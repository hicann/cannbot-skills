# IndexGather 范式 SIMT 线程排布模型

> **层级**：通用知识层（generic/），语言中立。读收集侧的线程/核数编排算法——同一颗芯片上排布策略与语言无关。语言适配层仅承载 API 写法（当前 AscendC：`SetBlockDim`）。

---

## 1 排布原则：一维 grid-stride（每输出元素一线程）

读收集（gather 族）的输出**连续无写冲突**——每个输出元素恰好由一个线程计算一次。因此最简且 cache-friendly 的排布是**一维线性化**：

```
tid = block_idx * thread_dim + thread_idx
stride = grid_dim * thread_dim
for i in range(tid, outSize, stride):
    out[i] = in[decode(i)]
```

- **为什么一维**：输出天然线性（无 (b,k,s) 类多维写散射任务分解需求）；一维 grid-stride 的地址模式连续，编译器可优化（无跨步跳转）。
- **为什么 grid-stride 而非分区间**：核数不必精确匹配 outSize，残余元素由前几核多跑一轮；编译器判界简单（`i < outSize` 即循环条件，无需 per-core 区间计算）。

## 2 核数两步法

多核切分的通用算法（语言中立公式）：

```
blockFactor = ceil(outSize / coreNum)
if blockFactor < PER_CORE_MIN:
    blockFactor = PER_CORE_MIN          # 下限保护（防每核元素太少启动开销占比过高）
needCoreNum = min(coreNum, ceil(outSize / blockFactor))
if needCoreNum <= 0:
    needCoreNum = 1                    # 空输出兜底（N==0 时仍须至少 1 核发射）
blockDim = needCoreNum
```

**设计理由**：
- **第一步（均分）**：outSize 平均分给 coreNum 核，得每核元素数 blockFactor。
- **PER_CORE_MIN 下限**：若 blockFactor 太小（如 outSize < coreNum），每核分到的元素数太少，核启动固定开销占比过高。设下限（如 1024）后自动收缩核数。
- **第二步（收缩）**：按修正后的 blockFactor 重算 needCoreNum，确保 ≤ coreNum 且 ≥ 1。
- **空输出兜底**：outSize == 0 时 needCoreNum = 1，kernel 侧 `usedCoreNum == 0` 短路（空 tensor 早退先于此公式）。

## 3 空 tensor 早退与核数的关系

`outSize == 0` 时**先走空 tensor 早退**，核数公式仅用于非空场景。早退须先于核数计算——空输出时 strides 可能含 0 除数（快除预计算同理）。

## 4 线程数选型（THREAD_NUM）

`thread=N`（或 `dim3(thread)`）决定每 block 线程数。选型依据 [simt-hardware-model.md](../../common/simt-hardware-model.md) §5（线程数翻倍→每线程寄存器减半）：

| 线程数 | 适用 | 风险 |
|---|---|---|
| 1024（默认） | 读收集/搬运类（寄存器需求低） | 极端计算密集可能溢出 |
| 2048 | 搬运密集（寄存器需求更低） | 同上加剧 |
| 更小（512/256） | 计算密集（解码链长、寄存器需求高） | 并行度降低 |

纯搬运算子（IndexGather）通常 1024 即可。
