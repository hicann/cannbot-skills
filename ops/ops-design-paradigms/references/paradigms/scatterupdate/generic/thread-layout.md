# ScatterUpdate 范式 SIMT 线程排布模型

> **层级**：通用知识层（generic/），语言中立。写散射侧的线程/核数编排算法——det 三阶段与 nondet 单阶段各有排布策略，同一颗芯片上排布策略与语言无关。语言适配层仅承载 API 写法（当前 AscendC：`SetBlockDim` / `SetScheduleMode`）。

---

## 1 det 路径：三阶段共用核排布

确定性三段式（Phase1 线性化 → Phase2 嵌入排序 → Phase3 段处理）**共用一个 blockDim**——每阶段内部自行消化核数需求：

```
phase1Cores = ceil(N / THREAD_NUM)           # N = totalIndexNum（散点数）
Phase2 并行度 = 排序库内建多核（tile 数决定）
phase3Cores = ceil(N_eff / THREAD_NUM)       # N_eff = b × k × s 任务总量

blockDim = max(phase3Cores, phase1Cores, sortActiveCores)
if blockDim < 1: blockDim = 1                  # 空 tensor 兜底
```

### 1.1 Phase1：全核 grid-stride 一维

每个散点恰一个线程处理（pos → 线性化 → 冲突 key + srcPos）。核内所有线程参与（无 guard），Phase1 需要全核写入 excl_bins/gh——**后续 Phase2/3 依赖全核数据**。

### 1.2 Phase2：排序库内建多核（核数需求独立）

排序（如 SortLib）的核数由**排序量 M（冲突单元数）**与排序库的 tile 容量决定：

```
sortActiveCores = min(coreCount, tileCount)    # tileCount = ceil(M / sortTileSize)
```

排序库内部管理 tile→核映射（库的分块域 = active_cores 运行期参数，[simt-hardware-model.md](../../common/simt-hardware-model.md) §6 超发核屏障参与）。宿主侧只需确保 `blockDim >= sortActiveCores`（否则排序阶段凑不齐屏障）。

### 1.3 Phase3：全核 grid-stride + (b,k,s) 一维化

Phase3 任务空间为三维 `(b, k, s)`（batch × 冲突段 × slice），一维化为 task ID 后 grid-stride 全核展开：

```
total_tasks = b * kNum * sNum
task = pos_base + tid
for task in range(task, total_tasks, stride):
    q = task // sNum        # 第一次快除（÷s）
    s = task - q * sNum
    b = q // kNum           # 第二次快除（÷k）
    k = q - b * kNum
```

| 场景 | 条件 | 快除次数 |
|---|---|---|
| **(b,k,s) 通用** | b>1 ∧ s>1 | 2 |
| (k,s) 二维 | b==1 | 1 |
| (b,k) 二维 | s==1 | 1 |

> **退化交叠**：`b == 1 && s == 1` 时任务空间退化为 `(k)`，任务量 `kNum`，零快除。

**排序量与任务量的区别**（[conflict-domain.md](conflict-domain.md) 核心原则）：
- 排序量 = kNum（冲突单元数 = 索引个数）——Phase2 排序的输入规模
- 任务量 = b × kNum × s（Phase3 线程并行展开）——N_eff 主导

**并行度 = b × k × s（越大越好）**——s 维线程级展开（每输出元素一任务），段尾任务 O(1) 自判、无串行扫描。

## 2 nondet 路径：单阶段核数两步法

非确定路径（原子直加）无排序无屏障，排布采用核数两步法：

```
blockDim = min(coreNum, max(1, ceil(N / THREAD_MIN)))
```

无 workspace、无 active_cores、无 SetScheduleMode。

## 3 空 tensor 早退与核数的关系

`totalIndexNum == 0 || totalVarElems == 0` 时**先走空 tensor 早退**（[conflict-domain.md](conflict-domain.md) §4），核数公式仅用于非空场景。空 tensor 时 blockDim = 1，kernel 侧短路。早退先于快除/排序 tiling 预计算。

## 4 超发核 guard（det 路径专用）

det 路径 `blockDim > sortActiveCores` 时，Phase1/Phase3 的 grid-stride 覆盖全核（每核各有任务），但排序阶段只需 sortActiveCores 核——超发核靠 `b >= active_cores` 跳过排序计算、直接到屏障（[simt-hardware-model.md](../../common/simt-hardware-model.md) §6）。超发核的 `ceil_div(tc - b, active)` 分子可为负，须钳 0。

## 5 线程数选型（THREAD_NUM）

同 [simt-hardware-model.md](../../common/simt-hardware-model.md) §5。det Phase3 段尾覆盖涉及 perm 寻址链与 slice 字节搬运——建议从 1024 起步，寄存器紧张时向下调（512/256）。
