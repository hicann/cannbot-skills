# compute-first-reduce-scatter 开发指导

> **场景 ID**：`compute-first-reduce-scatter`
> **适用路线**：`implementation_route=apace_custom`，`selected_scenario=compute-first-reduce-scatter`（DESIGN 冻结后编译进 PLAN 的记录值）
> **执行阶段**：Step 4（Implementation）只执行 PLAN，不改设计、不改接口/ABI；通用工程规范见 [`development-guide.md`](../../operator-design/development-guide.md)，架构原理见 [`fusion.md`](../../fundamentals/fusion.md) §6.2，本文只写本场景差异化要点。

---

## 1. 有序开发动作（PLAN §4 编译模板）

| # | 动作 | 验收锚点 | 证据 |
|:---|:---|:---|:---|
| A1 | 实现 FragmentTensor mm kernel（消 R 循环，R16） | FragTensorA/FragScaleA/FragTensorC 打包 R 个 rank 段地址 + `QmmMxBlockMmadFragment` per-fragment L1 隔离，一次调用覆盖 `R × curTileM` 行 | 编译通过；grep 无 R×T 子调用循环 |
| A2 | 实现 staging 即通信源 | mm 输出连续 `[M, N]` staging（GM workspace），rank 段 = chunk；PUT 钩子 src 偏移与 mm 写入偏移同源（`tileMOffset × axisN × sizeof(CType)`） | per-tile 契约不变量 3（[`development-guide.md`](../../operator-design/development-guide.md) §3.5） |
| A3 | 实现 AIV 严格分离 | 后 R 核通信 / 前核归约；分核公式 `jobIndex = GetBlockNum()-1-GetBlockIdx()`；零 tile 核无条件 SetFlag | AIV 分支骨架与 [`fusion.md`](../../fundamentals/fusion.md) §6.2.1 一致 |
| A4 | 实现增量归约 | 独立 `reduce_sum_ref.h`；手动 UB 批量形态：6-slot 布局 + src 双缓冲 + FP32 中间累加 + N 分段；禁止 TPipe/TQue 逐行模型 | R17/R18/R19（见 §5） |
| A5 | 实现 `Wait<BARRIER_NONE>` + 手动 CrossDevice | 禁止 `Wait<BARRIER_DEVICE>`（内建 CrossDevice step=rankSize 不轮询 remote → 80% 数据错误） | grep `BARRIER_DEVICE` 为空 |
| A6 | 完成 flag 编排 | flagId 避开保留区（[`fusion.md`](../../fundamentals/fusion.md) §3.3）；T=1 单次 SetFlag / T>1 逐轮配对，峰值 ≤15；SyncAll 在分核守卫外 | R5/R9 检查通过 |
| A7 | 完成最终 drain | 循环结束消费残留事件（含次数守卫）；最后一轮归约与 drain 全量覆盖 | 507014 挂死不出现 |

> 执行纪律：每完成一个动作立即冒烟（改完即测，禁止攒到最后）；bring-up 顺序 T=1/rank=2 → T>1 → 全量矩阵（[`workflow_integration.md`](../../workflow_integration.md) Step 3）。

---

## 2. 文件级合同（[MODIFY] 清单）

> 布局与命名规范以 [`development-guide.md`](../../operator-design/development-guide.md) §1.2 为唯一事实源；共享层零复制、CMake 直引 CANN 内置 apace。

| 目录 | 文件 | 职责（本场景） |
|:---|:---|:---|
| kernel/ | `{op_name}_tiling_data.h` | 单份完整 mm tiling + `commTilingData` + 通信派生字段（每卡行数/chunk 字节/staging 大小/归约粒度）+ 就地 `CommContext` 与 flag 常量；**无 `localMatmul` 字段** |
| kernel/ | `{op_name}_impl.h` | Impl：Init/Run 编排；AIC 统一 `for t` 循环（T=1 自然退化）+ SetFlag；AIV 逐轮 WaitFlag 门控 + Commit/Wait + SyncAll + 归约 |
| kernel/ | `{op_name}_frag_kernel.h` | FragmentTensor mm 内核（A1），独立 namespace（如 `ReduceScatterFragImpl`）。命名遵循 apace 惯例：`qmm_mx_kernel_{rs/ag/a2a}_frag.h` |
| kernel/ | `reduce_sum_ref.h` | 归约文件（A4），独立 namespace（如 `AiVReduceSumImpl`） |
| src/ | `kernel_launcher.h` | 4 个 dtype 变体 `__global__` 入口（E4M3E4M3/E5M2E5M2/E4M3E5M2/E5M2E4M3），各含 `KERNEL_TYPE_MIX_AIC_1_1` |
| src/ | `main.cpp` | 前置校验 9 项 → T 派生 → staging 分配（`M×N×sizeof(CType)`）→ dtype dispatch → fork 多 rank + TCP rootInfo + 建链 + launch（perf 模式内嵌 L2 flush） |
| src/ | `root_info_exchanger.h` | [REUSE] 从参考算子复制 |
| src/ | `utils.h` | [REUSE] 同上 |

---

## 3. Host 侧要求

### 3.1 前置校验清单（9 项，fork/建链前执行，main.cpp 与 gen_data.py 双侧）

| # | 校验项 | 违反后果 |
|:---|:---|:---|
| 1 | `M % rankSize == 0` | M 轴切分错位 |
| 2 | 对齐约束（`K%32`、`N%16` 等，按 dtype/fixpipe 32B 行对齐推导） | 跨卡 Win 槽污染 |
| 3 | `usedCoreNum ≥ rankSize` | TeamBarrier rendezvous 永不齐 → 挂死 |
| 4 | Win 容量 `M×N×sizeof(CType)` ≤ `HcclGetHcclBuffer` 实测值 | 通信越界 |
| 5 | flag 计数峰值 ≤ 15（T>1 时峰值 = T） | 硬件异常中断（R9） |
| 6 | 新增输入 buffer 预算（bias 等 L1/BT 兜底核算） | L1/BT 溢出 |
| 7 | `perRoundChunkBytes = tileM × N × sizeof(CType) ≤ 512KB` | UDMA 间歇 FAIL（R14） |
| 8 | `R × T ≤ 32`（MAX_FRAGMENT_COUNT） | fragment 地址数组越界 |
| 9 | `usedCoreNum - rankSize ≥ 1`（归约核至少 1 个） | 无核执行归约 → 输出缺失/挂死 |

### 3.2 T 派生

- 默认 `T | mSeg` 无尾块（R10）；`maxTileM = 512KB/(N×sizeof(CType))`（#7 推导值）为搜索基准，`T = mSeg/tileM ≤ min(15, 32/R)`（#5/#8 联合约束）双向搜索
- 语义无法保证无尾块时走策略 A：`paddedCurTileM` 32 对齐 + `realFragmentSize` 限读 + 三套 tiling（全量 `mmTilingData` + head `subMmTilingData` + tail `tailMmTilingData`）；两条路径均合法，见 [`fusion.md`](../../fundamentals/fusion.md) §6.2.7

> **约束联合推导原则**：单条约束各自满足 ≠ 联合可行。#5/#7/#8 联合作用派生出 shape 可行域——`tileM ≤ 512KB/(N×sizeof(CType))`（#7）与 `T = mSeg/tileM`，且 `T ≤ 15`（flag 峰值，#5）且 `R×T ≤ 32`（FragmentTensor，#8）联合推出 `M×N` 元素上限；T 派生必须以 `maxTileM`（#7 推导值）为搜索基准做双向搜索，而非仅按 headMSize 分档值派生。host 校验与 T 派生必须覆盖派生边界，cases.csv 必须包含约束边界用例（strided 场景 N>redUbN、T 边界、R 边界）——**硬件隐式上限类缺陷（静默丢零、无报错）只在边界用例下暴露**，缺失边界用例 = 缺陷逃逸到生产。
>
> **分档标定需 per-case 搜索兜底**：R 分档值（见 §5.6）只是起点——出现异常特征（tileCnt 非最优、SyncAll 占比偏高、GM 带宽争抢）的 case 需按通信覆盖率与收益实测搜索 headMSize，单点最优值不跨 case 复用（生产实证：某 R=4 大 K case 从 128 调整到 252 收益显著，其余 case 分档值合理）。

### 3.3 dtype dispatch

- host 按 `dtypeA`/`dtypeB` **运行期分派**到 4 变体入口，禁止硬编码单入口（生产实证：硬编码 → 异 dtype 字节流被错误模板解释，matched_ratio=0%，误差 ~36-59%）
- dispatch 宏模板与 SWAT tiling 引擎 dtype 说明见 [`development-guide.md`](../../operator-design/development-guide.md) §3.5
- Win 数据区偏移按 host 建链布局确定（官网布局从 0 可用；共享布局按约定偏移跳过头部），kernel/host 偏移同源（R14）

---

## 4. 验证矩阵（compute-first 特有项）

| 类别 | 项 | 达标条件 |
|:---|:---|:---|
| 精度 | T=1 与 T>1 双路径 | 只测 T=1 会掩盖多 tile 布局 bug，双路径必须全 PASS |
| 精度 | tail/非对齐 shape（如 M=4032, R=4）+ 多 commTurn | tail 路径特征信号暴露 |
| 精度 | per-tile 契约不变量 | 每轮 mm 只算本轮子区间（禁止"全量 mm × T 次"）；归约每轮只处理 `turn` 行区间（禁止 `(void)turn`）；T>1 违反 → raw_max 万级 ULP |
| 精度 | golden 语义 | 每卡"完整 K、切 M"，输出 `[M/R, N]`；golden FP32 累加路径 + 固定种子 |
| 精度 | dtype 全覆盖 | 4 变体各自 PASS；E5M2 matched_ratio=0% → 查 dispatch |
| 性能 | tileCnt 扫描 | `{1,2,4,8}`（上限受 flag 计数约束：compute-first `T ≤ 15` 且 `R×T ≤ 32`，见 fusion.md §6.2.10），切 tileCnt 必须重新调 `GetTilingData`；精度阶段先 tileCnt=1 串行基线 |
| 性能 | 投产门槛（R15） | 真实大 shape × R=2/4 双档 × 与 mc2 融合算子 / hccl 分步对标归档；toy shape 基线 = FAIL |
| 性能 | L2 flush 实接线（R20） | perf 每轮调 flush kernel，msprof 记录数 == 轮数 |

---

## 5. 实现细节模板（本场景专属落地形态）

> 通用方法论（编排原理、纪律、判据）以 [`fusion.md`](../../fundamentals/fusion.md) §6.2 为唯一事实源；本节收纳该场景已生产验证的**具体实现形态**——骨架代码、UB 布局、标定值。新算子可照搬结构，数值需按自身 shape/dtype 重新标定。

### 5.1 AIV 严格分离编排完整骨架

```
AIV (RunAllToAllAndReduceSum):
  jobIndex = GetBlockNum() - 1 - GetBlockIdx()      ← 后 R 核通信（最大 blockIdx → jobIndex 0，依次递减）
  isCommBlock    = (jobIndex < rankSize)             ← 通信核 = 后 R 核
  isComputeBlock = (blockIdx < usedCoreNum - rankSize)  ← 归约核 = 前 (核数-R) 核

  commTurn<=1 退化: WaitFlag → SyncAll → 通信核 Commit/Wait<BARRIER_NONE> → SyncAll
                    → teamBarrier_.CrossDevice()（仅 jobIndex=0）→ SyncAll
                    → 归约核 DoReduceSumBatch(tile0) → 通信核 Finalize
  Tile 0:      WaitFlag → SyncAll → 通信核 Commit/Wait<BARRIER_NONE> → SyncAll → CrossDevice → SyncAll
  Tiles 1..T-1: WaitFlag → SyncAll → 通信核 AllToAll(t) ‖ 归约核 ReduceSum(t-1)
                                    → SyncAll → CrossDevice → SyncAll
  Final:       归约核 ReduceSum(T-1)；通信核 Finalize
```

### 5.1a 多核通信失败链（生产归档）

> **以下为失败尝试（禁止使用）**，仅作历史根因归档。

| 尝试 | 现象 | 根因 |
|:---|:---|:---|
| totalJobs=R + winOffset=0 | Rank0 PASS、Rank1 FAIL（NaN/大面积元素错） | PUT 从偏移 0 写入覆盖 barrier counter 区，CrossDevice 读到数据值恒 ≥ count → 同步"假通过" |
| + winOffset=128 + `Wait<BARRIER_DEVICE>` | 仍大面积元素错 | 框架 CrossDeviceExecute `step = min(totalJobs, nranks)`：totalJobs=rankSize 时 step=rankSize，循环只遇到自己直接退出，不轮询任何 remote |
| + 额外 SyncAll | 同上 | **SyncAll 是本地核间同步，无跨设备能力**，加多少次都无法保证远端 PUT 到达 |
| 手动轮询 remote GM flag | 死循环超时 | **MTE 跨设备 GM 写可见性无框架 fence 保证**（本端 MTE2 读看不到远端 MTE3 写）；手动跨设备轮询不可行 |
| totalJobs=1 单核通信 + 真同步 | R=2 PASS、R=4 死锁 | 单核串行 PUT + Drain 在 R≥4 暴露死锁竞态；回退路径不可行 |

> **以下为已生产验证方案**。

**终局（方案B，已生产验证）**：teamBarrier `totalJobs=1`（jobIndex=0 的 CrossDevice step=1 正确轮询所有 remote）+ AllToAll `totalJobs=rankSize`（后 R 核各负责 1 个 target 并行 PUT）+ winOffset=128 + `Wait<BARRIER_NONE>`（仅 Drain）+ 手动 `teamBarrier_.CrossDevice()`——两个 totalJobs 正交分离，不改框架。

### 5.2 批量归约循环骨架（两路径强制提供，伪代码）

> **归约实现必须同时提供两路径**（`fusion.md` §6.2.6）：
> - **ExecuteReduceSum**（逐行）：T>1 多 tile 路径（归约核处理第 t-1 轮）
> - **ExecuteReduceSumBatch**（批量）：tileCnt=1 路径——一次处理多行摊薄 flag 次数（`fusion.md` §6.2.6 多行批量归约），否则逐行 flag 操作数 = R×tileM×(N/redUbN)×10，性能严重退化

**多核分治（必做，禁止单核）**：归约核集合（严格分离 = 前 `rsCoreNum = usedCoreNum - rankSize` 核）内按行块均分本轮 `tileM` 行，落地用 SplitToCore 形态：

```cpp
// 每归约核只处理 [startRowId, endRowId) 行；连续行块、余数前置核 +1
uint32_t startRowId, endRowId, rowNum;
SplitToCore(tileM, rsCoreNum, GetBlockIdx(), startRowId, endRowId, rowNum);
// rowNum = tileM / rsCoreNum; 余数 r = tileM % rsCoreNum 分给前 r 核（各 +1 行）
```

**禁止单核归约**（如 `GetBlockIdx()==0` 独立承担全量行）——R 个来源 × 全量行由单核串行处理，归约耗时独占且多核若误写同一 yGm 会产生写后写竞争；这是已发生产 bug 形态。

```
batchRows = UB预算 / (每行字节数 × 每元素 buffer 系数)   // host 按 UB 推导
for batch in 按 batchRows 分批（覆盖本核行区间 [startRowId, endRowId)）:
    for i = 0..R-1（R 个来源，pingpong 双缓冲）:
        2D DataCopyPad(srcBuf[i%2], 来源 i 的本批行区间, blockCount=本批行数, 64B pitch)
        Cast(srcFP32[i%2], srcBuf[i%2], CAST_NONE)
        PipeBarrier<PIPE_V>()
        Add(accFP32, accFP32, srcFP32[i%2])
    Cast(dstBF16, accFP32, CAST_RINT)
    2D DataCopyPad(yGm 本批行区间, dstBF16, blockCount=本批行数)
    // 一批数据一次 SetFlag/WaitFlag（MTE2/V/MTE3 FIFO 特性）
```

### 5.3 归约事件配对完整代码模板（R19 落地形态）

4 类 HardEvent 的语义与 Set/Wait 时机以 [`fusion.md`](../../fundamentals/fusion.md) §6.2.6（唯一事实源）为准；本节只留落地代码模板（pingpong slot = i%2）：

```cpp
// Init 时预初始化 MTE3_V（首次 Wait 前需有对应 Set）
SetFlag<HardEvent::MTE3_V>(0);
for i = 0..R-1:                                    // R 个来源
    slot = i % 2;
    if (i >= 2) WaitFlag<HardEvent::V_MTE2>(slot); // slot 复用前等 V 消费完
    DataCopyPad(srcBuf[slot], ...);                // MTE2 搬入
    SetFlag<HardEvent::MTE2_V>(slot);
    WaitFlag<HardEvent::MTE2_V>(slot);
    Cast(srcFP32[slot], srcBuf[slot], CAST_NONE);  // V：BF16→FP32（独立 srcFP32，禁止 in-place）
    PipeBarrier<PIPE_V>();                          // V-V 依赖防御
    Add(accFP32, accFP32, srcFP32[slot]);          // V：累加
    SetFlag<HardEvent::V_MTE2>(slot);              // 释放 src slot
// 循环结束消费残留 V_MTE2 事件（保持事件计数器平衡，漏消费 → 后续 kernel 挂死）
// 注意 R=1 边界：slot 1 从未 Set，Wait 无配对 = 未定义行为
WaitFlag<HardEvent::V_MTE2>(0);
if (R >= 2) { WaitFlag<HardEvent::V_MTE2>(1); }
// 输出路径
WaitFlag<HardEvent::MTE3_V>(0);                    // 等上一批 MTE3 搬出完成
Cast(dstBF16, accFP32, CAST_RINT);                 // V：FP32→BF16
SetFlag<HardEvent::V_MTE3>(0);
WaitFlag<HardEvent::V_MTE3>(0);
DataCopyPad(yGm, dstBF16, ...);                    // MTE3 搬出
SetFlag<HardEvent::MTE3_V>(0);                     // 为下一批预置
```

### 5.4 手动 UB 6-slot 布局与预算公式

> **UB 管理机制红线**：归约区 UB 管理**必须**与通信区使用同一机制。禁止 `TPipe::InitBuffer` 与 `Te::MakeMemPtr<Te::Location::UB>` 混用——两套机制偏移空间不共享，混用导致地址重叠 → MTE2 UB out of bounds（507015）。推荐 `MakeMemPtr` 手动偏移（与官方 AllGather/AllToAll 算子一致，`operator-anatomy.md` §4.3）。

| slot | buffer | 份数 | 每份字节数 | 用途 |
|:---|:---|:---|:---|:---|
| 1-2 | srcBuf[2] | 2（pingpong） | `redUbM × redUbN × sizeof(CType)` | 搬入 |
| 3-4 | srcFP32[2] | 2（pingpong） | `redUbM × redUbN × sizeof(float)` | FP32 Cast 目标（独立，禁止与 srcBuf 复用） |
| 5 | accBuf | 1（语义依赖） | `redUbM × redUbN × sizeof(float)` | FP32 累加器 |
| 6 | dstBuf | 1 | `redUbM × redUbN × sizeof(CType)` | 输出 |

host 侧 UB 预算推导（参数化公式，系数按 dtype 路径代入）：

```
perElemBytes = 2×sizeof(CType) + 2×sizeof(float) + sizeof(float) + sizeof(CType)
             // 示例 BF16 输出 + FP32 累加路径：2×2 + 2×4 + 4 + 2 = 18B
availableUB = UB 总量 - guard 通信区（尺寸按框架约定）
maxElements = availableUB / perElemBytes
redUbN      = min(N, 单次列宽上限) 且按对齐要求取整
redUbM      = min(tileM, maxElements / redUbN)
            // N > redUbN（strided 场景）时追加 redUbM ≤ 32（fusion.md §6.2.6 纪律 3 硬件限制）
```

> **另一种已验证形态**：整片申请 UB（如 192KB）后按 `[sum｜bf16×2｜f32×2｜out]` 划分，批量行数按 UB 预算自适应（逐 rank 搬运与计算 pingpong 重叠），N 超预算时按列分段——与 6-slot 布局同为合法形态，选择按 UB 预算与批量宽度需要。

### 5.5 多套 tiling 字段契约（策略 A/B 通用，tiling_data.h）

```cpp
struct {OpName}TilingData {                 // compute-first 多套 tiling 形态
    CommTilingData commTilingData;          // 通信切分（splitAxisTileSize/Cnt + splitAxisTailSize/Cnt + nonSplitAxisSize）
    QuantMatmulTilingData mmTilingData;     // 全量 tiling（GetTilingData(m, n, k)），T=1 退化用
    QuantMatmulTilingData subMmTilingData;  // head 子问题 tiling（GetTilingData(headMSize, n, k)），**T>1 时强制存在**（策略 B 无尾块也需要）
    QuantMatmulTilingData tailMmTilingData; // tail tile tiling（GetTilingData(rankSize*tailMSize, n, k)），T>1 且有 tail 时用；策略 B 无尾块时可空但字段必须声明
};
// kernel 侧选择：(commTurn == 1) ? mmT : (isHeadTile ? subMmT : tailMmT)；blockDim 恒用 mmTilingData.usedCoreNum
```

> **T>1 时 subMmTilingData 强制存在**：全量 tiling 的 baseM 可能大于 headMSize 导致 tile 分布不合理，必须为 headMSize 独立调用 `GetTilingData`（[`development-guide.md`](../../operator-design/development-guide.md) §3.5）。策略 B（T|mSeg 无尾块）时 tailMmTilingData 可空但字段必须声明。

### 5.6 headMSize 标定示例（dav-3510 一种已验证实现的实测值）

> 决策原则（通用）："分片小→减小 tile 让通信尽早启动；分片大→增大 tile 减少同步次数"。下表数值为示例标定，**新算子应按 tile 粒度与流水深度目标自行标定**，勿直接照搬。

| 场景 | headMSize（示例标定值） | 理由 |
|:---|:---|:---|
| R ≤ 2 | 保持较大 tile（示例：512） | 少 commTurn ⇒ 少 SyncAll，通信效率高 |
| R ≥ 4 | 压缩小 tile（示例：128） | 多 commTurn ⇒ 流水掩盖充分，通信时间短 |
| 联合约束 | `headMSize ≥ ceil(mSeg / 15)` | flag 计数峰值上限（T = mSeg/headMSize ≤ 15）；分档值导致 T>15 时必须上调 headMSize |
| 对齐 | 16 的倍数；`tailMSize = chunkM % headMSize` 非 0 时必须 16 对齐（Blaze 最小行粒度），host 拒绝非法值 | — |
| baseM 联动 | SWAT 输出 baseM=256 且 headMSize 可被 128 整除时，手动减半 baseM→128 改善 tile 分布（机制见 [`optimization-playbook.md`](../../troubleshooting/optimization-playbook.md) §2） | — |

### 5.7 tiling 结构体字段（host 填充，示例实现的典型形态）

| 字段 | 含义 | host 填充规则 |
|------|------|--------------|
| `commTilingData` | 通信切分（tileM/T/N，单对象） | 由 T/headMSize 派生填充（默认 `T \| mSeg` 无尾块；策略 A padding 为合法替代，fusion.md §6.2.7） |
| `mmTilingData` | 全量计算 tiling（**完整 {M, N, K}**） | SWAT tiling 引擎输出 `GetTilingData(m, n, k)`；T=1 退化路径用 |
| `subMmTilingData` | head 子问题 tiling（**T>1 时强制存在**） | `GetTilingData(headMSize, n, k)`；baseM 减半联动见 §5.6 |
| `tailMmTilingData` | tail tile tiling（T>1 且有 tail 时用；**策略 B 无尾块时可空但字段必须声明**） | `GetTilingData(rankSize × tailMSize, n, k)`；tailMSize 必须 16 对齐（host 校验） |
| `coreNum` | AIC 核数 | `mmTilingData.usedCoreNum`（blockDim 恒用全量 tiling 的核数）；**必须 ≥ rankSize**（否则 TeamBarrier rendezvous 挂死） |
| `rankSize` | R | 冗余备份（kernel 主用 udmaCtx.rankSize） |
| `mSeg` | M / R（本卡输出行数） | M 必须被 R 整除（host 校验） |
| `chunkBytes` | mSeg × N × sizeof(CType) | 与 PUT 钩子 chunkBytes_ 同源 |
| `tileMaxBytes` | tileM × N × sizeof(CType) | PUT 钩子 src 偏移步长（尾块会错位 ⇒ 无尾块） |
| `stagingSize` | R × chunkBytes = M×N×sizeof(CType) | workspaceGM 分配大小；同时是 Win 容量校验依据 |
| `redUbM` / `redUbN` | 归约 ubTile 行/列 | host 按 UB 预算核算（含 64B pitch 换算，公式见 §5.4） |

### 5.8 自研 FragmentTensor mm kernel 骨架（qmm_mx_kernel_{rs}_frag.h）

参考 AllGather `qmm_mx_kernel_ag_udma.h`（`QmmMxKernelAgUdma`）改造。**独立 namespace**（如 `ReduceScatterFragImpl`），禁止与 Impl 共用。改造时必须区分**必须保留**与**可省略**：

| 必须保留（砍了即错） | 可省略（compute-first 特有） |
|:---|:---|
| per-fragment L1 隔离（`QmmMxBlockMmadFragment` 跨 rank 边界自动切 fragment） | dependId 预触发 / `WaitFlag`（计算不依赖通信） |
| fragment 地址公式：A 段 `aGM + r×mPerRank×fullK`（**fullK = 完整 K 行 stride，A 不切分**），C 段 `stagingGm + r×mPerRank×cBytesPerM`；`cFragAddrs_` 保持原始 rank 顺序 | Win 区离散地址解析（A 为本卡 GM 连续 rank 段） |
| **localLast 编排（强制，见 §5.11）**：fragment 重排 `[remote..., local]` + 边界提前 SetFlag | winDataBase（无 Win 区读 A） |
| `SetL2Cache`（fullMBlock + 128B 对齐判定）+ tail tile `sch.UpdateTailTile` | — |

> **localLast 强制条款**：移除 localLast = **阻塞级错误**（峰值 2T + 丧失提前启动；错位根因是 `cFragAddrs_` 顺序写错，非 localLast 本身）。详见 §5.11。

> **TransA/TransB 参数化**（`operator-anatomy.md` §7.2）：Impl 模板必须含 TransA/TransB，Layout 条件选择。禁止固定 Layout。

> **地址公式红线**：A 行 stride = **完整 K**（不是 K/R）；fragment 按 M 轴 `chunkM` 打包 R 段。

> **3-region 调度**：策略 B per-tile 调用可走单 region；多 round 或策略 A 必须保留 HEAD/MAIN/TAIL（`operator-anatomy.md` §7.3）。

**Params 结构契约**（`fusion.md` §6.2.7 契约的落地形态）：

```cpp
struct Params {
    const QuantMatmulTilingData* mmTile;   // 当前轮 tiling
    QBMMTiling qbmmParams;                 // {baseM, baseN, baseK, dbL0C}
    uint32_t rankSize, rankId;             // R, 本 rank
    uint64_t mPerRank;                      // mSeg = M/R
    uint32_t tileM;                        // problem M = R × tileM
    uint64_t k, n, scaleKLen;              // k = 完整 K；scaleKLen = CeilDiv(K,64)×2
    GM_ADDR aGM, aScaleGM, bGM, bScaleGM;  // B/ScaleB 全 rank 共享
    GM_ADDR stagingGm;                     // C 输出（= PUT 通信源）
    uint64_t cBytesPerM, tileMOffset;     // 行步长, 本轮 M 偏移
};
```

**核心方法**：

| 方法 | 职责 |
|:---|:---|
| `BuildFragmentTensors` | `cFragAddrs_[r] = stagingGm + r×mPerRank×cBytesPerM`；A/ScaleA 按 `r×mPerRank×k` 步进；localLast 重排 addrList（`[remote..., local]`），`cFragAddrs_` 保持原始顺序 |
| `Run` | problem M = `R × tileM`；构造 scheduler + `FragL1Params`；`mmadFrag.Init(...)`；`BuildFragmentTensors` + `Process` |
| `Process` | `while (sch.GetTileIdx)` 循环：`SetL2Cache` → FragmentTensor Slice → `mmadFrag(..., cFragAddrs_, mPerRank, tileM, ..., blockC)`（17 参直接调用，输出经 `ScatterL0C2GM` 散射到 staging） |

### 5.9 AIV 编排两种形态（均合法，均有生产实例验证）

| 形态 | 结构 | 优点 |
|:---|:---|:---|
| **统一 for-t 循环（推荐）** | `for t { WaitFlag+SyncAll → 通信核 Commit/Wait → SyncAll → CrossDevice(jobIndex=0) → SyncAll → 归约核 DoReduceSumBatch(t-1) }` + 尾轮 `DoReduceSumBatch(T-1)` | 无分支、T=1 自然退化、代码最短 |
| **单 tile/多 tile 分支** | `commTurn≤1` 走单 tile 路径（4 级 SyncAll 链）；多 tile 时 tile 0 单独提出仅通信（通信尽早启动），`for t=1..T-1 { 通信核 AllToAll(t) ‖ 归约核 ReduceSum(t-1) }` | tile 0 通信提前一拍启动，首拍通信延迟略低；分支语义更显式 |

> 两形态性能等价（流水稳态后通信均被计算掩盖），统一 for-t 循环代码更简洁，推荐默认采用；若 profiling 显示首拍通信暴露明显，可切换分支形态。

### 5.10 per-tile 子区间契约代码模板（AIC RunMatmul，T>1 实现红线）

```cpp
for (uint32_t t = 0; t < commTurn; ++t) {
    uint32_t curTileM    = GetTileM(t);        // t < headTileCnt ? headTileM : tailTileM
    uint64_t tileMOffset = GetTileMOffset(t);  // t < headTileCnt ? t*headTileM : headTileCnt*headTileM
    const auto& tileT = (commTurn == 1) ? mmTiling : (isHeadTile ? subMmTiling : tailMmTiling);
    params.aGM      = aGm         + tileMOffset * axisK;                 // A 按行偏移
    params.aScaleGM = scaleAGm    + tileMOffset * scaleARowBytes;        // ScaleA 按行偏移
    params.cGM      = workspaceGm + tileMOffset * axisN * sizeof(CType); // C 按行偏移
    mmKernel_(params);                          // problem M = R × curTileM（FragmentTensor 打包）
    CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
}
```

不变量与违反后果见 [`development-guide.md`](../../operator-design/development-guide.md) §3.5 per-tile 契约表（唯一事实源）。

### 5.11 localLast 双 flag 通信提前启动（compute-first 默认编排）

机制原理（fusion.md §6.2.2/§6.2.3，compute-first 默认）的落地要点：

1. **localLast 重排只作用于 FragmentTensor addrList**：构造 `addrListA/Scale/C` 时跳过本 rank 段、最后追加；**`cFragAddrs_` 必须保持原始 rank 顺序**（mmadFrag 内部 L1 cache 管理依赖原始顺序）——写错 = 阻塞级选型错误。实际输出走 FragmentTensor Slice 得到的 blockC。
2. **边界预计算**：`localFragBoundary = headMainRows - fragM`（最后一个 fragment=本卡段的起始 M 位置）；调度循环内 `mPos >= localFragBoundary` 首次满足时 `CrossCoreSetFlag<0x2, PIPE_FIX>(flagA)`。
3. **兜底补 Set**：循环结束 `remoteFlagSet == false` 的核（无本卡 tile、未跨边界）必须补 Set flagA——否则 AIV `WaitFlag(flagA)` 挂死。
4. **AIV 两侧分等**：通信核 `WaitFlag(flagA)` 后启动 AllToAll；reduceSum 前 `WaitFlag(flagB)` 确认本卡段完成；两轮之间 SyncAll/CrossDevice 序列与基线协议一致。
5. **host 侧无新增校验**：每 flagId 峰值仍 = T，沿用 §3.1 #5 校验。

---

## 6. 合规映射（R1-R21 本场景重点项）

> 全量红线见 [`review-checklist.md`](../../review-checklist.md)；违反任意红线 = FAIL。本表列本场景最易踩中项。

| # | 约束 | 本场景落点 |
|:---|:---|:---|
| R9 | flag 计数峰值 ≤ 15 | A6；host 校验 #5 强制执行（T>1 峰值 = T）；若峰值 = 2T（每轮 Set 双 flag）= 性能 FAIL |
| R10 | 尾块策略 | §3.2 T 派生：默认 `T \| mSeg` 无尾块，否则策略 A padding 32 + realFragmentSize + 多套 tiling |
| R14 | Win 数据/元数据分离 + 单轮 PUT ≤ 512KB | Win 数据区偏移按 host 建链布局确定，三处同源（PUT 写/归约读/host 建链）；校验 #7 派生 `maxTileM` |
| R16 | mm 默认 FragmentTensor | A1；vendor R×T 子调用为例外，须 DESIGN 论证 SCALAR 占比（"vendor kernel + FragmentTensor C 输出"= 阻塞级错误） |
| R21 | localLast 编排禁止移除 | A1/§5.11：frag kernel 含 `[remote..., local]` 重排 + 边界提前 SetFlag(flagA)；移除 = 峰值 2T + 丧失提前启动 |
| R17 | 归约禁止逐行搬运 | A4：2D DataCopyPad blockCount=本批行数；strided（N>redUbN）redUbM ≤ 32 或 1D 退化 |
| R18 | 归约独立 srcFP32 双缓冲 | A4：禁止 in-place BF16→FP32 Cast |
| R19 | 归约事件配对完整 | A4/A7：MTE2_V/V_MTE2/V_MTE3/MTE3_V 同迭代配对 + 循环结束消费残留事件（缺 Wait → 507014 挂死） |
| R3 | 入口变体 + dtype dispatch | kernel_launcher.h 4 变体 + §3.3 运行期分派 |
| R12 | UB 静态通信区隔离 | 通信 commBuf/barrierBuf 与归约 buffer 物理隔离；**TPipe 与 MakeMemPtr 必须二选一，禁止混用 → 507015** |
| R20 | perf L2 flush 实接线 | §4 性能项 |

---

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`fusion.md`](../../fundamentals/fusion.md) §6.2 | compute-first 架构原理与编排模式（通用方法论唯一事实源） |
| [`operator-anatomy.md`](../../operator-design/operator-anatomy.md) §7 | 文件级契约与 MAX_FRAGMENT_COUNT |
| [`development-guide.md`](../../operator-design/development-guide.md) §3.5 | per-tile 子区间契约、dispatch 宏模板 |
| [`review-checklist.md`](../../review-checklist.md) | 全局红线 + 场景约束（本场景适用项）与 FAIL 诊断 |
| [`step4-implementation.md`](../../workflow/step4-implementation.md) | Step 4 门禁、修复循环、验收流程 |
