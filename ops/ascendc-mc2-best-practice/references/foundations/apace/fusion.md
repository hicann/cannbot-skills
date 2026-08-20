# apace 通算融合组合模式

> 本文档覆盖通算融合的组合层：通信与计算如何重叠、GET/PUT 模式选型、flag 编排模式、localMatmul 模式、多对象复用与扩展语义推导。前置阅读：communication.md（通信接口）、compute.md（计算接口）。

## 目录

1. [通算重叠原理](#1-通算重叠原理)
2. [GET vs PUT 模式选型](#2-get-vs-put-模式选型)
3. [Flag 编排模式](#3-flag-编排模式)
4. [PUT 算子级编排验收](#4-put-算子级编排验收)
5. [localMatmul 模式（0/1/2）](#5-localmatmul-模式012)
6. [多对象复用与扩展语义推导](#6-多对象复用与扩展语义推导)

---

## 1. 通算重叠原理

AIC（计算）与 AIV（通信）在同一 kernel 内并行执行，两侧通过 CrossCore flag 做逐 tile 精细同步（见 §3），通信延迟被计算掩盖。`Commit()` 的非阻塞特性是通算流水的关键——AIV 可以在 Commit 后、Wait 前插入其他指令（机制详见 communication.md）。

掩盖条件：每个 tile 的计算耗时 ≥ 通信耗时，且流水深度足够。

- **PUT 模式**：AIV 先推数据、AIC 后算（通信→计算）；localMatmul=1 时本地 A×B 前置计算与 AIV PUT 并行（见 §5）。
- **GET 模式**：AIC 先算、AIV 后拉（计算→通信），Win 槽位环形复用 + 回压（见 §3.4）。
- 重叠粒度受 comm tile 限制（CalcDependTileIdx + waitedMask 去重，见 §4.2）。

---

## 2. GET vs PUT 模式选型

**方向语义**：

- **GET 模式 = 计算→通信**：AIC 先算 C 写到 Win 区，AIV 从远端 Win 区拉回本 rank 的 C 段。
- **PUT 模式 = 通信→计算**：AIV 先推数据到远端 Win 区，AIC 从 Win 区读取计算。实现见 `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 的 `AllToAllCommPutImpl`（AllGather PUT 变体见 `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` 的 `AllGatherCommPutImpl`，钩子结构相同，仅地址公式不同）。

> **官网暂无 GET 算子样例**：GET 钩子基础设施（`apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` 的 `AllToAllCommGetImpl`）已就绪并注册进 `CollectiveCommHelper<AllToAll, GET, ...>` 分发（`apace/block/aiv_comm/collective_comm_api.h`），但官网 kernel/ 下两个算子均为 PUT 模式，无 GET 使用方。本文 GET 内容为钩子契约级描述，地址公式与 self 跳过规则均可在该头文件中直接验证。

### 2.1 钩子差异

| 钩子 | GET | PUT |
|:---|:---|:---|
| `PostInit()` | 空 | `CrossDevice()+CrossCore()`（通知对端即将写入） |
| `DoCommit()` | `CrossCore()+CrossDevice()` → `ReadNbi`（拉） | `WriteNbi`（推），**无**前置 barrier；self rank 直接 return |
| `DoWait()` | `Drain`（self 直接 return） | `Drain`（仅非 self）→ `CrossDevice()+CrossCore()`（含 self） |
| `DoFinalize()` | `CrossCore()+CrossDevice()` | 空 |

> **顺序差异**：GET 钩子内 barrier 顺序是先 Core 后 Device；PUT 的 PostInit/DoWait 是先 Device 后 Core。以 `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` / `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 实际代码为准。

PUT 的 src/dst 与 GET 完全镜像：GET 从远端读，PUT 往远端写。PUT 地址公式（`AllToAllCommPutImpl::DoCommit`）：`srcAddr = localAddr_ + targetRankId * chunkBytes_ + currentTileIdx_ * tileMaxByteSize_`；`dstAddr = commBufferAddrs[targetRankId] + winOffset_ + rankId * chunkBytes_ + tileByteOffset_`。

### 2.2 flag 编排对比

| 维度 | GET 模式（契约，官网无样例） | PUT 模式（官网实现） |
|:---|:---|:---|
| 发起方 | AIC 先 SetFlag → AIV WaitFlag | AIV 先通信 → SetFlag → AIC WaitFlag |
| Wait 参数 | Wait(true) 按 waitLast 早退语义：DoWait 仅在 currentTileIdx_ == totalTiles - 1 时执行一次（典型 Commit→Wait 循环中为倒数第二轮），末轮不 Drain（详见 communication.md） | `Wait<BARRIER_DEVICE>()`（BarrierMode 模板参数） |
| SyncAll | 不需要（CrossCore flag 已保证时序） | **每轮必须** `SyncAll<true>()`（WriteNbi 对端可见性） |
| flagId | `tid`（tile 索引） | `tid`（round 索引） |
| 回压 | AIC WaitFlag<0x2, PIPE_M>(tid-bufCnt) | 无（PUT 不需要环形回压） |

### 2.3 Wait 参数维度差异

GET 约定的 `Wait(true)` 中 `true` 是 `waitLast` 位置参数——Wait(true) 按 waitLast 早退语义：DoWait 仅在 currentTileIdx_ == totalTiles - 1 时执行一次（典型 Commit→Wait 循环中为倒数第二轮），末轮不 Drain（详见 communication.md）；PUT 的 `Wait<BARRIER_DEVICE>()` 是 `BarrierMode` 模板参数（控制 DoWait 中是否插入 CrossDevice barrier）。两者是完全不同的维度。

---

## 3. Flag 编排模式

CrossCore Flag 是 AIC↔AIV 跨核同步的核心机制。每个 flag 由 `<MODE, PIPE, flagId>` 三元组标识；MODE 常量与 PIPE 选项机制表见 communication.md（本文不复制）。

### 3.1 GET 编排不变量

> **官网暂无 GET 算子样例**：以下为 GET 模式的契约级编排模式（与 communication.md 的 GET 钩子契约配套），官网 kernel/ 下无可验证的 GET 使用方。

```
AIC (MatmulProcess)                            AIV (AllToAllProcess)
─────────────────────────────                  ─────────────────────────────
tile tid:
  [if ring: WaitFlag<0x2, PIPE_M>(tid-bufCnt)] ←── 回压
  RunMatmul(C → Win[tid % bufCnt])
  SetFlag<0x2, PIPE_FIX>(tid)          ──→    WaitFlag<0x2, PIPE_S>(tid)
                                               Commit(GET C[tid])
                                               Wait(true)
  [if ring: ...]                        ←──   SetFlag<0x2, PIPE_MTE3>(tid)
```

> GET 模式 `AllToAllProcess` 中不需要 `SyncAll`。同步完全由 CrossCore flag 负责。

### 3.2 PUT 编排不变量

官网实现：`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` 的 `RunAllToAll` / `RunMatmul`。

```
AIV (RunAllToAll)                              AIC (RunMatmul)
─────────────────────────────                  ─────────────────────────────
round tid:
  Commit(scale A[tid])
  Commit(data A[tid])
  Wait<BARRIER_DEVICE>()
  SyncAll<true>()
  SetFlag<0x2, PIPE_MTE3>(tid)          ──→    WaitFlag<0x2, PIPE_MTE2>(tid)
                                                从 Win 读 A[tid] → Matmul
```

> PUT 模式 AIC 侧 `UdmaCommWaitPolicy::WaitTile()`（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h`）使用 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)`——PIPE_MTE2 对应数据搬入流水，等待 AIV WriteNbi 数据到达。AllGather PUT 算子（`apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h` 的 `AllGatherProcess`）同一模式：循环外先预触发 `CrossCoreSetFlag<0x2, PIPE_MTE3>(0)`（自身数据始终就绪，AIC 可直接消费），循环内 `SyncAll<true>()` 后 `CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1)`（flagId 用 round+1 与预触发的 0 错开）。

### 3.3 flagId 选择规则

- 官网两个 PUT 算子均使用**轮次索引 `tid`/`round`** 作为 flagId（`CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)`）
- **硬件规则**（官方约束，详见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`）：
  - 模式 0/1/2 每核仅 **16 个 flagId（0-15）**，超出截断低 4bit——截断机制正是直接用无界 `tid` 作 flagId 能工作的硬件基础（Set/Wait 双方截断到同一 flag，配对保持）
  - 每个 flagId 对应计数器，Set/Wait 必须配对，否则未定义行为
  - **SyncAll 硬同步内部占用 flagId [11-14]**，官方不建议同时使用 CrossCoreSetFlag 与 SyncAll 硬同步——PUT 模式组合使用两者，当 `tid % 16 ∈ [11,14]` 时存在与 SyncAll 内部 flag 冲突的理论风险，移植到新平台需重新确认
  - Matmul 高阶 API 占用 flagId [0, 2N-1]（最多 [0,7]）；自定义 flagId 需避开此类保留区间
  - 同一核连续发出的 CrossCoreSetFlag，硬件不保证执行顺序
- `apace/utils/constant.h` 定义 `FLAG_ID_MAX = 16` 为**预留常量**（官网 apace 当前未使用）

### 3.4 环形回压与 bufferCount

> **GET 模式契约级描述**：环形回压是 GET 模式（计算→通信）Win 槽位复用的配套机制，官网 kernel/ 下暂无 GET 算子样例可验证；PUT 模式（官网两个算子）不需要环形回压。

#### 机制

`bufferCount` 个槽位环形复用 Win 区。当 `totalTileCnt > bufferCount` 时，需要回压机制防止 AIC 覆盖未消费的数据。

#### 两种模式

| 模式 | 条件 | 行为 |
|:---|:---|:---|
| **全量缓冲** | `bufferCount >= totalTileCnt` | 无回压，所有 tile 的 Win 槽位独立 |
| **环形复用** | `bufferCount < totalTileCnt` | AIC 等 AIV 释放旧槽位（回压 flag） |

#### 环形回压不变量

| 侧 | 不变量 |
|:---|:---|
| AIC | 当 `bufferCount < totalTileCnt && tid >= bufferCount` 时，`WaitFlag<0x2, PIPE_M>(tid - bufferCount)` 等待 AIV 释放 |
| AIV | 当 `bufferCount < totalTileCnt && tid < totalTileCnt - bufferCount` 时，`SetFlag<0x2, PIPE_MTE3>(tid)` 通知 AIC 可复用 |

---

## 4. PUT 算子级编排验收

### 4.1 AIV 通信流水线（PUT）验收

`RunAllToAll()`（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` `RunAllToAll`）是 AIV 侧主通信循环。

#### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | Commit 顺序 | scale 先、data 后（`allToAllScaleA_.Commit()` → `allToAllA_.Commit()`，同 channel 串行） |
| 2 | Wait 方式 | `allToAllA_.Wait<BARRIER_DEVICE>()`；scale 与 data 同 channel，只 Wait 一次 |
| 3 | SyncAll | 每轮 `SyncAll<true>()` |
| 4 | AIC 通知 | 每轮 `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` 通知 AIC 第 tid 个通信 tile 就绪 |
| 5 | 分核保护 | Commit/Wait 仅在 `GetBlockIdx() < rankSize` 的 block 执行；`SyncAll` 与 `CrossCoreSetFlag` 在守卫**之外**，全 AIV 执行 |
| 6 | Finalize | 循环结束后 `allToAllScaleA_.Finalize()` + `allToAllA_.Finalize()`（无分核 guard，全 AIV 执行） |
| 7 | 无环形回压 | 每轮独立，无 bufferCount 槽位复用（与 GET 的环形回压不同，见 §3.4） |

#### 流水线要素

| 要素 | 代码 | 含义 |
|:---|:---|:---|
| 通信 | `Commit()` × 2 + `Wait<BARRIER_DEVICE>()` × 1 | PUT 模式：把本 rank A/scaleA 推到远端 Win |
| 同步 | `SyncAll<true>()` | 保证所有 AIV block 完成本轮通信 |
| 通知 AIC | `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` | AIC 侧 `WaitTile(tid)` 的配对端 |
| 分核保护 | `GetBlockIdx() < rankSize` | 只有前 rankSize 个 block 下发通信（要求 rankSize ≤ BlockNum） |

### 4.2 AIC 等待机制

AIC 在 kernel 内经 `CommPolicy::WaitTile` 等待通信数据，策略类由模板注入。

#### 等待机制契约

| 要素 | 机制 | 锚点 |
|:---|:---|:---|
| 策略注入 | `UdmaCommWaitPolicy::WaitTile(tileIdx)` 底层为 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)` | `all_to_all_mx_quant_matmul_udma_impl.h` `struct UdmaCommWaitPolicy` |
| 依赖计算 | `CalcDependTileIdx(mPos + blockM - 1, headTileSize, totalTiles)` 由 block 末行 mPos 推导依赖的通信 tile，越界钳到 `totalTiles - 1` | `quant_matmul_mx_kernel.h` `CalcDependTileIdx` |
| 按位去重 | `waitedMask`（`uint32_t`）按位记录已 wait 的 tile，同一 tile 只 wait 一次 | `quant_matmul_mx_kernel.h` `ProcessSingleBatch` |
| 尾部兜底 | 循环结束后遍历全部 tile，对未 wait 的位补 `WaitTile`（drain 兜底，非 LOCAL 模式） | `quant_matmul_mx_kernel.h` `ProcessSingleBatch` 末尾 |

> ⚠️ **≤32 硬约束**：`waitedMask` 是 `uint32_t` → 通信 tile 总数 ≤ 32，超出会静默出错（高位 tile 的等待位溢出丢失）。

#### Flag 配对关系

| AIV 操作 | AIC 操作 | flagId | 含义 |
|:---|:---|:---|:---|
| `SetFlag<0x2, PIPE_MTE3>(tid)` | `WaitFlag<0x2, PIPE_MTE2>(tid)` | tid | 通信 tile tid 就绪，AIC 可消费 |

**配对规则**：AIV SetFlag 的 flagId 必须 == AIC WaitFlag 的 flagId；`SyncAll` 在 SetFlag 之前，保证 flag 可见性。

> Flag 机制（`<MODE, PIPE, flagId>` 三元组、完整编排图、flagId 选择规则）详见本文档 §3；上表为 AIC 消费者视角的锚点对照。

---

## 5. localMatmul 模式（0/1/2）

> 本节是 PUT 模式下 `localMatmul` 参数（0/1/2）的**完整决策参考**，覆盖模式选择、L0C 容量约束、AtomicAdd 精度风险、PipeBarrier 修复方案。性能/精度数字均为开发历史数据，仓内不可复现。
>
> **适用场景**：PUT 模式（`all_to_all_quant_matmul` 基底）下开发 ReduceScatter / AllToAll+Matmul 等通算融合算子时，需要选择 localMatmul 模式的 Architect 和 Developer。

### 5.1 三模式定义

`localMatmul` 是 Host 侧 tiling 中的 `uint32_t` 字段（`apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h` `allToAllMatmulTilingData::localMatmul`，官网注释："0：不使能；1：使能atomiadd"），控制 PUT 模式下 AIC 的计算编排。分支逻辑在 `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h`（`Run`/`SetupParams`）与 `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`（`QuantMatmulMxKernel::Init`/`ProcessSingleBatch`）：

| localMatmul | MatmulMode | AIC 执行顺序 | AtomicAdd | splitKNum | 适用场景 |
|:---:|:---|:---|:---:|:---:|:---|
| **0** | REMOTE-only | 遍历全部 rank 的 A；self rank 数据本地 GM 直读（`gmALocal`，`ProcessSingleBatch` REMOTE 分支 `rank==rankId && localMatmul!=1` 时改切 local 地址），但在 per-tile wait 之后才计算 → L0C 累加 → 单次 fixpipe → C | 不需要 | rankSize | 融合基线模式（官网 ST main.cpp 固定 `localMatmul = 0`） |
| **1** | LOCAL+REMOTE | **RunLocalMatmul**（本 rank A × B → C，直读 GM，首次写入）→ **SetAtomicAdd** → **RunMatmul**（远端 A × B → C，AtomicAdd 累加）→ **SetAtomicNone** | REMOTE 阶段开启 | rankSize-1 | **推荐**：本地 A 可前置计算，与 AIV PUT 并行 |
| **2** | DEFERRED_SYNC | per-tile 内：本 rank A × B → **L0C**（不 fixpipe）→ wait_flag → 远端 A × B → L0C 累加 → 最后一次触发单次 fixpipe → C | 不需要 | rankSize | L0C 容量足够时 |

> - `waitedMask` 为 `uint32_t`（`quant_matmul_mx_kernel.h` `ProcessSingleBatch` 内局部变量）→ 通信 tile 总数 ≤32 是硬约束。
> - LOCAL 阶段 `splitKNum=1`：本 rank 部分和单次 fixpipe 直写 C；仅 REMOTE 阶段逐 rank 在 L0C/GM 上累加。
> - `splitKNum` 赋值见 `all_to_all_mx_quant_matmul_udma_impl.h` `SetupParams`：localMatmul==1 → `rankSize - 1`；localMatmul==0/2 → `rankSize`。
> - mode 0 **不开启 AtomicAdd**：`isAtomicAdd_` 仅在 `matmulMode==REMOTE && localMatmul==1` 时置位（`quant_matmul_mx_kernel.h` `QuantMatmulMxKernel::Init`）。
> - ⚠️ 官网 ST（`apace/tests/st/all_to_all_quant_matmul/src/main.cpp`）仅覆盖 `localMatmul = 0`；mode 1/2 在 kernel 分支中存在但无 ST 样例。
> - ⚠️ **mode 2 仅 UDMA impl 可达**：`localMatmul==2` → `DEFERRED_SYNC` 的选择分支只存在于 `all_to_all_mx_quant_matmul_udma_impl.h` `RunMatmul()`。hcomm impl（`all_to_all_mx_quant_matmul_hcomm_impl.h`，即 APACE 通信基础 API 经 HCOMM 库驱动 CCU 引擎的变体）对任何 `localMatmul != 0` 都做 LOCAL 前置 + 恒 REMOTE（`splitKNum = rankDim - 1`、不开 AtomicAdd），而 kernel REMOTE 分支仅 `localMatmul==1` 才跳过 self → hcomm 下 `localMatmul==2` 的 self 被 LOCAL 前置与 REMOTE 各算一次、mmad 计数与 `splitKNum` 错配，**静默产生错误结果（无任何报错）**。使用 mode 2 必须确认走 UDMA impl。

### 5.2 模式选择决策树

```
算子语义需要 ReduceScatter（M 轴输出切分 + K 轴部分和累加）？
├── 是 → 优先 localMatmul=1（通算并行 + AtomicAdd）
│   ├── 风险：BF16 AtomicAdd 精度损失（开发历史数据：max_rel_diff ≈ 0.79%，容差 1e-2 内）
│   ├── 风险：MTE 异常（PipeBarrier 修复，见 §5.4）
│   └── 收益：通算并行（AIC local matmul 与 AIV PUT 重叠），性能最优
│
├── 精度要求极高（rtol < 1e-3）→ localMatmul=2（DEFERRED_SYNC）
│   ├── 前提：L0C 容量足够（见 §5.3）
│   ├── 收益：L0C 全 FP32 累加，单次 fixpipe，精度最好
│   └── 代价：self 计算与通信的重叠粒度为 per-tile，掩盖效果弱于 mode 1 的整段前置
│
└── 融合基线（官网 ST 默认）→ localMatmul=0（REMOTE-only）
    ├── 收益：最简单，无需 RunLocalMatmul
    └── 注意：self/remote 均在依赖 tile 的 wait 之后才计算，重叠粒度受 comm tile 限制（CalcDependTileIdx + waitedMask 去重）
```

**选择原则**：
1. **默认选 localMatmul=1** — 通算并行是 MC2 的核心价值，性能比 localMatmul=2 提升约 27%（开发历史数据）
2. **精度不达标时回退到 localMatmul=2** — 但必须先尝试 PipeBarrier 修复（见 §5.4），不可直接回退
3. **localMatmul=0 是融合基线模式** — 官网 ST 固定下发 0，不是纯 AllToAll 专用

### 5.3 L0C 容量约束（DAV_3510）

DAV_3510 的 L0C 容量为 **256 KB**。所有模式下，单个 tile 内各 rank 的部分和都**累加进同一块 L0C FP32 累加器**（mmad 第 8 参 `0` = reset、递增 = 累加、计满 `splitKNum` 触发单次 fixpipe；REMOTE 与 DEFERRED_SYNC 分支均如此，见 `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h` `ProcessSingleBatch` 及 Blaze `block_mmad_qbmm_mx.h`——Blaze 头文件来自 ops-tensor 仓，由 `cmake/third_party/ops-tensor.cmake` 按 pin 拉取，asc-devkit 安装内无 blaze 头文件），因此 L0C 需求**与 rankSize 无关、dtype 固定为 FP32，且对三种 localMatmul 模式完全相同**：

```
L0C 需求 = baseM × baseN × 4B（FP32）

约束：L0C 需求 ≤ 128KB（dbL0C ping-pong 半区）或 ≤ 256KB（单缓冲）

示例：
  baseM=128, baseN=128 → 128×128×4 = 64KB  ≤ 128KB ✅ 双缓冲可用 localMatmul=2
  baseM=256, baseN=256 → 256×256×4 = 256KB > 128KB ❌ 双缓冲不可用；=256KB ⚠️ 单缓冲临界
  baseM=128, baseN=256 → 128×256×4 = 128KB = 128KB ⚠️ 双缓冲临界
```

> `baseM`/`baseN` 由 `QuantMatmulTilingSwat::GetTilingData` 推导，Developer 可在 Host 侧打印确认（`QuantMatmulTilingBase::PrintTilingData` 会自动打印）。
> `dbL0c`（L0C 双缓冲）> 1 时实际可用容量减半（128KB ping-pong 半区），约束更严格。

**该约束对三种模式一视同仁，mode 2 相对 mode 1 没有额外 L0C 负担** — REMOTE 阶段（mode 0/1）与 DEFERRED_SYNC（mode 2）都是把各 rank 部分和累加进同一块 L0C 累加器（首份 reset、末份触发 fixpipe）；mode 1 的 AtomicAdd 发生在 fixpipe 写 GM 时，并不改变 L0C 占用；mode 1 的 LOCAL 阶段单发 mmad（`splitKNum=1`）同样只占这一块累加器。L0C 容量不足时三种模式同样不可用，只能调小 `baseM`/`baseN`。

### 5.4 MTE 异常修复方案（localMatmul=1 的关键修复）

#### 5.4.1 问题现象

localMatmul=1 模式下，RunLocalMatmul 和 RunMatmul 之间**如果没有 PipeBarrier**，可能触发（开发历史数据）：

```
aclError:507015 (timeout or trap error)
MTE error info: 非零
所有核心超时
```

#### 5.4.2 根因分析

```
AIC 侧时序（无 PipeBarrier）：
  RunLocalMatmul()
    └─ Blaze BlockMmad → L0C → fixpipe → MTE3 → GM（首次写入 C）
       ↑ MTE3 pipeline 仍在排空中...
  SetAtomicAdd<CType>()    ← AtomicAdd 配置生效
  RunMatmul()
    └─ Blaze BlockMmad → L0C → fixpipe → MTE3 → GM（AtomicAdd 累加）
       ↑ MTE3 pipeline 的 LOCAL fixpipe 尚未完成，
         AtomicAdd 的 read-modify-write 读取了未完全写入的 GM 值
         → MTE 硬件异常（aclError:507015）
```

**核心矛盾**：LOCAL 阶段的 fixpipe（MTE3 管线）和 REMOTE 阶段的 SetAtomicAdd + AtomicAdd fixpipe 共享同一 MTE3 管线，如果 LOCAL 的 MTE3 尚未排空，REMOTE 的 AtomicAdd 会在 GM 上做 read-modify-write 时读到未完全写入的旧值，触发 MTE 异常。

#### 5.4.3 修复方案（推荐补丁，官网未合入）

> ⚠️ **官网现状**：`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` `Run()` 当前为 `RunLocalMatmul()` 直接接 `RunMatmul()`，**两者之间无 PipeBarrier**。以下为推荐修复方案，Developer 在使用 localMatmul=1 时应添加。

在 `RunLocalMatmul()` 和 `RunMatmul()` 之间添加 `PipeBarrier<PIPE_ALL>()`：

```cpp
// 基于官网 Run() 结构的最小补丁（★ 行为新增）
void Run() {
    if ASCEND_IS_AIV {
        RunAllToAll();
    }

    if ASCEND_IS_AIC {
        if (tilingData_->localMatmul == 1) {
            RunLocalMatmul();  // LOCAL: fixpipe → GM（无 AtomicAdd）

            PipeBarrier<PIPE_ALL>();  // ★ 确保 LOCAL 阶段 MTE3 fixpipe 排空后再进 REMOTE

            RunMatmul();      // REMOTE: AtomicAdd 累加
        } else {
            RunMatmul();
        }
    }
}
```

> `PipeBarrier<PIPE_ALL>()` 是 AscendC API，强制等待所有 pipe（MTE2/MTE3/VEC/CUBE/FIX）的操作完成后再继续。
> 详见 `ascendc-api-best-practices` skill `references/api-pipeline.md`。

#### 5.4.4 修复验证

修复后的时序：

```
AIC 侧时序（有 PipeBarrier）：
  RunLocalMatmul()
    └─ Blaze BlockMmad → L0C → fixpipe → MTE3 → GM
       ↑ MTE3 pipeline 排空中...
  PipeBarrier<PIPE_ALL>()   ← ★ 等待所有 pipe 完成（LOCAL fixpipe 保证写入 GM）
  SetAtomicAdd<CType>()    ← AtomicAdd 配置生效（GM 中 C 值已完整）
  RunMatmul()
    └─ Blaze BlockMmad → L0C → fixpipe → MTE3 → GM（AtomicAdd read-modify-write）
       ↑ 读到完整的 LOCAL 写入值，正确累加
       → 无 MTE 异常 ✅
```

#### 5.4.5 实际案例（开发历史数据）

`quant_matmul_reduce_scatter` 算子开发过程中的修复记录（⚠️ 数据来自开发历史，仓内不可复现——官网无 RS 算子与 ST）：

| 阶段 | localMatmul | PipeBarrier | 结果 | Task Duration |
|:---|:---:|:---:|:---|:---|
| 初始开发 | 1 | 无 | ❌ aclError:507015（MTE 异常） | — |
| 添加 PipeBarrier | 1 | 有 | ✅ 6/6 PASS | 143 us |
| 错误回退（未重试） | 2 | — | ✅ 6/6 PASS | 196 us（-27%） |
| 修正回退 | 1 | 有 | ✅ 6/6 PASS | 143 us |

### 5.5 AtomicAdd 精度风险分析

#### 5.5.1 精度损失机制

localMatmul=1 的 AtomicAdd 累加分两阶段：

1. **LOCAL 阶段**：`C = A_local × B_local` → Blaze fixpipe 将 L0C 中的 FP32 结果截断为 BF16 写入 GM
2. **REMOTE 阶段**：每个远端 rank 的部分和 → L0C FP32 累加 → fixpipe 截断为 BF16 → AtomicAdd 到 GM（BF16 + BF16）

**理想情况**（localMatmul=2）：所有 rank 的部分和在 L0C 中以 FP32 累加，最后一次 fixpipe 截断为 BF16 → 精度最好。

**精度损失来源**：
- LOCAL 结果先截断为 BF16，再被 REMOTE AtomicAdd 累加（多一次 FP32→BF16 截断）
- AtomicAdd 对 BF16 做 read-modify-write，硬件实现的累加精度可能低于 FP32 加法

#### 5.5.2 已知精度基线（开发历史数据）

> ⚠️ 下表数值（6/6 PASS、143us vs 196us、-27%、0.79%/0.78%）来自开发历史，仓内不可复现（官网无 RS 算子与 ST）。

| 算子 | dtype | rankSize | max_rel_diff | 容差 | 达标 |
|:---|:---|:---:|:---|:---|:---|
| quant_matmul_reduce_scatter | FP8 E4M3FN × E4M3FN → BF16 | 4 | 0.79% | rtol=atol=1e-2 | ✅ |
| quant_matmul_reduce_scatter | 同上 | 2 | 0.78% | 同上 | ✅ |

> AtomicAdd 的精度损失在 FP8 量化的整体精度容差内可忽略（FP8 本身的量化误差远大于 AtomicAdd 截断误差）。

#### 5.5.3 不达标时的回退路径

如果 localMatmul=1 的精度不达标（rtol/atol 超容差）：

1. **确认已添加 PipeBarrier** — MTE 异常不是精度问题，但可能导致全零输出被误判为精度问题
2. **回退到 localMatmul=2** — L0C 全 FP32 累加，精度最好
3. **回退前提**：L0C 容量约束满足（见 §5.3）
4. **回退代价**：性能损失约 27%（无通算并行，开发历史数据）

### 5.6 Host 侧 localMatmul 配置

在 `main.cpp` 中设置 `tilingData.localMatmul`：

> **注意**：官网 ST（`apace/tests/st/all_to_all_quant_matmul/src/main.cpp`）固定使用 `localMatmul = 0`（REMOTE-only，无需 AtomicAdd，最简单）。`localMatmul = 1` 为推荐优化方向（通算并行），但需添加 PipeBarrier 修复（见 §5.4）。

| 场景 | 推荐值 | 说明 |
|:---|:---:|:---|
| 默认（官网 ST） | 0 | REMOTE-only，无需 AtomicAdd |
| 通算并行优化 | 1 | LOCAL+REMOTE+AtomicAdd，需 PipeBarrier 修复 |
| 精度优先 | 2 | DEFERRED_SYNC，L0C 全 FP32 累加 |
| 融合基线 | 0 | 官网 ST 默认 |

> **注意**：`all_to_all_matmul_tiling_data.h` 中的默认值 `uint32_t localMatmul{0}` 与实际使用值可能不一致，Developer 必须在 `main.cpp` 中显式赋值。

### 5.7 速查表

| 问题 | 答案 | 详见 |
|:---|:---|:---|
| 默认选哪个模式？ | localMatmul=1 | §5.2 |
| localMatmul=1 报 507015？ | 加 `PipeBarrier<PIPE_ALL>()` | §5.4 |
| 精度不达标？ | 先确认 PipeBarrier 已加，再回退 localMatmul=2 | §5.5.3 |
| localMatmul=2 什么时候能用？ | L0C 容量足够（见公式） | §5.3 |
| localMatmul=0 什么时候用？ | 融合基线模式（官网 ST 默认） | §5.2 |
| AtomicAdd 精度损失多大？ | max_rel_diff ≈ 0.79%（FP8→BF16, rank=4，开发历史数据） | §5.5.2 |
| 回退后性能损失多少？ | 约 27%（开发历史数据） | §5.4.5 |
| 有无不用 AtomicAdd 的方案？ | ReduceScatter 替代实现思路 | §6 |

### 5.8 替代方案：ReduceScatter 语义（无 AtomicAdd）

> 官网仓无 reduce_scatter 算子实现（无 epilogue/AIV ReduceAdd 代码可引用）。ReduceScatter 语义的替代实现思路（AllToAll PUT + AtomicAdd 路径的原理级推导，官网无对应算子样例）见本文档 §6。

| 条件 | 推荐方案 |
|:---|:---|
| K 轴切分 + 需要 AtomicAdd 通算并行 | localMatmul=1（§5.2） |
| 精度不达标且 L0C 容量足够 | localMatmul=2（§5.2） |
| 融合基线 | localMatmul=0（§5.2） |
| 不能用 AtomicAdd | 见本文档 §6 |

---

## 6. 多对象复用与扩展语义推导

### 6.1 winOffset 多对象复用

官网 PUT 算子（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` 的 `RunAllToAll`）有**两个通信对象**（data 和 scale），复用同一 channel，通过不同 `winOffset` 区分 Win 区数据段。

- **同 channel 单次 Wait**：scale 与 data 的通信复用同一 channel，channel 级 Drain 覆盖两个对象，故只对 `allToAllA_` 调一次 Wait 即可（代码注释："scale的通信和a矩阵的通信使用同一channel，因此只需要wait一次"）。

当 data 和 scale 两个通信对象复用同一 Win 区时，通过 `winOffset` 区分段（官网 `AllToAllMxQuantMatmulUdmaImpl::Init` 中 `allToAllScaleA_.Init(..., baseParams_.rankSize * baseParams_.rankDataBytes)`）：

```
rankDataBytes = axisM × axisKa × sizeof(AType)
winOffset_scale = rankSize × rankDataBytes
```

**Win 区布局（winOffset 复用）**：

```
commBufferAddrs[rankId] → ┌──────────────────────────────┐
                          │  data chunk[0]                │  ← winOffset=0
                          │  data chunk[1]                │
                          │  ...                          │
                          │  data chunk[rankSize-1]       │
                          │  scale chunk[0]               │  ← winOffset=rankSize×rankDataBytes
                          │  scale chunk[1]               │
                          │  ...                          │
                          │  scale chunk[rankSize-1]      │
                          └──────────────────────────────┘
```

> **注意**：每个通信对象需要独立的 UB commBuf（COMM_WORKSPACE_SIZE = 512B），两个对象共需双对象翻倍 + barrier UB（UB 预算详见 communication.md）。

> PUT 算子级编排验收标准见本文档 §4。

### 6.2 ReduceScatter 替代实现：AllToAll PUT + AtomicAdd

ReduceScatter 语义（每 rank 输出 C 的 M/rank 段，C = Σ_i A_i × B_i）可通过复用已有 AllToAll PUT 原语 + `SetAtomicAdd` 实现，**无需新增 block 原语**。本节为纯原理级模式推导，官网仓暂无该模式的算子样例。

> **API 参考**：`SetAtomicAdd` 详见 `ascendc-api-best-practices` skill `references/api-atomic.md`。

#### 适用条件（参数化决策）

```
算子语义是"多 rank 部分和累加 + 输出按轴切分"？
├── 是 → 可用 AllToAll PUT + AtomicAdd 模式
│   ├── A 按切分轴（如 K）切分到各 rank
│   ├── B 全量复制（每 rank 持有完整 B）或按同轴切分
│   ├── 输出按另一轴（如 M）切分
│   └── 需要跨 rank 累加部分和
└── 否 → 考虑 GET 模式或新原语
```

#### 数据分布模式

```
rank i 持有：A_i[M × K_local], B_full[K × N]（全量复制）
输出：rank r 获得 C[r*M/rank : (r+1)*M/rank, :] = Σ_i A_i × B_i 的 M/rank 段

通信：rank i 将 A_i 按 M/rank 段 PUT 到各 target rank
计算：每 rank 遍历所有 rank 的 A 段 × 对应 B 的 K 段，AtomicAdd 累加
```

#### Win 区布局

```
Win[rank r]:
  [src_0 的 A_0[r*M/rank:(r+1)*M/rank, :],  K_local 列]   ← winOffset=0
  [src_1 的 A_1[r*M/rank:(r+1)*M/rank, :],  K_local 列]
  ...
  [src_{R-1} 的 A_{R-1}[r*M/rank:..., :],   K_local 列]
  [src_0 的 scaleA_0[r*M/rank:..., :],      scaleK 列]    ← winOffset=rankSize×rankDataBytes
  ...
```

> `winOffset` 复用机制详见 §6.1。

#### AtomicAdd 时序不变量

```
AIC 侧计算编排：
  1. 本地部分先算：本 rank A_local × B 对应 K 段 → C[my_M]
     无需 AtomicAdd（首次写入，C 地址无旧值）
  2. SetAtomicAdd<CType>()  ← 开启原子加
  3. 远端部分：遍历远端 A 段 × 对应 B K 段 → C[my_M]
     Fixpipe 原子加自动累加到 C[my_M]
  4. 关闭 AtomicAdd
```

#### 泛化规则

任何"多 rank 计算部分和 + 输出按轴切分"语义的算子，都可按以下步骤推导实现方案：

1. **确定通信内容**：通信输入数据（PUT）还是结果数据（GET）？
   - 输入数据 PUT → 后续 AIC 计算 + AtomicAdd 累加
   - 结果数据 GET → 需要额外 reduce 步骤
2. **确定累加方式**：
   - Fixpipe AtomicAdd（PUT 模式）→ 无需额外 reduce
   - Host/kernel reduce（GET 模式）→ 增加 D2H 开销或 kernel 内 reduce 复杂度
3. **选择通信原语**：AllToAll PUT（已有）或 AllToAll GET（钩子已就绪，官网暂无算子样例）
4. **确定 B 分布**：PUT+AtomicAdd 要求 B 全量复制；GET 可 K-split

---

## 后续阅读

- communication.md — 通信接口契约（四段式/钩子/TeamBarrier/CrossCore flag 机制）
- compute.md — 计算接口（Blaze/MatmulMode/L0C）
- operator-anatomy.md — 完整算子骨架
- ascendc-api-best-practices skill references/api-atomic.md — SetAtomicAdd 约束
