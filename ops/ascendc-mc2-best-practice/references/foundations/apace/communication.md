# apace 通信原理与接口

> 本文档覆盖 apace 算子的通信侧：通信原理（URMA/Win 区模型、AIV 驱动通信）、通信框架接口（CollectiveComm 四段式 + GET/PUT 钩子）、同步接口（TeamBarrier/CrossCore flag/SyncAll）、host 侧建链机制。

## 目录

1. [通信原理：URMA 与 Win 区模型](#1-通信原理urma-与-win-区模型)
2. [通信框架接口：CollectiveComm 四段式](#2-通信框架接口collectivecomm-四段式)
3. [GET/PUT 钩子职责](#3-getput-钩子职责)
4. [同步接口](#4-同步接口)
   - 4.1 [TeamBarrier（跨卡）](#41-teambarrier跨卡)
   - 4.2 [CrossCoreSetFlag/WaitFlag（跨核）](#42-crosscoresetflagwaitflag跨核)
   - 4.3 [SyncAll（块间）](#43-syncall块间)
5. [通信上下文 CommContext](#5-通信上下文-commcontext)
6. [Host 侧建链机制](#6-host-侧建链机制)
7. [扩展通信原语指南](#7-扩展通信原语指南)
- [常见陷阱](#常见陷阱)
- [后续阅读](#后续阅读)

---

## 1. 通信原理：URMA 与 Win 区模型

```
kernel/<op>/<op>_impl.h
    │
    ├── CollectiveComm<Op, Mode, T, Barrier>   ← 统一类型别名（编译期分发）
    │       │
    │       ├── CollectiveCommBase<Impl,...>    ← CRTP 基类（公共逻辑）
    │       │       │
    │       │       └── 4 钩子: PostInit / DoCommit / DoWait / DoFinalize
    │       │
    │       ├── AllToAllCommGetImpl             ← GET 模式实现（ReadNbi，官网暂无算子使用）
    │       ├── AllToAllCommPutImpl             ← PUT 模式实现（WriteNbi）
    │       └── AllGatherCommPutImpl            ← AllGather PUT 实现
    │
    ├── TeamBarrier                             ← 跨卡同步原语（UBMEM 协议）
    │
    └── Hcomm<COMM_PROTOCOL_UBC_CTP>            ← 底层通信对象（ReadNbi/WriteNbi/Drain）
```

### 机制说明

- **Win 区共享 GM**：Win 区是各 rank 共享的 GM 区域。每 rank 的 Win 区基地址存于 `commBufferAddrs[]`，由 Host 侧 `CommChannelBuilder` 建链后填充（见 §5、§6）；kernel 侧按地址公式直接读写远端 Win 区（见 §3）。
- **URMA channel**：每 peer rank 的 URMA channel 句柄存于 `channelHandles[]`（self 不填充，见 §5），经 UDMA 引擎下发通信。
- **AIV 核驱动通信**：UDMA 引擎下通信由 AIV 核发起（ReadNbi/WriteNbi），AIC 专注 Matmul 计算，两侧经 CrossCore flag 同步（见 §4.2）。
- **Hcomm 基础原语**：底层通信对象为 `Hcomm<COMM_PROTOCOL_UBC_CTP>`（ReadNbi/WriteNbi/Drain）；完整签名与约束见 `ascendc-api-best-practices` skill `references/api-hcomm.md`。

### 文件位置（官网仓 `apace/` 相对路径）

| 组件 | 文件 |
|:---|:---|
| 统一 API + 编译期分发 | `apace/block/aiv_comm/collective_comm_api.h` |
| CRTP 基类 | `apace/block/aiv_comm/collective_comm_base.h` |
| 通信上下文结构体 | `apace/block/aiv_comm/collective_comm_context.h` |
| AllToAll GET | `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` |
| AllToAll PUT | `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` |
| AllGather PUT | `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` |
| TeamBarrier | `apace/block/aiv_comm/barrier/barrier_ubmem.h` |
| Host 建链 builder | `apace/utils/comm_channel_builder.h` |
| CommTilingData 定义 | `apace/tiling/comm_tiling_data.h` |
| PUT 算子样例（AllToAll） | `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` |
| PUT 算子样例（AllGather） | `apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h` |

### winOffset 多对象复用

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

> **注意**：每个通信对象需要独立的 UB commBuf（COMM_WORKSPACE_SIZE = 512B），两个对象共需双对象翻倍 + barrier UB（见 §4.1 UB 预算）。

### UDMA vs HCCL windows 双引擎

| 引擎 | 底层 API | CommContext | 使用场景 | 直调支持 |
|:---|:---|:---|:---|:---|
| **UDMA** | `Hcomm::ReadNbi/WriteNbi/Drain` | 需要 `CommContext{udmaCtx, ubmemCtx}` | AIV 核驱动通信 | **是** |
| **HCCL windows** | `GetHcclContext<0>()` | 不需要 `CommContext` | 委托模式 epilogue | 否（官网存在 CCU hcomm 变体 `all_to_all_mx_quant_matmul_hcomm_impl.h`，但无 `__global__` 直调入口） |

#### 验收条件

| 模式 | 入口签名特征 | tiling_data.h 特征（启发式判据） |
|:---|:---|:---|
| UDMA | 含 `__gm__ CommContext*` 参数 | 该模式使用的 tiling 结构含 `CommContext` 聚合体 |
| HCCL windows | 含 `GetHcclContext`，无 `CommContext` 参数 | 该模式使用的 tiling 结构不含 `CommContext` |

> tiling_data.h 判据是启发式：同一头文件可共存多种 tiling 结构（如 `all_to_all_matmul_tiling_data.h` 同时定义 `CommContext` 和 CCU 变体用的 `ccuAllToAllMatmulTilingData`），判据应针对具体使用的结构而非整个文件。

> **注意**：HCCL windows（`GetHcclContext`）是 kernel 级 API，与 blaze-shmem 路线禁止的 HCCL 高阶 API（`Hccl::AllReduce` 等服务端调度 API）不同。apace 路线允许 HCCL windows。

---

## 2. 通信框架接口：CollectiveComm 四段式

### 2.1 四段式语义

| 阶段 | 语义 | 阻塞性 |
|:---|:---|:---|
| `Init()` | 初始化通信对象，分配 targetRank，计算偏移 | 同步 |
| `Commit()` | 发起当前 tile 的通信（GET 拉 / PUT 推），非阻塞返回 | **非阻塞** |
| `Wait()` | 等待 Commit 发起的通信完成 | 阻塞 |
| `Finalize()` | 收尾（最终 barrier 等） | 同步 |

Commit 的非阻塞特性是通算流水的关键——AIV 可以在 Commit 后、Wait 前插入其他指令。

### 2.2 CRTP 基类 CollectiveCommBase

`CollectiveCommBase`（`apace/block/aiv_comm/collective_comm_base.h`）使用 CRTP 模式，提供公共逻辑，派生类只需实现 4 个钩子（`PostInit` / `DoCommit` / `DoWait` / `DoFinalize`）。

#### Init 完整签名（`CollectiveCommBase::Init`）

```cpp
template<uint8_t BarrierMode = BARRIER_BOTH>
__aicore__ inline void Init(
    __gm__ CommUdmaContext* udmaCtx,   // UDMA 通信上下文
    Barrier& barrier,                   // TeamBarrier 实例
    const CommTilingData& tilingData,   // 通信切分参数（5 字段，见 apace/tiling/comm_tiling_data.h）
    GM_ADDR localAddr,                  // 本地 GM 地址（GET=目标 cGM，PUT=源 aGM）
    __ubuf__ uint8_t* commbuf,          // UB 通信 workspace（COMM_WORKSPACE_SIZE = 512B）
    uint32_t totalJobs,                 // 参与通信的核数（通常=rankSize）
    uint32_t jobIndex,                  // 当前核索引（GetBlockIdx()）
    uint64_t winOffset = 0);            // Win 区偏移（多对象复用用）
// 返回值：void。totalTiles 通过 GetCommTurn() 获取，另有 GetCommByteSize() 访问器
```

BarrierMode 常量（同文件定义）：`BARRIER_NONE=0`、`BARRIER_DEVICE=1`、`BARRIER_CORE=2`、`BARRIER_BOTH=3`。

#### Init 不变量

| 不变量 | 说明 |
|:---|:---|
| 保存上下文 | udmaCtx、barrier、localAddr、tilingData、commBuf、winOffset |
| 底层 comm_ 初始化 | Hcomm 对象使用 commBuf 的 COMM_WORKSPACE_SIZE（512B）workspace |
| jobIndex → targetRank 自动分核映射 | `targetRankPerCore = ceil(rankSize / totalJobs)`；`targetRankStart = jobIndex * targetRankPerCore`；`targetRankCnt` 三分支：`targetRankStart + targetRankPerCore <= rankSize` 取 `targetRankPerCore`，`targetRankStart < rankSize` 取 `rankSize - targetRankStart`，否则钳到 0 |
| **早退语义** | `jobIndex >= totalJobs` 时 Init 提前 return，字段未初始化——调用方必须用 `GetBlockIdx() < rankSize` 守卫保护后续 Commit/Wait/Finalize（见本节末尾「AIV 分核保护惯例」） |
| chunk 大小计算 | 从 CommTilingData 的 5 字段推导：`chunkSize = splitAxisTileSize*splitAxisTileCnt + splitAxisTailSize*splitAxisTailCnt`；`chunkBytes_ = chunkSize * nonSplitAxisSize * sizeof(Dtype)`；`tileMaxByteSize_ = max(splitAxisTileSize, splitAxisTailSize) * nonSplitAxisSize * sizeof(Dtype)` |
| PostInit 钩子调用 | Init 末尾调用 `PostInit<BarrierMode>()`，派生类可在此插入前置逻辑（如 PUT 的 barrier） |
| 返回值 | **void**（totalTiles 经 `GetCommTurn()` 获取） |

#### 访问器

| 方法 | 语义 |
|:---|:---|
| `GetCommTurn()` | 返回 `splitAxisTileCnt + splitAxisTailCnt`（总通信轮次/tile 数） |
| `GetCommByteSize()` | 返回 `chunkBytes_ * rankSize`（本 rank 通信总字节数） |

#### Commit/Wait 不变量

| 阶段 | 不变量 |
|:---|:---|
| Commit | `remainingChunkSize_ <= 0` 则直接返回；计算当前 tile 大小（`currentTileIdx_ < splitAxisTileCnt` 取头块 `splitAxisTileSize`，否则取尾块 `splitAxisTailSize`，再钳到 `remainingChunkSize_`）；遍历 `targetRankCnt_` 个 targetRank 调用 `DoCommit<BarrierMode>(targetRankId, currentTileByteSize)`；更新 `currentTileIdx_++`、`slotByteOffset_ += rankSize * tileMaxByteSize_`、`tileByteOffset_`、`chunkByteOffset_ += chunkBytes_`、`remainingChunkSize_ -= currentTileSize` |
| Wait | 签名 `Wait<BarrierMode = BARRIER_BOTH>(bool waitLast = false)`。waitLast 早退语义（官网事实）：`if (waitLast && currentTileIdx_ != totalTiles - 1) return;`——在典型的 `Commit(); Wait(true);` 逐 tile 循环中（Commit 末尾 `currentTileIdx_++`），`DoWait` 仅在 `currentTileIdx_ == totalTiles - 1` 时执行一次（即倒数第二轮），**最后一轮通信不被 Drain**；`Wait(false)`（默认）每个 tile 都执行 `DoWait`。循环体：遍历 `targetRankCnt_` 个 targetRank 调用 `DoWait<BarrierMode>(targetRankId)`。⚠️ 使用 waitLast 模式（GET 语义）时需自行评估该行为是否满足时序要求 |

#### 受保护字段（基类提供，钩子可访问）

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `udmaCtx_` | `__gm__ CommUdmaContext*` | UDMA 通信上下文（rankId、rankSize、channelHandles、commBufferAddrs） |
| `tilingData_` | `const CommTilingData*` | 通信切分参数（5 字段） |
| `barrier_` | `Barrier` | 跨卡同步原语 |
| `comm_` | `Hcomm<COMM_PROTOCOL_UBC_CTP>` | 底层通信对象 |
| `localAddr_` | `GM_ADDR` | 本地 GM 地址（GET=目标，PUT=源） |
| `commBuf_` | `__ubuf__ uint8_t*` | UB 通信 workspace |
| `winOffset_` | `uint64_t` | Win 区偏移 |
| `chunkBytes_` | `uint64_t` | 一个完整 chunk 的字节数 |
| `currentTileIdx_` | `uint64_t` | 当前 tile 索引 |
| `tileByteOffset_` | `uint64_t` | 当前 tile 在 chunk 内的字节偏移 |
| `tileMaxByteSize_` | `uint64_t` | 最大 tile 字节数 |
| `slotByteOffset_` | `uint64_t` | 当前 slot 偏移（环形） |
| `targetRankStart_` | `uint32_t` | 本核负责的起始 targetRank |
| `targetRankCnt_` | `uint32_t` | 本核负责的 targetRank 数量 |
| `remainingChunkSize_` | `uint64_t` | 当前 chunk 剩余未通信字节数 |
| `chunkByteOffset_` | `uint64_t` | 当前 chunk 内字节偏移 |

#### AIV 分核保护惯例

多 block 场景下，AIV 侧的 Commit/Wait 必须包裹在 `if (GetBlockIdx() < rankSize)` 守卫中（配合 Init 早退语义——超出 rankSize 的 block 未初始化通信字段）。官网两个 PUT 算子的惯例（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` 的 `RunAllToAll`、`apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h` 的 `AllGatherProcess`）：守卫仅包 Commit/Wait，`SyncAll<true>()` 与 `CrossCoreSetFlag` **在守卫外**由所有 AIV block 执行，Finalize 也无守卫（全 AIV 执行）。移植时以官网 kernel 实现为准。

---

## 3. GET/PUT 钩子职责

GET/PUT 的算子级编排模式见 `fusion.md`，本节只描述钩子契约。

### 3.1 GET 钩子（AllToAllCommGetImpl）

> **官网暂无 GET 算子样例**：GET 钩子基础设施（`apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` 的 `AllToAllCommGetImpl`）已就绪并注册进 `CollectiveCommHelper<AllToAll, GET, ...>` 分发（`apace/block/aiv_comm/collective_comm_api.h`），但官网 kernel/ 下两个算子均为 PUT 模式，无 GET 使用方。本节为钩子契约级描述，地址公式与 self 跳过规则均可在该头文件中直接验证。

GET 模式 = 计算→通信：AIC 先算 C 写到 Win 区，AIV 从远端 Win 区拉回本 rank 的 C 段。

#### GET 钩子不变量表

| 钩子 | 不变量 | 违反后果 |
|:---|:---|:---|
| `PostInit()` | 空实现（GET 不需要前置 barrier） | 若加入 barrier，产生不必要的同步开销 |
| `DoCommit(targetRankId, tileByteSize)` | ① 按 `BarrierMode` 按需调用 `CrossCore()`/`CrossDevice()`（确保对端 AIC 已写入 Win 区） ② 跳过 `targetRankId == rankId`（self rank，直接 return） ③ `srcAddr = commBufferAddrs[targetRankId] + winOffset_ + slotByteOffset_ + rankId * tileMaxByteSize_` ④ `dstAddr = localAddr_ + targetRankId * chunkBytes_ + tileByteOffset_` ⑤ `ReadNbi` 返回值经 `ascendc_assert(ret == 0, ...)` 检查 | ① 遗漏 barrier → 读到未初始化数据 ② 未跳过 self → 自身 Win 区读写错误 |
| `DoWait(targetRankId)` | ① 跳过 `targetRankId == rankId`（self 直接 return） ② `Drain` 返回值经 `ascendc_assert(ret == 0, ...)` 检查 | ① 未跳过 self → 无谓 Drain |
| `DoFinalize()` | 按 `BarrierMode` 按需调用 `CrossCore()`/`CrossDevice()` | 遗漏 → 后续操作可能访问未完成通信的 buffer |

#### Barrier 机制

模板参数 `BarrierMode` 控制 barrier 行为（编译期 `if constexpr (BarrierMode & BARRIER_*)`，常量定义于 `apace/block/aiv_comm/collective_comm_base.h`）：
- `BARRIER_CORE`：跨核 barrier（`barrier_.CrossCore()`）
- `BARRIER_DEVICE`：跨设备 barrier（`barrier_.CrossDevice()`）

GET 的 `DoCommit()` 和 `DoFinalize()` 中 barrier 调用顺序为**先 Core 后 Device**。

#### GET 地址语义

```
远端 rank 的 Win 区布局:
┌─────────────────────────────────────────┐
│  rank 0 的槽位     rank 1 的槽位  ...   │  ← commBufferAddrs[targetRankId]
├─────────┬─────────┬─────────┬──────────┤
│ tileMax │ tileMax │ tileMax │  ...     │
│ ByteSize│ ByteSize│ ByteSize│          │
└─────────┴─────────┴─────────┴──────────┘
     ↑
     本 rank 的槽位 = slotByteOffset_ + rankId * tileMaxByteSize_

本地 cGM 布局:
┌─────────────────────────────────────────┐
│  rank 0 的 C 段   rank 1 的 C 段  ...   │  ← localAddr_
├─────────┬─────────┬────────────────────┤
│ chunkBytes │ chunkBytes │  ...          │
└─────────┴─────────┴────────────────────┘
     ↑
     本 tile 偏移 = targetRankId * chunkBytes_ + tileByteOffset_
```

#### 自跳过规则

GET 的 DoCommit/DoWait 和 PUT 的 DoCommit 都会跳过 `targetRankId == rankId`（self rank，直接 return），因为本 rank 的数据已在本地，不需要跨卡通信。注意 PUT 的 `DoWait` 对 self 仅跳过 `Drain`，**仍会执行** CrossDevice/CrossCore barrier（`apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 的 `AllToAllCommPutImpl::DoWait`：Drain 在 `if (targetRankId != rankId)` 内，barrier 在其外）。

### 3.2 PUT 钩子（AllToAllCommPutImpl）与 GET 差异

PUT 模式 = 通信→计算：AIV 先推数据到远端 Win 区，AIC 从 Win 区读取计算。实现见 `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 的 `AllToAllCommPutImpl`（AllGather PUT 变体见 `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` 的 `AllGatherCommPutImpl`，钩子结构相同，仅地址公式不同）。

#### 与 GET 的钩子差异

| 钩子 | GET | PUT |
|:---|:---|:---|
| `PostInit()` | 空 | `CrossDevice()+CrossCore()`（通知对端即将写入） |
| `DoCommit()` | `CrossCore()+CrossDevice()` → `ReadNbi`（拉） | `WriteNbi`（推），**无**前置 barrier；self rank 直接 return |
| `DoWait()` | `Drain`（self 直接 return） | `Drain`（仅非 self）→ `CrossDevice()+CrossCore()`（含 self） |
| `DoFinalize()` | `CrossCore()+CrossDevice()` | 空 |

> **顺序差异**：GET 钩子内 barrier 顺序是先 Core 后 Device；PUT 的 PostInit/DoWait 是先 Device 后 Core。以 `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` / `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 实际代码为准。

PUT 的 src/dst 与 GET 完全镜像：GET 从远端读，PUT 往远端写。PUT 地址公式（`AllToAllCommPutImpl::DoCommit`）：`srcAddr = localAddr_ + targetRankId * chunkBytes_ + currentTileIdx_ * tileMaxByteSize_`；`dstAddr = commBufferAddrs[targetRankId] + winOffset_ + rankId * chunkBytes_ + tileByteOffset_`。

> GET/PUT 的算子级编排模式见 `fusion.md`。

---

## 4. 同步接口

### 4.1 TeamBarrier（跨卡）

`TeamBarrier`（`apace/block/aiv_comm/barrier/barrier_ubmem.h`）是基于 UBMEM 协议的跨卡同步原语，替代 blaze-shmem 路线的 `aclshmemx_barrier_all_vec`。

#### 关键特性

- 基于 GM flag counter 递增 + 轮询远端 flag
- **支持部分核参与**（`totalJobs` / `jobIndex` 参数），非全核 barrier
- 仅 AIV 核执行（内部 `if ASCEND_IS_AIV` 保护；AIC 调用是空操作，不会报错，但应避免）
- UB 需求量固定 32 字节（`UB_SIZE = BARRIER_FLAG_SIZE = 32`）

#### 常量

```cpp
constexpr uint32_t BARRIER_FLAG_SIZE = 32;              // 每个同步槽 32B
constexpr uint32_t UB_SIZE = BARRIER_FLAG_SIZE;          // kernel 侧 ubOffset 累加用
constexpr uint32_t BARRIER_FLAG_ELEMS = 8;               // 32B / sizeof(int32_t)
```

#### Init

```cpp
__aicore__ inline void Init(
    __ubuf__ uint8_t* syncBuf,           // UB 同步缓冲（32B 对齐）
    __gm__ CommUbmemContext* ctx,        // barrier 通道上下文（含远端 flag 地址）
    uint32_t totalJobs,                   // 总 job 数（参与同步的核数）
    uint32_t jobIndex);                   // 当前核索引（GetBlockIdx()）
```

#### CrossDevice 机制（跨卡，仅 AIV）

`jobIndex_ >= totalJobs_` 时提前 return，否则：

1. 读本 rank per-job 槽（`commBufferAddrs[rankId] + 32 + jobIndex*32`），count+1
2. 先把 count 写到**基址 flag**（`commBufferAddrs[rankId]`，偏移 0），随后轮询**其他 rank 的基址 flag**（偏移 0）直到 ≥ count——注意轮询的是**跨步子集**：`step = min(totalJobs, rankSize)`，`for (i = jobIndex; i < nranks; i += step)`，跳过本 rank，并非轮询所有其他 rank 的 per-job counter
3. 轮询通过后才把 count 写回 per-job 槽
4. **无超时保护**：远端 rank 未就绪将无限等待挂死（不会 assert）。规避：确保所有 rank kernel 已 launch，且 `CreateDeviceContext` 后做了跨 rank host barrier（见 §6）

#### CrossCore 机制（跨核，仅 AIV）

`jobIndex_ >= totalJobs_` 时提前 return，否则：

1. 读本 job 的 localFlag（`commBufferAddrs[rankId] + 32 + totalJobs*32 + jobIndex*32`）
2. count+1 写回
3. 轮询其他 job 的 flag 直到 ≥ count（同样无超时，无限 do-while）

#### GM flag 区来源与预算

TeamBarrier 轮询的 GM flag 位于 `CreateDeviceContext` 内部分配的 2MB `BARRIER_BUF_SIZE` 区域（device ctx 之后），容量需求为 `(1 + 2×totalJobs) × 32B`/rank（基址 flag + CrossDevice 槽 + CrossCore 槽）。注意：`aclrtMemset` 清的是 HCCL **数据** buffer；barrier flag 区无显式 memset，零初值依赖 `HcclEngineCtxCreate` 的分配语义——新算子建议显式 memset 或验证该假设。

#### UB 预算

单通信对象：`COMM_WORKSPACE_SIZE`(512B) + `UB_SIZE`(32B) = **544B**。
data+scale 双通信对象：512×2 + 32 = **1056B**。

#### 常见错误

| 错误 | 后果 | 正确做法 |
|:---|:---|:---|
| `BARRIER_NONE` 但无外部 CrossCore flag 保证时序 | GET 读到 AIC 未写完的数据 | `Init<BARRIER_NONE>` 时必须靠 `CrossCoreWaitFlag` 保证 AIC 已写完 |
| barrier UB 按 64B 分配 | 多分配 32B | 按 `UB_SIZE = 32B` 分配 |
| 远端 rank 未启动 | CrossDevice **无限等待挂死**（无超时保护） | 确保所有 rank 已 launch + host 侧跨 rank barrier |
| AIC 侧调用 TeamBarrier | 空操作（内部 ASCEND_IS_AIV 保护），但语义混乱 | 仅在 AIV 侧调用 |

#### 与 SHMEM barrier 的区别

| 特性 | SHMEM `barrier_all_vec` | apace `TeamBarrier` |
|:---|:---|:---|
| 协议 | SHMEM/URMA | UBMEM |
| 参与者 | 全部核 | 支持部分核（jobIndex/totalJobs） |
| 实现机制 | SHMEM 库内部 | GM flag 轮询（用户可见） |
| 执行核 | AIV | AIV |
| 超时保护 | 依赖库实现 | **无**（无限等待） |

> 注：SHMEM 列以 SHMEM 文档为准，非 apace 仓实证。

### 4.2 CrossCoreSetFlag/WaitFlag（跨核）

CrossCore Flag 是 AIC↔AIV 跨核同步的核心机制。每个 flag 由 `<MODE, PIPE, flagId>` 三元组标识。完整签名、flagId 硬件规则与平台生效性见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`；GET/PUT 的 flag 编排不变量见 `fusion.md`。

#### MODE 常量

| MODE 值 | 常量名 | 含义 |
|:---|:---|:---|
| `0x2` | `CROSS_CORE_INNER_CUBE_VEC_SYNC` | Cube↔Vector 同核同步（apace 唯一使用） |

> 注：apace 代码中使用字面量 `0x2`（如 `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)`）；常量名 `CROSS_CORE_INNER_CUBE_VEC_SYNC` 是 MC2 框架惯例命名，非 apace 仓符号。

#### PIPE 选项

| PIPE | 含义 | 典型场景 |
|:---|:---|:---|
| `PIPE_FIX` | FixPipe（AIC 侧） | AIC 完成 fixpipe 输出后 SetFlag |
| `PIPE_M` | M（MAD/Cube 主流水，AIC 侧） | AIC 等待回压（WaitFlag） |
| `PIPE_S` | Scalar（AIV 侧） | AIV 等待 AIC 通知（WaitFlag） |
| `PIPE_MTE3` | Mte3（AIV 侧） | AIV 完成通信后 SetFlag 回压 |

#### flagId 选择规则

- 官网两个 PUT 算子均使用**轮次索引 `tid`/`round`** 作为 flagId（`CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)`）
- **硬件规则**（官方约束，详见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`）：
  - 模式 0/1/2 每核仅 **16 个 flagId（0-15）**，超出截断低 4bit——截断机制正是直接用无界 `tid` 作 flagId 能工作的硬件基础（Set/Wait 双方截断到同一 flag，配对保持）
  - 每个 flagId 对应计数器，Set/Wait 必须配对，否则未定义行为
  - **SyncAll 硬同步内部占用 flagId [11-14]**，官方不建议同时使用 CrossCoreSetFlag 与 SyncAll 硬同步——PUT 模式组合使用两者，当 `tid % 16 ∈ [11,14]` 时存在与 SyncAll 内部 flag 冲突的理论风险，移植到新平台需重新确认
  - Matmul 高阶 API 占用 flagId [0, 2N-1]（最多 [0,7]）；自定义 flagId 需避开此类保留区间
  - 同一核连续发出的 CrossCoreSetFlag，硬件不保证执行顺序
- `apace/utils/constant.h` 定义 `FLAG_ID_MAX = 16` 为**预留常量**（官网 apace 当前未使用）

### 4.3 SyncAll（块间）

`SyncAll<true>()` 是 AIV 块间硬同步原语，其内部占用 flagId [11-14]（见 §4.2 flagId 选择规则）。PUT 模式的逐轮编排用法（每轮 `SyncAll<true>()` 保证 WriteNbi 对端可见性）见 `fusion.md`；完整签名与平台生效性见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`。

---

## 5. 通信上下文 CommContext

聚合体 `CommContext{udmaCtx, ubmemCtx}` **不**在 `apace/block/aiv_comm/collective_comm_context.h` 中定义——该头文件仅定义 `CommUdmaContext` / `CommUbmemContext` 两个子结构及常量（`COMM_MAX_RANK_NUM`、`COMM_WORKSPACE_SIZE`）。聚合体 `CommContext` 由各算子在自己的 tiling_data.h 中定义：

| 算子 | 定义位置 | 命名空间 |
|:---|:---|:---|
| PUT（all_to_all_quant_matmul） | `apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h` | 全局命名空间 |
| AG（all_gather_quant_matmul） | `apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_tiling_data.h` | `Apace::AivComm`（文件内 `using Apace::AivComm::CommContext;` 导出到全局） |

### 结构

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `udmaCtx` | `CommUdmaContext` | UDMA 通信通道 |
| `ubmemCtx` | `CommUbmemContext` | Barrier 通道 |

### CommUdmaContext

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `rankId` | `uint32_t` | 本 rank ID |
| `rankSize` | `uint32_t` | 总 rank 数 |
| `channelHandles[]` | `uint64_t[COMM_MAX_RANK_NUM]` | 每 rank 的通信 channel 句柄 |
| `commBufferAddrs[]` | `uint64_t[COMM_MAX_RANK_NUM]` | 每 rank 的 Win 区基地址 |

### CommUbmemContext

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `rankId` | `uint32_t` | 本 rank ID |
| `rankSize` | `uint32_t` | 总 rank 数 |
| `commBufferAddrs[]` | `uint64_t[COMM_MAX_RANK_NUM]` | Barrier flag 的 GM 地址 |

> `COMM_MAX_RANK_NUM`（=64）和 `COMM_WORKSPACE_SIZE`（=512B）定义在 `apace/block/aiv_comm/collective_comm_context.h`。

### 填充细节

| 细节 | 说明 |
|:---|:---|
| `channelHandles[self]` 不填充 | builder 建链循环跳过 `peer == rankId`，本 rank 条目保持 0；kernel 侧 DoCommit/DoWait 必须跳过 self（见 §3 自跳过规则） |
| `commBufferAddrs[self]` | = 本地 HCCL buffer 地址（有效，用于本地 Win 区读写） |
| ctxTag 复用语义 | 同 tag 命中 `HcclEngineCtxGet` 会**直接复用已有 context 并跳过字段填充与建链**——这是特性；不同通信域/不同 group 必须用不同 tag |
| rankSize 上限 | `rankSize <= COMM_MAX_RANK_NUM`（64 卡），数组定长越界即静默错位 |

### CommContext 不变量

| 不变量 | 说明 |
|:---|:---|
| 传递方式 | 通过 `__gm__` 指针传递（`__global__` 入口的第一参数），不按值传递 |
| Host 构造 | Host 侧构造后写入 GM，kernel 通过指针读取 |
| tiling 按值 | tilingData 作为 `__global__` 入口参数按值传递 |

---

## 6. Host 侧建链机制

CommContext 的 `CommUdmaContext` 和 `CommUbmemContext` 不能手动赋值，必须通过 `CommChannelBuilder`（`apace/utils/comm_channel_builder.h`）创建 HCCL channel 后自动填充。

### Host 侧验收条件

| 验收条件 | 说明 |
|:---|:---|
| TCP 交换 RootInfo | rank0 生成 `HcclRootInfo` 并通过 TCP 广播给其他 rank（官网 ST 用 `apace/tests/st/utils/root_info_exchanger.h` 的 `RootInfoExchanger`） |
| 创建 HCCL comm | `HcclCommInitRootInfoConfig` 创建 `HcclComm` |
| HCCL 数据 buffer 清零 | builder 的 `AllocRegAndBuildChannels` 内部对 HCCL 内置 buffer 做 `aclrtMemset(buf, hcclBufSize, 0, hcclBufSize)`（清的是数据 buffer；barrier flag 区零初值依赖 engine 分配语义，见 §4.1） |
| CommChannelBuilder 填充 | 通过 `builder.CreateDeviceContext` 自动填充 `udmaCtx` 和 `ubmemCtx` |
| **跨 rank host barrier（强制）** | `CreateDeviceContext` 返回后必须做一次 rank 间 barrier（官网 ST 用 `RootInfoExchanger::Barrier()`），确保所有 channel 握手完成，再 launch kernel（`CreateDeviceContext` 头注释明确要求："调用方应在本函数返回后对 rank 间做一次 barrier"） |
| engine 一致性 | `HcclChannelAcquire` 与 `HcclEngineCtxCreate/Get/Copy` 必须使用同一 engine（apace 用 `BUILDER_COMM_ENGINE_AIV = 4`，定义于 `apace/utils/comm_channel_builder.h`），否则 `HcclEngineCtxGet` 复用失效 |
| ctxTag 唯一性 | 不同通信域用不同 ctxTag；同 tag 命中 `HcclEngineCtxGet` 会直接复用并跳过填充（见 `CommChannelBuilder::CreateDeviceContext` 头注释） |
| 资源生命周期 | devContext 由 HCCL engine 管理：随 `HcclCommDestroy` 释放，或显式 `HcclEngineCtxDestroy`（推断：释放路径未经官网验证；实证：AG ST 有 aclrtFree(devContext)，all_to_all ST 无——两份 ST 处置不一致）；builder 无清理接口。⚠️ 官网两份 ST 处置不一致：all_gather ST（`apace/tests/st/all_gather_quant_matmul/src/main.cpp`）有 `aclrtFree(devContext)`，all_to_all ST（`apace/tests/st/all_to_all_quant_matmul/src/main.cpp`）不释放——推荐范式：不单独释放，随 HcclCommDestroy 连带释放 |
| 禁止手动填充 | `channelHandles` 和 `commBufferAddrs` 必须由 `CommChannelBuilder` 通过 HCCL API 获取 |

### 禁止行为

| 禁止 | 原因 |
|:---|:---|
| 手动赋值 `channelHandles` / `commBufferAddrs` | 必须由 HCCL API 获取，手动赋值导致通信失败 |
| 依赖 `GetRankId()` / `GetRankSize()` 的时机 | builder 有独立 `Init()` 方法（内部调 `HcclGetRankId/HcclGetRankSize`）；`CreateDeviceContext` 内部也会获取。确保在 HCCL comm 创建之后调用 |

### CreateDeviceContext 不变量

| 步骤 | 不变量 |
|:---|:---|
| ctxTag 复用检查 | 先 `HcclEngineCtxGet(comm, ctxTag, engine, ...)`；命中已存在 context 直接返回复用，跳过建链与字段填充 |
| 创建 device context | `HcclEngineCtxCreate(comm, ctxTag, engine, totalSize, &devCtx)`；`totalSize = ctxSize + BARRIER_BUF_SIZE`（有 barrierCtx 时），`BARRIER_BUF_SIZE = 2MB`（函数内 constexpr） |
| 获取 rank 信息 | `HcclGetRankId` / `HcclGetRankSize`（自动获取，无需手动调用） |
| 填充 CommUdmaContext | `rankId`/`rankSize` + `AllocRegAndBuildChannels(URMA)` → `channelHandles[peer]` + `commBufferAddrs[peer]`（建链循环跳过 `peer == rankId`，self 的 channelHandle 保持 0；`commBufferAddrs[self]` 填本地 HCCL buffer 地址） |
| 填充 CommUbmemContext | `rankId`/`rankSize` + barrier buffer 取 `devCtx + ctxSize`（2MB 区域）+ `HcclCommMemReg` 注册 + `BuildChannels(UBMEM)` → `commBufferAddrs[peer]` |
| 拷贝到 device | `HcclEngineCtxCopy(comm, engine, ctxTag, hostCtx, ctxSize, 0)` 把 hostCtx 拷到 device GM |
| 返回 device 指针 | kernel 通过此指针访问 CommContext |

### 填充后的 CommContext 结构

```
CommContext (device GM)
├── CommUdmaContext udmaCtx
│   ├── rankId                                ← 本 rank ID
│   ├── rankSize                              ← 总 rank 数
│   ├── channelHandles[COMM_MAX_RANK_NUM]     ← 每 peer rank 的 URMA channel 句柄（self 不填充）
│   └── commBufferAddrs[COMM_MAX_RANK_NUM]    ← 每 rank 的 Win 区基地址
└── CommUbmemContext ubmemCtx
    ├── rankId
    ├── rankSize
    └── commBufferAddrs[COMM_MAX_RANK_NUM]    ← 每 rank 的 barrier flag GM 地址
```

> 字段顺序以 `apace/block/aiv_comm/collective_comm_context.h` 为准：`rankId`、`rankSize` 在前，数组在后。host 侧聚合初始化/布局推算必须按此顺序。

> 完整实现见官网 `apace/utils/comm_channel_builder.h` 的 `CommChannelBuilder::CreateDeviceContext()`。

---

## 7. 扩展通信原语指南

### 当前支持的通信原语

| 原语 | 模式 | 文件 | 状态 |
|:---|:---|:---|:---|
| AllToAll | GET | `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` | ✅ 钩子已实现（已注册分发；官网暂无 GET 算子使用方） |
| AllToAll | PUT | `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` | ✅ 已实现（all_to_all_quant_matmul 使用） |
| AllGather | PUT | `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` | ✅ 已实现（all_gather_quant_matmul 使用） |
| AllGather | GET | — | ❌ 未实现 |
| AllReduce | — | — | ❌ 未实现 |
| ReduceScatter | — | — | ❌ 未实现（`CommCollectiveOp::ReduceScatter` 枚举值已在 `apace/block/aiv_comm/collective_comm_api.h` 预留但无分发实现；可用 AllToAll PUT + AtomicAdd 替代，推导见 `fusion.md`） |

### 扩展边界

- **禁止修改现有文件**：`collective_comm_api.h`、`collective_comm_base.h`、`all_to_all_udma_get.h` 等已实现的文件不能改
- **允许新增文件**：可在 `apace/block/aiv_comm/` 下**新增**目录和文件（如 `apace/block/aiv_comm/all_reduce/all_reduce_udma.h`）
- 新增 block 文件属于"创建新通信原语"，超出常规开发范围

### 扩展决策树

```
需要新通信原语？
├── 是 AllToAll/AllGather 的变体（如 GET→PUT）
│   └── 参考现有实现，新增 block 文件
├── 是完全新原语（AllReduce/ReduceScatter/Broadcast）
│   ├── 评估是否超出常规开发范围
│   ├── 如继续：实现 4 个钩子 + 注册到分发器 + tiling 适配
│   └── 建议先在独立分支验证，再合入共享层
└── 只是使用方式不同（如换 dtype/shape）
    └── 不需要新增 block 文件，只改 kernel/<op>/ 下的文件
```

### 常见误区

| 误区 | 正确做法 |
|:---|:---|
| 修改 `all_to_all_udma_get.h` 适应新场景 | 新增 `all_to_all_udma_get_v2.h` 或在 kernel 层适配 |
| 在 kernel 中直接调用 `Hcomm::ReadNbi` | 使用 `CollectiveComm` 四段式 API，保持抽象一致性 |
| 跳过 `CollectiveCommHelper` 直接实例化实现类 | 通过 `CollectiveComm<Op, Mode, T, Barrier>` 编译期分发，保持类型安全 |

> ReduceScatter 替代实现（AllToAll PUT + AtomicAdd）的完整推导见 `fusion.md`。

---

## 常见陷阱

| # | 陷阱 | 后果 | 规避 |
|:---|:---|:---|:---|
| 1 | flagId 冲突（与保留区间冲突） | 同步紊乱 | 避开保留区间 [11,14]（SyncAll 硬同步内部占用）、Matmul 高阶 API 的 [0,2N-1] |
| 2 | UB 分配溢出 | 超出 UB 容量 | 单对象基线（COMM_WORKSPACE_SIZE + barrier UB_SIZE）+ kernel 专属 UB 需求，双对象翻倍；总量不能超过 UB 容量 |
| 3 | GET 模式 barrier 时序遗漏 | 读到未初始化数据 | `DoCommit` 中先 `CrossCore()+CrossDevice()` 再 `ReadNbi` |
| 4 | 自跳过规则遗漏 | 自身 Win 区读写错误 | DoCommit/DoWait 跳过 `targetRankId == rankId` |
| 5 | HCCL windows 模式误加 CommContext | 编译错误或内存浪费 | 使用 `GetHcclContext` 的 kernel 不需要 `CommContext` 结构 |
| 6 | TeamBarrier 远端 rank 未就绪 | CrossDevice **无限等待挂死**（无超时保护，不会 assert） | 确保所有 rank kernel 已 launch；`CreateDeviceContext` 后做跨 rank host barrier 再 launch |

---

## 后续阅读

- `fusion.md` — GET/PUT 编排模式、flag 编排、环形回压、localMatmul
- `operator-anatomy.md` — 算子完整骨架中的通信对象使用
- `host-and-testing.md` — host launcher 序列（建链调用时机）
- `ascendc-api-best-practices` skill `references/api-hcomm.md`、`references/api-crosscore-sync.md`、`references/api-hccl-host.md`
