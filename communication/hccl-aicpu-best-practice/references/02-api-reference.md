# 02 · API 与数据结构速查（AICPU template）

## 0. 按开发动作选择接口

自定义算法先按下表选动作，再读对应小节；不需要为了写 Ring 先研究整个 HCOMM 仓。
本表及 §9.4 于 2026-09-24 核对 HCCL `88a2a65c` 的 wrapper/channel/dlsym 与
HCOMM `ce7297c6` 的正式/实验头文件，来源指纹见 [source-baseline.json](source-baseline.json)。
头文件匹配不代表已安装库及目标设备支持全部接口，部署支持性仍由运行时检查和功能验证确定。

| 开发动作 | template 优先使用 | 底层 HCOMM / 契约入口 |
|---|---|---|
| 本地拷贝 | `LocalCopy` / `LocalCopySlices`（§4） | `HcommLocalCopyOnThread`；长度为字节 |
| 本地归约 | `LocalReduce`（§4） | `HcommLocalReduceOnThread`；count 为元素数；软归约另见 §4 |
| 向 peer 写 / 从 peer 读 | `SendWrite` / `RecvWrite`、`SendRecv*Write/Read` 及 Batch 变体 | `HcommWriteOnThread` / `HcommReadOnThread`；wrapper 含握手 |
| 跨 rank 归约 | §4 的 `*WriteReduce` / `*ReadReduce` | `HcommWriteReduceOnThread` / `HcommReadReduceOnThread`；归约目标须已初始化 |
| 主从线程同步 | `PreSyncInterThreads` / `PostSyncInterThreads` | `HcommThreadNotifyRecordOnThread` / `HcommThreadNotifyWaitOnThread` |
| peer 间通知 | 通常复用 Send/Recv wrapper 协议 | `HcommChannelNotifyRecordOnThread` / `HcommChannelNotifyWaitOnThread`；不能混用线程 notify |
| AICPU 读取设备结果前等待 | 软归约的已核对完成协议（§9、线程同步参考） | `HcommBatchModeEnd/Start`、实验接口 `HcommThreadJoin`；不是普通步骤的通用 barrier |
| 按指定 peer 申请通道 | `CreateChannelRequestByRankId`（§6） | 控制面准备，不放入 KernelRun；按实际协议/拓扑核对实现 |

原始 Read/Write 不等价于带握手的 wrapper；直接替换时必须重新设计通知与 buffer 生命周期。
HCCL wrapper 的 `LocalCopy/LocalReduce` 参数顺序是 **thread, src, dst**，底层 HCOMM 是 **thread, dst, src**。

源码位置：
- 基类　　`src/ops/op_common/algorithm/template/alg_v2_template_base.h` / `common_alg_template_base.h`
- 结构体　`src/ops/op_common/algorithm/template/template_utils.h`、`src/ops/op_common/inc/alg_param.h`
- 搬运原语`src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.h`
- channel 申请`src/ops/op_common/algorithm/executor/channel/channel.h`

## 1. 头文件该 include 什么

```cpp
#include "alg_v2_template_base.h"   // 基类 InsAlgTemplateBase，必须
#include "executor_base.h"          // 顺带带进 channel.h / topo.h / utils.h / log.h / sal.h
#include "alg_data_trans_wrapper.h" // SendRecv* / LocalCopy / LocalReduce / *SyncInterThreads
```

`.cc` 只需 `#include "自己的.h"`（必要时再补 `alg_data_trans_wrapper.h`）。
部分存量文件用 `executor_v2_base.h` 代替 `executor_base.h`，两者都行，跟邻居保持一致。

## 2. 基类可用的 protected 成员

`InsAlgTemplateBase`（构造函数自动填好，见 `alg_v2_template_base.cc`）：

| 成员 | 含义 |
|---|---|
| `u32 myRank_` | 通信域内的 userRank |
| `u32 templateRankSize_` | 本 template 视角的 rank 总数（多层时是各层乘积） |
| `std::vector<std::vector<u32>> subCommRanks_` | 分层 rank 列表；`subCommRanks_[0]` 是 level0 |
| `u32 root_` | scatter / broadcast 的 root |
| `OpMode opMode_` | `OPBASE` / `OFFLOAD` / `ACLGRAPH`；完整枚举以目标仓 `src/ops/op_common/inc/alg_param.h` 为准（`978d01b6` 核对） |
| `HcclReduceOp reduceOp_` | 归约类型 |
| `HcclDataType dataType_` | **构造时不填**，KernelRun 里自己从 `param.DataDes.dataType` 赋值 |
| `u32 threadNum_` | 同上，KernelRun 里自己赋值 |
| `std::vector<u32> notifyIdxMainToSub_` / `notifyIdxSubToMain_` | 给 `GetNotifyIdx*` 填 |
| `bool enableDetour_` / `enableRemoteMemAccess_` / `supportSymmetricMemory_` | 特性开关 |
| `u32 channelsPerRank_`（来自 `CommonAlgTemplateBase`） | 多 jetty 时每个远端 rank 的 channel 数 |

`CommonAlgTemplateBase` 还提供：
- `HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)` — 从 channels 反推并设置 `channelsPerRank_`
- `virtual HcclResult CalcDataSplitByPortGroup(totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset)` — 按端口组把一份数据切给多条 channel
- `static constexpr TemplateProp props = {};` — 覆写成 `{.algoType = AlgoType::NHR}` 之类；`TemplateProp` 只有 `algoType` 一个成员，枚举定义在 `src/common/alg_parse.h`

`InsAlgTemplateBase` 额外提供：
- `bool IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>>& channels)` — 判断是否 PCIe 链路（决定 write 还是 read 模式）

## 3. 核心数据结构

### DataSlice —— 一段待搬运的数据（`template_utils.h`）

```cpp
struct DataSlice {
    void* addr_;   // 基地址：本地 input/output/hcclBuff，或远端 channel.remoteCclMem.addr
    u64   offset_; // 相对 addr_ 的字节偏移
    u64   size_;   // 字节数
    u64   count_;  // 元素个数（归约类必须正确，纯拷贝可为 0）
    DataSlice(void* addr, u64 offset, u64 size, u64 count);
    DataSlice(void* addr, u64 offset, u64 size);          // count_ = 0
    std::string Describe() const;
};
```

### SliceInfo / RankSliceInfo —— 切片计划

```cpp
struct SliceInfo { u64 offset{0}; u64 size{0}; };
using RankSliceInfo = std::vector<std::vector<SliceInfo>>;  // [rankIdx][sliceIdx]
```

### BuffInfo —— 三块 buffer 的地址与偏移

```cpp
struct BuffInfo {
    void*   inputPtr;        // 当前阶段输入，不保证是 userIn
    void*   outputPtr;       // 当前阶段输出，不保证是 userOut
    HcclMem hcclBuff;        // 跨 rank 可见的 CCL buffer（scratch），{type, addr, size}
    BufferType inBuffType, outBuffType, hcclBuffType;   // INPUT / OUTPUT / HCCL_BUFFER
    u64 inputSize, outputSize, hcclBuffSize;
    u64 inBuffBaseOff, outBuffBaseOff, hcclBuffBaseOff; // 本轮 loop 的基偏移，由 executor 填
};
```

### TemplateDataParams —— executor 每轮 loop 传进来的数据描述

常用字段：

| 字段 | 含义 |
|---|---|
| `buffInfo` | 见上 |
| `count` | executor 为本轮填写的元素个数；是完整向量还是单 rank 片取决于契约 |
| `sliceSize` | 本轮切片字节数；不恒等于 count × dtSize（如 AllReduce 双模板 executor），见算子契约卡 |
| `tailSize` | 尾块字节数 |
| `inputSliceStride` / `outputSliceStride` | 相邻 rank 数据块的跨度（gather/scatter 类用） |
| `repeatNum` / `inputRepeatStride` / `outputRepeatStride` | 本次调用的逻辑组数与输入/输出 repeat 字节步长，按实际 executor 分支填写；见 `09-template-data-params.md` |
| `enableRemoteMemAccess` | 能否直接访问对端 input/output（图模式 OFFLOAD 下为 true） |
| `supportSymmetricMemory` | 对称内存开关 |
| `root` | broadcast/scatter/reduce 的 root |
| `dataType` | 数据类型（也可从 `param.DataDes.dataType` 取） |
| `allRankSliceSize` / `allRankDispls` | 变长算子（*_v）每个 rank 的大小与位移 |
| `sendCounts/recvCounts/sdispls/rdispls` | AlltoAllV 变长参数 |
| `stepSliceInfo` | OmniPipe 多 step 切片 |

### TemplateResource —— executor 给的资源

```cpp
struct TemplateResource {
    std::map<u32, std::vector<ChannelInfo>> channels; // key = 远端 userRank，value = 到它的多条 channel
    std::vector<ThreadHandle> threads;                // threads[0] 是主流，其余是从流
    std::vector<CcuKernelHandle> ccuKernels;          // CCU 专用
    std::vector<CcuKernelSubmitInfo> submitInfos;     // CCU 专用
    void* npu2DpuShmemPtr; void* dpu2NpuShmemPtr;     // DPU 专用
    void* aivCommInfoPtr;                             // AIV 专用
    double dieSplitRatio;
};
```

### ChannelInfo —— 一条链路（`alg_param.h`）

```cpp
struct ChannelInfo {
    bool isValid;
    u32  remoteRank;
    CommProtocol protocol;      // COMM_PROTOCOL_PCIE / _UB_CTP / ...
    EndpointLocType locationType;
    u32  notifyNum, portGroupSize, dieId;
    ChannelHandle handle;
    HcclMem remoteCclMem;          // ★ 对端 CCL buffer 地址，写模式的落点
    HcclMem remoteInputGraphMode;  // 图模式下对端 sendBuf
    HcclMem remoteOutputGraphMode; // 图模式下对端 recvBuf
    HcclMem remoteInput, remoteOutput; // A3 的 cclIn / cclOut
};
```

### AlgResourceRequest —— CalcRes 的输出

```cpp
struct AlgResourceRequest {
    u32 notifyNumOnMainThread;                          // 主流上的 notify 数
    u32 slaveThreadNum;                                 // 从流数量（= 总 thread 数 - 1）
    std::vector<u32> notifyNumPerThread;                // 每个从流的 notify 数，长度 = slaveThreadNum
    std::vector<std::vector<HcclChannelDesc>> channels; // 每层一组
    double dieSplitRatio;
    ParallelChannelPortInfo parallelPortInfo;
    std::vector<CcuKernelInfo> ccuKernelInfos;
    std::vector<u32> ccuKernelNum;
};
```

## 4. 搬运原语全表（`alg_data_trans_wrapper.h`）

以下 Send/Recv 搬运函数返回 `HcclResult`，最后一个参数是 `const ThreadHandle& thread`；
本地操作和同步的参数顺序见各自签名，不能按这个顺序套用。

### 写模式（Write / DMA push，默认路径）

| 函数 | 入参类型 | 说明 |
|---|---|---|
| `SendWrite` | `DataInfo` | 单向发，一组 slice |
| `SendBatchWrite` | `DataInfo` | 单向发，批量（多 slice 合并下发） |
| `RecvWrite` | `DataInfo` | 单向收 |
| `SendRecvWrite` | `SendRecvInfo` | 双向 |
| **`SendRecvBatchWrite`** | `SendRecvInfo` | **双向批量，Mesh/NHR 最常用** |
| `SendWriteReduce` / `SendBatchWriteReduce` / `RecvWriteReduce` | `DataReduceInfo` | 带归约 |
| `SendRecvWriteReduce` | `SendRecvReduceInfo` | 双向带归约 |
| **`SendRecvBatchWriteReduce`** | `SendRecvReduceInfo` | **双向批量带归约，ReduceScatter 步用** |

### 读模式（Read / DMA pull，PCIe 链路走这条）

`SendRead` · `RecvRead` · `RecvBatchRead` · `SendRecvRead` · `SendRecvBatchRead`
`SendReadReduce` · `RecvReadReduce` · `RecvBatchReadReduce` · `SendRecvReadReduce` · `SendRecvBatchReadReduce`

选择方式：
```cpp
isDmaRead_ = IsPcieProtocol(templateResource.channels);
if (isDmaRead_) { CHK_RET(SendRecvReadReduce(info, thread)); }
else            { CHK_RET(SendRecvBatchWriteReduce(info, thread)); }
```

### 本地操作

```cpp
HcclResult LocalCopy(const ThreadHandle&, const DataSlice& src, const DataSlice& dst);
HcclResult LocalReduce(const ThreadHandle&, const DataSlice& src, const DataSlice& dst,
                       HcclDataType, HcclReduceOp);          // dst = reduce(dst, src)
HcclResult LocalCopySlices(const ThreadHandle&, const std::vector<DataSlice>& src,
                                                const std::vector<DataSlice>& dst);
bool       IsContinuousSlice(const DataSlice& nxt, const DataSlice& curr);
```

### 主从流同步

```cpp
HcclResult PreSyncInterThreads (const ThreadHandle& mainThread,
                                const std::vector<ThreadHandle>& subThreads,
                                const std::vector<u32>& notifyIdxMainToSub);
HcclResult PostSyncInterThreads(const ThreadHandle& mainThread,
                                const std::vector<ThreadHandle>& subThreads,
                                const std::vector<u32>& notifyIdxSubToMain);
```

### AICPU 软归约（64-bit / PROD 兜底）

```cpp
HcclResult AicpuReduce(const ThreadHandle&, const DataSlice& src, const DataSlice& dst,
                       HcclDataType, HcclReduceOp);
HcclResult AicpuReduceFp16(u8* dst, u8* src, u64 size, HcclReduceOp);
template <typename T> HcclResult AicpuReduceTemplate(T* dst, u64 dstSize, T* src, u64 srcSize, HcclReduceOp);
```

## 5. 组装入参的三层结构

`SendRecvInfo` / `SendRecvReduceInfo` 是套娃出来的，聚合初始化写法：

```cpp
//              { 发channel, 收channel }   { { 发src[], 发dst[] }, { 收src[], 收dst[] } }
SendRecvInfo info{ {linkSend, linkRecv},   {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                   dataType_ };

SendRecvReduceInfo rinfo{ {sendChannel, recvChannel},
                          {{sendSrc, sendDst}, {recvSrc, recvDst}},
                          dataType_, reduceOp_ };

// 单向的：
DataInfo       dinfo{ channel, SlicesList{srcSlices, dstSlices}, dataType_ };
DataReduceInfo rdinfo{ channel, SlicesList{srcSlices, dstSlices}, dataType_, reduceOp_ };
```

底层定义：
```cpp
struct SlicesList     { std::vector<DataSlice> srcSlices_, dstSlices_; };
struct TxRxChannels   { ChannelInfo txChannel_, rxChannel_; };
struct TxRxSlicesList { SlicesList txSlicesList_, rxSlicesList_; };
```

## 6. channel 申请函数（`executor/channel/channel.h`）

自定义 peer 图可定向核对以下已存在的接口，不要求 Mesh/NHR：

```cpp
HcclResult CreateChannelRequestByRankId(
    HcclComm comm, const OpParam& param, u32 myRank, u32 remoteRank,
    std::vector<HcclChannelDesc>& channels, u32 channelRepeatNum = 1);
```

`myRank/remoteRank` 是实际 rank 身份；先完成逻辑索引映射。接口存在不代表适用于所有网络层/协议，
实际调用前按选定场景核对 `channel.cc` 的实现。它不自动完成 Ring 的 peer 去重、阶段和通知设计。

在 `CalcRes` 里调，产出 `std::vector<HcclChannelDesc>`：

```cpp
HcclResult CalcChannelRequestMesh1D(HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*,
                                    const std::vector<std::vector<u32>>& subcommInfo,
                                    std::vector<HcclChannelDesc>& channels);
```
同签名的还有：
`CalcChannelRequestMesh1DFullMesh` · `CalcChannelRequestMesh1DLevel0` · `CalcChannelRequestMesh1DLevel1`
`CalcChannelRequestMesh1DHighestHostRoce` · `CalcChannelRequestNhr` · `CalcChannelRequestNhrHighestHostRoce`
`CalcChannelRequestMesh2D`

带额外参数的：
```cpp
CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subcommInfo, channels, bool isIsolation = false);
CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subcommInfo, channels,
                                     bool isIsolation = false, bool execptMesh = true);
CreateChannelRequestByRankId(comm, param, myRank, remoteRank, channels, u32 channelRepeatNum = 1);
CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subcommInfo, channels, CommTopo priorityTopo);
CalcChannelRequestNHRWithPriorityTopo(...);
```

## 7. NHR 辅助（`template_utils.h`）

```cpp
u32        GetNHRStepNum(u32 rankSize);                              // NHR 的 step 数
HcclResult GetAlgRank(u32 virtRank, const std::vector<u32>& rankIds, u32& algRank); // userRank → 算法内序号
u32        CalcChannelsPerRank(const std::vector<HcclChannelDesc>& channels);
u32        CalcChannelsPerRankMin(const std::vector<HcclChannelDesc>& channels);

struct AicpuNHRStepInfo {
    u32 step, myRank, nSlices, toRank, fromRank;
    std::vector<u32> txSliceIdxs, rxSliceIdxs;
};
```

## 8. 常量与宏

```cpp
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;        // alg_template_base.h
constexpr u32 CUSTOM_TIMEOUT       = 1836;       // alg_param.h，HcommThreadJoin 超时秒数
constexpr u32 MAX_JETTY_NUM        = 4;          // template_utils.h
constexpr u32 SMALL_SIZE_512KB     = 512 * 1024;
constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED];  // alg_param.h，dataType → 字节数
struct HcclMem { HcclMemType type; void* addr; uint64_t size; };  // src/common/hccl_common.h
```

错误处理宏（`-Werror` 环境，一律用宏不要裸判断）：

```cpp
CHK_RET(expr);                                   // 非 HCCL_SUCCESS 直接 return
CHK_PTR_NULL(ptr);
CHK_PRT_RET(cond, HCCL_ERROR("..."), retVal);    // cond 为真则打日志并 return retVal
CHK_PRT(expr);
CHK_SAFETY_FUNC_RET(strcpy_s(...));
```

日志：`HCCL_DEBUG` / `HCCL_INFO` / `HCCL_WARNING` / `HCCL_ERROR`，
习惯前缀 `"[类名][方法名] ..."`，格式串用 `%u/%llu/%d/%s`（`u64` 用 `%llu`）。

## 9. wrapper 之下的 HCOMM 接口层（直调与 dlsym 机制）

wrapper 覆盖不到时才直调 HCOMM 原语；新增 HCOMM 接口调用必须经 dlsym 层。
事实来源：hcomm 仓 `include/hcomm_primitives.h`（正式 API）、`pkg_inc/hcomm/hcomm_primitives_expt.h`
（实验 API）、HCCL 仓 `src/common/hcomm_dlsym/hcomm_primitives_dl.h`。

### 9.1 template 可能直调的原语

| 原语 | 场景 |
|---|---|
| `HcommBatchModeEnd(param.algTag)` + `HcommBatchModeStart(param.algTag)` | KernelRun 多阶段之间刷新批量下发上下文（Start/End 须同线程）；64-bit/PROD 的 ThreadJoin 前也用（见 `08` §5.3） |
| `HcommThreadJoin(thread, timeout)` | `needAicpuReduce_`（INT64/UINT64/FP64/PROD）时 PostLocalReduce 前等待全部 thread 写完。实验 API（expt 头 + dlsym 声明） |
| `HcommAclrtNotifyRecord/WaitOnThread` | 主 stream 通知（kernel_launch 层为主） |
| `HcommThreadSynchronize` / `HcommSendRequest` / `HcommWaitResponse` | 仅 DPU/host_nic 系列 template，常规 AICPU 不用 |

### 9.2 dlsym 适配层（HCCL ↔ HCOMM 解耦的关键）

HCCL 与 HCOMM 两仓独立编译，跨仓接口全部经 `src/common/hcomm_dlsym/`：

```cpp
DECL_WEAK_FUNC(HcclResult, HcommThreadJoin, ThreadHandle thread, uint32_t timeout);  // weak 符号
DECL_SUPPORT_FLAG(HcommBatchTransferOnThread); // 已存在的可选能力查询示例
```

规则：
1. **按真实适配层核对支持性**：有 `DECL_SUPPORT_FLAG` 的可选接口使用其真实查询函数，false 时走替代路径。
   不是每个 weak 接口都有同名查询函数；当前头文件没有 `DECL_SUPPORT_FLAG(HcommThreadJoin)`，不能凭名字造函数。
   没有独立查询的接口须核对加载/错误路径及目标运行库支持，不能把 weak 声明当作支持证明。
2. **Default 超时封装**：优先 `HcommChannelNotifyWaitOnThreadWithDefaultTimeout` 等 Default 变体，
   HCCL 侧封装为 `HcclChannelNotifyWaitOnThreadDefault(thread, ch, idx, fallbackTimeout)`
3. **私有 ABI 兼容类型**：`HcclHcommTransferType` / `HcclHcommBatchTransferDesc` 是 HCCL 侧副本
   （比 hcomm 原版多 `NOTIFY_WAIT_WITH_DEFAULT_TIMEOUT`），旧 HCOMM 头未声明新类型时也能编译

### 9.3 wrapper 内置的 notify 协议与 batch 降级

来源为 §0 的 wrapper 指纹；具体实现入口 `DoSendRecvBatchTx/Rx`。
非对称收发可以让 tx/rx channel 指向不同 peer，无须先阅读整个 NHR 实现。

- **双向写 notify 时序**（`NOTIFY_IDX_ACK=0` / `NOTIFY_IDX_DATA_SIGNAL=1`，alg_param.h）：
  `record ACK(recv通道) → wait ACK(send通道) → Write … → record DATA_SIGNAL(send通道) → wait DATA_SIGNAL(recv通道)`
- **双向读 notify 时序**：
  `record ACK(send通道) → wait ACK(recv通道) → Read … → record DATA_SIGNAL(recv通道) → wait DATA_SIGNAL(send通道)`
- **入参最小形态**（示意，不预设算法的切片/线程）：写模式用
  `SendRecvInfo{{txChannel, rxChannel}, {{txSrc, txDst}, {{}, {}}}, dtype}`；
  读模式用 `SendRecvInfo{{txChannel, rxChannel}, {{{}, {}}, {rxSrc, rxDst}}, dtype}`。
  写只消费 tx slices，读只消费 rx slices，但两条 channel 都参与握手。邻居重合时复用同 peer 的 channel，
  并核对两端通知序列；多 jetty 每条 channel 上分别匹配数据和通知，不能从端口数相同推断分组相同。
- **batch 降级**：批量函数先查 `IsHcommBatchTransferOnThreadSupported()`；支持则组装
  `HcclHcommBatchTransferDesc[]` 一次 `HcommBatchTransferOnThread` 下发（notify 融合进最后一条
  WRITE_REDUCE desc），不支持逐条回退非 batch 版
- **HCOMM 原语签名要点**：`HcommLocalCopyOnThread` 的 len 是**字节**；`HcommLocalReduceOnThread`
  的 count 是**元素个数**；`HcommWriteOnThread` 的 dst 是对端地址（`ChannelInfo.remoteCclMem.addr`）；
  notify wait 的 timeOut 单位是秒。标 `experimental API` 的无兼容性保证，须按 §9.2 核对 dlsym 与支持条件

上述调用完成协议不自动证明下一步可覆盖哪一块 scratch；算法仍须论证各 rank 的步次匹配和 buffer 生命周期。

### 9.4 必要 HCOMM 签名（按动作取用）

以下来自上述 HCOMM 版本的 `include/hcomm_primitives.h`；返回值均为 `int32_t`，
在 HCCL 中通过适配层调用并处理错误。`len` 为字节，`count` 为元素数，`timeout` 为秒。
Read 的 src 为远端、dst 为本端；Write 的 src 为本端、dst 为远端。

```cpp
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len);
int32_t HcommLocalReduceOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t count,
                               HcommDataType dataType, HcommReduceOp reduceOp);
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel,
                          void* dst, const void* src, uint64_t len);
int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel,
                         void* dst, const void* src, uint64_t len);
int32_t HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
int32_t HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                               uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx);
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout);
int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx);
int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel,
                                      uint32_t localNotifyIdx, uint32_t timeout);
int32_t HcommBatchModeStart(const char* batchTag);
int32_t HcommBatchModeEnd(const char* batchTag);
```

批量 Start/End 及其中下发任务须在同一调用线程。普通接口返回成功不能直接证明跨线程数据可供 CPU 读取。
`HcommThreadJoin(ThreadHandle thread, uint32_t timeout)` 的 HCOMM 声明返回 `int32_t`，
位于 `pkg_inc/hcomm/hcomm_primitives_expt.h`；HCCL weak 声明的返回类型为 `HcclResult`，按适配层消费。
Batch/带通知融合等可选接口由 wrapper 的支持分支处理；只有自行扩展这些能力时才深入对应实现。
