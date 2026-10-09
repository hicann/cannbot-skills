# 00 · 领域速成：评审一个 AICPU template 之前要知道的

**这一篇让本 skill 自成体系** —— 读完它就能开始评审，不需要先去读别的 skill 或别的仓的文档。
所有断言都能在 `$HCCL` 里查证，文中给了文件路径。

---

## 1. template 是什么

HCCL 里的 **template ≠ C++ 模板**。它是分层架构最底层的**执行编排单元**：
一个具体通信算法在某个执行引擎上的实现，负责"这一轮数据，谁往谁发、走哪条 channel、
跑在哪个 thread、什么时候同步"。

```
hccl.h API → <op>_op.cc → Selector(选算法) → ExecutorRegistry → Executor → ★Template★
                                                                   ↑            ↑
                                                        "分几次搬、每次多大"  "这一次怎么搬"
```

本 skill 的评审对象是 **AICPU 引擎**的 template：

```
$HCCL/src/ops/{算子}/algorithm/template/aicpu/ins_temp_*.h / *.cc      基类 InsAlgTemplateBase
```

CCU（`template/ccu/ccu_temp_*`）、AIV（`template/aiv/aiv_temp_*`）、
以及 `src/ops/scatter/algo/template/` 下那批继承 `AlgTemplateBase` 的 **V1 遗留**都不在范围内。

template **通常不写注册宏**：executor 用 `REGISTER_EXEC_V2(cmdType, algName, ExecutorTmpl, TopoMatch, 本类名)`
把它当 C++ 模板参数吃进去。例外是 **DPU 侧展开**的，要在 `.cc` 末尾写
`REGISTER_TEMPLATE_V2("类名", 类名);`（`dpu/kernel_launch.cc` 按字符串反查）。

## 2. 接口契约（`src/ops/op_common/template/alg_v2_template_base.h`）

**纯虚，不实现编译不过**

- `std::string Describe() const`
- `void GetNotifyIdxMainToSub(std::vector<u32>&)`
- `void GetNotifyIdxSubToMain(std::vector<u32>&)`

**实质必须**（基类默认实现直接 `HCCL_ERROR` 返回失败）

- `HcclResult CalcRes(HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, AlgResourceRequest&)`
- `HcclResult KernelRun(const OpParam&, const TemplateDataParams&, TemplateResource&)`
- `u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)`

**按需**

- `GetRes(AlgResourceRequest&) const` + `u64 GetThreadNum() const` —— 线程数由 `channelsPerRank_` 推导时必须写
- `static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam)` —— **不是 virtual**，靠模板参数静态分发
- `static constexpr TemplateProp props = {.algoType = AlgoType::X};` —— `TemplateProp` **只有 `algoType` 一个成员**
  （历史上的 `isNhr` 已删除，仍这么写编不过）；枚举值在 `src/common/alg_parse.h`

构造签名被 executor 硬编码：`(const OpParam&, const u32 rankId /* 通信域 userRank */,
const std::vector<std::vector<u32>>&)`；走 FastLaunch 的 executor 还需要 `T() = default`。

基类给的成员：`myRank_`（userRank）、`templateRankSize_`、`subCommRanks_`、`opMode_`、`root_`、`buffInfo_`。

## 3. 数据怎么描述（`src/ops/op_common/template/template_utils.h`）

`KernelRun` 拿到两个入参结构：

```cpp
struct BuffInfo {                    // tempAlgParams.buffInfo
    void* inputPtr;  void* outputPtr;    // userIn / userOut
    HcclMem hcclBuff;                    // 跨 rank 缓存（scratch）
    BufferType inBuffType, outBuffType, hcclBuffType;
    u64 inBuffBaseOff, outBuffBaseOff, hcclBuffBaseOff;   // ★ 一切地址都要加上它
};
struct TemplateDataParams {
    BuffInfo buffInfo;
    u64 count, sliceSize, tailSize;                       // 本轮 loop 的元素数 / 字节数 / 尾块
    u64 inputSliceStride,  outputSliceStride;             // 同一 repeat 内相邻 alg rank 的间距
    u64 repeatNum, inputRepeatStride, outputRepeatStride; // 两级地址的外层
    bool enableRemoteMemAccess, supportSymmetricMemory;   // 图模式 / 对称内存
    StepSliceInfo stepSliceInfo;                          // OmniPipe
};
struct TemplateResource {
    std::map<u32, std::vector<ChannelInfo>> channels;     // ★ key 是通信域 userRank
    std::vector<ThreadHandle> threads;                    // threads[0] 是主流
};
```

地址句柄一共五种，评审时按这张表对：

| 指向 | 怎么写 |
|---|---|
| 本 rank userIn | `DataSlice(buffInfo.inputPtr, buffInfo.inBuffBaseOff + off, len, cnt)` |
| 本 rank userOut | `DataSlice(buffInfo.outputPtr, buffInfo.outBuffBaseOff + off, len, cnt)` |
| 本 rank scratch | `DataSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + off, len, cnt)` |
| **对端** scratch | `channels.at(p)[ch].remoteCclMem.addr` + 同样的 offset |
| 图模式下对端 user buffer | `channels.at(p)[ch].remoteOutputGraphMode` / `remoteInputGraphMode` |

`len` 是字节数，带归约的原语底层用元素数，所以 `cnt = len / DATATYPE_SIZE_TABLE[dataType]`。

## 4. 搬运原语

| 意图 | 用 |
|---|---|
| 双向交换一批 slice（**写模式**，默认） | `SendRecvBatchWrite(SendRecvInfo, thread)` |
| 交换并在对端做归约 | `SendRecvBatchWriteReduce(SendRecvReduceInfo, thread)` |
| PCIe 链路（`IsPcieProtocol(channels)` 为真）改**读模式** | `SendRecvBatchRead` / `SendRecvReadReduce` |
| 本地拷贝 / 本地归约 | `LocalCopy` / `LocalReduce(thread, src, dst, dataType, reduceOp)` |
| 主从流同步 | `PreSyncInterThreads` / `PostSyncInterThreads` |

**跨 rank 的边只有一个方向**：写模式底层只用 `txSlicesList_`，读模式只用 `rxSlicesList_`；
另一侧只贡献 channel 做 notify 握手。两边都填 = 填了不生效的参数。

以上都返回状态，必须 `CHK_RET` / `CHK_PRT_RET` 接住。

## 5. thread 与 notify

约定：**`threads[0]` 是主流**，`threads[1..n-1]` 是从流；本地 `LocalCopy`/`LocalReduce` 挂主流，
跨 rank 的收发挂从流（Mesh 下"第 i 个从流对第 i 个远端"）。

```
主流: ─ LocalCopy ─┬─ record ────────────────────┬─ wait ─ LocalReduce ─
                   │  PreSyncInterThreads        │  PostSyncInterThreads
从流i:             └─ wait ─ SendRecvBatch* ─ record ┘
```

`CalcRes`（或 `GetRes`）里这三个字段一起填：

```cpp
resourceRequest.slaveThreadNum        = threadNum - 1;
resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
resourceRequest.notifyNumOnMainThread = threadNum - 1;
```

## 6. scratch 倍数决定 loop 次数

```
scratchBoundDataSize = hcclBuff.size / CalcScratchMultiple()  向下对齐 HCCL_MIN_SLICE_ALIGN(128)
maxDataSizePerLoop   = min(transportBoundDataSize, scratchBoundDataSize)
loopTimes            = ceil(dataCount / (maxDataSizePerLoop / dataTypeSize))
```

**报小 → hcclBuff 越界；报大 → 白切 loop。** 且**永远按单次 repeat 报**，
放大是 executor 的活（`std::max(interMultiplier, intraMultiplier * rankSizeLevel1_)`）。

## 7. 两级地址：repeat 与 stride

```
外层 repeat  rpt ∈ 0..RPT-1   一个 rank 有 RPT 块互不相邻的数据，块间距 inputRepeatStride / outputRepeatStride
内层 alg rank r ∈ 0..N-1      一块之内第 r 个 rank 的数据在 inputSliceStride*r / outputSliceStride*r
```

这六个值**全由 executor 每轮填、template 只读**，单层算法恒为 `1 / 0 / 0 / S / S`；
被两三层 executor 当 intra/inter 构件复用时才不是。

- **user buffer 的跨度用入参**；
- **scratch 的跨度自己算 `sliceSize * templateRankSize_`**，
  除非 `buffInfo.inBuffType == BufferType::HCCL_BUFFER`（输入本来就在 CCL buffer 里），
  这时排布是上一层 executor 定的，必须改用 `inputRepeatStride` / `inputSliceStride`。

## 8. 两处 CMake 登记

AICPU template 同时被编进 **host 侧 `libhccl.so`** 和 **device 侧 `libscatter_aicpu_kernel.so`**：

1. `src/ops/{算子}/algorithm/template/aicpu/CMakeLists.txt` 的 `set(src_list ...)`；
2. `src/scatter_aicpu_kernel.cmake` 的 `add_library(scatter_aicpu_kernel SHARED ...)` 块。

**只加第 1 处：host 编得过，运行时 AICPU 侧找不到符号。** 本仓最高频的接线 bug。

## 9. 验证手段

```bash
bash build.sh --aicpu             # 只编 device 侧 kernel，最快验第 2 处 CMake
bash build.sh --pkg               # host 侧；-Werror，警告即失败
bash build.sh --st_ops=<算子>     # 单算子 ST；全量 --st
```

ST（`test/st/algorithm/`）是此独立 skill 原有的验证参考。接入 HCCLBot 后，功能验收以调用工作流为准，由 Verifier 执行 hccl-vm 等验证；静态评审不替代运行验证。
scratch 倍数、slice 偏移、notify 索引错位这三类错误编译期一律抓不到——
这正是本 skill 存在的理由。
