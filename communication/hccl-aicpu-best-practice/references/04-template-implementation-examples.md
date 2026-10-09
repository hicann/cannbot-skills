# 04 · AICPU Template 实现示例

本文仅收录四种已有 template 的 C++ 写法，供实现或检视时按需对照；算法阶段、片归属及资源决策由已确认的 Dataflow Spec 给出，不能由示例代码反推。新算法无需先匹配下列实现。实际接口和字段以目标版本源码为准。

---

## 1. Mesh 1D · One-Shot

**示例布局**：该实现把整份数据推给对端 scratch，再在本地归约；下列 scratch 倍数和线程数只描述该蓝本，不能外推到其他算法或拓扑，也不证明性能优劣。
**蓝本**：`src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc`

```
CalcScratchMultiple() → templateRankSize_
thread 数            → templateRankSize_
```

```cpp
// ① 切片计划：scratch 被划成 rankSize 段，第 r 段留给 rank r 的数据
HcclResult XXX::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec) const
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(templateRankSize_, tmp);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        sliceInfoVec[rankIdx][0] = SliceInfo{accumOff, dataSize};
        accumOff += dataSize;
    }
    CHK_PRT_RET(sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size
                    != dataSize * templateRankSize_,
                HCCL_ERROR("[XXX] Rank [%d], SliceInfo calculation error!", myRank_), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

// ② 主体：主流本地拷 in→out，从流 i 与第 i 个远端交换整份数据
HcclResult XXX::RunAllReduce(const OpParam& param,
                             const std::map<u32, std::vector<ChannelInfo>>& channels,
                             const std::vector<ThreadHandle>& threads,
                             const TemplateDataParams& tempAlgParams, const RankSliceInfo& sliceInfoVec)
{
    DataSlice usrIn (tempAlgParams.buffInfo.inputPtr,  tempAlgParams.buffInfo.inBuffBaseOff,  processSize_, count_);
    DataSlice usrOut(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], usrIn, usrOut)));   // 主流
    if (subCommRanks_[0].size() == 1) { return HCCL_SUCCESS; }               // 单 rank 早退

    for (u32 queIdx = 1; queIdx < threadNum_; queIdx++) {                    // 从流
        u32 nextRank = (myRank_ + queIdx) % templateRankSize_;               // 错峰，避免全员打同一个目标
        u32 peer = subCommRanks_[0][nextRank];
        const ChannelInfo& linkSend = channels.at(peer)[0];
        const ChannelInfo& linkRecv = channels.at(peer)[0];

        // 我的数据 → 对端 scratch 的第 myRank_ 段
        std::vector<DataSlice> txSrc{usrIn};
        std::vector<DataSlice> txDst{DataSlice(linkSend.remoteCclMem.addr,
            sliceInfoVec[myRank_][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff,
            sliceInfoVec[myRank_][0].size, count_)};
        // 对端数据 → 我 scratch 的第 peerIdx 段
        std::vector<DataSlice> rxSrc{usrIn};
        std::vector<DataSlice> rxDst{DataSlice(linkRecv.remoteCclMem.addr,
            sliceInfoVec[nextRank][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff,
            sliceInfoVec[nextRank][0].size, count_)};

        SendRecvInfo info{{linkSend, linkRecv}, {{txSrc, txDst}, {rxSrc, rxDst}}, dataType_};
        CHK_PRT_RET(SendRecvBatchWrite(info, threads[queIdx]),
                    HCCL_ERROR("[XXX] SendRecv failed"), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

// ③ 收尾：主流把 scratch 里其余 rankSize-1 段归约进 output
HcclResult XXX::PostLocalReduce(...)
{
    if (needAicpuReduce_) { /* 见 §5 的 64-bit 兜底 */ }
    DataSlice usrOut(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);
    for (u32 r = 0; r < subCommRanks_[0].size(); r++) {
        if (r == myRank_) { continue; }
        DataSlice src(tempAlgParams.buffInfo.hcclBuff.addr,
                      sliceInfoVec[r][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff,
                      sliceInfoVec[r][0].size, count_);
        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], src, usrOut, dataType_, reduceOp_)));
    }
    return HCCL_SUCCESS;
}
```

---

## 2. Mesh 1D · Two-Shot（ReduceScatter + AllGather）

本节只展示该蓝本的两阶段调用和切片写法；阶段语义与算法选择见设计 Skill 的[通信算法设计](../../hccl-aicpu-design/references/08-algorithm-design.md)。
**蓝本**：`src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc`

```
CalcScratchMultiple() → 2   （理论 1 份够；非均衡切分时不足，留余量）
thread 数            → templateRankSize_
```

主流程：

```cpp
HcclResult XXX::KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res)
{
    threadNum_ = res.threads.size();
    CHK_PRT_RET(threadNum_ != templateRankSize_, HCCL_ERROR("..."), HCCL_E_INTERNAL);
    CHK_PRT_RET(subCommRanks_.empty(), HCCL_ERROR("..."), HCCL_E_INTERNAL);
    rankList_ = subCommRanks_.at(0);
    CHK_RET(GetAlgRank(myRank_, rankList_, myRankIdx_));   // userRank → 算法内序号

    processSize_ = p.sliceSize; count_ = p.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    needAicpuReduce_ = /* §5 */;
    if (count_ == 0) { return HCCL_SUCCESS; }

    CHK_RET(SplitData());                                  // 切成 rankSize 片，尾片可短
    CHK_RET(RunReduceScatter(param, p, res.channels, res.threads));
    CHK_RET(RunAllGather(p, res.channels, res.threads));
    return HCCL_SUCCESS;
}

HcclResult XXX::RunReduceScatter(...)
{
    PreSync(threads);                       // 主→从
    CHK_RET(ScatterData(p, channels, threads));   // 从流 i 把第 i 片发给 rank i 的 scratch
    PostSync(threads);                      // 从→主
    if (needAicpuReduce_) { /* §5 */ }
    CHK_RET(ReduceData(p, threads));        // 主流把收到的 rankSize-1 份归约到第 0 片位置
    return HCCL_SUCCESS;
}
```

切片（向上取整 + 尾片截断，`RoundUp(count, sliceNum)`）：

```cpp
u64 sliceCount = RoundUp(count_, sliceNum);          // 每片元素数（向上取整）
u64 offsetCount = 0, offsetSize = 0;
for (u32 i = 0; i < sliceNum; ++i) {
    u64 cur = std::min(sliceCount, count_ - offsetCount);
    sliceInfoList_.emplace_back(offsetSize, cur * dataTypeSize_, cur);
    offsetCount += cur;
    offsetSize = offsetCount * dataTypeSize_;
}
```

`PreSync` / `PostSync` 抽成私有方法，两个阶段各调一次：

```cpp
HcclResult XXX::PreSync(const std::vector<ThreadHandle>& threads) {
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slave(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads.at(0), slave, notifyIdxMainToSub_));
    }
    return HCCL_SUCCESS;
}
```

---

## 3. NHR（多 step，对数级）

本节只展示该 NHR 蓝本的分步循环和收发接口；支持的 rank 数、每步 peer 与片归属应按设计 Skill 的算法检查、目标源码和本次 Spec 推导，不从示例代码推断性能。
**蓝本**：`src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_nhr.cc`

```
CalcScratchMultiple() → 1
thread 数            → channelsPerRank_（多 jetty），必须覆写 GetRes() + GetThreadNum()
头文件类体开头       → props = {.algoType = AlgoType::NHR}（实现了 CalcCostCoeff 的普遍会加；非必须）
```

```cpp
HcclResult XXX::KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res)
{
    rankList_ = subCommRanks_.at(0);
    CHK_RET(GetAlgRank(myRank_, rankList_, myRankIdx_));
    processSize_ = p.sliceSize; count_ = p.count;
    dataType_ = param.DataDes.dataType; dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    isDmaRead_ = IsPcieProtocol(res.channels);          // PCIe → 读模式
    if (count_ == 0) { return HCCL_SUCCESS; }

    sliceSize_ = (count_ / templateRankSize_) * dataTypeSize_;             // 向下取整
    tailSize_  = count_ * dataTypeSize_ - sliceSize_ * (templateRankSize_ - 1);

    CHK_RET(PrepareDataSplitForMultiChannel(res));      // 一份数据再按 channel 数二次切分
    threadNum_ = GetThreadNum();
    CHK_PRT_RET(threadNum_ > res.threads.size(), HCCL_ERROR("..."), HCCL_E_INTERNAL);

    CHK_RET(PreCopy(p, res.threads));                  // input → hcclBuff（NHR 原地跑在 scratch 上）
    /* PreSyncInterThreads */
    for (u32 ch = 0; ch < channelsPerRank_; ch++) {
        CHK_RET(RunReduceScatter(p, res.channels, res.threads, ch));
        CHK_RET(RunAllGather   (p, res.channels, res.threads, ch));
    }
    /* PostSyncInterThreads */
    CHK_RET(PostCopy(p, res.threads));                 // hcclBuff → output
    return HCCL_SUCCESS;
}
```

每个 step 的编排（ReduceScatter 阶段）：

```cpp
std::vector<NHRStepInfo> stepInfoList;
CHK_RET(GetReduceScatterStepInfoList(stepInfoList));   // 自己实现或抄蓝本，产出每步的 to/from/片索引
for (auto& s : stepInfoList) {
    CHK_PRT_RET(channels.count(rankList_.at(s.fromRank)) == 0, HCCL_ERROR("..."), HCCL_E_INTERNAL);
    CHK_PRT_RET(channels.count(rankList_.at(s.toRank))   == 0, HCCL_ERROR("..."), HCCL_E_INTERNAL);
    const ChannelInfo& recvCh = channels.at(rankList_.at(s.fromRank)).at(channelIdx);
    const ChannelInfo& sendCh = channels.at(rankList_.at(s.toRank)).at(channelIdx);

    std::vector<DataSlice> sSrc, sDst, rSrc, rDst;
    for (u32 i = 0; i < s.nSlices; ++i) {
        u32 tx = s.txSliceIdxs.at(i), rx = s.rxSliceIdxs.at(i);
        u64 txOff = (tx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 txSz  = (tx == templateRankSize_ - 1) ? dataSplitTail_ [channelIdx] : dataSplit_ [channelIdx];
        /* rx 同理 */
        sSrc.emplace_back(localHcclBuff, base + tx * sliceSize_ + txOff, txSz, txSz / dataTypeSize_);
        sDst.emplace_back(sendCh.remoteCclMem.addr, base + tx * sliceSize_ + txOff, txSz, txSz / dataTypeSize_);
        rSrc.emplace_back(recvCh.remoteCclMem.addr, base + rx * sliceSize_ + rxOff, rxSz, rxSz / dataTypeSize_);
        rDst.emplace_back(localHcclBuff, base + rx * sliceSize_ + rxOff, rxSz, rxSz / dataTypeSize_);
    }
    SendRecvReduceInfo info{{sendCh, recvCh}, {{sSrc, sDst}, {rSrc, rDst}}, dataType_, reduceOp_};
    if (isDmaRead_) { CHK_PRT_RET(SendRecvReadReduce(info, threads.at(channelIdx)), HCCL_ERROR("..."), HCCL_E_INTERNAL); }
    else            { CHK_PRT_RET(SendRecvBatchWriteReduce(info, threads.at(channelIdx)), HCCL_ERROR("..."), HCCL_E_INTERNAL); }
}
```

AllGather 阶段一样，只是换成 `SendRecvBatchWrite` / `SendRecvBatchRead`（不带 Reduce）。

---

## 4. Gather / Scatter 类（带 repeat 与 stride）

AllGather / ReduceScatter / AlltoAll 这类"每个 rank 一块、块与块之间有跨度"的算子，
用 `TemplateDataParams` 里的 stride 字段而不是自己算 `rankIdx * sliceSize`。
地址是**两级嵌套**的：外层 `repeat`（一个 rank 有几块不相邻的数据），内层 alg rank（一块之内第几片）。
五参数语义与反例见 `09-template-data-params.md`。repeat 是调用内逻辑组数，不是通信 step。

**特定蓝本示例**：`src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc:152`。
下列 scratch 分支仅描述该蓝本的布局；其他 template 必须按自己的生产者与消费者契约推导，不能直接照搬条件。

```cpp
for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {          // 一个 rank 的多块不连续数据
    const u64 inBaseOff  = tempAlgParams_.buffInfo.inBuffBaseOff  + rpt * tempAlgParams_.inputRepeatStride;
    const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;

    // 本蓝本的紧凑 scratch 布局；不是其他 template 的默认公式
    u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff
                    + rpt * (tempAlgParams_.sliceSize * templateRankSize_);
    u64 scratchSliceStride = tempAlgParams_.sliceSize;
    // 本蓝本在此分支复用上游 CCL 布局，采用输入 stride；其他实现需独立核对
    if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        scratchSliceStride = tempAlgParams_.inputSliceStride;
    }

    // 第 algRank 个 rank 的数据落点
    u64 inOff      = tempAlgParams_.inputSliceStride  * algRank + inBaseOff;
    u64 outOff     = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
    u64 scratchOff = scratchSliceStride               * algRank + scratchBase;
    ...
}
```

三条容易踩的：

1. **覆盖本次调用的全部 repeat。** 规则寻址使用上述循环；单次处理限制或特殊消费方式须在 spec 声明。
2. 输入和输出按各自 stride 寻址；scratch 单独推导。仅凭 `inBuffType` 不能决定所有中转布局。
3. **别把 `repeatNum` 乘进 `CalcScratchMultiple()`。** 那个数按单次 repeat 报，
   放大由 executor 做（`references/03` §2.1）。地址要乘 `rpt`，预算不要。

图模式 / 对称内存路径：

```cpp
enableRemoteMemAccess_  = tempAlgParams.enableRemoteMemAccess;   // OFFLOAD 时可直接写对端 output
supportSymmetricMemory_ = tempAlgParams.supportSymmetricMemory;
if (supportSymmetricMemory_) {
    inputSymWindow_ = param.inputSymWindow;  outputSymWindow_ = param.outputSymWindow;
    inputOffset_    = param.inputOffset;     outputOffset_    = param.outputOffset;
}
```
此时落点从 `channel.remoteCclMem` 换成 `channel.remoteOutputGraphMode` / `remoteInputGraphMode`，
且 `CalcScratchMultiple()` 通常在 `opMode_ != OPBASE` 时返回 0。

---

## 5. 通用注意点

### 5.1 64-bit 数据类型与 PROD 的软归约兜底

硬件归约不支持 INT64 / UINT64 / FP64 / PROD，必须先把所有流的搬运任务真正跑完，再在 AICPU 上软归约：

```cpp
needAicpuReduce_ = dataType_ == HCCL_DATA_TYPE_INT64  ||
                   dataType_ == HCCL_DATA_TYPE_UINT64 ||
                   dataType_ == HCCL_DATA_TYPE_FP64   ||
                   param.reduceType == HCCL_REDUCE_PROD;

if (needAicpuReduce_) {
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));    // 结束批下发
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));  // 重开一批
    for (const auto& t : threads) {
        CHK_RET(static_cast<HcclResult>(HcommThreadJoin(t, CUSTOM_TIMEOUT)));  // 等所有流落地
    }
}
// 之后再 LocalReduce / AicpuReduce
```

**顺序不能变**：`BatchModeEnd → BatchModeStart → ThreadJoin`，漏掉会读到未写完的 scratch。

### 5.2 单 rank 与空数据早退

```cpp
if (count_ == 0) { HCCL_WARNING("[XXX][KernelRun] data count is 0."); return HCCL_SUCCESS; }
if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) { return HCCL_SUCCESS; }
if (templateRankSize_ == 1 || subCommRanks_[0].size() == 1) { /* 只做 LocalCopy */ return HCCL_SUCCESS; }
```

### 5.3 蓝本中的错峰对端顺序

该 Mesh 蓝本用 `(myRank_ + queIdx) % templateRankSize_` 错开各 rank 同轮的目标；其他拓扑的 peer 顺序按已确认的算法设计实现。

### 5.4 channel 取用前必须校验

```cpp
CHK_PRT_RET(channels.count(peerRank) == 0,
            HCCL_ERROR("[XXX] remoteRank[%u] is not in channels.", peerRank), HCCL_E_INTERNAL);
const ChannelInfo& ch = channels.at(peerRank).at(channelIdx);
```
`channels` 的 key 是**通信域 userRank**（即 `subCommRanks_[0][idx]`），不是算法内序号。混用是高频 bug。

### 5.5 多 channel（多 jetty）数据二次切分

一个远端有多条 channel 时，把一份数据按端口组比例分给各 channel：

```cpp
std::vector<u64> elemCountOut, sizeOut, elemOffset;
CHK_RET(CalcDataSplitByPortGroup(totalCount, dataTypeSize_, channelsVec, elemCountOut, sizeOut, elemOffset));
// 之后 dataSplit_[ch] / dataOffset_[ch] 从这三个数组取
```
并且 `CalcRes` 里要 `channelsPerRank_ = CalcChannelsPerRank(channels);`，`GetThreadNum()` 返回它。

### 5.6 Describe()

一行自述，出现在日志和 DFX 里，务必带上 `templateRankSize_`：

```cpp
std::string Describe() const override {
    std::string info = "Template of all reduce (one-shot) 1D Mesh with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}
```
