{{LICENSE}}

#include "{{HEADER}}"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

{{CLASS}}::{{CLASS}}(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

{{CLASS}}::~{{CLASS}}() {}

std::vector<CostModelParam> {{CLASS}}::CalcCostCoeff(CalcCostCoeffParam param)
{
    if (param.rankSize > 8) { // one-shot 通信量随 rankSize 线性增长，大规模不适用
        return {};
    }
    int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int kernelNum = 1;
    int taskNum = CostModelManager::CalcTransTaskNum(param.rankSize)
                  + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2 + (param.rankSize - 1);
    float A = 0.0f;   // 传输项
    float B = 0.0f;   // 本地计算项
    float C = 0.0f;   // 延迟项
    float D = 0.0f;   // 下发项
    // executor 层统一传 two-shot 需要的一片数据大小，one-shot 在此乘回 rankSize
    float n = param.dataRatio * param.rankSize;
    float B1 = 0.0f;
    float B2 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(n, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {   // 输入不在 scratch 上时才有一次 LocalCopy
        CostModelManager::Global()->CalcLocalCopyParams(n, EngineType::AICPU, B1);
    }
    CostModelManager::Global()->CalcLocalReduceParams(n, EngineType::AICPU, B2);
    B = B1 + (param.rankSize - 1) * B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});   // 四个成员要填满，漏一个会被 -Werror=missing-field-initializers 拦下
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

HcclResult {{CLASS}}::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做 level0 层级
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1; // 主 thread 用接口传入的 stream
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_INFO("[{{CLASS}}][CalcRes] threadNum[%u] channelNum[%zu].",
              threadNum, level0Channels.size());
    return HCCL_SUCCESS;
}

u64 {{CLASS}}::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_; // 每个 rank 在我 scratch 里独占一段
    return scratchMultiple;
}

void {{CLASS}}::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    notifyIdxMainToSub.assign(threadNum - 1, 0);
}

void {{CLASS}}::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    for (u32 notifyIdx = 0; notifyIdx < threadNum - 1; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult {{CLASS}}::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec) const
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(templateRankSize_, tmp);

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        SliceInfo slice = {accumOff, dataSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += dataSize;
    }
    CHK_PRT_RET(
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size
         != dataSize * templateRankSize_),
        HCCL_ERROR("[{{CLASS}}][CalcSlice] Rank [%u], SliceInfo calculation error!",
                   myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult {{CLASS}}::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[{{CLASS}}][KernelRun] Start.");
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    needAicpuReduce_
        = dataType_ == HcclDataType::HCCL_DATA_TYPE_INT64 || dataType_ == HcclDataType::HCCL_DATA_TYPE_UINT64
          || dataType_ == HcclDataType::HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;

    if (count_ == 0) {
        HCCL_WARNING("[{{CLASS}}][KernelRun] data count is 0.");
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        subCommRanks_.empty(), HCCL_ERROR("[{{CLASS}}][KernelRun] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        threadNum_ != templateRankSize_,
        HCCL_ERROR("[{{CLASS}}][KernelRun] thread num is invalid, need[%u], actual[%u].",
                   templateRankSize_, threadNum_),
        HcclResult::HCCL_E_INTERNAL);

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(processSize_, sliceInfoVec));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunExchange(param, templateResource.channels, templateResource.threads, tempAlgParams, sliceInfoVec));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    CHK_RET(PostLocalReduce(param, templateResource.threads, tempAlgParams, sliceInfoVec));
    HCCL_INFO("[{{CLASS}}][KernelRun] finished: rank[%u] end.", myRank_);
    return HCCL_SUCCESS;
}

HcclResult {{CLASS}}::RunExchange(
    const OpParam& param, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams,
    const RankSliceInfo& sliceInfoVec)
{
    (void)param;
    HCCL_INFO("[{{CLASS}}][RunExchange] send/recv: rank[%u].", myRank_);

    DataSlice usrInSlices
        = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
    DataSlice usrOutSlices
        = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

    // 主流 - 本地拷贝
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], usrInSlices, usrOutSlices)));

    // 单 rank 场景
    if (subCommRanks_[0].size() == 1) {
        return HCCL_SUCCESS;
    }

    // 从流 - 与各对端交换整份数据
    for (u32 queIdx = 1; queIdx < threadNum_; queIdx++) {
        u32 peerIdx = (myRank_ + queIdx) % templateRankSize_; // 错峰，避免全员打同一目标
        u32 peerRank = subCommRanks_[0][peerIdx];
        CHK_PRT_RET(
            channels.count(peerRank) == 0,
            HCCL_ERROR("[{{CLASS}}][RunExchange] remoteRank[%u] is not in channels.",
                       peerRank),
            HcclResult::HCCL_E_INTERNAL);
        const ChannelInfo& linkRecv = channels.at(peerRank)[0]; // 从 peerRank 接收的链路
        const ChannelInfo& linkSend = channels.at(peerRank)[0]; // 向 peerRank 发送的链路

        // 我的数据 → 对端 scratch 的第 myRank_ 段
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        txSrcSlices.push_back(usrInSlices);
        txDstSlices.push_back(DataSlice(
            linkSend.remoteCclMem.addr, sliceInfoVec[myRank_][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff,
            sliceInfoVec[myRank_][0].size, count_));

        // 对端数据 → 我 scratch 的第 peerIdx 段
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        rxSrcSlices.push_back(usrInSlices);
        rxDstSlices.push_back(DataSlice(
            linkRecv.remoteCclMem.addr, sliceInfoVec[peerIdx][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff,
            sliceInfoVec[peerIdx][0].size, count_));

        SendRecvInfo sendRecvInfo{
            {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
        CHK_PRT_RET(
            SendRecvBatchWrite(sendRecvInfo, threads[queIdx]),
            HCCL_ERROR("[{{CLASS}}][RunExchange] SendRecvBatchWrite failed."),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult {{CLASS}}::PostLocalReduce(
    const OpParam& param, const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams,
    const RankSliceInfo& sliceInfoVec)
{
    HCCL_INFO("[{{CLASS}}][PostLocalReduce] reduce: rank[%u].", myRank_);
    // 64-bit 数据类型 / PROD 需要先把各流任务真正跑完，再做软归约
    if (needAicpuReduce_) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto& thread : threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }

    DataSlice usrOutSlices
        = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

    for (u32 curRank = 0; curRank < subCommRanks_[0].size(); curRank++) {
        if (curRank == myRank_) { // 自己那份已经通过 LocalCopy 进了 output
            continue;
        }
        DataSlice curSrcSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            sliceInfoVec[curRank][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff, sliceInfoVec[curRank][0].size,
            count_);
        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], curSrcSlice, usrOutSlices, dataType_, reduceOp_)));
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
