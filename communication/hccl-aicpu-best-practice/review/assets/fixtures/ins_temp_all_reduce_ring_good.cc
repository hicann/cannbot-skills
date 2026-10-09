/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_temp_all_reduce_ring_good.h"
namespace ops_hccl {

u64 InsTempAllReduceRingGood::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 1; // ring 在 scratch 上原地累加，一份就够
    return scratchMultiple;
}

HcclResult InsTempAllReduceRingGood::CalcRes(HcclComm comm, const OpParam& param,
                                             const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& req)
{
    req.slaveThreadNum = 1; // ring 同时只有一收一发，与 rankSize 无关
    req.notifyNumPerThread.assign(req.slaveThreadNum, 1);
    req.notifyNumOnMainThread = req.slaveThreadNum;
    std::vector<HcclChannelDesc> ringChannels;
    CHK_RET(CalcChannelRequestRing(comm, param, topoInfo, subCommRanks_, ringChannels));
    req.channels.push_back(ringChannels);
    return HCCL_SUCCESS;
}

std::string InsTempAllReduceRingGood::Describe() const
{
    std::string info = "Template of all reduce ring with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

HcclResult InsTempAllReduceRingGood::KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res)
{
    rankList_ = subCommRanks_.at(0);
    CHK_RET(GetAlgRank(myRank_, rankList_, myRankIdx_));
    count_ = p.count;
    if (count_ == 0) {
        return HCCL_SUCCESS;
    }
    if (templateRankSize_ == 1) {
        return LocalCopyOnly(res.threads);
    }

    const u32 nextIdx = (myRankIdx_ + 1) % templateRankSize_;
    const u32 prevIdx = (myRankIdx_ - 1 + templateRankSize_) % templateRankSize_;
    nextRank_ = rankList_.at(nextIdx);
    prevRank_ = rankList_.at(prevIdx);
    CHK_PRT_RET(res.channels.count(nextRank_) == 0 || res.channels.count(prevRank_) == 0,
                HCCL_ERROR("[RingGood] neighbour not in channels"), HCCL_E_INTERNAL);

    CHK_RET(SplitData()); // RS 与 AG 共用这一张分片表
    for (u32 rpt = 0; rpt < p.repeatNum; ++rpt) {
        // ReduceScatter：N-1 步，第 step 步发 (myRankIdx_ - step + N) % N 号片
        for (u32 step = 0; step < templateRankSize_ - 1; ++step) {
            const u32 txIdx = (myRankIdx_ - step + templateRankSize_) % templateRankSize_;
            const u32 rxIdx = (myRankIdx_ - step - 1 + templateRankSize_) % templateRankSize_;
            CHK_RET(PreSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxMainToSub_));
            CHK_RET(SendRecvBatchWriteReduce(MakeStepInfo(txIdx, rxIdx, rpt), res.threads[1]));
            CHK_RET(PostSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxSubToMain_));
        }
        // AllGather：同样 N-1 步，索引整体后移一位
        for (u32 step = 0; step < templateRankSize_ - 1; ++step) {
            const u32 txIdx = (myRankIdx_ - step + 1 + templateRankSize_) % templateRankSize_;
            const u32 rxIdx = (myRankIdx_ - step + templateRankSize_) % templateRankSize_;
            CHK_RET(PreSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxMainToSub_));
            CHK_RET(SendRecvBatchWrite(MakeStepInfo(txIdx, rxIdx, rpt), res.threads[1]));
            CHK_RET(PostSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxSubToMain_));
        }
    }
    if (needAicpuReduce_) {
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
