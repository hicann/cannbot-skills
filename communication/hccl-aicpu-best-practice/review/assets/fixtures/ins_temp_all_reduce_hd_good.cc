/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_temp_all_reduce_hd_good.h"
namespace ops_hccl {

u64 InsTempAllReduceHdGood::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 1; // 原地折半累加，一份就够
    return scratchMultiple;
}

std::string InsTempAllReduceHdGood::Describe() const
{
    std::string info = "Template of all reduce halving-doubling with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

HcclResult InsTempAllReduceHdGood::CalcRes(HcclComm comm, const OpParam& param,
                                           const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& req)
{
    req.slaveThreadNum = 1;
    req.notifyNumPerThread.assign(req.slaveThreadNum, 1);
    req.notifyNumOnMainThread = req.slaveThreadNum;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHdGood::CalcPartOneSizeAndBlockSize()
{
    blockSize_ = 1;
    while ((blockSize_ << 1) <= templateRankSize_) {
        blockSize_ = blockSize_ << 1;
    }
    part1Size_ = (templateRankSize_ - blockSize_) * 2; // 前 part1Size_ 个 rank 两两合并
    round_ = 0;
    for (u32 b = blockSize_; b > 1; b = b >> 1) {
        round_++;
    } // round_ = log2(blockSize_)
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHdGood::KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res)
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
    CHK_RET(CalcPartOneSizeAndBlockSize());

    for (u32 rpt = 0; rpt < p.repeatNum; ++rpt) {
        // ① part1 淘汰：偶数 rank 把数据交给 rank+1，之后不参与主循环
        if (myRankIdx_ < part1Size_ && (myRankIdx_ % 2) == 0) {
            const u32 peer = rankList_.at(myRankIdx_ + 1);
            CHK_PRT_RET(res.channels.count(peer) == 0, HCCL_ERROR("[HdGood] peer missing"), HCCL_E_INTERNAL);
            CHK_RET(SendRecvBatchWriteReduce(MakeInfo(peer, rpt), res.threads[1]));
            return PostRecvFromBlockRank(res, rpt); // ③ 末尾等 rank+1 把最终结果回传
        }
        // ② 主循环只在 blockSize_ 个 rank 上跑：halving 阶段每步数据量减半
        u64 stepSize = sliceSize_;
        for (u32 step = 0; step < round_; ++step) {
            const u32 peerIdx = blockIdx_ ^ (1u << step); // XOR 找对端
            const u32 peer = rankList_.at(BlockIdxToAlgIdx(peerIdx));
            CHK_PRT_RET(res.channels.count(peer) == 0, HCCL_ERROR("[HdGood] peer missing"), HCCL_E_INTERNAL);
            stepSize = stepSize / 2;
            CHK_RET(PreSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxMainToSub_));
            CHK_RET(SendRecvBatchWriteReduce(MakeStepInfo(peer, stepSize, rpt), res.threads[1]));
            CHK_RET(PostSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxSubToMain_));
        }
        // doubling 阶段：逆序回放，每步数据量翻倍
        for (u32 step = round_; step > 0; --step) {
            const u32 peerIdx = blockIdx_ ^ (1u << (step - 1));
            const u32 peer = rankList_.at(BlockIdxToAlgIdx(peerIdx));
            CHK_PRT_RET(res.channels.count(peer) == 0, HCCL_ERROR("[HdGood] peer missing"), HCCL_E_INTERNAL);
            CHK_RET(PreSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxMainToSub_));
            CHK_RET(SendRecvBatchWrite(MakeStepInfo(peer, stepSize, rpt), res.threads[1]));
            CHK_RET(PostSyncInterThreads(res.threads[0], {res.threads[1]}, notifyIdxSubToMain_));
            stepSize = stepSize * 2;
        }
        CHK_RET(SendBackToPartOneRank(res, rpt)); // ③ 把结果回传给被淘汰的偶数 rank
    }
    if (needAicpuReduce_) {
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
