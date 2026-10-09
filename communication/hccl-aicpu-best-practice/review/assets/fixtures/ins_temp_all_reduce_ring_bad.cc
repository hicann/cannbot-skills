/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* 评审器自测夹具（不参与编译）：把 references/05 的不变式逐条写错的 Ring template。
 * 每处错误后面标了它应该触发的规则号，selftest.sh 会核对是不是都报出来了。 */
#include "ins_temp_all_reduce_ring_bad.h"
namespace ops_hccl {

u64 InsTempAllReduceRingBad::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_ * repeatNum_; // R05 + C01：倍数错，还乘了 repeatNum
    return scratchMultiple;
}

HcclResult InsTempAllReduceRingBad::CalcRes(HcclComm comm, const OpParam& param,
                                            const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& req)
{
    req.slaveThreadNum = templateRankSize_ - 1; // R06：照搬 Mesh 的线程模型
    req.notifyNumPerThread.assign(req.slaveThreadNum, 1);
    req.notifyNumOnMainThread = req.slaveThreadNum;
    return HCCL_SUCCESS;
}

std::string InsTempAllReduceRingBad::Describe() const
{
    return "Template of all reduce ring"; // C13：没带 templateRankSize_
}

HcclResult InsTempAllReduceRingBad::KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res)
{
    rankList_ = subCommRanks_.at(0);
    count_ = p.count;
    // R01：对全体 rank 取 channel，且没有左右邻居的推导
    for (u32 i = 0; i < templateRankSize_; ++i) {
        const ChannelInfo& ch = res.channels.at(rankList_.at(i))[0]; // C11：没有 count() 校验
        (void)ch;
    }
    // R02：步数写死；R03：片索引直接用 i，没有 (idx - step + N) % N；R04：step 之间没有同步
    for (u32 step = 0; step < 7; ++step) {
        SendRecvBatchWriteReduce(MakeStepInfo(step, step), res.threads[1]); // C12：没被 CHK_RET 接住
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
