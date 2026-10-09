/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* 评审器自测夹具（不参与编译）：把 references/06 的不变式写错的 HD template。 */
#include "ins_temp_all_reduce_hd_bad.h"
namespace ops_hccl {

u64 InsTempAllReduceHdBad::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_; // H06：HD 不需要 N 倍 scratch
}

std::string InsTempAllReduceHdBad::Describe() const
{
    return "Template of all reduce hd"; // C13
}

HcclResult InsTempAllReduceHdBad::CalcRes(HcclComm comm, const OpParam& param,
                                          const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& req)
{
    req.slaveThreadNum = 1;
    req.notifyNumPerThread.assign(req.slaveThreadNum, 1);
    req.notifyNumOnMainThread = req.slaveThreadNum;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHdBad::KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res)
{
    rankList_ = subCommRanks_.at(0);
    count_ = p.count;
    // H01：直接假设 rankSize 是 2 的幂，没有 part1 / blockSize 的降级处理
    // H02 + H03：步数写死，对端用 idx + step 而不是 XOR
    // H04：每步数据量恒定
    for (u32 step = 0; step < 3; ++step) {
        const u32 peer = rankList_.at((myRankIdx_ + step) % templateRankSize_);
        CHK_RET(SendRecvBatchWriteReduce(MakeStepInfo(peer, sliceSize_), res.threads[1]));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
