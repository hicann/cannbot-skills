/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* 评审器自测夹具（不参与编译）：一个「写法正确」的 Ring template 骨架。
 * 仓内没有 AICPU ring 实现，这份夹具用来锁住 references/05 里那几条不变式的正例。 */
#ifndef INS_TEMP_ALL_REDUCE_RING_BAD_H
#define INS_TEMP_ALL_REDUCE_RING_BAD_H
#include "alg_v2_template_base.h"
namespace ops_hccl {
class InsTempAllReduceRingBad : public InsAlgTemplateBase {
public:
    InsTempAllReduceRingBad() = default;
    explicit InsTempAllReduceRingBad(const OpParam& param, const u32 rank,
                                     const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllReduceRingBad() override = default;
    std::string Describe() const override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& req) override;
    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& p, TemplateResource& res) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& idx) override
    {
        idx.assign(slaveThreadNum_, 0);
    }
    void GetNotifyIdxSubToMain(std::vector<u32>& idx) override
    {
        idx.clear();
        for (u32 i = 0; i < slaveThreadNum_; ++i) {
            idx.push_back(i);
        }
    }

private:
    u32 slaveThreadNum_ = 1;
    u32 nextRank_ = 0;
    u32 prevRank_ = 0;
};
} // namespace ops_hccl
#endif // INS_TEMP_ALL_REDUCE_RING_BAD_H
