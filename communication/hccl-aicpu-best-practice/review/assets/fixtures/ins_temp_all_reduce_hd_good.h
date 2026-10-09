/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* 评审器自测夹具（不参与编译）：写法正确的 HD（recursive halving-doubling）template 骨架。 */
#ifndef INS_TEMP_ALL_REDUCE_HD_GOOD_H
#define INS_TEMP_ALL_REDUCE_HD_GOOD_H
#include "alg_v2_template_base.h"
namespace ops_hccl {
class InsTempAllReduceHdGood : public InsAlgTemplateBase {
public:
    InsTempAllReduceHdGood() = default;
    explicit InsTempAllReduceHdGood(const OpParam& param, const u32 rank,
                                    const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllReduceHdGood() override = default;
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
    u32 blockSize_ = 0; // 不大于 rankSize 的最大 2 的幂
    u32 part1Size_ = 0; // 参与「两两合并」的 rank 数 = 2 * (rankSize - blockSize)
    u32 round_ = 0;     // log2(blockSize)
};
} // namespace ops_hccl
#endif // INS_TEMP_ALL_REDUCE_HD_GOOD_H
