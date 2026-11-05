/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file drop_out_v3_split_fusion_pass.h
 * \brief DropOutV3 split fusion pass: DropOutV3 -> StatelessDropOutGenMask + DropOutDoMask
 */

#ifndef MATH_DROP_OUT_V3_SPLIT_FUSION_PASS_H
#define MATH_DROP_OUT_V3_SPLIT_FUSION_PASS_H

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ge::fusion {

using namespace ge;

struct InputInfo {
    std::vector<int64_t> xDims;
    std::vector<int64_t> pDims;
    std::vector<int64_t> seedDims;
    std::vector<int64_t> offsetDims;
    std::vector<int64_t> noiseShapeDims;
    DataType xDtype;
    DataType pDtype;
    DataType seedDtype;
    DataType offsetDtype;
    Format fmt;
};

class __attribute__((visibility("default"))) DropOutV3SplitFusionPass : public FusionBasePass {
public:
    Status Run(GraphPtr &graph, CustomPassContext &pass_context) override;

private:
    bool CheckPlatform() const;
    bool CheckDtypes(const GNode &node) const;
    bool CheckNode(const GNode &node) const;
    InputInfo GetInputInfo(const GNode &node) const;
    void UpdateTensorDescs(const InputInfo &info,
                           const es::EsTensorHolder &rX,
                           const es::EsTensorHolder &rProb,
                           const es::EsTensorHolder &rSeed,
                           const es::EsTensorHolder &rOffset,
                           const es::EsTensorHolder &genMask,
                           const es::EsTensorHolder &doMask,
                           const es::EsTensorHolder &rShapeConst,
                           const es::EsTensorHolder &rSeed1) const;
    GraphUniqPtr CreateReplacement(const GNode &node);
    std::unique_ptr<SubgraphBoundary> ConstructBoundary(const GNode &node);
};

} // namespace ge::fusion
#endif // MATH_DROP_OUT_V3_SPLIT_FUSION_PASS_H
