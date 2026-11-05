/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file drop_out_v3_graph_infer.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "random/random_common/op_graph/random_graph_infer_base.h"

using namespace ge;
namespace ops {
static constexpr size_t DropOutV3_MASK_IDX = 1;

static ge::graphStatus InferDataTypeDropOutV3(gert::InferDataTypeContext* context)
{
    int32_t mode = ops::GraphCommon::MODE_INPUT_EQUAL_OUTPUT;
    int32_t dtypeIndex = 0;
    const std::vector<ops::GraphCommon::OutputSpec>& extraOutputMap = {{"mask", DropOutV3_MASK_IDX, ge::DT_UINT8}};
    return ops::GraphCommon::CommonInferType(context, mode, dtypeIndex, extraOutputMap);
}

IMPL_OP(DropOutV3).InferDataType(InferDataTypeDropOutV3);
} // namespace ops
