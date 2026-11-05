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
 * \file drop_out_v3_infershape.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_util.h"
#include "random/random_common/op_host/random_infershape_base.h"
using namespace ge;
namespace ops {
static constexpr size_t DropOutV3_X = 0;
static constexpr size_t DropOutV3_P = 2;
static constexpr size_t DropOutV3_SEED = 3;
static constexpr size_t DropOutV3_OFFSET = 4;
static constexpr size_t DropOutV3_Y = 0;
static constexpr size_t DropOutV3_MASK = 1;
static constexpr size_t DropOutV3_NOISE = 1;
static constexpr size_t MAX_DIM_NUM = 8;

static graphStatus InferShapeDropOutV3(gert::InferShapeContext* context)
{
    const std::unordered_map<std::string, size_t>& inputMap = {
        {"x", DropOutV3_X}, {"p", DropOutV3_P}, {"seed", DropOutV3_SEED}, {"offset", DropOutV3_OFFSET}};
    const std::unordered_map<std::string, size_t>& outputMap = {{"y", DropOutV3_Y}, {"mask", DropOutV3_MASK}};
    int32_t mode = ops::randomCommon::MODE_NO_DEPENDENCY;
    const gert::Shape* noiseInputShape = context->GetOptionalInputShape(DropOutV3_NOISE);
    if (noiseInputShape != nullptr) {
        if (noiseInputShape->GetDimNum() > MAX_DIM_NUM) {
            return ge::GRAPH_FAILED;
        }
    }
    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}
IMPL_OP_INFERSHAPE(DropOutV3).InferShape(InferShapeDropOutV3);

} // namespace ops
