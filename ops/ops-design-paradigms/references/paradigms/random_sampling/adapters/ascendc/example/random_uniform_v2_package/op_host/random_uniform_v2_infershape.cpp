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
 * \file random_uniform_v2.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "random/random_common/op_host/random_infershape_base.h"

using namespace ge;
namespace ops {
static constexpr size_t RANDOM_UNIFORM_V2_X = 0;
static constexpr size_t RANDOM_UNIFORM_V2_OFFSET = 1;
static constexpr size_t RANDOM_UNIFORM_V2_Y = 0;

static graphStatus InferShapeRandomUniformV2(gert::InferShapeContext* context)
{
    const std::unordered_map<std::string, size_t>& inputMap = {{"shape", RANDOM_UNIFORM_V2_X},
                                                               {"offset", RANDOM_UNIFORM_V2_OFFSET}};
    const std::unordered_map<std::string, size_t>& outputMap = {{"y", RANDOM_UNIFORM_V2_Y},
                                                                {"offset", RANDOM_UNIFORM_V2_OFFSET}};
    int32_t mode = ops::randomCommon::MODE_DEPENDENCY;
    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}
IMPL_OP_INFERSHAPE(RandomUniformV2).InputsDataDependency({0}).InferShape(InferShapeRandomUniformV2);

} // namespace ops
