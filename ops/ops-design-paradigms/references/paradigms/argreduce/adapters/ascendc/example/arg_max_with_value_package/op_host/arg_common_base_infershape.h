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
 * \file arg_common_base_infershape.h
 * \brief
 */

#ifndef ARG_COMMON_BASE_INFERSHAPE_H_
#define ARG_COMMON_BASE_INFERSHAPE_H_

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
const int64_t INPUT_IDX_AXIS = 1;
const int64_t INPUT_IDX_X = 0;
const int64_t OUT_IDX_INDICE = 0;
const int64_t OUT_IDX_VALUE = 1;

ge::graphStatus InferShapeForArgOps(gert::InferShapeContext* context);

ge::graphStatus InferShapeForArgOpsWithValue(gert::InferShapeContext* context);
}  // namespace ops

#endif // ARG_COMMON_BASE_INFERSHAPE_H_