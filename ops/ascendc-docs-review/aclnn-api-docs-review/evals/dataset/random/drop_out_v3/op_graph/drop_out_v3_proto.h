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
 * \file drop_out_v3_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DROP_OUT_V3_H_
#define OPS_OP_PROTO_INC_DROP_OUT_V3_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief During training, randomly zeroes some of the elements of the input tensor
 * with probability
 *
 * @par Inputs:
 * @li x: A tensor, support ND format. Must be one of the following data types: float32,float16,bfloat16,1 ~ 8-D.
 * @li noise_shape: A tensor, optional tensor. Must be int64,1-D.
 * @li p: A required input, should be const data. Must be one of the following data types: double,float32,float16,bfloat16.
 * @li seed: A required input, should be const data. Must be one of the following data types: int32,int64, 1-D.
 * @li offset: A required input, should be const data. Must be int64, 1-D.
 * Shape is 2 and the value of index 0 is 0. Value of index 1 must be a multiple of 4.
 *
 * @par Outputs:
 * @li y: A tensor with the same shape and type as "x".
 * @li mask_out: A tensor with the shape and type support uint8.
 */

REG_OP(DropOutV3)
    .INPUT(x, "T")
    .OPTIONAL_INPUT(noise_shape, TensorType({DT_INT64}))
    .INPUT(p, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, "T")
    .OUTPUT(mask, TensorType({DT_UINT8}))
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(DropOutV3)

} // namespace ge
#endif // OPS_OP_PROTO_INC_DROP_OUT_V3_H_
