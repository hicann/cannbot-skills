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
 * \file conv3ddx_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CONV3DDX_H_
#define OPS_OP_PROTO_INC_CONV3DDX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Conv3ddx：三维卷积输入梯度算子。给定正向卷积的输出梯度 out_backprop（dedy）与权重
 * filter，按 input_size 给出的形状计算对卷积输入的梯度 y（dedx）；数学上等价于转置卷积前向。
 *
 * @par Inputs:
 * Three inputs, including:
 * @li input_size: A 1D Tensor of length 5 (int32/int64). Element values give the shape
 * of y (= forward input dedx); axis order follows data_format (NCDHW: N, Cin, Din, Hin, Win;
 * NDHWC: N, Din, Hin, Win, Cin). Value-dependent shape descriptor, not consumed by compute.
 * @li filter: A 5D Tensor of the forward convolution weight. One of the following types:
 * float16, bfloat16. Registered layouts NCDHW/NDHWC/DHWCN are independent of data_format.
 * @li out_backprop: A 5D Tensor of the output gradient (dedy). One of the following types:
 * float16, bfloat16. Layout (NCDHW/NDHWC) is selected by attr data_format.
 *
 * @par Outputs:
 * @li y: A 5D Tensor of the input gradient (dedx). Has the same dtype as out_backprop;
 * format pairs strictly with out_backprop.
 *
 * @par Attributes:
 * @li strides: Required. A list of 5 ints. N and C positions must be 1; axis order follows
 * data_format (NCDHW: N, C, sD, sH, sW; NDHWC: N, sD, sH, sW, C).
 * @li pads: Required. A list of 6 ints [pD_front, pD_back, pH_top, pH_bottom, pW_left,
 * pW_right]; semantics are independent of data_format; single-side asymmetric padding supported.
 * @li dilations: Optional. A list of 5 ints, default {1, 1, 1, 1, 1}. N and C positions
 * must be 1.
 * @li groups: Optional. An int, default 1. Group count; requires Cin % groups == 0 and
 * Cout % groups == 0.
 * @li data_format: Optional. A string, default "NDHWC". One of {"NCDHW", "NDHWC"}.
 * @li enable_hf32: Optional. A bool, default false. FP32-path hf32 switch; reserved for
 * the FP32 extension (not in this delivery's dtype set).
 */
REG_OP(Conv3ddx)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3ddx)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CONV3DDX_H_
