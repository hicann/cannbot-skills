/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3dtranspose_proto.h
 * \brief Conv3DTranspose GEIR 原型注册（与 op_host/conv3dtranspose_def.cpp 同一 Interface.md 契约）
 */
#ifndef OPS_OP_PROTO_INC_CONV3DTRANSPOSE_H_
#define OPS_OP_PROTO_INC_CONV3DTRANSPOSE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Conv3DTranspose：三维转置卷积算子（反卷积）。给定特征图 x（dedy）、卷积核 filter、
 * 可选偏置 bias 与目标输出形状 input_size，输出与正向三维卷积输入同形状的特征图 y（dedx）；
 * 数学上是正向三维卷积关于输入的伴随（转置）映射（scatter-add 散射累加）。
 *
 * @par Inputs:
 * Four inputs, including:
 * @li input_size: A 1D Tensor of length 5. Must be one of the following types: int32, int64.
 * Element values give the shape of y; axis order follows data_format (NCDHW: N, Cin, Din,
 * Hin, Win; NDHWC: N, Din, Hin, Win, Cin). Value-dependent shape descriptor, not consumed
 * by compute.
 * @li x: A 5D Tensor of the transposed-convolution input (dedy). Must be one of the
 * following types: float16, bfloat16. Layout (NCDHW/NDHWC) is selected by attr
 * data_format and pairs strictly with y.
 * @li filter: A 5D Tensor of the weight (kernel window enters unit-stride cross
 * correlation after rot180). Must be one of the following types: float16, bfloat16.
 * Registered layouts NCDHW/NDHWC/DHWCN are independent of data_format.
 * @li bias: Optional. A 1D Tensor of per-channel bias broadcast along the y channel axis
 * (covers positions with no product contribution). Must be one of the following types:
 * float32, float16. Length equals Cin (= filter dim1 * groups); when omitted, y contains
 * the convolution term only and non-contribution positions are written as 0.
 *
 * (Reserved quantization slot offset_w is NOT registered as an input: the verified
 * whitebox case set materializes 4-input rows and TTK kernel mode enforces strict
 * input-count equality — see op_host/conv3dtranspose_def.cpp integration note and
 * tests/whitebox/branch-1/bringup/RECORD.md; the slot is never instantiated in this
 * delivery.)
 *
 * @par Outputs:
 * @li y: A 5D Tensor of the transposed-convolution output (dedx). Must be one of the
 * following types: float16, bfloat16. Has the same dtype as x; format pairs strictly with
 * x. Shape equals the element values of input_size.
 *
 * @par Attributes:
 * @li strides: Required. A list of 5 ints. N and C positions must be 1; axis order
 * follows data_format (NCDHW: N, C, sD, sH, sW; NDHWC: N, sD, sH, sW, C).
 * @li pads: Required. A list of 6 ints [pD_front, pD_back, pH_top, pH_bottom, pW_left,
 * pW_right]; semantics are independent of data_format; single-side asymmetric padding is
 * supported (pads are subtracted from the output geometry in the transposed direction).
 * @li dilations: Optional. A list of 5 ints, default {1, 1, 1, 1, 1}. N and C positions
 * must be 1.
 * @li groups: Optional. An int, default 1. Group count in [1, 65535]; requires
 * Cin % groups == 0 and Cout % groups == 0.
 * @li data_format: Optional. A string, default "NDHWC". One of {"NCDHW", "NDHWC"}.
 * @li output_padding: Optional. A list of 5 ints, default {0, 0, 0, 0, 0}. Output shape
 * increment q; N and C positions must be 0; per axis q < s or q < d.
 * @li offset_x: Optional. An int, default 0. Reserved attribute, not enabled.
 * @li enable_hf32: Optional. A bool, default false. FP32-path hf32 switch; reserved for
 * the FP32 extension (not in this delivery's dtype set).
 */
REG_OP(Conv3DTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DTranspose)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CONV3DTRANSPOSE_H_
