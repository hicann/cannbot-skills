/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DRAFT — 范式参考模板，未编译、未上板 */

/*!
 * \file max_pool_grad_with_argmax_proto.h
 * \brief 反向算子原型：argmax 路由梯度回传
 */

#ifndef OPS_OP_PROTO_INC_MAX_POOL_GRAD_WITH_ARGMAX_H_
#define OPS_OP_PROTO_INC_MAX_POOL_GRAD_WITH_ARGMAX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief SlidingWindow 反向 GradRoute·B（argmax 路由）参考模板。
 * @par Inputs:
 * @li x: 原输入（NCHW），值不消费，作为 y 的形状源。
 * @li grad: 正向输出同形梯度（N, C, outH, outW），float32 / float16。
 * @li argmax: 正向 winner 线性索引（口径 h*W+w），int32 / int64——与正向输出同 dtype 同口径。
 * @par Outputs:
 * y: 与 x 同形的梯度回传结果。
 * @par Attributes:
 * ksize / strides / pads / dilations：与正向同源。
 */
REG_OP(MaxPoolGradWithArgmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(strides, ListInt, {1, 1})
    .ATTR(pads, ListInt, {0, 0})
    .ATTR(dilations, ListInt, {1, 1})
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmax)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MAX_POOL_GRAD_WITH_ARGMAX_H_
