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
 * \file max_pool_with_argmax_proto.h
 * \brief 正向算子原型：I/O 契约（滑动窗口 max 聚合 + argmax mask 双输出）
 */

#ifndef OPS_OP_PROTO_INC_MAX_POOL_WITH_ARGMAX_H_
#define OPS_OP_PROTO_INC_MAX_POOL_WITH_ARGMAX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief SlidingWindow 正向 WindowAggregate（max 聚合）参考模板。
 * @par Inputs:
 * x: 4D Tensor（NCHW/NHWC，data_format 决定），float32 / float16。
 * @par Outputs:
 * @li y: 聚合值，与 x 同 dtype，shape = 窗口公式推导。
 * @li argmax: winner 线性索引（批次内 h*W+w 口径），int32 / int64。
 * @par Attributes:
 * @li ksize / strides / pads / dilations: 长度 2 列表 [H, W]。
 * @li data_format: "NCHW"（默认）或 "NHWC"。
 */
REG_OP(MaxPoolWithArgmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(strides, ListInt, {1, 1})
    .ATTR(pads, ListInt, {0, 0})
    .ATTR(dilations, ListInt, {1, 1})
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(MaxPoolWithArgmax)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MAX_POOL_WITH_ARGMAX_H_
