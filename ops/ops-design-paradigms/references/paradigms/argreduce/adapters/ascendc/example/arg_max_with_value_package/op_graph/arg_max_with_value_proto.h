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
 * \file arg_max_with_value_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ARG_MAX_WITH_VALUE_H_
#define OPS_OP_PROTO_INC_ARG_MAX_WITH_VALUE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns the maximum value of all elements in the input in the given
* dimension.

* @par Inputs:
* One input:
* x: A multi-dimensional tensor of type bfloat16, float16, float32, int64, int32. \n

* @par Attributes:
* @li dimension: An integer of type int32, specifying the axis information of
* the index with the maximum value. Required and no default value.
* @li keep_dims: A bool, specifying whether to keep dimensions for the output
* tensor. Optional and defaults to "false".
* @li indice_dtype: A Type, The output type, either "int32" or "int64". Defaults to "int32". \n

* @par Outputs:
* @li indice: A multi-dimensional tensor of type int32 or int64, specifying the index.
* (If "keep_dims" is set to "false", the output dimensions are reduced by
* "dimension" compared with that of "x". Otherwise, the output keeps the dimensions, but
* the length of corresponding dimension will be set to "1".)
* @li values: A ND tensor, specifying the maximum value. Has the same dimensions
* as "indice" and the same dtype as "x". \n

* @attention Constraints:
* @li If there are multiple maximum values, the index of the first maximum
* value is used.
* @li The value range of "dimension" is [-dims, dims - 1]. "dims" is the
* dimension length of "x". \n

* @par Third-party framework compatibility
* Compatible with the two output scenarios of PyTorch operator Max (the output
* sequence is opposite to that of PyTorch).
*/
REG_OP(ArgMaxWithValue)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32}))
    .OUTPUT(indice, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
    .ATTR(indice_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(ArgMaxWithValue)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ARG_MAX_WITH_VALUE_H_

