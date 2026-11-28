/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_SAMPLE_OP_H_
#define OPS_OP_PROTO_INC_SAMPLE_OP_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

// TODO(<op>): 按实际语义声明 IR（与 op_host 定义一致）
REG_OP(SampleOp)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SampleOp)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SAMPLE_OP_H_
