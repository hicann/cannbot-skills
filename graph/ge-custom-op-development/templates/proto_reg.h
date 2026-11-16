/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 同步原型、实现注册和构图侧 op type；显式注册时实现类名可不同。
// 将此头与 proto_reg.cpp 放入 src/，确保原型 TU 参与构建。
#ifndef CUSTOM_OP_PROTO_REG_H
#define CUSTOM_OP_PROTO_REG_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
// 将 MyCustomOp 替换为目标 op type；自动映射时同步实现类名，或使用显式注册。
REG_OP(MyCustomOp)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MyCustomOp);

// 多输出示例
// REG_OP(MyMultiOutOp)
//     .INPUT(x, TensorType({DT_FLOAT}))
//     .OUTPUT(y, TensorType({DT_FLOAT}))
//     .OUTPUT(z, TensorType({DT_FLOAT}))
//     .OP_END_FACTORY_REG(MyMultiOutOp);

// 动态输入示例（dynamic input）
// REG_OP(MyDynamicOp)
//     .DYNAMIC_INPUT(x, TensorType({DT_FLOAT}))
//     .OUTPUT(y, TensorType({DT_FLOAT}))
//     .OP_END_FACTORY_REG(MyDynamicOp);
} // namespace ge

#endif // CUSTOM_OP_PROTO_REG_H
