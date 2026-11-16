/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 同shape推导骨架：合并到最终实现类。数据/值依赖另按真实契约实现。
// 接口和选择条件见 references/interfaces.md、scenarios.md。
#include "graph/custom_op.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/types.h"

class MyShapeInferOp : public ge::ShapeInferOp {
public:
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override
    {
        const gert::Shape* x_shape = ctx->GetInputShape(0);
        if (x_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape* out_shape = ctx->GetOutputShape(0);
        if (out_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 形态 1：输出与输入同 shape
        *out_shape = *x_shape;

        // 数据依赖示例的 -1 表示unknown；实际分配上界与shape回填需在Execute实现
        // out_shape->SetDimNum(2);
        // out_shape->SetDim(0, x_shape->GetDim(0));
        // out_shape->SetDim(1, -1); // unknown 维：Execute 时由 kernel 写 shape buffer 回填
        // 秩本身未知时用 -2（value_dependent_shape_custom 的 kUnknownRankDim=-2）；
        // 需要按维追加时 AppendDim 比 SetDimNum+SetDim 更常用（两者都合法）。

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override
    {
        const ge::DataType x_dtype = ctx->GetInputDataType(0);
        return ctx->SetOutputDataType(0, x_dtype);
    }
};

// 与 EagerExecuteOp 同属一个算子时，只注册一个类即可（多重继承）：
//   class MyShapeInferOp : public ge::EagerExecuteOp, public ge::ShapeInferOp { ... };
//   REG_AUTO_MAPPING_OP(MyShapeInferOp);
//
// 值依赖 shape 场景额外补一行（借用 v1 注册声明数据依赖）：
//   IMPL_OP(MyShapeInferOp).InputsDataDependency({kConcatDimIrIndex});
REG_AUTO_MAPPING_OP(MyShapeInferOp);
