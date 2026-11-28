/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <iostream>
#include <typeinfo>
#include "register/op_def_registry.h"
#include "register/register.h"
#include "graph/operator.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "built-in/op_graph/inc/array_ops.h"
#include "built-in/op_graph/inc/math_ops.h"
#include "built-in/op_graph/inc/elewise_calculation_ops.h"
#include "built-in/op_graph/inc/selection_ops.h"
#include "built-in/op_graph/inc/nn_norm_ops.h"
#include "built-in/op_graph/inc/nonlinear_fuc_ops.h"
#include "built-in/op_graph/inc/fusion_ops.h"
#include "built-in/op_graph/inc/transformation_ops.h"
#include "built-in/op_graph/inc/matrix_calculation_ops.h"
#include "../op_graph/sample_op_proto.h"

using namespace ge;

namespace ge {
static graphStatus InferDataType4SampleOp(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SampleOp : public OpDef {
public:
    explicit SampleOp(const char* name) : OpDef(name)
    {
        // TODO(<op>): 按实际语义声明输入输出；DataType/Format/UnknownShapeFormat 列表长度必须一致
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // TODO(<op>): 可选属性
        // this->Attr("<attr_name>").AttrType(OPTIONAL).Bool(true);

        this->SetInferDataType(ge::InferDataType4SampleOp);

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "sample_op");
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
        this->AICore().AddConfig("ascend910_93", aicoreConfig910B);
    }
};
OP_ADD(SampleOp);
} // namespace ops
