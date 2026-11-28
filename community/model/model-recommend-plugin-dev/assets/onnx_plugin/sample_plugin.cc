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
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "register/op_def_registry.h"
#include "register/register.h"
#include "graph/operator.h"
#include "graph/utils/type_utils.h"
#include "built-in/op_graph/inc/array_ops.h"
#include "built-in/op_graph/inc/math_ops.h"
#include "built-in/op_graph/inc/elewise_calculation_ops.h"
#include "built-in/op_graph/inc/selection_ops.h"
#include "built-in/op_graph/inc/nn_norm_ops.h"
#include "built-in/op_graph/inc/nonlinear_fuc_ops.h"
#include "built-in/op_graph/inc/fusion_ops.h"
#include "built-in/op_graph/inc/transformation_ops.h"
#include "built-in/op_graph/inc/matrix_calculation_ops.h"
#include "sample_common.h"

using namespace ge;
using json = nlohmann::json;

namespace domi {

// 完整范例：<OpType>(x, y) 带 float 属性 scale
//   ParseParams  : 从 attribute JSON 取 scale
//   ParseOpToGraph: 展开为 Mul(x, y) -> Add(mul, scale)
// TODO(<domain>_<optype>): 把本范例替换为真实算子的语义映射
static Status ParseParamsSample(const ge::Operator& op_src, ge::Operator& op_dest)
{
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) != ge::GRAPH_SUCCESS) {
        std::cout << "Failed to get attribute JSON from op" << std::endl;
        return FAILED;
    }
    json attrs = json::parse(attrs_string.GetString());

    for (json& attr : attrs["attribute"]) {
        std::string attr_name = attr["name"];
        if (attr_name == "scale") {
            float scale = attr["f"].is_string() ? std::stof(attr["f"].get<std::string>())
                                                : attr["f"].get<float>();
            op_dest.SetAttr("scale", scale);
        }
        // TODO(<domain>_<optype>): 其它属性按类型解析
        //   int:    attr["i"]
        //   string: attr["s"]
        //   ints:   attr["ints"].get<std::vector<int>>()
        //   floats: attr["floats"].get<std::vector<float>>()
    }

    UpdateMetaByNode(op_dest, op_src, "<domain>::<version>::<OpType>");
    return SUCCESS;
}

static Status ParseOpToGraphSample(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (GetOriNameFromOperator(op, ori_name) != SUCCESS) {
        return FAILED;
    }

    float scale = 1.0f;
    op.GetAttr("scale", scale);

    // 每条输入边一个 Data 节点，set_attr_index(i) 对应第 i 条输入边
    auto data_x = ge::op::Data((ori_name + "_x").c_str()).set_attr_index(0);
    auto data_y = ge::op::Data((ori_name + "_y").c_str()).set_attr_index(1);

    // 需要固定值时用 Const
    auto scale_const = ge::op::Const((ori_name + "_scale").c_str())
                           .set_attr_value(CreateScalar<float>(scale, ge::DT_FLOAT));

    // 用 NPU 现有算子拼接目标语义
    auto mul = ge::op::Mul((ori_name + "_mul").c_str())
                   .set_input_x1(data_x)
                   .set_input_x2(data_y);
    auto add = ge::op::Add((ori_name + "_add").c_str())
                   .set_input_x1(mul)
                   .set_input_x2(scale_const);

    // SetInputs 放全部 Data 节点；SetOutputs 输出个数必须与原算子一致
    std::vector<ge::Operator> inputs{data_x, data_y};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs{{add, {0}}};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// 注册：子图展开统一 BUILDIN；直映射现有 GE 算子用 GELOCAL；TVM 不用
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(std::vector<ge::AscendString>{ge::AscendString("<domain>::1::<OpType>"),
                                                ge::AscendString("<domain>::8::<OpType>")
                                                // TODO(<domain>_<optype>): 覆盖该算子全部 opset 版本
                                                })
    .ParseParamsByOperatorFn(ParseParamsSample)
    .ParseOpToGraphFn(ParseOpToGraphSample)
    .ImplyType(ImplyType::BUILDIN);

} // namespace domi
