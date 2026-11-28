/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __PASS_UTILS_H__
#define __PASS_UTILS_H__

#include <initializer_list>
#include <string>
#include <vector>
#include "register/register_custom_pass.h"

namespace pass_utils {

inline bool IsNodeType(const ge::GNode& node, const char* type)
{
    ge::AscendString node_type;
    if (node.GetType(node_type) != ge::GRAPH_SUCCESS) {
        return false;
    }
    return node_type == type;
}

inline bool IsAnyNodeType(const ge::GNode& node, std::initializer_list<const char*> types)
{
    for (const char* type : types) {
        if (IsNodeType(node, type)) {
            return true;
        }
    }
    return false;
}

inline std::string GetNodeName(const ge::GNode& node)
{
    ge::AscendString name;
    node.GetName(name);
    return std::string(name.GetString());
}

inline void RemoveAllInputEdges(ge::GraphPtr& graph, ge::GNode& node)
{
    for (size_t i = 0; i < node.GetInputsSize(); ++i) {
        auto [in_node, in_idx] = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (in_node != nullptr) {
            graph->RemoveEdge(*in_node, in_idx, node, static_cast<int32_t>(i));
        }
    }
}

inline void RemoveAllOutputEdges(ge::GraphPtr& graph, ge::GNode& node)
{
    for (uint32_t out_idx = 0; out_idx < node.GetOutputsSize(); ++out_idx) {
        auto outs = node.GetOutDataNodesAndPortIndexs(out_idx);
        for (auto& [out_node, out_idx_of_consumer] : outs) {
            if (out_node != nullptr) {
                graph->RemoveEdge(node, static_cast<int32_t>(out_idx), *out_node, out_idx_of_consumer);
            }
        }
    }
}

} // namespace pass_utils

#endif // __PASS_UTILS_H__
