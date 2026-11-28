/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SAMPLE_COMMON_H__
#define __SAMPLE_COMMON_H__

#include <cstdint>
#include <string>
#include <vector>
#include "graph/operator.h"
#include "graph/utils/type_utils.h"

namespace domi {

template <typename T>
ge::Tensor CreateScalar(T val, ge::DataType dtype, ge::Format format = ge::FORMAT_ND)
{
    std::vector<int64_t> dims_scalar = {};
    ge::Shape shape(dims_scalar);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(&val), sizeof(T));
    return tensor;
}

inline ge::Tensor CreateConstInt64(const std::vector<int64_t>& vals)
{
    ge::Shape shape({static_cast<int64_t>(vals.size())});
    ge::TensorDesc desc(shape, ge::FORMAT_ND, ge::DT_INT64);
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(vals.data()), vals.size() * sizeof(int64_t));
}

inline void UpdateMetaByNode(ge::Operator& op_dest, const ge::Operator& op_src,
                             const std::string& original_type)
{
    std::vector<ge::AscendString> input_strings;
    op_src.GetAttr("input", input_strings);
    std::vector<ge::AscendString> output_strings;
    op_src.GetAttr("output", output_strings);
    op_dest.DynamicInputRegister("x", static_cast<int>(input_strings.size()));
    op_dest.DynamicOutputRegister("y", static_cast<int>(output_strings.size()));
    op_dest.SetAttr("name", op_src.GetName());
    op_dest.SetAttr("original_type", original_type);
}

inline ge::Status GetOriNameFromOperator(const ge::Operator& op, std::string& ori_name)
{
    if (op.GetAttr("name", ori_name) != ge::SUCCESS) {
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

} // namespace domi

#endif // __SAMPLE_COMMON_H__
