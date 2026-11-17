// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#ifndef CANNBOT_CCU_PATTERNS_HPP
#define CANNBOT_CCU_PATTERNS_HPP
#include <cstdint>

namespace cannbot_ccu_example {
// Var is the reviewed CCU Variable type. result must have an independent handle.
// factor is a HOST value. Prove stride*factor fits the destination/address range.
// Copy this helper into the algorithm if needed; do not depend on a Skill path.
template <class Var>
void EmitScaledStride(const Var& stride, uint16_t factor, const Var& result)
{
    result = uint64_t{0};
    if (factor == 0) {
        return;
    }
    Var doubled;
    doubled = stride; // assignment emits a value copy; construction would alias.
    for (uint32_t bits = factor; bits != 0; bits >>= 1U) {
        if ((bits & 1U) != 0) {
            result += doubled;
        }
        if (bits > 1) {
            doubled += doubled;
        }
    }
}
} // namespace cannbot_ccu_example
#endif
