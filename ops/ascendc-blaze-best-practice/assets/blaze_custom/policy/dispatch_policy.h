/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef BLAZE_CUSTOM_DISPATCH_POLICY_H
#define BLAZE_CUSTOM_DISPATCH_POLICY_H

#include <cstdint>

namespace Blaze {
namespace Gemm {

struct KernelMmadDualBranchGlu {};

template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, bool RefineNearZeroFp16_ = false>
struct MatmulDualBranchGlu {
    using ScheduleType = KernelMmadDualBranchGlu;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
    static constexpr bool REFINE_NEAR_ZERO_FP16 = RefineNearZeroFp16_;
};

} // namespace Gemm
} // namespace Blaze

#endif // BLAZE_CUSTOM_DISPATCH_POLICY_H
