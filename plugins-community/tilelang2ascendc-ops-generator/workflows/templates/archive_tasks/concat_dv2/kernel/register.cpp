/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("concat_dim0_1(Tensor x0) -> Tensor");
    m.def("concat_dim0_2(Tensor x0, Tensor x1) -> Tensor");
    m.def("concat_dim0_3(Tensor x0, Tensor x1, Tensor x2) -> Tensor");
    m.def("concat_dim0_4(Tensor x0, Tensor x1, Tensor x2, Tensor x3) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("concat_dim0_1", TORCH_FN(ascend_kernel::concat_dim0_1));
    m.impl("concat_dim0_2", TORCH_FN(ascend_kernel::concat_dim0_2));
    m.impl("concat_dim0_3", TORCH_FN(ascend_kernel::concat_dim0_3));
    m.impl("concat_dim0_4", TORCH_FN(ascend_kernel::concat_dim0_4));
}

} // namespace
