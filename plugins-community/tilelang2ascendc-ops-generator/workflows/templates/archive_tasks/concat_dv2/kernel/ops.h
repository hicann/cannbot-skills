/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor concat_dim0_1(const at::Tensor &x0);
at::Tensor concat_dim0_2(const at::Tensor &x0, const at::Tensor &x1);
at::Tensor concat_dim0_3(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2);
at::Tensor concat_dim0_4(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &x3);

} // namespace ascend_kernel

#endif // OPS_H
