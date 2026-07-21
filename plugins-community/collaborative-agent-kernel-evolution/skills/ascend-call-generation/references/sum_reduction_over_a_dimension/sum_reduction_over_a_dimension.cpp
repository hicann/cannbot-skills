/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor sum_reduction_over_a_dimension_custom_impl_npu(const at::Tensor& input, int64_t dim) {
    int64_t ndim = input.dim();
    int64_t adjusted_dim = dim;
    if (dim < 0) {
        adjusted_dim = dim + ndim;
    }

    auto input_shape = input.sizes().vec();
    auto output_shape = input_shape;
    output_shape[adjusted_dim] = 1;
    
    at::Tensor result = at::empty(output_shape, input.options());
    EXEC_NPU_CMD(aclnnSumReductionOverADimensionCustom, input, adjusted_dim, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_custom_impl_npu, "Sum Reduction Over A Dimension");
}