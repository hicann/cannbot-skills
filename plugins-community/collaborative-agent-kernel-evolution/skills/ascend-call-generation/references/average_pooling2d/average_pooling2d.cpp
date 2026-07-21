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

at::Tensor average_pooling_2d_custom_impl_npu(const at::Tensor& input, int64_t kernel_size) {
    // bf16 variant: input/output tensors are bfloat16
    // The kernel handles bf16↔f32 casting internally
    int64_t actual_stride = kernel_size ;
    
    // --------------------------------------------------
    // Input tensor layout (NHWC):
    //   dim 0: batch (N)
    //   dim 1: height (H)
    //   dim 2: width  (W)
    //   dim 3: channels (C)  <-- channel axis already moved here
    // --------------------------------------------------
    int64_t batch = input.size(0);
    int64_t height = input.size(1);
    int64_t width = input.size(2);
    int64_t channels = input.size(3);
    
    int64_t out_height = height / actual_stride;
    int64_t out_width = width / actual_stride;
    
    at::Tensor result = at::empty({batch, out_height, out_width, channels}, input.options());  // preserves bf16 dtype
    EXEC_NPU_CMD(aclnnAveragePooling2dCustom, input, kernel_size, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling_2d_custom", &average_pooling_2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("average_pooling_2d_custom", &average_pooling_2d_custom_impl_npu, "2D Average Pooling");
}