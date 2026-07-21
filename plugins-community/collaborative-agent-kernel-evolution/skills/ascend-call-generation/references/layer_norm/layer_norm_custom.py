# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch_npu
import custom_ops_lib


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return custom_ops_lib.layer_norm_custom(x, weight, bias, eps)


class ModelNew(nn.Module):

    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer parameters.

        Args:
        normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # bf16 variant: weights and biases must match input dtype
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        fn: Function to apply (defaults to module_fn)

        Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias)
