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
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Functional implementation of LayerNorm (bfloat16 variant).
    Input x, weight, bias are bfloat16. F.layer_norm handles bf16 natively.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        weight (torch.Tensor): Weight tensor of shape (normalized_shape).
        bias (torch.Tensor): Bias tensor of shape (normalized_shape).
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    # Get the normalized shape from the weight tensor
    normalized_shape = tuple(x.shape[-len(weight.shape):])
    return F.layer_norm(
        x, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """

    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        # bf16: weights and bias must be bfloat16 to match input dtype
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias)


BATCH_SIZE = 16
FEATURES = 64
DIM1 = 256
DIM2 = 256


def get_inputs():
    x = torch.randn(BATCH_SIZE, FEATURES, DIM1, DIM2, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [(FEATURES, DIM1, DIM2)]
