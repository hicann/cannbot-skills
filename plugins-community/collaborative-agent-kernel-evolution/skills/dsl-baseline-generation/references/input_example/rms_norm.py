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


def module_fn(x: torch.Tensor, num_features: int, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS Normalization to the input tensor using functional implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        num_features (int): Number of features in the input tensor.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
    """
    # Calculate the RMS along the feature dimension
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)

    # Normalize the input by dividing by the RMS
    return x / rms


class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        xt = x.transpose(1, -1).contiguous()
        res = fn(xt, self.num_features, self.eps)
        return res.transpose(-1, 1)


BATCH_SIZE = 16
FEATURES = 64
DIM1 = 256
DIM2 = 256


def get_inputs():
    x = torch.randn(BATCH_SIZE, FEATURES, DIM1, DIM2)
    return [x]


def get_init_inputs():
    return [FEATURES]
