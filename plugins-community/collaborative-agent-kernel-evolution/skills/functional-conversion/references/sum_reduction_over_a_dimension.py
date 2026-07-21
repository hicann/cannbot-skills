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


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies sum reduction over the specified dimension using functional implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim, ...).
        dim (int): Dimension to reduce over (should be -1 for the transposed tensor).

    Returns:
        torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
    """
    return torch.sum(x, dim=dim, keepdim=True)


class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Transpose so that the reduction dimension becomes the last axis
        xt = x.transpose(self.dim, -1).contiguous()
        # Apply the reduction on the last axis
        res = fn(xt, -1)
        # Transpose back to the original layout
        return res.transpose(-1, self.dim)


BATCH_SIZE = 16
DIM1 = 256
DIM2 = 256
REDUCE_DIM = 1


def get_inputs():
    x = torch.rand(BATCH_SIZE, DIM1, DIM2)
    return [x]


def get_init_inputs():
    return [REDUCE_DIM]
