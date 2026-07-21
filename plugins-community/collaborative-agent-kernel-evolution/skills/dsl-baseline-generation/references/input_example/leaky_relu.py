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


def module_fn(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Applies LeakyReLU activation to the input tensor using functional implementation.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        negative_slope (float): The negative slope of the activation function. Defaults to 0.01.

    Returns:
        torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
    """
    return F.leaky_relu(x, negative_slope=negative_slope)


class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return fn(x, self.negative_slope)


BATCH_SIZE = 16
DIM = 16384


def get_inputs():
    x = torch.randn(BATCH_SIZE, DIM)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
