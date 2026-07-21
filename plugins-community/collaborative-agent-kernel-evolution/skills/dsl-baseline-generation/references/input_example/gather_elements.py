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


def module_fn(input_tensor: torch.Tensor, index: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Applies gather to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features).

    Returns:
        torch.Tensor: Output tensor with gather applied, same shape as index tensor.
    """
    return torch.gather(input_tensor, dim=dim, index=index)


class Model(nn.Module):
    """
    Simple model that performs a gather_elements operation.
    """

    def __init__(self, dim: int = -1):
        """
        Initializes the gather_elements model.

        Args:
            dim (int, optional): The axis along which to gather. Defaults to 1.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor, index: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Gathers values from input tensor along the specified dimension using indices.

        Args:
            input_tensor (torch.Tensor): The source tensor to gather from.
            index (torch.Tensor): The indices tensor specifying which elements to gather.

        Returns:
            torch.Tensor: Output tensor with gathered elements, same shape as index tensor.
        """
        return fn(input_tensor, index=index, dim=self.dim)


BATCH_SIZE = 128
INPUT_SIZE = 8192
GATHER_SIZE = 32


def get_inputs():
    # Create input tensor
    input_tensor = torch.randn(BATCH_SIZE, INPUT_SIZE)

    # Create index tensor with valid indices for gathering along dim=1
    index_tensor = torch.randint(0, INPUT_SIZE, (BATCH_SIZE, GATHER_SIZE)).to(torch.int32)

    return [input_tensor, index_tensor]


def get_init_inputs():
    return [-1]  # Provide dim value for initialization
