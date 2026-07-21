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


def module_fn(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Functional implementation of Average Pooling 2D (bfloat16 variant).
    F.avg_pool2d does not support bf16, so we cast to float32 for computation.
    """
    orig_dtype = x.dtype
    x = x.permute(0, 3, 1, 2).contiguous()
    # F.avg_pool2d requires float32; cast bf16 → f32 → compute → cast back
    x = F.avg_pool2d(x.float(), kernel_size=kernel_size)
    x = x.to(orig_dtype)
    x = x.permute(0, 2, 3, 1).contiguous()
    return x


class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        x = x.permute(0, 2, 3, 1).contiguous()   # NCHW → NHWC
        y = fn(x, self.kernel_size)
        return y.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW


BATCH_SIZE = 16
CHANNELS = 64
HEIGHT = 256
WIDTH = 256
KERNEL_SIZE = 3


def get_inputs():
    x = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [KERNEL_SIZE]
