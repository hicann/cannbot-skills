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


class Model(nn.Module):
    """
    Simple model that performs a HardTanh activation (bfloat16 variant).
    F.hardtanh supports bfloat16 natively on Ascend.
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """
        Initializes the HardTanh model.

        Args:
            min_val (float, optional): Lower bound of the clamp range. Defaults to -1.0.
            max_val (float, optional): Upper bound of the clamp range. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


BATCH_SIZE = 16
DIM = 16384


def get_inputs():
    x = torch.rand(BATCH_SIZE, DIM, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [-1.0, 1.0]  # Provide min_val and max_val for initialization
