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
    Simple model that performs a linear transformation followed by softplus activation.
    Demonstrates proper weight initialization for evaluation compatibility.
    Formula: output = softplus(input @ weight^T)
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the model with a weight matrix.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
        """
        super(Model, self).__init__()
        # IMPORTANT: Use a fixed seed before random parameter initialization.
        # This ensures Model and ModelNew produce identical weights during evaluation,
        # since the evaluation framework creates them sequentially.
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies linear transformation and softplus activation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """
        logits = F.linear(x, self.weight)
        return F.softplus(logits)


BATCH_SIZE = 16
IN_FEATURES = 512
OUT_FEATURES = 8


def get_inputs():
    x = torch.rand(BATCH_SIZE, IN_FEATURES)
    return [x]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES]
