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


def module_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return custom_ops_lib.matmul_custom(x, weight)


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the model with a weight matrix.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
        """
        super(ModelNew, self).__init__()
        # IMPORTANT: Must match Model.__init__ exactly — same seed and same tensor creation
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies matrix multiplication: output = x @ weight^T

        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch, out_features].
        """
        return fn(x, self.weight)
