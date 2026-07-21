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


def module_fn(x: torch.Tensor, perm) -> torch.Tensor:
    """
    Permute (transpose) the dimensions of the input tensor.

    Pure layout transformation, NO arithmetic:
        out_shape[j] = in_shape[perm[j]]
        y[out_coords] reads x at input_coords where
            input_coords[perm[j]] = out_coords[j]
        Equivalently, input offset = sum_j out_coords[j] * in_stride[perm[j]].

    Args:
        x (torch.Tensor): Input tensor, rank 2..8, any dtype.
        perm (List[int]): Permutation of the axes (e.g. [1, 0] for a 2D
            transpose, [0, 2, 1, 3] for swapping the middle two axes).

    Returns:
        torch.Tensor: Permuted tensor, contiguous, with shape out_shape.
    """
    # `.contiguous()` is important: torch.permute returns a *view* with
    # non-contiguous strides; the Ascend kernel produces a materialized
    # contiguous output, so the reference must materialize too.
    return torch.permute(x, perm).contiguous()


class Model(nn.Module):
    """
    Simple model that performs an N-D transpose / permute operation.
    """

    def __init__(self, perm):
        """
        Args:
            perm (List[int]): The axis permutation applied in forward().
        """
        super(Model, self).__init__()
        self.perm = perm

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, perm=self.perm)


# Representative dominant case: a 2D transpose perm=[1, 0].
# (In practice the same kernel must also cover 3D-5D and arbitrary perm.)
DIM0 = 512
DIM1 = 2049          # deliberately non-aligned last dim
PERM = [1, 0]


def get_inputs():
    x = torch.randn(DIM0, DIM1)
    return [x]


def get_init_inputs():
    return [PERM]      # PERM is a construction-time attribute
