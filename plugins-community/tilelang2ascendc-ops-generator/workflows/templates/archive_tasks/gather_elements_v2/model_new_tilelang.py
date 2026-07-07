#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Template for: gather_elements_v2
import torch
import torch.nn as nn

from gather_elements_v2.design.tile_level.gather_elements_v2 import gather_elements_v2 as tl_gather_elements_v2
from gather_elements_v2.model import (
    compute_row_map,
    _prepare_gather,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._kernel_cache = {}

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int):
        r = _prepare_gather(x, index, dim)
        if r is None:
            return x.new_empty(index.shape)

        kernel = self._build_kernel(
            r.mode,
            r.idx_rows,
            r.x_rows,
            r.padding.x_gather_dim,
            r.padding.idx_gather_dim,
            r.padding.x_stride,
            r.padding.y_stride,
            x.dtype,
        )

        if r.mode == "last_dim":
            y_padded = kernel(r.padding.x_padded, r.padding.index_padded)
        else:
            row_map = compute_row_map(r.x_prefix_shape, r.idx_prefix_shape, index.device)
            y_padded = kernel(r.padding.x_padded, r.padding.index_padded, row_map)

        y_flat = y_padded[:, :r.padding.idx_gather_dim]
        y_perm = y_flat.reshape(r.index_perm.shape)
        return y_perm.permute(r.inverse_perm).contiguous()

    def _build_kernel(self, *args):
        mode, idx_rows, x_rows, x_gather_dim, idx_gather_dim, x_stride, y_stride, dtype = args
        dtype_name = str(dtype).split(".")[-1]
        cache_key = (mode, idx_rows, x_rows, x_gather_dim, idx_gather_dim, x_stride, y_stride, dtype_name)
        kernel = self._kernel_cache.get(cache_key)
        if kernel is None:
            kernel = tl_gather_elements_v2(
                idx_rows,
                x_rows,
                x_gather_dim,
                idx_gather_dim,
                x_stride,
                y_stride,
                mode=mode,
                dtype=dtype_name,
            )
            self._kernel_cache[cache_key] = kernel
        return kernel

