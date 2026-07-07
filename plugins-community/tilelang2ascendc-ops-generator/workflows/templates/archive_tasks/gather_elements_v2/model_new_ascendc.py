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

import sys
from pathlib import Path

# Template for: gather_elements_v2
import torch
# Template for: gather_elements_v2
import torch.nn as nn

from gather_elements_v2.model import (
    compute_row_map,
    _prepare_gather,
)

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _current_task_ext as _ext  


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _mode_str_to_int(mode: str) -> int:
        return {"last_dim": 0, "transpose": 1, "indexed": 2}[mode]

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int):
        r = _prepare_gather(x, index, dim)
        if r is None:
            return x.new_empty(index.shape)

        mode = self._mode_str_to_int(r.mode)
        if mode == 0:
            row_map = torch.zeros((r.idx_rows,), dtype=torch.int32, device=index.device)
        else:
            row_map = compute_row_map(r.x_prefix_shape, r.idx_prefix_shape, index.device)

        y_padded = _ext.run_gather_elements_v2(
            r.padding.x_padded, r.padding.index_padded, row_map, r.padding.idx_gather_dim, mode)
        y_flat = y_padded[:, :r.padding.idx_gather_dim]
        y_perm = y_flat.reshape(r.index_perm.shape)
        return y_perm.permute(r.inverse_perm).contiguous()
