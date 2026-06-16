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

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.kv_sort import kv_sort as tl_kv_sort


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, keys: torch.Tensor, values: torch.Tensor):
        if keys.ndim != 1 or values.ndim != 1:
            raise ValueError(f"Expected 1D inputs, got keys:{keys.ndim}D values:{values.ndim}D")
        if keys.shape[0] != values.shape[0]:
            raise ValueError(f"Length mismatch: keys={keys.shape[0]} vs values={values.shape[0]}")
        if keys.dtype != torch.int32:
            raise ValueError(f"keys must be int32, got {keys.dtype}")
        if values.dtype != torch.int32:
            raise ValueError(f"values must be int32, got {values.dtype}")

        kernel = self._build_kernel(keys)
        sorted_keys, sorted_values = kernel(keys, values)
        return sorted_keys, sorted_values

    def _build_kernel(self, keys: torch.Tensor):
        n = keys.shape[0]
        return tl_kv_sort(n, dtype="int32")
