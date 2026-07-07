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

from design.tile_level.avg_pool3_d import avg_pool3_d as tl_avg_pool3_d
from model import SCENARIO_BY_SHAPE


SPLIT_MODE_AUTO = 0
SPLIT_MODE_C = 1
SPLIT_MODE_W = 2
SPLIT_MODE_MULTI_W = 3


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input, got {x.ndim}D")
        if x.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Unsupported dtype {x.dtype}, expected float16 or float32")

        n, c, d, h, w = x.shape
        kernel, (_, _, o_d, o_h, o_w) = self._build_kernel(x)

        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_nhwc.reshape(n * d * h * w, c)
        y_flat = kernel(x_flat)
        y = y_flat.reshape(n, o_d, o_h, o_w, c).permute(0, 4, 1, 2, 3).contiguous()
        return y

    def _resolve_scenario(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape={tuple(x.shape)}")

        shape = tuple(int(dim) for dim in x.shape)
        scenario = SCENARIO_BY_SHAPE.get(shape)
        if scenario is None:
            supported = ", ".join(str(case_shape) for case_shape in SCENARIO_BY_SHAPE.keys())
            raise ValueError(f"Unsupported avg_pool3_d input shape {shape}. Supported shapes: {supported}")
        return scenario

    def _compute_output_size(self, *args):
        d, h, w, kernel_size, stride, padding, ceil_mode = args
        k_d, k_h, k_w = kernel_size
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding

        if not ceil_mode:
            o_d = (d + 2 * p_d - k_d) // s_d + 1
            o_h = (h + 2 * p_h - k_h) // s_h + 1
            o_w = (w + 2 * p_w - k_w) // s_w + 1
        else:
            o_d = (d + 2 * p_d - k_d + s_d - 1) // s_d + 1
            o_h = (h + 2 * p_h - k_h + s_h - 1) // s_h + 1
            o_w = (w + 2 * p_w - k_w + s_w - 1) // s_w + 1
            if (o_d - 1) * s_d >= d + p_d:
                o_d -= 1
            if (o_h - 1) * s_h >= h + p_h:
                o_h -= 1
            if (o_w - 1) * s_w >= w + p_w:
                o_w -= 1

        return o_d, o_h, o_w

    def _choose_block_c(self, c: int) -> int:
        for candidate in (256, 128, 64, 32):
            if candidate <= c and c % candidate == 0:
                return candidate
        return 0

    def _resolve_tilelang_mode(self, scenario: dict) -> int:
        mode = scenario.get("tilelang_mode", "auto")
        if mode == "split_c":
            return SPLIT_MODE_C
        if mode == "split_w":
            return SPLIT_MODE_W
        if mode == "multi_w":
            return SPLIT_MODE_MULTI_W
        return SPLIT_MODE_AUTO

    def _resolve_split_w_tile_kw(self, scenario: dict, k_w: int) -> int:
        tile_kw = int(scenario.get("split_w_tile_kw", 0))
        if tile_kw <= 0:
            return 0
        return min(tile_kw, k_w)

    def _resolve_multi_w_window_w_num(self, scenario: dict, o_w: int) -> int:
        window = int(scenario.get("multi_w_window_w_num", 1))
        if window <= 1:
            return 1
        return min(window, o_w)

    def _build_kernel(self, x: torch.Tensor):
        n, c, d, h, w = x.shape
        scenario = self._resolve_scenario(x)

        kernel_size = scenario["kernel_size"]
        stride = scenario["stride"]
        padding = scenario["padding"]
        ceil_mode = scenario["ceil_mode"]
        count_include_pad = scenario["count_include_pad"]
        divisor_override = scenario["divisor_override"]

        o_d, o_h, o_w = self._compute_output_size(d, h, w, kernel_size, stride, padding, ceil_mode)
        k_d, k_h, k_w = kernel_size
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding
        dtype = str(x.dtype).split(".")[-1]

        split_mode = self._resolve_tilelang_mode(scenario)
        block_c = 0
        if split_mode == SPLIT_MODE_C:
            block_c = self._choose_block_c(c)
            if block_c == 0:
                raise ValueError(f"split_c scenario requires divisible channel tiles, got C={c}")

        split_w_tile_kw = self._resolve_split_w_tile_kw(scenario, k_w)
        multi_w_window_w_num = self._resolve_multi_w_window_w_num(scenario, o_w)

        kernel = tl_avg_pool3_d(
            n,
            c,
            d,
            h,
            w,
            o_d,
            o_h,
            o_w,
            k_d,
            k_h,
            k_w,
            s_d,
            s_h,
            s_w,
            p_d,
            p_h,
            p_w,
            1 if count_include_pad else 0,
            divisor_override,
            block_c,
            dtype=dtype,
            split_mode=split_mode,
            split_w_tile_kw=split_w_tile_kw,
            multi_w_window_w_num=multi_w_window_w_num,
        )
        return kernel, (n, c, o_d, o_h, o_w)
