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

# Template for: avg_pool3_d
import torch
# Template for: avg_pool3_d
import torch.nn as nn
# Template for: avg_pool3_d
import torch.nn.functional as F


SCENARIOS = [
    {
        "name": "avg_pool3_d_ncdhw_big_kernel",
        "shape": (2, 16, 32, 32, 32),
        "kernel_size": (8, 8, 8),
        "stride": (4, 4, 4),
        "padding": (2, 2, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
    },
    {
        "name": "avg_pool3_d_ncdhw_normal",
        "shape": (4, 16, 8, 16, 16),
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
    },
    {
        "name": "avg_pool3_d_ncdhw_reduce_d",
        "shape": (2, 32, 16, 32, 32),
        "kernel_size": (3, 1, 1),
        "stride": (2, 1, 1),
        "padding": (1, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
    },
    {
        "name": "avg_pool3_d_ndhwc_multi_w",
        "shape": (2, 64, 16, 16, 16),
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
        "tilelang_mode": "multi_w",
        "multi_w_window_w_num": 2,
    },
    {
        "name": "avg_pool3_d_ndhwc_split_c",
        "shape": (2, 512, 8, 8, 8),
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
        "tilelang_mode": "split_c",
    },
    {
        "name": "avg_pool3_d_ndhwc_split_w",
        "shape": (2, 32, 8, 16, 64),
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 4),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
        "tilelang_mode": "split_w",
    },
]

SCENARIO_BY_SHAPE = {scenario["shape"]: scenario for scenario in SCENARIOS}


def resolve_scenario(x):
    """Return the scenario dict matching *x*'s shape, or raise ValueError."""
    if x.ndim != 5:
        raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape={tuple(x.shape)}")
    shape = tuple(int(dim) for dim in x.shape)
    scenario = SCENARIO_BY_SHAPE.get(shape)
    if scenario is None:
        supported = ", ".join(str(case["shape"]) for case in SCENARIOS)
        raise ValueError(f"Unsupported avg_pool3_d input shape {shape}. Supported shapes: {supported}")
    return scenario


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scenario = resolve_scenario(x)
        divisor_override = scenario["divisor_override"] or None
        return F.avg_pool3d(
            x,
            kernel_size=scenario["kernel_size"],
            stride=scenario["stride"],
            padding=scenario["padding"],
            ceil_mode=scenario["ceil_mode"],
            count_include_pad=scenario["count_include_pad"],
            divisor_override=divisor_override,
        )


def get_input_groups():
    input_groups = []
    for scenario in SCENARIOS:
        n, c, d, h, w = scenario["shape"]
        x = torch.rand(n, c, d, h, w, dtype=torch.float32)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
