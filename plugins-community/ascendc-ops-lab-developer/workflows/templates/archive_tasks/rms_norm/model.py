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

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor):
        x_fp32 = x.to(torch.float32)
        mean_sq = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        out = x_fp32 * inv_rms * gamma.to(torch.float32)
        return out.to(x.dtype), inv_rms.to(x.dtype)


RMS_NORM_CASES = [
    {"shape": [1024, 1024], "dtype": torch.float32,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2026, "gamma_seed": 3026},
    {"shape": [128, 4096], "dtype": torch.float16,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2027, "gamma_seed": 3027},
    {"shape": [64, 3584], "dtype": torch.bfloat16,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2028, "gamma_seed": 3028},
    {"shape": [16, 16384], "dtype": torch.float16,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2029, "gamma_seed": 3029},
    {"shape": [33, 16384], "dtype": torch.float32,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2030, "gamma_seed": 3030},
    {"shape": [2, 3, 12288], "dtype": torch.bfloat16,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2031, "gamma_seed": 3031},
    {"shape": [1, 32, 1024], "dtype": torch.float32,
     "x_mode": "randn", "gamma_mode": "ones",
     "x_seed": 2032, "gamma_seed": 3032},
    {"shape": [2, 16, 2048], "dtype": torch.float16,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2033, "gamma_seed": 3033},
    {"shape": [2, 8, 16, 256], "dtype": torch.float32,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2034, "gamma_seed": 3034},
    {"shape": [17, 1536], "dtype": torch.float32,
     "x_mode": "randn", "gamma_mode": "randn",
     "x_seed": 2035, "gamma_seed": 3035},
    {"shape": [4, 513], "dtype": torch.float32,
     "x_mode": "small", "gamma_mode": "randn",
     "x_seed": 2036, "gamma_seed": 3036},
    {"shape": [8, 1024], "dtype": torch.float32,
     "x_mode": "zeros", "gamma_mode": "randn",
     "x_seed": 2037, "gamma_seed": 3037},
]


def _make_tensor(shape, dtype, mode, seed):
    generator = torch.Generator().manual_seed(seed)
    if mode == "randn":
        return torch.randn(*shape, dtype=dtype, generator=generator)
    if mode == "small":
        return torch.randn(*shape, dtype=dtype, generator=generator) * 1e-4
    if mode == "zeros":
        return torch.zeros(*shape, dtype=dtype)
    if mode == "ones":
        return torch.ones(*shape, dtype=dtype)
    raise ValueError(f"Unsupported tensor mode: {mode}")


def get_input_groups():
    input_groups = []
    for idx, case in enumerate(RMS_NORM_CASES):
        shape = case["shape"]
        dtype = case["dtype"]
        hidden_size = shape[-1]
        x_seed = case.get("x_seed", 2026 + idx)
        gamma_seed = case.get("gamma_seed", 3026 + idx)
        x = _make_tensor(shape, dtype, case["x_mode"], seed=x_seed)
        gamma = _make_tensor([hidden_size], dtype, case["gamma_mode"], seed=gamma_seed)
        input_groups.append([x, gamma])
    return input_groups


def get_init_inputs():
    return []
