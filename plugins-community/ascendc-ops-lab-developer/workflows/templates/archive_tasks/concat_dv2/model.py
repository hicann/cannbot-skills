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


SUPPORTED_DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
)

INPUT_CASES = [
    {
        "name": "concat_dim0_single_input_fp16_2d",
        "shapes": [(8, 64)],
        "dtype": torch.float16,
    },
    {
        "name": "concat_dim0_two_input_fp32_2d",
        "shapes": [(3, 128), (5, 128)],
        "dtype": torch.float32,
    },
    {
        "name": "concat_dim0_three_input_bf16_3d",
        "shapes": [(2, 4, 32), (1, 4, 32), (3, 4, 32)],
        "dtype": torch.bfloat16,
    },
    {
        "name": "concat_dim0_two_input_fp16_4d",
        "shapes": [(2, 3, 8, 16), (1, 3, 8, 16)],
        "dtype": torch.float16,
    },
    {
        "name": "concat_dim0_four_input_fp32_2d_odd_width",
        "shapes": [(7, 33), (1, 33), (9, 33), (2, 33)],
        "dtype": torch.float32,
    },
    {
        "name": "concat_dim0_three_input_fp16_1d",
        "shapes": [(5,), (3,), (7,)],
        "dtype": torch.float16,
    },
    {
        "name": "concat_dim0_two_input_bf16_3d_large_tail",
        "shapes": [(16, 8, 64), (5, 8, 64)],
        "dtype": torch.bfloat16,
    },
    {
        "name": "concat_dim0_four_input_fp32_4d",
        "shapes": [(1, 2, 4, 8), (2, 2, 4, 8), (1, 2, 4, 8), (3, 2, 4, 8)],
        "dtype": torch.float32,
    },
]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, concat_dim: int = 0) -> torch.Tensor:
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValueError("inputs must be a non-empty list or tuple of tensors")
        if concat_dim != 0:
            raise ValueError(f"concat_dv2 TileLang reference only supports concat_dim=0, got {concat_dim}")

        reference = inputs[0]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("all inputs must be torch.Tensor")
        if reference.ndim < 1:
            raise ValueError("each input tensor must have rank >= 1")
        if reference.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {reference.dtype}")

        tail_shape = tuple(int(dim) for dim in reference.shape[1:])
        for tensor in inputs[1:]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("all inputs must be torch.Tensor")
            if tensor.ndim != reference.ndim:
                raise ValueError("all inputs must have the same rank")
            if tensor.dtype != reference.dtype:
                raise ValueError("all inputs must have the same dtype")
            if tuple(int(dim) for dim in tensor.shape[1:]) != tail_shape:
                raise ValueError("all non-concat dimensions must match")

        return torch.cat(tuple(inputs), dim=0)


def _make_tensor(shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randn(*shape, dtype=torch.float32, generator=generator)
    return tensor.to(dtype)


def get_input_groups():
    input_groups = []
    for idx, case in enumerate(INPUT_CASES):
        inputs = []
        for tensor_idx, shape in enumerate(case["shapes"]):
            seed = 2026 + idx * 17 + tensor_idx
            inputs.append(_make_tensor(shape, case["dtype"], seed))
        input_groups.append([inputs, 0])
    return input_groups


def get_init_inputs():
    return []
