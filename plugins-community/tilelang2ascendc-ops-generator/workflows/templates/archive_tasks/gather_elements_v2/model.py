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

from dataclasses import dataclass

import torch
import torch.nn as nn


SCENARIOS = [
    {
        "name": "last_dim_fp16_basic",
        "x_shape": (8, 32, 128),
        "index_shape": (8, 32, 64),
        "dim": 2,
        "dtype": torch.float16,
        "allow_negative_index": False,
        "expected_mode": "last_dim",
    },
    {
        "name": "last_dim_fp32_negative",
        "x_shape": (4, 97),
        "index_shape": (4, 33),
        "dim": -1,
        "dtype": torch.float32,
        "allow_negative_index": True,
        "expected_mode": "last_dim",
    },
    {
        "name": "middle_dim_fp16_post_large",
        "x_shape": (4, 128, 32),
        "index_shape": (4, 16, 32),
        "dim": 1,
        "dtype": torch.float16,
        "allow_negative_index": False,
        "expected_mode": "permute_last_dim",
    },
    {
        "name": "middle_dim_fp32_post_small",
        "x_shape": (4, 97, 2),
        "index_shape": (4, 15, 2),
        "dim": 1,
        "dtype": torch.float32,
        "allow_negative_index": False,
        "expected_mode": "permute_last_dim",
    },
    {
        "name": "dim0_fp16",
        "x_shape": (64, 8, 16),
        "index_shape": (12, 8, 16),
        "dim": 0,
        "dtype": torch.float16,
        "allow_negative_index": False,
        "expected_mode": "permute_last_dim",
    },
    {
        "name": "negative_dim_mid_fp32",
        "x_shape": (2, 5, 7, 16),
        "index_shape": (2, 3, 7, 16),
        "dim": -3,
        "dtype": torch.float32,
        "allow_negative_index": True,
        "expected_mode": "permute_last_dim",
    },
]


def normalize_dim(dim: int, rank: int) -> int:
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"Invalid dim={dim} for rank={rank}")
    return normalized


def build_scenario_key(x_shape, index_shape, dim: int, dtype) -> tuple:
    return (
        tuple(int(v) for v in x_shape),
        tuple(int(v) for v in index_shape),
        int(dim),
        str(dtype),
    )


SCENARIO_BY_KEY = {
    build_scenario_key(
        scenario["x_shape"],
        scenario["index_shape"],
        normalize_dim(scenario["dim"], len(scenario["x_shape"])),
        scenario["dtype"],
    ): scenario
    for scenario in SCENARIOS
}


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
        _, normalized_dim = self._resolve_scenario(x, index, dim)
        return torch.gather(x, normalized_dim, index.to(torch.int64))

    def _resolve_scenario(self, x: torch.Tensor, index: torch.Tensor, dim: int):
        if x.ndim != index.ndim:
            raise ValueError(f"Expected x.ndim == index.ndim, got {x.ndim} and {index.ndim}")

        normalized_dim = normalize_dim(dim, x.ndim)
        key = build_scenario_key(x.shape, index.shape, normalized_dim, x.dtype)
        scenario = SCENARIO_BY_KEY.get(key)
        if scenario is None:
            supported = ", ".join(
                f"(x={case['x_shape']}, index={case['index_shape']}, dim={case['dim']}, dtype={case['dtype']})"
                for case in SCENARIOS
            )
            raise ValueError(
                f"Unsupported gather_elements_v2 case x={tuple(x.shape)}, index={tuple(index.shape)}, "
                f"dim={dim}, dtype={x.dtype}. Supported cases: {supported}"
            )
        return scenario, normalized_dim


def _make_x(shape, dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)


def _make_index(shape, gather_dim_size: int, allow_negative_index: bool, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if allow_negative_index:
        return torch.randint(
            -gather_dim_size,
            gather_dim_size,
            shape,
            generator=generator,
            dtype=torch.int32,
        )
    return torch.randint(
        0,
        gather_dim_size,
        shape,
        generator=generator,
        dtype=torch.int32,
    )


def get_input_groups():
    input_groups = []
    for index, scenario in enumerate(SCENARIOS):
        normalized_dim = normalize_dim(scenario["dim"], len(scenario["x_shape"]))
        gather_dim_size = scenario["x_shape"][normalized_dim]
        x = _make_x(scenario["x_shape"], scenario["dtype"], seed=2026 + index)
        gather_index = _make_index(
            scenario["index_shape"],
            gather_dim_size,
            scenario["allow_negative_index"],
            seed=3036 + index,
        )
        input_groups.append([x, gather_index, int(scenario["dim"])])
    return input_groups


def product_dims(shape):
    """Return the product of all dimension extents in *shape*."""
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def compute_strides(shape):
    """Return prefix strides for a multi-dimensional shape.

    For shape (a, b, c), returns (b*c, c, 1).
    """
    strides = []
    running = 1
    for extent in reversed(shape):
        strides.append(running)
        running *= int(extent)
    return list(reversed(strides))


def validate_gather_inputs(x, index, normalized_dim):
    """Validate x dtype, index dtype, and per-axis extents."""
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Unsupported x dtype {x.dtype}, expected float16 or float32")
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Unsupported index dtype {index.dtype}, expected int32 or int64")
    for axis, (x_extent, index_extent) in enumerate(zip(x.shape, index.shape)):
        if axis != normalized_dim:
            if index_extent > x_extent:
                raise ValueError(f"index extent {index_extent} exceeds x extent {x_extent} on axis {axis}")


@dataclass
class GatherPadding:
    """Padded tensors and stride metadata for gather_elements_v2 kernel dispatch."""
    x_padded: torch.Tensor
    index_padded: torch.Tensor
    x_stride: int
    y_stride: int
    x_gather_dim: int
    idx_gather_dim: int


@dataclass
class GatherPrepareResult:
    """Result of _prepare_gather: permuted tensors, shape metadata, padding, mode, and inverse permutation."""
    x_perm: torch.Tensor
    index_perm: torch.Tensor
    x_prefix_shape: tuple
    idx_prefix_shape: tuple
    x_rows: int
    idx_rows: int
    padding: GatherPadding
    mode: str
    inverse_perm: list


TRANSPOSE_MODE_MAX_X_GATHER = 2048
TRANSPOSE_MODE_MAX_IDX_GATHER = 2048


def _prepare_gather(x, index, dim):
    """Common preparation logic for gather_elements_v2 forward methods.

    Returns:
        GatherPrepareResult or None if the output would be empty.
    """
    if x.ndim != index.ndim:
        raise ValueError(f"Expected x.ndim == index.ndim, got {x.ndim} and {index.ndim}")
    normalized_dim = normalize_dim(dim, x.ndim)
    validate_gather_inputs(x, index, normalized_dim)

    perm = [axis for axis in range(x.ndim) if axis != normalized_dim] + [normalized_dim]
    inverse_perm = [0] * x.ndim
    for new_axis, old_axis in enumerate(perm):
        inverse_perm[old_axis] = new_axis

    x_perm = x.permute(perm).contiguous()
    index_perm = index.permute(perm).contiguous().to(torch.int32)
    index_perm = torch.where(index_perm < 0, index_perm + x_perm.shape[-1], index_perm)

    x_prefix_shape = tuple(int(v) for v in x_perm.shape[:-1])
    idx_prefix_shape = tuple(int(v) for v in index_perm.shape[:-1])
    x_rows = product_dims(x_prefix_shape)
    idx_rows = product_dims(idx_prefix_shape)

    if idx_rows == 0 or x_perm.shape[-1] == 0:
        return None

    p = compute_gather_padding(x_perm, index_perm, x_rows, idx_rows)

    if x_prefix_shape == idx_prefix_shape:
        mode = "last_dim"
    elif (idx_rows >= 64
          and p.x_gather_dim <= TRANSPOSE_MODE_MAX_X_GATHER
          and p.idx_gather_dim <= TRANSPOSE_MODE_MAX_IDX_GATHER):
        mode = "transpose"
    else:
        mode = "indexed"

    return GatherPrepareResult(
        x_perm, index_perm, x_prefix_shape, idx_prefix_shape, x_rows, idx_rows, p, mode, inverse_perm)


def compute_gather_padding(x_perm, index_perm, x_rows, idx_rows):
    """Pad x_perm and index_perm to aligned strides."""
    x_gather_dim = int(x_perm.shape[-1])
    idx_gather_dim = int(index_perm.shape[-1])

    element_bytes = 2 if x_perm.dtype == torch.float16 else 4
    x_stride = ((x_gather_dim * element_bytes + 31) // 32) * (32 // element_bytes)
    y_stride = ((idx_gather_dim * 4 + 31) // 32) * 8

    x_rows_2d = x_perm.reshape(x_rows, x_gather_dim)
    index_rows_2d = index_perm.reshape(idx_rows, idx_gather_dim)

    x_padded = x_perm.new_zeros((x_rows, x_stride))
    x_padded[:, :x_gather_dim] = x_rows_2d

    index_padded = index_rows_2d.new_zeros((idx_rows, y_stride))
    index_padded[:, :idx_gather_dim] = index_rows_2d

    return GatherPadding(x_padded, index_padded, x_stride, y_stride, x_gather_dim, idx_gather_dim)


def compute_row_map(x_prefix_shape, idx_prefix_shape, device):
    """Build a row-mapping tensor from idx_prefix_shape → x_prefix_shape."""
    if len(idx_prefix_shape) == 0:
        return torch.zeros((1,), dtype=torch.int32, device=device)

    idx_rows = product_dims(idx_prefix_shape)
    x_prefix_strides = compute_strides(x_prefix_shape)
    idx_prefix_strides = compute_strides(idx_prefix_shape)

    row_ids = torch.arange(idx_rows, device=device, dtype=torch.int64)
    row_map = torch.zeros((idx_rows,), device=device, dtype=torch.int64)
    tmp = row_ids
    for axis, idx_stride in enumerate(idx_prefix_strides):
        coord = tmp // int(idx_stride)
        tmp = tmp - coord * int(idx_stride)
        row_map = row_map + coord * int(x_prefix_strides[axis])

    return row_map.to(torch.int32)


def get_init_inputs():
    return []
