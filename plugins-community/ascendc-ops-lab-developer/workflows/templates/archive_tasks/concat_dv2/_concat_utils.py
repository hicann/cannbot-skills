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

"""Shared validation utilities for concat_dv2 model implementations."""

import torch

SUPPORTED_DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
)


def validate_concat_inputs(inputs, concat_dim, implementation_name):
    """Validate and normalize concat inputs. Returns (normalized, tail_shape, dim0_sizes, reference)."""
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs must be a non-empty list or tuple of tensors")
    if concat_dim != 0:
        raise ValueError(f"concat_dv2 {implementation_name} implementation only supports concat_dim=0, "
                         f"got {concat_dim}")
    if len(inputs) > 4:
        raise ValueError(f"up to 4 inputs are supported, got {len(inputs)}")

    reference = inputs[0]
    if not isinstance(reference, torch.Tensor):
        raise TypeError("all inputs must be torch.Tensor")
    if reference.ndim < 1:
        raise ValueError("each input tensor must have rank >= 1")
    if reference.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {reference.dtype}")

    tail_shape = tuple(int(dim) for dim in reference.shape[1:])
    rank = reference.ndim
    dtype = reference.dtype

    normalized = []
    dim0_sizes = []
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("all inputs must be torch.Tensor")
        if tensor.ndim != rank:
            raise ValueError("all inputs must have the same rank")
        if tensor.dtype != dtype:
            raise ValueError("all inputs must have the same dtype")
        if tuple(int(dim) for dim in tensor.shape[1:]) != tail_shape:
            raise ValueError("all non-concat dimensions must match")

        dim0 = int(tensor.shape[0])
        normalized.append(tensor.contiguous().reshape(dim0, -1))
        dim0_sizes.append(dim0)

    return normalized, tail_shape, dim0_sizes, reference
