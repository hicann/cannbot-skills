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

# Template for: concat_dv2
import torch
import torch.nn as nn

from _concat_utils import SUPPORTED_DTYPES, validate_concat_inputs
from concat_dv2.design.tile_level.concat_dim0 import (
    concat_dim0_1 as tl_concat_dim0_1,
    concat_dim0_2 as tl_concat_dim0_2,
    concat_dim0_3 as tl_concat_dim0_3,
    concat_dim0_4 as tl_concat_dim0_4,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._kernel_cache = {}

    def forward(self, inputs, concat_dim: int = 0):
        inputs_2d, tail_shape, dim0_sizes = self._normalize_inputs(inputs, concat_dim)
        cache_key = (
            tuple(int(tensor.shape[0]) for tensor in inputs_2d),
            int(inputs_2d[0].shape[1]),
            str(inputs_2d[0].dtype).split(".")[-1],
        )
        kernel = self._build_kernel(inputs_2d)
        self._warmup_if_needed(cache_key, kernel, inputs_2d)
        y_2d = self._invoke_kernel(kernel, inputs_2d)

        output_shape = (sum(dim0_sizes), *tail_shape)
        return y_2d.reshape(output_shape)

    def _normalize_inputs(self, inputs, concat_dim: int):
        return validate_concat_inputs(inputs, concat_dim, "TileLang")

    def _build_kernel(self, inputs_2d):
        dim0_sizes = [int(tensor.shape[0]) for tensor in inputs_2d]
        same_dim_size = int(inputs_2d[0].shape[1])
        dtype = str(inputs_2d[0].dtype).split(".")[-1]

        cache_key = (tuple(dim0_sizes), same_dim_size, dtype)
        cached = self._kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        input_count = len(inputs_2d)
        if input_count == 1:
            kernel = tl_concat_dim0_1(dim0_sizes[0], same_dim_size, dtype=dtype)
        elif input_count == 2:
            kernel = tl_concat_dim0_2(dim0_sizes[0], dim0_sizes[1], same_dim_size, dtype=dtype)
        elif input_count == 3:
            kernel = tl_concat_dim0_3(dim0_sizes[0], dim0_sizes[1], dim0_sizes[2], same_dim_size, dtype=dtype)
        elif input_count == 4:
            kernel = tl_concat_dim0_4(
                dim0_sizes[0],
                dim0_sizes[1],
                dim0_sizes[2],
                dim0_sizes[3],
                same_dim_size,
                dtype=dtype,
            )
        else:
            raise ValueError(f"unsupported input count: {input_count}")

        self._kernel_cache[cache_key] = kernel
        return kernel

    def _invoke_kernel(self, kernel, inputs_2d):
        input_count = len(inputs_2d)
        if input_count == 1:
            return kernel(inputs_2d[0])
        if input_count == 2:
            return kernel(inputs_2d[0], inputs_2d[1])
        if input_count == 3:
            return kernel(inputs_2d[0], inputs_2d[1], inputs_2d[2])
        return kernel(inputs_2d[0], inputs_2d[1], inputs_2d[2], inputs_2d[3])

    def _warmup_if_needed(self, cache_key, kernel, inputs_2d):
        warmed_key = ("warmed", cache_key)
        if self._kernel_cache.get(warmed_key):
            return
        warm_inputs = [tensor.clone() for tensor in inputs_2d]
        _ = self._invoke_kernel(kernel, warm_inputs)
        self._kernel_cache[warmed_key] = True
