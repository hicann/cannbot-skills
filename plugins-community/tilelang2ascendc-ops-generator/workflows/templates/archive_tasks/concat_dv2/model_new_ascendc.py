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

# Template for: concat_dv2
import torch
# Template for: concat_dv2
import torch.nn as nn

from _concat_utils import SUPPORTED_DTYPES, validate_concat_inputs

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "_ascend_concat_dv2_ext*")
if _LIB_PATTERN not in "".join(sys.path):
    import glob as _glob
    _libs = _glob.glob(_LIB_PATTERN)
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, concat_dim: int = 0):
        inputs_2d, tail_shape, dim0_sizes, original_dtype = self._normalize_inputs(inputs, concat_dim)
        kernel_inputs = [tensor.to(torch.float32).contiguous() for tensor in inputs_2d]
        kernel_inputs, original_n = self._pad_width_if_needed(kernel_inputs)
        y_2d = self._run_kernel(kernel_inputs)
        if y_2d.shape[1] != original_n:
            y_2d = y_2d[:, :original_n].contiguous()
        if original_dtype != torch.float32:
            y_2d = y_2d.to(original_dtype)

        output_shape = (sum(dim0_sizes), *tail_shape)
        return y_2d.reshape(output_shape)

    def _normalize_inputs(self, inputs, concat_dim: int):
        normalized, tail_shape, dim0_sizes, reference = validate_concat_inputs(
            inputs, concat_dim, "AscendC")
        return normalized, tail_shape, dim0_sizes, reference.dtype

    def _pad_width_if_needed(self, inputs_2d):
        original_n = int(inputs_2d[0].shape[1])
        if original_n == 0:
            return inputs_2d, original_n

        padded_n = ((original_n + 7) // 8) * 8
        if padded_n == original_n:
            return inputs_2d, original_n

        padded_inputs = []
        for tensor in inputs_2d:
            padded = torch.zeros(
                (int(tensor.shape[0]), padded_n),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            padded[:, :original_n].copy_(tensor)
            padded_inputs.append(padded)
        return padded_inputs, original_n

    def _run_kernel(self, inputs_2d):
        input_count = len(inputs_2d)
        if input_count == 1:
            return torch.ops.npu.concat_dim0_1(inputs_2d[0])
        if input_count == 2:
            return torch.ops.npu.concat_dim0_2(inputs_2d[0], inputs_2d[1])
        if input_count == 3:
            return torch.ops.npu.concat_dim0_3(inputs_2d[0], inputs_2d[1], inputs_2d[2])
        return torch.ops.npu.concat_dim0_4(inputs_2d[0], inputs_2d[1], inputs_2d[2], inputs_2d[3])


def get_init_inputs():
    return []
