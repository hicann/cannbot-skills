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
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if a.dtype == b.dtype == torch.int8:
            # NPU torch.matmul does not support int32 inputs; float32 is exact for this int8 range.
            out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
            return F.leaky_relu(out, negative_slope=self.negative_slope)

        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        out = F.leaky_relu(out, negative_slope=self.negative_slope)
        if a.dtype == torch.float16:
            out = out.half()
        return out


def get_input_groups():
    cases = [
        ("float16", 1024, 1024, 1024),
        ("float16", 4096, 4096, 4096),
        ("float16", 512, 768, 512),
        ("int8", 1024, 1024, 1024),
    ]
    input_groups = []
    for dtype_name, m, n, k in cases:
        if dtype_name == "float16":
            a = torch.randn(m, k).half()
            b = torch.randn(k, n).half()
        elif dtype_name == "int8":
            a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
            b = torch.randint(-8, 8, (k, n), dtype=torch.int8)
        else:
            raise ValueError(f"Unsupported dtype case: {dtype_name}")
        input_groups.append([a, b])
    return input_groups


def get_init_inputs():
    return []
