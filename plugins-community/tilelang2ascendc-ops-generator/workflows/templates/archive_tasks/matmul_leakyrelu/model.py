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
    def __init__(self, negative_slope: float = 0.001) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        out = F.leaky_relu(out, negative_slope=self.negative_slope)
        if a.dtype == torch.float16:
            out = out.half()
        return out


def get_input_groups():
    cases = [
        ("float16", 1024, 1024, 1024),
        ("float16", 4096, 4096, 4096),
        ("float16", 2048, 2048, 2048),
        ("float16", 512, 512, 1024),
        ("float16", 768, 512, 256),
    ]
    input_groups = []
    for _, m, n, k in cases:
        a = torch.randn(m, k).half()
        b = torch.randn(k, n).half()
        input_groups.append([a, b])
    return input_groups


def get_init_inputs():
    return []
