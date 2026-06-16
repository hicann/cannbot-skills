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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor):
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        out = out * scale
        return out.to(torch.float16)


def get_input_groups():
    cases = [
        (1024, 1024, 1024),
    ]
    input_groups = []
    for m, n, k in cases:
        a = torch.randint(-128, 127, (m, k), dtype=torch.int8)
        b = torch.randint(-128, 127, (k, n), dtype=torch.int8)
        scale = torch.randn(n, dtype=torch.float32)
        input_groups.append([a, b, scale])
    return input_groups


def get_init_inputs():
    return []
