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

import csv
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
        return o


def get_input_groups():
    case_path = Path(__file__).resolve().parent / "general_case.csv"
    input_groups = []
    with case_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            batch_size = int(row["batch_size"])
            n_heads = int(row["n_heads"])
            seq_len_q = int(row["seq_len_q"])
            seq_len_kv = int(row["seq_len_kv"])
            d_k = int(row["d_k"])
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[row["dtype"]]
            q = torch.rand(batch_size, n_heads, seq_len_q, d_k, dtype=dtype)
            k = torch.rand(batch_size, n_heads, seq_len_kv, d_k, dtype=dtype)
            v = torch.rand(batch_size, n_heads, seq_len_kv, d_k, dtype=dtype)
            input_groups.append([q, k, v])
    return input_groups


def get_init_inputs():
    return []
