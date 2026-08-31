# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Small first-party KernelBench-style Add task.

This pair demonstrates the canonical task layout: a task ``.py`` (operator
implementation exposing ``Model`` + ``get_input_groups()``) plus a same-stem
``.json``/``.jsonl`` test-case sidecar.  It is a tutorial fixture, not an
upstream benchmark corpus case or an acceptance baseline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class Model(nn.Module):
    """Elementwise Add with the same three-argument ABI as the sidecar cases."""

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return torch.add(x, y, alpha=alpha)


def _load_cases() -> list[dict[str, Any]]:
    """Read JSONL from the same-stem sidecar, as old-format tasks commonly do."""

    path = Path(__file__).with_suffix(".json")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_tensor(spec: dict[str, Any], *, case_index: int, input_index: int) -> torch.Tensor:
    """Create a small deterministic CPU tensor from one old-format descriptor."""

    dtype_name = spec.get("dtype")
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported tutorial tensor dtype: {dtype_name!r}")
    shape = spec.get("shape")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"invalid tutorial tensor shape: {shape!r}")
    element_count = math.prod(shape)
    values = torch.arange(element_count, dtype=torch.float32)
    values = values.mul_(0.125).add_(case_index + input_index * 0.25)
    return values.reshape(shape).to(_DTYPES[dtype_name])


def get_input_groups() -> list[list[Any]]:
    """Produce one positional argument group per JSONL sidecar case."""

    groups: list[list[Any]] = []
    for case_index, case in enumerate(_load_cases()):
        inputs = case.get("inputs")
        if not isinstance(inputs, list):
            raise ValueError("tutorial case must contain an inputs list")
        group: list[Any] = []
        for input_index, spec in enumerate(inputs):
            if not isinstance(spec, dict):
                raise ValueError("tutorial input descriptor must be an object")
            if spec.get("type") == "tensor":
                group.append(_make_tensor(spec, case_index=case_index, input_index=input_index))
            elif spec.get("type") == "attr" and "value" in spec:
                group.append(spec["value"])
            else:
                raise ValueError(f"unsupported tutorial input descriptor: {spec!r}")
        groups.append(group)
    return groups


def get_init_inputs() -> list[Any]:
    """The Add task has no constructor arguments."""

    return []
