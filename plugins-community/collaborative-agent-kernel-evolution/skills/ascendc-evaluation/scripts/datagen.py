# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import re
from typing import Callable, List

import torch

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype):
    """Resolve dtype from string or torch.dtype."""
    if isinstance(dtype, str):
        return _DTYPE_MAP[dtype]
    return dtype


def parse_distr(distr: str) -> Callable:
    """Parse distribution string like 'normal(0,1)' or 'uniform(-1,1)'.

    Returns a callable(shape, dtype, device) -> Tensor.
    """
    m = re.match(r"(\w+)\(([^,]+),\s*([^)]+)\)", distr)
    if not m:
        raise ValueError(f"Cannot parse distribution: {distr!r}")
    name, a, b = m.group(1), float(m.group(2)), float(m.group(3))
    if name == "normal":
        def _gen(shape, dtype, device="cpu"):
            return torch.randn(shape, dtype=dtype, device=device) * b + a
        return _gen
    elif name == "uniform":
        def _gen(shape, dtype, device="cpu"):
            return torch.empty(shape, dtype=dtype, device=device).uniform_(a, b)
        return _gen
    else:
        raise ValueError(f"Unknown distribution: {name}")


def make_tensor(shape, dtype=torch.bfloat16, *, seed, device="cpu",
                distr="normal(0,1)"):
    """Create a random tensor with given shape, dtype, seed, and distribution.

    Args:
        shape: Tensor shape.
        dtype: Tensor dtype (string or torch.dtype).
        seed: Random seed (required).
        device: Target device.
        distr: Distribution string (e.g. 'normal(0,1)').
    """
    torch.manual_seed(seed)
    dtype = resolve_dtype(dtype)
    gen = parse_distr(distr)
    return gen(shape, dtype, device)


def require_grads(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Clone tensors and enable gradient tracking."""
    outs: List[torch.Tensor] = []
    for t in tensors:
        t2 = t.detach().clone()
        t2.requires_grad_(True)
        outs.append(t2)
    return outs
