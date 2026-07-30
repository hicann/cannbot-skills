# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Backward-mode input generator — edge-aware, deterministic, reproducible.

The backward op-gen reference (phase_o25_backward) historically seeded ONE
`torch.randn` input per (dtype × case). That gave thin coverage: a handful of
standard-normal points, no edge values (exact zeros, large/small magnitude,
op boundaries). Owner 2026-06-11: "为什么反向 case 这么少" → build a real
input_gen so backward gets the multi-case + edge-value coverage the forward
migration path gets from its input_gen/edge_dataset.

This module is the SINGLE source of truth for materializing backward inputs,
used by BOTH:
  - phase_o25_backward.provision_backward_reference (to build the fp64 truth)
  - the worker's verify_<op>.py (to reproduce IDENTICAL inputs on-device)

Reproduction contract: inputs are materialized here AND saved into
backward_cpu_truth.pt (records[i]["inputs"]). Verify LOADS them from the .pt
rather than re-seeding — robust to arbitrary edge transforms. The recipe
(seed + profile + binding) is also recorded in backward_ref.json so the whole
dataset is REGENERABLE from the spec alone (the .pt is a cache, slim-safe).

Determinism: each record seeds `torch.manual_seed(seed + idx)` once, so a
re-run of provision_backward_reference reproduces bit-identical inputs+grads.
"""
from __future__ import annotations

from typing import Any

import torch

# Ordered default value-profiles. A spec may override via
# BACKWARD_SPEC["input_profiles"]. "randn" first = the legacy standard-normal
# point (keeps a like-for-like baseline case in every sweep).
DEFAULT_PROFILES = ["randn", "zeros", "large", "small", "boundary"]

# Single-profile default for back-compat call sites / tests that want the
# legacy behavior (one standard-normal draw per case).
LEGACY_PROFILES = ["randn"]


def resolve_shape(shape: list, binding: dict) -> list[int]:
    """Resolve a symbolic shape (ints and/or symbol strings) via a case binding."""
    out = []
    for d in shape:
        if isinstance(d, str):
            if d not in binding:
                raise KeyError(f"shape symbol {d!r} not bound in case {binding}")
            out.append(int(binding[d]))
        else:
            out.append(int(d))
    return out


def _apply_profile(base: "torch.Tensor", profile: str) -> "torch.Tensor":
    """Deterministically transform a standard-normal `base` into a value-profile.

    Uses torch ops seeded by the caller's manual_seed — reproducible. Any extra
    randomness (masks) is drawn from the same RNG sequence, so a re-run with the
    same seed yields the identical tensor.
    """
    if profile == "randn":
        return base
    if profile == "zeros":
        # ~40% of elements forced to exact 0 (probes sign(0)/boundary handling,
        # e.g. abs dx at x==0). Mask drawn from the live RNG → reproducible.
        mask = torch.rand(base.shape) < 0.4
        return base.masked_fill(mask, 0.0)
    if profile == "large":
        # Large magnitude — stresses overflow/saturation. fp16 cases that go
        # non-finite are SKIPPED by the §5.4 degenerate guard (not failed).
        return base * 100.0
    if profile == "small":
        # Near-zero magnitude — stresses fp precision / underflow in the grad.
        return base * 1e-3
    if profile == "boundary":
        # Sprinkle exact op-agnostic boundary values: 0, +1, -1, +0.5, -0.5.
        flat = base.flatten().clone()
        n = flat.numel()
        vals = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5], dtype=flat.dtype)
        # Deterministic positions from the live RNG.
        if n > 0:
            k = max(1, n // 3)
            idx = torch.randperm(n)[:k]
            flat[idx] = vals[torch.arange(k) % vals.numel()].to(flat.dtype)
        return flat.reshape(base.shape)
    raise ValueError(f"unknown input profile: {profile!r}")


def materialize_inputs(
    inputs_decl: dict,
    binding: dict,
    dtype: "torch.dtype",
    profile: str,
    seed: int,
    idx: int,
) -> tuple[dict, dict]:
    """Materialize forward inputs for one (case × dtype × profile) record.

    Deterministic: seeds once with (seed + idx). Draws each forward input in
    declaration order as standard-normal, then applies the value-profile.

    Returns (inputs dict name->tensor, resolved_shapes dict name->list[int]).
    """
    torch.manual_seed(int(seed) + int(idx))
    out: dict[str, "torch.Tensor"] = {}
    shapes: dict[str, list[int]] = {}
    for name, decl in inputs_decl.items():
        shp = resolve_shape(list(decl["shape"]), binding)
        shapes[name] = shp
        if "dtype" in decl:
            from_name = decl["dtype"]
            in_dt = _DTYPES.get(from_name)
            if in_dt is None:
                raise ValueError(f"unknown dtype {from_name!r} for input {name!r}")
        else:
            in_dt = dtype
        base = torch.randn(shp, dtype=in_dt)
        out[name] = _apply_profile(base, profile)
    return out, shapes


def materialize_grad_outputs(
    forward, inputs: dict, idx: int, seed: int,
) -> tuple[Any, list]:
    """Draw explicit grad_outputs (gy) for the forward outputs of `inputs`.

    gy is standard-normal in each output's shape/dtype (the upstream gradient
    direction — value-profiles apply to forward INPUTS, not gy). Deterministic
    via (seed + idx + 500000) to keep gy's RNG stream distinct from inputs'.
    """
    with torch.no_grad():
        y = forward(**inputs)
    outs = tuple(y) if isinstance(y, (tuple, list)) else (y,)
    torch.manual_seed(int(seed) + int(idx) + 500000)
    gos = tuple(torch.randn(o.shape, dtype=o.dtype) for o in outs)
    grad_outputs = gos if len(gos) > 1 else gos[0]
    gy_shape = [list(o.shape) for o in outs]
    return grad_outputs, gy_shape


_DTYPES = {
    "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float64": torch.float64, "double": torch.float64,
}
