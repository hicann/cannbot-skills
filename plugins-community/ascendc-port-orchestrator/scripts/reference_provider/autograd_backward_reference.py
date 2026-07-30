# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Autograd-based backward reference oracle (B2-core, design §5.2/§5.4).

The `backward` op-gen mode's novelty: given a forward op expressed as a
differentiable PyTorch callable, compute the EXACT backward (gradient)
reference truth via `torch.autograd.grad` — for free, on CPU/fp64 — so the
generated AscendC backward kernel can be verified against it (reusing the
existing two-tier precision + multi-output verify path).

Pure CPU Python: NO NPU, NO A3, NO CANN. This is the reference-generation
half (Phase O2.5); building/running OUR AscendC kernel is B3 (needs the NPU).

Design refs:
- §5.2 autograd path (the clean reference oracle)
- §5.4 degenerate-reference guard (skip cases whose fp64 truth is non-finite —
  large-magnitude inputs that overflow the forward; LIG evidence: 4/38 cases).
  Without the guard those un-scoreable cases get mis-judged FAIL.
- §5.1 multi-output: one grad tensor per differentiable input (`wrt`).

Saved-tensor capture (§10.5) and finite-difference fallback (§5.3 / §10.6) are
deferred refinements — this core handles the self-contained differentiable
forward, which is the common case.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


class DegenerateReferenceError(ValueError):
    """Raised when the fp64 reference truth is non-finite (NaN/Inf) and the
    case is therefore un-scoreable. Callers (input_gen) should SKIP the case,
    not record it as FAIL (design §5.4)."""


def compute_backward_reference(
    forward_fn: Callable[..., "torch.Tensor | tuple"],
    inputs: dict[str, torch.Tensor],
    wrt: list[str],
    grad_outputs: "Optional[torch.Tensor | tuple]" = None,
    *,
    compute_dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Compute exact backward reference grads via torch.autograd.grad.

    Args:
        forward_fn: differentiable PyTorch forward. Called as
            `forward_fn(**inputs_cast)`; returns one tensor or a tuple of
            tensors (the forward outputs y).
        inputs: name -> tensor. Tensors named in `wrt` get requires_grad.
        wrt: input names to differentiate with respect to (multi-output grads).
            Order defines the returned dict's key set.
        grad_outputs: upstream grad(s) dL/dy. If None, uses ones_like for each
            forward output (i.e. reduce-sum loss). Must structurally match the
            forward outputs (single tensor or tuple).
        compute_dtype: reference compute dtype (default fp64 for a tight oracle;
            the generated kernel's fp16/bf16 output is compared with the
            fp16-aware coarser-dtype + per-output tolerance at verify time).

    Returns:
        dict name -> grad tensor (dL/d{name}) for each name in `wrt`,
        in `compute_dtype`.

    Raises:
        KeyError: a name in `wrt` is absent from `inputs`.
        DegenerateReferenceError: forward output or any grad is non-finite
            (§5.4 — caller should SKIP, not FAIL).
        RuntimeError: a `wrt` input does not actually affect the output
            (autograd returns None — the op is not differentiable w.r.t. it).
    """
    missing = [n for n in wrt if n not in inputs]
    if missing:
        raise KeyError(f"wrt names not in inputs: {missing}")
    if not wrt:
        raise ValueError("wrt must name at least one input to differentiate")

    # Cast + set requires_grad on the wrt leaves. Non-wrt inputs are passed
    # through (cast to compute_dtype only if floating — index/int tensors stay).
    cast: dict[str, torch.Tensor] = {}
    leaves: list[torch.Tensor] = []
    for name, t in inputs.items():
        if name in wrt:
            leaf = t.detach().to(compute_dtype).requires_grad_(True)
            cast[name] = leaf
            leaves.append(leaf)
        elif torch.is_floating_point(t):
            cast[name] = t.detach().to(compute_dtype)
        else:
            cast[name] = t.detach()

    with torch.enable_grad():
        y = forward_fn(**cast)
        outs = tuple(y) if isinstance(y, (tuple, list)) else (y,)
        for o in outs:
            if not torch.isfinite(o).all():
                raise DegenerateReferenceError(
                    "forward output is non-finite (NaN/Inf) — un-scoreable case"
                )
        if grad_outputs is None:
            gos = tuple(torch.ones_like(o) for o in outs)
        else:
            gos = tuple(grad_outputs) if isinstance(grad_outputs, (tuple, list)) else (grad_outputs,)
            gos = tuple(g.to(compute_dtype) for g in gos)
        grads = torch.autograd.grad(
            outputs=outs, inputs=leaves, grad_outputs=gos,
            retain_graph=False, allow_unused=True,
        )

    result: dict[str, torch.Tensor] = {}
    for name, g in zip(wrt, grads):
        if g is None:
            raise RuntimeError(
                f"grad w.r.t. {name!r} is None — forward is not differentiable "
                f"w.r.t. it (autograd allow_unused). Not a valid backward target."
            )
        if not torch.isfinite(g).all():
            raise DegenerateReferenceError(
                f"grad w.r.t. {name!r} is non-finite (NaN/Inf) — un-scoreable case"
            )
        result[name] = g
    return result
