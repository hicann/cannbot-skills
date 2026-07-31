#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""TEMPLATE — backward-mode CPU fp64 truth via autograd (B2-core increment 2).

The production pattern for the `backward` op-gen mode's reference generation.
A generated backward op customizes the SCHEMA below (its differentiable forward
+ which inputs to differentiate + how grad_outputs are formed), then runs:

    python3 backward_cpu_truth.py        # edge_inputs.pt -> cpu_truth_outputs.pt

This replaces the hand-written autograd boilerplate that each backward op's
cpu_truth_reference.py currently duplicates (fp64 promote, requires_grad,
torch.autograd.grad, finite-guard, dtype cast) — that boilerplate now lives in
`autograd_backward_reference.compute_backward_reference`, and the op only
declares its forward.

CPU-only (NO NPU / A3 / CANN): this is the reference-generation half (Phase
O2.5). Building/running the AscendC backward kernel is B3 (needs the NPU).

YOUR JOB (per op): fill SCHEMA — `forward`, `wrt`, optional `grad_outputs_fn`.
Everything else (the per-case loop, finite-guard skip, save) is generic.

Design refs: §5.1 (multi-output), §5.2 (autograd oracle), §5.4 (degenerate-ref
guard). Mirrors the SCHEMA-driven style of input_gen.template.py.
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_RP_DIR = HERE  # reference_provider/ (when copied into a workspace, adjust sys.path)
if str(_RP_DIR) not in sys.path:
    sys.path.insert(0, str(_RP_DIR))

from autograd_backward_reference import (  # noqa: E402
    compute_backward_reference,
    DegenerateReferenceError,
)


_SCHEMA_GUIDE = r"""
|# ───────────────────────── SCHEMA (customize per op) ─────────────────────────
|# Example shown for a 2-input elementwise multiply backward (mul_grad):
|#
|#   def _forward(x, y):            # the op's differentiable PyTorch forward
|#       return x * y
|#
|#   SCHEMA = {
|#       "op_name": "mul_grad",
|#       "forward": _forward,
|#       "wrt": ["x", "y"],          # grads to produce (multi-output)
|#       # grad_outputs_fn(outputs, inputs) -> tensor|tuple|None.
|#       # None => ones_like(forward output) (reduce-sum loss). Override when the
|#       # op's dy comes from a case input (e.g. inputs["dy"]).
|#       "grad_outputs_fn": None,
|#   }
|#
|# The mul example is exercised by tests/test_backward_cpu_truth_template.py.
""".replace("\n|#", "\n#")

SCHEMA: dict = {
    "op_name": "REPLACE_ME",
    "forward": None,          # Callable[..., Tensor|tuple]
    "wrt": [],                # list[str]
    "grad_outputs_fn": None,  # Optional[Callable[[outs, inputs], grad_outputs]]
}


def compute_one(case: dict, schema: dict = None) -> dict:
    """Compute the per-`wrt` grads for one case via the autograd oracle.

    `case["inputs"]` is a name->tensor (and scalars) dict, same shape as the
    cpu_truth_reference.py contract. Returns {wrt_name: grad_tensor (input dtype)}
    plus status. Non-finite fp64 truth => status='skipped' (§5.4), not 'error'.

    Only inputs matching the forward's parameter names are routed into the
    forward (auxiliary inputs like `dy` / `sparse_indices` are NOT forward args
    — they feed `grad_outputs_fn` or are unused). `wrt` must name forward args.

    schema=None resolves the module-level SCHEMA at CALL time (so callers that
    overwrite SCHEMA — e.g. a workspace's filled-in copy, or a test monkeypatch —
    are honored; a def-time default would bind the placeholder). DEBT-101 pattern.
    """
    if schema is None:
        schema = SCHEMA
    inputs = dict(case["inputs"])
    forward = schema["forward"]
    wrt = schema["wrt"]
    go_fn = schema.get("grad_outputs_fn")

    # Route ONLY the forward's declared params into the forward. Auxiliary
    # inputs (dy, indices, ...) stay out of the forward + the oracle leaves.
    fwd_params = set(inspect.signature(forward).parameters)
    fwd_tensors = {k: v for k, v in inputs.items()
                   if k in fwd_params and torch.is_tensor(v)}
    fwd_scalars = {k: v for k, v in inputs.items()
                   if k in fwd_params and not torch.is_tensor(v)}

    def forward_fn(**ti):
        return forward(**ti, **fwd_scalars)

    grad_outputs = None
    if go_fn is not None:
        # Recompute forward once (no grad) to let the op shape grad_outputs from
        # the (forward outputs, full inputs). Cheap; keeps the SCHEMA declarative.
        with torch.no_grad():
            preview = forward_fn(**{k: (v.to(torch.float64) if torch.is_floating_point(v) else v)
                                    for k, v in fwd_tensors.items()})
        grad_outputs = go_fn(preview, inputs)

    in_dtype = next((v.dtype for v in fwd_tensors.values()
                     if torch.is_floating_point(v)), torch.float32)
    try:
        grads = compute_backward_reference(forward_fn, fwd_tensors, wrt, grad_outputs)
    except DegenerateReferenceError as e:
        return {"status": "skipped", "reason": str(e)}
    out = {name: g.to(in_dtype) for name, g in grads.items()}
    out["status"] = "ok"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=str, default=str(HERE / "edge_inputs.pt"))
    ap.add_argument("--out", type=str, default=str(HERE / "cpu_truth_outputs.pt"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    data = torch.load(args.edge, map_location="cpu", weights_only=False)
    cases = data["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]

    outputs, t0 = [], time.time()
    for case in cases:
        res = compute_one(case)
        rec = {"idx": case["idx"], "name": case["name"], "shape": case.get("shape")}
        rec.update(res)
        outputs.append(rec)

    torch.save({
        "op": SCHEMA["op_name"],
        "n_cases": len(outputs),
        "outputs": outputs,
        "method": "CPU fp64 autograd backward reference (compute_backward_reference); "
                  "non-finite truth skipped per §5.4 degenerate-ref guard",
    }, args.out)
    n_ok = sum(1 for o in outputs if o["status"] == "ok")
    n_skip = sum(1 for o in outputs if o["status"] == "skipped")
    print(f"Done in {time.time()-t0:.1f}s: {n_ok} ok, {n_skip} skipped, "
          f"{len(outputs)} total → {args.out}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
