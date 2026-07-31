# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 reference provider for `backward` op-gen mode (B3.2).

Given a differentiable PyTorch *forward* spec (the `--backward <forward_spec.py>`
input), produce the **backward (gradient) reference truth** self-contained — no
edge_dataset, no NPU, no A3, no CANN. This is the harness wiring that makes
`/ascendc-op-gen --backward <forward_spec>` end-to-end able to emit the autograd
reference a worker then verifies its AscendC backward kernel against.

Design decision (BACKWARD_PLUGIN_DESIGN §5.5, owner-fixed 2026-05-29, NOT open):
B3.2 uses the **self-contained verify** path (the mul_grad / rms_norm_grad probe
style: the verify script inline-seeds randn forward inputs + grad_outputs and
calls the autograd oracle for truth) — NOT the benchmark edge_dataset path.
Rationale: (1) mul_grad already hardware-validated this path; (2) edge_dataset
depends on the benchmark submodule (github.com, unreachable from this env);
(3) the self-contained path has zero external dependency. The original §5.2
edge_dataset sketch is superseded for backward mode by §5.5.

Concern split (design §4): this is C1 (backward *generation* reference). C2
(backward *perf*, OL-200) is the cross-cutting brief block wired in PR2.

Phase boundary: B3.2 = produce the reference truth (this module) + wire it into
the orchestrator O2.5 dispatch. B3.3 = the worker authors the backward AscendC
kernel + verifies vs this autograd truth on real A3 hardware (npu-a3-back).

────────────────────────────────────────────────────────────────────────────
Forward-spec convention (the `--backward <.py>` contract)
────────────────────────────────────────────────────────────────────────────
The forward spec .py MUST define a module-level `forward` callable (the
differentiable PyTorch forward) and a `BACKWARD_SPEC` dict. Mirrors the
SCHEMA shape of reference_provider/backward_cpu_truth.template.py.

    import torch

    def forward(x, w):
        var = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * w

    BACKWARD_SPEC = {
        "wrt": ["x", "w"],                 # inputs to differentiate (grad outputs)
        "inputs": {                        # how to seed each forward input
            "x": {"shape": ["R", "H"], "dtype": "float32"},
            "w": {"shape": ["H"],      "dtype": "float32"},
        },
        "cases": [                         # symbolic-dim bindings → concrete shapes
            {"R": 4,  "H": 1024},
            {"R": 8,  "H": 2048},
            {"R": 16, "H": 4096},
        ],
        "dtypes": ["float32", "float16"],  # optional, default ["float32"]
        "grad_output": "explicit",         # "explicit" (gy fed as kernel input,
                                           #  seeded randn) | "ones" (reduce-sum loss)
        "seed": 1234,                      # optional, default 1234
    }

Field rules:
- `forward`            required callable. Its parameter names define input order.
- `BACKWARD_SPEC.wrt`  required non-empty list; each name MUST be a forward param.
- `inputs`             required; one entry per forward param. `shape` is a list of
                       ints and/or symbolic strings; symbols are resolved per case.
                       `dtype` is a torch dtype name ("float32"/"float16"/"bfloat16").
- `cases`              OPTIONAL list of symbol→int dicts. OMIT IT to auto-derive a
                       default shape sweep (small/medium/large) from the input
                       signature — a user generating a backward from a forward op
                       need not hand-write cases. Provide it only to override with
                       specific shapes (use `[{}]` when all shapes are concrete).
- `dtypes`             optional; floating dtypes to sweep. Default ["float32"].
- `grad_output`        optional; "ones" (default) or "explicit".
- `seed`               optional int (default 1234) — deterministic randn seeding.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# reference_provider is a sibling package dir; ensure it's importable.
_ORCH_DIR = Path(__file__).resolve().parent
_RP_DIR = _ORCH_DIR.parent / "reference_provider"
if str(_RP_DIR) not in sys.path:
    sys.path.insert(0, str(_RP_DIR))

_VALID_VERDICTS = (
    "READY",                 # reference truth produced; ≥1 scoreable case
    "FORWARD_SPEC_MISSING",  # backward_forward_source absent / unreadable
    "SPEC_INVALID",          # spec doesn't satisfy the BACKWARD_SPEC contract
    "NOT_DIFFERENTIABLE",    # forward not differentiable w.r.t. a wrt input
    "ALL_DEGENERATE",        # every case's fp64 truth was non-finite (§5.4)
)


@dataclass
class BackwardRefReport:
    verdict: str
    summary: str = ""
    op: str = ""
    n_cases: int = 0
    n_ok: int = 0
    n_skipped: int = 0
    wrt: list = field(default_factory=list)
    grad_output: str = "ones"
    errors: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)


_DTYPE_BY_NAME = None


def _torch_dtype(name: str):
    """Resolve a dtype name to a torch dtype (lazy — torch imported on demand)."""
    global _DTYPE_BY_NAME
    import torch
    if _DTYPE_BY_NAME is None:
        _DTYPE_BY_NAME = {
            "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float64": torch.float64, "double": torch.float64,
        }
    if name not in _DTYPE_BY_NAME:
        raise ValueError(f"unsupported dtype name {name!r} "
                         f"(known: {sorted(_DTYPE_BY_NAME)})")
    return _DTYPE_BY_NAME[name]


def _load_forward_spec(spec_path: Path):
    """Import the forward spec .py as a module. Returns the module object."""
    spec = importlib.util.spec_from_file_location(
        f"_backward_forward_spec_{abs(hash(str(spec_path)))}", spec_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load forward spec module from {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_shape(shape_decl: list, binding: dict) -> list:
    """Resolve a [dim|symbol,...] shape against a symbol→int binding."""
    out = []
    for d in shape_decl:
        if isinstance(d, int):
            out.append(d)
        elif isinstance(d, str):
            if d not in binding:
                raise KeyError(f"shape symbol {d!r} not bound in case {binding}")
            out.append(int(binding[d]))
        else:
            raise TypeError(f"shape dim must be int or symbol str, got {d!r}")
    return out


def _validate_spec(mod, op: str) -> tuple[Any, dict, list[str]]:
    """Validate the forward spec module against the BACKWARD_SPEC contract.

    Returns (forward, spec_dict, errors). errors non-empty → SPEC_INVALID.
    """
    import inspect
    errors: list[str] = []
    forward = getattr(mod, "forward", None)
    spec = getattr(mod, "BACKWARD_SPEC", None)
    if not callable(forward):
        errors.append("forward spec must define a module-level callable `forward`")
    if not isinstance(spec, dict):
        errors.append("forward spec must define a `BACKWARD_SPEC` dict")
        return forward, {}, errors

    wrt = spec.get("wrt")
    inputs = spec.get("inputs")
    cases = spec.get("cases")
    if not isinstance(wrt, list) or not wrt or not all(isinstance(n, str) for n in wrt):
        errors.append("BACKWARD_SPEC['wrt'] must be a non-empty list[str]")
    if not isinstance(inputs, dict) or not inputs:
        errors.append("BACKWARD_SPEC['inputs'] must be a non-empty dict")
    # cases is OPTIONAL (2026-06-16, owner directive): a user bringing a forward
    # inference op to GENERATE its backward should NOT have to hand-write test
    # cases. When omitted/empty, provision auto-derives a default coverage sweep
    # from the input signature (_auto_derive_cases). Only a present-but-non-list
    # value is an error; `cases` set explicitly still works as an override.
    if cases is not None and not isinstance(cases, list):
        errors.append("BACKWARD_SPEC['cases'], if provided, must be a list "
                      "(omit it entirely to auto-derive a default shape sweep)")

    if callable(forward) and isinstance(inputs, dict):
        signature = inspect.signature(forward)
        fwd_params = list(signature.parameters)
        unknown_inputs = [n for n in inputs if n not in fwd_params]
        if unknown_inputs:
            errors.append(f"BACKWARD_SPEC['inputs'] names not in forward params "
                          f"{fwd_params}: {unknown_inputs}")
        required_params = []
        excluded_kinds = (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        for name, parameter in signature.parameters.items():
            if (
                parameter.default is inspect.Parameter.empty
                and parameter.kind not in excluded_kinds
            ):
                required_params.append(name)
        missing_inputs = [name for name in required_params if name not in inputs]
        if missing_inputs:
            errors.append(
                "BACKWARD_SPEC['inputs'] is missing required forward params "
                f"{missing_inputs}"
            )
        if isinstance(wrt, list):
            bad_wrt = [n for n in wrt if n not in fwd_params]
            if bad_wrt:
                errors.append(f"BACKWARD_SPEC['wrt'] names not in forward params "
                              f"{fwd_params}: {bad_wrt}")

    go = spec.get("grad_output", "ones")
    if go not in ("ones", "explicit"):
        errors.append(f"BACKWARD_SPEC['grad_output'] must be 'ones' or 'explicit', "
                      f"got {go!r}")
    return forward, (spec if isinstance(spec, dict) else {}), errors


def _generate_model_py(op: str, spec: dict) -> str:
    """Generate the canonical backward-reference model.py (OL-160 entry-point).

    The generated Model.forward takes the forward inputs (in declared order) plus
    a trailing gy when grad_output=='explicit', and returns a tuple of grads —
    the EXACT same self-contained autograd computation `compute_backward_reference`
    performs (inlined so model.py is deployable to the container without the repo).
    A regression test pins model.py's grads bit-equal to compute_backward_reference.
    """
    input_order = list(spec["inputs"].keys())
    wrt = list(spec["wrt"])
    grad_output = spec.get("grad_output", "ones")
    return f'''"""Canonical backward reference for {op} (B3.2 generated — DO NOT EDIT).

Self-contained autograd backward truth: forward inputs (declared order) {"+ gy" if grad_output == "explicit" else ""}
-> tuple of grads dL/d{{{", ".join(wrt)}}}, computed via fp64 torch.autograd.grad
(the same oracle as reference_provider.autograd_backward_reference, inlined so
this file is deployable to the NPU container without the repo). OL-160 canonical
entry-point — anti-cheat scanners target this name.
"""
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from forward_spec import forward as _forward  # noqa: E402

_INPUT_ORDER = {input_order!r}
_WRT = {wrt!r}
_GRAD_OUTPUT = {grad_output!r}


class Model(nn.Module):
    """Backward (gradient) reference for {op} via fp64 autograd oracle."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Positional args = forward inputs in declared order{", then gy" if grad_output == "explicit" else ""}.
        if _GRAD_OUTPUT == "explicit":
            *fwd_args, gy = args
        else:
            fwd_args, gy = list(args), None
        if len(fwd_args) != len(_INPUT_ORDER):
            raise ValueError(
                f"expected {{len(_INPUT_ORDER)}} forward inputs "
                f"{{_INPUT_ORDER}}, got {{len(fwd_args)}}")
        named = dict(zip(_INPUT_ORDER, fwd_args))
        out_dtype = next((t.dtype for t in fwd_args
                          if torch.is_floating_point(t)), torch.float32)

        cast = {{}}
        leaves = []
        for name, t in named.items():
            if name in _WRT:
                leaf = t.detach().double().requires_grad_(True)
                cast[name] = leaf
                leaves.append(leaf)
            elif torch.is_floating_point(t):
                cast[name] = t.detach().double()
            else:
                cast[name] = t.detach()

        with torch.enable_grad():
            y = _forward(**cast)
            outs = tuple(y) if isinstance(y, (tuple, list)) else (y,)
            if gy is None:
                gos = tuple(torch.ones_like(o) for o in outs)
            else:
                gos = (gy.detach().double(),)
            grads = torch.autograd.grad(
                outputs=outs, inputs=leaves, grad_outputs=gos,
                retain_graph=False, allow_unused=False)
        return tuple(g.to(out_dtype) for g in grads)


def get_init_inputs():
    return []
'''


# Default shape sweep used when BACKWARD_SPEC omits `cases` (owner 2026-06-16:
# a user bringing a forward op to generate its backward must NOT hand-write cases).
_AUTO_CASE_SIZES = (64, 128, 256)  # small / medium / large


def _auto_derive_cases(inputs: dict) -> list:
    """Derive a default case (shape-binding) sweep from the input signature.

    Collect every symbolic dim (a `str` entry in any input's `shape`). If the
    shapes are fully concrete → a single empty binding `[{}]`. Otherwise emit one
    binding per default size (small/medium/large), each setting EVERY symbol to
    the same size — a capped, representative sweep with no combinatorial blowup
    (the dtype × edge-profile axes already multiply coverage). A user who wants
    specific shapes still sets `BACKWARD_SPEC['cases']` to override this.
    """
    syms: list = []
    for decl in (inputs or {}).values():
        shape = decl.get("shape", []) if isinstance(decl, dict) else []
        for d in shape:
            if isinstance(d, str) and d not in syms:
                syms.append(d)
    if not syms:
        return [{}]
    return [{s: sz for s in syms} for sz in _AUTO_CASE_SIZES]


def provision_backward_reference(
    workspace: Path,
    forward_spec: Path,
    *,
    op: str | None = None,
) -> BackwardRefReport:
    """Produce the self-contained backward reference truth for a backward-mode op.

    Steps:
      1. Load + validate the forward spec (BACKWARD_SPEC contract).
      2. For each (case × dtype): seed forward inputs (deterministic randn), form
         grad_outputs (ones or explicit randn gy), compute the fp64 autograd truth
         via compute_backward_reference. Non-finite truth → skip (§5.4 guard).
      3. Emit workspace artifacts: forward_spec.py (copied), model.py (canonical
         OL-160 reference), backward_ref.json (resolved cases + smoke result),
         backward_cpu_truth.pt (per-case grads).
      4. READY iff ≥1 scoreable case; else a typed block verdict.

    Pure CPU. No NPU/A3/CANN. The worker (B3.3) verifies its kernel vs this.
    """
    import torch  # local import — keeps module import cheap for non-backward paths
    from autograd_backward_reference import (
        compute_backward_reference,
        DegenerateReferenceError,
    )
    from backward_input_gen import (
        DEFAULT_PROFILES,
        materialize_inputs,
        materialize_grad_outputs,
    )

    workspace = Path(workspace)
    op = op or workspace.name
    forward_spec = Path(forward_spec)

    if not forward_spec.is_file():
        return BackwardRefReport(
            verdict="FORWARD_SPEC_MISSING", op=op,
            summary=f"backward_forward_source not found: {forward_spec}",
            errors=[f"forward spec missing: {forward_spec}"],
        )

    try:
        mod = _load_forward_spec(forward_spec)
    except Exception as e:
        return BackwardRefReport(
            verdict="SPEC_INVALID", op=op,
            summary=f"failed to import forward spec: {e!r}",
            errors=[f"import error: {e!r}"],
        )

    forward, spec, errors = _validate_spec(mod, op)
    if errors:
        return BackwardRefReport(
            verdict="SPEC_INVALID", op=op,
            summary="forward spec does not satisfy the BACKWARD_SPEC contract",
            errors=errors,
        )

    wrt = list(spec["wrt"])
    inputs_decl = spec["inputs"]
    # cases OPTIONAL (owner 2026-06-16): omitted/empty → auto-derive a default
    # shape sweep from the input signature; present → used verbatim (override).
    cases = spec.get("cases") or _auto_derive_cases(inputs_decl)
    dtypes = spec.get("dtypes", ["float32"])
    grad_output = spec.get("grad_output", "ones")
    seed = int(spec.get("seed", 1234))
    # Edge-aware input_gen (2026-06-11): each (dtype × case) expands over a set
    # of value-profiles (standard / zeros / large / small / boundary) so backward
    # verify covers edge values, not just a few randn points. Spec may override
    # via BACKWARD_SPEC["input_profiles"]; default is the rich set.
    profiles = spec.get("input_profiles", DEFAULT_PROFILES)
    if not isinstance(profiles, list) or not profiles:
        profiles = DEFAULT_PROFILES

    truth_records: list[dict] = []
    n_ok = n_skipped = 0
    case_idx = 0
    for dt_name in dtypes:
        try:
            dtype = _torch_dtype(dt_name)
        except ValueError as e:
            return BackwardRefReport(
                verdict="SPEC_INVALID", op=op,
                summary=str(e), errors=[str(e)],
            )
        for binding in cases:
            for profile in profiles:
                case_idx += 1
                # Edge-aware materialization (2026-06-11): deterministic per record
                # idx; inputs SAVED into backward_cpu_truth.pt so the worker's verify
                # LOADS them (robust to arbitrary edge transforms — no fragile
                # re-seed). The recipe (seed+profile+binding) in backward_ref.json
                # keeps the dataset regenerable from the spec alone (.pt = cache).
                try:
                    seeded, resolved_shapes = materialize_inputs(
                        inputs_decl, binding, dtype, profile, seed, case_idx)
                except Exception as e:
                    return BackwardRefReport(
                        verdict="SPEC_INVALID", op=op,
                        summary=f"input gen failed (case {binding} profile {profile}): {e!r}",
                        errors=[f"input_gen error: {e!r}"],
                    )

                grad_outputs = None
                gy_shape = None
                if grad_output == "explicit":
                    grad_outputs, gy_shape = materialize_grad_outputs(
                        forward, seeded, case_idx, seed)
                    if isinstance(grad_outputs, (tuple, list)):
                        return BackwardRefReport(
                            verdict="SPEC_INVALID",
                            op=op,
                            summary=(
                                "multi-output forward with explicit grad_output is not "
                                "supported by the generated kernel verification contract"
                            ),
                            errors=[
                                "BACKWARD_SPEC['grad_output']='explicit' requires a "
                                "single-output forward"
                            ],
                            wrt=wrt,
                            grad_output=grad_output,
                        )

                try:
                    grads = compute_backward_reference(
                        forward, seeded, wrt, grad_outputs)
                except DegenerateReferenceError:
                    n_skipped += 1
                    truth_records.append({
                        "idx": case_idx, "binding": dict(binding), "dtype": dt_name,
                        "profile": profile, "shapes": resolved_shapes,
                        "status": "skipped",
                        "reason": "non-finite fp64 truth (§5.4)",
                    })
                    continue
                except RuntimeError as e:
                    # autograd returned None for a wrt → forward not differentiable.
                    return BackwardRefReport(
                        verdict="NOT_DIFFERENTIABLE", op=op,
                        summary=f"forward not differentiable w.r.t. a wrt input: {e}",
                        errors=[str(e)], wrt=wrt, grad_output=grad_output,
                    )

                n_ok += 1
                truth_records.append({
                    "idx": case_idx, "binding": dict(binding), "dtype": dt_name,
                    "profile": profile, "shapes": resolved_shapes,
                    "gy_shape": gy_shape,
                    "status": "ok",
                    "grad_shapes": {n: list(g.shape) for n, g in grads.items()},
                    "inputs": {n: t.clone() for n, t in seeded.items()},
                    "grad_outputs": (
                        [g.clone() for g in grad_outputs]
                        if isinstance(grad_outputs, tuple)
                        else (grad_outputs.clone()
                              if grad_outputs is not None else None)
                    ),
                    "grads": {n: g.clone() for n, g in grads.items()},
                })

    if n_ok == 0:
        return BackwardRefReport(
            verdict="ALL_DEGENERATE", op=op,
            summary=f"all {n_skipped} cases produced non-finite fp64 truth "
                    f"(§5.4) — nothing scoreable",
            n_cases=case_idx, n_ok=0, n_skipped=n_skipped,
            wrt=wrt, grad_output=grad_output,
        )

    # ── Emit artifacts ──────────────────────────────────────────────────────
    workspace.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    # 1. forward_spec.py — copy verbatim so model.py can import `forward`.
    # Re-run safety: when re-invoking `--backward` on an EXISTING workspace, the passed
    # forward_spec IS already workspace/forward_spec.py → copyfile(src, src) raises
    # SameFileError. Skip the self-copy (the spec is already in place).
    spec_dst = workspace / "forward_spec.py"
    if Path(forward_spec).resolve() != spec_dst.resolve():
        shutil.copyfile(forward_spec, spec_dst)
    artifacts.append("forward_spec.py")

    # 2. model.py — canonical OL-160 reference (deployable, inline autograd).
    (workspace / "model.py").write_text(_generate_model_py(op, spec))
    artifacts.append("model.py")

    # 3. backward_cpu_truth.pt — per-case grads (the reference dataset).
    torch.save({
        "op": op, "wrt": wrt, "grad_output": grad_output,
        "method": "CPU fp64 autograd backward reference "
                  "(compute_backward_reference); non-finite truth skipped (§5.4)",
        "records": truth_records,
    }, workspace / "backward_cpu_truth.pt")
    artifacts.append("backward_cpu_truth.pt")

    # 4. backward_ref.json — the "ref_runnable" marker for backward mode.
    #    STUB-FORM (2026-06-16, #406 rework, fix-harness-for-next-customer): metadata
    #    + seeding_recipe ONLY. The per-case bindings/profiles enumeration is OMITTED
    #    — it is regenerable from forward_spec.cases x dtypes x input_profiles via the
    #    recipe, no harness consumer reads cases[], and verify loads inputs/grads from
    #    backward_cpu_truth.pt. Keeps the archive compact (the per-case metadata was
    #    ~2-5k regenerable lines/op; the real golden tensors are the gitignored .pt).
    (workspace / "backward_ref.json").write_text(json.dumps({
        "schema_version": 1,
        "op": op,
        "verdict": "RUNNABLE",
        "reference_method": "self_contained_autograd_fp64",
        "wrt": wrt,
        "grad_output": grad_output,
        "n_cases": case_idx,
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "dtypes": dtypes,
        "input_profiles": profiles,
        "seed": seed,
        "seeding_recipe": (
            f"edge-aware input_gen (backward_input_gen.materialize_inputs): per "
            f"record idx, torch.manual_seed({seed}+idx); torch.randn for each "
            f"forward input in order {list(inputs_decl.keys())}, then apply the "
            f"record's value-profile (one of {profiles}); "
            + ("explicit gy = torch.randn per forward output, seed {seed}+idx+500000"
               if grad_output == "explicit" else "grad_output = ones_like(y)")
            + ". The materialized inputs + grad_outputs are ALSO saved in "
            "backward_cpu_truth.pt (records[i]['inputs'/'grad_outputs']); B3.3 "
            "verify LOADS them from the .pt (do NOT re-seed) and compares the "
            "kernel output to records[i]['grads']."
        ),
        "cases_note": "per-case bindings/profiles enumeration omitted (stub-form, "
                      "#406 rework): regenerable from forward_spec.cases x dtypes x "
                      "input_profiles via seeding_recipe. Verify loads inputs/grads "
                      "from backward_cpu_truth.pt, NOT from here; no consumer reads cases[].",
        "note": "B3.2 self-contained backward reference (BACKWARD_PLUGIN_DESIGN "
                "§5.5). Kernel-gen + hardware verify vs this truth = B3.3.",
    }, indent=2))
    artifacts.append("backward_ref.json")

    return BackwardRefReport(
        verdict="READY", op=op,
        summary=f"backward reference truth produced self-contained: "
                f"{n_ok} scoreable / {n_skipped} skipped (§5.4) / {case_idx} total "
                f"cases; grads for {wrt} (grad_output={grad_output})",
        n_cases=case_idx, n_ok=n_ok, n_skipped=n_skipped,
        wrt=wrt, grad_output=grad_output, artifacts=artifacts,
    )


def format_block_message(op: str, rep: BackwardRefReport) -> str:
    """Human-readable block message for non-READY verdicts (mirrors phase_o25)."""
    lines = [
        "=" * 72,
        f"BACKWARD Phase O2.5 BLOCK — op: {op}  verdict: {rep.verdict}",
        "=" * 72,
        rep.summary or "",
    ]
    for e in rep.errors:
        lines.append(f"  - {e}")
    if rep.verdict == "SPEC_INVALID":
        lines.append("")
        lines.append("The --backward forward spec must define a `forward` callable "
                     "and a `BACKWARD_SPEC` dict (wrt / inputs / cases). See "
                     "phase_o25_backward.py module docstring for the convention.")
    lines.append("=" * 72)
    return "\n".join(lines)
