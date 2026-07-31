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

"""precision_eval_two_tier.py — Two-tier precision verdict per op archive.

Combines `precision_eval_cpu_ref.py` (ours vs CPU truth) and
`precision_eval_cann_vs_cpu.py` (CANN vs CPU truth) into a single judge
that emits a tiered verdict for each case + an op-level summary.

Tiers:
  Tier 1 (Strict)  — ours MERE < per-dtype threshold AND ours MARE < 10×threshold
                     (vs CPU truth, identical to existing precision_eval_cpu_ref).
  Tier 2 (Relative)— when Tier 1 fails, pass if ours_MERE ≤ CANN_MERE AND
                     ours_MARE ≤ CANN_MARE (both vs CPU truth) — i.e. we are
                     at least as accurate as CANN's reference impl.

Per-case verdict ladder (each output independently, then aggregated):
  PASS_T1   ours passes Tier 1 thresholds.
  PASS_T2   ours fails T1 but is ≤ CANN MERE/MARE (parity-or-better with CANN).
  FAIL      ours strictly worse than CANN.
  EVAL_ERR  one of {ours,CANN,CPU} crashed/produced wrong shape/etc.

Per-op summary aggregates per-case verdicts and reports:
  n_pass_t1, n_pass_t2, n_fail, n_err, beats_cann

  beats_cann = (ours_MERE STRICTLY <  CANN_MERE)
            AND (ours_MARE         ≤  CANN_MARE)

  Asymmetric on purpose: strict on average error (MERE) so beats_cann actually
  signals a win, but loose on worst-case (MARE) so the counter doesn't go to
  zero on Tier-2 ops where CANN sits at the hardware noise floor on MARE. A
  stricter "MARE strictly <" rule would lose its diagnostic value on those ops.

Threshold sources are identical to the production AscendC SKILL:
  fp16  9.77e-4 (2^-10)
  bf16  7.81e-3 (2^-7)
  fp32  1.22e-4 (2^-13)
  int{8,16,32,64} / bool: bit-exact required (Tier 1 only — no Tier 2 fallback
    because integer outputs admit no relative-tolerance framing; CANN passing
    bit-exact while ours doesn't is a real precision gap).

Usage:
  python3 precision_eval_two_tier.py <archive_dir> [--json out.json] [--quiet]

Designed for the same NPU container as the two source scripts (torch_npu loaded).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# task#15(b) fp16-aware: reuse the SINGLE-SOURCE coarser-dtype helper from the
# reference-provider verifier (do NOT duplicate the precision-rank table here).
# A lower-precision kernel output (fp16-FA) compared against a higher-precision
# oracle (fp32) must key its threshold to the COARSER dtype — see _coarser_float_dtype.
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference_provider"))
from verify import _coarser_float_dtype  # noqa: E402

# 生态 (ecosystem) DEFAULT grader — the VERBATIM-vendored cann-bench compare.py (byte-identical
# @007855b). The forward T1 verdict routes through THIS (the real ecosystem standard), replacing the
# drifted per-dtype tolerance classifier that false-FAILed fp32 near-zero. Single source of truth
# shared with the backward judge (precision_cannbot_adapter → same cannbench_grader).
sys.path.insert(0, str(Path(__file__).resolve().parent / "orchestrator" / "precision"))
from cannbench_grader import compare_tensors as _eco_compare_tensors  # noqa: E402

from precision_tier1 import (  # re-export: Tier-1 + shared consts moved to precision_tier1 (2026-07-05)
    _ecosystem_t1, compute_mere_mare, compute_small_value_error_count, classify_output,
    EPS, PRECISION_THRESHOLDS, INT_DTYPES, INT_LSB_TOLERANCE,
    SMALL_VALUE_THRESHOLDS, SMALL_VALUE_ERROR_THRESHOLDS,
)
from precision_tier2 import (  # re-export: Tier-2/3 moved to precision_tier2 (2026-07-05)
    compute_ref_self_drift, classify_output_t3, classify_index_list_t3,
)




# ---------------------------------------------------------------------------
# T3 — when CPU truth is structurally undefined (P0zz, 2026-05-06)
# ---------------------------------------------------------------------------
# T3 fires when Model.forward needs torch_npu (e.g. L4 fully-CANN-fused ops:
# npu_lstm, npu_quant_matmul, npu_fusion_attention, ...). The CPU run throws
# ImportError, so T1/T2 axis is unavailable. Reference becomes CANN-on-NPU,
# and the kernel is judged against it under T1-equivalent thresholds —
# loosened only when the reference itself is non-deterministic across 3
# successive runs.
#
# T3 trigger requires BOTH conditions (per OL-109 refinement 2026-05-06):
#   1. CPU truth structurally undefined (cpu_model.forward raised)
#   2. Reference IS non-deterministic on the test inputs (3× ref run check)
#
# If condition #2 fails (reference deterministic), T3 still applies but
# threshold = thresh_t1 strict — kernel must match CANN-on-NPU exactly within
# T1 tolerance, no admission for ref-self drift.
#
# CRITICAL SAFETY PROPERTY (P0zz, user-flagged 2026-05-06):
# T3 and T2 verdicts overlap on the "ours bit-matches CANN" case (both would
# return PASS). This is harmless ONLY because T3 is structurally unreachable
# when CPU truth is available — the trigger gate `_is_torch_npu_required_error`
# fires only when cpu_model.forward() raises. If CPU run succeeds (T1/T2 axis),
# T3 path NEVER enters.
#
# Why this matters: if T3 were ever invoked when CPU truth IS available, an op
# where "ours diverged from CPU-fp64 in the same direction as CANN" would pass
# T3 (ours bit-matches CANN) while genuinely FAILing T2 (ours_MERE > CANN_MERE
# vs CPU). T3 would mask real precision drift. The axis-stickiness + trigger
# gate prevent this. Do not add any code path that calls classify_output_t3
# when cpu_model.forward() has succeeded.




def _is_torch_npu_required_error(exc: Exception) -> bool:
    """Detect 'CPU run failed because op needs torch_npu / NPU' as opposed to
    a real eval bug. Used to switch from T1/T2 axis to T3 axis."""
    msg = str(exc).lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        if "torch_npu" in msg or "npu" in msg:
            return True
    if isinstance(exc, AttributeError):
        if "torch_npu" in msg or "npu_" in msg:
            return True
    if isinstance(exc, RuntimeError):
        # E.g. "Expected NPU device but got CPU"
        if "npu" in msg and ("device" in msg or "backend" in msg):
            return True
    return False


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().contiguous()
    if isinstance(v, (list, tuple)):
        return type(v)(to_cpu(x) for x in v)
    if isinstance(v, dict):
        return {k: to_cpu(x) for k, x in v.items()}
    return v


def to_npu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().npu().contiguous()
    if isinstance(v, (list, tuple)):
        return type(v)(to_npu(x) for x in v)
    if isinstance(v, dict):
        return {k: to_npu(x) for k, x in v.items()}
    return v


def normalize(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    if hasattr(out, "_asdict"):
        return [t for t in out._asdict().values() if isinstance(t, torch.Tensor)]
    return []


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_class(mod, prefer):
    if hasattr(mod, prefer):
        return getattr(mod, prefer)
    for a in dir(mod):
        if a.startswith("Model") and isinstance(getattr(mod, a), type):
            return getattr(mod, a)
    raise AttributeError(f"no Model in {mod}")


# ---------------------------------------------------------------------------
# Main eval — runs ours, CANN, CPU truth on each case
# ---------------------------------------------------------------------------
def _is_out_of_scope_raise(exc: Exception) -> bool:
    """True iff the exception is the kernel's TYPED out-of-scope declaration.

    Coverage-fraud guard (main, 2026-05-29): a case is SKIP-able ONLY when the
    kernel itself declared the scope boundary via the typed `_OutOfScope` sentinel
    (type named `_OutOfScope`, OR a message carrying the `_OutOfScope` sentinel +
    reason — shape > declared physical cap / feature unsupported). ANY other
    exception (RuntimeError, NotImplementedError, shape bug, etc.) is a real
    EVAL_ERR and MUST NOT be skipped — that would hide a fail as a skip.
    Matches the convention in fa_class_self_check.py (`'_OutOfScope' in error`)."""
    if type(exc).__name__ == "_OutOfScope":
        return True
    return "_OutOfScope" in (str(exc) or "")


def evaluate(archive_dir: Path, verbose: bool = True) -> dict[str, Any]:
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"archive dir not found: {archive_dir}")

    model_py = archive_dir / "model.py"
    cpu_truth_py = archive_dir / "model_cpu_truth.py"
    new_py = archive_dir / "model_new_ascendc.py"
    if not model_py.is_file():
        raise FileNotFoundError(f"missing {model_py}")
    if not new_py.is_file():
        raise FileNotFoundError(f"missing {new_py}")

    op_name = archive_dir.name
    has_cpu_truth = cpu_truth_py.is_file()
    if verbose:
        print(f"=== Two-tier precision eval for {op_name} ===")
        if has_cpu_truth:
            print("  CPU truth: model_cpu_truth.py (Path-A)")
        else:
            print("  CPU truth: model.py (same as CANN reference)")
        print("  Tier 1: ours MERE/MARE vs CPU truth, per-dtype thresholds")
        print("  Tier 2 (fallback): ours MERE/MARE ≤ CANN MERE/MARE (both vs CPU truth)")

    # Load three module incarnations:
    # - mod_ref_cpu : Model.forward on CPU (ground truth).
    #                 When model_cpu_truth.py exists (Path-A), loaded from it;
    #                 otherwise loaded from model.py (same as CANN ref).
    # - mod_ref_npu : Model.forward on NPU (CANN reference). ALWAYS from model.py
    #                 — model.py is the benchmark source, preserved for perf baseline.
    # - mod_cand    : ModelNew.forward (our kernel)
    mod_ref_cpu = load_module(cpu_truth_py if has_cpu_truth else model_py, f"{op_name}_ref_cpu")
    mod_ref_npu = load_module(model_py, f"{op_name}_ref_npu")
    mod_cand = load_module(new_py, f"{op_name}_cand")

    cls_ref_cpu = find_class(mod_ref_cpu, "Model")
    cls_ref_npu = find_class(mod_ref_npu, "Model")
    cls_cand = find_class(mod_cand, "ModelNew")

    torch.manual_seed(0)
    init = []
    if hasattr(mod_cand, "get_init_inputs"):
        init = mod_cand.get_init_inputs()
    elif hasattr(mod_ref_cpu, "get_init_inputs"):
        init = mod_ref_cpu.get_init_inputs()
    get_inputs = (
        getattr(mod_cand, "get_input_groups", None)
        or getattr(mod_ref_cpu, "get_input_groups", None)
    )
    if get_inputs is None:
        raise AttributeError("no get_input_groups in either module")
    groups = get_inputs()

    cpu_model = (cls_ref_cpu(*copy.deepcopy(init)) if init else cls_ref_cpu()).cpu().eval()
    npu_model = (cls_ref_npu(*copy.deepcopy(init)) if init else cls_ref_npu())
    cand_model = (cls_cand(*copy.deepcopy(init)) if init else cls_cand())
    if HAS_NPU:
        npu_model = npu_model.npu()
        cand_model = cand_model.npu()
    npu_model = npu_model.eval()
    cand_model = cand_model.eval()

    # P0zz (2026-05-06): tier-axis selection per case.
    # Default: T1/T2 axis (existing behavior — runs ours, CANN, CPU truth).
    # Fallback: T3 axis (when Model.forward needs torch_npu, runs ours,
    # CANN×3 with ref-self-drift gate; no CPU truth).
    op_tier_axis = None  # set on first case, then sticky for the op
    op_reference_source = None
    T3_REF_RUNS = 3  # canonical ref-run count for non-det gauge

    # Per-case eval
    results = []
    for idx, raw in enumerate(groups):
        try:
            cpu_in = to_cpu(raw)
            npu_in = to_npu(raw) if HAS_NPU else cpu_in

            # Try CPU truth (T1/T2 path). If it fails specifically because
            # Model.forward needs torch_npu / NPU device, switch this op to
            # T3 axis (CANN-on-NPU as reference, no CPU truth).
            cpu_out = None
            cpu_failure_t3_trigger = None
            with torch.no_grad():
                try:
                    cpu_out = (
                        cpu_model(*cpu_in) if isinstance(cpu_in, (list, tuple)) else cpu_model(cpu_in)
                    )
                except Exception as cpu_exc:
                    if _is_torch_npu_required_error(cpu_exc):
                        cpu_failure_t3_trigger = cpu_exc
                    else:
                        # Real eval failure unrelated to T3 axis — re-raise to
                        # the outer except where it becomes EVAL_ERR.
                        raise

                cann_runs = []
                if cpu_failure_t3_trigger is not None:
                    # T3 axis: run CANN reference T3_REF_RUNS times to gauge
                    # reference self-drift. The test inputs may or may not
                    # produce a deterministic reference; classify_output_t3
                    # uses the observed drift to size the threshold band.
                    for _ in range(T3_REF_RUNS):
                        ref_out = (
                            npu_model(*npu_in) if isinstance(npu_in, (list, tuple))
                            else npu_model(npu_in)
                        )
                        cann_runs.append(ref_out)
                    cann_out = cann_runs[0]
                else:
                    cann_out = (
                        npu_model(*npu_in) if isinstance(npu_in, (list, tuple))
                        else npu_model(npu_in)
                    )
                ours_out = (
                    cand_model(*npu_in) if isinstance(npu_in, (list, tuple)) else cand_model(npu_in)
                )

            # Sticky tier axis for the op (set on first case, never change
            # mid-op — mixed axes within one op would conflate metrics).
            this_axis = "T3" if cpu_failure_t3_trigger is not None else "T1_T2"
            if op_tier_axis is None:
                op_tier_axis = this_axis
                op_reference_source = "NPU+CANN" if this_axis == "T3" else "CPU+fp64"
            elif op_tier_axis != this_axis:
                results.append({
                    "case": idx,
                    "verdict": "EVAL_ERR",
                    "error": (
                        f"tier axis flipped mid-op: was {op_tier_axis}, this case is "
                        f"{this_axis} (cpu_run trigger: {cpu_failure_t3_trigger}). "
                        f"Tier-mixing rule violated; cannot mix T1/T2 and T3 in one op."
                    ),
                })
                continue

            cann_t = normalize(cann_out)
            ours_t = normalize(ours_out)
            if cpu_failure_t3_trigger is not None:
                # T3: cpu_t is empty; alignment check must use cann's count
                cpu_t = []
                # Normalize each cann_runs into per-output groups for T3 classifier.
                cann_runs_t = [normalize(r) for r in cann_runs]
            else:
                cpu_t = normalize(cpu_out)
                cann_runs_t = None

            # Output count alignment differs by axis
            if op_tier_axis == "T3":
                count_ok = (
                    cann_runs_t is not None
                    and all(len(rr) == len(cann_t) for rr in cann_runs_t)
                    and len(cann_t) == len(ours_t)
                )
                if not count_ok:
                    results.append({
                        "case": idx,
                        "verdict": "EVAL_ERR",
                        "error": (
                            f"T3 output count mismatch: cann_runs="
                            f"{[len(rr) for rr in (cann_runs_t or [])]} "
                            f"cann={len(cann_t)} ours={len(ours_t)}"
                        ),
                        "outputs": [],
                        "tier_axis": "T3",
                    })
                    continue
            else:
                if not (len(cpu_t) == len(cann_t) == len(ours_t)):
                    results.append({
                        "case": idx,
                        "verdict": "EVAL_ERR",
                        "error": f"output count mismatch: cpu={len(cpu_t)} cann={len(cann_t)} ours={len(ours_t)}",
                        "outputs": [],
                        "tier_axis": "T1_T2",
                    })
                    if verbose:
                        print(
                            f"case[{idx}]: EVAL_ERR  count cpu={len(cpu_t)} "
                            f"cann={len(cann_t)} ours={len(ours_t)}"
                        )
                    continue

            per_out = []
            verdicts = []
            if op_tier_axis == "T3":
                # Per-output T3 classification using ref_runs across the 3 NPU runs
                for j, ours_ in enumerate(ours_t):
                    ref_runs_for_out = [rr[j] for rr in cann_runs_t]
                    # Cast all to first-ref's dtype for comparison
                    ref_dtype = ref_runs_for_out[0].dtype
                    ours_cmp = (
                        ours_.cpu().to(ref_dtype) if ours_.dtype != ref_dtype else ours_.cpu()
                    )
                    refs_cmp = [
                        (r.cpu().to(ref_dtype) if r.dtype != ref_dtype else r.cpu())
                        for r in ref_runs_for_out
                    ]
                    m = classify_output_t3(ours_cmp, refs_cmp)
                    m["output_idx"] = j
                    m["shape"] = list(refs_cmp[0].shape)
                    per_out.append(m)
                    verdicts.append(m.get("verdict", "EVAL_ERR"))
            else:
                for j, (cpu_, cann_, ours_) in enumerate(zip(cpu_t, cann_t, ours_t)):
                    ref_dtype = cpu_.dtype
                    cann_cmp = cann_.cpu().to(ref_dtype) if cann_.dtype != ref_dtype else cann_.cpu()
                    ours_cmp = ours_.cpu().to(ref_dtype) if ours_.dtype != ref_dtype else ours_.cpu()
                    # Pass the candidate's ORIGINAL (pre-cast) dtype so the fp16-aware
                    # coarser-dtype threshold selection sees fp16 (ours_cmp is already
                    # cast to cpu_truth dtype above for comparable error MATH).
                    # BENCHMARK = 商用 scenario: model.py IS the live CANN op (a real vendor 标杆),
                    # so route="commercial" — T1 via vendored compare.py (生态 metric, fixes the
                    # drifted near-zero) + the vs-CANN T2 parity fallback (preserves benchmark's
                    # competitor-based small-value behavior; no near-zero regression, no native needed).
                    m = classify_output(ours_cmp, cann_cmp, cpu_, cand_orig_dtype=ours_.dtype,
                                        route="commercial")
                    m["output_idx"] = j
                    m["shape"] = list(cpu_.shape)
                    m["tier_axis"] = "T1_T2"
                    per_out.append(m)
                    verdicts.append(m.get("verdict", "EVAL_ERR"))

            # Case verdict = worst-of-outputs
            severity = {
                "PASS_T1": 0, "PASS_T2": 1, "PASS_T3": 1,
                "FAIL": 2, "EVAL_ERR": 3,
            }
            case_verdict = max(verdicts, key=lambda v: severity.get(v, 99))
            results.append({
                "case": idx,
                "verdict": case_verdict,
                "outputs": per_out,
                "tier_axis": op_tier_axis,
            })

            if verbose:
                summ = ", ".join(
                    f"o{m['output_idx']} {m.get('verdict', '?')} "
                    f"ours_M={m.get('ours_mere', 0):.2e}/{m.get('ours_mare', 0):.2e} "
                    f"cann_M={m.get('cann_mere', 0):.2e}/{m.get('cann_mare', 0):.2e}"
                    for m in per_out
                )
                print(f"case[{idx}]: {case_verdict}  {summ}")

        except Exception as e:
            # fix(a) three-way (2026-05-29, main coverage-fraud guard): a kernel that
            # declares a case OUT-OF-SCOPE by raising the TYPED `_OutOfScope` sentinel
            # (with reason — shape > declared physical cap e.g. S*D>UB, OR feature not
            # supported) is SKIPPED, NOT failed: there is no in-scope contract to verify.
            # Any OTHER exception is a real eval failure → EVAL_ERR. NEVER fold a real
            # fail/eval-error into skip (that would be coverage-fraud). The skip is
            # legitimate ONLY because the kernel itself declared the scope boundary.
            if _is_out_of_scope_raise(e):
                results.append({"case": idx, "verdict": "SKIP_OOS",
                                "error": f"{type(e).__name__}: {e}"})
                if verbose:
                    print(f"case[{idx}]: SKIP_OOS (kernel-declared out-of-scope)  {e}")
            else:
                results.append({"case": idx, "verdict": "EVAL_ERR", "error": f"{type(e).__name__}: {e}"})
                if verbose:
                    print(f"case[{idx}]: EVAL_ERR  {type(e).__name__}: {e}")
                    traceback.print_exc()

    # Aggregate
    n_total = len(results)
    n_pass_t1 = sum(1 for r in results if r["verdict"] == "PASS_T1")
    n_pass_t2 = sum(1 for r in results if r["verdict"] == "PASS_T2")
    n_pass_t3 = sum(1 for r in results if r["verdict"] == "PASS_T3")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_err = sum(1 for r in results if r["verdict"] == "EVAL_ERR")
    # fix(a) Layer-A honesty data (2026-05-29): SKIP_OOS = kernel-declared
    # out-of-scope (typed _OutOfScope). Excluded from the in-scope pass/fail
    # denominator (skip != pass AND skip != fail), but RECORDED so the
    # declared-subset coverage is VISIBLE (anti-masquerade). The full
    # PARTIAL-vs-full-pass contract-completeness verdict is Layer B (main's
    # CONTRACT_COMPLETENESS_GATE_DESIGN); Layer A surfaces the data.
    n_skip_oos = sum(1 for r in results if r["verdict"] == "SKIP_OOS")
    n_in_scope = n_total - n_skip_oos
    coverage_pct = round(100.0 * n_in_scope / n_total, 1) if n_total else 0.0
    beats_cann = sum(
        1 for r in results
        if r.get("outputs") and any(o.get("beats_cann", False) for o in r["outputs"])
    )

    # Op-level verdict — per OL-109 tier axes are exclusive (an op is EITHER
    # on T1/T2 axis OR on T3 axis based on whether CPU truth was available).
    # Op_tier_axis sticky from first case decides the verdict family.
    # fix(a): SKIP_OOS excluded from pass/fail; verdict is over the IN-SCOPE set.
    if n_in_scope == 0 and n_skip_oos > 0:
        op_verdict = "ALL_OOS"          # every case kernel-declared OOS; nothing verified
    elif n_in_scope > 0 and n_err == n_in_scope:
        op_verdict = "ERROR"
    elif op_tier_axis == "T3":
        if n_fail == 0 and n_err == 0:
            op_verdict = "OVERALL_T3"
        elif n_pass_t3 > 0:
            op_verdict = "PARTIAL"
        else:
            op_verdict = "FAIL"
    else:  # T1_T2 axis (default)
        if n_fail == 0 and n_err == 0:
            op_verdict = "OVERALL_T1" if n_pass_t2 == 0 else "OVERALL_T2"
        elif (n_pass_t1 + n_pass_t2) > 0:
            op_verdict = "PARTIAL"
        else:
            op_verdict = "FAIL"
    # fix(a) anti-masquerade (Layer A): an in-scope OVERALL with OOS skips present is
    # NOT a full-contract pass (declared-subset coverage). Flag it so no downstream
    # reads OVERALL as complete. The formal PARTIAL-vs-full-pass contract-completeness
    # verdict is Layer B (CONTRACT_COMPLETENESS_GATE_DESIGN, main).
    if n_skip_oos > 0 and op_verdict.startswith("OVERALL"):
        op_verdict = op_verdict + "_IN_SCOPE_ONLY"

    summary = {
        "op": op_name,
        "archive": str(archive_dir),
        "metric": (
            "two-tier MERE/MARE: T1=vs-CPU-thresholds, T2=ours≤CANN(both vs CPU)"
            if op_tier_axis != "T3"
            else "T3 MERE/MARE vs CANN-on-NPU (CPU truth structurally undefined; P0zz)"
        ),
        "tier_axis": op_tier_axis or "T1_T2",
        "reference_source": op_reference_source or "CPU+fp64",
        "thresholds": {str(k).replace("torch.", ""): v for k, v in PRECISION_THRESHOLDS.items()},
        "n_total": n_total,
        "n_pass_t1": n_pass_t1,
        "n_pass_t2": n_pass_t2,
        "n_pass_t3": n_pass_t3,
        "n_fail": n_fail,
        "n_err": n_err,
        # fix(a) honesty data: OOS skips + in-scope coverage (anti-masquerade).
        "n_skip_oos": n_skip_oos,
        "n_in_scope": n_in_scope,
        "coverage_pct": coverage_pct,
        "in_scope_only": n_skip_oos > 0,
        "beats_cann": beats_cann,
        "op_verdict": op_verdict,
        "results": results,
    }

    if verbose:
        print(f"\n--- Summary: {op_name} ---")
        print(f"  total: {n_total}")
        print(f"  tier_axis: {op_tier_axis or 'T1_T2'} (reference_source: {op_reference_source or 'CPU+fp64'})")
        if op_tier_axis == "T3":
            print(f"  PASS_T3 (T1 thresholds vs CANN-on-NPU):  {n_pass_t3}")
            print(f"  FAIL    (worse than CANN-on-NPU):        {n_fail}")
            print(f"  EVAL_ERR:                                {n_err}")
            print(f"  Op verdict: {op_verdict}  (CPU truth structurally undefined)")
        else:
            print(f"  PASS_T1 (strict, vs CPU thresholds):    {n_pass_t1}")
            print(f"  PASS_T2 (relative, ≤ CANN MERE/MARE):   {n_pass_t2}")
            print(f"  FAIL    (worse than CANN):              {n_fail}")
            print(f"  EVAL_ERR:                               {n_err}")
            print(f"  beats CANN (MERE<, MARE≤):              {beats_cann}")
            print(f"  Op verdict: {op_verdict}")

    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive_dir")
    p.add_argument("--json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    s = evaluate(Path(args.archive_dir).resolve(), verbose=not args.quiet)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, default=str))
        print(f"wrote json: {args.json}")
    # Exit 0 only if no FAIL or ERR; PASS_T1 + PASS_T2 both acceptable
    return 0 if (s["n_fail"] == 0 and s["n_err"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
