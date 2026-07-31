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

"""CDV precision grader for BACKWARD ops (owner-directed 2026-06-13, new method).

Consumes per-case error metrics (vs fp64 CPU autograd golden) produced by the
in-container cdv_collect step, applies the §4.6 contract-driven verdict:

  * validator selected DECLARATIVELY by (op_class × available_refs × tier) —
    NOT chosen here (anti-metric-shop). Uses contract_validator_poc.select_validator.
  * REPRESENTATIVE stream (profile=randn, large sample) → STATISTICAL 达标:
    bootstrap-median + 95% CI of MERE and MARE vs the selected threshold
    (NOT per-case hard-fail — that was the FA 20/68 artifact).
  * EDGE stream (zeros/large/small/boundary) → bug-finding, reported SEPARATELY,
    NOT counted in the precision pass/fail.
  * Emits verification.json-ready precision block with criterion_provenance.
  * FAIL-CLOSED: missing/incomplete provenance, or a non-declarative validator,
    or no representative cases → verdict REJECT.

T2 (double-baseline ratio, ADDITIVE — owner-requested "需要补", 2026-06-14): in addition to
the T1 verdict above (which stays STRICT + UNCHANGED), emit a SEPARATE `pass_a_t2` block that
grades the canonical v_double_baseline_ratio (contract_validator_poc, RATIO_THRESH v2.1 §4.5.1)
on per-record baseline_* data from same-precision CPU torch autograd
(competitor_kind=torch_same_dtype_cpu).
T2 is reported ALONGSIDE T1 as additional evidence; the OVERALL `verdict` is driven by T1 ONLY
(T2 never flips T1's status). Graceful N/A when competitor data is absent.

Usage:
  python3 cdv_grade.py --errors per_case_errors.json --op mul_grad \
      --op-class elementwise --tier L1 --refs fp64_golden --out cdv_precision.json
"""
from __future__ import annotations
import argparse
import json
import sys
import os
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # same dir as contract_validator_poc
from contract_validator_poc import (  # noqa: E402
    Contract, select_validator, VALIDATORS, SAME_DTYPE_THR, bootstrap_median_ci,
    grade_contract, RATIO_THRESH, v_double_baseline_ratio, _ratio,
)
from precision_cannbot_adapter import grade_batch  # noqa: E402  (cannbot single judge — owner 2026-06-18 完全照抄)

REPRESENTATIVE_PROFILES = {"randn", "representative"}


def _as_np(x):
    """Coerce one per-output grad (torch.Tensor | np.ndarray | array-like) to a RAW
    np.ndarray for the cannbot adapter. grade_batch casts dtype + computes MARE/MERE/
    ratio internally, so do NOT pre-reduce or floor here. None → None (生态 absent-competitor)."""
    if x is None:
        return None
    if hasattr(x, "detach"):            # torch.Tensor (cpu/npu) → host np
        x = x.detach().cpu()
        if "bfloat16" in str(getattr(x, "dtype", "")):  # numpy has no bf16 → lossless upcast (fp32 ⊃ bf16)
            x = x.float()
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _build_record_cases(npu_grads, golden_grads, competitor_grads, wrt, dtype, profile,
                        competitor_kind=None, native_grads=None):
    """PURE (CPU-unit-testable) per-record → cannbot `cases` mapper.

    No NPU / no autograd here: maps ALREADY-COMPUTED per-output arrays to one cannbot
    case dict per (output × record) for `grade_batch`. The kernel-run + same-precision-CPU
    competitor happen in the VERIFY-side caller (backward_cdv_collect.cases_from_records),
    which is NPU/torch-runtime; this case-shape + is_edge logic is isolated here so it is
    testable without a container (owner 2026-06-18: 完全照抄 cannbot — same case dict the
    collect-side collect_cases emits, ONE judge).

      npu_grads        wrt-ordered seq — raw kernel output per output
      golden_grads     dict name->fp64 truth (record["grads"]) OR wrt-ordered seq
      competitor_grads wrt-ordered independent comparator sequence OR None
      native_grads     wrt-ordered seq — the CPU-SAME-PRECISION reference for the 生态 (DEFAULT)
                       compare.py small-value/cancellation carve-out. Phase-2 wiring: this is the
                       same-dtype CPU torch autograd backward (the genuine native_output compare.py
                       wants). Emitted under the `native` case key + native_kind="cpu_same_precision"
                       (the adapter's provenance guard only honors that tag). None ⇒ carve-out strict.
      wrt              list[str] output names (defines order + the case `output` label)
      dtype            str
      profile          value-profile str → is_edge = profile NOT representative
                       (cannbot excludes is_edge from the statistical verdict; bug-find only)
      competitor_kind  provenance tag for this record's comparator
                       ("torch_same_dtype_cpu" | None). Threaded
                       into each case dict so the verdict block records which 标杆 graded it
                       (gate (e) competitor_kind honest).
    """
    is_edge = profile not in REPRESENTATIVE_PROFILES
    cases = []
    for oi, wname in enumerate(wrt):
        g = golden_grads[wname] if isinstance(golden_grads, dict) else golden_grads[oi]
        tp = None if competitor_grads is None else competitor_grads[oi]
        nat = None if native_grads is None else native_grads[oi]
        cases.append({
            "npu": _as_np(npu_grads[oi]),
            "golden": _as_np(g),
            "third_party": _as_np(tp),
            # 生态 DEFAULT native_output (CPU-same-precision) + its provenance guard tag.
            "native": _as_np(nat),
            "native_kind": "cpu_same_precision" if nat is not None else None,
            "dtype": dtype,
            "output": wname,
            "is_edge": is_edge,
            "competitor_kind": competitor_kind,
        })
    return cases


def _stat_verdict_ecosystem(recs, dtype):
    """Statistical 达标 for single/same-dtype threshold validators (absolute MERE/MARE).
    PASS ⟺ bootstrap-median CI-upper(MERE) < thr AND CI-upper(MARE) < 10·thr."""
    thr = SAME_DTYPE_THR.get(dtype, 2 ** -13)
    mere = [r["ours_mere"] for r in recs if r.get("ours_mere") is not None]
    mare = [r["ours_mare"] for r in recs if r.get("ours_mare") is not None]
    bm = bootstrap_median_ci(mere)
    ba = bootstrap_median_ci(mare)
    if bm is None or ba is None:
        return None
    mere_ok = bm["ci"][1] < thr
    mare_ok = ba["ci"][1] < 10 * thr
    return {
        "is_pass": bool(mere_ok and mare_ok),
        "thr": thr, "mare_thr": 10 * thr,
        "mere_bootstrap": bm, "mare_bootstrap": ba,
        "mere_ci_upper_ok": mere_ok, "mare_ci_upper_ok": mare_ok,
        "n_representative": len(recs),
    }


def _has_competitor(rec):
    """A record carries T2 comparator data iff all baseline fields are populated."""
    return all(rec.get(f"baseline_{m}") is not None for m in ("mere", "mare", "rmse"))


def grade_t2_double_baseline(errors, op, tier):
    """T2 (double-baseline ratio) verdict — ADDITIVE, SEPARATE from T1.

    Reuses the CANONICAL contract_validator_poc.grade_contract / v_double_baseline_ratio
    (RATIO_THRESH v2.1 §4.5.1: L0(10,2,2)/L1(5,1.5,1.5)/L2(2,1.2,1.2) for mare/mere/rmse).
    The ratio denominator comes from per-record baseline_* fields produced by the
    same-precision CPU torch-autograd comparator.

    Forces op_class="numerically_hard" + refs=(fp64_golden, independent_baseline) so select_validator
    routes to double_baseline_ratio (the WHOLE POINT of T2 — does fp32 meet vendor/competitor-
    equivalent?). T1's declarative selection is UNTOUCHED by this. Per-dtype: ratio verdict +
    bootstrap-median-ratio CI (the canonical grade_contract aggregate). Graceful N/A when the
    competitor is absent (collector run pre-T2 / competitor failed to compute).
    """
    rep = [r for r in errors if r.get("profile") in REPRESENTATIVE_PROFILES and _has_competitor(r)]
    if not rep:
        return {
            "status": "N/A",
            "reason": ("no representative records carry T2 competitor (标杆) data "
                       "(baseline_{mere,mare,rmse}); collector ran pre-T2 or comparator compute failed"),
            "tier": tier,
        }
    competitor_kinds = sorted({r.get("competitor_kind") for r in rep if r.get("competitor_kind")})
    dtypes = sorted({r["dtype"] for r in rep})
    per_dtype = {}
    all_pass = True
    for dt in dtypes:
        rep_dt = [r for r in rep if r["dtype"] == dt]
        # CANONICAL ratio grade via the PoC (numerically_hard + fp64_golden + independent_baseline
        # → double_baseline_ratio). Selection is forced by the contract, not chosen here.
        c = Contract(op=op, op_class="numerically_hard", dtype=dt, tier=tier,
                     refs_available=("fp64_golden", "independent_baseline"))
        audit = grade_contract(rep_dt, c, case_class="representative")
        if audit.validator != "double_baseline_ratio":
            raise ValueError(
                f"T2 expected double_baseline_ratio, got {audit.validator}"
            )
        # CI-upper of the bootstrap-median ratio vs RATIO_THRESH[tier] (locked aggregate
        # verdict, mirrors contract_e2e #3: pass iff CI-upper <= thr for mare AND mere AND rmse).
        mt, et, rt = RATIO_THRESH[tier]
        thr = {"mare": mt, "mere": et, "rmse": rt}
        boot = audit.bootstrap_median_ratio
        ci_ok = {}
        for m in ("mare", "mere", "rmse"):
            bm = boot.get(m)
            ci_ok[m] = bool(bm is not None and bm["ci"][1] <= thr[m])
        dt_pass = all(ci_ok.values())
        all_pass = all_pass and dt_pass
        per_dtype[dt] = {
            "is_pass": dt_pass,
            "tier": tier,
            "ratio_thresholds": thr,
            "bootstrap_median_ratio": boot,
            "ci_upper_within_thr": ci_ok,
            "n_cases": audit.n_cases,
            "n_pass_per_case": audit.n_pass,
        }
    return {
        "status": "PASS" if all_pass else "FAIL",
        "tier": tier,
        "validator": "double_baseline_ratio",
        "competitor_kind": competitor_kinds[0] if competitor_kinds else "unknown",
        "competitor_caveat": ("T2 uses a same-precision CPU torch-autograd comparator; "
                              "it is an independent numerical baseline, not a hardware route."),
        "ratio_thresholds_source": "contract_validator_poc.RATIO_THRESH (v2.1 §4.5.1)",
        "per_dtype": per_dtype,
        "all_dtypes_pass": all_pass,
        "method": ("canonical v_double_baseline_ratio via grade_contract; ratio = ours_err / "
                   "competitor_err vs the SAME fp64 golden; PASS ⟺ bootstrap-median-ratio CI-upper "
                   "<= RATIO_THRESH[tier] for mare AND mere AND rmse, per-dtype."),
    }


def grade(errors, op, op_class, tier, refs):
    dtypes = sorted({r["dtype"] for r in errors})
    rep = [r for r in errors if r.get("profile") in REPRESENTATIVE_PROFILES]
    edge = [r for r in errors if r.get("profile") not in REPRESENTATIVE_PROFILES]

    # ---- declarative validator selection (per dtype; the policy reads op_class/refs/tier) ----
    sel_validator, sel_rule = None, None
    per_dtype = {}
    for dt in dtypes:
        c = Contract(op=op, op_class=op_class, dtype=dt, tier=tier, refs_available=tuple(refs))
        vkey, rule = select_validator(c)
        sel_validator, sel_rule = vkey, rule  # same across dtypes (selection is dtype-independent here)
        rep_dt = [r for r in rep if r["dtype"] == dt]
        if vkey in ("single_baseline_threshold", "same_dtype_threshold"):
            sv = _stat_verdict_ecosystem(rep_dt, dt)
        elif vkey == "bit_exact":
            n_nonzero = sum(1 for r in rep_dt if (r.get("ours_max_abs_diff") or 0) != 0)
            sv = {"is_pass": n_nonzero == 0, "rule": "bit_exact", "n_nonzero": n_nonzero,
                  "n_representative": len(rep_dt)}
        else:  # double_baseline_ratio handled by the PoC's grade_contract elsewhere
            sv = {"is_pass": None, "note": f"validator {vkey} not graded by this ecosystem helper"}
        per_dtype[dt] = sv

    # ---- edge stream: bug-finding only (flag anomalies; NOT a pass/fail count) ----
    edge_report = []
    for r in edge:
        thr = SAME_DTYPE_THR.get(r["dtype"], 2 ** -13)
        flagged = (r.get("ours_mere") is not None and r["ours_mere"] >= thr) or \
                  (r.get("ours_max_abs_diff") is not None and r["ours_max_abs_diff"] != r["ours_max_abs_diff"])  # NaN
        if flagged:
            edge_report.append({"profile": r.get("profile"), "dtype": r["dtype"], "shape": r.get("shape"),
                                "output": r.get("output"), "mere": r.get("ours_mere"), "mare": r.get("ours_mare"),
                                "max_abs_diff": r.get("ours_max_abs_diff")})

    rep_pass = all(v.get("is_pass") for v in per_dtype.values())

    # ---- T2 (double-baseline ratio) — ADDITIVE, SEPARATE; does NOT touch T1's rep_pass/verdict ----
    # Owner-requested ("需要补", 2026-06-14): formally answer "does fp32 meet T2
    # (vendor/competitor-equivalent)?" via the canonical v_double_baseline_ratio. T2 is reported
    # alongside T1 as additional evidence; T1 stays the PRIMARY gate (T2 never flips T1's status).
    t2 = grade_t2_double_baseline(errors, op, tier)

    # ---- fail-closed provenance + verdict ----
    provenance = {
        "validator": sel_validator,
        "selection_rule": sel_rule,
        "selected_declaratively": True,
        "references": list(refs),
        "reference_note": ("fp64 CPU torch.autograd.grad golden; when no independent baseline is "
                           "declared, policy selects the ecosystem single-baseline validator"),
        "tier": tier, "op_class": op_class,
        "case_split": {"representative_profiles": sorted(REPRESENTATIVE_PROFILES),
                       "n_representative": len(rep), "n_edge": len(edge)},
        "method": ("representative stream: bootstrap-median + 95% CI of MERE/MARE vs declarative "
                   "threshold (statistical 达标, NOT per-case hard-fail); edge stream: bug-finding, "
                   "reported separately, NOT counted."),
    }
    # fail-closed checks
    fc = []
    if sel_validator is None or sel_rule is None:
        fc.append("no validator selected")
    if not refs:
        fc.append("no references declared")
    if len(rep) == 0:
        fc.append("no representative cases (cannot form statistical verdict)")
    verdict = "REJECT_FAIL_CLOSED" if fc else ("PASS" if rep_pass else "FAIL")

    return {
        "op": op,
        "verdict": verdict,  # T1 primary gate — UNCHANGED; T2 never flips this.
        "fail_closed_reasons": fc,
        # T1 (existing, strict same-dtype absolute threshold) — UNCHANGED.
        "representative_statistical": {"per_dtype": per_dtype, "all_dtypes_pass": rep_pass},
        # T2 (NEW, additive double-baseline ratio vs same-precision-CPU competitor).
        "pass_a_t2": t2,
        "edge_bug_findings": {"n_flagged": len(edge_report), "flagged": edge_report,
                              "note": "edge cases are bug-finding, NOT counted in pass/fail"},
        "criterion_provenance": provenance,
    }


def _persist_raw_outputs(cases, res, path, *, op, cap=200):
    """Persist a BOUNDED subset of raw per-case arrays for offline re-grade (Part B / criterion 3).

    Saves npu/golden/native(+native_kind)/dtype/output/is_edge/competitor_kind so a future re-grade
    under a corrected metric needs NO NPU re-run.

    CAP POLICY (main gate, keep-failures-first): the re-grade VALUE is in the FAILING cases — a cap
    that drops a failing case defeats the harness-defect fix. So when len(cases) > cap, persist in
    PRIORITY order and TOTAL-bound at `cap`:
      1. FAILING representative cases (is_pass False/None — exactly what a corrected-metric re-grade needs),
      2. ALL edge cases (bug-find stream, high-signal),
      3. then PASSING representative cases to fill the remaining budget.
    `res` = the grade_batch result (its per_case/edge carry the per-case is_pass, aligned in order to
    the representative/edge cases). A representative case whose verdict is unknown (length mismatch) is
    treated as FAILING (kept) — fail-safe toward preservation.
    """
    import torch  # local import: keep module import CPU/numpy-clean for the pure unit-test path
    rep = [c for c in cases if not c.get("is_edge")]
    edge = [c for c in cases if c.get("is_edge")]
    rep_pass = [bool(r.get("is_pass")) for r in (res.get("per_case") or [])]

    rep_fail, rep_ok = [], []
    for i, c in enumerate(rep):
        # unknown verdict (i >= len(rep_pass)) → treat as failing (preserve)
        (rep_ok if (i < len(rep_pass) and rep_pass[i]) else rep_fail).append(c)

    # priority order, TOTAL-bounded at cap (single total bound, not per-stream)
    prioritized = rep_fail + edge + rep_ok
    kept = prioritized[:cap]
    n_fail_total = len(rep_fail)
    # identity-based membership (dicts hold numpy arrays → `in`/`==` would raise ambiguous-truth)
    _fail_ids = {id(c) for c in rep_fail}
    n_fail_kept = sum(1 for c in kept if id(c) in _fail_ids)

    def _payload(c):
        payload = {}
        for key in (
            "npu",
            "golden",
            "native",
            "native_kind",
            "dtype",
            "output",
            "is_edge",
            "competitor_kind",
        ):
            payload[key] = c.get(key)
        return payload

    blob = {
        "schema": "backward_npu_outputs_v1",
        "op": op,
        "policy": (f"keep-failures-first, TOTAL cap={cap}: failing rep + all edge + passing rep "
                   "(re-grade offline, no NPU re-run)"),
        "n_total": len(cases), "n_representative": len(rep), "n_edge": len(edge),
        "n_failing_representative": n_fail_total, "n_failing_persisted": n_fail_kept,
        "n_persisted": len(kept), "capped": len(prioritized) > cap,
        "cases": [_payload(c) for c in kept],
    }
    torch.save(blob, path)
    return {"path": str(path), "schema": blob["schema"], "n_persisted": len(kept),
            "n_total": len(cases), "capped": blob["capped"],
            "n_failing_representative": n_fail_total, "n_failing_persisted": n_fail_kept,
            "all_failures_kept": n_fail_kept == n_fail_total,
            "policy": blob["policy"]}


def grade_cases(cases, op, *, op_class="float", precision_level="L1",
                persist_outputs_path=None, persist_cap=200):
    """Cannbot SINGLE-JUDGE backward grading — DEFAULT = 生态 cann-bench compare.py.

    Backward's in-process verify routes its per-(output×case) RAW arrays through the
    SAME canonical adapter `grade_batch` that pass_a / benchmark / port_a3 use — one
    source of truth, no backward-private statistics. `cases`: list of
    {npu, golden, native, native_kind, third_party, dtype, is_edge} (raw np.ndarray per
    output×case):
      * DEFAULT 生态 (vendored compare.py): golden=fp64 CPU truth; `native` = CPU-SAME-PRECISION
        reference (the §135 same-dtype CPU torch autograd backward, native_kind="cpu_same_precision")
        feeds compare.py's small-value/cancellation carve-out. grade_cases passes route by default
        ("ecosystem") — `third_party` is NOT consumed here.
      * OPTIONAL 商用 ratio is a separate route ("commercial") and is NOT the default — `third_party`
        (the 标杆) only matters there.
      * is_edge=True ⇒ excluded from the statistical verdict (bug-find stream only).

    PERSIST (Part B / main criterion 3): when `persist_outputs_path` is given, a BOUNDED subset of
    the raw arrays (npu/golden/native/dtype/output/is_edge/competitor_kind) is torch.saved so a
    failed op can be RE-GRADED OFFLINE under a corrected metric with NO NPU re-run. The cap is
    TOTAL-bounded and KEEP-FAILURES-FIRST (failing rep + all edge + passing rep) — persisted AFTER
    grading so the per-case verdicts drive the priority. Without this the archive stored only derived
    scalars (mere/mare) → offline re-grade impossible.

    FAIL-CLOSED: empty cases / no representative / inconclusive (is_pass None) → REJECT
    (never fabricate PASS). Returns a verification.json-ready block; `cannbot` carries the
    canonical grade_batch result (verdict_basis/bootstrap_valid/ci_upper/gate/scenario/
    pass_rate/n_representative/n_edge/per_case/edge).
    """
    _prov_base = {
        "standard": "cann-bench 生态 compare.py (vendored) via "
                    "precision_cannbot_adapter.grade_batch",
        "single_source_of_truth": "shared grade_batch with pass_a/benchmark/port_a3",
        "default_route": "ecosystem (生态 compare.py); native=CPU-same-precision; "
                         "third_party/商用 ratio is opt-in only",
        "op_class": op_class, "precision_level": precision_level,
    }
    if not cases:
        return {"op": op, "verdict": "REJECT_FAIL_CLOSED",
                "fail_closed_reasons": ["no cases collected (cannot verify)"],
                "cannbot": None, "criterion_provenance": _prov_base}
    res = grade_batch(cases, op_class=op_class, precision_level=precision_level)
    # Persist AFTER grading so the keep-failures-first cap can prioritize by per-case verdict.
    if persist_outputs_path:
        try:
            _prov_base["persisted_outputs"] = _persist_raw_outputs(
                cases, res, persist_outputs_path, op=op, cap=persist_cap)
        except Exception as e:  # persistence is best-effort; never block the verdict on it
            _prov_base["persist_error"] = f"{type(e).__name__}: {e}"
    fc = []
    if res.get("n_representative", 0) == 0:
        fc.append("no representative cases (cannot form statistical verdict)")
    if res.get("is_pass") is None and not fc:
        fc.append(f"inconclusive verdict (verdict_basis={res.get('verdict_basis')})")
    verdict = "REJECT_FAIL_CLOSED" if fc else ("PASS" if res["is_pass"] else "FAIL")
    return {
        "op": op,
        "verdict": verdict,
        "fail_closed_reasons": fc,
        "cannbot": res,
        "criterion_provenance": {**_prov_base, "scenario": res.get("scenario"),
                                 "verdict_basis": res.get("verdict_basis"),
                                 "selected_declaratively": True},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--op-class", required=True, choices=["int", "bool", "elementwise", "numerically_hard"])
    ap.add_argument("--tier", default="L1", choices=["L0", "L1", "L2"])
    ap.add_argument("--refs", action="append", default=[],
                    help="repeatable: fp64_golden / independent_baseline / same_dtype_vendor / model_forward")
    ap.add_argument("--out", default="cdv_precision.json")
    a = ap.parse_args()
    refs = a.refs or ["fp64_golden"]
    with open(a.errors, encoding="utf-8") as errors_file:
        errors = json.load(errors_file)
    result = grade(errors, a.op, a.op_class, a.tier, refs)
    with open(a.out, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["verdict"] == "PASS" else 1)
