#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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
on per-record `baseline_*` data from same-precision CPU torch autograd
(`competitor_kind=torch_same_dtype_cpu`).
T2 is reported ALONGSIDE T1 as additional evidence; the OVERALL `verdict` is driven by T1 ONLY
(T2 never flips T1's status). Graceful N/A when competitor data is absent.

Usage:
  python3 cdv_grade.py --errors per_case_errors.json --op mul_grad \
      --op-class elementwise --tier L1 --refs fp64_golden --out cdv_precision.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # same dir as contract_validator_poc
from contract_validator_poc import (  # noqa: E402
    Contract, select_validator, SAME_DTYPE_THR, bootstrap_median_ci,
    grade_contract, RATIO_THRESH,
)

REPRESENTATIVE_PROFILES = {"randn", "representative"}


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
    The ratio denominator comes from per-record `baseline_*` fields produced by the
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
            raise AssertionError(
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
                              "it is additive evidence and does not replace the fp64 primary gate."),
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
        "reference_note": ("fp64 CPU torch.autograd.grad golden; the declared CPU "
                           "same-precision comparator is used only for the additive T2 ratio"),
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
    with open(a.errors) as errors_file:
        errors = json.load(errors_file)
    result = grade(errors, a.op, a.op_class, a.tier, refs)
    with open(a.out, "w") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["verdict"] == "PASS" else 1)
