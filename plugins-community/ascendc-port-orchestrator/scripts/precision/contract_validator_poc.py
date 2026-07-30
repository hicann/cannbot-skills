# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Contract-driven precision-validator — PoC (owner-directed 2026-06-13).

Implements the verify side of docs/design/CONTRACT_DRIVEN_GENERATION_DESIGN.md §4.6:
  (A) pluggable PrecisionValidator strategies
  (B) declarative, contract-driven selection (NOT agent-choosable) — anti-metric-shop
  (C) audit record

This is a PoC: validators operate on per-case ERROR METRICS (mare/mere/rmse of each
engine vs the high-precision golden), which is the shape produced by the reference
pipeline (e.g. representative_fa_errors.json). A production wiring computes those
metrics from tensors via a shared metrics helper + the cannbot small-value/INF-NAN
companions; the SELECTION + AUDIT logic shown here is the load-bearing part.

Grounded in: cannbot ops-precision-standard (= PRECISION_STANDARD_v2.1.md).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Callable

# ---- L0/L1/L2 commercial ratio thresholds (mare, mere, rmse) — cannbot/Ascend 2.1 §4.5.1 ----
RATIO_THRESH = {"L0": (10.0, 2.0, 2.0), "L1": (5.0, 1.5, 1.5), "L2": (2.0, 1.2, 1.2)}
# same-dtype absolute thresholds (MERE<thr, MARE<10thr) — NPUKernelBench-style
SAME_DTYPE_THR = {"float16": 2 ** -10, "bfloat16": 2 ** -7, "float32": 2 ** -13}


@dataclass
class Contract:
    op: str
    op_class: str        # "int" | "elementwise" | "numerically_hard"
    dtype: str
    tier: str = "L1"     # L0/L1/L2 (importance; default L1 for LLM-class)
    refs_available: tuple = ()   # subset of {"fp64_golden","independent_baseline","same_dtype_vendor","model_forward"}


def _ratio(npu, ref):
    if ref > 0:
        return npu / ref
    return 1.0 if npu == 0 else float("inf")


# ---------------- (A) Validator strategies: grade(ours, refs, contract) -> dict ----------------
# `ours` / `refs[k]` are dicts {mare, mere, rmse} (errors vs the high-precision golden),
# except bit_exact which takes {max_abs_diff}.

def v_bit_exact(ours, refs, c):
    ok = ours.get("max_abs_diff", 1) == 0
    return {"is_pass": ok, "rule": "bit_exact", "detail": {"max_abs_diff": ours.get("max_abs_diff")}}


def v_same_dtype_threshold(ours, refs, c):
    thr = SAME_DTYPE_THR.get(c.dtype, 2 ** -13)
    ok = ours["mere"] < thr and ours["mare"] < 10 * thr
    return {"is_pass": ok, "rule": "same_dtype_threshold",
            "detail": {"mere": ours["mere"], "mare": ours["mare"], "thr": thr}}


def v_double_baseline_ratio(ours, refs, c):
    bench = refs["independent_baseline"]   # ratio denominator
    mt, et, rt = RATIO_THRESH[c.tier]
    mr = _ratio(ours["mare"], bench["mare"])
    er = _ratio(ours["mere"], bench["mere"])
    rr = _ratio(ours["rmse"], bench["rmse"])
    ok = mr <= mt and er <= et and rr <= rt
    return {"is_pass": ok, "rule": "double_baseline_ratio",
            "detail": {"tier": c.tier, "mare_ratio": round(mr, 3), "mere_ratio": round(er, 3), "rmse_ratio": round(rr, 3),
                       "thresholds": {"mare": mt, "mere": et, "rmse": rt}}}


def v_single_baseline_threshold(ours, refs, c):
    thr = SAME_DTYPE_THR.get(c.dtype, 2 ** -13)
    ok = ours["mere"] < thr and ours["mare"] < 10 * thr
    return {"is_pass": ok, "rule": "single_baseline_threshold(ecosystem)",
            "detail": {"mere": ours["mere"], "mare": ours["mare"], "thr": thr}}


VALIDATORS: dict[str, Callable] = {
    "bit_exact": v_bit_exact,
    "same_dtype_threshold": v_same_dtype_threshold,
    "double_baseline_ratio": v_double_baseline_ratio,
    "single_baseline_threshold": v_single_baseline_threshold,
}


# ---------------- (B) Declarative selection — contract-driven, NOT agent-choosable ----------------
def select_validator(c: Contract) -> tuple[str, str]:
    """Return (validator_key, selection_rule). Fixed policy table; the agent/op does NOT choose."""
    if c.op_class in ("int", "bool"):
        return "bit_exact", "int/bool -> bit_exact"
    if c.op_class == "elementwise" and "model_forward" in c.refs_available:
        return "same_dtype_threshold", "elementwise + same-dtype model.forward -> same_dtype_threshold"
    if c.op_class == "numerically_hard":
        if "fp64_golden" in c.refs_available and "independent_baseline" in c.refs_available:
            return (
                "double_baseline_ratio",
                "numerically_hard + fp64_golden + independent baseline -> double_baseline_ratio",
            )
        if "same_dtype_vendor" in c.refs_available:
            return "same_dtype_threshold", "numerically_hard + same_dtype_vendor -> same_dtype_threshold (vs vendor; circular — declare in audit)"
        if "fp64_golden" in c.refs_available:
            return "single_baseline_threshold", "numerically_hard + fp64_golden only (no 标杆) -> single_baseline_threshold (ecosystem)"
    # default: most conservative achievable
    if "model_forward" in c.refs_available:
        return "same_dtype_threshold", "default same-dtype"
    return "single_baseline_threshold", "default single-baseline"


# ---------------- (C) Audit record + statistical aggregate verdict ----------------
@dataclass
class PrecisionAudit:
    op: str
    validator: str
    selection_rule: str
    tier: str
    references: list
    case_class: str           # "representative" | "edge"
    n_cases: int
    n_pass: int
    bootstrap_median_ratio: dict = field(default_factory=dict)  # {mare:..,mere:..,rmse:..} + CI (representative+ratio only)


def bootstrap_median_ci(vals, n_boot=2000, cl=0.95, seed=0):
    import random
    import statistics
    vals = [v for v in vals if v != float("inf")]
    if not vals:
        return None
    rng = random.Random(seed)
    meds = []
    for _ in range(n_boot):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        meds.append(statistics.median(s))
    meds.sort()
    lo = meds[int((1 - cl) / 2 * n_boot)]
    hi = meds[int((1 + cl) / 2 * n_boot)]
    return {"median": round(statistics.median(vals), 4), "ci": [round(lo, 4), round(hi, 4)]}


def grade_contract(per_case: list[dict], contract: Contract, case_class: str = "representative") -> PrecisionAudit:
    """per_case[i] = {ours_{mare,mere,rmse}, baseline_{...}, (max_abs_diff), dtype}. Selection is FIXED by contract."""
    vkey, rule = select_validator(contract)
    fn = VALIDATORS[vkey]
    n_pass = 0
    ratios = {"mare": [], "mere": [], "rmse": []}
    for rec in per_case:
        ours = {"mare": rec.get("ours_mare"), "mere": rec.get("ours_mere"), "rmse": rec.get("ours_rmse"),
                "max_abs_diff": rec.get("ours_max_abs_diff")}
        refs = {
            "independent_baseline": {
                "mare": rec.get("baseline_mare"),
                "mere": rec.get("baseline_mere"),
                "rmse": rec.get("baseline_rmse"),
            }
        }
        res = fn(ours, refs, contract)
        n_pass += int(res["is_pass"])
        if vkey == "double_baseline_ratio":
            for m in ("mare", "mere", "rmse"):
                ratios[m].append(
                    _ratio(rec[f"ours_{m}"], rec[f"baseline_{m}"])
                )
    boot = {m: bootstrap_median_ci(ratios[m]) for m in ratios} if vkey == "double_baseline_ratio" else {}
    return PrecisionAudit(op=contract.op, validator=vkey, selection_rule=rule, tier=contract.tier,
                          references=list(contract.refs_available), case_class=case_class,
                          n_cases=len(per_case), n_pass=n_pass, bootstrap_median_ratio=boot)


if __name__ == "__main__":
    import json
    import sys
    # Demo on FA representative data (representative_fa_errors.json)
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rep_fa_errors.json"
    with open(path) as input_file:
        data = json.load(input_file)
    c = Contract(op="flash_attention_score", op_class="numerically_hard", dtype="float16",
                 tier="L1", refs_available=("fp64_golden", "independent_baseline"))
    audit = grade_contract(data, c, case_class="representative")
    print("SELECTED:", audit.validator, "|", audit.selection_rule)
    print(json.dumps(asdict(audit), indent=2, default=str))
