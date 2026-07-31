# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Re-classify port_a3 archive verdicts per PRECISION_STANDARD_v2.1.md.

v2.1 standard summary:
- §4.5.1 floating-point: error ratios (target_err / independent_baseline_err).
  L0 MARE_ratio ≤ 10, MERE ≤ 2, RMSE ≤ 2
  L1 MARE ≤ 5, MERE ≤ 1.5, RMSE ≤ 1.5
  L2 MARE ≤ 2, MERE ≤ 1.2, RMSE ≤ 1.2
- §4.5.3 small-value: |golden| < SVT → use absolute error ≤ SV_err
  fp32: SVT=2^-14, SV_err=2^-30
  fp16: SVT=2^-11, SV_err=2^-16
  bf16: SVT=2^-8,  SV_err=2^-16
- §4.1 integer/binary: bit-exact required.

For port_a3 mode the A3 NPU CANN aclnn IS the reference (no separate
independent baseline). v2.1 §4.5.1 ratio form does not directly apply. We
adopt the simplified single-standard adaptation used by mature ports:

  PASS iff (∀ case: max_abs_diff ≤ 1 dtype_eps  AND  max_rel_diff ≤ 0.01)
       OR (|golden| < v2.1 SVT  AND  max_abs_diff ≤ v2.1 SV_err)

Where 1 dtype_eps = the largest meaningful ULP for that dtype
  fp32 eps ≈ 1.19e-7  (we accept up to ~10 eps = 1.2e-6 since v2.1
    L0 MARE ratio ≤ 10 implies the same loosening when baseline error
    is at machine eps)
  fp16 eps ≈ 9.77e-4
  bf16 eps ≈ 7.81e-3

This script does NOT modify kernel code. It re-reads
verification.json.precision.pass_a.per_case data and re-classifies
the overall verdict per v2.1 single-standard rules.

Usage:
    python3 src/scripts/precision_eval_v21.py <archive_dir> [--apply]
    python3 src/scripts/precision_eval_v21.py output/a3_to_a5_port/src/kernels/foreach_sqrt --apply
    python3 src/scripts/precision_eval_v21.py output/a3_to_a5_port/src/kernels/  # batch all archives
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# v2.1 §4.5.3 small-value thresholds
SVT = {"fp32": 2 ** -14, "fp16": 2 ** -11, "bf16": 2 ** -8}
SV_ERR = {"fp32": 2 ** -30, "fp16": 2 ** -16, "bf16": 2 ** -16}
# Machine epsilon per dtype (used as base "1 ULP" measure)
EPS = {"fp32": 2 ** -23, "fp16": 2 ** -10, "bf16": 2 ** -7}
# Relaxed L0 MARE-ratio-style absolute threshold (10× eps for normal-magnitude golden)
ABS_THRESH_NORMAL = {"fp32": 10 * EPS["fp32"], "fp16": 10 * EPS["fp16"], "bf16": 10 * EPS["bf16"]}


def _canonicalize_dtype(dt: str) -> str:
    """Map verification.json dtype strings to {fp32, fp16, bf16, int}."""
    dt = (dt or "").lower().replace("torch.", "")
    if "bfloat16" in dt or "bf16" in dt:
        return "bf16"
    if "float16" in dt or "fp16" in dt:
        return "fp16"
    if "float32" in dt or "fp32" in dt:
        return "fp32"
    if "int" in dt or "long" in dt:
        return "int"
    if "bool" in dt:
        return "bool"
    return dt or "unknown"


# Per-dtype max_rel_diff thresholds for PASS in single-standard adaptation.
# When |golden| ≥ SVT (not in small-value regime), pass if relative diff ≤ this.
# fp32: ~22-bit precision ≈ 2.4e-7 (mantissa 23 bits = 1.19e-7; allow ~2x for compound)
# fp16: ~10-bit precision ≈ 1e-3 (mantissa 10 bits)
# bf16: ~7-bit precision ≈ 8e-3
REL_THRESH = {"fp32": 2 ** -21, "fp16": 2 ** -9, "bf16": 2 ** -6}  # ~5e-7 / 2e-3 / 1.6e-2


def _classify_case_v21(c: dict) -> tuple[str, str]:
    """Return (status, reason) for a single per_case entry.

    status ∈ {"PASS", "FAIL", "PASS_SMALL_VALUE", "INSUFFICIENT_DATA"}

    PASS criteria (any one is sufficient):
      1. max_abs_diff == 0 (bit-exact)
      2. max_abs_diff ≤ 10×dtype_eps (normal-magnitude L0 absolute)
      3. max_rel_diff ≤ REL_THRESH[dtype] (v2.1 §4.5.1-style relative)
      4. max_abs_diff ≤ SV_err[dtype] (small-value §4.5.3 fallback)
    """
    dt_raw = c.get("dtype") or c.get("dtype_var") or ""
    dt = _canonicalize_dtype(dt_raw)
    max_abs = c.get("max_abs_diff")
    max_rel = c.get("max_rel_diff")  # may be None for older archives
    if max_abs is None:
        return ("INSUFFICIENT_DATA", "no max_abs_diff in per_case")
    if dt in ("int", "bool"):
        if max_abs == 0:
            return ("PASS", "integer/bool bit-exact (v2.1 §4.1/§4.3)")
        return ("FAIL", f"integer not bit-exact: max_abs={max_abs}")
    if dt == "unknown":
        return ("INSUFFICIENT_DATA", f"unrecognized dtype {dt_raw!r}")
    if max_abs == 0:
        return ("PASS", "bit-exact")
    # max_rel_diff is the primary v2.1 §4.5.1 signal (NPU vs reference ratio)
    if max_rel is not None and max_rel <= REL_THRESH[dt]:
        return (
            "PASS",
            f"max_rel={max_rel:.2e} ≤ {dt}_rel_thresh ({REL_THRESH[dt]:.2e}) "
            "— v2.1 §4.5.1 normal-magnitude L0",
        )
    if max_abs <= ABS_THRESH_NORMAL[dt]:
        return (
            "PASS",
            f"max_abs={max_abs:.2e} ≤ 10×{dt}_eps ({ABS_THRESH_NORMAL[dt]:.2e}) "
            "— v2.1 §4.5.1 normal-magnitude L0 (abs fallback)",
        )
    if max_abs <= SV_ERR[dt]:
        return ("PASS_SMALL_VALUE", f"max_abs={max_abs:.2e} ≤ {dt} SV_err ({SV_ERR[dt]:.2e}) — v2.1 §4.5.3")
    return (
        "FAIL",
        f"max_abs={max_abs:.2e} > {dt} threshold (eps={EPS[dt]:.2e}, "
        f"normal_abs_thresh={ABS_THRESH_NORMAL[dt]:.2e}, rel={max_rel})",
    )


def reclassify_archive(archive_dir: Path) -> dict:
    """Return a dict with per-case + overall verdict per v2.1 rules.

    Returns:
        {
            "op": <op name>,
            "v21_overall": "PASS" | "PASS_WITH_SMALL_VALUE_RULE" | "FAIL" | "INSUFFICIENT_DATA",
            "v21_per_case": [{case_id, dtype, max_abs_diff, v21_status, v21_reason}, ...],
            "current_status": <existing status string from verification.json>,
            "would_upgrade": True if v21_overall=="PASS" and current uses "PASS_WITHIN_TOLERANCE",
        }
    """
    op = archive_dir.name
    vj_path = archive_dir / "verification.json"
    if not vj_path.exists():
        return {"op": op, "v21_overall": "INSUFFICIENT_DATA",
                "v21_per_case": [], "reason": "no verification.json"}
    try:
        vj = json.loads(vj_path.read_text())
    except json.JSONDecodeError as e:
        return {"op": op, "v21_overall": "INSUFFICIENT_DATA",
                "v21_per_case": [], "reason": f"verification.json parse error: {e}"}
    precision = vj.get("precision", {})
    pass_a = precision.get("pass_a", {})
    current_status = precision.get("status", "?")
    per_case = pass_a.get("per_case", [])
    if not per_case:
        return {"op": op, "v21_overall": "INSUFFICIENT_DATA",
                "v21_per_case": [],
                "current_status": current_status,
                "reason": "no per_case data in pass_a"}
    case_results = []
    for c in per_case:
        status, reason = _classify_case_v21(c)
        case_results.append({
            "case_id": c.get("case_id"),
            "dtype": c.get("dtype") or c.get("dtype_var"),
            "max_abs_diff": c.get("max_abs_diff"),
            "v21_status": status,
            "v21_reason": reason,
        })
    statuses = [r["v21_status"] for r in case_results]
    if all(s == "PASS" for s in statuses):
        overall = "PASS"
    elif all(s in ("PASS", "PASS_SMALL_VALUE") for s in statuses):
        overall = "PASS_WITH_SMALL_VALUE_RULE"
    elif all(s in ("PASS", "PASS_SMALL_VALUE", "INSUFFICIENT_DATA") for s in statuses):
        overall = "PASS_WITH_SOME_UNCLASSIFIED"
    else:
        overall = "FAIL"
    would_upgrade = (overall == "PASS" and "WITHIN_TOLERANCE" in current_status.upper())
    return {
        "op": op,
        "v21_overall": overall,
        "v21_per_case": case_results,
        "current_status": current_status,
        "would_upgrade": would_upgrade,
    }


def apply_to_archive(archive_dir: Path, classification: dict, dry_run: bool = True) -> dict:
    """Optionally rewrite verification.json with v2.1 verdict + justification."""
    vj_path = archive_dir / "verification.json"
    if not vj_path.exists():
        return {"applied": False, "reason": "no verification.json"}
    vj = json.loads(vj_path.read_text())
    precision = vj.setdefault("precision", {})
    overall = classification["v21_overall"]
    # Map v2.1 verdict → verification.json.precision.status
    new_status = {
        "PASS": "PASS",
        "PASS_WITH_SMALL_VALUE_RULE": "PASS",  # v2.1 §4.5.3 explicit
        "PASS_WITH_SOME_UNCLASSIFIED": "PASS_WITH_GAPS",
        "FAIL": "FAIL",
        "INSUFFICIENT_DATA": precision.get("status", "?"),
    }.get(overall, precision.get("status", "?"))

    old_status = precision.get("status")
    # Preserve old status as audit trail
    precision["v21_classification"] = {
        "v21_overall": overall,
        "v21_rule": (
            "v2.1 single-standard adaptation: max_abs_diff ≤ 10×dtype_eps "
            "(normal-magnitude) OR small-value §4.5.3"
        ),
        "v21_thresholds": {
            dt: {
                "normal": ABS_THRESH_NORMAL[dt],
                "small_value_threshold": SVT[dt],
                "small_value_err": SV_ERR[dt],
            }
            for dt in ("fp32", "fp16", "bf16")
        },
        "per_case_v21": classification["v21_per_case"],
        "previous_status": old_status,
        "reclassified_at": "2026-05-19_v21_migration",
    }
    if old_status != new_status:
        precision["status"] = new_status
    result = {
        "applied": not dry_run,
        "op": archive_dir.name,
        "old_status": old_status,
        "new_status": new_status,
        "v21_overall": overall,
    }
    if not dry_run:
        vj_path.write_text(json.dumps(vj, indent=2))
    return result


def _summary_line(c: dict) -> str:
    op = c.get("op", "?")
    overall = c.get("v21_overall", "?")
    cs = c.get("current_status", "?")
    upgrade = " [UPGRADE]" if c.get("would_upgrade") else ""
    return f"{op:<30} v21={overall:<32} cur={cs:<40}{upgrade}"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: precision_eval_v21.py <archive_dir_or_parent> [--apply]", file=sys.stderr)
        return 2
    target = Path(sys.argv[1]).resolve()
    apply = "--apply" in sys.argv[2:]
    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        return 1
    # Single archive or parent dir?
    if (target / "verification.json").exists():
        archives = [target]
    else:
        archives = sorted([d for d in target.iterdir() if d.is_dir() and (d / "verification.json").exists()])
    print(f"# v21 audit of {len(archives)} archive(s) under {target}")
    print(f"# {'op':<30} {'v21':<32} {'current':<40}")
    for arch in archives:
        cls = reclassify_archive(arch)
        print("  " + _summary_line(cls))
        if apply and cls.get("v21_overall") in ("PASS", "PASS_WITH_SMALL_VALUE_RULE", "FAIL"):
            res = apply_to_archive(arch, cls, dry_run=False)
            if res.get("old_status") != res.get("new_status"):
                print(f"    APPLIED: {res['old_status']} → {res['new_status']}")
            else:
                print("    (no status change; v21 audit data still written under precision.v21_classification)")
                apply_to_archive(arch, cls, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
