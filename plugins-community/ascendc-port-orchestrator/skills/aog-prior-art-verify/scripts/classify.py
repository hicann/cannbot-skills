# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 5 of /aog-prior-art-verify — classify a prior-art candidate.

Decision matrix (mirrors SKILL.md §Phase 5):

| precision         | perf ratio | determinism      | VERDICT              | Action                              |
|-------------------|------------|------------------|----------------------|-------------------------------------|
| PASS_T1 or PASS_T2| ≥ 0.6×     | observed=true    | CANDIDATE_PASS       | retain as provenance-bound seed     |
| any FAIL          | —          | —                | CANDIDATE_PRECISION_GAP | record counterexample             |
| PASS              | < 0.6×     | —                | CANDIDATE_PERF_GAP   | record optimization target          |
| —                 | —          | observed=false   | CANDIDATE_DET_GAP    | record determinism gap              |
| build FAILED      | —          | —                | CANDIDATE_BUILD_GAP  | record build-failure pattern        |

Every verdict is advisory. The caller must continue the mandatory fresh arch22
capture and the normal generation/post-verification pipeline; this module never
archives an operator or emits the customer-facing success verdict.

Special "BLOCKED" verdicts (cannot classify):
- NO_VERIFY_DATA: verification_prior_art.json missing
- VERIFY_SKIPPED: precision_status == SKIPPED (build or verifier didn't run)
- VERIFY_INVALID: unknown, malformed, unbound, or error-bearing verification
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

_PERF_FLOOR = 0.6


@dataclass
class ClassifyReport:
    op: str
    verdict: str = "UNKNOWN"
    # CANDIDATE_PASS / CANDIDATE_PRECISION_GAP / CANDIDATE_PERF_GAP
    # / CANDIDATE_DET_GAP / CANDIDATE_BUILD_GAP
    # / NO_VERIFY_DATA / VERIFY_SKIPPED / VERIFY_INVALID
    action: str = ""
    reasons: list[str] = field(default_factory=list)
    # Carries-through fields for KB / archive consumption
    precision_status: str = ""
    perf_ratio_median: float = 0.0
    perf_status: str = ""
    observed_deterministic: Optional[bool] = None
    n_pass_t1: int = 0
    n_pass_t2: int = 0
    n_total: int = 0
    # Determinism policy from .opgen_state.json (best_effort → advisory note)
    det_policy: str = "best_effort"


def _load_verify_payload(workspace: Path) -> tuple[Optional[dict], list[str]]:
    p = workspace / ".prior_art_candidate" / "verification_prior_art.json"
    if not p.is_file():
        return None, [f"verification_prior_art.json missing at {p}"]
    try:
        return json.loads(p.read_text()), []
    except Exception as e:
        return None, [f"parse error: {e}"]


def _load_det_policy(workspace: Path) -> str:
    state = workspace / ".opgen_state.json"
    if not state.is_file():
        return "best_effort"
    try:
        data = json.loads(state.read_text())
        return str(data.get("det_policy", "best_effort"))
    except Exception:
        return "best_effort"


def classify(op: str, workspace: Path) -> ClassifyReport:
    rep = ClassifyReport(op=op)
    payload, errors = _load_verify_payload(workspace)
    if payload is None:
        rep.verdict = "NO_VERIFY_DATA"
        rep.action = "rerun verify_candidate before classify"
        rep.reasons.extend(errors)
        return rep

    rep.det_policy = _load_det_policy(workspace)
    if not isinstance(payload, dict):
        rep.verdict = "VERIFY_INVALID"
        rep.action = "discard malformed verification and rerun verify_candidate"
        rep.reasons.append("verification payload is not an object")
        return rep

    build = payload.get("build")
    prec = payload.get("precision")
    perf = payload.get("performance")
    det = payload.get("determinism")
    build_status = build.get("status") if isinstance(build, dict) else None

    # A known build failure is an advisory counterexample.  It does not enter
    # the stricter measurement schema because no candidate was executed.
    if build_status in ("BUILD_FAILED", "TIMEOUT", "SCP_PUSH_FAILED",
                         "SCP_PULL_FAILED", "NO_CANDIDATE"):
        rep.verdict = "CANDIDATE_BUILD_GAP"
        rep.action = (
            "record prior-art build failure as a counterexample; continue mandatory "
            "fresh arch22 capture and normal generation"
        )
        rep.reasons.append(f"build status = {build_status}")
        return rep

    invalid_reasons: list[str] = []
    if payload.get("schema_version") != 2:
        invalid_reasons.append("schema_version must be 2")
    if payload.get("op") != op:
        invalid_reasons.append(
            f"verification op mismatch: {payload.get('op')!r} != {op!r}"
        )
    if build_status != "SUCCESS":
        invalid_reasons.append(f"unknown/non-success build status: {build_status!r}")
    if payload.get("errors"):
        invalid_reasons.append("verification contains errors")
    binding = payload.get("binding")
    binding_keys = {
        "candidate_path", "candidate_sha256", "candidate_digest",
        "manifest_sha256", "edge_dataset_path", "edge_dataset_sha256",
        "source_perf_path", "source_perf_sha256", "capture_id",
        "capture_metadata_sha256",
    }
    if not isinstance(binding, dict) or not binding_keys.issubset(binding):
        invalid_reasons.append("verification binding is missing required object digests")
    else:
        digest_keys = {
            "candidate_sha256", "candidate_digest", "manifest_sha256",
            "edge_dataset_sha256", "source_perf_sha256",
            "capture_metadata_sha256",
        }
        if any(not isinstance(binding.get(key), str) or not binding[key]
               for key in binding_keys):
            invalid_reasons.append("verification binding contains empty/non-string values")
        elif any(len(binding[key]) != 64
                 or any(char not in "0123456789abcdef" for char in binding[key])
                 for key in digest_keys):
            invalid_reasons.append("verification binding contains an invalid SHA-256")
    if not isinstance(prec, dict) or not isinstance(perf, dict) or not isinstance(det, dict):
        invalid_reasons.append("precision/performance/determinism sections must be objects")
        prec = prec if isinstance(prec, dict) else {}
        perf = perf if isinstance(perf, dict) else {}
        det = det if isinstance(det, dict) else {}

    rep.precision_status = str(prec.get("status", "UNKNOWN"))
    rep.perf_status = str(perf.get("status", "UNKNOWN"))
    rep.observed_deterministic = det.get("observed_deterministic")
    try:
        rep.perf_ratio_median = float(perf.get("ratio_median"))
        rep.n_pass_t1 = int(prec.get("n_pass_t1"))
        rep.n_pass_t2 = int(prec.get("n_pass_t2"))
        rep.n_total = int(prec.get("n_total"))
        det_runs = int(det.get("n_runs"))
        det_identical = int(det.get("n_identical"))
    except (TypeError, ValueError):
        invalid_reasons.append("verification counters/ratio are missing or non-numeric")
        det_runs = 0
        det_identical = 0

    if rep.precision_status == "SKIPPED" and not invalid_reasons:
        rep.verdict = "VERIFY_SKIPPED"
        rep.action = "verify phase did not produce data; check prereqs"
        rep.reasons.append("precision.status == SKIPPED")
        return rep

    if rep.precision_status not in {"PASS_T1", "PASS_T2", "FAIL"}:
        invalid_reasons.append(
            f"unknown precision status: {rep.precision_status!r}"
        )
    if not (rep.n_total > 0
            and 0 <= rep.n_pass_t1 <= rep.n_pass_t2 <= rep.n_total):
        invalid_reasons.append("precision counts are empty or inconsistent")
    per_case = prec.get("per_case")
    if not isinstance(per_case, list) or len(per_case) != rep.n_total:
        invalid_reasons.append("precision per_case length does not match n_total")
    if rep.precision_status == "PASS_T1" and not (
            rep.n_pass_t1 == rep.n_pass_t2 == rep.n_total):
        invalid_reasons.append("PASS_T1 counts are inconsistent")
    if rep.precision_status == "PASS_T2" and not (
            rep.n_pass_t1 < rep.n_total and rep.n_pass_t2 == rep.n_total):
        invalid_reasons.append("PASS_T2 counts are inconsistent")
    if rep.precision_status == "FAIL" and rep.n_pass_t2 >= rep.n_total:
        invalid_reasons.append("FAIL status has no failing T2 case")
    if rep.det_policy not in {"required", "best_effort"}:
        invalid_reasons.append(f"unknown determinism policy: {rep.det_policy!r}")
    if (det_runs < 2 or not 0 <= det_identical <= det_runs
            or not isinstance(rep.observed_deterministic, bool)):
        invalid_reasons.append("determinism measurement is missing or inconsistent")
    elif rep.observed_deterministic != (det_identical == det_runs):
        invalid_reasons.append("determinism verdict contradicts its counters")
    if (rep.perf_status not in {"PASS", "FAIL_FLOOR"}
            or not math.isfinite(rep.perf_ratio_median)
            or rep.perf_ratio_median <= 0):
        invalid_reasons.append(
            f"invalid performance evidence: status={rep.perf_status!r}, "
            f"ratio={rep.perf_ratio_median!r}"
        )
    elif ((rep.perf_status == "PASS" and rep.perf_ratio_median < _PERF_FLOOR)
          or (rep.perf_status == "FAIL_FLOOR"
              and rep.perf_ratio_median >= _PERF_FLOOR)):
        invalid_reasons.append("performance status contradicts ratio_median")

    if invalid_reasons:
        rep.verdict = "VERIFY_INVALID"
        rep.action = "discard unbound/malformed verification and rerun verify_candidate"
        rep.reasons.extend(invalid_reasons)
        return rep

    # Precision check (highest priority among PASS-eligibles)
    if rep.precision_status == "FAIL":
        rep.verdict = "CANDIDATE_PRECISION_GAP"
        rep.action = (
            f"prior art FAIL precision ({rep.n_pass_t2}/{rep.n_total} T2); "
            "record the gap in the worker brief and continue normal generation"
        )
        rep.reasons.append(f"precision.status=FAIL ({rep.n_pass_t2}/{rep.n_total} T2)")
        return rep

    # Determinism check — only matters if DET_POLICY is required, OR
    # the deterministic flag is explicitly false AND policy says we care.
    if rep.observed_deterministic is False and rep.det_policy == "required":
        rep.verdict = "CANDIDATE_DET_GAP"
        rep.action = (
            "DET_POLICY=required but observed_deterministic=false; "
            "record the gap and continue normal generation"
        )
        rep.reasons.append("observed_deterministic=false under required policy")
        return rep

    # Performance check
    if rep.perf_status == "FAIL_FLOOR":
        rep.verdict = "CANDIDATE_PERF_GAP"
        rep.action = (
            f"prior-art perf {rep.perf_ratio_median:.2f}x < {_PERF_FLOOR:.2f}x floor; "
            "record an optimization target and continue normal generation"
        )
        rep.reasons.append(
            f"perf median = {rep.perf_ratio_median:.2f}x (floor {_PERF_FLOOR:.2f}x)"
        )
        return rep

    # All checks passed.  This line is deliberately unreachable for UNKNOWN,
    # empty, error-bearing, or unbound measurements.
    rep.verdict = "CANDIDATE_PASS"
    rep.action = (
        "retain as a provenance-bound implementation seed; continue mandatory fresh "
        "arch22 capture, worker generation, and independent O4/O5 verification; "
        "do not copy this candidate verdict into customer verification.json"
    )
    rep.reasons.append(
        f"precision={rep.precision_status} ({rep.n_pass_t1}/{rep.n_total} T1, "
        f"{rep.n_pass_t2}/{rep.n_total} T2), perf={rep.perf_ratio_median:.2f}x ≥ "
        f"{_PERF_FLOOR:.2f}x"
    )
    if rep.observed_deterministic is False:
        rep.reasons.append(
            "soft-OK: observed_deterministic=false but DET_POLICY=best_effort"
        )
    return rep


def write_verdict(rep: ClassifyReport, workspace: Path) -> Path:
    out = workspace / "prior_art_verdict.json"
    out.write_text(json.dumps(asdict(rep), indent=2, sort_keys=True) + "\n")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--op", required=True)
    p.add_argument("--workspace", required=True, type=Path)
    args = p.parse_args(argv)
    rep = classify(args.op, args.workspace)
    out = write_verdict(rep, args.workspace)
    print(f"VERDICT={rep.verdict}  → {out}")
    for r in rep.reasons:
        print(f"  reason: {r}")
    print(f"action: {rep.action}")
    return 0 if rep.verdict == "CANDIDATE_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
