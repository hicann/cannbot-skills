# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 4 of /aog-prior-art-verify — verify staged build against edge_dataset.

Given a successful Phase 3 build (`.prior_art_build.json` verdict=SUCCESS
+ `candidate_dir/build/<op>.so`), runs the workspace's explicit
`verify_prior_art_candidate.py` adapter against that candidate `.so`, comparing
per-case outputs to `edge_dataset.pt`'s fresh source-NPU outputs.  The adapter
must accept explicit candidate/truth paths and digests and echo the exact binding
in its JSON result; an absent or legacy adapter fails closed.

Three signals captured:
- precision: per-case max_abs_err / max_rel_err against T1 (bit-exact)
  + T2 (atol=rtol=1e-3) tiers
- performance: median of N runs on A5, compared to a3_baseline_perf.json
  median per case → ratio = a3_median / a5_median
- determinism: 2 independent runs; bit-identical compare

Result: `workspace/<op>/.prior_art_candidate/verification_prior_art.json`.

The ordinary customer-verification runner is not reused because it has no
stable contract for loading an arbitrary prior-art library.  Candidate evidence
stays advisory and separate from customer-facing Phase O5 verification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from build_candidate import _load_candidate_binding, _sha256_file


# Tier thresholds — mirrors GATE_CONTRACT.md
_T1_ABS_TOL = 0.0
_T1_REL_TOL = 0.0
_T2_ABS_TOL = 1e-3
_T2_REL_TOL = 1e-3
_PERF_FLOOR = 0.6  # ratio threshold below which we record CANDIDATE_PERF_GAP


@dataclass
class PrecisionCase:
    case_id: str
    max_abs_err: float
    max_rel_err: float
    tier1_pass: bool
    tier2_pass: bool


@dataclass
class VerifyReport:
    op: str
    candidate_dir: Path
    # build precondition
    build_status: str = "UNKNOWN"
    binding: dict = field(default_factory=dict)
    # precision
    precision_status: str = "UNKNOWN"        # PASS_T1 / PASS_T2 / FAIL / SKIPPED
    n_pass_t1: int = 0
    n_pass_t2: int = 0
    n_total: int = 0
    per_case: list[PrecisionCase] = field(default_factory=list)
    # performance
    perf_ratios: list[float] = field(default_factory=list)
    perf_ratio_median: float = 0.0
    perf_status: str = "UNKNOWN"             # PASS / FAIL_FLOOR / SKIPPED
    # determinism
    determinism_n_runs: int = 0
    determinism_n_identical: int = 0
    observed_deterministic: Optional[bool] = None
    # overall
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


VerifierInvoker = Callable[[Path, Path, Path, dict], dict]


def _check_prereqs(workspace: Path, op: str) -> tuple[bool, list[str]]:
    """Returns (ready, errors). Verifies build verdict + edge_dataset presence."""
    errs = []
    build_json = workspace / ".prior_art_build.json"
    if not build_json.is_file():
        errs.append(".prior_art_build.json missing — run build_candidate first")
        return False, errs
    try:
        bd = json.loads(build_json.read_text())
    except Exception as e:
        errs.append(f"build.json parse error: {e}")
        return False, errs
    if bd.get("verdict") != "SUCCESS":
        errs.append(f"build verdict={bd.get('verdict')} — verify requires SUCCESS")
        return False, errs
    if bd.get("op") != op:
        errs.append(f"build op={bd.get('op')!r} does not match requested op={op!r}")
        return False, errs
    edge = workspace / "edge_dataset.pt"
    if not edge.is_file():
        errs.append("edge_dataset.pt missing — phase_o25_a3_ref must run first")
        return False, errs
    a3_perf = workspace / "a3_baseline_perf.json"
    if not a3_perf.is_file():
        errs.append("a3_baseline_perf.json missing — perf ratio cannot be computed")
        return False, errs
    runnable = workspace / "a3_reference_runnable.json"
    if not runnable.is_file():
        errs.append(
            "a3_reference_runnable.json missing — a fresh source-NPU capture is required"
        )
        return False, errs
    try:
        capture = json.loads(runnable.read_text())
    except Exception as exc:
        errs.append(f"a3_reference_runnable.json parse error: {exc}")
        return False, errs
    if capture.get("verdict") != "READY":
        errs.append(
            f"source-NPU capture verdict={capture.get('verdict')!r}; READY required"
        )
    if capture.get("a3_exec_attempted") is not True:
        errs.append("source-NPU capture does not record a live execution attempt")
    if not isinstance(capture.get("capture_id"), str) or not capture["capture_id"]:
        errs.append("source-NPU capture_id missing")
    if errs:
        return False, errs
    return True, errs


def _expected_binding(workspace: Path, candidate_dir: Path, build_data: dict,
                      so_path: Path) -> tuple[Optional[dict], list[str]]:
    """Bind verification to the staged source, built library, and fresh truth."""
    binding, candidate_errors = _load_candidate_binding(
        candidate_dir, str(build_data.get("op", ""))
    )
    if binding is None:
        return None, candidate_errors
    errors: list[str] = []
    if build_data.get("schema_version") != 2:
        errors.append("build report schema_version must be 2")
    if build_data.get("candidate_digest") != binding.candidate_digest:
        errors.append("build report candidate_digest does not match staged candidate")
    if build_data.get("manifest_sha256") != binding.manifest_sha256:
        errors.append("build report manifest_sha256 does not match staged manifest")
    try:
        resolved_so = so_path.resolve(strict=True)
        resolved_build = (candidate_dir / "build").resolve(strict=True)
        resolved_so.relative_to(resolved_build)
    except (OSError, ValueError):
        errors.append("candidate .so is not inside .prior_art_candidate/build")
        return None, errors
    so_sha = _sha256_file(resolved_so)
    if build_data.get("so_sha256") != so_sha:
        errors.append("candidate .so digest does not match build report")

    edge_dataset = workspace / "edge_dataset.pt"
    source_perf = workspace / "a3_baseline_perf.json"
    runnable_path = workspace / "a3_reference_runnable.json"
    try:
        capture = json.loads(runnable_path.read_text())
    except Exception as exc:
        errors.append(f"source capture metadata unreadable: {exc}")
        return None, errors
    if errors:
        return None, errors
    return {
        "candidate_path": str(resolved_so),
        "candidate_sha256": so_sha,
        "candidate_digest": binding.candidate_digest,
        "manifest_sha256": binding.manifest_sha256,
        "edge_dataset_path": str(edge_dataset.resolve()),
        "edge_dataset_sha256": _sha256_file(edge_dataset),
        "source_perf_path": str(source_perf.resolve()),
        "source_perf_sha256": _sha256_file(source_perf),
        "capture_id": capture["capture_id"],
        "capture_metadata_sha256": _sha256_file(runnable_path),
    }, []


def classify_precision_case(max_abs: float, max_rel: float) -> tuple[bool, bool]:
    """Returns (tier1_pass, tier2_pass)."""
    t1 = max_abs <= _T1_ABS_TOL and max_rel <= _T1_REL_TOL
    t2 = max_abs <= _T2_ABS_TOL and max_rel <= _T2_REL_TOL
    return t1, t2


def aggregate_precision_status(cases: list[PrecisionCase]) -> tuple[str, int, int, int]:
    """Returns (status, n_pass_t1, n_pass_t2, n_total).

    Status:
      PASS_T1 — all cases tier1_pass
      PASS_T2 — all cases tier2_pass (some not tier1)
      FAIL    — at least one case fails tier2
      SKIPPED — no cases
    """
    n = len(cases)
    if n == 0:
        return ("SKIPPED", 0, 0, 0)
    n_t1 = sum(1 for c in cases if c.tier1_pass)
    n_t2 = sum(1 for c in cases if c.tier2_pass)
    if n_t1 == n:
        return ("PASS_T1", n_t1, n_t2, n)
    if n_t2 == n:
        return ("PASS_T2", n_t1, n_t2, n)
    return ("FAIL", n_t1, n_t2, n)


def aggregate_perf_status(ratios: list[float]) -> tuple[str, float]:
    """Returns (status, median_ratio).

    PASS if median ≥ 0.6× (PB-33-aligned floor). FAIL_FLOOR otherwise.
    """
    if not ratios:
        return ("SKIPPED", 0.0)
    median = statistics.median(ratios)
    return ("PASS" if median >= _PERF_FLOOR else "FAIL_FLOOR", median)


def verify(op: str, workspace: Path,
           *,
           invoker: Optional[VerifierInvoker] = None,
           ) -> VerifyReport:
    """Phase 4 entry. Runs the candidate .so against edge_dataset.pt + a3_perf
    using `invoker` (production default invokes run_pass_b.py via subprocess).
    """
    candidate_dir = workspace / ".prior_art_candidate"
    rep = VerifyReport(op=op, candidate_dir=candidate_dir)

    ready, errs = _check_prereqs(workspace, op)
    if not ready:
        rep.errors.extend(errs)
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep

    build_data = json.loads((workspace / ".prior_art_build.json").read_text())
    rep.build_status = build_data["verdict"]
    so_path = Path(build_data["so_path"]) if build_data.get("so_path") else None
    if so_path is None or not so_path.exists():
        rep.errors.append(f"candidate .so not found: {build_data.get('so_path')}")
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep

    expected, binding_errors = _expected_binding(
        workspace, candidate_dir, build_data, so_path
    )
    if expected is None:
        rep.errors.extend(binding_errors)
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep
    rep.binding = expected

    edge_dataset = workspace / "edge_dataset.pt"
    a3_perf = workspace / "a3_baseline_perf.json"
    inv = invoker or _default_invoker
    try:
        raw = inv(so_path, edge_dataset, a3_perf, expected)
    except Exception as exc:
        rep.errors.append(f"candidate verifier raised: {exc!r}")
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep
    if not isinstance(raw, dict):
        rep.errors.append("candidate verifier returned a non-object payload")
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep
    if raw.get("error"):
        rep.errors.append(f"candidate verifier error: {raw['error']}")
    if raw.get("binding") != expected:
        rep.errors.append(
            "candidate verifier did not echo the exact candidate/truth binding"
        )
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        rep.errors.append("candidate verifier produced no cases")
    if rep.errors:
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep

    # Marshal cases
    seen_case_ids: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, dict):
            rep.errors.append("candidate verifier emitted a non-object case")
            continue
        case_id = str(case.get("case_id", "?"))
        if case_id in seen_case_ids:
            rep.errors.append(f"candidate verifier emitted duplicate case_id={case_id}")
            continue
        seen_case_ids.add(case_id)
        try:
            max_abs = float(case["max_abs_err"])
            max_rel = float(case["max_rel_err"])
        except (KeyError, TypeError, ValueError):
            rep.errors.append(f"candidate verifier case {case_id} lacks finite errors")
            continue
        if not math.isfinite(max_abs) or not math.isfinite(max_rel):
            rep.errors.append(f"candidate verifier case {case_id} has non-finite errors")
            continue
        t1, t2 = classify_precision_case(
            max_abs,
            max_rel,
        )
        rep.per_case.append(PrecisionCase(
            case_id=case_id,
            max_abs_err=max_abs,
            max_rel_err=max_rel,
            tier1_pass=t1,
            tier2_pass=t2,
        ))
        if "a5_time_s" in case and "a3_time_s" in case:
            try:
                a3 = float(case["a3_time_s"])
                a5 = float(case["a5_time_s"])
            except (TypeError, ValueError):
                rep.errors.append(f"candidate verifier case {case_id} has invalid timing")
                continue
            if math.isfinite(a3) and math.isfinite(a5) and a3 > 0 and a5 > 0:
                rep.perf_ratios.append(a3 / a5)
            else:
                rep.errors.append(f"candidate verifier case {case_id} has non-positive timing")
        else:
            rep.errors.append(f"candidate verifier case {case_id} lacks timing")

    if rep.errors or len(rep.per_case) != len(raw_cases):
        rep.precision_status = "SKIPPED"
        rep.perf_status = "SKIPPED"
        return rep

    status, n_t1, n_t2, n = aggregate_precision_status(rep.per_case)
    rep.precision_status = status
    rep.n_pass_t1 = n_t1
    rep.n_pass_t2 = n_t2
    rep.n_total = n
    rep.perf_status, rep.perf_ratio_median = aggregate_perf_status(rep.perf_ratios)

    # Read the determinism counters supplied by the verifier.
    try:
        rep.determinism_n_runs = int(raw.get("determinism_n_runs", 0))
        rep.determinism_n_identical = int(raw.get("determinism_n_identical", 0))
    except (TypeError, ValueError):
        rep.errors.append("candidate verifier emitted invalid determinism counters")
        return rep
    if (rep.determinism_n_runs < 2
            or not 0 <= rep.determinism_n_identical <= rep.determinism_n_runs):
        rep.errors.append(
            "candidate verifier must report at least two valid determinism runs"
        )
        return rep
    rep.observed_deterministic = (
        rep.determinism_n_identical == rep.determinism_n_runs
    )

    return rep


def _default_invoker(so_path: Path, edge_dataset: Path,
                      a3_perf: Path, binding: dict) -> dict:
    """Invoke an explicit candidate-aware adapter and require bound evidence.

    The ordinary migration runner is intentionally not used: it has no stable
    contract for loading an arbitrary library.  The adapter must accept the
    explicit arguments below and echo ``binding`` in its JSON result.
    """
    workspace = edge_dataset.parent
    runner = workspace / "verify_prior_art_candidate.py"
    if not runner.is_file():
        return {
            "cases": [],
            "error": (
                "candidate-aware adapter unavailable: "
                f"verify_prior_art_candidate.py not found in {workspace}"
            ),
        }
    try:
        env = os.environ.copy()
        env.update({
            "PRIOR_ART_SO": str(so_path),
            "PRIOR_ART_SO_SHA256": binding["candidate_sha256"],
            "PRIOR_ART_CANDIDATE_DIGEST": binding["candidate_digest"],
            "EDGE_DATASET": str(edge_dataset),
            "EDGE_DATASET_SHA256": binding["edge_dataset_sha256"],
            "A3_PERF": str(a3_perf),
            "A3_PERF_SHA256": binding["source_perf_sha256"],
            "PRIOR_ART_BINDING_JSON": json.dumps(binding, sort_keys=True),
        })
        command = [
            sys.executable,
            str(runner),
            "--candidate-so", str(so_path),
            "--candidate-sha256", binding["candidate_sha256"],
            "--candidate-digest", binding["candidate_digest"],
            "--edge-dataset", str(edge_dataset),
            "--edge-dataset-sha256", binding["edge_dataset_sha256"],
            "--source-perf", str(a3_perf),
            "--source-perf-sha256", binding["source_perf_sha256"],
            "--capture-id", binding["capture_id"],
        ]
        r = subprocess.run(
            command, capture_output=True, text=True, timeout=600, env=env,
            cwd=str(workspace),
        )
        if r.returncode != 0:
            return {
                "cases": [],
                "error": f"candidate adapter rc={r.returncode}: {r.stderr[:500]}",
            }
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"cases": [], "error": "no JSON found in runner stdout"}
    except subprocess.TimeoutExpired:
        return {"cases": [], "error": "verifier timeout"}
    except Exception as exc:
        return {"cases": [], "error": f"verifier invocation failed: {exc!r}"}


def write_verify_report(rep: VerifyReport, workspace: Path) -> Path:
    out = workspace / ".prior_art_candidate" / "verification_prior_art.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "op": rep.op,
        "build": {"status": rep.build_status},
        "binding": rep.binding,
        "precision": {
            "status": rep.precision_status,
            "n_pass_t1": rep.n_pass_t1,
            "n_pass_t2": rep.n_pass_t2,
            "n_total": rep.n_total,
            "per_case": [asdict(c) for c in rep.per_case],
        },
        "performance": {
            "status": rep.perf_status,
            "ratio_median": rep.perf_ratio_median,
            "per_case_ratios": rep.perf_ratios,
        },
        "determinism": {
            "n_runs": rep.determinism_n_runs,
            "n_identical": rep.determinism_n_identical,
            "observed_deterministic": rep.observed_deterministic,
        },
        "errors": rep.errors,
        "warnings": rep.warnings,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--op", required=True)
    p.add_argument("--workspace", required=True, type=Path)
    args = p.parse_args(argv)
    rep = verify(args.op, args.workspace)
    out = write_verify_report(rep, args.workspace)
    print(f"precision={rep.precision_status} ({rep.n_pass_t1}/{rep.n_total} T1, "
          f"{rep.n_pass_t2}/{rep.n_total} T2) perf={rep.perf_status} "
          f"ratio={rep.perf_ratio_median:.2f}x → {out}")
    return 0 if rep.precision_status in ("PASS_T1", "PASS_T2") else 1


if __name__ == "__main__":
    sys.exit(main())
