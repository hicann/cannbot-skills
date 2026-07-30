# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for classify — Phase 5 of aog-prior-art-verify (decision matrix)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from classify import classify, write_verdict  # noqa: E402


def _setup(tmp_path: Path, op: str = "op_a", payload: dict = None,
           det_policy: str = "best_effort") -> Path:
    ws = tmp_path / "ws" / op
    candidate = ws / ".prior_art_candidate"
    candidate.mkdir(parents=True)
    (candidate / "verification_prior_art.json").write_text(json.dumps(payload or {}))
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": op, "det_policy": det_policy,
    }))
    return ws


def _payload(build="SUCCESS", precision="PASS_T1", perf="PASS",
              ratio=1.5, det=True, n_t1=8, n_t2=8, n=8) -> dict:
    if precision == "FAIL" and n_t2 == n:
        n_t2 = n - 1
        n_t1 = min(n_t1, n_t2)
    binding = {
        "candidate_path": "/candidate.so",
        "candidate_sha256": "1" * 64,
        "candidate_digest": "2" * 64,
        "manifest_sha256": "3" * 64,
        "edge_dataset_path": "/edge_dataset.pt",
        "edge_dataset_sha256": "4" * 64,
        "source_perf_path": "/a3_baseline_perf.json",
        "source_perf_sha256": "5" * 64,
        "capture_id": "capture-current-run",
        "capture_metadata_sha256": "6" * 64,
    }
    return {
        "schema_version": 2,
        "op": "op_a",
        "build": {"status": build},
        "precision": {"status": precision, "n_pass_t1": n_t1,
                       "n_pass_t2": n_t2, "n_total": n,
                       "per_case": [{"case_id": index} for index in range(n)]},
        "performance": {"status": perf, "ratio_median": ratio},
        "determinism": {
            "observed_deterministic": det,
            "n_runs": 2,
            "n_identical": 2 if det else 1,
        },
        "binding": binding,
        "errors": [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Happy path: candidate passes its advisory quality gate
# ---------------------------------------------------------------------------
def test_pass_t1_with_good_perf_and_det(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload())
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_PASS"
    assert "provenance-bound implementation seed" in rep.action
    assert "do not copy" in rep.action


def test_pass_t2_with_good_perf(tmp_path: Path) -> None:
    """PASS_T2 also yields the advisory candidate-pass verdict."""
    ws = _setup(tmp_path, payload=_payload(precision="PASS_T2",
                                            n_t1=0, n_t2=8))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_PASS"


def test_pass_with_perf_exactly_at_floor(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(ratio=0.6))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_PASS"


# ---------------------------------------------------------------------------
# Candidate-gap paths
# ---------------------------------------------------------------------------
def test_candidate_precision_gap_overrides_perf(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(precision="FAIL", ratio=2.0))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_PRECISION_GAP"
    assert "prior art FAIL precision" in rep.action


def test_candidate_perf_gap_when_below_floor(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(ratio=0.4, perf="FAIL_FLOOR"))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_PERF_GAP"
    assert "0.40x" in rep.action


def test_candidate_build_gap_when_build_failed(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(build="BUILD_FAILED"))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_BUILD_GAP"


def test_candidate_build_gap_when_scp_push_failed(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(build="SCP_PUSH_FAILED"))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_BUILD_GAP"


def test_candidate_build_gap_when_timeout(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(build="TIMEOUT"))
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_BUILD_GAP"


# ---------------------------------------------------------------------------
# Determinism interaction with DET_POLICY
# ---------------------------------------------------------------------------
def test_candidate_det_gap_only_when_policy_required(tmp_path: Path) -> None:
    """observed_deterministic=false + policy=required → candidate gap."""
    ws = _setup(tmp_path, payload=_payload(det=False), det_policy="required")
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_DET_GAP"


def test_pass_when_det_false_but_policy_best_effort(tmp_path: Path) -> None:
    """best-effort determinism still permits an advisory candidate pass."""
    ws = _setup(tmp_path, payload=_payload(det=False), det_policy="best_effort")
    rep = classify("op_a", ws)
    assert rep.verdict == "CANDIDATE_PASS"
    assert any("soft-OK" in r for r in rep.reasons)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_no_verify_data(tmp_path: Path) -> None:
    ws = tmp_path / "ws" / "op_a"
    ws.mkdir(parents=True)
    rep = classify("op_a", ws)
    assert rep.verdict == "NO_VERIFY_DATA"


def test_verify_skipped(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(precision="SKIPPED"))
    rep = classify("op_a", ws)
    assert rep.verdict == "VERIFY_SKIPPED"


def test_classify_writes_verdict_file(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload())
    rep = classify("op_a", ws)
    out = write_verdict(rep, ws)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["verdict"] == "CANDIDATE_PASS"
    assert "action" in data
    assert "reasons" in data


def test_det_policy_defaults_when_state_missing(tmp_path: Path) -> None:
    """If .opgen_state.json missing, det_policy defaults to best_effort."""
    ws = tmp_path / "ws" / "op_a"
    candidate = ws / ".prior_art_candidate"
    candidate.mkdir(parents=True)
    (candidate / "verification_prior_art.json").write_text(
        json.dumps(_payload(det=False))
    )
    # NO .opgen_state.json
    rep = classify("op_a", ws)
    assert rep.det_policy == "best_effort"
    # det=False + best_effort = soft-OK candidate pass
    assert rep.verdict == "CANDIDATE_PASS"


def test_unknown_empty_measurement_cannot_be_candidate_pass(tmp_path: Path) -> None:
    """Regression: UNKNOWN + zero cases + perf PASS used to fall through."""
    payload = _payload()
    payload["precision"] = {
        "status": "UNKNOWN", "n_pass_t1": 0, "n_pass_t2": 0,
        "n_total": 0, "per_case": [],
    }
    ws = _setup(tmp_path, payload=payload)

    rep = classify("op_a", ws)

    assert rep.verdict == "VERIFY_INVALID"
    assert any("unknown precision" in reason or "empty" in reason
               for reason in rep.reasons)


def test_error_bearing_measurement_is_invalid(tmp_path: Path) -> None:
    payload = _payload()
    payload["errors"] = ["adapter did not load candidate"]
    ws = _setup(tmp_path, payload=payload)

    rep = classify("op_a", ws)

    assert rep.verdict == "VERIFY_INVALID"


def test_missing_binding_is_invalid(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("binding")
    ws = _setup(tmp_path, payload=payload)

    rep = classify("op_a", ws)

    assert rep.verdict == "VERIFY_INVALID"


def test_contradictory_pass_counts_are_invalid(tmp_path: Path) -> None:
    payload = _payload()
    payload["precision"]["n_pass_t1"] = 7
    ws = _setup(tmp_path, payload=payload)

    rep = classify("op_a", ws)

    assert rep.verdict == "VERIFY_INVALID"


def test_candidate_invalid_build_is_not_a_build_gap(tmp_path: Path) -> None:
    ws = _setup(tmp_path, payload=_payload(build="CANDIDATE_INVALID"))

    rep = classify("op_a", ws)

    assert rep.verdict == "VERIFY_INVALID"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
