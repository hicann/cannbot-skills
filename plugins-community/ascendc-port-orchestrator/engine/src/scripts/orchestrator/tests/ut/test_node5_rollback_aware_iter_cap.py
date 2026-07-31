# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""NODE-5 (2026-05-28): rollback-aware iter-cap exemption tests.

When O5 RUNNER_FAILED triggers a state-machine rollback whose root cause is
infrastructural (SCP timeout, oversized .pt, JSON parse fail on verifier
stdout, env issue), the re-entry SHOULD NOT consume the algorithm
iter_below_cap budget — the kernel didn't change.

Empirical anchor: independent review FA-A3 bg `bjfp4fi3e` 12:18-16:03Z consumed all 4
probe-iters across 3 infra-rollback cycles even though the kernel was
correct on probe-1 (10/10 PASS). NODE-4 raised probe iter_cap 4→8 as a
stopgap. NODE-5 is the cleaner fix: distinguish infra-rollback from
algorithm rollback via a `rollback_kind` field that flows through the
audit trail.

Test scope:
- `classify_runner_error` pattern coverage (infra strings → "infra";
  algorithm strings → "algorithm"; empty → "algorithm" conservative).
- `MeasuredResult.__post_init__` auto-classifies when runner_error is set.
- `O5Report.rollback_kind` field exists and propagates from MeasuredResult
  via `phase_o5.post_verify_for_finalize`.
- `TransitionDecision.rollback_kind` field + `record_transition` writes
  the tag into state_transitions.jsonl (only when non-None — back-compat).
- `iter_counts_from_log` SKIPS entries tagged rollback_kind="infra" from
  the iter counter; counts normal + algorithm-rollback entries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent.parent.parent  # repo root (a5_ops)
sys.path.insert(0, str(_HERE.parent.parent))                    # orchestrator/
sys.path.insert(0, str(_ROOT / "src" / "scripts" / "workflow"))  # workflow/


# ────────────────────── classify_runner_error ──────────────────────


from phase_o5 import classify_runner_error, MeasuredResult, O5Report  # noqa: E402


@pytest.mark.parametrize("msg", [
    "scp aborted: oversized payload .pt files exceed 100 MiB threshold",
    "ssh: connect to host 198.51.100.70 port 22: Connection refused",
    "ssh timeout after 60s",
    "pass_b: verifier stdout had no parseable JSON; tail='...'",
    "pre-O5 workspace sync failed: scp aborted",
    "tool missing (sshpass)",
    "verifier exit 9; stderr='missing CANN env'",
    "SSH+verifier timeout after 600s",
    "ConnectTimeout=60",
])
def test_classify_infra_patterns_return_infra(msg):
    """Infra-class fragments → 'infra'."""
    assert classify_runner_error(msg) == "infra"


@pytest.mark.parametrize("msg", [
    "verification.json missing — nothing to verify against",
    "verification.json malformed: Expecting value: line 1 column 1 (char 0)",
    "runner raised: AttributeError: Model has no attribute forward",
    "pass_a tier1_pass=12 differs from claim 50",
    "precision FAIL: max_abs_diff 8.2e-1 exceeds tolerance",
    "kernel emitted invalid LoadData params",
])
def test_classify_algorithm_messages_return_algorithm(msg):
    """Algorithm-class / unrecognized fragments → 'algorithm' (conservative)."""
    assert classify_runner_error(msg) == "algorithm"


def test_classify_empty_returns_algorithm():
    """Empty / None handled conservatively (no auto-free-iter)."""
    assert classify_runner_error("") == "algorithm"
    assert classify_runner_error(None) == "algorithm"  # type: ignore[arg-type]


# ────────────────────── MeasuredResult __post_init__ ──────────────────────


def test_measured_result_auto_classifies_infra_error():
    """MeasuredResult(runner_error=<infra>) → rollback_kind auto-set to 'infra'."""
    m = MeasuredResult(runner_error="scp aborted: oversized payload")
    assert m.rollback_kind == "infra"


def test_measured_result_auto_classifies_algorithm_error():
    """MeasuredResult(runner_error=<other>) → rollback_kind defaults to 'algorithm'."""
    m = MeasuredResult(runner_error="precision FAIL: max_abs_diff exceeds tol")
    assert m.rollback_kind == "algorithm"


def test_measured_result_no_error_no_rollback_kind():
    """Success path: no runner_error → rollback_kind stays None."""
    m = MeasuredResult(pass_a={"tier1_pass": 50, "total": 50})
    assert m.rollback_kind is None


def test_measured_result_explicit_kind_respected():
    """Explicit kind argument overrides auto-classification (escape hatch)."""
    m = MeasuredResult(
        runner_error="scp aborted",  # would be classified as infra
        rollback_kind="algorithm",   # explicit override
    )
    assert m.rollback_kind == "algorithm"


# ────────────────────── O5Report.rollback_kind propagation ──────────────────────


def test_o5report_has_rollback_kind_field():
    """O5Report has the new field (back-compat: default None)."""
    rep = O5Report(verdict="VERIFIED")
    assert hasattr(rep, "rollback_kind")
    assert rep.rollback_kind is None


def test_post_verify_for_finalize_propagates_infra_kind(tmp_path):
    """post_verify_for_finalize: when runner returns infra-tagged MeasuredResult,
    the resulting O5Report carries rollback_kind='infra'.
    """
    from phase_o5 import post_verify_for_finalize

    # Seed minimal verification.json so we get past the existence check
    workspace = tmp_path / "myop"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "op": "myop",
        "opgen_mode": "backward",
    }))
    (workspace / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"tier1_pass": 50, "total": 50, "status": "PASS"}},
    }))

    def fake_runner(ws, op, lane):
        # Infra error → MeasuredResult auto-classifies as infra
        return MeasuredResult(runner_error="scp aborted: oversized payload")

    rep = post_verify_for_finalize(workspace, "myop", lane=0, runner=fake_runner)
    assert rep.verdict == "RUNNER_FAILED"
    assert rep.rollback_kind == "infra"


def test_post_verify_for_finalize_propagates_algorithm_kind(tmp_path):
    """Algorithm-class runner error → rollback_kind='algorithm'."""
    from phase_o5 import post_verify_for_finalize

    workspace = tmp_path / "myop"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "op": "myop",
        "opgen_mode": "backward",
    }))
    (workspace / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"tier1_pass": 50, "total": 50, "status": "PASS"}},
    }))

    def fake_runner(ws, op, lane):
        return MeasuredResult(runner_error="some other failure")

    rep = post_verify_for_finalize(workspace, "myop", lane=0, runner=fake_runner)
    assert rep.verdict == "RUNNER_FAILED"
    assert rep.rollback_kind == "algorithm"


# ────────────────────── TransitionDecision + record_transition ──────────────────────


def test_transition_decision_has_rollback_kind_field():
    """TransitionDecision has the new field (back-compat: default None)."""
    import state_executor
    d = state_executor.TransitionDecision(
        next_state="await_worker",
        matched_transition_index=-1,
        rationale="x",
        from_state="finalize",
        handoff="",
    )
    assert hasattr(d, "rollback_kind")
    assert d.rollback_kind is None


def test_record_transition_writes_rollback_kind_when_present(tmp_path):
    """record_transition writes rollback_kind to jsonl entry ONLY when non-None
    (back-compat: legacy entries don't have the field).
    """
    import state_executor
    workspace = tmp_path / "myop"
    workspace.mkdir()

    # Case 1: rollback_kind=None → field NOT in entry
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state="await_worker",
            matched_transition_index=-1,
            rationale="normal transition",
            from_state="finalize",
            handoff="",
        ),
    )
    # Case 2: rollback_kind="infra" → field IS in entry
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state="await_worker",
            matched_transition_index=-1,
            rationale="infra rollback",
            from_state="finalize",
            handoff="",
            rollback_kind="infra",
        ),
    )

    lines = (workspace / "state_transitions.jsonl").read_text().splitlines()
    assert len(lines) == 2
    entry1 = json.loads(lines[0])
    entry2 = json.loads(lines[1])
    assert "rollback_kind" not in entry1  # legacy entry shape
    assert entry2.get("rollback_kind") == "infra"


# ────────────────────── iter_counts_from_log skip ──────────────────────


def test_iter_counts_skips_infra_rollback_entries(tmp_path):
    """iter_counts_from_log SKIPS entries tagged rollback_kind="infra"."""
    import state_machine

    workspace = tmp_path / "myop"
    workspace.mkdir()

    # Write 4 transitions: 1 normal, 2 infra-rollback (free), 1 algorithm
    # — counter "probe" should count to 2 (normal + algorithm), NOT 4.
    entries = [
        {"from_state": "await_worker", "to_state": "await_probe",
         "handoff": "→ orchestrator: done", "matched_transition_index": 0,
         "rationale": "first probe spawn", "iter_counts_snapshot": {}},
        {"from_state": "finalize", "to_state": "await_probe",
         "handoff": "", "matched_transition_index": -1,
         "rationale": "O5 infra rollback (SCP timeout)",
         "iter_counts_snapshot": {"probe": 1},
         "rollback_kind": "infra"},  # FREE — should be skipped
        {"from_state": "finalize", "to_state": "await_probe",
         "handoff": "", "matched_transition_index": -1,
         "rationale": "O5 infra rollback (JSON parse fail)",
         "iter_counts_snapshot": {"probe": 1},
         "rollback_kind": "infra"},  # FREE — should be skipped
        {"from_state": "finalize", "to_state": "await_probe",
         "handoff": "", "matched_transition_index": -1,
         "rationale": "MISMATCH algorithm rollback (real count mismatch)",
         "iter_counts_snapshot": {"probe": 1},
         "rollback_kind": "algorithm"},  # CONSUMES
    ]
    log = workspace / "state_transitions.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    # Load state machine + count
    sm = state_machine.load_state_machine()
    counts = state_machine.iter_counts_from_log(workspace, sm)
    # probe counter: 2 (entry 1 normal + entry 4 algorithm); infra ones skipped
    assert counts.get("probe") == 2, (
        f"iter_counts should skip infra-rollback entries; expected probe=2 "
        f"(1 normal + 1 algorithm; 2 infra skipped), got {counts.get('probe')}"
    )


def test_iter_counts_counts_legacy_entries_without_field(tmp_path):
    """Back-compat: entries WITHOUT rollback_kind field count normally (legacy
    pre-NODE-5 state_transitions.jsonl).
    """
    import state_machine

    workspace = tmp_path / "legacyop"
    workspace.mkdir()
    entries = [
        {"from_state": "await_worker", "to_state": "await_probe",
         "handoff": "→ orchestrator: done", "matched_transition_index": 0,
         "rationale": "x", "iter_counts_snapshot": {}},
        # NO rollback_kind field — legacy shape
        {"from_state": "finalize", "to_state": "await_probe",
         "handoff": "", "matched_transition_index": -1,
         "rationale": "y", "iter_counts_snapshot": {"probe": 1}},
    ]
    log = workspace / "state_transitions.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    sm = state_machine.load_state_machine()
    counts = state_machine.iter_counts_from_log(workspace, sm)
    assert counts.get("probe") == 2  # both entries count (no infra tag)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
