# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for state_executor.py.

Run: cd src/scripts/orchestrator && python3 -m pytest tests/test_state_executor.py -v
Or:  python3 -m pytest src/scripts/orchestrator/tests/test_state_executor.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add orchestrator dir to path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import state_executor as se  # noqa: E402


@pytest.fixture
def fresh_ws(tmp_path):
    """Empty workspace + minimal PROGRESS.md."""
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


@pytest.fixture
def ws_with_log(tmp_path):
    """Workspace with one canonical state_transitions.jsonl entry."""
    (tmp_path / "PROGRESS.md").write_text("# test\n")
    log = [
        {"ts": "2026-05-04T01:00:00Z", "from_state": "init", "to_state": "await_worker",
         "handoff": "init", "matched_transition_index": 0, "rationale": "cold-start"},
    ]
    (tmp_path / "state_transitions.jsonl").write_text(
        "\n".join(json.dumps(e) for e in log) + "\n"
    )
    return tmp_path


@pytest.fixture
def ws_finalized(tmp_path):
    """Workspace at finalize state with PASS verification."""
    (tmp_path / "PROGRESS.md").write_text("# done\n")
    log = [
        {"ts": "2026-05-04T01:00:00Z", "from_state": "init", "to_state": "await_worker",
         "handoff": "init", "matched_transition_index": 0, "rationale": "cold-start"},
        {"ts": "2026-05-04T01:30:00Z", "from_state": "await_worker", "to_state": "finalize",
         "handoff": "→ orchestrator: done — Pass A 50/50, perf 0.7×",
         "matched_transition_index": 2, "rationale": "PASS + perf above threshold"},
    ]
    (tmp_path / "state_transitions.jsonl").write_text(
        "\n".join(json.dumps(e) for e in log) + "\n"
    )
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_a": {"status": "PASS", "n_pass": 50, "n_total": 50}},
        "performance": {"status": "PASS", "ratio": 0.7},
    }))
    return tmp_path


# ---------------------------------------------------------------------------
# next_agent + is_terminal
# ---------------------------------------------------------------------------
def test_next_agent_for_each_state():
    assert se.next_agent("await_worker") == "aog-kernel-worker"
    assert se.next_agent("await_optimizer") == "aog-kernel-optimizer"
    assert se.next_agent("await_probe") == "aog-precision-probe"
    assert se.next_agent("await_fused_optimizer") == "aog-fused-optimizer"
    assert se.next_agent("await_researcher") == "aog-researcher"
    assert se.next_agent("await_det_analyzer") == "aog-determinism-analyzer"


def test_next_agent_terminal_states():
    # P0dd (2026-05-05): done is the new truly-terminal state. abort is also
    # terminal. finalize is now a non-terminal state with finalize_pipeline as
    # its in-process agent (orchestrator dispatches Python instead of CC).
    assert se.next_agent("done") is None
    assert se.next_agent("abort") is None
    assert se.next_agent("finalize") == "aog-finalize-pipeline"


def test_next_agent_unknown_state():
    # Unknown state returns None (not crash)
    assert se.next_agent("await_orchestrator_decision") is None
    assert se.next_agent("partial_persist") is None


def test_is_terminal():
    # P0dd (2026-05-05): finalize is no longer terminal; done replaces it.
    assert se.is_terminal("done")
    assert se.is_terminal("abort")
    assert not se.is_terminal("finalize")
    assert not se.is_terminal("await_worker")
    assert not se.is_terminal("await_probe")


# ---------------------------------------------------------------------------
# current_state
# ---------------------------------------------------------------------------
def test_fresh_workspace_starts_at_initial_state(fresh_ws):
    """Empty log → defaults to YAML's phase_o4_initial_state."""
    state = se.current_state(fresh_ws)
    # Per opgen_state_machine.yaml, this should be `await_worker`
    assert state == "await_worker"


def test_logged_workspace_returns_last_state(ws_with_log):
    state = se.current_state(ws_with_log)
    assert state == "await_worker"


def test_finalized_workspace_state(ws_finalized):
    assert se.current_state(ws_finalized) == "finalize"


# ---------------------------------------------------------------------------
# iter_count + at_iter_cap
# ---------------------------------------------------------------------------
def test_iter_count_fresh_zero(fresh_ws):
    assert se.iter_count(fresh_ws, "worker") == 0
    assert se.iter_count(fresh_ws, "probe") == 0


def test_iter_count_increments(ws_with_log):
    # ws_with_log has 1 transition into await_worker (counter "worker")
    assert se.iter_count(ws_with_log, "worker") == 1


def test_iter_cap_reads_yaml():
    # YAML has worker iter_cap=9 (V3.7.0 bump)
    assert se.iter_cap("await_worker") == 9
    # Optimizer cap=5
    assert se.iter_cap("await_optimizer") == 5
    # Researcher cap=2
    assert se.iter_cap("await_researcher") == 2


def test_at_iter_cap_false_when_under(ws_with_log):
    assert not se.at_iter_cap(ws_with_log, "await_worker")


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------
def test_snapshot_returns_correct_fields(ws_finalized):
    # P0dd: ws_finalized's log ends at finalize state. With finalize now
    # NON-terminal (it has the in-process pipeline as its agent), is_terminal
    # is False and the orchestrator will dispatch the pipeline next.
    snap = se.snapshot(ws_finalized)
    assert snap.op == ws_finalized.name
    assert snap.workspace == ws_finalized
    assert snap.current_state == "finalize"
    assert snap.is_terminal is False  # P0dd: finalize is non-terminal now
    assert "done" in snap.last_handoff
    assert "worker" in snap.iter_counts
    assert snap.iter_counts["worker"] == 1


def test_snapshot_empty_handoff_on_fresh(fresh_ws):
    snap = se.snapshot(fresh_ws)
    assert snap.last_handoff == ""


# ---------------------------------------------------------------------------
# next_state + record_transition
# ---------------------------------------------------------------------------
def test_next_state_dry_run_does_not_modify_log(ws_with_log):
    log_before = (ws_with_log / "state_transitions.jsonl").read_text()
    decision = se.next_state(ws_with_log, "→ orchestrator: done", dry_run=True)
    log_after = (ws_with_log / "state_transitions.jsonl").read_text()
    assert log_before == log_after, "dry_run must not modify state log"
    # Decision routing depends on YAML; for fresh kw-1 with `done` handoff +
    # no perf/det info, may go to await_probe (V3.3 safety) or finalize
    assert decision.next_state in ("finalize", "await_probe", "await_optimizer", "abort")


def test_next_state_records_transition(ws_with_log):
    log_path = ws_with_log / "state_transitions.jsonl"
    n_lines_before = len(log_path.read_text().splitlines())
    se.next_state(ws_with_log, "→ orchestrator: done", dry_run=False)
    n_lines_after = len(log_path.read_text().splitlines())
    assert n_lines_after == n_lines_before + 1


def test_record_transition_uses_canonical_keys(fresh_ws):
    decision = se.TransitionDecision(
        next_state="finalize",
        matched_transition_index=0,
        rationale="test",
        from_state="await_worker",
        handoff="→ orchestrator: done",
    )
    se.record_transition(fresh_ws, decision)
    log = (fresh_ws / "state_transitions.jsonl").read_text()
    entry = json.loads(log.strip().splitlines()[0])
    # Must have canonical keys
    assert "from_state" in entry
    assert "to_state" in entry
    assert "handoff" in entry
    assert "matched_transition_index" in entry
    assert "rationale" in entry
    assert "iter_counts_snapshot" in entry
    assert "ts" in entry
    # Must NOT have alias keys
    assert "from" not in entry
    assert "to" not in entry
    # ts must be ISO + Z
    assert entry["ts"].endswith("Z")


# ---------------------------------------------------------------------------
# validate_log
# ---------------------------------------------------------------------------
def test_validate_log_empty_workspace(fresh_ws):
    ok, errors = se.validate_log(fresh_ws)
    assert ok
    assert errors == []


def test_validate_log_canonical_passes(ws_with_log):
    ok, errors = se.validate_log(ws_with_log)
    assert ok, f"errors: {errors}"


def test_validate_log_detects_invalid_state(tmp_path):
    """Worker wrote a bogus to_state — validate_log catches it.
    P0dd (2026-05-05): `done` is now a valid YAML state (terminal). Use a
    truly bogus state name for this test.
    """
    (tmp_path / "PROGRESS.md").write_text("# test\n")
    log = [
        {"ts": "2026-05-04T01:00:00Z", "from_state": "await_worker", "to_state": "bogus_state",
         "handoff": "→ orchestrator: done", "matched_transition_index": 0, "rationale": ""},
    ]
    (tmp_path / "state_transitions.jsonl").write_text(
        "\n".join(json.dumps(e) for e in log) + "\n"
    )
    ok, errors = se.validate_log(tmp_path)
    assert not ok
    assert any("bogus_state" in e for e in errors), f"errors: {errors}"


def test_validate_log_detects_alias_keys(tmp_path):
    """Worker wrote `from`/`to` instead of `from_state`/`to_state` (DEBT-074)."""
    (tmp_path / "PROGRESS.md").write_text("# test\n")
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from": "await_worker",  # alias key — should be from_state
        "to": "finalize",        # alias key
        "verdict": "PASS",
    }) + "\n"
    (tmp_path / "state_transitions.jsonl").write_text(log_text)
    ok, errors = se.validate_log(tmp_path)
    assert not ok
    assert any("missing from_state/to_state" in e for e in errors), f"errors: {errors}"


def test_validate_log_detects_partial_persist(tmp_path):
    """worker wrote to_state=partial_persist → not a YAML state."""
    (tmp_path / "PROGRESS.md").write_text("# test\n")
    log = [
        {"ts": "2026-05-04T01:00:00Z", "from_state": "await_worker", "to_state": "partial_persist",
         "handoff": "PARTIAL_PERSIST", "matched_transition_index": 0, "rationale": ""},
    ]
    (tmp_path / "state_transitions.jsonl").write_text(
        "\n".join(json.dumps(e) for e in log) + "\n"
    )
    ok, errors = se.validate_log(tmp_path)
    assert not ok
    assert any("partial_persist" in e for e in errors), f"errors: {errors}"
