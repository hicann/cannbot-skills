# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for resume.py (Track C #4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import resume  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """Workspace with PROGRESS.md present (so it counts as a real op dir)."""
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    (tmp_path / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test",
        "opgen_mode": "backward",
    }))
    return tmp_path


def _seed_state_log(ws, *transitions):
    """Helper: write state_transitions.jsonl with given transitions."""
    log = ws / "state_transitions.jsonl"
    lines = []
    for from_s, to_s, handoff in transitions:
        lines.append(json.dumps({
            "ts": "2026-05-04T05:00:00Z",
            "from_state": from_s, "to_state": to_s,
            "handoff": handoff, "matched_transition_index": 0,
            "rationale": "test seed", "iter_counts_snapshot": {},
        }))
    log.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# diagnose — each ResumeAction code path
# ---------------------------------------------------------------------------
def test_diagnose_unknown_when_workspace_missing():
    s = resume.diagnose("nonexistent", workspace=Path("/tmp/no_such_ws_12345"))
    assert s.action == resume.ResumeAction.UNKNOWN


def test_diagnose_terminal_done(ws):
    # P0dd (2026-05-05): `done` is now the truly-terminal state. `finalize`
    # is non-terminal — has the in-process finalize_pipeline as its agent.
    _seed_state_log(ws, ("finalize", "done", "pipeline_done"))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.NONE_TERMINAL
    assert s.current_state == "done"


def test_diagnose_finalize_is_resumable(ws):
    # An op left at `finalize` state means the pipeline didn't run yet —
    # resume should treat it as RESUMABLE so orchestrator can run the pipeline.
    _seed_state_log(ws, ("await_worker", "finalize", "done"))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.RESUMABLE
    assert s.current_state == "finalize"


def test_diagnose_terminal_abort(ws):
    _seed_state_log(ws, ("await_worker", "abort", "contract violation"))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.NONE_TERMINAL
    assert s.current_state == "abort"


def test_diagnose_pause_state_no_decision_yet(ws):
    _seed_state_log(ws, ("await_worker", "await_user_decision", "soft judgment"))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.USER_DECISION_PENDING


def test_diagnose_pause_state_with_decision_ready(ws):
    _seed_state_log(ws, ("await_worker", "await_user_decision", "soft judgment"))
    (ws / "user_decision.md").write_text("next_state: await_optimizer\n")
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.USER_DECISION_READY


def test_diagnose_agent_died(ws):
    _seed_state_log(ws, ("init", "await_worker", "start"))
    (ws / ".agent_died_at_await_worker").write_text(json.dumps({
        "ts": "2026-05-04T05:00:00Z",
        "state": "await_worker",
        "reason": "claude exited 1: subprocess timeout",
    }, indent=2))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.AGENT_DIED
    assert s.died_at_state == "await_worker"
    assert "subprocess timeout" in s.died_reason


def test_diagnose_agent_died_takes_priority_over_terminal(ws):
    """Even if the log shows terminal, an unprocessed died-marker wins
    (operator must clear the marker explicitly).
    """
    _seed_state_log(ws,
                    ("await_worker", "finalize", "done"))
    (ws / ".agent_died_at_await_worker").write_text(json.dumps({
        "state": "await_worker", "reason": "leftover marker",
    }))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.AGENT_DIED


def test_diagnose_mid_flight_resumable(ws):
    """In mid-flight (await_probe), resume is offered."""
    _seed_state_log(ws,
                    ("init", "await_worker", "first spawn"),
                    ("await_worker", "await_probe", "@aog-precision-probe stuck"))
    s = resume.diagnose("test", workspace=ws)
    assert s.action == resume.ResumeAction.RESUMABLE
    assert s.current_state == "await_probe"


def test_diagnose_handoff_truncated(ws):
    """Long handoff text is truncated for the resume status."""
    long_handoff = "x" * 1000
    _seed_state_log(ws, ("await_worker", "await_probe", long_handoff))
    s = resume.diagnose("test", workspace=ws)
    assert s.last_handoff is not None
    assert len(s.last_handoff) <= 300


# ---------------------------------------------------------------------------
# execute — dry-run paths
# ---------------------------------------------------------------------------
def test_execute_terminal_returns_zero(ws):
    _seed_state_log(ws, ("await_worker", "finalize", "done"))
    rc = resume.execute("test", workspace=ws, dry_run=True)
    assert rc == 0


def test_execute_agent_died_returns_two_no_invoke(ws):
    """Codex C4: no auto-retry on agent_died. execute() must NOT spawn."""
    (ws / ".agent_died_at_await_worker").write_text(json.dumps({
        "state": "await_worker", "reason": "test",
    }))
    _seed_state_log(ws, ("init", "await_worker", "x"))
    rc = resume.execute("test", workspace=ws, dry_run=True)
    assert rc == 2  # surfaced as failure


def test_execute_user_decision_pending_returns_two(ws):
    _seed_state_log(ws, ("await_worker", "await_user_decision", "soft"))
    rc = resume.execute("test", workspace=ws, dry_run=True)
    assert rc == 2


def test_execute_dry_run_resumable(ws):
    """Mid-flight + dry_run=True → reports + exits 0 without invoking child."""
    _seed_state_log(ws, ("init", "await_worker", "x"))
    rc = resume.execute("test", workspace=ws, dry_run=True)
    assert rc == 0


# ---------------------------------------------------------------------------
# scan_all
# ---------------------------------------------------------------------------
def test_scan_all_returns_empty_for_no_workspace(tmp_path):
    """scan_all with empty workspace dir — empty list."""
    (tmp_path).mkdir(exist_ok=True)
    out = resume.scan_all(root=tmp_path)
    assert out == []


def test_scan_all_skips_dirs_without_progress(tmp_path):
    (tmp_path / "no_progress").mkdir()
    (tmp_path / "real_op").mkdir()
    (tmp_path / "real_op" / "PROGRESS.md").write_text("# x\n")
    out = resume.scan_all(root=tmp_path)
    assert len(out) == 1
    assert out[0].op == "real_op"


def test_scan_all_diagnoses_each(tmp_path):
    # P0dd (2026-05-05): `done` is now the terminal state, not `finalize`.
    (tmp_path / "op1").mkdir()
    (tmp_path / "op1" / "PROGRESS.md").write_text("# x\n")
    _seed_state_log(tmp_path / "op1", ("finalize", "done", "pipeline_done"))

    (tmp_path / "op2").mkdir()
    (tmp_path / "op2" / "PROGRESS.md").write_text("# x\n")
    (tmp_path / "op2" / ".agent_died_at_await_worker").write_text(
        json.dumps({"state": "await_worker", "reason": "test"})
    )
    _seed_state_log(tmp_path / "op2", ("init", "await_worker", "x"))

    out = resume.scan_all(root=tmp_path)
    by_op = {s.op: s for s in out}
    assert by_op["op1"].action == resume.ResumeAction.NONE_TERMINAL
    assert by_op["op2"].action == resume.ResumeAction.AGENT_DIED


# ── P135.RL (2026-05-18): resume preserves workspace lane ────


def test_p135rl_load_workspace_lane_reads_state(tmp_path):
    """P135.RL: _load_workspace_lane reads `.opgen_state.json.lane`."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "foo", "lane": 2})
    )
    assert getattr(resume, '_load_workspace_lane')(tmp_path) == 2


def test_p135rl_load_workspace_lane_missing_state_returns_none(tmp_path):
    """No .opgen_state.json → returns None (caller falls back to default)."""
    assert getattr(resume, '_load_workspace_lane')(tmp_path) is None


def test_p135rl_load_workspace_lane_no_lane_field_returns_none(tmp_path):
    """State exists but no `lane` field → returns None."""
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "foo"})
    )
    assert getattr(resume, '_load_workspace_lane')(tmp_path) is None


def test_p135rl_load_workspace_lane_invalid_value_returns_none(tmp_path):
    """Lane field has non-int / negative value → returns None
    (don't import garbage data; caller falls back to default 0).
    """
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "foo", "lane": "two"})
    )
    assert getattr(resume, '_load_workspace_lane')(tmp_path) is None
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "foo", "lane": -1})
    )
    assert getattr(resume, '_load_workspace_lane')(tmp_path) is None


def test_p135rl_load_workspace_lane_zero_is_valid_but_yields_default_behavior(tmp_path):
    """lane=0 in state file is technically valid (default lane);
    _load_workspace_lane returns 0 — caller's `if lane == 0:` then
    treats this same as 'no state', resulting in default 0 either way.
    This is intentional: lane=0 is the harness default, no override needed.
    """
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"schema_version": 1, "op": "foo", "lane": 0})
    )
    assert getattr(resume, '_load_workspace_lane')(tmp_path) == 0


def test_p135rl_load_workspace_lane_malformed_json_returns_none(tmp_path):
    """Malformed state file → returns None (defensive parse)."""
    (tmp_path / ".opgen_state.json").write_text("not json {")
    assert getattr(resume, '_load_workspace_lane')(tmp_path) is None
