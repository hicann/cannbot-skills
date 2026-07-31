# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0r (2026-05-05): resume.py auto-recovery for FW-transient agent_died.

Origin: op#10_layernorm 2026-05-05. Researcher hit FW transient ("API
returned an empty or malformed response (HTTP 200)") mid-investigation,
claude subprocess exited 1, orchestrator wrote .agent_died_at_await_researcher
marker. resume --resume correctly diagnosed agent_died but explicitly
refused to auto-retry per codex C4 — required human-in-the-loop "inspect +
clear marker + re-run".

User direction 2026-05-05: "we need 0 interaction experience. why can we
call scripts to resume?"

Fix: resume.diagnose() classifies failures. FW-transient pattern (matches
agent_transport's _FW_TRANSIENT_PATTERNS) + retry budget remaining →
ResumeAction.AUTO_RECOVERABLE. resume.execute() handles AUTO_RECOVERABLE
by archiving the marker (.cleaned-<ts> suffix), incrementing per-state
retry counter, then re-invoking orchestrator. Persistent failures
eventually exhaust the budget and fall back to AGENT_DIED (codex C4 guard
preserved for unknown failure modes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import resume  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """Workspace with a state log entry placing it at await_researcher."""
    log_entry = {
        "ts": "2026-05-05T07:00:00Z",
        "from_state": "await_user_decision",
        "to_state": "await_researcher",
        "handoff": "→ orchestrator: await_researcher per user_decision.md",
        "matched_transition_index": 0,
        "rationale": "test seed",
        "iter_counts_snapshot": {},
    }
    (tmp_path / "state_transitions.jsonl").write_text(json.dumps(log_entry) + "\n")
    (tmp_path / "PROGRESS.md").write_text("# test\nMode: backward\n")
    (tmp_path / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "backward",
    }))
    return tmp_path


def _write_died_marker(ws, state: str, reason: str):
    payload = {
        "ts": "2026-05-05T07:01:00Z",
        "state": state,
        "reason": reason,
    }
    (ws / f".agent_died_at_{state}").write_text(json.dumps(payload, indent=2))


def test_p0r_fw_transient_classified_auto_recoverable(ws):
    """Marker reason matches FW-transient pattern + retry count 0 →
    ResumeAction.AUTO_RECOVERABLE (not AGENT_DIED).
    """
    _write_died_marker(ws, "await_researcher",
                       "claude (stream-json) exited 1 for agent='aog-researcher'\nstderr: ")
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.AUTO_RECOVERABLE, \
        f"FW-transient should classify auto-recoverable, got {status.action}"
    assert status.died_at_state == "await_researcher"
    assert "FW-transient" in status.summary or "retry" in status.summary


def test_p0r_fw_transient_with_explicit_pattern(ws):
    """Direct match on the API-error pattern (not just exited-1 wrapper)."""
    _write_died_marker(ws, "await_researcher",
                       "API returned an empty or malformed response (HTTP 200) — check for a proxy")
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.AUTO_RECOVERABLE


def test_p0r_unknown_failure_keeps_codex_c4_guard(ws):
    """Reason that doesn't match any FW-transient pattern → still AGENT_DIED
    (codex C4 — no auto-retry on unknown failures).
    """
    _write_died_marker(ws, "await_researcher",
                       "Some real kernel bug: SIGSEGV in worker.cpp:42")
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.AGENT_DIED, \
        f"Unknown failure should NOT auto-retry, got {status.action}"


def test_p0r_retry_budget_exhausted_falls_back_to_agent_died(ws):
    """After FW_AUTO_RETRY_CAP retries, classify as AGENT_DIED (escalate)."""
    _write_died_marker(ws, "await_researcher",
                       "claude (stream-json) exited 1 for agent='aog-researcher'")
    # Pre-populate retry counter at the cap
    (ws / getattr(resume, "_RETRY_COUNT_FILE")).write_text(json.dumps({
        "await_researcher": resume.FW_AUTO_RETRY_CAP
    }))
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.AGENT_DIED
    assert "exhausted" in status.summary.lower() or "budget" in status.summary.lower()


def test_p0r_increment_retry_count(ws):
    """_increment_retry_count persists across calls."""
    assert getattr(resume, "_get_retry_count")(ws, "await_researcher") == 0
    getattr(resume, "_increment_retry_count")(ws, "await_researcher")
    assert getattr(resume, "_get_retry_count")(ws, "await_researcher") == 1
    getattr(resume, "_increment_retry_count")(ws, "await_researcher")
    assert getattr(resume, "_get_retry_count")(ws, "await_researcher") == 2
    # Per-state counters are independent
    assert getattr(resume, "_get_retry_count")(ws, "await_worker") == 0
    getattr(resume, "_increment_retry_count")(ws, "await_worker")
    assert getattr(resume, "_get_retry_count")(ws, "await_worker") == 1
    assert getattr(resume, "_get_retry_count")(ws, "await_researcher") == 2  # unchanged


def test_p0r_cleaned_markers_ignored(ws):
    """*.cleaned-* suffixed markers (post-recovery archives) are NOT treated
    as active died markers — only fresh ones.
    """
    # Archived (cleaned) marker — should be ignored
    payload = {"ts": "2026-05-05T07:00:00Z", "state": "await_researcher",
               "reason": "old transient"}
    (ws / ".agent_died_at_await_researcher.cleaned-1234567890").write_text(
        json.dumps(payload))
    # No active marker
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action != resume.ResumeAction.AGENT_DIED
    assert status.action != resume.ResumeAction.AUTO_RECOVERABLE


def test_p0r_execute_archives_marker_and_increments_counter(ws, monkeypatch):
    """End-to-end: AUTO_RECOVERABLE → execute archives marker (.cleaned-<ts>),
    increments counter, then would invoke orchestrator (we mock the subprocess).
    """
    _write_died_marker(ws, "await_researcher",
                       "claude (stream-json) exited 1 for agent='aog-researcher'")

    # Pre-flight: marker present, counter at 0
    assert (ws / ".agent_died_at_await_researcher").exists()
    assert getattr(resume, "_get_retry_count")(ws, "await_researcher") == 0

    # Mock subprocess.call to avoid actually invoking orchestrator
    import subprocess as sp
    calls = []

    def fake_call(cmd, *args, **kwargs):
        calls.append(cmd)
        return 0
    monkeypatch.setattr(sp, "call", fake_call)
    monkeypatch.setattr(resume.subprocess, "call", fake_call)

    rc = resume.execute("test_op", workspace=ws, lane=0)
    assert rc == 0

    # Marker archived (renamed with .cleaned-<ts> suffix)
    assert not (ws / ".agent_died_at_await_researcher").exists()
    cleaned = list(ws.glob(".agent_died_at_await_researcher.cleaned-*"))
    assert len(cleaned) == 1, f"Expected 1 cleaned marker, got {cleaned}"

    # Counter incremented
    assert getattr(resume, "_get_retry_count")(ws, "await_researcher") == 1

    # subprocess was invoked (orchestrator re-spawned)
    assert len(calls) == 1
    assert "10_layernorm" in calls[0] or "test_op" in calls[0]
