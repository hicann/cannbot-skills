# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for recover.py — orchestrator zombie diagnosis + cleanup."""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import recover  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_etime
# ---------------------------------------------------------------------------
def test_parse_etime_ss():
    """ps etime can be MM:SS or HH:MM:SS or DD-HH:MM:SS."""
    assert getattr(recover, '_parse_etime')("00:30") == 30
    assert getattr(recover, '_parse_etime')("01:30") == 90
    assert getattr(recover, '_parse_etime')("01:00:00") == 3600
    assert getattr(recover, '_parse_etime')("02-01:00:00") == 2 * 86400 + 3600
    assert getattr(recover, '_parse_etime')("") == 0
    assert getattr(recover, '_parse_etime')("garbage") == 0


# ---------------------------------------------------------------------------
# _classify_proc
# ---------------------------------------------------------------------------
def test_classify_batch_dispatcher():
    p = {"cmd": "python3 src/scripts/orchestrator/orchestrator.py --batch op1,op2 --max-lanes 2"}
    assert getattr(recover, '_classify_proc')(p) == recover.ProcKind.BATCH_DISPATCHER


def test_classify_single_op():
    p = {"cmd": "python3 /home/x/orchestrator.py 1_gelu --lane 0"}
    assert getattr(recover, '_classify_proc')(p) == recover.ProcKind.SINGLE_OP_ORCHESTRATOR


def test_classify_harness_agent():
    p = {"cmd": "claude --print --agent aog-kernel-worker --output-format stream-json"}
    assert getattr(recover, '_classify_proc')(p) == recover.ProcKind.HARNESS_AGENT


def test_classify_other():
    p = {"cmd": "python3 unrelated.py"}
    assert getattr(recover, '_classify_proc')(p) == recover.ProcKind.OTHER


# ---------------------------------------------------------------------------
# _extract_op_from_cmd
# ---------------------------------------------------------------------------
def test_extract_op_from_orchestrator_cmd():
    cmd = "python3 /home/x/orchestrator.py 1_gelu --lane 0"
    assert getattr(recover, '_extract_op_from_cmd')(cmd) == "1_gelu"


def test_extract_op_from_claude_g7_slug():
    cmd = "claude --print --agent aog-kernel-worker layernorm-kw-1 — kernel-worker spawn"
    assert getattr(recover, '_extract_op_from_cmd')(cmd) == "layernorm"


def test_extract_op_returns_none_when_no_match():
    cmd = "python3 unrelated.py"
    assert getattr(recover, '_extract_op_from_cmd')(cmd) is None


# ---------------------------------------------------------------------------
# Stale marker detection
# ---------------------------------------------------------------------------
def _write_marker(ws: Path, state: str, ts_str: str) -> Path:
    """Helper: create .agent_died_at_<state> marker."""
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / f".agent_died_at_{state}"
    p.write_text(json.dumps({"ts": ts_str, "state": state, "reason": "test"}))
    return p


def _set_mtime(p: Path, hours_ago: float) -> None:
    """Set both atime + mtime to N hours in the past."""
    import os
    target = _dt.datetime.now().timestamp() - hours_ago * 3600
    os.utime(p, (target, target))


def test_find_stale_markers_marker_old_no_log(tmp_path, monkeypatch):
    """No state_transitions.jsonl + marker > 6h old → STALE."""
    workspace = tmp_path / "workspace"
    op_dir = workspace / "test_op"
    marker = _write_marker(op_dir, "await_worker", "2026-05-04T01:00:00Z")
    _set_mtime(marker, hours_ago=10)

    # Patch _HERE so root resolves to our tmp
    monkeypatch.setattr(recover, "_HERE",
                          tmp_path / "src" / "scripts" / "orchestrator" / "recover.py")
    out = recover.find_stale_markers()
    assert len(out) == 1
    assert out[0].is_stale is True
    assert "6h" in out[0].reason


def test_find_stale_markers_marker_recent_no_log(tmp_path, monkeypatch):
    """Recent marker, no follow-up state log activity → NOT stale."""
    workspace = tmp_path / "workspace"
    op_dir = workspace / "test_op"
    marker = _write_marker(op_dir, "await_worker", "2026-05-04T01:00:00Z")
    _set_mtime(marker, hours_ago=1)

    monkeypatch.setattr(recover, "_HERE",
                          tmp_path / "src" / "scripts" / "orchestrator" / "recover.py")
    out = recover.find_stale_markers()
    assert len(out) == 1
    assert out[0].is_stale is False


def test_find_stale_markers_state_log_newer(tmp_path, monkeypatch):
    """Marker old, state_transitions.jsonl mtime newer → STALE."""
    workspace = tmp_path / "workspace"
    op_dir = workspace / "test_op"
    marker = _write_marker(op_dir, "await_worker", "2026-05-04T01:00:00Z")
    _set_mtime(marker, hours_ago=2)

    log = op_dir / "state_transitions.jsonl"
    log.write_text(json.dumps({
        "ts": "2026-05-04T05:00:00Z", "from_state": "await_probe",
        "to_state": "finalize", "handoff": "done",
        "matched_transition_index": 0, "rationale": "", "iter_counts_snapshot": {},
    }) + "\n")
    # log mtime is "now" (just written), marker mtime is 2h ago
    monkeypatch.setattr(recover, "_HERE",
                          tmp_path / "src" / "scripts" / "orchestrator" / "recover.py")
    out = recover.find_stale_markers()
    assert len(out) == 1
    assert out[0].is_stale is True
    assert "newer than marker" in out[0].reason


# ---------------------------------------------------------------------------
# kill_zombie behavior (dry-run only — real kills mocked elsewhere)
# ---------------------------------------------------------------------------
def test_kill_zombie_skips_non_zombie():
    proc = recover.ProcInfo(
        pid=1234, ppid=1, cmd="claude --print",
        elapsed_sec=600, kind=recover.ProcKind.HARNESS_AGENT,
        health=recover.ProcHealth.HEALTHY,
    )
    result = recover.kill_zombie(proc, dry_run=True)
    assert "skipped" in result
    assert "not a zombie" in result["skipped"]


def test_kill_zombie_skips_non_claude():
    """Kill strategy only applies to harness_agent; orchestrator kill is
    different (you'd kill the parent batch instead).
    """
    proc = recover.ProcInfo(
        pid=1234, ppid=1, cmd="python3 orchestrator.py 1_gelu --lane 0",
        elapsed_sec=600, kind=recover.ProcKind.SINGLE_OP_ORCHESTRATOR,
        health=recover.ProcHealth.ZOMBIE,
    )
    result = recover.kill_zombie(proc, dry_run=True)
    assert "skipped" in result
    assert "harness_agent" in result["skipped"]


# ---------------------------------------------------------------------------
# clean_stale_markers
# ---------------------------------------------------------------------------
def test_clean_stale_markers_dry_run(tmp_path):
    op_dir = tmp_path / "test_op"
    marker = _write_marker(op_dir, "await_worker", "2026-05-04T01:00:00Z")

    sm = recover.StaleMarker(
        workspace=op_dir, op="test_op", marker_path=marker,
        marker_ts="2026-05-04T01:00:00Z", last_state_log_ts=None,
        age_minutes=400, is_stale=True, reason="6h+ old",
    )
    out = recover.clean_stale_markers([sm], dry_run=True)
    assert len(out) == 1
    assert out[0]["dry_run"] is True
    assert "renamed_to" not in out[0]
    # Marker still exists
    assert marker.exists()


def test_clean_stale_markers_actual(tmp_path):
    op_dir = tmp_path / "test_op"
    marker = _write_marker(op_dir, "await_worker", "2026-05-04T01:00:00Z")

    sm = recover.StaleMarker(
        workspace=op_dir, op="test_op", marker_path=marker,
        marker_ts="2026-05-04T01:00:00Z", last_state_log_ts=None,
        age_minutes=400, is_stale=True, reason="6h+ old",
    )
    out = recover.clean_stale_markers([sm], dry_run=False)
    assert len(out) == 1
    assert "renamed_to" in out[0]
    # Original gone, .cleaned-* version exists
    assert not marker.exists()
    cleaned = list(op_dir.glob(".agent_died_at_await_worker.cleaned-*"))
    assert len(cleaned) == 1


def test_find_stale_markers_skips_already_cleaned(tmp_path, monkeypatch):
    """Idempotency: re-running on a workspace with .cleaned-* markers should
    NOT re-process them (was a real bug 2026-05-04 — re-cleaning created
    .cleaned-X.cleaned-Y filenames).
    """
    workspace = tmp_path / "workspace"
    op_dir = workspace / "test_op"
    op_dir.mkdir(parents=True)
    cleaned = op_dir / ".agent_died_at_await_worker.cleaned-20260504T120000Z"
    cleaned.write_text(json.dumps({"ts": "x", "state": "x", "reason": "x"}))

    monkeypatch.setattr(recover, "_HERE",
                          tmp_path / "src" / "scripts" / "orchestrator" / "recover.py")
    out = recover.find_stale_markers()
    assert out == [], f"already-cleaned marker should be skipped, got {out}"


def test_clean_stale_markers_skips_current(tmp_path):
    """Markers flagged is_stale=False should not be touched."""
    op_dir = tmp_path / "test_op"
    marker = _write_marker(op_dir, "await_worker", "2026-05-04T01:00:00Z")

    sm = recover.StaleMarker(
        workspace=op_dir, op="test_op", marker_path=marker,
        marker_ts="2026-05-04T01:00:00Z", last_state_log_ts=None,
        age_minutes=10, is_stale=False, reason="fresh",
    )
    out = recover.clean_stale_markers([sm], dry_run=False)
    assert out == []
    assert marker.exists()


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------
def test_format_report_empty():
    rep = recover.RecoverReport()
    text = recover.format_report(rep)
    assert "no orchestrator/harness backend processes running" in text


def test_format_report_with_zombie():
    proc = recover.ProcInfo(
        pid=2917304, ppid=2905065, cmd="claude --print --agent aog-kernel-optimizer",
        elapsed_sec=6900, kind=recover.ProcKind.HARNESS_AGENT,
        health=recover.ProcHealth.ZOMBIE, op="1_gelu",
        notes=["stream_log ends with result event (is_error=True, api_error_status=429)"],
    )
    rep = recover.RecoverReport(procs=[proc])
    text = recover.format_report(rep)
    assert "2917304" in text
    assert "zombie" in text
    assert "1_gelu" in text


def test_format_report_with_stale_marker():
    op_dir = Path("/tmp/test_op")  # path doesn't need to exist for formatter
    sm = recover.StaleMarker(
        workspace=op_dir, op="9_topktopp",
        marker_path=op_dir / ".agent_died_at_await_worker",
        marker_ts="2026-05-04T01:00:00Z", last_state_log_ts="2026-05-04T08:00:00Z",
        age_minutes=730, is_stale=True,
        reason="state_transitions.jsonl mtime is 12000s newer than marker",
    )
    rep = recover.RecoverReport(stale_markers=[sm])
    text = recover.format_report(rep)
    assert "STALE" in text
    assert "9_topktopp" in text


def test_format_report_with_resumable_ops():
    rep = recover.RecoverReport(resumable_ops=["1_gelu", "5_cumsum"])
    text = recover.format_report(rep)
    assert "Resumable ops (2)" in text
    assert "1_gelu" in text
    assert "--resume" in text


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def test_audit_log_writes_to_workspace(tmp_path):
    ws = tmp_path / "test_op"
    ws.mkdir()
    getattr(recover, '_audit_log')(ws, {"action": "test", "data": 42})
    log = ws / ".recover_log.jsonl"
    assert log.exists()
    entry = json.loads(log.read_text().splitlines()[-1])
    assert entry["action"] == "test"
    assert entry["data"] == 42
    assert "ts" in entry
    assert entry["ts"].endswith("Z")


def test_audit_log_appends(tmp_path):
    ws = tmp_path / "test_op"
    ws.mkdir()
    getattr(recover, '_audit_log')(ws, {"action": "first"})
    getattr(recover, '_audit_log')(ws, {"action": "second"})
    lines = (ws / ".recover_log.jsonl").read_text().splitlines()
    assert len(lines) == 2
