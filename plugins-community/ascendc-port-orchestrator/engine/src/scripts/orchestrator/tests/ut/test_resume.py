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


def _seed_npubench_repair_workspace(ws, died_reason):
    """Seed a port-aclnn NPUBench workspace stranded in candidate-repair shape."""
    (ws / "PROGRESS.md").write_text("# gelu\n")
    (ws / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "port_a3_to_a5",
        "reference": {"source": "npubench"},
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
    }))
    _seed_state_log(ws, ("finalize", "await_worker", "candidate repair"))
    (ws / ".rollback_history.jsonl").write_text(json.dumps({
        "gate": "phase_o5_npubench_candidate_contract",
        "rollback_state": "await_worker",
    }) + "\n")
    (ws / ".agent_died_at_await_worker").write_text(json.dumps({
        "state": "await_worker",
        "reason": died_reason,
    }))


def _seed_tilelang_source_workspace(ws):
    """Seed a mid-flight TileLang2AscendC port workspace parked at await_worker."""
    (ws / "PROGRESS.md").write_text("# gelu\n")
    (ws / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "port_a3_to_a5",
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
    }))
    _seed_state_log(ws, ("init", "await_worker", "start"))


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


def test_diagnose_npubench_candidate_repair_construction_failure(tmp_path):
    """Known NPUBench graybox retry failures are recoverable without input."""
    _seed_npubench_repair_workspace(tmp_path, (
        "graybox construction rejected target/answer-bearing curated input; "
        "see construction_manifest.json"
    ))

    status = resume.diagnose("gelu", workspace=tmp_path)

    assert status.action == resume.ResumeAction.NPUBENCH_CANDIDATE_REPAIR
    assert "stale candidate" in status.summary


def test_diagnose_npubench_manifest_drift_is_recoverable(tmp_path):
    """A stale graybox seal marker must not strand NPUBench resume."""
    _seed_npubench_repair_workspace(tmp_path, (
        "graybox construction manifest changed during worker dispatch: "
        "expected old, observed new"
    ))

    status = resume.diagnose("gelu", workspace=tmp_path)

    assert status.action == resume.ResumeAction.NPUBENCH_CANDIDATE_REPAIR


def test_diagnose_npubench_repair_after_o5_mismatch_gate(tmp_path):
    """2026-08-27 (flash_attention_score, port-a3-ops route): an O5 MISMATCH
    rollback keeps the candidate in-workspace for incremental repair; a
    deliverable-shaped candidate (op_kernel/arch35/*.h + kernel/pybind11.cpp)
    trips the answer gate on the next spawn.  The resume classifier must
    accept phase_o5_mismatch as a repair-producing gate too.
    """
    _seed_npubench_repair_workspace(tmp_path, (
        "graybox construction rejected target/answer-bearing curated input; "
        "see construction_manifest.json"
    ))
    (tmp_path / ".rollback_history.jsonl").write_text(json.dumps({
        "gate": "phase_o5_mismatch",
        "rollback_state": "await_worker",
    }) + "\n")

    status = resume.diagnose("gelu", workspace=tmp_path)

    assert status.action == resume.ResumeAction.NPUBENCH_CANDIDATE_REPAIR


def test_diagnose_npubench_repair_without_port_source_kind(tmp_path):
    """CANN ops-repo sources (port-a3-ops default route) record no
    port_source kind in .opgen_state.json; the classifier must not reject
    the repair crash class for that (2026-08-27 flash_attention_score:
    graybox seal-marker drift after a double-orchestrator collision, latest
    gate phase_o5_mismatch, port_source absent).
    """
    _seed_npubench_repair_workspace(tmp_path, (
        "graybox construction manifest changed during worker dispatch: "
        "expected old, observed new"
    ))
    state = json.loads((tmp_path / ".opgen_state.json").read_text())
    del state["port_source"]
    (tmp_path / ".opgen_state.json").write_text(json.dumps(state))

    status = resume.diagnose("gelu", workspace=tmp_path)

    assert status.action == resume.ResumeAction.NPUBENCH_CANDIDATE_REPAIR


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


def test_execute_npubench_candidate_repair_archives_and_reinvokes(tmp_path, monkeypatch):
    """Resume cleans the stale candidate before the next isolated spawn."""
    _seed_npubench_repair_workspace(tmp_path, (
        "graybox construction rejected target/answer-bearing curated input; "
        "see construction_manifest.json"
    ))
    (tmp_path / "kernel" / "arch35").mkdir(parents=True)
    (tmp_path / "kernel" / "arch35" / "pybind11.cpp").write_text("stale\n")
    repair_root = tmp_path / "repair-backups"
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(repair_root))
    monkeypatch.setattr(
        resume,
        "_verify_tilelang_source_stage",
        lambda workspace, state: (True, "ok", {}),
    )
    calls = []
    monkeypatch.setattr(
        resume.subprocess,
        "call",
        lambda command: calls.append(command) or 0,
    )

    rc = resume.execute("gelu", workspace=tmp_path)

    assert rc == 0
    assert not (tmp_path / ".agent_died_at_await_worker").exists()
    assert list(tmp_path.glob(".agent_died_at_await_worker.cleaned-*"))
    assert not (tmp_path / "kernel").exists()
    repair = json.loads(
        (tmp_path / ".npubench_candidate_repair.json").read_text()
    )
    assert repair["moved"] == ["kernel/"]
    assert (
        repair_root / tmp_path.name / repair["archive_id"] / "kernel" / "arch35" / "pybind11.cpp"
    ).is_file()
    assert calls


def test_execute_user_decision_pending_returns_two(ws):
    _seed_state_log(ws, ("await_worker", "await_user_decision", "soft"))
    rc = resume.execute("test", workspace=ws, dry_run=True)
    assert rc == 2


def test_execute_dry_run_resumable(ws):
    """Mid-flight + dry_run=True → reports + exits 0 without invoking child."""
    _seed_state_log(ws, ("init", "await_worker", "x"))
    rc = resume.execute("test", workspace=ws, dry_run=True)
    assert rc == 0


def test_execute_uses_tilelang_source_verifier_on_resume(tmp_path, monkeypatch):
    """TileLang2AscendC target snapshots must not go to the legacy arch22 verifier."""
    _seed_tilelang_source_workspace(tmp_path)
    calls = []

    monkeypatch.setattr(
        resume,
        "_verify_tilelang_source_stage",
        lambda workspace, state: (calls.append((workspace, state)) or (True, "ok", {})),
    )
    monkeypatch.setattr(
        resume,
        "verify_source_stage",
        lambda *args, **kwargs: pytest.fail("legacy arch22 verifier was selected"),
    )
    monkeypatch.setattr(resume.subprocess, "call", lambda command: 0)

    assert resume.execute("gelu_tilelang", workspace=tmp_path) == 0
    assert calls == [(tmp_path, json.loads((tmp_path / ".opgen_state.json").read_text()))]


def test_execute_dry_run_still_validates_tilelang_source(tmp_path, monkeypatch):
    """Dry-run keeps the source-stage validation contract for resumable work."""
    _seed_tilelang_source_workspace(tmp_path)
    calls = []
    monkeypatch.setattr(
        resume,
        "_verify_tilelang_source_stage",
        lambda workspace, state: (calls.append((workspace, state)) or (True, "ok", {})),
    )
    monkeypatch.setattr(
        resume.subprocess,
        "call",
        lambda *args: pytest.fail("dry-run spawned a child"),
    )

    assert resume.execute("gelu_tilelang", workspace=tmp_path, dry_run=True) == 0
    assert calls


def test_execute_returns_two_when_tilelang_verifier_raises(tmp_path, monkeypatch):
    """A verifier failure is a persisted resume failure, never a process crash."""
    _seed_tilelang_source_workspace(tmp_path)
    monkeypatch.setattr(
        resume,
        "_verify_tilelang_source_stage",
        lambda *args: (_ for _ in ()).throw(RuntimeError("broken verifier")),
    )

    assert resume.execute("gelu_tilelang", workspace=tmp_path) == 2


def test_execute_terminal_tilelang_workspace_skips_source_verifier(tmp_path, monkeypatch):
    """A terminal status must be reported without revalidating its old snapshot."""
    (tmp_path / "PROGRESS.md").write_text("# gelu\n")
    (tmp_path / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "port_a3_to_a5",
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
    }))
    _seed_state_log(tmp_path, ("finalize", "done", "pipeline_done"))
    monkeypatch.setattr(
        resume,
        "_verify_tilelang_source_stage",
        lambda *args: pytest.fail("terminal resume revalidated source stage"),
    )

    assert resume.execute("gelu_tilelang", workspace=tmp_path) == 0


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


# ---------------------------------------------------------------------------
# perf backfill guards (2026-08-25, codex review F1)
# ---------------------------------------------------------------------------
def _seed_done_with_deferred_perf(ws):
    """done op whose harness verification carries a DEFERRED perf placeholder."""
    _seed_state_log(ws, ("finalize", "done", "pipeline_done"))
    (ws / "verification.json").write_text(json.dumps({
        "performance": {"status": "DEFERRED", "perf_deferred": True},
    }))


def test_diagnose_perf_backfill_reenters_finalize(ws, monkeypatch):
    """BACKFILL=1 without SKIP_PERF re-enters finalize for the measurement."""
    _seed_done_with_deferred_perf(ws)
    monkeypatch.setenv("CANNBOT_NPUBENCH_PERF_BACKFILL", "1")
    monkeypatch.delenv("CANNBOT_NPUBENCH_SKIP_PERF", raising=False)

    s = resume.diagnose("test", workspace=ws)

    assert s.action == resume.ResumeAction.RESUMABLE
    assert s.current_state == "finalize"


def test_diagnose_perf_backfill_refused_when_skip_perf_still_set(ws, monkeypatch):
    """SKIP_PERF=1 + BACKFILL=1 must NOT re-enter finalize: the re-run O5
    would emit another DEFERRED placeholder and the op would stay 'done'
    with perf never measured (silent spin).
    """
    _seed_done_with_deferred_perf(ws)
    monkeypatch.setenv("CANNBOT_NPUBENCH_PERF_BACKFILL", "1")
    monkeypatch.setenv("CANNBOT_NPUBENCH_SKIP_PERF", "1")

    s = resume.diagnose("test", workspace=ws)

    assert s.action == resume.ResumeAction.NONE_TERMINAL
    assert s.current_state == "done"
    assert "CANNBOT_NPUBENCH_SKIP_PERF" in s.summary
    # No synthetic done→finalize transition was recorded.
    tail = json.loads(
        (ws / "state_transitions.jsonl").read_text().splitlines()[-1]
    )
    assert tail["to_state"] == "done"


# ---------------------------------------------------------------------------
# candidate repair keeps npubench_evidence/ (2026-08-25, codex review F2)
# ---------------------------------------------------------------------------
def test_candidate_repair_preserves_npubench_evidence(tmp_path, monkeypatch):
    """The repair reset must keep npubench_evidence/ in the workspace — the
    next worker brief reads preflight_target_receipt.json from it and is
    forbidden from re-running preflight (kw_brief_port_a3).
    """
    evidence = tmp_path / "npubench_evidence"
    evidence.mkdir()
    receipt = evidence / "preflight_target_receipt.json"
    receipt.write_text(json.dumps({"soc": "Ascend950DT_9582"}))
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "pybind11.cpp").write_text("stale\n")
    backup_root = tmp_path / "repair-backups"
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(backup_root))

    # Bind the protected helper by import (the module is already on sys.path);
    # done inside the test so any monkeypatching stays effective.
    from resume import _prepare_npubench_candidate_repair

    record = _prepare_npubench_candidate_repair(tmp_path)

    assert receipt.is_file()
    assert "npubench_evidence/" not in record["moved"]
    assert "kernel/" in record["moved"]
    archive = backup_root / tmp_path.name / record["archive_id"]
    assert not (archive / "npubench_evidence").exists()
