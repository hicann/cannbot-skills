# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration smoke tests — orchestrator + state machine + resume seam.

Each scenario mirrors a real bug class that escaped pure-Python unit tests:
- P0aa: await_optimizer iter_cap legitimate exhaustion → finalize PARTIAL_PERSIST
- P0bb: orchestrator iter=0 empty log + PROGRESS.md drift across re-derives
- P0cc: resume.diagnose short-circuits via bootstrap inference
- P0ab: P0t replay drops abort, log empty, bootstrap canonicalizes via PROGRESS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import orchestrator  # noqa: E402
import resume  # noqa: E402
import state_executor  # noqa: E402

from .conftest import (  # noqa: E402
    _mk_agent_result,
    append_progress_handoff,
    write_verification,
)


# ---------------------------------------------------------------------------
# P0bb regression — empty log + worker rewrites PROGRESS.md mid-session.
# Pre-fix symptom: snap.current_state at iter=0 returns await_worker. Worker
# runs, appends "→ orchestrator: done" to PROGRESS.md. Orchestrator's
# next_state internally calls current_state — re-bootstraps from PROGRESS.md,
# returns finalize, state_machine.next("finalize", "done") errors no-match.
# ---------------------------------------------------------------------------
def test_p0bb_orchestrator_iter0_drift(
    fake_agent_dispatch, stub_skills, make_workspace, monkeypatch
):
    ws = make_workspace("test_op_p0bb", progress_md="# PROGRESS\nMode: backward\n")

    def worker_step(*, workspace, **kw):
        # Real worker writes verification.json + appends final handoff to PROGRESS
        write_verification(
            workspace,
            performance={
                "ratio": 1.5, "status": "PASS",
                "method": "same_wrapper symmetric=true method_symmetric",
                "independent_re_measure": {"ran": True, "ratio": 1.5, "delta_vs_kw_self_report": 0.0},
            },
        )
        append_progress_handoff(
            workspace, "→ orchestrator: done — Pass A 51/51; perf 1.5x"
        )
        return _mk_agent_result(
            output_text="→ orchestrator: done — Pass A 51/51; perf 1.5x"
        )

    fake_agent_dispatch.add(worker_step)

    monkeypatch.chdir(ws.parent)
    exit_code = orchestrator.run_single_op(
        op="test_op_p0bb", workspace=ws, lane=0
    )

    # Pre-fix: would return 5 (state machine error from finalize).
    # Post-fix: bootstrap canonicalizes once, log non-empty on next read,
    # explicit next_state records await_worker→target correctly.
    assert exit_code == 0, f"orchestrator should finalize cleanly, got {exit_code}"

    log = (ws / "state_transitions.jsonl").read_text().strip().splitlines()
    transitions = [
        (json.loads(line)["from_state"], json.loads(line)["to_state"])
        for line in log
    ]
    assert ("await_worker", "finalize") in transitions or any(
        t[1] == "finalize" for t in transitions
    ), f"expected finalize transition; got {transitions}"


# ---------------------------------------------------------------------------
# P0cc regression — resume on workspace where worker completed but log empty.
# Pre-fix: resume.diagnose calls current_state, P0o bootstrap returns finalize,
# diagnose returns NONE_TERMINAL, exit 0 — but no audit trail written.
# Post-fix: bootstrap auto-persists init→initial→finalize chain.
# ---------------------------------------------------------------------------
def test_p0cc_resume_canonicalizes_completed_workspace(make_workspace):
    ws = make_workspace(
        "test_op_p0cc",
        progress_md=(
            "# PROGRESS\nMode: backward\n\n"
            "## Phase E\n\n"
            "→ orchestrator: done — Pass A 51/51 PASS_T1; perf 8.4x\n"
        ),
    )
    write_verification(ws)

    # Resume — should NOT crash, should canonicalize bootstrap chain
    resume.diagnose("test_op_p0cc", workspace=ws)

    log_path = ws / "state_transitions.jsonl"
    assert log_path.exists() and log_path.stat().st_size > 0, (
        "diagnose call must have triggered bootstrap canonicalization"
    )
    log_lines = log_path.read_text().strip().splitlines()
    assert len(log_lines) >= 2, f"expected canonical chain; got {log_lines}"
    first = json.loads(log_lines[0])
    assert first["from_state"] == "init"
    assert "P0bb" in first["rationale"]


# ---------------------------------------------------------------------------
# P0bb stability — calling current_state twice returns same result, even if
# PROGRESS.md is rewritten between calls.
# ---------------------------------------------------------------------------
def test_p0bb_current_state_stable_across_progress_edits(make_workspace):
    ws = make_workspace(
        "test_op_stable",
        progress_md="# PROGRESS\n\n→ orchestrator: done — first handoff\n",
    )

    state1 = state_executor.current_state(ws)

    # Simulate something rewriting PROGRESS.md (the original drift bug class)
    (ws / "PROGRESS.md").write_text(
        "# PROGRESS\n\n→ orchestrator: PARTIAL_PERSIST — different handoff\n"
    )

    state2 = state_executor.current_state(ws)

    assert state1 == state2, (
        f"current_state must be stable after canonicalization; "
        f"got {state1} → {state2}"
    )


# ---------------------------------------------------------------------------
# P0bb truly-fresh-workspace contract — no spurious log entries.
# ---------------------------------------------------------------------------
def test_p0bb_fresh_workspace_no_spurious_entries(make_workspace):
    ws = make_workspace("test_op_fresh", progress_md="# fresh\n")

    state = state_executor.current_state(ws)
    assert state == "await_worker"

    log_path = ws / "state_transitions.jsonl"
    # No log entry — fresh workspace should not generate phantom transitions
    assert (not log_path.exists()) or log_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# P0aa regression — await_optimizer iter_cap with full pipeline exhaustion
# routes to finalize PARTIAL_PERSIST instead of error 5.
# ---------------------------------------------------------------------------
def test_p0aa_await_optimizer_legitimate_exhaustion(
    fake_agent_dispatch, stub_skills, make_workspace, monkeypatch
):
    # Workspace at await_optimizer with full pipeline evidence already present:
    # - cann_strategy_inference.md (researcher ran)
    # - probe_result.json with classification=requirement
    # - verification.json with perf < 0.6
    # - 5 prior optimizer iters logged (= iter_cap=5)
    state_log = []
    state_log.append({
        "ts": "2026-05-05T01:00:00Z",
        "from_state": "init", "to_state": "await_worker",
        "handoff": "", "matched_transition_index": -1,
        "rationale": "init", "iter_counts_snapshot": {},
    })
    state_log.append({
        "ts": "2026-05-05T01:30:00Z",
        "from_state": "await_worker", "to_state": "await_probe",
        "handoff": "@aog-precision-probe", "matched_transition_index": 0,
        "rationale": "stuck precision", "iter_counts_snapshot": {"worker": 1},
    })
    state_log.append({
        "ts": "2026-05-05T02:00:00Z",
        "from_state": "await_probe", "to_state": "await_researcher",
        "handoff": "→ orchestrator: probe done — requirement",
        "matched_transition_index": 0, "rationale": "probe requirement",
        "iter_counts_snapshot": {"worker": 1, "probe": 1},
    })
    state_log.append({
        "ts": "2026-05-05T02:30:00Z",
        "from_state": "await_researcher", "to_state": "await_optimizer",
        "handoff": "→ orchestrator: research_done", "matched_transition_index": 0,
        "rationale": "researcher", "iter_counts_snapshot": {"worker": 1, "probe": 1, "researcher": 1},
    })
    # 5 optimizer iter records (each await_optimizer→await_optimizer to bump iter_counter)
    for i in range(5):
        state_log.append({
            "ts": f"2026-05-05T03:{i:02d}:00Z",
            "from_state": "await_optimizer", "to_state": "await_optimizer",
            "handoff": f"@aog-kernel-optimizer iter {i}",
            "matched_transition_index": 0,
            "rationale": "ko iter",
            "iter_counts_snapshot": {
                "worker": 1, "probe": 1, "researcher": 1, "optimizer": i + 1,
            },
        })

    ws = make_workspace(
        "test_op_p0aa",
        progress_md="# PROGRESS\n\n→ orchestrator: ko iter\n",
        state_log=state_log,
        verification={
            "precision": {"status": "PASS",
                          "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
                          "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11}},
            "performance": {
                "ratio": 0.385, "status": "BELOW_THRESHOLD",
                "independent_re_measure": {"ran": True, "ratio": 0.385, "delta_vs_kw_self_report": 0.0},
            },
            "determinism": {"policy_satisfied": True,
                            "n_identical_cases": 50, "n_cases_checked": 50},
        },
    )
    # Pipeline-exhausted markers
    (ws / "cann_strategy_inference.md").write_text("# researcher findings")
    (ws / "probe_result.json").write_text(json.dumps({
        "classification": "requirement", "confidence": "verified",
    }))

    monkeypatch.chdir(ws.parent)
    exit_code = orchestrator.run_single_op(
        op="test_op_p0aa", workspace=ws, lane=0
    )

    # Pre-fix: error 5 (state machine no transition match) or exit 2 (iter_cap).
    # Post-fix: P0aa detects legitimate exhaustion → finalize PARTIAL_PERSIST
    # → exit 0.
    assert exit_code == 0, (
        f"P0aa: legitimate ko exhaustion should finalize cleanly; got {exit_code}"
    )

    # verification.json should be tagged with persist_verdict
    v = json.loads((ws / "verification.json").read_text())
    assert v.get("precision", {}).get("persist_verdict") == "PARTIAL_PERSIST"


# ---------------------------------------------------------------------------
# P0ab regression — P0t replay drops single abort entry, log becomes empty,
# bootstrap canonicalizes from PROGRESS.md handoff.
# ---------------------------------------------------------------------------
def test_p0ab_p0t_replay_empty_log_via_bootstrap(make_workspace):
    state_log = [{
        "ts": "2026-05-05T22:00:00Z",
        "from_state": "await_worker", "to_state": "abort",
        "handoff": "→ orchestrator: probe done, kernel verified",
        "matched_transition_index": -1,
        "rationale": "worker exited without recognized handoff — contract violation",
        "iter_counts_snapshot": {"worker": 1},
    }]
    ws = make_workspace(
        "test_op_p0ab",
        progress_md=(
            "# PROGRESS\n\n## Phase D\n\n"
            "→ orchestrator: done — probe verified 31/31 semantic PASS\n"
        ),
        state_log=state_log,
    )
    write_verification(ws)

    result = resume._replay_buggy_abort_recovery(ws)
    assert result is True, "P0t replay must succeed via bootstrap canonicalization"

    log_lines = (ws / "state_transitions.jsonl").read_text().strip().splitlines()
    # Bootstrap canonicalized → log has init→await_worker AND await_worker→target
    assert len(log_lines) >= 2

    first = json.loads(log_lines[0])
    assert first["from_state"] == "init"
    assert first["to_state"] == "await_worker"
    assert "P0bb" in first["rationale"]


# ---------------------------------------------------------------------------
# Resume + orchestrator combined — agent_died at await_worker, P0r recovery
# clears marker, orchestrator picks up via empty-log bootstrap (no agent
# re-spawn since worker actually completed).
# ---------------------------------------------------------------------------
def test_resume_plus_orchestrator_completed_op_no_respawn(
    fake_agent_dispatch, stub_skills, make_workspace, monkeypatch
):
    # Workspace where worker completed (PROGRESS has done handoff,
    # verification.json present) but state log is empty (e.g. crash before
    # log write).
    ws = make_workspace(
        "test_op_completed",
        progress_md=(
            "# PROGRESS\n\n## Phase E\n\n"
            "→ orchestrator: done — Pass A 50/50; perf 2.0x\n"
        ),
    )
    write_verification(ws)

    monkeypatch.chdir(ws.parent)

    # First action: resume.diagnose should determine state via bootstrap
    resume.diagnose("test_op_completed", workspace=ws)
    # diagnose triggered current_state → bootstrap canonicalized chain
    log_path = ws / "state_transitions.jsonl"
    assert log_path.exists() and log_path.stat().st_size > 0

    # If bootstrap routed to finalize, status should be NONE_TERMINAL.
    # If routed to a non-terminal (e.g. await_probe per V3.3 rule), status
    # should be RESUMABLE — but importantly, no agent should have been
    # spawned yet.
    assert fake_agent_dispatch.steps_remaining() == 0, (
        "diagnose should not spawn agents"
    )
