# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration smoke: P0kk Phase O5 must block finalize when measured ≠ claimed.

This is the test that catches the worker-self-report-lies bug class. P0ee
caught "status=PASS with tier1_pass=0" via schema. P0kk catches "tier1_pass=50
claimed, runner measured 0" — schema-consistent but factually wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import orchestrator  # noqa: E402
import phase_o5  # noqa: E402


def _seed_workspace_at_finalize(ws: Path, *, prec: dict, **opts):
    """Workspace ready to enter finalize state with given verification claim."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": ws.name,
        "opgen_mode": "backward",
    }))
    (ws / "PROGRESS.md").write_text(
        "# done\n→ orchestrator: done — Pass A 50/50; perf 1.5x\n"
    )
    (ws / "verification.json").write_text(json.dumps({
        "precision": prec, "performance": {"status": "PASS", "ratio": 1.5},
        "determinism": {"policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50},
    }))
    # Phase O2.5 artifacts so O2.5 gate doesn't fire instead
    (ws / "input_gen.py").write_text("# stub\n")
    (ws / "edge_inputs.pt").write_bytes(b"\x80\x02tensor")
    (ws / "manifest.json").write_text(json.dumps({"op": "x", "data_sha256": "abc"}))
    (ws / "ref_runnable.json").write_text(json.dumps({
        "verdict": "RUNNABLE", "ref_call_path": "Model.forward",
        "recommendation": "PROCEED",
    }))
    (ws / "edge_dataset.pt").write_bytes(b"\x80\x02ds")
    (ws / "knowledge_update.md").write_text(
        "## Context\nStub.\n\n## Findings\n- Stub.\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\n- STUB-0\n\n## Anti-patterns avoided\nNone\n"
    )
    # State log: already at finalize
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"from_state": "init", "to_state": "await_worker",
                    "ts": "t1", "handoff": "", "matched_transition_index": 0,
                    "rationale": "", "iter_counts_snapshot": {}}) + "\n" +
        json.dumps({"from_state": "await_worker", "to_state": "finalize",
                    "ts": "t2", "handoff": "→ orchestrator: done",
                    "matched_transition_index": 0, "rationale": "",
                    "iter_counts_snapshot": {}}) + "\n"
    )
    (ws.parent / ".ascendc_env").write_text(
        "A5_HOST=test\nA5_USER=root\nA5_PASSWORD=t\n"
        "A5_CONTAINER=t\nCANN_PATH=/test\nSOC_VERSION=Ascend950PR_9579\n"
    )


def test_o5_mismatch_routes_back_to_worker(tmp_path, monkeypatch):
    """Route a measured precision mismatch back to the worker.

    A 0/50 runner measurement must prevent finalization of a 50/50 worker claim.
    """
    ws = tmp_path / "test_op"
    _seed_workspace_at_finalize(ws, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })

    # Inject a MISMATCH runner via monkeypatch on phase_o5_runner.ssh_runner.
    # Orchestrator now (Step 2.1) calls phase_o5.post_verify_for_finalize with
    # runner=phase_o5_runner.ssh_runner, so we replace ssh_runner directly.
    import phase_o5_runner

    def mismatch_runner(workspace, op, *args, **kwargs):
        return phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 0, "total": 50},
        )
    monkeypatch.setattr(phase_o5_runner, "ssh_runner", mismatch_runner)
    import importlib as _il
    monkeypatch.setattr(_il, "reload", lambda mod: mod)

    # Track agent_dispatch — should NOT be called for await_worker spawn since
    # we hit iter_cap or not depending on flow. Critical: finalize_pipeline
    # should NOT promote.
    spawn_calls = []
    import agent_dispatch as ad

    def fake_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        # If we're spawning kw, allow it but make it short-circuit
        import agent_transport
        return agent_transport.AgentResult(
            agent_type="aog-kernel-worker", success=True, is_error=False,
            output_text="→ orchestrator: PARTIAL_PERSIST — re-evaluation",
            duration_ms=100, cost_usd=0.1, session_id="t",
            terminal_reason="end_turn", raw_envelope={"type": "result"},
            tool_uses=[], progress_lines=[],
        )
    monkeypatch.setattr(ad, "spawn_for_state", fake_spawn)

    # Track finalize_pipeline calls — should NOT be called on MISMATCH
    import finalize_pipeline as fp
    finalize_called = [False]
    real_finalize = fp.finalize_op

    def wrapped_finalize(*args, **kwargs):
        finalize_called[0] = True
        return real_finalize(*args, **kwargs)
    monkeypatch.setattr(fp, "finalize_op", wrapped_finalize)

    # Stub critic + kb
    import critic_invoke
    import kb_invoke
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning",
                        lambda ws, s: False)
    monkeypatch.setattr(kb_invoke, "merge_one",
                        lambda ws: {"success": True, "merged": "test-stub"})

    monkeypatch.chdir(tmp_path)
    # Run — orchestrator will detect MISMATCH at finalize, route back to
    # await_worker. Worker spawn is mocked; loop continues until either
    # iter_cap or another terminal.
    orchestrator.run_single_op(
        op="test_op", workspace=ws, lane=0,
    )

    # Critical assertion: finalize_pipeline must NOT have run (MISMATCH
    # blocks promotion).
    assert not finalize_called[0], (
        "finalize_pipeline MUST NOT run when O5 MISMATCH — got "
        f"finalize_called={finalize_called[0]}"
    )

    # State log should record finalize → await_worker transition (P0kk rollback)
    log_entries = []
    for line in (ws / "state_transitions.jsonl").read_text().strip().splitlines():
        log_entries.append(json.loads(line))
    rollback = []
    for entry in log_entries:
        if (entry.get("from_state") == "finalize"
                and entry.get("to_state") == "await_worker"):
            rollback.append(entry)
    assert rollback, (
        f"O5 MISMATCH must record finalize → await_worker transition; "
        f"got log: {[(e['from_state'], e['to_state']) for e in log_entries]}"
    )
    assert "P0kk" in rollback[0].get("rationale", "")
    assert "MISMATCH" in rollback[0].get("rationale", "")


def test_o5_runner_failed_blocks_finalize(tmp_path, monkeypatch):
    """Block finalization when the O5 runner fails.

    P0aba (2026-05-07) routes RUNNER_FAILED to await_worker. Codex finding #3:
    fail-open silently allowed finalize when O5 runner errored — same
    bug shape as worker self-claim with no independent proof.
    """
    ws = tmp_path / "test_op"
    _seed_workspace_at_finalize(ws, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
        "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
    })

    # Stub ssh_runner to return RUNNER_FAILED directly (the test premise).
    import phase_o5_runner
    monkeypatch.setattr(
        phase_o5_runner, "ssh_runner",
        lambda workspace, op: phase_o5.MeasuredResult(
            runner_error="test stub: simulated runner failure"
        ),
    )
    import importlib as _il
    monkeypatch.setattr(_il, "reload", lambda mod: mod)

    import finalize_pipeline as fp
    finalize_called = [False]
    real_finalize = fp.finalize_op

    def wrapped_finalize(*args, **kwargs):
        finalize_called[0] = True
        return real_finalize(*args, **kwargs)
    monkeypatch.setattr(fp, "finalize_op", wrapped_finalize)

    # Bound worker respawns to 1 so test terminates quickly.
    import agent_dispatch as ad
    import agent_transport
    spawn_calls = [0]

    def fake_spawn(*args, **kwargs):
        spawn_calls[0] += 1
        return agent_transport.AgentResult(
            agent_type="aog-kernel-worker", success=True, is_error=False,
            output_text="→ orchestrator: PARTIAL_PERSIST — runner reconciliation needed",
            duration_ms=100, cost_usd=0.1, session_id="t",
            terminal_reason="end_turn", raw_envelope={"type": "result"},
            tool_uses=[], progress_lines=[],
        )
    monkeypatch.setattr(ad, "spawn_for_state", fake_spawn)

    import critic_invoke
    import kb_invoke
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning",
                        lambda ws, s: False)
    monkeypatch.setattr(kb_invoke, "merge_one",
                        lambda ws: {"success": True, "merged": "test-stub"})

    monkeypatch.chdir(tmp_path)
    orchestrator.run_single_op(op="test_op", workspace=ws, lane=0)

    # P0aba: RUNNER_FAILED must NOT allow finalize. Orchestrator routes back
    # to await_worker — finalize_pipeline.finalize_op is never invoked.
    assert not finalize_called[0], (
        "RUNNER_FAILED must block finalize (P0aba 2026-05-07). "
        "If finalize ran, the fail-open regression returned."
    )
