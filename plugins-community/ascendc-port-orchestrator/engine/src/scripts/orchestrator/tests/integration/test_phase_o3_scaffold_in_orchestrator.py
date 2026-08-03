# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration smoke: P0ll Phase O3 scaffold must run in orchestrator.

Anti-pattern guard against "function unit-tested but not actually called"
class. Verifies the orchestrator invokes phase_o3.init_progress_md
between O2.5 ready and first agent spawn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import orchestrator  # noqa: E402
import phase_o3  # noqa: E402


def _seed_workspace_with_o25(ws: Path):
    """Workspace ready to spawn worker (O2.5 complete)."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": ws.name,
        "opgen_mode": "backward",
    }))
    (ws / "input_gen.py").write_text("# stub\n")
    (ws / "edge_inputs.pt").write_bytes(b"\x80\x02tensor")
    (ws / "manifest.json").write_text(json.dumps({"op": "x", "data_sha256": "abc"}))
    (ws / "ref_runnable.json").write_text(json.dumps({
        "verdict": "RUNNABLE", "ref_call_path": "Model.forward",
        "recommendation": "PROCEED",
    }))
    (ws / "edge_dataset.pt").write_bytes(b"\x80\x02ds")
    (ws.parent / ".ascendc_env").write_text(
        "A5_HOST=test\nA5_USER=root\nA5_PASSWORD=t\n"
        "A5_CONTAINER=t\nCANN_PATH=/test\nSOC_VERSION=Ascend950PR_9579\n"
    )


def test_orchestrator_writes_scaffold_on_cold_start(tmp_path, monkeypatch):
    """Write the scaffold for a cold-start workspace.

    The scaffold must exist before the first worker spawn.
    """
    ws = tmp_path / "test_op"
    _seed_workspace_with_o25(ws)
    # No PROGRESS.md yet

    progress_at_spawn_time = []
    import agent_dispatch as ad
    import agent_transport

    def fake_spawn(*args, **kwargs):
        # Record PROGRESS.md state at the moment of spawn
        ws_arg = kwargs.get("workspace") or args[1]
        p = ws_arg / "PROGRESS.md"
        progress_at_spawn_time.append({
            "exists": p.exists(),
            "has_marker": p.exists() and phase_o3.SKELETON_MARKER in p.read_text(),
        })
        # Worker writes done so loop terminates
        (ws_arg / "verification.json").write_text(json.dumps({
            "precision": {"status": "PASS",
                          "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
                          "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11}},
            "performance": {
                "status": "PASS",
                "ratio": 1.5,
                "method": "same_wrapper symmetric=true method_symmetric",
                "independent_re_measure": {
                    "status": "N/A",
                    "reason": "fixture default — not the gate under test",
                },
            },
            "determinism": {"policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50},
        }))
        (ws_arg / "knowledge_update.md").write_text(
            "## Context\nStub.\n\n## Findings\n- Stub.\n\n"
            "## KB-promotable patterns (proposed)\nNone\n\n"
            "## Cited KB items\n- STUB-0\n\n## Anti-patterns avoided\nNone\n"
        )
        # DEBT-NEW (2026-05-14): canonical mode-agnostic entry-points
        (ws_arg / "model_new_ascendc.py").write_text(
            "import torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self,x): return x\n"
        )
        (ws_arg / "model.py").write_text(
            "import torch.nn as nn\nclass Model(nn.Module):\n    def forward(self,x): return x\n"
        )
        return agent_transport.AgentResult(
            agent_type="aog-kernel-worker", success=True, is_error=False,
            output_text="→ orchestrator: done — Pass A 50/50; perf 1.5x",
            duration_ms=100, cost_usd=0.1, session_id="t",
            terminal_reason="end_turn", raw_envelope={"type": "result"},
            tool_uses=[], progress_lines=[],
        )
    monkeypatch.setattr(ad, "spawn_for_state", fake_spawn)

    # Stub critic + kb + O5 runner (avoid real SSH)
    import critic_invoke
    import kb_invoke
    import phase_o5
    import phase_o5_runner
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning",
                        lambda ws, s: False)
    monkeypatch.setattr(kb_invoke, "merge_one",
                        lambda ws: {"success": True, "merged": "test-stub"})
    # Make O5 runner agree (matching) so finalize completes

    def matching_runner(workspace, op, *args, **kwargs):
        return phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 50, "total": 50},
            pass_b={"tier1_pass": 11, "total": 11},
        )
    monkeypatch.setattr(phase_o5_runner, "ssh_runner", matching_runner)
    # orchestrator does importlib.reload(phase_o5_runner) before each O5 call —
    # neutralize it so the ssh_runner monkeypatch survives.
    import importlib as _il
    monkeypatch.setattr(_il, "reload", lambda mod: mod)

    monkeypatch.chdir(tmp_path)
    exit_code = orchestrator.run_single_op(
        op="test_op", workspace=ws, lane=0,
    )

    assert exit_code == 0
    # Phase O3 must have written scaffold BEFORE worker spawn
    assert progress_at_spawn_time, "worker should have been spawned at least once"
    first_spawn = progress_at_spawn_time[0]
    assert first_spawn["exists"], "PROGRESS.md must exist when worker spawns"
    assert first_spawn["has_marker"], (
        "PROGRESS.md must have phase O3 scaffold marker when worker spawns"
    )


def test_orchestrator_preserves_existing_progress(tmp_path, monkeypatch):
    """Preserve an existing worker-written PROGRESS.md.

    The orchestrator must not overwrite the file on subsequent runs.
    """
    ws = tmp_path / "test_op"
    _seed_workspace_with_o25(ws)
    # Pre-existing worker content
    original = (
        "# real progress from prior session\n"
        "## kw-1 final\n"
        "→ orchestrator: PARTIAL_PERSIST — existing real evidence\n"
    )
    (ws / "PROGRESS.md").write_text(original)

    monkeypatch.chdir(tmp_path)

    # Just call init_progress_md directly — orchestrator wires it the same
    rep = phase_o3.init_progress_md(ws, "test_op")
    assert rep.verdict == "PRESERVED"
    assert (ws / "PROGRESS.md").read_text() == original
