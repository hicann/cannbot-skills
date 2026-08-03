# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression test for the 2026-05-13 rms_norm_quant gap:

Worker writes canonical "→ orchestrator: done" line in PROGRESS.md but the
`claude --print` result.output_text doesn't contain it (final response was
a tool call, not a text emission with the marker). Orchestrator's
extract_canonical_handoff returned the full stdout (no canonical prefix),
state machine routed to abort despite substantive PASS.

Fix: when stdout has no canonical handoff, fall back to scanning PROGRESS.md
tail for one before declaring contract violation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import orchestrator  # noqa: E402


def test_extract_canonical_handoff_returns_full_text_when_no_marker():
    """Return pure-text stdout unchanged when no handoff marker exists.

    The orchestrator then has to look elsewhere for a canonical handoff.
    """
    text = "did stuff\nMore stuff\nNo marker here\n"
    result = orchestrator.extract_canonical_handoff(text)
    # extractor's last-resort behavior: return the stripped full text
    assert result == text.strip()
    # Critically: it does NOT use the public orchestrator handoff form.
    assert not result.startswith("→ orchestrator:")


def test_extract_canonical_handoff_finds_marker_in_progress_tail():
    """Match a canonical marker through the PROGRESS.md fallback.

    This covers the marker position from the original rms_norm_quant case.
    """
    progress_tail = (
        "# rms_norm_quant\n\n"
        "## kw-1 done\n\n"
        "Some narrative...\n"
        "\n"
        "→ orchestrator: done — precision 8/8 PASS, perf 6.60×, det 8/8 satisfied.\n"
    )
    result = orchestrator.extract_canonical_handoff(progress_tail)
    assert result.startswith("→ orchestrator: done"), (
        f"expected canonical handoff prefix; got {result!r}"
    )
    assert "precision 8/8 PASS" in result


def test_run_single_op_falls_back_to_progress_md(tmp_path, monkeypatch):
    """Route using the PROGRESS.md handoff when worker stdout has no marker.

    The orchestrator must use the canonical handoff instead of aborting.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    # Pre-seed O2.5 artifacts + PROGRESS.md scaffold
    (ws / "input_gen.py").write_text("# stub")
    (ws / "edge_inputs.pt").write_bytes(b"\x80\x02tensor")
    (ws / "manifest.json").write_text(json.dumps({"op": "test_op", "data_sha256": "x"}))
    (ws / "ref_runnable.json").write_text(json.dumps({
        "verdict": "RUNNABLE", "ref_call_path": "Model.forward",
        "recommendation": "PROCEED",
    }))
    (ws / "edge_dataset.pt").write_bytes(b"\x80\x02ds")
    (ws.parent / ".ascendc_env").write_text(
        "A5_HOST=t\nA5_USER=root\nA5_PASSWORD=t\nA5_CONTAINER=t\n"
        "CANN_PATH=/t\nSOC_VERSION=Ascend950PR_9579\n"
    )

    (ws / "PROGRESS.md").write_text("# kw-1 work log\n\n")

    import agent_dispatch as ad
    import agent_transport

    def fake_spawn(*args, **kwargs):
        ws_arg = kwargs.get("workspace") or args[1]
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
        # PB-33 (2026-05-14): op_host/ completeness gate requires ≥ 3 files
        op_host = ws_arg / "op_host"
        op_host.mkdir(exist_ok=True)
        (op_host / "stub_def.cpp").write_text("// stub op_def\n")
        (op_host / "stub_tiling.cpp").write_text("// stub tiling\n")
        (op_host / "stub_tiling.h").write_text("// stub tiling header\n")
        with (ws_arg / "PROGRESS.md").open("a") as f:
            f.write(
                "Built kernel + verified.\n\n"
                "→ orchestrator: done — Pass A 50/50; perf 1.5x\n"
            )
        # CRITICAL: stdout does NOT contain the canonical handoff
        return agent_transport.AgentResult(
            agent_type="aog-kernel-worker", success=True, is_error=False,
            output_text="Bash: ls\nBash: ls again\nAll artifacts in place.",
            duration_ms=100, cost_usd=0.1, session_id="t",
            terminal_reason="end_turn", raw_envelope={"type": "result"},
            tool_uses=[], progress_lines=[],
        )
    monkeypatch.setattr(ad, "spawn_for_state", fake_spawn)

    # Standard stubs (mirror conftest.py)
    import critic_invoke
    import kb_invoke
    import phase_o5
    import phase_o5_runner
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning",
                        lambda ws, s: False)
    monkeypatch.setattr(kb_invoke, "merge_one",
                        lambda ws: {"success": True, "merged": "test"})

    def matching_runner(workspace, op, *args, **kwargs):
        return phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 50, "total": 50},
            pass_b={"tier1_pass": 11, "total": 11},
        )
    monkeypatch.setattr(phase_o5_runner, "ssh_runner", matching_runner)
    import importlib as _il
    monkeypatch.setattr(_il, "reload", lambda mod: mod)

    monkeypatch.chdir(tmp_path)
    exit_code = orchestrator.run_single_op(
        op="test_op", workspace=ws, lane=0,
    )

    # Before the fix: exit_code would be non-zero (route to abort).
    # After the fix: PROGRESS.md tail rescues the handoff, route to finalize.
    assert exit_code == 0, (
        f"orchestrator should rescue canonical handoff from PROGRESS.md "
        f"when stdout lacks it; got exit_code={exit_code}"
    )


def test_run_single_op_does_not_reuse_stale_progress_md_handoff(tmp_path, monkeypatch):
    """Do not reuse a stale PROGRESS.md handoff from a prior agent.

    This applies when stdout has no marker and the current spawn did not update
    PROGRESS.md.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    (ws / "input_gen.py").write_text("# stub")
    (ws / "edge_inputs.pt").write_bytes(b"\x80\x02tensor")
    (ws / "manifest.json").write_text(json.dumps({"op": "test_op", "data_sha256": "x"}))
    (ws / "ref_runnable.json").write_text(json.dumps({
        "verdict": "RUNNABLE", "ref_call_path": "Model.forward",
        "recommendation": "PROCEED",
    }))
    (ws / "edge_dataset.pt").write_bytes(b"\x80\x02ds")
    (ws.parent / ".ascendc_env").write_text(
        "A5_HOST=t\nA5_USER=root\nA5_PASSWORD=t\nA5_CONTAINER=t\n"
        "CANN_PATH=/t\nSOC_VERSION=Ascend950PR_9579\n"
    )
    (ws / "PROGRESS.md").write_text(
        "# old worker progress\n\n"
        "→ orchestrator: done — stale worker handoff, must not route this spawn\n"
    )

    import agent_dispatch as ad
    import agent_transport

    def fake_spawn(*args, **kwargs):
        ws_arg = kwargs.get("workspace") or args[1]
        (ws_arg / "verification.json").write_text(json.dumps({
            "precision": {"status": "PASS",
                          "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
                          "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11}},
            "performance": {
                "status": "PASS",
                "ratio": 1.5,
                "method": "fixture",
                "independent_re_measure": {
                    "status": "N/A",
                    "reason": "fixture",
                },
            },
            "determinism": {"policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50},
        }))
        (ws_arg / "knowledge_update.md").write_text(
            "## Context\nStub.\n\n## Findings\n- Stub.\n\n"
            "## KB-promotable patterns (proposed)\nNone\n\n"
            "## Cited KB items\n- STUB-0\n\n## Anti-patterns avoided\nNone\n"
        )
        (ws_arg / "model_new_ascendc.py").write_text(
            "import torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self,x): return x\n"
        )
        (ws_arg / "model.py").write_text(
            "import torch.nn as nn\nclass Model(nn.Module):\n    def forward(self,x): return x\n"
        )
        op_host = ws_arg / "op_host"
        op_host.mkdir(exist_ok=True)
        (op_host / "stub_def.cpp").write_text("// stub op_def\n")
        (op_host / "stub_tiling.cpp").write_text("// stub tiling\n")
        (op_host / "stub_tiling.h").write_text("// stub tiling header\n")
        return agent_transport.AgentResult(
            agent_type="aog-kernel-worker", success=True, is_error=False,
            output_text="All artifacts in place, but no canonical marker.",
            duration_ms=100, cost_usd=0.1, session_id="t",
            terminal_reason="end_turn", raw_envelope={"type": "result"},
            tool_uses=[], progress_lines=[],
        )

    monkeypatch.setattr(ad, "spawn_for_state", fake_spawn)

    import critic_invoke
    import kb_invoke
    import phase_o5
    import phase_o5_runner
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning",
                        lambda ws, s: False)
    monkeypatch.setattr(kb_invoke, "merge_one",
                        lambda ws: {"success": True, "merged": "test"})
    monkeypatch.setattr(
        phase_o5_runner, "ssh_runner",
        lambda workspace, op, *args, **kwargs: phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 50, "total": 50},
            pass_b={"tier1_pass": 11, "total": 11},
        ),
    )
    import importlib as _il
    monkeypatch.setattr(_il, "reload", lambda mod: mod)

    monkeypatch.chdir(tmp_path)
    exit_code = orchestrator.run_single_op(op="test_op", workspace=ws, lane=0)

    assert exit_code != 0
