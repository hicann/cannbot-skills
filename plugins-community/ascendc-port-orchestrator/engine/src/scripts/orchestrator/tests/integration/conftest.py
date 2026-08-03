# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration smoke fixtures — real orchestrator main loop, mocked agent dispatch.

Goal: catch bug classes that pure-Python unit tests miss because they live at
seams between modules (orchestrator + state_machine + state_executor + resume).
P0[s..cc] all surfaced from real op runs, not pytest. This harness lets us
script those scenarios in seconds.

Strategy:
  1. tmp_path workspace with PROGRESS.md + verification.json + .ascendc_env
  2. Monkeypatch agent_dispatch.spawn_for_state to return scripted AgentResult
     objects (keyed by spawn_index)
  3. Monkeypatch critic_invoke / kb_invoke to no-op (skill calls require live CC)
  4. Monkeypatch agent_dispatch.spawn_for_state to also write the agent's
     "output" to PROGRESS.md tail and verification.json — simulating what
     real agents do. The orchestrator then reads these as if a real agent ran.
  5. Run orchestrator.run_single_op or resume.execute on the workspace
  6. Assert final state, log entries, exit code
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import agent_dispatch  # noqa: E402
import agent_transport  # noqa: E402
import critic_invoke  # noqa: E402
import kb_invoke  # noqa: E402
import phase_o17_classify  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_scoped_reference_gate(monkeypatch):
    """Keep lifecycle integration tests below the customer-reference boundary.

    Live arch22 capture and backward autograd provisioning have their own tests.
    These integration cases exercise the state machine after a scoped reference
    has been accepted, so no hardware or forward-spec generation belongs here.
    """
    import fsm_phase_o25_dispatch

    monkeypatch.setattr(
        fsm_phase_o25_dispatch,
        "provision_reference",
        lambda *args, **kwargs: None,
    )

    # Phase O0 validates an installed Claude plugin and its live hook
    # registration.  That boundary has dedicated install/preflight tests; a
    # source-tree lifecycle test must not depend on the developer's config.
    import phase_o0

    monkeypatch.setattr(
        phase_o0,
        "check_hook_integrity",
        lambda workspace=None: phase_o0.O0Report(
            verdict="READY",
            summary="integration-test stub: scoped hook preflight ready",
        ),
    )

    # The lifecycle cases historically inject `ssh_runner` as their independent
    # measurement seam.  Backward mode selects `backward_verify_runner`; bridge
    # that scoped selector to the same injected seam without weakening the O5
    # comparison logic exercised by the tests.
    import phase_o5_runner

    def _scoped_test_runner(workspace, op, lane=0):
        return phase_o5_runner.ssh_runner(workspace, op, lane)

    monkeypatch.setattr(
        phase_o5_runner,
        "backward_verify_runner",
        _scoped_test_runner,
    )


@pytest.fixture(autouse=True)
def _isolate_ascendc_env(tmp_path, monkeypatch):
    """Integration tests must not read the real workspace/.ascendc_env.

    load_env() resolves DEFAULT_ASCENDC_ENV (absolute _PROJECT_ROOT/workspace/
    .ascendc_env). A checkout whose real .ascendc_env carries OPGEN_MODE could
    leak into integration tests and select a route unrelated to the scenario
    under test. Redirect load_env to a neutral per-test stub. Fixtures/tests
    that write their own tmp_path/.ascendc_env (make_workspace, the O2.5
    autogen tests) overwrite this stub at the same path, so their content still
    wins.
    """
    import briefs._common as _bc
    stub = tmp_path / ".ascendc_env"
    if not stub.exists():
        stub.write_text(
            "TARGET=a5\nA5_HOST=test-host\nA5_USER=root\nA5_PASSWORD=test\n"
            "A5_CONTAINER=test-container\nCANN_PATH=/test/cann\n"
            "SOC_VERSION=Ascend950PR_9579\n"
        )
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", stub)


@pytest.fixture(autouse=True)
def _stub_llm_subprocess_calls(monkeypatch):
    """Stub all subprocess-based LLM calls in integration tests.

    Integration tests exercise state-machine routing — they don't need
    `claude --print` subprocess calls to skills (aog-op-classify @120s,
    aog-self-critic @600s). Without these stubs, each test takes >120s
    minimum and hangs to 600s if any path reaches the post-worker audit.
    """
    # 1) Phase O1.7 classification
    def _stub_classify(workspace, *, force=False, timeout=120):
        return phase_o17_classify.OpClassification(
            op=workspace.name,
            op_class_tags=[],
            kb_recommendations=[],
            rationale="integration-test stub",
            source_signatures_observed=[],
            source_sha256="stub",
        )
    monkeypatch.setattr(phase_o17_classify, "classify", _stub_classify)

    # 2) Post-worker self-critic audit — write stub artifact so the
    #    finalize gate sees PASS without spawning claude --print.
    #    Also seed op_host/ to satisfy PB-33 OP_HOST_COMPLETENESS gate.
    import orchestrator as _orch

    def _stub_audit(workspace, *, lane=0):
        audit_doc = workspace / "audit_self_critic_post_worker.md"
        if not audit_doc.exists():
            audit_doc.write_text(
                "# Post-Worker Self-Critic Audit (integration-test stub)\n\n"
                "Verdict: PASS\n\n"
                "C13: no hallucinated metrics\n"
                "C18: no delegation cheating\n"
                "C25/C26: no anti-overfit concerns\n"
            )
        # Delegation scan marker (codex 2026-05-07 finalize gate)
        marker = workspace / ".delegation_scan_passed"
        if not marker.exists():
            marker.write_text("integration-test stub: scan passed\n")
        # PB-33 (2026-05-14): op_host/ completeness gate stub
        op_host = workspace / "op_host"
        op_host.mkdir(exist_ok=True)
        if not (op_host / "stub_def.cpp").exists():
            (op_host / "stub_def.cpp").write_text("// integration-test stub op_def\n")
            (op_host / "stub_tiling.cpp").write_text("// integration-test stub tiling\n")
            (op_host / "stub_tiling.h").write_text("// integration-test stub tiling header\n")
        state = json.loads((workspace / ".opgen_state.json").read_text())
        op = state.get("op", workspace.name)
        kernel = workspace / "kernel"
        kernel.mkdir(exist_ok=True)
        (kernel / f"lib{op}.so").write_bytes(b"compiled-extension")
        (workspace / f"verify_{op}.py").write_text(
            "from model_new_ascendc import ModelNew\n"
            "candidate = ModelNew()\n"
            "output = candidate(inputs)\n"
        )
    monkeypatch.setattr(_orch, "_ensure_audit_artifacts", _stub_audit)


def _mk_agent_result(
    *,
    agent_type: str = "aog-kernel-worker",
    output_text: str = "",
    success: bool = True,
    is_error: bool = False,
    cost_usd: float = 0.5,
    duration_ms: int = 60000,
) -> agent_transport.AgentResult:
    return agent_transport.AgentResult(
        agent_type=agent_type,
        success=success,
        is_error=is_error,
        output_text=output_text,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        session_id="test-session",
        terminal_reason="end_turn",
        raw_envelope={"type": "result", "subtype": "success"},
        tool_uses=[],
        progress_lines=[],
    )


@pytest.fixture
def fake_agent_dispatch(monkeypatch):
    """Provide a scripted spawn_for_state implementation.

    Tests register scenarios via the `script` callback, which receives
    (op, workspace, state, spawn_index) and returns either:
      - an AgentResult (handoff already in result.output_text), OR
      - a tuple (AgentResult, side_effect_fn) where side_effect_fn(workspace)
        runs before result is returned (simulates worker writing files).
    """
    scripts: list[Callable] = []

    def add_step(fn: Callable):
        scripts.append(fn)
        return fn

    def fake_spawn(op, workspace, state, *, lane, spawn_index, **kwargs):
        if not scripts:
            raise AssertionError(
                f"fake_agent_dispatch: spawn called for state={state} "
                f"spawn_index={spawn_index} but no scripted step registered"
            )
        step = scripts.pop(0)
        out = step(op=op, workspace=workspace, state=state, spawn_index=spawn_index)
        if isinstance(out, tuple):
            result, side_effect = out
            side_effect(workspace)
        else:
            result = out
        return result

    monkeypatch.setattr(agent_dispatch, "spawn_for_state", fake_spawn)

    class Harness:
        add = staticmethod(add_step)

        @staticmethod
        def steps_remaining() -> int:
            return len(scripts)

    return Harness


@pytest.fixture
def stub_skills(monkeypatch):
    """No-op the critic/kb skill invocations — they require a live CC session."""
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, trigger: {"verdict": "PASS"})
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning", lambda ws, state: False)
    monkeypatch.setattr(kb_invoke, "merge_one", lambda ws: {"success": True, "merged": "test"})
    # P0aay (2026-05-11): stub O5 runner for integration tests — SSH/scp
    # require a real NPU container. Return a no-op VERIFIED result.
    import phase_o5
    monkeypatch.setattr(
        phase_o5, "post_verify_for_finalize",
        lambda workspace, op, runner=None, skip=False, **kwargs: phase_o5.O5Report(
            verdict="VERIFIED",
            summary="test stub — O5 verified (no-op)",
        ),
    )
    # P0aay (2026-05-11): pre-write audit_self_critic_post_worker.md so
    # the orchestrator's audit step doesn't block finalize.
    import orchestrator as _orch
    _orig_run_single_op = _orch.run_single_op

    def _patched_run_single_op(op, *, workspace=None, lane=0, plan_only=False, cap_bumps=None):
        if workspace is not None and workspace.exists():
            audit_doc = workspace / "audit_self_critic_post_worker.md"
            if not audit_doc.exists():
                audit_doc.write_text(
                    "# Post-Worker Self-Critic Audit (test stub)\n\n"
                    "Verdict: PASS\n\n"
                    "C13: no hallucinated metrics\n"
                    "C18: no delegation cheating\n"
                    "C25/C26: no anti-overfit concerns\n"
                )
        return _orig_run_single_op(
            op, workspace=workspace, lane=lane, plan_only=plan_only,
            cap_bumps=cap_bumps,
        )
    monkeypatch.setattr(_orch, "run_single_op", _patched_run_single_op)


@pytest.fixture
def make_workspace(tmp_path):
    """Build a workspace dir with the minimum scaffolding for orchestrator.

    P0jj (2026-05-06): seed O2.5 artifacts by default so existing integration
    tests don't trip the new Phase O2.5 gate. Tests that specifically want
    to verify the gate (e.g. test_phase_o25_blocks_worker_spawn) can pass
    skip_o25_seed=True.
    """
    def _build(
        op: str,
        *,
        progress_md: str = "# PROGRESS\nMode: backward\n",
        verification: Optional[dict] = None,
        state_log: Optional[list[dict]] = None,
        analysis_md: str = "# analysis\n",
        ascendc_env: dict = None,
        skip_o25_seed: bool = False,
    ) -> Path:
        ws = tmp_path / op
        ws.mkdir(parents=True)
        (ws / ".opgen_state.json").write_text(json.dumps({
            "schema_version": 1,
            "op": op,
            "opgen_mode": "backward",
        }))
        (ws / "PROGRESS.md").write_text(progress_md)
        (ws / "analysis.md").write_text(analysis_md)
        if verification is not None:
            verification = dict(verification)
            verification.setdefault("harness_pristine", {
                "state": "CLEAN", "o5_verdict": "VERIFIED",
                "sampled_at": "o5_post_verify",
            })
            (ws / "verification.json").write_text(json.dumps(verification))
        if state_log is not None:
            with (ws / "state_transitions.jsonl").open("w") as f:
                for entry in state_log:
                    f.write(json.dumps(entry) + "\n")
        # .ascendc_env at workspace root parent (orchestrator looks at workspace/.ascendc_env)
        env_path = tmp_path / ".ascendc_env"
        env_lines = ascendc_env or {
            "A5_HOST": "test-host",
            "A5_USER": "root",
            "A5_PASSWORD": "test",
            "A5_CONTAINER": "test-container",
            "CANN_PATH": "/test/cann",
            "SOC_VERSION": "Ascend950PR_9579",
        }
        env_path.write_text("\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n")

        if not skip_o25_seed:
            # Seed O2.5 reference-provider artifacts so the gate doesn't block.
            # Tests that exercise the gate explicitly should pass skip_o25_seed=True.
            (ws / "input_gen.py").write_text("# input gen stub\n")
            (ws / "edge_inputs.pt").write_bytes(b"\x80\x02tensor")
            (ws / "manifest.json").write_text(json.dumps({"op": op, "data_sha256": "stub"}))
            (ws / "ref_runnable.json").write_text(json.dumps({
                "verdict": "RUNNABLE", "ref_call_path": "Model.forward (active)",
                "error": None, "alternate": {"present": False},
                "recommendation": "PROCEED",
            }))
            (ws / "edge_dataset.pt").write_bytes(b"\x80\x02dataset")
            # P0aay (2026-05-11): seed knowledge_update.md so schema_norm
            # pre-handoff gate (## Findings + structure check) doesn't block
            # integration tests that exercise worker→done→finalize flow.
            (ws / "knowledge_update.md").write_text(
                "## Context\nStub context for test.\n\n"
                "## Findings\n- Stub finding\n\n"
                "## KB-promotable patterns (proposed)\nNone\n\n"
                "## Cited KB items\n- STUB-0\n\n"
                "## Anti-patterns avoided\nNone\n"
            )
            # P0aay (2026-05-11): seed O5 verifier so post-verify runner
            # doesn't fail with "no Pass B verifier found". O5 runner
            # searches for run_pass_b.py (or pass_b_runner.py alias).
            (ws / "run_pass_b.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json; print(json.dumps({'status':'PASS','n_pass':50,'n_total':50}))\n"
            )
            # DEBT-NEW (2026-05-14, OL-160): _check_universal_entrypoints
            # gate requires canonical entry-point file names at workspace
            # root for ANY PASS verdict. Integration tests reach finalize
            # → must seed both files to satisfy the mode-agnostic gate.
            # Tests targeting the gate explicitly should skip_o25_seed=True
            # and not seed these.
            (ws / "model.py").write_text(
                "import torch.nn as nn\n"
                "class Model(nn.Module):\n    def forward(self,x): return x\n"
            )
            (ws / "model_new_ascendc.py").write_text(
                "import torch.nn as nn\n"
                "class ModelNew(nn.Module):\n    def forward(self,x): return x\n"
                "if __name__ == '__main__':\n    pass\n"
            )
            kernel = ws / "kernel"
            kernel.mkdir(exist_ok=True)
            (kernel / f"lib{op}.so").write_bytes(b"compiled-extension")
            (ws / f"verify_{op}.py").write_text(
                "from model_new_ascendc import ModelNew\n"
                "candidate = ModelNew()\n"
                "output = candidate(inputs)\n"
            )

        return ws

    return _build


def append_progress_handoff(workspace: Path, handoff: str) -> None:
    """Helper: simulate worker appending its final handoff to PROGRESS.md tail."""
    p = workspace / "PROGRESS.md"
    cur = p.read_text() if p.exists() else ""
    sep = "" if cur.endswith("\n\n") else ("\n" if cur.endswith("\n") else "\n\n")
    p.write_text(cur + sep + handoff + "\n")


def write_verification(workspace: Path, **kwargs) -> None:
    """Helper: write a minimal verification.json."""
    default = {
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
        },
        "performance": {
            "ratio": 1.5, "status": "PASS",
            "independent_re_measure": {"ran": True, "ratio": 1.5, "delta_vs_kw_self_report": 0.0},
        },
        "determinism": {
            "policy_satisfied": True,
            "n_identical_cases": 50,
            "n_cases_checked": 50,
        },
        "harness_pristine": {
            "state": "CLEAN", "o5_verdict": "VERIFIED",
            "sampled_at": "o5_post_verify",
        },
    }
    default.update(kwargs)
    (workspace / "verification.json").write_text(json.dumps(default))


__all__ = [
    "_mk_agent_result",
    "fake_agent_dispatch",
    "stub_skills",
    "make_workspace",
    "append_progress_handoff",
    "write_verification",
]
