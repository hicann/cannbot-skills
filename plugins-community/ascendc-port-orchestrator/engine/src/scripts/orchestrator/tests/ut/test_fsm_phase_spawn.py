# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Fast per-handler unit tests for fsm_phase_spawn.handle_spawn.

DEBT-201: the spawn + post-spawn cluster was extracted from run_single_op into
fsm_phase_spawn. These stub the sibling seams + a fake OrchestratorContext to
lock the handler's control-flow mapping (each original `continue` / `return N`
→ HandlerResult) AND the mutable-loop-state threading (spawn_count advance,
last_handoff set) WITHOUT the full run_single_op boot. The e2e routing paths
stay locked by test_run_single_op_fsm_characterization; these are the cheap
per-module complement.

Run: cd src/scripts/orchestrator && PYTHONPATH=.:plugins python3 -m pytest \
     tests/ut/test_fsm_phase_spawn.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

# Force the orchestrator.py MODULE identity into sys.modules["orchestrator"] so
# the ctx read-through accessors (ctx._resolve_env etc.) resolve the live module.
import orchestrator  # noqa: E402,F401
import agent_dispatch  # noqa: E402
import agent_transport  # noqa: E402
import critic_invoke  # noqa: E402
import events  # noqa: E402
import kb_invoke  # noqa: E402
import schema_norm  # noqa: E402
import state_executor  # noqa: E402
import fsm_phase_spawn as S  # noqa: E402
from fsm_context import OrchestratorContext  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------
class _Snap:
    def __init__(self, state="await_worker", iter_counts=None):
        self.current_state = state
        self.iter_counts = iter_counts or {}


def _worker_result(handoff="@aog-precision-probe iter1"):
    return agent_transport.AgentResult(
        agent_type="aog-kernel-worker", success=True, is_error=False,
        output_text="Built kernel.\n" + handoff + "\n",
        duration_ms=10, cost_usd=0.0, session_id="x",
        terminal_reason="end_turn", raw_envelope={"type": "result"},
        tool_uses=[], progress_lines=[],
    )


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text("{}")
    return OrchestratorContext(op="op", workspace=ws, lane=0)


@pytest.fixture
def happy_seams(monkeypatch):
    """Neutralize every sibling seam for a clean happy-path spawn."""
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(agent_dispatch, "spawn_for_state",
                        lambda *a, **k: _worker_result())
    monkeypatch.setattr(
        schema_norm, "normalize_workspace",
        lambda ws, fail_strict=False: schema_norm.NormalizationReport(
            events=[], rejected_terminal_aliases=[], files_modified=[]),
    )

    class _Dec:
        from_state = "await_worker"
        next_state = "await_probe"
        rationale = "r"
    monkeypatch.setattr(state_executor, "next_state", lambda *a, **k: _Dec())
    monkeypatch.setattr(kb_invoke, "merge_one", lambda ws: {"skipped": "nothing"})


# ---------------------------------------------------------------------------
# Control-flow branches
# ---------------------------------------------------------------------------
def test_no_agent_returns_exit_2(ctx, monkeypatch):
    monkeypatch.setattr(state_executor, "next_agent", lambda st: None)
    res = S.handle_spawn(ctx, _Snap())
    assert (res.action, res.exit_code) == ("return", 2)


def test_itercap_not_legitimate_returns_exit_2(ctx, monkeypatch):
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-precision-probe")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: True)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 3)
    monkeypatch.setattr(orchestrator, "_is_legitimate_pipeline_exhaustion",
                        lambda ws, st: False)
    res = S.handle_spawn(ctx, _Snap("await_probe"))
    assert (res.action, res.exit_code) == ("return", 2)


def test_itercap_legitimate_exhaustion_continues_and_sets_handoff(ctx, monkeypatch):
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-researcher")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: True)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 3)
    monkeypatch.setattr(orchestrator, "_is_legitimate_pipeline_exhaustion",
                        lambda ws, st: True)
    recorded = {}
    monkeypatch.setattr(orchestrator, "_record_partial_persist_finalize",
                        lambda ws, st, c, cap: recorded.setdefault("hit", (st, c, cap)))
    res = S.handle_spawn(ctx, _Snap("await_researcher", {"researcher": 3}))
    assert res.action == "continue"
    assert "P0y" in ctx.last_handoff
    assert recorded["hit"][0] == "await_researcher"


def test_plan_only_returns_exit_0(ctx, monkeypatch):
    ctx.plan_only = True
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    # stale-output archive is a ctx read-through — stub via orchestrator module
    monkeypatch.setattr(orchestrator, "_archive_stale_outputs_before_spawn",
                        lambda ws, st, idx: None)
    res = S.handle_spawn(ctx, _Snap())
    assert (res.action, res.exit_code) == ("return", 0)


def test_spawn_exception_returns_exit_3(ctx, monkeypatch):
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})

    def boom(*a, **k):
        raise RuntimeError("crash")
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", boom)
    marked = {}
    monkeypatch.setattr(orchestrator, "_mark_agent_died",
                        lambda ws, st, msg: marked.setdefault("m", msg))
    res = S.handle_spawn(ctx, _Snap())
    assert (res.action, res.exit_code) == ("return", 3)
    assert "crash" in marked["m"]


def test_silence_timeout_budget_exhausted_returns_exit_3(ctx, monkeypatch):
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})

    def silent(*a, **k):
        raise agent_transport.StreamSilenceTimeout(
            agent_type="aog-kernel-worker", silent_seconds=1, last_event_type="t")
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", silent)
    # already at budget → give up
    monkeypatch.setattr(orchestrator, "_load_silence_retry_count",
                        lambda ws, st: agent_transport.STREAM_SILENCE_RETRY_MAX)
    monkeypatch.setattr(orchestrator, "_mark_agent_died", lambda ws, st, msg: None)
    res = S.handle_spawn(ctx, _Snap())
    assert (res.action, res.exit_code) == ("return", 3)


def test_silence_timeout_under_budget_continues_and_bumps(ctx, monkeypatch):
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})

    def silent(*a, **k):
        raise agent_transport.StreamSilenceTimeout(
            agent_type="aog-kernel-worker", silent_seconds=1, last_event_type="t")
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", silent)
    monkeypatch.setattr(orchestrator, "_load_silence_retry_count", lambda ws, st: 0)
    bumped = {}
    monkeypatch.setattr(orchestrator, "_bump_silence_retry_count",
                        lambda ws, st: bumped.setdefault("b", True))
    res = S.handle_spawn(ctx, _Snap())
    assert res.action == "continue"
    assert bumped["b"] is True


def test_kw_1_only_returns_exit_0(ctx, monkeypatch, happy_seams):
    ctx.kw_1_only = True
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert (res.action, res.exit_code) == ("return", 0)


def test_schema_reject_returns_exit_4(ctx, monkeypatch, happy_seams):
    def reject(ws, fail_strict=False):
        if fail_strict:
            raise schema_norm.SchemaNormalizationError("drift")
        return schema_norm.NormalizationReport(
            events=[], rejected_terminal_aliases=[], files_modified=[])
    monkeypatch.setattr(schema_norm, "normalize_workspace", reject)
    res = S.handle_spawn(ctx, _Snap())
    assert (res.action, res.exit_code) == ("return", 4)


def test_state_machine_error_returns_exit_5(ctx, monkeypatch, happy_seams):
    def boom(*a, **k):
        raise state_executor.StateMachineError("bad route")
    monkeypatch.setattr(state_executor, "next_state", boom)
    res = S.handle_spawn(ctx, _Snap())
    assert (res.action, res.exit_code) == ("return", 5)


def test_happy_path_continues_and_advances_spawn_count(ctx, monkeypatch, happy_seams):
    assert ctx.spawn_count == 0
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert res.action == "continue"
    assert ctx.spawn_count == 1
    # handoff was extracted from the canned worker result
    assert ctx.last_handoff.startswith("@aog-precision-probe")


def test_post_spawn_never_merges_staged_knowledge(ctx, monkeypatch, happy_seams):
    """Worker-authored knowledge stays staged until finalize safety gates pass."""
    (ctx.workspace / "knowledge_update.md").write_text("unsafe candidate\n" * 20)

    def premature_merge(_workspace):
        raise AssertionError("KB merge ran before O5/static/critic finalize gates")

    monkeypatch.setattr(kb_invoke, "merge_one", premature_merge)
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert res.action == "continue"


def test_patch_bite_resolve_env_through_ctx(ctx, monkeypatch, happy_seams):
    """Breaking the ctx read-through dep (_resolve_env via orchestrator module)
    must reach the handler — proving the monkeypatch surface still bites.
    """
    called = {}

    def sentinel_env():
        called["hit"] = True
        raise RuntimeError("resolve_env sentinel")
    monkeypatch.setattr(orchestrator, "_resolve_env", sentinel_env)
    monkeypatch.setattr(orchestrator, "_mark_agent_died", lambda ws, st, msg: None)
    res = S.handle_spawn(ctx, _Snap())
    # the sentinel exception is caught by the generic spawn-exception guard → exit 3
    assert called.get("hit") is True
    assert (res.action, res.exit_code) == ("return", 3)
