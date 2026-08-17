# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Wiring contract for the dispatch-site stop gate.

Claude Code fires the agent stop gates from its own SubagentStop hook. Harnesses without
that event get them from the orchestrator, which owns the completion event. These tests pin
the wiring — not the gates' internal rules, which `hooks/v3/*` and the door suite cover.

The regression they exist for: the first implementation ran the dispatcher with
``cwd=workspace/<op>``. ``hooks/v3/_common.sh:find_active_workspace`` resolves
``${WORKSPACE_ROOT:-workspace}`` RELATIVE to cwd and returns "" when it is absent, and
``check_worker.sh`` treats "" as "nothing to check" and exits 0 — so every stop gate passed
unconditionally, for every gated agent, with nothing anywhere reporting a problem. The
early return also precedes the ``CLAUDE_ACTIVE_WORKSPACE`` branch, so setting that env var
alone does not rescue it. Claude Code never hit this because its hooks run with the engine
as cwd.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

# No per-file sys.path bootstrap: orchestrator/tests/conftest.py already anchors the module
# roots at collection time, and its own comments name per-file inserts as the thing it
# replaces (each one permanently widens sys.path for the rest of the session).
import agent_dispatch

# These tests exist to pin MODULE-PRIVATE behaviour, so they must name private members. Bind
# them once here instead of writing `agent_dispatch._x` at each call site: the checker's
# protected-access rule (G.CLS.11) fires per expression, and one documented seam reads better
# than twenty suppressions. Binding the function object is safe for the monkeypatching below —
# it resolves `_backend` and `subprocess` from its module globals at CALL time, so patching the
# module still takes effect.
_run_stop_gate = getattr(agent_dispatch, "_run_stop_gate")
_clear_stop_gate_marker = getattr(agent_dispatch, "_clear_stop_gate_marker")
STOP_GATE_MARKER = getattr(agent_dispatch, "_STOP_GATE_MARKER")


class _Envelope:
    def __init__(self):
        self.is_error = False
        self.output_text = "worker said done"


def _fake_backend(name):
    return types.SimpleNamespace(name=name)


def test_claude_code_does_not_double_fire_the_gate(monkeypatch, tmp_path):
    """CC already runs these gates from SubagentStop; running them again would double-fire."""
    calls = []
    monkeypatch.setattr(agent_dispatch, "_backend", _fake_backend("claude_code"))
    monkeypatch.setattr(agent_dispatch.subprocess, "run",
                        lambda *a, **k: calls.append(k) or types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    env = _Envelope()
    _run_stop_gate(tmp_path, "aog-kernel-worker", env)
    assert calls == [], "Claude Code path must not invoke the gate a second time"
    assert env.is_error is False


def test_gate_runs_from_the_engine_root_not_the_op_workspace(monkeypatch, tmp_path):
    """The cwd regression: from workspace/<op> the gate finds no workspace and always passes."""
    seen = {}
    monkeypatch.setattr(agent_dispatch, "_backend", _fake_backend("opencode"))

    def _capture(cmd, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_dispatch.subprocess, "run", _capture)
    workspace = tmp_path / "workspace" / "some_op"
    workspace.mkdir(parents=True)
    _run_stop_gate(workspace, "aog-kernel-worker", _Envelope())

    engine_root = Path(agent_dispatch.__file__).resolve().parents[3]
    assert Path(seen["cwd"]) == engine_root, (
        f"gate ran from {seen['cwd']}; from anywhere below workspace/ the relative "
        "WORKSPACE_ROOT lookup returns empty and the gate passes unconditionally"
    )
    # The workspace still has to reach the gate — via env, since cwd no longer carries it.
    assert seen["env"]["CLAUDE_ACTIVE_WORKSPACE"] == str(workspace)


def test_failed_gate_fails_the_dispatch_and_leaves_a_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_dispatch, "_backend", _fake_backend("opencode"))
    monkeypatch.setattr(
        agent_dispatch.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=2, stdout="", stderr="contract violation"),
    )
    env = _Envelope()
    _run_stop_gate(tmp_path, "aog-kernel-worker", env)
    assert env.is_error is True, "a rejected stop gate must fail the dispatch"
    assert "contract violation" in env.output_text
    assert (tmp_path / ".agent_gate_stop_failed_aog-kernel-worker").is_file()


def test_passing_gate_leaves_the_result_untouched(monkeypatch, tmp_path):
    """Paired with the test above: an always-failing implementation would satisfy that one."""
    monkeypatch.setattr(agent_dispatch, "_backend", _fake_backend("opencode"))
    monkeypatch.setattr(
        agent_dispatch.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    env = _Envelope()
    _run_stop_gate(tmp_path, "aog-kernel-worker", env)
    assert env.is_error is False
    assert not list(tmp_path.glob(".agent_gate_stop_failed_*"))


def test_unrunnable_gate_is_not_a_passing_gate(monkeypatch, tmp_path):
    """A dispatcher we cannot execute must fail closed, not be skipped."""
    monkeypatch.setattr(agent_dispatch, "_backend", _fake_backend("opencode"))

    def _boom(*a, **k):
        raise OSError("exec failed")

    monkeypatch.setattr(agent_dispatch.subprocess, "run", _boom)
    env = _Envelope()
    _run_stop_gate(tmp_path, "aog-kernel-worker", env)
    assert env.is_error is True
    assert (tmp_path / ".agent_gate_stop_failed_aog-kernel-worker").is_file()


def test_fsm_refuses_to_advance_after_a_rejected_stop_gate(tmp_path, monkeypatch, caplog):
    """A rejected gate must stop the run, not just be logged.

    Claude Code blocks the sub-agent's exit via SubagentStop. Dispatch-site enforcement can
    only mark the result — and marking alone was inert: `result.is_error` had no consumer in
    _post_spawn_transition, the canonical handoff line sits at the END of output_text so a
    prepended reason does not disturb extraction, and the marker file was read by nothing.
    The run advanced on artifacts the gate had just refused.
    """
    import fsm_phase_spawn
    post_spawn_transition = getattr(fsm_phase_spawn, "_post_spawn_transition")

    ws = tmp_path / "op"
    ws.mkdir()
    (ws / f"{STOP_GATE_MARKER}_aog-kernel-worker").write_text(
        "stop gate rejected aog-kernel-worker (rc=2): contract violation\n"
    )
    caplog.set_level(logging.ERROR)
    ctx = types.SimpleNamespace(workspace=ws, lane=0, kw_1_only=False)
    res = post_spawn_transition(
        ctx, object(), "aog-kernel-worker", _Envelope(), 1
    )
    assert res.action == "return" and res.exit_code == 7, (
        f"FSM advanced past a rejected stop gate: {res}"
    )
    assert "contract violation" in caplog.text
    assert (ws / f"{STOP_GATE_MARKER}_aog-kernel-worker").is_file(), (
        "marker must survive as the durable record that these artifacts were refused"
    )


def test_fsm_proceeds_when_no_gate_marker(tmp_path):
    """Paired control: an always-stopping implementation would satisfy the test above."""
    import fsm_phase_spawn
    post_spawn_transition = getattr(fsm_phase_spawn, "_post_spawn_transition")

    ws = tmp_path / "op"
    ws.mkdir()
    ctx = types.SimpleNamespace(workspace=ws, lane=0, kw_1_only=True)
    res = post_spawn_transition(
        ctx, object(), "aog-kernel-worker", _Envelope(), 1
    )
    assert not (res.action == "return" and res.exit_code == 7), (
        "clean spawn was treated as gate-rejected"
    )


def test_a_new_dispatch_clears_the_previous_verdict(tmp_path):
    """The marker describes ONE dispatch, so a retry must not inherit the last one's.

    `fsm_phase_spawn._post_spawn_transition` refuses to advance state whenever the marker is
    present, and it checks it after EVERY spawn. Left in place, the first failure wedged the
    workspace permanently: the retry that fixed the artifacts was killed on sight by the
    stale reason, and the operator saw a failure message describing an earlier attempt. It is
    cleared before the agent starts — not after the gate — so a dispatch that dies before ever
    reaching the gate cannot leave the previous verdict standing either.
    """
    marker = tmp_path / ".agent_gate_stop_failed_aog-kernel-worker"
    marker.write_text("previous attempt: verification.json was fabricated\n")

    _clear_stop_gate_marker(tmp_path, "aog-kernel-worker")

    assert not marker.exists(), "a stale stop-gate verdict outlived the dispatch it described"


def test_clearing_an_absent_marker_is_not_an_error(tmp_path):
    """The normal case is no marker at all; it must not raise on the happy path."""
    _clear_stop_gate_marker(tmp_path, "aog-kernel-worker")
    _clear_stop_gate_marker(tmp_path / "nonexistent", "aog-kernel-worker")
