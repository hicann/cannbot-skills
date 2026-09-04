# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""FSM-side unit tests for A.5 (structured api_error_status retry) and
A.7 (candidate-tree stall watchdog consumer) in fsm_phase_spawn.

A.5: the retry decision consumes ONLY Envelope.api_error_status.  An errored
result whose text says "Unexpected server error" but carries no structured
status must fall through to the legacy no-op-failure path untouched.

A.7: a CandidateTreeStallTimeout from the backend respawns in place, appends
a PROGRESS.md note, and feeds the P0-1 same-signature family counter —
reaching the park threshold escalates to the agent_died abort path.

Run: cd src/scripts/orchestrator && python3 -m pytest tests/ut/test_spawn_api_error_retry.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import agent_dispatch  # noqa: E402
import critic_invoke  # noqa: E402
import events  # noqa: E402
import state_executor  # noqa: E402
import fsm_phase_spawn as S  # noqa: E402
from backends.base import Envelope  # noqa: E402
from backends.opencode_backend import CandidateTreeStallTimeout  # noqa: E402
from fsm_context import OrchestratorContext  # noqa: E402

test_seams = SimpleNamespace(module=None)


class _Snap:
    def __init__(self, state="await_worker", iter_counts=None):
        self.current_state = state
        self.iter_counts = iter_counts or {}


@pytest.fixture(autouse=True)
def _use_orchestrator_module(monkeypatch):
    """Same live-module pattern as test_fsm_phase_spawn (ctx read-throughs)."""
    module_path = _HERE.parent.parent.parent / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("orchestrator", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "orchestrator", module)
    spec.loader.exec_module(module)
    test_seams.module = module


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text("{}")
    return OrchestratorContext(op="op", workspace=ws, lane=0)


@pytest.fixture
def spawn_seams(monkeypatch):
    """Neutralize the pre-spawn seams; the spawn stub itself is per-test."""
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda st: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, t: {})
    monkeypatch.setattr(S, "_sleep_backoff", lambda s: None)


@pytest.fixture
def routed_seams(monkeypatch, spawn_seams):
    """Neutralize post-spawn routing seams (schema/handoff/next_state) for failed turns."""
    # Only a turn that was NOT api-error-retried reaches these seams; with them
    # neutralized such a turn routes onward deterministically.
    import schema_norm

    monkeypatch.setattr(
        schema_norm, "normalize_workspace",
        lambda ws, fail_strict=False: schema_norm.NormalizationReport(
            events=[], rejected_terminal_aliases=[], files_modified=[]),
    )

    class _Dec:
        from_state = "await_worker"
        next_state = "await_user_decision"
        rationale = "canned"

    monkeypatch.setattr(state_executor, "next_state", lambda *a, **k: _Dec())


def _error_envelope(*, api_error_status=None, text="boom"):
    return Envelope(
        is_error=True,
        output_text=text,
        api_error_status=api_error_status,
        raw_envelope={"backend": "opencode"},
    )


def _assert_routes_on_without_api_error_retry(ctx, monkeypatch, envelope):
    """Shared body: the errored turn routes on with no A.5 retry counter bumped."""
    from fsm_phase_spawn import _api_error_retry_decision

    assert _api_error_retry_decision(ctx, _Snap("await_worker"), "aog-kernel-worker",
                                     envelope) is None
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", lambda *a, **k: envelope)
    bumped = []
    monkeypatch.setattr(test_seams.module, "_load_silence_retry_count", lambda ws, key: 0)
    monkeypatch.setattr(test_seams.module, "_bump_silence_retry_count",
                        lambda ws, key: bumped.append(key))
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert res.action == "continue"
    assert not any("__api_error_" in key for key in bumped)


# ---------------------------------------------------------------------------
# A.5 — status classification (the ONLY retry criterion)
# ---------------------------------------------------------------------------
def test_api_error_status_retryable_classification():
    assert S.api_error_status_retryable(429) is True
    assert S.api_error_status_retryable(500) is True
    assert S.api_error_status_retryable(599) is True
    assert S.api_error_status_retryable(600) is False
    assert S.api_error_status_retryable(403) is False
    assert S.api_error_status_retryable(None) is False
    assert S.api_error_status_retryable(True) is False
    assert S.api_error_status_retryable("500") is False


def test_api_error_500_respawns_with_backoff_and_own_counter(ctx, monkeypatch, spawn_seams):
    """Envelope carrying a structured 500 triggers the dedicated retry."""
    sleeps = []
    monkeypatch.setattr(S, "_sleep_backoff", lambda s: sleeps.append(s))
    monkeypatch.setattr(agent_dispatch, "spawn_for_state",
                        lambda *a, **k: _error_envelope(api_error_status=500))
    bumped = []
    monkeypatch.setattr(test_seams.module, "_load_silence_retry_count", lambda ws, key: 0)
    monkeypatch.setattr(test_seams.module, "_bump_silence_retry_count",
                        lambda ws, key: bumped.append(key))
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert res.action == "continue"
    assert bumped == ["await_worker__api_error_500"]
    assert sleeps == [S.API_ERROR_BACKOFF_SEC]  # first retry backoff


def test_api_error_text_without_status_never_retries(ctx, monkeypatch, routed_seams):
    """Errored text with NO api_error_status must not take the A.5 retry path."""
    # Such a turn routes onward as an ordinary failed turn, no retry counter bumped.
    envelope = _error_envelope(
        api_error_status=None,
        text='{"error":{"data":{"message":"Unexpected server error. '
             'Check server logs for details."}}}',
    )
    _assert_routes_on_without_api_error_retry(ctx, monkeypatch, envelope)


def test_api_error_403_not_retryable(ctx, monkeypatch, routed_seams):
    _assert_routes_on_without_api_error_retry(
        ctx, monkeypatch, _error_envelope(api_error_status=403))


def test_api_error_budget_exhausted_escalates_abort(ctx, monkeypatch, spawn_seams):
    monkeypatch.setattr(agent_dispatch, "spawn_for_state",
                        lambda *a, **k: _error_envelope(api_error_status=429))
    monkeypatch.setattr(test_seams.module, "_load_silence_retry_count",
                        lambda ws, key: S.API_ERROR_RETRY_MAX)
    marked = {}
    monkeypatch.setattr(test_seams.module, "_mark_agent_died",
                        lambda ws, st, msg: marked.setdefault("m", msg))
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert (res.action, res.exit_code) == ("return", 3)
    assert "429" in marked["m"]


# ---------------------------------------------------------------------------
# A.7 — candidate-tree stall watchdog consumer
# ---------------------------------------------------------------------------
def _stall(*a, **k):
    raise CandidateTreeStallTimeout(
        "aog-kernel-worker", 2700.0, "treeSha123", partial_output="thinking...")


def test_tree_stall_respawns_notes_progress_and_counts(ctx, monkeypatch, spawn_seams):
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", _stall)
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert res.action == "continue"
    note = (ctx.workspace / "PROGRESS.md").read_text()
    assert "candidate-tree stall watchdog" in note
    assert "treeSha123" in note
    entry = state_executor.load_same_signature_state(ctx.workspace)["same_signature"]
    assert entry["failure_class"] == "engine"
    assert entry["count"] == 1
    assert entry["last_tree_sha256"] == "treeSha123"


def test_tree_stall_consecutive_count_accumulates(ctx, monkeypatch, spawn_seams):
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", _stall)
    assert S.handle_spawn(ctx, _Snap("await_worker")).action == "continue"
    assert S.handle_spawn(ctx, _Snap("await_worker")).action == "continue"
    entry = state_executor.load_same_signature_state(ctx.workspace)["same_signature"]
    assert entry["count"] == 2


def test_tree_stall_escalates_at_same_signature_threshold(ctx, monkeypatch, spawn_seams):
    """Third identical stall crosses the P0-1 engine-family park threshold and aborts."""
    # There is no independent parking surface; the abort is the escalation for a human.
    monkeypatch.setattr(agent_dispatch, "spawn_for_state", _stall)
    monkeypatch.setattr(state_executor, "same_signature_park_threshold", lambda cls: 3)
    marked = {}
    monkeypatch.setattr(test_seams.module, "_mark_agent_died",
                        lambda ws, st, msg: marked.setdefault("m", msg))
    assert S.handle_spawn(ctx, _Snap("await_worker")).action == "continue"
    assert S.handle_spawn(ctx, _Snap("await_worker")).action == "continue"
    res = S.handle_spawn(ctx, _Snap("await_worker"))
    assert (res.action, res.exit_code) == ("return", 3)
    assert "stall" in marked["m"]
