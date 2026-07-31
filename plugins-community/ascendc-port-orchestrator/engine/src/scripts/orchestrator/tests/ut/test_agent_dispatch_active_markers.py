# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import agent_dispatch  # noqa: E402


class _FakeBackend:
    name = "opencode"

    def __init__(self, workspace: Path, marker_name: str):
        self.workspace = workspace
        self.marker_name = marker_name

    def dispatch(self, *args, **kwargs):
        assert (self.workspace / self.marker_name).exists()
        return SimpleNamespace(
            success=True,
            is_error=False,
            duration_ms=1,
            cost_usd=None,
            session_id=None,
            terminal_reason=None,
            raw_envelope={},
            output_text="done",
        )


def _env():
    return SimpleNamespace(
        opgen_mode="backward",
        port_a3_source=None,
        get_subagent_settings=lambda _agent_type: None,
    )


def _builder(op, workspace, **_kwargs):
    return f"brief for {op} in {workspace}"


def _patch_dispatch_basics(monkeypatch, workspace: Path, agent_type: str, marker_name: str):
    monkeypatch.setattr(agent_dispatch, "_backend", _FakeBackend(workspace, marker_name))
    monkeypatch.setattr(agent_dispatch.state_executor, "next_agent", lambda _state: agent_type)
    monkeypatch.setattr(agent_dispatch.state_executor, "iter_cap", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(agent_dispatch.state_executor, "iter_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(agent_dispatch, "load_env", _env)
    monkeypatch.setattr(agent_dispatch, "_build_extra_args", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(agent_dispatch.BRIEF_BUILDERS, agent_type, _builder)


def test_spawn_for_state_marks_kernel_worker_active_during_dispatch(monkeypatch, tmp_path):
    _patch_dispatch_basics(monkeypatch, tmp_path, "aog-kernel-worker", ".kernel_worker_active")

    agent_dispatch.spawn_for_state("op", tmp_path, "await_worker", lane=0, spawn_index=1)

    assert not (tmp_path / ".kernel_worker_active").exists()


def test_spawn_for_state_marks_optimizer_active_during_dispatch(monkeypatch, tmp_path):
    _patch_dispatch_basics(monkeypatch, tmp_path, "aog-kernel-optimizer", ".optimizer_active")

    agent_dispatch.spawn_for_state("op", tmp_path, "await_optimizer", lane=0, spawn_index=1)

    assert not (tmp_path / ".optimizer_active").exists()
