# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P95 regression tests — production paths broken by xulei partial commits.

Triggered by user catch (Discord 16:00Z 2026-05-15 msg 1504876140):
"你以后不能这么忽略这种改动,立刻评估影响".

xulei commit 12a1b71e (2026-05-13) added `env.get_subagent_settings(...)`
call in agent_dispatch.py:266 but never added the method/storage to
AscendCEnv. Latent bug surfaced when 3 cold-starts hit AttributeError
at orchestrator iter=0 worker spawn.

Sanity suite didn't catch: existing unit tests don't exercise the
spawn-side code path that calls _build_extra_args.

This file pins the production paths that need to work for ANY agent
spawn. Future xulei (or anyone) commit that breaks them → CI catches.

User directive 16:06Z: "之后要求其他agent进行代码review,你来补多层测试网".
This is the test-net layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))


# ── AscendCEnv contract surface ─────────────────────────────────────────

def test_ascendc_env_has_get_subagent_settings():
    """xulei 12a1b71e regression: AscendCEnv MUST expose this method."""
    from briefs._common import AscendCEnv
    assert hasattr(AscendCEnv, "get_subagent_settings"), (
        "AscendCEnv missing get_subagent_settings — xulei commit 12a1b71e "
        "added the call without adding the method"
    )
    # Constructor doesn't require subagent_settings (must default)
    env = AscendCEnv(
        target="a5", host="x", user="r", password="p", container="c",
        cann_path="/cann", soc_version="v", benchmark_root="/b",
        local_benchmark="/lb", local_project="/lp",
        archive_project="ap", build_archive_enabled=False,
    )
    # Default behavior: returns None for any agent_type when unconfigured
    assert env.get_subagent_settings("aog-kernel-worker") is None


def test_get_subagent_settings_with_config(tmp_path):
    """When SUBAGENT_SETTINGS_<NAME>=<path> is in .ascendc_env, method
    returns the path for matching agent_type.
    """
    from briefs._common import load_env
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "TARGET=a5\n"
        "A5_HOST=test\n"
        "A5_CONTAINER=test_c\n"
        "SUBAGENT_SETTINGS_AOG_KERNEL_WORKER=/path/to/kw_settings.json\n"
        "SUBAGENT_SETTINGS_AOG_PRECISION_PROBE=/path/to/probe_settings.json\n"
    )
    env = load_env(env_file)
    assert env.get_subagent_settings("aog-kernel-worker") == "/path/to/kw_settings.json"
    assert env.get_subagent_settings("aog-precision-probe") == "/path/to/probe_settings.json"
    # Unconfigured agent_type returns None
    assert env.get_subagent_settings("aog-researcher") is None


# ── agent_dispatch production path ──────────────────────────────────────

def test_agent_dispatch_build_extra_args_doesnt_crash(tmp_path):
    """Spawn-side code path that crashed in production (3 cold-starts
    simultaneously). Must not raise for any agent_type.
    """
    from briefs._common import load_env
    import agent_dispatch
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text("TARGET=a5\nA5_HOST=test\n")
    env = load_env(env_file)
    # Every agent type in production must produce a valid extra_args
    # (either None or a list[str]) without crashing.
    for agent_type in (
        "aog-kernel-worker", "aog-precision-probe", "aog-kernel-optimizer",
        "aog-fused-optimizer", "aog-researcher", "aog-determinism-analyzer",
        "aog-hardware-probe",
    ):
        result = getattr(agent_dispatch, '_build_extra_args')(env, agent_type)
        assert result is None or isinstance(result, list), (
            f"{agent_type}: _build_extra_args returned unexpected type {type(result)}"
        )


def test_agent_dispatch_build_extra_args_with_settings(tmp_path):
    """When configured, extra_args includes ['--settings', '<resolved-path>']."""
    from briefs._common import load_env
    import agent_dispatch
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "TARGET=a5\nA5_HOST=test\n"
        "SUBAGENT_SETTINGS_AOG_KERNEL_WORKER=/abs/path/kw.json\n"
    )
    env = load_env(env_file)
    result = getattr(agent_dispatch, '_build_extra_args')(env, "aog-kernel-worker")
    assert result is not None
    assert "--settings" in result
    assert "/abs/path/kw.json" in result


def test_agent_dispatch_build_extra_args_skips_settings_for_non_claude_backend(tmp_path, monkeypatch):
    """Claude --settings must not be passed to Codex/opencode backends."""
    from briefs._common import load_env
    import types
    import agent_dispatch
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "TARGET=a5\nA5_HOST=test\n"
        "SUBAGENT_SETTINGS_AOG_KERNEL_WORKER=/abs/path/kw.json\n"
    )
    env = load_env(env_file)
    monkeypatch.setattr(agent_dispatch, "_backend", types.SimpleNamespace(name="codex"))
    assert getattr(agent_dispatch, '_build_extra_args')(env, "aog-kernel-worker") is None


def test_agent_dispatch_backend_manifest_cmd_is_backend_neutral(monkeypatch):
    import types
    import agent_dispatch

    monkeypatch.setattr(agent_dispatch, "_backend", types.SimpleNamespace(name="codex"))
    assert getattr(agent_dispatch, '_backend_manifest_cmd')("aog-kernel-worker")[:2] == ["codex", "exec"]

    monkeypatch.setattr(agent_dispatch, "_backend", types.SimpleNamespace(name="opencode"))
    assert getattr(agent_dispatch, '_backend_manifest_cmd')("aog-kernel-worker")[:2] == ["opencode", "run"]


# ── load_env doesn't choke on missing SUBAGENT_SETTINGS_* keys ──────────

def test_load_env_without_subagent_settings(tmp_path):
    """Backward compat: existing .ascendc_env files without
    SUBAGENT_SETTINGS_* keys must still load.
    """
    from briefs._common import load_env
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text("TARGET=a5\nA5_HOST=test\nA5_CANN_PATH=/cann\n")
    env = load_env(env_file)
    assert env.subagent_settings is None
    assert env.get_subagent_settings("aog-kernel-worker") is None
