# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""The installed workflow critic must resolve the community-plugin layout."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve()
WORKFLOW_DIR = HERE.parents[1]
ENGINE_ROOT = HERE.parents[4]
PLUGIN_ROOT = HERE.parents[5]
sys.path.insert(0, str(WORKFLOW_DIR))

import workflow_critic_common as common  # noqa: E402
import workflow_critic as critic  # noqa: E402


AGENT_GATE = PLUGIN_ROOT / "hooks" / "agent-gate-dispatch.py"
PLUGIN_HOOKS = PLUGIN_ROOT / "hooks" / "hooks.json"


def test_workflow_critic_loads_the_bundled_fsm_from_real_layout():
    assert common.REPO_ROOT == ENGINE_ROOT
    assert common.YAML_PATH == PLUGIN_ROOT / "workflows" / "opgen_state_machine.yaml"
    assert common.YAML_PATH.is_file()
    machine = common.load_state_machine()
    assert machine["workflow_name"] == "ascendc-op-gen"
    assert machine.get("phases"), "state machine loaded but contains no phases"


def test_agent_gate_scripts_are_packaged_and_not_home_dependent():
    expected = {
        "_common.sh",
        "block_edit_on_infra.sh",
        "check_fused_optimizer_artifacts.sh",
        "check_optimizer_artifacts.sh",
        "check_probe_report.sh",
        "check_progress_signed.sh",
        "check_worker.sh",
    }
    hook_root = PLUGIN_ROOT / "hooks" / "v3"
    assert {path.name for path in hook_root.glob("*.sh")} == expected
    assert AGENT_GATE.is_file()
    for agent in (PLUGIN_ROOT / "agents").glob("*.md"):
        text = agent.read_text(encoding="utf-8")
        assert "$HOME/.claude/hooks" not in text


def test_agent_gate_dispatch_noops_for_unrelated_agent(tmp_path):
    result = subprocess.run(
        [sys.executable, str(AGENT_GATE), "stop"],
        input=json.dumps({"agent_type": "unrelated-agent"}),
        text=True,
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_agent_stop_gates_are_registered_for_subagents():
    hooks = json.loads(PLUGIN_HOOKS.read_text(encoding="utf-8"))["hooks"]
    assert "Stop" not in hooks
    commands = [
        hook.get("command", "")
        for matcher in hooks.get("SubagentStop", [])
        for hook in matcher.get("hooks", [])
    ]
    assert any("agent-gate-dispatch.py" in command and " stop" in command
               for command in commands)


def test_agent_gate_dispatch_preserves_worker_infra_edit_block(tmp_path):
    workspace = tmp_path / "workspace" / "demo"
    kernel = workspace / "kernel" / "demo_kernel.h"
    kernel.parent.mkdir(parents=True)
    (workspace / "PROGRESS.md").write_text("# progress\n", encoding="utf-8")
    (workspace / ".last_build.class").write_text("infra\n", encoding="utf-8")
    (workspace / ".last_build.stderr").write_text("network unavailable\n", encoding="utf-8")
    env = dict(os.environ, CLAUDE_ACTIVE_WORKSPACE=str(workspace))
    payload = {
        "agent_type": "aog-kernel-worker",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(kernel)},
    }
    result = subprocess.run(
        [sys.executable, str(AGENT_GATE), "pretool"],
        input=json.dumps(payload),
        text=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "infra block" in result.stderr


def test_staged_path_resolution_accepts_monorepo_and_standalone_prefixes():
    monorepo_root = PLUGIN_ROOT.parents[1]
    yaml_monorepo = common.YAML_PATH.relative_to(monorepo_root).as_posix()
    skill = PLUGIN_ROOT / "skills" / "ascendc-cross-gen-port" / "SKILL.md"
    skill_monorepo = skill.relative_to(monorepo_root).as_posix()
    assert getattr(critic, "_absolutize_staged_paths")(
        monorepo_root, [yaml_monorepo, skill_monorepo]
    ) == {common.YAML_PATH.resolve(), skill.resolve()}
    assert getattr(critic, "_absolutize_staged_paths")(
        PLUGIN_ROOT,
        ["workflows/opgen_state_machine.yaml", "skills/ascendc-cross-gen-port/SKILL.md"],
    ) == {common.YAML_PATH.resolve(), skill.resolve()}


def test_pre_commit_sync_accepts_yaml_with_affected_entry_skill(monkeypatch):
    monkeypatch.setattr(
        critic,
        "_read_staged_paths",
        lambda: {common.YAML_PATH.resolve(), critic.ENTRY_SKILL_PATHS[1].resolve()},
    )
    critic.mode_pre_commit_sync()


@pytest.mark.parametrize("changed", ["yaml", "skill"])
def test_pre_commit_sync_rejects_one_sided_drift(monkeypatch, capsys, changed):
    staged = ({common.YAML_PATH.resolve()} if changed == "yaml"
              else {critic.ENTRY_SKILL_PATHS[0].resolve()})
    monkeypatch.setattr(critic, "_read_staged_paths", lambda: staged)
    with pytest.raises(BaseException) as exc:
        critic.mode_pre_commit_sync()
    assert type(exc.value).__name__ == "SystemExit"
    assert exc.value.code == 2
    assert "rule DRIFT" in capsys.readouterr().err


def test_pre_commit_sync_checks_archive_after_monorepo_prefix_normalization(
        monkeypatch, tmp_path, capsys):
    engine = tmp_path / "plugins-community" / "ascendc-port-orchestrator" / "engine"
    archive = engine / "output" / "demo" / "src" / "kernels" / "demo_op"
    archive.mkdir(parents=True)
    kernel = archive / "kernel.cpp"
    kernel.write_text("// staged\n", encoding="utf-8")
    monkeypatch.setattr(critic, "REPO_ROOT", engine)
    monkeypatch.setattr(critic, "_read_staged_paths", lambda: {kernel.resolve()})
    with pytest.raises(BaseException) as exc:
        critic.mode_pre_commit_sync()
    assert type(exc.value).__name__ == "SystemExit"
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "rule SC1" in err
    assert "output/demo/src/kernels/demo_op" in err
