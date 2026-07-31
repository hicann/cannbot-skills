# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""The O0 hook gate must validate activation, not merely packaged files."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]
REAL_PLUGIN_ROOT = HERE.parents[4]
_BASH = shutil.which("bash")
sys.path.insert(0, str(SCRIPTS))

import preflight_install_hooks as checker  # noqa: E402


def _bash() -> str:
    if _BASH is None:
        pytest.skip("bash executable not found")
    return _BASH


def _plugin_tree(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    plugin = tmp_path / "plugin"
    (plugin / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin / "hooks").mkdir(exist_ok=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "ascendc-port-orchestrator"}), encoding="utf-8"
    )
    hooks = json.loads(
        (REAL_PLUGIN_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8")
    )
    (plugin / "hooks" / "hooks.json").write_text(
        json.dumps(hooks), encoding="utf-8"
    )
    monkeypatch.setattr(checker, "PLUGIN_ROOT", plugin)
    monkeypatch.setattr(checker, "PLUGIN_HOOKS", plugin / "hooks" / "hooks.json")
    monkeypatch.setattr(checker, "PROJECT_SETTINGS", plugin / "engine" / ".claude" / "settings.json")
    return plugin, hooks


def _marketplace(tmp_path: Path, monkeypatch, *, enabled: bool = True,
                 install_path: Path | None = None) -> tuple[Path, dict]:
    plugin, hooks = _plugin_tree(tmp_path, monkeypatch)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.setattr(checker, "REPO_OPS_ROOT", tmp_path / "absent-ops")
    monkeypatch.setattr(checker.shutil, "which", lambda _: "/fake/claude")
    listed = [{
        "id": "ascendc-port-orchestrator@cannbot",
        "enabled": enabled,
        "installPath": str(install_path or plugin),
    }]
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(listed), stderr=""
        ),
    )
    return plugin, hooks


def _direct(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    plugin, plugin_hooks = _plugin_tree(tmp_path, monkeypatch)
    ops = tmp_path / "ops"
    ops.mkdir()
    monkeypatch.setattr(checker, "REPO_OPS_ROOT", ops)
    direct_hooks = {}
    for event in ("PreToolUse", "PostToolUse", "SubagentStop"):
        direct_hooks[event] = copy.deepcopy(plugin_hooks["hooks"][event])
        for matcher in direct_hooks[event]:
            for hook in matcher["hooks"]:
                hook["command"] = hook["command"].replace(
                    '"${CLAUDE_PLUGIN_ROOT}', f'"{plugin}'
                )
                hook["_owner"] = "ascendc-port-orchestrator"
    settings = checker.PROJECT_SETTINGS
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"hooks": direct_hooks}), encoding="utf-8")
    return plugin, direct_hooks


def test_marketplace_registration_requires_enabled_current_install(tmp_path, monkeypatch):
    plugin, _ = _marketplace(tmp_path, monkeypatch)
    mode, declaration, errors = checker.check_current_registration()
    assert mode == "plugin"
    assert declaration == plugin / "hooks" / "hooks.json"
    assert errors == []

    _marketplace(tmp_path, monkeypatch, enabled=False)
    assert any("not installed and enabled" in error
               for error in checker.check_current_registration()[2])

    stale = tmp_path / "stale-cache"
    _marketplace(tmp_path, monkeypatch, install_path=stale)
    assert any("enabled plugin path" in error
               for error in checker.check_current_registration()[2])


def test_marketplace_project_scope_registry_fallback_from_engine_cwd(tmp_path, monkeypatch):
    """Project installs remain armed when the orchestrator runs from engine/."""
    plugin, _ = _plugin_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(checker, "REPO_OPS_ROOT", tmp_path / "absent-ops")
    monkeypatch.setattr(checker.shutil, "which", lambda _: "/fake/claude")
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "id": "ascendc-port-orchestrator@cannbot",
                "enabled": False,
                "installPath": str(plugin),
            }]),
            stderr="",
        ),
    )

    config_root = tmp_path / "claude-config"
    project = tmp_path / "customer-project"
    settings = project / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"ascendc-port-orchestrator@cannbot": True}}),
        encoding="utf-8",
    )
    registry = config_root / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"plugins": {"ascendc-port-orchestrator@cannbot": [{
            "scope": "project",
            "projectPath": str(project),
            "installPath": str(plugin),
        }]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_root))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))

    assert checker.check_current_registration()[2] == []

    settings.write_text(
        json.dumps({"enabledPlugins": {"ascendc-port-orchestrator@cannbot": False}}),
        encoding="utf-8",
    )
    assert any("not installed and enabled" in error
               for error in checker.check_current_registration()[2])


def test_marketplace_registry_rejects_other_project_and_invalid_install_path(tmp_path, monkeypatch):
    """A project switch only arms the project that actually runs the worker."""
    plugin, _ = _plugin_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(checker, "REPO_OPS_ROOT", tmp_path / "absent-ops")
    monkeypatch.setattr(checker.shutil, "which", lambda _: "/fake/claude")
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "id": "ascendc-port-orchestrator@cannbot",
                "enabled": False,
                "installPath": str(plugin),
            }]),
            stderr="",
        ),
    )
    config_root = tmp_path / "claude-config"
    registered = tmp_path / "registered-project"
    running = tmp_path / "running-project"
    settings = registered / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"ascendc-port-orchestrator@cannbot": True}}),
        encoding="utf-8",
    )
    registry = config_root / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"plugins": {"ascendc-port-orchestrator@cannbot": [{
            "scope": "project",
            "projectPath": str(registered),
            "installPath": str(plugin),
        }]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_root))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(running))
    errors = checker.check_current_registration()[2]
    assert any("does not match running project" in error for error in errors)

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(registered))
    registry.write_text(
        json.dumps({"plugins": {"ascendc-port-orchestrator@cannbot": [{
            "scope": "project",
            "projectPath": str(registered),
            "installPath": None,
        }]}}),
        encoding="utf-8",
    )
    errors = checker.check_current_registration()[2]
    assert any("installPath is invalid" in error for error in errors)


def _worker_workspace(tmp_path: Path, *, pybind: str,
                      kernel: str = "void kernel() { DataCopy(); }\n") -> Path:
    workspace = tmp_path / "workspace" / "gelu_backward"
    kernel_dir = workspace / "kernel"
    kernel_dir.mkdir(parents=True)
    (workspace / "PROGRESS.md").write_text("→ orchestrator: active\n", encoding="utf-8")
    (workspace / "analysis.md").write_text(
        "\n".join([
            "## Source & references",
            "- algorithm_family: test",
            "## Dtypes & shapes",
            "- dtypes: fp16",
            "## Precision-critical ops",
            "## Architecture decision",
            "- choice: test",
            "## KB Manifest",
            "## Precision traps",
            "## UB budget estimate",
        ]) + "\n",
        encoding="utf-8",
    )
    (kernel_dir / "pybind11.cpp").write_text(pybind, encoding="utf-8")
    (kernel_dir / "gelu_kernel.h").write_text(kernel, encoding="utf-8")
    return workspace


def _run_worker_hook(tmp_path: Path, workspace: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update({
        "WORKSPACE_ROOT": str(tmp_path / "workspace"),
        "CLAUDE_ACTIVE_WORKSPACE": str(workspace),
        # These fixtures exercise worker-hook contracts independently of the
        # separate AscendC static checker and a compilable kernel fixture.
        "LOCAL_PROJECT": str(tmp_path / "no-static-check"),
    })
    return subprocess.run(
        [_bash(), str(REAL_PLUGIN_ROOT / "hooks" / "v3" / "check_worker.sh")],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_worker_hook_allows_npu_pybind_and_blocks_invalid_artifacts(tmp_path):
    """A blocking hook needs both a valid pass control and malformed-artifact checks."""
    workspace = _worker_workspace(
        tmp_path,
        pybind=(
            "#include <torch_npu/csrc/core/npu/NPUStream.h>\n"
            "void launch() { auto stream = c10_npu::getCurrentNPUStream().stream(false); }\n"
        ),
    )
    assert _run_worker_hook(tmp_path, workspace).returncode == 0

    (workspace / "kernel" / "gelu_kernel.h").write_text(
        "void kernel() {}\n", encoding="utf-8"
    )
    primitive_result = _run_worker_hook(tmp_path, workspace)
    assert primitive_result.returncode == 2
    assert "not a real AscendC kernel" in primitive_result.stderr

    (workspace / "kernel" / "gelu_kernel.h").write_text(
        "void kernel() { DataCopy(); }\n", encoding="utf-8"
    )
    (workspace / "verification.json").write_text("{not-json\n", encoding="utf-8")
    verification_result = _run_worker_hook(tmp_path, workspace)
    assert verification_result.returncode == 2
    assert "verification.json schema errors" in verification_result.stderr


def test_marketplace_hook_closure_is_exact_and_read_only(tmp_path, monkeypatch):
    plugin, hooks = _marketplace(tmp_path, monkeypatch)
    declaration = plugin / "hooks" / "hooks.json"
    before = declaration.read_bytes()
    hooks["hooks"]["PreToolUse"][0]["hooks"].clear()
    declaration.write_text(json.dumps(hooks), encoding="utf-8")
    mutated = declaration.read_bytes()
    errors = checker.check_current_registration()[2]
    assert any("workflow_critic.py" in error for error in errors)
    assert any("closure size" in error for error in errors)
    assert declaration.read_bytes() == mutated
    assert before != mutated


def test_stop_gate_must_use_subagent_stop(tmp_path, monkeypatch):
    plugin, hooks = _marketplace(tmp_path, monkeypatch)
    hooks["hooks"]["Stop"] = hooks["hooks"].pop("SubagentStop")
    (plugin / "hooks" / "hooks.json").write_text(json.dumps(hooks), encoding="utf-8")
    errors = checker.check_current_registration()[2]
    assert any("SubagentStop" in error for error in errors)
    assert any("incorrectly registered on Stop" in error for error in errors)


def test_direct_registration_requires_exact_owned_closure(tmp_path, monkeypatch):
    _, hooks = _direct(tmp_path, monkeypatch)
    mode, _, errors = checker.check_current_registration()
    assert mode == "direct-settings"
    assert errors == []

    hooks["PreToolUse"][0]["hooks"][0]["_owner"] = "ascendc-op-gen"
    checker.PROJECT_SETTINGS.write_text(
        json.dumps({"hooks": hooks}), encoding="utf-8"
    )
    errors = checker.check_current_registration()[2]
    assert any("legacy ascendc-op-gen" in error for error in errors)
    assert any("closure size" in error for error in errors)


def test_legacy_default_installer_is_fail_closed(tmp_path, monkeypatch, capsys):
    plugin, _ = _plugin_tree(tmp_path, monkeypatch)
    try:
        checker.cmd_install(SimpleNamespace())
    except BaseException as exc:
        assert type(exc).__name__ == "SystemExit"
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("legacy installer unexpectedly returned")
    output = capsys.readouterr().out
    assert str(plugin / "init.sh") in output
    assert "/aog-preflight" not in output
