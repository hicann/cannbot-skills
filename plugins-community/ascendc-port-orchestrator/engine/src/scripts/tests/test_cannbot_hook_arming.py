# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""init.sh must produce an ARMED install — regression anchor for DEBT-253.

Background (2026-07-25): `init.sh` defaulted to `level=project`, writing
`settings.json` to `<plugin>/.claude/`. Claude Code loads project settings from
`<cwd>/.claude/settings.json` ONLY — it does not walk up from cwd, not even to the
git root (both measured with a marker hook). The engine always spawns subagents with
cwd=`<plugin>/engine`, so those hooks were never loaded: `output_read_guard` (④),
`workflow_critic` and `ship_claim_audit` were silently inert while the installer
printed `✓ installed` even though the runtime gates were inert. An a3-port e2e ran to a green verdict
that way before the gap was caught.

The two properties below are what make an install armed. They are asserted
separately on purpose: "the guard logic is correct" and "the guard is ever loaded"
look identical from outside, and only the second one was broken.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[4]          # <plugin>/
ENGINE = PLUGIN_ROOT / "engine"
INIT_SH = PLUGIN_ROOT / "init.sh"
GUARD = ENGINE / "src" / "scripts" / "workflow" / "output_read_guard.py"
OWNER = "ascendc-port-orchestrator"
REQUIRED_HOOKS = (
    "output_read_guard.py",
    "workflow_critic.py",
    "ship_claim_audit.py",
    "agent-gate-dispatch.py",
)


def _owned_commands(settings: Path) -> list[str]:
    data = json.loads(settings.read_text())
    commands = []
    for event in ("PreToolUse", "PostToolUse", "SubagentStop"):
        for matcher in data.get("hooks", {}).get(event, []):
            for hook in matcher.get("hooks", []):
                if hook.get("_owner") == OWNER:
                    commands.append(hook.get("command", ""))
    return commands


def _guard_verdict(payload: dict) -> int:
    """Run the guard hook exactly as CC would and return its exit code."""
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    return proc.returncode


# ---------------------------------------------------------------- LOCATION --
def test_init_registers_hooks_where_the_engine_runs():
    """init.sh must target the engine's own cwd, not just CONFIG_ROOT.

    Reads init.sh rather than running it: the installer touches $HOME (user KB)
    and CONFIG_ROOT, which a unit test must not do. The behavioural end of this
    property is proven at install time by init.sh's own step-4 self-check.
    """
    src = INIT_SH.read_text()
    assert 'ENGINE_SETTINGS="$PLUGIN_DIR/engine/.claude/settings.json"' in src, (
        "init.sh no longer registers hooks into the engine cwd. CC only reads "
        "<cwd>/.claude/settings.json and the engine runs from engine/, so hooks "
        "written elsewhere are never loaded (DEBT-253)."
    )
    assert '"$CONFIG_ROOT/settings.json" "$ENGINE_SETTINGS"' in src, (
        "hook registration must cover BOTH the chosen CONFIG_ROOT and the engine cwd"
    )


def test_installer_refuses_to_tick_ok_without_a_liveness_proof():
    """Counting installed files is what let a disarmed install print ✓."""
    src = INIT_SH.read_text()
    assert "hooks_live" in src, "install-time hook liveness check is gone"
    assert 'health_ok=false' in src.split("hooks_live=true", 1)[1], (
        "a failed hook liveness check must fail the install, not just warn"
    )
    for name in REQUIRED_HOOKS:
        assert name in src, f"{name} dropped from the installer's liveness check"
    assert 'hooks["SubagentStop"]' in src
    assert 'hooks["Stop"]' not in src


@pytest.mark.skipif(
    not (ENGINE / ".claude" / "settings.json").is_file(),
    reason="engine/.claude/settings.json only exists after init.sh has run here",
)
def test_installed_engine_settings_carry_all_three_hooks():
    cmds = _owned_commands(ENGINE / ".claude" / "settings.json")
    missing = [n for n in REQUIRED_HOOKS if not any(n in c for c in cmds)]
    assert not missing, f"engine-cwd settings missing hooks CC would load: {missing}"


# ---------------------------------------------------------------- BEHAVIOUR --
def test_guard_denies_subagent_output_read():
    """A generation subagent reading output/ is the cheat this guard exists for."""
    rc = _guard_verdict({
        "agent_id": "test-subagent",
        "tool_name": "Read",
        "tool_input": {"file_path": str(ENGINE / "output" / "proj" / "src" / "kernels" / "op" / "verification.json")},
        "cwd": str(ENGINE),
    })
    assert rc == 2, f"expected DENY (exit 2) for a subagent output/ read, got {rc}"


def test_guard_passes_main_agent():
    """Discrimination control: an always-deny guard is as broken as an always-allow
    one, and would make the coordinator agent unable to read its own archives.
    """
    rc = _guard_verdict({
        "tool_name": "Read",
        "tool_input": {"file_path": str(ENGINE / "output" / "proj" / "src" / "kernels" / "op" / "verification.json")},
        "cwd": str(ENGINE),
    })
    assert rc == 0, f"main agent (no agent_id) must be exempt, got exit {rc}"
