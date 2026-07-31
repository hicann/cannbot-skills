# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0pp (2026-05-06): Phase O0 hook integrity gate.

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 7.

Verifies KB + hook + deploy infrastructure is present BEFORE
orchestrator spawns agents. Lower-damage gap than O2.5 / O5 but
explicit pre-check fails loud rather than silent downstream symptoms.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o0  # noqa: E402


def test_ready_in_real_repo_when_registration_is_armed(monkeypatch):
    """Package files plus an active registration should be READY.

    The registration probe is mocked so clean CI never depends on a developer's
    ignored engine settings or personal marketplace registry.
    """
    monkeypatch.setattr(
        phase_o0,
        "_check_hook_registration",
        lambda: ("plugin", Path("hooks/hooks.json"), []),
    )
    rep = phase_o0.check_hook_integrity()
    assert rep.verdict == "READY"
    assert not rep.missing_files
    assert not rep.missing_scripts


def test_blocked_when_critical_missing(tmp_path, monkeypatch):
    """Simulate missing KB file → BLOCKED."""
    # Patch _PROJECT_ROOT to point at empty tmp_path
    monkeypatch.setattr(phase_o0, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        phase_o0,
        "_check_hook_registration",
        lambda: ("plugin", Path("hooks/hooks.json"), []),
    )
    rep = phase_o0.check_hook_integrity()
    assert rep.verdict == "BLOCKED"
    # Should list missing files
    assert any("KB_INDEX.md" in f or "ANTI_PRESSURE_PROTOCOLS.md" in f
               or "workflow_critic.py" in f for f in rep.missing_files)


def test_degraded_when_deploy_missing_only(tmp_path, monkeypatch):
    """Critical files present but deploy scripts missing → DEGRADED.

    Layout mirrors the real plugin: tmp_path is the plugin-root, engine/ is
    _PROJECT_ROOT, KB lives at <plugin_root>/kb/ and the FSM at
    <plugin_root>/workflows/ (both relocated 2026-07-05).
    """
    engine = tmp_path / "engine"
    monkeypatch.setattr(phase_o0, "_PROJECT_ROOT", engine)
    monkeypatch.setattr(phase_o0, "_kb_root", lambda: tmp_path / "kb")
    monkeypatch.setattr(
        phase_o0,
        "_check_hook_registration",
        lambda: ("plugin", Path("hooks/hooks.json"), []),
    )
    # Engine-relative critical hook files
    for rel in phase_o0.REQUIRED_FILES:
        full = engine / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("# stub")
    # KB critical files at <plugin_root>/kb/
    for rel in phase_o0.REQUIRED_KB_FILES:
        full = tmp_path / "kb" / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("# stub")
    # Workflow FSM at <plugin_root>/workflows/
    for rel in phase_o0.REQUIRED_WORKFLOW_FILES:
        full = tmp_path / "workflows" / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("# stub")
    # But DON'T create deploy scripts
    rep = phase_o0.check_hook_integrity()
    assert rep.verdict == "DEGRADED", rep
    assert rep.missing_scripts


def test_format_block_message_lists_files():
    rep = phase_o0.O0Report(
        verdict="BLOCKED",
        missing_files=["src/skills/references/KB_INDEX.md"],
        summary="test",
    )
    msg = phase_o0.format_block_message(rep)
    assert "KB_INDEX.md" in msg
    assert "test" in msg


def test_registration_error_blocks_and_explains_current_remediation(monkeypatch):
    monkeypatch.setattr(
        phase_o0,
        "_check_hook_registration",
        lambda: ("plugin", Path("hooks/hooks.json"), ["plugin is disabled"]),
    )
    rep = phase_o0.check_hook_integrity()
    assert rep.verdict == "BLOCKED"
    msg = phase_o0.format_block_message(rep)
    assert "plugin is disabled" in msg
    assert "init.sh" in msg
    assert "/aog-preflight" not in msg
    assert "/ascendc-op-gen" not in msg


def test_warnings_rendered_in_message():
    rep = phase_o0.O0Report(
        verdict="DEGRADED",
        warnings=["sample warning"],
        summary="test",
    )
    msg = phase_o0.format_block_message(rep)
    assert "sample warning" in msg
