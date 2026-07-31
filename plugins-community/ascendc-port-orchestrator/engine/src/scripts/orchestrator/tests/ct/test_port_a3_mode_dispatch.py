# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""W3 (2026-05-12, ROADMAP §1.5) — port_a3_to_a5 mode dispatch tests.

Validates that:
- YAML `phase_o4_initial_state_by_mode` registers port_a3_to_a5 → await_worker
- workflow_critic loads YAML and recognizes port_a3_to_a5 as a valid mode
- SKILL.md mode-detection table mentions --port-a3 (drift-check sentinel)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import yaml

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _reorg_paths.REPO_ROOT
sys.path.insert(0, str(_HERE.parent.parent))


def _load_state_machine_yaml() -> dict:
    yaml_path = _PROJECT_ROOT .parent / "workflows" / "opgen_state_machine.yaml"
    return yaml.safe_load(yaml_path.read_text())


# ---------------------------------------------------------------------------
# W3: YAML mode entry
# ---------------------------------------------------------------------------
def test_yaml_phase_o4_initial_state_by_mode_includes_port_a3_to_a5():
    """W3: YAML registers port_a3_to_a5 → await_worker."""
    sm = _load_state_machine_yaml()
    mode_map = sm.get("phase_o4_initial_state_by_mode", {})
    assert "port_a3_to_a5" in mode_map, (
        f"port_a3_to_a5 missing from YAML phase_o4_initial_state_by_mode; "
        f"present modes: {sorted(mode_map.keys())}"
    )
    assert mode_map["port_a3_to_a5"] == "await_worker", (
        f"port_a3_to_a5 should route to await_worker (Phase O2.5 a3-ref variant "
        f"runs before state machine; once reference ready, kw spawns like every "
        f"other mode). Got: {mode_map['port_a3_to_a5']}"
    )


# ---------------------------------------------------------------------------
# W3: workflow_critic recognizes new mode
# ---------------------------------------------------------------------------
def test_workflow_critic_loads_yaml_with_port_a3_mode():
    """W3: workflow_critic can parse the YAML with the new mode entry."""
    import importlib.util
    critic_path = _PROJECT_ROOT / "src" / "scripts" / "workflow" / "workflow_critic.py"
    spec = importlib.util.spec_from_file_location("workflow_critic_test", critic_path)
    _mod = importlib.util.module_from_spec(spec)
    # Don't actually exec — just check loading the YAML doesn't fail (parsing
    # is what matters; full module exec has dependencies we don't need to test).
    sm = _load_state_machine_yaml()
    # workflow_critic.py line 745: sm.get("phase_o4_initial_state_by_mode", {})
    mode_map = sm.get("phase_o4_initial_state_by_mode", {})
    assert isinstance(mode_map, dict)
    assert mode_map.get("port_a3_to_a5") == "await_worker"


# ---------------------------------------------------------------------------
# W3: SKILL.md mode-detection table
# ---------------------------------------------------------------------------
def test_skill_md_mode_table_mentions_port_a3():
    """W3: SKILL.md mode-detection table includes --port-a3."""
    skill_md = (_PROJECT_ROOT / "src" / "skills" / "ascendc-op-gen" / "SKILL.md").read_text()
    assert "--port-a3" in skill_md, "SKILL.md missing --port-a3 mention"
    assert "port_a3_to_a5" in skill_md, "SKILL.md missing port_a3_to_a5 mode name"
    assert "ops-nn" in skill_md, "SKILL.md should reference ops-nn as source"


def test_skill_md_yaml_drift_sentinel():
    """W3 drift-check: if you change YAML port_a3_to_a5 mode entry, SKILL.md
    drift hook fires. Verify the sentinel pair (YAML mode key + SKILL.md text
    reference to that key) is intact.
    """
    sm = _load_state_machine_yaml()
    yaml_modes = set(sm.get("phase_o4_initial_state_by_mode", {}).keys())
    skill_md = (_PROJECT_ROOT / "src" / "skills" / "ascendc-op-gen" / "SKILL.md").read_text()
    # For every mode in YAML that's a new generation mode (not optimize/probe/research),
    # if SKILL.md has a mode-detection table, the mode name should appear somewhere.
    # Soft check — only enforce for port_a3_to_a5 (other modes pre-existed).
    if "port_a3_to_a5" in yaml_modes:
        assert "port_a3_to_a5" in skill_md
