# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""CANN-LEARN-ON-RESEARCH-GAP Phase 2a: port_a3 plugin opt-in.

Design: docs/design/KB_DESIGN_NOTES.md#cann-learn-on-research-gap-design-2026-05-20 §3.7
Phase 1 (PRs #65 + #66 + #67) shipped the FSM + spawn fabric + CLI flag —
all defaults OFF. Phase 2a flips port_a3 to default-ON: port-from-CANN
scenarios benefit directly from CANN pattern extraction, and the carve-out
scanners (C34a-C35) protect against unsafe copy-paste.

These tests pin:
- PortA3Plugin.should_auto_cann_learn_on_gap returns True
- BackwardPlugin and BasePlugin still return False
- End-to-end: state machine condition eval_condition with port_a3 plugin
  + research_blocked handoff + iter_below_cap routes to await_cann_learn

NO Phase 1 routing change: same FSM as Phase 1a; this PR just makes one
plugin actually return True from the gate method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))


def test_port_a3_plugin_should_auto_cann_learn_on_gap_returns_true():
    """The port_a3 plugin opts in unconditionally — per-paradigm decision."""
    from plugins.port_a3 import PortA3Plugin
    p = PortA3Plugin()

    # Should return True regardless of op_class / op_complexity / mode
    assert p.should_auto_cann_learn_on_gap("FUSED SOFTMAX", "L4", "port_a3_to_a5") is True
    assert p.should_auto_cann_learn_on_gap("ELEMENTWISE", "L1", "port_a3_to_a5") is True
    assert p.should_auto_cann_learn_on_gap("unknown", "unknown", "port_a3_to_a5") is True


def test_other_plugins_still_default_false():
    """Research recovery is migration-only; backward and base stay disabled."""
    from plugins.base import BasePlugin
    from plugins.backward import BackwardPlugin

    for plug in [BasePlugin(), BackwardPlugin()]:
        assert plug.should_auto_cann_learn_on_gap("FA", "L4", "any") is False, (
            f"{plug.__class__.__name__} should retain the neutral False default"
        )


def test_eval_condition_routes_to_cann_learn_for_port_a3_workspace(tmp_path):
    """End-to-end: when port_a3 plugin is active AND researcher emits
    research_blocked AND iter cap available, state machine routes to
    await_cann_learn.
    """
    import state_machine as sm
    from plugins.port_a3 import PortA3Plugin

    ws = tmp_path / "ws_port_a3_research_blocked"
    ws.mkdir()

    snap = sm.snapshot(ws)
    snap["op_taxonomy"] = {"class": "ANY", "complexity": "L3"}

    ctx = {
        "handoff": "→ orchestrator: research_blocked",
        "snapshot": snap,
        "iter_counts": {"cann_learn": 0},
        "ws": ws,
        "sm": sm.load_state_machine(),
        "plugin": PortA3Plugin(),
    }

    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: research_blocked"},
            {"iter_below_cap": "cann_learn"},
            {"plugin_method": "should_auto_cann_learn_on_gap"},
        ]
    }
    assert sm.eval_condition(cond, ctx) is True


def test_eval_condition_routes_to_cann_learn_for_port_a3_research_partial(tmp_path):
    """Mirror — research_partial should also route to cann_learn under port_a3."""
    import state_machine as sm
    from plugins.port_a3 import PortA3Plugin

    ws = tmp_path / "ws_port_a3_research_partial"
    ws.mkdir()
    snap = sm.snapshot(ws)
    snap["op_taxonomy"] = {"class": "ANY", "complexity": "L2"}

    ctx = {
        "handoff": "→ orchestrator: research_partial",
        "snapshot": snap,
        "iter_counts": {"cann_learn": 0},
        "ws": ws,
        "sm": sm.load_state_machine(),
        "plugin": PortA3Plugin(),
    }
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: research_partial"},
            {"iter_below_cap": "cann_learn"},
            {"plugin_method": "should_auto_cann_learn_on_gap"},
        ]
    }
    assert sm.eval_condition(cond, ctx) is True


def test_eval_condition_skips_when_port_a3_iter_cap_hit(tmp_path):
    """Even with port_a3 plugin opt-in, iter_below_cap=False (cap=1, count=1)
    must skip the cann_learn transition — prevents re-entry loop.
    """
    import state_machine as sm
    from plugins.port_a3 import PortA3Plugin

    ws = tmp_path / "ws_iter_cap_hit"
    ws.mkdir()
    snap = sm.snapshot(ws)
    snap["op_taxonomy"] = {"class": "ANY", "complexity": "L2"}

    ctx = {
        "handoff": "→ orchestrator: research_blocked",
        "snapshot": snap,
        "iter_counts": {"cann_learn": 1},  # at cap=1
        "ws": ws,
        "sm": sm.load_state_machine(),
        "plugin": PortA3Plugin(),
    }
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: research_blocked"},
            {"iter_below_cap": "cann_learn"},
            {"plugin_method": "should_auto_cann_learn_on_gap"},
        ]
    }
    assert sm.eval_condition(cond, ctx) is False
