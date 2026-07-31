# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""CANN-LEARN-ON-RESEARCH-GAP Phase 1a: FSM `await_cann_learn` state +
plugin gate + state_executor mapping.

Design: docs/design/KB_DESIGN_NOTES.md#cann-learn-on-research-gap-design-2026-05-20
Owner direction (2026-05-20): green-light Phase 1, mechanical infra only;
no behavior change until Phase 2 plugin opt-ins.

These tests pin:
- YAML await_cann_learn state declared with iter_cap=1, iter_counter=cann_learn,
  agent=aog-cann-learner, and 4 exit transitions (cann_learn_done →
  await_researcher; cann_learn_empty/blocked → finalize; catch-all → finalize)
- await_researcher gains TWO new exit transitions (research_partial gate +
  research_blocked gate) BEFORE the existing fallbacks, both gated on
  plugin_method should_auto_cann_learn_on_gap + iter_below_cap cann_learn
- state_executor.STATE_TO_AGENT maps await_cann_learn → aog-cann-learner
- BasePlugin.should_auto_cann_learn_on_gap exists with default False
  (Phase 1 = no behavior change)
- PluginProtocol surfaces the method abstract (forward-compat for any
  paradigm plugin to override)

Phase 1b (cl_brief.py) and Phase 1c (CLI flag) are separate PRs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))

import state_machine as sm  # noqa: E402
import state_executor as se  # noqa: E402


# ──────────────────────────────────────────── state_executor mapping


def test_state_executor_maps_await_cann_learn_to_aog_cann_learner():
    """next_agent('await_cann_learn') must resolve to 'aog-cann-learner'.
    Without this, orchestrator emits 'no agent for state await_cann_learn'
    on the first auto-trigger event (Phase 2 / runtime gate flips True).
    """
    assert se.next_agent("await_cann_learn") == "aog-cann-learner"


def test_state_executor_await_cann_learn_is_not_pause_or_terminal():
    """await_cann_learn is a regular spawn state, not pause/terminal."""
    assert se.is_pause("await_cann_learn") is False
    assert se.is_terminal("await_cann_learn") is False


# ──────────────────────────────────────────── YAML FSM state declaration


def test_yaml_declares_await_cann_learn_state():
    """opgen_state_machine.yaml must declare the new await_cann_learn state
    with the design-doc-specified config: iter_cap=1, iter_counter=cann_learn,
    agent=aog-cann-learner.
    """
    sm_dict = sm.load_state_machine()
    awl = next(
        (s for s in sm_dict["phase_o4_states"] if s["id"] == "await_cann_learn"),
        None,
    )
    assert awl is not None, "await_cann_learn state missing from YAML"
    assert awl["agent"] == "aog-cann-learner"
    assert awl["iter_cap"] == 1
    assert awl["iter_counter"] == "cann_learn"


def test_yaml_await_cann_learn_exit_transitions_complete():
    """await_cann_learn exits MUST cover the 3 design-doc handoffs +
    catch-all: cann_learn_done → await_researcher; cann_learn_empty/blocked
    → finalize; always-true → finalize.
    """
    sm_dict = sm.load_state_machine()
    awl = next(s for s in sm_dict["phase_o4_states"] if s["id"] == "await_cann_learn")
    transitions = awl.get("exit_transitions", [])

    # Collect (handoff_match, goto) pairs from simple-condition transitions
    pairs = []
    for t in transitions:
        cond = t.get("condition", {})
        if isinstance(cond, dict):
            if "handoff_match" in cond:
                pairs.append((cond["handoff_match"], t["goto"]))
            elif cond.get("always") is True:
                pairs.append(("__always__", t["goto"]))

    handoff_to_goto = dict(pairs)
    assert handoff_to_goto.get("→ orchestrator: cann_learn_done") == "await_researcher"
    assert handoff_to_goto.get("→ orchestrator: cann_learn_empty") == "finalize"
    assert handoff_to_goto.get("→ orchestrator: cann_learn_blocked") == "finalize"
    assert handoff_to_goto.get("__always__") == "finalize"


# ──────────────────────────────────────────── await_researcher route-in


def test_yaml_await_researcher_routes_research_blocked_to_cann_learn_when_gated():
    """await_researcher MUST have a new exit transition that catches
    research_blocked + plugin_method gate + iter_below_cap before the
    existing research_blocked → abort fallback.
    """
    sm_dict = sm.load_state_machine()
    ar_state = next(
        s for s in sm_dict["phase_o4_states"] if s["id"] == "await_researcher"
    )
    transitions = ar_state["exit_transitions"]

    # Find the gated cann_learn transition (composite all_of with research_blocked)
    matched = None
    for idx, t in enumerate(transitions):
        cond = t.get("condition", {})
        if not isinstance(cond, dict):
            continue
        clauses = cond.get("all_of", [])
        has_blocked = any(
            isinstance(c, dict) and c.get("handoff_match") == "→ orchestrator: research_blocked"
            for c in clauses
        )
        has_plugin_gate = any(
            isinstance(c, dict) and c.get("plugin_method") == "should_auto_cann_learn_on_gap"
            for c in clauses
        )
        has_iter_gate = any(
            isinstance(c, dict) and c.get("iter_below_cap") == "cann_learn"
            for c in clauses
        )
        if has_blocked and has_plugin_gate and has_iter_gate:
            matched = (idx, t)
            break

    assert matched is not None, (
        "await_researcher missing research_blocked + plugin_method + iter_below_cap "
        "composite transition to await_cann_learn"
    )
    idx, t = matched
    assert t["goto"] == "await_cann_learn"

    # ORDERING: must come BEFORE the existing research_blocked → abort transition
    abort_idx = None
    for j, t2 in enumerate(transitions):
        c = t2.get("condition", {})
        if (
            isinstance(c, dict)
            and c.get("handoff_match") == "→ orchestrator: research_blocked"
            and t2.get("goto") == "abort"
        ):
            abort_idx = j
            break
    assert abort_idx is not None, "existing research_blocked → abort transition missing"
    assert idx < abort_idx, (
        f"cann_learn gate transition (idx={idx}) must come BEFORE research_blocked → abort "
        f"(idx={abort_idx}); otherwise the gate is unreachable"
    )


def test_yaml_await_researcher_routes_research_partial_to_cann_learn_when_gated():
    """Mirror of above for research_partial path."""
    sm_dict = sm.load_state_machine()
    ar_state = next(
        s for s in sm_dict["phase_o4_states"] if s["id"] == "await_researcher"
    )
    transitions = ar_state["exit_transitions"]

    matched = None
    for idx, t in enumerate(transitions):
        cond = t.get("condition", {})
        if not isinstance(cond, dict):
            continue
        clauses = cond.get("all_of", [])
        has_partial = any(
            isinstance(c, dict) and c.get("handoff_match") == "→ orchestrator: research_partial"
            for c in clauses
        )
        has_plugin_gate = any(
            isinstance(c, dict) and c.get("plugin_method") == "should_auto_cann_learn_on_gap"
            for c in clauses
        )
        has_iter_gate = any(
            isinstance(c, dict) and c.get("iter_below_cap") == "cann_learn"
            for c in clauses
        )
        if has_partial and has_plugin_gate and has_iter_gate:
            matched = (idx, t)
            break

    assert matched is not None, (
        "await_researcher missing research_partial + plugin_method + iter_below_cap "
        "composite transition to await_cann_learn"
    )
    assert matched[1]["goto"] == "await_cann_learn"


# ──────────────────────────────────────────── plugin gate


def test_baseplugin_should_auto_cann_learn_on_gap_default_false():
    """BasePlugin's default MUST be False — Phase 1 is no behavior change."""
    from plugins.base import BasePlugin
    p = BasePlugin()
    assert p.should_auto_cann_learn_on_gap("FA", "L4", "research_blocked") is False
    assert p.should_auto_cann_learn_on_gap("UNKNOWN", "unknown", "port_a3_to_a5") is False


def test_backward_plugin_inherits_default_false():
    """Backward generation must not enter migration-only prior-art recovery."""
    from plugins.backward import BackwardPlugin

    for plug in [BackwardPlugin()]:
        assert plug.should_auto_cann_learn_on_gap("FA", "L4", "test") is False, (
            f"{plug.__class__.__name__} should retain the neutral False default"
        )


def test_pluginprotocol_surfaces_should_auto_cann_learn_on_gap():
    """The method MUST be on PluginProtocol's abstract surface so any future
    paradigm-native plugin must implement it (or inherit BasePlugin's False).
    """
    from plugins.base import PluginProtocol
    assert hasattr(PluginProtocol, "should_auto_cann_learn_on_gap")


# ──────────────────────────────────────────── end-to-end gate evaluation


def test_eval_condition_plugin_method_routes_cann_learn_gate_when_optin(tmp_path):
    """End-to-end: composite YAML condition (handoff_match + iter_below_cap +
    plugin_method) evaluates True when a hypothetical opt-in plugin returns True.
    """
    from plugins.base import BasePlugin

    class _OptInPlugin(BasePlugin):
        name = "test_optin_for_cann_learn"

        def should_auto_cann_learn_on_gap(self, op_class, op_complexity, workspace_mode):
            return True

    ws = tmp_path / "ws_test_cann_learn"
    ws.mkdir()

    snap = sm.snapshot(ws)
    snap["op_taxonomy"] = {"class": "FUSED SOFTMAX TRANSCENDENTAL", "complexity": "L4"}

    ctx = {
        "handoff": "→ orchestrator: research_blocked",
        "snapshot": snap,
        "iter_counts": {"cann_learn": 0},  # below cap=1
        "ws": ws,
        "sm": sm.load_state_machine(),
        "plugin": _OptInPlugin(),
    }

    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: research_blocked"},
            {"iter_below_cap": "cann_learn"},
            {"plugin_method": "should_auto_cann_learn_on_gap"},
        ]
    }
    assert sm.eval_condition(cond, ctx) is True


def test_eval_condition_plugin_method_skips_cann_learn_when_default_optout(tmp_path):
    """With BasePlugin default (False), gate must NOT fire even when handoff
    + iter conditions match. Preserves Phase 1 no-behavior-change guarantee.
    """
    from plugins.base import BasePlugin

    class _DefaultPlugin(BasePlugin):
        name = "test_default_optout"
        # Inherits should_auto_cann_learn_on_gap default = False

    ws = tmp_path / "ws_test_default"
    ws.mkdir()
    snap = sm.snapshot(ws)
    snap["op_taxonomy"] = {"class": "FUSED SOFTMAX", "complexity": "L4"}

    ctx = {
        "handoff": "→ orchestrator: research_blocked",
        "snapshot": snap,
        "iter_counts": {"cann_learn": 0},
        "ws": ws,
        "sm": sm.load_state_machine(),
        "plugin": _DefaultPlugin(),
    }
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: research_blocked"},
            {"iter_below_cap": "cann_learn"},
            {"plugin_method": "should_auto_cann_learn_on_gap"},
        ]
    }
    assert sm.eval_condition(cond, ctx) is False


def test_eval_condition_plugin_method_skips_when_iter_cap_hit(tmp_path):
    """iter_below_cap=cann_learn returns False when iter_counts.cann_learn >= cap.
    Prevents loop where cann_learn fires multiple times for same op-gen run.
    """
    from plugins.base import BasePlugin

    class _OptInPlugin(BasePlugin):
        name = "test_optin_iter_cap"

        def should_auto_cann_learn_on_gap(self, op_class, op_complexity, workspace_mode):
            return True

    ws = tmp_path / "ws_test_iter_cap"
    ws.mkdir()
    snap = sm.snapshot(ws)
    snap["op_taxonomy"] = {"class": "FUSED SOFTMAX", "complexity": "L4"}

    ctx = {
        "handoff": "→ orchestrator: research_blocked",
        "snapshot": snap,
        "iter_counts": {"cann_learn": 1},  # at cap (cap=1 per design)
        "ws": ws,
        "sm": sm.load_state_machine(),
        "plugin": _OptInPlugin(),
    }
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: research_blocked"},
            {"iter_below_cap": "cann_learn"},
            {"plugin_method": "should_auto_cann_learn_on_gap"},
        ]
    }
    assert sm.eval_condition(cond, ctx) is False
