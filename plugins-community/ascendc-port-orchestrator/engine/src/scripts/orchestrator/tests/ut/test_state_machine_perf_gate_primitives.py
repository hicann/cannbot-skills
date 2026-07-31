# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for the 3 PerfGateProfile YAML primitives in state_machine._eval_condition.

Phase B3 of PERF_GATE_PROFILE_DESIGN_2026_05_20 §8. Verifies the typed
primitives bridge YAML transitions to PerfGateProfile fields:

  perf_gate_profile_allows: <name>   → profile.allow_<name>
  perf_gate_profile_requires: <name> → profile.require_<name>
  perf_gate_profile_measures: <name> → profile.measure_<name> OR include_<name>

Also covers `escalation_overrides[current_state]` auto-switch — IL chain
states transparently get PRECISION_ONLY without YAML referencing structural-rewrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

from state_machine import eval_condition  # noqa: E402
from perf_gate import write_profile_marker  # noqa: E402


def _ctx(ws: Path, current_state: str | None = None):
    """Minimal ctx for primitive testing — only needs ws + current_state."""
    return {
        "handoff": "",
        "snapshot": {},
        "iter_counts": {},
        "ws": ws,
        "sm": {},
        "plugin": None,
        "current_state": current_state,
    }


# ---- perf_gate_profile_allows ----


def test_default_profile_allows_ko_escalation(tmp_path):
    """No marker → DEFAULT → allow_ko_escalation=True."""
    cond = {"perf_gate_profile_allows": "ko_escalation"}
    assert eval_condition(cond, _ctx(tmp_path)) is True


def test_precision_only_blocks_ko_escalation(tmp_path):
    """--perf-threshold=0 marker → PRECISION_ONLY → allow_ko_escalation=False."""
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {"perf_gate_profile_allows": "ko_escalation"}
    assert eval_condition(cond, _ctx(tmp_path)) is False


def test_precision_only_blocks_ar_escalation(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {"perf_gate_profile_allows": "ar_escalation"}
    assert eval_condition(cond, _ctx(tmp_path)) is False


def test_precision_only_blocks_fo_escalation(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {"perf_gate_profile_allows": "fo_escalation"}
    assert eval_condition(cond, _ctx(tmp_path)) is False


def test_hero_strict_allows_all_escalations(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0.9)
    for name in ("ko_escalation", "ar_escalation", "fo_escalation"):
        cond = {"perf_gate_profile_allows": name}
        assert eval_condition(cond, _ctx(tmp_path)) is True, name


# ---- perf_gate_profile_requires ----


def test_default_requires_ratio_in_verification(tmp_path):
    cond = {"perf_gate_profile_requires": "ratio_in_verification"}
    assert eval_condition(cond, _ctx(tmp_path)) is True


def test_precision_only_drops_ratio_requirement(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {"perf_gate_profile_requires": "ratio_in_verification"}
    assert eval_condition(cond, _ctx(tmp_path)) is False


def test_precision_only_drops_perf_artifacts_requirement(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {"perf_gate_profile_requires": "perf_artifacts"}
    assert eval_condition(cond, _ctx(tmp_path)) is False


# ---- perf_gate_profile_measures ----


def test_default_measures_reference_perf(tmp_path):
    """profile.measure_reference_perf=True under DEFAULT."""
    cond = {"perf_gate_profile_measures": "reference_perf"}
    assert eval_condition(cond, _ctx(tmp_path)) is True


def test_precision_only_skips_reference_perf_measurement(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {"perf_gate_profile_measures": "reference_perf"}
    assert eval_condition(cond, _ctx(tmp_path)) is False


def test_measures_falls_back_to_include_prefix(tmp_path):
    """When measure_<name> doesn't exist, fall back to include_<name>.

    `perf_in_brief` only exists as `include_perf_in_brief`, not
    `measure_perf_in_brief`. The primitive should still resolve it.
    """
    cond = {"perf_gate_profile_measures": "perf_in_brief"}
    # DEFAULT → include_perf_in_brief=True
    assert eval_condition(cond, _ctx(tmp_path)) is True
    # PRECISION_ONLY → include_perf_in_brief=False
    write_profile_marker(tmp_path, perf_threshold=0)
    assert eval_condition(cond, _ctx(tmp_path)) is False


# ---- escalation_overrides[current_state] auto-switch (generic hook, empty) ----


def test_default_has_no_per_state_override(tmp_path):
    """DEFAULT carries empty escalation_overrides — no per-state auto-switch,
    so ko escalation stays allowed at every state.
    """
    cond = {"perf_gate_profile_allows": "ko_escalation"}
    assert eval_condition(cond, _ctx(tmp_path, current_state="await_worker")) is True
    assert eval_condition(cond, _ctx(tmp_path, current_state="await_probe")) is True


def test_hero_strict_no_auto_switch(tmp_path):
    """HERO_OP_STRICT has empty escalation_overrides — no per-state auto-switch.
    User explicitly chose strict; we don't second-guess.
    """
    write_profile_marker(tmp_path, perf_threshold=0.9)
    cond = {"perf_gate_profile_allows": "ko_escalation"}
    assert eval_condition(
        cond, _ctx(tmp_path, current_state="await_probe")
    ) is True


# ---- safety defaults ----


def test_unknown_field_returns_true_safe_default(tmp_path):
    """A primitive with no matching profile field returns True (safe default).

    Preserves legacy behavior — adding a primitive that references a
    non-existent field shouldn't accidentally block transitions.
    """
    cond = {"perf_gate_profile_allows": "made_up_escalation_xyz"}
    assert eval_condition(cond, _ctx(tmp_path)) is True


def test_no_workspace_returns_true_safe_default():
    """When ws=None (legacy callers / synthetic tests), primitive fails open."""
    cond = {"perf_gate_profile_allows": "ko_escalation"}
    ctx = {
        "handoff": "", "snapshot": {}, "iter_counts": {},
        "ws": None, "sm": {}, "plugin": None, "current_state": None,
    }
    assert eval_condition(cond, ctx) is True


# ---- composition with all_of (real YAML pattern) ----


def test_all_of_combination_blocks_when_profile_disallows(tmp_path):
    """Real YAML pattern from design doc §8:
       all_of: [handoff_match: "..., perf_gate_profile_allows: ko_escalation, ...]
    """
    write_profile_marker(tmp_path, perf_threshold=0)
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: done"},
            {"perf_gate_profile_allows": "ko_escalation"},
        ]
    }
    ctx = _ctx(tmp_path)
    ctx["handoff"] = "→ orchestrator: done"
    # PRECISION_ONLY → ko blocked even though handoff matches
    assert eval_condition(cond, ctx) is False


def test_all_of_combination_passes_under_default(tmp_path):
    """Same all_of, DEFAULT profile — both primitives pass → True."""
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: done"},
            {"perf_gate_profile_allows": "ko_escalation"},
        ]
    }
    ctx = _ctx(tmp_path)
    ctx["handoff"] = "→ orchestrator: done"
    assert eval_condition(cond, ctx) is True
