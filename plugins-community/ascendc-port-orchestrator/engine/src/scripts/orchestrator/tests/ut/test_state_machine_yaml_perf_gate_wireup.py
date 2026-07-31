# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for Phase B4 YAML wire-up: opgen_state_machine.yaml gates 2
await_worker → await_optimizer transitions with perf_gate_profile_allows: ko_escalation.

Verifies end-to-end routing changes under PRECISION_ONLY profile
(--perf-threshold=0):

  Implicit V3.8.4 anti-bypass:
    done + perf<thresh + iter<cap (no profile)        → await_optimizer
    done + perf<thresh + iter<cap + PRECISION_ONLY    → falls through to finalize

  Explicit @aog-kernel-optimizer handoff:
    @aog-kernel-optimizer + iter<cap (DEFAULT)         → await_optimizer
    @aog-kernel-optimizer + PRECISION_ONLY             → falls through to finalize
"""
from __future__ import annotations

import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest
import yaml

ROOT = _reorg_paths.REPO_ROOT
YAML_PATH = ROOT .parent / "workflows" / "opgen_state_machine.yaml"

sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

from state_machine import eval_condition  # noqa: E402
from perf_gate import write_profile_marker  # noqa: E402


@pytest.fixture(scope="module")
def sm():
    return yaml.safe_load(YAML_PATH.read_text())


def _find_state(sm: dict, state_id: str) -> dict | None:
    for s in sm.get("phase_o4_states", []):
        if s.get("id") == state_id:
            return s
    return None


def _ctx(ws: Path, *, handoff: str = "", vj: dict | None = None,
         iter_counts: dict | None = None, current_state: str = "await_worker"):
    snap = {"verification": vj or {}}
    return {
        "handoff": handoff,
        "snapshot": snap,
        "iter_counts": iter_counts or {},
        "ws": ws,
        "sm": {"phase_o4_states": [
            {"iter_counter": "optimizer", "iter_cap": 3},
        ]},
        "plugin": None,
        "current_state": current_state,
    }


# ---- Structural: confirm gate is present in YAML ----


def test_implicit_v3_8_4_anti_bypass_has_ko_gate(sm):
    """The 'done + perf<thresh + iter<cap → await_optimizer' transition
    in await_worker MUST include perf_gate_profile_allows: ko_escalation
    (added Phase B4).
    """
    s = _find_state(sm, "await_worker")
    assert s is not None
    found = False
    for trans in s.get("exit_transitions", []):
        cond = trans.get("condition", {})
        all_of = cond.get("all_of", [])
        primitives = [list(p.keys())[0] for p in all_of if isinstance(p, dict)]
        if ("handoff_match" in primitives
                and "verification_perf_below_threshold" in primitives
                and trans.get("goto") == "await_optimizer"):
            assert "perf_gate_profile_allows" in primitives, (
                "V3.8.4 anti-bypass transition missing perf_gate_profile_allows "
                "gate added by Phase B4"
            )
            # Verify the value is ko_escalation specifically
            for p in all_of:
                if isinstance(p, dict) and "perf_gate_profile_allows" in p:
                    assert p["perf_gate_profile_allows"] == "ko_escalation"
            found = True
            break
    assert found, "implicit V3.8.4 anti-bypass transition not located in YAML"


def test_explicit_ko_handoff_has_ko_gate(sm):
    """The '@aog-kernel-optimizer + iter<cap → await_optimizer' transition
    MUST include perf_gate_profile_allows: ko_escalation.
    """
    s = _find_state(sm, "await_worker")
    found = False
    for trans in s.get("exit_transitions", []):
        cond = trans.get("condition", {})
        all_of = cond.get("all_of", [])
        # Looking for the specific explicit-handoff transition (no
        # verification_perf check — just handoff + iter_cap)
        primitives = [list(p.keys())[0] for p in all_of if isinstance(p, dict)]
        if (primitives == ["handoff_match", "iter_below_cap",
                           "perf_gate_profile_allows"]
                and trans.get("goto") == "await_optimizer"):
            for p in all_of:
                if isinstance(p, dict) and "handoff_match" in p:
                    assert p["handoff_match"] == "@aog-kernel-optimizer"
                if isinstance(p, dict) and "perf_gate_profile_allows" in p:
                    assert p["perf_gate_profile_allows"] == "ko_escalation"
            found = True
            break
    assert found, (
        "explicit @aog-kernel-optimizer handoff transition missing "
        "perf_gate_profile_allows gate added by Phase B4"
    )


def test_fallback_explicit_ko_to_finalize_still_present(sm):
    """The fallback '@aog-kernel-optimizer → finalize' (cap exhausted OR
    profile blocks) must still exist after Phase B4 — it's the escape
    route when the gated transition above suppresses.
    """
    s = _find_state(sm, "await_worker")
    found = False
    for trans in s.get("exit_transitions", []):
        cond = trans.get("condition", {})
        all_of = cond.get("all_of", [])
        primitives = [list(p.keys())[0] for p in all_of if isinstance(p, dict)]
        # Fallback has just handoff_match alone (no iter_below_cap, no gate)
        if (primitives == ["handoff_match"] and trans.get("goto") == "finalize"):
            for p in all_of:
                if isinstance(p, dict) and "handoff_match" in p:
                    if p["handoff_match"] == "@aog-kernel-optimizer":
                        found = True
                        break
        if found:
            break
    assert found, "fallback @aog-kernel-optimizer → finalize transition missing"


# ---- Behavioral: simulate eval_condition under both profiles ----


def test_implicit_ko_route_fires_under_default(tmp_path, sm):
    """DEFAULT profile + done + perf<thresh + iter<cap → await_optimizer fires."""
    s = _find_state(sm, "await_worker")
    # Find the V3.8.4 transition
    trans = None
    for t in s["exit_transitions"]:
        cond = t.get("condition", {})
        primitives = [list(p.keys())[0] for p in cond.get("all_of", [])
                      if isinstance(p, dict)]
        if ("verification_perf_below_threshold" in primitives
                and "perf_gate_profile_allows" in primitives
                and t.get("goto") == "await_optimizer"):
            trans = t
            break
    assert trans is not None
    vj = {"performance": {"ratio": 0.3, "threshold": 0.6}}
    ctx = _ctx(tmp_path, handoff="→ orchestrator: done",
               vj=vj, iter_counts={"optimizer": 0})
    assert eval_condition(trans["condition"], ctx) is True


def test_implicit_ko_route_suppressed_under_precision_only(tmp_path, sm):
    """PRECISION_ONLY profile + done + perf<thresh + iter<cap → transition
    suppressed (profile.allow_ko_escalation=False).
    """
    write_profile_marker(tmp_path, perf_threshold=0)
    s = _find_state(sm, "await_worker")
    trans = None
    for t in s["exit_transitions"]:
        cond = t.get("condition", {})
        primitives = [list(p.keys())[0] for p in cond.get("all_of", [])
                      if isinstance(p, dict)]
        if ("verification_perf_below_threshold" in primitives
                and "perf_gate_profile_allows" in primitives
                and t.get("goto") == "await_optimizer"):
            trans = t
            break
    assert trans is not None
    # Even with a "bad" ratio in verification (which under DEFAULT would
    # escalate), PRECISION_ONLY suppresses the transition.
    vj = {"performance": {"ratio": 0.3, "threshold": 0.6}}
    ctx = _ctx(tmp_path, handoff="→ orchestrator: done",
               vj=vj, iter_counts={"optimizer": 0})
    assert eval_condition(trans["condition"], ctx) is False


def test_explicit_ko_handoff_fires_under_default(tmp_path, sm):
    s = _find_state(sm, "await_worker")
    trans = None
    for t in s["exit_transitions"]:
        cond = t.get("condition", {})
        primitives = [list(p.keys())[0] for p in cond.get("all_of", [])
                      if isinstance(p, dict)]
        if (primitives == ["handoff_match", "iter_below_cap",
                           "perf_gate_profile_allows"]
                and t.get("goto") == "await_optimizer"):
            trans = t
            break
    assert trans is not None
    ctx = _ctx(tmp_path, handoff="@aog-kernel-optimizer",
               iter_counts={"optimizer": 0})
    assert eval_condition(trans["condition"], ctx) is True


def test_explicit_ko_handoff_suppressed_under_precision_only(tmp_path, sm):
    """Worker insists on @aog-kernel-optimizer handoff — under PRECISION_ONLY
    the gated transition suppresses; the fallback (no iter_cap, no profile gate)
    catches it and routes to finalize.
    """
    write_profile_marker(tmp_path, perf_threshold=0)
    s = _find_state(sm, "await_worker")
    trans = None
    for t in s["exit_transitions"]:
        cond = t.get("condition", {})
        primitives = [list(p.keys())[0] for p in cond.get("all_of", [])
                      if isinstance(p, dict)]
        if (primitives == ["handoff_match", "iter_below_cap",
                           "perf_gate_profile_allows"]
                and t.get("goto") == "await_optimizer"):
            trans = t
            break
    assert trans is not None
    ctx = _ctx(tmp_path, handoff="@aog-kernel-optimizer",
               iter_counts={"optimizer": 0})
    # Gated transition does NOT fire
    assert eval_condition(trans["condition"], ctx) is False
