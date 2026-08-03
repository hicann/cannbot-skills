# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the optimizer plateau handoff regression.

The KO_PERF_PLATEAU transition in await_optimizer must fire on the canonical
arrow handoff `→ orchestrator: done — KO_PERF_PLATEAU`.

The bug (caught 2026-06-03, back-agent GroupNorm/Sum archives blocked; 5_Cumsum
origin): the YAML used `handoff_match: "KO_PERF_PLATEAU"`, but handoff_match is
PREFIX-only (`handoff.startswith(arg)`). The verdict token rides as a SUFFIX of
the canonical arrow form, so the condition silently never fired — the plateau
verdict fell through to `iter_below_cap: optimizer → await_optimizer`, respawning
ko-4/ko-5 redundantly and never letting the op reach finalize.

Fix: a new `handoff_contains` substring primitive (`arg in handoff`), with the
YAML transition pointing at it.

This test drives the REAL YAML transition through the REAL eval_condition — not a
hand-authored condition — so a future revert of either the primitive or the YAML
re-breaks it (guards against the isolation-test dead-no-op trap).
"""
from __future__ import annotations

import sys

import pytest
import yaml

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

ROOT = _reorg_paths.REPO_ROOT
YAML_PATH = ROOT .parent / "workflows" / "opgen_state_machine.yaml"

sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))
from state_machine import eval_condition  # noqa: E402

# The exact canonical arrow handoff an optimizer emits on a proven plateau.
# extract_canonical_handoff keeps it as-is ("done" is a valid arrow keyword),
# so the verdict token "KO_PERF_PLATEAU" is a SUFFIX, never a prefix.
PLATEAU_HANDOFF = "→ orchestrator: done — KO_PERF_PLATEAU"


@pytest.fixture(scope="module")
def sm():
    return yaml.safe_load(YAML_PATH.read_text())


def _find_state(sm: dict, state_id: str) -> dict | None:
    for s in sm.get("phase_o4_states", []):
        if s.get("id") == state_id:
            return s
    return None


def _find_plateau_transition(sm: dict) -> dict | None:
    s = _find_state(sm, "await_optimizer")
    assert s is not None, "await_optimizer state missing from YAML"
    for t in s.get("exit_transitions", []):
        if t.get("goto") != "finalize":
            continue
        for p in t.get("condition", {}).get("all_of", []):
            if isinstance(p, dict) and p.get("handoff_contains") == "KO_PERF_PLATEAU":
                return t
    return None


# ---- Primitive-level unit ----


def test_handoff_contains_is_substring():
    """handoff_contains matches a token anywhere in the line (substring)."""
    assert eval_condition({"handoff_contains": "KO_PERF_PLATEAU"},
                          {"handoff": PLATEAU_HANDOFF}) is True
    assert eval_condition({"handoff_contains": "KO_PERF_PLATEAU"},
                          {"handoff": "→ orchestrator: done — 8/8"}) is False


def test_handoff_match_stays_prefix_only():
    """Keep handoff_match prefix-only.

    A suffix token must never match this condition.
    """
    assert eval_condition({"handoff_match": "KO_PERF_PLATEAU"},
                          {"handoff": PLATEAU_HANDOFF}) is False
    assert eval_condition({"handoff_match": "→ orchestrator: done"},
                          {"handoff": PLATEAU_HANDOFF}) is True


# ---- Real YAML transition through real evaluator ----


def test_yaml_plateau_transition_uses_handoff_contains(sm):
    """Use handoff_contains for the plateau-to-finalize transition.

    A prefix-only handoff_match condition cannot detect the plateau suffix.
    """
    t = _find_plateau_transition(sm)
    assert t is not None, (
        "KO_PERF_PLATEAU plateau-accept transition not found with "
        "handoff_contains in await_optimizer — did it revert to handoff_match?"
    )


def test_yaml_plateau_condition_fires_on_canonical_handoff(sm):
    """Evaluate the located YAML condition against the canonical handoff.

    The condition must fire without redundant optimizer respawns.
    """
    t = _find_plateau_transition(sm)
    assert t is not None
    ctx = {"handoff": PLATEAU_HANDOFF, "snapshot": {}, "iter_counts": {},
           "ws": None, "sm": {}, "plugin": None}
    assert eval_condition(t["condition"], ctx) is True
