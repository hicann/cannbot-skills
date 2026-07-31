#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Characterization (truth-table) lock for `state_machine.eval_condition`.

Authored BEFORE decomposing the eval_condition god-function (2026-07-05). eval_condition
is the FSM condition evaluator — a per-`kind` dispatch over ~26 condition primitives. This
enumerates a representative (cond, ctx) input for EACH kind and pins the CURRENT result, so a
behavior-neutral decomposition is verifiable per-branch. The 17 kinds heavily exercised by the
existing suite (yaml_primitives/perf_gate/plugin_method/user_decision/handoff/il_gate/…) are
covered there; this file locks the recursion + parsing contract + the 9 kinds thin on coverage
(analysis_md/det_report/fused_analysis/probe_classification/trajectory/verification_*).
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

# workflow/ dir on path (matches sibling tests, e.g. test_gapc_handoff_markdown_bold.py) —
# there is no conftest that injects it, so do it here before importing state_machine.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import state_machine as sm  # noqa: E402


def _ctx(**kw):
    d = pathlib.Path(tempfile.mkdtemp())
    base = {"ws": d, "handoff": "", "snapshot": {}, "iter_counts": {}, "sm": {}}
    base.update(kw)
    return base


# --- recursion + parsing contract (kind-agnostic) ---
def test_eval_condition_primitives_contract():
    assert sm.eval_condition(True, _ctx()) is True
    assert sm.eval_condition(False, _ctx()) is False
    assert sm.eval_condition("not-a-dict", _ctx()) is False          # non-dict → False
    assert sm.eval_condition({"all_of": [True, True]}, _ctx()) is True
    assert sm.eval_condition({"all_of": [True, False]}, _ctx()) is False
    assert sm.eval_condition({"any_of": [False, True]}, _ctx()) is True
    assert sm.eval_condition({"any_of": [False, False]}, _ctx()) is False
    # nested recursion
    assert sm.eval_condition({"all_of": [{"any_of": [False, True]}, True]}, _ctx()) is True


# (kind, cond, ctx-kwargs, expected)  — captured pre-refactor
_TRUTH_TABLE = [
    ("always", {"always": True}, {}, True),
    ("handoff_match", {"handoff_match": "done"}, {"handoff": "done — ok"}, True),
    ("handoff_match_miss", {"handoff_match": "done"}, {"handoff": "nope"}, False),
    ("handoff_contains", {"handoff_contains": "PLATEAU"}, {"handoff": "done — KO_PLATEAU"}, True),
    ("iter_below_cap", {"iter_below_cap": {"counter": "kw", "cap": 5}}, {"iter_counts": {"kw": 2}}, True),
    ("path_exists_miss", {"path_exists": "nope.txt"}, {}, False),
    ("file_absent", {"file_absent": "nope.txt"}, {}, True),
    ("plugin_method_no_plugin", {"plugin_method": "nonexistent_method"}, {}, False),
    # --- the 9 thin-coverage kinds (captured) ---
    ("analysis_md_contains_any", {"analysis_md_contains_any": ["tuning"]}, {}, False),
    ("det_report_decision_in", {"det_report_decision_in": ["retry"]}, {}, False),
    ("fused_analysis_contains_tuning_candidate", {"fused_analysis_contains_tuning_candidate": True}, {}, False),
    ("fused_analysis_lacks_strategy_citation", {"fused_analysis_lacks_strategy_citation": True}, {}, True),
    ("probe_classification_in", {"probe_classification_in": ["kernel_bug"]}, {}, False),
    ("trajectory_mad_loose", {"trajectory_mad_loose": 0.1}, {}, False),
    ("trajectory_mad_tight", {"trajectory_mad_tight": 0.01}, {}, False),
    ("verification_det_policy_satisfied_miss", {"verification_det_policy_satisfied": True}, {}, False),
    ("verification_precision_status_in_miss", {"verification_precision_status_in": ["PASS"]}, {}, False),
    # --- True paths with state ---
    ("verification_precision_status_in_hit", {"verification_precision_status_in": ["PASS"]},
     {"snapshot": {"verification": {"precision": {"status": "PASS"}}}}, True),
    ("verification_det_policy_satisfied_hit", {"verification_det_policy_satisfied": True},
     {"snapshot": {"verification": {"determinism": {"policy_satisfied": True}}}}, True),
    ("verification_precision_status_not_in", {"verification_precision_status_not_in": ["FAIL"]},
     {"snapshot": {"verification": {"precision": {"status": "PASS"}}}}, True),
]


@pytest.mark.parametrize("label,cond,ctxkw,expected", _TRUTH_TABLE, ids=[c[0] for c in _TRUTH_TABLE])
def test_eval_condition_truth_table(label, cond, ctxkw, expected):
    assert sm.eval_condition(cond, _ctx(**ctxkw)) is expected, (
        f"eval_condition changed for {label}: {cond} → {sm.eval_condition(cond, _ctx(**ctxkw))!r} != {expected!r}"
    )
