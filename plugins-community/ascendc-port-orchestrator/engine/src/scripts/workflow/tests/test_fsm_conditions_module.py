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

"""Module-boundary lock for the fsm_conditions extraction (DEBT-201, 2026-07-06).

`eval_condition` + its four private helpers were extracted VERBATIM from
state_machine.py into fsm_conditions.py to keep state_machine.py <1000 lines.
This test guards the SPLIT itself (the truth-table behaviour is
characterization-locked by test_eval_condition_characterization.py, which stays
green pre AND post):

  1. fsm_conditions is independently importable and exposes the public API.
  2. state_machine re-exports the SAME function objects (identity), so every
     caller of `sm.eval_condition` / `from state_machine import eval_condition`
     / `sm._parse_worker_signal` keeps hitting the moved implementation.
  3. Leaf-invariant: fsm_conditions must NOT import state_machine (no cycle).
"""
from __future__ import annotations

import ast
import pathlib
import sys
import tempfile

# workflow/ dir on path (matches sibling tests; no conftest injects it).
_WF = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WF))

import fsm_conditions  # noqa: E402
import state_machine as sm  # noqa: E402


def _ctx(**kw):
    d = pathlib.Path(tempfile.mkdtemp())
    base = {"ws": d, "handoff": "", "snapshot": {}, "iter_counts": {}, "sm": {}}
    base.update(kw)
    return base


def test_fsm_conditions_public_api_importable():
    assert callable(fsm_conditions.eval_condition)
    for h in ("_resolved_precision_status", "_perf_ratio_and_threshold",
              "_trajectory_stats", "_parse_worker_signal"):
        assert callable(getattr(fsm_conditions, h)), h


def test_state_machine_reexports_same_objects():
    """Identity, not just presence — the re-import shim must forward the moved
    implementation so behaviour is byte-for-byte the pre-split code.
    """
    assert sm.eval_condition is fsm_conditions.eval_condition
    assert getattr(sm, '_parse_worker_signal') is getattr(fsm_conditions, '_parse_worker_signal')
    assert getattr(sm, '_resolved_precision_status') is getattr(fsm_conditions, '_resolved_precision_status')
    assert getattr(sm, '_perf_ratio_and_threshold') is getattr(fsm_conditions, '_perf_ratio_and_threshold')
    assert getattr(sm, '_trajectory_stats') is getattr(fsm_conditions, '_trajectory_stats')


def test_eval_condition_works_via_direct_module_import():
    ec = fsm_conditions.eval_condition
    assert ec(True, _ctx()) is True
    assert ec(False, _ctx()) is False
    assert ec("not-a-dict", _ctx()) is False
    assert ec({"all_of": [True, True]}, _ctx()) is True
    assert ec({"all_of": [True, False]}, _ctx()) is False
    assert ec({"any_of": [False, True]}, _ctx()) is True
    assert ec({"kind": "no_such_primitive_xyz"}, _ctx()) is False


def test_parse_worker_signal_contract():
    ps = getattr(fsm_conditions, '_parse_worker_signal')
    assert ps("→ orchestrator: done") == "done"
    assert ps("→ orchestrator: **abort** — build failed") == "abort"
    assert ps("→ aog-kernel-worker: tiling 5*8 in progress") == "unknown"


def test_fsm_conditions_is_a_leaf_no_cycle():
    """fsm_conditions must not import state_machine (edge is one-way)."""
    tree = ast.parse((_WF / "fsm_conditions.py").read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module)
    assert "state_machine" not in imported
