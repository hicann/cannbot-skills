# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for previously-unimplemented YAML condition primitives (TD #29).

V3.7.11 vendor-strategy escalation declared:
  - perf_ratio_below: <float>
  - optimization_log_lacks_strategy_citation: <bool>

Both fell into eval_condition's unknown-primitive branch (return False with
stderr warning). Day 4 implementation: backed by verification.json
performance.ratio and optimization_log.md grep respectively.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


def _ctx(ws, *, perf_ratio=None, optimization_log_text=None):
    """Build a ctx dict the way eval_condition expects."""
    snap = sm.snapshot(ws)
    if perf_ratio is not None or perf_ratio == 0:
        snap.setdefault("verification", {})
        snap["verification"]["performance"] = {"ratio": perf_ratio}
    if optimization_log_text is not None:
        (ws / "optimization_log.md").write_text(optimization_log_text)
        # snapshot() doesn't preload optimization_log; eval reads it directly
    return {"handoff": "", "snapshot": snap, "iter_counts": {},
            "ws": ws, "sm": {}}


# ---------------------------------------------------------------------------
# file_absent glob — P135.SI Bug B (2026-05-18)
#
# YAML routing rules may use `file_absent: workspace/{op}/*_strategy_inference.md`
# to gate "researcher hasn't run" checks. These tests pin the generic glob
# primitive independently of the supported workflow's canonical filename.
# ---------------------------------------------------------------------------
def test_file_absent_glob_returns_false_when_cann_strategy_present(ws):
    """Glob `*_strategy_inference.md` matches cann_strategy_inference.md → absent=False."""
    (ws / "cann_strategy_inference.md").write_text("# researcher findings")
    ctx = _ctx(ws)
    assert sm.eval_condition(
        {"file_absent": "workspace/{op}/*_strategy_inference.md"}, ctx
    ) is False


def test_file_absent_glob_returns_false_when_alternate_strategy_present(ws):
    """The primitive remains a real glob rather than a fixed-name check."""
    (ws / "alternate_strategy_inference.md").write_text("# researcher findings")
    ctx = _ctx(ws)
    assert sm.eval_condition(
        {"file_absent": "workspace/{op}/*_strategy_inference.md"}, ctx
    ) is False


def test_file_absent_glob_returns_true_when_nothing_matches(ws):
    """No *_strategy_inference.md anywhere → absent=True (researcher hasn't run)."""
    ctx = _ctx(ws)
    assert sm.eval_condition(
        {"file_absent": "workspace/{op}/*_strategy_inference.md"}, ctx
    ) is True


def test_file_absent_glob_returns_false_for_arbitrary_backend(ws):
    """Forward-compat: any future <backend>_strategy_inference.md matches."""
    (ws / "future_strategy_inference.md").write_text("# future backend")
    ctx = _ctx(ws)
    assert sm.eval_condition(
        {"file_absent": "workspace/{op}/*_strategy_inference.md"}, ctx
    ) is False


# ---------------------------------------------------------------------------
# perf_ratio_below
# ---------------------------------------------------------------------------
def test_perf_ratio_below_returns_true_when_below(ws):
    ctx = _ctx(ws, perf_ratio=0.4)
    assert sm.eval_condition({"perf_ratio_below": 0.5}, ctx) is True


def test_perf_ratio_below_returns_false_when_at_threshold(ws):
    ctx = _ctx(ws, perf_ratio=0.5)
    assert sm.eval_condition({"perf_ratio_below": 0.5}, ctx) is False


def test_perf_ratio_below_returns_false_when_above(ws):
    ctx = _ctx(ws, perf_ratio=0.9)
    assert sm.eval_condition({"perf_ratio_below": 0.5}, ctx) is False


def test_perf_ratio_below_returns_false_when_no_perf_data(ws):
    ctx = _ctx(ws)
    assert sm.eval_condition({"perf_ratio_below": 0.5}, ctx) is False


def test_perf_ratio_below_handles_string_arg(ws):
    """YAML parsers may pass numeric thresholds as strings; coerce."""
    ctx = _ctx(ws, perf_ratio=0.4)
    assert sm.eval_condition({"perf_ratio_below": "0.5"}, ctx) is True


def test_perf_ratio_below_invalid_arg_returns_false(ws):
    ctx = _ctx(ws, perf_ratio=0.4)
    assert sm.eval_condition({"perf_ratio_below": "not a number"}, ctx) is False


# ---------------------------------------------------------------------------
# optimization_log_lacks_strategy_citation
# ---------------------------------------------------------------------------
def test_optlog_lacks_when_file_absent(ws):
    """No optimization_log.md → lacks citation = True."""
    ctx = _ctx(ws)
    assert sm.eval_condition(
        {"optimization_log_lacks_strategy_citation": True}, ctx
    ) is True


def test_optlog_lacks_when_no_citation_keywords(ws):
    """Log exists but no citation phrase."""
    ctx = _ctx(ws, optimization_log_text="""\
# Optimization log

iter 1: tried tile size 256, perf 0.4×
iter 2: tried tile size 512, perf 0.45×
plateau identified.
""")
    assert sm.eval_condition(
        {"optimization_log_lacks_strategy_citation": True}, ctx
    ) is True


def test_optlog_present_with_vendor_uses_citation(ws):
    """Log mentions 'vendor uses' → citation present, lacks=False."""
    ctx = _ctx(ws, optimization_log_text="""\
# Optimization log

Per msprof, **vendor uses** MrgSort4 hierarchy with bitonic outer.
Our impl uses BlockReduce + StableSort; this explains the 3.35× gap.
""")
    assert sm.eval_condition(
        {"optimization_log_lacks_strategy_citation": True}, ctx
    ) is False


def test_optlog_cann_strategy_citation(ws):
    ctx = _ctx(ws, optimization_log_text="CANN strategy: radix sort top-down.")
    assert sm.eval_condition(
        {"optimization_log_lacks_strategy_citation": True}, ctx
    ) is False


def test_optlog_case_insensitive(ws):
    ctx = _ctx(ws, optimization_log_text="VENDOR USES Cooley-Tukey FFT.")
    assert sm.eval_condition(
        {"optimization_log_lacks_strategy_citation": True}, ctx
    ) is False


def test_optlog_inverse_arg(ws):
    """arg=False means: assert citation IS present."""
    ctx = _ctx(ws, optimization_log_text="vendor uses MrgSort.")
    assert sm.eval_condition(
        {"optimization_log_lacks_strategy_citation": False}, ctx
    ) is True


# ---------------------------------------------------------------------------
# Integration: V3.7.11 transition firing now actually works
# ---------------------------------------------------------------------------
def test_v3_7_11_transition_fires_when_all_conditions_met(ws):
    """V3.7.11 await_fused_optimizer → await_researcher gate:
    perf_below_threshold + perf_ratio_below 0.5 + iter_below_cap researcher
    + file_absent cann_strategy_inference.md + optimization_log_lacks_*
    All must be True. Verify the gate fires when so.
    """
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL"},
        "performance": {"status": "PASS", "ratio": 0.3, "threshold": 0.6},
    }))
    # No optimization_log.md → lacks citation
    # No cann_strategy_inference.md → file_absent True
    # No researcher iter_count → iter_below_cap True
    sm_yaml = sm.load_state_machine()
    spec = sm.get_state_spec(sm_yaml, "await_optimizer")
    snap_dict = sm.snapshot(ws)
    ctx = {"handoff": "", "snapshot": snap_dict, "iter_counts": {},
           "ws": ws, "sm": sm_yaml}

    # Find the V3.7.11 transition (one with perf_ratio_below)
    v3_7_11_trans = None
    for t in spec.get("exit_transitions", []):
        cond = t.get("condition") or {}
        ao = cond.get("all_of", [])
        if any("perf_ratio_below" in c for c in ao):
            v3_7_11_trans = t
            break
    assert v3_7_11_trans is not None, "V3.7.11 transition disappeared from YAML"

    # Now eval its condition — should be True (all subconditions met)
    assert sm.eval_condition(v3_7_11_trans["condition"], ctx) is True
    assert v3_7_11_trans["goto"] == "await_researcher"
