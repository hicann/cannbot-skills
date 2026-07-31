# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0xx (2026-05-06): pass_count_consistent YAML primitive blocks
done→finalize when worker self-declares PASS with contradictory counts.

Reported by DS agent: 30_NMS scenario worker wrote canonical V3.8.x
verification.json with:
  - top-level precision.status: PASS
  - precision.pass_a: {status: FAIL_EXPECTED, tier1_pass: 0, total: 31}
  - precision.pass_b: {status: PASS, tier1_pass: 31, total: 31}

The await_worker `done → finalize` transition matched on
`handoff_match: → orchestrator: done` alone (status=PASS check
passed because top-level status was set). P0ee schema_norm caught
the contradiction at finalize-time but the transition was already
committed.

Fix: new YAML primitive `pass_count_consistent` calls
schema_norm._check_pass_count_consistency. New transition with
`pass_count_consistent: false + iter_below_cap: probe` routes to
await_probe BEFORE finalize.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
_PROJECT = _reorg_paths.REPO_ROOT
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
sys.path.insert(0, str(_HERE.parent.parent))
import state_machine  # noqa: E402


def _ctx_with(prec: dict, *, ws=None, sm=None):
    return {
        "handoff": "→ orchestrator: done",
        "snapshot": {"verification": {"precision": prec}},
        "iter_counts": {},
        "ws": ws,
        "sm": sm or {},
    }


def test_pass_count_consistent_true_when_all_pass(tmp_path):
    """Genuine PASS: tier1_pass == total > 0 → primitive returns True."""
    ctx = _ctx_with({
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 31, "total": 31},
        "pass_b": {"status": "PASS", "tier1_pass": 31, "total": 31},
    }, ws=tmp_path)
    # primitive expects arg=true (i.e. "is consistent")
    assert state_machine.eval_condition(
        {"pass_count_consistent": True}, ctx) is True


def test_pass_count_consistent_false_for_contradictory_pass(tmp_path):
    """30_NMS scenario: status=PASS but pass_a tier1=0/31."""
    ctx = _ctx_with({
        "status": "PASS",
        "pass_a": {"status": "FAIL_EXPECTED", "tier1_pass": 0, "total": 31},
        "pass_b": {"status": "PASS", "tier1_pass": 31, "total": 31},
    }, ws=tmp_path)
    # arg=true → consistent? NO → returns False (primitive == arg fails)
    assert state_machine.eval_condition(
        {"pass_count_consistent": True}, ctx) is False
    # arg=false → consistent? NO → returns True (primitive == arg matches)
    assert state_machine.eval_condition(
        {"pass_count_consistent": False}, ctx) is True


def test_pass_count_consistent_allows_n_a_pass(tmp_path):
    """Path A (OL-68 case A): pass_a status=N/A, only pass_b counts."""
    ctx = _ctx_with({
        "status": "PASS",
        "pass_a": {"status": "N/A"},
        "pass_b": {"status": "PASS", "tier1_pass": 50, "total": 50},
    }, ws=tmp_path)
    assert state_machine.eval_condition(
        {"pass_count_consistent": True}, ctx) is True


def test_pass_count_consistent_partial_status_with_partial_counts_ok(tmp_path):
    """{status: PARTIAL, tier1_pass: 17/31} is genuinely consistent —
    status text admits partial. The primitive only flags contradiction
    (e.g. status=PASS but counts say 0 passed). For PARTIAL-with-PARTIAL,
    the existing `verification_precision_status_not_in: [PASS, ...]` rule
    in YAML routes done→await_probe before this primitive is reached.
    """
    ctx_partial = _ctx_with({
        "status": "PARTIAL",
        "pass_a": {"status": "PARTIAL", "tier1_pass": 17, "total": 31},
    }, ws=tmp_path)
    assert state_machine.eval_condition(
        {"pass_count_consistent": True}, ctx_partial) is True


def test_pass_count_consistent_handles_legacy_n_pass_n_total(tmp_path):
    """Legacy field names (n_pass / n_total) should also be consistent-checked."""
    ctx = _ctx_with({
        "status": "PASS",
        "pass_a": {"status": "PASS", "n_pass": 0, "n_total": 31},
    }, ws=tmp_path)
    # 0/31 is contradictory regardless of which field name
    assert state_machine.eval_condition(
        {"pass_count_consistent": False}, ctx) is True


def test_yaml_routing_p0aab_await_probe_escalates_on_inconsistent_counts(tmp_path):
    """P0aab (2026-05-06, DS report): probe-closed-loop transition must
    check pass_count_consistent BEFORE the bare status=PASS finalize rule.
    Origin: 30_NMS — probe fixed Pass B (9/9) but Pass A still had 14/31
    FAIL_EXPECTED. State machine routed to finalize despite contradiction.
    Now must escalate to await_researcher when counts contradict.
    """
    import yaml as _yaml
    yaml_path = _PROJECT .parent / "workflows" / "opgen_state_machine.yaml"
    raw = _yaml.safe_load(yaml_path.read_text())
    aw = next(s for s in raw.get("phase_o4_states", [])
              if s.get("id") == "await_probe")
    transitions = aw.get("exit_transitions", [])
    p0aab = None
    bare_probe_done_idx = None
    for i, t in enumerate(transitions):
        cond = t.get("condition", {})
        all_of = cond.get("all_of", [])
        if any("pass_count_consistent" in (sub if isinstance(sub, dict) else {})
               for sub in all_of):
            # Confirm it ALSO requires probe_report_has_actionable_fix to scope
            # to probe-closed-loop only
            if any(("probe_report_has_actionable_fix" in (sub if isinstance(sub, dict) else {}))
                   for sub in all_of):
                p0aab = i
        if (t.get("goto") == "finalize"
                and any(("probe_report_has_actionable_fix" in (sub if isinstance(sub, dict) else {}))
                        for sub in all_of)
                and not any(("pass_count_consistent" in (sub if isinstance(sub, dict) else {}))
                            for sub in all_of)):
            bare_probe_done_idx = i
    assert p0aab is not None, "P0aab transition not present in await_probe"
    assert bare_probe_done_idx is not None, "bare probe-closed-loop finalize transition vanished"
    assert p0aab < bare_probe_done_idx, (
        "P0aab must be ordered BEFORE bare probe-closed-loop finalize, "
        "otherwise the bare rule short-circuits and consistency check never runs."
    )


def test_yaml_routing_uses_p0xx_for_done_with_inconsistent_counts(tmp_path):
    """Integration check: the actual YAML await_worker.exit_transitions
    has the new transition before the bare `done → finalize` rule, so the
    contradictory case routes to await_probe, not finalize.
    """
    import yaml as _yaml
    yaml_path = _PROJECT .parent / "workflows" / "opgen_state_machine.yaml"
    raw = _yaml.safe_load(yaml_path.read_text())
    # Find await_worker spec
    aw = next(s for s in raw.get("phase_o4_states", [])
              if s.get("id") == "await_worker")
    transitions = aw.get("exit_transitions", [])
    # Find the new P0xx transition (pass_count_consistent: false → await_probe)
    p0xx = None
    bare_done_idx = None
    for i, t in enumerate(transitions):
        cond = t.get("condition", {})
        all_of = cond.get("all_of", [])
        # Detect P0xx
        if any("pass_count_consistent" in (sub if isinstance(sub, dict) else {})
               for sub in all_of):
            p0xx = i
        # Detect bare done→finalize (only handoff_match condition)
        if (t.get("goto") == "finalize"
                and len(all_of) == 1
                and all_of[0].get("handoff_match", "").startswith("→ orchestrator: done")):
            bare_done_idx = i
    assert p0xx is not None, "P0xx transition not present in YAML"
    assert bare_done_idx is not None, "bare done→finalize transition vanished"
    assert p0xx < bare_done_idx, (
        "P0xx must be ordered BEFORE bare done→finalize, otherwise the "
        "bare rule short-circuits and consistency check never runs."
    )
