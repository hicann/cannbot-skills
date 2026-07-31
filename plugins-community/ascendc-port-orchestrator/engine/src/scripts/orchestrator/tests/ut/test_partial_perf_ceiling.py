# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for PARTIAL_PERF_STRUCTURAL_CEILING handoff routing (P0g, op#10 finding).

Background: kw is NOT supposed to claim structural ceiling — that's ko/fo
territory after actual optimization attempt. But kw_brief had this in
EXIT HANDOFF OPTIONS by mistake. P0g fix:
  1. Remove from kw_brief.
  2. Add YAML safety net: if kw emits it anyway, treat like `done`
     (V3.8.4 forced escalation if perf below threshold, else finalize).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402
from briefs.kw_brief import build_worker_brief  # noqa: E402
from briefs._common import AscendCEnv  # noqa: E402


@pytest.fixture
def fake_env():
    return AscendCEnv(
        target="a5", host="198.51.100.35", user="root", password="x",
        container="npu_dev3", cann_path="/data/cann_b103/cann-9.0.0",
        soc_version="Ascend950PR_9579", benchmark_root="/home/x/bench",
        local_benchmark="/home/x/bench-local", local_project="/home/x/proj",
        archive_project="backward_ops", build_archive_enabled=True,
        opgen_mode="backward",
    )


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    (tmp_path / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "backward",
    }))
    return tmp_path


# ---------------------------------------------------------------------------
# Brief: PARTIAL_PERF_STRUCTURAL_CEILING removed + DO NOT WRITE warning added
# ---------------------------------------------------------------------------
def test_kw_brief_no_longer_advertises_structural_ceiling(fake_env, ws):
    """kw_brief EXIT HANDOFF OPTIONS list must NOT include
    PARTIAL_PERF_STRUCTURAL_CEILING (P0g).
    """
    brief = build_worker_brief("17_AdamW", ws, lane=0, spawn_index=1,
                                iter_cap_remaining=9, env=fake_env)
    # The DO NOT WRITE block explicitly forbids it
    assert "DO NOT write" in brief
    # The actual EXIT HANDOFF OPTIONS list (above the DO NOT block) should not
    # offer it as a valid option. We check the bullet-list pattern.
    options_section = brief.split("# EXIT HANDOFF OPTIONS")[1]
    options_until_dont = options_section.split("DO NOT write")[0]
    assert "PARTIAL_PERF_STRUCTURAL_CEILING" not in options_until_dont, (
        "kw_brief still offers PARTIAL_PERF_STRUCTURAL_CEILING as a valid "
        "EXIT HANDOFF OPTION (P0g regression — it belongs to ko/fo only)"
    )


def test_kw_brief_warns_against_structural_ceiling(fake_env, ws):
    brief = build_worker_brief("17_AdamW", ws, lane=0, spawn_index=1,
                                iter_cap_remaining=9, env=fake_env)
    # The "DO NOT" section should explicitly call this out
    after_dont = brief.split("DO NOT write")[1]
    assert "PARTIAL_PERF_STRUCTURAL_CEILING" in after_dont
    assert "RESERVED" in after_dont or "reserved" in after_dont


# ---------------------------------------------------------------------------
# YAML: PARTIAL_PERF_STRUCTURAL_CEILING from kw routes to optimizer or finalize
# ---------------------------------------------------------------------------
def _seed_minimal_log(ws, *, perf_ratio: float | None):
    """Seed verification.json + state_transitions.jsonl for state machine."""
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                       "pass_a": {"status": "PASS", "tier1_pass": 60, "total": 60},
                       "pass_b": {"status": "PASS", "tier1_pass": 16, "total": 16}},
        "performance": ({"ratio": perf_ratio} if perf_ratio is not None else {}),
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 60, "n_cases_checked": 60},
    }))
    (ws / "state_transitions.jsonl").write_text("")


def test_kw_partial_perf_ceiling_low_perf_routes_to_optimizer(ws):
    """V3.8.6 safety net: kw emits PARTIAL_PERF_STRUCTURAL_CEILING + perf
    below the parity threshold (1.0, owner-directed 2026-07-21) → route to
    await_optimizer (V3.8.4 spirit), not abort.
    """
    _seed_minimal_log(ws, perf_ratio=0.19)
    handoff = ("→ orchestrator: PARTIAL_PERF_STRUCTURAL_CEILING — "
               "Pass A 60/60 + Pass B 16/16 + det 60/60; perf 0.19x flat")
    result = sm.next_state(ws, "await_worker", handoff)
    assert "error" not in result, result
    assert result["next_state"] == "await_optimizer"


def test_kw_partial_perf_ceiling_high_perf_routes_to_finalize(ws):
    """If perf already at/above threshold, ceiling claim → finalize (no
    point escalating). Ratio is at/above the default parity threshold (1.0,
    owner-directed 2026-07-21; was 0.6) — a sub-parity ratio like 0.85 now
    correctly escalates instead (owner-accepted behavior change).
    """
    _seed_minimal_log(ws, perf_ratio=1.05)
    handoff = "→ orchestrator: PARTIAL_PERF_STRUCTURAL_CEILING — perf 1.05x"
    result = sm.next_state(ws, "await_worker", handoff)
    assert "error" not in result, result
    assert result["next_state"] == "finalize"


def test_kw_partial_perf_ceiling_no_perf_routes_to_finalize(ws):
    """No perf data → no escalation (perf_below_threshold returns False
    when ratio is None per state_machine._perf_ratio_and_threshold).
    """
    _seed_minimal_log(ws, perf_ratio=None)
    handoff = "→ orchestrator: PARTIAL_PERF_STRUCTURAL_CEILING — perf N/A"
    result = sm.next_state(ws, "await_worker", handoff)
    assert "error" not in result, result
    assert result["next_state"] == "finalize"


def test_kw_normal_done_still_routes_correctly(ws):
    """Sanity: don't break the canonical `→ orchestrator: done` path.
    Uses an at-parity ratio (>= 1.0 default parity threshold, owner-directed
    2026-07-21) so a satisfactory-perf `done` finalizes.
    """
    _seed_minimal_log(ws, perf_ratio=1.05)
    handoff = "→ orchestrator: done — Pass A 60/60, perf 1.05x"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "finalize"


def test_kw_done_low_perf_still_escalates_to_optimizer(ws):
    """V3.8.4 sanity: `done` + perf below threshold still routes to optimizer."""
    _seed_minimal_log(ws, perf_ratio=0.2)
    handoff = "→ orchestrator: done — perf 0.2x"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "await_optimizer"
