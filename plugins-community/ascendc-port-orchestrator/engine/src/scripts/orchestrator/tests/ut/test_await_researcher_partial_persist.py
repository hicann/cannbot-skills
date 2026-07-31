# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression: await_researcher PARTIAL_PERSIST → finalize transition.

V3.8.10 (2026-05-18). Before this transition existed, an
`aog-researcher` spawn that legitimately exhausted all mitigation
candidates would emit:

    → orchestrator: PARTIAL_PERSIST — <N>/<M> bit-exact ... All
    mitigation candidates (path A / path B / ...) FAIL on ...

and the state machine fell through to the `always: true → await_worker`
catchall, looping back to a worker that had no directive, eventually
killed by `iter_cap hit for await_researcher`.

Empirical anchor: 8_Sort 2026-05-17 — researcher correctly evaluated
4 mitigation paths (extension.sort wrapper / selection-sort / multi-pass
merge / radix sort) for the K_PADDED ≥ 8192 platform limit, all paths
failed on language feasibility / perf floor / spawn budget, emitted
PARTIAL_PERSIST. State machine looped, eventual `iter_cap=2` death.

Fix: handoff_match "→ orchestrator: PARTIAL_PERSIST" from
await_researcher routes to finalize (terminal-PARTIAL accepted).

This test pins the transition so it cannot be reordered or removed
without a deliberate state-machine revision.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


def test_researcher_partial_persist_routes_to_finalize(ws) -> None:
    """The empirical 8_Sort handoff string must route to finalize."""
    handoff_8_sort = (
        "→ orchestrator: PARTIAL_PERSIST — 30/31 Pass-A bit-exact "
        "(max_abs=0.0 across all 30), perf 0.887× overall vs aclnn "
        "(peak 1.385× case 22 [4096,18432] fp32 dim=0), det best_effort "
        "30/30 runnable PASS. All four mitigation candidates "
        "(extension.sort wrapper / selection-sort O(N²) / multi-pass "
        "merge / radix sort) FAIL on at least one of (language "
        "feasibility, 0.6× perf floor, spawn budget)."
    )
    result = sm.next_state(ws, "await_researcher", handoff_8_sort)
    assert result.get("next_state") == "finalize", (
        f"PARTIAL_PERSIST from researcher must go to finalize, "
        f"got next_state={result.get('next_state')!r}; rationale="
        f"{result.get('rationale')!r}"
    )
    # The new rationale string must mention 8_Sort + iter_cap to
    # document the historical motivation for future maintainers.
    rationale = result.get("rationale", "")
    assert "8_Sort" in rationale or "iter_cap" in rationale or "death-loop" in rationale, (
        f"transition matched but rationale doesn't reference the "
        f"motivating incident: {rationale!r}"
    )


def test_researcher_directive_present_still_routes_to_worker(ws) -> None:
    """Pre-existing behavior preserved: if researcher writes
    optimization_directive.md, route to worker REGARDLESS of handoff text.
    """
    (ws / "optimization_directive.md").write_text("# directive body")
    # Even with a PARTIAL_PERSIST handoff, the directive-present transition
    # is listed FIRST so it takes precedence.
    handoff = "→ orchestrator: PARTIAL_PERSIST — exploration done; directive written"
    result = sm.next_state(ws, "await_researcher", handoff)
    assert result.get("next_state") == "await_worker", (
        f"directive present must route to worker (V1 path), "
        f"got {result.get('next_state')!r}"
    )


def test_researcher_research_partial_still_loops_for_extension(ws) -> None:
    """Pre-existing behavior preserved: research_partial signal extends
    the researcher's iter budget (looping back to await_researcher).
    """
    handoff = "→ orchestrator: research_partial — exploring direction X"
    result = sm.next_state(ws, "await_researcher", handoff)
    assert result.get("next_state") == "await_researcher", (
        f"research_partial must loop back to await_researcher, "
        f"got {result.get('next_state')!r}"
    )


def test_researcher_research_blocked_still_aborts(ws) -> None:
    """Pre-existing behavior preserved: research_blocked signals abort
    when researcher cannot converge with available info.
    """
    handoff = "→ orchestrator: research_blocked — need user input on Y"
    result = sm.next_state(ws, "await_researcher", handoff)
    assert result.get("next_state") == "abort", (
        f"research_blocked must abort, got {result.get('next_state')!r}"
    )


def test_researcher_silent_default_still_routes_to_worker(ws) -> None:
    """Pre-existing legacy default: bare handoff with no recognized
    signal routes to worker (legacy researcher hypothesis report path).
    """
    handoff = "→ orchestrator: see researcher_notes.md for next steps"
    result = sm.next_state(ws, "await_researcher", handoff)
    assert result.get("next_state") == "await_worker", (
        f"default catchall must route to worker, got {result.get('next_state')!r}"
    )


def test_researcher_partial_persist_transition_documented_in_yaml() -> None:
    """Pin: the YAML must contain the PARTIAL_PERSIST → finalize transition
    on the await_researcher state, with a rationale mentioning the
    motivating incident class.
    """
    yaml_text = sm.YAML_PATH.read_text()
    # Locate the await_researcher state block
    idx = yaml_text.find("- id: await_researcher")
    assert idx >= 0, "await_researcher state block missing from YAML"
    # Take the next ~80 lines (until next state) and check for the transition
    next_state_idx = yaml_text.find("- id: await_det_analyzer", idx)
    if next_state_idx < 0:
        next_state_idx = idx + 4000
    block = yaml_text[idx:next_state_idx]
    assert "→ orchestrator: PARTIAL_PERSIST" in block, (
        "await_researcher state missing PARTIAL_PERSIST handoff_match"
    )
    # The transition's goto must be finalize (terminal-PARTIAL)
    # Find the position of PARTIAL_PERSIST and ensure the nearby `goto: finalize`
    pp_idx = block.find('"→ orchestrator: PARTIAL_PERSIST"')
    if pp_idx < 0:
        pp_idx = block.find("'→ orchestrator: PARTIAL_PERSIST'")
    assert pp_idx >= 0
    nearby = block[pp_idx:pp_idx + 400]
    assert "goto: finalize" in nearby, (
        "PARTIAL_PERSIST handoff_match found but goto is not finalize; "
        "this would re-introduce the 8_Sort iter_cap death-loop"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
