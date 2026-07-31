# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression test for the 2026-05-23 white-box fix:
structural_rewrite_needed → await_researcher (auto-research before user_decision).

Background: commit 149f3139 (2026-05-22) added a SOLE rule
`structural_rewrite_needed → await_user_decision`. Under emergency-mode
2026-05-23 (user direction "不能再完全黑盒跑这些实验"), independent review white-box
analysis found this over-conservative: fused_quant_mat_mul cold-start hit
await_user_decision and blocked, even though researcher budget remained and
no prior research had run.

Fix: mirror V3.8.9 PARTIAL_PERSIST pattern — first rule auto-routes to
await_researcher when budget remaining + no *_strategy_inference.md; second
rule (fallthrough) goes to await_user_decision only after researcher exhausted.

This test parses opgen_state_machine.yaml and asserts the rules are in the
correct order (await_researcher rule FIRST, await_user_decision second).
"""
from __future__ import annotations

import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest
import yaml

ROOT = _reorg_paths.REPO_ROOT
YAML_PATH = ROOT .parent / "workflows" / "opgen_state_machine.yaml"


@pytest.fixture(scope="module")
def sm():
    return yaml.safe_load(YAML_PATH.read_text())


def _find_state(sm: dict, state_id: str) -> dict | None:
    for s in sm.get("phase_o4_states", []):
        if s.get("id") == state_id:
            return s
    return None


def _structural_rewrite_transitions(sm: dict) -> list[dict]:
    """Find all exit_transitions whose handoff_match references
    structural_rewrite_needed in await_worker.exit_transitions."""
    await_worker = _find_state(sm, "await_worker")
    assert await_worker is not None, "await_worker state missing from YAML"
    matched = []
    for t in await_worker.get("exit_transitions", []):
        cond = t.get("condition") or {}
        all_of = cond.get("all_of") or []
        for clause in all_of:
            if isinstance(clause, dict) and clause.get("handoff_match", "").endswith(
                "structural_rewrite_needed"
            ):
                matched.append(t)
                break
    return matched


def test_structural_rewrite_has_at_least_two_rules(sm):
    """After 2026-05-23 fix, there should be at least 2 structural_rewrite rules
    (await_researcher for budget-remaining, await_user_decision fallthrough).
    """
    rules = _structural_rewrite_transitions(sm)
    assert len(rules) >= 2, (
        f"Expected ≥2 structural_rewrite rules (researcher + "
        f"user_decision); found {len(rules)}. Possible regression: someone "
        f"reverted the 2026-05-23 fix back to a single await_user_decision rule."
    )


def test_structural_rewrite_await_researcher_rule_present(sm):
    """The auto-research rule MUST be present: structural_rewrite_needed +
    iter_below_cap(researcher) + file_absent(*_strategy_inference.md) →
    await_researcher.
    """
    rules = _structural_rewrite_transitions(sm)
    matched = None
    for t in rules:
        if t.get("goto") != "await_researcher":
            continue
        all_of = t.get("condition", {}).get("all_of", [])
        has_iter = any(
            isinstance(c, dict) and c.get("iter_below_cap") == "researcher"
            for c in all_of
        )
        has_file_absent = any(
            isinstance(c, dict)
            and "strategy_inference" in str(c.get("file_absent", ""))
            for c in all_of
        )
        if has_iter and has_file_absent:
            matched = t
            break
    assert matched is not None, (
        "Missing the 2026-05-23 white-box fix rule: "
        "`structural_rewrite_needed + iter_below_cap(researcher) + "
        "file_absent(*_strategy_inference.md) → await_researcher`. "
        "Without this rule, structural_rewrite_needed falls through to "
        "await_user_decision blocking ops with researcher budget remaining."
    )


def test_structural_rewrite_user_decision_is_fallthrough(sm):
    """The await_user_decision rule MUST be the FALLTHROUGH (no extra conditions
    beyond handoff_match), positioned AFTER the await_researcher rule.
    """
    rules = _structural_rewrite_transitions(sm)
    user_decision_rule = None
    researcher_rule_idx = None
    user_decision_rule_idx = None
    for idx, t in enumerate(rules):
        if t.get("goto") == "await_researcher":
            researcher_rule_idx = idx
        elif t.get("goto") == "await_user_decision":
            user_decision_rule = t
            user_decision_rule_idx = idx

    assert user_decision_rule is not None, "await_user_decision fallthrough missing"
    assert researcher_rule_idx is not None, "await_researcher rule missing"
    assert researcher_rule_idx < user_decision_rule_idx, (
        f"await_researcher rule (idx={researcher_rule_idx}) MUST come BEFORE "
        f"await_user_decision rule (idx={user_decision_rule_idx}). "
        f"YAML evaluates rules top-down with first-match-wins; reversed order "
        f"means user_decision always wins and the regression reappears."
    )

    # Fallthrough = single condition (just handoff_match), no iter_below_cap or file_absent
    all_of = user_decision_rule.get("condition", {}).get("all_of", [])
    assert len(all_of) == 1, (
        f"await_user_decision fallthrough should have exactly 1 condition "
        f"(handoff_match only); found {len(all_of)} conditions. Extra conditions "
        f"would prevent the fallthrough from firing when researcher exhausted."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
