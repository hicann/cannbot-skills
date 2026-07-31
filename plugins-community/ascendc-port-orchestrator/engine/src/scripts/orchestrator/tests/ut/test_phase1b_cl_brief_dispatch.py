# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CANN-LEARN-ON-RESEARCH-GAP Phase 1b: cl_brief.py thin adapter +
agent_dispatch wire-up.

Design: docs/design/KB_DESIGN_NOTES.md#cann-learn-on-research-gap-design-2026-05-20 §3.5
Phase 1a (PR #65) merged the FSM routing fabric; this PR adds the SPAWN
fabric so that when Phase 2 plugin opt-ins flip defaults, the
orchestrator can actually invoke aog-cann-learner.

These tests pin:
- briefs/cl_brief.py exists with build_cann_learner_brief() callable
- The builder produces a string with the required structural sections
  (env, ITER BUDGET, EXIT HANDOFF sentinels)
- The brief surfaces the 3 exit handoff sentinels (cann_learn_done /
  cann_learn_empty / cann_learn_blocked)
- The brief reminds the agent of the carve-out's hard rules (sealed dir
  only, no nohup, no Agent sub-spawn)
- agent_dispatch.BRIEF_BUILDERS registers "aog-cann-learner"
- agent_dispatch.spawn_for_state has an elif branch for "aog-cann-learner"
- g7_slug recognizes "aog-cann-learner" with code "cl"
- workflow_critic._AGENT_NAME_PATTERN accepts the "cl" slug suffix

NO behavior change: Phase 1b adds the spawn fabric but the gate path
(per Phase 1a) is still unreachable until Phase 2 plugin opt-ins.

Phase 1c (CLI flag + persistence) is the final Phase 1 PR.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))


def _fake_env(target: str = "a3"):
    """Synthetic AscendCEnv for brief construction."""
    from briefs._common import AscendCEnv
    return AscendCEnv(
        target=target,
        host="dummy", user="root", password="x",
        container="cont",
        cann_path="/cann",
        soc_version="Ascend910_9382" if target == "a3" else "Ascend950PR",
        benchmark_root="/root/AscendOpGenAgent",
        local_benchmark="/local/benchmark",
        local_project="/local/proj",
        archive_project="testbench",
        build_archive_enabled=True,
    )


# ──────────────────────────────────────────── brief construction


def test_cl_brief_builds_without_error(tmp_path):
    from briefs.cl_brief import build_cann_learner_brief

    brief = build_cann_learner_brief(
        "3_FusionAttention", tmp_path,
        lane=0, spawn_index=1,
        iter_cap_remaining=1, env=_fake_env("a3"),
        handoff_from_prior_agent="→ orchestrator: research_blocked — no actionable strategy",
    )
    assert isinstance(brief, str)
    assert len(brief) > 800
    # G7 slug — `cl` code
    assert "fusionattention-cl-1" in brief
    # Structural sections common to all briefs
    assert "ITER BUDGET" in brief
    assert "EXIT HANDOFF" in brief
    assert "iter_cap_remaining = 1" in brief
    # The 3 exit handoff sentinels MUST all be listed
    assert "cann_learn_done" in brief
    assert "cann_learn_empty" in brief
    assert "cann_learn_blocked" in brief


def test_cl_brief_surfaces_carve_out_rules(tmp_path):
    """Brief MUST remind the agent of the hard rules — even though hooks
    G11/G12/SC10 enforce them, the agent should know what's forbidden so
    it doesn't try (better signal/cost than letting it bounce off hooks).
    """
    from briefs.cl_brief import build_cann_learner_brief
    brief = build_cann_learner_brief(
        "3_FusionAttention", tmp_path,
        lane=0, spawn_index=1, iter_cap_remaining=1, env=_fake_env("a3"),
    )
    # Hard rules
    assert "sealed" in brief.lower()  # sealed dir mentioned
    assert ".cann_learn_sealed" in brief or "sealed_<run_id>" in brief
    # Forbidden tools
    assert "nohup" in brief.lower()
    assert "Agent" in brief  # no sub-spawn (lowercase test would match too much)


def test_cl_brief_includes_researcher_handoff(tmp_path):
    """When the researcher's gap signal is passed in, the brief surfaces it
    so the agent knows what gap they're trying to close.
    """
    from briefs.cl_brief import build_cann_learner_brief
    handoff = "→ orchestrator: research_blocked — no CANN reference for fused FlashAttention on V220"
    brief = build_cann_learner_brief(
        "3_FusionAttention", tmp_path,
        lane=0, spawn_index=1, iter_cap_remaining=1, env=_fake_env("a3"),
        handoff_from_prior_agent=handoff,
    )
    assert "Handoff from researcher" in brief
    assert "no CANN reference for fused FlashAttention" in brief


def test_cl_brief_handles_no_handoff(tmp_path):
    """No handoff (e.g. test or direct invocation) — builder must not crash."""
    from briefs.cl_brief import build_cann_learner_brief
    brief = build_cann_learner_brief(
        "3_FusionAttention", tmp_path,
        lane=0, spawn_index=1, iter_cap_remaining=1, env=_fake_env("a3"),
        handoff_from_prior_agent=None,
    )
    assert "Handoff from researcher" not in brief


def test_cl_brief_one_shot_iter_budget_warning(tmp_path):
    """Brief should explicitly call out that cann_learner is one-shot (iter_cap=1)
    so the agent doesn't expect a second attempt.
    """
    from briefs.cl_brief import build_cann_learner_brief
    brief = build_cann_learner_brief(
        "3_FusionAttention", tmp_path,
        lane=0, spawn_index=1, iter_cap_remaining=1, env=_fake_env("a3"),
    )
    assert "ONE-SHOT" in brief or "one-shot" in brief.lower() or "iter_cap=1" in brief


# ──────────────────────────────────────────── dispatch registration


def test_cl_brief_registered_in_dispatch():
    """spawn_for_state must NOT NotImplementedError on aog-cann-learner."""
    import agent_dispatch as ad
    assert "aog-cann-learner" in ad.BRIEF_BUILDERS
    builder = ad.BRIEF_BUILDERS["aog-cann-learner"]
    assert callable(builder)


def test_cl_brief_dispatch_elif_branch_exists():
    """The kwargs-dispatch elif branch must accept the cl signature."""
    src = (_reorg_paths.ORCH_DIR / "agent_dispatch.py").read_text()
    assert 'elif agent_type == "aog-cann-learner":' in src
    branch = src.split('elif agent_type == "aog-cann-learner":', 1)[1]
    branch = branch.split("elif agent_type ==", 1)[0]
    branch = branch.split("else:", 1)[0]
    # Mirrors td/tt brief signature (handoff_from_prior_agent + directive_text + plugin)
    assert "handoff_from_prior_agent=handoff_from_prior" in branch
    assert "plugin=plugin" in branch


# ──────────────────────────────────────────── G7 slug + audit regex


def test_g7_slug_recognizes_cl_code():
    """g7_slug must map aog-cann-learner → cl so brief construction doesn't
    ValueError on missing code.
    """
    from briefs._common import g7_slug
    assert g7_slug("3_FusionAttention", "aog-cann-learner", 1) == "fusionattention-cl-1"
    assert g7_slug("10_LayerNorm", "aog-cann-learner", 3) == "layernorm-cl-3"


def test_workflow_critic_pattern_accepts_cl_slug():
    """workflow_critic._AGENT_NAME_PATTERN must accept fusionattention-cl-1
    so pre-spawn audits + find_active_workspace name-extraction don't fail.
    """
    sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
    import workflow_critic as wc
    pattern: re.Pattern = getattr(wc, '_AGENT_NAME_PATTERN')
    assert pattern.match("fusionattention-cl-1 — auto-triggered from research-gap")
    # Negative control: unknown code still fails
    assert pattern.match("fusionattention-xx-1") is None
