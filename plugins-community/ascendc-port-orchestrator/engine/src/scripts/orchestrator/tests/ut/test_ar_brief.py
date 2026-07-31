# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0q (2026-05-05): aog-researcher brief builder.

Origin: op#10_layernorm 2026-05-05. After P0o + P0p landed, orchestrator
correctly routed await_user_decision → await_researcher and tried to spawn
aog-researcher → AttributeError: brief builder for 'aog-researcher' not yet
implemented (Day 2 task). The TODO had been pending since DEBT-077 V1 init.

Fix: add briefs/ar_brief.py with build_researcher_brief, register in
agent_dispatch.BRIEF_BUILDERS, add researcher branch in spawn_for_state's
kwargs dispatch (handoff_from_prior_agent + directive_text).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))


def test_ar_brief_builds_without_error(tmp_path):
    from briefs.ar_brief import build_researcher_brief
    from briefs._common import AscendCEnv

    fake_env = AscendCEnv(
        target="a5", host="dummy", user="root", password="x",
        container="cont", cann_path="/cann", soc_version="Ascend950PR",
        benchmark_root="/root/AscendOpGenAgent",
        local_benchmark="/local/benchmark",
        local_project="/local/proj",
        archive_project="testbench",
        build_archive_enabled=True,
    )
    brief = build_researcher_brief(
        "10_layernorm", tmp_path,
        lane=0, spawn_index=1,
        iter_cap_remaining=2, env=fake_env,
        handoff_from_prior_agent="→ orchestrator: await_user_decision — investigate alternate vendor",
    )
    # Smoke checks — brief is non-empty, contains required structural blocks
    assert isinstance(brief, str)
    assert len(brief) > 500
    assert "layernorm-ar-1" in brief  # G7 slug (numeric prefix stripped by g7_slug)
    assert "researcher" in brief.lower()
    assert "ITER BUDGET" in brief
    assert "EXIT HANDOFF" in brief
    # Must list the 4 valid handoff forms
    assert "research_done" in brief
    assert "research_partial" in brief
    assert "research_blocked" in brief
    assert "PARTIAL_PERSIST" in brief


def test_ar_brief_registered_in_dispatch():
    """Sanity: aog-researcher must be in BRIEF_BUILDERS so spawn_for_state
    doesn't NotImplementedError on it.
    """
    import agent_dispatch as ad
    assert "aog-researcher" in ad.BRIEF_BUILDERS
    builder = ad.BRIEF_BUILDERS["aog-researcher"]
    assert callable(builder)


def test_ar_brief_handles_no_handoff(tmp_path):
    """Cold-start case: spawn_for_state may pass handoff_from_prior=None."""
    from briefs.ar_brief import build_researcher_brief
    from briefs._common import AscendCEnv

    fake_env = AscendCEnv(
        target="a5", host="dummy", user="root", password="x",
        container="cont", cann_path="/cann", soc_version="Ascend950PR",
        benchmark_root="/root/AscendOpGenAgent",
        local_benchmark="/local/benchmark",
        local_project="/local/proj",
        archive_project="testbench",
        build_archive_enabled=True,
    )
    brief = build_researcher_brief(
        "10_layernorm", tmp_path,
        lane=0, spawn_index=1,
        iter_cap_remaining=2, env=fake_env,
        handoff_from_prior_agent=None,
    )
    assert isinstance(brief, str)
    assert len(brief) > 500
    # When no handoff, should not contain "Handoff from prior agent" header
    assert "## Handoff from prior agent" not in brief
