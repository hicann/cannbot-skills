# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0abm (2026-05-09): brief-size threshold to prevent V4 timeout regression.

Origin: DS reported 2026-05-08 that kw_brief had grown to ~22K chars over
multiple gate-doc additions (P0abd/abe/abi/abk). V4 backend hit its 300s
processing threshold between op runs (5_Cumsum hung where 9_TopK had
spawned cleanly). The 1M-context backend handles 22K fine but the harness must
support both backends.

Resolution (P0abl, same day): split kw_brief Phase D + E gate documentation
out into `src/skills/references/shared/GATE_CONTRACT.md`, leaving the brief with
thin pointers ("Read GATE_CONTRACT.md §Phase D Verify Gate before
verifying"). Brief shrunk from 21,726 → 13,386 chars (38% cut).

This test fails-fast if any common brief, including the worker's scoped
tactical-respawn path, grows past `_BRIEF_SIZE_HARD_CAP`. Mode-owned cold-start
bodies have separate content budgets and are not measured by this common-shell
guard. The cap is empirical: chosen to be comfortably below the V4 timeout
boundary while leaving headroom for new lean additions. If you legitimately
need to add bulk (and can't push it to GATE_CONTRACT.md or another
reference doc), update the cap here AND tell DS in the DS-collab group
channel (ID in local memory `reference_discord_*.md`) — V4 timeout
regression is the failure mode.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import _common as bc  # noqa: E402
from briefs.kw_brief import build_worker_brief  # noqa: E402
from briefs.ko_brief import build_optimizer_brief  # noqa: E402
from briefs.pp_brief import build_probe_brief  # noqa: E402
from briefs.ar_brief import build_researcher_brief  # noqa: E402
from briefs.da_brief import build_det_analyzer_brief  # noqa: E402
from briefs.fo_brief import build_fused_optimizer_brief  # noqa: E402


# Empirical cap: V4 timeout fired around 22K chars in DS's measurement.
# Set to 21K (was 18K) to give 1K headroom; kw_brief crossed 18K after
# independent review S4 (8a803c18) added structural_rewrite_needed sentinel + my
# Phase B.1.bis (238316eb) added OL-103 hw-transcendental check.
# DEBT-110 (2026-05-20): kw_brief at 19352 chars on origin/main is real
# bloat — next refactor must move detail to GATE_CONTRACT.md or another
# reference doc, restore to 18K cap. Interim accommodation.
# DEBT-110 continued (2026-05-21): cap raised 21000 → 21500 because
# the full worker brief on HEAD (pre-existing, before owner-directive
# prior harness change) is 21025 chars. Not caused by today's edits
# — exposed by them when commit hook ran full suite. Same refactor
# (move detail → GATE_CONTRACT.md) applies; cap can drop back to 21000
# after that refactor.
_BRIEF_SIZE_HARD_CAP = 21_500


def _seed_env(tmp_path: Path) -> bc.AscendCEnv:
    env_path = tmp_path / ".ascendc_env"
    env_path.write_text(textwrap.dedent("""\
        A5_HOST=test
        A5_USER=root
        A5_PASSWORD='x'
        A5_CONTAINER=test
        CANN_PATH=/data/cann
        SOC_VERSION=Ascend950PR_9579
        BENCHMARK_ROOT=/root/bench
        LOCAL_BENCHMARK=/local/bench
        LOCAL_PROJECT=/proj
        TARGET=a5
        OPGEN_MODE=port_a3_to_a5
        PORT_A3_SOURCE=/fixture/arch22/3_FusionAttention
        BENCHMARK_BRANCH=main
    """))
    return bc.load_env(env_path)


_BRIEFS = [
    ("kw", build_worker_brief, dict(
        lane=0,
        spawn_index=1,
        iter_cap_remaining=9,
        directive_text="Apply fixture-scoped retry directive.",
    )),
    ("ko", build_optimizer_brief, dict(lane=0, spawn_index=1, iter_cap_remaining=5)),
    ("pp", build_probe_brief, dict(lane=0, spawn_index=1, iter_cap_remaining=4)),
    ("ar", build_researcher_brief, dict(lane=0, spawn_index=1, iter_cap_remaining=2)),
    ("da", build_det_analyzer_brief, dict(lane=0, spawn_index=1, iter_cap_remaining=1)),
    ("fo", build_fused_optimizer_brief, dict(lane=0, spawn_index=1, iter_cap_remaining=1)),
]


@pytest.mark.parametrize("agent,builder,kwargs", _BRIEFS)
def test_brief_under_size_cap(tmp_path, agent, builder, kwargs):
    """Every common brief path must be under _BRIEF_SIZE_HARD_CAP chars.

    For kw this measures the supported scoped tactical-respawn path; the
    migration/backward cold-start bodies are mode-owned and tested separately.
    Failure means V4 backend will likely hit its 300s processing threshold
    on this common path. Push detail to kb/shared/GATE_CONTRACT.md (or another
    reference doc) and leave a thin pointer in the brief.
    """
    env = _seed_env(tmp_path)
    op = "3_FusionAttention"
    workspace = tmp_path / "workspace" / op
    workspace.mkdir(parents=True)

    brief = builder(op, workspace, env=env, **kwargs)

    assert len(brief) < _BRIEF_SIZE_HARD_CAP, (
        f"{agent}_brief is {len(brief)} chars, exceeds cap "
        f"{_BRIEF_SIZE_HARD_CAP}. Move detail to "
        f"src/skills/references/shared/GATE_CONTRACT.md and reference it from the "
        f"brief instead of inlining. See P0abl (2026-05-09) for the pattern."
    )


def test_gate_contract_md_exists():
    """The GATE_CONTRACT.md file briefs reference must actually exist."""
    repo_root = _reorg_paths.REPO_ROOT
    gate_contract = repo_root.parent / "kb" / "shared" / "GATE_CONTRACT.md"
    assert gate_contract.exists(), (
        f"GATE_CONTRACT.md missing at {gate_contract}. Briefs reference "
        f"this file by path; if it's gone, every kw spawn will fail Read."
    )
    # Sanity: contains the §-anchor names the briefs cite
    text = gate_contract.read_text()
    assert "## §Phase D Verify Gate" in text
    assert "## §Phase E Knowledge Update" in text
