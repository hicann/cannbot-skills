# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O3 PROGRESS.md scaffold (P0ll 2026-05-06).

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 3.

Scope:
    Write a canonical PROGRESS.md skeleton at workspace start before
    first worker spawn. Workers append to ## Timeline section instead
    of inventing their own headers. Stable scaffold = stable handoff
    extraction (P0w-class bugs traced back to PROGRESS.md tail varying
    across workers).

Idempotent: if PROGRESS.md already exists with the canonical skeleton,
no-op. If it exists with non-canonical content, leave it alone (don't
overwrite real work).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SKELETON_MARKER = "<!-- PHASE_O3_SCAFFOLD_v1 -->"


@dataclass
class O3Report:
    verdict: str  # "WROTE" | "PRESERVED" | "ALREADY_CANONICAL"
    summary: str = ""


def init_progress_md(
    workspace: Path,
    op: str,
    *,
    opgen_mode: str = "scoped",
    perf_threshold: float = 1.0,  # parity default (owner-directed 2026-07-21, was 0.6)
    det_policy: str = "best_effort",
    hard_floors: Optional[dict] = None,
) -> O3Report:
    """Write canonical PROGRESS.md scaffold to workspace.

    Args:
        workspace: workspace dir
        op: op name
        opgen_mode: supported workflow identity written to the scaffold
        perf_threshold: V3.8.4 ratio floor (default 0.6×)
        det_policy: required / best_effort / n_a (from O1.5 once wired)
        hard_floors: {pass_a: "60/60", pass_b: "16/16", det: "60/60"} OR None

    Returns O3Report:
        WROTE: scaffold written (PROGRESS.md didn't exist)
        ALREADY_CANONICAL: scaffold marker found, no-op
        PRESERVED: PROGRESS.md exists with non-canonical content; left alone
    """
    p = workspace / "PROGRESS.md"
    if p.exists():
        existing = p.read_text(errors="replace")
        if SKELETON_MARKER in existing:
            return O3Report(
                verdict="ALREADY_CANONICAL",
                summary="PROGRESS.md already has phase O3 scaffold",
            )
        if existing.strip():
            # Has content but no marker — preserve. Workers may have appended
            # their own structure. Don't overwrite.
            return O3Report(
                verdict="PRESERVED",
                summary="PROGRESS.md has non-scaffold content; left untouched",
            )
        # File exists but empty — fall through and write scaffold

    floors_block = _format_hard_floors(hard_floors)
    content = _SKELETON_TEMPLATE.format(
        op=op,
        marker=SKELETON_MARKER,
        opgen_mode=opgen_mode,
        perf_threshold=perf_threshold,
        det_policy=det_policy,
        hard_floors_block=floors_block,
    )
    workspace.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return O3Report(
        verdict="WROTE",
        summary=f"PROGRESS.md scaffold written ({len(content)} bytes)",
    )


def _format_hard_floors(floors: Optional[dict]) -> str:
    if not floors:
        return "- (none — first iter; baseline established by kw-1)"
    lines = []
    for k in ("pass_a", "pass_b", "det", "perf"):
        if k in floors:
            lines.append(f"- {k}: {floors[k]}")
    return "\n".join(lines) if lines else "- (none)"


_SKELETON_TEMPLATE = """\
# PROGRESS — {op}

{marker}

Mode: {opgen_mode}
Perf threshold: {perf_threshold}× (V3.8.4 escalation rule)
DET_POLICY: {det_policy}

## Hard gate floors (preserve byte-identical or beat)

{hard_floors_block}

## Timeline (append-only — workers + orchestrator each add a section per spawn)

<!-- WORKERS: append your section below this line. Format:
###  [HH:MM] {{slug}}
{{narrative + final handoff line at the very end of your section}}
-->
"""
