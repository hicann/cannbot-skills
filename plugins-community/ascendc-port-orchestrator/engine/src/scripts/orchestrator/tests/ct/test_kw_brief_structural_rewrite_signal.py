# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""kw_brief.py emits guidance for the new `structural_rewrite_needed` worker handoff
sentinel (S4, 2026-05-20).

Worker emits this sentinel when scope spans ≥2 design axes AND at least one objective
signal fires; orchestrator routes it (structural_rewrite_needed → await_researcher →
await_user_decision). Tests pin the brief content + the 4-axis fire criteria text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

ROOT = _reorg_paths.REPO_ROOT

# Read the brief source directly — it's a Python module with prose embedded in
# docstrings/strings, simplest to load as text and assert substrings.
KW_BRIEF = ROOT / "src" / "scripts" / "orchestrator" / "briefs" / "kw_brief.py"


@pytest.fixture(scope="module")
def kw_brief_text() -> str:
    assert KW_BRIEF.is_file(), f"kw_brief.py missing at {KW_BRIEF}"
    return KW_BRIEF.read_text()


def test_brief_mentions_structural_rewrite_needed_sentinel(kw_brief_text: str):
    """kw_brief.py instructs the worker on emitting the new sentinel."""
    assert "structural_rewrite_needed" in kw_brief_text, (
        "kw_brief.py is missing structural_rewrite_needed sentinel guidance (§4.3)"
    )
    # Must be the FULL handoff line including the arrow form
    assert "→ orchestrator: structural_rewrite_needed" in kw_brief_text, (
        "structural_rewrite_needed guidance must specify the canonical handoff "
        "form `→ orchestrator: structural_rewrite_needed — <reason>`"
    )


def test_brief_lists_the_four_axes(kw_brief_text: str):
    """The brief enumerates the 4 design axes that scope must span ≥2 of."""
    # Per design doc §7 Q2 (main agent's resolution): scope spans ≥2 of
    # {algorithm design, tile structure decision, primitive selection,
    #  cross-core sync discipline}.
    for axis in [
        "algorithm design",
        "tile structure decision",
        "primitive selection",
        "cross-core sync discipline",
    ]:
        assert axis in kw_brief_text, (
            f"kw_brief.py structural_rewrite_needed section missing axis {axis!r}"
        )
    # And the explicit "≥2" threshold.
    assert "≥2" in kw_brief_text, (
        "Brief must state the ≥2-axis threshold for the structural_rewrite_needed gate"
    )


def test_brief_lists_objective_signals(kw_brief_text: str):
    """Brief lists the objective signals (codex SHOULD #3 — reduce LLM self-estimation)."""
    # At least one of these objective signals must fire alongside the axis count.
    objective_signal_markers = [
        "pass_count",   # pass_count <= current_baseline + 1 after iters
        "files",        # ≥2 distinct kernel files
        "phases",       # ≥2 distinct kernel phases
        "tiling",       # new tiling layout
    ]
    missing = [m for m in objective_signal_markers if m not in kw_brief_text]
    assert not missing, (
        f"kw_brief.py structural_rewrite_needed section missing objective signal "
        f"markers: {missing}. Codex SHOULD #3 requires concrete objective triggers, "
        f"not LLM self-estimation."
    )


def test_brief_provides_worked_examples(kw_brief_text: str):
    """Brief provides calibration examples (FA = yes-structural; small ops = no)."""
    # The examples are explicit in the design doc §7 Q2. Spot-check key terms.
    assert "FA" in kw_brief_text or "fused-attention" in kw_brief_text, (
        "Brief should cite FA-class as the yes-structural calibration example"
    )
    assert "foreach_sqrt" in kw_brief_text or "single-axis" in kw_brief_text, (
        "Brief should cite a no-structural calibration example "
        "(e.g. foreach_sqrt with partial green = stay PARTIAL_PERSIST)"
    )


def test_brief_distinguishes_from_partial_persist(kw_brief_text: str):
    """Brief makes clear when to use structural_rewrite_needed vs PARTIAL_PERSIST."""
    # Both sentinels documented; structural_rewrite_needed shouldn't replace PARTIAL_PERSIST.
    assert "PARTIAL_PERSIST" in kw_brief_text, (
        "PARTIAL_PERSIST guidance still required (structural_rewrite_needed is added "
        "as a DISTINCT sentinel, not a replacement)"
    )
    # The PARTIAL_PERSIST and structural_rewrite_needed bullets should both be present
    # — verifies they coexist in the exit-handoff section.
    p_count = kw_brief_text.count("PARTIAL_PERSIST")
    s_count = kw_brief_text.count("structural_rewrite_needed")
    assert p_count >= 1 and s_count >= 1, (
        f"Expected ≥1 occurrence of each sentinel; got PARTIAL_PERSIST={p_count}, "
        f"structural_rewrite_needed={s_count}"
    )


def test_brief_references_design_doc(kw_brief_text: str):
    """Brief cites the design doc so future readers can find the rationale."""
    assert "§4.3" in kw_brief_text, (
        "structural_rewrite_needed section should cite the design doc / §4.3 "
        "for the rationale, so future kw spawns / brief reviewers can find the "
        "criteria source."
    )


def test_brief_distinguishes_partial_persist_from_structural_rewrite_needed_examples(kw_brief_text: str):
    """The worked example for 'no-structural' (foreach_sqrt 6/8) routes to PARTIAL_PERSIST,
    confirming the brief explicitly tells workers WHEN to use PARTIAL_PERSIST instead.
    """
    # Find the CANONICAL structural_rewrite_needed handoff section (in handoff list,
    # not the A.2.7 tiling-struct anti-pattern reference that may also
    # contain the substring). Anchor on the unique handoff-list prefix.
    needle = "`→ orchestrator: structural_rewrite_needed —"
    idx = kw_brief_text.find(needle)
    assert idx >= 0, "canonical structural_rewrite_needed handoff section not found"
    # Section ends roughly at the next list item or major header. Look in a 2KB window.
    window = kw_brief_text[idx:idx + 2000]
    # The no-example must explicitly mention staying with PARTIAL_PERSIST.
    assert "PARTIAL_PERSIST" in window or "no-structural" in window.lower(), (
        "The structural_rewrite_needed section should tell workers when to use "
        "PARTIAL_PERSIST instead (worked no-example must mention PARTIAL_PERSIST)"
    )
