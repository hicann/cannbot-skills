# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""structural_rewrite_needed sentinel recognition in extract_canonical_handoff.

S4 (PR #31) added kw_brief guidance for workers to emit:
    → orchestrator: structural_rewrite_needed — <reason>

But the receiver-side recognizer (orchestrator.extract_canonical_handoff) was
missing the keyword in _VALID_ARROW_KEYWORDS, so the sentinel got dropped to
"contract violation → abort". Caught during S5 cold-start on 3_FusionAttention
where kw-1 correctly emitted the sentinel but the orchestrator aborted.

These tests pin the recognition contract so the regression can't recur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

import orchestrator as orch  # noqa: E402


def test_structural_rewrite_needed_in_valid_arrow_keywords():
    """The keyword list explicitly includes the sentinel."""
    assert "structural_rewrite_needed" in getattr(orch, '_VALID_ARROW_KEYWORDS'), (
        "_VALID_ARROW_KEYWORDS missing 'structural_rewrite_needed' — IL escalation "
        "handoff will be dropped as malformed (caught during S5 cold-start 2026-05-20)"
    )


def test_extract_recognizes_bare_sentinel():
    """Worker emits the bare sentinel form — recognizer returns the full canonical line."""
    output = """
some worker stdout
prose discussion ...
→ orchestrator: structural_rewrite_needed — FA-class fused-attention on cold-start
    """
    result = orch.extract_canonical_handoff(output)
    assert result.startswith("→ orchestrator: structural_rewrite_needed"), (
        f"Recognizer dropped the sentinel; got: {result!r}"
    )


def test_extract_recognizes_with_dash_reason():
    """Sentinel + reason form (canonical per S4 kw_brief)."""
    output = (
        "[aog-kernel-worker-1] | Phase A complete. Final handoff summary:\n"
        "→ orchestrator: structural_rewrite_needed — "
        "FA-class fused-attention on cold-start (61 cases, 4 layouts × 3 dtypes); "
        "scope spans algorithm design + tile structure + primitive selection + "
        "cross-core sync\n"
    )
    result = orch.extract_canonical_handoff(output)
    assert "structural_rewrite_needed" in result
    assert result.startswith("→ orchestrator: structural_rewrite_needed")


def test_extract_recognizes_markdown_wrapped_sentinel():
    """Sentinel wrapped in markdown formatting (per orchestrator's P0h handling)."""
    output = """
some worker stdout
**Exit handoff**: `→ orchestrator: structural_rewrite_needed — FA scope-spanning rewrite`
    """
    result = orch.extract_canonical_handoff(output)
    assert "structural_rewrite_needed" in result, (
        f"Markdown-wrapped sentinel should still be recognized; got: {result!r}"
    )


def test_extract_last_sentinel_wins_with_structural_rewrite():
    """Worker may discuss the sentinel earlier in stdout; only the LAST line counts."""
    output = """
[aog-kernel-worker-1] | Considering: do I emit structural_rewrite_needed here?
[aog-kernel-worker-1] | Let me check the 4-axis criteria first.
[aog-kernel-worker-1] | (analysis...)
[aog-kernel-worker-1] | Final decision:
→ orchestrator: structural_rewrite_needed — FA scope-spanning
    """
    result = orch.extract_canonical_handoff(output)
    assert result.startswith("→ orchestrator: structural_rewrite_needed")


def test_extract_distinguishes_structural_rewrite_from_partial_persist():
    """Both sentinels are valid; recognizer doesn't normalize one to the other."""
    structural_output = "→ orchestrator: structural_rewrite_needed — reason\n"
    partial_output = "→ orchestrator: PARTIAL_PERSIST — reason\n"

    r_s = orch.extract_canonical_handoff(structural_output)
    r_p = orch.extract_canonical_handoff(partial_output)

    assert "structural_rewrite_needed" in r_s
    assert "PARTIAL_PERSIST" in r_p
    # Sanity: neither sentinel collapses into the other
    assert "PARTIAL_PERSIST" not in r_s
    assert "structural_rewrite_needed" not in r_p
