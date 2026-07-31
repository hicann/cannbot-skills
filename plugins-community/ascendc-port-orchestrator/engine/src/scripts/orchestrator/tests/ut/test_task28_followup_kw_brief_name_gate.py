# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""task#33 (task#28 follow-up) — kw_brief FA-class emit block must use the op NAME, not tags.

Reference patch: blue `origin/blue/pr/kw-brief-fa-gate-name-align` (this test is
blue's, carried over verbatim + credited; the kw_brief fix was re-impl'd onto
current origin/main to avoid reverting #325's task#31 design doc — dev-owned merge
per owner's user=consumer / dev=developer division).

blue repro (live, 2026-06-01): after task#28 (PR #324) fixed the FA-class *routing*
gate (name-based `is_attention_named`), `/ascendc-op-gen hc_split_sinkhorn` was
correctly routed to await_worker (kw path). BUT the kw *brief* generator still
gated the FA-STOP block on the tag-based `is_fa_class(op_class)`:

    briefs/kw_brief.py:172  (before)
        from plugins.base import is_fa_class as _is_fa
        if not _is_fa(op_class): return None      # True for fused+softmax tags

hc_split_sinkhorn's tags `[fused, softmax, reduction, transcendental, ...]` made
`is_fa_class` True → the worker received a "# PHASES (FA-class autonomous IL
escalation)" brief saying **"STOP — DO NOT AUTHOR, emit structural_rewrite_needed
and EXIT"**. The two gates disagreed: routing said "kw, author normally", brief
said "don't author, hand off to IL" — which the FSM won't honor (name gate False)
→ no kernel produced (worker correctly refused, handed back await_user_decision).

Fix: gate `_fa_class_template_assembly_block` on `is_attention_named(op)`, the
same predicate as the routing gate. Then a pure-Vector fused op (sinkhorn) gets a
normal author brief; a real FA op (3_FusionAttention) still gets the FA-STOP block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from briefs.kw_brief import _fa_class_template_assembly_block  # noqa: E402


def _mk_workspace(tmp_path: Path, op_name: str, *, design_absent: bool = True) -> Path:
    """Create a minimal workspace dir with FUSED+SOFTMAX op_classification.json
    (the tag set that mis-fired the old gate) and no design/tile_level."""
    ws = tmp_path / op_name
    ws.mkdir()
    (ws / "op_classification.json").write_text(
        json.dumps(
            {
                "op": op_name,
                "op_class_tags": ["fused", "softmax", "reduction", "transcendental"],
            }
        )
    )
    if not design_absent:
        (ws / "design" / "tile_level").mkdir(parents=True)
    return ws


def test_pure_vector_fused_softmax_op_gets_normal_brief(tmp_path):
    """hc_split_sinkhorn (fused+softmax tags, NOT attention-named) must NOT get the
    FA-STOP block — _fa_class_template_assembly_block returns None → normal author.
    """
    ws = _mk_workspace(tmp_path, "hc_split_sinkhorn", design_absent=True)
    assert _fa_class_template_assembly_block("hc_split_sinkhorn", ws) is None


def test_real_fa_op_gets_template_assembly_block(tmp_path):
    """A real FA op (attention-named) must get the template-assembly recipe block
    (owner 2026-06-07: replaces the legacy IL-escalation 'STOP' block). The
    name-gate fix must not regress true FA detection.
    """
    ws = _mk_workspace(tmp_path, "3_FusionAttention", design_absent=True)
    block = _fa_class_template_assembly_block("3_FusionAttention", ws)
    assert block is not None
    assert "template-assembly" in block.lower()
    assert "wp_fa_regbase_impl" in block
    # NOT the legacy IL-escalation
    assert "structural_rewrite_needed" not in block


def test_fa_op_fires_regardless_of_design_dir(tmp_path):
    """structural-rewrite-IL disabled (owner 2026-06-07): there is no IL designer, so the
    legacy design/tile_level re-entry guard is gone — the recipe fires even with a
    stale design/ dir present (inverts the prior 'design present → None').
    """
    ws = _mk_workspace(tmp_path, "3_FusionAttention", design_absent=False)
    assert _fa_class_template_assembly_block("3_FusionAttention", ws) is not None


@pytest.mark.parametrize("op_name", ["grouped_query_attention", "flash_attention_score"])
def test_other_attention_named_ops_get_block(tmp_path, op_name):
    ws = _mk_workspace(tmp_path, op_name, design_absent=True)
    assert _fa_class_template_assembly_block(op_name, ws) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
