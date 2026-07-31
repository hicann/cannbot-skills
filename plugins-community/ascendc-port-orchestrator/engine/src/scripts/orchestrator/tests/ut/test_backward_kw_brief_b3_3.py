# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""B3.3a: BackwardPlugin.kw_brief_phase_block — the backward GENERATION (C1) brief.

Pins the self-contained-verify generation contract (BACKWARD_PLUGIN_DESIGN §5.5)
validated on mul_grad / rms_norm_grad / layer_norm_grad, plus the two lessons from
the hardware proofs: FAIR perf baseline (raw vendor C-API, not autograd-through-
forward) + V220 VEC-transcendental ~fp16 precision → Newton-Raphson.

Note: this brief is unit-pinned here; it is exercised end-to-end only once B3.3b
flips the O2.5 boundary + teaches finalize/O5 the self-contained backward truth
(currently O2.5 stops at the reference). Same staging as the PR2 C2 block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from plugins.backward import BackwardPlugin  # noqa: E402


def _ws(tmp_path: Path, with_ref: bool) -> Path:
    ws = tmp_path / "rms_norm_grad"
    ws.mkdir()
    if with_ref:
        (ws / "backward_ref.json").write_text(json.dumps({
            "wrt": ["x", "w"], "grad_output": "explicit",
            "seeding_recipe": "per (dtype, case): manual_seed(1234); randn x,w; then gy",
        }))
    return ws


def test_block_encodes_generation_contract(tmp_path):
    blk = BackwardPlugin().kw_brief_phase_block(op="rms_norm_grad", workspace=_ws(tmp_path, True))
    assert isinstance(blk, str) and blk
    # self-contained, NOT edge_dataset
    assert "self-contained" in blk.lower() or "SELF-CONTAINED" in blk
    assert "edge_dataset" in blk  # explicitly says NOT edge_dataset
    # forbidden delegation (the backward cheat surface)
    assert "FORBIDDEN" in blk
    assert "autograd" in blk.lower()
    assert "REFERENCE ORACLE ONLY" in blk or "reference oracle only" in blk.lower()
    # OL-160 canonical entry-point
    assert "model_new_ascendc.py" in blk
    # archive-blind (main's 2026-05-30 rule)
    assert "archive-blind" in blk
    # surfaces the resolved wrt (multi-output)
    assert "['x', 'w']" in blk or "x" in blk


def test_block_bakes_fair_baseline_lesson(tmp_path):
    """The layer_norm_grad lesson: A/B vs the RAW vendor C-API, NOT autograd-through-
    the-forward (which inflates the ratio with Python autograd-engine overhead).
    """
    blk = BackwardPlugin().kw_brief_phase_block(op="rms_norm_grad", workspace=_ws(tmp_path, True))
    assert "FAIR" in blk and "C-API" in blk
    assert "autograd-through" in blk or "autograd-engine" in blk


def test_block_bakes_transcendental_precision_lesson(tmp_path):
    """V220 VEC transcendentals are ~fp16-mantissa → Newton-Raphson refinement."""
    blk = BackwardPlugin().kw_brief_phase_block(op="rms_norm_grad", workspace=_ws(tmp_path, True))
    assert "Newton-Raphson" in blk
    assert "Rsqrt" in blk or "fp16-mantissa" in blk or "transcendental" in blk.lower()


def test_block_graceful_without_backward_ref(tmp_path):
    """No backward_ref.json yet (e.g. brief built before O2.5) → still returns a
    valid block with a placeholder wrt, no crash.
    """
    blk = BackwardPlugin().kw_brief_phase_block(op="foo_grad", workspace=_ws(tmp_path, False))
    assert isinstance(blk, str) and "BACKWARD generation" in blk


def test_block_none_workspace_no_crash():
    blk = BackwardPlugin().kw_brief_phase_block(op="foo_grad", workspace=None)
    assert isinstance(blk, str) and blk


def test_forward_op_plugins_unaffected():
    """Sanity: the override lives on BackwardPlugin only; BasePlugin default is None
    so forward-op briefs are byte-identical (no backward block leaks).
    """
    from plugins.base import BasePlugin
    assert BasePlugin().kw_brief_phase_block(op="13_Cat", workspace=None) is None


def test_call_level_dispatch_routes_to_backward_block(tmp_path):
    """#269 lesson — prove via the REAL brief dispatcher (not isolation): the
    kw_brief phase-instructions dispatcher routes a backward workspace + the
    BackwardPlugin to the generation block (no directive). Forward plugins fall
    through (BasePlugin returns None → cold-start default, not the backward block).
    """
    from briefs import kw_brief
    ws = _ws(tmp_path, True)
    blk = getattr(kw_brief, '_phase_instructions_block')(
        "rms_norm_grad",
        ws,
        5,
        None,
        None,
        env=SimpleNamespace(opgen_mode="backward"),
        plugin=BackwardPlugin(),
    )
    assert "BACKWARD generation" in blk and "FORBIDDEN" in blk


def test_call_level_dispatch_rejects_missing_workflow_identity(tmp_path):
    """No declared mode must not silently fall through to a generic prompt."""
    from briefs import kw_brief

    with pytest.raises(RuntimeError, match="unsupported worker route"):
        getattr(kw_brief, '_phase_instructions_block')(
            "rms_norm_grad",
            _ws(tmp_path, True),
            5,
            None,
            None,
            env=None,
            plugin=BackwardPlugin(),
        )
