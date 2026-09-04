# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""FA-class template-assembly emit in kw_brief (owner 2026-06-07).

When op is FA-class (attention-named OR FA-tagged) AND no directive_text,
kw_brief._phase_instructions_block returns the template-assembly recipe brief —
assemble a self-contained arch35 FA op from the arch22 spec + KB templates
(P-P103). The legacy alternate IL/DSL route is disabled; template assembly is
the supported FA/L4 path.

Owner architectural direction: FA is op-class, not mode — fires across all
source modes (port_a3 / backward).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # ../../ → orchestrator/

from briefs.kw_brief import (  # noqa: E402
    _fa_class_template_assembly_block,
    _fa_class_backward_stitch_block,
    _fa_class_backward_multilaunch_block,
    _fused_fa_backward_requested,
    _is_fa_class_backward,
    _phase_instructions_block,
)


def _write_classification(ws: Path, tags: list[str]) -> None:
    (ws / "op_classification.json").write_text(
        json.dumps({"op_class_tags": tags})
    )


def test_fa_class_emits_template_assembly_recipe(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX", "REDUCTION"])
    out = _fa_class_template_assembly_block("3_FusionAttention", ws)
    assert out is not None
    # Template-assembly paradigm, NOT the legacy IL-escalation
    assert "template-assembly" in out.lower()
    assert "structural_rewrite_needed" not in out
    # Recipe references the parameterized KB template + the inputs
    assert "wp_fa_regbase_impl" in out
    assert "arch22" in out
    # Target/prior context is permitted only as logged advisory evidence.
    assert "provenance-logged" in out
    assert "advisory context" in out
    assert "source-NPU capture" in out
    assert "arch35" in out


def test_fa_class_fires_regardless_of_stale_design_dir(tmp_path: Path):
    """With the legacy IL path disabled, its design/tile_level re-entry guard
    is gone — the recipe fires even if a stale
    design/ dir is present (inverts the prior 'design present → None' behavior).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX"])
    (ws / "design" / "tile_level").mkdir(parents=True)
    out = _fa_class_template_assembly_block("3_FusionAttention", ws)
    assert out is not None
    assert "template-assembly" in out.lower()


def test_non_fa_class_falls_through(tmp_path: Path):
    """Non-FA op (non-attention name + non-attention tag) never triggers the block."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["FUSED", "ELEMENTWISE"])
    out = _fa_class_template_assembly_block("not_fa", ws)
    assert out is None


def test_missing_classification_falls_through(tmp_path: Path):
    """Workspaces without op_classification.json never trip the block (the
    name-backstop reads tags from the file; no file → no false positive).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    out = _fa_class_template_assembly_block("anon", ws)
    assert out is None


def test_phase_block_uses_recipe_when_fa_class_no_directive(tmp_path: Path):
    """_phase_instructions_block honors BRANCH-5 (FA-class template-assembly)
    on first spawn when directive_text is None.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX"])
    brief = _phase_instructions_block(
        op="3_FusionAttention",
        workspace=ws,
        iter_cap_remaining=10,
        directive_text=None,
        handoff_from_prior=None,
        env=SimpleNamespace(opgen_mode="port_a3_to_a5", target="a5"),
        backend="ascendc",
        plugin=None,
    )
    assert "template-assembly" in brief.lower()
    assert "wp_fa_regbase_impl" in brief
    assert "structural_rewrite_needed" not in brief
    # Must NOT include the legacy cold-start PHASES preamble (the short-circuit
    # fully replaces it).
    assert "KB Manifest LOAD" not in brief


def _write_state(ws: Path, mode: str) -> None:
    (ws / ".opgen_state.json").write_text(json.dumps({"opgen_mode": mode}))


def test_fa_grad_backward_routes_away_from_forward_block(tmp_path: Path):
    """An FA-class BACKWARD op (attention-named + GRADIENT tag / *_grad name /
    opgen_mode==backward) must NOT get the FORWARD template-assembly recipe —
    the forward block returns None so the BackwardPlugin override fires and
    prepends the P-P103 BACKWARD stitch recipe instead.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX", "GRADIENT"])
    _write_state(ws, "backward")
    # The forward FA block must decline (return None) so the plugin can route it.
    assert _fa_class_template_assembly_block("flash_attention_score_grad", ws) is None
    assert _is_fa_class_backward(
        "flash_attention_score_grad", "ATTENTION FUSED SOFTMAX GRADIENT", ws
    ) is True


def test_fa_grad_backward_via_name_only(tmp_path: Path):
    """The *_grad NAME alone (no GRADIENT tag, no backward mode) is sufficient
    to mark an attention op as FA-class backward (belt-and-suspenders).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX"])  # no GRADIENT tag
    assert _is_fa_class_backward(
        "flash_attention_score_grad", "ATTENTION FUSED SOFTMAX", ws
    ) is True
    assert _fa_class_template_assembly_block("flash_attention_score_grad", ws) is None


def test_forward_fa_is_not_flagged_backward(tmp_path: Path):
    """A FORWARD FA op must NOT be mis-flagged as backward — it still gets the
    forward template-assembly recipe (no regression on the forward path).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX"])
    assert _is_fa_class_backward(
        "flash_attention_score", "ATTENTION FUSED SOFTMAX", ws
    ) is False
    out = _fa_class_template_assembly_block("flash_attention_score", ws)
    assert out is not None
    assert "wp_fa_regbase_impl" in out  # forward entry, forward recipe


def test_non_fa_backward_is_not_fa_class_backward(tmp_path: Path):
    """A non-attention backward op (rms_norm_grad) is NOT FA-class backward —
    it must keep the analytic-derive BackwardPlugin path (no mis-route).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ELEMENTWISE", "GRADIENT"])
    assert _is_fa_class_backward("rms_norm_grad", "ELEMENTWISE GRADIENT", ws) is False


def test_fa_backward_stitch_block_has_recipe_anchors(tmp_path: Path):
    """The FA-grad stitch brief points at the P-P103 BACKWARD recipe (NOT the
    forward entry / NOT analytic-derive), names the load-bearing steps, and
    bakes in the graybox-found 'backward has NO core-fill' clarification.
    """
    blk = _fa_class_backward_stitch_block(
        "flash_attention_score_grad", "ATTENTION FUSED SOFTMAX GRADIENT"
    )
    # The BACKWARD stitch paradigm, not the forward template-assembly.
    assert "BACKWARD template-stitch" in blk
    assert "wp_fa_regbase_impl" not in blk  # must NOT instruct the forward entry
    # The 6-step stitch recipe load-bearing anchors.
    assert "wp_fag" in blk
    assert "IterateMmDyV" in blk  # the 5-GEMM cube block
    assert "Pre/Base/Post" in blk  # the 3-phase entry
    assert "splitAxis" in blk and "BN2GS1S2" in blk  # the precision-bug rule
    # The graybox-found asset-gap, baked into the brief.
    assert "NO CORE-FILL" in blk
    assert "GetS1S2TemplateType" in blk
    # port_a3 discipline + K5 dual-pass FORBID.
    assert "ARCH35_WRAP_CHEAT" in blk
    assert "__NPU_ARCH__ 3510" in blk  # named as FORBIDDEN
    assert "copying source blocks or lines" in blk
    assert "allowed-input provenance review" in blk
    assert "COPY the CANN" not in blk
    assert "copy authorized" not in blk
    # Full-scope DEBUG-ON-TOP (the GAPs the directive wants closed).
    assert "BN2GS1S2 axis" in blk and "fp32" in blk


def test_fused_fa_backward_default_is_multilaunch_not_fused(tmp_path: Path):
    """Architecture default (C19): an FA-grad op with NO explicit fused opt-in is
    NOT fused-requested → it gets the multi-launch default, not the fused stitch.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX", "GRADIENT"])
    _write_state(ws, "backward")  # opgen_mode only, no fa_backward_arch
    assert _fused_fa_backward_requested("flash_attention_score_grad", ws) is False


def test_fused_fa_backward_opt_in_signals(tmp_path: Path):
    """The fused stitch is requested ONLY via an explicit signal:
    fa_backward_arch=="fused" OR fa_backward_large_s truthy.
    """
    ws_fused = tmp_path / "fused"
    ws_fused.mkdir()
    (ws_fused / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "backward", "fa_backward_arch": "fused"})
    )
    assert _fused_fa_backward_requested("flash_attention_score_grad", ws_fused) is True

    ws_ls = tmp_path / "ls"
    ws_ls.mkdir()
    (ws_ls / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "backward", "fa_backward_large_s": True})
    )
    assert _fused_fa_backward_requested("flash_attention_score_grad", ws_ls) is True

    # No state file → not requested (graceful default).
    ws_none = tmp_path / "none"
    ws_none.mkdir()
    assert _fused_fa_backward_requested("flash_attention_score_grad", ws_none) is False


def test_fa_backward_multilaunch_block_points_at_gqa_sibling(tmp_path: Path):
    """The multi-launch default brief points at CAND-FA-GQA-BWD-1 (the proven
    precision-core-complete path), states the 5-grad math, and does NOT instruct
    the fused single-launch stitch / MIX_AIC_1_2.
    """
    blk = _fa_class_backward_multilaunch_block(
        "flash_attention_score_grad", "ATTENTION FUSED SOFTMAX GRADIENT"
    )
    assert "MULTI-LAUNCH" in blk
    assert "CAND-FA-GQA-BWD-1" in blk
    assert "precision-core-complete" in blk
    # The 5 backward grads (math), NOT the fused 5-GEMM cube method names.
    assert "dV = Pᵀ@dO" in blk and "rowsum(dP∘P)" in blk
    assert "IterateMmDyV" not in blk  # that's the fused-cube block, not multi-launch
    # Must NOT instruct the fused single-launch / MIX path as the recipe.
    assert "MIX_AIC_1_2 single-launch" in blk  # named as what NOT to do
    # The C19 rationale is stated (so the kw understands the default choice).
    assert "C19" in blk


def test_phase_block_directive_text_wins_over_fa_class_recipe(tmp_path: Path):
    """Re-spawn from probe/optimizer passes directive_text; the directive leads
    as an overlay, but the FA-class template block MUST still be emitted
    (c341280d F1: directive used to early-return and silently drop it).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_classification(ws, ["ATTENTION", "FUSED", "SOFTMAX"])
    brief = _phase_instructions_block(
        op="3_FusionAttention",
        workspace=ws,
        iter_cap_remaining=10,
        directive_text="Probe directive: fix BF16 cast in line 234",
        handoff_from_prior=None,
        env=SimpleNamespace(opgen_mode="port_a3_to_a5", target="a5"),
        backend="ascendc",
        plugin=None,
    )
    # directive_text branch was taken
    assert "DIRECTIVE FROM PRIOR AGENT" in brief
    assert "fix BF16 cast" in brief
    # ...and the template-assembly recipe is STILL present (overlay, not
    # short-circuit) — see test_directive_preserves_fa_template_block.py
    assert "wp_fa_regbase_impl" in brief
