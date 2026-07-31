# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for C35 KB-overlap detection with reason codes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from cann_learn import kb_overlap as kbo  # noqa: E402


def test_extract_evidence_pointers():
    text = "Per OL-83 and P-P58 evidence; see EC-12, PB-22."
    out = getattr(kbo, '_extract_evidence_pointers')(text)
    assert out == {"OL-83", "P-P58", "EC-12", "PB-22"}


def test_extract_api_refs_namespace():
    text = "Use AscendC::DataCopy and AscendC::WholeReduceSum."
    out = getattr(kbo, '_extract_api_refs')(text)
    assert "AscendC::DataCopy" in out
    assert "DataCopy" in out
    assert "AscendC::WholeReduceSum" in out


def test_extract_api_refs_backtick():
    text = "Try `WholeReduceSum<float>` after the `DataCopy()` call."
    out = getattr(kbo, '_extract_api_refs')(text)
    assert "WholeReduceSum" in out
    assert "DataCopy" in out


def test_extract_op_classes():
    text = "This pattern applies to normalization and reduction ops."
    out = getattr(kbo, '_extract_op_classes')(text)
    assert "normalization" in out
    assert "reduction" in out


def test_parse_kb_entry_full():
    body = """
## Symptom

Slow performance on normalization with large reduction trees.
Multi-pass loads dominate.

## Solution

Use `AscendC::Normalize<U,T>` for batched dispatch via OL-110 evidence.
"""
    entry = kbo.parse_kb_entry("OL-83", "Reduction-tree fail-floor on Ascend950PR", body)
    assert entry.id == "OL-83"
    assert "OL-110" in entry.evidence_pointers
    assert "Normalize" in entry.api_refs


def test_check_overlap_2_reasons_match_triggers(tmp_path):
    """Same op_class + same evidence_family = ≥2 reasons → overlap."""
    kb_index = [kbo.parse_kb_entry(
        "OL-83",
        "fp32 unit-ULP floor",
        "Applies to normalization ops. Cite OL-110 evidence.",
    )]
    candidate = kbo.parse_candidate(
        "P-CAND-1",
        "Vendor uses batched normalization Normalize primitive",
        "On normalization ops with reduction-tree fail-floor (OL-110), vendor uses A=K dispatch.",
    )
    result = kbo.check_overlap(candidate, kb_index)
    assert result.has_overlap
    # Should match same_op_class (normalization) + same_evidence_family (OL-110)
    reasons_set = {r for m in result.matches for r in m.reasons}
    assert "same_op_class" in reasons_set
    assert "same_evidence_family" in reasons_set


def test_check_overlap_one_reason_does_not_trigger():
    """Only same_op_class match (no other reason) → not overlap."""
    kb_index = [kbo.parse_kb_entry(
        "OL-99",
        "convolution-class hint",
        "Applies to conv ops. No evidence pointers shared with normalization.",
    )]
    candidate = kbo.parse_candidate(
        "P-CAND-2",
        "Tip for conv tile sizes",
        "Conv ops benefit from larger tiles.",
    )
    # Only conv class shared, no api_refs / evidence overlap
    result = kbo.check_overlap(candidate, kb_index, min_reasons=2)
    assert not result.has_overlap


def test_check_overlap_metadata_fix_proposal_includes_diff():
    """Proposal text describes WHAT to add to existing entry."""
    kb_index = [kbo.parse_kb_entry(
        "OL-83",
        "fp32 floor",
        "Applies to reduction ops. References `Normalize` API. Cite OL-110.",
    )]
    candidate = kbo.parse_candidate(
        "P-CAND-3",
        "Normalize batched A=K",
        "Applies to normalization, attention ops. Uses `AscendC::Normalize` and `AscendC::DataCopy`. OL-110.",
    )
    result = kbo.check_overlap(candidate, kb_index)
    assert result.has_overlap, f"matches: {result.matches}"
    proposal = result.matches[0].metadata_fix_proposal
    # Should mention what's missing in existing — candidate adds normalization
    # + attention op_classes the existing entry doesn't have
    assert "op_classes" in proposal or "api_refs" in proposal


def test_parse_kb_index_walks_md_files(tmp_path):
    """parse_kb_index extracts entries from KB md files."""
    kb_dir = tmp_path / "references"
    kb_dir.mkdir()
    (kb_dir / "OL.md").write_text("""
# Operational Knowledge

## OL-83: fp32 unit-ULP floor

Applies to normalization ops. See OL-110 evidence chain.

## Solution
Use AscendC::Normalize.

## OL-110: reduction-tree fail-floor

Applies to reduction ops with output-dtype subfamily.
""")
    entries = kbo.parse_kb_index(kb_dir)
    ids = [e.id for e in entries]
    assert "OL-83" in ids
    assert "OL-110" in ids


def test_check_overlap_empty_kb_no_matches():
    candidate = kbo.parse_candidate("c", "title", "body")
    result = kbo.check_overlap(candidate, [])
    assert not result.has_overlap
    assert result.matches == []


# ── C35-fix 2026-06-05: deadlock-recipe-slip guard (PP107↔PB-35 DEBT) ──────────

def test_op_class_canonical_alias_mix_aic():
    """(i) Differently-worded same op-class must canonicalize + intersect.
    `MIX_AIC_1_2` and `mixed_aic_aiv_pattern_a` are the same class — the literal
    set-intersect missed it (the PP107↔PB-35 same_op_class miss).
    """
    a = getattr(kbo, '_extract_op_classes')("op_class=MIX_AIC_1_2 backward, hand-rolled CrossCore")
    b = getattr(kbo, '_extract_op_classes')("op_class=mixed_aic_aiv_pattern_a_tile_mmad")
    assert "mix_aic" in a and "mix_aic" in b
    assert a & b  # they now intersect


def test_construction_sig_extracted():
    """(ii) Distinctive sync-construction markers extracted (valence-agnostic)."""
    sig = getattr(kbo, '_extract_construction_sig')("Set with `CrossCoreSetFlag<2, PIPE>(id)`; 1:2 paired")
    assert "crosscore_setflag" in sig
    assert "crosscore_mode2" in sig  # the <2, ...> form now matches


def test_reject_conditions_parsed_not_dead():
    """(iii) candidate.reject_conditions must no longer be a hard-coded empty set."""
    cand = kbo.parse_candidate("c", "t",
        "Body.\n**Reject_cond**: do NOT apply on the high-level Matmul library path.")
    assert cand.reject_conditions  # non-empty
    assert "matmul" in cand.reject_conditions or "library" in cand.reject_conditions


def test_construction_collision_recipe_vs_platform_bug():
    """(ii) LOAD-BEARING regression: a RECIPE that re-walks a PLATFORM_BUG's
    construction must fire construction_collision EVEN THOUGH symptom vocabulary is
    opposite-valence (recipe says 'do X', bug says 'X deadlocks'). This is the
    PP107↔PB-35 deadlock-recipe-slip that the old C35 (op_class literal-match +
    symptom-only bridge) missed.
    """
    pb = kbo.parse_kb_entry(
        "PB-35",
        "event_t(0) cube-internal pipe sync collides with CrossCoreSetFlag chain → silent hang",
        "Symptom: kernel using manual `CrossCoreSetFlag<0x2>(flagId)` chain in "
        "KERNEL_TYPE_MIX_AIC_1_2 mode deadlocks; torch.npu.synchronize() hangs, no fault. "
        "op_class=mixed_aic_aiv_pattern_a_tile_mmad. SYNC MODE 2 (1:2 ratio) + a SHARED flag id.")
    # The recipe (positive framing) re-walking the same construction:
    recipe = kbo.parse_candidate(
        "CAND-PP107",
        "MIX_AIC software-pipeline GENERATION recipe — emittable staggered schedule",
        "op_class=MIX_AIC_1_2 backward; hand-rolled CrossCore. Set with "
        "`CrossCoreSetFlag<2, PIPE>(id)`; the <2> MODE = AIC-AIV 1:2 paired mode. "
        "Allocate ONE flag id per edge (shared flag id). This GENERATES the pipeline.")
    res = kbo.check_overlap(recipe, [pb])
    assert res.has_construction_collision, "recipe re-walking PB-35 must flag construction_collision"
    cc = res.construction_collisions[0]
    assert cc.kb_id == "PB-35"
    assert "same_construction_sig" in cc.reasons
    assert "PLATFORM_BUG" in cc.metadata_fix_proposal
    assert "DO NOT promote" in cc.metadata_fix_proposal


def test_no_false_collision_on_non_bug_overlap():
    """A recipe overlapping a normal P-P entry (not a PB) must NOT be tagged a
    construction_collision (that flag is reserved for PLATFORM_BUG re-walks).
    """
    pp = kbo.parse_kb_entry("P-P99", "dq/dk contraction pairing",
        "op_class=MIX_AIC_1_2 backward. Use `CrossCoreSetFlag<4>(id)` disjoint ids.")
    recipe = kbo.parse_candidate("CAND-X", "some MIX recipe",
        "op_class=MIX_AIC_1_2 backward. `CrossCoreSetFlag<4, PIPE>(id)` disjoint ids id+16.")
    res = kbo.check_overlap(recipe, [pp])
    assert not res.has_construction_collision  # P-P99 is not a PB-
