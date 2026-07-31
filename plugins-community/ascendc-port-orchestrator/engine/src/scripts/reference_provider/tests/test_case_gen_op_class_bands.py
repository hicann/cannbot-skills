# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Op-class-aware SEMANTIC shape bands (2026-06-08, owner case_gen-coverage directive).

case_gen must self-generate comprehensive DOMAIN-semantic coverage (not just generic
align/tile/prime edges), because real a3-port/op-gen agents have no benchmark reference —
only case_gen output. The bar: case_gen attention coverage ≥ the curated FA V2-64,
INCLUDING the head-dim bands (D640/768) the generic `_shape_plan` never emits (the exact
gap that produced the case-gen reconcile, docs/analysis/FA_VERIFY_SET_VS_SPEC64_RECONCILE_2026_06_08.md).

The mechanism is an OPT-IN op-class band registry: schemas declaring `op_class` get the
engine's domain bands prepended; ops without `op_class` are byte-identical (covered by the
155 existing case_gen tests staying green).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # reference_provider/

import case_gen as cg  # noqa: E402


# ---------------------------------------------------------------------------
# Band-emitter: covers the FA-semantic D/S/B/N bands incl the V2-64 gap (640/768)
# ---------------------------------------------------------------------------
def test_attention_bands_cover_head_dim_incl_640_768():
    """The load-bearing case: the generic plan emits NO D640/768; the attention band
    emitter MUST (that absence was the V2-64 coverage gap).
    """
    bands = getattr(cg, '_attention_shape_bands')("sign_off", torch.float16, 4)
    d_buckets = sorted({s["shape"][3] for s in bands})
    # spec V2-64 D-buckets are {64,128,512,640,768}; we cover those + 256 (multi-tile boundary)
    for d in (64, 128, 512, 640, 768):
        assert d in d_buckets, f"attention bands missing head-dim {d}: {d_buckets}"
    assert 640 in d_buckets and 768 in d_buckets, "missing the V2-64 head-dim gap (640/768)"


def test_attention_bands_cover_seqlen_tile_bands():
    bands = getattr(cg, '_attention_shape_bands')("sign_off", torch.float16, 4)
    s_bands = sorted({s["shape"][1] for s in bands})
    # seqlen tile multiples + tails (V2-64 used {64,128,192,256,384,512})
    for s in (64, 128, 192, 256, 384, 512):
        assert s in s_bands, f"attention bands missing seqlen {s}: {s_bands}"


def test_attention_bands_cover_batch_and_heads():
    bands = getattr(cg, '_attention_shape_bands')("sign_off", torch.float16, 4)
    b_vals = {s["shape"][0] for s in bands}
    n_vals = {s["shape"][2] for s in bands}
    assert {1, 2, 4} <= b_vals, f"batch bands incomplete: {b_vals}"
    assert {8, 16, 32} <= n_vals, f"head_num bands incomplete: {n_vals}"


def test_attention_bands_are_rank4_bsnd_base():
    bands = getattr(cg, '_attention_shape_bands')("sign_off", torch.float16, 4)
    assert bands, "expected attention bands at sign_off"
    assert all(len(s["shape"]) == 4 for s in bands), "attention base must be rank-4 [B,S,N,D]"


# ---------------------------------------------------------------------------
# Opt-in dispatch: only fires on registered op_class; never affects other ops
# ---------------------------------------------------------------------------
def test_dispatch_registered_op_classes():
    assert len(getattr(cg, '_op_class_shape_bands')("attention", "sign_off", torch.float16, 4)) > 0
    assert len(getattr(cg, '_op_class_shape_bands')("fa_class", "sign_off", torch.float16, 4)) > 0
    # case-insensitive
    assert len(getattr(cg, '_op_class_shape_bands')("ATTENTION", "sign_off", torch.float16, 4)) > 0


def test_dispatch_optin_returns_empty_for_non_op_class():
    assert getattr(cg, '_op_class_shape_bands')(None, "sign_off", torch.float16, 4) == []
    assert getattr(cg, '_op_class_shape_bands')("", "sign_off", torch.float16, 4) == []
    assert getattr(cg, '_op_class_shape_bands')("not_registered", "sign_off", torch.float16, 4) == []


def test_dispatch_rank_mismatch_returns_empty():
    # attention base is rank-4; a non-4 rank yields no attention bands (no crash)
    assert getattr(cg, '_op_class_shape_bands')("attention", "sign_off", torch.float16, 1) == []
    assert getattr(cg, '_op_class_shape_bands')("attention", "sign_off", torch.float16, 3) == []


# ---------------------------------------------------------------------------
# generate_cases: op_class schema gets the bands; non-op_class schema does NOT
# ---------------------------------------------------------------------------
def _bounded_fa_base(shape):
    """Keep this unit fixture source-realistic and below its memory budget."""
    return (
        len(shape) == 4
        and shape[0] <= 4
        and shape[2] <= 32
        and shape[3] <= 768
    )


def _fa_schema(op_class=None):
    schema = {
        "rank": 4,
        "tensor_inputs": [{"name": "query", "shape_derive": None}],
        "scalar_inputs": [],
        "base_shape_filter": _bounded_fa_base,
    }
    if op_class:
        schema["op_class"] = op_class
    return schema


def test_generate_cases_op_class_injects_attention_bands():
    """A rank-4 schema with op_class=attention gets the D640/768 bands; the same schema
    WITHOUT op_class does not (proving the bands come from the op-class dispatch).
    """
    with_oc = cg.generate_cases(_fa_schema("attention"), coverage_tier="sign_off",
                                dtype=torch.float16)
    without_oc = cg.generate_cases(_fa_schema(None), coverage_tier="sign_off",
                                   dtype=torch.float16)
    d_with = {c["shape"][3] for c in with_oc if len(c["shape"]) == 4}
    d_without = {c["shape"][3] for c in without_oc if len(c["shape"]) == 4}
    assert 640 in d_with and 768 in d_with, f"op_class did not inject 640/768: {sorted(d_with)}"
    assert 640 not in d_without and 768 not in d_without, (
        f"non-op_class schema unexpectedly has 640/768: {sorted(d_without)}"
    )
    assert len(with_oc) > len(without_oc), "op_class bands should add cases"


# ---------------------------------------------------------------------------
# increment-2: high-D bands CARRY their dropout + layout config (coverage =
# shape × dropout), via the generic opt-in per-shape-plan "scalars" override.
# Spec V2-64: 640 is BNSD kp0.9, 768 is SBH kp0.8 — high-D ARE the dropout cases.
# ---------------------------------------------------------------------------
def test_attention_high_d_bands_carry_dropout_and_layout():
    bands = getattr(cg, '_attention_shape_bands')("sign_off", torch.float16, 4)
    by_name = {b["name"]: b for b in bands}
    d640 = by_name["attn_b1_n4_s192_d640"]
    d768 = by_name["attn_b1_n2_s384_d768"]
    assert d640["scalars"]["keep_prob"] == 0.9 and d640["scalars"]["input_layout"] == "BNSD"
    assert d768["scalars"]["keep_prob"] == 0.8 and d768["scalars"]["input_layout"] == "SBH"
    # dropout also spans mid-D (spec's 6 kp<1 are not all high-D)
    assert any(b.get("scalars", {}).get("keep_prob", 1.0) < 1.0 and b["shape"][3] <= 512
               for b in bands), "expected at least one mid/low-D dropout band"


def _fa_like_schema(with_dropout_scalar):
    """A rank-4 attention schema; optionally declares the keep_prob scalar."""
    scalars = [{"name": "input_layout", "dtype": "str", "default": "BSH",
                "probe_values": ["BSH", "SBH", "BNSD"]}]
    if with_dropout_scalar:
        scalars.append({"name": "keep_prob", "dtype": "float", "default": 1.0,
                        "probe_values": [1.0, 0.9, 0.8]})
    return {"rank": 4, "op_class": "attention",
            "tensor_inputs": [{"name": "query", "shape_derive": None}],
            "scalar_inputs": scalars,
            "base_shape_filter": _bounded_fa_base}


def test_generate_cases_high_d_cases_get_dropout_when_declared():
    """When the schema declares keep_prob, the high-D bands' generated cases carry kp<1
    AND a non-BSH layout (the spec's high-D = dropout pairing).
    """
    cases = cg.generate_cases(_fa_like_schema(True), coverage_tier="sign_off",
                              dtype=torch.float16)
    hi = []
    for case in cases:
        if len(case["shape"]) != 4:
            continue
        if case["shape"][3] not in (640, 768):
            continue
        if case["inputs"].get("keep_prob", 1.0) < 1.0:
            hi.append(case)
    assert hi, "no high-D dropout case generated"
    kps = {c["inputs"]["keep_prob"] for c in cases}
    assert any(k < 1.0 for k in kps), "no kp<1 case at all"
    # the (640,BNSD,0.9) and (768,SBH,0.8) spec pairings are present
    pairs = {(c["shape"][3], c["inputs"].get("input_layout"), c["inputs"].get("keep_prob"))
             for c in cases if len(c["shape"]) == 4}
    assert (640, "BNSD", 0.9) in pairs, f"missing spec 640/BNSD/0.9: {sorted(p for p in pairs if p[0]==640)}"
    assert (768, "SBH", 0.8) in pairs, f"missing spec 768/SBH/0.8: {sorted(p for p in pairs if p[0]==768)}"


def test_per_plan_scalar_override_ignored_when_scalar_undeclared():
    """generic-safe: a schema WITHOUT keep_prob must not crash and must not inject an
    undeclared keep_prob — the band's scalar hint is silently dropped, shape still lands.
    """
    cases = cg.generate_cases(_fa_like_schema(False), coverage_tier="sign_off",
                              dtype=torch.float16)
    # high-D shapes still present (the SHAPE is never dropped)
    assert any(len(c["shape"]) == 4 and c["shape"][3] in (640, 768) for c in cases)
    # but keep_prob was never declared → never injected
    assert all("keep_prob" not in c["inputs"] for c in cases), \
        "undeclared keep_prob leaked into a case via the band override"


# ---------------------------------------------------------------------------
# increment-4: 2nd op-class (matmul) proves the mechanism generalizes — same
# registry, same per-plan override, different domain bands (M/N/K instead of D/S/B/N).
# ---------------------------------------------------------------------------
def test_matmul_bands_sweep_mnk_and_tails():
    bands = getattr(cg, '_matmul_shape_bands')("sign_off", torch.float16, 3)
    assert bands and all(len(b["shape"]) == 3 for b in bands), "matmul base must be rank-3 [M,N,K]"
    m_vals = {b["shape"][0] for b in bands}
    n_vals = {b["shape"][1] for b in bands}
    k_vals = {b["shape"][2] for b in bands}
    for v in (16, 64, 256, 512, 1024):
        assert v in m_vals and v in n_vals and v in k_vals, f"matmul M/N/K band {v} missing"
    names = {b["name"] for b in bands}
    assert any(n.startswith("mm_Mtail") for n in names) and any(n.startswith("mm_Ktail") for n in names)
    assert {"mm_tall_skinny", "mm_short_wide", "mm_deep_k"} <= names


def test_matmul_bands_rank_mismatch_returns_empty():
    assert getattr(cg, '_matmul_shape_bands')("sign_off", torch.float16, 4) == []
    assert getattr(cg, '_matmul_shape_bands')("sign_off", torch.float16, 2) == []


def test_dispatch_registered_matmul_aliases():
    assert len(getattr(cg, '_op_class_shape_bands')("matmul", "sign_off", torch.float16, 3)) > 0
    assert len(getattr(cg, '_op_class_shape_bands')("gemm", "sign_off", torch.float16, 3)) > 0
    assert len(getattr(cg, '_op_class_shape_bands')("GEMM", "sign_off", torch.float16, 3)) > 0
    # attention emitter is rank-4; matmul emitter is rank-3 — they don't cross-fire
    assert getattr(cg, '_op_class_shape_bands')("matmul", "sign_off", torch.float16, 4) == []
    assert getattr(cg, '_op_class_shape_bands')("attention", "sign_off", torch.float16, 3) == []


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
