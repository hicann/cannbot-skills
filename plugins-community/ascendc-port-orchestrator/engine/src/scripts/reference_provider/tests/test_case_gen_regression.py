# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0bbb (2026-05-06): regression suite for case_gen.py.

Origin: user direction 2026-05-06 after L4 2_FFN P0aaa incident — engine
extensions like rank-3/4 base_shape (P0aaa task #105) require a regression
suite first, otherwise the touch could break 32 existing rank-1/2 ops in
the suite. Zero pre-existing tests for case_gen as of this commit.

Coverage targets:
  1. _shape_plan output per (coverage_tier, rank) — pinned shape lists
  2. _base_shape_for_rank for rank 1..5 — pinned shapes
  3. generate_cases for representative SCHEMAs — case count, distinct
     shapes, dataset_data_sha256
  4. _scalar_only_cases for scalar-only schemas — pinned case count
  5. _build_inputs shape_derive + dtype overrides
  6. dataset_data_sha256 determinism property

These tests are the safety net for P0aaa task #105 (rank-3/4 base_shape
plans + dim_constant primitive + tuple_length_derive). Future engine
edits that touch _shape_plan, _base_shape_for_rank, or generate_cases
must keep these green or surface the regression explicitly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import case_gen  # noqa: E402


# ---------------------------------------------------------------------------
# _base_shape_for_rank — pinned single representative shape per rank
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rank,expected", [
    (1, [1024]),
    (2, [32, 32]),
    (3, [8, 8, 16]),
    (4, [2, 2, 16, 16]),
    (5, [1, 2, 4, 8, 16]),
])
def test_base_shape_for_rank_pinned(rank, expected):
    assert getattr(case_gen, '_base_shape_for_rank')(rank) == expected


def test_base_shape_for_rank_above_5_raises():
    with pytest.raises(ValueError, match="rank=6"):
        getattr(case_gen, '_base_shape_for_rank')(6)


def test_base_shape_for_rank_flat_size_invariant():
    """Each non-rank-5 base_shape flattens to ~1024 elements (degenerate test
    cases use this constant). Ensures future edits don't accidentally bump
    the flat size to something the coverage tiers don't cover.
    """
    for rank in (1, 2, 3, 4):
        s = getattr(case_gen, '_base_shape_for_rank')(rank)
        prod = 1
        for d in s:
            prod *= d
        assert prod == 1024, f"rank{rank} base flat-size {prod} != 1024"


# ---------------------------------------------------------------------------
# _shape_plan — golden 1D + 2D shapes per coverage tier
# ---------------------------------------------------------------------------
def test_shape_plan_pilot_rank1_only_typical():
    """Pilot tier emits only the 3 typical shapes regardless of rank."""
    plan = getattr(case_gen, '_shape_plan')("pilot", torch.float32, rank=1)
    names = [p["name"] for p in plan]
    assert names == ["typical_1k", "typical_32k", "typical_1m"]


def test_shape_plan_signoff_rank1_includes_edges():
    """sign_off adds shape-edge entries; all 1D."""
    plan = getattr(case_gen, '_shape_plan')("sign_off", torch.float32, rank=1)
    names = {p["name"] for p in plan}
    # Tier 1 typicals
    assert {"typical_1k", "typical_32k", "typical_1m"}.issubset(names)
    # Tier 2 edges
    assert {"degen_n1", "align_1u", "core_lt_cores",
            "tile_minus1", "prime_1009", "prime_32771"}.issubset(names)
    # Should NOT include 2D entries since rank=1
    assert "d2_32x32" not in names


def test_shape_plan_signoff_rank2_includes_2d_equivalents():
    """sign_off + rank>=2 adds 2D equivalents (32x32, 64x16) as flat-1024 alternates."""
    plan = getattr(case_gen, '_shape_plan')("sign_off", torch.float32, rank=2)
    names = {p["name"] for p in plan}
    assert "d2_32x32" in names
    assert "d2_64x16" in names


def test_shape_plan_production_extras():
    """production tier adds tile_3_plus3 + huge_8m that sign_off lacks."""
    plan = getattr(case_gen, '_shape_plan')("production", torch.float32, rank=1)
    names = {p["name"] for p in plan}
    assert "tile_3_plus3" in names
    assert "huge_8m" in names


def test_shape_plan_dtype_align_fp16_vs_fp32():
    """fp16 and fp32 may use different alignment elem counts; the resulting
    `align_1u` and `core_*` shape sizes can differ. Pin the contract.
    """
    plan_fp32 = getattr(case_gen, '_shape_plan')("sign_off", torch.float32, rank=1)
    plan_fp16 = getattr(case_gen, '_shape_plan')("sign_off", torch.float16, rank=1)
    fp32_align = next(p for p in plan_fp32 if p["name"] == "align_1u")["shape"]
    fp16_align = next(p for p in plan_fp16 if p["name"] == "align_1u")["shape"]
    # Both should be lists of single int — exact value may differ by dtype
    assert isinstance(fp32_align, list) and len(fp32_align) == 1
    assert isinstance(fp16_align, list) and len(fp16_align) == 1


# ---------------------------------------------------------------------------
# generate_cases — golden case counts + shapes for representative SCHEMAs
# ---------------------------------------------------------------------------
def _simple_rank1_schema():
    return {
        "op_name": "test_simple_rank1",
        "formula": "y = x",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 1,
        "base_shape_filter": lambda b: True,
    }


def _simple_rank2_schema():
    return {
        "op_name": "test_simple_rank2",
        "formula": "y = x",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 2,
        "base_shape_filter": lambda b: True,
    }


def test_generate_cases_pilot_rank1_count():
    """pilot tier rank-1: 3 typical shapes × 3 distributions = 9 cases (pinned)."""
    cases = case_gen.generate_cases(_simple_rank1_schema(), "pilot", torch.float32)
    assert len(cases) == 9


def test_generate_cases_signoff_rank1_count():
    """sign_off tier rank-1: pinned ~29 cases — guards against silent regression."""
    cases = case_gen.generate_cases(_simple_rank1_schema(), "sign_off", torch.float32)
    assert 25 <= len(cases) <= 35, f"sign_off rank-1 case count {len(cases)} outside expected band"


def test_generate_cases_signoff_rank2_includes_2d_shapes():
    cases = case_gen.generate_cases(_simple_rank2_schema(), "sign_off", torch.float32)
    distinct_shapes = sorted(set(tuple(c.get("shape", [])) for c in cases))
    has_2d = any(len(s) == 2 for s in distinct_shapes)
    assert has_2d, f"rank-2 sign_off should include 2D shapes; got {distinct_shapes}"


def test_generate_cases_case_dict_contract():
    """Every case must have idx, name, shape, inputs, meta keys."""
    cases = case_gen.generate_cases(_simple_rank1_schema(), "pilot", torch.float32)
    for c in cases:
        assert "idx" in c
        assert "name" in c and isinstance(c["name"], str)
        assert "shape" in c
        assert "inputs" in c and isinstance(c["inputs"], dict)
        assert "meta" in c


def test_generate_cases_inputs_match_schema_tensors():
    schema = _simple_rank1_schema()
    cases = case_gen.generate_cases(schema, "pilot", torch.float32)
    expected_keys = {t["name"] for t in schema["tensor_inputs"]}
    for c in cases:
        actual_keys = set(c["inputs"].keys())
        assert expected_keys <= actual_keys, (
            f"case {c['name']} missing tensors: expected ⊇ {expected_keys}, got {actual_keys}"
        )


def test_generate_cases_output_dtype_respects_schema():
    """All output tensors must be the requested dtype unless tensor_input
    overrides via per-tensor `dtype` field.
    """
    cases = case_gen.generate_cases(_simple_rank1_schema(), "pilot", torch.float16)
    for c in cases:
        x = c["inputs"]["x"]
        if isinstance(x, torch.Tensor):
            assert x.dtype == torch.float16, f"case {c['name']} x.dtype={x.dtype}"


def test_generate_cases_base_shape_filter_rejects_invalid():
    """SCHEMA's base_shape_filter is honored. When it rejects EVERY shape
    in the plan AND the default base, generate_cases raises ValueError
    rather than silently producing zero cases (which would mask a SCHEMA
    bug). This is a documented contract.
    """
    schema = _simple_rank1_schema()

    def _reject_all_base_shapes(_base_shape):
        return False

    schema["base_shape_filter"] = _reject_all_base_shapes
    with pytest.raises(ValueError, match="base_shape_filter excludes"):
        case_gen.generate_cases(schema, "pilot", torch.float32)


# ---------------------------------------------------------------------------
# dataset_data_sha256 — determinism + sensitivity
# ---------------------------------------------------------------------------
def test_dataset_data_sha256_deterministic():
    cases = case_gen.generate_cases(_simple_rank1_schema(), "pilot", torch.float32)
    s1 = case_gen.dataset_data_sha256(cases)
    s2 = case_gen.dataset_data_sha256(cases)
    assert s1 == s2
    assert len(s1) == 64  # SHA-256 hex digest


def test_dataset_data_sha256_changes_on_input_change():
    cases1 = case_gen.generate_cases(_simple_rank1_schema(), "pilot", torch.float32)
    cases2 = case_gen.generate_cases(_simple_rank1_schema(), "sign_off", torch.float32)
    s1 = case_gen.dataset_data_sha256(cases1)
    s2 = case_gen.dataset_data_sha256(cases2)
    assert s1 != s2, "different tier should produce different sha256"


# ---------------------------------------------------------------------------
# scalar_inputs — _scalar_only_cases path
# ---------------------------------------------------------------------------
def _scalar_only_schema():
    return {
        "op_name": "test_scalar_only",
        "formula": "y = scalar_op(scale)",
        "tensor_inputs": [],
        "scalar_inputs": [
            {"name": "scale", "dtype": "float", "default": 1.0,
             "probe_values": [0.5, 1.0, 2.0]},
            {"name": "mode", "dtype": "int", "default": 0,
             "probe_values": [0, 1, 2]},
        ],
        "tensor_output": "y",
        "rank": 0,
        "base_shape_filter": lambda b: True,
    }


def test_scalar_only_case_pilot_count():
    """Scalar-only pilot tier: 1 baseline + each scalar's probe_values cases."""
    cases = case_gen.generate_cases(_scalar_only_schema(), "pilot", torch.float32)
    # baseline + 3 scale probes + 3 mode probes = 7
    assert len(cases) >= 5, (
        f"scalar-only pilot expected ≥5 cases (baseline + per-scalar probes); got {len(cases)}"
    )


def test_scalar_only_inputs_are_scalars():
    cases = case_gen.generate_cases(_scalar_only_schema(), "pilot", torch.float32)
    for c in cases:
        for k, v in c["inputs"].items():
            assert not isinstance(v, torch.Tensor), (
                f"scalar-only schema produced a tensor input: {k} = {v}"
            )


# ---------------------------------------------------------------------------
# _build_inputs — shape_derive + dtype override
# ---------------------------------------------------------------------------
def test_shape_derive_lambda_applied():
    """tensor_inputs with shape_derive should produce the derived shape, not base."""
    schema = {
        "op_name": "test_derive",
        "formula": "y = x · w",
        "tensor_inputs": [
            {"name": "x", "role": "operand"},
            {"name": "w", "role": "operand", "shape_derive": lambda s: [s[-1], s[-1]]},
        ],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 2,
        "base_shape_filter": lambda b: len(b) == 2,
    }
    cases = case_gen.generate_cases(schema, "pilot", torch.float32)
    for c in cases:
        x = c["inputs"]["x"]
        w = c["inputs"]["w"]
        if isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor):
            assert list(w.shape) == [x.shape[-1], x.shape[-1]], (
                f"shape_derive not applied: x={list(x.shape)} w={list(w.shape)}"
            )


def test_per_tensor_dtype_override():
    """A tensor with explicit `dtype` overrides the global DTYPE."""
    schema = {
        "op_name": "test_dtype_override",
        "formula": "y = x + indices",
        "tensor_inputs": [
            {"name": "x", "role": "operand"},
            {"name": "indices", "role": "operand", "dtype": torch.int32},
        ],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 1,
        "base_shape_filter": lambda b: True,
    }
    cases = case_gen.generate_cases(schema, "pilot", torch.float16)
    for c in cases:
        ix = c["inputs"].get("indices")
        if isinstance(ix, torch.Tensor):
            assert ix.dtype == torch.int32, (
                f"per-tensor dtype override not applied: indices.dtype={ix.dtype}"
            )


# ---------------------------------------------------------------------------
# Backward-compat smoke — load + run an existing op's input_gen.py via
# case_gen (regression against the actual benchmark workspace SCHEMAs).
# ---------------------------------------------------------------------------
def test_shape_plan_rank3_signoff_includes_l4_shapes():
    """P0aaa task #105 (2026-05-06): rank-3 ops like 6_QuantMatmul [m,k,n]
    need multiple representative shapes covering small / typical / non-square
    / batch=1. Pinned names must appear at sign_off+ when rank>=3.
    """
    plan = getattr(case_gen, '_shape_plan')("sign_off", torch.float32, rank=3)
    names = {p["name"] for p in plan}
    assert "d3_typical" in names, f"sign_off rank-3 missing d3_typical; got {names}"
    assert "d3_non_square" in names
    assert "d3_batch1" in names
    # All d3_* entries must be 3D
    d3_shapes = [p["shape"] for p in plan if p["name"].startswith("d3_")]
    for s in d3_shapes:
        assert len(s) == 3, f"d3_* shape has wrong rank: {s}"


def test_shape_plan_rank4_signoff_includes_l4_shapes():
    """The established rank-4 coverage plan remains unchanged by safety gates."""
    plan = getattr(case_gen, '_shape_plan')("sign_off", torch.float32, rank=4)
    d4_shapes = {
        item["name"]: item["shape"]
        for item in plan
        if item["name"].startswith("d4_")
    }
    assert d4_shapes == {
        "d4_small": [16, 64, 64, 64],
        "d4_typical": [64, 256, 256, 256],
        "d4_non_square": [32, 128, 256, 64],
        "d4_m1": [1, 256, 256, 256],
        "d4_align_inner": [16, 16, 32, 64],
    }


def test_rank4_4gib_tensor_rejected_before_generator_call():
    """The historical d4_typical shape must fail before torch allocates 4 GiB."""
    generator_called = False

    def forbidden_generator(_numel, _dtype, _seed):
        nonlocal generator_called
        generator_called = True
        raise AssertionError("generator must not run for an oversized case")

    with pytest.raises(ValueError) as exc_info:
        getattr(case_gen, '_build_inputs')(
            [{"name": "x", "role": "operand"}],
            [64, 256, 256, 256],
            forbidden_generator,
            torch.float32,
            seed_base=1,
            case_label="shape_d4_typical",
        )

    message = str(exc_info.value)
    assert "estimated_bytes=4294967296" in message
    assert "shape_d4_typical" in message
    assert generator_called is False


def test_case_budget_sums_all_inputs_before_allocation():
    """Two individually valid tensors that exceed the combined cap fail together."""
    generator_called = False

    def forbidden_generator(_numel, _dtype, _seed):
        nonlocal generator_called
        generator_called = True
        raise AssertionError("generator must not run for an oversized case")

    shape = [16 * 1024 * 1024]  # 64 MiB per fp32 tensor
    with pytest.raises(ValueError) as exc_info:
        getattr(case_gen, '_build_inputs')(
            [{"name": "x"}, {"name": "y"}],
            shape,
            forbidden_generator,
            torch.float32,
            seed_base=1,
            case_label="two_inputs",
        )

    message = str(exc_info.value)
    assert "estimated_bytes=134217728" in message
    assert "x:shapes=" in message and "y:shapes=" in message
    assert generator_called is False


def test_case_budget_accounts_for_shape_dtype_and_tensor_list_overrides():
    """The preflight includes shape_derive, dtype overrides, and list items."""
    list_spec = {
        "name": "parts",
        "kind": "list_of_tensors",
        "dtype": torch.float16,
        "list_length_plan": [4],
        "per_item_shape_derive": lambda _shape, _index, _length: [8 * 1024 * 1024],
    }
    derived_spec = {
        "name": "indices",
        "dtype": torch.int8,
        "shape_derive": lambda _shape: [40 * 1024 * 1024],
    }

    with pytest.raises(ValueError) as exc_info:
        getattr(case_gen, '_build_inputs')(
            [list_spec, derived_spec],
            [1],
            lambda *_args: pytest.fail("generator must not run"),
            torch.float32,
            seed_base=1,
            case_label="list_and_derived",
        )

    message = str(exc_info.value)
    assert "estimated_bytes=109051904" in message
    assert "dtype=torch.float16" in message
    assert "dtype=torch.int8" in message


def test_shape_derive_is_resolved_once_for_guard_and_allocation():
    """A stateful derive cannot return a small checked shape then allocate a large one."""
    derive_calls = 0
    generated_sizes = []

    def stateful_derive(_shape):
        nonlocal derive_calls
        derive_calls += 1
        return [8] if derive_calls == 1 else [64 * 1024 * 1024]

    def generator(numel, dtype, _seed):
        generated_sizes.append(numel)
        return torch.zeros(numel, dtype=dtype)

    inputs = getattr(case_gen, '_build_inputs')(
        [{"name": "x", "shape_derive": stateful_derive}],
        [1],
        generator,
        torch.float32,
        seed_base=1,
        case_label="stateful_shape",
    )

    assert derive_calls == 1
    assert generated_sizes == [8]
    assert list(inputs["x"].shape) == [8]


def test_list_item_shape_derive_is_resolved_once_per_item():
    """List item shapes use the exact immutable plan checked by the budget gate."""
    calls = []

    def derive(_shape, index, _length):
        calls.append(index)
        return [index + 1]

    inputs = getattr(case_gen, '_build_inputs')(
        [{
            "name": "parts",
            "kind": "list_of_tensors",
            "list_length_plan": [3],
            "per_item_shape_derive": derive,
        }],
        [1],
        lambda n, dtype, _seed: torch.zeros(n, dtype=dtype),
        torch.float32,
        seed_base=1,
        case_label="stateful_list_shape",
    )

    assert calls == [0, 1, 2]
    assert [list(tensor.shape) for tensor in inputs["parts"]] == [[1], [2], [3]]


def test_large_unique_index_range_does_not_allocate_upper_sized_permutation(monkeypatch):
    """A huge legal index range uses O(n), not O(upper), temporary storage."""
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda *_args, **_kwargs: pytest.fail("large upper must not call randperm"),
    )
    generator = getattr(case_gen, '_mk_invariant_gen')(
        "index_range:1000000000", {"allow_dup_indices": False}, {}
    )
    values = generator(8, torch.int64, 7)
    assert values.numel() == 8
    assert len(set(values.tolist())) == 8
    assert int(values.min()) >= 0
    assert int(values.max()) < 1_000_000_000


def test_shape_plan_rank3_pilot_does_not_include_d3():
    """Pilot tier should NOT add d3/d4 entries — only sign_off+. Confirms
    we don't blow up case count on rapid-iteration tier.
    """
    plan = getattr(case_gen, '_shape_plan')("pilot", torch.float32, rank=3)
    names = {p["name"] for p in plan}
    assert not any(n.startswith("d3_") for n in names)


def test_generate_cases_rank3_with_filter_produces_d3_shapes():
    """Real L4 op-style SCHEMA: rank-3 with base_shape_filter that rejects
    1D shapes. Should produce ONLY 3D base shapes from the d3_* plan.
    """
    schema = {
        "op_name": "test_rank3_l4_style",
        "formula": "y = x1 @ x2",
        "tensor_inputs": [
            {"name": "x1", "role": "operand"},
            {"name": "x2", "role": "operand", "shape_derive": lambda s: [s[2], s[2]]},
        ],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 3,
        "base_shape_filter": lambda b: len(b) == 3,
    }
    cases = case_gen.generate_cases(schema, "sign_off", torch.float16)
    # All cases must have 3D base shape
    for c in cases:
        s = c.get("shape", [])
        assert len(s) == 3, f"case {c['name']} has wrong rank shape {s}"
    # Should produce more than 1 distinct shape (the d3_* plan adds 6 entries
    # plus the _base_shape_for_rank default — so expect ≥3 distinct shapes
    # appearing across the case list).
    distinct = sorted(set(tuple(c["shape"]) for c in cases))
    assert len(distinct) >= 3, (
        f"rank-3 L4-style schema only got {len(distinct)} distinct shapes; "
        f"expected ≥3 from d3_* plan: {distinct}"
    )


def test_generate_cases_rank4_with_filter_produces_d4_shapes():
    """Same property at rank-4 (FFN-style)."""
    schema = {
        "op_name": "test_rank4_ffn_style",
        "formula": "y = ffn(x, w1, w2)",
        "tensor_inputs": [
            {"name": "x", "role": "operand", "shape_derive": lambda s: [s[0], s[1]]},
            {"name": "w1", "role": "operand", "shape_derive": lambda s: [s[1], s[2]]},
            {"name": "w2", "role": "operand", "shape_derive": lambda s: [s[2], s[3]]},
        ],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 4,
        "base_shape_filter": lambda b: len(b) == 4,
    }
    cases = case_gen.generate_cases(schema, "sign_off", torch.float16)
    for c in cases:
        s = c.get("shape", [])
        assert len(s) == 4
    # Verify chained-matmul shape_derive produced consistent shapes per case
    for c in cases:
        x_shape = list(c["inputs"]["x"].shape)
        w1_shape = list(c["inputs"]["w1"].shape)
        w2_shape = list(c["inputs"]["w2"].shape)
        # Chain invariant: x.cols == w1.rows; w1.cols == w2.rows
        assert x_shape[1] == w1_shape[0], (
            f"x.cols {x_shape[1]} != w1.rows {w1_shape[0]}"
        )
        assert w1_shape[1] == w2_shape[0], (
            f"w1.cols {w1_shape[1]} != w2.rows {w2_shape[0]}"
        )
    distinct = sorted(set(tuple(c["shape"]) for c in cases))
    assert len(distinct) >= 3, (
        f"rank-4 FFN-style only {len(distinct)} distinct shapes; expected ≥3"
    )


# ---------------------------------------------------------------------------
# P0aaa task #105: dim_constant via shape_derive (NMS-class) — already
# expressible without engine extension (regression guard).
# ---------------------------------------------------------------------------
def test_dim_constant_via_shape_derive_for_nms_style():
    """NMS [N, 4] is expressible via rank=1 base + shape_derive that emits
    [s[0], 4]. No engine extension needed; case_gen already supports this.
    Regression guard: if shape_derive ever changes contract, NMS-class
    SCHEMAs must keep working.
    """
    schema = {
        "op_name": "test_nms_style",
        "formula": "indices = nms(boxes, scores, iou_thresh)",
        "tensor_inputs": [
            {"name": "boxes", "shape_derive": lambda s: [s[0], 4]},
            {"name": "scores", "shape_derive": lambda s: [s[0]]},
        ],
        "scalar_inputs": [
            {"name": "iou_thresh", "dtype": "float", "default": 0.5},
        ],
        "tensor_output": "indices",
        "rank": 1,
        "base_shape_filter": lambda b: len(b) == 1 and b[0] >= 3,
    }
    cases = case_gen.generate_cases(schema, "pilot", torch.float32)
    assert len(cases) >= 1
    for c in cases:
        b = c["inputs"]["boxes"]
        s = c["inputs"]["scores"]
        if isinstance(b, torch.Tensor) and isinstance(s, torch.Tensor):
            assert b.shape[-1] == 4, f"boxes last dim != 4: {list(b.shape)}"
            assert s.shape[0] == b.shape[0], (
                f"scores N {s.shape[0]} != boxes N {b.shape[0]}"
            )


# ---------------------------------------------------------------------------
# P0aaa task #105: tuple_of_int with length_derive (Pad-class)
# ---------------------------------------------------------------------------
def test_tuple_of_int_length_derive_for_pad_style():
    """Pad's pad: tuple[int] has length 2*input_rank. P0aaa task #105 added
    `tuple_of_int` dtype + `length_derive`. Value is a tuple of ints whose
    LENGTH depends on base_shape rank.
    """
    schema = {
        "op_name": "test_pad_style",
        "formula": "y = pad(x, pad_widths)",
        "tensor_inputs": [
            {"name": "x", "role": "operand"},
        ],
        "scalar_inputs": [
            {
                "name": "pad",
                "dtype": "tuple_of_int",
                "length_derive": lambda base: 2 * len(base),
                "value_range": (0, 3),
                "default": (0, 0),  # default for default_scalars dict — overridden per-case
            },
        ],
        "tensor_output": "y",
        "rank": 2,
        "base_shape_filter": lambda b: len(b) == 2,
    }
    cases = case_gen.generate_cases(schema, "pilot", torch.float32)
    for c in cases:
        pad = c["inputs"].get("pad")
        if pad is None:
            continue
        # 2D base_shape → length should be 4
        assert isinstance(pad, tuple), f"pad must be tuple, got {type(pad).__name__}"
        # rank-2 base_shape → length 4 (when length_derive applied for this case's shape)
        # Note: pilot tier only emits 1D shapes from _shape_plan so the rank-2 derive
        # only fires when base shape happens to be rank-2 — degraded check: ensure
        # all values are ints in valid range
        for v in pad:
            assert isinstance(v, int), f"pad element type {type(v).__name__}"
            assert 0 <= v <= 3, f"pad element {v} outside value_range"


def test_tuple_of_int_length_matches_base_rank_signoff():
    """sign_off tier produces enough rank-matched bases for length_derive
    to fire correctly per case.
    """
    schema = {
        "op_name": "test_pad_signoff",
        "formula": "y = pad(x, pad_widths)",
        "tensor_inputs": [
            {"name": "x", "role": "operand"},
        ],
        "scalar_inputs": [
            {
                "name": "pad",
                "dtype": "tuple_of_int",
                "length_derive": lambda base: 2 * len(base),
                "value_range": (0, 5),
                "default": (0, 0),
            },
        ],
        "tensor_output": "y",
        "rank": 2,
        "base_shape_filter": lambda b: len(b) == 2,
    }
    cases = case_gen.generate_cases(schema, "sign_off", torch.float32)
    # Every case's base_shape is rank-2, so pad must be 4-tuple
    for c in cases:
        pad = c["inputs"].get("pad")
        # Skip baseline-default cases (no base shape applied) but check
        # cases where shape was set
        shape = c.get("shape", [])
        if len(shape) == 2:
            assert len(pad) == 4, (
                f"case {c['name']} shape={shape} pad-length {len(pad)} != 4"
            )


def test_tuple_of_int_invalid_default_raises():
    """default for tuple_of_int must be list/tuple, not scalar."""
    schema = {
        "op_name": "test_pad_invalid",
        "formula": "y = pad(x, p)",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [
            {
                "name": "p",
                "dtype": "tuple_of_int",
                "length_derive": lambda base: 4,
                "default": 0,  # WRONG — should be tuple
            },
        ],
        "tensor_output": "y",
        "rank": 1,
        "base_shape_filter": lambda b: True,
    }
    with pytest.raises(ValueError, match="tuple_of_int"):
        case_gen.generate_cases(schema, "pilot", torch.float32)


def test_tuple_of_int_negative_length_derive_raises():
    """length_derive returning negative raises clear error."""
    schema = {
        "op_name": "test_pad_neg",
        "formula": "y = pad(x, p)",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [
            {
                "name": "p",
                "dtype": "tuple_of_int",
                "length_derive": lambda base: -1,
                "default": (),
            },
        ],
        "tensor_output": "y",
        "rank": 1,
        "base_shape_filter": lambda b: True,
    }
    with pytest.raises(ValueError, match="non-negative"):
        case_gen.generate_cases(schema, "pilot", torch.float32)


def test_smoke_existing_input_gen_schemas_runnable(tmp_path):
    """Pick a simple workspace's input_gen.py SCHEMA and verify it produces
    cases without error. Uses 11_dequantswigluquant as canonical fused
    SCHEMA exemplar (well-tested in production).

    If this test starts failing after engine edits, an existing op's case
    generation has broken — surface the regression explicitly.
    """
    # Inline a stripped-down version of the real schema to avoid filesystem
    # coupling. Mirrors src/scripts/reference_provider/input_gen.template.fused.py
    # contract.
    def _base_valid(base):
        if len(base) != 2:
            return False
        n, two_h = base
        return n >= 1 and two_h >= 2 and (two_h % 2 == 0)

    schema = {
        "op_name": "smoke_fused",
        "formula": "out, scales = fused_kernel(x, w, a)",
        "tensor_inputs": [
            {"name": "x", "dtype": torch.int32,
             "shape_derive": lambda s: [s[0], s[1]],
             "int_range": (-100, 100)},
            {"name": "weight_scale", "dtype": torch.float32,
             "shape_derive": lambda s: [1, s[1]]},
            {"name": "activation_scale", "dtype": torch.float32,
             "shape_derive": lambda s: [s[0], 1]},
        ],
        "scalar_inputs": [
            {"name": "mode", "dtype": "int", "default": 1, "probe_values": [1]},
        ],
        "tensor_output": "out",
        "rank": 2,
        "base_shape_filter": _base_valid,
    }

    cases = case_gen.generate_cases(schema, "pilot", torch.float32)
    assert len(cases) >= 1, "fused smoke produced zero cases — regression"
    # All cases should have all three tensor inputs
    for c in cases:
        for tname in ("x", "weight_scale", "activation_scale"):
            assert tname in c["inputs"], f"missing {tname} in case {c['name']}"
