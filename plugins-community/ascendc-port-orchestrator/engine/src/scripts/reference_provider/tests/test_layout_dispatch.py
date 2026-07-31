# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0aba_layout_dispatch (2026-05-07) — case_gen primitive for L4 attention-class
ops where the operator's input layout (BSH/SBH/BSND/BNSD/TND) is selected by
a string-typed scalar and dictates which permutation/collapse of the same
logical (B, S, N, D) dims a tensor uses.

Covers 4 of the 9 currently-blocked L4 ops: 3_FusionAttention,
5_LightningIndexer, 7_SparseFlashAttention, 10_FusedInferAttentionScore.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from case_gen import generate_cases, make_layout_dispatch  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests for the helper itself
# ---------------------------------------------------------------------------

def test_layout_dispatch_callable_layouts():
    """Standard usage — each layout is a callable(base) -> shape."""
    derive = make_layout_dispatch({
        "BSH": lambda s: [s[0], s[1], s[2] * s[3]],
        "BSND": lambda s: [s[0], s[1], s[2], s[3]],
    })
    base = [2, 64, 8, 32]
    assert derive(base, {"input_layout": "BSH"}) == [2, 64, 8 * 32]
    assert derive(base, {"input_layout": "BSND"}) == [2, 64, 8, 32]


def test_layout_dispatch_fixed_shape_layouts():
    """Fixed shape (list[int]) instead of callable — for layouts that
    don't depend on base.
    """
    derive = make_layout_dispatch({
        "FIXED_4D": [1, 2, 3, 4],
        "FIXED_3D": [10, 20, 30],
    })
    assert derive([99, 88, 77, 66], {"input_layout": "FIXED_4D"}) == [1, 2, 3, 4]
    assert derive([99, 88, 77, 66], {"input_layout": "FIXED_3D"}) == [10, 20, 30]


def test_layout_dispatch_unknown_layout_raises():
    """Layout not in the layouts dict → clear error (catches typo / missing)."""
    derive = make_layout_dispatch({"BSH": lambda s: list(s)[:3]})
    with pytest.raises(ValueError, match="not in declared layouts"):
        derive([2, 64, 8, 32], {"input_layout": "BNSD"})


def test_layout_dispatch_missing_scalar_raises():
    """scalar_name absent from scalars dict → clear error."""
    derive = make_layout_dispatch({"BSH": lambda s: list(s)[:3]})
    with pytest.raises(ValueError, match="scalar 'input_layout' missing"):
        derive([2, 64, 8, 32], {})


def test_layout_dispatch_custom_scalar_name():
    """Some L4 ops use 'layout_query' / 'layout_kv' instead of 'input_layout'."""
    derive_q = make_layout_dispatch(
        {"BSND": lambda s: [s[0], s[1], s[2], s[3]],
         "TND": lambda s: [s[0] * s[1], s[2], s[3]]},
        scalar_name="layout_query",
    )
    base = [2, 64, 8, 32]
    assert derive_q(base, {"layout_query": "BSND"}) == [2, 64, 8, 32]
    assert derive_q(base, {"layout_query": "TND"}) == [2 * 64, 8, 32]


def test_layout_dispatch_returns_list():
    """Result is always list (not tuple), per case_gen's strict check."""
    derive = make_layout_dispatch({"BSND": lambda s: tuple(s)})
    out = derive([1, 2, 3, 4], {"input_layout": "BSND"})
    assert isinstance(out, list)
    assert out == [1, 2, 3, 4]


def test_layout_dispatch_invalid_spec_raises():
    """Non-list non-callable spec → clear error."""
    derive = make_layout_dispatch({"BAD": "not a shape"})
    with pytest.raises(ValueError, match="must be list\\[int\\] or callable"):
        derive([1, 2, 3, 4], {"input_layout": "BAD"})


# ---------------------------------------------------------------------------
# Engine-integration test: 2-arg shape_derive + string scalar through generate_cases
# ---------------------------------------------------------------------------

def test_generate_cases_with_layout_dispatch_integration():
    """End-to-end: generate_cases honors make_layout_dispatch for a 4-layout
    attention-style schema. Verifies produced cases have correctly-shaped
    tensors per the layout scalar in each case.
    """
    layouts_qkv = {
        "BSH": lambda s: [s[0], s[1], s[2] * s[3]],
        "SBH": lambda s: [s[1], s[0], s[2] * s[3]],
        "BSND": lambda s: [s[0], s[1], s[2], s[3]],
        "BNSD": lambda s: [s[0], s[2], s[1], s[3]],
    }
    derive = make_layout_dispatch(layouts_qkv)

    schema = {
        "op_name": "test_layout_dispatch",
        "formula": "test",
        "tensor_inputs": [
            {"name": "query", "role": "operand", "shape_derive": derive},
            {"name": "key", "role": "operand", "shape_derive": derive},
            {"name": "value", "role": "operand", "shape_derive": derive},
        ],
        "scalar_inputs": [
            {"name": "input_layout", "dtype": "str",
             "default": "BSH",
             "probe_values": ["BSH", "SBH", "BSND", "BNSD"]},
        ],
        "tensor_output": "out",
        "rank": 4,  # base_shape is [B, S, N, D]
        "base_shape_filter": (lambda b: len(b) == 4 and all(d >= 1 for d in b)),
    }

    cases = generate_cases(schema, coverage_tier="pilot", dtype=torch.float16)
    assert len(cases) > 0, "should generate at least one case"

    # For each case, verify the query/key/value tensor shapes match what
    # the layout_dispatch should have produced for that case's base_shape +
    # input_layout scalar value.
    for c in cases:
        layout = c["inputs"]["input_layout"]
        base = c["shape"]
        assert len(base) == 4, f"base must be rank-4, got {base}"
        expected = layouts_qkv[layout](base)
        for tname in ("query", "key", "value"):
            actual = list(c["inputs"][tname].shape)
            assert actual == expected, (
                f"case {c['idx']} layout={layout} base={base}: tensor {tname} "
                f"shape {actual} != expected {expected}"
            )


def test_generate_cases_layout_dispatch_distinct_per_case():
    """Different cases with different layout values should have different
    tensor shapes (proves the dispatch IS varying, not constant).
    """
    derive = make_layout_dispatch({
        "RANK3": lambda s: [s[0], s[1], s[2] * s[3]],  # collapse N*D into H
        "RANK4": lambda s: list(s),
    })
    schema = {
        "op_name": "test_distinct",
        "formula": "test",
        "tensor_inputs": [
            {"name": "x", "role": "operand", "shape_derive": derive},
        ],
        "scalar_inputs": [
            {"name": "input_layout", "dtype": "str",
             "default": "RANK4",
             "probe_values": ["RANK3", "RANK4"]},
        ],
        "tensor_output": "out",
        "rank": 4,
        "base_shape_filter": (lambda b: len(b) == 4 and all(d >= 1 for d in b)),
    }
    cases = generate_cases(schema, coverage_tier="pilot", dtype=torch.float16)
    rank3_cases = [c for c in cases if c["inputs"]["input_layout"] == "RANK3"]
    rank4_cases = [c for c in cases if c["inputs"]["input_layout"] == "RANK4"]
    assert rank3_cases, "should have at least one RANK3 case"
    assert rank4_cases, "should have at least one RANK4 case"
    for c in rank3_cases:
        assert c["inputs"]["x"].dim() == 3
    for c in rank4_cases:
        assert c["inputs"]["x"].dim() == 4
