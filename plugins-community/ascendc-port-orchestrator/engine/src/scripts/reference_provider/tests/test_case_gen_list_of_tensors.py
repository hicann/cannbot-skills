# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for case_gen V1.7 `list_of_tensors` primitive (2026-05-21).

Closes the 13_Cat BLOCK class documented in SKILL.md V1 "Known limitations" #1.
Without this primitive, ops with `forward(tensors: list[Tensor], ...)`
signatures (cat / stack / block_diag class) can't be expressed in SCHEMA
and Phase O2.5 BLOCKs.

Coverage:
- list_length_plan single-value sweep
- list_length_plan multi-value sweep — case count = base * Π plans
- per-item uniform shape (default behavior)
- per-item non-uniform shape derive (3-arg form)
- per-item non-uniform shape derive (4-arg form with scalars)
- dataset hash protocol handles list-of-Tensors inputs
- per-case hash stable across runs (same seed → same bytes)
- error: list_length_plan missing
- error: per_item_shape_derive returns invalid type
- mixing list_of_tensors with normal tensor inputs in same SCHEMA
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
_RP = _HERE.parent.parent
sys.path.insert(0, str(_RP))

import case_gen  # noqa: E402


def _minimal_schema(plan, derive=None):
    schema = {
        "op_name": "cat_test",
        "formula": "torch.cat(tensors, dim=dim)",
        "tensor_inputs": [
            {
                "name": "tensors",
                "kind": "list_of_tensors",
                "list_length_plan": plan,
            },
        ],
        "scalar_inputs": [
            {"name": "dim", "dtype": "int", "default": 0, "probe_values": [0]},
        ],
        "tensor_output": "out",
        "rank": 1,
    }
    if derive is not None:
        schema["tensor_inputs"][0]["per_item_shape_derive"] = derive
    return schema


def test_list_of_tensors_single_length():
    schema = _minimal_schema([3])
    cases = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    assert len(cases) > 0
    for c in cases:
        assert isinstance(c["inputs"]["tensors"], list)
        assert len(c["inputs"]["tensors"]) == 3
        assert all(isinstance(t, torch.Tensor) for t in c["inputs"]["tensors"])


def test_list_of_tensors_multi_length_multiplies_case_count():
    # Two runs: plan=[2] then plan=[2,3,4] — second should produce 3x cases.
    cases_one = case_gen.generate_cases(_minimal_schema([2]),
                                          coverage_tier="pilot", dtype=torch.float32)
    cases_three = case_gen.generate_cases(_minimal_schema([2, 3, 4]),
                                            coverage_tier="pilot", dtype=torch.float32)
    assert len(cases_three) == len(cases_one) * 3
    seen_lengths = sorted({c["meta"].get("list_length_tensors") for c in cases_three})
    assert seen_lengths == [2, 3, 4]


def test_per_item_uniform_shape_default():
    """Without per_item_shape_derive, all list items have base_shape."""
    schema = _minimal_schema([3])
    cases = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    for c in cases:
        item_shapes = [list(t.shape) for t in c["inputs"]["tensors"]]
        assert all(s == item_shapes[0] for s in item_shapes), \
            f"expected uniform, got {item_shapes}"


def test_per_item_shape_derive_3arg():
    """3-arg per_item_shape_derive(base_shape, i, N) — varies along dim 0."""
    def derive(base_shape, i, num_items):
        s = list(base_shape)
        s[0] = s[0] + i * 16
        return s
    schema = _minimal_schema([2, 3], derive=derive)
    cases = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    for c in cases:
        items = c["inputs"]["tensors"]
        num_items = len(items)
        for i in range(1, num_items):
            assert items[i].shape[0] == items[0].shape[0] + i * 16, \
                f"per-item stride broken; got {[list(t.shape) for t in items]}"


def test_per_item_shape_derive_4arg_with_scalars():
    """4-arg derive(base_shape, i, N, scalars) — uses a scalar to size items."""
    def derive(base_shape, i, num_items, scalars):
        # depend on per-case 'dim' to vary which axis grows
        s = list(base_shape)
        d = int(scalars.get("dim", 0))
        s[d] = s[d] + i * 8
        return s
    schema = _minimal_schema([3], derive=derive)
    cases = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    assert len(cases) > 0
    for c in cases:
        items = c["inputs"]["tensors"]
        # each item differs from item 0 along SOME axis
        for i in range(1, len(items)):
            assert items[i].shape != items[0].shape, \
                f"4-arg derive should differ; got {[list(t.shape) for t in items]}"


def test_dataset_hash_includes_list_of_tensors():
    schema = _minimal_schema([2])
    cases = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    h1 = case_gen.dataset_data_sha256(cases)
    assert isinstance(h1, str) and len(h1) == 64
    # Re-run with same SCHEMA — hash must be stable
    cases2 = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    h2 = case_gen.dataset_data_sha256(cases2)
    assert h1 == h2, "hash unstable across identical runs"


def test_dataset_hash_differs_for_different_list_lengths():
    """Cases with list_length=2 vs list_length=3 must hash differently."""
    c2 = case_gen.generate_cases(_minimal_schema([2]),
                                   coverage_tier="pilot", dtype=torch.float32)
    c3 = case_gen.generate_cases(_minimal_schema([3]),
                                   coverage_tier="pilot", dtype=torch.float32)
    h2 = case_gen.dataset_data_sha256(c2)
    h3 = case_gen.dataset_data_sha256(c3)
    assert h2 != h3


def test_per_case_hash_stable():
    """case_data_sha256 stable for same case across re-generations."""
    schema = _minimal_schema([2])
    c1 = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    c2 = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    for a, b in zip(c1, c2):
        assert case_gen.case_data_sha256(a) == case_gen.case_data_sha256(b)


def test_list_length_plan_missing_raises():
    schema = _minimal_schema([1])  # build first, then break
    schema["tensor_inputs"][0].pop("list_length_plan")
    # depending on where it fails — may not fire until builder; either way must raise
    with pytest.raises((ValueError, KeyError)):
        case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)


def test_per_item_shape_derive_invalid_return_raises():
    def bad_derive(base_shape, i, num_items):
        return "not a list"  # noqa
    schema = _minimal_schema([2], derive=bad_derive)
    with pytest.raises(ValueError, match="per_item_shape_derive"):
        case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)


def test_mixed_list_and_normal_tensor_inputs():
    """A SCHEMA can have both a list_of_tensors AND a regular tensor."""
    schema = {
        "op_name": "mixed_test",
        "formula": "stack(tensors) + x",
        "tensor_inputs": [
            {"name": "x", "role": "operand"},
            {
                "name": "tensors",
                "kind": "list_of_tensors",
                "list_length_plan": [2, 3],
            },
        ],
        "tensor_output": "out",
        "rank": 1,
    }
    cases = case_gen.generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    assert len(cases) > 0
    for c in cases:
        assert "x" in c["inputs"] and isinstance(c["inputs"]["x"], torch.Tensor)
        assert "tensors" in c["inputs"] and isinstance(c["inputs"]["tensors"], list)
        assert all(isinstance(t, torch.Tensor) for t in c["inputs"]["tensors"])
