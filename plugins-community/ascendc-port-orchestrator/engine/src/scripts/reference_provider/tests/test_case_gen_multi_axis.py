# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for case_gen V1.5 multi-axis primitives (Gap #3 unblock).

Empirical anchor: 2026-05-17 / 2026-05-18 cold-start sweep had
`/aog-input-gen-builder` BLOCK on 5_Cumsum / 8_Sort / 9_TopK because
case_gen V1 lacked four primitives:

  1. multi-rank sweep within a single schema (the spec admits rank ∈
     {1, 2, 3, 4}).
  2. multi-dtype sweep within a single schema (5_Cumsum has a
     dtype-branched reference; running per-dtype would still hide the
     branch behavior).
  3. 2-arg scalar derive `lambda base_shape, scalars: value` so a
     later scalar can reference an earlier one (TopK's `k` depends on
     `dim` AND base_shape).
  4. rank-dependent probe_values via a callable (e.g.
     `lambda rank: list(range(-rank, rank))` for `dim`).

This test pin verifies each primitive lands and back-compat is
preserved for the V1 single-rank, single-dtype, 1-arg derive,
static-probe-list schemas.

Lives at: src/scripts/reference_provider/tests/test_case_gen_multi_axis.py
"""
from __future__ import annotations

import sys
import pathlib

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from case_gen import generate_cases  # noqa: E402


# ---------- V1 BACK-COMPAT (must not regress) -----------------------------

def test_v1_single_rank_single_dtype_unchanged() -> None:
    schema = {
        "op_name": "x",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [{"name": "k", "dtype": "int", "default": 1,
                           "probe_values": [2, 3]}],
        "tensor_output": "out",
        "rank": 2,
    }
    cases = generate_cases(schema, coverage_tier="sign_off",
                           dtype=torch.float32)
    assert len(cases) > 0
    for c in cases:
        # All cases share the same rank (= 2), dtype (= fp32)
        assert "rank" not in c["meta"] or c["meta"]["rank"] == 2
        # x tensor is fp32
        x = c["inputs"]["x"]
        assert x.dtype == torch.float32


def test_v1_1arg_derive_unchanged() -> None:
    schema = {
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [{"name": "dim", "dtype": "int", "default": 0,
                           "derive": lambda base_shape: -1}],
        "tensor_output": "out",
        "rank": 3,
    }
    cases = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    assert all(c["inputs"]["dim"] == -1 for c in cases)


# ---------- (1) Multi-rank sweep -----------------------------------------

def test_multi_rank_sweep_basic() -> None:
    """`schema['ranks'] = [1, 2, 3, 4]` → cases for each rank, merged."""
    schema = {
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [{"name": "dim", "dtype": "int", "default": 0,
                           "derive": lambda base_shape: -1}],
        "tensor_output": "out",
        "ranks": [1, 2, 3, 4],
    }
    cases = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    ranks_seen = {c["meta"].get("rank") for c in cases}
    assert ranks_seen == {1, 2, 3, 4}, f"missing ranks: got {ranks_seen}"
    # Names include rank prefix to distinguish in the merged list
    assert all(c["name"].startswith(f"r{c['meta']['rank']}_") for c in cases)


def test_multi_rank_and_rank_mutually_exclusive() -> None:
    bad_schema = {
        "tensor_inputs": [{"name": "x"}],
        "scalar_inputs": [],
        "rank": 1, "ranks": [1, 2],
    }
    with pytest.raises(ValueError, match="rank.*ranks"):
        generate_cases(bad_schema)


def test_multi_rank_empty_list_rejected() -> None:
    bad_schema = {
        "tensor_inputs": [{"name": "x"}],
        "scalar_inputs": [],
        "ranks": [],
    }
    with pytest.raises(ValueError, match="non-empty"):
        generate_cases(bad_schema)


# ---------- (2) Multi-dtype sweep ----------------------------------------

def test_multi_dtype_sweep_basic() -> None:
    """`schema['dtypes']` → cases for each dtype, merged."""
    schema = {
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [{"name": "dim", "dtype": "int", "default": 0,
                           "derive": lambda base_shape: -1}],
        "tensor_output": "out",
        "rank": 2,
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    dtype_tags = {c["meta"].get("dtype") for c in cases}
    # Tags are str repr of the dtype with "torch." prefix stripped.
    assert dtype_tags == {"float32", "float16", "bfloat16"}
    # Tensor data matches the declared dtype per case
    fp32_cases = [c for c in cases if c["meta"]["dtype"] == "float32"]
    assert all(c["inputs"]["x"].dtype == torch.float32 for c in fp32_cases)
    fp16_cases = [c for c in cases if c["meta"]["dtype"] == "float16"]
    assert all(c["inputs"]["x"].dtype == torch.float16 for c in fp16_cases)


def test_multi_rank_and_multi_dtype_combined() -> None:
    """Cross-product: ranks × dtypes. Each rank gets all dtypes."""
    schema = {
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [{"name": "dim", "dtype": "int", "default": 0,
                           "derive": lambda base_shape: -1}],
        "tensor_output": "out",
        "ranks": [1, 2],
        "dtypes": [torch.float32, torch.float16],
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    ranks_seen = {c["meta"].get("rank") for c in cases}
    dtypes_seen = {c["meta"].get("dtype") for c in cases}
    assert ranks_seen == {1, 2}
    assert dtypes_seen == {"float32", "float16"}


# ---------- (3) 2-arg scalar derive --------------------------------------

def test_2arg_derive_scalars_reference_earlier_scalars() -> None:
    """TopK pattern: `k = min(base_shape[dim], 8)` where dim is itself
    derived. The second derive takes (base_shape, scalars) and reads
    scalars['dim'].
    """
    def derive_dim(base_shape):
        # Pick first axis (always valid)
        return 0

    def derive_k(base_shape, scalars):
        dim = scalars["dim"]
        # Use the actual size at the chosen dim
        size_at_dim = base_shape[dim]
        return min(size_at_dim, 8)

    schema = {
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [
            {"name": "dim", "dtype": "int", "default": 0, "derive": derive_dim},
            {"name": "k", "dtype": "int", "default": 1, "derive": derive_k},
        ],
        "tensor_output": "out",
        "rank": 3,
    }
    cases = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    assert len(cases) > 0
    for c in cases:
        assert c["inputs"]["dim"] == 0
        # k must be valid for the case's actual base_shape[0]
        x_shape = list(c["inputs"]["x"].shape)
        assert 1 <= c["inputs"]["k"] <= max(1, x_shape[0])
        assert c["inputs"]["k"] <= 8  # respects min(size, 8)


def test_2arg_derive_back_compat_with_1arg() -> None:
    """A schema with one 1-arg derive AND one 2-arg derive mixes cleanly."""
    schema = {
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [
            {"name": "dim", "dtype": "int", "default": 0,
             "derive": lambda base_shape: -1},
            {"name": "k", "dtype": "int", "default": 1,
             "derive": lambda base_shape, scalars: scalars["dim"] + 100},
        ],
        "tensor_output": "out",
        "rank": 2,
    }
    cases = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    for c in cases:
        assert c["inputs"]["dim"] == -1
        assert c["inputs"]["k"] == 99  # -1 + 100


# ---------- (4) Callable probe_values ------------------------------------

def test_rank_dependent_probe_values_scalar_only() -> None:
    """Scalar-only schema: probe_values can be `lambda rank: [...]`.

    Each rank-loop iteration sees its own rank and emits its own probe set.
    """
    def probe_for_dim(rank):
        # Valid dim range for the given rank
        return list(range(-rank, rank))

    schema = {
        "tensor_inputs": [],
        "scalar_inputs": [
            {"name": "dim", "dtype": "int", "default": 0,
             "probe_values": probe_for_dim},
        ],
        "tensor_output": "out",
        "ranks": [1, 2, 3],
    }
    cases = generate_cases(schema, coverage_tier="sign_off")
    # For rank=3, dim probes should include -3, -2, -1, 0, 1, 2
    rank3_cases = [c for c in cases if c["meta"].get("rank") == 3]
    rank3_dims = {c["inputs"]["dim"] for c in rank3_cases if c["meta"].get("band") == "S_probe"}
    # baseline (default=0) is excluded from S_probe; expect 1, 2, -1, -2, -3
    assert {-3, -2, -1, 1, 2}.issubset(rank3_dims)
    # For rank=1, dim probes should be only {-1, 0} → S_probe excludes default 0 → {-1}
    rank1_cases = [c for c in cases if c["meta"].get("rank") == 1]
    rank1_dims = {c["inputs"]["dim"] for c in rank1_cases if c["meta"].get("band") == "S_probe"}
    assert rank1_dims == {-1}, f"rank=1 should produce {{-1}}, got {rank1_dims}"


def test_callable_probe_values_back_compat_static() -> None:
    """Static-list probe_values still works as before."""
    schema = {
        "tensor_inputs": [],
        "scalar_inputs": [
            {"name": "k", "dtype": "int", "default": 1, "probe_values": [2, 3, 5]},
        ],
        "tensor_output": "out",
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    ks = {c["inputs"]["k"] for c in cases}
    assert ks == {1, 2, 3, 5}  # baseline + 3 probes


# ---------- Combined: 5_Cumsum-style schema -----------------------------

def test_5_cumsum_style_full_schema() -> None:
    """5_Cumsum admits rank ∈ {1,2,3,4}, dtypes ∈ {fp32, fp16, bf16},
    dim is rank-aware. The full V1.5 schema should now succeed without
    silent coverage reduction.
    """
    schema = {
        "op_name": "5_Cumsum",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [
            {"name": "dim", "dtype": "int", "default": 0,
             "derive": lambda base_shape: -1},
        ],
        "tensor_output": "out",
        "ranks": [1, 2, 3, 4],
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    # Cross-product: 4 ranks × 3 dtypes × (>= 1 case each) → >= 12 cases
    assert len(cases) >= 12
    rank_dtype_pairs = {(c["meta"]["rank"], c["meta"]["dtype"]) for c in cases}
    expected_pairs = {(r, dt) for r in [1, 2, 3, 4]
                      for dt in ["float32", "float16", "bfloat16"]}
    assert rank_dtype_pairs == expected_pairs


# ---- V1.6.B optional-tensor primitive (2026-05-19 DEBT-069 Gap A) ----


def test_v1_6_b_single_optional_tensor_doubles_cases() -> None:
    """`optional=True` on one tensor → case count doubles (present + absent)."""
    schema_base = {
        "op_name": "with_optional",
        "tensor_inputs": [{"name": "x"}],
        "tensor_output": "y",
        "rank": 1,
    }
    base = generate_cases(schema_base, coverage_tier="pilot")
    n_base = len(base)

    schema_opt = {
        "op_name": "with_optional",
        "tensor_inputs": [{"name": "x"}, {"name": "bias", "optional": True}],
        "tensor_output": "y",
        "rank": 1,
    }
    expanded = generate_cases(schema_opt, coverage_tier="pilot")

    assert len(expanded) == n_base * 2, (
        f"1 optional tensor must double case count: got {len(expanded)} "
        f"vs base {n_base} × 2 = {n_base * 2}"
    )
    n_present = sum(1 for c in expanded if c["inputs"]["bias"] is not None)
    n_absent = sum(1 for c in expanded if c["inputs"]["bias"] is None)
    assert n_present == n_base, f"present cases: {n_present} (expected {n_base})"
    assert n_absent == n_base, f"absent cases: {n_absent} (expected {n_base})"


def test_v1_6_b_optional_tensor_none_in_inputs() -> None:
    """Absent path puts Python None in inputs[name], NOT a tensor."""
    schema = {
        "op_name": "test",
        "tensor_inputs": [{"name": "x"}, {"name": "bias", "optional": True}],
        "tensor_output": "y",
        "rank": 1,
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    absent_case = next(c for c in cases if not c["meta"]["optional_bias_present"])
    assert absent_case["inputs"]["bias"] is None
    assert absent_case["inputs"]["x"] is not None  # x stays materialized


def test_v1_6_b_meta_records_presence_state() -> None:
    """Each case meta exposes `optional_<name>_present` boolean for filtering."""
    schema = {
        "op_name": "test",
        "tensor_inputs": [{"name": "x"}, {"name": "w", "optional": True}],
        "tensor_output": "y",
        "rank": 1,
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    for c in cases:
        assert "optional_w_present" in c["meta"]
        assert isinstance(c["meta"]["optional_w_present"], bool)


def test_v1_6_b_two_optional_tensors_4x_fanout() -> None:
    """k optional tensors → 2^k cases per base. With k=2, expect 4× and all 4 patterns."""
    schema = {
        "op_name": "two_opt",
        "tensor_inputs": [
            {"name": "x"},
            {"name": "bias", "optional": True},
            {"name": "weight", "optional": True},
        ],
        "tensor_output": "y",
        "rank": 1,
    }
    base = generate_cases(
        {"op_name": "base", "tensor_inputs": [{"name": "x"}],
         "tensor_output": "y", "rank": 1},
        coverage_tier="pilot",
    )
    cases = generate_cases(schema, coverage_tier="pilot")
    assert len(cases) == len(base) * 4

    patterns = {
        (c["meta"]["optional_bias_present"], c["meta"]["optional_weight_present"])
        for c in cases
    }
    assert patterns == {(True, True), (True, False), (False, True), (False, False)}


def test_v1_6_b_back_compat_no_optional_unchanged() -> None:
    """SCHEMAs without `optional` field produce same case count as pre-V1.6.B."""
    schema = {
        "op_name": "no_optional",
        "tensor_inputs": [{"name": "x"}, {"name": "y"}],
        "tensor_output": "out",
        "rank": 1,
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    # No optional tensors → no forking; cases should all have x and y materialized
    for c in cases:
        assert c["inputs"]["x"] is not None
        assert c["inputs"]["y"] is not None
        assert "optional_x_present" not in c["meta"]
        assert "optional_y_present" not in c["meta"]


def test_v1_6_b_name_suffix_indicates_presence() -> None:
    """Case names include `_<tname>{T|F}` suffix per optional tensor."""
    schema = {
        "op_name": "test",
        "tensor_inputs": [{"name": "x"}, {"name": "w", "optional": True}],
        "tensor_output": "y",
        "rank": 1,
    }
    cases = generate_cases(schema, coverage_tier="pilot")
    present_names = [c["name"] for c in cases if c["meta"]["optional_w_present"]]
    absent_names = [c["name"] for c in cases if not c["meta"]["optional_w_present"]]
    assert all(n.endswith("_wT") for n in present_names), present_names[:3]
    assert all(n.endswith("_wF") for n in absent_names), absent_names[:3]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
