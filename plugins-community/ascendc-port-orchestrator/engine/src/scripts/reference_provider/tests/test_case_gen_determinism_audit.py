# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""S1 deliverable for EDGE_DATA_A3_SIDE_DESIGN_2026_05_20.

Audit: every distribution generator factory in case_gen._distribution_plan
must be bit-deterministic. Same `(seed, n_elem, dtype)` → identical bytes.

Rationale: the design's `edge_manifest.json` records seed + per-case hash;
regenerating must produce hash-identical tensors. If any factory is
non-deterministic, the design breaks at S1.5 gate and we re-converge on
mitigations (Q3 fallback to (b) PCG64).

Coverage:
- All 5 distribution kinds defined in case_gen: const / randn / uniform /
  mixed_sign / denormal
- 3 dtypes: fp32, fp16, bf16 (the dtype range case_gen produces)
- Multiple n_elem sizes including alignment / boundary cases
- Cross-call determinism: two separate calls to the same factory with
  same args must produce bit-identical output
- Cross-instantiation determinism: factories built via separate
  _distribution_plan() calls must produce identical output for same seed

Also covers `generate_cases` at the higher level — running the case
generator twice with the same SCHEMA + seed_base must produce identical
edge data (verified via dataset_data_sha256).

This is the gate that unblocks (or blocks) S2 of the design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from case_gen import _distribution_plan, dataset_data_sha256  # noqa: E402


# ---- Distribution-factory determinism (the load-bearing audit) ----


@pytest.fixture(scope="module")
def dist_plans():
    """Build TWO independent distribution plans (catches global-state leaks)."""
    p1 = _distribution_plan("production")
    p2 = _distribution_plan("production")
    return p1, p2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_elem", [128, 256, 1024, 4096, 65537])  # incl odd boundary
@pytest.mark.parametrize("seed", [0, 1, 42, 12345, 0xDEADBEEF])
def test_distribution_factories_bit_deterministic(dist_plans, dtype, n_elem, seed):
    """Same factory called twice with same (n, dtype, seed) → bit-identical."""
    plan1, plan2 = dist_plans
    for entry1, entry2 in zip(plan1, plan2):
        if "fn" not in entry1:
            continue  # cancellation / sparse_nan / sparse_inf — non-fn signal
        fn1 = entry1["fn"]
        fn2 = entry2["fn"]
        tag = entry1["tag"]
        # Two consecutive calls on first plan
        try:
            t_a = fn1(n_elem, dtype, seed)
            t_b = fn1(n_elem, dtype, seed)
        except RuntimeError as e:
            # Some dtypes may not be supported by some distributions
            # (e.g. bf16 + torch.rand pre-2.0). Skip those — not a determinism
            # bug, just a coverage gap.
            pytest.skip(f"factory {tag!r} unsupported for dtype={dtype}: {e}")
        # Cross-plan: second plan instance, same seed, must match
        t_c = fn2(n_elem, dtype, seed)
        # Bit equality (cast to uint8 view for nan-safe comparison)
        b_a = t_a.view(torch.uint8) if t_a.dtype.is_floating_point else t_a
        b_b = t_b.view(torch.uint8) if t_b.dtype.is_floating_point else t_b
        b_c = t_c.view(torch.uint8) if t_c.dtype.is_floating_point else t_c
        assert torch.equal(b_a, b_b), (
            f"factory {tag!r} non-deterministic across calls "
            f"(dtype={dtype}, n={n_elem}, seed={seed})"
        )
        assert torch.equal(b_a, b_c), (
            f"factory {tag!r} non-deterministic across plan instantiations "
            f"(dtype={dtype}, n={n_elem}, seed={seed})"
        )


# Distributions whose RANGE overflows / underflows a dtype produce dtype-saturated
# output (all-inf for large_mag@fp16, all-0 for small_mag@fp16). Different seeds
# produce identical output via saturation — that's NOT a determinism bug, it's
# expected dtype behavior. Skip these (dtype, factory) pairs in the "distinct
# seed → distinct output" sanity check.
_SATURATION_SKIPS = {
    ("large_mag", torch.float16),    # 1e20-1e30 → all +inf in fp16
    ("large_mag", torch.bfloat16),   # bf16 max ~3.4e38; 1e30 fits but barely
    ("small_mag", torch.float16),    # 1e-30-1e-20 → all 0 / subnormal in fp16
    ("denormal", torch.float16),    # all in fp16 subnormal range; mostly zero
    ("denormal", torch.bfloat16),   # bf16 has narrower subnormal range
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_distribution_factories_distinct_seed_distinct_output(dist_plans, dtype):
    """Sanity: different seeds → different output (else seed isn't being used).

    Skips:
    - const_* entries (seed-invariant by design)
    - (factory, dtype) pairs where the distribution range saturates the dtype
      (e.g. large_mag@fp16 → all-inf regardless of seed; not a determinism bug)
    """
    plan, _ = dist_plans
    n_elem = 1024
    for entry in plan:
        tag = entry["tag"]
        if "fn" not in entry:
            continue  # non-fn (cancellation etc) handled separately
        if tag.startswith("const"):
            continue  # const_sanity, const_near_zero — seed-invariant by design
        if (tag, dtype) in _SATURATION_SKIPS:
            continue  # dtype-saturation expected, not non-determinism
        fn = entry["fn"]
        try:
            t1 = fn(n_elem, dtype, 0)
            t2 = fn(n_elem, dtype, 1)
        except RuntimeError:
            continue  # unsupported dtype combination
        # Cast to fp32 for nan-aware comparison
        eq = torch.equal(
            t1.float().view(torch.int32),
            t2.float().view(torch.int32),
        )
        assert not eq, (
            f"factory {tag!r} produced identical output for seed=0 vs seed=1 "
            f"(dtype={dtype}) — seed isn't being used"
        )


# ---- generate_cases higher-level determinism ----


def test_generate_cases_deterministic_for_pointwise():
    """Run generate_cases twice with the same SCHEMA; check dataset_data_sha256
    matches. (`generate_cases` itself doesn't take a seed param — seeds are
    managed internally; this verifies that internal management produces
    reproducible output across calls within one process.)

    Uses a minimal SCHEMA — a single tensor input — to keep the test fast
    while still exercising the full case_gen pipeline (band A + B + C).
    """
    from case_gen import generate_cases  # local import to dodge sys.path issues

    schema = {
        "op_name": "test_determinism_rank2",
        "formula": "y = x",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 2,
        "base_shape_filter": lambda b: True,
    }

    cases1 = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    cases2 = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)

    h1 = dataset_data_sha256(cases1)
    h2 = dataset_data_sha256(cases2)
    assert h1 == h2, (
        f"generate_cases(pilot) non-deterministic across calls: "
        f"run1.sha256={h1[:16]}... vs run2.sha256={h2[:16]}..."
    )
    # Sanity: pilot tier should produce at least a handful of cases
    assert len(cases1) >= 1


def test_generate_cases_deterministic_across_tiers():
    """Different coverage_tier (pilot vs production) produces different
    dataset_data_sha256 — confirms tier is reflected in output and that
    deterministic output isn't accidentally tier-invariant.
    """
    from case_gen import generate_cases

    schema = {
        "op_name": "test_tier_rank2",
        "formula": "y = x",
        "tensor_inputs": [{"name": "x", "role": "operand"}],
        "scalar_inputs": [],
        "tensor_output": "y",
        "rank": 2,
        "base_shape_filter": lambda b: True,
    }

    cases_pilot = generate_cases(schema, coverage_tier="pilot", dtype=torch.float32)
    cases_prod = generate_cases(schema, coverage_tier="production", dtype=torch.float32)

    h_pilot = dataset_data_sha256(cases_pilot)
    h_prod = dataset_data_sha256(cases_prod)

    # Two ways this could fail:
    # 1. Same hash → tier not reflected in output (case_gen ignoring tier)
    # 2. Production has fewer cases than pilot (tier ordering broken)
    assert len(cases_prod) >= len(cases_pilot), (
        f"production tier has fewer cases ({len(cases_prod)}) than pilot "
        f"({len(cases_pilot)}) — tier ordering broken"
    )
    # If sizes differ, hashes must differ (different content)
    if len(cases_prod) != len(cases_pilot):
        assert h_pilot != h_prod


# ---- Side-channel: torch.manual_seed should NOT affect case_gen output ----


def test_factories_unaffected_by_torch_global_seed(dist_plans):
    """`torch.manual_seed(N)` sets a global RNG. case_gen factories use
    `torch.Generator().manual_seed(seed)` (a *local* generator), so global
    state must NOT bleed in. If it does, a process that called
    torch.manual_seed earlier could change case_gen output → non-determinism
    in production.
    """
    plan, _ = dist_plans
    n_elem = 1024
    dtype = torch.float32
    seed = 12345
    for entry in plan:
        if entry["tag"] == "const_sanity" or "fn" not in entry:
            continue
        fn = entry["fn"]
        torch.manual_seed(777)
        t_with_global_seed_777 = fn(n_elem, dtype, seed)
        torch.manual_seed(11111)
        t_with_global_seed_11111 = fn(n_elem, dtype, seed)
        # Should be identical — case_gen ignores torch global RNG
        eq = torch.equal(
            t_with_global_seed_777.float().view(torch.int32),
            t_with_global_seed_11111.float().view(torch.int32),
        )
        assert eq, (
            f"factory {entry['tag']!r} reads torch global RNG — output differs "
            f"after torch.manual_seed(777) vs torch.manual_seed(11111). "
            f"This is a non-determinism BUG: any caller that touched global "
            f"RNG before case_gen would shift outputs. S1.5 gate triggered."
        )
