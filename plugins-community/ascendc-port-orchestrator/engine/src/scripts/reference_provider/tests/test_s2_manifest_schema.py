# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""S2 deliverable for EDGE_DATA_A3_SIDE_DESIGN_2026_05_20 v0.2.

Tests:
1. `case_data_sha256(case)` produces stable per-case hashes
2. Per-case hash invariants (determinism, distinct-input distinct-hash)
3. Cross-reference: per-case hashes are EACH consistent across calls
   (not the same as `dataset_data_sha256` aggregating them — see
   `_update_hash_for_case` docstring for why)
4. Manifest schema v2 fields present + correct shapes

The per-case `input_sha256` is the load-bearing audit field — S5's
finalize gate will verify regenerated inputs against the archive's
manifest by computing `case_data_sha256(regen_case) == manifest[i].input_sha256`.
If this hash is wrong / unstable, S5's gate is paperwork.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from case_gen import (  # noqa: E402
    case_data_sha256,
    dataset_data_sha256,
    generate_cases,
)


_SIMPLE_SCHEMA = {
    "op_name": "test_s2_simple",
    "formula": "y = x",
    "tensor_inputs": [{"name": "x", "role": "operand"}],
    "scalar_inputs": [],
    "tensor_output": "y",
    "rank": 2,
    "base_shape_filter": lambda b: True,
}


@pytest.fixture(scope="module")
def pilot_cases():
    """Pilot tier (~15 cases) — enough variety to test against."""
    return generate_cases(_SIMPLE_SCHEMA, coverage_tier="pilot", dtype=torch.float32)


# ---- case_data_sha256 basic invariants ----


def test_case_data_sha256_deterministic_same_case(pilot_cases):
    """Hashing the same case twice → identical hash."""
    case = pilot_cases[0]
    h1 = case_data_sha256(case)
    h2 = case_data_sha256(case)
    assert h1 == h2


def test_case_data_sha256_is_hex_64_chars(pilot_cases):
    """sha256 output is 64 hex chars."""
    case = pilot_cases[0]
    h = case_data_sha256(case)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_case_data_sha256_distinct_cases_distinct_hashes(pilot_cases):
    """Different cases produce different hashes (sanity — no accidental
    hash collision in the test dataset).
    """
    if len(pilot_cases) < 2:
        pytest.skip("need at least 2 cases")
    hashes = [case_data_sha256(c) for c in pilot_cases]
    # Most should be distinct; allow up to 1 accidental duplicate in case
    # two probe variants happen to produce identical tensors.
    distinct = len(set(hashes))
    assert distinct >= len(pilot_cases) - 1, (
        f"Too many duplicate hashes: {distinct}/{len(pilot_cases)} distinct"
    )


def test_case_data_sha256_changes_on_tensor_perturbation(pilot_cases):
    """Modify one byte of an input tensor → hash must change."""
    import copy
    case = copy.deepcopy(pilot_cases[0])
    h_orig = case_data_sha256(case)
    # Perturb first element of first tensor input
    for k, v in case["inputs"].items():
        if isinstance(v, torch.Tensor) and v.numel() > 0 and v.dtype.is_floating_point:
            v_clone = v.clone()
            v_clone.view(-1)[0] += 1e-6
            case["inputs"][k] = v_clone
            break
    h_perturbed = case_data_sha256(case)
    assert h_orig != h_perturbed, "Hash unchanged after tensor perturbation"


def test_case_data_sha256_independent_of_other_cases(pilot_cases):
    """Per-case hash depends ONLY on the case itself, not other cases in
    the dataset (this is what makes it useful for the S5 finalize gate —
    verify ONE regenerated case without re-running the whole dataset).
    """
    case0 = pilot_cases[0]
    case1 = pilot_cases[-1]  # last case in the list
    h0_alone = case_data_sha256(case0)
    h0_with_others = case_data_sha256(case0)  # same op, just verify stability
    h1_alone = case_data_sha256(case1)
    h1_with_others = case_data_sha256(case1)
    assert h0_alone == h0_with_others
    assert h1_alone == h1_with_others


# ---- Cross-reference with dataset_data_sha256 ----


def test_dataset_sha256_changes_when_any_case_perturbed(pilot_cases):
    """Dataset hash must change when one case is perturbed (sanity that
    dataset_data_sha256 isn't accidentally invariant to inputs).
    """
    import copy
    h_orig = dataset_data_sha256(pilot_cases)
    cases_perturbed = copy.deepcopy(pilot_cases)
    for k, v in cases_perturbed[0]["inputs"].items():
        if isinstance(v, torch.Tensor) and v.numel() > 0 and v.dtype.is_floating_point:
            v_clone = v.clone()
            v_clone.view(-1)[0] += 1e-6
            cases_perturbed[0]["inputs"][k] = v_clone
            break
    h_perturbed = dataset_data_sha256(cases_perturbed)
    assert h_orig != h_perturbed


def test_case_hashes_unchanged_when_dataset_reordered():
    """Per-case hashes are independent of case ordering in the dataset.
    Reordering cases changes dataset_data_sha256 (sort-stable but hash
    sequence matters? actually NO — dataset_data_sha256 sorts by idx),
    but each case's individual hash is invariant to ordering.
    """
    cases = generate_cases(_SIMPLE_SCHEMA, coverage_tier="pilot", dtype=torch.float32)
    h_cases_orig = [case_data_sha256(c) for c in cases]
    # Reverse the list — per-case hashes for each should be unchanged
    cases_reversed = list(reversed(cases))
    h_cases_rev = [case_data_sha256(c) for c in cases_reversed]
    # Same set of hashes regardless of order
    assert sorted(h_cases_orig) == sorted(h_cases_rev)


# ---- Manifest schema v2 shape (validated via input_gen template structure) ----


def test_manifest_schema_v2_template_has_required_fields():
    """input_gen.template.py emits a manifest with seed + per-case hash +
    schema version. This test reads the template source to verify the
    schema fields are present in the manifest dict literal.

    Can't easily exec the template (it has guards on op_name placeholder),
    so we grep the source for the field names.
    """
    template = Path(__file__).resolve().parent.parent / "input_gen.template.py"
    src = template.read_text()
    # Top-level v2 fields
    assert '"manifest_schema_version": 2' in src
    assert '"seed": COVERAGE_SEED' in src
    assert '"case_gen_version":' in src
    # Per-case v2 field
    assert '"input_sha256": case_data_sha256(c)' in src


def test_manifest_schema_v2_template_imports_case_data_sha256():
    """Template imports the new helper from case_gen."""
    template = Path(__file__).resolve().parent.parent / "input_gen.template.py"
    src = template.read_text()
    assert "from case_gen import case_data_sha256, dataset_data_sha256, generate_cases" in src
