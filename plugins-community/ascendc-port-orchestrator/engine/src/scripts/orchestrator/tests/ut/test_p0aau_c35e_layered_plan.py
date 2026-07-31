# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for read_layered_plan — Tier 3 Stage 1 schema reader.

P0aau-c35.e (2026-05-09): op_classification.json gains
`algorithm_classification` + `layered_implementation_plan` fields.
read_layered_plan() returns LayeredPlan when fused + applicable, None
otherwise — fail-closed for malformed input so orchestrator routes
through standard await_worker path safely.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from briefs.op_taxonomy import LayeredPlan, LayerSpec, read_layered_plan  # noqa: E402


def _fa_classification() -> dict:
    """Realistic FA classification with valid 3-layer plan."""
    return {
        "op": "3_FusionAttention",
        "schema_version": 3,
        "source_sha256": "deadbeef" * 8,
        "op_class_tags": ["fused", "softmax", "transcendental"],
        "kb_recommendations": [
            {"path": "OPERATIONAL_KNOWLEDGE.md#OL-68", "reason": "Path A reference"},
        ],
        "algorithm_classification": "fused",
        "layered_implementation_plan": {
            "applicable": True,
            "layers": [
                {
                    "layer": 1,
                    "name": "qk_matmul",
                    "sub_op": "matmul",
                    "inputs": ["query", "key"],
                    "outputs_added": ["scores"],
                    "outputs_placeholder": ["softmax_max", "softmax_sum"],
                    "outputs_filled": [],
                    "reference_decomposition": "torch.matmul(query, key.transpose(-2, -1))",
                    "verify_against": "isolated_layer_ref",
                    "optional": False,
                },
                {
                    "layer": 2,
                    "name": "softmax_with_scale",
                    "sub_op": "softmax",
                    "inputs": ["scores", "scale"],
                    "outputs_added": ["attn"],
                    "outputs_placeholder": [],
                    "outputs_filled": ["softmax_max", "softmax_sum"],
                    "reference_decomposition": "torch.softmax(scores * scale, dim=-1)",
                    "verify_against": "isolated_layer_ref",
                    "optional": False,
                },
                {
                    "layer": 3,
                    "name": "av_matmul",
                    "sub_op": "matmul",
                    "inputs": ["attn", "value"],
                    "outputs_added": ["attention_out"],
                    "outputs_placeholder": [],
                    "outputs_filled": [],
                    "reference_decomposition": "torch.matmul(attn, value)",
                    "verify_against": "isolated_layer_ref",
                    "optional": False,
                },
            ],
        },
        "rationale": "fused attention",
    }


def test_returns_none_when_workspace_missing(tmp_path):
    assert read_layered_plan(tmp_path / "doesnt_exist") is None


def test_returns_none_when_classification_missing(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert read_layered_plan(workspace) is None


def test_returns_none_when_classification_unparseable(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "op_classification.json").write_text("not json")
    assert read_layered_plan(workspace) is None


def test_returns_none_for_single_op_classification(tmp_path):
    """Old schema_version=2 ops + single_op v3 ops → no layered routing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "op_classification.json").write_text(json.dumps({
        "op": "1_GELU",
        "schema_version": 2,
        "op_class_tags": ["transcendental"],
        "kb_recommendations": [],
    }))
    assert read_layered_plan(workspace) is None


def test_returns_none_for_v3_single_op(tmp_path):
    """v3 schema with algorithm_classification=single_op → None (route standard)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "op_classification.json").write_text(json.dumps({
        "op": "1_GELU",
        "schema_version": 3,
        "algorithm_classification": "single_op",
        "layered_implementation_plan": {
            "applicable": False,
            "rationale_when_inapplicable": "single primitive",
            "layers": [],
        },
    }))
    assert read_layered_plan(workspace) is None


def test_returns_none_when_applicable_false(tmp_path):
    """fused but applicable=false (e.g., RNG-driven) → None, route standard."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "op_classification.json").write_text(json.dumps({
        "op": "15_AttentionSoftmaxWithSoftcappingAndDropout",
        "schema_version": 3,
        "algorithm_classification": "other",
        "layered_implementation_plan": {
            "applicable": False,
            "rationale_when_inapplicable": "RNG-driven dropout breaks layer isolation",
            "layers": [],
        },
    }))
    assert read_layered_plan(workspace) is None


def test_parses_valid_fused_plan(tmp_path):
    """Real FA-style plan → LayeredPlan with 3 layers."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "op_classification.json").write_text(json.dumps(_fa_classification()))

    plan = read_layered_plan(workspace)
    assert plan is not None
    assert isinstance(plan, LayeredPlan)
    assert plan.applicable is True
    assert len(plan.layers) == 3

    l1 = plan.layers[0]
    assert l1.layer == 1
    assert l1.name == "qk_matmul"
    assert l1.outputs_placeholder == ["softmax_max", "softmax_sum"]
    assert l1.optional is False

    l2 = plan.layers[1]
    assert l2.outputs_filled == ["softmax_max", "softmax_sum"]


def test_returns_none_for_non_canonical_layer_numbering(tmp_path):
    """Layers must be numbered 1..N strictly. Skipped numbers → fail-closed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bad = _fa_classification()
    bad["layered_implementation_plan"]["layers"][1]["layer"] = 5  # skip 2
    (workspace / "op_classification.json").write_text(json.dumps(bad))

    assert read_layered_plan(workspace) is None


def test_returns_none_for_malformed_layer(tmp_path):
    """Missing required field per layer → fail-closed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bad = _fa_classification()
    del bad["layered_implementation_plan"]["layers"][0]["reference_decomposition"]
    (workspace / "op_classification.json").write_text(json.dumps(bad))

    assert read_layered_plan(workspace) is None


def test_returns_none_for_empty_layers_list(tmp_path):
    """applicable=True with empty layers — invalid, fail-closed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bad = _fa_classification()
    bad["layered_implementation_plan"]["layers"] = []
    (workspace / "op_classification.json").write_text(json.dumps(bad))

    assert read_layered_plan(workspace) is None


def test_optional_layer_parsed_correctly(tmp_path):
    """Optional feature-wiring layer → optional=True, verify_against=full_fixture."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cls = _fa_classification()
    cls["layered_implementation_plan"]["layers"].append({
        "layer": 4,
        "name": "feature_wiring",
        "sub_op": "feature_codepaths",
        "inputs": ["query", "key", "value", "scale", "atten_mask"],
        "outputs_added": [],
        "outputs_placeholder": [],
        "outputs_filled": [],
        "reference_decomposition": "model_full_attention(...)",
        "verify_against": "full_fixture",
        "optional": True,
    })
    (workspace / "op_classification.json").write_text(json.dumps(cls))

    plan = read_layered_plan(workspace)
    assert plan is not None
    assert len(plan.layers) == 4
    assert plan.layers[3].optional is True
    assert plan.layers[3].verify_against == "full_fixture"
