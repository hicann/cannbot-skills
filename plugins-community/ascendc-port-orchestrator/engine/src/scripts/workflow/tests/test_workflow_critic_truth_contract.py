# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression test for the migration schema-gap truth contract."""
from __future__ import annotations

import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

import workflow_critic as critic  # noqa: E402


def test_schema_gap_waives_only_the_standard_input_recipe() -> None:
    state_machine = critic.load_state_machine()
    phase = state_machine["phases"]["O2_5_reference_provider"]
    migration = phase["branches"]["port_a3_to_a5"]
    backward = phase["branches"]["backward"]
    reference_sources = migration["reference_sources"]
    a3_live = reference_sources["a3_live"]
    schema_gap = a3_live["exceptions"]["schema_gap"]

    assert "no_edge_cases_flag" not in phase
    assert "critic_gate" not in migration
    assert "critic_gate" not in backward
    assert migration["reference_source"] == {
        "default_for_new_invocation": "npubench",
        "persisted_field": "state.reference.source",
        "legacy_missing_reference": "a3_live",
        "accepted_values": ["npubench", "a3_live"],
    }
    assert set(reference_sources) == {"a3_live"}
    a3_artifacts = {item["path"] for item in a3_live["required_artifacts"]}
    assert "workspace/{op}/edge_dataset.pt" in a3_artifacts
    assert "workspace/{op}/a3_capture_manifest.json" in a3_artifacts
    assert any(
        "fresh source-arch NPU capture" in invariant["description"]
        for invariant in a3_live["invariants"]
    )
    assert schema_gap["fallback_allows"] == [
        "reuse a provenance-recorded deterministic input recipe, or provide a custom recipe for the unsupported schema",
        "waive only the standard case_gen/SCHEMA input-recipe form",
    ]
    assert any("current run" in rule for rule in schema_gap["still_requires"])
    assert any("arch22 source operator" in rule for rule in schema_gap["still_requires"])
    assert any("cached, archived or committed" in rule
               for rule in schema_gap["does_not_allow"])

    yaml_text = critic.YAML_PATH.read_text(encoding="utf-8")
    assert "--no-edge-cases" not in yaml_text
    assert "use the source operator's committed source-NPU fixture" not in yaml_text

    plugin_root = critic.YAML_PATH.parent.parent
    template = plugin_root / "scripts" / "reference_provider" / "input_gen.template.py"
    template_text = template.read_text(encoding="utf-8")
    assert "execute the staged arch22 source operator on a source NPU" in template_text
    assert "use the committed source-NPU fixture" not in template_text
