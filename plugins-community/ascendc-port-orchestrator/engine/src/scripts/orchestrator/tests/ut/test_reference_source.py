# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for the exhaustive, fail-closed provider registry."""
from __future__ import annotations

import json

import pytest

import reference_source


def _npubench_binding():
    return {
        "schema_version": 3,
        "source": "npubench",
        "semantic_binding": "npubench_old_format_task_bundle",
        "runner_contract_version": "npubench/v1",
        "bundle_manifest_path": "/workspace/reference_inputs/npubench/digest/bundle_manifest.json",
        "bundle_manifest_sha256": "a" * 64,
        "bundle_sha256": "b" * 64,
        "task_relative_path": "level1/3_Add.py",
        "task_sha256": "c" * 64,
        "sidecar_relative_path": "level1/3_Add.json",
        "sidecar_sha256": "d" * 64,
        "sidecar_encoding": "jsonl",
    }


def test_missing_reference_fails_closed_and_never_defaults_to_live_a3(tmp_path):
    workspace = tmp_path / "op"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "port_a3_to_a5"}))

    with pytest.raises(reference_source.ReferenceSourceError, match="no explicit reference"):
        reference_source.resolve_reference_source(workspace)


def test_all_registered_complete_reference_bindings_are_recognised():
    assert reference_source.resolve_reference_source(
        {"reference": reference_source.explicit_a3_live_binding()}
    ) == "a3_live"
    assert reference_source.resolve_reference_source(
        {"reference": _npubench_binding()}
    ) == "npubench"
    assert reference_source.uses_npubench_reference({"reference": _npubench_binding()})
    assert reference_source.resolve_reference_source(
        {"reference": reference_source.explicit_cannbench_binding()}
    ) == "cannbench"
    assert reference_source.uses_cannbench_reference(
        {"reference": reference_source.explicit_cannbench_binding()}
    )


@pytest.mark.parametrize(
    "reference",
    [{}, {"source": "offline_tensor"}, "a3_live", {"source": "a3_live"}],
)
def test_malformed_explicit_reference_fails_closed(reference):
    with pytest.raises(reference_source.ReferenceSourceError):
        reference_source.resolve_reference_source({"reference": reference})


@pytest.mark.parametrize(
    "field",
    ["schema_version", "source", "semantic_binding", "runner_contract_version"],
)
def test_a3_live_binding_missing_field_fails_closed(field):
    binding = reference_source.explicit_a3_live_binding()
    del binding[field]
    with pytest.raises(reference_source.ReferenceSourceError, match="a3_live"):
        reference_source.resolve_reference_binding({"reference": binding})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("semantic_binding", "npubench_old_format_task_bundle"),
        ("runner_contract_version", "npubench/v1"),
    ],
)
def test_a3_live_binding_altered_field_fails_closed(field, value):
    binding = reference_source.explicit_a3_live_binding()
    binding[field] = value
    with pytest.raises(reference_source.ReferenceSourceError):
        reference_source.resolve_reference_binding({"reference": binding})


def test_a3_live_binding_rejects_unexpected_extra_field():
    binding = reference_source.explicit_a3_live_binding()
    binding["task_sha256"] = "a" * 64
    with pytest.raises(reference_source.ReferenceSourceError):
        reference_source.resolve_reference_binding({"reference": binding})


def test_complete_a3_live_binding_is_the_only_live_truth_marker():
    state = {"reference": reference_source.explicit_a3_live_binding()}
    assert reference_source.uses_live_a3_reference(state)
    assert not reference_source.uses_npubench_reference(state)
    assert not reference_source.uses_cannbench_reference(state)


def test_explicit_legacy_migration_is_side_effect_free_and_required():
    original = {"opgen_mode": "port_a3_to_a5"}
    migrated = reference_source.migrate_legacy_a3_state(original)
    assert "reference" not in original
    assert migrated["reference"] == reference_source.explicit_a3_live_binding()
    with pytest.raises(reference_source.ReferenceSourceError):
        reference_source.migrate_legacy_a3_state(migrated)
    partial = reference_source.migrate_legacy_a3_state(
        {"opgen_mode": "port_a3_to_a5", "reference": {"source": "a3_live"}}
    )
    assert partial["reference"] == reference_source.explicit_a3_live_binding()
