# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-cpp-knowledge/scripts/record_knowledge.py"
SPEC = importlib.util.spec_from_file_location("cpp_knowledge_manager", SCRIPT)
knowledge = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(knowledge)
BUNDLE = ROOT / "knowledge"


def test_initialize_is_idempotent_and_preserves_project_changes(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    copied = knowledge.initialize(BUNDLE, target)
    assert copied
    concept = target / "catlass/block-mmad.md"
    concept.write_text("project-owned change\n", encoding="utf-8")
    copied_again = knowledge.initialize(BUNDLE, target)
    assert copied_again == []
    assert concept.read_text(encoding="utf-8") == "project-owned change\n"


def test_query_supports_workflow_type_and_family_alias(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    workflow = knowledge.query_bundle(
        target, "workflow", [], None, None, "CrossCore", True
    )
    assert workflow["status"] == "passed"
    assert workflow["count"] >= 1
    assert all(item["path"].startswith("workflow/") for item in workflow["results"])
    assert all("tags" not in item for item in workflow["results"])

    operator = knowledge.query_bundle(target, None, [], "gdn", None, "矩阵逆", False)
    assert operator["count"] == 1
    assert (
        operator["results"][0]["path"]
        == "operator/linear-attention/matrix-inverse-patterns.md"
    )
    assert operator["results"][0]["consumers"] == []


@pytest.mark.parametrize(
    "relative",
    [
        "catlass/block-mmad.md",
        "workflow/interface-and-golden-contract.md",
        "operator/linear-attention/matrix-inverse-patterns.md",
    ],
)
def test_editorial_metadata_cannot_be_omitted(tmp_path: Path, relative: str) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    concept = target / relative
    data, body = knowledge._frontmatter(concept)
    for field in ("status", "generated", "verified"):
        data.pop(field, None)
    concept.write_text(
        "---\n" + yaml.safe_dump(data, allow_unicode=True) + "---\n" + body,
        encoding="utf-8",
    )
    report = knowledge.validate_bundle(target)
    assert report["status"] == "failed"
    assert any("missing fields" in error for error in report["errors"])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("status", "approved", "invalid status"),
        ("status", [], "invalid status"),
        ("generated", [], "generated must be a mapping"),
        ("verified", {}, "verified must be a list"),
    ],
)
def test_editorial_metadata_values_are_validated(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    concept = target / "operator/sparse-attention/semantics-and-dispatch.md"
    data, body = knowledge._frontmatter(concept)
    data[field] = value
    concept.write_text(
        "---\n" + yaml.safe_dump(data, allow_unicode=True) + "---\n" + body,
        encoding="utf-8",
    )
    report = knowledge.validate_bundle(target)
    assert report["status"] == "failed"
    assert any(message in error for error in report["errors"])


@pytest.mark.parametrize(
    "sources,message",
    [
        ([], "sources must be a non-empty list"),
        ({}, "sources must be a non-empty list"),
        (None, "sources must be a non-empty list"),
        ([{}], "invalid source entry"),
        (
            [
                {
                    "id": "example",
                    "resource": "example.md",
                    "title": "Example",
                    "kind": "repository",
                }
            ],
            "missing source footnote example",
        ),
    ],
)
def test_sources_are_checked_when_declared(
    tmp_path: Path, sources: object, message: str
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    concept = target / "operator/sparse-attention/semantics-and-dispatch.md"
    data, body = knowledge._frontmatter(concept)
    data["sources"] = sources
    concept.write_text(
        "---\n" + yaml.safe_dump(data, allow_unicode=True) + "---\n" + body,
        encoding="utf-8",
    )
    report = knowledge.validate_bundle(target)
    assert report["status"] == "failed"
    assert any(message in error for error in report["errors"])


@pytest.mark.parametrize(
    "family",
    [
        "sparse-attention",
        "block_sparse_attention",
        "BlockSparseAttention",
        "block-sparse-attention",
        "block_sparse_attention-arch22",
        "块稀疏注意力",
    ],
)
def test_block_sparse_attention_aliases_resolve_after_runtime_reindex(
    tmp_path: Path, family: str
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    knowledge.reindex(target)
    result = knowledge.query_bundle(
        target, "operator", [], family, "DAV_2201", None, True
    )
    expected = {
        "scope-and-evidence.md",
        "semantics-and-dispatch.md",
        "scheduling-and-pipeline.md",
        "catlass-components.md",
        "vector-epilogue-design.md",
        "golden-and-validation.md",
        "execution-and-measurement.md",
        "performance-hypotheses.md",
        "blockxy-generalization.md",
        "device-pipeline-and-reduce-facts.md",
        "host-workspace-and-batching-patterns.md",
        "optimization-admission-rules.md",
        "task-boundary-sync-patterns.md",
        "throughput-structure-rules.md",
    }
    assert {item["path"] for item in result["results"]} == {
        "operator/sparse-attention/" + name for name in expected
    }
    for item in result["results"]:
        assert (
            "content" not in item
        )  # Compact query must not eagerly return experiment bodies.
        assert item.get("status") == "stable"
        concept = knowledge.get_concept(target, item["path"])
        assert concept["content"] == (BUNDLE / item["path"]).read_text(encoding="utf-8")


def test_block_sparse_attention_queries_do_not_expand_architecture_or_linear_attention_scope(
    tmp_path: Path,
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    assert (
        knowledge.query_bundle(
            target, None, [], "block_sparse_attention", "ascend950", None, True
        )["count"]
        == 0
    )
    assert (
        knowledge.query_bundle(
            target, None, [], "gdn", None, "block_sparse_attention", True
        )["count"]
        == 0
    )
    result = knowledge.query_bundle(
        target, None, [], None, None, "sparse-attention", True
    )
    assert result["count"] == 14
    assert all(
        item["path"].startswith("operator/sparse-attention/")
        for item in result["results"]
    )


def test_get_rejects_escape_and_indexes(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    with pytest.raises(ValueError):
        knowledge.get_concept(target, "../outside.md")
    with pytest.raises(ValueError):
        knowledge.get_concept(target, "index.md")
    result = knowledge.get_concept(target, "catlass/block-mmad.md")
    assert result["status"] == "passed"
    assert result["path"] == "catlass/block-mmad.md"
    workflow = knowledge.get_concept(target, "workflow/precision-policy.md")
    assert workflow["status"] == "passed"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support is required")
def test_get_rejects_symlink_escape(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = target / "catlass/escape.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError):
        knowledge.get_concept(target, "catlass/escape.md")


def test_validate_rejects_unknown_workflow_consumer(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    concept = target / "workflow/interface-and-golden-contract.md"
    text = concept.read_text(encoding="utf-8")
    concept.write_text(
        text.replace("catlass-cpp-interface", "catlass-cpp-unknown", 1),
        encoding="utf-8",
    )
    report = knowledge.validate_bundle(target)
    assert report["status"] == "failed"
    assert any(
        "consumers must name existing stage skills" in item for item in report["errors"]
    )


def test_reindex_preserves_a_valid_bundle(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    concepts_before = {
        path.relative_to(target): path.read_bytes()
        for path in knowledge._concept_paths(target)
    }
    result = knowledge.reindex(target)
    assert result["count"] == len(concepts_before)
    assert {
        path.relative_to(target): path.read_bytes()
        for path in knowledge._concept_paths(target)
    } == concepts_before
    assert knowledge.validate_bundle(target)["status"] == "passed"


def test_record_is_explicitly_unsupported_and_creates_no_partition(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "record", "--project-root", str(tmp_path)],
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "unsupported"
    assert not (tmp_path / ".catlass-cpp/knowledge/learned").exists()
