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
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-cpp-knowledge/scripts/record_knowledge.py"
SPEC = importlib.util.spec_from_file_location("cpp_knowledge", SCRIPT)
knowledge = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(knowledge)
BUNDLE = ROOT / "knowledge"


def test_builtin_bundle_validates() -> None:
    report = knowledge.validate_bundle(BUNDLE)
    assert report["status"] == "passed", report["errors"]
    assert report["okf_version"] == "0.2"
    assert report["count"] == 36


def test_business_partitions_are_exact() -> None:
    profile = knowledge._load_yaml(BUNDLE / "bundle-profile.yaml")
    assert set(profile["business_partitions"]) == {"catlass", "workflow", "operator"}
    assert {path.name for path in BUNDLE.iterdir() if path.is_dir()} == {
        "catlass",
        "workflow",
        "operator",
    }
    assert not (BUNDLE / "learned").exists()
    assert not (BUNDLE / "dsl").exists()


def test_all_concepts_have_required_sections() -> None:
    for path in knowledge._concept_paths(BUNDLE):
        data, body = knowledge._frontmatter(path)
        assert knowledge.REQUIRED_FIELDS <= set(data)
        for heading in knowledge.COMMON_HEADINGS:
            assert heading in body
        if data["type"] == "operator":
            for heading in knowledge.OPERATOR_HEADINGS:
                assert heading in body
        if data["type"] == "workflow":
            assert data["consumers"]
            assert all(item.startswith("catlass-cpp-") for item in data["consumers"])


def test_block_sparse_attention_stays_within_a2a3_scope() -> None:
    for path in (BUNDLE / "operator/sparse-attention").glob("*.md"):
        if path.name == "index.md":
            continue
        data, body = knowledge._frontmatter(path)
        assert "provenance" not in data
        assert data["architectures"] == ["DAV_2201"]
        # 归因仅用审计编号：脚注不得包含可读取的外部资源链接
        for footnote in re.findall(r"^\[\^[^\]]+\]:.*$", body, re.MULTILINE):
            assert "http" not in footnote, path
        # Public guidance must not require the private experiment ledger.
        assert not re.search(
            r"EXP-[A-Z0-9]+-\d+|\bround\s*\d+|\[S\d{2}|legacy_evidence_ids|/(?:home|root|Users)/",
            body,
            re.IGNORECASE,
        ), path


def test_bsag_knowledge_stays_in_its_ascend950_operator_family() -> None:
    family = BUNDLE / "operator/block-sparse-attention-grad"
    concepts = [path for path in family.glob("*.md") if path.name != "index.md"]
    assert [path.name for path in concepts] == ["block-sparse-patterns.md"]
    data, body = knowledge._frontmatter(concepts[0])
    assert data["operator_families"] == ["block-sparse-attention-grad"]
    assert data["architectures"] == ["ascend950"]
    assert "provenance" not in data
    assert not re.search(
        r"EXP-[A-Z0-9]+-\d+|\bround\s*\d+|legacy_evidence_ids|/(?:home|root|Users)/",
        body,
        re.IGNORECASE,
    )


def _heading_anchors(text: str) -> set[str]:
    """Resolve the simple ATX heading/inline-code form used by these documents."""
    anchors = set()
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE):
        base = re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")
        anchor = base
        suffix = 0
        while anchor in anchors:
            suffix += 1
            anchor = f"{base}-{suffix}"
        anchors.add(anchor)
    anchors.update(re.findall(r'<a\s+(?:name|id)=["\x27]([^"\x27]+)', text))
    return anchors


@pytest.mark.parametrize("family", ["sparse-attention", "sparse-flash-mla"])
def test_family_links_are_self_contained_after_runtime_reindex(
    tmp_path: Path,
    family: str,
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    knowledge.reindex(target)
    for path in (target / "operator" / family).rglob("*.md"):
        for link in re.findall(
            r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")
        ):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (
                (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            )
            assert destination.is_relative_to(target.resolve()), (path.name, link)
            assert destination.is_file(), (path.name, link)
            if parsed.fragment:
                assert unquote(parsed.fragment) in _heading_anchors(
                    destination.read_text(encoding="utf-8")
                ), (path.name, link)


def test_sparse_family_relative_links_resolve() -> None:
    family = BUNDLE / "operator/sparse-flash-mla"
    for document in family.glob("*.md"):
        for target in re.findall(
            r"\[[^\]]+\]\(([^)#]+)", document.read_text(encoding="utf-8")
        ):
            if "://" not in target:
                assert (document.parent / target).exists(), (document.name, target)


def test_sparse_family_runtime_contains_knowledge_without_executables(
    tmp_path: Path,
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    family = target / "operator/sparse-flash-mla"
    files = [path for path in family.rglob("*") if path.is_file()]
    assert files
    assert all(path.suffix == ".md" or path.name == "workflow.json" for path in files)
    assert not (family / "tools").exists()
    assert not (family / "materials").exists()


@pytest.mark.parametrize(
    "query_text, concept",
    [
        ("五阶段", "workflow.md"),
        ("aclnnSparseFlashMlaMetadataGetWorkspaceSize", "interface.md"),
        ("INT32", "metadata.md"),
        ("公共 暂存", "development.md"),
        ("INCONCLUSIVE", "pipeline.md"),
        ("minimum_speedup", "performance.md"),
        ("full-K manifest", "validation.md"),
    ],
)
def test_sparse_topics_are_retrievable_after_runtime_reindex(
    tmp_path: Path,
    query_text: str,
    concept: str,
) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    knowledge.reindex(target)
    report = knowledge.query_bundle(
        target,
        "operator",
        [],
        "sparse_flash_mla",
        "atlas_a2_a3",
        query_text,
        True,
    )
    expected = f"operator/sparse-flash-mla/{concept}"
    assert expected in {item["path"] for item in report["results"]}
    retrieved = knowledge.get_concept(target, expected)
    assert retrieved["content"] == (BUNDLE / expected).read_text(encoding="utf-8")
