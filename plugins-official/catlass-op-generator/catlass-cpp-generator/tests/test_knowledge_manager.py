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
    workflow = knowledge.query_bundle(target, "workflow", [], None, None, "CrossCore", True)
    assert workflow["status"] == "passed"
    assert workflow["count"] >= 1
    assert all(item["path"].startswith("workflow/") for item in workflow["results"])
    assert all("tags" not in item for item in workflow["results"])

    operator = knowledge.query_bundle(target, None, [], "gdn", None, "矩阵逆", False)
    assert operator["count"] == 1
    assert operator["results"][0]["path"] == "operator/linear-attention/matrix-inverse-patterns.md"
    assert operator["results"][0]["consumers"] == []


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
    assert any("consumers must name existing stage skills" in item for item in report["errors"])


def test_reindex_preserves_a_valid_bundle(tmp_path: Path) -> None:
    target = tmp_path / ".catlass-cpp/knowledge"
    knowledge.initialize(BUNDLE, target)
    result = knowledge.reindex(target)
    assert result["count"] == 13
    assert knowledge.validate_bundle(target)["status"] == "passed"


def test_record_is_explicitly_unsupported_and_creates_no_partition(tmp_path: Path) -> None:
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
