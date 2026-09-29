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
from pathlib import Path


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
    assert report["count"] == 13


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
