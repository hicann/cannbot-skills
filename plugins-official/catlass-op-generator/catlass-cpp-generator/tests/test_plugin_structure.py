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

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SKILLS = {
    "catlass-cpp-interface",
    "catlass-cpp-reference",
    "catlass-cpp-design",
    "catlass-cpp-develop",
    "catlass-cpp-test",
    "catlass-cpp-knowledge",
}


def test_public_skill_set_is_exact() -> None:
    actual = {path.name for path in (ROOT / "skills").iterdir() if path.is_dir()}
    assert actual == EXPECTED_SKILLS
    for name in actual:
        text = (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"name: {name}" in text.split("---", 2)[1]


def test_manifests_have_consistent_identity_and_version() -> None:
    manifests = [
        ROOT / ".claude-plugin/plugin.json",
        ROOT / ".codex-plugin/plugin.json",
        ROOT / ".cursor-plugin/plugin.json",
    ]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    assert {item["name"] for item in payloads} == {"catlass-cpp-generator"}
    assert {item["version"] for item in payloads} == {"0.1.0"}


def test_stage_assets_have_one_owner() -> None:
    expected = {
        "skills/catlass-cpp-interface/scripts/validate_workflow.py",
        "skills/catlass-cpp-interface/templates/workflow.json",
        "skills/catlass-cpp-reference/scripts/generate_definition.py",
        "skills/catlass-cpp-reference/scripts/validate_reference.py",
        "skills/catlass-cpp-design/scripts/validate_design.py",
        "skills/catlass-cpp-test/scripts/compare_precision.py",
    }
    assert all((ROOT / path).is_file() for path in expected)
    names = [
        "validate_workflow.py",
        "generate_definition.py",
        "validate_reference.py",
        "validate_design.py",
        "compare_precision.py",
    ]
    for name in names:
        assert len(list((ROOT / "skills").rglob(name))) == 1


def test_pr1069_evals_are_distributed_once() -> None:
    ids = []
    for path in (ROOT / "skills").glob("catlass-cpp-*/evals/evals.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["skill_name"] == path.parents[1].name
        ids.extend(item["id"] for item in payload["evals"])
    assert sorted(ids) == list(range(1, 11))
    assert len(ids) == len(set(ids))


def test_no_sixth_stage_or_duplicate_technical_references() -> None:
    assert not (ROOT / "skills/catlass-cpp-optimize").exists()
    for skill in EXPECTED_SKILLS - {"catlass-cpp-knowledge"}:
        assert not (ROOT / "skills" / skill / "references").exists()
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "性能不达标仍属于 validation" in agents
    assert not re.search(r"catlass-cpp-(?:optimize|bench)", agents)
