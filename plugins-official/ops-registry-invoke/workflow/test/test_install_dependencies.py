#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for ops-registry-invoke installer dependency coverage."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PLUGIN_ROOT.parents[1]
SHARED_OPS_ROOT = REPO_ROOT / "ops"


def parse_yaml_list(path: Path, key: str) -> list[str]:
    values: list[str] = []
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(rf"^{re.escape(key)}:", line):
            in_section = True
            continue
        if line and not line.startswith(" "):
            in_section = False
        if in_section and re.match(r"^ +- ", line):
            values.append(re.sub(r"^ +- ", "", line).strip())
    return [value for value in values if value]


def installed_skills_from_init_inputs() -> set[str]:
    direct_skills = parse_yaml_list(PLUGIN_ROOT / "AGENTS.md", "skills")
    direct_agents = parse_yaml_list(PLUGIN_ROOT / "AGENTS.md", "agents")
    skills = set(direct_skills)
    for agent in direct_agents:
        agent_path = PLUGIN_ROOT / "agents" / f"{agent}.md"
        if agent_path.is_file():
            skills.update(parse_yaml_list(agent_path, "skills"))
    return skills


def declared_skill_names() -> set[str]:
    skill_files = list(SHARED_OPS_ROOT.glob("*/SKILL.md"))
    skill_files.extend((PLUGIN_ROOT / "skills").glob("*/SKILL.md"))
    skill_files.append(PLUGIN_ROOT / "workflow" / "SKILL.md")

    names: set[str] = set()
    for path in skill_files:
        for line in path.read_text(encoding="utf-8").splitlines()[:8]:
            match = re.match(r"name:\s*([A-Za-z0-9_.-]+)", line.strip())
            if match:
                names.add(match.group(1))
                break
    return names


def workflow_referenced_skills() -> set[str]:
    workflow_files = [PLUGIN_ROOT / "workflow" / "SKILL.md"]
    workflow_files.extend((PLUGIN_ROOT / "workflow" / "resources").glob("*.md"))
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in workflow_files)
    return {name for name in declared_skill_names() if name in workflow_text}


class InstallDependencyTest(unittest.TestCase):
    def test_workflow_referenced_skills_are_installed(self) -> None:
        missing = workflow_referenced_skills() - installed_skills_from_init_inputs()

        self.assertEqual(set(), missing)


if __name__ == "__main__":
    unittest.main()
