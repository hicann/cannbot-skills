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
"""Regression tests for ops-registry-invoke workflow documentation."""

from __future__ import annotations

import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
PLUGIN_DIR = WORKFLOW_DIR.parent


class WorkflowDocsTest(unittest.TestCase):
    def test_spec_gen_commands_use_existing_skill_path(self) -> None:
        # 分解式设计增强后，spec 生成/校验脚本的引用从 task-prompts.md 下沉到 SKILL.md
        # 门禁（validate_spec.py 用正确全路径 ops/ops-spec-gen/scripts/）与 architect agent
        # （按脚本名引用 generate_spec.py / validate_spec.py）。原测试的意图保持不变：
        # 错误路径 ops/skills/ops-spec-gen 不得出现、两个 spec 脚本被正确引用。
        skill = (WORKFLOW_DIR / "SKILL.md").read_text(encoding="utf-8")
        prompt = (WORKFLOW_DIR / "resources/task-prompts.md").read_text(encoding="utf-8")
        architect = (PLUGIN_DIR / "agents/ascendc-ops-architect.md").read_text(encoding="utf-8")
        docs = skill + prompt + architect

        self.assertNotIn("ops/skills/ops-spec-gen", docs)
        self.assertIn("ops/ops-spec-gen/scripts/validate_spec.py", skill)
        self.assertIn("generate_spec.py", docs)
        self.assertIn("validate_spec.py", docs)

    def test_cp2_validator_requirements_file_exists_when_referenced(self) -> None:
        # 增强后新增 workflow/resources/requirements.txt（validate_workflow_state.py 依赖）。
        # 原测试断言 SKILL.md 不得引用该文件（因当时文件缺失，引用会导致 setup 失败）；
        # 现该文件已真实存在，守护点改为“被引用则必须存在”，保留原“不引用缺失文件”的意图。
        skill = (WORKFLOW_DIR / "SKILL.md").read_text(encoding="utf-8")
        req = WORKFLOW_DIR / "resources/requirements.txt"

        if "workflow/resources/requirements.txt" in skill:
            self.assertTrue(
                req.is_file(),
                "SKILL.md 引用了 workflow/resources/requirements.txt 但该文件不存在",
            )


if __name__ == "__main__":
    unittest.main()
