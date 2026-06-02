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


class WorkflowDocsTest(unittest.TestCase):
    def test_spec_gen_commands_use_existing_skill_path(self) -> None:
        prompt = (WORKFLOW_DIR / "resources/task-prompts.md").read_text(encoding="utf-8")

        self.assertNotIn("ops/skills/ops-spec-gen", prompt)
        self.assertIn("ops/ops-spec-gen/scripts/generate_spec.py", prompt)
        self.assertIn("ops/ops-spec-gen/scripts/validate_spec.py", prompt)

    def test_cp2_validator_setup_does_not_reference_missing_requirements_file(self) -> None:
        skill = (WORKFLOW_DIR / "SKILL.md").read_text(encoding="utf-8")

        self.assertNotIn("workflow/resources/requirements.txt", skill)


if __name__ == "__main__":
    unittest.main()
