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
"""Regression tests for the Step 6 pytest generator prompt."""

from __future__ import annotations

import unittest
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parents[1] / "references" / "prompts" / "S6-pytest-generator.md"


class S6PytestGeneratorPromptTest(unittest.TestCase):
    def test_cases_file_option_is_registered_before_test_module_load(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("conftest.py", prompt)
        self.assertIn("pytest_addoption", prompt)
        self.assertIn("--cases-file", prompt)
        self.assertIn("禁止在 `S6_test_{op_name}.py` 中定义 `pytest_addoption`", prompt)

    def test_prompt_declares_execution_order(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("执行顺序", prompt)
        self.assertIn("Step 6a", prompt)
        self.assertIn("Step 6b", prompt)
        self.assertIn("Step 6c", prompt)
        self.assertIn("前置", prompt)
        self.assertIn("完成标志", prompt)

    def test_prompt_requires_dtype_safe_generation_for_non_floating_inputs(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("dtype.is_floating_point", prompt)
        self.assertIn("torch.randint", prompt)
        self.assertIn("torch.bool", prompt)
        self.assertIn("randn", prompt)


if __name__ == "__main__":
    unittest.main()
