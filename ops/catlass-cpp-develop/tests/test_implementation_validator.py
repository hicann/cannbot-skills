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
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "catlass-cpp-design/tests"))
from test_design_validator import valid_design  # noqa: E402


SCRIPT = ROOT / "catlass-cpp-develop/scripts/validate_implementation.py"
SPEC = importlib.util.spec_from_file_location("implementation_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validator)

QUICK = """## 02 标杆运行

| 范围 | 标杆运行 | 计算语义 | 精度策略 | 证据 | 结论 |
|---|---|---|---|---|---|
| 正常、边界和非法输入 | 通过 | 通过 | 通过 | CPU 标杆运行报告 | 通过 |

## 04 快速精度验证

| Stage | 设计校验 | 构建/运行 | 精度 | 证据 | 结论 |
|---|---|---|---|---|---|
| S0 | 通过 | 通过 | 通过 | 构建日志和定向精度报告 | 通过 |
"""


class ImplementationValidatorTests(unittest.TestCase):
    def test_design_workflow_and_quick_record_pass(self) -> None:
        workflow = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "implementation",
            "issue_type": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            design = root / "design.md"
            workflow_path = root / "workflow.json"
            validation = root / "validation.md"
            design.write_text(valid_design(), encoding="utf-8")
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            validation.write_text(QUICK, encoding="utf-8")
            self.assertEqual([], validator.validate(design, workflow_path, validation))

    def test_interface_stage_is_rejected(self) -> None:
        workflow = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "interface",
            "issue_type": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            design = root / "design.md"
            workflow_path = root / "workflow.json"
            validation = root / "validation.md"
            design.write_text(valid_design(), encoding="utf-8")
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            validation.write_text(QUICK, encoding="utf-8")
            errors = validator.validate(design, workflow_path, validation)
            self.assertTrue(any("stage=implementation" in error for error in errors))

    def test_formal_serial_architecture_is_rejected(self) -> None:
        workflow = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "implementation",
            "issue_type": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            design = root / "design.md"
            workflow_path = root / "workflow.json"
            validation = root / "validation.md"
            design.write_text(
                valid_design().replace("多 work 并行，流水推进", "正式设备方案逐 work 串行"),
                encoding="utf-8",
            )
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            validation.write_text(QUICK, encoding="utf-8")
            errors = validator.validate(design, workflow_path, validation)
            self.assertTrue(
                any("formal Stage design cannot use per-work serial execution" in error for error in errors),
                errors,
            )

    def test_diagnostic_serial_path_is_not_formal_architecture(self) -> None:
        workflow = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "implementation",
            "issue_type": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            design = root / "design.md"
            workflow_path = root / "workflow.json"
            validation = root / "validation.md"
            design.write_text(
                valid_design().replace(
                    "多 work 并行，流水推进", "多 work 并行，流水推进；串行仅用于独立诊断"
                ),
                encoding="utf-8",
            )
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            validation.write_text(QUICK, encoding="utf-8")
            self.assertEqual([], validator.validate(design, workflow_path, validation))


if __name__ == "__main__":
    unittest.main()
