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
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "catlass-cpp-test/scripts/validate_validation.py"
SPEC = importlib.util.spec_from_file_location("validation_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validator)
TEMPLATE = ROOT / "catlass-cpp-test/templates/validation.md.template"
PERFORMANCE_REFERENCE = ROOT / "catlass-cpp-test/references/performance_validation.md"

REFERENCE = """## 02 标杆运行

| 范围 | 标杆运行 | 计算语义 | 精度策略 | 证据 | 结论 |
|---|---|---|---|---|---|
| 正常、边界和非法输入 | 通过 | 通过 | 通过 | CPU 标杆运行报告 | 通过 |
"""

QUICK = """## 04 快速精度验证

| Stage | 设计校验 | 构建/运行 | 精度 | 证据 | 结论 |
|---|---|---|---|---|---|
| S0-S3 | 通过 | 通过 | 通过 | 构建日志和定向精度报告 | 通过 |
"""

FULL = """## V1 全量验收

| 覆盖项 | 来源 | 取值/分支 | 工作流用例 | 证据 | 结论 |
|---|---|---|---|---|---|
| dtype、shape、分支和边界 | `docs/api.md`、`docs/design.md` | 已声明支持范围 | 工程原生全量用例 | 用例清单和执行报告 | 通过 |

| 范围 | 功能 | 精度 | 边界 | 异常 | 证据 | 结论 |
|---|---|---|---|---|---|---|
| `docs/api.md` 支持范围 | 通过 | 通过 | 通过 | 通过 | 全量测试报告 | 通过 |
"""

PERF = """## V2 性能验收

| 设备/输入/采样 | 指标 | baseline | candidate | 目标 | 证据 | 结论 |
|---|---|---:|---:|---:|---|---|
| 目标 SoC；主模型 case；普通 msprof 采样5次，取中位数 | `BasicInfo Task Duration(us)` | 4100 us | 3900 us | 4000 us | 性能采样摘要 | 达标 |
"""

FAILED_PERF = """## V2 性能验收

| 设备/输入/采样 | 指标 | baseline | candidate | 目标 | 证据 | 结论 |
|---|---|---:|---:|---:|---|---|
| 目标 SoC；主模型 case；普通 msprof 采样5次，取中位数 | `BasicInfo Task Duration(us)` | 4100 us | 4200 us | 4000 us | 性能采样摘要 | 未达标 |
"""

CANDIDATES = """## V3 流水分析

### 当前流水瓶颈

瓶颈位于 Cube，证据为流水采样摘要。

### 候选池

| ID | 优化方案 | 影响 Stage/Pipe | 局部收益依据 | 预计整体收益(us) | 状态 | 状态依据/失效条件 |
|---|---|---|---|---:|---|---|
| C01 | 调整 Cube 调度 | Stage 2 / Cube | Cube 活跃周期预计减少 30 us | 30 us | selected | 当前关键路径且收益最大 |
| C02 | 增加 MTE 预取 | Stage 1 / MTE | MTE 搬运预计减少 20 us | 0 us | deferred | 当前被 Cube 关键路径掩盖 |

## V4 候选选择

本轮选择：C01

排序依据：预计整体耗时收益最大

对应 baseline：第 1 轮
"""

NO_ACTIVE_CANDIDATES = """## V3 流水分析

### 当前流水瓶颈

瓶颈位于 Cube，证据为流水采样摘要。

### 候选池

| ID | 优化方案 | 影响 Stage/Pipe | 局部收益依据 | 预计整体收益(us) | 状态 | 状态依据/失效条件 |
|---|---|---|---|---:|---|---|
| C01 | 增加 MTE 预取 | Stage 1 / MTE | MTE 搬运预计减少 20 us | 0 us | deferred | 当前被 Cube 关键路径掩盖 |
| C02 | 扩大 Cube Tile | Stage 2 / Cube | 资源测算显示无法实例化 | 0 us | rejected | L1 空间不足 |

## V4 候选选择

本轮无可执行候选。

原因：当前没有能够缩短整体耗时的 active 候选。
"""

PENDING = """## V1 全量验收

| 覆盖项 | 来源 | 取值/分支 | 工作流用例 | 证据 | 结论 |
|---|---|---|---|---|---|
| 待执行 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |

| 范围 | 功能 | 精度 | 边界 | 异常 | 证据 | 结论 |
|---|---|---|---|---|---|---|
| 待执行 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |

## V2 性能验收

| 设备/输入/采样 | 指标 | baseline | candidate | 目标 | 证据 | 结论 |
|---|---|---:|---:|---:|---|---|
| 待执行 | `BasicInfo Task Duration(us)` | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |

## V3 流水分析

待执行。

## V4 候选选择

待执行。
"""


def state(stage: str) -> dict[str, str]:
    return {"stage": stage}


class ValidationValidatorTests(unittest.TestCase):
    def test_performance_reference_contains_executable_collection_recipe(self) -> None:
        text = " ".join(PERFORMANCE_REFERENCE.read_text(encoding="utf-8").split())
        for required in (
            "for sample in 1 2 3 4 5",
            "msprof --output=",
            "--aic-mode=task-based",
            "op_summary_*.csv",
            "完整 kernel 序列",
            "Task Duration(us)`，取 5 次中位数",
        ):
            self.assertIn(required, text)

    def test_reference_stage_requires_executed_evidence(self) -> None:
        self.assertEqual([], validator.validate(REFERENCE, state("reference")))
        template = TEMPLATE.read_text(encoding="utf-8")
        errors = validator.validate(template, state("reference"))
        self.assertTrue(
            any("reference execution evidence" in error for error in errors)
        )

    def test_implementation_requires_completed_quick_evidence(self) -> None:
        self.assertEqual(
            [], validator.validate(REFERENCE + QUICK, state("implementation"))
        )
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertTrue(validator.validate(template, state("implementation")))

    def test_pending_validation_accepts_unexecuted_v1_and_v2(self) -> None:
        self.assertEqual(
            [], validator.validate(REFERENCE + QUICK + PENDING, state("validation"))
        )

    def test_validation_acceptance_rejects_pending_evidence(self) -> None:
        workflow = state("validation")
        errors = validator.validate(
            REFERENCE + QUICK + PENDING, workflow, "validation_accepted"
        )
        self.assertTrue(any("requires passed V1" in error for error in errors))
        self.assertTrue(any("requires passed V2" in error for error in errors))
        self.assertEqual(
            [],
            validator.validate(
                REFERENCE + QUICK + FULL + PERF, workflow, "validation_accepted"
            ),
        )

    def test_complete_requires_evidence_without_placeholders(self) -> None:
        complete = state("complete")
        self.assertEqual(
            [], validator.validate(REFERENCE + QUICK + FULL + PERF, complete)
        )
        errors = validator.validate(TEMPLATE.read_text(encoding="utf-8"), complete)
        self.assertTrue(any("requires passed V1 evidence" in error for error in errors))
        self.assertTrue(any("requires passed V2 evidence" in error for error in errors))

    def test_v1_pass_requires_full_case_coverage_evidence(self) -> None:
        without_coverage = FULL.split("| 范围", maxsplit=1)
        invalid = without_coverage[0].split("| 覆盖项", maxsplit=1)[0]
        invalid += "| 范围" + without_coverage[1]
        errors = validator.validate(
            REFERENCE + QUICK + invalid + PERF, state("complete")
        )
        self.assertTrue(any("full-case coverage" in error for error in errors), errors)

    def test_performance_result_cannot_precede_v1_evidence(self) -> None:
        pending_v1 = PENDING.split("## V2 性能验收", maxsplit=1)[0]
        errors = validator.validate(
            REFERENCE + QUICK + pending_v1 + PERF, state("validation")
        )
        self.assertTrue(
            any("V1 full validation must pass" in error for error in errors)
        )

    def test_failed_performance_requires_analysis_and_candidates(self) -> None:
        workflow = state("validation")
        errors = validator.validate(REFERENCE + QUICK + FULL + FAILED_PERF, workflow)
        self.assertTrue(any("V3" in error for error in errors))
        self.assertTrue(any("V4" in error for error in errors))
        analyzed = REFERENCE + QUICK + FULL + FAILED_PERF + CANDIDATES
        self.assertEqual([], validator.validate(analyzed, workflow))
        self.assertEqual(
            [], validator.validate(analyzed, workflow, "performance_candidate")
        )

    def test_deferred_candidate_is_retained_but_not_selected(self) -> None:
        record = REFERENCE + QUICK + FULL + FAILED_PERF + CANDIDATES
        self.assertEqual([], validator.validate(record, state("validation")))

    def test_selection_requires_largest_expected_overall_benefit(self) -> None:
        lower_priority_selection = CANDIDATES.replace(
            "| C02 | 增加 MTE 预取 | Stage 1 / MTE | MTE 搬运预计减少 20 us | "
            "0 us | deferred | 当前被 Cube 关键路径掩盖 |",
            "| C02 | 增加 MTE 预取 | Stage 1 / MTE | MTE 搬运预计减少 50 us | "
            "50 us | active | 当前进入关键路径 |",
        )
        errors = validator.validate(
            REFERENCE + QUICK + FULL + FAILED_PERF + lower_priority_selection,
            state("validation"),
            "performance_candidate",
        )
        self.assertTrue(
            any("largest expected overall benefit" in error for error in errors), errors
        )

    def test_deferred_candidate_cannot_be_selected(self) -> None:
        deferred_selection = CANDIDATES.replace(
            "| C01 | 调整 Cube 调度 | Stage 2 / Cube | Cube 活跃周期预计减少 30 us | "
            "30 us | selected | 当前关键路径且收益最大 |",
            "| C01 | 调整 Cube 调度 | Stage 2 / Cube | Cube 活跃周期预计减少 30 us | "
            "30 us | active | 当前关键路径且收益最大 |",
        ).replace("本轮选择：C01", "本轮选择：C02")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + FAILED_PERF + deferred_selection,
            state("validation"),
            "performance_candidate",
        )
        self.assertTrue(any("selected status" in error for error in errors), errors)

    def test_no_active_candidate_preserves_deferred_without_transition(self) -> None:
        record = REFERENCE + QUICK + FULL + FAILED_PERF + NO_ACTIVE_CANDIDATES
        self.assertEqual([], validator.validate(record, state("validation")))
        errors = validator.validate(
            record, state("validation"), "performance_candidate"
        )
        self.assertTrue(any("requires a selected" in error for error in errors), errors)

    def test_candidate_pool_requires_local_benefit_evidence(self) -> None:
        missing_evidence = CANDIDATES.replace("Cube 活跃周期预计减少 30 us", "")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + FAILED_PERF + missing_evidence,
            state("validation"),
        )
        self.assertTrue(any("局部收益依据" in error for error in errors), errors)

    def test_performance_values_must_be_numeric(self) -> None:
        invalid = PERF.replace("4100 us", "四千微秒")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + invalid, state("complete")
        )
        self.assertTrue(any("numeric baseline" in error for error in errors), errors)

    def test_performance_values_and_conclusions_must_be_canonical(self) -> None:
        invalid_value = PERF.replace("3900 us", "5 samples; median 4200 us")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + invalid_value, state("complete")
        )
        self.assertTrue(any("numeric candidate" in error for error in errors), errors)

        invalid_conclusion = PERF.replace("达标 |", "通过率 0% |")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + invalid_conclusion, state("complete")
        )
        self.assertTrue(any("passed or failed conclusion" in error for error in errors))

    def test_performance_requires_plain_msprof_five_sample_median(self) -> None:
        invalid = PERF.replace("普通 msprof 采样5次，取中位数", "其它采集方式")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + invalid, state("complete")
        )
        self.assertTrue(any("plain msprof" in error for error in errors), errors)
        self.assertTrue(any("sample 5 times" in error for error in errors), errors)
        self.assertTrue(any("use the median" in error for error in errors), errors)

    def test_passed_performance_must_meet_target(self) -> None:
        invalid = PERF.replace("3900 us", "4200 us")
        errors = validator.validate(
            REFERENCE + QUICK + FULL + invalid, state("complete")
        )
        self.assertTrue(any("must not exceed" in error for error in errors), errors)

    def test_result_word_is_not_accepted_as_evidence(self) -> None:
        invalid = QUICK.replace("构建日志和定向精度报告", "通过")
        errors = validator.validate(REFERENCE + invalid, state("implementation"))
        self.assertTrue(any("not only a result" in error for error in errors), errors)

    def test_acceptance_sections_must_be_ordered(self) -> None:
        errors = validator.validate(REFERENCE + QUICK + PERF + FULL, state("complete"))
        self.assertTrue(
            any("must follow 02, 04, V1, V2" in error for error in errors), errors
        )

    def test_cli_runs_the_full_workflow_validator(self) -> None:
        workflow = {
            "algorithm_family": "linear_attention",
            "target_architecture": "atlas_a2_a3",
            "stage": "complete",
            "issue_type": "performance_optimize",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path = root / "validation.md"
            workflow_path = root / "workflow.json"
            validation_path.write_text(
                REFERENCE + QUICK + FULL + PERF, encoding="utf-8"
            )
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "validate_validation.py",
                        "--validation",
                        str(validation_path),
                        "--workflow",
                        str(workflow_path),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(1, validator.main())


if __name__ == "__main__":
    unittest.main()
