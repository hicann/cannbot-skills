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
"""Validate ops-registry-invoke workflow checklists.

Deterministic script replacements for LLM-based checklist verification.
Each --stage corresponds to a workflow checkpoint's checklist items.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent
if str(_RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_DIR))
from _output_log import get_logger

_LOGGER = get_logger("ops_registry_invoke.validate_checklist")


SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

STAGE_CHOICES = [
    "init",
    "requirements",
    "design",
    "design-review",
    "test-design",
    "test-review",
    "doc-examples",
    "wave1",
    "wave2",
]


class ChecklistValidator:
    # ==== construction ===========================================================================

    def __init__(self, operator_dir: Path) -> None:
        self.operator_dir = operator_dir
        self.errors: list[str] = []

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    # ==== public stage validators (dispatch order) ===============================================

    def validate_init(self) -> None:
        if not self._require_operator_dir():
            return

        dir_name = self.operator_dir.name
        if not SNAKE_CASE_PATTERN.match(dir_name):
            self.add_error(
                f"operator directory name is not snake_case: {dir_name!r} "
                f"(expected pattern: {SNAKE_CASE_PATTERN.pattern})"
            )

        self._require_non_empty_file("docs/LOG.md")

        if not (self.operator_dir / "issues").is_dir():
            self.add_error("missing required directory: issues/")

    def validate_requirements(self) -> None:
        if not self._require_operator_dir():
            return

        self._require_non_empty_file("docs/REQUIREMENTS.md")

        aclnn_docs = list((self.operator_dir / "docs").glob("aclnn*.md"))
        if not aclnn_docs:
            self.add_error("missing required file: docs/aclnn*.md (no aclnn*.md found)")

    def validate_design(self) -> None:
        if not self._require_operator_dir():
            return

        if self._require_non_empty_file("docs/DESIGN_PREP.md"):
            text = self._read_text("docs/DESIGN_PREP.md")
            self._require_sections(
                "docs/DESIGN_PREP.md",
                text,
                ["路线决策", "模板选型", "API 验证记录", "UB 预算"],
            )

        self._validate_design_sections()

        if self._require_non_empty_file("docs/DESIGN.md"):
            text = self._read_text("docs/DESIGN.md")
            self._require_sections(
                "docs/DESIGN.md",
                text,
                ["spec.yaml 一致性映射"],
            )

        self._require_non_empty_file("docs/PLAN.md")

        self._validate_paradigm_trace()

    def validate_design_review(self) -> None:
        if not self._require_operator_dir():
            return

        self._check_review_report(
            "tmp/checks/DESIGN_REVIEW.md",
            ["DESIGN-SPEC-1", "spec.yaml 一致性映射"],
        )

    def validate_test_design(self) -> None:
        if not self._require_operator_dir():
            return

        if self._require_non_empty_file("docs/TEST.md"):
            text = self._read_text("docs/TEST.md")
            self._require_sections("docs/TEST.md", text, ["spec.yaml 测试映射"])

        for name in (
            "03_参数定义.yaml",
            "04_测试因子.yaml",
            "05_约束定义.yaml",
        ):
            self._require_non_empty_file(f"tests/st/design/{name}")

        self._validate_testcase_levels()

    def validate_test_review(self) -> None:
        if not self._require_operator_dir():
            return

        self._check_review_report(
            "tmp/checks/TEST_REVIEW.md",
            ["TEST-SPEC", "spec.yaml 测试映射"],
        )

    def validate_doc_examples(self) -> None:
        if not self._require_operator_dir():
            return

        operator_name = self.operator_dir.name

        self._require_non_empty_file("README.md")

        examples_dir = self.operator_dir / "examples"
        if not examples_dir.is_dir():
            self.add_error("missing required directory: examples/")
            return

        for filename in (
            f"test_aclnn_{operator_name}.cpp",
            f"test_geir_{operator_name}.cpp",
            "CMakeLists.txt",
            "run.sh",
        ):
            self._require_non_empty_file(f"examples/{filename}")

        build_dir = examples_dir / "build"
        for binary_name in (
            f"test_aclnn_{operator_name}",
            f"test_geir_{operator_name}",
        ):
            if not (build_dir / binary_name).is_file():
                self.add_error(
                    f"missing build artifact: examples/build/{binary_name} "
                    f"(run 'examples/run.sh' to build)"
                )

        run_log_path = examples_dir / "run.log"
        if not run_log_path.is_file():
            self.add_error(
                "missing required file: examples/run.log "
                "(run 'examples/run.sh 2>&1 | tee examples/run.log')"
            )
        else:
            log_text = run_log_path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?i)\bfail\b", log_text):
                self.add_error("examples/run.log contains FAIL (example execution failed)")

    def validate_wave1(self) -> None:
        if not self._require_operator_dir():
            return

        self._validate_wave1_build()
        self._validate_wave1_st()
        self._validate_wave1_probe()

    def validate_wave2(self) -> None:
        if not self._require_operator_dir():
            return

        self._validate_wave2_ut_report()
        self._validate_wave2_probe()

    # ==== reporting ==============================================================================

    def print_result(self, stage: str) -> int:
        _LOGGER.info(f"stage: {stage}")
        if self.errors:
            _LOGGER.info("errors:")
            for error in self.errors:
                _LOGGER.info(f"- {error}")
            _LOGGER.info("STATUS: FAILED")
            return 1
        _LOGGER.info("STATUS: PASSED")
        return 0

    # ==== shared helpers =========================================================================

    def _require_operator_dir(self) -> bool:
        """Return True when the operator directory exists; record an error otherwise."""
        if self.operator_dir.is_dir():
            return True
        self.add_error(f"operator directory does not exist: {self.operator_dir}")
        return False

    def _require_non_empty_file(self, rel_path: str) -> bool:
        """Ensure a file exists and is non-empty. Return True when the check passes."""
        path = self.operator_dir / rel_path
        if not path.is_file():
            self.add_error(f"missing required file: {rel_path}")
            return False
        if path.stat().st_size == 0:
            self.add_error(f"{rel_path} is empty")
            return False
        return True

    def _read_text(self, rel_path: str) -> str:
        return (self.operator_dir / rel_path).read_text(encoding="utf-8", errors="replace")

    def _require_sections(self, rel_path: str, text: str, sections: list[str]) -> None:
        for section in sections:
            if section not in text:
                self.add_error(f"{rel_path} missing required section: {section}")

    def _check_review_report(self, rel_path: str, required_sections: list[str]) -> None:
        if not self._require_non_empty_file(rel_path):
            return
        text = self._read_text(rel_path)
        if not re.search(r"\*\*状态\*\*:", text):
            self.add_error(f"{rel_path} missing required field: **状态**:")
        self._require_sections(rel_path, text, required_sections)

    # ==== design helpers =========================================================================

    def _validate_design_sections(self) -> None:
        sections_dir = self.operator_dir / ".spec-to-design" / "sections"
        if not sections_dir.is_dir():
            self.add_error("missing required directory: .spec-to-design/sections/")
            return
        for name in (
            "01-overview-contract.md",
            "02-architecture.md",
            "03-implementation.md",
            "04-quality-plan.md",
            "05-plan.md",
        ):
            self._require_non_empty_file(f".spec-to-design/sections/{name}")

    def _validate_paradigm_trace(self) -> None:
        log = self.operator_dir / "docs" / "LOG.md"
        if log.is_file() and log.stat().st_size > 0:
            if "paradigm" not in self._read_text("docs/LOG.md").lower():
                self.add_error("docs/LOG.md missing paradigm trace in 1.3 record")
        else:
            self.add_error("docs/LOG.md not found or empty, cannot verify paradigm trace")

    # ==== test-design helpers ====================================================================

    def _validate_testcase_levels(self) -> None:
        testcase_dir = self.operator_dir / "tests" / "st" / "testcases"
        if not testcase_dir.is_dir():
            self.add_error("missing required directory: tests/st/testcases/")
            return

        level_patterns = {
            "L0": "*_l0_test_cases.csv",
            "L1": "*_l1_test_cases.csv",
            "L2": "*_l2_test_cases.csv",
        }
        for _, pattern in level_patterns.items():
            matches = list(testcase_dir.glob(pattern))
            if not matches:
                self.add_error(f"missing required file: tests/st/testcases/{pattern}")
            elif matches[0].stat().st_size == 0:
                self.add_error(f"tests/st/testcases/{matches[0].name} is empty")

    # ==== wave1 helpers ==========================================================================

    def _validate_wave1_build(self) -> None:
        build_dir = self.operator_dir / "build"

        if not (build_dir / "CMakeCache.txt").is_file():
            self.add_error("build/CMakeCache.txt not found (cmake configuration incomplete)")

        run_packages = list(build_dir.glob("custom_opp_*.run"))
        if not run_packages:
            self.add_error("build/custom_opp_*.run not found (build package not generated)")
        else:
            for pkg in run_packages:
                if pkg.stat().st_size == 0:
                    self.add_error(f"{pkg.relative_to(self.operator_dir)} is empty")

        host_libs = list((build_dir / "op_host").glob("libcust_*.so"))
        if not host_libs:
            self.add_error("build/op_host/libcust_*.so not found (host compilation incomplete)")

        kernel_glob = (build_dir / "op_kernel" / "ascendc_kernels" / "binary").glob("**/*.o")
        if not list(kernel_glob):
            self.add_error(
                "build/op_kernel/ascendc_kernels/binary/**/*.o not found "
                "(kernel binary not generated)"
            )

    def _validate_wave1_st(self) -> None:
        st_dir = self.operator_dir / "tests" / "st"
        if not st_dir.is_dir():
            self.add_error("tests/st/ not found (ST test project not generated)")
            return

        if not (st_dir / "CMakeLists.txt").is_file():
            self.add_error("tests/st/CMakeLists.txt not found")
        if not (st_dir / "run.sh").is_file():
            self.add_error("tests/st/run.sh not found")
        if not list(st_dir.glob("test_aclnn_*.cpp")):
            self.add_error("tests/st/test_aclnn_*.cpp not found")

        mock_build = st_dir / "build-mock"
        if not mock_build.is_dir():
            self.add_error("tests/st/build-mock/ not found (Mock compilation not executed)")
            return

        if not (mock_build / "CMakeCache.txt").is_file():
            self.add_error("tests/st/build-mock/CMakeCache.txt not found (Mock cmake incomplete)")
        mock_binaries = [
            b for b in mock_build.glob("test_aclnn_*") if b.is_file() and not b.suffix
        ]
        if not mock_binaries:
            self.add_error(
                "tests/st/build-mock/test_aclnn_* binary not found (Mock compilation incomplete)"
            )

    def _validate_wave1_probe(self) -> None:
        probe_dir = self.operator_dir / "probe"
        has_probe = probe_dir.is_dir() and any(probe_dir.iterdir())
        if not has_probe:
            return

        summary_path = probe_dir / "PROBE_SUMMARY.md"
        if not summary_path.is_file():
            self.add_error("probe/PROBE_SUMMARY.md not found")
        else:
            text = summary_path.read_text(encoding="utf-8", errors="replace")
            rows = re.findall(r"\|\s*\S.*?\s*\|\s*(✅|❌)\s*\|\s*\d+\s*\|", text)
            if not rows:
                self.add_error("probe/PROBE_SUMMARY.md contains no probe result rows")
            elif not all(status == "✅" for status in rows):
                failed = sum(1 for s in rows if s != "✅")
                self.add_error(f"probe success rate != 100% ({failed}/{len(rows)} failed)")

        for result_md in probe_dir.glob("*/RESULT.md"):
            content = result_md.read_text(encoding="utf-8", errors="replace")
            env_match = re.search(r"\*\*运行环境\*\*:\s*(.+)", content)
            if env_match and "Mock" in env_match.group(1):
                task_name = result_md.parent.name
                self.add_error(
                    f"probe/{task_name}/RESULT.md: runtime environment is Mock (expected NPU)"
                )

    # ==== wave2 helpers ==========================================================================

    def _validate_wave2_ut_report(self) -> None:
        rel_path = "tests/ut/test-report.json"
        if not self._require_non_empty_file(rel_path):
            return

        data = self._load_ut_report(rel_path)
        if data is None:
            return
        if not isinstance(data, dict):
            self.add_error(f"{rel_path} must be a JSON object")
            return

        status = str(data.get("status", "")).strip().lower()
        if status not in {"passed", "pass", "success", "ok"}:
            self.add_error(f"{rel_path} status is not passed: {data.get('status')!r}")

    def _load_ut_report(self, rel_path: str):
        try:
            return json.loads(self._read_text(rel_path))
        except ValueError as exc:
            self.add_error(f"{rel_path} is not valid JSON: {exc}")
            return None

    def _validate_wave2_probe(self) -> None:
        probe_path = self.operator_dir / "probe" / "PROBE_SUMMARY.md"
        if not probe_path.is_file():
            return

        text = probe_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\|\s*任务\s*\|", text):
            self.add_error("probe/PROBE_SUMMARY.md missing required table header: | 任务 |")
        if re.search(r"\|\s*❌\s*\|", text):
            self.add_error("probe/PROBE_SUMMARY.md contains failed tasks (❌)")

        for log_path in (self.operator_dir / "probe").glob("*.md"):
            if log_path.name == "PROBE_SUMMARY.md":
                continue
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            env_match = re.search(r"运行环境[：:]\s*(.+)", log_text)
            if env_match and env_match.group(1).strip().lower() == "mock":
                self.add_error(
                    f"probe/{log_path.name} runtime environment must not be Mock: "
                    f"{env_match.group(1).strip()!r}"
                )


STAGE_DISPATCH = {
    "init": ChecklistValidator.validate_init,
    "requirements": ChecklistValidator.validate_requirements,
    "design": ChecklistValidator.validate_design,
    "design-review": ChecklistValidator.validate_design_review,
    "test-design": ChecklistValidator.validate_test_design,
    "test-review": ChecklistValidator.validate_test_review,
    "doc-examples": ChecklistValidator.validate_doc_examples,
    "wave1": ChecklistValidator.validate_wave1,
    "wave2": ChecklistValidator.validate_wave2,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGE_CHOICES, required=True)
    parser.add_argument("--operator-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    operator_dir = args.operator_dir.resolve()
    validator = ChecklistValidator(operator_dir)
    STAGE_DISPATCH[args.stage](validator)
    return validator.print_result(args.stage)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
