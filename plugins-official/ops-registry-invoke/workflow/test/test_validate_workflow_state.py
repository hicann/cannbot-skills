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
"""Focused tests for validate_workflow_state.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "resources" / "validate_workflow_state.py"
SPEC = importlib.util.spec_from_file_location("validate_workflow_state", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
WorkflowValidator = MODULE.WorkflowValidator


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


class WorkflowValidatorTest(unittest.TestCase):
    @staticmethod
    def _whitebox_case_fixtures(
        *,
        include_data_range: bool,
        include_optional_absent_input: bool,
        high_expanded: bool,
    ) -> tuple[dict[str, object], list[dict[str, object]], list[str], list[str]]:
        input_spec = {"shape": [2, 8], "dtype": "float32"}
        if include_data_range:
            input_spec["_data_range"] = "normal"
        case = {
            "id": "case00001",
            "case_id": "case00001",
            "params": {"_group": "normal", "last_dim": 8},
            "tensors": {
                "inputs": {
                    "x": dict(input_spec),
                    "residual": dict(input_spec),
                    "gamma": dict(input_spec),
                },
                "outputs": {"y": {"shape": [2, 8], "dtype": "float32"}},
            },
        }
        if include_optional_absent_input:
            case["tensors"]["inputs"]["optional_bias"] = None
        high_case = json.loads(json.dumps(case))
        high_case["id"] = "case00001_x_zero"
        high_case["case_id"] = "case00001_x_zero"
        if include_data_range:
            high_case["tensors"]["inputs"]["x"]["_data_range"] = "zero"
        high_cases = [case, high_case] if high_expanded else [case]
        low_case_ids = [case["case_id"]]
        high_case_ids = [item["case_id"] for item in high_cases]
        return case, high_cases, low_case_ids, high_case_ids

    @staticmethod
    def _write_internal_whitebox_artifacts(
        whitebox_dir: Path,
        case: dict[str, object],
        include_auxiliary_whitebox_artifacts: bool,
    ) -> None:
        if include_auxiliary_whitebox_artifacts:
            write_text(whitebox_dir / "S2P0_scout_t.md", "# Scout tiling\n")
            write_text(whitebox_dir / "S2P0_scout_k.md", "# Scout kernel\n")
        write_json(
            whitebox_dir / "S2P0_file_manifest.json",
            {
                "verification": {"status": "pass"},
                "tiling": {"file_list": ["op_host/demo.cpp"]},
                "kernel": {"file_list": ["op_kernel/demo.h"], "total_key_count": 1},
            },
        )
        write_text(whitebox_dir / "S5_case_mapper.py", "def load_mapped_configs():\n    return []\n")
        if include_auxiliary_whitebox_artifacts:
            write_text(whitebox_dir / "S5_verify_mapper.py", "def main():\n    return 0\n")
        write_json(whitebox_dir / "S5_mapped_cases_path.json", {"cases": [case]})
        write_json(whitebox_dir / "S5_mapped_cases_network.json", {"cases": [case]})

    @staticmethod
    def _write_whitebox_contract_files(
        whitebox_dir: Path,
        case: dict[str, object],
        high_cases: list[dict[str, object]],
        include_internal_whitebox_artifacts: bool,
    ) -> None:
        write_json(
            whitebox_dir / "S2P2_param_def.json",
            {
                "tiling_keys": [101],
                "groups": [
                    {
                        "id": "group0",
                        "per_dtype": [{"dtype": "float32", "key": 101}],
                    }
                ],
            },
        )
        if include_internal_whitebox_artifacts:
            write_json(whitebox_dir / "S2P1_path_list.json", {"paths": []})
            write_json(whitebox_dir / "S2P1_operator_model.json", {"operator_model": {}})
            write_text(whitebox_dir / "S2P3_test_design.md", "# Whitebox design\n")
            write_json(whitebox_dir / "S3_verification_report.json", {"status": "pass", "checks": [], "issues": []})
        write_json(whitebox_dir / "S5_mapped_cases_low.json", {"cases": [case]})
        write_json(whitebox_dir / "S5_mapped_cases_high.json", {"cases": high_cases})
        write_json(
            whitebox_dir / "S6_tilingkey_coverage.json",
            {"total_keys": 1, "covered_keys": [101], "uncovered_keys": [], "coverage_rate": 1.0},
        )

    @staticmethod
    def _write_whitebox_result_files(
        results_dir: Path,
        include_st_result: bool,
        st_status: str,
        st_route: str,
        evidence_case_ids: list[str],
    ) -> None:
        if include_st_result:
            write_json(
                results_dir / "st_result.json",
                {
                    "results": [
                        {
                            "case_id": case_id,
                            "status": st_status,
                            "route": st_route,
                        }
                        for case_id in evidence_case_ids
                    ]
                },
            )
        write_json(results_dir / "pytest_collect.json", {"collected_case_ids": evidence_case_ids})
        write_json(
            results_dir / "pytest_result.json",
            {"results": [{"case_id": case_id, "status": "passed"} for case_id in evidence_case_ids]},
        )

    @staticmethod
    def _write_whitebox_execution_evidence(
        operator_dir: Path,
        st_evidence_mode: str,
        evidence_case_set: str,
        evidence_case_ids: list[str],
    ) -> None:
        evidence_ids = " ".join(evidence_case_ids)
        cases_file = "S5_mapped_cases_low.json" if evidence_case_set == "low" else "S5_mapped_cases_high.json"
        if st_evidence_mode == "id_marker_only":
            evidence_text = (
                f"IMPLEMENTATION_UNDER_TEST {cases_file} "
                "aclnnDemoGetWorkspaceSize op_host op_kernel\n"
                f"{evidence_ids}\n"
            )
        else:
            evidence_text = (
                "int main() {\n"
                f"  // IMPLEMENTATION_UNDER_TEST {cases_file} "
                f"aclnnDemoGetWorkspaceSize op_host op_kernel {evidence_ids}\n"
                f"  return RunCase(\"{evidence_case_ids[0]}\") ? 0 : 1;\n"
                "}\n"
            )
        write_text(operator_dir / "tests/st/test_whitebox_demo.cpp", evidence_text)

    @staticmethod
    def _write_report(operator_dir: Path, rel_path: str) -> None:
        write_text(operator_dir / rel_path, "# Report\n\n**状态**: ✅通过\n")

    @staticmethod
    def _write_ut_fixture(operator_dir: Path) -> None:
        write_json(
            operator_dir / "tests/ut/test-report.json",
            {
                "status": "passed",
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "results": [{"case_id": "ut_demo", "status": "passed"}],
            },
        )
        write_text(operator_dir / "tests/ut/test_demo.cpp", "aclnnDemo x residual gamma y\n")

    @staticmethod
    def _blackbox_passed_results() -> list[dict[str, object]]:
        return [
            {"case_id": "BB_L0_001", "level": "L0", "status": "passed", "route": "st_cpp"},
            {"case_id": "BB_L0_002", "level": "L0", "status": "passed", "route": "st_cpp"},
            {"case_id": "BB_L1_001", "level": "L1", "status": "passed", "route": "st_cpp"},
            {"case_id": "BB_L2_001", "level": "L2", "status": "passed", "route": "st_cpp"},
        ]

    @staticmethod
    def _write_evidence_index_fixture(operator_dir: Path) -> None:
        write_json(
            operator_dir / "tests/reports/evidence_index.json",
            {
                "status": "passed",
                "evidence_paths": [
                    "tests/st/case_manifest.json",
                    "tests/st/results/st_real_result.json",
                    "tests/whitebox/WORKFLOW_PROVENANCE.json",
                    "tests/whitebox/S2P2_param_def.json",
                    "tests/whitebox/S5_mapped_cases_low.json",
                    "tests/whitebox/S5_mapped_cases_high.json",
                    "tests/whitebox/S6_tilingkey_coverage.json",
                    "tests/whitebox/results/pytest_collect.json",
                    "tests/whitebox/results/pytest_result.json",
                    "tests/whitebox/results/st_result.json",
                ],
                "summary": {
                    "blackbox": {},
                    "whitebox": {},
                },
            },
        )

    @staticmethod
    def _write_blackbox_design_files(operator_dir: Path) -> None:
        write_text(
            operator_dir / "docs/TEST.md",
            """
# TEST

blackbox_case_targets:
  L0: 2
  L1: 1
  L2: 1
""",
        )
        write_json(
            operator_dir / "tests/st/LOW_COVERAGE_WAIVER.json",
            {
                "approved": True,
                "approved_by": "user",
                "reason": "Unit test fixture uses a small blackbox case set.",
                "scope": "blackbox_l1_minimum",
            },
        )
        write_text(
            operator_dir / "tests/st/testcases/blackbox_l0.csv",
            "case_id,level\nBB_L0_001,L0\nBB_L0_002,L0\n",
        )
        write_text(operator_dir / "tests/st/testcases/blackbox_l1.csv", "case_id,level\nBB_L1_001,L1\n")
        write_text(operator_dir / "tests/st/testcases/blackbox_l2.csv", "case_id,level\nBB_L2_001,L2\n")

    @staticmethod
    def _blackbox_manifest_cases() -> list[dict[str, object]]:
        return [
            {
                "case_id": "BB_L0_001",
                "level": "L0",
                "required": True,
                "accepted_routes": ["st_cpp"],
            },
            {
                "case_id": "BB_L0_002",
                "level": "L0",
                "required": True,
                "accepted_routes": ["st_cpp"],
            },
            {
                "case_id": "BB_L1_001",
                "level": "L1",
                "required": True,
                "accepted_routes": ["st_cpp"],
            },
            {
                "case_id": "BB_L2_001",
                "level": "L2",
                "required": True,
                "accepted_routes": ["st_cpp"],
            },
        ]

    @staticmethod
    def _write_st_target_count_generator(repo_dir: Path, default: int = 300) -> None:
        write_text(
            repo_dir / "ops/ascendc-st-design/scripts/generate_test_cases.py",
            f"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target-count", type=int, default={default})
""",
        )

    @staticmethod
    def _write_blackbox_target_document(operator_dir: Path, target_lines: str) -> Path:
        test_md = operator_dir / "docs/TEST.md"
        write_text(
            test_md,
            f"""
# TEST

blackbox_case_targets:
{target_lines}
""",
        )
        return test_md

    @staticmethod
    def _single_blackbox_execution_fixture(
        operator_dir: Path,
        source_text: str = "",
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        if source_text:
            write_text(operator_dir / "tests/st/test_demo.cpp", source_text)
        manifest = {
            "BB_L1_001": {
                "case_id": "BB_L1_001",
                "accepted_routes": ["st_cpp"],
            }
        }
        results = {"BB_L1_001": {"case_id": "BB_L1_001", "status": "passed", "route": "st_cpp"}}
        return manifest, results

    @staticmethod
    def _validate_l1_below_default_fixture(
        waiver: dict[str, object] | None = None,
        markdown_note: str = "",
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"
            operator_dir = repo_dir / "operators" / "demo"
            WorkflowValidatorTest._write_st_target_count_generator(repo_dir)
            test_md = WorkflowValidatorTest._write_blackbox_target_document(
                operator_dir,
                "  L0: 1\n  L1: 6\n  L2: 1",
            )
            if markdown_note:
                write_text(operator_dir / "tests/st/LOW_COVERAGE_JUSTIFICATION.md", markdown_note)
            if waiver is not None:
                write_json(operator_dir / "tests/st/LOW_COVERAGE_WAIVER.json", waiver)

            validator = WorkflowValidator(operator_dir)
            validator.blackbox_case_target_counts(test_md)
            return list(validator.errors)

    def test_blackbox_case_targets_use_ascendc_st_design_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"
            operator_dir = repo_dir / "operators" / "demo"
            self._write_st_target_count_generator(repo_dir)
            test_md = self._write_blackbox_target_document(operator_dir, "  L0: 1\n  L1: 299\n  L2: 1")

            validator = WorkflowValidator(operator_dir)
            targets = validator.blackbox_case_target_counts(test_md)

            self.assertEqual(targets["L1"], 299)
            self.assertTrue(
                any("minimum=300" in error for error in validator.errors),
                validator.errors,
            )

    def test_blackbox_case_targets_accept_ascendc_st_design_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"
            operator_dir = repo_dir / "operators" / "demo"
            self._write_st_target_count_generator(repo_dir)
            test_md = self._write_blackbox_target_document(operator_dir, "  L0: 1\n  L1: 300\n  L2: 1")

            validator = WorkflowValidator(operator_dir)
            targets = validator.blackbox_case_target_counts(test_md)

            self.assertEqual(targets["L1"], 300)
            self.assertEqual([], validator.errors)

    def test_blackbox_case_targets_accept_nested_target_count_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"
            operator_dir = repo_dir / "operators" / "demo"
            self._write_st_target_count_generator(repo_dir)
            test_md = self._write_blackbox_target_document(
                operator_dir,
                """  L0:
    target_count: 20
    file: operators/demo/tests/st/testcases/demo_L0_test_cases.csv
  L1:
    target_count: 300
    file: operators/demo/tests/st/testcases/demo_L1_test_cases.csv
  L2:
    target_count: 16
    file: operators/demo/tests/st/testcases/demo_L2_test_cases.csv""",
            )

            validator = WorkflowValidator(operator_dir)
            targets = validator.blackbox_case_target_counts(test_md)

            self.assertEqual({"L0": 20, "L1": 300, "L2": 16}, targets)
            self.assertEqual([], validator.errors)

    def test_blackbox_csv_rows_accept_testcase_name_as_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            write_text(
                operator_dir / "tests/st/testcases/aclnnDemo_l1_functional.csv",
                "testcase_name,api_name\n"
                "aclnnDemo_L1_001,aclnnDemo\n"
                ",aclnnDemo\n",
            )

            validator = WorkflowValidator(operator_dir)
            rows = validator.csv_case_rows()

            self.assertEqual(1, len(rows))
            self.assertEqual("aclnnDemo_L1_001", rows[0]["_case_id"])
            self.assertEqual("L1", rows[0]["_level"])

    def test_markdown_low_coverage_note_does_not_waive_l1_minimum(self) -> None:
        errors = self._validate_l1_below_default_fixture(
            markdown_note="# Low coverage\n\nDevelopment-only run with six L1 cases.\n",
        )

        self.assertTrue(any("minimum=300" in error for error in errors), errors)

    def test_user_approved_low_coverage_waiver_allows_l1_below_default(self) -> None:
        errors = self._validate_l1_below_default_fixture(
            {
                "approved": True,
                "approved_by": "user",
                "reason": "User explicitly requested a development-only low coverage run.",
                "scope": "blackbox_l1_minimum",
            },
        )

        self.assertFalse(any("minimum=300" in error for error in errors), errors)

    def test_unresolved_low_coverage_waiver_does_not_waive_l1_minimum(self) -> None:
        errors = self._validate_l1_below_default_fixture(
            {
                "approved": True,
                "approved_by": "user",
                "reason": "TODO",
                "scope": "blackbox_l1_minimum",
            },
        )

        self.assertTrue(any("minimum=300" in error for error in errors), errors)

    def test_invalid_low_coverage_waiver_scope_does_not_waive_l1_minimum(self) -> None:
        errors = self._validate_l1_below_default_fixture(
            {
                "approved": True,
                "approved_by": "user",
                "reason": "User approved a small case set.",
                "scope": "wrong_scope",
            },
        )

        self.assertTrue(any("minimum=300" in error for error in errors), errors)

    def test_blackbox_target_consistency_checks_csv_manifest_and_passed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = WorkflowValidator(Path(tmpdir))

            validator.validate_blackbox_target_consistency(
                "tests/st/results/st_real_result.json",
                targets={"L0": 1, "L1": 2, "L2": 1},
                csv_counts={"L0": 1, "L1": 1, "L2": 1},
                manifest_counts={"L0": 1, "L1": 2, "L2": 1},
                passed_level_counts={"L0": 1, "L1": 1, "L2": 1},
            )

            self.assertIn(
                "L1 blackbox CSV rows do not equal TEST.md target_count: actual=1, target=2",
                validator.errors,
            )
            self.assertIn(
                "tests/st/results/st_real_result.json L1 passed results do not equal "
                "TEST.md target_count: actual=1, target=2",
                validator.errors,
            )

    def test_ut_report_rejects_suite_only_results_without_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            write_json(
                operator_dir / "tests/ut/test-report.json",
                {
                    "status": "passed",
                    "summary": {"total": 1, "passed": 1, "failed": 0},
                    "results": [{"suite": "host_logic", "status": "passed"}],
                },
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_result_summary("tests/ut/test-report.json", label="UT")

            self.assertIn(
                "tests/ut/test-report.json UT result at index 0 lacks case_id",
                validator.errors,
            )

    def test_manifest_rejects_implemented_cases_schema_with_specific_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            write_json(
                operator_dir / "tests/st/case_manifest.json",
                {"implemented_cases": {"L0": [], "L1": [], "L2": []}},
            )

            validator = WorkflowValidator(operator_dir)
            validator.manifest_cases()

            self.assertIn(
                "tests/st/case_manifest.json uses unsupported implemented_cases schema; expected top-level cases list",
                validator.errors,
            )

    def test_blackbox_result_warns_when_st_dev_result_looks_like_debug_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_blackbox_fixture(
                operator_dir,
                results=[{"case_id": "BB_L0_001", "level": "L0", "status": "passed", "route": "st_cpp"}],
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_blackbox("tests/st/results/st_dev_result.json")

            self.assertTrue(
                any("appears to contain partial/debug results" in error for error in validator.errors),
                validator.errors,
            )

    def test_blackbox_design_constraints_reject_match_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            write_text(
                operator_dir / "tests/st/design/05_约束定义.yaml",
                """
constraints:
  - id: x_matches_y
    type: match
    sources: [x_shape]
    target: y_shape
""",
            )
            write_text(
                operator_dir / "tests/st/design/07_因子值.csv",
                "x_shape,y_shape\n[1, 2],[1, 3]\n",
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_blackbox_design_constraints()

            self.assertTrue(
                any("violate match constraint" in error for error in validator.errors),
                validator.errors,
            )

    def test_blackbox_execution_evidence_requires_st_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            manifest, results = self._single_blackbox_execution_fixture(operator_dir)

            validator = WorkflowValidator(operator_dir)
            validator.validate_blackbox_execution_evidence(set(manifest), manifest, results)

            self.assertTrue(
                any("blackbox execution evidence lacks ST case_id mappings" in error for error in validator.errors),
                validator.errors,
            )

    def test_blackbox_execution_evidence_requires_implementation_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            manifest, results = self._single_blackbox_execution_fixture(
                operator_dir,
                "int main() { /* BB_L1_001 */ return 0; }\n",
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_blackbox_execution_evidence(set(manifest), manifest, results)

            self.assertTrue(
                any("lacks implementation invocation marker" in error for error in validator.errors),
                validator.errors,
            )

    def test_cp2_accepts_full_positive_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_blackbox_fixture(operator_dir, results=self._blackbox_passed_results())
            self._write_ut_fixture(operator_dir)
            self._write_report(operator_dir, "tests/reports/iter3-integration-report.md")

            validator = WorkflowValidator(operator_dir)
            validator.validate_cp2()

            self.assertEqual([], validator.errors)

    def test_cp3_accepts_full_positive_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_cp3_positive_fixture(operator_dir)

            validator = WorkflowValidator(operator_dir)
            validator.validate_cp3()

            self.assertEqual([], validator.errors)

    def test_whitebox_accepts_new_skill_outputs_with_st_primary_and_pytest_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertEqual([], validator.errors)
            self.assertEqual(2, validator.whitebox_summary["enabled"])
            self.assertEqual(2, validator.whitebox_summary["passed"])
            self.assertEqual(2, validator.whitebox_summary["st_passed"])
            self.assertEqual(2, validator.whitebox_summary["pytest_passed"])

    def test_whitebox_rejects_name_only_generated_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)
            fake_case = {"case_id": "ADDRMS_WB_001", "name": "empty_tensor", "enabled": True}
            write_json(operator_dir / "tests/whitebox/S5_mapped_cases_low.json", {"cases": [fake_case]})
            write_json(operator_dir / "tests/whitebox/S5_mapped_cases_high.json", {"cases": [fake_case]})
            write_json(
                operator_dir / "tests/whitebox/results/st_result.json",
                {"results": [{"case_id": "ADDRMS_WB_001", "status": "passed", "route": "whitebox_st"}]},
            )
            write_json(
                operator_dir / "tests/whitebox/results/pytest_collect.json",
                {"collected_case_ids": ["ADDRMS_WB_001"]},
            )
            write_json(
                operator_dir / "tests/whitebox/results/pytest_result.json",
                {"results": [{"case_id": "ADDRMS_WB_001", "status": "passed"}]},
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("must contain Step 5 mapped params/tensors" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_rejects_mapped_cases_without_data_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, include_data_range=False)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("must include _data_range" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_accepts_absent_optional_input_without_data_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, include_optional_absent_input=True)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertEqual([], validator.errors)

    def test_whitebox_rejects_high_cases_without_data_range_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, high_expanded=False)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("must contain data_range-expanded cases beyond low" in error for error in validator.errors),
                validator.errors,
            )
            self.assertTrue(
                any("must include data_range-expanded cases with non-normal" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_rejects_whitebox_st_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, st_route="whitebox_st")

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("uses unsupported routes" in error and "whitebox_st" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_requires_real_st_execution_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, include_st_execution_evidence=False)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("whitebox ST execution evidence lacks" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_rejects_id_only_st_marker_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, st_evidence_mode="id_marker_only")

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any(
                    "whitebox ST execution evidence lacks executable case mappings" in error
                    for error in validator.errors
                ),
                validator.errors,
            )

    def test_whitebox_rejects_helper_script_that_writes_passed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)
            write_text(
                operator_dir / "scripts/generate_evidence.py",
                """
import json
from pathlib import Path

def write_json(path, data):
    Path(path).write_text(json.dumps(data))

write_json("tests/whitebox/results/st_result.json", {"results": [{"case_id": "case00001", "status": "passed"}]})
write_json("tests/reports/evidence_index.json", {"status": "passed"})
""",
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any(
                    "helper script appears to synthesize workflow pass evidence" in error
                    for error in validator.errors
                ),
                validator.errors,
            )

    def test_whitebox_rejects_simplified_json_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, include_full_skill_workflow=False)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("WORKFLOW_PROVENANCE.json" in error for error in validator.errors),
                validator.errors,
            )
            self.assertFalse(
                any("S2P0_file_manifest.json" in error for error in validator.errors),
                validator.errors,
            )
            self.assertFalse(
                any("S5_case_mapper.py" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_does_not_require_internal_skill_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, include_internal_whitebox_artifacts=False)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertEqual([], validator.errors)

    def test_whitebox_accepts_repo_relative_skill_workflow_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)
            provenance_path = operator_dir / "tests/whitebox/WORKFLOW_PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["workflow"] = "ops/ascendc-whitebox-design/SKILL.md"
            write_json(provenance_path, provenance)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertEqual([], validator.errors)

    def test_whitebox_rejects_external_workflow_provenance_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)
            provenance_path = operator_dir / "tests/whitebox/WORKFLOW_PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["workflow"] = (
                "/mnt/workspace/code/ai-infra/runs/old-run/repo/"
                "ops/ascendc-whitebox-design/references/workflow.md"
            )
            write_json(provenance_path, provenance)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("workflow must be repository-relative" in error for error in validator.errors),
                validator.errors,
            )

    def test_whitebox_requires_st_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, include_st_result=False)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any(
                    "missing required file: tests/whitebox/results/st_result.json" in error
                    for error in validator.errors
                ),
                validator.errors,
            )

    def test_whitebox_enabled_ids_come_from_high_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)

            validator = WorkflowValidator(operator_dir)

            self.assertEqual({"case00001", "case00001_x_zero"}, validator.enabled_whitebox_ids())

    def test_whitebox_gate_uses_high_cases_when_high_contains_data_range_expansions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertEqual([], validator.errors)
            self.assertEqual(2, validator.whitebox_summary["enabled"])
            self.assertEqual(2, validator.whitebox_summary["st_passed"])
            self.assertEqual(2, validator.whitebox_summary["pytest_passed"])

    def test_whitebox_rejects_low_only_evidence_when_high_contains_data_range_expansions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, evidence_case_set="low")

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any(
                    "pytest_collect.json case ids do not match enabled whitebox cases" in error
                    for error in validator.errors
                ),
                validator.errors,
            )
            self.assertTrue(
                any("st_result.json missing enabled case ids" in error for error in validator.errors),
                validator.errors,
            )
            self.assertTrue(
                any(
                    "whitebox ST execution evidence lacks executable case mappings" in error
                    for error in validator.errors
                ),
                validator.errors,
            )

    def test_whitebox_param_def_accepts_per_dtype_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir)
            write_json(
                operator_dir / "tests/whitebox/S2P2_param_def.json",
                {
                    "tiling_keys": [101],
                    "groups": [
                        {
                            "id": "group0",
                            "per_dtype": {
                                "float16": {"path": "tiling.cpp:42", "key": 101},
                                "float32": {"path": "tiling.cpp:43", "key": 101},
                            },
                        }
                    ],
                },
            )

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertEqual([], validator.errors)

    def test_whitebox_requires_every_st_result_row_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, st_status="failed")

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox()

            self.assertTrue(
                any("st_result.json failed enabled case ids" in error for error in validator.errors),
                validator.errors,
            )
            self.assertEqual(0, validator.whitebox_summary["st_passed"])
            self.assertEqual(2, validator.whitebox_summary["st_failed"])

    def test_whitebox_coverage_profile_can_lower_expected_gate_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, high_expanded=False)
            provenance_path = operator_dir / "tests/whitebox/WORKFLOW_PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance.pop("case_reduction_reason", None)
            provenance["coverage_profile"] = {
                "name": "simple_operator",
                "minimum_gate_cases": 1,
                "reason": "One tiling key and two reachable interface modes.",
            }
            write_json(provenance_path, provenance)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox_provenance(2)

            self.assertEqual([], validator.errors)

    def test_whitebox_coverage_profile_rejects_invalid_gate_case_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            operator_dir = Path(tmpdir)
            self._write_new_whitebox_outputs(operator_dir, high_expanded=False)
            provenance_path = operator_dir / "tests/whitebox/WORKFLOW_PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["coverage_profile"] = {
                "name": "simple_operator",
                "minimum_gate_cases": 0,
                "reason": "One tiling key and two reachable interface modes.",
            }
            write_json(provenance_path, provenance)

            validator = WorkflowValidator(operator_dir)
            validator.validate_whitebox_provenance(1)

            self.assertTrue(
                any("coverage_profile.minimum_gate_cases" in error for error in validator.errors),
                validator.errors,
            )

    def _write_new_whitebox_outputs(
        self,
        operator_dir: Path,
        *,
        include_st_result: bool = True,
        st_status: str = "passed",
        st_route: str = "st_cpp",
        include_full_skill_workflow: bool = True,
        include_auxiliary_whitebox_artifacts: bool = True,
        include_internal_whitebox_artifacts: bool = True,
        include_data_range: bool = True,
        include_optional_absent_input: bool = False,
        high_expanded: bool = True,
        include_st_execution_evidence: bool = True,
        st_evidence_mode: str = "executable",
        evidence_case_set: str = "high",
    ) -> None:
        whitebox_dir = operator_dir / "tests/whitebox"
        results_dir = whitebox_dir / "results"
        case, high_cases, low_case_ids, high_case_ids = self._whitebox_case_fixtures(
            include_data_range=include_data_range,
            include_optional_absent_input=include_optional_absent_input,
            high_expanded=high_expanded,
        )
        evidence_case_ids = low_case_ids if evidence_case_set == "low" else high_case_ids

        self._write_whitebox_workflow_files(
            whitebox_dir,
            case,
            low_case_ids,
            high_case_ids,
            high_cases,
            include_full_skill_workflow=include_full_skill_workflow,
            include_auxiliary_whitebox_artifacts=include_auxiliary_whitebox_artifacts,
            include_internal_whitebox_artifacts=include_internal_whitebox_artifacts,
        )
        self._write_whitebox_contract_files(
            whitebox_dir,
            case,
            high_cases,
            include_internal_whitebox_artifacts,
        )
        write_text(whitebox_dir / "S6_test_demo.py", "def test_placeholder():\n    pass\n")
        self._write_whitebox_result_files(results_dir, include_st_result, st_status, st_route, evidence_case_ids)
        if include_st_execution_evidence:
            self._write_whitebox_execution_evidence(
                operator_dir, st_evidence_mode, evidence_case_set, evidence_case_ids
            )

    def _write_whitebox_workflow_files(
        self,
        whitebox_dir: Path,
        case: dict[str, object],
        low_case_ids: list[str],
        high_case_ids: list[str],
        high_cases: list[dict[str, object]],
        *,
        include_full_skill_workflow: bool,
        include_auxiliary_whitebox_artifacts: bool,
        include_internal_whitebox_artifacts: bool,
    ) -> None:
        if not include_full_skill_workflow:
            return
        if include_internal_whitebox_artifacts:
            self._write_internal_whitebox_artifacts(
                whitebox_dir,
                case,
                include_auxiliary_whitebox_artifacts,
            )
        write_json(
            whitebox_dir / "WORKFLOW_PROVENANCE.json",
            {
                "skill": "ascendc-whitebox-design",
                "workflow": "ops/ascendc-whitebox-design/references/workflow.md",
                "coverage_profile": "skill-default",
                "steps": ["S2P0", "S2P1", "S2P2", "S3", "S5", "S6"],
                "step4": {
                    "decision": "skipped",
                    "reason": "Unit test fixture uses automatic checkpoint continuation.",
                },
                "case_reduction_reason": {
                    "reason": "Unit test fixture intentionally contains one whitebox case.",
                    "path_case_count": len(low_case_ids),
                    "network_case_count": len(low_case_ids),
                    "enabled_case_count": len(high_case_ids),
                    "low_case_count": 1,
                    "high_case_count": len(high_cases),
                },
            },
        )

    def _write_cp3_positive_fixture(self, operator_dir: Path) -> None:
        self._write_blackbox_fixture(
            operator_dir,
            results=self._blackbox_passed_results(),
            result_name="st_real_result.json",
        )
        self._write_ut_fixture(operator_dir)
        self._write_new_whitebox_outputs(operator_dir)
        self._write_report(operator_dir, "tests/reports/iter3-acceptance-report.md")
        self._write_report(operator_dir, "tests/reports/test-branches-merge-exec-report.md")
        self._write_evidence_index_fixture(operator_dir)

    def _write_blackbox_fixture(
        self,
        operator_dir: Path,
        *,
        results: list[dict[str, object]],
        result_name: str = "st_dev_result.json",
    ) -> None:
        self._write_blackbox_design_files(operator_dir)
        write_json(operator_dir / "tests/st/case_manifest.json", {"cases": self._blackbox_manifest_cases()})
        write_json(operator_dir / f"tests/st/results/{result_name}", {"results": results})
        write_text(
            operator_dir / "tests/st/test_aclnn_demo.cpp",
            "int main() {\n"
            "  // IMPLEMENTATION_UNDER_TEST aclnnDemoGetWorkspaceSize op_host op_kernel "
            "BB_L0_001 BB_L0_002 BB_L1_001 BB_L2_001\n"
            "  return RunCase(\"BB_L1_001\") ? 0 : 1;\n"
            "}\n",
        )

if __name__ == "__main__":
    unittest.main()
