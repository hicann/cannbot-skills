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
"""Validate ops-registry-invoke CP2/CP3 workflow evidence.

The validator treats Markdown reports as presentation material only. The gate is
driven by non-empty CSV files, JSON result evidence, whitebox ST result evidence,
pytest collect/run evidence, and evidence index checks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple


PASS_STATUS = "✅通过"
PASSED_VALUES = {"passed", "pass", "success", "ok", "通过", "✅通过"}
_UNSET = object()
BLACKBOX_LEVELS = ("L0", "L1", "L2")
ASCENDC_ST_DESIGN_GENERATOR = "ops/ascendc-st-design/scripts/generate_test_cases.py"
DEFAULT_WHITEBOX_GATE_CASES = 25
WHITEBOX_SKILL_PATH_PREFIX = "ops/ascendc-whitebox-design/"
WHITEBOX_NON_NORMAL_DATA_RANGES = {
    "zero",
    "extreme",
    "negative",
    "tiny_pos",
    "all_ones",
    "near_zero",
    "with_inf",
    "with_nan",
}
WHITEBOX_DATA_RANGES = {"normal"} | WHITEBOX_NON_NORMAL_DATA_RANGES
WHITEBOX_CONTRACT_FILES = (
    "tests/whitebox/WORKFLOW_PROVENANCE.json",
    "tests/whitebox/S2P2_param_def.json",
    "tests/whitebox/S5_mapped_cases_low.json",
    "tests/whitebox/S5_mapped_cases_high.json",
    "tests/whitebox/S6_tilingkey_coverage.json",
)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".py"}
EXECUTION_EVIDENCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
    ".sh",
    ".log",
    ".txt",
}
COMMON_OUTPUT_NAMES = {"y", "out", "output"}
ST_IMPLEMENTATION_EVIDENCE_PATTERNS = (
    re.compile(r"\baclnn[A-Za-z0-9_]*GetWorkspaceSize\b"),
    re.compile(r"\baclnn[A-Za-z0-9_]*\s*\("),
    re.compile(r"\baclrt[A-Za-z0-9_]*\b"),
    re.compile(r"\bcustom_opp\b"),
    re.compile(r"\blibcust[A-Za-z0-9_]*\.so\b"),
    re.compile(r"\bop_host\b"),
    re.compile(r"\bop_kernel\b"),
    re.compile(r"\bIMPLEMENTATION_UNDER_TEST\b"),
)
ST_EXECUTABLE_EVIDENCE_PATTERNS = (
    re.compile(r"\bint\s+main\s*\("),
    re.compile(r"\bRunCase\s*\("),
    re.compile(r"\bTEST(?:_F)?\s*\("),
    re.compile(r"\bdef\s+test_[A-Za-z0-9_]*\s*\("),
)
FORBIDDEN_HELPER_EVIDENCE_OUTPUTS = (
    "tests/st/results/st_dev_result.json",
    "tests/st/results/st_real_result.json",
    "tests/ut/test-report.json",
    "tests/whitebox/results/st_result.json",
    "tests/whitebox/results/pytest_result.json",
    "tests/reports/evidence_index.json",
)

LOGGER = logging.getLogger("validate_workflow_state")


class ResultSummaryCounts(NamedTuple):
    passed: int
    failed: int


class BlackboxResultStatus(NamedTuple):
    passed_ids: set[str]
    failed_ids: set[str]
    smoke_only_ids: set[str]
    route_errors: list[str]
    passed_level_counts: dict[str, int]


def configure_output_logging() -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


class WorkflowValidator:
    def __init__(self, operator_dir: Path) -> None:
        self.operator_dir = operator_dir
        self.errors: list[str] = []
        self.blackbox_summary: dict[str, int] = {}
        self.whitebox_summary: dict[str, int] = {}
        self.whitebox_evidence_paths: list[str] = []
        self._blackbox_l1_default_target: int | None | object = _UNSET

    @staticmethod
    def infer_level_from_path(path: Path) -> str:
        match = re.search(r"(?:^|[_/-])(L[0-2])(?:[_./-]|$)", path.as_posix(), re.I)
        return match.group(1).upper() if match else ""

    @staticmethod
    def csv_level_row_counts(csv_rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
        counts = {level: 0 for level in BLACKBOX_LEVELS}
        unknown: list[str] = []
        for row in csv_rows:
            level = str(row.get("_level") or "").upper()
            if level in counts:
                counts[level] += 1
            else:
                unknown.append(f"{row.get('_rel_path', '<csv>')}:{row.get('_case_id', '<missing>')}")
        return counts, unknown

    @staticmethod
    def canonical_factor_value(value: str) -> Any:
        raw = str(value).strip()
        if raw == "":
            return raw
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return raw

    @staticmethod
    def values_match_constraint(
        source_value: Any,
        target_value: Any,
        source_index: Any,
        target_index: Any,
    ) -> bool:
        if source_index is None and target_index is None:
            return source_value == target_value
        if not isinstance(source_index, int) or not isinstance(target_index, int):
            return False
        if not isinstance(source_value, list) or not isinstance(target_value, list):
            return False
        if source_index >= len(source_value) or target_index >= len(target_value):
            return False
        return source_value[source_index] == target_value[target_index]

    @staticmethod
    def match_constraint_targets(constraint: dict[str, Any]) -> list[str]:
        targets: list[str] = []
        if constraint.get("target"):
            targets.append(str(constraint["target"]))
        targets.extend(str(target) for target in constraint.get("targets", []))
        return targets

    @staticmethod
    def has_default_st_route(route: str, accepted_routes: Any) -> bool:
        return (
            not route
            and isinstance(accepted_routes, list)
            and any(str(item).startswith("st_") for item in accepted_routes)
        )

    @staticmethod
    def has_accepted_waiver_fields(
        approved: bool,
        approved_by: str,
        reason: str,
        scope: str,
        unresolved: re.Pattern[str],
    ) -> bool:
        if not approved:
            return False
        if approved_by not in {"user", "human", "用户"}:
            return False
        if not reason or unresolved.search(reason):
            return False
        return scope in {"blackbox_l1_minimum", "development_only_low_coverage"}

    @staticmethod
    def uses_unaccepted_route(route: Any, accepted_routes: Any) -> bool:
        if not route:
            return False
        if not isinstance(accepted_routes, list):
            return False
        return bool(accepted_routes) and route not in accepted_routes

    @staticmethod
    def is_passed_entry(entry: dict[str, Any]) -> bool:
        raw = entry.get("status", entry.get("result", ""))
        return WorkflowValidator.is_passed_status(raw)

    @staticmethod
    def is_passed_status(raw: Any) -> bool:
        return str(raw).strip().lower() in PASSED_VALUES or str(raw).strip() in PASSED_VALUES

    @staticmethod
    def whitebox_input_data_ranges(item: dict[str, Any]) -> list[str]:
        tensors = item.get("tensors")
        inputs = tensors.get("inputs") if isinstance(tensors, dict) else None
        if not isinstance(inputs, dict):
            return []
        ranges = []
        for spec in inputs.values():
            if spec is None:
                continue
            if isinstance(spec, dict):
                value = str(spec.get("_data_range") or "").strip()
                if value:
                    ranges.append(value)
        return ranges

    @staticmethod
    def record_evidence_index_path(paths: dict[str, dict[str, Any]], item: Any) -> None:
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            if path:
                paths[path] = item
            return
        if isinstance(item, str):
            paths[item] = {"path": item}


    def rel(self, rel_path: str) -> Path:
        return self.operator_dir / rel_path

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def require_file(self, rel_path: str) -> Path | None:
        path = self.rel(rel_path)
        if not path.is_file():
            self.add_error(f"missing required file: {rel_path}")
            return None
        return path

    def require_any_file(self, rel_paths: tuple[str, ...], label: str) -> Path | None:
        for rel_path in rel_paths:
            path = self.rel(rel_path)
            if path.is_file():
                return path
        self.add_error(f"missing required file: {label} (checked: {', '.join(rel_paths)})")
        return None

    def require_glob_file(self, pattern: str, label: str) -> Path | None:
        matches = sorted(path for path in self.operator_dir.glob(pattern) if path.is_file())
        if not matches:
            self.add_error(f"missing {label}: {pattern}")
            return None
        return matches[0]

    def load_json(self, rel_path: str) -> Any | None:
        path = self.require_file(rel_path)
        if path is None:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.add_error(f"invalid JSON in {rel_path}: {exc}")
            return None

    def test_design_file(self) -> Path | None:
        return self.require_any_file(("docs/TEST.md", "TEST.md"), "TEST.md")

    def repo_root_with_ascendc_st_design(self) -> Path | None:
        for root in [self.operator_dir, *self.operator_dir.parents]:
            if (root / ASCENDC_ST_DESIGN_GENERATOR).is_file():
                return root
        return None

    def blackbox_l1_default_target(self) -> int | None:
        if self._blackbox_l1_default_target is not _UNSET:
            return self._blackbox_l1_default_target if isinstance(self._blackbox_l1_default_target, int) else None

        repo_root = self.repo_root_with_ascendc_st_design()
        if repo_root is None:
            self.add_error(
                "cannot resolve ascendc-st-design --target-count default: "
                f"missing {ASCENDC_ST_DESIGN_GENERATOR}"
            )
            self._blackbox_l1_default_target = None
            return None

        generator = repo_root / ASCENDC_ST_DESIGN_GENERATOR
        try:
            tree = ast.parse(generator.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            self.add_error(f"cannot resolve ascendc-st-design --target-count default: {exc}")
            self._blackbox_l1_default_target = None
            return None

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            has_target_count = any(
                isinstance(arg, ast.Constant) and arg.value == "--target-count"
                for arg in node.args
            )
            if not has_target_count:
                continue
            for keyword in node.keywords:
                if keyword.arg != "default" or not isinstance(keyword.value, ast.Constant):
                    continue
                value = keyword.value.value
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    self._blackbox_l1_default_target = value
                    return value

        self.add_error("cannot resolve ascendc-st-design --target-count default: argument default not found")
        self._blackbox_l1_default_target = None
        return None

    def validate_report_status(self, rel_path: str) -> None:
        path = self.require_file(rel_path)
        if path is None:
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\*\*状态\*\*:\s*([^\n\r]+)", text)
        if not match:
            self.add_error(f"report lacks parsable status field: {rel_path}")
            return
        status = match.group(1).strip()
        if status != PASS_STATUS:
            self.add_error(f"report status is not {PASS_STATUS}: {rel_path} -> {status}")

    def csv_case_rows(self) -> list[dict[str, Any]]:
        base = self.rel("tests/st/testcases")
        if not base.is_dir():
            self.add_error("missing required directory: tests/st/testcases")
            return []

        rows: list[dict[str, Any]] = []
        csv_files = sorted(base.rglob("*.csv"))
        if not csv_files:
            self.add_error("no blackbox CSV files found under tests/st/testcases")
            return rows

        for path in csv_files:
            rows.extend(self.csv_rows_from_file(path))
        return rows

    def csv_rows_from_file(self, path: Path) -> list[dict[str, Any]]:
        rel_path = path.relative_to(self.operator_dir).as_posix()
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return self.csv_rows_from_reader(reader, rel_path, self.infer_level_from_path(path))
        except csv.Error as exc:
            self.add_error(f"invalid CSV in {rel_path}: {exc}")
            return []

    def csv_rows_from_reader(
        self,
        reader: csv.DictReader[str],
        rel_path: str,
        inferred_level: str,
    ) -> list[dict[str, Any]]:
        if not reader.fieldnames:
            self.add_error(f"CSV lacks header: {rel_path}")
            return []

        rows: list[dict[str, Any]] = []
        data_rows = 0
        for row in reader:
            data_rows += 1
            case_id = str(row.get("case_id") or row.get("testcase_name") or "").strip()
            if not case_id:
                continue
            normalized = dict(row)
            normalized["_case_id"] = case_id
            normalized["_level"] = str(row.get("level") or inferred_level or "").upper()
            normalized["_rel_path"] = rel_path
            rows.append(normalized)
        if data_rows == 0:
            self.add_error(f"CSV has no data rows: {rel_path}")
        return rows

    def csv_case_ids(self) -> set[str]:
        return {row["_case_id"] for row in self.csv_case_rows()}

    def record_blackbox_target(self, targets: dict[str, int], level: str, target_count: int) -> None:
        targets[level] = target_count
        if target_count <= 0:
            self.add_error(f"TEST.md {level} target_count must be positive")
        if level != "L1" or self.has_low_coverage_waiver():
            return

        default_target = self.blackbox_l1_default_target()
        if default_target is None or target_count >= default_target:
            return
        self.add_error(
            "TEST.md L1 target_count is below ascendc-st-design default target: "
            f"actual={target_count}, minimum={default_target}"
        )

    def parse_blackbox_case_target_lines(self, lines: list[str]) -> dict[str, int]:
        targets: dict[str, int] = {}
        current_level = ""
        for line in lines:
            direct_match = re.match(r"\s*-?\s*(L[0-2])\s*[:：]\s*(\d+)\b", line, re.I)
            if direct_match:
                level = direct_match.group(1).upper()
                self.record_blackbox_target(targets, level, int(direct_match.group(2)))
                current_level = ""
            else:
                nested_level = re.match(r"\s*-?\s*(L[0-2])\s*[:：]\s*$", line, re.I)
                nested_count = re.match(r"\s*(?:target_count|count)\s*[:：]\s*(\d+)\b", line, re.I)
                if nested_level:
                    current_level = nested_level.group(1).upper()
                elif current_level and nested_count:
                    self.record_blackbox_target(targets, current_level, int(nested_count.group(1)))
                elif targets and line.strip() and not line.startswith((" ", "\t", "-")):
                    break
            if all(level_name in targets for level_name in BLACKBOX_LEVELS):
                break
        return targets

    def blackbox_case_target_counts(self, test_design_path: Path) -> dict[str, int]:
        text = test_design_path.read_text(encoding="utf-8", errors="replace")
        marker = re.search(r"(?im)^\s*blackbox_case_targets\s*[:：]\s*$", text)
        if not marker:
            self.add_error("TEST.md missing blackbox_case_targets L0/L1/L2 target counts")
            return {}

        targets = self.parse_blackbox_case_target_lines(text[marker.end():].splitlines())
        for level in BLACKBOX_LEVELS:
            if level not in targets:
                self.add_error(f"TEST.md missing {level} target_count under blackbox_case_targets")
        return targets

    def has_low_coverage_waiver(self) -> bool:
        waiver_paths = [
            "tests/st/LOW_COVERAGE_WAIVER.json",
            "docs/LOW_COVERAGE_WAIVER.json",
        ]
        unresolved = re.compile(r"\b(TBD|TODO)\b|待确认|未确认|未批准")
        for rel_path in waiver_paths:
            path = self.rel(rel_path)
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.add_error(f"invalid JSON in {rel_path}: {exc}")
                continue
            if not isinstance(data, dict):
                self.add_error(f"{rel_path} must be a JSON object")
                continue

            approved = data.get("approved") is True
            approved_by = str(data.get("approved_by") or "").strip().lower()
            reason = str(data.get("reason") or "").strip()
            scope = str(data.get("scope") or "").strip()
            if self.has_accepted_waiver_fields(approved, approved_by, reason, scope, unresolved):
                return True
        return False

    def manifest_level_counts(
        self,
        manifest: dict[str, dict[str, Any]],
        csv_rows: list[dict[str, Any]],
    ) -> tuple[dict[str, int], dict[str, str]]:
        counts = {level: 0 for level in BLACKBOX_LEVELS}
        csv_level_by_id = {str(row["_case_id"]): str(row.get("_level") or "").upper() for row in csv_rows}
        level_by_id: dict[str, str] = {}

        for case_id, item in manifest.items():
            level = str(
                item.get("level")
                or item.get("case_level")
                or item.get("test_level")
                or csv_level_by_id.get(case_id, "")
            ).upper()
            if level not in counts:
                self.add_error(f"case_manifest.json case {case_id} lacks L0/L1/L2 level")
                continue
            counts[level] += 1
            level_by_id[case_id] = level
        return counts, level_by_id

    def validate_blackbox_target_consistency(
        self,
        result_rel_path: str,
        *,
        targets: dict[str, int],
        csv_counts: dict[str, int],
        manifest_counts: dict[str, int],
        passed_level_counts: dict[str, int],
    ) -> None:
        if not targets:
            return
        for level in BLACKBOX_LEVELS:
            target_count = targets.get(level)
            if target_count is None:
                continue
            csv_count = csv_counts.get(level, 0)
            manifest_count = manifest_counts.get(level, 0)
            passed_count = passed_level_counts.get(level, 0)
            if csv_count != target_count:
                self.add_error(
                    f"{level} blackbox CSV rows do not equal TEST.md target_count: "
                    f"actual={csv_count}, target={target_count}"
                )
            if manifest_count != target_count:
                self.add_error(
                    f"{level} case_manifest cases do not equal TEST.md target_count: "
                    f"actual={manifest_count}, target={target_count}"
                )
            if passed_count != target_count:
                self.add_error(
                    f"{result_rel_path} {level} passed results do not equal TEST.md target_count: "
                    f"actual={passed_count}, target={target_count}"
                )

    def load_yaml_file(self, rel_path: str) -> Any | None:
        try:
            import yaml
        except ModuleNotFoundError:
            self.add_error(
                "missing Python dependency: PyYAML. "
                "Install it with: python3 -m pip install PyYAML"
            )
            return None

        path = self.require_file(rel_path)
        if path is None:
            return None
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            self.add_error(f"invalid YAML in {rel_path}: {exc}")
            return None

    def load_blackbox_design_constraint_inputs(
        self,
        values_path: Path,
    ) -> tuple[list[Any], list[dict[str, str]], set[str]] | None:
        constraints_data = self.load_yaml_file("tests/st/design/05_约束定义.yaml")
        if not isinstance(constraints_data, dict):
            self.add_error("tests/st/design/05_约束定义.yaml must be a mapping")
            return None

        constraints = constraints_data.get("constraints")
        if not isinstance(constraints, list):
            self.add_error("tests/st/design/05_约束定义.yaml must contain a constraints list")
            return None

        try:
            with values_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = set(reader.fieldnames or [])
        except csv.Error as exc:
            self.add_error(f"invalid CSV in tests/st/design/07_因子值.csv: {exc}")
            return None

        if not rows:
            self.add_error("tests/st/design/07_因子值.csv has no factor rows")
            return None
        return constraints, rows, fieldnames

    def validate_match_constraint_rows(
        self,
        constraint: dict[str, Any],
        rows: list[dict[str, str]],
        fieldnames: set[str],
    ) -> None:
        sources = constraint.get("sources", [])
        if not sources:
            return

        source = sources[0]
        targets = self.match_constraint_targets(constraint)
        if not targets:
            return

        constraint_id = str(constraint.get("id") or "unnamed-match")
        required_columns = [source, *targets]
        missing_columns = [name for name in required_columns if name not in fieldnames]
        if missing_columns:
            self.add_error(
                "blackbox design factor values missing match constraint columns: "
                f"{constraint_id} -> {missing_columns}"
            )
            return

        source_index = constraint.get("source_index")
        target_index = constraint.get("target_index")
        for row_index, row in enumerate(rows, start=1):
            source_value = self.canonical_factor_value(row[source])
            for target in targets:
                target_value = self.canonical_factor_value(row[target])
                if not self.values_match_constraint(source_value, target_value, source_index, target_index):
                    self.add_error(
                        "blackbox design factor values violate match constraint: "
                        f"{constraint_id} row={row_index} {source}={row[source]} "
                        f"{target}={row[target]}"
                    )
                    return

    def validate_blackbox_design_constraints(self) -> None:
        constraints_path = self.rel("tests/st/design/05_约束定义.yaml")
        values_path = self.rel("tests/st/design/07_因子值.csv")
        if not constraints_path.exists() and not values_path.exists():
            return
        if not constraints_path.is_file() or not values_path.is_file():
            if not constraints_path.is_file():
                self.add_error("missing required file: tests/st/design/05_约束定义.yaml")
            if not values_path.is_file():
                self.add_error("missing required file: tests/st/design/07_因子值.csv")
            return

        inputs = self.load_blackbox_design_constraint_inputs(values_path)
        if inputs is None:
            return
        constraints, rows, fieldnames = inputs

        for constraint in constraints:
            if not isinstance(constraint, dict) or constraint.get("type") != "match":
                continue
            self.validate_match_constraint_rows(constraint, rows, fieldnames)

    def execution_evidence_entries(
        self,
        rel_base: str,
        *,
        excluded_parts: set[str],
        excluded_names: set[str],
    ) -> list[tuple[str, str]]:
        base = self.rel(rel_base)
        if not base.is_dir():
            return []

        entries: list[tuple[str, str]] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(self.operator_dir).as_posix()
            if any(part in rel_path for part in excluded_parts):
                continue
            if path.name in excluded_names or path.name.startswith("generate_"):
                continue
            if path.suffix not in EXECUTION_EVIDENCE_SUFFIXES and path.name != "CMakeLists.txt":
                continue
            entries.append((rel_path, path.read_text(encoding="utf-8", errors="replace")))
        return entries

    def blackbox_execution_evidence_text(self, rel_base: str) -> tuple[str, list[str]]:
        entries = self.execution_evidence_entries(
            rel_base,
            excluded_parts={
                "tests/st/testcases/",
                "tests/st/results/",
                "tests/st/design/",
            },
            excluded_names={"case_manifest.json"},
        )
        chunks = [text for _, text in entries]
        rel_paths = [rel_path for rel_path, _ in entries]
        return "\n".join(chunks), rel_paths

    def validate_blackbox_execution_evidence(
        self,
        required_ids: set[str],
        manifest: dict[str, dict[str, Any]],
        results: dict[str, dict[str, Any]],
    ) -> None:
        if not required_ids or not results:
            return

        st_text, st_paths = self.blackbox_execution_evidence_text("tests/st")
        ut_text, ut_paths = self.blackbox_execution_evidence_text("tests/ut")
        missing_st, missing_ut, missing_unknown, requires_st_implementation = (
            self.blackbox_execution_evidence_gaps(required_ids, manifest, results, st_text, ut_text)
        )

        self.add_blackbox_execution_evidence_errors(missing_st, missing_ut, missing_unknown, st_paths, ut_paths)
        has_st_marker = any(pattern.search(st_text) for pattern in ST_IMPLEMENTATION_EVIDENCE_PATTERNS)
        if requires_st_implementation and not has_st_marker:
            self.add_error(
                "blackbox ST execution evidence lacks implementation invocation marker; "
                "non-smoke ST routes must show ACLNN/custom_opp/libcust/op_host/op_kernel "
                f"or IMPLEMENTATION_UNDER_TEST evidence; checked files={st_paths}"
            )

    def blackbox_execution_evidence_gaps(
        self,
        required_ids: set[str],
        manifest: dict[str, dict[str, Any]],
        results: dict[str, dict[str, Any]],
        st_text: str,
        ut_text: str,
    ) -> tuple[list[str], list[str], list[str], bool]:
        combined_text = f"{st_text}\n{ut_text}"
        missing_st: list[str] = []
        missing_ut: list[str] = []
        missing_unknown: list[str] = []
        requires_st_implementation = False

        for case_id in sorted(required_ids):
            entry = results.get(case_id)
            if not entry or entry.get("smoke_only") is True:
                continue
            route = str(entry.get("route") or "").strip()
            accepted_routes = manifest[case_id].get("accepted_routes", [])
            has_default_st_route = self.has_default_st_route(route, accepted_routes)
            if route in {"st_cpp", "st_real"} or has_default_st_route:
                requires_st_implementation = True
                if case_id not in st_text:
                    missing_st.append(case_id)
            elif route == "ut_host":
                if case_id not in ut_text:
                    missing_ut.append(case_id)
            elif case_id not in combined_text:
                missing_unknown.append(case_id)
        return missing_st, missing_ut, missing_unknown, requires_st_implementation

    def add_blackbox_execution_evidence_errors(
        self,
        missing_st: list[str],
        missing_ut: list[str],
        missing_unknown: list[str],
        st_paths: list[str],
        ut_paths: list[str],
    ) -> None:
        if missing_st:
            self.add_error(
                "blackbox execution evidence lacks ST case_id mappings: "
                f"{missing_st}; checked files={st_paths}"
            )
        if missing_ut:
            self.add_error(
                "blackbox execution evidence lacks UT/Host case_id mappings: "
                f"{missing_ut}; checked files={ut_paths}"
            )
        if missing_unknown:
            self.add_error(
                "blackbox execution evidence lacks case_id mappings for unknown routes: "
                f"{missing_unknown}; checked files={sorted(set(st_paths + ut_paths))}"
            )

    def manifest_cases(self) -> dict[str, dict[str, Any]]:
        data = self.load_json("tests/st/case_manifest.json")
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list):
            if isinstance(data, dict) and "implemented_cases" in data:
                self.add_error(
                    "tests/st/case_manifest.json uses unsupported implemented_cases schema; "
                    "expected top-level cases list"
                )
            self.add_error("tests/st/case_manifest.json must contain a cases list")
            return {}

        result: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(cases):
            if not isinstance(item, dict):
                self.add_error(f"manifest case at index {index} is not an object")
                continue
            case_id = str(item.get("case_id") or item.get("id") or "").strip()
            if not case_id:
                self.add_error(f"manifest case at index {index} lacks case_id")
                continue
            result[case_id] = item
        if not result:
            self.add_error("case_manifest.json has no usable case ids")
        return result

    def result_entries(self, rel_path: str) -> dict[str, dict[str, Any]]:
        data = self.load_json(rel_path)
        entries: Any
        if isinstance(data, dict):
            entries = data.get("results", data.get("cases", []))
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        if not isinstance(entries, list):
            self.add_error(f"{rel_path} must contain a results list")
            return {}

        result: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(entries):
            if not isinstance(item, dict):
                self.add_error(f"{rel_path} result at index {index} is not an object")
                continue
            case_id = str(item.get("case_id") or item.get("id") or "").strip()
            if not case_id:
                self.add_error(f"{rel_path} result at index {index} lacks case_id")
                continue
            result[case_id] = item
        return result

    def numeric_values_for_keys(self, value: Any, keys: set[str]) -> list[float]:
        values: list[float] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key in keys and isinstance(item, (int, float)) and not isinstance(item, bool):
                    values.append(float(item))
                values.extend(self.numeric_values_for_keys(item, keys))
        elif isinstance(value, list):
            for item in value:
                values.extend(self.numeric_values_for_keys(item, keys))
        return values

    def first_numeric_for_keys(self, value: Any, keys: set[str]) -> float | None:
        values = self.numeric_values_for_keys(value, keys)
        return values[0] if values else None

    def count_result_summary_entries(
        self,
        rel_path: str,
        label: str,
        entries: list[Any],
    ) -> ResultSummaryCounts:
        passed_entries = 0
        failed_entries = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                self.add_error(f"{rel_path} {label} result at index {index} is not an object")
                continue
            case_id = str(entry.get("case_id") or entry.get("id") or "").strip()
            if not case_id:
                self.add_error(f"{rel_path} {label} result at index {index} lacks case_id")
                continue
            if self.is_passed_entry(entry):
                passed_entries += 1
            else:
                failed_entries += 1
        return ResultSummaryCounts(passed_entries, failed_entries)

    def validate_result_summary_counts(
        self,
        rel_path: str,
        label: str,
        summary: dict[str, Any],
        entries: list[Any],
        counts: ResultSummaryCounts,
    ) -> None:
        total = summary.get("total")
        passed = summary.get("passed")
        failed = summary.get("failed")
        if not isinstance(total, int) or total <= 0:
            self.add_error(f"{rel_path} summary.total must be a positive integer")
        if failed != 0:
            self.add_error(f"{rel_path} summary.failed expected 0, got {failed!r}")
        if isinstance(total, int) and passed != total:
            self.add_error(f"{rel_path} summary.passed expected {total}, got {passed!r}")
        if counts.failed:
            self.add_error(f"{rel_path} contains failed {label} results: {counts.failed}")
        if isinstance(total, int) and total != len(entries):
            self.add_error(f"{rel_path} summary.total does not match results length: {total} vs {len(entries)}")
        if isinstance(passed, int) and passed != counts.passed:
            self.add_error(f"{rel_path} summary.passed does not match passed results: {passed} vs {counts.passed}")

    def validate_result_summary(self, rel_path: str, *, label: str) -> None:
        data = self.load_json(rel_path)
        if data is None:
            return
        if not isinstance(data, dict):
            self.add_error(f"{rel_path} must be a JSON object")
            return

        status = data.get("status")
        if str(status).strip().lower() not in PASSED_VALUES and str(status).strip() not in PASSED_VALUES:
            self.add_error(f"{rel_path} status is not passed: {status!r}")

        entries = data.get("results", [])
        if not isinstance(entries, list) or not entries:
            self.add_error(f"{rel_path} must contain a non-empty results list")
            entries = []

        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            self.add_error(f"{rel_path} summary must be an object")
            return

        counts = self.count_result_summary_entries(rel_path, label, entries)
        self.validate_result_summary_counts(rel_path, label, summary, entries, counts)

    def aclnn_workspace_tensor_names(self) -> tuple[list[str], list[str]]:
        op_api_dir = self.operator_dir / "op_api"
        if not op_api_dir.is_dir():
            return [], []

        for header in sorted(op_api_dir.glob("aclnn_*.h")):
            text = header.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"\b\w+GetWorkspaceSize\s*\((.*?)\)\s*;", text, re.S)
            if not match:
                continue
            tensor_names = re.findall(r"\bconst\s+aclTensor\s*\*\s*([A-Za-z_]\w*)", match.group(1))
            if not tensor_names:
                continue
            if len(tensor_names) == 1:
                return tensor_names, []
            return tensor_names[:-1], [tensor_names[-1]]
        return [], []

    def ut_source_text(self) -> tuple[str, list[str]]:
        ut_dir = self.operator_dir / "tests/ut"
        if not ut_dir.is_dir():
            return "", []

        chunks: list[str] = []
        rel_paths: list[str] = []
        for path in sorted(ut_dir.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            rel_path = path.relative_to(self.operator_dir).as_posix()
            rel_paths.append(rel_path)
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(chunks), rel_paths

    def validate_ut_source_alignment(self) -> None:
        input_names, output_names = self.aclnn_workspace_tensor_names()
        source_text, source_paths = self.ut_source_text()

        if not source_paths:
            self.add_error("missing UT source files under tests/ut")
            return
        if not input_names:
            return

        allowed_names = set(input_names) | set(output_names) | COMMON_OUTPUT_NAMES
        missing_input_names = [
            name for name in input_names if not re.search(rf"\b{re.escape(name)}\b", source_text)
        ]
        if missing_input_names:
            self.add_error(
                "tests/ut source does not reference ACLNN input names: "
                f"{missing_input_names}"
            )

        stale_aliases = {"x1", "x2", "input1", "input2"} - allowed_names
        for alias in sorted(stale_aliases):
            if re.search(rf"\b{re.escape(alias)}\b", source_text):
                self.add_error(
                    "tests/ut source uses stale or undeclared tensor identifier "
                    f"{alias!r}; expected ACLNN inputs {input_names}"
                )

    def validate_ut(self) -> None:
        self.validate_result_summary("tests/ut/test-report.json", label="UT")
        self.validate_ut_source_alignment()

    def validate_blackbox_case_id_alignment(
        self,
        csv_ids: set[str],
        manifest_ids: set[str],
    ) -> None:
        if not csv_ids or not manifest_ids or csv_ids == manifest_ids:
            return
        self.add_error(
            "case_manifest.json case ids do not match CSV case ids: "
            f"missing_in_manifest={sorted(csv_ids - manifest_ids)}, "
            f"extra_in_manifest={sorted(manifest_ids - csv_ids)}"
        )

    def collect_blackbox_result_status(
        self,
        required_ids: set[str],
        manifest: dict[str, dict[str, Any]],
        manifest_level_by_id: dict[str, str],
        results: dict[str, dict[str, Any]],
    ) -> BlackboxResultStatus:
        passed_ids: set[str] = set()
        failed_ids: set[str] = set()
        smoke_only_ids: set[str] = set()
        route_errors: list[str] = []
        passed_level_counts = {level: 0 for level in BLACKBOX_LEVELS}

        for case_id in sorted(required_ids):
            entry = results.get(case_id)
            if not entry:
                continue
            if entry.get("smoke_only") is True:
                smoke_only_ids.add(case_id)
                continue
            accepted_routes = manifest[case_id].get("accepted_routes", [])
            route = entry.get("route")
            if self.uses_unaccepted_route(route, accepted_routes):
                route_errors.append(f"{case_id}:{route}")
            if self.is_passed_entry(entry):
                passed_ids.add(case_id)
                level = manifest_level_by_id.get(case_id)
                if level in passed_level_counts:
                    passed_level_counts[level] += 1
            else:
                failed_ids.add(case_id)
        return BlackboxResultStatus(
            passed_ids,
            failed_ids,
            smoke_only_ids,
            route_errors,
            passed_level_counts,
        )

    def validate_blackbox_result_coverage(
        self,
        result_rel_path: str,
        required_ids: set[str],
        results: dict[str, dict[str, Any]],
        status: BlackboxResultStatus,
    ) -> set[str]:
        missing_ids = required_ids - set(results)
        if missing_ids:
            if results:
                self.add_error(
                    f"{result_rel_path} appears to contain partial/debug results: "
                    f"results={len(results)}, required={len(required_ids)}. "
                    "Keep full required-case evidence in tests/st/results/st_dev_result.json "
                    "or tests/st/results/st_real_result.json; write single-case debug output under "
                    "tests/st/results/debug/ or st_real_debug_*.json instead."
                )
            self.add_error(f"{result_rel_path} missing required case ids: {sorted(missing_ids)}")
        if status.failed_ids:
            self.add_error(f"{result_rel_path} failed required case ids: {sorted(status.failed_ids)}")
        if status.smoke_only_ids:
            self.add_error(
                f"{result_rel_path} smoke_only results cannot count as passed: {sorted(status.smoke_only_ids)}"
            )
        if status.route_errors:
            self.add_error(f"{result_rel_path} uses routes outside accepted_routes: {status.route_errors}")
        return missing_ids

    def validate_blackbox(self, result_rel_path: str) -> None:
        test_design_path = self.test_design_file()
        targets = self.blackbox_case_target_counts(test_design_path) if test_design_path else {}
        csv_rows = self.csv_case_rows()
        csv_ids = {row["_case_id"] for row in csv_rows}
        csv_counts, unknown_levels = self.csv_level_row_counts(csv_rows)
        if unknown_levels:
            self.add_error(f"blackbox CSV rows lack L0/L1/L2 level: {unknown_levels[:20]}")
        manifest = self.manifest_cases()
        manifest_ids = set(manifest)
        manifest_counts, manifest_level_by_id = self.manifest_level_counts(manifest, csv_rows)
        self.validate_blackbox_case_id_alignment(csv_ids, manifest_ids)

        results = self.result_entries(result_rel_path)
        required_ids = {
            case_id
            for case_id, item in manifest.items()
            if bool(item.get("required", True))
        }
        result_status = self.collect_blackbox_result_status(required_ids, manifest, manifest_level_by_id, results)
        missing_ids = self.validate_blackbox_result_coverage(
            result_rel_path,
            required_ids,
            results,
            result_status,
        )
        self.validate_blackbox_target_consistency(
            result_rel_path,
            targets=targets,
            csv_counts=csv_counts,
            manifest_counts=manifest_counts,
            passed_level_counts=result_status.passed_level_counts,
        )

        self.blackbox_summary = {
            "required": len(required_ids),
            "passed": len(result_status.passed_ids),
            "failed": len(result_status.failed_ids) + len(result_status.smoke_only_ids),
            "missing": len(missing_ids),
        }

        self.validate_blackbox_execution_evidence(required_ids, manifest, results)
        self.validate_blackbox_design_constraints()

    def whitebox_case_ids(self, rel_path: str) -> set[str]:
        return {case_id for case_id, _ in self.whitebox_case_entries(rel_path)}

    def whitebox_case_entries(self, rel_path: str) -> list[tuple[str, dict[str, Any]]]:
        if not self.rel(rel_path).is_file():
            return []
        data = self.load_json(rel_path)
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list):
            self.add_error(f"{rel_path} must contain a cases list")
            return []
        enabled: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(cases):
            if not isinstance(item, dict):
                self.add_error(f"{rel_path} case at index {index} is not an object")
                continue
            if item.get("enabled", True) is False:
                continue
            case_id = str(item.get("case_id") or item.get("id") or "").strip()
            if not case_id:
                self.add_error(f"{rel_path} case at index {index} lacks case_id/id")
                continue
            enabled.append((case_id, item))
        if not enabled:
            self.add_error(f"{rel_path} has no enabled case ids")
        return enabled

    def enabled_whitebox_ids(self) -> set[str]:
        return self.whitebox_case_ids("tests/whitebox/S5_mapped_cases_high.json")

    def collected_whitebox_ids(self) -> set[str]:
        data = self.load_json("tests/whitebox/results/pytest_collect.json")
        if isinstance(data, dict):
            raw = data.get("collected_case_ids", data.get("case_ids", data.get("collected", [])))
        elif isinstance(data, list):
            raw = data
        else:
            raw = []

        collected: set[str] = set()
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    case_id = str(item.get("case_id") or item.get("id") or "").strip()
                else:
                    case_id = str(item).strip()
                if case_id:
                    collected.add(case_id)
        else:
            self.add_error("pytest_collect.json collected case ids must be a list")
        return collected

    def validate_whitebox(self) -> None:
        self.whitebox_evidence_paths = []
        for rel_path in WHITEBOX_CONTRACT_FILES:
            self.require_file(rel_path)
        self.validate_no_fabricated_evidence_writers()

        whitebox_tests = list((self.operator_dir / "tests/whitebox").glob("S6_test_*.py"))
        if not whitebox_tests:
            self.add_error("missing whitebox pytest file: tests/whitebox/S6_test_{op_name}.py")

        enabled_ids = self.enabled_whitebox_ids()
        self.validate_whitebox_skill_workflow(enabled_ids)
        st_passed_ids, st_failed_ids, st_missing_ids = self.validate_whitebox_st_evidence(enabled_ids)
        self.validate_whitebox_st_execution_evidence(enabled_ids)

        collected_ids = self.collected_whitebox_ids()
        passed_ids, failed_ids, missing_result_ids = self.validate_whitebox_pytest_evidence(
            enabled_ids,
            collected_ids,
        )

        self.validate_whitebox_reports()

        self.whitebox_summary = {
            "enabled": len(enabled_ids),
            "passed": len(st_passed_ids),
            "failed": len(st_failed_ids),
            "missing": len(st_missing_ids),
            "st_passed": len(st_passed_ids),
            "st_failed": len(st_failed_ids),
            "st_missing": len(st_missing_ids),
            "collected": len(collected_ids),
            "pytest_passed": len(passed_ids),
            "pytest_failed": len(failed_ids),
            "pytest_missing": len(missing_result_ids),
        }

    def validate_whitebox_pytest_evidence(
        self,
        enabled_ids: set[str],
        collected_ids: set[str],
    ) -> tuple[set[str], set[str], set[str]]:
        pytest_results = self.result_entries("tests/whitebox/results/pytest_result.json")
        if enabled_ids != collected_ids:
            self.add_error(
                "pytest_collect.json case ids do not match enabled whitebox cases: "
                f"missing_in_collect={sorted(enabled_ids - collected_ids)}, "
                f"extra_in_collect={sorted(collected_ids - enabled_ids)}"
            )

        passed_ids: set[str] = set()
        failed_ids: set[str] = set()
        for case_id in sorted(enabled_ids):
            entry = pytest_results.get(case_id)
            if not entry:
                continue
            if self.is_passed_entry(entry):
                passed_ids.add(case_id)
            else:
                failed_ids.add(case_id)
        missing_result_ids = enabled_ids - set(pytest_results)
        if missing_result_ids:
            self.add_error(f"pytest_result.json missing enabled case ids: {sorted(missing_result_ids)}")
        if failed_ids:
            self.add_error(f"pytest_result.json failed enabled case ids: {sorted(failed_ids)}")
        return passed_ids, failed_ids, missing_result_ids

    def validate_whitebox_skill_workflow(self, enabled_ids: set[str]) -> None:
        self.validate_whitebox_provenance(len(enabled_ids))
        self.validate_whitebox_param_def()
        self.validate_whitebox_case_generation(enabled_ids)

    def whitebox_expected_gate_cases(self, provenance: dict[str, Any], rel_path: str) -> int:
        profile = provenance.get("coverage_profile")
        if not isinstance(profile, dict):
            return DEFAULT_WHITEBOX_GATE_CASES

        value = profile.get("minimum_gate_cases", profile.get("minimum_low_cases"))
        reason = str(profile.get("reason") or "").strip()
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            self.add_error(f"{rel_path} coverage_profile.minimum_gate_cases must be a positive integer")
            return DEFAULT_WHITEBOX_GATE_CASES
        if value < DEFAULT_WHITEBOX_GATE_CASES and (
            len(reason) < 20 or re.search(r"\b(TBD|TODO)\b|待确认|未确认", reason, re.I)
        ):
            self.add_error(f"{rel_path} coverage_profile.reason must explain reduced gate-case expectation")
        return value

    def validate_whitebox_provenance_source(self, data: dict[str, Any], rel_path: str) -> None:
        skill = str(data.get("skill") or "").strip()
        if skill != "ascendc-whitebox-design":
            self.add_error(f"{rel_path} skill must be 'ascendc-whitebox-design', got {skill!r}")

        workflow = str(data.get("workflow") or data.get("skill_path") or data.get("source") or "").strip()
        if not workflow:
            self.add_error(f"{rel_path} must record a repository-relative whitebox skill workflow/source path")
        elif workflow.startswith("/") or workflow.startswith("../") or "/../" in workflow:
            self.add_error(
                f"{rel_path} workflow must be repository-relative under "
                f"{WHITEBOX_SKILL_PATH_PREFIX!r}, got {workflow!r}"
            )
        elif not workflow.startswith(WHITEBOX_SKILL_PATH_PREFIX):
            self.add_error(
                f"{rel_path} workflow must reference {WHITEBOX_SKILL_PATH_PREFIX!r}, got {workflow!r}"
            )

    def validate_whitebox_provenance_step4(self, data: dict[str, Any], rel_path: str) -> None:
        raw_steps = data.get("steps")
        if raw_steps is not None and not isinstance(raw_steps, list):
            self.add_error(f"{rel_path} steps must be a list")

        step4 = data.get("step4")
        if step4 is not None:
            if not isinstance(step4, dict):
                self.add_error(f"{rel_path} step4 decision must be recorded as an object")
                step4 = {}
            decision = str(step4.get("decision") or "").strip().lower()
            if decision not in {"approved", "skipped", "auto_continue"}:
                self.add_error(
                    f"{rel_path} step4.decision must be approved/skipped/auto_continue, got {decision!r}"
                )
            reason = str(step4.get("reason") or "").strip()
            if decision in {"skipped", "auto_continue"} and len(reason) < 10:
                self.add_error(f"{rel_path} step4.reason must explain automatic Step 4 continuation")

    def validate_whitebox_case_reduction(
        self,
        data: dict[str, Any],
        rel_path: str,
        enabled_count: int,
        expected_gate_cases: int,
    ) -> None:
        if not enabled_count or enabled_count >= expected_gate_cases:
            return

        reduction = data.get("case_reduction_reason")
        if not isinstance(reduction, dict):
            self.add_error(
                f"{rel_path} must include case_reduction_reason when enabled whitebox cases "
                f"are below {expected_gate_cases}: actual={enabled_count}"
            )
            return
        self.validate_whitebox_case_reduction_details(rel_path, reduction)

    def validate_whitebox_case_reduction_details(self, rel_path: str, reduction: dict[str, Any]) -> None:
        reason = str(reduction.get("reason") or "").strip()
        if len(reason) < 20 or re.search(r"\b(TBD|TODO)\b|待确认|未确认", reason, re.I):
            self.add_error(f"{rel_path} case_reduction_reason.reason is missing or unresolved")
        for key in ("enabled_case_count", "low_case_count", "high_case_count"):
            value = reduction.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                self.add_error(f"{rel_path} case_reduction_reason.{key} must be a non-negative integer")

    def validate_whitebox_provenance(self, enabled_count: int) -> None:
        rel_path = "tests/whitebox/WORKFLOW_PROVENANCE.json"
        if not self.rel(rel_path).is_file():
            return
        data = self.load_json(rel_path)
        if not isinstance(data, dict):
            if data is not None:
                self.add_error(f"{rel_path} must be a JSON object")
            return

        self.validate_whitebox_provenance_source(data, rel_path)
        self.validate_whitebox_provenance_step4(data, rel_path)
        expected_gate_cases = self.whitebox_expected_gate_cases(data, rel_path)
        self.validate_whitebox_case_reduction(data, rel_path, enabled_count, expected_gate_cases)

    def validate_whitebox_param_def(self) -> None:
        rel_path = "tests/whitebox/S2P2_param_def.json"
        if not self.rel(rel_path).is_file():
            return
        data = self.load_json(rel_path)
        if not isinstance(data, dict):
            if data is not None:
                self.add_error(f"{rel_path} must be a JSON object")
            return

        tiling_keys = data.get("tiling_keys")
        if not isinstance(tiling_keys, list) or not tiling_keys:
            self.add_error(f"{rel_path} must define non-empty top-level tiling_keys")

        groups = data.get("groups")
        if not isinstance(groups, list) or not groups:
            self.add_error(f"{rel_path} must define non-empty groups produced by whitebox Step 2")
            return

        per_dtype_key_found = False
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                self.add_error(f"{rel_path} groups[{index}] must be an object")
                continue
            per_dtype = group.get("per_dtype")
            if isinstance(per_dtype, list):
                entries = [(f"[{dtype_index}]", item) for dtype_index, item in enumerate(per_dtype)]
            elif isinstance(per_dtype, dict):
                entries = [(f"[{dtype_name}]", item) for dtype_name, item in per_dtype.items()]
            else:
                self.add_error(f"{rel_path} groups[{index}].per_dtype must be a non-empty list or object")
                continue
            if not entries:
                self.add_error(f"{rel_path} groups[{index}].per_dtype must be non-empty")
                continue
            for entry_label, item in entries:
                if not isinstance(item, dict):
                    self.add_error(f"{rel_path} groups[{index}].per_dtype{entry_label} must be an object")
                    continue
                if "key" in item:
                    per_dtype_key_found = True

        if not per_dtype_key_found:
            self.add_error(f"{rel_path} groups[].per_dtype entries must include tiling key entries")

    def validate_whitebox_case_generation(self, enabled_ids: set[str]) -> None:
        low_entries = self.whitebox_case_entries("tests/whitebox/S5_mapped_cases_low.json")
        high_entries = self.whitebox_case_entries("tests/whitebox/S5_mapped_cases_high.json")
        low_ids = {case_id for case_id, _ in low_entries}
        high_ids = {case_id for case_id, _ in high_entries}
        self.validate_whitebox_mapped_case_schema("tests/whitebox/S5_mapped_cases_low.json")
        self.validate_whitebox_mapped_case_schema("tests/whitebox/S5_mapped_cases_high.json")

        if low_ids and high_ids and len(high_ids) < len(low_ids):
            self.add_error(
                "S5_mapped_cases_high.json must include at least as many enabled cases as "
                f"S5_mapped_cases_low.json: high={len(high_ids)}, low={len(low_ids)}"
            )
        self.validate_whitebox_data_range_expansion(low_entries, high_entries)

    def validate_whitebox_data_range_expansion(
        self,
        low_entries: list[tuple[str, dict[str, Any]]],
        high_entries: list[tuple[str, dict[str, Any]]],
    ) -> None:
        non_empty_low_cases = []
        for case_id, item in low_entries:
            ranges = self.whitebox_input_data_ranges(item)
            if not ranges:
                continue
            non_empty_low_cases.append(case_id)
            invalid = [dr for dr in ranges if dr != "normal"]
            if invalid:
                self.add_error(
                    f"S5_mapped_cases_low.json case {case_id} must use normal _data_range for all inputs, "
                    f"got {invalid}"
                )

        high_ranges: list[str] = []
        for _, item in high_entries:
            high_ranges.extend(self.whitebox_input_data_ranges(item))
        non_normal_high_ranges = [dr for dr in high_ranges if dr in WHITEBOX_NON_NORMAL_DATA_RANGES]

        if non_empty_low_cases and high_entries:
            if len(high_entries) <= len(low_entries):
                self.add_error(
                    "S5_mapped_cases_high.json must contain data_range-expanded cases beyond low: "
                    f"high={len(high_entries)}, low={len(low_entries)}"
                )
            if not non_normal_high_ranges:
                self.add_error(
                    "S5_mapped_cases_high.json must include data_range-expanded cases with non-normal "
                    "input _data_range values"
                )

    def validate_whitebox_mapped_case_schema(self, rel_path: str) -> None:
        for case_id, item in self.whitebox_case_entries(rel_path):
            params = item.get("params")
            tensors = item.get("tensors")
            if not isinstance(params, dict) or not isinstance(tensors, dict):
                self.add_error(
                    f"{rel_path} case {case_id} must contain Step 5 mapped params/tensors"
                )
                continue

            inputs = tensors.get("inputs")
            outputs = tensors.get("outputs")
            if not isinstance(inputs, dict) or not isinstance(outputs, dict):
                self.add_error(
                    f"{rel_path} case {case_id} tensors must contain inputs and outputs objects"
                )
                continue
            if not any(spec is not None for spec in inputs.values()):
                self.add_error(f"{rel_path} case {case_id} tensors.inputs must contain at least one tensor spec")
            if not outputs:
                self.add_error(f"{rel_path} case {case_id} tensors.outputs must contain at least one tensor spec")

            self.validate_whitebox_tensor_specs(rel_path, case_id, "inputs", inputs, require_data_range=True)
            self.validate_whitebox_tensor_specs(rel_path, case_id, "outputs", outputs, require_data_range=False)

    def validate_whitebox_tensor_specs(
        self,
        rel_path: str,
        case_id: str,
        kind: str,
        tensors: dict[str, Any],
        *,
        require_data_range: bool,
    ) -> None:
        for name, spec in tensors.items():
            if spec is None:
                continue
            if not isinstance(spec, dict):
                self.add_error(f"{rel_path} case {case_id} tensors.{kind}.{name} must be an object or null")
                continue
            shape = spec.get("shape")
            dtype = spec.get("dtype")
            if not isinstance(shape, list) or not str(dtype or "").strip():
                self.add_error(
                    f"{rel_path} case {case_id} tensors.{kind}.{name} must include shape and dtype"
                )
            if require_data_range:
                data_range = str(spec.get("_data_range") or "").strip()
                if not data_range:
                    self.add_error(
                        f"{rel_path} case {case_id} tensors.{kind}.{name} must include _data_range"
                    )
                elif data_range not in WHITEBOX_DATA_RANGES:
                    self.add_error(
                        f"{rel_path} case {case_id} tensors.{kind}.{name} has unsupported _data_range "
                        f"{data_range!r}"
                    )

    def validate_whitebox_st_evidence(self, expected_ids: set[str]) -> tuple[set[str], set[str], set[str]]:
        rel_path = "tests/whitebox/results/st_result.json"
        path = self.require_file(rel_path)
        if path is not None:
            self.whitebox_evidence_paths.append(rel_path)
        else:
            return set(), set(), set(expected_ids)

        results = self.result_entries(rel_path)
        seen = set(results)
        missing = expected_ids - seen if expected_ids else set()
        extra = seen - expected_ids if expected_ids else set()
        passed: set[str] = set()
        failed: set[str] = set()
        route_errors: list[str] = []

        for case_id in sorted(expected_ids):
            entry = results.get(case_id)
            if not entry:
                continue
            route = str(entry.get("route", "")).strip()
            if route not in {"st_cpp", "st_real"}:
                route_errors.append(f"{case_id}:{route or '<missing>'}")
            if self.is_passed_entry(entry):
                passed.add(case_id)
            else:
                failed.add(case_id)

        if missing:
            self.add_error(f"{rel_path} missing enabled case ids: {sorted(missing)}")
        if extra:
            self.add_error(f"{rel_path} contains result ids outside enabled whitebox cases: {sorted(extra)}")
        if failed:
            self.add_error(f"{rel_path} failed enabled case ids: {sorted(failed)}")
        if route_errors:
            self.add_error(f"{rel_path} uses unsupported routes: {route_errors}")

        return passed, failed, missing

    def whitebox_execution_evidence_entries(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        excluded_parts = {
            "tests/st/testcases/",
            "tests/st/results/",
            "tests/st/design/",
            "tests/whitebox/results/",
        }
        excluded_names = {
            "S2P2_param_def.json",
            "S5_mapped_cases_low.json",
            "S5_mapped_cases_high.json",
            "S6_tilingkey_coverage.json",
            "WORKFLOW_PROVENANCE.json",
        }

        for rel_base in ("tests/st", "tests/whitebox"):
            entries.extend(
                self.execution_evidence_entries(
                    rel_base,
                    excluded_parts=excluded_parts,
                    excluded_names=excluded_names,
                )
            )
        return entries

    def whitebox_execution_evidence_text(self) -> tuple[str, list[str]]:
        entries = self.whitebox_execution_evidence_entries()
        chunks = [text for _, text in entries]
        rel_paths = [rel_path for rel_path, _ in entries]
        return "\n".join(chunks), rel_paths

    def validate_whitebox_st_execution_evidence(self, expected_ids: set[str]) -> None:
        if not expected_ids:
            return

        evidence_entries = self.whitebox_execution_evidence_entries()
        evidence_text = "\n".join(text for _, text in evidence_entries)
        evidence_paths = [rel_path for rel_path, _ in evidence_entries]
        missing_executable_ids = []
        for case_id in sorted(expected_ids):
            has_executable_mapping = any(
                case_id in text
                and any(pattern.search(text) for pattern in ST_IMPLEMENTATION_EVIDENCE_PATTERNS)
                and any(pattern.search(text) for pattern in ST_EXECUTABLE_EVIDENCE_PATTERNS)
                for _, text in evidence_entries
            )
            if not has_executable_mapping:
                missing_executable_ids.append(case_id)
        if missing_executable_ids:
            self.add_error(
                "whitebox ST execution evidence lacks executable case mappings: "
                f"{missing_executable_ids}; checked files={evidence_paths}"
            )
        if not any(pattern.search(evidence_text) for pattern in ST_IMPLEMENTATION_EVIDENCE_PATTERNS):
            self.add_error(
                "whitebox ST execution evidence lacks implementation invocation marker; "
                "whitebox ST routes must show ACLNN/custom_opp/libcust/op_host/op_kernel "
                f"or IMPLEMENTATION_UNDER_TEST evidence; checked files={evidence_paths}"
            )

    def validate_no_fabricated_evidence_writers(self) -> None:
        helper_roots = ("scripts", "tools")
        write_markers = ("write_json", "write_text", "json.dumps", "json.dump", "Path(")
        pass_markers = ('"passed"', "'passed'", '"status": "passed"', "'status': 'passed'")
        for helper_root in helper_roots:
            base = self.rel(helper_root)
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix not in {".py", ".sh"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                outputs = [name for name in FORBIDDEN_HELPER_EVIDENCE_OUTPUTS if name in text]
                if not outputs:
                    continue
                if any(marker in text for marker in write_markers) and any(marker in text for marker in pass_markers):
                    rel_path = path.relative_to(self.operator_dir).as_posix()
                    self.add_error(
                        f"{rel_path} helper script appears to synthesize workflow pass evidence: {outputs}"
                    )

    def validate_s3_verification_report(self) -> None:
        rel_path = "tests/whitebox/S3_verification_report.json"
        if not self.rel(rel_path).is_file():
            return

        verification = self.load_json(rel_path)
        if not isinstance(verification, dict):
            if verification is not None:
                self.add_error(f"{rel_path} must be a JSON object")
            return

        status = verification.get("status")
        if str(status).strip().lower() not in {"pass", "passed", "success", "ok", "pass_with_warnings"}:
            self.add_error(f"{rel_path} status is not passed: {status!r}")

        checks = verification.get("checks", [])
        if isinstance(checks, list):
            failed_checks = [
                str(check.get("id", index))
                for index, check in enumerate(checks)
                if isinstance(check, dict) and str(check.get("status", "")).strip().lower() == "fail"
            ]
            if failed_checks:
                self.add_error(f"{rel_path} failed checks: {failed_checks}")
        elif checks is not None:
            self.add_error(f"{rel_path} checks must be a list")

    def validate_whitebox_reports(self) -> None:
        self.validate_s3_verification_report()

        coverage = self.load_json("tests/whitebox/S6_tilingkey_coverage.json")
        if not isinstance(coverage, dict) and coverage is not None:
            self.add_error("tests/whitebox/S6_tilingkey_coverage.json must be a JSON object")

    def evidence_index_paths(self, entries: Any) -> dict[str, dict[str, Any]]:
        paths: dict[str, dict[str, Any]] = {}
        if not isinstance(entries, list):
            self.add_error("evidence_index.json evidence_paths must be a list")
            return paths
        for item in entries:
            self.record_evidence_index_path(paths, item)
        return paths

    def validate_required_evidence_paths(self, paths: dict[str, dict[str, Any]]) -> None:
        for rel_path in [
            "tests/st/case_manifest.json",
            "tests/st/results/st_real_result.json",
            "tests/whitebox/WORKFLOW_PROVENANCE.json",
            "tests/whitebox/S2P2_param_def.json",
            "tests/whitebox/S5_mapped_cases_low.json",
            "tests/whitebox/S5_mapped_cases_high.json",
            "tests/whitebox/S6_tilingkey_coverage.json",
            "tests/whitebox/results/pytest_collect.json",
            "tests/whitebox/results/pytest_result.json",
            *self.whitebox_evidence_paths,
        ]:
            if rel_path not in paths:
                self.add_error(f"evidence_index.json missing evidence path: {rel_path}")

    def validate_evidence_index(self) -> None:
        data = self.load_json("tests/reports/evidence_index.json")
        if not isinstance(data, dict):
            self.add_error("evidence_index.json must be a JSON object")
            return

        if data.get("status") != "passed":
            self.add_error(f"evidence_index.json status expected 'passed', got {data.get('status')!r}")

        paths = self.evidence_index_paths(data.get("evidence_paths", []))
        self.validate_required_evidence_paths(paths)

        summary = data.get("summary", {})
        if isinstance(summary, dict):
            self.compare_summary("blackbox", self.blackbox_summary, summary.get("blackbox"))
            self.compare_summary("whitebox", self.whitebox_summary, summary.get("whitebox"))
        else:
            self.add_error("evidence_index.json summary must be an object")

    def compare_summary(self, name: str, computed: dict[str, int], actual: Any) -> None:
        if not isinstance(actual, dict):
            self.add_error(f"evidence_index.json summary.{name} must be an object")
            return
        for key, expected_value in computed.items():
            if key in actual and actual[key] != expected_value:
                self.add_error(
                    f"evidence_index.json summary.{name}.{key} expected "
                    f"{expected_value}, got {actual[key]}"
                )

    def validate_cp2(self) -> None:
        self.validate_report_status("tests/reports/iter3-integration-report.md")
        self.validate_ut()
        self.validate_blackbox("tests/st/results/st_dev_result.json")

    def validate_cp3(self) -> None:
        self.validate_report_status("tests/reports/iter3-acceptance-report.md")
        self.validate_report_status("tests/reports/test-branches-merge-exec-report.md")
        self.validate_ut()
        self.validate_blackbox("tests/st/results/st_real_result.json")
        self.validate_whitebox()
        self.validate_evidence_index()

    def print_result(self, stage: str) -> int:
        configure_output_logging()
        LOGGER.info("stage: %s", stage)
        if self.blackbox_summary:
            LOGGER.info("blackbox: %s", json.dumps(self.blackbox_summary, sort_keys=True))
        if self.whitebox_summary:
            LOGGER.info("whitebox: %s", json.dumps(self.whitebox_summary, sort_keys=True))
        if self.errors:
            LOGGER.info("errors:")
            for error in self.errors:
                LOGGER.info("- %s", error)
            LOGGER.info("STATUS: FAILED")
            return 1
        LOGGER.info("STATUS: PASSED")
        return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cp2", "cp3"], required=True)
    parser.add_argument("--operator-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    operator_dir = args.operator_dir.resolve()
    validator = WorkflowValidator(operator_dir)
    if not operator_dir.is_dir():
        validator.add_error(f"operator directory does not exist: {operator_dir}")
    elif args.stage == "cp2":
        validator.validate_cp2()
    else:
        validator.validate_cp3()
    return validator.print_result(args.stage)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
