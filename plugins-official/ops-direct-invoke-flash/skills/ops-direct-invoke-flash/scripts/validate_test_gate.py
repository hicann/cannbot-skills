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
"""Validate ops-direct-invoke-flash test gate evidence.

Switch: `harness.test_gate` (on/off, default off) read from the Team AGENTS.md.

When `on`, the gate is bound to the REAL skill artifacts so it cannot be passed
with a hand-written "shadow" result file:

  * cases_paths  — the cases the skills actually generated (the case-id ground
                   truth): blackbox `*_L0/L1/L2_test_cases.csv` (ascendc-st-design),
                   whitebox `S5_mapped_cases_low.json` (ascendc-whitebox-design).
  * result_paths — the real execution results: blackbox ST result JSON
                   ({case_id,status}); whitebox TTK `*_result.csv` (precision_status).
  * log_paths    — a non-empty execution log (proof the cases were actually run).

Binding rule (per branch): every generated case_id must appear in the results
with a passing status; a result that omits a generated case, marks one non-pass,
or references a case_id that is NOT in the generated set fails the gate. The
self-reported `status` in test_gate.json is never trusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PASSED_VALUES = {"passed", "pass", "success", "ok", "通过", "✅通过"}
ID_KEYS = ("case_id", "testcase_name", "id")
STATUS_KEYS = ("precision_status", "status", "result")
BLACKBOX_PROVENANCE = "test-harness/blackbox/WORKFLOW_PROVENANCE.json"
WHITEBOX_PROVENANCE = "test-harness/whitebox/WORKFLOW_PROVENANCE.json"
TEST_GATE_RESULT = "test-harness/results/test_gate.json"
MAX_LISTED = 20


def _frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    return text[3:end] if end != -1 else ""


def _normalize_config_value(value: str) -> str:
    return value.split("#", 1)[0].strip().strip("\"'")


def _read_test_gate_config(agents_path: Path) -> str:
    if not agents_path.is_file():
        return "off"
    frontmatter = _frontmatter(agents_path.read_text(encoding="utf-8", errors="replace"))
    inline = re.search(r"(?m)^harness:\s*\{[^}]*test_gate\s*:\s*([^,}#]+)", frontmatter)
    if inline:
        return _normalize_config_value(inline.group(1))
    in_harness = False
    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if re.match(r"^\S", line):
            if line.startswith("harness:"):
                in_harness = True
                continue
            if in_harness:
                break
        if in_harness:
            m = re.match(r"\s+test_gate\s*:\s*(.+)$", line)
            if m:
                return _normalize_config_value(m.group(1))
    return "off"


def _first_key(row: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return row[k]
    return None


def _is_pass(raw: Any) -> bool:
    text = str(raw).strip()
    return text.lower() in PASSED_VALUES or text in PASSED_VALUES


class TestGateValidator:
    def __init__(self, plugin_root: Path, operator_doc_dir: Path) -> None:
        self.plugin_root = plugin_root
        self.operator_doc_dir = operator_doc_dir
        self.errors: list[str] = []
        self.summary: dict[str, str] = {}
        self.test_gate = _read_test_gate_config(self.plugin_root / "AGENTS.md")

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def load_json(self, path: Path) -> Any | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.add_error(f"invalid JSON in {path.name}: {exc}")
            return None

    def validate_provenance(self, rel_path: str, expected_skill: str) -> None:
        path = self._safe_path(rel_path, f"{expected_skill} provenance")
        if path is None:
            return
        data = self.load_json(path)
        if not isinstance(data, dict):
            if data is not None:
                self.add_error(f"{rel_path} must be a JSON object")
            return
        if str(data.get("skill", "")).strip() != expected_skill:
            self.add_error(f"{rel_path} skill must be {expected_skill!r}, got {str(data.get('skill','')).strip()!r}")
        workflow = str(data.get("workflow", "")).strip()
        if expected_skill not in workflow:
            self.add_error(f"{rel_path} workflow must reference {expected_skill}, got {workflow!r}")

    def case_ids(self, rel_paths: Any, branch: str) -> set[str]:
        ids: set[str] = set()
        if not isinstance(rel_paths, list) or not rel_paths:
            self.add_error(f"{TEST_GATE_RESULT} {branch}.cases_paths must be a non-empty list")
            return ids
        for rel in rel_paths:
            path = self._safe_path(rel, f"{branch} cases")
            if path is None:
                continue
            for row in self._rows(path):
                cid = _first_key(row, ID_KEYS)
                if cid is not None:
                    ids.add(str(cid).strip())
        if not ids:
            self.add_error(f"{branch} generated cases yielded no case ids (checked {rel_paths})")
        return ids

    def result_status(self, rel_paths: Any, branch: str) -> dict[str, str]:
        results: dict[str, str] = {}
        if not isinstance(rel_paths, list) or not rel_paths:
            self.add_error(f"{TEST_GATE_RESULT} {branch}.result_paths must be a non-empty list")
            return results
        for rel in rel_paths:
            path = self._safe_path(rel, f"{branch} result")
            if path is None:
                continue
            for row in self._rows(path):
                cid = _first_key(row, ID_KEYS)
                status = _first_key(row, STATUS_KEYS)
                if cid is None:
                    self.add_error(f"{branch} result row lacks a case id ({path.name})")
                    continue
                if status is None:
                    self.add_error(f"{branch} result for case {str(cid).strip()} lacks a status ({path.name})")
                    continue
                results[str(cid).strip()] = str(status).strip()
        return results

    def require_logs(self, rel_paths: Any, branch: str) -> None:
        if not isinstance(rel_paths, list) or not rel_paths:
            self.add_error(f"{TEST_GATE_RESULT} {branch}.log_paths must be a non-empty list (execution evidence)")
            return
        for rel in rel_paths:
            self._safe_path(rel, f"{branch} execution log")

    def validate_branch(self, branch: str, branch_data: dict[str, Any]) -> None:
        gen_ids = self.case_ids(branch_data.get("cases_paths"), branch)
        results = self.result_status(branch_data.get("result_paths"), branch)
        self.require_logs(branch_data.get("log_paths"), branch)

        if not gen_ids or not results:
            return  # specific errors already recorded

        missing = sorted(gen_ids - results.keys())
        if missing:
            self.add_error(
                f"{branch} results do not cover generated cases "
                f"(missing {len(missing)}): {missing[:MAX_LISTED]}"
            )
        not_pass = sorted(cid for cid in gen_ids & results.keys() if not _is_pass(results[cid]))
        if not_pass:
            self.add_error(f"{branch} cases did not pass ({len(not_pass)}): "
                           + ", ".join(f"{cid}={results[cid]!r}" for cid in not_pass[:MAX_LISTED]))
        unknown = sorted(results.keys() - gen_ids)
        if unknown:
            self.add_error(f"{branch} results reference case ids NOT in the generated set "
                           f"(possible fabricated evidence; {len(unknown)}): {unknown[:MAX_LISTED]}")

    def validate_test_gate_result(self) -> None:
        path = self._safe_path(TEST_GATE_RESULT, "test_gate")
        if path is None:
            return
        data = self.load_json(path)
        if not isinstance(data, dict):
            if data is not None:
                self.add_error(f"{TEST_GATE_RESULT} must be a JSON object")
            return
        for branch in ("blackbox", "whitebox"):
            branch_data = data.get(branch)
            if not isinstance(branch_data, dict):
                self.add_error(f"{TEST_GATE_RESULT} missing {branch} object")
                continue
            self.validate_branch(branch, branch_data)

    def validate_test_gate(self) -> None:
        if self.test_gate not in {"on", "off"}:
            self.add_error(f"harness.test_gate must be 'on' or 'off', got {self.test_gate!r}")
            return
        if self.test_gate == "off":
            self.summary["test_gate"] = "skipped_by_config"
            return
        self.validate_provenance(BLACKBOX_PROVENANCE, "ascendc-st-design")
        self.validate_provenance(WHITEBOX_PROVENANCE, "ascendc-whitebox-design")
        self.validate_test_gate_result()
        if not self.errors:
            self.summary.update({"test_gate": "passed", "blackbox": "passed", "whitebox": "passed"})

    def print_result(self) -> int:
        LOGGER.info("test_gate: %s", self.test_gate)
        if self.summary:
            LOGGER.info("summary: %s", json.dumps(self.summary, ensure_ascii=False, sort_keys=True))
        if self.errors:
            LOGGER.info("errors:")
            for error in self.errors:
                LOGGER.info("- %s", error)
            LOGGER.info("STATUS: FAILED")
            return 1
        LOGGER.info("STATUS: PASSED")
        return 0

    def _safe_path(self, rel_path: str, label: str) -> Path | None:
        """Resolve rel_path under the operator dir; reject escapes / missing / empty."""
        if not isinstance(rel_path, str) or not rel_path.strip():
            self.add_error(f"{label} contains an invalid path: {rel_path!r}")
            return None
        rel_path = rel_path.strip()
        path = (self.operator_doc_dir / rel_path).resolve()
        try:
            path.relative_to(self.operator_doc_dir.resolve())
        except ValueError:
            self.add_error(f"{label} path escapes operator doc dir: {rel_path}")
            return None
        if not path.is_file():
            self.add_error(f"{label} path is missing: {rel_path}")
            return None
        if path.stat().st_size == 0:
            self.add_error(f"{label} path is empty: {rel_path}")
            return None
        return path

    def _rows(self, path: Path) -> list[dict]:
        """Return a list of record dicts from a .csv or .json file."""
        if path.suffix.lower() == ".csv":
            try:
                with path.open(newline="", encoding="utf-8") as fh:
                    return list(csv.DictReader(fh))
            except (csv.Error, OSError) as exc:
                self.add_error(f"invalid CSV in {path.name}: {exc}")
                return []
        data = self.load_json(path)
        if isinstance(data, dict):
            entries = data.get("cases", data.get("results", []))
        elif isinstance(data, list):
            entries = data
        else:
            entries = []
        if not isinstance(entries, list):
            self.add_error(f"{path.name} must contain a cases/results list")
            return []
        return [e for e in entries if isinstance(e, dict)]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-doc-dir", required=True, type=Path)
    parser.add_argument("--plugin-root", type=Path, default=Path(__file__).resolve().parents[3],
                        help="ops-direct-invoke-flash plugin root containing AGENTS.md")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(argv)
    validator = TestGateValidator(args.plugin_root.resolve(), args.operator_doc_dir.resolve())
    if not args.operator_doc_dir.is_dir():
        validator.add_error(f"operator doc directory does not exist: {args.operator_doc_dir.resolve()}")
    else:
        validator.validate_test_gate()
    return validator.print_result()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
