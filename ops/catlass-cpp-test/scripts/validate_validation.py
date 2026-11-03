# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate the CATLASS C++ validation record and its ordered acceptance gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SECTION_NAMES = (
    "02 标杆运行",
    "04 快速精度验证",
    "V1 全量验收",
    "V2 性能验收",
    "V3 流水分析",
    "V4 候选选择",
)
PASS_RE = re.compile(r"(?:通过|达标|PASS(?:ED)?)", re.IGNORECASE)
FAIL_RE = re.compile(r"(?:未通过|不通过|未达标|失败|FAIL(?:ED)?)", re.IGNORECASE)
RESULT_ONLY_RE = re.compile(
    r"(?:通过|达标|未通过|不通过|未达标|失败|PASS(?:ED)?|FAIL(?:ED)?)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(
    r"(?:待填写|待执行|待补充|待确认|\bTODO\b|\bTBD\b|<[^>\n]+>|\{[^}\n]+\})",
    re.IGNORECASE,
)
DURATION_RE = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(us|ms|µs|μs)?", re.IGNORECASE)
CANDIDATE_HEADERS = (
    "ID",
    "优化方案",
    "影响 Stage/Pipe",
    "局部收益依据",
    "预计整体收益(us)",
    "状态",
    "状态依据/失效条件",
)
CANDIDATE_STATES = {"active", "deferred", "selected", "accepted", "rejected"}
CANDIDATE_ID_RE = re.compile(r"C\d+")


def _section(text: str, name: str) -> str | None:
    pattern = re.compile(rf"(?ms)^##\s+{re.escape(name)}\s*$\n?(.*?)(?=^##\s+|\Z)")
    match = pattern.search(text)
    return match.group(1) if match else None


def _position(text: str, name: str) -> int | None:
    match = re.search(rf"(?m)^##\s+{re.escape(name)}\s*$", text)
    return match.start() if match else None


def _has_passed(value: str) -> bool:
    return bool(PASS_RE.fullmatch(value.strip().strip("`* ")))


def _has_failed(value: str) -> bool:
    return bool(FAIL_RE.fullmatch(value.strip().strip("`* ")))


def _duration_us(value: str) -> float | None:
    match = DURATION_RE.fullmatch(value.strip().strip("`* "))
    if match is None:
        return None
    number = float(match.group(1).replace(",", ""))
    if number < 0:
        return None
    unit = (match.group(2) or "us").lower()
    return number * 1000 if unit == "ms" else number


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(body: str) -> list[tuple[list[str], list[list[str]]]]:
    lines = body.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index + 1 < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        headers = _cells(lines[index])
        separator = _cells(lines[index + 1])
        if len(separator) != len(headers) or not all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in separator
        ):
            index += 1
            continue
        rows: list[list[str]] = []
        index += 2
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            rows.append(_cells(lines[index]))
            index += 1
        tables.append((headers, rows))
    return tables


def _candidate_pool_errors(body: str) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    match = next(
        (
            (headers, rows)
            for headers, rows in _tables(body)
            if all(header in headers for header in CANDIDATE_HEADERS)
        ),
        None,
    )
    if match is None:
        return ["V3 candidate pool must use the required candidate table"], []
    headers, rows = match
    if not rows:
        return ["V3 candidate pool must contain at least one candidate"], []

    indexes = {header: headers.index(header) for header in CANDIDATE_HEADERS}
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != len(headers):
            errors.append(f"V3 candidate row {row_number} has an invalid column count")
            continue
        values = {header: row[indexes[header]].strip() for header in CANDIDATE_HEADERS}
        unresolved = [
            header
            for header, value in values.items()
            if not value or PLACEHOLDER_RE.search(value)
        ]
        if unresolved:
            errors.append(
                f"V3 candidate row {row_number} contains unresolved "
                + ", ".join(unresolved)
            )

        candidate_id = values["ID"].strip("`* ")
        if CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
            errors.append(f"V3 candidate row {row_number} must use an ID such as C01")
        elif candidate_id in seen_ids:
            errors.append(f"V3 candidate ID must be unique: {candidate_id}")
        else:
            seen_ids.add(candidate_id)

        status = values["状态"].strip("`* ").lower()
        if status not in CANDIDATE_STATES:
            errors.append(
                f"V3 candidate row {row_number} has invalid status: {values['状态']}"
            )
        benefit_us = _duration_us(values["预计整体收益(us)"])
        if benefit_us is None:
            errors.append(
                f"V3 candidate row {row_number} must record numeric expected overall benefit"
            )
        elif status in {"active", "selected"} and benefit_us <= 0:
            errors.append(
                f"V3 {status} candidate {candidate_id or row_number} must have "
                "positive expected overall benefit"
            )
        elif status == "deferred" and benefit_us != 0:
            errors.append(
                f"V3 deferred candidate {candidate_id or row_number} must record "
                "0 us current overall benefit"
            )

        if (
            CANDIDATE_ID_RE.fullmatch(candidate_id) is not None
            and status in CANDIDATE_STATES
            and benefit_us is not None
            and not unresolved
        ):
            candidates.append(
                {
                    "id": candidate_id,
                    "status": status,
                    "benefit_us": benefit_us,
                }
            )
    return errors, candidates


def _field_value(body: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}[：:]\s*(.+?)\s*$", body)
    return match.group(1).strip() if match else None


def _candidate_selection_errors(
    body: str,
    candidates: list[dict[str, Any]],
    *,
    require_selection: bool,
) -> list[str]:
    errors: list[str] = []
    selection = _field_value(body, "本轮选择")
    no_selection = "本轮无可执行候选" in body
    selected_rows = [row for row in candidates if row["status"] == "selected"]

    if selection is not None and no_selection:
        errors.append(
            "V4 must not record both a selected candidate and no executable candidate"
        )
        return errors

    if selection is not None:
        selected_id = selection.strip("`* ")
        if (
            PLACEHOLDER_RE.search(selection)
            or CANDIDATE_ID_RE.fullmatch(selected_id) is None
        ):
            errors.append("V4 must record a valid selected candidate ID")
            return errors
        if len(selected_rows) != 1:
            errors.append(
                "V4 selection requires exactly one V3 candidate with selected status"
            )
        selected = next(
            (candidate for candidate in candidates if candidate["id"] == selected_id),
            None,
        )
        if selected is None:
            errors.append(
                f"V4 selected candidate is missing from the V3 pool: {selected_id}"
            )
        elif selected["status"] != "selected":
            errors.append(
                f"V4 selected candidate must have selected status: {selected_id}"
            )
        else:
            eligible = [
                candidate
                for candidate in candidates
                if candidate["status"] in {"active", "selected"}
            ]
            max_benefit = max(
                (candidate["benefit_us"] for candidate in eligible), default=0
            )
            if selected["benefit_us"] < max_benefit:
                errors.append(
                    "V4 must select the candidate with the largest expected overall benefit"
                )
        for label in ("排序依据", "对应 baseline"):
            value = _field_value(body, label)
            if value is None or PLACEHOLDER_RE.search(value):
                errors.append(f"V4 selected candidate must record {label}")
        return errors

    if no_selection:
        if selected_rows:
            errors.append("V4 no-selection result must not leave a selected candidate")
        if any(candidate["status"] == "active" for candidate in candidates):
            errors.append(
                "V4 must select an active candidate before stopping the round"
            )
        reason = _field_value(body, "原因")
        if reason is None or PLACEHOLDER_RE.search(reason):
            errors.append("V4 no-selection result must record a reason")
        if require_selection:
            errors.append(
                "performance_candidate requires a selected optimization candidate"
            )
        return errors

    errors.append("V4 must record either 本轮选择 or 本轮无可执行候选")
    return errors


def _evidence_table_errors(
    body: str,
    label: str,
    required_headers: tuple[str, ...],
    *,
    numeric_headers: tuple[str, ...] = (),
    pass_headers: tuple[str, ...] = (),
    max_not_exceed: tuple[str, str] | None = None,
) -> tuple[list[str], str]:
    errors: list[str] = []
    match = next(
        (
            (headers, rows)
            for headers, rows in _tables(body)
            if all(header in headers for header in required_headers)
        ),
        None,
    )
    if match is None:
        return [f"{label} must use the required evidence table"], "invalid"
    headers, rows = match
    if not rows:
        return [
            f"{label} evidence table must contain at least one result row"
        ], "invalid"

    indexes = {header: headers.index(header) for header in required_headers}
    row_statuses: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != len(headers):
            errors.append(f"{label} row {row_number} has an invalid column count")
            row_statuses.append("invalid")
            continue
        conclusion = row[indexes["结论"]].strip()
        if not conclusion or PLACEHOLDER_RE.search(conclusion):
            row_statuses.append("pending")
            continue
        if _has_failed(conclusion):
            row_statuses.append("failed")
        elif _has_passed(conclusion):
            row_statuses.append("passed")
        else:
            row_statuses.append("invalid")
            errors.append(
                f"{label} row {row_number} must record a passed or failed conclusion"
            )
            continue
        for header, cell_index in indexes.items():
            value = row[cell_index].strip()
            if not value or PLACEHOLDER_RE.search(value):
                errors.append(
                    f"{label} row {row_number} contains unresolved {header} evidence"
                )
            elif header == "证据" and RESULT_ONLY_RE.fullmatch(value.strip("`* ")):
                errors.append(
                    f"{label} row {row_number} must record evidence, not only a result"
                )
        for header in numeric_headers:
            value = row[indexes[header]]
            if _duration_us(value) is None:
                errors.append(
                    f"{label} row {row_number} must record a numeric {header} value"
                )
        if row_statuses[-1] == "passed" and max_not_exceed is not None:
            value_header, limit_header = max_not_exceed
            value = _duration_us(row[indexes[value_header]])
            limit = _duration_us(row[indexes[limit_header]])
            if value is not None and limit is not None and value > limit:
                errors.append(
                    f"{label} row {row_number} {value_header} must not exceed "
                    f"{limit_header} for a passed result"
                )
        if row_statuses[-1] == "passed":
            for header in (*pass_headers, "结论"):
                if not _has_passed(row[indexes[header]]):
                    errors.append(
                        f"{label} row {row_number} must record {header} as passed"
                    )
    if "failed" in row_statuses:
        status = "failed"
    elif "invalid" in row_statuses:
        status = "invalid"
    elif row_statuses and all(status == "passed" for status in row_statuses):
        status = "passed"
    else:
        status = "pending"
    return errors, status


def _stage(workflow: dict[str, Any] | str) -> str:
    if isinstance(workflow, dict):
        return str(workflow.get("stage", ""))
    return workflow


def validate(
    text: str,
    workflow: dict[str, Any] | str = "validation",
    event: str | None = None,
) -> list[str]:
    errors: list[str] = []
    stage = _stage(workflow)
    if stage not in {"reference", "implementation", "validation", "complete"}:
        return [
            "validation record check requires stage=reference, implementation, validation, or complete"
        ]
    reference = _section(text, SECTION_NAMES[0])
    if reference is None:
        errors.append("validation record missing section: 02 标杆运行")
    else:
        reference_errors, reference_status = _evidence_table_errors(
            reference,
            "reference execution",
            ("范围", "标杆运行", "计算语义", "精度策略", "证据", "结论"),
            pass_headers=("标杆运行", "计算语义", "精度策略"),
        )
        errors.extend(reference_errors)
        if reference_status != "passed":
            errors.append("reference execution evidence must pass before design")

    if stage == "reference":
        return errors

    quick = _section(text, SECTION_NAMES[1])
    if quick is None:
        errors.append("validation record missing section: 04 快速精度验证")
    else:
        quick_errors, quick_status = _evidence_table_errors(
            quick,
            "quick precision",
            ("Stage", "设计校验", "构建/运行", "精度", "证据", "结论"),
            pass_headers=("设计校验", "构建/运行", "精度"),
        )
        errors.extend(quick_errors)
        if quick_status != "passed":
            errors.append("quick precision evidence must pass before validation")

    if stage == "implementation":
        return errors

    full = _section(text, SECTION_NAMES[2])
    performance = _section(text, SECTION_NAMES[3])
    positions = {name: _position(text, name) for name in SECTION_NAMES}
    document_order = sorted(
        (name for name in SECTION_NAMES if positions[name] is not None),
        key=lambda name: positions[name],
    )
    expected_order = [name for name in SECTION_NAMES if positions[name] is not None]
    if document_order != expected_order:
        errors.append(
            "validation record sections must follow 02, 04, V1, V2, V3, V4 order"
        )
    full_status = "invalid"
    coverage_status = "invalid"
    if full is None:
        errors.append("validation record missing section: V1 全量验收")
    else:
        coverage_errors, coverage_status = _evidence_table_errors(
            full,
            "V1 full-case coverage",
            ("覆盖项", "来源", "取值/分支", "工作流用例", "证据", "结论"),
        )
        errors.extend(coverage_errors)
        full_errors, full_status = _evidence_table_errors(
            full,
            "V1 full validation",
            ("范围", "功能", "精度", "边界", "异常", "证据", "结论"),
            pass_headers=("功能", "精度", "边界", "异常"),
        )
        errors.extend(full_errors)
        if full_status == "passed" and coverage_status != "passed":
            errors.append(
                "V1 full validation requires passed full-case coverage evidence"
            )

    performance_status = "invalid"
    if performance is None:
        errors.append("validation record missing section: V2 性能验收")
    else:
        if "BasicInfo Task Duration(us)" not in performance:
            errors.append("performance validation must use BasicInfo Task Duration(us)")
        performance_errors, performance_status = _evidence_table_errors(
            performance,
            "V2 performance validation",
            (
                "设备/输入/采样",
                "指标",
                "baseline",
                "candidate",
                "目标",
                "证据",
                "结论",
            ),
            numeric_headers=("baseline", "candidate", "目标"),
            max_not_exceed=("candidate", "目标"),
        )
        errors.extend(performance_errors)
        if performance_status not in {"pending", "invalid"}:
            for term, message in (
                ("普通 msprof", "performance validation must use plain msprof"),
                ("5次", "performance validation must sample 5 times"),
                ("中位数", "performance validation must use the median"),
            ):
                if term not in performance:
                    errors.append(message)

    if full_status != "passed" and performance_status not in {"pending", "invalid"}:
        errors.append("V1 full validation must pass before V2 performance validation")

    if stage == "complete":
        if full_status != "passed":
            errors.append("complete stage requires passed V1 evidence")
        if coverage_status != "passed":
            errors.append("complete stage requires passed full-case coverage evidence")
        if performance_status != "passed":
            errors.append("complete stage requires passed V2 evidence")

    if performance_status == "failed":
        analysis = _section(text, SECTION_NAMES[4])
        selection = _section(text, SECTION_NAMES[5])
        candidate_rows: list[dict[str, Any]] = []
        if (
            analysis is None
            or not any(term in analysis for term in ("瓶颈", "阻塞"))
            or PLACEHOLDER_RE.search(analysis)
        ):
            errors.append(
                "performance failure requires V3 pipeline bottleneck analysis"
            )
        if analysis is not None:
            candidate_errors, candidate_rows = _candidate_pool_errors(analysis)
            errors.extend(candidate_errors)
        if selection is None:
            errors.append("performance failure requires V4 candidate selection")
        else:
            errors.extend(
                _candidate_selection_errors(
                    selection,
                    candidate_rows,
                    require_selection=event == "performance_candidate",
                )
            )
    if event == "validation_accepted":
        if full_status != "passed" or coverage_status != "passed":
            errors.append("validation_accepted requires passed V1 evidence")
        if performance_status != "passed":
            errors.append("validation_accepted requires passed V2 evidence")
    elif event == "performance_candidate":
        if full_status != "passed" or coverage_status != "passed":
            errors.append("performance_candidate requires passed V1 evidence")
        if performance_status != "failed":
            errors.append("performance_candidate requires failed V2 evidence")
    return errors


def _load_workflow(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read workflow JSON: {path}: {exc}"]
    if not isinstance(data, dict):
        return None, ["workflow JSON must be an object"]
    return data, []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--event")
    args = parser.parse_args()
    errors: list[str] = []
    try:
        text = args.validation.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read validation record: {args.validation}: {exc}")
        text = ""
    workflow, workflow_errors = _load_workflow(args.workflow)
    errors.extend(workflow_errors)
    if workflow is not None:
        workflow_validator = _load(
            "catlass_workflow_validator",
            ROOT / "catlass-cpp-interface/scripts/validate_workflow.py",
        )
        errors.extend(workflow_validator.validate_workflow(workflow))
        errors.extend(validate(text, workflow, args.event))
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: validation record follows the CATLASS acceptance sequence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
