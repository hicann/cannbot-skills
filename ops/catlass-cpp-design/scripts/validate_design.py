# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate the required structure of a CATLASS C++ detailed design."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SECTIONS = (
    "目标",
    "变量与任务映射",
    "Stage 总览",
    "Stage 详细设计",
    "用户额外 workspace",
    "跨 Stage 同步契约",
)
TARGET_ARCHITECTURES = ("atlas_a2_a3", "ascend950")
ARCHITECTURE_PRODUCTS = {
    "atlas_a2_a3": ("Atlas A2系列产品", "Atlas A3系列产品"),
    "ascend950": ("Ascend 950PR&950DT系列产品",),
}
ARCHITECTURE_STAGE_RULES = {
    "atlas_a2_a3": tuple(f"R{index:02d}" for index in range(1, 10)),
    "ascend950": tuple(f"R{index:02d}" for index in range(1, 11)),
}
STAGE_RULE_DECISIONS = {"已应用", "不适用"}
REQUIRED_STAGE_SPACE_SECTIONS = (
    "本 Stage 产生/写出",
    "其它 Stage 提供、当前 Stage 读取或需保护",
    "本 Stage 临时/固定",
)
PLACEHOLDERS = (
    "{operator_name}",
    "{stage_ready}",
    "{stage_ready_or_free}",
    "{input_ready_or_free}",
    "{MTE2_or_MTE1_transfer}",
    "{Cube_or_Vector_compute}",
    "{Fixpipe_or_MTE3_write}",
    "{previous_stage_ready}",
    "{load_and_compute}",
    "{write_shared_record}",
    "{target_architecture}",
    "{not_applicable_reason}",
    "{official_product_series}",
    "{stage_rule_id}",
    "{stage_rule_decision}",
    "{stage_rule_evidence_or_inapplicable_reason}",
    "TODO",
    "TBD",
    "待补充",
    "待填写",
)
SYNC_TERMS = (
    r"wait|等待",
    r"ready|就绪",
    r"free|释放",
    r"set|设置",
    r"event|flag|Mutex|互斥",
)
STAGE_HEADING = r"S(\d+)(?:\b|\s*[·:：])"
STORAGE_ORDER = ("UB", "L1", "L0A", "L0B", "L0C", "GM")
L2_CACHE_VALUES = {"使能", "不使能", "不涉及"}
DURATION_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ns|us|[μµ]s|ms|s|纳秒|微秒|毫秒|秒)\b",
    re.IGNORECASE,
)


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Section:
    level: int
    title: str
    line: int
    end: int
    body: str


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]


def markdown_parts(
    text: str,
) -> tuple[list[tuple[int, str, int]], list[str], str, bool, str]:
    """Read headings and fenced code separately, preserving section boundaries."""
    headings, blocks, prose, visible = [], [], [], []
    fence = ""
    in_comment = False
    code: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not fence:
            if in_comment:
                end = line.find("-->")
                if end < 0:
                    visible.append("")
                    prose.append("")
                    continue
                comment_end = end + 3
                line, in_comment = line[comment_end:], False
            while "<!--" in line:
                prefix, comment = line.split("<!--", 1)
                if "-->" in comment:
                    line = prefix + comment.split("-->", 1)[1]
                else:
                    line, in_comment = prefix, True
                    break
        visible.append(line)
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                blocks.append("\n".join(code))
                code, fence = [], ""
            else:
                code.append(line)
            prose.append("")
        elif marker:
            fence = marker[1]
            prose.append("")
        else:
            prose.append(line)
            heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                title = re.sub(r"^\d+(?:\.(?:\d+|N))*[.、]?\s+", "", heading[2])
                headings.append((len(heading[1]), title, line_number))
    return headings, blocks, "\n".join(prose), bool(fence), "\n".join(visible)


def sections(text: str) -> list[Section]:
    headings, _, _, _, visible = markdown_parts(text)
    lines = visible.splitlines()
    result = []
    for index, (level, title, line) in enumerate(headings):
        next_index = index + 1
        end = next(
            (item[2] for item in headings[next_index:] if item[0] <= level),
            len(lines) + 1,
        )
        content_end = end - 1
        result.append(
            Section(level, title, line, end, "\n".join(lines[line:content_end]))
        )
    return result


def _cells(line: str) -> list[str]:
    return [
        cell.strip().strip("`*").strip()
        for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
    ]


def markdown_tables(text: str) -> list[Table]:
    """Parse visible Markdown tables without treating fenced examples as evidence."""
    _, _, prose, _, _ = markdown_parts(text)
    lines = prose.splitlines()
    result: list[Table] = []
    index = 0
    while index + 1 < len(lines):
        if "|" not in lines[index] or "|" not in lines[index + 1]:
            index += 1
            continue
        headers = _cells(lines[index])
        separators = _cells(lines[index + 1])
        if (
            len(headers) < 2
            or len(separators) != len(headers)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators)
        ):
            index += 1
            continue
        rows: list[list[str]] = []
        cursor = index + 2
        while cursor < len(lines) and "|" in lines[cursor]:
            row = _cells(lines[cursor])
            if len(row) != len(headers):
                break
            rows.append(row)
            cursor += 1
        result.append(Table(headers, rows))
        index = cursor
    return result


def _filled(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {
        "",
        "-",
        "—",
        "/",
        "n/a",
        "none",
        "待补充",
        "待填写",
        "todo",
        "tbd",
    }


def design_title(text: str) -> tuple[str | None, list[str]]:
    titles = [title for level, title, _ in markdown_parts(text)[0] if level == 1]
    if len(titles) != 1:
        return None, ["design must contain exactly one document title"]
    title = titles[0]
    products = tuple(
        product for values in ARCHITECTURE_PRODUCTS.values() for product in values
    )
    if not title.endswith("设计方案") or not any(
        product in title for product in products
    ):
        return title, [
            "design title must contain an official product series and end with 设计方案"
        ]
    return title, []


def target_architecture_from_title(title: str | None) -> str | None:
    if title is None:
        return None
    matches = [
        architecture
        for architecture, products in ARCHITECTURE_PRODUCTS.items()
        if any(product in title for product in products)
    ]
    return matches[0] if len(matches) == 1 else None


def workflow_alignment_errors(text: str, workflow: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if workflow.get("stage") not in (
        "design",
        "implementation",
        "validation",
        "complete",
    ):
        errors.append(
            "design validation requires workflow stage=design or a later stage"
        )
    title, _ = design_title(text)
    workflow_architecture = workflow.get("target_architecture")
    if workflow_architecture not in TARGET_ARCHITECTURES:
        errors.append("workflow target_architecture must be frozen before design")
    elif title is not None and not any(
        product in title for product in ARCHITECTURE_PRODUCTS[workflow_architecture]
    ):
        errors.append(
            "design title product series does not match workflow target_architecture"
        )
    return errors


def _find_table(
    body: str, header_patterns: tuple[str, ...]
) -> tuple[Table | None, list[int]]:
    for table in markdown_tables(body):
        indexes: list[int] = []
        for pattern in header_patterns:
            match = next(
                (
                    index
                    for index, header in enumerate(table.headers)
                    if re.search(pattern, header, re.IGNORECASE)
                ),
                None,
            )
            if match is None:
                break
            indexes.append(match)
        else:
            return table, indexes
    return None, []


def table_errors(body: str, label: str, header_patterns: tuple[str, ...]) -> list[str]:
    table, indexes = _find_table(body, header_patterns)
    if table is None:
        return [f"missing structured table: {label}"]
    if not any(all(_filled(row[index]) for index in indexes) for row in table.rows):
        return [f"structured table has no complete data row: {label}"]
    return []


def _duration_us(value: str) -> float | None:
    match = DURATION_RE.search(value.replace("`", ""))
    if not match:
        return None
    number = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit == "ns" or unit == "纳秒":
        return number / 1000.0
    if unit in {"ms", "毫秒"}:
        return number * 1000.0
    if unit in {"s", "秒"}:
        return number * 1_000_000.0
    return number


def theoretical_performance_errors(body: str) -> list[str]:
    table, indexes = _find_table(
        body,
        (
            r"^约束来源$",
            r"工作量|数据量",
            r"峰值|带宽",
            r"理论最低耗时",
            r"重叠|依赖",
            r"^证据$",
        ),
    )
    if table is None:
        return ["missing structured table: theoretical performance upper bound"]

    errors: list[str] = []
    source_index, _, _, duration_index, _, evidence_index = indexes
    required_sources = {
        "AIC/Cube": r"AIC|Cube",
        "AIV/Vector": r"AIV|Vector",
        "GM/MTE": r"GM|MTE",
        "dependency": r"依赖",
    }
    for name, pattern in required_sources.items():
        if not any(
            re.search(pattern, row[source_index], re.IGNORECASE) for row in table.rows
        ):
            errors.append(f"theoretical performance table missing bound: {name}")
    for row in table.rows:
        if not _filled(row[duration_index]):
            errors.append("theoretical performance row missing lower-bound duration")
        elif _duration_us(row[duration_index]) is None:
            errors.append(
                "theoretical performance lower-bound duration must include a supported unit"
            )
        if not _filled(row[evidence_index]):
            errors.append("theoretical performance row missing evidence")

    lower_match = re.search(r"理论最低耗时\s*[:：]\s*`?([^`\n]+)", body, re.IGNORECASE)
    target_match = re.search(r"目标耗时\s*[:：]\s*`?([^`\n]+)", body, re.IGNORECASE)
    feasibility_match = re.search(
        r"理论可达性\s*[:：]\s*`?([^`\n]+)", body, re.IGNORECASE
    )
    if not lower_match or _duration_us(lower_match.group(1)) is None:
        errors.append(
            "theoretical performance summary must provide numeric theoretical minimum duration"
        )
    if not target_match or _duration_us(target_match.group(1)) is None:
        errors.append(
            "theoretical performance summary must provide numeric target duration"
        )
    if not feasibility_match or feasibility_match.group(1).strip() not in {
        "理论可达",
        "理论不可达",
    }:
        errors.append(
            "theoretical performance summary must state 理论可达 or 理论不可达"
        )
    if lower_match and target_match:
        lower = _duration_us(lower_match.group(1))
        target = _duration_us(target_match.group(1))
        if lower is not None and target is not None and target < lower:
            errors.append("target duration is below the theoretical minimum duration")
        if (
            lower is not None
            and target is not None
            and target >= lower
            and feasibility_match
            and feasibility_match.group(1).strip() == "理论不可达"
        ):
            errors.append(
                "theoretical feasibility contradicts the lower-bound comparison"
            )
    return errors


def stage_parallel_relation_errors(body: str) -> list[str]:
    table, indexes = _find_table(
        body,
        (r"^Stage$", r"^执行单元$", r"^主要动作$", r"并行.*流水", r"^结束条件$"),
    )
    if table is None:
        return ["missing structured table: Stage parallel and pipeline relation"]
    relation_index = indexes[3]
    errors: list[str] = []
    for row in table.rows:
        relation = row[relation_index]
        if not _filled(relation) or not re.search(r"并行|流水", relation):
            errors.append(
                "Stage parallel and pipeline relation must describe parallelism or overlap"
            )
        if re.search(r"正式.*串行|串行.*正式|逐\s*work.*串行", relation, re.IGNORECASE):
            errors.append("formal Stage design cannot use per-work serial execution")
    return errors


def synchronization_contract_errors(body: str) -> list[str]:
    headers = (
        r"^数据边$",
        r"^Producer$",
        r"^Consumer$",
        r"^ready$",
        r"free.*复用",
        r"本地\s*pipe",
        r"回绕.*退出",
    )
    table, indexes = _find_table(body, headers)
    if table is None:
        return ["missing structured table: synchronization producer-consumer closure"]
    errors: list[str] = []
    for row in table.rows:
        if not all(_filled(row[index]) for index in indexes):
            errors.append(
                "synchronization contract rows must close producer, consumer, "
                "ready, free, local pipe and drain"
            )
    return errors


def stage_rule_application_errors(
    body: str, target_architecture: str | None
) -> list[str]:
    if target_architecture not in ARCHITECTURE_STAGE_RULES:
        return []
    table, indexes = _find_table(body, (r"^规则$", r"^结论$", r"^设计证据/不适用原因$"))
    if table is None:
        return ["missing chip rule application table"]

    rule_index, decision_index, evidence_index = indexes
    expected = ARCHITECTURE_STAGE_RULES[target_architecture]
    counts = {rule: 0 for rule in expected}
    errors: list[str] = []
    for row in table.rows:
        rule = row[rule_index].strip().upper()
        if rule not in counts:
            errors.append(
                f"unexpected chip rule application decision: {rule or '<empty>'}"
            )
            continue
        counts[rule] += 1
        decision = row[decision_index].strip()
        evidence = row[evidence_index].strip()
        if decision not in STAGE_RULE_DECISIONS:
            errors.append(f"invalid chip rule application decision: {rule}")
        if not _filled(evidence):
            errors.append(f"missing evidence or inapplicable reason: {rule}")

    for rule, count in counts.items():
        if count == 0:
            errors.append(f"missing chip rule application decision: {rule}")
        elif count > 1:
            errors.append(f"duplicate chip rule application decision: {rule}")
    return errors


def stage_formula_errors(stage: Section) -> list[str]:
    """Require a formula/order code block before the Stage space subsections."""
    marker = re.search(r"(?m)^\*\*本 Stage 产生/写出\*\*\s*$", stage.body)
    prefix = stage.body[: marker.start()] if marker else stage.body
    _, blocks, _, unclosed, _ = markdown_parts(prefix)
    if unclosed or not any(block.strip() for block in blocks):
        return [
            f"Stage section near line {stage.line} missing formula block before space tables"
        ]
    return []


def stage_space_table_errors(section: Section, stage: Section) -> list[str]:
    headers = (r"^对象$", r"^存储位置$", r"^分配地址$", r"^L2Cache$", r"^功能$")
    table, indexes = _find_table(section.body, headers)
    label = f"{stage.title}/{section.title}"
    if table is None:
        return [f"missing Stage space table: {label}"]
    if not any(all(_filled(row[index]) for index in indexes) for row in table.rows):
        return [f"Stage space table has no complete data row: {label}"]

    object_index = indexes[0]
    storage_index = indexes[1]
    address_index = indexes[2]
    l2_cache_index = indexes[3]
    errors: list[str] = []
    positions: list[int] = []
    for row in table.rows:
        storage = row[storage_index].upper().replace(" ", "")
        matched = [index for index, name in enumerate(STORAGE_ORDER) if name in storage]
        if not matched:
            if not re.fullmatch(r"(?:不适用|N/?A)", storage, re.IGNORECASE):
                errors.append(
                    f"Stage space table has unsupported storage location: {label}"
                )
        elif len(matched) > 1:
            errors.append(
                "Stage space table must use one storage location per row: "
                f"{label}/{row[object_index]}"
            )
        else:
            positions.append(matched[0])

        address = row[address_index].replace("`", "")
        l2_cache = row[l2_cache_index].strip().strip("`* ")
        if l2_cache not in L2_CACHE_VALUES:
            errors.append(
                "Stage space table L2Cache must be one of 使能、不使能、不涉及: "
                f"{label}/{row[object_index]}"
            )
        address_is_unused = re.fullmatch(
            r"(?:不适用|N/?A)", address.strip(), re.IGNORECASE
        )
        size_is_explicit = re.search(
            r"(?:\[[^\]]+?\)|\d+(?:\.\d+)?)\s*(?:KiB|MiB|GiB|KB|MB|GB|B)\b",
            address,
            re.IGNORECASE,
        )
        if not address_is_unused and not size_is_explicit:
            errors.append(
                "Stage space table allocation address must expose size with B/KiB/MiB: "
                f"{label}/{row[object_index]}"
            )
        row_text = " | ".join(row)
        if re.search(r"组件\s*(?:bank|分配|管理)", row_text, re.IGNORECASE):
            errors.append(
                "Stage space table must replace vague component-managed buffer wording "
                "with explicit buffering mode, addresses, count, and size: "
                f"{label}/{row[object_index]}"
            )
        uses_l0 = any(name in storage for name in ("L0A", "L0B", "L0C"))
        buffering_mode_is_explicit = re.search(
            r"(?:单缓冲|双缓冲|ping\s*/\s*pong|(?:\d+|N)\s*(?:份|slot))",
            row_text,
            re.IGNORECASE,
        )
        if uses_l0 and not address_is_unused and not buffering_mode_is_explicit:
            errors.append(
                "Stage L0 space row must state single buffering, ping/pong, or an "
                f"explicit slot count: {label}/{row[object_index]}"
            )
        if uses_l0 and not address_is_unused:
            l0_storage = next(name for name in ("L0A", "L0B", "L0C") if name in storage)
            intervals = re.findall(r"\[[^\]]+?\)", address)
            qualified_intervals = re.findall(
                rf"\b{l0_storage}\s*\[[^\]]+?\)", address, re.IGNORECASE
            )
            if (
                re.search(r"相对基址\s*0", address)
                or not intervals
                or len(qualified_intervals) != len(intervals)
            ):
                errors.append(
                    "Stage L0 allocation address must qualify every interval with "
                    f"its storage space ({l0_storage}): {label}/{row[object_index]}"
                )
        if re.search(r"(?:API\s*tensor|work\s*offset)", row_text, re.IGNORECASE):
            errors.append(
                "Stage API input/output row must replace placeholder addressing with "
                f"tensor base, offset, and transfer size: {label}/{row[object_index]}"
            )
        is_api_io = re.search(r"API\s*(?:输入|输出)", storage, re.IGNORECASE)
        api_offset_is_explicit = re.search(
            r"(?:\+|offset|偏移)", address, re.IGNORECASE
        )
        if is_api_io and not api_offset_is_explicit:
            errors.append(
                "Stage API input/output address must expose tensor base and offset: "
                f"{label}/{row[object_index]}"
            )
    if positions != sorted(positions):
        errors.append(
            f"Stage space table storage order must be UB,L1,L0A,L0B,L0C,GM: {label}"
        )
    return errors


def child_section(
    document: list[Section], parent: Section | None, title: str
) -> Section | None:
    if parent is None:
        return None
    matches = [
        section
        for section in document
        if section.level == parent.level + 1
        and parent.line < section.line < parent.end
        and section.title == title
    ]
    return matches[0] if len(matches) == 1 else None


def stage_space_sections(stage: Section) -> tuple[dict[str, Section], list[str]]:
    visible = markdown_parts(stage.body)[4]
    matches = list(re.finditer(r"(?m)^\*\*(.+?)\*\*\s*$", visible))
    titles = [match.group(1).strip() for match in matches]
    if titles != list(REQUIRED_STAGE_SPACE_SECTIONS):
        return {}, [
            f"Stage section must contain exactly the three bold space subsections in order: {stage.title}"
        ]
    found: dict[str, Section] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(visible)
        line = stage.line + visible[: match.start()].count("\n") + 1
        title = titles[index]
        found[title] = Section(0, title, line, stage.end, visible[start:end])
    return found, []


def validate(text: str) -> list[str]:
    errors: list[str] = []
    text = markdown_parts(text)[4]
    title, title_errors = design_title(text)
    errors.extend(title_errors)
    target_architecture = target_architecture_from_title(title)
    document = sections(text)
    top_level = [section.title for section in document if section.level == 2]
    if top_level != list(REQUIRED_SECTIONS):
        errors.append("design must contain exactly the six required chapters in order")
    chapters: dict[str, Section] = {}
    for title in REQUIRED_SECTIONS:
        matches = [
            section
            for section in document
            if section.level == 2 and section.title == title
        ]
        if len(matches) != 1:
            errors.append(f"expected one required section: {title}")
        else:
            chapters[title] = matches[0]
            if not any(
                line.strip() and not re.match(r"^\s*#", line)
                for line in matches[0].body.splitlines()
            ):
                errors.append(f"empty required section: {title}")

    objective = chapters.get("目标")
    if objective:
        data_types = child_section(document, objective, "数据类型")
        if data_types is None:
            errors.append("missing objective subsection: 数据类型")
        else:
            errors.extend(
                table_errors(
                    data_types.body,
                    "data types",
                    (r"^范围$", r"^数据$", r"^数据类型$"),
                )
            )
        for term in (r"主模型\s*case", r"性能目标"):
            if not re.search(term, objective.body, re.IGNORECASE):
                errors.append(f"target section missing: {term}")
        theory = child_section(document, objective, "理论性能上限分析")
        if theory is None:
            errors.append("missing objective subsection: 理论性能上限分析")
        else:
            errors.extend(theoretical_performance_errors(theory.body))

    variables = chapters.get("变量与任务映射")
    if variables:
        errors.extend(
            table_errors(
                variables.body,
                "global symbols",
                (r"^符号$", r"^含义$"),
            )
        )
        _, mapping_blocks, _, _, _ = markdown_parts(variables.body)
        if not any(block.strip() for block in mapping_blocks):
            errors.append("task mapping must contain a formula or pseudocode block")

    overview = chapters.get("Stage 总览")
    summary_table: Table | None = None
    summary_indexes: list[int] = []
    if overview:
        summary_headers = (
            r"^Stage$",
            r"^执行单元$",
            r"^主要动作$",
            r"并行.*流水",
            r"^结束条件$",
        )
        errors.extend(table_errors(overview.body, "Stage summary", summary_headers))
        summary_table, summary_indexes = _find_table(overview.body, summary_headers)
        errors.extend(stage_parallel_relation_errors(overview.body))
        errors.extend(stage_rule_application_errors(overview.body, target_architecture))

    stage_chapter = chapters.get("Stage 详细设计")
    children = [
        section
        for section in document
        if stage_chapter and stage_chapter.line < section.line < stage_chapter.end
    ]
    stages = [
        section
        for section in children
        if section.level == 3 and re.match(STAGE_HEADING, section.title)
    ]
    if not stages:
        errors.append("at least one detailed Stage section is required")
    stage_ids = [int(re.match(STAGE_HEADING, stage.title).group(1)) for stage in stages]
    if stage_ids and stage_ids != list(range(len(stage_ids))):
        errors.append(
            "detailed Stage sections must be ordered and numbered continuously from S0"
        )
    for stage in stages:
        errors.extend(stage_formula_errors(stage))
        stage_spaces, detail_errors = stage_space_sections(stage)
        errors.extend(detail_errors)
        for space in stage_spaces.values():
            errors.extend(stage_space_table_errors(space, stage))
        if summary_table is not None:
            number = re.match(STAGE_HEADING, stage.title)
            if number and not any(
                re.search(rf"\bS{number.group(1)}\b", row[summary_indexes[0]])
                for row in summary_table.rows
            ):
                errors.append(f"Stage summary missing row for S{number.group(1)}")

    workspace = chapters.get("用户额外 workspace")
    if workspace:
        zero_workspace = bool(
            re.search(
                r"workspace(?:_size)?\s*=\s*0(?:\D|$)", workspace.body, re.IGNORECASE
            )
        )
        if zero_workspace:
            has_workspace_table = bool(markdown_tables(workspace.body))
            has_nonzero_assignment = bool(
                re.search(
                    r"workspace(?:_size)?\s*=\s*(?!0(?:\D|$))\d",
                    workspace.body,
                    re.IGNORECASE,
                )
            )
            has_concrete_size = bool(
                re.search(r"\b\d+(?:\.\d+)?\s*MB\b", workspace.body, re.IGNORECASE)
            )
            if has_workspace_table or has_nonzero_assignment or has_concrete_size:
                errors.append(
                    "workspace_size=0 cannot be combined with workspace regions or nonzero totals"
                )
        else:
            errors.extend(
                table_errors(
                    workspace.body,
                    "workspace total",
                    (r"^区域$", r"数量.*单份大小", r"主(?:模型)?\s*case.*大小"),
                )
            )
            if not re.search(r"workspace(?:_size)?\s*=", workspace.body, re.IGNORECASE):
                errors.append("workspace total must define workspace or workspace_size")
            if not re.search(r"\b\d+(?:\.\d+)?\s*MB\b", workspace.body, re.IGNORECASE):
                errors.append(
                    "workspace total must provide a concrete model-case value in MB"
                )
            _, workspace_blocks, _, _, _ = markdown_parts(workspace.body)
            if re.search(
                r"system_workspace", "\n".join(workspace_blocks), re.IGNORECASE
            ):
                errors.append(
                    "workspace_size must only contain user extra workspace regions"
                )
        for term in (r"API\s*输入", r"最终输出", r"系统\s*workspace"):
            if not re.search(term, workspace.body, re.IGNORECASE):
                errors.append(f"workspace scope missing: {term}")

    sync_section = chapters.get("跨 Stage 同步契约")
    if sync_section:
        errors.extend(
            table_errors(
                sync_section.body,
                "synchronization contract",
                (
                    r"数据边",
                    r"Producer",
                    r"Consumer",
                    r"ready",
                    r"free.*复用",
                    r"本地\s*pipe",
                    r"回绕.*退出",
                ),
            )
        )
        errors.extend(synchronization_contract_errors(sync_section.body))
        _, blocks, _, _, _ = markdown_parts(sync_section.body)
        pseudocode = "\n".join(blocks)
        if not pseudocode.strip():
            errors.append(
                "synchronization contract must contain execution-order pseudocode"
            )
        elif not re.search(r"wait|等待", pseudocode, re.IGNORECASE):
            errors.append(
                "synchronization execution order must contain producer/consumer waits"
            )

    generated_placeholders = re.findall(
        r"\{(?:operator_name|official_product_series|target_[A-Za-z0-9_]*|catlass_[A-Za-z0-9_]*|"
        r"arch_tag|vector_[A-Za-z0-9_]*|cube_[A-Za-z0-9_]*|api_[A-Za-z0-9_]*|main_[A-Za-z0-9_]*|"
        r"performance_[A-Za-z0-9_]*|relay_[A-Za-z0-9_]*|symbol_\d+[A-Za-z0-9_]*|work_[A-Za-z0-9_]*|"
        r"tiling_[A-Za-z0-9_]*|supported_[A-Za-z0-9_]*|cg_[A-Za-z0-9_]*|tile_[A-Za-z0-9_]*|"
        r"buffer_[A-Za-z0-9_]*|selection_[A-Za-z0-9_]*|architecture_[A-Za-z0-9_]*|stage_\d+_[A-Za-z0-9_]*|"
        r"workspace_[A-Za-z0-9_]*|collective_[A-Za-z0-9_]*|sync_[A-Za-z0-9_]*|theoretical_[A-Za-z0-9_]*|"
        r"aic_[A-Za-z0-9_]*|aiv_[A-Za-z0-9_]*|gm_[A-Za-z0-9_]*|dependency_[A-Za-z0-9_]*|first_[A-Za-z0-9_]*|"
        r"stage_\d+_parallel_[A-Za-z0-9_]*|sync_[A-Za-z0-9_]*|"
        r"reused_[A-Za-z0-9_]*|protected_[A-Za-z0-9_]*|producer_[A-Za-z0-9_]*)\}",
        text,
    )
    for placeholder in (*PLACEHOLDERS, *generated_placeholders):
        if placeholder in text:
            errors.append(f"unresolved placeholder: {placeholder}")
    if markdown_parts(text)[3]:
        errors.append("unclosed fenced code block")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    args = parser.parse_args()
    if not args.design.is_file():
        print(f"ERROR: design file not found: {args.design}")
        return 1
    text = args.design.read_text(encoding="utf-8")
    errors = validate(text)
    if not args.workflow.is_file():
        errors.append(f"workflow file not found: {args.workflow}")
    else:
        try:
            workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
            if not isinstance(workflow, dict):
                errors.append("workflow JSON must be an object")
            else:
                workflow_validator = _load(
                    "catlass_workflow_validator",
                    ROOT / "catlass-cpp-interface/scripts/validate_workflow.py",
                )
                errors.extend(workflow_validator.validate_workflow(workflow))
                errors.extend(workflow_alignment_errors(text, workflow))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid workflow JSON: {exc}")
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: detailed design structure is complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
