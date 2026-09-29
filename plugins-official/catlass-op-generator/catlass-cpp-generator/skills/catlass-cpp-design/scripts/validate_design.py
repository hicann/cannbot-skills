# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate the required structure of a Linear Attention detailed design."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SECTIONS = (
    "目标与数学语义",
    "Stage 总览与完整详设",
)
DESIGN_RULE_VERSION = "V1"
TARGET_ARCHITECTURES = ("atlas_a2_a3", "ascend950")
REQUIRED_STAGE_SECTIONS = (
    "全局符号、任务域与模型 case 代入",
    "依赖图与 Stage 结果",
    "数据搬运汇总",
    "Stage 间同步方案",
    "Kernel 组件与调用",
    "Workspace 总量",
)
PLACEHOLDERS = (
    "{operator_name}", "{stage_ready}", "{stage_ready_or_free}", "{input_ready_or_free}",
    "{MTE2_or_MTE1_transfer}", "{Cube_or_Vector_compute}",
    "{Fixpipe_or_MTE3_write}", "{previous_stage_ready}",
    "{load_and_compute}", "{write_shared_record}", "{target_architecture}",
    "{not_applicable_reason}",
    "TODO", "TBD", "待补充", "待填写",
)
SYNC_TERMS = (r"wait|等待", r"ready|就绪", r"free|释放", r"set|设置", r"event|flag|Mutex|互斥")
SYNC_FLOW_TERMS = (
    r"initialize|初始化|allocate|分配",
    r"for\s|while\s|稳态|循环",
    r"slot|bank|ping|pong|回绕|复用",
    r"tail|partial|尾",
    r"empty|空任务|无效任务",
    r"drain|final|排空|最终",
    r"release\s+(?:event|flag|Mutex|resources?)|释放.*(?:event|flag|Mutex|资源)|资源释放",
)
STAGE_TERMS = (
    r"公式",
    r"shape",
    r"dtype",
    r"地址",
    r"生命周期",
    r"task|head|映射",
    r"执行顺序",
    r"tail|partial|varlen|空任务",
    r"同步|依赖",
)
ARCHITECTURE_TERMS = (
    r"依赖",
    r"生命周期",
    r"执行单元",
    r"AIC|AIV",
    r"映射",
)


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


def markdown_parts(text: str) -> tuple[list[tuple[int, str, int]], list[str], str, bool, str]:
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
                line, in_comment = line[end + 3:], False
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
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
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
        end = next((item[2] for item in headings[index + 1:] if item[0] <= level), len(lines) + 1)
        result.append(Section(level, title, line, end, "\n".join(lines[line:end - 1])))
    return result


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("`*").strip()
            for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


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
        if (len(headers) < 2 or len(separators) != len(headers)
                or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators)):
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
    return normalized not in {"", "-", "—", "/", "n/a", "none", "待补充", "待填写", "todo", "tbd"}


def target_architecture(text: str) -> tuple[str | None, list[str]]:
    match = re.search(r"`target_architecture`\s*:\s*`([^`]+)`", text)
    if match is None:
        return None, ["missing target_architecture"]
    value = match.group(1).strip()
    if value not in TARGET_ARCHITECTURES:
        return value, [
            "target_architecture must be one of: " + ", ".join(TARGET_ARCHITECTURES)
        ]
    return value, []


def _field_value(body: str, label: str) -> str | None:
    match = re.search(rf"(?mi)^\s*-\s*{label}\s*[：:]\s*(.*?)\s*$", body)
    if match is None or not _filled(match.group(1)):
        return None
    return match.group(1).strip()


def _ordered(value: str, terms: tuple[str, ...]) -> bool:
    cursor = 0
    for term in terms:
        match = re.search(term, value[cursor:], re.IGNORECASE)
        if match is None:
            return False
        cursor += match.end()
    return True


def _not_applicable_with_reason(value: str) -> bool:
    match = re.search(r"不适用\s*[：:]\s*(.+)", value)
    return bool(match and match.group(1).strip())


def architecture_errors(body: str, architecture: str | None) -> list[str]:
    errors: list[str] = []
    fields = {
        "soc": _field_value(body, r"目标\s*SoC"),
        "arch": _field_value(body, r"CATLASS\s*架构标识"),
        "vector": _field_value(body, r"Vector\s*执行模型"),
        "cube_to_vector": _field_value(body, r"Cube\s*(?:→|->)\s*Vector\s*数据路径"),
        "vector_to_cube": _field_value(body, r"Vector\s*(?:→|->)\s*Cube\s*数据路径"),
    }
    labels = {
        "soc": "目标 SoC",
        "arch": "CATLASS 架构标识",
        "vector": "Vector 执行模型",
        "cube_to_vector": "Cube→Vector 数据路径",
        "vector_to_cube": "Vector→Cube 数据路径",
    }
    for key, value in fields.items():
        if value is None:
            errors.append(f"missing architecture field: {labels[key]}")
    if architecture not in TARGET_ARCHITECTURES or any(value is None for value in fields.values()):
        return errors

    arch = fields["arch"] or ""
    vector = fields["vector"] or ""
    cube_to_vector = fields["cube_to_vector"] or ""
    vector_to_cube = fields["vector_to_cube"] or ""
    if architecture == "atlas_a2_a3":
        if not re.search(r"CATLASS_ARCH\s*=\s*2201", arch) or "Arch::AtlasA2" not in arch:
            errors.append("atlas_a2_a3 requires CATLASS_ARCH=2201 and Arch::AtlasA2")
        if re.search(r"3510|Arch::Ascend950", arch):
            errors.append("atlas_a2_a3 architecture field must not contain A5 identifiers")
        if "MemBase" not in vector and not _not_applicable_with_reason(vector):
            errors.append("A2/A3 Vector execution model must select MemBase or explain why it is not applicable")
        if (not _not_applicable_with_reason(cube_to_vector)
                and not _ordered(cube_to_vector, (r"L0C", r"GM", r"UB"))):
            errors.append("A2/A3 Cube→Vector path must be L0C -> GM -> UB or explain why it is not applicable")
        if (not _not_applicable_with_reason(vector_to_cube)
                and not _ordered(vector_to_cube, (r"UB", r"GM", r"L1"))):
            errors.append("A2/A3 Vector→Cube path must be UB -> GM workspace -> L1 or explain why it is not applicable")
    elif architecture == "ascend950":
        if not re.search(r"CATLASS_ARCH\s*=\s*3510", arch) or "Arch::Ascend950" not in arch:
            errors.append("ascend950 requires CATLASS_ARCH=3510 and Arch::Ascend950")
        if re.search(r"2201|Arch::AtlasA2", arch):
            errors.append("ascend950 architecture field must not contain A2/A3 identifiers")
        if (not re.search(r"MemBase|RegBase|VF", vector)
                and not _not_applicable_with_reason(vector)):
            errors.append("A5 Vector execution model must select MemBase or RegBase/VF, or explain why it is not applicable")
        if (not _not_applicable_with_reason(cube_to_vector)
                and not _ordered(cube_to_vector, (r"L0C", r"UB"))):
            errors.append("A5 Cube→Vector path must connect L0C to UB or explain why it is not applicable")
        if (not _not_applicable_with_reason(vector_to_cube)
                and not _ordered(vector_to_cube, (r"UB", r"L1"))):
            errors.append("A5 Vector→Cube path must connect UB to L1 or explain why it is not applicable")
    return errors


def workflow_alignment_errors(text: str, workflow: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if workflow.get("workflow_id") != "catlass-linear-attention-v1":
        errors.append("workflow_id must be catlass-linear-attention-v1")
    design_architecture, _ = target_architecture(text)
    workflow_architecture = workflow.get("target_architecture")
    if workflow_architecture not in TARGET_ARCHITECTURES:
        errors.append("workflow target_architecture must be frozen before design")
    elif design_architecture in TARGET_ARCHITECTURES and workflow_architecture != design_architecture:
        errors.append("design target_architecture does not match workflow target_architecture")
    return errors


def _find_table(body: str, header_patterns: tuple[str, ...]) -> tuple[Table | None, list[int]]:
    for table in markdown_tables(body):
        indexes: list[int] = []
        for pattern in header_patterns:
            match = next((index for index, header in enumerate(table.headers)
                          if re.search(pattern, header, re.IGNORECASE)), None)
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


def table_or_na_errors(body: str, label: str, header_patterns: tuple[str, ...], na_pattern: str) -> list[str]:
    if re.search(na_pattern, body, re.IGNORECASE):
        return []
    return table_errors(body, label, header_patterns)


def child_section(document: list[Section], parent: Section | None, title: str) -> Section | None:
    if parent is None:
        return None
    matches = [section for section in document
               if section.level == parent.level + 1
               and parent.line < section.line < parent.end
               and section.title == title]
    return matches[0] if len(matches) == 1 else None


def subsection_errors(document: list[Section], parent: Section | None,
                      titles: tuple[str, ...], group: str) -> tuple[dict[str, Section], list[str]]:
    found: dict[str, Section] = {}
    errors: list[str] = []
    for title in titles:
        section = child_section(document, parent, title)
        if section is None:
            errors.append(f"missing {group} subsection: {title}")
        else:
            found[title] = section
    return found, errors


def has_structured_address_map(body: str) -> bool:
    table, indexes = _find_table(
        body,
        (
            r"存储",
            r"绝对半开区间|地址区间",
            r"tensor|slot|bank",
            r"shape|dtype|layout",
            r"大小|对齐",
            r"首次写入",
            r"最后消费",
            r"释放|复用条件",
        ),
    )
    if table is not None and any(all(_filled(row[index]) for index in indexes) for row in table.rows):
        return True

    _, blocks, _, _, _ = markdown_parts(body)
    address_pattern = r"(?:L1|UB|L0A|L0B|L0C|GM|workspace)\s*\[[^,\n]+,[^\]\)\n]+[\]\)]"
    required = (r"KiB|MiB|字节|bytes?|大小", r"写入|搬入|生产", r"最后|末次|消费", r"释放|复用|保留至")
    return any(re.search(address_pattern, block, re.IGNORECASE)
               and all(re.search(term, block, re.IGNORECASE) for term in required)
               for block in blocks)


def validate(text: str) -> list[str]:
    errors: list[str] = []
    text = markdown_parts(text)[4]
    if not re.search(r"`workflow_id`\s*:\s*`catlass-linear-attention-v1`", text):
        errors.append("workflow_id must be catlass-linear-attention-v1")
    version = re.search(r"`design_rule_version`\s*:\s*`([^`]+)`", text)
    if version is None:
        errors.append("missing design_rule_version")
    elif version.group(1) != DESIGN_RULE_VERSION:
        errors.append(f"design_rule_version must be {DESIGN_RULE_VERSION}")
    architecture, architecture_metadata_errors = target_architecture(text)
    errors.extend(architecture_metadata_errors)
    if "docs/api.md" not in text:
        errors.append("design must reference docs/api.md as the interface contract")
    if "docs/validation.md" not in text:
        errors.append("design must route development evidence to docs/validation.md")

    document = sections(text)
    top_level = [section.title for section in document if section.level == 2]
    if top_level != list(REQUIRED_SECTIONS):
        errors.append("design must contain exactly the two required chapters in order")
    chapters: dict[str, Section] = {}
    for title in REQUIRED_SECTIONS:
        matches = [section for section in document if section.level == 2 and section.title == title]
        if len(matches) != 1:
            errors.append(f"expected one required section: {title}")
        else:
            chapters[title] = matches[0]
            if not any(line.strip() and not re.match(r"^\s*#", line)
                       for line in matches[0].body.splitlines()):
                errors.append(f"empty required section: {title}")

    stage_chapter = chapters.get(REQUIRED_SECTIONS[1])
    if stage_chapter:
        for term in ARCHITECTURE_TERMS:
            if not re.search(term, stage_chapter.body, re.IGNORECASE):
                errors.append(f"Stage overview missing architecture evidence: {term}")
        if "PipeBarrier" in stage_chapter.body:
            errors.append("synchronization design must use paired HardEvent/CrossCore notifications")
    children = [section for section in document if stage_chapter
                and stage_chapter.line < section.line < stage_chapter.end]
    stages = [section for section in children if section.level == 3
              and re.match(r"Stage\s+\d+(?:\b|[：:])", section.title)]
    if not stages:
        errors.append("at least one detailed Stage section is required")
    for stage in stages:
        for term in STAGE_TERMS:
            if not re.search(term, stage.body, re.IGNORECASE):
                errors.append(f"Stage section near line {stage.line} missing: {term}")
        if not has_structured_address_map(stage.body):
            errors.append(f"Stage section near line {stage.line} missing structured address map")

    objective_chapter = chapters.get(REQUIRED_SECTIONS[0])
    objective_sections, detail_errors = subsection_errors(
        document,
        objective_chapter,
        ("目标、融合范围与目标场景", "API 契约引用与内部映射", "完整数学语义与数值边界"),
        "objective",
    )
    errors.extend(detail_errors)
    objective = objective_sections.get("目标、融合范围与目标场景")
    if objective:
        errors.extend(architecture_errors(objective.body, architecture))
        errors.extend(table_or_na_errors(
            objective.body,
            "performance target cases",
            (r"模型\s*case", r"shape|属性", r"基线", r"目标值", r"测量|统计口径", r"可达性"),
            r"性能目标.*不适用|不适用\s*[：:].*(?:性能目标|性能基线)",
        ))
    math = objective_sections.get("完整数学语义与数值边界")
    if math:
        errors.extend(table_or_na_errors(
            math.body,
            "numerical risk mapping",
            (r"中间量|高风险", r"CPU.*标杆", r"实现运算序列", r"计算\s*dtype", r"输入|中间值范围", r"NaN|Inf", r"mask|cast|保护"),
            r"高风险运算.*不适用|不适用\s*[：:].*(?:高风险|数值风险)",
        ))

    stage_sections, detail_errors = subsection_errors(
        document, stage_chapter, REQUIRED_STAGE_SECTIONS, "Stage detail")
    errors.extend(detail_errors)
    symbols = stage_sections.get("全局符号、任务域与模型 case 代入")
    if symbols:
        errors.extend(table_errors(
            symbols.body,
            "global symbols",
            (r"符号", r"定义", r"单位|取值域", r"来源|约束"),
        ))
        errors.extend(table_errors(
            symbols.body,
            "model case scheduling",
            (r"模型\s*case", r"Nbase", r"CG", r"Nwork", r"blockDim", r"wave", r"每核任务", r"尾"),
        ))
        if not re.search(r"TilingKey", symbols.body, re.IGNORECASE):
            errors.append("global task domain must enumerate reachable TilingKey conditions")
    overview = stage_sections.get("依赖图与 Stage 结果")
    if overview:
        summary_headers = (
            r"Stage", r"执行单元|owner", r"功能|公式", r"前驱|可并行",
            r"输入来源", r"输出落点", r"任务映射|任务数", r"主要空间",
        )
        errors.extend(table_errors(
            overview.body,
            "Stage result summary",
            summary_headers,
        ))
        summary_table, summary_indexes = _find_table(overview.body, summary_headers)
        if summary_table is not None:
            for stage in stages:
                number = re.match(r"Stage\s+(\d+)", stage.title)
                if (number and not any(re.search(rf"\bStage\s+{number.group(1)}\b", row[summary_indexes[0]])
                                       for row in summary_table.rows)):
                    errors.append(f"Stage result summary missing row for Stage {number.group(1)}")
        errors.extend(table_errors(
            overview.body,
            "cross-Stage data",
            (r"数据", r"shape|dtype|layout", r"生产\s*Stage|owner", r"消费\s*Stage|最后消费者", r"存放位置|份数", r"释放|复用条件"),
        ))
    movement = stage_sections.get("数据搬运汇总")
    if movement:
        errors.extend(table_errors(
            movement.body,
            "data movement summary",
            (r"tensor", r"GM.*读|写次数", r"字节", r"GM.*L1|UB", r"驻留|复用", r"批量搬运|stride", r"L2.*Cache", r"最后消费者"),
        ))

    sync_section = stage_sections.get("Stage 间同步方案")
    if sync_section is None:
        errors.append("missing Stage synchronization section")
    else:
        errors.extend(table_errors(
            sync_section.body,
            "cross-Stage synchronization edges",
            (r"数据边", r"生产\s*Stage|owner|pipe", r"消费\s*Stage|owner|pipe", r"存放位置|slot", r"ready", r"free", r"空任务"),
        ))
        errors.extend(table_errors(
            sync_section.body,
            "synchronization resources",
            (r"同步资源", r"所属核|范围", r"类型|方向", r"保护的数据|slot", r"数量|初始状态", r"set|wait|lock|unlock", r"复用|释放条件", r"硬件上限"),
        ))
        _, blocks, _, _, _ = markdown_parts(sync_section.body)
        pseudocode = "\n".join(blocks)
        if not pseudocode.strip():
            errors.append("synchronization section must contain pseudocode")
        for term in SYNC_TERMS:
            if not re.search(term, pseudocode, re.IGNORECASE):
                errors.append(f"synchronization pseudocode missing: {term}")
        for term in SYNC_FLOW_TERMS:
            if not re.search(term, pseudocode, re.IGNORECASE):
                errors.append(f"synchronization pseudocode missing workflow coverage: {term}")

    workspace = stage_sections.get("Workspace 总量")
    stage_level_sections = [section for section in children if section.level == 3]
    if stage_level_sections and stage_level_sections[-1].title != "Workspace 总量":
        errors.append("Workspace 总量 must be the final Stage detail subsection")
    if workspace and not re.search(r"workspace_size\s*=\s*0", workspace.body, re.IGNORECASE):
        errors.extend(table_errors(
            workspace.body,
            "workspace total",
            (r"Workspace\s*区域", r"offset|索引公式", r"record|bank", r"单份大小|对齐", r"总量贡献", r"生产者|最后消费者", r"释放条件"),
        ))
        if not re.search(r"workspace_size\s*=", workspace.body, re.IGNORECASE):
            errors.append("workspace total must define workspace_size")

    for placeholder in (*PLACEHOLDERS, *re.findall(r"\{stage_\d+_name\}", text)):
        if placeholder in text:
            errors.append(f"unresolved placeholder: {placeholder}")
    if markdown_parts(text)[3]:
        errors.append("unclosed fenced code block")
    return errors


def validate_kernel(path: Path) -> list[str]:
    """Check generated device code for the workflow's synchronization contract."""
    if not path.is_file():
        return [f"kernel file not found: {path}"]
    text = path.read_text(encoding="utf-8")
    if "PipeBarrier" in text:
        return ["kernel synchronization must use paired HardEvent/CrossCore notifications"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--kernel", type=Path,
                        help="optional device kernel source to validate synchronization")
    args = parser.parse_args()
    if not args.design.is_file():
        print(f"ERROR: design file not found: {args.design}")
        return 1
    text = args.design.read_text(encoding="utf-8")
    errors = validate(text)
    if args.workflow:
        if not args.workflow.is_file():
            errors.append(f"workflow file not found: {args.workflow}")
        else:
            try:
                workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
                if not isinstance(workflow, dict):
                    errors.append("workflow JSON must be an object")
                else:
                    errors.extend(workflow_alignment_errors(text, workflow))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid workflow JSON: {exc}")
    if args.kernel:
        errors.extend(validate_kernel(args.kernel))
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: detailed design structure is complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
