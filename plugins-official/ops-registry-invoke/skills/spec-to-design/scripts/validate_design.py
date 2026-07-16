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
"""Validate generated DESIGN.md and PLAN.md structure for spec-to-design."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESOURCES_DIR = Path(__file__).resolve().parents[3] / "workflow" / "resources"
if _RESOURCES_DIR.is_dir() and str(_RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_DIR))
from frontmatter_utils import parse_yaml_frontmatter
from assemble_design import extract_op_name, read_text
from _output_log import get_logger

_LOGGER = get_logger("ops_registry_invoke.validate_design")


RAW_TEMPLATE_SNIPPETS = [
    "本模板用于",
    "生成修订记录表",
    "用表格列出",
    "不得把未验证 API 写成",
    "{{",
    "}}",
    "{op}",
    "{operator_name}",
]

REQUIRED_DESIGN_TERMS = [
    "spec.yaml 一致性映射",
    "API 验证记录",
    "UB 容量验证",
    "Tiling",
    "交付件清单",
]

REQUIRED_PLAN_HEADINGS_BY_ITERATION = {
    1: [
        "## 修订记录",
        "## 全覆盖目标",
        "## 穿刺结果判定",
    ],
    2: [
        "## 修订记录",
        "## 迭代一穿刺列表",
        "## 迭代二整合与全覆盖目标",
        "## 穿刺结果判定",
    ],
    3: [
        "## 修订记录",
        "## 迭代一穿刺列表",
        "## 迭代二整合目标",
        "## 迭代二穿刺列表",
        "## 迭代三全覆盖目标",
        "## 穿刺结果判定",
    ],
}


def template_sections(template_text: str) -> list[str]:
    return [m.group(0).strip() for m in re.finditer(r"^## .+$", template_text, re.MULTILINE)]


def markdown_sections(markdown: str) -> list[str]:
    return [m.group(0).strip() for m in re.finditer(r"^## .+$", markdown, re.MULTILINE)]


def validate_fences(label: str, text: str, errors: list[str]) -> None:
    fence_count = sum(1 for line in text.splitlines() if line.startswith("```"))
    if fence_count % 2 != 0:
        errors.append(f"{label}: unclosed fenced code block")


def validate_design(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    spec_text = read_text(args.spec)
    template_text = read_text(args.template)
    design_text = read_text(args.design)
    op_name = extract_op_name(spec_text)

    expected_title = f"# {op_name} 设计文档"
    first_line = design_text.splitlines()[0].strip() if design_text.splitlines() else ""
    if first_line != expected_title:
        errors.append(f"expected title {expected_title!r}, found {first_line!r}")

    expected_sections = template_sections(template_text)
    actual_sections = markdown_sections(design_text)
    if actual_sections != expected_sections:
        missing = [section for section in expected_sections if section not in actual_sections]
        extra = [section for section in actual_sections if section not in expected_sections]
        if missing:
            errors.append("missing sections: " + ", ".join(missing))
        if extra:
            errors.append("unexpected sections: " + ", ".join(extra))
        if not missing and not extra:
            errors.append("sections are present but not in template order")

    for term in REQUIRED_DESIGN_TERMS:
        if term not in design_text:
            errors.append(f"DESIGN.md missing required term: {term}")

    for snippet in RAW_TEMPLATE_SNIPPETS:
        if snippet in design_text:
            errors.append(f"DESIGN.md raw template marker remains: {snippet!r}")

    validate_fences("DESIGN.md", design_text, errors)
    if args.design.name != "DESIGN.md":
        errors.append("design file must be named DESIGN.md")

    return errors


def _validate_plan_iterations(iterations: object, iteration_count: object, errors: list[str]) -> None:
    """Check the frontmatter `iterations` list shape and required per-item fields."""
    if iterations is None:
        errors.append("PLAN.md frontmatter missing iterations field")
        return
    if not (isinstance(iterations, list) and iteration_count is not None):
        return
    if len(iterations) != iteration_count:
        errors.append(
            f"PLAN.md iterations list length ({len(iterations)}) "
            f"!= iteration_count ({iteration_count})"
        )
    for idx, iteration in enumerate(iterations):
        if not isinstance(iteration, dict):
            errors.append(f"PLAN.md iterations[{idx}] is not an object")
            continue
        for field in ("id", "goal", "wave1", "wave2", "acceptance"):
            if field not in iteration:
                errors.append(f"PLAN.md iterations[{idx}] missing field: {field}")


def _validate_plan_frontmatter(frontmatter: dict, errors: list[str]) -> None:
    """Validate the PLAN.md YAML frontmatter (iteration_count and iterations)."""
    if not frontmatter:
        errors.append("PLAN.md missing YAML frontmatter")
        return
    iteration_count = frontmatter.get("iteration_count")
    if iteration_count is None:
        errors.append("PLAN.md frontmatter missing iteration_count field")
    elif iteration_count not in (1, 2, 3):
        errors.append(f"PLAN.md iteration_count must be 1, 2, or 3, got {iteration_count!r}")
    _validate_plan_iterations(frontmatter.get("iterations"), iteration_count, errors)


def _normalize_iteration_count(frontmatter: dict) -> int:
    """Coerce the frontmatter iteration_count into an int, defaulting to 3."""
    iteration_count = frontmatter.get("iteration_count", 3)
    if isinstance(iteration_count, str):
        return int(iteration_count) if iteration_count.isdigit() else 3
    return iteration_count


def _validate_plan_body(body: str, op_name: str, frontmatter: dict, errors: list[str]) -> None:
    """Validate the PLAN.md body: title, required headings, markers, and fences."""
    expected_title = f"# {op_name} 迭代执行计划"
    first_line = body.splitlines()[0].strip() if body.splitlines() else ""
    if first_line != expected_title:
        errors.append(f"PLAN.md expected title {expected_title!r}, found {first_line!r}")

    iteration_count = _normalize_iteration_count(frontmatter)
    required_headings = REQUIRED_PLAN_HEADINGS_BY_ITERATION.get(
        iteration_count, REQUIRED_PLAN_HEADINGS_BY_ITERATION[3]
    )
    for heading in required_headings:
        if heading not in body:
            errors.append(f"PLAN.md missing heading: {heading}")
    for snippet in RAW_TEMPLATE_SNIPPETS:
        if snippet in body:
            errors.append(f"PLAN.md raw template marker remains: {snippet!r}")
    validate_fences("PLAN.md", body, errors)


def validate_plan(plan_path: Path, op_name: str) -> list[str]:
    errors: list[str] = []
    plan_text = read_text(plan_path)
    frontmatter, body = parse_yaml_frontmatter(plan_text)

    _validate_plan_frontmatter(frontmatter, errors)
    _validate_plan_body(body, op_name, frontmatter, errors)
    if plan_path.name != "PLAN.md":
        errors.append("plan file must be named PLAN.md")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()

    errors = validate_design(args)
    op_name = extract_op_name(read_text(args.spec))
    if args.plan:
        errors.extend(validate_plan(args.plan, op_name))

    if errors:
        for error in errors:
            _LOGGER.info(f"ERROR: {error}")
        return 1

    _LOGGER.info(f"OK: {args.design}")
    if args.plan:
        _LOGGER.info(f"OK: {args.plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
