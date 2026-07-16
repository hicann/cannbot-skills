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
"""Assemble generated DESIGN.md sections and PLAN.md for ops-registry-invoke."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESOURCES_DIR = Path(__file__).resolve().parents[3] / "workflow" / "resources"
if _RESOURCES_DIR.is_dir() and str(_RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_DIR))
from frontmatter_utils import parse_yaml_frontmatter
from _output_log import get_logger

_LOGGER = get_logger("ops_registry_invoke.assemble_design")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_op_name(spec_text: str) -> str:
    in_op = False
    for line in spec_text.splitlines():
        if line.startswith("op:"):
            in_op = True
            continue
        if in_op and line and not line.startswith((" ", "\t", "#")):
            break
        if in_op:
            match = re.match(r"\s+name:\s*[\"']?([^\"'#\s]+)", line)
            if match:
                return match.group(1)
    raise ValueError("could not find op.name in spec")


def template_order(template_text: str) -> list[str]:
    return [m.group(0).removeprefix("## ").strip() for m in re.finditer(r"^## .+$", template_text, re.MULTILINE)]


def extract_sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## .+$", markdown, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = match.group(0).removeprefix("## ").strip()
        body = markdown[start:end].strip()
        if title in sections:
            raise ValueError(f"duplicate section returned: {title}")
        sections[title] = body
    return sections


def is_plan_document(path: Path, text: str) -> bool:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return path.name == "05-plan.md" or (first.startswith("# ") and "迭代执行计划" in first)


def expected_section_filenames(sections_dir: Path) -> list[str]:
    """Discover the bundle filenames that the agent was expected to materialize.

    The slice step writes instruction bundles to a sibling `bundles/` directory
    and creates an empty `sections/`. Each bundle filename (e.g. `01-overview-
    contract.md`) is the filename the agent must write back into `sections/`.
    Returns an empty list if no bundles directory exists.
    """
    bundles_dir = sections_dir.parent / "bundles"
    if not bundles_dir.is_dir():
        return []
    return sorted(path.name for path in bundles_dir.iterdir() if path.suffix.lower() == ".md")


def collect_sections(sections_dir: Path) -> tuple[dict[str, str], str | None]:
    if not sections_dir.is_dir():
        raise ValueError(f"sections path is not a directory: {sections_dir}")

    merged: dict[str, str] = {}
    plan_text: str | None = None
    files = sorted(path for path in sections_dir.iterdir() if path.suffix.lower() == ".md")
    if not files:
        expected = expected_section_filenames(sections_dir)
        hint_lines = [
            f"sections directory is empty: {sections_dir}",
            "",
            "The slice step only writes instruction bundles to ../bundles/ and",
            "leaves sections/ empty. You must read each bundle and write the",
            "generated markdown back to sections/ BEFORE running assemble_design.py.",
        ]
        if expected:
            hint_lines.append("")
            hint_lines.append("Expected files (one per bundle, same filename):")
            for name in expected:
                hint_lines.append(f"  - {sections_dir / name}")
        raise ValueError("\n".join(hint_lines))

    expected = expected_section_filenames(sections_dir)
    if expected:
        present = {path.name for path in files}
        missing = [name for name in expected if name not in present]
        if missing:
            hint_lines = [
                f"sections directory is missing {len(missing)} of {len(expected)} expected file(s): {sections_dir}",
                "",
                "Generate markdown for the listed bundles before re-running:",
            ]
            for name in missing:
                hint_lines.append(f"  - {sections_dir / name}  (bundle: {sections_dir.parent / 'bundles' / name})")
            raise ValueError("\n".join(hint_lines))

    for path in files:
        text = read_text(path)
        if is_plan_document(path, text):
            plan_text = text.strip() + "\n"
            continue
        for title, body in extract_sections(text).items():
            if title in merged:
                raise ValueError(f"section {title!r} appears in more than one file")
            merged[title] = body
    return merged, plan_text


def first_dtype(spec_text: str) -> str:
    match = re.search(r"dtype_set:\s*\[([^\]]+)\]", spec_text)
    if not match:
        return "主要 dtype"
    return match.group(1).split(",")[0].strip()


def _strip_conditional_blocks(text: str, iteration_count: int) -> str:
    """Remove conditional blocks from template based on iteration_count.

    Blocks are delimited by paired markers:
        <!-- BEGIN <condition> --> ... <!-- END <condition> -->

    Supported conditions:
        iteration_count >= N  — keep block if iteration_count >= N
        iteration_count = N   — keep block if iteration_count == N
    """
    pattern = re.compile(
        r"<!--\s*BEGIN\s+(iteration_count\s*(?:>=|=)\s*\d+)\s*-->"
        r"(.*?)"
        r"<!--\s*END\s+\1\s*-->",
        re.DOTALL,
    )

    def should_keep(condition: str) -> bool:
        match = re.match(r"iteration_count\s*(>=|=)\s*(\d+)", condition.strip())
        if not match:
            return False
        op, value = match.group(1), int(match.group(2))
        if op == ">=":
            return iteration_count >= value
        return iteration_count == value

    def replacer(m: re.Match) -> str:
        condition = m.group(1)
        content = m.group(2)
        if should_keep(condition):
            return content.rstrip("\n")
        return ""

    return pattern.sub(replacer, text)


def render_default_plan(op_name: str, spec_text: str, plan_template: str | None) -> str:
    dtype = first_dtype(spec_text)
    if plan_template:
        frontmatter, body = parse_yaml_frontmatter(plan_template)
        iteration_count = frontmatter.get("iteration_count", 3)
        if isinstance(iteration_count, str):
            iteration_count = int(iteration_count) if iteration_count.isdigit() else 3
        text = body.replace("{{ op.name }}", op_name)
        text = text.replace("YYYY-MM-DD", "待填写")
        text = text.replace("从 spec 选择主要 dtype", dtype)
        text = _strip_conditional_blocks(text, iteration_count)
        if iteration_count == 1:
            text = text.replace("迭代{N}全覆盖目标", "全覆盖目标")
        elif iteration_count == 2:
            text = text.replace("迭代{N}全覆盖目标", "迭代二整合与全覆盖目标")
        elif iteration_count == 3:
            text = text.replace("迭代{N}全覆盖目标", "迭代三全覆盖目标")
        frontmatter_yaml = render_frontmatter_yaml(iteration_count, dtype)
        return frontmatter_yaml + text.rstrip() + "\n"
    return render_default_plan_no_template(op_name, dtype)


def _probe(task_name: str, tiling_key: str, dtype: str, memory_strategy: str) -> dict:
    """Build one穿刺 (probe) definition; centralizes the shared probe schema."""
    return {
        "task_name": task_name,
        "tiling_key": tiling_key,
        "dtype": dtype,
        "memory_strategy": memory_strategy,
    }


_PROBE_TILING1 = _probe("probe_tiling1", "TilingKey_1", "{dtype}", "single_buffer")


_ITERATION_TEMPLATES = {
    1: [
        {
            "goal": "全功能实现 + 全覆盖",
            "a1_main_scope": "全功能实现",
            "probes": [],
            "b_scope": "L0+L1 全覆盖",
            "a2_scope": "全覆盖 UT",
            "ut": "全覆盖且无回归",
            "st": "全部必需用例已执行",
        },
    ],
    2: [
        {
            "goal": "骨架搭建 + 穿刺验证",
            "a1_main_scope": "单 TilingKey 骨架",
            "probes": [_PROBE_TILING1],
            "b_scope": "L0 基础用例",
            "a2_scope": "核心路径 UT",
            "ut": "核心路径 UT 通过",
            "st": "L0 用例覆盖完整",
        },
        {
            "goal": "整合 + 全功能 + 全覆盖",
            "a1_main_scope": "整合穿刺结果 + 全功能实现",
            "probes": [],
            "b_scope": "L0+L1 全覆盖",
            "a2_scope": "全覆盖 UT",
            "ut": "全覆盖且无回归",
            "st": "全部必需用例已执行",
        },
    ],
    3: [
        {
            "goal": "骨架搭建",
            "a1_main_scope": "单 TilingKey 骨架",
            "probes": [_PROBE_TILING1],
            "b_scope": "L0 基础用例",
            "a2_scope": "核心路径 UT",
            "ut": "核心路径 UT 通过",
            "st": "L0 用例覆盖完整",
        },
        {
            "goal": "策略整合",
            "a1_main_scope": "整合迭代一穿刺结果 → 多 TilingKey 实现",
            "probes": [_probe("probe_full_dtype", "TilingKey_2", "float32", "double_buffer")],
            "b_scope": "C++ 多 shape 用例",
            "a2_scope": "Tiling 分支 UT 覆盖",
            "ut": "Tiling 分支 UT 覆盖达标",
            "st": "多 shape 用例通过",
        },
        {
            "goal": "全量覆盖",
            "a1_main_scope": "全功能实现",
            "probes": [],
            "b_scope": "C++ 全量用例",
            "a2_scope": "全覆盖 UT",
            "ut": "UT 全覆盖且无回归",
            "st": "全部必需用例已执行",
        },
    ],
}


def _render_iteration_yaml(iterations: list[dict], dtype: str) -> str:
    """Render a list of iteration definitions as YAML frontmatter content."""
    lines: list[str] = []
    for idx, it in enumerate(iterations):
        lines.append(f"  - id: {idx + 1}")
        lines.append(f'    goal: "{it["goal"]}"')
        lines.append("    wave1:")
        lines.append("      a1_main:")
        lines.append(f'        scope: "{it["a1_main_scope"].format(dtype=dtype)}"')
        if it["probes"]:
            lines.append("      a1_p:")
            for probe in it["probes"]:
                lines.append(f'        - task_name: "{probe["task_name"]}"')
                lines.append(f'          tiling_key: "{probe["tiling_key"]}"')
                lines.append(f'          dtype: "{probe["dtype"].format(dtype=dtype)}"')
                lines.append(f'          memory_strategy: "{probe["memory_strategy"]}"')
        else:
            lines.append("      a1_p: []")
        lines.append("      b:")
        lines.append(f'        scope: "{it["b_scope"].format(dtype=dtype)}"')
        lines.append("    wave2:")
        lines.append("      a2:")
        lines.append(f'        scope: "{it["a2_scope"].format(dtype=dtype)}"')
        lines.append("    acceptance:")
        lines.append(f'      ut: "{it["ut"]}"')
        lines.append(f'      st: "{it["st"]}"')
    return "\n".join(lines)


def render_frontmatter_yaml(iteration_count: int, dtype: str) -> str:
    """Generate YAML frontmatter for PLAN.md based on iteration_count."""
    templates = _ITERATION_TEMPLATES.get(iteration_count, _ITERATION_TEMPLATES[3])
    iterations_yaml = _render_iteration_yaml(templates, dtype)
    return f"---\niteration_count: {iteration_count}\niterations:\n{iterations_yaml}\n---\n\n"


def render_default_plan_no_template(op_name: str, dtype: str) -> str:
    """Generate default PLAN.md when no template is provided (backward compatible, iteration_count=3)."""
    frontmatter = render_frontmatter_yaml(3, dtype)
    body = f"""# {op_name} 迭代执行计划

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 基于 spec.yaml 生成初始迭代计划 | 待填写 | 待填写 |

## 迭代一穿刺列表

| 任务类型 | TilingKey | Dtype | Shape/场景 | Memory Strategy | 验收点 |
| --- | --- | --- | --- | --- | --- |
| 主线 | TilingKey_0 | {dtype} | 核心正常 shape | 待设计确认 | 编译通过 + L0 核心用例通过 |

## 迭代二整合目标

整合迭代一成功穿刺项，实现多 TilingKey 和单 dtype 多 shape 覆盖。

## 迭代二穿刺列表

| 任务类型 | 验证目标 | 输入规格 | dtype | 预期结论 |
| --- | --- | --- | --- | --- |
| 穿刺1 | 全 dtype 或关键边界 | 来自 spec.yaml | 来自 supported_combinations | 成功/风险/需修订 |

## 迭代三全覆盖目标

| 覆盖维度 | 内容 | 来源 | 说明 |
| --- | --- | --- | --- |
| 全 dtype | supported_combinations | spec.yaml | 覆盖所有输出 dtype |
| shape/broadcast | shape_rule/broadcast | spec.yaml | 覆盖正常、边界和广播 case |

## 穿刺结果判定

| 状态 | 判定标准 | 后续处理 |
| --- | --- | --- |
| 成功 | 编译通过，核心用例通过，设计假设成立 | 复用到主线 |
| 部分成功 | 部分 case 通过，存在明确限制 | 记录限制并调整计划 |
| 失败 | 编译失败、精度失败或资源预算不可行 | 归档问题并重做 |
"""
    return frontmatter + body


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--sections", type=Path)
    parser.add_argument("--work-dir", type=Path, help="directory created by slice_design_inputs.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, help="directory for DESIGN.md and PLAN.md")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--plan-template", type=Path)
    parser.add_argument("--operator", help="optional operator name sanity check")
    return parser


def load_manifest(work_dir: Path | None) -> dict:
    """Load manifest.json from work_dir; return {} when unavailable."""
    if not work_dir:
        return {}
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(read_text(manifest_path))


def _manifest_path(manifest: dict, key: str) -> Path | None:
    """Return manifest[key] as a Path when the key is present, else None."""
    value = manifest.get(key)
    return Path(value) if value else None


def resolve_paths(args: argparse.Namespace, manifest: dict, skill_dir: Path) -> dict:
    """Resolve every input/output path from CLI args, manifest, then defaults."""
    default_template = skill_dir / "templates" / "DESIGN.md.templ"

    output_path = args.output
    if output_path is None and args.output_dir:
        output_path = args.output_dir / "DESIGN.md"
    if output_path is None:
        output_path = _manifest_path(manifest, "suggested_design_output")

    plan_output_path = args.plan_output
    if plan_output_path is None and args.output_dir:
        plan_output_path = args.output_dir / "PLAN.md"
    if plan_output_path is None:
        plan_output_path = _manifest_path(manifest, "suggested_plan_output")

    return {
        "spec": args.spec or _manifest_path(manifest, "spec"),
        "template": args.template or _manifest_path(manifest, "template") or default_template,
        "sections": args.sections or (args.work_dir / "sections" if args.work_dir else None),
        "output": output_path,
        "plan_output": plan_output_path,
    }


def write_design(op_name: str, template_text: str, sections: dict[str, str], output_path: Path) -> None:
    """Order sections against the template and write DESIGN.md."""
    order = template_order(template_text)
    missing = [title for title in order if title not in sections]
    extra = [title for title in sections if title not in order]
    if missing:
        raise ValueError("missing required sections: " + ", ".join(missing))
    if extra:
        raise ValueError("unknown sections not present in template: " + ", ".join(extra))

    ordered_body = "\n\n".join(sections[title] for title in order).rstrip()
    design = "# " + op_name + " 设计文档\n\n" + ordered_body + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(design, encoding="utf-8")
    _LOGGER.info(f"Wrote {output_path}")


@dataclass
class PlanWriteInputs:
    """聚合写 PLAN.md 所需的参数（避免长参数列表）。"""

    op_name: str
    spec_text: str
    plan_text: str | None
    plan_output_path: Path
    args: argparse.Namespace
    manifest: dict


def write_plan(inputs: PlanWriteInputs) -> None:
    """Write PLAN.md, rendering a default from the plan template when needed."""
    plan_text = inputs.plan_text
    if plan_text is None:
        plan_template_path = inputs.args.plan_template or _manifest_path(inputs.manifest, "plan_template")
        template_text_for_plan = read_text(plan_template_path) if plan_template_path else None
        plan_text = render_default_plan(inputs.op_name, inputs.spec_text, template_text_for_plan)
    inputs.plan_output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs.plan_output_path.write_text(plan_text, encoding="utf-8")
    _LOGGER.info(f"Wrote {inputs.plan_output_path}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return _run(parser, args)
    except ValueError as exc:
        _LOGGER.error(str(exc))
        return 1


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    skill_dir = Path(__file__).resolve().parents[1]
    manifest = load_manifest(args.work_dir)
    paths = resolve_paths(args, manifest, skill_dir)

    if paths["spec"] is None:
        parser.error("missing spec path; pass --spec or --work-dir with manifest.json")
    if paths["sections"] is None:
        parser.error("missing sections path; pass --sections or --work-dir")
    if paths["output"] is None:
        parser.error("missing output path; pass --output or --output-dir")

    spec_text = read_text(paths["spec"])
    template_text = read_text(paths["template"])
    op_name = extract_op_name(spec_text)
    if args.operator and args.operator != op_name:
        raise ValueError(f"operator mismatch: --operator {args.operator!r}, spec op.name {op_name!r}")

    sections, plan_text = collect_sections(paths["sections"])
    write_design(op_name, template_text, sections, paths["output"])

    if paths["plan_output"]:
        write_plan(
            PlanWriteInputs(
                op_name=op_name,
                spec_text=spec_text,
                plan_text=plan_text,
                plan_output_path=paths["plan_output"],
                args=args,
                manifest=manifest,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
