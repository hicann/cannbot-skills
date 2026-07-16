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
"""Create phased input bundles for ops-registry-invoke spec-to-design."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import yaml

_RESOURCES_DIR = Path(__file__).resolve().parents[3] / "workflow" / "resources"
if _RESOURCES_DIR.is_dir() and str(_RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_DIR))
from _output_log import get_logger

_LOGGER = get_logger("ops_registry_invoke.slice_design_inputs")


# 各输入包 spec 切片共用的字段片段，避免多处重复相同的字段名字面量序列。
_IO_BLOCKS = ["op", "inputs", "attributes", "outputs"]
_CONTRACT_BLOCKS = ["numerical_tolerance", "boundary_conditions", "extreme_inputs", "determinism"]
_PERF_BLOCKS = ["performance_budget", "performance_baseline"]


def _compose_blocks(*groups: list[str]) -> list[str]:
    """按顺序拼接若干字段片段为单个 spec 切片字段列表。"""
    composed: list[str] = []
    for group in groups:
        composed.extend(group)
    return composed


DESIGN_BUNDLES = [
    {
        "file": "01-overview-contract.md",
        "sections": ["修订记录", "1. 概述"],
        "blocks": _compose_blocks(
            _IO_BLOCKS,
            ["reduction", "dtype_policy", "broadcast", "math_semantics"],
            _CONTRACT_BLOCKS,
        ),
    },
    {
        "file": "02-architecture.md",
        "sections": ["2. 架构设计"],
        "blocks": _compose_blocks(
            _IO_BLOCKS,
            ["reduction", "shape_constraints", "dtype_policy", "broadcast", "interface_binding"],
        ),
    },
    {
        "file": "03-implementation.md",
        "sections": ["3. 实现方案"],
        "blocks": _compose_blocks(
            _IO_BLOCKS,
            ["reduction", "shape_constraints", "dtype_policy", "broadcast", "math_semantics",
             "numerical_stability"],
            _PERF_BLOCKS,
            ["interface_binding"],
        ),
    },
    {
        "file": "04-quality-plan.md",
        "sections": ["4. 性能优化", "5. 风险评估", "6. 交付件清单", "7. 迭代规划", "8. Design Contract"],
        "blocks": _compose_blocks(
            _IO_BLOCKS,
            ["dtype_policy", "broadcast", "math_semantics", "numerical_stability"],
            _CONTRACT_BLOCKS,
            _PERF_BLOCKS,
        ),
    },
]

PLAN_BUNDLE = {
    "file": "05-plan.md",
    "blocks": _compose_blocks(
        _IO_BLOCKS,
        ["reduction", "shape_constraints", "dtype_policy", "broadcast", "math_semantics"],
        _CONTRACT_BLOCKS,
        _PERF_BLOCKS,
    ),
}

TOP_LEVEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)")

# 参考材料区块的公共前言（设计包与计划包共用，仅结尾用途词不同）。
_REFS_PREAMBLE = (
    "所有 `patterns.md` 位于本 skill 的 references/ 下。**必须先逐个 Read 阅读**，"
    "再根据各 `patterns.md` 内部的相对路径**继续 Read 阅读二级文件**。"
    "所有设计模式、API 约束、避坑指南必须取自这条完整路径，禁止跳过任何一层"
)

_CATEGORY_REFS_HEADER = [
    "### 主参考材料（category 路由 — 必须优先阅读）",
    "",
    "以下由 op.category 路由，是算子主特征决定的**核心设计模式**。",
    "**必须最先 Read 阅读**，所有架构决策必须与此对齐。",
    "",
]

_PARADIGM_REFS_HEADER = [
    "### 补充参考材料（paradigm 路由 — 组合叠加）",
    "",
    "以下由 op.paradigms 自动路由，是叠加在主特征上的**补充知识**。",
    "在主参考材料基础上，按需 Read 阅读以获取额外的设计约束和避坑指南。",
    "",
]


@dataclass
class ParadigmResolveInputs:
    """聚合 paradigm 解析所需的相关参数（避免长参数列表）。"""

    routes: dict
    op_name: str
    category: str
    skill_dir: Path
    paradigms: list[str] = field(default_factory=list)
    routing: dict = field(default_factory=dict)
    paradigm_groups: list[dict] = field(default_factory=list)


@dataclass
class DesignBundleContent:
    """聚合写设计输入包所需的相关内容（避免长参数列表）。"""

    op_name: str
    bundle: dict[str, object]
    spec_slice: str
    req_text: str
    template_excerpt: str
    category_refs: list[str] | None = None
    paradigm_refs: list[str] | None = None
    paradigm_partitions: list[dict] | None = None


@dataclass
class PlanBundleContent:
    """聚合写计划输入包所需的相关内容（避免长参数列表）。"""

    op_name: str
    spec_slice: str
    req_text: str
    plan_template: str
    category_refs: list[str] | None = None
    paradigm_refs: list[str] | None = None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_op_block_lines(spec_text: str):
    """Yield the lines nested under the top-level ``op:`` mapping in spec.yaml."""
    in_op = False
    for line in spec_text.splitlines():
        if line.startswith("op:"):
            in_op = True
            continue
        if in_op and line and not line.startswith((" ", "\t", "#")):
            break
        if in_op:
            yield line


def extract_op_name(spec_text: str) -> str:
    for line in _iter_op_block_lines(spec_text):
        match = re.match(r"\s+name:\s*[\"']?([^\"'#\s]+)", line)
        if match:
            return match.group(1)
    raise ValueError("could not find op.name in spec")


def extract_top_level_blocks(spec_text: str) -> dict[str, str]:
    lines = spec_text.splitlines()
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = TOP_LEVEL_RE.match(line)
        if match:
            starts.append((match.group(1), index))

    blocks: dict[str, str] = {}
    for pos, (name, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(lines)
        blocks[name] = "\n".join(lines[start:end]).rstrip()
    return blocks


def extract_template_sections(template_text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## .+$", template_text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(template_text)
        title = match.group(0).removeprefix("## ").strip()
        sections[title] = template_text[start:end].strip()
    return sections


def render_spec_slice(blocks: dict[str, str], names: list[str]) -> str:
    selected = [blocks[name] for name in names if name in blocks]
    return "\n\n".join(selected).strip()


def render_template_excerpt(sections: dict[str, str], names: list[str]) -> str:
    missing = [name for name in names if name not in sections]
    if missing:
        raise ValueError(f"template is missing sections: {', '.join(missing)}")
    return "\n\n".join(sections[name] for name in names).strip()


def default_outputs(spec: Path, op_name: str) -> tuple[Path, Path]:
    if spec.name == "spec.yaml" and spec.parent.name == "docs":
        return spec.parent / "DESIGN.md", spec.parent / "PLAN.md"
    return spec.parent / "DESIGN.md", spec.parent / "PLAN.md"


def requirements_excerpt(path: Path | None) -> str:
    if not path:
        return (
            "未提供 REQUIREMENTS.md。只能使用 spec.yaml 字段生成设计，"
            "需求背景、ACLNN 接口自然语言说明、性能目标和资源约束需标记待补充。"
        )
    if not path.exists():
        return (
            f"REQUIREMENTS.md 不存在：{path}。"
            "只能使用 spec.yaml 字段生成设计，缺失需求信息需标记待补充。"
        )
    text = read_text(path)
    # Keep enough context for design while avoiding huge bundles.
    lines = text.splitlines()
    if len(lines) <= 420:
        return text
    head = "\n".join(lines[:260])
    tail = "\n".join(lines[-120:])
    return head + "\n\n<!-- 中间内容已省略；如需要，请直接读取 REQUIREMENTS.md 原文。 -->\n\n" + tail


def _read_yaml_routes(routes_file: Path, result: dict) -> None:
    """从 routes_file 读取 routes 并写入 result（存在时）。"""
    with open(routes_file) as f:
        routes_data = yaml.safe_load(f) or {}
        result["routes"] = routes_data.get("routes", {})


def load_paradigm_refs(skill_dir: Path) -> dict:
    """Load paradigm-refs.yaml and routes.yaml.

    - skill_homes: from spec-to-design/references/paradigm-refs.yaml
    - routes: 优先读取插件内置 spec-to-design/references/paradigms/routes.yaml，
      否则回退到 <skill_home>/references/paradigms/routes.yaml
    """
    result: dict = {}

    # Load skill_homes from paradigm-refs.yaml
    refs_file = skill_dir / "references" / "paradigm-refs.yaml"
    if refs_file.exists():
        with open(refs_file) as f:
            data = yaml.safe_load(f) or {}
            result["skill_homes"] = data.get("skill_homes", {})

    # Load routes: 优先插件内置 routes.yaml，其次 <skill_home>/references/paradigms/routes.yaml
    skill_homes = result.get("skill_homes", {})
    if not skill_homes:
        return result

    # Use first skill_home to locate the routes file
    home = next(iter(skill_homes.values()))
    routes_candidates = [
        skill_dir / "references" / "paradigms" / "routes.yaml",
        skill_dir.parent / home / "references" / "paradigms" / "routes.yaml",
        skill_dir.parent.parent / ".opencode" / "skills" / home / "references" / "paradigms" / "routes.yaml",
        skill_dir.parent.parent.parent.parent / "ops" / home / "references" / "paradigms" / "routes.yaml",
    ]
    for routes_file in routes_candidates:
        if routes_file.exists():
            _read_yaml_routes(routes_file, result)
            break

    return result


def extract_paradigm_info(spec_text: str) -> tuple[str, list[str]]:
    """Extract op.category and op.paradigms from spec.yaml text."""
    category = ""
    paradigms: list[str] = []
    for line in _iter_op_block_lines(spec_text):
        if "category:" in line:
            category = line.split(":", 1)[1].strip()
        if "paradigms:" in line:
            paradigms = re.findall(r"(\w+)", line.split(":", 1)[1])
    return category, paradigms


def extract_paradigm_routing(spec: Path) -> dict | None:
    """Extract op.paradigm_routing from spec.yaml, return None if absent."""
    try:
        data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    except Exception:
        return None
    routing = (data or {}).get("op", {}).get("paradigm_routing")
    if not routing or not isinstance(routing, dict):
        return None
    return routing


def extract_paradigm_groups(spec: Path) -> list[dict]:
    """Extract op.paradigm_groups from spec.yaml, return [] if absent."""
    try:
        data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    except Exception:
        return []
    groups = (data or {}).get("op", {}).get("paradigm_groups")
    if not groups or not isinstance(groups, list):
        return []
    return groups


def _route_candidate_paths(skill_dir: Path, home: str, ref_path: str) -> list[Path]:
    """给定 skill_home 与相对路径，返回候选绝对路径列表。"""
    return [
        skill_dir.parent / home / ref_path,
        skill_dir.parent.parent / ".opencode" / "skills" / home / ref_path,
        skill_dir.parent.parent.parent.parent / "ops" / home / ref_path,
    ]


def _local_paradigm_dirs(paradigms_dir: Path) -> dict[str, str]:
    """返回本地 references/paradigms/ 下的目录名映射 {lower: original}。"""
    if not paradigms_dir.exists():
        return {}
    return {d.name.lower(): d.name for d in paradigms_dir.iterdir() if d.is_dir()}


def _first_existing_candidate(skill_dir: Path, home: str, ref_path: str) -> str | None:
    """返回 ref_path 在候选目录中首个存在文件的绝对路径字符串。"""
    for full in _route_candidate_paths(skill_dir, home, ref_path):
        if full.exists():
            return str(full)
    return None


def _resolve_routed_paradigm(
    para_routes: dict,
    skill_homes: dict,
    skill_dir: Path,
    paradigm: str,
) -> list[str]:
    """Level 1: 为单个 paradigm 解析 routes 命中的绝对路径列表。"""
    home = skill_homes.get(paradigm, skill_homes.get("default"))
    if not home:
        return []
    resolved: list[str] = []
    for ref_path in para_routes.get(paradigm, []):
        full = _first_existing_candidate(skill_dir, home, ref_path)
        if full:
            resolved.append(full)
    return resolved


def _resolve_local_paradigm(existing_dirs: dict[str, str], paradigms_dir: Path, paradigm: str) -> str | None:
    """Level 2: 为单个 paradigm 解析本地 references/paradigms/ 下的 patterns.md。"""
    dir_name = existing_dirs.get(paradigm.lower())
    if not dir_name:
        return None
    pat_file = paradigms_dir / dir_name / "patterns.md"
    return str(pat_file) if pat_file.exists() else None


def _resolve_paradigm_refs_for(
    routes: dict,
    skill_dir: Path,
    paras: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve paradigm reference file paths for a list of paradigm names.

    Returns (refs, matched) where:
      - refs: list of absolute paths to resolved reference files
      - matched: {paradigm_name: [resolved_paths]} per paradigm
    """
    para_routes = (routes.get("routes", {}) or {}).get("paradigms", {})
    skill_homes = routes.get("skill_homes", {})
    refs: list[str] = []
    matched: dict[str, list[str]] = {}
    seen: set[str] = set()
    # Level 1: routes.yaml
    for p in paras:
        for full in _resolve_routed_paradigm(para_routes, skill_homes, skill_dir, p):
            if full not in seen:
                refs.append(full)
                seen.add(full)
            matched.setdefault(p, []).append(full)
    # Level 2: local references/paradigms/ fallback
    paradigms_dir = skill_dir / "references" / "paradigms"
    existing_dirs = _local_paradigm_dirs(paradigms_dir)
    for p in paras:
        if p in matched:
            continue
        pat_file = _resolve_local_paradigm(existing_dirs, paradigms_dir, p)
        if pat_file:
            refs.append(pat_file)
            matched[p] = [pat_file]
    return refs, matched


def resolve_paradigm_refs_from_groups(inputs: ParadigmResolveInputs) -> list[dict]:
    """Resolve paradigm references from paradigm_groups combination entries.

    Each kind:combination entry becomes one partition.

    Returns a list of partitions, each with:
      - label: str  (e.g. 'reduction={"none"}')
      - active_paradigms: list[str]
      - paradigm_refs: list[str]  (absolute paths resolved for this group)
      - matched: dict[str, list[str]]
    """
    partitions: list[dict] = []
    for group in inputs.paradigm_groups:
        if group.get("kind") != "combination":
            continue
        switch = group.get("switch", "?")
        when = group.get("when", "*")
        active = group.get("paradigms", [])
        label = f'{switch}={{"{when}"}}'
        refs, matched = _resolve_paradigm_refs_for(inputs.routes, inputs.skill_dir, active)
        partitions.append({
            "label": label,
            "active_paradigms": active,
            "paradigm_refs": refs,
            "matched": matched,
        })
    return partitions


def resolve_paradigm_refs_partitioned(inputs: ParadigmResolveInputs) -> list[dict]:
    """Resolve paradigm references per paradigm_routing case.

    Returns a list of partitions, each with:
      - label: str  (e.g. 'reduction="none"')
      - active_paradigms: list[str]
      - paradigm_refs: list[str]  (absolute paths resolved for this case)
      - matched: dict[str, list[str]]
    """
    routing = inputs.routing
    partitions: list[dict] = []
    for case in routing.get("cases", []):
        values = case.get("values", [])
        if "value" in case:
            values = [case["value"]]
        active = case.get("active_paradigms", [])
        label_values = " | ".join(f'"{v}"' for v in values)
        label = f'{routing["attribute"]}={{{label_values}}}'
        refs, matched = _resolve_paradigm_refs_for(inputs.routes, inputs.skill_dir, active)
        partitions.append({
            "label": label,
            "active_paradigms": active,
            "paradigm_refs": refs,
            "matched": matched,
        })
    return partitions


def _resolve_ref_path(skill_dir: Path, home: str, ref_path: str) -> str | None:
    """在 sibling / .opencode / ops 三处依次查找 ref_path，返回首个存在的绝对路径。"""
    sibling, oc, ops = _route_candidate_paths(skill_dir, home, ref_path)
    if sibling.exists():
        return str(sibling)
    if oc.exists():
        return str(oc)
    if ops.exists():
        return str(ops)
    return None


def resolve_paradigm_refs(inputs: ParadigmResolveInputs) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Resolve paradigm reference paths.

    Matches op.paradigms list item by item, results deduplicated.

    Returns:
        (category_refs, paradigm_refs, matched)
        category_refs:  always empty (reserved for future use)
        paradigm_refs:  paradigm 路由到的参考材料（组合叠加）
        matched:        {paradigm_name: [absolute_paths]} for each paradigm that resolved
    """
    routes = inputs.routes
    skill_dir = inputs.skill_dir
    cat_refs: list[str] = []
    para_refs: list[str] = []
    matched: dict[str, list[str]] = {}
    route_routes = routes.get("routes", {})
    skill_homes = routes.get("skill_homes", {})
    name_routes = route_routes.get("name_patterns", {})
    para_routes = route_routes.get("paradigms", {})

    seen: set[str] = set()

    def _resolve(ref_paths, target_list: list[str], paradigm: str = "") -> list[str]:
        resolved: list[str] = []
        for ref_path in ref_paths:
            home = skill_homes.get(paradigm, skill_homes.get("default")) if paradigm else None
            if not home:
                continue
            full = _resolve_ref_path(skill_dir, home, ref_path)
            if not full:
                continue
            if full not in seen:
                target_list.append(full)
                seen.add(full)
            resolved.append(full)
        return resolved

    # paradigms 路由 — 组合叠加
    for p in inputs.paradigms:
        resolved = _resolve(para_routes.get(p, []), para_refs, p)
        if resolved:
            matched[p] = resolved

    # 3. name_patterns — 兜底（regex 匹配算子名）
    for pattern, ref_paths in name_routes.items():
        if re.search(pattern, inputs.op_name):
            _resolve(ref_paths, para_refs)

    return cat_refs, para_refs, matched


def detect_horizontal_combination(paradigm_partitions: list[dict]) -> dict | None:
    """Detect 横向组合（paradigm_groups combination / paradigm_routing 产生 ≥2 条不同计算流）.

    准入条件：spec.yaml 声明了 op.paradigm_routing 且产生 ≥2 条不同的计算流即触发，
    不要求范式互斥（与 advanced-guide.md 一致）。
    每个 distinct active_paradigms 集合视为一条计算流，按首次出现顺序分配 partition 值。

    Returns:
        dict | None — None 表示非横向组合；否则含:
          is_combination: True
          n_flows:  计算流条数
          assignments: [(partition_label, partition_id, constant_name), ...]
    """
    flow_map: dict[frozenset, int] = {}
    seen: set[str] = set()
    assignments: list[tuple[str, int, str]] = []
    for part in paradigm_partitions:
        key = frozenset(part.get("active_paradigms", []) or [])
        if key not in flow_map:
            flow_map[key] = len(flow_map)
        fid = flow_map[key]
        active = part.get("active_paradigms", []) or []
        base = active[0].upper() if active else "FLOW"
        const_name = f"PARTITION_{base}"
        if const_name in seen:
            const_name = f"PARTITION_{base}_{fid}"
        seen.add(const_name)
        assignments.append((part.get("label", "?"), fid, const_name))
    n_flows = len(flow_map)
    if n_flows < 2:
        return None
    return {"is_combination": True, "n_flows": n_flows, "assignments": assignments}


def _build_combination_sel_block(n: int, assignments: list, const_names: list[str]) -> str:
    """构造 ASCENDC_TPL_SEL 各分区行块。"""
    sel_rows = []
    for fid in range(n):
        labels = ", ".join(lbl for lbl, f, _ in assignments if f == fid)
        cname = const_names[fid]
        sel_rows.append(
            f"    ASCENDC_TPL_ARGS_SEL(\n"
            f"        ASCENDC_TPL_UINT_SEL(partition, ASCENDC_TPL_UI_LIST, {cname}),   // 分区：{labels}\n"
            f"        // 该分区专属参数按 patterns.md 完整填写\n"
            f"        // 其他分区专属参数取固定默认值（如 ASCENDC_TPL_BOOL_SEL(isEmpty, 0)）\n"
            f"    ),"
        )
    return "\n".join(sel_rows)


def _build_combination_entry_skeleton(n: int, assignments: list, const_names: list[str]) -> str:
    """构造单 Kernel 入口按 partition 分发的骨架代码。"""
    entry_lines = []
    for fid in range(n):
        labels = ", ".join(lbl for lbl, f, _ in assignments if f == fid)
        cname = const_names[fid]
        kw = "if" if fid == 0 else "} else if"
        entry_lines.append(f"    {kw} constexpr (partition == {cname}) {{")
        entry_lines.append(
            f"        GET_TILING_DATA_WITH_STRUCT(<流{fid} 的 TilingData struct 名>, tilingData, tiling);"
            f"  // 分区：{labels}"
        )
        entry_lines.append(
            "        // 该分区计算体（内层按 isEmpty/group 子模板分发 + 实例化对应 Kernel 类并 Process）"
        )
    entry_lines.append("    }")
    return "\n".join(entry_lines)


def build_combination_directive_lines(combination: dict) -> list[str]:
    """Build the ⛔ 横向组合强制约束 markdown block (list of lines)."""
    n = combination["n_flows"]
    assignments = combination["assignments"]

    const_names = [a[2] for a in assignments]
    unique_consts = list(dict.fromkeys(const_names))
    consts_str = ", ".join(unique_consts)
    decl_sig = f"partition, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, {consts_str}"

    mapping = "\n".join(
        f"- `{label}` → partition = {cname}" for label, _, cname in assignments
    )

    sel_block = _build_combination_sel_block(n, assignments, const_names)
    entry_skeleton = _build_combination_entry_skeleton(n, assignments, const_names)

    block = f"""### ⛔ 横向组合强制约束（检测到 {n} 条不同计算流）

本算子 paradigm_routing 各分区激活了不同的范式子集 = {n} 条不同的计算流（对应 spec.yaml `op.paradigm_groups` 的 `kind: combination` 横向组合）。一个算子只有**一个 kernel 入口**，多条计算流必须在同一入口经 TilingKey 分发。

**分区 → 分区选择维度取值映射**（首字段 `partition` 的取值）：

{mapping}

生成 §3.2 模板划分 / TilingKey 声明 与 §3.5 Kernel 入口时**强制**：

1. **必须生成【完整】的统一 `ASCENDC_TPL_ARGS_DECL` 定义**：第一个参数固定为"分区选择维度"字段 `partition`（kernel 入口最先读它决定走哪条流），其后接全部子模板字段（isEmpty/isGroup 等按各分区 patterns.md 完整枚举）。**字段必须填全，禁止省略、禁止 `// ...` 占位**。分区选择维度统一使用 `ASCENDC_TPL_UINT_DECL`（位宽 `ASCENDC_TPL_8_BW`，模式 `ASCENDC_TPL_UI_LIST`）。**禁止**为每条流各写独立 DECL，**禁止**声明多个 kernel 入口。
2. **必须生成【完整】的统一 `ASCENDC_TPL_SEL`**，枚举**全部** TilingKey 组合（所有分区 × 所有子模板），每条 `ASCENDC_TPL_ARGS_SEL` 首先用 `partition` 取值标注所属分区；**不适用于当前分区的参数取固定默认值**。
3. dtype 不进 TilingKey（编译期 DTYPE_X 实例化）。
4. **单 Kernel 入口（§3.5）取 TilingData 必须用"未注册 struct"模式**：因每条计算流的 TilingData struct 不同（无法注册单个 struct），入口必须：
   - 首行 `REGISTER_NONE_TILING;`（**禁止** `REGISTER_TILING_DEFAULT` / `REGISTER_TILING_FOR_TILINGKEY`，且与二者**不可混用**）
   - 仅可配 `GET_TILING_DATA_WITH_STRUCT`（**禁止** `GET_TILING_DATA`）
   - `GET_TILING_DATA_WITH_STRUCT(<该流 TilingData struct 名>, tilingData, tiling)` 必须放在**每个 `if constexpr (partition == PARTITION_X)` 分支内**，按该流 struct 类型分别取（编译期依赖 partition，分支外不可一次性取）；struct 名取自各流 §3.3 的 TilingData 定义
   - 入口模板参数顺序与 `ASCENDC_TPL_ARGS_DECL` 字段一致，首参为 `partition`

示例（{n} 条流，首字段 = 分区选择维度）：

```cpp
ASCENDC_TPL_ARGS_DECL(
    OpName,
    ASCENDC_TPL_UINT_DECL({decl_sig}),   // ⭐分区选择（首参）：入口最先读它分发到对应流
    // 其余子模板字段按各分区 patterns.md 完整填写（禁止省略）
);
ASCENDC_TPL_SEL(
{sel_block}
);
```

示例（{n} 条流，单 Kernel 入口取 TilingData）：

```cpp
template <uint32_t partition, ...>   // 首参 partition = 分区选择维度，字段顺序与 ASCENDC_TPL_ARGS_DECL 一致
__global__ __aicore__ void OpName(GM_ADDR /*inputs*/, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{{
    REGISTER_NONE_TILING;   // 用未注册的自定义 C++ TilingData（每流 struct 不同，不能注册单个）
{entry_skeleton}
}}
```"""
    return block.split("\n")


def _wrap_refs_block(inner: str, purpose_tail: str) -> str:
    """将参考材料内容包裹进带前言的标准区块。

    purpose_tail: 前言结尾的用途词（如 "直接动笔" / "直接编写 PLAN.md"）。
    """
    return f"""
## 参考材料（必须阅读）

{_REFS_PREAMBLE}{purpose_tail}。

{inner}
"""


def _category_refs_lines(category_refs: list[str], bold: bool = True) -> list[str]:
    """构造 category 主参考材料的行列表。"""
    lines = list(_CATEGORY_REFS_HEADER)
    for ref in category_refs:
        lines.append(f"- **{ref}**" if bold else f"- {ref}")
    lines.append("")
    return lines


def _paradigm_refs_lines(paradigm_refs: list[str]) -> list[str]:
    """构造 paradigm 补充参考材料的行列表。"""
    lines = list(_PARADIGM_REFS_HEADER)
    for ref in paradigm_refs:
        lines.append(f"- {ref}")
    return lines


def _build_partitioned_refs(
    category_refs: list[str] | None,
    paradigm_partitions: list[dict],
) -> str:
    """构造分区模式（paradigm_groups combination）的参考材料区块。"""
    lines: list[str] = []
    if category_refs:
        lines.extend(_category_refs_lines(category_refs))
    lines.append("### 条件范式路由（paradigm_groups combination — 分区激活）")
    lines.append("")
    lines.append("本算子按属性值激活不同范式子集，每个分区对应一套独立的 Kernel 模板。")
    lines.append("请为每个分区分别 Read 对应的 patterns.md，并在 §3.2 模板划分总览中按分区规划 TilingKey。")
    lines.append("")
    for part in paradigm_partitions:
        lines.append(f"#### 分区：{part['label']}")
        lines.append("")
        lines.append(f"- 激活范式：`{part['active_paradigms']}`")
        lines.append("- 参考材料：")
        for ref in part["paradigm_refs"]:
            lines.append(f"  - {ref}")
        lines.append("")
    # 横向组合检测（≥2 互斥计算流）→ 注入统一 ASCENDC_TPL_ARGS_DECL 强制约束
    combination = detect_horizontal_combination(paradigm_partitions)
    if combination:
        lines.extend(build_combination_directive_lines(combination))
    return _wrap_refs_block("\n".join(lines), "直接动笔")


def _build_simple_refs(
    category_refs: list[str] | None,
    paradigm_refs: list[str] | None,
    purpose_tail: str,
) -> str:
    """构造非分区模式（category + paradigm 叠加）的参考材料区块。"""
    if not (category_refs or paradigm_refs):
        return ""
    lines: list[str] = []
    if category_refs:
        lines.extend(_category_refs_lines(category_refs))
    if paradigm_refs:
        lines.extend(_paradigm_refs_lines(paradigm_refs))
    return _wrap_refs_block("\n".join(lines), purpose_tail)


def write_design_bundle(path: Path, content: DesignBundleContent) -> None:
    bundle = content.bundle
    section_list = "\n".join(f"- `## {section}`" for section in bundle["sections"])

    if content.paradigm_partitions:
        refs_block = _build_partitioned_refs(content.category_refs, content.paradigm_partitions)
    else:
        refs_block = _build_simple_refs(content.category_refs, content.paradigm_refs, "直接动笔")

    body = f"""# spec-to-design 输入包：{path.stem}

算子：`{content.op_name}`

请只生成以下 DESIGN.md 章节：

{section_list}

规则：
- 必须使用简体中文。
- 必须使用模板摘录中的 `##` 标题，标题文字不可改名。
- 只能使用本输入包中的 spec 切片、需求摘要和模板摘录。
- 不要包含文档标题；只返回指定章节 markdown。
- 不要保留模板说明性文字或占位符。
- 对无法从 spec 或需求确认的信息，写"待补充/需回到 spec-generation 修订"，不要编造。
- API 支持状态必须来自可信来源；如果本输入包没有验证证据，只能写"待验证"。
- **必须阅读「参考材料」中列出的所有文件**，将其中的设计模式、避坑指南、API 约束应用到本章节内容中。
{refs_block}
## spec.yaml 切片

```yaml
{content.spec_slice}
```

## REQUIREMENTS.md 摘要

```markdown
{content.req_text}
```

## 模板摘录

```markdown
{content.template_excerpt}
```
"""
    path.write_text(body, encoding="utf-8")


def write_plan_bundle(path: Path, content: PlanBundleContent) -> None:
    refs_block = _build_simple_refs(
        content.category_refs, content.paradigm_refs, "直接编写 PLAN.md"
    )

    body = f"""# spec-to-design 输入包：{path.stem}

算子：`{content.op_name}`

请生成完整 `PLAN.md`，首行必须是 `# {content.op_name} 迭代执行计划`。

规则：
- 必须使用简体中文。
- 只能使用本输入包中的 spec 切片、需求摘要和 PLAN 模板。
- 迭代计划必须与 DESIGN.md 的模板划分/TilingKey 策略保持一致。
- dtype、shape、broadcast、boundary、extreme、tolerance 必须来自 spec.yaml。
- 对无法确认的信息，写"待补充/需回到方案设计修订"，不要编造。
- 返回完整 markdown 文档。
- **必须阅读「参考材料」中列出的所有文件**，将其中的设计模式、迭代策略、避坑指南应用到 PLAN.md 中。
{refs_block}
## spec.yaml 切片

```yaml
{content.spec_slice}
```

## REQUIREMENTS.md 摘要

```markdown
{content.req_text}
```

## PLAN 模板

```markdown
{content.plan_template}
```
"""
    path.write_text(body, encoding="utf-8")


def reset_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise ValueError(f"output directory already exists: {path}")
        shutil.rmtree(path)
    (path / "bundles").mkdir(parents=True)
    (path / "sections").mkdir()


def update_manifest(output_dir: Path, values: dict[str, object]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(read_text(manifest_path)) if manifest_path.exists() else {}
    manifest.update(values)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _print_category_trace(category: str, category_refs: list) -> None:
    _LOGGER.info(f"  category:         {category}")
    if category_refs:
        _LOGGER.info("  category refs (主要):")
        for ref in category_refs:
            _LOGGER.info(f"    -> {ref}")
    else:
        _LOGGER.info("  category refs:    (none)")


def _print_paradigm_matches(partitions, matched: dict) -> None:
    if partitions:
        _LOGGER.info("  paradigm_partitions (分区激活):")
        for part in partitions:
            _LOGGER.info(f"    {part['label']}:")
            _LOGGER.info(f"      active_paradigms: {part['active_paradigms']}")
            for ref in part["refs"]:
                _LOGGER.info(f"        -> {ref}")
    elif matched:
        _LOGGER.info("  paradigm matched (补充):")
        for p, ref_list in matched.items():
            for ref in ref_list:
                _LOGGER.info(f"    {p} -> {ref}")
    else:
        _LOGGER.info("  paradigm matched:   (none)")


def print_paradigm_trace(trace: dict[str, object]) -> None:
    """Print a human-readable paradigm-routing summary for the architect's LOG.md."""
    category = trace.get("category") or ""
    category_refs = trace.get("category_refs") or []
    declared = trace.get("declared") or []
    matched = trace.get("matched") or {}
    unmatched = trace.get("unmatched") or []
    fallback_used = bool(trace.get("fallback_used"))
    partitions = trace.get("partitions")

    _LOGGER.info("Paradigm trace:")
    _print_category_trace(category, category_refs)
    _LOGGER.info(f"  paradigms declared: {list(declared) if declared else '(none)'}")
    _print_paradigm_matches(partitions, matched)
    if unmatched:
        _LOGGER.info(f"  unmatched: {unmatched}")
    if fallback_used:
        _LOGGER.info("  fallback:  general-methodology.md (no paradigm matched)")
    else:
        _LOGGER.info("  fallback:  not used")


@dataclass
class ParadigmRouting:
    """paradigm 路由计算结果，供 bundle 写入与 trace 打印共用。"""

    category: str
    category_refs: list[str] = field(default_factory=list)
    paradigm_refs: list[str] = field(default_factory=list)
    paradigms_declared: list[str] = field(default_factory=list)
    matched: dict[str, list[str]] = field(default_factory=dict)
    partitions: list[dict] | None = None
    routing: dict | None = None
    paradigm_groups: list[dict] = field(default_factory=list)
    fallback_used: bool = False
    fallback_file: Path | None = None


@dataclass
class BundleWriteInputs:
    """聚合写全部输入包所需的参数（避免长参数列表）。"""

    output_dir: Path
    op_name: str
    blocks: dict[str, str]
    template_sections: dict[str, str]
    req_text: str
    plan_template: str
    plan_output: Path
    pr: ParadigmRouting


def _resolve_local_paradigm_fallback(
    references_dir: Path,
    paradigms_declared: list[str],
    matched: dict[str, list[str]],
    paradigm_refs: list[str],
) -> None:
    """Level 2: references/paradigms/{paradigm}/patterns.md for unmatched paradigms."""
    paradigms_dir = references_dir / "paradigms"
    existing_dirs = _local_paradigm_dirs(paradigms_dir)
    for p in paradigms_declared:
        if p in matched:
            continue
        dir_name = existing_dirs.get(p.lower())
        if not dir_name:
            continue
        pat_file = paradigms_dir / dir_name / "patterns.md"
        if pat_file.exists():
            paradigm_refs.append(str(pat_file))
            matched[p] = [str(pat_file)]


def compute_paradigm_routing(spec: Path, spec_text: str, op_name: str, skill_dir: Path) -> ParadigmRouting:
    """Compute paradigm routing (category/paradigm refs + partitions + fallback)."""
    references_dir = skill_dir / "references"

    category, paradigms_declared = extract_paradigm_info(spec_text)
    routing = extract_paradigm_routing(spec)
    paradigm_groups = extract_paradigm_groups(spec)
    combination_groups = [g for g in paradigm_groups if g.get("kind") == "combination"]

    category_refs: list[str] = []
    paradigm_refs: list[str] = []
    matched: dict[str, list[str]] = {}
    partitions: list[dict] | None = None

    # Level 1: paradigm-refs.yaml (category → primary, paradigms → supplementary)
    refs_config = load_paradigm_refs(skill_dir)
    if refs_config:
        base_inputs = ParadigmResolveInputs(
            routes=refs_config,
            op_name=op_name,
            category=category,
            skill_dir=skill_dir,
            paradigms=paradigms_declared,
            routing=routing or {},
            paradigm_groups=paradigm_groups,
        )
        category_refs, paradigm_refs, matched = resolve_paradigm_refs(base_inputs)

        if combination_groups:
            # Primary: derive partitions from paradigm_groups combination
            partitions = resolve_paradigm_refs_from_groups(base_inputs)
        elif routing:
            # Backward compat: old specs still using paradigm_routing
            partitions = resolve_paradigm_refs_partitioned(base_inputs)

    if partitions is None:
        _resolve_local_paradigm_fallback(references_dir, paradigms_declared, matched, paradigm_refs)

    # Level 3: general-methodology.md fallback (→ supplementary when nothing matched)
    fallback_used = False
    fallback_file = (
        skill_dir.parent.parent.parent.parent
        / "ops" / "ascendc-regbase-best-practice" / "references" / "paradigms" / "general-methodology.md"
    )
    nothing_matched = not partitions and not category_refs and not paradigm_refs
    if nothing_matched and fallback_file.exists():
        paradigm_refs = [str(fallback_file)]
        fallback_used = True

    return ParadigmRouting(
        category=category,
        category_refs=category_refs,
        paradigm_refs=paradigm_refs,
        paradigms_declared=paradigms_declared,
        matched=matched,
        partitions=partitions,
        routing=routing,
        paradigm_groups=paradigm_groups,
        fallback_used=fallback_used,
        fallback_file=fallback_file if fallback_used else None,
    )


def build_paradigm_trace(pr: ParadigmRouting) -> dict[str, object]:
    """Build the manifest paradigm_trace dict from a ParadigmRouting result."""
    unmatched = [p for p in pr.paradigms_declared if p not in pr.matched]
    trace: dict[str, object] = {
        "category": pr.category,
        "category_refs": pr.category_refs,
        "declared": pr.paradigms_declared,
        "matched": pr.matched,
        "unmatched": unmatched,
        "fallback_used": pr.fallback_used,
    }
    if pr.routing:
        trace["paradigm_routing"] = pr.routing
    if pr.paradigm_groups:
        trace["paradigm_groups"] = pr.paradigm_groups
    if pr.partitions:
        trace["partitions"] = [
            {"label": p["label"], "active_paradigms": p["active_paradigms"], "refs": p["paradigm_refs"]}
            for p in pr.partitions
        ]
    if pr.fallback_used and pr.fallback_file is not None:
        trace["fallback_file"] = str(pr.fallback_file)
    return trace


def _write_one_design_bundle(
    inputs: BundleWriteInputs,
    bundle: dict,
    category_refs: list[str] | None,
    paradigm_refs: list[str] | None,
) -> dict[str, object]:
    """Write a single design bundle and return its manifest entry."""
    blocks = inputs.blocks
    bundle_path = inputs.output_dir / "bundles" / str(bundle["file"])
    spec_slice = render_spec_slice(blocks, list(bundle["blocks"]))
    template_excerpt = render_template_excerpt(inputs.template_sections, list(bundle["sections"]))
    write_design_bundle(
        bundle_path,
        DesignBundleContent(
            op_name=inputs.op_name,
            bundle=bundle,
            spec_slice=spec_slice,
            req_text=inputs.req_text,
            template_excerpt=template_excerpt,
            category_refs=category_refs,
            paradigm_refs=paradigm_refs,
            paradigm_partitions=inputs.pr.partitions,
        ),
    )
    return {
        "file": str(bundle_path),
        "sections": bundle["sections"],
        "spec_blocks": [name for name in bundle["blocks"] if name in blocks],
        "phase": "design",
    }


def _write_the_plan_bundle(
    inputs: BundleWriteInputs,
    category_refs: list[str] | None,
    paradigm_refs: list[str] | None,
) -> dict[str, object]:
    """Write the plan bundle and return its manifest entry."""
    blocks = inputs.blocks
    plan_bundle_path = inputs.output_dir / "bundles" / str(PLAN_BUNDLE["file"])
    plan_spec_slice = render_spec_slice(blocks, list(PLAN_BUNDLE["blocks"]))
    write_plan_bundle(
        plan_bundle_path,
        PlanBundleContent(
            op_name=inputs.op_name,
            spec_slice=plan_spec_slice,
            req_text=inputs.req_text,
            plan_template=inputs.plan_template,
            category_refs=category_refs,
            paradigm_refs=paradigm_refs,
        ),
    )
    return {
        "file": str(plan_bundle_path),
        "output": str(inputs.plan_output),
        "spec_blocks": [name for name in PLAN_BUNDLE["blocks"] if name in blocks],
        "phase": "plan",
    }


def write_all_bundles(inputs: BundleWriteInputs) -> list[dict[str, object]]:
    """Write all design bundles + the plan bundle, returning manifest entries."""
    pr = inputs.pr
    all_paradigm_refs = pr.paradigm_refs if pr.paradigm_refs else None
    all_category_refs = pr.category_refs if pr.category_refs else None

    bundles: list[dict[str, object]] = [
        _write_one_design_bundle(inputs, bundle, all_category_refs, all_paradigm_refs)
        for bundle in DESIGN_BUNDLES
    ]
    bundles.append(_write_the_plan_bundle(inputs, all_category_refs, all_paradigm_refs))
    return bundles


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_pos", nargs="?", type=Path, help="operators/{op}/docs/spec.yaml")
    parser.add_argument("template_pos", nargs="?", type=Path, help="DESIGN.md template")
    parser.add_argument("output_dir_pos", nargs="?", type=Path, help="temporary work directory")
    parser.add_argument("--spec", dest="spec_opt", type=Path, help="operators/{op}/docs/spec.yaml")
    parser.add_argument("--template", dest="template_opt", type=Path, help="DESIGN.md template")
    parser.add_argument(
        "--work-dir", "--output-dir", dest="output_dir_opt", type=Path, help="temporary work directory"
    )
    parser.add_argument("--requirements", type=Path, help="operators/{op}/docs/REQUIREMENTS.md")
    parser.add_argument("--plan-template", type=Path, help="PLAN.md template")
    parser.add_argument("--phase", choices=["base"], default="base")
    parser.add_argument("--force", action="store_true", help="replace output_dir")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        return _run(parser, args)
    except ValueError as exc:
        _LOGGER.error(str(exc))
        return 1


@dataclass
class ResolvedPaths:
    """`_run` 解析后的输入/输出路径集合。"""

    skill_dir: Path
    spec: Path
    template: Path
    output_dir: Path
    plan_template_path: Path


def _resolve_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ResolvedPaths:
    """解析并校验命令行给定的路径参数（缺失时经 parser.error 退出）。"""
    skill_dir = Path(__file__).resolve().parents[1]
    spec_arg = args.spec_opt or args.spec_pos
    template_arg = args.template_opt or args.template_pos or skill_dir / "templates" / "DESIGN.md.templ"
    output_dir_arg = args.output_dir_opt or args.output_dir_pos
    if spec_arg is None:
        parser.error("missing spec path; pass positional spec or --spec")
    if output_dir_arg is None:
        parser.error("missing output directory; pass positional output_dir or --work-dir")

    template = template_arg.resolve()
    plan_template_path = (
        args.plan_template.resolve()
        if args.plan_template
        else template.parent / "PLAN.md.templ"
    )
    return ResolvedPaths(
        skill_dir=skill_dir,
        spec=spec_arg.resolve(),
        template=template,
        output_dir=output_dir_arg.resolve(),
        plan_template_path=plan_template_path,
    )


class ManifestValueInputs(NamedTuple):
    """`_run_manifest_values` 的聚合入参（G.FNM.03：避免函数参数过多）。"""

    paths: ResolvedPaths
    args: argparse.Namespace
    op_name: str
    outputs: tuple[Path, Path]
    bundles: list[dict[str, object]]
    paradigm_trace: dict[str, object]


def _run_manifest_values(inputs: ManifestValueInputs) -> dict[str, object]:
    """组装写入 manifest.json 的键值。"""
    paths = inputs.paths
    args = inputs.args
    design_output, plan_output = inputs.outputs
    return {
        "op_name": inputs.op_name,
        "spec": str(paths.spec),
        "requirements": str(args.requirements.resolve()) if args.requirements else None,
        "template": str(paths.template),
        "plan_template": str(paths.plan_template_path),
        "suggested_design_output": str(design_output),
        "suggested_plan_output": str(plan_output),
        "bundles": inputs.bundles,
        "paradigm_trace": inputs.paradigm_trace,
    }


def _emit_run_summary(output_dir: Path, bundles: list, paradigm_trace: dict[str, object]) -> None:
    """输出运行摘要（bundle 数量、sections 路径、paradigm trace）。"""
    _LOGGER.info(f"Wrote {len(bundles)} bundles to {output_dir / 'bundles'}")
    _LOGGER.info(f"Write generated sections and PLAN.md to {output_dir / 'sections'}")
    print_paradigm_trace(paradigm_trace)


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    paths = _resolve_paths(parser, args)

    spec_text = read_text(paths.spec)
    template_text = read_text(paths.template)
    plan_template = read_text(paths.plan_template_path)
    op_name = extract_op_name(spec_text)
    blocks = extract_top_level_blocks(spec_text)
    template_sections = extract_template_sections(template_text)
    req_text = requirements_excerpt(args.requirements.resolve() if args.requirements else None)
    design_output, plan_output = default_outputs(paths.spec, op_name)

    reset_output_dir(paths.output_dir, args.force)

    # ---- paradigm routing: inject references from this skill's references/ ----
    pr = compute_paradigm_routing(paths.spec, spec_text, op_name, paths.skill_dir)
    paradigm_trace = build_paradigm_trace(pr)

    bundles = write_all_bundles(
        BundleWriteInputs(
            output_dir=paths.output_dir,
            op_name=op_name,
            blocks=blocks,
            template_sections=template_sections,
            req_text=req_text,
            plan_template=plan_template,
            plan_output=plan_output,
            pr=pr,
        )
    )

    update_manifest(
        paths.output_dir,
        _run_manifest_values(
            ManifestValueInputs(
                paths=paths,
                args=args,
                op_name=op_name,
                outputs=(design_output, plan_output),
                bundles=bundles,
                paradigm_trace=paradigm_trace,
            )
        ),
    )
    _emit_run_summary(paths.output_dir, bundles, paradigm_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
