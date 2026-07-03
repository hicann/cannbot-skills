#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Parse LOG.md status table and output current stage, next step, blocking items as JSON."""

import json
import logging
import os
import re
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

SUBAGENT_MAP = {
    "开发准备": "general",
    "需求分析": "ascendc-ops-architect",
    "1.2 需求分析": "ascendc-ops-architect",
    "spec生成": "ascendc-ops-architect",
    "1.2.5 spec 生成": "ascendc-ops-architect",
    "spec自审": "ascendc-ops-architect",
    "1.2.5R spec 评审": "ascendc-ops-architect",
    "CP1.5 spec确认": "ascendc-ops-architect",
    "方案设计": "ascendc-ops-architect",
    "1.3 方案设计": "ascendc-ops-architect",
    "1.3a 设计准备": "ascendc-ops-architect",
    "1.3b 切片分析": "ascendc-ops-architect",
    "1.3c 并行分段": "ascendc-ops-architect",
    "1.3d 组装校验": "ascendc-ops-architect",
    "方案评审": "ascendc-ops-architect",
    "1.4 测试设计": "ascendc-ops-tester",
    "测试设计": "ascendc-ops-tester",
    "测试设计评审": "ascendc-ops-tester",
    "骨架搭建": "ascendc-ops-developer",
    "穿刺验证": "ascendc-ops-developer",
    "核心路径UT": "ascendc-ops-developer",
    "策略整合": "ascendc-ops-developer",
    "Tiling分支UT": "ascendc-ops-developer",
    "全功能实现": "ascendc-ops-developer",
    "全覆盖UT": "ascendc-ops-developer",
    "汇合验证": "ascendc-ops-developer",
    "白盒测试": "ascendc-ops-tester",
    "C++标准用例": "ascendc-ops-tester",
    "C++多shape用例": "ascendc-ops-tester",
    "C++全量用例": "ascendc-ops-tester",
    "ST测试开发": "ascendc-ops-tester",
    "PyTorch ST测试": "ascendc-ops-tester",
    "测试工程师验收": "ascendc-ops-tester",
    "精度验收": "ascendc-ops-tester",
    "3.1 最终精度验收": "ascendc-ops-tester",
    "性能达标验收": "ascendc-ops-developer",
    "3.2 性能达标验收": "ascendc-ops-developer",
    "代码检视": "ascendc-ops-reviewer",
    "4.2 代码检视": "ascendc-ops-reviewer",
    "开发总结": "general",
    "4.3 开发总结": "general",
    "文档与示例": "general",
    "4.1 文档与示例": "general",
}

SCENE_MAP = {
    "1.1 开发准备": "dev_prep",
    "1.2 需求分析": "requirements_analysis",
    "1.2.5 spec 生成": "spec_generation",
    "1.2.5R spec 评审": "spec_review",
    "CP1.5 spec确认": "spec_confirmation",
    "1.3a 设计准备": "design_prep",
    "1.3b 切片分析": "tiling_analysis",
    "1.3c 并行分段": "parallel_sections",
    "1.3d 组装校验": "design_assembly",
    "1.4 测试设计": "test_design",
    "测试设计评审": "test_design_review",
}

STAGE_RULES = [
    ("阶段一：需求与设计", ("1.", "需求", "设计", "spec", "开发准备")),
    ("阶段二：开发", ("迭代", "骨架", "穿刺", "策略", "Tiling", "全功能", "全覆盖", "汇合")),
    ("阶段三：验收", ("3.", "精度", "性能", "验收")),
    ("阶段四：上库", ("4.", "上库", "检视", "总结", "文档")),
]


def _emit_json(obj):
    logging.info(json.dumps(obj, ensure_ascii=False, indent=2))


def extract_operator_name(log_path):
    parts = log_path.split("/")
    ops_idx = parts.index("operators") if "operators" in parts else -1
    if ops_idx >= 0 and ops_idx + 1 < len(parts):
        return parts[ops_idx + 1]
    return "unknown"


def extract_status_rows(content):
    rows = []
    for line in content.split("\n"):
        if not line.startswith("|"):
            continue
        if line.strip().startswith("|---") or line.strip().startswith("| --"):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= 2:
            step = cells[0]
            status = cells[1]
            rows.append({"step": step, "status": status})
    return rows


def extract_current_status_table(content):
    result = {}
    for line in content.split("\n"):
        if not line.startswith("|"):
            continue
        if line.strip().startswith("|---") or line.strip().startswith("| --"):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= 2:
            key = cells[0].replace("**", "").strip()
            val = cells[1].replace("**", "").strip()
            result[key] = val
    return result


def find_section_rows(content, section_header):
    lines = content.split("\n")
    start_idx = -1
    for i, line in enumerate(lines):
        if section_header in line:
            start_idx = i
            break
    if start_idx < 0:
        return []

    rows = []
    for line in lines[start_idx + 1:]:
        if line.startswith("##") or line.startswith("###"):
            if rows:
                break
            continue
        if not line.startswith("|"):
            continue
        if line.strip().startswith("|---") or line.strip().startswith("| --"):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= 2:
            step = cells[0].strip()
            status = cells[1].strip()
            rows.append({"step": step, "status": status})
    return rows


def clean_step_name(step):
    step = step.replace("├─", "").replace("└─", "").strip()
    step = re.sub(r"\s+", " ", step)
    return step


def _matches_step(step, keyword):
    if keyword.endswith("."):
        return step.startswith(keyword)
    return keyword in step


def determine_stage_name(step):
    for stage_name, keywords in STAGE_RULES:
        if any(_matches_step(step, k) for k in keywords):
            return stage_name
    return "未知阶段"


def _find_last_index(rows, predicate):
    for i in range(len(rows) - 1, -1, -1):
        if predicate(rows[i]["status"]):
            return i
    return -1


def _clean_rows(rows):
    return [
        {"step": clean_step_name(r["step"]), "status": r["status"]}
        for r in rows
    ]


def _read_log(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)
    return open(log_path, "r", encoding="utf-8").read()


def _build_result(operator_name, cleaned_rows, current_status):
    last_completed_idx = _find_last_index(cleaned_rows, lambda s: s.startswith("✅"))
    last_failed_idx = _find_last_index(cleaned_rows, lambda s: s.startswith("❌"))

    blocking_items = []
    if last_failed_idx >= 0:
        failed = cleaned_rows[last_failed_idx]
        blocking_items.append(f"步骤失败需重做: {failed['step']} ({failed['status']})")

    base = {
        "operator_name": operator_name,
        "blocking_items": blocking_items,
        "status_rows": cleaned_rows,
        "current_status_summary": current_status,
    }

    if last_completed_idx < 0:
        next_step = cleaned_rows[0]["step"] if cleaned_rows else "未知"
        base.update({
            "stage_name": "未开始",
            "last_completed_step": "",
            "last_completed_subagent": "",
            "next_step": next_step,
            "next_subagent": SUBAGENT_MAP.get(next_step, "unknown"),
            "next_scene": SCENE_MAP.get(next_step, ""),
        })
        return base

    last_completed_step = cleaned_rows[last_completed_idx]["step"]
    if last_completed_idx + 1 < len(cleaned_rows):
        next_step = cleaned_rows[last_completed_idx + 1]["step"]
    else:
        next_step = "已完成"

    base.update({
        "stage_name": determine_stage_name(last_completed_step),
        "last_completed_step": last_completed_step,
        "last_completed_subagent": SUBAGENT_MAP.get(last_completed_step, "unknown"),
        "next_step": next_step,
        "next_subagent": SUBAGENT_MAP.get(next_step, "unknown"),
        "next_scene": SCENE_MAP.get(next_step, ""),
    })
    return base


def main():
    if len(sys.argv) < 2:
        _emit_json({"error": "Usage: parse_log.py <log_path>"})
        sys.exit(1)

    log_path = sys.argv[1]
    try:
        content = _read_log(log_path)
    except FileNotFoundError:
        _emit_json({"error": f"LOG.md not found: {log_path}"})
        sys.exit(1)
    operator_name = extract_operator_name(log_path)

    four_stage_rows = find_section_rows(content, "四阶段进度")
    if not four_stage_rows:
        four_stage_rows = find_section_rows(content, "进度")
    if not four_stage_rows:
        four_stage_rows = extract_status_rows(content)

    cs_section = find_section_rows(content, "当前开发状态")
    current_status = {row["step"]: row["status"] for row in cs_section} if cs_section else {}

    cleaned_rows = _clean_rows(four_stage_rows)
    result = _build_result(operator_name, cleaned_rows, current_status)
    _emit_json(result)


if __name__ == "__main__":
    main()
