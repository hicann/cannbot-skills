# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Evals Markdown Parser — 将 MD 格式的评测用例文件解析为 dict

格式说明：
  - 文件使用 YAML frontmatter 定义 skill_name
  - 每个用例以 # Case <id>: <name> 标题开头
  - ## Prompt / ## Expected Output / ## Expectations 三个章节
  - Expectations 以无序列表书写: - [type] pattern
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

EXPECTED_SECTIONS = {"prompt", "expected output", "expectations"}


def _parse_frontmatter(content: str) -> dict:
    """从 MD 内容中提取 YAML frontmatter"""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(content[3:end])
    except yaml.YAMLError:
        logger.warning("Failed to parse frontmatter YAML")
        return {}


def _parse_expectation_line(line: str) -> Optional[Dict[str, str]]:
    """解析单行 expectation: - [contains] pattern"""
    line = line.strip()
    m = re.match(r'^- *\[([a-z_]+)\] (.+)$', line)
    if not m:
        return None
    return {"type": m.group(1), "pattern": m.group(2).strip()}


def _parse_config_lines(content: str) -> Dict[str, str]:
    """解析 ## Config 段的键值对: - Key: value"""
    config = {}
    for line in content.strip().split("\n"):
        m = re.match(r'^- *([^:]+):\s*(.+)$', line.strip())
        if m:
            config[m.group(1).strip().lower()] = m.group(2).strip()
    return config


def _split_cases(content: str) -> List[str]:
    """按 # Case <id>: <name> 标题拆分内容为单个用例块"""
    # 去掉 frontmatter
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].lstrip("\n")

    # 按 # Case 标题拆分
    case_blocks = re.split(r'^# Case \d+:.*$', body, flags=re.MULTILINE)
    # 第一个 split 结果在 # Case 之前（无意义），去掉
    return [b.strip() for b in case_blocks if b.strip()]


def _parse_case_title(line: str) -> tuple:
    """从 '# Case 1: 用例名称' 中提取 (id, name)"""
    m = re.match(r'^# Case (\d+): (.+)$', line.strip())
    if not m:
        return (0, "")
    return (int(m.group(1)), m.group(2).strip())


def _parse_sections(block: str) -> Dict[str, str]:
    """将用例块按 ## <Section> 标题分割为 {section_name: content}"""
    sections = {}
    # 按 ## 标题分割
    parts = re.split(r'^## ', block, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 第一行是标题名，之后是内容
        lines = part.split("\n", 1)
        section_name = lines[0].strip().lower()
        content = lines[1].strip() if len(lines) > 1 else ""
        sections[section_name] = content
    return sections


def _parse_expectations(exp_raw: str) -> List[Dict[str, str]]:
    """解析 expectations 文本块为结构化列表"""
    expectations = []
    for line in exp_raw.split("\n"):
        parsed = _parse_expectation_line(line)
        if parsed:
            expectations.append(parsed)
        elif line.strip():
            logger.debug("Unparseable expectation line: %s", line.strip())
    return expectations


def _parse_single_case(block: str, case_id: int, case_name: str,
                       default_eval_mode: str) -> Dict[str, Any]:
    """解析单个评测用例块"""
    eval_case: Dict[str, Any] = {"id": case_id, "case_name": case_name}
    sections = _parse_sections(block)

    raw_config = sections.get("config", "")
    case_config = _parse_config_lines(raw_config)

    case_eval_mode = case_config.get("eval mode", default_eval_mode)
    if case_eval_mode not in ("text", "file_based"):
        logger.warning("Invalid case eval_mode '%s' in case %d, falling back to '%s'",
                       case_eval_mode, case_id, default_eval_mode)
        case_eval_mode = default_eval_mode
    eval_case["eval_mode"] = case_eval_mode

    eval_case["prompt"] = sections.get("prompt", "")
    eval_case["expected_output"] = sections.get("expected output", "")
    eval_case["expectations"] = _parse_expectations(sections.get("expectations", ""))
    return eval_case


def parse_evals_md(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    解析 MD 评测用例文件

    Returns:
        {
            "skill_name": "...",
            "eval_mode": "text",
            "evals": [
                {"id": 1, "case_name": "...", "prompt": "...",
                 "expected_output": "...", "expectations": [...]}
            ]
        }
        解析失败返回 None
    """
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
    except IOError as e:
        logger.error("Error reading %s: %s", file_path, e)
        return None

    frontmatter = _parse_frontmatter(content)
    skill_name = frontmatter.get("skill_name", "")

    eval_mode = frontmatter.get("eval_mode", "text").strip().lower()
    if eval_mode not in ("text", "file_based"):
        logger.warning("Invalid eval_mode '%s', falling back to 'text'", eval_mode)
        eval_mode = "text"

    bullet_titles = re.findall(r'^# Case (\d+): (.+)$', content, re.MULTILINE)
    case_blocks = _split_cases(content)

    evals: List[Dict[str, Any]] = []
    for idx, block in enumerate(case_blocks):
        case_id = int(bullet_titles[idx][0]) if idx < len(bullet_titles) else (idx + 1)
        case_name = bullet_titles[idx][1] if idx < len(bullet_titles) else ""
        evals.append(_parse_single_case(block, case_id, case_name, eval_mode))

    return {
        "skill_name": skill_name,
        "eval_mode": eval_mode,
        "evals": evals,
    }
