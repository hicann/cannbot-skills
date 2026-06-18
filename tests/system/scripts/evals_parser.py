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


def _split_cases(content: str) -> List[Dict[str, Any]]:
    """按 # Case <id>: <name> 标题拆分内容为单个用例块。

    返回的每个 dict 包含 case_id、case_name 和 raw_block，
    消除了 parse_evals_md 中 split 与 findall 双路径解析的不一致风险。
    """
    # 去掉 frontmatter
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].lstrip("\n")

    # 按 # Case 标题拆分（用捕获组保留标题行）
    raw_parts = re.split(r'^# Case (\d+): (.+)$', body, flags=re.MULTILINE)

    # raw_parts 格式: [before_first_case, id1, name1, block1, id2, name2, block2, ...]
    # 第一个元素（before_first_case）跳过
    cases = []
    i = 1
    while i + 2 <= len(raw_parts):
        try:
            case_id = int(raw_parts[i].strip())
        except (ValueError, TypeError):
            case_id = len(cases) + 1
        case_name = raw_parts[i + 1].strip()
        block = (raw_parts[i + 2] or "").strip()
        if block:  # 仅保留有内容的用例块
            cases.append({
                "case_id": case_id,
                "case_name": case_name,
                "raw_block": block,
            })
        i += 3
    return cases


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
    return expectations


def _parse_max_tokens_config(case_config: Dict[str, str], eval_case: Dict[str, Any],
                               case_id: int) -> None:
    """解析 max_tokens 和 max_tokens_by_model 配置（从 case_config 读取）。"""
    max_tokens_str = case_config.get("max tokens", "")
    if max_tokens_str:
        try:
            eval_case["max_tokens"] = int(max_tokens_str)
        except ValueError:
            logger.warning("Invalid max_tokens '%s' in case %d, ignoring", max_tokens_str, case_id)

    max_tokens_by_model = {}
    for key, value in case_config.items():
        m = re.match(r'^max tokens \(([^)]+)\)$', key)
        if m:
            model_name = m.group(1).strip()
            try:
                max_tokens_by_model[model_name] = int(value)
            except ValueError:
                logger.warning("Invalid max_tokens for model '%s': %s in case %d, ignoring",
                               model_name, value, case_id)
    if max_tokens_by_model:
        eval_case["max_tokens_by_model"] = max_tokens_by_model


def _parse_case_config(case_config: Dict[str, str], case_id: int,
                       default_eval_mode: str) -> Dict[str, Any]:
    """解析 Config 块中的配置项，返回已填充的 eval_case 字典片段。"""
    eval_case: Dict[str, Any] = {}

    case_eval_mode = case_config.get("eval mode", default_eval_mode)
    if case_eval_mode not in ("text", "file_based"):
        logger.warning("Invalid case eval_mode '%s' in case %d, falling back to '%s'",
                       case_eval_mode, case_id, default_eval_mode)
        case_eval_mode = default_eval_mode
    eval_case["eval_mode"] = case_eval_mode

    _parse_max_tokens_config(case_config, eval_case, case_id)

    timeout_str = case_config.get("timeout", "")
    if timeout_str:
        try:
            timeout_val = int(timeout_str)
            if timeout_val > 0:
                eval_case["timeout"] = timeout_val
            else:
                logger.warning("Invalid timeout '%s' (must be > 0) in case %d, ignoring",
                               timeout_str, case_id)
        except ValueError:
            logger.warning("Invalid timeout '%s' in case %d, ignoring", timeout_str, case_id)

    disabled_raw = case_config.get("disabled", "").strip().lower()
    if disabled_raw in ("true", "yes", "1"):
        eval_case["disabled"] = True

    truncate_len_str = case_config.get("truncate length", "")
    if truncate_len_str:
        try:
            truncate_len_val = int(truncate_len_str)
            if truncate_len_val > 0:
                eval_case["truncate_len"] = truncate_len_val
            else:
                logger.warning("Invalid truncate_len '%s' (must be > 0) in case %d, ignoring",
                               truncate_len_str, case_id)
        except ValueError:
            logger.warning("Invalid truncate_len '%s' in case %d, ignoring",
                           truncate_len_str, case_id)

    distractor_raw = case_config.get("distractor skills", "")
    distractor_skills = [s.strip() for s in distractor_raw.split(";") if s.strip()]
    eval_case["distractor_skills"] = distractor_skills if distractor_skills else []

    ascend_platform_raw = case_config.get("ascend platform", "")
    ascend_platforms = [p.strip().upper() for p in ascend_platform_raw.split(";") if p.strip()]
    eval_case["ascend_platforms"] = ascend_platforms

    dim_thresholds = _parse_dim_thresholds(case_config, case_id)
    if dim_thresholds:
        eval_case["dim_thresholds"] = dim_thresholds

    return eval_case


def _parse_dim_thresholds(case_config: Dict[str, str], case_id: int) -> Dict[str, int]:
    """解析维度阈值配置项，返回 {维度名: 阈值} 字典。"""
    dim_thresholds: Dict[str, int] = {}
    _dim_threshold_config_keys = {
        "覆盖度阈值": "覆盖度",
        "准确性阈值": "准确性",
        "质量阈值": "质量",
        "token阈值": "Token",
    }
    for config_key, dim_name in _dim_threshold_config_keys.items():
        val = case_config.get(config_key)
        if val is not None:
            try:
                dim_thresholds[dim_name] = int(val)
            except ValueError:
                logger.warning("Invalid dimension threshold '%s=%s' in case %d, ignoring",
                               config_key, val, case_id)
    return dim_thresholds


def _parse_single_case(block: str, case_id: int, case_name: str,
                       default_eval_mode: str) -> Dict[str, Any]:
    """解析单个评测用例块"""
    sections = _parse_sections(block)
    raw_config = sections.get("config", "")
    case_config = _parse_config_lines(raw_config)

    eval_case: Dict[str, Any] = {"id": case_id, "case_name": case_name}
    eval_case.update(_parse_case_config(case_config, case_id, default_eval_mode))
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
    team_name = frontmatter.get("team_name", "")

    if not skill_name and not team_name:
        logger.warning("No skill_name or team_name in frontmatter of %s", file_path)
        return None

    eval_mode = frontmatter.get("eval_mode", "text").strip().lower()
    if eval_mode not in ("text", "file_based"):
        logger.warning("Invalid eval_mode '%s', falling back to 'text'", eval_mode)
        eval_mode = "text"

    cases = _split_cases(content)

    evals: List[Dict[str, Any]] = []
    for case_info in cases:
        evals.append(_parse_single_case(
            case_info["raw_block"], case_info["case_id"],
            case_info["case_name"], eval_mode,
        ))

    result: Dict[str, Any] = {
        "eval_mode": eval_mode,
        "evals": evals,
    }
    if team_name:
        result["team_name"] = team_name
        result["target_type"] = "team"
        result["target_name"] = team_name
    else:
        result["skill_name"] = skill_name
        result["target_type"] = "skill"
        result["target_name"] = skill_name
    return result
