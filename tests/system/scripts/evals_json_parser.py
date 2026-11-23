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
Evals JSON Parser — 将 JSON 格式的评测用例文件解析为与 evals_parser.parse_evals_md()
相同结构的 dict，确保 ST 框架各模块（test_skill_evals.py、conftest.py、main.py 等）
无需修改即可直接使用 JSON 格式的评测数据。

格式说明：
  - 顶层字段: skill_name / team_name, eval_mode, evals[]
  - 每个用例: id, title, config{}, prompt, expected_output, files[], expectations[]
  - expectations: [{type, pattern, description}, ...]

与 parse_evals_md() 输出对齐：
  - JSON title          →  dict case_name
  - JSON config.xxx     →  平铺到 dict 顶层（max_tokens, disabled 等）
  - JSON expectations[] →  只保留 type + pattern（description 供 skill-creator 使用）
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
EVAL_MODES = ("text", "file_based", "cann_bench")
ASSERTION_TYPES = (
    "contains", "not_contains", "file_exists", "file_list", "file_contains", "skill_activated",
)


def _iter_case_expectations(item, case_index, *, require_fields, check_values=False, require_file_assertions=False):
    field = f"evals[{case_index}].expectations"
    case_id = item.get("id")
    case_label = f" (case id {case_id})" if type(case_id) is int else ""
    expectations = item.get("expectations", [])
    if not isinstance(expectations, list):
        yield f"{field}{case_label}: expected an array"
        return
    if require_file_assertions and not expectations:
        yield f"{field}{case_label}: enabled file_based case requires at least one source assertion"
    for index, expectation in enumerate(expectations):
        if not isinstance(expectation, dict):
            yield f"{field}[{index}]{case_label}: expected an object"
            continue
        if require_fields:
            missing = {"type", "pattern"} - expectation.keys()
            if missing:
                names = ", ".join(sorted(missing))
                yield f"{field}[{index}]{case_label}: missing {names}"
        if check_values:
            value_field = f"evals[{case_index}]{case_label}.expectations[{index}]"
            if "type" in expectation and expectation["type"] not in ASSERTION_TYPES:
                yield f"{value_field}.type: unsupported assertion type"
            if "pattern" in expectation and not _nonblank(expectation["pattern"]):
                yield f"{value_field}.pattern: expected a non-empty string"


def iter_invalid_expectations(data, *, require_fields=False):
    """Locate invalid assertion containers or entries without reporting their values."""
    if not isinstance(data, dict) or not isinstance(data.get("evals", []), list):
        return
    for case_index, item in enumerate(data.get("evals", [])):
        if isinstance(item, dict):
            yield from _iter_case_expectations(item, case_index, require_fields=require_fields)


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _id(value):
    return type(value) is int and -(2**31) <= value < 2**31


def _relative_file(value):
    return (
        _nonblank(value)
        and "\\" not in value
        and ":" not in value
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
        and all(part not in ("", ".", "..") for part in value.split("/"))
    )


def _case_errors(case, field):
    for key in ("title", "prompt", "expected_output"):
        if not _nonblank(case.get(key)):
            yield f"{field}.{key}: expected a non-empty string"
    if "timeout" in case:
        yield f"{field}.timeout: use config.timeout (integer seconds)"
    if "assertions" in case:
        yield f"{field}.assertions: use expectations for structured checks or expected_output for prose"
    config = case.get("config", {})
    if not isinstance(config, dict):
        yield f"{field}.config: expected an object (use {{}} when empty)"
    else:
        if "timeout" in config and (not _id(config["timeout"]) or config["timeout"] <= 0):
            yield f"{field}.config.timeout: expected integer seconds in 1..2147483647"
        if "eval_mode" in config and config["eval_mode"] not in EVAL_MODES:
            yield f"{field}.config.eval_mode: expected text, file_based or cann_bench"
        if "disabled" in config and type(config["disabled"]) is not bool:
            yield f"{field}.config.disabled: expected a boolean"
        for key in ("ascend_platforms", "distractor_skills"):
            if key in config and (not isinstance(config[key], list) or not all(map(_nonblank, config[key]))):
                yield f"{field}.config.{key}: expected an array of non-empty strings"
    files = case.get("files", [])
    if not isinstance(files, list):
        yield f"{field}.files: expected an array of relative path strings"
    else:
        for index, value in enumerate(files):
            if not _relative_file(value):
                yield f"{field}.files[{index}]: expected a normalized relative path"


def iter_pr_evals_errors(data):
    """Validate every raw case in a suite selected by PR UT."""
    if not isinstance(data, dict) or not isinstance(data.get("evals"), list):
        yield "evals: expected an array in a JSON object"
        return

    target = data.get("team_name") or data.get("skill_name")
    if not _nonblank(target):
        yield "expected a non-empty top-level skill_name or team_name"
    suite_mode = data.get("eval_mode", "text")
    if suite_mode not in EVAL_MODES:
        yield "eval_mode: expected text, file_based or cann_bench"

    seen_ids = set()
    for index, case in enumerate(data["evals"]):
        field = f"evals[{index}]"
        if not isinstance(case, dict):
            yield f"{field}: expected an object"
            continue
        case_id = case.get("id")
        if not _id(case_id):
            yield f"{field}.id: expected a signed 32-bit integer"
        else:
            field = f"{field} (case id {case_id})"
            if case_id in seen_ids:
                yield f"{field}.id: duplicate ID {case_id}"
            seen_ids.add(case_id)
        yield from _case_errors(case, field)
        config = case.get("config", {})
        require_file_assertions = (
            isinstance(config, dict)
            and config.get("eval_mode", suite_mode) == "file_based"
            and config.get("disabled") is not True
        )
        yield from _iter_case_expectations(
            case, index, require_fields=True, check_values=True, require_file_assertions=require_file_assertions,
        )


def _has_valid_containers(data) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("evals", []), list):
        return False
    return all(
        isinstance(item, dict) and isinstance(item.get("config", {}), dict)
        for item in data.get("evals", [])
    ) and next(iter_invalid_expectations(data), None) is None


def parse_evals_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    解析 evals.json，返回与 evals_parser.parse_evals_md() 相同结构的 dict。

    Returns:
        {
            "skill_name" / "team_name": "...",
            "eval_mode": "text",
            "target_type": "skill" | "team",
            "target_name": "...",
            "evals": [
                {
                    "id": 1,
                    "case_name": "...",
                    "prompt": "...",
                    "expected_output": "...",
                    "expectations": [{"type": "contains", "pattern": "..."}],
                    "max_tokens": 200000,
                    "max_tokens_by_model": {...},
                    "disabled": False,
                    "distractor_skills": [...],
                    "timeout": None,
                    "ascend_platforms": [...],
                    "eval_mode": "text",
                    "dim_thresholds": {},
                    "truncate_len": None,
                    "files": [...],
                }
            ]
        }
        解析失败返回 None
    """
    if not file_path.exists():
        return None

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Error reading %s: %s", file_path, e)
        return None

    if not _has_valid_containers(data):
        logger.error("Invalid evaluation JSON containers in %s", file_path)
        return None

    skill_name = data.get("skill_name", "")
    team_name = data.get("team_name", "")

    if not skill_name and not team_name:
        logger.warning("No skill_name or team_name in %s", file_path)
        return None

    eval_mode = data.get("eval_mode", "text")
    if not isinstance(eval_mode, str):
        logger.warning(
            "eval_mode is %s (not str), falling back to 'text'",
            type(eval_mode).__name__,
        )
        eval_mode = "text"
    else:
        eval_mode = eval_mode.strip().lower()
        if eval_mode not in EVAL_MODES:
            logger.warning("Invalid eval_mode '%s', falling back to 'text'", eval_mode)
            eval_mode = "text"

    raw_evals = data.get("evals", [])
    evals = [_parse_eval_item(item, eval_mode) for item in raw_evals]

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


def _parse_config_fields(config: dict, eval_case: dict, eval_mode: str) -> None:
    """
    将 config 字段平铺到 eval_case dict，只写非空值。
    保持与 evals_parser.py 行为一致：下游通过 .get(key, default) 获取默认值，
    key 不存在时才会触发默认值，因此不写入 None/False/{}。
    """
    eval_case["eval_mode"] = config.get("eval_mode", eval_mode)

    max_tokens = config.get("max_tokens")
    if max_tokens is not None:
        eval_case["max_tokens"] = max_tokens

    if config.get("max_tokens_by_model"):
        eval_case["max_tokens_by_model"] = config["max_tokens_by_model"]

    if config.get("disabled"):
        eval_case["disabled"] = True

    if config.get("distractor_skills"):
        eval_case["distractor_skills"] = config["distractor_skills"]

    timeout = config.get("timeout")
    if timeout is not None and timeout > 0:
        eval_case["timeout"] = timeout

    ascend_platforms = config.get("ascend_platforms")
    if ascend_platforms:
        eval_case["ascend_platforms"] = [p.upper() for p in ascend_platforms]

    if config.get("dim_thresholds"):
        eval_case["dim_thresholds"] = config["dim_thresholds"]

    truncate_len = config.get("truncate_len")
    if truncate_len is not None and truncate_len > 0:
        eval_case["truncate_len"] = truncate_len

    # cann_bench 模式特有字段
    if eval_case.get("eval_mode") == "cann_bench":
        _parse_cann_bench_config(config, eval_case, eval_case.get("id"))


def _parse_cann_bench_config(config: dict, eval_case: dict, case_id) -> None:
    """解析 cann_bench 模式专属配置项。"""
    eval_case["cann_bench_operator"] = config.get("cann_bench_operator", "").strip()
    eval_case["cann_bench_level"] = config.get("cann_bench_level", "level1").strip()
    eval_case["cann_bench_device"] = config.get("cann_bench_device", "0").strip()
    eval_case["cann_bench_no_perf"] = config.get("cann_bench_no_perf") or False

    warmup_str = config.get("cann_bench_warmup")
    if warmup_str is not None:
        try:
            eval_case["cann_bench_warmup"] = int(warmup_str)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid cann_bench_warmup '%s' in case %s, ignoring",
                warmup_str,
                case_id,
            )

    repeat_str = config.get("cann_bench_repeat")
    if repeat_str is not None:
        try:
            eval_case["cann_bench_repeat"] = int(repeat_str)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid cann_bench_repeat '%s' in case %s, ignoring",
                repeat_str,
                case_id,
            )


def _parse_eval_item(item: dict, eval_mode: str) -> dict:
    """解析单个 JSON eval item 为 ST 框架期望的 dict 格式。"""
    config = item.get("config", {})
    eval_case = {
        "id": item.get("id"),
        "case_name": item.get("title", ""),
        "prompt": item.get("prompt", ""),
        "expected_output": item.get("expected_output", ""),
        "expectations": _extract_expectations(item.get("expectations", [])),
        "files": item.get("files", []),
    }
    _parse_config_fields(config, eval_case, eval_mode)
    return eval_case


def _extract_expectations(raw_exps: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    从 JSON expectations 中提取 type + pattern。
    description 字段供 skill-creator 使用，ST 框架不需要。
    """
    results = []
    for exp in raw_exps:
        if "type" in exp and "pattern" in exp:
            results.append(
                {
                    "type": exp["type"],
                    "pattern": exp["pattern"],
                }
            )
    return results
