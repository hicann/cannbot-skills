# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase 2: Team AI 语义评测。

与 skill eval 共享相同的验证逻辑（validate_output、create_review_prompt 等），
差异在于沙箱创建方式：team 通过 init.sh 在沙箱中安装完整 team 环境。
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest

from conftest import (
    get_team_path, get_teams_with_evals, load_team_evals_md, REPO_ROOT,
    FRAMEWORK_DIR, SANDBOX_DIR,
    create_opencode_runner, _platform_matches,
)
from opencode_runner import OpencodeRunner
from sandbox_manager import SandboxManager

# 复用 skill eval 的所有验证逻辑
from test_skill_evals import (
    ValidationContext,
    ExpectationContext,
    ReviewPromptContext,
    validate_output,
    extract_ai_text,
    extract_ai_text_from_session,
    extract_reasoning,
    create_review_prompt,
    _run_review_session,
    _check_contains_pattern,
    _check_not_contains_pattern,
    _check_file_exists,
    _check_file_list,
    _check_file_contains,
    _parse_file_contains_pattern,
    _validate_expected_output,
    _validate_expectation,
    _validate_skill_activated,
    collect_generated_files,
    _check_token_budget,
    _validate_prompt,
    _log_eval_case_header,
    _collect_exec_output,
    _run_and_extract_text,
    _EvalInputs,
    _unpack_eval_inputs,
    _create_and_validate,
    FILE_BASED_HINT,
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _setup_team_eval_sandbox(sandbox_manager: SandboxManager, team_name: str,
                              eval_id, team_dir: Path,
                              timeout: Optional[int] = None):
    """为 team 创建沙箱并初始化 opencode runner。

    与 skill 不同：team 沙箱通过 init.sh 完整安装 team 环境，
    包括 AGENTS.md、skills/agents symlink、workflows、asc-devkit 等。
    """
    sandbox_path = sandbox_manager.create_team_sandbox(
        team_name, eval_id, REPO_ROOT, team_dir,
    )
    try:
        opencode_runner = create_opencode_runner(
            sandbox_manager, sandbox_path, timeout=timeout,
        )
    except Exception:
        # create_opencode_runner 失败时清理已创建的沙箱目录
        import shutil
        shutil.rmtree(str(sandbox_path), ignore_errors=True)
        raise
    return opencode_runner, sandbox_path


def _build_team_eval_test_case(team_name: str, team_dir: Optional[Path],
                                eval_item: Dict[str, Any]) -> Dict[str, Any]:
    """从单条 team eval 用例构造 pytest 参数化字典"""
    return {
        "team_name": team_name,
        "eval_mode": eval_item.get("eval_mode", "text"),
        "eval": eval_item,
        "team_dir": team_dir,
    }


def pytest_generate_tests(metafunc):
    if "team_eval_case" not in metafunc.fixturenames:
        return

    team_names = metafunc.config.getoption("--team", None)
    eval_id = metafunc.config.getoption("--eval-id", None)
    ascend_platforms = metafunc.config.getoption("--ascend-platform", None)

    test_cases: List[Dict[str, Any]] = []
    ids: List[str] = []

    for team_name in get_teams_with_evals():
        if team_names and team_name not in team_names:
            continue
        evals_data = load_team_evals_md(team_name)
        if not evals_data:
            continue
        team_dir = get_team_path(team_name)
        if team_dir is None:
            logger.warning("Team directory not found for %s, skipping eval cases", team_name)
            continue
        for eval_item in evals_data.get("evals", []):
            if eval_id and str(eval_item.get("id")) != str(eval_id):
                continue
            if not _platform_matches(ascend_platforms, eval_item):
                continue
            test_cases.append(_build_team_eval_test_case(team_name, team_dir, eval_item))
            ids.append(f"{team_name}::eval_{eval_item.get('id')}")

    if test_cases:
        metafunc.parametrize("team_eval_case", test_cases, ids=ids, scope="function")
    else:
        # 无匹配用例时，生成一个占位用例并立即跳过，避免 fixture 未定义的 ERROR
        dummy = {"_skip": True}
        metafunc.parametrize("team_eval_case", [dummy], ids=["no-matching-cases"], scope="function")


def test_team_eval_case(team_eval_case: Dict[str, Any], sandbox_manager: SandboxManager):
    """Team AI 语义评测 — 主测试函数。

    流程与 skill eval 一致：沙箱创建 → opencode 执行 → 输出验证 → token 检查。
    差异在于沙箱创建使用 create_team_sandbox()（通过 init.sh 安装 team）。
    """
    # 占位用例过滤：无匹配 case 时跳过
    if team_eval_case.get("_skip"):
        pytest.skip("No matching eval cases for the current filter")

    if os.environ.get("REPORT_ONLY") == "1":
        logger.info("[%s] REPORT_ONLY 模式，跳过测试执行 (eval %s)",
                    team_eval_case["team_name"], team_eval_case["eval"].get("id"))
        return

    eval_data = team_eval_case["eval"]
    if eval_data.get("disabled"):
        pytest.skip(f"Eval {eval_data.get('id')} marked as Disabled - skipping")

    # 构造与 skill eval 兼容的输入结构
    inputs = _unpack_eval_inputs({
        "skill_name": team_eval_case["team_name"],
        "eval_mode": team_eval_case.get("eval_mode", "text"),
        "eval": eval_data,
        "skill_dir": team_eval_case["team_dir"],
        "distractor_skill_dirs": [],
    })
    # 覆盖 skill_name → team_name（_unpack_eval_inputs 使用 skill_name）
    inputs.skill_name = team_eval_case["team_name"]

    team_dir = team_eval_case["team_dir"]
    opencode_runner, sandbox_path = _setup_team_eval_sandbox(
        sandbox_manager, inputs.skill_name, inputs.eval_id, team_dir,
        timeout=inputs.eval_data.get("timeout"),
    )

    _log_eval_case_header(inputs.skill_name, inputs.eval_id, inputs.prompt,
                          inputs.expected_output)

    # Team 执行：不传 skill 参数（opencode 从 AGENTS.md 自动发现 team）
    # 使用 "." 表示当前目录（沙箱根目录），其中 AGENTS.md 已由 init.sh 安装
    max_retries = int(os.environ.get("EVAL_EXEC_RETRIES", "1"))
    last_error = None
    best_session_file = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning("[RETRY %d/%d] 重试执行 team eval %s",
                           attempt, max_retries, inputs.eval_id)

        try:
            full_output, session_name, session_file, ai_text = _run_and_extract_text(
                opencode_runner, inputs.prompt, ".",
                inputs.skill_name, str(inputs.eval_id),
            )
        except AssertionError as e:
            last_error = e
            logger.warning("[EXEC FAIL] opencode 执行失败 (attempt %d): %s",
                           attempt, str(e)[:200])
            continue

        logger.debug("--- AI Response (team eval %s, attempt %d) ---",
                     inputs.eval_id, attempt)
        logger.debug(ai_text[:1000])
        logger.debug("--- End AI Response ---")

        _check_token_budget(inputs.eval_data, inputs.eval_id,
                            opencode_runner, session_name)

        try:
            _create_and_validate(opencode_runner, session_name, full_output,
                                 inputs, ai_text, sandbox_path)
            last_error = None
            best_session_file = session_file
            break  # 评测通过
        except AssertionError as e:
            last_error = e
            logger.warning("[VALIDATE FAIL] 评测不通过 (attempt %d): %s",
                           attempt, str(e)[:200])
            if not best_session_file:
                best_session_file = session_file

    if last_error is not None:
        raise last_error
    logger.info("Session file: %s", best_session_file)
