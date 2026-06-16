#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
Skill 脚本执行 Hook (PreToolUse 协议)。

对 Bash 工具调用进行拦截：
- 命中 INTERCEPTED_PATTERNS: 直接由本 hook 执行脚本，并以 PreToolUse 协议返回
  permissionDecision=deny + additionalContext，把执行结果（exit_code/stdout/stderr）
  注入回 agent 的上下文，阻止 harness 二次执行。
- 不命中: 输出 permissionDecision=allow，让 harness 正常执行（不在 hook 中重复执行）。

输入：从 stdin 读取 harness PreToolUse JSON：
    {"tool_name": "Bash", "tool_input": {"command": "..."}, ...}

输出：单行 PreToolUse JSON 到 stdout。
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path


_protocol_logger = logging.getLogger("skill_script_hook.protocol")
_protocol_logger.propagate = False
_protocol_logger.setLevel(logging.INFO)
_protocol_handler = logging.StreamHandler(sys.stdout)
_protocol_handler.setFormatter(logging.Formatter("%(message)s"))
_protocol_logger.addHandler(_protocol_handler)


# 每个条目: (interpreter_regex, script_basename_regex)
# 只匹配“某个解释器 + 脚本路径”这种真正的脚本调用，
# 不匹配命令字符串里仅出现脚本名的情况（如 cat / grep / echo JSON 等）。
_PY_INTERP = r"(?:python|python3|python3\.\d+)"
_SH_INTERP = r"(?:bash|sh)"

INTERCEPTED_PATTERNS = [
    (_PY_INTERP, r"validate_tilelang_impl\.py"),
    (_PY_INTERP, r"validate_ascendc_impl\.py"),
    (_SH_INTERP, r"evaluate_tilelang\.sh"),
    (_SH_INTERP, r"evaluate_ascendc\.sh"),
    (_PY_INTERP, r"performance\.py"),
    (_PY_INTERP, r"build_ascendc\.py"),
    (_PY_INTERP, r"verification_ascendc\.py"),
    (_PY_INTERP, r"verification_tilelang\.py"),
]

# 命令前缀允许的部分（环境变量赋值、cd ... &&、source ... &&）
# 这里只是用于推断"真实的脚本调用"，不需要覆盖所有 shell 语法。
_LEADING_PREFIX = (
    r"^\s*"
    r"(?:export\s+)?"                                 # 可选的 export 关键字
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"           # ENV=VAL ...
    r"(?:cd\s+\S+\s*&&\s*)?"                          # cd <dir> &&
    r"(?:export\s+)?"                                 # 可选的 export 关键字
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"           # 再次允许 ENV=VAL
    r"(?:&&\s+)*"                                     # 支持 && 链式连接
)

# 项目根目录：优先从环境变量获取，fallback 到 __file__ 计算
# 避免 hook 被以相对路径调用时 __file__ 解析错误
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def should_intercept(command: str) -> bool:
    # 仅当命令的"实际执行段"是 <interpreter> <path-to-script> 形式时才拦截。
    # 兼容：环境变量前缀、可选的 `cd <dir> &&` 前缀。
    for interp, script in INTERCEPTED_PATTERNS:
        # 路径里允许出现 / 和非空白字符；脚本名用 \b 边界
        pattern = (
            _LEADING_PREFIX
            + interp
            + r"\s+"
            + r"(?:\S*/)?"
            + script
            + r"(?:\s|$)"
        )
        if re.search(pattern, command):
            return True
    return False


def emit_pretooluse(permission_decision: str, *, reason: str = "", additional_context: str = ""):
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": permission_decision,
        }
    }
    if reason:
        output["hookSpecificOutput"]["permissionDecisionReason"] = reason
    if additional_context:
        output["hookSpecificOutput"]["additionalContext"] = additional_context
    _protocol_logger.info(json.dumps(output, ensure_ascii=False))


def truncate(text: str, limit: int = 8000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return f"{head}\n\n... [truncated {len(text) - limit} chars] ...\n\n{tail}"


def _run_command(command: str, cwd: str):
    """Execute a shell command and return (exit_code, stdout, stderr, duration_ms)."""
    start_time = time.time()
    try:
        import shlex
        cmd_parts = shlex.split(command)
        proc = subprocess.run(
            cmd_parts, shell=False, cwd=cwd,
            capture_output=True, text=True, timeout=1800,
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        exit_code = 124
        stdout = (e.stdout if e.stdout else "")
        stderr = (e.stderr if e.stderr else "")
        stderr += "\n[HOOK] 命令执行超时（30 分钟）"
    except Exception as e:
        exit_code = 1
        stdout = ""
        stderr = f"[HOOK] 执行异常: {type(e).__name__}: {e}"

    duration_ms = int((time.time() - start_time) * 1000)
    return exit_code, stdout, stderr, duration_ms

# ── Precision Gate (D-class enforcement) ────────────────────
_PRECISION_GATE_STATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".claude", "hooks", "state", "precision_gate.json",
)

# Patterns for D-class detection in evaluate_ascendc.sh output
_A_CLASS_ERROR_PATTERNS = [
    r"\berror:\s",           # compilation error
    r"\bfatal error:\s",     # fatal compilation error
    r"undefined reference",  # linker error
    r"Segmentation fault",
    r"core dumped",
    r"cannot find -l",       # library not found
    r"No such file or directory",
    r"CMake Error",
    r"make\[\d+\]: \*\*\*",  # make build error
]
_A_CLASS_PATTERN_RE = re.compile("|".join(_A_CLASS_ERROR_PATTERNS), re.IGNORECASE)

_D_CLASS_PASS_RE = re.compile(r"^\s*Result:\s*(pass|fail)\s*$", re.MULTILINE | re.IGNORECASE)
_D_CLASS_NUMERIC_RE = re.compile(r"max_abs_diff\s*=\s*[\d.e+\-]+|MERE\s*=\s*[\d.e+\-]+|matched_ratio\s*=\s*[\d.]+")

_EVAL_ASCENDC_SCRIPT_RE = re.compile(r"evaluate_ascendc\.sh")


def _is_evaluate_ascendc(command: str) -> bool:
    return bool(_EVAL_ASCENDC_SCRIPT_RE.search(command))


def _extract_output_dir(command: str) -> str:
    """Extract output_dir argument from evaluate_ascendc.sh command."""
    m = re.search(r"evaluate_ascendc\.sh\s+(\S+)", command)
    return m.group(1) if m else ""


def _classify_result(exit_code: int, stdout: str, stderr: str) -> str:
    """
    Classify evaluate_ascendc.sh result as "D" (precision failure),
    "A" (code/environment error), or "PASS".
    """
    combined = f"{stdout}\n{stderr}"

    # Check for PASS
    pass_matches = _D_CLASS_PASS_RE.findall(combined)
    if pass_matches:
        last_result = pass_matches[-1].lower()
        if last_result == "pass":
            return "PASS"

    # Check for A-class (compilation/runtime error)
    if _A_CLASS_PATTERN_RE.search(combined):
        return "A"

    # Check for D-class: has numeric diff output and Result: fail
    if _D_CLASS_NUMERIC_RE.search(combined):
        return "D"

    # Non-zero exit without clear A-class or D-class patterns → A
    if exit_code != 0:
        return "A"

    return "UNKNOWN"


def _ensure_precision_gate_dir() -> None:
    os.makedirs(os.path.dirname(_PRECISION_GATE_STATE), exist_ok=True)


def _read_precision_gate() -> dict:
    """Read existing precision gate state to preserve counters across iterations."""
    try:
        with open(_PRECISION_GATE_STATE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _update_precision_gate(classification: str, stdout: str, output_dir: str) -> None:
    """Write or clear precision gate state based on evaluate_ascendc.sh result.
    On D-class: resets edits_allowed/skill_just_called but PRESERVES
    stage/debug_call_count/tune_call_count across iterations.
    On PASS/A: full reset.
    """
    if classification == "D":
        # Extract failure summary (first few lines of case diffs)
        summary_lines = []
        for line in stdout.splitlines():
            if "max_abs_diff" in line or "Result:" in line:
                summary_lines.append(line.strip())
        failure_summary = "\n".join(summary_lines[:5])

        # Preserve stage and counters from previous iteration
        existing = _read_precision_gate()
        stage = existing.get("stage", "D1")
        debug_count = existing.get("debug_call_count", 0)
        tune_count = existing.get("tune_call_count", 0)

        # Auto-advance stage based on accumulated counts
        if debug_count >= 7:
            stage = "D2"

        state = {
            "d_class_active": True,
            "output_dir": output_dir,
            "edits_allowed": 0,
            "skill_just_called": False,
            "stage": stage,
            "debug_call_count": debug_count,
            "tune_call_count": tune_count,
            "failure_summary": failure_summary,
        }
    else:
        # PASS or A-class: full reset
        state = {"d_class_active": False, "edits_allowed": 0}

    _ensure_precision_gate_dir()
    with open(_PRECISION_GATE_STATE, "w") as fh:
        json.dump(state, fh)


def execute_intercepted(command: str) -> None:
    cwd = os.getcwd()
    if os.path.isdir(PROJECT_ROOT):
        cwd = PROJECT_ROOT

    exit_code, stdout, stderr, duration_ms = _run_command(command, cwd)

    # Update precision gate state if this is an evaluate_ascendc.sh call
    is_eval_ascendc = _is_evaluate_ascendc(command)
    classification = None
    if is_eval_ascendc:
        classification = _classify_result(exit_code, stdout, stderr)
        output_dir = _extract_output_dir(command)
        _update_precision_gate(classification, stdout, output_dir)

    additional_context = (
        f"[skill_script_hook intercepted execution]\n"
        f"command: {command}\n"
        f"cwd: {cwd}\n"
        f"exit_code: {exit_code}\n"
        f"duration_ms: {duration_ms}\n"
        f"--- stdout ---\n{truncate(stdout)}\n"
        f"--- stderr ---\n{truncate(stderr)}\n"
    )

    status = "成功" if exit_code == 0 else "失败"

    class_note = ""
    if is_eval_ascendc and classification:
        class_labels = {"D": "D类-精度不匹配", "A": "A类-代码/编译错误", "PASS": "通过"}
        label = class_labels.get(classification, classification)
        class_note = f" | 错误分类: {label}"
        if classification == "D":
            class_note += (
                " | [precision_gate] kernel 编辑已锁定，修改代码前必须先调用 "
                "ascendc-precision-debug 或 ascendc-precision-tuning Skill"
            )

    reason = (
        f"[Hook 拦截提示] 该命令命中 skill_script_hook 拦截规则，"
        f"已由 hook 代为执行（{status}，exit_code={exit_code}）。{class_note}"
        f"执行结果见下方 additionalContext，原命令已被替换为 no-op，不会重复执行。"
    )

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
            "additionalContext": additional_context,
            "updatedInput": {
                "command": f"echo '[hook-noop] original command was intercepted and executed by skill_script_hook'"
            },
        }
    }
    _protocol_logger.info(json.dumps(output, ensure_ascii=False))


def read_stdin_command():
    try:
        raw = sys.stdin.read()
    except Exception:
        return None, None
    if not raw or not raw.strip():
        return None, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    tool_name = data.get("tool_name")
    tool_input = data.get("tool_input") or {}
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    return tool_name, command


def main():
    tool_name, command = read_stdin_command()

    # 没有命令或不是 Bash —— allow，由 harness 处理
    if not command or tool_name != "Bash":
        emit_pretooluse("allow")
        return

    if should_intercept(command):
        execute_intercepted(command)
    else:
        # 不拦截 —— 让 harness 正常处理（不在此重复执行）
        emit_pretooluse("allow")


if __name__ == "__main__":
    main()
