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
Documentation Gate Hook — enforces asc-devkit doc reading before kernel code edits.

Mode: --mode pre       (PreToolUse):  gates Edit/Write to kernel/**/*.{cpp,h}
      --mode post      (PostToolUse): tracks Read calls to asc-devkit/docs/ and asc-devkit/examples/
      --mode skill_post (PostToolUse): tracks precision Skill calls, grants precision edit budget

State file: .claude/hooks/state/doc_gate.json

Protocol: reads tool invocation JSON from stdin, outputs PreToolUse/PostToolUse response JSON to stdout.
"""
import json
import os
import sys
import logging

LOGGER = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    "CLAUDE_PLUGIN_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
STATE_DIR = os.path.join(PROJECT_ROOT, ".claude", "hooks", "state")
STATE_FILE = os.path.join(STATE_DIR, "doc_gate.json")
PRECISION_GATE_FILE = os.path.join(STATE_DIR, "precision_gate.json")

# Budget: how many kernel-file Edit/Write ops are allowed per coverage level.
# Proportional to foundational doc category coverage (0-4 categories).
BUDGET_BY_COVERAGE = {
    0: 0,    # blocked
    1: 10,   # minimal
    2: 20,   # partial
    3: 35,   # good
    4: 50,   # full (all 4 categories including example code)
}

# Files under these path segments are considered kernel source code.
KERNEL_SEGMENTS = ("/kernel/", "/op_kernel/", "/op_host/")
KERNEL_EXTENSIONS = (".cpp", ".h", ".hpp", ".cc")

# Files under these path segments are considered documentation.
DOC_SEGMENTS = ("/asc-devkit/docs/", "/asc-devkit/examples/")

# ── foundational doc categories ──────────────────────────────
# Each category has patterns (path substrings) and a human-readable description.
# An agent must read at least ONE doc matching each category to unlock full budget.
FOUNDATIONAL_CATEGORIES = {
    "矢量编程流水线指南": {
        "patterns": [
            "/guide/算子实践参考/SIMD算子实现/矢量编程/",
        ],
        "desc": "CopyIn→Compute→CopyOut 正确流水线模式",
        "example": "asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md",
    },
    "UB缓冲区/TBuf-TQue管理指南": {
        "patterns": [
            "/guide/算子实践参考/SIMD算子实现/矢量编程/TBuf",
            "/guide/编程指南/编程模型/AI-Core-SIMD编程/基于TPipe-TQue框架编程/",
        ],
        "desc": "UB 临时缓冲区正确分配与 TQue/TBuf 使用模式",
        "example": "asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/TBuf的使用.md",
    },
    "API参考文档": {
        "patterns": [
            "/api/SIMD-API/",
            "/api/tensor_api/",
        ],
        "desc": "所用 API 的完整签名、dtype 支持矩阵、参数约束",
        "example": "asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopyPad(ISASI).md",
    },
    "官方示例代码": {
        "patterns": [
            "/asc-devkit/examples/01_simd_cpp_api/",
        ],
        "desc": "官方 CopyIn→Compute→CopyOut 完整 kernel 实现，验证 TQue 管道的正确用法",
        "example": (
            "asc-devkit/examples/01_simd_cpp_api/02_features/00_compilation/"
            "custom_op/op_kernel/add_custom/add_custom_kernel.cpp"
        ),
    },
}


# ── helpers ──────────────────────────────────────────────────
def ensure_state_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)


def read_state() -> dict:
    try:
        with open(STATE_FILE) as fh:
            state = json.load(fh)
    except Exception:
        return {"docs_read": [], "edits_remaining": 0}

    # Migrate old format (single last_doc_read → list docs_read)
    if "last_doc_read" in state and "docs_read" not in state:
        old = state.pop("last_doc_read")
        state["docs_read"] = [old] if old else []
    if "docs_read" not in state:
        state["docs_read"] = []
    if "edits_remaining" not in state:
        state["edits_remaining"] = 0

    return state


def write_state(state: dict) -> None:
    ensure_state_dir()
    with open(STATE_FILE, "w") as fh:
        json.dump(state, fh)


def is_kernel_file(file_path: str) -> bool:
    """Check whether file_path is a kernel source file under output_dir/kernel/."""
    if not file_path:
        return False
    if not file_path.endswith(KERNEL_EXTENSIONS):
        return False
    normalized = os.path.normpath(file_path)
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return any(seg in normalized for seg in KERNEL_SEGMENTS)


def is_doc_file(file_path: str) -> bool:
    """Check whether file_path is an asc-devkit documentation or example."""
    if not file_path:
        return False
    return any(seg in file_path for seg in DOC_SEGMENTS)


def get_covered_categories(docs_read: list) -> set:
    """Return the set of foundational category names that have at least one doc read."""
    covered = set()
    for name, cat in FOUNDATIONAL_CATEGORIES.items():
        for doc_path in docs_read:
            if any(pattern in doc_path for pattern in cat["patterns"]):
                covered.add(name)
                break
    return covered


def get_missing_categories(docs_read: list) -> dict:
    """Return {name: category_dict} for categories that have zero docs read."""
    covered = get_covered_categories(docs_read)
    return {name: cat for name, cat in FOUNDATIONAL_CATEGORIES.items() if name not in covered}


def emit(response: dict) -> None:
    LOGGER.info(json.dumps(response, ensure_ascii=False))


# ── precision gate helpers (unchanged) ───────────────────────
def read_precision_state() -> dict:
    try:
        with open(PRECISION_GATE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {"d_class_active": False, "edits_allowed": 0}


def write_precision_state(state: dict) -> None:
    ensure_state_dir()
    with open(PRECISION_GATE_FILE, "w") as fh:
        json.dump(state, fh)


def check_precision_gate() -> tuple:
    """Check precision gate with D1/D2 stage enforcement.

    Returns (denied: bool, reason: str | None).
    """
    state = read_precision_state()
    if not state.get("d_class_active", False):
        return False, None

    remaining = state.get("edits_allowed", 0)
    if remaining > 0:
        state["edits_allowed"] = remaining - 1
        write_precision_state(state)
        return False, None

    stage = state.get("stage", "D1")
    debug_count = state.get("debug_call_count", 0)
    tune_count = state.get("tune_call_count", 0)
    total = debug_count + tune_count

    if total >= 12:
        return True, (
            "[Precision Gate] TERMINAL: Phase 4 D 类精度修复已耗尽全部 12 次机会\n"
            f"  D-1 (ascendc-precision-debug): {min(debug_count, 7)}/7 次\n"
            f"  D-2 (ascendc-precision-tuning): {min(tune_count, 5)}/5 次\n"
            "\nPhase 4 失败，请进入 Phase 7 记录 trace。"
        )

    if stage == "D1":
        used = debug_count
        max_calls = 7
        budget = 7
        required_skill = "ascendc-precision-debug"
        stage_info = f"D-1 阶段: ascendc-precision-debug, 已用 {used}/{max_calls} 次"
        if debug_count >= 7:
            stage_info += "\n[D-1 已耗尽，下次请调用 ascendc-precision-tuning 进入 D-2 深度审计]"
    else:
        used = tune_count
        max_calls = 5
        budget = 5
        required_skill = "ascendc-precision-tuning"
        stage_info = f"D-2 阶段: ascendc-precision-tuning, 已用 {used}/{max_calls} 次"

    return True, (
        "[Precision Gate] BLOCKED: D 类精度失败禁止直接修改 kernel 代码\n"
        "\n"
        "evaluate_ascendc.sh 判定当前结果为 D 类（精度不匹配）。\n"
        f"{stage_info}\n"
        "\n"
        f"你必须先调用 {required_skill} Skill 进行结构化根因分析：\n"
        "  1. 使用 Skill 工具调用上述 skill\n"
        "  2. 等待 Skill 返回诊断结论和修复建议\n"
        f"  3. Skill 调用后自动授予 {budget} 次编辑预算用于实施修复\n"
        "\n"
        f"最近 D 类失败摘要:\n{state.get('failure_summary', '')}"
    )


def _deny_doc_gate(coverage, missing, budget):
    """Build and emit the doc gate denial message."""
    lines = [
        "[Doc Gate] BLOCKED: 未完成必备文档查阅就尝试修改 kernel 代码",
        "",
        f"已覆盖 {coverage}/4 类必备文档。缺少以下类别：",
        "",
    ]
    for i, (name, cat) in enumerate(missing.items(), 1):
        lines.append(f"  [{i}] {name} — {cat['desc']}")
        lines.append(f"      示例: {cat['example']}")
        lines.append("")

    lines.append("请先完成上述文档的查阅，然后再修改 kernel 代码。")
    lines.append(f"当前覆盖率: {coverage}/4 → 编辑预算: {budget}")

    emit({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "\n".join(lines),
        }
    })


def _consume_edit_and_allow(state, coverage, budget, covered):
    """Consume one edit from budget and emit allow decision."""
    remaining = state.get("edits_remaining", budget)
    if coverage == 4 and remaining > budget:
        remaining = budget
    state["edits_remaining"] = max(remaining - 1, 0)
    write_state(state)

    covered_names = ", ".join(sorted(covered))
    emit({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": (
                f"[Doc Gate] 全部 {coverage}/4 类必备文档已覆盖 ({covered_names}), "
                f"剩余编辑预算: {state['edits_remaining']}"
            ),
        }
    })


# ── handlers ─────────────────────────────────────────────────
def handle_pre(tool_name: str, tool_input: dict) -> None:
    """PreToolUse: gate Edit/Write to kernel files.

    Enforces two independent gates:
    1. Doc Gate — foundational doc categories must be covered
    2. Precision Gate — D-class precision failures require structured debugging
    """
    if tool_name not in ("Edit", "Write"):
        emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}})
        return

    file_path = tool_input.get("file_path", "")
    if not is_kernel_file(file_path):
        emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}})
        return

    state = read_state()
    docs_read = state.get("docs_read", [])
    covered = get_covered_categories(docs_read)
    missing = get_missing_categories(docs_read)
    coverage = len(covered)
    budget = BUDGET_BY_COVERAGE.get(coverage, 0)

    if coverage < 4:
        _deny_doc_gate(coverage, missing, budget)
        return

    denied, precision_reason = check_precision_gate()
    if denied:
        emit({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": precision_reason,
            }
        })
        return

    _consume_edit_and_allow(state, coverage, budget, covered)


def handle_post(tool_name: str, tool_input: dict) -> None:
    """PostToolUse: track doc reads, grant edit budget proportionally."""
    if tool_name != "Read":
        emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "permissionDecision": "allow"}})
        return

    file_path = tool_input.get("file_path", "")
    if not is_doc_file(file_path):
        emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "permissionDecision": "allow"}})
        return

    state = read_state()
    docs_read = state.get("docs_read", [])

    if file_path not in docs_read:
        docs_read.append(file_path)
        state["docs_read"] = docs_read

    # Grant budget based on current coverage
    covered = get_covered_categories(docs_read)
    coverage = len(covered)
    budget = BUDGET_BY_COVERAGE.get(coverage, 0)
    missing = get_missing_categories(docs_read)

    # Reset budget when coverage level changes (don't accumulate)
    old_remaining = state.get("edits_remaining", 0)
    if coverage == 4:
        state["edits_remaining"] = budget
    elif old_remaining < budget:
        state["edits_remaining"] = budget

    write_state(state)

    if coverage < 4:
        missing_lines = "\n".join(
            f"  - {name}: {cat['example']}"
            for name, cat in missing.items()
        )
        msg = (
            f"[Doc Gate] 文档已记录 ({coverage}/4 类已覆盖)，编辑预算: {budget}\n"
            f"\n还需查阅:\n{missing_lines}"
        )
    else:
        covered_names = ", ".join(sorted(covered))
        msg = (
            f"[Doc Gate] 全部 {coverage}/4 类必备文档已覆盖 ({covered_names})\n"
            f"编辑预算: {budget} 次"
        )

    emit({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "permissionDecision": "allow",
            "additionalContext": msg,
        }
    })


def handle_skill_post(tool_name: str, tool_input: dict) -> None:
    """PostToolUse for Skill: track precision skill calls and grant edit budget."""
    if tool_name != "Skill":
        emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "permissionDecision": "allow"}})
        return

    skill_name = tool_input.get("skill", "")
    is_precision = skill_name in ("ascendc-precision-debug", "ascendc-precision-tuning")
    if not is_precision:
        emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "permissionDecision": "allow"}})
        return

    state = read_precision_state()
    if not state.get("d_class_active", False):
        emit({"hookSpecificOutput": {"permissionDecision": "allow"}})
        return

    if skill_name == "ascendc-precision-debug":
        state["debug_call_count"] = state.get("debug_call_count", 0) + 1
        budget = 7
        if state["debug_call_count"] >= 7:
            state["stage"] = "D2"
    elif skill_name == "ascendc-precision-tuning":
        state["tune_call_count"] = state.get("tune_call_count", 0) + 1
        budget = 5
        state["stage"] = "D2"

    state["skill_just_called"] = True
    state["edits_allowed"] = budget
    write_precision_state(state)

    emit({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "permissionDecision": "allow",
            "additionalContext": (
                f"[Precision Gate] 检测到 {skill_name} Skill 调用完成。\n"
                f"已授予 {budget} 次 kernel 编辑预算。请根据 Skill 的修复建议修改代码。"
            ),
        }
    })


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    args = sys.argv[1:]
    mode = "pre"
    for a in args:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]
        elif a == "--mode" or a == "-m":
            idx = args.index(a)
            if idx + 1 < len(args):
                mode = args[idx + 1]

    raw = sys.stdin.read()
    if not raw or not raw.strip():
        emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}})
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}})
        return

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}

    if mode == "post":
        handle_post(tool_name, tool_input)
    elif mode == "skill_post":
        handle_skill_post(tool_name, tool_input)
    else:
        handle_pre(tool_name, tool_input)


if __name__ == "__main__":
    main()
