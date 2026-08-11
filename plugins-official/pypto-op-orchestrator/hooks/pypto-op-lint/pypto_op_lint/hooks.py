#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import subprocess

from . import checks  # noqa: F401
from .core import (
    GIT_BIN,
    MODE_ENV,
    OP_WORKSPACE_DIR,
    POST_EDIT_BLOCK_ENV,
    SPEC_FILE,
    TEST_COMMAND_PATTERN,
    CheckContext,
    Finding,
    _run_checks,
)
from .infer import (
    _build_context,
    _find_nearest_op_dir,
    _get_current_stage,
    _infer_op_dir,
    _infer_stage_from_artifacts,
    _infer_stage_from_filename,
    _load_hook_input,
    _rule_ids_for_filename,
)
from .observability import (
    _emit_gate_event,
    _output_hook_json,
)


def _post_edit_target(data: dict) -> tuple[CheckContext, str] | None:
    file_path = data.get("tool_input", {}).get("file_path", "")
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        return None
    basename = os.path.basename(file_path)
    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_filename(basename)
        if stage == 0:
            return None
    ctx = _build_context(op_dir, stage)
    if basename.endswith("_impl.py"):
        try:
            ctx.file_scope = os.path.relpath(os.path.abspath(file_path), op_dir)
        except ValueError:
            ctx.file_scope = basename
    return ctx, basename


def _split_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding], list[Finding], list[Finding]]:
    fails = [finding for finding in findings if finding.status == "FAIL"]
    error_fails = [finding for finding in fails if finding.severity in ("S0", "S1")]
    warns = [finding for finding in findings if finding.status == "WARN"]
    infos = [finding for finding in findings if finding.status == "INFO"]
    return fails, error_fails, warns, infos


def _finding_context(
    fails: list[Finding],
    warns: list[Finding],
    infos: list[Finding],
) -> str:
    sections: list[str] = []
    if fails:
        lines = [_format_finding(finding) for finding in fails]
        sections.append(
            "[pypto-op-lint] 以下规则违规，请立即修正后重新写入文件：\n"
            + "\n".join(lines)
            + "\n\n参考 pypto-op-develop 技能的 references/execution-constraints.md"
        )
    if warns:
        lines = [_format_finding(finding) for finding in warns]
        sections.append(
            "[pypto-op-lint] 以下提醒建议确认：\n"
            + "\n".join(lines)
            + "\n\n请确认以上提醒项是否需要处理。"
        )
    if infos:
        lines = [_format_finding(finding) for finding in infos]
        sections.append("[pypto-op-lint] 以下信息提示（不影响门禁）：\n" + "\n".join(lines))
    return "\n\n".join(sections)


def _blocking_reason(error_fails: list[Finding]) -> str:
    lines = [_format_finding(finding) for finding in error_fails]
    hint_lines = [
        f"  - {finding.rule_id}: {_rule_fix_hint(finding.rule_id)}"
        for finding in error_fails
    ]
    blocking_rules = sorted({finding.rule_id for finding in error_fails})
    footer = (
        "\n\n**⛔ 修正流程：阅读上方 fix_hints → 修复 file:line 指出的违规 → "
        "对【同一文件】重新执行 Write/Edit。不可使用 bash 绕过 lint，"
        "不可移动文件到其他路径。**"
    )
    return (
        "[pypto-op-lint] 产物写入后即时门禁未通过（S0/S1）：\n"
        + "\n".join(lines)
        + "\n\nblocking_rules: "
        + ", ".join(blocking_rules)
        + "\nfix_hints:\n"
        + "\n".join(hint_lines)
        + footer
    )


def hook_post_edit() -> int:
    """PostToolUse[Write|Edit] — 按文件类型 lint。

    默认开启“产物生成即阻断”模式：当检测到 S0/S1 FAIL 时返回 decision=block，
    由插件侧拦截本次工具调用。
    """
    os.environ[MODE_ENV] = "post-edit"
    target = _post_edit_target(_load_hook_input())
    if target is None:
        return 0
    ctx, basename = target
    # ── SPEC.md 冻结：Stage 1（需求规划）完成后锁定。──
    # 在 Stage 1-7 模型中，SPEC.md 在 Stage 1 产出后便不可修改。
    # 如需合法修订规格，orchestrator 必须先通过 state_transition 回退到 Stage 1。
    if basename == SPEC_FILE and ctx.stage is not None and ctx.stage >= 2:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=(
                f"[pypto-op-lint] {SPEC_FILE} 在 Stage 1 完成后已冻结。"
                "如需修订规格，请通过 state_transition 回退到 target_stage=1。"
            ),
        )
        return 0

    rule_ids = _rule_ids_for_filename(basename)
    if not rule_ids:
        return 0

    findings, _ = _run_checks(ctx, rule_ids)
    fails, error_fails, warns, infos = _split_findings(findings)
    if not fails and not warns and not infos:
        return 0
    context_msg = _finding_context(fails, warns, infos)
    strict_block = os.environ.get(POST_EDIT_BLOCK_ENV, "1") == "1"
    if strict_block and error_fails:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=_blocking_reason(error_fails),
            additionalContext=context_msg,
        )
        # 不返回非零，避免插件侧拿不到结构化结果；实际阻断由插件根据 decision 实施。
        return 0

    _output_hook_json(
        "PostToolUse", decision="allow", reason="", additionalContext=context_msg
    )
    return 0


def hook_post_bash() -> int:
    """PostToolUse[Bash] — 三态解析测试输出"""
    os.environ[MODE_ENV] = "post-bash"
    data = _load_hook_input()
    command = data.get("tool_input", {}).get("command", "")
    if not TEST_COMMAND_PATTERN.search(command):
        return 0

    stdout = data.get("tool_result", {}).get("stdout", "")
    stderr = data.get("tool_result", {}).get("stderr", "")
    exit_code = data.get("tool_result", {}).get("exit_code", 0)

    verdict = _parse_verdict(stdout, stderr, exit_code)
    detail = _verdict_detail(verdict)
    context_msg = (
        f"[pypto-op-lint parse-result] 确定性判定: {verdict}。{detail}。"
        "请以此结果为准，不要自行解读测试输出。"
    )
    _output_hook_json("PostToolUse", additionalContext=context_msg)
    return 0


def hook_pre_edit_backup() -> int:
    """PreToolUse[Write|Edit] — 状态文件保护 + Stage 5 cleanup 编辑 impl 前 git auto-commit"""
    data = _load_hook_input()
    file_path = data.get("tool_input", {}).get("file_path", "")

    # 拦截对 .orchestrator_state.json 的直接写入
    if os.path.basename(file_path) == ".orchestrator_state.json":
        _output_hook_json(
            "PreToolUse",
            decision="block",
            reason="禁止直接修改 .orchestrator_state.json，"
            "请通过 state_transition 工具操作阶段状态。",
        )
        return 0

    if not file_path.endswith("_impl.py"):
        return 0
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        return 0
    if _get_current_stage(op_dir) != 6:
        return 0

    impl_file = os.path.basename(file_path)
    try:
        _ensure_git_init(op_dir)
        subprocess.run(
            [_git_executable(), "add", impl_file],
            cwd=op_dir,
            check=True,
            capture_output=True,
        )
        attempt = _get_stage5_cleanup_attempt(op_dir)
        subprocess.run(
            [
                _git_executable(),
                "commit",
                "-m",
                f"backup: {impl_file} before stage6 attempt {attempt}",
            ],
            cwd=op_dir,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # git commit 失败（可能无变更），不拦截
    return 0


def _git_executable() -> str:
    if GIT_BIN:
        return GIT_BIN
    return "/usr/bin/git"


def _ensure_git_init(op_dir: str):
    """确保算子工作区根目录有 git 仓库"""
    current = op_dir
    while current and os.path.basename(current) != OP_WORKSPACE_DIR:
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    git_dir = current if os.path.basename(current) == OP_WORKSPACE_DIR else op_dir
    if not os.path.isdir(os.path.join(git_dir, ".git")):
        subprocess.run(
            [_git_executable(), "init"], cwd=git_dir, check=True, capture_output=True
        )
        subprocess.run(
            [_git_executable(), "add", "."],
            cwd=git_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [_git_executable(), "commit", "-m", "init: operator workspace"],
            cwd=git_dir,
            check=True,
            capture_output=True,
        )


def _get_stage5_cleanup_attempt(op_dir: str) -> int:
    """从 .orchestrator_state.json 获取 Stage 5 重试次数（cleanup 使用）"""
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("stage_retry_count", {}).get("5", 0)) + 1
    except (ValueError, FileNotFoundError):
        return 1


def hook_stop() -> int:
    """Stop — agent 结束前交付门禁"""
    os.environ[MODE_ENV] = "stop"
    data = _load_hook_input()
    cwd = data.get("cwd", os.getcwd())

    op_dir = _find_nearest_op_dir(cwd)
    if not op_dir:
        return 0

    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_artifacts(op_dir)
        if stage == 0:
            return 0
    ctx = _build_context(op_dir, stage)
    applicable = [r["id"] for r in ctx.rules if ctx.stage in r.get("stages", [])]
    findings, invocation_id = _run_checks(ctx, applicable)
    error_fails = []
    for finding in findings:
        if finding.status == "FAIL" and finding.severity in ("S0", "S1"):
            error_fails.append(finding)

    if error_fails:
        blocking_rules = [f.rule_id for f in error_fails]
        lines = [_format_finding(f) for f in error_fails]
        hint_lines = [f"  - {f.rule_id}: {_rule_fix_hint(f.rule_id)}" for f in error_fails]
        _emit_gate_event(ctx, blocked=True, blocking_rules=blocking_rules,
                         invocation_id=invocation_id)
        _output_hook_json(
            "Stop",
            decision="block",
            reason="[pypto-op-lint] 交付门禁未通过，存在 ERROR（S0/S1）级违规：\n"
            + "\n".join(lines)
            + "\n\nblocking_rules: "
            + ", ".join(blocking_rules)
            + "\nfix_hints:\n"
            + "\n".join(hint_lines)
            + "\n\ndocs_ref: pypto-op-lint 的 rules.json"
            + "\n\n**⛔ 门禁已阻断：请先修复上述 ERROR 级违规，再继续后续操作。"
            "修复后重新运行即可。**",
        )
        return 2
    _emit_gate_event(ctx, blocked=False, blocking_rules=[], invocation_id=invocation_id)
    return 0


def _format_finding(finding) -> str:
    """统一格式化单条 finding 输出：``  [OLxx][Sx] file:line message``。

    ``finding.file`` / ``finding.line`` 为可选字段；缺失时退化为
    ``  [OLxx][Sx] message``。
    """
    loc = ""
    f_path = getattr(finding, "file", None)
    f_line = getattr(finding, "line", None)
    if f_path:
        loc = f" {f_path}"
        if f_line:
            loc += f":{f_line}"
    return f"  [{finding.rule_id}][{finding.severity}]{loc} {finding.message}"


def _rule_fix_hint(rule_id: str) -> str:
    hints = {
        # D1 框架约束（impl 文件最常触发）
        "OL01": (
            "kernel 装饰器必须字面写成 @pypto.frontend.jit（或带参数 @pypto.frontend.jit(...)），"
            "每个 impl 文件须有且仅有 1 个（Layer J）；任何别名形式均被拒绝——包括 "
            "@pt.frontend.jit、@F.jit、@frontend.jit、@jit 等。"
            "import 必须用 `import pypto`，不要用 as 子句或 from-import"
        ),
        "OL02": "输出写回须用 y[:] = expr / pypto.move() / pypto.assemble()，不可写成 y = expr",
        "OL03": "kernel 内禁止使用 Python 原生 for/while 控制流，请改用 pypto.loop()",
        "OL04": (
            "JIT 入口及其同文件可达的 Layer I/H helper 内必须调用 "
            "pypto.set_vec_tile_shapes 或 pypto.set_cube_tile_shapes；"
            "允许 thin @pypto.frontend.jit 只委托 helper"
        ),
        "OL05": "kernel 的张量参数必须带 pypto.Tensor([...], pypto.DT_xxx) 类型注解",
        "OL06": "kernel 内禁用 Python 原生 min()/max()，请改用 pypto.minimum/maximum",
        "OL07": (
            "impl 文件顶层只能使用正规 `import pypto`；"
            "禁止 `import pypto as ...`、`import pypto.frontend as ...` "
            "和 `from pypto... import ...`"
        ),
        "OL08": "kernel 内不可调用 print/logging 等宿主侧函数",
        "OL25": "pypto.Tensor() 不可为空参数，须填写 shape 与 dtype，如 pypto.Tensor([N, M], pypto.DT_FP32)",
        "OL26": "kernel 参数必须张量在前、标量在后",
        # D3 三文件分离
        "OL16": "impl 文件禁止 import golden 模块；如需对照，请在 test 文件中导入",
        "OL17": "test 文件中禁止定义 @pypto.frontend.jit 函数",
        "OL18": "test 文件须同时 import impl wrapper 与 golden 函数",
        # D5 一致性
        "OL31": "在 DESIGN.md front matter 设置 dynamic_axes，并在 impl Tensor 注解使用 pypto.DYNAMIC",
        "OL34": (
            "在 SPEC.md front matter 的 p0_shapes 填写 P0 形状，并在 test 用例覆盖。"
            "格式: [[1024,128]] 或 [{x: [4,2560]}]（value 必须是 list，不支持 {B:4} 纯标量）"
        ),
        "OL39": "为 SPEC.md/DESIGN.md/API_REPORT.md 添加 front matter 块（--- 包裹）",
        "OL40": "补齐 front matter 必填字段并修正字段类型（list/dict）",
        "OL41": "删除代码文件中的 lint 门禁输出文本，确保仅保留可执行源码/文档内容",
        "OL43": "DESIGN.md 声明 dynamic_axes 时，所有 impl 文件的 kernel 内必须存在 pypto.loop()",
        "OL50": (
            "生产 wrapper 的显式参数必须严格等于 module_interfaces.yaml 的 "
            "primary_inputs 顺序；调试扩展走 **kwargs 或 _debug/ 产物，"
            "不进入生产 ABI"
        ),
        "OL51": (
            "每个 YAML 输出至少要有一个真实写回点：pypto.assemble(..., out)、"
            "out.move(...) 或 out[:] = ...；不要只重新绑定局部变量"
        ),
    }
    return hints.get(rule_id, "参考 rules.json 中该规则说明修复")


def _parse_verdict(stdout: str, stderr: str, exit_code: int) -> str:
    combined = f"{stdout}\n{stderr}"
    # FAIL 优先于 PASS：多 case 测试中部分失败应判定为整体失败
    if "[PRECISION_FAIL]" in combined:
        return "precision_fail"
    if "[PRECISION_PASS]" in combined:
        return "precision_pass"
    if exit_code != 0:
        return "runtime_error"
    return "no_marker"


def _verdict_detail(verdict: str) -> str:
    details = {
        "precision_pass": "输出中包含 [PRECISION_PASS] 标记，精度通过",
        "precision_fail": "输出中包含 [PRECISION_FAIL] 标记，精度失败",
        "runtime_error": "测试进程返回非零退出码，运行失败",
        "no_marker": "测试运行完毕但未检测到精度标记（[PRECISION_PASS]/[PRECISION_FAIL]）",
    }
    return details.get(verdict, f"未知判定结果: {verdict}")
