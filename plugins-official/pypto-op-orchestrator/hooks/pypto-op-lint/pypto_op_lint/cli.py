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

import argparse

from . import checks  # noqa: F401
from .core import (
    CONSISTENCY_RULE_IDS,
    GOLDEN_RULE_IDS,
    IMPL_RULE_IDS,
    TEST_RULE_IDS,
    Finding,
    _has_error_fail,
    _run_checks,
)
from .hooks import hook_post_bash, hook_post_edit, hook_pre_edit_backup, hook_stop
from .infer import _build_context
from .observability import (
    _print_findings,
)


def _cmd_run(findings: list[Finding]) -> int:
    _print_findings(findings)
    return 2 if _has_error_fail(findings) else 0


def cmd_lint_impl(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, IMPL_RULE_IDS)
    return _cmd_run(findings)


def cmd_lint_golden(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, GOLDEN_RULE_IDS)
    return _cmd_run(findings)


def cmd_lint_test(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, TEST_RULE_IDS)
    return _cmd_run(findings)


def cmd_lint_consistency(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, CONSISTENCY_RULE_IDS)
    return _cmd_run(findings)


def cmd_check_gate(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    gate_rules: list[str] = []
    for rule in ctx.rules:
        # 门禁检查应覆盖当前 stage 的全部交付规则，而不仅是 gate。
        # 否则会出现 Stage 5/6/7 对 impl/test 关键 S1 规则不阻断的问题。
        if rule.get("target") not in ("gate", "impl", "test", "golden"):
            continue
        if stage not in rule.get("stages", []):
            continue
        gate_rules.append(rule["id"])
    findings, _ = _run_checks(ctx, gate_rules)
    return _cmd_run(findings)


def cmd_check_phase_gate(op_dir: str, phase: str, stage: int = 5) -> int:
    """Stage 5 ``complete_phase`` 用的 phase 限定门禁。

    与 :func:`cmd_check_gate` 不同, 本命令:

    1. 仅运行 ``target == "impl"`` 的规则 (OL01-08, OL16, OL25, OL26, OL28,
       OL29, OL43, OL45-49 等), 以及 ``target == "gate"`` 中对 phase 有意义
       的少数规则 (OL53, OL54)。
       test/golden 规则跳过 — 它们针对的是 Stage 5 cleanup 才会就绪的
       整合 artifact, 在 phase 边界检查只会因为 stub 文件报假阳性。
    2. 通过 ``ctx.phase_scope = phase`` 让
       :func:`utils._impl_files_to_scan` 只返回当前 phase 对应的累积模块
       impl 文件 (如 ``M1`` → ``modules/<op>_module1_impl.py``)。其他
       phase 的模块文件与顶层 ``<op>_impl.py`` 都不参与扫描。

    入口: ``state_transition(action=complete_phase, phase=M_k)`` 由
    ``pypto-state-transition.ts`` 中的 plugin 转译为本命令。违规情况
    与 PostToolUse hook 的 in-band block 一样, 以 exit code 2 + JSON
    返回, 由 plugin 侧捕获并报告。
    """
    ctx = _build_context(op_dir, stage)
    ctx.phase_scope = phase
    # Only the explicitly listed gate rules are meaningful at a phase boundary;
    # the remaining gate rules examine cleanup artifacts that do not exist yet.
    phase_gate_allow = {"OL53", "OL54"}
    phase_rules: list[str] = []
    for rule in ctx.rules:
        target = rule.get("target")
        rid = rule["id"]
        is_phase_rule = target == "impl" or (target == "gate" and rid in phase_gate_allow)
        if not is_phase_rule:
            continue
        if stage not in rule.get("stages", []):
            continue
        phase_rules.append(rid)
    findings, _ = _run_checks(ctx, phase_rules)
    return _cmd_run(findings)


def cmd_check_design_gate(op_dir: str, stage: int = 4) -> int:
    """Stage 4 ``submit_design`` 用的设计阶段门禁。

    Stage 4 由两个 subagent 串联执行: Designer 先产出 ``DESIGN.md`` /
    ``module_interfaces.yaml``, 之后 Verifier 在 scaffolding mode 中产出
    adversarial harness。本门禁在两者之间触发, 让 Designer 的产出在
    Verifier 启动**之前**就被 lint 捕获, 避免 Verifier 的工作因 Designer
    侧的 typo / API 不存在而作废。

    与 :func:`cmd_check_phase_gate` 的对偶: phase gate 在 Coder 之后,
    design gate 在 Designer 之后。

    扫描范围: ``DESIGN.md`` (含 fenced Python code block) 内的:

    - **OL12** — DESIGN.md 结构性校验 (计算图 / Tiling / 验证方案章节)
    - **OL55** — PyPTO API 存在性 (阻止 ``pypto.empty`` 这类 typo)
    - **OL56** — unroll_list 单一值 (阻止 Stage 6 之前出现多值 unroll_list)

    入口: ``state_transition(action=submit_design, stage=4)`` 由
    ``pypto-state-transition.ts`` 中的 plugin 转译为本命令。违规以
    exit code 2 + JSON 返回, 由 plugin 抛出, 让 Orchestrator 再次
    dispatch Designer 而不是先 dispatch Verifier。
    """
    ctx = _build_context(op_dir, stage)
    design_rules: list[str] = []
    for rule in ctx.rules:
        rid = rule["id"]
        if rid not in ("OL12", "OL55", "OL56"):
            continue
        if stage not in rule.get("stages", []):
            continue
        design_rules.append(rid)
    findings, _ = _run_checks(ctx, design_rules)
    return _cmd_run(findings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyPTO 算子开发流程确定性检查工具")
    parser.add_argument("--hook",
                        choices=["post-edit", "post-bash",
                                 "pre-edit-backup", "stop"],
                        help="Hook 模式（从 stdin 读 JSON）")
    parser.add_argument("--lint-impl", action="store_true",
                        help="检查 impl 文件")
    parser.add_argument("--lint-golden", action="store_true",
                        help="检查 golden 文件")
    parser.add_argument("--lint-test", action="store_true",
                        help="检查 test 文件")
    parser.add_argument("--lint-consistency", action="store_true",
                        help="检查跨文件一致性（D5 规则）")
    parser.add_argument("--check-gate", action="store_true",
                        help="检查阶段门禁")
    parser.add_argument("--check-phase-gate", action="store_true",
                        help="检查 Stage 5 phase 门禁 (仅扫描当前 phase 的累积模块 impl)")
    parser.add_argument("--check-design-gate", action="store_true",
                        help="检查 Stage 4 设计阶段门禁 (Designer 完成后, Verifier 启动前)")
    parser.add_argument("--phase",
                        help="Stage 5 phase 标识 (如 M1, M2, ...), 与 --check-phase-gate 配合")
    parser.add_argument("--op-dir", help="算子工作目录")
    parser.add_argument("--stage", type=int, default=5,
                        help="当前阶段 (1-7)")
    return parser


def _require_op_dir(parser: argparse.ArgumentParser, args: argparse.Namespace, option: str) -> str:
    if not args.op_dir:
        parser.error(f"{option} 需要 --op-dir")
    return args.op_dir


def _dispatch_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    hook_commands = {
        "post-edit": hook_post_edit,
        "post-bash": hook_post_bash,
        "pre-edit-backup": hook_pre_edit_backup,
        "stop": hook_stop,
    }
    if args.hook:
        hook_command = hook_commands.get(args.hook)
        if hook_command is None:
            parser.error(f"unsupported hook: {args.hook}")
        return hook_command()

    lint_commands = (
        ("lint_impl", "--lint-impl", cmd_lint_impl),
        ("lint_golden", "--lint-golden", cmd_lint_golden),
        ("lint_test", "--lint-test", cmd_lint_test),
        ("lint_consistency", "--lint-consistency", cmd_lint_consistency),
        ("check_gate", "--check-gate", cmd_check_gate),
    )
    for attribute, option, command in lint_commands:
        if getattr(args, attribute):
            op_dir = _require_op_dir(parser, args, option)
            return command(op_dir, args.stage)

    if args.check_phase_gate:
        op_dir = _require_op_dir(parser, args, "--check-phase-gate")
        if not args.phase:
            parser.error("--check-phase-gate 需要 --phase (如 M1, M2, ...)")
        return cmd_check_phase_gate(op_dir, args.phase, args.stage)

    if args.check_design_gate:
        op_dir = _require_op_dir(parser, args, "--check-design-gate")
        stage = args.stage if args.stage != 5 else 4
        return cmd_check_design_gate(op_dir, stage)

    parser.print_help()
    return 0


def main() -> int:
    parser = _build_parser()
    return _dispatch_command(parser, parser.parse_args())
