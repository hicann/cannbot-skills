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

import ast
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .ast_helpers import _resolve_pypto_aliases
from .observability import (
    _emit_metric_event_buffered,
    _emit_summary_event,
    _flush_metrics_batch,
    _RunMeta,
)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_COMMAND_PATTERN = re.compile(r"python3?\s+.*test_\w+\.py")
PYTHON_BIN = os.path.abspath(sys.executable)
GIT_BIN = shutil.which("git")

SPEC_FILE = "SPEC.md"

API_REPORT_FILE = "API_REPORT.md"

DESIGN_FILE = "DESIGN.md"

GOLDEN_PERF_REPORT_FILE = "GOLDEN_PERF_REPORT.md"

OP_WORKSPACE_DIR = "pypto"

IMPL_RULE_IDS = [
    "OL01", "OL02", "OL03", "OL04", "OL05", "OL06", "OL07", "OL08",
    "OL16", "OL23", "OL25", "OL26", "OL28", "OL29", "OL37", "OL48", "OL49",
    "OL55",  # PyPTO API 存在性 (typo 防止: pypto.empty 等)
    "OL56",  # Stage 6 之前 unroll_list 只能含单一值 (默认 [1])
    "OL57",  # JIT 图代码内允许 pypto.loop / pypto.loop_unroll / range 循环 (禁止 while 和非 range 的 for)
    "OL58",  # Layer K wrapper output buffer 必须 torch.* 预分配 (禁止 host pypto.zeros)
    "OL61",  # Preflight code scan: cast path / Element wrap / scalar arg / zeros dtype
    "OL62",  # impl 内 torch 仅限 layout/alloc/cast; 数值计算必须在 @jit 图内 (dummy-JIT 防护)
]

# DESIGN.md post-edit 适用规则 (在 Stage 4 Designer 编辑 DESIGN.md 时即时校验)。
# OL55 (PyPTO API 存在性) + OL56 (unroll_list 单一值) 进入此列表; OL12 等 DESIGN
# 结构性检查在 stop / complete_stage gate 中校验, 不在 post-edit 阶段触发,
# 避免半成品 block。
DESIGN_RULE_IDS = ["OL55", "OL56"]

GOLDEN_RULE_IDS = ["OL15"]

TEST_RULE_IDS = ["OL17", "OL18", "OL19", "OL20", "OL21", "OL22", "OL42", "OL60"]

CONSISTENCY_RULE_IDS = ["OL30", "OL31", "OL32", "OL33", "OL34", "OL39", "OL40", "OL41", "OL43"]

# Subset of CONSISTENCY_RULE_IDS safe for post-edit: only rules that check
# the edited file itself, not cross-file dependencies.  Cross-file rules
# (OL30-OL34, OL39, OL40, OL43) must wait for the gate / stop check.
POST_EDIT_CONSISTENCY_RULE_IDS = ["OL41"]

STRICT_ENV = "PYPTO_OP_LINT_STRICT"

MODE_ENV = "PYPTO_OP_LINT_MODE"

HOOK_INPUT_ENV = "PYPTO_OP_LINT_HOOK_INPUT"

POST_EDIT_BLOCK_ENV = "PYPTO_OP_LINT_POST_EDIT_BLOCK"


@dataclass
class Finding:
    rule_id: str = ""
    severity: str = ""
    dimension: str = ""
    fix_effort: str = ""
    status: str = "SKIP"
    message: str = ""
    file: str = ""
    line: int = 0


@dataclass
class CheckContext:
    op_dir: str
    op_name: str
    stage: int
    rules: list[dict[str, Any]]
    # ── file-scope filter (post-edit hook) ──
    # 设置后, ``_impl_files_to_scan`` 仅返回这一个 impl 文件; 其他 module
    # impl 与顶层 ``<op>_impl.py`` 都排除在外。post-edit hook 在 Coder
    # Write/Edit 单文件后调用 lint 时启用此字段, 避免对其他 module 的
    # stub 文件触发跨文件级规则 (OL01/OL07 等会在空 stub 上 FAIL)。
    # phase_scope 与同时设置时, file_scope 优先级更高 (post-edit 一般
    # 比 complete_phase 更具体)。``None`` 表示无文件限制。
    file_scope: Optional[str] = None
    # ── phase-scope filter (complete_phase gate) ──
    # 设置后, ``_impl_files_to_scan`` 仅返回当前 Phase 对应的累积模块
    # impl 文件 (如 ``M1`` → ``modules/<op>_module1_impl.py``, ``M2`` →
    # ``modules/<op>_module12_impl.py``); 顶层 ``<op>_impl.py`` 与其他
    # module impl 都排除在外。``--check-phase-gate`` CLI / orchestrator 的
    # ``complete_phase`` 网关使用此字段, 避免在 phase 边界检查 Stage 5
    # cleanup 才产出的整合 artifact。``None`` 表示无 phase 限制 (默认)。
    phase_scope: Optional[str] = None
    _ast_cache: dict[str, ast.Module] = field(default_factory=dict, repr=False)
    _parse_errors: dict[str, str] = field(default_factory=dict, repr=False)
    _pypto_aliases_cache: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def file_path(self, filename: str) -> str:
        return os.path.join(self.op_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(self.file_path(filename))

    def read_file(self, filename: str) -> str:
        path = self.file_path(filename)
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_file(self, filename: str) -> Optional[ast.Module]:
        if filename in self._ast_cache:
            return self._ast_cache[filename]
        source = self.read_file(filename)
        if not source:
            return None
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            self._parse_errors[filename] = str(e)
            return None
        self._ast_cache[filename] = tree
        return tree

    def parse_error(self, filename: str) -> str:
        return self._parse_errors.get(filename, "")

    def pypto_aliases(self, filename: str) -> set[str]:
        """获取指定文件中 pypto 包的 import 别名集合（带缓存）。"""
        if filename in self._pypto_aliases_cache:
            return self._pypto_aliases_cache[filename]
        tree = self.parse_file(filename)
        if tree is None:
            aliases = {"pypto"}
        else:
            aliases = _resolve_pypto_aliases(tree)
        self._pypto_aliases_cache[filename] = aliases
        return aliases

    def get_rule(self, rule_id: str) -> dict[str, Any]:
        for r in self.rules:
            if r["id"] == rule_id:
                return r
        return {}

    def make_finding(self, rule_id: str, status: str, message: str,
                     file: str = "", line: int = 0) -> Finding:
        rule = self.get_rule(rule_id)
        return Finding(
            rule_id=rule_id,
            severity=rule.get("severity", "S2"),
            dimension=rule.get("dimension", ""),
            fix_effort=rule.get("fix_effort", ""),
            status=status,
            message=message,
            file=file,
            line=line,
        )


CHECKERS: dict[str, Callable[[CheckContext], Finding]] = {}


def register(rule_id: str):
    """装饰器：将检查函数注册到规则 ID"""
    def decorator(fn: Callable[[CheckContext], Finding]):
        CHECKERS[rule_id] = fn
        return fn
    return decorator


def _load_rules() -> list[dict[str, Any]]:
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger(__name__).error("[pypto-op-lint FATAL] rules.json 加载失败: %s", e)
        return []
    return data.get("rules", [])


def _run_checks(ctx: CheckContext, rule_ids: list[str]) -> tuple[list[Finding], str]:
    invocation_id = uuid.uuid4().hex[:8]
    findings = []
    run_meta = _RunMeta(
        mode=os.environ.get(MODE_ENV, "cli"),
        strict=os.environ.get(STRICT_ENV, "1") == "1",
        invocation_id=invocation_id,
    )
    total_duration_ms = 0.0
    try:
        for rid in rule_ids:
            start = time.perf_counter()
            rule = ctx.get_rule(rid)
            if ctx.stage not in rule.get("stages", []):
                finding = ctx.make_finding(rid, "SKIP", "当前阶段不适用")
                findings.append(finding)
                dur = (time.perf_counter() - start) * 1000
                total_duration_ms += dur
                _emit_metric_event_buffered(ctx, finding, run_meta, dur)
                continue
            checker = CHECKERS.get(rid)
            if not checker:
                finding = ctx.make_finding(rid, "SKIP", "检查函数未注册")
                findings.append(finding)
                dur = (time.perf_counter() - start) * 1000
                total_duration_ms += dur
                _emit_metric_event_buffered(ctx, finding, run_meta, dur)
                continue
            finding = checker(ctx)
            findings.append(finding)
            dur = (time.perf_counter() - start) * 1000
            total_duration_ms += dur
            _emit_metric_event_buffered(ctx, finding, run_meta, dur)
    finally:
        _flush_metrics_batch()
    _emit_summary_event(ctx, findings, run_meta, total_duration_ms)
    return findings, invocation_id


def _has_error_fail(findings: list[Finding]) -> bool:
    return any(
        finding.status == "FAIL" and finding.severity in ("S0", "S1")
        for finding in findings
    )
