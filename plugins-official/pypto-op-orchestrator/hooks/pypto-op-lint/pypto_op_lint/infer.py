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
import select
import sys
from typing import Any, Optional

from .core import (
    API_REPORT_FILE,
    DESIGN_FILE,
    DESIGN_RULE_IDS,
    GOLDEN_RULE_IDS,
    HOOK_INPUT_ENV,
    IMPL_RULE_IDS,
    POST_EDIT_CONSISTENCY_RULE_IDS,
    SPEC_FILE,
    TEST_RULE_IDS,
    CheckContext,
    _load_rules,
)


def _infer_op_dir(file_path: str) -> Optional[str]:
    """Resolve the operator directory that owns ``file_path``.

    覆盖三类布局:

    * 标准布局：``<op_dir>/<op>_impl.py`` 等顶层产物 → ``<op_dir>``
    * 模块开发布局 (Stage 5 Phase M_k)：
      ``<op_dir>/modules/<op>_module*_impl.py`` 等子目录产物 → 仍返回 ``<op_dir>``。
      若不向上解析，``modules/`` 自身会被误判为算子目录，使得
      ``_impl_files_to_scan`` 拿不到 ``operator_name``，所有 module 级
      lint 规则因 "无 impl 文件可供检查" 而 SKIP。
    * 无状态布局 (尚未生成 ``.orchestrator_state.json``)：
      通过 ``_looks_like_stateless_op_dir`` 启发式匹配。
    """
    if not file_path:
        return None
    op_dir = os.path.dirname(os.path.abspath(file_path))
    if os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        return op_dir
    # ── modules/ 子目录 → 向上解析到算子主目录 ──
    # 模块开发阶段写入的 modules/<op>_module*_impl.py 须挂到正确的 op_dir,
    # 才能触发 _impl_files_to_scan 的模块级 lint 覆盖。
    if os.path.basename(op_dir) == "modules":
        parent = os.path.dirname(op_dir)
        if parent and os.path.isfile(os.path.join(parent, ".orchestrator_state.json")):
            return parent
        parent_basename = os.path.basename(parent)
        if parent_basename and _looks_like_stateless_op_dir(parent, parent_basename):
            return parent
    basename = os.path.basename(file_path)
    inferred_op_name = _infer_op_name_from_filename(basename)
    if inferred_op_name and _looks_like_stateless_op_dir(op_dir, inferred_op_name):
        return op_dir
    return None


def _infer_op_name_from_filename(filename: str) -> str:
    if filename.startswith("test_") and filename.endswith(".py"):
        return filename[len("test_"):-len(".py")]
    if filename.endswith("_impl.py"):
        return filename[:-len("_impl.py")]
    if filename.endswith("_golden.py"):
        return filename[:-len("_golden.py")]
    return ""


def _looks_like_stateless_op_dir(op_dir: str, op_name: str) -> bool:
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return False
    expected = {
        f"{op_name}_impl.py",
        f"{op_name}_golden.py",
        f"test_{op_name}.py",
        SPEC_FILE,
        API_REPORT_FILE,
        DESIGN_FILE,
        "README.md",
    }
    return len(files & expected) >= 2


def _infer_stage_from_filename(filename: str) -> int:
    if filename == SPEC_FILE:
        return 1
    if filename == API_REPORT_FILE:
        return 2
    if filename == DESIGN_FILE:
        return 4
    if filename.endswith("_golden.py"):
        return 3
    if filename.endswith("_impl.py") or (filename.startswith("test_") and filename.endswith(".py")):
        return 5
    return 0


def _infer_stage_from_artifacts(op_dir: str) -> int:
    op_name = os.path.basename(op_dir)
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return 0

    if len({f"{op_name}_impl.py", f"test_{op_name}.py", "README.md"} & files) >= 2:
        return 5
    if DESIGN_FILE in files or f"{op_name}_golden.py" in files:
        return 4
    if API_REPORT_FILE in files:
        return 3
    if SPEC_FILE in files:
        return 2
    return 0


def _get_current_stage(op_dir: str) -> int:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return 0
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("current_stage", 0))
    except ValueError:
        return 0


def _state_op_name(state_path: str) -> str:
    try:
        with open(state_path, "r", encoding="utf-8") as state_file:
            return json.load(state_file).get("operator_name", "")
    except ValueError:
        return ""


def _get_op_name(op_dir: str) -> str:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if os.path.isfile(state_path):
        name = _state_op_name(state_path)
        if name:
            return name
    return os.path.basename(op_dir)


def _build_context(op_dir: str, stage: Optional[int] = None) -> CheckContext:
    rules = _load_rules()
    op_dir = os.path.abspath(op_dir)
    if stage is None:
        stage = _get_current_stage(op_dir)
    op_name = _get_op_name(op_dir)
    return CheckContext(op_dir=op_dir, op_name=op_name, stage=stage, rules=rules)


def _load_hook_input() -> dict[str, Any]:
    raw = ""
    # Try stdin first — non-blocking probe. Works for any caller that pipes data.
    # select(0) returns immediately: data ready → read; no data → skip to env var.
    # On a TTY stdin, select(0) also returns immediately (TTY has no pending data),
    # so this is safe for interactive terminals too — no isatty() check needed.
    if not sys.stdin.closed:
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.read()
        except (ValueError, OSError):
            pass
    # Fall back to env var (OpenCode plugin path)
    if not raw.strip():
        raw = os.environ.get(HOOK_INPUT_ENV, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


SPEC_RULE_IDS = ["OL09"]


def _rule_ids_for_filename(filename: str) -> list[str]:
    """Return rule IDs for post-edit checks on a single file.

    Only includes rules that validate the edited file itself.  Cross-file
    consistency rules (target=gate, D5 dimension: OL30-OL34, OL39, OL40,
    OL43) are deferred to the gate/stop check so that half-written
    artefacts don't cause spurious blocks during implementation.
    """
    if filename.endswith("_impl.py"):
        return IMPL_RULE_IDS + POST_EDIT_CONSISTENCY_RULE_IDS
    if filename.endswith("_golden.py"):
        return GOLDEN_RULE_IDS + POST_EDIT_CONSISTENCY_RULE_IDS
    is_test_file = filename.startswith("test_") and filename.endswith(".py")
    if is_test_file:
        return TEST_RULE_IDS + POST_EDIT_CONSISTENCY_RULE_IDS
    if filename == SPEC_FILE:
        return SPEC_RULE_IDS
    if filename == DESIGN_FILE:
        # DESIGN.md 编辑后即时校验 PyPTO API 存在性 (OL55), 阻止 Designer
        # 在伪代码块写出不存在的 pypto.<attr> (如 `pypto.empty`)。
        return DESIGN_RULE_IDS
    return []


_MAX_OP_DIR_SEARCH_DEPTH = 8


def _find_nearest_op_dir(cwd: str) -> Optional[str]:
    """从 cwd 向上查找所属算子目录，避免跨算子误判。"""
    abs_cwd = os.path.abspath(cwd)
    current = abs_cwd
    for _ in range(_MAX_OP_DIR_SEARCH_DEPTH):
        if os.path.isfile(os.path.join(current, ".orchestrator_state.json")):
            return current
        if _looks_like_stateless_op_dir(current, os.path.basename(current)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None
