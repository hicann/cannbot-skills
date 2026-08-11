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

"""D1 维度: PyPTO API 存在性检查 (OL55)。

通过对比源代码中出现的 ``pypto.<attr>`` 与运行时 ``dir(pypto)`` 集合,
在 post-edit hook 阶段就阻止 typo (如 ``pypto.empty`` / ``pypto.empty_like``
— 这些 API 不存在, JIT 内应使用 ``pypto.zeros`` / ``pypto.ones`` /
``pypto.full``)。这是为了防止 typo 走到运行时才被 verifier 捕获,
从而避免 Verify → Debug → Coder fix → Verify 的循环开销。

对象文件:
- ``DESIGN.md``: 仅扫描 Markdown 中 ```python``` / ``` ``` 代码块内容。
- ``<op>_impl.py`` / ``<op>_module*_impl.py``: 扫描整个 AST。
- 其他文件: SKIP (不在 OL55 适用范围)。

PyPTO 在 lint 环境不可导入时返回 SKIP (而非 FAIL), 避免无 PyPTO 的环境
下 hook 持续报错。
"""

from __future__ import annotations

import ast
import difflib
import os
from typing import Iterable

from ..ast_helpers import _resolve_pypto_aliases  # 复用现有别名解析
from ..core import CheckContext, Finding, register
from ..pypto_attrs import (
    extract_pypto_attrs,
    extract_python_blocks,
    get_pypto_attrs,
)

# 推荐的等价替换 (针对最常见的 typo); 找不到 close match 时作为兜底建议。
# 注意: 不要把 from_torch 作为建议 —— 本仓库算子强制走 JIT 经路, 输入张量直接
# 用原始 torch (torch.randn(...) 等) 传入 kernel, from_torch 在算子开发里没有使用场景。
_KNOWN_TYPO_HINTS: dict[str, list[str]] = {
    "empty": ["zeros", "ones", "full"],
    "empty_like": ["zeros", "ones", "full"],
    "Empty": ["zeros", "ones", "full"],
    "tensor": ["zeros", "ones"],
    "Tensor_": ["Tensor"],
    "constant": ["full"],
}

# 从 torch 习惯带来的 typo: PyPTO 没有随机生成 / 通用 tensor 构造 API。
# 命中时附加一条说明, 引导算子直接用原始 torch 输入, 而不是去找 pypto 等价物。
_RAW_TORCH_INPUT_TYPOS: frozenset[str] = frozenset({"rand", "randn", "tensor"})
_MISSING_CREATION_APIS: frozenset[str] = frozenset({"empty", "empty_like"})


def _suggest_alternatives(missing: str, pypto_attrs: set[str]) -> list[str]:
    """为不存在的属性给出候选名建议 (排序 by 相似度)。"""
    # 先尝试已知 typo 表
    known = _KNOWN_TYPO_HINTS.get(missing, [])
    # difflib 找相似名 (cutoff=0.6 较宽容; 调到 0.7 会漏掉 empty→zeros 等)
    fuzzy = difflib.get_close_matches(missing, pypto_attrs, n=5, cutoff=0.6)
    # 合并去重, 保留顺序
    seen: set[str] = set()
    out: list[str] = []
    for cand in [*known, *fuzzy]:
        if cand in pypto_attrs and cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out[:5]


def _format_failures(
    missing: Iterable[str], pypto_attrs: set[str], file_label: str
) -> str:
    """组装 FAIL 文案。包含每个不存在属性的候选建议。"""
    lines = [f"{file_label} 内出现 PyPTO 中不存在的属性:"]
    for attr in sorted(missing):
        suggestions = _suggest_alternatives(attr, pypto_attrs)
        if suggestions:
            hint = ", ".join(f"pypto.{s}" for s in suggestions)
            lines.append(f"  • pypto.{attr} → 不存在 (可能想用: {hint})")
        else:
            lines.append(f"  • pypto.{attr} → 不存在 (无相似名建议)")
        if attr in _RAW_TORCH_INPUT_TYPOS:
            # 这类 typo 来自 torch 习惯; PyPTO 没有随机 / 通用 tensor 构造 API。
            lines.append(
                "    注: PyPTO 没有随机 / 通用张量构造 API。算子输入张量直接用原始 "
                "torch (如 torch.randn(...)) 传入 JIT kernel; kernel 内部需要新建张量时用 "
                "pypto.zeros / pypto.ones / pypto.full。不要用 from_torch。"
            )
        if attr in _MISSING_CREATION_APIS:
            lines.append(
                "    注: 当前 PyPTO 公开 API 没有 pypto.empty / pypto.empty_like。"
                "Layer K host wrapper 分配输出 buffer 时使用 torch.empty / torch.empty_like；"
                "JIT 内需要初始化张量时使用 pypto.zeros / pypto.ones / pypto.full。"
            )
    return "\n".join(lines)


def _scan_design_md(ctx: CheckContext, filename: str) -> tuple[set[str], list[str]]:
    """提取 DESIGN.md 中所有 Python 代码块的 pypto.<attr> 集合与块对应别名。

    Returns:
        (all_attrs, _) — 第二项是每个块用到的别名集合的并集 (此处合并后只返回属性)。
    """
    source = ctx.read_file(filename)
    if not source:
        return set(), []
    all_attrs: set[str] = set()
    for block in extract_python_blocks(source):
        try:
            block_tree = ast.parse(block)
            aliases = _resolve_pypto_aliases(block_tree)
        except SyntaxError:
            aliases = {"pypto"}
        all_attrs |= extract_pypto_attrs(block, aliases)
    return all_attrs, []


def _scan_python_file(ctx: CheckContext, filename: str) -> set[str]:
    """扫描整个 Python 文件, 返回 pypto.<attr> 集合 (含别名解析)。"""
    tree = ctx.parse_file(filename)
    if tree is None:
        return set()
    aliases = ctx.pypto_aliases(filename)
    source = ctx.read_file(filename)
    return extract_pypto_attrs(source, aliases)


def _target_files(ctx: CheckContext) -> list[tuple[str, str]]:
    """列出 OL55 适用的文件 ``(relative_filename, file_type)``。

    file_type 取值: ``"design"`` / ``"impl"``。其他文件不进入此列表
    (golden / test / yaml / runner 等都不在 OL55 范围内)。
    """
    out: list[tuple[str, str]] = []

    # DESIGN.md
    if ctx.file_exists("DESIGN.md"):
        # file_scope 限制下, 仅当 DESIGN.md 是被编辑的文件时才检查
        if ctx.file_scope is None or _basename_match(ctx.file_scope, "DESIGN.md"):
            out.append(("DESIGN.md", "design"))

    # <op>_impl.py (Stage 5 cleanup 后的整合 impl)
    top_impl = f"{ctx.op_name}_impl.py"
    if ctx.file_exists(top_impl):
        if ctx.file_scope is None or _basename_match(ctx.file_scope, top_impl):
            out.append((top_impl, "impl"))

    # modules/<op>_module<suffix>_impl.py
    modules_dir = os.path.join(ctx.op_dir, "modules")
    if os.path.isdir(modules_dir):
        for entry in sorted(os.listdir(modules_dir)):
            if not entry.endswith("_impl.py"):
                continue
            if not entry.startswith(f"{ctx.op_name}_module"):
                continue
            rel = os.path.join("modules", entry)
            if ctx.file_scope is None or _basename_match(ctx.file_scope, entry):
                out.append((rel, "impl"))

    return out


def _basename_match(file_scope: str, candidate: str) -> bool:
    """file_scope (post-edit hook 传入的绝对/相对路径) 与候选基名是否匹配。"""
    return os.path.basename(file_scope) == os.path.basename(candidate)


@register("OL55")
def check_ol55(ctx: CheckContext) -> Finding:
    """禁止使用 PyPTO 中不存在的 ``pypto.<attr>``。"""
    targets = _target_files(ctx)
    if not targets:
        return ctx.make_finding(
            "OL55", "SKIP", "未发现 OL55 适用的文件 (DESIGN.md / *_impl.py)"
        )

    pypto_attrs = get_pypto_attrs()
    if pypto_attrs is None:
        return ctx.make_finding(
            "OL55",
            "SKIP",
            "lint 环境中 PyPTO 不可导入, 跳过 API 存在性检查 "
            "(本地装好 pypto 后会自动恢复)",
        )

    # 累计违规
    failures: list[tuple[str, set[str]]] = []
    total_checked = 0
    for rel, ftype in targets:
        if ftype == "design":
            attrs, _ = _scan_design_md(ctx, rel)
        else:
            attrs = _scan_python_file(ctx, rel)
        if not attrs:
            continue
        total_checked += len(attrs)
        missing = attrs - pypto_attrs
        if missing:
            failures.append((rel, missing))

    if failures:
        msg_blocks: list[str] = []
        for rel, missing in failures:
            msg_blocks.append(_format_failures(missing, pypto_attrs, rel))
        return ctx.make_finding(
            "OL55",
            "FAIL",
            "\n".join(msg_blocks),
            file=failures[0][0],
        )

    return ctx.make_finding(
        "OL55",
        "PASS",
        f"所有 pypto.<attr> (共 {total_checked} 处) 均存在于 dir(pypto)",
    )
