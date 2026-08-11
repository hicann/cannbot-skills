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

from ..ast_helpers import _get_jit_functions
from ..core import CheckContext, Finding, register
from ..utils import _impl_files_to_scan


def _is_dot_t_call(node: ast.AST) -> bool:
    """判定无参数的 `.t()` 调用（拆分布尔条件以满足 G.CTL.03）。"""
    if not isinstance(node, ast.Call) or node.args:
        return False
    return isinstance(node.func, ast.Attribute) and node.func.attr == "t"


def _imported_module_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module:
        return {node.module}
    return set()


def _matches_module(module_name: str, expected: str) -> bool:
    return module_name == expected or module_name.endswith(f".{expected}")


def _find_pypto_import(tree: ast.Module) -> tuple[str, int] | None:
    for node in ast.iter_child_nodes(tree):
        module_names = _imported_module_names(node)
        if not any(name == "pypto" or name.startswith("pypto.") for name in module_names):
            continue
        if isinstance(node, ast.ImportFrom):
            return "golden 文件禁止 from pypto import ...", node.lineno
        return "golden 文件禁止 import pypto", node.lineno
    return None


def _find_module_import(tree: ast.Module, expected: str) -> ast.AST | None:
    for node in ast.iter_child_nodes(tree):
        if expected in _imported_module_names(node):
            return node
    return None


@register("OL15")
def check_ol15(ctx: CheckContext) -> Finding:
    """golden 文件须为纯 torch 规范化实现：禁止 import pypto，禁止 `.T` / `.t()`（须用 torch.transpose）。"""
    golden_file = f"{ctx.op_name}_golden.py"
    tree = ctx.parse_file(golden_file)
    if tree is None:
        return ctx.make_finding("OL15", "SKIP", f"{golden_file} 不存在或无法解析")
    bad_import = _find_pypto_import(tree)
    if bad_import is not None:
        message, line = bad_import
        return ctx.make_finding("OL15", "FAIL", message, file=golden_file, line=line)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "T":
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 `.T`，请用 torch.transpose(t, d0, d1) 并注记 `# pypto: b_trans=True`",
                file=golden_file, line=node.lineno)
        if _is_dot_t_call(node):
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 `.t()`，请用 torch.transpose(t, d0, d1)",
                file=golden_file, line=node.lineno)
    return ctx.make_finding("OL15", "PASS",
        "golden 文件未导入 pypto 且无 `.T` / `.t()`", file=golden_file)


@register("OL16")
def check_ol16(ctx: CheckContext) -> Finding:
    """impl 文件不应导入 golden 模块。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL16", "SKIP", "无 impl 文件可供检查")
    golden_module = f"{ctx.op_name}_golden"
    parsed_any = False
    for impl_file in impl_files:
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        parsed_any = True
        bad_import = _find_module_import(tree, golden_module)
        if bad_import is not None:
            return ctx.make_finding(
                "OL16",
                "FAIL",
                f"{impl_file} 不应导入 {golden_module}",
                file=impl_file,
                line=bad_import.lineno,
            )
    if not parsed_any:
        return ctx.make_finding("OL16", "SKIP", "无 impl 文件可解析")
    return ctx.make_finding(
        "OL16", "PASS", f"所有 impl 文件均未导入 golden（共 {len(impl_files)} 个）"
    )


@register("OL17")
def check_ol17(ctx: CheckContext) -> Finding:
    """test 文件不应包含 kernel 实现代码"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL17", "SKIP", f"{test_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(test_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if jit_funcs:
        func = jit_funcs[0]
        return ctx.make_finding("OL17", "FAIL",
            f"test 文件包含 @pypto.frontend.jit 装饰的函数: {func.name}",
            file=test_file, line=func.lineno)
    return ctx.make_finding("OL17", "PASS",
        "test 文件未包含 kernel 实现", file=test_file)


@register("OL18")
def check_ol18(ctx: CheckContext) -> Finding:
    """test 文件必须从 impl 和 golden 分别导入"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL18", "SKIP", f"{test_file} 不存在或无法解析")
    impl_module = f"{ctx.op_name}_impl"
    golden_module = f"{ctx.op_name}_golden"
    imported_names = {
        name
        for node in ast.walk(tree)
        for name in _imported_module_names(node)
    }
    has_impl = any(_matches_module(name, impl_module) for name in imported_names)
    has_golden = any(_matches_module(name, golden_module) for name in imported_names)
    missing = []
    if not has_impl:
        missing.append(impl_module)
    if not has_golden:
        missing.append(golden_module)
    if missing:
        return ctx.make_finding("OL18", "FAIL",
            f"test 文件缺少导入: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL18", "PASS",
        "test 文件正确导入了 impl 和 golden", file=test_file)
