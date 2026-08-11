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
import os
import re

from ..ast_helpers import _get_jit_functions, _has_test_level_markers
from ..core import CheckContext, Finding, register
from ..utils import _check_npu_available, _impl_files_to_scan


@register("OL19")
def check_ol19(ctx: CheckContext) -> Finding:
    """test 必须使用 assert_allclose 或 detailed_tensor_compare 做精度比对。"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL19", "SKIP", f"{test_file} 不存在")
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL19", "SKIP", f"{test_file} 无法解析")
    # 接受任一标准比对工具: numpy assert_allclose 或 verifier 的 detailed_tensor_compare
    _compare_helpers = ("assert_allclose", "detailed_tensor_compare")
    has_compare = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_str = ast.dump(node.func)
            if any(h in call_str for h in _compare_helpers):
                has_compare = True
                break
    if not has_compare:
        return ctx.make_finding("OL19", "FAIL",
            "未找到 assert_allclose / detailed_tensor_compare 调用，禁止手写 assert max_diff",
            file=test_file)
    return ctx.make_finding("OL19", "PASS",
        "使用了 assert_allclose / detailed_tensor_compare", file=test_file)


@register("OL20")
def check_ol20(ctx: CheckContext) -> Finding:
    """test 必须处理 TILE_FWK_DEVICE_ID 并调用 set_device"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL20", "SKIP", f"{test_file} 不存在")
    has_device_id = "TILE_FWK_DEVICE_ID" in source
    has_set_device = "set_device" in source
    if has_device_id and has_set_device:
        return ctx.make_finding("OL20", "PASS",
            "找到 TILE_FWK_DEVICE_ID 处理和 set_device 调用", file=test_file)
    missing = []
    if not has_device_id:
        missing.append("TILE_FWK_DEVICE_ID 环境变量处理")
    if not has_set_device:
        missing.append("set_device 调用")
    return ctx.make_finding("OL20", "FAIL",
        f"缺少: {', '.join(missing)}", file=test_file)


@register("OL21")
def check_ol21(ctx: CheckContext) -> Finding:
    """test 必须有 Level 0 和 Level 1 两级测试函数"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在")
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在或无法解析")
    has_level0, has_level1 = _has_test_level_markers(tree, source)
    missing = []
    if not has_level0:
        missing.append("level0")
    if not has_level1:
        missing.append("level1")
    if missing:
        return ctx.make_finding("OL21", "FAIL",
            f"缺少测试级别: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL21", "PASS",
        "包含 Level 0 和 Level 1 测试", file=test_file)


@register("OL22")
def check_ol22(ctx: CheckContext) -> Finding:
    """test 应设置 torch.manual_seed 保证可复现"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL22", "SKIP", f"{test_file} 不存在")
    if "manual_seed" in source:
        return ctx.make_finding("OL22", "PASS",
            "找到 manual_seed 设置", file=test_file)
    return ctx.make_finding("OL22", "WARN",
        "未设置 torch.manual_seed，建议显式设置以保证测试可复现",
        file=test_file)


# =============================================================================
# OL60 — test_<op>.py imports must reach @pypto.frontend.jit transitively
# =============================================================================


def _local_call_name(node: ast.Call) -> str | None:
    """Extract the bare function name for a local ``Name(id=...)`` callee."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _called_local_names(func: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            callee = _local_call_name(node)
            if callee is not None:
                names.add(callee)
    return names


def _ol60_test_failure(ctx, test_file, impl_stems):
    impl_imports = _called_impl_imports(ctx, test_file, set(impl_stems))
    if not impl_imports:
        return None, False
    for impl_stem, names in impl_imports.items():
        impl_file = impl_stems[impl_stem]
        graph = _impl_call_graph(ctx, impl_file)
        if graph is None:
            continue
        for entry_name in names:
            failure = _ol60_entry_failure(
                ctx, test_file, impl_stem, entry_name, graph,
            )
            if failure is not None:
                return failure, True
    return None, True


def _reaches_jit(
    entry: ast.FunctionDef,
    function_defs: dict[str, ast.FunctionDef],
    jit_names: set[str],
    max_depth: int = 8,
) -> bool:
    """Return True if ``entry`` transitively calls any function in ``jit_names``.

    Walks the same-file call graph from ``entry``, following bare-name calls
    (``foo(...)``) that resolve to a local ``FunctionDef``. Stops when any
    visited function is itself a JIT entry, or when ``max_depth`` is exceeded.
    Cycles are blocked by ``visited``.
    """
    visited: set[str] = set()
    stack: list[ast.FunctionDef] = [entry]

    while stack:
        if len(visited) > max_depth:
            return False
        func = stack.pop()
        if func.name in visited:
            continue
        visited.add(func.name)
        if func.name in jit_names:
            return True

        callees = _called_local_names(func) - visited
        if callees & jit_names:
            return True
        stack.extend(function_defs[name] for name in callees if name in function_defs)
    return False


def _collect_impl_imports(
    test_tree: ast.Module, impl_module_stems: set[str]
) -> dict[str, list[str]]:
    """Collect ``from <stem> import X`` mappings from the test file.

    Returns ``{stem: [imported_name_or_asname, ...]}``. Only ``from ... import``
    forms are honoured (the canonical pattern in test_*.py templates); plain
    ``import <mod>`` aliases are intentionally skipped because they would require
    attribute resolution that this rule does not attempt.
    """
    result: dict[str, list[str]] = {}
    for node in ast.walk(test_tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or node.module not in impl_module_stems:
            continue
        result.setdefault(node.module, []).extend(
            alias.asname or alias.name for alias in node.names
        )
    return result


def _collect_called_in_test(
    test_tree: ast.Module, imported_names: set[str]
) -> set[str]:
    """Return the subset of ``imported_names`` that are actually invoked in the test."""
    called: set[str] = set()
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in imported_names:
                called.add(node.func.id)
    return called


def _collect_test_files(ctx: CheckContext) -> list[str]:
    """Return the test files OL60 should scan, relative to ``ctx.op_dir``.

    Two file shapes are inspected, mirroring the workflow's two test surfaces:

    - ``test_<op>.py``                                 — Stage 6 integrated test
    - ``modules/test_<op>_module<k>.py`` (any ``<k>``) — Stage 5 per-module test

    Stage 5 is where Phase verifiers exercise individual module impls; failing
    to scan those test files leaves the flash_kda bypass undetected through the
    entire Stage 5 inner loop and only surfaces at Stage 6 integration. By
    scanning both surfaces OL60 honours its declared ``stages: [5, 6]`` window.
    """
    result: list[str] = []
    integrated = f"test_{ctx.op_name}.py"
    if ctx.file_exists(integrated):
        result.append(integrated)

    modules_dir = ctx.file_path("modules")
    if os.path.isdir(modules_dir):
        prefix = f"test_{ctx.op_name}_module"
        for name in sorted(os.listdir(modules_dir)):
            if name.startswith(prefix) and name.endswith(".py"):
                result.append(os.path.join("modules", name))
    return result


def _impl_stem_map(impl_files: list[str]) -> dict[str, str]:
    return {
        impl_file.removesuffix(".py").rsplit("/", 1)[-1]: impl_file
        for impl_file in impl_files
    }


def _called_impl_imports(
    ctx: CheckContext,
    test_file: str,
    impl_stems: set[str],
) -> dict[str, list[str]]:
    test_tree = ctx.parse_file(test_file)
    if test_tree is None:
        return {}
    impl_imports = _collect_impl_imports(test_tree, impl_stems)
    all_imported = {name for names in impl_imports.values() for name in names}
    called = _collect_called_in_test(test_tree, all_imported)
    return {
        stem: [name for name in names if name in called]
        for stem, names in impl_imports.items()
        if any(name in called for name in names)
    }


def _impl_call_graph(
    ctx: CheckContext,
    impl_file: str,
) -> tuple[str, set[str], dict[str, ast.FunctionDef]] | None:
    impl_tree = ctx.parse_file(impl_file)
    if impl_tree is None:
        return None
    aliases = ctx.pypto_aliases(impl_file)
    jit_names = {func.name for func in _get_jit_functions(impl_tree, aliases)}
    function_defs = {
        node.name: node
        for node in ast.iter_child_nodes(impl_tree)
        if isinstance(node, ast.FunctionDef)
    }
    return impl_file, jit_names, function_defs


def _ol60_entry_failure(
    ctx: CheckContext,
    test_file: str,
    impl_stem: str,
    entry_name: str,
    graph: tuple[str, set[str], dict[str, ast.FunctionDef]],
) -> Finding | None:
    impl_file, jit_names, function_defs = graph
    if not jit_names:
        return ctx.make_finding(
            "OL60",
            "FAIL",
            f"{test_file} 从 `{impl_stem}` import 并调用 `{entry_name}`, "
            f"但 {impl_file} 中没有任何 @pypto.frontend.jit 函数。"
            f"测试将完全不执行 PyPTO 内核 (flash_kda 类失败模式: "
            f"agent 把所有计算都放在纯 PyTorch 路径上)。",
            file=test_file,
        )
    entry_func = function_defs.get(entry_name)
    if entry_func is None or _reaches_jit(entry_func, function_defs, jit_names):
        return None
    return ctx.make_finding(
        "OL60",
        "FAIL",
        f"{test_file} 调用 `{entry_name}` (来自 {impl_stem}), "
        f"但 `{entry_name}` 同文件可达调用链中无法到达任何 "
        f"@pypto.frontend.jit 函数。"
        f"impl 中存在的 JIT 入口: {sorted(jit_names)}。"
        f"该测试实际跳过了 PyPTO 内核 (flash_kda 类失败模式)。"
        f"修复方式: 让 `{entry_name}` 或它调用的某个 helper 显式调用 "
        f"@pypto.frontend.jit 入口, 或把 test 改为 import 那个 "
        f"通过 wrapper 调用 JIT 的入口。",
        file=test_file,
        line=entry_func.lineno,
    )


@register("OL60")
def check_ol60(ctx: CheckContext) -> Finding:
    """test_<op>.py / modules/test_<op>_module<k>.py 调用的入口函数必须可达 @jit。

    flash_kda 类失败模式的"测试端"补强检查 (对应 OL51.b 的"impl 端"检查): 即便
    impl 文件里写了一个 @pypto.frontend.jit 函数, agent 仍可能让 test 去 import
    并调用另一个纯 PyTorch 入口函数, 使整个 verification 跳过 PyPTO 内核。

    覆盖范围 (与 stages=[5, 6] 一致):

    - Stage 6 集成 test: ``test_<op>.py``
    - Stage 5 per-module test: ``modules/test_<op>_module<k>.py`` (任意 ``<k>``)

    规则: 对每个被发现的 test 文件, 它中每个 ``from <stem>_impl import X`` 形式
    的导入, 若 ``X`` 真的在 test 函数体内被调用, 则 ``X`` 在 ``<stem>_impl.py``
    同文件可达调用链中必须能到达至少一个 @pypto.frontend.jit 函数。

    若 impl 文件根本没有 @jit 函数, 或所有被调入口都绕过 @jit, 则 FAIL。
    """
    test_files = _collect_test_files(ctx)
    if not test_files:
        return ctx.make_finding(
            "OL60",
            "SKIP",
            f"未发现 test_{ctx.op_name}.py 或 modules/test_{ctx.op_name}_module*.py",
        )

    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL60", "SKIP", "无 impl 文件可供检查")
    impl_stems = _impl_stem_map(impl_files)
    checked_any = False
    last_pass: str | None = None
    for test_file in test_files:
        failure, checked = _ol60_test_failure(ctx, test_file, impl_stems)
        if failure is not None:
            return failure
        if checked:
            checked_any = True
            last_pass = test_file
    if not checked_any:
        return ctx.make_finding(
            "OL60",
            "SKIP",
            "test 文件中没有发现对 *_impl 模块的有效 from-import + 调用对",
        )

    return ctx.make_finding(
        "OL60",
        "PASS",
        f"被检查的 {len(test_files)} 个 test 文件中, 被调用的 impl 入口均可达 "
        f"@pypto.frontend.jit",
        file=last_pass or "",
    )


@register("OL42")
def check_ol42(ctx: CheckContext) -> Finding:
    """NPU 环境下 test 不得硬编码 sim 模式"""
    test_file = f"test_{ctx.op_name}.py"
    content = ctx.read_file(test_file)
    if not content:
        return ctx.make_finding("OL42", "SKIP", f"{test_file} 不存在")

    # 检测是否有 NPU 环境
    if not _check_npu_available():
        return ctx.make_finding("OL42", "SKIP",
            "未检测到 NPU 环境（npu-smi 不可用），跳过 sim 模式检查")

    # 在有 NPU 的环境下，检查是否硬编码了 sim 模式
    problems = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        # 跳过注释行
        if stripped.startswith("#"):
            continue
        # 检查 default="sim" 或 default='sim' (argparse default)
        if re.search(r"""default\s*=\s*['"]sim['"]""", line):
            problems.append(f"L{i}: argparse default 设置为 sim")
        # 检查 run_mode="sim" 或 run_mode='sim' 的硬编码赋值
        elif re.search(r"""run_mode\s*=\s*['"]sim['"]""", line):
            problems.append(f"L{i}: run_mode 硬编码为 sim")

    if problems:
        return ctx.make_finding("OL42", "FAIL",
            f"NPU 环境下不应使用 sim 模式: {'; '.join(problems)}",
            file=f"test_{ctx.op_name}.py")
    return ctx.make_finding("OL42", "PASS",
        "未发现 sim 模式硬编码", file=f"test_{ctx.op_name}.py")
