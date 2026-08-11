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
from collections.abc import Iterator

from ..ast_helpers import (
    _dynamic_tensor_annotation_status,
    _extract_symbolic_dynamic_aliases,
    _get_jit_functions,
    _get_primary_jit_functions,
    _has_loop_structure,
    _is_fp32_only_call,
    _is_non_tensor_annotation,
    _is_pypto_tensor_annotation,
    _resolve_pypto_aliases,
)
from ..core import CheckContext, Finding, register
from ..pypto_attrs import extract_python_blocks
from ..utils import _impl_files_to_scan, _syntax_error_finding

_TILE_SHAPE_CALLS = {"set_vec_tile_shapes", "set_cube_tile_shapes"}

ParsedImpl = tuple[str, ast.Module, set[str]]
JitImpl = tuple[str, ast.Module, set[str], list[ast.FunctionDef]]


def _iter_parsed_impls(ctx: CheckContext, impl_files: list[str]) -> Iterator[ParsedImpl]:
    for impl_file in impl_files:
        tree = ctx.parse_file(impl_file)
        if tree is not None:
            yield impl_file, tree, ctx.pypto_aliases(impl_file)


def _iter_jit_impls(ctx: CheckContext, impl_files: list[str]) -> Iterator[JitImpl]:
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        jit_funcs = _get_jit_functions(tree, aliases)
        if jit_funcs:
            yield impl_file, tree, aliases, jit_funcs


def _is_tile_shape_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _TILE_SHAPE_CALLS
    )


def _local_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _tile_call_and_local_callees(
    func: ast.FunctionDef,
    function_defs: dict[str, ast.FunctionDef],
    visited: set[str],
) -> tuple[ast.Call | None, list[ast.FunctionDef]]:
    callees: list[ast.FunctionDef] = []
    for node in ast.walk(func):
        if _is_tile_shape_call(node):
            return node, callees  # type: ignore[return-value]
        if not isinstance(node, ast.Call):
            continue
        callee = _local_call_name(node)
        if callee and callee in function_defs and callee not in visited:
            callees.append(function_defs[callee])
    return None, callees


def _find_reachable_tile_shape_call(
    entry: ast.FunctionDef,
    function_defs: dict[str, ast.FunctionDef],
) -> ast.Call | None:
    """Return the first tile-shape call in the same-file call chain.

    Layer J is allowed to be a thin @jit entry that delegates to Layer I/H
    helpers. OL04 therefore checks the local helper chain reachable from the
    JIT entry, not only the literal @jit function body.
    """
    visited: set[str] = set()
    stack: list[ast.FunctionDef] = [entry]

    while stack:
        func = stack.pop()
        if func.name in visited:
            continue
        visited.add(func.name)

        tile_call, callees = _tile_call_and_local_callees(func, function_defs, visited)
        if tile_call is not None:
            return tile_call
        stack.extend(callees)
    return None


def _ol01_failure(
    ctx: CheckContext,
    impl_file: str,
    jit_funcs: list[ast.FunctionDef],
) -> Finding | None:
    if len(jit_funcs) == 1:
        return None
    if len(jit_funcs) > 1:
        names = ", ".join(func.name for func in jit_funcs)
        return ctx.make_finding(
            "OL01",
            "FAIL",
            f"{impl_file} 中存在 {len(jit_funcs)} 个 @pypto.frontend.jit 装饰的函数 "
            f"({names})；项目惯例要求每个 impl 文件有且仅有 1 个 JIT 入口（Layer J）。"
            f"请将多余的 JIT 入口合并为单一 kernel，子计算用普通函数（Layer H/I）承担。",
            file=impl_file,
            line=jit_funcs[0].lineno,
        )
    return ctx.make_finding(
        "OL01",
        "FAIL",
        f"{impl_file} 中未找到字面 @pypto.frontend.jit 装饰的函数。"
        f"OL01 仅接受唯一正规形 **@pypto.frontend.jit**（允许 @pypto.frontend.jit(...) "
        f"带参数调用语法）。**任何别名形式都被拒绝**，包括但不限于:\n"
        f"  - `import pypto as pt` + @pt.frontend.jit        ← 顶层包别名禁用\n"
        f"  - `import pypto.frontend as F` + @F.jit          ← 子模块别名禁用\n"
        f"  - `from pypto import frontend` + @frontend.jit   ← 子模块直接绑定禁用\n"
        f"  - `from pypto.frontend import jit` + @jit        ← 函数级 from-import 禁用\n"
        f"修复方式：导入语句改为 `import pypto`，装饰器严格写成 @pypto.frontend.jit。"
        f"不要保留任何别名。这是项目唯一约定，使 AST 静态分析、grep、IDE 跳转保持一致。",
        file=impl_file,
    )


@register("OL01")
def check_ol01(ctx: CheckContext) -> Finding:
    """kernel 函数必须有且仅有一个 @pypto.frontend.jit 装饰器。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    每个 impl 文件须各自满足 "有且仅有 1 个 JIT 入口"——module 文件即使在
    Stage 5 阶段单独开发，也应保持单 JIT 结构，避免拖到 Stage 6 集成时才暴露。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL01", "SKIP", "无 impl 文件可供检查")
    last_pass: tuple[str, ast.FunctionDef] | None = None
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        jit_funcs = _get_jit_functions(tree, aliases)
        failure = _ol01_failure(ctx, impl_file, jit_funcs)
        if failure is not None:
            return failure
        last_pass = impl_file, jit_funcs[0]
    if last_pass is None:
        return ctx.make_finding("OL01", "SKIP", "无 impl 文件可解析")
    impl_file, func = last_pass
    return ctx.make_finding(
        "OL01",
        "PASS",
        f"所有 impl 文件均含单一 @pypto.frontend.jit 入口（共 {len(impl_files)} 个）",
        file=impl_file,
        line=func.lineno,
    )


@register("OL02")
def check_ol02(ctx: CheckContext) -> Finding:
    """输出写回必须用 [:]/move()/assemble()，禁止 out = expr。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL02", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, _, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            failure = _ol02_function_failure(ctx, impl_file, func)
            if failure is not None:
                return failure
    if not saw_jit:
        return ctx.make_finding("OL02", "SKIP", "无 jit 函数")
    return ctx.make_finding("OL02", "PASS", "所有 impl 文件的输出写回方式正确")


def _ol02_function_failure(
    ctx: CheckContext,
    impl_file: str,
    func: ast.FunctionDef,
) -> Finding | None:
    param_names = {arg.arg for arg in func.args.args}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            target = next(
                (item for item in node.targets if isinstance(item, ast.Name) and item.id in param_names),
                None,
            )
            if target is not None:
                return ctx.make_finding(
                    "OL02",
                    "FAIL",
                    f"{impl_file} 中禁止 `{target.id} = expr` 写回，"
                    f"应使用 `{target.id}[:] = ...` 或 `{target.id}.move(...)`",
                    file=impl_file,
                    line=node.lineno,
                )
        if isinstance(node, ast.AugAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id in param_names:
                return ctx.make_finding(
                    "OL02",
                    "FAIL",
                    f"{impl_file} 中禁止 `{target.id} += expr` 写回，"
                    f"应使用 `{target.id}[:] = {target.id} + ...`",
                    file=impl_file,
                    line=node.lineno,
                )
    return None


@register("OL03")
def check_ol03(ctx: CheckContext) -> Finding:
    """kernel 函数不能有 return 语句。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL03", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, _, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            return_node = next((node for node in ast.walk(func) if isinstance(node, ast.Return)), None)
            if return_node is not None:
                return ctx.make_finding(
                    "OL03",
                    "FAIL",
                    f"{impl_file} 中 jit 函数 {func.name} 内存在 return 语句",
                    file=impl_file,
                    line=return_node.lineno,
                )
    if not saw_jit:
        return ctx.make_finding("OL03", "SKIP", "无 jit 函数")
    return ctx.make_finding("OL03", "PASS", "所有 impl 文件的 jit 函数均无 return 语句")


@register("OL04")
def check_ol04(ctx: CheckContext) -> Finding:
    """必须在 JIT 入口可达的 kernel/helper 中配置 tile shapes。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    每个 module impl 的 JIT 内核都需要自己的 tile 配置——Stage 5 阶段单独
    跑 module 测试时若漏配 tile shapes，会立即在编译期失败。Layer J 可以是
    一层很薄的 @jit 入口；tile 配置允许放在它调用到的 Layer I/H helper 中。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL04", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    last_pass = None
    for impl_file, tree, _, jit_funcs in _iter_jit_impls(ctx, impl_files):
        function_defs = {
            node.name: node
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef)
        }
        saw_jit = True
        found_tile_call = None
        for func in jit_funcs:
            found_tile_call = _find_reachable_tile_shape_call(func, function_defs)
            if found_tile_call is not None:
                break
        if found_tile_call is None:
            return ctx.make_finding(
                "OL04",
                "FAIL",
                f"{impl_file} 中 @pypto.frontend.jit 入口及其同文件可达 helper 内未找到 "
                "set_vec_tile_shapes 或 set_cube_tile_shapes 调用。"
                "修复方式：在 JIT 入口调用到的 Layer I `_kernel_impl` 或 Layer H "
                "`pypto_*` 子内核中设置 tile shapes；不要放在未被 JIT 调用链触达的死代码里。",
                file=impl_file,
            )
        last_pass = (impl_file, found_tile_call)
    if not saw_jit:
        return ctx.make_finding("OL04", "SKIP", "无 jit 函数")
    impl_file, node = last_pass
    return ctx.make_finding(
        "OL04",
        "PASS",
        f"所有 impl 文件的 JIT 调用链均含 tile shapes 配置调用（共 {len(impl_files)} 个）",
        file=impl_file,
        line=node.lineno,
    )


@register("OL05")
def check_ol05(ctx: CheckContext) -> Finding:
    """kernel 张量参数必须有 pypto.Tensor 类型注解。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL05", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            failure = _ol05_function_failure(ctx, impl_file, func, aliases)
            if failure is not None:
                return failure
    if not saw_jit:
        return ctx.make_finding("OL05", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL05", "PASS", "jit 函数张量参数均有 pypto.Tensor 类型注解"
    )


def _ol05_function_failure(ctx, impl_file, func, aliases):
    for arg in func.args.args:
        failure = _ol05_arg_failure(ctx, impl_file, func, arg, aliases)
        if failure is not None:
            return failure
    return None


def _ol05_arg_failure(
    ctx: CheckContext,
    impl_file: str,
    func: ast.FunctionDef,
    arg: ast.arg,
    aliases: set[str],
) -> Finding | None:
    if arg.annotation is None:
        return ctx.make_finding(
            "OL05",
            "FAIL",
            f"{impl_file} 中 jit 函数参数 `{arg.arg}` 缺少类型注解",
            file=impl_file,
            line=func.lineno,
        )
    if _is_non_tensor_annotation(arg.annotation, aliases):
        return None
    if _is_pypto_tensor_annotation(arg.annotation, aliases):
        return None
    return ctx.make_finding(
        "OL05",
        "FAIL",
        f"{impl_file} 中 jit 函数张量参数 `{arg.arg}` 注解必须为 pypto.Tensor",
        file=impl_file,
        line=func.lineno,
    )


@register("OL06")
def check_ol06(ctx: CheckContext) -> Finding:
    """kernel 内禁用 Python 原生 min()/max()。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL06", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, _, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            bad_call = _native_min_max_call(func)
            if bad_call is not None:
                return ctx.make_finding(
                    "OL06",
                    "FAIL",
                    f"{impl_file} 中 jit 函数内使用了 Python 原生 "
                    f"{bad_call.func.id}()，应使用 pypto 等价函数",
                    file=impl_file,
                    line=bad_call.lineno,
                )
    if not saw_jit:
        return ctx.make_finding("OL06", "SKIP", "无 jit 函数")
    return ctx.make_finding("OL06", "PASS", "所有 impl 文件均未使用原生 min/max")


def _native_min_max_call(func):
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
            return node
    return None


@register("OL07")
def check_ol07(ctx: CheckContext) -> Finding:
    """必须使用唯一正规导入 `import pypto`。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL07", "SKIP", "无 impl 文件可供检查")
    parsed_any = False
    for impl_file, tree, _ in _iter_parsed_impls(ctx, impl_files):
        parsed_any = True
        found_canonical, bad_import, bad_text = _pypto_import_status(tree)
        if bad_import is not None:
            return ctx.make_finding(
                "OL07",
                "FAIL",
                f"{impl_file} 使用了非正规 PyPTO 导入 {bad_text}。"
                "本 agent 算子开发流程只允许文件顶层写 `import pypto`，"
                "禁止 alias、`import pypto.frontend as F` 与 from-import。"
                "这与 OL01 的字面 @pypto.frontend.jit 要求保持一致，"
                "便于 AST 静态分析、grep 与自动修复。",
                file=impl_file,
                line=getattr(bad_import, "lineno", 0),
            )
        if not found_canonical:
            return ctx.make_finding(
                "OL07",
                "FAIL",
                f"{impl_file} 中未 import pypto——"
                "这是 PyPTO 算子实现的基础前提，缺少 import 说明该文件不是合法的 kernel 实现。"
                "修复方式：在文件顶层添加唯一正规导入 `import pypto`；不要使用 alias 或 from-import。",
                file=impl_file,
            )
    if not parsed_any:
        return ctx.make_finding("OL07", "SKIP", "无 impl 文件可解析")
    return ctx.make_finding(
        "OL07", "PASS", f"所有 impl 文件均使用正规 `import pypto`（共 {len(impl_files)} 个）"
    )


def _pypto_import_node_status(
    node: ast.AST,
) -> tuple[bool, ast.AST | None, str]:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "pypto" or module.startswith("pypto."):
            return False, node, f"`from {module} import ...`"
        return False, None, ""
    if not isinstance(node, ast.Import):
        return False, None, ""
    found_canonical = False
    bad_alias = None
    for alias in node.names:
        if alias.name == "pypto" and alias.asname is None:
            found_canonical = True
        if (
            (alias.name == "pypto" and alias.asname is not None)
            or alias.name.startswith("pypto.")
        ):
            bad_alias = alias
            break
    if bad_alias is None:
        return found_canonical, None, ""
    text = f"`import {bad_alias.name}"
    text += f" as {bad_alias.asname}`" if bad_alias.asname else "`"
    return found_canonical, node, text


def _pypto_import_status(tree: ast.Module) -> tuple[bool, ast.AST | None, str]:
    found_canonical = False
    for node in ast.iter_child_nodes(tree):
        node_canonical, bad_import, bad_text = _pypto_import_node_status(node)
        found_canonical = found_canonical or node_canonical
        if bad_import is not None:
            return found_canonical, bad_import, bad_text
    return found_canonical, None, ""


@register("OL08")
def check_ol08(ctx: CheckContext) -> Finding:
    """wrapper 函数必须导出且以 _wrapper 结尾。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    每个 module impl 文件也应自带 `<op>_module<k>_wrapper`——这是 Stage 5
    module 测试与 Stage 6 集成的统一入口约定。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL08", "SKIP", "无 impl 文件可供检查")
    last_pass: tuple[str, ast.FunctionDef] | None = None
    for impl_file, tree, _ in _iter_parsed_impls(ctx, impl_files):
        wrapper = next(
            (
                node for node in ast.iter_child_nodes(tree)
                if isinstance(node, ast.FunctionDef) and node.name.endswith("_wrapper")
            ),
            None,
        )
        if wrapper is None:
            return ctx.make_finding(
                "OL08",
                "FAIL",
                f"{impl_file} 中未找到以 _wrapper 结尾的模块级函数",
                file=impl_file,
            )
        last_pass = (impl_file, wrapper)
    if last_pass is None:
        return ctx.make_finding("OL08", "SKIP", "无 impl 文件可解析")
    impl_file, wrapper = last_pass
    return ctx.make_finding(
        "OL08",
        "PASS",
        f"所有 impl 文件均含 _wrapper 函数（共 {len(impl_files)} 个）",
        file=impl_file,
        line=wrapper.lineno,
    )


@register("OL23")
def check_ol23(ctx: CheckContext) -> Finding:
    """impl 中需检测到 loop 相关结构，否则 WARN。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。Production
    kernel 4 要素之一就是 pypto.loop——若 module 漏写 loop，会在 production
    shape 上 workspace estimator INT32 溢出，必须在 Stage 5 module 测试
    阶段就被警示。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL23", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    files_without_loop: list[str] = []
    last_pass_file = None
    for impl_file in impl_files:
        syntax_error = _syntax_error_finding(ctx, "OL23", impl_file)
        if syntax_error:
            return syntax_error
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        aliases = ctx.pypto_aliases(impl_file)
        jit_funcs = _get_primary_jit_functions(tree, aliases)
        if not jit_funcs:
            continue
        saw_jit = True
        if any(_has_loop_structure(func) for func in jit_funcs):
            last_pass_file = impl_file
            continue
        files_without_loop.append(impl_file)
    if not saw_jit:
        return ctx.make_finding("OL23", "SKIP", "无 jit 函数")
    if files_without_loop:
        return ctx.make_finding(
            "OL23",
            "WARN",
            "以下 impl 文件未检测到 loop 相关结构；若算子需要分块或迭代，"
            "请确认设计已说明无需 loop：" + ", ".join(files_without_loop),
            file=files_without_loop[0],
        )
    return ctx.make_finding(
        "OL23", "PASS", "所有 impl 文件均检测到 loop 相关结构", file=last_pass_file
    )


@register("OL25")
def check_ol25(ctx: CheckContext) -> Finding:
    """Tensor 注解完整性检查：无参数或缺少 dtype 时告警。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。空 [] 注解
    在模块开发阶段（Stage 5 Phase M_k）即被拦截，避免下沉到集成 cleanup
    才暴露。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL25", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            finding = _ol25_function_finding(ctx, impl_file, func, aliases)
            if finding is not None:
                return finding
    if not saw_jit:
        return ctx.make_finding("OL25", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL25", "PASS", "JIT Tensor 注解均包含 shape 与 dtype"
    )


def _ol25_function_finding(ctx, impl_file, func, aliases):
    for arg in func.args.args:
        finding = _ol25_arg_finding(ctx, impl_file, arg, aliases)
        if finding is not None:
            return finding
    return None


def _ol25_arg_finding(
    ctx: CheckContext,
    impl_file: str,
    arg: ast.arg,
    aliases: set[str],
) -> Finding | None:
    annotation = arg.annotation
    if not isinstance(annotation, ast.Call):
        return None
    if not _is_pypto_tensor_annotation(annotation, aliases):
        return None
    if not annotation.args:
        return ctx.make_finding(
            "OL25",
            "FAIL",
            f"{impl_file} 中参数 `{arg.arg}` 使用 pypto.Tensor()（无参数形式）；"
            "动态轴必须显式标 pypto.DYNAMIC，静态轴写常量整数，"
            "禁止使用空注解。例：pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)",
            file=impl_file,
            line=annotation.lineno,
        )
    shape = annotation.args[0]
    if isinstance(shape, ast.List) and not shape.elts:
        return ctx.make_finding(
            "OL25",
            "FAIL",
            f"{impl_file} 中参数 `{arg.arg}` 使用 pypto.Tensor([], ...) 空 shape 注解；"
            "动态轴必须显式标 pypto.DYNAMIC，静态轴写常量整数。"
            "（per-shape compile 写法已废弃，详见 DEBUG_GUIDEBOOK §9.13）",
            file=impl_file,
            line=annotation.lineno,
        )
    if len(annotation.args) != 1:
        return None
    return ctx.make_finding(
        "OL25",
        "WARN",
        f"{impl_file} 中参数 `{arg.arg}` 只声明了 shape，缺少 dtype。"
        "建议写成 pypto.Tensor([shape], dtype)",
        file=impl_file,
        line=annotation.lineno,
    )


@register("OL26")
def check_ol26(ctx: CheckContext) -> Finding:
    """JIT 函数中张量参数必须在非张量参数之前。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL26", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            bad_arg = _misordered_tensor_arg(func, aliases)
            if bad_arg is not None:
                return ctx.make_finding(
                    "OL26",
                    "FAIL",
                    f"{impl_file} 中 jit 函数 {func.name} 的张量参数 `{bad_arg.arg}` "
                    f"出现在非张量参数之后，JIT 要求张量参数在前、非张量参数在后",
                    file=impl_file,
                    line=func.lineno,
                )
    if not saw_jit:
        return ctx.make_finding("OL26", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL26",
        "PASS",
        "所有 impl 文件的 jit 函数参数顺序正确（张量在前、标量在后）",
    )


def _misordered_tensor_arg(func: ast.FunctionDef, aliases: set[str]) -> ast.arg | None:
    seen_non_tensor = False
    for arg in func.args.args:
        annotation = arg.annotation
        if annotation is None:
            continue
        if _is_non_tensor_annotation(annotation, aliases):
            seen_non_tensor = True
        elif seen_non_tensor and _is_pypto_tensor_annotation(annotation, aliases):
            return arg
    return None


@register("OL28")
def check_ol28(ctx: CheckContext) -> Finding:
    """sigmoid/softmax/sin/cos 仅支持 DT_FP32，非 FP32 dtype 时警告。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL28", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            finding = _ol28_function_finding(ctx, impl_file, func, aliases)
            if finding is not None:
                return finding
    if not saw_jit:
        return ctx.make_finding("OL28", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL28", "PASS", "所有 impl 文件中 FP32-only API 与 dtype 注解一致"
    )


def _ol28_function_finding(
    ctx: CheckContext,
    impl_file: str,
    func: ast.FunctionDef,
    aliases: set[str],
) -> Finding | None:
    used_apis = {
        node.func.attr
        for node in ast.walk(func)
        if _is_fp32_only_call(node, aliases)
    }
    if not used_apis:
        return None
    for arg in func.args.args:
        annotation = arg.annotation
        if not isinstance(annotation, ast.Call):
            continue
        if not _is_pypto_tensor_annotation(annotation, aliases) or len(annotation.args) < 2:
            continue
        if "DT_FP32" in ast.dump(annotation.args[1]):
            continue
        return ctx.make_finding(
            "OL28",
            "WARN",
            f"{impl_file} 中 jit 函数使用了仅支持 DT_FP32 的 API "
            f"({', '.join(sorted(used_apis))})，"
            f"但参数 `{arg.arg}` 的 dtype 不是 DT_FP32，"
            "请确认已正确处理 dtype 转换（cast）",
            file=impl_file,
            line=func.lineno,
        )
    return None


@register("OL29")
def check_ol29(ctx: CheckContext) -> Finding:
    """Tensor 注解的 shape 中应声明 pypto.DYNAMIC/pypto.DYN 维度。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。每个文件
    单独评估，任一文件均缺少 DYNAMIC 声明时返回 WARN（按文件名定位）。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL29", "SKIP", "无 impl 文件可供检查")
    files_without_dynamic: list[str] = []
    files_with_tensor_count = 0
    last_file_seen = ""
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        jit_funcs = _get_primary_jit_functions(tree, aliases)
        if not jit_funcs:
            continue
        dynamic_aliases = _extract_symbolic_dynamic_aliases(tree, aliases)
        tensor_count, has_dynamic = _dynamic_tensor_annotation_status(
            jit_funcs, aliases, dynamic_aliases,
        )
        if tensor_count == 0:
            continue
        files_with_tensor_count += 1
        last_file_seen = impl_file
        if not has_dynamic:
            files_without_dynamic.append(impl_file)
    if files_with_tensor_count == 0:
        return ctx.make_finding("OL29", "SKIP", "无 Tensor 注解")
    if files_without_dynamic:
        return ctx.make_finding(
            "OL29",
            "WARN",
            f"以下文件的 Tensor 注解均未声明 pypto.DYNAMIC/pypto.DYN: "
            f"{', '.join(files_without_dynamic)}。"
            "若有输入维度在运行时可变，必须标记为 DYNAMIC 以避免重编译",
            file=files_without_dynamic[0],
        )
    return ctx.make_finding(
        "OL29", "PASS", "Tensor 注解中包含 DYNAMIC 维度声明", file=last_file_seen
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL45 / OL46 / OL47 辅助函数 — Layer K、pypto.loop、tile-shape 作用域
# ─────────────────────────────────────────────────────────────────────────────


# `_impl_files_to_scan` 已迁移至 `..utils`，供 D1/D3/D5 各检查共享使用。


def _is_wrapper_function(name: str, op_name: str) -> bool:
    """Layer K 包装函数命名模式 — `host_wrapper`、`<op>_wrapper`、
    `<op>_module<suffix>_wrapper`、`launch_*`、`run_*`。"""
    if name == "host_wrapper":
        return True
    if name == f"{op_name}_wrapper":
        return True
    if name.startswith(f"{op_name}_module") and name.endswith("_wrapper"):
        return True
    if name.startswith("launch_") or name.startswith("run_"):
        return True
    return False


def _is_kernel_impl_function(name: str) -> bool:
    """Layer I 实现函数体 — 以 `_kernel_impl` 结尾或符合
    design-format 约定的 `_impl` 后缀命名。"""
    return "_kernel_impl" in name or name.endswith("_impl")


def _calls_to_kernel_or_jit(node: ast.AST, jit_names: set[str]) -> bool:
    """若 for 循环体中包含对 JIT 入口函数的调用则返回 True。"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id in jit_names:
                return True
            if isinstance(f, ast.Attribute) and f.attr in jit_names:
                return True
    return False


def _is_pypto_loop_call(node: ast.AST, aliases) -> tuple[bool, int | None]:
    """识别 `pypto.loop(N)` / `pypto.loop(...)` 调用。返回 (is_loop, N 或 None)。"""
    if not isinstance(node, ast.Call):
        return (False, None)
    f = node.func
    is_loop = False
    if isinstance(f, ast.Attribute) and f.attr == "loop":
        if isinstance(f.value, ast.Name) and f.value.id in aliases:
            is_loop = True
    if not is_loop:
        return (False, None)
    n_val: int | None = None
    if node.args:
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, int):
            n_val = a.value
    return (True, n_val)


def _calls_pypto_helper(node: ast.AST) -> int:
    """统计 `node` 内直接调用以 `pypto_` 开头的函数次数
    （Layer H 子 kernel 命名约定），排除 `pypto.<attr>(...)` API 调用。"""
    count = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id.startswith("pypto_"):
                count += 1
    return count


def _function_defs(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    return (
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _forbidden_wrapper_loop(
    wrapper: ast.FunctionDef | ast.AsyncFunctionDef,
    jit_names: set[str],
) -> ast.For | None:
    for node in ast.walk(wrapper):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            continue
        if _calls_to_kernel_or_jit(node, jit_names):
            return node
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OL45 — Layer K（宿主 wrapper）中禁止用 Python `for ... in range(...)` 分块调用 kernel
# ─────────────────────────────────────────────────────────────────────────────


@register("OL45")
def check_ol45(ctx: CheckContext) -> Finding:
    """Layer K 禁止包含通过 Python `for ... in range(...)` 逐块调用
    kernel 的循环。分块逻辑应通过 pypto.loop(N) + pypto.view offsets
    放在 `_kernel_impl` 内部。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL45", "SKIP", "无 impl 文件可供检查")
    for impl_file, tree, _, jit_funcs in _iter_jit_impls(ctx, impl_files):
        jit_names = {f.name for f in jit_funcs}
        for top in _function_defs(tree):
            if not _is_wrapper_function(top.name, ctx.op_name):
                continue
            bad_loop = _forbidden_wrapper_loop(top, jit_names)
            if bad_loop is not None:
                return ctx.make_finding(
                    "OL45",
                    "FAIL",
                    f"Layer K 包装函数 `{top.name}` 包含 Python "
                    f"`for ... in range(...)` 逐块调用 JIT kernel。"
                    f"请将分块迭代移入 `_kernel_impl`，改用 "
                    f"`pypto.loop(NT)` + `pypto.view(..., offsets=[nt*BT, ...])`。"
                    f"包装函数必须仅调用 kernel 一次。",
                    file=impl_file,
                    line=bad_loop.lineno,
                )
    return ctx.make_finding(
        "OL45",
        "PASS",
        "Layer K 包装函数未通过 Python 循环分块调用 kernel",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL46 — 冗余的 `pypto.loop(1)` 包装内层 pypto.loop(N)
# ─────────────────────────────────────────────────────────────────────────────


@register("OL46")
def check_ol46(ctx: CheckContext) -> Finding:
    """`pypto.loop(1)` 仅在作用域内不存在其他 pypto.loop(N) 时才合法。
    用 `pypto.loop(1)` 包装内层的 `pypto.loop(N)` 是冗余且禁止的。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL46", "SKIP", "无 impl 文件可供检查")
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        for top in _function_defs(tree):
            if not _is_kernel_impl_function(top.name):
                continue
            loops = _kernel_loops(top, aliases)
            if not loops:
                continue
            has_loop_one = any(n == 1 for n, _ in loops)
            has_loop_n = any(
                n is None or (isinstance(n, int) and n != 1) for n, _ in loops
            )
            if has_loop_one and has_loop_n:
                line = next((ln for n, ln in loops if n == 1), -1)
                return ctx.make_finding(
                    "OL46",
                    "WARN",
                    f"{impl_file}: `{top.name}` 用 `pypto.loop(1)` 包装了内层 "
                    f"`pypto.loop(N)`。请移除外层 `pypto.loop(1)` — 它仅用于"
                    f"作用域内不存在其他 pypto.loop 的场景（布局检查要求的"
                    f"vector-pipe 简单算子）。",
                    file=impl_file,
                    line=line if line > 0 else None,
                )
    return ctx.make_finding(
        "OL46",
        "PASS",
        "未检测到冗余的 pypto.loop(1) 包装",
    )


def _kernel_loops(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: set[str],
) -> list[tuple[int | None, int]]:
    loops: list[tuple[int | None, int]] = []
    for node in ast.walk(func):
        is_loop, count = _is_pypto_loop_call(node, aliases)
        if is_loop:
            loops.append((count, getattr(node, "lineno", -1)))
    return loops


# ─────────────────────────────────────────────────────────────────────────────
# OL47 — _kernel_impl 中仅设置一次全局 tile-shape 却调用多个子 kernel
# ─────────────────────────────────────────────────────────────────────────────


@register("OL47")
def check_ol47(ctx: CheckContext) -> Finding:
    """当 `_kernel_impl` 自身调用 `set_*_tile_shapes(...)` 同时调用了
    2 个及以上 `pypto_*` 子 kernel 时，可能存在逐阶段 tile shape 优化
    机会被遗漏。建议将 tile 配置下推到每个子 kernel 内部。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL47", "SKIP", "无 impl 文件可供检查")
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        for top in _function_defs(tree):
            if not _is_kernel_impl_function(top.name):
                continue
            tile_in_impl = _top_level_tile_count(top, aliases)
            helper_calls = _calls_pypto_helper(top)
            if tile_in_impl >= 1 and helper_calls >= 2:
                return ctx.make_finding(
                    "OL47",
                    "INFO",
                    f"{impl_file}: `{top.name}` 在 `_kernel_impl` 顶层设置了 tile shapes"
                    f"同时调用了 {helper_calls} 个 `pypto_*` 子 kernel。"
                    f"建议将每个 `set_*_tile_shapes` 移到对应的子 kernel 内部，"
                    f"使各阶段的 matmul/vec 操作能使用各自最优的 tile 布局。",
                    file=impl_file,
                )
    return ctx.make_finding(
        "OL47",
        "PASS",
        "tile-shape 作用域配置看起来与 kernel 结构匹配",
    )


def _top_level_tile_count(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: set[str],
) -> int:
    values = [
        node.value for node in ast.iter_child_nodes(func)
        if isinstance(node, (ast.Expr, ast.Assign))
    ]
    return sum(_is_tile_call(value, aliases) for value in values)


# ─────────────────────────────────────────────────────────────────────────────
# OL48 — set_*_tile_shapes 参数必须编译期静态可知
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_to_const_int(node: ast.AST, scope: dict, max_depth: int = 10) -> bool:
    """节点能否解析到编译期 int 常量。

    允许：
    - ast.Constant(int) 字面量
    - ast.Name，其名字在 scope 中解析到上述形式（递归）

    禁止：函数参数、Subscript（x.shape[i]）、Attribute、Call、BinOp、
    Compare、SymbolicScalar 表达式等非编译期值。
    """
    if max_depth <= 0:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return True
    if isinstance(node, ast.Name) and node.id in scope:
        return _resolve_to_const_int(scope[node.id], scope, max_depth - 1)
    return False


def _collect_module_const_assigns(tree: ast.Module) -> dict:
    """模块顶层 `name = <expr>` 的 expr 节点（用于后续解析）。"""
    scope: dict = {}
    for n in ast.iter_child_nodes(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            target = n.targets[0]
            if isinstance(target, ast.Name):
                scope[target.id] = n.value
    return scope


def _collect_func_local_assigns(func: ast.FunctionDef, before_lineno: int) -> dict:
    """函数体内出现在 `before_lineno` 之前的 `name = <expr>` 赋值。
    按源代码行号顺序覆盖，后赋值的胜出。函数参数本身不会进入此 scope。
    """
    relevant = sorted(
        [n for n in ast.walk(func)
         if isinstance(n, ast.Assign) and getattr(n, "lineno", 0) < before_lineno],
        key=lambda n: n.lineno,
    )
    scope: dict = {}
    for n in relevant:
        if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            scope[n.targets[0].id] = n.value
    return scope


def _iter_tile_args(call_node: ast.Call):
    """flatten tile call args：list 元素也展开（cube tile 的 [L0, L1] 形式）。"""
    for arg in call_node.args:
        if isinstance(arg, ast.List):
            for elt in arg.elts:
                yield elt
        else:
            yield arg


def _is_tile_call(node: ast.AST, aliases) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in (
        "set_vec_tile_shapes",
        "set_cube_tile_shapes",
    ):
        if isinstance(f.value, ast.Name) and f.value.id in aliases:
            return True
    return False


def _format_expr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return f"<{type(node).__name__}>"


def _get_positional_or_kw(call_node: ast.Call, pos_index: int, kw_name: str):
    """取 call 的位置参数 pos_index 或同名 keyword 参数节点，缺失返回 None。"""
    if len(call_node.args) > pos_index:
        return call_node.args[pos_index]
    for kw in call_node.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _int_literal_value(node) -> int | None:
    """正/任意 int 字面量的值；非 int 字面量 (含 bool) 返回 None。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _cube_tile_pair_errors(axis: str, node) -> list[str]:
    """校验 set_cube_tile_shapes 的某一轴 (m/k/n)：必须是 2 元素 list `[L0, L1]`，
    且当 L0/L1 为 int 字面量时 0 < L0 <= L1 且 L1 % L0 == 0。

    非 literal list (变量引用) 时跳过 (literal 性由主检查覆盖)。
    """
    errs: list[str] = []
    if not isinstance(node, ast.List):
        return errs
    n = len(node.elts)
    if n != 2:
        errs.append(
            f"`{axis}` 必须是 2 元素 list `[{axis}L0, {axis}L1]`，实际为 {n} 元素 list "
            f"(set_cube_tile_shapes 每轴需 [L0, L1]，不能是单元素 [L0])"
        )
        return errs
    l0 = _int_literal_value(node.elts[0])
    l1 = _int_literal_value(node.elts[1])
    if l0 is None or l1 is None:
        return errs
    if l0 <= 0 or l1 <= 0:
        errs.append(f"`{axis}` 取值必须为正：得到 [{l0}, {l1}]")
        return errs
    if l0 > l1:
        errs.append(f"`{axis}` 要求 {axis}L0 <= {axis}L1，得到 [{l0}, {l1}]")
    if l1 % l0 != 0:
        errs.append(
            f"`{axis}` 要求 {axis}L1 % {axis}L0 == 0，得到 [{l0}, {l1}] "
            f"({l1} % {l0} = {l1 % l0})"
        )
    return errs


TileViolation = tuple[str, str, int]
CubeViolation = tuple[str, int]


def _tile_call_issues(
    func: ast.FunctionDef,
    aliases: set[str],
    module_scope: dict,
) -> tuple[list[TileViolation], list[CubeViolation]]:
    violations: list[TileViolation] = []
    cube_violations: list[CubeViolation] = []
    for node in ast.walk(func):
        if not _is_tile_call(node, aliases):
            continue
        local_scope = dict(module_scope)
        local_scope.update(_collect_func_local_assigns(func, node.lineno))
        call_name = node.func.attr  # type: ignore[union-attr]
        violations.extend(
            (call_name, _format_expr(arg), node.lineno)
            for arg in _iter_tile_args(node)
            if not _resolve_to_const_int(arg, local_scope)
        )
        if call_name == "set_cube_tile_shapes":
            cube_violations.extend(_cube_call_issues(node))
    return violations, cube_violations


def _cube_call_issues(node: ast.Call) -> list[CubeViolation]:
    issues = []
    for axis, index in (("m", 0), ("k", 1), ("n", 2)):
        pair = _get_positional_or_kw(node, index, axis)
        for message in _cube_tile_pair_errors(axis, pair):
            issues.append((message, node.lineno))
    return issues


def _tile_violation_message(
    impl_file: str,
    violations: list[TileViolation],
    cube_violations: list[CubeViolation],
) -> str:
    lines = [f"{impl_file} 中 tile 参数违规："]
    if violations:
        lines.append(
            "· 非编译期静态值（必须是 Python int 字面量或解析到字面量的局部/模块 Assign）："
        )
        lines.extend(
            f"  - {tile_call} 第 {line} 行: `{expression}`"
            for tile_call, expression, line in violations
        )
    if cube_violations:
        lines.append(
            "· set_cube_tile_shapes 的 tile 列表结构 / 整除违规 "
            "(参考 docs/zh/api/config/pypto-set_cube_tile_shapes.md)："
        )
        lines.extend(f"  - 第 {line} 行: {message}" for message, line in cube_violations)
    lines.append(
        "禁止用 kernel 入参、tensor.shape[i]、SymbolicScalar、运行时计算等动态值作为 "
        "tile shape；set_cube_tile_shapes 每轴须为 [L0, L1] 且 0<L0<=L1、L1%L0==0。"
    )
    return "\n".join(lines)


@register("OL48")
def check_ol48(ctx: CheckContext) -> Finding:
    """set_vec_tile_shapes / set_cube_tile_shapes 的每个 tile 参数（含 list 元素）
    必须是 Python int 字面量或解析到字面量的局部/模块 Assign。

    禁止：函数参数、tensor.shape[i]、SymbolicScalar、运行时计算。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。module impl
    的 tile 参数同样必须编译期静态——否则 Stage 5 module 测试一编译就会失败。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL48", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, tree, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        module_scope = _collect_module_const_assigns(tree)
        violations: list[TileViolation] = []
        cube_struct: list[CubeViolation] = []
        for func in jit_funcs:
            func_violations, func_cube_struct = _tile_call_issues(
                func, aliases, module_scope,
            )
            violations.extend(func_violations)
            cube_struct.extend(func_cube_struct)
        if violations or cube_struct:
            first_line = violations[0][2] if violations else cube_struct[0][1]
            return ctx.make_finding(
                "OL48",
                "FAIL",
                _tile_violation_message(impl_file, violations, cube_struct),
                file=impl_file,
                line=first_line,
            )
    if not saw_jit:
        return ctx.make_finding("OL48", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL48",
        "PASS",
        f"所有 impl 文件的 tile 参数均为编译期静态值（共 {len(impl_files)} 个）",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL49 — unroll_list 只能出现在最内层 pypto.loop
# ─────────────────────────────────────────────────────────────────────────────


def _is_pypto_loop_for_node(node: ast.AST, aliases) -> ast.Call | None:
    """若 `node` 是 `for x in pypto.loop(...):` 形式，返回 pypto.loop 的 Call 节点；否则返回 None。"""
    if not isinstance(node, ast.For):
        return None
    iter_call = node.iter
    if not isinstance(iter_call, ast.Call):
        return None
    f = iter_call.func
    if not isinstance(f, ast.Attribute) or f.attr != "loop":
        return None
    if not isinstance(f.value, ast.Name):
        return None
    if f.value.id not in aliases:
        return None
    return iter_call


def _has_unroll_list_kwarg(call_node: ast.Call) -> bool:
    for kw in call_node.keywords:
        if kw.arg == "unroll_list":
            return True
    return False


def _has_inner_pypto_loop(for_node: ast.For, aliases) -> bool:
    """检查 `for_node.body` / `for_node.orelse` 内是否还嵌套了另一个 `for ... in pypto.loop(...):`。
    注意：只扫描 body 和 orelse，不扫描 iter（避免误把 outer 自己当 inner）。"""
    for child in list(for_node.body) + list(for_node.orelse):
        for inner in ast.walk(child):
            if _is_pypto_loop_for_node(inner, aliases) is not None:
                return True
    return False


def _outer_unroll_loop(func: ast.FunctionDef, aliases: set[str]) -> ast.Call | None:
    for node in ast.walk(func):
        loop_call = _is_pypto_loop_for_node(node, aliases)
        if loop_call is None or not _has_unroll_list_kwarg(loop_call):
            continue
        if _has_inner_pypto_loop(node, aliases):
            return loop_call
    return None


@register("OL49")
def check_ol49(ctx: CheckContext) -> Finding:
    """`unroll_list` 只能出现在最内层 `pypto.loop`。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。

    嵌套 `pypto.loop` 场景下，若外层 loop 含 `unroll_list`，会触发：
      - 编译路径爆炸（指数级 root function 数）
      - 寄存器拷贝 pass bug 引发的精度异常（参见 pypto-precision-debug Issue #223, #341）
    项目惯例：unroll_list 仅放在最内层 pypto.loop；外层禁止。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL49", "SKIP", "无 impl 文件可供检查")
    saw_jit = False
    for impl_file, _, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            loop_call = _outer_unroll_loop(func, aliases)
            if loop_call is not None:
                return ctx.make_finding(
                    "OL49",
                    "FAIL",
                    f"{impl_file} 第 {loop_call.lineno} 行：`pypto.loop(..., unroll_list=...)` "
                    f"出现在外层（其 body 内还嵌套了另一个 pypto.loop）。"
                    f"unroll_list **只能放在最内层** pypto.loop —— 外层加 unroll_list 会触发"
                    f"编译路径爆炸或寄存器拷贝 pass 引起的精度异常。"
                    f"请把 unroll_list 移到最内层 pypto.loop（或删除外层 unroll_list）。",
                    file=impl_file,
                    line=loop_call.lineno,
                )
    if not saw_jit:
        return ctx.make_finding("OL49", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL49",
        "PASS",
        "所有 unroll_list 均位于最内层 pypto.loop",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL56 — Stage 6 之前 unroll_list 只能含单一值（默认 [1]）
# ─────────────────────────────────────────────────────────────────────────────


def _unroll_list_value_count(loop_call: ast.Call) -> int | None:
    """返回 `pypto.loop(..., unroll_list=[...])` 中 unroll_list 的元素个数。

    仅当 unroll_list 是 List / Tuple 字面量时返回元素数；若为变量、函数
    调用等非字面量（无法静态判断），返回 None（视为不可判定，按 PASS 处理，
    避免误报）。
    """
    for kw in loop_call.keywords:
        if kw.arg == "unroll_list":
            value = kw.value
            if isinstance(value, (ast.List, ast.Tuple)):
                return len(value.elts)
            return None
    return None


def _find_multivalue_unroll(tree: ast.AST, aliases) -> list[int]:
    """遍历 tree，返回所有 unroll_list 元素数 >= 2 的 pypto.loop 行号。"""
    hits: list[int] = []
    for node in ast.walk(tree):
        loop_call = _is_pypto_loop_for_node(node, aliases)
        if loop_call is None:
            continue
        count = _unroll_list_value_count(loop_call)
        if count is not None and count >= 2:
            hits.append(loop_call.lineno)
    return hits


def _scan_design_md_for_multivalue_unroll(ctx: CheckContext, filename: str) -> bool:
    """扫描 DESIGN.md 内的 ```python``` 代码块，若任一块存在多值 unroll_list 返回 True。"""
    source = ctx.read_file(filename)
    if not source:
        return False
    for block in extract_python_blocks(source):
        try:
            block_tree = ast.parse(block)
            aliases = _resolve_pypto_aliases(block_tree)
        except SyntaxError:
            continue
        if _find_multivalue_unroll(block_tree, aliases):
            return True
    return False


@register("OL56")
def check_ol56(ctx: CheckContext) -> Finding:
    """Stage 6 之前 `pypto.loop` 的 `unroll_list` 只能含单一值。

    覆盖范围（与 OL55 相同的两个着火点）：
    - `DESIGN.md`：仅扫描 Markdown 中的 ```python``` 代码块（Designer 写完
      DESIGN.md 后着火）。
    - `<op>_impl.py` / `modules/<op>_module*_impl.py`：扫描 JIT 函数体
      （Coder 写完 impl 后着火）。

    多值 `unroll_list`（如 `[16, 8, 4, 2, 1]`）会为每个迭代次数生成一条
    编译路径，导致编译路径爆炸、显著拖慢编译，进而使开发流程超时。Stage 6
    之前应固定单一值（默认 `[1]`，关闭循环展开；有依据时也可用其它单值）。
    多值展开调优仅允许在 Stage 7 optimization 阶段进行——故本规则的 stages
    为 [4, 5, 6]，不含 7。
    """
    # ── DESIGN.md（design 着火点）─────────────────────────────────────────
    design_scope_ok = (
        ctx.file_scope is None
        or _basename_match_ol56(ctx.file_scope, "DESIGN.md")
    )
    if ctx.file_exists("DESIGN.md") and design_scope_ok:
        if _scan_design_md_for_multivalue_unroll(ctx, "DESIGN.md"):
            return ctx.make_finding(
                "OL56",
                "FAIL",
                "DESIGN.md 的 ```python``` 代码块中存在多值 "
                "`pypto.loop(..., unroll_list=[...])`。Stage 6 之前 unroll_list "
                "只能含单一值（默认 `[1]`；有依据时可用其它单值），多值会触发"
                "编译路径爆炸、拖慢编译并导致开发超时。多值展开调优请留到 "
                "Stage 7 optimization。",
                file="DESIGN.md",
            )

    # ── impl（implementation 着火点）──────────────────────────────────────
    impl_files = _impl_files_to_scan(ctx)
    saw_jit = False
    for impl_file, _, aliases, jit_funcs in _iter_jit_impls(ctx, impl_files):
        saw_jit = True
        for func in jit_funcs:
            hits = _find_multivalue_unroll(func, aliases)
            if hits:
                return ctx.make_finding(
                    "OL56",
                    "FAIL",
                    f"{impl_file} 第 {hits[0]} 行："
                    f"`pypto.loop(..., unroll_list=[...])` 含 2 个及以上值。"
                    f"Stage 6 之前 unroll_list 只能含单一值（默认 `[1]`；有依据"
                    f"时可用其它单值）——多值会触发编译路径爆炸、拖慢编译并导致"
                    f"开发超时。请把 unroll_list 改成单一值（多值展开调优留到 "
                    f"Stage 7 optimization）。",
                    file=impl_file,
                    line=hits[0],
                )

    if not ctx.file_exists("DESIGN.md") and not saw_jit:
        return ctx.make_finding("OL56", "SKIP", "无 DESIGN.md / jit 函数可供检查")
    return ctx.make_finding(
        "OL56",
        "PASS",
        "所有 unroll_list 均为单一值（Stage 6 之前约束）",
    )


def _basename_match_ol56(file_scope: str, candidate: str) -> bool:
    """file_scope（post-edit hook 传入的路径）与候选基名是否匹配。"""
    return os.path.basename(file_scope) == os.path.basename(candidate)


# ─────────────────────────────────────────────────────────────────────────────
# OL52 — pypto.view(t, shape=[...], offsets=[...]) 的 list 长度一致
# ─────────────────────────────────────────────────────────────────────────────


def _is_pypto_view_call(node: ast.AST, aliases) -> bool:
    """True iff node is `pypto.view(...)` (with alias support)."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if not isinstance(f, ast.Attribute) or f.attr != "view":
        return False
    if not isinstance(f.value, ast.Name) or f.value.id not in aliases:
        return False
    return True


def _get_view_arg(call: ast.Call, pos_index: int, kw_name: str) -> ast.AST | None:
    """Return the AST node for the arg at `pos_index` or matching keyword `kw_name`."""
    if len(call.args) > pos_index:
        return call.args[pos_index]
    for kw in call.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _literal_list_len(node: ast.AST | None) -> int | None:
    """Return len(elts) if node is a literal `[...]` or `(...,)`; else None."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    return None


def _view_rank_finding(
    ctx: CheckContext,
    impl_file: str,
    node: ast.Call,
) -> Finding | None:
    shape_len = _literal_list_len(_get_view_arg(node, 1, "shape"))
    offsets_len = _literal_list_len(_get_view_arg(node, 2, "offsets"))
    valid_len = _literal_list_len(_get_view_arg(node, 3, "valid_shape"))
    line = getattr(node, "lineno", -1)
    if shape_len is not None and offsets_len is not None and shape_len != offsets_len:
        return ctx.make_finding(
            "OL52",
            "FAIL",
            f"{impl_file} 第 {line} 行: `pypto.view(...)` 的 shape/offsets "
            f"rank 不一致 (shape={shape_len} dims, offsets={offsets_len} dims)。"
            f"pypto.view 不是 reshape, 而是抽取 **同 rank 的 sub-view** 的 API。"
            f"请将两个 list 的长度对齐。若要改变 rank, 应使用 `pypto.reshape(...)`。"
            f"(参考 `docs/zh/api/operation/pypto-view.md`, "
            f"`skills/pypto-general-debug/references/DEBUG_GUIDEBOOK.md` §9.4)",
            file=impl_file,
            line=line if line > 0 else None,
        )
    valid_differs = (
        shape_len is not None
        and valid_len is not None
        and valid_len != 0
        and shape_len != valid_len
    )
    if not valid_differs:
        return None
    return ctx.make_finding(
        "OL52",
        "FAIL",
        f"{impl_file} 第 {line} 行: `pypto.view(...)` 的 shape/valid_shape "
        f"rank 不一致 (shape={shape_len} dims, valid_shape={valid_len} dims)。"
        f"shape, offsets, valid_shape 三者必须 rank 一致。"
        f"(参考 `docs/zh/api/operation/pypto-view.md`)",
        file=impl_file,
        line=line if line > 0 else None,
    )


@register("OL52")
def check_ol52(ctx: CheckContext) -> Finding:
    """`pypto.view(t, shape=[...], offsets=[...])` 的 shape / offsets / valid_shape
    必须 rank 一致 (list 长度相同)。

    pypto.view 是同 rank 的 sub-view 抽取 API, 不是改变 rank 的 reshape。
    当三个参数均为 list literal 时, 通过静态检查捕获长度不一致。

    (参考 `docs/zh/api/operation/pypto-view.md`,
     `skills/pypto-general-debug/references/DEBUG_GUIDEBOOK.md` §9.4)
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL52", "SKIP", "无 impl 文件可供检查")
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        for node in ast.walk(tree):
            if not _is_pypto_view_call(node, aliases):
                continue
            finding = _view_rank_finding(ctx, impl_file, node)
            if finding is not None:
                return finding
    return ctx.make_finding(
        "OL52",
        "PASS",
        "pypto.view 的 shape/offsets/valid_shape 全部 rank 一致 (或非 literal)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL57 — @jit 图代码内允许 pypto.loop / pypto.loop_unroll / range 循环；禁止 while 和非 range 的 for
# ─────────────────────────────────────────────────────────────────────────────


def _is_pypto_loop_iter(node: ast.AST, aliases) -> bool:
    """判定 `for x in pypto.loop(...)` / `pypto.loop_unroll(...)` 的 iter。"""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in ("loop", "loop_unroll"):
        if isinstance(f.value, ast.Name) and f.value.id in aliases:
            return True
    return False


def _is_range_iter(node: ast.AST) -> bool:
    """判定 `for x in range(...)` 的 iter。"""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name) and f.id == "range":
        return True
    return False


def _contains_pypto_call(node: ast.AST, aliases) -> bool:
    """node 子树内是否含 `pypto.<attr>(...)` 调用（compute / api / loop）。"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            v = sub.func.value
            if isinstance(v, ast.Name) and v.id in aliases:
                return True
    return False


def _local_func_callees(fn: ast.AST, func_defs: dict[str, ast.AST]) -> set[str]:
    callees = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in func_defs:
            callees.add(node.func.id)
    return callees


def _collect_jit_code_funcs(func_defs, jit_names, aliases, op):
    """返回 JIT 图代码函数名集合：从 @jit 函数出发调用图可达 ∪ 含 pypto 算子的
    函数，并排除 Layer K host wrapper（其循环由 OL45 管辖）。"""
    reachable: set[str] = set()
    work = list(jit_names)
    while work:
        name = work.pop()
        if name in reachable:
            continue
        reachable.add(name)
        fn = func_defs.get(name)
        if fn is None:
            continue
        work.extend(_local_func_callees(fn, func_defs) - reachable)
    # 含 pypto 算子的函数也视为 JIT 图代码（捕获调用图解析漏掉的 helper）
    for name, fn in func_defs.items():
        if _is_wrapper_function(name, op):
            continue
        if _contains_pypto_call(fn, aliases):
            reachable.add(name)
    # 排除 host wrapper
    return {n for n in reachable if not _is_wrapper_function(n, op)}


def _find_forbidden_loop(fn, aliases):
    """在 fn body 内查找非 pypto.loop / range 的 Python 循环 / 含 pypto 算子的推导式。
    返回 (kind, lineno) 或 None。"""
    for stmt in fn.body:
        for node in ast.walk(stmt):
            kind = _forbidden_loop_kind(node, aliases)
            if kind is not None:
                return kind, node.lineno
    return None


def _forbidden_loop_kind(node: ast.AST, aliases: set[str]) -> str | None:
    if isinstance(node, ast.While):
        return "while"
    if isinstance(node, (ast.For, ast.AsyncFor)):
        if not _is_pypto_loop_iter(node.iter, aliases) and not _is_range_iter(node.iter):
            return "for"
    comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    if isinstance(node, comprehensions) and _contains_pypto_call(node, aliases):
        return "comprehension"
    return None


@register("OL57")
def check_ol57(ctx: CheckContext) -> Finding:
    """@pypto.frontend.jit 配下的图代码（kernel 本体 + 其调用到的所有函数 /
    含 pypto 算子的函数）内只允许 `pypto.loop` / `pypto.loop_unroll` /
    `range(...)` 循环；其它 Python `for` / `while`（及含 pypto 算子的
    推导式）一律禁止。

    迭代可用 `pypto.loop(...)`（迭代间有依赖时加 `submit_before_loop=True`）
    或 `for ... in range(...)`（编译期全展开）。
    静态展开（如 inverse 类分块）不得用 Python while。Layer K host
    wrapper 的 kernel 驱动循环由 OL45 管辖, 不在本规则范围。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL57", "SKIP", "无 impl 文件可供检查")
    op = ctx.op_name
    saw_jit = False
    for impl_file in impl_files:
        syntax_error = _syntax_error_finding(ctx, "OL57", impl_file)
        if syntax_error:
            return syntax_error
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        aliases = ctx.pypto_aliases(impl_file)
        jit_funcs = _get_jit_functions(tree, aliases)
        if not jit_funcs:
            continue
        saw_jit = True
        func_defs = {
            n.name: n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        jit_names = {f.name for f in jit_funcs}
        code_funcs = _collect_jit_code_funcs(func_defs, jit_names, aliases, op)
        for name in sorted(code_funcs):
            fn = func_defs.get(name)
            if fn is None:
                continue
            hit = _find_forbidden_loop(fn, aliases)
            if hit:
                kind, lineno = hit
                return ctx.make_finding(
                    "OL57",
                    "FAIL",
                    f"{impl_file}: JIT 图代码函数 `{name}` 内出现非 pypto.loop/loop_unroll/range 的 "
                    f"Python {kind} 循环 (第 {lineno} 行)。@pypto.frontend.jit 配下"
                    f"(kernel 本体及其调用的所有函数) 的迭代可用 `pypto.loop(...)` / `pypto.loop_unroll(...)` "
                    f"或 `for ... in range(...)`; "
                    f"迭代间有数据依赖时加 `submit_before_loop=True`。静态展开 (含 "
                    f"inverse 类分块) 不得用 Python while。Layer K host wrapper "
                    f"的 kernel 驱动循环另由 OL45 管辖。",
                    file=impl_file,
                    line=lineno,
                )
    if not saw_jit:
        return ctx.make_finding("OL57", "SKIP", "无 jit 函数")
    return ctx.make_finding(
        "OL57", "PASS", "JIT 图代码内未发现非 pypto.loop/loop_unroll/range 的 Python 循环"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL58 — Layer K wrapper: output buffer must be torch.* pre-allocated before JIT call
# ─────────────────────────────────────────────────────────────────────────────


_PYPTO_CREATION_APIS = ("zeros", "empty", "ones", "full")
_TORCH_ALLOC_APIS = (
    "empty",
    "zeros",
    "ones",
    "full",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "empty_strided",
)


def _is_pypto_creation_call(node: ast.AST, aliases) -> ast.Call | None:
    """Return the call node if it is `pypto.zeros / empty / ones / full`, else None."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in _PYPTO_CREATION_APIS:
        if isinstance(f.value, ast.Name) and f.value.id in aliases:
            return node
    return None


def _is_torch_alloc_call(node: ast.AST) -> bool:
    """Detect `torch.empty / torch.zeros / torch.ones / torch.full` and *_like variants."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in _TORCH_ALLOC_APIS:
        if isinstance(f.value, ast.Name) and f.value.id == "torch":
            return True
    return False


def _collect_wrapper_assigns(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, ast.AST]:
    """name -> RHS for `name = expr` single-target assigns in wrapper body."""
    out: dict[str, ast.AST] = {}
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name):
                out[tgt.id] = stmt.value
    return out


def _jit_call_arg_nodes(call: ast.Call) -> list[ast.AST]:
    """Return positional + keyword arg expressions."""
    args = list(call.args)
    for kw in call.keywords:
        args.append(kw.value)
    return args


def _resolve_to_alloc_origin(
    name: str,
    local_assigns: dict[str, ast.AST],
    wrapper_params: set[str],
    aliases,
    max_depth: int = 6,
) -> tuple[str, ast.AST | None]:
    """Trace `name` back through wrapper-local assigns to determine its origin.

    Returns (origin_kind, evidence_node):
      - ("param", None)              name is a wrapper parameter
      - ("torch_alloc", rhs_call)    name was assigned from torch.* allocation API
      - ("pypto_creation", rhs_call) name was assigned from pypto.zeros/empty/ones/full
      - ("other", rhs_node)           name resolves to something else (method call, expr)
      - ("unknown", None)             name not assigned in wrapper body
    """
    if name in wrapper_params:
        return ("param", None)
    seen: set[str] = set()
    cur = name
    for _ in range(max_depth):
        if cur in seen:
            return ("unknown", None)
        seen.add(cur)
        rhs = local_assigns.get(cur)
        if rhs is None:
            return ("unknown", None)
        if _is_torch_alloc_call(rhs):
            return ("torch_alloc", rhs)
        if _is_pypto_creation_call(rhs, aliases) is not None:
            return ("pypto_creation", rhs)
        # Follow single-name alias chain (`b = a` → resolve a)
        if isinstance(rhs, ast.Name):
            cur = rhs.id
            if cur in wrapper_params:
                return ("param", None)
            continue
        return ("other", rhs)
    return ("unknown", None)


def _wrapper_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    params = {arg.arg for arg in func.args.args}
    params.update(arg.arg for arg in func.args.kwonlyargs)
    if func.args.vararg:
        params.add(func.args.vararg.arg)
    if func.args.kwarg:
        params.add(func.args.kwarg.arg)
    return params


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _jit_calls(
    wrapper: ast.FunctionDef | ast.AsyncFunctionDef,
    jit_names: set[str],
) -> Iterator[tuple[ast.Call, str]]:
    for node in ast.walk(wrapper):
        if not isinstance(node, ast.Call):
            continue
        callee = _call_name(node)
        if callee in jit_names:
            yield node, callee


def _looks_like_output(name: str) -> bool:
    return name in ("out", "output") or name.startswith("out_") or name.endswith("_out")


JitArgViolation = tuple[ast.Call, str, ast.Name, str, ast.AST | None]


def _jit_arg_violation(
    wrapper: ast.FunctionDef | ast.AsyncFunctionDef,
    jit_names: set[str],
    aliases: set[str],
) -> JitArgViolation | None:
    local_assigns = _collect_wrapper_assigns(wrapper)
    params = _wrapper_params(wrapper)
    for call, callee in _jit_calls(wrapper, jit_names):
        for arg in _jit_call_arg_nodes(call):
            if not isinstance(arg, ast.Name):
                continue
            origin, evidence = _resolve_to_alloc_origin(
                arg.id, local_assigns, params, aliases,
            )
            if origin == "pypto_creation":
                return call, callee, arg, origin, evidence
            if origin == "unknown" and _looks_like_output(arg.id):
                return call, callee, arg, origin, evidence
    return None


def _ol58_creation_finding(
    ctx: CheckContext,
    impl_file: str,
    wrapper: ast.FunctionDef | ast.AsyncFunctionDef,
    bad_call: ast.Call,
) -> Finding:
    api_name = bad_call.func.attr  # type: ignore[union-attr]
    return ctx.make_finding(
        "OL58",
        "FAIL",
        f"Layer K wrapper `{wrapper.name}` (第 {bad_call.lineno} 行) "
        f"调用 `pypto.{api_name}(...)`。`pypto.{api_name}` 是 JIT-context "
        f"creation API, 仅在 `@pypto.frontend.jit` 函数体内合法; "
        f"在 host wrapper 调用会 runtime crash "
        f"(`device=` kwarg 不接受, 或 `F21003 INVALID_TYPE`)。"
        f"host wrapper 内 output buffer 必须用 `torch.{api_name}(...)` "
        f"等 torch 等价物预先分配 (显式 `dtype=` 与 `device=`), "
        f"再传给 JIT 入口。",
        file=impl_file,
        line=bad_call.lineno,
    )


def _ol58_jit_arg_finding(
    ctx: CheckContext,
    impl_file: str,
    wrapper: ast.FunctionDef | ast.AsyncFunctionDef,
    violation: JitArgViolation,
) -> Finding:
    call, callee, arg, origin, evidence = violation
    if origin == "pypto_creation":
        evidence_line = evidence.lineno if evidence is not None else arg.lineno
        api_name = evidence.func.attr if evidence is not None else "?"  # type: ignore[union-attr]
        return ctx.make_finding(
            "OL58",
            "FAIL",
            f"Layer K wrapper `{wrapper.name}` 调用 JIT kernel `{callee}` "
            f"(第 {call.lineno} 行) 时传入 `{arg.id}`, 但 `{arg.id}` 来自 "
            f"`pypto.{api_name}(...)` (第 {evidence_line} 行)。"
            f"host wrapper 内 output buffer 必须用 torch.* 预分配 "
            f"(`torch.empty / torch.zeros / torch.empty_like` 等), "
            f"`pypto.{api_name}` 仅在 JIT 图内合法。",
            file=impl_file,
            line=evidence_line,
        )
    return ctx.make_finding(
        "OL58",
        "FAIL",
        f"Layer K wrapper `{wrapper.name}` 调用 JIT kernel `{callee}` "
        f"(第 {call.lineno} 行) 时传入 output `{arg.id}`, 但 `{arg.id}` "
        f"未在 wrapper 内分配, 也不是 wrapper 参数。"
        f"output buffer 必须用 `torch.empty / torch.zeros / "
        f"torch.empty_like` 等 torch allocation API 创建后, "
        f"再传给 JIT 入口。",
        file=impl_file,
        line=arg.lineno,
    )


@register("OL58")
def check_ol58(ctx: CheckContext) -> Finding:
    """Layer K host wrapper: output buffer must be torch.* pre-allocated before JIT call.

    Two checks:
    - (A) `pypto.zeros / pypto.empty / pypto.ones / pypto.full` are JIT-context creation
          APIs; calling them inside the Layer K wrapper body runtime-crashes
          (`device=` kwarg unsupported, or `F21003 INVALID_TYPE`). Forbid them.
    - (B) Every Name argument passed to a JIT-decorated kernel call inside the wrapper
          must resolve to (i) a wrapper parameter, or (ii) a `torch.empty / torch.zeros
          / torch.empty_like / ...` allocation. If it resolves to a `pypto.zeros/empty/
          ones/full` call (alias chain) it is flagged. Output buffers must be allocated
          via torch.* with explicit dtype= and device= before being passed to the JIT
          kernel.

    Other origins (method calls like `.reshape()`, `.contiguous()`, or complex
    expressions on wrapper inputs) are permitted — they are valid torch transforms
    of wrapper parameters and not output buffers being allocated.
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL58", "SKIP", "无 impl 文件可供检查")
    saw_wrapper = False
    for impl_file in impl_files:
        syntax_error = _syntax_error_finding(ctx, "OL58", impl_file)
        if syntax_error:
            return syntax_error
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        aliases = ctx.pypto_aliases(impl_file)
        jit_names = {func.name for func in _get_jit_functions(tree, aliases)}
        for top in _function_defs(tree):
            if not _is_wrapper_function(top.name, ctx.op_name):
                continue
            saw_wrapper = True
            bad_call = next(
                (
                    call for node in ast.walk(top)
                    if (call := _is_pypto_creation_call(node, aliases)) is not None
                ),
                None,
            )
            if bad_call is not None:
                return _ol58_creation_finding(ctx, impl_file, top, bad_call)
            if not jit_names:
                continue
            violation = _jit_arg_violation(top, jit_names, aliases)
            if violation is not None:
                return _ol58_jit_arg_finding(ctx, impl_file, top, violation)
    if not saw_wrapper:
        return ctx.make_finding("OL58", "SKIP", "未检测到 Layer K wrapper 函数")
    return ctx.make_finding(
        "OL58",
        "PASS",
        "Layer K wrapper output buffer 已用 torch.* 预分配; 未发现 pypto.zeros/empty/ones/full 误用",
    )
