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
from typing import Optional


def _resolve_pypto_aliases(tree: ast.Module) -> set[str]:
    """解析模块中 pypto 包的所有 import 别名。

    支持的模式:
    - ``import pypto``            → {"pypto"}
    - ``import pypto as pt``      → {"pt"}
    - ``import pypto.frontend``   → {"pypto"}  (取顶层名)
    - ``from pypto import frontend``  → 不影响包别名（frontend 不是 pypto 的别名）

    返回所有指代 pypto 顶层包的名称集合，始终至少包含 "pypto" 自身。
    """
    aliases: set[str] = {"pypto"}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Import):
            continue
        pypto_imports = (
            alias for alias in node.names
            if alias.name == "pypto" or alias.name.startswith("pypto.")
        )
        for alias in pypto_imports:
            # import pypto as pt → pt；import pypto → pypto
            resolved = alias.asname if alias.asname else alias.name.split(".")[0]
            aliases.add(resolved)
    return aliases


def _is_jit_decorator(dec: ast.AST, pypto_aliases: set[str]) -> bool:
    """精确判断装饰器是否为字面 ``@pypto.frontend.jit``（**严格模式，唯一正规形**）。

    唯一接受的写法:

    - ``@pypto.frontend.jit``
    - ``@pypto.frontend.jit(...)``     （带参数调用语法）

    **明确拒绝**的写法（这些会触发 OL01 [S0] FAIL）:

    - ``import pypto as pt`` 后 ``@pt.frontend.jit``
        → 顶层包别名也不接受，根名必须是字面 ``pypto``
    - ``import pypto.frontend as F`` 后 ``@F.jit``
        → 子模块别名不接受
    - ``from pypto import frontend`` 后 ``@frontend.jit``
        → 子模块直接绑定也不接受
    - ``from pypto.frontend import jit`` 后 ``@jit``
        → 函数级 from-import 不接受

    背景: 项目约定统一为字面三段属性 ``pypto.frontend.jit``——这是 AST
    静态分析最稳健的形式，也使 grep/IDE 跳转一致。OL01 强制此唯一形式。

    ``pypto_aliases`` 参数保留以维持函数签名稳定，但不再参与 JIT 装饰器
    判定（其他规则仍可使用别名集合，如 ``pypto.Tensor`` 注解）。
    """
    # 若装饰器是一个调用（如 @pypto.frontend.jit(...)），剥离到被调用对象
    target = dec
    if isinstance(target, ast.Call):
        target = target.func

    # 严格期望结构: Attribute(value=Attribute(value=Name(id='pypto'), attr='frontend'), attr='jit')
    if not isinstance(target, ast.Attribute) or target.attr != "jit":
        return False
    mid = target.value
    if not isinstance(mid, ast.Attribute) or mid.attr != "frontend":
        return False
    root = mid.value
    if not isinstance(root, ast.Name):
        return False
    return root.id == "pypto"  # 字面匹配，不接受任何别名


def _get_jit_functions(tree: ast.Module,
                       pypto_aliases: set[str] | None = None) -> list[ast.FunctionDef]:
    """找到所有被 @pypto.frontend.jit 装饰的函数（支持别名）"""
    if pypto_aliases is None:
        pypto_aliases = _resolve_pypto_aliases(tree)
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if _is_jit_decorator(dec, pypto_aliases):
                result.append(node)
                break
    return result


def _get_wrapper_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    wrappers: list[ast.FunctionDef] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_wrapper"):
            wrappers.append(node)
    return wrappers


def _resolve_primary_kernel_names(tree: ast.Module,
                                   pypto_aliases: set[str] | None = None) -> set[str]:
    """解析主 kernel 名称（优先取 wrapper 内直接调用的 jit 函数）。"""
    jit_funcs = _get_jit_functions(tree, pypto_aliases)
    if not jit_funcs:
        return set()
    jit_names = {f.name for f in jit_funcs}

    resolved: set[str] = set()
    for wrapper in _get_wrapper_functions(tree):
        for node in ast.walk(wrapper):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in jit_names:
                resolved.add(node.func.id)
    if resolved:
        return resolved

    # 无法解析 wrapper 调用关系时，降级为"全部 jit"。
    return jit_names


def _get_primary_jit_functions(tree: ast.Module,
                                pypto_aliases: set[str] | None = None) -> list[ast.FunctionDef]:
    primary_names = _resolve_primary_kernel_names(tree, pypto_aliases)
    if not primary_names:
        return []
    return [f for f in _get_jit_functions(tree, pypto_aliases) if f.name in primary_names]


def _has_loop_structure(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return True
        if isinstance(node, ast.Call):
            call_str = ast.dump(node.func).lower()
            if "loop" in call_str:
                return True
            if any(keyword.arg in ("unroll_list", "submit_before_loop")
                   for keyword in node.keywords if keyword.arg):
                return True
    return False


def _is_pypto_tensor_annotation(annotation: ast.AST,
                                pypto_aliases: set[str] | None = None) -> bool:
    """检查注解是否为 pypto.Tensor 类型（支持别名）"""
    if pypto_aliases is None:
        pypto_aliases = {"pypto"}
    func: ast.AST
    if isinstance(annotation, ast.Call):
        func = annotation.func
    else:
        func = annotation

    if isinstance(func, ast.Attribute):
        return isinstance(func.value, ast.Name) and func.value.id in pypto_aliases and func.attr == "Tensor"
    if isinstance(func, ast.Name):
        return func.id == "Tensor"
    return False


def _is_non_tensor_annotation(annotation: ast.AST,
                              pypto_aliases: set[str] | None = None) -> bool:
    """检查注解是否为显式声明的非 Tensor 参数类型。"""
    return annotation is not None and not _is_pypto_tensor_annotation(annotation, pypto_aliases)


def _has_test_level_markers(tree: ast.Module, source: str) -> tuple[bool, bool]:
    """识别仓内常见的两级测试命名方式。

    支持：
    - 函数名中的 level0 / level1
    - 函数名中的 _l0 / _l1（test_template.py.tmpl 推荐的命名约定）
    - 功能_P0 / 性能_P0（含 func_p0 / perf_p0）这类 case 命名或元数据
    """
    func_names = [
        node.name.lower() for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    has_level0 = any("level0" in name or "_l0" in name for name in func_names)
    has_level1 = any("level1" in name or "_l1" in name for name in func_names)
    if has_level0 and has_level1:
        return True, True

    lower = source.lower()
    p0_level0_patterns = (
        "功能_p0",
        "func_p0",
        "test_p0",
    )
    p0_level1_patterns = (
        "性能_p0",
        "perf_p0",
    )
    has_level0 = has_level0 or any(pattern in lower for pattern in p0_level0_patterns)
    has_level1 = has_level1 or any(pattern in lower for pattern in p0_level1_patterns)
    return has_level0, has_level1


FP32_ONLY_OPS = {"sigmoid", "softmax", "sin", "cos"}


def _is_fp32_only_call(node: ast.AST,
                       pypto_aliases: set[str] | None = None) -> bool:
    if pypto_aliases is None:
        pypto_aliases = {"pypto"}
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    owner = node.func.value
    if not isinstance(owner, ast.Name):
        return False
    if owner.id not in pypto_aliases:
        return False
    return node.func.attr in FP32_ONLY_OPS


def _extract_symbolic_dynamic_aliases(tree: ast.Module,
                                      pypto_aliases: set[str] | None = None) -> set[str]:
    """提取模块级别指向 pypto.DYNAMIC/pypto.DYN 的符号名（支持别名）。"""
    if pypto_aliases is None:
        pypto_aliases = _resolve_pypto_aliases(tree)
    aliases: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        value = node.value
        if isinstance(value, ast.Attribute):
            if (isinstance(value.value, ast.Name)
                and value.value.id in pypto_aliases
                and value.attr in ("DYNAMIC", "DYN")):
                aliases.add(target)
        elif isinstance(value, ast.Name) and value.id in ("DYNAMIC", "DYN"):
            aliases.add(target)
    return aliases


def _shape_has_dynamic(shape_node: ast.AST, dynamic_aliases: set[str],
                       pypto_aliases: set[str] | None = None) -> bool:
    """递归判断 shape 注解中是否包含动态维度声明（支持别名）。"""
    if pypto_aliases is None:
        pypto_aliases = {"pypto"}
    if isinstance(shape_node, ast.Attribute):
        owner = shape_node.value
        if isinstance(owner, ast.Name) and owner.id in pypto_aliases and shape_node.attr in ("DYNAMIC", "DYN"):
            return True
    if isinstance(shape_node, ast.Name):
        return shape_node.id in dynamic_aliases or shape_node.id in ("DYNAMIC", "DYN")
    if isinstance(shape_node, (ast.List, ast.Tuple)):
        return any(_shape_has_dynamic(elem, dynamic_aliases, pypto_aliases) for elem in shape_node.elts)
    for child in ast.iter_child_nodes(shape_node):
        if _shape_has_dynamic(child, dynamic_aliases, pypto_aliases):
            return True
    return False


def _dynamic_tensor_annotation_status(
    jit_funcs: list[ast.FunctionDef],
    pypto_aliases: set[str],
    dynamic_aliases: set[str],
) -> tuple[int, bool]:
    """Return the Tensor-annotation count and whether one has a dynamic shape."""
    tensor_count = 0
    has_dynamic = False
    for func in jit_funcs:
        for arg in func.args.args:
            annotation = arg.annotation
            if not isinstance(annotation, ast.Call):
                continue
            if not _is_pypto_tensor_annotation(annotation, pypto_aliases):
                continue
            tensor_count += 1
            if annotation.args and _shape_has_dynamic(
                annotation.args[0], dynamic_aliases, pypto_aliases,
            ):
                has_dynamic = True
    return tensor_count, has_dynamic


def _get_func_param_count(tree: ast.Module, func_name: str) -> Optional[int]:
    """获取指定函数的必需参数个数"""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            total = len(node.args.args)
            defaults = len(node.args.defaults)
            required = total - defaults
            if required > 0 and node.args.args[0].arg in ("self", "cls"):
                required -= 1
            return required
    return None


def _const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _first_two_ints(nodes: list[ast.AST]) -> tuple[int, int] | None:
    if len(nodes) < 2:
        return None
    first = _const_int(nodes[0])
    second = _const_int(nodes[1])
    if first is None or second is None:
        return None
    return first, second


def _run_check_shape(node: ast.Call) -> tuple[int, int] | None:
    if not isinstance(node.func, ast.Name) or node.func.id != "_run_and_check":
        return None
    return _first_two_ints(node.args[1:3])


def _torch_factory_shape(node: ast.Call) -> tuple[int, int] | None:
    if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
        return None
    if node.func.value.id != "torch":
        return None
    if node.func.attr not in ("randn", "zeros", "ones", "empty", "full"):
        return None
    if node.func.attr == "randn":
        direct_shape = _first_two_ints(node.args[:2])
        if direct_shape is not None:
            return direct_shape
    if not node.args or not isinstance(node.args[0], (ast.Tuple, ast.List)):
        return None
    return _first_two_ints(node.args[0].elts[:2])


def _extract_shapes_from_test_ast(tree: ast.Module) -> set[tuple[int, ...]]:
    """从测试 AST 中提取 shape 覆盖。

    支持模式：
    1) _run_and_check(name, n, m, ...)
    2) torch.randn(n, m, ...) / torch.zeros((n, m), ...) 等
    """
    shapes: set[tuple[int, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for shape in (_run_check_shape(node), _torch_factory_shape(node)):
            if shape is not None:
                shapes.add(shape)
    return shapes


def _extract_jit_assigned_names(tree: ast.Module,
                                pypto_aliases: set[str] | None = None) -> set[str]:
    names: set[str] = set()
    for func in _get_jit_functions(tree, pypto_aliases):
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names
