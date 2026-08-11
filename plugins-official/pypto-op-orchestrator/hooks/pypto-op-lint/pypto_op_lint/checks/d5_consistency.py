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
from collections.abc import Iterator

try:
    import yaml
except ImportError:  # PyYAML 是可选依赖
    yaml = None

from ..ast_helpers import (
    _dynamic_tensor_annotation_status,
    _extract_jit_assigned_names,
    _extract_shapes_from_test_ast,
    _extract_symbolic_dynamic_aliases,
    _get_func_param_count,
    _get_jit_functions,
    _get_primary_jit_functions,
    _has_loop_structure,
)
from ..core import (
    API_REPORT_FILE,
    DESIGN_FILE,
    SPEC_FILE,
    STRICT_ENV,
    CheckContext,
    Finding,
    register,
)
from ..utils import (
    _extract_design_identifiers,
    _extract_shapes_from_text,
    _extract_spec_dtypes_from_meta,
    _extract_test_dtypes,
    _extract_tolerance,
    _impl_files_to_scan,
    _load_doc_meta,
    _load_tolerance_schema,
    _parse_front_matter,
    _syntax_error_finding,
    _validate_doc_schema,
)

ParsedImpl = tuple[str, ast.Module, set[str]]


def _iter_parsed_impls(ctx: CheckContext, impl_files: list[str]) -> Iterator[ParsedImpl]:
    for impl_file in impl_files:
        tree = ctx.parse_file(impl_file)
        if tree is not None:
            yield impl_file, tree, ctx.pypto_aliases(impl_file)


def _jit_has_dynamic_shape(
    jit_funcs: list[ast.FunctionDef],
    aliases: set[str],
    dynamic_aliases: set[str],
) -> bool:
    return _dynamic_tensor_annotation_status(
        jit_funcs, aliases, dynamic_aliases,
    )[1]


def _ol31_syntax_error(ctx: CheckContext, impl_files: list[str]) -> Finding | None:
    for impl_file in impl_files:
        finding = _syntax_error_finding(ctx, "OL31", impl_file)
        if finding is not None:
            return finding
    return None


@register("OL30")
def check_ol30(ctx: CheckContext) -> Finding:
    """SPEC.md front matter 的 supported_dtypes 必须在测试文件中覆盖。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL30", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL30", "SKIP", f"{test_file} 不存在")

    spec_dtypes = _extract_spec_dtypes_from_meta(spec_meta)
    if not spec_dtypes:
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} front matter 中未声明 supported_dtypes",
            file=SPEC_FILE)

    test_dtypes = _extract_test_dtypes(test_source)
    missing = spec_dtypes - test_dtypes
    if missing:
        canonical_names = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        missing_names = sorted(canonical_names.get(d, d) for d in missing)
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} 声明支持的 dtype ({', '.join(missing_names)}) "
            "在测试文件中未覆盖；请补充对应 dtype 的测试用例",
            file=test_file)
    return ctx.make_finding("OL30", "PASS", "spec dtype 覆盖与 test 一致")


@register("OL31")
def check_ol31(ctx: CheckContext) -> Finding:
    """DESIGN.md front matter dynamic_axes 与 impl 动态注解一致。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。每个文件
    单独评估，任一文件均未使用 pypto.DYNAMIC literal 时返回 FAIL。
    这样在模块开发阶段（Stage 5 Phase M_k）即可发现违规，避免下沉
    到 Stage 6 集成阶段才暴露。
    """
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL31", "SKIP", f"{DESIGN_FILE} 不存在")
    design_meta = _load_doc_meta(ctx, DESIGN_FILE)
    schema_errors = _validate_doc_schema("DESIGN", design_meta)
    if schema_errors:
        return ctx.make_finding("OL31", "FAIL",
            f"{DESIGN_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=DESIGN_FILE)

    dynamic_axes = design_meta.get("dynamic_axes", [])
    if not isinstance(dynamic_axes, list) or not dynamic_axes:
        return ctx.make_finding("OL31", "PASS",
            f"{DESIGN_FILE} front matter 未声明动态轴，无需检查")

    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL31", "SKIP", "无 impl 文件可供检查")
    syntax_error = _ol31_syntax_error(ctx, impl_files)
    if syntax_error is not None:
        return syntax_error

    files_without_dynamic: list[str] = []
    saw_jit = False
    last_file_seen = ""
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        jit_funcs = _get_primary_jit_functions(tree, aliases)
        if not jit_funcs:
            continue
        saw_jit = True
        last_file_seen = impl_file
        dynamic_aliases = _extract_symbolic_dynamic_aliases(tree, aliases)
        if not _jit_has_dynamic_shape(jit_funcs, aliases, dynamic_aliases):
            files_without_dynamic.append(impl_file)

    if not saw_jit:
        return ctx.make_finding("OL31", "SKIP", "无 jit 函数")
    if files_without_dynamic:
        return ctx.make_finding("OL31", "FAIL",
            f"{DESIGN_FILE} front matter 声明了动态轴，但以下 impl 文件的 "
            f"Tensor 注解中均未使用 pypto.DYNAMIC/pypto.DYN: "
            f"{', '.join(files_without_dynamic)}。"
            "动态轴必须在类型注解中显式标记。",
            file=files_without_dynamic[0])
    return ctx.make_finding("OL31", "PASS",
        "design 动态轴声明与 impl 注解一致", file=last_file_seen)


@register("OL43")
def check_ol43(ctx: CheckContext) -> Finding:
    """DESIGN 声明动态轴时 impl 必须包含 pypto.loop 调用。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    DESIGN 一旦声明动态轴，每个 module impl 都必须自带 pypto.loop——这是
    "production kernel 4 要素" 的核心一环。module 漏写 loop 会在 Stage 5
    集成 / Stage 6 production shape 上 workspace estimator INT32 溢出。
    """
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL43", "SKIP", f"{DESIGN_FILE} 不存在")

    design_meta = _load_doc_meta(ctx, DESIGN_FILE)
    dynamic_axes = design_meta.get("dynamic_axes") or design_meta.get("dynamic_axis")

    # 若 front matter 无声明，在正文中搜索动态轴相关关键词
    if not dynamic_axes:
        content = ctx.read_file(DESIGN_FILE)
        has_dynamic_keyword = bool(
            re.search(r'(?:pypto\.DYNAMIC|pypto\.DYN|动态轴|dynamic.{0,10}axis)', content, re.IGNORECASE)
        )
        if not has_dynamic_keyword:
            return ctx.make_finding("OL43", "SKIP",
                "DESIGN.md 未声明动态轴，无需检查 pypto.loop")

    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL43", "SKIP", "无 impl 文件可供检查")

    saw_jit = False
    files_without_loop: list[str] = []
    last_pass_file = None
    for impl_file, _, _, jit_funcs in _iter_jit_impls_d5(ctx, impl_files):
        saw_jit = True
        impl_source = ctx.read_file(impl_file) or ""
        has_pypto_loop = bool(re.search(r'pypto\.loop\s*\(', impl_source))
        if not has_pypto_loop:
            has_pypto_loop = _has_loop_structure(jit_funcs[0])
        if has_pypto_loop:
            last_pass_file = impl_file
        else:
            files_without_loop.append(impl_file)

    if not saw_jit:
        return ctx.make_finding("OL43", "SKIP", "未找到 JIT 函数")

    if files_without_loop:
        return ctx.make_finding(
            "OL43",
            "FAIL",
            "DESIGN.md 声明了动态轴，但以下 impl 文件中未找到 "
            "pypto.loop / pypto.lang.loop 调用：" + ", ".join(files_without_loop) +
            "。OL43 是硬性门禁；NPU 运行通过不能替代动态轴 loop。"
            "请在对应 JIT kernel 中补齐遍历动态轴的真实 pypto.loop，"
            "并结合 NPU 错误码与 traceback 继续修复。",
            file=files_without_loop[0],
        )
    return ctx.make_finding(
        "OL43",
        "PASS",
        "DESIGN 声明动态轴，所有 impl 文件均含 loop 结构",
        file=last_pass_file,
    )


def _iter_jit_impls_d5(
    ctx: CheckContext,
    impl_files: list[str],
) -> Iterator[tuple[str, ast.Module, set[str], list[ast.FunctionDef]]]:
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        jit_funcs = _get_jit_functions(tree, aliases)
        if jit_funcs:
            yield impl_file, tree, aliases, jit_funcs


def _matched_tolerance_keys(
    tolerance_meta: dict,
) -> tuple[list[str], str | None]:
    schemas = _load_tolerance_schema()
    if not schemas:
        return ["atol", "rtol"], None
    for schema in schemas:
        required = schema.get("required", [])
        if all(key in tolerance_meta for key in required):
            return required, None
    modes = [
        f"{schema.get('mode', '?')}({', '.join(schema.get('required', []))})"
        for schema in schemas
    ]
    return [], " | ".join(modes)


def _tolerance_mismatches(
    tolerance_meta: dict,
    test_source: str,
    matched_keys: list[str],
) -> list[str]:
    mismatches: list[str] = []
    for key in matched_keys:
        if key not in tolerance_meta:
            continue
        try:
            spec_value = float(tolerance_meta[key])
        except (ValueError, TypeError):
            continue
        test_values = _extract_tolerance(test_source, key)
        if not test_values:
            continue
        test_loosest = max(test_values)
        if test_loosest > spec_value * 3:
            mismatches.append(
                f"{key}: spec 最严={spec_value}, test 最松={test_loosest}"
            )
    return mismatches


@register("OL32")
def check_ol32(ctx: CheckContext) -> Finding:
    """SPEC.md front matter tolerance 须与 test 文件一致（支持 tolerance_schema 多模式）。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL32", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL32", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL32", "SKIP", f"{test_file} 不存在")

    tolerance_meta = spec_meta.get("tolerance", {})
    if not isinstance(tolerance_meta, dict):
        return ctx.make_finding("OL32", "FAIL",
            f"{SPEC_FILE} front matter 中 tolerance 必须是 dict",
            file=SPEC_FILE)

    matched_keys, modes = _matched_tolerance_keys(tolerance_meta)
    if modes is not None:
        return ctx.make_finding(
            "OL32", "FAIL",
            f"{SPEC_FILE} tolerance 不匹配任何已知模式: {modes}",
            file=SPEC_FILE,
        )
    mismatches = _tolerance_mismatches(tolerance_meta, test_source, matched_keys)
    if mismatches:
        return ctx.make_finding("OL32", "WARN",
            f"{SPEC_FILE} 与 test 的精度容差差距较大: {'; '.join(mismatches)}",
            file=test_file)
    return ctx.make_finding("OL32", "PASS", "spec 精度容差与 test 一致")


@register("OL33")
def check_ol33(ctx: CheckContext) -> Finding:
    """golden 函数签名须与 impl wrapper 函数兼容"""
    golden_file = f"{ctx.op_name}_golden.py"
    impl_file = f"{ctx.op_name}_impl.py"
    syntax_error = _syntax_error_finding(ctx, "OL33", impl_file)
    if syntax_error:
        return syntax_error
    golden_tree = ctx.parse_file(golden_file)
    impl_tree = ctx.parse_file(impl_file)
    if golden_tree is None:
        return ctx.make_finding("OL33", "SKIP", f"{golden_file} 不存在或无法解析")
    if impl_tree is None:
        return ctx.make_finding("OL33", "SKIP", f"{impl_file} 不存在或无法解析")
    golden_func = f"{ctx.op_name}_golden"
    wrapper_func = f"{ctx.op_name}_wrapper"
    golden_count = _get_func_param_count(golden_tree, golden_func)
    wrapper_count = _get_func_param_count(impl_tree, wrapper_func)
    if golden_count is None:
        return ctx.make_finding("OL33", "SKIP", f"未找到 {golden_func} 函数")
    if wrapper_count is None:
        return ctx.make_finding("OL33", "SKIP", f"未找到 {wrapper_func} 函数")
    if golden_count != wrapper_count:
        return ctx.make_finding("OL33", "WARN",
            f"{golden_func} 需要 {golden_count} 个必需参数，"
            f"但 {wrapper_func} 需要 {wrapper_count} 个必需参数，接口可能不兼容",
            file=impl_file)
    return ctx.make_finding("OL33", "PASS",
        f"golden ({golden_count} 参数) 与 wrapper ({wrapper_count} 参数) 签名兼容")


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(item) for item in value)
    except (ValueError, TypeError):
        return None


def _spec_shape_item(item: object) -> set[tuple[int, ...]]:
    if isinstance(item, dict):
        candidates = item.values()
    elif isinstance(item, (list, tuple)):
        direct = _shape_tuple(item)
        if direct is not None:
            return {direct} if len(direct) >= 2 else set()
        candidates = item
    else:
        return set()
    shapes = {_shape_tuple(candidate) for candidate in candidates}
    return {shape for shape in shapes if shape is not None and len(shape) >= 2}


def _spec_p0_shapes(raw_shapes: list) -> set[tuple[int, ...]]:
    shapes: set[tuple[int, ...]] = set()
    for item in raw_shapes:
        shapes.update(_spec_shape_item(item))
    return shapes


@register("OL34")
def check_ol34(ctx: CheckContext) -> Finding:
    """SPEC.md front matter p0_shapes 应在 test 文件中覆盖。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL34", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL34", "SKIP", f"{test_file} 不存在")
    test_tree = ctx.parse_file(test_file)
    if test_tree is None:
        return ctx.make_finding("OL34", "SKIP", f"{test_file} 不存在或无法解析")
    raw_shapes = spec_meta.get("p0_shapes", [])
    if not isinstance(raw_shapes, list):
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter 中 p0_shapes 必须是 list",
            file=SPEC_FILE)
    spec_p0_shapes = _spec_p0_shapes(raw_shapes)

    if not spec_p0_shapes:
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter 中未找到有效 p0_shapes。\n"
            "期望格式: [[1024, 128], [1024, 256]] 或 [{x: [4, 2560], y: [4, 1024]}]\n"
            "注意: {B: 4, C: 64} 格式不支持（value 必须是 list，不能是标量）",
            file=SPEC_FILE)
    test_shapes = _extract_shapes_from_test_ast(test_tree) | _extract_shapes_from_text(test_source)
    missing = spec_p0_shapes - test_shapes
    if missing:
        missing_str = ", ".join(str(list(s)) for s in sorted(missing))
        return ctx.make_finding("OL34", "WARN",
            f"{SPEC_FILE} 中 P0 配置的 shape {missing_str} 在测试文件中未覆盖",
            file=test_file)
    return ctx.make_finding("OL34", "PASS",
        "spec P0 配置 shape 在 test 中均有覆盖")


@register("OL39")
def check_ol39(ctx: CheckContext) -> Finding:
    """strict 模式下，三个文档必须包含 front matter。"""
    if os.environ.get(STRICT_ENV, "1") != "1":
        return ctx.make_finding("OL39", "SKIP", "strict 模式关闭")
    for filename in (SPEC_FILE, DESIGN_FILE, API_REPORT_FILE):
        content = ctx.read_file(filename)
        if not content:
            return ctx.make_finding("OL39", "FAIL", f"{filename} 不存在", file=filename)
        meta, _ = _parse_front_matter(content)
        if not meta:
            return ctx.make_finding("OL39", "FAIL",
                f"{filename} 缺少 front matter（必须以 --- 开头）",
                file=filename)
    return ctx.make_finding("OL39", "PASS", "front matter 完整")


@register("OL40")
def check_ol40(ctx: CheckContext) -> Finding:
    """strict 模式下，三个文档 front matter 必填字段必须完整。"""
    if os.environ.get(STRICT_ENV, "1") != "1":
        return ctx.make_finding("OL40", "SKIP", "strict 模式关闭")
    docs = ((SPEC_FILE, "SPEC"), (DESIGN_FILE, "DESIGN"), (API_REPORT_FILE, "API_REPORT"))
    for filename, doc_type in docs:
        content = ctx.read_file(filename)
        if not content:
            return ctx.make_finding("OL40", "FAIL", f"{filename} 不存在", file=filename)
        meta, _ = _parse_front_matter(content)
        errors = _validate_doc_schema(doc_type, meta)
        if errors:
            return ctx.make_finding("OL40", "FAIL",
                f"{filename} front matter schema 非法: {'; '.join(errors)}",
                file=filename)
    return ctx.make_finding("OL40", "PASS", "front matter schema 完整")


@register("OL41")
def check_ol41(ctx: CheckContext) -> Finding:
    """禁止将 lint/门禁输出文本污染到代码工件。"""
    suspicious_tokens = (
        "[pypto-op-lint]",
        "交付门禁阻断",
        "以下规则违规",
        "fix_hints:",
        "blocking_rules:",
        "docs_ref:",
    )
    targets = [
        f"{ctx.op_name}_impl.py",
        f"{ctx.op_name}_golden.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    modules_dir = os.path.join(ctx.op_dir, "modules")
    if os.path.isdir(modules_dir):
        for entry in sorted(os.listdir(modules_dir)):
            is_module_file = entry.endswith(("_impl.py", "_golden.py"))
            is_test_file = entry.startswith("test_") and entry.endswith(".py")
            if is_module_file or is_test_file:
                targets.append(os.path.join("modules", entry))
    for filename in targets:
        source = ctx.read_file(filename)
        if not source:
            continue
        lower = source.lower()
        for token in suspicious_tokens:
            if token.lower() in lower:
                return ctx.make_finding(
                    "OL41",
                    "FAIL",
                    f"{filename} 检测到 lint 输出污染片段: {token}",
                    file=filename,
                )
    return ctx.make_finding("OL41", "PASS", "未检测到 lint 输出污染")


@register("OL37")
def check_ol37(ctx: CheckContext) -> Finding:
    """design 与 impl 的关键命名可追溯性检查（信息提示）。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    各 impl 文件中的 JIT 内局部变量名取并集后再与 DESIGN 关键命名求交集。
    在 Stage 5 仅有 module 文件的状态下也能给出可追溯性反馈。
    """
    design_content = ctx.read_file(DESIGN_FILE)
    if not design_content:
        return ctx.make_finding("OL37", "SKIP", f"{DESIGN_FILE} 不存在")

    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL37", "SKIP", "无 impl 文件可供检查")

    design_names = _extract_design_identifiers(design_content)
    impl_blacklist = {
        "i", "j", "k", "n", "m", "x", "y", "z", "tmp", "temp", "result", "out",
        "input", "output",
    }
    impl_names: set[str] = set()
    last_impl_file = None
    for impl_file in impl_files:
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        last_impl_file = impl_file
        aliases = ctx.pypto_aliases(impl_file)
        names = _extract_jit_assigned_names(tree, aliases)
        impl_names |= {n for n in names if n not in impl_blacklist}

    if last_impl_file is None:
        return ctx.make_finding("OL37", "SKIP", "无 impl 文件可解析")

    # design 中缺少可对齐变量名时直接跳过，避免误报
    if len(design_names) < 3:
        return ctx.make_finding(
            "OL37",
            "SKIP",
            f"{DESIGN_FILE} 中可用于对齐的代码变量名不足（<3），跳过可追溯性检查",
        )

    if not impl_names:
        return ctx.make_finding("OL37", "SKIP", "未检测到 jit 内局部变量赋值")

    overlap = sorted(design_names & impl_names)
    # 命中 >=2 视为具备可追溯性；该规则为 S3 信息提示，避免因 design 关键名
    # 抽取范围较大导致"有重合却仍提示不重合"的假阳性。
    if len(overlap) >= 2:
        return ctx.make_finding(
            "OL37",
            "PASS",
            f"design/impl 命名可追溯性良好（命中 {len(overlap)} 个：{', '.join(overlap[:5])}）",
            file=last_impl_file,
        )

    sample_impl = ", ".join(sorted(list(impl_names))[:5])
    return ctx.make_finding(
        "OL37",
        "INFO",
        "design 与 impl 的关键命名重合较少，建议对齐中间变量命名以提升可追溯性；"
        f"当前 impl 示例变量：{sample_impl}",
        file=last_impl_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL50/OL51 — module_interfaces.yaml 契约与 impl 的一致性
# ─────────────────────────────────────────────────────────────────────────────

def _load_module_interfaces(ctx: CheckContext) -> dict | None:
    """Load `<op_dir>/eval/module_interfaces.yaml`. Return None if absent / invalid."""
    yaml_path = os.path.join(ctx.op_dir, "eval", "module_interfaces.yaml")
    if not os.path.isfile(yaml_path) or yaml is None:
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _suffix_to_module_ids(suffix: str) -> list[int]:
    """Convert cumulative suffix string ('1', '12', '123', ..., '12345678910')
    to list of module ids. Handles two-digit ids by interpreting the suffix as
    decimal concatenation: '12' = [1, 2], '12345678910' = [1..10].
    """
    out: list[int] = []
    i = 0
    n = 1
    while i < len(suffix):
        # Greedy: assume single digit unless that would skip beyond range
        # Simple model: digits 1..9 are single-char, 10+ uses two chars.
        # Heuristic: try single digit first; ids are emitted in order so the
        # next id should always equal previous+1.
        if i + 1 < len(suffix) and suffix[i:i + 2].isdigit():
            two = int(suffix[i:i + 2])
            if two == n:
                out.append(two)
                i += 2
                n += 1
                continue
        if suffix[i].isdigit():
            one = int(suffix[i])
            if one == n:
                out.append(one)
                i += 1
                n += 1
                continue
        # Inconsistent — give up
        return []
    return out


def _impl_file_to_module_suffix(impl_file: str, op_name: str) -> str | None:
    """Extract the cumulative suffix from a module impl filename.

    `modules/<op>_module<suffix>_impl.py` → '<suffix>'.
    Top-level `<op>_impl.py` → None (covers ALL modules).
    """
    basename = os.path.basename(impl_file)
    m = re.match(rf"^{re.escape(op_name)}_module(\d+)_impl\.py$", basename)
    if m:
        return m.group(1)
    return None


def _primary_inputs_for_modules(
    interfaces: dict, module_ids: list[int]
) -> list[str]:
    """Return the ordered list of primary-input names referenced by the given
    module ids. Order is determined by the `primary_inputs` declaration order.
    """
    primary_decl = interfaces.get("primary_inputs", [])
    if not isinstance(primary_decl, list):
        return []
    declared_order = [
        p.get("name")
        for p in primary_decl
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    ]
    modules = interfaces.get("modules", [])
    if not isinstance(modules, list):
        return []
    referenced: set[str] = set()
    id_filter = set(module_ids) if module_ids else None
    for mod in modules:
        if not isinstance(mod, dict):
            continue
        mid = mod.get("id")
        if id_filter is not None and mid not in id_filter:
            continue
        for inp in mod.get("inputs", []) or []:
            if not isinstance(inp, dict):
                continue
            if inp.get("source") == "primary" and isinstance(inp.get("name"), str):
                referenced.add(inp["name"])
    return [n for n in declared_order if n in referenced]


def _outputs_for_modules(
    interfaces: dict, module_ids: list[int]
) -> list[str]:
    """Return the names of outputs declared by the given module ids.
    For cumulative modules, the final output(s) of the last module
    are the externally-visible outputs (others are intermediate).
    """
    modules = interfaces.get("modules", [])
    if not isinstance(modules, list) or not module_ids:
        return []
    final_id = max(module_ids)
    for mod in modules:
        if isinstance(mod, dict) and mod.get("id") == final_id:
            outs = mod.get("outputs", []) or []
            return [
                out_entry["name"]
                for out_entry in outs
                if isinstance(out_entry, dict) and isinstance(out_entry.get("name"), str)
            ]
    return []


def _final_outputs_for_top_level(interfaces: dict) -> list[str]:
    """For top-level <op>_impl.py: use `final_outputs` (which lists every
    return tensor of the user golden) to enumerate the externally-visible
    output names."""
    fos = interfaces.get("final_outputs", [])
    if not isinstance(fos, list):
        return []
    return [
        fo["name"] for fo in fos
        if isinstance(fo, dict) and isinstance(fo.get("name"), str)
    ]


def _find_wrapper_function(
    tree: ast.Module, op_name: str, suffix: str | None
) -> ast.FunctionDef | None:
    """Find the Layer K wrapper function in `tree`.

    For module impl: prefer `<op>_module<suffix>_wrapper`.
    For top-level impl: prefer `host_wrapper`, then `<op>_wrapper`,
    then any function named `launch_*` / `run_*`.
    """
    candidates: list[str] = []
    if suffix is not None:
        candidates.append(f"{op_name}_module{suffix}_wrapper")
    else:
        candidates.extend(["host_wrapper", f"{op_name}_wrapper"])

    def_by_name: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            def_by_name[node.name] = node  # type: ignore[assignment]
    for cand in candidates:
        if cand in def_by_name:
            return def_by_name[cand]
    # Fallback: launch_* / run_* only for top-level
    if suffix is None:
        for name, fn in def_by_name.items():
            if name.startswith("launch_") or name.startswith("run_"):
                return fn
    return None


def _func_arg_names(func: ast.FunctionDef) -> list[str]:
    """Positional + keyword-only arg names (skip *args / **kwargs)."""
    args = func.args
    names: list[str] = []
    for a in list(args.posonlyargs) + list(args.args):
        names.append(a.arg)
    for a in args.kwonlyargs:
        names.append(a.arg)
    return names


def _top_level_primary_inputs(interfaces: dict) -> list[str] | None:
    primary_decl = interfaces.get("primary_inputs", [])
    if not isinstance(primary_decl, list):
        return None
    return [
        item["name"]
        for item in primary_decl
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def _invalid_suffix_finding(
    ctx: CheckContext,
    impl_file: str,
    suffix: str,
) -> Finding:
    cumulative_example = (
        "".join(str(index) for index in range(1, int(suffix) + 1))
        if suffix.isdigit() else "123..."
    )
    return ctx.make_finding(
        "OL50", "FAIL",
        f"{impl_file}: 模块 suffix '{suffix}' 不满足累积命名规约。\n"
        f"累积命名规则: 文件名 suffix 必须是从 1 开始的连续模块 ID 拼接。\n"
        f"  合法示例: module1(M1), module12(M1+M2), module123(M1+M2+M3)\n"
        f"  当前 suffix '{suffix}' 无法解析为累积序列。\n"
        f"修正方式:\n"
        f"  - 如果这是第 {suffix} 个 Phase 的累积产物（M1–M{suffix}），"
        f"文件名应为 module{cumulative_example}_impl.py\n"
        f"  - 如果这是 standalone 模块（不参与累积），应将其逻辑合并到 "
        f"顶层 {ctx.op_name}_impl.py 中，或重新设计模块分解使其参与累积 Phase 链",
        file=impl_file,
    )


def _expected_primary_inputs(
    ctx: CheckContext,
    interfaces: dict,
    impl_file: str,
) -> tuple[list[str] | None, str | None, Finding | None]:
    suffix = _impl_file_to_module_suffix(impl_file, ctx.op_name)
    if suffix is None:
        return _top_level_primary_inputs(interfaces), suffix, None
    module_ids = _suffix_to_module_ids(suffix)
    if not module_ids:
        return None, suffix, _invalid_suffix_finding(ctx, impl_file, suffix)
    return _primary_inputs_for_modules(interfaces, module_ids), suffix, None


def _wrapper_contract_finding(
    ctx: CheckContext,
    interfaces: dict,
    impl_file: str,
    tree: ast.Module,
) -> Finding | None:
    expected, suffix, invalid_suffix = _expected_primary_inputs(ctx, interfaces, impl_file)
    if invalid_suffix is not None:
        return invalid_suffix
    if expected is None:
        return None
    wrapper = _find_wrapper_function(tree, ctx.op_name, suffix)
    if wrapper is None:
        return ctx.make_finding(
            "OL50", "FAIL",
            f"{impl_file}: 未发现 Layer K wrapper 函数 "
            f"(期望 `{ctx.op_name}_module{suffix}_wrapper` "
            f"或 `host_wrapper`)。",
            file=impl_file,
        )
    actual = _func_arg_names(wrapper)
    if actual == expected:
        return None
    fix_hint = ""
    if not expected and "primary_inputs" not in interfaces:
        fix_hint = (
            "\n提示: 当前 expected=[] 是因为 YAML 缺少顶层 `primary_inputs`；"
            "请按 pypto-op-designer 的 "
            "`module_interfaces.yaml` schema 补回该字段，"
            "`source: primary` 不能替代它。"
        )
    return ctx.make_finding(
        "OL50", "FAIL",
        f"{impl_file} 第 {wrapper.lineno} 行: wrapper `{wrapper.name}` 的 "
        f"参数顺序与 module_interfaces.yaml 不一致。\n"
        f"  expected (primary_inputs 顺序): {expected}\n"
        f"  actual:                         {actual}\n"
        f"修正方针: wrapper 显式参数应只包含 expected 列表，并保持同顺序。"
        f"{fix_hint}",
        file=impl_file,
        line=wrapper.lineno,
    )


@register("OL50")
def check_ol50(ctx: CheckContext) -> Finding:
    """host wrapper 的签名（参数名与顺序）必须与
    `eval/module_interfaces.yaml` 的 `primary_inputs` 一致。

    适用对象:
      - `modules/<op>_module<suffix>_impl.py`: 当前 phase 引用的 primary inputs
      - 顶层 `<op>_impl.py`: 全部 primary inputs

    YAML 不存在、PyYAML 未安装、wrapper 函数未发现时 SKIP。
    """
    interfaces = _load_module_interfaces(ctx)
    if interfaces is None:
        return ctx.make_finding(
            "OL50", "SKIP",
            "eval/module_interfaces.yaml 不存在或 PyYAML 未安装 — 无法核对契约",
        )
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL50", "SKIP", "无 impl 文件可供检查")

    for impl_file, tree, _ in _iter_parsed_impls(ctx, impl_files):
        failure = _wrapper_contract_finding(ctx, interfaces, impl_file, tree)
        if failure is not None:
            return failure
    return ctx.make_finding(
        "OL50", "PASS", "wrapper 参数顺序与 module_interfaces.yaml 一致",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL51 — 输出张量写回漏检
# ─────────────────────────────────────────────────────────────────────────────


def _named_keyword_targets(node: ast.Call, names: tuple[str, ...]) -> set[str]:
    return {
        keyword.value.id
        for keyword in node.keywords
        if keyword.arg in names and isinstance(keyword.value, ast.Name)
    }


def _assemble_target(node: ast.Call) -> ast.AST | None:
    if len(node.args) >= 3:
        return node.args[2]
    if len(node.args) >= 2 and isinstance(node.args[0], (ast.List, ast.Tuple)):
        return node.args[1]
    return None


def _call_writeback_targets(node: ast.Call) -> set[str]:
    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return set()
    receiver = func.value.id
    targets: set[str] = set()
    if receiver == "pypto" and func.attr == "assemble":
        target = _assemble_target(node)
        if isinstance(target, ast.Name):
            targets.add(target.id)
        targets.update(_named_keyword_targets(node, ("target", "dst", "out")))
    elif receiver == "pypto" and func.attr == "move":
        target = node.args[1] if len(node.args) >= 2 else None
        if isinstance(target, ast.Name):
            targets.add(target.id)
        targets.update(_named_keyword_targets(node, ("target", "dst", "out")))
    elif receiver == "pypto" and func.attr in _INPLACE_PYPTO_OPS:
        if node.args and isinstance(node.args[0], ast.Name):
            targets.add(node.args[0].id)
        targets.update(_named_keyword_targets(node, ("input", "y")))
    elif func.attr in ("move", "assemble", "index_add_", "scatter_", "axpy_"):
        targets.add(receiver)
    return targets


def _assigned_subscript_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    else:
        return set()
    return {
        target.value.id
        for target in targets
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
    }


def _collected_writeback_targets(tree: ast.Module) -> set[str]:
    """收集 impl 中"被写回到输出张量的 identifier 名"。

    识别模式:
      - `pypto.assemble(src, offsets, <name>)`
      - `pypto.assemble([(src, offsets), ...], <name>)`
      - `<name>[:] = expr` (Subscript Slice)
      - `pypto.move(src, <name>)`
      - `<name>.move(src)` / `<name>.assemble(src, offsets)`
      - `<name>[idx] = expr` 一般写法 (Subscript any)
      - `pypto.index_add_(<name>, ...)` — inplace 索引累加
      - `pypto.index_add__ub(<name>, ...)` — inplace UB 变体索引累加
      - `pypto.index_put_(<name>, ...)` — inplace 索引写入
      - `pypto.scatter_(<name>, ...)` — inplace 散射写入
      - `pypto.axpy_(<name>, ...)` — inplace AXPY (y = alpha*x + y)
      - `<name>.index_add_(...)` / `.scatter_(...)` / `.axpy_(...)`
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            out.update(_call_writeback_targets(node))
        out.update(_assigned_subscript_targets(node))
    return out


def _is_pypto_assemble_call(node: ast.AST) -> bool:
    """识别 `pypto.assemble(...)` 调用 (字面三段属性, OL01 风格)。"""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if not isinstance(f, ast.Attribute) or f.attr != "assemble":
        return False
    return isinstance(f.value, ast.Name) and f.value.id == "pypto"


def _local_call_name(node: ast.Call) -> str | None:
    """For `foo(...)` 形式调用, 返回 `foo`; 否则 None。"""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


_INPLACE_PYPTO_OPS = (
    "index_add_", "index_add__ub", "index_put_",
    "scatter_", "axpy_",
)


WritebackPair = tuple[str, ast.AST | None]


def _assemble_pairs(node: ast.Call) -> list[WritebackPair]:
    if not _is_pypto_assemble_call(node):
        return []
    if len(node.args) >= 3 and isinstance(node.args[2], ast.Name):
        return [(node.args[2].id, node.args[0])]
    is_batch = (
        len(node.args) >= 2
        and isinstance(node.args[0], (ast.List, ast.Tuple))
        and isinstance(node.args[1], ast.Name)
    )
    if not is_batch:
        return []
    target = node.args[1].id
    return [
        (target, segment.elts[0])
        for segment in node.args[0].elts
        if isinstance(segment, (ast.Tuple, ast.List)) and segment.elts
    ]


def _attribute_call_pairs(node: ast.Call) -> list[WritebackPair]:
    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return []
    receiver = func.value.id
    is_pypto_namespace = receiver == "pypto"
    is_inplace = func.attr in _INPLACE_PYPTO_OPS
    if not is_pypto_namespace and func.attr in ("move", "assemble") and node.args:
        return [(receiver, node.args[0])]
    if is_pypto_namespace and is_inplace:
        if node.args and isinstance(node.args[0], ast.Name):
            return [(node.args[0].id, None)]
        return []
    if not is_pypto_namespace and is_inplace:
        return [(receiver, None)]
    return []


def _assignment_writeback_pairs(node: ast.AST) -> list[WritebackPair]:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            return [(target.value.id, node.value)]
    if isinstance(node, ast.AugAssign):
        target = node.target
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            return [(target.value.id, node.value)]
    return []


def _collect_jit_writeback_pairs(
    entry: ast.FunctionDef,
    function_defs: dict[str, ast.FunctionDef],
    max_depth: int = 8,
) -> list[tuple[str, ast.AST | None]]:
    """从 @pypto.frontend.jit 入口出发, 沿同文件可达 call chain 收集
    所有写回操作的 (target_name, source_expr) 二元组。

    识别模式 (与 `_collected_writeback_targets` 一致, 但只在 JIT 调用图
    内部统计, 且额外保留 RHS 源表达式以便 OL51.b 检测平凡写入):

    - `pypto.assemble(src, [offsets], target)` → (target, src)
    - `pypto.assemble([(src, offsets), ...], target, ...)` → 每段一对
    - `target[:] = expr` / `target[idx] = expr` → (target, expr)
    - `target += expr` (AugAssign 形式) → (target, expr)
    - `target.move(src)` / `target.assemble(src, ...)` → (target, src)
    - `pypto.index_add_(target, ...)` / `pypto.scatter_(target, ...)` /
      `pypto.axpy_(target, ...)` 等就地累积 → (target, None)
    - `target.index_add_(...)` / `target.scatter_(...)` / `target.axpy_(...)`
      就地累积 → (target, None)

    source_expr 为 None 表示就地累积语义 (output = f(target_before, ...)),
    按 API 契约本质上不可能是 trivial pass-through, 故 OL51.b 不再检查。

    `max_depth` 限制 transitive 递归深度, 防止异常调用图导致代价爆炸。
    """
    pairs: list[WritebackPair] = []
    visited: set[str] = set()
    stack: list[ast.FunctionDef] = [entry]

    while stack:
        if len(visited) > max_depth:
            break
        func = stack.pop()
        if func.name in visited:
            continue
        visited.add(func.name)
        for node in ast.walk(func):
            pairs.extend(
                _node_writeback_pairs(node, function_defs, visited, stack)
            )
    return pairs


def _node_writeback_pairs(node, function_defs, visited, stack):
    if not isinstance(node, ast.Call):
        return _assignment_writeback_pairs(node)
    pairs = _assemble_pairs(node) + _attribute_call_pairs(node)
    callee = _local_call_name(node)
    if callee and callee in function_defs and callee not in visited:
        stack.append(function_defs[callee])
    return pairs


def _map_outputs_to_jit_params(
    expected_outputs: list[str],
    jit_func: ast.FunctionDef,
) -> dict[str, str]:
    """把 YAML 声明的 output 名映射到 JIT 函数引数。

    映射策略 (按优先级):
      1. 全员精确名匹配 — YAML 声明的所有 output 名都出现在 JIT 引数列表里 →
         直接同名映射 (推荐做法, agent prompt 也建议此风格)。
      2. 位置 fallback — 否则取 JIT 引数列表的末尾 N 个作为 output slot
         (与 OL50 wrapper 参数顺序规约保持一致)。
      3. JIT 引数数量不足 — 返回空 dict, 由调用方报错。
    """
    jit_params = [arg.arg for arg in jit_func.args.args]
    if len(jit_params) < len(expected_outputs):
        return {}
    name_matched = {o: o for o in expected_outputs if o in jit_params}
    if len(name_matched) == len(expected_outputs):
        return name_matched
    n = len(expected_outputs)
    last_params = jit_params[-n:]
    return dict(zip(expected_outputs, last_params))


def _direct_assignment(node: ast.AST) -> tuple[str, ast.AST] | None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    return target.id, node.value


def _modified_assignment_name(node: ast.AST) -> str | None:
    target = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AugAssign):
        target = node.target
    if isinstance(target, ast.Name) and isinstance(node, ast.AugAssign):
        return target.id
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return target.value.id
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    receiver = node.func.value
    if (
        isinstance(receiver, ast.Name)
        and receiver.id != "pypto"
        and node.func.attr in ("move", "assemble", "index_add_", "scatter_", "axpy_")
    ):
        return receiver.id
    return None


def _collect_function_local_assigns(
    func: ast.FunctionDef,
) -> dict[str, ast.AST]:
    """收集函数体内 `name = expr` 形式的直接赋值, 返回 name → expr 映射。

    用于 OL51.b 跟踪经过中间局部变量的平凡写入 (例如
    `z = pypto.zeros(...); pypto.assemble(z, ..., out)` 的 `z` 经回溯
    可识别为零值源)。

    保守排除规则: 如果某个 name 后续被 `name[:] = expr` (subscript-slice
    再赋值) 或 `name.move(...)` / `name[idx] = expr` 等方式修改, 则不能
    简单地用最初的 `name = expr` 判断其语义内容 (经典 Task_0527 风格:
    `acc = pypto.tensor(...); acc[:] = partial; pypto.assemble(acc, ..., out)`
    的 acc 实际承载的是 matmul 结果, 不是初始空 tensor)。这种被覆写过的
    name 从 map 中剔除, OL51.b 对其判定回退为"非平凡"。

    限制: 仅识别简单单目标赋值; 不展开元组解包; 不跨函数边界跟踪。
    如果同一 name 被多次直接赋值且未被 subscript 修改, 保留最后一次。
    """
    direct_assigns: dict[str, ast.AST] = {}
    subscript_modified: set[str] = set()
    for node in ast.walk(func):
        assignment = _direct_assignment(node)
        if assignment is not None:
            name, value = assignment
            direct_assigns[name] = value
        modified_name = _modified_assignment_name(node)
        if modified_name is not None:
            subscript_modified.add(modified_name)
    return {
        k: v for k, v in direct_assigns.items()
        if k not in subscript_modified
    }


def _is_trivial_write_source(
    source: ast.AST,
    jit_input_params: set[str],
    local_assigns: dict[str, ast.AST] | None = None,
    _depth: int = 0,
) -> bool:
    """识别"无实质计算"的写入源表达式 (OL51.b 用途)。

    平凡写入模式 (返回 True):
      - 直接引用一个 JIT 输入参数: `out[:] = x`
      - 中间变量经过赋值最终来自上述 (local_assigns 回溯): 例如
        `z = pypto.zeros(...); pypto.assemble(z, ..., out)`
      - JIT 上下文未初始化分配: `pypto.zeros/ones/empty/full/tensor(...)`
      - 输入参数的浅层 view / reshape: `x.reshape(...)` /
        `pypto.view(x, ...)` / `pypto.reshape(x, ...)`
      - 输入参数的下标取值: `x[idx]`

    非平凡写入 (返回 False): 任何 compute op (`pypto.matmul` / `pypto.exp` /
    `pypto.add` 等) 的返回值; 来自未被本规则识别的局部变量 (保守假设是计算
    结果, 避免误报)。

    `_depth` 限制 local_assigns 链上的递归深度, 避免病态赋值循环导致栈溢出。
    """
    if _depth > 4:
        return False
    # 直接 Name 引用 → 是 input param 才算平凡; 否则尝试 local_assigns 回溯
    if isinstance(source, ast.Name):
        if source.id in jit_input_params:
            return True
        if local_assigns is not None and source.id in local_assigns:
            return _is_trivial_write_source(
                local_assigns[source.id], jit_input_params, local_assigns, _depth + 1,
            )
        return False
    # 输入参数的下标访问形式
    if isinstance(source, ast.Subscript) and isinstance(source.value, ast.Name):
        return source.value.id in jit_input_params
    if isinstance(source, ast.Call):
        f = source.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            # pypto.zeros / ones / empty / full / tensor — 未初始化或零值
            if f.value.id == "pypto" and f.attr in (
                "zeros", "ones", "empty", "full", "tensor", "zeros_like",
                "empty_like", "ones_like",
            ):
                return True
            # 输入参数的浅层视图：pypto 命名空间的 view 或 reshape
            first_arg_is_name = (
                bool(source.args) and isinstance(source.args[0], ast.Name)
            )
            if (
                f.value.id == "pypto"
                and f.attr in ("view", "reshape")
                and first_arg_is_name
            ):
                return source.args[0].id in jit_input_params
            # 张量方法形式的浅层视图：输入参数的 reshape、view、contiguous、clone、detach
            if (
                f.attr in ("reshape", "view", "contiguous", "clone", "detach")
                and isinstance(f.value, ast.Name)
            ):
                return f.value.id in jit_input_params
    return False


def _expected_impl_outputs(
    interfaces: dict,
    impl_file: str,
    op_name: str,
) -> list[str]:
    suffix = _impl_file_to_module_suffix(impl_file, op_name)
    if suffix is None:
        return _final_outputs_for_top_level(interfaces)
    module_ids = _suffix_to_module_ids(suffix)
    if not module_ids:
        return []
    return _outputs_for_modules(interfaces, module_ids)


def _ol51_count_failure(
    ctx: CheckContext,
    impl_file: str,
    expected_outputs: list[str],
    writebacks: set[str],
) -> Finding | None:
    if len(writebacks) >= len(expected_outputs):
        return None
    return ctx.make_finding(
        "OL51", "FAIL",
        f"[OL51.a] {impl_file}: YAML 声明了 {len(expected_outputs)} 个输出 "
        f"{expected_outputs}, 但 impl 中只检测到 "
        f"{len(writebacks)} 个写回点 (writebacks={sorted(writebacks)})。\n"
        f"修正方针: 对每个输出, 在 JIT 函数内 / 任意 Layer I-J 中执行 "
        f"`pypto.assemble(src, offsets, <buffer>)`、`<buffer>.move(src)` "
        f"或 `<buffer>[:] = expr`。"
        f"漏写回 = Verify 阶段「全零输出」型精度 FAIL 的典型原因。",
        file=impl_file,
    )


def _jit_trivial_hits(
    expected_outputs: list[str],
    jit_funcs: list[ast.FunctionDef],
    function_defs: dict[str, ast.FunctionDef],
) -> list[str]:
    trivial_hits: list[str] = []
    for jit_func in jit_funcs:
        mapping = _map_outputs_to_jit_params(expected_outputs, jit_func)
        if not mapping:
            continue
        jit_params = [arg.arg for arg in jit_func.args.args]
        jit_pairs = _collect_jit_writeback_pairs(jit_func, function_defs)
        jit_inputs = set(jit_params) - set(mapping.values())
        local_assigns = _collect_function_local_assigns(jit_func)
        for output, param in mapping.items():
            sources = [source for name, source in jit_pairs if name == param]
            if sources and all(
                source is not None
                and _is_trivial_write_source(source, jit_inputs, local_assigns)
                for source in sources
            ):
                trivial_hits.append(f"{jit_func.name}:{output}")
    return trivial_hits


def _ol51_trivial_failure(
    ctx: CheckContext,
    impl_file: str,
    trivial_hits: list[str],
) -> Finding | None:
    if not trivial_hits:
        return None
    return ctx.make_finding(
        "OL51", "FAIL",
        f"[OL51.b] {impl_file}: JIT 仅以平凡 pass-through "
        f"(直接复制输入 / pypto.zeros / 浅层 reshape) 写出 {trivial_hits}。\n"
        f"修正方针: 输出值必须由真实 PyPTO compute op (pypto.matmul / "
        f"pypto.exp / pypto.add / ...) 产生 "
        f"(Issue #2083 _layout_check_placeholder 类占位实现 = 此处 FAIL)。",
        file=impl_file,
    )


@register("OL51")
def check_ol51(ctx: CheckContext) -> Finding:
    """对 `eval/module_interfaces.yaml` 声明的 N 个输出, impl 必须同时满足:

    OL51.a — 数量层:
        impl 文件全体至少存在 N 个写回点
        (`pypto.assemble(..., t)` / `t[:] = ...` / `pypto.move(..., t)` 等)。
        及早捕获"完全漏写 assemble → 全零输出 → Verify FAIL"型 bug。

    OL51.b — 非平凡写入层:
        JIT 内每个 output 的写入源不得是平凡 pass-through:
          - 直接复制输入参数 (`out[:] = x` 且 x 是 JIT 输入)
          - `pypto.zeros / empty / tensor(...)` 等未初始化值
          - 仅 reshape / view 的浅层变换
        实际计算必须由 `pypto.matmul / pypto.exp / ...` 等 compute op 产生。
        此层封堵 Issue #2083 描述的 _layout_check_placeholder 类占位实现。

    背景: 代码中 output buffer 不必沿用 YAML 同名 identifier, 经常使用
    `y0`, `out`, `C_npu` 等本地名。OL51.a 不强制名称一致, 只检查"写回点的数量";
    OL51.b 借助 output→JIT-param 映射定位写入源, 仅判定其是否平凡。

    适用对象:
      - `modules/<op>_module<suffix>_impl.py`: 当前 module 的 outputs[*]
      - 顶层 `<op>_impl.py`: final_outputs[*]

    YAML 不存在 / PyYAML 未安装 / 无输出声明时 SKIP。
    """
    interfaces = _load_module_interfaces(ctx)
    if interfaces is None:
        return ctx.make_finding(
            "OL51", "SKIP",
            "eval/module_interfaces.yaml 不存在或 PyYAML 未安装 — 无法核对写回",
        )
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL51", "SKIP", "无 impl 文件可供检查")

    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        expected_outputs = _expected_impl_outputs(
            interfaces, impl_file, ctx.op_name,
        )
        if not expected_outputs:
            continue

        # ── OL51.a — 数量层: 写回点总数 ≥ N (file scope) ──────────────────
        writebacks = _collected_writeback_targets(tree)
        failure = _ol51_count_failure(
            ctx, impl_file, expected_outputs, writebacks,
        )
        if failure:
            return failure

        # ── OL51.b — 非平凡写入层: JIT 写出的 output 不得是平凡 pass-through ──
        jit_funcs = _get_jit_functions(tree, aliases)
        # 无 JIT 函数 → 跳过 .c (纯 host 实现由 OL60/OL62 把关)。
        if not jit_funcs:
            continue
        function_defs = {
            node.name: node
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef)
        }
        trivial_hits = _jit_trivial_hits(
            expected_outputs, jit_funcs, function_defs,
        )
        failure = _ol51_trivial_failure(ctx, impl_file, trivial_hits)
        if failure:
            return failure
    return ctx.make_finding(
        "OL51", "PASS",
        "impl 写回点数 ≥ YAML 输出数, 且 (有 JIT 时) 写出非平凡 (OL51.a + OL51.b)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL62 — impl 内 torch 仅限 layout/alloc/cast; 数值计算必须在 @jit 图内
# ─────────────────────────────────────────────────────────────────────────────

# torch.<X>(...) 允许的非算术调用 (alloc / 构造 / layout / 随机初始化 / 设备)
_OL62_TORCH_OK = frozenset({
    "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
    "full_like", "tensor", "as_tensor", "from_numpy", "arange", "eye",
    "randn", "rand", "randint", "normal", "randn_like", "rand_like",
    "reshape", "cat", "stack", "chunk", "split", "unbind", "movedim",
    "swapaxes", "swapdims", "broadcast_to", "flatten", "squeeze", "unsqueeze",
    "permute", "transpose", "pad", "tril", "triu", "tril_indices",
    "triu_indices", "meshgrid", "repeat", "tile",
    "set_device", "manual_seed", "device", "get_default_dtype",
    "set_default_dtype", "no_grad", "set_grad_enabled", "is_available",
    "device_count", "current_device",
})
# 张量算术方法 (recv.<X>(...)) —— 命中即 FAIL (recv 非 pypto/np/math 时)
_OL62_METHOD_ARITH = frozenset({
    "matmul", "mm", "bmm", "dot", "mv", "einsum", "exp", "expm1", "log",
    "log1p", "log2", "log10", "softmax", "log_softmax", "sum", "mean",
    "median", "std", "var", "prod", "cumsum", "cumprod", "rsqrt", "sqrt",
    "sigmoid", "tanh", "relu", "gelu", "silu", "pow", "neg", "abs", "maximum",
    "minimum", "amax", "amin", "argmax", "argmin", "where", "clamp",
    "clamp_min", "clamp_max", "erf", "sin", "cos", "tan", "sign", "floor",
    "ceil", "round", "reciprocal", "addmm", "baddbmm", "addbmm", "scatter",
    "scatter_add", "gather", "index_select", "index_add", "norm", "dist",
    "cross", "outer", "masked_fill", "add", "sub", "mul", "div",
})
# 这些 receiver 上的 .<arith>() 不是 torch 张量算术 (pypto op / scalar / 模块)
_OL62_SKIP_ROOTS = frozenset({"np", "numpy", "math", "os", "sys", "self", "kwargs"})
# 非生产函数 (self-test / demo) —— 不参与 OL62 检查
_OL62_NONPROD = frozenset({"_validate", "_make_inputs", "main", "_self_test",
                           "_demo", "_test"})


def _ol62_root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _ol62_production_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """生产路径节点: 跳过 `if __name__ == '__main__'` 块与 self-test / demo 函数。"""
    for top in tree.body:
        if isinstance(top, ast.If):
            t = top.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"):
                continue
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and top.name in _OL62_NONPROD:
            continue
        yield from ast.walk(top)


def _ol62_hit(
    node: ast.AST,
    skip_roots: set[str] | frozenset[str],
) -> tuple[int, str] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return node.lineno, "@ (matmul)"
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    attr = node.func.attr
    root = _ol62_root_name(node.func.value)
    if root == "torch" and attr not in _OL62_TORCH_OK:
        return node.lineno, f"torch.{attr}(...)"
    if root in ("F", "functional"):
        return node.lineno, f"F.{attr}(...)"
    if (
        attr in _OL62_METHOD_ARITH
        and root not in skip_roots
        and root != "torch"
    ):
        return node.lineno, f"{root or ''}.{attr}(...)"
    return None


def _ol62_hits(
    tree: ast.Module,
    skip_roots: set[str] | frozenset[str],
) -> list[tuple[int, str]]:
    return [
        hit
        for node in _ol62_production_nodes(tree)
        if (hit := _ol62_hit(node, skip_roots)) is not None
    ]


@register("OL62")
def check_ol62(ctx: CheckContext) -> Finding:
    """`<op>_impl.py` 内 torch 仅可用于 layout / alloc / cast / reshape;
    任何 torch 张量算术 (torch.matmul / .exp / .sum / `@` / ...) 即 FAIL。

    算子的数值计算必须落在 @pypto.frontend.jit 图内 (用 `pypto.*` op)。host
    侧 (wrapper / 非 JIT helper) 只允许打包、分配、布局、dtype 转换。

    封堵 dummy-JIT 作弊全谱: dead JIT + host 全 torch / 小 JIT + torch 主计算 /
    torch 旁路再计算。AST 检测, 自动忽略注释与字符串; 跳过 `__main__` 块与
    `_validate` / `_make_inputs` 等自测函数。`pypto.* / np.* / math.*` 接收者的
    同名方法 (如 pypto.matmul) 不误判。

    YAML 无关; 只看 impl 源码。无 impl 文件时 SKIP。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL62", "SKIP", "无 impl 文件可供检查")
    for impl_file, tree, aliases in _iter_parsed_impls(ctx, impl_files):
        skip_roots = _OL62_SKIP_ROOTS | aliases
        hits = _ol62_hits(tree, skip_roots)
        if hits:
            uniq = sorted(set(hits))[:8]
            detail = "; ".join(f"L{ln}:{m}" for ln, m in uniq)
            return ctx.make_finding(
                "OL62", "FAIL",
                f"{impl_file}: 检测到 host 侧 torch 张量算术 ({detail})。\n"
                f"修正方针: 算子的数值计算必须在 @pypto.frontend.jit 图内用 "
                f"`pypto.*` op 实现; host (wrapper / 非 JIT helper) 仅允许 "
                f"layout / alloc / cast / reshape。torch 主计算 = dummy-JIT 作弊。",
                file=impl_file,
            )
    return ctx.make_finding(
        "OL62", "PASS",
        "impl 内 torch 仅用于 layout/alloc/cast; 数值计算在 JIT 图内",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL53 — MEMORY.md → Golden function inventory: 全行均 ✅ 校验
# ─────────────────────────────────────────────────────────────────────────────


_INVENTORY_HEADING_RE = re.compile(
    r"^\s*##\s+Golden\s+function\s+inventory.*$",
    re.IGNORECASE | re.MULTILINE,
)
_INVENTORY_NEXT_HEADING_RE = re.compile(r"^\s*##\s+")
# Match a markdown table row that ends with a Status cell containing ✅ or ❌
# (Verify last `|`-cell of the row contains one of the markers; ignore separator rows.)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _extract_inventory_rows(memory_text: str) -> list[tuple[int, str]]:
    """Return [(line_no, row_text), …] of rows inside the
    "Golden function inventory" section of MEMORY.md. Excludes:
      - header row (the column titles)
      - separator row (`|---|---|…`)
      - the comment / how-to lines around the table
    Only data rows are returned.
    """
    lines = memory_text.splitlines()
    in_section = False
    rows: list[tuple[int, str]] = []
    seen_header = False
    seen_sep = False
    for idx, line in enumerate(lines, start=1):
        if not in_section:
            if _INVENTORY_HEADING_RE.match(line):
                in_section = True
                seen_header = False
                seen_sep = False
            continue
        if _INVENTORY_NEXT_HEADING_RE.match(line):
            break
        if not _TABLE_ROW_RE.match(line):
            continue
        # First table row is header, second is separator
        if not seen_header:
            seen_header = True
            continue
        if not seen_sep:
            # Heuristic: separator row has only `|`, `-`, `:`, spaces
            content = line.replace("|", "").replace("-", "").replace(":", "").strip()
            if not content:
                seen_sep = True
                continue
        rows.append((idx, line))
    return rows


def _row_status(row_text: str) -> str:
    """Pick the status cell (✅ / ❌ / other) from the row's last non-empty cell."""
    cells = [c.strip() for c in row_text.strip().strip("|").split("|")]
    if not cells:
        return ""
    # Walk from right, skip empty trailing cells
    for c in reversed(cells):
        if c:
            return c
    return ""


@register("OL53")
def check_ol53(ctx: CheckContext) -> Finding:
    """`custom/<op>/MEMORY.md` 中 "Golden function inventory" 必须每行在
    Status 列均标记为 ✅, 否则 FAIL (残留 ❌ 是实现遗漏的典型信号)。

    执行时机:
      - Stage 5 / 6 的 complete_stage 门禁
      - 任意 stage 的 complete_phase 时该规则 SKIP
        (phase 范围外的行保留 ❌ 属正常)

    MEMORY.md 或该章节不存在时 SKIP (由 OL09 等其他规则覆盖)。
    """
    memory_file = "MEMORY.md"
    if not ctx.file_exists(memory_file):
        return ctx.make_finding("OL53", "SKIP", f"{memory_file} 不存在")
    text = ctx.read_file(memory_file)
    if not _INVENTORY_HEADING_RE.search(text):
        return ctx.make_finding(
            "OL53", "SKIP",
            f"{memory_file}: 缺少 'Golden function inventory' 章节",
            file=memory_file,
        )
    # phase_scope 已设置时, 未启动 module 的 ❌ 残留属正常情况
    if getattr(ctx, "phase_scope", None):
        return ctx.make_finding(
            "OL53", "SKIP",
            f"phase_scope={ctx.phase_scope}: 允许未启动 module 的 ❌",
            file=memory_file,
        )
    rows = _extract_inventory_rows(text)
    if not rows:
        return ctx.make_finding(
            "OL53", "SKIP",
            f"{memory_file}: Golden function inventory 无数据行",
            file=memory_file,
        )
    unresolved = [(ln, _row_status(r)) for ln, r in rows]
    bad = [
        (ln, st)
        for ln, st in unresolved
        if "❌" in st or st in {"", "❌", "TODO", "todo", "未实现", "WIP"}
    ]
    if bad:
        n_rows = len(rows)
        bad_lines = ", ".join(f"L{ln}" for ln, _ in bad[:8])
        more = f" 等共 {len(bad)} 行" if len(bad) > 8 else ""
        return ctx.make_finding(
            "OL53", "FAIL",
            f"{memory_file}: Golden function inventory 共 {n_rows} 行中 "
            f"{len(bad)} 行带 ❌/未解决标记 ({bad_lines}{more})。\n"
            f"修正方针: 将每行更新为 ✅ 并附 impl 中对应 PyPTO 调用的行号; "
            f"如果有意省略, 请在另一行注明理由。残留 ❌ 时 complete_stage 不通过。",
            file=memory_file,
        )
    return ctx.make_finding(
        "OL53", "PASS",
        f"Golden function inventory 全 {len(rows)} 行均已 ✅",
        file=memory_file,
    )
