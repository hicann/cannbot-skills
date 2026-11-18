# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch7 约束说明 — 严格按四维表提取。

四维表映射:
- basic_api: 文件(头+check+impl)✅ 函数(CheckFunc+Impl)✅ 代码段(arch+CheckFunc+static_assert)✅ 注释段❌
- adv_api: 文件(头+api_check)✅ 函数(文件名)✅ 代码段(arch+文件名)✅ 注释段❌
- simt_api: 文件(头)✅ 函数(声明)✅ 代码段❌ 注释段❌
- tensor_api: 文件(头)✅ 函数(声明)✅ 代码段❌ 注释段❌
- c_api: 文件(头)✅ 函数(声明)✅ 代码段(arch)✅ 注释段❌
- aicpu_api: 全部无
- utils: 文件(头)✅ 函数(声明)✅ 代码段❌ 注释段❌
"""

from config import CH_CONSTRAINTS, IMPL_CHECK_GLOB
from evidence import Evidence

CHAPTER_NAME = CH_CONSTRAINTS


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取：文件 + 函数 + 代码段 + 注释段。"""
    extractors = {
        "basic_api": _extract_basic,
        "adv_api": _extract_adv,
        "simt_api": _extract_signature,
        "tensor_api": _extract_signature,
        "c_api": _extract_c,
        "aicpu_api": _extract_aicpu,
        "utils": _extract_signature,
    }
    return extractors[api_type](api_name, api_type, decl_file, impl_files, extractor)


# ==================== 四维独立提取 ====================


def _extract_basic(api_name, api_type, decl_file, impl_files, e):
    """basic_api: 文件(头+check+impl)✅ 函数(CheckFunc+Impl)✅ 代码段(arch+CheckFunc+static_assert)✅"""
    evidence = []
    evidence.extend(_file_basic(api_name, decl_file, impl_files))
    evidence.extend(_function_basic(api_name, decl_file, impl_files, e))
    evidence.extend(_code_basic(decl_file, impl_files, e))
    return evidence


def _file_basic(api_name, decl_file, impl_files):
    """文件维度：头文件 + check 文件 + impl 文件。"""
    import glob

    evidence = []
    if decl_file:
        evidence.append(
            Evidence(
                dimension="file",
                file=decl_file,
                line=1,
                content=decl_file,
                metadata={"type": "header_file"},
            )
        )
    # check 文件
    for cf in sorted(glob.glob(IMPL_CHECK_GLOB)):
        evidence.append(
            Evidence(
                dimension="file",
                file=cf,
                line=1,
                content=cf,
                metadata={"type": "check_file"},
            )
        )
    for impl in impl_files:
        evidence.append(
            Evidence(
                dimension="file",
                file=impl,
                line=1,
                content=impl,
                metadata={"type": "impl_file"},
            )
        )
    return evidence


def _function_basic(api_name, decl_file, impl_files, e):
    """函数维度：CheckFunc + 声明 + Impl，合并所有 overload。"""
    evidence = []
    check_funcs = e.extract_check_functions(api_name, "basic_api", impl_files)
    evidence.extend(check_funcs)
    decl_snippets = e.extract_all_decl_snippets(api_name, decl_file)
    if decl_snippets:
        merged = "\n".join(s.content for s in decl_snippets)
        evidence.append(
            Evidence(
                dimension="function",
                file=decl_file,
                line=decl_snippets[0].line,
                content=f"{api_name} (位于 {decl_file})",
                metadata={
                    "type": "impl_function",
                    "name": api_name,
                    "snippet": merged,
                    "overload_count": len(decl_snippets),
                },
            )
        )
    if impl_files:
        evidence.append(
            Evidence(
                dimension="function",
                file=impl_files[0],
                line=1,
                content=f"{api_name}Impl (位于 {impl_files[0]})",
                metadata={"type": "impl_function", "name": f"{api_name}Impl"},
            )
        )
    return evidence


def _code_basic(decl_file, impl_files, e):
    """代码段维度：arch_condition + static_assert。"""
    evidence = []
    evidence.extend(e.extract_arch_conditions(decl_file))
    if impl_files:
        evidence.extend(e.extract_static_asserts(impl_files))
    return evidence


def _extract_adv(api_name, api_type, decl_file, impl_files, e):
    """adv_api: 文件(头+api_check)✅ 函数(文件名)✅ 代码段(arch+文件名)✅"""
    evidence = []
    evidence.extend(_file_adv(decl_file))
    evidence.extend(_function_adv(api_name, decl_file, impl_files, e))
    evidence.extend(_code_adv(decl_file, e))
    return evidence


def _file_adv(decl_file):
    """文件维度：头文件 + api_check 文件。"""
    evidence = []
    if decl_file:
        evidence.append(
            Evidence(
                dimension="file",
                file=decl_file,
                line=1,
                content=decl_file,
                metadata={"type": "header_file"},
            )
        )
    return evidence


def _function_adv(api_name, decl_file, impl_files, e):
    """函数维度：api_check + 声明 + impl，合并所有 overload。"""
    evidence = []
    api_checks = e.extract_check_functions(api_name, "adv_api")
    evidence.extend(api_checks)
    decl_snippets = e.extract_all_decl_snippets(api_name, decl_file)
    if decl_snippets:
        merged = "\n".join(s.content for s in decl_snippets)
        evidence.append(
            Evidence(
                dimension="function",
                file=decl_file,
                line=decl_snippets[0].line,
                content=f"{api_name} (位于 {decl_file})",
                metadata={
                    "type": "impl_function",
                    "name": api_name,
                    "snippet": merged,
                    "overload_count": len(decl_snippets),
                },
            )
        )
    if impl_files:
        evidence.append(
            Evidence(
                dimension="function",
                file=impl_files[0],
                line=1,
                content=f"{api_name} (位于 {impl_files[0]})",
                metadata={"type": "impl_function", "name": api_name},
            )
        )
    return evidence


def _code_adv(decl_file, e):
    """代码段维度：arch_condition。"""
    return e.extract_arch_conditions(decl_file)


def _extract_signature(api_name, api_type, decl_file, impl_files, e):
    """simt_api / tensor_api / utils: 文件(头)✅ 函数(声明)✅ 代码段❌ 注释段❌"""
    evidence = []
    evidence.extend(_file_signature(decl_file))
    evidence.extend(_function_signature(api_name, decl_file, impl_files, e))
    # 代码段：无
    # 注释段：无
    return evidence


def _file_signature(decl_file):
    """文件维度：头文件。"""
    if decl_file:
        return [
            Evidence(
                dimension="file",
                file=decl_file,
                line=1,
                content=decl_file,
                metadata={"type": "header_file"},
            )
        ]
    return []


def _function_signature(api_name, decl_file, impl_files, e):
    """函数维度：声明，合并所有 overload。"""
    evidence = []
    decl_snippets = e.extract_all_decl_snippets(api_name, decl_file)
    if decl_snippets:
        merged = "\n".join(s.content for s in decl_snippets)
        evidence.append(
            Evidence(
                dimension="function",
                file=decl_file,
                line=decl_snippets[0].line,
                content=f"{api_name} (位于 {decl_file})",
                metadata={
                    "type": "impl_function",
                    "name": api_name,
                    "snippet": merged,
                    "overload_count": len(decl_snippets),
                },
            )
        )
    if impl_files:
        evidence.append(
            Evidence(
                dimension="function",
                file=impl_files[0],
                line=1,
                content=f"{api_name} (位于 {impl_files[0]})",
                metadata={"type": "impl_function", "name": api_name},
            )
        )
    return evidence


def _extract_c(api_name, api_type, decl_file, impl_files, e):
    """c_api: 文件(头)✅ 函数(声明)✅ 代码段(arch)✅"""
    evidence = []
    evidence.extend(_file_signature(decl_file))
    evidence.extend(_function_signature(api_name, decl_file, impl_files, e))
    evidence.extend(_code_c(decl_file, e))
    return evidence


def _code_c(decl_file, e):
    """代码段维度：arch_condition。"""
    return e.extract_arch_conditions(decl_file)


def _extract_aicpu(api_name, api_type, decl_file, impl_files, e):
    """aicpu_api: 全部无"""
    return []
