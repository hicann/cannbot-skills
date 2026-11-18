# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch5 参数说明 — 严格按四维表提取。

四维表映射:
- basic_api: 文件(头+impl)✅ 函数(声明+Impl)✅ 代码段(static_assert)✅ 注释段(@param)✅
- 其他 6 种: 文件(头)✅ 函数(声明)✅ 代码段(参数列表)✅ 注释段❌
"""

from config import CH_PARAM_DESC
from evidence import Evidence

CHAPTER_NAME = CH_PARAM_DESC


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取：文件 + 函数 + 代码段 + 注释段。"""
    e = extractor
    evidence = []
    evidence.extend(_file_dimension(api_type, decl_file, impl_files))
    evidence.extend(_function_dimension(api_name, api_type, decl_file, impl_files, e))
    evidence.extend(_code_dimension(api_name, api_type, decl_file, impl_files, e))
    evidence.extend(_comment_dimension(api_type, api_name, decl_file, e))
    return evidence


# ==================== 四维独立提取 ====================


def _file_dimension(api_type, decl_file, impl_files):
    """文件维度：basic_api 提取头+impl，其他提取头文件。"""
    evidence = []
    if api_type == "basic_api":
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
    else:
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


def _function_dimension(api_name, api_type, decl_file, impl_files, e):
    """函数维度：basic_api 提取声明+Impl，其他提取声明，合并所有 overload。"""
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
    if api_type == "basic_api" and impl_files:
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


def _code_dimension(api_name, api_type, decl_file, impl_files, e):
    """代码段维度：basic_api 提取 static_assert，其他提取参数列表。"""
    if api_type == "basic_api":
        if impl_files:
            return e.extract_static_asserts(impl_files)
        return []
    else:
        return e.extract_function_params(api_name, decl_file)


def _comment_dimension(api_type, api_name, decl_file, e):
    """注释段维度：basic_api 提取 @param，其他无注释段。"""
    if api_type == "basic_api":
        return e.extract_doxygen_params(api_name, decl_file)
    return []
