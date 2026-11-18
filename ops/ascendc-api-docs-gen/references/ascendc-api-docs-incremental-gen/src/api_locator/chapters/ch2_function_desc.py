# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch2 功能说明 — 严格按四维表提取。

四维表映射:
- basic_api: 文件(头)✅ 函数(声明)✅ 代码段❌ 注释段(@brief)✅
- 其他 6 种: 文件(头)✅ 函数(声明)✅ 代码段(签名前200字符)✅ 注释段❌
"""

from config import CH_FUNCTION_DESC
from evidence import Evidence

CHAPTER_NAME = CH_FUNCTION_DESC


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取：文件 + 函数 + 代码段 + 注释段。"""
    evidence = []
    evidence.extend(_file_dimension(api_type, decl_file))
    evidence.extend(_function_dimension(api_name, api_type, decl_file, extractor))
    evidence.extend(_code_dimension(api_name, api_type, decl_file, extractor))
    evidence.extend(_comment_dimension(api_name, api_type, decl_file, extractor))
    return evidence


# ==================== 四维独立提取 ====================


def _file_dimension(api_type, decl_file):
    """文件维度：头文件存在即生成。"""
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


def _function_dimension(api_name, api_type, decl_file, e):
    """函数维度：独立提取声明，合并所有 overload。"""
    decl_snippets = e.extract_all_decl_snippets(api_name, decl_file)
    if not decl_snippets:
        return []
    merged = "\n".join(s.content for s in decl_snippets)
    return [
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
    ]


def _code_dimension(api_name, api_type, decl_file, e):
    """代码段维度：basic_api 无代码段，其他类型提取函数签名片段，合并所有 overload。"""
    if api_type == "basic_api":
        return []
    snippets = e.extract_all_decl_snippets(api_name, decl_file)
    if not snippets:
        return []
    merged = "\n".join(s.content for s in snippets)
    return [
        Evidence(
            dimension="code_snippet",
            file=decl_file,
            line=snippets[0].line,
            content=merged,
            metadata={"type": "function_signature_code", "overload_count": len(snippets)},
        )
    ]


def _comment_dimension(api_name, api_type, decl_file, e):
    """注释段维度：basic_api 提取 @brief，其他无注释段。"""
    if api_type == "basic_api":
        return e.extract_doxygen_brief(api_name, decl_file)
    return []
