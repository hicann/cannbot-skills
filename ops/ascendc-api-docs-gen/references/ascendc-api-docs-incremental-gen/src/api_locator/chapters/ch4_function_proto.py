# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch4 函数原型 — 严格按四维表提取。

四维表映射:
- 全部 7 种: 文件(头)✅ 函数(声明)✅ 代码段(完整声明)✅ 注释段❌
"""

from config import CH_FUNCTION_PROTO
from evidence import Evidence

CHAPTER_NAME = CH_FUNCTION_PROTO


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取：文件 + 函数 + 代码段 + 注释段。"""
    evidence = []
    evidence.extend(_file_dimension(decl_file))
    evidence.extend(_function_dimension(api_name, decl_file, extractor))
    evidence.extend(_code_dimension(api_name, decl_file, extractor))
    # 注释段：无
    return evidence


# ==================== 四维独立提取 ====================


def _file_dimension(decl_file):
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


def _function_dimension(api_name, decl_file, e):
    """函数维度：独立提取声明，合并所有 overload。"""
    snippets = e.extract_all_decl_snippets(api_name, decl_file)
    if not snippets:
        return []
    merged = "\n".join(s.content for s in snippets)
    return [
        Evidence(
            dimension="function",
            file=decl_file,
            line=snippets[0].line,
            content=f"{api_name} (位于 {decl_file})",
            metadata={"type": "impl_function", "name": api_name, "snippet": merged, "overload_count": len(snippets)},
        )
    ]


def _code_dimension(api_name, decl_file, e):
    """代码段维度：完整函数声明片段，合并所有 overload。"""
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
            metadata={"type": "declaration_snippet", "overload_count": len(snippets)},
        )
    ]
