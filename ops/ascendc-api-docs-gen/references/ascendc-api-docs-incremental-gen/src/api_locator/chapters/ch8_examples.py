# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ch8 调用示例 — 严格按四维表提取。

四维表映射:
- 全部 7 种: 文件(tests+examples)✅ 函数(调用函数名)✅ 代码段(完整调用代码)✅ 注释段❌
"""

import os
import re

import config
from evidence import Evidence

CHAPTER_NAME = config.CH_EXAMPLES


def extract(api_name, api_type, decl_file, impl_files, test_files, extractor, export_root):
    """四维独立提取：文件 + 函数 + 代码段 + 注释段。"""
    evidence = []
    evidence.extend(_file_dimension(test_files, export_root, api_name))
    evidence.extend(_function_dimension(api_name, test_files, export_root))
    evidence.extend(_code_dimension(api_name, test_files, export_root, extractor))
    # 注释段：无
    return evidence


# ==================== 四维独立提取 ====================


def _file_dimension(test_files, export_root, api_name):
    """文件维度：test 文件 + example 文件。"""
    evidence = []
    for tf in test_files:
        evidence.append(
            Evidence(
                dimension="file",
                file=tf,
                line=1,
                content=tf,
                metadata={"type": "test_file"},
            )
        )
    example_files = _find_example_files(api_name, export_root)
    for ef in example_files:
        evidence.append(
            Evidence(
                dimension="file",
                file=ef,
                line=1,
                content=ef,
                metadata={"type": "example_file"},
            )
        )
    return evidence


def _function_dimension(api_name, test_files, export_root):
    """函数维度：从 test 和 example 文件中提取包含 API 名的函数。"""
    evidence = []
    seen = set()
    for tf in test_files:
        _extract_functions_from_file(api_name, tf, export_root, evidence, seen)
    example_files = _find_example_files(api_name, export_root)
    for ef in example_files:
        _extract_functions_from_file(api_name, ef, export_root, evidence, seen)
    return evidence


def _extract_functions_from_file(api_name, file_path, base_path, evidence, seen):
    """从单个文件中提取包含 API 调用的函数名作为 function 证据。"""
    import os

    full_path = os.path.join(base_path, file_path) if not os.path.isabs(file_path) else file_path
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError):
        return
    for i, line in enumerate(lines, 1):
        if re.search(rf"\b{re.escape(api_name)}\b", line):
            # 往上找最近的函数定义（含 TEST/TEST_F 宏和普通函数）
            for j in range(i - 2, max(0, i - 30), -1):
                # 匹配 TEST_F/TEST 宏
                tm = re.match(r"\s*(TEST_F|TEST)\s*\((\w+)", lines[j])
                if tm:
                    func_name = tm.group(2)
                    key = f"{func_name}@{file_path}"
                    if key not in seen:
                        seen.add(key)
                        evidence.append(
                            Evidence(
                                dimension="function",
                                file=file_path,
                                line=j + 1,
                                content=f"{func_name} (位于 {file_path})",
                                metadata={"type": "impl_function", "name": func_name},
                            )
                        )
                    break
                # 匹配普通函数定义
                m = re.match(r"(\s*)(?:[\w:*&<>,\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{?", lines[j])
                if m and m.group(2) not in ("if", "for", "while", "switch", "else"):
                    func_name = m.group(2)
                    key = f"{func_name}@{file_path}"
                    if key not in seen:
                        seen.add(key)
                        evidence.append(
                            Evidence(
                                dimension="function",
                                file=file_path,
                                line=j + 1,
                                content=f"{func_name} (位于 {file_path})",
                                metadata={"type": "impl_function", "name": func_name},
                            )
                        )
                    break


def _code_dimension(api_name, test_files, export_root, e):
    """代码段维度：从 test 和 example 文件中提取调用代码。"""
    evidence = []
    test_calls = e.extract_test_calls(api_name, test_files)
    evidence.extend(test_calls)
    example_files = _find_example_files(api_name, export_root)
    for ef in example_files:
        example_calls = e.extract_test_calls(api_name, [ef])
        for call in example_calls:
            call.metadata["type"] = "example_file_call"
            evidence.append(call)
    return evidence


def _file_contains_api_call(file_path, api_name):
    """检查文件内容是否包含 API 调用。"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # 匹配 API 调用模式：asc_abs( 或 asc_abs<
        pattern = rf"\b{re.escape(api_name)}\s*[(<]"
        return bool(re.search(pattern, content))
    except Exception:
        return False


def _find_example_files(api_name, export_root):
    """在 export_root 的 examples 子目录下，通过内容匹配查找示例文件。"""
    examples = []
    examples_root = os.path.join(export_root, config.OUTPUT_EXAMPLES_DIR)

    if not os.path.isdir(examples_root):
        return []

    # 扫描所有示例文件，通过内容匹配查找
    for root, dirs, files in os.walk(examples_root):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(config.EXAMPLE_EXTENSIONS):
                full_path = os.path.join(root, f)
                if _file_contains_api_call(full_path, api_name):
                    rel = os.path.relpath(full_path, export_root)
                    examples.append(rel)

    return sorted(examples)[:10]
