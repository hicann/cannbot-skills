# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""跨平台文件搜索工具 — 用 Python 原生实现替代 grep/find 子进程调用。

对外接口：
    search_in_files()  — 替代 grep -rnE
    find_files()       — 替代 find <dir> -name <pattern>

返回格式与 grep/find 的 stdout 格式兼容，下游代码无需改动解析逻辑。
"""

import fnmatch
import os
import re


def search_in_files(
    root: str,
    patterns: list[str],
    dir_prefix: str = "",
    extensions: tuple[str, ...] = (".h", ".hpp", ".hxx"),
) -> list[str]:
    """跨平台替代 ``grep -rnE <pattern> <dir>``。

    遍历 root/dir_prefix 下所有 *extensions 文件，对每行尝试匹配
    patterns 中的任一正则表达式，返回 ``["rel_path:line_num:line_text", ...]``
    格式（兼容 grep -rnE 输出格式）。

    Args:
        root:        仓库根目录的绝对路径（对应 grep 的 cwd 参数）
        patterns:    正则表达式列表（对应 grep -E 的模式）
        dir_prefix:  在 root 下搜索的子目录前缀，如 "include" 或 "include/basic_api"
                     对应 grep 的搜索路径参数
        extensions:  只扫描这些后缀的文件，默认 (".h", ".hpp", ".hxx")

    Returns:
        列表，每个元素格式为 ``"rel_path:line_num:line_content"``，
        rel_path 相对于 root，使用 os.path 相对路径计算。
    """
    results = []
    compiled = [re.compile(p) for p in patterns]
    search_root = os.path.join(root, dir_prefix) if dir_prefix else root

    if not os.path.isdir(search_root):
        return results

    for dirpath, dirs, filenames in os.walk(search_root):
        # 排序确保子目录和文件的遍历顺序在所有平台上一致
        # (Windows NTFS 按字母序, Mac APFS / Linux ext4 按 inode 序, 不保证字母序)
        dirs.sort()
        for fname in sorted(filenames):
            if not fname.endswith(extensions):
                continue
            full_path = os.path.join(dirpath, fname)
            # 统一使用正斜杠，兼容 Windows/Linux/macOS
            rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        for cp in compiled:
                            if cp.search(line):
                                results.append(f"{rel_path}:{i}:{line.rstrip()}")
                                break  # 一行只报一次匹配
            except OSError:
                continue

    return results


def find_files(
    root: str,
    dir_prefix: str = "",
    name_pattern: str = "*.h",
) -> list[str]:
    """跨平台替代 ``find <dir> -name <pattern>``。

    遍历 root/dir_prefix 下所有匹配 name_pattern 的文件，
    返回相对路径列表（相对于 root）。

    Args:
        root:         仓库根目录的绝对路径
        dir_prefix:   在 root 下搜索的子目录前缀，如 "include"
        name_pattern: fnmatch 通配符模式，如 "*.h"

    Returns:
        列表，每个元素为相对路径（相对于 root）。
    """
    results = []
    search_root = os.path.join(root, dir_prefix) if dir_prefix else root

    if not os.path.isdir(search_root):
        return results

    for dirpath, dirs, filenames in os.walk(search_root):
        # 排序确保子目录和文件的遍历顺序在所有平台上一致
        dirs.sort()
        for fname in sorted(filenames):
            if fnmatch.fnmatch(fname, name_pattern):
                full_path = os.path.join(dirpath, fname)
                # 统一使用正斜杠，兼容 Windows/Linux/macOS
                rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
                results.append(rel_path)

    return results
