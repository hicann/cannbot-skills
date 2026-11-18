#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""一键脚本：输入变更前后的代码目录 → 输出受影响的 API。

用法:
    # 两个目录对比（最常用）
    python pipeline.py --old /path/to/old-repo --new /path/to/new-repo

    # 使用已有的索引文件（跳过构建，秒级）
    python pipeline.py --old /path/to/old-repo --new /path/to/new-repo \
        --index /path/to/api_reverse_index.json

    # git diff 模式
    python pipeline.py --git-range HEAD~3

    # 使用内置测试数据
    python pipeline.py --test-data

输出:
    变更文件列表
    受影响的 API 列表（含每个 API 被哪些变更文件影响）
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_PATH_CONFIG, REPO_ROOT, PathConfig

from api_change_detect import (
    _rp,
    build_reverse_index,
    compile_api_name_pattern,
    extract_api_name_from_doc,
    extract_apis_from_content,
    find_affected_apis,
    find_affected_by_pattern,
    get_commit_id,
    incremental_update_index,
    load_index,
    path_matches_apis,
    save_index,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def find_changed_files_between_dirs(old_dir: str, new_dir: str, path_config=None) -> list:
    """对比两个目录，找出变更的文件。

    返回相对于目录根的路径列表。
    """
    if path_config is None:
        path_config = DEFAULT_PATH_CONFIG or PathConfig()

    old_dir = os.path.abspath(old_dir)
    new_dir = os.path.abspath(new_dir)

    exclude_dirs = tuple(path_config.walk_exclude_dirs)
    exclude_dir_suffixes = tuple(path_config.walk_exclude_dir_suffixes)
    exclude_file_suffixes = tuple(path_config.changed_file_exclude_suffixes)
    file_prefixes = tuple(path_config.changed_file_prefixes)

    def _should_exclude_dir(d):
        return d in exclude_dirs or (exclude_dir_suffixes and d.endswith(exclude_dir_suffixes))

    def _should_include_file(rel_path):
        return rel_path.startswith(file_prefixes) if file_prefixes else True

    old_files = {}
    for root, dirs, files in os.walk(old_dir):
        dirs[:] = sorted(d for d in dirs if not _should_exclude_dir(d))
        for f in sorted(files):
            if f.endswith(exclude_file_suffixes):
                continue
            full = os.path.join(root, f)
            rel = _rp(full, old_dir)
            if not _should_include_file(rel):
                continue
            try:
                with open(full, "rb") as fh:
                    old_files[rel] = hashlib.md5(fh.read(), usedforsecurity=False).hexdigest()
            except (FileNotFoundError, PermissionError):
                continue

    changed = []
    new_files = {}
    for root, dirs, files in os.walk(new_dir):
        dirs[:] = sorted(d for d in dirs if not _should_exclude_dir(d))
        for f in sorted(files):
            if f.endswith(exclude_file_suffixes):
                continue
            full = os.path.join(root, f)
            rel = _rp(full, new_dir)
            if not _should_include_file(rel):
                continue
            try:
                with open(full, "rb") as fh:
                    new_hash = hashlib.md5(fh.read(), usedforsecurity=False).hexdigest()
            except (FileNotFoundError, PermissionError):
                continue
            new_files[rel] = new_hash

            if rel not in old_files or old_files[rel] != new_hash:
                changed.append(rel)

    for rel in old_files:
        if rel not in new_files:
            changed.append(rel)

    return sorted(set(changed))


def find_changed_files_git(diff_range: str, repo_root: str) -> list:
    """通过 git diff 获取变更文件。"""
    if ".." in diff_range:
        cmd = ["git", "diff", "--name-only", diff_range]
    elif " " in diff_range:
        parts = diff_range.split()
        cmd = ["git", "diff", "--name-only"] + parts
    else:
        cmd = ["git", "diff", "--name-only", diff_range]

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"git 命令失败: {' '.join(cmd)}\n{result.stderr}")

    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def run_pipeline(
    old_dir=None, new_dir=None, index_path=None, git_range=None, repo_root=None, path_config=None, index_output_dir=None
):
    """完整管道：构建/增量更新索引 → 找变更 → 查 API。

    Args:
        old_dir: 旧代码目录
        new_dir: 新代码目录
        index_path: 已有的索引文件路径（跳过构建或走增量）
        git_range: git diff 范围
        repo_root: 仓库根目录（git 模式使用）
        path_config: PathConfig 路径配置
        index_output_dir: 索引输出目录

    Returns:
        (changed_files, affected_apis_dict)
    """
    if path_config is None:
        path_config = DEFAULT_PATH_CONFIG or PathConfig()

    t0 = time.time()

    # 索引输出目录
    save_dir = index_output_dir or OUTPUT_DIR

    # Step 1: 构建或增量更新索引
    index = None
    api_file_map = None
    summary = None

    if index_path and os.path.exists(index_path):
        print(f"[1/3] 加载旧索引: {index_path}")
        old_index_raw = load_index(index_path)
        old_api_file_map = {}
        saved_fingerprint = old_index_raw.pop("__fingerprint__", None)
        for f, apis in old_index_raw.items():
            for api in apis:
                old_api_file_map.setdefault(api, set()).add(f)

        # commit ID 校验
        old_commit = get_commit_id(os.path.abspath(old_dir)) if old_dir else None
        if saved_fingerprint and old_commit and saved_fingerprint == old_commit:
            print(f"  commit 匹配: {saved_fingerprint[:10]}")
            if old_dir and new_dir:
                # 增量更新
                print(f"  增量更新索引: {old_dir} → {new_dir}")
                t_build = time.time()
                index, api_file_map, summary = incremental_update_index(
                    os.path.abspath(old_dir), os.path.abspath(new_dir), old_index_raw, old_api_file_map, path_config
                )
                save_index(index, save_dir)
                print(f"  耗时: {time.time() - t_build:.1f}s")
            else:
                print(f"  使用已有索引")
                index = old_index_raw
                api_file_map = old_api_file_map
                summary = {"mode": "from_index"}
        else:
            reason = "commit 不匹配" if saved_fingerprint else "索引无 commit"
            print(
                f"  {reason}: 索引={saved_fingerprint[:10] if saved_fingerprint else 'None'}, "
                f"实际={old_commit[:10] if old_commit and old_commit != 'unknown' else old_commit}"
            )
            print(f"  降级为全量构建: {new_dir}")
            t_build = time.time()
            index, api_file_map = build_reverse_index(os.path.abspath(new_dir), path_config)
            save_index(index, save_dir)
            print(f"  构建耗时: {time.time() - t_build:.0f}s")
            summary = {"mode": "full_build"}
    else:
        # 无索引 → 全量构建
        if not old_dir:
            raise ValueError("需要提供 --old 旧代码目录或 --index 已有索引")
        print(f"[1/3] 全量构建索引: {old_dir}")
        t_build = time.time()
        index, api_file_map = build_reverse_index(os.path.abspath(old_dir), path_config)
        save_index(index, save_dir)
        print(f"  构建耗时: {time.time() - t_build:.0f}s")
        summary = {"mode": "full_build"}

    # Step 2: 找出变更文件
    print(f"\n[2/3] 查找变更文件")
    t_diff = time.time()

    if old_dir and new_dir:
        changed_files = find_changed_files_between_dirs(old_dir, new_dir, path_config)
        print(f"  对比: {old_dir} → {new_dir}")
    elif git_range:
        changed_files = find_changed_files_git(git_range, repo_root)
        print(f"  git diff: {git_range}")
    else:
        raise ValueError("需要提供 --new 新代码目录 或 --git-range")

    print(f"  变更文件: {len(changed_files)} 个")
    for f in changed_files:
        print(f"    {f}")
    print(f"  查找耗时: {time.time() - t_diff:.1f}s")

    # Step 3: 查索引找受影响 API（只查代码文件变更，过滤掉 docs/figures 等）
    print(f"\n[3/3] 查询受影响 API")
    t_query = time.time()

    # 只保留代码目录的变更用于 API 判定
    api_prefixes = tuple(path_config.api_change_prefixes)
    code_changes = [f for f in changed_files if f.startswith(api_prefixes)]
    doc_changes = [f for f in changed_files if not f.startswith(api_prefixes)]
    print(f"  代码变更: {len(code_changes)} 个（用于 API 判定）")
    print(f"  文档/图片变更: {len(doc_changes)} 个（不计入 API 变更）")

    exact_affected = find_affected_apis(code_changes, index)
    pattern_affected = find_affected_by_pattern(code_changes, api_file_map)

    all_affected = {}
    for api in exact_affected:
        all_affected[api] = list(set(list(all_affected.get(api, [])) + list(exact_affected[api])))
    for api in pattern_affected:
        if api not in all_affected:
            all_affected[api] = list(pattern_affected[api])
        else:
            all_affected[api] = list(set(list(all_affected[api]) + list(pattern_affected[api])))

    print(f"  受影响 API: {len(all_affected)} 个")
    for api_name in sorted(all_affected.keys()):
        files = sorted(all_affected[api_name])
        print(f"    {api_name}")
        for f in files:
            print(f"      ← {f}")

    print(f"  查询耗时: {time.time() - t_query:.1f}s")
    print(f"\n总耗时: {time.time() - t0:.1f}s")
    print(f"共 {len(all_affected)} 个 API 受影响")

    return changed_files, all_affected


# 索引文件默认名称（保存在代码目录下）
DEFAULT_INDEX_FILENAME = "api_reverse_index.json"


def detect_api_changes(old_dir: str, new_dir: str, path_config=None, verbose: bool = True) -> dict:
    """检测代码变更影响的 API（智能模式）。

    自动选择全量构建或增量更新，索引文件保存在代码目录下：

    - 如果 old_dir 下已存在索引 → 增量更新，新索引保存到 new_dir
    - 如果 old_dir 下不存在索引 → 全量构建，索引保存到 old_dir

    Args:
        old_dir: 旧代码目录路径
        new_dir: 新代码目录路径
        path_config: PathConfig 路径配置（可选，默认使用 DEFAULT_PATH_CONFIG）
        verbose: 是否输出详细日志（默认 True）

    Returns:
        dict: 受影响的 API 及其文档路径，格式：
            {
                "Add": "docs/api/Add.md",
                "Sub": "docs/api/Sub.md",
                ...
            }

    Example:
        >>> from pipeline import detect_api_changes

        # 第一次使用（old_dir 无索引）
        >>> affected = detect_api_changes("/path/to/v1.0", "/path/to/v1.1")
        # 索引保存在: /path/to/v1.0/api_reverse_index.json
        受影响的 API: ['Add', 'Sub']

        # 第二次使用（old_dir 有索引）
        >>> affected = detect_api_changes("/path/to/v1.1", "/path/to/v1.2")
        # 使用增量更新，新索引保存在: /path/to/v1.2/api_reverse_index.json
        受影响的 API: ['Mul']

    Workflow:
        ┌─────────────────────────────────────────────────────────┐
        │  old_dir/api_reverse_index.json 存在？                   │
        ├─────────────────────────────────────────────────────────┤
        │  YES → 增量更新                                          │
        │       1. 加载 old_dir 的索引                              │
        │       2. diff(old_dir, new_dir) 找变更文件               │
        │       3. 增量更新索引                                     │
        │       4. 新索引保存到 new_dir/api_reverse_index.json     │
        ├─────────────────────────────────────────────────────────┤
        │  NO  → 全量构建                                          │
        │       1. 基于 old_dir 构建完整索引                        │
        │       2. 索引保存到 old_dir/api_reverse_index.json       │
        │       3. diff(old_dir, new_dir) 找变更文件               │
        │       4. 查索引找受影响 API                               │
        └─────────────────────────────────────────────────────────┘

    Note:
        - 索引文件名默认为 api_reverse_index.json
        - 索引保存在代码目录下，方便后续增量更新
        - 增量更新要求索引的 commit ID 与 old_dir 匹配
    """
    old_dir = os.path.abspath(old_dir)
    new_dir = os.path.abspath(new_dir)

    # 紮引文件路径
    old_index_path = os.path.join(old_dir, DEFAULT_INDEX_FILENAME)
    new_index_path = os.path.join(new_dir, DEFAULT_INDEX_FILENAME)

    # 判断使用哪种模式：old_dir 下是否已有索引
    use_incremental = os.path.exists(old_index_path)

    if verbose:
        if use_incremental:
            print(f"[智能模式] 使用增量更新")
            print(f"  加载索引: {old_index_path}")
            print(f"  保存索引: {new_index_path}")
        else:
            print(f"[智能模式] 使用全量构建")
            print(f"  构建索引: {old_dir}")
            print(f"  保存索引: {old_index_path}")

    # Step 1: 构建或增量更新索引
    if use_incremental:
        # 增量更新模式：加载 old_dir 的索引，更新后保存到 new_dir
        if verbose:
            print(f"\n[1/3] 增量更新索引")

        # 调用 run_pipeline，指定索引路径和输出目录
        if not verbose:
            import io
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                _, affected_apis = run_pipeline(
                    old_dir=old_dir,
                    new_dir=new_dir,
                    index_path=old_index_path,
                    path_config=path_config,
                    index_output_dir=new_dir,  # 新索引保存到 new_dir
                )
        else:
            _, affected_apis = run_pipeline(
                old_dir=old_dir,
                new_dir=new_dir,
                index_path=old_index_path,
                path_config=path_config,
                index_output_dir=new_dir,  # 新索引保存到 new_dir
            )

        # 使用 new_dir 下的新索引来提取文档路径
        actual_index_path = new_index_path
    else:
        # 全量构建模式：基于 old_dir 构建，保存到 old_dir
        if verbose:
            print(f"\n[1/3] 全量构建索引")

        if not verbose:
            import io
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                _, affected_apis = run_pipeline(
                    old_dir=old_dir,
                    new_dir=new_dir,
                    index_path=None,  # 无索引，全量构建
                    path_config=path_config,
                    index_output_dir=old_dir,  # 索引保存到 old_dir
                )
        else:
            _, affected_apis = run_pipeline(
                old_dir=old_dir,
                new_dir=new_dir,
                index_path=None,
                path_config=path_config,
                index_output_dir=old_dir,
            )

        # 使用 old_dir 下的新索引来提取文档路径
        actual_index_path = old_index_path

    # Step 2: 从索引提取文档路径
    if not os.path.exists(actual_index_path):
        if verbose:
            print(f"警告: 索引文件不存在: {actual_index_path}")
        return {api: None for api in affected_apis.keys()}

    # 加载索引并重建 api_file_map
    raw_index = load_index(actual_index_path)
    raw_index.pop("__fingerprint__", None)

    api_file_map = {}
    for file_path, apis in raw_index.items():
        for api in apis:
            if api not in api_file_map:
                api_file_map[api] = set()
            api_file_map[api].add(file_path)

    # 为每个受影响的 API 提取文档路径
    result = {}
    for api_name in affected_apis.keys():
        if api_name in api_file_map:
            files = api_file_map[api_name]
            # 筛选 docs/ 目录下的 .md 文件
            doc_files = [f for f in files if f.startswith("docs/") and f.endswith(".md")]
            if doc_files:
                result[api_name] = sorted(doc_files)[0]
            else:
                result[api_name] = None
        else:
            result[api_name] = None

    if verbose:
        print(f"\n[结果] 受影响的 API: {len(result)} 个")
        for api, doc_path in sorted(result.items()):
            if doc_path:
                print(f"  {api}: {doc_path}")

    return result


def detect_api_changes_incremental(
    old_dir: str, new_dir: str, index_path: str, path_config=None, verbose: bool = True, index_output_dir: str = None
) -> dict:
    """检测代码变更影响的 API（增量更新模式）。

    使用已有索引文件进行增量检测，速度更快（约 5-8 秒）。
    如果索引文件的 commit ID 与旧代码目录不匹配，会自动降级为全量构建。

    Args:
        old_dir: 旧代码目录路径
        new_dir: 新代码目录路径
        index_path: 已有的反向索引文件路径
        path_config: PathConfig 路径配置（可选，默认使用 DEFAULT_PATH_CONFIG）
        verbose: 是否输出详细日志（默认 True）
        index_output_dir: 索引输出目录（可选，默认 output/）

    Returns:
        dict: 受影响的 API 及其文档路径，格式：
            {
                "Add": "docs/api/Add.md",
                "Sub": "docs/api/Sub.md",
                ...
            }

    Raises:
        FileNotFoundError: 索引文件不存在时会自动降级为全量构建

    Example:
        >>> from pipeline import detect_api_changes_incremental
        >>> affected_apis = detect_api_changes_incremental(
        ...     old_dir="/path/to/old-repo",
        ...     new_dir="/path/to/new-repo",
        ...     index_path="output/api_reverse_index.json"
        ... )
        >>> print(f"受影响的 API: {list(affected_apis.keys())}")
        >>> for api, doc_path in affected_apis.items():
        ...     print(f"{api}: {doc_path}")
        Add: docs/api/Add.md

    Note:
        - 索引文件必须包含 __fingerprint__（commit ID）
        - 旧代码目录的 HEAD commit 必须与索引指纹匹配才能增量更新
        - 不匹配时自动降级为全量构建（约 90 秒）
    """
    # 如果不需要日志，临时抑制 stdout
    if not verbose:
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            _, affected_apis = run_pipeline(
                old_dir=old_dir,
                new_dir=new_dir,
                index_path=index_path,
                path_config=path_config,
                index_output_dir=index_output_dir,
            )
    else:
        _, affected_apis = run_pipeline(
            old_dir=old_dir,
            new_dir=new_dir,
            index_path=index_path,
            path_config=path_config,
            index_output_dir=index_output_dir,
        )

    # 转换为文档路径映射：{API名称: 文档路径}
    # 使用传入的索引文件
    if not os.path.exists(index_path):
        print(f"警告: 索引文件不存在: {index_path}")
        return {api: None for api in affected_apis.keys()}

    # 加载索引并重建 api_file_map
    raw_index = load_index(index_path)
    raw_index.pop("__fingerprint__", None)

    api_file_map = {}
    for file_path, apis in raw_index.items():
        for api in apis:
            if api not in api_file_map:
                api_file_map[api] = set()
            api_file_map[api].add(file_path)

    # 为每个受影响的 API 提取文档路径
    result = {}
    for api_name in affected_apis.keys():
        if api_name in api_file_map:
            files = api_file_map[api_name]
            # 筛选 docs/ 目录下的 .md 文件
            doc_files = [f for f in files if f.startswith("docs/") and f.endswith(".md")]
            if doc_files:
                result[api_name] = sorted(doc_files)[0]
            else:
                result[api_name] = None
        else:
            result[api_name] = None

    return result


def get_api_doc_path(api_name: str, index_path: str = None) -> str | None:
    """获取某个 API 的文档路径。

    从反向索引中查找 API 对应的文档路径（docs/api/ 下的 .md 文件）。

    Args:
        api_name: API 名称（如 "Add", "Sub", "operator+"）
        index_path: 索引文件路径（可选，默认使用 output/api_reverse_index.json）

    Returns:
        str | None: 文档路径（相对路径，如 "docs/api/Add.md"），找不到返回 None

    Example:
        >>> from pipeline import get_api_doc_path
        >>> doc_path = get_api_doc_path("Add", "output/api_reverse_index.json")
        >>> print(doc_path)
        docs/api/Add.md

        >>> doc_path = get_api_doc_path("operator+")
        >>> print(doc_path)
        docs/api/operator.md
    """
    # 加载索引
    if index_path is None:
        index_path = os.path.join(OUTPUT_DIR, "api_reverse_index.json")

    if not os.path.exists(index_path):
        print(f"错误: 索引文件不存在: {index_path}")
        return None

    # 加载索引并重建 api_file_map
    raw_index = load_index(index_path)
    raw_index.pop("__fingerprint__", None)  # 移除指纹字段

    api_file_map = {}
    for file_path, apis in raw_index.items():
        for api in apis:
            if api not in api_file_map:
                api_file_map[api] = set()
            api_file_map[api].add(file_path)

    # 查找 API 的文档路径
    if api_name not in api_file_map:
        print(f"未找到 API: {api_name}")
        return None

    files = api_file_map[api_name]

    # 筛选文档路径：docs/ 目录下的 .md 文件
    doc_files = [f for f in files if f.startswith("docs/") and f.endswith(".md")]

    if not doc_files:
        print(f"API {api_name} 未找到关联文档")
        return None

    # 返回第一个文档（通常只有一个主文档）
    return sorted(doc_files)[0]


def get_all_api_doc_paths(index_path: str = None) -> dict:
    """获取所有 API 的文档路径映射。

    Args:
        index_path: 索引文件路径（可选，默认使用 output/api_reverse_index.json）

    Returns:
        dict: {api_name: doc_path} 映射，如：
            {
                "Add": "docs/api/Add.md",
                "Sub": "docs/api/Sub.md",
                ...
            }

    Example:
        >>> from pipeline import get_all_api_doc_paths
        >>> doc_map = get_all_api_doc_paths()
        >>> print(f"共 {len(doc_map)} 个 API 文档")
    """
    # 加载索引
    if index_path is None:
        index_path = os.path.join(OUTPUT_DIR, "api_reverse_index.json")

    if not os.path.exists(index_path):
        print(f"错误: 索引文件不存在: {index_path}")
        return {}

    raw_index = load_index(index_path)
    raw_index.pop("__fingerprint__", None)

    api_file_map = {}
    for file_path, apis in raw_index.items():
        for api in apis:
            if api not in api_file_map:
                api_file_map[api] = set()
            api_file_map[api].add(file_path)

    # 为每个 API 提取文档路径
    doc_map = {}
    for api_name, files in api_file_map.items():
        doc_files = [f for f in files if f.startswith("docs/") and f.endswith(".md")]
        if doc_files:
            doc_map[api_name] = sorted(doc_files)[0]

    return doc_map


def _collect_api_names_from_docs(repo_dir: str, path_config) -> tuple[set, dict]:
    """从 docs/ 目录收集所有 API 名，并构建 api_name → doc_rel 映射。"""
    import fnmatch

    doc_dir = os.path.join(repo_dir, path_config.doc_dir)
    if not os.path.isdir(doc_dir):
        return set(), {}
    docs = []
    if path_config.doc_recursive:
        import glob as _glob

        ext = path_config.doc_extensions[0] if path_config.doc_extensions else ".md"
        docs = _glob.glob(os.path.join(doc_dir, "**", "*" + ext), recursive=True)
    exclude = path_config.doc_exclude_patterns
    docs = [d for d in docs if not any(fnmatch.fnmatch(os.path.basename(d), pat) for pat in exclude)]

    api_names = set()
    api_to_doc = {}
    for d in docs:
        name = extract_api_name_from_doc(d)
        if not name:
            continue
        api_names.add(name)
        rel = _rp(d, repo_dir)
        if name not in api_to_doc:
            api_to_doc[name] = rel
    return api_names, api_to_doc


def _find_changed_line_numbers(old_lines: list, new_lines: list) -> tuple[set, set]:
    """找出变更的行号。

    使用 difflib.SequenceMatcher 对比新旧文件，返回两组行号：
    - new_changed: 新文件中变更的行号（replace/insert 对应的新行，从 0 开始）
    - old_deleted: 旧文件中被删除/替换的行号（delete/replace 对应的旧行，从 0 开始）
    """
    import difflib

    new_changed = set()
    old_deleted = set()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert", "delete"):
            # 新文件中变更的行
            for line_num in range(j1, j2):
                new_changed.add(line_num)
        # 旧文件中被删除/替换的行
        if tag in ("replace", "delete"):
            for line_num in range(i1, i2):
                old_deleted.add(line_num)
    return new_changed, old_deleted


def detect_api_changes_lightweight(old_dir: str, new_dir: str, path_config=None, changed_files=None) -> dict:
    """轻量版 API 变更检测：不建全量索引，只对改动文件做内容扫描 + 差集。

    相比 detect_api_changes：
      - 速度：从 ~70-100s 降到 ~5-10s（只扫改动文件，不扫全仓）
      - 准确度：用"对称差集"过滤掉两边都存在的 API 调用（如 asc_init/asc_sync）
      - 召回：路径匹配 + 内容匹配互补，不会漏掉 asc_abs 这类"路径名不含 API 名
              但内容调用了该 API"的情况

    Args:
        old_dir: 旧代码目录（"before"）
        new_dir: 新代码目录（"now"）
        path_config: 路径配置（可选）
        changed_files: 预先提供的改动文件相对路径列表（正斜杠）。传入时跳过
            find_changed_files_between_dirs 的全量 os.walk + MD5 对比（大仓库上
            需读取两份完整工作区，耗时可达分钟级）。调用方可用
            `git diff --name-only <base> <head>` 毫秒级获得同等内容。

    Returns:
        {api_name: doc_rel_path} — 受影响的 API 及其文档路径
    """
    if path_config is None:
        path_config = DEFAULT_PATH_CONFIG or PathConfig()

    # 1. 从 new_dir 的 docs 拿 API 名列表（快）
    api_names, api_to_doc = _collect_api_names_from_docs(new_dir, path_config)
    if not api_names:
        return {}
    pattern = compile_api_name_pattern(api_names)

    # 2. diff 找出改动文件（调用方已提供时直接复用，跳过全量 MD5 对比）
    if changed_files is None:
        changed_files = find_changed_files_between_dirs(old_dir, new_dir, path_config)
    # 只关心代码目录（include/impl/tests/examples）
    api_prefixes = tuple(path_config.api_change_prefixes)
    code_changes = [f for f in changed_files if f.startswith(api_prefixes)]

    affected = set()

    for rel in code_changes:
        old_path = os.path.join(old_dir, rel)
        new_path = os.path.join(new_dir, rel)
        old_exists = os.path.isfile(old_path)
        new_exists = os.path.isfile(new_path)

        # 路径匹配：文件名/目录名暗示的 API（无论哪边存在都算）
        path_match = path_matches_apis(rel, api_names)
        affected.update(path_match)

        # 内容扫描：只在文本文件上做（跳过二进制）
        old_content = _safe_read(old_path) if old_exists else None
        new_content = _safe_read(new_path) if new_exists else None

        if old_content is None and new_content is None:
            continue  # 二进制或不可读，仅用 path_match

        # 找出变更行号，提取变更行 ±100 行范围内的 API
        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines() if new_content else []
        new_changed, old_deleted = _find_changed_line_numbers(old_lines, new_lines)

        # 从变更行 ±100 行范围内提取 API
        context_range = 100
        apis_in_context = set()
        # 新文件变更行上下文（检测新增/修改的 API 调用）
        for line_num in new_changed:
            start = max(0, line_num - context_range)
            end = min(len(new_lines), line_num + context_range + 1)
            context_lines = new_lines[start:end]
            apis_in_context.update(extract_apis_from_content("\n".join(context_lines), pattern))
        # 旧文件删除行上下文（检测被删除的 API 调用）
        for line_num in old_deleted:
            start = max(0, line_num - context_range)
            end = min(len(old_lines), line_num + context_range + 1)
            context_lines = old_lines[start:end]
            apis_in_context.update(extract_apis_from_content("\n".join(context_lines), pattern))

        affected.update(apis_in_context)

    # 过滤掉没有文档的 API
    return {api: api_to_doc[api] for api in affected if api in api_to_doc}


def _safe_read(path: str):
    """读取文件文本内容，二进制/不可读返回 None。"""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
        if b"\x00" in head:
            return None  # 二进制
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description="API 变更检测：输入变更前后代码，输出受影响的 API")
    parser.add_argument("--old", help="旧代码目录路径")
    parser.add_argument("--new", help="新代码目录路径")
    parser.add_argument("--index", help="已有的反向索引文件路径（跳过构建）")
    parser.add_argument("--git-range", help="git diff 范围，如 HEAD~3 或 abc123 def456")
    parser.add_argument("--repo", help="仓库根目录（git 模式使用，默认使用 config.py 中的配置）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    parser.add_argument("--output", "-o", help="JSON 结果输出路径，默认不写文件")
    parser.add_argument("--index-output", help="索引文件输出目录，默认 output/")
    args = parser.parse_args()

    if args.repo is None:
        args.repo = REPO_ROOT

    if not args.old and not args.new and not args.git_range:
        args.git_range = "HEAD"

    changed_files, result = run_pipeline(
        old_dir=args.old,
        new_dir=args.new,
        index_path=args.index,
        git_range=args.git_range,
        repo_root=args.repo,
        index_output_dir=args.index_output,
    )

    output_data = {
        "changed_files": changed_files,
        "affected_apis": result,
    }

    if args.json:
        print("\n--- JSON Output ---")
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已写入: {args.output}")


if __name__ == "__main__":
    main()
