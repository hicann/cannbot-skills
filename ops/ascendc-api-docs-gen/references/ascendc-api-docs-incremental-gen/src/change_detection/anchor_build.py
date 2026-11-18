# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import shutil
from pathlib import Path

from anchor_adaptor import SUB_PART_TO_CHAPTER, _adapt_output, _ensure_anchor_fields, _find_doc_path
from rich.console import Console

from src.logger import debug, error, info, warning


def statistic_anchor_info(anchor_path: str):
    """统计 anchor 输出目录中各 JSON 文件的 anchor 数量和类型分布。

    通用版本：遍历 anchor_path 下所有 .json 文件，
    不依赖固定文件名列表（PathConfig），兼容 8 章节 sub_part 命名。
    """
    statistic_info = []
    if not os.path.isdir(anchor_path):
        warning(f"anchor 目录不存在: {anchor_path}")
        return statistic_info

    for filename in os.listdir(anchor_path):
        if not filename.endswith(".json"):
            continue
        sub_part = filename.replace(".json", "")
        filepath = os.path.join(anchor_path, filename)

        _info = {"doc_part": sub_part, "file": [], "function": [], "section": [], "note": []}

        with open(filepath, "r", encoding="utf-8") as f:
            anchor = json.load(f)

        for anchor_tuple_item in anchor.get("anchor_tuple", []):
            type_ = anchor_tuple_item.get("anchor_information", {}).get("type", "unknown")
            file_path = anchor_tuple_item.get("anchor_information", {}).get("level1", {}).get("file_path", "")
            if type_ in _info:
                _info[type_].append(file_path)
        statistic_info.append(_info)

    return statistic_info


def load_snapshot(
    operator_path: str,
    out_path: str,
    operator_name: str | None = None,
    doc_part_control=None,
    file_dir_list=None,
    old_tree_id_path=None,
):
    """调用 api_locator + 适配层，替代原 snapshot_construction 4步流水线。

    原 4 步: cann_doc_struct → code_map → build_tree_id(LLM) → AnchorPipeline
    新流程: api_locator.cli.run() → _adapt_output() → _ensure_anchor_fields()

    参数:
        operator_path: 算子源码根目录
        out_path: 输出目录
        operator_name: 算子名称
        doc_part_control: sub_part 列表（如 ['support', 'summary']），
                          通过 SUB_PART_TO_CHAPTER 转为中文章节名后传给
                          api_locator 的 selected_chapters，仅提取并生成
                          指定章节的 anchor，节省章节证据提取和 anchor 生成时间。
        file_dir_list: 变更文件目录列表（保留签名兼容，暂不生效）
        old_tree_id_path: 旧 tree_id 路径（保留签名兼容，暂不生效）
    """
    console = Console()

    # --- Step 1: 定位接口文档 ---
    console.print("    [dim]> 定位接口文档[/dim]")
    doc_path = _find_doc_path(operator_path, operator_name=operator_name)
    info(f"接口文档: {doc_path}")

    # --- Step 2: 调用 api_locator ---
    console.print("    [dim]> 执行 api_locator[/dim]")
    from api_locator.cli import run as api_locator_run

    # 将 sub_part 列表转换为中文章节名，传给 api_locator 的 selected_chapters 按章节筛选
    selected_chapters = None
    if doc_part_control:
        selected_chapters = [SUB_PART_TO_CHAPTER[sp] for sp in doc_part_control if sp in SUB_PART_TO_CHAPTER]
        unknown = set(doc_part_control) - set(SUB_PART_TO_CHAPTER.keys())
        if unknown:
            warning(f"doc_part_control 中有未知 sub_part（已跳过）: {unknown}")
        info(f"按章节筛选: {doc_part_control} → {selected_chapters}")

    # api_locator 需要 repo_root 搜索 include/impl 目录，默认路径不存在
    # operator_path 就是算子仓库根目录，直接作为 repo_root 传入
    export_root = api_locator_run(doc_path, repo_root=operator_path, selected_chapters=selected_chapters)
    info(f"api_locator 输出: {export_root}")

    # --- Step 3: 适配输出目录结构 ---
    console.print("    [dim]> 适配输出格式[/dim]")
    _adapt_output(export_root, out_path)

    # --- Step 4: 补齐 anchor 字段 ---
    anchor_dir = os.path.join(out_path, "anchor")
    for filename in os.listdir(anchor_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(anchor_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        _ensure_anchor_fields(data)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return statistic_anchor_info(anchor_dir)


def anchor_rebuild(
    operator_name, operator_path, data_path, filter_list=None, file_dir_list=None, old_tree_id_path=None
):
    operator_storage_path = f"{data_path}/operator_storage/{operator_name}"

    # operator_version_split(operator_path, operator_storage_path)
    import shutil

    if os.path.exists(operator_storage_path):
        shutil.rmtree(operator_storage_path)
    shutil.copytree(operator_path, operator_storage_path)

    info(operator_storage_path)

    echo_info = load_snapshot(
        f"{operator_storage_path}",
        f"{data_path}",
        operator_name=operator_name,
        doc_part_control=filter_list,
        file_dir_list=file_dir_list,
        old_tree_id_path=old_tree_id_path,
    )
    return echo_info


def anchor_rebuild_cli(operator_path, data_path, filter_list):
    operator_storage_path = f"{data_path}/operator_storage"

    # operator_version_split(operator_path, operator_storage_path)
    import shutil

    if os.path.exists(operator_storage_path):
        shutil.rmtree(operator_storage_path)
    shutil.copytree(operator_path, operator_storage_path)

    info(operator_storage_path)

    # operator_name 从路径 basename 推导（anchor_rebuild_cli 无显式 operator_name 参数）
    operator_name = os.path.basename(operator_path.rstrip("/"))
    echo_info = load_snapshot(
        f"{operator_storage_path}", f"{data_path}", operator_name=operator_name, doc_part_control=filter_list
    )
    return echo_info
