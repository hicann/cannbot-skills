# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Entry script for anchor redirect mapping."""

import json
import os
import sys
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.logger import debug, error, info, warning

from .config import MappingContext
from .three_level_mapper import ThreeLevelMapper

# 路径归一化（原截断逻辑已不适用新路径格式）
_truncate_keywords = []


def _truncate_path(path: str | list) -> str | list:
    """路径归一化：统一使用正斜杠，兼容 Windows/Linux/macOS。

    注意：原截断逻辑（op_api/op_host/op_kernel/op_graph）已不适用新路径格式，
    现在路径格式与 api_locator 一致，都是纯相对路径。

    兼容 str 和 list 两种输入格式：
    - str: 直接归一化返回
    - list: 归一化每个元素，返回第一个元素（用于 unchanged_file_paths 集合匹配）
    """
    if isinstance(path, list):
        # 多文件场景：归一化所有路径
        normalized = [p.replace("\\", "/") for p in path]
        return normalized[0] if normalized else ""  # 返回第一个用于集合匹配
    else:
        return path.replace("\\", "/")


def process(
    anchor_path: str,
    origin_map_base: str,
    change_map_base: str,
    change_repo_path: str,
    output_dir: str,
    verbose: bool = False,
    changed_files: set = None,
    unchanged_file_paths: set = None,
) -> dict:
    """
    从单个 anchor JSON 文件中处理所有 type="function" 和 type="file" 的 anchor。
    遍历每个 file-type anchor，对应的变更后文件路径（来自 file 阶段的更新），
    调用 ThreeLevelMapper 进行函数级重定向。

    参数:
        anchor_path: anchor JSON 文件路径（四个阶段共用）
        origin_map_base: 变更前 code_map 目录
        change_map_base: 变更后 code_map 目录
        change_repo_path: 变更后算子仓库路径
        output_dir: 输出目录
        verbose: 是否输出详细日志
        changed_files: 变更文件路径集合（用于候选范围缩小）
        unchanged_file_paths: 未变更文件路径集合（用于跳过 anchor）

    返回:
        更新后的 anchor JSON 数据（含 processed/skipped/remaining/status 统计字段），
        由调用方负责写入文件
    """
    from .utils import load_json

    # Load the anchor JSON once
    try:
        anchor_data = load_json(anchor_path)
        anchor_path_name = Path(anchor_path).name

    except Exception as e:
        # print(f"  [func 阶段] 加载 anchor JSON 失败: {e}")
        anchor_data = {
            "anchor_tuple": [],
            "processed": 0,
            "skipped": 0,
            "remaining": 0,
            "status": f"load error: {e}",
        }
        return anchor_data

    # Collect all file-type anchors to process

    function_anchors = []
    for item in anchor_data.get("anchor_tuple", []):
        anchor_info = item.get("anchor_information", {})
        if anchor_info.get("type") == "function":
            level1 = anchor_info.get("level1", {})
            file_path = level1.get("file_path", "")
            function_name = level1.get("function_name", "")

            # Level 2: 如果 anchor 的源文件未变更，跳过该 anchor
            if unchanged_file_paths is not None and file_path:
                truncated = _truncate_path(file_path)
                if truncated in unchanged_file_paths:
                    debug(f"[func] 跳过未变更 anchor: {function_name} @ {file_path}")
                    continue

            if file_path:
                # 兼容 file_path 为 list 或 str：file_name 取第一个路径的文件名
                if isinstance(file_path, list):
                    first_path = file_path[0]
                    file_name = os.path.basename(first_path) if first_path else ""
                else:
                    file_name = os.path.basename(file_path)

                function_anchors.append(
                    {"item": item, "file_path": file_path, "file_name": file_name, "function_name": function_name}
                )
    skipped_by_unchanged = (
        len(anchor_data.get("anchor_tuple", [])) - len(function_anchors) if unchanged_file_paths else 0
    )
    info(
        f"找到 {len(function_anchors)} 个 function 类型的 anchor（已跳过 {skipped_by_unchanged} 个未变更文件的 anchor）"
    )
    processed = 0
    skipped = 0
    new_anchors = []

    info(f"[func 阶段] 开始处理 {len(function_anchors)} 个 function anchor")

    anchor_label = Path(anchor_path).stem
    with Progress(
        TextColumn("    "),
        TextColumn(f"[bold blue]函数重定向匹配 ({anchor_label})"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TextColumn("("),
        TimeRemainingColumn(),
        TextColumn(","),
        TimeElapsedColumn(),
        TextColumn(")"),
        disable=not sys.stdout.isatty(),
    ) as progress:
        task_id = progress.add_task("函数重定向", total=len(function_anchors))

        for fa in function_anchors:
            progress.update(task_id, advance=1)

            # 兼容 file_path 为 list 或 str 两种格式
            raw_path = fa["file_path"]
            if isinstance(raw_path, list):
                # 多文件场景：取第一个路径（通常是最主要的实现文件）
                file_path = Path(raw_path[0])
            else:
                file_path = Path(raw_path)
            file_name = fa["file_name"]

            if not file_name:
                file_name = os.path.basename(file_path)

            # Normalize ops_name from file path
            ops_name = file_name.replace(".cpp", "").replace(".h", "")
            ops_name = ops_name.replace("aclnn_", "").replace("aclnn", "")

            if verbose:
                pass
                # # print(f"  [func] 处理文件: {file_path} (ops={ops_name})")

            # Skip if file doesn't exist on disk (构建绝对路径检查)
            full_file_path = os.path.join(change_repo_path, str(file_path))
            if not os.path.exists(full_file_path):
                if verbose:
                    pass
                    # # print(f"  [func] 文件不存在，跳过: {file_path}")
                skipped += 1
                continue

            # Create a temp copy of the anchor JSON for the mapper
            # (ThreeLevelMapper loads from disk and saves to output_dir)
            # import tempfile
            # output_file = os.path.join(output_dir, 'func_result.json')
            # with open(output_file, "w", encoding='utf-8') as fp:
            #     json.dump(anchor_data, fp, ensure_ascii=False, indent=2)

            try:
                context = MappingContext(
                    ops_name=ops_name,
                    file_path=file_path,
                    code_map_dir=change_map_base,
                    new_operator_path=change_repo_path,
                    anchor_path=anchor_path,
                    output_dir=output_dir,
                )

                mapper = ThreeLevelMapper(context, verbose=verbose)
                result = mapper.map()
                new_anchors.append(result.update_anchor)

                # if result.update_anchor:
                #         # save files
                #         output_func_dir = os.path.join(output_dir, f"func_{anchor_path_name}")
                #         with open(output_func_dir, "w", encoding="utf-8") as f:
                #             json.dump(result, f, ensure_ascii=False, indent=2)

                if result.matched:
                    processed += 1
                    # # print("ori_file_name:", result.old_file_path)
                    # # print("new_file_name:", result.new_file_path)
                    # # print("ori_functions:", result.old_function_names)
                    # # print("new_functions:", result.new_function_names)
                    # # print("update_anchors:", result.update_anchor)

                else:
                    if verbose:
                        warning(f"  [func] 映射失败 ({ops_name}): {result.message}")

            except Exception as e:
                if verbose:
                    # print(f"  [func] 处理异常 ({file_path}): {e}")
                    import traceback

                    traceback.print_exc()

                    # 单独获取 当前异常的 文件名、行号
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    fname = exc_tb.tb_frame.f_code.co_filename  # 报错文件
                    line_no = exc_tb.tb_lineno  # 报错行号

                    # print(f"\n报错文件：{fname}")
                    # print(f"报错行号：{line_no}")
                    # print(f"错误信息：{e}")

    info(f"[func 阶段] 处理完成: processed={processed}, skipped={skipped}")
    from copy import deepcopy

    anchor_update_output = deepcopy(anchor_data)
    anchor_update_output["anchor_tuple"] = new_anchors

    # 保存anchor
    output_func_dir = os.path.join(output_dir, f"func_{anchor_path_name}")
    # print("FUNC", output_func_dir)
    with open(output_func_dir, "w", encoding="utf-8") as f:
        json.dump(anchor_update_output, f, ensure_ascii=False, indent=2)

    remaining = len(function_anchors) - processed - skipped
    anchor_data["processed"] = processed
    anchor_data["skipped"] = skipped
    anchor_data["remaining"] = remaining
    anchor_data["status"] = "completed"

    return anchor_data
