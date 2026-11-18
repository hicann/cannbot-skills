#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Anchor Redirect Pipeline - 统一入口函数

按顺序执行四个重定向阶段:
1. file 阶段: 文件级重定向 (type="file")
2. func 阶段: 函数级重定向 (type="function")
3. note 阶段: 注释重定向 (type="note")
4. section 阶段: 章节重定向 (type="section")
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console

from src.logger import debug, error, info, warning

_truncate_keywords = []  # 路径格式已统一，不再需要截断


def _truncate_path(path: str | list) -> str:
    """路径归一化：统一使用正斜杠，兼容 Windows/Linux/macOS。

    注意：原截断逻辑（op_api/op_host/op_kernel/op_graph）已不适用新路径格式，
    现在路径格式与 api_locator 一致，都是纯相对路径。

    兼容 str 和 list 两种输入格式（多架构文件聚合时 file_path 可能为 list）：
    - str: 直接归一化返回
    - list: 归一化每个元素，返回第一个元素（用于 unchanged_file_paths 集合匹配）
    """
    if isinstance(path, list):
        normalized = [p.replace("\\", "/") for p in path]
        return normalized[0] if normalized else ""
    else:
        return path.replace("\\", "/")


# 确保各子模块可被导入
sys.path.insert(0, str(Path(__file__).parent))
# 确保 api_change_detect 可被导入
_api_detect_path = str(Path(__file__).parent.parent)
if _api_detect_path not in sys.path:
    sys.path.insert(0, _api_detect_path)

# 抑制子模块中 logging.INFO/WARNING/ERROR 的输出，只保留 CRITICAL
logging.basicConfig(level=logging.CRITICAL, format="%(message)s")
for logger_name in ["", __name__]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def _detect_changed_files(origin_repo: str, change_repo: str) -> set:
    """MD5 对比新旧仓库，返回变更文件相对路径集合（仅含 change_repo 中存在且发生变更的文件）

    Args:
        origin_repo: 变更前算子仓库路径
        change_repo: 变更后算子仓库路径

    Returns:
        变更文件的相对路径集合（相对于 change_repo）
    """
    _cpp_h_ext = (".cpp", ".h", ".hpp", ".cc", ".cxx")
    changed = set()

    old_hashes = {}
    for root, dirs, files in os.walk(origin_repo):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith(_cpp_h_ext):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, origin_repo).replace("\\", "/")
            try:
                with open(fp, "rb") as fh:
                    old_hashes[rel] = hashlib.md5(fh.read(), usedforsecurity=False).hexdigest()
            except (FileNotFoundError, PermissionError):
                pass

    for root, dirs, files in os.walk(change_repo):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith(_cpp_h_ext):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, change_repo).replace("\\", "/")
            try:
                with open(fp, "rb") as fh:
                    new_hash = hashlib.md5(fh.read(), usedforsecurity=False).hexdigest()
                if rel not in old_hashes or old_hashes[rel] != new_hash:
                    changed.add(rel)
            except (FileNotFoundError, PermissionError):
                pass

    return changed


def run_anchor_redirect_pipeline(
    anchor_path: str,
    origin_map_base: str,
    change_map_base: str,
    origin_repo_path: str,
    change_repo_path: str,
    output_dir: str = "anchor_redirect_output",
    verbose: bool = False,
    anchor_type_filter: list = None,
    anchor_file_filter: list = None,
) -> dict:
    """
    顺序执行 file → func → note → section 四个重定向阶段。

    参数:
        anchor_path: anchor JSON 文件路径（四个阶段共用）
        origin_map_base: 变更前 code_map 目录 (file/func 共用)
        change_map_base: 变更后 code_map 目录 (file/func 共用)
        change_repo_path: 变更后算子仓库路径 (note/section 用作 data_dir)
        output_dir: 输出目录，默认 "anchor_redirect_output"
        verbose: 是否输出详细日志
        anchor_type_filter: 只处理文件名匹配 {item}.json 的 anchor，为空时处理全部

    返回:
        汇总报告字典
    """
    console = Console()
    output_dir = os.path.join(output_dir, "redirect")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    info("重定向输出路径：", output_dir)

    # 初始化汇总报告（必须在循环外，确保 anchor_files 为空时 report 也有值）
    report = {
        "anchor_path": anchor_path,
        "origin_map_base": origin_map_base,
        "change_map_base": change_map_base,
        "change_repo_path": change_repo_path,
        "output_dir": output_dir,
        "stages": {},
    }

    # 读取初始 anchor JSON
    anchor_files = os.listdir(anchor_path)
    # 按 anchor_type_filter 过滤：只保留文件名匹配 {item}.json 的 anchor
    if anchor_type_filter:
        allowed_names = {f"{item}.json" for item in anchor_type_filter}
        before_count = len(anchor_files)
        anchor_files = [f for f in anchor_files if f in allowed_names]
        skipped = before_count - len(anchor_files)
        if skipped > 0:
            info(f"anchor_type_filter 过滤：跳过 {skipped} 个 anchor 文件，保留 {len(anchor_files)} 个")
    for anchor_file in anchor_files:
        with open(os.path.join(anchor_path, anchor_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        anchors = data.get("anchor_tuple", [])
        debug(f"  {anchor_file}加载 {len(anchors)} 个 anchor")

        # 统计 anchor 类型
        type_counts = {}
        for a in anchors:
            t = a.get("anchor_information", {}).get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        type_parts = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
        # print(f"  类型分布: {type_parts}")

        debug(json.dumps(report, indent=2, ensure_ascii=False))

    # =====================
    # Stage 0: 变更文件预检测
    # =====================

    changed_files = None
    if anchor_file_filter:
        changed_files = anchor_file_filter
    else:
        if (
            origin_repo_path
            and change_repo_path
            and os.path.isdir(origin_repo_path)
            and os.path.isdir(change_repo_path)
        ):
            console.print("    [dim]【Stage 0】变更文件预检测[/dim]")
            changed_files = _detect_changed_files(origin_repo_path, change_repo_path)
            info(f"[Stage 0] 发现 {len(changed_files)} 个变更文件（.cpp/.h），将用于缩小重定向候选范围")
        else:
            info("[Stage 0] 跳过：缺少新旧仓库路径，降级为全量匹配")

    # =====================
    # Level 1: 无变更 → 跳过全部重定向阶段
    # =====================
    if changed_files is not None and len(changed_files) == 0:
        info("[Level 1] 无变更文件，跳过所有重定向阶段")
        console.print("    [green]✓ 无变更文件，所有 anchor 标记为 unchanged[/green]")
        # 把所有 anchor 标记为 unchanged 并写入输出目录
        for anchor_file in anchor_files:
            anchor_file_path = os.path.join(anchor_path, anchor_file)
            with open(anchor_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for anchor in data.get("anchor_tuple", []):
                anchor["change_type"] = "unchanged"
            out_file = os.path.join(output_dir, f"unchanged_{anchor_file}")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        report["stages"]["overall"] = {
            "status": "skipped_no_changes",
            "changed_files_count": 0,
        }
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        info("Pipeline 完成（无变更）")
        return report

    # =====================
    # Level 2: 计算未变更文件路径集合，用于后续阶段跳过
    # =====================
    unchanged_file_paths = set()
    if changed_files is not None and len(changed_files) > 0:
        for anchor_file in anchor_files:
            anchor_file_path = os.path.join(anchor_path, anchor_file)
            with open(anchor_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for anchor in data.get("anchor_tuple", []):
                file_path = anchor.get("anchor_information", {}).get("level1", {}).get("file_path", "")
                if file_path:
                    truncated = _truncate_path(file_path)
                    if truncated not in changed_files:
                        unchanged_file_paths.add(truncated)
        info(f"[Level 2] {len(unchanged_file_paths)} 个文件路径未变更，对应 anchor 将在 func/note/section 阶段跳过")

    console.print("    [dim]【File 阶段】文件级重定向[/dim]")
    for anchor_file in anchor_files:
        # =====================
        # Stage 1: file 阶段
        # =====================
        import anchor_redirect_file.ancor_match_file_0422 as file_module

        report_return, file_data = file_module.process(
            os.path.join(anchor_path, anchor_file),
            origin_map_base,
            change_map_base,
            change_repo_path,
            output_dir,
            changed_files=changed_files,
        )

        # 同步 anchor 数据到输出目录
        with open(os.path.join(output_dir, f"file_{anchor_file}"), "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
    info(f"[Stage 1] file 阶段 - 完成")

    info("====" * 20)

    # for anchor_file in anchor_files:
    # =====================
    # Stage 2: func 阶段
    # =====================
    console.print("    [dim]【Function 阶段】函数级重定向[/dim]")
    func_result = anchor_redirect_file_try_func_process(
        anchor_path,
        origin_map_base,
        change_map_base,
        change_repo_path,
        output_dir,
        verbose,
        changed_files=changed_files,
        unchanged_file_paths=unchanged_file_paths,
        anchor_files=anchor_files,
    )

    info(f"[Stage 2] func 阶段 - 完成")
    info("====" * 20)
    # =====================
    # Stage 3: note 阶段
    # =====================
    console.print("    [dim]【Note 阶段】注释级重定向[/dim]")
    try:
        import anchor_redirect_note.anchor_retarget_note as note_module

        result = note_module.process_anchor_data(
            anchor_path,
            change_repo_path,
            output_dir,
            verbose,
            unchanged_file_paths=unchanged_file_paths,
            anchor_files=anchor_files,
        )

        info(f"[Stage 3] note 阶段 - 完成")
        # print(result)
        # print(f"  处理: {stats['processed']} 个 | 找到: {stats['found']} 个 | 未找到: {stats['not_found']} 个")
    except Exception as e:
        report["stages"]["note"] = {
            "status": f"error: {e}",
        }
        # 打印异常详细信息：文件名、行号、堆栈
        import sys
        import traceback

        traceback.print_exc()

        # 单独获取 当前异常的 文件名、行号
        exc_type, exc_value, exc_tb = sys.exc_info()
        fname = exc_tb.tb_frame.f_code.co_filename  # 报错文件
        line_no = exc_tb.tb_lineno  # 报错行号

        error(f"\n报错文件：{fname}")
        error(f"报错行号：{line_no}")
        error(f"错误信息：{e}")
        info(f"[Stage 3] note 阶段 - 完成")
        error(f"  错误: {e}")
    info("====" * 20)
    # =====================
    # Stage 4: section 阶段
    # =====================
    console.print("    [dim]【Section 阶段】章节级重定向[/dim]")
    try:
        import anchor_redirect_section.anchor_retarget_section as section_module

        result = section_module.process_anchor_data(
            anchor_path,
            change_repo_path,
            output_dir,
            verbose,
            unchanged_file_paths=unchanged_file_paths,
            anchor_files=anchor_files,
        )

        info(f"[Stage 4] section 阶段 - 完成")
        # print(result)
        # print(f"  处理: {stats['processed']} 个 | 找到: {stats['found']} 个 | 未找到: {stats['not_found']} 个")
    except Exception as e:
        report["stages"]["section"] = {
            "status": f"error: {e}",
        }
        info(f"[Stage 4] section 阶段 - 完成")
        error(f"  错误: {e}")
        # 打印异常详细信息：文件名、行号、堆栈
        import sys
        import traceback

        traceback.print_exc()

    # 保存汇总报告
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    info("Pipeline 完成")
    info(f"  汇总报告: {summary_path}")

    return report


def anchor_redirect_file_try_func_process(
    anchor_path: str,
    origin_map_base: str,
    change_map_base: str,
    change_repo_path: str,
    output_dir: str,
    verbose: bool,
    changed_files: set = None,
    unchanged_file_paths: set = None,
    anchor_files: list = None,
):
    """尝试调用 func 模块的 process 函数，失败时返回 None 并打印错误。"""
    import anchor_redirect_func.function_redirect_v1 as func_module

    # 如果提供了 anchor_files（已过滤），使用它；否则自己扫描
    if anchor_files is None:
        anchor_files = os.listdir(anchor_path)
    function_data_result = []
    for anchor in anchor_files:
        anchor_json_path = os.path.join(anchor_path, anchor)
        func_data = func_module.process(
            anchor_json_path,
            origin_map_base,
            change_map_base,
            change_repo_path,
            output_dir,
            verbose,
            changed_files=changed_files,
            unchanged_file_paths=unchanged_file_paths,
        )
        function_data_result.append(func_data)

    # print(function_data_result)
    return function_data_result


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Anchor Redirect Pipeline - 顺序执行 file→func→note→section 四个重定向阶段"
    )
    parser.add_argument(
        "--anchor-path",
        default="./out/base/anchor/",
        type=str,
        # required=True,
        help="anchor JSON 文件路径（四个阶段共用）",
    )
    parser.add_argument(
        "--origin-map-base",
        default="out/base/code_map/base",
        type=str,
        # required=True,
        help="变更前 code_map 目录（file/func 共用）",
    )
    parser.add_argument(
        "--change-map-base",
        default="out/base/code_map/base",
        type=str,
        # required=True,
        help="变更后 code_map 目录（file/func 共用）",
    )
    parser.add_argument(
        "--origin-repo-path",
        default="./out/operator_storage/roi_align/base/op_host/op_api/aclnn_roi_align.cpp",
        type=str,
        # required=True,
        help="变更前算子仓库路径（note/section 用作 data_dir）",
    )
    parser.add_argument(
        "--change-repo-path",
        default="./out/operator_storage/roi_align/base",
        type=str,
        # required=True,
        help="变更后算子仓库路径（note/section 用作 data_dir）",
    )
    # parser.add_argument()
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录（默认: anchor_redirect_output）")
    parser.add_argument("--verbose", action="store_true", default=True, help="显示详细输出")
    parser.add_argument(
        "--anchor-type-filter",
        nargs="+",
        default=None,
        help="只处理文件名匹配 {item}.json 的 anchor，为空时处理全部（例: support summary constraint）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_anchor_redirect_pipeline(
        anchor_path=args.anchor_path,
        origin_map_base=args.origin_map_base,
        change_map_base=args.change_map_base,
        origin_repo_path=args.origin_repo_path,
        change_repo_path=args.change_repo_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        anchor_type_filter=args.anchor_type_filter,
    )


if __name__ == "__main__":
    main()
