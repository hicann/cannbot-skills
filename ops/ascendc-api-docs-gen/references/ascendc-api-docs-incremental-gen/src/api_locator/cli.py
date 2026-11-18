# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""API 代码定位工具命令行入口。

使用方式：
    python cli.py docs/zh/api/**/Add.md
    python cli.py /abs/path/to/Add.md --repo /abs/path/to/asc-devkit --output data/out
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formatter import AnchorFormatter

# 默认仓库根目录（从 config.py 导入，支持 ASCENDC_REPO_ROOT 环境变量覆盖）
from config import (
    ADV_API_EXTRA_CHAPTER,
    CHAPTERS,
    OUTPUT_ANCHOR_DIR,
    OUTPUT_CODE_MAP_DIR,
    OUTPUT_DOC_JSON_DIR,
    OUTPUT_EXTRA_DIR,
)
from config import REPO_ROOT as _DEFAULT_REPO_ROOT
from doc_parser import get_sub_part, load_chapters_json, save_chapters_json
from evidence import APILocator
from file_regroup import export_regroup


def run(
    doc_path: str,
    repo_root: str = _DEFAULT_REPO_ROOT,
    output_dir: str = "data/out",
    api_type: str = None,
    selected_chapters: list = None,
) -> str:
    """完整流程：打包重组文件夹 → 从其中提取八章证据 → 生成章节 JSON。

    Args:
        doc_path: 接口文档路径（相对路径基于 repo_root，或绝对路径）
        repo_root: 代码仓库根目录（asc-devkit 路径）
        output_dir: 输出目录（相对路径基于运行目录，或绝对路径）
        api_type: API 类型，不传则自动识别
        selected_chapters: 指定提取的章节名列表

    Returns:
        导出目录的绝对路径
    """
    repo_root = os.path.abspath(repo_root)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isabs(doc_path):
        doc_path = os.path.join(repo_root, doc_path)
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"文档不存在: {doc_path}")

    # 1. 打包重组（调用独立脚本模块）
    export_root = export_regroup(doc_path, api_type=api_type, repo_root=repo_root, output_dir=output_dir)
    regroup_dir = os.path.dirname(export_root)

    # 2. 提取 API 名和类型
    api_name = APILocator.extract_api_name_from_doc(doc_path)
    rel_doc_path = os.path.relpath(doc_path, repo_root)

    locator = APILocator(repo_root=repo_root)
    step1 = locator.determine_api_type(api_name)
    api_type = step1["api_type"]
    decl_file = step1["decl_file"]

    # 3. 解析接口文档，输出中间 JSON 到 doc_json/ 子目录
    doc_json_dir = os.path.join(regroup_dir, OUTPUT_DOC_JSON_DIR)
    os.makedirs(doc_json_dir, exist_ok=True)
    doc_json_path = save_chapters_json(doc_path, doc_json_dir)
    print(f"  文档 JSON: {doc_json_path}")

    # 3.5 构建 tree_id（重组目录的目录树）
    from tree_id import build_folder_tree

    tree_id_text = build_folder_tree(export_root)

    # 3.6 生成 code_map（输出到 extra/ 子目录）
    import shutil

    from code_map import CodeMapLoader, build_code_map

    code_map_dir = build_code_map(export_root)  # 生成在 export_root/code_map/
    extra_dir = os.path.join(regroup_dir, OUTPUT_EXTRA_DIR)
    os.makedirs(extra_dir, exist_ok=True)
    extra_code_map = os.path.join(extra_dir, OUTPUT_CODE_MAP_DIR)
    if os.path.exists(extra_code_map):
        shutil.rmtree(extra_code_map)
    shutil.move(code_map_dir, extra_code_map)
    code_map_dir = extra_code_map
    print(f"  Code Map: {code_map_dir}")
    code_map_loader = CodeMapLoader(code_map_dir)

    # 4. 从重组文件夹重新构建 sections（只提取文档中存在的章节）
    locator2 = APILocator(repo_root=export_root)
    impl_result = locator2.find_impl_files(api_type, decl_file, api_name)
    test_files = locator2.find_test_files(api_type, api_name)
    sections = locator2._build_sections(
        api_name,
        api_type,
        decl_file,
        impl_result,
        test_files,
        read_base=export_root,
        doc_chapters_json_path=doc_json_path,
        selected_chapters=selected_chapters,
    )

    # 5. 从 JSON 读取章节内容，生成章节 anchor JSON
    doc_chapters = load_chapters_json(doc_json_path)

    # 5. 生成章节 JSON
    fmt = AnchorFormatter(code_map_loader=code_map_loader, tree_id_text=tree_id_text, repo_root=export_root)
    function_proto_content = ""
    decl_snippets = sections.get("函数原型", {}).get("sources", [])
    if decl_snippets:
        function_proto_content = decl_snippets[0].get("content", decl_snippets[0].get("snippet", ""))

    chapters_to_output = [ch for ch in CHAPTERS if ch != ADV_API_EXTRA_CHAPTER or api_type == "adv_api"]
    if selected_chapters:
        chapters_to_output = [ch for ch in chapters_to_output if ch in selected_chapters]

    anchor_dir = os.path.join(regroup_dir, OUTPUT_ANCHOR_DIR)
    os.makedirs(anchor_dir, exist_ok=True)

    for chapter in chapters_to_output:
        sources = sections.get(chapter, {}).get("sources", [])
        fmt._anchor_counter = 0
        fmt._ev_counter = 0
        # chip_arch 是代码层面扫描的元数据，不属于四维表证据，不生成 anchor
        anchor_sources = [s for s in sources if s.get("type") != "chip_arch"]
        sub_part = get_sub_part(chapter)
        anchors = fmt.sources_to_anchors(anchor_sources, sub_part)

        doc_content = doc_chapters.get(chapter, "")

        out_path = fmt.write_chapter_json(
            anchor_dir,
            api_name,
            rel_doc_path,
            chapter,
            anchors,
            sources=sources,
            doc_content=doc_content,
            function_proto=function_proto_content,
        )
        print(f"  章节 JSON: {out_path}")

    return regroup_dir


def main():
    parser = argparse.ArgumentParser(description="API 代码定位工具：输入接口文档，输出重组文件夹及章节 JSON")
    parser.add_argument("doc", help="接口文档路径（相对路径基于 --repo，或绝对路径）")
    parser.add_argument("--repo", default=_DEFAULT_REPO_ROOT, help=f"代码仓库根目录（默认: {_DEFAULT_REPO_ROOT}）")
    parser.add_argument("--output", default="data/out", help="输出目录（相对路径基于运行目录，默认: data/out）")
    parser.add_argument("--chapters", help="指定提取章节，逗号分隔（如 '函数原型,参数说明'）")
    args = parser.parse_args()

    selected = None
    if args.chapters:
        selected = [c.strip() for c in args.chapters.split(",") if c.strip()]

    try:
        path = run(args.doc, repo_root=args.repo, output_dir=args.output, selected_chapters=selected)
        count = sum(len(f) for _, _, f in os.walk(path))
        print(f"\n导出完成: {path}")
        print(f"共 {count} 个文件")
        if selected:
            print(f"提取章节: {', '.join(selected)}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
