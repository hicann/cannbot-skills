# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""适配层：将 api_locator 输出转换为 change_detection 可消费的格式。

api_locator 不能修改，所有适配逻辑在此完成。

核心职责：
1. 定位接口文档路径供 api_locator 使用
2. 将 api_locator 输出目录映射到 change_detection 期望的结构
3. 补齐 api_locator anchor 中缺失的字段，确保 change_detection 下游能正常消费
"""

import glob as glob_module
import json
import os
import re
import shutil
from pathlib import Path

from src.logger import debug, info, warning

# 章节中文名 → 英文 sub_part 映射（与 api_locator 的 DOC_TO_SUB_PART 对应）
SUB_PART_TO_CHAPTER = {
    "support": "产品支持情况",
    "summary": "功能说明",
    "impl_principle": "实现原理",
    "function_proto": "函数原型",
    "para_clarify": "参数说明",
    "return_clarify": "返回值说明",
    "constraint": "约束说明",
    "example": "调用示例",
}


def _find_doc_path(operator_path: str, operator_name: str | None = None) -> str:
    """从算子目录中定位接口文档 MD 文件供 api_locator.cli.run() 使用。

    api_locator 需要 doc_path（接口文档路径）作为入口参数。
    AscendC 仓惯例：文档在 docs/api/ 目录下，文件名可能是大写首字母（如 Add.md）。

    查找优先级：
    1. operator_name 已提供 → 在 docs/api/ 下按文件名精确匹配（大小写不敏感）
    2. aclnn 开头的文档（CANN 仓惯例）
    3. docs/api/ 下任意 MD 文件（取第一个）
    4. 全仓扫描 fallback

    Args:
        operator_path: 算子源码根目录
        operator_name: 算子名称（如 "Add"），提供后优先按名匹配接口文档

    Returns:
        接口文档 MD 文件的绝对路径

    Raises:
        FileNotFoundError: 未找到任何 MD 文件
    """
    # --- 优先级 1: 按 operator_name 在 docs/api/ 下精确匹配 ---
    if operator_name:
        api_md_files = glob_module.glob(os.path.join(operator_path, "docs", "api", "**", "*.md"), recursive=True)
        name_lower = operator_name.lower()
        for f in api_md_files:
            basename = os.path.splitext(os.path.basename(f))[0]
            # 文件名去掉中文括号、-侧接口、接口等后缀后再比较
            cleaned = re.sub(r"[（(].*?[）)]", "", basename)
            cleaned = re.sub(r"-.*侧接口$", "", cleaned)
            cleaned = re.sub(r"接口$", "", cleaned)
            cleaned = re.sub(r"-\d+$", "", cleaned)
            if cleaned.strip().lower() == name_lower:
                info(f"按算子名匹配到接口文档: {f}")
                return f
        warning(f"按算子名 '{operator_name}' 未在 docs/api/ 下匹配到文档，降级到全局搜索")

    # --- 优先级 2/3: docs/ 下搜索 ---
    md_files = glob_module.glob(os.path.join(operator_path, "docs", "**", "*.md"), recursive=True)
    if md_files:
        # 优先选 aclnn 开头的文档（CANN 仓惯例）
        aclnn_docs = [f for f in md_files if os.path.basename(f).startswith("aclnn")]
        if aclnn_docs:
            return aclnn_docs[0]
        # 优先取 docs/api/ 下的文件（接口文档，而非顶层指南）
        api_docs = [f for f in md_files if "api" in f.replace("\\", "/").split("/")]
        if api_docs:
            return api_docs[0]
        return md_files[0]

    # --- 优先级 4: 全仓 fallback ---
    md_files = glob_module.glob(os.path.join(operator_path, "**", "*.md"), recursive=True)
    aclnn_docs = [f for f in md_files if os.path.basename(f).startswith("aclnn")]
    if aclnn_docs:
        return aclnn_docs[0]
    if md_files:
        return md_files[0]

    raise FileNotFoundError(f"未找到接口文档: {operator_path}")


def _extract_api_name_from_export_root(export_root: str) -> str:
    """从 api_locator 输出目录路径中提取 API 名称。

    export_root 为 regroup_dir: .../{api_name}/  （含 refactor/anchor/doc_json/extra）
    或其子目录: .../{api_name}/refactor/ 等
    """
    export_root = export_root.rstrip("/\\")
    parts = [p for p in export_root.replace("\\", "/").split("/") if p]
    subdirs = {"refactor", "anchor", "doc_json", "extra", "code_map", "examples", "files"}
    while parts and parts[-1] in subdirs:
        parts.pop()
    return parts[-1] if parts else ""


def _adapt_output(export_root: str, out_path: str):
    """将 api_locator 输出目录映射到 change_detection 期望的结构。

    api_locator 输出 (regroup_dir):
        {api_name}/
        ├── anchor/{sub_part}.json            ← 8个章节 anchor
        ├── doc_json/{api_name}_doc.json      ← 文档章节内容
        ├── extra/code_map/                   ← code_map 数据
        ├── refactor/include/ impl/ tests/... ← 重组后源码
        └── refactor/docs/                    ← 重组后文档

    change_detection 期望:
        {out_path}/
        ├── anchor/{sub_part}.json            ← anchor JSON 文件
        ├── code_map/files/*.json             ← code_map 数据
        └── code_map/functions.json

    Args:
        export_root: api_locator 输出目录的绝对路径
        out_path: change_detection 期望的输出根目录
    """
    anchor_dir = os.path.join(out_path, "anchor")
    os.makedirs(anchor_dir, exist_ok=True)

    code_map_dir = os.path.join(out_path, "code_map")
    os.makedirs(code_map_dir, exist_ok=True)

    api_name = _extract_api_name_from_export_root(export_root)

    # --- 1. Anchor 文件复制到 anchor 目录 ---
    for sub_part, chapter_name in SUB_PART_TO_CHAPTER.items():
        dst_file = os.path.join(anchor_dir, f"{sub_part}.json")
        src_candidates = [
            os.path.join(export_root, "anchor", f"{sub_part}.json"),  # 新格式（anchor 子目录，refactor 同级）
            os.path.join(export_root, f"{api_name}anchor_{chapter_name}.json"),  # 旧格式
            os.path.join(export_root, f"{sub_part}.json"),  # 旧格式（直接在 export_root 下）
            os.path.join(export_root, "refactor", f"{sub_part}.json"),  # 历史格式（refactor 子目录）
        ]
        found = False
        for src_file in src_candidates:
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                info(f"  anchor: {os.path.basename(src_file)} → {sub_part}.json")
                found = True
                break
        if not found:
            debug(f"  章节 anchor 不存在（可能该 API 无此章节）: {sub_part}")

    # --- 2. Code Map 路径映射 ---
    # api_locator 将 code_map 输出到 extra/code_map/ 目录下
    src_code_map = os.path.join(export_root, "extra", "code_map")
    if os.path.isdir(src_code_map):
        # 复制 files/ 子目录中的每文件 map
        src_files_dir = os.path.join(src_code_map, "files")
        dst_files_dir = os.path.join(code_map_dir, "files")
        if os.path.isdir(src_files_dir):
            os.makedirs(dst_files_dir, exist_ok=True)
            for f in os.listdir(src_files_dir):
                if f.endswith(".json"):
                    shutil.copy2(os.path.join(src_files_dir, f), os.path.join(dst_files_dir, f))

        # 复制 functions.json
        src_func_json = os.path.join(src_code_map, "functions.json")
        if os.path.exists(src_func_json):
            shutil.copy2(src_func_json, os.path.join(code_map_dir, "functions.json"))

        # 复制 code_map.json (如果存在)
        src_map_json = os.path.join(src_code_map, "code_map.json")
        if os.path.exists(src_map_json):
            shutil.copy2(src_map_json, os.path.join(code_map_dir, "code_map.json"))
    else:
        # fallback: export_root 下直接有 code_map/ (api_locator 某些版本可能未移到 extra/)
        src_code_map_alt = os.path.join(export_root, "code_map")
        if os.path.isdir(src_code_map_alt):
            if os.path.isdir(os.path.join(src_code_map_alt, "files")):
                shutil.copytree(
                    os.path.join(src_code_map_alt, "files"), os.path.join(code_map_dir, "files"), dirs_exist_ok=True
                )
            src_func_json = os.path.join(src_code_map_alt, "functions.json")
            if os.path.exists(src_func_json):
                shutil.copy2(src_func_json, os.path.join(code_map_dir, "functions.json"))


def _normalize_file_path(fp):
    """将 file_path 从 list 归一化为 str。

    api_locator 的去重逻辑可将 file 字段变为 [path1, path2]，
    但 change_detection 所有重定向模块假定 file_path 是字符串。

    规则：
      - list 且非空 → 取第一个元素（最相关）
      - list 且为空 → 返回空字符串
      - str → 原样返回
    """
    if isinstance(fp, list):
        return fp[0] if fp else ""
    return fp


def _ensure_anchor_fields(anchor_data: dict) -> dict:
    """补齐 api_locator anchor 中缺失的字段，确保 change_detection 下游能正常消费。

    处理项：
    1. file_path 从 list 归一化为 str（api_locator 去重后 file 可为 list）
    2. file_name 从 file_path 推导（api_locator 仅 file 类型自带 file_name）
    3. level2.file_功能 补空值（file 阶段 Part2/Part3 需要）
    4. level3.tree_id 补空值（file 阶段 Part3 需要）
    5. function 类型补齐 function_name, call/callee/return, function_sign/function_hash
    6. note 类型补齐 behind_code, note_hash
    7. section 类型补齐 level2.token_hash（从 level1.token_hash 复制）
    8. code_snippet 类型降级为 section（change_detection 不识别 code_snippet）
    9. evidence 补齐 file_content/content/file_path（去重后 file_path 也可能为 list）
    10. evidence 列表至少保证 1 条记录（note/section 模块访问 evidence[0]）

    Args:
        anchor_data: 从 api_locator 输出读取的 anchor JSON dict

    Returns:
        补齐后的 anchor_data（原地修改，也返回引用）
    """
    for entry in anchor_data.get("anchor_tuple", []):
        info_dict = entry.setdefault("anchor_information", {})
        level1 = info_dict.setdefault("level1", {})
        level2 = info_dict.setdefault("level2", {})
        level3 = info_dict.setdefault("level3", {})
        type_ = info_dict.get("type", "")

        # --- file_path 从 list 归一化为 str ---
        # api_locator 去重后 file 可变为 [path1, path2]，但重定向模块全部假定字符串
        if "file_path" in level1:
            level1["file_path"] = _normalize_file_path(level1["file_path"])

        # --- file_name 从 file_path 推导 ---
        # api_locator 仅 file 类型自带 file_name；function/note/section 类型缺失
        if "file_name" not in level1:
            fp = level1.get("file_path", "")
            if isinstance(fp, str) and fp:
                level1["file_name"] = os.path.basename(fp)
            elif isinstance(fp, list) and fp:
                level1["file_name"] = os.path.basename(fp[0])
        else:
            # file_name 本身也可能是 list（api_locator _resolve_file_path）
            if isinstance(level1["file_name"], list) and level1["file_name"]:
                level1["file_name"] = os.path.basename(level1["file_name"][0])
            elif isinstance(level1["file_name"], list):
                level1["file_name"] = ""

        # --- 确保 level2/level3 有键（空值即可，但不能缺键） ---
        level2.setdefault("file_功能", "")
        level3.setdefault("tree_id", "")

        # --- type 特定补齐 ---
        if type_ == "function":
            level1.setdefault("function_name", "")
            # call/callee/return 可能是空列表 [] 或空字符串 ""，统一为 []
            for key in ("call", "callee", "return"):
                val = level2.get(key)
                if val == "" or val is None:
                    level2[key] = []
                elif not isinstance(val, list):
                    level2[key] = [val] if val else []
            level3.setdefault("function_sign", "")
            level3.setdefault("function_hash", "")

        elif type_ == "note":
            level1.setdefault("behind_code", "")
            level2.setdefault("note_hash", level2.get("token_hash", ""))

        elif type_ == "section":
            # level2.token_hash 可能缺失，从 level1.token_hash 复制
            if "token_hash" not in level2 and "token_hash" in level1:
                level2["token_hash"] = level1["token_hash"]

        elif type_ == "code_snippet":
            # change_detection 的 4 阶段不识别 code_snippet，降级为 section
            info_dict["type"] = "section"
            if "token_hash" not in level2 and "token_hash" in level1:
                level2["token_hash"] = level1["token_hash"]

        # --- evidence 补齐 ---
        # 确保 evidence 列表至少有 1 条记录（note/section 模块访问 evidence[0]）
        evidence_list = entry.get("evidence", [])
        if not evidence_list:
            evidence_list = [
                {
                    "evidence_id": "",
                    "evidence_information": {
                        "file_path": level1.get("file_path", ""),
                        "file_name": level1.get("file_name", ""),
                        "file_content": "",
                        "content": "",
                    },
                }
            ]
            entry["evidence"] = evidence_list

        for ev in evidence_list:
            ev_info = ev.setdefault("evidence_information", {})
            # evidence_information.file_path 也可能为 list（同 level1.file_path）
            if "file_path" in ev_info:
                ev_info["file_path"] = _normalize_file_path(ev_info["file_path"])
            # file_name 同理
            if "file_name" in ev_info:
                if isinstance(ev_info["file_name"], list) and ev_info["file_name"]:
                    ev_info["file_name"] = os.path.basename(ev_info["file_name"][0])
                elif isinstance(ev_info["file_name"], list):
                    ev_info["file_name"] = ""
            # merge_evidence._get_evidence_content() 优先读 file_content，其次读 content
            ev_info.setdefault("file_content", "")
            # content 在 function 类型中是列表，在其他类型中是字符串
            if "content" not in ev_info:
                ev_info["content"] = ""

    return anchor_data


def adapt_and_ensure(out_path: str):
    """对 out_path/anchor/ 下所有 JSON 文件执行字段补齐。"""

    anchor_dir = os.path.join(out_path, "anchor")
    if not os.path.isdir(anchor_dir):
        warning(f"anchor 目录不存在: {anchor_dir}")
        return

    for filename in os.listdir(anchor_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(anchor_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            _ensure_anchor_fields(data)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except (json.JSONDecodeError, IOError) as e:
            warning(f"处理 anchor 文件失败: {filepath}, 错误: {e}")
