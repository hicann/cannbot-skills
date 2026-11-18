# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""接口文档解析器：从 docs/zh/api 下的 {api_name}.md 提取每个章节的原始内容。"""

import json
import os
import re
import shutil

from config import DOC_TO_SUB_PART, OUTPUT_DOC_JSON_SUFFIX  # noqa: F401  re-export for backward compatibility


def extract_chapters(doc_path: str) -> dict:
    """从接口文档按章节标题分割，提取每个章节的原始内容（含 ## 章节标题行）。

    Args:
        doc_path: 文档的绝对路径

    Returns:
        {章节名: 章节原始内容字符串}，内容以 "## 章节名" 标题行开头
    """
    if not os.path.exists(doc_path):
        return {}

    with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # 按 ## 章节名 分割
    # 章节标题格式：## 功能说明<a name="..."></a>
    pattern = re.compile(r"^##\s+([^<\n]+)", re.MULTILINE)
    matches = list(pattern.finditer(content))

    chapters = {}
    for i, m in enumerate(matches):
        chapter_name = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[start:end].strip()
        if section_content:
            chapters[chapter_name] = section_content

    return chapters


def save_chapters_json(doc_path: str, output_dir: str) -> str:
    """解析文档并保存为 JSON，供后续证据生成阶段读取。

    同时将源 md 文件复制到输出目录，便于人工对照原文。

    Args:
        doc_path: 文档绝对路径
        output_dir: 输出目录

    Returns:
        生成的 JSON 文件路径
    """
    chapters = extract_chapters(doc_path)
    api_name = os.path.splitext(os.path.basename(doc_path))[0]
    api_name = re.sub(r"[（(].*?[）)]", "", api_name)
    api_name = re.sub(r"-.*侧接口$", "", api_name)
    api_name = re.sub(r"接口$", "", api_name)
    api_name = re.sub(r"-\d+$", "", api_name)
    api_name = api_name.strip()

    json_path = os.path.join(output_dir, f"{api_name}{OUTPUT_DOC_JSON_SUFFIX}")
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False, indent=2)
    if os.path.exists(doc_path):
        _ = shutil.copy2(doc_path, output_dir)
    return json_path


def load_chapters_json(json_path: str) -> dict:
    """从已解析的 JSON 文件读取章节内容。

    Args:
        json_path: JSON 文件路径

    Returns:
        {章节名: 原文内容} 字典
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sub_part(doc_part: str) -> str:
    """将文档章节名映射为 sub_part。"""
    return DOC_TO_SUB_PART.get(doc_part, doc_part.lower())
