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
anchor_retarget.py - 从 JSON 文件中提取 anchor_tuple 并在源文件中查找 note_hash 内容

功能：
1. 遍历指定目录（out/doc_json）下的所有 JSON 文件
2. 读取每个 JSON 文件中的 anchor_tuple 字段
3. 过滤只处理 anchor_information.type 为 "note"的数据
4. 从 level1 获取 file_path，从 level2 获取 note_hash
5. 在 data/ 目录下查找对应的源文件
6. 在文件中查找 note_hash 内容：
   - 首先尝试直接在文件中匹配 note_hash，找到则直接输出
   - 未找到则提取最相似的字符串片段，和 level1.behind_code 前的代码片段并分别输出
7. 输出查找结果，包括是否找到、开始行号、结束行号和内容
"""

import argparse
import json
import os
from difflib import SequenceMatcher
from pathlib import Path

from src.logger import debug, error, info, warning


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在源文件中查找 anchor_tuple 中的 note_hash 内容")
    parser.add_argument("--json-dir", type=str, default="out/doc_json", help="JSON 文件目录路径 (默认: out/doc_json)")
    parser.add_argument("--data-dir", type=str, default="data", help="源文件根目录路径 (默认: data)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径 (默认: 输出到控制台)")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    return parser.parse_args()


def find_json_files(json_dir):
    """遍历目录下的所有 JSON 文件"""
    json_path = Path(json_dir)
    if not json_path.exists():
        error(f"目录不存在: {json_dir}")
        return []

    json_files = list(json_path.glob("*.json"))
    return sorted(json_files)


def read_json_file(json_file):
    """读取 JSON 文件"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        warning(f"JSON 解析错误 {json_file}: {e}")
        return None
    except Exception as e:
        warning(f"读取文件失败 {json_file}: {e}")
        return None


def extract_note_anchors(data):
    """提取 anchor_tuple 中 type 为 note 的数据"""
    anchors = []

    if not isinstance(data, dict):
        return anchors

    anchor_tuple = data.get("anchor_tuple", [])
    if not isinstance(anchor_tuple, list):
        return anchors

    for item in anchor_tuple:
        if not isinstance(item, dict):
            continue

        anchor_info = item.get("anchor_information", {})
        if anchor_info.get("type") != "note":
            continue

        level1 = anchor_info.get("level1", {})
        level2 = anchor_info.get("level2", {})

        file_path = level1.get("file_path", "")
        behind_code = level1.get("behind_code", "")
        note_hash = level2.get("note_hash", "")

        if file_path and note_hash:
            anchors.append(
                {
                    "file_path": file_path,
                    "behind_code": behind_code,
                    "note_hash": note_hash,
                    "anchor_id": item.get("anchor_id", ""),
                    "note": item.get("note", ""),
                    "anchor_information": anchor_info,
                    "evidence": item.get("evidence", []),
                }
            )

    return anchors


def find_source_file(data_dir, file_path):
    """在 data 目录下查找源文件"""
    # 尝试不同的路径组合
    possible_paths = [
        Path(data_dir) / file_path,
        Path(data_dir) / file_path.lstrip("/"),
        Path(file_path),
    ]

    for path in possible_paths:
        if path.exists() and path.is_file():
            return path

    return None


def read_source_file(file_path):
    """读取源文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), f.readlines()
    except Exception as e:
        warning(f"读取源文件失败 {file_path}: {e}")
        return None, None


def find_note_hash_in_file(content, lines, note_hash):
    """
    在文件中查找 note_hash 内容
    返回: (是否找到, 开始行号, 结束行号, 匹配内容)
    """
    if not note_hash or not content:
        return False, -1, -1, ""

    # 尝试直接匹配
    if note_hash in content:
        start_idx = content.index(note_hash)
        # 计算行号
        start_line = content[:start_idx].count("\n") + 1
        end_line = start_line + note_hash.count("\n")
        return True, start_line, end_line, note_hash

    # 如果直接匹配失败，尝试找到最相似的片段
    best_match = find_best_match(content, note_hash)
    if best_match:
        start_idx = content.index(best_match)
        start_line = content[:start_idx].count("\n") + 1
        end_line = start_line + best_match.count("\n")
        return False, start_line, end_line, best_match

    return False, -1, -1, ""


def find_best_match(content, target):
    """找到与目标最相似的字符串片段"""
    # 将内容分割成行
    content_lines = content.split("\n")
    target_lines = target.split("\n")
    if not target_lines or not content_lines:
        return None
    best_similarity = 0
    best_match = None
    best_start = 0

    # 使用滑动窗口查找最相似的片段
    window_size = len(target_lines)
    for i in range(len(content_lines) - window_size + 1):
        window = "\n".join(content_lines[i: i + window_size])
        similarity = calculate_similarity(window, target)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = window
            best_start = i

    # 如果相似度太低，尝试更小的窗口
    if best_similarity < 0.5 and window_size > 1:
        # 尝试逐行匹配
        for i, line in enumerate(content_lines):
            if target_lines[0] in line or target_lines[0].strip() in line.strip():
                # 找到可能的起始行
                match_lines = []
                for j in range(min(len(target_lines), len(content_lines) - i)):
                    if i + j < len(content_lines):
                        match_lines.append(content_lines[i + j])
                if match_lines:
                    match_str = "\n".join(match_lines)
                    similarity = calculate_similarity(match_str, target)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = match_str
                        best_start = i

    # 只返回相似度高于阈值的匹配
    if best_similarity >= 0.3:
        return best_match

    return None


def calculate_similarity(a, b):
    """计算两个字符串的相似度"""
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b).ratio()


def find_behind_code_context(content, behind_code, context_lines=5):
    """查找 behind_code 前的代码片段"""
    if not behind_code or not content:
        return None, -1, -1

    # 查找 behind_code 在文件中的位置
    if behind_code in content:
        idx = content.index(behind_code)
        line_num = content[:idx].count("\n") + 1

        # 获取前面的代码片段
        lines = content.split("\n")
        start_line = max(0, line_num - context_lines - 1)
        end_line = line_num - 1

        context = "\n".join(lines[start_line:end_line])
        return context, start_line + 1, end_line

    return None, -1, -1


def process_anchor(anchor, data_dir, verbose=False):
    """处理单个 anchor"""
    file_path = anchor["file_path"]
    behind_code = anchor["behind_code"]
    note_hash = anchor["note_hash"]
    anchor_id = anchor.get("anchor_id", "")
    from copy import deepcopy

    copy_anchor = deepcopy(anchor)
    # 查找源文件

    source_file = find_source_file(data_dir, file_path)
    if not source_file:
        return {
            "anchor_id": anchor_id,
            "file_path": file_path,
            "found": False,
            "error": f"源文件未找到: {file_path}",
        }, None

    # 读取源文件
    content, lines = read_source_file(source_file)
    if content is None:
        return {
            "anchor_id": anchor_id,
            "file_path": file_path,
            "found": False,
            "error": f"无法读取源文件: {source_file}",
        }, None

    # 查找 note_hash
    found, start_line, end_line, matched_content = find_note_hash_in_file(content, lines, note_hash)

    # 查找 behind_code 的上下文
    behind_context, behind_start, behind_end = find_behind_code_context(content, behind_code)

    # 将复制的旧的anchor内容更新
    anchor_information_level1 = copy_anchor["anchor_information"]["level1"]
    anchor_information_level1["file_path"] = file_path
    anchor_information_level1["behind_code"] = ""

    anchor_information_level2 = copy_anchor["anchor_information"]["level2"]
    anchor_information_level2["token_hash"] = matched_content if found else None
    anchor_information_level2["note_hash"] = matched_content if found else None

    # 保护：evidence 列表可能为空（api_locator 去重后或原始数据缺失）
    evidence_list = copy_anchor.get("evidence", [])
    if evidence_list:
        anchor_information_evidence = evidence_list[0]
        anchor_information_evidence["evidence_information"]["file_path"] = file_path
        anchor_information_evidence["evidence_information"]["content"] = matched_content if found else None
        # 同步更新 file_content（_get_evidence_content 优先取此字段，不更新会残留修改前内容）
        anchor_information_evidence["evidence_information"]["file_content"] = matched_content if found else None
    else:
        # 无 evidence 时创建默认条目
        copy_anchor["evidence"] = [
            {
                "evidence_id": "",
                "evidence_information": {
                    "file_path": file_path,
                    "content": matched_content if found else None,
                    "file_content": matched_content if found else None,
                },
            }
        ]

    result = {
        "anchor_id": anchor_id,
        "file_path": file_path,
        "source_file": str(source_file),
        "found": found,
        "note_hash": {
            "start_line": start_line,
            "end_line": end_line,
            "content": matched_content if found else None,
            "similar_content": matched_content if not found else None,
        },
    }

    if behind_context:
        result["behind_code_context"] = {"start_line": behind_start, "end_line": behind_end, "content": behind_context}

    if verbose:
        result["behind_code"] = behind_code
        result["note_hash_original"] = note_hash

    return result, copy_anchor


def format_result(result):
    """格式化输出结果"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Anchor ID: {result.get('anchor_id', 'N/A')}")
    lines.append(f"源文件路径: {result.get('file_path', 'N/A')}")
    lines.append(f"实际文件: {result.get('source_file', 'N/A')}")

    status = "找到" if result.get("found") else "未找到"
    lines.append(f"note_hash 查找结果: {status}")

    note_info = result.get("note_hash", {})
    if note_info.get("start_line", -1) > 0:
        lines.append(f"  开始行号: {note_info['start_line']}")
        lines.append(f"  结束行号: {note_info['end_line']}")

    if note_info.get("content"):
        lines.append("  匹配内容:")
        for line in note_info["content"].split("\n"):
            lines.append(f"    | {line}")

    if note_info.get("similar_content") and not result.get("found"):
        lines.append("  最相似内容:")
        for line in note_info["similar_content"].split("\n"):
            lines.append(f"    | {line}")

    behind_ctx = result.get("behind_code_context")
    if behind_ctx and behind_ctx.get("start_line", -1) > 0:
        lines.append(f"behind_code 上下文 (行 {behind_ctx['start_line']}-{behind_ctx['end_line']}):")
        for line in behind_ctx["content"].split("\n"):
            lines.append(f"    | {line}")

    if result.get("error"):
        lines.append(f"错误: {result['error']}")

    lines.append("")

    return "\n".join(lines)


def process_anchor_data(
    anchor_path,
    data_dir: str,
    output_dir: str,
    verbose: bool = False,
    unchanged_file_paths: set = None,
    anchor_files: list = None,
) -> dict:
    # 查找所有 JSON 文件
    # 如果提供了 anchor_files（已过滤），直接构造路径；否则扫描目录
    if anchor_files is None:
        json_files = find_json_files(anchor_path)
    else:
        json_files = [Path(anchor_path) / f for f in anchor_files if f.endswith(".json")]
    if not json_files:
        warning("未找到 JSON 文件")
        return

    info(f"找到 {len(json_files)} 个 JSON 文件")

    # 路径归一化辅助函数（原截断逻辑已不适用新路径格式）
    _truncate_keywords = []

    def _truncate_path(path: str | list) -> str:
        """路径归一化：统一使用正斜杠，兼容 Windows/Linux/macOS。

        兼容 str 和 list 两种输入格式（多架构文件聚合时 file_path 可能为 list）：
        - str: 直接归一化返回
        - list: 归一化每个元素，返回第一个元素（用于 unchanged_file_paths 集合匹配）
        """
        if isinstance(path, list):
            normalized = [p.replace("\\", "/") for p in path]
            return normalized[0] if normalized else ""
        else:
            return path.replace("\\", "/")

    all_results = []
    skipped_unchanged = 0

    # 处理每个 JSON 文件
    for json_file in json_files:
        # 读取 JSON 文件
        data = read_json_file(json_file)
        if data is None:
            continue

        # 提取 note 类型的 anchor
        anchors = extract_note_anchors(data)
        # if not anchors:
        #     continue

        info(f"  找到 {len(anchors)} 个 note 类型的 anchor")
        from copy import deepcopy

        update_anchor = deepcopy(data)
        update_anchor["anchor_tuple"] = []
        # 处理每个 anchor

        for anchor in anchors:
            # Level 2: 如果 anchor 的源文件未变更，跳过该 anchor
            if unchanged_file_paths is not None:
                file_path = anchor.get("file_path", "")
                if file_path:
                    truncated = _truncate_path(file_path)
                    if truncated in unchanged_file_paths:
                        skipped_unchanged += 1
                        continue

            result, an_ev_tuple = process_anchor(anchor, data_dir, verbose)
            result["json_file"] = json_file.name
            all_results.append(result)
            if an_ev_tuple:
                update_anchor["anchor_tuple"].append(an_ev_tuple)

        if output_dir:
            json_file_name = Path(json_file).name
            output_note_dir = os.path.join(output_dir, f"note_{json_file_name}")
            with open(output_note_dir, "w", encoding="utf-8") as f:
                json.dump(update_anchor, f, ensure_ascii=False, indent=2)
            info(f"结果已保存到: {output_dir}")

    if skipped_unchanged > 0:
        info(f"已跳过 {skipped_unchanged} 个未变更文件的 note anchor")

    info(f"共处理 {len(all_results)} 个 anchor")

    # 输出结果
    output = []
    for result in all_results:
        output.append(format_result(result))

    output_text = "\n".join(output)

    # 统计信息
    found_count = sum(1 for r in all_results if r.get("found"))
    info(f"统计: 找到 {found_count}/{len(all_results)} 个")
    return all_results
