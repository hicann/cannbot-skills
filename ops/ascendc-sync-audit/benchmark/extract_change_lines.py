#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""从修复前后配对/PR diff 提取缺陷行号（defect_lines），写入 labels.json。

来源一（real_pairs）：defect_pr_xxx.cpp ↔ correct_pr_xxx.cpp 同名配对，
   统一 diff 的「- 行」即修复前文件被修改行号（同步修复锚点）。
来源二（real_pr）：从 data/sync_cases.jsonl 按 pr_id + file_path 匹配 PR diff，
   解析 diff_patch 中各 hunk 的「- 行」映射到仓库文件（含提取头注释偏移）。
过滤规则：仅含注释/空白/花括号的变更行不计为缺陷锚点。

用法（在 ops/ascendc-sync-audit 目录下）：
  python3 benchmark/extract_change_lines.py [--write]
（不带 --write 时仅打印统计，不落盘）
"""

import argparse
import difflib
import json
import logging
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sync_logging import init_logging  # noqa: E402

TESTS = ROOT / "tests"
LABELS = TESTS / "labels.json"
CASES = ROOT / "data" / "sync_cases.jsonl"

LOGGER = logging.getLogger('extract_change_lines')
STDERR_LOGGER = logging.getLogger('extract_change_lines.stderr')
init_logging(LOGGER, STDERR_LOGGER)

# extract 脚本注入的头注释行（pr_pair/pr_full 提取时写入），不算缺陷锚点
HEADER_NOISE = re.compile(r"^\s*(//|/\*|\*|/\* )")

# unified diff 元信息行前缀（不占用目标文件行号）
DIFF_META_PREFIXES = ('diff --git', 'index ', '+++', 'new file', 'rename')


def _is_diff_meta_line(line: str) -> bool:
    return line.startswith(DIFF_META_PREFIXES)


def parse_unified_removed_lines(diff_lines):
    """从 unified diff 行列表提取修复前文件被修改/删除的行号。

    跳过 diff 元信息行（diff --git / index / --- / +++ / new file 等），
    这些行不占用目标文件行号。
    """
    removed = []
    cur = 0
    for line in diff_lines:
        if _is_diff_meta_line(line):
            continue
        if line.startswith('@@'):
            m = re.search(r'-(\d+)(?:,\d+)?', line)
            if m:
                cur = int(m.group(1))
            continue
        if line.startswith('-') and not line.startswith('---'):
            removed.append(cur)
            cur += 1
        elif line.startswith('+') and not line.startswith('+++'):
            continue
        else:
            cur += 1
    return removed


def filter_noise(lines, removed):
    """剔除只含注释/空白/括号的变更行。"""
    keep = []
    for ln in removed:
        if 1 <= ln <= len(lines):
            text = lines[ln - 1].strip()
            if not text or HEADER_NOISE.match(text) or text in ('{', '}'):
                continue
            keep.append(ln)
    return keep


def extract_from_pairs():
    """real_pairs 配对 diff：返回 {文件名: defect_lines}。"""
    result = {}
    pair_dir = TESTS / "real_pairs"
    if not pair_dir.is_dir():
        return result
    for defect in sorted(pair_dir.glob("defect_pr_*.cpp")):
        correct = pair_dir / ("correct" + defect.name[len("defect"):])
        if not correct.exists():
            continue
        d = defect.read_text(errors="ignore").splitlines()
        c = correct.read_text(errors="ignore").splitlines()
        diff = difflib.unified_diff(d, c, lineterm="", n=0)
        removed = parse_unified_removed_lines(list(diff))
        lines = filter_noise(d, removed)
        result[defect.name] = lines
    return result


def _normalize(text: str) -> str:
    """内容指纹：去空白后的大写化片段，用于行级对齐校验。"""
    return re.sub(r"\s+", "", text).upper()


def _parse_pr_header(text_lines: list, fallback_name: str):
    """从 real_pr 文件头注释解析 (pr_id, file_path)。"""
    if len(text_lines) < 6:
        return '', ''
    m = re.search(r"pr#(\d+)", text_lines[3] or "")
    pr_id = m.group(1) if m else ""
    if not pr_id:
        m2 = re.match(r"pr_(\d+)_", fallback_name)
        pr_id = m2.group(1) if m2 else ""
    mf = re.search(r"FILE: (\S+)", text_lines[4] or "")
    return pr_id, (mf.group(1) if mf else "")


def _align_case_diff(diff_patch: str, text_lines: list, header_offset: int) -> list:
    """diff 变更行与 real_pr 文件指纹对齐校验，返回缺陷行号（不可靠则空）。"""
    removed = parse_unified_removed_lines(diff_patch.splitlines())
    removed = filter_noise(text_lines, removed)
    if not removed:
        return []
    diff_ctx = _normalize(diff_patch)
    for ln in removed:
        target = ln + header_offset
        if 1 <= target <= len(text_lines):
            if _normalize(text_lines[target - 1]) not in diff_ctx:
                return []
    return [ln + header_offset for ln in removed]


def extract_from_case_diff():
    """real_pr：据文件头 // FILE 路径 + // SOURCE pr# 匹配 case diff_patch。

    diff 的「- 行」是修复前行内容，应与 real_pr 文件（含 5 行提取头注释）
    对应行指纹一致；指纹全部对齐才采用（防内容被改写导致行号错位）。
    """
    if not CASES.exists():
        return {}
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines()]
    by_pr = {}
    for case in cases:
        by_pr.setdefault(str(case.get("pr_id")), []).append(case)
    result = {}
    pr_dir = TESTS / "real_pr"
    if not pr_dir.is_dir():
        return result
    for defect in sorted(pr_dir.glob("pr_*.cpp")):
        text_lines = defect.read_text(errors="ignore").splitlines()
        pr_id, file_path = _parse_pr_header(text_lines, defect.name)
        if not pr_id or not file_path:
            continue
        for case in by_pr.get(pr_id, []):
            if (case.get("file_path") or "") != file_path:
                continue
            aligned = _align_case_diff(case.get("diff_patch") or "", text_lines, 5)
            if aligned:
                result[defect.name] = aligned
                break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="提取缺陷行号并写入 labels.json")
    parser.add_argument("--write", action="store_true", help="写回 labels.json")
    args = parser.parse_args()

    pairs = extract_from_pairs()
    full = extract_from_case_diff()
    LOGGER.info("real_pairs 配对提取: %d 个文件（均含 defect_lines 的行号）", len(pairs))
    LOGGER.info("real_pr  案例 diff 提取: %d 个文件", len(full))

    if not args.write:
        sample = next(iter(pairs.items())) if pairs else ("", [])
        LOGGER.info("样例: %s -> 缺陷行 %s", sample[0], sample[1])
        LOGGER.info("（加 --write 写回 labels.json）")
        return 0

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    updated = 0
    for fname, lines in {**pairs, **full}.items():
        for key in labels["files"]:
            if key.endswith("/" + fname):
                labels["files"][key]["defect_lines"] = lines
                updated += 1
                break
    LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("已写回 labels.json：%d 个文件获得 defect_lines", updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())