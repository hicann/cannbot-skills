#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""校验 ops-knowledge-optimization-ingest 编译出的单算子劣化实测优化点库。

目标 runbook：`<root>/runbooks/operator-optimization/single-op-degradation.md`
（本 skill 自带此校验；不复用 ops-knowledge-vv-ingest 的脚本，因其 runbook 名硬编码为
vv-fusion-common.md）。挖矿产出的 curated 层只有该 runbook；`ops/<cat>/<op>.md` 算子卡
非本 skill 产出，故可选——若通过 --ops 传入则一并校验其对 OPT 的引用与锚点闭合。

用法::

    python3 validate_degradation_knowledge.py [--knowledge-root DIR] [--ops A.md B.md ...]

`--knowledge-root` 缺省 = CANNBOT_KNOWLEDGE_ROOT / KNOWLEDGE_ROOT / 当前工作目录。

退出码：0 = 全过（仅 WARN 不致命）；1 = 有 HARD 错误；2 = 找不到目标 runbook。

检查项：
  [HARD] 悬空锚          —— runbook / ops 里 `](rel#锚点)` 的锚点须在目标文件有同名标题
  [HARD] 引用未定义       —— ops 引的 OPT-N/CT-N/AP-N 须在 runbook 有 `## OPT-N` 等
  [HARD] 标题未瘦身       —— `## OPT/CT/AP` 标题行不得含【标签】或破锚符号 `` `<>``
  [HARD] frontmatter 缺失 —— runbook 缺 optimization_runbook profile 必备字段
  [WARN] OPT 字段         —— OPT 缺 摘要/触发/优化维度/置信度/实测收益
  [WARN] 坏实践必填        —— 某 OPT 无「坏实践（反例）」字段（无则写「待补充」）
  [WARN] 已知实例          —— 某 OPT 无「已知实例」字段（知识库无算子卡时写「待补充」）
  [WARN] CT 字段          —— CT-* 缺 症状/根因/规避/预防
  [WARN] # 相关托管块      —— runbook 缺文末 okf:related 托管块
  [WARN] 导航 index       —— 缺 runbooks/operator-optimization/index.md
"""
import argparse
import glob
import logging
import os
import re
import sys

RUNBOOK_REL = os.path.join("runbooks", "operator-optimization", "single-op-degradation.md")
INDEX_REL = os.path.join("runbooks", "operator-optimization", "index.md")

# optimization_runbook frontmatter 必备字段
FM_FIELDS = ["schema_version", "kind", "type", "source_family",
             "title", "description", "tags", "created_at", "updated_at"]

HARD, WARN = [], []


def _emit(message, *args):
    """Write one unprefixed log record to stdout, preserving the CLI contract."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(__name__, logging.INFO, "", 0, message, args, None)
    handler.handle(record)
    handler.close()


def _read_text(path):
    with open(path, encoding="utf-8") as source_file:
        return source_file.read()


def hard(msg):
    HARD.append(msg)


def warn(msg):
    WARN.append(msg)


def headings(path):
    hs = set()
    for line in _read_text(path).splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            hs.add(m.group(1))
    return hs


def extract_links(text):
    """返回所有 markdown 链接的 url（含尖括号 <...> 与裸形式）。"""
    urls = re.findall(r"\]\(<([^>]+)>\)", text)
    urls += re.findall(r"\]\((?!<)([^)\s]+)\)", text)
    return urls


def opt_blocks(runbook_text):
    """把 runbook 按 `## ` 切块，返回 (id, block) 列表（id 形如 OPT-3 / CT-1 / AP-2）。"""
    out = []
    for block in re.split(r"\n## ", runbook_text):
        match = re.match(r"(OPT-\d+|CT-\d+|AP-\d+)", block)
        if match:
            out.append((match.group(1), block))
    return out


def _parse_args():
    default_root = os.path.abspath(
        os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
        or os.environ.get("KNOWLEDGE_ROOT")
        or os.getcwd()
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge-root", "--root", dest="root", default=default_root,
        help="知识根目录（含 runbooks/operator-optimization/single-op-degradation.md），缺省=仓库根",
    )
    parser.add_argument(
        "--ops", nargs="*", default=None,
        help="额外校验这些算子卡对 OPT 的引用/锚点（相对 root 或绝对路径）；缺省=不校验 ops",
    )
    return parser.parse_args()


def _resolve_ops(root, requested_ops):
    if not requested_ops:
        return []
    return sorted(
        os.path.abspath(path if os.path.isabs(path) else os.path.join(root, path))
        for path in requested_ops
    )


def _skip_anchor_check(url):
    return "gitcode.com" in url or url.startswith("http") or "#" not in url


def _check_anchors(root, md_files, headings_by_path):
    for source in md_files:
        base = os.path.dirname(source)
        for url in extract_links(_read_text(source)):
            if _skip_anchor_check(url):
                continue
            path, anchor = url.split("#", 1)
            target = source if path == "" else os.path.normpath(os.path.join(base, path))
            if target in headings_by_path and anchor not in headings_by_path[target]:
                hard("悬空锚: [%s] -> %s #%s" % (
                    os.path.relpath(source, root), os.path.relpath(target, root), anchor))


def _check_definitions(root, ops_files, runbook_text):
    definitions = set(re.findall(r"^## (OPT-\d+|AP-\d+|CT-\d+)", runbook_text, re.M))
    for operator_file in ops_files:
        references = set(re.findall(r"OPT-\d+|AP-\d+|CT-\d+", _read_text(operator_file)))
        for reference in sorted(references - definitions):
            hard("引用未定义: %s 引用 %s，但 runbook 无 `## %s`" % (
                os.path.relpath(operator_file, root), reference, reference))


def _check_titles(runbook_text):
    for line in runbook_text.splitlines():
        if not re.match(r"^## (OPT-|AP-|CT-)", line):
            continue
        if "【" in line:
            hard("标题未瘦身（含【标签】）: %s" % line.strip())
        if "`" in line or "<" in line or ">" in line:
            hard("标题含破锚符号: %s" % line.strip())


def _check_frontmatter(root, runbook):
    text = _read_text(runbook)
    relpath = os.path.relpath(runbook, root)
    if not text.startswith("---"):
        warn("%s: 缺 YAML frontmatter（应含 %s）" % (relpath, "/".join(FM_FIELDS)))
        return
    parts = text.split("---", 2)
    frontmatter = parts[1] if len(parts) >= 3 else ""
    for field in FM_FIELDS:
        if not re.search(r"^%s\s*:" % field, frontmatter, re.M):
            warn("%s: frontmatter 缺字段 `%s`" % (relpath, field))
    if not re.search(r"^type\s*:\s*optimization_runbook", frontmatter, re.M):
        warn("%s: frontmatter type 应为 optimization_runbook" % relpath)


def _check_opt_fields(runbook_text):
    for opt_id, block in opt_blocks(runbook_text):
        if not opt_id.startswith("OPT-"):
            continue
        for field in ("摘要", "触发", "优化维度", "置信度", "实测收益"):
            if "- **%s**" % field not in block:
                warn("标签块缺字段: runbook %s 缺「%s」" % (opt_id, field))
        if "坏实践" not in block:
            warn("坏实践缺失: runbook %s 无「坏实践（反例）」字段（无则写「待补充」）" % opt_id)
        if "已知实例" not in block:
            warn("已知实例缺失: runbook %s 无「已知实例」字段（无算子卡时写「待补充」）" % opt_id)


def _check_ct_fields(runbook_text):
    for ct_id, block in opt_blocks(runbook_text):
        if not ct_id.startswith("CT-"):
            continue
        for field in ("症状", "根因", "规避", "预防"):
            if field not in block:
                warn("CT 缺字段: runbook %s 缺「%s」" % (ct_id, field))


def _check_related(root, runbook):
    text = _read_text(runbook)
    relpath = os.path.relpath(runbook, root)
    has_block = "<!-- okf:related:start -->" in text
    has_heading = re.search(r"^#\s+相关\s*$", text, re.M) is not None
    if not has_block or not has_heading:
        warn("%s: 缺文末 `# 相关` 托管块（okf:related:start/end + `# 相关`）" % relpath)


def _report(root, runbook, ops_files):
    _emit("== 校验根目录: %s ==", root)
    ops_rel = ", ".join(os.path.relpath(path, root) for path in ops_files) or "(无)"
    _emit("   runbook: %s；ops: %s", os.path.relpath(runbook, root), ops_rel)
    if HARD:
        _emit("\n[HARD 错误 × %d]", len(HARD))
        for message in HARD:
            _emit("  ✗ %s", message)
    if WARN:
        _emit("\n[WARN 提示 × %d]", len(WARN))
        for message in WARN:
            _emit("  ! %s", message)
    if not HARD and not WARN:
        _emit("\n✅ 全部通过（无 HARD、无 WARN）")
    elif not HARD:
        _emit("\n✅ 无 HARD 错误（%d 条 WARN，请人工确认是否误报）", len(WARN))
    else:
        _emit("\n❌ %d 条 HARD 错误，须修正后重出", len(HARD))
    return 1 if HARD else 0


def main():
    args = _parse_args()
    root = os.path.abspath(args.root)
    runbook = os.path.join(root, RUNBOOK_REL)
    if not os.path.exists(runbook):
        _emit("[FATAL] 找不到 %s", runbook)
        return 2

    ops_files = [path for path in _resolve_ops(root, args.ops) if os.path.exists(path)]
    md_files = [runbook] + ops_files
    headings_by_path = {path: headings(path) for path in md_files}
    runbook_text = _read_text(runbook)

    _check_anchors(root, md_files, headings_by_path)
    _check_definitions(root, ops_files, runbook_text)
    _check_titles(runbook_text)
    _check_frontmatter(root, runbook)
    _check_opt_fields(runbook_text)
    _check_ct_fields(runbook_text)
    _check_related(root, runbook)
    if not os.path.exists(os.path.join(root, INDEX_REL)):
        warn("缺 runbooks/operator-optimization/index.md 渐进导航（OKF）")
    return _report(root, runbook, ops_files)


if __name__ == "__main__":
    raise SystemExit(main())
