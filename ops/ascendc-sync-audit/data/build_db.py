#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""从 sync_cases.jsonl（source of truth）重建 sync_cases.db。

用法:
  python3 data/build_db.py            # jsonl → db（覆盖重建）
  python3 data/build_db.py --export   # db → jsonl（反向导出，编辑 db 后同步明文）
  python3 data/build_db.py --verify   # 校验 jsonl 与 db 行数/内容一致

修改 case 数据的流程：编辑 sync_cases.jsonl → 运行本脚本重建 db → --verify 确认一致。
"""

import argparse
import json
import logging
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, 'sync_cases.jsonl')
DB = os.path.join(HERE, 'sync_cases.db')

_LOGGER = logging.getLogger('build_db')
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False
if not _LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter('%(message)s'))
    _LOGGER.addHandler(_handler)


def _fail(msg: str) -> None:
    """以带信息异常终止；由 main 捕获后以非零退出码结束（G.ERR.11 允许入口退出）。"""
    raise RuntimeError(msg)

COLUMNS = ['id', 'repo', 'issue_number', 'pr_id', 'pr_title', 'pr_desc', 'is_merged',
           'file_path', 'base_sha', 'head_sha', 'diff_patch', 'keywords', 'sync_apis',
           'index_vars', 'fix_type', 'fix_summary']

SCHEMA = """CREATE TABLE sync_cases (
    id INTEGER PRIMARY KEY,
    repo TEXT,
    issue_number TEXT,
    pr_id TEXT,
    pr_title TEXT,
    pr_desc TEXT,
    is_merged TEXT,
    file_path TEXT,
    base_sha TEXT,
    head_sha TEXT,
    diff_patch TEXT,
    keywords TEXT,
    sync_apis TEXT,
    index_vars TEXT,
    fix_type TEXT,
    fix_summary TEXT
)"""


def load_jsonl():
    rows = []
    with open(JSONL, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                _fail(f"错误: {JSONL}:{ln} JSON 解析失败: {e}")
    return rows


def build():
    if not os.path.exists(JSONL):
        _fail(f"错误: 未找到 {JSONL}")
    rows = load_jsonl()
    tmp = DB + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute(SCHEMA)
    con.executemany(
        f"INSERT INTO sync_cases ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
        [[r.get(c) for c in COLUMNS] for r in rows])
    con.commit()
    con.close()
    os.replace(tmp, DB)
    _LOGGER.info("重建完成: %s（%d 条 case）", DB, len(rows))


def export():
    if not os.path.exists(DB):
        _fail(f"错误: 未找到 {DB}")
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM sync_cases ORDER BY id").fetchall()
    with open(JSONL, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False) + '\n')
    _LOGGER.info("导出完成: %s（%d 条 case）", JSONL, len(rows))


def verify():
    rows = load_jsonl()
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    db_rows = [dict(r) for r in con.execute("SELECT * FROM sync_cases ORDER BY id")]
    if len(rows) != len(db_rows):
        _fail(f"不一致: jsonl {len(rows)} 条 vs db {len(db_rows)} 条")
    for i, (a, b) in enumerate(zip(rows, db_rows)):
        for c in COLUMNS:
            if a.get(c) != b.get(c):
                _fail(f"不一致: 第 {i + 1} 条 case（id={a.get('id')}）字段 {c} 不同")
    _LOGGER.info("一致: jsonl 与 db 均为 %d 条 case，逐字段相同", len(rows))


def main():
    p = argparse.ArgumentParser(description='sync_cases.jsonl ↔ sync_cases.db 构建/导出/校验')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--export', action='store_true', help='db → jsonl 反向导出')
    g.add_argument('--verify', action='store_true', help='校验 jsonl 与 db 一致')
    args = p.parse_args()
    try:
        if args.export:
            export()
        elif args.verify:
            verify()
        else:
            build()
    except RuntimeError as exc:
        _LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == '__main__':
    main()
