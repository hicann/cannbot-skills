#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Replace one CSV group from a JSON row array on stdin, preserving concurrent groups."""
import argparse
import csv
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path

from casebook import COLUMNS, SHEETS, validate_rows


def update_group(test_dir, sheet, incoming):
    test_dir = Path(test_dir)
    if sheet not in SHEETS or not isinstance(incoming, list):
        raise ValueError("expected a supported sheet and JSON row array")
    normalized = []
    for row in incoming:
        if not isinstance(row, dict) or set(row) - set(COLUMNS):
            raise ValueError("unknown CSV columns")
        item = {key: row.get(key, "") for key in COLUMNS}
        if any(not isinstance(value, str) for value in item.values()):
            raise ValueError("CSV cells must be strings; encode JSON recipes first")
        if item["sheet"] != sheet:
            raise ValueError("cannot update another group")
        normalized.append(item)
    validate_rows(normalized)
    test_dir.mkdir(parents=True, exist_ok=True)
    path = test_dir / "testcase.csv"
    # A stable sidecar inode is needed: the CSV inode changes on atomic replace.
    with (test_dir / ".testcase.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != COLUMNS:
                    raise ValueError("invalid testcase.csv header")
                existing = validate_rows(list(reader))
        rows = [row for row in existing if row["sheet"] != sheet] + normalized
        validate_rows(rows)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=test_dir, delete=False,
            ) as handle:
                tmp = handle.name
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sheet", choices=SHEETS, required=True)
    args = parser.parse_args()
    update_group(args.test_dir, args.sheet, json.load(sys.stdin))


if __name__ == "__main__":
    main()
