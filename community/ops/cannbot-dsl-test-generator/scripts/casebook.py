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

"""Read the single durable CANNBotDSL test design and execution casebook."""

import csv

from generator_support import ArtifactPaths, test_dir

SHEETS = ("blackbox", "whitebox", "tdd")
COLUMNS = (
    "sheet",
    "case_id",
    "source",
    "branch_ids",
    "unit_id",
    "inputs_json",
    "output_json",
    "seed",
    "expected",
    "assertions_json",
    "obligation_ids",
    "note",
    "status",
    "design_ref",
    "condition",
    "exclude_reason",
)


def read_casebook(work: ArtifactPaths, op: str) -> list[dict]:
    path = test_dir(work, op) / "testcase.csv"
    if not path.is_file():
        raise ValueError(f"missing {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"invalid testcase.csv columns: expected {COLUMNS}")
        rows = list(reader)
    keys = [(row["sheet"], row["case_id"]) for row in rows]
    if len(keys) != len(set(keys)) or any(not case_id for _, case_id in keys):
        raise ValueError("duplicate or empty sheet/case_id in testcase.csv")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed testcase.csv row")
        if row["sheet"] not in SHEETS or row["source"] != row["sheet"]:
            raise ValueError(f"invalid sheet/source for {row['case_id']}")
        if row["status"] not in ("active", "excluded"):
            raise ValueError(f"{row['case_id']}: status must be active or excluded")
        if row["status"] == "excluded" and (
            not row["exclude_reason"].strip() or row["inputs_json"].strip()
        ):
            raise ValueError(
                f"{row['case_id']}: excluded row needs a reason and no executable inputs"
            )
        if row["status"] == "active" and (
            not row["inputs_json"].strip() or row["exclude_reason"].strip()
        ):
            raise ValueError(
                f"{row['case_id']}: active row needs inputs and no exclusion reason"
            )
    return rows
