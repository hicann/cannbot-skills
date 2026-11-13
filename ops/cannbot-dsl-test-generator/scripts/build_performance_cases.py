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

"""Validate user-supplied performance recipes; never generate cases or time kernels."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
from case_contract import case_calls

from build_blackbox_cases import required_refs, validate_case
from casebook import read_casebook
from generator_support import add_io_arguments, artifact_paths, load_yaml_mapping

PERF_ID = re.compile(r"^PERF-[A-Za-z0-9][A-Za-z0-9-]{0,63}$")


def validate_performance(rows, spec):
    refs, seen = required_refs(spec), set()
    for row in rows:
        if row["sheet"] != "performance":
            continue
        if row["status"] != "active":
            raise ValueError(f"{row['case_id']}: user performance cases cannot be excluded")
        if any(row[key] for key in ("unit_id", "obligation_ids", "branch_ids")):
            raise ValueError(f"{row['case_id']}: performance cases do not bind Units or branches")
        if row["expected"] not in ("match_golden", "match_golden_nan"):
            raise ValueError(f"{row['case_id']}: performance requires a numerical reference")
        raw = {
            "id": row["case_id"], "level": "L0", "purpose": row["note"],
            "spec_refs": [x.strip() for x in row["design_ref"].split(",") if x.strip()],
            "inputs": json.loads(row["inputs_json"]),
            "output_tensors": json.loads(row["output_json"] or "{}"),
            "seed": int(row["seed"] or "42"), "expected": row["expected"],
        }
        _, _, calls = case_calls(raw["inputs"], raw["seed"], json.loads(row["assertions_json"] or "[]"))
        for index, call in enumerate(calls):
            validate_case({**raw, "inputs": call["inputs"], "seed": call["seed"]},
                          spec, refs, seen if index == 0 else set(), id_pattern=PERF_ID)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_arguments(parser)
    args = parser.parse_args()
    paths = artifact_paths(args)
    spec = load_yaml_mapping(paths.spec)
    count = validate_performance(read_casebook(paths, spec["op"]["name"]), spec)
    print(f"validated {count} user performance cases" + (" (not provided)" if not count else ""))


if __name__ == "__main__":
    main()
