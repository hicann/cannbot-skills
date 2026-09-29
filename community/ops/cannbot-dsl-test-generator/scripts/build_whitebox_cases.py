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

"""Validate DESIGN-backed whitebox rows in testcase.csv."""

from __future__ import annotations

import logging
import sys
import argparse
import json
import re

from casebook import read_casebook
from recipe_validation import validate_tensor, validate_tensor_list
from generator_support import (
    add_io_arguments,
    artifact_paths,
    design_path as input_design_path,
    spec_path as input_spec_path,
    load_yaml_mapping,
)

BRANCH_ID = re.compile(
    r"^B(?:[A-Za-z0-9][A-Za-z0-9_-]{0,63}|-[A-Za-z0-9][A-Za-z0-9-]{0,63})$"
)


LOGGER = logging.getLogger(__name__)


def fail(message):
    raise ValueError(message)


def validated_branches(work):
    spec_path, design_path = input_spec_path(work), input_design_path(work)
    spec = load_yaml_mapping(spec_path)
    op = (spec.get("op") or {}).get("name") or ""
    if not design_path.is_file():
        fail("a design Markdown file is required")
    rows = [row for row in read_casebook(work, op) if row["sheet"] == "whitebox"]
    if not rows:
        fail("whitebox testcase.csv needs branches")
    design = design_path.read_text(encoding="utf-8")
    styles = {
        row["note"] or "return_value" for row in rows if row["status"] == "active"
    }
    if len(styles) != 1 or not styles <= {"return_value", "dst_args"}:
        fail("whitebox call_style must be consistent")
    call_style = styles.pop()
    data = {"branches": [parse_branch_row(row) for row in rows]}
    branches, seen = [], set()
    for raw in data["branches"]:
        branches.append(validate_branch(raw, seen, spec, design, call_style))
    if not any(item["status"] == "draft" for item in branches):
        fail("no runnable whitebox branch")
    payload = {
        "operator": op,
        "phase": "draft",
        "branch_source": "testcase.csv checked against supplied design document",
        "call_style": call_style,
        "branches": branches,
    }
    return payload


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_arguments(parser, design=True, units=False)
    args = parser.parse_args()
    work = artifact_paths(args)
    data = validated_branches(work)
    LOGGER.info(f"validated {len(data['branches'])} whitebox branches")


def parse_branch_row(row):
    if not row["case_id"].startswith("WB-") or row["branch_ids"] != row["case_id"][3:]:
        fail(f"{row['case_id']}: whitebox case and branch ID disagree")
    try:
        suggested = json.loads(row["inputs_json"]) if row["status"] == "active" else {}
        outputs = (
            json.loads(row["output_json"] or "{}") if row["status"] == "active" else {}
        )
    except ValueError as exc:
        fail(f"{row['case_id']}: invalid JSON: {exc}")
    return {
        "id": row["case_id"][3:],
        "design_ref": row["design_ref"],
        "condition": row["condition"],
        "suggested_input": suggested,
        "output_tensors": outputs,
        "status": "draft" if row["status"] == "active" else "excluded",
        "exclude_reason": row["exclude_reason"],
    }


def validate_branch(raw, seen, spec, design, call_style):
    if not isinstance(raw, dict):
        fail("each branch must be a mapping")
    branch_id = raw.get("id")
    if (
        not isinstance(branch_id, str)
        or not BRANCH_ID.fullmatch(branch_id)
        or branch_id in seen
    ):
        fail(f"invalid or duplicate branch id: {branch_id}")
    seen.add(branch_id)
    status = raw.get("status", "draft")
    if status not in ("draft", "excluded"):
        fail(f"{branch_id}: invalid status")
    if status == "excluded" and not raw.get("exclude_reason"):
        fail(f"{branch_id}: excluded branch needs a reason")
    suggested = raw.get("suggested_input") or {}
    outputs = raw.get("output_tensors") or {}
    if status == "draft":
        validate_branch_inputs(branch_id, suggested, outputs, spec, call_style)
    branch = {
        "branch_id": branch_id,
        "design_ref": str(raw.get("design_ref") or ""),
        "condition": str(raw.get("condition") or ""),
        "suggested_input": suggested if status == "draft" else {},
        "output_tensors": outputs if status == "draft" else {},
        "status": status,
        "exclude_reason": str(raw.get("exclude_reason") or ""),
    }
    if not branch["design_ref"] or not branch["condition"]:
        fail(f"{branch_id}: design_ref and condition are required")
    if branch["design_ref"] not in design:
        fail(f"{branch_id}: design_ref not found in supplied design document")
    return branch


def validate_branch_inputs(branch_id, suggested, outputs, spec, call_style):
    input_meta = {item["name"]: item for item in spec.get("inputs") or []}
    attributes = {item["name"] for item in spec.get("attributes") or []}
    required_attrs = {
        item["name"] for item in spec.get("attributes") or [] if "default" not in item
    }
    output_names = {item["name"] for item in spec.get("outputs") or []}
    if (
        not isinstance(suggested, dict)
        or not suggested
        or set(suggested) - (set(input_meta) | attributes)
    ):
        fail(f"{branch_id}: suggested_input must use public spec names")
    if {name for name, meta in input_meta.items() if not meta.get("optional")} - set(
        suggested
    ):
        fail(f"{branch_id}: suggested_input must cover all spec inputs")
    if required_attrs - set(suggested):
        fail(f"{branch_id}: suggested_input must cover required attributes")
    for name, item in suggested.items():
        if name in input_meta and input_meta[name].get("role", "tensor") == "tensor":
            validate_tensor(branch_id, name, item, input_meta[name], None)
        elif name in input_meta and input_meta[name].get("role") == "tensor_list":
            validate_tensor_list(branch_id, name, item, input_meta[name], None)
    if call_style == "dst_args":
        if set(outputs) != output_names:
            fail(f"{branch_id}: output_tensors must cover spec outputs")
        for name, item in outputs.items():
            validate_tensor(
                branch_id, name, item, {"dtype_set": [item.get("dtype")]}, None
            )
    elif outputs:
        fail(f"{branch_id}: output_tensors only applies to dst_args")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
