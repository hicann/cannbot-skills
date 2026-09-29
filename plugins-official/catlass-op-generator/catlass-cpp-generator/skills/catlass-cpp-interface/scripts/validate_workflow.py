# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate Linear Attention workflow state and recovery routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKFLOW_ID = "catlass-linear-attention-v1"
ALGORITHM_FAMILY = "linear_attention"
TARGET_ARCHITECTURES = ("pending", "atlas_a2_a3", "ascend950")
STAGES = ("interface", "reference", "design", "implementation", "validation", "complete")
CONTRACT_STATES = ("provisional", "frozen")
ISSUE_TO_RESUME = {
    "precision_debug": "implementation",
    "design_issue": "design",
    "reference": "reference",
    "interface": "interface",
    "performance_optimize": "validation",
}


def validate_workflow(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("workflow_id") != WORKFLOW_ID:
        errors.append(f"workflow_id must be {WORKFLOW_ID}")
    if data.get("algorithm_family") != ALGORITHM_FAMILY:
        errors.append(f"algorithm_family must be {ALGORITHM_FAMILY}")

    stage = data.get("stage")
    if stage not in STAGES:
        errors.append(f"stage must be one of: {', '.join(STAGES)}")

    target_architecture = data.get("target_architecture")
    if target_architecture not in TARGET_ARCHITECTURES:
        errors.append(
            "target_architecture must be one of: "
            + ", ".join(TARGET_ARCHITECTURES)
        )
    elif stage in STAGES[1:] and target_architecture == "pending":
        errors.append(f"{stage} requires a frozen target_architecture")

    for field in ("operator_contract", "golden_contract"):
        if data.get(field) not in CONTRACT_STATES:
            errors.append(f"{field} must be provisional or frozen")

    issue_type = data.get("issue_type")
    resume_from = data.get("resume_from")
    if issue_type is not None and issue_type not in ISSUE_TO_RESUME:
        errors.append(f"unsupported issue_type: {issue_type}")
    if resume_from is not None and resume_from not in STAGES[:-1]:
        errors.append(f"resume_from must be one of: {', '.join(STAGES[:-1])}")
    if issue_type in ISSUE_TO_RESUME and resume_from != ISSUE_TO_RESUME[issue_type]:
        errors.append(
            f"issue_type={issue_type} requires resume_from={ISSUE_TO_RESUME[issue_type]}"
        )
    if issue_type in ISSUE_TO_RESUME and stage != ISSUE_TO_RESUME[issue_type]:
        errors.append(
            f"issue_type={issue_type} requires stage={ISSUE_TO_RESUME[issue_type]}"
            f"; current stage={stage}"
        )
    if issue_type is None and resume_from is not None:
        errors.append("resume_from requires issue_type")

    scope = data.get("validation_scope")
    if scope not in ("full", "precision_targeted"):
        errors.append("validation_scope must be full or precision_targeted")
    if scope == "precision_targeted" and issue_type != "precision_debug":
        errors.append("precision_targeted requires issue_type=precision_debug")
    if issue_type == "design_issue" and scope != "full":
        errors.append("design_issue must reset validation_scope to full")

    if stage in ("design", "implementation", "validation", "complete"):
        if data.get("operator_contract") != "frozen":
            errors.append(f"{stage} requires operator_contract=frozen")
        if data.get("golden_contract") != "frozen":
            errors.append(f"{stage} requires golden_contract=frozen")
    if stage == "complete" and (issue_type is not None or resume_from is not None):
        errors.append("complete stage cannot contain an unresolved issue or recovery route")
    if stage == "complete" and scope != "full":
        errors.append("complete stage requires validation_scope=full")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    args = parser.parse_args()
    if not args.workflow.is_file():
        print(f"FAIL\n- workflow file not found: {args.workflow}")
        return 1
    try:
        data = json.loads(args.workflow.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL\n- invalid workflow JSON: {exc}")
        return 1
    if not isinstance(data, dict):
        print("FAIL\n- workflow JSON must be an object")
        return 1
    errors = validate_workflow(data)
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: workflow state and recovery route are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
