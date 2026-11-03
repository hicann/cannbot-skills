# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate the CATLASS C++ workflow routing state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALGORITHM_FAMILIES = (
    "general",
    "linear_attention",
    "block_sparse_attention",
    "sparse_flash_mla",
    "block_sparse_attention_grad",
    "flash_attention_grad",
    "flash_attention",
)
TARGET_ARCHITECTURES = {
    "atlas_a2_a3": "Atlas A2系列产品、Atlas A3系列产品",
    "ascend950": "Ascend 950PR&950DT系列产品",
}
TARGET_ARCHITECTURE_VALUES = ("pending", *TARGET_ARCHITECTURES)
WORKFLOW_FIELDS = {
    "algorithm_family",
    "target_architecture",
    "stage",
    "issue_type",
}
STAGES = (
    "interface",
    "reference",
    "design",
    "implementation",
    "validation",
    "complete",
)
VALID_STATES = {
    ("interface", None),
    ("reference", None),
    ("design", None),
    ("design", "performance_optimize"),
    ("implementation", None),
    ("implementation", "precision_debug"),
    ("validation", None),
    ("complete", None),
}


def validate_workflow(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(WORKFLOW_FIELDS - data.keys())
    unexpected = sorted(data.keys() - WORKFLOW_FIELDS)
    if missing:
        errors.append(f"workflow is missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"workflow contains unsupported fields: {', '.join(unexpected)}")

    family = data.get("algorithm_family")
    if family not in ALGORITHM_FAMILIES:
        errors.append("algorithm_family must be a registered CATLASS C++ family")

    stage = data.get("stage")
    issue_type = data.get("issue_type")
    if (stage, issue_type) not in VALID_STATES:
        errors.append(
            "stage and issue_type must form a registered CATLASS C++ workflow state"
        )

    target_architecture = data.get("target_architecture")
    if target_architecture not in TARGET_ARCHITECTURE_VALUES:
        supported = "; ".join(
            f"{name} ({products})" for name, products in TARGET_ARCHITECTURES.items()
        )
        errors.append(f"target_architecture must be one of: pending; {supported}")
    elif stage != "interface" and target_architecture == "pending":
        errors.append(f"{stage} requires a selected target_architecture")
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
    print("PASS: workflow routing state is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
