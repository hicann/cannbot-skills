# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Select legacy or dedicated (Linear Attention / BSA Arch22) workflow without name-based guessing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WORKFLOW_ID = "catlass-linear-attention-v1"
BSA_WORKFLOW_ID = "catlass-block-sparse-attention-arch22-v1"
LINEAR_FAMILY = "linear_attention"
BSA_FAMILY = "block_sparse_attention"
LEGACY_FAMILY = "legacy"
PENDING_FAMILY = "pending"
DEDICATED_WORKFLOWS = {LINEAR_FAMILY: WORKFLOW_ID, BSA_FAMILY: BSA_WORKFLOW_ID}
SUPPORTED_FAMILIES = {LINEAR_FAMILY, BSA_FAMILY, LEGACY_FAMILY, PENDING_FAMILY}
SAFE_OPERATOR = re.compile(r"^[a-z][a-z0-9_]*$")


def _result(route: str, reason: str, marker: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {"route": route, "reason": reason}
    if marker is not None:
        value["marker"] = marker
    return value


def select(
    workspace: Path, operator_name: str, algorithm_family: str | None
) -> dict[str, object]:
    if not SAFE_OPERATOR.fullmatch(operator_name):
        return _result("blocked", "operator_name must be a safe snake_case name")

    operator_dir = workspace.resolve() / "operators" / operator_name
    marker = operator_dir / "docs" / "workflow.json"

    if marker.exists():
        if marker.is_symlink():
            return _result(
                "blocked", "workflow marker must not be a symlink", str(marker)
            )
        if not marker.is_file():
            return _result(
                "blocked", "workflow marker is not a regular file", str(marker)
            )
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _result("blocked", f"invalid workflow marker: {exc}", str(marker))
        if not isinstance(data, dict):
            return _result(
                "blocked", "workflow marker must be a JSON object", str(marker)
            )
        workflow_id = data.get("workflow_id")
        family = data.get("algorithm_family")
        if family in DEDICATED_WORKFLOWS and workflow_id == DEDICATED_WORKFLOWS[family]:
            return _result(family, "existing dedicated workflow marker", str(marker))
        if workflow_id is not None:
            return _result("blocked", "unsupported workflow marker", str(marker))
        return _result(
            "legacy", "existing project without dedicated marker", str(marker)
        )

    if operator_dir.exists():
        return _result("legacy", "existing project takes legacy precedence")

    if algorithm_family is None:
        return _result(
            "pending", "new project requires mathematical algorithm classification"
        )
    normalized = algorithm_family.strip().lower()
    if normalized == PENDING_FAMILY or not normalized:
        return _result(
            "pending", "new project requires mathematical algorithm classification"
        )
    if normalized not in SUPPORTED_FAMILIES:
        return _result("blocked", f"unsupported algorithm family: {algorithm_family}")
    if normalized in DEDICATED_WORKFLOWS:
        return _result(
            normalized,
            "intake classified the mathematical algorithm as a dedicated-workflow family",
        )
    return _result("legacy", "intake classified the mathematical algorithm as legacy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--operator-name", required=True)
    parser.add_argument("--algorithm-family", choices=tuple(sorted(SUPPORTED_FAMILIES)))
    args = parser.parse_args()
    result = select(args.workspace, args.operator_name, args.algorithm_family)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["route"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
