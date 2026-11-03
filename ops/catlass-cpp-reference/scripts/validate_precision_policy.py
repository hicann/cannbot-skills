# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate the schema of a CATLASS precision policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "precision_policy_version",
    "global_matched_ratio",
    "region_matched_ratio",
    "critical_region_matched_ratio",
    "strict_ulp",
    "dtype",
)
DTYPE_FIELDS = ("atol", "rtol", "max_abs_limit", "max_ulp_limit")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate(policy_path: Path) -> list[str]:
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read precision policy JSON: {policy_path}: {exc}"]
    if not isinstance(policy, dict):
        return ["precision policy must be a JSON object"]

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in policy:
            errors.append(f"precision policy missing required field: {field}")
    version = policy.get("precision_policy_version")
    if not isinstance(version, str) or not version.strip():
        errors.append("precision_policy_version must be a non-empty string")
    for field in (
        "global_matched_ratio",
        "region_matched_ratio",
        "critical_region_matched_ratio",
    ):
        value = policy.get(field)
        if not _finite_number(value) or not 0 <= value <= 1:
            errors.append(f"{field} must be a finite number in [0, 1]")
    if not isinstance(policy.get("strict_ulp"), bool):
        errors.append("strict_ulp must be a boolean")

    dtype = policy.get("dtype")
    if not isinstance(dtype, dict) or not dtype:
        errors.append("dtype must be a non-empty object")
    else:
        for name, config in dtype.items():
            if not isinstance(name, str) or not name.strip():
                errors.append("dtype names must be non-empty strings")
                continue
            if not isinstance(config, dict):
                errors.append(f"dtype.{name} must be an object")
                continue
            for field in DTYPE_FIELDS:
                if field not in config:
                    errors.append(f"dtype.{name} missing required field: {field}")
            for field in ("atol", "rtol", "max_abs_limit"):
                value = config.get(field)
                if not _finite_number(value) or value < 0:
                    errors.append(
                        f"dtype.{name}.{field} must be a finite non-negative number"
                    )
            max_ulp = config.get("max_ulp_limit")
            if max_ulp is not None and (not _finite_number(max_ulp) or max_ulp < 0):
                errors.append(
                    f"dtype.{name}.max_ulp_limit must be null or a finite non-negative number"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.policy)
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: precision policy schema is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
