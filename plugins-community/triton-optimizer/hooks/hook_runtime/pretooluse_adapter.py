# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from hook_runtime.tool_use_decision import deny_reason_for_tool_use


_JSON_INPUT_ERRORS = (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError)
_POLICY_EVALUATION_ERRORS = (OSError, TypeError, ValueError)


def build_denial_output(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], data)


def run_policy_file_wrapper(
    *,
    argv: list[str] | None,
    failure_prefix: str,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args(argv)

    try:
        policy = load_json_object(Path(args.policy))
        payload = json.load(sys.stdin)
    except _JSON_INPUT_ERRORS as exc:
        _print_fail_open(failure_prefix, exc)
        return 0

    return run_with_policy(
        policy=policy,
        payload=payload,
        failure_prefix=failure_prefix,
    )


def run_with_policy(
    *,
    policy: dict[str, Any],
    payload: Any,
    failure_prefix: str,
) -> int:
    try:
        if not isinstance(payload, dict):
            raise ValueError("expected JSON object payload on stdin")
        reason = deny_reason_for_tool_use(policy, cast(dict[str, Any], payload))
    except _POLICY_EVALUATION_ERRORS as exc:
        _print_fail_open(failure_prefix, exc)
        return 0

    if reason is not None:
        json.dump(build_denial_output(reason), sys.stdout)
    return 0


def _print_fail_open(prefix: str, exc: Exception) -> None:
    print(f"{prefix} failed open: {exc}", file=sys.stderr)
