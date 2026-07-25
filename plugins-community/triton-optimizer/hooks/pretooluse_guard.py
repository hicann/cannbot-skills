#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Claude Code PreToolUse hook wrapper for the standalone optimize plugin.

This variant computes its workspace policy dynamically from the active cwd
instead of relying on a runner-generated policy.json file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from hook_runtime.pretooluse_adapter import run_with_policy
from state_bootstrap import (
    resolve_workspace,
)


def _read_hook_payload() -> dict[str, object] | None:
    try:
        payload = json.load(sys.stdin)
    except (OSError, TypeError, ValueError) as exc:
        print(f"triton-agent claude plugin PreToolUse failed open: {exc}", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


def main() -> int:
    payload = _read_hook_payload()
    if payload is None:
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0

    return run_with_policy(
        policy=_policy(workspace),
        payload=payload,
        failure_prefix="triton-agent claude plugin PreToolUse",
    )


def _policy(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    return {
        "workspace_root": str(root),
        "allow_read_roots": ["/"],
        "deny_read_globs": [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
