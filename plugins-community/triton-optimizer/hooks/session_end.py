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

from __future__ import annotations

import json
import sys

from state_bootstrap import cleanup_runtime_tree, resolve_workspace


_JSON_INPUT_ERRORS = (json.JSONDecodeError, OSError, UnicodeDecodeError)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except _JSON_INPUT_ERRORS as exc:
        print(f"triton-agent claude plugin SessionEnd failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    try:
        cleanup_runtime_tree(workspace / ".triton-agent")
    except OSError as exc:
        print(f"triton-agent claude plugin SessionEnd failed open: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
