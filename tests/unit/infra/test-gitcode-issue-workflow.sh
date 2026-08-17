#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Test: GitCode Issue Workflow ==="
echo ""
echo "Runs the local unit and contract tests for gitcode-issue-handler"
echo "and gitcode-toolkit. No CLI or network access is required."
echo ""

if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "  [FAIL] pytest is required; install tests/system/scripts/requirements.txt"
    exit 1
fi

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
    -p no:cacheprovider \
    -q \
    "$REPO_ROOT/infra/gitcode-issue-handler/tests" \
    "$REPO_ROOT/infra/gitcode-toolkit/tests"
