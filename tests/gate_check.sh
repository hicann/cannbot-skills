# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FRAMEWORK_DIR="$REPO_ROOT/tests/system"
TARGET_BRANCH="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-${BASE_BRANCH:-master}}"

# =========================================================================
# Phase 1: 环境初始化
# =========================================================================
echo "=== Phase 1: Environment Setup ==="
echo "Repository root: $REPO_ROOT"

# =========================================================================
# Phase 2: 检测变更文件
# =========================================================================
echo "=== Phase 2: Detect Changed Files ==="
# 优先从 pr_filelist.txt 读取（CI 流水线在项目根目录生成）
PR_FILELIST="$REPO_ROOT/pr_filelist.txt"
if [ -f "$PR_FILELIST" ]; then
    CHANGED_FILES=$(grep -v '^\s*$' "$PR_FILELIST" || true)
    echo "[from pr_filelist.txt]"
fi

if [ -z "${CHANGED_FILES:-}" ]; then
    git fetch origin "$TARGET_BRANCH" --depth=1 2>/dev/null || true
    CHANGED_FILES=$(git diff --name-only "origin/$TARGET_BRANCH"...HEAD 2>/dev/null || true)
fi

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed files detected. Exiting."
    exit 0
fi
echo "Changed files:"
echo "$CHANGED_FILES"

readarray -t changed_files_array <<< "$CHANGED_FILES"

# =========================================================================
# Phase 3: 安装依赖
# =========================================================================
echo "=== Phase 3: Install Dependencies ==="
pip install -r "$FRAMEWORK_DIR/scripts/requirements.txt" --quiet

# =========================================================================
# Phase 4: 执行门禁检查
# =========================================================================
echo "=== Phase 4: Run Gate Check ==="

python3 "$FRAMEWORK_DIR/scripts/main.py" \
    --repo-root "$REPO_ROOT" \
    --changed-files "${changed_files_array[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Gate check PASSED"
else
    echo "Gate check FAILED with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE