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

# 预检 opencode CLI（Phase 2 AI 语义评测的前置依赖）
if ! command -v opencode &> /dev/null; then
    echo "ERROR: opencode CLI not found in PATH."
    echo "Phase 2 (AI 语义评测) 需要 opencode，请安装后再运行 gate_check。"
    exit 1
fi
echo "  opencode: $(opencode --version 2>&1 | head -1)"

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
pip install -r "$FRAMEWORK_DIR/scripts/requirements.txt" --quiet --break-system-packages

# =========================================================================
# Phase 4: 执行门禁检查
# =========================================================================
echo "=== Phase 4: Run Gate Check ==="

# 后台心跳进程：每 60 秒输出一次时间戳，保持 SSH 连接（CI 执行机通过 SSH
# 连接测试执行机，长时间无 stdout 输出会导致会话中断）
heartbeat_pid=""
if [ -z "${GATE_CHECK_NO_HEARTBEAT:-}" ]; then
    (
        while true; do
            echo "[HEARTBEAT $(date '+%Y-%m-%d %H:%M:%S')] gate_check running..."
            sleep 60
        done
    ) &
    heartbeat_pid=$!
    trap 'kill "$heartbeat_pid" 2>/dev/null' EXIT
fi

python3 "$FRAMEWORK_DIR/scripts/main.py" \
    --repo-root "$REPO_ROOT" \
    --changed-files "${changed_files_array[@]}" \
    --parallel auto

EXIT_CODE=$?

# 停止心跳
if [ -n "$heartbeat_pid" ]; then
    kill "$heartbeat_pid" 2>/dev/null || true
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Gate check PASSED"
else
    echo "Gate check FAILED with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE