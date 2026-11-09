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

# =============================================================================
# Test: test-safe-install.sh 迁移韧性看护
# =============================================================================
# 插件内容持续向 cannbot 仓迁移（ops-registry-invoke 等已迁移/冻结待删），
# test-safe-install.sh 若硬编码单一插件的 init.sh 路径，将在该插件删除时
# 突然变红。本看护保证其"动态发现"形态：
#
#   SI-01: 不再硬编码 ops-registry-invoke（或任何单一插件）路径
#   SI-02: 按插件目录遍历发现 safe_install_file() 实现的检索逻辑存在
#   SI-03: 沙箱模拟"插件已全部迁走"时，必须 SKIP（exit 0）而非 FAIL
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET_TEST="$SCRIPT_DIR/test-safe-install.sh"

PASS_COUNT=0
FAIL_COUNT=0

run_check() {
    local name="$1"
    shift
    if "$@"; then
        print_pass "$name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "$name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# =============================================================================
# Static rules
# =============================================================================
print_section "Static: test-safe-install.sh 不依赖单一插件"

si01_no_hardcode() {
    ! grep -q "ops-registry-invoke" "$TARGET_TEST"
}
run_check "SI-01: 无 ops-registry-invoke 硬编码路径" si01_no_hardcode

run_check "SI-02: 按插件目录遍历发现 safe_install_file 实现" \
    grep -qF 'grep -q "^safe_install_file"' "$TARGET_TEST"

# =============================================================================
# Behavioral rule：沙箱模拟插件全部迁走
# =============================================================================
print_section "Behavior: 无插件环境必须 SKIP"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

mkdir -p "$SANDBOX/tests/unit/install" "$SANDBOX/tests/lib"
cp "$TARGET_TEST" "$SANDBOX/tests/unit/install/test-safe-install.sh"
cp "$SCRIPT_DIR/../../lib/"*.sh "$SANDBOX/tests/lib/"
# 不创建任何 plugins-official / plugins-community 目录 —— 模拟插件全部迁走

SB_OUT="$(cd "$SANDBOX" && bash "$SANDBOX/tests/unit/install/test-safe-install.sh" </dev/null 2>&1)" \
    && SB_RC=0 || SB_RC=$?

run_check "SI-03a: 无插件环境 → exit 0（SKIP 而非 FAIL）" test "$SB_RC" -eq 0
si03_skip_msg() { printf '%s' "$SB_OUT" | grep -qE "SKIP|跳过"; }
run_check "SI-03b: 无插件环境 → 输出含 SKIP/跳过说明" si03_skip_msg

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Safe Install Resilience Test Summary"
echo "========================================"
echo "  Passed:  $PASS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
