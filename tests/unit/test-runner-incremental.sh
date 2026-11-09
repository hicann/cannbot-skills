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
# Test: run-tests.sh 增量模式技能检测看护
# =============================================================================
# 插件迁移后技能的现行布局为域目录（ops/<skill>/、community/ops/<skill>/、
# communication/<skill>/ 等），run-tests.sh 增量模式的旧检测正则期望
# skills/<name>/，对域技能全部漏检——混合 PR（技能 + 插件改动）在增量
# 模式下会静默跳过技能测试（P0 基线已用探针复现）。
#
# 本看护从 run-tests.sh 提取 detect_skill_name_from_path()（awk 括号
# 深度跟踪，模式同 test-safe-install.sh 提取 safe_install_file）做纯函数
# 级验证：
#   RI-01: 提取物完整（SKILL_ROOTS 数组 + 函数存在）
#   RI-02: ops/<skill>/… 被识别
#   RI-03: community/ops/<skill>/… 被识别（迁移后布局）
#   RI-04: communication/<skill>/… 被识别（st-test.config 漏配但确有技能）
#   RI-05: 插件内嵌 skills/<name>/… 布局保持识别（旧行为保留，如 triton）
#   RI-06: 域根直属文件（ops/README.md）不误判
#   RI-07: agents/ 路径不误判（由 agent 模式处理）
#   RI-08: tests/ 路径不误判（由 framework 模式处理）
#   RI-09: analyze_changes 实际调用该函数（接线静态断言）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

RUNNER="$SCRIPT_DIR/../run-tests.sh"

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
# 从 run-tests.sh 提取 SKILL_ROOTS + detect_skill_name_from_path
# =============================================================================
EXTRACT="$(mktemp)"
trap 'rm -f "$EXTRACT"' EXIT

print_section "Extract: SKILL_ROOTS + detect_skill_name_from_path"

ri01_extract() {
    grep -E '^SKILL_ROOTS=' "$RUNNER" > "$EXTRACT" || return 1
    awk '
        /^detect_skill_name_from_path\(\)/ { found=1; depth=0 }
        found {
            print
            for (i = 1; i <= length($0); i++) {
                c = substr($0, i, 1)
                if (c == "{") depth++
                if (c == "}") depth--
            }
            if (depth <= 0 && $0 ~ /^}/) exit
        }
    ' "$RUNNER" >> "$EXTRACT"
    # shellcheck disable=SC1090
    source "$EXTRACT"
    # 基本完整性：根数组非空 + 函数可调用
    [ "${#SKILL_ROOTS[@]}" -ge 7 ] && detect_skill_name_from_path "ops/x/SKILL.md" >/dev/null
}
run_check "RI-01: SKILL_ROOTS 与函数可提取且完整" ri01_extract

# =============================================================================
# 纯函数检测用例
# =============================================================================
print_section "Detect: 域技能 / 插件内嵌技能 / 非技能路径"

expect_skill() {
    # expect_skill <期望技能名> <路径>
    local expected="$1" path="$2"
    local got
    got="$(detect_skill_name_from_path "$path")" || return 1
    [ "$got" = "$expected" ]
}

expect_not_skill() {
    local path="$1"
    ! detect_skill_name_from_path "$path" >/dev/null 2>&1
}

run_check "RI-02: ops/<skill>/ 被识别" \
    expect_skill "ascendc-api-best-practices" "ops/ascendc-api-best-practices/SKILL.md"
run_check "RI-03: community/ops/<skill>/ 被识别（迁移后布局）" \
    expect_skill "ascendc-evolution-strategies" "community/ops/ascendc-evolution-strategies/references/guide.md"
run_check "RI-04: communication/<skill>/ 被识别" \
    expect_skill "hccl-env-check" "communication/hccl-env-check/SKILL.md"
run_check "RI-10: community/model/<skill>/ 被识别" \
    expect_skill "model-recommend-structure-analysis" "community/model/model-recommend-structure-analysis/SKILL.md"
run_check "RI-05: 插件内嵌 skills/<name>/ 保持识别（旧行为）" \
    expect_skill "triton-npu-optimize" "plugins-community/triton-optimizer/skills/triton-npu-optimize/scripts/x.py"
run_check "RI-06: 域根直属文件不误判" \
    expect_not_skill "ops/README.md"
run_check "RI-07: agents/ 路径不误判" \
    expect_not_skill "plugins-official/ops-registry-invoke/agents/architect.md"
run_check "RI-08: tests/ 路径不误判" \
    expect_not_skill "tests/unit/install/test-init-install.sh"

# =============================================================================
# 接线静态断言：analyze_changes 必须调用该函数（防回归为内联正则）
# =============================================================================
run_check "RI-09: analyze_changes 实际调用 detect_skill_name_from_path" \
    grep -qF 'detect_skill_name_from_path "$file"' "$RUNNER"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Runner Incremental Detection Summary"
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
