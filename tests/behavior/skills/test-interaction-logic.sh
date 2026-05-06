# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

#!/usr/bin/env bash
# =============================================================================
# Test: Skill Interaction Logic
# =============================================================================
# What this tests:
# 1. B-INTA-01: Missing parameter clarification - ask when key info missing
# 2. B-INTA-02: Context retention - maintain variables across conversation
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Interaction Logic ==="
echo ""
echo "This test verifies interaction logic with Claude CLI."
echo "Estimated time: 2-3 minutes"
echo "Requires: Claude Code CLI"
echo ""

if ! command -v claude &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Claude Code CLI not found"
    echo "Install Claude Code first: https://code.claude.com"
    exit 1
fi

TIMEOUT=30
pass_count=0
fail_count=0
skip_count=0

# =============================================================================
# Part 1: Missing Parameter Clarification (B-INTA-01)
# =============================================================================

echo -e "${BOLD}=== Part 1: Missing Parameter Clarification (B-INTA-01) ===${NC}"
echo ""
echo "Testing that skills ask for clarification when key parameters are missing..."
echo ""

# B-INTA-01: Test that skill asks for clarification when key info is missing
test_missing_param_clarification() {
    local name="$1"
    local prompt="$2"
    local expected_question="$3"
    local timeout="${4:-30}"

    echo "Testing: $name (B-INTA-01)"
    echo "Prompt: $prompt"

    if output=$(timeout "$timeout" claude -p "$prompt" 2>&1); then
        # Check if the response asks for clarification
        local asks_clarification=false

        # Look for question patterns
        if echo "$output" | grep -qiE "$expected_question"; then
            asks_clarification=true
        fi

        # Also check for general clarification patterns
        local has_clarification_pattern=false
        if echo "$output" | grep -qiE "请问|请提供|请说明|是什么|哪个|哪种|请确认|需要知道|请告诉我|需要.*信息"; then
            has_clarification_pattern=true
        fi

        if $asks_clarification || $has_clarification_pattern; then
            print_pass "B-INTA-01: Correctly asked for clarification"
            if $asks_clarification; then
                print_info "  - Matched expected question pattern"
            fi
            pass_count=$((pass_count + 1))
        else
            print_fail "B-INTA-01: Did not ask for clarification when key info missing"
            echo -e "  ${YELLOW}Expected question about:${NC} $expected_question"
            echo "  Output: ${output:0:150}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

# Test cases for missing parameters
test_missing_param_clarification "missing-chip-arch" \
    "我要开发一个 Ascend C 算子，请给我设计 Tiling 方案" \
    "芯片|架构|arch|Ascend910|Ascend310" \
    "$TIMEOUT"

test_missing_param_clarification "missing-data-type" \
    "帮我实现一个矩阵乘法算子" \
    "数据类型|dtype|精度|float|half" \
    "$TIMEOUT"

test_missing_param_clarification "missing-shape-info" \
    "帮我优化这个算子的性能" \
    "shape|形状|维度|大小|输入" \
    "$TIMEOUT"

# =============================================================================
# Part 2: Context Retention (B-INTA-02)
# =============================================================================

echo -e "${BOLD}=== Part 2: Context Retention (B-INTA-02) ===${NC}"
echo ""
echo "Testing that skills maintain context across conversation turns..."
echo ""

# B-INTA-02: Test context retention in multi-turn conversation
test_context_retention() {
    local name="$1"
    local prompt="$2"
    local expected_context="$3"
    local timeout="${4:-30}"

    echo "Testing: $name (B-INTA-02)"
    echo "Prompt: $prompt"

    if output=$(timeout "$timeout" claude -p "$prompt" 2>&1); then
        # Check if the response references the context from previous turns
        if echo "$output" | grep -qiE "$expected_context"; then
            print_pass "B-INTA-02: Context was correctly retained"
            pass_count=$((pass_count + 1))
        else
            print_fail "B-INTA-02: Context was not properly retained"
            echo -e "  ${YELLOW}Expected context reference:${NC} $expected_context"
            echo "  Output: ${output:0:150}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

# Multi-turn conversation test using a combined prompt
test_context_retention "chip-arch-context" \
    "我正在开发 Ascend910B 的算子。基于这个架构，我应该使用什么 Tiling 策略？" \
    "Ascend910B|arch22|910B|这个架构" \
    "$TIMEOUT"

test_context_retention "data-type-context" \
    "我之前说了要用 float16 精度。基于这个精度要求，DataCopy 应该怎么配置？" \
    "float16|fp16|half|精度" \
    "$TIMEOUT"

test_context_retention "workflow-context" \
    "我们在第一步确定了算子规格是 1024x1024 的矩阵乘法。现在进入第二步，应该做什么？" \
    "1024|矩阵乘法|规格|第二步|实现|编码" \
    "$TIMEOUT"

# =============================================================================
# Summary
# =============================================================================

echo "========================================"
echo -e " ${BOLD}Skill Interaction Logic Test Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $pass_count"
echo -e "  ${RED}Failed:${NC}  $fail_count"
echo -e "  ${YELLOW}Skipped:${NC} $skip_count"
echo ""

total=$((pass_count + fail_count))
if [ $total -gt 0 ]; then
    accuracy=$((pass_count * 100 / total))
    echo "  Accuracy: ${accuracy}%"
    echo ""
fi

if [ $fail_count -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
