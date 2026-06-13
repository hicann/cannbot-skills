#!/usr/bin/env bash
# =============================================================================
# Test: Agent Trigger Correctness
# =============================================================================
# What this tests:
# 1. Agent response accuracy (architect, developer, tester, reviewer)
# 2. Negative tests (irrelevant prompts should not trigger agent behavior)
# 3. Trigger mechanism tests (explicit/invalid agent requests)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Agent Behavior Test ==="
echo ""
echo "This test verifies agents behavior with Claude CLI."
echo "Estimated time: 1-2 minutes"
echo "Requires: Claude Code CLI"
echo ""

if ! command -v claude &> /dev/null; then
    echo -e "${YELLOW}[SKIP]${NC} Claude Code CLI not found — skipping behavior test"
    exit 0
fi

TIMEOUT=25
pass_count=0
fail_count=0
skip_count=0

# =============================================================================
# Part 1: Agent Behavior Tests
# =============================================================================

echo -e "${BOLD}=== Part 1: Agent Behavior Tests ===${NC}"
echo ""

run_behavior_test "ascendc-ops-architect" \
    "你是一个算子架构师。用户需要开发一个 Add 算子，你应该首先做什么？" \
    "需求|分析|设计|requirement|架构|规格|明确|定义" \
    "$TIMEOUT"

run_behavior_test "ascendc-ops-developer" \
    "你是一个算子开发工程师。设计方案已确定，你应该按什么顺序开发？" \
    "迭代|开发|实现|Kernel|Tiling|编码" \
    "$TIMEOUT"

run_behavior_test "ascendc-ops-tester" \
    "你是一个算子测试工程师。算子代码已完成，你应该如何设计测试？" \
    "测试|test|用例|ST|UT|覆盖|验证" \
    "$TIMEOUT"

run_behavior_test "ascendc-ops-reviewer" \
    "你是一个代码检视专家。检视 Ascend C 代码时主要关注哪些方面？" \
    "检视|review|规范|安全|质量|代码|内存|性能|并行|优化" \
    "$TIMEOUT"

run_behavior_test "model-infer-analyzer" \
    "你是 NPU 推理优化分析专家。给定模型代码和 profiling 数据，你怎么分析架构和并行策略，是否会改模型代码？" \
    "架构|并行|策略|profiling|只读|不改|不修改" \
    "$TIMEOUT"

run_behavior_test "model-infer-implementer" \
    "你是 NPU 推理优化实施专家。方案已确认，你应该怎么推进？" \
    "实施|改造|实现|调试|修复|验证|代码" \
    "$TIMEOUT"

run_behavior_test "model-infer-reviewer" \
    "你是 NPU 推理优化验证专家。改造完成后你怎么验证是否达标，输出什么交给实施工程师？" \
    "验证|精度|性能|诊断|报告|基线" \
    "$TIMEOUT"

# =============================================================================
# Part 2: Negative Tests (Non-Agent Prompts)
# =============================================================================

echo -e "${BOLD}=== Part 2: Negative Tests ===${NC}"
echo ""

run_behavior_test "travel advice (should not trigger agent)" \
    "推荐一个适合度假的旅游目的地。" \
    "旅游|目的地|度假|景点|推荐|风景|海滩|山" \
    "$TIMEOUT"

run_behavior_test "fitness advice (should not trigger agent)" \
    "如何制定一个健身计划？" \
    "健身|运动|锻炼|计划|肌肉|训练|有氧|力量" \
    "$TIMEOUT"

run_behavior_test "recipe request (should not trigger agent)" \
    "推荐一道家常菜的做法。" \
    "菜|做法|食材|烹饪|家常|炒|煮|调味" \
    "$TIMEOUT"

# =============================================================================
# Part 3: Trigger Mechanism Tests
# =============================================================================

echo -e "${BOLD}=== Part 3: Trigger Mechanism Tests ===${NC}"
echo ""

run_behavior_test "explicit agent request" \
    "请让 ascendc-ops-architect agent 来分析这个算子需求。" \
    "需求|分析|设计|架构|规格" \
    "$TIMEOUT"

run_behavior_test "invalid agent handling" \
    "请让 nonexistent-agent-xyz agent 来处理。" \
    "不存在|not.*found|无法|unknown|没有|抱歉|找不到" \
    "$TIMEOUT"

run_behavior_test "agent name typo handling" \
    "请让 ascendc-ops-architekt agent 来分析需求。" \
    "不存在|not.*found|无法|unknown|没有|抱歉|找不到|不在|需求|分析" \
    "$TIMEOUT"

# =============================================================================
# Summary
# =============================================================================

echo "========================================"
echo -e " ${BOLD}Agent Behavior Test Summary${NC}"
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