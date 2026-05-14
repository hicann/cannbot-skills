#!/usr/bin/env bash
# =============================================================================
# Test: Skill Trigger Correctness
# =============================================================================
# What this tests:
# 1. Knowledge Skills response accuracy
# 2. Debug Skills response accuracy
# 3. Tool Skills response accuracy
# 4. Negative tests (irrelevant prompts should not trigger skill content)
# 5. Trigger mechanism tests (explicit/invalid skill requests)
# 6. B-TRIG-01: Precise trigger - core keywords should trigger correct skill
# 7. B-TRIG-02: Fuzzy trigger - colloquial terms should still map to skill
# 8. B-BND-01: Negative rejection (irrelevant prompts should be politely rejected)
# 9. B-BND-02: Hallucination defense (fake API/wrong hardware should be caught)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Behavior Test ==="
echo ""
echo "This test verifies skills behavior with Claude CLI."
echo "Estimated time: 2-3 minutes"
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
# Part 1: Knowledge Skills
# =============================================================================

echo -e "${BOLD}=== Part 1: Knowledge Skills ===${NC}"
echo ""

run_behavior_test "npu-arch" \
    "Ascend910B 和 Ascend950 分别对应什么架构？" \
    "arch22|arch35|达芬奇|DaVinci|架构" \
    "$TIMEOUT"

run_behavior_test "ascendc-api-best-practices" \
    "Ascend C 中 DataCopy 的主要作用是什么？" \
    "搬运|copy|数据|传输" \
    "$TIMEOUT"

run_behavior_test "ascendc-operator-kernel-design" \
    "Ascend C Kernel 的核函数签名需要哪些关键修饰符？" \
    "__global__|__aicore__" \
    "$TIMEOUT"

run_behavior_test "ascendc-code-review" \
    "为什么要避免使用 GM_ADDR 直接计算地址？" \
    "对齐|align|越界|boundary|安全|风险|边界|类型检查|缓冲区" \
    "$TIMEOUT"

run_behavior_test "ascendc-custom-op-enhance" \
    "开发 Ascend C 自定义算子的基本步骤是什么？" \
    "设计|实现|测试|开发|编译|构建" \
    "$TIMEOUT"

# =============================================================================
# Part 2: Debug Skills
# =============================================================================

echo -e "${BOLD}=== Part 2: Debug Skills ===${NC}"
echo ""

run_behavior_test "ascendc-runtime-debug" \
    "算子运行时报错，我该怎么调试？" \
    "运行时|错误|错误码|aclnn|error|调试|排查" \
    "$TIMEOUT"

run_behavior_test "ascendc-precision-debug" \
    "算子精度不达标，应该怎么排查？" \
    "精度|precision|误差|对比|检查|数据" \
    "$TIMEOUT"

run_behavior_test "ascendc-crash-debug" \
    "算子执行时卡死崩溃了怎么办？" \
    "卡死|崩溃|crash|hang|死锁|挂起|调试|定位" \
    "$TIMEOUT"

run_behavior_test "ascendc-perf-analysis" \
    "算子性能太慢，如何优化？" \
    "性能|performance|优化|optimize|瓶颈|并行" \
    "$TIMEOUT"

run_behavior_test "ascendc-env-check" \
    "如何检查 Ascend NPU 环境是否正常？" \
    "npu-smi|环境|检查|NPU|设备" \
    "$TIMEOUT"

# =============================================================================
# Part 3: Tool Skills
# =============================================================================

echo -e "${BOLD}=== Part 3: Tool Skills ===${NC}"
echo ""

run_behavior_test "ascendc-kernel-develop-workflow" \
    "Ascend C Kernel 开发的第一阶段是什么？" \
    "环境|准备|phase|分析|设计|需求|Kernel" \
    "$TIMEOUT"

run_behavior_test "ascendc-ut-develop" \
    "Ascend C UT 测试应该放在哪个目录？" \
    "tests/ut|ut/|ut 目录|unit/|单元测试|tests/|ut" \
    "$TIMEOUT"

# =============================================================================
# Part 3.5: Trigger Accuracy Tests (B-TRIG-01, B-TRIG-02)
# =============================================================================

echo -e "${BOLD}=== Part 3.5: Trigger Accuracy Tests ===${NC}"
echo ""

# B-TRIG-01: Precise trigger - core keywords should trigger correct skill
test_precise_trigger() {
    local name="$1"
    local prompt="$2"
    local expected_skill="$3"
    local expected_content="$4"
    local timeout="${5:-25}"

    echo "Testing: $name (B-TRIG-01)"
    echo "Prompt: $prompt"
    echo "Expected skill: $expected_skill"

    if output=$(timeout "$timeout" claude -p "$prompt 用简短回答。" 2>&1); then
        # Check if expected content is present
        if echo "$output" | grep -qiE "$expected_content"; then
            print_pass "B-TRIG-01: Skill triggered correctly with core keyword"
            pass_count=$((pass_count + 1))
        else
            print_fail "B-TRIG-01: Expected content not found"
            echo -e "  ${YELLOW}Expected pattern:${NC} $expected_content"
            echo "  Output: ${output:0:100}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

# B-TRIG-02: Fuzzy trigger - colloquial terms should still map to skill
test_fuzzy_trigger() {
    local name="$1"
    local prompt="$2"
    local expected_content="$3"
    local timeout="${4:-25}"

    echo "Testing: $name (B-TRIG-02)"
    echo "Prompt: $prompt"

    if output=$(timeout "$timeout" claude -p "$prompt 用简短回答。" 2>&1); then
        # Check if the response shows understanding despite colloquial terms
        if echo "$output" | grep -qiE "$expected_content"; then
            print_pass "B-TRIG-02: Correctly interpreted colloquial/unclear query"
            pass_count=$((pass_count + 1))
        else
            print_fail "B-TRIG-02: Failed to map colloquial terms to correct skill"
            echo -e "  ${YELLOW}Expected pattern:${NC} $expected_content"
            echo "  Output: ${output:0:100}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

echo -e "${BOLD}B-TRIG-01: Precise Trigger Tests${NC}"
echo ""

test_precise_trigger "npu-arch-precise" \
    "Ascend910B芯片架构信息" \
    "npu-arch" \
    "arch22|达芬奇|DaVinci|架构" \
    "$TIMEOUT"

test_precise_trigger "runtime-debug-precise" \
    "Ascend C算子运行时错误码排查" \
    "ascendc-runtime-debug" \
    "错误码|aclnn|调试|debug|错误|排查" \
    "$TIMEOUT"

test_precise_trigger "crash-debug-precise" \
    "Ascend C算子卡死崩溃排查" \
    "ascendc-crash-debug" \
    "卡死|崩溃|crash|hang|死锁|挂起|调试|定位" \
    "$TIMEOUT"

test_precise_trigger "perf-analysis-precise" \
    "NPU算子性能优化分析" \
    "ascendc-perf-analysis" \
    "性能|performance|优化|optimize|瓶颈" \
    "$TIMEOUT"

echo -e "${BOLD}B-TRIG-02: Fuzzy Trigger Tests${NC}"
echo ""

test_fuzzy_trigger "colloquial-debug" \
    "我的代码跑不动了，帮忙看看" \
    "调试|debug|错误|排查|问题|运行" \
    "$TIMEOUT"

test_fuzzy_trigger "colloquial-perf" \
    "算子太慢了，怎么提速" \
    "性能|优化|加速|并行|性能" \
    "$TIMEOUT"

test_fuzzy_trigger "colloquial-precision" \
    "结果不对，数值偏差很大" \
    "精度|precision|误差|对比|正确|数值" \
    "$TIMEOUT"

# =============================================================================
# Part 4: Negative Tests (B-BND-01)
# =============================================================================

echo -e "${BOLD}=== Part 4: Negative Tests (B-BND-01) ===${NC}"
echo ""
echo "Testing that irrelevant prompts are politely rejected..."
echo ""

# B-BND-01: Test negative rejection - irrelevant prompts should not trigger professional skills
# The response should either politely decline or not contain CANN-specific technical content
test_negative_rejection() {
    local name="$1"
    local prompt="$2"
    local timeout="${3:-25}"

    echo "Testing: $name"
    echo "Prompt: $prompt"

    if output=$(timeout "$timeout" claude -p "$prompt" 2>&1); then
        # Check that no CANN-specific technical skills were triggered
        # The response should NOT contain detailed Ascend C code or deep technical content
        local has_cann_content=false
        if echo "$output" | grep -qiE "Ascend C|Kernel.*开发|Tiling.*实现|GM_ADDR|__aicore__|Ascend910B|npu-smi"; then
            has_cann_content=true
        fi

        # Check for polite rejection indicators
        local has_rejection=false
        if echo "$output" | grep -qiE "抱歉|无法|不在.*范围|不能.*帮助|不.*相关|无法.*回答|超出.*领域|专业领域|CANN.*生态"; then
            has_rejection=true
        fi

        # For creative tasks, check if response is appropriate (not technical)
        local is_creative_response=false
        if echo "$output" | grep -qiE "诗|诗歌|菜谱|红烧|天气|晴|雨"; then
            is_creative_response=true
        fi

        # Pass if: polite rejection OR creative response (not CANN technical content)
        if $has_rejection || ($is_creative_response && ! $has_cann_content); then
            print_pass "B-BND-01: Correctly handled irrelevant prompt"
            pass_count=$((pass_count + 1))
        elif ! $has_cann_content; then
            print_pass "B-BND-01: No CANN technical content triggered"
            pass_count=$((pass_count + 1))
        else
            print_fail "B-BND-01: Incorrectly triggered CANN skills for irrelevant prompt"
            echo -e "  ${YELLOW}Response should politely decline or not trigger CANN skills${NC}"
            echo "  Output: ${output:0:100}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

test_negative_rejection "weather (B-BND-01)" "今天天气怎么样？"
test_negative_rejection "poetry (B-BND-01)" "帮我写一首关于春天的诗"
test_negative_rejection "cooking (B-BND-01)" "如何做一道红烧肉？"
test_negative_rejection "unrelated-tech (B-BND-01)" "如何学习 Python 编程？"

# =============================================================================
# Part 5: Hallucination Defense Tests (B-BND-02)
# =============================================================================

echo -e "${BOLD}=== Part 5: Hallucination Defense Tests (B-BND-02) ===${NC}"
echo ""
echo "Testing that fake APIs and wrong hardware are caught..."
echo ""

# B-BND-02: Test hallucination defense - fake API/wrong hardware should be caught
test_hallucination_defense() {
    local name="$1"
    local prompt="$2"
    local expected_error="$3"
    local timeout="${4:-25}"

    echo "Testing: $name"
    echo "Prompt: $prompt"

    if output=$(timeout "$timeout" claude -p "$prompt" 2>&1); then
        # Check if the response catches the error
        local caught_error=false
        if echo "$output" | grep -qiE "$expected_error"; then
            caught_error=true
        fi

        # Check for various error indication patterns
        local has_error_indication=false
        if echo "$output" | grep -qiE "不存在|没有.*API|错误|无效|不正确|不存在该|未知|无法识别|请.*核实|检查.*是否正确|可能.*错误|not.*exist|invalid|unknown|错误.*型号"; then
            has_error_indication=true
        fi

        if $caught_error || $has_error_indication; then
            print_pass "B-BND-02: Correctly caught hallucination/error"
            if $caught_error; then
                print_info "  - Detected expected error pattern: $expected_error"
            fi
            pass_count=$((pass_count + 1))
        else
            print_fail "B-BND-02: Failed to catch hallucination/error"
            echo -e "  ${YELLOW}Expected to detect error in: $expected_error${NC}"
            echo "  Output: ${output:0:150}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

test_hallucination_defense "fake-api (B-BND-02)" \
    "请用 Ascend C 的 DataCopyUltraFast 接口实现数据搬运" \
    "不存在|没有.*API|无效|无法识别|不正确|请.*核实|可能.*错误"

test_hallucination_defense "wrong-chip (B-BND-02)" \
    "Ascend910Z 芯片支持什么架构？" \
    "不存在|错误.*型号|无效|未知|不正确|没有.*型号|请.*核实"

test_hallucination_defense "fake-function (B-BND-02)" \
    "请使用 Ascend C 的 SetTensorMagic 函数初始化张量" \
    "不存在|没有.*函数|无效|无法识别|不正确|请.*核实|可能.*错误"

test_hallucination_defense "wrong-api-params (B-BND-02)" \
    "Ascend C 的 DataCopy 接口支持 5 维张量吗？" \
    "不支持|维度|正确|错误|请.*检查|参数"

# =============================================================================
# Part 6: Trigger Mechanism Tests
# =============================================================================

echo -e "${BOLD}=== Part 6: Trigger Mechanism Tests ===${NC}"
echo ""

run_behavior_test "explicit skill request" \
    "请使用 ascendc-runtime-debug 技能分析运行时错误。" \
    "运行时|runtime|错误|error|调试" \
    "$TIMEOUT"

run_behavior_test "invalid skill handling" \
    "请使用 nonexistent-skill-xyz 技能。" \
    "不存在|not.*found|无法|unknown|没有|抱歉|找不到" \
    "$TIMEOUT"

run_behavior_test "skill name typo handling" \
    "请使用 ascendc-runtime-debub 技能。" \
    "不存在|not.*found|无法|unknown|没有|抱歉|找不到|运行时|runtime" \
    "$TIMEOUT"

# =============================================================================
# Summary
# =============================================================================

echo "========================================"
echo -e " ${BOLD}Skill Behavior Test Summary${NC}"
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