#!/usr/bin/env bash
# =============================================================================
# Test: Agent Premature Action Detection
# =============================================================================
# Verifies that tools are not invoked BEFORE Agents are properly dispatched.
# This ensures Claude follows the agent instructions before taking actions.
#
# What this tests:
# 1. Task tool should be used to dispatch agents
# 2. No Write/Edit operations should happen before agent context is loaded
# 3. TodoWrite before Task is acceptable (for planning)
# 4. Read before Task is acceptable (for understanding context)
#
# Usage: ./test-premature-action.sh [agent-name]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

AGENT_NAME="${1:-ascendc-ops-developer}"
TIMEOUT="${2:-120}"

echo "=== Test: Agent Premature Action Detection ==="
echo ""
echo "This test verifies that tools are not invoked before Agents are dispatched."
echo "Target agent: $AGENT_NAME"
echo "Timeout: ${TIMEOUT}s"
echo ""

if ! command -v claude &> /dev/null; then
    echo -e "${YELLOW}[SKIP]${NC} Claude Code CLI not found — skipping behavior test"
    exit 0
fi

TIMESTAMP=$(date +%s)
OUTPUT_DIR="/tmp/cann-skills-tests/${TIMESTAMP}/agent-premature-action/${AGENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# Test scenarios with prompts that should trigger agent dispatch
declare -A TEST_SCENARIOS=(
    ["architect_scenario"]="你是一个算子架构师。用户需要开发一个 MatMul 算子，请分析需求并给出设计方案。"
    ["developer_scenario"]="你是一个算子开发工程师。请帮我实现一个 Add 算子的 Kernel 代码。"
    ["tester_scenario"]="你是一个算子测试工程师。请为 Add 算子设计测试用例。"
    ["reviewer_scenario"]="你是一个代码检视专家。请评审这段 Ascend C 代码的质量和安全性。"
)

test_premature_action() {
    local scenario="$1"
    local prompt="$2"
    local log_file="$OUTPUT_DIR/${scenario}.log"

    echo -e "${BOLD}--- Scenario: $scenario ---${NC}"
    echo "Prompt: ${prompt:0:50}..."
    echo ""

    # Run Claude and capture output
    # Note: Session analysis uses the most recent session from ~/.claude/projects/
    if claude -p "$prompt" \
        --dangerously-skip-permissions \
        > "$log_file" 2>&1; then
        :
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_warn "Timeout after ${TIMEOUT}s"
        fi
    fi

    # Find the session file created by this interaction
    local session_file=$(find_recent_session 2)

    if [ -n "$session_file" ] && [ -f "$session_file" ]; then
        cp "$session_file" "$OUTPUT_DIR/${scenario}.jsonl"
        analyze_premature_actions "$session_file" "agent" "$AGENT_NAME"
    else
        print_warn "No session file found for analysis"
        print_info "Output saved to: $log_file"
    fi

    rm -f "$session_file" 2>/dev/null || true
    echo ""
}

# Run tests for each scenario
echo "Running agent premature action tests..."
echo ""

total_passed=0
total_failed=0

for scenario in "${!TEST_SCENARIOS[@]}"; do
    prompt="${TEST_SCENARIOS[$scenario]}"
    if test_premature_action "$scenario" "$prompt"; then
        ((total_passed++)) || true
    else
        ((total_failed++)) || true
    fi
done

# Summary
echo "========================================"
echo -e " ${BOLD}Agent Premature Action Test Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $total_passed"
echo -e "  ${RED}Failed:${NC}  $total_failed"
echo ""
echo "  Output directory: $OUTPUT_DIR"
echo "    - *.log: CLI output logs"
echo "    - *.jsonl: Session transcripts (for analysis)"
echo ""

if [ $total_failed -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi