#!/usr/bin/env bash
# =============================================================================
# Test: Agent Content
# =============================================================================
# Validates content quality for all agents.
# Rules tested (all via skill_validator.py validate-agent --subset=content):
#   error level (blocking):
#     A-CON-01: name matches directory/file name
#     A-CON-02: description contains trigger keywords (skipped if disable-model-invocation)
#   warn level (advisory):
#     A-CON-03: description contains trigger conditions (skipped for mode: subagent)
#     A-CON-04: long files link to supporting files (progressive disclosure)
#     A-CON-05: no anti-pattern phrases in description
#
# Supports incremental testing via INCREMENTAL_AGENTS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Agent Content ==="
echo ""
echo "This test validates content quality for all agents."
echo "Run time: ~10 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed agents"
    echo ""
fi

# Counters
total_agents=0
pass_count=0
fail_count=0
skip_count=0

# Get agents to test (filtered if in incremental mode)
AGENTS_TO_TEST=$(get_agents_to_test)
total_agents=$(echo "$AGENTS_TO_TEST" | grep -c . || echo "0")

# Pre-compute skill paths once to avoid repeated full-repo scans
CACHED_SKILL_PATHS=$(get_all_skills_with_paths | cut -d: -f2-)

echo "Agents to test: $total_agents"
echo ""

# ============================================
# Validate agents content
# ============================================
print_section_header "Test: Agent Content (A-CON-01 to A-CON-09)"

# Collect files for batch validation
batch_files=()
for agent in $AGENTS_TO_TEST; do
    [ -z "$agent" ] && continue

    if is_incremental_mode && ! should_test_agent "$agent"; then
        print_skip "$agent: Not in changed list"
        ((skip_count++)) || true
        continue
    fi

    agent_file=$(find_agent_file "$agent")

    if [ ! -f "$agent_file" ]; then
        print_skip "$agent: AGENT.md not found"
        ((skip_count++)) || true
        continue
    fi

    batch_files+=("$agent_file")
done

if [ ${#batch_files[@]} -gt 0 ]; then
    skill_paths_csv=$(echo "$CACHED_SKILL_PATHS" | tr '\n' ',')
    batch_output=$(validate_agents_content_batch "$skill_paths_csv" "${batch_files[@]}" 2>&1) || true
    echo "$batch_output"
    pass_count=$(echo "$batch_output" | grep -c '\[PASS\]' || echo 0)
    fail_count=$(echo "$batch_output" | grep -c '\[FAIL\]' || echo 0)
    [[ "$pass_count" =~ ^[0-9]+$ ]] || pass_count=0
    [[ "$fail_count" =~ ^[0-9]+$ ]] || fail_count=0
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Agent Content Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total agents: $total_agents"
echo -e "  ${GREEN}Passed:${NC}       $pass_count"
echo -e "  ${RED}Failed:${NC}       $fail_count"
[ $skip_count -gt 0 ] && echo -e "  ${YELLOW}Skipped:${NC}      $skip_count"
echo ""

if [ $fail_count -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the failed content checks."
    exit 1
else
    print_status_passed
    exit 0
fi