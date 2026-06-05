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
#     A-CON-03: description contains trigger conditions
#     A-CON-04: contains actionable instructions (code blocks, numbered steps)
#     A-CON-05: contains error handling / troubleshooting section
#     A-CON-06: contains examples / scenario section
#     A-CON-07: long files link to references/ (progressive disclosure)
#     A-CON-08: description follows three-segment structure
#     A-CON-09: no anti-pattern phrases in description
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

for agent in $AGENTS_TO_TEST; do
    [ -z "$agent" ] && continue

    # In incremental mode, check if this agent should be tested
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

    if validate_agent_content "$agent_file" "$CACHED_SKILL_PATHS"; then
        ((pass_count++)) || true
    else
        ((fail_count++)) || true
    fi
done

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