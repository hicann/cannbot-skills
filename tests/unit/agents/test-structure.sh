#!/usr/bin/env bash
# =============================================================================
# Test: Agent Structure
# =============================================================================
# Validates structure correctness for all agents.
# Rules tested:
# - A-STR-01: YAML Front Matter format
# - A-STR-02: name/description/mode fields exist
# - A-STR-03: mode is primary or subagent
# - A-STR-04: All skill dependencies exist
# - A-STR-05: name length 1-64 characters
# - A-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
# - A-STR-07: description length 1-1024 characters
# - A-STR-08: All links point to existing files
#
# Supports incremental testing via INCREMENTAL_AGENTS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Agent Structure ==="
echo ""
echo "This test validates structure for all agents."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed agents"
    echo ""
fi

# Counters
total_agents=0
structure_pass=0
structure_fail=0
link_pass=0
link_fail=0
skip_count=0

# Get agents to test (filtered if in incremental mode)
AGENTS_TO_TEST=$(get_agents_to_test)
total_agents=$(echo "$AGENTS_TO_TEST" | grep -c . || echo "0")

echo "Agents to test: $total_agents"
echo ""

# ============================================
# Test 1: Agent Structure Validation
# ============================================
print_section_header "Test: Agent Structure (A-STR-01 to A-STR-07)"

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

    if validate_agent_structure "$agent_file"; then
        ((structure_pass++)) || true
    else
        ((structure_fail++)) || true
    fi
done

echo ""

# ============================================
# Test 2: Link Validity
# ============================================
print_section_header "Test: Link Validity (A-STR-08)"

while IFS=: read -r aname apath; do
    [ -z "$aname" ] && continue

    # In incremental mode, only check links for changed agents
    if is_incremental_mode && ! should_test_agent "$aname"; then
        continue
    fi

    if [ -f "$apath" ]; then
        if check_file_links "$apath" "agent"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done <<< "$(get_all_agents_with_paths)"

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Agent Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total agents: $total_agents"
echo -e "  Structure tests: ${GREEN}$structure_pass passed${NC}, ${RED}$structure_fail failed${NC}"
echo -e "  Link tests:      ${GREEN}$link_pass passed${NC}, ${RED}$link_fail failed${NC}"
[ $skip_count -gt 0 ] && echo -e "  ${YELLOW}Skipped:${NC}        $skip_count"
echo ""

if [ $((structure_fail + link_fail)) -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the failed structure checks."
    exit 1
else
    print_status_passed
    exit 0
fi