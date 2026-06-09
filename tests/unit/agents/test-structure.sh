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
# - A-STR-09: Agent name uniqueness across all agents
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

# Pre-compute skill paths once to avoid repeated full-repo scans
CACHED_SKILL_PATHS=$(get_all_skills_with_paths | cut -d: -f2-)

echo "Agents to test: $total_agents"
echo ""

# ============================================
# Test 1: Agent Structure Validation
# ============================================
print_section_header "Test: Agent Structure (A-STR-01 to A-STR-07)"

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
    batch_output=$(validate_agents_structure_batch "$skill_paths_csv" "${batch_files[@]}" 2>&1) || true
    echo "$batch_output"
    structure_pass=$(echo "$batch_output" | grep -c '\[PASS\]' || echo 0)
    structure_fail=$(echo "$batch_output" | grep -c '\[FAIL\]' || echo 0)
    [[ "$structure_pass" =~ ^[0-9]+$ ]] || structure_pass=0
    [[ "$structure_fail" =~ ^[0-9]+$ ]] || structure_fail=0
fi

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
# Test 3: Global Uniqueness
# ============================================
print_section_header "Test: Agent Name Uniqueness (A-STR-09)"

uniqueness_fail=0
if ! validate_global_uniqueness "agent"; then
    uniqueness_fail=1
fi

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
echo -e "  Uniqueness:      ${GREEN}$([ $uniqueness_fail -eq 0 ] && echo "passed" || echo "failed")${NC}"
[ $skip_count -gt 0 ] && echo -e "  ${YELLOW}Skipped:${NC}        $skip_count"
echo ""

if [ $((structure_fail + link_fail + uniqueness_fail)) -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the failed structure checks."
    exit 1
else
    print_status_passed
    exit 0
fi