#!/usr/bin/env bash
# =============================================================================
# Test: Team Structure
# =============================================================================
# Validates structure correctness for all teams.
# Rules tested:
# - T-STR-01: YAML Front Matter format (---wrapped)
# - T-STR-02: description field exists
# - T-STR-03: mode field exists and is "primary"
# - T-STR-04: skills field exists
# - T-STR-05: All skill dependencies exist
# - T-STR-06: description length 1-1024 characters
# - T-STR-07: references/ directory not empty (if exists)
# - T-STR-07: Team name uniqueness across all teams
# - T-STR-08: All links point to existing files
#
# Supports incremental testing via INCREMENTAL_TEAMS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Team Structure ==="
echo ""
echo "This test validates structure for all teams."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed teams"
    echo ""
fi

# Counters
total_teams=0
structure_pass=0
structure_fail=0
link_pass=0
link_fail=0
skip_count=0

# Get teams to test (filtered if in incremental mode)
TEAMS_TO_TEST=$(get_teams_to_test)
total_teams=$(echo "$TEAMS_TO_TEST" | grep -c . || echo "0")

echo "Teams to test: $total_teams"
echo ""

# ============================================
# Test 1: Team Structure Validation
# ============================================
print_section_header "Test: Team Structure (T-STR-01 to T-STR-07)"

for team in $TEAMS_TO_TEST; do
    [ -z "$team" ] && continue

    # In incremental mode, check if this team should be tested
    if is_incremental_mode && ! should_test_team "$team"; then
        print_skip "$team: Not in changed list"
        ((skip_count++)) || true
        continue
    fi

    team_file=$(find_team_file "$team")

    if [ ! -f "$team_file" ]; then
        print_skip "$team: AGENTS.md not found"
        ((skip_count++)) || true
        continue
    fi

    if validate_team_structure "$team_file"; then
        ((structure_pass++)) || true
    else
        ((structure_fail++)) || true
    fi
done

echo ""

# ============================================
# Test 2: Link Validity
# ============================================
print_section_header "Test: Link Validity (T-STR-08)"

while IFS=: read -r tname tpath; do
    [ -z "$tname" ] && continue

    # In incremental mode, only check links for changed teams
    if is_incremental_mode && ! should_test_team "$tname"; then
        continue
    fi

    if [ -f "$tpath" ]; then
        if check_file_links "$tpath" "team"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done <<< "$(get_all_teams_with_paths)"

echo ""

# ============================================
# Test 3: Global Uniqueness
# ============================================
print_section_header "Test: Team Name Uniqueness (T-STR-07)"

uniqueness_fail=0
if ! validate_global_uniqueness "team"; then
    uniqueness_fail=1
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Team Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total teams: $total_teams"
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