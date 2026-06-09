#!/usr/bin/env bash
# =============================================================================
# Test: Team Content
# =============================================================================
# Validates content quality for all teams.
# Rules tested (all via skill_validator.py validate-team --subset=content):
#   error level (blocking):
#     T-CON-01: directory naming format ^[a-z0-9]+(-[a-z0-9]+)*$
#     T-CON-02: description contains trigger keywords (skipped if disable-model-invocation)
#   warn level (advisory):
#     T-CON-03: description contains trigger conditions
#
# Supports incremental testing via INCREMENTAL_TEAMS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Team Content ==="
echo ""
echo "This test validates content quality for all teams."
echo "Run time: ~10 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed teams"
    echo ""
fi

# Counters
total_teams=0
pass_count=0
fail_count=0
skip_count=0

# Get teams to test (filtered if in incremental mode)
TEAMS_TO_TEST=$(get_teams_to_test)
total_teams=$(echo "$TEAMS_TO_TEST" | grep -c . || echo "0")

echo "Teams to test: $total_teams"
echo ""

# ============================================
# Validate teams content
# ============================================
print_section_header "Test: Team Content (T-CON-01 to T-CON-03)"

# Collect files for batch validation
batch_files=()
for team in $TEAMS_TO_TEST; do
    [ -z "$team" ] && continue

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

    batch_files+=("$team_file")
done

if [ ${#batch_files[@]} -gt 0 ]; then
    skill_paths_csv=$(get_all_skills_with_paths | cut -d: -f2- | tr '\n' ',')
    batch_output=$(validate_teams_content_batch "$skill_paths_csv" "${batch_files[@]}" 2>&1) || true
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
echo -e " ${BOLD}Team Content Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total teams: $total_teams"
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