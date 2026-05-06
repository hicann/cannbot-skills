#!/usr/bin/env bash
# =============================================================================
# Test: Skill Structure
# =============================================================================
# Validates structure correctness for all skills.
# Rules tested:
# - S-STR-01: YAML Front Matter format (---包裹)
# - S-STR-02: name field exists
# - S-STR-03: description field exists
# - S-STR-04: references/ directory not empty (if exists)
# - S-STR-05: name length 1-64 characters
# - S-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
# - S-STR-07: description length 1-1024 characters
# - S-STR-08: All links point to existing files
#
# Supports incremental testing via INCREMENTAL_SKILLS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Structure ==="
echo ""
echo "This test validates structure for all skills."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed skills"
    echo ""
fi

# Counters
total_skills=0
structure_pass=0
structure_fail=0
link_pass=0
link_fail=0
skip_count=0

# Get skills to test (filtered if in incremental mode)
SKILLS_TO_TEST=$(get_skills_to_test)
total_skills=$(echo "$SKILLS_TO_TEST" | grep -c . || echo "0")

echo "Skills to test: $total_skills"
echo ""

# ============================================
# Test 1: Skill Structure Validation
# ============================================
print_section_header "Test: Skill Structure (S-STR-01 to S-STR-07)"

for skill in $SKILLS_TO_TEST; do
    [ -z "$skill" ] && continue

    # In incremental mode, check if this skill should be tested
    if is_incremental_mode && ! should_test_skill "$skill"; then
        print_skip "$skill: Not in changed list"
        ((skip_count++)) || true
        continue
    fi

    skill_file=$(find_skill_file "$skill")

    if [ ! -f "$skill_file" ]; then
        print_skip "$skill: SKILL.md not found"
        ((skip_count++)) || true
        continue
    fi

    if validate_skill_structure "$skill_file"; then
        ((structure_pass++)) || true
    else
        ((structure_fail++)) || true
    fi
done

echo ""

# ============================================
# Test 2: Link Validity
# ============================================
print_section_header "Test: Link Validity (S-STR-08)"

# Get skills with paths for link testing
while IFS=: read -r sname spath; do
    [ -z "$sname" ] && continue

    # In incremental mode, only check links for changed skills
    if is_incremental_mode && ! should_test_skill "$sname"; then
        continue
    fi

    if [ -f "$spath" ]; then
        if check_file_links "$spath" "skill"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done <<< "$(get_all_skills_with_paths)"

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Skill Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total skills: $total_skills"
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