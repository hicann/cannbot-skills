#!/usr/bin/env bash
# =============================================================================
# Test: Skill Content
# =============================================================================
# Validates content quality for all skills.
# Rules tested (all via skill_validator.py validate-skill --subset=content):
#   Aligned with Agent Skills / OpenCode / Claude Code specifications.
#   error level (blocking):
#     S-CON-01: name matches directory name
#     S-CON-02: description contains trigger keywords (skipped if disable-model-invocation)
#   warn level (advisory):
#     S-CON-03: description contains trigger conditions
#     S-CON-04: long files link to supporting files (progressive disclosure, ≤500 lines)
#     S-CON-05: no anti-pattern phrases in description
#     S-CON-06: file references kept one level deep (Agent Skills spec)
#
# Supports incremental testing via INCREMENTAL_SKILLS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Content ==="
echo ""
echo "This test validates content quality for all skills."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed skills"
    echo ""
fi

# Counters
total_skills=0
pass_count=0
fail_count=0
skip_count=0

# Get skills to test (filtered if in incremental mode)
SKILLS_TO_TEST=$(get_skills_to_test)
total_skills=$(echo "$SKILLS_TO_TEST" | grep -c . || echo "0")

echo "Skills to test: $total_skills"
echo ""

# ============================================
# Validate skills content
# ============================================
print_section_header "Test: Skill Content (S-CON-01 to S-CON-09, S-STR-13)"

# Collect files for batch validation
batch_files=()
for skill in $SKILLS_TO_TEST; do
    [ -z "$skill" ] && continue

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

    batch_files+=("$skill_file")
done

if [ ${#batch_files[@]} -gt 0 ]; then
    batch_output=$(validate_skills_content_batch "${batch_files[@]}" 2>&1) || true
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
echo -e " ${BOLD}Skill Content Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total skills: $total_skills"
echo -e "  ${GREEN}Passed:${NC}       $pass_count"
echo -e "  ${RED}Failed:${NC}       $fail_count"
echo -e "  ${YELLOW}Skipped:${NC}      $skip_count"
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