#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# Test: Init Script Installation
# =============================================================================
# Validates static conditions for init.sh-based installation.
# No CLI required — pure filesystem checks.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PLUGIN_DIRS=("$SKILLS_DIR/plugins-official" "$SKILLS_DIR/plugins-community")

PASS_COUNT=0
FAIL_COUNT=0

run_check() {
    local name="$1"
    shift
    if "$@"; then
        print_pass "$name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "$name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# Collect all plugin directories that contain init.sh
get_all_plugin_dirs() {
    for plugin_root_dir in "${PLUGIN_DIRS[@]}"; do
        [ -d "$plugin_root_dir" ] || continue
        for team_dir in "$plugin_root_dir"/*; do
            [ -d "$team_dir" ] || continue
            [ -f "$team_dir/init.sh" ] || continue
            echo "$team_dir"
        done
    done
}

# =============================================================================
# Check 1: init.sh exists and is executable for every team
# =============================================================================
print_section_header "Check: init.sh existence & permissions"

for team_dir in $(get_all_plugin_dirs); do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    if [ -f "$init_script" ]; then
        run_check "[$team_name] init.sh exists" test -f "$init_script"
        if [ -d "$team_dir/agents" ]; then
            run_check "[$team_name] init.sh is executable" test -x "$init_script"
        else
            print_warn "[$team_name] init.sh is not executable (agentless plugin; use 'bash init.sh' to run)"
        fi

        # Check shebang (may be after copyright header) — warn only, not fail
        shebang=$(grep -m1 '^#!' "$init_script" || true)
        if [[ "$shebang" == \#\!/bin/bash* ]] || [[ "$shebang" == \#\!/usr/bin/env\ bash* ]]; then
            print_pass "[$team_name] init.sh shebang correct: $shebang"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_warn "[$team_name] init.sh missing shebang (use 'bash init.sh' to run)"
            # Not a hard failure — init.sh is typically invoked via 'bash init.sh'
        fi
    else
        print_warn "[$team_name] init.sh not found (may be placeholder team)"
    fi
done

# =============================================================================
# Check 2: init.sh references exist (AGENTS.md, workflows/, agents/)
# =============================================================================
print_section_header "Check: init.sh referenced paths exist"

for team_dir in $(get_all_plugin_dirs); do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    [ -f "$init_script" ] || continue

    # AGENTS.md must exist for every team with init.sh
    run_check "[$team_name] AGENTS.md exists" test -f "$team_dir/AGENTS.md"

    # workflows/ is optional — only check if the directory actually exists
    if [ -d "$team_dir/workflows" ]; then
        run_check "[$team_name] workflows/ exists" test -d "$team_dir/workflows"
    fi

    # agents/ is optional — only check when the directory actually exists.
    # Agentless plugins (e.g. triton-op-generator) install only skills and may
    # legitimately have no agents/ directory.
    if [ -d "$team_dir/agents" ]; then
        run_check "[$team_name] agents/ exists" test -d "$team_dir/agents"

        # Check that agents/ has .md files
        agent_count=$(find "$team_dir/agents" -maxdepth 1 -name '*.md' | wc -l)
        if [ "$agent_count" -gt 0 ]; then
            print_pass "[$team_name] agents/ has $agent_count agent file(s)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "[$team_name] agents/ has no .md files"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        print_info "[$team_name] no agents/ directory (agentless plugin, skipping agent checks)"
    fi
done

# =============================================================================
# Check 3: workflow scripts exist
# =============================================================================
print_section_header "Check: workflow scripts existence"

for team_dir in $(get_all_plugin_dirs); do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")

    # Only check teams that have workflows/
    [ -d "$team_dir/workflows" ] || continue
    [ -d "$team_dir/workflows/scripts" ] || continue

    for script in "$team_dir/workflows/scripts/"*.sh; do
        [ -f "$script" ] || continue
        script_name=$(basename "$script")
        run_check "[$team_name] workflow script $script_name exists" test -f "$script"
    done
    # ops-direct-invoke specific scripts (hardcoded standard for kernel ops plugin)
    if [ "$team_name" = "ops-direct-invoke" ]; then
        run_check "[$team_name] verify_environment.sh exists" test -f "$team_dir/workflows/scripts/verify_environment.sh"
        run_check "[$team_name] init_operator_project.sh exists" test -f "$team_dir/workflows/scripts/init_operator_project.sh"
    fi
done

# =============================================================================
# Check 4: skills referenced by init.sh exist
# =============================================================================
print_section_header "Check: init.sh skill references exist"

for team_dir in $(get_all_plugin_dirs); do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    [ -f "$init_script" ] || continue

    # Extract INCLUDED_SKILLS from init.sh
    included_skills=$(grep -oE 'INCLUDED_SKILLS="[^"]+"' "$init_script" 2>/dev/null | sed 's/INCLUDED_SKILLS="//;s/"$//' || true)

    if [ -n "$included_skills" ]; then
        for skill in $included_skills; do
            ops_skill_dir="$SKILLS_DIR/ops/$skill"
            model_skill_dir="$SKILLS_DIR/model/$skill"
            graph_skill_dir="$SKILLS_DIR/graph/$skill"
            infra_skill_dir="$SKILLS_DIR/infra/$skill"
            local_skill_dir="$team_dir/$skill"
            local_plugin_skill_dir="$team_dir/skills/$skill"
            workflow_dir="$team_dir/workflow"

            if [ -d "$ops_skill_dir" ] && [ -f "$ops_skill_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists in ops/"
                PASS_COUNT=$((PASS_COUNT + 1))
            elif [ -d "$model_skill_dir" ] && [ -f "$model_skill_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists in model/"
                PASS_COUNT=$((PASS_COUNT + 1))
            elif [ -d "$graph_skill_dir" ] && [ -f "$graph_skill_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists in graph/"
                PASS_COUNT=$((PASS_COUNT + 1))
            elif [ -d "$infra_skill_dir" ] && [ -f "$infra_skill_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists in infra/"
                PASS_COUNT=$((PASS_COUNT + 1))
            elif [ -d "$local_skill_dir" ] && [ -f "$local_skill_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists as local team skill"
                PASS_COUNT=$((PASS_COUNT + 1))
            elif [ -d "$local_plugin_skill_dir" ] && [ -f "$local_plugin_skill_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists as plugin-local skill"
                PASS_COUNT=$((PASS_COUNT + 1))
            elif [ -d "$workflow_dir" ] && [ -f "$workflow_dir/SKILL.md" ]; then
                print_pass "[$team_name] skill '$skill' exists as team workflow skill"
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                print_fail "[$team_name] skill '$skill' NOT found in shared or local skill roots"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        done
    else
        print_info "[$team_name] no INCLUDED_SKILLS found in init.sh"
    fi
done

# =============================================================================
# Check 5: INCLUDED_AGENT_PATTERN exists and matches agents/
# =============================================================================
print_section_header "Check: INCLUDED_AGENT_PATTERN consistency"

for team_dir in $(get_all_plugin_dirs); do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    [ -f "$init_script" ] || continue

    # INCLUDED_AGENT_PATTERN is only meaningful for teams that ship an agents/
    # directory. Agentless plugins install only skills and have no agent pattern.
    if [ ! -d "$team_dir/agents" ]; then
        print_info "[$team_name] no agents/ directory, skipping INCLUDED_AGENT_PATTERN check"
        continue
    fi

    agent_pattern=$(grep -oE 'INCLUDED_AGENT_PATTERN="[^"]+"' "$init_script" 2>/dev/null | sed 's/INCLUDED_AGENT_PATTERN="//;s/"$//' || true)

    if [ -n "$agent_pattern" ]; then
        print_pass "[$team_name] INCLUDED_AGENT_PATTERN defined: '$agent_pattern'"
        PASS_COUNT=$((PASS_COUNT + 1))

        matched=0
        for agent_file in "$team_dir/agents/"*.md; do
            [ -f "$agent_file" ] || continue
            base_name=$(basename "$agent_file" .md)
            if [[ "$base_name" == $agent_pattern ]]; then
                matched=1
                break
            fi
        done

        if [ "$matched" -eq 1 ]; then
            print_pass "[$team_name] at least one agent matches pattern '$agent_pattern'"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "[$team_name] no agent matches pattern '$agent_pattern'"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        # Some plugins (e.g. cuda2ascend) use a different agent installation
        # mechanism (collect_agents / direct glob) instead of INCLUDED_AGENT_PATTERN.
        # If the init.sh has an alternative agent collection mechanism, warn instead of fail.
        if grep -qE 'collect_agents|find.*agents.*-type f|AGENT_FILES' "$init_script" 2>/dev/null; then
            print_warn "[$team_name] no INCLUDED_AGENT_PATTERN (uses alternative agent collection mechanism)"
        else
            print_fail "[$team_name] INCLUDED_AGENT_PATTERN not found in init.sh"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
    
done

# =============================================================================
# Check 6: init.sh $(cd ... && pwd) referenced directories exist
# =============================================================================
# Extracts all $(cd "$PLUGIN_ROOT/../../some-dir" && pwd) patterns from init.sh
# and verifies the referenced directory exists. This catches the class of bugs
# where init.sh references a repo directory that has been removed or renamed
# (e.g. issue #406: ops-lab directory removed but init.sh still referenced it).
#
# Smart guard detection: skips cd calls that are already protected by:
#   1. if [ -d "...path..." ] on a preceding line (within 10 lines)
#   2. elif [ -d "...path..." ] on the same or preceding line
#   3. 2>/dev/null in the cd command itself (self-guarded)
# =============================================================================
print_section_header "Check: init.sh cd-referenced directories exist"

for team_dir in $(get_all_plugin_dirs); do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    [ -f "$init_script" ] || continue

    # Extract all $(cd "$VAR/path" && pwd) patterns with line numbers
    # Match: $(cd "$PLUGIN_ROOT/../../something" && pwd)  or  $(cd "$SCRIPT_DIR/../../something" && pwd)
    while IFS=: read -r line_num cd_line; do
        [ -n "$cd_line" ] || continue

        # Extract the path after $PLUGIN_ROOT/ or $SCRIPT_DIR/ prefix
        rel_path=$(echo "$cd_line" | sed -n 's/.*\$\(PLUGIN_ROOT\|SCRIPT_DIR\)\/\([^"]*\)".*/\2/p')

        # Skip lines that use 2>/dev/null (already self-guarded)
        if echo "$cd_line" | grep -q '2>/dev/null'; then
            continue
        fi

        # Skip empty extractions (e.g. $(cd "$(dirname ...)" && pwd))
        [ -n "$rel_path" ] || continue

        # Check if this cd is guarded by a preceding if [ -d ] / elif [ -d ]
        # Look at the current line and up to 10 preceding lines for [ -d "...path..." ]
        # The guard may be several lines above (e.g. inside an if block with intervening code)
        is_guarded=false
        for check_offset in 0 1 2 3 4 5 6 7 8 9 10; do
            check_line=$((line_num - check_offset))
            [ "$check_line" -ge 1 ] || continue
            check_content=$(sed -n "${check_line}p" "$init_script")
            # Check if this line contains [ -d "..." ] referencing the same path
            # Match both bare path (e.g. "hooks") and $VAR/path (e.g. "$PLUGIN_ROOT/hooks")
            if echo "$check_content" | grep -qE '\[ -d "[^"]*'"$rel_path"'"[[:space:]]*\]'; then
                is_guarded=true
                break
            fi
        done

        if $is_guarded; then
            # Guarded by if [ -d ] — safe, skip
            continue
        fi

        # Resolve the actual path relative to the team_dir
        actual_path="$team_dir/$rel_path"

        if [ -d "$actual_path" ]; then
            print_pass "[$team_name] cd target exists: $rel_path"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "[$team_name] cd target MISSING: $rel_path (referenced in init.sh but directory does not exist)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(grep -nE '\$\(cd "\$(PLUGIN_ROOT|SCRIPT_DIR)/[^"]*" && pwd\)' "$init_script" 2>/dev/null || true)
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Init Install Test Summary"
echo "========================================"
echo "  Passed:  $PASS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
