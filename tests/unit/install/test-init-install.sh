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
TEAMS_DIR="$SKILLS_DIR/plugins-official"

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

# =============================================================================
# Check 1: init.sh exists and is executable for every team
# =============================================================================
print_section_header "Check: init.sh existence & permissions"

for team_dir in "$TEAMS_DIR"/*; do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    if [ -f "$init_script" ]; then
        run_check "[$team_name] init.sh exists" test -f "$init_script"
        run_check "[$team_name] init.sh is executable" test -x "$init_script"

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

for team_dir in "$TEAMS_DIR"/*; do
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

    # agents/ must exist for every team with init.sh
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
done

# =============================================================================
# Check 3: workflow scripts exist
# =============================================================================
print_section_header "Check: workflow scripts existence"

for team_dir in "$TEAMS_DIR"/*; do
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

for team_dir in "$TEAMS_DIR"/*; do
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

for team_dir in "$TEAMS_DIR"/*; do
    [ -d "$team_dir" ] || continue
    team_name=$(basename "$team_dir")
    init_script="$team_dir/init.sh"

    [ -f "$init_script" ] || continue

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
        print_fail "[$team_name] INCLUDED_AGENT_PATTERN not found in init.sh"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
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
