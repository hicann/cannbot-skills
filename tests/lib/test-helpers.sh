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
# Helper functions for CANN Skills tests
# =============================================================================
# Supports: Claude Code + OpenCode dual platform
# Features: Test isolation, CI/CD ready, JSON output
# =============================================================================

set -euo pipefail

# Get the directories
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$LIB_DIR/.." && pwd)"
SKILLS_DIR="$(cd "$LIB_DIR/../.." && pwd)"

# =============================================================================
# Color Output
# =============================================================================
# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if colors should be enabled (respects NO_COLOR env var and non-TTY)
# Supports FORCE_COLOR=1 to force enable colors in CI/non-TTY environments
setup_colors() {
    if [ -n "${NO_COLOR:-}" ]; then
        disable_colors
    elif [ -n "${FORCE_COLOR:-}" ]; then
        enable_colors
    elif [ -t 1 ]; then
        enable_colors
    else
        disable_colors
    fi
}

enable_colors() {
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
}

disable_colors() {
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
}

# Print colored status - these are the main functions to use
print_pass() {
    echo -e "  ${GREEN}[PASS]${NC} $*"
}

print_fail() {
    echo -e "  ${RED}[FAIL]${NC} $*"
}

print_skip() {
    echo -e "  ${YELLOW}[SKIP]${NC} $*"
}

print_info() {
    echo -e "  ${BLUE}[INFO]${NC} $*"
}

print_warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $*"
}

print_error() {
    echo -e "  ${RED}[ERROR]${NC} $*"
}

print_section_header() {
    local name="$1"
    echo ""
    echo -e "${BOLD}${CYAN}=== $name ===${NC}"
    echo ""
}

# Parse captured test output and print a compact summary.
# Extracts [PASS] / [FAIL] / [SKIP] / [WARN] line counts, shows failure
# details inline. When show_all is true, also prints the full captured output.
#
# Usage: print_compact_result "$output" "$status" "$duration" [show_all]
print_compact_result() {
    local output="$1"
    local status="$2"
    local duration="$3"
    local show_all="${4:-false}"

    local n_pass=0 n_fail=0 n_skip=0 n_warn=0
    if [[ -n "$output" ]]; then
        n_pass=$(echo "$output" | grep -cE '\[PASS\]' 2>/dev/null || echo 0)
        n_fail=$(echo "$output" | grep -cE '\[FAIL\]' 2>/dev/null || echo 0)
        n_skip=$(echo "$output" | grep -cE '\[SKIP\]' 2>/dev/null || echo 0)
        n_warn=$(echo "$output" | grep -cE '\[WARN\]' 2>/dev/null || echo 0)
        # Ensure valid numbers
        [[ "$n_pass" =~ ^[0-9]+$ ]] || n_pass=0
        [[ "$n_fail" =~ ^[0-9]+$ ]] || n_fail=0
        [[ "$n_skip" =~ ^[0-9]+$ ]] || n_skip=0
        [[ "$n_warn" =~ ^[0-9]+$ ]] || n_warn=0
    fi

    # Build compact counts line
    local counts=""
    [[ $n_pass -gt 0 ]] && counts+=" ${GREEN}${n_pass} passed${NC}"
    [[ $n_fail -gt 0 ]] && counts+=" ${RED}${n_fail} failed${NC}"
    [[ $n_skip -gt 0 ]] && counts+=" ${YELLOW}${n_skip} skipped${NC}"
    [[ $n_warn -gt 0 ]] && counts+=" ${YELLOW}${n_warn} warnings${NC}"

    case "$status" in
        pass)
            if [[ -z "$counts" ]]; then
                echo -e "  ${GREEN}[PASS]${NC} (${duration}s)"
            else
                echo -e "  ${GREEN}[PASS]${NC} (${counts}, ${duration}s)"
            fi
            # Show warnings if any
            if [[ $n_warn -gt 0 ]]; then
                echo ""
                echo "$output" | grep -E '\[WARN\]' | head -20 | sed 's/^/    /'
                echo ""
            fi
            ;;
        fail)
            echo -e "  ${RED}[FAIL]${NC} (${counts}, ${duration}s)"

            # Extract and show failure lines
            if [[ $n_fail -gt 0 ]]; then
                echo ""
                echo -e "  ${YELLOW}── Failures ──${NC}"
                echo "$output" | grep -E '\[FAIL\]' | head -30 | sed 's/^/    /'
                echo ""
            fi

            # Also show ERROR lines if any
            local n_err
            n_err=$(echo "$output" | grep -cE '\[ERROR\]' 2>/dev/null || echo 0)
            [[ "$n_err" =~ ^[0-9]+$ ]] || n_err=0
            if [[ $n_err -gt 0 ]]; then
                echo -e "  ${RED}── Errors ──${NC}"
                echo "$output" | grep -E '\[ERROR\]' | head -20 | sed 's/^/    /'
                echo ""
            fi
            ;;
    esac

    # Show full output when verbose
    if $show_all; then
        echo ""
        echo -e "  ${BLUE}── Full Output ──${NC}"
        echo "$output" | sed 's/^/  /'
        echo -e "  ${BLUE}── End ──${NC}"
        echo ""
    fi
}

# Print colored summary status
print_status_passed() {
    echo -e "${GREEN}${BOLD}STATUS: PASSED${NC}"
}

print_status_failed() {
    echo -e "${RED}${BOLD}STATUS: FAILED${NC}"
}

# Initialize colors at source time
setup_colors

# =============================================================================
# Configuration
# =============================================================================

# Default configuration
DEFAULT_TIMEOUT=60
DEFAULT_PLATFORM="opencode"  # claude, opencode, or all

# Test results tracking
declare -g TEST_PASSED=0
declare -g TEST_FAILED=0
declare -g TEST_SKIPPED=0
declare -g TEST_START_TIME=0

# =============================================================================
# Platform Detection
# =============================================================================

# Detect available platforms

# Check if a specific platform is available
# Usage: is_platform_available "claude"
is_platform_available() {
    local platform="$1"
    case "$platform" in
        claude)  command -v claude &> /dev/null ;;
        opencode) command -v opencode &> /dev/null ;;
        *) return 1 ;;
    esac
}

# Get platform version
# Usage: get_platform_version "claude"
get_platform_version() {
    local platform="$1"
    case "$platform" in
        claude)  claude --version 2>/dev/null || echo "unknown" ;;
        opencode) opencode --version 2>/dev/null || echo "unknown" ;;
        *) echo "unknown" ;;
    esac
}

# =============================================================================
# Test Project Management
# =============================================================================

# Create a temporary test project directory
# Usage: test_project=$(create_test_project [prefix])
create_test_project() {
    local prefix="${1:-cann-test}"
    local test_dir=$(mktemp -d -t "${prefix}.XXXXXX")
    echo "$test_dir"
}

# Cleanup test project
# Usage: cleanup_test_project "$test_dir"
cleanup_test_project() {
    local test_dir="$1"
    if [ -d "$test_dir" ]; then
        rm -rf "$test_dir"
    fi
}

# =============================================================================
# Platform-specific Runners
# =============================================================================

# Run Claude Code with a prompt and capture output
# Usage: run_claude "prompt text" [timeout_seconds] [allowed_tools]
run_claude() {
    local prompt="$1"
    local timeout="${2:-$DEFAULT_TIMEOUT}"
    local allowed_tools="${3:-}"
    local output_file=$(mktemp)

    # Build command
    local cmd="claude -p \"$prompt\""
    if [ -n "$allowed_tools" ]; then
        cmd="$cmd --allowed-tools=$allowed_tools"
    fi

    # Run Claude in headless mode with timeout
    if timeout "$timeout" bash -c "$cmd" > "$output_file" 2>&1; then
        cat "$output_file"
        rm -f "$output_file"
        return 0
    else
        local exit_code=$?
        cat "$output_file" >&2
        rm -f "$output_file"
        return $exit_code
    fi
}


# =============================================================================
# Assertions
# =============================================================================

# Check if pattern A appears before pattern B
# Usage: assert_order "output" "pattern_a" "pattern_b" "test name"
assert_order() {
    local output="$1"
    local pattern_a="$2"
    local pattern_b="$3"
    local test_name="${4:-test}"

    local line_a=$(echo "$output" | grep -inE "$pattern_a" | head -1 | cut -d: -f1)
    local line_b=$(echo "$output" | grep -inE "$pattern_b" | head -1 | cut -d: -f1)

    if [ -z "$line_a" ]; then
        print_fail "$test_name: pattern A not found: $pattern_a"
        return 1
    fi

    if [ -z "$line_b" ]; then
        print_fail "$test_name: pattern B not found: $pattern_b"
        return 1
    fi

    if [ "$line_a" -lt "$line_b" ]; then
        print_pass "$test_name (A at line $line_a, B at line $line_b)"
        return 0
    elif [ "$line_a" -eq "$line_b" ]; then
        # Same line: check character position order
        local the_line
        the_line=$(echo "$output" | sed -n "${line_a}p")
        local pos_a pos_b
        pos_a=$(echo "$the_line" | grep -iobE "$pattern_a" | head -1 | cut -d: -f1)
        pos_b=$(echo "$the_line" | grep -iobE "$pattern_b" | head -1 | cut -d: -f1)
        if [ -n "$pos_a" ] && [ -n "$pos_b" ] && [ "$pos_a" -le "$pos_b" ]; then
            print_pass "$test_name (both on line $line_a, A at pos $pos_a, B at pos $pos_b)"
            return 0
        elif [ -n "$pos_a" ] && [ -n "$pos_b" ]; then
            print_fail "$test_name"
            echo -e "  ${YELLOW}Expected '$pattern_a' before '$pattern_b' on line $line_a${NC}"
            echo "  But found A at pos $pos_a, B at pos $pos_b"
            return 1
        else
            print_pass "$test_name (both on line $line_a)"
            return 0
        fi
    else
        print_fail "$test_name"
        echo -e "  ${YELLOW}Expected '$pattern_a' before '$pattern_b'${NC}"
        echo "  But found A at line $line_a, B at line $line_b"
        return 1
    fi
}

# Assert file exists
# Usage: assert_file_exists "/path/to/file" "test name"
assert_file_exists() {
    local file="$1"
    local test_name="${2:-file exists}"

    if [ -f "$file" ]; then
        print_pass "$test_name"
        return 0
    else
        print_fail "$test_name"
        echo "  File not found: $file"
        return 1
    fi
}

# =============================================================================
# File Link Validation
# =============================================================================

# Check link validity in a file (for SKILL.md and agent <name>.md)
# Usage: check_file_links "/path/to/file.md" "skill|agent"
# Returns: 0 if all links valid, 1 if broken links found
check_file_links() {
    local file="$1"
    local file_type="$2"
    local file_dir="$(dirname "$file")"
    local item_name
    # For flat agent layout (agents/<name>.md), basename(dirname) is the
    # generic "agents" directory, which leaks no information. Use the file
    # stem instead. For skills/teams the file itself is SKILL.md / AGENTS.md,
    # so basename(dirname) is still the semantically meaningful name.
    if [ "$file_type" = "agent" ]; then
        item_name=$(basename "$file" .md)
    else
        item_name=$(basename "$file_dir")
    fi
    local broken_links=()

    # Delegate link extraction to Python — shell sed chokes on URLs that
    # contain '/' which must not be treated as the sed delimiter.
    local links
    links=$(python3 - "$file" <<'PY'
import re, sys
path = sys.argv[1]
try:
    text = open(path, encoding="utf-8", errors="replace").read()
except OSError:
    sys.exit(0)
out = set()
# [text](references/...) or [text](./...)
for m in re.finditer(r'\]\((references/[^)\s]+|\./[^)\s]+)\)', text):
    out.add(m.group(1))
# {file:./references/...}
for m in re.finditer(r'\{file:([^}\s]+)\}', text):
    out.add(m.group(1))
for raw in out:
    link = raw.split('#', 1)[0]
    if link.startswith('./'):
        link = link[2:]
    if link:
        print(link)
PY
)

    while IFS= read -r link; do
        [ -z "$link" ] && continue
        if [ ! -e "$file_dir/$link" ]; then
            broken_links+=("$link")
        fi
    done <<< "$links"

    if [ ${#broken_links[@]} -gt 0 ]; then
        print_fail "$file_type/$item_name: Broken links:"
        echo -e "    ${YELLOW}${broken_links[*]}${NC}"
        return 1
    else
        print_pass "$file_type/$item_name: All links valid"
        return 0
    fi
}

# =============================================================================
# Behavior Test Helpers
# =============================================================================

# Run a behavior test with Claude CLI
# Usage: run_behavior_test "test_name" "prompt" "expected_pattern" [timeout]
# Globals: pass_count, fail_count, skip_count should be defined before calling
run_behavior_test() {
    local name="$1"
    local prompt="$2"
    local expected="$3"
    local timeout="${4:-25}"
    
    echo "Testing: $name"
    
    if output=$(timeout "$timeout" claude -p "$prompt 用 1 行回答。" 2>&1); then
        if echo "$output" | grep -qiE "$expected"; then
            print_pass "Correct response"
            pass_count=$((pass_count + 1))
        else
            print_fail "Incorrect response"
            echo -e "  ${YELLOW}Expected:${NC} $expected"
            echo "  Output: ${output:0:80}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

# =============================================================================
# Premature Action Analysis
# =============================================================================

# Analyze premature actions in a session file
# Usage: analyze_premature_actions "$session_file" "skill|agent" "target_name"
analyze_premature_actions() {
    local session_file="$1"
    local target_type="$2"  # "skill" or "agent"
    local target_name="$3"
    
    if [ ! -f "$session_file" ]; then
        echo "  [SKIP] Session file not found"
        return 0
    fi
    
    local passed=true
    local tool_invoked=false
    local first_tool_line=""
    
    # Find the first tool invocation (Skill or Task)
    local search_pattern='"name":"Skill"'
    [ "$target_type" = "agent" ] && search_pattern='"name":"Task"'
    
    if grep -q "$search_pattern" "$session_file" 2>/dev/null; then
        tool_invoked=true
        first_tool_line=$(grep -n "$search_pattern" "$session_file" 2>/dev/null | head -1 | cut -d: -f1)
    fi
    
    # Check if the target was invoked
    if $tool_invoked; then
        local target_pattern
        if [ "$target_type" = "skill" ]; then
            target_pattern='"skill":"([^"]*:)?'"${target_name}"'"'
        else
            target_pattern="\"subagent_type\":\"[^\"]*${target_name}[^\"]*\""
        fi
        
        if grep -qE "$target_pattern" "$session_file" 2>/dev/null; then
            print_pass "Target $target_type '$target_name' was invoked"
        else
            print_info "Other ${target_type}s were invoked instead of '$target_name'"
            if [ "$target_type" = "skill" ]; then
                local triggered=$(grep -o '"skill":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/^/    - /')
            else
                local triggered=$(grep -o '"subagent_type":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/^/    - /')
            fi
            if [ -n "$triggered" ]; then
                print_info "Triggered ${target_type}s:"
                echo "$triggered"
            fi
        fi
    else
        if [ "$target_type" = "skill" ]; then
            print_warn "No Skill tool was invoked in this session"
        else
            print_info "No explicit Task tool invocation (agent may be triggered implicitly)"
        fi
    fi
    
    # Check for premature Write/Edit operations
    if $tool_invoked && [ -n "$first_tool_line" ]; then
        local premature_tools=$(head -n "$first_tool_line" "$session_file" 2>/dev/null | \
            grep '"type":"tool_use"' 2>/dev/null | \
            grep -v '"name":"Skill"' 2>/dev/null | \
            grep -v '"name":"Task"' 2>/dev/null | \
            grep -v '"name":"TodoWrite"' 2>/dev/null | \
            grep -v '"name":"Read"' 2>/dev/null | \
            grep -v '"name":"Glob"' 2>/dev/null | \
            grep -v '"name":"Grep"' 2>/dev/null | \
            grep -E '"name":"(Write|Edit|Bash)"' 2>/dev/null || true)
        
        if [ -n "$premature_tools" ]; then
            print_fail "Premature actions detected BEFORE $target_type invocation:"
            echo "$premature_tools" | head -5 | sed 's/^/    /'
            passed=false
        else
            print_pass "No premature Write/Edit/Bash actions before $target_type"
        fi
    fi
    
    # Check for TodoWrite usage (acceptable before tool)
    local todo_before=0
    if $tool_invoked && [ -n "$first_tool_line" ]; then
        todo_before=$(head -n "$first_tool_line" "$session_file" 2>/dev/null | \
            grep -c '"name":"TodoWrite"' 2>/dev/null | tr -cd '0-9' || true)
        todo_before="${todo_before:-0}"
    fi
    
    if [ "$todo_before" -gt 0 ]; then
        print_info "TodoWrite used $todo_before time(s) before $target_type (acceptable for planning)"
    fi
    
    # Check for Read usage before tool (acceptable for context)
    local read_before=0
    if $tool_invoked && [ -n "$first_tool_line" ]; then
        read_before=$(head -n "$first_tool_line" "$session_file" 2>/dev/null | \
            grep -c '"name":"Read"' 2>/dev/null | tr -cd '0-9' || true)
        read_before="${read_before:-0}"
    fi
    
    if [ "$read_before" -gt 0 ]; then
        print_info "Read used $read_before time(s) before $target_type (acceptable for context)"
    fi
    
    # Summary
    if $passed; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Skill & Agent Queries
# =============================================================================

# Get list of all skills with their full paths
# Returns: skill_name:full_path per line
# Return 0 if $file resides inside a nested git repo (not the repo root itself)
_is_inside_nested_git_repo() {
    local file="$1"
    local dir
    dir=$(dirname "$file")
    while [ "$dir" != "$SKILLS_DIR" ] && [ "$dir" != "/" ]; do
        if [ -d "$dir/.git" ]; then
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

get_all_skills_with_paths() {
    local cache_file="${_TEST_CACHE_DIR:-}/skills_cache"
    if [ -n "${_TEST_CACHE_DIR:-}" ] && [ -f "$cache_file" ]; then
        cat "$cache_file"
        return
    fi

    local tmpfile
    tmpfile=$(mktemp)
    # Prune teams/ so nested team SKILL.md files (e.g.
    # plugins-official/ops-registry-invoke/workflow/SKILL.md) are not mistakenly
    # discovered as top-level skills.
    find "$SKILLS_DIR" \
        \( -name "node_modules" -o -name ".git" -o -name "teams" \) -prune -o \
        \( -path "*/skills/*" -o -path "*/ops/*" -o -path "*/graph/*" -o -path "*/model/*" -o -path "*/infra/*" \) \
        -iname "skill.md" \
        -print 2>/dev/null > "$tmpfile" || true

    local result=""
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        # Skip skills inside nested external git repos (e.g. pypto, asc-devkit)
        if _is_inside_nested_git_repo "$f"; then
            continue
        fi
        result+="$(basename "$(dirname "$f")"):$f"$'\n'
    done < "$tmpfile"
    rm -f "$tmpfile"

    result=$(printf '%s' "$result" | sort -u -t: -k1,1)
    if [ -n "${_TEST_CACHE_DIR:-}" ]; then
        printf '%s\n' "$result" > "$cache_file"
    fi
    printf '%s\n' "$result"
}

# Get list of all skills
get_all_skills() {
    get_all_skills_with_paths | cut -d: -f1
}

# Check if incremental mode is active
is_incremental_mode() {
    [ -n "${INCREMENTAL_SKILLS:-}${INCREMENTAL_AGENTS:-}${INCREMENTAL_TEAMS:-}" ]
}

# Check if a specific skill should be tested in incremental mode
should_test_skill() {
    local skill_name="$1"

    # If not in incremental mode, test everything
    if ! is_incremental_mode; then
        return 0
    fi

    # If INCREMENTAL_SKILLS is set, only test those skills
    if [ -n "${INCREMENTAL_SKILLS:-}" ]; then
        echo " $INCREMENTAL_SKILLS " | grep -q " $skill_name "
        return $?
    fi

    # No skills specified in incremental mode, skip this skill
    return 1
}

# Check if a specific agent should be tested in incremental mode
should_test_agent() {
    local agent_name="$1"

    # If not in incremental mode, test everything
    if ! is_incremental_mode; then
        return 0
    fi

    # If INCREMENTAL_AGENTS is set, only test those agents
    if [ -n "${INCREMENTAL_AGENTS:-}" ]; then
        echo " $INCREMENTAL_AGENTS " | grep -q " $agent_name "
        return $?
    fi

    # No agents specified in incremental mode, skip this agent
    return 1
}

# Check if a specific team should be tested in incremental mode
should_test_team() {
    local team_name="$1"

    # If not in incremental mode, test everything
    if ! is_incremental_mode; then
        return 0
    fi

    # If INCREMENTAL_TEAMS is set, only test those teams
    if [ -n "${INCREMENTAL_TEAMS:-}" ]; then
        echo " $INCREMENTAL_TEAMS " | grep -q " $team_name "
        return $?
    fi

    # No teams specified in incremental mode, skip this team
    return 1
}

# Get filtered list of skills for incremental testing
# Returns: skill_name per line (only changed skills if in incremental mode)
get_skills_to_test() {
    local all_skills
    all_skills=$(get_all_skills)

    if ! is_incremental_mode || [ -z "${INCREMENTAL_SKILLS:-}" ]; then
        echo "$all_skills"
        return
    fi

    # Filter to only changed skills
    for skill in $all_skills; do
        if should_test_skill "$skill"; then
            echo "$skill"
        fi
    done
}

# Get filtered list of agents for incremental testing
get_agents_to_test() {
    local all_agents
    all_agents=$(get_all_agents)

    if ! is_incremental_mode || [ -z "${INCREMENTAL_AGENTS:-}" ]; then
        echo "$all_agents"
        return
    fi

    # Filter to only changed agents
    for agent in $all_agents; do
        if should_test_agent "$agent"; then
            echo "$agent"
        fi
    done
}

# Get filtered list of teams for incremental testing
get_teams_to_test() {
    local all_teams
    all_teams=$(get_all_teams)

    if ! is_incremental_mode || [ -z "${INCREMENTAL_TEAMS:-}" ]; then
        echo "$all_teams"
        return
    fi

    # Filter to only changed teams
    for team in $all_teams; do
        if should_test_team "$team"; then
            echo "$team"
        fi
    done
}

# Find skill file by name
# Usage: find_skill_file "skill-name"
# Returns: full path to SKILL.md
find_skill_file() {
    local skill_name="$1"
    local result
    result=$(get_all_skills_with_paths | grep "^${skill_name}:" | head -1 | cut -d: -f2-)
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "$SKILLS_DIR/skills/$skill_name/SKILL.md"
    fi
}

# Get list of all agents with their full paths
# Returns: agent_name:full_path per line
# Layout: agents/<name>.md (flat). Directory-based agents/<name>/AGENT.md is
# NOT a valid layout and will be ignored by discovery.
get_all_agents_with_paths() {
    local cache_file="${_TEST_CACHE_DIR:-}/agents_cache"
    if [ -n "${_TEST_CACHE_DIR:-}" ] && [ -f "$cache_file" ]; then
        cat "$cache_file"
        return
    fi

    local tmpfile
    tmpfile=$(mktemp)
    # Flat layout: agents/<name>.md (exclude AGENTS.md team files)
    find "$SKILLS_DIR" -path "*/agents/*.md" -not -name "AGENTS.md" \
        -not -path "*/node_modules/*" -not -path "*/.git/*" \
        -not -path "*/operators/*" 2>/dev/null >> "$tmpfile" || true

    local result=""
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        # Skip agents inside nested external git repos (e.g. pypto, asc-devkit)
        if _is_inside_nested_git_repo "$f"; then
            continue
        fi
        local name
        name=$(basename "$f" .md)
        result+="${name}:${f}"$'\n'
    done < "$tmpfile"
    rm -f "$tmpfile"

    result=$(printf '%s' "$result" | sort -u -t: -k1,1)
    if [ -n "${_TEST_CACHE_DIR:-}" ]; then
        printf '%s\n' "$result" > "$cache_file"
    fi
    printf '%s\n' "$result"
}

# Get list of all agents
get_all_agents() {
    get_all_agents_with_paths | cut -d: -f1
}

# Find agent file by name
# Usage: find_agent_file "agent-name"
# Returns: full path to <name>.md under any agents/ directory
find_agent_file() {
    local agent_name="$1"
    local result
    result=$(get_all_agents_with_paths | grep "^${agent_name}:" | head -1 | cut -d: -f2-)
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "$SKILLS_DIR/agents/${agent_name}.md"
    fi
}

# Get list of all teams with their full paths
# Returns: team_name:full_path per line
get_all_teams_with_paths() {
    local cache_file="${_TEST_CACHE_DIR:-}/teams_cache"
    if [ -n "${_TEST_CACHE_DIR:-}" ] && [ -f "$cache_file" ]; then
        cat "$cache_file"
        return
    fi

    local tmpfile
    tmpfile=$(mktemp)
    # Prune .opencode / .claude* / node_modules / .git so OpenCode/Claude
    # auxiliary trees under a team (e.g. <team>/.opencode/AGENTS.md) are not
    # mistaken for top-level teams.
    find "$SKILLS_DIR" \
        \( -name ".opencode" -o -name ".claude" -o -name ".claude-plugin" \
           -o -name "node_modules" -o -name ".git" \) -prune -o \
        -path "*/teams/*/AGENTS.md" -print 2>/dev/null > "$tmpfile" || true

    # Also discover teams under plugins-official/ (top-level AGENTS.md)
    if [ -d "$SKILLS_DIR/plugins-official" ]; then
        find "$SKILLS_DIR/plugins-official" -maxdepth 2 -name "AGENTS.md" \
            -not -path "*/.opencode/*" -not -path "*/.claude/*" \
            -not -path "*/.claude-plugin/*" 2>/dev/null >> "$tmpfile" || true
    fi

    local result=""
    while IFS= read -r f; do
        [ -f "$f" ] && result+="$(basename "$(dirname "$f")"):$f"$'\n'
    done < "$tmpfile"
    rm -f "$tmpfile"

    result=$(printf '%s' "$result" | sort -u -t: -k1,1)
    if [ -n "${_TEST_CACHE_DIR:-}" ]; then
        printf '%s\n' "$result" > "$cache_file"
    fi
    printf '%s\n' "$result"
}

# Get list of all teams
get_all_teams() {
    get_all_teams_with_paths | cut -d: -f1
}

# Find team file by name
# Usage: find_team_file "team-name"
# Returns: full path to AGENTS.md
find_team_file() {
    local team_name="$1"
    local result
    result=$(get_all_teams_with_paths | grep "^${team_name}:" | head -1 | cut -d: -f2-)
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "$SKILLS_DIR/teams/$team_name/AGENTS.md"
    fi
}

# =============================================================================
# Structure & Content Validation Functions (Auto-scan)
# =============================================================================

# Python validator script path
SKILL_VALIDATOR="$LIB_DIR/skill_validator.py"

# Parse JSONL output from skill_validator.py into tab-separated level/rule/msg lines
_parse_jsonl() {
    python3 - <<'PYEOF' "$1"
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            level = obj.get("level", "")
            rule = obj.get("rule", "")
            msg = obj.get("msg", "")
            print(f"{level}\t{rule}\t{msg}")
        except json.JSONDecodeError:
            pass
PYEOF
}

# Run the Python validator and dispatch each JSONL finding to print helpers.
# Usage: _run_validator <item_name> <subcmd> <file> [extra_args...]
# Returns: 0 if no error-level findings, 1 otherwise.
_run_validator() {
    local item_name="$1"
    local subcmd="$2"
    local file="$3"
    shift 3
    local tmp
    tmp=$(mktemp)
    if ! python3 "$SKILL_VALIDATOR" "$subcmd" "$file" "$@" >"$tmp" 2>&1; then
        print_fail "$item_name: validator invocation failed"
        cat "$tmp" >&2
        rm -f "$tmp"
        return 1
    fi

    local had_error=0 err_count=0 warn_count=0
    # Batch-parse all JSONL lines in a single python3 invocation
    local parsed
    parsed=$(_parse_jsonl "$tmp")

    while IFS=$'\t' read -r level rule msg; do
        [ -z "$level" ] && continue
        case "$level" in
            error)
                had_error=1
                err_count=$((err_count + 1))
                print_error "${rule}: ${msg}"
                ;;
            warn)
                warn_count=$((warn_count + 1))
                echo -e "    ${YELLOW}[WARN]${NC} ${item_name}: ${rule}: ${msg}"
                ;;
            *)
                echo "  ${level}: ${rule}: ${msg}"
                ;;
        esac
    done <<< "$parsed"
    rm -f "$tmp"

    if [ $had_error -ne 0 ]; then
        print_fail "${item_name}: ${err_count} error(s), ${warn_count} warning(s)"
        return 1
    fi
    if [ $warn_count -gt 0 ]; then
        print_pass "${item_name}: valid (${warn_count} warning(s))"
    else
        print_pass "${item_name}: valid"
    fi
    return 0
}

# Run the Python validator in batch mode for multiple files.
# Usage: _run_validator_batch <subcmd> <subset> [skill_paths_csv] <file1> <file2> ...
# Returns: 0 if no error-level findings across all files, 1 otherwise.
_run_validator_batch() {
    local subcmd="$1"
    local subset="$2"
    local skill_paths_csv="$3"
    shift 3
    local files=("$@")

    if [ ${#files[@]} -eq 0 ]; then
        return 0
    fi

    local tmp
    tmp=$(mktemp)
    local batch_args=(--batch "--subset=$subset")
    if [ -n "$skill_paths_csv" ]; then
        batch_args+=("--skill-paths=$skill_paths_csv")
    fi

    if ! python3 "$SKILL_VALIDATOR" "$subcmd" "${batch_args[@]}" "${files[@]}" >"$tmp" 2>&1; then
        print_fail "batch validator invocation failed"
        cat "$tmp" >&2
        rm -f "$tmp"
        return 1
    fi

    # Convert JSONL + FILE: markers to TSV in a single Python pass
    local parsed
    parsed=$(python3 - <<'PYEOF' "$tmp"
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("FILE:"):
            print(f"FILE\t-\t{line[5:]}")
            continue
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            print(f"{obj.get('level','')}\t{obj.get('rule','')}\t{obj.get('msg','')}")
        except json.JSONDecodeError:
            pass
PYEOF
)

    local had_error=0
    local current_file=""
    local file_errors=0
    local file_warns=0
    local item_name=""

    while IFS=$'\t' read -r level rule msg; do
        [ -z "$level" ] && continue

        if [ "$level" = "FILE" ]; then
            # Flush previous file results
            if [ -n "$current_file" ]; then
                if [ $file_errors -gt 0 ]; then
                    print_fail "${item_name}: ${file_errors} error(s), ${file_warns} warning(s)"
                elif [ $file_warns -gt 0 ]; then
                    print_pass "${item_name}: valid (${file_warns} warning(s))"
                else
                    print_pass "${item_name}: valid"
                fi
            fi
            current_file="$msg"
            file_errors=0
            file_warns=0
            case "$subcmd" in
                validate-skill) item_name=$(basename "$(dirname "$current_file")") ;;
                validate-agent) item_name=$(basename "$current_file" .md) ;;
                validate-team)  item_name=$(basename "$(dirname "$current_file")") ;;
            esac
            continue
        fi

        case "$level" in
            error)
                had_error=1
                file_errors=$((file_errors + 1))
                print_error "${rule}: ${msg}"
                ;;
            warn)
                file_warns=$((file_warns + 1))
                echo -e "    ${YELLOW}[WARN]${NC} ${item_name}: ${rule}: ${msg}"
                ;;
        esac
    done <<< "$parsed"

    # Flush last file
    if [ -n "$current_file" ]; then
        if [ $file_errors -gt 0 ]; then
            print_fail "${item_name}: ${file_errors} error(s), ${file_warns} warning(s)"
        elif [ $file_warns -gt 0 ]; then
            print_pass "${item_name}: valid (${file_warns} warning(s))"
        else
            print_pass "${item_name}: valid"
        fi
    fi

    rm -f "$tmp"
    [ $had_error -eq 0 ]
}

# Validate skill STRUCTURE + CONTENT in one pass (Python-backed).
# Rules: S-STR-01..16, S-CON-01..09
# Returns: 0 if valid, 1 if errors found
validate_skill_structure() {
    local skill_file="$1"
    local skill_name
    skill_name=$(basename "$(dirname "$skill_file")")
    _run_validator "$skill_name" validate-skill "$skill_file" --subset=structure
}

validate_skill_content() {
    local skill_file="$1"
    local skill_name
    skill_name=$(basename "$(dirname "$skill_file")")
    _run_validator "$skill_name" validate-skill "$skill_file" --subset=content
}

# Batch validate multiple skill files in a single Python invocation.
# Usage: validate_skills_structure_batch file1 file2 ...
# Returns: 0 if no errors, 1 if any errors found.
validate_skills_structure_batch() {
    _run_validator_batch validate-skill structure "" "$@"
}

validate_skills_content_batch() {
    _run_validator_batch validate-skill content "" "$@"
}

# Validate agent STRUCTURE + CONTENT (Python-backed).
# Rules: A-STR-01..07,09,14 + A-CON-01..09
# Optional $2: pre-computed skill_paths (avoids repeated full-repo scans)
validate_agent_structure() {
    local agent_file="$1"
    local agent_name
    agent_name=$(basename "$agent_file" .md)
    local skill_paths="${2:-$(get_all_skills_with_paths | cut -d: -f2-)}"
    local -a skill_arr
    read -ra skill_arr <<< "$skill_paths"
    _run_validator "$agent_name" validate-agent "$agent_file" --subset=structure "${skill_arr[@]}"
}

validate_agent_content() {
    local agent_file="$1"
    local agent_name
    agent_name=$(basename "$agent_file" .md)
    local skill_paths="${2:-$(get_all_skills_with_paths | cut -d: -f2-)}"
    local -a skill_arr
    read -ra skill_arr <<< "$skill_paths"
    _run_validator "$agent_name" validate-agent "$agent_file" --subset=content "${skill_arr[@]}"
}

# Batch validate multiple agent files in a single Python invocation.
# Usage: validate_agents_structure_batch skill_paths_csv file1 file2 ...
# Returns: 0 if no errors, 1 if any errors found.
validate_agents_structure_batch() {
    local skill_paths_csv="$1"
    shift
    _run_validator_batch validate-agent structure "$skill_paths_csv" "$@"
}

validate_agents_content_batch() {
    local skill_paths_csv="$1"
    shift
    _run_validator_batch validate-agent content "$skill_paths_csv" "$@"
}

# Validate team STRUCTURE + CONTENT (Python-backed).
# Rules: T-STR-01..05,07 + T-CON-01..03
validate_team_structure() {
    local team_file="$1"
    local team_name
    team_name=$(basename "$(dirname "$team_file")")
    local team_dir
    team_dir=$(dirname "$team_file")
    local skill_paths
    skill_paths=$(get_all_skills_with_paths | cut -d: -f2-)
    # Optimization: Support local skills bundled in team directory
    local local_skills
    local_skills=$(find "$team_dir" -name "SKILL.md" 2>/dev/null || true)
    local -a all_paths
    read -ra all_paths <<< "$skill_paths $local_skills"
    _run_validator "$team_name" validate-team "$team_file" --subset=structure "${all_paths[@]}"
}

validate_team_content() {
    local team_file="$1"
    local team_name
    team_name=$(basename "$(dirname "$team_file")")
    local team_dir
    team_dir=$(dirname "$team_file")
    local skill_paths
    skill_paths=$(get_all_skills_with_paths | cut -d: -f2-)
    local local_skills
    local_skills=$(find "$team_dir" -name "SKILL.md" 2>/dev/null || true)
    local -a all_paths
    read -ra all_paths <<< "$skill_paths $local_skills"
    _run_validator "$team_name" validate-team "$team_file" --subset=content "${all_paths[@]}"
}

# Batch validate multiple team files in a single Python invocation.
# Usage: validate_teams_structure_batch skill_paths_csv file1 file2 ...
# Returns: 0 if no errors, 1 if any errors found.
validate_teams_structure_batch() {
    local skill_paths_csv="$1"
    shift
    _run_validator_batch validate-team structure "$skill_paths_csv" "$@"
}

validate_teams_content_batch() {
    local skill_paths_csv="$1"
    shift
    _run_validator_batch validate-team content "$skill_paths_csv" "$@"
}

# Cross-file uniqueness check.
# Usage: validate_global_uniqueness <skill|agent|team>
# Returns: 0 if all names unique, 1 if duplicates found.
validate_global_uniqueness() {
    local kind="$1"
    local paths
    case "$kind" in
        skill) paths=$(get_all_skills_with_paths | cut -d: -f2-) ;;
        agent) paths=$(get_all_agents_with_paths | cut -d: -f2-) ;;
        team)  paths=$(get_all_teams_with_paths  | cut -d: -f2-) ;;
        *) print_error "validate_global_uniqueness: unknown kind '$kind'"; return 1 ;;
    esac

    if [ -z "$paths" ]; then
        return 0
    fi

    local tmp
    tmp=$(mktemp)
    local -a path_arr
    read -ra path_arr <<< "$paths"
    python3 "$SKILL_VALIDATOR" check-uniqueness "$kind" "${path_arr[@]}" >"$tmp"

    local had_error=0
    # Batch-parse all JSONL lines in a single python3 invocation
    local parsed
    parsed=$(_parse_jsonl "$tmp")

    while IFS=$'\t' read -r level rule msg; do
        [ -z "$level" ] && continue
        if [ "$level" = "error" ]; then
            had_error=1
            print_error "${rule}: ${msg}"
        fi
    done <<< "$parsed"
    rm -f "$tmp"

    if [ $had_error -ne 0 ]; then
        print_fail "${kind}: uniqueness check failed"
        return 1
    fi
    print_pass "${kind}: uniqueness OK"
    return 0
}

# =============================================================================
# Team Structure & Content Validation Functions
# =============================================================================

# =============================================================================
# Session Analysis
# =============================================================================

# Find the most recent session file
# Usage: session_file=$(find_recent_session [minutes_old])
find_recent_session() {
    local minutes_old="${1:-30}"
    local session_dir="$HOME/.claude/projects"

    # Find the most recent session file
    find "$session_dir" -name "*.jsonl" -type f -mmin -"$minutes_old" 2>/dev/null | sort -r | head -1
}

# Check if a skill was invoked in a session
# Usage: verify_skill_invoked "$session_file" "skill-name"
verify_skill_invoked() {
    local session_file="$1"
    local skill_name="$2"

    if [ ! -f "$session_file" ]; then
        print_fail "Session file not found: $session_file"
        return 1
    fi

    # Look for Skill tool invocation with the skill name
    # Match both "skill":"skillname" and "skill":"namespace:skillname"
    local skill_pattern='"skill":"([^"]*:)?'"${skill_name}"'"'

    if grep -q '"name":"Skill"' "$session_file" && grep -qE "$skill_pattern" "$session_file"; then
        print_pass "Skill '$skill_name' was invoked"
        return 0
    else
        print_fail "Skill '$skill_name' was NOT invoked"
        return 1
    fi
}

# Check if an agent was dispatched in a session
# Usage: verify_agent_dispatched "$session_file" "agent-name"
verify_agent_dispatched() {
    local session_file="$1"
    local agent_name="$2"

    if [ ! -f "$session_file" ]; then
        print_fail "Session file not found: $session_file"
        return 1
    fi

    # Look for Agent tool invocation or Task with agent subagent_type
    if grep -qE '"subagent_type":"'"$agent_name"'"|"name":"'"$agent_name"'"' "$session_file"; then
        print_pass "Agent '$agent_name' was dispatched"
        return 0
    else
        print_fail "Agent '$agent_name' was NOT dispatched"
        return 1
    fi
}

# Count tool invocations in a session
# Usage: count=$(count_tool_invocations "$session_file" "ToolName")
count_tool_invocations() {
    local session_file="$1"
    local tool_name="$2"

    if [ ! -f "$session_file" ]; then
        echo "0"
        return
    fi

    # grep -c prints one count per input file; when grep exits non-zero (no
    # match) under `set -e` the fallback can leak "0\n0". Force-feed via stdin,
    # strip non-digits, and emit a single integer.
    local count
    count=$(grep -c "\"name\":\"$tool_name\"" "$session_file" 2>/dev/null | tr -cd '0-9' || true)
    echo "${count:-0}"
}

# Check for premature action (tools invoked before skill)
# Whitelist: Skill, TodoWrite, TaskOutput (strict — used by tools/analyze-*.sh)
# See also: analyze_premature_actions() which uses a broader whitelist for behavior tests
# Usage: check_premature_action "$session_file" "skill-name"
check_premature_action() {
    local session_file="$1"
    local skill_name="$2"

    if [ ! -f "$session_file" ]; then
        print_skip "Session file not found"
        return 0
    fi

    # Find the line number of the first Skill invocation
    local first_skill_line=$(grep -n '"name":"Skill"' "$session_file" 2>/dev/null | head -1 | cut -d: -f1 || true)

    if [ -z "$first_skill_line" ]; then
        print_warn "No Skill invocation found"
        return 0
    fi

    # Check for tool invocations before the Skill invocation
    local premature_tools=$(head -n "$first_skill_line" "$session_file" 2>/dev/null | \
        grep '"type":"tool_use"' 2>/dev/null | \
        grep -v '"name":"Skill"' 2>/dev/null | \
        grep -v '"name":"TodoWrite"' 2>/dev/null | \
        grep -v '"name":"TaskOutput"' 2>/dev/null || true)

    if [ -n "$premature_tools" ]; then
        print_warn "Tools invoked BEFORE Skill '$skill_name':"
        echo "$premature_tools" | head -3 | sed 's/^/    /' || true
        return 1
    else
        print_pass "No premature tool invocations detected"
        return 0
    fi
}

# Get list of triggered skills from session
# Usage: get_triggered_skills "$session_file"
get_triggered_skills() {
    local session_file="$1"
    if [ ! -f "$session_file" ]; then
        return
    fi
    grep -o '"skill":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/"skill":"//;s/"$//' || true
}

# =============================================================================
# Advanced Session Analysis
# =============================================================================


# Analyze tool call chain in session
# Usage: analyze_tool_chain "$session_file"
# Returns: tool call sequence with timestamps
analyze_tool_chain() {
    local session_file="$1"
    
    if [ ! -f "$session_file" ]; then
        echo "No session file"
        return 1
    fi
    
    echo "=== Tool Call Chain ==="
    
    local line_num=0
    local tool_count=0
    local skill_invoked=false
    local skill_line=0
    
    while IFS= read -r line; do
        ((line_num++)) || true
        
        # Check for Skill invocation
        if echo "$line" | grep -q '"name":"Skill"'; then
            skill_invoked=true
            skill_line=$line_num
            local skill_name=$(echo "$line" | grep -o '"skill":"[^"]*"' | sed 's/"skill":"//;s/"$//' | head -1)
            echo "  [SKILL] Line $line_num: $skill_name"
            ((tool_count++)) || true
        fi
        
        # Check for other tool invocations
        if echo "$line" | grep -q '"type":"tool_use"'; then
            local tool_name=$(echo "$line" | grep -o '"name":"[^"]*"' | head -1 | sed 's/"name":"//;s/"$//')
            if [ -n "$tool_name" ] && [ "$tool_name" != "Skill" ]; then
                local marker="TOOL"
                if $skill_invoked && [ $line_num -lt $skill_line ]; then
                    marker="PREMATURE"
                fi
                echo "  [$marker] Line $line_num: $tool_name"
                ((tool_count++)) || true
            fi
        fi
        
    done < "$session_file"
    
    echo ""
    echo "Total tool calls: $tool_count"
}

# Analyze cost breakdown by subagent (requires jq)
# Usage: analyze_cost_breakdown "$session_file"
analyze_cost_breakdown() {
    local session_file="$1"
    
    if [ ! -f "$session_file" ]; then
        echo "[ERROR] Session file not found"
        return 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo "[ERROR] jq is required for cost analysis"
        return 1
    fi
    
    echo "=== Cost Breakdown Analysis ==="
    echo ""
    
    # Main session usage
    local main_usage=$(jq -s '[.[] | select(.type == "result")] | last | .usage // {}' "$session_file" 2>/dev/null)
    
    if [ -n "$main_usage" ] && [ "$main_usage" != "{}" ]; then
        echo "Main Session:"
        echo "$main_usage" | jq -r '"  Input: \(.input_tokens // 0), Output: \(.output_tokens // 0), Cache Read: \(.cache_read_input_tokens // 0)"' 2>/dev/null || echo "  (unable to parse)"
        echo ""
    fi
    
    # Subagent usage
    local subagent_count=$(jq -s '[.[] | select(.type == "user") | .toolUseResult? | select(. != null) | select(.agentId != null)] | length' "$session_file" 2>/dev/null || echo "0")
    
    if [ "$subagent_count" -gt 0 ]; then
        echo "Subagents ($subagent_count total):"
        jq -s '[.[] | select(.type == "user") | .toolUseResult? | select(. != null) | select(.agentId != null)] | 
            .[] | {agentId, input: .usage.input_tokens // 0, output: .usage.output_tokens // 0, prompt: .prompt[0:50]}' \
            "$session_file" 2>/dev/null | \
            jq -r '"  \(.agentId): Input=\(.input), Output=\(.output) - \(.prompt)..."'
        echo ""
    fi
    
    # Total estimation (approximate pricing)
    local total_input=$(jq -s '[.[] | .usage.input_tokens // 0, .toolUseResult?.usage?.input_tokens // 0] | add' "$session_file" 2>/dev/null || echo "0")
    local total_output=$(jq -s '[.[] | .usage.output_tokens // 0, .toolUseResult?.usage?.output_tokens // 0] | add' "$session_file" 2>/dev/null || echo "0")
    local total_cache=$(jq -s '[.[] | .usage.cache_read_input_tokens // 0] | add' "$session_file" 2>/dev/null || echo "0")
    
    echo "Totals:"
    echo "  Input tokens: $total_input"
    echo "  Output tokens: $total_output"
    echo "  Cache read tokens: $total_cache"
    echo ""
}

# =============================================================================
# Plugin Version Management
# =============================================================================

# Version state directory
VERSION_STATE_DIR="$TESTS_DIR/.version-state"

# Get team plugin.json path
# Usage: get_team_plugin_json "ops-direct-invoke"
# Returns: full path to plugin.json
get_team_plugin_json() {
    local team_name="$1"
    local plugin_path="$SKILLS_DIR/plugins-official/$team_name/.claude-plugin/plugin.json"
    if [ -f "$plugin_path" ]; then
        echo "$plugin_path"
    else
        echo ""
    fi
}

# Extract version from plugin.json
# Usage: extract_plugin_version "/path/to/plugin.json"
# Returns: version string (e.g., "1.0.0")
extract_plugin_version() {
    local plugin_json="$1"
    if [ -f "$plugin_json" ]; then
        grep '"version"' "$plugin_json" | head -1 | sed 's/.*"version":[[:space:]]*"\([^"]*\)".*/\1/'
    fi
}

# Extract array items from plugin.json by field name
# Usage: _extract_plugin_json_array "/path/to/plugin.json" "skills"
_extract_plugin_json_array() {
    local plugin_json="$1"
    local field_name="$2"
    if [ -f "$plugin_json" ]; then
        local in_array=false
        while IFS= read -r line; do
            if echo "$line" | grep -q "\"${field_name}\""; then
                in_array=true
                continue
            fi
            if $in_array; then
                if echo "$line" | grep -q '^\s*\]'; then
                    break
                fi
                local item=$(echo "$line" | sed 's/.*"\(\.[^"]*\)".*/\1/' | tr -d '[:space:]')
                if [ -n "$item" ] && [ "$item" != "$line" ]; then
                    echo "$item"
                fi
            fi
        done < "$plugin_json"
    fi
}

# Extract skills list from plugin.json
# Usage: extract_plugin_skills "/path/to/plugin.json"
# Returns: space-separated list of skill paths (relative)
extract_plugin_skills() {
    _extract_plugin_json_array "$1" "skills"
}

# Extract agents list from plugin.json
# Usage: extract_plugin_agents "/path/to/plugin.json"
# Returns: space-separated list of agent file paths (relative)
extract_plugin_agents() {
    _extract_plugin_json_array "$1" "agents"
}

# Validate SemVer format
# Usage: validate_semver "1.0.0"
# Returns: 0 if valid, 1 if invalid
validate_semver() {
    local version="$1"
    if echo "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
        return 0
    else
        return 1
    fi
}

# Compute SHA256 hash of a file (first 16 chars).
# Used only in local mode — .version-state/ is gitignored, machine-local.
# Usage: compute_file_hash "/path/to/file"
# Returns: short hash string
compute_file_hash() {
    local file="$1"
    if [ -f "$file" ]; then
        if command -v sha256sum &>/dev/null; then
            sha256sum "$file" 2>/dev/null | cut -c1-16 || echo "MISSING"
        elif command -v shasum &>/dev/null; then
            shasum -a 256 "$file" 2>/dev/null | cut -c1-16 || echo "MISSING"
        else
            echo "MISSING"
        fi
    else
        echo "MISSING"
    fi
}

# Recommend version bump based on changes
# Usage: recommend_version_bump "1.0.0" true false
# Args: current_version, skill_changed, agent_changed
# Returns: recommended version string
recommend_version_bump() {
    local current="$1"
    local skill_changed="$2"
    local agent_changed="$3"

    if ! validate_semver "$current"; then
        echo "INVALID"
        return 1
    fi

    local major minor patch
    IFS='.' read -r major minor patch <<< "$current"

    # Any change (skill or agent) → bump PATCH
    if [ "$skill_changed" = "true" ] || [ "$agent_changed" = "true" ]; then
        patch=$((patch + 1))
    fi

    echo "${major}.${minor}.${patch}"
}

# =============================================================================
# Team Version Bump Helpers
# =============================================================================

# Get list of all teams (names only)
# Usage: get_teams_list
# Returns: one team name per line
get_teams_list() {
    for plugin_json in "$SKILLS_DIR"/plugins-official/*/.claude-plugin/plugin.json; do
        [ -f "$plugin_json" ] || continue
        local team_dir
        team_dir=$(dirname "$(dirname "$plugin_json")")
        basename "$team_dir"
    done
}

# Resolve team skill & agent file paths from marketplace.json + plugin.json
# Usage: resolve_team_file_paths "ops-direct-invoke"
# Returns: relative paths (from SKILLS_DIR), one per line
resolve_team_file_paths() {
    local team_name="$1"
    local marketplace_json="$SKILLS_DIR/.claude-plugin/marketplace.json"

    python3 <<PYEOF
import json, os

# Resolve skills from marketplace.json
try:
    marketplace = json.load(open("${marketplace_json}"))
except Exception:
    marketplace = {"plugins": []}

skills_packages = {}
team_entry = None
for p in marketplace.get("plugins", []):
    if p.get("category") == "skills":
        skills_packages[p["name"]] = p
    elif p.get("name") == "${team_name}":
        team_entry = p

if team_entry:
    for dep_name in team_entry.get("dependencies", []):
        sp = skills_packages.get(dep_name)
        if sp:
            source = sp.get("source", "./ops").lstrip("./")
            for skill_rel in sp.get("skills", []):
                skill_dir = skill_rel.lstrip("./")
                print(f"{source}/{skill_dir}/SKILL.md")

# Resolve agents from plugin.json
try:
    plugin_json_path = os.path.join(
        "${SKILLS_DIR}", "plugins-official", "${team_name}",
        ".claude-plugin", "plugin.json")
    plugin = json.load(open(plugin_json_path))
    for agent_rel in plugin.get("agents", []):
        agent_rel = agent_rel.lstrip("./")
        print(f"plugins-official/${team_name}/{agent_rel}")
except Exception:
    pass
PYEOF
}

# Detect whether a team's dependency files changed against a base ref
# Usage: detect_team_changes "ops-direct-invoke" "origin/master"
# Returns: "changed" if any dependency file changed, empty otherwise
detect_team_changes() {
    local team_name="$1"
    local base_ref="$2"

    local changed_files
    changed_files=$(git -C "$SKILLS_DIR" diff --name-only "${base_ref}...HEAD" 2>/dev/null || true)
    [ -z "$changed_files" ] && return

    local team_files
    team_files=$(resolve_team_file_paths "$team_name")
    [ -z "$team_files" ] && return

    while IFS= read -r rel; do
        [ -z "$rel" ] && continue
        if echo "$changed_files" | grep -qF "$rel"; then
            echo "changed"
            return
        fi
    done <<< "$team_files"
}

# Bump version in a team's plugin.json
# Usage: bump_plugin_json "ops-direct-invoke" "1.1.11"
# Returns: 0 on success, 1 on failure
bump_plugin_json() {
    local team_name="$1"
    local new_version="$2"
    local plugin_json
    plugin_json=$(get_team_plugin_json "$team_name")

    if [ -z "$plugin_json" ] || [ ! -f "$plugin_json" ]; then
        echo "[WARN] plugin.json not found for $team_name" >&2
        return 1
    fi

    python3 <<PYEOF
import json, sys
path = "${plugin_json}"
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    old = data.get("version", "")
    data["version"] = "${new_version}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[BUMP] ${team_name}: plugin.json {old} -> ${new_version}")
except Exception as e:
    print(f"[ERROR] Failed to bump plugin.json for ${team_name}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# Bump version in marketplace.json for a given team
# Usage: bump_marketplace_json "ops-direct-invoke" "1.1.11"
# Returns: 0 on success, 1 on failure
bump_marketplace_json() {
    local team_name="$1"
    local new_version="$2"
    local marketplace_json="$SKILLS_DIR/.claude-plugin/marketplace.json"

    if [ ! -f "$marketplace_json" ]; then
        echo "[WARN] marketplace.json not found" >&2
        return 1
    fi

    python3 <<PYEOF
import json, sys
path = "${marketplace_json}"
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    found = False
    for p in data.get("plugins", []):
        if p.get("name") == "${team_name}":
            old = p.get("version", "")
            p["version"] = "${new_version}"
            found = True
            print(f"[BUMP] ${team_name}: marketplace.json {old} -> ${new_version}")
            break
    if not found:
        print(f"[WARN] ${team_name}: not found in marketplace.json", file=sys.stderr)
        sys.exit(1)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
except Exception as e:
    print(f"[ERROR] Failed to bump marketplace.json for ${team_name}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# Compare two SemVer versions
# Usage: semver_compare "1.0.0" "1.0.1"
# Returns: -1 if first < second, 0 if equal, 1 if first > second
semver_compare() {
    local v1="$1"
    local v2="$2"

    local major1 minor1 patch1 major2 minor2 patch2
    IFS='.' read -r major1 minor1 patch1 <<< "$v1"
    IFS='.' read -r major2 minor2 patch2 <<< "$v2"

    if [ "$major1" -gt "$major2" ] 2>/dev/null; then echo "1"; return; fi
    if [ "$major1" -lt "$major2" ] 2>/dev/null; then echo "-1"; return; fi
    if [ "$minor1" -gt "$minor2" ] 2>/dev/null; then echo "1"; return; fi
    if [ "$minor1" -lt "$minor2" ] 2>/dev/null; then echo "-1"; return; fi
    if [ "$patch1" -gt "$patch2" ] 2>/dev/null; then echo "1"; return; fi
    if [ "$patch1" -lt "$patch2" ] 2>/dev/null; then echo "-1"; return; fi
    echo "0"
}

# =============================================================================
# Token Usage Analysis
# =============================================================================

# Extract token usage from session file (requires jq)
# Usage: extract_token_usage "$session_file"
extract_token_usage() {
    local session_file="$1"
    if [ ! -f "$session_file" ]; then
        echo "{}"
        return 1
    fi

    if command -v jq &> /dev/null; then
        # Extract usage from the last result message
        jq -s '[.[] | select(.type == "result")] | last | .usage // {}' "$session_file" 2>/dev/null || echo "{}"
    else
        echo "{}"
    fi
}

# =============================================================================
# Test Result Tracking
# =============================================================================
# These functions provide basic test tracking for individual test scripts.
# Note: run-tests.sh uses its own TEST_RESULTS array and print_summary/output_json
# functions for more detailed tracking. Individual test scripts should use these
# functions or implement their own tracking as needed.
# =============================================================================

# Initialize shared test cache (avoids repeated find/python3 calls across test scripts)
# Usage: init_test_cache
init_test_cache() {
    if [ -z "${_TEST_CACHE_DIR:-}" ]; then
        export _TEST_CACHE_DIR=$(mktemp -d -t cann-test-cache.XXXXXX)
    fi
}

# Cleanup shared test cache
# Usage: cleanup_test_cache
cleanup_test_cache() {
    if [ -n "${_TEST_CACHE_DIR:-}" ] && [ -d "${_TEST_CACHE_DIR}" ]; then
        rm -rf "$_TEST_CACHE_DIR"
        unset _TEST_CACHE_DIR
    fi
}

# Initialize test tracking
# Usage: init_test_tracking
init_test_tracking() {
    TEST_PASSED=0
    TEST_FAILED=0
    TEST_SKIPPED=0
    TEST_START_TIME=$(date +%s)
}

# Record a test result
# Usage: record_test "pass" "test_name" ["duration"]
record_test() {
    local result="$1"
    local test_name="$2"
    local duration="${3:-0}"

    case "$result" in
        pass|PASS)  ((TEST_PASSED++)) || true ;;
        fail|FAIL)  ((TEST_FAILED++)) || true ;;
        skip|SKIP)  ((TEST_SKIPPED++)) || true ;;
    esac
}

# Print test summary
# Usage: print_test_summary
print_test_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - TEST_START_TIME))

    echo ""
    echo "========================================"
    echo -e " ${BOLD}Test Results Summary${NC}"
    echo "========================================"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}  $TEST_PASSED"
    echo -e "  ${RED}Failed:${NC}  $TEST_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $TEST_SKIPPED"
    echo "  Duration: ${duration}s"
    echo ""

    if [ "$TEST_FAILED" -gt 0 ]; then
        print_status_failed
        return 1
    else
        print_status_passed
        return 0
    fi
}

# Output test results as JSON (for CI)
# Usage: output_test_json > results.json
output_test_json() {
    local end_time=$(date +%s)
    local duration=$((end_time - TEST_START_TIME))

    cat <<EOF
{
  "status": "$([ "$TEST_FAILED" -gt 0 ] && echo "failed" || echo "passed")",
  "passed": $TEST_PASSED,
  "failed": $TEST_FAILED,
  "skipped": $TEST_SKIPPED,
  "duration": $duration,
  "timestamp": "$(date -Iseconds)"
}
EOF
}

# =============================================================================
# Utility Functions
# =============================================================================

# Print a test section header
# Usage: print_section "Section Name"
print_section() {
    local name="$1"
    echo ""
    echo "----------------------------------------"
    echo " $name"
    echo "----------------------------------------"
}

# Print test start banner
# Usage: print_test_banner "Test Name" ["description"]
print_test_banner() {
    local name="$1"
    local description="${2:-}"

    echo ""
    echo "========================================"
    echo " $name"
    echo "========================================"
    [ -n "$description" ] && echo "" && echo "$description"
    echo ""
}

# =============================================================================
# Export Functions
# =============================================================================

export -f is_platform_available
export -f get_platform_version
export -f create_test_project
export -f cleanup_test_project
export -f run_claude
export -f assert_order
export -f assert_file_exists
export -f check_file_links
export -f run_behavior_test
export -f analyze_premature_actions
export -f get_all_skills
export -f get_all_skills_with_paths
export -f find_skill_file
export -f get_all_agents
export -f get_all_agents_with_paths
export -f find_agent_file
export -f get_all_teams
export -f get_all_teams_with_paths
export -f find_team_file
export -f is_incremental_mode
export -f should_test_skill
export -f should_test_agent
export -f should_test_team
export -f get_skills_to_test
export -f get_agents_to_test
export -f get_teams_to_test
export -f validate_skill_structure
export -f validate_skill_content
export -f validate_agent_structure
export -f validate_agent_content
export -f validate_team_structure
export -f validate_team_content
export -f validate_global_uniqueness
export -f validate_skills_structure_batch
export -f validate_skills_content_batch
export -f validate_agents_structure_batch
export -f validate_agents_content_batch
export -f validate_teams_structure_batch
export -f validate_teams_content_batch
export -f find_recent_session
export -f verify_skill_invoked
export -f verify_agent_dispatched
export -f count_tool_invocations
export -f check_premature_action
export -f get_triggered_skills
export -f analyze_tool_chain
export -f analyze_cost_breakdown
export -f extract_token_usage
export -f init_test_cache
export -f cleanup_test_cache
export -f init_test_tracking
export -f record_test
export -f print_test_summary
export -f output_test_json
export -f print_section
export -f print_test_banner
export -f setup_colors
export -f enable_colors
export -f disable_colors
export -f print_compact_result
export -f print_pass
export -f print_fail
export -f print_skip
export -f print_info
export -f print_warn
export -f print_error
export -f print_section_header
export -f print_status_passed
export -f print_status_failed
export -f get_team_plugin_json
export -f extract_plugin_version
export -f extract_plugin_skills
export -f extract_plugin_agents
export -f validate_semver
export -f compute_file_hash
export -f recommend_version_bump
export -f semver_compare
export LIB_DIR TESTS_DIR SKILLS_DIR

# Force enable colors when sourced (unless NO_COLOR is set)
if [ -z "${NO_COLOR:-}" ]; then
    FORCE_COLOR=1
    setup_colors
fi