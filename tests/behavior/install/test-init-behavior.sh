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
# Behavior Test: Init Script Installation (L2)
# =============================================================================
# Executes init.sh in isolated environments and verifies the installed
# artifacts match expectations for all 4 level×tool combinations.
#
# Isolation strategy:
#   - global mode:  override $HOME to a temp directory
#   - project mode:  run from a temp working directory
#   - git repos:     create fake local directories to avoid network clones
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Populated per-team in main()
TEAM=""
TEAM_DIR=""
INIT_SCRIPT=""

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Track fake repos (git-clone targets) created during tests
FAKE_REPOS_CREATED=""

# =============================================================================
# Helpers
# =============================================================================

# Detect repo names that init.sh symlinks into CONFIG_ROOT in global mode.
# These are typically git-clone targets (asc-devkit, pypto, etc.).
get_git_repo_names() {
    local init="$1"
    grep -E 'ln -sfn.*"\$?CONFIG_ROOT/' "$init" 2>/dev/null | sed -E 's/.*CONFIG_ROOT\/([^"[:space:]]+).*/\1/' | grep -vxE 'AGENTS.md|workflows|agents|skills' | grep -v '^skills/' | sort -u
}

# Create fake repo directories to avoid network clones during tests
setup_fake_repos() {
    local team_dir="$1"
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        local repo_dir="$team_dir/$repo"
        if [ ! -d "$repo_dir" ]; then
            mkdir -p "$repo_dir/docs" "$repo_dir/examples"
            touch "$repo_dir/docs/README.md"
            touch "$repo_dir/examples/example.py"
            touch "$repo_dir/.test-fake-repo"
            # Initialize as a real git repo so that git commands in init.sh
            # (e.g. `git checkout .`, `git checkout master`) operate on this
            # fake repo instead of falling through to the parent cann_skills
            # repository (which would switch branches / discard changes).
            git init --quiet "$repo_dir"
            git -C "$repo_dir" add -A
            git -C "$repo_dir" -c user.name="test" -c user.email="test@test.com" \
                commit --quiet -m "test fixture"
            FAKE_REPOS_CREATED="$FAKE_REPOS_CREATED $repo"
        fi
    done < <(get_git_repo_names "$team_dir/init.sh")
}

cleanup_fake_repos() {
    for repo in $FAKE_REPOS_CREATED; do
        # Only delete directories we created (marked with .test-fake-repo).
        # A real git clone will not have this marker.
        if [ -f "$TEAM_DIR/$repo/.test-fake-repo" ]; then
            rm -rf "$TEAM_DIR/$repo"
        fi
    done
    FAKE_REPOS_CREATED=""
}

# Remove artifacts created by init.sh inside the given temp dir; never touch $TEAM_DIR (committed config).
cleanup_team_artifacts() {
    local dir="${1:-}"
    if [ -z "$dir" ] || [ "$dir" = "$TEAM_DIR" ]; then
        return 0
    fi
    rm -rf "$dir/.opencode" "$dir/.claude" "$dir/.trae" "$dir/.marscode" "$dir/.traecli" "$dir/.codeartsdoer" "$dir/.cursor" "$dir/.github" "$dir/.copilot"
}

get_expected_skill_count() {
    local skills
    skills=$(grep 'INCLUDED_SKILLS="' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*INCLUDED_SKILLS="//;s/"$//' || true)
    if [ -z "$skills" ]; then
        echo "-1"
    else
        echo "$skills" | wc -w
    fi
}

get_expected_agent_count() {
    local pattern
    pattern=$(grep 'INCLUDED_AGENT_PATTERN=' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*="//;s/"$//' || true)
    if [ -z "$pattern" ]; then
        if [ -d "$TEAM_DIR/agents" ]; then
            echo "-1"
        else
            echo "0"
        fi
        return
    fi
    local count=0
    for f in "$TEAM_DIR/agents/"*.md; do
        [ -f "$f" ] || continue
        local base
        base=$(basename "$f" .md)
        if [[ "$base" == $pattern ]]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# Detect whether init.sh explicitly rejects global + trae (design choice,
# not a bug). Such teams exit 1 instead of installing; the issue #355
# regression guard for them is "must not silently fall through to ~/.claude".
trae_global_unsupported() {
    grep -q 'Global installation is not supported for Trae' "$INIT_SCRIPT" 2>/dev/null
}

# Detect whether init.sh supports a given TOOL argument.
# Some plugins (e.g. cuda2ascend) only support opencode and reject other tools.
supports_tool() {
    local tool="$1"
    grep -qE "(^|[^a-z])${tool}([^a-z]|$)" "$INIT_SCRIPT" 2>/dev/null
}

# Resolve the actual CONFIG_ROOT for project mode.
# Most plugins use $PWD/.xxx, but some (e.g. tilelang2ascendc-ops-generator)
# use $PLUGIN_ROOT/.xxx which means CONFIG_ROOT is under the plugin directory.
# Returns the config_root path via echo.
resolve_project_config_root() {
    local tool="$1"
    local pwd_dir="$2"
    local tool_dir_name

    case "$tool" in
        opencode) tool_dir_name=".opencode" ;;
        claude)   tool_dir_name=".claude" ;;
        trae)     tool_dir_name=".trae" ;;
        codearts) tool_dir_name=".codeartsdoer" ;;
        cursor)   tool_dir_name=".cursor" ;;
        copilot)  tool_dir_name=".github" ;;
        *)        tool_dir_name=".$tool" ;;
    esac

    # Check if init.sh uses PLUGIN_ROOT for project CONFIG_ROOT
    if grep -q "CONFIG_ROOT=\"\$PLUGIN_ROOT/${tool_dir_name}" "$INIT_SCRIPT" 2>/dev/null; then
        echo "$TEAM_DIR/$tool_dir_name"
    else
        echo "$pwd_dir/$tool_dir_name"
    fi
}

# Verify repos (asc-devkit, etc.) are symlinked into CONFIG_ROOT in global mode.
# Mirrors the inline repo-symlink assertions in scenario_global_opencode/claude.
verify_global_repo_symlinks() {
    local config_root="$1"
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ -L "$config_root/$repo" ]; then
            print_pass "$repo symlinked into CONFIG_ROOT (global mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo NOT symlinked into CONFIG_ROOT (global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")
}

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

# Negative assertion: init.sh output must not contain bash parse diagnostics.
# Parse-time bash warnings (e.g. "unterminated here-document" from a heredoc
# closed on the same line as its $( opening) only warn on new bash but are
# hard syntax errors on older bash, so they must fail the test on any version
# (see https://gitcode.com/cann/cannbot-skills/issues/608).
assert_no_shell_warnings() {
    local output="$1"
    local hits
    hits=$(echo "$output" | grep -E "warning: (command substitution|here-document)|syntax error" || true)
    if [ -n "$hits" ]; then
        print_fail "init.sh emitted bash parse diagnostics"
        echo "$hits" | sed 's/^/        /'
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "init.sh output free of bash parse diagnostics"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

# Verify every symlink under a directory points to an existing target
verify_symlinks_valid() {
    local dir="$1"
    local label="$2"
    local broken=0
    if [ -d "$dir" ]; then
        for link in "$dir"/*; do
            [ -e "$link" ] || [ -L "$link" ] || continue
            if [ -L "$link" ]; then
                local target
                target=$(readlink "$link")
                if [ ! -e "$target" ]; then
                    print_fail "$label: broken symlink '$link' → '$target'"
                    FAIL_COUNT=$((FAIL_COUNT + 1))
                    broken=$((broken + 1))
                fi
            fi
        done
    fi
    if [ "$broken" -eq 0 ]; then
        print_pass "$label: all symlinks resolve to existing targets"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

# Convert a POSIX path to a Windows-native path when running under MSYS2
# with a Windows-native python3. Returns the original path on Linux/macOS.
py_path() {
    local p="$1"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$p" 2>/dev/null || echo "$p"
    else
        echo "$p"
    fi
}

# Verify manifest JSON structure and consistency
verify_manifest() {
    local config_root="$1"
    local expected_skills="$2"
    local expected_agents="$3"
    local tool="${4:-}"
    local manifest="$config_root/cannbot-manifest.json"
    local py_manifest
    py_manifest=$(py_path "$manifest")

    if [ ! -f "$manifest" ]; then
        print_fail "manifest not found: $manifest"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Validate JSON
    if python3 -c "import json; json.load(open(r'''$py_manifest'''))" 2>/dev/null; then
        print_pass "manifest is valid JSON"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "manifest is invalid JSON"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Check skill count
    local actual_skills
    actual_skills=$(python3 -c "import json; print(len(json.load(open(r'''$py_manifest''')).get('installed_skills', [])))" 2>/dev/null || echo 0)
    if [ "$expected_skills" -eq -1 ]; then
        print_pass "manifest: installed_skills count = $actual_skills (dynamic, no INCLUDED_SKILLS)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ "$actual_skills" -eq "$expected_skills" ]; then
        print_pass "manifest: installed_skills count = $actual_skills (expected $expected_skills)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "manifest: installed_skills count = $actual_skills (expected $expected_skills)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Check agent count
    local actual_agents
    actual_agents=$(python3 -c "import json; print(len(json.load(open(r'''$py_manifest''')).get('installed_agents', [])))" 2>/dev/null || echo 0)
    # Some plugins install an extra primary agent (PRIMARY_AGENT_NAME) in
    # opencode mode, so the actual count may be expected + 1.
    local agent_tolerance=0
    if [ "$tool" = "opencode" ] && grep -q 'PRIMARY_AGENT_NAME=' "$INIT_SCRIPT" 2>/dev/null; then
        agent_tolerance=1
    fi
    if [ "$expected_agents" -eq -1 ]; then
        print_pass "manifest: installed_agents count = $actual_agents (dynamic, no INCLUDED_AGENT_PATTERN)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ "$actual_agents" -eq "$expected_agents" ] || \
         [ "$actual_agents" -eq "$((expected_agents + agent_tolerance))" ]; then
        print_pass "manifest: installed_agents count = $actual_agents (expected $expected_agents, tolerance +$agent_tolerance)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "manifest: installed_agents count = $actual_agents (expected $expected_agents, tolerance +$agent_tolerance)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Check brand/level/tool fields
    for field in brand level tool; do
        local val
        val=$(python3 -c "import json; print(json.load(open(r'''$py_manifest''')).get('$field',''))" 2>/dev/null || true)
        if [ -n "$val" ]; then
            print_pass "manifest: '$field' = '$val'"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "manifest: missing '$field' field"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
}

# Verify that installed skill/agent names match whitelist expectations
verify_installed_names() {
    local dir="$1"
    local label="$2"
    local expected_list="$3"

    if [ ! -d "$dir" ]; then
        print_fail "$label: directory not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    if [ -z "$expected_list" ]; then
        print_pass "$label: no whitelist to verify (dynamic skill list)"
        PASS_COUNT=$((PASS_COUNT + 1))
        return 0
    fi

    local mismatch=0
    for item in "$dir"/*; do
        [ -e "$item" ] || [ -L "$item" ] || continue
        local name
        name=$(basename "$item")
        if ! echo "$expected_list" | grep -qw "$name"; then
            print_fail "$label: unexpected item '$name' installed"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            mismatch=$((mismatch + 1))
        fi
    done

    if [ "$mismatch" -eq 0 ]; then
        print_pass "$label: no unexpected items installed"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

# =============================================================================
# Scenario helpers
# =============================================================================

# Expected counts are set per-team in main() before each scenario runs.

check_common_artifacts() {
    local config_root="$1"
    local tool="$2"

    # 1. Config root directory exists
    run_check "CONFIG_ROOT exists: $config_root" test -d "$config_root"
    if [ ! -d "$config_root" ]; then
        print_warn "CONFIG_ROOT missing, skipping remaining artifact checks"
        return 0
    fi

    # 2. skills/ directory exists with expected count
    local skill_dir="$config_root/skills"
    if [ -d "$skill_dir" ]; then
        local actual_skills
        actual_skills=$(find "$skill_dir" -maxdepth 1 \( -type l -o -type d \) | wc -l)
        # subtract 1 for the directory itself
        actual_skills=$((actual_skills - 1))
        if [ "$EXPECTED_SKILL_COUNT" -eq -1 ]; then
            print_pass "skills/ contains $actual_skills item(s) (dynamic count, no INCLUDED_SKILLS)"
            PASS_COUNT=$((PASS_COUNT + 1))
        elif [ "$actual_skills" -eq "$EXPECTED_SKILL_COUNT" ]; then
            print_pass "skills/ contains $actual_skills item(s) (expected $EXPECTED_SKILL_COUNT)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "skills/ contains $actual_skills item(s) (expected $EXPECTED_SKILL_COUNT)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        print_fail "skills/ directory not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # 3. agents/ directory exists with expected count
    local agent_dir="$config_root/agents"
    # Some plugins install an extra primary agent (PRIMARY_AGENT_NAME) in
    # opencode mode, so the actual count may be expected + 1.
    local agent_tolerance=0
    if [ "$tool" = "opencode" ] && grep -q 'PRIMARY_AGENT_NAME=' "$INIT_SCRIPT" 2>/dev/null; then
        agent_tolerance=1
    fi
    if [ "$EXPECTED_AGENT_COUNT" -eq -1 ]; then
        if [ -d "$agent_dir" ]; then
            local actual_agents
            actual_agents=$(find "$agent_dir" -maxdepth 1 -mindepth 1 | wc -l)
            print_pass "agents/ contains $actual_agents item(s) (dynamic count, no INCLUDED_AGENT_PATTERN)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "agents/ directory not found"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    elif [ "$EXPECTED_AGENT_COUNT" -eq 0 ]; then
        print_pass "agents/ not installed (agentless plugin, expected 0)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ -d "$agent_dir" ]; then
        local actual_agents
        actual_agents=$(find "$agent_dir" -maxdepth 1 -mindepth 1 | wc -l)
        if [ "$actual_agents" -eq "$EXPECTED_AGENT_COUNT" ] || \
           [ "$actual_agents" -eq "$((EXPECTED_AGENT_COUNT + agent_tolerance))" ]; then
            print_pass "agents/ contains $actual_agents item(s) (expected $EXPECTED_AGENT_COUNT, tolerance +$agent_tolerance)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "agents/ contains $actual_agents item(s) (expected $EXPECTED_AGENT_COUNT, tolerance +$agent_tolerance)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        print_fail "agents/ directory not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # 4. All symlinks resolve to existing targets
    verify_symlinks_valid "$skill_dir" "skills/"
    verify_symlinks_valid "$agent_dir" "agents/"

    # 5. No whitelist-violating items installed
    verify_installed_names "$skill_dir" "skills/" "$EXPECTED_SKILLS_LIST"

    # 6. Workflows symlink exists (if team has workflows/)
    if [ -d "$TEAM_DIR/workflows" ]; then
        local wf_link="$config_root/workflows"
        if [ -L "$wf_link" ]; then
            local wf_target
            wf_target=$(readlink "$wf_link")
            if [ -d "$wf_target" ]; then
                print_pass "workflows symlink resolves to existing directory"
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                print_fail "workflows symlink target does not exist: $wf_target"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            print_fail "workflows symlink not found"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi

    # 7. Manifest is correct
    verify_manifest "$config_root" "$EXPECTED_SKILL_COUNT" "$EXPECTED_AGENT_COUNT" "$tool"
}

# =============================================================================
# Verify via opencode CLI that installed agents are recognized
# =============================================================================
verify_opencode_cli_agents() {
    local scan_dir="${1:-$TEAM_DIR}"
    if ! command -v opencode &>/dev/null; then
        print_skip "opencode CLI not available, skipping CLI agent recognition check"
        return 0
    fi

    local pattern
    pattern=$(grep 'INCLUDED_AGENT_PATTERN=' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*="//;s/"$//' || true)
    local expected=()
    for f in "$TEAM_DIR/agents/"*.md; do
        [ -f "$f" ] || continue
        local base
        base=$(basename "$f" .md)
        if [[ "$base" == $pattern ]]; then
            expected+=("$base")
        fi
    done

    if [ ${#expected[@]} -eq 0 ]; then
        print_info "No agents match pattern for CLI check"
        return 0
    fi

    # Pre-check: verify agent symlinks exist and resolve
    local agent_dir="$scan_dir/.opencode/agents"
    if [ ! -d "$agent_dir" ]; then
        # Some plugins use PLUGIN_ROOT instead of PWD for CONFIG_ROOT
        agent_dir="$TEAM_DIR/.opencode/agents"
    fi
    if [ ! -d "$agent_dir" ]; then
        print_fail "Agent directory not found: $agent_dir"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
    for agent in "${expected[@]}"; do
        local link="$agent_dir/$agent.md"
        if [ ! -e "$link" ]; then
            print_fail "Agent symlink missing: $agent.md"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        elif [ ! -f "$link" ]; then
            print_fail "Agent symlink broken (target not found): $agent.md → $(readlink "$link")"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done

    local output
    # opencode agent list scans the current directory's .opencode/ config
    output=$(cd "$scan_dir" && opencode agent list 2>&1 || true)

    # If not all agents recognized on first try, wait and retry once
    local all_recognized=true
    for agent in "${expected[@]}"; do
        if ! echo "$output" | grep -q "^$agent "; then
            all_recognized=false
            break
        fi
    done
    if ! $all_recognized; then
        sleep 2
        output=$(cd "$scan_dir" && opencode agent list 2>&1 || true)
    fi

    for agent in "${expected[@]}"; do
        local match_count
        match_count=$(echo "$output" | grep -cE "^[[:space:]]*${agent} " || echo 0)
        if [ "$match_count" -gt 0 ]; then
            print_pass "opencode CLI recognizes agent: $agent"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "opencode CLI does NOT recognize agent: $agent"
            echo "    Diagnostic: searching for '$agent' in opencode agent list output:"
            echo "    Total output lines: $(echo "$output" | wc -l)"
            echo "    Lines containing '$agent':"
            echo "$output" | grep "$agent" | head -5 | sed 's/^/      /' || echo "      (none found)"
            echo "    Hex dump of matching lines:"
            echo "$output" | grep "$agent" | head -2 | od -c | head -10 | sed 's/^/      /' || true
            echo "    Grep test (count=$match_count):"
            echo "$output" | grep -E "^[[:space:]]*${agent} " | head -2 | sed 's/^/      /' || echo "      (no match with pattern)"
            echo "    Symlink status:"
            ls -la "$agent_dir/$agent.md" 2>&1 | sed 's/^/      /' || true
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
}

# =============================================================================
# Simulate OpenCode discovery scan
# =============================================================================
verify_opencode_discovery() {
    local config_root="$1"
    local found=0
    local fail=0

    # OpenCode scans agents/<agent>.md
    local agent_dir="$config_root/agents"
    found=0; fail=0
    for link in "$agent_dir"/*; do
        [ -e "$link" ] || [ -L "$link" ] || continue
        local name
        name=$(basename "$link")
        if [ -f "$link" ]; then
            found=$((found + 1))
        else
            print_fail "OpenCode scan: agent '$name' → not a readable file"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -eq 0 ]; then
        print_pass "OpenCode scan: all $found agents are readable .md files"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # OpenCode reads AGENTS.md from project root (upward traversal)
    # No need to check CONFIG_ROOT/AGENTS.md since opencode does not use rules/
}

# =============================================================================
# Simulate Claude Code discovery scan
# =============================================================================
verify_claude_discovery() {
    local config_root="$1"
    local level="$2"
    local tmp_pwd="$3"
    local found=0
    local fail=0

    # Claude scans agents/<agent>.md
    local agent_dir="$config_root/agents"
    found=0; fail=0
    for link in "$agent_dir"/*; do
        [ -e "$link" ] || [ -L "$link" ] || continue
        local name
        name=$(basename "$link")
        if [ -f "$link" ]; then
            found=$((found + 1))
        else
            print_fail "Claude scan: agent '$name' → not a readable file"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -eq 0 ]; then
        print_pass "Claude scan: all $found agents are readable .md files"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # Claude reads CLAUDE.md in project root or CONFIG_ROOT
    local claude_md
    if [ "$level" = "project" ]; then
        claude_md="$tmp_pwd/CLAUDE.md"
        if [ ! -f "$claude_md" ]; then
            claude_md="$config_root/CLAUDE.md"
        fi
    else
        claude_md="$config_root/CLAUDE.md"
    fi
    if [ -f "$claude_md" ] && [ -s "$claude_md" ]; then
        print_pass "Claude scan: CLAUDE.md is present and non-empty"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "Claude scan: CLAUDE.md missing or empty"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# =============================================================================
# Scenario 1: Project + OpenCode
# =============================================================================
scenario_project_opencode() {
    print_section_header "Scenario: project + opencode"

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    # Cleanup on exit or error
    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project opencode <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$(resolve_project_config_root "opencode" "$tmp_pwd")"
    check_common_artifacts "$config_root" "opencode"

    # Project OpenCode specific: AGENTS.md in PWD or CONFIG_ROOT
    # When install_path differs from plugin dir, AGENTS.md is a copy with absolute paths
    if [ -e "$tmp_pwd/AGENTS.md" ] || [ -e "$config_root/AGENTS.md" ]; then
        print_pass "AGENTS.md exists (PWD or CONFIG_ROOT)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "AGENTS.md is missing (not in PWD or CONFIG_ROOT)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    local _agents_md="$tmp_pwd/AGENTS.md"
    [ -f "$_agents_md" ] || _agents_md="$config_root/AGENTS.md"
    if [ -f "$_agents_md" ] && [ ! -L "$_agents_md" ] && grep -q "$TEAM_DIR" "$_agents_md" 2>/dev/null; then
        print_pass "AGENTS.md contains absolute paths (project mode rewrite)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ -L "$_agents_md" ]; then
        print_pass "AGENTS.md is a symlink (plugin dir = PWD)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # Repos should NOT be symlinked into CONFIG_ROOT in project mode
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ ! -e "$config_root/$repo" ]; then
            print_pass "$repo not symlinked into CONFIG_ROOT (project mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo unexpectedly present in CONFIG_ROOT (project mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")

    # Verify OpenCode can discover installed content
    verify_opencode_discovery "$config_root"

    # Verify via opencode CLI (project mode only; global mode agents live in
    # ~/.config/opencode which opencode agent list does not scan from a temp dir)
    verify_opencode_cli_agents "$tmp_pwd"

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 2: Global + OpenCode
# =============================================================================
scenario_global_opencode() {
    print_section_header "Scenario: global + opencode"

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global opencode <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.config/opencode"
    check_common_artifacts "$config_root" "opencode"

    # Global OpenCode specific: AGENTS.md is a copy with absolute paths
    # (only if init.sh performs sed rewrite)
    if grep -q 'ESCAPED_ROOT' "$INIT_SCRIPT" 2>/dev/null; then
        local config_file="$config_root/AGENTS.md"
        if [ -f "$config_file" ] && [ ! -L "$config_file" ]; then
            print_pass "AGENTS.md is a regular file (global mode copy, not symlink)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "AGENTS.md is missing or is a symlink (expected regular file in global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        if grep -q "$TEAM_DIR" "$config_file" 2>/dev/null; then
            print_pass "AGENTS.md contains absolute paths (global mode rewrite detected)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_warn "AGENTS.md does not contain absolute paths (may be OK if no relative refs)"
            WARN_COUNT=$((WARN_COUNT + 1))
        fi
    fi

    # Repos SHOULD be symlinked into CONFIG_ROOT in global mode
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ -L "$config_root/$repo" ]; then
            print_pass "$repo symlinked into CONFIG_ROOT (global mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo NOT symlinked into CONFIG_ROOT (global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")

    # Verify OpenCode can discover installed content
    verify_opencode_discovery "$config_root"

    # -------------------------------------------------------------------------
    # Safe-install assertions (idempotency + backup)
    # -------------------------------------------------------------------------

    # Idempotency: re-run init.sh with existing config → no backup created
    local idempotent_output
    local idempotent_exit=0
    idempotent_output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global opencode <<< "y" 2>&1) || idempotent_exit=$?
    if [ "$idempotent_exit" -eq 0 ]; then
        print_pass "Idempotent re-run exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "Idempotent re-run exited with code $idempotent_exit"
        echo "$idempotent_output" | tail -10
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local bak_count
    bak_count=$(find "$config_root" -maxdepth 1 -name 'AGENTS.md.bak.*' 2>/dev/null | wc -l)
    if [ "$bak_count" -eq 0 ]; then
        print_pass "Idempotent re-run: no AGENTS.md backup created"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "Idempotent re-run: unexpected AGENTS.md backup found ($bak_count)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Backup trigger: pre-write custom content → backup created on re-run
    # Only test teams that have safe_install_file AND use it for AGENTS.md
    if [ -f "$config_root/AGENTS.md" ] && grep -q 'safe_install_file.*AGENTS\.md\|safe_install_file.*config_target\|safe_install_file.*config_name' "$INIT_SCRIPT" 2>/dev/null; then
        echo "# User custom header" > "$config_root/AGENTS.md"
        echo "custom user content" >> "$config_root/AGENTS.md"

        local backup_output
        local backup_exit=0
        backup_output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global opencode <<< "y" 2>&1) || backup_exit=$?
        if [ "$backup_exit" -eq 0 ]; then
            print_pass "Backup-trigger re-run exited with code 0"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "Backup-trigger re-run exited with code $backup_exit"
            echo "$backup_output" | tail -10
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        bak_count=$(find "$config_root" -maxdepth 1 -name 'AGENTS.md.bak.*' 2>/dev/null | wc -l)
        if [ "$bak_count" -ge 1 ]; then
            print_pass "Backup-trigger re-run: AGENTS.md backup created ($bak_count)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "Backup-trigger re-run: AGENTS.md backup NOT created"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        # Verify the new plugin content was written
        if grep -q "custom user content" "$config_root/AGENTS.md" 2>/dev/null; then
            print_warn "Backup-trigger: old custom content still present (plugin may not have overwritten)"
            WARN_COUNT=$((WARN_COUNT + 1))
        else
            print_pass "Backup-trigger: plugin content successfully overwritten"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    fi

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 3: Project + Claude
# =============================================================================
scenario_project_claude() {
    print_section_header "Scenario: project + claude"

    if ! supports_tool "claude"; then
        print_info "Skipping: init.sh does not support 'claude' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project claude <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$(resolve_project_config_root "claude" "$tmp_pwd")"
    check_common_artifacts "$config_root" "claude"

    # Project Claude specific: CLAUDE.md in project root or CONFIG_ROOT
    local claude_md="$tmp_pwd/CLAUDE.md"
    local claude_md_alt="$config_root/CLAUDE.md"
    if [ -e "$claude_md" ] || [ -e "$claude_md_alt" ]; then
        print_pass "CLAUDE.md exists (project root or CONFIG_ROOT)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "CLAUDE.md is missing (not in project root or CONFIG_ROOT)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    if { [ -f "$claude_md" ] && [ ! -L "$claude_md" ] && grep -q "$TEAM_DIR" "$claude_md" 2>/dev/null; } || \
       { [ -f "$claude_md_alt" ] && [ ! -L "$claude_md_alt" ] && grep -q "$TEAM_DIR" "$claude_md_alt" 2>/dev/null; }; then
        print_pass "CLAUDE.md contains absolute paths (project mode rewrite)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ -L "$claude_md" ] || [ -L "$claude_md_alt" ]; then
        print_pass "CLAUDE.md is a symlink (plugin dir = PWD)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # Repos should NOT be symlinked into CONFIG_ROOT in project mode
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ ! -e "$config_root/$repo" ]; then
            print_pass "$repo not symlinked into CONFIG_ROOT (project mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo unexpectedly present in CONFIG_ROOT (project mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")

    # Verify Claude can discover installed content
    verify_claude_discovery "$config_root" "project" "$tmp_pwd"

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 4: Global + Claude
# =============================================================================
scenario_global_claude() {
    print_section_header "Scenario: global + claude"

    if ! supports_tool "claude"; then
        print_info "Skipping: init.sh does not support 'claude' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global claude <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.claude"
    check_common_artifacts "$config_root" "claude"

    # Global Claude specific: CLAUDE.md in CONFIG_ROOT
    # (only if init.sh performs sed rewrite)
    if grep -q 'ESCAPED_ROOT' "$INIT_SCRIPT" 2>/dev/null; then
        local config_file="$config_root/CLAUDE.md"
        if [ -f "$config_file" ] && [ ! -L "$config_file" ]; then
            print_pass "CLAUDE.md is a regular file (global mode copy, not symlink)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "CLAUDE.md is missing or is a symlink (expected regular file in global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        if grep -q "$TEAM_DIR" "$config_file" 2>/dev/null; then
            print_pass "CLAUDE.md contains absolute paths (global mode rewrite detected)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_warn "CLAUDE.md does not contain absolute paths (may be OK if no relative refs)"
            WARN_COUNT=$((WARN_COUNT + 1))
        fi
    fi

    # Repos SHOULD be symlinked into CONFIG_ROOT in global mode
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ -L "$config_root/$repo" ]; then
            print_pass "$repo symlinked into CONFIG_ROOT (global mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo NOT symlinked into CONFIG_ROOT (global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")

    # Verify Claude can discover installed content
    verify_claude_discovery "$config_root" "global" ""

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 5: Project + Trae IDE (auto-detect .trae)
# =============================================================================
scenario_project_trae_ide() {
    print_section_header "Scenario: project + trae (IDE path)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    # Pre-create ~/.trae to simulate TRAE IDE environment
    mkdir -p "$tmp_home/.trae"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project trae <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$(resolve_project_config_root "trae" "$tmp_pwd")"
    if [ -d "$config_root" ] || [ -d "$tmp_pwd/.trae" ]; then
        [ -d "$config_root" ] || config_root="$tmp_pwd/.trae"
        print_pass "Artifacts installed to .trae/ (IDE path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".trae/ directory not found after IDE-path installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    if echo "$output" | grep -q "Detected: TRAE IDE"; then
        print_pass "Output contains TRAE IDE detection message"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "Output missing TRAE IDE detection message"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi

    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 6: Project + Trae Plugin (auto-detect .marscode)
# =============================================================================
scenario_project_trae_plugin() {
    print_section_header "Scenario: project + trae (Plugin path)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    # Pre-create ~/.marscode (but NOT ~/.trae) to simulate TRAE Plugin environment
    mkdir -p "$tmp_home/.marscode"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project trae <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Some plugins don't have detect_trae_variant() and always use .trae
    local expected_dir=".marscode"
    if ! grep -q 'detect_trae_variant' "$INIT_SCRIPT" 2>/dev/null; then
        expected_dir=".trae"
    fi

    local config_root="$tmp_pwd/$expected_dir"
    if grep -q "CONFIG_ROOT=\"\$PLUGIN_ROOT/${expected_dir}" "$INIT_SCRIPT" 2>/dev/null; then
        config_root="$TEAM_DIR/$expected_dir"
    fi
    if [ -d "$config_root" ] || [ -d "$tmp_pwd/$expected_dir" ]; then
        [ -d "$config_root" ] || config_root="$tmp_pwd/$expected_dir"
        print_pass "Artifacts installed to ${expected_dir}/ (Plugin path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "${expected_dir}/ directory not found after Plugin-path installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    if echo "$output" | grep -q "Detected: TRAE Plugin"; then
        print_pass "Output contains TRAE Plugin detection message"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "Output missing TRAE Plugin detection message"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi

    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 7: Project + Trae CLI (auto-detect .traecli)
# =============================================================================
scenario_project_trae_cli() {
    print_section_header "Scenario: project + trae (CLI path)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    # Pre-create ~/.traecli to simulate TRAE CLI environment
    mkdir -p "$tmp_home/.traecli"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project trae <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Some plugins don't have detect_trae_variant() and always use .trae
    local expected_dir=".traecli"
    if ! grep -q 'detect_trae_variant' "$INIT_SCRIPT" 2>/dev/null; then
        expected_dir=".trae"
    fi

    local config_root="$tmp_pwd/$expected_dir"
    if grep -q "CONFIG_ROOT=\"\$PLUGIN_ROOT/${expected_dir}" "$INIT_SCRIPT" 2>/dev/null; then
        config_root="$TEAM_DIR/$expected_dir"
    fi
    if [ -d "$config_root" ] || [ -d "$tmp_pwd/$expected_dir" ]; then
        [ -d "$config_root" ] || config_root="$tmp_pwd/$expected_dir"
        print_pass "Artifacts installed to ${expected_dir}/ (CLI path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "${expected_dir}/ directory not found after CLI-path installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    if echo "$output" | grep -q "Detected: TRAE CLI"; then
        print_pass "Output contains TRAE CLI detection message"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "Output missing TRAE CLI detection message"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi

    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 8: Global + Trae IDE (auto-detect ~/.trae-cn)
# =============================================================================
# Regression guard for issue #355: global + trae must enter
# detect_trae_variant branch (previously dead code due to
# `&& [ "$LEVEL" = "project" ]`), installing to ~/.trae-cn, NOT ~/.claude.
scenario_global_trae_ide() {
    print_section_header "Scenario: global + trae (IDE path)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    # Pre-create ~/.trae-cn to simulate TRAE IDE environment (global detect target)
    mkdir -p "$tmp_home/.trae-cn"

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global trae <<< "y" 2>&1) || exit_code=$?
    assert_no_shell_warnings "$output"

    # Teams that explicitly reject global + trae (design choice): assert exit 1
    # and no silent fall-through to ~/.claude, then skip install assertions.
    if trae_global_unsupported; then
        if [ "$exit_code" -ne 0 ]; then
            print_pass "init.sh rejected global + trae as designed (exit $exit_code)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "init.sh should reject global + trae but exited 0"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        if [ -d "$tmp_home/.claude" ]; then
            print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        rm -rf "$tmp_home"
        cleanup_team_artifacts
        trap - EXIT
        return 0
    fi

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.trae-cn"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .trae-cn/ (IDE path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".trae-cn/ directory not found after IDE-path installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Regression core: must NOT fall through to ~/.claude (the bug from issue #355)
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    if echo "$output" | grep -q "Detected: TRAE IDE"; then
        print_pass "Output contains TRAE IDE detection message"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "Output missing TRAE IDE detection message"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi

    verify_global_repo_symlinks "$config_root"
    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 9: Global + Trae Plugin (auto-detect ~/.marscode)
# =============================================================================
scenario_global_trae_plugin() {
    print_section_header "Scenario: global + trae (Plugin path)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    # Pre-create ~/.marscode (but NOT ~/.trae-cn) to simulate TRAE Plugin environment
    mkdir -p "$tmp_home/.marscode"

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global trae <<< "y" 2>&1) || exit_code=$?
    assert_no_shell_warnings "$output"

    # Teams that explicitly reject global + trae (design choice): assert exit 1
    # and no silent fall-through to ~/.claude, then skip install assertions.
    if trae_global_unsupported; then
        if [ "$exit_code" -ne 0 ]; then
            print_pass "init.sh rejected global + trae as designed (exit $exit_code)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "init.sh should reject global + trae but exited 0"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        if [ -d "$tmp_home/.claude" ]; then
            print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        rm -rf "$tmp_home"
        cleanup_team_artifacts
        trap - EXIT
        return 0
    fi

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.marscode"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .marscode/ (Plugin path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".marscode/ directory not found after Plugin-path installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Regression core: must NOT fall through to ~/.claude (issue #355)
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    if echo "$output" | grep -q "Detected: TRAE Plugin"; then
        print_pass "Output contains TRAE Plugin detection message"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "Output missing TRAE Plugin detection message"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi

    verify_global_repo_symlinks "$config_root"
    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 10: Global + Trae CLI (auto-detect ~/.traecli)
# =============================================================================
scenario_global_trae_cli() {
    print_section_header "Scenario: global + trae (CLI path)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    # Pre-create ~/.traecli (but NOT ~/.trae-cn / ~/.marscode) to simulate TRAE CLI environment
    mkdir -p "$tmp_home/.traecli"

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global trae <<< "y" 2>&1) || exit_code=$?
    assert_no_shell_warnings "$output"

    # Teams that explicitly reject global + trae (design choice): assert exit 1
    # and no silent fall-through to ~/.claude, then skip install assertions.
    if trae_global_unsupported; then
        if [ "$exit_code" -ne 0 ]; then
            print_pass "init.sh rejected global + trae as designed (exit $exit_code)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "init.sh should reject global + trae but exited 0"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        if [ -d "$tmp_home/.claude" ]; then
            print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        rm -rf "$tmp_home"
        cleanup_team_artifacts
        trap - EXIT
        return 0
    fi

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.traecli"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .traecli/ (CLI path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".traecli/ directory not found after CLI-path installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Regression core: must NOT fall through to ~/.claude (issue #355)
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    if echo "$output" | grep -q "Detected: TRAE CLI"; then
        print_pass "Output contains TRAE CLI detection message"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "Output missing TRAE CLI detection message"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi

    verify_global_repo_symlinks "$config_root"
    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 11: Global + Trae unknown (no ~/.trae-cn|~/.marscode|~/.traecli)
# =============================================================================
# Covers the unknown fallback branch of detect_trae_variant. Before the
# issue #355 fix this path was unreachable (dead code) and silently fell
# through to ~/.claude; after the fix it correctly lands on ~/.trae-cn with
# a warning. This is the 4th detect_trae_variant exit path (ide/plugin/cli
# covered by scenarios 8/9/10).
scenario_global_trae_unknown() {
    print_section_header "Scenario: global + trae (unknown fallback)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    # Intentionally do NOT pre-create any TRAE directory → TRAE_VARIANT=unknown

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global trae <<< "y" 2>&1) || exit_code=$?
    assert_no_shell_warnings "$output"

    # Teams that explicitly reject global + trae (design choice): assert exit 1
    # and no silent fall-through to ~/.claude, then skip install assertions.
    if trae_global_unsupported; then
        if [ "$exit_code" -ne 0 ]; then
            print_pass "init.sh rejected global + trae as designed (exit $exit_code)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "init.sh should reject global + trae but exited 0"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        if [ -d "$tmp_home/.claude" ]; then
            print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        rm -rf "$tmp_home"
        cleanup_team_artifacts
        trap - EXIT
        return 0
    fi

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Unknown fallback must land on ~/.trae-cn (NOT ~/.claude)
    local config_root="$tmp_home/.trae-cn"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .trae-cn/ (unknown fallback path)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".trae-cn/ directory not found after unknown-fallback installation"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Regression core: must NOT fall through to ~/.claude (issue #355)
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    verify_global_repo_symlinks "$config_root"
    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 12: Global + Trae detection priority (.trae-cn beats .marscode)
# =============================================================================
# detect_trae_variant checks ~/.trae-cn → ~/.marscode → ~/.traecli in order,
# first match wins. When multiple dirs coexist, IDE (.trae-cn) must take
# precedence over Plugin (.marscode). This branch of detect_trae_variant
# (the priority chain) was unreachable before the issue #355 fix and is not
# covered by single-dir scenarios 8-11.
scenario_global_trae_priority() {
    print_section_header "Scenario: global + trae (detection priority: .trae-cn > .marscode)"

    if ! supports_tool "trae"; then
        print_info "Skipping: init.sh does not support 'trae' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    # Pre-create BOTH ~/.trae-cn and ~/.marscode → IDE must win (first match)
    mkdir -p "$tmp_home/.trae-cn" "$tmp_home/.marscode"

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global trae <<< "y" 2>&1) || exit_code=$?
    assert_no_shell_warnings "$output"

    # Teams that explicitly reject global + trae (design choice): assert exit 1
    # and no silent fall-through to ~/.claude, then skip install assertions.
    if trae_global_unsupported; then
        if [ "$exit_code" -ne 0 ]; then
            print_pass "init.sh rejected global + trae as designed (exit $exit_code)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "init.sh should reject global + trae but exited 0"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        if [ -d "$tmp_home/.claude" ]; then
            print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        rm -rf "$tmp_home"
        cleanup_team_artifacts
        trap - EXIT
        return 0
    fi

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Priority core: must install to ~/.trae-cn (IDE wins over Plugin)
    local config_root="$tmp_home/.trae-cn"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .trae-cn/ (IDE priority over Plugin)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".trae-cn/ not found (priority broken: Plugin may have won)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # And must NOT have installed to ~/.marscode
    if [ -d "$tmp_home/.marscode/skills" ] || [ -d "$tmp_home/.marscode/agents" ]; then
        print_fail "Artifacts leaked into .marscode/ (priority broken)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "No artifacts in .marscode/ (IDE priority correct)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # Regression core: must NOT fall through to ~/.claude (issue #355)
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (regression: trae branch not reached, issue #355)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (trae branch reached, issue #355 fixed)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    verify_global_repo_symlinks "$config_root"
    check_common_artifacts "$config_root" "trae"

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 13: Project + CodeArts
# =============================================================================
scenario_project_codearts() {
    print_section_header "Scenario: project + codearts"

    if ! supports_tool "codearts"; then
        print_info "Skipping: init.sh does not support 'codearts' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project codearts <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$(resolve_project_config_root "codearts" "$tmp_pwd")"
    check_common_artifacts "$config_root" "codearts"

    # Project CodeArts specific: AGENTS.md in PWD or CONFIG_ROOT (codearts uses AGENTS.md, not CLAUDE.md)
    if [ -e "$tmp_pwd/AGENTS.md" ] || [ -e "$config_root/AGENTS.md" ]; then
        print_pass "AGENTS.md exists (PWD or CONFIG_ROOT)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "AGENTS.md is missing (not in PWD or CONFIG_ROOT)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    local _agents_md="$tmp_pwd/AGENTS.md"
    [ -f "$_agents_md" ] || _agents_md="$config_root/AGENTS.md"
    if [ -f "$_agents_md" ] && [ ! -L "$_agents_md" ] && grep -q "$TEAM_DIR" "$_agents_md" 2>/dev/null; then
        print_pass "AGENTS.md contains absolute paths (project mode rewrite)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ -L "$_agents_md" ]; then
        print_pass "AGENTS.md is a symlink (plugin dir = PWD)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # CodeArts must NOT create CLAUDE.md
    if [ -e "$tmp_pwd/CLAUDE.md" ]; then
        print_fail "CLAUDE.md created (codearts should use AGENTS.md)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "CLAUDE.md not created (codearts uses AGENTS.md)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # CodeArts must NOT create ~/.claude
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (codearts should use ~/.codeartsdoer)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (codearts branch reached correctly)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # Repos should NOT be symlinked into CONFIG_ROOT in project mode
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ ! -e "$config_root/$repo" ]; then
            print_pass "$repo not symlinked into CONFIG_ROOT (project mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo unexpectedly present in CONFIG_ROOT (project mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 14: Global + CodeArts
# =============================================================================
scenario_global_codearts() {
    print_section_header "Scenario: global + codearts"

    if ! supports_tool "codearts"; then
        print_info "Skipping: init.sh does not support 'codearts' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global codearts <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.codeartsdoer"
    check_common_artifacts "$config_root" "codearts"

    # Global CodeArts specific: AGENTS.md (not CLAUDE.md)
    if grep -q 'ESCAPED_ROOT' "$INIT_SCRIPT" 2>/dev/null; then
        local config_file="$config_root/AGENTS.md"
        if [ -f "$config_file" ] && [ ! -L "$config_file" ]; then
            print_pass "AGENTS.md is a regular file (global mode copy, not symlink)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "AGENTS.md is missing or is a symlink (expected regular file in global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        if grep -q "$TEAM_DIR" "$config_file" 2>/dev/null; then
            print_pass "AGENTS.md contains absolute paths (global mode rewrite detected)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_warn "AGENTS.md does not contain absolute paths (may be OK if no relative refs)"
            WARN_COUNT=$((WARN_COUNT + 1))
        fi
    fi

    # CodeArts must NOT create ~/.claude
    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (codearts should use ~/.codeartsdoer)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (codearts uses ~/.codeartsdoer)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    # Repos SHOULD be symlinked into CONFIG_ROOT in global mode
    local repo
    while IFS= read -r repo; do
        [ -n "$repo" ] || continue
        if [ -L "$config_root/$repo" ]; then
            print_pass "$repo symlinked into CONFIG_ROOT (global mode, correct)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "$repo NOT symlinked into CONFIG_ROOT (global mode)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < <(get_git_repo_names "$INIT_SCRIPT")

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 15: Project + Cursor
# =============================================================================
scenario_project_cursor() {
    print_section_header "Scenario: project + cursor"

    if ! supports_tool "cursor"; then
        print_info "Skipping: init.sh does not support 'cursor' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project cursor <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$(resolve_project_config_root "cursor" "$tmp_pwd")"
    check_common_artifacts "$config_root" "cursor"

    if [ -e "$tmp_pwd/AGENTS.md" ] || [ -e "$config_root/AGENTS.md" ]; then
        print_pass "AGENTS.md exists (PWD or CONFIG_ROOT)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "AGENTS.md is missing (not in PWD or CONFIG_ROOT)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    if [ -e "$tmp_pwd/CLAUDE.md" ]; then
        print_fail "CLAUDE.md created (cursor should use AGENTS.md)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "CLAUDE.md not created (cursor uses AGENTS.md)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 16: Global + Cursor
# =============================================================================
scenario_global_cursor() {
    print_section_header "Scenario: global + cursor"

    if ! supports_tool "cursor"; then
        print_info "Skipping: init.sh does not support 'cursor' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global cursor <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.cursor"
    check_common_artifacts "$config_root" "cursor"

    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (cursor should use ~/.cursor)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (cursor uses ~/.cursor)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Scenario 17: Project + Copilot
# =============================================================================
scenario_project_copilot() {
    print_section_header "Scenario: project + copilot"

    if ! supports_tool "copilot"; then
        print_info "Skipping: init.sh does not support 'copilot' tool"
        return 0
    fi

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project copilot <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$(resolve_project_config_root "copilot" "$tmp_pwd")"
    check_common_artifacts "$config_root" "copilot"

    if [ -e "$tmp_pwd/AGENTS.md" ] || [ -e "$config_root/AGENTS.md" ]; then
        print_pass "AGENTS.md exists (PWD or CONFIG_ROOT)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "AGENTS.md is missing (not in PWD or CONFIG_ROOT)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    if [ -e "$tmp_pwd/CLAUDE.md" ]; then
        print_fail "CLAUDE.md created (copilot should use AGENTS.md)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "CLAUDE.md not created (copilot uses AGENTS.md)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    rm -rf "$tmp_home" "$tmp_pwd"
    cleanup_team_artifacts "$tmp_pwd"
    trap - EXIT
}

# =============================================================================
# Scenario 18: Global + Copilot
# =============================================================================
scenario_global_copilot() {
    print_section_header "Scenario: global + copilot"

    if ! supports_tool "copilot"; then
        print_info "Skipping: init.sh does not support 'copilot' tool"
        return 0
    fi

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global copilot <<< "y" 2>&1) || exit_code=$?

    assert_no_shell_warnings "$output"
    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_home/.copilot"
    check_common_artifacts "$config_root" "copilot"

    if [ -d "$tmp_home/.claude" ]; then
        print_fail "~/.claude created (copilot should use ~/.copilot)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        print_pass "~/.claude not created (copilot uses ~/.copilot)"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi

    rm -rf "$tmp_home"
    cleanup_team_artifacts
    trap - EXIT
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "========================================"
    echo "  Behavior Test: Init Script Install"
    echo "========================================"
    echo ""

    # Defensive guard: record current git branch before running any test.
    # Some init.sh scripts run `git checkout` on sub-repo directories that
    # may not be real git repos (fake repos created by setup_fake_repos),
    # causing git to fall through to the parent cann_skills repo and switch
    # its branch. This guard detects such leaks and restores the branch.
    local guard_branch=""
    if git -C "$SKILLS_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
        guard_branch=$(git -C "$SKILLS_DIR" branch --show-current 2>/dev/null || echo "")
        if [ -n "$guard_branch" ]; then
            echo -e "${CYAN}[GUARD]${NC} Recording current branch: '$guard_branch'"
            echo ""
        fi
    fi

    # Find all teams with init.sh (scan both plugins-official and plugins-community)
    local teams=()
    local team_roots=()
    for plugin_root_dir in "$SKILLS_DIR/plugins-official" "$SKILLS_DIR/plugins-community"; do
        [ -d "$plugin_root_dir" ] || continue
        for team_dir in "$plugin_root_dir"/*; do
            [ -d "$team_dir" ] || continue
            local init="$team_dir/init.sh"
            [ -f "$init" ] || continue
            [ -x "$init" ] || continue
            teams+=("$(basename "$team_dir")")
            team_roots+=("$plugin_root_dir")
        done
    done

    if [ ${#teams[@]} -eq 0 ]; then
        print_fail "No teams with init.sh found"
        exit 1
    fi

    echo "Teams to test: ${teams[*]}"
    echo ""

    local idx
    for idx in "${!teams[@]}"; do
        team="${teams[$idx]}"
        local team_root="${team_roots[$idx]}"
        TEAM="$team"
        TEAM_DIR="$team_root/$TEAM"
        INIT_SCRIPT="$TEAM_DIR/init.sh"

        EXPECTED_SKILL_COUNT=$(get_expected_skill_count)
        EXPECTED_AGENT_COUNT=$(get_expected_agent_count)
        EXPECTED_SKILLS_LIST=$(grep 'INCLUDED_SKILLS="' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*INCLUDED_SKILLS="//;s/"$//' || true)

        echo "========================================"
        echo "  Team: $TEAM"
        echo "  Expected skills: $EXPECTED_SKILL_COUNT"
        echo "  Expected agents: $EXPECTED_AGENT_COUNT"
        echo "========================================"
        echo ""

        # Pre-check
        if [ ! -f "$INIT_SCRIPT" ]; then
            print_fail "init.sh not found: $INIT_SCRIPT"
            continue
        fi
        if [ ! -x "$INIT_SCRIPT" ]; then
            print_fail "init.sh not executable: $INIT_SCRIPT"
            continue
        fi

        # Clean any leftover artifacts from previous interrupted runs
        cleanup_team_artifacts
        cleanup_fake_repos

        scenario_project_opencode
        scenario_global_opencode
        scenario_project_claude
        scenario_global_claude

        scenario_project_trae_ide
        scenario_project_trae_plugin
        scenario_project_trae_cli

        scenario_global_trae_ide
        scenario_global_trae_plugin
        scenario_global_trae_cli
        scenario_global_trae_unknown
        scenario_global_trae_priority

        scenario_project_codearts
        scenario_global_codearts

        scenario_project_cursor
        scenario_global_cursor

        scenario_project_copilot
        scenario_global_copilot

        # Final cleanup per team
        cleanup_team_artifacts
        cleanup_fake_repos
    done

    # Defensive guard: verify branch was not changed by init.sh git operations.
    if [ -n "$guard_branch" ]; then
        local final_branch
        final_branch=$(git -C "$SKILLS_DIR" branch --show-current 2>/dev/null || echo "")
        if [ "$final_branch" != "$guard_branch" ]; then
            echo ""
            echo -e "${RED}[GUARD ERROR]${NC} Git branch changed during test: '$guard_branch' → '$final_branch'"
            echo -e "${RED}[GUARD ERROR]${NC} This indicates a test isolation bug (init.sh git ops leaked to parent repo)"
            echo -e "${YELLOW}[GUARD]${NC} Restoring branch to '$guard_branch'"
            git -C "$SKILLS_DIR" checkout "$guard_branch" 2>/dev/null || true
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            echo -e "${GREEN}[GUARD]${NC} Branch unchanged: '$guard_branch'"
        fi
    fi

    echo ""
    echo "========================================"
    echo "  Init Behavior Test Summary"
    echo "========================================"
    echo "  Teams: ${#teams[@]}"
    echo "  Passed: $PASS_COUNT"
    echo "  Failed: $FAIL_COUNT"
    echo ""

    if [ "$FAIL_COUNT" -gt 0 ]; then
        print_status_failed
        exit 1
    else
        print_status_passed
        exit 0
    fi
}

main "$@"
