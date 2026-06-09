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

# Remove artifacts created by init.sh inside the team directory (or given dir)
cleanup_team_artifacts() {
    local dir="${1:-$TEAM_DIR}"
    rm -rf "$dir/.opencode" "$dir/.claude" "$dir/.trae" "$dir/.marscode" "$dir/.traecli"
}

get_expected_skill_count() {
    local skills
    skills=$(grep 'INCLUDED_SKILLS=' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*="//;s/"$//' || true)
    echo "$skills" | wc -w
}

get_expected_agent_count() {
    local pattern
    pattern=$(grep 'INCLUDED_AGENT_PATTERN=' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*="//;s/"$//' || true)
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

# Verify manifest JSON structure and consistency
verify_manifest() {
    local config_root="$1"
    local expected_skills="$2"
    local expected_agents="$3"
    local manifest="$config_root/cannbot-manifest.json"

    if [ ! -f "$manifest" ]; then
        print_fail "manifest not found: $manifest"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Validate JSON
    if python3 -c "import json; json.load(open('$manifest'))" 2>/dev/null; then
        print_pass "manifest is valid JSON"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "manifest is invalid JSON"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Check skill count
    local actual_skills
    actual_skills=$(python3 -c "import json; print(len(json.load(open('$manifest')).get('installed_skills', [])))" 2>/dev/null || echo 0)
    if [ "$actual_skills" -eq "$expected_skills" ]; then
        print_pass "manifest: installed_skills count = $actual_skills (expected $expected_skills)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "manifest: installed_skills count = $actual_skills (expected $expected_skills)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Check agent count
    local actual_agents
    actual_agents=$(python3 -c "import json; print(len(json.load(open('$manifest')).get('installed_agents', [])))" 2>/dev/null || echo 0)
    if [ "$actual_agents" -eq "$expected_agents" ]; then
        print_pass "manifest: installed_agents count = $actual_agents (expected $expected_agents)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "manifest: installed_agents count = $actual_agents (expected $expected_agents)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Check brand/level/tool fields
    for field in brand level tool; do
        local val
        val=$(python3 -c "import json; print(json.load(open('$manifest')).get('$field',''))" 2>/dev/null || true)
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

    # 2. skills/ directory exists with expected count
    local skill_dir="$config_root/skills"
    if [ -d "$skill_dir" ]; then
        local actual_skills
        actual_skills=$(find "$skill_dir" -maxdepth 1 \( -type l -o -type d \) | wc -l)
        # subtract 1 for the directory itself
        actual_skills=$((actual_skills - 1))
        if [ "$actual_skills" -eq "$EXPECTED_SKILL_COUNT" ]; then
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
    if [ -d "$agent_dir" ]; then
        local actual_agents
        actual_agents=$(find "$agent_dir" -maxdepth 1 -type l | wc -l)
        if [ "$actual_agents" -eq "$EXPECTED_AGENT_COUNT" ]; then
            print_pass "agents/ contains $actual_agents symlink(s) (expected $EXPECTED_AGENT_COUNT)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            print_fail "agents/ contains $actual_agents symlink(s) (expected $EXPECTED_AGENT_COUNT)"
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
    verify_manifest "$config_root" "$EXPECTED_SKILL_COUNT" "$EXPECTED_AGENT_COUNT"
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
    pattern=$(grep 'INCLUDED_AGENT_PATTERN=' "$INIT_SCRIPT" | head -1 | sed 's/.*="//;s/"$//')
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

    # Claude reads CLAUDE.md in project root
    local claude_md
    if [ "$level" = "project" ]; then
        claude_md="$tmp_pwd/CLAUDE.md"
    else
        claude_md="$config_root/CLAUDE.md"
    fi
    if [ -f "$claude_md" ] && [ -s "$claude_md" ]; then
        print_pass "Claude scan: CLAUDE.md in project root is present and non-empty"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "Claude scan: CLAUDE.md in project root missing or empty"
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

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_pwd/.opencode"
    check_common_artifacts "$config_root" "opencode"

    # Project OpenCode specific: AGENTS.md in PWD (discovered by upward traversal)
    # When install_path differs from plugin dir, AGENTS.md is a copy with absolute paths
    if [ -e "$tmp_pwd/AGENTS.md" ]; then
        print_pass "PWD/AGENTS.md exists"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "PWD/AGENTS.md is missing"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    if [ -f "$tmp_pwd/AGENTS.md" ] && [ ! -L "$tmp_pwd/AGENTS.md" ] && grep -q "$TEAM_DIR" "$tmp_pwd/AGENTS.md" 2>/dev/null; then
        print_pass "PWD/AGENTS.md contains absolute paths (project mode rewrite)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ -L "$tmp_pwd/AGENTS.md" ]; then
        print_pass "PWD/AGENTS.md is a symlink (plugin dir = PWD)"
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
    # Only test teams that have safe_install_file (skip legacy init.sh)
    if [ -f "$config_root/AGENTS.md" ] && grep -q 'safe_install_file()' "$INIT_SCRIPT" 2>/dev/null; then
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

    local tmp_home tmp_pwd
    tmp_home=$(mktemp -d)
    tmp_pwd=$(mktemp -d)

    trap "rm -rf '$tmp_home' '$tmp_pwd'; cleanup_team_artifacts '$tmp_pwd'" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts "$tmp_pwd"

    local output
    local exit_code=0
    output=$(cd "$tmp_pwd" && HOME="$tmp_home" bash "$INIT_SCRIPT" project claude <<< "y" 2>&1) || exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_pwd/.claude"
    check_common_artifacts "$config_root" "claude"

    # Project Claude specific: CLAUDE.md in project root
    local claude_md="$tmp_pwd/CLAUDE.md"
    if [ -e "$claude_md" ]; then
        print_pass "CLAUDE.md in project root exists"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "CLAUDE.md in project root is missing"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    if [ -f "$claude_md" ] && [ ! -L "$claude_md" ] && grep -q "$TEAM_DIR" "$claude_md" 2>/dev/null; then
        print_pass "CLAUDE.md contains absolute paths (project mode rewrite)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ -L "$claude_md" ]; then
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

    local tmp_home
    tmp_home=$(mktemp -d)

    trap "rm -rf '$tmp_home'; cleanup_team_artifacts" EXIT

    setup_fake_repos "$TEAM_DIR"
    cleanup_team_artifacts

    local output
    local exit_code=0
    output=$(HOME="$tmp_home" bash "$INIT_SCRIPT" global claude <<< "y" 2>&1) || exit_code=$?

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

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_pwd/.trae"
    if [ -d "$config_root" ]; then
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

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_pwd/.marscode"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .marscode/ (Plugin path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".marscode/ directory not found after Plugin-path installation"
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

    if [ "$exit_code" -eq 0 ]; then
        print_pass "init.sh exited with code 0"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail "init.sh exited with code $exit_code"
        echo "$output" | tail -20
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    local config_root="$tmp_pwd/.traecli"
    if [ -d "$config_root" ]; then
        print_pass "Artifacts installed to .traecli/ (CLI path detected)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_fail ".traecli/ directory not found after CLI-path installation"
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
# Main
# =============================================================================
main() {
    echo "========================================"
    echo "  Behavior Test: Init Script Install"
    echo "========================================"
    echo ""

    # Find all teams with init.sh
    local teams=()
    for team_dir in "$SKILLS_DIR/plugins-official"/*; do
        [ -d "$team_dir" ] || continue
        local init="$team_dir/init.sh"
        [ -f "$init" ] || continue
        [ -x "$init" ] || continue
        teams+=("$(basename "$team_dir")")
    done

    if [ ${#teams[@]} -eq 0 ]; then
        print_fail "No teams with init.sh found"
        exit 1
    fi

    echo "Teams to test: ${teams[*]}"
    echo ""

    for team in "${teams[@]}"; do
        TEAM="$team"
        TEAM_DIR="$SKILLS_DIR/plugins-official/$TEAM"
        INIT_SCRIPT="$TEAM_DIR/init.sh"

        EXPECTED_SKILL_COUNT=$(get_expected_skill_count)
        EXPECTED_AGENT_COUNT=$(get_expected_agent_count)
        EXPECTED_SKILLS_LIST=$(grep 'INCLUDED_SKILLS=' "$INIT_SCRIPT" 2>/dev/null | head -1 | sed 's/.*="//;s/"$//' || true)

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

        # Final cleanup per team
        cleanup_team_artifacts
        cleanup_fake_repos
    done

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
