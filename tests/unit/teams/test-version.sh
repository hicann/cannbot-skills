#!/usr/bin/env bash
# =============================================================================
# Test: Team Plugin Version Care
# =============================================================================
# Validates that plugin.json version is correctly bumped when dependencies change.
#
# Rules:
# - PATCH (3rd digit): Skill dependency changed (content hash or list changed)
# - MINOR (2nd digit): Agent dependency changed (content hash or list changed)
# - MAJOR (1st digit): Breaking team interface changes (not auto-detected)
#
# Version state is stored in tests/.version-state/<team-name>.json
#
# Supports incremental testing via INCREMENTAL_TEAMS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Team Plugin Version Care ==="
echo ""
echo "Validates plugin.json version correctness against dependency changes."
echo "Run time: ~5 seconds (no CLI needed)"
echo ""

# Check for incremental mode
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed teams"
    echo ""
fi

TEAM_BASE="$SKILLS_DIR/ops/teams"

# Counters
total_teams=0
pass_count=0
fail_count=0
skip_count=0

# Get teams to test (filtered if in incremental mode)
TEAMS_TO_TEST=$(get_teams_to_test)
total_teams=$(echo "$TEAMS_TO_TEST" | grep -c . || echo "0")

echo "Teams to check: $total_teams"
echo ""

# ============================================
# Helper: build hash map as "name:hash" lines
# ============================================
build_skill_hashes() {
    local plugin_dir="$1"
    local plugin_json="$2"

    while IFS= read -r skill_rel_path; do
        [ -z "$skill_rel_path" ] && continue
        skill_rel_path="${skill_rel_path#./}"
        local skill_full="$plugin_dir/$skill_rel_path"
        if [ -L "$skill_full" ]; then
            skill_full=$(readlink -f "$skill_full")
        fi
        local skill_file="$skill_full/SKILL.md"
        local skill_name
        skill_name=$(basename "$skill_full")
        local hash
        hash=$(compute_file_hash "$skill_file")
        echo "$skill_name:$hash"
    done < <(extract_plugin_skills "$plugin_json")
}

build_agent_hashes() {
    local plugin_dir="$1"
    local plugin_json="$2"

    while IFS= read -r agent_rel_path; do
        [ -z "$agent_rel_path" ] && continue
        agent_rel_path="${agent_rel_path#./}"
        local agent_full="$plugin_dir/$agent_rel_path"
        local agent_name
        agent_name=$(basename "$agent_full" .md)
        local hash
        hash=$(compute_file_hash "$agent_full")
        echo "$agent_name:$hash"
    done < <(extract_plugin_agents "$plugin_json")
}

# ============================================
# Helper: compare hash maps
# Returns: lines describing changes
# ============================================
compare_hashes() {
    local current_file="$1"
    local loaded_lines="$2"

    local changes=""

    # Check for removed or modified entries
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local name="${line%%:*}"
        local old_hash="${line#*:}"
        local current_hash
        current_hash=$(grep "^${name}:" "$current_file" 2>/dev/null | head -1 | cut -d: -f2- || true)
        if [ -z "$current_hash" ]; then
            changes="${changes}REMOVED: ${name}\n"
        elif [ "$current_hash" != "$old_hash" ]; then
            changes="${changes}MODIFIED: ${name} (${old_hash} -> ${current_hash})\n"
        fi
    done <<< "$loaded_lines"

    # Check for added entries
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local name="${line%%:*}"
        local found
        found=$(grep "^${name}:" <<< "$loaded_lines" 2>/dev/null || true)
        if [ -z "$found" ]; then
            changes="${changes}ADDED: ${name}\n"
        fi
    done < "$current_file"

    echo -e "$changes" | sed '/^$/d'
}

# ============================================
# Helper: save state using temp files
# ============================================
save_team_state() {
    local team_name="$1"
    local version="$2"
    local skills_file="$3"
    local agents_file="$4"

    mkdir -p "$VERSION_STATE_DIR"
    local state_file="$VERSION_STATE_DIR/$team_name.json"

    # Build skills JSON object
    local skills_obj="{"
    local first=true
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local name="${line%%:*}"
        local hash="${line#*:}"
        if $first; then first=false; else skills_obj+=","; fi
        skills_obj+="\"$name\":\"$hash\""
    done < "$skills_file"
    skills_obj+="}"

    # Build agents JSON object
    local agents_obj="{"
    first=true
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local name="${line%%:*}"
        local hash="${line#*:}"
        if $first; then first=false; else agents_obj+=","; fi
        agents_obj+="\"$name\":\"$hash\""
    done < "$agents_file"
    agents_obj+="}"

    cat > "$state_file" <<EOF
{
  "version": "$version",
  "timestamp": "$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')",
  "skills": $skills_obj,
  "agents": $agents_obj
}
EOF
}

# ============================================
# Main: Check each team's version
# ============================================
print_section_header "Version Check"

for team in $TEAMS_TO_TEST; do
    [ -z "$team" ] && continue

    # In incremental mode, check if this team should be tested
    if is_incremental_mode && ! should_test_team "$team"; then
        print_skip "$team: Not in changed list"
        ((skip_count++)) || true
        continue
    fi

    print_section "Team: $team"

    plugin_json=$(get_team_plugin_json "$team")
    if [ -z "$plugin_json" ] || [ ! -f "$plugin_json" ]; then
        print_skip "$team: plugin.json not found"
        continue
    fi

    current_version=$(extract_plugin_version "$plugin_json")
    if [ -z "$current_version" ]; then
        print_fail "$team: No version field in plugin.json"
        ((fail_count++)) || true
        continue
    fi

    if ! validate_semver "$current_version"; then
        print_fail "$team: Invalid SemVer format: $current_version"
        ((fail_count++)) || true
        continue
    fi

    print_info "Current version: $current_version"

    # Resolve plugin directory
    plugin_dir="$(dirname "$plugin_json")"
    # Relative paths in plugin.json (e.g., "./skills/...") are relative to the team root,
    # not the .claude-plugin directory. Resolve from team root instead.
    team_dir="$(dirname "$(dirname "$plugin_json")")"

    # Build current hashes into temp files
    current_skills_file=$(mktemp)
    current_agents_file=$(mktemp)
    build_skill_hashes "$team_dir" "$plugin_json" > "$current_skills_file"
    build_agent_hashes "$team_dir" "$plugin_json" > "$current_agents_file"

    skill_changed=false
    agent_changed=false
    skill_details=""
    agent_details=""
    should_save_state=false

    # Compare with saved state
    state_file="$VERSION_STATE_DIR/$team.json"
    if [ -f "$state_file" ]; then
        # Load previous state
        loaded_version=$(extract_plugin_version "$state_file")

        # Parse skills from state JSON using python3 or grep fallback
        if command -v python3 &>/dev/null; then
            loaded_skills=$(python3 <<PYEOF
import json
with open("${state_file}") as f:
    data = json.load(f)
for k,v in data.get('skills',{}).items():
    print(f'{k}:{v}')
PYEOF
)
            loaded_agents=$(python3 <<PYEOF
import json
with open("${state_file}") as f:
    data = json.load(f)
for k,v in data.get('agents',{}).items():
    print(f'{k}:{v}')
PYEOF
)
        else
            loaded_skills=""
            loaded_agents=""
        fi

        # Compare skills
        if [ -n "$loaded_skills" ]; then
            skill_details=$(compare_hashes "$current_skills_file" "$loaded_skills")
            if [ -n "$skill_details" ]; then
                skill_changed=true
            fi
        fi

        # Compare agents
        if [ -n "$loaded_agents" ]; then
            agent_details=$(compare_hashes "$current_agents_file" "$loaded_agents")
            if [ -n "$agent_details" ]; then
                agent_changed=true
            fi
        fi
    else
        # First run — no state, initialize
        print_info "First run — initializing version state"
    fi

    # ============================================
    # Determine if version bump is needed
    # ============================================
    # should_save_state already initialized above

    if [ "$skill_changed" = "true" ] || [ "$agent_changed" = "true" ]; then
        # Compare current version against the recorded version in state
        recorded_version="${loaded_version:-}"
        if [ -z "$recorded_version" ]; then
            # No state file — this shouldn't happen if changes detected, but handle gracefully
            print_warn "No previous state found, initializing"
            should_save_state=true
        else
            # Compute recommended version from the RECORDED version, not current
            recommended=$(recommend_version_bump "$recorded_version" "$skill_changed" "$agent_changed")

            echo ""
            if [ "$skill_changed" = "true" ]; then
                print_info "Skill changes detected:"
                echo -e "$skill_details" | sed 's/^/    - /'
            fi
            if [ "$agent_changed" = "true" ]; then
                print_info "Agent changes detected:"
                echo -e "$agent_details" | sed 's/^/    - /'
            fi
            echo ""

            cmp=$(semver_compare "$current_version" "$recommended")
            if [ "$cmp" = "-1" ]; then
                # current < recommended → FAIL
                version_line=$(grep -n '"version"' "$plugin_json" | head -1)
                print_fail "Version $current_version should be >= $recommended"
                echo "    Action: update the version field in $plugin_json"
                echo "    Location: $plugin_json:$version_line"
                ((fail_count++)) || true
                # FAIL: do NOT update state — keep old snapshot so next run still catches this
            else
                # current >= recommended → PASS
                print_pass "Version $current_version is up-to-date (was $recorded_version, now >= $recommended)"
                ((pass_count++)) || true
                should_save_state=true
            fi
        fi
    else
        # No changes — verify version matches saved state
        if [ -f "$state_file" ]; then
            loaded_version=$(extract_plugin_version "$state_file")
            if [ "$current_version" != "$loaded_version" ]; then
                cmp=$(semver_compare "$current_version" "$loaded_version")
                if [ "$cmp" = "-1" ]; then
                    print_fail "Version $current_version is lower than recorded $loaded_version"
                    ((fail_count++)) || true
                    # FAIL: do NOT update state
                else
                    print_pass "Version $current_version (upgraded from $loaded_version)"
                    ((pass_count++)) || true
                    should_save_state=true
                fi
            else
                print_pass "Version $current_version is consistent (no changes detected)"
                ((pass_count++)) || true
                # Consistent: no need to rewrite same state
            fi
        else
            # First run: initialize state
            print_info "First run — initializing version state"
            should_save_state=true
        fi
    fi

    # Only save state when test passes (new state or version upgrade confirmed)
    if $should_save_state; then
        save_team_state "$team" "$current_version" "$current_skills_file" "$current_agents_file"
    fi

    # Cleanup temp files
    rm -f "$current_skills_file" "$current_agents_file"

    echo ""
done

# ============================================
# Marketplace Version Consistency Check
# ============================================
print_section_header "Marketplace Version Consistency"

# package.json (OpenCode) and marketplace.json (Claude) are what users see
# when browsing the marketplace. If these don't match plugin.json, users
# won't see the updated version and won't know to upgrade.
PACKAGE_JSON="$SKILLS_DIR/package.json"
MARKETPLACE_JSON="$SKILLS_DIR/.claude-plugin/marketplace.json"

for manifest_file in "$PACKAGE_JSON" "$MARKETPLACE_JSON"; do
    manifest_name="$(basename "$(dirname "$manifest_file")")/$(basename "$manifest_file")"
    if [ ! -f "$manifest_file" ]; then
        print_skip "$manifest_name: not found"
        continue
    fi

    print_section "Checking: $manifest_name"

    if ! command -v python3 &>/dev/null; then
        print_warn "python3 not found, skipping"
        continue
    fi

    manifest_fail=false

    for team in $TEAMS_TO_TEST; do
        [ -z "$team" ] && continue

        # In incremental mode, skip teams not in the changed list
        if is_incremental_mode && ! should_test_team "$team"; then
            continue
        fi

        plugin_json=$(get_team_plugin_json "$team")
        [ -z "$plugin_json" ] || [ ! -f "$plugin_json" ] && continue

        plugin_version=$(extract_plugin_version "$plugin_json")

        manifest_version=$(python3 <<PYEOF
import json
with open("${manifest_file}") as f:
    data = json.load(f)
for p in data.get("plugins", []):
    if p.get("name") == "${team}":
        print(p.get("version", ""))
        break
PYEOF
)

        if [ -z "$manifest_version" ]; then
            print_warn "$team: not found in $manifest_name"
            continue
        fi

        if [ "$plugin_version" = "$manifest_version" ]; then
            print_pass "$team: version $plugin_version matches"
        else
            print_fail "$team: version mismatch — plugin.json=$plugin_version, $manifest_name=$manifest_version"
            manifest_fail=true
        fi
    done

    if $manifest_fail; then
        ((fail_count++)) || true
    else
        ((pass_count++)) || true
    fi
done

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Version Care Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total teams: $total_teams"
echo -e "  ${GREEN}Passed:${NC}   $pass_count"
echo -e "  ${RED}Failed:${NC}   $fail_count"
[ $skip_count -gt 0 ] && echo -e "  ${YELLOW}Skipped:${NC}  $skip_count"
echo ""

if [ $fail_count -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Action: Update plugin.json version and re-run to regenerate state."
    exit 1
else
    print_status_passed
    exit 0
fi
