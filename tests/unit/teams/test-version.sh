#!/usr/bin/env bash
# =============================================================================
# Test: Team Plugin Version Care
# =============================================================================
# Validates that plugin.json version is bumped when skill/agent dependencies change.
# Uses git diff against a base ref (origin/master or HEAD~1) to detect file changes,
# then compares plugin.json version with the base to verify it was bumped.
#
# Rules:
# - PATCH (3rd digit): Skill or agent files changed → bump required (WARN, non-blocking)
# - MINOR / MAJOR: manual upgrade by developer
# - Version decrease: FAIL (blocking)
# - Marketplace version mismatch: WARN (non-blocking)
#
# Environment variables:
#   CI_MERGE_REQUEST_TARGET_BRANCH_NAME  base ref for CI (default: origin/master)
#
# Supports incremental testing via INCREMENTAL_TEAMS environment variable.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

MARKETPLACE_JSON="$SKILLS_DIR/.claude-plugin/marketplace.json"

# Parse arguments
AUTO_FIX=false
for arg in "$@"; do
    case "$arg" in
        --auto-fix) AUTO_FIX=true ;;
    esac
done

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

# Counters
total_teams=0
pass_count=0
fail_count=0
skip_count=0
warn_count=0
fail_decreased=0
warn_unbumped=0
warn_mismatch=0

# Get teams to test (filtered if in incremental mode)
TEAMS_TO_TEST=$(get_teams_to_test)
total_teams=$(echo "$TEAMS_TO_TEST" | grep -c . || echo "0")

echo "Teams to check: $total_teams"
echo ""

# ============================================
# Helper: resolve ALL teams' skill & agent file paths in a single Python call
# Output: TEAM:<team_name> followed by relative paths, one per line
# ============================================
declare -gA TEAM_FILE_PATHS_CACHE=()
_resolve_all_team_file_paths() {
    if [ ${#TEAM_FILE_PATHS_CACHE[@]} -gt 0 ]; then
        return
    fi

    local teams_csv=""
    for team in $TEAMS_TO_TEST; do
        [ -z "$team" ] && continue
        teams_csv+="$team,"
    done
    teams_csv="${teams_csv%,}"

    [ -z "$teams_csv" ] && return

    local result
    result=$(python3 <<PYEOF
import json, os

marketplace_json = "${MARKETPLACE_JSON}"
skills_dir = "${SKILLS_DIR}"
teams = "${teams_csv}".split(",")

try:
    marketplace = json.load(open(marketplace_json))
except Exception:
    marketplace = {"plugins": []}

skills_packages = {}
for p in marketplace.get("plugins", []):
    if p.get("category") == "skills":
        skills_packages[p["name"]] = p

for team_name in teams:
    if not team_name:
        continue
    print(f"TEAM:{team_name}")

    team_entry = None
    for p in marketplace.get("plugins", []):
        if p.get("name") == team_name:
            team_entry = p
            break

    if team_entry:
        for dep_name in team_entry.get("dependencies", []):
            sp = skills_packages.get(dep_name)
            if sp:
                source = sp.get("source", "./ops").lstrip("./")
                for skill_rel in sp.get("skills", []):
                    skill_dir = skill_rel.lstrip("./")
                    print(f"{source}/{skill_dir}/SKILL.md")

    try:
        plugin_json_path = os.path.join(
            skills_dir, "plugins-official", team_name,
            ".claude-plugin", "plugin.json")
        plugin = json.load(open(plugin_json_path))
        for agent_rel in plugin.get("agents", []):
            agent_rel = agent_rel.lstrip("./")
            print(f"plugins-official/{team_name}/{agent_rel}")
    except Exception:
        pass
PYEOF
)

    local current_team=""
    local current_paths=""
    while IFS= read -r line; do
        if [[ "$line" == TEAM:* ]]; then
            if [ -n "$current_team" ]; then
                TEAM_FILE_PATHS_CACHE["$current_team"]="$current_paths"
            fi
            current_team="${line#TEAM:}"
            current_paths=""
        else
            [ -n "$line" ] && current_paths+="$line"$'\n'
        fi
    done <<< "$result"
    if [ -n "$current_team" ]; then
        TEAM_FILE_PATHS_CACHE["$current_team"]="$current_paths"
    fi
}

# Get cached file paths for a team
resolve_team_file_paths() {
    local team_name="$1"
    _resolve_all_team_file_paths
    printf '%s' "${TEAM_FILE_PATHS_CACHE[$team_name]:-}"
}

# ============================================
# Main: Check each team's version via git diff
# ============================================

# Determine base ref for comparison
# Priority: 1) CI env var  2) origin/master  3) HEAD~1
BASE_REF="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-}"
if [ -z "$BASE_REF" ]; then
    if git -C "$SKILLS_DIR" rev-parse --verify "origin/master" &>/dev/null; then
        BASE_REF="origin/master"
    else
        BASE_REF="HEAD~1"
    fi
fi

# Verify the ref actually exists locally
if ! git -C "$SKILLS_DIR" rev-parse --verify "$BASE_REF" &>/dev/null; then
    echo -e "  ${YELLOW}[WARN]${NC} Base ref '$BASE_REF' not found in local git repository"
    echo -e "  ${YELLOW}[WARN]${NC} Version comparison skipped (run 'git fetch origin' or set CI_MERGE_REQUEST_TARGET_BRANCH_NAME)"
    echo ""
    print_section_header "Version Check"
    for team in $TEAMS_TO_TEST; do
        [ -z "$team" ] && continue
        if is_incremental_mode && ! should_test_team "$team"; then
            continue
        fi
        print_skip "$team: base ref unavailable"
        ((skip_count++)) || true
    done
    echo ""
    echo "========================================"
    echo -e " ${BOLD}Version Care Test Summary${NC}"
    echo "========================================"
    echo ""
    echo "  Total teams: $total_teams"
    echo -e "  ${GREEN}Passed:${NC}   $pass_count"
    echo -e "  ${RED}Failed:${NC}   $fail_count"
    [ $skip_count -gt 0 ] && echo -e "  ${YELLOW}Skipped:${NC}  $skip_count"
    echo ""
    print_status_passed
    exit 0
fi

# Safe to diff: three-dot diff against base ref
CHANGED_FILES=$(git -C "$SKILLS_DIR" diff --name-only "${BASE_REF}...HEAD" 2>/dev/null || true)

# Warn about uncommitted changes (git diff only sees committed state)
UNCOMMITTED=$(git -C "$SKILLS_DIR" diff --name-only 2>/dev/null || true)
if [ -n "$UNCOMMITTED" ]; then
    echo -e "  ${YELLOW}[WARN]${NC} Uncommitted changes detected — only committed changes are checked"
fi

echo "Comparing against: ${BASE_REF}"
echo "Changed files: $(echo "$CHANGED_FILES" | grep -c . || echo 0)"
echo ""

# Detect fork sync status for helpful diagnostics
if git -C "$SKILLS_DIR" rev-parse --verify "upstream/master" &>/dev/null; then
    if [ "$BASE_REF" = "origin/master" ]; then
        origin_upstream_diff=$(git -C "$SKILLS_DIR" log --oneline "upstream/master..origin/master" 2>/dev/null | wc -l || echo 0)
        if [ "$origin_upstream_diff" -gt 0 ]; then
            echo -e "  ${CYAN}[INFO]${NC} upstream/master detected and diverges from origin/master."
            echo -e "  ${CYAN}[INFO]${NC} If local tests pass but CI fails, ensure you have rebased to upstream."
            echo ""
        fi
    fi
fi

print_section_header "Version Check"

for team in $TEAMS_TO_TEST; do
    [ -z "$team" ] && continue

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

    # Get base version from the reference
    base_version=$(git -C "$SKILLS_DIR" show "${BASE_REF}:plugins-official/${team}/.claude-plugin/plugin.json" 2>/dev/null | grep '"version"' | sed 's/.*"version":[[:space:]]*"\([^"]*\)".*/\1/' || true)

    if [ -z "$base_version" ]; then
        print_info "No base version found (new team?), skipping"
        ((pass_count++)) || true
        echo ""
        continue
    fi

    # Check which team files changed against BASE_REF
    team_files=$(resolve_team_file_paths "$team")
    changed_items=""
    has_change=false

    while IFS= read -r rel; do
        [ -z "$rel" ] && continue
        if echo "$CHANGED_FILES" | grep -qF "$rel"; then
            has_change=true
            if echo "$rel" | grep -q '/SKILL.md$'; then
                changed_items+="  - SKILL: $rel"$'\n'
            else
                changed_items+="  - AGENT: $rel"$'\n'
            fi
        fi
    done <<< "$team_files"

    if $has_change; then
        if [ "$current_version" = "$base_version" ]; then
            if $AUTO_FIX; then
                new_version=$(recommend_version_bump "$current_version" true true)
                bump_plugin_json "$team" "$new_version"
                bump_marketplace_json "$team" "$new_version"
                print_pass "Version auto-bumped: $base_version → $new_version"
                echo -e "$changed_items" | sed 's/^/    /'
                ((pass_count++)) || true
            else
                new_version=$(recommend_version_bump "$current_version" true true)
                print_warn "Files changed but version not bumped (base: $base_version, current: $current_version)"
                echo -e "$changed_items" | sed 's/^/    /'
                echo ""
                echo -e "  ${CYAN}Suggested fix:${NC} bump PATCH to $new_version"
                echo -e "  ${CYAN}Quick fix:${NC}   tests/run-tests.sh --fast --auto-fix"
                ((warn_count++)) || true
                ((warn_unbumped++)) || true
            fi
        else
            cmp=$(semver_compare "$current_version" "$base_version")
            if [ "$cmp" = "1" ] || [ "$cmp" = "0" ]; then
                print_pass "Version bumped: $base_version → $current_version"
                echo -e "$changed_items" | sed 's/^/    /'
                ((pass_count++)) || true
            else
                suggested=$(recommend_version_bump "$base_version" true true)
                print_fail "Version decreased: $base_version → $current_version"
                echo -e "  ${CYAN}Fix:${NC} Version must be >= base ($base_version). Suggested: $suggested"
                ((fail_count++)) || true
                ((fail_decreased++)) || true
            fi
        fi
    else
        if [ "$current_version" != "$base_version" ]; then
            print_pass "Version changed: $base_version → $current_version (no file changes in this team)"
        else
            print_pass "Version $current_version consistent (no changes detected)"
        fi
        ((pass_count++)) || true
    fi
    echo ""
done

# ============================================
# Marketplace Version Consistency Check
# ============================================
print_section_header "Marketplace Version Consistency"

# package.json (OpenCode) and marketplace.json (Claude) are what users see
# when browsing the marketplace. If these don't match plugin.json, users
# won't see the updated version and won't know to upgrade.

for manifest_file in "$SKILLS_DIR/package.json" "$MARKETPLACE_JSON"; do
    if [ ! -f "$manifest_file" ]; then
        continue
    fi
    manifest_name="$(basename "$(dirname "$manifest_file")")/$(basename "$manifest_file")"

    print_section "Checking: $manifest_name"

    if ! command -v python3 &>/dev/null; then
        print_warn "python3 not found, skipping"
        continue
    fi

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
            ((pass_count++)) || true
        else
            print_warn "$team: version mismatch — plugin.json=$plugin_version, $manifest_name=$manifest_version"
            echo -e "  ${CYAN}Fix:${NC} Update $manifest_name to match plugin.json ($plugin_version)"
            ((warn_count++)) || true
            ((warn_mismatch++)) || true
        fi
    done
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
echo -e "  ${YELLOW}Warnings:${NC} $warn_count"
[ $skip_count -gt 0 ] && echo -e "  ${YELLOW}Skipped:${NC}  $skip_count"
echo ""

if [ $fail_decreased -gt 0 ]; then
    print_status_failed
    echo ""
    echo "========================================"
    echo -e " ${BOLD}Recommended Actions${NC}"
    echo "========================================"
    echo ""

    echo -e "${RED}1. Version decreased:${NC}"
    echo "   Version can only increase. Restore or bump higher."
    echo ""

    # fork sync hint
    if git -C "$SKILLS_DIR" rev-parse --verify "upstream/master" &>/dev/null; then
        origin_upstream_diff=$(git -C "$SKILLS_DIR" log --oneline "upstream/master..origin/master" 2>/dev/null | wc -l || echo 0)
        if [ "$origin_upstream_diff" -gt 0 ]; then
            echo -e "${YELLOW}[FORK SYNC]${NC} origin/master diverges from upstream/master."
            echo "            CI may use upstream/master as base. Rebase and push first:"
            echo "            git fetch upstream && git rebase upstream/master && git push --force-with-lease"
            echo ""
        fi
    fi

    exit 1
elif [ $warn_count -gt 0 ]; then
    echo -e "${YELLOW}Warnings (non-blocking):${NC}"
    if [ "$warn_unbumped" -gt 0 ]; then
        echo -e "  - $warn_unbumped team(s) have file changes but version not bumped"
        echo -e "    Run: ${CYAN}tests/run-tests.sh --fast --auto-fix${NC}"
    fi
    if [ "$warn_mismatch" -gt 0 ]; then
        echo -e "  - $warn_mismatch team(s) have marketplace version mismatch"
        echo -e "    Update marketplace.json to match plugin.json"
    fi
    echo ""
    print_status_passed
    exit 0
else
    print_status_passed
    exit 0
fi
