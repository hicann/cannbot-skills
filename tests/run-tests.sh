#!/usr/bin/env bash
# =============================================================================
# CANN Skills Test Runner v0.2
# =============================================================================
# Unified test runner supporting Claude Code and OpenCode platforms.
#
# Usage:
#   ./run-tests.sh                    # Run all fast tests
#   ./run-tests.sh --integration      # Run integration tests
#   ./run-tests.sh --platform claude  # Run only Claude tests
#   ./run-tests.sh --test test-name   # Run specific test
#   ./run-tests.sh --output json      # JSON output for CI
#   ./run-tests.sh --incremental      # Only test changed skills/agents/teams
#
# CI/CD Integration:
#   This script supports incremental testing for PR workflows.
#   Use --incremental to only test changed components.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source libraries
source "$SCRIPT_DIR/lib/test-helpers.sh"

# =============================================================================
# Configuration
# =============================================================================

RUN_INTEGRATION=false
RUN_FAST=false
RUN_ALL=false
RUN_EVAL_RESULTS=false
PLATFORM="opencode"
OUTPUT_FORMAT="text"
VERBOSE=true
TIMEOUT=300
SPECIFIC_TEST=""
CATEGORY=""
TEST_RESULTS=()

# Incremental testing options
INCREMENTAL_MODE=false
BASE_BRANCH="${BASE_BRANCH:-master}"
FORCE_FULL_TEST=false

# Eval-specific options
EVAL_WORKSPACE=""
EVAL_ITERATION=""
EVAL_THRESHOLD=""
EVAL_DETECT_REGRESSION=false
EVAL_INCREMENTAL=false
EVAL_BASE_BRANCH="master"

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
    cat <<EOF
CANN Skills Test Runner v0.2

Usage: $0 [OPTIONS]

Options:
  -h, --help           Show this help message
  --fast               Run only fast tests (no CLI required)
  --integration        Run integration tests (may take several minutes)
  --all                Run all tests including integration
  --platform PLATFORM  Specify platform: claude, opencode (default: opencode)
  --test TEST          Run specific test file
  --category CAT       Run tests in specific category
  --output FORMAT      Output format: text, json (default: text)
  --timeout SECONDS    Test timeout (default: 300)
  --verbose            Enable verbose output
  --list               List available tests

Incremental Testing Options (for CI/CD):
  --incremental        Only test changed skills/agents/teams
  --base-branch BRANCH Base branch for comparison (default: master)
  --force-full         Force full test run even in incremental mode

Skill Evaluation Options:
  --eval-results       Run skill evaluation results check (workspace benchmark validation)
  --workspace PATH     Specify a specific workspace for eval results check
  --iteration N        Specify iteration version (default: latest)
  --threshold RATE     Override pass rate threshold (0.0-1.0)
  --detect-regression  Enable regression detection between iterations

Test Categories:
  unit          - Unit tests (structure, dependencies, content)
  behavior      - Behavior tests (requires CLI)
  integration   - End-to-end integration tests (use --integration)
  all           - Run all test categories

Examples:
  $0                              # Run unit tests (L1)
  $0 --fast                       # Unit tests only (no CLI needed)
  $0 --integration                # Run all tests including L3
  $0 --category behavior          # Run behavior tests
  $0 --test unit/skills/test-structure.sh
  $0 --output json                # JSON output
  $0 --incremental                # Only test changed components (CI/CD)
  $0 --incremental --base-branch develop
  $0 --eval-results               # Check skill evaluation results
  $0 --eval-results --workspace ../skills/ascendc-stc-design-workspace
  $0 --eval-results --threshold 0.9 --detect-regression

EOF
    exit 0
}

list_tests() {
    echo "========================================"
    echo " Available Tests"
    echo "========================================"
    echo ""
    
    echo "L1 Unit Tests - Repo Hygiene:"
    for f in "$SCRIPT_DIR"/unit/test-*.sh; do
        [ -f "$f" ] && echo "  unit/$(basename "$f")"
    done
    echo ""

    echo "L1 Unit Tests - Skills:"
    for f in "$SCRIPT_DIR"/unit/skills/test-*.sh; do
        [ -f "$f" ] && echo "  unit/skills/$(basename "$f")"
    done
    echo ""
    
    echo "L1 Unit Tests - Agents:"
    for f in "$SCRIPT_DIR"/unit/agents/test-*.sh; do
        [ -f "$f" ] && echo "  unit/agents/$(basename "$f")"
    done
    echo ""
    
    echo "L1 Unit Tests - Teams:"
    for f in "$SCRIPT_DIR"/unit/teams/test-*.sh; do
        [ -f "$f" ] && echo "  unit/teams/$(basename "$f")"
    done
    echo ""

    echo "L1 Unit Tests - Install:"
    for f in "$SCRIPT_DIR"/unit/install/test-*.sh; do
        [ -f "$f" ] && echo "  unit/install/$(basename "$f")"
    done
    echo ""

    echo "L2 Behavior Tests - Skills:"
    for f in "$SCRIPT_DIR"/behavior/skills/test-*.sh; do
        [ -f "$f" ] && echo "  behavior/skills/$(basename "$f")"
    done
    echo ""
    
    echo "L2 Behavior Tests - Agents:"
    for f in "$SCRIPT_DIR"/behavior/agents/test-*.sh; do
        [ -f "$f" ] && echo "  behavior/agents/$(basename "$f")"
    done
    echo ""

    echo "L2 Behavior Tests - Install:"
    for f in "$SCRIPT_DIR"/behavior/install/test-*.sh; do
        [ -f "$f" ] && echo "  behavior/install/$(basename "$f")"
    done
    echo ""

    echo "L3 Integration Tests:"
    for f in "$SCRIPT_DIR"/integration/test-*.sh; do
        [ -f "$f" ] && echo "  integration/$(basename "$f")"
    done
    echo ""
}

parse_args() {
    local has_mode_flag=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                ;;
            --fast|-f)
                RUN_FAST=true
                has_mode_flag=true
                shift
                ;;
            --integration|-i)
                RUN_INTEGRATION=true
                has_mode_flag=true
                shift
                ;;
            --all)
                RUN_ALL=true
                has_mode_flag=true
                shift
                ;;
            --eval-results)
                RUN_EVAL_RESULTS=true
                has_mode_flag=true
                shift
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --test|-t)
                SPECIFIC_TEST="$2"
                has_mode_flag=true
                shift 2
                ;;
            --category|-c)
                CATEGORY="$2"
                has_mode_flag=true
                shift 2
                ;;
            --output)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --list|-l)
                list_tests
                exit 0
                ;;
            --workspace)
                EVAL_WORKSPACE="$2"
                shift 2
                ;;
            --iteration)
                EVAL_ITERATION="$2"
                shift 2
                ;;
            --threshold)
                EVAL_THRESHOLD="$2"
                shift 2
                ;;
            --detect-regression)
                EVAL_DETECT_REGRESSION=true
                shift
                ;;
            --incremental)
                EVAL_INCREMENTAL=true
                shift
                ;;
            --base-branch)
                EVAL_BASE_BRANCH="$2"
                BASE_BRANCH="$2"
                shift 2
                ;;
            --incremental-ci)
                INCREMENTAL_MODE=true
                has_mode_flag=true
                shift
                ;;
            --incremental)
                # Alias for --incremental-ci
                INCREMENTAL_MODE=true
                has_mode_flag=true
                shift
                ;;
            --force-full)
                FORCE_FULL_TEST=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Default to --fast if no mode flag specified
    if ! $has_mode_flag && [ -z "$SPECIFIC_TEST" ]; then
        RUN_FAST=true
    fi

    # If category is integration, auto-enable RUN_INTEGRATION
    if [ "$CATEGORY" == "integration" ]; then
        RUN_INTEGRATION=true
    fi

    if [[ "$PLATFORM" == "auto" ]]; then
        if is_platform_available "claude"; then
            PLATFORM="claude"
        elif is_platform_available "opencode"; then
            PLATFORM="opencode"
        else
            echo -e "${YELLOW}[WARN]${NC} No AI CLI found - will run fast tests only"
            RUN_FAST=true
            PLATFORM="none"
        fi
    fi
}

# =============================================================================
# Incremental Testing Functions
# =============================================================================

# Check if we're in a git repository
is_git_repo() {
    git -C "$SKILLS_DIR" rev-parse --is-inside-work-tree &>/dev/null
}

# Get list of changed files compared to base branch
get_changed_files() {
    local base="${1:-master}"

    # Try to find the merge base
    if ! git -C "$SKILLS_DIR" rev-parse --verify "$base" &>/dev/null; then
        echo ""
        return 1
    fi

    # Get changed files (added, modified, renamed, deleted)
    git -C "$SKILLS_DIR" diff --name-status "$base"...HEAD 2>/dev/null || \
    git -C "$SKILLS_DIR" diff --name-status HEAD~1 HEAD 2>/dev/null || \
    echo ""
}

# Parse changed files and extract affected skills, agents, teams
# Returns: changed_skills, changed_agents, changed_teams (global arrays)
declare -gA CHANGED_SKILLS=()
declare -gA CHANGED_AGENTS=()
declare -gA CHANGED_TEAMS=()
declare -g FRAMEWORK_CHANGED=false
declare -g GLOBAL_CONFIG_CHANGED=false

analyze_changes() {
    local base_branch="${1:-master}"
    CHANGED_SKILLS=()
    CHANGED_AGENTS=()
    CHANGED_TEAMS=()
    FRAMEWORK_CHANGED=false
    GLOBAL_CONFIG_CHANGED=false

    if ! is_git_repo; then
        echo -e "${YELLOW}[WARN]${NC} Not a git repository, running full tests"
        return 1
    fi

    local changed_files
    changed_files=$(get_changed_files "$base_branch")

    if [ -z "$changed_files" ]; then
        echo -e "${YELLOW}[WARN]${NC} Could not detect changes, running full tests"
        return 1
    fi

    echo -e "${CYAN}=== Incremental Test Analysis ===${NC}"
    echo ""
    echo "Base branch: $base_branch"
    echo ""

    # Framework patterns - changes here require full test
    local framework_patterns=(
        "^tests/"
        "^tests/lib/"
        "\.sh$"
        "package\.json$"
        "\.claude-plugin/"
        "\.opencode/"
    )

    # Global config patterns
    local global_config_patterns=(
        "^README\.md$"
        "^CLAUDE\.md$"
        "^AGENTS\.md$"
        "^\.git"
    )

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        local status="${line:0:1}"
        local file="${line:1}"
        file="${file#"${file%%[![:space:]]*}"}"  # Trim leading whitespace

        # Skip deleted files for component detection (but track for framework)
        if [ "$status" == "D" ]; then
            continue
        fi

        # Check for framework changes
        for pattern in "${framework_patterns[@]}"; do
            if echo "$file" | grep -qE "$pattern"; then
                FRAMEWORK_CHANGED=true
                echo -e "  ${YELLOW}[FRAMEWORK]${NC} $file"
                break
            fi
        done

        # Check for global config changes
        for pattern in "${global_config_patterns[@]}"; do
            if echo "$file" | grep -qE "$pattern"; then
                GLOBAL_CONFIG_CHANGED=true
                break
            fi
        done

        # Detect skill changes
        # Patterns: skills/<name>/, ops/skills/<name>/, ops-lab/*/skills/<name>/, model/skills/<name>/
        if [[ "$file" =~ (skills/([^/]+)/) ]] || \
           [[ "$file" =~ /skills/([^/]+)/SKILL\.md$ ]] || \
           [[ "$file" =~ /skills/([^/]+)/references/ ]]; then
            local skill_name="${BASH_REMATCH[2]:-${BASH_REMATCH[1]}}"
            if [ -n "$skill_name" ]; then
                CHANGED_SKILLS["$skill_name"]=1
                echo -e "  ${GREEN}[SKILL]${NC} $skill_name <- $file"
            fi
        fi

        # Detect agent changes
        # Patterns: agents/<name>/, ops/agents/<name>/, ops-lab/*/agent/
        if [[ "$file" =~ /agents/([^/]+)/AGENT\.md$ ]] || \
           [[ "$file" =~ /agents/([^/]+)/ ]]; then
            local agent_name="${BASH_REMATCH[1]}"
            if [ -n "$agent_name" ]; then
                CHANGED_AGENTS["$agent_name"]=1
                echo -e "  ${BLUE}[AGENT]${NC} $agent_name <- $file"
            fi
        fi

        # Detect team changes
        # Patterns: teams/<name>/, ops/teams/<name>/
        if [[ "$file" =~ /teams/([^/]+)/AGENTS\.md$ ]] || \
           [[ "$file" =~ /teams/([^/]+)/ ]]; then
            local team_name="${BASH_REMATCH[1]}"
            if [ -n "$team_name" ]; then
                CHANGED_TEAMS["$team_name"]=1
                echo -e "  ${CYAN}[TEAM]${NC} $team_name <- $file"
            fi
        fi

    done <<< "$changed_files"

    echo ""
}

# Check if incremental testing should be used
should_run_incremental() {
    if ! $INCREMENTAL_MODE; then
        return 1
    fi

    if $FORCE_FULL_TEST; then
        echo -e "${YELLOW}[INFO]${NC} Force full test requested"
        return 1
    fi

    if $FRAMEWORK_CHANGED; then
        echo -e "${YELLOW}[INFO]${NC} Test framework changed, running full tests"
        return 1
    fi

    if $GLOBAL_CONFIG_CHANGED; then
        echo -e "${YELLOW}[INFO]${NC} Global config changed, running full tests"
        return 1
    fi

    # Check if there are any changes
    if [ ${#CHANGED_SKILLS[@]} -eq 0 ] && [ ${#CHANGED_AGENTS[@]} -eq 0 ] && [ ${#CHANGED_TEAMS[@]} -eq 0 ]; then
        echo -e "${GREEN}[INFO]${NC} No skill/agent/team changes detected"
        return 1
    fi

    return 0
}

# Generate test list for changed components
get_incremental_tests() {
    local tests=""

    # Always run structure and content tests for changed skills
    if [ ${#CHANGED_SKILLS[@]} -gt 0 ]; then
        tests+="unit/skills/test-structure.sh:fast\n"
        tests+="unit/skills/test-content.sh:fast\n"
    fi

    # Always run structure and content tests for changed agents
    if [ ${#CHANGED_AGENTS[@]} -gt 0 ]; then
        tests+="unit/agents/test-structure.sh:fast\n"
        tests+="unit/agents/test-content.sh:fast\n"
    fi

    # Run all team tests for changed teams (including version check)
    if [ ${#CHANGED_TEAMS[@]} -gt 0 ]; then
        tests+="unit/teams/test-structure.sh:fast\n"
        tests+="unit/teams/test-content.sh:fast\n"
        tests+="unit/teams/test-version.sh:fast\n"
    fi

    echo -e "$tests" | grep -v '^$' || true
}

# Export changed component lists for test scripts to use
export_changed_components() {
    local skills_list=""
    local agents_list=""
    local teams_list=""

    for skill in "${!CHANGED_SKILLS[@]}"; do
        skills_list="$skills_list $skill"
    done

    for agent in "${!CHANGED_AGENTS[@]}"; do
        agents_list="$agents_list $agent"
    done

    for team in "${!CHANGED_TEAMS[@]}"; do
        teams_list="$teams_list $team"
    done

    export INCREMENTAL_SKILLS="${skills_list# }"
    export INCREMENTAL_AGENTS="${agents_list# }"
    export INCREMENTAL_TEAMS="${teams_list# }"
}

# =============================================================================
# Test Definitions
# =============================================================================

get_tests_for_category() {
    local cat="$1"

    case "$cat" in
        unit)
            echo "unit/test-line-endings.sh:fast"
            echo "unit/skills/test-structure.sh:fast"
            echo "unit/skills/test-content.sh:fast"
            echo "unit/agents/test-structure.sh:fast"
            echo "unit/agents/test-content.sh:fast"
            echo "unit/teams/test-structure.sh:fast"
            echo "unit/teams/test-content.sh:fast"
            echo "unit/teams/test-version.sh:fast"
            echo "unit/install/test-init-install.sh:fast"
            ;;
        behavior)
            # test-universal.sh contains all 9 behavior rules (B-TRIG, B-SAFE, B-INTA, B-BND)
            # Other test files can be run individually via --test flag
            echo "behavior/skills/test-universal.sh:medium"
            echo "behavior/install/test-init-behavior.sh:fast"
            ;;
        integration)
            for f in "$SCRIPT_DIR"/integration/test-*.sh; do
                [ -f "$f" ] && echo "integration/$(basename "$f"):slow"
            done
            ;;
        all)
            get_tests_for_category "unit"
            if ! $RUN_FAST; then
                get_tests_for_category "behavior"
            else
                # In fast mode, still run behavior tests marked as :fast
                get_tests_for_category "behavior" | grep ':fast$' || true
            fi
            if $RUN_INTEGRATION || $RUN_ALL; then
                get_tests_for_category "integration"
            fi
            ;;
        *)
            get_tests_for_category "all"
            ;;
    esac
}

# =============================================================================
# Test Execution
# =============================================================================

run_test_file() {
    local test_file="$1"
    local speed="$2"
    local test_path="$SCRIPT_DIR/$test_file"
    local start_time=$(date +%s)
    local status="pass"
    local output=""
    local warning_count=0

    if [[ ! -f "$test_path" ]]; then
        echo "  [SKIP] Test file not found: $test_file"
        TEST_RESULTS+=("skip:$test_file:0:0")
        return 0
    fi

    print_section "Running: $test_file"

    if $VERBOSE; then
        if timeout $TIMEOUT bash "$test_path"; then
            status="pass"
        else
            status="fail"
        fi
    else
        if output=$(timeout $TIMEOUT bash "$test_path" 2>&1); then
            status="pass"
        else
            status="fail"
        fi
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Count warnings in output
    if [[ -n "$output" ]]; then
        warning_count=$(echo "$output" | grep -cE "\[WARN\]" 2>/dev/null || true)
        # Ensure it's a valid number
        [[ "$warning_count" =~ ^[0-9]+$ ]] || warning_count=0
    fi

    case "$status" in
        pass)
            print_pass "(${duration}s)"
            # Show warnings if present in output
            if [[ "$warning_count" -gt 0 ]]; then
                echo ""
                echo "$output" | grep -E "\[WARN\]" | sed 's/^/    /'
                echo ""
            fi
            TEST_RESULTS+=("pass:$test_file:$duration:$warning_count")
            record_test "pass" "$test_file" "$duration"
            ;;
        fail)
            print_fail "(${duration}s)"
            if [[ -n "$output" ]]; then
                echo ""
                echo -e "  ${YELLOW}--- Failure Details ---${NC}"
                echo "$output" | sed 's/^/    /'
                echo -e "  ${YELLOW}--- End ---${NC}"
                echo ""
            else
                echo "  (run with --verbose for more details)"
            fi
            TEST_RESULTS+=("fail:$test_file:$duration:$warning_count")
            record_test "fail" "$test_file" "$duration"
            ;;
    esac

    [ "$status" == "pass" ]
}

run_all_tests() {
    local total_failed=0
    local tests_run=0

    # Check for incremental mode
    local use_incremental=false
    if $INCREMENTAL_MODE; then
        analyze_changes "$BASE_BRANCH"
        if should_run_incremental; then
            use_incremental=true
            export_changed_components
        fi
    fi

    local test_banner_info="Repository: $SKILLS_DIR
Test time: $(date '+%Y-%m-%d %H:%M:%S')
Platform: $PLATFORM"

    if $use_incremental; then
        test_banner_info+="
Mode: INCREMENTAL (changed components only)"
    fi

    print_test_banner "CANN Skills Test Suite v0.2" "$test_banner_info"

    echo ""
    echo "Platform versions:"
    case "$PLATFORM" in
        claude)
            echo "  Claude Code: $(get_platform_version claude)"
            ;;
        opencode)
            echo "  OpenCode: $(get_platform_version opencode)"
            ;;
        none)
            echo "  (no CLI - fast tests only)"
            ;;
    esac
    echo ""

    local tests
    if $use_incremental; then
        tests=$(get_incremental_tests)
        if [ -z "$tests" ]; then
            echo -e "${GREEN}No tests to run (no relevant changes detected)${NC}"
            print_summary 0
            return 0
        fi
    elif [[ -n "$CATEGORY" ]]; then
        tests=$(get_tests_for_category "$CATEGORY")
    else
        tests=$(get_tests_for_category "all")
    fi

    local test_count=$(echo "$tests" | grep -c ':' || echo "0")

    if $use_incremental; then
        echo "Changed components:"
        [ ${#CHANGED_SKILLS[@]} -gt 0 ] && echo "  Skills: ${!CHANGED_SKILLS[*]}"
        [ ${#CHANGED_AGENTS[@]} -gt 0 ] && echo "  Agents: ${!CHANGED_AGENTS[*]}"
        [ ${#CHANGED_TEAMS[@]} -gt 0 ] && echo "  Teams: ${!CHANGED_TEAMS[*]}"
        echo ""
    fi

    echo "Tests to run: $test_count"
    echo ""

    local test_array=()
    while IFS=':' read -r test_file speed; do
        [[ -n "$test_file" ]] && test_array+=("$test_file:$speed")
    done <<< "$tests"

    for test_entry in "${test_array[@]}"; do
        IFS=':' read -r test_file speed <<< "$test_entry"

        if [[ "$speed" == "slow" ]] && ! $RUN_INTEGRATION && ! $RUN_ALL; then
            print_skip "$test_file (slow test, use --integration)"
            continue
        fi

        if [[ "$speed" != "fast" ]] && ($RUN_FAST || [[ "$PLATFORM" == "none" ]]); then
            print_skip "$test_file (requires CLI)"
            continue
        fi

        tests_run=$((tests_run + 1))
        if ! run_test_file "$test_file" "$speed"; then
            total_failed=$((total_failed + 1))
        fi
    done

    print_summary $tests_run
    return $total_failed
}

run_specific_test() {
    local test_name="$SPECIFIC_TEST"
    local test_path=""

    if [[ "$test_name" != */* ]]; then
        for dir in unit behavior integration; do
            if [[ -f "$SCRIPT_DIR/$dir/$test_name" ]]; then
                test_path="$dir/$test_name"
                break
            fi
        done
        if [[ -z "$test_path" ]] && [[ -f "$SCRIPT_DIR/$test_name" ]]; then
            test_path="$test_name"
        fi
    else
        test_path="$test_name"
    fi

    if [[ -z "$test_path" ]]; then
        echo "[ERROR] Test not found: $test_name"
        exit 1
    fi

    print_test_banner "Single Test: $test_path" "
Platform: $PLATFORM
Repository: $SKILLS_DIR
"

    run_test_file "$test_path" "medium"
}

print_summary() {
    local tests_run="${1:-0}"
    local passed=0
    local failed=0
    local skipped=0
    local warnings=0
    local total_duration=0

    for result in "${TEST_RESULTS[@]}"; do
        IFS=':' read -r status file duration warn_count <<< "$result"
        case "$status" in
            pass) ((passed++)) || true ;;
            fail) ((failed++)) || true ;;
            skip) ((skipped++)) || true ;;
        esac
        # Add warning count (field 4)
        if [[ -n "$warn_count" ]] && [[ "$warn_count" =~ ^[0-9]+$ ]]; then
            warnings=$((warnings + warn_count))
        fi
        total_duration=$((total_duration + duration))
    done

    echo ""
    echo "========================================"
    echo -e " ${BOLD}Test Results Summary${NC}"
    echo "========================================"
    echo ""
    echo "  Tests run: $tests_run"
    echo -e "  ${GREEN}Passed:${NC}    $passed"
    echo -e "  ${RED}Failed:${NC}    $failed"
    echo -e "  ${YELLOW}Skipped:${NC}   $skipped"
    echo -e "  ${YELLOW}Warnings:${NC} $warnings"
    echo "  Duration:  ${total_duration}s"
    echo ""

    if $RUN_FAST; then
        echo "Note: Only fast tests were run (--fast flag)."
        echo ""
    fi

    if ! $RUN_INTEGRATION && ! $RUN_ALL && [ -d "$SCRIPT_DIR/integration" ]; then
        local integration_count=$(find "$SCRIPT_DIR/integration" -name "test-*.sh" -type f 2>/dev/null | wc -l)
        if [ "$integration_count" -gt 0 ]; then
            echo "Note: Integration tests were not run."
            echo "Use --integration flag to run them."
            echo ""
        fi
    fi

    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        output_json "$passed" "$failed" "$skipped" "$warnings" "$total_duration"
    fi

    if [[ $failed -gt 0 ]]; then
        print_status_failed
        return 1
    else
        print_status_passed
        return 0
    fi
}

output_json() {
    local passed="$1"
    local failed="$2"
    local skipped="$3"
    local warnings="$4"
    local duration="$5"

    local test_json="["
    local first=true
    for result in "${TEST_RESULTS[@]}"; do
        IFS=':' read -r status file dur warn_cnt <<< "$result"
        if $first; then
            first=false
        else
            test_json+=","
        fi
        test_json+="{"\"name\"": "\"$file\"", "\"status\"": "\"$status\"", "\"duration\"": $dur, "\"warnings\"": ${warn_cnt:-0}}"
    done
    test_json+="]"

    cat <<EOF
{
  "status": "$([ "$failed" -gt 0 ] && echo "failed" || echo "passed")",
  "passed": $passed,
  "failed": $failed,
  "skipped": $skipped,
  "warnings": $warnings,
  "duration": $duration,
  "timestamp": "$(date -Iseconds)",
  "platform": "$PLATFORM",
  "tests": $test_json
}
EOF
}

# =============================================================================
# Skill Evaluation Results Test
# =============================================================================

run_eval_results() {
    local eval_test="$SCRIPT_DIR/integration/test-skill-eval-results.sh"

    if [ ! -f "$eval_test" ]; then
        echo "[ERROR] Eval results test script not found: $eval_test"
        return 1
    fi

    print_test_banner "Skill Evaluation Results Check" "
Repository: $SKILLS_DIR
Test time: $(date '+%Y-%m-%d %H:%M:%S')"

    # Build command with optional parameters
    local cmd="bash \"$eval_test\""

    if [ -n "$EVAL_WORKSPACE" ]; then
        cmd="$cmd --workspace \"$EVAL_WORKSPACE\""
    fi

    if [ -n "$EVAL_ITERATION" ]; then
        cmd="$cmd --iteration $EVAL_ITERATION"
    fi

    if [ -n "$EVAL_THRESHOLD" ]; then
        cmd="$cmd --threshold $EVAL_THRESHOLD"
    fi

    if $EVAL_DETECT_REGRESSION; then
        cmd="$cmd --detect-regression"
    fi

    if $EVAL_INCREMENTAL; then
        cmd="$cmd --incremental --base-branch $EVAL_BASE_BRANCH"
    fi

    if $VERBOSE; then
        cmd="$cmd --verbose"
    fi

    local start_time=$(date +%s)
    local status="pass"
    local output=""

    print_section "Running: test-skill-eval-results.sh"

    if output=$(eval "$cmd" 2>&1); then
        status="pass"
    else
        status="fail"
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Print output
    echo "$output"

    case "$status" in
        pass)
            echo ""
            print_pass "(${duration}s)"
            ;;
        fail)
            echo ""
            print_fail "(${duration}s)"
            ;;
    esac

    [ "$status" == "pass" ]
}

# =============================================================================
# Main
# =============================================================================

master() {
    parse_args "$@"
    init_test_tracking

    case "$PLATFORM" in
        claude)
            if ! is_platform_available "claude"; then
                echo "[ERROR] Claude Code CLI not found"
                exit 1
            fi
            ;;
        opencode)
            if ! is_platform_available "opencode"; then
                echo "[ERROR] OpenCode CLI not found"
                echo "[DEBUG] Environment information:"
                echo "  - Node.js version: $(node --version 2>/dev/null || echo 'not installed')"
                echo "  - npm version: $(npm --version 2>/dev/null || echo 'not installed')"
                echo "  - opencode version: $(opencode --version 2>/dev/null || echo 'not installed')"
                echo "  - which node: $(which node 2>/dev/null || echo 'not found')"
                echo "  - which npm: $(which npm 2>/dev/null || echo 'not found')"
                echo "  - which opencode: $(which opencode 2>/dev/null || echo 'not found')"
                echo "  - PATH: $PATH"
                exit 1
            fi
            ;;
    esac

    local exit_code=0

    # Handle --eval-results mode
    if $RUN_EVAL_RESULTS; then
        run_eval_results || exit_code=$?
        exit $exit_code
    fi

    if [[ -n "$SPECIFIC_TEST" ]]; then
        run_specific_test || exit_code=$?
    else
        run_all_tests || exit_code=$?
    fi

    exit $exit_code
}

master "$@"