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
HTML_OUTPUT_PATH=""
VERBOSE=true
TIMEOUT=600
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

# Auto-fix option (applies to all tests that support --fix)
AUTO_FIX=false

# Parallel execution
PARALLEL_JOBS=1

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
  --output FORMAT      Output format: text, json, html (default: text)
  --output-path PATH   Write html report to path (default: tests/test-ut-report.html)
  --timeout SECONDS    Test timeout (default: 600)
  --verbose            Enable verbose output
  --list               List available tests

Incremental Testing Options (for CI/CD):
  --incremental        Only test changed skills/agents/teams
  --base-branch BRANCH Base branch for comparison (default: master)
  --force-full         Force full test run even in incremental mode
  --auto-fix           Auto-fix issues in tests that support it (CRLF, version bump, etc.)

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
  $0 --output html                # HTML report for local debugging
  $0 --output html --output-path report.html
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
            --output-path)
                HTML_OUTPUT_PATH="$2"
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
                INCREMENTAL_MODE=true
                has_mode_flag=true
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
            --force-full)
                FORCE_FULL_TEST=true
                shift
                ;;
            --auto-fix)
                AUTO_FIX=true
                shift
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
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
        "^tests/.*\.sh$"
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
        # Patterns: agents/<name>.md (flat layout)
        if [[ "$file" =~ /agents/([^/.]+)\.md$ ]] && [[ ! "$file" =~ /AGENTS\.md$ ]]; then
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
            echo "unit/test-dependency-graph.sh:fast"
            echo "unit/skills/test-structure.sh:fast"
            echo "unit/skills/test-content.sh:fast"
            echo "unit/agents/test-structure.sh:fast"
            echo "unit/agents/test-content.sh:fast"
            echo "unit/teams/test-structure.sh:fast"
            echo "unit/teams/test-content.sh:fast"
            echo "unit/teams/test-version.sh:fast"
            echo "unit/install/test-init-install.sh:fast"
            echo "unit/install/test-safe-install.sh:fast"
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
        TEST_RESULTS+=("skip:$test_file:0:0:")
        return 0
    fi

    print_section "Running: $test_file"

    # Always capture output silently; compact summary printed after.
    local test_outfile=$(mktemp)
    trap 'rm -f "$test_outfile"' RETURN
    local exit_code=0
    local extra_args=()
    if $AUTO_FIX; then
        extra_args+=("--auto-fix")
    fi
    timeout $TIMEOUT bash "$test_path" "${extra_args[@]}" > "$test_outfile" 2>&1 || exit_code=$?
    output=$(cat "$test_outfile")
    rm -f "$test_outfile"
    trap - RETURN

    if [[ $exit_code -ne 0 ]]; then
        status="fail"
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Count warnings in output
    if [[ -n "$output" ]]; then
        warning_count=$(echo "$output" | grep -cE "\[WARN\]" 2>/dev/null || true)
        [[ "$warning_count" =~ ^[0-9]+$ ]] || warning_count=0
    fi

    # Base64-encode output for HTML report
    local output_b64=""
    if [[ -n "$output" ]]; then
        output_b64=$(printf '%s' "$output" | base64 -w0)
    fi

    # Print compact result (verbose mode never dumps full raw output;
    # failure details are already extracted inline by print_compact_result)
    print_compact_result "$output" "$status" "$duration" false

    TEST_RESULTS+=("$status:$test_file:$duration:$warning_count:$output_b64")
    record_test "$status" "$test_file" "$duration"

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

    if [[ "$PARALLEL_JOBS" -gt 1 ]]; then
        # Parallel execution mode
        local eligible_tests=()
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

            eligible_tests+=("$test_file:$speed")
        done

        tests_run=${#eligible_tests[@]}
        local parallel_dir=$(mktemp -d)
        local all_test_files=()
        local running_pids=()
        local job_idx=0

        for test_entry in "${eligible_tests[@]}"; do
            IFS=':' read -r test_file speed <<< "$test_entry"

            # Wait if we've reached the parallel limit
            while [[ ${#running_pids[@]} -ge $PARALLEL_JOBS ]]; do
                local new_pids=()
                for i in "${!running_pids[@]}"; do
                    if ! kill -0 "${running_pids[$i]}" 2>/dev/null; then
                        wait "${running_pids[$i]}" 2>/dev/null || true
                        unset 'running_pids[$i]'
                    fi
                done
                running_pids=("${running_pids[@]}")
                [[ ${#running_pids[@]} -ge $PARALLEL_JOBS ]] && sleep 0.5
            done

            # Launch test in background
            local result_file="$parallel_dir/result_${job_idx}"
            all_test_files+=("$test_file")
            (
                local test_path="$SCRIPT_DIR/$test_file"
                local start_time=$(date +%s)
                local exit_code=0
                local extra_args=()
                if $AUTO_FIX; then extra_args+=("--auto-fix"); fi
                timeout $TIMEOUT bash "$test_path" "${extra_args[@]}" > "$result_file.out" 2>&1 || exit_code=$?
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                echo "$exit_code:$duration" > "$result_file.meta"
            ) &
            running_pids+=($!)
            job_idx=$((job_idx + 1))
        done

        # Wait for all remaining jobs
        for pid in "${running_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        # Collect results
        for i in $(seq 0 $((job_idx - 1))); do
            local test_file="${all_test_files[$i]}"
            local result_file="$parallel_dir/result_${i}"
            local exit_code=0
            local duration=0

            if [ -f "$result_file.meta" ]; then
                IFS=':' read -r exit_code duration < "$result_file.meta"
            fi

            local output=""
            if [ -f "$result_file.out" ]; then
                output=$(cat "$result_file.out")
            fi

            local status="pass"
            [[ $exit_code -ne 0 ]] && status="fail"

            local warning_count=0
            if [[ -n "$output" ]]; then
                warning_count=$(echo "$output" | grep -cE "\[WARN\]" 2>/dev/null || true)
                [[ "$warning_count" =~ ^[0-9]+$ ]] || warning_count=0
            fi

            local output_b64=""
            if [[ -n "$output" ]]; then
                output_b64=$(printf '%s' "$output" | base64 -w0)
            fi

            print_compact_result "$output" "$status" "$duration" false
            TEST_RESULTS+=("$status:$test_file:$duration:$warning_count:$output_b64")
            record_test "$status" "$test_file" "$duration"

            if [[ "$status" == "fail" ]]; then
                total_failed=$((total_failed + 1))
            fi
        done

        rm -rf "$parallel_dir"
    else
        # Serial execution mode (default)
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
    fi

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
    print_summary 1
}

print_summary() {
    local tests_run="${1:-0}"
    local passed=0
    local failed=0
    local skipped=0
    local warnings=0
    local total_duration=0

    for result in "${TEST_RESULTS[@]}"; do
        IFS=':' read -r status file duration warn_count _rest <<< "$result"
        case "$status" in
            pass) ((passed++)) || true ;;
            fail) ((failed++)) || true ;;
            skip) ((skipped++)) || true ;;
        esac
        # Add warning count (field 4; field 5 is base64 output)
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

    # Show failure recap when there are failures
    if [[ $failed -gt 0 ]]; then
        echo ""
        echo "----------------------------------------"
        echo -e " ${RED}${BOLD}Failed Tests Recap${NC}"
        echo "----------------------------------------"
        echo ""
        local idx=0
        for result in "${TEST_RESULTS[@]}"; do
            IFS=':' read -r status file duration warn_count output_b64 <<< "$result"
            if [[ "$status" != "fail" ]]; then
                continue
            fi
            idx=$((idx + 1))
            echo -e "  ${RED}${idx}.${NC} ${file} (${duration}s)"

            # Decode and extract key failure info
            if [[ -n "$output_b64" ]]; then
                local decoded
                decoded=$(printf '%s' "$output_b64" | base64 -d 2>/dev/null || echo "")
                if [[ -n "$decoded" ]]; then
                    # Show FAIL lines first
                    local fail_lines
                    fail_lines=$(echo "$decoded" | grep -E '\[FAIL\]' | head -10)
                    if [[ -n "$fail_lines" ]]; then
                        echo "$fail_lines" | sed 's/^/      /'
                    fi
                    # Show ERROR lines if present
                    local err_lines
                    err_lines=$(echo "$decoded" | grep -E '\[ERROR\]' | head -5)
                    if [[ -n "$err_lines" ]]; then
                        echo "$err_lines" | sed 's/^/      /'
                    fi
                fi
            fi
            echo ""
        done
    fi

    # Always generate HTML report for CI pipeline consumption (non-fatal on failure)
    output_html "$passed" "$failed" "$skipped" "$warnings" "$total_duration" || true

    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        output_json "$passed" "$failed" "$skipped" "$warnings" "$total_duration"
    fi

    # Print HTML report path so developer/CI can locate it
    local report_path="${HTML_OUTPUT_PATH:-$SCRIPT_DIR/test-ut-report.html}"
    echo ""
    echo -e "  ${BLUE}HTML Report:${NC} file://$report_path"

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
        IFS=':' read -r status file dur warn_cnt _rest <<< "$result"
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
# ANSI to HTML conversion (lightweight)
# =============================================================================

ansi_to_html() {
    sed \
        -e 's/\x1b\[0;31m/<span class="a-r">/g' \
        -e 's/\x1b\[0;32m/<span class="a-g">/g' \
        -e 's/\x1b\[0;33m/<span class="a-y">/g' \
        -e 's/\x1b\[0;34m/<span class="a-b">/g' \
        -e 's/\x1b\[0;36m/<span class="a-c">/g' \
        -e 's/\x1b\[2m//g' \
        -e 's/\x1b\[1m/<span class="a-B">/g' \
        -e 's/\x1b\[0m/<\/span>/g'
}

# =============================================================================
# HTML Report Generator
# =============================================================================

output_html() {
    local passed="$1"
    local failed="$2"
    local skipped="$3"
    local warnings="$4"
    local duration="$5"
    local report_path="${HTML_OUTPUT_PATH:-$SCRIPT_DIR/test-ut-report.html}"

    # Serialize TEST_RESULTS to a temp file for Python to consume
    local data_file=$(mktemp)
    for result in "${TEST_RESULTS[@]}"; do
        printf '%s\n' "$result" >> "$data_file"
    done

    python3 - "$passed" "$failed" "$skipped" "$warnings" "$duration" "$report_path" "$PLATFORM" "$data_file" <<'PYEOF'
import sys, base64, html as html_module, datetime, re

passed, failed, skipped, warnings, duration, report_path, platform, data_file = sys.argv[1:9]

# ---------------------------------------------------------------------------
# Parse serialized test results
# ---------------------------------------------------------------------------
tests = []
with open(data_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 4)
        status = parts[0]
        name = parts[1]
        dur = parts[2]
        warn_cnt = parts[3] if len(parts) > 3 else "0"
        output_b64 = parts[4] if len(parts) > 4 else ""
        output_text = ""
        if output_b64:
            try:
                output_text = base64.b64decode(output_b64).decode("utf-8", errors="replace")
            except Exception:
                pass
        tests.append({
            "status": status,
            "name": name,
            "duration": dur,
            "warnings": warn_cnt,
            "output": output_text,
        })

# Sort: fail first, then skip, then pass
order = {"fail": 0, "skip": 1, "pass": 2}
tests.sort(key=lambda t: (order.get(t["status"], 3), t["name"]))

status_meta = {
    "pass": ("通过", "pass"),
    "fail": ("失败", "fail"),
    "skip": ("跳过", "skip"),
}

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------
# ANSI to HTML helper
# ---------------------------------------------------------------------------
def ansi_to_html(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    ansi_pattern = re.compile(r"\x1b\[([0-9;]*)m")

    def attrs_to_classes(attrs):
        classes = []
        for attr in attrs:
            if attr in ("31", "0;31"):
                classes.append("a-r")
            elif attr in ("32", "0;32"):
                classes.append("a-g")
            elif attr in ("33", "0;33"):
                classes.append("a-y")
            elif attr in ("34", "0;34"):
                classes.append("a-b")
            elif attr in ("36", "0;36"):
                classes.append("a-c")
            elif attr == "1":
                classes.append("a-B")
            # "2" (DIM) is ignored
        return classes

    result = []
    open_spans = []
    last_end = 0

    for match in ansi_pattern.finditer(text):
        result.append(text[last_end:match.start()])
        code = match.group(1)
        parts = code.split(";") if code else ["0"]
        has_reset = ("0" in parts) or (code == "")
        attrs = [p for p in parts if p != "0"]
        classes = attrs_to_classes(attrs)

        if has_reset:
            while open_spans:
                result.append("</span>")
                open_spans.pop()

        for cls in classes:
            result.append(f'<span class="{cls}">')
            open_spans.append(cls)

        last_end = match.end()

    result.append(text[last_end:])

    while open_spans:
        result.append("</span>")
        open_spans.pop()

    return "".join(result)

# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
html_body = []
html_body.append(f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CANNBot-skills UT 测试报告</title>
<style>
:root {{
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #c9d1d9;
  --muted: #8b949e;
  --pass: #3fb950;
  --fail: #f85149;
  --skip: #d29922;
  --warn: #d29922;
  --info: #58a6ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}
.page-header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--bg);
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}}
.dashboard {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 1rem max(1.5rem, calc((100% - 1200px) / 2));
}}
.dashboard h1 {{
  margin: 0 0 0.5rem;
  font-size: 1.25rem;
  letter-spacing: 0.02em;
  text-align: center;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
  gap: 0.75rem;
  max-width: 600px;
  margin: 0 auto;
}}
.stat {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 1rem;
  text-align: center;
  transition: all 0.15s ease;
}}
.stat.clickable {{
  cursor: pointer;
}}
.stat.clickable:hover {{
  background: rgba(88,166,255,0.08);
  border-color: var(--info);
  transform: translateY(-1px);
}}
.stat.clickable.active {{
  background: rgba(88,166,255,0.15);
  border-color: var(--info);
  box-shadow: 0 0 0 2px rgba(88,166,255,0.2);
}}
.stat .count {{
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
}}
.stat.pass {{ border-color: var(--pass); }}
.stat.pass .count {{ color: var(--pass); }}
.stat.fail {{ border-color: var(--fail); }}
.stat.fail .count {{ color: var(--fail); }}
.stat.skip {{ border-color: var(--skip); }}
.stat.skip .count {{ color: var(--skip); }}
.stat.warn {{ border-color: var(--warn); }}
.stat.warn .count {{ color: var(--warn); }}
.meta {{
  margin-top: 0.5rem;
  color: var(--muted);
  font-size: 0.85rem;
  text-align: center;
}}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 0.75rem max(1.5rem, calc((100% - 1200px) / 2));
  border-bottom: 1px solid var(--border);
  background: var(--bg);
  align-items: center;
}}
.toolbar input {{
  flex: 1;
  min-width: 180px;
  padding: 0.4rem 0.75rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
}}
.toolbar button {{
  padding: 0.4rem 0.9rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  cursor: pointer;
  transition: all 0.15s ease;
  font-size: 0.85rem;
}}
.toolbar button:hover {{
  border-color: var(--info);
}}
.toolbar button.active {{
  background: var(--info);
  border-color: var(--info);
  color: #fff;
}}
.toolbar .nav-btn {{
  background: rgba(248,81,73,0.1);
  border-color: var(--fail);
  color: var(--fail);
}}
.toolbar .nav-btn:hover {{
  background: rgba(248,81,73,0.2);
}}
.toolbar .nav-btn:disabled {{
  opacity: 0.4;
  cursor: not-allowed;
}}
.toolbar .sort-select {{
  padding: 0.4rem 0.6rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 0.85rem;
  cursor: pointer;
}}
.toolbar-sep {{
  width: 1px;
  height: 24px;
  background: var(--border);
  margin: 0 0.25rem;
}}
#test-list {{
  padding: 1.5rem 1.5rem 3rem;
  max-width: 1200px;
  margin: 0 auto;
}}
.test-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 0.75rem;
  overflow: hidden;
  transition: border-color 0.3s ease;
}}
.test-card.highlight {{
  animation: pulse-border 1.5s ease;
}}
@keyframes pulse-border {{
  0% {{ border-color: var(--info); box-shadow: 0 0 0 3px rgba(88,166,255,0.3); }}
  100% {{ border-color: var(--border); box-shadow: none; }}
}}
.test-card summary {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  cursor: pointer;
  list-style: none;
  user-select: none;
}}
.test-card summary::-webkit-details-marker {{ display: none; }}
.badge {{
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  text-transform: uppercase;
  min-width: 48px;
  text-align: center;
}}
.badge.pass {{ background: rgba(63,185,80,0.15); color: var(--pass); }}
.badge.fail {{ background: rgba(248,81,73,0.15); color: var(--fail); box-shadow: 0 0 6px rgba(248,81,73,0.25); }}
.badge.skip {{ background: rgba(210,153,34,0.15); color: var(--skip); }}
.test-card .name {{
  flex: 1;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.9rem;
}}
.test-card .duration {{
  color: var(--muted);
  font-size: 0.85rem;
}}
.test-card .warn-count {{
  color: var(--warn);
  font-size: 0.8rem;
}}
.test-card .duration-bar {{
  width: 60px;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 2px;
}}
.test-card .duration-bar-fill {{
  height: 100%;
  background: var(--info);
  border-radius: 2px;
}}
.log {{
  margin: 0;
  padding: 0;
  background: var(--bg);
  border-top: 1px solid var(--border);
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.82rem;
  line-height: 1.5;
  max-height: 600px;
  overflow: auto;
}}
.log-line {{
  display: flex;
  padding: 0 1rem;
}}
.log-line:hover {{
  background: rgba(88,166,255,0.05);
}}
.log-lineno {{
  width: 40px;
  text-align: right;
  padding-right: 0.75rem;
  color: var(--muted);
  font-size: 0.78rem;
  user-select: none;
  flex-shrink: 0;
}}
.log-content {{
  flex: 1;
  white-space: pre-wrap;
  word-break: break-word;
}}
.log-error {{
  background: rgba(248,81,73,0.08);
  border-left: 2px solid var(--fail);
  padding-left: calc(1rem - 2px);
}}
.log-fold {{
  text-align: center;
  padding: 0.3rem;
  color: var(--muted);
  font-size: 0.8rem;
  cursor: pointer;
}}
.log-fold:hover {{
  color: var(--text);
  background: rgba(88,166,255,0.05);
}}
.a-r {{ color: #f85149; }}
.a-g {{ color: #3fb950; }}
.a-y {{ color: #d29922; }}
.a-b {{ color: #58a6ff; }}
.a-c {{ color: #39c5cf; }}
.a-B {{ font-weight: 700; }}
.empty-tip {{
  text-align: center;
  color: var(--muted);
  padding: 3rem;
}}
.empty-tip .icon {{ font-size: 2rem; margin-bottom: 0.5rem; }}
.fix-guide {{
  max-width: 1200px;
  margin: 0.75rem auto 1rem;
  background: var(--surface);
  border: 1px solid var(--fail);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}}
.fix-guide::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--fail), #ff7b72);
  box-shadow: 0 0 12px rgba(248,81,73,0.35);
}}
.fix-guide.top-placement {{
  margin-top: 0;
  margin-bottom: 0;
  animation: slideDown 0.4s ease;
}}
@keyframes slideDown {{
  from {{ opacity: 0; transform: translateY(-12px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.fix-guide summary {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.85rem 1rem;
  cursor: pointer;
  list-style: none;
  user-select: none;
  background: rgba(248,81,73,0.08);
}}
.fix-guide .fail-count-badge {{
  background: var(--fail);
  color: #fff;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.2rem 0.6rem;
  border-radius: 12px;
  margin-right: 0.3rem;
}}
.fix-guide .guide-actions {{
  display: flex;
  gap: 0.5rem;
  align-items: center;
}}
.fix-guide .jump-btn {{
  padding: 0.35rem 0.75rem;
  background: rgba(248,81,73,0.15);
  border: 1px solid var(--fail);
  border-radius: 6px;
  color: var(--fail);
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  transition: all 0.15s ease;
}}
.fix-guide .jump-btn:hover {{
  background: rgba(248,81,73,0.25);
}}
.fix-guide summary::-webkit-details-marker {{ display: none; }}
.fix-guide .guide-title {{
  flex: 1;
  font-weight: 700;
  color: var(--fail);
}}
.fix-guide .guide-body {{
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.88rem;
  line-height: 1.7;
}}
.fix-guide .guide-body h3 {{
  font-size: 0.95rem;
  color: var(--info);
  margin: 1rem 0 0.5rem;
}}
.fix-guide .guide-body h3:first-child {{ margin-top: 0; }}
.fix-guide .guide-body code {{
  background: var(--bg);
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.82rem;
}}
.fix-guide .guide-body pre {{
  background: var(--bg);
  padding: 0.75rem 1rem;
  border-radius: 6px;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.8rem;
  overflow-x: auto;
  white-space: pre-wrap;
}}
.fix-guide .do-dont {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin: 0.75rem 0;
}}
.fix-guide .do, .fix-guide .dont {{
  padding: 0.6rem 0.9rem;
  border-radius: 6px;
  font-size: 0.82rem;
}}
.fix-guide .do  {{ background: rgba(63,185,80,0.1); border-left: 3px solid var(--pass); }}
.fix-guide .dont {{ background: rgba(248,81,73,0.1); border-left: 3px solid var(--fail); }}
.fix-guide .do ul, .fix-guide .dont ul {{
  margin: 0.3rem 0 0;
  padding-left: 1.2rem;
}}
.fix-guide .copy-btn {{
  display: inline-block;
  padding: 0.25rem 0.7rem;
  background: var(--info);
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.78rem;
  font-weight: 600;
}}
.fix-guide .copy-btn:hover {{ opacity: 0.85; }}
.fix-guide .copy-btn.copied {{ background: var(--pass); }}
.all-pass-banner {{
  max-width: 1200px;
  margin: 0.75rem auto 1rem;
  background: var(--surface);
  border: 1px solid var(--pass);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  animation: slideDown 0.4s ease;
}}
.all-pass-banner::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--pass), #56d364);
  box-shadow: 0 0 12px rgba(63,185,80,0.35);
}}
.all-pass-banner .banner-body {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.85rem 1rem;
  background: rgba(63,185,80,0.08);
}}
.all-pass-banner .banner-body .icon {{
  font-size: 1.25rem;
}}
.all-pass-banner .banner-body .text {{
  flex: 1;
  font-weight: 700;
  color: var(--pass);
}}
@media (max-width: 640px) {{
  .stats {{ grid-template-columns: repeat(3, 1fr); max-width: 100%; }}
  .toolbar {{ position: relative; }}
  .toolbar input {{ width: 100%; flex: none; }}
  .toolbar-sep {{ display: none; }}
  .fix-guide .do-dont {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="page-header">
<header class="dashboard">
  <h1>CANNBot-skills UT 测试报告</h1>
  <div class="stats">
    <div class="stat pass clickable" data-filter="pass" onclick="filter('pass')"><span class="count">{passed}</span>通过</div>
    <div class="stat fail clickable" data-filter="fail" onclick="filter('fail')"><span class="count">{failed}</span>失败</div>
    <div class="stat skip clickable" data-filter="skip" onclick="filter('skip')"><span class="count">{skipped}</span>跳过</div>
    <div class="stat warn clickable" data-filter="warn" onclick="filter('warn')"><span class="count">{warnings}</span>警告</div>
    <div class="stat time"><span class="count">{duration}s</span>耗时</div>
  </div>
  <div class="meta">Platform: {platform} | {timestamp}</div>
</header>''')

# Banner container: show fix-guide when failures exist, show pass banner otherwise
if int(failed) > 0:
    fix_prompt = """\\
# 任务：修复 UT 测试失败

`./tests/run-tests.sh --fast` 当前存在失败项，请根据以下指南分析并修复。

## 快速修复流程

### Step 1: 运行测试并尝试自动修复
```bash
cd <项目根目录>
./tests/run-tests.sh --fast
```

**部分失败支持一键自动修复：**
```bash
./tests/run-tests.sh --fast --auto-fix
```
可自动修复：
- 版本号未 bump（plugin.json / marketplace.json）
- CRLF 换行符（自动转换为 LF）

自动修复后请重跑 `./tests/run-tests.sh --fast` 确认通过。
若仍有失败，进入 Step 2。

### Step 2: 阅读失败日志定位根因
每个失败测试的输出格式：
```
[FAIL] 规则编号: 错误描述
    - 涉及文件: 路径
```

**定位方法：**
1. 找到 `[FAIL]` 或 `[ERROR]` 标记的行
2. 注意前面的规则编号（如 S-STR-06、A-CON-03）
3. 根据规则编号查下表确定修复方法
4. 注意文件名和路径提示

### Step 3: 按失败类型修复

| 测试 | 规则 | 常见失败 | 修复方法 |
|---|---|---|---|
| **Line Endings** | - | 文件含 CRLF 换行符 | `--auto-fix` 或 `dos2unix <file>` |
| **Version Care** | - | 改了文件未 bump 版本号 | `--auto-fix` 自动 bump plugin.json |
| **Version Care** | fork-sync | 本地 bump 后 CI 仍报未 bump | 远程主线已更新，需 `git fetch upstream && git rebase upstream/master` 后再 bump |
| **Version Care** | origin-drift | rebase 后本地测试仍失败 | `origin/master` 还是旧 commit，先 `git push --force-with-lease` 再测试 |
| **Version Care** | mismatch | marketplace.json 版本不一致 | 手动同步 marketplace.json 与 plugin.json 版本 |
| **Version Care** | decrease | 版本号降低 | 版本只能升高，建议恢复到 base 版本或更高 |
| **Skill Structure** | S-STR-01 | YAML Front Matter 格式错误 | 确保 `---` 包裹 metadata |
| | S-STR-02/03 | name/description 字段缺失 | 补充对应字段 |
| | S-STR-05 | name 长度超 64 字符 | 缩短 name |
| | S-STR-06 | name 格式错误 | 使用小写+连字符，如 `my-skill-name` |
| | S-STR-07 | description 长度超 1024 | 精简描述或移到 references/ |
| | S-STR-08 | 链接指向不存在的文件 | 修复链接路径或创建目标文件 |
| **Skill Content** | S-CON-01 | name 与目录名不匹配 | 统一 name 和目录名 |
| | S-CON-02/03 | description 缺少触发关键词/条件 | 添加 `Use when...` 或 `当...时` 触发条件 |
| | S-CON-04 | 缺少可操作指令 | 添加代码块或编号步骤 |
| | S-CON-08 | description 非三段式 | 按 `概述→触发条件→能力范围` 组织 |
| **Agent Structure** | A-STR-01~09 | 同 Skill Structure 规则 | 修复 AGENT.md 对应字段 |
| | A-STR-03 | mode 不是 primary/subagent | 修正 mode 字段 |
| | A-STR-04 | skill 依赖不存在 | 确认 skill 路径正确或移除无效依赖 |
| **Agent Content** | A-CON-01~09 | 同 Skill Content 规则 | 修复 AGENT.md 对应内容 |
| **Team Structure** | T-STR-01~08 | YAML/字段/链接问题 | 修复对应 team 的 SKILL.md |
| | T-STR-03 | mode 不是 primary | 修正为 `mode: primary` |
| **Team Content** | T-CON-01 | 目录命名格式错误 | 使用小写+连字符命名目录 |
| | T-CON-03 | description 缺少触发条件 | 同 S-CON-03 |

**关键修复原则：**
- name 格式: `^[a-z0-9]+(-[a-z0-9]+)*$`（全小写，连字符分隔）
- description 必须包含触发条件（如 `Use when...` 或 `当...时`）
- 修改 skill/agent/team 文件后，**必须**同步 bump 对应 plugin.json 版本号
- 链接必须指向实际存在的文件

### Step 4: 验证
```bash
./tests/run-tests.sh --fast
```
确认 `Failed: 0` 且 `STATUS: PASSED`。

## 约束
- **允许修改**: SKILL.md / AGENT.md / AGENTS.md / plugin.json / init.sh / 目录结构 / marketplace.json
- **禁止修改**: tests/ 目录下任何文件（测试脚本、lib 库、配置）
- **禁止**: 跳过/绕过/禁用任何测试规则

## 修复后检查清单
修复完成后，在提交前确认：
- [ ] `./tests/run-tests.sh --fast` 输出 `Failed: 0`
- [ ] 若修改了 skill/agent/team 内容，plugin.json 版本号已 bump
- [ ] 若修改了文件路径，所有链接已同步更新
- [ ] 若 CI 失败但本地已通过，检查 `origin/master` 是否与 `upstream/master` 同步
- [ ] rebase 后先 push 再运行本地测试，避免 origin/master 假阳性
- [ ] 没有修改 tests/ 目录下的任何文件

## 分析框架（五问法）
1. **哪个测试失败了？** → 看测试名称和规则编号
2. **涉及哪个文件？** → 看 `[FAIL]` 后的文件路径提示
3. **违反了什么规则？** → 对照上表查规则含义
4. **怎么最小修复？** → 只改必要字段，不重构无关内容
5. **全量重跑验证了吗？** → 确认 `./tests/run-tests.sh --fast` 全绿\\
"""
    html_body.append(f'''<div id="banner-container">
<details class="fix-guide top-placement" id="fix-guide-banner">
    <summary>
      <span>&#128736;</span>
      <span class="fail-count-badge" id="fail-count-badge">{failed} 个测试失败，需要修复</span>
      <span class="guide-title">UT Test 失败修复指南 — 将此提示词粘贴给 AI 自动修复</span>
      <div class="guide-actions">
        <button class="jump-btn" onclick="scrollToFirstFail()" title="跳转到第一个失败测试">&#9660; 跳转第一个失败</button>
        <button class="copy-btn" onclick="copyFixPrompt()" id="copy-btn">复制提示词</button>
      </div>
    </summary>
    <div class="guide-body">
      <h3>执行流程</h3>
      <p><strong>Step 1:</strong> 运行 <code>./tests/run-tests.sh --fast</code> 收集所有 <code>[FAIL]</code> 项。部分失败可先用 <code>./tests/run-tests.sh --fast --auto-fix</code> 一键自动修复（支持版本号 bump 和 CRLF 转换），修复后重跑确认。</p>
      <p><strong>Step 2:</strong> 按失败信息反向定位源文件 — 组件名 → 源文件路径 → 判断修改内容。</p>
      <p><strong>Step 3:</strong> 只修改项目源文件（SKILL.md、AGENT.md、plugin.json、init.sh 等），最小化修复。</p>
      <p><strong>Step 4:</strong> 重新运行 <code>./tests/run-tests.sh --fast</code>，确认 <code>Failed: 0</code>。</p>

      <div class="do-dont">
        <div class="do">
          <strong>&#10004; 允许</strong>
          <ul>
            <li>修改 SKILL.md / AGENT.md / AGENTS.md</li>
            <li>修改 plugin.json / 目录结构</li>
            <li>修改 init.sh / marketplace.json</li>
            <li>修复失效链接、补充缺失字段</li>
          </ul>
        </div>
        <div class="dont">
          <strong>&#10008; 禁止</strong>
          <ul>
            <li>修改 tests/ 目录下任何文件</li>
            <li>修改测试脚本逻辑或阈值</li>
            <li>修改测试 lib 库或 helper</li>
            <li>跳过/绕过/禁用测试规则</li>
          </ul>
        </div>
      </div>

      <h3>分析框架</h3>
      <p>测试要求什么 → 哪个文件没满足 → 为什么没满足 → 怎么最小修复 → 全量重跑验证</p>

      <h3>&#128203; 可复制提示词（粘贴给 AI 助手）</h3>
      <pre id="fix-prompt-text">{html_module.escape(fix_prompt)}</pre>
    </div>
  </details>
</div>''')
else:
    html_body.append('''<div id="banner-container">
<div class="all-pass-banner" id="all-pass-banner">
  <div class="banner-body">
    <span class="icon">&#127881;</span>
    <span class="text">全部测试通过！无需修复操作。</span>
  </div>
</div>
</div>''')

html_body.append(f'''<div class="toolbar">
  <input type="text" id="search" placeholder="搜索测试名称..." oninput="doSearch()">
  <button id="btn-all" onclick="filter('all')">全部</button>
  <button id="btn-fail" onclick="filter('fail')">仅失败</button>
  <button id="btn-pass" onclick="filter('pass')">仅通过</button>
  <button id="btn-skip" onclick="filter('skip')">仅跳过</button>
  <div class="toolbar-sep"></div>
  <button onclick="expandAll()">展开全部</button>
  <button onclick="collapseAll()">收起全部</button>
  <div class="toolbar-sep"></div>
  <select class="sort-select" id="sort-select" onchange="sortTests(this.value)">
    <option value="default">默认排序</option>
    <option value="duration-desc">耗时最长</option>
    <option value="duration-asc">耗时最短</option>
  </select>
  <div class="toolbar-sep"></div>
  <button class="nav-btn" id="nav-prev" onclick="navigateFail(-1)" title="上一个失败 (k)">&#9664; 上一个失败</button>
  <button class="nav-btn" id="nav-next" onclick="navigateFail(1)" title="下一个失败 (j)">下一个失败 &#9654;</button>
</div>
</div>
<main id="test-list">
''')

max_dur = max(int(t["duration"]) for t in tests) if tests else 1
for t in tests:
    st_label, st_cls = status_meta.get(t["status"], (t["status"], t["status"]))
    warn_badge = f'<span class="warn-count">&#9888; {t["warnings"]} 警告</span>' if t["warnings"] and int(t["warnings"]) > 0 else ''
    output_html_text = ansi_to_html(t["output"])
    dur_pct = min(100, int(int(t["duration"]) / max_dur * 100)) if max_dur > 0 else 0
    # Build log with line numbers and folding
    log_lines = output_html_text.split('\n')
    log_html_parts = []
    total_lines = len(log_lines)
    fold_threshold = 30
    fold_start = 15
    fold_end = 15
    for i, line in enumerate(log_lines):
        lineno = i + 1
        is_error = '[FAIL]' in line or '[ERROR]' in line
        error_cls = ' log-error' if is_error else ''
        if total_lines > fold_threshold and i == fold_start:
            fold_count = total_lines - fold_start - fold_end
            log_html_parts.append(f'<div class="log-fold" onclick="toggleFold(this)">... {fold_count} 行已折叠，点击展开 ...</div>')
            log_html_parts.append(f'<div class="fold-content" style="display:none">')
        if total_lines > fold_threshold and i == total_lines - fold_end:
            log_html_parts.append('</div>')
        log_html_parts.append(f'<div class="log-line{error_cls}"><span class="log-lineno">{lineno}</span><span class="log-content">{line or " "}</span></div>')
    log_html = '\n'.join(log_html_parts)
    html_body.append(f'''  <details class="test-card {st_cls}" data-name="{html_module.escape(t["name"])}" data-status="{st_cls}" data-duration="{t["duration"]}">
    <summary>
      <span class="badge {st_cls}">{st_label}</span>
      <span class="name">{html_module.escape(t["name"])}</span>
      {warn_badge}
      <span class="duration">{t["duration"]}s<div class="duration-bar"><div class="duration-bar-fill" style="width:{dur_pct}%"></div></div></span>
    </summary>
    <div class="log">
{log_html}
    </div>
  </details>
''')

html_body.append(f'''</main>
''')
html_body.append(f'''<script>
let currentFailIndex = -1;
let failCards = [];
let currentFilter = 'all';

function updateEmptyTip(visible, context) {{
  const list = document.getElementById('test-list');
  let tip = list.querySelector('.empty-tip');
  if (visible === 0) {{
    if (!tip) {{
      tip = document.createElement('div');
      tip.className = 'empty-tip';
      list.appendChild(tip);
    }}
    let msg = '没有匹配的测试';
    let icon = '&#128269;';
    if (context === 'fail') {{ msg = '&#127881; 恭喜，没有失败的测试！'; icon = ''; }}
    else if (context === 'pass') {{ msg = '当前没有通过的测试，请先处理失败项'; icon = '&#128679;'; }}
    else if (context === 'skip') {{ msg = '当前没有跳过的测试'; icon = '&#9989;'; }}
    else if (context === 'search') {{ msg = '未找到包含该关键词的测试'; icon = '&#128269;'; }}
    tip.innerHTML = icon ? '<div class="icon">' + icon + '</div><div>' + msg + '</div>' : '<div>' + msg + '</div>';
    tip.style.display = '';
  }} else if (tip) {{
    tip.style.display = 'none';
  }}
}}

function setButtonActive(status) {{
  document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
  const idMap = {{'all':'btn-all','fail':'btn-fail','pass':'btn-pass','skip':'btn-skip'}};
  const btn = document.getElementById(idMap[status]);
  if (btn) btn.classList.add('active');
}}

function setStatActive(status) {{
  document.querySelectorAll('.stat.clickable').forEach(s => s.classList.remove('active'));
  if (status !== 'all' && status !== 'warn') {{
    const s = document.querySelector('.stat[data-filter="' + status + '"]');
    if (s) s.classList.add('active');
  }}
}}

function filter(status) {{
  currentFilter = status;
  const cards = document.querySelectorAll('.test-card');
  let visible = 0;
  cards.forEach(c => {{
    const show = status === 'all' || c.dataset.status === status;
    c.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  setButtonActive(status);
  setStatActive(status);
  updateEmptyTip(visible, status);
  updateNavButtons();
}}

function doSearch() {{
  const q = document.getElementById('search').value.toLowerCase();
  const cards = document.querySelectorAll('.test-card');
  let visible = 0;
  cards.forEach(c => {{
    const show = !q || c.dataset.name.toLowerCase().includes(q);
    c.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.stat.clickable').forEach(s => s.classList.remove('active'));
  updateEmptyTip(visible, 'search');
}}

function expandAll() {{
  document.querySelectorAll('.test-card').forEach(c => c.open = true);
}}
function collapseAll() {{
  document.querySelectorAll('.test-card').forEach(c => c.open = false);
}}

function updateNavButtons() {{
  failCards = Array.from(document.querySelectorAll('.test-card.fail')).filter(c => c.style.display !== 'none');
  const prev = document.getElementById('nav-prev');
  const next = document.getElementById('nav-next');
  if (prev) prev.disabled = failCards.length === 0;
  if (next) next.disabled = failCards.length === 0;
}}

function navigateFail(direction) {{
  failCards = Array.from(document.querySelectorAll('.test-card.fail')).filter(c => c.style.display !== 'none');
  if (failCards.length === 0) return;
  currentFailIndex += direction;
  if (currentFailIndex < 0) currentFailIndex = failCards.length - 1;
  if (currentFailIndex >= failCards.length) currentFailIndex = 0;
  const card = failCards[currentFailIndex];
  card.open = true;
  card.scrollIntoView({{behavior: 'smooth', block: 'center'}});
  card.classList.remove('highlight');
  void card.offsetWidth; // force reflow
  card.classList.add('highlight');
  setTimeout(() => card.classList.remove('highlight'), 1500);
}}

function sortTests(order) {{
  const list = document.getElementById('test-list');
  const cards = Array.from(list.querySelectorAll('.test-card'));
  if (order === 'duration-desc') {{
    cards.sort((a, b) => parseInt(b.dataset.duration) - parseInt(a.dataset.duration));
  }} else if (order === 'duration-asc') {{
    cards.sort((a, b) => parseInt(a.dataset.duration) - parseInt(b.dataset.duration));
  }} else {{
    cards.sort((a, b) => {{
      const oa = {{'fail':0,'skip':1,'pass':2}}[a.dataset.status] || 3;
      const ob = {{'fail':0,'skip':1,'pass':2}}[b.dataset.status] || 3;
      if (oa !== ob) return oa - ob;
      return a.dataset.name.localeCompare(b.dataset.name);
    }});
  }}
  cards.forEach(c => list.appendChild(c));
}}

function toggleFold(el) {{
  const content = el.nextElementSibling;
  if (content && content.classList.contains('fold-content')) {{
    const isHidden = content.style.display === 'none';
    content.style.display = isHidden ? '' : 'none';
    el.textContent = isHidden ? '... 点击折叠 ...' : '... ' + (content.children.length) + ' 行已折叠，点击展开 ...';
  }}
}}

// Auto-expand failed tests on load
document.querySelectorAll('.test-card.fail').forEach(c => c.open = true);

// Scroll to first fail on load
setTimeout(() => {{
  const firstFail = document.querySelector('.test-card.fail');
  if (firstFail) {{
    firstFail.scrollIntoView({{behavior: 'smooth', block: 'center'}});
    firstFail.classList.add('highlight');
    setTimeout(() => firstFail.classList.remove('highlight'), 1500);
  }}
}}, 300);

updateNavButtons();

// Keyboard shortcuts
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'j' || e.key === 'J') {{ e.preventDefault(); navigateFail(1); }}
  if (e.key === 'k' || e.key === 'K') {{ e.preventDefault(); navigateFail(-1); }}
}});

function copyFixPrompt() {{
  const el = document.getElementById('fix-prompt-text');
  const btn = document.getElementById('copy-btn');
  if (!el || !btn) return;
  const txt = document.createElement('textarea');
  txt.innerHTML = el.innerHTML;
  navigator.clipboard.writeText(txt.value).then(() => {{
    btn.textContent = '已复制!';
    btn.classList.add('copied');
    setTimeout(() => {{ btn.textContent = '复制提示词'; btn.classList.remove('copied'); }}, 2000);
  }}).catch(() => {{
    txt.style.position = 'fixed'; txt.style.left = '-9999px';
    document.body.appendChild(txt);
    txt.select(); document.execCommand('copy');
    document.body.removeChild(txt);
    btn.textContent = '已复制!';
    btn.classList.add('copied');
    setTimeout(() => {{ btn.textContent = '复制提示词'; btn.classList.remove('copied'); }}, 2000);
  }});
}}

function scrollToFirstFail() {{
  const firstFail = document.querySelector('.test-card.fail');
  if (firstFail) {{
    firstFail.open = true;
    firstFail.scrollIntoView({{behavior: 'smooth', block: 'center'}});
    firstFail.classList.remove('highlight');
    void firstFail.offsetWidth;
    firstFail.classList.add('highlight');
    setTimeout(() => firstFail.classList.remove('highlight'), 1500);
  }}
}}
</script>
</body>
</html>''')

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html_body))

print(f"HTML report written to: {report_path}")

PYEOF

    rm -f "$data_file"
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

    # Build command with optional parameters (array-based to avoid eval injection)
    local cmd=(bash "$eval_test")

    if [ -n "$EVAL_WORKSPACE" ]; then
        cmd+=(--workspace "$EVAL_WORKSPACE")
    fi

    if [ -n "$EVAL_ITERATION" ]; then
        cmd+=(--iteration "$EVAL_ITERATION")
    fi

    if [ -n "$EVAL_THRESHOLD" ]; then
        cmd+=(--threshold "$EVAL_THRESHOLD")
    fi

    if $EVAL_DETECT_REGRESSION; then
        cmd+=(--detect-regression)
    fi

    if $EVAL_INCREMENTAL; then
        cmd+=(--incremental --base-branch "$EVAL_BASE_BRANCH")
    fi

    if $VERBOSE; then
        cmd+=(--verbose)
    fi

    local start_time=$(date +%s)
    local status="pass"
    local output=""

    print_section "Running: test-skill-eval-results.sh"

    if output=$("${cmd[@]}" 2>&1); then
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
    init_test_cache

    if $RUN_FAST && [[ "$PLATFORM" != "none" ]] && ! is_platform_available "$PLATFORM"; then
        echo -e "${YELLOW}[WARN]${NC} No AI CLI found - will run fast tests only"
        PLATFORM="none"
    fi

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

    # Auto-open HTML report in browser when in desktop environment
    if [[ "$OUTPUT_FORMAT" == "html" ]]; then
        local report_path="${HTML_OUTPUT_PATH:-$SCRIPT_DIR/test-ut-report.html}"
        if [[ -f "$report_path" ]]; then
            if command -v xdg-open &>/dev/null; then
                xdg-open "$report_path" &>/dev/null &
            elif command -v open &>/dev/null; then
                open "$report_path" &>/dev/null &
            fi
        fi
    fi

    cleanup_test_cache
    exit $exit_code
}

master "$@"
