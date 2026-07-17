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
# Test: Tool Adaptation for init.sh (L1 Static) — Unified
# =============================================================================
# Validates that every plugin init.sh properly supports a given tool
# across all integration points. No CLI required — pure static checks.
#
# Usage:
#   bash test-init-tool.sh --tool opencode
#   bash test-init-tool.sh --tool claude
#   bash test-init-tool.sh --all
#
# Rules (XX = tool prefix):
#   XX-01: Parameter parsing accepts tool as TOOL (error)
#   XX-02: Global CONFIG_ROOT maps to correct path (error)
#   XX-03: Project CONFIG_ROOT maps to correct path (error)
#   XX-04: Tool uses correct MD file (error)
#   XX-05: Quick Start section has tool branch (warn)
#   XX-06: Help text mentions tool (warn)
#   XX-07: .gitignore contains tool path (error)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# =============================================================================
# Argument Parsing
# =============================================================================

TOOL=""
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool)  TOOL="$2"; shift 2 ;;
        --all)   RUN_ALL=true; shift ;;
        *)       echo "Unknown option: $1"; exit 1 ;;
    esac
done

ALL_TOOLS=(opencode claude trae cursor copilot codearts)

if $RUN_ALL; then
    PASS_TOTAL=0
    FAIL_TOTAL=0
    for t in "${ALL_TOOLS[@]}"; do
        echo ""
        echo "########################################################################"
        echo "# Running tool: $t"
        echo "########################################################################"
        if bash "$0" --tool "$t"; then
            PASS_TOTAL=$((PASS_TOTAL + 1))
        else
            FAIL_TOTAL=$((FAIL_TOTAL + 1))
        fi
    done
    echo ""
    echo "========================================"
    echo "  All Tools Summary"
    echo "========================================"
    echo "  Tools passed: $PASS_TOTAL / ${#ALL_TOOLS[@]}"
    echo "  Tools failed: $FAIL_TOTAL"
    if [ "$FAIL_TOTAL" -gt 0 ]; then
        print_status_failed
        exit 1
    else
        print_status_passed
        exit 0
    fi
fi

if [ -z "$TOOL" ]; then
    echo "Usage: $0 --tool <tool> | --all"
    echo "Tools: ${ALL_TOOLS[*]}"
    exit 1
fi

# =============================================================================
# Tool Configuration Table
# =============================================================================

case "$TOOL" in
    opencode)
        PREFIX="OC"
        TOOL_LABEL="OpenCode"
        GITIGNORE_PATTERN='\.opencode'
        USES_CLAUDE_MD=false
        ;;
    claude)
        PREFIX="CL"
        TOOL_LABEL="Claude Code"
        GITIGNORE_PATTERN='\.claude'
        USES_CLAUDE_MD=true
        ;;
    trae)
        PREFIX="TR"
        TOOL_LABEL="TRAE"
        GITIGNORE_PATTERN='\.trae|\.marscode|\.traecli'
        USES_CLAUDE_MD=false
        ;;
    cursor)
        PREFIX="CU"
        TOOL_LABEL="Cursor"
        GITIGNORE_PATTERN='\.cursor'
        USES_CLAUDE_MD=false
        ;;
    copilot)
        PREFIX="CO"
        TOOL_LABEL="Copilot"
        GITIGNORE_PATTERN='\.github|\.copilot'
        USES_CLAUDE_MD=false
        ;;
    codearts)
        PREFIX="CA"
        TOOL_LABEL="CodeArts"
        GITIGNORE_PATTERN='codeartsdoer'
        USES_CLAUDE_MD=false
        ;;
    *)
        echo "Unknown tool: $TOOL"
        echo "Supported: ${ALL_TOOLS[*]}"
        exit 1
        ;;
esac

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

run_warn() {
    local name="$1"
    shift
    if "$@"; then
        print_pass "$name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        print_warn "$name"
    fi
}

# =============================================================================
# XX-01: Parameter parsing accepts tool as TOOL
# =============================================================================

check_01() {
    local init="$1"
    case "$TOOL" in
        opencode)
            grep -q 'TOOL="opencode"' "$init" || grep -qE 'opencode\).*TOOL' "$init"
            ;;
        codearts)
            grep -qE 'codearts\).*TOOL="?\$arg' "$init"
            ;;
        *)
            grep -qE "${TOOL}\).*TOOL" "$init" || grep -qE "${TOOL}.*TOOL=\"\\\$arg\"" "$init"
            ;;
    esac
}

# =============================================================================
# XX-02: Global CONFIG_ROOT maps to correct path
# =============================================================================

check_02() {
    local init="$1"
    case "$TOOL" in
        opencode)
            grep -q 'CONFIG_ROOT.*\.config/opencode' "$init" && grep -q 'HOME' "$init"
            ;;
        claude)
            grep -qE 'CONFIG_ROOT="[^"]*HOME[^"]*\.claude"' "$init"
            ;;
        trae)
            grep -q 'Global installation is not supported for Trae' "$init" || \
            grep -qE 'CONFIG_ROOT.*HOME.*(\.trae|\.marscode|\.traecli|\.trae-cn)' "$init" || \
            { grep -q 'detect_trae_variant' "$init" && grep -qE 'CONFIG_ROOT.*HOME.*(\.trae|\.marscode|\.traecli|\.trae-cn)' "$init"; }
            ;;
        cursor)
            awk '
                /"\$TOOL"[[:space:]]*=[[:space:]]*"cursor"/ { tool_line = NR }
                tool_line && NR <= tool_line + 2 && /\.cursor/ && /HOME/ { found = 1 }
                END { exit (found ? 0 : 1) }
            ' "$init"
            ;;
        copilot)
            awk '
                /"\$TOOL"[[:space:]]*=[[:space:]]*"copilot"/ { tool_line = NR }
                tool_line && NR <= tool_line + 2 && /\.copilot/ && /HOME/ { found = 1 }
                END { exit (found ? 0 : 1) }
            ' "$init"
            ;;
        codearts)
            awk '
                /"\$TOOL"[[:space:]]*=[[:space:]]*"codearts"/ { tool_line = NR }
                tool_line && NR <= tool_line + 2 && /\.codeartsdoer/ && /HOME/ { found = 1 }
                END { exit (found ? 0 : 1) }
            ' "$init"
            ;;
    esac
}

# =============================================================================
# XX-03: Project CONFIG_ROOT maps to correct path
# =============================================================================

check_03() {
    local init="$1"
    case "$TOOL" in
        opencode)
            grep -qE 'CONFIG_ROOT.*\.opencode' "$init"
            ;;
        claude)
            grep -qE 'CONFIG_ROOT="[^"]*\.claude"' "$init"
            ;;
        trae)
            grep -qE 'CONFIG_ROOT.*\.trae|CONFIG_ROOT.*\.marscode|CONFIG_ROOT.*\.traecli|detect_trae_variant' "$init"
            ;;
        cursor)
            awk '
                /"\$TOOL"[[:space:]]*=[[:space:]]*"cursor"/ { tool_line = NR }
                tool_line && NR <= tool_line + 2 && /\.cursor/ && !/HOME/ { found = 1 }
                END { exit (found ? 0 : 1) }
            ' "$init"
            ;;
        copilot)
            awk '
                /"\$TOOL"[[:space:]]*=[[:space:]]*"copilot"/ { tool_line = NR }
                tool_line && NR <= tool_line + 2 && /\.github/ && !/HOME/ { found = 1 }
                END { exit (found ? 0 : 1) }
            ' "$init"
            ;;
        codearts)
            awk '
                /"\$TOOL"[[:space:]]*=[[:space:]]*"codearts"/ { tool_line = NR }
                tool_line && NR <= tool_line + 2 && /\.codeartsdoer/ && !/HOME/ { found = 1 }
                END { exit (found ? 0 : 1) }
            ' "$init"
            ;;
    esac
}

# =============================================================================
# XX-04: Tool uses correct MD file
# =============================================================================

check_04() {
    local init="$1"
    if $USES_CLAUDE_MD; then
        grep -q 'CLAUDE\.md' "$init"
    else
        ! grep -qE "${TOOL}.*CLAUDE\.md|CLAUDE\.md.*${TOOL}" "$init"
    fi
}

# =============================================================================
# XX-05: Quick Start section has tool branch
# =============================================================================

check_05() {
    local init="$1"
    case "$TOOL" in
        codearts)
            grep -qE '(elif|if).*"\$TOOL"[[:space:]]*=[[:space:]]*"codearts"' "$init"
            ;;
        *)
            grep -qE "TOOL.*=.*\"${TOOL}\"" "$init" || grep -q "$TOOL" "$init"
            ;;
    esac
}

# =============================================================================
# XX-06: Help text mentions tool
# =============================================================================

check_06() {
    local init="$1"
    case "$TOOL" in
        codearts)
            grep -qE 'Target tool.*codearts' "$init" || \
            grep -qE 'init\.sh.*codearts' "$init" || \
            grep -q 'CodeArts:' "$init"
            ;;
        *)
            grep -q "$TOOL" "$init"
            ;;
    esac
}

# =============================================================================
# XX-07: .gitignore contains tool path
# =============================================================================

check_07() {
    local init="$1"; local dir="$2"
    local gitignore="$dir/.gitignore"
    [ -f "$gitignore" ] || return 1
    grep -qE "$GITIGNORE_PATTERN" "$gitignore"
}

# =============================================================================
# Check: Tool adaptation for every plugin with init.sh
# =============================================================================
print_section_header "Check: ${TOOL_LABEL} tool adaptation"

for base_dir in "$SKILLS_DIR/plugins-official" "$SKILLS_DIR/plugins-community"; do
    [ -d "$base_dir" ] || continue
    for team_dir in "$base_dir"/*; do
        [ -d "$team_dir" ] || continue
        [ -f "$team_dir/init.sh" ] || continue

        team_name=$(basename "$team_dir")
        init_script="$team_dir/init.sh"

        # Skip plugins that do not support this tool.
        if ! grep -qE "(^|[^a-z])${TOOL}([^a-z]|$)" "$init_script" 2>/dev/null; then
            print_info "[$team_name] does not support '${TOOL}', skipping"
            echo ""
            continue
        fi

        print_section_header "Plugin: $team_name"

        run_check "[$team_name] ${PREFIX}-01: parameter parsing accepts ${TOOL} as tool" \
            check_01 "$init_script"

        run_check "[$team_name] ${PREFIX}-02: global config_root maps ${TOOL} to correct path" \
            check_02 "$init_script"

        run_check "[$team_name] ${PREFIX}-03: project config_root maps ${TOOL} to correct path" \
            check_03 "$init_script"

        run_check "[$team_name] ${PREFIX}-04: ${TOOL} uses correct md file" \
            check_04 "$init_script"

        run_warn "[$team_name] ${PREFIX}-05: quick start section has ${TOOL} branch" \
            check_05 "$init_script"

        run_warn "[$team_name] ${PREFIX}-06: help text mentions ${TOOL}" \
            check_06 "$init_script"

        run_check "[$team_name] ${PREFIX}-07: .gitignore contains ${TOOL} path" \
            check_07 "$init_script" "$team_dir"

        echo ""
    done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  ${TOOL_LABEL} Adaptation Test Summary"
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
