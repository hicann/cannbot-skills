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
# Test: Line Endings (CRLF Detection)
# =============================================================================
# Scans all text files in the repository for DOS-style CRLF line endings.
# CRLF line endings cause:
#   - Increased file size (1 extra byte per line)
#   - Extra \r characters in log output
#   - CI hash mismatches due to autocrlf/smudge behavior
#
# This is a global repo-hygiene check — not per-skill/agent/team.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

# Parse arguments
AUTO_FIX=false
for arg in "$@"; do
    case "$arg" in
        --auto-fix) AUTO_FIX=true ;;
    esac
done

print_test_banner "Line Endings Check" "Scanning repository for CRLF (DOS-style) line endings..."
init_test_tracking

# Directories to exclude from scanning
EXCLUDE_DIRS=(
    ".git"
    "node_modules"
    "__pycache__"
    ".version-state"
    "CI"
    "asc-devkit"
    "pypto"
    "cann-recipes-infer"
    "operators"
    ".opencode"
    ".claude"
    ".trae"
    ".marscode"
    ".traecli"
    ".cursor"
)

# Check if a file is binary using the 'file' command (with fallback for containers without it)
is_binary() {
    local f="$1"
    local mime
    if command -v file &>/dev/null; then
        mime=$(file -b --mime-type "$f" 2>/dev/null || echo "unknown")
    else
        # Fallback: judge by extension when 'file' is unavailable
        case "$f" in
            *.md|*.sh|*.py|*.json|*.yaml|*.yml|*.js|*.ts|*.xml|*.txt|*.cfg|*.ini|*.toml)
                mime="text/plain" ;;
            *)
                mime="unknown" ;;
        esac
    fi
    case "$mime" in
        text/*|application/json|application/xml|application/javascript|application/x-shellscript|inode/x-empty)
            return 1 ;;
        *)
            return 0 ;;
    esac
}

print_info "Scanning repository at: $SKILLS_DIR"
echo ""

# Build exclude-dir arguments for grep
grep_excludes=()
for ex in "${EXCLUDE_DIRS[@]}"; do
    grep_excludes+=("--exclude-dir=$ex")
done

# Scan for CRLF in a single grep pass across common text file types
cr_files=$(mktemp)
cr_count=0
scanned=0

# Use grep -rlP for a single-pass scan across text file extensions
grep -rlP '\r' "$SKILLS_DIR" \
    --include='*.md' --include='*.sh' --include='*.py' \
    --include='*.json' --include='*.yaml' --include='*.yml' \
    --include='*.js' --include='*.ts' --include='*.xml' \
    --include='*.txt' --include='*.cfg' --include='*.ini' \
    --include='*.toml' --include='*.h' --include='*.c' \
    --include='*.cpp' --include='*.hpp' --include='*.html' \
    "${grep_excludes[@]}" 2>/dev/null > "$cr_files" || true

cr_count=$(wc -l < "$cr_files" | tr -d ' ')
# Count total files scanned (approximate: all text files in repo)
scanned=$(find "$SKILLS_DIR" \( \
    -name ".git" -o -name "node_modules" -o -name "__pycache__" -o \
    -name ".version-state" -o -name "CI" -o -name "asc-devkit" -o \
    -name "pypto" -o -name "cann-recipes-infer" -o -name "operators" -o \
    -name ".opencode" -o -name ".claude" -o -name ".trae" -o \
    -name ".marscode" -o -name ".traecli" -o -name ".cursor" \
\) -prune -o -type f -print 2>/dev/null | wc -l | tr -d ' ')

print_info "Files scanned: $scanned"

if [ "$cr_count" -eq 0 ]; then
    print_pass "All text files use Unix (LF) line endings"
    rm -f "$cr_files"
    echo ""
    print_test_summary
    exit 0
fi

echo ""

if $AUTO_FIX; then
    # Auto-fix: convert CRLF to LF for each affected file
    fixed_count=0
    while IFS= read -r abs_path; do
        [ -z "$abs_path" ] && continue
        if [ -f "$abs_path" ]; then
            rel_path="${abs_path#$SKILLS_DIR/}"
            if sed --version 2>/dev/null | grep -q GNU; then
                sed -i 's/\r$//' "$abs_path"
            else
                sed -i '' 's/\r$//' "$abs_path"
            fi
            echo -e "  ${GREEN}●${NC} Fixed: $rel_path"
            fixed_count=$((fixed_count + 1))
        fi
    done < "$cr_files"

    rm -f "$cr_files"
    echo ""
    print_pass "Auto-fixed $fixed_count file(s) — CRLF converted to LF"
    echo ""
    print_test_summary
    exit 0
else
    print_fail "Found $cr_count file(s) with CRLF (DOS) line endings:"
    echo ""

    while IFS= read -r abs_path; do
        [ -z "$abs_path" ] && continue
        rel_path="${abs_path#$SKILLS_DIR/}"
        echo -e "  ${RED}●${NC} $rel_path"
    done < "$cr_files"

    echo ""
    print_info "To fix these files, run:"
    echo "  ./tests/run-tests.sh --auto-fix"
    echo "  # Or manually:"
    echo "  dos2unix <file>"
    echo '  grep -rlP "\r" . --include="*.md" --include="*.py" --include="*.h" --include="*.sh" | xargs dos2unix'
    echo ""

    rm -f "$cr_files"

    # Record failure and exit (use direct assignment to avoid ((0)) set -e issue)
    TEST_FAILED=$((TEST_FAILED + 1))
    print_test_summary
    exit 1
fi
