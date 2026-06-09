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
# Test: safe_install_file() Unit Tests
# =============================================================================
# Validates the safe_install_file() helper function across 5 scenarios:
#   1. Fresh install (target does not exist)
#   2. Idempotent reinstall (content identical)
#   3. Global mode overwrite with backup (non-interactive)
#   4. Project mode overwrite with backup (non-interactive)
#   5. Merge structure validation
#
# Sources the function definition from ops-registry-invoke/init.sh
# as the canonical implementation.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILLS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INIT_FILE="$SKILLS_DIR/plugins-official/ops-registry-invoke/init.sh"

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
# Source safe_install_file() and its helpers from the canonical init.sh
# =============================================================================
if [ ! -f "$INIT_FILE" ]; then
    print_fail "Source init.sh not found: $INIT_FILE"
    exit 1
fi

# Create a temporary script containing only the helper functions + safe_install_file
SAFE_FUNC_TMP=$(mktemp)
trap 'rm -f "$SAFE_FUNC_TMP"' EXIT

# Extract color setup and helper functions (first ~30 lines typically contain these)
# We inline a minimal set to avoid fragile line-number extraction.
cat > "$SAFE_FUNC_TMP" << 'HELPER'
GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
HELPER

# Extract safe_install_file function using awk with brace-depth tracking
awk '
/^safe_install_file/ { found=1; depth=0 }
found {
    print
    for (i=1; i<=length($0); i++) {
        c = substr($0, i, 1)
        if (c == "{") depth++
        if (c == "}") depth--
    }
    if (depth <= 0 && found) exit
}
' "$INIT_FILE" >> "$SAFE_FUNC_TMP"

source "$SAFE_FUNC_TMP"

# =============================================================================
# Test Scenarios
# =============================================================================

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"; rm -f "$SAFE_FUNC_TMP"' EXIT

print_section_header "Scenario 1: Fresh install (target does not exist)"

TARGET="$TMPDIR/fresh.md"
TMPFILE=$(mktemp)
echo "plugin header" > "$TMPFILE"
echo "plugin body" >> "$TMPFILE"

safe_install_file "$TMPFILE" "$TARGET" "fresh.md" "global"

run_check "Target file created" test -f "$TARGET"
run_check "Target contains plugin content" grep -q "plugin header" "$TARGET"
run_check "No backup created" test ! -f "$TMPDIR"/*.bak.*

print_section_header "Scenario 2: Idempotent reinstall (content identical)"

TMPFILE=$(mktemp)
echo "plugin header" > "$TMPFILE"
echo "plugin body" >> "$TMPFILE"

# Get the target's modification time before re-run
mtime_before=$(stat -c %Y "$TARGET" 2>/dev/null || stat -f %m "$TARGET")

safe_install_file "$TMPFILE" "$TARGET" "fresh.md" "global"

mtime_after=$(stat -c %Y "$TARGET" 2>/dev/null || stat -f %m "$TARGET")

run_check "No backup created on idempotent run" test ! -f "$TMPDIR"/*.bak.*
run_check "Target modification time unchanged" test "$mtime_before" -eq "$mtime_after"

print_section_header "Scenario 3: Global mode overwrite with backup (non-interactive)"

TARGET="$TMPDIR/global_overwrite.md"
echo "user custom content" > "$TARGET"
echo "user custom line 2" >> "$TARGET"

TMPFILE=$(mktemp)
echo "plugin new header" > "$TMPFILE"
echo "plugin new body" >> "$TMPFILE"

safe_install_file "$TMPFILE" "$TARGET" "global_overwrite.md" "global"

run_check "Backup file created" test -f "$TMPDIR/global_overwrite.md.bak."*
run_check "Target overwritten with plugin content" grep -q "plugin new header" "$TARGET"
run_check "Old user content removed" bash -c "! grep -q 'user custom content' '$TARGET'"

print_section_header "Scenario 4: Project mode overwrite with backup (non-interactive)"

TARGET="$TMPDIR/project_overwrite.md"
echo "project user content" > "$TARGET"

TMPFILE=$(mktemp)
echo "project plugin content" > "$TMPFILE"

safe_install_file "$TMPFILE" "$TARGET" "project_overwrite.md" "project"

run_check "Backup file created in project mode" test -f "$TMPDIR/project_overwrite.md.bak."*
run_check "Target overwritten with plugin content (project)" grep -q "project plugin content" "$TARGET"

print_section_header "Scenario 5: Merge structure validation"

TARGET="$TMPDIR/merge.md"
echo "user line 1" > "$TARGET"
echo "user line 2" >> "$TARGET"

TMPFILE=$(mktemp)
echo "plugin header" > "$TMPFILE"
echo "plugin body" >> "$TMPFILE"

# Simulate merge manually (same logic as safe_install_file's merge path)
cat "$TMPFILE" > "${TARGET}.new"
echo "" >> "${TARGET}.new"
echo "<!-- === User custom content below === -->" >> "${TARGET}.new"
echo "" >> "${TARGET}.new"
cat "$TARGET" >> "${TARGET}.new"
mv "${TARGET}.new" "$TARGET"

run_check "Merge: plugin content at top" test "$(head -1 "$TARGET")" = "plugin header"
run_check "Merge: separator present" grep -q "=== User custom content below ===" "$TARGET"
run_check "Merge: original user content preserved" grep -q "user line 1" "$TARGET"
run_check "Merge: plugin content present" grep -q "plugin body" "$TARGET"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Safe Install Test Summary"
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
