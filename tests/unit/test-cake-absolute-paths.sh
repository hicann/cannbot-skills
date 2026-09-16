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
# Test: cake plugin developer-specific absolute paths
# =============================================================================
# A skill that hardcodes the path of the machine it was written on works for its
# author and for nobody else. This check rejects such paths in the cake plugin.
#
# Denied:   /data0/<user>/...   /root/...   /home/<user>/...
# Allowed:  /usr/local/Ascend           CANN's own install location
#           /home/developer             the fixed user inside hdspace containers,
#                                       a platform fact rather than an account
#           $HOME, ${HOME}, ~           resolved at run time, which is the point
#           /data0/*/, /home/*/         wildcards name no account, and environment
#                                       discovery chains legitimately search them;
#                                       the pattern's character class excludes '*'.
#
# SCOPE — deliberately the cake plugin, not the repository. Other plugins carry
# several hundred such paths (/home/npu_user, /root/AscendOpGenAgent, /home/ssdz
# and more); a repo-wide gate would fail on content this port does not own and
# cannot fix. Widening it is a separate repo-hygiene proposal.
#
# Resolve a finding by routing through the machinery the plugin already ships:
#   - the skill's own files      -> SK from skills/tlcheck/checks/env/paths.sh
#   - the user's operator tree   -> OUT / REPO_ROOT from the same script
#   - the TileLang checkout      -> TL_INSTALL_DIR via skills/tlcheck/tl_pin.env
#   - the easyasc library        -> EASYASC_DSL_HOME (docs/EXTERNAL_DEPENDENCIES.md)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCAN_ROOT="$REPO_ROOT/plugins-community/collaborative-agent-kernel-evolution"

print_test_banner "cake Absolute Paths" "Scanning the cake plugin for developer-specific absolute paths..."
init_test_tracking

if [ ! -d "$SCAN_ROOT" ]; then
    print_fail "scan root does not exist: ${SCAN_ROOT#"$REPO_ROOT"/}"
    TEST_FAILED=$((TEST_FAILED + 1))
    print_test_summary
    exit 1
fi

hits="$(mktemp)"
trap 'rm -f "$hits"' EXIT

# /home/developer is the hdspace container user; everything else under /home is an account.
grep -rnE '(/data0/[A-Za-z0-9_.-]+|/root/[A-Za-z0-9_.-]|/home/[A-Za-z0-9_.-]+)' "$SCAN_ROOT" \
    --exclude-dir=__pycache__ --exclude='*.pyc' 2>/dev/null \
    | grep -vE '/home/developer(/|$|[^A-Za-z0-9_.-])' \
    > "$hits" || true

scanned="$(find "$SCAN_ROOT" -type f ! -path '*__pycache__*' ! -name '*.pyc' | wc -l | tr -d ' ')"
count="$(wc -l < "$hits" | tr -d ' ')"
files="$(cut -d: -f1 "$hits" | sort -u | wc -l | tr -d ' ')"

print_info "Files scanned: $scanned"

if [ "$count" = "0" ]; then
    print_pass "No developer-specific absolute paths in the cake plugin"
    print_test_summary
    exit 0
fi

print_fail "Found $count occurrence(s) in $files file(s):"
sed "s|^$REPO_ROOT/||" "$hits" | sed 's/\(.\{140\}\).*/\1.../' | sed 's/^/    /'
echo ""
print_info "See the header of this test for the substitution each case needs."
echo ""

TEST_FAILED=$((TEST_FAILED + 1))
print_test_summary
exit 1
