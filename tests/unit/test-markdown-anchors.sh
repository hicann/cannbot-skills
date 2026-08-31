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
# Test: Markdown Anchor Link Integrity (rule ANCHOR)
# =============================================================================
# Guards the PR #915 defect class: intra-repo markdown links whose #anchor no
# longer matches any heading in the target .md file (heading renamed/reworded
# without updating back-references).
#
# Rule:
# - ANCHOR: every `[text](file.md#frag)` / `[text](#frag)` / `[id]: file.md#frag`
#   with a non-empty fragment must resolve to a heading anchor in the target
#   .md file (GitHub/GitCode slug rules, incl. duplicate-heading -N suffix).
#
# Scope is intentionally tight: only anchor mismatches on EXISTING .md targets
# are flagged. Missing files, non-.md targets, external URLs, images, and links
# in code/HTML comments are skipped (separate concerns, not this guard).
# This is a global repo-hygiene check — not per-skill/agent/team.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

print_test_banner "Markdown Anchor Link Check" "Scanning repository for dead intra-doc #anchor links..."

VALIDATOR="$LIB_DIR/markdown_anchor_validator.py"

if [ ! -f "$VALIDATOR" ]; then
    print_error "markdown_anchor_validator.py not found: $VALIDATOR"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    print_error "python3 not found on PATH"
    exit 1
fi

tmp=$(mktemp)
exit_code=0
python3 "$VALIDATOR" "$SKILLS_DIR" > "$tmp" 2>&1 || exit_code=$?

if [ "$exit_code" -ne 0 ] || ! grep -q '"summary"' "$tmp"; then
    print_error "Validator failed (exit $exit_code), no summary found:"
    cat "$tmp"
    rm -f "$tmp"
    exit 1
fi

error_count=0
summary_md=0
summary_links=0

parsed=$(python3 - <<'PYEOF' "$tmp"
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in obj:
            s = obj["summary"]
            print(f"SUMMARY\t{s.get('md_files',0)}\t{s.get('links_checked',0)}\t{s.get('errors',0)}")
        else:
            file_ = obj.get("file", "")
            line_no = obj.get("line", "")
            target = obj.get("target", "")
            anchor = obj.get("anchor", "")
            suggest = obj.get("suggest", "")
            print(f"ERROR\t{file_}\t{line_no}\t{target}\t{anchor}\t{suggest}")
PYEOF
)

while IFS=$'\t' read -r f1 f2 f3 f4 f5 f6; do
    [ -z "$f1" ] && continue
    if [ "$f1" = "SUMMARY" ]; then
        summary_md="$f2"
        summary_links="$f3"
        continue
    fi
    if [ "$f1" = "ERROR" ]; then
        ((error_count++)) || true
        hint=""
        [ -n "$f6" ] && hint="  (nearest valid anchor: '$f6')"
        print_error "$f2:$f3  → $f4 has no heading '#$f5'${hint}"
    fi
done <<< "$parsed"

rm -f "$tmp"

echo ""
echo "========================================"
echo -e " ${BOLD}Markdown Anchor Test Summary${NC}"
echo "========================================"
echo ""
echo "  Markdown files:   $summary_md"
echo "  Anchor links:      $summary_links"
echo -e "  Dead anchors:     ${RED}$error_count${NC}"
echo ""

if [ $error_count -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Fix by either restoring the target heading or re-pointing the link to the"
    echo "current anchor (run the validator directly for the full list):"
    echo "  python3 tests/lib/markdown_anchor_validator.py \"\$PWD\""
    exit 1
else
    print_status_passed
    exit 0
fi
