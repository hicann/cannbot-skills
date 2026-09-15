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
# Test: MD-01 Markdown Relative Links (plugin-scoped)
# =============================================================================
# Plugin-scoped version of the MD-01 gate. Scans every Markdown file in the
# ascendc-port-orchestrator plugin directory and fails if any relative link
# (inline link, image, or reference-style definition) points to a file or
# directory that does not exist on disk.
#
# The scanning rules live in scripts/md_link_validator.py; its default scan
# root is this plugin's directory, so no --root is passed here.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "MD-01 Markdown Relative Links — scanning plugin at: $PLUGIN_ROOT"
echo ""

report=$(mktemp)
if python3 "$SCRIPT_DIR/md_link_validator.py" > "$report" 2>&1; then
    scanned=$(find "$PLUGIN_ROOT" -name '*.md' -not -path '*/.git/*' -not -path '*/node_modules/*' | wc -l | tr -d ' ')
    echo "[PASS] All Markdown relative links resolve ($scanned Markdown files scanned)"
    rm -f "$report"
    exit 0
fi

# Validator exits 1 when broken links were found; the report lists them all.
broken_count=$(grep -c ': broken link ' "$report" || true)
echo "[FAIL] Found $broken_count broken Markdown relative link(s):"
echo ""
grep ': broken link ' "$report" | while IFS= read -r line; do
    echo "  - $line"
done
echo ""
echo "Fix the links above, then re-run:"
echo "  bash $SCRIPT_DIR/check_md_links.sh"
echo "  # Or scan directly:"
echo "  python3 $SCRIPT_DIR/md_link_validator.py"

rm -f "$report"
exit 1
