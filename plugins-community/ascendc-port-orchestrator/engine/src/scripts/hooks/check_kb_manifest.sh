#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# KB Manifest gate — blocks agent Stop/commit if KB Manifest is missing from PROGRESS.md.
#
# The worker MUST write a structured KB Manifest during Phase 0b, documenting:
#   - LOADED files (with pattern IDs)
#   - AVAILABLE files (for later reference)
#   - Classification tags
#   - Architecture decision
#
# Exit codes:
#   0 = pass (allow)
#   2 = BLOCKING (agent must fix before proceeding)

set -e

WORKSPACE_ROOT="${WORKSPACE_ROOT:-workspace}"
if [ ! -d "$WORKSPACE_ROOT" ]; then
    exit 0  # No workspace — not in a generation workflow
fi

# Find most recent PROGRESS.md (modified within last 4 hours)
PROGRESS=$(find "$WORKSPACE_ROOT" -name "PROGRESS.md" -mmin -240 2>/dev/null | head -1)

if [ -z "$PROGRESS" ]; then
    exit 0  # No PROGRESS.md — not in an active workflow
fi

ERRORS=0
MESSAGES=""

# Check for KB Manifest section
if ! grep -q "KB Manifest" "$PROGRESS" 2>/dev/null; then
    MESSAGES="$MESSAGES\n  - Missing 'KB Manifest' section in PROGRESS.md"
    ERRORS=$((ERRORS + 1))
fi

# Check for LOADED section
if ! grep -q "LOADED" "$PROGRESS" 2>/dev/null; then
    MESSAGES="$MESSAGES\n  - Missing 'LOADED' section — which KB files were read?"
    ERRORS=$((ERRORS + 1))
fi

# Check for AVAILABLE section with sufficient entries (at least 3 files listed)
if ! grep -q "AVAILABLE" "$PROGRESS" 2>/dev/null; then
    MESSAGES="$MESSAGES\n  - Missing 'AVAILABLE' section — no map for targeted loading"
    ERRORS=$((ERRORS + 1))
else
    AVAIL_COUNT=$(grep -c "AVAILABLE" "$PROGRESS" 2>/dev/null || echo "0")
    if [ "$AVAIL_COUNT" -lt 3 ]; then
        MESSAGES="$MESSAGES\n  - AVAILABLE section too short ($AVAIL_COUNT entries, need >=3) — list ALL unloaded KB files"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check for Tags
if ! grep -q "Tags:" "$PROGRESS" 2>/dev/null; then
    MESSAGES="$MESSAGES\n  - Missing 'Tags:' — no classification tags from Phase 0a"
    ERRORS=$((ERRORS + 1))
fi

# Check for Architecture decision (SIMT or SIMD mentioned)
if ! grep -qiE "Architecture.*:.*SI[MD]T|SIMT|SIMD" "$PROGRESS" 2>/dev/null; then
    MESSAGES="$MESSAGES\n  - Missing Architecture decision (SIMT/SIMD) in KB Manifest"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo -e "⚠️ KB MANIFEST INCOMPLETE ($ERRORS issues):$MESSAGES" >&2
    echo "  Worker must fill in the KB Manifest template in PROGRESS.md." >&2
    echo "  Required: Tags, Architecture, LOADED files, AVAILABLE files." >&2
    exit 2  # BLOCKING
fi

exit 0
