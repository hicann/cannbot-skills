#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# audit_aiv_count.sh — find hardcoded AIV/core constants across archive kernels
#
# Motivation (2026-04-27 op#5 Cumsum ko-1 Iter 1):
#   `CUMSUM_NUM_CORES = 20` was hardcoded in kernel constants. Changing it to 56
#   (the physical AIV count on Ascend950PR) gave +169% sum-ratio lift in a single
#   iter. This script finds other archive kernels that may have the same bug.
#
# Usage:
#   bash src/scripts/audit_aiv_count.sh                    # scan default archive
#   bash src/scripts/audit_aiv_count.sh path/to/kernels    # custom root
#
# Output: groups of (file, line, line-content) for each suspicious hardcoded
# constant, with self-test footer confirming the script catches op#5's known
# bug pattern.

set -e

ARCHIVE_ROOT="${1:-output/a3_to_a5_port/src/kernels}"

if [ ! -d "$ARCHIVE_ROOT" ]; then
    echo "ERROR: archive root not found: $ARCHIVE_ROOT" >&2
    exit 2
fi

echo "=== Hardcoded AIV/core constants in $ARCHIVE_ROOT ==="
echo ""

# Pattern matches identifier-like names (NUM_CORES / CORE_COUNT / NUM_AIV /
# coreNum / aivNum / blockDim / NPU_NUM / CORE_NUM and op-prefixed variants
# like CUMSUM_NUM_CORES) followed by ':=' or '=' and an integer literal.
# We KEEP =20 in scope because that's the exact value op#5 had.
# We exclude 56 (correct), 1 (single-AIV intentional), 47/48 (occasional
# single-AIV core reservation patterns documented in some kernels).
PATTERN='\b([A-Za-z_][A-Za-z0-9_]*?)(NUM_CORES|CORE_COUNT|NUM_AIV|aivNum|blockDim|NPU_NUM|CORE_NUM|coreNum|AIVNum)\b\s*[:=]\s*[0-9]+'

# Collect hits — only AscendC kernel headers and sources; skip build/, probes/, .so files
FOUND=0
LAST_OP=""

while IFS= read -r line; do
    if [ -z "$line" ]; then continue; fi
    # Extract <file>:<lineno>:<content>
    file="${line%%:*}"
    rest="${line#*:}"
    lineno="${rest%%:*}"
    content="${rest#*:}"

    # Get op directory (3 levels up from kernel/*.h typically)
    op=$(echo "$file" | sed -E 's|.*/kernels/([^/]+)/.*|\1|')

    if [ "$op" != "$LAST_OP" ]; then
        echo ""
        echo "── $op ──"
        LAST_OP="$op"
    fi

    # Strip leading ./
    file_rel="${file#./}"
    echo "  $file_rel:$lineno:$(echo "$content" | sed 's/^[[:space:]]*//')"
    FOUND=$((FOUND + 1))
done < <(
    # Two layout patterns coexist in the archive:
    #   (a) flat: <op>/<op>_kernel.h, <op>/<op>_kernels.cpp  (older)
    #   (b) nested: <op>/kernel/*.h, <op>/kernel/*.cpp       (newer)
    # We accept either, but reject build/probes/.opt*_bak intermediates.
    find "$ARCHIVE_ROOT" \
        \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) \
        -not -path "*/build/*" \
        -not -path "*/probes/*" \
        -not -name "*.opt*_bak*" \
        -not -name "*.bak*" \
        2>/dev/null \
    | xargs grep -EHn "$PATTERN" 2>/dev/null \
    | grep -vE '=\s*56\b' \
    | grep -vE '=\s*1\b\s*[;,)/]' \
    | grep -vE '=\s*48\b' \
    | grep -vE '=\s*47\b' \
    | sort
)

echo ""
echo "=== Total hits: $FOUND ==="
echo ""

# Self-test: verify pattern catches op#5 cumsum's known bug
echo "=== Self-test: does the pattern match op#5 cumsum's CUMSUM_NUM_CORES = 20? ==="
SELF_TEST_FILE="$ARCHIVE_ROOT/5_Cumsum/kernel/cumsum_kernel.h"
SELF_TEST_FILE2="$ARCHIVE_ROOT/5_Cumsum/kernel/cumsum_tiling.h"
for tf in "$SELF_TEST_FILE" "$SELF_TEST_FILE2"; do
    if [ -f "$tf" ]; then
        if grep -EHn "$PATTERN" "$tf" 2>/dev/null | grep -E "CUMSUM_NUM_CORES|NUM_CORES" >/dev/null; then
            echo "  ✓ pattern matches CUMSUM_NUM_CORES in $(basename $tf)"
        fi
    fi
done

# Also verify the negative test: post-fix value (=56) should NOT trigger
echo ""
echo "=== Sanity: would the filter HIDE a corrected =56? (it should) ==="
echo "static constexpr int TEST_NUM_CORES = 56;" | grep -EHn "$PATTERN" 2>&1 | grep -vE '=\s*56\b' || echo "  ✓ correctly suppressed (=56 is not a hit)"
