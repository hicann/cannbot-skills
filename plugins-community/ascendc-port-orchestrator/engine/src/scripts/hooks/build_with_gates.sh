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
# Gated build script — checks prerequisites THEN runs build_ascendc.py.
# Worker MUST use this script instead of calling build_ascendc.py directly.
#
# Usage (from A5 container):
#   bash /root/AscendOpGenAgent/build_with_gates.sh [build_ascendc.py args...]
#
# Gates checked:
#   1. PROGRESS.md exists with KB Manifest (Tags + LOADED + AVAILABLE)
#   2. All passed → runs build_ascendc.py with forwarded args
#   3. Any gate fails → prints what's missing, exits 1

set -e

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/root/AscendOpGenAgent}"
PROGRESS_FILE=""

# Find PROGRESS.md in the workspace (local machine, not container)
# This script runs ON the A5 container, but PROGRESS.md is on the local machine.
# So we check a gate flag file instead.
GATE_FILE="$BENCHMARK_ROOT/current_task/.kb_manifest_done"

if [ ! -f "$GATE_FILE" ]; then
    echo "============================================" >&2
    echo "BUILD BLOCKED: KB Manifest gate not passed." >&2
    echo "" >&2
    echo "Before building, you must:" >&2
    echo "  1. Complete Phase 0b (KB Scan)" >&2
    echo "  2. Write KB Manifest to PROGRESS.md" >&2
    echo "  3. Create gate file: touch $GATE_FILE" >&2
    echo "" >&2
    echo "The gate file signals that KB scan is complete." >&2
    echo "============================================" >&2
    exit 1
fi

# Gate passed — run the actual build
echo "KB gate: PASS (manifest done)"
cd "$BENCHMARK_ROOT"
python3 utils/build_ascendc.py "$@"
