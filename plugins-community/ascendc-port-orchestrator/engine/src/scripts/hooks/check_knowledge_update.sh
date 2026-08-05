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
# Knowledge update gate — blocks agent Stop/commit if knowledge_update.md is missing.
#
# After each completed op, the agent MUST produce knowledge_update.md documenting
# what was learned and what knowledge base files were updated (or explicitly marked
# "no change").
#
# Exit codes:
#   0 = pass (allow)
#   1 = error (logged only)
#   2 = BLOCKING (agent must fix before proceeding)

set -e

WORKSPACE_ROOT="${WORKSPACE_ROOT:-workspace}"
if [ ! -d "$WORKSPACE_ROOT" ]; then
    exit 0  # No workspace — not in a generation workflow
fi

# Find knowledge_update.md modified within last 4 hours
KU_FILE=$(find "$WORKSPACE_ROOT" -name "knowledge_update.md" -mmin -240 2>/dev/null | head -1)

# Only enforce if there's an active workflow (gate_decision.md exists)
GATE_FILE=$(find "$WORKSPACE_ROOT" -name "gate_decision.md" -mmin -240 2>/dev/null | head -1)
if [ -z "$GATE_FILE" ]; then
    exit 0  # No active workflow
fi

if [ -z "$KU_FILE" ]; then
    echo "⚠️ KNOWLEDGE UPDATE MISSING: No knowledge_update.md found in workspace." >&2
    echo "  After completing an op, write knowledge_update.md with:" >&2
    echo "    - ERROR_CORRECTIONS: new entries or 'no change'" >&2
    echo "    - PATTERN_INDEX: validated/new patterns or 'no change'" >&2
    echo "    - OPERATIONAL_KNOWLEDGE: lessons learned or 'no change'" >&2
    echo "    - PLATFORM_BUGS: new bugs found or 'no change'" >&2
    exit 2  # BLOCKING
fi

# Check required sections exist
ERRORS=0
MESSAGES=""

for section in "ERROR_CORRECTIONS" "PATTERN_INDEX" "OPERATIONAL_KNOWLEDGE" "PLATFORM_BUGS"; do
    if ! grep -qi "$section" "$KU_FILE" 2>/dev/null; then
        MESSAGES="$MESSAGES\n  - Missing section: $section"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    echo -e "⚠️ KNOWLEDGE UPDATE INCOMPLETE ($ERRORS missing sections):$MESSAGES" >&2
    echo "  Each section must have a concrete update or explicit 'no change'." >&2
    exit 2  # BLOCKING
fi

exit 0
