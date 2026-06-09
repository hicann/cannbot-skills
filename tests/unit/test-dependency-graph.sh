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
# Test: Dependency Graph Integrity
# =============================================================================
# Validates cross-references between marketplace.json, plugin.json, AGENTS.md,
# and agent .md files.
#
# Rules tested:
# - DG-01: marketplace.json skills paths exist
# - DG-02: marketplace.json dependencies valid
# - DG-03: plugin.json agents paths exist
# - DG-04: plugin.json dependencies valid
# - DG-05: AGENTS.md skills references exist
# - DG-06: Agent .md skills references exist
# - DG-07: Orphaned skills detection (warn)
# - DG-08: Orphaned agents detection (warn)
# - DG-09: Circular dependency detection
# - DG-10: init.sh INCLUDED_SKILLS covers all marketplace-declared skills
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "=== Test: Dependency Graph Integrity ==="
echo ""
echo "Validates cross-references between marketplace.json, plugin.json,"
echo "AGENTS.md, and agent .md files."
echo "Run time: ~5 seconds (no CLI needed)"
echo ""

DEPENDENCY_VALIDATOR="$LIB_DIR/dependency_validator.py"

if [ ! -f "$DEPENDENCY_VALIDATOR" ]; then
    print_error "dependency_validator.py not found: $DEPENDENCY_VALIDATOR"
    exit 1
fi

print_section_header "Dependency Graph Validation (DG-01 to DG-09)"

tmp=$(mktemp)
exit_code=0
python3 "$DEPENDENCY_VALIDATOR" "$SKILLS_DIR" > "$tmp" 2>&1 || exit_code=$?

error_count=0
warn_count=0
summary_skills=0
summary_agents=0

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
            print(f"SUMMARY\t{s.get('total_skills',0)}\t{s.get('total_agents',0)}")
        else:
            level = obj.get("level", "")
            rule = obj.get("rule", "")
            msg = obj.get("msg", "")
            print(f"{level}\t{rule}\t{msg}")
PYEOF
)

while IFS=$'\t' read -r field1 field2 field3; do
    [ -z "$field1" ] && continue
    if [ "$field1" = "SUMMARY" ]; then
        summary_skills="$field2"
        summary_agents="$field3"
        continue
    fi
    case "$field1" in
        error)
            ((error_count++)) || true
            print_error "${field2}: ${field3}"
            ;;
        warn)
            ((warn_count++)) || true
            print_warn "${field2}: ${field3}"
            ;;
    esac
done <<< "$parsed"

rm -f "$tmp"

echo ""
echo "========================================"
echo -e " ${BOLD}Dependency Graph Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total skills:     $summary_skills"
echo "  Total agents:     $summary_agents"
echo -e "  Errors:           ${RED}$error_count${NC}"
echo -e "  Warnings:         ${YELLOW}$warn_count${NC}"
echo ""

if [ $error_count -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the broken dependency references."
    exit 1
else
    print_status_passed
    exit 0
fi
