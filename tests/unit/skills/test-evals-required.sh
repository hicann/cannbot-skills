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
# Test: Skill evals/evals.json Requirement (S-EVAL-01)
# =============================================================================
# Enforces that every skill ships a valid evals/evals.json file.
#
# Rules tested:
# - S-EVAL-01: every skill must have evals/evals.json where the file:
#     - exists
#     - parses as valid JSON
#     - declares a non-empty top-level "skill_name"
#   A "skill_name" mismatch against the directory name is a warning only.
#
# This is a full-repo check: it iterates ALL one-level skills under
# ops/ model/ graph/ runtime/ infra/ regardless of incremental mode,
# mirroring test-dependency-graph.sh.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill evals/evals.json Requirement (S-EVAL-01) ==="
echo ""
echo "Validates that every skill ships evals/evals.json (exists, valid JSON,"
echo "with a non-empty top-level skill_name)."
echo "Run time: ~5 seconds (no CLI needed)"
echo ""

# Domain dirs containing one-level skills (mirrors ST skill_dirs, plus runtime/)
DOMAIN_DIRS=(ops model graph runtime infra)

total_skills=0
pass_count=0
fail_count=0
warn_count=0

# Build the domain dir arguments: <SKILLS_DIR>/ops, <SKILLS_DIR>/model, ...
domain_args=()
for d in "${DOMAIN_DIRS[@]}"; do
    [ -d "$SKILLS_DIR/$d" ] && domain_args+=("$SKILLS_DIR/$d")
done

SKILL_FILES=$(find "${domain_args[@]}" -maxdepth 2 -iname "SKILL.md" -print 2>/dev/null || true)

while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || continue

    # Skip skills inside nested external git repos (e.g. pypto, asc-devkit)
    if _is_inside_nested_git_repo "$f"; then
        continue
    fi

    skill_dir=$(dirname "$f")
    skill_name=$(basename "$skill_dir")
    total_skills=$((total_skills + 1))

    evals_file="$skill_dir/evals/evals.json"
    if [ ! -f "$evals_file" ]; then
        print_fail "S-EVAL-01: $skill_dir missing evals/evals.json"
        fail_count=$((fail_count + 1))
        continue
    fi

    # Validate JSON parses and declares a non-empty top-level skill_name
    reason=""
    if ! reason=$(python3 -c '
import json, sys
p = sys.argv[1]
try:
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
except Exception as e:
    sys.exit("invalid JSON: %s" % e)
if not isinstance(d, dict) or not isinstance(d.get("skill_name"), str) or not d["skill_name"].strip():
    sys.exit("missing non-empty top-level skill_name")
' "$evals_file" 2>&1); then
        print_fail "S-EVAL-01: $skill_dir evals/evals.json ($reason)"
        fail_count=$((fail_count + 1))
        continue
    fi

    # skill_name vs. directory name consistency (warning only)
    declared_name=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("skill_name",""))' "$evals_file" 2>/dev/null || true)
    if [ -n "$declared_name" ] && [ "$declared_name" != "$skill_name" ]; then
        print_warn "S-EVAL-01: $skill_dir evals/evals.json skill_name='$declared_name' != directory name '$skill_name'"
        warn_count=$((warn_count + 1))
    fi

    print_pass "S-EVAL-01: $skill_dir"
    pass_count=$((pass_count + 1))
done <<< "$SKILL_FILES"

echo ""
echo "========================================"
echo -e " ${BOLD}Skill evals/evals.json Requirement Summary${NC}"
echo "========================================"
echo ""
echo "  Total skills:  $total_skills"
echo -e "  Passed:        ${GREEN}$pass_count${NC}"
echo -e "  Failed:        ${RED}$fail_count${NC}"
echo -e "  Warnings:      ${YELLOW}$warn_count${NC}"
echo ""

if [ "$fail_count" -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please add a valid evals/evals.json (valid JSON with non-empty"
    echo "top-level skill_name) for the skills listed above."
    exit 1
else
    print_status_passed
    exit 0
fi
