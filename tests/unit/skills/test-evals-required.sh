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
#   In PR UT, validate all cases in affected Skills and directly changed evals files.
#
# This is a full-repo check: it iterates ALL one-level skills under
# ops/ model/ graph/ runtime/ infra/ tools/ regardless of incremental mode,
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
DOMAIN_DIRS=(ops model graph runtime infra tools)

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

# The PR UT Job downloads this manifest alongside ut.sh.
PR_FILELIST="$SKILLS_DIR/pr_filelist.txt"
if [ -f "$PR_FILELIST" ]; then
    if ! python3 -B - "$SKILLS_DIR" "$PR_FILELIST" "$TESTS_DIR/system/scripts" "${DOMAIN_DIRS[@]}" <<'PY'
import json
import os
from pathlib import Path
import sys

repo, manifest, parser_dir = map(Path, sys.argv[1:4])
sys.path.insert(0, str(parser_dir))
from evals_json_parser import iter_pr_evals_errors

repo = repo.resolve()
# The PR scan also covers existing community, communication and embedded Skills.
domains = tuple(map(Path, [*sys.argv[4:], "community/ops", "community/model", "communication"]))


def owning_skill(path):
    owner = None
    for directory in path.parents:
        if directory == repo:
            break
        if (directory / ".git").exists():
            return None
        if owner is None and (directory / "SKILL.md").is_file():
            owner = directory
    if owner is not None:
        relative = owner.relative_to(repo)
        if any(relative != domain and relative.is_relative_to(domain) for domain in domains):
            return owner
        if "skills" in relative.parts[:-1]:
            return owner
    return None


def skill_suites(skill):
    for directory, subdirs, files in os.walk(skill):
        path = Path(directory)
        if path != skill and ("SKILL.md" in files or ".git" in files or ".git" in subdirs):
            subdirs.clear()
            continue
        subdirs[:] = [name for name in subdirs if name not in (".git", "node_modules")]
        if path.name == "evals" and "evals.json" in files:
            yield path / "evals.json"


suites = {}
affected_skills = set()
for changed in dict.fromkeys(manifest.read_text(encoding="utf-8-sig").splitlines()):
    if not changed:
        continue
    path = Path(os.path.abspath(repo / changed))
    if Path(changed).parts[-2:] == ("evals", "evals.json"):
        suites[path] = changed
    if path.is_relative_to(repo) and path.resolve().is_relative_to(repo):
        owner = owning_skill(path)
        if owner is not None:
            affected_skills.add(owner)

for skill in sorted(affected_skills):
    for path in sorted(skill_suites(skill)):
        suites.setdefault(path, path.relative_to(repo).as_posix())

failed = False
for source, changed in suites.items():
    if not source.resolve().is_relative_to(repo):
        print(f"[ERROR] {changed}: evals.json must be inside the repository", file=sys.stderr)
        failed = True
        continue
    if not source.is_file():  # Deleted suites are outside the PR check.
        continue
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"[ERROR] {changed}: could not read valid JSON", file=sys.stderr)
        failed = True
        continue
    for error in iter_pr_evals_errors(data):
        print(f"[ERROR] {changed}: {error}", file=sys.stderr)
        failed = True
sys.exit(int(failed))
PY
    then
        print_fail "S-EVAL-01: invalid evaluation input in changed files"
        fail_count=$((fail_count + 1))
    fi
fi

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
