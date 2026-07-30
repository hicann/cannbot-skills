# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate every tag→KB path cited in SKILL.md §Step 2 tables exists in plugin kb/.

Run after SKILL.md edits to ensure the curated tables don't reference dead anchors.
Exit 0 if all paths exist, 2 if any missing.

Usage:
    python3 skills/aog-op-classify/validate_kb_paths.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[2]
_REFS = _PLUGIN_ROOT / "kb"
# P88 (2026-05-15) KB reorg: all AscendC KB files moved under target/ascendc/.
# This mirrors the canonical resolution in
# src/scripts/orchestrator/briefs/op_taxonomy.py (_CANONICAL_KB_PATHS).
_ASCENDC = _REFS / "target" / "ascendc"
_SKILL_MD = _HERE.parent / "SKILL.md"


def extract_kb_paths(skill_md_text: str) -> list[str]:
    """Pull every `OL-XXX` / `PB-XXX` / `EC-XXX` / `P-PXX` / `CAND-PPXX` reference
    out of the SKILL.md tag→KB tables. Also pull bare path references like
    `patterns/domains/memory_access.md`.
    """
    paths: set[str] = set()
    # OL/PB/EC/P-P/CAND-PP anchor refs
    for m in re.finditer(r"(OL-\d+|PB-\d+|EC-\d+|P-P\d+|CAND-PP\d+)", skill_md_text):
        paths.add(m.group(1))
    # Bare relative paths (foo/bar.md)
    for m in re.finditer(r"`(patterns/[a-zA-Z0-9_/.\-]+\.md(?:#[A-Z0-9\-]+)?)`", skill_md_text):
        paths.add(m.group(1))
    return sorted(paths)


def verify_anchor_exists(anchor: str) -> tuple[bool, str]:
    """For OL-XXX → grep OPERATIONAL_KNOWLEDGE.md.
    For PB-XXX → grep PLATFORM_BUGS.md.
    For EC-XXX → grep ERROR_CORRECTIONS.md.
    For P-PXX → grep patterns/PATTERN_INDEX.md.
    For CAND-PPXX → grep patterns/unverified/candidates.md.
    For bare paths (patterns/...) → check file exists; if anchor present, grep file for #anchor.
    """
    if anchor.startswith("OL-"):
        target = _ASCENDC / "OPERATIONAL_KNOWLEDGE.md"
        pattern = f"## {anchor}"
    elif anchor.startswith("PB-"):
        target = _ASCENDC / "PLATFORM_BUGS.md"
        pattern = f"### {anchor}"
    elif anchor.startswith("EC-"):
        target = _ASCENDC / "ERROR_CORRECTIONS.md"
        pattern = f"### {anchor}"
    elif anchor.startswith("P-P"):
        target = _ASCENDC / "patterns" / "PATTERN_INDEX.md"
        pattern = f"| {anchor} "
    elif anchor.startswith("CAND-PP"):
        target = _ASCENDC / "patterns" / "unverified" / "candidates.md"
        pattern = anchor
    elif "/" in anchor:  # bare path like patterns/domains/foo.md
        path_part, _, _ = anchor.partition("#")
        # SKILL.md cites bare paths relative to the AscendC KB root
        # (e.g. `patterns/domains/sort.md`); some rows cite repo-relative
        # paths already prefixed with target/ascendc/. Try both.
        for base in (_ASCENDC, _REFS):
            target = base / path_part
            if target.exists():
                return True, str(target)
        return False, f"file not found under target/ascendc or plugin kb/: {path_part}"
    else:
        return False, f"unrecognized anchor format: {anchor}"

    if not target.exists():
        return False, f"target file missing: {target}"
    text = target.read_text()
    if pattern in text:
        return True, str(target)
    return False, f"anchor `{pattern}` not found in {target.name}"


def main() -> int:
    if not _SKILL_MD.exists():
        print(f"ERROR: SKILL.md not found at {_SKILL_MD}", file=sys.stderr)
        return 2
    text = _SKILL_MD.read_text()
    anchors = extract_kb_paths(text)
    print(f"Found {len(anchors)} unique KB references in SKILL.md")
    failures: list[tuple[str, str]] = []
    for anchor in anchors:
        ok, detail = verify_anchor_exists(anchor)
        marker = "✓" if ok else "✗"
        print(f"  {marker} {anchor:<25} {detail}")
        if not ok:
            failures.append((anchor, detail))
    if failures:
        print(f"\n{len(failures)} broken references:")
        for anchor, detail in failures:
            print(f"  - {anchor}: {detail}")
        return 2
    print("\nAll references valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
