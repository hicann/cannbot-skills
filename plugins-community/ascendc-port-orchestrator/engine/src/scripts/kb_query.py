#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""KB query CLI — tag-aware grep wrapper (NODE-21 Phase A, 2026-05-28).

Usage:
    # Filter by arch_family + keyword
    python3 src/scripts/kb_query.py --arch=arch35 --keyword="WholeReduceMax"

    # Filter by SoC
    python3 src/scripts/kb_query.py --soc=Ascend950PR_9579

    # JSON output for brief construction
    python3 src/scripts/kb_query.py --arch=arch35 --keyword="DataCopy" --json

    # Show all entries for a target (no keyword filter)
    python3 src/scripts/kb_query.py --arch=arch35 --soc=Ascend950PR_9579

    # Show related entries (loosen one tier at a time)
    python3 src/scripts/kb_query.py --arch=arch35 --keyword="softmax" --show-related

Behavior:
    1. Load KB_INDEX.md, parse rows with optional tag column
    2. Apply target filter (--arch, --soc, --npu-arch, --cann, --bisheng,
       --dtype, --op-class, --paradigm)
    3. Within filtered subset, grep keyword across linked file anchor sections
    4. Return matching rows + brief content excerpt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# KB_INDEX.md lives at <plugin_root>/kb/ (2026-07-05 relocation).
# kb_query.py is at <plugin_root>/engine/src/scripts/; _PROJECT_ROOT == engine/,
# so _PROJECT_ROOT.parent == <plugin_root>.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_KB_INDEX = _PROJECT_ROOT.parent / "kb" / "KB_INDEX.md"

# Tier → CLI flag mapping
_CLI_TIERS = {
    "paradigm": "paradigm",
    "arch_family": "arch",
    "soc": "soc",
    "npu_arch": "npu_arch",
    "cann": "cann",
    "bisheng": "bisheng",
    "dtype": "dtype",
    "op_class": "op_class",
}

# KB_INDEX row pattern: | [ID](path#anchor) | one-liner | [tags] | L[123] |
# Tag column is optional (backward compat).
_ROW_RE = re.compile(
    r"^\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|"   # [ID](path)
    r"\s*(.+?)\s*\|"                          # one-liner
    r"\s*(.*?)\s*\|"                          # optional tag column
    r"\s*(L[123])\s*\|"                       # level
)

# Simple anchor extraction: find the heading that matches the anchor
_ANCHOR_RE = re.compile(r"^#{1,4}\s+")


def _parse_kb_index(path: Path) -> list[dict]:
    """Parse KB_INDEX.md into a list of row dicts via the shared
    kb_schema.parse_kb_index_row (tolerates 3-col AND 4-col rows — the
    legacy 4-col-only regex dropped every real 3-col row → recall returned 0).
    """
    if not path.is_file():
        return []
    from kb_schema import parse_kb_index_row
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        parsed = parse_kb_index_row(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _resolve_path(file_path: str) -> Optional[Path]:
    """Resolve a relative path from KB_INDEX to a full filesystem path."""
    kb_dir = _KB_INDEX.parent
    candidate = kb_dir / file_path
    if candidate.is_file():
        return candidate
    # Try relative to project root
    candidate = _PROJECT_ROOT / file_path
    if candidate.is_file():
        return candidate
    return None


def _extract_section(file_path: Path, anchor: str, max_chars: int = 500) -> str:
    """Extract the content of a markdown anchor section.

    Reads from the heading matching `anchor` until the next heading of
    equal or higher level. Returns up to max_chars characters.
    """
    if not file_path.is_file():
        return ""
    text = file_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Find the anchor heading
    anchor_pattern = re.compile(rf"^#+\s+{re.escape(anchor)}\s*$", re.IGNORECASE)
    start = -1
    heading_level = 0
    for i, line in enumerate(lines):
        m = anchor_pattern.match(line.strip())
        if m:
            start = i
            heading_level = line.count("#", 0, 6)
            break

    if start < 0:
        return ""

    # Collect content until next heading of same or higher level
    content_lines = []
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if _ANCHOR_RE.match(line):
            level = line.count("#", 0, 6)
            if level <= heading_level:
                break
        content_lines.append(line)

    content = "\n".join(content_lines).strip()
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    return content


def query(
    *,
    target: Optional[dict] = None,
    keyword: Optional[str] = None,
    show_related: bool = False,
) -> list[dict]:
    """Run a tag-aware KB query.

    Args:
        target: target env dict (e.g. {"arch_family": "arch35"})
        keyword: optional keyword to grep in matched entries
        show_related: if True, also include entries that are one tier
            looser than the target (to surface potential over-narrow tagging)

    Returns:
        list of matching entry dicts with keys:
        id, path, anchor, hook, tags, level, excerpt (first 500 chars of
        the anchor section, if keyword matched).
    """
    from kb_schema import kb_entry_applies, sort_by_specificity

    if target is None:
        target = {}

    all_rows = _parse_kb_index(_KB_INDEX)

    # Filter by target
    applicable = []
    for row in all_rows:
        if kb_entry_applies(row["tags"], target):
            applicable.append(row)
        elif show_related:
            # Loosen one tier at a time, most-specific first
            for tier in ("npu_arch", "soc", "arch_family", "paradigm"):
                looser = dict(target)
                looser.pop(tier, None)
                if looser != target and kb_entry_applies(row["tags"], looser):
                    row["related_via_loose"] = tier
                    applicable.append(row)
                    break

    if not applicable:
        return []

    # Filter by keyword — uses alias_match so a query for any name variant
    # (V300 / arch35 / Ascend950PR) recalls entries written under any other
    # variant (2026-05-28 name-mismatch recall fix). Word-boundary matching
    # inside alias_match prevents the V300/V300x cross-chip collision.
    from kb_schema import alias_match
    results = []
    for row in applicable:
        if keyword:
            excerpt = ""
            match = False
            search_text = f"{row['id']} {row['hook']} {row.get('related_via_loose', '')}"
            if alias_match(keyword, search_text):
                match = True
            else:
                # Look in the linked file's anchor section
                resolved = _resolve_path(row["path"])
                if resolved and row.get("anchor"):
                    excerpt = _extract_section(resolved, row["anchor"])
                    if alias_match(keyword, excerpt):
                        match = True
            if not match:
                continue
        else:
            resolved = _resolve_path(row["path"])
            excerpt = _extract_section(resolved, row["anchor"]) if resolved and row.get("anchor") else ""

        results.append({
            "id": row["id"],
            "path": row["path"],
            "anchor": row.get("anchor", ""),
            "hook": row["hook"],
            "tags": row["tags"],
            "level": row.get("level", "L0"),
            "excerpt": excerpt[:500] if excerpt else "",
        })

    # Sort by specificity (most specific first)
    tagged = [(r["tags"], r) for r in results]
    return [r for _, r in sort_by_specificity(tagged)]


def _build_target(args: argparse.Namespace) -> dict[str, str]:
    """Build target dict from CLI args."""
    target: dict[str, str] = {}
    for tier, cli_flag in _CLI_TIERS.items():
        val = getattr(args, cli_flag.replace("-", "_"), None)
        if val:
            target[tier] = val
    return target


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KB query CLI — tag-aware grep wrapper (NODE-21 Phase A)",
    )
    ap.add_argument("--paradigm", help="T0 filter: ascendc/any")
    ap.add_argument("--arch", dest="arch", help="T1 filter: arch22/arch35/arch38/any")
    ap.add_argument("--soc", help="T2 filter: Ascend910_9382/Ascend950PR_9579/...")
    ap.add_argument("--npu-arch", dest="npu_arch", help="T3 filter: 3003/3113/3510/5102/...")
    ap.add_argument("--cann", help="CANN version filter")
    ap.add_argument("--bisheng", help="Bisheng version filter")
    ap.add_argument("--dtype", help="dtype filter: fp16/fp32/bf16/...")
    ap.add_argument("--op-class", dest="op_class", help="op_class filter: matmul/elementwise/...")
    ap.add_argument("--keyword", "-k", help="Keyword to grep in matched entries")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--show-related", action="store_true",
                    help="Loosen one tier to surface over-narrow tagging")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Include full excerpt (up to 2KB) instead of truncated")

    args = ap.parse_args()
    target = _build_target(args)

    if not target:
        ap.error("At least one filter flag is required (--arch, --soc, etc.)")

    results = query(target=target, keyword=args.keyword, show_related=args.show_related)

    if args.json:
        output = []
        for r in results:
            d = dict(r)
            d["tags"] = {k: v for k, v in d["tags"].items()}
            if args.verbose:
                pass  # keep full excerpt
            output.append(d)
        json.dump(output, sys.stdout, indent=2, default=str)
        print()
        return

    if not results:
        target_str = ", ".join(f"{k}={v}" for k, v in target.items())
        print(f"No KB entries matched target ({target_str})" +
              (f" + keyword={args.keyword!r}" if args.keyword else ""))
        return

    target_str = ", ".join(f"{k}={v}" for k, v in target.items())
    kw_str = f", keyword={args.keyword!r}" if args.keyword else ""
    print(f"KB entries for target=({target_str}){kw_str}:")
    print("-" * 70)

    for r in results:
        from kb_schema import format_tag_column
        tag_str = format_tag_column(r["tags"])
        related = f" [related via loose {r.get('related_via_loose', '')}]" if r.get("related_via_loose") else ""
        print(f"{r['id']} ({tag_str}){related} — {r['hook']}")
        print(f"  → {r['path']}#{r['anchor']}")
        if r["excerpt"]:
            max_chars = 2000 if args.verbose else 300
            ex = r["excerpt"][:max_chars]
            for line in ex.split("\n")[:5]:
                print(f"    {line[:120]}")
        print()


if __name__ == "__main__":
    main()
