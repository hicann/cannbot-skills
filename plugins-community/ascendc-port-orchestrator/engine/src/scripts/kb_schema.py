# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""KB Tag+Index Taxonomy Schema — NODE-21 Phase A, 2026-05-28.

Per docs/design/KB_TAG_INDEX_TAXONOMY_DESIGN.md:
  4-tier taxonomy (T0 paradigm / T1 arch_family / T2 soc / T3 npu_arch)
  + optional cross-cutting dimensions (cann / bisheng / dtype / op_class).

Provides:
  - parse_applies_to(): parse a KB entry's applies_to block into a dict
  - kb_entry_applies(): deterministic filter — entry applies to target?
  - specificity_score(): rank entries by scope specificity
  - VALID_VALUES: known-good values per tier (informational, not enforced)
"""
from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Known-good tier values (informational catalog, not enforcement)
# ---------------------------------------------------------------------------

VALID_VALUES: dict[str, set[str]] = {
    "paradigm": {"ascendc", "any"},
    "arch_family": {"arch22", "arch35", "arch38", "any"},
    "soc": {
        "Ascend910_9382", "Ascend910_9392", "Ascend910B1", "Ascend910B2",
        "Ascend950PR", "Ascend950PR_9579", "Ascend950PR_9589",
        "Atlas200I_A2", "Atlas500_A2",
        "any",
    },
    "npu_arch": {"3003", "3113", "3510", "5102", "any"},
    "cann": set(),   # freeform version strings
    "bisheng": set(),
    "dtype": {"fp16", "fp32", "bf16", "int8", "int32", "int64", "any"},
    "op_class": {
        "matmul", "elementwise", "reduction", "softmax", "attention",
        "quant-matmul", "attention-fwd", "attention-bwd", "moe", "norm",
        "any",
    },
}

# Ordered from least to most specific for precedence ranking.
_TIER_SPECIFICITY_ORDER = ("paradigm", "arch_family", "soc", "npu_arch",
                           "cann", "bisheng", "dtype", "op_class")


def parse_applies_to(text: str) -> dict[str, str]:
    """Parse a KB entry's applies_to YAML block into a flat {tier: value} dict.

    Expected input (one or more lines from a markdown code block):
        paradigm: ascendc
        arch_family: arch35
        npu_arch: any
        cann: 9.0.0+

    Returns:
        dict with string values. Missing tiers are absent from the dict
        (absent = universal scope on that dimension per the fallback rule).
        Empty dict for empty/malformed input.
    """
    result: dict[str, str] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip().rstrip("+")  # strip trailing "+" (e.g. "9.0.0+")
        # Strip YAML-style inline comments: "value  # comment"
        if " #" in val:
            val = val.split(" #")[0].strip()
        if key and val:
            result[key] = val
    return result


def kb_entry_applies(entry_tags: dict[str, str], target: dict[str, str]) -> bool:
    """Deterministic filter: does this KB entry apply to the target?

    An entry applies if EVERY declared tag either:
      - matches the target's value exactly, OR
      - equals 'any', OR
      - is absent from the entry (absent = universal scope on that dimension).

    Args:
        entry_tags: parsed applies_to dict from the entry
        target: target environment dict (e.g. {"arch_family": "arch35", ...})

    Returns:
        True if the entry applies to the target.
    """
    for tier in _TIER_SPECIFICITY_ORDER:
        if tier not in entry_tags:
            continue  # absent = universal scope on this dimension
        if entry_tags[tier] == "any":
            continue
        if tier not in target:
            # Entry declares this tier but target doesn't — entry is
            # more specific than our knowledge; include it (conservative).
            continue
        if entry_tags[tier] != target[tier]:
            return False
    return True


def specificity_score(entry_tags: dict[str, str]) -> int:
    """Rank an entry by how specific its scope is.

    Higher score = more specific (more tiers declared with concrete values).
    'any' counts as 0 (universal on that dimension).
    Each concrete non-'any' tier contributes 1 point, weighted by tier depth.
    """
    score = 0
    weight = 1
    for tier in _TIER_SPECIFICITY_ORDER:
        val = entry_tags.get(tier)
        if val and val != "any":
            score += weight
        weight += 1
    return score


def sort_by_specificity(
    entries: list[tuple[dict[str, str], Any]],
) -> list[tuple[dict[str, str], Any]]:
    """Sort (tags, payload) pairs by specificity descending.

    More specific entries first — when two entries conflict, the more
    specific one should take precedence.
    """
    return sorted(entries, key=lambda x: specificity_score(x[0]), reverse=True)


def format_tag_column(entry_tags: dict[str, str]) -> str:
    """Format an entry's tags as a KB_INDEX tag-column string.

    Format: "paradigm=ascendc; arch_family=arch35; soc=Ascend950PR_9579"
    Multi-valued lists: "npu_arch=3510,5102"
    Semicolon-separated, no trailing semicolon.
    """
    parts = []
    for tier in _TIER_SPECIFICITY_ORDER:
        val = entry_tags.get(tier)
        if val and val != "any":
            parts.append(f"{tier}={val}")
    if not parts:
        parts.append("paradigm=any")
    return "; ".join(parts)


def parse_tag_column(text: str) -> dict[str, str]:
    """Parse a KB_INDEX tag column string back into a dict.

    Inverse of format_tag_column. Handles multi-valued lists
    (npu_arch=3510,5102) by taking only the first value.
    """
    result: dict[str, str] = {}
    for part in text.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip().lower()
        val = val.strip().split(",")[0].strip()  # take first value if multi
        if key and val:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Canonical alias families (2026-05-28) — the name-mismatch recall fix.
#
# Problem this solves: the SAME A5 knowledge is written under many names
# (arch35 / V351x / Ascend950PR / project-internal "V300" / __NPU_ARCH__ 3510).
# A worker grepping "V300" found a DIFFERENT subset of entries than one
# grepping "arch35" or "Ascend950PR" — name mismatch fragmented recall.
# Query-time alias expansion makes a query for ANY member of a family match
# entries written under ANY other member, so recall is name-invariant.
#
# CRITICAL collision guard: hiascend's "V300x" (Atlas 200I/500 A2) is a
# DIFFERENT chip and is NOT in the A5 family. The project-internal "V300"
# (== A5/arch35) IS. Callers MUST match expanded terms with WORD-BOUNDARY
# semantics (\bv300\b) so "v300" does not false-match "v300x". See
# alias_match() which enforces this.
# ---------------------------------------------------------------------------

_ALIAS_FAMILIES: list[set[str]] = [
    {  # A5 / arch35 family
        "a5", "arch35", "v351x", "v300",          # "v300" = project-internal A5 label (NOT hiascend V300x)
        "ascend950pr", "ascend950pr_9579", "ascend950pr_9589", "950pr",
        "3510", "5102",
    },
    {  # A3 / arch22 family
        "a3", "arch22", "v220",
        "ascend910c", "ascend910_9382", "ascend910_9392", "910c",
        "3003", "3113",
    },
]


def expand_aliases(term: str) -> set[str]:
    """Return all name variants of `term` within its canonical family.

    If `term` belongs to a known alias family, returns the whole family
    (lowercased). Otherwise returns just the lowercased term. Name-invariant
    recall: expand_aliases("V300") == expand_aliases("arch35") == the A5 set.
    """
    t = term.strip().lower()
    for fam in _ALIAS_FAMILIES:
        if t in fam:
            return set(fam)
    return {t}


def alias_match(term: str, text: str) -> bool:
    """True if `text` mentions `term` OR any of its canonical aliases, using
    WORD-BOUNDARY matching so short ambiguous tokens don't false-match
    (e.g. "v300" must not match "v300x" — the different-chip collision).

    Non-alias terms fall back to a plain case-insensitive substring test
    (preserves existing keyword behavior for normal keywords).
    """
    import re as _re
    variants = expand_aliases(term)
    if variants == {term.strip().lower()}:
        # not an alias term — plain substring (back-compat for normal keywords)
        return term.strip().lower() in text.lower()
    low = text.lower()
    for v in variants:
        # word-boundary match; \b around alphanumerics handles v300 vs v300x
        if _re.search(rf"(?<![a-z0-9]){_re.escape(v)}(?![a-z0-9])", low):
            return True
    return False


# ---------------------------------------------------------------------------
# KB_INDEX row parser (2026-05-28) — single source of truth, tolerates BOTH
# the legacy 3-column format `| [ID](path) | hook | L1 |` AND the 4-column
# tagged format `| [ID](path) | hook | tags | L1 |`. Previously kb_query and
# briefs/_common each had a regex that REQUIRED 4 columns, so every real
# (3-column) KB_INDEX row was silently dropped and recall returned ~0. This
# parser reads the format the KB_INDEX actually has.
# ---------------------------------------------------------------------------

def parse_kb_index_row(line: str) -> Optional[dict]:
    """Parse one KB_INDEX markdown table row. Returns dict with
    id / path / anchor / hook / tags(dict) / level / category, or None if the
    line is not a data row (header, separator, prose).

    Handles 3-col (no tag column → tags={}) and 4-col (tag column parsed).
    """
    import re as _re
    s = line.strip()
    if not s.startswith("|"):
        return None
    # cells between pipes
    cells = [c.strip() for c in s.strip("|").split("|")]
    if len(cells) < 3:
        return None
    # first cell must be a [ID](path) link
    m = _re.match(r"\[([^\]]+)\]\(([^)]+)\)", cells[0])
    if not m:
        return None
    entry_id = m.group(1).strip()
    file_path = m.group(2).strip()
    # last cell is the level if it looks like L1/L2/L3; else no level
    level = ""
    tail = cells[-1]
    if _re.fullmatch(r"L[0-9]", tail):
        level = tail
        mid = cells[1:-1]          # cells between ID and level
    else:
        mid = cells[1:]            # no level column
    # mid[0] = hook; mid[1] (if present) = tag column
    hook = mid[0] if mid else ""
    tag_text = mid[1] if len(mid) >= 2 else ""
    tags = parse_tag_column(tag_text) if tag_text and "=" in tag_text else {}
    anchor = ""
    if "#" in file_path:
        file_path, _, anchor = file_path.partition("#")
    # category from ID prefix
    cat = ""
    if entry_id.startswith("P-P") or entry_id.startswith("P-"):
        cat = "P-P"
    elif entry_id.startswith("CAND"):
        cat = "CAND"
    elif entry_id[:2] in ("OL", "EC", "PB"):
        cat = entry_id[:2]
    elif entry_id.startswith("DEBT"):
        cat = "DEBT"
    return {"id": entry_id, "path": file_path, "anchor": anchor,
            "hook": hook, "tags": tags, "level": level or "L0", "category": cat}
