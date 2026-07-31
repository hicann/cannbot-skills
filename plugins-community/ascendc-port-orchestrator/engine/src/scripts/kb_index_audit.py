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
"""KB_INDEX orphan audit — detect any entry in KB files (EC/PB/OL/P-P/CAND)
that does not have a corresponding row in KB_INDEX.md.

Per user direction 2026-05-22T09:22Z: 'kb 可能出现严重 orphon 问题，
你们明明有 kb pr 的协议，为什么没有人遵守？你那里pr merge的时候为什么
不用kb maintianer扫一遍kb index？是否kb maintainer没有这个能力？
如果是，必须加上，否则kb index问题无法修复！'

Root cause this addresses: aog-knowledge-maintain Mode 1 step 4.5 (added
2026-05-16) syncs the index for NEW entries written during the
invocation, but does NOT audit pre-existing entries. Historical entries
created before that rule are orphan until someone runs a bulk-sync PR.

Usage:
  python3 src/scripts/kb_index_audit.py              # full audit, exit 1 if orphans
  python3 src/scripts/kb_index_audit.py --strict     # fail on any orphan (default)
  python3 src/scripts/kb_index_audit.py --report-only  # informational, exit 0 always
  python3 src/scripts/kb_index_audit.py --json        # machine-readable output

Integration:
  - Pre-commit hook: runs on KB-touching commits to block orphan addition
  - PR merge gate: orchestrator/CI runs this on merge requests
  - CLAUDE.md protocol: A5 main agent runs before merging KB-touching PR

Exit codes:
  0 — no orphans (or --report-only)
  1 — orphans detected in strict mode
  2 — usage / file-not-found error
"""
from __future__ import annotations
import logging
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Compute project root from script location
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent  # src/scripts/kb_index_audit.py → engine/
# 2026-07-05: KB relocated to <plugin_root>/kb/ (PROJECT_ROOT.parent == plugin_root).
KB_ROOT = PROJECT_ROOT.parent / "kb"

KB_DIRS = {
    "ascendc": KB_ROOT / "target" / "ascendc",
}
INDEX_PATH = KB_ROOT / "KB_INDEX.md"


@dataclass
class AuditResult:
    """One KB-file's audit outcome against KB_INDEX.md (produced by `audit_backend`).

    Attributes:
        backend: KB backend name (``ascendc``).
        file_type: entry class this row covers — ``EC`` / ``PB`` / ``OL`` /
            ``P-P`` / ``CAND``.
        file_path: absolute path of the canonical KB file audited.
        total_entries: distinct entry IDs found in the KB file.
        indexed_entries: how many of those also appear in KB_INDEX.md.
        orphans: entry IDs present in the KB file but MISSING from the index
            (the strict-blocking finding).
        duplicate_ids: ``[(id, [line_nums])]`` for IDs defined more than once.
        dangling_index_refs: IDs cited by KB_INDEX.md with no canonical entry.
        missing_applies_to: entry IDs lacking (or with an invalid) ``applies_to`` tag.
        id_sequence_violations: ``[(prefix, prev_id, outlier_id, jump)]`` for
            entry-header ids that jump wildly beyond the family's established
            sequence (the OL-984 picker/typo class — see ``check_id_sequence``).
    """
    backend: str
    file_type: str  # EC / PB / OL / P-P / CAND
    file_path: str
    total_entries: int
    indexed_entries: int
    orphans: list[str] = field(default_factory=list)
    duplicate_ids: list[tuple] = field(default_factory=list)  # [(id, [line_nums])]
    dangling_index_refs: list[str] = field(default_factory=list)
    missing_applies_to: list[str] = field(default_factory=list)
    id_sequence_violations: list[tuple] = field(default_factory=list)  # [(prefix, prev, outlier, jump)]


def extract_entries_with_lines(path: Path, regex: str) -> list[tuple[str, int]]:
    """Like extract_entries_<x> but returns [(id, line_no), ...] for duplicate
    detection. Useful for any 'must be unique' KB entry type.
    """
    if not path.is_file():
        return []
    out: list[tuple[str, int]] = []
    pattern = re.compile(regex)
    for i, line in enumerate(path.read_text().splitlines(), 1):
        m = pattern.match(line)
        if m:
            out.append((m.group(1), i))
    return out


def find_duplicates(entries_with_lines: list[tuple[str, int]]) -> list[tuple]:
    """Return [(id, [line_nums])] for IDs appearing more than once."""
    from collections import defaultdict
    by_id: dict[str, list[int]] = defaultdict(list)
    for entry_id, line in entries_with_lines:
        by_id[entry_id].append(line)
    return [(eid, lines) for eid, lines in sorted(by_id.items()) if len(lines) > 1]


def check_applies_to(path: Path, entries_with_lines: list[tuple[str, int]]) -> list[str]:
    """Check if each entry has an `applies_to:` line within ~15 lines after the
    heading. Returns list of IDs missing applies_to per P0aax (2026-05-07).

    NODE-21 Phase E (2026-05-28): also validates tier values against
    kb_schema.VALID_VALUES when the file is importable. Invalid values are
    appended to the returned list as "ID (invalid: tier=value)".
    """
    if not path.is_file() or not entries_with_lines:
        return []
    lines = path.read_text().splitlines()
    missing: list[str] = []
    for entry_id, start in entries_with_lines:
        # Look at the next 15 lines after the heading for an applies_to: marker
        window = lines[start:start + 15]
        if not any("applies_to" in l.lower() for l in window):
            missing.append(f"{entry_id} (missing applies_to)")
            continue
        # NODE-21 Phase E: validate tier values
        for line in window:
            if not re.match(r"^\s*\w+\s*:", line):
                continue
            if "applies_to" in line.lower():
                continue  # the applies_to header itself
            m = re.match(r"^\s*(\w+)\s*:\s*(.+)$", line)
            if not m:
                continue
            tier, value = m.group(1).strip().lower(), m.group(2).strip()
            # Try to validate against VALID_VALUES
            try:
                from kb_schema import VALID_VALUES as _vv
                valid_set = _vv.get(tier)
                if valid_set and value not in valid_set:
                    recognized = sorted(valid_set)[:6]
                    missing.append(
                        f"{entry_id} (invalid {tier}={value!r}, "
                        f"recognized: {recognized})"
                    )
            except ImportError:
                pass  # kb_schema not importable — skip validation
    return missing


# ---------------------------------------------------------------------------
# SoC scope-consistency lint (2026-07-17)
# ---------------------------------------------------------------------------
# Invariant: every SoC an entry claims POSITIVE evidence on (`verified_on:` /
# `confirmed_on:`) must be inside that entry's declared `applies_to: soc=` set.
#
# Why it must be mechanical: PB-35 declared `applies_to: soc=Ascend910_9382`
# (V220 only) while its own `confirmed_on:` recorded a CONFIRMED reproduction on
# Ascend950PR_9579 (V351/A5). Harmless while composers hardcode injection — but
# DEBT-208 makes composers HONOR `applies_to: soc=`, at which point the entry is
# SUPPRESSED on the one SoC where it is confirmed. A fix for an over-block would
# silently become an under-block. Presence of applies_to != correctness of it.
#
# Deliberately NOT checked:
#   - `unverified_on:` / `verified_does_not_reproduce_on:` /
#     `verified_does_not_apply_on:` — these are NEGATIVE-evidence fields whose
#     whole job is to name SoCs OUTSIDE applies_to. Flagging them would invert
#     the invariant. (The KB keeps a distinct vocabulary for negative evidence,
#     which is what makes verified_on/confirmed_on unambiguously positive.)
#   - SoC names appearing in FREE PROSE inside an evidence line. Empirically
#     (738-entry survey, 2026-07-17) prose mentions are provenance, not claims:
#     CAND-POOL-LAYOUT-BRIDGE says `verified_on: adaptive_avg_pool3d (V220->A5
#     L1 port)` — a port DIRECTION, verified on A5; and
#     CAND-V351-arch35-RegBase's verified_on names V220 only inside the
#     cross-ref ID `CAND-V220-to-V351-PortPattern-CubeVecFusedOp`. Both are
#     correct entries; a prose scan red-flags them. A lint that fires on honest
#     entries is worse than no lint, so we read a SoC from an evidence line only
#     in an IDENTIFYING position (see _evidence_socs).

# Any SoC-naming token: Ascend9xx family names, or the V220/V351 arch shorthands.
_SOC_TOKEN_RE = re.compile(r"\b(Ascend9\d{2}[A-Za-z0-9_]*|V220|V351)\b")

# Positive-evidence fields only. Anchored + explicit alternation so the
# negative-evidence fields (`verified_does_not_reproduce_on:`) never match.
_EVIDENCE_FIELD_RE = re.compile(r"^`?(verified_on|confirmed_on)\s*:\s*(.*?)`?\s*$")
_APPLIES_FIELD_RE = re.compile(r"^`?applies_to\s*:\s*(.*?)`?\s*$")

# Entry headings across every supported KB file class.
_ANY_ENTRY_HEAD_RE = re.compile(
    r"^#{2,4}\s+((?:EC|PB|OL|P-P|F-P|F-AP|CAND)[-A-Za-z0-9_]*)\b"
)


def _soc_family(token: str) -> str | None:
    """Normalize a SoC token to its coarse arch family (`V220` / `V351`).

    Family granularity is the only defensible comparison level: the KB writes
    `soc=` values as free text with ~20 distinct spellings for two families
    (`Ascend910_9382`, `Ascend910_V220`, `Ascend910C`, `Ascend910B4` all = V220;
    `Ascend950PR`, `Ascend950PR_9579`, `Ascend950PR_957b`, `V351` all = V351).
    Exact-token comparison would false-positive on every entry that declares
    `soc=Ascend950PR` and verifies on `Ascend950PR_9579`.
    """
    t = token.lower()
    if "950" in t or "v351" in t:
        return "V351"
    if "910" in t or "v220" in t:
        return "V220"
    return None


def _families_in(text: str) -> set[str]:
    """Every SoC family named anywhere in `text`."""
    return {f for f in (_soc_family(t) for t in _SOC_TOKEN_RE.findall(text)) if f}


def _applies_to_socs(value: str) -> tuple[set[str] | None, str | None]:
    """Parse the `soc=` clause of an applies_to line into a family set.

    Returns `({"*"}, raw)` for `soc=all` / `soc=any` (universal — never a
    violation), `(None, None)` when no `soc=` clause or no recognizable SoC
    token is present (unscoped — nothing to check against).
    """
    m = re.search(r"soc\s*=\s*([^;`]*)", value)
    if not m:
        return None, None
    raw = m.group(1).strip()
    if re.match(r"^(all|any)\b", raw, re.IGNORECASE):
        return {"*"}, raw
    return (_families_in(raw) or None), raw


def _evidence_socs(value: str) -> set[str]:
    """SoC families an evidence line CLAIMS, read only from an identifying position.

    Two identifying positions, both established KB conventions:
      1. a structured `soc=` clause  — `verified_on: soc=Ascend950PR; cann=9.0.0`
      2. a SoC token at the START of the value —
         `confirmed_on: Ascend950PR_9579 (V351 / A5) - kw-gb2 ...`

    Anything else (a SoC named mid-prose) is ignored — see the module note above.
    """
    m = re.search(r"soc\s*=\s*([^;`]*)", value)
    if m:
        return _families_in(m.group(1))
    lead = _SOC_TOKEN_RE.match(value.strip())
    if lead:
        f = _soc_family(lead.group(1))
        return {f} if f else set()
    return set()


def check_soc_scope(path: Path) -> list[tuple]:
    """Find entries whose verified_on/confirmed_on names a SoC applies_to excludes.

    Returns `[(entry_id, applies_raw, field, [out_of_scope_families], excerpt)]`.
    """
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    heads = [i for i, l in enumerate(lines) if _ANY_ENTRY_HEAD_RE.match(l)]
    out: list[tuple] = []
    for n, start in enumerate(heads):
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        entry_id = _ANY_ENTRY_HEAD_RE.match(lines[start]).group(1)
        applies: set[str] | None = None
        applies_raw: str | None = None
        evidence: list[tuple[str, str]] = []
        for line in lines[start:end]:
            s = line.strip()
            am = _APPLIES_FIELD_RE.match(s)
            if am and applies is None:
                applies, applies_raw = _applies_to_socs(am.group(1))
                continue
            em = _EVIDENCE_FIELD_RE.match(s)
            if em:
                evidence.append((em.group(1), em.group(2)))
        # No soc scope declared, or universal scope → nothing to contradict.
        if not applies or applies == {"*"}:
            continue
        for evidence_field, value in evidence:
            claimed = _evidence_socs(value)
            out_of_scope = claimed - applies
            if out_of_scope:
                out.append((entry_id, applies_raw, evidence_field,
                            sorted(out_of_scope), value[:110]))
    return out


# ---------------------------------------------------------------------------
# Domain-template arch-scope invariant (DEBT-222, 2026-07-17)
# ---------------------------------------------------------------------------
# Every domain template (patterns/domains/*.md) is an arch-scoped body a brief
# composer can hand to a worker. Owner-reported bug (2026-07-17): "kb 里这些
# template 连目标 soc 或者架构 tag 都没有" — a worker could receive a template for
# the wrong architecture because scope lived only as prose (or nowhere), not as a
# machine-checkable field a composer/audit enforces. This is the same defect
# class as PB-34's over-block (DEBT-208), one layer up: knowledge mis-delivered
# because its scope is not machine-readable.
#
# Invariant: every domain template MUST declare a machine-readable
# `applies_to: soc=...` in its header zone — either a concrete SoC set
# (soc=Ascend950PR) or an EXPLICIT soc=all for a genuinely arch-neutral doc.
# "Buried in a prose reference line" and "absent" both FAIL: an explicit neutral
# tag means the composer never has to GUESS whether an untagged file is neutral.
#
# The tag is read with the SAME `_applies_to_socs` parser the composer honors
# (briefs/kb_scope.py) and the SoC-scope lint uses — one parser, no drift.
_DOMAIN_TEMPLATES_DIR = KB_DIRS["ascendc"] / "patterns" / "domains"
_TEMPLATE_SCOPE_HEAD_LINES = 20  # frontmatter / top-of-file header zone


def _template_header_applies_to(lines: list[str]) -> str | None:
    """First machine-parseable `applies_to:` value in a template's header zone.

    Tolerates an optional leading blockquote `>` (a common KB convention — the
    FA and GMM templates carry their tag inside a top `>` block) so a prominent
    blockquoted tag still counts. A tag that appears only as bulleted PROSE
    (`- **applies_to**: ...`) or only deep in the body (past the header zone)
    does NOT match `_APPLIES_FIELD_RE` — that is exactly the "buried" state
    DEBT-222 rejects. Returns the raw value (post-`applies_to:`) or None.
    """
    for raw in lines[:_TEMPLATE_SCOPE_HEAD_LINES]:
        s = raw.strip()
        if s.startswith(">"):
            s = s[1:].strip()  # see through a top-of-file blockquote
        m = _APPLIES_FIELD_RE.match(s)
        if m:
            return m.group(1)
    return None


def check_domain_template_scope(domains_dir: Path = _DOMAIN_TEMPLATES_DIR) -> list[tuple]:
    """Flag domain templates lacking a machine-readable header `applies_to: soc=`.

    Returns `[(filename, reason)]`. A template passes iff its header zone carries
    an `applies_to:` line whose `soc=` clause parses to a concrete family set OR
    to universal (`soc=all`). Absent, buried-in-prose, or an unparseable `soc=`
    all fail — presence at a machine-readable location AND a valid scope value.
    """
    out: list[tuple] = []
    if not domains_dir.is_dir():
        return out
    for path in sorted(domains_dir.glob("*.md")):
        skip_current_item = False
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        val = _template_header_applies_to(lines)
        if val is None:
            out.append((path.name,
                        f"no machine-readable `applies_to: soc=` in the first "
                        f"{_TEMPLATE_SCOPE_HEAD_LINES} lines (absent or buried in prose)"))
            continue
        socs, _raw = _applies_to_socs(val)
        if socs is None:
            out.append((path.name,
                        f"`applies_to` present but its `soc=` clause is missing or "
                        f"names no recognizable SoC: {val!r} — use soc=<Ascend…> or soc=all"))
    return out


def extract_entries_ec(path: Path) -> list[str]:
    """Extract `### EC-N: title` lines from ERROR_CORRECTIONS.md."""
    if not path.is_file():
        return []
    return re.findall(r"^### (EC-\d+):", path.read_text(), re.MULTILINE)


def extract_entries_pb(path: Path) -> list[str]:
    """Extract `### PB-N: title` lines from PLATFORM_BUGS.md."""
    if not path.is_file():
        return []
    return re.findall(r"^### (PB-\d+):", path.read_text(), re.MULTILINE)


def extract_entries_ol(path: Path) -> list[str]:
    """Extract `## OL-N: title` lines from OPERATIONAL_KNOWLEDGE.md.
    Tolerates optional `[REVISED ...]` / `[ARCHIVED ...]` suffix between
    the ID and the colon (e.g., `## OL-83 [REVISED 2026-04-22]: title`).
    Robustness pass 2026-05-22 (task #68).
    """
    if not path.is_file():
        return []
    return re.findall(r"^## (OL-\d+)(?:\s*\[[^\]]+\])?:", path.read_text(), re.MULTILINE)


def extract_entries_pp(path: Path) -> list[str]:
    """Extract pattern IDs from PATTERN_INDEX.md (table rows + h2/h3 headers) +
    walk patterns/domains/*.md for h2/h3-headed P-P entries that don't appear in
    PATTERN_INDEX.md table form. Robustness pass 2026-05-22 (task #68).
    """
    found: set[str] = set()
    if path.is_file():
        text = path.read_text()
        # Form 1: table row `| P-P93 |`
        found.update(re.findall(r"^\|\s*(P-P\d+|F-P\d+|F-AP\d+)\s*\|", text, re.MULTILINE))
        # Form 2: h2/h3 header `## P-P93:` or `### P-P48:`
        found.update(re.findall(r"^#{2,3} (P-P\d+|F-P\d+|F-AP\d+):", text, re.MULTILINE))
    # Form 3: also walk patterns/domains/*.md for h2/h3-headed P-P entries
    domains_dir = path.parent / "domains"
    if domains_dir.is_dir():
        for domain_file in domains_dir.glob("*.md"):
            text = domain_file.read_text()
            found.update(re.findall(r"^#{2,3} (P-P\d+|F-P\d+|F-AP\d+):", text, re.MULTILINE))
    return sorted(found)


def find_pp_title_conflicts(pattern_index_path: Path) -> list[tuple]:
    """Detect a P-P ID used for TWO DIFFERENT patterns in PATTERN_INDEX.md: a
    `| P-PNNN | **table title** |` row AND a `## P-PNNN: heading title` whose titles
    DIFFER. The set-based extract_entries_pp collapses these to one ID, so the orphan/
    duplicate checks miss the collision (owner-caught 2026-06-29: P-P107/108 each had a
    swi_glu ## section + a top_k table-row = two patterns sharing a number).

    Same ID + SAME title (a normal index-row + its ## definition, e.g. P-P101) is NOT a
    conflict — only a title mismatch is. Returns [(id, table_title, heading_title), ...].
    """
    if not pattern_index_path.is_file():
        return []
    text = pattern_index_path.read_text()
    table = {}
    for m in re.finditer(r"^\|\s*(P-P\d+)\s*\|\s*\*\*(.+?)\*\*", text, re.MULTILINE):
        table.setdefault(m.group(1), m.group(2).strip())
    heading = {}
    for m in re.finditer(r"^## (P-P\d+):\s*(.+)$", text, re.MULTILINE):
        heading.setdefault(m.group(1), m.group(2).strip())
    conflicts = []
    for pid in sorted(set(table) & set(heading)):
        t, h = table[pid].lower(), heading[pid].lower()
        # same pattern if one title is a prefix of / shares the leading key-phrase
        key = min(len(t), len(h), 25)
        if t[:key] != h[:key]:
            conflicts.append((pid, table[pid][:55], heading[pid][:55]))
    return conflicts


def extract_entries_cand(path: Path) -> list[str]:
    """Extract `## CAND-XXX: title` OR `### CAND-XXX: title` lines from candidates.md.
    Both h2 and h3 are valid CAND headers in practice. Trailing
    identifier-allowed chars only.
    """
    if not path.is_file():
        return []
    return re.findall(r"^#{2,3} (CAND-[A-Za-z0-9_-]+)", path.read_text(), re.MULTILINE)


def extract_index_refs(text: str, *, prefixes: list[str]) -> set[str]:
    """Find IDs of given prefixes mentioned anywhere in KB_INDEX.md.

    Returns IDs verbatim as they appear in KB files, for example ``EC-13``
    and ``P-P90``. The KB-file extractors return the same form, so direct
    set comparison works without further normalization.
    """
    found: set[str] = set()
    for prefix in prefixes:
        # ID-format rules:
        #  - `EC`, `PB`, `OL` (no dash in prefix): natural form `EC-13`
        #  - `P-P`, `F-P`, `F-AP` (dash in prefix):
        #     natural form `P-P90` (digits directly after prefix)
        if "-" in prefix:
            pattern = rf"\b{re.escape(prefix)}(\d+)\b"
            id_fmt = f"{prefix}{{num}}"
        else:
            pattern = rf"\b{re.escape(prefix)}-(\d+)\b"
            id_fmt = f"{prefix}-{{num}}"
        for m in re.finditer(pattern, text):
            found.add(id_fmt.format(num=m.group(1)))
    return found


def extract_index_cand(text: str) -> set[str]:
    """Find CAND-XXX IDs mentioned in KB_INDEX.md."""
    return set(re.findall(r"CAND-[A-Za-z0-9_-]+", text))


# ---------------------------------------------------------------------------
# ID-sequence continuity check (2026-07-23, OL-984 picker/typo incident)
# ---------------------------------------------------------------------------
# The gap this closes: KB PR #237 nearly merged with a mis-numbered `OL-984`.
# The author's id-picker ran `grep -oE 'OL-[0-9]+' | sort | tail -1` and matched
# an `OL-983`-class id sitting in PROSE (a cross-reference), not a title — so it
# picked 984 as "highest+1" instead of the real next id (OL-280; existing OL
# titles ran OL-1..OL-279). The orphan/dup/dangling/applies_to checks all passed;
# only a manual index/content review caught it. This makes the MACHINE catch it.
#
# What "wildly out-of-sequence" means here — grounded in a full disk survey
# (2026-07-23) of every numeric single-prefix family (EC/PB/OL/P-P/F-P/F-AP):
# ids run essentially 1..max with only SMALL internal
# gaps (archived/renumbered entries). The LARGEST legitimate consecutive gap
# anywhere is 3 (P-P 13→16, OL 124→127). There are NO reserved ranges and NO
# large intentional forward-jumps. So a "must be exactly max+1" rule would be
# WRONG (it would false-positive on every legit archived-id gap) — but a large
# jump (the OL-984 case was 279→984, a jump of 705) is unambiguously a picker
# error, never an intentional gap.
#
# Rule: within each numeric family, sort the ENTRY-HEADER ids (parsed from the
# `### EC-N` / `## OL-N` headings by the extract_entries_* functions — NOT a
# blind prose grep, so this does not repeat the author's own mistake) and flag
# any id whose jump from the previous id EXCEEDS ID_SEQUENCE_MARGIN. The offender
# is itself the new max, so a naive "max+margin" test can't see it — the
# consecutive-jump form detects the outlier regardless.
#
# MARGIN=50 gives >16x headroom over the largest observed legit gap (3) while
# catching the 705-jump defect and even a moderate ~70 picker slip. It is a named
# constant so a future reserved-range convention can widen it deliberately.
#
# Default-FAIL (no opt-in flag), mirroring the SoC-scope / template-scope lints:
# the current clean KB has ZERO violations, so default-blocking cannot brick an
# honest commit — and a wildly mis-numbered id is a data defect, not a backlog
# item. CAND is deliberately excluded: its ids are NAMED (CAND-FA-COREDIST-1,
# CAND-PP74, CAND-A-P-NONALIGNED-DIVERGENCE), many independent sub-namespaces,
# not one monotonic sequence — a sequence rule is meaningless there.
ID_SEQUENCE_MARGIN = 50

# Split a canonical entry id into (alpha-prefix, trailing-int). Handles every
# family form: OL-280 → ("OL",280); TR-OL-18 → ("TR-OL",18); P-P93 → ("P-P",93);
# F-AP1 → ("F-AP",1). Grouping by this prefix keeps P-P / F-P / F-AP (which share
# one AuditResult) as independent sequences.
_ID_SPLIT_RE = re.compile(r"^(.*?)-?(\d+)$")


def _split_id(entry_id: str) -> tuple[str, int] | None:
    m = _ID_SPLIT_RE.match(entry_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def check_id_sequence(entries: list[str],
                      margin: int = ID_SEQUENCE_MARGIN) -> list[tuple]:
    """Flag entry-header ids that jump wildly beyond their family's sequence.

    `entries` are canonical ids already parsed from entry HEADINGS by the
    extract_entries_* functions (never from prose), so this cannot repeat the
    OL-984 author's prose-mismatch bug. Ids are grouped by alpha-prefix so a
    combined family (P-P / F-P / F-AP) is checked as separate sequences.

    Returns `[(prefix, prev_id, outlier_id, jump)]` sorted by prefix then id —
    one row per id that sits more than `margin` above the previous id in its
    sorted sequence (e.g. OL 279→984, jump 705). Small archived-id gaps
    (jump ≤ margin) are legitimate and never flagged.
    """
    from collections import defaultdict
    by_prefix: dict[str, set[int]] = defaultdict(set)
    for eid in entries:
        parsed = _split_id(eid)
        if parsed:
            by_prefix[parsed[0]].add(parsed[1])
    out: list[tuple] = []
    for prefix, nums in by_prefix.items():
        ordered = sorted(nums)
        for prev, cur in zip(ordered, ordered[1:]):
            if cur - prev > margin:
                out.append((prefix, prev, cur, cur - prev))
    return sorted(out)


def audit_backend(backend: str, kb_dir: Path, index_text: str) -> list[AuditResult]:
    """Audit one backend's KB files against KB_INDEX.md."""
    results: list[AuditResult] = []
    if not kb_dir.is_dir():
        return results

    if backend == "ascendc":
        ec_path = kb_dir / "ERROR_CORRECTIONS.md"
        pb_path = kb_dir / "PLATFORM_BUGS.md"
        ol_path = kb_dir / "OPERATIONAL_KNOWLEDGE.md"
        pp_path = kb_dir / "patterns/PATTERN_INDEX.md"
        cand_path = kb_dir / "patterns/unverified/candidates.md"

        index_ec = extract_index_refs(index_text, prefixes=["EC"])
        index_pb = extract_index_refs(index_text, prefixes=["PB"])
        index_ol = extract_index_refs(index_text, prefixes=["OL"])
        # P-P prefix matching: also match F-P and F-AP via pattern
        index_pp = (extract_index_refs(index_text, prefixes=["P-P"])
                    | extract_index_refs(index_text, prefixes=["F-P"])
                    | extract_index_refs(index_text, prefixes=["F-AP"]))
        index_cand = extract_index_cand(index_text)

        for ftype, path, entries, indexed, dup_regex, check_at in [
            ("EC", ec_path, extract_entries_ec(ec_path), index_ec, r"^### (EC-\d+):", True),
            ("PB", pb_path, extract_entries_pb(pb_path), index_pb, r"^### (PB-\d+):", True),
            ("OL", ol_path, extract_entries_ol(ol_path), index_ol, r"^## (OL-\d+):", True),
            ("P-P", pp_path, extract_entries_pp(pp_path), index_pp, None, False),
            ("CAND", cand_path, extract_entries_cand(cand_path), index_cand, r"^#{2,3} (CAND-[A-Za-z0-9_-]+)", False),
        ]:
            entries_set = set(entries)
            orphans = sorted(entries_set - indexed)
            # Anchor-missing (a.k.a. dangling index refs, task #58):
            # KB_INDEX.md references entry that doesn't exist in canonical KB file.
            # Reverse of orphan check.
            dangling = sorted(indexed - entries_set)
            # Duplicate detection: only for EC / PB / OL / CAND (P-P uses table form)
            dups: list[tuple] = []
            missing_at: list[str] = []
            if dup_regex:
                entries_with_lines = extract_entries_with_lines(path, dup_regex)
                dups = find_duplicates(entries_with_lines)
                if check_at:
                    missing_at = check_applies_to(path, entries_with_lines)
            # ID-sequence continuity: numeric single/grouped-prefix families only.
            # CAND is excluded — its ids are NAMED, not one monotonic sequence.
            id_seq = check_id_sequence(entries) if ftype != "CAND" else []
            results.append(AuditResult(
                backend=backend, file_type=ftype, file_path=str(path),
                total_entries=len(entries_set),
                indexed_entries=len(entries_set & indexed),
                orphans=orphans,
                duplicate_ids=dups,
                dangling_index_refs=dangling,
                missing_applies_to=missing_at,
                id_sequence_violations=id_seq,
            ))

    return results


def check_tag_pollution(index_path: Path) -> list[tuple]:
    """Detect KB_INDEX rows whose STRICT tag values contain free-text (space/paren).

    A strict tag value with prose (e.g. `soc=Ascend950PR (method verified` or
    `op_class=all DB chunk-loops`) SILENTLY fails the brief's exact-match env filter
    (kb_schema.kb_entry_applies) → the entry is NEVER RECALLED into worker/ko/fo briefs,
    defeating its purpose. The orphan audit does NOT catch this (presence != recallability).
    FIX: keep each strict tag value a CLEAN TOKEN (soc=Ascend950PR, op_class=all); prose
    belongs in the hook cell, not the tag value. (2026-07-03, ss-bwd graybox: 13 entries
    incl. CAND-DB-COARSE-FENCE / CAND-EXPOSED were silently unrecalled this way.)
    """
    if not index_path.is_file():
        return []
    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parent)
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from kb_schema import parse_kb_index_row
    except Exception:
        return []
    STRICT = {"soc", "cann", "arch_family", "op_class", "paradigm",
              "bisheng", "dtype", "arch", "kernel_type"}
    bad = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        row = parse_kb_index_row(line)
        if not row:
            continue
        for k, v in (row.get("tags") or {}).items():
            if k in STRICT and v and (" " in v or "(" in v or ")" in v):
                bad.append((str(row.get("id", "?")).split("#")[-1], k, v[:40]))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", default=True,
                    help="exit non-zero on any orphan (default)")
    ap.add_argument("--report-only", action="store_true",
                    help="informational only, exit 0 even if orphans found")
    ap.add_argument("--strict-duplicates", action="store_true",
                    help="also exit 1 on duplicate IDs (default: warn-only, "
                         "since semantic judgment is needed to resolve)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON report instead of text")
    ap.add_argument(
        "--strict-dangling",
        action="store_true",
        help=(
            "also exit 1 on dangling index refs (KB_INDEX cites an entry that "
            "does not exist in canonical KB). Default off because the existing "
            "CAND/P-P/OL extractors miss some valid header patterns (h3 CANDs, "
            "REVISED suffix OLs, patterns/domains/* P-Ps). Enable after extractor "
            "robustness pass. Task #58 follow-up tracks the cleanup."
        ),
    )
    ap.add_argument("--strict-applies-to", action="store_true",
                    help="NODE-21 Phase E (2026-05-28): exit 1 on entries missing applies_to tag. "
                         "Default off for backward compat; pre-commit hook enables this.")
    args = ap.parse_args()

    if not INDEX_PATH.is_file():
        print(f"ERROR: KB_INDEX.md not found at {INDEX_PATH}", file=sys.stderr)
        return 2

    index_text = INDEX_PATH.read_text()

    all_results: list[AuditResult] = []
    for backend, kb_dir in KB_DIRS.items():
        all_results.extend(audit_backend(backend, kb_dir, index_text))

    total_orphans = sum(len(r.orphans) for r in all_results)
    total_entries = sum(r.total_entries for r in all_results)
    total_duplicates = sum(len(r.duplicate_ids) for r in all_results)
    total_missing_at = sum(len(r.missing_applies_to) for r in all_results)
    total_dangling = sum(len(r.dangling_index_refs) for r in all_results)
    total_id_seq = sum(len(r.id_sequence_violations) for r in all_results)
    tag_pollution = check_tag_pollution(INDEX_PATH)
    # SoC scope-consistency: sweep every KB file in every backend subtree, not
    # just the five canonical registries — P-P bodies (patterns/domains/*.md),
    # fa_class/*.md and candidates.md all carry applies_to/verified_on too.
    soc_violations: list[tuple] = []
    for _kb_dir in KB_DIRS.values():
        if _kb_dir.is_dir():
            for _md in sorted(_kb_dir.rglob("*.md")):
                soc_violations += [(str(_md.relative_to(KB_ROOT.parent)),) + v
                                   for v in check_soc_scope(_md)]

    # DEBT-222: every domain template must carry a machine-readable header
    # applies_to: soc= (concrete soc set or explicit soc=all).
    template_scope_violations = check_domain_template_scope()

    if args.json:
        report = {
            "kb_index_path": str(INDEX_PATH),
            "total_entries": total_entries,
            "total_orphans": total_orphans,
            "total_duplicates": total_duplicates,
            "total_missing_applies_to": total_missing_at,
            "total_dangling_index_refs": total_dangling,
            "total_id_sequence_violations": total_id_seq,
            "id_sequence_violations": [
                {"backend": r.backend, "file_type": r.file_type,
                 "prefix": pfx, "prev_id": prev, "outlier_id": cur, "jump": jump}
                for r in all_results
                for pfx, prev, cur, jump in r.id_sequence_violations
            ],
            "total_soc_scope_violations": len(soc_violations),
            "soc_scope_violations": [
                {"file": f, "id": eid, "applies_to_soc": ap,
                 "field": fld, "out_of_scope": oos, "excerpt": ex}
                for f, eid, ap, fld, oos, ex in soc_violations
            ],
            "total_template_scope_violations": len(template_scope_violations),
            "template_scope_violations": [
                {"template": name, "reason": reason}
                for name, reason in template_scope_violations
            ],
            "results": [
                {
                    "backend": r.backend,
                    "file_type": r.file_type,
                    "file_path": r.file_path,
                    "total_entries": r.total_entries,
                    "indexed_entries": r.indexed_entries,
                    "orphans": r.orphans,
                    "orphan_count": len(r.orphans),
                    "duplicate_ids": [
                        {"id": eid, "line_nos": lines}
                        for eid, lines in r.duplicate_ids
                    ],
                    "dangling_index_refs": r.dangling_index_refs,
                    "missing_applies_to": r.missing_applies_to,
                    "id_sequence_violations": [
                        {"prefix": pfx, "prev_id": prev,
                         "outlier_id": cur, "jump": jump}
                        for pfx, prev, cur, jump in r.id_sequence_violations
                    ],
                }
                for r in all_results
            ],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"KB_INDEX multi-error audit — {INDEX_PATH.relative_to(KB_ROOT.parent)}")
        print(f"{'─' * 80}")
        print(f"{'backend':<10} {'type':<8} {'total':>6} {'indexed':>8} {'orphan':>7} "
              f"{'dup':>5} {'dangle':>7} {'no_at':>6}")
        for r in all_results:
            ok = (not r.orphans and not r.duplicate_ids and not r.dangling_index_refs
                  and not r.id_sequence_violations)
            indicator = "✓" if ok else "❌"
            print(f"{r.backend:<10} {r.file_type:<8} {r.total_entries:>6} "
                  f"{r.indexed_entries:>8} {len(r.orphans):>7} "
                  f"{len(r.duplicate_ids):>5} {len(r.dangling_index_refs):>7} "
                  f"{len(r.missing_applies_to):>6} {indicator}")
        print(f"{'─' * 80}")
        # P-P title-conflict check (same ID → two different patterns; the set-based
        # extraction's blind spot — owner-caught 2026-06-29 P-P107/108).
        pp_conflicts = []
        for _pi in KB_ROOT.rglob("PATTERN_INDEX.md"):
            pp_conflicts += find_pp_title_conflicts(_pi)
        print(f"TOTAL entries: {total_entries}; orphans: {total_orphans}; "
              f"duplicates: {total_duplicates}; dangling: {total_dangling}; "
              f"missing applies_to: {total_missing_at}; pp-id-conflicts: {len(pp_conflicts)}; "
              f"polluted-tags: {len(tag_pollution)}; soc-scope-violations: {len(soc_violations)}; "
              f"template-scope-violations: {len(template_scope_violations)}; "
              f"id-sequence-violations: {total_id_seq}")
        if total_id_seq:
            print(f"\n❌ ID-SEQUENCE OUT-OF-RANGE ({total_id_seq}): an entry-header id jumps "
                  f"more than {ID_SEQUENCE_MARGIN} beyond its family's sequence — a picker/typo "
                  f"error (the OL-984 class), NOT an intentional archived-id gap.")
            for r in all_results:
                for pfx, prev, cur, jump in r.id_sequence_violations:
                    print(f"    - {r.backend}/{r.file_type}: {pfx}-{cur} sits {jump} above "
                          f"{pfx}-{prev} (prior id in sequence)")
            print("FIX: the id was mis-picked (e.g. an id-picker matched a cross-ref in PROSE,")
            print("  not a title). Renumber the entry to the real next id (max title id + 1),")
            print("  and update any cross-references. Parse ENTRY HEADERS to pick the next id,")
            print("  never a blind `grep 'OL-[0-9]+' | tail -1` (that matched OL-983 in prose).")
        if template_scope_violations:
            print(f"\n❌ DOMAIN-TEMPLATE ARCH-SCOPE ({len(template_scope_violations)}): a "
                  f"patterns/domains/*.md template lacks a machine-readable header "
                  f"`applies_to: soc=` (DEBT-222).")
            for name, reason in template_scope_violations:
                print(f"    - {name}: {reason}")
            print("FIX: add an `applies_to: soc=<Ascend…>` (or explicit `soc=all` for a")
            print("  genuinely arch-neutral doc) to the template's frontmatter / header zone.")
            print("WHY THIS BLOCKS: an arch-fixed template with no machine-readable scope can")
            print("  be handed to a worker for the wrong architecture — the same mis-delivery")
            print("  class as DEBT-208, now guarded at the template layer. An explicit soc=all")
            print("  means the composer never has to GUESS whether an untagged file is neutral.")
        if soc_violations:
            print(f"\n❌ SoC SCOPE CONTRADICTION ({len(soc_violations)}): an entry claims POSITIVE "
                  f"evidence (verified_on/confirmed_on) on a SoC its own applies_to EXCLUDES.")
            for f, eid, ap, fld, oos, ex in soc_violations:
                print(f"    - {eid} ({f})")
                print(f"        applies_to: soc={ap}  → covers {sorted(_families_in(ap or ''))}")
                print(f"        {fld}: names {oos} — OUTSIDE the declared scope")
                print(f"        {fld} excerpt: {ex}")
            print("FIX: the two fields disagree — decide which is true and make them agree.")
            print("  (a) evidence is right → WIDEN applies_to's soc= to include that SoC")
            print("      (comma-separated clean tokens: soc=Ascend910_9382,Ascend950PR_9579);")
            print("  (b) evidence is mis-scoped → move it to unverified_on /")
            print("      verified_does_not_reproduce_on (the negative-evidence fields,")
            print("      which this lint deliberately does NOT check).")
            print("WHY THIS BLOCKS: once brief composers honor applies_to: soc= (DEBT-208),")
            print("an entry confirmed on a SoC its applies_to excludes is SILENTLY SUPPRESSED")
            print("on the one SoC where it is proven. Presence of applies_to != correctness.")
        if tag_pollution:
            print(f"\n❌ POLLUTED tag values ({len(tag_pollution)}): a STRICT tag value has free-text "
                  f"(space/paren) → SILENTLY not recalled by briefs (kb_entry_applies exact-match).")
            for eid, k, v in tag_pollution[:30]:
                print(f"    - {eid}: {k}={v!r}")
            if len(tag_pollution) > 30:
                print(f"    ... ({len(tag_pollution) - 30} more)")
            print("FIX: keep each strict tag value a CLEAN TOKEN (soc=Ascend950PR, op_class=all);"
                  " move descriptive prose into the hook cell. Presence != recallability.")
        if pp_conflicts:
            print("\n❌ P-P ID CONFLICTS (one number used for TWO different patterns):")
            for pid, tt, ht in pp_conflicts:
                print(f"    - {pid}: table='{tt}' vs ##section='{ht}'")
            print("FIX: renumber the unregistered (not-in-KB_INDEX) one to next free P-P + update cross-refs.")
        if total_duplicates:
            print("\n❌ DUPLICATE IDs (same ID used for different entries — cross-refs ambiguous):")
            for r in all_results:
                if r.duplicate_ids:
                    print(f"  {r.backend}/{r.file_type} ({len(r.duplicate_ids)}):")
                    for eid, line_nos in r.duplicate_ids:
                        print(f"    - {eid} at lines {line_nos}")
            print("\nFIX: rename the later/colliding entry to next available ID;")
            print("     update cross-refs in other KB files + workspace/.")
        if total_orphans:
            print("\n❌ ORPHAN entries (in KB file but not in KB_INDEX):")
            for r in all_results:
                if r.orphans:
                    print(f"  {r.backend}/{r.file_type} ({len(r.orphans)}):")
                    for entry in r.orphans[:50]:
                        print(f"    - {entry}")
                    if len(r.orphans) > 50:
                        print(f"    ... ({len(r.orphans) - 50} more)")
            print()
            print("FIX: add KB_INDEX.md rows for each orphan. Format:")
            print("  | [<file>#<id>](target/<backend>/<file>) | **<id>** — title | tier |")
            print("Tier guideline: L1=critical/cross-op, L2=med, L3=narrow/unverified, L4=L4-class")
        if total_dangling:
            print("\n❌ DANGLING index refs (KB_INDEX cites entry that doesn't exist in canonical KB file):")
            for r in all_results:
                if r.dangling_index_refs:
                    print(f"  {r.backend}/{r.file_type} ({len(r.dangling_index_refs)}):")
                    for entry in r.dangling_index_refs[:50]:
                        print(f"    - {entry}")
                    if len(r.dangling_index_refs) > 50:
                        print(f"    ... ({len(r.dangling_index_refs) - 50} more)")
            print()
            print("FIX: either (a) the dangling KB_INDEX row was authored before the")
            print("canonical entry was created — write the canonical entry now, OR (b)")
            print("the canonical entry was archived/renamed — remove the dangling row")
            print("from KB_INDEX. Anchor-missing entries can mislead grep-driven readers.")
        if total_missing_at:
            print(
                "\n⚠ MISSING/INVALID applies_to (NODE-21 Phase E): entries lack "
                "scope tag or have invalid tier values"
            )
            print("  Count by file_type:")
            for r in all_results:
                if r.missing_applies_to:
                    print(f"    {r.backend}/{r.file_type}: {len(r.missing_applies_to)} entries")
                    for e in r.missing_applies_to[:10]:
                        print(f"      - {e}")
                    if len(r.missing_applies_to) > 10:
                        print(f"      ... ({len(r.missing_applies_to) - 10} more)")
            if args.strict_applies_to:
                print("  Mode: STRICT — commit blocked. Fix or use --report-only to audit without blocking.")
            else:
                print("  Mode: informational (use --strict-applies-to to block commits)")

    # Strict-mode policy (2026-05-22 refinement, updated 2026-05-28 NODE-21 Phase E):
    #   - ORPHAN entries → strict-blocking
    #   - DUPLICATE IDs → warn-only (unless --strict-duplicates)
    #   - DANGLING index refs → warn-only (unless --strict-dangling)
    #   - missing_applies_to → warn-only (unless --strict-applies-to, NODE-21 Phase E)
    #   - SoC scope contradiction → strict-blocking BY DEFAULT (2026-07-17). No
    #     opt-in flag, unlike --strict-applies-to: that one is opt-in because a
    #     backlog of entries genuinely lacked applies_to, so blocking by default
    #     would have bricked unrelated commits. This check has ZERO pre-existing
    #     violations once PB-35 is reconciled (738-entry sweep, 188 evidence
    #     lines evaluated, 1 violation = PB-35), so default-blocking cannot brick
    #     an honest commit — and a self-contradicting entry is a data defect, not
    #     a backlog item.
    if total_duplicates and getattr(args, "strict_duplicates", False) and not args.report_only:
        return 1
    if soc_violations and not args.report_only:
        return 1
    # ID-sequence out-of-range (2026-07-23, OL-984): default-blocking, no opt-in
    # flag — same rationale as the SoC-scope/template-scope lints: ZERO current
    # violations, so default-block cannot brick an honest commit; a wildly
    # mis-numbered id is a data defect, not a backlog item.
    if total_id_seq and not args.report_only:
        return 1
    # DEBT-222: default-blocking (no opt-in flag), mirroring the SoC-scope lint.
    # Safe to default-block because the backfill leaves ZERO violations, so an
    # honest commit is never bricked; a newly-added untagged template is a data
    # defect, not a backlog item.
    if template_scope_violations and not args.report_only:
        return 1
    if tag_pollution and not args.report_only:
        return 1
    if total_orphans and not args.report_only:
        return 1
    _pp_conf = []
    for _pi in KB_ROOT.rglob("PATTERN_INDEX.md"):
        _pp_conf += find_pp_title_conflicts(_pi)
    if _pp_conf and not args.report_only:
        return 1
    if total_dangling and args.strict_dangling and not args.report_only:
        return 1
    if total_missing_at and args.strict_applies_to and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
