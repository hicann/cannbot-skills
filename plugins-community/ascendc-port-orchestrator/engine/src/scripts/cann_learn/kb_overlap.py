# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""C35: KB-overlap with reason codes — self-evolution feedback loop.

When learner extracts a pattern that already exists in the KB, the right
response is NOT to add a redundant entry — it's to identify WHY the existing
entry didn't surface during op-gen and propose a metadata fix.

Reason codes (from v2 design):
- same_op_class: existing entry tagged for the same op_taxonomy class
- same_symptom_keyword: symptom-keyword overlap with candidate
- same_public_api: references the same AscendC API call
- same_reject_condition: same anti-pattern / when-NOT-to-apply
- same_evidence_family: references same hardware behavior / OL-class

≥2 reason codes match = overlap → metadata-fix proposal, not new entry.
"""
from __future__ import annotations
import logging

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REASON_CODES = (
    "same_op_class",
    "same_symptom_keyword",
    "same_public_api",
    "same_reject_condition",
    "same_evidence_family",
    "same_construction_sig",  # C35-fix 2026-06-05: valence-agnostic code-construction
                              # match (catches a RECIPE that re-walks a known BUG's
                              # construction — symptom keywords are opposite-valence so
                              # cannot bridge recipe↔bug; this can). See PP107↔PB-35 DEBT.
)


@dataclass
class KBEntry:
    id: str            # OL-N, P-PN, EC-N, etc.
    title: str
    body: str
    op_classes: set[str] = field(default_factory=set)
    symptom_keywords: set[str] = field(default_factory=set)
    api_refs: set[str] = field(default_factory=set)
    reject_conditions: set[str] = field(default_factory=set)
    evidence_pointers: set[str] = field(default_factory=set)
    construction_sigs: set[str] = field(default_factory=set)


@dataclass
class CandidatePattern:
    candidate_id: str
    title: str
    body: str
    op_classes: set[str] = field(default_factory=set)
    symptom_keywords: set[str] = field(default_factory=set)
    api_refs: set[str] = field(default_factory=set)
    reject_conditions: set[str] = field(default_factory=set)
    evidence_pointers: set[str] = field(default_factory=set)
    construction_sigs: set[str] = field(default_factory=set)


@dataclass
class OverlapMatch:
    kb_id: str
    reasons: list[str]
    metadata_fix_proposal: str  # short text describing what to add to existing entry
    construction_collision: bool = False  # C35-fix: candidate re-walks a PLATFORM_BUG's
                                          # construction (shared api + construction_sig with
                                          # a PB-N) — HIGH severity, fires regardless of the
                                          # 2-reason threshold (a deadlock-recipe must never
                                          # slip-promote).


@dataclass
class OverlapResult:
    candidate_id: str
    matches: list[OverlapMatch]

    @property
    def has_overlap(self) -> bool:
        return any(len(m.reasons) >= 2 for m in self.matches)

    @property
    def has_construction_collision(self) -> bool:
        """C35-fix: a candidate re-walks a PLATFORM_BUG's construction — HIGH severity,
        the caller MUST NOT auto-promote (deadlock-recipe-slip guard).
        """
        return any(m.construction_collision for m in self.matches)

    @property
    def construction_collisions(self) -> list[OverlapMatch]:
        return [m for m in self.matches if m.construction_collision]


# Built-in op-class taxonomy mirror (subset; full version in op_taxonomy.py)
KNOWN_OP_CLASSES = {
    "normalization", "reduction", "activation", "attention", "rope",
    "quantization", "conv", "matmul", "scatter-gather", "fused",
    "fft", "data_movement", "softmax", "sort", "topk",
}


# OL-class evidence prefixes — used for evidence_pointer matching
OL_CLASS_RE = re.compile(r"\bOL-\d+\b")
PP_CLASS_RE = re.compile(r"\bP-P\d+\b")
EC_CLASS_RE = re.compile(r"\bEC-\d+\b")
PB_CLASS_RE = re.compile(r"\bPB-\d+\b")


def _extract_evidence_pointers(text: str) -> set[str]:
    """Extract OL-N / P-PN / EC-N / PB-N references from a text body."""
    out = set()
    for pattern in (OL_CLASS_RE, PP_CLASS_RE, EC_CLASS_RE, PB_CLASS_RE):
        out.update(pattern.findall(text))
    return out


def _extract_api_refs(text: str) -> set[str]:
    """Extract AscendC public API call patterns from text.

    Catches:
      - `AscendC::Foo` namespace-qualified
      - Backtick-quoted symbols `DataCopy()` / `WholeReduceSum<T>`
      - Function-call shapes Foo(...) where Foo is UpperCamel
    """
    out = set()
    # Namespace-qualified
    for m in re.finditer(r"\bAscendC::([A-Z][A-Za-z0-9_]*)", text):
        out.add(f"AscendC::{m.group(1)}")
        out.add(m.group(1))  # also bare name
    # Backtick-quoted
    for m in re.finditer(r"`([A-Z][A-Za-z0-9_]*(?:<[^>`]*>)?(?:\(\))?)`", text):
        bare = re.sub(r"[<(].*$", "", m.group(1))
        out.add(bare)
    # UpperCamel followed by `<` or `(`
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]+)(?=[<(])", text):
        out.add(m.group(1))
    return out


# C35-fix (i): canonical op-class aliases — different wordings of the SAME class.
# Without this, `MIX_AIC_1_2` and `mixed_aic_aiv_pattern_a` (the same MIX op-class)
# don't set-intersect, so `same_op_class` silently misses (PP107↔PB-35 DEBT).
_OP_CLASS_ALIASES = (
    # (canonical_token, regex that matches any wording of it)
    ("mix_aic", re.compile(r"mix(?:ed)?[_\- ]?aic", re.I)),
    ("cross_core", re.compile(r"cross[_\- ]?core", re.I)),
    ("cube_vec", re.compile(r"cube[_\- /]*vec(?:tor)?", re.I)),
    ("flash_attention", re.compile(r"\b(?:flash[_\- ]?attn|flash[_\- ]?attention|\bfa[_\- ]?class)\b", re.I)),
)


def _extract_op_classes(text: str) -> set[str]:
    """Find op_taxonomy class names mentioned in text body (substring + canonical alias)."""
    out = set()
    text_lower = text.lower()
    for cls in KNOWN_OP_CLASSES:
        if cls.lower() in text_lower:
            out.add(cls)
    # C35-fix (i): canonical aliases so differently-worded same-class entries intersect
    for canon, rx in _OP_CLASS_ALIASES:
        if rx.search(text):
            out.add(canon)
    return out


# C35-fix (ii) LOAD-BEARING: distinctive code-construction markers. A RECIPE (positive
# "do X this way") and a PLATFORM_BUG (negative "X deadlocks") describe the SAME
# construction with OPPOSITE symptom vocabulary, so `same_symptom_keyword` structurally
# cannot bridge them. These valence-agnostic construction signatures CAN. Curated +
# targeted (cross-core/cube sync constructions, where deadlock-recipes live) — NOT a
# general semantic matcher (per main 2026-06-05: don't over-engineer).
_CONSTRUCTION_SIG_RES = (
    # cross-core sync MODE (the load-bearing one for PP107↔PB-35): normalize <2>/<0x2>→mode2
    ("crosscore_mode2", re.compile(r"CrossCoreSetFlag\s*<\s*(?:2|0x2)\s*[,>]", re.I)),
    ("crosscore_mode4", re.compile(r"CrossCoreSetFlag\s*<\s*(?:4|0x4)\s*[,>]", re.I)),
    ("crosscore_setflag", re.compile(r"\bCrossCoreSetFlag\b")),
    ("shared_flag_id", re.compile(r"shared[_\- ]?flag[_\- ]?id|share[ds]?\s+(?:a\s+)?flag\s+id", re.I)),
    ("disjoint_flag_id", re.compile(r"disjoint\s+(?:per-sub-block\s+)?(?:flag\s+)?ids?|id\s*\+\s*16", re.I)),
    ("hardevent_eventt", re.compile(r"SetFlag\s*<\s*HardEvent|event_t\s*\(\s*\d", re.I)),
    ("sync_mode_paired", re.compile(r"\bMODE\s*[24]\b|1:2\s*(?:paired|ratio)|1:1\s*(?:individually)?", re.I)),
)


def _extract_construction_sig(text: str) -> set[str]:
    """Extract distinctive code-construction signature tokens (valence-agnostic)."""
    out = set()
    for tok, rx in _CONSTRUCTION_SIG_RES:
        if rx.search(text):
            out.add(tok)
    return out


# C35-fix (iii): the candidate's own reject conditions were never parsed (hardcoded
# `set()`), so `same_reject_condition` was a DEAD reason code for all candidates.
_REJECT_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*>\s]*)?\**\s*reject[_\- ]?cond(?:ition)?s?\**\s*[:：]?(.*)", re.I)


def _extract_reject_conditions(body: str) -> set[str]:
    """Parse a `Reject_cond`/`reject_condition` line into distinctive tokens."""
    out: set[str] = set()
    for m in _REJECT_LINE_RE.finditer(body):
        seg = m.group(1)[:600].lower()
        for tok in re.findall(r"\b[a-z][a-z0-9_]{4,}\b", seg):
            out.add(tok)
    return out


def _extract_symptom_keywords(text: str) -> set[str]:
    """Heuristic: keywords from ## Symptom or ## Trigger sections of KB entry."""
    out = set()
    # Look for "## Symptom" / "## Trigger" sections
    for m in re.finditer(r"##\s+(?:Symptom|Trigger|When applies)\s*\n([^\n]+(?:\n(?!##)[^\n]+)*)",
                          text, re.IGNORECASE):
        section = m.group(1)
        # Extract noun-phrase-ish tokens (length ≥ 4, alpha)
        for tok in re.findall(r"\b[a-z][a-z_-]{3,}\b", section.lower()):
            out.add(tok)
    return out


def parse_kb_entry(entry_id: str, title: str, body: str) -> KBEntry:
    """Build a KBEntry from raw text by auto-extracting all metadata fields."""
    full = title + "\n" + body
    return KBEntry(
        id=entry_id,
        title=title,
        body=body,
        op_classes=_extract_op_classes(full),
        symptom_keywords=_extract_symptom_keywords(body),
        api_refs=_extract_api_refs(full),
        reject_conditions=_extract_reject_conditions(body),  # C35-fix (iii)
        evidence_pointers=_extract_evidence_pointers(full),
        construction_sigs=_extract_construction_sig(full),   # C35-fix (ii)
    )


def parse_candidate(cand_id: str, title: str, body: str) -> CandidatePattern:
    """Build a CandidatePattern by auto-extracting metadata fields."""
    full = title + "\n" + body
    return CandidatePattern(
        candidate_id=cand_id,
        title=title,
        body=body,
        op_classes=_extract_op_classes(full),
        symptom_keywords=_extract_symptom_keywords(body),
        api_refs=_extract_api_refs(full),
        reject_conditions=_extract_reject_conditions(body),  # C35-fix (iii)
        evidence_pointers=_extract_evidence_pointers(full),
        construction_sigs=_extract_construction_sig(full),   # C35-fix (ii)
    )


def parse_kb_index(kb_root: Path) -> list[KBEntry]:
    """Walk src/skills/references/*.md, extract each ## OL-N / ### EC-N / etc.
    entry as a KBEntry. Best-effort: regex-based, doesn't validate structure.
    """
    entries: list[KBEntry] = []
    if not kb_root.exists():
        return entries
    for md in kb_root.rglob("*.md"):
        skip_current_item = False
        try:
            text = md.read_text(errors="replace")
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        # Match ## OL-N / ### EC-N / ### PB-N / ## P-PN style headers
        # Keep title + body until next header
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r"^#{1,3}\s+(OL-\d+|P-P\d+|EC-\d+|PB-\d+)\s*[:\s]+(.*)$", line)
            if not m:
                i += 1
                continue
            entry_id = m.group(1)
            title = m.group(2).strip()
            # Body is everything until next ##/### header at same/higher level
            body_lines = []
            i += 1
            while i < len(lines):
                if re.match(r"^#{1,3}\s+", lines[i]):
                    break
                body_lines.append(lines[i])
                i += 1
            entries.append(parse_kb_entry(entry_id, title, "\n".join(body_lines)))
    return entries


def _set_intersect_size(a: set, b: set) -> int:
    return len(a & b)


def check_overlap(
    candidate: CandidatePattern,
    kb_index: list[KBEntry],
    *,
    min_reasons: int = 2,
) -> OverlapResult:
    """Compare candidate against every KB entry; return matches with ≥ min_reasons."""
    matches: list[OverlapMatch] = []
    for entry in kb_index:
        reasons: list[str] = []
        if _set_intersect_size(candidate.op_classes, entry.op_classes) > 0:
            reasons.append("same_op_class")
        if _set_intersect_size(candidate.symptom_keywords, entry.symptom_keywords) >= 2:
            reasons.append("same_symptom_keyword")
        if _set_intersect_size(candidate.api_refs, entry.api_refs) > 0:
            reasons.append("same_public_api")
        if _set_intersect_size(candidate.reject_conditions, entry.reject_conditions) > 0:
            reasons.append("same_reject_condition")
        if _set_intersect_size(candidate.evidence_pointers, entry.evidence_pointers) > 0:
            reasons.append("same_evidence_family")
        shared_sig = candidate.construction_sigs & entry.construction_sigs
        if shared_sig:
            reasons.append("same_construction_sig")  # C35-fix (ii)
        # C35-fix (ii) LOAD-BEARING: a candidate sharing a public-API AND a construction
        # signature with a PLATFORM_BUG (PB-N) is re-walking a documented bug's construction
        # (valence-agnostic — works even though the recipe's symptom vocabulary is the
        # opposite of the bug's). This is the PP107↔PB-35 deadlock-recipe-slip class. HIGH
        # severity: a deadlock-recipe must never slip-promote.
        construction_collision = (
            entry.id.startswith("PB-")
            and "same_public_api" in reasons
            and "same_construction_sig" in reasons
        )
        if len(reasons) >= min_reasons or construction_collision:
            if construction_collision:
                proposal = (
                    f"⚠ CONSTRUCTION-COLLISION with PLATFORM_BUG {entry.id} "
                    f"({entry.title[:60]}): this candidate re-walks a DOCUMENTED bug's "
                    f"construction (shared api {sorted(candidate.api_refs & entry.api_refs)[:2]} "
                    f"+ construction-sig {sorted(shared_sig)}). DO NOT promote until the "
                    f"candidate (a) cross-refs {entry.id}, (b) carries a reject_cond against "
                    f"the bug's construction, and (c) replaces it with the documented fix. "
                    f"Auto-metadata-fix is NOT appropriate — this is a safety hold, not a dedup."
                )
            else:
                # Build a short metadata-fix proposal
                missing_in_existing = []
                if candidate.op_classes - entry.op_classes:
                    missing_in_existing.append(
                        f"op_classes: add {sorted(candidate.op_classes - entry.op_classes)}"
                    )
                if candidate.symptom_keywords - entry.symptom_keywords:
                    new_sk = sorted(candidate.symptom_keywords - entry.symptom_keywords)[:5]
                    missing_in_existing.append(f"symptom_keywords: add {new_sk}")
                if candidate.api_refs - entry.api_refs:
                    missing_in_existing.append(
                        f"api_refs: add {sorted(candidate.api_refs - entry.api_refs)[:3]}"
                    )
                proposal = (
                    f"Existing entry {entry.id} ({entry.title[:50]}) overlaps; "
                    f"to make it discoverable for this op-class, suggest: "
                    + "; ".join(missing_in_existing)
                ) if missing_in_existing else (
                    f"Existing entry {entry.id} fully covers candidate; "
                    "drop candidate, no metadata fix needed."
                )
            matches.append(OverlapMatch(
                kb_id=entry.id,
                reasons=reasons,
                metadata_fix_proposal=proposal,
                construction_collision=construction_collision,
            ))
    return OverlapResult(candidate_id=candidate.candidate_id, matches=matches)
