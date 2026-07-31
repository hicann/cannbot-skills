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
"""KB tiering PoC — gauge implementation difficulty of the §14 contract (KB_TIERING_DESIGN.md v0.5).

Exploratory (owner 2026-07-02: "做一些 PoC 看看难度再做决定"). NOT production. Exercises:
  - Entry envelope + gate (admission/override/promotion) + Arbiter (resolve/write/promote)
  - an in-memory c-tier adapter (trivial)
  - an a5_ops b-tier adapter that parses the REAL src/skills/references/KB_INDEX.md  ← the hard part
Run: python3 src/scripts/kb_tiering/poc/kb_tiering_poc.py
Prints VERDICT lines + a DIFFICULTY assessment.
"""
from __future__ import annotations
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

# ── §14.1 Entry envelope ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entry:
    id: str
    tier: str
    role: str
    kind: str
    scope: dict
    key: str
    claim: str
    evidence: dict = field(default_factory=dict)
    trust: str = "unverified"
    provenance: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    tombstoned: bool = False

    @property
    def content_hash(self) -> str:                      # = hash(claim + key + SCOPE)  (scan②)
        norm = f"{self.claim.strip().lower()}|{self.key}|{sorted(self.scope.items())}"
        return hashlib.sha1(norm.encode()).hexdigest()[:12]


def hard_key(text: str) -> str:                         # narrow dedup key: error-codes + snake_case
    codes = re.findall(r"\b\d{5,6}\b", text)
    syms = re.findall(r"\b[a-z]+_[a-z_]+\b", text.lower())
    return " ".join(sorted(set(codes)) + sorted(set(syms))[:6]) or full_sig(text)[:40]


def full_sig(text: str) -> str:                         # wide resolve signature: content words (scan①)
    return " ".join(sorted(set(re.findall(r"[a-z0-9]{4,}", text.lower())))[:12])

# ── §14.4 gate (one gate, three modes) ──────────────────────────────────────


def gate(e: Entry, mode: str, providers: list) -> tuple[bool, str]:
    if e.kind not in ("positive_pattern", "anti_pattern", "experience"):
        return False, f"bad kind {e.kind!r} (config is NOT KB, §4)"
    if e.kind in ("positive_pattern", "anti_pattern") and not e.evidence:
        return False, f"{e.kind} needs evidence (unbacked assertion)"        # evidence-by-kind
    for p in providers:                                                       # tombstone check (§7)
        if e.content_hash in p.tombstones():
            return False, f"tombstoned→promoted (see {p.tombstones()[e.content_hash]})"
    if mode in ("override", "promotion") and e.trust != "verified":
        return False, f"{mode} requires trust=verified"
    return True, "ok"

# ── §14.2 KBProvider (per-tier adapter) ─────────────────────────────────────


class KBProvider(Protocol):
    tier: str
    role: str

    def resolve(self, signature: str) -> Optional[Entry]:
        ...

    def admit(self, e: Entry) -> tuple[bool, str]:
        ...

    def put(self, e: Entry) -> None:
        ...

    def tombstones(self) -> dict:
        ...


class MemProvider:                                       # c-tier: trivial in-memory
    def __init__(self, tier, role):
        self.tier, self.role = tier, role
        self._d = {}
        self._ts = {}

    def resolve(self, signature):
        toks = set(signature.split())
        best, bs = None, 0.0
        for e in self._d.values():
            o = toks & set(full_sig(e.claim).split())
            j = len(o) / max(1, len(toks | set(full_sig(e.claim).split())))
            if j > bs:
                best, bs = e, j
        return best if bs >= 0.3 else None

    @staticmethod
    def admit(e):
        return gate(e, "admission", [self])  # THIN delegate to shared gate

    @staticmethod
    def put(e):
        self._d[e.content_hash] = e

    def tombstones(self):
        return self._ts


class A5opsKBIndexProvider:                             # b-tier: parse the REAL KB_INDEX.md  ← hard part
    ROW = re.compile(
        r"\|\s*\[[^\]]*#(?P<id>[A-Za-z0-9_\-]+)\][^|]*\|\s*"
        r"(?:\*\*[^*]+\*\*\s*[—-]\s*)?(?P<claim>.+?)\s*\|\s*"
        r"(?P<scope>[^|]*)\|"
    )

    def __init__(self, path: Path):
        self.tier, self.role = "b", "official-canonical"
        self._ts = {}
        self._e = []
        n_tot = n_ok = 0
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith("| ["):  # index rows only
                continue
            n_tot += 1
            m = self.ROW.match(line)
            if not m:
                continue
            n_ok += 1
            claim = m["claim"][:200]
            self._e.append(Entry(id=m["id"], tier="b", role=self.role,
                kind="anti_pattern" if "anti" in claim.lower() else "positive_pattern",
                scope={"raw": m["scope"].strip()[:80]}, key=hard_key(claim), claim=claim,
                evidence={"canonical": True}, trust="verified"))
        self.parse_total, self.parse_ok = n_tot, n_ok

    def resolve(self, signature):
        toks = set(signature.split())
        best, bs = None, 0.0
        for e in self._e:
            o = toks & set(full_sig(e.claim).split())
            j = len(o) / max(1, len(toks))
            if j > bs:
                best, bs = e, j
        return best if bs >= 0.25 else None

    def admit(self, e):
        return False, "b is promotion-gate-only, no direct write"

    def put(self, e):
        raise RuntimeError("b write only via promote()")

    def tombstones(self):
        return self._ts

# ── §14.3 Arbiter ───────────────────────────────────────────────────────────


class Arbiter:
    def __init__(self, providers):
        self.providers = providers  # ORDERED by precedence (N-tier)

    def resolve(self, signature):
        hits = [(p, p.resolve(signature)) for p in self.providers]
        hits = [(p, e) for p, e in hits if e]
        if not hits:
            return ("MISS", None, None)
        (p0, e0) = hits[0]
        for _, e in hits[1:]:                                        # kind-aware conflict-surfacing (§5)
            if e.role == "official-canonical" and e0.kind in ("positive_pattern", "anti_pattern") \
               and e0.role != "official-canonical" and e0.claim.strip() != e.claim.strip():
                return ("CONFLICT_SURFACED", e0, e)                 # not silent
        return ("RESOLVED", e0, p0)

    def write(self, e, target_tier):
        p = next(p for p in self.providers if p.tier == target_tier)
        ok, why = p.admit(e)                                        # admit → shared gate
        if not ok:
            return ("REJECTED", why)
        p.put(e)
        return ("WRITTEN", e.id)

# ── demo / difficulty gauge ─────────────────────────────────────────────────


def main():
    root = Path(__file__).resolve().parents[4]  # engine/
    kbindex = root.parent / "kb" / "KB_INDEX.md"  # 2026-07-05: <plugin_root>/kb/
    b = A5opsKBIndexProvider(kbindex)
    c = MemProvider("c", "user-local")
    arb = Arbiter([c, b])                                           # c > b precedence

    print(f"VERDICT parse: a5_ops KB_INDEX rows parsed {b.parse_ok}/{b.parse_total} into Entry "
          f"({100*b.parse_ok//max(1,b.parse_total)}%)")
    # 1) resolve hits b (official)
    q = full_sig("arch35 wrap cheat include")
    st, e, _ = arb.resolve(q)
    print(f"VERDICT resolve(b): {st} -> {e.id if e else None}")
    # 2) write a c positive_pattern (admit→gate), then resolve prefers c
    ce = Entry(id="customer:x1", tier="c", role="user-local", kind="positive_pattern",
               scope={"applies_to": "elementwise", "chip_scope": "A5"}, key=hard_key("use chunk 256 for D192"),
               claim="use chunk 256 tiling for D=192 elementwise on A5", evidence={"ab": "+2%"}, trust="verified")
    print(f"VERDICT write(c): {arb.write(ce, 'c')}")
    st2, e2, _ = arb.resolve(full_sig(ce.claim))
    print(f"VERDICT resolve(c-first): {st2} -> {e2.id if e2 else None}")
    # 3) admission blocks an unbacked anti_pattern (config-ish / no evidence)
    bad = Entry(id="customer:x2", tier="c", role="user-local", kind="anti_pattern",
                scope={}, key="k", claim="X fails", evidence={})  # no repro evidence
    print(f"VERDICT admit(no-evidence): {arb.write(bad, 'c')}")
    # 4) content_hash folds scope (two scope-distinct = different hash)
    h1 = ce.content_hash
    h2 = Entry(**{**ce.__dict__, "scope": {"applies_to": "reduction", "chip_scope": "A5"}}).content_hash
    print(f"VERDICT content_hash scope-fold: distinct={h1 != h2} ({h1} vs {h2})")

    print("\nDIFFICULTY: arbiter+gate+Entry+mem-adapter = trivial (~120 lines, pure logic). "
          "a5_ops KB_INDEX adapter = the real cost — one regex handles the common index-row shape; "
          "the long tail (OL/EC/PB vs CAND vs P-P multi-line blocks, embedded pipes) needs per-row-type "
          "parsing + a reindex()/writer for round-trip. Verdict: arbiter LIB is a few days; the a5_ops "
          "ADAPTER (faithful OL-N/KB_INDEX read+write+reindex) is the bulk of main's work.")


if __name__ == "__main__":
    main()
