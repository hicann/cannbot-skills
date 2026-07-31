# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""KB tiering — the stable CONTRACT (arbiter + provider + gate + Entry).

Reference implementation of `docs/design/KB_TIERING_DESIGN.md` §14 (v0.5, three-way APPROVED).
Each project implements a `KBProvider` adapter against THIS; the `Arbiter` + `gate` are shared
(a5_ops-core). Language/transport-agnostic contract (an MCP skin can wrap the Arbiter later).

Build split (24h MVP): main = this lib + a5_ops adapter; cannbot = cannbot adapter; autoport = autoport
adapter. This module is the interface both adapters + the arbiter agree on — keep it stable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

KINDS = ("positive_pattern", "anti_pattern", "experience")  # LEARNED knowledge only — NOT config (§4)
ROLES = ("base", "official-canonical", "user-local", "intermediate")


# ── §14.1 Entry — the unified envelope (adapters translate their raw format to/from this) ──
@dataclass(frozen=True)
class Entry:
    id: str                                   # tier-namespaced: "customer:<hash>" | "OL-N"/"EC-N"/… | …
    tier: str                                 # provider's own tier name
    role: str                                 # one of ROLES (§13 N-tier: role, not fixed position)
    kind: str                                 # one of KINDS
    scope: dict = field(default_factory=dict)  # RESOLVE routing/filter: {applies_to, chip_scope, target}
    key: str = ""                             # DEDUP/lookup ONLY (narrow hard-key); NOT the resolve signature
    claim: str = ""                           # the knowledge text; source of the wide resolve full-signature
    evidence: dict = field(default_factory=dict)   # works-proof | reproducible-failure | A/B | {}
    trust: str = "unverified"                 # "verified" | "unverified"
    provenance: dict = field(default_factory=dict)  # {source, b_version_anchored, ts, agent?}
    meta: dict = field(default_factory=dict)   # extensible — adapter round-trips its own fields
    tombstoned: bool = False

    @property
    def content_hash(self) -> str:
        """Idempotency + tombstone key = hash(normalized claim + key + SCOPE).

        Scope is FOLDED IN (scan②) so scope-distinct entries (op_class A vs B / different chip) do
        NOT false-collide on put/tombstone. Idempotency = "same lesson + same scope → same hash".
        """
        norm = f"{self.claim.strip().lower()}|{self.key}|{sorted(self.scope.items())}"
        return hashlib.sha1(norm.encode()).hexdigest()[:12]


# ── signature helpers: DEDUP (narrow hard-key) vs RESOLVE (wide full-sig) are SEPARATE (scan①) ──
def hard_key(text: str) -> str:
    """Narrow DEDUP key: error codes + snake_case symbols. Falls back to full_sig when sparse."""
    codes = re.findall(r"\b\d{5,6}\b", text)
    syms = re.findall(r"\b[a-z]+_[a-z_]+\b", text.lower())
    hk = " ".join(sorted(set(codes)) + sorted(set(syms))[:6])
    return hk or full_sig(text)[:40]


def full_sig(text: str) -> str:
    """Wide RESOLVE signature: content words (recalls entries with no hard identifier)."""
    return " ".join(sorted(set(re.findall(r"[a-z0-9]{4,}", text.lower())))[:12])


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


# ── §14.4 gate — ONE gate, three modes (= §3's three uses) ──
def gate(e: Entry, mode: str, providers: list["KBProvider"]) -> tuple[bool, str]:
    """Shared evidence check. mode ∈ {"admission","override","promotion"}. Never excludes a KIND —
    excludes UNBACKED assertions. Providers' `admit` delegate here (one-gate invariant).
    """
    if e.kind not in KINDS:
        return False, f"bad kind {e.kind!r} (config is NOT KB — §4 boundary)"
    if e.kind in ("positive_pattern", "anti_pattern") and not e.evidence:
        return False, f"{e.kind} needs evidence (unbacked assertion)"       # evidence-by-kind
    for p in providers:                                                      # tombstone check (§7)
        ts = p.tombstones()
        if e.content_hash in ts:
            return False, f"tombstoned→already promoted (see {ts[e.content_hash]})"
    if mode in ("override", "promotion") and e.trust != "verified":
        return False, f"{mode} requires trust=verified (A/B for efficiency, repro for correctness)"
    return True, "ok"


# ── §14.2 KBProvider — per-tier adapter (anti-corruption layer; hides raw storage+index) ──
@runtime_checkable
class KBProvider(Protocol):
    tier: str
    role: str

    def resolve(self, signature: str) -> Optional[Entry]:
        """WIDE full-sig match (computed from claim) IN THIS tier; recalls no-hard-id entries."""
        ...

    def lookup(self, key: str) -> Optional[Entry]:
        """NARROW dedup/lookup by hard-key."""
        ...

    def index_rows(self) -> list:
        """This tier's index, normalized. MUST be orphan-free per tier."""
        ...

    def reindex(self) -> None:
        ...

    def admit(self, e: Entry) -> tuple[bool, str]:
        """THIN delegate to shared gate(e,"admission",...) + tier-specific STORAGE checks only.
        Do NOT re-implement admission logic here (one-gate invariant).
        """
        ...

    def put(self, e: Entry) -> None:
        """Idempotent by content_hash (scope-folded)."""
        ...

    def tombstones(self) -> dict:
        """{content_hash: to_serial} for entries promoted out."""
        ...

    def tombstone(self, content_hash: str, to_serial: str) -> None:
        ...

    # c-tier user-retract of own mistake = scan④, post-MVP follow.


# ── §14.3 Arbiter — shared a5_ops-core; composes an ORDERED provider list (N-tier) ──
@dataclass
class Resolution:
    status: str                 # "RESOLVED" | "CONFLICT_SURFACED" | "MISS"
    entry: Optional[Entry] = None
    conflict_with: Optional[Entry] = None
    tier: Optional[str] = None


class Arbiter:
    def __init__(self, providers: list[KBProvider]):
        self.providers = providers            # ORDERED by precedence (index 0 = highest, e.g. c)

    def resolve(self, signature: str) -> Resolution:
        hits = [(p, p.resolve(signature)) for p in self.providers]
        hits = [(p, e) for p, e in hits if e is not None]
        if not hits:
            return Resolution("MISS")
        p0, e0 = hits[0]
        for _, e in hits[1:]:                 # kind-aware conflict-surfacing (§5) — canonical kinds only
            if (e.role == "official-canonical"
                    and e0.kind in ("positive_pattern", "anti_pattern")
                    and e0.role != "official-canonical"
                    and e0.claim.strip() != e.claim.strip()):
                return Resolution("CONFLICT_SURFACED", entry=e0, conflict_with=e)  # not silent
        return Resolution("RESOLVED", entry=e0, tier=p0.tier)

    def write(self, e: Entry, target_tier: str) -> tuple[str, str]:
        p = self._provider(target_tier)
        ok, why = p.admit(e)                  # admit → shared gate
        if not ok:
            return ("REJECTED", why)
        p.put(e)
        return ("WRITTEN", e.id)

    def promote(self, e: Entry, from_tier: str, to_tier: str) -> tuple[str, str]:
        """Owner-gated, lower→higher (§13). gate(promotion) → reassign id → tombstone-from → put-to."""
        ok, why = gate(e, "promotion", self.providers)
        if not ok:
            return ("REJECTED", why)
        src, dst = self._provider(from_tier), self._provider(to_tier)
        new_id = f"{to_tier}:{e.content_hash}"  # adapter reassigns to its serial scheme in put()
        promoted = Entry(**{**e.__dict__, "id": new_id, "tier": to_tier,
                            "role": dst.role, "trust": "verified"})
        dst.put(promoted)
        src.tombstone(e.content_hash, new_id)   # closes the resurrection loop (§7)
        return ("PROMOTED", new_id)

    def reconcile_on_upgrade(self, new_official: KBProvider) -> list:
        """b ships new anti-patterns on upgrade → cross-check c entries → FLAG conflicts (NOT delete).
        §10#4. Returns a list of flagged (c_entry, matching_b_anti) — surfaced, never auto-deleted.
        """
        flagged = []
        b_antis = [r for r in new_official.index_rows()]
        for p in self.providers:
            if p.role != "user-local":
                continue
            for e in getattr(p, "entries", lambda: [])():
                for anti in b_antis:
                    if getattr(anti, "kind", "") == "anti_pattern" and jaccard(e.key, getattr(anti, "key", "")) >= 0.6:
                        flagged.append((e, anti))
        return flagged

    def index(self) -> dict:
        return {p.tier: p.index_rows() for p in self.providers}

    def _provider(self, tier: str) -> KBProvider:
        for p in self.providers:
            if p.tier == tier:
                return p
        raise KeyError(f"no provider for tier {tier!r}")
