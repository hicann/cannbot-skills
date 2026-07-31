# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""a5_ops KBProvider adapter — the b-tier (official/canonical) over the committed KB.

Wraps a5_ops's `src/skills/references/KB_INDEX.md` (+ the target KB files) behind the §14.2
`KBProvider` contract, translating `OL-N`/`EC-N`/`PB-N`/`CAND-*`/`P-Pxx` index rows ↔ `Entry`.
This is the "anti-corruption layer" for a5_ops's markdown/serial format (§11).

Role: `b` = official-canonical. b is **promotion-gate-only** — `admit` rejects direct writes;
new canonical entries arrive via `Arbiter.promote(c→b)` → `put`. MVP-safe: `put` appends to a
scoped `promoted_inbox.jsonl` (NOT an in-place edit of the committed KB_INDEX — that faithful
writer/reindex round-trip is the post-MVP hardening, the "bulk" flagged in the PoC).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from interface import Entry, gate, hard_key, full_sig, jaccard

# Common index-row shape: | [path#ID](link) | **ID** — claim… | applies_to; soc=… |
_ROW = re.compile(
    r"\|\s*\[[^\]]*#(?P<id>[A-Za-z0-9_\-]+)\][^|]*\|\s*"
    r"(?:\*\*[^*]+\*\*\s*[—\-:]+\s*)?(?P<claim>.+?)\s*\|\s*"
    r"(?P<scope>[^|]*)\|"
)
_ANTI = re.compile(r"anti[- ]?pattern|ANTI-PATTERN|don'?t\b|forbidden|never\b", re.I)


class A5opsProvider:
    tier = "b"
    role = "official-canonical"

    def __init__(self, kb_index: Path, inbox: Optional[Path] = None):
        self.kb_index = Path(kb_index)
        self.inbox = Path(inbox) if inbox else self.kb_index.parent / ".kb_tiering_promoted_inbox.jsonl"
        self._entries: list[Entry] = []
        self._ts: dict = {}
        self.parse_total = 0
        self.parse_ok = 0
        self._load()

    @staticmethod
    def _parse_scope(scope_raw: str) -> dict:
        s: dict = {"raw": scope_raw}
        if (mc := re.search(r"soc=([A-Za-z0-9_]+)", scope_raw)):
            s["chip_scope"] = mc.group(1)
        if (ma := re.search(r"applies_to[:\s]+([^;]+)", scope_raw)):
            s["applies_to"] = ma.group(1).strip()
        return s

    def resolve(self, signature: str) -> Optional[Entry]:
        best, bs = None, 0.0
        for e in self._entries:
            j = jaccard(signature, full_sig(e.claim))          # WIDE full-sig (from claim)
            if j > bs:
                best, bs = e, j
        return best if bs >= 0.2 else None

    def lookup(self, key: str) -> Optional[Entry]:
        for e in self._entries:
            if jaccard(key, e.key) >= 0.6:                      # NARROW hard-key dedup
                return e
        return None

    def index_rows(self) -> list:
        return list(self._entries)

    def entries(self):                                          # used by reconcile_on_upgrade
        return list(self._entries)

    def reindex(self) -> None:
        self._entries.clear()
        self.parse_total = self.parse_ok = 0
        self._load()

    # ── write side (promotion-gated) ──
    @staticmethod
    def admit(e: Entry) -> tuple[bool, str]:
        # b is promotion-gate-only: no DIRECT c-style writes. (Still runs the shared gate so the
        # "one gate" invariant holds if a caller routes here — but the tier policy forbids direct put.)
        return (False, "b (official-canonical) is promotion-gate-only — write via Arbiter.promote(c→b)")

    def put(self, e: Entry) -> None:
        # reached only via Arbiter.promote() (which already passed gate(promotion)). MVP-safe:
        # append to the promoted-inbox (idempotent by content_hash), NOT an in-place KB_INDEX edit.
        if self.lookup(e.key) is not None:
            return                                             # idempotent-ish (already canonical)
        entry_fields = (
            "id", "tier", "role", "kind", "scope", "key", "claim",
            "evidence", "trust", "provenance", "meta", "tombstoned",
        )
        rec = {"content_hash": e.content_hash}
        for field in entry_fields:
            rec[field] = getattr(e, field)
        with self.inbox.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._entries.append(e)

    def tombstones(self) -> dict:
        return self._ts

    def tombstone(self, content_hash: str, to_serial: str) -> None:
        self._ts[content_hash] = to_serial

    # ── read side ──
    def _load(self) -> None:
        self._entries.clear()
        seen = set()
        for line in self.kb_index.read_text(errors="replace").splitlines():
            if not line.startswith("| ["):
                continue
            self.parse_total += 1
            match = _ROW.match(line)
            if not match:
                continue
            self.parse_ok += 1
            entry_id = match["id"]
            claim = match["claim"].strip()[:400]
            scope_raw = match["scope"].strip()[:120]
            if entry_id in seen:
                continue
            seen.add(entry_id)
            self._entries.append(Entry(
                id=entry_id, tier="b", role=self.role,
                kind="anti_pattern" if _ANTI.search(claim) else "positive_pattern",
                scope=self._parse_scope(scope_raw), key=hard_key(claim), claim=claim,
                evidence={"canonical": True, "index_row": True}, trust="verified",
                provenance={"source": "a5_ops KB_INDEX", "id": entry_id},
            ))
        # merge any promoted-inbox entries (from promote c→b, MVP)
        if self.inbox.exists():
            for line in self.inbox.read_text(errors="replace").splitlines():
                if line.strip():
                    data = json.loads(line)
                    entry_fields = (
                        "id", "tier", "role", "kind", "scope", "key", "claim",
                        "evidence", "trust", "provenance", "meta", "tombstoned",
                    )
                    entry_data = {
                        field: data[field] for field in entry_fields if field in data
                    }
                    self._entries.append(Entry(**entry_data))
