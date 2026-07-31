# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""autoport c-tier KBProvider adapter — VENDORED from scan's npu-autoport-pipeline.

Source: `npu-autoport-pipeline @ d042326` `pipeline/kb/adapters/autoport_c.py::AutoportCProvider`
(scan, three-way-APPROVED §14 contract). Vendored here so the a5_ops `Arbiter([c,b,a])` demo is
**self-contained** (no cross-repo runtime import — a5_ops stays an independent harness). The LOGIC is
byte-identical to scan's file; the ONLY change is the import line:
    scan (package-relative):  `from ..interface import ...`
    a5_ops (flat convention): `from interface import ...`   ← this file
a5_ops's kb_tiering uses flat imports (`sys.path` + `from interface import`, see adapter_a5ops.py +
tests), so the flat form binds to a5_ops's own byte-identical `interface.py` (@ 8165e991) and avoids a
double-import Entry-identity split. On a scan re-sync, re-vendor from scan's latest + re-adapt this one line.

tier='c', role='user-local' → upgrade-safe: user learnings live OUTSIDE the shipped harness, so a
harness upgrade never clobbers them. Persists to `INDEX.jsonl` (survives restart — real round-trip proof).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from interface import Entry, KBProvider, gate, full_sig, jaccard

# Persisted Entry fields. `content_hash` is a DERIVED property (hash of claim+key+scope) — never
# stored, always recomputed on load → idempotency holds across restart (same fields → same hash).
_FIELDS = ("id", "tier", "role", "kind", "scope", "key", "claim",
           "evidence", "trust", "provenance", "meta", "tombstoned")


class AutoportCProvider:
    """autoport user-local (c) tier. Storage = one JSON object per line in `INDEX.jsonl`
    (the index IS the entry store → orphan-free by construction); tombstones in `TOMBSTONES.json`.
    """

    tier = "c"
    role = "user-local"

    def __init__(self, root, resolve_threshold: float = 0.3):
        self.root = Path(root)
        self.index_path = self.root / "INDEX.jsonl"
        self.tombstone_path = self.root / "TOMBSTONES.json"
        self._threshold = resolve_threshold
        self.root.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── KBProvider contract (§14.2) ──
    def resolve(self, signature: str) -> Optional[Entry]:
        """WIDE full-sig match computed FROM claim (recalls no-hard-id entries; scan①). NOT the key."""
        best, best_score = None, 0.0
        for e in self._d.values():
            if e.tombstoned:
                continue
            j = jaccard(signature, full_sig(e.claim))
            if j > best_score:
                best, best_score = e, j
        return best if best_score >= self._threshold else None

    def lookup(self, key: str) -> Optional[Entry]:
        """NARROW dedup/lookup by hard-key."""
        return next((e for e in self._d.values() if e.key == key), None)

    def index_rows(self) -> list:
        return list(self._d.values())

    def reindex(self) -> None:
        self._load()

    def admit(self, e: Entry) -> tuple[bool, str]:
        """THIN delegate to the shared gate (one-gate invariant) + a tier-specific
        STORAGE check only. No admission LOGIC re-implemented here.
        """
        if not os.access(self.root, os.W_OK):
            return False, f"c-tier root {self.root} not writable"
        return gate(e, "admission", [self])

    def put(self, e: Entry) -> None:
        """Idempotent by content_hash (scope-folded)."""
        self._d[e.content_hash] = e
        self._flush()

    def tombstones(self) -> dict:
        return self._ts

    def tombstone(self, content_hash: str, to_serial: str) -> None:
        self._ts[content_hash] = to_serial
        self._flush()

    def entries(self) -> list:
        """For `Arbiter.reconcile_on_upgrade` (user-local scan)."""
        return list(self._d.values())

    # ── storage <-> Entry envelope (the anti-corruption boundary) ──
    @staticmethod
    def _from_json(d: dict) -> Entry:
        return Entry(**{k: d[k] for k in _FIELDS if k in d})

    @staticmethod
    def _to_json(e: Entry) -> dict:
        return {k: getattr(e, k) for k in _FIELDS}

    def _load(self) -> None:
        self._d: dict[str, Entry] = {}   # content_hash -> Entry
        if self.index_path.exists():
            for line in self.index_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                e = self._from_json(json.loads(line))
                self._d[e.content_hash] = e
        self._ts: dict[str, str] = {}
        if self.tombstone_path.exists():
            self._ts = json.loads(self.tombstone_path.read_text())

    def _flush(self) -> None:
        # rewrite the whole index (per-deployment KB is small); INDEX.jsonl IS the index → orphan-free
        self.index_path.write_text(
            "".join(json.dumps(self._to_json(e), ensure_ascii=False) + "\n" for e in self._d.values()))
        self.tombstone_path.write_text(json.dumps(self._ts, ensure_ascii=False))


def make_autoport_c(root, resolve_threshold: float = 0.3) -> AutoportCProvider:
    """Factory (mirrors scan's `make_autoport_c`): `Arbiter([make_autoport_c(user_kb_dir), b, a])`."""
    return AutoportCProvider(root, resolve_threshold=resolve_threshold)
