# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""cannbot c-tier (user-local) KBProvider adapter.

Implements the frozen contract (`..interface`) for cannbot's **user-local** KB —
the upgrade-safe tier a distributed plugin deployment writes to (design §2/§6).

Storage (cannbot-local; NOT the contract — main owns only `interface.py`):
  <c_root>/entries/<content_hash>.json   one self-describing Entry per file (lossless round-trip)
  <c_root>/INDEX.md                      orphan-free human index (one row per entry file)
  <c_root>/.tombstones.json              {content_hash: to_serial} for promoted-out entries

`c_root` = `ASCENDC_PORT_USER_KB` or `~/.ascendc-port/user_kb`, **resolved at RUNTIME per call**
(never import-cached — the deployment may create it after import; design §6).

Write path (locked three-way): the `aog-knowledge-maintain` LLM skill produces an Entry-shaped
CANDIDATE; `Arbiter.write(e, "customer")` is the sole deterministic entry point → `admit` (THIN
delegate to shared `gate`) → `put`. This adapter never bypasses the gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..interface import Entry, KBProvider, gate, full_sig, hard_key, jaccard

TIER = "customer"
ROLE = "user-local"
_RESOLVE_THRESHOLD = 0.5   # wide full-sig jaccard (matches scan PoC resolve gate)


def resolve_c_root(explicit: Optional[str] = None) -> Path:
    """c-root, RUNTIME-resolved (never import-cached). Precedence: explicit arg >
    $ASCENDC_PORT_USER_KB > ~/.ascendc-port/user_kb.
    """
    root = explicit or os.environ.get("ASCENDC_PORT_USER_KB") or str(Path.home() / ".ascendc-port" / "user_kb")
    return Path(root)


def kb_write_root() -> str:
    """Deployment target-tier resolver (§6). cannbot deployment: c when a user_kb
    root is configured (ASCENDC_PORT_USER_KB set OR the default c-root exists — created by
    init.sh), else b (bundled). Resolved per-invocation.
    """
    if os.environ.get("ASCENDC_PORT_USER_KB"):
        return "customer"
    if (Path.home() / ".ascendc-port" / "user_kb").is_dir():
        return "customer"
    return "b"


class CannbotCProvider:
    """c-tier adapter. Duck-types the `KBProvider` Protocol (runtime_checkable → isinstance passes)."""

    tier = TIER
    role = ROLE

    def __init__(self, root: Optional[str] = None, providers: Optional[list] = None):
        # root may be None → resolved at RUNTIME per access (deployment may create it post-init).
        self._explicit_root = root
        # provider list for admit→gate tombstone check; defaults to [self] (c's own resurrection guard).
        self._providers = providers if providers is not None else [self]

    def set_gate_providers(self, providers: list) -> None:
        """Configure the ordered provider set consulted by the admission gate."""
        self._providers = providers

    def entries(self) -> list:
        """All live c entries (used by Arbiter.reconcile_on_upgrade)."""
        out = []
        ed = self._entries_dir()
        for f in sorted(ed.glob("*.json")):
            try:
                out.append(self._to_entry(json.loads(f.read_text())))
            except Exception:
                continue   # skip a corrupt/hand-broken file rather than crash the read path
        return out

    # ── KBProvider contract ──────────────────────────────────────────────────
    def resolve(self, signature: str) -> Optional[Entry]:
        """WIDE full-sig match (computed from each entry's claim) within c."""
        best, best_j = None, _RESOLVE_THRESHOLD
        for e in self.entries():
            j = jaccard(full_sig(e.claim), signature)
            if j >= best_j:
                best, best_j = e, j
        return best

    def lookup(self, key: str) -> Optional[Entry]:
        """NARROW dedup/lookup by hard-key (exact key, or hard_key(claim) fallback)."""
        for e in self.entries():
            if (e.key or hard_key(e.claim)) == key:
                return e
        return None

    def index_rows(self) -> list:
        """Orphan-free index (one per entry file), normalized to `Entry` — consistent with the
        b-adapter so the read bridge merges tiers uniformly.
        """
        return self.entries()

    def reindex(self) -> None:
        """Rebuild INDEX.md from the entry files (orphan-free by construction)."""
        lines = ["# 用户本地 KB 索引 (c 层) — id → key/kind (由 adapter 维护，勿手改此表)",
                 "", "| id | key | kind |", "|---|---|---|"]
        for e in self.index_rows():
            lines.append(f"| {e.id} | {e.key or hard_key(e.claim)} | {e.kind} |")
        (self._root()).mkdir(parents=True, exist_ok=True)
        (self._root() / "INDEX.md").write_text("\n".join(lines) + "\n")

    def admit(self, e: Entry) -> tuple[bool, str]:
        """THIN delegate to the shared gate (one-gate invariant) + c-storage writability.
        Does NOT re-implement admission logic.
        """
        ok, why = gate(e, "admission", self._providers)
        if not ok:
            return ok, why
        try:
            self._entries_dir()   # ensure c-root writable
        except OSError as exc:
            return False, f"c-root not writable: {exc}"
        return True, "ok"

    def put(self, e: Entry) -> None:
        """Idempotent by content_hash (scope-folded): same lesson+scope → same file."""
        h = e.content_hash
        eid = e.id if e.id.startswith(f"{self.tier}:") else f"{self.tier}:{h}"
        stored = {**e.__dict__, "id": eid}
        (self._entries_dir() / f"{h}.json").write_text(json.dumps(stored, ensure_ascii=False, indent=2))
        self.reindex()

    def tombstones(self) -> dict:
        f = self._tombstone_file()
        if not f.is_file():
            return {}
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}

    def tombstone(self, content_hash: str, to_serial: str) -> None:
        """Mark promoted-out (closes the §7 resurrection loop) + drop the live entry file."""
        ts = self.tombstones()
        ts[content_hash] = to_serial
        self._tombstone_file().write_text(json.dumps(ts, ensure_ascii=False, indent=2))
        entry_file = self._entries_dir() / f"{content_hash}.json"
        if entry_file.is_file():
            entry_file.unlink()
        self.reindex()

    # ── storage helpers (runtime-resolved paths) ─────────────────────────────
    def _root(self) -> Path:
        return resolve_c_root(self._explicit_root)

    def _entries_dir(self) -> Path:
        directory = self._root() / "entries"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _tombstone_file(self) -> Path:
        return self._root() / ".tombstones.json"

    @staticmethod
    def _to_entry(data: dict) -> Entry:
        # content_hash is a computed property — never stored/passed to the ctor.
        entry_data = {key: value for key, value in data.items() if key != "content_hash"}
        return Entry(**entry_data)


def make_cannbot_c(root: Optional[str] = None, providers: Optional[list] = None) -> CannbotCProvider:
    """Factory mirroring scan's make_autoport_c — for `build_arbiter(tmp, cannbot_c=make_cannbot_c(...))`."""
    return CannbotCProvider(root=root, providers=providers)
