# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""e2e: a5_ops adapter (real KB_INDEX b-tier) + mock c-tier through the Arbiter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interface import Entry, gate, Arbiter, KBProvider, hard_key, full_sig, jaccard  # noqa: E402
from adapter_a5ops import A5opsProvider  # noqa: E402

_KBINDEX = Path(__file__).resolve().parents[5] / "kb/KB_INDEX.md"


class MemC:
    tier, role = "c", "user-local"

    def __init__(self):
        self.entries_by_hash = {}
        self.tombstones_by_hash = {}

    def resolve(self, sig):
        best, bs = None, 0.0
        for e in self.entries_by_hash.values():
            j = jaccard(sig, full_sig(e.claim))
            if j > bs:
                best, bs = e, j
        return best if bs >= 0.3 else None

    def lookup(self, k):
        return next((e for e in self.entries_by_hash.values() if jaccard(k, e.key) >= 0.6), None)

    def index_rows(self):
        return list(self.entries_by_hash.values())

    def entries(self):
        return list(self.entries_by_hash.values())

    def reindex(self):
        pass

    def admit(self, e):
        return gate(e, "admission", [self])

    def put(self, e):
        self.entries_by_hash[e.content_hash] = e

    def tombstones(self):
        return self.tombstones_by_hash

    def tombstone(self, h, s):
        self.tombstones_by_hash[h] = s


def test_a5ops_adapter_parses_real_kbindex(tmp_path: Path):
    b = A5opsProvider(_KBINDEX, inbox=tmp_path / "promoted_inbox.jsonl")
    assert b.parse_total > 100
    assert b.parse_ok / b.parse_total >= 0.85          # ≥85% of index rows → Entry
    assert isinstance(b, KBProvider)


def test_a5ops_resolve_hits_real_entry(tmp_path: Path):
    b = A5opsProvider(_KBINDEX, inbox=tmp_path / "promoted_inbox.jsonl")
    r = b.resolve(full_sig("arch35 wrap cheat include kernel"))
    assert r is not None and r.tier == "b" and r.trust == "verified"


def test_a5ops_direct_write_rejected_promotion_only(tmp_path: Path):
    b = A5opsProvider(_KBINDEX, inbox=tmp_path / "promoted_inbox.jsonl")
    e = Entry(id="x", tier="b", role="official-canonical", kind="positive_pattern",
              key="k", claim="c", evidence={"x": 1}, trust="verified")
    ok, why = b.admit(e)
    assert not ok and "promotion-gate-only" in why


def test_e2e_arbiter_c_over_b_and_promote(tmp_path: Path):
    inbox = tmp_path / "promoted_inbox.jsonl"
    b = A5opsProvider(_KBINDEX, inbox=inbox)
    c = MemC()
    arb = Arbiter([c, b])                               # c > b
    ce = Entry(id="customer:e1", tier="c", role="user-local", kind="positive_pattern",
               scope={"applies_to": "elementwise", "chip_scope": "A5"},
               key=hard_key("chunk 256 tiling D192"), claim="chunk 256 tiling for D192 elementwise A5",
               evidence={"ab": "+2%"}, trust="verified")
    assert arb.write(ce, "c")[0] == "WRITTEN"
    r = arb.resolve(full_sig(ce.claim))
    assert r.status in ("RESOLVED", "CONFLICT_SURFACED") and r.entry.tier == "c"   # c wins precedence
    st, new_id = arb.promote(ce, "c", "b")
    assert st == "PROMOTED" and new_id.startswith("b:")
    assert ce.content_hash in c.tombstones()           # resurrection guard
    # promoted entry now visible in b
    b2 = A5opsProvider(_KBINDEX, inbox=inbox)
    assert any(e.claim == ce.claim for e in b2.index_rows())
