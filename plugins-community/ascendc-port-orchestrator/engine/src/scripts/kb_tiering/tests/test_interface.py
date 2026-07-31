# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Smoke tests for the KB-tiering CONTRACT (interface.py) — the stable surface adapters build on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interface import Entry, gate, Arbiter, KBProvider, hard_key, full_sig  # noqa: E402


class Mem:
    def __init__(self, tier, role):
        self.tier, self.role = tier, role
        self.entries_by_hash = {}
        self.tombstones_by_hash = {}

    def resolve(self, sig):
        best, bs = None, 0.0
        for e in self.entries_by_hash.values():
            from interface import jaccard
            j = jaccard(sig, full_sig(e.claim))
            if j > bs:
                best, bs = e, j
        return best if bs >= 0.3 else None

    def lookup(self, key):
        return next((e for e in self.entries_by_hash.values() if e.key == key), None)

    def index_rows(self):
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

    def entries(self):
        return list(self.entries_by_hash.values())


def _pos(claim, **kw):
    return Entry(id="customer:x", tier="c", role="user-local", kind="positive_pattern",
                 key=hard_key(claim), claim=claim, evidence={"ab": "+2%"}, trust="verified", **kw)


def test_entry_content_hash_folds_scope():
    a = _pos("use chunk 256", scope={"applies_to": "elementwise"})
    b = _pos("use chunk 256", scope={"applies_to": "reduction"})
    assert a.content_hash != b.content_hash          # scope-distinct → distinct (scan②)


def test_isinstance_provider():
    assert isinstance(Mem("c", "user-local"), KBProvider)


def test_gate_rejects_unbacked_anti():
    e = Entry(id="c:1", tier="c", role="user-local", kind="anti_pattern", claim="X fails", evidence={})
    ok, why = gate(e, "admission", [])
    assert not ok and "evidence" in why


def test_gate_rejects_config_kind():
    e = Entry(id="c:1", tier="c", role="user-local", kind="site_config", claim="ip", evidence={"x": 1})
    ok, why = gate(e, "admission", [])
    assert not ok and "NOT KB" in why


def test_arbiter_write_admit_then_resolve():
    c = Mem("c", "user-local")
    arb = Arbiter([c])
    st, _id = arb.write(_pos("use chunk 256 tiling for D192 elementwise A5"), "c")
    assert st == "WRITTEN"
    r = arb.resolve(full_sig("chunk 256 tiling D192 elementwise"))
    assert r.status == "RESOLVED" and r.entry is not None


def test_arbiter_conflict_surfaced():
    c, b = Mem("c", "user-local"), Mem("b", "official-canonical")
    b.put(Entry(id="OL-1", tier="b", role="official-canonical", kind="positive_pattern",
                key=hard_key("chunk 512"), claim="use chunk 512 for D192 elementwise A5",
                evidence={"canon": True}, trust="verified"))
    c.put(_pos("use chunk 256 for D192 elementwise A5"))
    arb = Arbiter([c, b])                                  # c > b
    r = arb.resolve(full_sig("chunk D192 elementwise A5"))
    assert r.status == "CONFLICT_SURFACED" and r.conflict_with is not None   # not silent (§5)


def test_arbiter_promote_tombstones_source():
    c, b = Mem("c", "user-local"), Mem("b", "official-canonical")
    arb = Arbiter([c, b])
    e = _pos("regbase fuse gz gu gd chain")
    arb.write(e, "c")
    st, new_id = arb.promote(e, "c", "b")
    assert st == "PROMOTED"
    assert e.content_hash in c.tombstones()               # resurrection guard (§7)
    ok, why = gate(e, "admission", [c, b])                # re-learn same → blocked
    assert not ok and "tombstoned" in why
