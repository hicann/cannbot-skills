# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Contract + behavior tests for the cannbot c-tier adapter.

Run: PYTHONPATH=<engine> python3 src/scripts/kb_tiering/tests/test_cannbot_c.py
 or: pytest this file.
"""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # engine/src/scripts on path

from kb_tiering.interface import Entry, KBProvider, Arbiter, full_sig
from kb_tiering.adapters.cannbot_c import make_cannbot_c, resolve_c_root

LOGGER = logging.getLogger(__name__)


def _exp(claim, key="", scope=None):
    return Entry(id="", tier="customer", role="user-local", kind="experience",
                 claim=claim, key=key, scope=scope or {}, evidence={"note": "site"})


def test_isinstance_kbprovider():
    with tempfile.TemporaryDirectory() as d:
        assert isinstance(make_cannbot_c(d), KBProvider)


def test_put_resolve_lookup_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = make_cannbot_c(d)
        e = _exp("lane wedges under high load then restart clears it", key="lane_wedge restart")
        p.put(e)
        got = p.resolve(full_sig(e.claim))
        assert got is not None and "wedge" in got.claim
        assert p.lookup("lane_wedge restart") is not None


def test_persist_reload():
    with tempfile.TemporaryDirectory() as d:
        make_cannbot_c(d).put(_exp("aclInit returns 500000 unless set_env sourced", key="500000 set_env"))
        # fresh instance, same root → survives (upgrade-safe persistence)
        p2 = make_cannbot_c(d)
        assert p2.lookup("500000 set_env") is not None
        assert len(p2.index_rows()) == 1   # orphan-free


def test_idempotent_by_content_hash():
    with tempfile.TemporaryDirectory() as d:
        p = make_cannbot_c(d)
        e = _exp("same lesson same scope", key="same_lesson")
        p.put(e)
        p.put(e)
        assert len(list((Path(d) / "entries").glob("*.json"))) == 1


def test_admit_delegates_to_gate():
    with tempfile.TemporaryDirectory() as d:
        p = make_cannbot_c(d)
        # unbacked positive_pattern (no evidence) → gate rejects
        bad = Entry(id="", tier="customer", role="user-local", kind="positive_pattern",
                    claim="do X it works", evidence={})
        ok, why = p.admit(bad)
        assert not ok and "evidence" in why
        # experience with evidence → admitted
        ok2, _ = p.admit(_exp("host class Y wedges under Z", key="host_y wedge"))
        assert ok2


def test_tombstone_resurrection_guard():
    with tempfile.TemporaryDirectory() as d:
        p = make_cannbot_c(d)
        e = _exp("reworded lesson about buffer overrun repro", key="buffer_overrun")
        p.put(e)
        assert p.admit(e)[0]                       # admits pre-tombstone
        p.tombstone(e.content_hash, "OL-42")       # promoted out
        ok, why = p.admit(e)                       # re-learn same lesson
        assert not ok and "tombstone" in why.lower()   # §7 resurrection loop closed
        assert p.lookup("buffer_overrun") is None      # live entry removed


def test_arbiter_two_tier_c_over_b():
    """c>b precedence via the real Arbiter with a minimal in-memory b stub."""
    with tempfile.TemporaryDirectory() as d:
        c = make_cannbot_c(d)
        c.put(_exp("target host ip is REDACTED_IP for this deployment", key="host_ip"))

        class _BStub:                      # minimal read-only b (official-canonical)
            tier, role = "b", "official-canonical"

            @staticmethod
            def resolve(sig):
                e = Entry(id="OL-1", tier="b", role="official-canonical", kind="experience",
                          claim="target host ip is REDACTED_IP default", key="host_ip")
                return e if any(w in sig for w in ("host", "target", "deployment")) else None

            @staticmethod
            def lookup(k):
                return None

            @staticmethod
            def index_rows():
                return []

            def reindex(self):
                pass

            @staticmethod
            def admit(e):
                return True, "ok"

            def put(self, e):
                pass

            @staticmethod
            def tombstones():
                return {}

            def tombstone(self, h, s):
                pass

        arb = Arbiter([c, _BStub()])       # ordered: c highest
        r = arb.resolve(full_sig("target host ip for deployment"))
        assert r.status == "RESOLVED" and r.tier == "customer"   # local c overrides b (experience, silent)
        assert "REDACTED_IP" in r.entry.claim


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        LOGGER.info("  [OK] %s", fn.__name__)
    LOGGER.info("ALL %d cannbot-c contract tests PASS", len(fns))


if __name__ == "__main__":
    _run()
