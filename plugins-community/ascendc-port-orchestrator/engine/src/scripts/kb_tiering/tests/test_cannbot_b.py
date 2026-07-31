# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""cannbot b-tier adapter tests — reads the REAL bundled references KB + Arbiter([c, b]) e2e."""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # engine/src/scripts

from kb_tiering.interface import Entry, KBProvider, Arbiter, full_sig
from kb_tiering.adapters.cannbot_b import make_cannbot_b
from kb_tiering.adapters.cannbot_c import make_cannbot_c

LOGGER = logging.getLogger(__name__)


def test_b_isinstance_and_reads_real_bundled_kb():
    b = make_cannbot_b()                          # default: bundled src/skills/references
    assert isinstance(b, KBProvider)
    rows = b.index_rows()
    assert len(rows) > 50, f"expected many b entries from real KB_INDEX, got {len(rows)}"
    assert all(getattr(e, "role", None) == "official-canonical" for e in rows)


def test_b_resolve_hits_real_entry():
    b = make_cannbot_b()
    # a signature built from real KB content (TQue/DataCopy are core AscendC terms in the KB)
    hit = b.resolve(full_sig("TQue TBuf DataCopy sync pipeline"))
    assert hit is not None and hit.tier == "b"


def test_b_is_promotion_gate_only():
    b = make_cannbot_b()
    e = Entry(id="", tier="b", role="official-canonical", kind="experience", claim="x")
    ok, why = b.admit(e)
    assert not ok and "promot" in why.lower()    # b rejects direct writes


def test_arbiter_c_over_b_real_b():
    """Real two-tier: c (user-local) overrides b (bundled) on the same experience key."""
    with tempfile.TemporaryDirectory() as d:
        c = make_cannbot_c(d)
        c.put(Entry(id="", tier="customer", role="user-local", kind="experience",
                    claim="on this deployment DataCopy alignment quirk needs 32B pad",
                    key="datacopy_align", evidence={"note": "site"}))
        b = make_cannbot_b()
        arb = Arbiter([c, b])                     # c highest precedence
        r = arb.resolve(full_sig("DataCopy alignment pad deployment"))
        assert r.status in ("RESOLVED", "CONFLICT_SURFACED")
        if r.status == "RESOLVED":
            assert r.tier == "customer"           # local c wins for experience (silent override)


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        LOGGER.info("  [OK] %s", fn.__name__)
    LOGGER.info("ALL %d cannbot-b tests PASS", len(fns))


if __name__ == "__main__":
    _run()
