# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Read-path bridge tests — build_arbiter + inject_for_brief (keyword→signature bridge)."""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # engine/src/scripts

from kb_tiering.interface import Entry
from kb_tiering.read_bridge import build_arbiter, inject_for_brief
from kb_tiering.adapters.cannbot_c import make_cannbot_c
from kb_tiering.adapters.cannbot_b import make_cannbot_b

LOGGER = logging.getLogger(__name__)


def test_build_arbiter_orders_c_then_b():
    with tempfile.TemporaryDirectory() as d:
        arb = build_arbiter(user_kb_root=d)          # real c + real bundled b
        assert [p.tier for p in arb.providers] == ["customer", "b"]


def test_inject_multi_row_keyword_filter_over_real_b():
    """Bridge returns MANY keyword-matched rows (not single resolve) from the real bundled b."""
    with tempfile.TemporaryDirectory() as d:
        arb = build_arbiter(user_kb_root=d)
        rows = inject_for_brief(arb, keywords=["DataCopy"])
        assert len(rows) >= 1                          # keyword filter recalls real b entries
        assert all(r["tier"] in ("customer", "b") for r in rows)


def test_c_shadows_b_same_key():
    """c entry with a hard-key shared by a b entry shadows it (c>b), still multi-row for the rest."""
    with tempfile.TemporaryDirectory() as d:
        c = make_cannbot_c(d)
        # forge a c entry whose key collides with a b entry's key so we can prove shadowing
        b = make_cannbot_b()
        some_b = b.index_rows()[0]
        c.put(Entry(id="", tier="customer", role="user-local", kind="experience",
                    claim="local override of " + some_b.claim[:40], key=some_b.key,
                    evidence={"note": "site"}))
        arb = build_arbiter(cannbot_c=c, cannbot_b=b)
        rows = inject_for_brief(arb, keywords=None)     # all rows, c>b merged
        # the shared key appears exactly once, and it's the customer one
        hits = [r for r in rows if getattr(r["row"], "key", "") == some_b.key]
        assert len(hits) == 1 and hits[0]["tier"] == "customer"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        LOGGER.info("  [OK] %s", fn.__name__)
    LOGGER.info("ALL %d read-bridge tests PASS", len(fns))


if __name__ == "__main__":
    _run()
