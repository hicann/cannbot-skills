# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Read-path bridge tests — build_arbiter + inject_for_brief (keyword→signature bridge, c-only)."""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # engine/src/scripts

from kb_tiering.interface import Entry
from kb_tiering.read_bridge import build_arbiter, inject_for_brief
from kb_tiering.adapters.cannbot_c import make_cannbot_c

LOGGER = logging.getLogger(__name__)


def test_build_arbiter_is_c_only():
    """OKF-only 后 b-tier 已摘除：Arbiter 只组装 user-kb（c-tier）。"""
    with tempfile.TemporaryDirectory() as d:
        arb = build_arbiter(user_kb_root=d)
        assert [p.tier for p in arb.providers] == ["customer"]


def test_gate_providers_wired_to_c():
    """c 的 admit-gate 上下文即 [c]：tombstone 过的条目再写入被 gate 拒绝（resurrection guard）。"""
    with tempfile.TemporaryDirectory() as d:
        c = make_cannbot_c(d)
        arb = build_arbiter(cannbot_c=c)
        e = Entry(id="", tier="customer", role="user-local", kind="experience",
                  claim="gelu tail needs 32B pad", key="gelu_tail_pad",
                  evidence={"note": "site"})
        assert arb.write(e, "customer")[0] == "WRITTEN"
        c.tombstone(e.content_hash, "customer:superseded")
        status, why = arb.write(e, "customer")
        assert status == "REJECTED" and "tombstoned" in why


def test_inject_multi_row_keyword_filter_over_c():
    """Bridge returns MANY keyword-matched rows (not single resolve) from the c-tier."""
    with tempfile.TemporaryDirectory() as d:
        c = make_cannbot_c(d)
        c.put(Entry(id="", tier="customer", role="user-local", kind="experience",
                    claim="gelu on this deployment needs 32B DataCopy pad on tail",
                    key="gelu_tail_pad", evidence={"note": "site"}))
        c.put(Entry(id="", tier="customer", role="user-local", kind="experience",
                    claim="DataCopy alignment requires 512B multiples on arch35",
                    key="datacopy_align", evidence={"note": "site"}))
        c.put(Entry(id="", tier="customer", role="user-local", kind="experience",
                    claim="unrelated softmax tiling note",
                    key="softmax_tile", evidence={"note": "site"}))
        arb = build_arbiter(cannbot_c=c)
        rows = inject_for_brief(arb, keywords=["DataCopy"])
        assert len(rows) == 2                          # keyword filter recalls both c lessons
        assert all(r["tier"] == "customer" for r in rows)
        rows_all = inject_for_brief(arb, keywords=None)  # empty keywords → all rows
        assert len(rows_all) == 3


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        LOGGER.info("  [OK] %s", fn.__name__)
    LOGGER.info("ALL %d read-bridge tests PASS", len(fns))


if __name__ == "__main__":
    _run()
