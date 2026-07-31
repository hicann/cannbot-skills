# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""c>b>a read-path wiring: _c_tier_lessons_block is config-gated (default-b byte-unchanged)."""
import os
import tempfile
from pathlib import Path


def test_default_b_block_empty_when_no_user_kb(monkeypatch):
    """No c-tier user_kb → block is empty → kb_manifest_block stays byte-unchanged."""
    monkeypatch.delenv("ASCENDC_PORT_USER_KB", raising=False)
    # point the default c-root probe at a nonexistent dir so a stray real ~/.ascendc-port doesn't leak in
    monkeypatch.setenv("HOME", tempfile.mkdtemp())
    from briefs._common import _c_tier_lessons_block
    assert _c_tier_lessons_block("gelu", None, "a5") == ""


def test_c_tier_injected_when_user_kb_active(monkeypatch):
    """c-tier user_kb with a matching lesson → block carries it, c>b>a header."""
    import sys
    scripts_root = Path(__file__).resolve().parents[3]  # src/scripts
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("ASCENDC_PORT_USER_KB", d)
        from kb_tiering.adapters.cannbot_c import make_cannbot_c
        from kb_tiering.interface import Entry
        make_cannbot_c(d).put(Entry(
            id="", tier="customer", role="user-local", kind="experience",
            claim="gelu on this deployment needs 32B DataCopy pad on tail",
            key="gelu_tail_pad", evidence={"note": "site"}))
        from briefs._common import _c_tier_lessons_block
        blk = _c_tier_lessons_block("gelu", None, "a5")
        assert "32B DataCopy pad" in blk
        assert "c>b>a" in blk
