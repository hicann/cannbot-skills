# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0kk (2026-05-29): consume user_decision.md after applying, to break the
finalize<->await_user_decision infinite loop.

Origin: FA-class 3_FusionAttention 2026-05-29 regenerate-validation. The
orchestrator pause branch advances when user_decision.md is present (P0p) but
never CONSUMED it. A decision with `next_state: finalize`, while finalize kept
rolling back on the SAME O5 count gate (and await_worker is excluded for FA
no-kw), re-advanced the identical stale decision every main-loop iteration ->
infinite loop (observed 99x, ~hours of wall-clock burned).

Fix: _consume_applied_user_decision renames user_decision.md ->
.user_decision_consumed.md right after the route is applied, so a loop that
returns to await_user_decision finds no file -> genuine PAUSE instead of
infinite stale re-advance.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH = _HERE.parent.parent.parent / "orchestrator"
sys.path.insert(0, str(_ORCH))

import orchestrator as orch  # noqa: E402


def test_consume_renames_present_decision(tmp_path):
    ud = tmp_path / "user_decision.md"
    ud.write_text("next_state: finalize\nreason: test\n")
    assert getattr(orch, '_consume_applied_user_decision')(tmp_path) is True
    assert not ud.exists(), "user_decision.md must be gone after consume"
    consumed = tmp_path / ".user_decision_consumed.md"
    assert consumed.exists()
    assert "finalize" in consumed.read_text()


def test_consume_noop_when_absent(tmp_path):
    # No file present: must be a safe no-op (no error), returns False.
    assert getattr(orch, '_consume_applied_user_decision')(tmp_path) is False


def test_loop_broken_second_visit_has_no_file(tmp_path):
    """The infinite-loop guard: after one consume, the file is gone, so a loop
    that returns to await_user_decision cannot re-advance the same stale
    decision (the orchestrator pause guard `exists() and st_size>0` is now
    False -> it PAUSES/exits instead of looping).
    """
    ud = tmp_path / "user_decision.md"
    ud.write_text("next_state: finalize\nreason: loops back to finalize\n")
    # First visit: present -> consumed.
    assert getattr(orch, '_consume_applied_user_decision')(tmp_path) is True
    assert not ud.exists()
    # Second visit (loop returned to await_user_decision): file gone -> not
    # re-consumable -> the stale decision can never re-advance. Loop broken.
    assert getattr(orch, '_consume_applied_user_decision')(tmp_path) is False
