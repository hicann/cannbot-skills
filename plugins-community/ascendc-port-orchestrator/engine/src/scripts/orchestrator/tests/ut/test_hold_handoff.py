# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""FOLLOWUPS v3.1 A.2 (2026-08-30, 2_FFN_evo lesson): the `hold` handoff.

A worker whose mandate requires NO code change (e.g. waiting on operator-supplied
evidence or a pending external probe) previously had no honest exit handoff:
improvised lines were rejected as malformed by extract_canonical_handoff or fell
through routing. `→ orchestrator: hold — <reason>` is now a canonical keyword.
Routing is unchanged: `hold` matches no specific YAML rule and lands on the
P0abk `→ orchestrator:` catch-all → await_user_decision with the reason preserved.

These tests pin (a) recognizer acceptance of both arrow and @-mention forms,
(b) the prefix-matching caveat the brief warns about (variants like
`probe-pending hold` are NOT recognized), and (c) downstream catch-all routing.
"""
from __future__ import annotations

import handoff_audit
import state_machine as sm

_VALID_ARROW_KEYWORDS = getattr(handoff_audit, "_VALID_ARROW_KEYWORDS")


def test_hold_is_a_valid_arrow_keyword():
    assert "hold" in _VALID_ARROW_KEYWORDS


def test_arrow_hold_handoff_is_accepted_verbatim():
    raw = "Some prose...\n→ orchestrator: hold — waiting on operator int8 probe evidence\n"
    out = handoff_audit.extract_canonical_handoff(raw)
    assert out == "→ orchestrator: hold — waiting on operator int8 probe evidence"


def test_at_form_hold_normalizes_to_arrow_form():
    raw = "Final line:\n@orchestrator: hold — no code change this spawn\n"
    out = handoff_audit.extract_canonical_handoff(raw)
    assert out.startswith("→ orchestrator: hold")


def test_hold_variant_prefix_is_not_recognized():
    """The router prefix-matches the keyword: `probe-pending hold` must stay
    unextractable so the brief's exact-prefix instruction is load-bearing.
    """
    raw = "→ orchestrator: probe-pending hold — waiting\n"
    out = handoff_audit.extract_canonical_handoff(raw)
    assert not out.startswith("→ orchestrator: hold")


def test_hold_routes_via_orchestrator_catch_all():
    """End-to-end: the extracted hold line downstream-matches the P0abk
    catch-all condition (`handoff_match: "→ orchestrator:"`) whose goto is
    await_user_decision in opgen_state_machine.yaml.
    """
    extracted = handoff_audit.extract_canonical_handoff(
        "→ orchestrator: hold — waiting on operator probe\n"
    )
    ctx = {"handoff": extracted, "snapshot": {}, "iter_counts": {}, "ws": None, "sm": {}}
    assert sm.eval_condition({"handoff_match": "→ orchestrator:"}, ctx) is True
    # ...and matches nothing more specific than the catch-all.
    assert sm.eval_condition({"handoff_match": "→ orchestrator: done"}, ctx) is False
    assert sm.eval_condition(
        {"handoff_match": "→ orchestrator: await_user_decision"}, ctx
    ) is False
