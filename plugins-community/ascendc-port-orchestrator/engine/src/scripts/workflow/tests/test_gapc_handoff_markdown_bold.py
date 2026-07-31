# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""gap(c) 2026-06-16 (celu port_a3, independent review-caught): a worker handoff whose
verdict token is in markdown bold — `→ orchestrator: **done**, precision ...` —
was classified "unknown" by `_parse_worker_signal` and missed by `handoff_match`
(prefix), causing a spurious `await_worker → abort` BEFORE O5 ran.

Root cause: inline `**` sits between the space and the token (`: **done`), so
`endswith("done")` / `" done" in h` / `startswith("→ orchestrator: done")` all
miss. Fix: strip inline `**` before verdict recognition (both `_parse_worker_signal`
and the `handoff_match`/`handoff_contains` eval branches).

These tests use the EXACT failing celu handoff string.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import state_machine as sm  # noqa: E402


# The exact string that aborted celu (state_transitions.jsonl, await_worker→abort).
CELU_BOLD_DONE = (
    "→ orchestrator: **done**, precision **9/9 T1 PASS** "
    "(fp16, bit-exact, max_abs_diff=0.0), perf **N/A** (A3 ref MISSING_ENTRY)."
)


# ---------------------------------------------------------------------------
# _parse_worker_signal
# ---------------------------------------------------------------------------
def test_bold_done_classified_done():
    """The exact celu handoff (markdown-bold **done**) must classify as done."""
    assert getattr(sm, '_parse_worker_signal')(CELU_BOLD_DONE) == "done"


def test_plain_done_still_classified_done():
    assert getattr(sm, '_parse_worker_signal')("→ orchestrator: done, precision 9/9 PASS") == "done"


def test_trailing_done_still_classified_done():
    assert getattr(sm, '_parse_worker_signal')("→ orchestrator: done") == "done"


def test_bold_abort_classified_abort():
    assert getattr(sm, '_parse_worker_signal')("→ orchestrator: **abort** — build failed") == "abort"


def test_bold_partial_persist_classified():
    assert getattr(sm, '_parse_worker_signal')(
        "→ orchestrator: **PARTIAL_PERSIST** (1 dtype below floor)"
    ) == "partial_persist"


def test_non_done_handoff_not_misclassified():
    """Bold-strip must NOT over-match: a probe-routing handoff is not 'done'."""
    sig = getattr(sm, '_parse_worker_signal')("→ aog-precision-probe: precision stuck on bf16")
    assert sig != "done"
    assert sig != "abort"


def test_single_star_not_stripped():
    """Only paired `**` is stripped; a lone `*` (e.g. a 5*8 shape note) is intact
    and must not fabricate a verdict.
    """
    # No verdict token present → unknown; the lone '*' must not change that.
    assert getattr(sm, '_parse_worker_signal')("→ aog-kernel-worker: tiling 5*8 in progress") == "unknown"


# ---------------------------------------------------------------------------
# eval_condition handoff_match / handoff_contains
# ---------------------------------------------------------------------------
def test_handoff_match_prefix_bold_done():
    """handoff_match `→ orchestrator: done` (prefix) must match the bolded form."""
    ctx = {"handoff": CELU_BOLD_DONE}
    assert sm.eval_condition({"handoff_match": "→ orchestrator: done"}, ctx) is True


def test_handoff_match_prefix_plain_done_still_matches():
    ctx = {"handoff": "→ orchestrator: done, precision 9/9 PASS"}
    assert sm.eval_condition({"handoff_match": "→ orchestrator: done"}, ctx) is True


def test_handoff_contains_bold_done():
    ctx = {"handoff": CELU_BOLD_DONE}
    assert sm.eval_condition({"handoff_contains": "done"}, ctx) is True


def test_handoff_match_negative_still_false():
    """A non-matching prefix must still be False after bold-strip (no false positive)."""
    ctx = {"handoff": CELU_BOLD_DONE}
    assert sm.eval_condition({"handoff_match": "→ aog-precision-probe:"}, ctx) is False
