# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-103 regression: extract_canonical_handoff normalizes arrow-form `→ aog-X`
to @-form `@aog-X` so state_machine handoff_match (which keys on @-form) hits.

Caught 2026-05-20 by independent review 10_LayerNorm E2E under PR #21 paradigm-native flow:
kw-1 emitted `→ aog-kernel-optimizer:` instead of `@aog-kernel-optimizer`, the
parser didn't recognize the arrow form, orchestrator aborted with
"no recognized handoff — contract violation".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from orchestrator import extract_canonical_handoff  # noqa: E402


@pytest.mark.parametrize(
    "arrow_line,expected_at_form",
    [
        ("→ aog-kernel-optimizer", "@aog-kernel-optimizer"),
        ("→ aog-precision-probe", "@aog-precision-probe"),
        ("→ aog-fused-optimizer", "@aog-fused-optimizer"),
        ("→ aog-determinism-analyzer", "@aog-determinism-analyzer"),
        ("→ aog-researcher", "@aog-researcher"),
        # With trailing colon + content
        ("→ aog-kernel-optimizer: perf 0.375x", "@aog-kernel-optimizer: perf 0.375x"),
        # With markdown wrap (P0h-style)
        ("**Exit handoff**: `→ aog-kernel-optimizer`", "@aog-kernel-optimizer"),
        # In multi-line block: last line wins
        (
            "Some prose\n→ aog-precision-probe: investigate fp16 ULP\n",
            "@aog-precision-probe: investigate fp16 ULP",
        ),
    ],
)
def test_arrow_form_normalized_to_at_form(arrow_line, expected_at_form):
    """Arrow form `→ aog-X` must normalize to `@aog-X` in returned handoff."""
    result = extract_canonical_handoff(arrow_line)
    assert result == expected_at_form, (
        f"arrow form not normalized: input={arrow_line!r}, "
        f"got={result!r}, expected={expected_at_form!r}"
    )


def test_at_form_still_works():
    """Existing @-form must still pass through unchanged."""
    out = extract_canonical_handoff("@aog-kernel-optimizer: perf 0.375x REDUCTION")
    assert out == "@aog-kernel-optimizer: perf 0.375x REDUCTION"


def test_arrow_orchestrator_done_still_works():
    """Existing `→ orchestrator: done` form must still pass through unchanged."""
    out = extract_canonical_handoff("→ orchestrator: done — 8/8 PASS")
    assert out == "→ orchestrator: done — 8/8 PASS"


def test_arrow_orchestrator_wraps_at_form_still_works():
    """P0s case: `→ orchestrator: handoff to @aog-X` still unwraps to @-form."""
    out = extract_canonical_handoff(
        "→ orchestrator: handoff to @aog-kernel-optimizer per V3.8.4"
    )
    assert out == "@aog-kernel-optimizer per V3.8.4"
