# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for schema_norm._resolve_perf_threshold delegation to PerfGateProfile.

Phase B2 of PERF_GATE_PROFILE_DESIGN. Verifies:
- No marker → plugin band-aware path unchanged (PR #21 behavior preserved)
- PRECISION_ONLY marker → finalize_threshold=0.0 returned (gate accepts any)
- HERO_OP_STRICT marker → finalize_threshold=0.9 returned
- custom_t<N> marker → finalize_threshold=N returned
- DEFAULT marker (finalize_threshold=None) → falls through to plugin
- Profile resolution failure → falls through to plugin (no hard error)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_gate import write_profile_marker  # noqa: E402
from schema_norm import _resolve_perf_threshold  # noqa: E402


# ---- Profile-driven threshold (B2 new behavior) ----


def test_precision_only_profile_yields_zero_threshold(tmp_path):
    """--perf-threshold=0 marker → gate accepts any ratio (threshold=0.0)."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    write_profile_marker(ws, perf_threshold=0)
    assert _resolve_perf_threshold(ws, vj={}) == 0.0


def test_hero_op_strict_profile_yields_0_9(tmp_path):
    ws = tmp_path / "test_op"
    ws.mkdir()
    write_profile_marker(ws, perf_threshold=0.9)
    assert _resolve_perf_threshold(ws, vj={}) == 0.9


def test_custom_threshold_marker_yields_synthesized_value(tmp_path):
    ws = tmp_path / "test_op"
    ws.mkdir()
    write_profile_marker(ws, perf_threshold=0.45)
    assert _resolve_perf_threshold(ws, vj={}) == 0.45


def test_custom_low_threshold_marker(tmp_path):
    """A user passing --perf-threshold=0.15 (below all band thresholds) is honored."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    write_profile_marker(ws, perf_threshold=0.15)
    assert _resolve_perf_threshold(ws, vj={}) == 0.15


# ---- No marker → plugin band-aware path unchanged ----


def test_no_marker_ascendc_workspace_falls_through_to_default(tmp_path):
    """AscendC workspace → 1.0 parity default.

    Owner-directed 2026-07-21: the DEFAULT (base plugin, paradigm-uniform
    AscendC) perf target was raised 0.6 → 1.0 (parity) so op-gen optimizes
    harder toward parity. Explicit profile thresholds remain covered below.
    """
    ws = tmp_path / "test_op"  # no legacy backend suffix
    ws.mkdir()
    # No marker, no plugin override → 1.0 parity default
    threshold = _resolve_perf_threshold(ws, vj={})
    # Base default for AscendC is parity, 1.0.
    assert threshold == 1.0


# ---- Profile + plugin interaction ----


def test_profile_threshold_overrides_plugin_band_aware(tmp_path):
    """Profile finalize_threshold takes precedence even when plugin would
    return a different band-aware value.
    """
    ws = tmp_path / "test_NLLLoss"
    ws.mkdir()
    # User explicitly requests --perf-threshold=0.8
    write_profile_marker(ws, perf_threshold=0.8)
    # Explicit profile wins over the default.
    assert _resolve_perf_threshold(ws, vj={}) == 0.8


def test_malformed_marker_falls_through_safely(tmp_path):
    """Malformed marker → resolve_profile returns DEFAULT (finalize_threshold=None)
    → falls through to plugin band-aware. No crash.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    # Write malformed marker
    (ws / ".perf_gate_profile.json").write_text("not valid JSON {")
    # Should fall through to the AscendC parity default.
    threshold = _resolve_perf_threshold(ws, vj={})
    assert threshold == 1.0


def test_default_marker_falls_through_to_plugin(tmp_path):
    """If marker is DEFAULT-shaped (finalize_threshold=None), plugin path taken."""
    ws = tmp_path / "test_NLLLoss"
    ws.mkdir()
    # Manually write a DEFAULT-equivalent marker
    (ws / ".perf_gate_profile.json").write_text(json.dumps(
        {"name": "default", "finalize_threshold": None}
    ))
    # Falls through to the AscendC parity default.
    assert _resolve_perf_threshold(ws, vj={}) == 1.0
