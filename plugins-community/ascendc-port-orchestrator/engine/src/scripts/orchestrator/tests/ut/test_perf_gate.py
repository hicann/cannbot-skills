# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for perf_gate.py (Phase A foundation).

Covers:
- Built-in profiles' field values
- Marker round-trip (write → read)
- Threshold → profile mapping (0 / 0.9 / arbitrary)
- escalation_overrides resolution (generic per-state profile switch)
- DEFAULT fallback on missing / malformed marker
- Custom profile synthesis + name format
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make perf_gate importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_gate import (  # noqa: E402
    DEFAULT,
    HERO_OP_STRICT,
    MARKER_FILENAME,
    PRECISION_ONLY,
    PerfGateProfile,
    _profile_for_threshold,
    remove_profile_marker,
    resolve_profile,
    synthesize_custom,
    write_profile_marker,
)


# ---- Built-in profile field invariants ----


def test_default_profile_full_perf_chain():
    """DEFAULT preserves current behavior — all gates active, threshold deferred to plugin."""
    assert DEFAULT.name == "default"
    assert DEFAULT.measure_reference_perf is True
    assert DEFAULT.include_perf_in_brief is True
    assert DEFAULT.finalize_threshold is None  # plugin band-aware
    assert DEFAULT.allow_ko_escalation is True
    assert DEFAULT.allow_ar_escalation is True
    assert DEFAULT.allow_fo_escalation is True
    assert DEFAULT.require_ratio_in_verification is True
    assert DEFAULT.require_perf_artifacts is True


def test_precision_only_profile_kills_all_perf_paths():
    """PRECISION_ONLY suppresses every perf-related gate."""
    assert PRECISION_ONLY.name == "precision_only"
    assert PRECISION_ONLY.measure_reference_perf is False
    assert PRECISION_ONLY.include_perf_in_brief is False
    assert PRECISION_ONLY.finalize_threshold == 0.0
    assert PRECISION_ONLY.allow_ko_escalation is False
    assert PRECISION_ONLY.allow_ar_escalation is False
    assert PRECISION_ONLY.allow_fo_escalation is False
    assert PRECISION_ONLY.require_ratio_in_verification is False
    assert PRECISION_ONLY.require_perf_artifacts is False


def test_hero_op_strict_uses_global_override():
    """HERO_OP_STRICT keeps full chain but raises threshold to 0.9 globally."""
    assert HERO_OP_STRICT.name == "hero_op_strict"
    assert HERO_OP_STRICT.measure_reference_perf is True
    assert HERO_OP_STRICT.finalize_threshold == 0.9
    assert HERO_OP_STRICT.allow_ko_escalation is True
    assert HERO_OP_STRICT.allow_ar_escalation is True


def test_default_has_empty_escalation_overrides():
    """DEFAULT carries no per-state profile overrides (generic hook, unused)."""
    assert DEFAULT.escalation_overrides == {}


def test_profile_is_frozen():
    """Profile dataclasses are immutable — mutation must raise."""
    with pytest.raises(Exception):  # FrozenInstanceError
        DEFAULT.allow_ko_escalation = False  # type: ignore


# ---- Threshold → profile mapping ----


def test_threshold_zero_maps_to_precision_only():
    assert _profile_for_threshold(0) is PRECISION_ONLY
    assert _profile_for_threshold(0.0) is PRECISION_ONLY


def test_threshold_0_9_maps_to_hero_op_strict_shortcut():
    assert _profile_for_threshold(0.9) is HERO_OP_STRICT


def test_threshold_arbitrary_synthesizes_custom():
    p = _profile_for_threshold(0.3)
    assert p.name == "custom_t0.30"
    assert p.finalize_threshold == 0.3
    # Clone of DEFAULT otherwise
    assert p.measure_reference_perf == DEFAULT.measure_reference_perf
    assert p.allow_ko_escalation == DEFAULT.allow_ko_escalation


def test_synthesize_custom_name_format():
    """Custom profile name uses `custom_t{N:.2f}` per main agent Q2."""
    assert synthesize_custom(0.5).name == "custom_t0.50"
    assert synthesize_custom(0.75).name == "custom_t0.75"
    assert synthesize_custom(1.0).name == "custom_t1.00"


# ---- Marker round-trip ----


def test_write_marker_for_precision_only(tmp_path):
    p = write_profile_marker(tmp_path, perf_threshold=0)
    assert p is PRECISION_ONLY
    marker = tmp_path / MARKER_FILENAME
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert data["name"] == "precision_only"
    assert data["finalize_threshold"] == 0.0


def test_write_marker_for_custom_threshold(tmp_path):
    p = write_profile_marker(tmp_path, perf_threshold=0.3)
    assert p.name == "custom_t0.30"
    data = json.loads((tmp_path / MARKER_FILENAME).read_text())
    assert data["name"] == "custom_t0.30"
    assert data["finalize_threshold"] == 0.3


def test_write_marker_with_none_returns_default_writes_no_marker(tmp_path):
    p = write_profile_marker(tmp_path, perf_threshold=None)
    assert p is DEFAULT
    # No marker file should be written
    assert not (tmp_path / MARKER_FILENAME).exists()


def test_remove_profile_marker(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    assert (tmp_path / MARKER_FILENAME).exists()
    removed = remove_profile_marker(tmp_path)
    assert removed is True
    assert not (tmp_path / MARKER_FILENAME).exists()
    # Second call returns False (nothing to remove)
    assert remove_profile_marker(tmp_path) is False


# ---- Resolver behavior ----


def test_resolve_profile_no_marker_returns_default(tmp_path):
    assert resolve_profile(tmp_path) is DEFAULT


def test_resolve_profile_with_precision_only_marker(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0)
    assert resolve_profile(tmp_path) is PRECISION_ONLY


def test_resolve_profile_with_custom_marker_round_trip(tmp_path):
    write_profile_marker(tmp_path, perf_threshold=0.45)
    p = resolve_profile(tmp_path)
    assert p.name == "custom_t0.45"
    assert p.finalize_threshold == 0.45


def test_resolve_profile_malformed_marker_falls_back_to_default(tmp_path):
    (tmp_path / MARKER_FILENAME).write_text("not valid JSON {")
    assert resolve_profile(tmp_path) is DEFAULT


def test_resolve_profile_unknown_name_falls_back_to_default(tmp_path):
    (tmp_path / MARKER_FILENAME).write_text(json.dumps({"name": "made_up_profile"}))
    assert resolve_profile(tmp_path) is DEFAULT


# ---- escalation_overrides (generic per-state hook, currently unused) ----


def test_no_override_for_state_returns_outer_profile(tmp_path):
    # current_state not in escalation_overrides (empty) → outer profile returned
    p = resolve_profile(tmp_path, current_state="await_worker")
    assert p is DEFAULT


def test_resolve_profile_without_current_state_returns_default(tmp_path):
    # current_state None → no override consideration
    p = resolve_profile(tmp_path)
    assert p is DEFAULT
    assert p.escalation_overrides == {}  # no per-state overrides configured
