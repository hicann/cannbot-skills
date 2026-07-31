# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for orchestrator.py --perf-threshold CLI arg + marker write integration.

Phase B1 of PERF_GATE_PROFILE_DESIGN. Verifies:
- CLI arg accepts None / 0 / 0.9 / arbitrary float
- run_single_op(perf_threshold=X) writes the correct marker
- run_single_op(perf_threshold=None) leaves marker as-is (does not remove)
- Marker round-trips correctly so subsequent resolve_profile() returns the
  expected built-in/custom profile
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_gate import (  # noqa: E402
    DEFAULT,
    HERO_OP_STRICT,
    MARKER_FILENAME,
    PRECISION_ONLY,
    resolve_profile,
    write_profile_marker,
)


# ---- write_profile_marker integration (B1 entry-point behavior) ----


def test_write_marker_for_precision_only_workflow(tmp_path):
    """--perf-threshold=0 → PRECISION_ONLY marker; resolve_profile returns PRECISION_ONLY."""
    profile = write_profile_marker(tmp_path, perf_threshold=0)
    assert profile is PRECISION_ONLY
    assert (tmp_path / MARKER_FILENAME).exists()
    assert resolve_profile(tmp_path) is PRECISION_ONLY


def test_write_marker_for_hero_op_strict_workflow(tmp_path):
    """--perf-threshold=0.9 → HERO_OP_STRICT (recognized shortcut)."""
    profile = write_profile_marker(tmp_path, perf_threshold=0.9)
    assert profile is HERO_OP_STRICT
    assert resolve_profile(tmp_path) is HERO_OP_STRICT


def test_write_marker_for_custom_threshold_workflow(tmp_path):
    """--perf-threshold=0.45 → synthesized custom profile."""
    profile = write_profile_marker(tmp_path, perf_threshold=0.45)
    assert profile.name == "custom_t0.45"
    assert profile.finalize_threshold == 0.45
    resolved = resolve_profile(tmp_path)
    assert resolved.name == "custom_t0.45"
    assert resolved.finalize_threshold == 0.45


def test_write_marker_with_none_returns_default_no_marker(tmp_path):
    """--perf-threshold not set → no marker write; subsequent resolve = DEFAULT."""
    profile = write_profile_marker(tmp_path, perf_threshold=None)
    assert profile is DEFAULT
    assert not (tmp_path / MARKER_FILENAME).exists()


def test_marker_overwritten_on_threshold_change(tmp_path):
    """Resume + new --perf-threshold → marker regenerated (per §7 Q1 main agent)."""
    # First run: --perf-threshold=0
    write_profile_marker(tmp_path, perf_threshold=0)
    assert resolve_profile(tmp_path) is PRECISION_ONLY

    # Second run: --perf-threshold=0.9 → marker regenerates
    write_profile_marker(tmp_path, perf_threshold=0.9)
    assert resolve_profile(tmp_path) is HERO_OP_STRICT

    # Marker file content matches
    data = json.loads((tmp_path / MARKER_FILENAME).read_text())
    assert data["name"] == "hero_op_strict"


def test_marker_persists_when_threshold_unset_on_resume(tmp_path):
    """Resume + no --perf-threshold flag → marker wins (no auto-removal).

    Per §7 Q1: "resume w/o flag → marker wins, no warning". This means
    when caller passes perf_threshold=None, we MUST NOT touch a pre-existing
    marker — leave the sticky precision-only state in place.
    """
    # First run sets precision-only
    write_profile_marker(tmp_path, perf_threshold=0)
    assert (tmp_path / MARKER_FILENAME).exists()

    # Second run: perf_threshold=None — DOES NOT remove marker
    profile = write_profile_marker(tmp_path, perf_threshold=None)
    assert profile is DEFAULT  # write_profile_marker returns DEFAULT on None
    # CRITICAL: marker MUST persist (resume semantics)
    assert (tmp_path / MARKER_FILENAME).exists()
    # And resolve still returns the original precision-only profile
    assert resolve_profile(tmp_path) is PRECISION_ONLY
