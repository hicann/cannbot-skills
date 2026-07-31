# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-104 residual: _is_legitimate_pipeline_exhaustion delegates to
canonical band-aware threshold resolver (schema_norm._resolve_perf_threshold).

The residual hardcoded `ratio >= 0.6` was band-unaware. A reduction-class
plugin can correctly emit 0.40× within its declared band ≥0.30; pre-fix the
legacy guard misclassified it as "below floor →
legitimate exhaustion → PARTIAL_PERSIST" because 0.40 < 0.6. Post-fix:
0.40 above band 0.30 → "perf satisfactory, gate firing is suspicious" →
False → orchestrator does NOT mark PARTIAL_PERSIST inappropriately.

The fix uses `schema_norm._resolve_perf_threshold` — the existing canonical
resolver (PerfGateProfile → plugin.
ko_escalation_threshold → 0.6 fallback) — avoiding divergent threshold
paths. This PR ships exactly that substitution.

Gate semantics (re-read from orchestrator.py:2242-2293):
- True  = "yes, iter_cap exhaustion is legitimate → finalize PARTIAL_PERSIST
          with explicit scope evidence (probe + researcher full cycle ran
          + perf still below threshold)"
- False = "exhaustion is NOT legitimate (perf at/above threshold, or pipeline
          incomplete) → caller takes another path"

So:
- REDUCTION 0.40 with band 0.30 → above band → False (post-fix; was True
  pre-fix when 0.6 hardcoded)
- REDUCTION 0.20 with band 0.30 → below band → True (eligible for PARTIAL)
- ELEMENTWISE 0.50 with default 0.6 → below default → True
- ELEMENTWISE 0.70 with default 0.6 → above default → False (unchanged)

Resolver's internal threshold logic is independently tested in
`test_schema_norm_perf_gate.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent / "plugins"))

from orchestrator import _is_legitimate_pipeline_exhaustion  # noqa: E402


def _seed_workspace(
    tmp_path: Path,
    *,
    ratio: float,
    op_class_tags: list[str] | None = None,
    add_probe_requirement: bool = True,
    precision_status: str = "PASS",
) -> None:
    (tmp_path / "verification.json").write_text(json.dumps({
        "performance": {"ratio": ratio},
        "precision": {"status": precision_status},
    }))
    if op_class_tags is not None:
        (tmp_path / "op_classification.json").write_text(json.dumps({
            "op": "test", "op_class_tags": op_class_tags,
        }))
    if add_probe_requirement:
        (tmp_path / "probe_result.json").write_text(json.dumps({
            "classification": "requirement",
        }))


@pytest.fixture
def stub_researcher(tmp_path):
    """Seed the gate's first-precondition file. `_has_researcher_output` is
    nested in `_is_legitimate_pipeline_exhaustion` (can't monkeypatch); the
    file IS the structurally-correct stub.
    """
    (tmp_path / "cann_strategy_inference.md").write_text(
        "stub researcher output for DEBT-104 band-aware threshold test"
    )


@pytest.fixture
def stub_band_plugin(monkeypatch):
    """Install plugin whose `ko_escalation_threshold(op_class)` returns 0.30
    for REDUCTION/SCAN substrings, 0.6 otherwise. Patches
    `plugins.detect_plugin` which the canonical
    resolver consumes via `from plugins import detect_plugin`.
    """
    from plugins.base import BasePlugin

    class _BandPlugin(BasePlugin):
        name = "test_band"

        def detect(self, workspace):
            return True

        def ko_escalation_threshold(self, op_class: str = "unknown") -> float:
            up = op_class.upper()
            if "REDUCTION" in up or "SCAN" in up:
                return 0.30
            return 0.6

    import plugins  # noqa: E402
    monkeypatch.setattr(plugins, "detect_plugin", lambda ws: _BandPlugin())


# ---------------------------------------------------------------------------
# Behavior change introduced by DEBT-104 fold
# ---------------------------------------------------------------------------
# Pre-fix: REDUCTION 0.40 with hardcoded 0.6 → 0.40 < 0.6 → continue checks
#          → likely True (legit exhaustion → PARTIAL_PERSIST)
# Post-fix: REDUCTION 0.40 above band 0.30 → return False (perf satisfactory,
#           iter_cap exhaustion is not a structural ceiling — caller takes a
#           different path that doesn't fabricate PARTIAL_PERSIST evidence)
# ---------------------------------------------------------------------------

def test_reduction_above_band_not_legitimate_exhaustion(
    tmp_path, stub_researcher, stub_band_plugin
):
    """ratio 0.40 above band 0.30 → perf is satisfactory → False (not
    legitimate exhaustion). Pre-fix would have returned True (wrongly
    marked PARTIAL_PERSIST because 0.40 < hardcoded 0.6).
    """
    _seed_workspace(tmp_path, ratio=0.40, op_class_tags=["REDUCTION"])
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is False


def test_reduction_below_band_legitimate_exhaustion(
    tmp_path, stub_researcher, stub_band_plugin
):
    """ratio 0.20 below band 0.30 → perf truly below structural ceiling
    after full cycle → True (legit exhaustion, eligible PARTIAL_PERSIST).
    """
    _seed_workspace(tmp_path, ratio=0.20, op_class_tags=["REDUCTION"])
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is True


def test_scan_class_band_aware(tmp_path, stub_researcher, stub_band_plugin):
    """SCAN class shares 0.30 band — 0.35 above band → not legitimate (perf
    satisfactory). Mirrors REDUCTION semantics.
    """
    _seed_workspace(tmp_path, ratio=0.35, op_class_tags=["SCAN"])
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is False


def test_non_band_op_below_default_threshold_eligible(
    tmp_path, stub_researcher, stub_band_plugin
):
    """ELEMENTWISE (no band override) → falls through to 0.6 default.
    ratio 0.50 < 0.6 → True (legitimate exhaustion, same as legacy).
    """
    _seed_workspace(tmp_path, ratio=0.50, op_class_tags=["ELEMENTWISE"])
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is True


def test_non_band_op_above_default_threshold_not_eligible(
    tmp_path, stub_researcher, stub_band_plugin
):
    """ELEMENTWISE 0.70 > default 0.6 → False (perf satisfactory,
    legacy-compatible behavior preserved).
    """
    _seed_workspace(tmp_path, ratio=0.70, op_class_tags=["ELEMENTWISE"])
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is False


def test_missing_op_classification_uses_default(
    tmp_path, stub_researcher, stub_band_plugin
):
    """No op_classification.json → resolver returns plugin's 'unknown'
    default 0.6 → 0.50 < 0.6 → True (legitimate exhaustion under default).
    """
    _seed_workspace(tmp_path, ratio=0.50, op_class_tags=None)
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is True


def test_missing_ratio_returns_false(
    tmp_path, stub_researcher, stub_band_plugin
):
    """No performance.ratio in verification.json → gate cannot decide,
    returns False (not legitimate exhaustion). Defensive default.
    """
    (tmp_path / "verification.json").write_text(json.dumps({
        "performance": {},
        "precision": {"status": "PASS"},
    }))
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is False


def test_no_plugin_layer_falls_back_to_0_6(
    tmp_path, stub_researcher, monkeypatch
):
    """Plugin layer unavailable → resolver falls back to 0.6 (legacy
    AscendC default). Defense — don't crash exhaustion gate on plugin
    bootstrap issues.
    """
    _seed_workspace(tmp_path, ratio=0.50, op_class_tags=["REDUCTION"])
    import plugins  # noqa: E402

    monkeypatch.setattr(plugins, "detect_plugin", lambda ws: None)
    # 0.50 < 0.6 fallback → True (legitimate exhaustion under default)
    assert _is_legitimate_pipeline_exhaustion(
        tmp_path, "await_optimizer"
    ) is True
