# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase B5 — schema_norm.P0uu accepts ratio=null under PRECISION_ONLY profile.

Closes the finalize-side hole left after B4 YAML wire-up:

  B4 suppresses kw→ko routing under PRECISION_ONLY → done route fires
    BUT
  P0uu still rejected ratio=null at finalize → finalize fails

  This PR: profile.require_ratio_in_verification=False (PRECISION_ONLY) →
  accept ratio=null without explicit N/A or skip_reason justification.

Backwards-compat: DEFAULT profile keeps require_ratio_in_verification=True
so legacy P0uu behavior preserved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

from perf_gate import write_profile_marker  # noqa: E402
from schema_norm import _check_evidence_for_terminal  # noqa: E402


def _minimal_workspace(tmp_path: Path, *, ratio=None, skipped=False,
                       skip_reason=None, status=None) -> Path:
    """Create a workspace with verification.json shaped to hit the P0uu gate.

    For P0uu null-ratio testing we need:
      - precision.status PASS (else other gates short-circuit)
      - pass count consistent (else pass-count gate fires first)
      - performance.ratio set per test args (None to hit P0uu)
    """
    ws = tmp_path / "test_op__backward"
    ws.mkdir()
    perf_block: dict = {}
    if ratio is not None:
        perf_block["ratio"] = ratio
    if skipped:
        perf_block["skipped"] = True
    if skip_reason is not None:
        perf_block["skip_reason"] = skip_reason
    if status is not None:
        perf_block["status"] = status
    vj = {
        "precision": {
            "status": "PASS",
            "pass_a": {"tier1_pass": 5, "total": 5, "status": "PASS"},
        },
        "performance": perf_block,
        "determinism": {"policy": "n/a", "observed": True, "policy_satisfied": True},
    }
    (ws / "verification.json").write_text(json.dumps(vj))
    # P0aay requires knowledge_update.md with ## Findings section + ≥100 bytes
    ku = (
        "# Knowledge Update\n\n"
        "## Context\nTest op for B5 P0uu null-ratio gate.\n\n"
        "## Findings\nratio=null path traversed.\n\n"
        "## KB-promotable patterns\nnone\n\n"
        "## Cited KB items\nnone\n\n"
        "## Anti-patterns avoided\nnone\n"
    )
    (ws / "knowledge_update.md").write_text(ku)
    # P0qq requires `## Self-introspection` section in PROGRESS.md with
    # specific subsections. Scaffold minimal compliance.
    progress = (
        "# PROGRESS\n\n"
        "## Self-introspection\n\n"
        "### Pressure modes I felt\nnone material\n\n"
        "### Decisions I almost rationalized\nnone\n\n"
        "### Verifications I might have skipped\nnone\n\n"
        "### Confidence calibration\nhigh\n"
    )
    (ws / "PROGRESS.md").write_text(progress)
    return ws


def _call_p0uu(ws: Path) -> dict:
    """Invoke the P0uu evidence-for-terminal gate via the public-ish entry."""
    return _check_evidence_for_terminal(
        ws, alias="done", target="finalize", entry={}
    )


# ---- Legacy behavior preserved under DEFAULT profile ----


def test_no_marker_null_ratio_rejected(tmp_path):
    """No marker (DEFAULT profile) + ratio=null + no N/A + no skip_reason
    → P0uu rejects (legacy behavior preserved).
    """
    ws = _minimal_workspace(tmp_path, ratio=None)
    result = _call_p0uu(ws)
    assert result.get("passes") is False
    assert "performance.ratio missing" in (result.get("reason") or "")


def test_no_marker_explicit_na_accepted(tmp_path):
    """No marker + ratio=null + status=N/A → accepted (existing Path A)."""
    ws = _minimal_workspace(tmp_path, ratio=None, status="N/A")
    result = _call_p0uu(ws)
    assert result.get("passes") is True


def test_no_marker_skipped_with_reason_accepted(tmp_path):
    """No marker + ratio=null + skipped + skip_reason → accepted (existing path)."""
    ws = _minimal_workspace(
        tmp_path, ratio=None, skipped=True,
        skip_reason="reference op unrunnable on this card",
    )
    result = _call_p0uu(ws)
    assert result.get("passes") is True


# ---- New B5 behavior under PRECISION_ONLY ----


def test_precision_only_null_ratio_accepted(tmp_path):
    """PRECISION_ONLY marker + ratio=null (no N/A, no skip_reason) → ACCEPTED.

    This is the new B5 behavior. Worker doesn't have to fabricate an N/A
    or skip_reason — the profile itself encodes "we explicitly opted out
    of perf measurement".
    """
    ws = _minimal_workspace(tmp_path, ratio=None)
    write_profile_marker(ws, perf_threshold=0)
    result = _call_p0uu(ws)
    assert result.get("passes") is True, result.get("reason")
    assert "precision_only" in (result.get("reason") or "").lower()


def test_precision_only_does_not_affect_present_ratio(tmp_path):
    """PRECISION_ONLY + ratio=0.3 → should pass the P0uu gate specifically.

    B2 makes threshold=0.0 under PRECISION_ONLY so ratio=0.3 is above
    threshold → no early reject. B5 only modifies the ratio=null path,
    so this test confirms ratio-present case still flows through P0uu
    unchanged (rejection reason should NOT mention "performance.ratio missing").
    """
    ws = _minimal_workspace(tmp_path, ratio=0.3)
    write_profile_marker(ws, perf_threshold=0)
    result = _call_p0uu(ws)
    # The result may still fail on downstream gates (P0aay schema checks),
    # but it must NOT fail with the P0uu "performance.ratio missing" reason.
    reason = result.get("reason") or ""
    assert "performance.ratio missing" not in reason, (
        f"Expected P0uu to accept ratio=0.3, but got: {reason}"
    )


# ---- HERO_OP_STRICT preserves strict requirement ----


def test_hero_op_strict_null_ratio_still_rejected(tmp_path):
    """HERO_OP_STRICT has require_ratio_in_verification=True →
    null ratio still rejected (legacy P0uu behavior). User explicitly
    chose strict; the gate enforces it.
    """
    ws = _minimal_workspace(tmp_path, ratio=None)
    write_profile_marker(ws, perf_threshold=0.9)
    result = _call_p0uu(ws)
    assert result.get("passes") is False
    assert "performance.ratio missing" in (result.get("reason") or "")


# ---- Custom-threshold profile clones DEFAULT (require=True) ----


def test_custom_threshold_profile_requires_ratio(tmp_path):
    """Custom threshold (e.g. --perf-threshold=0.45) clones DEFAULT
    fields → require_ratio_in_verification=True → null ratio rejected.
    """
    ws = _minimal_workspace(tmp_path, ratio=None)
    write_profile_marker(ws, perf_threshold=0.45)
    result = _call_p0uu(ws)
    assert result.get("passes") is False


# ---- Safety: malformed marker still rejects (no crash) ----


def test_malformed_marker_falls_through_to_legacy_reject(tmp_path):
    """Malformed marker → resolve_profile returns DEFAULT
    → require_ratio_in_verification=True → P0uu still rejects null ratio
    (no crash, legacy behavior preserved).
    """
    ws = _minimal_workspace(tmp_path, ratio=None)
    (ws / ".perf_gate_profile.json").write_text("not valid JSON {")
    result = _call_p0uu(ws)
    assert result.get("passes") is False
    assert "performance.ratio missing" in (result.get("reason") or "")
