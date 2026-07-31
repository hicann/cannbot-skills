# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""O5-BESTEFFORT-DET-COUNT-TOLERANCE (2026-07-21): best-effort determinism
count tolerance on the Phase O5 tier1_pass reconciliation.

Bug: a best-effort NONDETERMINISTIC op whose tier1_pass count varies run-to-run
(e.g. 13-16 of 16) stochastically TRIPS phase_o5's exact-equality count
reconciliation → O5 MISMATCH → the FSM rolls back to await_worker and respawns,
burning the whole worker budget then hard-failing — even though the kernel never
regressed.

Fix (scoped, MUST NOT weaken the verifier): for `tier1_pass` ONLY, when
det_policy=="best_effort" AND an author-declared immutable `det_floor` exists,
MISMATCH only when the MEASURED count drops BELOW the floor; tolerate a measured
count at/above the floor. Everything else (the `total` field, deterministic
ops, best_effort ops with no declared floor) keeps exact equality.

Anti-launder invariant: the tolerance ONLY stops the stochastic
MISMATCH→respawn churn. It does NOT mark the op deterministic/clean and NEVER
touches the determinism sub-block (`determinism.policy_satisfied`) — that stays
a SEPARATE required gate which still honestly reports policy_satisfied=false.

Sibling of test_p0kk_phase_o5_post_verify.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o5  # noqa: E402
import phase_o15  # noqa: E402


def _seed(ws: Path, *, prec: dict, determinism: dict | None = None):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "op",
        "opgen_mode": "backward",
    }))
    doc = {"precision": prec, "performance": {"status": "PASS", "ratio": 1.5}}
    if determinism is not None:
        doc["determinism"] = determinism
    (ws / "verification.json").write_text(json.dumps(doc))


def _claim(tier1_pass: int, total: int) -> dict:
    return {"status": "PASS", "tier1_pass": tier1_pass, "total": total}


# --------------------------------------------------------------------------
# Phase O5 count-reconciliation tolerance (the four discriminating cases)
# --------------------------------------------------------------------------

def test_besteffort_below_floor_still_mismatch(tmp_path):
    """best_effort + det_floor=14, measured tier1_pass=13 (<floor) → MISMATCH.

    The floor still catches a real drop — a precision regression that pushes
    the measured count below the declared floor is NOT tolerated.
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "best_effort", "det_floor": 14,
                       "observed_deterministic": False, "policy_satisfied": False})

    def runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 13, "total": 16})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    assert rep.verdict == "MISMATCH"
    assert any("BELOW declared det_floor" in m and "13" in m for m in rep.mismatches)


def test_besteffort_at_or_above_floor_tolerated_and_not_laundered(tmp_path):
    """best_effort + det_floor=14, measured=15 (>=floor), claimed=16 →
    NO count MISMATCH (tolerated), verdict stable (NOT MISMATCH → no respawn),
    AND the determinism sub-block is NOT forced true (nondeterminism recorded).
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "best_effort", "det_floor": 14,
                       "observed_deterministic": False, "policy_satisfied": False})

    def runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 15, "total": 16})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    # Stable, non-respawn verdict: the count discrepancy did NOT become MISMATCH.
    assert rep.verdict != "MISMATCH"
    assert rep.mismatches == []
    # The tolerated discrepancy is recorded honestly (not hidden).
    assert any("TOLERATED" in t and "15" in t for t in rep.det_tolerated)
    assert "PARTIAL_PERSIST" in rep.summary

    # Anti-launder: phase_o5 must NOT have written/forced the determinism
    # sub-block true — policy_satisfied stays false, so the SEPARATE
    # determinism gate still reports the op nondeterministic.
    det = json.loads((tmp_path / "verification.json").read_text())["determinism"]
    assert det["policy_satisfied"] is False
    assert det["observed_deterministic"] is False


def test_required_policy_keeps_exact_equality(tmp_path):
    """det_policy='required', measured=15 claimed=16 → MISMATCH.

    Deterministic ops are unchanged: exact equality preserved even though a
    det_floor happens to be declared (a required op must be bit-identical).
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "required", "det_floor": 14,
                       "observed_deterministic": True, "policy_satisfied": True})

    def runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 15, "total": 16})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    assert rep.verdict == "MISMATCH"
    assert rep.det_tolerated == []
    assert any("claimed=16 measured=15" in m for m in rep.mismatches)


def test_besteffort_without_floor_falls_back_to_exact(tmp_path):
    """det_policy='best_effort' but NO det_floor → exact equality (no silent
    tolerance). A best_effort op that never declared a floor is treated exactly
    as today — measured != claimed still MISMATCHes.
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "best_effort",
                       "observed_deterministic": False, "policy_satisfied": False})

    def runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 15, "total": 16})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    assert rep.verdict == "MISMATCH"
    assert rep.det_tolerated == []


def test_besteffort_exact_match_is_plain_verified(tmp_path):
    """best_effort + det_floor but measured EXACTLY equals claim → plain
    VERIFIED, no tolerated entry (tolerance only fires when counts differ).
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "best_effort", "det_floor": 14,
                       "observed_deterministic": True, "policy_satisfied": True})

    def runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 16, "total": 16})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    assert rep.verdict == "VERIFIED"
    assert rep.det_tolerated == []


def test_total_field_never_tolerated(tmp_path):
    """Tolerance is tier1_pass-ONLY: a `total` mismatch still MISMATCHes even
    under best_effort + det_floor (only the pass count varies stochastically;
    a changed total is a real discrepancy).
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "best_effort", "det_floor": 14,
                       "observed_deterministic": False, "policy_satisfied": False})

    def runner(ws, op, lane=0):
        # pass count above floor (tolerable) but total differs (must catch).
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 15, "total": 20})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    assert rep.verdict == "MISMATCH"
    assert any("total" in m and "16" in m and "20" in m for m in rep.mismatches)


def test_floor_read_from_durable_state_fallback(tmp_path):
    """det_floor absent from verification.json determinism block but present in
    the durable .opgen_state.json (written at O1.5) → still applied.
    """
    _seed(tmp_path, prec={"status": "PASS", "pass_a": _claim(16, 16)},
          determinism={"policy": "best_effort",
                       "observed_deterministic": False, "policy_satisfied": False})
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({
            "op": "op",
            "opgen_mode": "backward",
            "det_policy": "best_effort",
            "det_floor": 14,
        }))

    def runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 15, "total": 16})

    rep = phase_o5.post_verify_for_finalize(tmp_path, "op", runner=runner)
    assert rep.verdict != "MISMATCH"
    assert any("TOLERATED" in t for t in rep.det_tolerated)


# --------------------------------------------------------------------------
# Phase O1.5 det_floor declaration (immutable INPUT, never derived)
# --------------------------------------------------------------------------

def test_o15_parses_det_floor_from_analysis_md(tmp_path):
    (tmp_path / "analysis.md").write_text(
        "DET_POLICY: best_effort\nDET_FLOOR: 14\n")
    rep = phase_o15.classify_det_policy(tmp_path, "op")
    assert rep.policy == "best_effort"
    assert rep.det_floor == 14


def test_o15_no_det_floor_is_none(tmp_path):
    (tmp_path / "analysis.md").write_text("DET_POLICY: best_effort\n")
    rep = phase_o15.classify_det_policy(tmp_path, "op")
    assert rep.det_floor is None


def test_o15_explicit_floor_wins(tmp_path):
    (tmp_path / "analysis.md").write_text("DET_FLOOR: 14\n")
    rep = phase_o15.classify_det_policy(tmp_path, "op", explicit="best_effort",
                                        explicit_floor=12)
    assert rep.det_floor == 12


def test_o15_store_in_durable_state_writes_floor(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(json.dumps({"op": "op"}))
    phase_o15.store_in_durable_state(tmp_path, "best_effort", det_floor=14)
    state = json.loads((tmp_path / ".opgen_state.json").read_text())
    assert state["det_policy"] == "best_effort"
    assert state["det_floor"] == 14


def test_o15_store_omits_floor_when_none(tmp_path):
    """Undeclared floor leaves the field ABSENT (not a fabricated 0)."""
    (tmp_path / ".opgen_state.json").write_text(json.dumps({"op": "op"}))
    phase_o15.store_in_durable_state(tmp_path, "best_effort", det_floor=None)
    state = json.loads((tmp_path / ".opgen_state.json").read_text())
    assert "det_floor" not in state
