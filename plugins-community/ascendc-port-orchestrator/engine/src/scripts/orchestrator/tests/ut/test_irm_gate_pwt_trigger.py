# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression: the independent-re-measure (irm) gate in
`finalize_pipeline._check_post_worker_audit` must fire for
perf.status=PASS_WITHIN_TOLERANCE, not only PASS / BELOW_THRESHOLD.

Incident (back, KB-retirement backward PoC, 2026-06-17): a backward archive
shipped with `performance.status = PASS_WITHIN_TOLERANCE`, a worker-self-
reported `ratio = 0.8648`, and `independent_re_measure` MISSING — yet the
finalize perf-irm gate passed with 0 rollback. Root cause: the irm trigger
set was `("PASS", "BELOW_THRESHOLD")` and excluded PASS_WITHIN_TOLERANCE, so
a pass-level perf verdict labelled PASS_WITHIN_TOLERANCE bypassed the
CLAUDE.md hard rule "NEVER trust skill-reported performance numbers".

The fix STRENGTHENS the gate (adds PASS_WITHIN_TOLERANCE to the trigger set);
the Path-A / cpu-truth carve-out is preserved. These tests pin both the new
rejection and the carve-out, and guard the pre-existing PASS behaviour.

Pure CPU, no hardware, no mocks of the gate itself.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import finalize_pipeline as fp  # noqa: E402


_AUDIT_PASS = (
    "# Post-worker self-critic audit\n\n"
    "Reviewed kw output against C13/C18/C25/C26 watchpoints.\n\n"
    "## Verdict\n\nPASS — kernel is faithful, pybind pure, no overfitting.\n"
)


def _mk_audited_ws(tmp_path: Path, perf: dict) -> tuple[Path, dict]:
    """Build a workspace + verification.json that satisfies every
    _check_post_worker_audit precondition UP TO the perf block, so the only
    variable under test is the `performance` dict."""
    ws = tmp_path / "bw_op"
    ws.mkdir()
    # precond 1: substantive audit doc with PASS verdict
    (ws / "audit_self_critic_post_worker.md").write_text(_AUDIT_PASS, encoding="utf-8")
    # precond 2: fresh delegation-scan marker (no kernel files → newest mtime 0,
    # marker is strictly newer, so the staleness check passes trivially)
    marker = ws / ".delegation_scan_passed"
    marker.write_text("ok\n")
    # make sure the marker mtime is comfortably > 0
    now = time.time()
    import os
    os.utime(marker, (now, now))
    # precond 3: valid pass_b status; precision PASS
    vj = {
        "precision": {
            "status": "PASS",
            "pass_b": {"status": "PASS", "tier1_pass": 6, "total": 6},
        },
        "performance": perf,
    }
    return ws, vj


def test_pwt_perf_without_irm_is_rejected(tmp_path):
    """The incident shape: PASS_WITHIN_TOLERANCE + self-reported ratio, no
    independent_re_measure → gate MUST reject (this is the closed gap).
    """
    ws, vj = _mk_audited_ws(
        tmp_path,
        {"status": "PASS_WITHIN_TOLERANCE", "ratio": 0.8648},
    )
    msg = getattr(fp, '_check_post_worker_audit')(ws, vj)
    assert msg is not None, (
        "PASS_WITHIN_TOLERANCE perf without independent_re_measure must be "
        "rejected (the back 2026-06-17 gap)"
    )
    assert "independent_re_measure" in msg


def test_pwt_perf_with_irm_is_accepted(tmp_path):
    """Same PWT verdict, but with a real independent re-measure → gate passes.

    DEBT-217 (2026-07-17): the irm now needs a `source` naming the ORCHESTRATOR
    as measurer. This fixture previously omitted it — which, post-DEBT-217, is
    indistinguishable from a worker self-reporting on its own kernel, and is
    exactly the shape the gate must now reject (see
    test_debt_217_perf_irm_provenance.py). The stamp added here is what makes
    this fixture "a REAL independent re-measure" as the docstring always
    claimed; the test's intent is unchanged.
    """
    ws, vj = _mk_audited_ws(
        tmp_path,
        {
            "status": "PASS_WITHIN_TOLERANCE",
            "ratio": 0.8648,
            "independent_re_measure": {
                "ran": True,
                "ratio": 0.86,
                "delta_vs_kw_self_report": 0.0048,
                "source": (
                    "phase_o5 post_verify: orchestrator re-ran verify_<op>.py "
                    "on NPU and parsed its perf block (independent of worker "
                    "verification.json self-report)"
                ),
            },
        },
    )
    assert getattr(fp, '_check_post_worker_audit')(ws, vj) is None


def test_pwt_perf_path_a_cpu_truth_is_exempt(tmp_path):
    """Path-A / cpu-truth carve-out still applies to PWT — no irm required when
    the baseline is explicitly cpu-truth.
    """
    ws, vj = _mk_audited_ws(
        tmp_path,
        {
            "status": "PASS_WITHIN_TOLERANCE",
            "ratio": 0.8648,
            "ratio_baseline": "cpu-truth synthetic edge dataset (Path A)",
        },
    )
    assert getattr(fp, '_check_post_worker_audit')(ws, vj) is None


def test_pass_perf_still_requires_irm(tmp_path):
    """Regression guard: pre-existing PASS behaviour is unchanged — PASS
    without irm is still rejected.
    """
    ws, vj = _mk_audited_ws(
        tmp_path,
        {"status": "PASS", "ratio": 1.42},
    )
    msg = getattr(fp, '_check_post_worker_audit')(ws, vj)
    assert msg is not None and "independent_re_measure" in msg


def test_below_threshold_perf_still_requires_irm(tmp_path):
    """Regression guard: BELOW_THRESHOLD also unchanged."""
    ws, vj = _mk_audited_ws(
        tmp_path,
        {"status": "BELOW_THRESHOLD", "ratio": 0.4},
    )
    msg = getattr(fp, '_check_post_worker_audit')(ws, vj)
    assert msg is not None and "independent_re_measure" in msg
