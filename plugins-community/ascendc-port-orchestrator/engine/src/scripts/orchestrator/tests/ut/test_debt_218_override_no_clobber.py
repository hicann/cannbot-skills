# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-218 — `AOG_PERF_CAPTURE_OVERRIDE_WORKER=1` must not destroy the archive.

The override really did force a real `measure_op_perf` run — but
`fsm_phase_finalize.py` then did `vj["performance"] = perf_result`, a WHOLESALE
REPLACEMENT. `perf_result` carries no `independent_re_measure` and no
`ratio_baseline`, so both were dropped. Call order is `_o5_post_verify` →
`_run_perf_capture` → `_run_finalize_prep`/eligibility, so the very next gate
read `perf["independent_re_measure"]` → None → finalize ROLLBACK.

The one documented way to force an honest independent perf capture destroyed
the archive it was meant to strengthen — anyone reaching for the honest path
got punished for it.

These tests drive `_run_perf_capture` end-to-end (real verification.json on
disk, real gate afterwards) and assert the override now yields BOTH:
  1. a source-stamped `independent_re_measure` (DEBT-217), and
  2. a CLEAN gate verdict — not a rollback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import perf_irm_provenance  # noqa: E402


MEASURED = {
    "method": "phase_o5_perf_capture: profiler schedule(warmup=5, active=5)",
    "status": "PASS",
    "ratio": 2.4,
    "reference_us": 240.0,
    "candidate_us": 100.0,
}

UNMEASURABLE = {
    "method": "phase_o5_perf_capture: profiler schedule(warmup=5, active=5)",
    "status": "N/A",
    "reason": "harness exit 1: ModuleNotFoundError: No module named 'torch_npu'",
}


def _workspace(tmp_path: Path, perf: dict) -> Path:
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# Post-worker self-critic\n\nVerdict: PASS\n\nFindings: reviewed.\n"
    )
    (ws / ".delegation_scan_passed").write_text("ok")
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_b": {"status": "PASS"}},
        "performance": perf,
    }, indent=2))
    return ws


def _worker_perf() -> dict:
    """What a benchmark worker ships today: its own ratio + an irm it authored
    about its own kernel (exactly what kw_brief.py asked for)."""
    return {
        "status": "PASS",
        "ratio": 2.7,
        "ratio_baseline": "eager torch_npu on A5",
        "independent_re_measure": {"ran": True, "ratio": 2.7},
    }


def _run_capture(monkeypatch, ws: Path, perf_result: dict, *, override: bool):
    import fsm_phase_finalize as fpf
    import finalize_pipeline

    monkeypatch.setattr(
        finalize_pipeline, "_get_active_plugin",
        lambda _ws: SimpleNamespace(
            name="benchmark",
            should_run_phase_o5_perf_capture=lambda: True,
        ),
    )
    monkeypatch.setattr(
        fpf.phase_o5_perf_capture, "measure_op_perf",
        lambda *a, **k: dict(perf_result),
    )
    monkeypatch.setattr(fpf.events, "emit", lambda *a, **k: None)
    if override:
        monkeypatch.setenv("AOG_PERF_CAPTURE_OVERRIDE_WORKER", "1")
    else:
        monkeypatch.delenv("AOG_PERF_CAPTURE_OVERRIDE_WORKER", raising=False)

    getattr(fpf, '_run_perf_capture')(SimpleNamespace(op="myop", workspace=ws, lane=0))
    return json.loads((ws / "verification.json").read_text())


def _gate(ws: Path, vj: dict):
    import finalize_pipeline as fp
    return getattr(fp, '_check_post_worker_audit')(ws, vj)


# --------------------------------------------------------------------------
# DEBT-218: the override path, end to end
# --------------------------------------------------------------------------

def test_override_preserves_irm_and_stamps_it(monkeypatch, tmp_path):
    """The DEBT-218 regression: irm must SURVIVE the capture (it used to be
    dropped by the wholesale replacement) and carry an orchestrator stamp.
    """
    ws = _workspace(tmp_path, _worker_perf())
    vj = _run_capture(monkeypatch, ws, MEASURED, override=True)
    perf = vj["performance"]

    irm = perf.get("independent_re_measure")
    assert irm is not None, "DEBT-218: capture DROPPED independent_re_measure"
    assert irm["ran"] is True
    assert irm["ratio"] == 2.4
    assert perf_irm_provenance.is_orchestrator_measured(irm)
    # delta vs the worker's 2.7 self-report is what makes the two comparable
    assert irm["delta_vs_kw_self_report"] == pytest.approx(-0.3)


def test_override_preserves_ratio_baseline(monkeypatch, tmp_path):
    """`ratio_baseline` is the key the finalize gate reads for its Path-A
    carve-out — the wholesale replacement dropped it too.
    """
    ws = _workspace(tmp_path, _worker_perf())
    vj = _run_capture(monkeypatch, ws, MEASURED, override=True)
    assert vj["performance"]["ratio_baseline"] == "eager torch_npu on A5"


def test_override_promotes_orchestrator_ratio_and_keeps_worker_aux(
    monkeypatch, tmp_path
):
    ws = _workspace(tmp_path, _worker_perf())
    vj = _run_capture(monkeypatch, ws, MEASURED, override=True)
    perf = vj["performance"]
    assert perf["ratio"] == 2.4                       # orchestrator primary
    assert perf["worker_authored_aux"]["ratio"] == 2.7  # worker preserved


def test_override_finalizes_clean_not_rollback(monkeypatch, tmp_path):
    """The headline DEBT-218 proof: the honest path must not be punished.
    Override → real capture → gate must return None (clean), not a rollback
    reason.
    """
    ws = _workspace(tmp_path, _worker_perf())
    vj = _run_capture(monkeypatch, ws, MEASURED, override=True)
    err = _gate(ws, vj)
    assert err is None, f"override path still rolls back the archive: {err}"


def test_override_when_unmeasurable_is_honest_and_still_clean(
    monkeypatch, tmp_path
):
    """On the SSH fleet the harness cannot import torch_npu, so measure_op_perf
    honestly returns N/A. That must produce ran=false + reason (never a
    fabricated ratio) and STILL finalize clean.
    """
    ws = _workspace(tmp_path, _worker_perf())
    vj = _run_capture(monkeypatch, ws, UNMEASURABLE, override=True)
    irm = vj["performance"]["independent_re_measure"]
    assert irm["ran"] is False
    assert irm["status"] == "N/A"
    assert "torch_npu" in irm["reason"]
    assert "ratio" not in irm
    assert _gate(ws, vj) is None


# --------------------------------------------------------------------------
# DEBT-217: the default (no-override) benchmark path
# --------------------------------------------------------------------------

def test_default_path_relabels_worker_authored_irm(monkeypatch, tmp_path):
    """No override + worker shipped a ratio → capture is skipped, so the irm
    present is the worker's own. It must be relabelled truthfully rather than
    left asserting an independence that never happened.
    """
    ws = _workspace(tmp_path, _worker_perf())
    vj = _run_capture(monkeypatch, ws, MEASURED, override=False)
    irm = vj["performance"]["independent_re_measure"]

    assert irm["ran"] is False, "worker self-report still claims ran=true"
    assert irm["source"] == perf_irm_provenance.SOURCE_WORKER_AUTHORED
    assert "SELF-REPORT" in irm["reason"]
    # DATA preserved — only the CLAIM was downgraded.
    assert irm["self_reported_ratio"] == 2.7
    assert vj["performance"]["ratio"] == 2.7
    # and the archive still finalizes clean — no retro-blame.
    assert _gate(ws, vj) is None


def test_default_path_leaves_orchestrator_stamp_untouched(monkeypatch, tmp_path):
    """backward / port_a3 A3-golden get a genuine orchestrator irm from
    phase_o5._write_perf_independent_re_measure. The relabel must NOT clobber
    it.
    """
    stamped = {
        "ran": True,
        "ratio": 1.9,
        "source": ("phase_o5 post_verify: orchestrator re-ran verify_<op>.py "
                   "on NPU and parsed its perf block"),
    }
    perf = _worker_perf()
    perf["independent_re_measure"] = stamped
    ws = _workspace(tmp_path, perf)
    vj = _run_capture(monkeypatch, ws, MEASURED, override=False)
    assert vj["performance"]["independent_re_measure"] == stamped
    assert _gate(ws, vj) is None


def test_capture_disabled_plugin_still_gets_truthful_stamp(monkeypatch, tmp_path):
    """A scoped plugin may set should_run_phase_o5_perf_capture()=False. The
    stamp must still be applied — it is about truth-in-labelling, not about the
    capture — otherwise a capture-disabled mode carrying a worker-authored
    ran=true would hit the new gate and roll back.
    """
    import fsm_phase_finalize as fpf
    import finalize_pipeline

    ws = _workspace(tmp_path, _worker_perf())
    monkeypatch.setattr(
        finalize_pipeline, "_get_active_plugin",
        lambda _ws: SimpleNamespace(
            name="custom_mode",
            should_run_phase_o5_perf_capture=lambda: False,
        ),
    )
    monkeypatch.setattr(fpf.events, "emit", lambda *a, **k: None)
    getattr(fpf, '_run_perf_capture')(SimpleNamespace(op="myop", workspace=ws, lane=0))

    vj = json.loads((ws / "verification.json").read_text())
    irm = vj["performance"]["independent_re_measure"]
    assert irm["ran"] is False
    assert "custom_mode" in irm["reason"]
    assert _gate(ws, vj) is None


def test_no_verification_json_is_a_noop(monkeypatch, tmp_path):
    import fsm_phase_finalize as fpf
    import finalize_pipeline

    ws = tmp_path / "op"
    ws.mkdir()
    monkeypatch.setattr(
        finalize_pipeline, "_get_active_plugin",
        lambda _ws: SimpleNamespace(
            name="benchmark",
            should_run_phase_o5_perf_capture=lambda: True,
        ),
    )
    getattr(fpf, '_run_perf_capture')(SimpleNamespace(op="myop", workspace=ws, lane=0))
    assert not (ws / "verification.json").exists()
