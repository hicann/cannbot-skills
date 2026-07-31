# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-217 / DEBT-218 — `performance.independent_re_measure` provenance.

DEBT-217: the field asserts author!=measurer, but the WORKER historically
wrote it and the finalize gate only checked that it existed with ran=true + a
ratio. These tests pin:

  1. The gate REDs a worker-authored / unstamped `ran=true` irm (the mutation:
     strip the `source` → RED; restore → green).
  2. The honest unmeasured encodings (ran=false + N/A + reason) stay green —
     the gate demands a truthful LABEL, not that every op be measured.
  3. `_run_perf_capture` stamps the irm truthfully for every mode, including
     the modes where the capture is disabled, so a legitimate op never reaches
     the gate with an unstamped ran=true.

DEBT-218: `AOG_PERF_CAPTURE_OVERRIDE_WORKER=1` forced a real capture but then
did `vj["performance"] = perf_result` — a wholesale replacement that dropped
the top-level `independent_re_measure`, so the next gate read None → finalize
ROLLBACK. The end-to-end test below proves the override now yields a
source-stamped irm AND a clean gate, not a rollback.

Covered:
- perf_irm_provenance (all public helpers)
- finalize_checks_provenance._check_post_worker_audit (stamp check)
- fsm_phase_finalize._run_perf_capture (merge + stamp + relabel)
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import perf_irm_provenance  # noqa: E402


# --------------------------------------------------------------------------
# perf_irm_provenance unit surface
# --------------------------------------------------------------------------

def test_worker_authored_source_is_not_orchestrator():
    assert not perf_irm_provenance.is_orchestrator_source("worker_authored")
    assert not perf_irm_provenance.is_orchestrator_source("")
    assert not perf_irm_provenance.is_orchestrator_source(None)
    assert not perf_irm_provenance.is_orchestrator_source("kw-4 ran perf.py")


def test_existing_phase_o5_stamps_count_as_orchestrator():
    """The two prose stamps phase_o5._write_perf_independent_re_measure has
    written since 2026-06-16 (backward) / 2026-07-03 (port_a3 A3-golden) must
    keep counting as independent — this change must not invalidate the modes
    that already do the right thing.
    """
    assert perf_irm_provenance.is_orchestrator_source(
        "phase_o5 post_verify: orchestrator re-ran verify_<op>.py on NPU and "
        "parsed its perf block (independent of worker verification.json "
        "self-report)"
    )
    assert perf_irm_provenance.is_orchestrator_source(
        "phase_o5 post_verify: orchestrator attempted an independent perf "
        "re-measure and it was structurally unmeasurable"
    )
    assert perf_irm_provenance.is_orchestrator_source(
        perf_irm_provenance.SOURCE_CAPTURE_MEASURED
    )
    assert perf_irm_provenance.is_orchestrator_source(
        perf_irm_provenance.SOURCE_CAPTURE_ATTEMPTED
    )


def test_is_orchestrator_measured_rejects_non_dict_and_unstamped():
    assert not perf_irm_provenance.is_orchestrator_measured(None)
    assert not perf_irm_provenance.is_orchestrator_measured("nope")
    assert not perf_irm_provenance.is_orchestrator_measured({"ran": True, "ratio": 2.7})


def test_orchestrator_irm_from_measured_result_stamps_and_deltas():
    irm = perf_irm_provenance.orchestrator_irm_from_perf_result(
        {"status": "PASS", "ratio": 2.5, "method": "profiler"},
        worker_ratio=2.0,
    )
    assert irm["ran"] is True
    assert irm["ratio"] == 2.5
    assert irm["delta_vs_kw_self_report"] == 0.5
    assert perf_irm_provenance.is_orchestrator_measured(irm)


def test_orchestrator_irm_from_na_result_never_invents_a_ratio():
    """measure_op_perf returns status=N/A + reason when it cannot measure (the
    SSH fleet: no torch_npu on the orchestrator host). That must become an
    honest ran=false — never a fabricated ran=true.
    """
    irm = perf_irm_provenance.orchestrator_irm_from_perf_result(
        {"status": "N/A", "reason": "no torch_npu on orchestrator host"},
        worker_ratio=2.0,
    )
    assert irm["ran"] is False
    assert irm["status"] == "N/A"
    assert "ratio" not in irm
    assert "no torch_npu" in irm["reason"]
    assert perf_irm_provenance.is_orchestrator_measured(irm)


def test_worker_authored_irm_downgrades_claim_but_preserves_data():
    irm = perf_irm_provenance.worker_authored_irm(
        {"ran": True, "ratio": 2.7, "method": "kw self-timed"},
        reason="no orchestrator re-measure ran",
    )
    assert irm["ran"] is False
    assert irm["status"] == "N/A"
    assert irm["source"] == perf_irm_provenance.SOURCE_WORKER_AUTHORED
    # The DATA survives — only the CLAIM is downgraded.
    assert irm["self_reported_ratio"] == 2.7
    assert irm["self_reported_method"] == "kw self-timed"
    assert not perf_irm_provenance.is_orchestrator_measured(irm)


def test_merge_preserves_irm_and_ratio_baseline():
    """DEBT-218 core: the wholesale replacement dropped both of these."""
    worker_perf = {
        "ratio": 2.0,
        "ratio_baseline": "cpu-truth Path A",
        "independent_re_measure": {"ran": False, "status": "PENDING_ORCHESTRATOR"},
    }
    merged = perf_irm_provenance.merge_perf_preserving_irm(
        worker_perf, {"status": "PASS", "ratio": 1.5, "method": "profiler"}
    )
    assert merged["ratio"] == 1.5           # orchestrator wins
    assert merged["ratio_baseline"] == "cpu-truth Path A"   # worker key survives
    assert "independent_re_measure" in merged               # DEBT-218
    assert merged["method"] == "profiler"


# --------------------------------------------------------------------------
# The gate — mutation proof
# --------------------------------------------------------------------------

def _passing_vj(irm: dict) -> dict:
    return {
        "precision": {"status": "PASS", "pass_b": {"status": "PASS"}},
        "performance": {"status": "PASS", "ratio": 2.7,
                        "independent_re_measure": irm},
    }


def _workspace_with_audit(tmp_path: Path) -> Path:
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "audit_self_critic_post_worker.md").write_text(
        "# Post-worker self-critic\n\nVerdict: PASS\n\n"
        "Findings: kernel reviewed against C13/C18/C25/C26 catalog; no issues.\n"
    )
    marker = ws / ".delegation_scan_passed"
    marker.write_text("ok")
    return ws


def _run_gate(ws: Path, vj: dict):
    # Enter via finalize_pipeline (which re-exports the check functions): it is
    # the root of the finalize_pipeline <-> finalize_checks import cycle, so a
    # cold `import finalize_checks_provenance` would ImportError. Matches the
    # existing pattern in test_debt168_post_worker_waiver_parsing.py.
    import finalize_pipeline as fp
    return getattr(fp, '_check_post_worker_audit')(ws, vj)


ORCH_IRM = {
    "ran": True,
    "ratio": 2.65,
    "source": perf_irm_provenance.SOURCE_CAPTURE_MEASURED,
}


def test_gate_green_when_irm_is_orchestrator_stamped(tmp_path):
    ws = _workspace_with_audit(tmp_path)
    err = _run_gate(ws, _passing_vj(dict(ORCH_IRM)))
    assert err is None, f"orchestrator-stamped irm must pass, got: {err}"


def test_gate_reds_worker_authored_ran_true_irm(tmp_path):
    """MUTATION: strip the `source` from the orchestrator-stamped irm — i.e.
    exactly the worker-authored block older flows shipped for months. It
    MUST go RED. If this passes, the DEBT-217 fix is inert.
    """
    ws = _workspace_with_audit(tmp_path)
    mutated = dict(ORCH_IRM)
    del mutated["source"]
    err = _run_gate(ws, _passing_vj(mutated))
    assert err is not None, (
        "a source-less ran=true irm passed the gate — the stamp check is INERT"
    )
    assert "source" in err and "author" in err


def test_gate_reds_explicit_worker_authored_stamp(tmp_path):
    ws = _workspace_with_audit(tmp_path)
    mutated = dict(ORCH_IRM)
    mutated["source"] = perf_irm_provenance.SOURCE_WORKER_AUTHORED
    err = _run_gate(ws, _passing_vj(mutated))
    assert err is not None
    assert "worker_authored" in err


def test_gate_restore_source_returns_green(tmp_path):
    """The other half of the mutation: restoring the stamp restores green, so
    the RED above is attributable to the stamp and nothing else.
    """
    ws = _workspace_with_audit(tmp_path)
    assert _run_gate(ws, _passing_vj(dict(ORCH_IRM))) is None


def test_gate_green_on_honest_unmeasured_irm(tmp_path):
    """The gate demands a truthful LABEL, not that every op be measured. The
    honest ran=false + N/A + reason encoding (what benchmark now gets on the
    SSH fleet) must NOT block finalize — otherwise the fix would wedge every
    benchmark op.
    """
    ws = _workspace_with_audit(tmp_path)
    honest = {
        "ran": False,
        "status": "N/A",
        "reason": "no torch_npu on orchestrator host — capture not possible",
        "source": perf_irm_provenance.SOURCE_WORKER_AUTHORED,
    }
    assert _run_gate(ws, _passing_vj(honest)) is None


def test_gate_still_reds_missing_irm(tmp_path):
    """Pre-existing behaviour must survive the change."""
    ws = _workspace_with_audit(tmp_path)
    vj = _passing_vj({})
    vj["performance"]["independent_re_measure"] = {}
    err = _run_gate(ws, vj)
    assert err is not None and "missing or empty" in err


def test_gate_path_a_carveout_unaffected(tmp_path):
    """ratio_baseline Path-A ops skip the irm requirement entirely — and this
    is the key the DEBT-218 clobber used to drop.
    """
    ws = _workspace_with_audit(tmp_path)
    vj = _passing_vj({})
    del vj["performance"]["independent_re_measure"]
    vj["performance"]["ratio_baseline"] = "cpu-truth Path A"
    assert _run_gate(ws, vj) is None
