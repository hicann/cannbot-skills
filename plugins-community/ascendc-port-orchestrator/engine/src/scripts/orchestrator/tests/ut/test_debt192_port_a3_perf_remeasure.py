# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-192 (independent review 2026-07-04): port_a3 A3-golden orchestrator-side
independent perf re-measure (worker-perf-report contract).

Incident: port_a3 splits verification into precision (pass_a/pass_b_runner.py)
and perf (perf_runner.py, standalone). ssh_runner re-measured only precision,
so measured.perf stayed None → phase_o5 never wrote
performance.independent_re_measure → the worker's honest
{ran:false, status:PENDING_ORCHESTRATOR} irm sat unfulfilled → the finalize
irm gate (_check_post_worker_audit) rejected the archive forever (feeding the
finalize death-loop that main's DEBT-192 engine-half fail-fasts).

This pins the POSITIVE fill (orchestrator runs perf_runner.py in its own
context → irm ran:true) and the fail-loud N/A (structurally unmeasurable →
irm {ran:false, status:N/A, reason}, NEVER a bare self-report PASS).

Pure CPU, no hardware — _run_verifier is monkeypatched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import phase_o5  # noqa: E402
import phase_o5_runner as por  # noqa: E402


def _mk_port_a3_ws(tmp_path: Path, with_perf_runner: bool = True) -> Path:
    ws = tmp_path / "op"
    ws.mkdir()
    # _port_a3_claims_pass_a keys off verification.json precision.pass_a claim.
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_a": {"status": "PASS",
                                                   "tier1_pass": 43, "total": 43}},
        "performance": {"ratio": 1.48, "status": "PASS"},
        "truth_source": "a3_capture",
    }))
    if with_perf_runner:
        # A perf_runner that does NOT read verification.json (anti-cycle clean).
        (ws / "perf_runner.py").write_text(
            "import json\nprint(json.dumps({'ratio':1.42,'ratio_min':0.5,'ratio_max':1.9}))\n"
        )
    return ws


# ---- helper: scope + routing -------------------------------------------------

def test_non_port_a3_returns_none(tmp_path, monkeypatch):
    """Not a port_a3 op → helper is a no-op (leave perf to the mode's own path)."""
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda ws: False)
    assert getattr(por, '_maybe_port_a3_perf_remeasure')(tmp_path, "op", {}, lane=0) is None


def test_missing_perf_runner_is_na_not_pass(tmp_path, monkeypatch):
    """port_a3 op with NO perf_runner.py → fail-loud N/A signal, never a PASS."""
    ws = _mk_port_a3_ws(tmp_path, with_perf_runner=False)
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda w: True)
    out = getattr(por, '_maybe_port_a3_perf_remeasure')(ws, "op", {}, lane=0)
    assert out is not None and out.get("_remeasure_na") is True
    assert out["status"] == "N/A" and "no perf_runner.py" in out["reason"]


def test_successful_remeasure_returns_perf_dict(tmp_path, monkeypatch):
    """perf_runner.py runs + parses → the canonical perf dict flows back
    (phase_o5 will turn it into irm ran:true).
    """
    ws = _mk_port_a3_ws(tmp_path, with_perf_runner=True)
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda w: True)
    monkeypatch.setattr(por, "_verify_runner_independence", lambda w, s: None)
    monkeypatch.setattr(
        por, "_run_verifier",
        lambda w, op, env, script, label, lane=0, raw=False: {"ratio": 1.42, "ratio_min": 0.5},
    )
    out = getattr(por, '_maybe_port_a3_perf_remeasure')(ws, "op", {}, lane=0)
    assert out == {"ratio": 1.42, "ratio_min": 0.5}
    assert not out.get("_remeasure_na")


def test_perf_uses_raw_run_verifier(tmp_path, monkeypatch):
    """Regression (FA device e2e 2026-07-04): the perf re-measure MUST call
    _run_verifier with raw=True. The default normalization folds output into
    {tier1_pass,total,status} and STRIPS the ratio, so perf_runner.py's real
    ratio (1.2118 on device) came back as {'status':'PASS'} → false-N/A a
    measurable op. Pin that raw=True is passed.
    """
    ws = _mk_port_a3_ws(tmp_path, with_perf_runner=True)
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda w: True)
    monkeypatch.setattr(por, "_verify_runner_independence", lambda w, s: None)
    seen = {}

    def _capture(w, op, env, script, label, lane=0, raw=False):
        seen["raw"] = raw
        return {"ratio": 1.2118, "ratio_min": 0.79, "ratio_max": 1.66}

    monkeypatch.setattr(por, "_run_verifier", _capture)
    out = getattr(por, '_maybe_port_a3_perf_remeasure')(ws, "op", {}, lane=0)
    assert seen.get("raw") is True, "perf re-measure must pass raw=True to _run_verifier"
    assert out["ratio"] == 1.2118 and not out.get("_remeasure_na")


def test_normalized_dict_without_ratio_is_na(tmp_path, monkeypatch):
    """The precise false-N/A bug shape: if _run_verifier returns a dict with a
    status but NO ratio (the normalized precision shape), my guard must still
    N/A it (never accept a bare {'status':'PASS'} as a measured perf).
    """
    ws = _mk_port_a3_ws(tmp_path, with_perf_runner=True)
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda w: True)
    monkeypatch.setattr(por, "_verify_runner_independence", lambda w, s: None)
    monkeypatch.setattr(
        por, "_run_verifier",
        lambda w, op, env, script, label, lane=0, raw=False: {"status": "PASS"},
    )
    out = getattr(por, '_maybe_port_a3_perf_remeasure')(ws, "op", {}, lane=0)
    assert out.get("_remeasure_na") is True and out["status"] == "N/A"


def test_runner_error_string_is_na_not_pass(tmp_path, monkeypatch):
    """perf_runner.py ran but produced no parseable ratio (error string) →
    fail-loud N/A, never a bare self-report PASS survives.
    """
    ws = _mk_port_a3_ws(tmp_path, with_perf_runner=True)
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda w: True)
    monkeypatch.setattr(por, "_verify_runner_independence", lambda w, s: None)
    monkeypatch.setattr(
        por, "_run_verifier",
        lambda w, op, env, script, label, lane=0, raw=False: "SSH exit 2: device unreachable",
    )
    out = getattr(por, '_maybe_port_a3_perf_remeasure')(ws, "op", {}, lane=0)
    assert out.get("_remeasure_na") is True and out["status"] == "N/A"


def test_anti_cycle_perf_runner_is_na(tmp_path, monkeypatch):
    """A perf_runner that reads verification.json (self-citing) → N/A."""
    ws = _mk_port_a3_ws(tmp_path, with_perf_runner=True)
    monkeypatch.setattr(por, "_port_a3_claims_pass_a", lambda w: True)
    monkeypatch.setattr(
        por, "_verify_runner_independence",
        lambda w, s: "self-citing: perf_runner.py reads verification.json",
    )
    out = getattr(por, '_maybe_port_a3_perf_remeasure')(ws, "op", {}, lane=0)
    assert out.get("_remeasure_na") is True and "self-citing" in out["reason"]


# ---- phase_o5 irm-writer: positive vs N/A ------------------------------------

def _run_irm_writer(tmp_path, measured_perf: dict) -> dict:
    """Drive phase_o5.post_verify_for_finalize's irm-writer with a stub runner
    that returns pass_a matching the claim + the given measured.perf."""
    ws = tmp_path / "op2"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "port_a3_to_a5",
    }))
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                      "pass_a": {"status": "PASS", "tier1_pass": 5, "total": 5}},
        "performance": {"ratio": 1.48, "status": "PASS"},
    }))

    def _stub_runner(w, op, lane=0):
        return phase_o5.MeasuredResult(
            pass_a={"status": "PASS", "tier1_pass": 5, "total": 5},
            perf=measured_perf,
        )

    phase_o5.post_verify_for_finalize(ws, "op2", lane=0, runner=_stub_runner)
    return json.loads((ws / "verification.json").read_text())["performance"]["independent_re_measure"]


def test_irm_writer_positive_sets_ran_true(tmp_path):
    irm = _run_irm_writer(tmp_path, {"ratio": 1.42, "ratio_min": 0.5})
    assert irm["ran"] is True and irm["ratio"] == 1.42


def test_irm_writer_na_signal_sets_ran_false_with_reason(tmp_path):
    irm = _run_irm_writer(
        tmp_path,
        {"status": "N/A", "reason": "no perf_runner.py", "_remeasure_na": True},
    )
    assert irm["ran"] is False and irm["status"] == "N/A"
    assert irm["reason"] == "no perf_runner.py"
