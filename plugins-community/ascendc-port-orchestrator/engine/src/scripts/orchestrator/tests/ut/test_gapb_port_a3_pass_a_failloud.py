# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""gap-(b) 2026-06-16 (independent review-caught, elu): port_a3 O5 pass_a re-measure.

A port_a3 op that CLAIMS precision.pass_a but emitted NO canonical Python
pass_a verifier (e.g. a stale/divergent CPU-truth run that authored only a C++
`<op>_runner.cpp`, invisible to phase_o5_runner._find_verifier's Python search
list) used to fall through to measured.pass_a=None. phase_o5 then reported the
OPAQUE "pass_a: claimed but not measured" and rolled back WITHOUT telling the
worker what to emit → silent finalize-rollback loop.

The fix (phase_o5_runner.ssh_runner) FAILS LOUD + ACTIONABLE in that case:
RUNNER_FAILED with a message naming the missing pass_a_runner.py + the canonical
judge to import + the C++ runner found instead. It does NOT weaken the O5 gate —
the gate still refuses finalize on a fraudulent claim (negative test below).

Anti-cheat invariants asserted here:
  1. Claims-but-no-verifier → RUNNER_FAILED with actionable message (not silent None).
  2. The gate STILL rejects a fraudulent pass_a claim (claim 33/33, measure 20/33
     → MISMATCH). The fix produces a real measurement; it does not rubber-stamp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o5  # noqa: E402
import phase_o5_runner  # noqa: E402


class _FakePortA3Plugin:
    name = "port_a3_to_a5"

    def pass_b_required(self) -> bool:
        return False


def _seed_port_a3_workspace(ws: Path, *, pass_a_status="PASS", tier1_pass=33,
                            total=33, files=None):
    ws.mkdir(parents=True, exist_ok=True)
    for f in (files or []):
        (ws / f).write_text(f"// {f} stub\n")
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "pass_a": {"status": pass_a_status, "tier1_pass": tier1_pass,
                       "total": total},
            "pass_b": {"status": "N/A", "reason": "port_a3 mode: pass_a subsumes"},
        }
    }))
    (ws.parent / ".ascendc_env").write_text(
        "TARGET=a5\nA5_HOST=test-host\nA5_USER=root\nA5_PASSWORD=test\n"
        "A5_CONTAINER=test-container\nCANN_PATH=/test/cann\n"
        "BENCHMARK_ROOT=/root/AscendOpGenAgent\n"
    )


# ---------------------------------------------------------------------------
# 1. helper: distinguish a real pass_a claim from N/A
# ---------------------------------------------------------------------------
def test_port_a3_claims_pass_a_true_for_real_claim(tmp_path, monkeypatch):
    monkeypatch.setattr("plugins.detect_plugin", lambda ws: _FakePortA3Plugin())
    ws = tmp_path / "elu"
    _seed_port_a3_workspace(ws, pass_a_status="PASS", tier1_pass=33, total=33)
    assert getattr(phase_o5_runner, '_port_a3_claims_pass_a')(ws) is True


def test_port_a3_claims_pass_a_false_when_na(tmp_path, monkeypatch):
    monkeypatch.setattr("plugins.detect_plugin", lambda ws: _FakePortA3Plugin())
    ws = tmp_path / "op_na"
    ws.mkdir(parents=True)
    (ws / "verification.json").write_text(json.dumps(
        {"precision": {"pass_a": {"status": "N/A"}}}))
    (ws.parent / ".ascendc_env").write_text("TARGET=a5\nA5_HOST=h\n")
    assert getattr(phase_o5_runner, '_port_a3_claims_pass_a')(ws) is False


def test_port_a3_claims_pass_a_false_when_not_port_a3(tmp_path, monkeypatch):
    # No migration plugin → never enter the migration-only fail-loud path.
    monkeypatch.setattr("plugins.detect_plugin", lambda ws: None)
    ws = tmp_path / "1_gelu"
    _seed_port_a3_workspace(ws, pass_a_status="PASS", tier1_pass=50, total=50)
    assert getattr(phase_o5_runner, '_port_a3_claims_pass_a')(ws) is False


# ---------------------------------------------------------------------------
# 2. fail-loud: claims pass_a but only a C++ runner present → RUNNER_FAILED
# ---------------------------------------------------------------------------
def test_failloud_when_claims_pass_a_but_only_cpp_runner(tmp_path, monkeypatch):
    monkeypatch.setattr("plugins.detect_plugin", lambda ws: _FakePortA3Plugin())
    # canonical pass_a skips for port_a3 (as in production) → fall to verifier search
    monkeypatch.setattr(phase_o5_runner, "_run_canonical_pass_a",
                        lambda ws, op, env, lane=0: "canonical skipped: port_a3 mode")
    # bypass the real scp workspace-resync (no NPU host in unit env)
    monkeypatch.setattr(phase_o5_runner, "_resync_workspace_to_container",
                        lambda ws, env, lane=0: None)
    ws = tmp_path / "elu"
    # elu's stale CPU-truth run authored ONLY a C++ runner + build script —
    # NO pass_a_runner.py / edge_verify.py / verify_edge.py.
    _seed_port_a3_workspace(ws, files=["elu_runner.cpp", "build_runner.sh"])

    result = phase_o5_runner.ssh_runner(ws, "elu", lane=0)

    assert result.runner_error is not None, "must fail loud, not silently None"
    msg = result.runner_error
    assert "claims precision.pass_a" in msg
    assert "pass_a_runner.py" in msg          # names the fix
    assert "precision_eval_port_a3_two_tier" in msg  # names the canonical judge
    assert "elu_runner.cpp" in msg            # names what was found instead
    # And it must NOT have produced a (None) measured pass_a that the gate would
    # report opaquely — runner_error short-circuits before any pass_a is set.
    assert result.pass_a is None


def test_na_pass_a_does_not_failloud(tmp_path, monkeypatch):
    # A genuine N/A pass_a (nothing claimed) must NOT trip the fail-loud path —
    # it returns normally with pass_a None (nothing to cross-check).
    monkeypatch.setattr("plugins.detect_plugin", lambda ws: _FakePortA3Plugin())
    monkeypatch.setattr(phase_o5_runner, "_run_canonical_pass_a",
                        lambda ws, op, env, lane=0: "canonical skipped: port_a3 mode")
    monkeypatch.setattr(phase_o5_runner, "_resync_workspace_to_container",
                        lambda ws, env, lane=0: None)
    ws = tmp_path / "op_na"
    ws.mkdir(parents=True)
    (ws / "verification.json").write_text(json.dumps(
        {"precision": {"pass_a": {"status": "N/A"},
                       "pass_b": {"status": "N/A", "reason": "subsumed"}}}))
    (ws.parent / ".ascendc_env").write_text(
        "TARGET=a5\nA5_HOST=h\nA5_USER=root\nA5_PASSWORD=t\n"
        "A5_CONTAINER=c\nCANN_PATH=/c\nBENCHMARK_ROOT=/r\n")
    result = phase_o5_runner.ssh_runner(ws, "op_na", lane=0)
    # No actionable fail-loud about pass_a (claim is N/A). pass_a stays None.
    if result.runner_error is not None:
        assert "claims precision.pass_a" not in result.runner_error
    assert result.pass_a is None


# ---------------------------------------------------------------------------
# 3. NEGATIVE anti-cheat: the gate STILL rejects a fraudulent claim
# ---------------------------------------------------------------------------
def test_gate_still_rejects_fraudulent_pass_a_claim(tmp_path):
    """The fix makes a real measurement happen — it must NOT rubber-stamp the
    worker's claim. Claim says 33/33 PASS; an independent re-measure returns
    20/33 → phase_o5 MUST report MISMATCH and refuse finalize.
    """
    ws = tmp_path / "fraud_op"
    ws.mkdir(parents=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "port_a3_to_a5",
    }))
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "pass_a": {"status": "PASS", "tier1_pass": 33, "total": 33},
            "pass_b": {"status": "N/A", "reason": "subsumed"},
        }
    }))

    def fraud_runner(workspace, op, lane=0):
        # independent re-measure disagrees with the claim
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 20, "total": 33})

    rep = phase_o5.post_verify_for_finalize(ws, "fraud_op", runner=fraud_runner)
    assert rep.verdict == "MISMATCH", rep.summary
    assert any("pass_a" in m for m in rep.mismatches)


def test_gate_verifies_matching_pass_a_claim(tmp_path):
    """Sanity: when the independent re-measure MATCHES the claim, VERIFIED."""
    ws = tmp_path / "honest_op"
    ws.mkdir(parents=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "opgen_mode": "port_a3_to_a5",
    }))
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "pass_a": {"status": "PASS", "tier1_pass": 33, "total": 33},
            "pass_b": {"status": "N/A", "reason": "subsumed"},
        }
    }))

    def honest_runner(workspace, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 33, "total": 33})

    rep = phase_o5.post_verify_for_finalize(ws, "honest_op", runner=honest_runner)
    assert rep.verdict == "VERIFIED", rep.summary
