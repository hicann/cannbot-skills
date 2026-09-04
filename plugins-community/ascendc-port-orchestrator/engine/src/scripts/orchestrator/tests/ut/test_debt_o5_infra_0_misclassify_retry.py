# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-O5-INFRA-0-MISCLASSIFY regression (2026-07-20, owner=main).

Two-part harness fix so an INFRA-class O5 RUNNER_FAILED (e.g. a stale/missing
NPU_PYTHON_BIN → exit-127 `<bin>/python3: No such file or directory`) is
re-attempted in place instead of pointlessly rolling back to await_worker +
respawning the base worker on an intact artifact.

PART 1 — `phase_o5.classify_runner_error`:
  - the interpreter-127 signature → classified "infra";
  - a legit kernel-file "No such file" / a script-open ("can't open file")
    failure → NOT classified infra (tight-scope guard).

PART 2 — `fsm_phase_finalize._o5_post_verify` INFRA-RETRY branch:
  - infra RUNNER_FAILED → re-attempts O5 (does NOT immediately rollback /
    respawn); on retry-success → finalize proceeds (VERIFIED);
  - on bounded-retry-exhaustion → still rolls to await_worker (fail loud);
  - a retry that surfaces a real MISMATCH → handled by the MISMATCH arm;
  - an "algorithm" RUNNER_FAILED → unchanged (no retry, rolls to await_worker).

The O5 gate is never skipped: every re-attempt runs the full real
post_verify_for_finalize — this test only stubs its RESULT sequence.

Run: cd src/scripts/orchestrator && PYTHONPATH=. python3 -m pytest \
     tests/ut/test_debt_o5_infra_0_misclassify_retry.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # orchestrator/

import orchestrator  # noqa: E402,F401  (force module identity for fsm_context read-through)
import events  # noqa: E402
import fsm_phase_finalize as F  # noqa: E402
import phase_o5  # noqa: E402
import state_executor  # noqa: E402
import finalize_pipeline  # noqa: E402
from fsm_context import OrchestratorContext  # noqa: E402
from phase_o5 import classify_runner_error  # noqa: E402

_resolve_o5_report = getattr(F, "_resolve_o5_report")


# ─────────────────────────── PART 1: classify_runner_error ───────────────────────────


@pytest.mark.parametrize("msg", [
    # bash exec-not-found form: `<bin>/python3: No such file or directory` (exit 127).
    # Deliberately WITHOUT the pre-existing "verifier exit" token so this exercises
    # the DEBT-O5-INFRA-0-MISCLASSIFY rule, not the NODE-5 catalog.
    "pass_a re-measure: /usr/local/py311/bin/python3: No such file or directory (exit 127)",
    "bash: line 1: /opt/a5/py3/bin/python3: No such file or directory",
    "backward re-measure: /home/env/bin/python: No such file or directory",
    "/usr/local/py311/bin/python3: cannot execute: required file not found",
    "resolve failed: NPU_PYTHON_BIN=/stale/bin points at a missing python — No such file or directory",
])
def test_interpreter_127_signature_is_infra(msg):
    """Interpreter-resolution failure (stale NPU_PYTHON_BIN, exit 127) → 'infra'."""
    assert classify_runner_error(msg) == "infra"


@pytest.mark.parametrize("msg", [
    # A legit kernel/source file-not-found: bare "No such file" must NOT match.
    "compile error: my_kernel.h: No such file or directory",
    "fatal error: pass_a_runner.hpp: No such file or directory",
    # A python3 dir merely appearing in an unrelated path must NOT match
    # (`/python3.11/...kernel.cpp:` is not the interpreter itself).
    "/opt/python3.11/site-packages/foo/kernel.cpp: No such file or directory",
    # Script-open failure (run_pass_b.py discoverability) — the class that
    # legitimately rolls to await_worker; "can't open file" must veto.
    "python3: can't open file '/root/task/run_pass_b.py': [Errno 2] No such file or directory",
    # Plain exit-127 with no interpreter/file signature at all.
    "verifier returned 127",
])
def test_non_interpreter_not_found_is_not_infra(msg):
    """Tight-scope guard: bare/kernel-file/script-open 'No such file' → 'algorithm'."""
    assert classify_runner_error(msg) == "algorithm"


def test_existing_infra_catalog_still_classifies():
    """Regression guard: the NODE-5 catalog patterns keep classifying 'infra'."""
    assert classify_runner_error("scp aborted: oversized payload") == "infra"
    assert classify_runner_error("pass_a: verifier exit 127; stderr='...'") == "infra"


# ─────────────────────────── PART 2: _o5_post_verify INFRA-RETRY ───────────────────────────


class _O5:
    """Minimal O5Report stub mirroring the fields the handler reads."""

    def __init__(self, verdict, **kw):
        self.verdict = verdict
        self.claimed = kw.get("claimed", {})
        self.measured = kw.get("measured", {})
        self.mismatches = kw.get("mismatches", [])
        self.summary = kw.get("summary", "")
        self.rollback_kind = kw.get("rollback_kind", None)
        self.harness_git_state = kw.get("harness_git_state", "UNKNOWN")
        self.harness_dirty = kw.get("harness_dirty", [])


class _Snap:
    iter_counts: dict = {}


@pytest.fixture
def stub_common(monkeypatch):
    """Neutralize reload + emit + truth-source + rollback recording seams;
    return the list of rollback target states recorded (via record_transition).
    """
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(F, "_sleep", lambda seconds: None)  # no real backoff in ut
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "benchmark")
    monkeypatch.setattr(phase_o5, "format_block_message", lambda op, o5: "")
    monkeypatch.setattr(phase_o5, "record_harness_state", lambda ws, rep: True)
    monkeypatch.setattr(finalize_pipeline, "record_rollback", lambda *a, **k: None)
    recorded: list = []
    monkeypatch.setattr(
        state_executor, "record_transition",
        lambda ws, dec: recorded.append(dec.next_state),
    )
    # below-cap by default (so an exhausted-retry rollback records, not exit-2)
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    return recorded


def _ctx(tmp_path) -> OrchestratorContext:
    ws = tmp_path / "op"
    ws.mkdir()
    return OrchestratorContext(op="op", workspace=ws, lane=0)


def _seq_runner(monkeypatch, results):
    """Patch post_verify_for_finalize to return `results` in order (last repeats),
    and return a call-counter list so the test can assert how many O5 re-measures
    ran (i.e. that a retry actually happened / did not)."""
    calls: list = []
    seq = list(results)

    def _fake(ws, op, lane, runner):
        calls.append(1)
        return seq[len(calls) - 1] if len(calls) <= len(seq) else seq[-1]

    monkeypatch.setattr(phase_o5, "post_verify_for_finalize", _fake)
    return calls


def test_infra_runner_failed_retries_then_succeeds(stub_common, monkeypatch, tmp_path):
    """Infra RUNNER_FAILED → re-attempt O5 in place; retry succeeds (VERIFIED)
    → finalize proceeds (None), NO await_worker rollback recorded.
    """
    calls = _seq_runner(monkeypatch, [
        _O5("RUNNER_FAILED", rollback_kind="infra", summary="/bin/python3: No such file or directory"),
        _O5("VERIFIED", harness_git_state="CLEAN", summary="ok on re-attempt"),
    ])
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is None, "retry-success must proceed to finalize (return None)"
    assert stub_common == [], "must NOT roll back to await_worker when retry succeeds"
    assert len(calls) == 2, "expected 1 initial + 1 retry O5 re-measure"


def test_infra_runner_failed_exhausts_then_rolls_to_await_worker(stub_common, monkeypatch, tmp_path):
    """Infra RUNNER_FAILED that never recovers → after the bounded retries, still
    rolls to await_worker (fail loud, not silent).
    """
    calls = _seq_runner(monkeypatch, [
        _O5("RUNNER_FAILED", rollback_kind="infra", summary="/bin/python3: No such file or directory"),
    ])  # always infra-fail
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is not None and res.action == "continue"
    assert stub_common == ["await_worker"], "exhausted infra retry must fail loud → await_worker"
    # initial + _O5_INFRA_MAX_RETRIES re-attempts, all still infra-failing
    assert len(calls) == 1 + getattr(F, '_O5_INFRA_MAX_RETRIES')


def test_infra_retry_surfaces_real_mismatch_is_handled(stub_common, monkeypatch, tmp_path):
    """A re-attempt that reveals a genuine MISMATCH flows through the MISMATCH
    arm (rolls await_worker) — the retry does not mask a real discrepancy.
    """
    calls = _seq_runner(monkeypatch, [
        _O5("RUNNER_FAILED", rollback_kind="infra", summary="/bin/python3: No such file or directory"),
        _O5("MISMATCH", mismatches=[1, 2], summary="counts differ", claimed={}, measured={}),
    ])
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is not None and res.action == "continue"
    assert stub_common == ["await_worker"]
    assert len(calls) == 2, "one retry then the MISMATCH verdict is dispatched"


def test_algorithm_runner_failed_unchanged_no_retry(stub_common, monkeypatch, tmp_path):
    """Non-infra ('algorithm') RUNNER_FAILED → NO retry (exactly one O5
    re-measure), rolls to await_worker exactly as before.
    """
    calls = _seq_runner(monkeypatch, [
        _O5("RUNNER_FAILED", rollback_kind="algorithm", summary="precision FAIL"),
    ])
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is not None and res.action == "continue"
    assert stub_common == ["await_worker"]
    assert len(calls) == 1, "algorithm RUNNER_FAILED must NOT be retried"


def test_infra_runner_failed_at_cap_still_exits_2(stub_common, monkeypatch, tmp_path):
    """Fail-closed: if await_worker is already iter_cap-exhausted, an infra
    RUNNER_FAILED surviving the retries still hits the loop-guard exit 2 (never
    an infinite loop, never a silent pass).
    """
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: True)
    _seq_runner(monkeypatch, [
        _O5("RUNNER_FAILED", rollback_kind="infra", summary="/bin/python3: No such file or directory"),
    ])
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert (res.action, res.exit_code) == ("return", 2)
    assert stub_common == [], "loop-guard exit must not record a rollback"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_host_transient_mismatch_retries_as_infra(stub_common, monkeypatch, tmp_path):
    """MISMATCH caused by a host H2D copy fault re-attempts in place as infra
    (2026-08-23 CoT) instead of rolling back to await_worker.

    Note (P0-1, 2026-08-28): copy-fault reasons carrying a device error code
    (5070xx/aivec) are now split into the device class (zero backoff,
    same-signature parking) — covered in tests/ut/test_same_signature_parking.py.
    """
    calls = _seq_runner(monkeypatch, [
        _O5(
            "MISMATCH",
            summary="precision status is 'ERROR', not PASS",
            measured={
                "precision": {
                    "reason": (
                        "RuntimeError: copy_between_host_and_device_opapi "
                        "... host bus timeout, connection reset by peer"
                    )
                }
            },
        ),
        _O5("VERIFIED", harness_git_state="CLEAN", summary="ok on re-attempt"),
    ])
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is None, "transient copy-fault mismatch must retry in place, not roll back"
    assert stub_common == [], "must NOT roll back to await_worker"
    assert len(calls) == 2, "expected 1 initial + 1 infra re-attempt"


# ─────────────── PART 3: host copy-fault escalation guard (2026-08-27, 42_CoTAttention) ───────────────
# torch_npu launches asynchronously: a candidate device-kernel hard fault
# (507035/507015, e.g. aivec error 340) surfaces at the NEXT host op — often an
# H2D copy — so the measured `reason` carries the copy-fault markers even when
# the candidate caused it.  42_CoTAttention looped 10+ identical O5 failures
# classified "INFRA, do not edit the candidate" while cot_softmax_f32 was
# deterministically wedging the device.  The guard counts copy-fault
# classifications per candidate binding; past the cap it stops re-attempting
# as infra and dispatches the MISMATCH with a deterministic-fault note.
#
# P0-1 (2026-08-28): reasons carrying a device error code (5070xx / aivec) are
# now classified DEVICE before this host-transient branch and no longer reach
# this guard — they park via the same-signature counter (threshold 2, zero
# backoff; see tests/ut/test_same_signature_parking.py).  This guard still
# covers copy-fault texts WITHOUT a device code, so its fixtures use one.


def _copy_fault_o5(binding: str) -> _O5:
    return _O5(
        "MISMATCH",
        summary="precision status is 'ERROR', not PASS",
        measured={
            "precision": {
                "reason": (
                    "RuntimeError: copy_between_host_and_device_opapi "
                    "... host copy queue timeout"
                ),
                "binding_sha256": binding,
            }
        },
    )


def test_host_copy_fault_escalates_after_cap(stub_common, monkeypatch, tmp_path):
    """Same candidate binding tripping the copy-fault marker past the cap is
    reclassified as a DETERMINISTIC candidate fault: no infra re-attempt, and
    the dispatched MISMATCH carries the debug-recipe note for the worker.
    """
    ctx = _ctx(tmp_path)
    calls = _seq_runner(monkeypatch, [_copy_fault_o5("bindX")])

    cap = getattr(F, "_host_copy_fault_max_transient")()
    for fire in range(1, cap + 1):
        o5 = _resolve_o5_report(ctx, None)
        assert "NOT infra" not in o5.summary, (
            f"fire {fire} (<= cap {cap}) must still get the infra re-attempt"
        )
        assert len(calls) == 2 * fire, (
            "each sub-cap fire = 1 initial O5 + 1 infra re-attempt"
        )

    o5 = _resolve_o5_report(ctx, None)
    assert len(calls) == 2 * cap + 1, "escalated fire must NOT re-attempt O5"
    assert o5.verdict == "MISMATCH"
    assert "NOT infra" in o5.summary
    assert "ASCEND_LAUNCH_BLOCKING=1" in o5.summary, "worker debug recipe attached"

    import json as _json
    ledger = _json.loads(
        (ctx.workspace / getattr(F, "_HOST_COPY_FAULT_ESCALATION_FILE")).read_text()
    )
    assert ledger["bindX"]["count"] == cap + 1


def test_host_copy_fault_escalation_is_per_binding(stub_common, monkeypatch, tmp_path):
    """A DIFFERENT candidate binding starts its own counter — a new candidate
    must not inherit the previous one's escalation (the new candidate could be
    hitting a genuine transient host window).
    """
    ctx = _ctx(tmp_path)
    cap = getattr(F, "_host_copy_fault_max_transient")()
    calls = _seq_runner(monkeypatch, [_copy_fault_o5("bindA")])
    for _ in range(cap):
        _resolve_o5_report(ctx, None)
    assert len(calls) == 2 * cap

    calls_b = _seq_runner(monkeypatch, [_copy_fault_o5("bindB")])
    o5 = _resolve_o5_report(ctx, None)
    assert "NOT infra" not in o5.summary, "fresh binding must not inherit escalation"
    assert len(calls_b) == 2, "fresh binding still gets its infra re-attempt"
