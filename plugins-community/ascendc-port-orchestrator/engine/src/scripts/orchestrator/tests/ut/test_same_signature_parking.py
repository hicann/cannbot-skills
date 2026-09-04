# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""PR875 equiv-review engine fixes — A-stream regression tests (2026-08-28).

Covers the DSH-ruling items implemented in state_executor.py /
fsm_phase_finalize.py / orchestrator.py:

- P0-1: per-op PERSISTENT same-signature failure parking (survives exit 77 /
  process restart — the counter is a workspace dotfile, re-read per call);
  signature = (failure_class, normalized reason, candidate_tree_sha256);
  engine/infra park at >=3, device at >=2 with zero backoff; candidate-class
  MISMATCH feeds a separate "same tree + same case signature >=3" escalation.
- P0-2: O5 repair-reset race guard (candidate stability probes before eval).
- P1-4: engine-block diagnostic handoff accepted only at engine count >=3.
- P2-5: lifetime spawn-cost warning trigger (15 / same-signature engine >=2).
- P0-3 finalize side: repeat_fingerprint near-miss admission gate.

Run: cd src/scripts && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
     orchestrator/tests/ut/test_same_signature_parking.py -q
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # orchestrator/

import orchestrator  # noqa: E402,F401  (force module identity for fsm_context read-through)
import agent_dispatch  # noqa: E402
import events  # noqa: E402
import fsm_phase_finalize as F  # noqa: E402
import phase_o5  # noqa: E402
import state_executor  # noqa: E402
import finalize_pipeline  # noqa: E402
from fsm_context import OrchestratorContext  # noqa: E402


def _module_attr(module, name):
    # Resolve protected module members at call time so monkeypatches remain visible.
    return getattr(module, name)


# ---------------------------------------------------------------------------
# state_executor: signature normalization + persistent counters
# ---------------------------------------------------------------------------


def test_normalize_failure_reason_strips_volatile_tokens():
    a = "run-20260828T015441 eval failed: binding 49ae1123f00dbeef at 2026-08-28T01:54:41Z"
    b = "run-20260101T000000 eval failed: binding 0000000000000000 at 2025-12-31T23:59:59Z"
    assert state_executor.normalize_failure_reason(a) == state_executor.normalize_failure_reason(b)


def test_same_signature_counts_consecutive_and_resets(tmp_path):
    ws = tmp_path
    e1 = state_executor.record_same_signature_failure(ws, "engine", "boom run-aaa", "treeA")
    assert e1["count"] == 1
    # A per-eval run_id / hash churn must NOT fragment the signature.
    e2 = state_executor.record_same_signature_failure(ws, "engine", "boom run-bbb", "treeA")
    assert e2["count"] == 2
    # A different reason inserts a different signature → reset.
    e3 = state_executor.record_same_signature_failure(ws, "engine", "different failure", "treeA")
    assert e3["count"] == 1
    # candidate_tree change is part of the signature → reset.
    e4 = state_executor.record_same_signature_failure(ws, "engine", "different failure", "treeB")
    assert e4["count"] == 1


def test_same_signature_state_persists_across_reads(tmp_path):
    """The counter is file-backed: a 'restarted process' (fresh read) sees it."""
    state_executor.record_same_signature_failure(tmp_path, "infra", "scp down", "treeA")
    state_executor.record_same_signature_failure(tmp_path, "infra", "scp down", "treeA")
    # Simulate process restart: nothing but the workspace file.
    assert state_executor.same_signature_count(tmp_path, "infra") == 2
    assert state_executor.same_signature_count(tmp_path, "engine") == 0
    payload = json.loads((tmp_path / state_executor.SAME_SIGNATURE_STATE_FILE).read_text())
    assert payload["same_signature"]["failure_class"] == "infra"


def test_clear_same_signature_breaks_chain(tmp_path):
    state_executor.record_same_signature_failure(tmp_path, "engine", "boom", "treeA")
    state_executor.record_same_signature_failure(tmp_path, "engine", "boom", "treeA")
    state_executor.clear_same_signature_state(tmp_path)
    assert state_executor.same_signature_count(tmp_path, "engine") == 0
    entry = state_executor.record_same_signature_failure(tmp_path, "engine", "boom", "treeA")
    assert entry["count"] == 1


def test_park_thresholds():
    assert state_executor.same_signature_park_threshold("device") == 2
    assert state_executor.same_signature_park_threshold("engine") == 3
    assert state_executor.same_signature_park_threshold("infra") == 3


def test_candidate_case_counter_resets_on_tree_change(tmp_path):
    sig = "case1:FAIL:mere"
    e1 = state_executor.record_candidate_case_failure(tmp_path, "treeA", sig)
    e2 = state_executor.record_candidate_case_failure(tmp_path, "treeA", sig)
    assert (e1["count"], e2["count"]) == (1, 2)
    e3 = state_executor.record_candidate_case_failure(tmp_path, "treeB", sig)
    assert e3["count"] == 1
    e4 = state_executor.record_candidate_case_failure(tmp_path, "treeB", "case9:FAIL:other")
    assert e4["count"] == 1


# ---------------------------------------------------------------------------
# P1-4: engine-block diagnostic handoff gate
# ---------------------------------------------------------------------------


def _seed_engine_failures(ws, count):
    for _ in range(count):
        state_executor.record_same_signature_failure(ws, "engine", "contract skew", "treeA")


def test_engine_block_handoff_accepted_at_threshold(tmp_path):
    _seed_engine_failures(tmp_path, state_executor.same_signature_park_threshold("engine"))
    decision = state_executor.next_state(
        tmp_path, "→ orchestrator: engine-block-3-identical: brief contract skew",
        from_state="await_worker", dry_run=True,
    )
    assert decision.next_state == "await_user_decision"
    assert "P1-4" in decision.rationale
    # dry_run must not write the transition log.
    assert not (tmp_path / "state_transitions.jsonl").exists()


def test_engine_block_handoff_rejected_below_threshold(tmp_path):
    """Below the engine-class threshold the P1-4 override must NOT fire; the
    handoff keeps its legacy YAML routing (P0abk diagnostic catch-all).
    """
    _seed_engine_failures(tmp_path, 1)
    decision = state_executor.next_state(
        tmp_path, "→ orchestrator: engine-block: premature",
        from_state="await_worker", dry_run=True,
    )
    assert "P1-4" not in decision.rationale, (
        "below-threshold handoff must not take the P1-4 override path"
    )
    assert decision.matched_transition_index != -1 or "P0abk" in decision.rationale


def test_failure_signature_classification_mapping():
    """The DSH §4.b class mapping, locked at the classifier itself."""
    assert _module_attr(F, "_o5_failure_signature_input")(_O5("VERIFIED")) is None
    assert _module_attr(F, "_o5_failure_signature_input")(
        _O5("MISMATCH", measured={})
    )[0] == "candidate"
    assert _module_attr(F, "_o5_failure_signature_input")(_device_o5())[0] == "device"
    assert _module_attr(F, "_o5_failure_signature_input")(_O5(
        "MISMATCH",
        measured={"precision": {"reason": "copy_between_host_and_device timeout"}},
    ))[0] == "infra"
    assert _module_attr(F, "_o5_failure_signature_input")(
        _O5("RUNNER_FAILED", rollback_kind="infra", summary="scp down")
    )[0] == "infra"
    assert _module_attr(F, "_o5_failure_signature_input")(
        _O5("RUNNER_FAILED", summary="unknown verdict", rollback_kind=None)
    )[0] == "engine"
    # candidate_contract re-entries are engine-class even with an infra kind.
    assert _module_attr(F, "_o5_failure_signature_input")(
        _O5("RUNNER_FAILED", summary="candidate rejected before build",
            rollback_kind="infra", failure_kind="candidate_contract")
    )[0] == "engine"
    # The reclassify prefixes survive the measured-payload drop.
    assert _module_attr(F, "_o5_failure_signature_input")(_O5(
        "RUNNER_FAILED", summary=_module_attr(F, "_DEVICE_RECLASSIFY_PREFIX") + "x",
        rollback_kind="infra"
    ))[0] == "device"
    assert _module_attr(F, "_o5_failure_signature_input")(_O5(
        "RUNNER_FAILED", summary=_module_attr(F, "_HOST_TRANSIENT_RECLASSIFY_PREFIX") + "x",
        rollback_kind="infra"
    ))[0] == "infra"


def test_engine_block_handoff_ignored_without_counter(tmp_path):
    decision = state_executor.next_state(
        tmp_path, "→ orchestrator: engine-block: no evidence",
        from_state="await_worker", dry_run=True,
    )
    assert "P1-4" not in decision.rationale


def test_engine_block_handoff_persists_transition_when_applied(tmp_path):
    _seed_engine_failures(tmp_path, 3)
    state_executor.next_state(
        tmp_path, "→ orchestrator: engine-block-3-identical",
        from_state="await_worker", dry_run=False,
    )
    line = (tmp_path / "state_transitions.jsonl").read_text().splitlines()[-1]
    assert json.loads(line)["to_state"] == "await_user_decision"


# ---------------------------------------------------------------------------
# finalize-side seams (shared stubs, mirroring test_fsm_phase_finalize.py)
# ---------------------------------------------------------------------------


_O5 = phase_o5.O5Report


class _Snap:
    iter_counts: dict = {}


@pytest.fixture
def stub_common(monkeypatch, tmp_path):
    import importlib
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(tmp_path / "repair-backups"))
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    sleeps: list = []
    monkeypatch.setattr(F, "_sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "benchmark")
    monkeypatch.setattr(phase_o5, "format_block_message", lambda op, o5: "")
    monkeypatch.setattr(phase_o5, "record_harness_state", lambda ws, rep: True)
    monkeypatch.setattr(finalize_pipeline, "record_rollback", lambda *a, **k: None)
    recorded = []
    monkeypatch.setattr(
        state_executor,
        "record_transition",
        lambda ws, dec: recorded.append(dec.next_state),
    )
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    return recorded, sleeps


def _ctx(tmp_path) -> OrchestratorContext:
    ws = tmp_path / "op"
    ws.mkdir(exist_ok=True)
    return OrchestratorContext(op="op", workspace=ws, lane=0)


def _seq_runner(monkeypatch, results):
    calls: list = []
    seq = list(results)

    def _fake(ws, op, lane, runner):
        calls.append(1)
        return seq[len(calls) - 1] if len(calls) <= len(seq) else seq[-1]

    monkeypatch.setattr(phase_o5, "post_verify_for_finalize", _fake)
    return calls


def _device_o5(tree: str = "t" * 64) -> _O5:
    return _O5(
        "MISMATCH",
        summary="precision status is 'ERROR', not PASS",
        measured={
            "precision": {
                "reason": (
                    "RuntimeError: copy_between_host_and_device_opapi "
                    "... 507035 device error type 3"
                ),
            },
            "performance": {"evaluation_binding": {"candidate_tree_sha256": tree}},
        },
    )


# ---------------------------------------------------------------------------
# P0-1 device-class split: zero backoff + park at 2
# ---------------------------------------------------------------------------


def test_device_mismatch_retries_with_zero_backoff(stub_common, monkeypatch, tmp_path):
    recorded, sleeps = stub_common
    calls = _seq_runner(monkeypatch, [
        _device_o5(),
        _O5("VERIFIED", harness_git_state="CLEAN", summary="recovered"),
    ])
    res = _module_attr(F, "_o5_post_verify")(_ctx(tmp_path), _Snap())
    assert res is None
    assert recorded == []
    assert len(calls) == 2
    assert 600 not in sleeps, "device-class retry must not wait out a 600s window"


def test_device_same_signature_parks_at_two(stub_common, monkeypatch, tmp_path):
    """Two consecutive identical device-class failures → await_user_decision,
    with the device probe/reset/lane advisory — not another worker rollback.
    """
    recorded, sleeps = stub_common
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [_device_o5()])

    res1 = _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert res1.action == "continue"
    assert recorded == ["await_worker"], "first occurrence still rolls back"
    assert state_executor.same_signature_count(ctx.workspace, "device") == 1

    res2 = _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert res2.action == "continue"
    assert recorded == ["await_worker", "await_user_decision"]
    assert 600 not in sleeps


def test_device_park_rationale_carries_advisory(stub_common, monkeypatch, tmp_path):
    recorded = []

    def _capture(ws, dec):
        recorded.append(dec)

    monkeypatch.setattr(state_executor, "record_transition", _capture)
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [_device_o5()])
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    park = recorded[-1]
    assert park.next_state == "await_user_decision"
    assert "ASCEND_LAUNCH_BLOCKING=1" in park.rationale
    assert "different lane" in park.rationale
    assert "507035" in park.rationale or "device" in park.rationale


# ---------------------------------------------------------------------------
# P0-1 engine/infra parking at 3
# ---------------------------------------------------------------------------


def _engine_o5() -> _O5:
    return _O5("RUNNER_FAILED", summary="KeyError: 'binding_sha256' in evidence payload")


def test_engine_same_signature_parks_at_three(stub_common, monkeypatch, tmp_path):
    recorded, _ = stub_common
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [_engine_o5()])
    for round_no in (1, 2):
        res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())
        assert res.action == "continue"
        assert recorded == ["await_worker"] * round_no
    res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert res.action == "continue"
    assert recorded == ["await_worker", "await_worker", "await_user_decision"]
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 3


def test_success_breaks_the_consecutive_chain(stub_common, monkeypatch, tmp_path):
    recorded, _ = stub_common
    ctx = _ctx(tmp_path)
    calls = _seq_runner(monkeypatch, [
        _engine_o5(), _engine_o5(),
        _O5("VERIFIED", harness_git_state="CLEAN", summary="ok"),
        _engine_o5(),
    ])
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 2
    res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert res is None, "VERIFIED proceeds"
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 0
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 1
    assert recorded == ["await_worker"] * 3, "no parking after a broken chain"


def test_infra_same_signature_parks_at_three_with_backoff(stub_common, monkeypatch, tmp_path):
    recorded, sleeps = stub_common
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [
        _O5("RUNNER_FAILED", rollback_kind="infra", summary="scp aborted: oversized payload"),
    ])
    for _ in range(2):
        _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert recorded == ["await_worker"] * 2
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert recorded == ["await_worker"] * 2 + ["await_user_decision"]
    assert state_executor.same_signature_count(ctx.workspace, "infra") == 3
    assert sleeps and all(s == 600 for s in sleeps), "infra keeps the transient-window backoff"


def test_different_class_insertion_resets_engine_chain(stub_common, monkeypatch, tmp_path):
    stub_common
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [
        _engine_o5(),
        _engine_o5(),
        _O5("MISMATCH", mismatches=[1], summary="real precision mismatch", measured={}),
        _engine_o5(),
    ])
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 2
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())  # candidate-class MISMATCH → chain reset
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 0
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 1


# ---------------------------------------------------------------------------
# P0-1 candidate-class case escalation ("改不动=卡死")
# ---------------------------------------------------------------------------


def _candidate_mismatch(tree: str) -> _O5:
    return _O5(
        "MISMATCH",
        mismatches=[1],
        summary="precision FAIL: case 1 MERE over gate",
        measured={
            "precision": {
                "reason": "case 1: MERE 1.0372304916381836 over frozen gate",
                "cases": [
                    {"case_idx": 0, "status": "PASS"},
                    {"case_idx": 1, "status": "FAIL", "reason": "MERE 1.0372304916381836"},
                ],
            },
            "performance": {"evaluation_binding": {"candidate_tree_sha256": tree}},
        },
    )


def test_candidate_case_escalation_parks_at_three(stub_common, monkeypatch, tmp_path):
    recorded, _ = stub_common
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [_candidate_mismatch("a" * 64)])
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert recorded == ["await_worker"] * 2, "candidate MISMATCH does NOT same-signature park"
    assert state_executor.same_signature_count(ctx.workspace, "engine") == 0
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert recorded == ["await_worker"] * 2 + ["await_user_decision"]


def test_candidate_case_counter_resets_on_new_tree(stub_common, monkeypatch, tmp_path):
    recorded, _ = stub_common
    ctx = _ctx(tmp_path)
    _seq_runner(monkeypatch, [_candidate_mismatch("a" * 64)])
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    # Worker changed something (new tree) → progress signal → count restarts.
    _seq_runner(monkeypatch, [_candidate_mismatch("b" * 64)])
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert recorded == ["await_worker"] * 3
    state = state_executor.load_same_signature_state(ctx.workspace)
    assert state["candidate_case"]["count"] == 1
    assert state["candidate_case"]["last_tree_sha256"] == "b" * 64
    _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert recorded == ["await_worker"] * 4, "one round after reset must not park"


# ---------------------------------------------------------------------------
# P0-2: repair-reset race guard
# ---------------------------------------------------------------------------


def _fresh_repair_record(ws: Path, age_seconds: float = 1.0) -> None:
    created = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=age_seconds)
    (ws / ".npubench_candidate_repair.json").write_text(json.dumps({
        "schema": "cannbot.npubench_candidate_repair/v1",
        "created_at_utc": created.isoformat(),
        "moved": ["model_new_ascendc.py"],
    }))


def test_guard_skips_non_npubench_route(stub_common, monkeypatch, tmp_path):
    recorded, _ = stub_common  # expected_truth_source stubbed to "benchmark"
    ctx = _ctx(tmp_path)
    calls = _seq_runner(monkeypatch, [_O5("VERIFIED", harness_git_state="CLEAN")])
    res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())
    assert res is None
    assert len(calls) == 1


def test_guard_suspends_fresh_repair_reset_then_fails_loud(stub_common, monkeypatch, tmp_path):
    recorded, sleeps = stub_common
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "npubench")
    monkeypatch.setenv("AOG_O5_CANDIDATE_GUARD_INTERVAL_SEC", "3")
    ctx = _ctx(tmp_path)
    _fresh_repair_record(ctx.workspace, age_seconds=1.0)
    calls = _seq_runner(monkeypatch, [_O5("VERIFIED", harness_git_state="CLEAN")])

    res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())

    assert res is not None and res.action == "continue"
    assert recorded == ["await_worker"], "exhausted guard fails loud as infra"
    assert len(calls) == 0, "evaluation must never start on an unstable candidate"
    assert len(sleeps) == _module_attr(F, "_o5_candidate_guard_max_retries")()


def test_guard_passes_with_settled_candidate(stub_common, monkeypatch, tmp_path):
    stub_common
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "npubench")
    ctx = _ctx(tmp_path)
    _fresh_repair_record(ctx.workspace, age_seconds=3600.0)  # outside the 60s window
    (ctx.workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    calls = _seq_runner(monkeypatch, [_O5("VERIFIED", harness_git_state="CLEAN")])

    res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())

    assert res is None
    assert len(calls) == 1


def test_guard_detects_tree_churn_between_probes(stub_common, monkeypatch, tmp_path):
    stub_common
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "npubench")
    monkeypatch.setenv("AOG_O5_CANDIDATE_GUARD_INTERVAL_SEC", "0")
    ctx = _ctx(tmp_path)
    (ctx.workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    # Simulate a mid-guard candidate rewrite.
    original_probe = _module_attr(F, "_o5_candidate_stability_probe")
    probed = []

    def churning_probe(workspace, previous):
        issues, tree = original_probe(workspace, previous)
        probed.append(1)
        if len(probed) == 1:
            (workspace / "kernel_marker.txt").write_text("churn\n")
        return issues, tree

    monkeypatch.setattr(F, "_o5_candidate_stability_probe", churning_probe)
    calls = _seq_runner(monkeypatch, [_O5("VERIFIED", harness_git_state="CLEAN")])

    res = _module_attr(F, "_o5_post_verify")(ctx, _Snap())

    assert res is None
    assert len(probed) >= 2, "the churn probe must have forced a re-probe"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# P0-3 finalize side: repeat_fingerprint near-miss admission
# ---------------------------------------------------------------------------


def _write_precision_report(ws: Path, cases: list) -> None:
    evidence = ws / "npubench_evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "precision_report.json").write_text(json.dumps({"cases": cases}))


def test_repeat_fingerprint_admission_classes(tmp_path):
    _write_precision_report(tmp_path, [
        {"case_idx": 0, "repeat_fingerprint": {"class": "bimodal"}},
        {"case_idx": 1, "repeat_fingerprint": {"class": "deterministic-fail"}},
        {"case_idx": 2},
        {"case_idx": 3, "repeat_fingerprint": {"class": "reference-unstable"}},
        {"case_idx": 4, "repeat_fingerprint": {"class": "stable-pass"}},
    ])
    assert _module_attr(F, "_near_miss_inadmissible_cases")(tmp_path) == {
        0: "bimodal",
        3: "reference-unstable",
    }
    assert F.near_miss_case_admitted(tmp_path, 0) is False
    assert F.near_miss_case_admitted(tmp_path, 3) is False
    assert F.near_miss_case_admitted(tmp_path, 1) is True
    assert F.near_miss_case_admitted(tmp_path, 4) is True
    # No fingerprint field → legacy behavior (admitted here).
    assert F.near_miss_case_admitted(tmp_path, 2) is True


def test_repeat_fingerprint_missing_report_is_noop(tmp_path):
    assert _module_attr(F, "_repeat_fingerprint_classes")(tmp_path) == {}
    assert _module_attr(F, "_near_miss_inadmissible_cases")(tmp_path) == {}
    assert F.near_miss_case_admitted(tmp_path, 7) is True


def test_mismatch_event_carries_near_miss_inadmissible_cases(stub_common, monkeypatch, tmp_path):
    recorded, _ = stub_common
    emitted = []
    monkeypatch.setattr(events, "emit", lambda ws, name, **kw: emitted.append((name, kw)))
    ctx = _ctx(tmp_path)
    _write_precision_report(ctx.workspace, [
        {"case_idx": 5, "repeat_fingerprint": {"class": "bimodal"}},
    ])
    _seq_runner(monkeypatch, [_candidate_mismatch("a" * 64)])

    _module_attr(F, "_o5_post_verify")(ctx, _Snap())

    block = next(kw["data"] for n, kw in emitted if n == "orchestrator.phase_o5_block")
    assert block["near_miss_inadmissible_cases"] == {"5": "bimodal"}
    assert recorded == ["await_worker"]


# ---------------------------------------------------------------------------
# P2-5: lifetime spawn-cost warning trigger
# ---------------------------------------------------------------------------


def test_high_spawn_cost_trigger_thresholds():
    assert _module_attr(orchestrator, "_high_spawn_cost_trigger")(0, 0) is None
    assert _module_attr(orchestrator, "_high_spawn_cost_trigger")(14, 1) is None
    assert _module_attr(orchestrator, "_high_spawn_cost_trigger")(15, 0) == "lifetime_spawn_count>=15"
    assert _module_attr(orchestrator, "_high_spawn_cost_trigger")(0, 2) == "same_signature_engine>=2"
    assert _module_attr(orchestrator, "_high_spawn_cost_trigger")(30, 5) == "lifetime_spawn_count>=15"


# ---------------------------------------------------------------------------
# P1-2: whitelisted engine-exception bounded re-spawn (agent_dispatch)
# ---------------------------------------------------------------------------


class _Result:
    success = True
    is_error = False
    output_text = "ok"


def test_engine_crash_whitelisted_exception_respawns_and_recovers(monkeypatch, tmp_path):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise KeyError("binding_sha256")
        return _Result()

    monkeypatch.setattr(agent_dispatch, "_spawn_for_state_once", flaky)
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / ".agent_died_at_await_worker").write_text('{"reason": "prior crash"}')

    result = agent_dispatch.spawn_for_state("op", ws, "await_worker", lane=0, spawn_index=1)

    assert isinstance(result, _Result)
    assert len(calls) == 3
    assert not (ws / ".agent_died_at_await_worker").exists(), (
        "recovered spawn must clear the stale agent_died marker"
    )


def test_engine_crash_unknown_exception_propagates_immediately(monkeypatch, tmp_path):
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise ValueError("genuine worker/input problem")

    monkeypatch.setattr(agent_dispatch, "_spawn_for_state_once", broken)
    with pytest.raises(ValueError):
        agent_dispatch.spawn_for_state("op", tmp_path, "await_worker", lane=0, spawn_index=1)
    assert len(calls) == 1, "unknown exceptions keep the manual agent_died path"


def test_engine_crash_whitelist_exhaustion_propagates(monkeypatch, tmp_path):
    calls = []

    def always_broken(*args, **kwargs):
        calls.append(1)
        raise NameError("name 'npubench_evidence' is not defined")

    monkeypatch.setattr(agent_dispatch, "_spawn_for_state_once", always_broken)
    with pytest.raises(NameError):
        agent_dispatch.spawn_for_state("op", tmp_path, "await_worker", lane=0, spawn_index=1)
    assert len(calls) == 1 + _module_attr(agent_dispatch, "_ENGINE_CRASH_RESPAWN_MAX")


def test_engine_crash_contract_schema_message_whitelisted(monkeypatch, tmp_path):
    calls = []

    def contract_skew(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("evidence payload violates contract schema v6: missing binding_sha256")

    monkeypatch.setattr(agent_dispatch, "_spawn_for_state_once", contract_skew)
    with pytest.raises(RuntimeError):
        agent_dispatch.spawn_for_state("op", tmp_path, "await_worker", lane=0, spawn_index=1)
    assert len(calls) == 1 + _module_attr(agent_dispatch, "_ENGINE_CRASH_RESPAWN_MAX"), (
        "contract-schema message markers are whitelisted even on non-NameError/KeyError types"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
