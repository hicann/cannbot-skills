# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Fast per-handler unit tests for fsm_phase_finalize.handle_finalize.

DEBT-201: the finalize FSM state slice was extracted from run_single_op into
fsm_phase_finalize. These stub the sibling seams + a fake OrchestratorContext to
lock the handler's control-flow mapping (each original `continue` / `return N`
→ HandlerResult) WITHOUT the full run_single_op boot. The e2e finalize paths
stay locked by test_p0ww_partial_persist_drives_finalize +
test_phase_o5_blocks_finalize_on_mismatch; these are the cheap complement.

Run: cd src/scripts/orchestrator && PYTHONPATH=. python3 -m pytest \
     tests/ut/test_fsm_phase_finalize.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

# Force the orchestrator.py MODULE identity into sys.modules["orchestrator"]
# (the dir is on sys.path, so the file wins over the bare package __init__).
# fsm_context._orch() read-through resolves against this identity; without the
# import, a pytest session that only loaded the package shell leaves _orch()
# pointing at a stub lacking the private FSM helpers. Mirrors the
# characterization test's top-level `import orchestrator`.
import orchestrator  # noqa: E402,F401
import events  # noqa: E402
import fsm_phase_finalize as F  # noqa: E402
import kb_invoke  # noqa: E402
import phase_o5  # noqa: E402
import phase_o5_runner  # noqa: E402
import state_executor  # noqa: E402
import finalize_pipeline  # noqa: E402
from fsm_context import OrchestratorContext  # noqa: E402

_bump_host_copy_fault_escalation = getattr(F, "_bump_host_copy_fault_escalation")
_o5_candidate_tree_key = getattr(F, "_o5_candidate_tree_key")
_is_direct_npubench_candidate_failure = getattr(
    F, "_is_direct_npubench_candidate_failure"
)


class _O5:
    def __init__(self, verdict, **kw):
        self.verdict = verdict
        self.claimed = kw.get("claimed", {})
        self.measured = kw.get("measured", {})
        self.mismatches = kw.get("mismatches", [])
        self.summary = kw.get("summary", "")
        self.rollback_kind = kw.get("rollback_kind", None)
        self.failure_kind = kw.get("failure_kind", None)
        # DEBT-213(b): mirror O5Report's defaults exactly — the handler reads
        # these on the VERIFIED/PROVISIONAL paths, and a stub that silently
        # disagrees with the real dataclass is how a stub starts lying.
        self.harness_git_state = kw.get("harness_git_state", "UNKNOWN")
        self.harness_dirty = kw.get("harness_dirty", [])


@pytest.fixture
def stub_common(monkeypatch, tmp_path):
    """Neutralize the reload + emit + truth-source seams shared by every path."""
    import importlib
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(tmp_path / "repair-backups"))
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(F, "_sleep", lambda seconds: None)  # no real backoff in ut
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "benchmark")
    monkeypatch.setattr(phase_o5, "format_block_message", lambda op, o5: "")
    monkeypatch.setattr(finalize_pipeline, "record_rollback", lambda *a, **k: None)
    recorded = []
    monkeypatch.setattr(
        state_executor, "record_transition",
        lambda ws, dec: recorded.append(dec.next_state),
    )
    return recorded


def _ctx(tmp_path) -> OrchestratorContext:
    ws = tmp_path / "op"
    ws.mkdir()
    return OrchestratorContext(op="op", workspace=ws, lane=0)


class _Snap:
    iter_counts: dict = {}


def _stub_candidate_contract_failure(monkeypatch, summary, *, at_cap=False):
    """Stub post_verify_for_finalize with a pre-build candidate-contract rejection.

    Shared by the candidate-contract rollback tests so the identical stub
    wiring lives in exactly one place.
    """
    monkeypatch.setattr(
        phase_o5,
        "post_verify_for_finalize",
        lambda ws, op, lane, runner: _O5(
            "RUNNER_FAILED",
            summary=summary,
            rollback_kind="infra",
            failure_kind="candidate_contract",
        ),
    )
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: at_cap)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)


def test_o5_mismatch_worker_at_cap_returns_exit_2(stub_common, monkeypatch, tmp_path):
    """MISMATCH + await_worker already at iter_cap → loop-guard exit 2 (no
    rollback recorded).
    """
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: _O5("MISMATCH", mismatches=[1]))
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: True)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    res = F.handle_finalize(_ctx(tmp_path), _Snap())
    assert (res.action, res.exit_code) == ("return", 2)
    assert stub_common == []


def test_o5_mismatch_rollback_continues(stub_common, monkeypatch, tmp_path):
    """MISMATCH + worker below cap → record rollback + continue."""
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: _O5("MISMATCH", mismatches=[1]))
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    res = F.handle_finalize(_ctx(tmp_path), _Snap())
    assert res.action == "continue"
    assert stub_common == ["await_worker"]  # legacy rollback target


def test_o5_runner_failed_rollback_continues(stub_common, monkeypatch, tmp_path):
    """RUNNER_FAILED + worker below cap → record rollback + continue."""
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: _O5("RUNNER_FAILED", summary="ssh down"))
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    res = F.handle_finalize(_ctx(tmp_path), _Snap())
    assert res.action == "continue"
    assert stub_common == ["await_worker"]


def test_o5_runner_failed_at_cap_returns_exit_2(stub_common, monkeypatch, tmp_path):
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: _O5("RUNNER_FAILED", summary="x"))
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: True)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    res = F.handle_finalize(_ctx(tmp_path), _Snap())
    assert (res.action, res.exit_code) == ("return", 2)


def test_o5_direct_910_capability_stop_returns_exit_2_without_rollback(
    stub_common, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        phase_o5,
        "post_verify_for_finalize",
        lambda ws, op, lane, runner: _O5(
            "RUNNER_FAILED",
            summary="A5_SOC_UNSUPPORTED_FOR_VALIDATION: Ascend910B3",
            rollback_kind="target_capability",
        ),
    )
    res = F.handle_finalize(_ctx(tmp_path), _Snap())
    assert (res.action, res.exit_code) == ("return", 2)
    assert stub_common == []


def test_tilelang_candidate_rejection_rolls_back_to_await_worker(stub_common, monkeypatch, tmp_path):
    """Anti-copy independence-gate rejections are worker-fixable: route
    back to await_worker (re-author independently) instead of the terminal
    direct-npubench stop.
    """
    ctx = _ctx(tmp_path)
    (ctx.workspace / ".opgen_state.json").write_text(
        '{"port_source":{"kind":"port-aclnn-tilelang2ascendc"},'
        '"reference":{"source":"npubench"}}',
        encoding="utf-8",
    )
    (ctx.workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    _stub_candidate_contract_failure(
        monkeypatch,
        "candidate rejected before build: TileLang2AscendC candidate "
        "only changes comments/formatting from staged kernel source "
        "(kernel/op_kernel/cot_act1.cpp): kernel/op_kernel/cot_act1.cpp",
    )

    result = F.handle_finalize(ctx, _Snap())

    assert result.action == "continue"
    assert stub_common == ["await_worker"]


@pytest.mark.parametrize("failure_kind", ["target_build", "target_device", "evaluator"])
def test_npubench_target_or_evaluator_failure_stops_without_worker_respawn(
    stub_common, monkeypatch, tmp_path, failure_kind
):
    ctx = _ctx(tmp_path)
    state = {
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
        "reference": {"source": "npubench"},
    }
    (ctx.workspace / ".opgen_state.json").write_text(json.dumps(state), encoding="utf-8")
    (ctx.workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        phase_o5,
        "post_verify_for_finalize",
        lambda ws, op, lane, runner: _O5(
            "RUNNER_FAILED",
            summary=f"npubench runner reported error: {failure_kind} failed",
            rollback_kind="infra",
            failure_kind=failure_kind,
        ),
    )

    result = F.handle_finalize(ctx, _Snap())

    assert (result.action, result.exit_code) == ("return", 2)
    assert stub_common == []


def test_npubench_candidate_failure_is_not_retried_as_infra(
    stub_common, monkeypatch, tmp_path
):
    """The receipt category is dispatched before bounded infra retries."""
    ctx = _ctx(tmp_path)
    (ctx.workspace / ".opgen_state.json").write_text(
        json.dumps({
            "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
            "reference": {"source": "npubench"},
        }),
        encoding="utf-8",
    )
    (ctx.workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    calls = []

    def verify(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _O5(
                "RUNNER_FAILED",
                summary="target compiler failed transiently",
                rollback_kind="infra",
                failure_kind="target_build",
            )
        return _O5(
            "RUNNER_FAILED",
            summary="candidate rejected before build: missing CMakeLists.txt",
            rollback_kind="infra",
            failure_kind="candidate_contract",
        )

    monkeypatch.setattr(F, "_o5_runner_for_workspace", lambda *a, **k: object())
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize", verify)
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)

    # Late binding: resolves after the monkeypatches above, so the real helper runs.
    from fsm_phase_finalize import _o5_post_verify

    result = _o5_post_verify(ctx, _Snap())

    assert result.action == "continue"
    assert len(calls) == 2
    assert stub_common == ["await_worker"]


def test_npubench_candidate_failure_persists_worker_handoff_and_reason(
    monkeypatch, tmp_path
):
    """The real handler writer and rollback brief reader share one durable record."""
    ctx = _ctx(tmp_path)
    repair_backup_root = tmp_path / "repair-backups"
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(repair_backup_root))
    (ctx.workspace / "kernel" / "arch35").mkdir(parents=True)
    (ctx.workspace / "kernel" / "arch35" / "pybind11.cpp").write_text(
        "stale candidate\n", encoding="utf-8"
    )
    (ctx.workspace / "op_host" / "arch35").mkdir(parents=True)
    (ctx.workspace / "op_host" / "arch35" / "stale.cpp").write_text(
        "stale candidate\n", encoding="utf-8"
    )
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    o5 = _O5(
        "RUNNER_FAILED",
        summary=(
            "npubench runner reported error: candidate rejected before build: "
            "TileLang2AscendC candidate requires a regular kernel/CMakeLists.txt"
        ),
        rollback_kind="infra",
        failure_kind="candidate_contract",
    )

    from fsm_phase_finalize import _handle_npubench_candidate_failure

    result = _handle_npubench_candidate_failure(ctx, _Snap(), o5)

    assert result.action == "continue"
    transition = json.loads(
        (ctx.workspace / "state_transitions.jsonl").read_text().splitlines()[-1]
    )
    rollback = json.loads(
        (ctx.workspace / ".rollback_history.jsonl").read_text().splitlines()[-1]
    )
    assert transition["to_state"] == "await_worker"
    assert transition["rollback_kind"] == "algorithm"
    assert "legacy pass_b verifier" in transition["rationale"]
    assert rollback["gate"] == "phase_o5_npubench_candidate_contract"
    assert "kernel/CMakeLists.txt" in rollback["reason"]
    repair = json.loads(
        (ctx.workspace / ".npubench_candidate_repair.json").read_text()
    )
    assert repair["schema"] == "cannbot.npubench_candidate_repair/v1"
    assert repair["failure_kind"] == "candidate_contract"
    assert "kernel/CMakeLists.txt" in repair["failure_reason"]
    assert set(repair["moved"]) == {"kernel/", "op_host/"}
    assert not (ctx.workspace / "kernel").exists()
    assert not (ctx.workspace / "op_host").exists()
    assert (repair_backup_root / "op" / repair["archive_id"] / "kernel" / "arch35" / "pybind11.cpp").is_file()

    from briefs._common import rollback_context_block

    block = rollback_context_block(ctx.workspace)
    assert "Previous spawn rejected by finalize gate" in block
    assert "kernel/CMakeLists.txt" in block


def test_real_measured_result_report_and_fsm_route_candidate_failure(
    monkeypatch, tmp_path
):
    """The provider taxonomy survives MeasuredResult -> O5Report -> FSM."""
    import importlib

    ctx = _ctx(tmp_path)
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(tmp_path / "repair-backups"))
    (ctx.workspace / ".opgen_state.json").write_text(
        json.dumps(
            {
                "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
                "reference": {"source": "npubench"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(importlib, "reload", lambda module: module)
    monkeypatch.setattr(events, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda ws: "npubench")
    monkeypatch.setattr(phase_o5, "format_block_message", lambda op, o5: "")
    monkeypatch.setattr(F, "_o5_runner_for_workspace", lambda *args, **kwargs: object())
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)

    measured = phase_o5.MeasuredResult(
        runner_error=(
            "candidate rejected before build: "
            "A5_SOC_UNSUPPORTED_FOR_VALIDATION appeared in candidate diagnostics"
        ),
        rollback_kind="infra",
        failure_kind="candidate_contract",
    )

    from phase_o5 import _npubench_o5_report

    def real_post_verify(workspace, op, *, lane, runner):
        return _npubench_o5_report(workspace, op, lane, lambda *_args: measured)

    monkeypatch.setattr(phase_o5, "post_verify_for_finalize", real_post_verify)

    # Late binding: resolves after the monkeypatches above, so the real helper runs.
    from fsm_phase_finalize import _o5_post_verify

    result = _o5_post_verify(ctx, _Snap())

    assert result.action == "continue"
    transition = json.loads(
        (ctx.workspace / "state_transitions.jsonl").read_text().splitlines()[-1]
    )
    assert transition["to_state"] == "await_worker"
    assert transition["rollback_kind"] == "algorithm"


def test_npubench_candidate_contract_failure_reenters_worker(
    stub_common, monkeypatch, tmp_path
):
    """A pre-build candidate defect is repairable by the authoring worker."""
    ctx = _ctx(tmp_path)
    state = {
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
        "reference": {"source": "npubench"},
    }
    (ctx.workspace / ".opgen_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (ctx.workspace / "model_new_ascendc.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    _stub_candidate_contract_failure(
        monkeypatch,
        "npubench runner reported error: candidate rejected before build: "
        "TileLang2AscendC candidate requires a regular kernel/CMakeLists.txt",
    )

    result = F.handle_finalize(ctx, _Snap())

    assert result.action == "continue"
    assert stub_common == ["await_worker"]


def test_npubench_candidate_failure_at_worker_cap_stops_without_handoff(
    stub_common, monkeypatch, tmp_path
):
    """A capped candidate repair must terminate without another rollback."""
    ctx = _ctx(tmp_path)
    (ctx.workspace / ".opgen_state.json").write_text(
        json.dumps(
            {
                "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
                "reference": {"source": "npubench"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(F, "_o5_runner_for_workspace", lambda *args, **kwargs: object())
    _stub_candidate_contract_failure(
        monkeypatch,
        "candidate rejected before build: missing kernel/CMakeLists.txt",
        at_cap=True,
    )

    # Late binding: resolves after the monkeypatches above, so the real helper runs.
    from fsm_phase_finalize import _o5_post_verify

    result = _o5_post_verify(ctx, _Snap())

    assert (result.action, result.exit_code) == ("return", 2)
    assert stub_common == []
    assert not (ctx.workspace / ".rollback_history.jsonl").exists()
    assert not (ctx.workspace / "state_transitions.jsonl").exists()


def test_legacy_runner_failure_keeps_await_worker_rollback(
    stub_common, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        phase_o5,
        "post_verify_for_finalize",
        lambda ws, op, lane, runner: _O5("RUNNER_FAILED", summary="ssh down"),
    )
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)

    result = F.handle_finalize(_ctx(tmp_path), _Snap())

    assert result.action == "continue"
    assert stub_common == ["await_worker"]


def test_scheme_drift_and_missing_snapshot_are_refreezable():
    """Digest-scheme drift / missing snapshot are harness-state issues the
    finalize self-heals (clear + re-run O5), not worker-fixable rollbacks.
    """
    from fsm_phase_finalize import _eligibility_is_harness_refreezable

    assert _eligibility_is_harness_refreezable(
        {"reason": "NPUBENCH_EVIDENCE_INVALID: candidate digest scheme drift "
         "(evidence frozen under scheme 'npubench-candidate-scope/v2', ...)"}
    )
    assert _eligibility_is_harness_refreezable(
        {"reason": "NPUBENCH_EVIDENCE_INVALID: immutable candidate snapshot is "
         "missing or escapes workspace"}
    )
    assert not _eligibility_is_harness_refreezable(
        {"reason": "NPUBENCH_EVIDENCE_INVALID: current candidate scope differs "
         "from the frozen evaluation snapshot"}
    )
    assert not _eligibility_is_harness_refreezable({"reason": "some worker fixable issue"})


def test_o5_mismatch_rollback_target_is_await_worker(stub_common, monkeypatch, tmp_path):
    """O5 MISMATCH rollback always targets await_worker (re-iterate emission
    via the worker).
    """
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: _O5("MISMATCH", mismatches=[1]))
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    res = F.handle_finalize(_ctx(tmp_path), _Snap())
    assert res.action == "continue"
    assert stub_common == ["await_worker"]


# ---------------------------------------------------------------------------
# DEBT-213(b): the PROVISIONAL arm of _o5_post_verify.
#
# These exist because deleting the entire PROVISIONAL branch from the handler
# killed ZERO tests: the branch returns None and falls through, so nothing
# above notices, yet it holds the ONLY call site of record_harness_state —
# the one thing that makes a dirty-harness finding durable. An unguarded arm
# implementing "a check whose finding nobody can see later is no check at
# all" is exactly the rot this ticket is about.
#
# Driven at _o5_post_verify (not handle_finalize) because PROVISIONAL is a
# proceed path: handle_finalize would run on into real archive promotion.
# ---------------------------------------------------------------------------


@pytest.fixture
def o5_observers(monkeypatch):
    """Make the two PROVISIONAL side effects observable.

    stub_common silences events.emit; re-patch it here (per-test patch is
    applied after the fixture and wins) so the emission is asserted, not
    assumed.
    """
    emitted: list = []
    stamped: list = []
    monkeypatch.setattr(events, "emit",
                        lambda ws, name, **kw: emitted.append((name, kw)))
    monkeypatch.setattr(phase_o5, "record_harness_state",
                        lambda ws, rep: (stamped.append((ws, rep)), True)[1])
    return emitted, stamped


def test_o5_stub_mirrors_real_o5report_harness_defaults():
    """Guard against the stub drifting from the dataclass it impersonates."""
    real = phase_o5.O5Report(verdict="VERIFIED")
    stub = _O5("VERIFIED")
    assert stub.harness_git_state == real.harness_git_state
    assert stub.harness_dirty == real.harness_dirty


def test_o5_provisional_stamps_emits_and_proceeds(
    stub_common, o5_observers, monkeypatch, tmp_path
):
    """PROVISIONAL: no rollback, event emitted, harness state recorded,
    finalize proceeds. Dies if the PROVISIONAL branch is deleted.
    """
    emitted, stamped = o5_observers
    dirty = ["src/scripts/orchestrator/phase_o5_runner.py"]
    o5 = _O5("PROVISIONAL", harness_git_state="DIRTY", harness_dirty=dirty,
             summary="O5 PROVISIONAL (DEBT-213): ...")
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: o5)
    ctx = _ctx(tmp_path)

    res = getattr(F, '_o5_post_verify')(ctx, _Snap())

    # proceeds — PROVISIONAL is not a rollback (part (c) re-verifies, not (b))
    assert res is None
    assert stub_common == [], "PROVISIONAL must not record a rollback"
    # the finding is durable
    assert len(stamped) == 1, "record_harness_state must be called"
    assert stamped[0][0] == ctx.workspace
    assert stamped[0][1] is o5
    # the finding is visible
    names = [n for n, _ in emitted]
    assert "orchestrator.phase_o5_provisional" in names
    data = next(kw["data"] for n, kw in emitted
                if n == "orchestrator.phase_o5_provisional")
    assert data["harness_git_state"] == "DIRTY"
    assert data["harness_dirty"] == dirty


def test_o5_verified_stamps_harness_state(
    stub_common, o5_observers, monkeypatch, tmp_path
):
    """VERIFIED also records — a missing harness_pristine block must mean
    "no check ran", not "check passed".
    """
    _, stamped = o5_observers
    o5 = _O5("VERIFIED", harness_git_state="CLEAN", summary="O5 VERIFIED: ...")
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: o5)
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is None
    assert len(stamped) == 1, "VERIFIED must record the CLEAN evidence too"


def test_o5_unknown_verdict_fails_closed(stub_common, monkeypatch, tmp_path):
    monkeypatch.setattr(
        phase_o5, "post_verify_for_finalize", lambda ws, op, lane, runner: _O5("SURPRISE")
    )
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda ws, st: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda st, workspace=None: 9)
    result = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert result.action == "continue"
    assert stub_common == ["await_worker"]


def test_o5_skipped_does_not_stamp(stub_common, o5_observers, monkeypatch, tmp_path):
    """SKIPPED never re-measured, so there is no verdict to qualify."""
    _, stamped = o5_observers
    monkeypatch.setattr(phase_o5, "post_verify_for_finalize",
                        lambda ws, op, lane, runner: _O5("SKIPPED"))
    res = getattr(F, '_o5_post_verify')(_ctx(tmp_path), _Snap())
    assert res is None
    assert stamped == []


# ---------------------------------------------------------------------------
# KB merge safety ordering: worker return never promotes knowledge.  Finalize
# must pass O5, post-worker critic/eligibility, and delivery-scoped static check
# before archive promotion and merge.
# ---------------------------------------------------------------------------


class _BackwardDelivery:
    name = "backward"

    @staticmethod
    def kernel_cpp_dirs():
        return ("kernel",)


def _write_clean_delivery(ctx: OrchestratorContext) -> None:
    kernel = ctx.workspace / "kernel"
    kernel.mkdir()
    (kernel / "op_kernel.h").write_text(
        "#include <kernel_operator.h>\n"
        "using namespace AscendC;\n"
        "class OpKernel {\n"
        "  GlobalTensor<float> gm;\n"
        "  TQue<QuePosition::VECIN, 1> queue;\n"
        "  __aicore__ void Process() {\n"
        "    LocalTensor<float> x;\n"
        "    DataCopy(x, gm, 8);\n"
        "    Add(x, x, x, 8);\n"
        "    EnQue(queue, x);\n"
        "  }\n"
        "};\n"
    )


def test_o5_failure_never_reaches_static_or_merge(monkeypatch, tmp_path):
    """O5 rejection must stop before any static check or knowledge merge."""
    ctx = _ctx(tmp_path)
    calls = []
    monkeypatch.setattr(F, "_o5_post_verify",
                        lambda *_: (calls.append("o5_fail"), F.HandlerResult.cont())[1])
    monkeypatch.setattr(F, "_run_delivery_static_check",
                        lambda *_: calls.append("static"))
    monkeypatch.setattr(kb_invoke, "merge_one", lambda *_: calls.append("merge"))

    res = F.handle_finalize(ctx, _Snap())

    assert res.action == "continue"
    assert calls == ["o5_fail"]


def test_critic_eligibility_failure_never_reaches_static_or_merge(
    monkeypatch, tmp_path
):
    """Critic/eligibility rejection must leave staged knowledge untouched."""
    ctx = _ctx(tmp_path)
    calls = []
    monkeypatch.setattr(F, "_o5_post_verify", lambda *_: calls.append("o5"))
    monkeypatch.setattr(F, "_run_perf_capture", lambda *_: calls.append("perf"))
    monkeypatch.setattr(
        F, "_run_finalize_prep", lambda *_: (calls.append("critic"), None)[1]
    )
    monkeypatch.setattr(
        F,
        "_check_eligibility_and_rollback",
        lambda *_: (calls.append("eligibility_fail"), F.HandlerResult.cont())[1],
    )
    monkeypatch.setattr(F, "_run_delivery_static_check",
                        lambda *_: calls.append("static"))
    monkeypatch.setattr(kb_invoke, "merge_one", lambda *_: calls.append("merge"))

    res = F.handle_finalize(ctx, _Snap())

    assert res.action == "continue"
    assert calls == ["o5", "perf", "critic", "eligibility_fail"]


def test_datacopy_byte_count_static_failure_blocks_promotion_and_merge(
    monkeypatch, tmp_path
):
    """A concrete memory-safety finding from the existing checker fails closed."""
    ctx = _ctx(tmp_path)
    kernel = ctx.workspace / "kernel"
    kernel.mkdir()
    (kernel / "op_kernel.h").write_text(
        "#include <kernel_operator.h>\n"
        "using namespace AscendC;\n"
        "class OpKernel {\n"
        "  GlobalTensor<float> gm;\n"
        "  TQue<QuePosition::VECIN, 1> queue;\n"
        "  __aicore__ void Process() {\n"
        "    LocalTensor<float> x;\n"
        "    unsigned copyBytes = 32;\n"
        "    DataCopy(x, gm, copyBytes);\n"
        "    Add(x, x, x, 8);\n"
        "    EnQue(queue, x);\n"
        "  }\n"
        "};\n"
    )
    (ctx.workspace / "knowledge_update.md").write_text("candidate\n" * 20)
    monkeypatch.setattr(finalize_pipeline, "_get_active_plugin",
                        lambda _ws: _BackwardDelivery())
    calls = []
    monkeypatch.setattr(finalize_pipeline, "finalize_op",
                        lambda *_: calls.append("promotion"))
    monkeypatch.setattr(kb_invoke, "merge_one", lambda *_: calls.append("merge"))
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)

    report = getattr(F, '_run_delivery_static_check')(ctx.workspace)
    res = getattr(F, '_promote_and_route')(ctx)

    assert report["passed"] is False
    assert any(
        not item["report"]["checks"]["datacopy_byte_count"]["passed"]
        for item in report["reports"]
    )
    # 2026-08-30 (PR13 WP-A / A.4.2): a harness-side gate failure lands on an
    # FSM transition to await_user_decision, never a bare exit 7.
    assert res.action == "continue"
    assert state_executor.current_state(ctx.workspace) == "await_user_decision"
    assert calls == []


def test_static_check_does_not_scan_unrelated_or_build_only_sources(
    monkeypatch, tmp_path
):
    """No current delivery source must not fall back to a workspace-wide scan."""
    ctx = _ctx(tmp_path)
    unrelated = ctx.workspace / "prior_art"
    unrelated.mkdir()
    (unrelated / "unsafe_old.h").write_text(
        "void old() { unsigned copyBytes = 32; DataCopy(x, y, copyBytes); }\n"
    )
    build = ctx.workspace / "kernel" / "build"
    build.mkdir(parents=True)
    (build / "generated_old.h").write_text(
        "void generated() { unsigned copyBytes = 32; DataCopy(x, y, copyBytes); }\n"
    )
    monkeypatch.setattr(finalize_pipeline, "_get_active_plugin",
                        lambda _ws: _BackwardDelivery())

    report = getattr(F, '_run_delivery_static_check')(ctx.workspace)

    assert report["passed"] is True
    assert report["skipped"] == "no declared present AscendC delivery source"


def test_success_order_o5_critic_static_promotion_then_merge(monkeypatch, tmp_path):
    """A successful finalize runs every independent gate before KB merge."""
    ctx = _ctx(tmp_path)
    _write_clean_delivery(ctx)
    (ctx.workspace / "knowledge_update.md").write_text("validated knowledge\n" * 20)
    monkeypatch.setattr(finalize_pipeline, "_get_active_plugin",
                        lambda _ws: _BackwardDelivery())
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    order = []
    monkeypatch.setattr(F, "_o5_post_verify", lambda *_: order.append("o5"))
    monkeypatch.setattr(F, "_run_perf_capture", lambda *_: order.append("perf"))
    monkeypatch.setattr(
        F, "_run_finalize_prep", lambda *_: (order.append("critic"), None)[1]
    )
    monkeypatch.setattr(
        F,
        "_check_eligibility_and_rollback",
        lambda *_: (order.append("eligibility"), None)[1],
    )
    real_static_check = getattr(F, '_run_delivery_static_check')

    def checked(workspace):
        order.append("static")
        return real_static_check(workspace)

    monkeypatch.setattr(F, "_run_delivery_static_check", checked)

    def promote(op, workspace):
        order.append("promotion")
        return finalize_pipeline.FinalizeReport(
            op=op, workspace=workspace, archive_dir=workspace / "archive"
        )

    monkeypatch.setattr(finalize_pipeline, "finalize_op", promote)
    monkeypatch.setattr(
        kb_invoke,
        "merge_one",
        lambda *_: (order.append("merge"), {"success": True})[1],
    )

    class _Decision:
        from_state = "finalize"
        next_state = "done"
        rationale = "validated"

    monkeypatch.setattr(
        state_executor,
        "next_state",
        lambda *a, **k: (order.append("route"), _Decision())[1],
    )

    res = F.handle_finalize(ctx, _Snap())

    assert res.action == "continue"
    assert order == [
        "o5", "perf", "critic", "eligibility", "static",
        "promotion", "merge", "route",
    ]


def test_finalize_promotion_errors_block_merge_and_done(monkeypatch, tmp_path):
    """A partial archive must never be routed to pipeline_done."""
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(F, "_check_delivery_static_safety", lambda *_: None)
    monkeypatch.setattr(
        finalize_pipeline,
        "finalize_op",
        lambda op, workspace: finalize_pipeline.FinalizeReport(
            op=op,
            workspace=workspace,
            archive_dir=workspace / "archive",
            errors=["copy failed"],
        ),
    )
    calls = []
    monkeypatch.setattr(kb_invoke, "merge_one", lambda *_: calls.append("merge"))
    monkeypatch.setattr(state_executor, "next_state", lambda *_args, **_kwargs: calls.append("route"))
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)

    result = getattr(F, "_promote_and_route")(ctx)

    # 2026-08-30 (PR13 WP-A / A.4.2): non-worker-fixable promotion errors park
    # at await_user_decision via a recorded FSM transition, not a bare exit 7.
    assert result.action == "continue"
    assert state_executor.current_state(ctx.workspace) == "await_user_decision"
    assert calls == []


def test_finalize_does_not_trust_entries_token_marker(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.workspace / "knowledge_update.md").write_text(
        "validated knowledge\n" * 20
    )
    (ctx.workspace / ".kb_merged").write_text("entries=3\n")
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
    monkeypatch.setattr(
        finalize_pipeline,
        "finalize_op",
        lambda op, workspace: finalize_pipeline.FinalizeReport(
            op=op,
            workspace=workspace,
            archive_dir=workspace / "archive",
        ),
    )
    calls = []
    monkeypatch.setattr(
        kb_invoke,
        "merge_one",
        lambda *_: (calls.append("merge"), {"success": True})[1],
    )

    class _Decision:
        from_state = "finalize"
        next_state = "done"
        rationale = "validated"

    monkeypatch.setattr(
        state_executor,
        "next_state",
        lambda *a, **k: _Decision(),
    )

    res = getattr(F, '_promote_and_route')(ctx)

    assert res.action == "continue"
    assert calls == ["merge"]
    assert not (ctx.workspace / ".kb_merged").exists()
    assert list(ctx.workspace.glob(".kb_merged.invalid-*"))


def test_o5_rollback_so_files_backed_up_not_unlinked(tmp_path, monkeypatch):
    """codex review F3 (2026-08-25): the O5 rollback cleanup mirrors
    cold-start — workspace *.so are MOVED to a backup dir (recoverable),
    never unlinked; npubench_evidence/ stays untouched.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "libkernel.so").write_text("binary\n")
    evidence = ws / "npubench_evidence"
    evidence.mkdir()
    receipt = evidence / "preflight_target_receipt.json"
    receipt.write_text("{}")
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("NPUBENCH_REPAIR_BACKUP_ROOT", str(backup_root))

    from fsm_phase_finalize import _clear_harness_build_artifacts

    removed = _clear_harness_build_artifacts(ws)

    assert "libkernel.so" in removed
    assert not (ws / "libkernel.so").exists()
    backups = list((backup_root / ws.name).glob("o5-rollback-*/libkernel.so"))
    assert len(backups) == 1
    assert backups[0].read_text() == "binary\n"
    assert receipt.is_file()


def test_host_copy_fault_escalation_counts_per_candidate_tree(tmp_path):
    """Same tree accumulates across evals (fresh binding each time); new tree resets."""
    tree_a, tree_b = "a" * 64, "b" * 64

    def o5_for(tree, binding):
        return _O5(
            "MISMATCH",
            measured={
                "precision": {
                    "binding_sha256": binding,
                    "reason": "copy_between_host_and_device failed",
                },
                "performance": {"evaluation_binding": {"candidate_tree_sha256": tree}},
            },
        )

    assert _bump_host_copy_fault_escalation(tmp_path, o5_for(tree_a, "1" * 64)) == 1
    # A fresh per-eval binding for the SAME tree must keep accumulating — the
    # pre-2026-08-27 per-binding keying stayed at 1 forever and never escalated.
    assert _bump_host_copy_fault_escalation(tmp_path, o5_for(tree_a, "2" * 64)) == 2
    # A re-authored candidate (new tree) restarts the count.
    assert _bump_host_copy_fault_escalation(tmp_path, o5_for(tree_b, "3" * 64)) == 1


def test_host_copy_fault_key_falls_back_to_precision_binding():
    o5 = _O5("MISMATCH", measured={"precision": {"binding_sha256": "c" * 64}})
    assert _o5_candidate_tree_key(o5) == "c" * 64
    assert _o5_candidate_tree_key(_O5("MISMATCH", measured={})) == "unknown"


def _write_durable_state(ws, state):
    (ws / ".opgen_state.json").write_text(json.dumps(state))


def _write_generic_kernel_project(ws):
    (ws / "kernel").mkdir(exist_ok=True)
    (ws / "kernel" / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
    (ws / "model_new_ascendc.py").write_text("# candidate entry\n")


def test_direct_npubench_candidate_failure_generic_kernel_route(tmp_path):
    """port_a3_to_a5 (no port_source) + npubench reference + generic kernel
    project must classify as a direct npubench candidate route so an
    authenticated candidate_contract build failure routes to the repair
    worker instead of burning infra retries (2026-08-27 flash_attention_score).
    """
    _write_durable_state(tmp_path, {"reference": {"source": "npubench"}})
    _write_generic_kernel_project(tmp_path)
    assert _is_direct_npubench_candidate_failure(tmp_path) is True


def test_direct_npubench_candidate_failure_generic_route_requires_project(tmp_path):
    """No kernel/ project on disk -> not a candidate route (fail closed)."""
    _write_durable_state(tmp_path, {"reference": {"source": "npubench"}})
    assert _is_direct_npubench_candidate_failure(tmp_path) is False


def test_direct_npubench_candidate_failure_generic_route_requires_npubench(tmp_path):
    """A non-npubench reference on the generic route stays out of the taxonomy."""
    _write_durable_state(tmp_path, {"reference": {"source": "torch_ref"}})
    _write_generic_kernel_project(tmp_path)
    assert _is_direct_npubench_candidate_failure(tmp_path) is False


def test_direct_npubench_candidate_failure_tilelang_route_unchanged(tmp_path):
    """The original TileLang2AscendC gate keeps classifying as before."""
    _write_durable_state(tmp_path, {
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
        "reference": {"source": "npubench"},
    })
    assert _is_direct_npubench_candidate_failure(tmp_path) is True
    # TileLang2AscendC kind without the npubench reference is still rejected.
    _write_durable_state(tmp_path, {
        "port_source": {"kind": "port-aclnn-tilelang2ascendc"},
        "reference": {"source": "torch_ref"},
    })
    assert _is_direct_npubench_candidate_failure(tmp_path) is False


def test_direct_npubench_candidate_failure_unknown_source_kind_rejected(tmp_path):
    """An unrecognized port_source kind never enters the candidate taxonomy."""
    _write_durable_state(tmp_path, {
        "port_source": {"kind": "some-future-kind"},
        "reference": {"source": "npubench"},
    })
    _write_generic_kernel_project(tmp_path)
    assert _is_direct_npubench_candidate_failure(tmp_path) is False
