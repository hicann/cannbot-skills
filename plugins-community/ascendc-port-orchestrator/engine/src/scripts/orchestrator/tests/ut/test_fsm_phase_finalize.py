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


class _O5:
    def __init__(self, verdict, **kw):
        self.verdict = verdict
        self.claimed = kw.get("claimed", {})
        self.measured = kw.get("measured", {})
        self.mismatches = kw.get("mismatches", [])
        self.summary = kw.get("summary", "")
        self.rollback_kind = kw.get("rollback_kind", None)
        # DEBT-213(b): mirror O5Report's defaults exactly — the handler reads
        # these on the VERIFIED/PROVISIONAL paths, and a stub that silently
        # disagrees with the real dataclass is how a stub starts lying.
        self.harness_git_state = kw.get("harness_git_state", "UNKNOWN")
        self.harness_dirty = kw.get("harness_dirty", [])


@pytest.fixture
def stub_common(monkeypatch, tmp_path):
    """Neutralize the reload + emit + truth-source seams shared by every path."""
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    monkeypatch.setattr(events, "emit", lambda *a, **k: None)
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
    assert (res.action, res.exit_code) == ("return", 7)
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
