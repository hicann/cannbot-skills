# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for fsm_context.OrchestratorContext (DEBT-201 dependency inversion).

The context is the seam that lets the run_single_op FSM phase-handlers reach
orchestrator-module-level deps WITHOUT a load-time `import orchestrator` — its
orchestrator-global accessors are READ-THROUGH (resolve `orchestrator.<name>`
on every access). These tests lock that read-through contract so the STEP-2
handler extraction can rely on `monkeypatch.setattr(orchestrator, X)` still
biting through `ctx.X`.

Run: cd src/scripts/orchestrator && PYTHONPATH=. python3 -m pytest \
     tests/ut/test_fsm_context.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # src/scripts/orchestrator

import orchestrator  # noqa: E402
from fsm_context import OrchestratorContext, _orch  # noqa: E402


# All orchestrator-module-level names the FSM loop reaches through the context.
_READ_THROUGH_NAMES = {
    "total_spawn_cap_per_op": "TOTAL_SPAWN_CAP_PER_OP",
    "workspace_root": "WORKSPACE_ROOT",
    "ensure_audit_artifacts": "_ensure_audit_artifacts",
    "generate_timing_report": "_generate_timing_report",
    "record_partial_persist_finalize": "_record_partial_persist_finalize",
    "is_legitimate_pipeline_exhaustion": "_is_legitimate_pipeline_exhaustion",
    "archive_stale_outputs_before_spawn": "_archive_stale_outputs_before_spawn",
    "mark_agent_died": "_mark_agent_died",
    "load_silence_retry_count": "_load_silence_retry_count",
    "bump_silence_retry_count": "_bump_silence_retry_count",
    "extract_canonical_handoff": "extract_canonical_handoff",
    "canonical_handoff_prefixes": "_CANONICAL_HANDOFF_PREFIXES",
    "consume_applied_user_decision": "_consume_applied_user_decision",
    "extract_kb_draft_from_user_decision": "_extract_kb_draft_from_user_decision",
    "resolve_env": "_resolve_env",
    "agent_timeout_for_target": "_agent_timeout_for_target",
}


def _ctx() -> OrchestratorContext:
    return OrchestratorContext(op="x", workspace=Path("/tmp/x"), lane=0)


def test_orch_resolves_to_the_live_orchestrator_module():
    assert _orch() is orchestrator


def test_all_read_through_accessors_resolve_to_live_orchestrator_attrs():
    ctx = _ctx()
    for context_name, module_name in _READ_THROUGH_NAMES.items():
        assert getattr(ctx, context_name) is getattr(orchestrator, module_name), context_name


def test_scalar_global_read_through_bites_monkeypatch(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(orchestrator, "TOTAL_SPAWN_CAP_PER_OP", 7)
    assert ctx.total_spawn_cap_per_op == 7  # read at access time, not construction


def test_callable_global_read_through_bites_monkeypatch(monkeypatch):
    ctx = _ctx()
    sentinel = object()
    monkeypatch.setattr(orchestrator, "_mark_agent_died", lambda *a, **k: sentinel)
    assert ctx.mark_agent_died() is sentinel


def test_invariant_and_mutable_fields_are_plain_attributes():
    ctx = OrchestratorContext(
        op="addmm", workspace=Path("/w"), lane=2, plan_only=True,
        timing=True, kw_1_only=True, extra_lanes=[3, 4],
        runtime_kwargs={"force_x": True},
        spawn_count=5, last_handoff="@x", lifetime_spawn_count=9,
    )
    assert ctx.op == "addmm" and ctx.lane == 2
    assert ctx.plan_only and ctx.timing and ctx.kw_1_only
    assert ctx.extra_lanes == [3, 4]
    assert ctx.runtime_kwargs == {"force_x": True}
    assert (ctx.spawn_count, ctx.last_handoff, ctx.lifetime_spawn_count) == (5, "@x", 9)
    # mutable loop state is writable
    ctx.spawn_count += 1
    ctx.last_handoff = "@y"
    assert (ctx.spawn_count, ctx.last_handoff) == (6, "@y")
