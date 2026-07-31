# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""fsm_context.py — OrchestratorContext for the run_single_op FSM decomposition.

DEBT-201 god-function split (dependency inversion). `run_single_op` in
orchestrator.py is the 1792-line FSM DRIVER. Its phase-loop references a dozen
orchestrator-MODULE-LEVEL names that tests monkeypatch through the orchestrator
module (`monkeypatch.setattr(orchestrator, X, ...)`, plus the
`monkeypatch.setitem(sys.modules, "orchestrator", <alias>)` launch aliasing that
orchestrator_cmds documents). To extract the loop's phase-handlers into focused
sibling modules WITHOUT breaking that latent monkeypatch coupling, the handlers
must NOT `import orchestrator` and call `orchestrator.X` at module load time —
that binds a stale reference. Instead they receive an `OrchestratorContext` whose
orchestrator-global accessors are READ-THROUGH: each resolves the live
`orchestrator.<name>` at ATTRIBUTE-ACCESS time via `_orch()` (the same
sys.modules lookup orchestrator_cmds uses), so a test that patches
`orchestrator.TOTAL_SPAWN_CAP_PER_OP` (or any re-exported helper) STILL BITES
through the context.

Design contract (why this is behavior-neutral):
- Per-run INVARIANT inputs (op / workspace / lane / flags / runtime_kwargs) are
  plain attributes set once at construction.
- MUTABLE loop state (spawn_count / last_handoff / lifetime_spawn_count) lives on
  the context so extracted handlers can read+advance it.
- ORCHESTRATOR-GLOBAL seams (the monkeypatch surface) are read-through
  @property accessors resolving `_orch().<name>` on every access.
- SIBLING modules (state_executor, events, finalize_pipeline, agent_dispatch, …)
  are NOT proxied here — handlers import those directly; patching the sibling
  module object bites the handler's direct reference with no indirection needed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Sentinel attribute identifying the orchestrator MODULE among sys.modules
# candidates (the `orchestrator` package __init__ does not define it; only
# orchestrator.py does). Mirrors orchestrator_cmds._ORCH_MARKER.
_ORCH_MARKER = "run_single_op"


def _orch():
    """Live handle to the orchestrator MODULE, resolved lazily at CALL time.

    Resolved via `sys.modules` (NOT a module-scope `import orchestrator`, which
    would bind a fixed reference and defeat monkeypatch-through-module). Because
    every orchestrator-global accessor below calls this on each access,
    `monkeypatch.setattr(orchestrator, "<name>", ...)` and
    `monkeypatch.setitem(sys.modules, "orchestrator", <alias>)` both stay
    effective. orchestrator.py can appear under several names by launch path
    (`orchestrator`, `orchestrator.orchestrator`, `__main__`); pick the first
    candidate that actually carries the orchestrator globals.
    """
    for _name in ("orchestrator", "orchestrator.orchestrator", "__main__"):
        _m = sys.modules.get(_name)
        if _m is not None and hasattr(_m, _ORCH_MARKER):
            return _m
    return sys.modules.get("orchestrator") or sys.modules["__main__"]


@dataclass
class HandlerResult:
    """Control-flow signal returned by an extracted FSM phase-handler.

    The run_single_op loop drives handlers and acts on this:
      - action="continue" → `continue` the while loop (re-snapshot next iter)
      - action="return"    → `return exit_code` from run_single_op
    This preserves the original `continue` / `return N` control-flow verbatim
    while letting the slice live in a sibling module.
    """
    action: str  # "continue" | "return"
    exit_code: Optional[int] = None

    @staticmethod
    def cont() -> "HandlerResult":
        return HandlerResult(action="continue")

    @staticmethod
    def ret(code: int) -> "HandlerResult":
        return HandlerResult(action="return", exit_code=code)


@dataclass
class OrchestratorContext:
    """Per-run context threaded through the extracted FSM phase-handlers.

    Constructed once in `run_single_op` after the workspace is resolved; the
    loop and its handlers read invariant inputs + mutate loop state through it,
    and reach the monkeypatch-surface orchestrator globals via the read-through
    accessors below.
    """

    # --- per-run invariant inputs (set once) -------------------------------
    op: str
    workspace: Path
    lane: int = 0
    plan_only: bool = False
    timing: bool = False
    kw_1_only: bool = False
    extra_lanes: list[int] = field(default_factory=list)
    runtime_kwargs: dict[str, Any] = field(default_factory=dict)

    # --- mutable loop state (advanced by handlers) -------------------------
    spawn_count: int = 0
    last_handoff: str = ""
    lifetime_spawn_count: int = 0

    # -----------------------------------------------------------------------
    # Read-through orchestrator-global accessors (the monkeypatch surface).
    # Each resolves the LIVE orchestrator module attribute on every access so
    # `monkeypatch.setattr(orchestrator, <name>, ...)` bites through the ctx.
    # -----------------------------------------------------------------------
    @property
    def total_spawn_cap_per_op(self) -> int:
        return getattr(_orch(), "TOTAL_SPAWN_CAP_PER_OP")

    @property
    def workspace_root(self) -> Path:
        return getattr(_orch(), "WORKSPACE_ROOT")

    # Module-level helper functions / constants re-imported into orchestrator's
    # namespace and used by the FSM loop. Exposed as read-through callables so a
    # handler calls `ctx.<name>(...)` and a test patching `orchestrator.<name>`
    # is honored. (Attribute access returns the live function object.)
    @property
    def ensure_audit_artifacts(self):
        return getattr(_orch(), "_ensure_audit_artifacts")

    @property
    def generate_timing_report(self):
        return getattr(_orch(), "_generate_timing_report")

    @property
    def record_partial_persist_finalize(self):
        return getattr(_orch(), "_record_partial_persist_finalize")

    @property
    def is_legitimate_pipeline_exhaustion(self):
        return getattr(_orch(), "_is_legitimate_pipeline_exhaustion")

    @property
    def archive_stale_outputs_before_spawn(self):
        return getattr(_orch(), "_archive_stale_outputs_before_spawn")

    @property
    def mark_agent_died(self):
        return getattr(_orch(), "_mark_agent_died")

    @property
    def load_silence_retry_count(self):
        return getattr(_orch(), "_load_silence_retry_count")

    @property
    def bump_silence_retry_count(self):
        return getattr(_orch(), "_bump_silence_retry_count")

    @property
    def extract_canonical_handoff(self):
        return _orch().extract_canonical_handoff

    @property
    def canonical_handoff_prefixes(self):
        return getattr(_orch(), "_CANONICAL_HANDOFF_PREFIXES")

    @property
    def consume_applied_user_decision(self):
        return getattr(_orch(), "_consume_applied_user_decision")

    @property
    def extract_kb_draft_from_user_decision(self):
        return getattr(_orch(), "_extract_kb_draft_from_user_decision")

    @property
    def resolve_env(self):
        return getattr(_orch(), "_resolve_env")

    @property
    def agent_timeout_for_target(self):
        return getattr(_orch(), "_agent_timeout_for_target")
