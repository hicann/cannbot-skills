#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Harness-backend abstraction — the `Backend` interface + canonical `Envelope`.

Ratified in docs/design/HARNESS_BACKEND_ABSTRACTION_DESIGN.md (ea3def12), refined from the
CC-coupling survey (docs/design/CC_COUPLING_SURVEY.md). One implementation per harness
(CCBackend first — extracts the current command-runner coupling; other harness
adapters remain independent of the operator programming model.)

This module is PURE INTERFACE (no behavior) — adding it changes nothing until a site funnels
through it. Canonical gate behavior must stay stable
across the whole refactor.

**Boundary invariant (never violate):** a Backend only WIRES/adapts the harness (dispatch /
agent-format / hook-trigger / envelope). It NEVER re-implements or weakens a canonical gate —
gate/precision/provider check LOGIC stays in the orchestrator.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Envelope:
    """Canonical, backend-agnostic result of dispatching an agent/skill.

    REQUIRED (orchestrator control flow — retry/error/resume — may depend ONLY on these;
    every backend MUST populate them):
    """
    is_error: bool
    output_text: str
    api_error_status: Optional[int] = None   # 429 etc. — required for retry/quota control flow

    # OPTIONAL (telemetry / logs ONLY — NEVER control flow). A backend maps what it can, None else.
    # Using any of these for a retry/error/resume decision = abstraction leak (see design §Envelope).
    session_id: Optional[str] = None
    num_turns: Optional[int] = None
    total_cost_usd: Optional[float] = None
    permission_denials: list = field(default_factory=list)
    stop_reason: Optional[str] = None
    terminal_reason: Optional[str] = None
    duration_ms: Optional[int] = None
    raw_envelope: dict = field(default_factory=dict)

    # -- guard: the required fields are the ONLY control-flow surface --
    CONTROL_FLOW_FIELDS = ("is_error", "output_text", "api_error_status")

    # -- compat aliases (derived from canonical fields; backend-agnostic names) so existing
    #    consumers that read AgentResult-style `.success`/`.cost_usd` (telemetry/log only) keep
    #    working when a call site is funneled to return an Envelope. NOT new state.
    @property
    def success(self) -> bool:
        return not self.is_error

    @property
    def cost_usd(self):
        return self.total_cost_usd


class Backend(ABC):
    """One per harness (Claude Code / opencode / …). Swappable adapter layer ONLY."""

    name: str = "abstract"

    @abstractmethod
    def dispatch(self, target: str, prompt: str, *, kind: str = "agent",
                 settings: Optional[str] = None, session: Optional[str] = None,
                 timeout: Optional[float] = None) -> Envelope:
        """Invoke an agent (kind='agent') or skill (kind='skill'; e.g. kb_invoke --skill).
        Returns a normalized Envelope. This is the single funnel all ~7 invoke sites use.
        """

    @abstractmethod
    def normalize(self, raw: Any) -> Envelope:
        """Map a harness-native raw result → canonical Envelope (required + optional fields)."""

    @abstractmethod
    def format_agent(self, agent_def: dict) -> Any:
        """Render an a5ops agent-def to the harness-native agent form (+ tool-name map)."""

    @abstractmethod
    def wire_safety(self, checkers: list) -> Any:
        """Produce the harness TRIGGER-config that fires the canonical checkers (CC: a --settings
        file; opencode: a tool.execute.before plugin). Produces the trigger ONLY — check LOGIC
        stays canonical. Airtight backstop remains the orchestrator/bwrap sandbox.
        """

    @abstractmethod
    def resume(self, session_id: str, prompt: str) -> Envelope:
        """Resume a prior session (CC: --resume; opencode: --continue)."""

    # ---- RECOVERY coupling (survey finding #2 — a dispatch-only abstraction MISSES this) ----
    @abstractmethod
    def identify_cmd(self, cmd: str) -> bool:
        """Is `cmd` an orphaned process of THIS backend? (recover.py hardcoded 'claude --print')."""

    @abstractmethod
    def parse_op_from_cmd(self, cmd: str) -> Optional[str]:
        """Extract the op slug from a backend process cmdline (recover.py)."""
