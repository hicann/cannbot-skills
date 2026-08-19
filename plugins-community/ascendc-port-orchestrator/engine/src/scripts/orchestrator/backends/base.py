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

This module owns the interface and small backend-independent value transforms only.  It does
not launch a harness or implement a canonical gate; canonical gate behavior stays stable
across the whole refactor.

**Boundary invariant (never violate):** a Backend only WIRES/adapts the harness (dispatch /
agent-format / hook-trigger / envelope). It NEVER re-implements or weakens a canonical gate —
gate/precision/provider check LOGIC stays in the orchestrator.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── cross-backend stream-silence contract (G1) ──────────────────────────────────────────────
# Every streaming backend (Claude Code, opencode, ...) SIGTERMs its subprocess and signals
# "mid-work stdout silence" through the SAME exception so the FSM's respawn budget logic
# (fsm_phase_spawn.py) is harness-agnostic.  Fields are backend-normalized:
#   agent_type      — the dispatched agent name (backend maps what it has; may be the backend name)
#   silent_seconds  — observed silence duration before SIGTERM
#   last_event_type — last streamed event type when tracked, else None
STREAM_SILENCE_RETRY_MAX = int(os.environ.get("AOG_STREAM_SILENCE_RETRY_MAX", "2"))
# Same default for every streaming harness. A backend may offer a narrower
# explicit override, but `None` must still preserve this watchdog contract.
STREAM_SILENCE_TIMEOUT_SEC = int(os.environ.get("AOG_STREAM_SILENCE_TIMEOUT_SEC", "1800"))


class StreamSilenceTimeout(Exception):
    """Raised when a subprocess stdout stream has been silent longer than the
    backend's silence watchdog allows.  The subprocess has been SIGTERMed by
    the time this exception is raised.  The caller (orchestrator FSM) catches
    this distinctly from generic Exception to enable bounded auto-respawn.
    """

    def __init__(self, agent_type: str, silent_seconds: float,
                 last_event_type: Optional[str] = None, *, partial_output: str = "",
                 raw_envelope: Optional[dict] = None):
        self.agent_type = agent_type
        self.silent_seconds = silent_seconds
        self.last_event_type = last_event_type
        # Streaming callers that choose exception-based recovery still retain the
        # evidence already received before the watchdog fired.  Existing callers
        # only consume the first three fields, so this remains backward compatible.
        self.partial_output = partial_output
        self.raw_envelope = raw_envelope or {}
        super().__init__(
            f"{agent_type}: stdout silent for {silent_seconds:.0f}s "
            f"(last event type: {last_event_type or 'none'}); "
            f"SIGTERMed subprocess"
        )


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


def normalize_backend_envelope(raw: Any, backend_name: str) -> Envelope:
    """Apply the backend-independent Envelope normalization contract."""
    if isinstance(raw, Envelope):
        return raw
    if isinstance(raw, dict):
        return Envelope(
            is_error=bool(raw.get("is_error")),
            output_text=raw.get("result") or raw.get("output_text") or "",
            api_error_status=raw.get("api_error_status"),
            session_id=raw.get("session_id"),
            raw_envelope=raw,
        )
    return Envelope(
        is_error=False,
        output_text=str(raw),
        raw_envelope={"backend": backend_name},
    )


def format_backend_agent(agent_def: dict, backend_name: str) -> dict:
    """Return a backend-labelled copy of a harness-neutral agent definition."""
    rendered = dict(agent_def)
    rendered["harness_backend"] = backend_name
    return rendered


@dataclass
class TranscriptSkills:
    """Backend-parsed skill invocations from a dispatch transcript (G7 / CBA route gate).

    parseable=False means the backend could not even confirm its NATIVE transcript format
    — no claim of "skills missing" may be derived from that; the gate must surface an
    explicit BLOCKED note instead of a silent skip or a false missing list.
    """
    invoked: set = field(default_factory=set)
    # Names observed in a native but non-terminal tool event. The route gate
    # decides whether one intersects a required route; unrelated in-flight work
    # must not invalidate independently proven required use.
    unproven: set = field(default_factory=set)
    parseable: bool = True
    note: str = ""


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

    # ---- CBA transcript coupling (G7 — the tier-a route gate must parse BOTH formats) ----
    @abstractmethod
    def transcript_skills(self, transcript_path: Path) -> TranscriptSkills:
        """Parse a native transcript without raising for unreadable or foreign input."""

    # ---- RECOVERY coupling (survey finding #2 — a dispatch-only abstraction MISSES this) ----
    @abstractmethod
    def identify_cmd(self, cmd: str) -> bool:
        """Is `cmd` an orphaned process of THIS backend? (recover.py hardcoded 'claude --print')."""

    @abstractmethod
    def parse_op_from_cmd(self, cmd: str) -> Optional[str]:
        """Extract the op slug from a backend process cmdline (recover.py)."""
