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

"""CCBackend — the first `Backend` plugin: a5ops's existing Claude-Code coupling behind the
interface. FAITHFUL WRAPPER — it delegates to the existing canonical code paths
(`agent_transport.spawn_agent_foreground`, recover's cmd-parsing) so behavior is unchanged;
the conformance oracle must stay bit-identical as sites funnel through this.

Step 2 of the refactor (additive: nothing funnels through CCBackend yet → oracle unaffected).
skill/resume are wired at their respective funnel steps (kb_invoke / resume site) to reuse the
exact existing logic rather than reimplement subprocess handling prematurely.
"""
from __future__ import annotations
import os
import re
from typing import Any, Optional

from .base import Backend, Envelope

# recover.py's identify/op-extract logic, mirrored so the RECOVERY coupling is backend-owned
# (survey finding #2). Kept identical to recover._classify_proc / _extract_op_from_cmd.
_CLAUDE_PRINT_MARKERS = ("claude --print", "claude  --print")
_OP_SLUG_RE = re.compile(r"\b([a-z][a-z0-9_]*)-(?:kw|pp|ko|fo|ar|da|bs)-\d+")


class CCBackend(Backend):
    name = "claude_code"

    # ---- dispatch (agent path faithfully delegates to the canonical transport) ----
    def dispatch(self, target: str, prompt: str, *, kind: str = "agent", mode: str = "foreground",
                 settings: Optional[str] = None, session: Optional[str] = None,
                 timeout: Optional[float] = None, sandbox_prefix: Optional[list] = None,
                 progress_callback=None, tee_path=None, silence_timeout: Optional[int] = None,
                 extra_args: Optional[list] = None, output_file=None, cwd=None,
                 permission_mode: str = "bypassPermissions", output_format: Optional[str] = "json",
                 stdin_prompt: bool = False) -> Any:
        # sandbox_prefix (main-ratified 2026-07-05): an OPAQUE orchestrator-built argv-prefix that the
        # backend VERBATIM-prepends, with ZERO bwrap knowledge (airtight stays orchestrator-owned, inv#2).
        # mode: foreground (blocking→Envelope) | streaming (blocking+progress→Envelope) |
        #       background (fire-and-forget→raw Popen handle, NOT an Envelope). streaming/background carry
        #       sandbox_prefix (agent_transport supports it there; foreground does not).
        if kind == "agent":
            import agent_transport as _at  # canonical CC run (owns cmd-build + bwrap-prefix + envelope parse)
            extra = list(extra_args or [])
            if settings:
                extra += ["--settings", settings]
            extra = extra or None
            tkw: dict = {"timeout_sec": int(timeout)} if timeout is not None else {}
            if mode == "foreground":
                if sandbox_prefix:
                    raise NotImplementedError(
                        "foreground + sandbox_prefix unsupported by agent_transport.spawn_agent_foreground; "
                        "the graybox/bwrap path uses mode='streaming' or 'background'")
                return self.normalize(_at.spawn_agent_foreground(target, prompt, extra_args=extra, cwd=cwd, **tkw))
            if mode == "streaming":
                return self.normalize(_at.spawn_agent_streaming(
                    target, prompt, tee_path=tee_path, extra_args=extra, cwd=cwd,
                    progress_callback=progress_callback, silence_timeout_sec=silence_timeout,
                    sandbox_prefix=sandbox_prefix, **tkw))
            if mode == "background":
                if output_file is None:
                    raise ValueError("dispatch(mode='background') requires output_file")
                # fire-and-forget: returns the raw Popen handle (NOT an Envelope) — caller manages
                # lifecycle + reads output_file, exactly as spawn_for_state did pre-funnel.
                return _at.spawn_agent_background(target, prompt, output_file,
                                                  extra_args=extra, sandbox_prefix=sandbox_prefix)
            raise ValueError(f"unknown dispatch mode: {mode!r}")
        if kind == "skill":
            import subprocess
            cmd = self._build_skill_cmd(prompt, sandbox_prefix=sandbox_prefix,
                                        permission_mode=permission_mode,
                                        output_format=output_format, stdin_prompt=stdin_prompt)
            run_kw = {"capture_output": True, "text": True,
                      "timeout": int(timeout) if timeout else None}
            if stdin_prompt:
                run_kw["input"] = prompt  # divergent shape: prompt via stdin, not argv
            try:
                completed = subprocess.run(cmd, **run_kw)
            except subprocess.TimeoutExpired:
                return Envelope(is_error=True, output_text="",
                                raw_envelope={"timed_out": True, "returncode": None,
                                              "stderr": "(skill dispatch timed out)"})
            except FileNotFoundError as e:
                # faithful to sites that catch a missing CLI (phase_o17_classify) — surface, don't raise
                return Envelope(is_error=True, output_text="",
                                raw_envelope={"not_found": True, "returncode": None,
                                              "stderr": str(e)})
            return Envelope(
                is_error=(completed.returncode != 0),
                output_text=completed.stdout or "",
                raw_envelope={"returncode": completed.returncode,
                              "stderr": completed.stderr or "", "cmd_kind": "skill"},
            )
        raise ValueError(f"unknown dispatch kind: {kind!r}")

    @staticmethod
    def _build_skill_cmd(prompt: str, *, sandbox_prefix: Optional[list] = None,
                         permission_mode: str = "bypassPermissions",
                         output_format: Optional[str] = "json",
                         stdin_prompt: bool = False) -> list:
        """Faithful kb_invoke-style skill cmd (was hardcoded 'claude' → now CLAUDE_BIN-parameterized;
        sandbox_prefix opaque verbatim-prepend). Testable without spawning.

        Divergent-shape params (added for phase_o17_classify, which uses acceptEdits + plain-text
        output + prompt-via-stdin): `permission_mode` (default bypassPermissions), `output_format`
        (default 'json'; None → omit the flag entirely = plain text), `stdin_prompt` (default False =
        prompt in argv; True → prompt goes to subprocess stdin, so it is NOT appended to the cmd)."""
        base = [os.environ.get("CLAUDE_BIN", "claude"), "--print"]
        if output_format:
            base += ["--output-format", output_format]
        base += ["--permission-mode", permission_mode]
        if not stdin_prompt:
            base += [prompt]
        return list(sandbox_prefix or []) + base

    # ---- normalize: agent_transport.AgentResult (or raw envelope dict) -> canonical Envelope ----
    def normalize(self, raw: Any) -> Envelope:
        if isinstance(raw, dict):
            env = raw
            return Envelope(
                is_error=bool(env.get("is_error")),
                output_text=env.get("result") or env.get("output_text") or "",
                api_error_status=env.get("api_error_status"),
                session_id=env.get("session_id"),
                num_turns=env.get("num_turns"),
                total_cost_usd=env.get("total_cost_usd") or env.get("cost_usd"),
                permission_denials=env.get("permission_denials") or [],
                stop_reason=env.get("stop_reason"),
                terminal_reason=env.get("subtype") or env.get("terminal_reason"),
                duration_ms=env.get("duration_ms"),
                raw_envelope=env,
            )
        # agent_transport.AgentResult
        re_env = getattr(raw, "raw_envelope", None) or {}
        return Envelope(
            is_error=bool(getattr(raw, "is_error", True)),
            output_text=getattr(raw, "output_text", "") or "",
            api_error_status=re_env.get("api_error_status"),
            session_id=getattr(raw, "session_id", None),
            num_turns=re_env.get("num_turns"),
            total_cost_usd=getattr(raw, "cost_usd", None),
            permission_denials=re_env.get("permission_denials") or [],
            stop_reason=re_env.get("stop_reason"),
            terminal_reason=getattr(raw, "terminal_reason", None),
            duration_ms=getattr(raw, "duration_ms", None),
            raw_envelope=re_env,
        )

    # ---- agent-format: a5ops agents ARE already CC-native (Read/Write/... tool names) -> identity ----
    def format_agent(self, agent_def: dict) -> dict:
        return agent_def

    # ---- wire_safety: CC triggers canonical checkers via a --settings file (hooks); logic stays canonical
    def wire_safety(self, checkers: list) -> Optional[str]:
        # CC's trigger config is the settings.json hooks file already emitted by
        # agent_dispatch._build_extra_args (--settings). checkers are canonical Python, untouched.
        return None  # passthrough — CC hook wiring lives in the settings file, not synthesized here

    @staticmethod
    def _build_resume_cmd(session_id: str, prompt: str, *, output_format: Optional[str] = "json") -> list:
        """CC session-resume argv (`claude --resume <session_id> --print [--output-format json] <prompt>`).
        opencode mirrors this with `--continue`. Static + arg-free → unit-testable without spawning."""
        base = [os.environ.get("CLAUDE_BIN", "claude"), "--resume", session_id, "--print"]
        if output_format:
            base += ["--output-format", output_format]
        base += [prompt]
        return base

    def resume(self, session_id: str, prompt: str) -> Envelope:
        """Resume a prior CC session. NOTE (survey finding, 2026-07-05): a5ops has **no live caller**
        for harness session-resume — op-gen's `--resume` is the orchestrator's own state-machine resume
        (re-run from `.opgen_state.json`), independent of the harness; `session_id` captured at
        agent_dispatch is telemetry-only, never fed back. This method exists for interface-completeness
        (so OpencodeBackend can mirror `--continue`) and is exercised by unit test (cmd shape), NOT by a
        live-site conformance funnel. Behaviorally faithful to the skill-dispatch subprocess pattern.
        """
        import subprocess
        cmd = self._build_resume_cmd(session_id, prompt)
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as e:
            return Envelope(is_error=True, output_text="",
                            raw_envelope={"not_found": True, "returncode": None, "stderr": str(e)})
        return self.normalize({
            "is_error": completed.returncode != 0,
            "result": completed.stdout or "",
            "returncode": completed.returncode,
            "stderr": completed.stderr or "",
            "cmd_kind": "resume",
        })

    # ---- RECOVERY coupling (mirrors recover._classify_proc / _extract_op_from_cmd) ----
    def identify_cmd(self, cmd: str) -> bool:
        return any(m in cmd for m in _CLAUDE_PRINT_MARKERS)

    def parse_op_from_cmd(self, cmd: str) -> Optional[str]:
        m = _OP_SLUG_RE.search(cmd)
        return m.group(1) if m else None
