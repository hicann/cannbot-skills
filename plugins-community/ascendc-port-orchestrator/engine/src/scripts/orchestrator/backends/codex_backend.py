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

"""Codex CLI backend plugin.

This is the harness adapter layer only. It deliberately does not translate the
canonical gates or reference-provider logic; those remain in orchestrator
Python. The first production use should be a small structured-output spike
before routing long-running op-gen workers through this backend.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .base import (
    Backend, Envelope, TranscriptSkills, format_backend_agent,
    normalize_backend_envelope,
)
from .skill_context import load_skill_context

_CODEX_EXEC_MARKERS = ("codex exec", "codex  exec")
_OP_SLUG_RE = re.compile(r"\b([a-z][a-z0-9_]*)-(?:kw|pp|ko|fo|ar|da|bs|td|tt|tpo|cl)-\d+")


class CodexBackend(Backend):
    name = "codex"

    def __init__(self, codex_bin: str | None = None):
        self.codex_bin = codex_bin or os.environ.get("AOG_CODEX_BIN", "codex")

    def dispatch(self, target: str, prompt: str, *, kind: str = "agent", mode: str = "foreground",
                 settings: Optional[str] = None, session: Optional[str] = None,
                 timeout: Optional[float] = None, sandbox_prefix: Optional[list] = None,
                 progress_callback=None, tee_path=None, silence_timeout: Optional[int] = None,
                 extra_args: Optional[list] = None, output_file=None, cwd=None,
                 permission_mode: str = "bypassPermissions", output_format: Optional[str] = "json",
                 stdin_prompt: bool = False) -> Envelope:
        if mode == "background":
            raise NotImplementedError("CodexBackend does not support background dispatch yet")
        if sandbox_prefix:
            raise NotImplementedError("CodexBackend does not support orchestrator bwrap sandbox_prefix yet")

        formatted = self._format_prompt(target, prompt, kind=kind)
        with tempfile.NamedTemporaryFile(prefix="aog-codex-last-", suffix=".txt", delete=False) as tmp:
            last_path = tmp.name
        try:
            cmd = self._build_exec_cmd(
                last_path,
                cwd=str(cwd) if cwd is not None else None,
                sandbox=os.environ.get("AOG_CODEX_SANDBOX"),
                extra_args=extra_args,
            )
            completed = subprocess.run(
                cmd,
                input=formatted,
                capture_output=True,
                text=True,
                timeout=int(timeout) if timeout else None,
            )
            if tee_path:
                Path(tee_path).parent.mkdir(parents=True, exist_ok=True)
                Path(tee_path).write_text(completed.stdout or "")
            output_text = ""
            last = Path(last_path)
            if last.exists():
                output_text = last.read_text(errors="replace")
            if not output_text:
                output_text = completed.stdout or ""
            return Envelope(
                is_error=completed.returncode != 0,
                output_text=output_text,
                raw_envelope={
                    "returncode": completed.returncode,
                    "stderr": completed.stderr or "",
                    "stdout": completed.stdout or "",
                    "cmd_kind": kind,
                    "backend": self.name,
                    "mode": mode,
                },
            )
        except subprocess.TimeoutExpired:
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={"timed_out": True, "backend": self.name, "cmd_kind": kind},
            )
        except FileNotFoundError as e:
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={"not_found": True, "stderr": str(e), "backend": self.name},
            )
        finally:
            try:
                Path(last_path).unlink()
            except OSError:
                pass

    def _build_exec_cmd(self, last_message_path: str, *, cwd: str | None = None,
                        sandbox: str | None = None, extra_args: Optional[list] = None) -> list:
        cmd = [self.codex_bin, "exec", "--json", "-o", last_message_path]
        if cwd:
            cmd += ["-C", cwd]
        if sandbox:
            cmd += ["--sandbox", sandbox]
        if extra_args:
            cmd += list(extra_args)
        cmd += ["-"]
        return cmd

    def _format_prompt(self, target: str, prompt: str, *, kind: str) -> str:
        skill_context = load_skill_context(target) if kind == "skill" else None
        skill_block = f"\n\n{skill_context}" if skill_context else ""
        return (
            f"You are running as the a5_ops harness backend target {target!r} "
            f"(kind={kind!r}) through Codex CLI.\n\n"
            "Follow the a5_ops orchestrator brief below. Return only the final "
            "worker/skill result text expected by the orchestrator."
            f"{skill_block}\n\n"
            "----- a5_ops prompt -----\n"
            f"{prompt}"
        )

    def normalize(self, raw: Any) -> Envelope:
        return normalize_backend_envelope(raw, self.name)

    def format_agent(self, agent_def: dict) -> dict:
        return format_backend_agent(agent_def, self.name)

    def wire_safety(self, checkers: list) -> None:
        return None

    def resume(self, session_id: str, prompt: str) -> Envelope:
        formatted = self._format_prompt("resume", prompt, kind="resume")
        with tempfile.NamedTemporaryFile(prefix="aog-codex-resume-last-", suffix=".txt", delete=False) as tmp:
            last_path = tmp.name
        try:
            cmd = [self.codex_bin, "exec", "resume", session_id, "--json", "-o", last_path, "-"]
            completed = subprocess.run(cmd, input=formatted, capture_output=True, text=True)
            output_text = Path(last_path).read_text(errors="replace") if Path(last_path).exists() else completed.stdout
            return Envelope(
                is_error=completed.returncode != 0,
                output_text=output_text or "",
                raw_envelope={
                    "returncode": completed.returncode,
                    "stderr": completed.stderr or "",
                    "stdout": completed.stdout or "",
                    "cmd_kind": "resume",
                    "backend": self.name,
                },
            )
        except FileNotFoundError as e:
            return Envelope(is_error=True, output_text="", raw_envelope={"not_found": True, "stderr": str(e)})
        finally:
            try:
                Path(last_path).unlink()
            except OSError:
                pass

    def transcript_skills(self, transcript_path) -> TranscriptSkills:
        """Report that Codex transcripts cannot prove tier-a skill invocation."""
        return TranscriptSkills(
            parseable=False,
            note="codex transcripts carry no skill-tool events — CBA route gate cannot prove tier-a USE",
        )

    def identify_cmd(self, cmd: str) -> bool:
        return any(marker in cmd for marker in _CODEX_EXEC_MARKERS)

    def parse_op_from_cmd(self, cmd: str) -> Optional[str]:
        m = _OP_SLUG_RE.search(cmd)
        return m.group(1) if m else None
