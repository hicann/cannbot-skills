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
"""opencode CLI backend plugin."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging

import os
import json
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from .base import Backend, Envelope
from .skill_context import load_skill_context

_OPENCODE_RUN_MARKERS = ("opencode run", "opencode  run")
_OP_SLUG_RE = re.compile(r"\b([a-z][a-z0-9_]*)-(?:kw|pp|ko|fo|ar|da|bs|td|tt|tpo|cl)-\d+")


@dataclass
class _StreamState:
    """Mutable state collected while consuming an opencode event stream."""

    last_output_at: float
    invalid_tool_limit: int
    raw_lines: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    session_id: str | None = None
    total_cost: float | None = None
    num_turns: int | None = None
    invalid_tool_count: int = 0
    invalid_tool_event: bool = False
    pending: str = ""


class OpencodeBackend(Backend):
    name = "opencode"

    def __init__(self, opencode_bin: str | None = None):
        self.opencode_bin = opencode_bin or os.environ.get("AOG_OPENCODE_BIN", "opencode")

    def dispatch(self, target: str, prompt: str, *, kind: str = "agent", mode: str = "foreground",
                 settings: Optional[str] = None, session: Optional[str] = None,
                 timeout: Optional[float] = None, sandbox_prefix: Optional[list] = None,
                 progress_callback=None, tee_path=None, silence_timeout: Optional[int] = None,
                 extra_args: Optional[list] = None, output_file=None, cwd=None,
                 permission_mode: str = "bypassPermissions", output_format: Optional[str] = "json",
                 stdin_prompt: bool = False) -> Envelope:
        if mode == "background":
            if output_file is None:
                raise ValueError("dispatch(mode='background') requires output_file")

        formatted = self._format_prompt(target, prompt, kind=kind)
        use_json_events = mode == "streaming" or (
            kind == "skill" and os.environ.get("AOG_OPENCODE_SKILL_FORMAT", "").lower() == "json"
        )
        cmd = self._build_run_cmd(
            session_id=session,
            auto=os.environ.get("AOG_OPENCODE_AUTO") == "1",
            cwd=str(cwd) if cwd is not None else None,
            extra_args=extra_args,
            format_json=use_json_events,
            model=self._select_model(target, kind),
            agent=self._select_agent(target, kind),
            variant=self._select_variant(target, kind),
        )
        if sandbox_prefix:
            cmd = list(sandbox_prefix) + cmd
        env = self._build_env(target, formatted)
        if mode == "background":
            return self._dispatch_background(cmd, formatted, env, output_file)
        if mode == "streaming" or use_json_events:
            return self._run_streaming(
                cmd,
                formatted,
                kind=kind,
                mode=mode,
                env=env,
                timeout=timeout,
                silence_timeout=silence_timeout,
                progress_callback=progress_callback,
                tee_path=tee_path,
            )
        return self._dispatch_foreground(cmd, formatted, env, timeout, tee_path, kind, mode)

    @staticmethod
    def _dispatch_background(cmd: list, formatted: str, env: dict, output_file) -> subprocess.Popen:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Popen duplicates the descriptor for the child; close the
        # parent-side handle as soon as the process has started.
        with out_path.open("w") as out_f:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=out_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        if proc.stdin is not None:
            proc.stdin.write(formatted)
            proc.stdin.close()
        return proc

    def _dispatch_foreground(self, cmd: list, formatted: str, env: dict, timeout: Optional[float], tee_path,
                             kind: str, mode: str) -> Envelope:
        try:
            completed = subprocess.run(
                cmd,
                input=formatted,
                capture_output=True,
                text=True,
                timeout=int(timeout) if timeout else None,
                env=env,
            )
            if tee_path:
                Path(tee_path).parent.mkdir(parents=True, exist_ok=True)
                Path(tee_path).write_text(completed.stdout or "")
            return Envelope(
                is_error=completed.returncode != 0,
                output_text=completed.stdout or "",
                raw_envelope={
                    "returncode": completed.returncode,
                    "stderr": completed.stderr or "",
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

    def _build_run_cmd(self, prompt: str | None = None, *, session_id: str | None = None,
                       auto: bool = False, cwd: str | None = None,
                       extra_args: Optional[list] = None, format_json: bool = False,
                       model: str | None = None, agent: str | None = None,
                       variant: str | None = None) -> list:
        # Backward-compatible wrapper for tests/callers from v3.13.0. The
        # prompt parameter is intentionally ignored: opencode run receives the
        # full a5_ops brief on stdin, not as a positional argv message.
        cmd = [self.opencode_bin, "run"]
        if session_id:
            cmd += ["--session", session_id]
        if auto:
            cmd += ["--auto"]
        if cwd:
            cmd += ["--dir", cwd]
        env_model = model if model is not None else os.environ.get("AOG_OPENCODE_MODEL")
        if env_model and not self._has_model_arg(extra_args or []):
            cmd += ["--model", env_model]
        env_agent = agent if agent is not None else os.environ.get("AOG_OPENCODE_AGENT")
        if env_agent and not self._has_agent_arg(extra_args or []):
            cmd += ["--agent", env_agent]
        env_variant = variant if variant is not None else os.environ.get("AOG_OPENCODE_VARIANT")
        if env_variant and not self._has_variant_arg(extra_args or []):
            cmd += ["--variant", env_variant]
        if format_json and not self._has_format_arg(extra_args or []):
            cmd += ["--format", "json"]
        if extra_args:
            cmd += list(extra_args)
        return cmd

    @staticmethod
    def _has_model_arg(extra_args: list) -> bool:
        return any(arg == "--model" or arg == "-m" or str(arg).startswith("--model=") for arg in extra_args)

    @staticmethod
    def _has_format_arg(extra_args: list) -> bool:
        return any(arg == "--format" or str(arg).startswith("--format=") for arg in extra_args)

    @staticmethod
    def _has_agent_arg(extra_args: list) -> bool:
        return any(arg == "--agent" or str(arg).startswith("--agent=") for arg in extra_args)

    @staticmethod
    def _has_variant_arg(extra_args: list) -> bool:
        return any(arg == "--variant" or str(arg).startswith("--variant=") for arg in extra_args)

    @staticmethod
    def _select_model(target: str, kind: str) -> str | None:
        return OpencodeBackend._select_env_value("MODEL", target, kind)

    @staticmethod
    def _select_agent(target: str, kind: str) -> str | None:
        return OpencodeBackend._select_env_value("AGENT", target, kind)

    @staticmethod
    def _select_variant(target: str, kind: str) -> str | None:
        return OpencodeBackend._select_env_value("VARIANT", target, kind)

    @staticmethod
    def _select_env_value(name: str, target: str, kind: str) -> str | None:
        safe_target = re.sub(r"[^A-Za-z0-9]+", "_", target).upper().strip("_")
        if safe_target:
            specific = os.environ.get(f"AOG_OPENCODE_{name}_{safe_target}")
            if specific:
                return specific
        kind_specific = os.environ.get(f"AOG_OPENCODE_{kind.upper()}_{name}")
        if kind_specific:
            return kind_specific
        return os.environ.get(f"AOG_OPENCODE_{name}")

    @staticmethod
    def _build_env(target: str, prompt: str | None = None) -> dict:
        env = os.environ.copy()
        env.setdefault("AOG_HOOK_AGENT_ID", f"opencode:{target}")
        env.setdefault("AOG_HOOK_AGENT_TYPE", target)
        workspace = OpencodeBackend._extract_workspace_from_prompt(prompt or "")
        if workspace:
            env.setdefault("ASCENDC_WORKSPACE", workspace)
            env.setdefault("CLAUDE_ACTIVE_WORKSPACE", workspace)
        return env

    @staticmethod
    def _extract_workspace_from_prompt(prompt: str) -> str | None:
        patterns = (
            r"\bASCENDC_WORKSPACE:\s*(/[^\s`]+)",
            r"\bworkspace\s*[:=]\s*(/[^\s`]+)",
            r"\bworkspace\s+(/[^\s`]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1).rstrip(".,)")
        return None

    def _run_streaming(self, cmd: list, prompt: str, *, kind: str, mode: str, env: dict,
                       timeout: Optional[float], silence_timeout: Optional[int] = None,
                       progress_callback=None,
                       tee_path=None) -> Envelope:
        started = time.monotonic()
        state, timeout_sec, deadline, silence_timeout_sec = self._new_stream_state(
            started, timeout, silence_timeout
        )
        tee = None
        proc: subprocess.Popen | None = None
        sel: selectors.BaseSelector | None = None
        try:
            tee = self._open_tee(tee_path)
            proc, sel, stdout_fd = self._open_stream_process(cmd, prompt, env)
            timed_out, silence_timed_out = self._monitor_stream(
                proc,
                sel,
                stdout_fd,
                state,
                deadline=deadline,
                silence_timeout_sec=silence_timeout_sec,
                tee=tee,
                progress_callback=progress_callback,
            )
            if timed_out or silence_timed_out or state.invalid_tool_event:
                self._terminate_process_group(proc)
                return self._stream_failure_envelope(
                    cmd,
                    state,
                    proc,
                    started=started,
                    timeout_sec=timeout_sec,
                    silence_timeout_sec=silence_timeout_sec,
                    silence_timed_out=silence_timed_out,
                    kind=kind,
                    mode=mode,
                    tee=tee,
                )
            rc = proc.wait()
            return self._stream_success_envelope(cmd, state, rc, started, kind, mode)
        except FileNotFoundError as e:
            return Envelope(
                is_error=True,
                output_text="",
                raw_envelope={"not_found": True, "stderr": str(e), "backend": self.name},
            )
        finally:
            self._close_stream_resources(sel, tee)

    def _new_stream_state(self, started: float, timeout: Optional[float], silence_timeout: Optional[int]
                          ) -> tuple[_StreamState, int | None, float | None, int | None]:
        timeout_sec = int(timeout) if timeout else None
        return (
            _StreamState(last_output_at=started, invalid_tool_limit=self._invalid_tool_limit()),
            timeout_sec,
            started + timeout_sec if timeout_sec else None,
            self._stream_silence_timeout(silence_timeout),
        )

    @staticmethod
    def _open_tee(tee_path):
        if not tee_path:
            return None
        tee_file = Path(tee_path)
        tee_file.parent.mkdir(parents=True, exist_ok=True)
        return tee_file.open("w")

    @staticmethod
    def _close_stream_resources(selector: selectors.BaseSelector | None, tee) -> None:
        if selector is not None:
            selector.close()
        if tee:
            tee.close()

    @staticmethod
    def _open_stream_process(cmd: list, prompt: str, env: dict) -> tuple[subprocess.Popen, selectors.BaseSelector, int]:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            env=env,
            bufsize=0,
            start_new_session=True,
        )
        if proc.stdin is not None:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        selector = selectors.DefaultSelector()
        if proc.stdout is None:
            raise RuntimeError("opencode backend did not provide a stdout pipe")
        stdout_fd = proc.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        selector.register(proc.stdout, selectors.EVENT_READ)
        return proc, selector, stdout_fd

    def _monitor_stream(self, proc: subprocess.Popen, selector: selectors.BaseSelector, stdout_fd: int,
                        state: _StreamState, *, deadline: float | None, silence_timeout_sec: int | None,
                        tee, progress_callback) -> tuple[bool, bool]:
        while True:
            timed_out, silence_timed_out = self._stream_timeout_state(
                state.last_output_at, deadline, silence_timeout_sec
            )
            if timed_out or silence_timed_out:
                return timed_out, silence_timed_out
            if proc.poll() is not None:
                self._drain_stream_stdout(state, stdout_fd, tee, progress_callback)
                if state.pending and not state.invalid_tool_event:
                    self._record_stream_line(state, state.pending, tee, progress_callback)
                    state.pending = ""
                return False, False
            if selector.select(self._stream_wait_interval(state.last_output_at, deadline, silence_timeout_sec)):
                self._drain_stream_stdout(state, stdout_fd, tee, progress_callback)
                if state.invalid_tool_event:
                    return False, False

    @staticmethod
    def _stream_timeout_state(last_output_at: float, deadline: float | None,
                              silence_timeout_sec: int | None) -> tuple[bool, bool]:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return True, False
        if silence_timeout_sec is not None and now - last_output_at >= silence_timeout_sec:
            return False, True
        return False, False

    @staticmethod
    def _stream_wait_interval(last_output_at: float, deadline: float | None,
                              silence_timeout_sec: int | None) -> float:
        wait = 0.5
        now = time.monotonic()
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - now))
        if silence_timeout_sec is not None:
            wait = min(wait, max(0.0, silence_timeout_sec - (now - last_output_at)))
        return wait

    def _drain_stream_stdout(self, state: _StreamState, stdout_fd: int, tee, progress_callback) -> None:
        while True:
            try:
                chunk = os.read(stdout_fd, 65536)
            except BlockingIOError:
                return
            if not chunk:
                return
            self._process_stream_chunk(state, chunk, tee, progress_callback)
            if state.invalid_tool_event:
                return

    def _process_stream_chunk(self, state: _StreamState, chunk: bytes, tee, progress_callback) -> None:
        state.pending += chunk.decode("utf-8", errors="replace")
        while "\n" in state.pending:
            line, state.pending = state.pending.split("\n", 1)
            self._record_stream_line(state, line + "\n", tee, progress_callback)
            if state.invalid_tool_event:
                return

    def _record_stream_line(self, state: _StreamState, line: str, tee, progress_callback) -> None:
        state.last_output_at = time.monotonic()
        session_id, cost, turns, invalid = self._consume_stream_line(
            line, state.raw_lines, state.text_parts, tee, progress_callback
        )
        state.session_id = session_id or state.session_id
        state.total_cost = cost if cost is not None else state.total_cost
        state.num_turns = turns if turns is not None else state.num_turns
        if invalid:
            state.invalid_tool_count += 1
            state.invalid_tool_event = state.invalid_tool_count >= state.invalid_tool_limit

    def _stream_failure_envelope(self, cmd: list, state: _StreamState, proc: subprocess.Popen, *,
                                 started: float, timeout_sec: int | None, silence_timeout_sec: int | None,
                                 silence_timed_out: bool, kind: str, mode: str, tee) -> Envelope:
        duration_ms = int((time.monotonic() - started) * 1000)
        output_text = self._joined_output(state.text_parts)
        raw_tail = "".join(state.raw_lines)[-4000:]
        if tee:
            tee.write(
                json.dumps({
                    "type": "a5_ops_timeout" if not state.invalid_tool_event else "a5_ops_invalid_tool_event",
                    "timeout_sec": timeout_sec,
                    "silence_timeout_sec": silence_timeout_sec,
                    "invalid_tool_event": state.invalid_tool_event,
                    "partial_output_tail": output_text[-1000:],
                }) + "\n"
            )
        return Envelope(
            is_error=True,
            output_text=output_text,
            session_id=state.session_id,
            duration_ms=duration_ms,
            raw_envelope={
                "timed_out": True,
                "timeout_sec": timeout_sec,
                "silence_timed_out": silence_timed_out,
                "silence_timeout_sec": silence_timeout_sec,
                "invalid_tool_event": state.invalid_tool_event,
                "invalid_tool_limit": state.invalid_tool_limit,
                "backend": self.name,
                "cmd_kind": kind,
                "mode": mode,
                "event_format": "json",
                "returncode": proc.returncode,
                "stdout_tail": raw_tail,
                "cmd": self._redacted_cmd(cmd),
            },
        )

    def _stream_success_envelope(self, cmd: list, state: _StreamState, returncode: int, started: float,
                                 kind: str, mode: str) -> Envelope:
        output_text = self._joined_output(state.text_parts) or "".join(state.raw_lines)
        return Envelope(
            is_error=returncode != 0,
            output_text=output_text,
            session_id=state.session_id,
            total_cost_usd=state.total_cost,
            num_turns=state.num_turns,
            duration_ms=int((time.monotonic() - started) * 1000),
            raw_envelope={
                "returncode": returncode,
                "stdout_tail": "".join(state.raw_lines)[-4000:],
                "cmd_kind": kind,
                "backend": self.name,
                "mode": mode,
                "event_format": "json",
                "cmd": self._redacted_cmd(cmd),
            },
        )

    def _consume_stream_line(self, line: str, raw_lines: list[str], text_parts: list[str],
                             tee, progress_callback) -> tuple[str | None, float | None, int | None, bool]:
        raw_lines.append(line)
        if len(raw_lines) > 1000:
            del raw_lines[: len(raw_lines) - 1000]
        if tee:
            tee.write(line)
            tee.flush()
        session_id = None
        total_cost = None
        num_turns = None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None, None, None, False
        session_id = event.get("sessionID") or event.get("session_id")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if event.get("type") == "text" or part.get("type") == "text":
            text = part.get("text") or event.get("text") or ""
            if text:
                text_parts.append(text)
                self._emit_text_progress(progress_callback, text)
        tool_name = (
            part.get("tool")
            or part.get("toolName")
            or part.get("name")
            or event.get("tool")
            or event.get("toolName")
        )
        if tool_name:
            tool_input = part.get("input") or part.get("args") or event.get("input") or {}
            self._emit_tool_progress(progress_callback, str(tool_name), tool_input)
        invalid_tool = str(tool_name).lower() == "invalid"
        if event.get("type") == "step_finish" or part.get("type") == "step-finish":
            if isinstance(part.get("cost"), (int, float)):
                total_cost = float(part["cost"])
            tokens = part.get("tokens")
            if isinstance(tokens, dict) and isinstance(tokens.get("total"), int):
                num_turns = tokens["total"]
        return session_id, total_cost, num_turns, invalid_tool

    @staticmethod
    def _stream_silence_timeout(silence_timeout: Optional[int]) -> int | None:
        value = silence_timeout
        if value is None:
            env_value = os.environ.get("AOG_OPENCODE_STREAM_SILENCE_TIMEOUT_SEC")
            if env_value:
                try:
                    value = int(env_value)
                except ValueError:
                    value = None
        if value is None or value <= 0:
            return None
        return int(value)

    @staticmethod
    def _invalid_tool_limit() -> int:
        raw = os.environ.get("AOG_OPENCODE_INVALID_TOOL_LIMIT", "1")
        try:
            value = int(raw)
        except ValueError:
            value = 1
        return max(1, value)

    @staticmethod
    def _joined_output(parts: list[str]) -> str:
        return "\n".join(part.strip("\n") for part in parts if part is not None)

    @staticmethod
    def _emit_text_progress(progress_callback, text: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    @staticmethod
    def _emit_tool_progress(progress_callback, name: str, tool_input: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input or {}}]},
            })
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

    @staticmethod
    def _redacted_cmd(cmd: list) -> list:
        return [str(part) for part in cmd]

    def _format_prompt(self, target: str, prompt: str, *, kind: str) -> str:
        skill_context = load_skill_context(target) if kind == "skill" else None
        skill_block = f"\n\n{skill_context}" if skill_context else ""
        guard_block = self._compatibility_guard(target, kind)
        return (
            f"You are running as the a5_ops harness backend target {target!r} "
            f"(kind={kind!r}) through opencode CLI.\n\n"
            "Follow the a5_ops orchestrator brief below. Return only the final "
            "worker/skill result text expected by the orchestrator."
            f"{guard_block}"
            f"{skill_block}\n\n"
            "----- a5_ops prompt -----\n"
            f"{prompt}"
        )

    @staticmethod
    def _compatibility_guard(target: str, kind: str) -> str:
        if kind != "agent":
            return ""
        runtime_guard = ""
        if target in {"aog-kernel-worker", "aog-kernel-optimizer", "aog-precision-probe"}:
            runtime_guard = OpencodeBackend._runtime_smoke_guard()
        if target == "aog-kernel-optimizer":
            return runtime_guard + (
                "\n\n=== opencode a5_ops kernel-optimizer compatibility guard ===\n"
                "- Optimize only the current `ASCENDC_WORKSPACE`. Do not read, "
                "grep, glob, copy, tar, or manually sync any other `workspace/<op>` "
                "directory or old remote `current_task` archive.\n"
                "- Do not hand-roll deployment with tar/scp/manual copies into "
                "`current_task`. Use the project wrapper from the local repo: "
                "`ASCENDC_WORKSPACE=<workspace/op> bash src/scripts/deploy_to_npu_lane.sh "
                "--lane <LANE> --build`. Keep its output direct; do not pipe it "
                "through `tail`/`head`/`grep`/`sed`/`awk` and do not append marker "
                "words such as `unpiped`.\n"
                "- After a build, run precision/perf from the deployed current_task "
                "using the runtime smoke guard above. If using `a5_exec.py`, pass "
                "only the in-container command; do not include `docker exec` in "
                "that command.\n"
                "- If a simple elementwise kernel is precision-correct but cannot "
                "meet the perf threshold after a bounded optimization attempt, "
                "write an honest `verification.json`/`PROGRESS.md` update and "
                "handoff with the measured ratio and reason. Do not keep retrying "
                "environment setup or replace the kernel with a host fallback.\n"
                "=== end opencode kernel-optimizer compatibility guard ==="
            )
        if target != "aog-kernel-worker":
            return runtime_guard
        return runtime_guard + OpencodeBackend._kernel_worker_guard()

    @staticmethod
    def _runtime_smoke_guard() -> str:
        return (
            "\n\n=== opencode a5_ops runtime smoke guard ===\n"
            "- When validating a direct-pybind AscendC build on the A5 remote "
            "container and the extension is compiled for Python 3.11, run the "
            "smoke with `/root/miniconda3/envs/py311/bin/python3.11`. Set "
            "`CANN_ROOT=/usr/local/Ascend/cann-9.1.T500`, "
            "`PYTHONPATH=$CANN_ROOT/python/site-packages`, "
            "`ASCEND_HOME_PATH=$CANN_ROOT`, and "
            "`ASCEND_OPP_PATH=$CANN_ROOT/opp`.\n"
            "- Use this `LD_LIBRARY_PATH` ordering for that smoke: "
            "`$CANN_ROOT/x86_64-linux/lib64:$CANN_ROOT/lib64:"
            "/usr/local/Ascend/driver/lib64/driver:"
            "/usr/local/Ascend/driver/lib64/common:"
            "/root/miniconda3/envs/py311/lib/python3.11/site-packages/torch/lib:"
            "/root/miniconda3/envs/py311/lib/python3.11/site-packages/torch_npu/lib:"
            "/root/miniconda3/envs/py311/lib:"
            "/usr/local/Ascend/8.5.0/x86_64-linux/lib64:$LD_LIBRARY_PATH`. "
            "Keep the 9.1.T500 runtime before the 8.5.0 fallback: 9.1.T500 "
            "provides `aclrtLaunchKernelWithHostArgs`; 8.5.0 is only a "
            "fallback provider for `libacl_dvpp.so`. Putting 8.5.0 first "
            "causes undefined-symbol failures.\n"
            "- `~/.claude/skills/a5_op/scripts/a5_exec.py` already executes "
            "inside the configured A5 container. Do not nest `docker exec` "
            "inside an `a5_exec.py` command. If using `a5_exec.py`, pass a "
            "plain in-container command such as `cd ... && source ... && "
            "export ... && /root/miniconda3/envs/py311/bin/python3.11 ...`. "
            "If using raw ssh, then and only then wrap the command in a "
            "single `docker exec <container> bash -c ...`.\n"
            "- Do not debug `libhccl.so`/`libtorch_python.so` by trying random "
            "activation commands or `conda run`. Use the exact CANN/PYTHONPATH/"
            "LD_LIBRARY_PATH order above; missing `libtorch_python.so` means "
            "the torch site-packages `torch/lib` directory was omitted from "
            "`LD_LIBRARY_PATH`.\n"
            "=== end opencode runtime smoke guard ==="
        )

    @staticmethod
    def _kernel_worker_guard() -> str:
        return "".join((
            OpencodeBackend._kernel_worker_scope_guard(),
            OpencodeBackend._kernel_worker_pybind_guard(),
            OpencodeBackend._kernel_worker_kernel_guard(),
            OpencodeBackend._kernel_worker_verification_guard(),
        ))

    @staticmethod
    def _kernel_worker_scope_guard() -> str:
        return (
            "\n\n=== opencode a5_ops kernel-worker compatibility guard ===\n"
            "- This is a real AscendC generation task. Do not satisfy precision "
            "with CPU/PyTorch/NumPy fallback code in model_new_ascendc.py, "
            "pybind11.cpp, kernels.cpp, or any runner. Pybind may only do "
            "metadata, allocation/contiguous conversion, launch, and sync; all "
            "operator arithmetic must be in AscendC kernel code.\n"
            "- For A3/V220 kernels, do not emit foreign-backend-looking "
            "syntax. Use the project AscendC surface: #include \"kernel_operator.h\", "
            "using namespace AscendC, and a bare extern \"C\" __global__ __aicore__ "
            "entry. Do not use OPENVINO_HIDDEN, __opencl__, KernelTensor, "
            "lowercase ascendc namespace, or includes such as \" ascendc/...\".\n"
            "- Prefer repo KB/templates, declared source, and SDK headers named by "
            "the brief. Direct raw archive scans are blocked; use only staged, "
            "provenance-tracked prior-art/prestage context, and never treat it as "
            "truth or replace the kernel with a host fallback.\n"
            "- Do not scan project-wide `output/` archives or copy answer-bearing "
            "runners such as archived `pass_a_runner.py`/`verification.json`. "
            "Inside the current workspace, the kernel worker owns code generation, "
            "one real build, runner generation, and verification artifact writing "
            "before a `done` handoff.\n"
            "- Do not read, grep, glob, or Bash-inspect any other `workspace/<op>` "
            "directory from previous runs. Use only the current `ASCENDC_WORKSPACE`, "
            "KB, SDK headers, and source inputs named by the brief.\n"
            "- Write direct-pybind/backward kernel sources under workspace/<op>/kernel/ "
            "(`kernel.h`, `kernels.cpp`, `pybind11.cpp`). `model_new_ascendc.py` "
            "belongs at workspace root and imports the built extension from "
            "kernel/build. The deploy wrapper syncs only the kernel/ subtree for "
            "kernel sources. `build_ascendc.py` does not auto-generate "
            "`pybind11.cpp`; missing `workspace/<op>/kernel/pybind11.cpp` is a "
            "hard build failure.\n"
        )

    @staticmethod
    def _kernel_worker_pybind_guard() -> str:
        return "".join((
            OpencodeBackend._kernel_worker_pybind_api_guard(),
            OpencodeBackend._kernel_worker_pybind_layout_guard(),
        ))

    @staticmethod
    def _kernel_worker_pybind_api_guard() -> str:
        return (
            "- For pybind, use the project A3 pattern: include the generated "
            "`torch/extension.h` and `torch_npu/csrc/core/npu/NPUStream.h`, "
            "never include nonexistent `torch/pybind.h`, declare the generated "
            "`extern \"C\" uint32_t aclrtlaunch_<kernel>(uint32_t blockDim, "
            "void* stream, ...)` stub or include the generated "
            "`aclrtlaunch_<kernel>.h`, get "
            "`auto stream = c10_npu::getCurrentNPUStream().stream(false)`, "
            "use `torch::Tensor` or `at::Tensor` function signatures, "
            "do not use device-side `__gm__` qualifiers in pybind host code "
            "or launch stub declarations, "
            "allocate the output tensor with `at::empty_like` or "
            "`torch::empty`, call `aclrtlaunch_<kernel>(blockDim, stream, ...)` "
            "or `ACLRT_LAUNCH_KERNEL(<kernel>)(blockDim, stream, ...)`, check "
            "the return code when using the explicit stub with a compilable "
            "status variable, e.g. `uint32_t ret = aclrtlaunch_...(...); "
            "TORCH_CHECK(ret == 0, \"aclrtlaunch failed\", ret);`, return the output "
            "tensor from `run_<op>` (not a `uint32_t` launch status), and expose "
            "`m.def(\"run_<op>\", &run_<op>)` from the "
            "literal `PYBIND11_MODULE(_<op>_ext, m)`. Do not expose "
            "`&ACLRT_LAUNCH_KERNEL(...)` directly in `m.def`. Do not call raw "
            "`aclrtLaunchKernel(...)` directly; do not "
            "include `torch_npu/csrc/aten/common/ACLRTLauchKernel.h`; do not "
            "use `py::tensor` or `py::object` for NPU tensors. "
            "For torch_npu device checks and tensor options, use "
            "`c10::DeviceType::PrivateUse1` / `at::kPrivateUse1`. Kernel tiling/workspace arguments "
            "are GM addresses: create NPU tensors for tiling/workspace and "
            "pass their `data_ptr` values; never pass a host stack array such "
            "as `uint64_t tiling[2]` or `reinterpret_cast<uint64_t>(tiling)`. "
            "pass the project launcher's expected blockDim/grid values exactly "
            "as the local examples do. Do not include nonexistent pybind "
            "headers such as `pybind11/strict_rcward.h`. Do not pass a host "
            "stack pointer such as `reinterpret_cast<GM_ADDR>(&tiling)` as "
            "kernel GM tiling/workspace. Keep `kernels.cpp` as kernel/source "
            "glue; put the `PYBIND11_MODULE` binding only in `pybind11.cpp`.\n"
        )

    @staticmethod
    def _kernel_worker_pybind_layout_guard() -> str:
        return (
            "- In `model_new_ascendc.py`, insert `workspace/<op>/kernel/build` "
            "into `sys.path`, import the literal pybind module "
            "`_<op>_ext` where `<op>` is the workspace directory basename "
            "(for example `_opencode_e2e_agent17_add_a3_ext`), and call its `run_<op>(...)` wrapper from "
            "`ModelNew.forward`. Do not use `from kernel import ...`.\n"
            "- For direct-pybind tasks, do not create PR4778 "
            "`op_host/`, `op_kernel/`, or Ascend950 config scaffolds unless the "
            "brief explicitly asks for a vendor OPP package. The verifier path "
            "uses `workspace/<op>/kernel/` plus `model_new_ascendc.py`.\n"
            "- In this direct-pybind path, do not mkdir, touch, read, or write "
            "`op_host/` or `op_kernel/`; those scaffolds are a different backend "
            "shape and the opencode hook will block them.\n"
            "- In the direct-pybind path, never write `kernel_module_t`, "
            "`KernelAddParams`, or OPP registration scaffolds. The local deploy "
            "wrapper builds the `workspace/<op>/kernel/` subtree directly.\n"
        )

    @staticmethod
    def _kernel_worker_kernel_guard() -> str:
        return "".join((
            OpencodeBackend._kernel_worker_memory_guard(),
            OpencodeBackend._kernel_worker_elementwise_guard(),
            OpencodeBackend._kernel_worker_partition_guard(),
        ))

    @staticmethod
    def _kernel_worker_memory_guard() -> str:
        return (
            "- In kernel code, keep GM base pointers from the entry arguments "
            "and add element offsets to those bases. Do not fabricate GM "
            "pointers from numeric offsets such as "
            "`reinterpret_cast<__gm__ float*>(offset)`.\n"
        )

    @staticmethod
    def _kernel_worker_elementwise_guard() -> str:
        return (
            "- For simple fp32 elementwise add, use the proven direct-pybind "
            "shape: `kernel/kernels.cpp` includes literal `\"kernel.h\"`; "
            "`kernel/kernels.cpp` must define the actual `extern \"C\" "
            "__global__ __aicore__` kernel body, not just declare it; "
            "`kernel.h` must not declare or define that extern kernel entry; "
            "`kernel.h` must use its own project-specific include guard such as "
            "`OP_ADD_KERNEL_H`, never `KERNEL_OPERATOR_H` or "
            "`__KERNEL_OPERATOR_H__` because that collides with the SDK header; "
            "`kernel.h` uses `using namespace AscendC`, `TPipe`, "
            "`TQue<QuePosition::VECIN,...>`, `GlobalTensor<float>`, and "
            "calls `SetGlobalBuffer(reinterpret_cast<__gm__ float*>(arg) + "
            "start, elems)` in `Init`. `TPipe::InitBuffer` calls must pass "
            "queue, depth, and a fixed tile byte-size such as "
            "`kTileElems * sizeof(float)`, never `totalElems * sizeof(float)` "
            "or another full-input dynamic size. The kernel entry must take "
            "`GM_ADDR` parameters, e.g. `extern \"C\" __global__ __aicore__ "
            "void add_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, int64_t total)`; "
            "cast to `__gm__` pointers only inside the operator `Init`. Keep "
            "`TPipe pipe_` as a member and call `pipe_.InitBuffer(...)`; do "
            "not use `GetTPipe()`. Write the complete "
            "DataCopy/Add/DataCopy tile loop over chunks. Generated C/C++ "
            "must be ASCII only and must not contain TODOs, ellipses, "
            "placeholder comments, or thinking text. `TPipe` only initializes "
            "queue buffers for this path; do not call "
            "`InitBuffer(xTbuf_, depth, bytes)` or mix unused `TBuf` scratch "
            "members with a TQue DataCopy/Add pipeline. `TPipe` does not have "
            "`EnQue`/`DeQue`. Use the sequence "
            "`AllocTensor` -> `DataCopy(LocalTensor, GlobalTensor, count)` -> "
            "`EnQue` -> `DeQue` -> `Add` -> `EnQue` -> `DeQue` -> "
            "`DataCopy(GlobalTensor, LocalTensor, count)` -> `FreeTensor`, "
            "with each `DeQue` stored in a LocalTensor variable exactly once. "
            "Pass the current tile count directly to vector intrinsics such as "
            "`Add(..., tileLen)` or `Add(..., remaining)`; do not invent "
            "`epilogue_len` or subtract an undeclared tail value from the count. "
            "Do not emit typo identifiers such as `yDequeonge`, and never call "
            "`EnQue` twice on the same LocalTensor without an intervening `DeQue`. Keep queues and tensors inside "
            "the kernel operator object, not as file-scope globals. Split work by scalar `GetBlockIdx()` "
            "and `GetBlockNum()`; do not invent `coreCoord_t`, `IN_QUE_NUM`, "
            "`OUT_QUE_NUM`, `pipe.Barrier()`, or `TQue::WaitAllDone()`. "
        )

    @staticmethod
    def _kernel_worker_partition_guard() -> str:
        return (
            "If DataCopy counts are rounded up to 8 fp32 elements, the block "
            "ranges themselves must be 8-aligned and non-overlapping. Do not "
            "compute `base = total / blockNum`, `count = base + remainder`, "
            "then round each block's `count` to 8 and copy at `start + offset`; "
            "also do not compute `blockSize = (total + blockNum - 1) / blockNum`, "
            "`start = blockIdx * blockSize`, then use `tileLenAligned` for "
            "`DataCopy(...[start + offset], ..., tileLenAligned)`. "
            "for small or tail cases this makes block0 write 0..7, block1 "
            "write 1..8, etc. Use one block for tiny inputs, or partition the "
            "padded element range into disjoint 8-aligned spans and only return "
            "the original prefix. "
            "When splitting fp32 DataCopy/Add work across blocks, do not use "
            "`perBlock = (total + blockNum - 1) / blockNum` directly with "
            "`blockDim=56`; DataCopy element counts must stay 8-element aligned "
            "for the normal vector path. For the simple add smoke either launch "
            "with `blockDim = 8` for the provided 8-aligned cases or round "
            "per-block/tile counts up to an 8-element boundary and handle the "
            "tail explicitly.\n"
            "- In `pybind11.cpp`, `aclrtlaunch_<kernel>` host stubs should use "
            "`void*` tensor `data_ptr` arguments for GM addresses, matching the "
            "generated stub signature. Do not declare GM address parameters as "
            "`uint64_t` and do not cast tensor data pointers through `uint64_t`.\n"
        )

    @staticmethod
    def _kernel_worker_verification_guard() -> str:
        return (
            "- Build with the project wrapper by setting/using ASCENDC_WORKSPACE "
            "and running `bash src/scripts/deploy_to_npu_lane.sh --lane <LANE> "
            "--build`. Do not pipe any `deploy_to_npu*.sh` output through "
            "`tail`/`head`/`grep`/`sed`/`awk`; that masks exit status, can hang "
            "post-build sync, and hides build markers. Do not invent unsupported "
            "flags such as `--workspace`, do not append unsupported marker words "
            "such as `unpiped`, and do not manually copy files into LOCAL_TASK.\n"
            "- Before reporting done, run the mandated pre-build checks and one "
            "real build with direct output, then generate and run the workspace-local "
            "`pass_a_runner.py` and `pass_b_runner.py` or set "
            "`precision.pass_b.status` to `N/A` with a non-empty reason when the "
            "mode has no independent Pass B. If you create `pass_b_runner.py`, "
            "then `verification.json.precision.pass_b.status` must be real "
            "`PASS` or `FAIL`, not `N/A`. For an `edge_dataset.pt` fixture, "
            "support the dict shape "
            "`{'cases': [{'inputs': {...}, 'outputs': {...}}, ...]}` as well "
            "as legacy list cases; use the first output tensor from `outputs` "
            "as the oracle CPU truth when the output key is not obvious. Do "
            "not silently replace that oracle by recomputing a lower-precision "
            "truth just to pass an edge case; if the oracle precision policy is "
            "incompatible with the kernel dtype, report the Pass B result "
            "honestly as FAIL with the observed max diff. Write `verification.json` with "
            "`precision`, `determinism`, and `performance`; include "
            "`performance.independent_re_measure` with either a real ratio/ran "
            "field or an explicit skipped reason. Run "
            "`python3 src/scripts/orchestrator/check_verification_schema.py "
            "workspace/<op>/verification.json` and fix any schema failure before "
            "the handoff. If the implementation uses host fallback, cannot build "
            "a real AscendC kernel, or cannot produce honest verification "
            "artifacts, report FAIL/handoff instead of done.\n"
            "=== end opencode compatibility guard ==="
        )

    def normalize(self, raw: Any) -> Envelope:
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
        return Envelope(is_error=False, output_text=str(raw), raw_envelope={"backend": self.name})

    def format_agent(self, agent_def: dict) -> dict:
        rendered = dict(agent_def)
        rendered["harness_backend"] = self.name
        return rendered

    def wire_safety(self, checkers: list) -> dict:
        plugin_path = Path(__file__).resolve().parents[3] / "opencode" / "a5_ops_hooks.mjs"
        return {
            "backend": self.name,
            "kind": "host-hook-plugin",
            "plugin_path": str(plugin_path),
            "events": [
                "tool.execute.before",
                "tool.execute.after",
                "permission.ask",
            ],
            "checkers": list(checkers or []),
        }

    def resume(self, session_id: str, prompt: str) -> Envelope:
        return self.dispatch("resume", prompt, kind="resume", session=session_id)

    def identify_cmd(self, cmd: str) -> bool:
        return any(marker in cmd for marker in _OPENCODE_RUN_MARKERS)

    def parse_op_from_cmd(self, cmd: str) -> Optional[str]:
        m = _OP_SLUG_RE.search(cmd)
        return m.group(1) if m else None
