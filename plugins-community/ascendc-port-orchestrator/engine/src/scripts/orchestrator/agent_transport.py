# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Agent transport layer — spawn CC subagents from Python via claude CLI.

Validated 2026-05-04 by docs/design/CONTRACT_AND_MATURITY_NOTES.md#spike-cc-agent-transport:
- `claude --print --agent <agent_type> --output-format json "prompt"` returns
  a deterministic JSON envelope on stdout
- Hooks fire normally inside the spawned CC session (PreToolUse, PostToolUse,
  Stop, G1-G7, SC*)
- Both foreground (subprocess.run) and background (subprocess.Popen) work

Result envelope shape (from spike):
{
  "type": "result",
  "subtype": "success" | "error_*",
  "is_error": bool,
  "result": str,           # the agent's final text output
  "duration_ms": int,
  "duration_api_ms": int,
  "num_turns": int,
  "stop_reason": str,
  "session_id": str,
  "total_cost_usd": float,
  "usage": {...},
  "permission_denials": [...],
  "terminal_reason": "completed" | "timeout" | "error" | ...,
  "uuid": str,
}
"""
from __future__ import annotations
import logging

import json
import os
import select
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logging_config import get_logger  # cv-agent style logger
log = get_logger(__name__)


# P0m (2026-05-05): post-result-event hang.
# claude subprocess emits a `result` event on stdout, then exits — but its
# child processes (bun, node, playwright workers, ssh sessions) inherit the
# stdout pipe fd. After parent claude exits, those orphans keep the fd open;
# parent's `for raw in proc.stdout:` blocks forever waiting for EOF.
# Fix: after `result` event arrives, give a short grace window for natural
# exit, then SIGTERM the whole process group (start_new_session=True puts the
# child in its own session, killpg() reaches all descendents in one call).
RESULT_EVENT_GRACE_SEC = 15
PROCESS_GROUP_TERM_TIMEOUT_SEC = 5

# P0aal (2026-05-19, clipped_swiglu 9h-hang incident): the RESULT_EVENT_GRACE_SEC
# mechanism above had a dead-zone bug — the grace-elapsed check fired only when
# the NEXT stdout line arrived (i.e., inside `for raw in proc.stdout:` body).
# When the claude subprocess emitted `result` then went silent (do_epoll_wait
# state — 25 threads sleeping, telemetry/MCP/async-runtime cleanup never
# completing), no further stdout line ever arrived → the for-loop blocked
# forever inside readline. Fix: use select() with poll interval so the loop
# wakes every STREAM_POLL_INTERVAL_SEC even with no stdout activity, allowing
# the grace timer to be evaluated.
#
# Empirical: clipped_swiglu hung 9 hours under the pre-P0aal logic; with
# select()-based polling the same scenario should hit grace_sec + 1 poll
# interval ≈ 16 seconds.
STREAM_POLL_INTERVAL_SEC = 1.0

# P0aal-2 (2026-05-19, user direction same incident): mid-work stdout-silence
# watchdog. If the subprocess emits NO stream events for STREAM_SILENCE_TIMEOUT_SEC
# (default 10 min), SIGTERM the process group + raise StreamSilenceTimeout.
# Orchestrator main loop catches this exception, emits silence_killed event,
# increments a per-state silence-retry counter, and re-spawns up to
# STREAM_SILENCE_RETRY_MAX times before giving up.
#
# Failure modes this catches (NOT the P0aal-1 post-result-silence case):
#   - claude API call stuck on rate-limit retry
#   - subprocess wedged inside agent tool (long Bash, MCP server hang)
#   - infinite loop with no observable output
#   - lost network mid-call with no error surface
#
# Override via env: AOG_STREAM_SILENCE_TIMEOUT_SEC, AOG_STREAM_SILENCE_RETRY_MAX
STREAM_SILENCE_TIMEOUT_SEC = int(os.environ.get("AOG_STREAM_SILENCE_TIMEOUT_SEC", "1800"))
STREAM_SILENCE_RETRY_MAX = int(os.environ.get("AOG_STREAM_SILENCE_RETRY_MAX", "2"))


class StreamSilenceTimeout(Exception):
    """Raised when subprocess stdout has been silent for longer than
    STREAM_SILENCE_TIMEOUT_SEC. The subprocess has been SIGTERMed by
    the time this exception is raised. Caller (orchestrator) should
    catch this distinctly from generic Exception to enable bounded
    auto-respawn.
    """

    def __init__(self, agent_type: str, silent_seconds: float,
                 last_event_type: Optional[str] = None):
        self.agent_type = agent_type
        self.silent_seconds = silent_seconds
        self.last_event_type = last_event_type
        super().__init__(
            f"{agent_type}: stdout silent for {silent_seconds:.0f}s "
            f"(last event type: {last_event_type or 'none'}); "
            f"SIGTERMed subprocess"
        )


# P0n (2026-05-05): orchestrator-interrupt → claude-killed propagation.
# When orchestrator is killed (SIGTERM from CC, SIGINT from user, atexit),
# active claude subagent process groups MUST be reaped — otherwise they
# survive detached and continue running (consuming tokens, NPU lanes, etc.)
# until they hit their own timeout or external kill. Track active pgids in a
# module-level set; signal handlers + atexit drain it.
_ACTIVE_AGENT_PGIDS: set[int] = set()


def _reap_active_agents(reason: str = "exit") -> None:
    """Kill all tracked subagent process groups. Idempotent."""
    pgids = list(_ACTIVE_AGENT_PGIDS)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    # Brief escalation window before SIGKILL fallback
    if pgids:
        time.sleep(1)
        for pgid in pgids:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
    _ACTIVE_AGENT_PGIDS.clear()


_cleanup_handlers_installed = False


def _install_cleanup_handlers() -> None:
    """Install once-per-process: atexit + SIGTERM/SIGINT propagate to subagents."""
    global _cleanup_handlers_installed
    if _cleanup_handlers_installed:
        return
    import atexit
    atexit.register(_reap_active_agents, "atexit")

    def _handle_signal(signum: int, frame) -> None:  # noqa: ARG001
        _reap_active_agents(f"signal {signum}")
        # Re-raise default handler so process exits with conventional code
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            # Not in main thread / signal already set externally — best-effort
            pass

    _cleanup_handlers_installed = True


# P0k (Day 4 finding): Huawei firewall occasionally returns HTTP 200 with an
# empty or malformed body, surfacing as the following claude CLI error string.
# These are transient — wait + retry generally clears them.
_FW_TRANSIENT_PATTERNS = (
    "API returned an empty or malformed response",
    "check for a proxy or gateway intercepting the request",
    # 2026-05-27 (owner directive after 3_FusionAttention spawn): Huawei
    # internal corporate proxy occasionally emits this text as a stream-json
    # error event then claude exits 1. Re-spawn within seconds usually
    # succeeds. Pattern matches BOTH the prefix-only form
    # "API Error: corporate proxy Notification" (seen verbatim in worker spawn
    # 2026-05-26 17:43:43Z) and any future variant carrying the proxy name.
    "corporate proxy Notification",
    "API Error",
    # Add other firewall/network-transient patterns as observed
)
DEFAULT_FW_RETRY_WAIT_SEC = 30  # was 120; corporate proxy errors clear fast
DEFAULT_FW_MAX_RETRIES = 3


def _is_fw_transient_error(text: str) -> bool:
    """Return True if `text` (stderr / stdout / exception message) looks like
    a Huawei-firewall HTTP-200-empty-body kind of transient error."""
    if not text:
        return False
    return any(pat in text for pat in _FW_TRANSIENT_PATTERNS)


@dataclass
class AgentResult:
    agent_type: str
    success: bool
    is_error: bool
    output_text: str          # the agent's final stdout text (parsed from `result` field)
    duration_ms: int
    cost_usd: float
    session_id: str
    terminal_reason: str
    raw_envelope: dict        # full JSON envelope, for debugging
    tool_uses: list = None    # V2 #3: list of {tool_name, input, ts} when stream-json used
    progress_lines: list = None  # V2 #2: tail of mid-run text (last 50 lines)


class AgentTransportError(RuntimeError):
    """Raised when claude CLI itself fails (not when the agent reports an error)."""


def spawn_agent_foreground(
    agent_type: str,
    prompt: str,
    *,
    permission_mode: str = "bypassPermissions",
    timeout_sec: int = 3600,
    cwd: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
) -> AgentResult:
    """Spawn a CC subagent and BLOCK until completion.

    Args:
        agent_type: must match a name in `claude agents` (e.g. "aog-kernel-worker")
        prompt: the full prompt body (G7 slug should be encoded by caller)
        permission_mode: CC permission mode. Default "bypassPermissions" — workers
            need bash for deploy/build/ssh; "acceptEdits" only auto-approves file
            edits and would deny EVERY bash subprocess (Day 2 e2e finding 2026-05-04:
            33 Bash tool calls denied including `bash --version`). Override to a
            stricter mode only if the spawned agent is provably read-only.
        timeout_sec: hard wall-clock timeout (default 1h; increase for kernel-build runs)
        cwd: working directory for claude CLI (default: process cwd)
        extra_args: additional claude CLI flags (e.g. ["--add-dir", "/extra/path"])

    Returns:
        AgentResult parsed from the JSON envelope

    Raises:
        AgentTransportError: claude CLI failed to start or returned non-zero exit
        subprocess.TimeoutExpired: agent exceeded timeout_sec
        json.JSONDecodeError: claude returned non-JSON on stdout (unexpected)
    """
    # P0aau-c35.f (2026-05-10): feed prompt via stdin, NOT argv. Empirical finding
    # — 3.5K+ char prompts via argv cause claude --print to hang in epoll_wait
    # forever (zero stream output, claude alive at <1s CPU). 4140-char prompt via
    # stdin produces first stream events at +1.9s. Issue likely in claude's argv
    # handling for long prompts; stdin is the safe path for any non-trivial brief.
    cmd = [
        os.environ.get("CLAUDE_BIN", "claude"),
        "--print",
        "--agent", agent_type,
        "--output-format", "json",
        "--permission-mode", permission_mode,
    ]
    if extra_args:
        cmd.extend(extra_args)
    # NOTE: prompt is fed via stdin (input=prompt), NOT appended to cmd.

    completed = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=cwd,
    )

    if completed.returncode != 0:
        raise AgentTransportError(
            f"claude exited {completed.returncode} for agent={agent_type!r}\n"
            f"stderr: {completed.stderr[:2000]}"
        )

    envelope = json.loads(completed.stdout)
    return _parse_envelope(agent_type, envelope)


def spawn_agent_background(
    agent_type: str,
    prompt: str,
    output_file: Path,
    *,
    permission_mode: str = "bypassPermissions",
    cwd: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
    sandbox_prefix: Optional[list[str]] = None,
) -> subprocess.Popen:
    """Spawn a CC subagent in BACKGROUND. Returns the Popen handle.

    The caller is responsible for:
      - polling popen.poll() / popen.wait() / waitpid
      - reading output_file once popen exits
      - calling parse_background_result(output_file) to get AgentResult

    output_file will receive the JSON envelope on stdout (and stderr merged).

    Args:
        agent_type: must match a name in `claude agents`
        prompt: full prompt body
        output_file: where to redirect stdout+stderr
        permission_mode: CC permission mode
        cwd: working directory
        extra_args: additional claude flags

    Returns:
        subprocess.Popen handle (caller manages lifecycle)
    """
    # P0aau-c35.f: prompt via stdin to avoid argv-length hang.
    cmd = [
        os.environ.get("CLAUDE_BIN", "claude"),
        "--print",
        "--agent", agent_type,
        "--output-format", "json",
        "--permission-mode", permission_mode,
    ]
    if extra_args:
        cmd.extend(extra_args)

    # gap#2 airtight graybox isolation (a-fs): when sandbox_prefix is given, run the
    # whole claude-CLI spawn inside the bwrap mount-namespace it encodes. When None
    # (every non-graybox spawn), cmd is byte-identical to before — zero behavior change.
    if sandbox_prefix:
        cmd = list(sandbox_prefix) + cmd

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Popen duplicates the descriptor for the child, so the parent-side file
    # handle can be closed immediately after the process has started.
    with output_file.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
        )
    proc.stdin.write(prompt)
    proc.stdin.close()
    return proc


def parse_background_result(
    agent_type: str, output_file: Path
) -> AgentResult:
    """Parse the JSON envelope from a completed background spawn.

    Call this AFTER popen.wait() returns. Reads output_file, parses, returns
    AgentResult.
    """
    raw = output_file.read_text()
    if not raw.strip():
        raise AgentTransportError(
            f"empty output file from background agent {agent_type!r}: {output_file}"
        )
    envelope = json.loads(raw)
    return _parse_envelope(agent_type, envelope)


def _parse_envelope(agent_type: str, envelope: dict, *,
                     tool_uses: Optional[list] = None,
                     progress_lines: Optional[list] = None) -> AgentResult:
    return AgentResult(
        agent_type=agent_type,
        success=(envelope.get("subtype") == "success" and not envelope.get("is_error", False)),
        is_error=bool(envelope.get("is_error", False)),
        output_text=envelope.get("result", ""),
        duration_ms=int(envelope.get("duration_ms", 0)),
        cost_usd=float(envelope.get("total_cost_usd", 0.0)),
        session_id=envelope.get("session_id", ""),
        terminal_reason=envelope.get("terminal_reason", "unknown"),
        raw_envelope=envelope,
        tool_uses=tool_uses or [],
        progress_lines=progress_lines or [],
    )


# ---------------------------------------------------------------------------
# Streaming variant — V2 #2 + #3 (DEBT-077 Day 4)
# ---------------------------------------------------------------------------
def spawn_agent_streaming(
    agent_type: str,
    prompt: str,
    *,
    tee_path: Optional[Path] = None,
    permission_mode: str = "bypassPermissions",
    timeout_sec: int = 5400,
    cwd: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
    progress_callback: Optional[callable] = None,
    silence_timeout_sec: Optional[int] = None,
    sandbox_prefix: Optional[list[str]] = None,
) -> AgentResult:
    """Spawn a CC subagent with stream-json output for streaming progress
    AND per-tool-call trace.

    Args:
        agent_type: agent name (must match `claude agents`)
        prompt: full brief
        tee_path: if provided, raw stream-json appended line-by-line for forensics
        progress_callback: optional fn(event_dict) called per stream-json line
                           (lets caller print mid-run status, send Discord, etc.)
        timeout_sec: hard wall clock; defaults higher than non-streaming (90min)
                     since cold-start workers can run that long.
        silence_timeout_sec: optional override for the stream-silence watchdog
                     (P0aal-2). When None, uses module default
                     STREAM_SILENCE_TIMEOUT_SEC (currently 600s, env-
                     overridable). A caller may raise this for long build and
                     remote-execution cycles that legitimately produce silent
                     windows longer than the default.

    Returns:
        AgentResult with tool_uses[] + progress_lines[] populated.

    Raises:
        AgentTransportError on claude CLI non-zero exit OR no result event seen
        subprocess.TimeoutExpired on wall clock exceeded
    """
    # P0aau-c35.f: prompt via stdin to avoid argv-length hang.
    cmd = [
        os.environ.get("CLAUDE_BIN", "claude"),
        "--print",
        "--agent", agent_type,
        "--output-format", "stream-json",
        "--verbose",  # required by stream-json mode
        "--permission-mode", permission_mode,
    ]
    if extra_args:
        cmd.extend(extra_args)

    # gap#2 airtight graybox isolation (a-fs): prepend the bwrap mount-namespace prefix
    # when given. None (every non-graybox spawn) → cmd byte-identical to before.
    if sandbox_prefix:
        cmd = list(sandbox_prefix) + cmd

    # P0k (Day 4): wrap with FW transient retry. Non-FW errors propagate
    # immediately (no retry).
    return _run_streaming_with_fw_retry(
        cmd, agent_type, prompt=prompt, tee_path=tee_path,
        timeout_sec=timeout_sec, cwd=cwd,
        progress_callback=progress_callback,
        silence_timeout_sec=silence_timeout_sec,
    )


def _run_streaming_with_fw_retry(
    cmd: list[str],
    agent_type: str,
    *,
    prompt: str,  # P0aau-c35.f: fed via stdin, NOT argv (long argv hangs claude)
    tee_path: Optional[Path],
    timeout_sec: int,
    cwd: Optional[Path],
    progress_callback: Optional[callable],
    max_retries: int = DEFAULT_FW_MAX_RETRIES,
    wait_sec: int = DEFAULT_FW_RETRY_WAIT_SEC,
    sleep_fn=time.sleep,  # injectable for tests
    silence_timeout_sec: Optional[int] = None,
) -> AgentResult:
    """Wrapper around _run_streaming that retries on Huawei-firewall transient
    errors (HTTP 200 + empty/malformed body).

    Retry policy (P0k 2026-05-04):
      - On AgentTransportError whose message matches _FW_TRANSIENT_PATTERNS
        → sleep wait_sec, retry up to max_retries times
      - On any other AgentTransportError → propagate immediately (no retry —
        claude CLI exit code 1 with real error like quota / agent crash needs
        human attention, not silent retry)
      - On TimeoutExpired → propagate (caller wall-clock budget; retrying
        would burn another full timeout cycle)

    Attempts are 0-indexed; total attempts = max_retries + 1 (initial + N retries).
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return _run_streaming(
                cmd, agent_type, prompt=prompt,
                tee_path=tee_path, timeout_sec=timeout_sec,
                cwd=cwd, progress_callback=progress_callback,
                silence_timeout_sec=silence_timeout_sec,
            )
        except AgentTransportError as e:
            msg = str(e)
            if _is_fw_transient_error(msg) and attempt < max_retries:
                last_err = e
                log.info(
                    f"[transport] FW transient error on attempt {attempt+1}/"
                    f"{max_retries+1} for agent={agent_type!r}; sleeping "
                    f"{wait_sec}s before retry. Snippet: {msg[:200]}"
                )
                sleep_fn(wait_sec)
                continue
            raise  # non-FW error OR retries exhausted — propagate
    # Defensive: shouldn't reach here (loop either returns or raises)
    raise AgentTransportError(
        f"FW retry loop exhausted for agent={agent_type!r}: {last_err}"
    )


def _select_polling_lines(stdout, fd, proc_alive_cb, *,
                          poll_interval: float, tick_callback=None):
    """Generator that yields stdout lines OR `None` (poll tick) every
    `poll_interval` seconds. Allows the consumer to evaluate timers
    (grace, wall-clock) even when the subprocess goes silent — the
    fix for P0aal pre-2026-05-19 dead-zone bug.

    Yields:
        - str line (with trailing newline) when stdout has a line
        - None when poll_interval expired with no stdout activity
        - StopIteration when stdout closes (EOF) AND proc has exited

    Uses select() so we don't busy-spin. Reads one line at a time via
    readline() once select() reports the fd is readable.
    """
    buf = ""
    while True:
        ready, _, _ = select.select([fd], [], [], poll_interval)
        if not ready:
            # Poll tick — let caller check timers
            yield None
            # If proc exited AND no more data buffered, generator ends.
            if not proc_alive_cb() and not buf:
                # Drain remaining stdout one last time in case data
                # raced the exit; select would normally surface it.
                try:
                    remainder = stdout.read()
                except Exception:
                    remainder = ""
                if remainder:
                    for ln in remainder.splitlines(keepends=True):
                        yield ln
                return
            continue
        # fd readable — read one line. readline() blocks until \n or EOF
        # but select said data is ready, so this returns promptly.
        try:
            line = stdout.readline()
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            line = ""
        if not line:
            # EOF (subprocess closed stdout naturally — either exited or
            # killpg fired post-result). Drain any other state and return.
            return
        yield line


def _check_timers(result_event_seen_at, start, timeout_sec,
                  proc_alive_cb, killpg_cb, cmd):
    """Stub — main loop already handles timer checks. Kept as a hook for
    future extension (e.g., per-tick metrics). Returns nothing."""
    return None


def _run_streaming(
    cmd: list[str],
    agent_type: str,
    *,
    prompt: str,  # P0aau-c35.f: fed via stdin (avoids argv-length hang)
    tee_path: Optional[Path],
    timeout_sec: int,
    cwd: Optional[Path],
    progress_callback: Optional[callable],
    silence_timeout_sec: Optional[int] = None,
) -> AgentResult:
    """Run cmd with line-buffered stdout; parse stream-json events as they arrive.

    P0m (2026-05-05): start_new_session=True puts the claude child in its own
    process group/session. This serves two purposes:
      1. After we see the `result` event, we can `os.killpg(pgid, SIGTERM)` to
         reap any orphan grandchildren (bun/node/ssh) that inherited the stdout
         pipe and would otherwise prevent natural EOF, hanging the orchestrator
         indefinitely.
      2. If the orchestrator itself is killed (parent dies, SIGTERM/SIGINT
         from CC), the cleanup path can killpg() to ensure no zombie subagents
         continue running detached. (atexit handler in orchestrator main loop.)
    """
    # P0n: ensure cleanup handlers installed before spawning anything
    _install_cleanup_handlers()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,  # P0aau-c35.f: prompt fed via stdin
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=cwd,
        start_new_session=True,  # P0m: own process group for clean killpg
    )
    # Write prompt to stdin then close — claude reads to EOF then begins.
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except OSError:
        # Claude may exit before we can write (e.g. invalid args). Fall through
        # to normal stdout-read; error will surface as zero output → exit code.
        pass
    # Defensive: tests use FakePopen that doesn't have .pid; real Popen does.
    try:
        pgid = os.getpgid(proc.pid)
        _ACTIVE_AGENT_PGIDS.add(pgid)  # P0n: track for orchestrator-interrupt cleanup
    except (AttributeError, OSError):
        pgid = None

    def _killpg_safe(sig: int) -> None:
        if pgid is None:
            return
        try:
            os.killpg(pgid, sig)
        except OSError:
            pass

    def _proc_alive() -> bool:
        # FakePopen doesn't have poll(); treat it as exited (stream exhausted).
        try:
            return proc.poll() is None
        except AttributeError:
            return False

    final_envelope: Optional[dict] = None
    tool_uses: list[dict] = []
    progress_lines: list[str] = []
    result_event_seen_at: Optional[float] = None

    # P0aal-2 (2026-05-19): track last stream event timestamp for the
    # mid-work silence watchdog. Initialized to start time so the
    # subprocess gets the full timeout to emit its FIRST event.
    last_event_at: float = time.time()
    last_event_type: Optional[str] = None

    tee_f = open(tee_path, "w") if tee_path else None
    start = time.time()

    # P0aal (2026-05-19): use select()-based polling instead of blocking
    # `for raw in proc.stdout:` so the grace-timer + wall-clock checks fire
    # even when the subprocess goes silent post-`result` event.
    #
    # Tests use FakePopen.stdout which is an iterator, not a real fd —
    # detect via fileno() availability and fall through to the old path.
    try:
        stdout_fd = proc.stdout.fileno()
    except (AttributeError, OSError):
        stdout_fd = None

    try:
        if stdout_fd is None:
            # Test path / non-real-pipe fallback: keep old iterator behavior.
            # Tests can simulate the post-result hang by having FakePopen
            # signal exit via poll(); they don't exercise the dead-zone bug.
            line_source = iter(proc.stdout)
        else:
            line_source = _select_polling_lines(
                proc.stdout, stdout_fd, _proc_alive,
                poll_interval=STREAM_POLL_INTERVAL_SEC,
                tick_callback=lambda: _check_timers(
                    result_event_seen_at, start, timeout_sec,
                    _proc_alive, _killpg_safe, cmd,
                ),
            )

        for raw in line_source:
            # P0m: post-result grace check (now runs every iteration —
            # including synthetic POLL_TICK iterations from the select-based
            # source, so silence-after-result no longer hangs).
            if (result_event_seen_at is not None
                    and time.time() - result_event_seen_at > RESULT_EVENT_GRACE_SEC):
                if _proc_alive():
                    _killpg_safe(signal.SIGTERM)
                break

            # P0aal-2 (2026-05-19): mid-work stdout-silence watchdog. If
            # NO stream events for STREAM_SILENCE_TIMEOUT_SEC AND we haven't
            # yet seen the `result` event (post-result silence is handled
            # by the grace check above), SIGTERM + raise StreamSilenceTimeout.
            # Orchestrator catches this exception and decides whether to
            # respawn (up to STREAM_SILENCE_RETRY_MAX times).
            # silence_timeout_sec parameter (per-spawn override) wins over the
            # module-default STREAM_SILENCE_TIMEOUT_SEC. Callers that do not
            # override it inherit the 600s default.
            _silence_thresh = (
                silence_timeout_sec
                if silence_timeout_sec is not None
                else STREAM_SILENCE_TIMEOUT_SEC
            )
            if (result_event_seen_at is None
                    and time.time() - last_event_at > _silence_thresh):
                silent_sec = time.time() - last_event_at
                if _proc_alive():
                    _killpg_safe(signal.SIGTERM)
                raise StreamSilenceTimeout(
                    agent_type, silent_sec, last_event_type
                )

            # Wall-clock check (also lifted out of the line-receive path)
            if time.time() - start > timeout_sec:
                _killpg_safe(signal.SIGTERM)
                raise subprocess.TimeoutExpired(cmd, timeout_sec)

            # Polling tick — no actual line content; keep looping.
            if raw is None:
                continue

            # P0aal-2: any non-tick line resets the silence timer.
            last_event_at = time.time()

            # tee raw line for forensics
            if tee_f:
                tee_f.write(raw)
                tee_f.flush()

            # Parse JSON; non-JSON lines (rare) just pass through
            line = raw.strip()
            if not line:
                continue
            skip_current_item = False
            try:
                event = json.loads(line)
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue

            etype = event.get("type")
            last_event_type = etype  # P0aal-2: track for diagnostic on silence

            # Result envelope = last event before exit
            if etype == "result":
                final_envelope = event
                result_event_seen_at = time.time()
                if progress_callback:
                    try:
                        progress_callback(event)
                    except Exception as error:
                        logging.getLogger(__name__).debug(
                            "Recoverable operation failed.", exc_info=error
                        )
                continue

            # Tool-use events: trace
            if etype == "assistant":
                msg = event.get("message", {}) or {}
                content = msg.get("content", []) or []
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "tool_use":
                        tool_uses.append({
                            "ts_offset_s": round(time.time() - start, 2),
                            "tool_name": blk.get("name"),
                            "input": blk.get("input"),
                            "id": blk.get("id"),
                        })
                    elif blk.get("type") == "text":
                        text = blk.get("text", "")
                        if text:
                            for ln in text.splitlines():
                                progress_lines.append(ln)
                            # Keep tail
                            if len(progress_lines) > 50:
                                progress_lines = progress_lines[-50:]

            if progress_callback:
                try:
                    progress_callback(event)
                except Exception as error:
                    logging.getLogger(__name__).debug(
                        "Recoverable operation failed.", exc_info=error
                    )
    finally:
        if tee_f:
            tee_f.close()
        # Defensive: if we exit the loop for any reason and proc still alive,
        # killpg to prevent orphans from running detached after orchestrator exits.
        if _proc_alive():
            _killpg_safe(signal.SIGTERM)

    try:
        proc.wait(timeout=PROCESS_GROUP_TERM_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        # SIGTERM didn't take — escalate to SIGKILL on the group
        _killpg_safe(signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    # P0n: this pgid finished cleanly; remove from active set so the cleanup
    # handlers don't try to re-kill an already-dead group on orchestrator exit.
    if pgid is not None:
        _ACTIVE_AGENT_PGIDS.discard(pgid)

    if proc.returncode != 0:
        stderr_tail = ""
        try:
            stderr_tail = proc.stderr.read()[-2000:]  # type: ignore
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
        # 2026-05-27: include progress_lines tail in error message so
        # `_is_fw_transient_error` can detect stream-json error events
        # like "API Error: corporate proxy Notification" that surface ONLY via
        # progress_lines (claude exits 1 with empty stderr in that case).
        # Without this, FW-transient retry never triggers for corporate proxy
        # blips because the signature is invisible to the detector.
        progress_tail = "\n".join(progress_lines[-10:]) if progress_lines else ""
        raise AgentTransportError(
            f"claude (stream-json) exited {proc.returncode} for agent={agent_type!r}\n"
            f"stderr: {stderr_tail}\n"
            f"progress_tail: {progress_tail}"
        )

    if final_envelope is None:
        raise AgentTransportError(
            f"claude (stream-json) returned without a final result event "
            f"for agent={agent_type!r}"
        )

    return _parse_envelope(agent_type, final_envelope,
                           tool_uses=tool_uses, progress_lines=progress_lines)


def parse_stream_json_events(text: str) -> tuple[Optional[dict], list[dict], list[str]]:
    """Pure parser for testing — takes captured stream-json text, returns
    (final_envelope, tool_uses, progress_lines). Mirrors _run_streaming logic.
    """
    final_envelope: Optional[dict] = None
    tool_uses: list[dict] = []
    progress_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        skip_current_item = False
        try:
            event = json.loads(line)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        etype = event.get("type")
        if etype == "result":
            final_envelope = event
            continue
        if etype == "assistant":
            msg = event.get("message", {}) or {}
            for blk in msg.get("content") or []:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "tool_use":
                    tool_uses.append({
                        "tool_name": blk.get("name"),
                        "input": blk.get("input"),
                        "id": blk.get("id"),
                    })
                elif blk.get("type") == "text":
                    for ln in blk.get("text", "").splitlines():
                        progress_lines.append(ln)
    return final_envelope, tool_uses, progress_lines


# ---------------------------------------------------------------------------
# CLI for smoke-testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Smoke-test agent_transport.py")
    ap.add_argument("--agent", default="general-purpose",
                    help="agent type (must be in `claude agents`)")
    ap.add_argument("--prompt", default="Reply with exactly: TRANSPORT_TEST_OK. Then exit.",
                    help="prompt body")
    ap.add_argument("--mode", choices=["foreground", "background"], default="foreground")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    if args.mode == "foreground":
        t0 = time.time()
        result = spawn_agent_foreground(
            args.agent, args.prompt, timeout_sec=args.timeout
        )
        wall = time.time() - t0
        log.info(f"[transport] success={result.success} is_error={result.is_error}")
        log.info(f"[transport] duration_ms={result.duration_ms} wall_s={wall:.1f}")
        log.info(f"[transport] cost_usd={result.cost_usd:.4f}")
        log.info(f"[transport] terminal_reason={result.terminal_reason}")
        log.info(f"[transport] output: {result.output_text!r}")
        sys.exit(0 if result.success else 1)
    else:
        out_file = Path("/tmp/agent_transport_smoke.json")
        popen = spawn_agent_background(args.agent, args.prompt, out_file)
        log.info(f"[transport] background pid={popen.pid} output_file={out_file}")
        try:
            popen.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            popen.kill()
            log.info(f"[transport] TIMEOUT after {args.timeout}s; killed.")
            sys.exit(2)
        result = parse_background_result(args.agent, out_file)
        log.info(f"[transport] success={result.success} cost_usd={result.cost_usd:.4f}")
        log.info(f"[transport] output: {result.output_text!r}")
        sys.exit(0 if result.success else 1)
