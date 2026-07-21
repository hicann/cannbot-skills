# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch driver for /autoresearch.

Loads a manifest from <batch_dir>/manifest.{yaml,json}, resolves the op
list against the <op_name>_{ref,kernel}.py naming convention, then drives each
op end-to-end via headless `claude --print`. Streams stdout to console and
batch.log, updates batch_progress.json after every op.

Usage:
    python scripts/batch/run.py <batch_dir> --devices N \\
        [--max-rounds N] [--eval-timeout S] [--timeout-min M] \\
        [--only op1,op2] [--limit N] [--retry-errored] [--cooldown-sec S]
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from op_autoresearch.utils.console import emit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import manifest as mf
import phase_machine
import task_handle
from op_autoresearch.utils.process_utils import (
    popen_process_group_kwargs,
    terminate_process_tree,
)
from utils.settings import (
    batch_cooldown_sec,
    batch_run_timeout_min,
    batch_transient_retries,
    default_eval_timeout,
    default_max_rounds,
    recorded_speedup,
)

logger = logging.getLogger(__name__)

# Force UTF-8 on this script's own stdout/stderr. claude.cmd prints
# tokens like `µs`, box-drawing rules, and Chinese rationale text;
# on Chinese-locale Windows the default GBK codec can't encode them
# and console output raises mid-batch, killing
# the supervisor while ops are still queued. Sister fix to the
# subprocess-read encoding pin already on Popen below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _DiscardingTextStream:
    """Minimal text sink used after the controlling console disappears."""

    encoding = "utf-8"

    @staticmethod
    def write(text: str) -> int:
        return len(text)

    @staticmethod
    def flush() -> None:
        return None


def _resolve_claude_bin(name: str) -> str:
    """Resolve `name` to a real path Popen can execute.

    Bare 'claude' on Windows fails: Popen(list) calls CreateProcess
    directly and CreateProcess does not apply PATHEXT, so it won't find
    `claude.cmd`. shutil.which DOES walk PATHEXT and returns the full
    path (including `.cmd` on Windows). On POSIX this is also fine —
    shutil.which returns the resolved absolute path, which Popen handles
    identically to the bare name.
    """
    if os.path.isabs(name) or os.sep in name:
        return name  # caller already gave a path
    resolved = shutil.which(name)
    # Fall through on a miss; Popen then raises the original, useful error.
    return resolved or name


# Force line-buffered stdout so logs flush in real time when run via nohup.
try:
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(line_buffering=True)
    else:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except OSError:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _console_write(text: str, *, flush: bool = True) -> None:
    """Write to the interactive console without making it part of the
    batch transaction.

    A foreground batch is commonly launched through ``ssh ... | tee``.  If
    that controlling SSH connection disappears, the remote agent and its eval
    children can keep running, but the inherited stdout pipe eventually raises
    ``BrokenPipeError``.  Console output is only an observer; losing it must not
    prevent the driver from harvesting a completed task into
    ``batch_progress.json``.  After the first sink failure, replace stdout with
    ``os.devnull`` so later console writes in the batch are harmless.
    """
    try:
        emit(text, end="", flush=flush)
    except (OSError, ValueError):
        sys.stdout = _DiscardingTextStream()


def _emit(log_fp, text: str) -> None:
    """Persist output first, then mirror it to the best-effort console."""
    log_fp.write(text)
    log_fp.flush()
    _console_write(text)


# Keep the prompt to the slash command itself. Claude passes all text after
# `/autoresearch` as `$ARGUMENTS`; appending extra prose here corrupts the
# argument vector consumed by scripts/engine/parse_args.py.
PROMPT_TEMPLATE = (
    "/autoresearch --ref {ref} --kernel {kernel} --op-name {op} {hw} "
    "--max-rounds {rounds} --eval-timeout {timeout}"
)

LOCK_FILENAME = ".batch.lock"


@dataclass(frozen=True)
class CaseRequest:
    """Inputs shared by the Claude and OpenCode case drivers."""

    batch_dir: Path
    case: dict
    args: argparse.Namespace
    hw_arg: str
    log_fp: Any
    prev_task_dir: Optional[str] = None


@dataclass(frozen=True)
class StreamRequest:
    """One supervised child process invocation."""

    cmd: list[str]
    cwd: str
    started: float
    timeout_s: float
    log_fp: Any
    line_cb: Optional[Callable[[str], None]] = None
    extra_env: Optional[dict[str, str]] = None


@dataclass
class CaseResult:
    """Mutable outcome while a case is retried and finalized."""

    task_dir: Path
    phase: str
    result: dict
    status: str
    rc: int
    interrupted: bool
    consistency_note: str = ""
    retries: int = 0


class BatchLockError(RuntimeError):
    """The batch directory is already owned by another runner."""


class BatchConfigError(ValueError):
    """The batch command line or manifest is invalid."""


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            synchronize_access = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(synchronize_access, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            # Can't tell — err on the safe side and assume alive so the user
            # has to confirm by removing the lock manually.
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _unlink_stale_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError as exc:
        logger.debug("stale batch lock already removed %s: %s", lock, exc)


def acquire_lock(batch_dir: Path) -> Path:
    """Prevent two run.py instances racing on the same batch_progress.json.
    Stale locks (PID gone) are auto-cleared; live locks abort with a hint.

    Uses os.open(O_CREAT|O_EXCL) — atomic create-or-fail on every OS
    Python supports (POSIX + Windows). The old "exists() then write_text"
    pattern was a check-then-act race: two run.py instances starting in
    parallel could both see no lock, both write their PID, and both
    proceed to corrupt batch_progress.json. Atomic create eliminates
    that race; the stale-lock retry path is bounded to one cycle so two
    racers both finding a stale lock can't ping-pong it forever — the
    loser of the second atomic create aborts with a clear hint.
    """
    lock = batch_dir / LOCK_FILENAME
    for attempt in range(2):
        try:
            fd = os.open(str(lock),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            try:
                pid = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and pid_alive(pid):
                raise BatchLockError(
                    f"\nanother batch run is active on this batch dir "
                    f"(pid={pid}, lock={lock}).\n"
                    f"if you're sure no run.py is running, remove {lock} "
                    f"and retry.\n"
                ) from exc
            if attempt == 0:
                _unlink_stale_lock(lock)
                continue
            raise BatchLockError(
                f"\nbatch dir {batch_dir} is being claimed concurrently "
                f"(another run.py won the lock race).\n"
                f"retry in a moment.\n"
            ) from exc
        else:
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            return lock
    raise BatchLockError(f"failed to acquire batch lock: {lock}")


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError as exc:
        logger.debug("failed to release batch lock %s: %s", lock, exc)


def recover_stale_running(progress: dict) -> tuple[int, int]:
    """Demote 'running' cases that are demonstrably orphaned. We hold the
    batch dir lock when this fires, so anything still 'running' was left
    by a previous run.py. "Previous run.py" can mean any of:
    SIGKILLed, OOM-killed, machine rebooted (true orphans) OR another
    runner that's still alive but raced us through the (now atomic)
    lock OR a case whose runner is gone but whose claude --print is
    still finishing its last tool call (state.json keeps getting
    touched on the task_dir).

    Demoting all "running" indiscriminately means --retry-errored
    later re-launches a case that's still in flight, putting two
    Claude processes on the same task and the same worker — a silent
    double-run footgun. Check before demoting:
      - if the case carries a `runner_pid` and that pid is alive,
        it's not orphaned — skip.
      - if the task_dir is_task_active (owner + fresh heartbeat in
        state.json), /autoresearch is still writing — skip.
      - once both owners are dead, harvest an already-FINISH task from its
        authoritative state instead of launching the whole optimization again.
      - otherwise the case is a real incomplete orphan; demote with a note.

    Returns ``(demoted, harvested)``.
    """
    cases = progress.get("cases", {})
    demoted = harvested = 0
    now = mf.now_iso()
    for c in cases.values():
        if c.get("status") != "running":
            continue
        # Skip if the previous runner is still alive.
        runner_pid = c.get("runner_pid")
        if isinstance(runner_pid, int) and runner_pid > 0 and pid_alive(runner_pid):
            continue
        # Skip if the task is still active — claude --print may have
        # lost its runner pid (e.g. detached) but the agent loop is
        # still bumping state.last_touched.
        td = c.get("task_dir")
        if td and phase_machine.is_task_active(td):
            continue
        if td:
            task_dir = Path(td)
            if mf.read_phase(task_dir) == "FINISH":
                c.update({
                    "status": "done",
                    "finished_at": now,
                    "final_phase": "FINISH",
                    "rc": 0,
                    "result": mf.read_task_state(task_dir),
                })
                existing = (c.get("note") or "").strip()
                tag = ("harvested completed task on batch restart "
                       "after runner exit")
                c["note"] = f"{existing}; {tag}" if existing else tag
                harvested += 1
                continue
        c["status"] = "error"
        c["finished_at"] = now
        existing = (c.get("note") or "").strip()
        tag = ("stale running, demoted on batch restart "
               "(no live runner_pid, task not active in state.json)")
        c["note"] = f"{existing}; {tag}" if existing else tag
        demoted += 1
    return demoted, harvested


def build_prompt(case: dict, hw_arg: str,
                 max_rounds: int, eval_timeout: int) -> str:
    """Quote every value-bearing flag with shlex.quote so paths with
    spaces (e.g. batch dir under `C:\\Users\\Foo Bar\\...`, or
    `--output-dir "my tasks"`) reach /autoresearch as one argv each.
    """
    return PROMPT_TEMPLATE.format(
        ref=shlex.quote(case["ref"]),
        kernel=shlex.quote(case["kernel"]),
        op=shlex.quote(case["op_name"]),
        hw=hw_arg,
        rounds=max_rounds,
        timeout=eval_timeout,
    )


def build_claude_cmd(args: argparse.Namespace, prompt: str) -> list[str]:
    cmd = [
        args.claude_bin,
        "--print",
        "--permission-mode", "acceptEdits",
        "--output-format", "text",
    ]
    if args.model:
        cmd += ["--model", args.model]
    cmd += args.extra_claude_arg
    cmd += [prompt]
    return cmd


def env_with_no_proxy(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    extras = "127.0.0.1,localhost"
    existing = env.get("NO_PROXY", "")
    env["NO_PROXY"] = f"{existing},{extras}".strip(",") if existing else extras
    env["no_proxy"] = env["NO_PROXY"]
    env["PYTHONIOENCODING"] = "utf-8"  # propagates UTF-8 to claude --print + its Bash-tool subprocs
    if extra:
        env.update(extra)
    return env


def _begin_case(batch_dir: Path, case: dict,
                prev_task_dir: Optional[str]):
    """Shared manifest/ownership setup for every agent driver."""
    op = case["op_name"]
    mf.update_case(
        batch_dir, op, status="running", started_at=mf.now_iso(),
        finished_at=None, task_dir=None, final_phase=None, rc=None,
        runner_pid=os.getpid(), note="",
    )
    pre_task_dirs = mf.snapshot_task_dirs()
    if phase_machine.clear_active_task(expected_task_dir=prev_task_dir):
        return op, mf.repo_root(), time.time(), pre_task_dirs
    _console_write(
        f"[run] op={op}: refusing to start — another session is active on "
        "this checkout. Stop it before retrying.\n")
    mf.update_case(batch_dir, op, status="error", finished_at=mf.now_iso(),
                   note="aborted: prior owner still active")
    return None


def _find_task_dir(batch_dir: Path, op: str, pre_task_dirs: set,
                   candidate: Optional[str] = None) -> Optional[Path]:
    """Resolve the task produced by either agent, newest evidence first."""
    recorded = (mf.load_progress(batch_dir).get("cases", {}).get(op, {})
                .get("task_dir"))
    for raw in (candidate, recorded):
        if not isinstance(raw, str):
            continue
        task_dir = Path(raw)
        if (task_dir.is_dir()
                and mf.task_dir_belongs_to_op(task_dir.name, op)):
            return task_dir.resolve()
    found = mf.pick_new_task_dir(pre_task_dirs, op)
    return found.resolve() if found is not None else None


def _read_case_result(task_dir: Path, interrupted: bool):
    """Replay an interrupted journal, then read one canonical outcome."""
    consistency_note = ""
    try:
        with task_handle.open_task(
                str(task_dir), role=task_handle.Role.SUPERVISOR):
            pass
    except task_handle.TaskConsistencyError as exc:
        consistency_note = f"; post-run heal refused: {exc}"
    phase = mf.read_phase(task_dir)
    result = mf.read_task_state(task_dir)
    status = ("done" if phase == "FINISH" and not interrupted
              and not consistency_note else "error")
    return phase, result, status, consistency_note


def _finish_case(request: CaseRequest, op: str, outcome: CaseResult) -> int:
    note = ""
    if outcome.status == "error":
        note = f"phase={outcome.phase} rc={outcome.rc}"
        if outcome.interrupted:
            note += "; interrupted"
        note += outcome.consistency_note
    if outcome.retries:
        retry_note = f"transient_retries={outcome.retries}"
        note = f"{retry_note}; {note}" if note else retry_note
    mf.update_case(
        request.batch_dir,
        op,
        status=outcome.status,
        task_dir=str(outcome.task_dir.resolve()),
        finished_at=mf.now_iso(),
        final_phase=outcome.phase,
        rc=outcome.rc,
        result=outcome.result,
        note=note,
    )
    _console_write(
        f"[run] result: op={op} task_dir={outcome.task_dir} "
        f"phase={outcome.phase} status={outcome.status}\n")
    return 130 if outcome.interrupted else (0 if outcome.status == "done" else 1)


def _run_driver(
    request: CaseRequest,
    process: StreamRequest,
    launch_name: str,
    agent: str = "",
) -> tuple[int, bool]:
    op = request.case["op_name"]
    agent_tag = f" (agent={agent})" if agent else ""
    header = (
        f"\n{'=' * 72}\n"
        f"[run {datetime.now(timezone.utc).astimezone().replace(tzinfo=None).isoformat(timespec='seconds')}] op={op} "
        f"{request.hw_arg} rounds={request.args.max_rounds}{agent_tag}\n"
        f"[run] launching: {launch_name} (cwd={process.cwd}, "
        f"timeout={request.args.timeout_min}min)\n{'─' * 72}\n")
    _emit(request.log_fp, header)
    rc, interrupted = stream_subprocess(process)
    footer = (f"{'─' * 72}\n[run] {launch_name} exited rc={rc} after "
              f"{time.time() - process.started:.0f}s\n")
    _emit(request.log_fp, footer)
    return rc, interrupted


def _stream_request(
    request: CaseRequest,
    cmd: list[str],
    cwd: Path,
    started: float,
    line_cb: Optional[Callable[[str], None]] = None,
) -> StreamRequest:
    return StreamRequest(
        cmd=cmd,
        cwd=str(cwd),
        started=started,
        timeout_s=request.args.timeout_min * 60,
        log_fp=request.log_fp,
        line_cb=line_cb,
        extra_env={
            "AR_BATCH_DIR": str(request.batch_dir.resolve()),
            "AR_BATCH_OP": request.case["op_name"],
        },
    )


def run_one(request: CaseRequest) -> int:
    batch_dir = request.batch_dir
    case = request.case
    args = request.args
    hw_arg = request.hw_arg
    prev_task_dir = request.prev_task_dir
    context = _begin_case(batch_dir, case, prev_task_dir)
    if context is None:
        return 2
    op, repo_root, started, pre_task_dirs = context
    prompt = build_prompt(case, hw_arg,
                          args.max_rounds, args.eval_timeout)
    cmd = build_claude_cmd(args, prompt)

    process = _stream_request(request, cmd, repo_root, started)
    last_rc, interrupted = _run_driver(
        request,
        process,
        f"{args.claude_bin} --print",
    )

    task_dir = _find_task_dir(batch_dir, op, pre_task_dirs)
    if task_dir is None:
        mf.update_case(batch_dir, op,
                       status="error",
                       finished_at=mf.now_iso(),
                       rc=last_rc,
                       note=f"no task_dir found; rc={last_rc}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    phase, result, final_status, consistency_note = _read_case_result(
        task_dir, interrupted)

    outcome = CaseResult(
        task_dir,
        phase,
        result,
        final_status,
        last_rc,
        interrupted,
        consistency_note,
    )
    _retry_claude_case(request, repo_root, outcome)
    return _finish_case(request, op, outcome)


def _retry_claude_case(
    request: CaseRequest,
    repo_root: Path,
    outcome: CaseResult,
) -> None:
    driver_failed = outcome.status == "error" and outcome.rc != 0
    state_can_resume = (
        not outcome.interrupted
        and not outcome.consistency_note
        and outcome.phase != "FINISH"
    )
    progress_exists = (outcome.result or {}).get("progress_initialized") is True
    if not (driver_failed and state_can_resume and progress_exists):
        return

    max_retries = batch_transient_retries()
    resume_prompt = f"/autoresearch --resume {outcome.task_dir} --force"
    resume_cmd = build_claude_cmd(request.args, resume_prompt)
    while (
        outcome.retries < max_retries
        and outcome.status == "error"
        and outcome.phase != "FINISH"
    ):
        outcome.retries += 1
        started = time.time()
        _emit(
            request.log_fp,
            f"[run] transient claude crash (rc={outcome.rc}, "
            f"phase={outcome.phase}); resuming attempt "
            f"{outcome.retries}/{max_retries} via --resume --force\n",
        )
        process = _stream_request(request, resume_cmd, repo_root, started)
        rc, interrupted = stream_subprocess(process)
        _console_write(
            f"[run] resume attempt {outcome.retries} exited rc={rc} after "
            f"{time.time() - started:.0f}s\n"
        )
        _heal_after_retry(outcome.task_dir)
        outcome.phase = mf.read_phase(outcome.task_dir)
        outcome.result = mf.read_task_state(outcome.task_dir)
        outcome.rc = rc
        outcome.interrupted = outcome.interrupted or interrupted
        outcome.status = (
            "done"
            if outcome.phase == "FINISH" and not outcome.interrupted
            else "error"
        )
        if interrupted:
            break


def _heal_after_retry(task_dir: Path) -> None:
    try:
        with task_handle.open_task(
            str(task_dir),
            role=task_handle.Role.SUPERVISOR,
        ):
            pass
    except task_handle.TaskConsistencyError as exc:
        logger.debug("task state remains inconsistent after batch recovery: %s", exc)


def stream_subprocess(request: StreamRequest) -> tuple[int, bool]:
    """Run a command, tee its combined stdout, and enforce a deadline.

    wall-clock cap, and invoke line_cb(line) per line. Reader-thread + queue
    poll so a silent child still hits the deadline (Windows can't select on
    pipes). Returns (returncode, interrupted). This is the agent-neutral
    streaming primitive used by both Claude and OpenCode drivers.

    The child is spawned in its own process group/session so a wall-clock or
    Ctrl-C kill takes down the entire `run_loop.py -> opencode run -> shell ->
    pipeline/build` tree, not just the driver.
    """
    proc = _spawn_stream_process(request)
    output_queue, done = _start_output_reader(proc)
    interrupted = False
    try:
        _drain_process_output(proc, request, output_queue, done)
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        interrupted = True
        terminate_process_tree(proc)
    return proc.returncode, interrupted


def _spawn_stream_process(request: StreamRequest) -> subprocess.Popen:
    child_env = env_with_no_proxy(request.extra_env)
    if os.name == "posix":
        child_env["PWD"] = os.path.abspath(request.cwd)
    return subprocess.Popen(
        request.cmd,
        cwd=request.cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
        **popen_process_group_kwargs(),
    )


def _start_output_reader(
    proc: subprocess.Popen,
) -> tuple["queue.Queue[str]", threading.Event]:
    output_queue: "queue.Queue[str]" = queue.Queue()
    done = threading.Event()

    def _reader() -> None:
        try:
            if proc.stdout is None:
                raise RuntimeError("batch child process stdout pipe was not created")
            for line in proc.stdout:
                output_queue.put(line)
        finally:
            done.set()

    threading.Thread(target=_reader, daemon=True).start()
    return output_queue, done


def _drain_process_output(
    proc: subprocess.Popen,
    request: StreamRequest,
    output_queue: "queue.Queue[str]",
    done: threading.Event,
) -> None:
    while True:
        try:
            line = output_queue.get(timeout=5)
        except queue.Empty:
            if time.time() - request.started > request.timeout_s:
                _emit(
                    request.log_fp,
                    "[run] WALL-CLOCK TIMEOUT, killing agent driver "
                    "+ its process tree\n",
                )
                terminate_process_tree(proc)
                return
            if done.is_set() and output_queue.empty():
                return
            continue
        _emit(request.log_fp, line)
        if request.line_cb:
            request.line_cb(line)


def run_one_opencode(request: CaseRequest) -> int:
    """Drive one op to FINISH with opencode. opencode 1.17.7 has no Stop
    hook, so a single `opencode run` can't self-loop to FINISH like
    `claude --print`. We delegate to the proven headless driver
    `.opencode/run_loop.py`, which scaffolds the task and re-invokes
    `opencode run --session <id>` until the phase machine reaches FINISH. This
    wrapper supplies the same batch bookkeeping run_one does (ownership
    handoff, task_dir binding, post-run heal, status/result recording) so
    both agents share the manifest / queue / summary orchestration verbatim.
    """
    batch_dir = request.batch_dir
    case = request.case
    context = _begin_case(batch_dir, case, request.prev_task_dir)
    if context is None:
        return 2
    op, repo_root, started, pre_task_dirs = context

    cmd = _build_opencode_command(request, repo_root, op)
    if cmd is None:
        return 2

    bound: dict[str, Optional[str]] = {"td": None}
    line_cb = partial(_capture_task_dir, request, op, bound)
    process = _stream_request(request, cmd, repo_root, started, line_cb)
    rc, interrupted = _run_driver(
        request,
        process,
        "run_loop.py",
        agent="opencode",
    )

    task_dir = _find_task_dir(
        batch_dir, op, pre_task_dirs, bound["td"])
    if task_dir is None:
        mf.update_case(batch_dir, op, status="error", finished_at=mf.now_iso(),
                       rc=rc,
                       note=f"no task_dir from run_loop; rc={rc}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    phase, result, status, consistency_note = _read_case_result(
        task_dir, interrupted)
    outcome = CaseResult(
        task_dir,
        phase,
        result,
        status,
        rc,
        interrupted,
        consistency_note,
    )
    return _finish_case(request, op, outcome)


def _capture_task_dir(
    request: CaseRequest,
    op: str,
    bound: dict[str, Optional[str]],
    line: str,
) -> None:
    if bound["td"] is not None:
        return
    marker = "[run_loop] task_dir="
    text = line.strip()
    if not text.startswith(marker):
        return
    task_dir = text[len(marker):].strip()
    if task_dir:
        bound["td"] = task_dir
        mf.update_case(
            request.batch_dir,
            op,
            task_dir=str(Path(task_dir).resolve()),
        )


def _build_opencode_command(
    request: CaseRequest,
    repo_root: Path,
    op: str,
) -> Optional[list[str]]:
    run_loop = repo_root / ".opencode" / "run_loop.py"
    if not run_loop.is_file():
        mf.update_case(
            request.batch_dir,
            op,
            status="error",
            finished_at=mf.now_iso(),
            note=f"opencode driver missing: {run_loop}",
        )
        return None
    cmd = [
        sys.executable,
        str(run_loop),
        "--ref",
        request.case["ref"],
        "--kernel",
        request.case["kernel"],
        "--op-name",
        op,
        "--max-rounds",
        str(request.args.max_rounds),
        "--eval-timeout",
        str(request.args.eval_timeout),
    ]
    cmd += shlex.split(request.hw_arg)
    if request.args.model:
        os.environ["AR_OPENCODE_MODEL"] = request.args.model
    return cmd


def filter_queue(progress: dict, args: argparse.Namespace) -> list[dict]:
    statuses = {"pending"}
    if args.retry_errored:
        statuses.add("error")
    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    out: list[dict] = []
    for v in progress.get("cases", {}).values():
        if v.get("status") not in statuses:
            continue
        if only and v.get("op_name") not in only:
            continue
        out.append(v)
    return out


def print_summary(batch_dir: Path, total_elapsed: float,
                  hw_arg: str) -> None:
    """Compact end-of-batch report + concrete next-step commands.

    Status lines: just done / error counts (skip / pending only shown when
    nonzero). Speedup distribution collapses into a single line — regress
    cases are part of `done`, not called out separately.

    Next-step commands echo back enough of the original invocation that
    the user can paste directly: batch dir path + the hardware flag we
    were called with. mode is read from the manifest by run.py so we
    don't repeat it.
    """
    progress = mf.load_progress(batch_dir)
    cases = progress.get("cases", {})
    counts, speedups = _case_counts_and_speedups(cases)

    emit()
    emit("=" * 72)
    emit(f"[batch done] elapsed={total_elapsed/60:.1f}min")
    speed_note = _format_speed_note(speedups)
    emit(f"  done : {counts['done']}{speed_note}")
    emit(f"  error: {counts['error']}")
    if counts["skip"]:
        emit(f"  skip : {counts['skip']}")
    if counts["pending"]:
        emit(f"  pending: {counts['pending']}")

    suggestions = _next_step_suggestions(batch_dir, hw_arg, counts)
    if suggestions:
        emit()
        emit("next steps:")
        for label, cmd in suggestions:
            emit(f"  {label}:")
            emit(f"    {cmd}")
    emit("=" * 72)


def _case_counts_and_speedups(cases: dict) -> tuple[dict[str, int], list[float]]:
    counts = {"done": 0, "error": 0, "skip": 0, "pending": 0, "running": 0}
    speedups: list[float] = []
    for v in cases.values():
        s = v.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
        if s != "done":
            continue
        r = v.get("result") or {}
        sp = recorded_speedup(r)
        if sp is not None:
            speedups.append(sp)
    return counts, speedups


def _format_speed_note(speedups: list[float]) -> str:
    if speedups:
        import statistics
        return (
            f"  (median {statistics.median(speedups):.2f}x, "
            f"best {max(speedups):.2f}x, worst {min(speedups):.2f}x; "
            f"{len(speedups)} with metric)"
        )
    return ""


def _next_step_suggestions(
    batch_dir: Path,
    hw_arg: str,
    counts: dict[str, int],
) -> list[tuple[str, str]]:
    repo_root = mf.repo_root()
    try:
        ws_str = str(batch_dir.relative_to(repo_root))
    except ValueError:
        ws_str = str(batch_dir)

    suggestions: list[tuple[str, str]] = []
    if counts["error"]:
        suggestions.append((
            f"retry {counts['error']} errored ops",
            f"python scripts/batch/run.py {ws_str} "
            f"{hw_arg} --retry-errored",
        ))
    if counts["pending"]:
        suggestions.append((
            f"resume {counts['pending']} pending ops",
            f"python scripts/batch/run.py {ws_str} {hw_arg}",
        ))
    return suggestions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch driver for /autoresearch.")
    parser.add_argument("batch_dir", help="dir containing manifest.yaml/json")
    _add_execution_options(parser)
    _add_agent_options(parser)
    return parser


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worker-url",
        default="",
        help=(
            "Comma-separated worker URLs (host:port). When set, eval routes "
            "to remote HTTP worker(s) and --devices is not required."
        ),
    )
    parser.add_argument(
        "--devices",
        default="",
        help=(
            "device ids, e.g. 0 or 0,1; required only for local eval. "
            "With --worker-url, this is an optional expected-device filter."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=default_max_rounds())
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=default_eval_timeout(),
        help=(
            "per-shape verify/profile budget in seconds; multi-shape calls "
            "scale the budget by their case count"
        ),
    )
    parser.add_argument(
        "--timeout-min",
        type=int,
        default=batch_run_timeout_min(),
        help="hard wall-clock cap per op in minutes",
    )
    _add_queue_options(parser)


def _add_queue_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--only", default="", help="comma-separated op names")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after N ops (0 = no limit)",
    )
    parser.add_argument(
        "--retry-errored",
        action="store_true",
        help="also queue ops with status=error",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=int,
        default=batch_cooldown_sec(),
        help="seconds to sleep between ops",
    )


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent",
        choices=["claude", "opencode"],
        default="claude",
        help=(
            "agent harness: Claude uses its Stop hook; OpenCode uses "
            ".opencode/run_loop.py. Both share the same batch state."
        ),
    )
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Claude model id, or AR_OPENCODE_MODEL for the OpenCode driver"
        ),
    )
    parser.add_argument(
        "--extra-claude-arg",
        action="append",
        default=[],
        help="extra arg to pass to claude (repeatable)",
    )


def _load_batch_setup(
    args: argparse.Namespace,
) -> tuple[Path, str, list[dict], str]:
    if args.agent == "claude":
        args.claude_bin = _resolve_claude_bin(args.claude_bin)
    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        raise BatchConfigError(f"batch dir not found: {batch_dir}")
    try:
        manifest_path = mf.find_manifest(batch_dir)
        manifest_data = mf.load_manifest(manifest_path)
        cases = mf.resolve_cases(batch_dir, manifest_data, "ref-kernel")
    except mf.ManifestError as exc:
        raise BatchConfigError(str(exc)) from exc
    return batch_dir, _hardware_arg(args), cases, "ref-kernel"


def _hardware_arg(args: argparse.Namespace) -> str:
    if not args.devices and not args.worker_url:
        raise BatchConfigError(
            "--devices (local eval) or --worker-url (remote worker) is required"
        )
    if not args.worker_url:
        return f"--devices {args.devices}"
    remote = f"--worker-url {args.worker_url}"
    return f"--devices {args.devices} {remote}" if args.devices else remote


def _refresh_batch_progress(
    batch_dir: Path,
    cases: list[dict],
    mode: str,
) -> dict:
    progress = mf.load_progress(batch_dir)
    demoted, harvested = recover_stale_running(progress)
    progress, dropped = mf.merge_cases(progress, cases, mode)
    mf.save_progress(batch_dir, progress)
    if demoted:
        emit(
            f"[batch] demoted {demoted} stale 'running' op(s) "
            "from a previous run -> error"
        )
    if harvested:
        emit(
            f"[batch] harvested {harvested} completed op(s) from "
            "authoritative task state"
        )
    if dropped:
        preview = ", ".join(dropped[:5])
        if len(dropped) > 5:
            preview += f", ... (+{len(dropped) - 5} more)"
        emit(f"[batch] dropped {len(dropped)} op(s) no longer in manifest: {preview}")
    return progress


def _run_locked_batch(
    batch_dir: Path,
    args: argparse.Namespace,
    hw_arg: str,
    cases: list[dict],
    mode: str,
) -> int:
    progress = _refresh_batch_progress(batch_dir, cases, mode)
    pending_cases = filter_queue(progress, args)
    if not pending_cases:
        emit("nothing to run.")
        return 0
    if args.limit:
        pending_cases = pending_cases[: args.limit]
    now = datetime.now(timezone.utc).astimezone().replace(tzinfo=None)
    emit(
        f"[batch {now.isoformat(timespec='seconds')}] "
        f"batch_dir={batch_dir}  {hw_arg}\n"
        f"[batch] queue size: {len(pending_cases)}  rounds={args.max_rounds}"
    )
    started = time.time()
    log_path = batch_dir / mf.LOG_FILENAME
    with log_path.open("a", encoding="utf-8", buffering=1) as log_fp:
        result = _run_pending_cases(
            batch_dir,
            args,
            hw_arg,
            pending_cases,
            log_fp,
        )
    print_summary(batch_dir, time.time() - started, hw_arg)
    return result


def _run_pending_cases(
    batch_dir: Path,
    args: argparse.Namespace,
    hw_arg: str,
    pending_cases: list[dict],
    log_fp: Any,
) -> int:
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    started = time.time()
    previous_task_dir: Optional[str] = None
    total = len(pending_cases)
    for index, case in enumerate(pending_cases, 1):
        request = CaseRequest(
            batch_dir,
            case,
            args,
            hw_arg,
            log_fp,
            previous_task_dir,
        )
        rc, previous_task_dir = _execute_case(
            request,
            index,
            total,
            started,
        )
        if rc is None:
            counts["skipped"] += 1
            continue
        if rc == 0:
            counts["ok"] += 1
        else:
            counts["failed"] += 1
        _print_running_totals(index, total, case["op_name"], rc, counts)
        if rc == 130:
            emit("\n[batch] op interrupted, stopping.")
            return 130
        if index < total and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)
    return 0 if counts["failed"] == 0 else 1


def _execute_case(
    request: CaseRequest,
    index: int,
    total: int,
    batch_started: float,
) -> tuple[Optional[int], Optional[str]]:
    op = request.case["op_name"]
    current = filter_queue(mf.load_progress(request.batch_dir), request.args)
    if not any(case["op_name"] == op for case in current):
        emit(f"[{index}/{total}] {op}: status changed underfoot, skipping")
        return None, request.prev_task_dir
    emit(
        f"\n[{index}/{total}] starting op={op}  "
        f"elapsed_total={(time.time() - batch_started) / 60:.1f}min"
    )
    driver = run_one_opencode if request.args.agent == "opencode" else run_one
    try:
        rc = driver(request)
    except KeyboardInterrupt:
        emit("\n[batch] Ctrl-C — current op recorded, stopping.")
        return 130, request.prev_task_dir
    settled = (
        mf.load_progress(request.batch_dir)
        .get("cases", {})
        .get(op, {})
        .get("task_dir")
    )
    return rc, settled or request.prev_task_dir


def _print_running_totals(
    index: int,
    total: int,
    op: str,
    rc: int,
    counts: dict[str, int],
) -> None:
    emit(
        f"[{index}/{total}] {op} done rc={rc}  running totals: "
        f"ok={counts['ok']} fail={counts['failed']} "
        f"skipped={counts['skipped']}"
    )


def main() -> int:
    args = _build_parser().parse_args()
    try:
        batch_dir, hw_arg, cases, mode = _load_batch_setup(args)
    except BatchConfigError as exc:
        emit(exc, file=sys.stderr)
        return 2
    try:
        lock_path = acquire_lock(batch_dir)
    except BatchLockError as exc:
        emit(exc, file=sys.stderr)
        return 1
    try:
        return _run_locked_batch(batch_dir, args, hw_arg, cases, mode)
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    sys.exit(main())
