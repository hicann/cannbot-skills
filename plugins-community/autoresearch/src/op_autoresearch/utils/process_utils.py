# Copyright 2025 Huawei Technologies Co., Ltd
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

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from op_autoresearch.core.worker.eval_config import (
    resolve_kill_drain_s,
    resolve_kill_grace_s,
)

logger = logging.getLogger(__name__)

_REGISTRY_ENV = "OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY"
_registry_lock = threading.Lock()


@dataclass(frozen=True)
class CommandCaptureOptions:
    """Execution controls for :func:`run_command_capture`."""

    shell: bool = False
    env: Optional[Dict[str, str]] = None
    timeout: Optional[float] = None
    cwd: Optional[str] = None
    cancel_event: Any = None


def popen_process_group_kwargs():
    """Popen kwargs that isolate the child in its own process group."""
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def _process_start_token(pid: int) -> Optional[str]:
    """Linux process start tick used to reject a reused PID/PGID."""
    if os.name != "posix":
        return None
    try:
        # /proc/<pid>/stat field 22.  Split after the final ')' because comm
        # may itself contain spaces or parentheses.
        rest = Path(f"/proc/{pid}/stat").read_text(
            encoding="utf-8"
        ).rsplit(")", 1)[1].split()
        return rest[19]
    except (OSError, IndexError):
        return None


def _registry_path() -> Optional[Path]:
    raw = os.environ.get(_REGISTRY_ENV, "").strip()
    return Path(raw) if raw and os.name == "posix" else None


def _load_registry(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        groups = data.get("groups") if isinstance(data, dict) else None
        return groups if isinstance(groups, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_registry(path: Path, groups: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"version": 1, "groups": groups}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def register_process_group(process) -> None:
    """Persist an eval PGID while a worker request owns it.

    The registry is opt-in via ``OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY``.  It is not a
    scheduler or another process state machine: it is only a crash-recovery
    index for trees that the kernel reparents to PID 1 when the worker daemon
    itself is SIGKILLed.
    """
    path = _registry_path()
    pid = getattr(process, "pid", None)
    if path is None or not isinstance(pid, int) or pid <= 0:
        return
    token = _process_start_token(pid)
    owner_pid = os.getpid()
    owner_token = _process_start_token(owner_pid)
    if token is None or owner_token is None:
        return
    record = {
        "pgid": pid,
        "start_token": token,
        "owner_pid": owner_pid,
        "owner_start_token": owner_token,
        "registered_at": time.time(),
    }
    with _registry_lock:
        groups = _load_registry(path)
        groups[str(pid)] = record
        _save_registry(path, groups)


def unregister_process_group(process) -> None:
    """Remove this exact process generation from the crash registry."""
    path = _registry_path()
    pid = getattr(process, "pid", None)
    if path is None or not isinstance(pid, int) or pid <= 0:
        return
    with _registry_lock:
        groups = _load_registry(path)
        record = groups.get(str(pid))
        if (isinstance(record, dict)
                and record.get("start_token") == _process_start_token(pid)):
            groups.pop(str(pid), None)
            _save_registry(path, groups)
        elif record is not None and _process_start_token(pid) is None:
            # Normal completion: /proc entry is already gone, and only this
            # daemon could have reached this finally for the recorded group.
            groups.pop(str(pid), None)
            _save_registry(path, groups)


def _group_alive(pgid: int) -> bool:
    proc_root = Path("/proc")
    if proc_root.is_dir():
        # killpg(..., 0) also succeeds when a group contains only zombies.
        # Zombies execute no work and cannot hold an NPU context; treating
        # them as live makes successor cleanup falsely retain stale records
        # until an unrelated parent happens to wait().
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                rest = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                state, process_group = rest[0], int(rest[2])
            except (OSError, IndexError, ValueError):
                continue
            if process_group == pgid and state != "Z":
                return True
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _registered_group(record) -> Optional[tuple[int, int]]:
    if not isinstance(record, dict):
        return None
    try:
        return int(record["pgid"]), int(record["owner_pid"])
    except (KeyError, TypeError, ValueError):
        return None


def _record_matches_process(record: dict, pgid: int) -> bool:
    leader_token = _process_start_token(pgid)
    return (leader_token is None
            or leader_token == record.get("start_token"))


def _record_owner_alive(record: dict, owner_pid: int) -> bool:
    return (_process_start_token(owner_pid)
            == record.get("owner_start_token"))


def _wait_for_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _group_alive(pgid)


def _terminate_orphan_group(pgid: int) -> bool:
    """Return whether the process group exited after TERM/KILL."""
    os.killpg(pgid, signal.SIGTERM)
    if _wait_for_group_exit(pgid, resolve_kill_grace_s()):
        return True
    os.killpg(pgid, signal.SIGKILL)
    return _wait_for_group_exit(pgid, max(resolve_kill_grace_s(), 1.0))


def _surviving_registry_record(record: Any, reaped: list[int]) -> Optional[dict]:
    group = _registered_group(record)
    if group is None:
        return None
    pgid, owner_pid = group
    if not _record_matches_process(record, pgid):
        return None
    if _record_owner_alive(record, owner_pid):
        return record
    if not _group_alive(pgid):
        return None
    try:
        terminated = _terminate_orphan_group(pgid)
    except ProcessLookupError:
        return None
    except PermissionError:
        return record
    if terminated:
        reaped.append(pgid)
        return None
    logger.warning("Orphan eval PGID %s survived SIGKILL", pgid)
    return record


def reap_orphaned_process_groups() -> list[int]:
    """Kill verified eval groups left by a dead predecessor daemon.

    A live owner is never touched.  If the group leader PID exists, its Linux
    start tick must match the registry, which prevents PID reuse from targeting
    an unrelated process.  The leader may already be gone while grandchildren
    retain the PGID; that is precisely why cleanup targets the recorded group.
    """
    path = _registry_path()
    if path is None:
        return []
    reaped: list[int] = []
    with _registry_lock:
        groups = _load_registry(path)
        survivors = {}
        for key, record in groups.items():
            survivor = _surviving_registry_record(record, reaped)
            if survivor is not None:
                survivors[key] = survivor
        _save_registry(path, survivors)
    return reaped


def _returncode(process):
    poll = getattr(process, "poll", None)
    if callable(poll):
        return poll()
    return getattr(process, "returncode", None)


def _process_tree_alive(process) -> bool:
    """Whether the isolated child process *group* still has members.

    On POSIX the group leader may exit while a grandchild keeps inherited
    stdout/stderr pipes open.  Looking only at ``Popen.returncode`` then says
    "done" even though the workload is still running.  Every caller creates
    the child with :func:`popen_process_group_kwargs`, so the leader PID is
    also the process-group id and remains a valid ``killpg`` target after the
    leader exits.
    """
    pid = getattr(process, "pid", None)
    if pid is None:
        return False
    if os.name == "posix":
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return _returncode(process) is None


def _wait_process_tree(process, timeout: float) -> bool:
    """Synchronously wait until the whole isolated tree is gone."""
    deadline = time.monotonic() + max(0.0, timeout)
    while _process_tree_alive(process):
        # poll() reaps the direct child when it has exited; group liveness is
        # deliberately checked separately for surviving grandchildren.
        poll = getattr(process, "poll", None)
        if callable(poll):
            poll()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    poll = getattr(process, "poll", None)
    if callable(poll):
        poll()
    return True


async def _wait_process_tree_async(process, timeout: float) -> bool:
    """Async twin of :func:`_wait_process_tree`."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    while _process_tree_alive(process):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.05, remaining))
    await process.wait()  # reap the direct child if it exited before its group
    return True


def _taskkill_process_tree(pid: int, *, force: bool) -> bool:
    taskkill = shutil.which("taskkill")
    if taskkill is None:
        logger.warning("taskkill was not found on PATH; falling back")
        return False
    args = [taskkill]
    if force:
        args.append("/F")
    args += ["/T", "/PID", str(pid)]
    try:
        subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        logger.warning("taskkill failed (%s); falling back", exc)
        return False
    return True


def kill_process_tree(process, *, force=False):
    """Send SIGTERM/SIGKILL (or Win32 taskkill /T) to the child's group."""
    pid = getattr(process, "pid", None)
    if pid is None:
        return
    try:
        if os.name == "posix":
            sig = signal.SIGKILL if force else signal.SIGTERM
            # Do not call os.getpgid(pid): the group leader may already have
            # exited while descendants remain.  start_new_session=True makes
            # the original leader PID the stable PGID for the whole tree.
            os.killpg(pid, sig)
            return
        if _returncode(process) is not None:
            return
        if os.name == "nt" and _taskkill_process_tree(pid, force=force):
            return
        if force:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        return


def terminate_process_tree(process, *, grace_s=None):
    """SIGTERM → grace → SIGKILL. ``grace_s=None`` reads config (default 5s).

    The grace matters on NPU: PyTorch+CANN ACL teardown often needs 2-5s
    to release the device context; too short and SIGKILL leaves residual
    TS state on the card (same path as the 6/17 device-5 wedge).
    """
    if grace_s is None:
        grace_s = resolve_kill_grace_s()
    kill_process_tree(process, force=False)
    if _wait_process_tree(process, grace_s):
        return
    kill_process_tree(process, force=True)
    force_wait = grace_s if grace_s > 0 else 1.0
    if not _wait_process_tree(process, force_wait):
        logger.warning("Process tree did not exit after SIGKILL/taskkill: pid=%s",
                       getattr(process, "pid", None))


def _command_environment(env, cwd):
    command_env = (os.environ if env is None else env).copy()
    command_env["PYTHONUNBUFFERED"] = "1"
    if cwd is not None and os.name == "posix":
        command_env["PWD"] = os.path.abspath(cwd)
    return command_env


def _communicate_until_stop(process, timeout, cancel_event):
    if cancel_event is None:
        try:
            return process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    deadline = None if timeout is None else time.monotonic() + timeout
    while not cancel_event.is_set():
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            break
        wait_s = 0.1 if remaining is None else min(0.1, remaining)
        try:
            return process.communicate(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return None


def _drain_terminated_process(process):
    terminate_process_tree(process)
    try:
        return process.communicate(timeout=resolve_kill_drain_s())
    except subprocess.TimeoutExpired:
        return "", ""


def run_command_capture(cmd, options: Optional[CommandCaptureOptions] = None):
    """Run a command, kill its group on timeout/cancel, and drain pipes.

    ``cancel_event`` is intentionally a small duck-typed hook for commands
    dispatched through an executor thread.  It lets the owning coroutine
    stop the real process tree and wait for the thread before releasing a
    device lease; cancelling an executor Future alone cannot stop its thread.
    """
    options = options or CommandCaptureOptions()
    process = subprocess.Popen(
        cmd,
        shell=options.shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        env=_command_environment(options.env, options.cwd),
        cwd=options.cwd,
        **popen_process_group_kwargs(),
    )
    register_process_group(process)
    try:
        completed = _communicate_until_stop(
            process,
            options.timeout,
            options.cancel_event,
        )
        if completed is not None:
            stdout, stderr = completed
            return process.returncode, stdout, stderr, False
        stdout, stderr = _drain_terminated_process(process)
        return process.returncode, stdout, stderr, True
    finally:
        unregister_process_group(process)


async def _stop_async_process_tree(process, task_id: str, action: str) -> None:
    grace_s = resolve_kill_grace_s()
    kill_process_tree(process, force=False)
    if await _wait_process_tree_async(process, grace_s):
        return
    kill_process_tree(process, force=True)
    force_wait = grace_s if grace_s > 0 else 1.0
    if not await _wait_process_tree_async(process, force_wait):
        logger.warning(
            "[%s] %s child did not exit after force kill: pid=%s",
            task_id,
            action,
            getattr(process, "pid", None),
        )


async def _communicate_or_stop(process, timeout: Optional[int],
                               task_id: str, action: str
                               ) -> Tuple[bytes, bytes, bool]:
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return stdout, stderr, False
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        await _stop_async_process_tree(process, task_id, action)
        if isinstance(exc, asyncio.CancelledError):
            logger.warning("[%s] %s cancelled; child process tree terminated",
                           task_id, action)
            raise
        logger.error("[%s] %s timed out.", task_id, action)
        return b"", b"", True


async def communicate_or_kill(process, timeout: Optional[int],
                              task_id: str, action: str
                              ) -> Tuple[bytes, bytes, bool]:
    """Await an asyncio subprocess; kill its process group on timeout/cancel.

    Returns ``(stdout, stderr, timed_out)``. On asyncio cancel the child tree
    is torn down and CancelledError re-raised (caller's teardown stays
    correct). Grace before SIGKILL is ``resolve_kill_grace_s()`` (default 5s).
    Async twin of :func:`terminate_process_tree`; single owner of the
    "kill the tree on timeout/cancel" policy for async subprocesses.
    """
    register_process_group(process)
    try:
        return await _communicate_or_stop(process, timeout, task_id, action)
    finally:
        unregister_process_group(process)
