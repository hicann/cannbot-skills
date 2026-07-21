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

"""Standalone worker lifecycle CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from op_autoresearch.cli.service import remote_dispatch
from op_autoresearch.cli.service.worker_config import (
    WorkerConfig,
    probe_local_arch,
)
from op_autoresearch.cli.utils.paths import get_process_log_dir
from op_autoresearch.cli.utils.worker_state import (
    get_worker_entry,
    live_worker_pid,
    load_worker_state,
    pid_alive,
    remove_worker_entry,
    save_worker_state,
    set_worker_entry,
    terminate_pid,
)
from op_autoresearch.core.worker.eval_config import eval_defaults
from op_autoresearch.utils.console import emit
from op_autoresearch.utils.process_utils import reap_orphaned_process_groups

logger = logging.getLogger(__name__)


def _status(port: int, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/status", timeout=timeout
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _registry_path(port: int) -> str:
    return str(Path(tempfile.gettempdir()) /
               f"op_autoresearch_worker_{port}_process_groups.json")


def _stop_local(port: int) -> int:
    state = load_worker_state()
    entry = get_worker_entry(state, port)
    pid = live_worker_pid(port)
    registry_pid = entry.get("pid") if entry else None
    use_registry_pid = pid is None and isinstance(registry_pid, int)
    if use_registry_pid and pid_alive(registry_pid):
        pid = entry["pid"]
    if pid:
        terminate_pid(pid, timeout=eval_defaults().kill_grace_s)
    os.environ["OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY"] = _registry_path(port)
    reaped = reap_orphaned_process_groups()
    remove_worker_entry(state, port)
    save_worker_state(state)
    emit(json.dumps({"stopped_pid": pid, "reaped_process_groups": reaped}))
    return 0


@dataclass(frozen=True)
class LocalTarget:
    backend: str
    arch: str
    device_ids: list[int]


def _local_target(args, config: WorkerConfig) -> LocalTarget:
    devices = args.devices or config.devices
    device_ids = [
        int(item.strip()) for item in devices.split(",") if item.strip()
    ]
    backend = (args.backend or config.backend).lower()
    arch = args.arch or probe_local_arch(
        backend, device_ids[0]) or config.arch
    return LocalTarget(backend=backend, arch=arch, device_ids=device_ids)


def _local_worker_environment(args, config: WorkerConfig,
                              target: LocalTarget, log_path: Path) -> dict:
    environment = os.environ.copy()
    environment.update(config.timing.as_env())
    environment.update(eval_defaults().as_env())
    environment.update({
        "WORKER_HOST": args.host,
        "WORKER_PORT": str(args.port),
        "WORKER_BACKEND": target.backend,
        "WORKER_ARCH": target.arch,
        "WORKER_DEVICES": ",".join(str(item) for item in target.device_ids),
        "OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY": _registry_path(args.port),
        "OP_AUTORESEARCH_WORKER_LOG_FILE": str(log_path),
    })
    return environment


def _local_spawn_options() -> tuple[int, dict]:
    if os.name == "posix":
        return 0, {"preexec_fn": os.setsid}
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP |
             getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
    return flags, {}


def _spawn_local_worker(environment: dict, log_path: Path):
    flags, options = _local_spawn_options()
    with open(log_path, "ab", buffering=0) as log_handle:
        return subprocess.Popen(
            [sys.executable, "-m", "op_autoresearch.worker.server"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            creationflags=flags,
            **options,
        )


def _record_local_worker(args, process, target: LocalTarget,
                         log_path: Path) -> None:
    state = load_worker_state()
    set_worker_entry(state, args.port, {
        "pid": process.pid,
        "backend": target.backend,
        "arch": target.arch,
        "devices": target.device_ids,
        "log_file": str(log_path),
        "started_at": time.time(),
    })
    save_worker_state(state)


def _wait_local_ready(args, config: WorkerConfig, process) -> Optional[dict]:
    deadline = time.time() + config.timing.ready_timeout
    while time.time() < deadline:
        status = _status(args.port, config.timing.ready_probe_timeout)
        if status and status.get("status") == "ready":
            return status
        if process.poll() is not None:
            return None
        time.sleep(0.5)
    return None


def _log_tail(log_path: Path) -> str:
    try:
        return "\n".join(
            log_path.read_text(errors="replace").splitlines()[-30:])
    except OSError as exc:
        logger.debug("Could not read worker log tail: %s", exc)
        return ""


def _start_local(args, cfg: WorkerConfig) -> int:
    existing = _status(args.port, cfg.timing.status_timeout)
    if existing and existing.get("status") == "ready":
        emit(json.dumps(existing, indent=2))
        return 0
    if live_worker_pid(args.port):
        _stop_local(args.port)

    target = _local_target(args, cfg)
    log_path = get_process_log_dir() / f"worker_{args.port}.log"
    environment = _local_worker_environment(args, cfg, target, log_path)
    process = _spawn_local_worker(environment, log_path)
    _record_local_worker(args, process, target, log_path)
    status = _wait_local_ready(args, cfg, process)
    if status is not None:
        emit(json.dumps(status, indent=2))
        return 0

    _stop_local(args.port)
    emit(f"worker failed to become ready; log={log_path}\n{_log_tail(log_path)}",
          file=sys.stderr)
    return 1


def _worker(args) -> int:
    cfg = WorkerConfig.load(args.config)
    args.port = args.port or cfg.port
    if args.remote_host:
        host_cfg = remote_dispatch.load_remote_host_config(
            args.remote_host, args.config
        )
        if host_cfg is None:
            emit(f"unknown remote worker host: {args.remote_host}", file=sys.stderr)
            return 2
        if args.start:
            return remote_dispatch.dispatch_start(
                remote_dispatch.RemoteStartRequest(
                    alias=args.remote_host,
                    host_cfg=host_cfg,
                    backend=args.backend,
                    arch=args.arch,
                    devices=args.devices,
                    port=args.port,
                    dsl=args.dsl,
                )
            )
        if args.stop:
            return remote_dispatch.dispatch_stop(args.remote_host, host_cfg,
                                                   args.port)
        return remote_dispatch.dispatch_status(
            args.remote_host, host_cfg, args.port,
            backend=args.backend, dsl=args.dsl,
        )
    if args.start:
        return _start_local(args, cfg)
    if args.stop:
        return _stop_local(args.port)
    status = _status(args.port, cfg.timing.status_timeout)
    if status is None:
        emit(f"worker 127.0.0.1:{args.port} is unreachable", file=sys.stderr)
        return 1
    emit(json.dumps(status, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="op-autoresearch")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="manage a local or remote worker")
    action = worker.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true")
    action.add_argument("--stop", action="store_true")
    action.add_argument("--status", action="store_true")
    worker.add_argument("--backend")
    worker.add_argument("--arch")
    worker.add_argument("--devices")
    worker.add_argument("--dsl")
    worker.add_argument("--host", default="127.0.0.1")
    worker.add_argument("--port", type=int)
    worker.add_argument("--remote-host")
    worker.add_argument("--config")
    worker.set_defaults(func=_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
