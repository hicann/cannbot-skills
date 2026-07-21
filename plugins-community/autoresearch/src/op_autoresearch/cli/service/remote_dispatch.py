# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""``op-autoresearch worker --remote-host`` dispatch — thin orchestration layer.

Responsibilities: idempotent ``--start``, ``--stop``, ``--status``,
``--reconnect``. All heavy lifting delegated to siblings:

  - ``tunnel.py``           — local ssh -L lifecycle, port ownership
  - ``remote_probe.py``     — one-shot SSH probe → raw facts
  - ``diagnostics.py``      — facts → ``list[Finding]`` + stderr render

Module name was ``worker_remote`` previously — too easy to confuse with
``core/worker/remote_worker.py`` (the HTTP client class). Renamed so the
two layers can't be mistyped into each other.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

from op_autoresearch.core.worker.eval_config import eval_defaults
from op_autoresearch.utils.console import emit

from .diagnostics import classify, has_fatal, render_findings
from .remote_env import source_env_script_bash
from .remote_probe import RemoteProbeRequest, probe_remote
from .tunnel import kill_pid_hint, tunnel_start, tunnel_stop_silent, who_holds_port
from .worker_config import WorkerConfig, WorkerTiming, worker_timing


@dataclass(frozen=True)
class RemoteStartRequest:
    alias: str
    host_cfg: dict
    backend: Optional[str]
    arch: Optional[str]
    devices: Optional[str]
    port: int
    dsl: Optional[str] = None


@dataclass
class StartContext:
    request: RemoteStartRequest
    ssh_alias: str
    log_file: str
    env_script: Optional[str]
    repo_path: str
    backend: Optional[str]
    dsl: Optional[str]
    timing: WorkerTiming
    probe_device_ids: Optional[list[int]]
    arch: Optional[str]
    devices: Optional[str]

# Back-compat thin wrappers — misc.py / eval_bridge.py still import these
# names. New code should construct ``WorkerConfig.load(...)`` directly.


def load_remote_host_config(alias: str,
                            config_path: Optional[str]) -> Optional[dict]:
    return WorkerConfig.load(config_path).host(alias)


def load_default_port(config_path: Optional[str]) -> Optional[int]:
    return WorkerConfig.load(config_path).port


# ---------------------------------------------------------------------------
# HTTP probes (local-tunnel-side)
# ---------------------------------------------------------------------------


def _curl_status(host: str, port: int,
                 timeout: Optional[float] = None) -> Optional[dict]:
    """``/api/v1/status`` probe. ``timeout`` defaults to ``status_timeout``
    from config —— the ready loop should explicitly pass
    ``ready_probe_timeout`` instead since the two have different roles.
    """
    import urllib.request
    if timeout is None:
        timeout = worker_timing().status_timeout
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/status", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _is_ready(st: Optional[dict]) -> bool:
    """True iff ``/status`` 返回 dict 且 status 字段是 ready/ok。daemon
    刚 spawn 时 server.py 返回 ``initializing``（HTTP 已通但 worker 还
    没装好）——这个状态既不能跳过 spawn，也不能算 poll-loop 完成。
    """
    if not isinstance(st, dict):
        return False
    return str(st.get("status", "")).lower() in ("ready", "ok")


def _curl_health(host: str, port: int,
                 timeout: Optional[float] = None) -> Optional[dict]:
    """``/health``：非阻塞 device queue 探活。

    ``timeout`` 默认比 daemon 侧 health_timeout 多留一段 client 余量。
    传输失败或老 daemon 缺少 endpoint 时返回 None。
    """
    import urllib.request
    if timeout is None:
        timing = worker_timing()
        timeout = (
            max(timing.status_timeout, timing.health_timeout)
            + timing.http_read_margin
        )
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/health", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Remote spawn helpers
# ---------------------------------------------------------------------------


def _build_remote_start_cmd(context: StartContext) -> str:
    """Compose the bash payload sent over SSH to spawn the daemon. The
    recursive ``op-autoresearch`` on remote goes through the local branch of
    ``worker_cmd`` → ``worker_service.start``, which Popen-detaches the
    daemon (``preexec_fn=os.setsid`` + ``stdin=DEVNULL``) so this SSH
    returns promptly.

    ``PYTHONPATH`` is pinned to ``<repo_path>/src`` so the
    daemon runs the checkout source, not whatever pip pinned.

    ``env_script`` may contain plain ``conda activate``; bootstrap the
    conda shell hook before sourcing it so non-interactive SSH behaves like
    the user's login shell.

    worker.* timing 通过 env 透传，所以远端 ``worker_service.start`` 不
    再硬编码固定启动等待值 —— config.yaml worker.* 一处改、本机和递归远端都
    生效。
    """
    repo_path = context.repo_path
    env_script = context.env_script

    parts: list = [source_env_script_bash(env_script)]
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/src:"
        f"${{PYTHONPATH:-}}"
    )
    # daemon 只绑 loopback（tunnel 转发 :<port> 到远端 127.0.0.1）。
    parts.append("export WORKER_HOST=127.0.0.1")
    # 递归远端 op-autoresearch 跳过启动表格和心跳噪声；本机命令负责用户可见输出。
    parts.append("export OP_AUTORESEARCH_CLI_QUIET=1")
    for key, value in context.timing.as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    for key, value in eval_defaults().as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    parts.append(
        " ".join([
            "python", "-m", "op_autoresearch.cli.cli", "worker",
            "--start",
            "--backend", shlex.quote(context.backend or "ascend"),
            "--arch", shlex.quote(context.arch or ""),
            "--devices", shlex.quote(context.devices or "0"),
            "--port", str(context.request.port),
        ])
    )
    return "\n".join(parts)


def _build_remote_stop_cmd(host_cfg: dict, port: int) -> str:
    """Compose exact daemon termination plus predecessor-tree cleanup.

    A single SIGTERM is not a completed stop: Uvicorn waits for an in-flight
    eval, while that eval may run for many minutes.  Escalate the one listener
    PID after the configured NPU teardown grace, then invoke the shared,
    PID-fingerprinted eval-group reaper from the same checkout/environment.
    """
    repo_path = host_cfg["repo_path"]
    env_script = host_cfg.get("env_script")
    defaults = eval_defaults()
    polls = max(1, int(defaults.kill_grace_s * 10) + 1)
    registry = f"/tmp/op_autoresearch_worker_{port}_process_groups.json"
    state_lookup = (
        "from op_autoresearch.cli.utils.worker_state import live_worker_pid; "
        f"print(live_worker_pid({port}) or '')"
    )
    cleanup = (
        "import json; "
        "from op_autoresearch.utils.process_utils import "
        "reap_orphaned_process_groups; "
        "from op_autoresearch.cli.utils.worker_state import "
        "load_worker_state, remove_worker_entry, save_worker_state; "
        "reaped = reap_orphaned_process_groups(); "
        "state = load_worker_state(); "
        f"remove_worker_entry(state, {port}); "
        "save_worker_state(state); "
        "print(json.dumps({'reaped_process_groups': reaped}))"
    )
    parts: list[str] = [source_env_script_bash(env_script)]
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/src:"
        f"${{PYTHONPATH:-}}"
    )
    for key, value in defaults.as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    parts.append(
        f"export OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY={shlex.quote(registry)}"
    )
    parts.extend([
        f'listener_pid="$(lsof -tiTCP:{port} -sTCP:LISTEN | head -n 1)"',
        f"state_pid=\"$(python -c {shlex.quote(state_lookup)})\"",
        'pid="${listener_pid:-$state_pid}"',
        (
            'if [ -n "$pid" ]; then '
            'kill -TERM "$pid" 2>/dev/null || true; '
            f'for _ in $(seq 1 {polls}); do '
            'kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done; '
            'if kill -0 "$pid" 2>/dev/null; then '
            'kill -KILL "$pid" 2>/dev/null || true; fi; '
            'fi'
        ),
        f"python -c {shlex.quote(cleanup)}",
    ])
    return "\n".join(parts)


def _ssh_dispatch(ssh_alias: str, bash_cmd: str) -> int:
    """SSH-run bash_cmd on alias，stdout 透传给本机终端（让远端 op-autoresearch
    递归 print 流回来）。``-o LogLevel=ERROR`` 抑制 SSH banner 和无关
    RemoteForward warning，保留真实 ssh 错误。
    """
    ssh = shutil.which("ssh")
    if ssh is None:
        raise FileNotFoundError("ssh was not found on PATH")
    return subprocess.call([
        ssh, "-o", "LogLevel=ERROR",
        ssh_alias, f"bash -lc {shlex.quote(bash_cmd)}",
    ])


def _step(msg: str) -> None:
    """Step log to stderr with ``flush=True`` — Windows terminals
    occasionally buffer stderr until the producer terminates, which makes
    a 30s probe look "frozen". Flushing per line eliminates that.
    """
    emit(f"[op-autoresearch] {msg}", file=sys.stderr, flush=True)


def _device_ids_from_arg(devices: Optional[str]) -> Optional[list[int]]:
    if devices is None:
        return None
    ids = [int(p.strip()) for p in str(devices).split(",") if p.strip()]
    return ids or None


def _start_context(request: RemoteStartRequest) -> StartContext:
    config = WorkerConfig.load(None)
    return StartContext(
        request=request,
        ssh_alias=request.host_cfg.get("ssh_alias") or request.alias,
        log_file=f"/tmp/op_autoresearch_worker_{request.port}.log",
        env_script=request.host_cfg.get("env_script"),
        repo_path=request.host_cfg.get("repo_path", ""),
        backend=request.backend or config.backend,
        dsl=request.dsl or config.dsl,
        timing=worker_timing(),
        probe_device_ids=_device_ids_from_arg(request.devices),
        arch=request.arch,
        devices=request.devices,
    )


def _probe_request(context: StartContext) -> RemoteProbeRequest:
    return RemoteProbeRequest(
        ssh_alias=context.ssh_alias,
        env_script=context.env_script,
        port=context.request.port,
        log_file=context.log_file,
        repo_path=context.repo_path,
        device_ids=context.probe_device_ids,
    )


def _probe_start_host(context: StartContext) -> dict:
    return probe_remote(_probe_request(context))


def _render_start_diagnostics(context: StartContext, facts: dict) -> None:
    findings = classify(
        facts,
        context.request.port,
        backend=context.backend,
        dsl=context.dsl,
        for_start=True,
    )
    render_findings(findings, facts.get("LOG_TAIL", ""))


def _print_ready_status(status: dict) -> None:
    emit(json.dumps(status, indent=2, ensure_ascii=False))


def _reuse_existing_daemon(context: StartContext) -> Optional[int]:
    port = context.request.port
    _step(f"[1/4] probing 127.0.0.1:{port}/api/v1/status ...")
    status = _curl_status(
        "127.0.0.1", port, timeout=context.timing.status_timeout)
    if _is_ready(status):
        _step("[1/4] daemon already ready; nothing to do")
        _print_ready_status(status)
        return 0
    _step(f"[2/4] rebuilding local ssh -L :{port} -> "
          f"{context.ssh_alias} ...")
    tunnel_stop_silent(port, context.ssh_alias)
    tunnel_pid = tunnel_start(context.ssh_alias, port)
    if tunnel_pid == 0:
        _step("[2/4] tunnel failed; running reverse SSH diagnostics")
        _render_start_diagnostics(context, _probe_start_host(context))
        return 1
    _step(f"[2/4] tunnel pid={tunnel_pid}; probing /status again ...")
    status = _curl_status(
        "127.0.0.1", port, timeout=context.timing.status_timeout)
    if _is_ready(status):
        _step("[2/4] remote daemon was already running; complete")
        _print_ready_status(status)
        return 0
    return None


def _validate_start_host(context: StartContext) -> Optional[int]:
    _step("[3/4] remote diagnostics (env/backend/dsl/disk/port/log) ...")
    facts = _probe_start_host(context)
    findings = classify(
        facts,
        context.request.port,
        backend=context.backend,
        dsl=context.dsl,
        for_start=True,
    )
    if has_fatal(findings):
        _step("[3/4] fatal preflight finding; daemon will not be started")
        render_findings(findings, facts.get("LOG_TAIL", ""))
        return 1
    context.backend = context.backend or "ascend"
    if context.arch is None:
        context.arch = (facts.get("ARCH") or "").strip().lower() or None
    if context.arch is None:
        _step("[3/4] could not infer Ascend architecture; pass --arch")
        render_findings(findings, facts.get("LOG_TAIL", ""))
        return 1
    context.devices = context.devices or "0"
    _step(f"[3/4] probe OK: backend={context.backend}, "
          f"arch={context.arch}, devices={context.devices}, "
          f"dsl={context.dsl or '(any)'}")
    return None


def _wait_for_remote_ready(context: StartContext) -> int:
    port = context.request.port
    deadline = time.time() + context.timing.ready_timeout
    last_beat = time.time()
    while time.time() < deadline:
        status = _curl_status(
            "127.0.0.1", port,
            timeout=context.timing.ready_probe_timeout,
        )
        if _is_ready(status):
            _step("[4/4] /status ready; complete")
            _print_ready_status(status)
            return 0
        now = time.time()
        if now - last_beat >= context.timing.ready_poll_interval:
            elapsed = int(now - deadline + context.timing.ready_timeout)
            _step(f"   /status not ready ({elapsed}s/"
                  f"{context.timing.ready_timeout}s) ...")
            last_beat = now
        time.sleep(1)
    _step(f"[4/4] /status not ready after "
          f"{context.timing.ready_timeout}s; rerunning diagnostics")
    _render_start_diagnostics(context, _probe_start_host(context))
    return 1


def _launch_remote_daemon(context: StartContext) -> int:
    _step(f"[4/4] starting remote daemon on "
          f"{context.ssh_alias}:{context.request.port} ...")
    return_code = _ssh_dispatch(
        context.ssh_alias, _build_remote_start_cmd(context))
    if return_code != 0:
        _step(f"[4/4] remote daemon launch rc={return_code}; diagnosing")
        _render_start_diagnostics(context, _probe_start_host(context))
        return return_code
    _step(f"[4/4] daemon spawned; polling /status ready "
          f"(up to {context.timing.ready_timeout}s) ...")
    return _wait_for_remote_ready(context)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_start(request: RemoteStartRequest) -> int:
    """SSH-dispatch worker --start with idempotent recovery.

    Flow: probe /status → rebuild tunnel + reprobe → run diagnostic probe
    (also fills missing CLI defaults) → spawn daemon → poll /status
    with heartbeat. Any fatal finding aborts before spawn.

    ``backend / arch / devices`` may be None. Arch is filled from ``npu-smi``
    unless the caller passes ``--arch``. ``dsl`` drives diagnostics:
    Triton requires its runtime, while AscendC and CATLASS do not. Pass None
    to read ``defaults.dsl`` from config.yaml.
    """
    if "repo_path" not in request.host_cfg:
        _step(f"remote_worker.hosts.{request.alias} missing repo_path")
        return 2
    context = _start_context(request)

    # Resolve all defaults up front so every classify call sees a
    # consistent effective_backend / dsl / timing — without this,
    # tunnel-fail diagnostics used user-passed `backend` (often None)
    # and read different defaults than the probe-success path.

    reused = _reuse_existing_daemon(context)
    if reused is not None:
        return reused

    invalid = _validate_start_host(context)
    if invalid is not None:
        return invalid

    return _launch_remote_daemon(context)


def dispatch_stop(alias: str, host_cfg: dict, port: int) -> int:
    """Tear down the tunnel, stop the exact listener, reap its eval trees."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    emit(f"[op-autoresearch] tore down local tunnel for :{port}")
    if "repo_path" not in host_cfg:
        emit(f"[op-autoresearch] remote_worker.hosts.{alias} 缺 repo_path",
              file=sys.stderr)
        return 2
    rc = _ssh_dispatch(ssh_alias, _build_remote_stop_cmd(host_cfg, port))
    if rc != 0:
        emit(f"[op-autoresearch] remote daemon stop rc={rc}", file=sys.stderr)
        return rc
    emit(f"[op-autoresearch] stopped remote daemon and reaped owned eval trees on "
          f"{ssh_alias}:{port}")
    return 0


def _report_unreachable_local_port(port: int) -> None:
    holder = who_holds_port(port)
    if holder is None:
        emit(
            f"Worker 127.0.0.1:{port} is unreachable and the port is free; "
            "run --start."
        )
        return
    emit(
        f"Worker 127.0.0.1:{port} is unreachable; the port is held by "
        f"PID={holder['pid']}\n"
        f"  cmdline: {holder['cmdline'][:120]}\n"
        f"  stop the stale tunnel with {kill_pid_hint(holder['pid'])}, "
        "then run --start"
    )


def _probe_unreachable_remote(
    alias: str,
    host_cfg: dict,
    port: int,
    backend: Optional[str],
    dsl: Optional[str],
) -> None:
    ssh_alias = host_cfg.get("ssh_alias") or alias
    if ssh_alias == "local":
        return
    config = WorkerConfig.load(None)
    facts = probe_remote(
        RemoteProbeRequest(
            ssh_alias=ssh_alias,
            env_script=host_cfg.get("env_script"),
            port=port,
            log_file=f"/tmp/op_autoresearch_worker_{port}.log",
            repo_path=host_cfg.get("repo_path"),
        )
    )
    render_findings(
        classify(
            facts,
            port,
            backend=backend or config.backend,
            dsl=dsl or config.dsl,
            for_start=False,
        ),
        facts.get("LOG_TAIL", ""),
    )


def _health_summary(health: dict) -> dict:
    return {
        "healthy": bool(health.get("healthy")),
        "probed_device": health.get("probed_device"),
        "free": health.get("free"),
        "note": health.get("note"),
        "error": health.get("error"),
    }


def dispatch_status(
    alias: str,
    host_cfg: dict,
    port: int,
    *,
    backend: Optional[str] = None,
    dsl: Optional[str] = None,
) -> int:
    """Display tunneled worker status and health diagnostics."""
    status = _curl_status("127.0.0.1", port)
    if status is None:
        _report_unreachable_local_port(port)
        _probe_unreachable_remote(
            alias, host_cfg, port, backend, dsl
        )
        return 1

    health = _curl_health("127.0.0.1", port)
    output = dict(status)
    if health is not None:
        output["health"] = _health_summary(health)
    emit(json.dumps(output, indent=2, ensure_ascii=False))
    if health is None or health.get("healthy"):
        return 0
    emit(
        "\n[op-autoresearch] /status is available but /health is "
        f"degraded: {health.get('error')!r}",
        file=sys.stderr,
    )
    return 1


def dispatch_reconnect_tunnel(alias: str, host_cfg: dict, port: int) -> int:
    """Rebuild only the local tunnel; leave remote daemon alone. Use when
    a long batch silently lost its tunnel (server-side SSH reset / network
    drop) but the daemon is still alive. Falls back to --stop+--start if
    the daemon is also gone.
    """
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if pid:
        emit(f"[op-autoresearch] ssh -L 127.0.0.1:{port} → "
              f"{ssh_alias}:{port} reconnected (tunnel pid={pid})")
    st = _curl_status("127.0.0.1", port)
    if st is None:
        emit(
            "[op-autoresearch] /status 仍不通；daemon 可能也已停 — 用 --stop + --start。",
            file=sys.stderr,
        )
        return 1
    emit(json.dumps(st, indent=2, ensure_ascii=False))
    return 0
