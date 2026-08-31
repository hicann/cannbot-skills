# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Shared A5 target transport primitives.

Endpoint resolution (``.ascendc_env`` driven) plus the ssh/scp/sudo/run
wrappers used by the target-evaluation transports.  Extracted from the
removed ``model_reference_target.py`` so ``npubench_target.py`` keeps a
supported import source after the model_reference golden path was excised.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class A5TargetTransportError(RuntimeError):
    """A5 target endpoint resolution or remote command execution failed."""


@dataclass(frozen=True)
class _Target:
    """Resolved A5 endpoint without credentials in any persisted artifact."""

    name: str
    host: str
    user: str
    password: str
    container: str
    cann_path: str
    benchmark_root: str
    host_mode: bool
    visible_device: int
    ssh_options: tuple[str, ...]
    env: Mapping[str, str]


def _resolve_target(workspace: Path, lane: int) -> _Target:
    """Resolve the same A5 endpoint/configuration that O5 uses.

    The O5 helpers are bound by ``from ... import`` *inside* this function on
    purpose: the late import keeps the phase_o5 import cycle broken and makes
    every call re-resolve the names, so UT monkeypatching of the O5 module
    attributes still takes effect.
    """
    from phase_o5_runner import (
        _a5_build_cann_path, _a5_build_container, _a5_build_host,
        _lane_aware_benchmark_root, _normalise_target, _read_ascendc_env,
        _resolve_ssh_key_opts,
    )
    from phase_o5_verify import _resolve_visible_device

    env = _read_ascendc_env(workspace)
    if not env:
        raise A5TargetTransportError("missing .ascendc_env for A5 target resolution")
    _target_lower, target_name = _normalise_target(env)
    host = _a5_build_host(env, workspace, target_name)
    container = _a5_build_container(env, workspace, target_name)
    if not host and container.lower() != "local":
        raise A5TargetTransportError(f"missing {target_name}_HOST (and no A5_HOST fallback)")
    cann_path = _a5_build_cann_path(env, workspace, target_name)
    benchmark_root = _lane_aware_benchmark_root(env, lane)
    host_mode = str(env.get("A5_HOST_MODE", "")).strip().lower() in (
        "1", "true", "yes",
    )
    # A direct local target does not need (and may not permit) an SSH health
    # probe.  Its lane is already the target-visible device.  Remote targets
    # retain O5's health-aware device routing.
    visible_device = (
        lane if container.lower() == "local" else _resolve_visible_device(env, workspace, lane)
    )
    ssh_options = tuple(
        [
            *_resolve_ssh_key_opts(env, target_name),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=60",
            "-o", "LogLevel=ERROR",
            "-o", "ControlMaster=no",
        ]
    )
    return _Target(
        name=target_name,
        host=host,
        user=env.get(f"{target_name}_USER") or env.get("A5_USER", "root"),
        password=env.get(f"{target_name}_PASSWORD") or env.get("A5_PASSWORD", ""),
        container=container,
        cann_path=cann_path,
        benchmark_root=benchmark_root,
        host_mode=host_mode,
        visible_device=visible_device,
        ssh_options=ssh_options,
        env=dict(env),
    )


def _ssh_command(remote_command: str, target: _Target) -> list[str]:
    login = f"{target.user}@{target.host}"
    command = ["ssh", *target.ssh_options, login, remote_command]
    return ["sshpass", "-p", target.password, *command] if target.password else command


def _scp_command(source: str, destination: str, target: _Target) -> list[str]:
    command = ["scp", *target.ssh_options, source, destination]
    return ["sshpass", "-p", target.password, *command] if target.password else command


def _maybe_sudo(command: str, target: _Target) -> str:
    # Late, function-local binding (see _resolve_target): keeps the import cycle
    # broken and re-resolves the name per call so UT patches still apply.
    from phase_o5_runner import _maybe_sudo_wrap_remote

    return _maybe_sudo_wrap_remote(command, dict(target.env), target.name)


def _run(command: list[str], *, timeout: int, what: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise A5TargetTransportError(f"{what} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise A5TargetTransportError(f"{what} tool is unavailable: {exc}") from exc
