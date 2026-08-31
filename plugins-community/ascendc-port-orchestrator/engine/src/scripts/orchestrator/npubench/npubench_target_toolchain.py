# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Target toolchain resolution and the controlled direct-build process.

This module owns everything the controlled build needs from the machine it runs
on: which Python and CANN ``set_env.sh`` to use, which ``npu-smi`` binary is
approved, the device SoC probe, the pre-CMake runtime preflight, and the
session-isolated build process whose whole process group can be stopped on a
timeout.
"""
from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from a5_target_capability import a5_soc_version, parse_npu_smi_soc, soc_product_family
from a5_target_transport import _Target
from npubench.npubench_target_base import TargetTransportError, _DirectBuildTimeout


def _run_direct_build_process(
    shell_command: str,
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    """Run the build in its own session so timeout can stop every descendant.

    The build genuinely needs a login shell: it must source the CANN
    ``set_env.sh`` and chain the build invocation after it.  The interpreter is
    the approved absolute bash path and every value interpolated into
    ``shell_command`` by the caller is ``shlex.quote``d or ``int()``-cast, so no
    caller-controlled shell metacharacter can reach the shell.
    """
    command = [_bash_binary(), "-lc", shell_command]
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_direct_build_process_group(process)
        raise _DirectBuildTimeout(
            timeout_sec,
            _process_output_text(stdout if stdout is not None else exc.stdout),
            _process_output_text(stderr if stderr is not None else exc.stderr),
        ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _terminate_direct_build_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate then forcibly reap the isolated direct-build process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def _process_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _target_python(env: Mapping[str, Any]) -> Path:
    explicit = env.get("A5_PYTHON") or env.get("A5_HOST_PYTHON")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit.strip()).expanduser()
    directory = env.get("A5_NPU_PYTHON_BIN") or env.get("NPU_PYTHON_BIN")
    if isinstance(directory, str) and directory.strip():
        candidate = Path(directory.strip()).expanduser()
        if candidate.is_dir():
            return candidate / "python3"
        return candidate
    return Path(sys.executable)


def _resolve_cann_set_env(cann_path: Path) -> Path | None:
    """Resolve the active CANN environment script without trusting stale config."""
    candidates = [
        cann_path / "set_env.sh",
        Path("/usr/local/Ascend/cann/set_env.sh"),
        Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"),
    ]
    for candidate in candidates:
        try:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _npu_smi_command(target: _Target) -> str:
    configured = target.env.get("A5_NPU_SMI_BIN") or target.env.get("NPU_SMI_BIN")
    if isinstance(configured, str) and configured.strip():
        candidate = Path(configured.strip()).expanduser()
        if candidate.is_symlink() or not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise TargetTransportError(
                f"configured npu-smi is unavailable or unsafe: {candidate}"
            )
        return str(candidate.resolve())
    for candidate in (Path("/usr/local/bin/npu-smi"), Path("/usr/bin/npu-smi")):
        if candidate.is_file() and not candidate.is_symlink() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise TargetTransportError(
        "npu-smi is unavailable at an approved absolute path; refusing PATH lookup"
    )


def _positive_timeout_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        raise TargetTransportError(f"{name} must be a positive integer, got an empty value")
    try:
        timeout = int(value)
    except ValueError as exc:
        raise TargetTransportError(
            f"{name} must be a positive integer, got {raw!r}"
        ) from exc
    if timeout <= 0:
        raise TargetTransportError(f"{name} must be greater than zero, got {raw!r}")
    return timeout


def _bash_binary() -> str:
    """Return an approved absolute bash path (never rely on PATH lookup)."""
    for candidate in (Path("/bin/bash"), Path("/usr/bin/bash")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise TargetTransportError("bash is unavailable at an approved absolute path")


def _soc_probe_argv(target: _Target, set_env: Path | None) -> list[str]:
    """Build the npu-smi probe argv, using a shell only to source ``set_env``.

    A shell is genuinely required only when a CANN ``set_env.sh`` has to be
    sourced into the probe environment.  Without one the approved absolute
    ``npu-smi`` binary is executed directly as argv, so no shell is involved at
    all.  In the sourcing case the interpreter is the absolute bash path, the
    two interpolated strings are ``shlex.quote``d and the device index is
    ``int()``-cast, so no caller-controlled shell metacharacter can reach it.
    """
    npu_smi = _npu_smi_command(target)
    device = int(target.visible_device)
    if set_env is None:
        return [npu_smi, "info", "-t", "board", "-i", str(device)]
    command = f"{shlex.quote(npu_smi)} info -t board -i {device}"
    command = f"source {shlex.quote(str(set_env))} >/dev/null 2>&1 && {command}"
    return [_bash_binary(), "-lc", command]


def _run_soc_probe(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=_positive_timeout_from_env("CANNBOT_DIRECT_SOC_PROBE_TIMEOUT_SEC", 30),
    )


def _soc_probe_result(
    target: _Target, argv: Sequence[str]
) -> subprocess.CompletedProcess[str] | None:
    """Run the probe with one retry; ``None`` means 'trust the configured SOC'."""
    try:
        return _run_soc_probe(argv)
    except subprocess.TimeoutExpired:
        # Shared-infra contention (2026-08-22, source-migration flow): npu-smi
        # can hang when concurrent campaigns saturate the device manager.
        # Retry once below, then trust the configured SOC policy (authoritative
        # in .ascendc_env) so a healthy build is not blocked by probe latency.
        pass
    except (OSError, subprocess.SubprocessError) as exc:
        raise TargetTransportError(
            f"direct-launch target SoC probe could not start: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        return _run_soc_probe(argv)
    except (OSError, subprocess.SubprocessError) as exc:
        configured = a5_soc_version(target.env)
        if configured:
            logging.getLogger(__name__).warning(
                "direct-launch SoC probe timed out twice under contention; "
                "trusting configured SOC %s", configured
            )
            return None
        raise TargetTransportError(
            f"direct-launch target SoC probe could not start: {type(exc).__name__}: {exc}"
        ) from exc


def _chip_name_from_board_output(output: str) -> str:
    """Read ``Chip Name`` from the 25.1.rc1 per-device board listing.

        NPU ID    : 5
        Chip Name : Ascend950DT
    """
    for line in output.splitlines():
        if "Chip Name" not in line:
            continue
        parts = line.split(":")
        if len(parts) >= 2:
            name = parts[-1].strip().split()
            if name:
                return name[0]
    return ""


def _probe_target_soc(target: _Target, set_env: Path | None) -> str:
    """Bind configured SoC policy to the target device's reported model."""
    # Per-device board query.  The FULL npu-smi info listing iterates
    # every physical device and hangs as soon as one unrelated card wedges
    # its management path (2026-08-22 A5 campaign: device 4 died Critical
    # and froze the whole listing; lanes 1/2/3 were healthy but every
    # full-listing probe timed out at 30s).  The per-device query stays
    # responsive for healthy devices.
    completed = _soc_probe_result(target, _soc_probe_argv(target, set_env))
    if completed is None:
        return a5_soc_version(target.env)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        suffix = detail[-1] if detail else "no diagnostic"
        raise TargetTransportError(
            f"direct-launch target SoC probe failed (rc={completed.returncode}): {suffix}"
        )
    output = completed.stdout or ""
    observed = parse_npu_smi_soc(output, target.visible_device)
    if not observed:
        observed = _chip_name_from_board_output(output)
    configured_family = soc_product_family(a5_soc_version(target.env))
    observed_family = soc_product_family(observed)
    if not observed or observed_family is None:
        raise TargetTransportError(
            "direct-launch target SoC probe returned no recognized model for "
            f"device {target.visible_device}"
        )
    if configured_family != observed_family:
        raise TargetTransportError(
            "direct-launch target SoC does not match configuration: "
            f"configured={a5_soc_version(target.env)!r}, observed={observed!r}, "
            f"device={target.visible_device}"
        )
    return observed


def _direct_build_preflight(python: Path, target: _Target) -> str | None:
    """Fail before CMake with an actionable target-runtime diagnostic."""
    if not python.is_file() or not os.access(python, os.X_OK):
        return f"direct candidate target Python is unavailable or not executable: {python}"
    cann_path = Path(target.cann_path).expanduser()
    if not cann_path.is_dir():
        return f"direct candidate target CANN path does not exist: {cann_path}"
    probe = (
        "import importlib.util,sysconfig; "
        "mods=('torch','torch_npu'); "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "assert not missing, 'missing Python modules: '+','.join(missing); "
        "inc=sysconfig.get_path('include'); "
        "assert inc and __import__('os').path.isfile(__import__('os').path.join(inc,'Python.h')), "
        "'Python development headers are missing'"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            text=True,
            capture_output=True,
            check=False,
            timeout=_positive_timeout_from_env("CANNBOT_DIRECT_PREFLIGHT_TIMEOUT_SEC", 60),
        )
    except (TargetTransportError, OSError, subprocess.SubprocessError) as exc:
        return f"direct candidate target Python preflight could not start: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = detail[-1] if detail else "no diagnostic"
        return (
            f"direct candidate target Python preflight failed for {python}: {suffix}; "
            "install/use a target environment containing torch, torch_npu and Python.h"
        )
    return None
