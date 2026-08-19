#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""opencode 运行时自检 + 进程组清理（G5/G6）。

G5 — 运行时安全网自检（可执行文件/行为探针 fail-closed，版本仅 advisory）：
  * 一次性屏障：threading.Lock + 记忆化（成功与失败都记忆化），并发 dispatch 不会在
    探针出结论前放行；
  * 探针超时写死（60s），超时 = 拒绝拉起；
  * JS runner 与 init.sh 同源判定：node 优先、bun 次之；
  * 版本建议线：取 opencode --version 输出第一段，按数字段 int 元组比较，
    预发布后缀截断并 warn；默认建议线 (1, 18, 18)（后端行为均以 1.18.18 实测锚定），
    可用 AOG_OPENCODE_MIN_VERSION 覆盖。低于建议线、无法解析或查询失败只告警，
    不因版本本身阻断 dispatch；
  * 行为探针 engine/src/opencode/probe_safety_net.mjs（deny/allow 成对）：
    OK → 放行；FAIL/超时 → 拒绝（fail-closed）；SKIP → 放行并告警（Phase O0 每次
    运行仍会以进程内 deny/allow 对复证）。

G6 — 进程组清理语义（平台明确）：
  * 子进程固定以新 session 启动（start_new_session=True）；
  * 清理主路径 = os.killpg(pgid)：TERM → 宽限 2s → KILL；
  * 不扫描 /proc：会话/进程组是唯一可靠且不会误伤无关进程的清理边界；
  * 清理函数绝不向调用方抛异常（记录 debug）。

纯标准库，可独立单测。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# G5 advisory version line. Backend behaviour (--auto semantics, debug CLI, stream events) is
# anchored on opencode 1.18.18 measurements; older versions are allowed with a visible warning.
DEFAULT_RECOMMENDED_VERSION = (1, 18, 18)
# G5 probe timeout (seconds). A probe that hangs is indistinguishable from a broken install.
PROBE_TIMEOUT_SEC = 60.0

_ENGINE_ROOT = Path(__file__).resolve().parents[4]  # backends/ -> orchestrator/ -> scripts/ -> src/ -> engine/
_PROBE_SCRIPT = _ENGINE_ROOT / "src" / "opencode" / "probe_safety_net.mjs"

# ---- one-shot barrier (G5/B8): both success AND failure are memoized --------------------
_PROBE_LOCK = threading.Lock()
_PROBE_STATE = {}  # (opencode binary, advisory version line) -> RuntimeCheck
_JS_RUNTIME = None  # type: str | None


@dataclass
class RuntimeCheck:
    """Result of the one-shot opencode runtime self-check."""
    ok: bool
    reason: str = ""
    warnings: list = field(default_factory=list)
    version: tuple | None = None
    js_runtime: str | None = None


def reset_runtime_state() -> None:
    """Test seam: forget the memoized probe verdict + runner choice."""
    global _PROBE_STATE, _JS_RUNTIME
    with _PROBE_LOCK:
        _PROBE_STATE = {}
        _JS_RUNTIME = None


def pick_js_runtime() -> str | None:
    """Canonical runner rule, shared with init.sh js_runtime(): node first, bun second."""
    global _JS_RUNTIME
    if _JS_RUNTIME is None:
        _JS_RUNTIME = shutil.which("node") or shutil.which("bun")
    return _JS_RUNTIME


def _parse_version_token(token: str) -> tuple | None:
    """Parse the first whitespace token of opencode --version into an int tuple.

    Leading 'v' is stripped; a prerelease/build suffix (after the first '-' or '+')
    is truncated and flagged by the caller. Numeric-segment comparison: '1.18.9' <
    '1.18.18' MUST hold (never string comparison).
    """
    token = token.strip().lstrip("vV")
    if "-" in token:
        token = token.split("-", 1)[0]
    if "+" in token:
        token = token.split("+", 1)[0]
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", token)
    if not m:
        return None
    parts = [int(x) for x in m.groups() if x is not None]
    return tuple(parts)


def _version_ge(got: tuple, floor: tuple) -> bool:
    for g, f in zip(got, floor):
        if g != f:
            return g > f
    return len(got) >= len(floor)


def _record_version_advisory(completed, recommended_version: tuple,
                             warnings: list) -> tuple | None:
    """Parse a completed version command and append non-blocking advisory warnings."""
    if completed.returncode != 0:
        warnings.append(
            f"opencode --version exited {completed.returncode}; version check is advisory"
        )
        return None
    tokens = (completed.stdout or "").split()
    if not tokens:
        warnings.append("opencode --version produced no output; version check is advisory")
        return None
    token = tokens[0]
    if "-" in token or "+" in token:
        warnings.append(f"opencode version token has prerelease/build suffix: {token!r}")
    version = _parse_version_token(token)
    if version is None:
        warnings.append(
            f"cannot parse opencode version token {token!r}; version check is advisory"
        )
    elif not _version_ge(version, recommended_version):
        warnings.append(
            f"opencode {'.'.join(map(str, version))} is below the recommended "
            f"{'.'.join(map(str, recommended_version))}; continuing because "
            "the version check is advisory"
        )
    return version


def _advisory_version_check(opencode_bin: str, recommended_version: tuple) -> tuple:
    """Return an optional hard executable failure plus advisory version metadata."""
    warnings: list = []
    try:
        completed = subprocess.run(
            [opencode_bin, "--version"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC, check=False,
        )
    except FileNotFoundError as e:
        return RuntimeCheck(ok=False, reason=f"opencode binary not found: {e}"), warnings, None
    except subprocess.TimeoutExpired:
        warnings.append(
            f"opencode --version timed out after {PROBE_TIMEOUT_SEC:.0f}s; "
            "continuing because the version check is advisory"
        )
        return None, warnings, None
    except OSError as e:
        warnings.append(f"opencode --version could not run ({e}); version check is advisory")
        return None, warnings, None
    return None, warnings, _record_version_advisory(completed, recommended_version, warnings)


def _run_safety_probe(js: str, warnings: list, version: tuple | None) -> RuntimeCheck:
    """Run the fail-closed deny/allow probe after executable availability is established."""
    try:
        probe = subprocess.run(
            [js, str(_PROBE_SCRIPT)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC,
            cwd=str(_ENGINE_ROOT), check=False,
        )
    except subprocess.TimeoutExpired:
        return RuntimeCheck(ok=False,
                            reason=f"safety-net probe timed out after {PROBE_TIMEOUT_SEC:.0f}s")
    except OSError as error:
        return RuntimeCheck(ok=False, reason=f"safety-net probe could not run: {error}")
    out = ((probe.stdout or "") + (probe.stderr or "")).strip()
    if probe.returncode == 0 and out == "OK":
        return RuntimeCheck(ok=True, reason="safety net ENFORCES",
                            warnings=warnings, version=version, js_runtime=js)
    if probe.returncode == 2 and out.startswith("SKIP:"):
        warnings.append(f"safety-net probe could not set up here ({out or 'SKIP'}) — "
                        "Phase O0 re-proves the deny/allow pair every run")
        return RuntimeCheck(ok=True, reason="safety-net setup skipped", warnings=warnings,
                            version=version, js_runtime=js)
    return RuntimeCheck(
        ok=False,
        reason=f"safety-net probe failed: {out or probe.returncode}",
        warnings=warnings,
        version=version,
        js_runtime=js,
    )


def _probe_once(opencode_bin: str, recommended_version: tuple) -> RuntimeCheck:
    """Run the advisory version check and mandatory safety-net proof under the barrier."""
    failure, warnings, version = _advisory_version_check(opencode_bin, recommended_version)
    if failure is not None:
        return failure

    # 2) behavioural probe (deny/allow pair). OK → pass; FAIL/timeout → refuse;
    #    SKIP → pass with warning (Phase O0 re-proves the pair in-process every run).
    if not _PROBE_SCRIPT.is_file():
        return RuntimeCheck(ok=False, reason=f"safety-net probe missing: {_PROBE_SCRIPT}")
    js = pick_js_runtime()
    if not js:
        return RuntimeCheck(
            ok=False,
            reason="node/bun runtime required for the behavioural safety-net probe",
            warnings=warnings,
            version=version,
            js_runtime=None,
        )
    # Keep this exact verdict contract in lockstep with init.sh: (0, "OK") is
    # pass; (2, "SKIP:…") is the only non-fatal setup result. A stray
    # diagnostic or mismatched exit status must not split installer/runtime.
    return _run_safety_probe(js, warnings, version)


def ensure_opencode_runtime(opencode_bin: str, *,
                            min_version: tuple | None = None) -> RuntimeCheck:
    """One-shot, concurrency-safe runtime self-check (G5). Memoized pass AND fail."""
    global _PROBE_STATE
    with _PROBE_LOCK:
        floor = min_version
        if floor is None:
            raw_floor = os.environ.get("AOG_OPENCODE_MIN_VERSION", "").strip()
            floor = _parse_version_token(raw_floor) if raw_floor else None
            floor = floor or DEFAULT_RECOMMENDED_VERSION
        key = (str(opencode_bin), tuple(floor))
        if key not in _PROBE_STATE:
            _PROBE_STATE[key] = _probe_once(opencode_bin, floor)
        return _PROBE_STATE[key]


# ---- G6 process-group cleanup -------------------------------------------------------------

def spawn_new_session_kwargs() -> dict:
    """Popen kwargs putting the child in its own session: the cleanup contract (G6)."""
    return {"start_new_session": True}


def terminate_process_group(proc, *, grace_sec: float = 2.0) -> None:
    """Kill the child's whole process group; NEVER raises to the caller (G6).

    The child is started with start_new_session=True, so os.killpg is portable and
    covers the child plus normal descendants.  Deliberately do not scrape /proc:
    after reaping the parent that evidence may be gone, and parsing proc stat fields
    can target an unrelated process.  Every path waits after SIGKILL to avoid zombies.
    """
    pid = getattr(proc, "pid", None)
    if not pid:
        return
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return  # already reaped

    def _signal(sig):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return

    _signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_sec)
        return
    except Exception as error:
        _log.debug("Recoverable operation failed.", exc_info=error)
    _signal(signal.SIGKILL)
    try:
        proc.wait(timeout=grace_sec)
    except Exception as error:
        _log.debug("Recoverable operation failed.", exc_info=error)
