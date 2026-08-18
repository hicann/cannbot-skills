# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Local evaluation client (stdlib-only, no network).

Replaces the former remote HTTP evaluation service. For one operator working
copy it:

  1. builds it (operator project's own build step — build/run are separable);
  2. runs it under ``msprof`` (only the run step is wrapped, so compile time
     never pollutes the kernel µs);
  3. parses ``Task Duration(us)`` (= score) and aic-metrics into a
     ``performance_metrics`` dict.

The returned shape matches what the old service returned, so miner's downstream
logic is unchanged:

    {success, score, duration, device_id, performance_metrics, error_info}

The operator project is expected to expose separable build/run entry points
(``run.sh``/``build.sh``/``run.py``/CMake). When auto-detection does not fit a
project's convention, pass explicit ``build_cmd`` / ``run_cmd``.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from . import msprof_collect

BUILD_TIMEOUT: int = 3600
FAIL_SCORE: float = -10000.0


class _MsprofOptions(NamedTuple):
    """msprof collection knobs, bundled so ``_collect`` stays within the arg-count limit."""
    metric_groups: Tuple[str, ...]
    quick: bool
    device: Optional[int]


# ---------------------------------------------------------------------------
# build / run entry-point detection
# ---------------------------------------------------------------------------

def _as_argv(cmd) -> List[str]:
    if isinstance(cmd, (list, tuple)):
        return [str(c) for c in cmd]
    return shlex.split(str(cmd))


def detect_build_cmd(op_dir: str) -> Optional[List[str]]:
    """Detect the operator project's build entry point (build/run separable)."""
    if os.path.isfile(os.path.join(op_dir, "build.sh")):
        return ["bash", "build.sh"]
    if os.path.isfile(os.path.join(op_dir, "run.sh")):
        return ["bash", "run.sh", "build"]
    if os.path.isfile(os.path.join(op_dir, "CMakeLists.txt")):
        return ["bash", "-c", "cmake -B build && cmake --build build -j"]
    return None


def detect_run_cmd(op_dir: str) -> Optional[List[str]]:
    """Detect the operator project's run entry point (the step msprof wraps)."""
    if os.path.isfile(os.path.join(op_dir, "run.sh")):
        return ["bash", "run.sh", "run"]
    if os.path.isfile(os.path.join(op_dir, "run.py")):
        return ["python3", "run.py"]
    for exe in ("main", "test", "run"):
        cand = os.path.join(op_dir, exe)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return ["./%s" % exe]
    return None


# ---------------------------------------------------------------------------
# build + collect
# ---------------------------------------------------------------------------

def _run_build(op_dir: str, build_cmd: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            build_cmd, cwd=op_dir, capture_output=True, text=True, timeout=BUILD_TIMEOUT
        )
    except FileNotFoundError as exc:
        return False, "build command not found: %s" % exc
    except subprocess.SubprocessError as exc:
        return False, "build error: %s" % exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        return False, "build exited %d: %s" % (proc.returncode, tail)
    return True, ""


def _fail(
    op_name: str, device_id: Optional[int], error_info: str, *,
    kernel_breakdown: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "success": False,
        "op_name": op_name,
        "score": FAIL_SCORE,
        "duration": None,
        "device_id": device_id,
        "performance_metrics": {},
        "error_info": error_info,
        "kernel_breakdown": kernel_breakdown or [],
    }


def _resolve_build_and_run(
    op_dir: str, build_cmd, run_cmd, skip_build: bool
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Build (unless skipped) and resolve the run argv. Returns (run_argv, error)."""
    if not skip_build:
        resolved_build = _as_argv(build_cmd) if build_cmd else detect_build_cmd(op_dir)
        if resolved_build is None:
            return None, "no build entry detected in %s (pass build_cmd)" % op_dir
        ok, err = _run_build(op_dir, resolved_build)
        if not ok:
            return None, err

    resolved_run = _as_argv(run_cmd) if run_cmd else detect_run_cmd(op_dir)
    if resolved_run is None:
        return None, "no run entry detected in %s (pass run_cmd)" % op_dir
    return resolved_run, None


def _collect(
    op_dir: str, run_argv: List[str], base_out: str, opts: _MsprofOptions,
) -> Dict[str, Any]:
    """Run the operator under msprof from inside op_dir; always restore cwd."""
    prev = os.getcwd()
    try:
        os.chdir(op_dir)
        return msprof_collect.run_msprof(
            run_argv, base_out, metrics=opts.metric_groups, quick=opts.quick, device=opts.device
        )
    finally:
        os.chdir(prev)


def _parse_profile(
    prof_dirs: Dict[str, str], base_out: str, quick: bool, kernel_match: Optional[str],
) -> Tuple[Optional[float], Optional[str], Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    """Parse duration + performance_metrics. Returns (dur, kname, perf, breakdown, error)."""
    if quick:
        dur, kname, derr, breakdown = msprof_collect.parse_duration_quick(
            prof_dirs.get("quick", base_out), kernel_match=kernel_match
        )
    else:
        first = next(iter(prof_dirs.values()), base_out)
        dur, kname, derr, breakdown = msprof_collect.parse_duration(
            first, kernel_match=kernel_match
        )
    if dur is None:
        return None, None, {}, breakdown, "duration parse failed: %s" % derr

    if quick:
        perf: Dict[str, Any] = {"latency": {"task_duration_us": dur}}
    else:
        perf = msprof_collect.parse_aic_metrics(prof_dirs, dur, kernel_match=kernel_match)
    return dur, kname, perf, breakdown, None


def evaluate_op_local(
    op_dir: str,
    op_name: str,
    *,
    build_cmd=None,
    run_cmd=None,
    metrics: Tuple[str, ...] = msprof_collect.METRICS_DEFAULT,
    full: bool = False,
    quick: bool = False,
    device: Optional[int] = None,
    workdir: Optional[str] = None,
    skip_build: bool = False,
    kernel_match: Optional[str] = None,
) -> Dict[str, Any]:
    """Build + msprof-profile one operator working copy; return a normalized result.

    op_dir       : operator working-copy root (contains build/run entry points).
    op_name      : "<category>/<op>" identity, echoed into the result.
    build_cmd/run_cmd : override auto-detection (str or argv list).
    full         : use all 7 aic-metrics groups (else the default 3).
    quick        : task-time only, 1 pass (performance_metrics degrades to latency).
    device       : NPU id; default picks the idlest via npu-smi.
    kernel_match : selector matched against op_summary kernel names (default: bare
                   op name from ``op_name``); pass "all" to sum every compute kernel.
    """
    if not os.path.isdir(op_dir):
        return _fail(op_name, None, "op_dir not found: %s" % op_dir)

    if not msprof_collect.msprof_available():
        return _fail(op_name, None, "msprof not found; source CANN set_env.sh first")

    run_argv, err = _resolve_build_and_run(op_dir, build_cmd, run_cmd, skip_build)
    if run_argv is None:
        return _fail(op_name, None, err)

    selector = kernel_match or op_name.rsplit("/", 1)[-1]
    metric_groups = msprof_collect.METRICS_FULL if full else metrics
    base_out = workdir or tempfile.mkdtemp(prefix="msprof_%s_" % op_name.replace("/", "_"))
    os.makedirs(base_out, exist_ok=True)
    collected = _collect(op_dir, run_argv, base_out, _MsprofOptions(tuple(metric_groups), quick, device))

    device_id = collected.get("device_id")
    if not collected.get("success"):
        return _fail(op_name, device_id, collected.get("error_info", "msprof failed"))

    dur, kname, perf, breakdown, perr = _parse_profile(
        collected["prof_dirs"], base_out, quick, selector
    )
    if perr is not None:
        return _fail(op_name, device_id, perr, kernel_breakdown=breakdown)

    return {
        "success": True,
        "op_name": op_name,
        "score": float(dur),
        "duration": dur,
        "device_id": device_id,
        "performance_metrics": perf,
        "kernel_name": kname,
        "error_info": "",
        "kernel_breakdown": breakdown,
    }
