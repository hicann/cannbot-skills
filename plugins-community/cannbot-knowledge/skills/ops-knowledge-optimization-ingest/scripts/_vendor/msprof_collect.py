# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Local msprof collection helpers (stdlib-only).

Vendored & distilled from the sibling ``ops-profiling`` skill so this skill folder
stays self-contained. Responsibilities:

- pick an idle NPU (``pick_idle_npu``);
- run an operator command under ``msprof`` (``run_msprof``);
- parse the operator's kernel duration in微秒 from msprof artifacts
  (``parse_duration`` / ``parse_duration_quick``);
- fold aic-metrics ``op_summary`` rows into a normalized ``performance_metrics``
  dict aligned with what the previous remote service returned
  (``parse_aic_metrics``).

Depends only on the standard library + the external ``msprof`` / ``npu-smi``
binaries (present on a real Ascend NPU host).
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

# aic-metrics groups. Default (3) covers what miner's performance_metrics needs;
# "full" (7) mirrors the standard ops-profiling collection.
METRICS_DEFAULT: Tuple[str, ...] = ("PipeUtilization", "Memory", "L2Cache")
METRICS_FULL: Tuple[str, ...] = (
    "PipeUtilization", "ArithmeticUtilization", "Memory",
    "MemoryL0", "MemoryUB", "L2Cache", "ResourceConflictRatio",
)

# msprof binary timeout (seconds) for a single collection pass.
MSPROF_TIMEOUT: int = 1800


# ---------------------------------------------------------------------------
# safe conversions
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# kernel selection (filter op_summary/task_time rows to the target operator)
# ---------------------------------------------------------------------------

def _normalize_kernel_name(name: str) -> str:
    """lowercase + strip non-alphanumerics, e.g. ``AddRmsNormQuant_ab12`` -> ``addrmsnormquantab12``."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def kernel_matches(kernel_name: str, kernel_match: Optional[str]) -> bool:
    """True when a row's kernel name matches the selector.

    ``kernel_match`` of ``None``/``"all"`` matches every kernel (legacy sum-all behavior).
    """
    if not kernel_match or kernel_match == "all":
        return True
    needle = _normalize_kernel_name(kernel_match)
    return (not needle) or needle in _normalize_kernel_name(kernel_name)


def _build_breakdown(
    kernel_durs: Dict[str, List[float]], kernel_match: Optional[str]
) -> List[Dict[str, Any]]:
    """[{kernel_name, duration_us, count, selected}], selected first, then by duration desc."""
    out = [
        {
            "kernel_name": name,
            "duration_us": sum(durs),
            "count": len(durs),
            "selected": kernel_matches(name, kernel_match),
        }
        for name, durs in kernel_durs.items()
    ]
    out.sort(key=lambda b: (not b["selected"], -b["duration_us"]))
    return out


# ---------------------------------------------------------------------------
# NPU device selection (npu-smi info)
# ---------------------------------------------------------------------------

def pick_idle_npu(default: int = 0) -> int:
    """Pick the least-busy NPU by parsing ``npu-smi info``; fall back to ``default``."""
    for smi in ("/usr/local/bin/npu-smi", "npu-smi"):
        try:
            p = subprocess.run([smi, "info"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if p.returncode != 0:
            continue

        devices: Dict[int, Tuple[int, float]] = {}
        cur: Optional[int] = None
        head_re = re.compile(r"^\|\s+(\d+)\s+\S+\s+\|\s+\w+\s+\|")
        bus_re = re.compile(
            r"^\|\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?"
        )
        for line in p.stdout.splitlines():
            m2 = bus_re.match(line)
            if m2 and cur is not None:
                aicore = int(m2.group(1))
                mem_used = int(m2.group(2))
                mem_total = max(int(m2.group(3)), 1)
                hbm_used = int(m2.group(4)) if m2.group(4) else 0
                hbm_total = max(int(m2.group(5)), 1) if m2.group(5) else 1
                mem_ratio = max(mem_used / mem_total, hbm_used / hbm_total)
                devices[cur] = (aicore, mem_ratio)
                cur = None
                continue
            m1 = head_re.match(line)
            if m1:
                cur = int(m1.group(1))

        if devices:
            best_id, _ = min(devices.items(), key=lambda kv: (kv[1][0], kv[1][1]))
            return best_id
    return default


# ---------------------------------------------------------------------------
# msprof invocation
# ---------------------------------------------------------------------------

def msprof_available() -> bool:
    """Return True if the ``msprof`` binary is on PATH."""
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path, "msprof")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return True
    return False


def _msprof_cmd(run_cmd: List[str], out_dir: str, metric: Optional[str], quick: bool) -> List[str]:
    """Assemble one msprof command line wrapping ``run_cmd``.

    quick: only task-time (no aic-metrics). Otherwise one pass per aic ``metric``.
    """
    cmd = ["msprof", "--output=%s" % out_dir, "--ai-core=on", "--task-time=on", "--ascendcl=on"]
    if not quick and metric:
        cmd.append("--aic-metrics=%s" % metric)
    cmd.extend(run_cmd)
    return cmd


def run_msprof(
    run_cmd: List[str],
    base_out_dir: str,
    *,
    metrics: Tuple[str, ...] = METRICS_DEFAULT,
    quick: bool = False,
    device: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run ``run_cmd`` under msprof.

    quick=True → one task-time-only pass. Otherwise one pass per aic metric.
    Returns {"success", "prof_dirs": {metric|"quick": dir}, "error_info"}.
    """
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    dev = device if device is not None else pick_idle_npu()
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = str(dev)

    prof_dirs: Dict[str, str] = {}
    passes = [("quick", True, None)] if quick else [(m, False, m) for m in metrics]

    for key, is_quick, metric in passes:
        sub = os.path.join(base_out_dir, "PROF_%s" % key)
        os.makedirs(sub, exist_ok=True)
        cmd = _msprof_cmd(run_cmd, sub, metric, is_quick)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=MSPROF_TIMEOUT, env=run_env
            )
        except FileNotFoundError:
            return {"success": False, "prof_dirs": prof_dirs, "device_id": dev,
                    "error_info": "msprof not found; source CANN set_env.sh first"}
        except subprocess.SubprocessError as exc:
            return {"success": False, "prof_dirs": prof_dirs, "device_id": dev,
                    "error_info": "msprof run error: %s" % exc}
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            return {"success": False, "prof_dirs": prof_dirs, "device_id": dev,
                    "error_info": "msprof exited %d: %s" % (proc.returncode, tail)}
        prof_dirs[key] = sub

    return {"success": True, "prof_dirs": prof_dirs, "device_id": dev, "error_info": ""}


# ---------------------------------------------------------------------------
# duration parsing (kernel time, µs)
# ---------------------------------------------------------------------------

def _extract_compute_row(r: dict) -> Optional[Tuple[float, str]]:
    """From an op_summary row return (duration_us, op_name) for compute kernels only."""
    task_type = r.get("Task Type", "")
    if "AI_CORE" not in task_type and "AIV" not in task_type and "MIX" not in task_type:
        return None
    try:
        duration = float(r.get("Task Duration(us)", 0) or 0)
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration, r.get("Op Name", "unknown")


def parse_duration(
    prof_dir: str, *, kernel_match: Optional[str] = None
) -> Tuple[Optional[float], Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Standard mode: sum ``Task Duration(us)`` of compute rows matching ``kernel_match``.

    Returns ``(duration_us, kernel_name, error, kernel_breakdown)``. ``kernel_match``
    of ``None``/``"all"`` reproduces the legacy sum-all-compute-rows behavior.
    """
    csv_files = sorted(glob.glob(
        os.path.join(prof_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv"),
        recursive=True,
    ))
    if not csv_files:
        return None, None, "no op_summary csv found", []
    try:
        with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        return None, None, "read csv error: %s" % exc, []
    if not rows:
        return None, None, "empty csv", []
    compute_rows = [cr for cr in (_extract_compute_row(r) for r in rows) if cr is not None]
    if not compute_rows:
        return None, None, "no compute rows found", []

    grouped: Dict[str, List[float]] = {}
    for duration, name in compute_rows:
        grouped.setdefault(name, []).append(duration)
    breakdown = _build_breakdown(grouped, kernel_match)
    selected = [b for b in breakdown if b["selected"]]
    if not selected:
        return (None, None,
                "no kernel matched selector %r; kernels seen: [%s]" % (
                    kernel_match, ", ".join(sorted(grouped))),
                breakdown)

    total = sum(b["duration_us"] for b in selected)
    name = selected[0]["kernel_name"] if len(selected) == 1 else "multiple_kernels"
    return total, name, None, breakdown


def parse_duration_quick(
    prof_dir: str, *, kernel_match: Optional[str] = None
) -> Tuple[Optional[float], Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Quick mode: sum ``task_time(us)`` of real kernels matching ``kernel_match``."""
    patterns = [
        os.path.join(prof_dir, "mindstudio_profiler_output", "task_time_*.csv"),
        os.path.join(prof_dir, "**", "mindstudio_profiler_output", "task_time_*.csv"),
    ]
    meta = {"PROFILING_ENABLE", "PROFILING_DISABLE", "TASK_TIMEOUT_SET", ""}
    for pattern in patterns:
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            continue
        try:
            with open(files[0], "r", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        kernel_times: Dict[str, List[float]] = {}
        for r in rows:
            if r.get("kernel_type", "") in meta:
                continue
            t = r.get("task_time(us)", "")
            if not t:
                continue
            try:
                d = float(t)
            except ValueError:
                continue
            if d > 0:
                kernel_times.setdefault(r.get("kernel_name", "unknown"), []).append(d)
        if not kernel_times:
            continue
        breakdown = _build_breakdown(kernel_times, kernel_match)
        selected = [b for b in breakdown if b["selected"]]
        if not selected:
            return (None, None,
                    "no kernel matched selector %r in task_time csv; kernels seen: [%s]" % (
                        kernel_match, ", ".join(sorted(kernel_times))),
                    breakdown)
        total = sum(b["duration_us"] for b in selected)
        if total > 0:
            name = selected[0]["kernel_name"] if len(selected) == 1 else "multiple_kernels"
            return total, name, None, breakdown
    return None, None, "no task_time csv found", []


# ---------------------------------------------------------------------------
# aic-metrics → performance_metrics
# ---------------------------------------------------------------------------

def _merge_op_summary(prof_dir: str, *, kernel_match: Optional[str] = None) -> Dict[str, float]:
    """Average numeric columns across compute rows matching ``kernel_match``."""
    csv_files = sorted(glob.glob(
        os.path.join(prof_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv"),
        recursive=True,
    ))
    if not csv_files:
        return {}
    try:
        with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        task_type = r.get("Task Type", "")
        if "AI_CORE" not in task_type and "AIV" not in task_type and "MIX" not in task_type:
            continue
        if not kernel_matches(r.get("Op Name", ""), kernel_match):
            continue
        for key, val in r.items():
            fv = safe_float(val, None)
            if fv is None:
                continue
            sums[key] = sums.get(key, 0.0) + fv
            counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def parse_aic_metrics(
    prof_dirs: Dict[str, str], latency_us: Optional[float], *, kernel_match: Optional[str] = None
) -> Dict[str, Any]:
    """Fold per-metric op_summary rows into a normalized performance_metrics dict.

    Keys mirror the previous remote service so miner's downstream logic is unchanged:
      latency / aiv_utilization / pipeline_stalls / memory_read_bw / memory_write_bw /
      cache_l2 (+ critical_path omitted: no msprof equivalent).
    """
    merged: Dict[str, float] = {}
    for key, d in prof_dirs.items():
        if key == "quick":
            continue
        merged.update(_merge_op_summary(d, kernel_match=kernel_match))

    perf: Dict[str, Any] = {}
    if latency_us is not None:
        perf["latency"] = {"task_duration_us": latency_us}

    if not merged:
        return perf

    # PipeUtilization → aiv_utilization / pipeline_stalls
    aiv_vec = safe_float(merged.get("aiv_vec_ratio"))
    if aiv_vec:
        perf["aiv_utilization"] = aiv_vec
    mte2 = safe_float(merged.get("aic_mte2_ratio"))
    mte3 = safe_float(merged.get("aic_mte3_ratio"))
    if mte2 or mte3:
        perf["pipeline_stalls"] = {"mte2_ratio": mte2, "mte3_ratio": mte3}

    # Memory → memory_*_bw
    mem: Dict[str, float] = {}
    for src, dst in (
        ("aic_main_mem_read_bw(GB/s)", "read_bw_gbps"),
        ("aiv_main_mem_write_bw(GB/s)", "write_bw_gbps"),
        ("aic_main_mem_write_bw(GB/s)", "aic_write_bw_gbps"),
    ):
        v = safe_float(merged.get(src))
        if v:
            mem[dst] = v
    if mem:
        perf["memory_bw"] = mem

    # L2Cache → cache_l2 hit rate
    hit = safe_float(merged.get("aic_read_local_l2_hit"))
    miss = safe_float(merged.get("aic_read_local_l2_miss"))
    if hit or miss:
        total = hit + miss
        perf["cache_l2"] = {
            "read_hit": hit,
            "read_miss": miss,
            "hit_rate": (hit / total * 100.0) if total > 0 else 0.0,
        }

    return perf
