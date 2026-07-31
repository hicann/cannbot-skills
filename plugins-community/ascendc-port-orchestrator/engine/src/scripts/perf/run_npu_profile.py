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
"""NPU kernel-time profiler (PR-3 / F10.D.3) — torch_npu.profiler Level0 wrapper.

Ported from cv-agent `utils/run_perf.py` (1058 LOC) to a5_ops focused form.
This module extracts kernel-time (NOT wall-time) for an AscendC kernel
invocation, enabling apples-to-apples perf comparison across runs/containers.

Why this exists:
  Today perf measurement is ad-hoc — pass_a_runner uses time.time() (wall),
  some ops use msprof, others use raw forward call timing. Mixing wall +
  kernel times produces incomparable ratios. cv-agent demonstrated empirically
  that torch_npu.profiler Level0 isolates kernel-only time consistently.

Two modes:
  profiler  — torch_npu.profiler with Level0 + extract op_summary.csv kernel ms
  time      — time.perf_counter() wall-time fallback (when profiler unavailable)

Container probe pre-condition (per RFC v4 §5):
  Before relying on profiler mode in production, each target container MUST be
  probed for torch_npu.profiler availability via probe_torch_npu_profiler().
  If unavailable → fallback to time mode + flag in output.

Usage:
  from run_npu_profile import measure_kernel_time

  result = measure_kernel_time(
      model=my_module,        # callable taking *inputs
      inputs=(x, y),          # input tensors on NPU
      warmup=3, repeat=10,
      mode="profiler",        # or "time"
      kernel_op="my_kernel",  # name to extract from op_summary; None = all-op aggregate
  )
  # result = {"mode": "profiler", "median_ms": float, "p99_ms": float,
  #          "raw_ms": [...], "n_repeats": int, "all_repeats_passed": bool}

Per success-criteria gates F10.E.1:
  - n_repeats ≥ 3 required for reproducibility (default 10)
  - all_repeats_passed = True only if all repeats produced kernel-time data
"""
import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    print("FATAL: torch not available", file=sys.stderr)
    sys.exit(1)

# torch_npu import is lazy — probe-able for container availability
_TORCH_NPU_AVAILABLE = False
_TORCH_NPU_PROFILER_AVAILABLE = False
try:
    import torch_npu  # noqa: F401
    _TORCH_NPU_AVAILABLE = True
    try:
        from torch_npu import profiler as _torch_npu_profiler  # noqa: F401
        _TORCH_NPU_PROFILER_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


# ----------------------------------------------------------------------------
# Container probe (hard pre-condition gate)
# ----------------------------------------------------------------------------

def probe_torch_npu_profiler():
    """Return dict describing torch_npu.profiler availability + version.

    Use this BEFORE relying on profiler mode in production. If fails, fallback
    to time mode + document the gap.
    """
    result = {"torch_npu_available": _TORCH_NPU_AVAILABLE,
              "profiler_available": _TORCH_NPU_PROFILER_AVAILABLE,
              "torch_npu_version": None,
              "profiler_features": []}
    if _TORCH_NPU_AVAILABLE:
        result["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
        if _TORCH_NPU_PROFILER_AVAILABLE:
            features = []
            for attr in ("profile", "schedule", "tensorboard_trace_handler",
                         "ProfilerActivity", "ProfilerLevel",
                         "_ExperimentalConfig", "ExportType", "AiCMetrics"):
                if hasattr(torch_npu.profiler, attr):
                    features.append(attr)
            result["profiler_features"] = features
            required = {"profile", "schedule", "ProfilerLevel", "ProfilerActivity"}
            result["all_required_present"] = required.issubset(set(features))
    return result


# ----------------------------------------------------------------------------
# Profile + kernel-time extraction
# ----------------------------------------------------------------------------

def _synchronize(device):
    """NPU synchronize (or noop if cpu)."""
    if device.type == "npu":
        torch_npu.npu.synchronize(device)


def _find_op_summary_csv(prof_dir):
    """Find ALL profiler CSV(s) inside profiler output dir.

    Profiler output structure (two formats observed in the wild):
      prof_dir/<timestamp_pid>/mindstudio_profiler_output/op_summary_*.csv
        — torch_npu ≤ 2.7 (legacy)
      prof_dir/<timestamp_pid>/mindstudio_profiler_output/kernel_details_*.csv
        — torch_npu ≥ 2.8 (current on A5/A3 fleet, 2026-05+)

    Returns list of Paths covering both formats (DEBT-136, 2026-05-26).
    `finalize_pipeline._PROFILER_CSV_TOKENS` already accepts both per
    DEBT-128; this driver was the asymmetric half. Review Gap-2:
    previously returned first match only — kept the multi-subdir aggregation.
    """
    prof_dir = Path(prof_dir)
    if not prof_dir.is_dir():
        return []
    found = []
    for sub in prof_dir.iterdir():
        if not sub.is_dir():
            continue
        ms_dir = sub / "mindstudio_profiler_output"
        if ms_dir.is_dir():
            for pat in ("op_summary_*.csv", "kernel_details_*.csv"):
                for csv_file in ms_dir.glob(pat):
                    found.append(csv_file)
    return found


# DEBT-136 (2026-05-26): torch_npu 2.7 op_summary_*.csv vs 2.8+ kernel_details_*.csv
# have different column names. Resolve by case-insensitive header probing so a
# single parser handles both. Order in tuple = preference (first match wins).
_DURATION_COL_CANDIDATES = (
    "Task Duration(us)",       # op_summary_*.csv (torch_npu 2.7)
    "Duration(us)",            # kernel_details_*.csv variant
    "Total Duration(us)",      # kernel_details_*.csv variant
    "Aiv Duration(us)",        # kernel_details_*.csv (Vector core path)
    "Aic Duration(us)",        # kernel_details_*.csv (Cube core path)
    "Kernel Duration(us)",     # fallback variant
)
_STEPID_COL_CANDIDATES = ("Step Id", "Step ID", "StepId", "step_id")
_TYPE_COL_CANDIDATES = ("Type", "Op Type", "op_type")
_NAME_COL_CANDIDATES = ("Name", "Op Name", "Kernel Name", "kernel_name", "op_name")


def _resolve_col(fieldnames, candidates):
    """Return first candidate present (case-insensitive) in fieldnames, or None."""
    if not fieldnames:
        return None
    lc_map = {fn.lower().strip(): fn for fn in fieldnames if fn}
    for c in candidates:
        hit = lc_map.get(c.lower().strip())
        if hit:
            return hit
    return None


def _parse_op_durations(csv_path, kernel_op=None, kernel_op_match="substring"):
    """Parse op_summary OR kernel_details CSV; return list of per-step ms durations.

    DEBT-136 (2026-05-26): extended to accept kernel_details_*.csv (torch_npu
    ≥ 2.8) in addition to op_summary_*.csv (torch_npu ≤ 2.7). Column-name
    resolution is by case-insensitive header probing (`_resolve_col` against
    `_*_COL_CANDIDATES` tuples) so adding a future format = one tuple edit.
    `finalize_pipeline._PROFILER_CSV_TOKENS` accepts both per DEBT-128; this
    parser is the symmetric driver-side change.

    kernel_op + kernel_op_match (review Gap-1):
      None        → aggregate all op durations per step (sum)
      "<name>" + "substring" (default) → row.Type or row.Name contains <name>
      "<name>" + "exact" → row.Type == <name> OR row.Name == <name>
      "<re>" + "regex" → regex match on Type or Name (use anchors for exact)

    Use 'exact' to avoid false-attribution between shared-prefix op names
    (e.g. "Add" matching "ReduceAdd"). For kernel_details_*.csv the Name field
    falls back to "Kernel Name" (PyTorch 2.8+ rename) — substring/exact match
    semantics are unchanged.
    """
    durations_per_step = []
    current_step_total = 0.0
    last_step_id = None
    import re as _re
    if kernel_op_match == "regex" and kernel_op is not None:
        _pattern = _re.compile(kernel_op)
    else:
        _pattern = None

    def _match(type_f, name_f):
        if kernel_op is None:
            return True
        if kernel_op_match == "exact":
            return type_f == kernel_op or name_f == kernel_op
        if kernel_op_match == "regex":
            return bool(_pattern.search(type_f)) or bool(_pattern.search(name_f))
        # default substring
        return kernel_op in type_f or kernel_op in name_f

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dur_col = _resolve_col(reader.fieldnames, _DURATION_COL_CANDIDATES)
        step_col = _resolve_col(reader.fieldnames, _STEPID_COL_CANDIDATES)
        type_col = _resolve_col(reader.fieldnames, _TYPE_COL_CANDIDATES)
        name_col = _resolve_col(reader.fieldnames, _NAME_COL_CANDIDATES)
        # Duration is the only HARD requirement — if neither legacy nor new
        # format's duration col is present, the CSV is unrecognized and we
        # return empty (caller will fall back to the inline timer path).
        if dur_col is None:
            return []
        for row in reader:
            try:
                duration_us = float(row.get(dur_col, "0") or "0")
            except (TypeError, ValueError):
                duration_us = 0.0
            # Step Id may be absent in kernel_details_*.csv — single-step
            # aggregation is the safe fallback (one bucket = whole run).
            step_id = row.get(step_col, "0") if step_col else "0"
            type_field = row.get(type_col, "") if type_col else ""
            name_field = row.get(name_col, "") if name_col else ""

            if not _match(type_field, name_field):
                continue

            if last_step_id is not None and step_id != last_step_id:
                durations_per_step.append(current_step_total / 1000.0)  # us -> ms
                current_step_total = 0.0
            current_step_total += duration_us
            last_step_id = step_id

    if current_step_total > 0 and last_step_id is not None:
        durations_per_step.append(current_step_total / 1000.0)
    return durations_per_step


def _benchmark_profiler(model, inputs, device, warmup, repeat, kernel_op=None):
    """Run model under torch_npu.profiler Level0 and return per-step kernel-time ms.

    Returns: list of ms (one per active step) OR None if profiler failed.
    """
    if not _TORCH_NPU_PROFILER_AVAILABLE:
        return None

    with tempfile.TemporaryDirectory(prefix="npu_prof_") as prof_dir:
        exp_cfg = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False, op_attr=False, data_simplification=False,
        )

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=repeat, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=True, with_flops=False,
            experimental_config=exp_cfg,
        ) as prof:
            with torch.no_grad():
                for _ in range(warmup + repeat):
                    model(*inputs)
                    _synchronize(device)
                    prof.step()

        csv_paths = _find_op_summary_csv(prof_dir)
        if not csv_paths:
            return None
        # Aggregate across all sub-dirs (multi-session safety per Gap-2)
        all_durations = []
        for csv_path in csv_paths:
            all_durations.extend(_parse_op_durations(csv_path, kernel_op=kernel_op))
        return all_durations


def _benchmark_time(model, inputs, device, warmup, repeat):
    """Fallback wall-time measurement via time.perf_counter. Returns ms list."""
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        _synchronize(device)

        timings = []
        for _ in range(repeat):
            start = time.perf_counter()
            model(*inputs)
            _synchronize(device)
            end = time.perf_counter()
            timings.append((end - start) * 1e3)
    return timings


def measure_kernel_time(model, inputs, *, warmup=3, repeat=10,
                        mode="profiler", kernel_op=None,
                        kernel_op_match="substring", device=None,
                        raise_on_empty=False):
    """Measure kernel time for `model(*inputs)`.

    Args:
      model:    callable taking *inputs
      inputs:   tuple of tensors (must already be on target device)
      warmup:   profiler warmup steps (default 3)
      repeat:   active measurement steps (default 10) — must be ≥ 3 per F10.E.1 #11
      mode:     'profiler' (Level0 kernel-time) or 'time' (wall-time fallback)
      kernel_op: optional filter for kernel name in op_summary (None = all)
      kernel_op_match: 'substring' (default) / 'exact' / 'regex' — Gap-1 fix
      device:   torch.device for synchronize; inferred from inputs if None
      raise_on_empty: per Gap-3 — when True, raise RuntimeError if all_repeats_passed=False
                     (use this in pass_a_runner integration where empty timing IS a bug)

    Returns dict:
      {mode, median_ms, p99_ms, mean_ms, raw_ms, n_repeats, all_repeats_passed,
       profiler_available, kernel_op, kernel_op_match}

    Gap-3 caller-edge: callers MUST check all_repeats_passed before using median_ms.
    If all_repeats_passed=False and raise_on_empty=False, median_ms may be None.
    """
    if repeat < 3:
        raise ValueError(f"repeat must be >= 3 (reproducibility gate), got {repeat}")

    if device is None:
        for arg in inputs:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
        if device is None:
            device = torch.device("cpu")

    result = {"mode": mode, "median_ms": None, "p99_ms": None, "mean_ms": None,
              "raw_ms": [], "n_repeats": repeat,
              "all_repeats_passed": False,
              "profiler_available": _TORCH_NPU_PROFILER_AVAILABLE,
              "kernel_op": kernel_op,
              "kernel_op_match": kernel_op_match}

    if mode == "profiler":
        timings = _benchmark_profiler(model, inputs, device, warmup, repeat, kernel_op)
        if timings is None or len(timings) == 0:
            result["mode"] = "time_fallback_profiler_unavailable"
            timings = _benchmark_time(model, inputs, device, warmup, repeat)
    elif mode == "time":
        timings = _benchmark_time(model, inputs, device, warmup, repeat)
    else:
        raise ValueError(f"mode must be 'profiler' or 'time', got {mode!r}")

    result["raw_ms"] = timings
    result["all_repeats_passed"] = len(timings) == repeat and all(t > 0 for t in timings)

    if timings:
        sorted_t = sorted(timings)
        result["median_ms"] = statistics.median(sorted_t)
        result["mean_ms"] = statistics.fmean(sorted_t)
        result["p99_ms"] = sorted_t[int(len(sorted_t) * 0.99)] if len(sorted_t) > 1 else sorted_t[0]

    if raise_on_empty and not result["all_repeats_passed"]:
        raise RuntimeError(
            f"measure_kernel_time: all_repeats_passed=False (timings={len(timings)}/{repeat}); "
            f"raise_on_empty=True requested. profiler_available={_TORCH_NPU_PROFILER_AVAILABLE}, "
            f"mode={result['mode']}. Caller should check all_repeats_passed OR set raise_on_empty=False "
            f"to get partial result back."
        )

    return result


# ----------------------------------------------------------------------------
# CLI (for container-probe diagnostics)
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="torch_npu.profiler container-probe + kernel-time wrapper (PR-3)")
    parser.add_argument("--probe", action="store_true",
                        help="Probe torch_npu.profiler availability + print JSON (no measurement)")
    parser.add_argument("--model-py", help="Path to model_new_*.py to measure (TODO: not yet wired)")
    parser.add_argument("--inputs-pt", help="Path to .pt file with input tensors (TODO: not yet wired)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10,
                        help="Active measurement steps (reproducibility gate: must be >= 3)")
    parser.add_argument("--mode", choices=["profiler", "time"], default="profiler")
    parser.add_argument("--kernel-op", default=None,
                        help="Optional substring filter for kernel name in op_summary")
    args = parser.parse_args()

    if args.probe:
        probe = probe_torch_npu_profiler()
        print(json.dumps(probe, indent=2))
        sys.exit(0 if probe.get("all_required_present", False) else 2)

    if not args.model_py:
        print("Either --probe or --model-py required", file=sys.stderr)
        sys.exit(1)

    print("CLI measurement mode not yet wired (model_new_*.py loader scaffolding pending).",
          file=sys.stderr)
    print("Use measure_kernel_time(model, inputs, ...) from Python instead.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
