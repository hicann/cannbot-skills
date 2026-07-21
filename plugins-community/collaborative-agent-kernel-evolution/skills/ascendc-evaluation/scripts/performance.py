# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Performance measurement using msprof hardware profiling.

Measures kernel execution time via AdvancedPerformanceEngine from ascend_performance_test.py.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import torch_npu

from ascend_performance_test import AdvancedPerformanceEngine


def _copy_profiling_to_flat_dir(msprof_case_dir: Path, flat_dir: Path, case_name: str) -> None:
    """
    WF-012: Copy op_summary*.csv from deep _msprof_work/{case}/custom/.../ tree to
    a flat profiling/ directory at the work_dir level.

    This allows Advisor AIC-1 and PMR-2 to find profiling data via a simple
    `ls profiling/op_summary*.csv` without traversing the nested msprof tree.

    Output: flat_dir/op_summary_{safe_case_name}.csv  (first matching custom CSV)
    """
    # Sanitize case_name: allow only [A-Za-z0-9_.-], replace everything else with '_'
    # This prevents directory-traversal via '../' or embedded path separators.
    safe_case_name = re.sub(r'[^A-Za-z0-9_.\-]', '_', case_name)
    flat_dir.mkdir(parents=True, exist_ok=True)
    custom_dir = msprof_case_dir / "custom"
    if not custom_dir.exists():
        return
    # Find any op_summary*.csv under the custom profiling tree
    for csv in sorted(custom_dir.rglob("op_summary*.csv")):
        dst = flat_dir / f"op_summary_{safe_case_name}.csv"
        # Guard against path traversal: dst must stay inside flat_dir
        if not dst.resolve().is_relative_to(flat_dir.resolve()):
            logging.warning(f"WF-012: dst {dst} escapes flat_dir, skipping")
            return
        shutil.copy2(str(csv), str(dst))
        logging.debug("WF-012: copied %s → %s", csv.name, dst)
        return  # only need one representative CSV per case


def _select_target_row(df):
    """Pick the row most likely to be the target custom kernel.

    Use the row with the longest Task Duration(us); fall back to iloc[0].
    """
    import pandas as pd
    if "Task Duration(us)" in df.columns:
        dur = pd.to_numeric(df["Task Duration(us)"], errors="coerce").dropna()
        if not dur.empty:
            return df.loc[dur.idxmax()]
    return df.iloc[0]


def _ratio_getter(row):
    """Return a function reading a numeric ratio column from row (0.0 if missing)."""
    def _f(col: str) -> float:
        v = row.get(col)
        return float(v) if v is not None and str(v) not in ("", "nan") else 0.0
    return _f


def _vector_bottleneck_candidates(ratio) -> dict:
    vec = ratio("aiv_vec_ratio")
    mte2 = ratio("aiv_mte2_ratio")
    mte3 = ratio("aiv_mte3_ratio")
    scal = ratio("aiv_scalar_ratio")
    return {
        "Compute-bound": (vec, f"aiv_vec_ratio={vec:.2f}"),
        "Memory-bound": (mte2 + mte3, f"aiv_mte2_ratio={mte2:.2f}+aiv_mte3_ratio={mte3:.2f}={mte2+mte3:.2f}"),
        "Instruction-bound": (scal, f"aiv_scalar_ratio={scal:.2f}"),
    }


def _cube_bottleneck_candidates(ratio) -> dict:
    mac = ratio("aic_mac_ratio")
    mte2 = ratio("aic_mte2_ratio")
    scal = ratio("aic_scalar_ratio")
    return {
        "Compute-bound": (mac, f"aic_mac_ratio={mac:.2f}"),
        "Memory-bound": (mte2, f"aic_mte2_ratio={mte2:.2f}"),
        "Instruction-bound": (scal, f"aic_scalar_ratio={scal:.2f}"),
    }


def _mix_bottleneck_candidates(ratio) -> dict:
    vec = ratio("aiv_vec_ratio")
    mac = ratio("aic_mac_ratio")
    mte2a = ratio("aiv_mte2_ratio")
    mte3a = ratio("aiv_mte3_ratio")
    mte2c = ratio("aic_mte2_ratio")
    scal_v = ratio("aiv_scalar_ratio")
    scal_c = ratio("aic_scalar_ratio")
    compute = max(vec, mac)
    memory = max(mte2a + mte3a, mte2c)
    scalar = max(scal_v, scal_c)
    compute_ev = f"aiv_vec_ratio={vec:.2f}" if vec >= mac else f"aic_mac_ratio={mac:.2f}"
    memory_ev = (f"aiv_mte2_ratio={mte2a:.2f}+aiv_mte3_ratio={mte3a:.2f}={mte2a+mte3a:.2f}"
                 if mte2a + mte3a >= mte2c else f"aic_mte2_ratio={mte2c:.2f}")
    scalar_ev = f"aiv_scalar_ratio={scal_v:.2f}" if scal_v >= scal_c else f"aic_scalar_ratio={scal_c:.2f}"
    return {
        "Compute-bound": (compute, compute_ev),
        "Memory-bound": (memory, memory_ev),
        "Instruction-bound": (scalar, scalar_ev),
    }


def _classify_bottleneck_from_csv(csv_path: Path) -> str:
    """
    Derive a bottleneck classification with evidence from an op_summary CSV row.

    Uses dominant-ratio approach (no fixed thresholds): whichever of
    Compute / Memory / Instruction has the highest ratio wins.  Only guards
    against rows where all ratios are < 0.10 (non-kernel host rows).

    Returns "<Class>: <evidence> 为最大比率" or "Unknown: <reason>".
    Mirrors profiling_extractor.py — kept in sync manually (no cross-skill import).
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.empty:
            return "Unknown: CSV 为空"
        row = _select_target_row(df)
        task_type = str(row.get("Task Type", ""))
        ratio = _ratio_getter(row)

        if "VECTOR" in task_type and "MIX" not in task_type:
            candidates = _vector_bottleneck_candidates(ratio)
        elif "CORE" in task_type and "VECTOR" not in task_type and "MIX" not in task_type:
            candidates = _cube_bottleneck_candidates(ratio)
        elif "MIX" in task_type:
            candidates = _mix_bottleneck_candidates(ratio)
        else:
            candidates = {}

        if not candidates:
            return f"Unknown: task_type='{task_type}' 无法识别"

        dominant = max(candidates, key=lambda k: candidates.get(k, (0.0, ""))[0])
        dom_ratio, dom_ev = candidates.get(dominant, (0.0, ""))
        if dom_ratio < 0.10:
            return f"Unknown: 所有比率均低于 0.10（最高={dom_ratio:.2f}），可能为非 AI Core 执行行"
        return f"{dominant}: {dom_ev} 为最大比率"

    except Exception as e:
        logging.debug("bottleneck classification failed: %s", e)
    return "Unknown: 分类异常"


def measure_performance_msprof(
    ref_model: torch.nn.Module,
    custom_model: torch.nn.Module,
    inputs: list,
    work_dir: Path,
    case_name: str = "test",
    device_id: int = 0,
    num_trials: int = 10,
) -> Dict[str, Any]:
    """
    Measure performance using AdvancedPerformanceEngine with msprof profiling.

    Args:
        ref_model: Reference model instance
        custom_model: Custom model instance
        inputs: List of input tensors (already on NPU)
        work_dir: Working directory for profiling artifacts
        case_name: Test case name (for file naming)
        device_id: NPU device ID
        num_trials: Number of measurement trials

    Returns:
        Dict with:
            - ref_time_us: Reference kernel time (microseconds)
            - custom_time_us: Custom kernel time (microseconds)
            - speedup: Speedup ratio
    """
    logger = logging.getLogger(__name__)
    engine = AdvancedPerformanceEngine(logger)

    # Sanitize case_name before using it as a path component to prevent path traversal.
    safe_case_name = re.sub(r'[^A-Za-z0-9_.\-]', '_', case_name)
    prof_work = work_dir / "_msprof_work" / safe_case_name
    prof_work.mkdir(parents=True, exist_ok=True)

    # Profile reference
    logging.info(f"  Profiling reference model...")
    ref_time_us, _, ref_output_path = engine.warmup_and_measure(
        model=ref_model,
        inputs=inputs,
        device_id=device_id,
        profile_root=prof_work / "ref",
        num_trials=num_trials,
        task_type="vector",
        model_tag="Model"
    )

    # Profile custom
    logging.info(f"  Profiling custom model...")
    custom_time_us, _, custom_output_path = engine.warmup_and_measure(
        model=custom_model,
        inputs=inputs,
        device_id=device_id,
        profile_root=prof_work / "custom",
        num_trials=num_trials,
        task_type="vector",
        model_tag="ModelNew"
    )

    if ref_time_us is None or custom_time_us is None:
        raise RuntimeError("Performance measurement failed - could not extract timing data")

    speedup = ref_time_us / custom_time_us if custom_time_us > 0 else 0.0

    # WF-012: copy op_summary CSV to flat profiling/ for Advisor / PMR
    flat_profiling_dir = work_dir / "profiling"
    _copy_profiling_to_flat_dir(
        msprof_case_dir=prof_work,
        flat_dir=flat_profiling_dir,
        case_name=case_name,
    )

    # Classify bottleneck from the custom kernel's profiling CSV
    safe_case_name = re.sub(r'[^A-Za-z0-9_.\-]', '_', case_name)
    custom_csv = flat_profiling_dir / f"op_summary_{safe_case_name}.csv"
    bottleneck_class = _classify_bottleneck_from_csv(custom_csv) if custom_csv.exists() else "Unknown"

    return {
        "ref_time_us": ref_time_us,
        "custom_time_us": custom_time_us,
        "speedup": speedup,
        "bottleneck_class": bottleneck_class,
    }
