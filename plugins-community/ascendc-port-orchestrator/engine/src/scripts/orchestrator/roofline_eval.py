# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Roofline-mode perf self-eval — V3 Frontier (NODE-9 Phase B1, 2026-05-28).

Per docs/design/PERF_METHODOLOGY_NOTES.md#roofline-perf-eval-v3-design:
  efficiency = actual_throughput / roofline_throughput

Provides:
  - Per-SoC roofline profiles (A3 Ascend910_9382, A5 Ascend950PR)
  - OI (operational intensity) calculation per op type
  - Efficiency classification (4-tier: >80% skip, 60-80% light, 30-60% full, <30% escalate)
  - Integration-ready API consumed by perf_gate.py / ko_brief / verification.json

Non-fatal by design — any failure returns a safe default that falls through
to existing band-aware threshold behavior.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# SoC profiles — per ROOFLINE_MODEL.md + design doc §3.4
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SocRoofline:
    """Hardware roofline parameters for one SoC target.

    A3 parameters marked as provisional — pending /aog-hardware-probe calibration.
    """
    name: str
    peak_fp16_tflops: float    # VEC (vector-unit) fp16 peak — elementwise/reduction/softmax
    peak_fp32_tflops: float    # VEC (vector-unit) fp32 peak
    peak_bw_gb_s: float        # HBM bandwidth (GB/s)
    ub_per_core_kb: int        # unified buffer per core
    l2_cache_mb: int           # shared L2 cache
    core_count: int
    # CUBE (matmul-unit) peaks — govern matmul / attention op_types (the cube
    # unit, not the vector unit, drives those). Default 0.0 → analyze() falls
    # back to the VEC peak above for any SoC not yet cube-calibrated.
    # On A5 the cube fp16 peak is ~6.7× the VEC peak, so using the VEC peak for
    # a cube-bound FA/matmul op understates the ceiling ~6.7× → efficiency_pct
    # is inflated → perf_gate wrongly skips the optimizer (efficiency≥80%).
    peak_cube_fp16_tflops: float = 0.0
    peak_cube_fp32_tflops: float = 0.0

    @property
    def ridge_fp16(self) -> float:
        """Ridge point: OI above which fp16 kernels become compute-bound.

        Units: peak_fp16_tflops in TFLOPS (10^12), peak_bw_gb_s in GB/s (10^9).
        Ridge (FLOP/byte) = TFLOPS × 10^12 / (GB/s × 10^9)
                          = TFLOPS × 1000 / (GB/s)
        Example: 56 TFLOPS / 1500 GB/s = 56000/1500 = 37.3 FLOP/byte.
        """
        return self.peak_fp16_tflops * 1000 / self.peak_bw_gb_s

    @property
    def ridge_fp32(self) -> float:
        return self.peak_fp32_tflops * 1000 / self.peak_bw_gb_s


# Provisional A3 values per design doc §3.4 — to be calibrated.
SOC_A3 = SocRoofline(
    name="Ascend910_9382",
    # Empirically calibrated 2026-05-28 via /tmp/roofline_calib.py on A3 NPU 2:
    #   - fp16 matmul 8192^3: 3.76 ms → 292 TFLOPS (cube peak)
    #   - fp32 matmul 4096^3: 1.68 ms → 82 TFLOPS (cube peak)
    #   - HBM memcpy 256M fp16: 0.91 ms → 590 GB/s
    #   - HBM triad 256M fp16: 2.31 ms → 697 GB/s
    # NOTE (2026-06-12): A3 VEC peaks below are NOT yet calibrated — the
    # 290/82 values are actually the measured CUBE peaks left in place to
    # preserve pre-existing A3 behavior (zero regression). Effect: A3 VEC-bound
    # ops see an over-high ceiling → understated efficiency → over-optimize
    # (the SAFE direction). TODO: calibrate A3 VEC peak (~20-40 TFLOPS fp16 per
    # the 2026-05-28 note) on the A3 host, then lower these.
    peak_fp16_tflops=290.0,
    peak_fp32_tflops=82.0,
    # CUBE peaks (measured 2026-05-28): fp16 matmul 8192^3 → 292, fp32 → 82.
    peak_cube_fp16_tflops=290.0,
    peak_cube_fp32_tflops=82.0,
    peak_bw_gb_s=700.0,    # empirical triad BW
    ub_per_core_kb=192,
    l2_cache_mb=48,
    core_count=32,
)

# A5 VEC peaks from ROOFLINE_MODEL.md (56 AI Vector Cores × 512 FLOPS/cyc × 1GHz).
# A5 CUBE peaks measured 2026-06-12 on .171 NPU 1 (Ascend950PR_957b, cann-9.1.T500,
# torch.matmul fp16/bf16/fp32 8192^3, achieved-peak method mirroring the A3 calib):
#   fp16 → 373 TFLOPS, bf16 → 368, fp32 → 24  (script: /tmp/a5_cube_calib.py)
# The cube fp16 peak (373) is ~6.7× the VEC peak (56); for fp32 the cube unit
# is NOT favored (24 ≈ VEC 28) since A5 cube is fp16/bf16-optimized.
SOC_A5 = SocRoofline(
    name="Ascend950PR",
    peak_fp16_tflops=56.0,    # VEC fp16
    peak_fp32_tflops=28.0,    # VEC fp32
    peak_cube_fp16_tflops=373.0,   # measured (covers bf16 ≈ 368)
    peak_cube_fp32_tflops=24.0,    # measured
    peak_bw_gb_s=1500.0,   # 1.5 TB/s → 1500 GB/s
    ub_per_core_kb=192,
    l2_cache_mb=48,
    core_count=56,
)

_SOC_BY_NAME: dict[str, SocRoofline] = {
    "a3": SOC_A3,
    "a5": SOC_A5,
    "ascend910_9382": SOC_A3,
    "ascend950pr": SOC_A5,
    "ascend950pr_9589": SOC_A5,
}


def soc_for_target(target: str) -> SocRoofline:
    """Resolve roofline profile for a target name (case-insensitive prefix match)."""
    key = target.lower().replace("-ds", "")
    if key in _SOC_BY_NAME:
        return _SOC_BY_NAME[key]
    # Prefix match accepts target variants such as "a3_ds".
    for k, v in _SOC_BY_NAME.items():
        if key.startswith(k) or k.startswith(key):
            return v
    return SOC_A5  # safe default: A5 (higher ceiling = conservative for skip-gate)


# ---------------------------------------------------------------------------
# OI calculation — per design doc §3.3
# ---------------------------------------------------------------------------

def _oi_matmul(matrix_rows: int, reduction_dim: int, matrix_cols: int, itemsize: int) -> float:
    """C = A[M,K] @ B[K,N] — 2*M*K*N FLOP, read A+B, write C."""
    flops = 2.0 * matrix_rows * reduction_dim * matrix_cols
    bytes_rw = (matrix_rows * reduction_dim + reduction_dim * matrix_cols + matrix_rows * matrix_cols) * itemsize
    return flops / max(bytes_rw, 1)


def _oi_elementwise(itemsize: int) -> float:
    """1 unary or binary op per element: ~2 FLOP per element, 3 memory ops."""
    return 2.0 / (3.0 * itemsize)


def _oi_reduction(element_count: int, itemsize: int) -> float:
    """Reduce N elements → scalar. ~N FLOP, read N, write 1."""
    return float(element_count) / ((element_count + 1) * itemsize)


def _oi_softmax(row_count: int, feature_dim: int, itemsize: int) -> float:
    """Softmax over last dim: 3 passes (max, exp+sum, div) per row."""
    flops = 3.0 * row_count * feature_dim  # max-sub, exp, sum, div = ~3 ops per element
    bytes_rw = (2 * row_count + 1) * feature_dim * itemsize  # read scores, write P (fp16)
    return flops / max(bytes_rw, 1)


def _oi_attention(query_seq_len: int, key_seq_len: int, head_dim: int, itemsize: int) -> float:
    """FA: Q@K^T + softmax + P@V. 2 matmuls + 1 softmax."""
    oi1 = _oi_matmul(query_seq_len, head_dim, key_seq_len, itemsize)
    oi_soft = _oi_softmax(query_seq_len, key_seq_len, itemsize)
    oi2 = _oi_matmul(query_seq_len, key_seq_len, head_dim, itemsize)
    return (oi1 + oi_soft + oi2) / 3.0  # simple average


# Per-op-type OI dispatcher.
_OI_DISPATCH = {
    "matmul": _oi_matmul,
    "elementwise": _oi_elementwise,
    "reduction": _oi_reduction,
    "softmax": _oi_softmax,
    "attention": _oi_attention,
}


def _oi_args_matmul(op_shape: dict, itemsize: int) -> tuple[int, int, int, int]:
    """Build the positional OI arguments for a matmul shape."""
    return (
        int(op_shape.get("M", 0)),
        int(op_shape.get("K", 0)),
        int(op_shape.get("N", 0)),
        itemsize,
    )


def _oi_args_elementwise(_op_shape: dict, itemsize: int) -> tuple[int]:
    """Build the positional OI arguments for an elementwise shape."""
    return (itemsize,)


def _oi_args_reduction(op_shape: dict, itemsize: int) -> tuple[int, int]:
    """Build the positional OI arguments for a reduction shape."""
    return int(op_shape.get("N", 0)), itemsize


def _oi_args_softmax(op_shape: dict, itemsize: int) -> tuple[int, int, int]:
    """Build the positional OI arguments for a softmax shape."""
    return int(op_shape.get("N", 0)), int(op_shape.get("D", 0)), itemsize


def _oi_args_attention(op_shape: dict, itemsize: int) -> tuple[int, int, int, int]:
    """Build the positional OI arguments for an attention shape."""
    return (
        int(op_shape.get("Sq", 0)),
        int(op_shape.get("Sk", 0)),
        int(op_shape.get("D", 0)),
        itemsize,
    )


_OI_ARGUMENT_BUILDERS = {
    "matmul": _oi_args_matmul,
    "elementwise": _oi_args_elementwise,
    "reduction": _oi_args_reduction,
    "softmax": _oi_args_softmax,
    "attention": _oi_args_attention,
}


def _calculate_oi(op_type: str, op_shape: dict, itemsize: int) -> float:
    """Calculate OI with the existing zero-default shape-key behavior."""
    return _OI_DISPATCH[op_type](*_OI_ARGUMENT_BUILDERS[op_type](op_shape, itemsize))

# Op types whose compute ceiling is the CUBE (matmul) unit, not the vector unit.
# attention = QK^T + softmax + P@V — dominated by the two matmuls → cube-bound.
_CUBE_OP_TYPES = frozenset({"matmul", "attention"})


def _peak_tflops(soc: "SocRoofline", op_type: str, itemsize: int) -> float:
    """Governing compute ceiling (TFLOPS) for an op on this SoC.

    matmul/attention → CUBE peak (cube unit drives them); everything else →
    VEC peak. Falls back to the VEC peak when the SoC has no cube calibration
    (peak_cube_* == 0.0), so adding the cube fields is backward-compatible.
    """
    if op_type in _CUBE_OP_TYPES:
        cube = soc.peak_cube_fp16_tflops if itemsize <= 2 else soc.peak_cube_fp32_tflops
        if cube > 0.0:
            return cube
    return soc.peak_fp16_tflops if itemsize <= 2 else soc.peak_fp32_tflops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Bound(str, Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    UNCLEAR = "unclear"


@dataclass
class RooflineResult:
    """Per-op roofline self-eval result. Always constructible — no exceptions."""
    op_type: str = ""
    oi_flop_per_byte: float = 0.0
    bound: Bound = Bound.UNCLEAR
    roofline_throughput: float = 0.0   # theoretical max (same unit as actual)
    actual_throughput: float = 0.0     # measured
    efficiency_pct: float = 0.0        # 0..100+
    recommendation: str = ""
    soc_name: str = ""
    actionable: bool = False            # True if analysis is grounded (not default)


def _classify_bound(
    oi_flop_per_byte: float,
    ridge_flop_per_byte: float,
    msprof_vec_ratio: Optional[float],
    msprof_hbm_bw_util_pct: Optional[float],
) -> tuple[Bound, Optional[str]]:
    """Choose the measured bound when profiler data is available."""
    static_bound = Bound.COMPUTE if oi_flop_per_byte > ridge_flop_per_byte else Bound.MEMORY
    if msprof_hbm_bw_util_pct is None or msprof_vec_ratio is None:
        return static_bound, None
    if msprof_vec_ratio > 0.6 and msprof_hbm_bw_util_pct < 50.0:
        return Bound.COMPUTE, None
    if msprof_hbm_bw_util_pct > 50.0 and msprof_vec_ratio < 0.4:
        return Bound.MEMORY, None
    if msprof_vec_ratio < 0.3 and msprof_hbm_bw_util_pct < 30.0:
        return Bound.UNCLEAR, "both compute and memory util < 30% — bottleneck unclear; skip roofline gate"
    return static_bound, None


def _matmul_memory_bytes(op_shape: dict, itemsize: int) -> int:
    """Return the matmul read/write byte count using required shape keys."""
    matrix_rows = int(op_shape["M"])
    reduction_dim = int(op_shape["K"])
    matrix_cols = int(op_shape["N"])
    return (matrix_rows * reduction_dim + reduction_dim * matrix_cols + matrix_rows * matrix_cols) * itemsize


def _elementwise_memory_bytes(op_shape: dict, itemsize: int) -> int:
    """Return the elementwise read/write byte count."""
    return int(op_shape.get("N", 1)) * itemsize * 3


def _reduction_memory_bytes(op_shape: dict, itemsize: int) -> int:
    """Return the reduction read/write byte count."""
    return (int(op_shape.get("N", 1)) + 1) * itemsize


def _softmax_memory_bytes(op_shape: dict, itemsize: int) -> int:
    """Return the softmax read/write byte count using required shape keys."""
    return (2 * int(op_shape["N"]) + 1) * int(op_shape["D"]) * itemsize


def _attention_memory_bytes(op_shape: dict, itemsize: int) -> int:
    """Return the attention read/write byte count using required shape keys."""
    query_seq_len = int(op_shape["Sq"])
    key_seq_len = int(op_shape["Sk"])
    head_dim = int(op_shape["D"])
    total_elements = (
        query_seq_len * head_dim
        + key_seq_len * head_dim
        + query_seq_len * key_seq_len * 2
        + query_seq_len * head_dim
    )
    return total_elements * itemsize


_MEMORY_BYTES_CALCULATORS = {
    "matmul": _matmul_memory_bytes,
    "elementwise": _elementwise_memory_bytes,
    "reduction": _reduction_memory_bytes,
    "softmax": _softmax_memory_bytes,
    "attention": _attention_memory_bytes,
}


def _matmul_flops(op_shape: dict) -> float:
    """Return the matmul FLOP count using required shape keys."""
    matrix_rows = int(op_shape["M"])
    reduction_dim = int(op_shape["K"])
    matrix_cols = int(op_shape["N"])
    return 2.0 * matrix_rows * reduction_dim * matrix_cols


def _elementwise_flops(op_shape: dict) -> float:
    """Return the elementwise FLOP count."""
    return float(op_shape.get("N", 1)) * 2


def _reduction_flops(op_shape: dict) -> float:
    """Return the reduction FLOP count."""
    return float(op_shape.get("N", 1))


def _softmax_flops(op_shape: dict) -> float:
    """Return the softmax FLOP count using required shape keys."""
    return 3.0 * int(op_shape["N"]) * int(op_shape["D"])


def _attention_flops(op_shape: dict) -> float:
    """Return the attention FLOP count using required shape keys."""
    query_seq_len = int(op_shape["Sq"])
    key_seq_len = int(op_shape["Sk"])
    head_dim = int(op_shape["D"])
    return (
        2.0 * query_seq_len * head_dim * key_seq_len
        + 3.0 * query_seq_len * key_seq_len
        + 2.0 * query_seq_len * key_seq_len * head_dim
    )


_FLOP_CALCULATORS = {
    "matmul": _matmul_flops,
    "elementwise": _elementwise_flops,
    "reduction": _reduction_flops,
    "softmax": _softmax_flops,
    "attention": _attention_flops,
}


def _actual_throughput(bound: Bound, op_type: str, op_shape: dict, itemsize: int, measured_time_s: float) -> float:
    """Calculate actual bandwidth or FLOPs/s with the legacy shape-key behavior."""
    if bound == Bound.MEMORY:
        work = _MEMORY_BYTES_CALCULATORS.get(op_type, lambda *_: 1)(op_shape, itemsize)
    else:
        work = _FLOP_CALCULATORS.get(op_type, lambda *_: 1.0)(op_shape)
    return work / measured_time_s


def _efficiency_recommendation(result: RooflineResult) -> str:
    """Build the existing four-tier optimization recommendation."""
    if result.efficiency_pct > 80:
        return (
            f"efficiency {result.efficiency_pct:.0f}% > 80% — near hardware limit; "
            f"skip optimizer, finalize with roofline_efficiency metadata"
        )
    if result.efficiency_pct > 60:
        optimization_focus = (
            "data movement (MTE/DataCopy)"
            if result.bound == Bound.MEMORY
            else "vectorization (VEC pipeline overlap)"
        )
        return f"efficiency {result.efficiency_pct:.0f}% 60-80% — light optimize; focus on {optimization_focus}"
    if result.efficiency_pct > 30:
        primary_target = "memory bandwidth" if result.bound == Bound.MEMORY else "compute throughput"
        return f"efficiency {result.efficiency_pct:.0f}% 30-60% — full optimize; primary target: {primary_target}"
    return (
        f"efficiency {result.efficiency_pct:.0f}% < 30% — algorithmic redesign likely needed; "
        f"escalate to researcher for architecture-level gap analysis"
    )


def _prepare_analysis(
    op_type: str,
    soc_target: str,
    itemsize: int,
    op_shape: Optional[dict],
) -> tuple[RooflineResult, Optional[SocRoofline], Optional[float]]:
    """Create a result and compute OI, returning safe failures unchanged."""
    soc = soc_for_target(soc_target)
    result = RooflineResult(op_type=op_type, soc_name=soc.name)
    if op_type not in _OI_DISPATCH or op_shape is None:
        result.recommendation = f"unknown op_type={op_type!r} or missing op_shape; fall back to band-aware threshold"
        return result, None, None
    try:
        oi_flop_per_byte = _calculate_oi(op_type, op_shape, itemsize)
    except (ValueError, TypeError, KeyError):
        result.recommendation = "OI calculation failed — missing or invalid op_shape keys"
        return result, None, None
    result.oi_flop_per_byte = oi_flop_per_byte
    return result, soc, oi_flop_per_byte


def _classify_analysis(
    result: RooflineResult,
    soc: SocRoofline,
    op_type: str,
    itemsize: int,
    oi_flop_per_byte: float,
    msprof_vec_ratio: Optional[float],
    msprof_hbm_bw_util_pct: Optional[float],
) -> Optional[str]:
    """Set the bound and roofline ceiling, returning an early recommendation."""
    peak_flops = _peak_tflops(soc, op_type, itemsize)
    ridge_flop_per_byte = peak_flops * 1000 / soc.peak_bw_gb_s
    result.bound, recommendation = _classify_bound(
        oi_flop_per_byte,
        ridge_flop_per_byte,
        msprof_vec_ratio,
        msprof_hbm_bw_util_pct,
    )
    if recommendation is None:
        result.roofline_throughput = soc.peak_bw_gb_s * 1e9 if result.bound == Bound.MEMORY else peak_flops * 1e12
    return recommendation


def _actionable_failure(result: RooflineResult, recommendation: str) -> RooflineResult:
    """Record an actionable early-exit result."""
    result.recommendation, result.actionable = recommendation, True
    return result


def analyze(
    *,
    op_type: str,
    soc_target: str = "a5",
    dtype_itemsize: int = 2,  # fp16 default
    measured_time_s: float = 0.0,
    op_shape: Optional[dict] = None,
    msprof_vec_ratio: Optional[float] = None,
    msprof_mte2_ratio: Optional[float] = None,
    msprof_hbm_bw_util_pct: Optional[float] = None,
) -> RooflineResult:
    """Run roofline self-eval for one op.

    Args:
        op_type: one of "matmul", "elementwise", "reduction", "softmax", "attention"
        soc_target: target name ("a3", "a5", "ascend910_9382", etc.)
        dtype_itemsize: element size in bytes (2=fp16, 4=fp32)
        measured_time_s: measured kernel execution time in seconds
        op_shape: dict with op-type-specific keys (M/K/N for matmul, etc.)
        msprof_vec_ratio: aiv_vec_ratio from msprof (0..1)
        msprof_mte2_ratio: aiv_mte2_ratio from msprof (0..1)
        msprof_hbm_bw_util_pct: HBM BW utilization % from msprof

    Returns:
        RooflineResult — always valid, never raises
    """
    itemsize = dtype_itemsize
    result, soc, oi = _prepare_analysis(op_type, soc_target, itemsize, op_shape)
    if soc is None or oi is None:
        return result
    recommendation = _classify_analysis(result, soc, op_type, itemsize, oi, msprof_vec_ratio, msprof_hbm_bw_util_pct)
    if recommendation is not None:
        return _actionable_failure(result, recommendation)
    if measured_time_s <= 0:
        return _actionable_failure(result, "no measured_time_s provided; cannot compute efficiency")
    try:
        result.actual_throughput = _actual_throughput(result.bound, op_type, op_shape, itemsize, measured_time_s)
    except (ValueError, TypeError, KeyError):
        return _actionable_failure(result, "throughput calculation failed — invalid op_shape")

    result.efficiency_pct = (
        (result.actual_throughput / result.roofline_throughput) * 100.0
        if result.roofline_throughput > 0
        else 0.0
    )
    result.recommendation = _efficiency_recommendation(result)
    result.actionable = True
    return result


def should_skip_optimizer(result: RooflineResult, *, threshold_pct: float = 80.0) -> bool:
    """Pre-ko gate: given roofline analysis, should we skip the optimizer?

    True when efficiency is already near hardware ceiling — further micro-optimization
    will yield diminishing returns. Only actionable results participate.
    """
    return result.actionable and result.efficiency_pct >= threshold_pct


def attach_to_verification(result: RooflineResult) -> dict:
    """Pack roofline metadata for inclusion in verification.json performance section."""
    return {
        "roofline_efficiency_pct": round(result.efficiency_pct, 1),
        "roofline_bound": result.bound.value,
        "roofline_oi_flop_per_byte": round(result.oi_flop_per_byte, 2),
        "roofline_recommendation": result.recommendation,
        "roofline_soc": result.soc_name,
    }


def from_msprof_summary(msprof_json_path: Path, soc_target: str = "a5") -> Optional[RooflineResult]:
    """Construct RooflineResult from a msprof op_summary / op_statistic JSON dump.

    Reads a JSON file with keys: op_type, measured_time_s, op_shape, vec_ratio,
    mte2_ratio, hbm_bw_util_pct. Returns None if file missing or unparseable.
    """
    if not msprof_json_path.is_file():
        return None
    try:
        data = json.loads(msprof_json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return analyze(
        op_type=data.get("op_type", ""),
        soc_target=soc_target,
        dtype_itemsize=data.get("dtype_itemsize", 2),
        measured_time_s=data.get("measured_time_s", 0.0),
        op_shape=data.get("op_shape"),
        msprof_vec_ratio=data.get("vec_ratio"),
        msprof_mte2_ratio=data.get("mte2_ratio"),
        msprof_hbm_bw_util_pct=data.get("hbm_bw_util_pct"),
    )
