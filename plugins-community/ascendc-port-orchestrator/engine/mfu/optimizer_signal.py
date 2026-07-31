#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MFU 优化信号：给 ko/fo agent 用。输入算子实测(shape/dtype/device-time)+硬件目标,
输出 {current_mfu, achievable_mfu, gap, bottleneck, lever, done} —— 让优化器以
'逼近 achievable MFU' 为迭代目标/停止判据, 替代纯 ratio≥1.12x / 5-iter 的相对指标。

用法(库): from optimizer_signal import mfu_signal
  sig = mfu_signal(flops, hbm_bytes, device_us, hw_name='910C_die', dtype='fp16',
                   op_kind='matmul', n_aic_used=1)
done 判据: sig['done'] == (current_mfu >= tau*achievable_mfu) 或 杠杆耗尽。
"""
import sys
import json
from dataclasses import dataclass

from mfu_model import HW, ETA_UTIL, achievable_mfu_with_overhead, DTYPE_BYTES


# op-class -> compute engine. **准度关键(main review 2026-07-01)**: peak 必须按算子类取对。
# matmul/FA/fused/conv -> CUBE peak; elementwise/softmax/norm/reduce/vector -> VEC peak。
# 用错 peak(matmul 拿 VEC) => MFU 错 => headroom 错 (roofline_eval 犯过, PR#400 修)。
OP_ENGINE = {
    "matmul": "cube", "flash_attention": "cube", "fused": "cube", "conv": "cube",
    "elementwise": "vector", "softmax": "vector", "rmsnorm": "vector",
    "layernorm": "vector", "reduce": "vector", "vector": "vector",
}


def _peak_for_op_class(hw, op_kind, dtype):
    """按算子类取正确的引擎 peak。fail-loud: vec 算子缺 dtype peak 就报错, 绝不静默回落 cube
    (静默回落=main 警告的 PR#400-类 bug: 错 peak 错 headroom)。"""
    engine = OP_ENGINE.get(op_kind)
    if engine is None:
        raise ValueError(f"unknown op_kind '{op_kind}' — 显式加进 OP_ENGINE 并确认引擎(cube/vector)")
    if engine == "cube":
        return hw.cube_peak(dtype)
    # vector
    if dtype not in hw.vec_peak:
        raise ValueError(f"{hw.name}: vector op '{op_kind}' 无 dtype '{dtype}' 的 vec_peak — "
                         f"须先标定(不静默回落cube, 否则错估 headroom)")
    return hw.vec_peak[dtype]


@dataclass(frozen=True)
class _MfuMetrics:
    """MFU 信号计算中与输入形状无关的中间度量。"""

    hw: object
    peak: float
    elapsed_s: float
    current_tflops: float
    current_mfu: float
    achievable: float
    overhead_bound: bool


def _calculate_metrics(flops, device_us, hw_name, dtype, op_kind, t_overhead_us):
    """计算当前 MFU、可达 MFU 与固定开销边界标记。"""
    hw = HW[hw_name]
    peak = _peak_for_op_class(hw, op_kind, dtype)
    elapsed_s = device_us * 1e-6
    current_tflops = flops / elapsed_s / 1e12
    current_mfu = flops / (peak * elapsed_s)
    t_compute_ideal = flops / peak
    overhead_s = t_overhead_us * 1e-6 if t_overhead_us is not None else 0.0
    achievable = achievable_mfu_with_overhead(
        flops,
        peak,
        op_kind,
        hw_name,
        t_overhead_s=overhead_s,
        default_eta=0.85,
    )["achievable_mfu"]
    return _MfuMetrics(
        hw=hw,
        peak=peak,
        elapsed_s=elapsed_s,
        current_tflops=current_tflops,
        current_mfu=current_mfu,
        achievable=achievable,
        overhead_bound=t_compute_ideal / elapsed_s < 0.1,
    )


def _classify_bottleneck(flops, hbm_bytes, comm_bytes, metrics):
    """按 roofline 和通信时间确定当前的主要瓶颈。"""
    arithmetic_intensity = flops / hbm_bytes if hbm_bytes else float("inf")
    ridge = metrics.peak / metrics.hw.hbm_bw
    if (
        comm_bytes
        and metrics.hw.interconnect_bw
        and comm_bytes / metrics.hw.interconnect_bw > metrics.elapsed_s * 0.5
    ):
        return "comm", arithmetic_intensity, ridge
    if arithmetic_intensity < ridge:
        return "memory", arithmetic_intensity, ridge
    return "compute", arithmetic_intensity, ridge


def _direction_advice(
    op_kind,
    n_aic_used,
    comm_bytes,
    bottleneck,
    arithmetic_intensity,
    ridge,
    metrics,
    tau,
):
    """只为已验证的 Cube 算子产生具体优化方向。"""
    engine = OP_ENGINE.get(op_kind)
    validated_direction = op_kind in ("matmul", "mm_grad", "flash_attention", "fused")
    if not validated_direction:
        note = (
            f"⚠方向弃权: op-class '{op_kind}' 未验证 —— roofline 对 vector/issue-bound "
            "会误分类(busy-ratio≠bound; 盲 issue-count+latency-hiding/double-buffer). "
            "只给 ceiling/stop; 定 bound + lever 需实测 msprof pipe-ratio."
        )
        return [], note, False

    levers = []
    if engine == "cube" and n_aic_used is not None and n_aic_used < metrics.hw.n_aic:
        multiplier = metrics.hw.n_aic / n_aic_used
        levers.append(
            f"multi-AIC: blockDim {n_aic_used}->{metrics.hw.n_aic} "
            f"(~{multiplier:.0f}x, P-P74 M/N tiling)"
        )
    if bottleneck == "memory":
        levers.append(
            f"raise arithmetic intensity (AI={arithmetic_intensity:.0f}<ridge={ridge:.0f}): "
            "增大tile/合并/算子融合"
        )
    if bottleneck == "compute" and metrics.current_mfu < metrics.achievable * tau:
        levers.append("compute-bound但低于achievable: 查tiling/尾块/流水(η gap)")
    if comm_bytes and bottleneck == "comm":
        levers.append("comm-bound: 梯度累积/通信压缩/通算重叠")
    return levers, None, True


def _build_verdict(metrics, tau, applicable_direction, direction_note, levers):
    """保持 ceiling-based 停止判据和面向调用方的 verdict 文本。"""
    gap = metrics.current_mfu / metrics.achievable if metrics.achievable else 0.0
    done = metrics.current_mfu >= tau * metrics.achievable
    if not applicable_direction:
        verdict = (
            f"ABSTAIN(direction): at {metrics.current_mfu * 100:.1f}% MFU, "
            f"{gap * 100:.0f}% of achievable; ceiling given, lever withheld — {direction_note}"
        )
    elif done:
        verdict = "DONE (≥%.0f%% achievable)" % (tau * 100)
    else:
        verdict = "CONTINUE: at %.1f%% MFU, %.0f%% of achievable — %s" % (
            metrics.current_mfu * 100,
            gap * 100,
            levers[0] if levers else "?",
        )
    return gap, done, verdict


def mfu_signal(flops, hbm_bytes, device_us, hw_name="910C_die", dtype="fp16",
               op_kind="matmul", n_aic_used=None, comm_bytes=0.0, tau=0.8,
               t_overhead_us=None):
    """返回优化信号 dict。n_aic_used: 实际用的 cube 核数(单AIC算子=1)，用于诊断 multi-AIC 杠杆。
    t_overhead_us: launch/tail 固定开销(若知)。**iteration-2 教训(2026-06-30)**: 小/overhead-bound
    算子用 flat η 会 over-state headroom(mm_grad tiny shape: compute 0.74us 但跑 67us, vendor 也仅
    1.1% MFU -> 真 achievable≈1.1% 非 88%)。传 t_overhead_us 后用 overhead-corrected achievable,
    避免在 overhead-bound 算子上追幻影 headroom。
    """
    metrics = _calculate_metrics(flops, device_us, hw_name, dtype, op_kind, t_overhead_us)
    bottleneck, arithmetic_intensity, ridge = _classify_bottleneck(
        flops,
        hbm_bytes,
        comm_bytes,
        metrics,
    )
    levers, direction_note, applicable_direction = _direction_advice(
        op_kind,
        n_aic_used,
        comm_bytes,
        bottleneck,
        arithmetic_intensity,
        ridge,
        metrics,
        tau,
    )
    if metrics.overhead_bound and t_overhead_us is None:
        levers = ["⚠overhead-bound (compute<10%实测): 传 t_overhead_us 才能得真achievable; "
                  "flat-η会over-state headroom(iter2教训)"] + levers
    gap, done, verdict = _build_verdict(
        metrics,
        tau,
        applicable_direction,
        direction_note,
        levers,
    )
    return {
        "current_mfu": round(metrics.current_mfu, 4),
        "current_TFLOPS": round(metrics.current_tflops, 1),
        "achievable_mfu": round(metrics.achievable, 4),
        "peak_TFLOPS": round(metrics.peak / 1e12, 1),
        "gap_to_achievable": round(gap, 3),
        "bottleneck": bottleneck,
        "overhead_bound": metrics.overhead_bound,
        "applicable_direction": applicable_direction,
        "direction_note": direction_note,
        "levers": levers,
        "done": done,
        "verdict": verdict,
    }


if __name__ == "__main__":
    # demo: mm_grad [512,256,512] fp16 单AIC, 78us, 2 GEMM
    M, K, N = 512, 256, 512
    flops = 2 * M * K * N * 2
    hbm = (M * K + K * N + M * N) * DTYPE_BYTES["fp16"] * 2
    sig = mfu_signal(flops, hbm, 78.0, "910C_die", "fp16", "matmul", n_aic_used=1)
    print("mm_grad[512,256,512] fp16 单AIC 78us:")
    print(json.dumps(sig, ensure_ascii=False, indent=2))
