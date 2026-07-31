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

"""算子理论最大 MFU 求解器 v0 (L0 硬件模型 + L1 原语 + L3 roofline 求解).

数学口径见 docs/design/OPERATOR_MFU_DEFINITION.md 与 L0_HARDWARE_CONSTANTS.md。
- 理论最大 MFU = T_compute_ideal / T_total_optimal
- T_total_optimal = max(T_compute, T_mem, T_comm)   # 完全通算掩盖上界(理论天花板)
- η_util(tiling/尾块/流水有效利用率) 默认 1.0, 待 msprof 真机标定后下调。

硬件常数来源: docs/archive/A5_HARDWARE_DETAILS.md + ASCEND_FLEET_910C_950DT_SUPERPOD384.md
cube peak 用"1拍 = 2·M·K·N FLOPs"口径(910C 源实测, 950PR FP8 推导≈官方1PFLOPS 交叉验证)。
"""
from dataclasses import dataclass, field
from typing import Optional

# 每个 cube dtype 的"单拍 FLOPs" = 2·M·K·N (矩阵粒度), 来自 L0 常数表
CUBE_FLOPS_PER_CYCLE = {
    "fp32": 2 * 16 * 1 * 16,     # [16x1x16]
    "tf32": 2 * 16 * 8 * 16,     # [16x8x16]
    "fp16": 2 * 16 * 16 * 16,    # [16x16x16] = 8192
    "bf16": 2 * 16 * 16 * 16,    # 8192
    "fp8": 2 * 16 * 32 * 16,     # [16x32x16] = 16384
    "int8": 2 * 16 * 32 * 16,    # 16384
    "mxfp4": 2 * 16 * 64 * 16,   # [16x64x16] = 32768
    "int4": 2 * 16 * 64 * 16,    # 32768
}
DTYPE_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "mxfp4": 0.5, "int4": 0.5}


@dataclass
class HardwareModel:
    """描述设备算力、带宽和片上存储容量的简化硬件模型。"""

    name: str
    n_aic: int                    # cube core 数
    freq_hz: float                # 主频
    hbm_bw: float                 # HBM 带宽 B/s
    hbm_bytes: float              # HBM 容量 B
    interconnect_bw: float = 0.0  # 卡间/die间 互联 B/s (单向有效)
    arch: str = "SIMD"            # SIMD(A3) / SIMT(A5)
    l2_bw: float = 0.0            # L2 读带宽 B/s (聚合); 0=不建模L2级
    l2_bytes: float = 0.0         # L2 容量 B
    l1_bytes: float = 0.0         # L1 容量 B (matmul 复用级)
    l0c_bytes: float = 0.0        # L0C(输出累加器)容量 B
    # 直接给定的 peak(若有官方值, 覆盖推导), key=dtype -> FLOP/s
    peak_override: dict = field(default_factory=dict)
    vec_peak: dict = field(default_factory=dict)  # 向量 dtype -> FLOP/s

    def cube_peak(self, dtype: str) -> float:
        """返回指定数据类型的 Cube 峰值算力；显式配置优先。"""

        if dtype in self.peak_override:
            return self.peak_override[dtype]
        fpc = CUBE_FLOPS_PER_CYCLE.get(dtype)
        if fpc is None:
            raise ValueError(f"{self.name}: cube 不支持 dtype {dtype}")
        return self.n_aic * fpc * self.freq_hz

    def ridge_ai(self, dtype: str, engine="cube") -> float:
        """返回指定计算引擎的 Roofline 拐点算术强度。"""

        peak = self.cube_peak(dtype) if engine == "cube" else self.vec_peak[dtype]
        return peak / self.hbm_bw


@dataclass
class OpSpec:
    """单算子的某个阶段(fwd 或 bwd)的 FLOPs/bytes/通信 记账。"""
    name: str
    flops: float                  # useful FLOPs
    hbm_bytes: float              # 最小 HBM 访存字节(理想 tiling, 每数据读一次)
    dtype: str = "bf16"
    engine: str = "cube"          # cube / vector
    comm_bytes: float = 0.0       # 分布式通信字节(单向有效量)


def solve_mfu(op: OpSpec, hw: HardwareModel, eta_util: float = 1.0) -> dict:
    """求理论最大 MFU + 瓶颈归因。eta_util=有效利用率(默认1=纯理论天花板)。"""
    peak = hw.cube_peak(op.dtype) if op.engine == "cube" else hw.vec_peak[op.dtype]
    t_compute = op.flops / peak
    t_mem = op.hbm_bytes / hw.hbm_bw
    t_comm = op.comm_bytes / hw.interconnect_bw if (op.comm_bytes and hw.interconnect_bw) else 0.0
    t_total = max(t_compute, t_mem, t_comm)            # 完全掩盖上界
    mfu_max = (t_compute / t_total) * eta_util if t_total > 0 else 0.0
    times = {"compute": t_compute, "mem": t_mem, "comm": t_comm}
    bottleneck = max(times, key=times.get)
    ai = op.flops / op.hbm_bytes if op.hbm_bytes else float("inf")
    return {
        "op": op.name, "hw": hw.name, "dtype": op.dtype,
        "mfu_max": round(mfu_max, 4),
        "bottleneck": bottleneck,
        "arithmetic_intensity": round(ai, 2),
        "ridge_ai": round(hw.ridge_ai(op.dtype, op.engine), 2),
        "peak_TFLOPS": round(peak / 1e12, 1),
        "t_compute_us": round(t_compute * 1e6, 3),
        "t_mem_us": round(t_mem * 1e6, 3),
        "t_comm_us": round(t_comm * 1e6, 3),
        "eta_util": eta_util,
    }


# ---------- L1 原语: matmul 前向/后向 FLOPs/bytes 记账 ----------
def matmul_ops(M, N, K, dtype="bf16"):
    """Y=X·W, X[M,K] W[K,N]. 返回 (fwd, bwd) 两个 OpSpec。
    bytes = 理想最小 HBM 流量 = 各 tensor 读/写一次。
    """
    sz = DTYPE_BYTES[dtype]
    fwd = OpSpec("matmul.fwd", flops=2 * M * N * K,
                 hbm_bytes=(M * K + K * N + M * N) * sz, dtype=dtype)
    # bwd: dX = dY·Wᵀ (2MNK, 读 dY[M,N]+W[K,N], 写 dX[M,K]); dW = Xᵀ·dY (2MNK, 读 X[M,K]+dY[M,N], 写 dW[K,N])
    bwd = OpSpec("matmul.bwd", flops=4 * M * N * K,
                 hbm_bytes=(M * N + K * N + M * K + M * K + M * N + K * N) * sz, dtype=dtype)
    return fwd, bwd


# ---------- L1 原语: FlashAttention 前向/后向 (cube+vector 混合) ----------
def flash_attention_ops(B, H, S, D, dtype="bf16", causal=False):
    """Q,K,V [B,H,S,D]. FA tiling 不物化 S=[B,H,S,S], 故 HBM 仅读 QKV 写 O。
    cube FLOPs 主导(QKᵀ + P·V), softmax 走 vector(量级小, 此处计入但归一仍用 cube)。
    bwd ≈ 2.5× fwd cube FLOPs(重算 + dQ/dK/dV)。causal 减半有效 S² 工作量。
    """
    sz = DTYPE_BYTES[dtype]
    causal_factor = 0.5 if causal else 1.0
    # fwd: QKᵀ=2BHS²D, P·V=2BHS²D
    fwd_cube = 4 * B * H * S * S * D * causal_factor
    fwd = OpSpec("FA.fwd", flops=fwd_cube,
                 hbm_bytes=(3 * B * H * S * D + B * H * S * D) * sz,  # 读QKV + 写O
                 dtype=dtype, engine="cube")
    bwd = OpSpec("FA.bwd", flops=2.5 * fwd_cube,
                 hbm_bytes=(5 * B * H * S * D + 3 * B * H * S * D) * sz,  # 读QKVO·dO + 写dQdKdV
                 dtype=dtype, engine="cube")
    return fwd, bwd


# ---------- L2(部分): 分布式集合通信 = 通算掩盖输入 ----------
def allreduce_bytes(msg_bytes, n_dev, algo="ring"):
    """带宽最优 AllReduce 每设备通信量。ring/halving-doubling 同为 2·(N-1)/N·M。"""
    if n_dev <= 1:
        return 0.0
    return 2.0 * (n_dev - 1) / n_dev * msg_bytes


def with_data_parallel(op: OpSpec, grad_bytes: float, n_dev: int):
    """训练数据并行: 在算子(通常 bwd 的 dW)上叠加梯度 AllReduce 通信, 走通算掩盖。
    返回带 comm_bytes 的新 OpSpec(浅拷贝)。
    """
    comm = allreduce_bytes(grad_bytes, n_dev)
    return OpSpec(op.name + f".dp{n_dev}", flops=op.flops, hbm_bytes=op.hbm_bytes,
                  dtype=op.dtype, engine=op.engine, comm_bytes=comm)


# ---------- L4: 单算子 -> 训练 e2e MFU 聚合 (OPERATOR_MFU_DEFINITION §4 契约) ----------
def _matmul_time(M, N, K, hw, dtype, flop_mult=1.0):
    """matmul 的 (useful_flops, T_total_optimal秒)。flop_mult: fwd=1, fwd+bwd≈3。"""
    flops, hbm, l2 = matmul_tiled_levels(M, N, K, hw, dtype)
    flops *= flop_mult
    r = solve_mfu_multilevel(flops, hw.cube_peak(dtype),
                             [(hw.hbm_bw, hbm * flop_mult), (hw.l2_bw, l2 * flop_mult)])
    return flops, max(r["t_us"].values()) / 1e6


def _vec_time(op, hw, flop_mult=1.0):
    r = solve_mfu(op, hw)
    t = max(r["t_compute_us"], r["t_mem_us"], r["t_comm_us"]) / 1e6 * flop_mult
    return op.flops * flop_mult, t


def transformer_layer_e2e(B, S, d_model, d_ff, n_head, hw, dtype="bf16", training=True):
    """一层 Transformer 的训练 e2e MFU = Σuseful_FLOPs / (cube_peak × Σ各算子最优时间)。
    展示单算子 MFU 如何聚合到 e2e: 向量算子(norm/softmax)FLOPs 小但 memory-bound 耗时,
    会拖低 e2e MFU —— 这正是 e2e MFU agent 要复用的契约。
    """
    M = B * S
    fm = 3.0 if training else 1.0          # matmul fwd+bwd≈3x flops
    fam = 3.5 if training else 1.0         # attention fwd+bwd≈3.5x
    items = []  # (name, flops, t)
    for name, N, K in [("qkv_proj", 3 * d_model, d_model), ("o_proj", d_model, d_model),
                       ("ffn_up", d_ff, d_model), ("ffn_down", d_model, d_ff)]:
        items.append((name, *_matmul_time(M, N, K, hw, dtype, fm)))
    fwd, bwd = flash_attention_ops(B, n_head, S, d_model // n_head, dtype)
    items.append(("attention", *_vec_time(fwd, hw, fam)))     # FA cube 主导, 用其 flops/time×mult
    for name, opf, args in [("rmsnorm_attn", rmsnorm_op, (M, d_model)),
                            ("rmsnorm_ffn", rmsnorm_op, (M, d_model)),
                            ("softmax_skip", elementwise_op, (M * d_ff,))]:  # 占位非线性
        items.append((name, *_vec_time(opf(*args), hw, 2.0 if training else 1.0)))
    total_flops = sum(f for _, f, _ in items)
    total_time = sum(t for _, _, t in items)
    cube_peak = hw.cube_peak(dtype)
    mfu = total_flops / (cube_peak * total_time) if total_time else 0.0
    # 各算子耗时占比(找瓶颈)
    breakdown = sorted(((n, round(t / total_time * 100, 1)) for n, _, t in items),
                       key=lambda x: -x[1])
    return {"hw": hw.name, "e2e_MFU": round(mfu, 4),
            "total_TFLOPs": round(total_flops / 1e12, 2),
            "step_us": round(total_time * 1e6, 1),
            "time_pct": breakdown}


# ---------- L0 硬件实例 (常数源: docs/design/L0_HARDWARE_CONSTANTS.md) ----------
_L2_BW = 5.28e12  # L2 读带宽聚合: 910C 48核×110GB/s≈5.28TB/s; 950PR 官方5.28TB/s (950DT 待核实,暂同)
HW = {
    # 910C 整卡(双die ~48 AIC), HBM 3.2TB/s 128GB, 互联 784GB/s 双向→单向~392
    "910C": HardwareModel("910C", n_aic=48, freq_hz=1.85e9,
                          hbm_bw=3.2e12, hbm_bytes=128 * 2**30,
                          interconnect_bw=784e9 / 2, arch="SIMD",
                          l2_bw=_L2_BW, l2_bytes=192 * 2**20, l1_bytes=512 * 2**10, l0c_bytes=128 * 2**10),
    # 950PR PG(实测card2: 28 cube/56 vec, L2 112MB, soc 957b), HBM 1.6TB/s 128GB; AG变体=36 AIC
    "950PR": HardwareModel("950PR", n_aic=28, freq_hz=1.65e9,
                           hbm_bw=1.6e12, hbm_bytes=128 * 2**30,
                           interconnect_bw=2e12, arch="SIMT",
                           l2_bw=_L2_BW, l2_bytes=112 * 2**20, l1_bytes=512 * 2**10, l0c_bytes=256 * 2**10),
    # 950DT: 官方 FP8 1P/MXFP4 2P -> 反推 cube; HBM 4TB/s 144GB, 互联 2TB/s
    "950DT": HardwareModel("950DT", n_aic=36, freq_hz=1.65e9,
                           hbm_bw=4e12, hbm_bytes=144 * 2**30,
                           interconnect_bw=2e12, arch="SIMT",
                           l2_bw=_L2_BW, l2_bytes=128 * 2**20, l1_bytes=512 * 2**10, l0c_bytes=256 * 2**10,
                           peak_override={"fp8": 1e15, "mxfp4": 2e15,
                                          "bf16": 0.5e15, "fp16": 0.5e15}),
}
# SuperPod384 = Atlas900 A3 SuperPoD: 单卡=910C, 但超节点内卡间互联 784GB/s 双向(1:1无收敛)
HW["910C_superpod"] = HardwareModel(
    "910C@SuperPod384", n_aic=48, freq_hz=1.85e9,
    hbm_bw=3.2e12, hbm_bytes=128 * 2**30,
    interconnect_bw=784e9, arch="SIMD",
    l2_bw=_L2_BW, l2_bytes=192 * 2**20, l1_bytes=512 * 2**10, l0c_bytes=128 * 2**10)

# 910C 单 die (Ascend910_9392, torch_npu 单设备粒度 = 24 AIC/64GB; HW['910C']是整卡48AIC)
# 实测标定 2026-06-26 A3: bf16 matmul 实测峰值 ~321 TFLOPS (n=4096), 理论 363 -> η_util≈0.88, 天花板不破。
HW["910C_die"] = HardwareModel(
    "910C_die", n_aic=24, freq_hz=1.85e9,
    hbm_bw=1.6e12, hbm_bytes=64 * 2**30,           # 单die 64GB; 单die HBM BW 待msprof(暂半卡)
    interconnect_bw=784e9 / 2, arch="SIMD",
    l2_bw=_L2_BW / 2, l2_bytes=96 * 2**20, l1_bytes=512 * 2**10, l0c_bytes=128 * 2**10)
# 实测标定的有效利用率 η_util (2026-06-26 A3 Ascend910_9392 单die):
#   matmul 大方阵 compute-bound: 0.88 (纯 cube)
#   FlashAttention(fusion_attn 最优核): ~0.53 vs cube peak —— cube+vector 交织代价,
#     softmax(向量)阶段无法与 cube 完全掩盖 -> 混合算子天花板 < 纯 cube peak。
ETA_UTIL = {
    # 实测标定 2026-06-26. 关键: A5(SIMT) FA η 远高于 A3(SIMD) -> SIMT cube+vector 掩盖更好。
    "matmul": {"910C_die": 0.88, "950PR": 0.98},
    "flash_attention": {"910C_die": 0.53, "950PR": 0.87},
}
# η-provenance (main review 2026-07-01 要求): 每个 η 的来源+日期+re-validate 触发。
# stale-η -> 错估 headroom, 故硬件/CANN/bisheng 版本变时必须 re-validate。
ETA_PROVENANCE = {
    ("matmul", "910C_die"): {
        "value": 0.88,
        "measured": "2026-06-26",
        "method": "A3 Ascend910_9392 bf16 matmul n=4096 device-time vs 理论363TF",
        "revalidate_on": "SOC/CANN/bisheng 版本变 或 主频变",
    },
    ("matmul", "950PR"): {
        "value": 0.98,
        "measured": "2026-06-26",
        "method": "A5 950PR_957b bf16 matmul n=8192 372.7 vs 378.6TF",
        "revalidate_on": "同上",
    },
    ("flash_attention", "910C_die"): {
        "value": 0.53,
        "measured": "2026-06-26",
        "method": "A3 npu_fusion_attention 170TF vs cube peak 363, msprof cube_mac 0.50",
        "revalidate_on": "同上 + FA kernel 版本变",
    },
    ("flash_attention", "950PR"): {
        "value": 0.87,
        "measured": "2026-06-26",
        "method": "A5 npu_fusion_attention 330TF vs 378.6, msprof cube_mac 0.98",
        "revalidate_on": "同上",
    },
}


def eta_with_provenance(op_kind, hw_name):
    """返回 (η, provenance)。无 provenance 的 η = 未标定/stale 风险, 调用方应警惕。"""
    p = ETA_PROVENANCE.get((op_kind, hw_name))
    eta = ETA_UTIL.get(op_kind, {}).get(hw_name)
    return eta, p


def achievable_mfu_with_overhead(flops, peak, op_kind, hw_name, t_overhead_s=0.0, default_eta=0.85):
    """v2: 理论 compute 时间 / η + 固定 launch/tail overhead -> 可达 MFU。
    B组(2026-06-29 A3 FA)实测暴露: v1理想roofline的陡峭ridge过乐观, 小算子受
    kernel-launch/尾块固定开销主导。加 T_overhead 项后, A3 FA MFU-vs-S 全程拟合实测±几%。
    A3 FA fusion_attn 拟合: t_overhead≈0.115ms, η≈0.53。
    """
    eta = ETA_UTIL.get(op_kind, {}).get(hw_name, default_eta)
    t_compute_ideal = flops / peak
    t_total = t_compute_ideal / eta + t_overhead_s
    return {"achievable_mfu": round(t_compute_ideal / t_total, 4),
            "eta_util": eta, "t_overhead_s": t_overhead_s,
            "t_compute_ideal_s": t_compute_ideal}


def achievable_mfu(theoretical_mfu_max, op_kind, hw_name, default_eta=0.85):
    """把实测标定的 η_util 套到理论天花板上 -> 预测"可达 MFU"。
    返回 (theoretical_ceiling, eta, achievable)。优化时: 实测<achievable=>还有实现空间;
    实测≈achievable=>已接近该算子类的实际上限; 要再高需改算法(降 η 的机理)。
    """
    eta = ETA_UTIL.get(op_kind, {}).get(hw_name, default_eta)
    return {"theoretical_ceiling": theoretical_mfu_max,
            "eta_util": eta,
            "achievable_mfu": round(theoretical_mfu_max * eta, 4)}


# 向量算力 (源: 950PR FP16 54TOPS FMA / 27 单算; 910C 推导~22.7T 待msprof; 950DT 暂同950PR)
HW["950PR"].vec_peak = {"fp16": 54e12, "bf16": 54e12, "fp32": 27e12}
HW["950DT"].vec_peak = {"fp16": 54e12, "bf16": 54e12, "fp32": 27e12}
HW["910C"].vec_peak = {"fp16": 22.7e12, "bf16": 22.7e12, "fp32": 11.3e12}
HW["910C_superpod"].vec_peak = HW["910C"].vec_peak


# ---------- L1 原语: 非 matmul (向量主导, 通常 memory-bound) ----------
def elementwise_op(n_elem, dtype="bf16", flops_per_elem=1, n_read=1, n_write=1, name="elementwise"):
    """逐元素算子(add/gelu/激活...). 向量引擎. bytes=读n_read+写n_write个张量。"""
    sz = DTYPE_BYTES[dtype]
    return OpSpec(name, flops=n_elem * flops_per_elem,
                  hbm_bytes=n_elem * (n_read + n_write) * sz, dtype=dtype, engine="vector")


def softmax_op(rows, cols, dtype="bf16"):
    """softmax: 读1写1(融合) + 每元素~5 FLOPs(max,exp,sum,div). 强 memory-bound。"""
    n = rows * cols
    return OpSpec("softmax", flops=5 * n, hbm_bytes=n * 2 * DTYPE_BYTES[dtype],
                  dtype=dtype, engine="vector")


def rmsnorm_op(rows, cols, dtype="bf16"):
    """RMSNorm: 读x+读weight+写y, 每元素~4 FLOPs(平方/和/rsqrt/乘)。"""
    n = rows * cols
    return OpSpec("rmsnorm", flops=4 * n,
                  hbm_bytes=(n + cols + n) * DTYPE_BYTES[dtype], dtype=dtype, engine="vector")


# ---------- L2 调度演算: 多级内存 roofline + matmul tiling ----------
def _matmul_io_lb(M, N, K, cap_bytes, sz):
    """matmul I/O 下界 (Hong-Kung): 容量 cap 的快存下, 慢存最少搬运字节。
    = max( 2MNK/sqrt(cap_elems) ,  读A+读B+写C 各一次 )。这是**最优 tiling**的搬运量,
    故对应"理论最优调度"(理论最大 MFU), 而非某个具体次优 tiling。"""
    import math
    cap_elems = max(cap_bytes / sz, 1.0)
    q_reuse = 2.0 * M * N * K / math.sqrt(cap_elems)
    q_min = M * K + K * N + M * N
    return max(q_reuse, q_min) * sz


def matmul_tiled_levels(M, N, K, hw: HardwareModel, dtype="bf16"):
    """matmul 多级访存量(理论最优 tiling, I/O 下界):
      - hbm_bytes: HBM↔L2 搬运, 快存=L2 容量复用
      - l2_bytes : L2↔L1 搬运, 快存=L1 容量复用
    返回 (flops, hbm_bytes, l2_bytes)。
    """
    sz = DTYPE_BYTES[dtype]
    hbm_bytes = _matmul_io_lb(M, N, K, hw.l2_bytes, sz) if hw.l2_bytes else (M * K + K * N + M * N) * sz
    l2_bytes = _matmul_io_lb(M, N, K, hw.l1_bytes, sz) if hw.l1_bytes else 0.0
    return 2 * M * N * K, hbm_bytes, l2_bytes


def solve_mfu_multilevel(flops, peak, mem_levels, comm=(0.0, 0.0), eta_util=1.0):
    """多级 roofline: mem_levels=[(bw,bytes),...] 各级并行可掩盖, comm=(bw,bytes)。
    T_total = max(T_compute, 各级 T_mem, T_comm) (完全掩盖上界)。
    """
    t_compute = flops / peak
    t_levels = {f"mem_L{i}": (b / bw if bw else 0.0) for i, (bw, b) in enumerate(mem_levels)}
    t_comm = comm[1] / comm[0] if (comm[0] and comm[1]) else 0.0
    times = {"compute": t_compute, "comm": t_comm, **t_levels}
    t_total = max(times.values())
    mfu = (t_compute / t_total) * eta_util if t_total > 0 else 0.0
    bottleneck = max(times, key=times.get)
    return {"mfu_max": round(mfu, 4), "bottleneck": bottleneck,
            "peak_TFLOPS": round(peak / 1e12, 1),
            "t_us": {k: round(v * 1e6, 2) for k, v in times.items()}, "eta_util": eta_util}


def solve_matmul_tiled(M, N, K, hw: HardwareModel, dtype="bf16", n_dev=1, eta_util=1.0):
    """matmul 端到端: tiling 多级访存 + (可选)数据并行梯度通信。"""
    flops, hbm_b, l2_b = matmul_tiled_levels(M, N, K, hw, dtype)
    comm = (0.0, 0.0)
    if n_dev > 1:
        comm = (hw.interconnect_bw, allreduce_bytes(K * N * DTYPE_BYTES[dtype], n_dev))
    peak = hw.cube_peak(dtype)
    r = solve_mfu_multilevel(flops, peak, [(hw.hbm_bw, hbm_b), (hw.l2_bw, l2_b)], comm, eta_util)
    r.update({"op": f"matmul[{M}x{N}x{K}]", "hw": hw.name, "dtype": dtype})
    return r
