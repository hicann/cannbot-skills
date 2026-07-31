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

"""MFU 求解器 demo: matmul 在 910C/950PR/950DT 上的理论最大 MFU(前向+后向)。"""
from mfu_model import (HW, matmul_ops, flash_attention_ops, with_data_parallel,
                       solve_mfu, solve_matmul_tiled, DTYPE_BYTES)


def run(M, N, K, dtype):
    print(f"\n==== matmul M={M} N={N} K={K} dtype={dtype} ====")
    fwd, bwd = matmul_ops(M, N, K, dtype)
    hdr = f"{'hw':7} {'stage':11} {'MFU_max':>7} {'bottleneck':>10} {'AI':>9} {'AI*':>9} {'peakTF':>7}"
    print(hdr)
    for hwname, hw in HW.items():
        for op in (fwd, bwd):
            try:
                r = solve_mfu(op, hw)
            except ValueError as e:
                print(f"{hwname:7} {op.name:11} skip ({e})")
                continue
            print(f"{r['hw']:7} {r['op']:11} {r['mfu_max']:>7} {r['bottleneck']:>10} "
                  f"{r['arithmetic_intensity']:>9} {r['ridge_ai']:>9} {r['peak_TFLOPS']:>7}")


def run_fa(B, H, S, D, dtype="bf16", causal=False):
    print(f"\n==== FlashAttention B={B} H={H} S={S} D={D} dtype={dtype} causal={causal} ====")
    fwd, bwd = flash_attention_ops(B, H, S, D, dtype, causal)
    print(f"{'hw':18} {'stage':8} {'MFU_max':>7} {'bottleneck':>10} {'AI':>8} {'AI*':>8}")
    for hw in (HW["910C"], HW["950PR"], HW["950DT"]):
        for op in (fwd, bwd):
            r = solve_mfu(op, hw)
            print(f"{r['hw']:18} {r['op']:8} {r['mfu_max']:>7} {r['bottleneck']:>10} "
                  f"{r['arithmetic_intensity']:>8} {r['ridge_ai']:>8}")


def run_dist_train(M, N, K, dtype, n_dev):
    """训练数据并行: bwd 的 dW 叠加梯度 AllReduce, 看通信是否被 compute 掩盖。"""
    print(f"\n==== 分布式训练 matmul M={M} N={N} K={K} dtype={dtype} DP={n_dev} (910C@SuperPod384) ====")
    _, bwd = matmul_ops(M, N, K, dtype)
    grad_bytes = K * N * DTYPE_BYTES[dtype]      # dW 梯度大小
    bwd_dp = with_data_parallel(bwd, grad_bytes, n_dev)
    hw = HW["910C_superpod"]
    r = solve_mfu(bwd_dp, hw)
    print(f"  MFU_max={r['mfu_max']} bottleneck={r['bottleneck']} "
          f"t_compute={r['t_compute_us']}us t_comm={r['t_comm_us']}us "
          f"-> 通信{'被掩盖' if r['bottleneck']!='comm' else '未掩盖(comm-bound)'}")


if __name__ == "__main__":
    # 大方阵(compute-bound 应 MFU_max≈1) vs 瘦长(memory-bound 应 <1)
    run(4096, 4096, 4096, "bf16")
    run(4096, 4096, 4096, "fp8")
    run(1, 4096, 4096, "bf16")   # GEMV-like, memory-bound
    # FlashAttention (cube+vector 混合, FA tiling 高算术强度)
    run_fa(2, 32, 4096, 128, "bf16")
    run_fa(2, 32, 512, 128, "bf16")   # 短序列 -> AI 降
    # 分布式训练通算掩盖: 大矩阵 compute 多 -> 通信易掩盖; 小矩阵 -> 易 comm-bound
    run_dist_train(8192, 8192, 8192, "bf16", 8)
    run_dist_train(512, 8192, 8192, "bf16", 64)
    # L2 调度演算: 多级 roofline (tiling 受 L0C 限制 -> L2 re-read)
    print("\n==== tiled matmul (多级 roofline: HBM+L2, L0C限tile) ====")
    print("  (绝对值待 msprof 校准 + 加 L1 复用级; 结构示意 L0C小->L2-bound 机理)")
    for M, N, K in [(4096, 4096, 4096), (256, 256, 8192)]:
        for hwn in ("910C", "950PR", "950DT"):
            r = solve_matmul_tiled(M, N, K, HW[hwn], "bf16")
            print(f"  {hwn:6} {r['op']:22} MFU={r['mfu_max']:<6} bn={r['bottleneck']}")
