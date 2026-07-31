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

"""把 MFU 天花板信号**机械注入** verification.json —— ko/fo 读的正是这个文件。
这样 MFU 不是"agent 记得跑脚本才用"，而是**每次 perf 后自动进 ko 读的数据**（owner:
建模若 ko/fo 不用就等于没用 -> 把它变成 ko 必然读到的字段）。

集成点（pipeline）:
  perf 步骤产出 verification.json 后 -> 调 inject_mfu_ceiling() -> 写入 mfu_ceiling 块 ->
  ko/fo 的 done 判据读 verification.json['mfu_ceiling']（不依赖 agent 记得跑脚本）。

FLOPs 是 op-语义: 优先读 verification 里 perf-case 的 'flops' 字段(perf 步骤知道 op 时填);
无则用 builtin 公式(matmul/mm_grad); 都没有则该 case skip 并记原因(fail-safe, 不静默造假)。
"""
import json
import sys
import statistics
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from optimizer_signal import mfu_signal
from mfu_model import DTYPE_BYTES

# builtin FLOPs 公式(op_kind -> f(M,K,N)). 扩展新 op 在此加, 或让 perf 步骤直接填 flops 字段。
BUILTIN_FLOPS = {
    "matmul": lambda M, K, N: 2 * M * K * N,
    "mm_grad": lambda M, K, N: 2 * M * K * N * 2,   # dA + dB 两个 GEMM
}
BUILTIN_HBM = {  # 最小 HBM 字节(理想), 用于 memory-bound 判据
    "matmul": lambda M, K, N, sz: (M * K + K * N + M * N) * sz,
    "mm_grad": lambda M, K, N, sz: (M * N + K * N + M * K + M * K + M * N + K * N) * sz,
}


def _parse_mkn(shape_key):
    """'[512, 256, 512]' -> (512,256,512)."""
    return tuple(int(x) for x in shape_key.strip("[]").split(","))


def inject_mfu_ceiling(vjson_path, op_kind="matmul", hw_name="910C_die",
                       n_aic_used=None, t_overhead_us=None, engine_op_kind=None,
                       write=True):
    """读 verification.json -> 每个 perf case 算 MFU 信号 -> 写入 mfu_ceiling 块。
    engine_op_kind: 传给 mfu_signal 的算子类(选 CUBE/VEC peak); 默认同 op_kind。
    返回注入的 mfu_ceiling dict。
    """
    with open(vjson_path) as f:
        v = json.load(f)
    perf = v.get("performance", {})
    ours = perf.get("ours_us_two_runs") or perf.get("ours_us") or {}
    # dtype: 从 precision.cases 按 MKN 匹配, 缺省 fp16
    dtype_by_mkn = {}
    for c in v.get("precision", {}).get("cases", []):
        mkn = tuple(c.get("MKN", [])) if "MKN" in c else None
        if mkn:
            dtype_by_mkn[mkn] = c.get("dtype", "fp16")

    eok = engine_op_kind or ("matmul" if op_kind in ("matmul", "mm_grad") else op_kind)
    per_case, all_done, skipped = {}, [], []
    for shape_key, us_val in ours.items():
        try:
            M, K, N = _parse_mkn(shape_key)
        except Exception:
            skipped.append((shape_key, "unparseable shape"))
            continue
        us = statistics.median(us_val) if isinstance(us_val, list) else float(us_val)
        dt = dtype_by_mkn.get((M, K, N), "fp16")
        dtl = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}.get(dt, dt)
        # flops: 优先 case 内 'flops', 否则 builtin
        ff = BUILTIN_FLOPS.get(op_kind)
        if ff is None:
            skipped.append((shape_key, f"no flops formula for op_kind '{op_kind}'"))
            continue
        flops = ff(M, K, N)
        hf = BUILTIN_HBM.get(op_kind, BUILTIN_HBM["matmul"])
        hbm = hf(M, K, N, DTYPE_BYTES.get(dtl, 2))
        sig = mfu_signal(flops, hbm, us, hw_name, dtl, eok,
                         n_aic_used=n_aic_used, t_overhead_us=t_overhead_us)
        per_case[shape_key] = sig
        all_done.append(sig["done"])

    ceiling = {
        "generated_by": "mfu/verification_hook.py (mechanical, per perf step)",
        "hw": hw_name, "op_kind": op_kind, "engine": eok,
        "per_case": per_case,
        "all_cases_done": bool(all_done) and all(all_done),
        "verdict": ("ALL cases >= 0.8*achievable -> perf DONE (ceiling-aware)"
                    if (all_done and all(all_done))
                    else "CONTINUE: some case below achievable ceiling (see per_case levers)"),
        "skipped": skipped,
        "note": "ko/fo 读此块作 MFU-gated done; 非仅 ratio. peak 按 op_kind 引擎取(fail-loud).",
    }
    v["mfu_ceiling"] = ceiling
    if write:
        with open(vjson_path, "w") as f:
            json.dump(v, f, ensure_ascii=False, indent=2)
    return ceiling


if __name__ == "__main__":
    # demo/self-test on a copy of a real verification.json
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_mmgrad_vj_demo.json"
    c = inject_mfu_ceiling(path, op_kind="mm_grad", hw_name="910C_die", n_aic_used=1)
    print(json.dumps(c, ensure_ascii=False, indent=2))
