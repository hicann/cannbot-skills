# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# A5 (Ascend950PR_957b) cube fp16/fp32 peak calibration — ground-truth on-NPU
# Mirrors the A3 method (2026-05-28): time large square matmuls, TFLOPS=2MNK/t.
import time
import torch
import torch_npu  # noqa: E401,F401
dev = "npu:0"  # pinned via ASCEND_RT_VISIBLE_DEVICES


def bench(matrix_size, dtype, iters=30, warm=10):
    a = torch.randn(matrix_size, matrix_size, dtype=dtype, device=dev)
    b = torch.randn(matrix_size, matrix_size, dtype=dtype, device=dev)
    for _ in range(warm):
        _ = torch.matmul(a, b)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    torch.npu.synchronize()
    dts = (time.time() - t0) / iters
    tflops = 2.0 * matrix_size * matrix_size * matrix_size / dts / 1e12
    return dts * 1e3, tflops


print("soc=Ascend950PR_957b  matmul cube-peak calibration")
for dt, name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    best = 0.0
    for n in (2048, 4096, 8192):
        try:
            ms, tf = bench(n, dt)
            best = max(best, tf)
            print(f"  {name}  N={n:5d}  {ms:8.3f} ms  {tf:8.1f} TFLOPS")
        except Exception as e:
            print(f"  {name}  N={n:5d}  ERR {e}")
    print(f"  => {name} achieved cube peak ~ {best:.0f} TFLOPS")
