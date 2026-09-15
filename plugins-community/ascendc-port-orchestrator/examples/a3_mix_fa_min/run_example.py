#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# a3 MIX_AIC_1_2 cross-core SYNC-WITNESS — build-and-run harness.
#
# Run this AFTER building the pybind extension (see README.md) to WITNESS the
# AIC<->AIV MIX cross-core handshake close WITHOUT deadlock. The HEADLINE result
# is that torch.npu.synchronize() RETURNS (does not hang) — that is the whole
# point of this witness. The AIV compute between the flags is a PLACEHOLDER
# identity copy (P = S), so the device computes O = (Q @ K^T) @ V (NOT softmax
# attention). We additionally check O against that placeholder-consistent golden
# only to confirm the two GENUINE cubes really ran and the handshake delivered
# S -> P -> cube#2; it is NOT an operator correctness test.
# ============================================================================
import logging
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

# The built _a3_mix_fa_min_example*.so lands in ./build next to this script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import _a3_mix_fa_min_example as ext  # noqa: E402

DEV = "npu:0"  # ASCEND_RT_VISIBLE_DEVICES selects an idle device -> visible index 0
SEQ, D = 128, 64  # the one fixed witness shape
LOGGER = logging.getLogger(__name__)


def run_case(seed=0):
    torch.manual_seed(seed)
    query = torch.randn(SEQ, D, dtype=torch.float16)
    key = torch.randn(SEQ, D, dtype=torch.float16)
    value = torch.randn(SEQ, D, dtype=torch.float16)

    # Placeholder-consistent golden (fp32 compute): the AIV step is identity
    # P = S, so the device chain is O = (Q @ K^T) @ V — NO scale, NO softmax.
    scores = query.float() @ key.float().transpose(-1, -2)
    golden = (scores @ value.float()).half()

    query_npu, key_npu, value_npu = query.to(DEV), key.to(DEV), value.to(DEV)
    t0 = time.time()
    out_npu = ext.run_example(query_npu, key_npu, value_npu)
    torch.npu.synchronize()  # <-- if the MIX handshake deadlocked, we HANG HERE
    dt = (time.time() - t0) * 1e3  # reaching this line == handshake closed
    out = out_npu.cpu().float()

    g = golden.float()
    max_abs = (out - g).abs().max().item()
    denom = g.abs().max().item() + 1e-6
    rel = max_abs / denom
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), g.flatten(), dim=0).item()
    # Cube-chain sanity: the two GENUINE cubes over a large-magnitude fp16 matmul
    # chain; cosine is the whole-tensor signal, rel is scale-normalized max-abs.
    cube_ok = cos > 0.999 and rel < 5e-2
    LOGGER.info(f"[seq={SEQ} d={D}] handshake_closed=True (sync returned in "
                f"{dt:.2f}ms) | cube-chain: max_abs={max_abs:.4e} rel={rel:.4e} "
                f"cosine={cos:.8f} -> {'OK' if cube_ok else 'CHECK'}")
    return cube_ok, out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    LOGGER.info("device: %s visible= %s", DEV, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    # The witness: if the MIX_AIC_1_2 handshake were mis-wired (single-setter
    # reverse), this call would hang forever at torch.npu.synchronize().
    cube_ok, out1 = run_case(seed=0)
    LOGGER.info("WITNESS: MIX_AIC_1_2 AIC<->AIV handshake closed DEADLOCK-FREE "
                "(torch.npu.synchronize returned)")
    LOGGER.info(f"CUBE-CHAIN SANITY: {'OK' if cube_ok else 'CHECK'}")

    # Determinism witness: re-run in-proc and compare bit-exact.
    _, out2 = run_case(seed=0)
    same = torch.equal(out1, out2)
    LOGGER.info(f"DETERMINISM (in-proc re-run bit-exact): {same}")
