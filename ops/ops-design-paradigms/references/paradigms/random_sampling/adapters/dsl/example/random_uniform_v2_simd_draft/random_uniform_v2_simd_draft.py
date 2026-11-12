# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference implementation DRAFT: RandomUniformV2 via SIMD VF (Philox4x32-10).

!!! DRAFT —— 未编译、未上板、未与流基准对拍 !!!

RandomSampling 范式 A/C 族（TF 对标）SIMD 路线的结构参考实现，与 AscendC
侧 example/random_uniform_v2_package 同算子对齐（语义契约一致，实现载体
按语言）：
  1. seed/seed2 属性 → host 侧折叠为 Constexpr key 字（构图期常量）
  2. offset 输入张量 → counter 起点换算（⌈offset/4⌉ 字，设备侧完成）
  3. offset 输出 = offset 输入 + N（元素计数递进协议，单写者）
  4. Philox 轮函数 → reg 算子映射（vmull/vbitwise_xor/vadd）
  5. u 构造 → 位重解释口径链（vbitwise_and/vbitwise_or/vreinterpret/vadds）

首次真实接入前必须完成 prng-implementation.md §6 五项对拍义务；
本文件所有"待验证"标注点见同目录 README.md，不得作为已验证实现引用。

依据：prng-model.md §1（流模型/常数）· prng-implementation.md §1/§3 ·
simd-path.md §2/§4 · state-management.md §3（offset 递进）·
compute.md「矢量计算」/ vector-registers.md
"""

import cannbotdsl as cbd
from cannbotdsl import Tensor, dtypes, MemLoc
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.tensor import tile_slice
from cannbotdsl.ops.reg import (
    full_mask,
    varange,
    vdups,
    vmull,
    vadd,
    vadds,
    vbitwise_xor,
    vbitwise_and,
    vbitwise_or,
    vreinterpret,
    vstore,
)
from cannbotdsl.ops.memcpy import mem_copy

# ---------------------------------------------------------------------------
# 常数池（prng-model.md §1.1，语言无关）
# ---------------------------------------------------------------------------
PHILOX_ROUNDS = 10
W32_A = 0x9E3779B9  # key bump W0
W32_B = 0xBB67AE85  # key bump W1
M4X32_A = 0xD2511F53  # mulhilo M0
M4X32_B = 0xCD9E8D57  # mulhilo M1

VL = 64  # fp32：64 lane / 256B（vector-registers.md）
CHUNK_ELEMS = 4 * VL  # k=4 步产出 → 每实例 4 字 × 64 实例 = 256

# u 构造口径：位重解释 [0,1)（含 0；R7——与流基准同路径）
FLOAT_MAN_MASK = 0x7FFFFF
FLOAT_EXP_ONE = 127 << 23


@jit
def _word_to_uniform(w, buf, offset, m):
    """u 构造位重解释口径链：u32 → 尾数位拼接 → float → −1.0。

    依据 prng-implementation.md §3；口径一经选定全程一致（R7 硬规则）。
    """
    bits = vbitwise_or(
        vbitwise_and(w, vdups(FLOAT_MAN_MASK, dtypes.uint32, mask=m), mask=m),
        vdups(FLOAT_EXP_ONE, dtypes.uint32, mask=m),
        mask=m,
    )
    u = vreinterpret(bits, dtypes.float32)
    vstore(buf, offset, vadds(u, -1.0, mask=m), m)


@kernel
def _random_uniform_v2_simd(
    offset_in: Tensor,
    offset_out: Tensor,
    out: Tensor,
    chunk_loops: cbd.Constexpr[int],
    key_w0: cbd.Constexpr[int],
    key_w1: cbd.Constexpr[int],
):
    """RandomUniformV2 SIMD 路线：每个 lane 一个 Philox4x32 实例。

    key：seed/seed2 属性 → host 折叠 Constexpr 字（构图期常量）。
    offset：输入张量运行期读取 → counter 起点换算（⌈offset/4⌉ 字）；
    输出 = 输入 + out.shape[0]（元素计数递进，单写者：本 kernel 单 block）。
    流位置映射（示意）：lane l 第 j 个 chunk 覆盖输出
        [j*CHUNK_ELEMS + w*VL + l]，w∈{0,1,2,3} 对应产出字 c0..c3。
    """
    # ① offset 递进（state-management §3 元素计数口径；单写者）
    offset_out[0] = offset_in[0] + out.shape[0]

    buf = Buffer(MemLoc.UB, (CHUNK_ELEMS,), dtypes.float32)

    k0 = vdups(key_w0, dtypes.uint32, mask=full_mask())  # seed 字：构图期常量
    k1 = vdups(key_w1, dtypes.uint32, mask=full_mask())  # seed2 字：构图期常量
    # counter 起点换算（示意：⌈offset/4⌉ 字；逐位对拍锁死——README A4）
    base_words = dtypes.int64(offset_in[0]) // 4
    lane_i = varange(0, dtypes.int32)  # varange 不支持 uint32
    lane = vreinterpret(lane_i, dtypes.uint32)  # int32 → uint32 位等价

    for j in range(chunk_loops):  # Hardware Loop：从 0 步长 1，体内无运行期分支
        with vf(mode="simd"):
            m = full_mask()
            # counter 字装配（示意：c0 = j*VL + base + lane；c1..c3 由流位置派生）
            c0 = vadd(lane, vdups(j * VL + base_words, dtypes.uint32, mask=m), mask=m)
            c1 = vdups(0, dtypes.uint32, mask=m)
            c2 = vdups(0, dtypes.uint32, mask=m)
            c3 = vdups(0, dtypes.uint32, mask=m)
            m0 = vdups(M4X32_A, dtypes.uint32, mask=m)
            m1 = vdups(M4X32_B, dtypes.uint32, mask=m)

            for _r in cbd.range_constexpr(PHILOX_ROUNDS):
                # mulhilo32 —— vmull 返回 (high, low) 序：待对拍锁死（README §A1）
                hi0, lo0 = vmull(c0, m0, dtypes.uint32, mask=m)
                hi1, lo1 = vmull(c1, m1, dtypes.uint32, mask=m)
                # 单轮：n0 = hi1^c2^k0; n1 = hi0^c3^k1; n2 = lo1; n3 = lo0
                n0 = vbitwise_xor(vbitwise_xor(hi1, c2, mask=m), k0, mask=m)
                n1 = vbitwise_xor(vbitwise_xor(hi0, c3, mask=m), k1, mask=m)
                c0, c1, c2, c3 = n0, n1, lo1, lo0
                # key bump（Weyl 序列）
                k0 = vadd(k0, vdups(W32_A, dtypes.uint32, mask=m), mask=m)
                k1 = vadd(k1, vdups(W32_B, dtypes.uint32, mask=m), mask=m)

            _word_to_uniform(c0, buf, 0 * VL, m)
            _word_to_uniform(c1, buf, 1 * VL, m)
            _word_to_uniform(c2, buf, 2 * VL, m)
            _word_to_uniform(c3, buf, 3 * VL, m)

        mem_copy(tile_slice(out, (CHUNK_ELEMS,), (j,)), buf)  # 搬出在 VF 作用域外


@host
def random_uniform_v2(
    offset_in: Tensor, offset_out: Tensor, out: Tensor, seed: int = 0, seed2: int = 0
) -> None:
    """host 入口：seed/seed2 折叠 Constexpr；chunk_loops 由 out 形状推导。"""
    chunk_loops = (out.shape[0] + CHUNK_ELEMS - 1) // CHUNK_ELEMS
    _random_uniform_v2_simd[1](
        offset_in,
        offset_out,
        out,
        chunk_loops,
        seed & 0xFFFFFFFF,
        seed2 & 0xFFFFFFFF,
    )
