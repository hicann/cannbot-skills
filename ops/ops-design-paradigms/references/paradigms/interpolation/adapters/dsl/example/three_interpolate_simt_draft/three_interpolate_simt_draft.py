# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference implementation DRAFT: ThreeInterpolate (IndexedInterpolate family) via SIMT VF.

!!! DRAFT —— 未编译、未上板、未实测 !!!

Interpolation 范式 IndexedInterpolate 算子族的 DSL 结构参考实现，与 AscendC 侧
example/three_interpolate_package 同算子对齐（语义契约一致，实现载体按语言）：
  1. grid-stride 输出元素级遍历 + int64 坐标分解（(b, n, c) 整除/取模）
  2. 显式索引防御：idx 读值升 int64 后双向 clamp（契约保证合法仍保留）
  3. 加权和分步绑定（FMA 收缩行为未实测前的保守写法，capability-mapping §1）
  4. fp32 累加器（half 输入升位；项序与 golden 的 ((p0+p1)+p2) 一致）
  5. 正向逐位直写——无零初始化段（grad 反向才有两段 launch，kernel-entrance §4）

覆盖算子 README 公式 y[b,n,c] = sum_k weight[b,n,k] * features[b, idx[b,n,k], c]
（布局 (B, M, C) -> (B, N, C)）的公共骨架；编译实例管理（lru_cache 键设计）与
多 dtype 分档以 AscendC 样例包为准。首例接入前完成 verification.md 双轨验证
（越界 clamp 一致性 / 权重不归一 / NaN 传播 / 与 AscendC 样例包 golden 对拍）。

依据：sampling-ops.md §1/§2/§4 · kernel-entrance.md §2/§6 ·
capability-mapping.md §1（CAP_FMA_CONTROL 待验证）
"""

import cannbotdsl as cb
from cannbotdsl import Tensor, dtypes
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel

THREADS = 512  # 采样类线程档（坐标映射 + 邻居索引寄存器压力）；按目标核数/形状重核定
CORES = (
    8  # 示意值；按 splitting.md grid-stride 映射核定（vector_core_num 查询，禁写死）
)


@kernel
def _three_interpolate_simt(
    features: Tensor, idx: Tensor, weights: Tensor, output: Tensor
):
    """正向：y[i] = sum_k w[k] * features[b, idx[k], c] —— 输出元素级 grid-stride。

    (b, n, c) 坐标经整除/取模分解（DSL 无 UintDiv 函数族，热点场景实测除法开销，
    必要时 params Tensor 手工 magic/shift）；idx 值运行期直读（host 构图期不可读 GM）。
    """
    with cb.vf(mode="simt"):
        tid = dtypes.int64(cb.simt.thread_idx()[0])
        blk = dtypes.int64(cb.simt.block_idx()[0])
        tpb = dtypes.int64(cb.simt.thread_dim()[0])
        nblk = dtypes.int64(cb.simt.grid_dim()[0])
        start = blk * tpb + tid
        step = nblk * tpb
        bs = features.shape[0]
        ms = features.shape[1]
        cs = features.shape[2]
        ns = idx.shape[1]
        nsc = ns * cs  # (n, c) 平面跨度；b = i // nsc
        total = bs * nsc
        max_idx = dtypes.int64(ms - 1)
        for i in range(start, total, step):
            b = i // nsc
            r = i - b * nsc
            n = r // cs
            c = r - n * cs
            idx_base = b * ns * 3 + n * 3  # int64 域批次基址（b*ms*cs 同防溢出）
            feat_off = b * ms * cs
            raw0 = dtypes.int64(idx[idx_base + 0])
            raw1 = dtypes.int64(idx[idx_base + 1])
            raw2 = dtypes.int64(idx[idx_base + 2])
            safe0 = raw0 if raw0 >= 0 else dtypes.int64(0)  # 双向 clamp 防御
            safe0 = safe0 if safe0 <= max_idx else max_idx  # int64 负值/大值防 OOB
            safe1 = raw1 if raw1 >= 0 else dtypes.int64(0)
            safe1 = safe1 if safe1 <= max_idx else max_idx
            safe2 = raw2 if raw2 >= 0 else dtypes.int64(0)
            safe2 = safe2 if safe2 <= max_idx else max_idx
            w0 = weights[idx_base + 0]
            w1 = weights[idx_base + 1]
            w2 = weights[idx_base + 2]
            f0 = features[feat_off + safe0 * cs + c]
            f1 = features[feat_off + safe1 * cs + c]
            f2 = features[feat_off + safe2 * cs + c]
            # 分步绑定（禁单表达式）：乘积各自独立舍入后再加——FMA 收缩行为
            # 未实测（capability-mapping CAP_FMA_CONTROL 待验证），保守写法与
            # golden 的分步 mul-then-add 同构；项序固定 (p0+p1)+p2。
            p0 = dtypes.float32(w0) * dtypes.float32(f0)
            p1 = dtypes.float32(w1) * dtypes.float32(f1)
            p2 = dtypes.float32(w2) * dtypes.float32(f2)
            output[i] = (
                p0 + p1
            ) + p2  # half 输入升 fp32 累加；出口按 output dtype 落型


@host
def three_interpolate(
    features: Tensor,
    idx: Tensor,
    weights: Tensor,
    output: Tensor,
    cores: cb.Constexpr[int],
) -> None:
    """host 入口：单路径正向 launch（grid-stride 免核间协调，空输出直接不发起）。

    output 以 1D 平铺视图传入（numel = bs*ns*cs）；多 dtype/路径实例经
    cb.compile + lru_cache 管理（缓存键 = 路径/shape/dtype/cores，kernel-entrance §3）。
    """
    if (
        features.shape[0] * idx.shape[1] * features.shape[2] > 0
    ):  # dim3(0) 非法，空输出不 launch
        _three_interpolate_simt[cb.dim3(cores), cb.dim3(THREADS)](
            features, idx, weights, output
        )
