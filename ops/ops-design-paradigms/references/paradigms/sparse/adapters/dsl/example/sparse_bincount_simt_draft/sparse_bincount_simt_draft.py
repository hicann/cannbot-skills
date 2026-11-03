# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference implementation DRAFT: SparseBincount (Count family) via SIMT VF.

!!! DRAFT —— 未编译、未上板、未实测 !!!

Sparse 范式 Count 算子族的 DSL 结构参考实现，与 AscendC 侧
example/sparse_bincount_package 同算子对齐（语义契约一致，实现载体按语言）：
  1. 两段顺序 launch：零初始化 + 散射累积（同 stream 排序免 SyncAll）
  2. host 侧路径分派：计数 / 加权 / binary_output 三路径独立编译实例
  3. 索引驱动访存：grid-stride + cb.simt.atomic_add（标量点加轨）
  4. binary_output 幂等直写消除原子
  5. 值依赖输入（size / dense_shape）设备侧运行期读取

覆盖算子 README 两公式（1D: output[v] += w_i；Multi-dim: output[batch][v]
+= w_i，v/branch 越界丢弃）的公共骨架，batch 取 indices 首维折叠示意；
多维 stride 完整展开与 -1 维推断以 AscendC 样例包为准。首例接入前完成
verification.md 双轨验证（重复索引守恒 / 越界丢弃 / binary 幂等）。

依据：access-model.md §3（冲突决策树）· atomic-ops.md §1/§2/§4 ·
kernel-entrance.md §2/§4/§5 · compute.md「SIMT 模式」
"""

import cannbotdsl as cb
from cannbotdsl import Tensor, dtypes
from cannbotdsl.lang.kernel import kernel, host

THREADS = 256  # 计算类（原子/分支）线程档；具体值按目标核数/形状重核定
CORES = 8  # 示意值；按 splitting.md grid-stride 映射核定


@kernel
def _bincount_zero(output: Tensor):
    """阶段 ①：零初始化（幂等常量写；与阶段 ② 同 stream 顺序免栅栏）。"""
    with cb.vf(mode="simt"):
        tid = dtypes.int64(cb.simt.thread_idx()[0])
        blk = dtypes.int64(cb.simt.block_idx()[0])
        tpb = dtypes.int64(cb.simt.thread_dim()[0])
        nblk = dtypes.int64(cb.simt.grid_dim()[0])
        start = blk * tpb + tid
        step = nblk * tpb
        for j in range(start, output.shape[0], step):
            output[j] = 0


@kernel
def _bincount_count(
    values: Tensor,
    indices: Tensor,
    size_t: Tensor,
    dense_shape_t: Tensor,
    output: Tensor,
):
    """阶段 ②（计数路径）：output[batch][v] += 1 —— 标量原子点加。

    v = values[i]（bin index，运行期读）；batch = indices[i, 0]（首维折叠
    示意，1D 模式为其 batch=0 特例）；size / dense_shape[0] 为值依赖输入，
    设备侧运行期读取（host 构图期不可读 GM——kernel-entrance §5）。
    """
    with cb.vf(mode="simt"):
        tid = dtypes.int64(cb.simt.thread_idx()[0])
        blk = dtypes.int64(cb.simt.block_idx()[0])
        tpb = dtypes.int64(cb.simt.thread_dim()[0])
        nblk = dtypes.int64(cb.simt.grid_dim()[0])
        start = blk * tpb + tid
        step = nblk * tpb
        n = values.shape[0]
        rank = indices.shape[1]
        size = dtypes.int64(size_t[0])  # 值依赖：运行期读
        batch_count = dtypes.int64(dense_shape_t[0])  # 值依赖：运行期读
        for i in range(start, n, step):
            v = dtypes.int64(values[i])
            if v >= 0 and v < size:
                batch = dtypes.int64(indices[i * rank])
                if batch >= 0 and batch < batch_count:
                    off = batch * size + v
                    cb.simt.atomic_add(output.ptr(off), dtypes.int64(1))


@kernel
def _bincount_weighted(
    values: Tensor,
    indices: Tensor,
    weights: Tensor,
    size_t: Tensor,
    dense_shape_t: Tensor,
    output: Tensor,
):
    """阶段 ②（加权路径）：output[batch][v] += w_i —— float32 原子点加。

    weights 为 fp32；fp16/bf16 输出域走 fp32 累积 + 出口 cast 或批量随路
    （atomic-ops.md §1 双轨选型），本草稿不展开。
    """
    with cb.vf(mode="simt"):
        tid = dtypes.int64(cb.simt.thread_idx()[0])
        blk = dtypes.int64(cb.simt.block_idx()[0])
        tpb = dtypes.int64(cb.simt.thread_dim()[0])
        nblk = dtypes.int64(cb.simt.grid_dim()[0])
        start = blk * tpb + tid
        step = nblk * tpb
        n = values.shape[0]
        rank = indices.shape[1]
        size = dtypes.int64(size_t[0])
        batch_count = dtypes.int64(dense_shape_t[0])
        for i in range(start, n, step):
            v = dtypes.int64(values[i])
            if v >= 0 and v < size:
                batch = dtypes.int64(indices[i * rank])
                if batch >= 0 and batch < batch_count:
                    off = batch * size + v
                    cb.simt.atomic_add(output.ptr(off), weights[i])


@kernel
def _bincount_binary(
    values: Tensor,
    indices: Tensor,
    size_t: Tensor,
    dense_shape_t: Tensor,
    output: Tensor,
):
    """阶段 ②（binary_output 路径）：命中即写 1 —— 幂等直写，免原子。"""
    with cb.vf(mode="simt"):
        tid = dtypes.int64(cb.simt.thread_idx()[0])
        blk = dtypes.int64(cb.simt.block_idx()[0])
        tpb = dtypes.int64(cb.simt.thread_dim()[0])
        nblk = dtypes.int64(cb.simt.grid_dim()[0])
        start = blk * tpb + tid
        step = nblk * tpb
        n = values.shape[0]
        rank = indices.shape[1]
        size = dtypes.int64(size_t[0])
        batch_count = dtypes.int64(dense_shape_t[0])
        for i in range(start, n, step):
            v = dtypes.int64(values[i])
            if v >= 0 and v < size:
                batch = dtypes.int64(indices[i * rank])
                if batch >= 0 and batch < batch_count:
                    output[batch * size + v] = 1  # 全核同值幂等（I3 同款论证）


@host
def sparse_bincount(
    values: Tensor,
    indices: Tensor,
    dense_shape: Tensor,
    size: Tensor,
    output: Tensor,
    weights: Tensor = None,
    binary_output: bool = False,
) -> None:
    """host 入口：路径分派（atomic-ops §2）+ 两段顺序 launch。

    output 以 1D 平铺视图传入（numel = batch_count * size，2D 语义经
    shape 元数据表达）；计数 +1 的 dtype 按 output 分派（int32 输出改
    dtypes.int32(1)），本草稿按 int64 输出示意。
    """
    _bincount_zero[cb.dim3(CORES), cb.dim3(THREADS)](output)  # ① 零初始化
    if binary_output:
        _bincount_binary[cb.dim3(CORES), cb.dim3(THREADS)](
            values, indices, size, dense_shape, output
        )  # ② 幂等直写
    elif weights is not None:
        _bincount_weighted[cb.dim3(CORES), cb.dim3(THREADS)](
            values, indices, weights, size, dense_shape, output
        )  # ② 加权原子
    else:
        _bincount_count[cb.dim3(CORES), cb.dim3(THREADS)](
            values, indices, size, dense_shape, output
        )  # ② 计数原子
