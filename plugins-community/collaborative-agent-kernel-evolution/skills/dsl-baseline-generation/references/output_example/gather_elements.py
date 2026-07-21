# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import tile.language as tl


@ascend_kernel
def gather_element_kernel(
    x_ptr,                    # [x_pre_dim, x_gather_dim]
    idx_ptr,                  # [idx_pre_dim, idx_gather_dim]
    y_ptr,                    # [idx_pre_dim, idx_gather_dim]
    x_pre_dim,
    x_gather_dim,
    idx_pre_dim,
    idx_gather_dim,
    rows_per_core,
    tile_rows,
    tile_loops
):
    pid = tl.program_id(0)
    start_row = pid * rows_per_core

    # ------------------------------------------------------------
    # 计算数据类型大小
    # ------------------------------------------------------------
    dtype_x_size = 4  # float32
    dtype_idx_size = 4  # int32

    actual_tile_rows = tile_rows

    actual_loops = tile_loops

    # 分配UB缓冲区
    x_ub = tl.alloc_ub(actual_tile_rows * x_gather_dim, dtype=tl.float32)
    idx_ub = tl.alloc_ub(actual_tile_rows * idx_gather_dim, dtype=tl.int32)
    y_ub = tl.alloc_ub(actual_tile_rows * idx_gather_dim, dtype=tl.float32)

    # ------------------------------------------------------------
    # 主处理循环
    # ------------------------------------------------------------
    for loop_idx in range(actual_loops):
        current_row = start_row + loop_idx * actual_tile_rows
        current_rows = min(actual_tile_rows, rows_per_core - loop_idx * actual_tile_rows)
        x_offsets = current_row * x_gather_dim + tl.arange(0, x_gather_dim * actual_tile_rows)
        idx_offsets = current_row * idx_gather_dim + tl.arange(0, idx_gather_dim * actual_tile_rows)
        y_offsets = current_row * idx_gather_dim + tl.arange(0, idx_gather_dim * actual_tile_rows)

        if current_rows <= 0:
            break

        # 计算当前批次的实际元素数
        x_elements = current_rows * x_gather_dim
        idx_elements = current_rows * idx_gather_dim
        y_elements = current_rows * idx_gather_dim

        # -------------------------------
        # COPYIN: 加载x和idx数据
        # -------------------------------
        with tl.copyin():
            # 加载x数据
            tl.load(x_ptr + x_offsets, x_ub)

            # 加载idx数据
            tl.load(idx_ptr + idx_offsets, idx_ub)

        # -------------------------------
        # COMPUTE: 执行gather
        # -------------------------------
        with tl.compute():
            # 处理每一行
            for i in range(current_rows):
                # to gather data, you need to use vmul first to calculate the index
                # offset, then use the offset in the gather api
                index_row_offset = i * idx_gather_dim
                x_row_offset = i * x_gather_dim
                tl.vmul_scalar(
                    idx_ub[index_row_offset],
                    idx_ub[index_row_offset],
                    dtype_idx_size,
                    idx_gather_dim
                )

                tl.gather(y_ub[index_row_offset], x_ub[x_row_offset], idx_ub[index_row_offset], 0, idx_gather_dim)

        # -------------------------------
        # COPYOUT: 写回结果
        # -------------------------------
        with tl.copyout():
            tl.store(y_ptr + y_offsets, y_ub)


def gather_element_host(
    x: torch.Tensor,          # [x_pre_dim, x_gather_dim]
    idx: torch.Tensor,        # [idx_pre_dim, idx_gather_dim]
    y: torch.Tensor           # [idx_pre_dim, idx_gather_dim]
):
    # 将shape展平成2维，将除了最后一维的前几维，合成一维
    x_pre_dim = 1
    idx_pre_dim = 1
    for dim in range(len(x.shape) - 1):
        x_pre_dim = x_pre_dim * x.shape[dim]
        idx_pre_dim = idx_pre_dim * idx.shape[dim]

    x_gather_dim = x.shape[-1]
    idx_gather_dim = idx.shape[-1]

    # ------------------------------------------------------------
    # 核间切分策略
    # ------------------------------------------------------------
    n_cores = min(16, idx_pre_dim)  # 最大16个核心，不超过行数
    rows_per_core = (idx_pre_dim + n_cores - 1) // n_cores

    # ------------------------------------------------------------
    # 核内切分策略
    # ------------------------------------------------------------
    # 假设UB容量计算
    ub_capacity = 180 * 1024  # ub大小180kb

    # 计算每行需要的内存
    dtype_x_size = 4  # float32
    dtype_idx_size = 4  # int32

    mem_per_row = x_gather_dim * dtype_x_size + idx_gather_dim * dtype_idx_size + idx_gather_dim * dtype_x_size

    # 计算每轮可以处理的最大行数
    tile_rows = max(1, ub_capacity // mem_per_row)

    # 计算每个核心需要循环的次数
    tile_loops = (rows_per_core + tile_rows - 1) // tile_rows

    # ------------------------------------------------------------
    # 启动内核
    # ------------------------------------------------------------
    gather_element_kernel[n_cores](
        x,
        idx,
        y,
        x_pre_dim,
        x_gather_dim,
        idx_pre_dim,
        idx_gather_dim,
        rows_per_core,
        tile_rows,
        tile_loops
    )
