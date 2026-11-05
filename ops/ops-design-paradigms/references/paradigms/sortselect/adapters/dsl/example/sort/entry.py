# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""公开 Sort 包装：校验输入、选择路线、分配工作区并恢复输出布局。"""

import math

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401
from cannbotdsl.aot import export
from sort_common import (
    GLOBAL_RADIX_TILES,
    LOCAL_RADIX_COUNT_MAX,
    RADIX_BINS,
    DEFAULT_RADIX_TILE,
    INSERTION_MAX_BATCH_ROWS,
    LOCAL_RADIX_ALIGNMENT,
    SIMT_SCRATCH_BYTES,
    key_dtype,
)
from sort_host import (
    ROUTE_INSERTION,
    ROUTE_RADIX_ONE_CORE,
    ROUTE_RADIX_MORE_CORE,
    ROUTE_TWO_STAGE_INVERSE,
    ROUTE_TWO_STAGE_PACKED,
    sort_entry,
    validate_route,
)
from sort_transpose import TRANSPOSE_TILE, transpose_entry

SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)
DSL_DTYPES = {
    dtype: getattr(cb.dtypes, str(dtype).removeprefix("torch."))
    for dtype in SUPPORTED_DTYPES
}
INSERTION_TIERS = {
    torch.int64: ((8, 1), (16, 4)),
    torch.uint64: ((8, 1), (16, 4)),
    torch.int32: ((8, 2), (11, 4)),
    torch.uint32: ((8, 2), (11, 4)),
    torch.int16: ((4, 2), (8, 4)),
    torch.uint16: ((4, 2), (8, 4)),
    torch.int8: ((4, 2), (8, 7)),
    torch.uint8: ((4, 2), (8, 7)),
    # 沿用浮点基线门槛；gate=1 并非已测得的性能拐点。低并发时 Radix 可能更快，需按目标调优。
    torch.float32: ((4, 1), (8, 1)),
    torch.float16: ((4, 1), (8, 1)),
    torch.bfloat16: ((4, 1), (8, 1)),
}
INSERTION_LIMIT = {dtype: tiers[-1][0] for dtype, tiers in INSERTION_TIERS.items()}


# 整数档位与 AscendC 的 kSmallAxisRules 对齐；DSL 尚未适配 MergeSort，浮点类型沿用本实现的两阶段档位。
# 先选择首个匹配轴长的档位，再检查并发门槛；缓冲容量约束由各语言实现分别检查。
TWO_STAGE_TIERS = {
    torch.int64: ((15, 8), (128, 4), (512, 8)),
    torch.uint64: ((15, 8), (128, 4), (512, 8)),
    torch.int32: ((11, 8), (64, 4), (384, 8)),
    torch.uint32: ((11, 8), (64, 4), (384, 8)),
    torch.int16: ((7, 8), (64, 4), (192, 12)),
    torch.uint16: ((7, 8), (64, 4), (192, 12)),
    torch.int8: ((3, 8), (64, 7), (128, 16)),
    torch.uint8: ((3, 8), (64, 7), (128, 16)),
    torch.float32: ((24, 64), (64, 64), (256, 64)),
    torch.float16: ((54, 64), (64, 64), (256, 64)),
    torch.bfloat16: ((54, 64), (64, 64), (256, 64)),
}
TWO_STAGE_RANK_INVERSE_MAX_N = 64
FLOAT_TWO_STAGE_RANK_INVERSE_MAX_N = 32
TWO_STAGE_U16_KEY_SPACE = 65536


def min_segments(tiers, length):
    return next((need for maximum, need in tiers if length <= maximum), None)


def insertion_allowed(dtype, length, rows, cores):
    minimum = min_segments(INSERTION_TIERS[dtype], length)
    return minimum is not None and (
        length == 1 or (rows + cores - 1) // cores >= minimum
    )


def use_rank_inverse(dtype, length):
    limit = (
        FLOAT_TWO_STAGE_RANK_INVERSE_MAX_N
        if dtype in (torch.float16, torch.bfloat16, torch.float32)
        else TWO_STAGE_RANK_INVERSE_MAX_N
    )
    return length <= limit


def two_stage_batch(dtype, length, rows, cores, usable):
    """返回可行批次行数，0 表示未命中；usable 为扣除 SIMT 保留后的 UB 字节数。"""
    per_core = (rows + cores - 1) // cores
    minimum = min_segments(TWO_STAGE_TIERS[dtype], length)
    if length <= 1 or minimum is None or per_core < minimum:
        return 0
    key_bytes = 8 if dtype in (torch.int64, torch.uint64) else 4
    # 每元素两份键，加 19 字节的索引/排列/字节缓冲；1024 覆盖固定区和对齐余量。
    # 65280 是 uint16 计数上限向下对齐到 256 后的最大元素数。
    max_elements = min(
        65280,
        ((usable - 1024) // (2 * key_bytes + 19) // LOCAL_RADIX_ALIGNMENT)
        * LOCAL_RADIX_ALIGNMENT,
    )
    max_batch = min(per_core, max_elements // length)
    if not use_rank_inverse(dtype, length):
        # 紧凑键最大为 B²*N-1：B²*N≤键空间，所以 B≤floor(sqrt(键空间/N))。
        # 使用整数平方根，避免浮点舍入在键空间边界放行超出 uint16 的批次。
        max_batch = min(max_batch, math.isqrt(TWO_STAGE_U16_KEY_SPACE // length))
    best = 0
    best_idle = rows
    first_rounds = None
    for candidate in range(max_batch, 1, -1):
        batches = (rows + candidate - 1) // candidate
        active = min(cores, batches)
        rounds = (batches + active - 1) // active
        if first_rounds is None:
            first_rounds = rounds
        # 只在首个可行批次的执行轮数内减少空闲槽；同分保留较大批次。
        if rounds != first_rounds:
            break
        idle = rounds * active - batches
        if idle < best_idle:
            best, best_idle = candidate, idle
    return best


def build_program(dtype, route, width, groups, descending):
    validate_route(route, groups)
    elements = cb.Dim("elements", min=1, max=2**63 - 1)
    scratch = cb.Dim("scratch", min=1, max=2**63 - 1)
    bins = cb.Dim("bins", min=1, max=2**63 - 1)
    value_spec = cb.TensorSpec((elements,), DSL_DTYPES[dtype])
    output_indices = cb.TensorSpec((elements,), cb.dtypes.int64)
    key_spec = cb.TensorSpec((scratch,), key_dtype(DSL_DTYPES[dtype]))
    index_spec = cb.TensorSpec((scratch,), cb.dtypes.int64)
    bin_spec = cb.TensorSpec((bins,), cb.dtypes.int64)
    rank_spec = cb.TensorSpec((scratch,), cb.dtypes.uint32)
    return cb.compile(
        sort_entry,
        # 输入、输出值、输出索引。
        value_spec,
        value_spec,
        output_indices,
        # 工作区：ping、pong、计数、偏移和排名，与 sort_entry 顺序一致。
        key_spec,
        index_spec,
        key_spec,
        index_spec,
        bin_spec,
        bin_spec,
        rank_spec,
        # 运行参数：rows、length、inner、blocks。
        cb.dtypes.int64,
        cb.dtypes.int64,
        cb.dtypes.int64,
        cb.dtypes.int64,
        # 编译参数：路线、局部宽度、批次行数、方向。
        route,
        width,
        groups,
        descending,
    )


def build_transpose_program(dtype):
    elements = cb.Dim("elements", min=1, max=2**63 - 1)
    tensor_spec = cb.TensorSpec((elements,), DSL_DTYPES[dtype])
    return cb.compile(
        transpose_entry,
        tensor_spec,
        tensor_spec,
        cb.dtypes.int64,
        cb.dtypes.int64,
        cb.dtypes.int64,
        cb.dtypes.int64,
    )


class PreparedSort:
    """稳定排序；多次调用复用输出缓冲，后一次运行会覆盖前一次结果。"""

    def __init__(self, x, dim=-1, descending=False, stable=True):
        if x.device.type != "npu" or not x.is_contiguous():
            raise ValueError("requires contiguous NPU input")
        if x.dtype not in SUPPORTED_DTYPES:
            raise ValueError("unsupported Sort dtype")
        if not isinstance(dim, int) or not -x.ndim <= dim < x.ndim:
            raise ValueError("dim must identify an input axis")
        if not isinstance(descending, bool) or not isinstance(stable, bool):
            raise TypeError("descending and stable must be bool")
        # stable=False 允许任意等值顺序；本例统一使用稳定实现，两种标志值共用路线。
        dim %= x.ndim
        self.x = x.view(-1)
        self.values = torch.empty_like(x)
        self.indices = torch.empty(x.shape, dtype=torch.int64, device=x.device)
        self.empty = x.numel() == 0
        self.active_rows = 0
        self.workspace_bytes = 0
        self.program = None
        self.transpose_program = None
        self.index_transpose_program = None
        if self.empty:
            self.route = "empty"
            self.kernel_count = 0
            return
        length = x.shape[dim]
        tail_start = dim + 1
        inner = math.prod(x.shape[tail_start:])
        rows = x.numel() // length
        stream = torch.npu.current_stream(x.device)
        info = cb.get_platform_info(stream=stream)
        usable = cb.get_mem_size("ub") - SIMT_SCRATCH_BYTES
        if not info.available or info.vector_core_num <= 0 or usable <= 0:
            raise RuntimeError("vector core capacity unavailable")
        key_bytes = 8 if x.dtype in (torch.int64, torch.uint64) else 4
        width = 0
        groups = 0
        self.needs_transpose = length > 1 and inner > 1
        batch_rows = two_stage_batch(
            x.dtype, length, rows, int(info.vector_core_num), usable
        )
        if batch_rows:
            route = (
                ROUTE_TWO_STAGE_INVERSE
                if use_rank_inverse(x.dtype, length)
                else ROUTE_TWO_STAGE_PACKED
            )
            self.route = "two_stage"
            groups = batch_rows
            width = (
                (length * groups + LOCAL_RADIX_ALIGNMENT - 1) // LOCAL_RADIX_ALIGNMENT
            ) * LOCAL_RADIX_ALIGNMENT
            tasks = (rows + groups - 1) // groups
        elif insertion_allowed(x.dtype, length, rows, int(info.vector_core_num)):
            route = ROUTE_INSERTION
            self.route = "insertion"
            width = INSERTION_LIMIT[x.dtype]
            groups = min(
                INSERTION_MAX_BATCH_ROWS,
                usable // (width * (x.element_size() + key_bytes + 4)),
            )
            if groups == 0:
                raise RuntimeError("insufficient UB for insertion Sort")
            tasks = (rows + groups - 1) // groups
        else:
            width = (
                (length + LOCAL_RADIX_ALIGNMENT - 1) // LOCAL_RADIX_ALIGNMENT
            ) * LOCAL_RADIX_ALIGNMENT
            # 单核比两阶段少一份 stage1_ids，因此每元素附加空间从 19 减为 15 字节。
            required = width * (2 * key_bytes + 15) + 1024
            if width <= LOCAL_RADIX_COUNT_MAX and required <= usable:
                route = ROUTE_RADIX_ONE_CORE
                self.route = "radix_one_core"
                groups = 1
                tasks = rows
            else:
                route = ROUTE_RADIX_MORE_CORE
                self.route = "radix_more_core"
                width = DEFAULT_RADIX_TILE
                # 多核的每元素 UB 为 digits(1)+permutation(4)+scratch(1)+byte_order(1)。
                # 1024 覆盖桶偏移、warp 总量和固定 scratch；两波任务是并发策略门槛。
                # 候选升序排列，覆盖赋值后留下满足容量与并发门槛的最大块。
                for candidate in GLOBAL_RADIX_TILES:
                    candidate_tasks = rows * ((length + candidate - 1) // candidate)
                    if (
                        candidate * 7 + 1024 <= usable
                        and candidate_tasks >= 2 * info.vector_core_num
                    ):
                        width = candidate
                # 两波任务是性能偏好；即使使用默认块，容量也必须满足。
                if width * 7 + 1024 > usable:
                    raise RuntimeError("insufficient UB for radix Sort tile")
                tasks = rows * ((length + width - 1) // width)
        # 全局屏障协议要求所有行在同一次 launch 中完成。改变工作区复用方式前须对比性能。
        self.active_rows = rows  # 本实现 P_act=M，不做跨 launch 的有界行工作区复用。
        tiles = (length + width - 1) // width
        blocks = min(int(info.vector_core_num), max(1, tasks))
        count = self.active_rows * length if route == ROUTE_RADIX_MORE_CORE else 1
        bins = (
            self.active_rows * tiles * RADIX_BINS
            if route == ROUTE_RADIX_MORE_CORE
            else 1
        )
        torch_key_dtype = torch.uint64 if key_bytes == 8 else torch.uint32
        # 参数顺序与 sort_entry 一致：ping键/索引、pong键/索引、桶计数、全局偏移、打包排名。
        # 仅多核路线分配完整工作区；其他路线只保留调用签名所需的最小占位。
        self.workspace = (
            torch.empty(count, dtype=torch_key_dtype, device=x.device),
            torch.empty(count, dtype=torch.int64, device=x.device),
            torch.empty(count, dtype=torch_key_dtype, device=x.device),
            torch.empty(count, dtype=torch.int64, device=x.device),
            torch.empty(bins, dtype=torch.int64, device=x.device),
            torch.empty(bins, dtype=torch.int64, device=x.device),
            torch.empty(count, dtype=torch.uint32, device=x.device),
        )
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in self.workspace)
        self.rows = rows
        self.length = length
        self.kernel_count = 1
        if self.needs_transpose:
            self.kernel_count += 3
        self.sort_input = self.x
        self.output_values = self.values.view(-1)
        self.output_indices = self.indices.view(-1)
        if self.needs_transpose:
            self.sort_input = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
            self.output_values = torch.empty_like(self.sort_input)
            self.output_indices = torch.empty(
                x.numel(), dtype=torch.int64, device=x.device
            )
            transpose_tasks = (
                rows
                // inner
                * ((length + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE)
                * ((inner + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE)
            )
            transpose_blocks = min(int(info.vector_core_num), transpose_tasks)
            self.layout_shape = rows // inner, length, inner, transpose_blocks
            self.restore_shape = rows // inner, inner, length, transpose_blocks
            self.transpose_program = build_transpose_program(x.dtype)
            self.index_transpose_program = build_transpose_program(torch.int64)
        # 公开入口经布局规范化后始终传 inner=1；低层 Kernel 的跨步寻址仅供直接调用。
        self.runtime_shape = rows, length, 1, blocks
        self.program = build_program(x.dtype, route, width, groups, descending)

    def run(self):
        if not self.empty:
            if self.needs_transpose:
                self.transpose_program(self.x, self.sort_input, *self.layout_shape)
            self.program(
                self.sort_input,
                self.output_values,
                self.output_indices,
                *self.workspace,
                *self.runtime_shape,
            )
            if self.needs_transpose:
                self.transpose_program(
                    self.output_values, self.values.view(-1), *self.restore_shape
                )
                self.index_transpose_program(
                    self.output_indices, self.indices.view(-1), *self.restore_shape
                )
        return self.values, self.indices

    def close(self):
        """释放编译程序；调用方必须先等待使用这些程序的设备任务完成。"""
        if self.program is not None:
            self.program.close()
            self.program = None
        if self.transpose_program is not None:
            self.transpose_program.close()
            self.transpose_program = None
        if self.index_transpose_program is not None:
            self.index_transpose_program.close()
            self.index_transpose_program = None


def sort(x, dim=-1, descending=False, stable=True):
    """稳定排序并返回值和原轴索引；等待设备完成后释放临时程序。"""
    prepared = PreparedSort(x, dim, descending, stable)
    try:
        result = prepared.run()
        torch.npu.synchronize(x.device)
        return result
    finally:
        prepared.close()


@export("sort_example")
def export_sort_example():
    for descending in (False, True):
        with build_program(torch.float32, ROUTE_INSERTION, 8, 1024, descending):
            pass
    with build_program(torch.float32, ROUTE_RADIX_ONE_CORE, 8192, 1, False):
        pass
    with build_program(torch.float32, ROUTE_TWO_STAGE_INVERSE, 1280, 74, True):
        pass
    with build_program(torch.int32, ROUTE_TWO_STAGE_PACKED, 1792, 25, False):
        pass
    with build_program(torch.float32, ROUTE_RADIX_MORE_CORE, 8192, 0, True):
        pass


@export("sort_transpose")
def export_sort_transpose():
    for dtype in SUPPORTED_DTYPES:
        with build_transpose_program(dtype):
            pass
