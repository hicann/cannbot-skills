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
    ROUTE_TWO_STAGE_GROUPED,
    sort_entry,
    validate_route,
)
from sort_transpose import (
    TRANSPOSE_TILE,
    select_transpose_tile,
    transpose_entry,
    transpose_buffer_elements,
)

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
TWO_STAGE_RANK_INVERSE_MAX_N = 32
TWO_STAGE_ROW_KEY_BINS = 256
# 软件驻留目标，不是硬件容量；核间任务均衡可要求更大的最小工作集。
RADIX_WORKSPACE_TARGET_BYTES = 64 * 1024**2


def min_segments(tiers, length):
    return next((need for maximum, need in tiers if length <= maximum), None)


def insertion_allowed(dtype, length, rows, cores):
    minimum = min_segments(INSERTION_TIERS[dtype], length)
    return minimum is not None and (rows + cores - 1) // cores >= minimum


def use_rank_inverse(length):
    return length <= TWO_STAGE_RANK_INVERSE_MAX_N


def two_stage_batch(dtype, length, rows, cores, usable):
    """返回可行批次行数，0 表示未命中；usable 为扣除 SIMT 保留后的 UB 字节数。"""
    per_core = (rows + cores - 1) // cores
    minimum = min_segments(TWO_STAGE_TIERS[dtype], length)
    if length <= 0 or minimum is None or per_core < minimum:
        return 0
    key_bytes = 8 if dtype in (torch.int64, torch.uint64) else 4
    # 每元素两份键，加 14 字节的索引/排列/字节缓冲；1024 覆盖固定区和对齐余量。
    # 65280 是 uint16 计数上限向下对齐到 256 后的最大元素数。
    max_elements = min(
        65280,
        ((usable - 1024) // (2 * key_bytes + 14) // LOCAL_RADIX_ALIGNMENT)
        * LOCAL_RADIX_ALIGNMENT,
    )
    max_batch = min(per_core, max_elements // length)
    if not use_rank_inverse(length):
        # 第二阶段稳定地按行号分组，前一阶段的数值顺序自然保留。
        # 将批次限制在 256 行以内，只需一个 8-bit digit。
        max_batch = min(max_batch, TWO_STAGE_ROW_KEY_BINS)
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


def build_program(dtype, route, width, groups, descending, narrow_counts=False):
    """编译排序；直接调用默认 int64 工作区，受限轴长可显式选择 int32。"""
    validate_route(route, groups)
    elements = cb.Dim("elements", min=1, max=2**63 - 1)
    scratch = cb.Dim("scratch", min=1, max=2**63 - 1)
    bins = cb.Dim("bins", min=1, max=2**63 - 1)
    states = cb.Dim("states", min=1, max=2**63 - 1)
    value_spec = cb.TensorSpec((elements,), DSL_DTYPES[dtype])
    output_indices = cb.TensorSpec((elements,), cb.dtypes.int64)
    key_spec = cb.TensorSpec(
        (scratch,),
        DSL_DTYPES[dtype]
        if route == ROUTE_RADIX_MORE_CORE
        else key_dtype(DSL_DTYPES[dtype]),
    )
    counter_dtype = cb.dtypes.int32 if narrow_counts else cb.dtypes.int64
    index_spec = cb.TensorSpec((scratch,), counter_dtype)
    bin_spec = cb.TensorSpec((bins,), counter_dtype)
    state_spec = cb.TensorSpec((states,), counter_dtype)
    return cb.compile(
        sort_entry,
        # 输入、输出值、输出索引。
        value_spec,
        value_spec,
        output_indices,
        # 工作区：ping、pong、计数和 lookback 状态，与 sort_entry 顺序一致。
        key_spec,
        index_spec,
        key_spec,
        index_spec,
        bin_spec,
        state_spec,
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


class WordPairTranspose:
    """64 位 Tensor 仅以 uint32 别名绑定到 NDDMA，不复制、不转换数值。"""

    def __init__(self, program, dtype):
        self.program = program
        self.dtype = dtype

    def __call__(self, x, output, *args, **kwargs):
        if x.dtype != self.dtype or output.dtype != self.dtype:
            raise TypeError("transpose input and output must match the compiled dtype")
        return self.program(
            x.view(torch.uint32), output.view(torch.uint32), *args, **kwargs
        )

    def close(self):
        self.program.close()

    def __enter__(self):
        self.program.__enter__()
        return self

    def __exit__(self, *args):
        return self.program.__exit__(*args)


def build_transpose_program(dtype, tile_size=TRANSPOSE_TILE):
    if tile_size not in (32, 64):
        raise ValueError("unsupported transpose tile")
    words = 2 if dtype in (torch.int64, torch.uint64) else 1
    # NDDMA 检查 GM 根参数类型，不能仅在 kernel 内 reinterpret。
    raw_dtype = torch.uint32 if words == 2 else dtype
    elements = cb.Dim("elements", min=1, max=2**63 - 1)
    tensor_spec = cb.TensorSpec((elements,), DSL_DTYPES[raw_dtype])
    program = cb.compile(
        transpose_entry,
        tensor_spec,
        tensor_spec,
        cb.dtypes.int64,
        cb.dtypes.int64,
        cb.dtypes.int64,
        cb.dtypes.int64,
        tile_size,
        words,
    )
    return WordPairTranspose(program, dtype) if words == 2 else program


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
                if use_rank_inverse(length)
                else ROUTE_TWO_STAGE_GROUPED
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
            # 1/2 字节用局部排列表示；32/64 位仍采用键和索引双缓冲。
            local_bytes = (
                6
                if x.element_size() == 1
                else 12
                if x.element_size() == 2
                else 2 * key_bytes + 14
            )
            required = width * local_bytes + 1024
            if width <= LOCAL_RADIX_COUNT_MAX and required <= usable:
                route = ROUTE_RADIX_ONE_CORE
                self.route = "radix_one_core"
                groups = 1
                tasks = rows
            else:
                route = ROUTE_RADIX_MORE_CORE
                self.route = "radix_more_core"
                width = DEFAULT_RADIX_TILE
                # UB 包含局部排列、digit、原值和索引 tile。
                # 固定预算覆盖桶偏移、scatter bases 和对齐；两波任务是并发策略门槛。
                # 候选升序排列，覆盖赋值后留下满足容量与并发门槛的最大块。
                radix_tile_bytes = 6 + x.element_size() + (4 if length < 2**30 else 8)
                for candidate in GLOBAL_RADIX_TILES:
                    candidate_tasks = rows * ((length + candidate - 1) // candidate)
                    if (
                        candidate * radix_tile_bytes + 4096 <= usable
                        and candidate_tasks >= 2 * info.vector_core_num
                    ):
                        width = candidate
                # 两波任务是性能偏好；即使使用默认块，容量也必须满足。
                if width * radix_tile_bytes + 4096 > usable:
                    raise RuntimeError("insufficient UB for radix Sort tile")
                tasks = rows * ((length + width - 1) // width)
        # 单次 launch 内按批内行号复用 scratch，输入/输出仍使用全局行号。
        tiles = (length + width - 1) // width
        blocks = min(int(info.vector_core_num), max(1, tasks))
        self.active_rows = rows
        if route == ROUTE_RADIX_MORE_CORE:
            # P*K 按 blocks 整除；按原值、索引和桶状态计算每行工作集。
            counter_bytes = 4 if length < 2**30 else 8
            per_row_workspace = length * (
                2 * x.element_size() + 2 * counter_bytes
            ) + counter_bytes * RADIX_BINS * (tiles + x.element_size() * (tiles + 1))
            row_quantum = blocks // math.gcd(blocks, tiles)
            target_rows = max(
                row_quantum,
                ((RADIX_WORKSPACE_TARGET_BYTES // per_row_workspace) // row_quantum)
                * row_quantum,
            )
            if rows >= 2 * target_rows:
                self.active_rows = target_rows
                groups = target_rows
        count = self.active_rows * length if route == ROUTE_RADIX_MORE_CORE else 1
        bins = (
            self.active_rows * tiles * RADIX_BINS
            if route == ROUTE_RADIX_MORE_CORE
            else 1
        )
        states_count = (
            self.active_rows * (tiles + 1) * RADIX_BINS * x.element_size()
            if route == ROUTE_RADIX_MORE_CORE
            else 1
        )
        torch_key_dtype = (
            x.dtype
            if route == ROUTE_RADIX_MORE_CORE
            else torch.uint64
            if key_bytes == 8
            else torch.uint32
        )
        # 计数、前缀和原轴索引均限制在单行 N 内，保留 int32 累加余量。
        # 行号、GM 地址和最终输出索引仍为 int64；大轴保留宽工作区。
        narrow_counts = route == ROUTE_RADIX_MORE_CORE and length < 2**30
        workspace_index_dtype = torch.int32 if narrow_counts else torch.int64
        # 多核 ping/pong 与输入同 dtype，保存原值位模式；其他路线使用最小键占位。
        # 参数顺序：ping值/索引、pong值/索引、桶计数和 lookback 状态。
        # 仅多核路线分配完整工作区；其他路线只保留调用签名所需的最小占位。
        self.workspace = (
            torch.empty(count, dtype=torch_key_dtype, device=x.device),
            torch.empty(count, dtype=workspace_index_dtype, device=x.device),
            torch.empty(count, dtype=torch_key_dtype, device=x.device),
            torch.empty(count, dtype=workspace_index_dtype, device=x.device),
            torch.empty(bins, dtype=workspace_index_dtype, device=x.device),
            torch.empty(states_count, dtype=workspace_index_dtype, device=x.device),
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
            tile_size = select_transpose_tile(length, inner)
            if transpose_buffer_elements(tile_size) * 8 > usable:
                raise RuntimeError("insufficient UB for Sort transpose")
            transpose_tasks = (
                rows
                // inner
                * ((length + tile_size - 1) // tile_size)
                * ((inner + tile_size - 1) // tile_size)
            )
            transpose_blocks = min(int(info.vector_core_num), transpose_tasks)
            self.layout_shape = rows // inner, length, inner, transpose_blocks
            self.restore_shape = rows // inner, inner, length, transpose_blocks
            self.transpose_program = build_transpose_program(x.dtype, tile_size)
            self.index_transpose_program = build_transpose_program(
                torch.int64, tile_size
            )
        # 公开入口经布局规范化后始终传 inner=1；低层 Kernel 的跨步寻址仅供直接调用。
        self.runtime_shape = rows, length, 1, blocks
        self.program = build_program(
            x.dtype, route, width, groups, descending, narrow_counts
        )

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
    with build_program(torch.int32, ROUTE_TWO_STAGE_GROUPED, 1792, 25, False):
        pass
    with build_program(torch.float32, ROUTE_RADIX_MORE_CORE, 8192, 224, True, True):
        pass


@export("sort_transpose")
def export_sort_transpose():
    for dtype in SUPPORTED_DTYPES:
        with build_transpose_program(dtype):
            pass
