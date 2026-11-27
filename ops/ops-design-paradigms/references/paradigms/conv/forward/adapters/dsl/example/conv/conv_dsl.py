# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Conv1D / Conv2D / Conv3D 推理算子的 CANNBotDSL 参考实现。

单文件、单计算主体：
- 1D 升维为 (D=1, H=1)，2D 升维为 (D=1)，3D 保留 D 轴；
- Host 侧做规格校验、指令字段预检、片上容量核算、代价模型选块
  （MT/NT/CT × B 驻留 × L1/L0 双缓冲深度 × M 超块子块数 × 每任务输出行数）；
- Kernel 侧按 batch → 输出行组 → n 分组 → kd 分块 → ci 分块 迭代：
  A 侧整行带一次搬入 L1（单条 multi-row DMA，左右 pad 交给 load3d），
  B 侧在容量允许时按 n 分组驻留复用；一个任务可跨多个输出行（展平 M），
  并用 M 超块把同一 feature/weight 分块复用到多个 M 子块上。

与 ../conv2d/ascendc 参考实现同构，但走**底层接口**，与 AscendC 指令一一对应：
    c.load3d(Load3DFeatureMap/Window/Destination)   L1 → L0A 滑窗展开（对应 Load3D/FMatrix）
    from_tensor(..., physical_layout=...)         手写 NZ 物理布局（对应 DataCopy 分形排布）
    cube_fill_l1_zero + cube_sync_pipe             行带/权重零填充与 MTE2 屏障
    c.mem_copy / c.matmul / tensor.convert         L1→L0B、MMAD、L0C→GM
不使用 cannbotdsl.ops.conv 的 Conv2dSpec/conv2d_load_* 专用接口：该组接口在 N 方向
尾块（BN > n_actual）上存在数值缺陷（见 ../../capability-mapping.md §5 G1）。

规格范围：FP16、NCW/NCHW/NCDHW + OIW/OIHW/OIDHW、groups=1、pad < kernel、
stride 1~63、dilation 1~255、非对称 pad、可选一维 bias、合法空输出（不编译、不 launch）。
"""

from dataclasses import dataclass, replace

import torch

import cannbotdsl as c
from cannbotdsl import MemLoc, Tensor, dtypes, make_layout
from cannbotdsl.ops.memcpy import cube_fill_l1_zero
from cannbotdsl.ops.sync import PIPE
from cannbotdsl.ring_buffer import RingBuffer

MAX_INDEX = (1 << 31) - 1
LOAD3D_MAX_STRIDE = 63
LOAD3D_MAX_DILATION = 255
LOAD3D_MAX_PAD = 255
LOAD3D_MAX_FEATURE = 32767
MAX_FILTER = 255
UINT16_MAX = 65535
MAX_IMMEDIATE = 4095

ITEM_BYTES = 2
C0 = 32 // ITEM_BYTES
MT_STEPS = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224)
NT_STEPS = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224)
CT_STEPS = (16, 32, 48, 64, 96, 112, 128)
# 每任务/每 K 循环的固定开销当量（按 22 例 msprof 抽样校准：大 MT/NT 减少任务数带来的收益
# 明显高于离散负载不均的惩罚，原 1KB/0.5KB 会误选小 tile）。
TASK_OVERHEAD_BYTES = 4096
KLOOP_OVERHEAD_BYTES = 1024
SUB_OVERHEAD_BYTES = 4096
# 单缓冲流水 + L0 容量：MT 上限 224（k_stride_fractals=MT/16≤14），NT 由 L0B 容量再收紧。
MAX_TILE = 224
MAX_SUBS = 8


@dataclass(frozen=True)
class Problem:
    """Host 侧已归一化的卷积规格（1D/2D 已升维到 D/H 轴，D=H=1）。"""

    rank: int
    shape: tuple
    weight_shape: tuple
    batch: int
    ci: int
    depth: int
    height: int
    width: int
    co: int
    kd: int
    kh: int
    kw: int
    sd: int
    sh: int
    sw: int
    pd0: int
    pd1: int
    pt: int
    pb: int
    pl: int
    pr: int
    dd: int
    dh: int
    dw: int
    has_bias: bool
    od: int
    oh: int
    ow: int

    @property
    def rs(self):
        return self.kh * self.kw

    @property
    def empty(self):
        return not self.od or not self.oh or not self.ow

    @property
    def output_shape(self):
        if self.rank == 1:
            return (self.batch, self.co, self.ow)
        if self.rank == 2:
            return (self.batch, self.co, self.oh, self.ow)
        return (self.batch, self.co, self.od, self.oh, self.ow)

    @property
    def pad_tuple(self):
        return (self.pd0, self.pd1, self.pt, self.pb, self.pl, self.pr)


@dataclass(frozen=True)
class Config:
    """一次 tiling 决策的完整结果（Host 计算结果，Kernel 编译期常量）。"""

    mt: int
    nt: int
    ct: int
    resident: bool
    patch_height: int
    kt: int
    width_tiles: int
    channel_tiles: int
    ci_tiles: int
    spatial_tasks: int
    groups: int
    tasks: int
    reduction_steps: int
    cost: float
    l1_bytes: int
    pipe_depth: int = 1
    nsubs: int = 1
    mrows: int = 1

    @property
    def tile(self):
        return (self.mt, self.nt, self.ct)


def _int(value, name, minimum=1, maximum=MAX_INDEX):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not bool")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _axes(value, name, rank):
    values = (value,) * rank if type(value) is int else value
    if not isinstance(values, tuple) or len(values) != rank:
        raise ValueError(f"{name} must be an integer or a {rank}-tuple")
    return tuple(_int(v, name) for v in values)


def _pad_pairs(rank, value, kernel):
    """padding: 标量 / 每轴对称 / 逐侧成对。返回 [(before, after)] × 3（D,H,W）。"""
    if type(value) is int:
        value = (value,) * rank
    if not isinstance(value, tuple) or len(value) not in (rank, 2 * rank):
        raise ValueError(
            f"padding must be an integer, a {rank}-tuple or a {2 * rank}-tuple"
        )
    values = tuple(_int(v, "padding", 0, LOAD3D_MAX_PAD) for v in value)
    if len(values) == rank:
        pairs = [(v, v) for v in values]
    else:
        pairs = [(values[2 * i], values[2 * i + 1]) for i in range(rank)]
    if rank == 1:
        pairs = [(0, 0), (0, 0), pairs[0]]
    elif rank == 2:
        pairs = [(0, 0), pairs[0], pairs[1]]
    for axis, (before, after) in enumerate(pairs):
        if before >= kernel[axis] or after >= kernel[axis]:
            raise ValueError(
                f"axis {axis}: each padding side must be less than kernel size"
            )
    return pairs


def _normalize(x, weight, bias, stride, padding, dilation):
    rank = x.dim() - 2
    if rank not in (1, 2, 3) or weight.dim() != x.dim():
        raise ValueError("Conv1D/2D/3D expects rank-3/4/5 input and weight")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("weight input channels must equal input channels (groups=1)")
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError("bias must have shape (output_channels,)")
    for name, tensor in (("input", x), ("weight", weight), ("bias", bias)):
        if tensor is None:
            continue
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise ValueError(f"{name} must be contiguous with zero storage offset")
        if tensor.device != x.device or tensor.dtype != x.dtype:
            raise ValueError(f"{name} must match input device and dtype")
    if x.device.type != "npu":
        raise ValueError("only NPU inputs are supported; no fallback")
    if x.dtype != torch.float16:
        raise TypeError("this reference implementation supports float16 only")
    shapes = (x.shape[1],) + (1,) * (3 - rank) + tuple(x.shape[2:])
    kernels = (weight.shape[1],) + (1,) * (3 - rank) + tuple(weight.shape[2:])
    batch = int(x.shape[0])
    ci, depth, height, width = (int(v) for v in shapes)
    co, _, kd, kh, kw = (int(v) for v in (weight.shape[0],) + kernels)
    strides = (1,) * (3 - rank) + _axes(stride, "stride", rank)
    dilations = (1,) * (3 - rank) + _axes(dilation, "dilation", rank)
    pads = _pad_pairs(rank, padding, (kd, kh, kw))
    if kh > MAX_FILTER or kw > MAX_FILTER or kd > MAX_FILTER:
        raise ValueError(f"filter side must not exceed {MAX_FILTER}")
    sd, sh, sw = strides
    dd, dh, dw = dilations
    od = max(0, (depth + pads[0][0] + pads[0][1] - dd * (kd - 1) - 1) // sd + 1)
    oh = max(0, (height + pads[1][0] + pads[1][1] - dh * (kh - 1) - 1) // sh + 1)
    ow = max(0, (width + pads[2][0] + pads[2][1] - dw * (kw - 1) - 1) // sw + 1)
    if max(depth + pads[0][0] + pads[0][1], dd * (kd - 1) + 1) > MAX_INDEX:
        raise ValueError("shape arithmetic exceeds the signed 32-bit index limit")
    return Problem(
        rank,
        tuple(int(v) for v in x.shape),
        tuple(int(v) for v in weight.shape),
        batch,
        ci,
        depth,
        height,
        width,
        co,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pads[0][0],
        pads[0][1],
        pads[1][0],
        pads[1][1],
        pads[2][0],
        pads[2][1],
        dd,
        dh,
        dw,
        bias is not None,
        od,
        oh,
        ow,
    )


def _ceil16(value):
    return (value + 15) // 16 * 16


def _weight_elements(problem, nt, ct):
    kt = ct * problem.rs
    return nt * kt + (_ceil16(problem.rs) - problem.rs) * nt * C0


def _reduction_steps(problem, ct):
    ci_tiles = (problem.ci + ct - 1) // ct
    return ci_tiles, problem.kd * ci_tiles


def _resources(problem, mt, nt, ct, resident, depth=1, nsubs=1, mrows=1):
    patch_height = (problem.kh - 1) * problem.dh + 1
    band_rows = patch_height + (mrows - 1) * problem.sh
    band = band_rows * problem.width
    feature = ((ct // C0 - 1) * band + _ceil16(band)) * C0
    _, steps = _reduction_steps(problem, ct)
    weight = _weight_elements(problem, nt, ct) * (steps if resident else 1)
    if not resident:
        weight *= depth
    kt = ct * problem.rs
    return {
        "L1": ITEM_BYTES * (feature * depth + weight + (nt if problem.has_bias else 0)),
        "L0A": ITEM_BYTES * mt * kt * depth,
        "L0B": ITEM_BYTES * nt * kt * depth,
        "L0C": 4 * mt * nt * nsubs,
        "BIAS": 4 * nt if problem.has_bias else 0,
    }


def _instruction_rejections(problem, mt, nt, ct, mrows=1):
    """LOAD3D / MMAD / DMA / Fixpipe 字段上限（与 ascendc 适配层约束表对齐）。"""
    patch_height = (problem.kh - 1) * problem.dh + 1
    band_rows = patch_height + (mrows - 1) * problem.sh
    band = band_rows * problem.width
    kt = ct * problem.rs
    fields = (
        ("load3d.feature.width", problem.width, LOAD3D_MAX_FEATURE),
        ("load3d.feature.height", band_rows, LOAD3D_MAX_FEATURE),
        ("load3d.stride.h", problem.sh, LOAD3D_MAX_STRIDE),
        ("load3d.stride.w", problem.sw, LOAD3D_MAX_STRIDE),
        ("load3d.dilation.h", problem.dh, LOAD3D_MAX_DILATION),
        ("load3d.dilation.w", problem.dw, LOAD3D_MAX_DILATION),
        ("load3d.pad.left", problem.pl, LOAD3D_MAX_PAD),
        ("load3d.pad.right", problem.pr, LOAD3D_MAX_PAD),
        ("load3d.channels", ct, UINT16_MAX),
        ("load3d.m_extent", _ceil16(min(mt, problem.ow)), UINT16_MAX),
        ("load3d.k_extent", kt, UINT16_MAX),
        ("mmad.m", mt, MAX_IMMEDIATE),
        ("mmad.n", nt, MAX_IMMEDIATE),
        ("mmad.k", kt, MAX_IMMEDIATE),
        ("dma.feature.n_value", band, UINT16_MAX),
        ("dma.feature.dst_nz_c0_stride", band, UINT16_MAX),
        ("dma.weight.n_value", problem.rs, UINT16_MAX),
        ("dma.weight.d_value", nt * ct, 4294967295),
        ("dma.weight.dst_nz_n_stride", nt, UINT16_MAX),
        ("dma.weight.dst_nz_c0_stride", problem.rs * nt, UINT16_MAX),
        ("fixpipe.dst_stride", problem.od * problem.oh * problem.ow, 4294967295),
    )
    return [
        f"{name}={value} exceeds {limit}"
        for name, value, limit in fields
        if value > limit
    ]


def _cost(problem, config, cores):
    """解析代价（字节当量）：A/B/Out 搬运 + 任务与归约循环开销，再乘 负载不均因子。"""
    rows = min(config.patch_height + (config.mrows - 1) * problem.sh, problem.height)
    cols = problem.width
    a_bytes = (
        config.tasks * config.reduction_steps * rows * cols * config.ct * ITEM_BYTES
    )
    if config.resident:
        weight_loads = max(config.groups, min(cores, config.tasks))
        w_bytes = (
            weight_loads * config.reduction_steps * config.nt * config.kt * ITEM_BYTES
        )
    else:
        w_bytes = (
            config.tasks * config.reduction_steps * config.nt * config.kt * ITEM_BYTES
        )
    o_bytes = (
        problem.batch * problem.co * problem.od * problem.oh * problem.ow * ITEM_BYTES
    )
    cost = float(a_bytes + w_bytes + o_bytes)
    cost += (
        TASK_OVERHEAD_BYTES * config.tasks
        + KLOOP_OVERHEAD_BYTES * config.tasks * config.reduction_steps
        + SUB_OVERHEAD_BYTES * config.tasks * config.reduction_steps * config.nsubs
    )
    # N 方向尾块浪费：最后一个 channel tile 只用到 Co % NT 个通道，但 A/B 与 MMAD 仍按 NT 付代价。
    n_waste = config.channel_tiles * config.nt / problem.co
    cost *= n_waste * n_waste
    # M 方向尾块浪费：最后一个 width tile 只用到 ow - k*MS 列，但 A/B/MMAD 仍按整块付代价。
    m_waste = (
        config.width_tiles * config.nsubs * config.mt / (config.mrows * problem.ow)
    )
    cost *= m_waste
    waves = -(-config.tasks // cores) if cores > 0 else 1
    imbalance = waves * cores / max(config.tasks, 1)
    # 离散任务不均按半权计入：实测 per-task 成本随 MT 增大而摊薄，全权惩罚会系统性误选小 tile。
    cost *= 1.0 + 0.5 * (imbalance - 1.0)
    # 任务数不足一核一个时（极小的输出）核空转是主要损失，按 sqrt 额外惩罚。
    if config.tasks < cores:
        cost *= (cores / config.tasks) ** 0.5
    # 双缓冲消除同缓冲 WAR 依赖，收益随归约步数增加（只有 1 步时没有可重叠的窗口）。
    if config.pipe_depth > 1:
        steps = max(config.reduction_steps, 1)
        cost *= 1.0 - 0.25 * (steps - 1) / steps
    return cost


def _candidates(steps, cap, upper):
    limit = min(cap, upper, MAX_TILE)
    values = {v for v in steps if 16 <= v <= limit}
    edge = (limit // 16) * 16
    if edge >= 16:
        values.add(edge)
    return sorted(values)


def _search(problem, capacities, cores):
    """枚举可行配置，按代价模型选最优；容量与指令字段是硬约束。"""
    core_budget = max(cores, 1)
    l0c_elems = capacities["L0C"] // 4
    failures = []
    best = None
    for ct in CT_STEPS:
        kt = ct * problem.rs
        if kt > MAX_IMMEDIATE or kt > UINT16_MAX:
            failures.append(f"CT={ct}: mmad.k={kt} exceeds field limit")
            continue
        mt_cap = min(capacities["L0A"] // (ITEM_BYTES * kt), l0c_elems // 16)
        nt_cap = min(
            capacities["L0B"] // (ITEM_BYTES * kt),
            (capacities["BIAS"] // 4) if problem.has_bias else 1024,
            l0c_elems // 16,
        )
        for nt in _candidates(NT_STEPS, nt_cap, _ceil16(problem.co)):
            for mt in _candidates(MT_STEPS, mt_cap, _ceil16(problem.ow)):
                if 4 * mt * nt > capacities["L0C"]:
                    continue
                rejected = _instruction_rejections(problem, mt, nt, ct)
                if rejected:
                    failures.append(f"tile {(mt, nt, ct)}: " + "; ".join(rejected))
                    continue
                nsubs_cap = min(capacities["L0C"] // (4 * mt * nt), MAX_SUBS)
                # 同一 task 覆盖多行输出（仅在 ow == width 的步长/对称 pad 场景下坐标可展平）。
                mrows_list = (1, 2, 3, 4) if problem.ow == problem.width else (1,)
                mrows_list = tuple(m for m in mrows_list if m <= problem.oh)
                for resident in (True, False):
                    for depth in (2, 1):
                        for mrows in mrows_list:
                            total_m = mrows * problem.ow
                            if (total_m + mt - 1) // mt > nsubs_cap:
                                continue
                            nsubs = (total_m + mt - 1) // mt
                            required = _resources(
                                problem, mt, nt, ct, resident, depth, nsubs, mrows
                            )
                            over = [
                                f"{level} requires {need} bytes; capacity {capacities.get(level, 0)}"
                                for level, need in required.items()
                                if need > capacities.get(level, 0)
                            ]
                            if over:
                                failures.append(
                                    f"tile {(mt, nt, ct)} resident={resident} depth={depth} "
                                    f"nsubs={nsubs} mrows={mrows}: " + "; ".join(over)
                                )
                                continue
                            width_tiles = 1
                            channel_tiles = (problem.co + nt - 1) // nt
                            ci_tiles, steps = _reduction_steps(problem, ct)
                            oh_groups = (problem.oh + mrows - 1) // mrows
                            spatial_tasks = problem.od * oh_groups
                            groups = problem.batch * channel_tiles
                            config = Config(
                                mt,
                                nt,
                                ct,
                                resident,
                                (problem.kh - 1) * problem.dh + 1,
                                kt,
                                width_tiles,
                                channel_tiles,
                                ci_tiles,
                                spatial_tasks,
                                groups,
                                groups * spatial_tasks,
                                steps,
                                0.0,
                                required["L1"],
                                depth,
                                nsubs,
                                mrows,
                            )
                            config = replace(
                                config, cost=_cost(problem, config, core_budget)
                            )
                            key = (
                                config.cost,
                                0 if resident else 1,
                                -depth,
                                -mrows,
                                -nsubs,
                                required["L1"],
                            )
                            if best is None or key < best[0]:
                                best = (key, config)
    if best is None:
        raise NotImplementedError(
            "no admitted tiling fits: " + " | ".join(failures[-6:])
        )
    return best[1]


class ConvKernel:
    """单计算主体：A 为列窗口行带，B 在容量允许时按 n 分组驻留；D 轴按 kd 分块归约。"""

    def __init__(self, problem, config):
        self.p = problem
        self.config = config
        self.mt, self.nt, self.ct = config.tile
        self.resident = config.resident
        self.kt = config.kt
        self.patch_height = config.patch_height
        self.mrows = config.mrows
        self.band_rows = self.patch_height + (self.mrows - 1) * problem.sh
        self.band = self.band_rows * problem.width
        self.feature_elements = (
            (self.ct // C0 - 1) * self.band + _ceil16(self.band)
        ) * C0
        self.weight_elements = _weight_elements(problem, self.nt, self.ct)
        self.width_tiles = config.width_tiles
        self.channel_tiles = config.channel_tiles
        self.ci_tiles = config.ci_tiles
        self.spatial_tasks = config.spatial_tasks
        self.tasks = config.tasks
        self.steps = config.reduction_steps
        self.plane = problem.depth * problem.height * problem.width
        self.hw = problem.height * problem.width
        self.out_plane = problem.od * problem.oh * problem.ow
        self.pipe_depth = config.pipe_depth
        self.nsubs = config.nsubs
        self.ms = config.nsubs * self.mt

    def _make_accumulators(self):
        accumulators = []
        for _ in range(self.nsubs):
            accumulators.append(
                c.Buffer(MemLoc.L0C, dtypes.float32, make_layout((self.mt, self.nt)))
            )
        return accumulators

    @staticmethod
    def _fill_zero(buffer, total_blocks):
        """按 32767 块上限分片清零（编译期展开），供启动时的一次性缓冲初始化。"""
        done = 0
        while done < total_blocks:
            count = min(32767, total_blocks - done)
            cube_fill_l1_zero(buffer, offset_elems=done * C0, blk_num=count)
            done += count

    @c.jit
    def _load_feature(
        self,
        feature: Tensor,
        x: Tensor,
        batch,
        input_depth,
        input_row,
        band_rows,
        channel_tile,
    ):
        channel = channel_tile * self.ct
        count = min(self.ct, self.p.ci - channel)
        valid_begin = max(0, input_row)
        valid_end = min(self.p.height, input_row + band_rows)
        outside = (input_depth < 0) or (input_depth >= self.p.depth)
        # 左右 padding 由 load3d 的 fmap pad 字段处理；上下/深度越界需要真实零行。
        padded = (valid_begin > input_row) or (valid_end < input_row + band_rows)
        # Ci 尾块只写 count 个通道，但 load3d 按 channels=CT 读取；未写的通道若是
        # 上一步/上一条任务残留（可能含 NaN/Inf），0×NaN 仍会污染输出，必须确定性清零。
        if padded or outside or count < self.ct:
            cube_fill_l1_zero(feature, blk_num=self.feature_elements // C0)
            c.cube_sync_pipe(PIPE.MTE2)
        if not outside and valid_end > valid_begin:
            source = c.from_tensor(
                x,
                layout=make_layout((self.p.ci, self.plane), stride=(self.plane, 1)),
                offset=batch * self.p.ci * self.plane * ITEM_BYTES,
            )
            base = input_depth * self.hw
            begin = base + valid_begin * self.p.width
            end = base + valid_end * self.p.width
            row_begin = (valid_begin - input_row) * self.p.width
            row_end = (valid_end - input_row) * self.p.width
            channel_end = channel + count
            c.mem_copy(
                feature[row_begin:row_end, 0:count],
                source[channel:channel_end, begin:end].permute(1, 0),
            )

    @c.jit
    def _load_weight(
        self,
        arena: Tensor,
        weight: Tensor,
        channel_tile,
        kd,
        channel_of_k,
        offset_bytes,
        scrub,
    ):
        channel = channel_of_k * self.ct
        count = min(self.ct, self.p.ci - channel)
        channel_start = channel_tile * self.nt
        channel_count = min(self.nt, self.p.co - channel_start)
        packed = c.from_tensor(
            arena, layout=make_layout((self.weight_elements,)), offset=offset_bytes
        )
        # 流式权重缓冲被所有 K 步复用：Ci 尾块只写 count 个通道，通道尾部会残留上一步
        # 的真实权重（K 归约会把 feature 尾通道乘上错误权重），必须在写入前整体清零。
        # 驻留缓冲每步是独立分块且启动时已整体清零，无需 scrub。
        if c.const_expr(scrub) and count < self.ct:
            cube_fill_l1_zero(packed, blk_num=self.weight_elements // C0)
            c.cube_sync_pipe(PIPE.MTE2)
        weight_source = c.from_tensor(
            weight,
            layout=make_layout(
                (self.p.co, self.p.ci, self.p.kd * self.p.rs),
                stride=(self.p.ci * self.p.kd * self.p.rs, self.p.kd * self.p.rs, 1),
            ),
        )
        weight_view = c.from_tensor(
            packed,
            layout=make_layout((self.nt, self.p.rs, self.ct)),
            data_format="nz",
            physical_layout=make_layout(
                (self.nt, self.ct // C0, _ceil16(self.p.rs) // 16, 16, C0),
                stride=(
                    C0,
                    self.p.rs * self.nt * C0,
                    16 * self.nt * C0,
                    self.nt * C0,
                    1,
                ),
            ),
        )
        channel_end = channel_start + channel_count
        channel_in_end = channel + count
        k_begin = kd * self.p.rs
        k_end = (kd + 1) * self.p.rs
        c.mem_copy(
            weight_view[0:channel_count, None, 0:count],
            weight_source[
                channel_start:channel_end,
                channel:channel_in_end,
                k_begin:k_end,
            ].permute(0, 2, 1),
        )

    @c.jit
    def _load_bias(
        self,
        bias: Tensor,
        bias_l1: Tensor,
        bias_table: Tensor,
        channel_start,
        channel_count,
    ):
        cube_fill_l1_zero(bias_l1, blk_num=self.nt // C0)
        c.cube_sync_pipe(PIPE.MTE2)
        channel_end = channel_start + channel_count
        c.mem_copy(
            bias_l1[0:channel_count],
            bias[channel_start:channel_end],
        )
        c.mem_copy(bias_table, bias_l1)

    @c.jit
    def _load_a(self, a_write: Tensor, feature: Tensor, out_col, row_count):
        c.load3d(
            a_write,
            feature,
            feature=c.Load3DFeatureMap(
                height=self.band_rows,
                width=self.p.width,
                channels=self.ct,
                pad_left=self.p.pl,
                pad_right=self.p.pr,
            ),
            window=c.Load3DWindow(
                m_start=out_col, m_extent=row_count, k_extent=self.kt
            ),
            filter=(self.p.kh, self.p.kw),
            stride=(self.p.sh, self.p.sw),
            dilation=(self.p.dh, self.p.dw),
            destination=c.Load3DDestination(k_stride_fractals=self.mt // 16),
        )

    @c.kernel
    def kernel(self, x: Tensor, weight: Tensor, bias: Tensor, output: Tensor):
        features = RingBuffer(
            MemLoc.L1,
            dtypes.float16,
            make_layout((self.band, self.ct)),
            depth=self.pipe_depth,
            data_format="nz",
            physical_layout=make_layout(
                (self.ct // C0, _ceil16(self.band) // 16, 16, C0),
                stride=(self.band * C0, 16 * C0, C0, 1),
            ),
        )
        weights = None
        if c.const_expr(self.resident):
            weights = c.Buffer(
                MemLoc.L1,
                dtypes.float16,
                make_layout((self.steps * self.weight_elements,)),
                data_format="nd",
            )
        else:
            weights = RingBuffer(
                MemLoc.L1,
                dtypes.float16,
                make_layout((self.weight_elements,)),
                depth=1,
                data_format="nd",
            )
        left = RingBuffer(
            MemLoc.L0A,
            dtypes.float16,
            make_layout((self.mt, self.kt)),
            depth=self.pipe_depth,
        )
        right = RingBuffer(
            MemLoc.L0B,
            dtypes.float16,
            make_layout((self.nt, self.kt)),
            depth=self.pipe_depth,
        )
        accumulators = self._make_accumulators()
        if c.const_expr(self.p.has_bias):
            bias_l1 = c.Buffer(
                MemLoc.L1, dtypes.float16, make_layout((self.nt,)), data_format="nd"
            )
            bias_table = c.Buffer(
                MemLoc.BIAS, dtypes.float32, make_layout((self.nt,)), data_format="nd"
            )
        # 权重缓冲区只在启动时整体清零一次：后续每步只覆写有效的 rs 行/通道，
        # 分形尾部保持零，避免每步重复清零（见 capability-mapping G7）。
        if c.const_expr(self.resident):
            self._fill_zero(weights, self.steps * self.weight_elements // C0)
        else:
            for _ in c.range_constexpr(self.pipe_depth):
                self._fill_zero(weights.next_write(), self.weight_elements // C0)
        c.cube_sync_pipe(PIPE.MTE2)

        # 每核处理一段连续任务区间：n 分组在外、输出平面/行在内，权重按分组复用。
        block = c.get_block_idx()
        base = self.tasks // c.get_block_num()
        extra = self.tasks % c.get_block_num()
        shifted = block if block < extra else extra
        begin = block * base + shifted
        end = begin + base + (1 if block < extra else 0)
        last_group = c.dtypes.int64(-1)
        for task in range(begin, end):
            group = task // self.spatial_tasks
            spatial = task % self.spatial_tasks
            batch = group // self.channel_tiles
            channel_tile = group % self.channel_tiles
            row_index = spatial // self.width_tiles
            out_col = (spatial % self.width_tiles) * self.ms
            oh_groups = (self.p.oh + self.mrows - 1) // self.mrows
            out_depth = row_index // oh_groups
            out_row = (row_index % oh_groups) * self.mrows
            row_count = min(self.mrows, self.p.oh - out_row)
            total_m = row_count * self.p.ow
            if c.const_expr(self.mrows > 1):
                last_begin = max(out_col, out_col + total_m - self.mt)
            else:
                last_begin = max(0, self.p.ow - self.mt)
            band_rows = self.patch_height + (row_count - 1) * self.p.sh
            row_base = out_depth * self.p.oh * self.p.ow + out_row * self.p.ow
            input_row = out_row * self.p.sh - self.p.pt
            input_depth = out_depth * self.p.sd - self.p.pd0
            channel_start = channel_tile * self.nt
            channel_count = min(self.nt, self.p.co - channel_start)
            if c.const_expr(self.resident):
                if group != last_group:
                    for kd in range(self.p.kd):
                        for chunk in range(self.ci_tiles):
                            self._load_weight(
                                weights,
                                weight,
                                channel_tile,
                                kd,
                                chunk,
                                (kd * self.ci_tiles + chunk)
                                * self.weight_elements
                                * ITEM_BYTES,
                                False,
                            )
                    if c.const_expr(self.p.has_bias):
                        self._load_bias(
                            bias, bias_l1, bias_table, channel_start, channel_count
                        )
                    last_group = group
            else:
                if c.const_expr(self.p.has_bias):
                    self._load_bias(
                        bias, bias_l1, bias_table, channel_start, channel_count
                    )
            step = 0
            for kd in range(self.p.kd):
                for chunk in range(self.ci_tiles):
                    feature = features.next_write()
                    self._load_feature(
                        feature,
                        x,
                        batch,
                        input_depth + kd * self.p.dd,
                        input_row,
                        band_rows,
                        chunk,
                    )
                    if c.const_expr(self.resident):
                        packed = c.from_tensor(
                            weights,
                            layout=make_layout((self.weight_elements,)),
                            offset=(kd * self.ci_tiles + chunk)
                            * self.weight_elements
                            * ITEM_BYTES,
                        )
                    else:
                        packed = weights.next_write()
                        self._load_weight(
                            packed, weight, channel_tile, kd, chunk, 0, True
                        )
                    b_write = right.next_write()
                    weight_matrix = c.from_tensor(
                        packed, layout=make_layout((self.nt, self.kt)), data_format="nz"
                    )
                    c.mem_copy(b_write, weight_matrix)
                    # B 侧每个 step 只读一次，供所有 M 子块共用（RingBuffer 读写游标必须配对）。
                    b = right.next_read()
                    # 同一 feature/weight 分块被 nsubs 个 M 子块共享：A 侧 DMA 与 B 侧装载均摊。
                    for sub in c.range_constexpr(self.nsubs):
                        sub_begin = min(out_col + sub * self.mt, last_begin)
                        sub_extent = min(self.mt, out_col + total_m - sub_begin)
                        a_write = left.next_write()
                        self._load_a(a_write, feature, sub_begin, sub_extent)
                        a = left.next_read()
                        acc = accumulators[sub]
                        if c.const_expr(self.p.has_bias):
                            c.matmul(acc, a, b, bias=bias_table, init=(step == 0))
                        else:
                            c.matmul(acc, a, b, init=(step == 0))
                    step = step + 1
            target = c.from_tensor(
                output,
                layout=make_layout(
                    (self.p.co, self.out_plane), stride=(self.out_plane, 1)
                ),
                offset=batch * self.p.co * self.out_plane * ITEM_BYTES,
            )
            for sub in c.range_constexpr(self.nsubs):
                sub_begin = min(out_col + sub * self.mt, last_begin)
                sub_extent = min(self.mt, out_col + total_m - sub_begin)
                channel_end = channel_start + channel_count
                dest_begin = row_base + sub_begin
                dest_end = dest_begin + sub_extent
                destination = target[
                    channel_start:channel_end, dest_begin:dest_end
                ].permute(1, 0)
                c.mem_copy(
                    destination,
                    accumulators[sub][0:sub_extent, 0:channel_count].convert(
                        dtypes.float16
                    ),
                )

    @c.host
    def run(self, x, weight, bias, output, core_num: int):
        self.kernel[core_num](x, weight, bias, output)

    @c.host
    def run_without_bias(self, x, weight, output, core_num: int):
        self.kernel[core_num](x, weight, x, output)


_PROGRAMS = {}
_PLANS = {}


def _target(x):
    stream = torch.npu.current_stream(x.device)
    info = c.get_platform_info(stream=stream)
    if not info.available or info.cube_core_num <= 0:
        raise RuntimeError("cannot determine effective target resources")
    capacities = {
        name: c.get_mem_size("bt" if name == "BIAS" else name.lower())
        for name in ("L1", "L0A", "L0B", "L0C", "BIAS")
    }
    return info.npu_arch, int(info.cube_core_num), capacities, stream


def _cached_plan(problem, cores, capacities):
    key = (problem, cores, tuple(sorted(capacities.items())))
    config = _PLANS.get(key)
    if config is None:
        config = _PLANS[key] = _search(problem, capacities, cores)
    return config


def _compile(problem, config):
    kernel = ConvKernel(problem, config)
    x_spec = c.TensorSpec(problem.shape, dtypes.float16)
    w_spec = c.TensorSpec(problem.weight_shape, dtypes.float16)
    y_spec = c.TensorSpec(problem.output_shape, dtypes.float16)
    if problem.has_bias:
        b_spec = c.TensorSpec((problem.co,), dtypes.float16)
        return c.compile(kernel.run, x_spec, w_spec, b_spec, y_spec, c.dtypes.int64)
    return c.compile(kernel.run_without_bias, x_spec, w_spec, y_spec, c.dtypes.int64)


def _conv(rank, x, weight, bias, stride, padding, dilation):
    problem = _normalize(x, weight, bias, stride, padding, dilation)
    if problem.rank != rank:
        raise ValueError(f"expected {rank}-D convolution, got {problem.rank}-D")
    output = torch.empty(problem.output_shape, dtype=x.dtype, device=x.device)
    if problem.empty:
        return output
    arch, cores, capacities, stream = _target(x)
    if arch not in ("dav-3510", "dav3510"):
        raise NotImplementedError(
            f"target architecture {arch!r} has not been validated"
        )
    config = _cached_plan(problem, cores, capacities)
    launch_cores = max(1, min(cores, config.tasks))
    key = (problem, config.tile, config.resident, str(x.device), c.__version__)
    program = _PROGRAMS.get(key)
    if program is None:
        program = _PROGRAMS[key] = _compile(problem, config)
    try:
        if problem.has_bias:
            program(x, weight, bias, output, launch_cores)
        else:
            program(x, weight, output, launch_cores)
    finally:
        stream.synchronize()
    return output


def conv1d(x, weight, bias=None, *, stride=1, padding=0, dilation=1):
    """Conv1D 前向（无核翻转 cross-correlation），返回新分配的 NPU 输出。"""
    return _conv(1, x, weight, bias, stride, padding, dilation)


def conv2d(x, weight, bias=None, *, stride=1, padding=0, dilation=1):
    """Conv2D 前向（无核翻转 cross-correlation），返回新分配的 NPU 输出。"""
    return _conv(2, x, weight, bias, stride, padding, dilation)


def conv3d(x, weight, bias=None, *, stride=1, padding=0, dilation=1):
    """Conv3D 前向（无核翻转 cross-correlation），返回新分配的 NPU 输出。"""
    return _conv(3, x, weight, bias, stride, padding, dilation)


def close():
    """释放缓存程序；公开调用同步返回，close 后可重新编译。"""
    for program in _PROGRAMS.values():
        program.close()
    _PROGRAMS.clear()
    _PLANS.clear()
