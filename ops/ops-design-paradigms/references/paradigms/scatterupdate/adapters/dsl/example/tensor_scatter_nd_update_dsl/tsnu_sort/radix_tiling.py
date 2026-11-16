# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""radix_sort_simd DSL 移植版 tiling —— AscendC SortTilingCompute 的 Python 对应物。

与 AscendC sort_lib_tiling.h 的差异（设计文档 §12.4 P1 / §14.5）：
- UB 布局按 DSL 侧自有缓冲重算：SortDSL 组件物理 4B/元素（val 1 原地
  §19.63 + idx8 乒乓 2 + pristine 1），tile 定尺保守取 5（单波区防
  关键路径拉长）；I/O 通道 io_depth=1（v1 串行）
- tmpUb 概念删除：AscendC::Sort 的共享临时区在 DSL 侧为 SortDSL 自管缓冲（已含在
  每元素成本与 FIXED_OVERHEAD 中），无独立 tmpUbSize 迭代收缩环节
- 单核快路径（§20，= ASC TrySingleCore 同构）：ub_requirement_full(npad, ds)
  ≤ usable 即命中——含输入通道/val_a/val_b/idx_out(u32)+SortImpl 内部缓冲，
  为 ASC「2·xUb + idxTotal(×2 for i64) + scTmp」的并集重排（§19.61：
  i64 半区核内构造——idx_out 双倍，= ASC idxTotal ×2 for i64）；命中时
  workspace=0、tileCount=1、activeCores=1；force_multi 可强制多核（测试/归因）
- 空输入：host 侧跳过 kernel 发射（= AscendC kernel early-return 语义等价）
- UB 容量运行时查询 get_mem_size("ub")（文档口径，勿硬编码）；
  混合 SIMT 需另扣 DCache 32KB（R5 风险项，P2 编译验证）

六段 workspace 布局与 AscendC 逐段同公式（设计文档 §4.1 双侧契约）：
  [0] exclBins    256×rounds×counterSize
  [1] globalHist  rounds×tileCount×256×counterSize（lookback 介质，带状态标志）
  [2] idxDb       N×counterSize（轮间索引双缓冲，u32 宽度驻留）
  [3] histTile + histCumsum   各 tileCount×256×u16（两段）
  [4] xB8         tileCount×numTileData（当前字节缓存）
  [5] valueDb     N×dtypeSize（轮间值双缓冲）

算子 dtype 矩阵（op_def）：值 i/u16/32/64（无 8 位、无 fp），索引 i32/i64。
"""

from dataclasses import dataclass, field

import cannbotdsl

# SortDSL 组件内部固定开销（lib/sort_dsl.py FIXED_OVERHEAD，唯一真值）
from .sort_dsl import FIXED_OVERHEAD as _SORTER_FIXED, ub_requirement_full

BIN_NUM = 256  # 每轮 256 bin
DCACHE_SIZE = 32 * 1024  # 混合 SIMT 的 DCache 预留（AscendC SetLocalMemorySize 同口径）
UB_ALIGN = 32  # 段内对齐（= AscendC UB_ALIGN_BYTES）
WS_ALIGN = 128  # workspace 总对齐（= AscendC WORKSPACE_ALIGN_BYTES）
INT32_SAFE_LIMIT = 1 << 30  # 32 位计数安全上限（元素数，= AscendC INT32_SAFE_LIMIT）

# ── DSL 侧 UB 布局成本（kernel 缓冲声明必须与本表一致——单一真值）──
# 每元素（§19.5 C6：B 相散射改通道中转）：
#   ch ds / ch_val ds / b8_stage 1 / ch_b8 1 / sorter 5 / ch_sv 1 / ch_si 4 / ch_idx 4
# 固定开销见 fixed_cost()：sorter 516 + 2×(u16×256) + 4×(int32×256) + 2×(u16×256) + 2×(int32×256)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def per_elem_cost(dtype_size: int, counter_size: int, index_size: int = 4) -> int:
    """每元素 UB 字节数（io_depth=1）。kernel 缓冲布局的单一真值，改布局必须同步改此表。

    §17.6 统一架构：B 相散射 GM 直读直写（无值/索引通道），A 相值为 u32 打包
    通道（= ds B/元素）。§19.39：ch_idx 宽 = index_size（追逐域统一 IdxT）。
    """
    return (
        dtype_size  # ch（A 相 u32 打包值通道；ds=2 原生同为 ds B）
        + dtype_size  # ch_val（B 相原生值通道，C6 中转）
        + 1  # b8_stage（A 相字节暂存）
        + 1  # ch_b8（B 相排序源）
        + 5  # sorter 内部定尺（物理 4B §19.63 val 原地；tile 取保守 5——
        # 单波区 tile 增大直接拉长块关键路径）
        + 1  # ch_sv（排序字节输出）
        + 4  # ch_si（排序索引输出）
        + index_size
    )  # ch_idx（B 相追逐通道——宽=IdxT，§19.39）


def fixed_cost(counter_size: int) -> int:
    """固定 UB 字节开销（kernel 常驻缓冲，§16 布局单一真值）。

    sorter 固定 516 + hist/cumsum 暂存(u16×256×2) + 四个 int32×256 工作区
    （excl_acc/hist32/cum32/pos_table）+ ch_hist/ch_cum(u16×256×2) +
    ch_gh/ch_excl(int32×256×2)。
    """
    return (
        _SORTER_FIXED
        + 2 * BIN_NUM * 2  # hist_ub + cumsum_ub
        + 4 * BIN_NUM * counter_size  # excl_acc/hist32/cum32/pos_table
        + 2 * BIN_NUM * 2  # ch_hist + ch_cum
        + 2 * BIN_NUM * counter_size
    )  # ch_gh + ch_excl


def workspace_offsets(
    total_elements: int,
    tile_count: int,
    num_tile_data: int,
    dtype_size: int,
    counter_size: int,
    index_size: int = 4,
) -> dict:
    """六段 workspace 偏移（字节）。host 视图切分与 kernel 段对齐的唯一真值
    （设计文档 §14.5：三处一致收敛为单一函数）。

    与 AscendC ComputeWorkspaceBytes 同构（§19.39/§19.40）：
    - [2]/[4]/[5] 单段且 **n 基**（§19.40 M15 翻案：mem_copy = DataCopyPad
      等价，tail-aware tile_slice 变长装载；输出参与 ping-pong）；
      idx 段宽 = index_size（追逐域统一 IdxT）
    """
    rounds = dtype_size
    off = {}
    cur = 0
    cur += _align_up(BIN_NUM * rounds * counter_size, UB_ALIGN)  # [0] exclBins
    off["excl_bins"] = 0
    off["excl_bins_end"] = cur
    off["global_hist"] = cur
    cur += _align_up(
        rounds * tile_count * BIN_NUM * counter_size, UB_ALIGN
    )  # [1] globalHist
    off["global_hist_end"] = cur
    off["idx_db"] = cur
    cur += _align_up(
        total_elements * index_size, UB_ALIGN
    )  # [2] idxDb 单份（宽=IdxT；n 基，§19.40）
    off["idx_db_end"] = cur
    off["hist_tile"] = cur
    cur += _align_up(tile_count * BIN_NUM * 2, UB_ALIGN)  # [3a] histTile
    off["hist_tile_end"] = cur
    off["hist_cumsum"] = cur
    cur += _align_up(tile_count * BIN_NUM * 2, UB_ALIGN)  # [3b] histCumsum
    off["hist_cumsum_end"] = cur
    off["x_b8"] = cur
    cur += _align_up(total_elements, UB_ALIGN)  # [4] xB8（n 基，§19.40）
    off["x_b8_end"] = cur
    off["value_db"] = cur
    cur += _align_up(
        total_elements * dtype_size, UB_ALIGN
    )  # [5] valueDb 单份（n 基，§19.40）
    off["value_db_end"] = cur
    off["total"] = _align_up(cur, WS_ALIGN)
    return off


@dataclass
class RadixTiling:
    """LSD radix tiling 结果（对应 AscendC SortTilingResult，去 tmpUb）。"""

    total_elements: int
    dtype_size: int
    index_size: int  # 4 (i32) / 8 (i64)
    counter_size: int  # 4 (u32 计数) / 8 (i64 计数)
    rounds: int  # = dtype_size
    descending: bool
    num_tile_data: int
    tile_count: int
    active_cores: int
    workspace_bytes: int
    ws_offsets: dict = field(default_factory=dict)
    ub_budget: int = 0  # 本次规划采用的 UB 预算（诊断用）
    single_core: bool = False  # §20 单核快路径命中（= ASC isSingleCore）
    npad: int = 0  # 单核模式 ceil256(N)（full 模式缓冲容量）

    @property
    def is_int32_safe(self) -> bool:
        return self.counter_size == 4


def try_single_core(
    total_elements: int, dtype_size: int, usable: int, index_size: int = 4
) -> int:
    """单核快路径判定（= ASC sort_lib_tiling.h TrySingleCore，§20 同构）。

    ASC 预算「2·xUbSize + idxTotalUb(i64 ×2) + scTmpUb」在 DSL 侧并集重排为
    ub_requirement_full(npad, ds, index_size)（ch 输入 + val_a/val_b +
    idx_out(u32，idx64 双倍) + SortImpl 内部 = ASC 的 IO 与 tmp 之并）。
    命中返回 npad，否则 0。
    """
    if total_elements <= 0 or total_elements > 0xFFFFFFFF:
        return 0
    npad = _ceil_div(total_elements, 256) * 256
    if ub_requirement_full(npad, dtype_size, index_size) <= usable:
        return npad
    return 0


def radix_tiling(
    total_elements: int,
    dtype_size: int,
    index_size: int,
    descending: bool = False,
    core_count: int = None,
    ub_bytes: int = None,
    dcache_size: int = DCACHE_SIZE,
    tile_override: int = None,
    force_multi: bool = False,
    counter_mode: int = None,
) -> RadixTiling:
    """tiling 主入口（AscendC SortTilingCompute 的 DSL 布局版）。

    total_elements == 0 时返回占位结果（host 侧跳过 kernel 发射）。
    ub_bytes 默认运行时查询 get_mem_size("ub")（文档口径，勿硬编码）。
    """
    if total_elements < 0:
        raise ValueError(f"total_elements 须非负，得到 {total_elements}")
    if dtype_size not in (2, 4, 8):
        raise ValueError(f"dtype_size 须为 2/4/8（算子 dtype 矩阵），得到 {dtype_size}")
    if index_size not in (4, 8):
        raise ValueError(f"index_size 须为 4/8，得到 {index_size}")

    if core_count is None:
        core_count = cannbotdsl.get_platform_info().vector_core_num
    if ub_bytes is None:
        ub_bytes = cannbotdsl.get_mem_size("ub")

    counter_size = (
        int(counter_mode)
        if counter_mode in (4, 8)
        else (4 if total_elements <= INT32_SAFE_LIMIT else 8)
    )
    rounds = dtype_size

    if total_elements == 0:
        return RadixTiling(
            0,
            dtype_size,
            index_size,
            counter_size,
            rounds,
            descending,
            1,
            1,
            1,
            0,
            {},
            ub_bytes,
        )

    usable = ub_bytes - dcache_size
    if usable <= fixed_cost(counter_size):
        raise ValueError(
            f"UB 不足：usable={usable}B ≤ 固定开销 {fixed_cost(counter_size)}B"
            f"（ub={ub_bytes}B − dcache={dcache_size}B）"
        )

    if not force_multi:
        sc_npad = try_single_core(total_elements, dtype_size, usable, index_size)
        if sc_npad:
            return RadixTiling(
                total_elements,
                dtype_size,
                index_size,
                counter_size,
                rounds,
                descending,
                total_elements,
                1,
                1,
                0,
                {},
                usable,
                single_core=True,
                npad=sc_npad,
            )

    per_elem = per_elem_cost(dtype_size, counter_size, index_size)
    if tile_override:
        tile = tile_override // BIN_NUM * BIN_NUM
        if tile <= 0:
            raise ValueError(f"tile_override={tile_override} 过小")
    else:
        tile = (usable - fixed_cost(counter_size)) // per_elem
        tile = tile // BIN_NUM * BIN_NUM
    if tile <= 0:
        raise ValueError(
            f"UB 不足以容纳最小 tile（usable={usable}B，per_elem={per_elem}B，"
            f"固定 {fixed_cost(counter_size)}B）"
        )

    tile_count = _ceil_div(total_elements, tile)
    active_cores = min(core_count, tile_count)
    off = workspace_offsets(
        total_elements, tile_count, tile, dtype_size, counter_size, index_size
    )

    return RadixTiling(
        total_elements,
        dtype_size,
        index_size,
        counter_size,
        rounds,
        descending,
        tile,
        tile_count,
        active_cores,
        off["total"],
        off,
        usable,
    )
