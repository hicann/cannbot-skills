# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""tensor_scatter_nd_update DSL wrapper（双门控分派 + inplace 语义 + 视图 pack）。

分派（= ASC v2.0 tiling registry 双模板 IsCapable 同构）：
  torch.are_deterministic_algorithms_enabled() → det（三段式段尾覆盖）
  否则 → nondet（直写覆盖；无 dtype 白名单、无 dupRatio——v2.0 实现同构）

视图 pack（= ASC ComputePackParam 的 wrapper 层等价物，零拷贝）：
  合轴 + pack 本质只是 shape/dtype 的重解释——对连续张量直接建视图：
    var_v = var.reshape(shape[:K] + (slice,)).view(packed_dtype)
    upd_v = upd.reshape(N, slice).view(packed_dtype)
  kernel 零改动：dtype 泛化的整型纯拷贝天然承载 pack；lin 直接在 packed 域
  （视图 strides 预除好，比 ASC 还省写侧 key/W 除法）。任务量、每元素除法、
  metadata 读同步 ÷W。
  W 递降择优：slice 整除 + **首地址 pack 后元素对齐**（SIMT 硬约束：非元素
  对齐地址不可访问，0x3 按 int8 可读、按 int32 不可读；访问地址 = base + k·W
  恒 W 对齐 → 仅需查 base）。var 为 inplace 不可拷贝对齐 → 不满足降档，
  全不满足 W=1（元素级）。位级别名（f8/bool → uint8）与 pack 同机制家族。

det tiling（= ASC DetTiling，MDE §11.5.2）：
  - key_dt：totalVarElems 三档（i16/i32/i64；pack 后域 = 视图元素数）
  - sort：radix_tiling(n_idx, key_ds, 4, force_multi=True)；单核判定 sc_cap
  - tile/rounds/counter8/is_single_core 决定 ws 定形与程序变体；
    active_cores/launch 由 @host 闭式派生（SC 变体恒 1）

R2/R3-E 帧契约（零标量零 H2D）：
  - 结构化 8 维预留（1-padding）直传：var / indices(N,K) / updates(N,sNum)
  - kernel VF 外按 8 维预留静态读取 var metadata（= ASC 定长数组），
    维度循环展开卫（= `for i in range(8): if i >= rank: continue`）；
    n_idx/k_dim/s_num 由 @host 从 shape 派生 → nondet 3 槽 / det 13 槽
  - K 仍运行期变量：shape 变更零重编译（8 维预留布局在 wrapper 侧变化）

workspace 单张量布局（32B 对齐段；无 srcpos——nd 无 SUBSET）：
  [sort 段 ×5] + db_v + db_i + lin + sorted + perm；NPU 设备端创建（免 MB 级 H2D）。

inplace：var 原地覆盖（非连续 var 不支持）；越界/负索引 skip（无 err flag，
双路径语义一致）。

异步语义（= 标准 torch 算子契约）：enqueue-and-return——本算子只在当前流上
入队（ws 设备填充 / kernel 同流有序），**不做任何内部
synchronize**；写入可见性由流序保证，同步权归调用者（.cpu()/.item()/显式
sync 天然覆盖）。graph-capture 友好（capture 区间内 sync 非法——本算子无）。
"""

import os
import sys

# 自包含：本目录入 path（文件夹可整体移动；平名导入防连字符目录名）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from cannbotdsl import get_mem_size, dtypes

from tsnu_sort.radix_tiling import radix_tiling
from tensor_scatter_nd_update_kernel import (
    build_det_program,
    build_nondet_program,
    key_dtype_of,
    MAXD,
)


def _pad8(t):
    """8 维预留视图（1-padding；= ASC 定长 shape/stride 数组的承载形态）。"""
    return t.view(*t.shape, *([1] * (MAXD - t.dim())))


# = ASC 10 dtype 减 hifloat8（torch 接口不可表达）
_VAL_DT = {
    torch.bfloat16: "bfloat16",
    torch.bool: "bool",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int64: "int64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.float8_e5m2: "float8_e5m2",
    torch.float8_e4m3fn: "float8_e4m3fn",
}
_IDX_DT = {torch.int32: "int32", torch.int64: "int64"}

# 位级别名（f8：DSL 无可执行标量 wrapper；bool：统一入 1B pack 族）——
# 纯搬运语义按位解释为同宽 uint8（bit 保真；与 u8 共享编译程序）
_DTYPE_ALIAS = {
    torch.float8_e5m2: torch.uint8,
    torch.float8_e4m3fn: torch.uint8,
    torch.bool: torch.uint8,
}

# pack 后字节宽 → 视图 dtype（整型纯拷贝承载，bit 保真）
_PACK_DT = {2: torch.uint16, 4: torch.uint32, 8: torch.uint64}


def _align32b(x):
    return (x + 31) // 32 * 32


def _pack_param(slice_size, elem_bytes, var_ptr, upd_ptr):
    """W 递降择优（= ASC ComputePackParam 同构 + SIMT 对齐硬约束）。

    条件：slice_size % W == 0 且 var/upd 首地址 % (W·elem) == 0
    （SIMT 无法访问非元素对齐地址；访问地址 = base + k·W 恒 W 对齐，
    仅需查 base。var inplace 不可拷贝对齐 → 不满足降档，全不满足 W=1）。
    """
    w = 8 // elem_bytes
    while w > 1:
        pb = w * elem_bytes
        if slice_size % w == 0 and var_ptr % pb == 0 and upd_ptr % pb == 0:
            return w
        w //= 2
    return 1


def tensor_scatter_nd_update(var, indices, updates):
    """inplace：var 前 K 维按 indices 覆盖写 updates（确定性 = 排序段尾覆盖）。

    = TF tensor_scatter_nd_update 语义（越界/负索引跳过该 update；重复索引
    last-write-wins——nondet 路径下为任意赢）。
    输入契约：var/indices/updates 须为 NPU 张量且同设备（不做隐式搬移）。
    """
    if (
        var.device.type != "npu"
        or indices.device.type != "npu"
        or updates.device.type != "npu"
    ):
        raise ValueError(
            "tensor_scatter_nd_update requires NPU var/indices/updates tensors"
        )
    if indices.device != var.device or updates.device != var.device:
        raise ValueError("var/indices/updates must share the same NPU device")
    if var.dtype not in _VAL_DT:
        raise TypeError(
            f"var dtype {var.dtype} 不支持，支持 {sorted(str(d) for d in _VAL_DT)}"
        )
    if indices.dtype not in _IDX_DT:
        raise TypeError(f"indices dtype {indices.dtype} 须为 int32/int64")
    if updates.dtype != var.dtype:
        raise TypeError("updates dtype 须与 var 一致")
    rank = var.dim()
    if not (1 <= rank <= 8):
        raise ValueError(f"rank(var) 须在 1~8，got {rank}")
    idx_rank = indices.dim()
    if idx_rank < 1:
        raise ValueError(f"rank(indices) 须 ≥ 1，got {idx_rank}")
    k_dim = int(indices.shape[-1])
    if not (1 <= k_dim <= rank):
        raise ValueError(f"K=indices.shape[-1]={k_dim} 须在 [1, rank={rank})")
    # updates.shape == indices.shape[:-1] + var.shape[K:]（逐维，= ASC ValidateShape）
    expected = tuple(indices.shape[:-1]) + tuple(var.shape[k_dim:])
    if tuple(updates.shape) != expected:
        raise ValueError(
            f"updates.shape {tuple(updates.shape)} 须为 "
            f"indices.shape[:-1] + var.shape[K:] = {expected}"
        )

    if not var.is_contiguous():
        raise ValueError("var 非连续暂不支持（需原址写）")
    idx_c = indices if indices.is_contiguous() else indices.contiguous()
    upd_c = updates if updates.is_contiguous() else updates.contiguous()

    n_idx = idx_c.numel() // k_dim
    # 空输入早退（= ASC SE §5.4：indices 空 / var 含 0 维 → 输出 = var 本身）
    if n_idx == 0 or var.numel() == 0:
        return var

    # ── 位级别名（f8/bool → uint8，同 storage 原址写）──
    alias = _DTYPE_ALIAS.get(var.dtype)
    var_arg = var.view(alias) if alias is not None else var
    upd_arg = upd_c.view(alias) if alias is not None else upd_c

    # ── 视图 pack（合轴 + dtype 重解释，零拷贝）──
    s_num = 1
    for d in range(k_dim, rank):
        s_num *= var.shape[d]
    elem_bytes = var_arg.element_size()
    w = _pack_param(s_num, elem_bytes, var_arg.data_ptr(), upd_arg.data_ptr())
    if w > 1:
        packed = _PACK_DT[w * elem_bytes]
        var_v = var_arg.reshape(var_arg.shape[:k_dim] + (s_num,)).view(packed)
        upd_v = upd_arg.reshape(n_idx, s_num).view(packed)
        s_num //= w
        val_dt = str(packed).split(".")[-1]
        key_domain = var_v.numel()
    else:
        var_v, upd_v = var_arg, upd_arg
        val_dt = _VAL_DT[alias] if alias is not None else _VAL_DT[var.dtype]
        key_domain = var.numel()
    idx_dt = _IDX_DT[indices.dtype]

    # ── 分派（双门控：v2.0 = ASC IsCapable 唯一门控——确定性开关）──
    deterministic = torch.are_deterministic_algorithms_enabled()
    use_nondet = not deterministic

    # ── R3-E：结构化 8 维预留视图（metadata 直读替代 params；零 H2D）──
    var_s = _pad8(var_v)
    idx2 = idx_c if idx_c.dim() == 2 else idx_c.reshape(1, k_dim)
    idx_s = _pad8(idx2)
    upd_s = _pad8(upd_v.reshape(n_idx, s_num))

    if use_nondet:
        # launch 由 @host 闭式派生（min(cores, ceil(N·sNum/1024))）
        prog = build_nondet_program(val_dt, idx_dt)
        prog(var_s, idx_s, upd_s)
        return var

    # ── det 路径 ──
    key_dt = key_dtype_of(key_domain)
    key_ds = getattr(dtypes, key_dt).width // 8
    t = radix_tiling(n_idx, key_ds, 4, force_multi=True)
    tile, rounds = t.num_tile_data, t.rounds
    counter8 = t.counter_size == 8
    tc = t.tile_count
    # 单核判定（= ASC TrySingleCore / scatter_add 同款公式）：n_idx ≤ 单核容量
    usable = get_mem_size("ub") - 32768
    sc_cap = (usable - 1024) // (3 * key_ds + 15)  # 复合版余量
    sc_cap = sc_cap // 256 * 256
    is_single_core = n_idx <= sc_cap
    tile_for_sort = sc_cap if is_single_core else tile
    tc_sort = (n_idx + tile_for_sort - 1) // tile_for_sort

    # ── workspace 布局（单张量 32B 段；无 srcpos；NPU 设备端创建免 H2D）──
    segs = {}
    cur = 0
    for name, dt, shape_ in (
        ("x_b8", torch.uint8, (tc_sort * tile_for_sort,)),
        ("hist_tile", torch.uint16, (tc, 256)),
        ("hist_cumsum", torch.uint16, (tc, 256)),
        ("excl_bins", torch.int64 if counter8 else torch.int32, (rounds, 256)),
        ("global_hist", torch.int64 if counter8 else torch.int32, (rounds * tc, 256)),
    ):
        need = 1
        for x in shape_:
            need *= x
        span = _align32b(need * dt.itemsize)
        segs[name] = (cur, span, dt, shape_)
        cur += span
    ws = torch.zeros(cur, dtype=torch.uint8, device="npu")

    def seg(name):
        off, span, dt, shape_ = segs[name]
        t_ = ws.narrow(0, off, span).view(dt)
        need = 1
        for x in shape_:
            need *= x
        if t_.numel() != need:
            t_ = t_[:need]
        return t_.view(shape_)

    # ── 发射（blockDim = max(phaseCores, sortActive)；SC 变体全核冗余）──
    # phaseCores 按 N·sNum 口径（= ASC det PostTiling phase3Cores）
    # launch/active_cores 由 @host 闭式派生（tile/is_single_core 为变体常量）
    # SC 变体 sort tile = 单核容量（= scatter_add 同款：MC tiling tile 会使
    # SC 缓冲贴满 UB）
    prog = build_det_program(
        val_dt, idx_dt, key_dt, is_single_core, counter8, tile_for_sort, rounds
    )
    # SIMT 寻址段独立张量（ws 视图 offset 丢失导致散射写飞）
    kdt = getattr(torch, key_dt)
    lin = torch.zeros(n_idx, dtype=kdt, device="npu")
    sorted_ = torch.zeros(n_idx, dtype=kdt, device="npu")
    perm = torch.zeros(n_idx, dtype=torch.int32, device="npu")
    db_v = torch.zeros(n_idx, dtype=kdt, device="npu")
    db_i = torch.zeros(n_idx, dtype=torch.int32, device="npu")
    prog(
        var_s,
        idx_s,
        upd_s,
        lin,
        sorted_,
        perm,
        seg("x_b8"),
        seg("hist_tile"),
        seg("hist_cumsum"),
        seg("excl_bins"),
        seg("global_hist"),
        db_v,
        db_i,
    )
    return var
