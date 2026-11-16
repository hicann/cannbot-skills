# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""tensor_scatter_nd_update DSL 版（det + nondet，ASC v2.0 同构，全动态 shape）。

ASC 源：ops-design-paradigms/references/paradigms/scatterupdate 基例
examples-tensor-scatter-nd-update-code（det 三段式段尾覆盖 + nondet 直写覆盖，
MDE §11 v2.0）。
接口（inplace）：tensor_scatter_nd_update(var, indices, updates) → var（入口见 tensor_scatter_nd_update.py）
  var: bf16/bool/f16/f32/i64/i8/u8/f8e5m2/f8e4m3fn（= ASC 10 dtype 减
  hifloat8——torch 接口不可表达该 dtype，记为接口约束偏差）；
  indices: i32/i64；updates 同 var dtype。

det（确定性，= ASC det_simt 三段式，MDE §11.5）：
  Phase1 SIMT(1024) grid-stride：pos → indices[pos·K+d] 直读（nd 形态无除法
    反解）→ lin = Σ idx[d]·varStrides[d]；越界/负索引（j<0 或 j≥varDims[d]）
    → 哨兵 KeyT max（= ASC SENTINEL，skip 语义；**无 err flag**——与
    scatter_add 的 error 形态形成对照）
  Phase2 内嵌排序（自包含排序库 tsnu_sort）：lin 升序 →
    sorted + perm(i32)，哨兵沉底；多核 = 超发安全（active_cores 运行期）；
    单核 = 全核冗余
  R3-E：var strides/dims 由 kernel VF 外按 8 维预留静态读取（= ASC 定长数组），
    维度循环 8 块展开 + 运行时卫（= `for i in range(8): if i >= rank: continue`）；
    结构化 8 维预留契约（1-padding），n_idx/k_dim/s_num 由 @host 派生。
  Phase3 SIMT(1024) grid-stride：task = k·sNum + s（元素级，DSL 无 Pack 视图
    ——ASC W∈{1,2,4,8} 打包的 W=1 通用兜底形态）；**哨兵前置跳过 + 段尾判据**
    （k==N-1 || sorted[k+1]!=key，O(1) 无扫描）→ 段尾任务直写自己：
    gm_var[key·sNum+s] = gm_upd[perm[k]·sNum+s]（last-write-wins，无原子；
    var 终态写 = tracked 普通写，键/段判读 = barrier 稳定普通读——对表
    volatile 清单）

nondet（非确定，= ASC nondet_simt）：task 反解 (k,s) + 内联 nd 线性化 →
  直接写覆盖（无原子无排序无屏障，多写者任意赢；**无 dtype 白名单**——覆盖
  语义仅需对齐单条 store 原子，全 dtype 安全，MDE §11.6.4）

分派（= ASC v2.0 双模板 IsCapable 同构）：
  torch.are_deterministic_algorithms_enabled() → det；否则 nondet。
  （ASC v2.0 实现即唯一确定性开关门控；dupRatio 阈值待 perf 标定，不发明数值）

KeyT（= ASC keyDtype 判定，**域 = totalVarElems**——nd 无 srcPos，区别于
  scatter_add 的 max(var,src)）：≤32767→int16；≤2^31-1→int32；否则 int64。
  哨兵 = KeyT 最大值（32767 / 2^31-1 / 2^63-1）。
"""

import os
import sys

# 自包含：本目录入 path（文件夹可整体移动；平名导入防连字符目录名）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cannbotdsl
from cannbotdsl import Dim, TensorSpec, dtypes
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.info import get_platform_info

from tsnu_sort.sort_kernel import (
    SortCfg,
    sort_multicore_phase,
    sort_singlecore_phase,
    _barrier,
)

THREAD_NUM = 1024  # = ASC THREAD_NUM
MAXD = 8  # 8 维预留（= ASC shape/stride 定长数组）

# ASC 10 dtype 减 hifloat8（torch 接口不可表达）
_VAL_DTYPES = (
    "float32",
    "float16",
    "bfloat16",
    "int64",
    "int8",
    "uint8",
    "bool",
    "float8_e5m2",
    "float8_e4m3fn",
)
_IDX_DTYPES = ("int32", "int64")

# 哨兵（= ASC SENTINEL：KeyT 最大值；bounds §4——键值域最大值保证哨兵恒空闲）
_SENTINEL_OF = {"int16": 32767, "int32": 2147483647, "int64": 9223372036854775807}


def key_dtype_of(total_var_elems):
    """= ASC keyDtype 判定（域 = totalVarElems，nd 无 srcPos）。"""
    if total_var_elems <= 32767:
        return "int16"
    if total_var_elems <= 2**31 - 1:
        return "int32"
    return "int64"


def _sentinel(key_dt):
    """KeyT 最大值常量（哨兵）。"""
    return getattr(dtypes, key_dt)(_SENTINEL_OF[key_dt])


# ══════════════════ Phase1：nd 线性化（SIMT grid-stride，任务域 N）══════════════════


@jit
def _phase1_linearize_nd(
    var_st, var_dm, gm_index, gm_lin, n_idx, k_dim, pos_base, step, idx64, key_dt
):
    """pos → indices[pos·K+d] 直读（末维 K 连续，无除法）→ lin = Σ j·varStrides[d]。

    越界/负索引（j<0 或 j≥varDims[d]）→ 存哨兵 KeyT max（skip 语义，与
    nondet 直跳一致——双路径越界语义一致约束）；valid 存 lin（cast KeyT 收窄，
    = ASC static_cast<KeyT>；totalVarElems 判定保证无截断）。
    跨核消费（Phase2 排序）→ l2cache_store。

    R3-E：var 的 strides/dims 由 kernel VF 外按 8 维预留静态读取后传入
    （= ASC 定长数组）；维度循环 8 块展开 + 运行时卫 `d < k_dim`
    （= `for i in range(8): if i >= rank: continue`）。

    边界三件套全部 i64：@host 侧 n_idx/k_dim/s_num 已用 dtypes.int64 显式
    构造（脱离 shape dim 溯源），0.8.3 支持 SIMT VF i64 归纳变量
    （issue #79 已修），VF 内不再生成 i32 窄化。
    """
    with vf(mode="simt", thread=THREAD_NUM):
        t = simt.thread_idx()[0]
        # 三参 grid-stride 铁律（language-rules §3：手动步进阻止编译器判界与
        # 优化，while+手动步进为禁止形态）；start/step 为 kernel 级预计算
        # （K68 规避：线程无关子式跨 vf 域重复会被 CSE 跨域合并 → 崩）
        t64 = simt.cast(t, dtype=dtypes.int64)
        for pos in cannbotdsl.range(pos_base + t64, n_idx, step):
            base = pos * k_dim
            lin = dtypes.int64(0)
            valid = 1
            for d in cannbotdsl.range_constexpr(MAXD):
                if d < k_dim:
                    j = gm_index[base + d]
                    j = j if const_expr(idx64) else simt.cast(j, dtype=dtypes.int64)
                    dim_size = var_dm[d]
                    if (j < 0) or (j >= dim_size):
                        valid = 0
                    else:
                        lin = lin + j * var_st[d]
            if valid == 1:
                simt.l2cache_store(
                    gm_lin.ptr(pos),
                    simt.cast(lin, dtype=getattr(dtypes, key_dt)),
                )
            else:
                simt.l2cache_store(gm_lin.ptr(pos), _sentinel(key_dt))


# ══════════════════ Phase3：段尾覆盖（SIMT grid-stride，任务域 N·sNum）══════════════════


@jit
def _phase3_scatter_update(
    gm_var, gm_updates, gm_sorted, gm_perm, n_idx, s_num, pos_base, step, key_dt
):
    """task = k·sNum + s 元素级段尾写（last-write-wins，无原子无扫描）。

    哨兵前置跳过（bounds §4——漏判会以哨兵值+偏移巨值越界写）→ 段尾判据
    O(1)：写条件 = k==N-1 || sorted[k+1]!=key（与下一索引不同或全局末位）。
    段尾 perm[k] = 段内原始行号最大（稳定排序保证）= 胜者；判据在 s 维冗余
    执行（O(1) 幂等无害）。var 终态写 = tracked 普通写（单写者 + host 唯一
    读者）；sorted/perm 读 = barrier 稳定普通读——对表 volatile 清单。
    """
    sent = _sentinel(key_dt)
    n_last = n_idx - 1
    with vf(mode="simt", thread=THREAD_NUM):
        t = simt.thread_idx()[0]
        n_total = n_idx * s_num
        # 三参 grid-stride 铁律（language-rules §3，同 Phase1）；
        # 边界同 Phase1 统一 i64（0.8.3 i64 归纳变量）
        t64 = simt.cast(t, dtype=dtypes.int64)
        for task in cannbotdsl.range(pos_base + t64, n_total, step):
            k = task // s_num
            s = task - k * s_num
            key = gm_sorted[k]
            is_write = 1
            if key == sent:
                is_write = 0
            else:
                if k < n_last:
                    if gm_sorted[k + 1] == key:
                        is_write = 0
            if is_write == 1:
                target = simt.cast(key, dtype=dtypes.int64)
                pk64 = simt.cast(gm_perm[k], dtype=dtypes.int64)
                # var 侧 = lin + s（lin 已是 var 空间元素偏移——slice 起点；
                # slice 连续占 [lin, lin+sNum)，= ASC varVec[key/W + sw] 的 W=1 形态）
                gm_var[target + s] = gm_updates[pk64 * s_num + s]


# ══════════════════ det 复合 kernel ═════════════════


@kernel
class TensorScatterNdUpdateDetKernel:
    """三阶段复合 kernel（det）。launch = max(phaseCores, sortActive)。

    is_single_core 为编译期程序变体（= ASC OneCore/TrySingleCore）：
    True → Phase2 全核冗余；False → 超发安全多核（active_cores 运行期）。
    """

    def __init__(
        self, key_dt, tile, counter8, idx64, is_single_core, spin_limit=2_000_000
    ):
        self.sort_cfg = SortCfg(key_dt, tile, counter8=counter8, spin_limit=spin_limit)
        self.key_dt = key_dt
        self.idx64 = bool(idx64)
        self.is_single_core = bool(is_single_core)

    def __call__(
        self,
        gm_var,
        gm_index,
        gm_updates,
        gm_lin,
        gm_sorted,
        gm_perm,
        ws_b8,
        ws_hist,
        ws_cumsum,
        ws_excl,
        ws_gh,
        db_v,
        db_i,
        n_idx,
        k_dim,
        s_num,
        active_cores,
    ):
        b = get_block_idx()
        nb = get_block_num()
        # K68 规避（U9 推广形态）：线程无关子式 kernel 级预计算
        b64 = b + dtypes.int64(0)
        pos_base = b64 * THREAD_NUM
        step = (nb + dtypes.int64(0)) * THREAD_NUM
        # ── Phase0（R3-E）：var metadata 8 维预留静态读取（VF 外；替代 params）──
        var_st = [gm_var.stride[d] for d in range(MAXD)]
        var_dm = [gm_var.shape[d] for d in range(MAXD)]
        n_var = var_dm[0]
        for d in cannbotdsl.range_constexpr(MAXD - 1):
            n_var = n_var * var_dm[d + 1]
        var_f = gm_var.view(n_var)
        idx_f = gm_index.view(gm_index.shape[0] * gm_index.shape[1])
        upd_f = gm_updates.view(gm_updates.shape[0] * s_num)
        # ── Phase1：nd 线性化（全核 grid-stride，任务域 N）──
        _phase1_linearize_nd(
            var_st,
            var_dm,
            idx_f,
            gm_lin,
            n_idx,
            k_dim,
            pos_base,
            step,
            self.idx64,
            self.key_dt,
        )
        _barrier()
        # ── Phase2：排序（升序 lin → sorted + perm；哨兵沉底）──
        if const_expr(self.is_single_core):
            # 全核冗余（= ASC OneCore：无守卫无屏障，幂等共写）
            sort_singlecore_phase(self.sort_cfg, gm_lin, gm_sorted, gm_perm, n_idx)
        else:
            sort_multicore_phase(
                self.sort_cfg,
                b,
                n_idx,
                active_cores,
                gm_lin,
                gm_sorted,
                gm_perm,
                ws_b8,
                ws_hist,
                ws_cumsum,
                ws_excl,
                ws_gh,
                db_v,
                db_i,
            )
        _barrier()
        # ── Phase3：段尾覆盖（全核 grid-stride，任务域 N·sNum 元素级）──
        _phase3_scatter_update(
            var_f, upd_f, gm_sorted, gm_perm, n_idx, s_num, pos_base, step, self.key_dt
        )


# ══════════════════ nondet kernel（直写覆盖，无排序无屏障）══════════════════


@kernel
class TensorScatterNdUpdateNondetKernel:
    """非确定路径：task 反解 (k,s) + 内联 nd 线性化 + 直接写覆盖。

    launch = min(cores, ceil(N·sNum/1024))（无内部屏障，无需超发）。
    越界/负索引跳过（与 det 哨兵语义一致）；无原子、无 dtype 白名单
    （对齐单条 store 天然防撕裂，全 dtype 安全）。
    """

    def __init__(self, idx64):
        self.idx64 = bool(idx64)

    def __call__(self, gm_var, gm_index, gm_updates, n_idx, k_dim, s_num):
        b = get_block_idx()
        nb = get_block_num()
        # R3-E：var metadata 8 维预留静态读取（VF 外；替代 params）
        var_st = [gm_var.stride[d] for d in range(MAXD)]
        var_dm = [gm_var.shape[d] for d in range(MAXD)]
        n_var = var_dm[0]
        for d in cannbotdsl.range_constexpr(MAXD - 1):
            n_var = n_var * var_dm[d + 1]
        var_f = gm_var.view(n_var)
        idx_f = gm_index.view(gm_index.shape[0] * gm_index.shape[1])
        upd_f = gm_updates.view(gm_updates.shape[0] * s_num)
        with vf(mode="simt", thread=THREAD_NUM):
            t = simt.thread_idx()[0]
            n_total = n_idx * s_num
            # 三参 grid-stride 铁律（language-rules §3）；单 vf 域内联预计算
            # start/step（同 scatter_add nondet 形态，无跨域 K68 顾虑）
            task_base = (b + dtypes.int64(0)) * THREAD_NUM + t
            step = (nb + dtypes.int64(0)) * THREAD_NUM
            for task in cannbotdsl.range(task_base, n_total, step):
                k = task // s_num
                s = task - k * s_num
                base = k * k_dim
                lin = dtypes.int64(0)
                valid = 1
                for d in cannbotdsl.range_constexpr(MAXD):
                    if d < k_dim:
                        j = idx_f[base + d]
                        j = (
                            j
                            if const_expr(self.idx64)
                            else simt.cast(j, dtype=dtypes.int64)
                        )
                        dim_size = var_dm[d]
                        if (j < 0) or (j >= dim_size):
                            valid = 0
                        else:
                            lin = lin + j * var_st[d]
                if valid == 1:
                    # var 侧 = lin + s（slice 起点元素偏移 + slice 内偏移；
                    # = ASC NondetScatterUpdateKernelPack 的 key/W + sw，W=1）
                    target = lin + s
                    src_off = k * s_num + s
                    var_f[target] = upd_f[src_off]


# ══════════════════ 程序构建 ═════════════════

_DET_PROG_CACHE = {}
_NONDET_PROG_CACHE = {}


def build_det_program(
    val_dt, idx_dt, key_dt, is_single_core, counter8, tile, rounds, spin_limit=2_000_000
):
    """det 程序（编译期变体：SC/MC × dtype 族 × KeyT；N/K/sNum 全动态）。

    编译键无 is_subset/dup_mode（nd 无 SUBSET；assign 无 warp 归约）——
    与 scatter_add 的变体差异点。
    """
    key = (val_dt, idx_dt, key_dt, bool(is_single_core), bool(counter8), tile)
    if key in _DET_PROG_CACHE:
        return _DET_PROG_CACHE[key]
    kern = TensorScatterNdUpdateDetKernel(
        key_dt, tile, counter8, idx_dt == "int64", is_single_core, spin_limit
    )
    vd = getattr(dtypes, val_dt)
    idt = getattr(dtypes, idx_dt)
    kdt = getattr(dtypes, key_dt)
    cdt = dtypes.int64 if counter8 else dtypes.int32

    @host
    def run(
        gm_var,
        gm_index,
        gm_updates,
        gm_lin,
        gm_sorted,
        gm_perm,
        ws_b8,
        ws_hist,
        ws_cumsum,
        ws_excl,
        ws_gh,
        db_v,
        db_i,
    ):
        """@host：shape 派生 n_idx/k_dim/s_num + 闭式 active/launch（R2/R3-E）。"""
        # shape 派生值显式构造为 i64 标量（脱离 dim 溯源），
        # 避免 SIMT VF 内 range 边界被按 i32 索引域窄化
        n_idx = dtypes.int64(gm_index.shape[0])
        k_dim = dtypes.int64(gm_index.shape[1])
        s_num = dtypes.int64(gm_updates.shape[1])
        for d in cannbotdsl.range_constexpr(MAXD - 2):
            s_num = s_num * gm_updates.shape[d + 2]
        s_num = dtypes.int64(s_num)
        cores = get_platform_info().vector_core_num
        task_total = n_idx * s_num
        phase_cores = (task_total + THREAD_NUM - 1) // THREAD_NUM
        if phase_cores < 1:
            phase_cores = 1
        if phase_cores > cores:
            phase_cores = cores
        if const_expr(is_single_core):
            active_cores = 1
            launch = phase_cores if phase_cores > 1 else 1
        else:
            tc = (n_idx + tile - 1) // tile
            active_cores = tc if tc < cores else cores
            launch = phase_cores if phase_cores > active_cores else active_cores
        kern[launch](
            gm_var,
            gm_index,
            gm_updates,
            gm_lin,
            gm_sorted,
            gm_perm,
            ws_b8,
            ws_hist,
            ws_cumsum,
            ws_excl,
            ws_gh,
            db_v,
            db_i,
            n_idx,
            k_dim,
            s_num,
            dtypes.int32(active_cores),
        )

    nd = Dim("N", min=1)  # 散点数（lin/sorted/perm 域）
    tc = (nd + tile - 1) // tile
    ts = TensorSpec
    pv = [Dim(f"V{d}", min=1) for d in range(MAXD)]  # var 8 维预留
    pn = [Dim(f"I{d}", min=1) for d in range(MAXD)]  # indices 8 维预留
    pu = [Dim(f"U{d}", min=1) for d in range(MAXD)]  # updates 8 维预留
    prog = cannbotdsl.compile(
        run,
        ts(tuple(pv), vd),  # var（结构 8 维预留，inplace）
        ts(tuple(pn), idt),  # indices（(N,K) 8 维预留）
        ts(tuple(pu), vd),  # updates（(N,sNum) 8 维预留）
        ts((nd,), kdt),
        ts((nd,), kdt),
        ts((nd,), dtypes.int32),  # lin/sorted/perm
        ts(
            (tc * tile,), dtypes.uint8
        ),  # x_b8（加宽 tc*TILE：规避 ub2gm 固定满 tile 回写越界）
        ts((tc, 256), dtypes.uint16),
        ts((tc, 256), dtypes.uint16),
        ts((rounds, 256), cdt),
        ts((rounds * tc, 256), cdt),
        ts((nd,), kdt),
        ts((nd,), dtypes.int32),  # db_v / db_i
    )
    _DET_PROG_CACHE[key] = prog
    return prog


def build_nondet_program(val_dt, idx_dt):
    key = (val_dt, idx_dt)
    if key in _NONDET_PROG_CACHE:
        return _NONDET_PROG_CACHE[key]
    kern = TensorScatterNdUpdateNondetKernel(idx_dt == "int64")
    vd = getattr(dtypes, val_dt)
    idt = getattr(dtypes, idx_dt)

    @host
    def run(gm_var, gm_index, gm_updates):
        """@host：shape 派生 n_idx/k_dim/s_num + 闭式 launch（R2/R3-E）。"""
        # shape 派生值显式构造为 i64 标量（脱离 dim 溯源）
        n_idx = dtypes.int64(gm_index.shape[0])
        k_dim = dtypes.int64(gm_index.shape[1])
        s_num = dtypes.int64(gm_updates.shape[1])
        for d in cannbotdsl.range_constexpr(MAXD - 2):
            s_num = s_num * gm_updates.shape[d + 2]
        s_num = dtypes.int64(s_num)
        cores = get_platform_info().vector_core_num
        launch = (n_idx * s_num + THREAD_NUM - 1) // THREAD_NUM
        if launch < 1:
            launch = 1
        if launch > cores:
            launch = cores
        kern[launch](gm_var, gm_index, gm_updates, n_idx, k_dim, s_num)

    ts = TensorSpec
    pv = [Dim(f"V{d}", min=1) for d in range(MAXD)]
    pn = [Dim(f"I{d}", min=1) for d in range(MAXD)]
    pu = [Dim(f"U{d}", min=1) for d in range(MAXD)]
    prog = cannbotdsl.compile(
        run,
        ts(tuple(pv), vd),  # var（结构 8 维预留，inplace）
        ts(tuple(pn), idt),  # indices（(N,K) 8 维预留）
        ts(tuple(pu), vd),  # updates（(N,sNum) 8 维预留）
    )
    _NONDET_PROG_CACHE[key] = prog
    return prog
