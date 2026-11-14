# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""custom_index_select（DSL 版）——signed 归一 + @host shape 派生 + 编排。

torch.index_select 语义。架构：
  - Python wrapper：校验 + signed 归一（view） + pack + 输出分配
    → 结构化 reshape (outer, gds, inner) / (outer, m, inner)，零标量传参
  - @host：从 Tensor shape 派生 inner/m/gds/total（cannir.extract_shape
    运行期值）→ 运行期 blocks → 显式维度 view 展平 → 启动 kernel
  - @kernel：标量参数直用（inner/m/gds/total），三因子坐标分解（零改动）

帧契约 3 槽（张量×3，无标量）：@host 派生值与 use64 同域（i64 / 窄化 i32）。
动态 shape 下 view 必须用显式运行期维度（view(-1) 不支持）。
所有 dtype 按字节宽度归一为 signed 整型（bisheng unsigned 常量 bug 绕开）。
越界零写 = dt(0) 编译期常量（zero_t 删除）。
"""

import torch
import torch_npu  # noqa: F401

import cannbotdsl
from cannbotdsl import Dim, Tensor, TensorSpec, dtypes
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops.info import get_platform_info

THREAD_NUM = 2048
PER_CORE_MIN = 1024
MAX_RANK = 8
INT32_MAX = 2**31 - 1
USE64_MARGIN = 2 * 64 * 1024

_SIGNED_BY_BYTES = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_DSL_SIGNED = {1: dtypes.int8, 2: dtypes.int16, 4: dtypes.int32, 8: dtypes.int64}

_INDEX_DTYPES = (torch.int32, torch.int64)


def _pack_param(inner_size, elem_bytes, x_ptr):
    """W 递降择优（SIMT 对齐硬约束 + inner 整除；out 由 wrapper 分配恒对齐）。"""
    w = 8 // elem_bytes
    while w > 1:
        pb = w * elem_bytes
        if inner_size % w == 0 and x_ptr % pb == 0:
            return w
        w //= 2
    return 1


@kernel
class IndexSelectKernel:
    """SIMT index_select：grid-stride + 三因子坐标分解。

    标量 TensorSpec 随 use64 走 int32/int64 → range 参数同域 → 循环变量同域 →
    坐标分解算术全链自然同域（无锚定、无 cast）。越界检查在索引原值域（铁律）。
    """

    def __init__(self, use64, data_dt):
        self._use64 = bool(use64)
        self._data_dt = data_dt

    def __call__(
        self,
        gm_x: Tensor,
        gm_index: Tensor,
        gm_y: Tensor,
        inner_size,
        gather_size,
        gds,
        total,
    ):
        with vf(mode="simt", thread=THREAD_NUM):
            t = simt.thread_idx()[0]
            tdim = simt.thread_dim()[0]
            stride = simt.grid_dim()[0] * tdim
            tid = simt.block_idx()[0] * tdim + t
            if const_expr(self._use64):
                tid = simt.cast(tid, dtype=dtypes.int64)
                stride = simt.cast(stride, dtype=dtypes.int64)
            # 标量参数与 tid/stride 同域 → 循环变量同域 → 全链算术同域
            for i in cannbotdsl.range(tid, total, stride):
                q = i // inner_size
                outer_idx = q // gather_size
                j = q - outer_idx * gather_size
                inner_off = i - q * inner_size
                k = gm_index[j]
                # 越界检查在索引原值域（铁律），先于一切变换
                if (k < 0) or (k >= gds):
                    gm_y[i] = self._data_dt(0)
                else:
                    src = (outer_idx * gds + k) * inner_size + inner_off
                    gm_y[i] = gm_x[src]


class IndexSelectProgram:
    """动态程序：@host 内计算 total/blocks，标量直传 kernel（官方模式）。"""

    def __init__(self, use64, data_dt, data_dtype, idx_dtype):
        self._use64 = bool(use64)
        self._data_dt = data_dt
        self._dt = data_dtype
        self._idx_dt = idx_dtype
        self._program = None

    @property
    def program(self):
        if self._program is None:
            self._program = self._build()
        return self._program

    def _build(self):
        # 结构 shape：(outer, gds, inner) / (m,) / (outer, m, inner)
        # 尺寸全由 @host 从 shape 派生（运行期值），帧契约零标量
        nd_outer = Dim("N_OUTER", min=1)
        nd_gds = Dim("N_GDS", min=1)
        nd_inner = Dim("N_INNER", min=1)
        nd_m = Dim("N_M", min=1)
        return cannbotdsl.compile(
            self.run,
            TensorSpec((nd_outer, nd_gds, nd_inner), self._dt),
            TensorSpec((nd_m,), self._idx_dt),
            TensorSpec((nd_outer, nd_m, nd_inner), self._dt),
        )

    @host
    def run(self, gm_x, gm_index, gm_y):
        """@host：从 Tensor shape 派生尺寸（运行期值）→ blocks → 启动 kernel。"""
        outer = gm_x.shape[0]
        gds = gm_x.shape[1]
        inner = gm_x.shape[2]
        m = gm_y.shape[1]
        total = outer * m * inner
        if const_expr(not self._use64):
            # int32 真快速路径（spec 无标量，域由 shape 派生值窄化决定）
            inner = dtypes.int32(inner)
            m = dtypes.int32(m)
            gds = dtypes.int32(gds)
            total = dtypes.int32(total)
        cores = get_platform_info().vector_core_num
        block_factor = (total + cores - 1) // cores
        if block_factor < PER_CORE_MIN:
            block_factor = PER_CORE_MIN
        blocks = (total + block_factor - 1) // block_factor
        if blocks < 1:
            blocks = 1
        if blocks > cores:
            blocks = cores
        # 显式运行期维度展平（动态 shape 下 view(-1) 不支持）
        x_flat = gm_x.view(outer * gds * inner)
        y_flat = gm_y.view(total)
        IndexSelectKernel(self._use64, self._data_dt)[blocks](
            x_flat, gm_index, y_flat, inner, m, gds, total
        )


_cache = {}


def _get_program(use64, data_dt, data_dtype, idx_dtype):
    key = (str(data_dtype), str(idx_dtype), use64)
    if key not in _cache:
        _cache[key] = IndexSelectProgram(use64, data_dt, data_dtype, idx_dtype)
    return _cache[key]


def custom_index_select(
    x: torch.Tensor, index: torch.Tensor, dim: int = 0, *, _use64=None
) -> torch.Tensor:
    """按 ``dim`` 轴索引选择，对标 ``torch.index_select``。

    所有 dtype 按字节宽度归一为 signed 整型进出 kernel（位保真）。
    视图 pack：inner 可整除且对齐时按 W 合轴打包。
    use64=False 时全链 int32 算术（标量 TensorSpec 也为 int32）。
    """
    if x.device.type != "npu" or index.device.type != "npu":
        raise ValueError("custom_index_select requires NPU x/index tensors")
    if index.device != x.device:
        raise ValueError("x and index must share the same NPU device")

    orig_dtype = x.dtype
    elem_bytes = x.element_size()
    if elem_bytes not in _SIGNED_BY_BYTES:
        raise TypeError(f"unsupported x dtype {orig_dtype} (elem_bytes={elem_bytes})")
    if index.dtype not in _INDEX_DTYPES:
        raise TypeError(f"index dtype must be int32/int64, got {index.dtype}")

    rank = x.dim()
    if rank > MAX_RANK:
        raise ValueError(f"x rank must be <= {MAX_RANK}, got {rank}")
    if index.dim() > 1:
        raise ValueError(f"index dim must be <= 1, got {index.dim()}")
    if not x.is_contiguous() or not index.is_contiguous():
        raise ValueError("x and index must be contiguous")

    norm_rank = rank if rank > 0 else 1
    dim = int(dim)
    if dim < 0:
        dim += norm_rank
    if not 0 <= dim < norm_rank:
        raise ValueError(f"dim out of range [-{norm_rank}, {norm_rank})")

    m = index.numel()
    if rank == 0 and m > 1:
        raise ValueError(f"index to scalar can have only 1 value, got {m}")

    # --- 元信息 ---
    shape = list(x.shape)
    outer = 1
    for d in shape[:dim]:
        outer *= d
    inner = 1
    after_dim = dim + 1
    for d in shape[after_dim:]:
        inner *= d
    gather_dim_size = shape[dim] if rank > 0 else 1
    x_numel = x.numel() if rank > 0 else 1
    out_shape = () if rank == 0 else tuple(shape[:dim] + [m] + shape[after_dim:])

    # --- signed 归一（位重解释，零拷贝）---
    work_dtype = _SIGNED_BY_BYTES[elem_bytes]
    x_flat = x.reshape(-1).view(work_dtype)
    out_flat = torch.empty(outer * m * inner, dtype=work_dtype, device=x.device)
    total = outer * m * inner
    if total == 0:
        return out_flat.view(orig_dtype).reshape(out_shape)

    # --- 视图 pack（signed 宽度 → signed 宽度）---
    w = _pack_param(inner, elem_bytes, x_flat.data_ptr())
    if w > 1:
        packed_bytes = w * elem_bytes
        packed = _SIGNED_BY_BYTES[packed_bytes]
        x_flat = x_flat.view(packed)
        out_flat = out_flat.view(packed)
        inner //= w
        total = outer * m * inner
        elem_bytes = packed_bytes

    # use64 判定：仅规模驱动
    if _use64 is None:
        use64 = (total > INT32_MAX - USE64_MARGIN) or (
            x_numel > INT32_MAX - USE64_MARGIN
        )
    else:
        use64 = bool(_use64)

    idx_dt = dtypes.int32 if index.dtype is torch.int32 else dtypes.int64
    data_dt = _DSL_SIGNED[elem_bytes]

    # 结构化 reshape（各维 >=1，空输入已早退）——尺寸由 @host 从 shape 派生
    x_r = x_flat.view(outer, gather_dim_size, inner)
    out_r = out_flat.view(outer, m, inner)
    launcher = _get_program(use64, data_dt, _DSL_SIGNED[elem_bytes], idx_dt)
    launcher.program(x_r, index.reshape(-1), out_r)

    return out_flat.view(orig_dtype).reshape(out_shape)
