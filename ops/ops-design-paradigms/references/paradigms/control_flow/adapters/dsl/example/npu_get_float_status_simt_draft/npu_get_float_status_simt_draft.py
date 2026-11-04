# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference implementation DRAFT: FloatStatus-family signal kernels via SIMT/SIMD VF.

!!! DRAFT —— 未编译、未上板、未实测 !!!

ControlFlow 范式设备核路线的结构参考实现（对照 AscendC 侧
ops-nn/control/npu_*_float_status），展示三个形态：
  1. alloc/get：纯 SIMT kernel，幂等写规范零值（降级态 I5）
  2. clear：SIMD 副作用段 + SIMT 数据面段的**混合形态 kernel**
  3. 退化形态交付契约：常量 launch 几何 / 单编译实例

依据：signal-model.md（I3/I4/I5 不变量）· status-kernel.md §1-§5 ·
capability-mapping.md §1（降级态真值）· compute.md「两种 SIMT 编程模型」
"""

import cannbotdsl as cbd
from cannbotdsl import Tensor, dtypes, MemLoc, dim3
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.kernel import kernel, host
from cannbotdsl.ops.reg import full_mask, vdups, vstore

STATUS_ELEMS = 8  # [8]×fp32 固定契约（CAP_STATUS_LAYOUT）
STATUS_THREADS = 32  # 8 元素最小有效档
CLEAR_THREADS = 128
VEC_DUP_SIZE = 38400  # 副作用段缓冲（V1 行为对齐参数，逐平台重论证）
SIDE_EFFECT_DUPS = 6  # 同上


@kernel
def _alloc_zero_simt(output: Tensor):
    """alloc：分配清零（纯 SIMT，幂等写；框架保证 GM 写一致性）。"""
    with vf(mode="simt"):
        if cbd.simt.thread_idx()[0] < STATUS_ELEMS:
            output[cbd.simt.thread_idx()[0]] = 0.0


@kernel
def _get_zero_simt(data: Tensor, output: Tensor):
    """get：状态读取空实现——恒写规范零值（降级态 I5）。

    DSL API 面未暴露 FP 异常状态读取（capability-mapping §1）；
    能力补齐后的接入点：kernel 内取值传标量进 VF（对照 status-kernel.md §2）。
    """
    with vf(mode="simt"):
        for i in range(
            cbd.simt.thread_idx()[0], STATUS_ELEMS, cbd.simt.thread_dim()[0]
        ):
            output[i] = 0.0


@kernel
def _clear_side_effect(data: Tensor, output: Tensor, dup_size: cbd.Constexpr[int]):
    """clear：副作用段（SIMD 向量活动）+ 数据面段（SIMT 写零）——混合形态。

    同一 kernel 内含 SIMD VF 与 SIMT VF → host 以 kernel[block_count] 启动，
    SIMT VF 显式给 thread=N（compute.md 混合编程规则）。
    副作用通道无接口保证（V1 行为对齐，脆弱性声明见 status-kernel.md §3）；
    DSL 侧指令序列与 AscendC Duplicate 不同——结论不可平移，逐平台闭环
    （clear→get==0）。
    """
    buf = Buffer(MemLoc.UB, (dup_size,), dtypes.float16)
    with vf(mode="simd"):  # ① 副作用段：向量单元活动（值无意义）
        m = full_mask()
        for i in cbd.range_constexpr(SIDE_EFFECT_DUPS):
            vstore(buf, 0, vdups(3 + i, dtypes.float16, mask=m), m)
    with vf(mode="simt", thread=CLEAR_THREADS):  # ② 数据面段：写 [8] 零值
        for i in range(
            cbd.simt.thread_idx()[0], STATUS_ELEMS, cbd.simt.thread_dim()[0]
        ):
            output[i] = 0.0


@host
def run_alloc(output: Tensor) -> None:
    # 纯 SIMT：dim3(block), dim3(thread) 成对
    _alloc_zero_simt[dim3(1), dim3(STATUS_THREADS)](output)


@host
def run_get(data: Tensor, output: Tensor) -> None:
    _get_zero_simt[dim3(1), dim3(STATUS_THREADS)](data, output)


@host
def run_clear(data: Tensor, output: Tensor) -> None:
    # 混合形态：方括号内只给 block 数（thread 维在 SIMT VF 参数）
    _clear_side_effect[1](data, output, VEC_DUP_SIZE)
