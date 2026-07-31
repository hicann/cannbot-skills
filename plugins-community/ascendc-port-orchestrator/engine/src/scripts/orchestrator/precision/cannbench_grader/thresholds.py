#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
精度阈值配置

职责：
1. 管理各 dtype 的精度阈值表
2. 提供阈值查询函数（按 dtype，大小写不敏感，未知 dtype 回退 float32）

从 utils/precision.py 拆分出来，独立管理阈值数据。
"""

from typing import Dict


# 精度阈值表（采用生态算子开源精度标准）
PRECISION_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-10,      # ≈ 0.000976
    'bfloat16': 2**-7,      # ≈ 0.007812
    'float32': 2**-13,      # ≈ 0.000122
    'float64': 2**-13,      # 使用float32阈值
    'hifloat32': 2**-11,    # ≈ 0.000488
    # F083: 删除死代码 `float8_e4m3` 键 — PyTorch 实际 dtype 是 float8_e4m3fn
    # 字典查找用 str(t.dtype) 仅会命中 'float8_e4m3fn'，旧键永不触发
    'float8_e4m3fn': 2**-3,  # ≈ 0.125
    'float8_e5m2': 2**-2,   # ≈ 0.25
    'int8': 0,              # 完全相等
    'int16': 0,
    'int32': 0,
    'int64': 0,
    'uint8': 0,
    'uint16': 0,
    'uint32': 0,
    'uint64': 0,
}

# 小值域阈值表（来自 docs/kernel_bench_design_v1.0.md）
# 当 |golden| < small_value_threshold 时，采用小值域标准
SMALL_VALUE_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-11,      # ≈ 4.88e-4
    'bfloat16': 2**-8,      # ≈ 3.91e-3
    'float32': 2**-14,      # ≈ 6.10e-5
    'float64': 2**-14,      # 使用float32阈值
    'hifloat32': 2**-12,    # ≈ 2.44e-4
    'float8_e4m3fn': 2**-4,  # ≈ 0.0625 — F083 删 `float8_e4m3` 死键
    'float8_e5m2': 2**-3,   # ≈ 0.125
}

# 小值域误差阈值表（来自 docs/kernel_bench_design_v1.0.md）
# 当 |golden| < small_value_threshold 且 |actual - golden| > small_value_error 时，计入 ErrorCount
SMALL_VALUE_ERROR_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-16,      # ≈ 1.53e-5
    'bfloat16': 2**-16,     # ≈ 1.53e-5
    'float32': 2**-30,      # ≈ 9.31e-10
    'float64': 2**-30,      # 使用float32阈值
    'hifloat32': 2**-28,    # ≈ 3.73e-9
    'float8_e4m3fn': 2**-6,  # ≈ 1.56e-2 — F083 删 `float8_e4m3` 死键
    'float8_e5m2': 2**-5,   # ≈ 3.12e-2
}

# ============================================================================
# 相消精度边界阈值表（基于 IEEE 754 精度位数理论）
# ============================================================================
#
# 理论依据：
# 1. IEEE 754 标准：不同 dtype 的尾数位数决定了有效数字范围
#    - FP32: 23 位尾数，相对精度 ~2^-23 ≈ 10^-7，约 7 位有效数字
#    - FP16: 10 位尾数，相对精度 ~2^-10 ≈ 10^-3，约 3 位有效数字
#    - BF16: 7 位尾数，相对精度 ~2^-7 ≈ 10^-2，约 2 位有效数字
#
# 2. Kahan 灾难性相消理论：
#    当两个接近的大数相减时，结果的有效位数急剧丢失。
#    例如：FP32 中两个 ~10^4 的数相减得到 ~10^-3，但精度只够表示 7 位，
#    结果相对于原操作数丢失精度，可能输出为 0。
#
# 3. 相消判定条件：
#    - output ≈ 0：因相消丢失精度，结果接近零
#    - golden 在精度边界附近：非零小值，但小于 dtype 能可靠表示的范围
#    - 不在小值域内（排除极小值）
#
# 阈值选择原则：
#    cancel_boundary 应覆盖因精度位数丢失可能导致相消的范围。
#    对于 FP32，当操作数规模 ~10^4 时，结果 ~10^-3 可能相消丢失。
#    设置 cancel_boundary = 2^-8 ≈ 0.004，覆盖常见相消场景。
#
CANCEL_BOUNDARY_THRESHOLDS: Dict[str, float] = {
    # FP32: 精度 ~7 位，设置 2^-8 ≈ 0.004
    # 当 golden < 0.004 且 output ≈ 0 时，可能是 FP32 相消导致
    'float32': 2**-8,       # ≈ 3.91e-3 ≈ 0.004
    'float64': 2**-8,       # 使用float32阈值

    # FP16: 精度 ~3 位，设置 2^-5 ≈ 0.031
    # 当 golden < 0.031 且 output ≈ 0 时，可能是 FP16 相消导致
    'float16': 2**-5,       # ≈ 3.12e-2 ≈ 0.031

    # BF16: 精度 ~2 位，设置 2^-3 ≈ 0.125
    # 当 golden < 0.125 且 output ≈ 0 时，可能是 BF16 相消导致
    'bfloat16': 2**-3,      # ≈ 1.25e-1 ≈ 0.125

    'hifloat32': 2**-8,     # ≈ 3.91e-3
    'float8_e4m3fn': 2**-1,  # ≈ 0.5 — F083 删 `float8_e4m3` 死键
    'float8_e5m2': 2**-0,   # ≈ 1.0
}

# 相消 output 零值判定阈值
# 当 |output| < cancel_zero_threshold 时，判定 output ≈ 0（因相消丢失精度）
CANCEL_ZERO_THRESHOLDS: Dict[str, float] = {
    # 与 cancel_boundary 一致，确保 output 接近零时判定为相消
    'float32': 2**-8,       # ≈ 0.004
    'float64': 2**-8,
    'float16': 2**-5,       # ≈ 0.031
    'bfloat16': 2**-3,      # ≈ 0.125
    'hifloat32': 2**-8,
    'float8_e4m3fn': 2**-1,  # F083 删 `float8_e4m3` 死键
    'float8_e5m2': 2**-0,
}


def get_threshold(dtype_str: str) -> float:
    """获取精度阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in PRECISION_THRESHOLDS:
        return PRECISION_THRESHOLDS['float32']
    return PRECISION_THRESHOLDS[dtype_lower]


def get_small_value_threshold(dtype_str: str) -> float:
    """获取小值域阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in SMALL_VALUE_THRESHOLDS:
        return SMALL_VALUE_THRESHOLDS['float32']
    return SMALL_VALUE_THRESHOLDS[dtype_lower]


def get_small_value_error(dtype_str: str) -> float:
    """获取小值域误差阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in SMALL_VALUE_ERROR_THRESHOLDS:
        return SMALL_VALUE_ERROR_THRESHOLDS['float32']
    return SMALL_VALUE_ERROR_THRESHOLDS[dtype_lower]


def get_cancel_boundary(dtype_str: str) -> float:
    """
    获取相消精度边界阈值（基于 IEEE 754 精度位数理论）

    当 |golden| < cancel_boundary 且 |output| ≈ 0 时，判定为潜在相消位置。

    理论依据：
    - IEEE 754 尾数位数决定了有效数字范围
    - Kahan 灾难性相消理论：接近大数相减导致精度丢失
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower not in CANCEL_BOUNDARY_THRESHOLDS:
        return CANCEL_BOUNDARY_THRESHOLDS['float32']
    return CANCEL_BOUNDARY_THRESHOLDS[dtype_lower]


def get_cancel_zero_threshold(dtype_str: str) -> float:
    """
    获取相消 output 零值判定阈值

    当 |output| < cancel_zero_threshold 时，判定 output ≈ 0（因相消丢失精度）。
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower not in CANCEL_ZERO_THRESHOLDS:
        return CANCEL_ZERO_THRESHOLDS['float32']
    return CANCEL_ZERO_THRESHOLDS[dtype_lower]
