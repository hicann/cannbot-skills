#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except
# in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of
# the License.
# ----------------------------------------------------------------------------
"""
精度验证脚本模板（强制使用）

规则：
1. Developer 必须基于本模板生成 verify_result.py
2. 只允许修改 === 可修改区域 === 内的内容
3. 精度判定逻辑、阈值、输出格式禁止修改
4. 通过条件：两套标准（MERE/MARE 和 atol/rtol/error_ratio）
   通过任一即 PASS

阈值来源：ops-precision-standard/reference/float_compute_community.md
         ops-precision-standard/reference/quantization_community.md
"""

import logging
import os
import sys

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ============================================================
# 精度阈值（禁止修改）
# 来源: ops-precision-standard 决策树
# ============================================================

DTYPE_THRESHOLDS = {
    "float16": 2 ** (-10),   # ~9.766e-4
    "bfloat16": 2 ** (-7),   # ~7.813e-3
    "float32": 2 ** (-13),   # ~1.221e-4
    "float64": 2 ** (-13),
    "float8_e4m3": 2 ** (-3),
    "float8_e5m2": 2 ** (-2),
}

ATOL = 1e-3
RTOL = 1e-3
ERROR_RATIO_THRESHOLD = 1e-3


# ============================================================
# 精度计算函数（禁止修改）
# ============================================================

def calculate_mere(actual, golden):
    """Mean Element-wise Relative Error."""
    actual = actual.astype(np.float32).flatten()
    golden = golden.astype(np.float32).flatten()
    relative_errors = np.abs(actual - golden) / (np.abs(golden) + 1e-7)
    return float(np.mean(relative_errors))


def calculate_mare(actual, golden):
    """Max Absolute Relative Error."""
    actual = actual.astype(np.float32).flatten()
    golden = golden.astype(np.float32).flatten()
    relative_errors = np.abs(actual - golden) / (np.abs(golden) + 1e-7)
    return float(np.max(relative_errors))


def check_mere_mare(actual, golden, output_dtype="float16"):
    """Criterion 1: MERE/MARE Threshold.

    Pass condition: MERE < Threshold AND MARE < 10 * Threshold.
    """
    threshold = DTYPE_THRESHOLDS.get(output_dtype, 2 ** (-10))
    mare_threshold = 10 * threshold
    mere = calculate_mere(actual, golden)
    mare = calculate_mare(actual, golden)
    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold
    return {
        "pass": mere_pass and mare_pass,
        "mere": mere,
        "mare": mare,
        "threshold": threshold,
        "mare_threshold": mare_threshold,
        "mere_pass": mere_pass,
        "mare_pass": mare_pass,
    }


def check_error_ratio(actual, golden):
    """Criterion 2: atol/rtol/error_ratio.

    Pass condition: error_ratio <= ERROR_RATIO_THRESHOLD.
    """
    actual_f = actual.astype(np.float32).flatten()
    golden_f = golden.astype(np.float32).flatten()
    abs_diff = np.abs(actual_f - golden_f)
    tolerance = ATOL + RTOL * np.abs(golden_f)
    error_count = int(np.sum(abs_diff > tolerance))
    total = len(actual_f)
    error_ratio = error_count / total
    return {
        "pass": error_ratio <= ERROR_RATIO_THRESHOLD,
        "error_count": error_count,
        "total": total,
        "error_ratio": error_ratio,
    }


# ============================================================
# 验证主函数（禁止修改判定逻辑和输出格式）
# ============================================================

def verify(output_data, golden_data, output_dtype="float16"):
    """Run both criteria. PASS if either passes.

    Returns (passed: bool, report: str).
    """
    if np.any(np.isnan(output_data)):
        return False, "FAIL: Output contains NaN"
    if np.any(np.isinf(output_data)):
        return False, "FAIL: Output contains Inf"

    lines = []
    lines.append(f"Shape: {output_data.shape}, dtype: {output_dtype}")

    c1 = check_mere_mare(output_data, golden_data, output_dtype)
    lines.append("")
    lines.append("--- Criterion 1: MERE/MARE Threshold ---")
    lines.append(f"  Threshold ({output_dtype}): {c1['threshold']:.6e}")
    p1 = 'PASS' if c1['mere_pass'] else 'FAIL'
    lines.append(f"  MERE: {c1['mere']:.6e} ({p1}, < {c1['threshold']:.6e})")
    p2 = 'PASS' if c1['mare_pass'] else 'FAIL'
    lines.append(f"  MARE: {c1['mare']:.6e} ({p2}, < {c1['mare_threshold']:.6e})")
    lines.append(f"  Criterion 1: {'PASS' if c1['pass'] else 'FAIL'}")

    c2 = check_error_ratio(output_data, golden_data)
    lines.append("")
    lines.append("--- Criterion 2: atol/rtol/error_ratio ---")
    t = f"atol={ATOL}, rtol={RTOL}, threshold={ERROR_RATIO_THRESHOLD}"
    lines.append(f"  {t}")
    lines.append(f"  Error count: {c2['error_count']} / {c2['total']}")
    p3 = 'PASS' if c2['pass'] else 'FAIL'
    t2 = f"<= {ERROR_RATIO_THRESHOLD}"
    lines.append(f"  Error ratio: {c2['error_ratio']:.6e} ({p3}, {t2})")
    lines.append(f"  Criterion 2: {'PASS' if c2['pass'] else 'FAIL'}")

    abs_diff = np.abs(
        output_data.astype(np.float32) - golden_data.astype(np.float32)
    )
    lines.append("")
    lines.append("--- Additional metrics ---")
    lines.append(f"  Max absolute error: {abs_diff.max():.6e}")
    lines.append(f"  Mean absolute error: {abs_diff.mean():.6e}")

    passed = c1["pass"] or c2["pass"]
    criteria_passed = []
    if c1["pass"]:
        criteria_passed.append("Criterion 1 (MERE/MARE)")
    if c2["pass"]:
        criteria_passed.append("Criterion 2 (error_ratio)")

    lines.append("")
    if passed:
        lines.append(f"RESULT: PASS (via {', '.join(criteria_passed)})")
    else:
        lines.append("RESULT: FAIL")

    report = "\n".join(lines)
    return passed, report


# ============================================================
# === 可修改区域 === CLI 入口与数据加载
# Developer 只允许修改以下内容：
# 1. output_shape 的计算逻辑（如 SwiGLU 输出 [m, n // 2]）
# 2. output_dtype（如 bf16 算子改为 "bfloat16"）
# 3. 文件路径/命名（如有特殊需求）
# 4. CLI 参数解析（如需要额外参数）
# ============================================================

def main():
    if len(sys.argv) < 5:
        log.info("Usage: %s M N K data_dir", sys.argv[0])
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    data_dir = sys.argv[4]

    # --- Developer 可修改：输出 shape 和 dtype ---
    output_shape = (m, n)          # 例: SwiGLU 改为 (m, n // 2)
    output_dtype = "float16"       # 例: bf16 算子改为 "bfloat16"
    # --- 可修改区域结束 ---

    out_path = os.path.join(data_dir, "out.bin")
    golden_path = os.path.join(data_dir, "golden.bin")

    if not os.path.exists(out_path):
        log.info("FAIL: Output file not found: %s", out_path)
        sys.exit(1)
    if not os.path.exists(golden_path):
        log.info("FAIL: Golden file not found: %s", golden_path)
        sys.exit(1)

    try:
        if output_dtype == "bfloat16":
            if ml_dtypes is None:
                log.info("FAIL: bfloat16 requires ml_dtypes.")
                log.info("Install: pip install ml-dtypes")
                sys.exit(1)
            np_dtype = np.dtype(ml_dtypes.bfloat16)
        elif output_dtype == "float8_e4m3":
            if ml_dtypes is None:
                log.info("FAIL: float8_e4m3 requires ml_dtypes.")
                log.info("Install: pip install ml-dtypes")
                sys.exit(1)
            np_dtype = np.dtype(ml_dtypes.float8_e4m3fn)
        elif output_dtype == "float8_e5m2":
            if ml_dtypes is None:
                log.info("FAIL: float8_e5m2 requires ml_dtypes.")
                log.info("Install: pip install ml-dtypes")
                sys.exit(1)
            np_dtype = np.dtype(ml_dtypes.float8_e5m2)
        else:
            np_dtype = np.dtype(output_dtype)
    except TypeError as e:
        log.info("FAIL: dtype '%s' not supported: %s", output_dtype, e)
        sys.exit(1)
    output_data = np.fromfile(out_path, dtype=np_dtype).reshape(output_shape)
    golden_data = np.fromfile(golden_path, dtype=np_dtype).reshape(output_shape)

    passed, report = verify(output_data, golden_data, output_dtype)
    log.info(report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
