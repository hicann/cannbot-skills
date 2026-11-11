#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""设备输出 vs golden 精度比对参考脚本

参考脚本（非通用工具）：演示"二进制结果文件 vs golden 精度比对"的判定与报告模式，
供整个 skill 在回传输出比对时参考。判定模式由用户通过 --mode 显式指定（不依赖 dtype）：

- bitwise  : 按位模式逐元素比较（适合 bool/整数等需全一致的类型）；浮点按 IEEE 位模式，
  NaN payload 不同或 +0/-0 差异会判为不一致，反转元素=0 即 PASS
- tolerance: 容差判定，不匹配元素 |a-b| > atol 且 |a-b| > rtol×|golden|；
  NaN 同位视为一致、Inf 需同号（rtol/atol 取值参考 ops-precision-standard 的 dtype 精度标准）
- 无论何种模式，输出字节数 != shape×dtype 字节数时直接报警终止

用法:
  python3 compare_output.py output.bin golden.bin --shape 128,512 --dtype uint8 --mode bitwise
  python3 compare_output.py output.bin golden.bin --shape 128,512 --dtype float16 \
      --mode tolerance --rtol 1e-3 --atol 1e-3

退出码: 0=PASS  1=FAIL  2=参数错误
"""

import argparse
import sys

import numpy as np

DTYPES = {
    "bool": np.bool_,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
}


def as_bits(a):
    """按位模式重解释数组：bool/int/float 统一为同字节宽度的无符号整数。"""
    a = np.ascontiguousarray(a)
    if a.dtype.itemsize == 1:
        return a.view(np.uint8)
    return a.view(np.dtype(f"u{a.dtype.itemsize}"))


def load_bin(path, dtype, shape):
    data = np.fromfile(path, dtype=dtype)
    expect = int(np.prod(shape))
    if data.size != expect:
        print(
            f"[ERROR] 字节数自校验失败: {path} 元素数 {data.size} != shape {shape} 期望 {expect}"
            f"（输出字节 = 元素数 × dtype 字节数，不符先查传输/shape）"
        )
        sys.exit(1)
    return data


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("result", help="待比对结果文件（二进制，row-major 裸数据）")
    ap.add_argument("golden", help="golden 数据文件（二进制，row-major 裸数据）")
    ap.add_argument("--shape", required=True, help="逗号分隔，如 128,512")
    ap.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    ap.add_argument(
        "--mode",
        required=True,
        choices=["bitwise", "tolerance"],
        help="判定模式：bitwise（按位模式逐元素比较）/ tolerance（容差，配合 --rtol/--atol）",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="相对误差阈值（tolerance 模式，默认 1e-3）",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="绝对误差阈值（tolerance 模式，默认 1e-3）",
    )
    return ap.parse_args()


def report_bitwise(result, golden, r, g, n, dtype_name):
    diff_mask = as_bits(result) != as_bits(golden)
    n_diff = int(diff_mask.sum())
    if dtype_name in ("bool", "uint8"):
        print(f"golden True(1)     : {int(golden.astype(np.uint8).sum())}")
        print(f"output True(1)     : {int(result.astype(np.uint8).sum())}")
        print("-" * 60)
    print("判定模式           : bitwise（按位模式逐元素比较，反转元素=0 即 PASS）")
    if dtype_name in ("float16", "float32", "float64"):
        print(
            "说明               : 浮点按 IEEE 位模式比较；NaN payload 不同或 +0/-0 差异会判 FAIL，"
            "语义一致请用 tolerance 模式"
        )
    print(f"反转元素数量       : {n_diff}")
    print(f"反转元素比例       : {n_diff / n:.6f}")
    ng, nr = np.linalg.norm(g), np.linalg.norm(r)
    cos = float(np.dot(g, r) / (ng * nr)) if ng > 0 and nr > 0 else float("nan")
    print(f"余弦相似度         : {cos:.6f}")
    if n_diff > 0:
        flat = np.flatnonzero(diff_mask.reshape(-1))
        first_idx = int(flat[0])
        diffs = np.abs(r - g).reshape(-1)[flat]
        max_idx = (
            first_idx
            if np.all(np.isnan(diffs))
            else int(flat[int(np.nanargmax(diffs))])
        )
        print(
            f"第一个差异点       : flat[{first_idx}] golden={g.reshape(-1)[first_idx]:.6g}"
            f" output={r.reshape(-1)[first_idx]:.6g}"
        )
        print(
            f"最大差异点         : flat[{max_idx}] golden={g.reshape(-1)[max_idx]:.6g}"
            f" output={r.reshape(-1)[max_idx]:.6g}"
        )
    else:
        print("第一个差异点       : 无")
        print("最大差异点         : 无")
    print("-" * 60)
    verdict = "PASS (bitwise 全一致)" if n_diff == 0 else f"FAIL ({n_diff}/{n} 不一致)"
    print(f"结论               : {verdict}")
    print("=" * 60)
    sys.exit(0 if n_diff == 0 else 1)


def report_tolerance(result, golden, r, g, n, rtol, atol):
    with np.errstate(invalid="ignore"):
        diff = np.abs(r - g)
        rel = diff / np.maximum(np.abs(g), 1e-6)
    one_nan = np.isnan(r) ^ np.isnan(g)
    opp_inf = np.isinf(r) & np.isinf(g) & (np.sign(r) != np.sign(g))
    bad_mask = ((diff > atol) & (rel > rtol)) | one_nan | opp_inf
    n_bad = int(bad_mask.sum())
    ng, nr = np.linalg.norm(g), np.linalg.norm(r)
    cos = float(np.dot(g, r) / (ng * nr)) if ng > 0 and nr > 0 else float("nan")
    with np.errstate(invalid="ignore"):
        noise = float(np.mean((r - g) ** 2))
    print(f"判定模式           : 容差（rtol={rtol}, atol={atol}）")
    print(f"绝对误差(平均)     : {diff.mean():.6f}")
    print(f"相对误差(平均/最大): {rel.mean():.6f} / {rel.max():.6f}")
    print(f"余弦相似度         : {cos:.6f}")
    print(f"MSE/RMSE           : {noise:.6g} / {np.sqrt(noise):.6g}")
    print(
        f"NaN/Inf 数量       : result {int(np.isnan(result).sum())}/{int(np.isinf(result).sum())}"
        f", golden {int(np.isnan(golden).sum())}/{int(np.isinf(golden).sum())}"
    )
    print(
        f"不匹配元素         : {n_bad}/{n} ({n_bad / n:.4%})  "
        f"[判定: >atol 且 >rtol×|golden|，NaN 同位一致/Inf 须同号]"
    )
    if n_bad > 0:
        idx = int(np.argmax(bad_mask))
        print(
            f"首个不匹配点       : flat[{idx}] golden={g[idx]:.6g} output={r[idx]:.6g}"
        )
    print("-" * 60)
    verdict = "PASS" if n_bad == 0 else "FAIL"
    print(f"结论               : {verdict}")
    print("=" * 60)
    sys.exit(0 if verdict == "PASS" else 1)


def main():
    args = parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))
    dt = DTYPES[args.dtype]
    rtol, atol = args.rtol, args.atol

    result = load_bin(args.result, dt, shape)
    golden = load_bin(args.golden, dt, shape)
    r, g = result.astype(np.float64), golden.astype(np.float64)
    n = r.size

    print("=" * 60)
    print(f"文件名(result)     : {args.result.split('/')[-1]}")
    print(f"golden             : {args.golden}")
    print(f"shape              : {shape}  dtype: {args.dtype}")
    print(f"元素总数           : {n}")
    print("-" * 60)

    if args.mode == "bitwise":
        report_bitwise(result, golden, r, g, n, args.dtype)
    else:
        report_tolerance(result, golden, r, g, n, rtol, atol)


if __name__ == "__main__":
    main()
