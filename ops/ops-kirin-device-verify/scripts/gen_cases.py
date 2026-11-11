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
"""设备验证用例数据生成参考脚本（输入 bin + golden）

参考脚本（非通用工具）：演示"seed 可复现 + 随机底板 + 特殊值注入"的 golden 生成模式，
供整个 skill 在生成用例数据时参考。目标算子不在内置清单、或非逐元素算子时，
复制本脚本后按算子语义修改 golden 计算与注入逻辑。

- seed 固定可复现；随机底板 + 特殊值注入（float，种类见 --special，与算术语义解耦）：
  nan（NaN）、inf（±Inf）、equal（跨输入等值对）、ulp（单 ulp 相邻对）、
  spread（±10 大差异）、zpair（+0/-0 对）；跨输入类注入仅限二元输入同 shape
- golden 与设备输出同构：bool 结果以 uint8(0/1) 存盘，其余按实际 dtype 存盘

用法:
  python3 gen_cases.py --golden-op less --shape 128,512 --dtype float16 \
      --seed 97 --special nan,inf,equal,ulp,spread,zpair --out-dir ./case_data

产物: <out-dir>/x1.bin x2.bin golden.bin（二元算子）；一元算子产 x1.bin/golden.bin
"""

import argparse
import os

import numpy as np

BINARY_OP_MAP = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "less": np.less,
    "greater": np.greater,
    "less_equal": np.less_equal,
    "greater_equal": np.greater_equal,
    "equal": np.equal,
    "not_equal": np.not_equal,
}
UNARY_OP_MAP = {
    "abs": np.abs,
    "neg": np.negative,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "is_finite": np.isfinite,
    "isinf": np.isinf,
    "isnan": np.isnan,
}

DTYPES = {
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

FLOAT_KINDS = {"nan", "inf", "equal", "ulp", "spread", "zpair"}
CROSS_KINDS = {"equal", "ulp", "spread", "zpair"}


def _idx(rng, n, ratio):
    n_sp = min(max(int(n * ratio), 8), n)
    return rng.choice(n, n_sp, replace=False), n_sp


def inject_nan(arrays, dt, rng, ratio):
    for arr in arrays:
        idx, _ = _idx(rng, arr.size, ratio)
        arr.reshape(-1)[idx] = np.nan


def inject_inf(arrays, dt, rng, ratio):
    flat = arrays[0].reshape(-1)
    idx, n = _idx(rng, arrays[0].size, ratio)
    half = n // 2
    flat[idx[:half]] = np.inf
    flat[idx[half:]] = -np.inf


def inject_equal(arrays, dt, rng, ratio):
    f0, f1 = arrays[0].reshape(-1), arrays[1].reshape(-1)
    idx, _ = _idx(rng, arrays[0].size, ratio)
    f1[idx] = f0[idx]


def inject_ulp(arrays, dt, rng, ratio):
    f0, f1 = arrays[0].reshape(-1), arrays[1].reshape(-1)
    idx, n = _idx(rng, arrays[0].size, ratio)
    a = np.asarray(rng.uniform(-2, 2, n), dtype=dt)
    b = np.nextafter(a, np.inf, dtype=dt)
    f0[idx] = a
    f1[idx] = b


def inject_spread(arrays, dt, rng, ratio):
    f0, f1 = arrays[0].reshape(-1), arrays[1].reshape(-1)
    idx, n = _idx(rng, arrays[0].size, ratio)
    half = n // 2
    c0, c1 = idx[:half], idx[half:]
    f0[c0] = np.where(
        np.isfinite(f1[c0]), f1[c0] + np.dtype(dt).type(10.0), np.dtype(dt).type(5.0)
    )
    f0[c1] = np.where(
        np.isfinite(f1[c1]), f1[c1] - np.dtype(dt).type(10.0), np.dtype(dt).type(-5.0)
    )


def inject_zpair(arrays, dt, rng, ratio):
    f0, f1 = arrays[0].reshape(-1), arrays[1].reshape(-1)
    idx, _ = _idx(rng, arrays[0].size, ratio)
    f0[idx] = np.dtype(dt).type(0.0)
    f1[idx] = -np.dtype(dt).type(0.0)


def inject_specials(arrays, dt, rng, special_ratio, kinds):
    """在输入数组上按种类分块注入特殊值（跨输入类仅限二元同 shape 时生效）"""
    same_size = len({a.size for a in arrays}) == 1
    active = [
        k
        for k in kinds
        if k in FLOAT_KINDS
        and (k not in CROSS_KINDS or (len(arrays) >= 2 and same_size))
    ]
    for k in active:
        globals()[f"inject_{k}"](arrays, dt, rng, special_ratio)
    return arrays


def save_case(out_dir, inputs, golden):
    os.makedirs(out_dir, exist_ok=True)
    for i, arr in enumerate(inputs, 1):
        arr.astype(arr.dtype, copy=False).tofile(os.path.join(out_dir, f"x{i}.bin"))
    if golden.dtype == np.bool_:
        golden = golden.astype(np.uint8)
    golden.tofile(os.path.join(out_dir, "golden.bin"))
    n = inputs[0].size
    if np.issubdtype(golden.dtype, np.integer) and np.all(
        (golden == 0) | (golden == 1)
    ):
        print(
            f"  golden: True(1)={int(golden.sum())} False(0)={n - int(golden.sum())} "
            f"存储为 {golden.dtype.name}"
        )
    print(
        f"  元素数={n}  产物: {os.path.abspath(out_dir)}/"
        + ", ".join(f"x{i}.bin" for i in range(1, len(inputs) + 1))
        + ", golden.bin"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--golden-op",
        required=True,
        choices=sorted(set(BINARY_OP_MAP) | set(UNARY_OP_MAP)),
        help="内置 numpy 逐元素函数名（golden 计算方式），如 less/add/is_finite；"
        "其他算子请复制脚本后按语义修改",
    )
    ap.add_argument(
        "--shape", required=True, help="逗号分隔，如 128,512；必须与 spec 一致"
    )
    ap.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    ap.add_argument("--seed", type=int, default=97, help="随机种子（固定保证可复现）")
    ap.add_argument(
        "--special",
        default="nan,inf,equal,ulp,spread,zpair",
        help="特殊值注入种类（逗号分隔，float 专用；0 或空字符串关闭）",
    )
    ap.add_argument(
        "--special-ratio", type=float, default=0.12, help="特殊值注入比例（默认 0.12）"
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    dt = DTYPES[args.dtype]
    rng = np.random.default_rng(args.seed)
    is_float = np.issubdtype(dt, np.floating)
    lo, hi = (-2.0, 2.0) if args.dtype == "float16" else (-100.0, 100.0)
    if not is_float:
        lo, hi = 0, min(int(np.iinfo(dt).max), 1000)

    special_ratio = args.special_ratio
    kinds = (
        [k.strip() for k in args.special.split(",") if k.strip()]
        if args.special
        else []
    )
    if is_float and special_ratio > 0 and int(np.prod(shape)) < 64:
        print(f"[WARN] 元素数 {int(np.prod(shape))} 过小，跳过特殊值注入")
        kinds = []

    if args.golden_op in UNARY_OP_MAP:
        inputs = [rng.uniform(lo, hi, shape).astype(dt)]
        if is_float and kinds and special_ratio > 0:
            inputs = inject_specials(inputs, dt, rng, special_ratio, kinds)
        golden = UNARY_OP_MAP[args.golden_op](inputs[0])
        print(
            f"[case] {args.golden_op} shape={shape} dtype={args.dtype} seed={args.seed}"
        )
    else:
        x1 = rng.uniform(lo, hi, shape).astype(dt)
        x2 = rng.uniform(lo, hi, shape).astype(dt)
        if is_float and kinds and special_ratio > 0:
            x1, x2 = inject_specials([x1, x2], dt, rng, special_ratio, kinds)
        inputs = [x1, x2]
        golden = BINARY_OP_MAP[args.golden_op](x1, x2)
        print(
            f"[case] {args.golden_op} shape={shape} dtype={args.dtype} seed={args.seed}"
        )

    if np.issubdtype(golden.dtype, np.bool_):
        n_true = int(np.asarray(golden, dtype=np.uint8).sum())
        print(
            f"  golden 为 bool：True(1)={n_true} False(0)={inputs[0].size - n_true}，将存为 uint8"
        )
    if is_float:
        print(
            f"  NaN(x1)={np.isnan(inputs[0]).sum()}"
            + (f" NaN(x2)={np.isnan(inputs[1]).sum()}" if len(inputs) > 1 else "")
            + f" Inf={np.isinf(inputs[0]).sum() + (np.isinf(inputs[1]).sum() if len(inputs) > 1 else 0)}"
        )

    save_case(args.out_dir, inputs, golden)


if __name__ == "__main__":
    main()
