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
"""Compare a CPU/reference tensor against a plugin subgraph output.

Supports .npy directly. For raw .bin, pass --dtype and --shape.
int/bool compare exactly; float compare with rtol/atol.
"""

import argparse
import sys

import numpy as np

DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}


def load(path, dtype, shape):
    if path.endswith(".npy"):
        return np.load(path)
    np_dtype = DTYPES.get(dtype)
    if np_dtype is None:
        sys.exit(f"[ERROR] dtype for .bin must be one of {sorted(DTYPES)}; got: {dtype}")
    if not shape:
        sys.exit("[ERROR] --shape is required for .bin, e.g. --shape 1,50")
    arr = np.fromfile(path, dtype=np_dtype)
    return arr.reshape([int(x) for x in shape.split(",")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", help="CPU/reference tensor (.npy or .bin)")
    parser.add_argument("actual", help="plugin subgraph output (.npy or .bin)")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--dtype", default=None, help="for .bin, e.g. float32")
    parser.add_argument("--shape", default=None, help="for .bin, e.g. 1,50")
    args = parser.parse_args()

    expected = load(args.expected, args.dtype, args.shape)
    actual = load(args.actual, args.dtype, args.shape)

    if expected.shape != actual.shape:
        print(f"[FAIL] shape mismatch: expected={expected.shape} actual={actual.shape}")
        return 1

    if np.issubdtype(expected.dtype, np.integer) or expected.dtype == np.bool_:
        equal = np.array_equal(expected, actual)
        mismatch = int(np.count_nonzero(expected != actual))
        print(f"[{'PASS' if equal else 'FAIL'}] exact | shape={expected.shape} "
              f"dtype={expected.dtype} mismatch={mismatch}")
        return 0 if equal else 1

    exp = expected.astype(np.float64)
    act = actual.astype(np.float64)
    close = np.isclose(exp, act, rtol=args.rtol, atol=args.atol, equal_nan=True)
    abs_err = np.abs(exp - act)
    max_abs = float(np.nanmax(abs_err)) if abs_err.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(np.abs(exp) > 0, abs_err / np.abs(exp), 0.0)
    max_rel = float(np.nanmax(rel_err)) if rel_err.size else 0.0
    print(f"[{'PASS' if np.all(close) else 'FAIL'}] float | rtol={args.rtol} atol={args.atol} "
          f"shape={expected.shape} dtype={expected.dtype} "
          f"mismatch={int(np.count_nonzero(~close))}")
    print(f"        max_abs_err={max_abs:.6e} max_rel_err={max_rel:.6e}")
    return 0 if np.all(close) else 1


if __name__ == "__main__":
    sys.exit(main())
