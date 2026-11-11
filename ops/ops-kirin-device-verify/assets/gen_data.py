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
"""Generate AddCustom input tensors and the golden output.

The template writes three raw float16 files for a two-input AddCustom kernel:

- ``x.bin``: first input tensor
- ``y.bin``: second input tensor
- ``z.bin``: golden output (``x + y``)

The default tensor shape is ``[8, 2048]``. Adjust ``--shape`` when the target
operator uses a different shape.
"""

import argparse
from pathlib import Path

import numpy as np

DEFAULT_SHAPE = (8, 2048)
DEFAULT_SEED = 0


def build_tensors(shape, seed):
    """Create the two inputs and the golden output."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-100.0, 100.0, size=shape).astype(np.float16)
    y = rng.uniform(-100.0, 100.0, size=shape).astype(np.float16)
    z = (x + y).astype(np.float16)
    return x, y, z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="directory used for x.bin, y.bin and z.bin",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=DEFAULT_SHAPE,
        metavar=("M", "N"),
        help="tensor shape (default: 8 2048)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, z = build_tensors(tuple(args.shape), args.seed)
    for name, array in (("x.bin", x), ("y.bin", y), ("z.bin", z)):
        (out_dir / name).write_bytes(array.tobytes())

    print(
        f"generated x.bin, y.bin, z.bin in {out_dir} "
        f"(shape={tuple(args.shape)}, dtype=float16)"
    )


if __name__ == "__main__":
    main()
