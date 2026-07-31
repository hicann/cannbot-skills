#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""A/B ratio analysis for source-NPU and generated-operator timing output.

Usage:
    python3 perf_ab.py <ref_output.txt> <asc_output.txt>

Input format: output of `python3 utils/performance.py current_task <impl> <warmup> <repeat> <seed>`.
Each case line expected: `case[N] mean=X ms, median=Y ms, samples(ms): [...]`.

Output: ref/asc sum + median, per-case ratio distribution (sum/median/geomean),
min/max ratios, distribution buckets, worst/best 5 cases.

Paired with EC-33 benchmarking methodology — use warmup=1 repeat=2 × 3 runs for
ops affected by CANN VMS 343 crash (e.g. 9_TopKTopP reference).
"""
import math
import re
import statistics
import sys


def parse(path):
    vals = []
    for line in open(path):
        m = re.search(r"case\[(\d+)\]\s+mean=([\d.]+)\s*ms", line)
        if m:
            vals.append((int(m.group(1)), float(m.group(2))))
    vals.sort()
    return [v for _, v in vals]


def main():
    if len(sys.argv) != 3:
        print("Usage: perf_ab.py <ref_output.txt> <asc_output.txt>", file=sys.stderr)
        sys.exit(1)

    ref = parse(sys.argv[1])
    asc = parse(sys.argv[2])
    if len(ref) == 0 or len(asc) == 0:
        print(
            f"ERROR: parsed ref={len(ref)} cases, asc={len(asc)} cases "
            f"(one or both empty — likely a CANN ref crash or other run-time failure)",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(ref) != len(asc):
        n = min(len(ref), len(asc))
        print(
            f"WARN: case count mismatch — ref={len(ref)} asc={len(asc)}. Using first {n}.",
            file=sys.stderr,
        )
        ref = ref[:n]
        asc = asc[:n]

    print(f"ref: n={len(ref)} sum={sum(ref):.3f}ms median={statistics.median(ref):.3f}ms")
    print(f"asc: n={len(asc)} sum={sum(asc):.3f}ms median={statistics.median(asc):.3f}ms")

    ratios = [r / a for r, a in zip(ref, asc) if a > 0]
    print(f"per-case ratio (ref/asc) median={statistics.median(ratios):.3f}x")
    print(f"geomean={math.exp(statistics.mean([math.log(x) for x in ratios])):.3f}x")
    print(f"sum-ratio={sum(ref) / sum(asc):.3f}x")
    print(f"min={min(ratios):.3f}x max={max(ratios):.3f}x")

    below_06 = sum(1 for r in ratios if r < 0.6)
    in_06_1 = sum(1 for r in ratios if 0.6 <= r < 1.0)
    above_1 = sum(1 for r in ratios if r >= 1.0)
    print(f"distribution: <0.6x:{below_06} 0.6-1.0x:{in_06_1} >=1.0x:{above_1}")

    indexed = sorted(enumerate(ratios), key=lambda x: x[1])
    print(f"worst 5: {[(i, round(r, 3)) for i, r in indexed[:5]]}")
    print(f"best 5: {[(i, round(r, 3)) for i, r in indexed[-5:]]}")


if __name__ == "__main__":
    main()
