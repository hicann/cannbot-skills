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
"""Estimate the per-AIV UB budget of a kernel on A2 (DAV_2201) and scale A5 tiles.

The A5 (DAV_3510) UB is 248KB per AIV and RegBase/SIMT paths reserve 40KB for the
SIMT DCache, leaving ~208KB; A2 (DAV_2201) offers 192KB and has no SIMT reserve.
Copying A5 tile constants to A2 therefore overflows. This script does the
arithmetic only; it cannot see padding, alignment or access patterns, so a PASS
still has to be confirmed on device.

Usage:
    python3 a2_ub_budget.py --buffer in_x:float16:32768:2 --buffer out:float32:32768:2
    python3 a2_ub_budget.py --scale-from-a5 --tile-elements 32768 --dtype float16
    python3 a2_ub_budget.py --buffer a:bf16:16384:2 --json

Exit codes:
    0  fits within the budget
    1  over budget (or a single buffer larger than the budget)
    2  usage error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

SCHEMA = "ascend-op-arch-migration/ub-budget/v1"
TOOL_VERSION = "1.0.0"

DTYPE_BYTES: Dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "float": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "half": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int4": 0.5,
    "int4b": 0.5,
    "int16": 2.0,
    "uint16": 2.0,
    "short": 2.0,
    "int32": 4.0,
    "uint32": 4.0,
    "int": 4.0,
    "int64": 8.0,
    "uint64": 8.0,
    "double": 8.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "hifloat8": 1.0,
    "hif8": 1.0,
}

DEFAULT_UB_KB = 192.0  # DAV_2201 typical user-visible UB per AIV
DEFAULT_A5_UB_KB = 248.0  # DAV_3510 typical user-visible UB per AIV
DEFAULT_A5_RESERVE_KB = 40.0  # SIMT DCache reserve on A5 RegBase/SIMT paths


@dataclass
class Buffer:
    name: str
    dtype: str
    elements: int
    copies: int

    @property
    def element_bytes(self) -> float:
        return DTYPE_BYTES[self.dtype]

    @property
    def total_bytes(self) -> int:
        return int(math.ceil(self.elements * self.element_bytes * self.copies))

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "elements": self.elements,
            "copies": self.copies,
            "element_bytes": self.element_bytes,
            "total_bytes": self.total_bytes,
        }


def parse_buffer(spec: str) -> Buffer:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"buffer spec must be name:dtype:elements:copies, got '{spec}'"
        )
    name, dtype, elements, copies = parts
    key = dtype.strip().lower()
    if key not in DTYPE_BYTES:
        raise ValueError(
            f"unknown dtype '{dtype}'; known: {', '.join(sorted(DTYPE_BYTES))}"
        )
    try:
        elems = int(elements)
        cps = int(copies)
    except ValueError as exc:
        raise ValueError(
            f"elements/copies must be integers in '{spec}': {exc}"
        ) from exc
    if elems <= 0 or cps <= 0:
        raise ValueError(f"elements/copies must be positive in '{spec}'")
    if not name.strip():
        raise ValueError(f"buffer name must not be empty in '{spec}'")
    return Buffer(name.strip(), key, elems, cps)


def align_down(value: int, align: int) -> int:
    if align <= 1:
        return value
    return (value // align) * align


def budget_report(
    buffers: Sequence[Buffer], ub_kb: float, reserve_kb: float
) -> Dict[str, object]:
    available = int(ub_kb * 1024)
    reserve = int(reserve_kb * 1024)
    usable = available - reserve
    used = sum(b.total_bytes for b in buffers)
    remaining = usable - used
    fits = used <= usable and all(b.total_bytes <= usable for b in buffers)
    largest = max(buffers, key=lambda b: b.total_bytes) if buffers else None

    suggestions: List[str] = []
    if largest is not None and not fits:
        if used > usable and used > 0:
            ratio = usable / used
            suggestions.append(
                f"scale every tile by {ratio:.3f} (floor) so the set fits exactly; "
                f"check tail proportions after scaling"
            )
        if largest.total_bytes > usable:
            per_element_copies = largest.element_bytes * largest.copies
            max_elements = (
                max(1, int(usable // per_element_copies))
                if per_element_copies
                else largest.elements
            )
            suggestions.append(
                f"buffer '{largest.name}' alone needs {largest.total_bytes} B > {usable} B; "
                f"reduce elements from {largest.elements} to <= {max_elements} (or reduce copies)"
            )
        else:
            suggestions.append(
                f"largest buffer '{largest.name}' uses {largest.total_bytes} B; "
                f"halving its tile frees {largest.total_bytes // 2} B"
            )

    return {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "policy": {
            "ub_kb": ub_kb,
            "reserve_kb": reserve_kb,
            "available_bytes": available,
            "usable_bytes": usable,
        },
        "buffers": [b.to_dict() for b in buffers],
        "totals": {
            "used_bytes": used,
            "remaining_bytes": remaining,
            "used_pct": round(100.0 * used / usable, 2) if usable > 0 else None,
        },
        "verdict": "PASS" if fits else "OVER",
        "suggestions": suggestions,
    }


def scale_report(
    tile_elements: int,
    dtype: str,
    ub_kb: float,
    reserve_kb: float,
    a5_ub_kb: float,
    a5_reserve_kb: float,
    align_bytes: int,
) -> Dict[str, object]:
    key = dtype.strip().lower()
    if key not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype '{dtype}'")
    elem = DTYPE_BYTES[key]
    a2_usable = int(ub_kb * 1024) - int(reserve_kb * 1024)
    a5_usable = int(a5_ub_kb * 1024) - int(a5_reserve_kb * 1024)
    scale = a2_usable / a5_usable if a5_usable > 0 else 1.0
    align_elems = max(1, int(round(align_bytes / elem)))
    scaled = tile_elements * scale
    recommended = max(align_elems, align_down(int(math.floor(scaled)), align_elems))
    return {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "mode": "scale-from-a5",
        "dtype": key,
        "element_bytes": elem,
        "a5": {
            "ub_kb": a5_ub_kb,
            "reserve_kb": a5_reserve_kb,
            "usable_bytes": a5_usable,
        },
        "a2": {"ub_kb": ub_kb, "reserve_kb": reserve_kb, "usable_bytes": a2_usable},
        "scale": round(scale, 4),
        "tile_elements_input": tile_elements,
        "align_bytes": align_bytes,
        "align_elements": align_elems,
        "tile_elements_recommended": recommended,
        "bytes_per_copy_input": int(math.ceil(tile_elements * elem)),
        "bytes_per_copy_recommended": int(math.ceil(recommended * elem)),
    }


def render_text(report: Dict[str, object]) -> str:
    lines: List[str] = []
    if report.get("mode") == "scale-from-a5":
        a2 = report["a2"]  # type: ignore[index]
        a5 = report["a5"]  # type: ignore[index]
        lines.append("A5 -> A2 tile scaling")
        lines.append(
            f"  A5 usable UB : {a5['usable_bytes']} B ({a5['ub_kb']} KB - {a5['reserve_kb']} KB reserve)"
        )
        lines.append(
            f"  A2 usable UB : {a2['usable_bytes']} B ({a2['ub_kb']} KB - {a2['reserve_kb']} KB reserve)"
        )
        lines.append(f"  scale        : {report['scale']}")
        lines.append(
            f"  dtype        : {report['dtype']} ({report['element_bytes']} B/element)"
        )
        lines.append(
            f"  tile         : {report['tile_elements_input']} -> {report['tile_elements_recommended']} elements "
            f"({report['bytes_per_copy_input']} B -> {report['bytes_per_copy_recommended']} B per copy)"
        )
        lines.append(
            f"  align        : {report['align_elements']} elements ({report['align_bytes']} B)"
        )
        lines.append("")
        lines.append(
            "Check tail proportions and copy counts after scaling; PASS is arithmetic only."
        )
        return "\n".join(lines)

    policy = report["policy"]  # type: ignore[index]
    totals = report["totals"]  # type: ignore[index]
    lines.append("A2 per-AIV UB budget")
    lines.append(
        f"  available: {policy['available_bytes']} B  "
        f"reserve: {policy['reserve_kb']} KB  usable: {policy['usable_bytes']} B"
    )
    lines.append("")
    lines.append("| buffer | dtype | elements | copies | bytes |")
    lines.append("|---|---|---|---|---|")
    for b in report["buffers"]:  # type: ignore[union-attr]
        lines.append(
            f"| {b['name']} | {b['dtype']} | {b['elements']} | {b['copies']} | {b['total_bytes']} |"
        )
    lines.append("")
    lines.append(
        f"  used: {totals['used_bytes']} B ({totals['used_pct']}%)  remaining: {totals['remaining_bytes']} B"
    )
    lines.append(f"  verdict: {report['verdict']}")
    for s in report["suggestions"]:  # type: ignore[union-attr]
        lines.append(f"  - {s}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A2 (DAV_2201) per-AIV UB budget calculator for A5->A2 migration."
    )
    parser.add_argument(
        "--buffer",
        action="append",
        default=[],
        help="name:dtype:elements:copies (repeatable)",
    )
    parser.add_argument(
        "--ub-kb",
        type=float,
        default=DEFAULT_UB_KB,
        help="A2 usable UB in KB (default 192)",
    )
    parser.add_argument(
        "--reserve-kb", type=float, default=0.0, help="extra reserve in KB (default 0)"
    )
    parser.add_argument(
        "--scale-from-a5",
        action="store_true",
        help="report the A5->A2 tile scale instead of a budget table",
    )
    parser.add_argument(
        "--tile-elements", type=int, help="A5 tile element count (for --scale-from-a5)"
    )
    parser.add_argument(
        "--dtype", default="float16", help="tile dtype (for --scale-from-a5)"
    )
    parser.add_argument(
        "--a5-ub-kb",
        type=float,
        default=DEFAULT_A5_UB_KB,
        help="A5 usable UB in KB (default 248)",
    )
    parser.add_argument(
        "--a5-reserve-kb",
        type=float,
        default=DEFAULT_A5_RESERVE_KB,
        help="A5 SIMT/RegBase reserve in KB (default 40)",
    )
    parser.add_argument(
        "--align-bytes",
        type=int,
        default=32,
        help="alignment for the scaled tile (default 32)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    try:
        if args.scale_from_a5:
            if not args.tile_elements or args.tile_elements <= 0:
                parser.error("--scale-from-a5 requires a positive --tile-elements")
            report = scale_report(
                args.tile_elements,
                args.dtype,
                args.ub_kb,
                args.reserve_kb,
                args.a5_ub_kb,
                args.a5_reserve_kb,
                args.align_bytes,
            )
            print(
                json.dumps(report, ensure_ascii=False, indent=2)
                if args.json
                else render_text(report)
            )
            return 0

        if not args.buffer:
            parser.error("provide at least one --buffer, or use --scale-from-a5")
        buffers = [parse_buffer(spec) for spec in args.buffer]
        report = budget_report(buffers, args.ub_kb, args.reserve_kb)
        print(
            json.dumps(report, ensure_ascii=False, indent=2)
            if args.json
            else render_text(report)
        )
        return 0 if report["verdict"] == "PASS" else 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
