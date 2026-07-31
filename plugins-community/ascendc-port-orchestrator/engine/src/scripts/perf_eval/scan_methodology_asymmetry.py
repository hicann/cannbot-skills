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

"""Retroactive scanner for P97 PERF_METHODOLOGY_ASYMMETRY in already-shipped archives.

P97 finalize-time gate (commit 39a86ba) only blocks NEW archives from shipping
with asymmetric methodology. It does NOT scan existing archives on main.
This script does — runs over output/*/src/kernels/*/verification.json and
flags any that match the asymmetric-method signature.

Exit codes:
  0  no asymmetric archives found
  1  ≥1 archive flagged
  2  scan error

Usage:
  python3 scan_methodology_asymmetry.py [--archive-root output] [--strict]

The --strict mode requires every PASS/PASS_WITHIN_TOLERANCE archive to declare
a method field with at least one of: torch_npu.profiler / torch.npu.Event /
explicit symmetric tooling. Empty method is rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Asymmetric signals — explicit C++ chrono-style timing without symmetric profiler
ASYMMETRIC_SIGNALS = (
    "std::chrono",
    "high_resolution_clock",
    "aclrtevent",  # C++-side aclrtEvent timing
    "aclrtsynchronizestream",
)
# Symmetric signals — same NPU-side primitive on both sides
SYMMETRIC_SIGNALS = (
    "torch_npu.profiler",
    "torch.npu.event",
    "torch.npu.Event",
)


def scan_one(vj_path: Path, strict: bool) -> tuple[str, str] | None:
    """Return (severity, message) if archive is flagged, else None."""
    try:
        vj = json.loads(vj_path.read_text())
    except Exception as e:
        return ("ERROR", f"cannot parse: {e}")

    prec = (vj.get("precision") or {}).get("status", "")
    perf = vj.get("performance") or {}
    perf_status = perf.get("status", "")

    # Only scan archives that CLAIM a performance number
    if perf_status in ("N/A", "SKIPPED", "NA", "", None,
                       "RE_EVALUATED_PENDING_A3_BASELINE",
                       "RE_EVALUATED_OP_UNAVAILABLE"):
        return None
    if prec not in ("PASS", "PASS_WITHIN_TOLERANCE", "PARTIAL"):
        return None

    method = (perf.get("method") or perf.get("method_note") or "").lower()

    has_asymmetric = any(s in method for s in ASYMMETRIC_SIGNALS)
    has_symmetric = any(s.lower() in method for s in SYMMETRIC_SIGNALS)

    if has_asymmetric and not has_symmetric:
        return ("FAIL",
                f"asymmetric method declared (ratio={perf.get('ratio')}): {method[:200]}")

    if strict and not has_symmetric:
        return ("STRICT_FAIL",
                f"no symmetric profiler declared (ratio={perf.get('ratio')}): "
                f"{(method or '<empty>')[:200]}")

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", default="output", type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="Reject archives that don't declare a symmetric profiler")
    args = ap.parse_args()

    root = args.archive_root.resolve()
    if not root.is_dir():
        print(f"FATAL: archive root not found: {root}", file=sys.stderr)
        return 2

    vj_files = sorted(root.glob("*/src/kernels/*/verification.json"))
    if not vj_files:
        print(f"WARN: no verification.json under {root}", file=sys.stderr)
        return 0

    flagged = []
    for vj in vj_files:
        result = scan_one(vj, args.strict)
        if result is None:
            continue
        severity, msg = result
        rel = vj.relative_to(root.parent if root.parent != Path() else Path("."))
        flagged.append((severity, rel, msg))

    if not flagged:
        print(f"OK: {len(vj_files)} archives scanned, none asymmetric.")
        return 0

    print(f"\n=== P97 RETROACTIVE SCAN — {len(flagged)} archives flagged ===")
    for sev, rel, msg in flagged:
        print(f"[{sev}] {rel}")
        print(f"   {msg}")
    print(f"\nTotal: {len(flagged)}/{len(vj_files)} archives need re-evaluation "
          f"with symmetric methodology (PR #103 + torch.npu.Event / torch_npu.profiler).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
