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

"""
V220 pre-build correctness check — automated safety net.

Catches the 8 known "compiles clean, crashes at runtime" V220 error classes
BEFORE the build step. Runs on kernel source files in the current workspace.

Exit codes: 0 = all checks passed, 1 = check failed (block build), 2 = infra error.
"""

import sys
import os
import re
import glob


def find_kernel_files(ws_root):
    """Find kernel .h and .cpp files in workspace."""
    patterns = [
        os.path.join(ws_root, "kernel", "*.h"),
        os.path.join(ws_root, "kernel", "*.cpp"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return files


def check_tbuf_initbuffer(files):
    """EC-62: Every TBuf declaration must have a matching InitBuffer call."""
    tbuf_decls = []
    init_calls = set()
    for f in files:
        with open(f) as fh:
            content = fh.read()
        for m in re.finditer(r'TBuf<[^>]+>\s+(\w+)_\s*;', content):
            tbuf_decls.append((f, m.group(1)))
        for m in re.finditer(r'InitBuffer\((\w+)_\s*,', content):
            init_calls.add(m.group(1))

    missing = []
    for fpath, name in tbuf_decls:
        if name not in init_calls:
            missing.append(f"{fpath}: TBuf {name}_ declared but no pipe_.InitBuffer({name}_, ...)")
    return missing


def check_blockdim(files):
    """EC-47/EC-60: pybind11 nblk must be >= 1."""
    for f in files:
        if 'pybind11.cpp' not in f:
            continue
        with open(f) as fh:
            content = fh.read()
        if re.search(r'nblk\s*=\s*0[^.]', content):
            return [f"{f}: nblk = 0 found — will cause ACL_ERROR_RT_PARAM_INVALID"]
        if re.search(r'block_dim\s*=\s*0[^.]', content):
            return [f"{f}: block_dim = 0 found — will cause ACL_ERROR_RT_PARAM_INVALID"]
    return []


def check_pragma_pack(files):
    """OL-178: Tiling struct must have #pragma pack."""
    issues = []
    for f in files:
        if 'tiling' not in os.path.basename(f).lower():
            continue
        with open(f) as fh:
            content = fh.read()
        has_struct = bool(re.search(r'struct\s+\w+Tiling', content))
        has_pack = 'pragma pack' in content
        if has_struct and not has_pack:
            issues.append(f"{f}: Tiling struct without #pragma pack(1) — host/device ABI risk")
    return issues


def check_nblk_appropriateness(files):
    """Reward-hack guard: nblk=1 with large workloads = suspicious single-core cheat."""
    warnings = []
    for f in files:
        if 'pybind11.cpp' not in f:
            continue
        with open(f) as fh:
            content = fh.read()
        # Check if nblk=1 is hardcoded (not computed from workload)
        if re.search(r'nblk\s*=\s*1\s*;', content):
            # Check if there's a total_elements computation nearby
            if re.search(r'total\s*=\s*[Nn].*\*', content):
                warnings.append(
                    f"{f}: nblk=1 hardcoded with multi-dimensional workload — "
                    "suspicious single-core. Justify or compute dynamic nblk."
                )
    return warnings


def check_host_compute_laundering(files):
    """Reward-hack guard: business logic in pybind11 instead of kernel."""
    warnings = []
    compute_patterns = [
        (r'torch\.(sum|mean|max|min|sort|argsort|topk|matmul|mm|bmm)', 'torch compute'),
        (r'torch\.nn\.functional\.', 'torch functional compute'),
        (r'for\s+\w+\s+in\s+range.*:', 'Python loop on tensor data'),
        (r'\.numpy\(\)', 'CPU round-trip'),
    ]
    for f in files:
        if 'pybind11.cpp' not in f and 'model_new' not in f:
            continue
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                for pattern, desc in compute_patterns:
                    if re.search(pattern, line):
                        warnings.append(
                            f"{f}:{i}: {desc} in host code — reward-hack risk. All compute must be in kernel.")
    return warnings


def check_datacopy_alignment(files):
    """PB-22: DataCopy element counts should be compile-time aligned or use AlignUp pattern."""
    warnings = []
    for f in files:
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                if 'DataCopy(' in line and 'AlignUp' not in line and 'aligned' not in line.lower():
                    warnings.append(f"{f}:{i}: DataCopy without visible alignment — verify count is AlignUp32")
    return warnings  # non-blocking warning (can't verify runtime values statically)


def check_std_string(files):
    """EC-63: pybind11 std::string → basic_string null SIGSEGV on V220 ARM64."""
    issues = []
    for f in files:
        if 'pybind11.cpp' not in f:
            continue
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                if 'std::string' in line and 'PYBIND11_MODULE' not in line:
                    issues.append(
                        f"{f}:{i}: std::string in pybind11 — ABI crash on V220 ARM64. "
                        "Use const char* instead."
                    )
    return issues


def main():
    ws_root = os.environ.get("ASCENDC_WORKSPACE", os.getcwd())
    files = find_kernel_files(ws_root)

    if not files:
        print("[v220_prebuild_check] No kernel files found — skipping")
        return 0

    results = {
        "EC-62 (TBuf InitBuffer)": check_tbuf_initbuffer(files),
        "EC-60 (blockDim>0)": check_blockdim(files),
        "OL-178 (pragma pack)": check_pragma_pack(files),
        "EC-63 (std::string)": check_std_string(files),
        "RW-01 (nblk=1 cheat)": check_nblk_appropriateness(files),
        "RW-02 (host compute)": check_host_compute_laundering(files),
        "PB-22 (DataCopy align)": check_datacopy_alignment(files),
    }

    blocking = []
    warnings = []

    for check_name, issues in results.items():
        if check_name.startswith("RW-") or check_name == "PB-22 (DataCopy align)":
            warnings.extend(issues)
        elif issues:
            blocking.extend(issues)

    if warnings:
        print(f"[v220_prebuild_check] WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")

    if blocking:
        print(f"[v220_prebuild_check] BLOCKING ({len(blocking)}):")
        for b in blocking:
            print(f"  {b}")
        print("[v220_prebuild_check] Build BLOCKED — fix the above before rebuilding.")
        return 1

    print(f"[v220_prebuild_check] OK — {len(files)} files checked, 0 blocking issues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
