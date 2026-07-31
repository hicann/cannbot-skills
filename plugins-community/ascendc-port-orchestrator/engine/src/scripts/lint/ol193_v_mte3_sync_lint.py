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
"""OL-193 V→MTE3 sync lint.

Scans AscendC kernel sources for the V220→V351 port gotcha codified in OL-193:
the `Cast(low_prec, fp32_buf); PipeBarrier<PIPE_V>(); DataCopy(gm_out, low_prec)`
pattern works on V220 (implicit/benign timing) but on V351 MTE3 fires before V
completes → stale UB read → GM gets garbage (typically fp16 1.0 saturated on
first chunk of inner loop).

Flag conditions: Cast statement followed within ~3 lines by `PipeBarrier<PIPE_V>`
followed within ~6 lines by `DataCopy*(gm, ...)`, with NO existing V→MTE3 sync
in between. V→MTE3 sync is recognized as:
  - Explicit: `SetFlag<HardEvent::V_MTE3>(...) + WaitFlag<HardEvent::V_MTE3>(...)`
              or `SetWaitFlag<HardEvent::V_MTE3>()`
  - Implicit: `TQue.EnQue(...) + TQue.DeQue<...>()` pair

Usage:
  python3 ol193_v_mte3_sync_lint.py [path1] [path2] ...

If no paths given, scans:
  output/a3_to_a5_port/src/kernels/
  workspace/

Exit 0 with empty hit list = all sites compliant.
Exit 0 with non-empty hit list = candidate sites for human review.
"""
import logging
import re
import sys
from pathlib import Path

GM_PATTERN = re.compile(r'\w*[Gg]m\w*|GlobalTensor')
V_MTE3_SYNC_PATTERN = re.compile(
    r'(SetFlag|WaitFlag|SetWaitFlag).*V_MTE3'
    r'|V_MTE3.*(SetFlag|WaitFlag)'
)
TQUE_SYNC_PATTERN = re.compile(r'\b\w+\.EnQue\(.*\)|\b\w+\.template\s+DeQue<|\b\w+\.DeQue<')

DEFAULT_ROOTS = [
    "output/a3_to_a5_port/src/kernels",
    "workspace",
]


def find_candidates_in_text(text: str) -> list[tuple[int, int]]:
    """Return (pipe_barrier_line, datacopy_line) pairs that need OL-193 retrofit."""
    hits = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'PipeBarrier<PIPE_V>' not in line:
            continue
        cast_idx = next(
            (j for j in range(max(0, i - 3), i) if 'Cast(' in lines[j]),
            -1,
        )
        if cast_idx == -1:
            continue
        dc_idx = -1
        for j in range(i + 1, min(len(lines), i + 7)):
            l = lines[j]
            if not re.search(r'DataCopy[A-Za-z]*\(', l):
                continue
            arg_match = re.search(r'DataCopy[A-Za-z]*[<\w]*\(([^,)]+)', l)
            if arg_match and GM_PATTERN.search(arg_match.group(1)):
                dc_idx = j
                break
            if GM_PATTERN.search(l) and ('Gm[' in l or 'gm[' in l):
                dc_idx = j
                break
        if dc_idx == -1:
            continue
        between = '\n'.join(lines[i + 1:dc_idx])
        # Compliant if explicit V_MTE3 SetFlag/WaitFlag OR TQue.EnQue/DeQue
        if V_MTE3_SYNC_PATTERN.search(between):
            continue
        has_enq = 'EnQue(' in between
        has_deq = re.search(r'\.DeQue<|\.template\s+DeQue<', between)
        if has_enq and has_deq:
            continue
        hits.append((i + 1, dc_idx + 1))
    return hits


def scan_paths(roots: list[Path]) -> dict[str, list[str]]:
    hits_per_file = {}
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob("*.h")) + list(root.rglob("*.cpp")):
            skip_current_item = False
            try:
                text = f.read_text(errors='ignore')
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
            file_hits = find_candidates_in_text(text)
            if file_hits:
                hits_per_file[str(f)] = [f"PipeBarrier:L{p} → DataCopy:L{d}" for p, d in file_hits]
    return hits_per_file


def main() -> int:
    roots = [Path(p) for p in (sys.argv[1:] or DEFAULT_ROOTS)]
    hits = scan_paths(roots)
    n_files = len(hits)
    n_sites = sum(len(v) for v in hits.values())
    print(f"OL-193 V→MTE3 sync lint — scanned {len(roots)} roots")
    print(f"  candidate sites: {n_sites} across {n_files} files")
    if not hits:
        print("  all sites compliant (or no bare PipeBarrier<PIPE_V> + DataCopy-to-Gm patterns)")
        return 0
    print()
    for path, sites in sorted(hits.items()):
        print(f"  {path}")
        for s in sites:
            print(f"    {s}")
    print()
    print("Review each site: confirm V→MTE3 sync is missing, then add")
    print("  `SetWaitFlag<HardEvent::V_MTE3>()` or `SetFlag/WaitFlag<HardEvent::V_MTE3>`")
    print("  between the V-pipe op and the MTE3 DataCopy. See OL-193 in KB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
