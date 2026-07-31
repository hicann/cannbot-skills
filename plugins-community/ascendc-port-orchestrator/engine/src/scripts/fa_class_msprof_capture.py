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

"""FA-class msprof capture — Phase 1 skeleton (msprof wrapper, deferred to Phase 1.5).

Per design §6 Phase 1b + main R2 (`When` column finalize-gate #8). Wraps
msprof per-case to capture:
- kernel_binary_name actually dispatched (e.g. `flash_attention_score_s1s2_bn2gs1`
  vs `flash_attention_score_s1s2_bn2gs1_sab` vs `flash_attention_var_len_score`)
- (UB0, UB1, Block) host tiling tuple actually selected

Output writes `case_variant_map.json` conforming to `fa_class_schemas.CaseVariantMap`.

**Phase 1 skeleton scope**: this script provides the schema-conforming
wrapper + argparse / output path / msprof-availability detection +
graceful-degrade JSON when msprof is missing. Actual msprof invocation
+ output parsing is deferred to Phase 1.5 / Phase 3 v1.0 cold-start
when A3 NPU access is wired.

**Phase 1.5 follow-up (NOT IN THIS COMMIT)**:
- Locate msprof binary in CANN install or via `which msprof`
- Per-case: build msprof command line with --output + --aic-metrics
- Run msprof with the kernel binary + per-case input shapes
- Parse msprof output (CSV / JSON) → extract kernel_binary_name
- Parse host tiling debug log → extract (UB0, UB1, Block) tuple
- Write `case_variant_map.json` with one CaseVariantEntry per case

This skeleton writes `msprof_unavailable` status for ALL cases so that
downstream consumers (`fa_class_self_check.py` finalize gate #8) can
distinguish "msprof failure" from "kernel succeeded but msprof not run".

Usage:
  python3 src/scripts/fa_class_msprof_capture.py <workspace> [--out <path>] [--quiet]

Exit codes:
  0 — capture complete (either real or graceful-degrade)
  1 — workspace not found / unrecoverable error
  2 — usage error
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from fa_class_schemas import (  # noqa: E402
    CaseVariantEntry, CaseVariantMap, DispatchStatus,
)


def _detect_msprof() -> str | None:
    """Try to find msprof binary in PATH or CANN install."""
    p = shutil.which("msprof")
    if p:
        return p
    # Future Phase 1.5: also check /usr/local/Ascend/cann/tools/msprof
    return None


def _read_fixture(workspace: Path) -> list[int]:
    """Read case indices from workspace fixture file. Returns list of case_idx."""
    fixture_path = workspace / f"{workspace.name}.json"
    if not fixture_path.exists():
        # Try canonical_p2t_summary.json for count
        p2t = workspace / "canonical_p2t_summary.json"
        if p2t.exists():
            try:
                d = json.loads(p2t.read_text())
                return list(range(d.get("n_total", 0)))
            except Exception:
                return []
        return []
    # JSONL fixture (one JSON per line)
    cases = []
    for i, line in enumerate(fixture_path.read_text().splitlines()):
        if line.strip():
            cases.append(i)
    return cases


def capture_variants(workspace: Path) -> CaseVariantMap:
    """Phase 1 skeleton — writes msprof_unavailable for all cases."""
    msprof_bin = _detect_msprof()
    case_indices = _read_fixture(workspace)

    m = CaseVariantMap(
        op=workspace.name,
        workspace=str(workspace),
        captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        msprof_version=None,
    )

    if msprof_bin is None:
        # Graceful degrade — emit MSPROF_UNAVAILABLE for every case
        for case_idx in case_indices:
            m.cases.append(CaseVariantEntry(
                case_idx=case_idx,
                status=DispatchStatus.MSPROF_UNAVAILABLE,
                error="msprof binary not in PATH (Phase 1 skeleton — actual msprof "
                      "invocation deferred to Phase 1.5)",
            ))
        return m

    # msprof found but Phase 1 skeleton doesn't actually invoke it yet.
    # Emit a clear marker case so consumers know this script ran but didn't capture.
    m.msprof_version = "detected_but_not_invoked"
    for case_idx in case_indices:
        m.cases.append(CaseVariantEntry(
            case_idx=case_idx,
            status=DispatchStatus.MSPROF_UNAVAILABLE,
            error=f"msprof detected at {msprof_bin} but Phase 1 skeleton does "
                  "not invoke it; actual invocation deferred to Phase 1.5",
        ))
    return m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FA-class msprof per-case dispatch capture (Phase 1 skeleton)"
    )
    parser.add_argument(
        "workspace", type=Path,
        help="Path to workspace dir (e.g. workspace/3_FusionAttention)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: <workspace>/case_variant_map.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress non-error stdout output",
    )
    args = parser.parse_args(argv)

    if not args.workspace.exists():
        print(f"ERROR: workspace not found: {args.workspace}", file=sys.stderr)
        return 1

    out_path = args.out or (args.workspace / "case_variant_map.json")
    m = capture_variants(args.workspace)
    out_path.write_text(m.to_json(), encoding="utf-8")

    if not args.quiet:
        print(f"wrote {out_path}")
        print(f"  cases: {len(m.cases)}")
        print(f"  msprof_version: {m.msprof_version!r}")
        status_counts = {}
        for c in m.cases:
            status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1
        for s, n in sorted(status_counts.items()):
            print(f"    {n:>3}  {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
