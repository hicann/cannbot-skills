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

"""ATK loader — read upstream CANN ops-nn ATK JSON, sample S/M/L cases.

Fallback chain (highest authority first):
  1. ATK JSON  (tests/st/aclnn<Op>/atk_*.json)            ← primary, this loader
  2. aclnn doc (docs/aclnn<Op>.md SUPPORTED_DTYPES table) ← future
  3. design.md / <op>-test-cases.md (PR #103 path)        ← future
  4. inferred from Model.forward (aog-input-gen-builder)  ← existing fallback

Freshness gate (HARD requirement, per user 2026-05-15 directive):
  - cann/ops-nn working tree MUST be clean
  - cann/ops-nn MUST be at origin/master (no drift)
  - Loader fails fast if either condition violated — no silent stale-baseline.

Output format (per PR #103 ascendc-operator-performance-eval skill schema):
  - <op>_perf_cases.jsonl    -- one JSON object per line, {case_id, inputs, attrs}
  - <op>_case_provenance.json -- metadata: upstream_sha, atk_path, sampling_rule, dtype_matrix

Usage:
  python3 atk_loader.py --cann-root ~/workspace/cann/ops-nn \\
      --atk pooling/adaptive_avg_pool3d/tests/st/aclnnAdaptiveAvgPool3d/atk_aclnnAdaptiveAvgPool3d.json \\
      --op-name adaptive_avg_pool3d \\
      --out-dir output/a3_to_a5_port/src/kernels/adaptive_avg_pool3d \\
      --sample-per-dtype 2  # 2 cases per (dtype, scale-bucket) → ≤ 3*2*N_dtype cases
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# Scale buckets by element count (numel) of the LARGEST tensor in a case.
# Tuned per PR #103 layer-norm reference cases: S ~smoke / M ~typical / L ~stress.
SCALE_BUCKETS = {
    "S": (1, 64 * 1024),                # ≤ 64K elements
    "M": (64 * 1024, 4 * 1024 * 1024),  # 64K .. 4M
    "L": (4 * 1024 * 1024, math.inf),   # > 4M
}


def freshness_gate(cann_root: Path) -> dict:
    """Hard fail if cann/ops-nn is dirty OR drifted from origin/master.

    Returns provenance dict on success: {sha, fetched_at, is_clean, behind_count}.
    """
    if not (cann_root / ".git").exists():
        sys.exit(f"FATAL: {cann_root} is not a git repo")
    git_executable = str(Path(shutil.which("git") or "git").resolve())

    # 1. Clean working tree
    dirty = subprocess.run(
        [git_executable, "-C", str(cann_root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        sys.exit(
            f"FATAL: {cann_root} has uncommitted changes:\n{dirty}\n"
            f"ATK loader refuses to read a dirty upstream tree."
        )

    # 2. Behind origin/master? Fetch first to be sure.
    subprocess.run(
        [git_executable, "-C", str(cann_root), "fetch", "origin", "master"],
        check=True, capture_output=True,
    )
    behind = int(subprocess.run(
        [git_executable, "-C", str(cann_root), "rev-list", "--count", "HEAD..origin/master"],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    if behind > 0:
        sys.exit(
            f"FATAL: {cann_root} is {behind} commits behind origin/master.\n"
            f"Run `git -C {cann_root} pull --ff-only origin master` first, then re-invoke."
        )

    sha = subprocess.run(
        [git_executable, "-C", str(cann_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    return {
        "upstream_sha": sha,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_clean": True,
        "behind_count": 0,
    }


def case_numel(case: dict) -> int:
    """Return max tensor numel across all tensor inputs in the case."""
    max_n = 0
    for inp in case.get("inputs", []):
        # ATK has both dict-shaped and list-nested-attrs forms
        items = inp if isinstance(inp, list) else [inp]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tensor":
                continue
            shape = item.get("shape")
            if not isinstance(shape, list):
                continue
            n = 1
            for d in shape:
                if not isinstance(d, int) or d <= 0:
                    n = 0
                    break
                n *= d
            max_n = max(max_n, n)
    return max_n


def case_dtype(case: dict) -> str:
    """Primary tensor dtype = first required tensor input's dtype."""
    for inp in case.get("inputs", []):
        items = inp if isinstance(inp, list) else [inp]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tensor" and item.get("required"):
                return item.get("dtype", "unknown")
    return "unknown"


def bucket_of(numel: int) -> str:
    for label, (lo, hi) in SCALE_BUCKETS.items():
        if lo <= numel < hi:
            return label
    return "L"  # numel >= L lower bound


def sample_cases(atk_cases: list, per_dtype_per_bucket: int, seed: int = 42) -> list:
    """Stratified sample: at most `per_dtype_per_bucket` cases per (dtype, S/M/L).

    Preserves original ATK case_id. Drops cases with 0 numel (malformed).
    """
    by_key: dict = {}
    for c in atk_cases:
        n = case_numel(c)
        if n == 0:
            continue
        dt = case_dtype(c)
        bk = bucket_of(n)
        by_key.setdefault((dt, bk), []).append((c, n))

    sampled = []
    for _, lst in sorted(by_key.items()):
        if len(lst) <= per_dtype_per_bucket:
            picked = lst
        else:
            # Sort by numel; pick spread across the bucket (smallest, median, largest, ...)
            lst.sort(key=lambda x: x[1])
            if per_dtype_per_bucket == 1:
                picked = [lst[len(lst) // 2]]
            else:
                # Stratified pick across sorted numel
                idxs = [round(i * (len(lst) - 1) / (per_dtype_per_bucket - 1))
                        for i in range(per_dtype_per_bucket)]
                picked = [lst[i] for i in idxs]
        for c, n in picked:
            sampled.append({**c, "_scale_bucket": bucket_of(n), "_numel": n})
    return sampled


def to_perf_case(atk_case: dict, op_name: str) -> dict:
    """Convert one ATK case → PR #103 JSONL schema."""
    inputs = []
    for inp in atk_case.get("inputs", []):
        items = inp if isinstance(inp, list) else [inp]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tensor":
                inputs.append({
                    "name": item["name"],
                    "type": "tensor",
                    "dtype": item["dtype"],
                    "shape": item["shape"],
                    "range_values": item.get("range_values"),
                })
            elif item.get("type") in ("attr", "attrs"):
                inputs.append({
                    "name": item["name"],
                    "type": "attr",
                    "dtype": item.get("dtype"),
                    "value": item.get("range_values"),
                })
    return {
        "case_id": atk_case["id"],
        "op": op_name,
        "atk_aclnn_name": atk_case.get("aclnn_name"),
        "scale_bucket": atk_case["_scale_bucket"],
        "numel": atk_case["_numel"],
        "inputs": inputs,
        "is_boundary": bool(atk_case.get("is_boundary")),
        "atk_perf_key": (atk_case.get("standard") or {}).get("perf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cann-root", required=True, type=Path,
                    help="Path to cann/ops-nn repo (e.g., ~/workspace/cann/ops-nn)")
    ap.add_argument("--atk", required=True,
                    help="ATK path relative to cann-root")
    ap.add_argument("--op-name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sample-per-dtype", type=int, default=2,
                    help="Cases per (dtype × scale-bucket); default 2 → ≤ 6 × N_dtype total")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-freshness", action="store_true",
                    help="DANGER: skip clean+up-to-date gate. For testing only.")
    args = ap.parse_args()

    cann_root = args.cann_root.expanduser().resolve()
    if not args.skip_freshness:
        prov = freshness_gate(cann_root)
    else:
        prov = {"upstream_sha": "SKIPPED", "fetched_at_utc": "SKIPPED",
                "is_clean": "SKIPPED", "behind_count": "SKIPPED"}

    atk_path = (cann_root / args.atk).resolve()
    if not atk_path.is_file():
        sys.exit(f"FATAL: ATK not found: {atk_path}")
    with atk_path.open() as f:
        atk_cases = json.load(f)
    if not isinstance(atk_cases, list) or not atk_cases:
        sys.exit(f"FATAL: ATK has no cases: {atk_path}")

    sampled = sample_cases(atk_cases, args.sample_per_dtype, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / f"{args.op_name}_perf_cases.jsonl"
    prov_path = args.out_dir / f"{args.op_name}_case_provenance.json"

    with jsonl_path.open("w") as f:
        for c in sampled:
            f.write(json.dumps(to_perf_case(c, args.op_name)) + "\n")

    by_bucket = {"S": 0, "M": 0, "L": 0}
    by_dtype: dict = {}
    for c in sampled:
        by_bucket[c["_scale_bucket"]] += 1
        dt = case_dtype(c)
        by_dtype[dt] = by_dtype.get(dt, 0) + 1

    prov_full = {
        "case_source": "atk",
        "upstream_repo": "https://gitcode.com/cann/ops-nn",
        **prov,
        "atk_path_relative": args.atk,
        "atk_total_cases": len(atk_cases),
        "sampled_count": len(sampled),
        "sampling_rule": {
            "per_dtype_per_bucket": args.sample_per_dtype,
            "scale_buckets": {k: [v[0], None if v[1] == math.inf else v[1]]
                              for k, v in SCALE_BUCKETS.items()},
            "stratified": "across numel within each (dtype, bucket)",
            "seed": args.seed,
        },
        "case_distribution": {
            "by_scale_bucket": by_bucket,
            "by_dtype": by_dtype,
        },
        "schema_compat": "PR #103 ascendc-operator-performance-eval",
    }
    prov_path.write_text(json.dumps(prov_full, indent=2) + "\n")

    print(f"[atk-loader] {args.op_name}: {len(sampled)} cases sampled from {len(atk_cases)} ATK")
    print(f"  upstream_sha: {prov['upstream_sha'][:12]}")
    print(f"  by_bucket:    {by_bucket}")
    print(f"  by_dtype:     {by_dtype}")
    print(f"  jsonl:        {jsonl_path}")
    print(f"  provenance:   {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
