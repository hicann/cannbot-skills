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
"""Compute tiling key coverage from plog and S2P2_param_def.json.

Usage:
    python compute_tilingkey_coverage.py \
        --log-path /path/to/{op_name}_full.log \
        --param-def /path/to/S2P2_param_def.json \
        --output-dir /path/to/output/

Output:
    {output_dir}/S6_tilingkey_coverage.json
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path


LOGGER = logging.getLogger("compute_tilingkey_coverage")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute tiling key coverage")
    parser.add_argument("--log-path", required=True, help="Path to full plog log file")
    parser.add_argument("--param-def", required=True, help="Path to S2P2_param_def.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for coverage JSON")
    return parser.parse_args()


def parse_tiling_key(line):
    match = re.search(r"Tiling Key:\s*(\d+)", line)
    return int(match.group(1)) if match else None


def extract_keys_from_plog(log_path):
    """Extract all tiling key values from plog file, in execution order."""
    path = Path(log_path)
    if not path.is_file():
        return []

    keys = []
    with path.open(errors="ignore") as f:
        for line in f:
            key = parse_tiling_key(line)
            if key is not None:
                keys.append(key)
    return keys


def compute_coverage(expected_keys, actual_keys):
    """Compute coverage statistics."""
    expected = set(expected_keys)
    actual = set(actual_keys)
    covered = expected & actual
    uncovered = expected - actual

    per_key = []
    for k in sorted(expected):
        per_key.append({
            "key": k,
            "hit": k in actual,
            "case_count": actual_keys.count(k) if k in actual else 0,
        })

    total = len(expected)
    covered_count = len(covered)
    rate = covered_count / total if total > 0 else 0.0

    return {
        "total_keys": total,
        "expected_keys": sorted(expected),
        "covered_keys": sorted(covered),
        "uncovered_keys": sorted(uncovered),
        "coverage_rate": round(rate, 4),
        "details": per_key,
    }


def compute_per_group_coverage(groups, actual_set):
    """逐 group 计算覆盖率，消费 groups[].group_tilingkeys。"""
    per_group = {}
    for g in groups:
        expected = g.get("group_tilingkeys", [])
        if not expected:
            continue
        expected_set = set(expected)
        covered = sorted(expected_set & actual_set)
        uncovered = sorted(expected_set - actual_set)
        per_group[g["id"]] = {
            "expected": sorted(expected),
            "covered": covered,
            "uncovered": uncovered,
            "rate": round(len(covered) / len(expected), 4) if expected else 0.0,
        }
    return per_group


def main():
    configure_logging()
    args = parse_args()

    if not os.path.exists(args.log_path):
        LOGGER.info("SKIPPED: no plog found at %s", args.log_path)
        sys.exit(0)

    if not os.path.exists(args.param_def):
        LOGGER.info("ERROR: param-def not found at %s", args.param_def)
        sys.exit(1)

    with open(args.param_def) as f:
        param_def = json.load(f)

    expected_keys = param_def.get("tiling_keys", [])
    if not expected_keys:
        LOGGER.info("WARNING: no tiling_keys field in param-def, coverage cannot be computed")
        sys.exit(0)

    actual_keys = extract_keys_from_plog(args.log_path)

    report = compute_coverage(expected_keys, actual_keys)
    report["operator"] = os.path.basename(args.log_path).replace("_full.log", "")
    report["total_cases_executed"] = len(actual_keys)

    groups = param_def.get("groups", [])
    report["per_group"] = compute_per_group_coverage(groups, set(actual_keys))

    output_path = os.path.join(args.output_dir, "S6_tilingkey_coverage.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    uncovered = report["uncovered_keys"]
    LOGGER.info(
        "TilingKey coverage: %.1f%% (%s/%s)",
        report["coverage_rate"] * 100,
        len(report["covered_keys"]),
        report["total_keys"],
    )
    if uncovered:
        LOGGER.info("Uncovered keys: %s", uncovered)


if __name__ == "__main__":
    main()
