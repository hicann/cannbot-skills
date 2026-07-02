#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Run a single pytest case and extract its tiling key from plog.

Usage:
    python tilingkey_single.py \
        --op-path /path/to/operator \
        --case-id case00000
        [--log-dir /path/to/logs]

Output:
    stdout: case ID and tiling key value
    log file: {log_dir}/{op_name}_{case_id}.log
"""
import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys

_logger = logging.getLogger(__name__)


PLOG_DIR = os.path.expanduser("~/ascend/log/debug/plog")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract tiling key for single case")
    parser.add_argument("--op-path", required=True, help="Operator source directory")
    parser.add_argument("--case-id", required=True, help="Test case ID (e.g. case00000)")
    parser.add_argument("--log-dir", default=None, help="Output directory for per-case log")
    return parser.parse_args()


def deduce_op_name(op_path):
    """Deduce operator name from directory name."""
    return os.path.basename(os.path.abspath(op_path))


def find_test_file(op_path):
    """Find S6_test_*.py file in tests/whitebox/."""
    whitebox = os.path.join(op_path, "tests", "whitebox")
    candidates = glob.glob(os.path.join(whitebox, "S6_test_*.py"))
    if not candidates:
        raise FileNotFoundError("no S6_test_*.py found in tests/whitebox/")
    return candidates[0]


def clear_plog():
    """Remove all plog log files to ensure clean capture."""
    if not os.path.exists(PLOG_DIR):
        return
    for f in glob.glob(os.path.join(PLOG_DIR, "plog-*.log")):
        try:
            os.remove(f)
        except OSError:
            pass


def check_npu():
    """Check if NPU is available."""
    result = subprocess.run(
        [sys.executable, "-c", "import torch_npu; print(torch_npu.npu.is_available())"],
        capture_output=True, text=True, timeout=30,
    )
    if "True" in result.stdout:
        return True
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.npu.is_available())"],
        capture_output=True, text=True, timeout=30,
    )
    return "True" in result.stdout


def run_single_case(test_file, case_id, op_path):
    """Run a single pytest case with logging enabled. Returns stdout+stderr."""
    env = os.environ.copy()
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q", "--tb=short", "--case-id", case_id],
        cwd=op_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result


_TILING_KEY_RE = re.compile(r"Tiling Key:\s*(\d+)")


def _read_lines(fpath):
    try:
        with open(fpath, errors="ignore") as f:
            return f.readlines()
    except OSError:
        return []


def _search_key_in_file(fpath):
    for line in _read_lines(fpath):
        m = _TILING_KEY_RE.search(line)
        if m:
            return m.group(1)
    return None


def extract_key_from_plog():
    """Extract first 'Tiling Key: N' from the latest plog file."""
    files = sorted(
        glob.glob(os.path.join(PLOG_DIR, "plog-*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    for fpath in files:
        key = _search_key_in_file(fpath)
        if key is not None:
            return key, fpath
    return "NOT_FOUND", None


def save_log(log_dir, op_name, case_id, plog_path):
    """Copy plog to {log_dir}/{op_name}_{case_id}.log."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{op_name}_{case_id}.log")
    if plog_path and os.path.exists(plog_path):
        with open(plog_path, errors="ignore") as src:
            with open(log_path, "w") as dst:
                dst.write(src.read())
    return log_path


def main():
    logging.basicConfig(format="%(message)s")
    args = parse_args()
    op_name = deduce_op_name(args.op_path)
    log_dir = args.log_dir or os.path.join(args.op_path, "tests", "whitebox", "tilingkey_logs")

    if not check_npu():
        _logger.info("SKIPPED: NPU unavailable")
        sys.exit(0)

    try:
        test_file = find_test_file(args.op_path)
    except FileNotFoundError as e:
        _logger.error("ERROR: %s", e)
        sys.exit(1)
    clear_plog()

    result = run_single_case(test_file, args.case_id, args.op_path)
    key, plog_path = extract_key_from_plog()
    log_path = save_log(log_dir, op_name, args.case_id, plog_path)

    _logger.info("%s | tiling_key=%s", args.case_id, key)
    _logger.info("Log saved: %s", log_path)

    if key == "NOT_FOUND":
        sys.exit(1)


if __name__ == "__main__":
    main()
