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
import logging
import os
import re
import subprocess
import sys


PLOG_DIR = os.path.expanduser("~/ascend/log/debug/plog")
LOGGER = logging.getLogger("tilingkey_single")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


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
    torch_npu_check = "import torch_npu; raise SystemExit(0 if torch_npu.npu.is_available() else 1)"
    result = subprocess.run(
        [sys.executable, "-c", torch_npu_check],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return True
    torch_check = "import torch; raise SystemExit(0 if torch.npu.is_available() else 1)"
    result = subprocess.run(
        [sys.executable, "-c", torch_check],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def run_single_case(test_file, case_id, op_path):
    """Run a single pytest case with logging enabled. Returns stdout+stderr."""
    env = os.environ.copy()
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q", "--tb=short", "-k", case_id],
        cwd=op_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result


def extract_key_from_plog():
    """Extract first 'Tiling Key: N' from the latest plog file."""
    files = sorted(
        glob.glob(os.path.join(PLOG_DIR, "plog-*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    pattern = re.compile(r"Tiling Key:\s*(\d+)")
    for fpath in files:
        key = extract_key_from_file(fpath, pattern)
        if key:
            return key, fpath
    return "NOT_FOUND", None


def extract_key_from_file(fpath, pattern):
    for line in iter_file_lines(fpath):
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def iter_file_lines(fpath):
    try:
        with open(fpath, errors="ignore") as f:
            yield from f
    except OSError:
        return


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
    configure_logging()
    args = parse_args()
    op_name = deduce_op_name(args.op_path)
    log_dir = args.log_dir or os.path.join(args.op_path, "tests", "whitebox", "tilingkey_logs")

    if not check_npu():
        LOGGER.info("SKIPPED: NPU unavailable")
        sys.exit(0)

    try:
        test_file = find_test_file(args.op_path)
    except FileNotFoundError as exc:
        LOGGER.info("ERROR: %s", exc)
        sys.exit(1)
    clear_plog()

    result = run_single_case(test_file, args.case_id, args.op_path)
    key, plog_path = extract_key_from_plog()
    log_path = save_log(log_dir, op_name, args.case_id, plog_path)

    LOGGER.info("%s | tiling_key=%s", args.case_id, key)
    LOGGER.info("Log saved: %s", log_path)

    if key == "NOT_FOUND":
        sys.exit(1)


if __name__ == "__main__":
    main()
