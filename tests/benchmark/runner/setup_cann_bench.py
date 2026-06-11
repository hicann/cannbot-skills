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

"""cann-bench 仓库下载与管理。

评测运行前自动 clone cann-bench, 支持更新/重置。
可被其他模块 import, 也可独立运行。

用法:
  python runner/setup_cann_bench.py
  python runner/setup_cann_bench.py --update
  python runner/setup_cann_bench.py --reset
  python runner/setup_cann_bench.py --branch dev
"""

import argparse
import os
import shutil
import subprocess
import sys

CANN_BENCH_URL = "https://gitcode.com/cann/cann-bench.git"
DEFAULT_BRANCH = "master"

BENCHMARK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANN_BENCH_DIR = os.path.join(BENCHMARK_ROOT, "cann-bench")


def _run(cmd, cwd=None, check=True):
    return subprocess.run(
        cmd, cwd=cwd or CANN_BENCH_DIR,
        capture_output=True, text=True, check=check,
    )


def is_valid() -> bool:
    if not os.path.isdir(CANN_BENCH_DIR):
        return False
    git_dir = os.path.join(CANN_BENCH_DIR, ".git")
    if not os.path.exists(git_dir):
        return False
    result = _run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
    return result.returncode == 0


def clone(branch: str = DEFAULT_BRANCH) -> str:
    print(f"[setup] Cloning cann-bench ({branch}) ...")
    if os.path.exists(CANN_BENCH_DIR):
        shutil.rmtree(CANN_BENCH_DIR)
    subprocess.run(
        ["git", "clone", "--branch", branch, "--depth", "1",
         CANN_BENCH_URL, CANN_BENCH_DIR],
        check=True,
    )
    print(f"[setup] Clone complete: {CANN_BENCH_DIR}")
    return CANN_BENCH_DIR


def update(branch: str = DEFAULT_BRANCH) -> str:
    if not is_valid():
        return clone(branch)
    print(f"[setup] Updating cann-bench ({branch}) ...")
    _run(["git", "fetch", "origin", branch])
    _run(["git", "checkout", branch], check=False)
    _run(["git", "reset", "--hard", f"origin/{branch}"])
    print("[setup] Update complete.")
    return CANN_BENCH_DIR


def reset(branch: str = DEFAULT_BRANCH) -> str:
    print("[setup] Resetting cann-bench ...")
    return clone(branch)


def ensure_cann_bench(branch: str = DEFAULT_BRANCH, force_update: bool = False) -> str:
    if force_update:
        return update(branch)
    if is_valid():
        print(f"[setup] cann-bench already exists: {CANN_BENCH_DIR}")
        return CANN_BENCH_DIR
    return clone(branch)


def main():
    parser = argparse.ArgumentParser(description="cann-bench 仓库管理")
    parser.add_argument("--update", action="store_true",
                        help="拉取最新代码")
    parser.add_argument("--reset", action="store_true",
                        help="删除并重新克隆")
    parser.add_argument("--branch", default=DEFAULT_BRANCH,
                        help=f"分支名 (默认: {DEFAULT_BRANCH})")
    args = parser.parse_args()

    if args.reset:
        reset(args.branch)
    elif args.update:
        update(args.branch)
    else:
        ensure_cann_bench(args.branch)

    return 0


if __name__ == "__main__":
    sys.exit(main())
