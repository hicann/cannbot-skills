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

"""清理脚本：重置 cann-bench 和 results 到干净状态。

操作:
  1. cann-bench: git checkout -- . (还原 modified tracked files)
  2. cann-bench: git clean -fdx examples/ tasks/ (删除所有 untracked + gitignored)
  3. cann-bench: 确保在 master 分支
  4. 删除 results/ 目录

用法:
  python runner/cleanup.py --dry-run
  python runner/cleanup.py --force
"""

import argparse
import os
import shutil
import subprocess
import sys

BENCHMARK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (y/N): ")
    return resp.lower() in ("y", "yes")


def get_cann_bench_root():
    from setup_cann_bench import ensure_cann_bench
    return ensure_cann_bench()


def get_current_branch(cann_bench_root: str) -> str:
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cann_bench_root)
    return result.stdout.strip()


def show_what_will_be_cleaned(cann_bench_root: str):
    print("=" * 60)
    print("[DRY-RUN] 以下内容将被清理:")
    print("=" * 60)

    result = _run(["git", "diff", "--name-only"], cwd=cann_bench_root, check=False)
    modified = [l for l in result.stdout.strip().split("\n") if l]
    if modified:
        print(f"\n[cann-bench] 还原被修改的 tracked files ({len(modified)} 个):")
        for f in modified:
            print(f"  {f}")

    result = _run(["git", "ls-files", "--others", "--ignored", "--exclude-standard",
                   "examples/", "tasks/"], cwd=cann_bench_root, check=False)
    untracked = [l for l in result.stdout.strip().split("\n") if l]
    if untracked:
        print(f"\n[cann-bench] 删除 untracked + ignored 文件 ({len(untracked)} 个):")
        for f in untracked:
            print(f"  {f}")
    else:
        print("\n[cann-bench] 没有需要删除的 untracked 文件。")

    results_dir = os.path.join(BENCHMARK_ROOT, "results")
    if os.path.isdir(results_dir):
        print(f"\n[benchmark] 删除 results/ 目录")


def do_cleanup(cann_bench_root: str, force: bool = False):
    if not force:
        show_what_will_be_cleaned(cann_bench_root)
        if not confirm("\n确认执行清理?"):
            print("已取消。")
            return

    print("\n>>> 清理 cann-bench ...")

    branch = get_current_branch(cann_bench_root)
    print(f"  当前分支: {branch}")

    print("  还原 modified tracked files ...")
    _run(["git", "checkout", "--", "."], cwd=cann_bench_root, check=False)

    print("  删除 untracked + ignored 文件 ...")
    _run(["git", "clean", "-fdx", "examples/"], cwd=cann_bench_root, check=False)
    _run(["git", "clean", "-fdx", "tasks/"], cwd=cann_bench_root, check=False)

    if branch != "master":
        print(f"  切回 master 分支 ...")
        _run(["git", "checkout", "master"], cwd=cann_bench_root, check=False)

    results_dir = os.path.join(BENCHMARK_ROOT, "results")
    if os.path.isdir(results_dir):
        print(">>> 删除 results/ ...")
        shutil.rmtree(results_dir)

    result = _run(["git", "status", "--porcelain"], cwd=cann_bench_root, check=False)
    if result.stdout.strip():
        print(f"\n警告: cann-bench 仍有未跟踪文件:")
        print(result.stdout)
    else:
        print(f"\n✓ cann-bench 已完全干净。")

    print("清理完成。")


def main():
    parser = argparse.ArgumentParser(description="清理 cann-bench 和评测 workspace")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览, 不实际执行")
    parser.add_argument("--force", "-f", action="store_true",
                        help="跳过确认直接执行")
    args = parser.parse_args()

    cann_bench_root = get_cann_bench_root()

    if args.dry_run:
        show_what_will_be_cleaned(cann_bench_root)
        return 0

    do_cleanup(cann_bench_root, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
