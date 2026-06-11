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

"""归档脚本：将评测生成的代码和产物通过 git 分支归档到 remote。

流程:
  1. 在 cann-bench 中创建归档分支
  2. 提交所有生成的源码 + __init__.py + .whl
  3. 推送到 origin
  4. 切回原始分支, 删除本地归档分支 (磁盘零残留)

用法:
  python runner/archive_run.py --name run-001
  python runner/archive_run.py --dry-run
  python runner/archive_run.py --name run-001 --force
"""

import argparse
import os
import subprocess
import sys

BENCHMARK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_OPS = {
    "aclnn_launch_example": {"add", "sqrt", "mish", "_common"},
    "direct_launch_example": {"add", "sqrt"},
}


def _run(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def get_cann_bench_root():
    from setup_cann_bench import ensure_cann_bench
    return ensure_cann_bench()


def detect_generated_ops(cann_bench_root: str) -> dict[str, list[str]]:
    generated = {}
    examples_dir = os.path.join(cann_bench_root, "examples")
    for example_name, baseline in BASELINE_OPS.items():
        ops_dir = os.path.join(examples_dir, example_name, "csrc", "ops")
        if not os.path.isdir(ops_dir):
            continue
        all_ops = {d for d in os.listdir(ops_dir)
                   if os.path.isdir(os.path.join(ops_dir, d))}
        extra = all_ops - baseline
        if extra:
            generated[example_name] = sorted(extra)
    return generated


def detect_whl_files(cann_bench_root: str) -> list[str]:
    whl_files = []
    tasks_dir = os.path.join(cann_bench_root, "tasks")
    for root, dirs, files in os.walk(tasks_dir):
        for f in files:
            if f.endswith(".whl"):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, cann_bench_root)
                whl_files.append(rel_path)
    return sorted(whl_files)


def get_current_branch(cann_bench_root: str) -> str:
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cann_bench_root)
    return result.stdout.strip()


def archive_submodule(cann_bench_root: str, branch_name: str, remote: str,
                      generated_ops: dict[str, list[str]],
                      whl_files: list[str], dry_run: bool = False) -> bool:
    original_branch = get_current_branch(cann_bench_root)

    add_paths = []

    for example_name, ops in generated_ops.items():
        for op in ops:
            ops_src = os.path.join("examples", example_name, "csrc", "ops", op)
            add_paths.append(ops_src)

    for example_name in generated_ops:
        init_py = os.path.join("examples", example_name, "cann_bench", "__init__.py")
        if os.path.exists(os.path.join(cann_bench_root, init_py)):
            add_paths.append(init_py)

    add_paths.extend(whl_files)

    if not add_paths:
        print("没有检测到生成的产物, 无需归档。")
        return False

    print(f"原始分支: {original_branch}")
    print(f"归档分支: {branch_name}")
    print(f"\n将添加以下文件到归档分支:")
    for p in add_paths:
        print(f"  {p}")
    print(f"\n共 {len(add_paths)} 个路径")

    if dry_run:
        print("\n[DRY-RUN] 不会实际执行。")
        return True

    print(f"\n>>> 创建分支 {branch_name} ...")
    _run(["git", "checkout", "-b", branch_name], cwd=cann_bench_root)

    print(">>> 添加文件到暂存区 ...")
    for p in add_paths:
        _run(["git", "add", "-f", p], cwd=cann_bench_root)
    _run(["git", "add", "-f"] + add_paths, cwd=cann_bench_root)

    result = _run(["git", "status", "--short"], cwd=cann_bench_root, check=False)
    print(">>> 暂存区状态:")
    print(result.stdout)

    print(f">>> 提交 ...")
    commit_msg = f"eval({branch_name}): {len(generated_ops)} examples, {sum(len(v) for v in generated_ops.values())} operators generated"
    _run(["git", "commit", "-m", commit_msg], cwd=cann_bench_root)

    print(f">>> 推送到 {remote} ...")
    _run(["git", "push", remote, branch_name], cwd=cann_bench_root)

    print(f">>> 切回 {original_branch} ...")
    _run(["git", "checkout", original_branch], cwd=cann_bench_root)

    print(f">>> 删除本地分支 {branch_name} ...")
    _run(["git", "branch", "-D", branch_name], cwd=cann_bench_root)

    print(f"\n归档完成! 分支 {branch_name} 已推送到 {remote}, 本地已删除。")
    return True


def archive_results(branch_name: str, dry_run: bool = False):
    results_dir = os.path.join(BENCHMARK_ROOT, "results")
    if not os.path.isdir(results_dir) or not os.listdir(results_dir):
        print("\nresults/ 为空或不存在, 跳过归档。")
        return

    print(f"\n>>> results/ 目录包含评测结果, 请手动备份到: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="归档评测生成物到 git 远程分支")
    parser.add_argument("--name", default=None,
                        help="归档名称 (默认: eval-run-<auto> 自动检测序号)")
    parser.add_argument("--remote", default="origin",
                        help="远程仓库名称 (默认: origin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式, 不实际执行")
    parser.add_argument("--force", "-f", action="store_true",
                        help="跳过确认提示")
    args = parser.parse_args()

    cann_bench_root = get_cann_bench_root()

    if args.name is None:
        result = _run(["git", "branch", "-a"], cwd=cann_bench_root, check=False)
        existing = [line.strip() for line in result.stdout.split("\n")
                    if "eval-run-" in line]
        seq = 1
        while any(f"eval-run-{seq:03d}" in b for b in existing):
            seq += 1
        args.name = f"eval-run-{seq:03d}"

    generated_ops = detect_generated_ops(cann_bench_root)
    whl_files = detect_whl_files(cann_bench_root)

    if not generated_ops and not whl_files:
        print("没有检测到任何生成物 (生成的算子源码、.whl 文件)。")
        print("可能已经清理过, 或者还没有运行过评测。")
        return 0

    print("=" * 60)
    print("检测到的生成物:")
    print("=" * 60)
    for example, ops in generated_ops.items():
        print(f"\n[{example}] 生成的算子:")
        for op in ops:
            print(f"  - csrc/ops/{op}/")
        print(f"  - cann_bench/__init__.py (modified)")
    if whl_files:
        print(f"\n[.whl 交付件] ({len(whl_files)} 个):")
        for w in whl_files:
            print(f"  - {w}")
    print()

    if not args.force and not args.dry_run:
        resp = input("确认归档? (y/N): ")
        if resp.lower() not in ("y", "yes"):
            print("已取消。")
            return 0

    success = archive_submodule(cann_bench_root, args.name, args.remote,
                                generated_ops, whl_files, dry_run=args.dry_run)
    if success:
        archive_results(args.name, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
