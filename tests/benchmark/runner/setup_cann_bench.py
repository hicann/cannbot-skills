#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""cann-bench 仓库下载与管理。

评测运行前自动 clone cann-bench, 支持更新/重置。
可被其他模块 import, 也可独立运行。

钉版本: 框架的 prompt/隔离基线/示例工程结构均以 PINNED_COMMIT 为已知良好基线。
上游 master 漂移不再静默进入评测环境 —— clone/update 后自动 checkout 到 pin,
HEAD 与 pin 不一致或 pin 无法获取时 fail-fast 显式报警。
确认上游变更后请更新 PINNED_COMMIT; 也可用 CANN_BENCH_COMMIT=none 显式关闭
钉版本 (不推荐)。

用法:
  python runner/setup_cann_bench.py
  python runner/setup_cann_bench.py --update
  python runner/setup_cann_bench.py --reset
  python runner/setup_cann_bench.py --branch dev
  python runner/setup_cann_bench.py --commit <sha>
"""

import argparse
import os
import shutil
import subprocess
import sys

CANN_BENCH_URL = "https://gitcode.com/cann/cann-bench.git"
DEFAULT_BRANCH = "master"

# 已知良好基线 commit (上游 master 2026-08-04, 950PR/9.1.0 适配版)。
# 框架的 prompt / 隔离基线 / 示例工程结构均以此为参照; 上游有新提交时请
# 人工确认示例工程结构与任务定义无漂移后, 再更新本 pin。
PINNED_COMMIT = "d0dafee75e30f4d46dddb9ca45742e2cb69a8ff6"

BENCHMARK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANN_BENCH_DIR = os.path.join(BENCHMARK_ROOT, "cann-bench")


def _run(cmd, cwd=None, check=True):
    return subprocess.run(
        cmd, cwd=cwd or CANN_BENCH_DIR,
        capture_output=True, text=True, check=check,
    )


def resolve_pinned_commit(commit: str | None = None) -> str:
    """解析钉版 commit: 显式参数 > CANN_BENCH_COMMIT 环境变量 > PINNED_COMMIT。

    值为 "none"/"off"/空串 时返回 "" 表示关闭钉版本 (跟随分支最新, 不推荐)。
    """
    raw = commit if commit is not None else os.environ.get("CANN_BENCH_COMMIT")
    if raw is None:
        raw = PINNED_COMMIT
    raw = raw.strip()
    if raw.lower() in ("", "none", "off"):
        return ""
    return raw


def current_head() -> str:
    result = _run(["git", "rev-parse", "HEAD"], check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _has_commit(commit: str) -> bool:
    result = _run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                  check=False)
    return result.returncode == 0


def checkout_commit(commit: str, branch: str = DEFAULT_BRANCH) -> None:
    """将工作区切换到指定 commit (detached); 本地缺对象时按需 fetch。

    fetch 顺序: 浅 fetch 单 commit (省流量, 需服务端允许按 sha 抓取)
    → 全量 fetch 分支; 两者都拿不到则 fail-fast 显式报警。
    """
    if not _has_commit(commit):
        print(f"[setup] 本地无 commit {commit[:12]}, 尝试浅 fetch ...")
        result = _run(["git", "fetch", "--depth", "1", "origin", commit],
                      check=False)
        if result.returncode != 0 or not _has_commit(commit):
            print(f"[setup] 浅 fetch 失败, 尝试全量 fetch {branch} ...")
            _run(["git", "fetch", "origin", branch])
    if not _has_commit(commit):
        raise RuntimeError(
            f"[setup] 无法获取 pin commit {commit}: 上游可能已删除/改写该提交。\n"
            f"  请人工确认上游 cann-bench 变更后更新 setup_cann_bench.py 的 "
            f"PINNED_COMMIT, 或 CANN_BENCH_COMMIT=none 显式关闭钉版本 (不推荐)。")
    _run(["git", "checkout", "-f", commit])
    print(f"[setup] 已切换到 pin commit: {commit[:12]}")


def verify_commit(commit: str) -> None:
    """校验 HEAD 与 pin 一致; 不一致则 fail-fast 显式报警。"""
    if not commit:
        return
    head = current_head()
    if head != commit:
        raise RuntimeError(
            f"[ALARM] cann-bench HEAD 与 pin 不一致, 评测环境不可信!\n"
            f"  期望 (pin):  {commit}\n"
            f"  实际 (HEAD): {head or '未知'}\n"
            f"  处理: 运行 runner/setup_cann_bench.py --update 回退到 pin;"
            f" 若需跟随上游新版本, 请人工确认示例工程/任务定义无漂移后更新 "
            f"PINNED_COMMIT。")


def is_valid() -> bool:
    if not os.path.isdir(CANN_BENCH_DIR):
        return False
    git_dir = os.path.join(CANN_BENCH_DIR, ".git")
    if not os.path.exists(git_dir):
        return False
    result = _run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
    return result.returncode == 0


def clone(branch: str = DEFAULT_BRANCH, commit: str | None = None) -> str:
    print(f"[setup] Cloning cann-bench ({branch}) ...")
    if os.path.exists(CANN_BENCH_DIR):
        shutil.rmtree(CANN_BENCH_DIR)
    subprocess.run(
        ["git", "clone", "--branch", branch, "--depth", "1",
         CANN_BENCH_URL, CANN_BENCH_DIR],
        check=True,
    )
    print(f"[setup] Clone complete: {CANN_BENCH_DIR}")
    if commit:
        checkout_commit(commit, branch)
    return CANN_BENCH_DIR


def update(branch: str = DEFAULT_BRANCH, commit: str | None = None) -> str:
    if not is_valid():
        return clone(branch, commit)
    print(f"[setup] Updating cann-bench ({branch}) ...")
    _run(["git", "fetch", "origin", branch])
    _run(["git", "checkout", branch], check=False)
    _run(["git", "reset", "--hard", f"origin/{branch}"])
    if commit:
        checkout_commit(commit, branch)
    print("[setup] Update complete.")
    return CANN_BENCH_DIR


def reset(branch: str = DEFAULT_BRANCH, commit: str | None = None) -> str:
    print("[setup] Resetting cann-bench ...")
    return clone(branch, commit)


def ensure_cann_bench(branch: str = DEFAULT_BRANCH, force_update: bool = False,
                      commit: str | None = None) -> str:
    commit = resolve_pinned_commit(commit)
    if force_update:
        update(branch, commit)
    elif is_valid():
        print(f"[setup] cann-bench already exists: {CANN_BENCH_DIR}")
    else:
        clone(branch, commit)
    verify_commit(commit)
    return CANN_BENCH_DIR


def main():
    parser = argparse.ArgumentParser(description="cann-bench 仓库管理")
    parser.add_argument("--update", action="store_true",
                        help="拉取最新代码并回退到 pin commit")
    parser.add_argument("--reset", action="store_true",
                        help="删除并重新克隆")
    parser.add_argument("--branch", default=DEFAULT_BRANCH,
                        help=f"分支名 (默认: {DEFAULT_BRANCH})")
    parser.add_argument("--commit", default=None,
                        help="钉版 commit (默认读 CANN_BENCH_COMMIT 环境变量,"
                             " 否则用内置 PINNED_COMMIT; 传 'none' 关闭钉版本)")
    args = parser.parse_args()

    commit = resolve_pinned_commit(args.commit)
    try:
        if args.reset:
            reset(args.branch, commit)
        elif args.update:
            update(args.branch, commit)
        else:
            ensure_cann_bench(args.branch, commit=commit)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
