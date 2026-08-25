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

"""隔离检查脚本：验证评测环境干净、无残留。

检查项:
  1. example 目录无残留生成物 (verify_isolation)
  2. tasks/*/dist/ 无跨轮次遗留交付 whl (check_task_dist_residue)
  3. 评测端口无残留 opencode serve 进程 (check_stale_serve)

用法:
  python runner/isolation_check.py
"""

import logging
import os
import signal
import subprocess
import sys
import time

_log = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SERVE_PORT = 4096

# 基线算子目录的兜底值 (当前 PINNED_COMMIT 对应的模板内置算子)。
# verify_isolation 优先按 git ls-files 自动发现 (与 HEAD 一致),
# git 不可用/无输出时才回退到此兜底。
BASELINE_OPS_FALLBACK = {
    "aclnn_launch_example": {"add", "sqrt", "_common"},
    "direct_launch_example": {"add", "sqrt"},
}


def _discover_baseline_ops(cann_bench_root: str, example_name: str) -> set:
    """按当前模板自动发现基线算子目录。

    用 git ls-files 取 examples/<name>/csrc/ops 下被跟踪的顶层目录 ——
    基线随 HEAD(pin) 自动对齐, 上游移除/新增内置示例算子不再需要手工同步;
    git 失败或无输出时回退 BASELINE_OPS_FALLBACK。
    """
    ops_rel = f"examples/{example_name}/csrc/ops"
    fallback = set(BASELINE_OPS_FALLBACK[example_name])
    result = subprocess.run(
        ["git", "ls-files", "--", ops_rel],
        cwd=cann_bench_root, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return fallback
    top = set()
    prefix = ops_rel + "/"
    for line in result.stdout.splitlines():
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix):]
        if "/" in rest:
            top.add(rest.split("/", 1)[0])
    return top or fallback


def _check_head_pinned(cann_bench_root: str) -> bool:
    """钉版本校验: HEAD 必须与 pin 一致, 上游漂移显式报警。"""
    from setup_cann_bench import resolve_pinned_commit
    pinned = resolve_pinned_commit()
    if not pinned:
        return True
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cann_bench_root, capture_output=True, text=True)
    head = result.stdout.strip() if result.returncode == 0 else ""
    if head == pinned:
        return True
    print("\n[隔离检查 FAILED] cann-bench HEAD 与 pin 不一致 (上游或本地已变更):")
    print(f"  期望 (pin):  {pinned}")
    print(f"  实际 (HEAD): {head or '未知'}")
    print("  处理: 运行 runner/setup_cann_bench.py --update 回退到 pin,")
    print("        或人工确认上游变更后更新 setup_cann_bench.PINNED_COMMIT。")
    return False


def _check_examples_tree_clean(cann_bench_root: str) -> list[str]:
    """基线签名: examples/ 树必须与 HEAD(pin) 完全一致。

    任何 tracked 修改 (M) 或未跟踪文件 (??) 都是污染 —— 例如 agent
    直接改写参考工程 (在 examples/<template>/csrc/ops/ 下新增算子、
    修改 cann_bench/__init__.py 注册接口)。build/dist 等 gitignore 目录
    不在 porcelain 视野内, 由 verify_isolation 的目录残留检查覆盖。
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "examples/"],
        cwd=cann_bench_root, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return []
    violations = ["  examples/ 树与 HEAD 不一致 (参考工程被修改/污染):"]
    violations.extend(f"    {ln}" for ln in lines)
    return violations


def verify_isolation(cann_bench_root: str = None) -> bool:
    if cann_bench_root is None:
        from setup_cann_bench import ensure_cann_bench
        cann_bench_root = ensure_cann_bench()

    examples_dir = os.path.join(cann_bench_root, "examples")
    violations = []

    # 钉版本校验 (上游漂移显式报警)
    if not _check_head_pinned(cann_bench_root):
        return False

    # 基线签名: examples/ 树完整性
    violations.extend(_check_examples_tree_clean(cann_bench_root))

    for example_name in BASELINE_OPS_FALLBACK:
        ops_dir = os.path.join(examples_dir, example_name, "csrc", "ops")
        if not os.path.isdir(ops_dir):
            continue
        baseline = _discover_baseline_ops(cann_bench_root, example_name)
        all_ops = {d for d in os.listdir(ops_dir)
                   if os.path.isdir(os.path.join(ops_dir, d))}
        extra = all_ops - baseline
        if extra:
            violations.append(f"  {example_name}/csrc/ops/ 存在非 baseline 目录: {extra}")

    for root, dirs, files in os.walk(examples_dir):
        for d in ["build", "build_py", "dist", "__pycache__"]:
            if d in dirs:
                violations.append(
                    f"  {os.path.relpath(os.path.join(root, d), cann_bench_root)} 残留")

    tgz = os.path.join(cann_bench_root, "examples.tgz")
    if os.path.exists(tgz):
        violations.append(f"  examples.tgz 残留 (可能包含生成源码)")

    if violations:
        print("\n[隔离检查 FAILED] 检测到以下污染源:")
        for v in violations:
            print(v)
        print("\n请先运行 cleanup.py 清理。")
        return False

    print("[隔离检查 OK] 所有 example 目录干净。")
    return True


def check_task_dist_residue(cann_bench_root: str = None) -> bool:
    """检查 tasks/*/dist/ 下是否存在跨轮次遗留的交付 whl。

    历史问题: tasks/{op}/dist/ 是算子交付位置 (prompt 要求把 whl 放到这里),
    但跨轮次评测会残留先前模型的交付 whl —— 一方面后续 agent 可直接解包
    获取参考实现信息 (接口签名/kernel 符号), 另一方面 delivery_complete
    会把残留 whl 误判为本轮交付, 绕过硬性门禁。评测前必须确认该目录干净。
    """
    if cann_bench_root is None:
        from setup_cann_bench import ensure_cann_bench
        cann_bench_root = ensure_cann_bench()

    tasks_dir = os.path.join(cann_bench_root, "tasks")
    if not os.path.isdir(tasks_dir):
        print("[task dist 检查 OK] 无 tasks 目录。")
        return True

    residue = []
    for root, dirs, files in os.walk(tasks_dir):
        if "dist" not in root.split(os.sep):
            continue
        for fn in files:
            if fn.endswith(".whl"):
                residue.append(os.path.relpath(os.path.join(root, fn),
                                               cann_bench_root))

    if residue:
        print("\n[隔离检查 FAILED] 检测到跨轮次遗留交付 whl (可能泄露参考实现):")
        for r in sorted(residue):
            print(f"  {r}")
        print("\n请先运行 cleanup.py 清理, 或由 run_eval.py 评测前自动清理。")
        return False

    print("[task dist 检查 OK] 无跨轮次遗留交付 whl。")
    return True


# ══════════════════════════════════════════════════════════════════════
#  残留 opencode serve 检查
# ══════════════════════════════════════════════════════════════════════
#
# 历史问题: 旧评测遗留的 opencode serve 进程会持续监听评测端口, 劫持新
# runner 的会话请求 —— session 被创建到旧 serve 的工作目录 (往往已被删
# 除), message POST 返回 500 ServeError, 评测直接失败。因此每次评测前
# 必须确保端口空闲。

def _serve_healthy(port: int) -> bool:
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/global/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _collect_listen_inodes(port: int) -> set[str]:
    """收集监听指定端口的 socket inode (扫描 /proc/net/tcp{,6})。"""
    target = f"{port:04X}"
    inodes = set()
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net) as f:
                lines = f.readlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            # parts[3] == "0A" 即 LISTEN 状态
            if (len(parts) > 9 and parts[3] == "0A"
                    and parts[1].rsplit(":", 1)[-1].upper() == target):
                inodes.add(parts[9])
    return inodes


def _scan_proc_fds_for_inodes(fd_dir: str, inodes: set) -> bool:
    """扫描 /proc/<pid>/fd, 命中任一监听 inode 即返回 True (无权读则 False)。"""
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        # PermissionError 是 OSError 子类, 一并捕获 (无权读 /proc/<pid>/fd)
        return False
    for fd in fds:
        try:
            link = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if link.startswith("socket:[") and link[8:-1] in inodes:
            return True
    return False


def _find_listen_pid(port: int) -> int | None:
    """通过 /proc 查找监听指定 TCP 端口的进程 PID (Linux)。"""
    inodes = _collect_listen_inodes(port)
    if not inodes:
        return None
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if _scan_proc_fds_for_inodes(f"/proc/{pid}/fd", inodes):
            return int(pid)
    return None


def check_stale_serve(port: int = DEFAULT_SERVE_PORT) -> bool:
    """检测并清理占用评测端口的残留 opencode serve 进程。"""
    if requests is None:
        return True
    if not _serve_healthy(port):
        print(f"[serve 检查 OK] 端口 {port} 空闲。")
        return True

    print(f"[serve 检查] 端口 {port} 被残留 serve 占用, 尝试优雅关闭 ...")
    try:
        requests.post(f"http://127.0.0.1:{port}/shutdown", timeout=3)
    except Exception as e:
        _log.debug("shutdown 请求异常 (忽略, 后续重试健康检查): %s", e)
    for _ in range(10):
        time.sleep(1)
        if not _serve_healthy(port):
            print("[serve 检查] 残留 serve 已优雅关闭。")
            return True

    pid = _find_listen_pid(port)
    if pid is None:
        print(f"[serve 检查 FAILED] 端口 {port} 仍被占用, 且无法定位监听进程。")
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmdline = ""
    if "opencode" not in cmdline:
        print(f"[serve 检查 FAILED] 端口 {port} 被非 opencode 进程占用: "
              f"PID {pid} ({cmdline[:80]}), 请手动处理。")
        return False

    print(f"[serve 检查] 优雅关闭超时, 强杀残留 serve: PID {pid} ({cmdline[:80]})")
    os.kill(pid, signal.SIGTERM)
    for _ in range(5):
        time.sleep(1)
        if not _serve_healthy(port):
            print("[serve 检查] 残留 serve 已清理。")
            return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(1)
    if not _serve_healthy(port):
        print("[serve 检查] 残留 serve 已清理。")
        return True
    print(f"[serve 检查 FAILED] 端口 {port} 仍被占用。")
    return False


if __name__ == "__main__":
    try:
        ok = verify_isolation()
        ok = check_task_dist_residue() and ok
        ok = check_stale_serve(int(os.environ.get("OPENCODE_SERVE_PORT",
                                                  DEFAULT_SERVE_PORT))) and ok
    except RuntimeError as e:  # 钉版本校验 fail-fast 报警
        print(str(e))
        ok = False
    sys.exit(0 if ok else 1)
