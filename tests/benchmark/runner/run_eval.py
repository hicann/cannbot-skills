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

"""批量评测执行器。

读取 eval config 或自动扫描 cann-bench, 遍历每个算子, 用提示词模板生成 prompt,
通过 opencode serve REST API 执行算子开发任务, 收集结果。

架构:
  runner → 创建隔离工作目录 → init.sh 安装 CANNBot 工作流 →
  opencode serve (cwd=隔离目录) → CANNBot multi-agent (architect→developer→reviewer) →
  产出代码迁移到 cann-bench 格式 → 编译 .whl

用法:
  python runner/run_eval.py -c config/eval_config_mini.yaml
  python runner/run_eval.py --all
  OPS_FILTER="level1/exp" python runner/run_eval.py -c config/eval_config_mini.yaml
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isolation_check import verify_isolation
from setup_cann_bench import ensure_cann_bench, CANN_BENCH_DIR

# ── 路径常量 ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OPS_CACHE_DIR = os.path.join(PROJECT_ROOT, "operators")

DEFAULT_PROMPT_TEMPLATE = "prompts/op_dev_prompt.txt"
DEFAULT_EXAMPLE = "cann-bench/examples/aclnn_launch_example"
DEFAULT_WORKFLOW = "ops-direct-invoke"

# ── opencode serve ────────────────────────────────────────────────────
SERVE_PORT = int(os.environ.get("OPENCODE_SERVE_PORT", "4096"))
SERVE_URL = f"http://127.0.0.1:{SERVE_PORT}"
OP_TIMEOUT = int(os.environ.get("OP_TIMEOUT", "21600"))
SERVE_RETRY = int(os.environ.get("SERVE_RETRY", "3"))


class ServeManager:
    """管理 opencode serve 进程的生命周期。"""

    def __init__(self, port: int = SERVE_PORT, url: str = SERVE_URL):
        self.port = port
        self.url = url
        self._cwd: str | None = None
        self._proc: subprocess.Popen | None = None
        self._log_path: str | None = None
        self._log_fh: object | None = None

    @property
    def cwd(self) -> str | None:
        return self._cwd

    def _health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/global/health", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _close_log_fh(self):
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _terminate_proc(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            except Exception:
                pass
            self._proc = None
            self._cwd = None

    def ensure_running(self, cwd: str) -> bool:
        """确保 opencode serve 在指定 cwd 运行。"""
        if not requests:
            return False

        cwd_real = os.path.realpath(cwd)

        # ── 检查现有 serve ──
        if self._proc is not None:
            poll = self._proc.poll()
            if poll is not None:
                print(f"  [SERVE] 旧 serve 已退出 (exit={poll})")
                self._proc = None
                self._cwd = None
            elif self._cwd is not None and self._cwd != cwd_real:
                print(f"  [SERVE] cwd 变化, 重启: {os.path.basename(self._cwd)} → {os.path.basename(cwd_real)}")
                self._terminate_proc()
                self._close_log_fh()
            elif self._health_check():
                return True
            else:
                self._proc = None
                self._cwd = None
                self._close_log_fh()

        # ── 健康检查 (无本地进程记录时) ──
        if self._proc is None and self._health_check():
            print(f"  [SERVE] 端口 {self.port} 已有外部 serve, 重新接管 ...")
            try:
                requests.post(f"{self.url}/shutdown", timeout=3)
            except Exception:
                pass
            time.sleep(2)

        # ── 启动 serve ──
        print(f"  [SERVE] 启动 opencode serve --port {self.port} (cwd={os.path.basename(cwd_real)}) ...")

        log_dir = os.path.join(RESULTS_DIR, ".serve_logs")
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(
            log_dir, f"serve_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._log_fh = open(self._log_path, "w")

        self._proc = subprocess.Popen(
            ["opencode", "serve", "--port", str(self.port),
             "--hostname", "127.0.0.1"],
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            cwd=cwd_real,
        )
        self._cwd = cwd_real

        for _ in range(30):
            time.sleep(2)
            if self._health_check():
                print(f"  [SERVE] 就绪 (port {self.port}, cwd={os.path.basename(cwd_real)}, log={self._log_path})")
                return True
        print(f"  [SERVE] 启动失败, 日志: {self._log_path}")
        self._proc = None
        self._cwd = None
        self._close_log_fh()
        return False

    def shutdown(self):
        """评测结束时清理 serve 进程。"""
        if self._proc is not None:
            print(f"  [SERVE] 关闭 serve (cwd={os.path.basename(self._cwd) if self._cwd else '?'}) ...")
            self._terminate_proc()
        self._close_log_fh()
        self._log_path = None


_serve_mgr = ServeManager()

# ── 工作流安装 ────────────────────────────────────────────────────────
_SKILLS_FORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
WORKFLOW_ROOT = os.path.join(_SKILLS_FORK, "plugins-official", "ops-direct-invoke")
WORKFLOW_INIT = os.path.join(WORKFLOW_ROOT, "init.sh")


def generate_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ══════════════════════════════════════════════════════════════════════
#  算子源码持久化
# ══════════════════════════════════════════════════════════════════════

def _op_short_name(op_name: str) -> str:
    return os.path.basename(op_name)


def _is_valid_op_dir(op_dir: str) -> bool:
    """检查算子目录是否包含有效源码（排除示例/空目录）。"""
    total_lines = 0
    for root, _, files in os.walk(op_dir):
        for fn in files:
            if fn.endswith((".cpp", ".h", ".hpp")):
                try:
                    with open(os.path.join(root, fn)) as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    pass
    return total_lines >= 30


def persist_op_code(iso_dir: str, op_name: str):
    """将隔离副本中的算子源码持久化到 operators/{short_name}/。"""
    op_slug = _op_short_name(op_name)
    src_ops_dir = os.path.join(iso_dir, "csrc", "ops")
    if not os.path.isdir(src_ops_dir):
        return

    for entry in sorted(os.listdir(src_ops_dir)):
        if entry in ("add", "sqrt", "CMakeLists.txt"):
            continue
        src_op_dir = os.path.join(src_ops_dir, entry)
        if not os.path.isdir(src_op_dir):
            continue
        if not _is_valid_op_dir(src_op_dir):
            continue

        dst_op_dir = os.path.join(OPS_CACHE_DIR, op_slug, "csrc", "ops", entry)
        if os.path.exists(dst_op_dir):
            shutil.rmtree(dst_op_dir)
        os.makedirs(os.path.dirname(dst_op_dir), exist_ok=True)
        shutil.copytree(src_op_dir, dst_op_dir)
        print(f"  [PERSIST] {entry} → {dst_op_dir}")

    init_src = os.path.join(iso_dir, "cann_bench", "__init__.py")
    if os.path.isfile(init_src):
        init_dst = os.path.join(OPS_CACHE_DIR, op_slug, "cann_bench", "__init__.py")
        os.makedirs(os.path.dirname(init_dst), exist_ok=True)
        shutil.copy2(init_src, init_dst)


def restore_op_code(example_dir: str, op_name: str):
    """从 operators/{short_name}/ 恢复算子源码到 example 目录。"""
    op_slug = _op_short_name(op_name)
    cache_ops_dir = os.path.join(OPS_CACHE_DIR, op_slug, "csrc", "ops")
    if not os.path.isdir(cache_ops_dir):
        return

    dst_ops_dir = os.path.join(example_dir, "csrc", "ops")
    os.makedirs(dst_ops_dir, exist_ok=True)

    for entry in sorted(os.listdir(cache_ops_dir)):
        if entry == "CMakeLists.txt":
            continue
        src_d = os.path.join(cache_ops_dir, entry)
        dst_d = os.path.join(dst_ops_dir, entry)
        if not os.path.isdir(src_d):
            continue
        if not _is_valid_op_dir(src_d):
            continue

        if os.path.exists(dst_d):
            shutil.rmtree(dst_d)
        shutil.copytree(src_d, dst_d)


# ══════════════════════════════════════════════════════════════════════
#  配置 / 扫描 / prompt
# ══════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_category(proto_path: str) -> str:
    if not os.path.isfile(proto_path):
        return ""
    with open(proto_path, "r") as f:
        proto = yaml.safe_load(f)
    return proto.get("operator", {}).get("category", "")


def scan_all_ops(cann_bench_root: str) -> list[dict]:
    ops = []
    pattern = os.path.join(cann_bench_root, "tasks", "level*", "*")
    for op_path in sorted(glob.glob(pattern)):
        if not os.path.isdir(op_path):
            continue
        parts = op_path.split(os.sep)
        level, op_name = parts[-2], parts[-1]
        proto_path = os.path.join(op_path, "proto.yaml")
        ops.append({
            "op_name": f"cann-bench/tasks/{level}/{op_name}",
            "category": read_category(proto_path),
            "example_path": DEFAULT_EXAMPLE,
        })
    return ops


def resolve_paths(op: dict, cann_bench_root: str) -> dict:
    op_name = op["op_name"]
    example_path = op["example_path"]
    if op_name.startswith("cann-bench/"):
        op_name = op_name[len("cann-bench/"):]
    if example_path.startswith("cann-bench/"):
        example_path = example_path[len("cann-bench/"):]
    op["op_name_abs"] = os.path.join(cann_bench_root, op_name)
    op["example_path_abs"] = os.path.join(cann_bench_root, example_path)
    op["cann_bench_root"] = cann_bench_root
    return op


def build_prompt(template_path: str, op: dict) -> str:
    with open(template_path, "r") as f:
        template = f.read()
    vals = defaultdict(str)
    vals["op_name"] = op["op_name_abs"]
    vals["example_path"] = op["example_path_abs"]
    vals["category"] = op.get("category", "")
    vals["op_short_name"] = _op_short_name(op["op_name"])
    vals["cann_bench_root"] = op.get("cann_bench_root", "")
    return template.format_map(vals)


# ══════════════════════════════════════════════════════════════════════
#  隔离工作目录 — init.sh 安装 CANNBot workflow
# ══════════════════════════════════════════════════════════════════════

def prepare_isolated_work_dir(op_name: str, op_op_path_abs: str,
                                example_path_abs: str, run_id: str,
                                workflow: str = DEFAULT_WORKFLOW) -> str:
    """创建隔离工作目录并安装 CANNBot 工作流。

    结构:
      work_dir/
        ├── .opencode/{skills,agents,workflows}/  ← init.sh 安装
        ├── AGENTS.md                              ← init.sh 安装 (CANNBot)
        ├── asc-devkit/                            ← init.sh 安装
        ├── operators/{name}/                      ← 算子任务定义
        └── example/                               ← 隔离的参考工程副本
    """
    op_slug = op_name.replace("/", "_")
    work_dir = os.path.join(RESULTS_DIR, ".workdir", run_id, op_slug)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # 0) init.sh 要求目标目录已存在
    # (os.makedirs 已确保)

    # 1) 运行 init.sh 安装 CANNBot workflow 到隔离目录
    workflow_init = os.path.join(
        _SKILLS_FORK, "plugins-official", workflow, "init.sh")
    if not os.path.isfile(workflow_init):
        print(f"  [WARN] workflow init.sh not found: {workflow_init}")
        workflow_init = WORKFLOW_INIT  # fallback to default

    if os.path.isfile(workflow_init):
        print(f"  [INIT] 安装 CANNBot 工作流到 {work_dir} ...")
        result = subprocess.run(
            ["bash", workflow_init, "project", "opencode", work_dir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [WARN] init.sh 返回 {result.returncode}")
            # 继续尝试 —— 可能部分安装仍然可用
    else:
        print(f"  [WARN] {workflow_init} 不存在, 跳过工作流安装")

    # 2) 复制算子任务定义到 operators/{short_name}/
    op_short = _op_short_name(op_name)
    op_operators_dir = os.path.join(work_dir, "operators", op_short)
    os.makedirs(op_operators_dir, exist_ok=True)

    if os.path.isdir(op_op_path_abs):
        for fname in os.listdir(op_op_path_abs):
            src = os.path.join(op_op_path_abs, fname)
            dst = os.path.join(op_operators_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"  [COPY] 算子定义 → operators/{op_short}/")
    else:
        print(f"  [WARN] 算子目录不存在: {op_op_path_abs}")

    # 3) 复制隔离的参考工程到 example/
    iso_example = os.path.join(work_dir, "example")
    if os.path.exists(iso_example):
        shutil.rmtree(iso_example)
    shutil.copytree(
        example_path_abs, iso_example, symlinks=True,
        ignore=shutil.ignore_patterns(
            "build", "build_py", "dist",
            "__pycache__", "*.pyc", "*.egg-info",
            "_C.abi3.so",
            "CMakeCache.txt", "CMakeFiles", "Makefile",
            "cmake_install.cmake", "*.o", "*.os",
        ),
    )

    # 恢复持久化的算子源码 (如有)
    restore_op_code(iso_example, op_name)

    # 4) 将 AGENTS.md 软链接为 .opencode/agents/cannbot.md
    #    使 opencode serve 注册 cannbot 为命名 agent，支持 REST API "agent": "cannbot"
    #    必须指向 work_dir/AGENTS.md（init.sh sed 替换后的绝对路径版），而非
    #    PLUGIN_ROOT/AGENTS.md 原始版，避免相对路径在隔离工作目录中无法解析
    agents_md = os.path.join(work_dir, "AGENTS.md")
    agents_dir = os.path.join(work_dir, ".opencode", "agents")
    cannbot_link = os.path.join(agents_dir, "cannbot.md")
    if os.path.isfile(agents_md) and os.path.isdir(agents_dir):
        if os.path.lexists(cannbot_link):
            os.remove(cannbot_link)
        os.symlink(os.path.realpath(agents_md), cannbot_link)
        print(f"  [LINK] AGENTS.md → .opencode/agents/cannbot.md")

    print(f"  [WORKDIR] {work_dir}")
    return work_dir


# ══════════════════════════════════════════════════════════════════════
#  算子执行
# ══════════════════════════════════════════════════════════════════════

def _sse_monitor(session_id: str, stop_event: threading.Event):
    """Background thread: stream SSE events and log subagent dispatch."""
    try:
        resp = requests.get(
            f"{SERVE_URL}/session/{session_id}/event",
            stream=True, timeout=OP_TIMEOUT,
        )
        for line in resp.iter_lines(decode_unicode=True):
            if stop_event.is_set():
                break
            if not line:
                continue
            if line.startswith("data:"):
                try:
                    evt = json.loads(line[5:].strip())
                    etype = evt.get("type", "")
                    tool_name = evt.get("toolName") or evt.get("tool_name", "")
                    if etype == "tool-invocation" and tool_name == "task":
                        inp = evt.get("input", {})
                        agent = inp.get("subagent_type") or inp.get("subagentType", "?")
                        print(f"    [SSE] subagent dispatch: {agent}")
                    elif etype == "session.update":
                        status = evt.get("status", "")
                        if status in ("busy", "idle"):
                            print(f"    [SSE] session {status}")
                except (json.JSONDecodeError, AttributeError):
                    pass
    except Exception:
        pass


def _run_via_serve(op_name: str, prompt: str, output_dir: str) -> dict:
    result = {
        "op_name": op_name,
        "start_time": time.time(),
        "status": "running",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }

    session_id = None
    last_err = None

    for attempt in range(1, SERVE_RETRY + 1):
        try:
            op_slug = op_name.replace("/", "_")
            resp = requests.post(
                f"{SERVE_URL}/session",
                json={"title": f"eval-{op_slug}"},
                timeout=10,
            )
            resp.raise_for_status()
            session_id = resp.json()["id"]
            result["session_id"] = session_id
            print(f"  [session] {session_id} (attempt {attempt}/{SERVE_RETRY})")

            stop_sse = threading.Event()
            sse_thread = threading.Thread(
                target=_sse_monitor, args=(session_id, stop_sse), daemon=True)
            sse_thread.start()

            resp = requests.post(
                f"{SERVE_URL}/session/{session_id}/message",
                json={
                    "agent": "cannbot",
                    "parts": [{"type": "text", "text": prompt}],
                },
                timeout=OP_TIMEOUT,
            )
            resp.raise_for_status()

            stop_sse.set()
            sse_thread.join(timeout=5)

            data = resp.json()
            parts = data.get("parts", [])
            text_parts = [p["text"] for p in parts if p.get("type") == "text"]
            result["stdout"] = "\n".join(text_parts)[-50000:]
            result["status"] = "success"
            result["returncode"] = 0

            info = data.get("info", {})
            result["tokens"] = info.get("tokens", {})
            result["cost"] = info.get("cost", 0)
            break

        except requests.Timeout:
            result["status"] = "timeout"
            result["stderr"] = f"Timed out after {OP_TIMEOUT}s"
            if session_id:
                try:
                    requests.post(f"{SERVE_URL}/session/{session_id}/abort", timeout=5)
                except Exception:
                    pass
            break
        except (requests.HTTPError, requests.ConnectionError) as e:
            last_err = e
            print(f"  [SERVE] attempt {attempt} failed: {e}")
            if session_id and attempt < SERVE_RETRY:
                try:
                    requests.delete(f"{SERVE_URL}/session/{session_id}", timeout=5)
                except Exception:
                    pass
                session_id = None
            if attempt < SERVE_RETRY:
                time.sleep(5)
                if _serve_mgr.cwd:
                    _serve_mgr.ensure_running(_serve_mgr.cwd)
            else:
                result["status"] = "error"
                result["stderr"] = f"All {SERVE_RETRY} attempts failed. Last: {last_err}"
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = str(e)
            break

    result["end_time"] = time.time()
    result["duration_s"] = result["end_time"] - result["start_time"]
    return result


def _run_via_pipe(op_name: str, prompt: str, output_dir: str) -> dict:
    result = {
        "op_name": op_name,
        "start_time": time.time(),
        "status": "running",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }

    try:
        prompt_file = os.path.join(output_dir, "prompt.txt")
        if not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"prompt.txt not found: {prompt_file}")
        with open(prompt_file) as stdin_fh:
            proc = subprocess.run(
                ["opencode", "run",
                 "请按照附件中的指令完成算子开发任务。"],
                cwd=output_dir,
                stdin=stdin_fh,
                capture_output=True, text=True,
                timeout=21600,
            )
        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout[-50000:]
        result["stderr"] = proc.stderr[-10000:]
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["stderr"] = "Timed out after 6 hours"
    except FileNotFoundError:
        result["status"] = "error"
        result["stderr"] = "opencode CLI not found."

    result["end_time"] = time.time()
    result["duration_s"] = result["end_time"] - result["start_time"]
    return result


def run_single_op(op_name: str, prompt: str, output_dir: str,
                   work_dir: str, use_serve: bool = True) -> dict:
    """执行单个算子开发任务。"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    if use_serve and _serve_mgr.ensure_running(cwd=work_dir):
        print(f"  [MODE] serve API (CANNBot multi-agent)")
        return _run_via_serve(op_name, prompt, output_dir)

    print(f"  [MODE] pipe (fallback)")
    return _run_via_pipe(op_name, prompt, output_dir)


# ══════════════════════════════════════════════════════════════════════
#  结果
# ══════════════════════════════════════════════════════════════════════

def save_result(result: dict, output_dir: str):
    op_slug = result["op_name"].replace("/", "_")
    result_path = os.path.join(output_dir, f"{op_slug}.yaml")
    with open(result_path, "w") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)


def print_summary(results: list[dict]):
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout"))
    print(f"\n{'='*60}")
    print(f"评测完成: {total} 算子, 成功 {success}, 失败 {failed}, 异常 {errors}")
    print(f"{'='*60}")
    for r in results:
        if r["status"] != "success":
            print(f"  {r['status']:8s} | {r['op_name']}")


# ══════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="批量算子评测执行器")
    parser.add_argument("-c", "--config",
                        help="评测配置文件路径")
    parser.add_argument("--all", action="store_true",
                        help="自动扫描 cann-bench 全量算子")
    parser.add_argument("--keep-isolated", action="store_true",
                        help="保留隔离目录 (调试用)")
    parser.add_argument("--skip-isolation-check", action="store_true",
                        help="跳过运行前的隔离检查")
    parser.add_argument("--cann-bench-branch", default="master",
                        help="cann-bench 分支")
    parser.add_argument("--update-cann-bench", action="store_true",
                        help="强制更新 cann-bench")
    parser.add_argument("--no-serve", action="store_true",
                        help="回退到 opencode run pipe 模式")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW,
                        help=f"工作流名称 (plugins-official/ 下的目录, 默认: {DEFAULT_WORKFLOW})")

    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("请指定 -c/--config 或 --all")

    cann_bench_root = ensure_cann_bench(
        branch=args.cann_bench_branch,
        force_update=args.update_cann_bench,
    )

    if args.all:
        ops = scan_all_ops(cann_bench_root)
        template_path = os.path.join(PROJECT_ROOT, DEFAULT_PROMPT_TEMPLATE)
        print(f"自动扫描: 发现 {len(ops)} 个算子")
    else:
        config = load_config(args.config)
        template_path = os.path.join(PROJECT_ROOT, config["prompt_template"])
        ops = config["ops"]

    ops_filter = os.environ.get("OPS_FILTER", "")
    if ops_filter:
        ops = [op for op in ops if ops_filter in op["op_name"]]
        if not ops:
            print(f"No operators match OPS_FILTER='{ops_filter}'")
            return 0
        print(f"Filtered to {len(ops)} operator(s) matching '{ops_filter}'")

    if not args.skip_isolation_check:
        if not verify_isolation(cann_bench_root):
            return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = generate_run_id()
    print(f"评测 Run ID: {run_id}")
    print(f"工作流: {args.workflow}")

    use_serve = not args.no_serve

    results = []
    for i, op in enumerate(ops):
        op_name = op["op_name"]
        print(f"\n[{i+1}/{len(ops)}] 评测: {op_name}")

        op = resolve_paths(op, cann_bench_root)
        op_output_dir = os.path.join(RESULTS_DIR, op_name.replace("/", "_"))

        # ── 创建隔离工作目录 + 安装 CANNBot workflow ──
        work_dir = prepare_isolated_work_dir(
            op_name, op["op_name_abs"], op["example_path_abs"],
            run_id, workflow=args.workflow,
        )
        # prompt 中的 {example_path} 指向隔离目录内的 example
        op["example_path_abs"] = os.path.join(work_dir, "example")

        prompt = build_prompt(template_path, op)
        result = run_single_op(op_name, prompt, op_output_dir,
                                work_dir=work_dir, use_serve=use_serve)
        save_result(result, op_output_dir)
        results.append(result)

        # 持久化算子源码 (从隔离目录的 example/)
        persist_op_code(os.path.join(work_dir, "example"), op_name)

        if args.keep_isolated:
            print(f"  [KEEP] {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

        icons = {"success": "OK", "failed": "FAIL",
                 "timeout": "TIMEOUT", "error": "ERROR"}
        print(f"  [{icons[result['status']]}] {result['duration_s']:.0f}s")

    # 清理
    workdir_root = os.path.join(RESULTS_DIR, ".workdir")
    if os.path.isdir(workdir_root) and not args.keep_isolated:
        shutil.rmtree(workdir_root, ignore_errors=True)

    # 关闭 serve
    _serve_mgr.shutdown()

    summary_path = os.path.join(RESULTS_DIR, "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(results, f, allow_unicode=True, default_flow_style=False)

    print_summary(results)
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
