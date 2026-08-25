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
  python runner/run_eval.py -c config/eval_config_mini.yaml --model zhipuai-coding-plan/glm-5.2
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
from dataclasses import dataclass

try:
    import requests
except ImportError:
    requests = None

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isolation_check import verify_isolation, check_stale_serve, check_task_dist_residue
from setup_cann_bench import ensure_cann_bench, CANN_BENCH_DIR
from progress import EvalProgress
from report import generate_report

# ── 路径常量 ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OPS_CACHE_DIR = os.path.join(PROJECT_ROOT, "operators")

DEFAULT_PROMPT_TEMPLATE = "prompts/op_dev_prompt.txt"
# 默认参考工程与 config (eval_config_mini.yaml) 保持一致:
# 历史 DEFAULT_EXAMPLE=aclnn_launch_example 是框架初版遗留, 137e7c8 把
# config 全量切到 direct 时漏改 --all 路径, 导致两种启动方式模板分裂;
# 且此前所有成功交付均为 direct 扁平结构, aclnn 模板从未被验证过。
DEFAULT_EXAMPLE = "cann-bench/examples/direct_launch_example"
DEFAULT_WORKFLOW = "ops-direct-invoke"

# ── harness ─────────────────────────────────────────────────────────────
HARNESS_CHOICES = ["opencode", "claude", "kimi"]
HARNESS_DEFAULT = "opencode"

# ── opencode serve ────────────────────────────────────────────────────
SERVE_PORT = int(os.environ.get("OPENCODE_SERVE_PORT", "4096"))
SERVE_URL = f"http://127.0.0.1:{SERVE_PORT}"
OP_TIMEOUT = int(os.environ.get("OP_TIMEOUT", "21600"))
SERVE_RETRY = int(os.environ.get("SERVE_RETRY", "3"))
# 每算子最大尝试次数 (含首次), 评分/交付异常时重跑; 1 = 不重跑
OP_RETRY = int(os.environ.get("OP_RETRY", "1"))


def parse_model(model: str, harness: str = "opencode") -> tuple[str, str]:
    """解析 provider/model 格式 (如 zhipuai-coding-plan/glm-5.2)。

    非 opencode harness (claude/kimi) 不要求 provider/model 格式,
    直接返回 ("", model)。
    """
    if harness != "opencode":
        return "", model
    provider_id, sep, model_id = model.partition("/")
    if not sep or not provider_id or not model_id:
        raise ValueError(f"--model 格式应为 provider/model, 收到: {model!r}")
    return provider_id, model_id


def resolve_allowed_models(config: dict | None,
                           cli_allowed: str | None = None) -> list[str]:
    """解析模型白名单: CLI --allowed-models (逗号分隔) 优先于 config allowed_models。"""
    if cli_allowed:
        return [m.strip() for m in cli_allowed.split(",") if m.strip()]
    if config and config.get("allowed_models"):
        return [str(m).strip() for m in config["allowed_models"] if str(m).strip()]
    return []


def validate_model_allowed(model: str, allowed: list[str]):
    """白名单非空时校验模型在列表内, 否则放行。"""
    if allowed and model not in allowed:
        raise ValueError(
            f"模型 {model!r} 不在白名单内, 允许的模型: {', '.join(allowed)}")


def resolve_op_timeout(op: dict, config: dict | None = None) -> int:
    """解析单算子超时 (秒)。

    优先级: op.timeout (per-op) > config category_timeouts[category]
            > config default_timeout > OP_TIMEOUT 环境变量 (默认 21600)。
    """
    if op.get("timeout"):
        return int(op["timeout"])
    if config:
        category = op.get("category", "")
        cat_timeouts = config.get("category_timeouts") or {}
        if category and category in cat_timeouts:
            return int(cat_timeouts[category])
        if config.get("default_timeout"):
            return int(config["default_timeout"])
    return OP_TIMEOUT


def prompt_for_model(harness: str = "opencode",
                     allowed: list[str] | None = None) -> str:
    """--model 未提供时发起问询: 列出可用模型并等待用户输入。

    harness="opencode" 时运行 opencode models 列出模型;
    非 openencode harness (claude/kimi) 直接提示输入模型名。
    allowed 非空时仅接受白名单内的模型。
    """
    if not sys.stdin.isatty():
        raise ValueError(
            "必须通过 --model 指定评测模型"
            " (用 opencode models 查看可用模型)")
    if harness == "opencode":
        try:
            out = subprocess.run(["opencode", "models"],
                                 capture_output=True, text=True, timeout=15)
            models = [
                line.strip()
                for line in out.stdout.splitlines()
                if "/" in line
            ]
        except Exception:
            models = []
        if allowed:
            models = [m for m in models if m in allowed]
        if models:
            header = ("可用模型 (opencode models, 白名单内):" if allowed
                      else "可用模型 (opencode models):")
            print(header)
            for m in models:
                print(f"  {m}")
        if allowed:
            print("白名单 (allowed_models):")
            for m in allowed:
                print(f"  {m}")
        model_prompt = "请输入评测模型 (provider/model): "
    else:
        print(f"Harness: {harness}")
        if allowed:
            print("白名单模型:")
            for m in allowed:
                print(f"  {m}")
        model_prompt = f"请输入 {harness} 模型名称: "
    while True:
        model = input(model_prompt).strip()
        if not model:
            print("模型不能为空, 请重新输入。")
            continue
        if allowed and model not in allowed:
            print(f"模型 {model!r} 不在白名单内, 请从上述列表选择。")
            continue
        return model


class ServeManager:
    """管理 opencode serve 进程的生命周期。"""

    def __init__(self, port: int = SERVE_PORT, url: str = SERVE_URL,
                 model: str | None = None):
        self.port = port
        self.url = url
        self.model = model  # provider/model, None = opencode 默认 (上次选用)
        self._cwd: str | None = None
        self._proc: subprocess.Popen | None = None
        self._log_path: str | None = None
        self._log_fh: object | None = None

    @property
    def cwd(self) -> str | None:
        return self._cwd

    @property
    def log_path(self) -> str | None:
        """serve 日志文件路径 (供外部诊断读取, e.g. 启动失败时打印日志)。"""
        return self._log_path

    def ensure_running(self, cwd: str) -> bool:
        """确保 opencode serve 在指定 cwd 运行。"""
        if not requests:
            return False
        cwd_real = os.path.realpath(cwd)
        if self._reuse_or_reset_existing(cwd_real):
            return True
        self._takeover_external_if_any()
        return self._launch(cwd_real)

    def shutdown(self):
        """评测结束时清理 serve 进程。"""
        if self._proc is not None:
            print(f"  [SERVE] 关闭 serve (cwd={os.path.basename(self._cwd) if self._cwd else '?'}) ...")
            self._terminate_proc()
        self._close_log_fh()
        self._log_path = None

    # ── 私有辅助 ────────────────────────────────────────────────────────

    def _health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/global/health", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _write_model_config(self, cwd_real: str):
        """将指定模型写入工作目录 .opencode/opencode.json (合并已有配置)。

        opencode serve 不支持 -m 参数, 模型需通过项目级配置钉住;
        该配置同时对 cannbot 主 agent 和 task 分发的 subagent 生效。
        """
        if not self.model:
            return
        cfg_dir = os.path.join(cwd_real, ".opencode")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, "opencode.json")
        cfg = {}
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except (json.JSONDecodeError, OSError):
                cfg = {}
        if cfg.get("model") == self.model:
            return
        cfg["model"] = self.model
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  [SERVE] 模型写入 {cfg_path}: {self.model}")

    def _reuse_or_reset_existing(self, cwd_real: str) -> bool:
        """已有 serve 进程: 复用 / 因 cwd 变化或退出而重置。True = 已就绪。"""
        if self._proc is None:
            return False
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
        return False

    def _takeover_external_if_any(self):
        """无本地进程记录但端口已有外部 serve: 关闭以便接管。"""
        if self._proc is None and self._health_check():
            print(f"  [SERVE] 端口 {self.port} 已有外部 serve, 重新接管 ...")
            try:
                requests.post(f"{self.url}/shutdown", timeout=3)
            except Exception:
                pass
            time.sleep(2)

    def _launch(self, cwd_real: str) -> bool:
        """启动新的 serve 进程并等待就绪。"""
        print(f"  [SERVE] 启动 opencode serve --port {self.port} "
              f"(cwd={os.path.basename(cwd_real)}, model={self.model or 'default'}) ...")

        log_dir = os.path.join(RESULTS_DIR, ".serve_logs")
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(
            log_dir, f"serve_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._log_fh = open(self._log_path, "w")

        self._write_model_config(cwd_real)
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

def op_short_name(op_name: str) -> str:
    """算子短名 (用于 operators/ 缓存目录与 prompt 占位符)。

    路径形如 cann-bench/tasks/level{N}/{op} 时返回 level{N}_{op},
    避免不同 level 同名算子 (如 level2/top_k vs level3/top_k) 在
     operators/ 缓存与归档中互相覆盖; 其他形态退化为 basename。
    """
    parts = op_name.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2 and parts[-2].lower().startswith("level"):
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


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


def _copy_whl(src_dir: str, dst_dir: str) -> int:
    """复制 src_dir 下所有 .whl 到 dst_dir, 返回复制数量。"""
    if not os.path.isdir(src_dir):
        return 0
    copied = 0
    os.makedirs(dst_dir, exist_ok=True)
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".whl"):
            shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
            copied += 1
    return copied


def clean_task_dist_whl(op_name_abs: str) -> int:
    """清空算子任务目录 dist/ 下的遗留 .whl 交付件。

    历史问题: tasks/{op}/dist/ 是交付位置, 跨轮次评测会残留先前模型的
    交付 whl —— 一方面 agent 可直接解包获取参考实现信息 (接口签名/kernel
    符号), 另一方面 delivery_complete 会把残留 whl 误判为本轮交付,
    绕过硬性门禁 (与 operators/ 缓存残留同样的问题)。因此在每轮评测
    (含重跑) 开始前清空该目录, 要求 agent 本轮真实产出交付件。
    """
    if not op_name_abs:
        return 0
    dist_dir = os.path.join(op_name_abs, "dist")
    if not os.path.isdir(dist_dir):
        return 0
    removed = 0
    for fn in sorted(os.listdir(dist_dir)):
        if fn.endswith(".whl"):
            try:
                os.remove(os.path.join(dist_dir, fn))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  [CLEAN] 清理 {os.path.basename(os.path.dirname(dist_dir))}/dist "
              f"遗留 .whl × {removed}")
    return removed


def _persist_subdirs(src_root: str, dst_root: str, skip: tuple[str, ...] = (),
                     need_valid: bool = False, label: str = ""):
    """遍历 src_root 下子目录, 跳过 skip, (可选) 校验后覆盖拷入 dst_root。"""
    if not os.path.isdir(src_root):
        return
    for entry in sorted(os.listdir(src_root)):
        if entry in skip:
            continue
        src = os.path.join(src_root, entry)
        if not os.path.isdir(src):
            continue
        if need_valid and not _is_valid_op_dir(src):
            continue
        dst = os.path.join(dst_root, entry)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"  [PERSIST] {label}{entry} → {dst}")


def persist_op_code(work_dir: str, op_name: str, op_name_abs: str | None = None):
    """将隔离工作目录中的算子产物持久化到 operators/{short_name}/。

    持久化范围 (覆盖 CANNBot 工作流的全部产物):
      - example/csrc/ops/{op}/        算子源码 (kernel/plugin)
      - example/cann_bench/__init__.py 注册入口
      - example/tests/{op}/           算子测试 (跳过模板自带的 add/sqrt)
      - dist/*.whl                    交付件 (example/dist 与 cann-bench 算子目录 dist)
      - operators/{op}/docs/          DESIGN/PLAN/WALKTHROUGH/environment/perf 文档
      - operators/{op}/README.md      算子文档 (如有)
    """
    op_slug = op_short_name(op_name)
    iso_dir = os.path.join(work_dir, "example")

    # ── 算子源码 ──
    _persist_subdirs(
        os.path.join(iso_dir, "csrc", "ops"),
        os.path.join(OPS_CACHE_DIR, op_slug, "csrc", "ops"),
        skip=("add", "sqrt", "CMakeLists.txt"), need_valid=True)

    # ── 注册入口 ──
    init_src = os.path.join(iso_dir, "cann_bench", "__init__.py")
    if os.path.isfile(init_src):
        init_dst = os.path.join(OPS_CACHE_DIR, op_slug, "cann_bench", "__init__.py")
        os.makedirs(os.path.dirname(init_dst), exist_ok=True)
        shutil.copy2(init_src, init_dst)

    # ── 算子测试 ──
    _persist_subdirs(
        os.path.join(iso_dir, "tests"),
        os.path.join(OPS_CACHE_DIR, op_slug, "tests"),
        skip=("add", "sqrt"), label="tests/")

    # ── 交付件 .whl (build 产物 → example/dist; 按 prompt 要求 → cann-bench 算子目录 dist) ──
    whl_dst = os.path.join(OPS_CACHE_DIR, op_slug, "dist")
    n = _copy_whl(os.path.join(iso_dir, "dist"), whl_dst)
    if op_name_abs:
        n += _copy_whl(os.path.join(op_name_abs, "dist"), whl_dst)
    if n:
        print(f"  [PERSIST] {n} 个 .whl → {whl_dst}")

    # ── 设计/计划/串讲/性能文档 ──
    src_docs = os.path.join(work_dir, "operators", op_slug, "docs")
    if os.path.isdir(src_docs):
        dst_docs = os.path.join(OPS_CACHE_DIR, op_slug, "docs")
        if os.path.exists(dst_docs):
            shutil.rmtree(dst_docs)
        os.makedirs(os.path.dirname(dst_docs), exist_ok=True)
        shutil.copytree(src_docs, dst_docs)
        print(f"  [PERSIST] docs/ → {dst_docs}")

    # ── 算子 README ──
    src_readme = os.path.join(work_dir, "operators", op_slug, "README.md")
    if os.path.isfile(src_readme):
        readme_dst = os.path.join(OPS_CACHE_DIR, op_slug, "README.md")
        os.makedirs(os.path.dirname(readme_dst), exist_ok=True)
        shutil.copy2(src_readme, readme_dst)


def restore_op_code(work_dir: str, op_name: str):
    """从 operators/{short_name}/ 恢复算子产物到隔离工作目录 (persist 的逆操作)。"""
    op_slug = op_short_name(op_name)
    example_dir = os.path.join(work_dir, "example")

    # ── 算子源码 ──
    cache_ops_dir = os.path.join(OPS_CACHE_DIR, op_slug, "csrc", "ops")
    if os.path.isdir(cache_ops_dir):
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

    # ── 注册入口 ──
    cache_init = os.path.join(OPS_CACHE_DIR, op_slug, "cann_bench", "__init__.py")
    if os.path.isfile(cache_init):
        init_dst = os.path.join(example_dir, "cann_bench", "__init__.py")
        os.makedirs(os.path.dirname(init_dst), exist_ok=True)
        shutil.copy2(cache_init, init_dst)

    # ── 算子测试 ──
    cache_tests_dir = os.path.join(OPS_CACHE_DIR, op_slug, "tests")
    if os.path.isdir(cache_tests_dir):
        for entry in sorted(os.listdir(cache_tests_dir)):
            src_t = os.path.join(cache_tests_dir, entry)
            if not os.path.isdir(src_t):
                continue
            dst_t = os.path.join(example_dir, "tests", entry)
            if os.path.exists(dst_t):
                shutil.rmtree(dst_t)
            os.makedirs(os.path.dirname(dst_t), exist_ok=True)
            shutil.copytree(src_t, dst_t)

    # ── 设计/计划等文档 + README ──
    cache_docs = os.path.join(OPS_CACHE_DIR, op_slug, "docs")
    if os.path.isdir(cache_docs):
        dst_docs = os.path.join(work_dir, "operators", op_slug, "docs")
        if os.path.exists(dst_docs):
            shutil.rmtree(dst_docs)
        os.makedirs(os.path.dirname(dst_docs), exist_ok=True)
        shutil.copytree(cache_docs, dst_docs)

    cache_readme = os.path.join(OPS_CACHE_DIR, op_slug, "README.md")
    if os.path.isfile(cache_readme):
        shutil.copy2(cache_readme,
                     os.path.join(work_dir, "operators", op_slug, "README.md"))

    # ── 交付件 .whl ──
    # 重跑续作时恢复上次产出的 whl, 避免 agent 不再重新 build 导致
    # delivery_complete 误判交付不完整
    cache_dist = os.path.join(OPS_CACHE_DIR, op_slug, "dist")
    if os.path.isdir(cache_dist):
        n = _copy_whl(cache_dist, os.path.join(example_dir, "dist"))
        if n:
            print(f"  [RESTORE] {n} 个 .whl → example/dist/")


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


def scan_all_ops(cann_bench_root: str,
                 config_example_map: dict | None = None) -> list[dict]:
    """扫描 tasks/levelN/* 自动发现算子。

    example_path 与 config 行为一致: config 中显式配置过该算子的
    example_path 则沿用, 未配置的算子回退 DEFAULT_EXAMPLE。
    """
    config_example_map = config_example_map or {}
    ops = []
    pattern = os.path.join(cann_bench_root, "tasks", "level*", "*")
    for op_path in sorted(glob.glob(pattern)):
        if not os.path.isdir(op_path):
            continue
        parts = op_path.split(os.sep)
        level, op_name = parts[-2], parts[-1]
        proto_path = os.path.join(op_path, "proto.yaml")
        op_key = f"cann-bench/tasks/{level}/{op_name}"
        ops.append({
            "op_name": op_key,
            "category": read_category(proto_path),
            "example_path": config_example_map.get(op_key, DEFAULT_EXAMPLE),
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


def build_prompt(template_path: str, op: dict,
                 example_path: str | None = None) -> str:
    """渲染提示词模板。

    example_path 显式覆盖 op["example_path_abs"] (评测主流程在重试循环中
    把 prompt 的 {example_path} 指向隔离 work_dir/example, 但不应原地改写
    op 字典污染后续读取)。
    """
    with open(template_path, "r") as f:
        template = f.read()
    vals = defaultdict(str)
    vals["op_name"] = op["op_name_abs"]
    vals["example_path"] = example_path or op["example_path_abs"]
    vals["category"] = op.get("category", "")
    vals["op_short_name"] = op_short_name(op["op_name"])
    vals["cann_bench_root"] = op.get("cann_bench_root", "")
    return template.format_map(vals)


# ══════════════════════════════════════════════════════════════════════
#  隔离工作目录 — init.sh 安装 CANNBot workflow
# ══════════════════════════════════════════════════════════════════════

def prepare_isolated_work_dir(op_name: str, op_op_path_abs: str,
                                example_path_abs: str, run_id: str,
                                workflow: str = DEFAULT_WORKFLOW,
                                harness: str = HARNESS_DEFAULT) -> str:
    """创建隔离工作目录并安装 CANNBot 工作流。

    根据 harness 选择 init.sh 的 tool 参数:
      opencode → .opencode/{skills,agents,workflows}/ + AGENTS.md
      claude   → .claude/{skills,agents,workflows}/   + CLAUDE.md
      kimi     → .claude/{skills,agents,workflows}/   + CLAUDE.md (复用 claude 结构)

    结构:
      work_dir/
        ├── .opencode/ 或 .claude/{skills,agents,workflows}/  ← init.sh 安装
        ├── AGENTS.md 或 CLAUDE.md                            ← init.sh 安装 (CANNBot)
        ├── asc-devkit/                                       ← init.sh 安装
        ├── operators/{name}/                                 ← 算子任务定义
        └── example/                                          ← 隔离的参考工程副本
    """
    # harness → init.sh tool 映射: kimi 复用 claude 的项目结构
    _harness_tool = {"opencode": "opencode", "claude": "claude", "kimi": "claude"}
    init_tool = _harness_tool.get(harness, "opencode")

    op_slug = op_name.replace("/", "_")
    work_dir = os.path.join(RESULTS_DIR, ".workdir", run_id, op_slug)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # 1) 运行 init.sh 安装 CANNBot workflow 到隔离目录
    workflow_init = os.path.join(
        _SKILLS_FORK, "plugins-official", workflow, "init.sh")
    if not os.path.isfile(workflow_init):
        print(f"  [WARN] workflow init.sh not found: {workflow_init}")
        workflow_init = WORKFLOW_INIT  # fallback to default

    if os.path.isfile(workflow_init):
        print(f"  [INIT] 安装 CANNBot 工作流到 {work_dir} "
              f"(harness={harness}, tool={init_tool}) ...")
        result = subprocess.run(
            ["bash", workflow_init, "project", init_tool, work_dir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [WARN] init.sh 返回 {result.returncode}")
            # 继续尝试 —— 可能部分安装仍然可用
    else:
        print(f"  [WARN] {workflow_init} 不存在, 跳过工作流安装")

    # 2) 复制算子任务定义到 operators/{short_name}/
    op_short = op_short_name(op_name)
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

    # 恢复持久化的算子产物 (源码/测试/文档, 如有)
    restore_op_code(work_dir, op_name)

    # 4) 仅 opencode harness: 将 AGENTS.md 软链接为 .opencode/agents/cannbot.md
    if harness == "opencode":
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


def delivery_complete(work_dir: str, op: dict,
                       op_attempt: int = 1) -> bool:
    """交付完整性检查: 交付件 .whl 已产出。

    检查处所随尝试次数递进:
      - 始终检查: 隔离目录 example/dist、cann-bench 算子目录 dist
        (要求 agent 本轮真实产出)
      - 仅重跑续作 (op_attempt > 1) 纳入: operators/ 持久化 dist
        (上次尝试已产出并持久化, 本次 agent 基于恢复的产物收尾, 未必重新 build)

    首次尝试不纳入 operators/ 缓存, 避免缓存残留 whl 绕过 prompt 的硬性门禁。
    """
    search_dirs = [os.path.join(work_dir, "example", "dist")]
    if op.get("op_name_abs"):
        search_dirs.append(os.path.join(op["op_name_abs"], "dist"))
    if op_attempt > 1 and op.get("op_name"):
        search_dirs.append(
            os.path.join(OPS_CACHE_DIR, op_short_name(op["op_name"]), "dist"))
    for d in search_dirs:
        if os.path.isdir(d) and any(f.endswith(".whl") for f in os.listdir(d)):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
#  算子执行
# ══════════════════════════════════════════════════════════════════════

def _open_sse_log(log_path: str | None):
    """打开 (按需创建父目录) SSE 事件日志文件, 失败返回 None。"""
    if not log_path:
        return None
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        return open(log_path, "a", encoding="utf-8")
    except OSError:
        return None


def _handle_sse_event(raw: str, log_fh):
    """解析并处理一条 SSE data 事件 (落盘 + 打印 subagent dispatch / session 状态)。"""
    try:
        evt = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        return
    if log_fh:
        log_fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
        log_fh.flush()
    etype = evt.get("type", "")
    tool_name = evt.get("toolName") or evt.get("tool_name", "")
    if etype == "tool-invocation" and tool_name == "task":
        inp = evt.get("input", {})
        agent = inp.get("subagent_type") or inp.get("subagentType", "?")
        print(f"    [SSE] subagent dispatch: {agent}")
    elif etype == "session.update" and evt.get("status") in ("busy", "idle"):
        print(f"    [SSE] session {evt.get('status')}")


def _close_quietly(log_fh):
    """best-effort 关闭日志文件句柄。"""
    if log_fh:
        try:
            log_fh.close()
        except Exception:
            pass


def _drain_sse(resp, stop_event: threading.Event, log_fh):
    """消费 SSE 事件流直到 stop_event 置位 (抽出降低 _sse_monitor 嵌套深度)。"""
    for line in resp.iter_lines(decode_unicode=True):
        if stop_event.is_set():
            break
        if line and line.startswith("data:"):
            _handle_sse_event(line[5:].strip(), log_fh)


def _sse_monitor(session_id: str, stop_event: threading.Event,
                 log_path: str | None = None, timeout: int = OP_TIMEOUT):
    """Background thread: stream SSE events, log subagent dispatch, 事件落盘 jsonl。"""
    log_fh = _open_sse_log(log_path)
    try:
        resp = requests.get(
            f"{SERVE_URL}/session/{session_id}/event",
            stream=True, timeout=timeout,
        )
        _drain_sse(resp, stop_event, log_fh)
    except Exception:
        pass
    finally:
        _close_quietly(log_fh)


class _SseGuard:
    """管理单次 run_via_serve 的 SSE 监控线程生命周期 (start/stop+join, 幂等)。"""

    def __init__(self):
        self.event = None
        self.thread = None

    def start(self, session_id: str, output_dir: str, req_timeout: int):
        self.event = threading.Event()
        self.thread = threading.Thread(
            target=_sse_monitor,
            args=(session_id, self.event,
                  os.path.join(output_dir, "sse_events.jsonl"), req_timeout),
            name=f"sse-{session_id}", daemon=True)
        self.thread.start()

    def stop(self):
        if self.event is not None:
            self.event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.event = None
        self.thread = None


def _new_serve_result(op_name: str, model: str | None, req_timeout: int) -> dict:
    return {
        "op_name": op_name,
        "start_time": time.time(),
        "status": "running",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "model": model or "default",
        "timeout_s": req_timeout,
    }


def _serve_abort_session(session_id: str | None) -> str | None:
    """best-effort abort 会话 (超时重试时), 返回 None。"""
    if not session_id:
        return None
    try:
        requests.post(f"{SERVE_URL}/session/{session_id}/abort", timeout=5)
    except Exception:
        pass
    return None


def _serve_delete_session(session_id: str | None) -> str | None:
    """best-effort delete 会话 (HTTP/连接错误重试时), 返回 None。"""
    if not session_id:
        return None
    try:
        requests.delete(f"{SERVE_URL}/session/{session_id}", timeout=5)
    except Exception:
        pass
    return None


def _serve_retry_or_finalize(result: dict, attempt: int, stop_sse,
                             status: str, stderr: str):
    """重试前的统一收尾: 停 SSE → 重试则 sleep+确保 serve; 末次则写入失败状态。"""
    stop_sse()
    if attempt < SERVE_RETRY:
        time.sleep(5)
        if _serve_mgr.cwd:
            _serve_mgr.ensure_running(_serve_mgr.cwd)
        return
    result["status"] = status
    result["stderr"] = stderr


def _collect_session_metrics(session_id: str, result: dict):
    """采集会话级累计 tokens/cost (含 subagent 消耗); 失败时保留消息级回退。"""
    try:
        sresp = requests.get(f"{SERVE_URL}/session/{session_id}", timeout=10)
        sresp.raise_for_status()
        sdata = sresp.json()
        sess_tokens = sdata.get("tokens")
        if isinstance(sess_tokens, dict) and "input" in sess_tokens:
            result["tokens"] = sess_tokens
        if isinstance(sdata.get("cost"), (int, float)):
            result["cost"] = sdata["cost"]
    except (requests.HTTPError, requests.ConnectionError,
            requests.Timeout, ValueError):
        pass


def _extract_actual_model(info: dict) -> str:
    """从消息响应 info 解析 serve 实际使用的模型 (兼容平铺/嵌套两种结构)。"""
    nested = info.get("model", {}) if isinstance(info.get("model"), dict) else {}
    provider = info.get("providerID") or nested.get("providerID", "")
    model = info.get("modelID") or nested.get("modelID", "")
    return f"{provider}/{model}" if model else ""


def run_via_serve(op_name: str, prompt: str, output_dir: str,
                  model: str | None = None,
                  timeout: int | None = None) -> dict:
    req_timeout = timeout if timeout is not None else OP_TIMEOUT
    result = _new_serve_result(op_name, model, req_timeout)
    sse = _SseGuard()
    last_err = None
    attempt = 0
    for attempt in range(1, max(1, SERVE_RETRY) + 1):
        try:
            _serve_post_message(prompt, output_dir, attempt, result, sse)
            sse.stop()
            break
        except requests.Timeout:
            last_err = _serve_timeout_msg(req_timeout, attempt)
            # abort 会先关闭事件流解除监控线程的读阻塞, 随后 join 才能生效;
            # session_id 取 result["session_id"] (_serve_post_message 建会话后即写入,
            # 早于可能超时的 message POST, 保证 abort 命中正确会话)
            _serve_abort_session(result.get("session_id"))
            _serve_retry_or_finalize(
                result, attempt, sse.stop, "timeout",
                f"全部 {SERVE_RETRY} 次尝试均超时 (每次 {req_timeout}s)")
        except (requests.HTTPError, requests.ConnectionError) as e:
            last_err = e
            print(f"  [SERVE] attempt {attempt} failed: {e}")
            _serve_delete_session(result.get("session_id"))
            _serve_retry_or_finalize(
                result, attempt, sse.stop, "error",
                f"All {SERVE_RETRY} attempts failed. Last: {last_err}")
        except Exception as e:
            sse.stop()
            result["status"] = "error"
            result["stderr"] = str(e)
            break

    sse.stop()  # 兜底: 任何退出路径都不遗留 SSE 监控线程
    result["attempts"] = attempt
    result["end_time"] = time.time()
    result["duration_s"] = result["end_time"] - result["start_time"]
    return result


def _serve_timeout_msg(req_timeout: int, attempt: int) -> str:
    """超时重试时的统一日志 (返回 last_err 文案)。"""
    print(f"  [SERVE] attempt {attempt}/{SERVE_RETRY} 超时 ({req_timeout}s)")
    return f"Timed out after {req_timeout}s"


def _serve_post_message(prompt: str, output_dir: str, attempt: int,
                        result: dict, sse: _SseGuard) -> str:
    """单次尝试: 建会话 → 起 SSE → 发消息 → 采集指标; 成功填 result, 返回 session_id。

    op_name / req_timeout / model 从 result 读取 (避免长参数列表)。
    """
    op_name = result["op_name"]
    req_timeout = result["timeout_s"]
    model = result.get("model")
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

    sse.start(session_id, output_dir, req_timeout)

    payload = {"agent": "cannbot",
               "parts": [{"type": "text", "text": prompt}]}
    # "default" = 未指定模型的哨兵 (真模型需 provider/model 格式, 不可能等于 "default")
    if model and model != "default":
        provider_id, model_id = parse_model(model)
        payload["model"] = {"providerID": provider_id, "modelID": model_id}
    resp = requests.post(
        f"{SERVE_URL}/session/{session_id}/message",
        json=payload, timeout=req_timeout,
    )
    resp.raise_for_status()

    data = resp.json()
    text_parts = [
        p["text"]
        for p in data.get("parts", [])
        if p.get("type") == "text"
    ]
    result["stdout"] = "\n".join(text_parts)[-50000:]
    result["status"] = "success"
    result["returncode"] = 0

    # 消息响应的 info.tokens 只覆盖最后一条 assistant 消息, 不含 task 工具分发的
    # subagent 消息; 任务完成后的真实消耗以会话级累计口径为准 (GET /session/{id})
    info = data.get("info", {})
    result["tokens"] = info.get("tokens", {})
    result["cost"] = info.get("cost", 0)
    _collect_session_metrics(session_id, result)
    actual = _extract_actual_model(info)
    if actual:
        result["model_actual"] = actual
    return session_id


def run_via_pipe(op_name: str, prompt: str, output_dir: str,
                  model: str | None = None,
                  timeout: int | None = None,
                  harness: str = HARNESS_DEFAULT,
                  work_dir: str | None = None) -> dict:
    req_timeout = timeout if timeout is not None else OP_TIMEOUT
    result = {
        "op_name": op_name,
        "start_time": time.time(),
        "status": "running",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "model": model or "default",
        "timeout_s": req_timeout,
        "harness": harness,
    }

    try:
        prompt_file = os.path.join(output_dir, "prompt.txt")
        if not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"prompt.txt not found: {prompt_file}")

        if harness == "opencode":
            cmd = ["opencode", "run",
                   "请按照附件中的指令完成算子开发任务。"]
            if model:
                cmd += ["-m", model]
            with open(prompt_file) as stdin_fh:
                proc = subprocess.run(
                    cmd,
                    cwd=output_dir,
                    stdin=stdin_fh,
                    capture_output=True, text=True,
                    timeout=req_timeout,
                )
        elif harness == "claude":
            # claude 在 work_dir 下运行: cwd 内含 .claude/ + CLAUDE.md, 自动加载
            # CANNBot 多 Agent 工作流; 若用 output_dir (results/{op}, 无 .claude 配置)
            # 等于以裸 agent 运行, 完全不加载工作流 (与 kimi 分支对齐)
            cmd = ["claude", "-p", prompt]
            if model:
                cmd += ["--model", model]
            cwd = work_dir or output_dir
            agent_file = os.path.join(cwd, "CLAUDE.md")
            if os.path.isfile(agent_file):
                cmd += ["--agent-file", agent_file]
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True, text=True,
                timeout=req_timeout,
            )
        elif harness == "kimi":
            # kimi 在 work_dir 下运行，通过 --agent-file 加载 CANNBot 工作流
            cmd = ["kimi", "-p", prompt]
            if model:
                cmd += ["--model", model]
            cwd = work_dir or output_dir
            agent_file = os.path.join(cwd, "CLAUDE.md")
            if os.path.isfile(agent_file):
                cmd += ["--agent-file", agent_file]
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True, text=True,
                timeout=req_timeout,
            )
        else:
            result["status"] = "error"
            result["stderr"] = f"Unknown harness: {harness}"
            result["end_time"] = time.time()
            result["duration_s"] = result["end_time"] - result["start_time"]
            return result

        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout[-50000:]
        result["stderr"] = proc.stderr[-10000:]
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["stderr"] = f"Timed out after {req_timeout}s"
    except FileNotFoundError:
        result["status"] = "error"
        result["stderr"] = f"{harness} CLI not found."

    result["end_time"] = time.time()
    result["duration_s"] = result["end_time"] - result["start_time"]
    return result


@dataclass
class OpRunConfig:
    """单算子执行的运行配置 (use_serve/model/timeout/harness 相关参数封装)。"""
    use_serve: bool = True
    model: str | None = None
    timeout: int | None = None
    harness: str = HARNESS_DEFAULT


def run_single_op(op_name: str, prompt: str, output_dir: str,
                  work_dir: str, cfg: OpRunConfig) -> dict:
    """执行单个算子开发任务。

    opencode harness: 优先 serve API (multi-agent), 回退 pipe (CLI).
    非 opencode harness (claude/kimi): 直接走 CLI pipe 模式.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    if (cfg.harness == "opencode" and cfg.use_serve
            and _serve_mgr.ensure_running(cwd=work_dir)):
        print(f"  [MODE] serve API (CANNBot multi-agent)")
        return run_via_serve(op_name, prompt, output_dir, model=cfg.model,
                             timeout=cfg.timeout)

    print(f"  [MODE] pipe ({cfg.harness} CLI)")
    return run_via_pipe(op_name, prompt, output_dir, model=cfg.model,
                        timeout=cfg.timeout, harness=cfg.harness,
                        work_dir=work_dir)


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

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量算子评测执行器")
    parser.add_argument("-c", "--config", help="评测配置文件路径")
    parser.add_argument("--all", action="store_true",
                        help="自动扫描 cann-bench 全量算子")
    parser.add_argument("--keep-isolated", action="store_true",
                        help="保留隔离目录 (调试用)")
    parser.add_argument("--skip-isolation-check", action="store_true",
                        help="跳过运行前的隔离检查")
    parser.add_argument("--cann-bench-branch", default="master",
                        help="cann-bench 分支")
    parser.add_argument("--cann-bench-commit", default=None,
                        help="cann-bench 钉版 commit (默认读 CANN_BENCH_COMMIT 环境变量,"
                             " 否则用 setup_cann_bench.PINNED_COMMIT; 传 'none' 关闭钉版本)")
    parser.add_argument("--update-cann-bench", action="store_true",
                        help="强制更新 cann-bench")
    parser.add_argument("--no-serve", action="store_true",
                        help="回退到 opencode run pipe 模式")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW,
                        help=f"工作流名称 (plugins-official/ 下的目录, 默认: {DEFAULT_WORKFLOW})")
    parser.add_argument("--model",
                        help="评测模型, 格式 provider/model (如 zhipuai-coding-plan/glm-5.2); "
                             "不提供时交互问询; 用 opencode models 查看可用模型")
    parser.add_argument("--allowed-models",
                        help="模型白名单, 逗号分隔 (覆盖 config allowed_models; "
                             "--all 模式下唯一指定方式)")
    parser.add_argument("--prompt-template",
                        help="提示词模板路径 (覆盖 config prompt_template; "
                             "如 prompts/op_dev_resume_prompt.txt)")
    parser.add_argument("--harness", default=HARNESS_DEFAULT,
                        choices=HARNESS_CHOICES,
                        help=f"评测 harness (默认: {HARNESS_DEFAULT})")
    return parser


def _resolve_template_path(args, config: dict | None) -> str:
    if args.prompt_template:
        return os.path.join(PROJECT_ROOT, args.prompt_template)
    if args.all:
        return os.path.join(PROJECT_ROOT, DEFAULT_PROMPT_TEMPLATE)
    return os.path.join(PROJECT_ROOT, config["prompt_template"])


def _load_ops(args, config: dict | None, cann_bench_root: str) -> list[dict]:
    """加载算子列表 (扫描或 config)。OPS_FILTER 过滤由 main 处理 (涉及早退语义)。"""
    if args.all:
        # example_path 与 config 对齐: config 显式配置过的算子沿用其
        # example_path, 未配置的回退 DEFAULT_EXAMPLE (--all 不强制 -c,
        # 无 config 时全部为 DEFAULT_EXAMPLE)
        cfg_example_map = {}
        if args.config:
            for op in (load_config(args.config) or {}).get("ops", []) or []:
                if op.get("op_name"):
                    cfg_example_map[op["op_name"]] = op.get("example_path")
        ops = scan_all_ops(cann_bench_root, cfg_example_map)
        print(f"自动扫描: 发现 {len(ops)} 个算子")
    else:
        ops = config["ops"]
    return ops


def _filter_ops(ops: list[dict]) -> tuple[list[dict], bool]:
    """按 OPS_FILTER 过滤; 返回 (过滤后 ops, 是否命中)。未设过滤时返回 (ops, True)。"""
    ops_filter = os.environ.get("OPS_FILTER", "")
    if not ops_filter:
        return ops, True
    filtered = [op for op in ops if ops_filter in op["op_name"]]
    if filtered:
        print(f"Filtered to {len(filtered)} operator(s) matching '{ops_filter}'")
    else:
        print(f"No operators match OPS_FILTER='{ops_filter}'")
    return filtered, bool(filtered)


def _run_isolation_checks(args, harness: str, cann_bench_root: str) -> bool:
    if args.skip_isolation_check:
        return True
    if not verify_isolation(cann_bench_root):
        return False
    if not check_task_dist_residue(cann_bench_root):
        return False
    # 仅 opencode harness: 评测前清理占用端口的残留 serve 进程
    if harness == "opencode" and not args.no_serve \
            and not check_stale_serve(SERVE_PORT):
        return False
    return True


@dataclass
class _EvalCtx:
    """单次评测的共享上下文 (供逐算子处理时避免长参数列表)。"""
    cann_bench_root: str
    config: dict | None
    template_path: str
    run_id: str
    harness: str
    model: str
    use_serve: bool
    keep_isolated: bool
    workflow: str = DEFAULT_WORKFLOW


def _attempt_op(op: dict, ctx: _EvalCtx) -> tuple[dict, bool]:
    """单算子的尝试循环: 重跑直到 success + 交付完整, 或耗尽 OP_RETRY。"""
    op = resolve_paths(op, ctx.cann_bench_root)
    op_name = op["op_name"]
    op_output_dir = os.path.join(RESULTS_DIR, op_name.replace("/", "_"))
    op_timeout = resolve_op_timeout(op, ctx.config)
    print(f"  [TIMEOUT] {op_timeout}s"
          f" (op={op.get('timeout', '-')}, category={op.get('category', '-')})")

    example_src = op["example_path_abs"]
    result, delivery_ok = None, False
    for op_attempt in range(1, max(1, OP_RETRY) + 1):
        if op_attempt > 1:
            print(f"  [RETRY] {op_name} 第 {op_attempt}/{OP_RETRY} 次尝试"
                  f" (上次: {result['status']}, 交付完整: {delivery_ok})")
        clean_task_dist_whl(op["op_name_abs"])
        work_dir = prepare_isolated_work_dir(
            op_name, op["op_name_abs"], example_src,
            ctx.run_id, workflow=ctx.workflow,
            harness=ctx.harness,
        )
        iso_example_path = os.path.join(work_dir, "example")
        prompt = build_prompt(ctx.template_path, op, example_path=iso_example_path)
        cfg = OpRunConfig(use_serve=ctx.use_serve, model=ctx.model,
                          timeout=op_timeout, harness=ctx.harness)
        result = run_single_op(op_name, prompt, op_output_dir,
                               work_dir=work_dir, cfg=cfg)
        result["op_attempt"] = op_attempt
        persist_op_code(work_dir, op_name, op_name_abs=op["op_name_abs"])
        delivery_ok = delivery_complete(work_dir, op, op_attempt)
        if ctx.keep_isolated:
            print(f"  [KEEP] {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
        if result["status"] == "success" and delivery_ok:
            break
    return result, delivery_ok


def _finalize_run(results: list[dict], ctx: _EvalCtx, workflow: str) -> int:
    """收尾: 清理 workdir/serve, 写 summary, 生成报告, 返回退出码。"""
    workdir_root = os.path.join(RESULTS_DIR, ".workdir")
    if os.path.isdir(workdir_root) and not ctx.keep_isolated:
        shutil.rmtree(workdir_root, ignore_errors=True)
    if ctx.harness == "opencode":
        _serve_mgr.shutdown()
    summary_path = os.path.join(RESULTS_DIR, "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(results, f, allow_unicode=True, default_flow_style=False)
    try:
        report_paths = generate_report(results, ctx.run_id,
                                       f"[{ctx.harness}] {ctx.model}",
                                       workflow, RESULTS_DIR)
        print(f"汇总报告: {report_paths['md']}")
        print(f"          {report_paths['html']}")
    except Exception as e:
        print(f"[WARN] 报告生成失败: {e}")
    print_summary(results)
    passed = all(r["status"] == "success" and r.get("delivery_ok", True)
                 for r in results)
    return 0 if passed else 1


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not args.config and not args.all:
        parser.error("请指定 -c/--config 或 --all")

    harness = args.harness
    config = None if args.all else load_config(args.config)
    allowed_models = resolve_allowed_models(config, args.allowed_models)
    model = (args.model or "").strip()
    try:
        if not model:
            model = prompt_for_model(harness=harness, allowed=allowed_models)
        parse_model(model, harness=harness)
        validate_model_allowed(model, allowed_models)
    except ValueError as e:
        parser.error(str(e))
    if harness == "opencode":
        _serve_mgr.model = model

    try:
        cann_bench_root = ensure_cann_bench(
            branch=args.cann_bench_branch, force_update=args.update_cann_bench,
            commit=args.cann_bench_commit)
    except RuntimeError as e:
        print(str(e))
        return 1
    template_path = _resolve_template_path(args, config)
    ops = _load_ops(args, config, cann_bench_root)
    ops, matched = _filter_ops(ops)
    if not matched:
        # OPS_FILTER 设了但无命中: 立即退出 (不做隔离检查/报告, 与原行为一致)
        return 0
    if not _run_isolation_checks(args, harness, cann_bench_root):
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = generate_run_id()
    print(f"评测 Run ID: {run_id}")
    print(f"Harness: {harness}")
    print(f"工作流: {args.workflow}")
    print(f"模型: {model}")
    if allowed_models:
        print(f"模型白名单: {', '.join(allowed_models)}")
    if OP_RETRY > 1:
        print(f"算子级重跑: 最多 {OP_RETRY} 次/算子 (OP_RETRY)")

    ctx = _EvalCtx(cann_bench_root=cann_bench_root, config=config,
                   template_path=template_path, run_id=run_id, harness=harness,
                   model=model, use_serve=(harness == "opencode" and not args.no_serve),
                   keep_isolated=args.keep_isolated, workflow=args.workflow)
    progress = EvalProgress(len(ops))
    icons = {"success": "OK", "failed": "FAIL",
             "timeout": "TIMEOUT", "error": "ERROR"}
    results = []
    for i, op in enumerate(ops):
        progress.start_op(i, op["op_name"])
        result, delivery_ok = _attempt_op(op, ctx)
        result["delivery_ok"] = delivery_ok
        save_result(result, os.path.join(RESULTS_DIR, op["op_name"].replace("/", "_")))
        results.append(result)
        suffix = "" if delivery_ok else " (交付不完整: 缺 .whl)"
        progress.write(f"  [{icons.get(result['status'], '?')}] "
                       f"{result['duration_s']:.0f}s{suffix}")
        progress.finish_op(result["status"], result["duration_s"])
    progress.close()

    return _finalize_run(results, ctx, args.workflow)


if __name__ == "__main__":
    sys.exit(main())
