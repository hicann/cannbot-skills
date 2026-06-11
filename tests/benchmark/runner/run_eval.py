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

读取 eval config 或自动扫描 cann-bench, 遍历每个算子, 用提示词模板生成 prompt.txt,
通过 opencode 执行算子开发任务, 收集结果。

架构: runner → opencode (code generator)

用法:
  python tests/benchmark/runner/run_eval.py --all
  python tests/benchmark/runner/run_eval.py -c tests/benchmark/config/eval_config_mini.yaml
  OPS_FILTER="level2/softmax" python tests/benchmark/runner/run_eval.py -c tests/benchmark/config/eval_config_mini.yaml
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isolation_check import verify_isolation
from setup_cann_bench import ensure_cann_bench, CANN_BENCH_DIR

BENCHMARK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT_TEMPLATE = "prompts/op_dev_prompt.txt"
DEFAULT_EXAMPLE = "cann-bench/examples/aclnn_launch_example"
RESULTS_DIR = os.path.join(BENCHMARK_ROOT, "results")


def generate_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_isolated_example(example_path_abs: str, op_name: str, run_id: str) -> str:
    op_slug = op_name.replace("/", "_")
    iso_dir = os.path.join(RESULTS_DIR, ".isolated", run_id, op_slug)

    if os.path.exists(iso_dir):
        shutil.rmtree(iso_dir)

    shutil.copytree(
        example_path_abs,
        iso_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "build", "build_py",
            "dist",
            "__pycache__", "*.pyc",
            "*.egg-info",
            "_C.abi3.so",
            "CMakeCache.txt", "CMakeFiles", "Makefile",
            "cmake_install.cmake",
            "*.o", "*.os",
        ),
    )
    return iso_dir


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_category(proto_path: str) -> str:
    """从 proto.yaml 读取算子 category。"""
    if not os.path.isfile(proto_path):
        return ""
    with open(proto_path, "r") as f:
        proto = yaml.safe_load(f)
    return proto.get("operator", {}).get("category", "")


def scan_all_ops(cann_bench_root: str) -> list[dict]:
    """扫描 cann-bench/tasks/level*/ 目录, 自动构建算子列表。"""
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
    return op


def build_prompt(template_path: str, op: dict) -> str:
    with open(template_path, "r") as f:
        template = f.read()
    vals = defaultdict(str)
    vals["op_name"] = op["op_name_abs"]
    vals["example_path"] = op["example_path_abs"]
    vals["category"] = op.get("category", "")
    return template.format_map(vals)


def prepare_output_dir(output_dir: str, prompt: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    prompt_file = os.path.join(output_dir, "prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)

    return prompt_file


def run_single_op(op_name: str, prompt: str, output_dir: str) -> dict:
    prompt_file = prepare_output_dir(output_dir, prompt)

    result = {
        "op_name": op_name,
        "start_time": time.time(),
        "status": "running",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }

    try:
        proc = subprocess.run(
            ["opencode", "run", "请按照附件中的指令完成算子开发任务。",
             "-f", prompt_file],
            cwd=output_dir,
            capture_output=True,
            text=True,
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
        result["stderr"] = "opencode CLI not found. Please install OpenCode first."

    result["end_time"] = time.time()
    result["duration_s"] = result["end_time"] - result["start_time"]
    return result


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


def main():
    parser = argparse.ArgumentParser(description="批量算子评测执行器")
    parser.add_argument("-c", "--config",
                        help="评测配置文件路径 (如: config/eval_config_mini.yaml)")
    parser.add_argument("--all", action="store_true",
                        help="自动扫描 cann-bench 全量算子 (无需配置文件)")
    parser.add_argument("--keep-isolated", action="store_true",
                        help="保留每个算子的隔离 example 副本 (用于调试)")
    parser.add_argument("--skip-isolation-check", action="store_true",
                        help="跳过运行前的隔离检查")
    parser.add_argument("--cann-bench-branch", default="master",
                        help="cann-bench 分支 (默认: master)")
    parser.add_argument("--update-cann-bench", action="store_true",
                        help="强制更新 cann-bench")
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("请指定 -c/--config 或 --all")

    cann_bench_root = ensure_cann_bench(
        branch=args.cann_bench_branch,
        force_update=args.update_cann_bench,
    )

    if args.all:
        ops = scan_all_ops(cann_bench_root)
        template_path = os.path.join(BENCHMARK_ROOT, DEFAULT_PROMPT_TEMPLATE)
        print(f"自动扫描: 发现 {len(ops)} 个算子")
    else:
        config = load_config(args.config)
        template_path = os.path.join(BENCHMARK_ROOT, config["prompt_template"])
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

    results = []
    for i, op in enumerate(ops):
        op_name = op["op_name"]
        print(f"\n[{i+1}/{len(ops)}] 评测: {op_name}")

        op = resolve_paths(op, cann_bench_root)

        iso_dir = prepare_isolated_example(op["example_path_abs"], op_name, run_id)
        op["example_path_abs"] = iso_dir

        prompt = build_prompt(template_path, op)
        op_output_dir = os.path.join(RESULTS_DIR, op_name.replace("/", "_"))
        result = run_single_op(op_name, prompt, op_output_dir)
        save_result(result, op_output_dir)
        results.append(result)

        if args.keep_isolated:
            print(f"  [KEEP] 隔离副本保留在: {iso_dir}")
        else:
            shutil.rmtree(iso_dir, ignore_errors=True)

        icons = {"success": "OK", "failed": "FAIL", "timeout": "TIMEOUT", "error": "ERROR"}
        print(f"  [{icons[result['status']]}] {result['duration_s']:.0f}s")

    isolated_root = os.path.join(RESULTS_DIR, ".isolated")
    if os.path.isdir(isolated_root) and not args.keep_isolated:
        shutil.rmtree(isolated_root, ignore_errors=True)

    summary_path = os.path.join(RESULTS_DIR, "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(results, f, allow_unicode=True, default_flow_style=False)

    print_summary(results)
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
