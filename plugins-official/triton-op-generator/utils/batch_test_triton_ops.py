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

"""
批量测试 level3/level4 最终版本 Triton 算子的精度与性能。

使用 triton-op-verifier 的 scripts/verify.py 与 scripts/benchmark.py：
  1. 优先使用每个算子目录下的 {op_name}_generated.py（Phase 5 最终代码）
  2. 若不存在 *_generated.py，则回退到 iter*/opt_iter* 目录，
     挑选精度验证全过（passed_cases == total_cases）且几何平均加速比最高的实现文件
对每个算子进行验证和性能测试，并将汇总结果写入 CSV。
"""

import argparse
import csv
import dataclasses
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time


SKILL_DIR = os.environ.get("TRITON_OP_VERIFIER_SKILL_DIR")
if SKILL_DIR:
    VALIDATE_SCRIPT = os.path.join(SKILL_DIR, "scripts", "validate_triton_impl.py")
    VERIFY_SCRIPT = os.path.join(SKILL_DIR, "scripts", "verify.py")
    BENCHMARK_SCRIPT = os.path.join(SKILL_DIR, "scripts", "benchmark.py")
else:
    VALIDATE_SCRIPT = None
    VERIFY_SCRIPT = None
    BENCHMARK_SCRIPT = None

logger = logging.getLogger("batch_test_triton_ops")


def _setup_logger():
    """配置 logger：优先复用 triton-op-verifier 的 _log_utils.setup_logger。

    SKILL_DIR 不可用或 _log_utils 导入失败时，退回等价的基础 stdout/stderr 分流配置
    （INFO/DEBUG → stdout，WARNING+ → stderr）。
    """
    if logger.handlers:
        return
    if SKILL_DIR:
        scripts_dir = os.path.join(SKILL_DIR, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        try:
            from _log_utils import setup_logger
            setup_logger(logger)
            return
        except ImportError:
            pass
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)
    logger.addHandler(stderr_handler)


_setup_logger()


def _make_timestamp():
    """本地时区感知时间戳，用于输出文件命名。"""
    return datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def walk_excluding_noise(entry_path):
    """遍历目录，排除 session_dir / __pycache__ 等噪音目录。"""
    for dirpath, dirnames, filenames in os.walk(entry_path):
        if "session_dir" in dirnames:
            dirnames.remove("session_dir")
        if "__pycache__" in dirnames:
            dirnames.remove("__pycache__")
        yield dirpath, dirnames, filenames


def find_task_file(entry_path):
    """在算子目录根下查找任务描述文件（*.py，排除生成/实现文件）。"""
    candidates = []
    for f in os.listdir(entry_path):
        if not f.endswith(".py"):
            continue
        if f.endswith("_generated.py") or "_triton_" in f:
            continue
        full = os.path.join(entry_path, f)
        if os.path.isfile(full):
            candidates.append(full)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    dir_name = os.path.basename(entry_path)
    # 优先匹配目录名
    for c in candidates:
        if os.path.basename(c) == f"{dir_name}.py":
            return c
    # 优先包含 class Model / get_inputs 的文件
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as f:
                content = f.read()
            if "class Model" in content or "def get_inputs" in content:
                return c
        except (OSError, UnicodeDecodeError) as e:
            logger.debug("读取候选任务文件失败，跳过: %s (%s)", c, e)
    return candidates[0]


def _match_pattern_files(dirpath, verify_dir, filenames, pat):
    """收集单个 pattern 在 iter 目录与 verify/ 子目录下的全部匹配文件。"""
    matches = []
    if pat.endswith(".py") and not pat.startswith("_"):
        # 如 generated_code.py / optimized_code.py
        full = os.path.join(dirpath, pat)
        if os.path.isfile(full):
            matches.append(full)
    # 在 verify/ 子目录里也可能有
    if os.path.isdir(verify_dir):
        for f in os.listdir(verify_dir):
            if f.endswith(pat):
                matches.append(os.path.join(verify_dir, f))
    # 当前目录下匹配后缀的实现文件
    for f in filenames:
        if f.endswith(pat):
            matches.append(os.path.join(dirpath, f))
    return matches


def _dedup(paths):
    """去重并保持顺序。"""
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def _collect_impl_files(dirpath, verify_dir, filenames, is_opt):
    """收集 iter/opt_iter 目录下的候选实现文件（去重、保持顺序）。"""
    if is_opt:
        # opt_iter: 优先 optimized，不要 baseline
        patterns = ["_triton_optimized.py", "optimized_code.py"]
    else:
        patterns = ["_triton_ascend_impl.py", "generated_code.py"]

    impl_files = []
    for pat in patterns:
        impl_files.extend(_match_pattern_files(dirpath, verify_dir, filenames, pat))
    return _dedup(impl_files)


def _load_candidate_results(dirpath, verify_dir, is_opt):
    """读取候选目录的 verify/perf 结果 JSON，返回 (passed, speedup, verify_json, perf_json)。"""
    if is_opt:
        verify_json = (
            os.path.join(verify_dir, "verify_result_optimized.json")
            if os.path.isdir(verify_dir)
            else None
        )
        perf_json = os.path.join(dirpath, "optimized_perf_result.json")
    else:
        verify_json = (
            os.path.join(verify_dir, "verify_result.json")
            if os.path.isdir(verify_dir)
            else None
        )
        perf_json = os.path.join(dirpath, "perf_result.json")

    verify_data = load_json(verify_json) if verify_json else None
    perf_data = load_json(perf_json) if perf_json and os.path.isfile(perf_json) else None

    passed = (
        verify_data is not None
        and "_load_error" not in verify_data
        and verify_data.get("passed_cases") == verify_data.get("total_cases")
        and verify_data.get("total_cases", 0) > 0
    )
    speedup = (
        perf_data.get("speedup_vs_torch")
        if perf_data and "_load_error" not in perf_data
        else None
    )
    return passed, speedup, verify_json, perf_json


def find_fallback_impl(entry_path):
    """当没有 *_generated.py 时，从 iter*/opt_iter* 目录中挑选精度全过且性能最好的实现。

    返回 dict:
      - gen_file: 选中的 triton 实现文件路径
      - op_name: 用于 verify.py 的算子名（目录名）
      - task_file: 任务描述文件路径
      - source_desc: 来源描述（如 output/iter_0/verify/...）
    或 None（未找到合适候选）
    """
    candidates = []
    for dirpath, _dirnames, filenames in walk_excluding_noise(entry_path):
        basename = os.path.basename(dirpath)
        if not (basename.startswith("iter") or basename.startswith("opt_iter")):
            continue

        is_opt = basename.startswith("opt_iter")
        verify_dir = os.path.join(dirpath, "verify")

        unique_impl_files = _collect_impl_files(dirpath, verify_dir, filenames, is_opt)
        if not unique_impl_files:
            continue

        passed, speedup, verify_json, perf_json = _load_candidate_results(dirpath, verify_dir, is_opt)
        for impl_file in unique_impl_files:
            candidates.append({
                "impl_file": impl_file,
                "passed": passed,
                "speedup": speedup,
                "dirpath": dirpath,
                "verify_json": verify_json,
                "perf_json": perf_json,
            })

    if not candidates:
        return None

    # 只保留精度全过的候选；按 speedup 降序，speedup 缺失视为 -inf
    passed_candidates = [c for c in candidates if c["passed"]]
    if not passed_candidates:
        return None

    def sort_key(c):
        return c["speedup"] if c["speedup"] is not None else float("-inf")

    best = max(passed_candidates, key=sort_key)

    task_file = find_task_file(entry_path)
    if not task_file:
        return None

    return {
        "gen_file": best["impl_file"],
        "op_name": os.path.basename(entry_path),
        "task_file": task_file,
        "task_dir": os.path.dirname(task_file),
        "source_desc": f"{os.path.relpath(best['dirpath'], entry_path)} (speedup={best['speedup']})",
    }


def _find_generated_op(root, entry, entry_path, generated_files):
    """基于 *_generated.py 构建算子信息；缺任务文件时告警并返回 None。"""
    # 优先选择直接位于 entry_path 下的生成代码，否则取最新
    direct_gen = [p for p, _ in generated_files if os.path.dirname(p) == entry_path]
    if direct_gen:
        gen_file = direct_gen[0]
    else:
        gen_file = max(generated_files, key=lambda x: x[1])[0]

    gen_basename = os.path.basename(gen_file)
    base_name = gen_basename[: -len("_generated.py")]

    # 查找与生成代码同 base_name 的任务描述文件
    task_file = None
    direct_task = os.path.join(entry_path, f"{base_name}.py")
    if os.path.isfile(direct_task):
        task_file = direct_task
    else:
        for dirpath, _dirnames, filenames in walk_excluding_noise(entry_path):
            if f"{base_name}.py" in filenames:
                task_file = os.path.join(dirpath, f"{base_name}.py")
                break

    if not task_file:
        logger.warning("[WARN] 未找到任务描述文件 %s.py，跳过: %s", base_name, entry_path)
        return None

    return {
        "level": os.path.basename(root),
        "dir": entry_path,
        "dir_name": entry,
        "op_name": base_name,
        "gen_file": gen_file,
        "task_file": task_file,
        "task_dir": os.path.dirname(task_file),
        "source_desc": gen_file,
    }


def _scan_entry(root, entry, entry_path):
    """扫描单个算子目录，返回算子信息 dict；不可测试时告警并返回 None。"""
    # 递归查找所有 *_generated.py
    generated_files = []
    for dirpath, _dirnames, filenames in walk_excluding_noise(entry_path):
        for f in filenames:
            if f.endswith("_generated.py"):
                full_path = os.path.join(dirpath, f)
                generated_files.append((full_path, os.path.getmtime(full_path)))

    if generated_files:
        return _find_generated_op(root, entry, entry_path, generated_files)

    # 回退：从 iter*/opt_iter* 中挑选精度全过且性能最好的实现
    fallback_info = find_fallback_impl(entry_path)
    if fallback_info:
        return {
            "level": os.path.basename(root),
            "dir": entry_path,
            "dir_name": entry,
            "op_name": fallback_info["op_name"],
            "gen_file": fallback_info["gen_file"],
            "task_file": fallback_info["task_file"],
            "task_dir": fallback_info["task_dir"],
            "source_desc": fallback_info["source_desc"],
        }

    logger.warning(
        "[WARN] 未找到生成代码 *_generated.py，也未找到 iter/opt_iter 回退候选: %s", entry_path
    )
    return None


def find_operators(root_dirs):
    """扫描目录，返回所有包含最终生成代码的算子信息。

    支持以下复杂布局：
      - {op_dir}/{op_name}_generated.py + {op_dir}/{op_name}.py
      - {op_dir}/{base_name}_generated.py，其中 base_name 与目录名不同
      - {op_dir}/op_{op_name}_{timestamp}_{rid}/{op_name}_generated.py
      - {op_dir}/output/iter_*/{generated_code.py | verify/*_triton_ascend_impl.py}
      - {op_dir}/output/opt_iter_*/{optimized_code.py | verify/*_triton_optimized.py}
      - 同一算子目录下存在多个历史工作目录时，按 mtime 取最新
    """
    ops = []
    for root in root_dirs:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            logger.warning("[WARN] 目录不存在，跳过: %s", root)
            continue
        for entry in sorted(os.listdir(root)):
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue
            op = _scan_entry(root, entry, entry_path)
            if op:
                ops.append(op)
    return ops


def run_command(cmd_list, timeout, cwd=None, env=None):
    """运行命令并返回 (rc, stdout, stderr)。"""
    try:
        proc = subprocess.run(
            cmd_list,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -2, "", str(e)


def load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}


def summarize_failures(verify_data, max_items=5):
    failures = verify_data.get("failures", []) if isinstance(verify_data, dict) else []
    if not failures:
        return ""
    items = []
    for failure in failures[:max_items]:
        case_idx = failure.get("case_idx", "?")
        error_type = failure.get("error_type", "Unknown")
        error_msg = failure.get("error_msg", "")
        msg = error_msg[:120].replace("\n", " ")
        items.append(f"case {case_idx}: {error_type} ({msg})")
    if len(failures) > max_items:
        items.append(f"... ({len(failures) - max_items} more)")
    return "; ".join(items)


def _new_result(op, target_speedup):
    """构造单个算子的初始结果 dict。"""
    return {
        "level": op["level"],
        "op_name": op["op_name"],
        "dir_name": op["dir_name"],
        "op_dir": op["dir"],
        "source_desc": op.get("source_desc", ""),
        "ast_valid": None,
        "ast_regression_type": None,
        "verify_total_cases": 0,
        "verify_passed_cases": 0,
        "verify_failed_cases": 0,
        "verify_status": "NOT_RUN",
        "verify_error_summary": "",
        "speedup_vs_torch": None,
        "impl_avg_latency_ms": None,
        "framework_avg_latency_ms": None,
        "benchmark_status": "NOT_RUN",
        "benchmark_error": "",
        "target_speedup": target_speedup,
        "target_reached": None,
        "per_shape_summary": "",
        "elapsed_seconds": 0.0,
    }


def _copy_task_jsons(op, verify_dir):
    """复制任务文件同目录下的所有 JSON（保留原文件名，因为 task 里可能硬编码 json 名）。"""
    json_copied = []
    for f in os.listdir(op["task_dir"]):
        if f.endswith(".json"):
            shutil.copy(os.path.join(op["task_dir"], f), os.path.join(verify_dir, f))
            json_copied.append(f)
    return json_copied


def _align_op_name_json(op, verify_dir, json_copied):
    """按 op_name 对齐 JSON 文件名，分别匹配 {op_name}.py 与 {op_name}_torch.py。"""
    task_base_name = os.path.splitext(os.path.basename(op["task_file"]))[0]
    source_json = f"{task_base_name}.json"
    for suffix in ["", "_torch"]:
        expected_json = f"{op['op_name']}{suffix}.json"
        if expected_json not in json_copied and source_json in json_copied:
            shutil.copy(os.path.join(op["task_dir"], source_json), os.path.join(verify_dir, expected_json))


def _prepare_verify_dir(op, verify_dir):
    """复制任务文件/实现文件/配套 JSON 到验证目录。"""
    # verify.py 导入 {op_name}_torch.py；基线闸门检查 {op_name}.py
    # 两者内容必须一致，因此分别复制同一份任务文件
    shutil.copy(op["task_file"], os.path.join(verify_dir, f"{op['op_name']}_torch.py"))
    shutil.copy(op["task_file"], os.path.join(verify_dir, f"{op['op_name']}.py"))
    shutil.copy(op["gen_file"], os.path.join(verify_dir, f"{op['op_name']}_triton_ascend_impl.py"))
    json_copied = _copy_task_jsons(op, verify_dir)
    _align_op_name_json(op, verify_dir, json_copied)


def _run_freeze(op, verify_dir, result, start_time):
    """基线冻结（verify.py / benchmark.py 会检查锚文件）；失败时填充 result 并返回 False。"""
    freeze_cmd = [
        sys.executable,
        os.path.join(SKILL_DIR, "scripts", "freeze_baseline.py"),
        "--op_name", op["op_name"],
        "--work_dir", verify_dir,
        "--mode", "auto",
    ]
    rc, _stdout, stderr = run_command(freeze_cmd, timeout=60, cwd=verify_dir)
    if rc != 0:
        result["verify_status"] = "FREEZE_FAIL"
        result["verify_error_summary"] = f"freeze_baseline.py failed (rc={rc}): {stderr[:300]}"
        result["elapsed_seconds"] = time.time() - start_time
        return False
    return True


def _run_ast_check(op, verify_dir, result):
    """AST 退化预检查，结果写入 result。"""
    ast_cmd = [
        sys.executable,
        VALIDATE_SCRIPT,
        op["gen_file"],
        "--json",
    ]
    rc, stdout, stderr = run_command(ast_cmd, timeout=120, cwd=verify_dir)
    try:
        ast_data = json.loads(stdout.strip())
    except Exception:
        ast_data = {"valid": False, "suggestion": (stdout + stderr)[:500]}
    result["ast_valid"] = ast_data.get("valid", False)
    result["ast_regression_type"] = ast_data.get("regression_type")
    if not result["ast_valid"]:
        result["verify_status"] = "AST_FAIL"
        result["verify_error_summary"] = ast_data.get("suggestion", "AST validation failed")


def _build_op_aliases(op):
    """兼容 op_name / dir_name / 去掉数字前缀的别名，避免 --non-compute-ops 静默失效。"""
    dir_name = op["dir_name"]
    op_aliases = {op["op_name"], dir_name}
    for alias in [op["op_name"], dir_name]:
        if "_" in alias:
            op_aliases.add(alias.split("_", 1)[1])
    return op_aliases


@dataclasses.dataclass
class OpRunContext:
    """单个算子测试运行的共享上下文。"""

    op: dict
    args: argparse.Namespace
    work_dir: str
    test_env: dict
    result: dict


def _run_verify(ctx, ast_failed):
    """精度验证，结果写入 result。"""
    op, args = ctx.op, ctx.args
    verify_dir, test_env, result = ctx.work_dir, ctx.test_env, ctx.result
    op_aliases = _build_op_aliases(op)
    non_compute_flag = ["--non-compute"] if op_aliases & args.non_compute_ops else []
    if non_compute_flag:
        logger.info("[INFO] 对算子 %s 启用 --non-compute", op["op_name"])
    verify_output = os.path.join(verify_dir, "verify_result.json")
    verify_cmd = [
        sys.executable,
        VERIFY_SCRIPT,
        "--op_name", op["op_name"],
        "--verify_dir", verify_dir,
        "--triton_impl_name", "triton_ascend_impl",
        "--timeout", str(args.verify_timeout),
        "--output", verify_output,
    ] + non_compute_flag
    rc, _stdout, stderr = run_command(
        verify_cmd,
        timeout=args.verify_timeout + 120,
        cwd=verify_dir,
        env=test_env,
    )
    verify_data = load_json(verify_output)
    if verify_data is not None and "_load_error" not in verify_data:
        result["verify_total_cases"] = verify_data.get("total_cases", 0)
        result["verify_passed_cases"] = verify_data.get("passed_cases", 0)
        result["verify_failed_cases"] = verify_data.get("failed_cases", 0)
        if result["verify_passed_cases"] == result["verify_total_cases"] and result["verify_total_cases"] > 0:
            if not ast_failed:
                result["verify_status"] = "PASS"
        else:
            if not ast_failed:
                result["verify_status"] = "FAIL"
                result["verify_error_summary"] = summarize_failures(verify_data)
    else:
        if not ast_failed:
            result["verify_status"] = "FAIL"
            result["verify_error_summary"] = f"verify_result.json 读取失败或不存在 (rc={rc}): {stderr[:300]}"


def _run_benchmark(ctx, target_speedup):
    """性能测试，结果写入 result。"""
    op, args = ctx.op, ctx.args
    verify_dir, test_env, result = ctx.work_dir, ctx.test_env, ctx.result
    perf_output = os.path.join(verify_dir, "perf_result.json")
    benchmark_cmd = [
        sys.executable,
        BENCHMARK_SCRIPT,
        "--op_name", op["op_name"],
        "--verify_dir", verify_dir,
        "--triton_impl_name", "triton_ascend_impl",
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--output", perf_output,
    ]
    rc, _stdout, stderr = run_command(
        benchmark_cmd,
        timeout=args.benchmark_timeout,
        cwd=verify_dir,
        env=test_env,
    )
    perf_data = load_json(perf_output)
    if perf_data is not None and "_load_error" not in perf_data:
        result["speedup_vs_torch"] = perf_data.get("speedup_vs_torch")
        # benchmark.py 在全用例失败时会写入显式 null，需用 (x or {}) 防御
        result["impl_avg_latency_ms"] = (perf_data.get("implementation") or {}).get("avg_latency_ms")
        result["framework_avg_latency_ms"] = (perf_data.get("framework") or {}).get("avg_latency_ms")
        result["per_shape_summary"] = json.dumps(
            perf_data.get("per_shape_results", []), ensure_ascii=False, default=str
        )
        bench_passed = perf_data.get("passed_cases", 0)
        bench_total = perf_data.get("total_cases", 0)
        result["benchmark_status"] = "PASS" if bench_passed == bench_total else "FAIL"
        if result["benchmark_status"] == "FAIL":
            result["benchmark_error"] = stderr[:300]
        # 目标加速比判定
        sp = result["speedup_vs_torch"]
        if target_speedup is not None and sp is not None:
            result["target_reached"] = sp >= target_speedup
    else:
        result["benchmark_status"] = "FAIL"
        result["benchmark_error"] = f"perf_result.json 读取失败或不存在 (rc={rc}): {stderr[:300]}"


def test_operator(op, args, target_speedup):
    """对单个算子执行完整测试流程。"""
    result = _new_result(op, target_speedup)
    work_dir = tempfile.mkdtemp(prefix=f"verify_{op['op_name']}_")
    start_time = time.time()
    test_env = os.environ.copy()
    if args.ascend_device is not None:
        test_env["ASCEND_DEVICE_ID"] = str(args.ascend_device)
    ctx = OpRunContext(op=op, args=args, work_dir=work_dir, test_env=test_env, result=result)
    try:
        # 1. 准备验证项目文件
        _prepare_verify_dir(op, work_dir)

        # 1.5 基线冻结
        if not _run_freeze(op, work_dir, result, start_time):
            return result

        # 2. AST 退化预检查
        if not args.skip_ast:
            _run_ast_check(op, work_dir, result)

        ast_failed = result["verify_status"] == "AST_FAIL"
        if ast_failed and args.stop_on_ast_fail:
            result["elapsed_seconds"] = time.time() - start_time
            return result

        # 3. 精度验证（AST 退化但用户未要求停止时仍继续）
        _run_verify(ctx, ast_failed)

        # 4. 性能测试（仅精度全过才执行）
        if result["verify_passed_cases"] == result["verify_total_cases"] and result["verify_total_cases"] > 0:
            _run_benchmark(ctx, target_speedup)

        result["elapsed_seconds"] = time.time() - start_time
        return result

    finally:
        if not args.keep_work_dirs:
            shutil.rmtree(work_dir, ignore_errors=True)


def write_csv_row(result, output_path, fieldnames):
    """追加写入一行结果到 CSV；文件不存在时先写表头。"""
    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def write_csv(results, output_path):
    """将结果写入 CSV。"""
    fieldnames = [
        "level",
        "dir_name",
        "op_name",
        "source_desc",
        "verify_status",
        "verify_total_cases",
        "verify_passed_cases",
        "verify_failed_cases",
        "verify_error_summary",
        "ast_valid",
        "ast_regression_type",
        "benchmark_status",
        "benchmark_error",
        "speedup_vs_torch",
        "impl_avg_latency_ms",
        "framework_avg_latency_ms",
        "target_speedup",
        "target_reached",
        "elapsed_seconds",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return fieldnames


def print_summary(results):
    """打印汇总统计。"""
    total = len(results)
    verify_pass = sum(1 for r in results if r["verify_status"] == "PASS")
    verify_fail = sum(1 for r in results if r["verify_status"] == "FAIL")
    ast_fail = sum(1 for r in results if r["verify_status"] == "AST_FAIL")
    benchmark_pass = sum(1 for r in results if r["benchmark_status"] == "PASS")
    target_reached = sum(1 for r in results if r["target_reached"] is True)

    logger.info("\n========== 批量测试汇总 ==========")
    logger.info("总算子数: %d", total)
    logger.info("AST 退化失败: %d", ast_fail)
    logger.info("精度验证通过: %d", verify_pass)
    logger.info("精度验证失败: %d", verify_fail)
    logger.info("性能测试通过: %d", benchmark_pass)
    if any(r["target_speedup"] is not None for r in results):
        logger.info("达到目标加速比 (≥%s): %d", results[0]["target_speedup"], target_reached)

    if verify_fail or ast_fail:
        logger.info("\n失败的算子:")
        for r in results:
            if r["verify_status"] in ("FAIL", "AST_FAIL"):
                label = f"{r['level']}/{r['dir_name']}"
                logger.info("  - %s: %s - %s", label, r["verify_status"], r["verify_error_summary"][:120])

    # 打印加速比 Top 5 / Bottom 5
    speedups = [
        (r["dir_name"], r["level"], r["speedup_vs_torch"])
        for r in results
        if r["speedup_vs_torch"] is not None
    ]
    if speedups:
        speedups_sorted = sorted(speedups, key=lambda x: x[2] if x[2] is not None else -1)
        logger.info("\n加速比 Top 5:")
        for name, level, sp in speedups_sorted[-5:][::-1]:
            logger.info("  %s/%s: %.4fx", level, name, sp)
        logger.info("\n加速比 Bottom 5:")
        for name, level, sp in speedups_sorted[:5]:
            logger.info("  %s/%s: %.4fx", level, name, sp)


def _build_arg_parser():
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="批量测试 level3/level4 最终版本 Triton 算子的精度与性能")
    parser.add_argument("--skill-dir", default=os.environ.get("TRITON_OP_VERIFIER_SKILL_DIR"),
                        help="triton-op-verifier skill 目录，默认从环境变量 TRITON_OP_VERIFIER_SKILL_DIR 读取")
    parser.add_argument("--dirs", default=os.environ.get("TRITON_BATCH_DIRS"),
                        help="逗号分隔的算子扫描目录，默认从环境变量 TRITON_BATCH_DIRS 读取")
    parser.add_argument("--output", default=None,
                        help="输出 CSV 路径或目录；目录场景自动生成 triton_benchmark_results_{timestamp}.csv")
    parser.add_argument("--non-compute-ops", default="",
                        help="逗号分隔的非计算类算子名列表，这些算子会传 --non-compute 给 verify.py")
    parser.add_argument("--warmup", type=int, default=5, help="benchmark warmup 次数")
    parser.add_argument("--repeats", type=int, default=50, help="benchmark 正式测试次数")
    parser.add_argument("--verify-timeout", type=int, default=900, help="verify.py 超时秒数")
    parser.add_argument("--benchmark-timeout", type=int, default=1800, help="benchmark.py 超时秒数")
    parser.add_argument("--skip-ast", action="store_true", help="跳过 AST 退化预检查")
    parser.add_argument("--stop-on-ast-fail", action="store_true", help="AST 退化失败时跳过 verify/benchmark")
    parser.add_argument("--keep-work-dirs", action="store_true", help="保留临时工作目录用于调试")
    parser.add_argument("--ascend-device", "--npu", dest="ascend_device", type=int, default=None,
                        help="指定 NPU 设备 ID (ASCEND_DEVICE_ID)，--npu 为其别名")
    parser.add_argument("--config", default="config.json", help="项目 config.json 路径，读取 target_speedup")
    parser.add_argument("--level", type=str, default="all",
                        help="只测试指定 level（如 level1/level2/level3/level4），默认 all")
    parser.add_argument("--op-filter", default="",
                        help="逗号分隔的算子目录名过滤，仅测试匹配的算子（如 '1_BatchMatmul,2_GroupedMatmul'）")
    return parser


def _resolve_skill_dir(args):
    """解析 skill 目录（resolve 为绝对路径，避免 cwd 切换到临时目录后相对路径失效）。"""
    global SKILL_DIR, VALIDATE_SCRIPT, VERIFY_SCRIPT, BENCHMARK_SCRIPT
    if args.skill_dir:
        SKILL_DIR = os.path.abspath(args.skill_dir)
    if not SKILL_DIR:
        raise RuntimeError(
            "未指定 triton-op-verifier skill 目录，"
            "请通过 --skill-dir 或环境变量 TRITON_OP_VERIFIER_SKILL_DIR 指定"
        )
    SKILL_DIR = os.path.abspath(SKILL_DIR)
    VALIDATE_SCRIPT = os.path.join(SKILL_DIR, "scripts", "validate_triton_impl.py")
    VERIFY_SCRIPT = os.path.join(SKILL_DIR, "scripts", "verify.py")
    BENCHMARK_SCRIPT = os.path.join(SKILL_DIR, "scripts", "benchmark.py")


def _load_target_speedup(config_path):
    """从项目 config.json 读取目标加速比；读取失败时告警并返回 None。"""
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("target_speedup")
        except Exception as e:
            logger.warning("[WARN] 读取 config.json 失败: %s", e)
    return None


def _resolve_root_dirs(dirs_arg):
    """解析逗号分隔的扫描目录列表；为空时抛错。"""
    root_dirs = [d.strip() for d in dirs_arg.split(",") if d.strip()] if dirs_arg else []
    if not root_dirs:
        raise RuntimeError("未指定任何扫描目录，请通过 --dirs 或环境变量 TRITON_BATCH_DIRS 指定")
    return root_dirs


def _filter_ops(ops, args):
    """按 level / op-filter 过滤算子列表；过滤后为空时抛错。"""
    if args.level != "all":
        ops = [op for op in ops if op["level"] == args.level]
        if not ops:
            raise RuntimeError(f"按 level={args.level} 过滤后未找到匹配的算子")

    if args.op_filter_set:
        ops = [op for op in ops if op["dir_name"] in args.op_filter_set]
        if not ops:
            raise RuntimeError(f"过滤后未找到匹配的算子: {sorted(args.op_filter_set)}")
        logger.info("[INFO] 过滤后剩余 %d 个算子待测试", len(ops))
    else:
        logger.info("[INFO] 共发现 %d 个算子待测试", len(ops))
    if args.non_compute_ops:
        logger.info("[INFO] 非计算类算子: %s", sorted(args.non_compute_ops))
    return ops


def _resolve_output_path(output_arg):
    """确定输出 CSV 路径；目录场景自动生成带时间戳的文件名。"""
    if output_arg:
        if os.path.isdir(output_arg):
            # 已存在的目录：自动生成文件名
            output_path = os.path.join(output_arg, f"triton_benchmark_results_{_make_timestamp()}.csv")
        elif output_arg.lower().endswith(".csv") or os.path.isfile(output_arg):
            # 显式 .csv 文件路径或已存在的文件
            output_path = output_arg
        else:
            # 不存在的路径且不带 .csv 后缀：视为目录，自动生成文件名
            os.makedirs(output_arg, exist_ok=True)
            output_path = os.path.join(output_arg, f"triton_benchmark_results_{_make_timestamp()}.csv")
    else:
        output_path = f"triton_benchmark_results_{_make_timestamp()}.csv"
    output_path = os.path.abspath(output_path)
    # 若输出目录不存在则提前创建
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_path


def _error_result(op, target_speedup, exc):
    """算子测试抛出未捕获异常时的兜底结果。"""
    return {
        "level": op["level"],
        "op_name": op["op_name"],
        "dir_name": op["dir_name"],
        "verify_status": "ERROR",
        "verify_total_cases": 0,
        "verify_passed_cases": 0,
        "verify_failed_cases": 0,
        "verify_error_summary": str(exc),
        "benchmark_status": "NOT_RUN",
        "speedup_vs_torch": None,
        "impl_avg_latency_ms": None,
        "framework_avg_latency_ms": None,
        "target_speedup": target_speedup,
        "target_reached": False,
        "elapsed_seconds": 0.0,
    }


def _log_op_result(result):
    """单个算子完成后输出一行进度。"""
    status_str = result["verify_status"]
    speedup_str = f"{result['speedup_vs_torch']:.4f}x" if result["speedup_vs_torch"] is not None else "N/A"
    logger.info(
        "      验证: %s | 性能: %s | 加速比: %s | 耗时: %.1fs",
        status_str,
        result["benchmark_status"],
        speedup_str,
        result["elapsed_seconds"],
    )


def _run_all_ops(ops, args, target_speedup, output_path):
    """逐个测试算子并逐行追加 CSV（超时也能保留部分结果），返回全部结果。"""
    results = []
    csv_fieldnames = None
    for idx, op in enumerate(ops, 1):
        source_hint = ""
        src = op.get("source_desc", "")
        if src and not src.endswith("_generated.py"):
            source_hint = f" [来源: {src}]"
        logger.info("\n[%d/%d] 测试算子: %s/%s%s", idx, len(ops), op["level"], op["dir_name"], source_hint)
        try:
            result = test_operator(op, args, target_speedup)
        except Exception as e:
            logger.error("[ERROR] 算子 %s/%s 测试异常: %s", op["level"], op["dir_name"], e)
            result = _error_result(op, target_speedup, e)
        results.append(result)
        # 每完成一个算子立即追加到 CSV
        if csv_fieldnames is None:
            csv_fieldnames = write_csv([result], output_path)
        else:
            write_csv_row(result, output_path, csv_fieldnames)
        _log_op_result(result)
    return results


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        _resolve_skill_dir(args)

        # 解析非计算类算子列表
        args.non_compute_ops = set(x.strip() for x in args.non_compute_ops.split(",") if x.strip())

        # 解析算子目录名过滤
        args.op_filter_set = set(x.strip() for x in args.op_filter.split(",") if x.strip()) or None

        # 读取目标加速比
        target_speedup = _load_target_speedup(args.config)

        # 确定要扫描的目录
        root_dirs = _resolve_root_dirs(args.dirs)

        ops = find_operators(root_dirs)
        if not ops:
            logger.error("[ERROR] 未找到任何可测试的算子目录")
            sys.exit(1)

        ops = _filter_ops(ops, args)

        output_path = _resolve_output_path(args.output)
        results = _run_all_ops(ops, args, target_speedup, output_path)
    except RuntimeError as e:
        logger.error("[ERROR] %s", e)
        sys.exit(1)

    logger.info("\n[INFO] 结果已写入: %s", output_path)

    print_summary(results)


if __name__ == "__main__":
    main()
