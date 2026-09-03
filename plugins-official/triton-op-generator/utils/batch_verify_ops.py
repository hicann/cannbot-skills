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
批量校验 Triton 算子的 AST、精度与性能。

三个阶段可通过 --stages 任选（默认全开）：
  ast       —— validate_triton_impl.py，PyTorch 退化 / 作弊静态检查
  verify    —— verify.py，逐用例精度验证
  benchmark —— benchmark.py，性能测试，输出几何平均加速比

benchmark 依赖 verify 的结果作为闸门（benchmark.py 拒绝为未通过精度的实现测速）。
单独指定 --stages benchmark 时会自动附加 --verify_not_required 放行，此时性能数据
仅供参考，不代表实现正确。

两个入参，按算子编号配对：

  --kernel-path     待校验的 Triton kernel 实现。可以是单个 .py 文件，
                    也可以是目录（递归查找 {N}_{Name}{后缀}.py）。
  --benchmark-path  基线 benchmark。可以是单个 {N}_{Name}.py（同目录需有
                    同名 .json），也可以是目录。

精度与性能都用同一对来源对拍：framework 侧取 benchmark 的算子实现，
impl 侧取 kernel 实现，用例来自 benchmark 的 .json。

实现文件名的后缀不限（_generated / _triton_optimized / _optimized_code 等，
见 IMPL_SUFFIXES），同一编号有多份时按该列表优先级取第一个。

对每个算子依次执行：
  freeze_baseline.py（冻结基线锚文件）-> 按 --stages 选定的阶段依次执行
结果逐行追加写入 CSV，中途中断也能保留已完成部分。

算子在多张 NPU 上并行分发：每张卡一个 worker 线程，从共享队列取任务，
天然做到负载均衡（快的卡多跑几个），比预先静态切分更省时间。

设备隔离说明（重要）：
  torch_npu 只认 ASCEND_RT_VISIBLE_DEVICES，单设 ASCEND_DEVICE_ID 不起隔离作用，
  会让所有进程都落到物理 0 卡上互相抢显存，表现为 OOM / ACL 507xxx / 通过数随机波动。
  本脚本两个都设，以 ASCEND_RT_VISIBLE_DEVICES 为准。
"""

import argparse
import csv
import dataclasses
import datetime
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time


# 实现文件名后缀，按优先级排列（同一算子若有多份，取靠前的）
IMPL_SUFFIXES = [
    "_triton_optimized.py",
    "_optimized_code.py",
    "_generated.py",
    "_generated_code.py",
    "_triton_ascend_impl.py",
]

# 禁止使用的 NPU（留给其它任务）；--devices 指定到这些卡会直接报错退出
FORBIDDEN_DEVICES = {14, 15}

# scripts 目录默认值：本脚本位于 <repo>/plugins-official/triton-op-generator/utils/，
# verifier 脚本位于 <repo>/ops/triton-op-verifier/scripts/
_DEFAULT_SCRIPTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "ops", "triton-op-verifier", "scripts",
))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger  # noqa: E402

logger = logging.getLogger("batch_verify_ops")
_print_lock = threading.Lock()
_csv_lock = threading.Lock()

setup_logger(logger)


def _log(level, msg, *args):
    """加锁输出，避免多线程日志交错。"""
    with _print_lock:
        logger.log(level, msg, *args)


def _make_timestamp():
    """本地时区感知时间戳，用于输出文件命名。"""
    return datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def parse_impl_filename(filename):
    """从实现文件名解析 (op_num, op_base_name)；无法解析返回 None。

    如 '25_StridedAttention_triton_optimized.py' -> (25, '25_StridedAttention')
    """
    for suffix in IMPL_SUFFIXES:
        if filename.endswith(suffix):
            base = filename[: -len(suffix)]
            match = re.match(r"^(\d+)_", base)
            if match:
                return int(match.group(1)), base
            return None
    return None


def parse_op_range(spec):
    """解析算子编号范围，支持 '1-53'、'1,3,5'、'1-10,20' 混合形式。"""
    nums = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            nums.update(range(int(lo), int(hi) + 1))
        else:
            nums.add(int(part))
    return nums


def _as_dir(path, what):
    """把"文件或目录"统一成目录；传文件时返回其所在目录。"""
    if os.path.isdir(path):
        return path, None
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    raise RuntimeError(f"{what} 不存在: {path}")


# 遍历 kernel 目录时跳过的子目录
_SKIP_SCAN_DIRS = ("__pycache__", "session_dir", "output", "verify")


@dataclasses.dataclass
class KernelScan:
    """kernel 目录扫描参数。"""
    kernel_dir: str
    kernel_file: str = None
    wanted_nums: set = dataclasses.field(default_factory=set)


def _kernel_entry(dirpath, filename, scan: KernelScan, unparsed: list):
    """把一个候选文件解析成 by_num 条目；不合格返回 None。"""
    if not filename.endswith(".py"):
        return None
    if scan.kernel_file is not None and filename != scan.kernel_file:
        return None
    parsed = parse_impl_filename(filename)
    if parsed is None:
        if dirpath == scan.kernel_dir:
            unparsed.append(filename)
        return None
    op_num, op_base = parsed
    if op_num not in scan.wanted_nums:
        return None
    # 同编号多份实现：按 IMPL_SUFFIXES 优先级取第一个
    rank = next(i for i, suffix in enumerate(IMPL_SUFFIXES) if filename.endswith(suffix))
    return {
        "rank": rank,
        "op_num": op_num,
        "op_name": op_base,
        "impl_file": os.path.join(dirpath, filename),
    }


def _scan_kernel_files(scan: KernelScan):
    """递归遍历 kernel 目录，返回 ({op_num: 条目}, 无法解析的文件名)。

    既支持扁平目录（{N}_{Name}.py 直接摊开），也支持每算子一个子目录
    （如 0819_wjp/45_CrossformerAttention/45_CrossformerAttention_generated.py）。
    """
    by_num, unparsed = {}, []
    for dirpath, dirnames, filenames in os.walk(scan.kernel_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_SCAN_DIRS]
        for filename in sorted(filenames):
            entry = _kernel_entry(dirpath, filename, scan, unparsed)
            if entry is None:
                continue
            prev = by_num.get(entry["op_num"])
            if prev is None or entry["rank"] < prev["rank"]:
                by_num[entry["op_num"]] = entry
    return by_num, unparsed


def _pair_with_benchmark(by_num: dict, bench_dir: str):
    """给每份实现配上 benchmark 的 .py/.json，返回 (ops, problems)。"""
    ops, problems = [], []
    for op_num in sorted(by_num):
        info = by_num[op_num]
        task_py = os.path.join(bench_dir, f"{info['op_name']}.py")
        task_json = os.path.join(bench_dir, f"{info['op_name']}.json")
        if not os.path.isfile(task_py) or not os.path.isfile(task_json):
            problems.append({
                "op_num": op_num,
                "op_name": info["op_name"],
                "reason": "benchmark 缺少 %s" % (
                    ".py" if not os.path.isfile(task_py) else ".json"
                ),
            })
            continue
        info["task_py"] = task_py
        info["task_json"] = task_json
        ops.append(info)
    return ops, problems


def _filter_by_bench_file(ops: list, bench_file: str, problems: list):
    """benchmark 指到单个文件时，校验并剔除与之不同算子的实现。"""
    for info in ops:
        if os.path.basename(info["task_py"]) != bench_file:
            problems.append({
                "op_num": info["op_num"], "op_name": info["op_name"],
                "reason": (f"--benchmark-path 指向 {bench_file}，"
                           f"与 kernel 对应的 {os.path.basename(info['task_py'])} 不是同一算子"),
            })
    return [o for o in ops if os.path.basename(o["task_py"]) == bench_file]


def discover_ops(kernel_path, benchmark_path, wanted_nums):
    """配对被测实现与基线任务/用例，返回 (可测算子列表, 问题列表, 无法解析的文件名)。"""
    kernel_dir, kernel_file = _as_dir(kernel_path, "--kernel-path")
    bench_dir, bench_file = _as_dir(benchmark_path, "--benchmark-path")

    # 直接指到单个 kernel 文件时，只测它对应的那个算子
    if kernel_file is not None:
        parsed = parse_impl_filename(kernel_file)
        if parsed is None:
            raise RuntimeError(
                f"--kernel-path 指向的文件名无法解析实现后缀（需以 "
                f"{'/'.join(IMPL_SUFFIXES)} 之一结尾）: {kernel_file}")
        wanted_nums = {parsed[0]}

    by_num, unparsed = _scan_kernel_files(
        KernelScan(kernel_dir=kernel_dir, kernel_file=kernel_file, wanted_nums=set(wanted_nums)))
    ops, problems = _pair_with_benchmark(by_num, bench_dir)

    if bench_file is not None:
        ops = _filter_by_bench_file(ops, bench_file, problems)

    for op_num in sorted(set(wanted_nums) - set(by_num)):
        problems.append({"op_num": op_num, "op_name": "", "reason": "kernel 路径下无该编号的文件"})

    return ops, problems, unparsed


def run_command(cmd_list, timeout, cwd=None, env=None):
    """运行命令并返回 (rc, stdout, stderr)。"""
    try:
        proc = subprocess.run(
            cmd_list, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -2, "", str(e)


def load_json(path):
    """读 JSON；文件不存在返回 None，解析失败返回 {'_load_error': ...}。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}


def summarize_failures(verify_data, max_items=3):
    """把 verify_result.json 的 failures 压成一行摘要。"""
    failures = verify_data.get("failures", []) if isinstance(verify_data, dict) else []
    if not failures:
        return ""
    items = []
    for failure in failures[:max_items]:
        case_idx = failure.get("case_idx", "?")
        error_type = failure.get("error_type", "Unknown")
        error_msg = (failure.get("error_msg") or "").strip().splitlines()
        tail = error_msg[-1][:400] if error_msg else ""
        items.append(f"case {case_idx}: {error_type} ({tail})")
    if len(failures) > max_items:
        items.append(f"... (另有 {len(failures) - max_items} 例)")
    return "; ".join(items)


def _new_result(op, device):
    """构造单个算子的初始结果 dict。"""
    return {
        "op_num": op["op_num"],
        "op_name": op["op_name"],
        "device": device,
        "impl_file": op["impl_file"],
        "task_file": op["task_py"],
        "ast_valid": None,
        "ast_regression_type": None,
        "ast_reason": "",
        "verify_status": "NOT_RUN",
        "verify_total_cases": 0,
        "verify_passed_cases": 0,
        "verify_failed_cases": 0,
        "verify_error_summary": "",
        "benchmark_status": "NOT_RUN",
        "speedup_vs_torch": None,
        "impl_avg_latency_ms": None,
        "framework_avg_latency_ms": None,
        "benchmark_error": "",
        "elapsed_seconds": 0.0,
    }


def _prepare_work_dir(op, work_dir):
    """铺好验证目录：基线 .py 两份 + 实现 .py + 用例 .json 两份。

    verify.py 会 import {op_name}_torch.py，基线闸门校验 {op_name}.py，
    两者内容必须一致，因此同一份任务文件复制两次。
    """
    op_name = op["op_name"]
    shutil.copy(op["task_py"], os.path.join(work_dir, f"{op_name}_torch.py"))
    shutil.copy(op["task_py"], os.path.join(work_dir, f"{op_name}.py"))
    shutil.copy(op["impl_file"], os.path.join(work_dir, f"{op_name}_triton_ascend_impl.py"))
    # 任务里可能按任一名字硬编码读 json，两个名字都放一份
    shutil.copy(op["task_json"], os.path.join(work_dir, f"{op_name}.json"))
    shutil.copy(op["task_json"], os.path.join(work_dir, f"{op_name}_torch.json"))


@dataclasses.dataclass
class StageContext:
    """单个算子在 freeze / ast / verify / benchmark 各阶段共享的运行上下文。"""
    scripts_dir: str
    op: dict
    work_dir: str
    result: dict
    env: dict

    def script(self, name: str) -> str:
        return os.path.join(self.scripts_dir, name)

    def work_file(self, name: str) -> str:
        return os.path.join(self.work_dir, name)


def _run_freeze(ctx: StageContext):
    """冻结基线锚文件（verify.py 的前置闸门）；失败时填 result 并返回 False。"""
    op, result = ctx.op, ctx.result
    cmd = [
        sys.executable, ctx.script("freeze_baseline.py"),
        "--op_name", op["op_name"], "--work_dir", ctx.work_dir, "--mode", "auto",
    ]
    rc, _stdout, stderr = run_command(cmd, timeout=120, cwd=ctx.work_dir, env=ctx.env)
    if rc != 0:
        result["verify_status"] = "FREEZE_FAIL"
        result["verify_error_summary"] = f"freeze_baseline.py 失败 (rc={rc}): {stderr[:300]}"
        return False
    return True


def _run_ast_check(ctx: StageContext):
    """AST 退化检查，结果写入 result。"""
    result = ctx.result
    cmd = [sys.executable, ctx.script("validate_triton_impl.py"),
           ctx.op["impl_file"], "--json"]
    rc, stdout, stderr = run_command(cmd, timeout=180, cwd=ctx.work_dir, env=ctx.env)
    try:
        ast_data = json.loads(stdout.strip())
    except ValueError:
        ast_data = {"valid": False, "suggestion": (stdout + stderr)[:500] or f"rc={rc}"}
    result["ast_valid"] = ast_data.get("valid", False)
    result["ast_regression_type"] = ast_data.get("regression_type")
    if not result["ast_valid"]:
        result["ast_reason"] = (ast_data.get("suggestion") or "AST 校验失败")[:600]


def _run_verify(ctx: StageContext, timeout, non_compute):
    """精度验证，结果写入 result。"""
    result = ctx.result
    output_json = ctx.work_file("verify_result.json")
    cmd = [
        sys.executable, ctx.script("verify.py"),
        "--op_name", ctx.op["op_name"], "--verify_dir", ctx.work_dir,
        "--triton_impl_name", "triton_ascend_impl",
        "--timeout", str(timeout), "--output", output_json,
    ]
    if non_compute:
        cmd.append("--non-compute")
    rc, _stdout, stderr = run_command(cmd, timeout=timeout + 180, cwd=ctx.work_dir, env=ctx.env)

    verify_data = load_json(output_json)
    if verify_data is None or "_load_error" in verify_data:
        result["verify_status"] = "FAIL"
        result["verify_error_summary"] = (
            f"verify_result.json 未生成或读取失败 (rc={rc}): {stderr[-300:]}"
        )
        return

    result["verify_total_cases"] = verify_data.get("total_cases", 0)
    result["verify_passed_cases"] = verify_data.get("passed_cases", 0)
    result["verify_failed_cases"] = verify_data.get("failed_cases", 0)
    all_passed = (
        result["verify_passed_cases"] == result["verify_total_cases"]
        and result["verify_total_cases"] > 0
    )
    result["verify_status"] = "PASS" if all_passed else "FAIL"
    if not all_passed:
        result["verify_error_summary"] = summarize_failures(verify_data)


def _run_benchmark(ctx: StageContext, args):
    """性能测试，结果写入 result。

    benchmark.py 自带 verify 闸门：未通过精度的实现会被 exit 2 拒绝。仅当用户
    显式跳过 verify 阶段时才附加 --verify_not_required 放行。
    """
    result = ctx.result
    output_json = ctx.work_file("perf_result.json")
    cmd = [
        sys.executable, ctx.script("benchmark.py"),
        "--op_name", ctx.op["op_name"], "--verify_dir", ctx.work_dir,
        "--triton_impl_name", "triton_ascend_impl",
        "--warmup", str(args.warmup), "--repeats", str(args.repeats),
        "--output", output_json,
    ]
    if "verify" not in args.stage_set:
        cmd.append("--verify_not_required")
    rc, _stdout, stderr = run_command(cmd, timeout=args.benchmark_timeout,
                                      cwd=ctx.work_dir, env=ctx.env)

    perf = load_json(output_json)
    if perf is None or "_load_error" in perf:
        result["benchmark_status"] = "FAIL"
        result["benchmark_error"] = f"perf_result.json 未生成或读取失败 (rc={rc}): {stderr[-300:]}"
        return

    result["speedup_vs_torch"] = perf.get("speedup_vs_torch")
    # 全用例失败时 benchmark.py 会写入显式 null，用 (x or {}) 防御
    result["impl_avg_latency_ms"] = (perf.get("implementation") or {}).get("avg_latency_ms")
    result["framework_avg_latency_ms"] = (perf.get("framework") or {}).get("avg_latency_ms")
    bench_passed = perf.get("passed_cases", 0)
    bench_total = perf.get("total_cases", 0)
    result["benchmark_status"] = "PASS" if (bench_total and bench_passed == bench_total) else "FAIL"
    if result["benchmark_status"] == "FAIL":
        result["benchmark_error"] = f"{bench_passed}/{bench_total} 用例通过; {stderr[-200:]}"


def test_operator(op, device, args):
    """对单个算子执行 AST + 精度校验，返回结果 dict。"""
    result = _new_result(op, device)
    start_time = time.time()
    work_dir = tempfile.mkdtemp(prefix=f"archverify_{op['op_name']}_")

    env = os.environ.copy()
    # torch_npu 只认 ASCEND_RT_VISIBLE_DEVICES；ASCEND_DEVICE_ID 一并设上但不能只设它
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    env["ASCEND_DEVICE_ID"] = str(device)

    try:
        _prepare_work_dir(op, work_dir)
        ctx = StageContext(scripts_dir=args.scripts_dir, op=op, work_dir=work_dir,
                           result=result, env=env)

        if not _run_freeze(ctx):
            return result

        if "ast" in args.stage_set:
            _run_ast_check(ctx)

        ast_failed = result["ast_valid"] is False
        if ast_failed and args.stop_on_ast_fail:
            result["verify_status"] = "SKIPPED_AST_FAIL"
            return result

        if "verify" in args.stage_set:
            non_compute = (op["op_name"] in args.non_compute_ops
                           or str(op["op_num"]) in args.non_compute_ops)
            _run_verify(ctx, args.verify_timeout, non_compute)

        if "benchmark" in args.stage_set:
            # 精度未全过时不测速：benchmark.py 会拒绝，且此时的延时没有意义
            verify_ok = (result["verify_status"] == "PASS"
                         or "verify" not in args.stage_set)
            if verify_ok:
                _run_benchmark(ctx, args)
            else:
                result["benchmark_status"] = "SKIPPED_VERIFY_FAIL"
        return result
    except Exception as e:
        result["verify_status"] = "ERROR"
        result["verify_error_summary"] = f"{type(e).__name__}: {e}"
        return result
    finally:
        result["elapsed_seconds"] = round(time.time() - start_time, 1)
        if not args.keep_work_dirs:
            shutil.rmtree(work_dir, ignore_errors=True)


CSV_FIELDS = [
    "op_num", "op_name", "device",
    "ast_valid", "ast_regression_type",
    "verify_status", "verify_total_cases", "verify_passed_cases", "verify_failed_cases",
    "benchmark_status", "speedup_vs_torch",
    "impl_avg_latency_ms", "framework_avg_latency_ms",
    "elapsed_seconds", "ast_reason", "verify_error_summary", "benchmark_error",
    "impl_file", "task_file",
]


def append_csv_row(result, output_path):
    """线程安全地追加一行到 CSV；文件不存在时先写表头。"""
    with _csv_lock:
        exists = os.path.isfile(output_path)
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(result)


@dataclasses.dataclass
class WorkerContext:
    """所有工作线程共享的运行参数。"""
    args: object
    results: list
    output_path: str
    total: int


def worker(device, task_queue, wctx: WorkerContext):
    """单张卡的工作线程：从队列取算子，跑完写 CSV。"""
    results = wctx.results
    while True:
        try:
            op = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            _log(logging.INFO, "[NPU %s] 开始 %s", device, op["op_name"])
            result = test_operator(op, device, wctx.args)
            results.append(result)
            append_csv_row(result, wctx.output_path)
            sp = result["speedup_vs_torch"]
            sp_txt = f"{sp:.4f}x" if isinstance(sp, (int, float)) else "-"
            _log(
                logging.INFO,
                "[NPU %s] 完成 %s -> %s | 用例 %s/%s | AST %s | 加速比 %s | %.1fs (%d/%d)",
                device, op["op_name"], result["verify_status"],
                result["verify_passed_cases"], result["verify_total_cases"],
                result["ast_valid"], sp_txt, result["elapsed_seconds"],
                len(results), wctx.total,
            )
        finally:
            task_queue.task_done()


def _fmt_ms(v):
    """延迟格式化：非数值统一显示为 "-"。"""
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def write_report(results, problems, output_path, report_path, elapsed):
    """写人类可读的汇总报告。"""
    def bucket(r):
        if r["verify_status"] == "PASS" and r["ast_valid"]:
            return "全通过"
        if r["verify_status"] == "PASS" and not r["ast_valid"]:
            return "精度过但 AST 不过"
        if r["verify_status"] in ("FAIL", "ERROR", "FREEZE_FAIL", "SKIPPED_AST_FAIL"):
            return "精度不过"
        return "其他"

    groups = {}
    for r in results:
        groups.setdefault(bucket(r), []).append(r)

    lines = []
    lines.append("# Triton 算子 AST + 精度校验报告\n")
    lines.append(f"- 生成时间: {_make_timestamp()}")
    lines.append(f"- 总耗时: {elapsed / 60:.1f} 分钟")
    lines.append(f"- 实测算子数: {len(results)}")
    lines.append(f"- 明细 CSV: {output_path}")
    lines.append("- 对拍口径: framework 侧 = benchmark 算子实现，impl 侧 = kernel 实现，"
                 "用例取自 benchmark 的 .json；精度与性能共用同一对来源\n")

    lines.append("## 汇总\n")
    lines.append("| 分类 | 数量 | 算子编号 |")
    lines.append("| --- | --- | --- |")
    for name in ["全通过", "精度过但 AST 不过", "精度不过", "其他"]:
        rows = sorted(groups.get(name, []), key=lambda r: r["op_num"])
        if not rows:
            continue
        nums = ", ".join(str(r["op_num"]) for r in rows)
        lines.append(f"| {name} | {len(rows)} | {nums} |")
    lines.append("")

    lines.append("## 逐算子明细\n")
    lines.append("| 编号 | 算子 | AST | 精度 | 用例 | 加速比 | impl(ms) | torch(ms) | 耗时 | 说明 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(results, key=lambda r: r["op_num"]):
        ast_mark = "OK" if r["ast_valid"] else f"FAIL(type={r['ast_regression_type']})"
        sp = r["speedup_vs_torch"]
        sp_txt = f"{sp:.4f}x" if isinstance(sp, (int, float)) else "-"
        note = (r["ast_reason"] or r["verify_error_summary"]
                or r["benchmark_error"] or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['op_num']} | {r['op_name']} | {ast_mark} | {r['verify_status']} | "
            f"{r['verify_passed_cases']}/{r['verify_total_cases']} | {sp_txt} | "
            f"{_fmt_ms(r['impl_avg_latency_ms'])} | {_fmt_ms(r['framework_avg_latency_ms'])} | "
            f"{r['elapsed_seconds']}s | {note[:200]} |"
        )
    lines.append("")

    if problems:
        lines.append("## 未能测试的算子\n")
        lines.append("| 编号 | 算子 | 原因 |")
        lines.append("| --- | --- | --- |")
        for p in sorted(problems, key=lambda p: p["op_num"]):
            lines.append(f"| {p['op_num']} | {p['op_name']} | {p['reason']} |")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(results, problems):
    """终端汇总。"""
    total = len(results)
    ast_ok = sum(1 for r in results if r["ast_valid"])
    verify_ok = sum(1 for r in results if r["verify_status"] == "PASS")
    both_ok = sum(1 for r in results if r["verify_status"] == "PASS" and r["ast_valid"])

    logger.info("\n========== 校验汇总 ==========")
    logger.info("实测算子数: %d", total)
    logger.info("AST 通过:   %d", ast_ok)
    logger.info("精度通过:   %d", verify_ok)
    logger.info("两项全过:   %d", both_ok)
    sps = [r["speedup_vs_torch"] for r in results
           if isinstance(r["speedup_vs_torch"], (int, float))]
    if sps:
        logger.info("性能通过:   %d | 加速比 >=1.0: %d",
                    sum(1 for r in results if r["benchmark_status"] == "PASS"),
                    sum(1 for x in sps if x >= 1.0))
    if problems:
        logger.info("未能测试:   %d", len(problems))

    bad = [r for r in results if r["verify_status"] != "PASS" or not r["ast_valid"]]
    if bad:
        logger.info("\n有问题的算子:")
        for r in sorted(bad, key=lambda r: r["op_num"]):
            note = (r["ast_reason"] or r["verify_error_summary"] or "").replace("\n", " ")
            logger.info(
                "  %3d %-40s AST=%-5s %-8s %s/%s  %s",
                r["op_num"], r["op_name"], r["ast_valid"], r["verify_status"],
                r["verify_passed_cases"], r["verify_total_cases"], note[:100],
            )


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="批量校验 Triton 算子的 AST、精度与性能"
    )
    parser.add_argument("--kernel-path", "--impl-dir", dest="kernel_path", required=True,
                        help="待校验的 Triton kernel：单个 .py 文件，或递归查找的目录"
                             "（--impl-dir 为兼容别名）")
    parser.add_argument("--benchmark-path", "--bench-dir", dest="benchmark_path", required=True,
                        help="基线 benchmark：单个 {N}_{Name}.py（同目录需有同名 .json），"
                             "或包含它们的目录（--bench-dir 为兼容别名）")
    parser.add_argument("--scripts-dir", default=_DEFAULT_SCRIPTS_DIR,
                        help="triton-op-verifier 的 scripts 目录"
                             "（含 validate_triton_impl.py / verify.py / freeze_baseline.py），"
                             "默认按本脚本在仓库中的相对位置推导")
    parser.add_argument("--ops", default="1-53", help="算子编号范围，如 '1-53' 或 '1,5,10-20'")
    parser.add_argument("--devices", default="", help="逗号分隔的 NPU 编号；留空则自动挑选空闲卡")
    parser.add_argument("--num-devices", type=int, default=2, help="自动挑选时使用的卡数（默认 2）")
    parser.add_argument("--output", default=None, help="输出 CSV 路径或目录")
    parser.add_argument("--verify-timeout", type=int, default=1800, help="verify.py 超时秒数")
    parser.add_argument("--non-compute-ops", default="", help="逗号分隔的非计算类算子名或编号")
    parser.add_argument("--stages", default="ast,verify,benchmark",
                        help="逗号分隔的执行阶段，可选 ast / verify / benchmark，默认全开")
    parser.add_argument("--warmup", type=int, default=5, help="benchmark warmup 次数")
    parser.add_argument("--repeats", type=int, default=5, help="benchmark 正式测试次数")
    parser.add_argument("--benchmark-timeout", type=int, default=3600,
                        help="benchmark.py 超时秒数")
    parser.add_argument("--skip-ast", action="store_true",
                        help="[兼容旧用法] 等价于从 --stages 中去掉 ast")
    parser.add_argument("--stop-on-ast-fail", action="store_true", help="AST 不过时跳过精度验证")
    parser.add_argument("--keep-work-dirs", action="store_true", help="保留临时工作目录便于排查")
    parser.add_argument("--dry-run", action="store_true", help="只打印配对结果，不实际执行")
    return parser


# npu-smi 不在 PATH 时的常见安装位置（避免依赖 PATH 定位可执行文件）
_NPU_SMI_CANDIDATES = [
    "/usr/local/Ascend/driver/tools/npu-smi",
    "/usr/local/sbin/npu-smi",
    "/usr/local/bin/npu-smi",
    "/usr/bin/npu-smi",
]


def _npu_smi_path():
    """npu-smi 的绝对路径：先查 PATH，再遍历候选目录；找不到返回 None。"""
    found = shutil.which("npu-smi")
    if found:
        return os.path.abspath(found)
    for path in _NPU_SMI_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _query_npu_proc(npu_smi, dev):
    """查询单卡进程占用；命令失败记 warning 并返回 None。"""
    try:
        proc = subprocess.run(
            [npu_smi, "info", "-t", "proc-mem", "-i", str(dev)],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[WARN] npu-smi 查询设备 %d 失败，跳过该卡: %s: %s",
                       dev, type(e).__name__, e)
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _npu_free_devices():
    """用 npu-smi 找出无进程占用的卡编号（排除禁用卡）。"""
    npu_smi = _npu_smi_path()
    if npu_smi is None:
        logger.warning("[WARN] 未找到 npu-smi 可执行文件，无法自动挑选空闲卡，"
                       "请用 --devices 显式指定")
        return []
    free = []
    for dev in range(16):
        if dev in FORBIDDEN_DEVICES:
            continue
        stdout = _query_npu_proc(npu_smi, dev)
        if stdout is not None and "Process id:" not in stdout:
            free.append(dev)
    return free


def _resolve_devices(args):
    """确定要用的 NPU 列表，并拦截禁用卡。"""
    if args.devices.strip():
        devices = [int(x) for x in args.devices.split(",") if x.strip()]
        bad = sorted(set(devices) & FORBIDDEN_DEVICES)
        if bad:
            raise RuntimeError(f"NPU {bad} 被禁止使用，请换其他卡")
        if not devices:
            raise RuntimeError("--devices 解析后为空")
        return devices

    free = _npu_free_devices()
    if len(free) < args.num_devices:
        raise RuntimeError(
            f"空闲 NPU 不足：需要 {args.num_devices} 张，实际空闲 {free}（已排除 {sorted(FORBIDDEN_DEVICES)}）"
        )
    return free[: args.num_devices]


def _resolve_output_path(output_arg):
    """确定输出 CSV 路径；目录场景自动生成带时间戳的文件名。"""
    default_name = f"verify_results_{_make_timestamp()}.csv"
    if not output_arg:
        output_path = os.path.abspath(default_name)
    elif os.path.isdir(output_arg):
        output_path = os.path.join(os.path.abspath(output_arg), default_name)
    elif output_arg.lower().endswith(".csv"):
        output_path = os.path.abspath(output_arg)
    else:
        os.makedirs(output_arg, exist_ok=True)
        output_path = os.path.join(os.path.abspath(output_arg), default_name)
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    return output_path


_VALID_STAGES = {"ast", "verify", "benchmark"}
# 各阶段依赖的 verifier 脚本
_STAGE_SCRIPTS = {
    "ast": "validate_triton_impl.py",
    "verify": "verify.py",
    "benchmark": "benchmark.py",
}


def _normalize_args(args):
    """把逗号分隔的参数解析成集合，并处理 --skip-ast 兼容项。"""
    args.non_compute_ops = {x.strip() for x in args.non_compute_ops.split(",") if x.strip()}
    args.stage_set = {x.strip() for x in args.stages.split(",") if x.strip()}
    if args.skip_ast:
        args.stage_set.discard("ast")


def _validate_args(args):
    """校验路径、阶段与依赖脚本；不合法抛 RuntimeError。"""
    for name, path in [("--kernel-path", args.kernel_path),
                       ("--benchmark-path", args.benchmark_path)]:
        if not os.path.exists(path):
            raise RuntimeError(f"{name} 不存在: {path}")
    if not os.path.isdir(args.scripts_dir):
        raise RuntimeError(f"--scripts-dir 不存在: {args.scripts_dir}")

    bad_stages = args.stage_set - _VALID_STAGES
    if bad_stages:
        raise RuntimeError(
            f"--stages 含未知阶段 {sorted(bad_stages)}，可选 {sorted(_VALID_STAGES)}")
    if not args.stage_set:
        raise RuntimeError("--stages 解析后为空，至少要选一个阶段")

    needed = ["freeze_baseline.py"]
    needed += [_STAGE_SCRIPTS[st] for st in ("ast", "verify", "benchmark") if st in args.stage_set]
    for script in needed:
        if not os.path.isfile(os.path.join(args.scripts_dir, script)):
            raise RuntimeError(f"--scripts-dir 下缺少 {script}: {args.scripts_dir}")


@dataclasses.dataclass
class RunPlan:
    """一次运行的计划：配对结果、设备与输出路径。"""
    ops: list
    problems: list
    devices: list
    output_path: str
    report_path: str


def _prepare_run(args) -> "RunPlan":
    """校验参数、配对算子、定设备与输出路径，返回运行计划。"""
    _validate_args(args)

    wanted = parse_op_range(args.ops)
    ops, problems, unparsed = discover_ops(args.kernel_path, args.benchmark_path, wanted)
    if unparsed:
        logger.warning("[WARN] %d 个文件名无法解析实现后缀，已跳过: %s",
                       len(unparsed), ", ".join(unparsed[:5]))
    if not ops:
        raise RuntimeError("没有配对成功的算子，请检查 --kernel-path / --benchmark-path / --ops")

    output_path = _resolve_output_path(args.output)
    return RunPlan(
        ops=ops,
        problems=problems,
        devices=_resolve_devices(args),
        output_path=output_path,
        report_path=os.path.splitext(output_path)[0] + "_report.md",
    )


def _log_plan(args, plan: "RunPlan"):
    """开跑前打印本次运行计划。"""
    logger.info("待测算子: %d 个（编号 %s）", len(plan.ops), args.ops)
    logger.info("kernel 路径:    %s", args.kernel_path)
    logger.info("benchmark 路径: %s", args.benchmark_path)
    logger.info("执行阶段: %s", ", ".join(x for x in ("ast", "verify", "benchmark")
                                        if x in args.stage_set))
    logger.info("使用 NPU: %s（禁用 %s）", plan.devices, sorted(FORBIDDEN_DEVICES))
    logger.info("明细 CSV: %s", plan.output_path)
    logger.info("汇总报告: %s", plan.report_path)
    if plan.problems:
        logger.warning("[WARN] %d 个算子无法测试:", len(plan.problems))
        for p in sorted(plan.problems, key=lambda p: p["op_num"]):
            logger.warning("       %3d %-38s %s", p["op_num"], p["op_name"], p["reason"])


def _log_dry_run(ops):
    logger.info("\n--dry-run 配对结果:")
    for op in ops:
        logger.info("  %3d %-38s kernel=%s", op["op_num"], op["op_name"],
                    os.path.basename(op["impl_file"]))
        logger.info("      %-38s bench =%s + %s", "", os.path.basename(op["task_py"]),
                    os.path.basename(op["task_json"]))


def _run_all(ops, devices, args, output_path):
    """多卡并行跑完全部算子，返回 (results, elapsed)。"""
    task_queue = queue.Queue()
    for op in ops:
        task_queue.put(op)

    results = []
    start = time.time()
    wctx = WorkerContext(args=args, results=results, output_path=output_path, total=len(ops))
    threads = [
        threading.Thread(target=worker, args=(dev, task_queue, wctx),
                         name=f"npu{dev}", daemon=True)
        for dev in devices
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.time() - start


def main():
    args = _build_arg_parser().parse_args()
    _normalize_args(args)

    try:
        plan = _prepare_run(args)
    except RuntimeError as e:
        logger.error("[ERROR] %s", e)
        sys.exit(1)

    _log_plan(args, plan)

    if args.dry_run:
        _log_dry_run(plan.ops)
        return

    results, elapsed = _run_all(plan.ops, plan.devices, args, plan.output_path)

    write_report(results, plan.problems, plan.output_path, plan.report_path, elapsed)
    print_summary(results, plan.problems)
    logger.info("\n明细 CSV: %s", plan.output_path)
    logger.info("汇总报告: %s", plan.report_path)


if __name__ == "__main__":
    main()
