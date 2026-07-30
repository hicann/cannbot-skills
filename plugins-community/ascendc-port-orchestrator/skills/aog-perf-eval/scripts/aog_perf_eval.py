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

"""AscendC 性能测试脚本。

使用 NPU profiler 获取 device 侧算子级时延，并附带
``time.perf_counter`` 兜底机制。
"""

import argparse
import copy
import importlib.util
import inspect
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50


# ============================================================================
# 模型加载与输入解析
# ============================================================================

def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, preferred_name: str):
    candidate = getattr(module, preferred_name, None)
    if inspect.isclass(candidate) and issubclass(candidate, nn.Module):
        return candidate
    for _, value in vars(module).items():
        if inspect.isclass(value) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise AttributeError(f"No nn.Module subclass found in {module.__file__}")


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def _synchronize(device):
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _extract_scalar_from_json(inp: dict):
    """从 JSON scalar/attr 描述中提取具体值（兼容 range_values 回退）。"""
    val = inp.get("value")
    if val is not None:
        return val
    rv = inp.get("range_values")
    if isinstance(rv, (int, float, bool, str)):
        return rv
    if isinstance(rv, list) and len(rv) > 0:
        return rv[0]
    dtype = inp.get("dtype", "")
    if dtype == "bool":
        return True
    if dtype.startswith("int") or dtype.startswith("uint"):
        return 1
    return 1.0


def _get_input_groups_from_json(output_dir: Path):
    """从 output_dir 下的 .json 文件读取输入 cases。"""
    json_files = sorted(output_dir.glob("*.json"))
    json_path = None
    for f in json_files:
        if not f.name.endswith("_all_case.json") and not f.name.endswith(".json.bak"):
            json_path = f
            break
    if json_path is None:
        raise FileNotFoundError(f"No suitable JSON case file found in {output_dir}")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "uint64": torch.uint64,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        group = []
        for inp in inputs:
            if inp["type"] == "tensor":
                dtype = dtype_map.get(inp["dtype"], torch.float32)
                shape = inp["shape"]
                if dtype == torch.bool:
                    t = torch.randint(0, 2, shape, dtype=dtype)
                elif dtype.is_floating_point:
                    t = torch.randn(shape, dtype=dtype)
                elif str(inp.get("dtype", "")).startswith("uint"):
                    t = torch.randint(0, 10, shape, dtype=torch.int64).to(dtype)
                elif str(inp.get("dtype", "")).startswith("int"):
                    t = torch.randint(-10, 10, shape, dtype=dtype)
                elif str(inp.get("dtype", "")).startswith("complex"):
                    t = torch.randn(shape, dtype=dtype)
                else:
                    t = torch.randn(shape, dtype=dtype)
                group.append(t)
            elif inp["type"] in ("attr", "scalar"):
                val = _extract_scalar_from_json(inp)
                dtype = inp.get("dtype", "")
                if dtype == "bool":
                    group.append(bool(val))
                elif dtype in ("float", "double", "fp32", "fp64", "float32", "float64"):
                    group.append(float(val))
                elif dtype in ("int", "int64", "int32", "int16", "int8", "uint8", "uint16", "uint32", "uint64"):
                    group.append(int(val))
                elif str(dtype).startswith("complex"):
                    group.append(complex(val))
                else:
                    group.append(val)
            else:
                group.append(inp.get("value"))
        input_groups.append(group)

    return input_groups, str(json_path)


def _get_input_groups_from_module(module):
    """优先使用 model.py 自带的 get_input_groups / get_inputs 生成输入。"""
    if hasattr(module, "get_input_groups"):
        groups = module.get_input_groups()
        if isinstance(groups, list) and groups:
            return groups
    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if isinstance(inputs, list) and inputs:
            return [inputs]
    return None


def _load_impl(output_dir: Path, impl: str):
    if impl == "reference":
        module_path = output_dir / "model.py"
        preferred_class = "Model"
    elif impl == "ascendc":
        module_path = output_dir / "model_new_ascendc.py"
        preferred_class = "ModelNew"
    else:
        raise ValueError(f"Unsupported impl: {impl}")

    if not module_path.is_file():
        raise FileNotFoundError(f"missing {impl} model: {module_path}")

    module = _load_module(module_path, f"perf_{impl}_model")
    model_cls = _find_model_class(module, preferred_class)
    return module, model_cls, module_path


# ============================================================================
# 性能分析逻辑
# ============================================================================

def _find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _cleanup_profile_path(profile_path: str) -> None:
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)


def _parse_kernel_latency(profile_path: str, active_count: int, keep_traces: bool = False) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """从 kernel_details.csv 提取硬件 kernel 层总时延。

    为什么用 kernel_details.csv 而非 operator_details.csv：
    - operator_details 在 PyTorch dispatcher 层采集，只能看到 aten/aclnn API
    - 自定义 AscendC kernel 通过 aclrtlaunch 直接调用，绕过 dispatcher，
      在 operator_details 中不可见（只看到 aclnnInplaceCopy 这类 memcpy wrapper）
    - kernel_details 在 ACL runtime 层采集，记录所有 NPU kernel launch，
      包括自定义 kernel（如 gelu_fp16_tanh）和 CANN kernel（如 aclnnGelu_Gelu_Gelu）
    - 过滤 Accelerator Core = AI_VECTOR_CORE / AI_CORE，排除 Type=TensorMove (memcpy)
    """
    try:
        import pandas as pd
    except ImportError:
        if not keep_traces:
            _cleanup_profile_path(profile_path)
        return None, None

    kernel_details_file = _find_profile_file(profile_path, "kernel_details.csv")
    if not kernel_details_file or not os.path.exists(kernel_details_file):
        if not keep_traces:
            _cleanup_profile_path(profile_path)
        return None, None

    try:
        df = pd.read_csv(kernel_details_file)
    except Exception:
        if not keep_traces:
            _cleanup_profile_path(profile_path)
        return None, None

    required_columns = ["Name", "Duration(us)", "Accelerator Core"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        if not keep_traces:
            _cleanup_profile_path(profile_path)
        return None, None

    # 取计算类 kernel（AI_VECTOR_CORE / AI_CORE），包括 TensorMove
    # TensorMove (aclnnInplaceCopy) 是 pybind11 中 copy_() 产生的 memcpy 开销，
    # 属于 AscendC 实现的真实成本，不应排除
    compute_kernels = df[df["Accelerator Core"].isin(["AI_VECTOR_CORE", "AI_CORE"])]
    if compute_kernels.empty:
        compute_kernels = df  # 兜底：没有分类信息时取全部

    total_us = compute_kernels["Duration(us)"].sum()
    avg_us = total_us / active_count
    avg_ms = avg_us / 1000.0

    # 按 kernel 名分组统计
    kernel_avg_times = {}
    grouped = compute_kernels.groupby("Name")["Duration(us)"].sum()
    for name, total in grouped.items():
        kernel_avg_times[name] = round(total / active_count, 4)

    if not keep_traces:
        _cleanup_profile_path(profile_path)

    return kernel_avg_times, round(avg_ms, 4)


def _run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str, trace_root: str = "") -> str:
    """运行 NPU profiler 并返回生成的性能分析目录路径。

    Args:
        trace_root: trace 根目录，默认 os.getcwd()
    """
    import torch_npu

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )

    test_fn()
    torch.npu.synchronize()

    skip_first = 1 + warmup
    total_steps = skip_first + warmup + repeats

    timestamp = int(time.time() * 1000)
    root = trace_root if trace_root else os.getcwd()
    os.makedirs(root, exist_ok=True)
    profile_path = os.path.join(root, f"{profile_name}_{timestamp}")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=repeats, repeat=1, skip_first=skip_first
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total_steps):
            test_fn()
            prof.step()
            torch.npu.synchronize()

    return profile_path


def _measure_single_with_profiler(model, inputs, warmup: int, repeats: int, profile_name: str, device, keep_traces: bool = False, trace_root: str = "") -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """使用 torch_npu.profiler 测量单次性能。

    Args:
        trace_root: trace 输出根目录，默认 os.getcwd()
    """

    # warmup + 同步
    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()

    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()

    try:
        profile_path = _run_profiler_with_config(test_fn, warmup, repeats, profile_name, trace_root=trace_root)
        operators, latency_ms = _parse_kernel_latency(profile_path, repeats, keep_traces=keep_traces)
    except Exception as e:
        print(f"  torch_npu.profiler 获取数据失败: {e}，使用兜底测试机制...")
        operators, latency_ms = None, None

    if operators is None or latency_ms is None or latency_ms <= 0.0001:
        print(f"  警告: kernel_details.csv 无法获取有效数据（当前:{latency_ms} ms），使用 time.perf_counter() 兜底...")
        return _measure_single_fallback(model, inputs, warmup, repeats, device, trace_root=trace_root)

    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return operators, latency_ms, round(peak_memory, 2)


def _measure_single_fallback(model, inputs, warmup: int, repeats: int, device, trace_root: str = "") -> Tuple[Dict[str, float], float, float]:
    """使用 time.perf_counter() 的兜底测试机制。"""

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    torch.npu.synchronize()

    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    avg_latency_ms = statistics.mean(latencies)
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return {}, round(avg_latency_ms, 4), round(peak_memory, 2)


def _measure_single_wallclock(model, inputs, warmup: int, repeats: int, device) -> Dict:
    """使用 time.perf_counter() 测量 wall-clock 时间（包含 CPU 侧 kernel launch + sync 开销）。

    Returns:
        dict: {"mean_ms": float, "median_ms": float, "min_ms": float, "max_ms": float,
               "stdev_ms": float, "latencies_ms": [float]}
    """
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    _synchronize(device)

    latencies = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(*inputs)
        _synchronize(device)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return {
        "mean_ms": round(statistics.mean(latencies), 4),
        "median_ms": round(statistics.median(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "stdev_ms": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "latencies_ms": [round(v, 4) for v in latencies],
    }


# ============================================================================
# 主测试逻辑
# ============================================================================

def run_performance(output_dir: str, warmup: int = WARMUP_DEFAULT, repeats: int = REPEATS_DEFAULT, seed: int = 0, keep_traces: bool = False, cases_json: str = ""):
    """对指定 output_dir 进行 reference vs ascendc 性能测试。

    Args:
        keep_traces: 是否保留 profiler trace 目录（默认 False，解析后自动清理）。
        cases_json: 显式指定测试用例 JSON/JSONL 文件路径；为空时自动检测。

    Returns:
        dict: 包含每个 case 的 latency、operators、speedup 等。
    """
    output_dir_path = Path(output_dir).resolve()
    trace_root = str(output_dir_path / "profiler_traces") if keep_traces else ""
    device = _get_device()

    # 加载 reference 和 ascendc 实现
    ref_module, ref_cls, ref_path = _load_impl(output_dir_path, "reference")
    asc_module, asc_cls, asc_path = _load_impl(output_dir_path, "ascendc")

    init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()

    # 优先使用 model.py 自带的输入生成函数，否则回退到 JSON 解析
    input_groups = _get_input_groups_from_module(ref_module)
    if input_groups is not None:
        if cases_json:
            json_path = cases_json
        else:
            # auto-detect: prefer model.json, then first non-excluded .json
            json_files = sorted(output_dir_path.glob("*.json"))
            json_path = None
            for f in json_files:
                if f.name.endswith("_all_case.json") or f.name.endswith(".json.bak"):
                    continue
                if f.name == "model.json":
                    json_path = str(f)
                    break
            if json_path is None:
                for f in json_files:
                    if not f.name.endswith("_all_case.json") and not f.name.endswith(".json.bak"):
                        json_path = str(f)
                        break
            if json_path is None:
                json_path = str(output_dir_path / f"{output_dir_path.name}.json")
    else:
        if cases_json:
            input_groups, _ = _get_input_groups_from_json(Path(cases_json).parent)
            json_path = cases_json
        else:
            input_groups, json_path = _get_input_groups_from_json(output_dir_path)

    report = {
        "op": output_dir_path.name,
        "output_dir": str(output_dir_path),
        "json_path": json_path,
        "device": str(device),
        "warmup": warmup,
        "repeats": repeats,
        "seed": seed,
        "method_profiler": "V5: kernel_details.csv → AI_VECTOR_CORE/AI_CORE (incl. TensorMove) → sum(Duration)/active_count",
        "method_wallclock": "time.perf_counter() with synchronize before/after each call",
        "reference": {
            "model_path": str(ref_path),
            "profiler": {"case_results": [], "ok": False, "error": ""},
            "wallclock": {"case_results": [], "ok": False, "error": ""},
        },
        "ascendc": {
            "model_path": str(asc_path),
            "profiler": {"case_results": [], "ok": False, "error": ""},
            "wallclock": {"case_results": [], "ok": False, "error": ""},
        },
        "per_case_speedup": {"profiler": [], "wallclock": []},
        "overall_speedup": {"profiler": None, "wallclock": None},
    }

    # 测试 reference
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed(seed)
        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()

        for idx, inputs in enumerate(input_groups):
            model_inputs = _move_to_device(_clone_value(inputs), device)
            # Profiler measurement
            operators, latency_ms, peak_mem = _measure_single_with_profiler(
                ref_model, model_inputs, warmup, repeats, f"ref_profile_case{idx}", device, keep_traces=keep_traces, trace_root=trace_root
            )
            report["reference"]["profiler"]["case_results"].append({
                "index": idx, "latency_ms": latency_ms,
                "peak_memory_mb": peak_mem, "operators": operators or {},
            })
            # Wall clock measurement
            wc = _measure_single_wallclock(ref_model, model_inputs, warmup, repeats, device)
            report["reference"]["wallclock"]["case_results"].append({
                "index": idx, "mean_ms": wc["mean_ms"], "median_ms": wc["median_ms"],
                "min_ms": wc["min_ms"], "max_ms": wc["max_ms"], "stdev_ms": wc["stdev_ms"],
            })
        report["reference"]["profiler"]["ok"] = True
        report["reference"]["wallclock"]["ok"] = True
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        report["reference"]["profiler"]["error"] = err
        report["reference"]["wallclock"]["error"] = err
        import traceback
        traceback.print_exc()

    # 测试 ascendc
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed(seed)
        asc_model = asc_cls(*_clone_value(init_inputs)).to(device).eval()

        for idx, inputs in enumerate(input_groups):
            model_inputs = _move_to_device(_clone_value(inputs), device)
            # Profiler measurement
            operators, latency_ms, peak_mem = _measure_single_with_profiler(
                asc_model, model_inputs, warmup, repeats, f"asc_profile_case{idx}", device, keep_traces=keep_traces, trace_root=trace_root
            )
            report["ascendc"]["profiler"]["case_results"].append({
                "index": idx, "latency_ms": latency_ms,
                "peak_memory_mb": peak_mem, "operators": operators or {},
            })
            # Wall clock measurement
            wc = _measure_single_wallclock(asc_model, model_inputs, warmup, repeats, device)
            report["ascendc"]["wallclock"]["case_results"].append({
                "index": idx, "mean_ms": wc["mean_ms"], "median_ms": wc["median_ms"],
                "min_ms": wc["min_ms"], "max_ms": wc["max_ms"], "stdev_ms": wc["stdev_ms"],
            })
        report["ascendc"]["profiler"]["ok"] = True
        report["ascendc"]["wallclock"]["ok"] = True
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        report["ascendc"]["profiler"]["error"] = err
        report["ascendc"]["wallclock"]["error"] = err
        import traceback
        traceback.print_exc()

    # 计算 profiler 与 wallclock 两种计时方式的加速比。
    for method in ("profiler", "wallclock"):
        ref_key = "latency_ms" if method == "profiler" else "mean_ms"
        ref_results = report["reference"][method]["case_results"]
        asc_results = report["ascendc"][method]["case_results"]
        if ref_results and asc_results:
            speedups = []
            for ref_case, asc_case in zip(ref_results, asc_results):
                ref_lat = ref_case[ref_key]
                asc_lat = asc_case[ref_key]
                speedup = ref_lat / asc_lat if asc_lat and asc_lat > 0 else float("inf")
                speedups.append(speedup)
                report["per_case_speedup"][method].append({
                    "index": ref_case["index"],
                    "reference_ms": ref_lat,
                    "ascendc_ms": asc_lat,
                    "speedup": round(speedup, 4),
                })
            report["overall_speedup"][method] = round(statistics.mean(speedups), 4) if speedups else None

    return report


def _print_report(report: dict):
    print("=" * 88)
    print("Performance Report (AscendC) — Profiler + Wall Clock")
    print("=" * 88)
    print(f"Operator    : {report['op']}")
    print(f"Output Dir  : {report['output_dir']}")
    print(f"Device      : {report['device']}")
    print(f"Warmup      : {report['warmup']}")
    print(f"Repeat      : {report['repeats']}")
    print("-" * 88)

    for method, label in [("profiler", "Device-side (kernel_details.csv)"), ("wallclock", "Host-side (time.perf_counter)")]:
        sp = report["overall_speedup"].get(method)
        print(f"\n--- {label} ---")
        print(f"{'Case':<8} {'Ref(ms)':>12} {'AscendC(ms)':>14} {'Speedup':>10}")
        print("-" * 48)
        for case in report["per_case_speedup"].get(method, []):
            print(f"[{case['index']:<5}] {case['reference_ms']:>12.4f} {case['ascendc_ms']:>14.4f} {case['speedup']:>10.2f}x")
        print("-" * 48)
        if sp is not None:
            print(f"Overall speedup ({method}): {sp:.2f}x")
    print("=" * 88)



def _generate_html_report(report: dict, cases_info: list, output_path: str) -> str:
    """Generate dual-method (profiler + wallclock) HTML performance report."""
    def med(values):
        return sorted(values)[len(values) // 2]

    sp_prof = report["overall_speedup"]["profiler"] or 0
    sp_wc = report["overall_speedup"]["wallclock"] or 0
    prof_cases = report["per_case_speedup"]["profiler"]
    wc_cases = report["per_case_speedup"]["wallclock"]
    prof_sp = [c["speedup"] for c in prof_cases]
    wc_sp = [c["speedup"] for c in wc_cases]
    n = len(prof_cases)

    trows = ""
    for i in range(n):
        pc = prof_cases[i]
        wc = wc_cases[i]
        c = cases_info[i] if i < len(cases_info) else {}
        shape = str(c.get("shape", "?"))
        dtype = c.get("dtype", "?")
        approx = c.get("approx", "?")
        elems = 1
        sh = c.get("shape", [1])
        if isinstance(sh, list):
            for d in sh:
                if isinstance(d, int):
                    elems *= d
        p_bw = max(5, min(150, int(pc["speedup"] * 100)))
        w_bw = max(5, min(150, int(wc["speedup"] * 100)))
        p_cls = "bar-g" if pc["speedup"] >= 1 else ("bar-o" if pc["speedup"] >= 0.5 else "bar-b")
        w_cls = "bar-g" if wc["speedup"] >= 1 else ("bar-o" if wc["speedup"] >= 0.5 else "bar-b")
        trows += f"""<tr>
<td class="c">{i}</td><td class="l">{shape}</td><td class="c">{dtype}</td><td class="c">{approx}</td><td class="c">{elems:,}</td>
<td>{pc['reference_ms']*1000:.1f}</td><td>{pc['ascendc_ms']*1000:.1f}</td>
<td><span class="bar {p_cls}" style="width:{p_bw}px"></span> {pc['speedup']:.2f}x</td>
<td>{wc['reference_ms']*1000:.1f}</td><td>{wc['ascendc_ms']*1000:.1f}</td>
<td><span class="bar {w_cls}" style="width:{w_bw}px"></span> {wc['speedup']:.2f}x</td></tr>"""

    fn_fast = sum(1 for s in prof_sp if s >= 1.0)
    wc_fast = sum(1 for s in wc_sp if s >= 1.0)

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>{report['op']} Performance Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5;color:#333}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:10px}}h2{{color:#2563eb;margin-top:30px}}h3{{color:#444}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:12px}}
th,td{{padding:5px 8px;text-align:right;border:1px solid #ddd}}
th{{background:#2563eb;color:#fff;text-align:center}}
td.c,th.c{{text-align:center}}td.l,th.l{{text-align:left}}
tr:nth-child(even){{background:#f9f9f9}}tr:hover{{background:#e8f0fe}}
.summary{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.1);flex:1;min-width:150px}}
.card .value{{font-size:26px;font-weight:bold;color:#2563eb}}.card .label{{color:#666;font-size:12px}}
.bar{{display:inline-block;height:12px;border-radius:2px;vertical-align:middle}}
.bar-b{{background:#ef4444}}.bar-o{{background:#f59e0b}}.bar-g{{background:#22c55e}}
.info{{background:#dbeafe;border-left:4px solid #2563eb;padding:10px;margin:15px 0;font-size:12px}}
.note{{background:#fff3cd;border-left:4px solid #f59e0b;padding:10px;margin:15px 0;font-size:12px}}
</style></head><body>
<h1>{report['op']} Performance Report</h1>
<p><strong>AscendC vs CANN Reference</strong> | Device: {report['device']} | Warmup={report['warmup']}, Repeat={report['repeats']}</p>
<div class="info">
<strong>Dual-method measurement</strong>:<br>
<b>Profiler (device-side)</b>: <code>kernel_details.csv</code> &rarr; AI_VECTOR_CORE/AI_CORE (incl. TensorMove) &rarr; sum(Duration)/active_count &mdash; NPU kernel time only<br>
<b>Wall Clock (host-side)</b>: <code>time.perf_counter()</code> with <code>torch.npu.synchronize()</code> &mdash; end-to-end (kernel + launch + sync overhead)
</div>
<div class="summary">
<div class="card"><div class="label">Profiler Speedup</div><div class="value" style="color:{'#ef4444' if sp_prof<1 else '#22c55e'}">{sp_prof:.2f}x</div><div class="label">device kernel time</div></div>
<div class="card"><div class="label">Wall Clock Speedup</div><div class="value" style="color:{'#ef4444' if sp_wc<1 else '#22c55e'}">{sp_wc:.2f}x</div><div class="label">host end-to-end</div></div>
<div class="card"><div class="label">Profiler &ge;1.0x</div><div class="value">{fn_fast}</div><div class="label">/ {n}</div></div>
<div class="card"><div class="label">WC &ge;1.0x</div><div class="value">{wc_fast}</div><div class="label">/ {n}</div></div>
</div>
<h2>Per-Case Comparison</h2>
<table><thead><tr><th rowspan="2">#</th><th rowspan="2" class="l">Shape</th><th rowspan="2" class="c">dtype</th><th rowspan="2" class="c">approx</th><th rowspan="2" class="c">Elems</th>
<th colspan="3">Profiler (device-side, &mu;s)</th><th colspan="3">Wall Clock (host-side, &mu;s)</th></tr>
<tr><th>Ref</th><th>Asc</th><th>Sp</th><th>Ref</th><th>Asc</th><th>Sp</th></tr></thead><tbody>
{trows}</tbody></table>
<div class="note">
<strong>Profiler method</strong>: <code>torch_npu.profiler.Level1</code> &rarr; <code>kernel_details.csv</code> &rarr; AI_VECTOR_CORE/AI_CORE (incl. TensorMove) &rarr; sum(Duration) / active_count({report['repeats']})<br>
<strong>Wall Clock method</strong>: <code>time.perf_counter()</code> with <code>torch.npu.synchronize()</code> before/after each call. Includes kernel launch + sync overhead on host side.
</div>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AscendC performance eval — Profiler + Wall Clock dual-method")
    parser.add_argument("--output_dir", required=True, help="Operator output directory (model.py + model_new_ascendc.py + .json)")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="Warmup iterations (default 5)")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="Measurement iterations (default 50)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", help="Output JSON report path (dir auto-appends perf_result.json)")
    parser.add_argument("--html", help="Output HTML report path (dir auto-appends perf_report.html)")
    parser.add_argument("--keep-traces", action="store_true", help="Keep profiler trace directories")
    parser.add_argument("--cases", default="", help="Explicit JSON/JSONL test cases file (e.g. model.json). Required for reliable HTML generation when output_dir contains multiple .json files.")
    args = parser.parse_args()

    report = run_performance(args.output_dir, args.warmup, args.repeats, args.seed, keep_traces=args.keep_traces, cases_json=args.cases)
    _print_report(report)

    if args.output:
        save_path = os.path.join(args.output, "perf_result.json") if os.path.isdir(args.output) else args.output
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"JSON saved: {save_path}")

    if args.html:
        cases_info = []
        # Prefer explicit --cases for reliable case metadata; fall back to auto-detected json_path
        cases_src = args.cases if args.cases else report.get("json_path", "")
        if cases_src and os.path.exists(cases_src):
            try:
                with open(cases_src) as f:
                    content = f.read(256)
                    f.seek(0)
                    if content.strip().startswith("{"):
                        # Single JSON object (e.g. manifest.json) — not usable per-line
                        raise ValueError(f"{cases_src} is a single JSON object, not JSONL")
                    for line in f:
                        if line.strip():
                            c = json.loads(line)
                            inp0 = c["inputs"][0]
                            approx_val = ""
                            for inp in c["inputs"]:
                                if inp.get("name") not in ("x", "finished"):
                                    approx_val = str(inp.get("value", inp.get("dtype", "")))
                                    break
                            cases_info.append({"shape": inp0["shape"], "dtype": inp0.get("dtype", ""),
                                              "approx": approx_val})
            except Exception:
                cases_info = [{"shape": "?", "dtype": "?", "approx": ""}] * len(report["per_case_speedup"]["profiler"])
        html_path = os.path.join(args.html, "perf_report.html") if os.path.isdir(args.html) else args.html
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        _generate_html_report(report, cases_info, html_path)
        print(f"HTML saved: {html_path}")


if __name__ == "__main__":
    main()
