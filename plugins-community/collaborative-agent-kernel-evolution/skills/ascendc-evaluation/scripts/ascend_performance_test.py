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
AscendPerformanceTest - 算子性能测试工具
用于对单个算子进行独立的性能测试，测量NPU算子的执行延迟和性能指标

核心特性:
1. extract_prof_rows_tasktype - 根据task_type智能提取性能列
2. extract_perf_info_from_profiling_v2 - Pattern区间统计（中位数，排除冷启动）
3. 科学预热与清缓存 - 大矩阵预热提升频率，清空L2缓存
4. 多测试用例遍历 - 支持test_cases.csv
"""

import csv
import datetime
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from io import StringIO
from pathlib import Path

import acl
import pandas as pd
import torch
import torch_npu

logger = logging.getLogger(__name__)

PERF_TAIL_NUM_ERROR_CODE = -1  # 用于标识性能测试失败的特殊返回值


def remove(path):
    """Remove directory if exists."""
    if os.path.exists(path):
        shutil.rmtree(path)


class ProfilingContext:
    """Profiling上下文管理器"""

    def __init__(self, output_path, device_id):
        self.output_path = output_path
        self.device_id = device_id
        self.prof_config = None
        self.step_info = None

    def __enter__(self):
        # 初始化Profiling
        ret = acl.prof.init(self.output_path)
        if ret != 0:
            raise RuntimeError("Failed to initialize profiling")

        # 配置Profiling
        acl_prof_acl_api = 0x0001
        acl_prof_task_time = 0x0002
        acl_prof_aicore_metrics = 0x0004
        device_list = [self.device_id]
        self.prof_config = acl.prof.create_config(
            device_list,
            1,
            0,
            acl_prof_acl_api |
            acl_prof_task_time |
            acl_prof_aicore_metrics
        )

        ret = acl.prof.start(self.prof_config)
        if ret != 0:
            raise RuntimeError("Failed to start profiling")

        self.step_info = acl.prof.create_step_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 停止Profiling并清理资源
        if self.step_info:
            acl.prof.destroy_step_info(self.step_info)

        if self.prof_config:
            ret = acl.prof.stop(self.prof_config)
            ret = acl.prof.destroy_config(self.prof_config)

        ret = acl.prof.finalize()


def msprof_info_gen(path_prefix, path_suffix):
    """解析msprof性能信息，生成profiling汇总文件 - op_summary_*.csv"""
    try:
        command = ["msprof", "--export=on", f"--output={path_suffix}"]
        subprocess.run(command, shell=False, check=True, cwd=path_prefix)
    except subprocess.CalledProcessError as e:
        logger.info(f"msprof 命令执行失败，退出码: {e.returncode}")
    except FileNotFoundError:
        logger.info(f"错误: 目录 '{path_prefix}' 不存在。")
    except Exception as e:
        logger.info(f"执行 msprof 命令时发生错误: {e}")


def locate_op_summary_file(output_dir):
    """定位op_summary文件，确保只有一个文件且返回其绝对路径"""
    op_summary_files = glob.glob(os.path.join(output_dir, "**", "op_summary_*.csv"), recursive=True)

    if not op_summary_files:
        raise FileNotFoundError("未在输出目录及其子目录中找到op_summary文件")
    elif len(op_summary_files) > 1:
        raise RuntimeError(f"找到多个op_summary文件: {op_summary_files}")

    return os.path.abspath(op_summary_files[0])


def extract_prof_rows_tasktype(median_rows_str: str, task_type: str = "vector"):
    """
    Parse median_rows_str (CSV-like text) and extract columns according to task_type.

    根据task_type提取对应的性能指标列，支持vector/cube 两种类型。

    Args:
        median_rows_str: CSV text containing either:
            - header line + one or more data rows, or
            - one or more data rows matching the expected column order.
        task_type: one of "vector", "cube", "unknown" (case-insensitive).

    Returns:
        List[dict]: each dict maps expected column title -> cell value (str) or None if missing.
    """
    # Expected column titles per task type
    headers_by_type = {
        "vector": [
            "OP Type", "OP State", "Task Type", "Task Duration(us)", "Block Dim",
            "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
            "Output Data Types", "Output Formats", "aiv_time(us)", "aiv_total_cycles",
            "aiv_vec_time(us)", "aiv_vec_ratio", "aiv_scalar_time(us)", "aiv_scalar_ratio",
            "aiv_mte2_time(us)", "aiv_mte2_ratio", "aiv_mte3_time(us)", "aiv_mte3_ratio",
            "aiv_icache_miss_rate"
        ],
        "cube": [
            "OP Type", "OP State", "Task Type", "Task Duration(us)", "Block Dim",
            "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
            "Output Data Types", "Output Formats", "aicore_time(us)", "aic_total_cycles",
            "aic_mac_time(us)", "aic_mac_ratio", "aic_scalar_time(us)", "aic_scalar_ratio",
            "aic_mte1_time(us)", "aic_mte1_ratio", "aic_mte2_time(us)", "aic_mte2_ratio",
            "aic_fixpipe_time(us)", "aic_fixpipe_ratio", "aic_icache_miss_rate", "cube_utilization(%)"
        ],
    }

    t = task_type.lower()
    if t not in headers_by_type:
        t = "vector"

    expected_headers = headers_by_type[t]

    # Normalize input
    text = median_rows_str.strip()
    if not text:
        return []

    # Use csv reader to handle quoted fields and commas in fields
    reader = csv.reader(StringIO(text), skipinitialspace=True)
    rows = list(reader)
    if not rows:
        return []

    # Detect whether first row is header by matching some expected header tokens (case-insensitive)
    first_row = [c.strip() for c in rows[0]]
    first_lower = [c.lower() for c in first_row]
    expected_lower = [h.lower() for h in expected_headers]

    header_is_present = False
    # if a majority of expected headers appear in first row -> treat as header
    matches = sum(1 for h in expected_lower if h in first_lower)
    if matches >= max(1, len(expected_lower) // 3):  # heuristic: at least 1/3 match
        header_is_present = True

    if header_is_present:
        header = first_row
        data_rows = rows[1:]
    else:
        header = expected_headers[:]  # assume data columns in expected order
        data_rows = rows

    # Build index map for header (case-insensitive matching)
    header_index = {}
    for i, col in enumerate(header):
        header_index[col.strip().lower()] = i

    results = []
    for row in data_rows:
        # pad row to header length to avoid IndexError
        values = list(row) + [None] * max(0, len(header) - len(row))
        entry = {}
        for col in expected_headers:
            idx = header_index.get(col.lower())
            if idx is None:
                entry[col] = None
            else:
                val = values[idx]
                entry[col] = val if val is not None and val != "" else None
        results.append(entry)

    return results


def extract_perf_info_from_profiling_v2(csv_path, task_type="vector"):
    """
    按 ReduceMax 到 MatMulV3 区间提取 pattern 序列，并统计 pattern序列耗时的中位数及对应表格行。

    自动过滤 advanced-perf 模式中的预热算子（MatMulV3 + ReduceMax）。

    用于复杂算子的性能分析，通过标记点（ReduceMax/MatMulV3）识别核心计算区间，
    排除第一次冷启动，取中位数作为稳定性能指标。

    Args:
        csv_path (str): op_summary.csv文件的路径。
        task_type (str): 算子类型，用于extract_prof_rows_tasktype解析

    Returns:
        tuple: (pattern耗时中位数, pattern序列对应的表格行字符串)
    """
    # 先读取原始数据（含预热标记算子），用于 pattern 区间检测
    # 注意：必须在 filter_warmup_ops 之前读取，否则标记算子已被删除，无法识别区间
    df = pd.read_csv(csv_path)
    op_types = df['OP Type'].values

    # 精确匹配预热形状的 ReduceMax/MatMulV3 作为区间标记（避免误判用户算子）
    warmup_matmul_shape = '"10240,10240;10240,10240"'
    warmup_reduce_shape = '"96,1024,1024;3"'

    input_shapes = df['Input Shapes'].astype(str).values if 'Input Shapes' in df.columns else [''] * len(df)

    # 找到所有预热 ReduceMax 和 MatMulV3 的索引
    start_idxs = [i for i, (op, shape) in enumerate(zip(op_types, input_shapes))
                  if op == 'ReduceMax' and warmup_reduce_shape in shape]
    end_idxs = [i for i, (op, shape) in enumerate(zip(op_types, input_shapes))
                if op == 'MatMulV3' and warmup_matmul_shape in shape]

    # 只保留成对的区间（每个ReduceMax后面最近的MatMulV3）
    pattern_intervals = []
    for start in start_idxs:
        ends_after_start = [e for e in end_idxs if e > start]
        if ends_after_start:
            end = ends_after_start[0]
            pattern_intervals.append((start, end))

    # 提取所有pattern序列（区间内不含头尾）
    pattern_durations = []
    pattern_rows = []
    for start, end in pattern_intervals:
        # 区间必须有至少2个元素（去掉头尾）
        if end - start > 1:
            pattern_slice = df.iloc[start + 1:end]
            pattern_duration = pattern_slice['Task Duration(us)'].sum()
            pattern_durations.append(pattern_duration)
            pattern_rows.append(pattern_slice)

    # 统计除第一个pattern序列外的所有pattern序列耗时（排除冷启动）
    if len(pattern_durations) <= 1:
        # pattern 不足时也需要清理文件，再回退到简单统计
        filter_warmup_ops(csv_path)
        return None, "Not enough pattern sequences found."
    durations_to_stat = pattern_durations[1:]
    rows_to_stat = pattern_rows[1:]

    # 计算中位数
    sorted_idx = sorted(range(len(durations_to_stat)), key=lambda i: durations_to_stat[i])
    n = len(durations_to_stat)
    median_idx = n // 2
    median_duration = durations_to_stat[sorted_idx[median_idx]]
    median_rows = rows_to_stat[sorted_idx[median_idx]]

    # 转为CSV字符串并应用task_type筛选
    median_rows_str = median_rows.to_csv(index=False)
    median_rows_filtered = extract_prof_rows_tasktype(median_rows_str, task_type)

    # pattern 提取完成后，清理 op_summary 中的预热算子（用于归档查看）
    filter_warmup_ops(csv_path)

    return median_duration, median_rows_filtered


def filter_warmup_ops(csv_path):
    """
    过滤掉 advanced-perf 模式中的预热算子（MatMulV3 + ReduceMax）。

    Advanced 模式的 profiling 序列为：
    - MatMulV3 (预热 + 清 cache) - 固定 shape: 10240x10240
    - ReduceMax (预热 + 清 cache) - 固定 shape: 96x1024x1024
    - 目标算子 (真正评测)
    - (循环)

    此函数通过检测 OP Type + Input Shapes 精确识别预热算子，
    避免误判用户要测试的 MatMul/ReduceMax 算子。

    **直接覆盖原始文件**，避免生成多个文件造成索引混乱。

    Args:
        csv_path (str): op_summary CSV 文件路径

    Returns:
        Path: 原始文件路径（已就地过滤）
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # 预热算子的精确特征（OP Type + Input Shapes）
    warmup_matmul_shape = '"10240,10240;10240,10240"'
    warmup_reduce_shape = '"96,1024,1024;3"'

    # 标记预热算子行
    is_warmup = pd.Series([False] * len(df), index=df.index)

    for i, row in df.iterrows():
        op_type = row['OP Type']
        input_shapes = str(row.get('Input Shapes', ''))

        # 精确匹配预热算子
        if op_type == 'MatMulV3' and warmup_matmul_shape in input_shapes:
            is_warmup[i] = True
        elif op_type == 'ReduceMax' and warmup_reduce_shape in input_shapes:
            is_warmup[i] = True

    # 检查是否有预热模式（至少有一对 MatMul + ReduceMax）
    warmup_count = is_warmup.sum()
    if warmup_count < 2:
        # 没有检测到预热模式，直接返回原文件
        return csv_path

    # 过滤掉预热算子，保留目标算子
    df_filtered = df[~is_warmup].copy()

    if len(df_filtered) == 0:
        # 全部被过滤，返回原文件（避免空文件）
        return csv_path

    # 覆盖前先备份原始文件到 origin_data/
    origin_dir = csv_path.parent / "origin_data"
    origin_dir.mkdir(exist_ok=True)
    shutil.copy2(csv_path, origin_dir / csv_path.name)

    # 直接覆盖原始文件（就地过滤）
    df_filtered.to_csv(csv_path, index=False)

    return csv_path


def extract_simple_perf_info(csv_path, task_type="vector"):
    """
    简化版性能提取，计算 Task Duration 的中位数。

    自动过滤 advanced-perf 模式中的预热算子（MatMulV3 + ReduceMax）。

    用于简单算子（如FastGELU）的性能分析，不需要pattern匹配。
    返回中位数而不是总和，因为我们关心单次执行的稳定性能。

    Args:
        csv_path (str): op_summary.csv文件的路径。
        task_type (str): 算子类型，用于extract_prof_rows_tasktype解析

    Returns:
        tuple: (中位数耗时us, 提取后的行数据列表)
    """
    try:
        # 先过滤预热算子
        filtered_path = filter_warmup_ops(csv_path)
        df = pd.read_csv(filtered_path)
    except Exception as e:
        return None, str(e)

    if "Task Duration(us)" not in df.columns:
        return None, "Task Duration(us) column not found"

    # 计算中位数（更能反映稳定性能，排除异常值）
    median_duration = df["Task Duration(us)"].median()

    # 转为CSV字符串并应用task_type筛选
    csv_str = df.to_csv(index=False)
    rows_filtered = extract_prof_rows_tasktype(csv_str, task_type)

    return median_duration, rows_filtered


class MsprofProfiler:
    """
    Msprof-based profiler for single model runs.

    简单性能测试器，用于快速测试单个模型的性能。
    不包含预热和清缓存逻辑，适用于快速验证。
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id

    def profile_model(
        self,
        model,
        inputs,
        profile_root: Path,
        sync_fn=None,
        num_warmup: int = 10,
        num_trials: int = 20,
    ):
        """Profile a model and return (op_summary_path, total_task_duration_us)."""
        remove("/root/atc_data")

        profile_root = Path(profile_root).resolve()
        profile_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        model_tag = model.__class__.__name__
        path_suffix = f"{model_tag}_device{self.device_id}_{timestamp}"
        output_path = profile_root / path_suffix

        for _ in range(num_warmup):
            _ = model(*inputs)
            self._synchronize(sync_fn)

        with ProfilingContext(str(output_path), self.device_id):
            for _ in range(num_trials):
                _ = model(*inputs)
                self._synchronize(sync_fn)

        msprof_info_gen(str(profile_root), path_suffix)
        op_summary_path = locate_op_summary_file(str(output_path))
        total_us = self._sum_task_duration(op_summary_path)

        return Path(op_summary_path), total_us

    @staticmethod
    def _synchronize(sync_fn=None):
        if sync_fn:
            sync_fn()
        else:
            torch_npu.npu.synchronize()

    @staticmethod
    def _sum_task_duration(op_summary_path: str):
        try:
            df = pd.read_csv(op_summary_path)
        except Exception:
            return None

        if "Task Duration(us)" not in df.columns:
            return None

        try:
            return float(df["Task Duration(us)"].sum())
        except Exception:
            return None


class AdvancedPerformanceEngine:
    """
    高级性能测试引擎

    特性:
    1. 科学预热 - 大矩阵运算提升频率
    2. 清空L2缓存 - 每次测试前执行大矩阵运算
    3. Pattern区间统计 - 支持复杂算子的性能分析
    4. 多测试用例遍历 - 支持test_cases.csv
    """

    def __init__(self, logger_instance):
        self.logger = logger_instance

    def warmup_and_measure(
        self,
        model,
        inputs,
        device_id,
        profile_root,
        num_trials=20,
        task_type="vector",
        model_tag="Model"
    ):
        """
        预热模型并测量执行时间。

        使用大矩阵预热提升频率，每次测试前清空L2缓存。

        Args:
            model: 要测试的模型
            inputs: 模型输入
            device_id: NPU设备ID
            profile_root: profiling输出目录
            num_trials: 测试次数
            task_type: 算子类型（vector/cube）
            model_tag: 模型标签（用于输出文件命名）

        Returns:
            tuple: (耗时中位数us, 性能数据列表)
        """
        remove("/root/atc_data")

        profile_root = Path(profile_root).resolve()
        profile_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        path_suffix = f"{model_tag}_device{device_id}_{timestamp}"
        output_path = profile_root / path_suffix

        # 准备预热张量
        mm1, mm2, reduce_input = self._warmup_and_clear_cache()

        # run and profile
        with ProfilingContext(str(output_path), device_id):
            for _ in range(num_trials):
                # 用于提升频率及清空L2缓存
                res = torch.matmul(mm1, mm2)
                torch.npu.synchronize()
                res = torch.max(reduce_input)
                torch.npu.synchronize()

                # 性能测试
                npu_out = model(*inputs)
                torch.npu.synchronize()

        # profiling 解析
        msprof_info_gen(str(profile_root), path_suffix)
        op_summary_file = locate_op_summary_file(str(output_path))

        # 尝试使用pattern统计（如果有ReduceMax/MatMulV3标记）
        median_time, perf_data = extract_perf_info_from_profiling_v2(op_summary_file, task_type)

        # 如果没有pattern，使用简单统计
        if median_time is None:
            median_time, perf_data = extract_simple_perf_info(op_summary_file, task_type)

        return median_time, perf_data, output_path

    @staticmethod
    def _warmup_and_clear_cache():
        """
        准备预热用的张量，用于提升频率和清空L2缓存。

        使用大矩阵（10240x10240）确保充分预热。
        """
        mm1 = torch.rand((10240, 10240), dtype=torch.float16).npu()
        mm2 = torch.rand((10240, 10240), dtype=torch.float16).npu()
        reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).npu()
        return mm1, mm2, reduce_input


# 为兼容性保留的简单测试器类（已被MsprofProfiler替代，但保留旧接口）
class AscendPerformanceTester:
    """
    Ascend算子性能测试器（兼容旧接口）

    注意：此类主要为了兼容agenticKernelGenerator的接口。
    对于CAKE2项目，建议直接使用MsprofProfiler或AdvancedPerformanceEngine。
    """

    def __init__(self):
        unique_id = uuid.uuid4().hex
        self.logger = logging.getLogger(f"AscendPerformanceTest_{unique_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)


def main():
    """命令行入口（用于独立运行性能测试）"""
    import argparse

    parser = argparse.ArgumentParser(description="Ascend算子性能测试工具")
    parser.add_argument("op_name", help="算子名称，如: FastGelu, Add")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--num-trials", type=int, default=20, help="性能测试重复次数")
    parser.add_argument("--task-type", type=str, default="vector",
                        help="算子类型: 'vector', 'cube'")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    run_logger = logging.getLogger("AscendPerformanceTest")
    engine = AdvancedPerformanceEngine(run_logger)

    run_logger.info(f"Performance test for {args.op_name}")
    run_logger.info("This is a simplified interface for standalone testing.")


if __name__ == "__main__":
    main()
