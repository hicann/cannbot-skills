# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NPU Profiler 模块。

提供 NPU 性能分析功能，支持：
• 精确的执行时间测量

• L2 cache 清除（可选）

• 自动过滤无关的 warning 输出

"""

import contextlib
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import pandas as pd

# 导入 L2 cache 清除相关功能
from .l2_cache_clear import (
    L2_CACHE_CLEAR_KERNEL_NAME,
    DslType,
    clear_l2_cache_warnings,
    get_l2_cache_warnings,
)
from .l2_cache_clear import (
    clear_l2_cache as run_l2_cache_clear,
)

logger = logging.getLogger(__name__)

try:
    from op_autoresearch.op.utils.triton_autotune_patch import (
        OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME,
    )
except ImportError:
    OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME = "op_autoresearch_restore_copy_kernel"

# 预编译正则表达式，提高性能
# 过滤 profiler 相关的噪声输出
_FILTER_PATTERNS = re.compile(
    r'('
    r'Please DO NOT tune args|'
    r'Invalid parameter export_type|'
    r'Start parsing profiling data|'
    r'CANN profiling data parsed|'
    r'All profiling data parsed|'
    r'\[WARNING\]|'
    r'\[INFO\]|'
    r'profiler\.py:|'
    # 过滤 triton 编译相关的 warning
    r'WARNING:\s*Grid.*physical limit|'
    r'WARNING:\s*Grid.*performance'
    r')'
)

_SYMBOL_PATTERN = re.compile(r'^[\\\|/\-_=+*#~`!@$%^&()\[\]{}.,;:\'"<>?\s]+$')
_DECORATION_PATTERN = re.compile(r'[\\\|\-=/]{3,}')


@dataclass(frozen=True)
class ProfilerConfig:
    """One NPU profiling run and its result-filtering policy."""

    warmup: int = 25
    active: int = 100
    prof_dir_name: Optional[str] = None
    keep_res: bool = False
    suppress_warnings: bool = True
    clear_l2_cache: bool = False
    dsl: DslType = "other"
    filter_restore_copy: bool = False
    framework: str = "torch"


def _profile_run_settings(config: ProfilerConfig) -> Tuple[int, int, int, int, int]:
    wait = 0
    profiler_warmup = 0
    repeat = 1
    skip_first = 1 + config.warmup
    total = skip_first + (wait + profiler_warmup + config.active) * repeat
    return wait, profiler_warmup, repeat, skip_first, total


def _profile_path(directory_name: Optional[str]) -> str:
    timestamp = int(time.time() * 1000)
    stem = directory_name or "profile_results"
    return os.path.join(os.getcwd(), f"{stem}_{timestamp}")


def suppress_output():
    """
    创建输出抑制上下文管理器，过滤特定的 WARNING/INFO 输出。
    
    注意：此过滤器不会过滤 L2 cache 相关的警告消息，
    这些消息通过 l2_cache_clear 模块收集并在 profiler 结束后输出。
    """
    class OutputFilter:
        def __init__(self, original_stream):
            self.original_stream = original_stream
            self.suppress_next_lines = 0

        def __getattr__(self, name):
            return getattr(self.original_stream, name)

        def write(self, text):
            # 如果正在抑制后续行，减少计数器
            if self.suppress_next_lines > 0:
                self.suppress_next_lines -= 1
                if not text.strip():
                    return

            # 使用预编译的正则表达式快速匹配
            if _FILTER_PATTERNS.search(text):
                self.suppress_next_lines = 2
                return

            stripped_text = text.strip()

            # 完全空行
            if not stripped_text:
                return

            # 使用正则表达式快速检查符号行
            if len(stripped_text) <= 50 and _SYMBOL_PATTERN.match(stripped_text):
                unique_chars = set(stripped_text.replace(' ', '').replace('\t', ''))
                if len(unique_chars) <= 3:
                    return

            # 使用正则表达式检查装饰线
            if _DECORATION_PATTERN.search(stripped_text):
                return

            # 其他内容正常输出
            self.original_stream.write(text)

        def flush(self):
            if hasattr(self.original_stream, 'flush'):
                self.original_stream.flush()

    @contextlib.contextmanager
    def output_suppressor():
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = OutputFilter(old_stdout)
            sys.stderr = OutputFilter(old_stderr)
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return output_suppressor()


def profiler_npu_core(
    fn: Callable,
    config: ProfilerConfig,
) -> Tuple[float, str]:
    """
    NPU profiler 核心函数（PyTorch 版）。
    
    Args:
        fn: 要 profile 的函数
        config: Profiling schedule, output, and filtering settings.
        DSL 类型决定 L2 cache 清除方式
             ▪ "triton_ascend": 使用专用 triton kernel（推荐，可精确过滤）

             ▪ 其他: 使用 tensor.zero_()（fallback，有误判风险）

    
    Returns:
        Tuple[float, str]: (执行时间(微秒), profile结果目录路径)
    """
    import torch
    import torch_npu

    fn()
    torch.npu.synchronize()

    experimental_config_type = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_type(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    wait, profiler_warmup, repeat, skip_first, total = _profile_run_settings(config)
    profile_path = _profile_path(config.prof_dir_name)

    if config.clear_l2_cache:
        run_l2_cache_clear(config.dsl, framework="torch")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=profiler_warmup,
            active=config.active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            if config.clear_l2_cache:
                run_l2_cache_clear(config.dsl, framework="torch")
            fn()
            prof.step()
            torch.npu.synchronize()

    exec_time = collect_time(profile_path, config)
    return exec_time, profile_path


def profiler_npu_mindspore_core(
    fn: Callable,
    config: ProfilerConfig,
) -> Tuple[float, str]:
    """
    NPU profiler 核心函数（MindSpore 版）。

    与 PyTorch 版的关键差异：
    1. 枚举类名: AicoreMetrics (非 AiCMetrics)
    2. schedule 参数必须使用关键字传参
    3. data_simplification 默认值为 True，需显式设为 False
    4. profile() 不支持 with_flops / with_modules 参数
    5. 同步接口: ms.runtime.synchronize() (非 torch.npu.synchronize())

    Args:
        fn: 要 profile 的函数
        config: Profiling schedule, output, and filtering settings.

    Returns:
        Tuple[float, str]: (执行时间(微秒), profile结果目录路径)
    """
    import mindspore as ms
    import mindspore.profiler as ms_profiler

    fn()
    ms.runtime.synchronize()

    experimental_config_type = getattr(ms_profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_type(
        aic_metrics=ms_profiler.AicoreMetrics.PipeUtilization,
        profiler_level=ms_profiler.ProfilerLevel.Level0,
        l2_cache=False,
        data_simplification=False
    )
    wait, profiler_warmup, repeat, skip_first, total = _profile_run_settings(config)
    profile_path = _profile_path(config.prof_dir_name)

    if config.clear_l2_cache:
        run_l2_cache_clear(config.dsl, framework="mindspore")

    with ms_profiler.profile(
        activities=[ms_profiler.ProfilerActivity.NPU],
        schedule=ms_profiler.schedule(
            wait=wait, warmup=profiler_warmup, active=config.active,
            repeat=repeat, skip_first=skip_first,
        ),
        on_trace_ready=ms_profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            if config.clear_l2_cache:
                run_l2_cache_clear(config.dsl, framework="mindspore")
            fn()
            prof.step()
            ms.runtime.synchronize()

    exec_time = collect_time(profile_path, config)
    return exec_time, profile_path


def profiler_npu(
    fn: Callable,
    config: Optional[ProfilerConfig] = None,
    **legacy_options: Any,
) -> float:
    """
    NPU profiler 主函数。

    Args:
        fn: 要 profile 的函数
        config: Profiling schedule, output, and filtering settings.
        legacy_options: Backward-compatible keyword fields for ProfilerConfig.
        DSL 类型决定 L2 cache 清除方式
             ▪ "triton_ascend": 使用专用 triton kernel（推荐，可精确过滤）

             ▪ 其他: 使用 tensor.zero_()（fallback，有误判风险）

    Returns:
        float: 平均执行时间（微秒）
    """
    if config is not None and legacy_options:
        raise TypeError("pass either config or legacy profiler options, not both")
    config = config or ProfilerConfig(**legacy_options)

    # --trace / OP_AUTORESEARCH_PROF_KEEP_RES: keep the msprof trace dir (timeline + CSVs).
    keep_res = config.keep_res or os.environ.get("OP_AUTORESEARCH_PROF_KEEP_RES") == "1"
    clear_l2_cache_warnings()

    core_fn = (
        profiler_npu_mindspore_core
        if config.framework == "mindspore"
        else profiler_npu_core
    )

    if config.suppress_warnings:
        with suppress_output():
            exec_time, profile_path = core_fn(fn, config)
    else:
        exec_time, profile_path = core_fn(fn, config)

    warnings_list = get_l2_cache_warnings()
    if warnings_list:
        for warning_msg in warnings_list:
            logger.warning("%s", warning_msg)

    if not keep_res and os.path.exists(profile_path):
        shutil.rmtree(profile_path)

    return exec_time


def _read_profile_frame(target_file: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(target_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as exc:
        logger.warning("Failed to read %s: %s", target_file, exc)
        return None


def _mindspore_time(
    frame: pd.DataFrame,
    target_file: str,
    active: int,
) -> Optional[float]:
    if "Duration(us)" not in frame.columns:
        logger.warning(
            "Missing 'Duration(us)' in %s; found %s",
            target_file,
            list(frame.columns),
        )
        return None
    if "Step ID" in frame.columns:
        all_steps = sorted(frame["Step ID"].dropna().unique())
        active_steps = all_steps[-active:] if len(all_steps) > active else all_steps
        frame = frame[frame["Step ID"].isin(active_steps)]
    if frame.empty:
        logger.warning("No valid rows in %s", target_file)
        return None
    num_steps = len(frame["Step ID"].unique()) if "Step ID" in frame.columns else active
    return frame["Duration(us)"].sum() / num_steps


def _torch_time(
    frame: pd.DataFrame,
    target_file: str,
    active: int,
) -> Optional[float]:
    required_columns = ["Count", "Total Time(us)"]
    if not all(column in frame.columns for column in required_columns):
        logger.warning(
            "Missing required columns in %s; found %s",
            target_file,
            list(frame.columns),
        )
        return None
    valid_ops = frame[frame["Count"] % active == 0]
    if valid_ops.empty:
        logger.warning("No valid operations found in %s", target_file)
        return None
    total_time = valid_ops["Total Time(us)"].sum()
    if pd.isna(total_time) or total_time <= 0:
        logger.warning("Invalid timing data in %s", target_file)
        return None
    return total_time / active


def _frame_time(
    frame: pd.DataFrame,
    target_file: str,
    config: ProfilerConfig,
) -> Optional[float]:
    try:
        if config.framework == "mindspore":
            return _mindspore_time(frame, target_file, config.active)
        return _torch_time(frame, target_file, config.active)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        logger.warning(
            "Error processing timing data in %s: %s", target_file, exc
        )
        return None


def collect_time(base_dir: str, config: ProfilerConfig) -> float:
    """
    从 profiling 结果中收集时间信息。

    - torch: 读 op_statistic.csv，按 Count % active == 0 过滤，求 Total Time(us) / active
    - mindspore (Level0): 读 kernel_details.csv，按 Step ID 取后 active 步，求 Duration(us) / steps

    Args:
        base_dir: profiling 结果目录
        config: Profiling run and filtering settings.

    Returns:
        float: 平均执行时间(微秒)，失败时返回 float('inf')
    """
    if not os.path.exists(base_dir):
        logger.warning("Base directory not found: %s", base_dir)
        return float('inf')

    target_csv = (
        "kernel_details.csv"
        if config.framework == "mindspore"
        else "op_statistic.csv"
    )

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != target_csv:
                continue

            target_file = os.path.join(root, file)
            frame = _read_profile_frame(target_file)
            if frame is None:
                continue

            if config.clear_l2_cache or config.filter_restore_copy:
                frame = _filter_l2_cache_clear_ops(
                    frame,
                    config.dsl,
                    framework=config.framework,
                    filter_restore_copy=config.filter_restore_copy,
                )

            result = _frame_time(frame, target_file, config)
            if result is not None:
                return result

    logger.warning(
        "No valid timing data (%s) found in %s", target_csv, base_dir
    )
    return float('inf')


def _filter_l2_cache_clear_ops(df: pd.DataFrame, dsl: DslType,
                                framework: str = "torch",
                                filter_restore_copy: bool = False) -> pd.DataFrame:
    """
    从 profiling 结果中过滤掉 OP_AUTORESEARCH 框架内部操作。

    同时支持 op_statistic.csv（torch）和 kernel_details.csv（mindspore）的列名。

    过滤内容：
    - L2 cache 清除 kernel（OP_AUTORESEARCH_l2cache_clear / ZerosLike）
    - restore_value 的 copy kernel（filter_restore_copy=True 时）
    """
    if dsl == "triton_ascend":
        col = None
        if 'OP Type' in df.columns:
            col = 'OP Type'
        elif 'Name' in df.columns:
            col = 'Name'

        if col is not None:
            keep = pd.Series(True, index=df.index)
            keep &= ~df[col].str.contains(
                L2_CACHE_CLEAR_KERNEL_NAME, case=False, na=False, regex=False)
            if filter_restore_copy:
                keep &= ~df[col].str.contains(
                    OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME, case=False, na=False, regex=False)
            if framework == "mindspore":
                keep &= ~df[col].str.contains(
                    r'ZerosLike', case=False, na=False, regex=False)
            return df[keep]
        return df

    if 'OP Type' in df.columns:
        return df[~df['OP Type'].str.contains(r'^ZerosLike$', case=False, na=False, regex=True)]
    if 'Type' in df.columns:
        return df[~df['Type'].str.contains(r'^ZerosLike$', case=False, na=False, regex=True)]
    return df
