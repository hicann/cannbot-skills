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
Profiling utilities shared between KernelVerifier and LocalWorker.
Contains methods for running msprof and analyzing Ascend profiling data.
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from op_autoresearch.core.worker.eval_config import resolve_eval_timeout
from op_autoresearch.utils.process_utils import (
    CommandCaptureOptions,
    run_command_capture,
)

logger = logging.getLogger(__name__)


# Both the Python-script and msprof paths return the same profile section.
# It carries the average latency, a non-empty list of per-case latencies, and
# the timer method. Static-shape measurements still use a one-item list so
# downstream consumers can handle every shape uniformly. The collection API
# returns optional base and generation sections; absence consistently means a
# skipped or failed measurement, avoiding multiple sentinel conventions.


def make_profile_section(avg_us: float,
                         per_case_us: Optional[List[float]] = None,
                         method: Optional[str] = None) -> Dict[str, Any]:
    """Build a canonical profile section. Use this everywhere we synthesize
    a per-shape section from a single aggregate measurement (override
    baseline or msprof path) so the schema stays consistent.
    """
    if per_case_us is None or not per_case_us:
        per_case_us = [float(avg_us)]
    return {
        "avg_us": float(avg_us),
        "per_case_us": [float(t) for t in per_case_us],
        "method": method,
    }


def _finite(x: Any) -> Optional[float]:
    """Coerce to a finite float; None on inf/nan/non-numeric. Used to
    sanitize values read out of profile JSON before they propagate.
    """
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def read_profile_result_from_json(verify_dir: str,
                                  json_filename: str) -> Optional[Dict[str, Any]]:
    """Read a profile-result JSON written by ``prof_{base,generation}_template_refactored.j2``.

    Returns a canonical section dict (see module docstring) or ``None`` when
    the file is absent / unparsable / inf-only. Templates emit
    ``per_case_us`` (always a list, length 1 for static-shape); we fall back
    to wrapping ``execution_time_us`` so older JSON written by the previous
    template revision still parses (transitional — drop once all task dirs
    have been re-profiled).
    """
    json_path = os.path.join(verify_dir, json_filename)
    if not os.path.exists(json_path):
        logger.error('profile JSON not found: %s', json_path)
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error('profile JSON unreadable %s: %s', json_filename, e)
        return None

    avg = _finite(data.get("avg_time_us")) or _finite(data.get("execution_time_us"))
    if avg is None:
        return None

    raw_per_case = data.get("per_case_us")
    if isinstance(raw_per_case, list) and raw_per_case:
        per_case = [c for c in (_finite(t) for t in raw_per_case) if c is not None]
    else:
        per_case = []
    if not per_case:
        per_case = [avg]
    return {
        "avg_us": avg,
        "per_case_us": per_case,
        "method": data.get("method"),
    }


async def _load_base_profile(
    verify_dir: str,
    op_name: str,
    run_script: Any,
    task_id: str,
    override: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    base_section = None
    if (override is not None
            and isinstance(override.get("avg_us"), (int, float))
            and 0 < override["avg_us"] < float("inf")):
        base_section = override
        logger.info(
            "[%s: %s] 使用缓存的 baseline: %.2f us (per_shape len=%s)",
            op_name,
            task_id,
            base_section["avg_us"],
            len(base_section.get("per_case_us") or []),
        )
        return base_section
    script_name = f"profile_{op_name}_base.py"
    if os.path.exists(os.path.join(verify_dir, script_name)):
        if await run_script(script_name, "base_profile"):
            return read_profile_result_from_json(
                verify_dir, "base_profile_result.json")
        logger.error('[%s: %s] 基准性能脚本执行失败', op_name, task_id)
        return None
    logger.info(
        "[%s: %s] 基准性能脚本不存在"
        "（使用缓存 baseline 或跨后端场景），跳过 base profile",
        op_name,
        task_id,
    )
    return None


async def _load_generation_profile(
    verify_dir: str,
    op_name: str,
    run_script: Any,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    script_name = f"profile_{op_name}_generation.py"
    if os.path.exists(os.path.join(verify_dir, script_name)):
        if await run_script(script_name, "generation_profile"):
            return read_profile_result_from_json(
                verify_dir, "generation_profile_result.json")
        logger.error('[%s: %s] 生成代码性能脚本执行失败', op_name, task_id)
        return None
    logger.info(
        '[%s: %s] 生成代码性能脚本不存在，跳过 generation profile',
        op_name,
        task_id,
    )
    return None


def _log_profile_results(
    op_name: str,
    task_id: str,
    base_section: Optional[Dict[str, Any]],
    gen_section: Optional[Dict[str, Any]],
) -> None:
    base_avg = base_section["avg_us"] if base_section else float("inf")
    gen_avg = gen_section["avg_us"] if gen_section else float("inf")
    logger.info(
        "[%s: %s] Read profile results: base=%.2f us, gen=%.2f us "
        "(base_cases=%s, gen_cases=%s)",
        op_name,
        task_id,
        base_avg,
        gen_avg,
        len(base_section["per_case_us"]) if base_section else 0,
        len(gen_section["per_case_us"]) if gen_section else 0,
    )


async def run_profile_scripts_and_collect_results(
    verify_dir: str, op_name: str, run_script, *, task_id: str = "0",
    override_base_section: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run base and generation scripts and return canonical sections."""
    base_section = await _load_base_profile(
        verify_dir,
        op_name,
        run_script,
        task_id,
        override_base_section,
    )
    gen_section = await _load_generation_profile(
        verify_dir,
        op_name,
        run_script,
        task_id,
    )
    _log_profile_results(op_name, task_id, base_section, gen_section)
    return {"base": base_section, "gen": gen_section}


def _profile_output_path(stdout: str) -> Optional[str]:
    marker = "[INFO] Process profiling data complete. Data is saved in"
    for line in stdout.splitlines():
        if marker not in line:
            continue
        match = re.search(r"Data is saved in (.+)$", line)
        if match:
            return match.group(1).strip()
    return None


def run_msprof(script_path: str, op_name: str = "", task_id: str = "0",
               timeout: Optional[int] = None,
               cancel_event=None) -> Tuple[bool, str, Optional[str]]:
    """运行msprof性能分析
    
    Args:
        script_path: Python脚本路径
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        timeout: 超时时间（秒）
        
    Returns:
        (success, error_msg, prof_path): 是否成功，错误信息，prof数据路径
    """
    timeout = resolve_eval_timeout(timeout)
    try:
        returncode, stdout, stderr, timed_out = run_command_capture(
            ["msprof", f"--application=python {script_path}"],
            CommandCaptureOptions(timeout=timeout, cancel_event=cancel_event),
        )
        if timed_out:
            return False, f"msprof timed out after {timeout} seconds", None
        if returncode != 0:
            return False, stderr or stdout or f"msprof exited with {returncode}", None

        output_path = _profile_output_path(stdout)
        if output_path:
            return True, "", output_path
        return False, "未找到数据保存路径", None
    except Exception as e:
        logger.error('[%s:%s] msprof执行错误: %s', task_id, op_name, e)
        return False, f"执行错误: {str(e)}", None


def analyze_prof_data(
    prof_path: str, warmup_times: int, run_times: int,
    op_name: str = "", task_id: str = "0",
) -> Tuple[bool, str, float]:
    """分析PROF数据
    
    Args:
        prof_path: prof数据目录路径
        warmup_times: 预热次数
        run_times: 实际运行次数
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        
    Returns:
        (success, error_msg, avg_time_us): 是否成功，错误信息，平均时间（微秒）
    """
    try:
        csv_files = list(Path(prof_path).glob("mindstudio_profiler_output/op_summary_*.csv"))
        if not csv_files:
            return False, "未找到CSV文件", 0.0

        df = pd.read_csv(csv_files[0])

        # 移除特定的Op
        df_filtered = df[~df["Op Name"].str.contains("aclnnIsClose_IsCloseAiCpu_IsClose|aclnnAll_ReduceAll_ReduceAll",
                                                     regex=True, na=False)]

        total_count = warmup_times + run_times
        op_counts = df_filtered["Op Name"].value_counts()
        valid_ops = op_counts[op_counts == total_count]

        if len(valid_ops) == 0:
            return False, "没有找到符合预期次数的Op", float('inf')

        # 检查不匹配的Op
        invalid_ops = op_counts[op_counts != total_count]
        if len(invalid_ops) > 0:
            logger.warning('[%s:%s] 发现%s个Op次数不匹配', task_id, op_name, len(invalid_ops))

        # 计算平均时间
        df_valid = df_filtered[df_filtered["Op Name"].isin(valid_ops.index)]
        total_avg_time = 0.0

        for op_name_iter in valid_ops.index:
            op_data = df_valid[df_valid["Op Name"] == op_name_iter]["Task Duration(us)"].tolist()
            if len(op_data) > warmup_times:
                valid_data = op_data[warmup_times:]
                avg_time = sum(valid_data) / len(valid_data)
                total_avg_time += avg_time

        return True, "", total_avg_time

    except Exception as e:
        logger.error('[%s:%s] 分析prof数据时出错: %s', task_id, op_name, e)
        return False, f"分析数据时出错: {str(e)}", float('inf')
