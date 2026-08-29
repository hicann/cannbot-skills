#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
Profiling 解析脚本 — 推荐模型昇腾 NPU 性能分析

功能：
  1. 自动识别 MindStudio Profiler 输出目录中的 CSV / JSON 文件
  2. 计算 H2D 耗时、ModelExecute 耗时、D2H 耗时
  3. 计算 Iteration 算子耗时和，判断调度 bound
  4. 判断是否含动态 shape
  5. 计算各种 ratio 指标（aic_scalar_ratio, aiv_scalar_ratio, cube_utilization, aic_mac_ratio）
  6. 识别 AICPU 算子
  7. 输出结构化分析报告及优化建议

用法:
  python profiling_parser.py <profiling_dir> [--output report.md]

依赖: 仅 Python 标准库（csv, json, os, re, glob, argparse）
"""

import csv
import json
import os
import re
import glob
import argparse
import logging
from collections import defaultdict, OrderedDict
from typing import Optional, List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ============================================================
#  文件发现
# ============================================================

class ProfilingFileLocator:
    """在 profiling 目录中自动发现各类性能数据文件"""

    FILE_PATTERNS = {
        "op_summary": ["*op_summary*.csv", "kernel_details.csv"],
        "op_statistic": ["*op_statistic*.csv", "op_statistic.csv"],
        "api_statistic": ["api_statistic_*.csv", "api_statistic.csv"],
        "operator_details": ["operator_details.csv"],
        "step_trace": ["step_trace_*.csv", "step_trace_time.csv"],
        "msprof_json": ["msprof_*.json", "trace_view.json"],
        "step_trace_json": ["step_trace_*.json"],
        "fusion_op": ["fusion_op_*.csv"],
        "task_time": ["task_time_*.csv", "task_time.csv"],
        "aicore_util": ["ai_core_utilization_*.csv"],
        "aivector_util": ["ai_vector_core_utilization_*.csv"],
        "aicpu": ["aicpu_*.csv"],
    }

    def __init__(self, profiling_dir: str):
        self.profiling_dir = profiling_dir
        self.files: Dict[str, Optional[str]] = {}


    @staticmethod
    def _has_non_empty_op_name(reader, header: List[str]) -> bool:
        """检查 reader 中是否有非空 Op Name 值"""
        idx = header.index("Op Name")
        for row in reader:
            if idx < len(row) and row[idx].strip():
                return True
        return False

    def discover(self) -> Dict[str, Optional[str]]:
        """递归搜索 profiling 目录，返回 {数据类型: 文件路径}"""
        for data_type, patterns in self.FILE_PATTERNS.items():
            self.files[data_type] = self._find_matching_file(data_type, patterns)
        return self.files

    def get(self, data_type: str) -> Optional[str]:
        return self.files.get(data_type)

    def summary(self) -> str:
        lines = ["### 发现的 Profiling 文件\n"]
        lines.append("| 数据类型 | 文件路径 |")
        lines.append("|----------|----------|")
        for dt, path in self.files.items():
            status = path if path else "(未找到)"
            lines.append(f"| {dt} | {status} |")
        return "\n".join(lines)

    def _find_matching_file(self, data_type: str, patterns: List[str]) -> Optional[str]:
        """在 patterns 中查找第一个匹配的文件"""
        for pattern in patterns:
            matches = glob.glob(
                os.path.join(self.profiling_dir, "**", pattern),
                recursive=True
            )
            if matches:
                if data_type == "op_summary" and len(matches) > 1:
                    return self._pick_best_op_summary(matches)
                return matches[0]
        return None

    def _score_op_summary_header(self, reader, header: List[str]) -> int:
        """根据 header 内容评分"""
        score = 0
        if "Op Name" in header:
            score += 10
            if self._has_non_empty_op_name(reader, header):
                score += 5
        return score

    def _try_read_op_summary_header(self, path: str, encoding: str) -> Optional[int]:
        """尝试用指定编码读取 header 并评分，失败返回 None"""
        try:
            with open(path, "r", encoding=encoding) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return 0
                return self._score_op_summary_header(reader, header)
        except (UnicodeDecodeError, csv.Error, OSError):
            logging.warning("读取 CSV 失败 (编码 %s): %s", encoding, path)
            return None

    def _check_op_summary_columns(self, path: str) -> int:
        """检查文件是否含 "Op Name" 列且有非空数据，返回加分"""
        for encoding in ["utf-8-sig", "utf-8", "gbk"]:
            result = self._try_read_op_summary_header(path, encoding)
            if result is not None:
                return result
        return 0

    def _score_op_summary_file(self, path: str) -> int:
        """对单个 op_summary 候选文件评分"""
        score = 0
        if "_output_" in os.path.basename(path).lower():
            score -= 100
        if "_no_op_name" in os.path.basename(path).lower():
            score -= 50
        try:
            score += self._check_op_summary_columns(path)
        except Exception:
            logging.warning("评分 op_summary 文件时发生异常: %s", path)
            pass
        return score

    def _pick_best_op_summary(self, matches: List[str]) -> str:
        """从多个 op_summary 候选文件中选取最佳：
        优先选含 "Op Name" 列且数据非空的文件，避免选到 _no_op_name 或 _output_ 文件。
        如果所有候选都被过滤，仍返回最优的一个（而非 matches[0]）。
        """
        scored: List[Tuple[int, str]] = []
        for path in matches:
            scored.append((self._score_op_summary_file(path), path))
        if not scored:
            return matches[0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


# ============================================================
#  CSV 读取工具
# ============================================================

def _normalize_csv_row(row: List[str], header_len: int) -> Optional[List[str]]:
    """对齐 CSV 行长度到 header_len，跳过空行"""
    if not row or all(c.strip() == "" for c in row):
        return None
    if len(row) < header_len:
        row = row + [""] * (header_len - len(row))
    elif len(row) > header_len:
        row = row[:header_len]
    return row


def _read_csv_with_encoding(filepath: str, encoding: str) -> Optional[Tuple[List[str], List[Dict[str, str]]]]:
    """用指定编码读取 CSV，失败返回 None"""
    with open(filepath, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return [], []
        header_len = len(header)
        rows = []
        for row in reader:
            normed = _normalize_csv_row(row, header_len)
            if normed is not None:
                rows.append(dict(zip(header, normed)))
    return header, rows


def read_csv_safe(filepath: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """安全读取 CSV 文件，返回 (header, rows_as_dicts)

    健壮性保障：
    - 多编码回退（utf-8-sig → utf-8 → gbk → latin-1）
    - 行列数不匹配时按 header 长度截断或补空，不丢弃整行
    - 跳过完全空行
    """
    if not filepath or not os.path.exists(filepath):
        return [], []
    for encoding in ["utf-8-sig", "utf-8", "gbk", "latin-1"]:
        try:
            return _read_csv_with_encoding(filepath, encoding)
        except (UnicodeDecodeError, csv.Error, OSError):
            logging.warning("读取 CSV 失败 (编码 %s): %s", encoding, filepath)
            continue
    return [], []


def safe_float(val: str) -> float:
    """安全转换为 float，处理 N/A、空值等"""
    if val is None:
        return 0.0
    val = str(val).strip()
    if val in ("", "N/A", "n/a", "NA", "None", "null"):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def safe_int(val: str) -> int:
    return int(safe_float(val))


# ============================================================
#  Profiling 解析器
# ============================================================

class ProfilingParser:
    """解析 MindStudio Profiler 输出，计算推荐模型核心性能指标"""

    AIC_SCALAR_FIELDS = ["aic_scalar_ratio", "scalar_ratio"]

    AIV_SCALAR_FIELDS = ["aiv_scalar_ratio", "scalar_ratio"]

    MAC_FIELDS = ["aic_mac_ratio", "mac_ratio", "mac_fp16_ratio"]

    VEC_FIELDS = ["aic_vec_ratio", "vec_ratio", "vec_fp32_ratio"]

    CUBE_UTIL_FIELDS = ["cube_utilization(%)", "cube_utilization"]

    MTE2_AIC_FIELDS = ["aic_mte2_ratio", "mte2_ratio"]

    MTE2_AIV_FIELDS = ["aiv_mte2_ratio", "mte2_ratio"]

    MTE3_FIELDS = ["aiv_mte3_ratio", "mte3_ratio"]

    RATIO_IS_PERCENT = {"cube_utilization(%)": True, "cube_utilization": True}

    SCALAR_THRESHOLD = 0.30      # 30% -> 0.30

    CUBE_UTIL_THRESHOLD = 20.0   # 20% -> 20 (percent field)

    MAC_RATIO_THRESHOLD = 0.30   # 30% -> 0.30

    FUSION_PREFIXES = ["autofuse_", "autofused_", "triton_poi", "triton_per", "triton_unk_fused", "dvm_"]

    def __init__(self, profiling_dir: str, graph_file: Optional[str] = None):
        self.locator = ProfilingFileLocator(profiling_dir)
        self.locator.discover()
        self.report: Dict[str, Any] = OrderedDict()
        self._api_rows_cache: Optional[List[Dict[str, str]]] = None
        self._h2d_cache: Optional[Dict[str, Any]] = None
        self.graph_file = graph_file

    @staticmethod
    def _check_model_execute_in_api_rows(api_rows: List[Dict[str, str]]) -> bool:
        """检查 api_statistic 中是否有 count > 1 的 ModelExecute"""
        if not api_rows:
            return False
        for row in api_rows:
            api_name = str(row.get("API Name", "")).strip()
            if "ModelExecute" in api_name:
                count = safe_int(row.get("Count", 0))
                if count > 1:
                    return True
        return False

    @staticmethod
    def _infer_from_infer_id(rows: List[Dict[str, str]], header: List[str]) -> Optional[Dict[str, Any]]:
        if "Infer ID" not in header:
            return None
        infer_ids = set()
        for row in rows:
            val = str(row.get("Infer ID", "")).strip()
            if val and val not in ("", "N/A", "0"):
                infer_ids.add(val)
        if len(infer_ids) > 1:
            return {"count": len(infer_ids), "source": f"op_summary Infer ID unique count={len(infer_ids)}"}
        return None

    @staticmethod
    def _infer_from_step_id(rows: List[Dict[str, str]], header: List[str]) -> Optional[Dict[str, Any]]:
        if "Step Id" not in header:
            return None
        step_ids = set()
        for row in rows:
            val = str(row.get("Step Id", "")).strip()
            if val and val not in ("", "N/A"):
                step_ids.add(val)
        if len(step_ids) > 1:
            return {"count": len(step_ids), "source": f"kernel_details Step Id unique count={len(step_ids)}"}
        return None

    @staticmethod
    def _infer_from_op_name_mode(rows: List[Dict[str, str]], header: List[str]) -> Optional[Dict[str, Any]]:
        from collections import Counter as _Counter
        if "Op Name" not in header:
            return None
        op_name_counter = _Counter()
        for row in rows:
            name = str(row.get("Op Name", "")).strip()
            if name:
                op_name_counter[name] += 1
        if op_name_counter:
            count_dist = _Counter(op_name_counter.values())
            mode_iter_count = count_dist.most_common(1)[0][0]
            unique_op_names = len(op_name_counter)
            if mode_iter_count > 1:
                return {
                    "count": mode_iter_count,
                    "source": (
                        f"op_summary: Op Name 众数出现次数={mode_iter_count}"
                        f" (唯一OpName={unique_op_names}, 总行数={len(rows)})"
                    ),
                }
        return None

    @staticmethod
    def _infer_iter_from_api_sync(api_rows: List[Dict[str, str]]) -> int:
        """从 api_statistic 获取迭代次数"""
        if not api_rows:
            return 0
        for row in api_rows:
            api_name = str(row.get("API Name", "")).strip()
            if "ModelExecute" in api_name:
                c = safe_int(row.get("Count", 0))
                if c > 1:
                    return c
        for row in api_rows:
            api_name = str(row.get("API Name", "")).strip()
            if api_name in ("StreamSynchronize", "RunGraphAsync"):
                c = safe_int(row.get("Count", 0))
                if c > 1:
                    return c
        best = 0
        for row in api_rows:
            name = str(row.get("API Name", "")).strip()
            if "Synchronize" in name or name.startswith("aclrtSync"):
                c = safe_int(row.get("Count", 0))
                if c > best:
                    best = c
        if best > 1:
            return best
        return 0

    @staticmethod
    def _load_json_with_encodings(json_file: str):
        """尝试多种编码读取 JSON，成功返回 data，全失败返回 None"""
        for encoding in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                with open(json_file, "r", encoding=encoding) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                logging.warning("读取 JSON 失败 (编码 %s): %s", encoding, json_file)
                continue
        return None

    @staticmethod
    def _match_json_events(trace_events, name_patterns: List[str]) -> List[Dict]:
        """从 trace_events 中提取匹配 name_patterns 的事件"""
        events = []
        for evt in trace_events:
            evt_name = str(evt.get("name", ""))
            for pattern in name_patterns:
                if pattern.lower() in evt_name.lower():
                    events.append({
                        "name": evt_name,
                        "dur": safe_float(evt.get("dur", 0)),
                        "ts": safe_float(evt.get("ts", 0)),
                        "cat": evt.get("cat", ""),
                        "args": evt.get("args", {}),
                    })
                    break
        return events

    @staticmethod
    def _extract_json_events(json_file: str, name_patterns: List[str]) -> List[Dict]:
        """从 msprof_*.json (chrome trace 格式) 中提取匹配的事件"""
        events = []
        try:
            data = ProfilingParser._load_json_with_encodings(json_file)
            if data is None:
                return events

            trace_events = data.get("traceEvents", data) if isinstance(data, dict) else data
            if not isinstance(trace_events, list):
                return events

            events = ProfilingParser._match_json_events(trace_events, name_patterns)
        except Exception:
            logging.warning("解析 JSON 事件时发生异常: %s", json_file)

        return events

    @staticmethod
    def _calc_step_trace_eager(rows):
        """从 step_trace_time.csv (eager/inductor) 提取 Computing 和 Wall 统计"""
        computing_vals, wall_vals, free_vals, comm_vals, prep_vals = [], [], [], [], []
        for row in rows:
            wall = safe_float(row.get("Stage", 0))
            if wall > 0:
                computing_vals.append(safe_float(row.get("Computing", 0)))
                wall_vals.append(wall)
                free_vals.append(safe_float(row.get("Free", 0)))
                comm_vals.append(safe_float(row.get("Communication", 0)))
                prep_vals.append(safe_float(row.get("Preparing", 0)))
        eager_summary = None
        if wall_vals:
            n = len(wall_vals)
            eager_summary = {
                "wall_us": sum(wall_vals),
                "wall_avg_us": sum(wall_vals) / n,
                "computing_us": sum(computing_vals),
                "computing_avg_us": sum(computing_vals) / n,
                "free_us": sum(free_vals),
                "comm_us": sum(comm_vals),
                "prep_us": sum(prep_vals),
                "step_count": n,
            }
        return computing_vals, eager_summary

    @staticmethod
    def _calc_step_trace_ge(rows):
        """从 step_trace_*.csv (GE) 提取逐次迭代耗时"""
        iter_times = []
        for row in rows:
            for col in ["Iteration Time(us)", "Iteration Time", "iteration_time"]:
                if col in row and str(row[col]).strip() not in ("", "N/A"):
                    iter_times.append(safe_float(row[col]))
                    break
        return iter_times

    @staticmethod
    def _strip_compile_iter(iter_times, is_step_trace_time):
        """检测并移除首次编译迭代，返回 (iter_times, compile_iter_us, source)"""
        compile_iter_us = 0.0
        source = ""
        if len(iter_times) >= 2:
            rest_avg = sum(iter_times[1:]) / (len(iter_times) - 1)
            if rest_avg > 0 and iter_times[0] > 5 * rest_avg:
                compile_iter_us = iter_times[0]
                iter_times = iter_times[1:]
        if compile_iter_us > 0:
            source = ("step_trace_time.csv: Computing (排除首次编译迭代)"
                      if is_step_trace_time
                      else "step_trace.csv: Iteration Time(us) (排除首次编译迭代)")
        else:
            source = ("step_trace_time.csv: Computing (NPU活跃)"
                      if is_step_trace_time
                      else "step_trace.csv: Iteration Time(us)")
        return iter_times, compile_iter_us, source

    @staticmethod
    def _calc_from_api_model_execute(api_rows):
        """方法2: api_statistic ModelExecute"""
        for row in api_rows:
            api_name = str(row.get("API Name", "")).strip()
            if "ModelExecute" in api_name:
                count = safe_int(row.get("Count", 0))
                avg = safe_float(row.get("Avg(us)", 0))
                if count > 1 and avg > 0:
                    return avg, count, "api_statistic: ModelExecute"
        return None

    @staticmethod
    def _calc_from_api_run_graph(api_rows):
        """方法3: RunGraphAsync + StreamSynchronize"""
        run_graph = None
        stream_sync = None
        for row in api_rows:
            name = str(row.get("API Name", "")).strip()
            if name == "RunGraphAsync":
                run_graph = row
            elif name == "StreamSynchronize":
                stream_sync = row
        if run_graph and stream_sync:
            rg_total = safe_float(run_graph.get("Time(us)", 0))
            ss_total = safe_float(stream_sync.get("Time(us)", 0))
            count = safe_int(stream_sync.get("Count", 0))
            if count > 1:
                return ((rg_total + ss_total) / count, count,
                        "api_statistic: RunGraphAsync + StreamSynchronize")
        return None

    @staticmethod
    def _find_best_sync_count(api_rows):
        """从 api_statistic 找最大的 Synchronize 调用次数"""
        best = 0
        for row in api_rows:
            name = str(row.get("API Name", "")).strip()
            if "Synchronize" in name or name.startswith("aclrtSync"):
                c = safe_int(row.get("Count", 0))
                if c > best:
                    best = c
        return best

    @staticmethod
    def _build_col_map(header: List[str]) -> Dict[str, str]:
        is_kernel_details = "Name" in header and "Accelerator Core" in header
        if is_kernel_details:
            return {
                "Op Name": "Name", "OP Type": "Type",
                "Task Type": "Accelerator Core",
                "Task Duration(us)": "Duration(us)",
                "OP State": "OP State", "Input Shapes": "Input Shapes",
                "Input Data Types": "Input Data Types",
                "Output Shapes": "Output Shapes",
                "Output Data Types": "Output Data Types",
            }
        col_map = {}
        for k in ["Op Name", "OP Type", "Task Type", "Task Duration(us)",
                   "OP State", "Input Shapes", "Input Data Types",
                   "Output Shapes", "Output Data Types"]:
            col_map[k] = k
        return col_map

    @staticmethod
    def _agg_problem(target_list, ctx, col_map):
        ptype = ctx["ptype"]
        pvalue = ctx["pvalue"]
        op_name = ctx["op_name"]
        op_type = ctx["op_type"]
        task_type = ctx["task_type"]
        task_dur = ctx["task_dur"]
        in_shapes = ctx["row"].get(col_map.get("Input Shapes", "Input Shapes"), "")
        out_shapes = ctx["row"].get(col_map.get("Output Shapes", "Output Shapes"), "")
        key = (ptype, op_name, op_type, task_type, in_shapes, out_shapes)
        if key not in target_list:
            target_list[key] = {
                "problem_type": ptype, "problem_value": pvalue,
                "op_name": op_name, "op_type": op_type, "task_type": task_type,
                "input_shapes": in_shapes, "output_shapes": out_shapes,
                "count": 0, "total_us": 0.0,
            }
        target_list[key]["count"] += 1
        target_list[key]["total_us"] += task_dur

    @staticmethod
    def _build_op_type_distribution(op_type_counter, result):
        sorted_types = sorted(op_type_counter.items(), key=lambda x: x[1]["total_us"], reverse=True)
        for op_type, stats in sorted_types[:20]:
            result["op_type_distribution"][op_type] = stats

    @staticmethod
    def _classify_fusion_prefix(prefix: str, result: Dict[str, Any]):
        """根据融合前缀设置对应的 enabled 标志"""
        if prefix in ("autofuse_", "autofused_"):
            result["autofuse_enabled"] = True
        elif prefix in ("triton_poi", "triton_per", "triton_unk_fused"):
            result["triton_fusion_enabled"] = True
        elif prefix == "dvm_":
            result["dvm_fusion_enabled"] = True

    @staticmethod
    def _detect_from_autofuse(fusion: Dict[str, Any], flags: Dict[str, bool],
                              result: Dict[str, Any]) -> bool:
        """从融合算子前缀推断执行模式，返回 True 表示已判定"""
        has_triton = flags.get("has_triton", False)
        has_dvm = flags.get("has_dvm", False)
        has_autofused = flags.get("has_autofused", False)
        has_model_execute = flags.get("has_model_execute", False)
        fusion_count = len(fusion.get('fusion_op_types', []))
        if has_triton:
            result["mode"] = "graph_inductor_triton"
            result["mode_label"] = "图模式 (Inductor + Triton)"
            result["evidence"].append(
                f"op_statistic: 含 triton_poi/triton_per/triton_unk_fused 前缀融合算子 ({fusion_count}种)")
            return True
        if has_dvm:
            result["mode"] = "graph_inductor_dvm"
            result["mode_label"] = "图模式 (Inductor + DVM)"
            result["evidence"].append(f"op_statistic: 含 dvm_ 前缀融合算子 ({fusion_count}种)")
            return True
        if has_autofused and has_model_execute:
            result["mode"] = "graph_autofuse_ge"
            result["mode_label"] = "图模式 (GE/TorchAir/ATC AutoFuse)"
            result["evidence"].append(f"op_statistic: 含 autofuse_/autofused_ 前缀融合算子 ({fusion_count}种)")
            result["evidence"].append("api_statistic: ModelExecute")
            return True
        if has_autofused and not has_model_execute:
            result["mode"] = "graph_inductor_ascendc"
            result["mode_label"] = "图模式 (Inductor + AscendC)"
            result["evidence"].append(f"op_statistic: 含 autofuse_/autofused_ 前缀融合算子 ({fusion_count}种)，无 ModelExecute")
            return True
        return False

    @staticmethod
    def _detect_from_model_execute(has_model_execute: bool, result: Dict[str, Any]) -> bool:
        """从 ModelExecute 推断执行模式，返回 True 表示已判定"""
        if has_model_execute:
            result["mode"] = "graph_ge_torchair"
            result["mode_label"] = "图模式 (GE/TorchAir)"
            result["evidence"].append("api_statistic: ModelExecute (走GE/TorchAir)")
            return True
        return False

    @staticmethod
    def _format_eager_summary(s: Dict[str, Any]) -> List[str]:
        """生成 eager summary 相关的表格行"""
        lines = []
        eager_sum = s.get("eager_summary")
        if not eager_sum or eager_sum.get("wall_us", 0) <= 0:
            return lines
        step_cnt = eager_sum.get("step_count", 1) or 1
        comp_per = eager_sum.get(
            "computing_avg_us",
            eager_sum["computing_us"] / step_cnt,
        )
        free_per = eager_sum["free_us"] / step_cnt
        wall_per = eager_sum.get(
            "wall_avg_us",
            eager_sum["wall_us"] / step_cnt,
        )
        lines.append(
            f"| &emsp;Computing (NPU活跃) | {comp_per:.2f} |"
            f" {comp_per/wall_per*100:.1f}% (占Wall) |"
            " step_trace_time per-step |"
        )
        lines.append(
            f"| &emsp;Free (NPU空闲) | {free_per:.2f} |"
            f" {free_per/wall_per*100:.1f}% (占Wall) |"
            " step_trace_time per-step |"
        )
        lines.append(
            f"| &emsp;Wall (含空闲) | {wall_per:.2f} |"
            " - | step_trace_time per-step |"
        )
        return lines

    @staticmethod
    def _format_metrics_table(r: Dict[str, Any], s: Dict[str, Any]) -> List[str]:
        """生成核心指标表格行"""
        lines = []
        h2d_src = r.get('h2d', {}).get('source', '')
        lines.append(
            f"| H2D 平均耗时 | {s.get('h2d_avg_us', 0):.2f} |"
            f" {s.get('h2d_ratio_pct', 0):.1f}% | {h2d_src} |"
        )
        me_src = r.get('model_execute', {}).get('source', '')
        lines.append(
            f"| ModelExecute 平均 | {s.get('model_execute_avg_us', 0):.2f} |"
            f" {s.get('model_execute_ratio_pct', 0):.1f}% | {me_src} |"
        )
        d2h_src = r.get('d2h', {}).get('source', '')
        lines.append(
            f"| D2H 平均耗时 | {s.get('d2h_avg_us', 0):.2f} |"
            f" {s.get('d2h_ratio_pct', 0):.1f}% | {d2h_src} |"
        )
        lines.append(
            f"| **单实例迭代总耗时** | **{s.get('total_iter_avg_us', 0):.2f}**"
            f" | - | H2D + ModelExecute + D2H |"
        )
        op_src = r.get('op_total_time', {}).get('source', '')
        lines.append(
            f"| Iteration 算子耗时和 | {s.get('op_total_avg_us', 0):.2f} |"
            f" {s.get('op_total_ratio_pct', 0):.1f}% (占ModelExecute) | {op_src} |"
        )
        return lines

    @staticmethod
    def _format_fusion_section(r: Dict[str, Any]) -> List[str]:
        """生成执行模式/融合/吞吐/Free警告"""
        lines = []
        s = r.get("summary", {})
        eager_sum = s.get("eager_summary")
        lines.append(f"- **含动态 shape**: {'是' if s.get('has_dynamic_shape') else '否'}")
        fusion = r.get("fusion", {})
        exec_mode = r.get("execution_mode", {})
        any_fusion = (fusion.get("autofuse_enabled", False)
                      or fusion.get("triton_fusion_enabled", False)
                      or fusion.get("ascendc_fusion_enabled", False))
        fusion_type = fusion.get("fusion_type", "")
        lines.append(f"- **执行模式**: {exec_mode.get('mode_label', '未知')}")
        if exec_mode.get("evidence"):
            lines.append(f"  - 判断依据: {'; '.join(exec_mode['evidence'])}")
        fusion_label = ('是 (' + fusion_type + ')') if any_fusion else '否'
        lines.append(f"- **是否开启自动融合**: {fusion_label}")
        if fusion.get("fusion_op_types"):
            ftypes = fusion['fusion_op_types'][:5]
            suffix = '...' if len(fusion['fusion_op_types']) > 5 else ''
            lines.append(
                f"  - 融合算子类型: {', '.join(ftypes)}{suffix}"
            )
        lines.append(f"- **迭代次数**: {s.get('iteration_count', 0)}")
        lines.append(f"- **数据来源**: {r.get('model_execute', {}).get('source', '')}")
        total_us = s.get("total_iter_avg_us", 0)
        if total_us > 0:
            single_throughput = 1000.0 / total_us
            lines.append(f"- **单实例吞吐(估)**: {single_throughput:.2f} iter/s")
        if eager_sum and eager_sum.get("free_us", 0) > 0 and eager_sum.get("wall_us", 0) > 0:
            free_pct = eager_sum["free_us"] / eager_sum["wall_us"] * 100
            if free_pct > 30:
                lines.append(f"\n> ⚠️ **NPU空闲占比{free_pct:.0f}%**，建议多实例并行或增大BS填满NPU")
        lines.append("")
        return lines

    @staticmethod
    def _format_op_distribution(r: Dict[str, Any]) -> List[str]:
        """生成算子类型分布 + Top 耗时算子"""
        lines = []
        s = r.get("summary", {})
        iter_count = s.get("iteration_count", 1) or 1
        op_m = r.get("op_metrics", {})
        lines.append("## 3. 算子类型分布 (Top 20)\n")
        lines.append("| OP Type | 总Count | Count/iter | Total(us) | us/iter |"
                     " 占算子耗时比 |")
        lines.append("|---------|:-------:|:----------:|:---------:|:-------:|:----------:|")
        op_total_all = sum(stats['total_us'] for stats in op_m.get("op_type_distribution", {}).values())
        for op_type, stats in op_m.get("op_type_distribution", {}).items():
            per_iter_us = stats['total_us'] / iter_count
            per_iter_count = stats['count'] // iter_count
            pct = (stats['total_us'] / op_total_all * 100) if op_total_all > 0 else 0
            lines.append(
                f"| {op_type} | {stats['count']} | {per_iter_count} |"
                f" {stats['total_us']:.1f} | {per_iter_us:.1f} | {pct:.1f}% |"
            )
        lines.append("")
        lines.append("## 4. Top 20 耗时算子（按 算子类型+shape+dtype 去重）\n")
        lines.append("| # | Op Name | OP Type | Task Type | Dur(us) |"
                     " Shape | Dtype | scalar | mac | mte2 | cube |")
        lines.append("|:--:|---------|---------|-----------|"
                     ":-------:|-------|-------|:------:|:---:|:----:|:----:|")
        for i, op in enumerate(op_m.get("top_ops_by_duration", [])[:20], 1):
            shapes = op.get("input_shapes", "")[:30]
            dtypes = op.get("input_dtypes", "")[:15]
            sr = op.get("aic_scalar_ratio", 0) or op.get("aiv_scalar_ratio", 0)
            mr = op.get("aic_mac_ratio", 0)
            mte2 = op.get("aic_mte2_ratio", 0)
            cu = op.get("cube_utilization", 0)
            lines.append(
                f"| {i} | `{op['op_name'][:50]}` | {op['op_type']} |"
                f" {op['task_type']} | {op['task_duration_us']:.1f} |"
                f" {shapes} | {dtypes} | {sr:.1%} | {mr:.1%} |"
                f" {mte2:.1%} | {cu:.1f}% |"
            )
        lines.append("")
        return lines

    @staticmethod
    def _format_problem_ops(r: Dict[str, Any]) -> List[str]:
        """生成问题算子汇总表"""
        lines = []
        s = r.get("summary", {})
        iter_count = s.get("iteration_count", 1) or 1
        op_m = r.get("op_metrics", {})
        problem_agg = op_m.get("problem_ops_agg", {})
        if not problem_agg:
            return lines
        agg_list = list(problem_agg.values())
        agg_list.sort(key=lambda x: x["total_us"], reverse=True)
        lines.append(f"## 5. 问题算子汇总 ({len(agg_list)} 组)\n")
        lines.append("| 问题类型 | Op Name | OP Type | Task Type |"
                     " Input Shapes | Output Shapes | 重复次数/iter |"
                     " 总耗时/iter(us) |")
        lines.append("|----------|---------|---------|-----------|"
                     "-------------|--------------|:------------:|"
                     ":--------------:|")
        for op in agg_list[:50]:
            count_per_iter = round(op["count"] / iter_count, 1) if iter_count > 0 else op["count"]
            us_per_iter = op["total_us"] / iter_count if iter_count > 0 else op["total_us"]
            in_s = op.get("input_shapes", "")[:60]
            out_s = op.get("output_shapes", "")[:60]
            lines.append(
                f"| {op['problem_type']} | `{op.get('op_name', '')[:50]}` |"
                f" {op.get('op_type', '')} | {op.get('task_type', '')} |"
                f" {in_s} | {out_s} | {count_per_iter} |"
                f" {us_per_iter:.1f} |"
            )
        lines.append("")
        return lines

    @staticmethod
    def _format_host_overhead(r: Dict[str, Any]) -> List[str]:
        """生成 Host 侧开销分析表"""
        lines = []
        host_apis = r.get("host_overhead", [])
        if not host_apis:
            return lines
        lines.append("## 6. Host侧开销分析 (Top 15)\n")
        lines.append("> ModelExecute内部 = 算子耗时 + Host调度/框架开销。以下为Host侧主要耗时API\n")
        lines.append("| Host API | Level | per-iter(us) | calls/iter | 总耗时(us) |")
        lines.append("|:---------|:-----:|:----------:|:----------:|:---------:|")
        for api in host_apis:
            lines.append(
                f"| {api['api_name']} | {api['level']} |"
                f" {api['per_iter_us']:.1f} | {api['calls_per_iter']} |"
                f" {api['total_us']:.1f} |"
            )
        lines.append("")
        return lines

    def calc_h2d(self) -> Dict[str, Any]:
        """计算 H2D（Host to Device）耗时
        方法1: api_statistic 中 InputCopy
        方法2: api_statistic 中 Memcpy* 系列（客户自调aclrtMemcpy）
        方法3: msprof.json 中 InputCopy
        """
        result = {"h2d_total_us": 0.0, "h2d_avg_us": 0.0, "h2d_count": 0, "source": ""}

        api_rows = self._get_api_rows()

        if api_rows:
            for row in api_rows:
                api_name = str(row.get("API Name", "")).strip()
                if "InputCopy" in api_name or "H2D" in api_name.upper():
                    total = safe_float(row.get("Total(us)", row.get("Total Time(us)", row.get("Time(us)", 0))))
                    count = safe_int(row.get("Count", 0))
                    result["h2d_total_us"] = total
                    result["h2d_avg_us"] = (total / count) if count > 0 else 0
                    result["h2d_count"] = count
                    result["source"] = f"api_statistic: {api_name}"
                    self._h2d_cache = result
                    return result

        iter_count = self._get_iter_count()
        if api_rows and iter_count > 0:
            memcpy_total = 0.0
            memcpy_count = 0
            memcpy_details = []
            for row in api_rows:
                api_name = str(row.get("API Name", "")).strip()
                if "Memcpy" in api_name or "MemCopy" in api_name:
                    total = safe_float(row.get("Total(us)", row.get("Time(us)", 0)))
                    count = safe_int(row.get("Count", 0))
                    avg = safe_float(row.get("Avg(us)", 0))
                    memcpy_total += total
                    memcpy_count += count
                    memcpy_details.append(f"{api_name}:{total:.1f}us/{count}calls")
            if memcpy_total > 0:
                result["h2d_total_us"] = memcpy_total
                result["h2d_avg_us"] = memcpy_total / iter_count
                result["h2d_count"] = memcpy_count
                result["source"] = f"api_statistic Memcpy fallback ({'; '.join(memcpy_details)})"
                self._h2d_cache = result
                return result

        json_file = self.locator.get("msprof_json")
        if json_file:
            result = self._extract_h2d_from_json(json_file)

        return result

    def calc_d2h(self) -> Dict[str, Any]:
        """计算 D2H（Device to Host）耗时
        方法1: api_statistic 中 OutputCopy
        方法2: api_statistic 中 Memcpy 系列（客户自调aclrtMemcpy，与H2D共享）
        方法3: msprof.json 中 OutputCopy
        """
        result = {"d2h_total_us": 0.0, "d2h_avg_us": 0.0, "d2h_count": 0, "source": ""}

        api_rows = self._get_api_rows()

        # 方法1: api_statistic_*.csv 中的 OutputCopy
        if api_rows:
            for row in api_rows:
                api_name = str(row.get("API Name", "")).strip()
                if "OutputCopy" in api_name or "D2H" in api_name.upper():
                    total = safe_float(row.get("Total(us)", row.get("Total Time(us)", row.get("Time(us)", 0))))
                    count = safe_int(row.get("Count", 0))
                    result["d2h_total_us"] = total
                    result["d2h_avg_us"] = (total / count) if count > 0 else 0
                    result["d2h_count"] = count
                    result["source"] = f"api_statistic: {api_name}"
                    return result

        # 方法2: 若无 OutputCopy，H2D 已通过 Memcpy 统计了全部拷贝
        h2d = self._h2d_cache if self._h2d_cache else self.calc_h2d()
        if h2d["source"] and "Memcpy" in h2d["source"]:
            result["source"] = "D2H merged into H2D Memcpy (无法单独拆分)"
            result["d2h_avg_us"] = 0.0
            return result

        # 方法3: msprof_*.json 中的 OutputCopy
        json_file = self.locator.get("msprof_json")
        if json_file:
            d2h_events = self._extract_json_events(json_file, ["OutputCopy", "D2H"])
            if d2h_events:
                total = sum(e["dur"] for e in d2h_events)
                result["d2h_total_us"] = total * 1000  # ms -> us
                result["d2h_avg_us"] = (total / len(d2h_events) * 1000) if d2h_events else 0
                result["d2h_count"] = len(d2h_events)
                result["source"] = "msprof.json: OutputCopy"

        return result

    def calc_model_execute(self) -> Dict[str, Any]:
        """计算 ModelExecute / Iteration 耗时"""
        result = {"iteration_avg_us": 0.0, "iteration_count": 0,
                  "source": "", "eager_summary": None}
        step_file = self.locator.get("step_trace")
        if step_file:
            header, rows = read_csv_safe(step_file)
            is_step_trace_time = "Stage" in header and "Computing" in header
            if is_step_trace_time:
                iter_times, eager_summary = self._calc_step_trace_eager(rows)
            else:
                iter_times = self._calc_step_trace_ge(rows)
                eager_summary = None
            if eager_summary:
                result["eager_summary"] = eager_summary
            if iter_times:
                iter_times, compile_us, source = self._strip_compile_iter(iter_times, is_step_trace_time)
                result["iteration_avg_us"] = sum(iter_times) / len(iter_times)
                result["iteration_count"] = len(iter_times)
                if compile_us > 0:
                    result["compile_iter_us"] = compile_us
                result["source"] = source
                return result
        api_rows = self._get_api_rows()
        if api_rows:
            me = self._calc_from_api_model_execute(api_rows)
            if me:
                result["iteration_avg_us"], result["iteration_count"], result["source"] = me
                return result
            rg = self._calc_from_api_run_graph(api_rows)
            if rg:
                result["iteration_avg_us"], result["iteration_count"], result["source"] = rg
                return result
        if result.get("eager_summary") and api_rows:
            sync_count = self._find_best_sync_count(api_rows)
            es = result["eager_summary"]
            if sync_count > 1 and es["wall_us"] > 0:
                result["iteration_avg_us"] = es["wall_us"] / sync_count
                result["iteration_count"] = sync_count
                result["source"] = f"eager: step_trace_time wall/{sync_count} sync calls"
                return result
        self._calc_me_from_msprof_json(result)
        self._calc_me_fallback(result, api_rows)
        return result

    def calc_op_total_time(self, iteration_count: int) -> Dict[str, Any]:
        """计算单次 Iteration 的算子耗时和"""
        result = {"op_total_avg_us": 0.0, "source": ""}

        # 方法1: op_statistic_*.csv 的 Total Time(us) 求和 / iteration_count
        stat_file = self.locator.get("op_statistic")
        if stat_file and iteration_count > 0:
            _, rows = read_csv_safe(stat_file)
            total = sum(safe_float(row.get("Total Time(us)", 0)) for row in rows)
            result["op_total_avg_us"] = total / iteration_count
            result["source"] = "op_statistic.csv: Total Time(us) sum / iter_count"
            return result

        # 方法2: op_summary_*.csv 或 kernel_details.csv 的 Duration 求和 / iteration_count
        op_file = self.locator.get("op_summary")
        if op_file and iteration_count > 0:
            header, rows = read_csv_safe(op_file)
            # kernel_details用Duration(us), op_summary用Task Duration(us)
            dur_col = "Duration(us)" if "Duration(us)" in (header or []) else "Task Duration(us)"
            total = sum(safe_float(row.get(dur_col, 0)) for row in rows)
            result["op_total_avg_us"] = total / iteration_count
            result["source"] = f"{os.path.basename(op_file)}: {dur_col} sum / iter_count"

        return result

    def check_dynamic_shape(self) -> Dict[str, Any]:
        """判断是否含动态 shape"""
        result = {"has_dynamic_shape": False, "evidence": []}

        # 方法1: op_summary/kernel_details 中的 OP State 含 dynamic
        op_file = self.locator.get("op_summary")
        if op_file:
            _, rows = read_csv_safe(op_file)
            dynamic_count = 0
            for row in rows:
                op_state = str(row.get("OP State", "")).strip().lower()
                if "dynamic" in op_state:
                    dynamic_count += 1
            if dynamic_count > 0:
                result["has_dynamic_shape"] = True
                result["evidence"].append(
                    f"op_summary: {dynamic_count} 个算子 OP State=dynamic"
                )

        # 方法2: msprof_*.json 中有 ModelExecute 且有 infershape / aclnn 调用
        json_file = self.locator.get("msprof_json")
        if json_file and not result["has_dynamic_shape"]:
            infershape_events = self._extract_json_events(
                json_file, ["infershape", "InferShape", "aclnn"]
            )
            if infershape_events:
                result["has_dynamic_shape"] = True
                result["evidence"].append(
                    f"msprof.json: 发现 {len(infershape_events)} 个 infershape/aclnn 事件"
                )

        return result

    def analyze_op_metrics(self) -> Dict[str, Any]:
        """分析 op_summary / kernel_details 中的各种 ratio 指标，识别问题算子"""
        result = {
            "top_ops_by_duration": [], "scalar_ratio_ops": [],
            "low_cube_utilization_ops": [], "low_mac_ratio_ops": [],
            "aicpu_ops": [], "op_type_distribution": {}, "problem_ops_agg": {},
        }
        op_file = self.locator.get("op_summary")
        if not op_file:
            return result
        header, rows = read_csv_safe(op_file)
        if not header or not rows:
            return result
        col_map = self._build_col_map(header)
        dur_col = col_map["Task Duration(us)"]
        result["top_ops_by_duration"] = self._build_top_ops(rows, col_map, dur_col)
        op_type_counter = defaultdict(lambda: {"count": 0, "total_us": 0.0})
        for row in rows:
            op_name = row.get(col_map.get("Op Name", "Op Name"), "")
            op_type = row.get(col_map.get("OP Type", "OP Type"), "")
            task_type = str(row.get(col_map.get("Task Type", "Task Type"), "")).strip()
            task_dur = safe_float(row.get(dur_col, 0))
            op_type_counter[op_type]["count"] += 1
            op_type_counter[op_type]["total_us"] += task_dur
            is_aicore = "AI_CORE" in task_type or "MIX_AIC" in task_type
            is_aivector = "AI_VECTOR" in task_type or "AIV" in task_type or "MIX_AIV" in task_type
            is_aicpu = "AI_CPU" in task_type
            info = {"op_name": op_name, "op_type": op_type, "task_type": task_type,
                    "task_dur": task_dur, "is_aicore": is_aicore,
                    "is_aivector": is_aivector, "is_aicpu": is_aicpu}
            result["col_map"] = col_map
            self._check_ratio_problems(row, info, result)
        self._build_op_type_distribution(op_type_counter, result)
        return result

    def analyze_fusion(self) -> Dict[str, Any]:
        """分析融合信息

        检测来源（按优先级）：
        1. op_statistic_*.csv 中 OP Type 含 autofuse_/autofused_/triton_poi/triton_per/dvm_ 前缀
        2. fusion_op_*.csv 中的融合算子明细
        """
        result = {
            "fusion_ops": [], "total_fusion_count": 0,
            "autofuse_enabled": False, "triton_fusion_enabled": False,
            "dvm_fusion_enabled": False, "fusion_type": "", "fusion_op_types": [],
        }
        self._detect_fusion_from_op_statistic(result)
        self._detect_fusion_from_fusion_op(result)
        types = []
        if result["autofuse_enabled"]:
            types.append("AutoFuse")
        if result["triton_fusion_enabled"]:
            types.append("Triton融合")
        if result["dvm_fusion_enabled"]:
            types.append("DVM融合")
        result["fusion_type"] = " + ".join(types) if types else ""
        return result

    def detect_execution_mode(self) -> Dict[str, Any]:
        """检测执行模式：图模式 / 非图模式(Eager) / 无法判断

        判断逻辑（按优先级）：
        1. op_statistic 含 triton_poi/triton_per 前缀 → 图模式 (Inductor + Triton)
        2. op_statistic 含 dvm_ 前缀 → 图模式 (Inductor + DVM)
        3. op_statistic 含 autofuse_/autofused_ 前缀 + 有 ModelExecute → 图模式 (GE/TorchAir/ATC AutoFuse)
        4. op_statistic 含 autofuse_/autofused_ 前缀 + 无 ModelExecute → 图模式 (Inductor + AscendC)
        5. api_statistic 含 ModelExecute (无融合算子) → 图模式 (GE/TorchAir)
        6. 提供了 dump 图文件 (pbtxt / runnable.py) → 图模式
        7. 以上均不满足 → 无法判断

        注意：
        - 有自动融合算子 → 一定是图模式
        - 走GE/TorchAir (有ModelExecute) → 就是图模式
        - 有 dump 图文件 → 基本是图模式
        - 其他情况不瞎猜，直接报"无法判断"
        - 图模式下仍可能有部分 aclnn 算子（torch.compile 未完全覆盖），不影响图模式判定
        -autofuse_/autofused_ 前缀: GE / TorchAir / Torch Inductor+AscendC / ATC 均可能产生
        - ModelExecute: GE(TF后端/原生接口) 和 TorchAir(torch.compile后端) 都会有
        """
        result = {"mode": "unknown", "mode_label": "无法判断", "evidence": []}
        fusion = self.analyze_fusion()
        has_autofused = fusion.get("autofuse_enabled", False)
        has_triton = fusion.get("triton_fusion_enabled", False)
        has_dvm = fusion.get("dvm_fusion_enabled", False)
        api_rows = self._get_api_rows()
        has_model_execute = self._check_model_execute_in_api_rows(api_rows)
        flags = {
            "has_triton": has_triton, "has_dvm": has_dvm,
            "has_autofused": has_autofused, "has_model_execute": has_model_execute,
        }
        if self._detect_from_autofuse(fusion, flags, result):
            return result
        if self._detect_from_model_execute(has_model_execute, result):
            return result
        if self._detect_from_dump_graph(result):
            return result
        result["evidence"].append("无自动融合算子、无 ModelExecute、无 dump 图文件，无法确定执行模式")
        return result

    def generate_report(self) -> Dict[str, Any]:
        """生成完整的性能分析报告"""
        self.report = self._run_analysis()
        return self.report

    def format_text_report(self) -> str:
        """生成 Markdown 格式的分析报告"""
        r = self.report
        lines = []
        lines.extend(self._format_summary_section(r))
        lines.extend(self._format_fusion_section(r))
        lines.extend(self._format_op_distribution(r))
        lines.extend(self._format_problem_ops(r))
        lines.extend(self._format_host_overhead(r))
        return "\n".join(lines)


    def _get_api_rows(self) -> List[Dict[str, str]]:
        """缓存 api_statistic 行数据，避免多次 read_csv_safe"""
        if self._api_rows_cache is not None:
            return self._api_rows_cache
        api_file = self.locator.get("api_statistic")
        if api_file:
            _, rows = read_csv_safe(api_file)
            self._api_rows_cache = rows
        else:
            self._api_rows_cache = []
        return self._api_rows_cache

    def _infer_iter_from_op_summary(self) -> Optional[Dict[str, Any]]:
        """从 op_summary/kernel_details 推断迭代数（无 step_trace/api_statistic 时）

        策略（按优先级）：
        1. op_summary 含 Infer ID 列 → 唯一 Infer ID 数 = 迭代数
        2. kernel_details 含 Step Id 列 → 唯一 Step Id 数 = 迭代数
        3. GE Profiling op_summary: 非aclnn算子的 "Op Name" 在单次迭代中唯一存在，
           同一个 Op Name 出现的次数 = 迭代数（取众数）。

        注意：kernel_details "Name" 和 op_summary "OP Type" 不能用于推断迭代数，
        因为这些实体在单次迭代内出现多次（如 MatMulV2 每迭代 101 个），
        众数 = per_iter_count × iter_count ≠ 迭代数。

        无法推断时返回 None，调用方默认设为 1 次迭代并用算子耗时估算。
        """
        op_file = self.locator.get("op_summary")
        if not op_file:
            return None
        header, rows = read_csv_safe(op_file)
        if not header or not rows:
            return None
        for strategy in (self._infer_from_infer_id, self._infer_from_step_id, self._infer_from_op_name_mode):
            result = strategy(rows, header)
            if result:
                return result
        return None

    def _infer_iter_from_step_trace(self) -> int:
        """从 step_trace 获取迭代次数"""
        step_file = self.locator.get("step_trace")
        if not step_file:
            return 0
        header, rows = read_csv_safe(step_file)
        count = 0
        is_step_trace_time = "Stage" in header and "Computing" in header
        for row in rows:
            if is_step_trace_time:
                if safe_float(row.get("Stage", 0)) > 0:
                    count += 1
                continue
            for col in ["Iteration Time(us)", "Iteration Time", "iteration_time"]:
                if col in row and str(row[col]).strip() not in ("", "N/A"):
                    count += 1
                    break
        return count

    def _get_iter_count(self) -> int:
        """从 step_trace 或 api_statistic 获取迭代次数"""
        count = self._infer_iter_from_step_trace()
        if count > 0:
            return count
        api_rows = self._get_api_rows()
        count = self._infer_iter_from_api_sync(api_rows)
        if count > 0:
            return count
        inferred = self._infer_iter_from_op_summary()
        if inferred:
            return inferred["count"]
        return 0

    def _get_ratio(self, row: Dict[str, str], field_candidates: List[str]) -> float:
        """从行中尝试多个字段名，返回第一个有效的 ratio 值
        自动处理 0~1 小数 和 0~100 百分比 两种值域

        健壮性保障：
        - 百分比字段值<1时 ×100（如 0.5 → 50）
        - 非百分比字段值>1时 /100（如 50 → 0.5），但仅当值在 1~100 之间
        - 值>100 的非百分比字段视为异常，返回 0
        """
        for field in field_candidates:
            val = row.get(field)
            if val is None or str(val).strip() in ("", "N/A", "n/a"):
                continue
            fval = safe_float(val)
            if fval == 0.0:
                continue
            if field in self.RATIO_IS_PERCENT:
                if 0 < fval < 1.0:
                    fval *= 100
                elif fval > 100:
                    fval = 0.0
            else:
                if 1.0 < fval <= 100.0:
                    fval /= 100
                elif fval > 100:
                    fval = 0.0
            return fval
        return 0.0

    def _analyze_host_overhead(self, iter_count: int) -> List[Dict[str, Any]]:
        """分析api_statistic中Host侧开销Top API（per-iter耗时排序）

        排除编译/加载等一次性开销（BuildGraph, ModelLoad等）。
        """
        compile_apis = {"BuildGraph", "ModelLoad", "CompileGraph", "Init"}
        if iter_count <= 0:
            iter_count = self._get_iter_count() or 1
        result = []
        api_rows = self._get_api_rows()
        if not api_rows:
            return result
        for row in api_rows:
            api_name = row.get("API Name", "").strip()
            if api_name in compile_apis:
                continue
            total_us = safe_float(row.get("Time(us)", 0))
            if total_us < 1000:
                continue
            count = safe_int(row.get("Count", 0))
            result.append({
                "api_name": api_name,
                "level": row.get("Level", "").strip(),
                "total_us": total_us,
                "count": count,
                "per_iter_us": total_us / iter_count,
                "calls_per_iter": count // iter_count,
            })
        result.sort(key=lambda x: x["per_iter_us"], reverse=True)
        return result[:15]

    def _extract_h2d_from_json(self, json_file: str) -> Dict[str, Any]:
        """从 msprof.json 提取 H2D 事件并返回结果 dict"""
        result = {"h2d_total_us": 0.0, "h2d_avg_us": 0.0, "h2d_count": 0, "source": ""}
        h2d_events = self._extract_json_events(json_file, ["InputCopy", "H2D"])
        if h2d_events:
            total = sum(e["dur"] for e in h2d_events)
            result["h2d_total_us"] = total * 1000
            result["h2d_avg_us"] = (total / len(h2d_events) * 1000) if h2d_events else 0
            result["h2d_count"] = len(h2d_events)
            result["source"] = "msprof.json: InputCopy"
        return result

    def _calc_me_from_msprof_json(self, result: Dict[str, Any]):
        """从 msprof.json ModelExecute 事件填充 result"""
        json_file = self.locator.get("msprof_json")
        if not json_file:
            return
        me_events = self._extract_json_events(json_file, ["ModelExecute"])
        if me_events:
            durs = [e["dur"] * 1000 for e in me_events]
            result["iteration_avg_us"] = sum(durs) / len(durs)
            result["iteration_count"] = len(durs)
            result["source"] = "msprof.json: ModelExecute"

    def _calc_me_fallback(self, result: Dict[str, Any], api_rows: List[Dict[str, str]]):
        """当 step_trace/api 均无结果时的 fallback 推断"""
        if result["iteration_count"] != 0:
            return
        sync_count = self._find_best_sync_count(api_rows) if api_rows else 0
        if sync_count > 1:
            result["iteration_count"] = sync_count
            result["source"] = f"api_statistic: {sync_count} sync calls"
        else:
            inferred = self._infer_iter_from_op_summary()
            if inferred:
                result["iteration_count"] = inferred["count"]
                result["source"] = inferred["source"]
        if result["iteration_count"] > 0 and result.get("eager_summary") and \
                result["eager_summary"].get("wall_us", 0) > 0:
            result["iteration_avg_us"] = result["eager_summary"]["wall_us"] / result["iteration_count"]

    def _build_top_ops(self, rows, col_map, dur_col):
        sorted_rows = sorted(rows, key=lambda r: safe_float(r.get(dur_col, 0)), reverse=True)
        seen_keys = set()
        top_ops = []
        for row in sorted_rows:
            op_type = row.get(col_map.get("OP Type", "OP Type"), "")
            shapes = row.get(col_map.get("Input Shapes", "Input Shapes"), "")
            dtypes = row.get(col_map.get("Input Data Types", "Input Data Types"), "")
            dedup_key = (op_type, shapes, dtypes)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            top_ops.append({
                "op_name": row.get(col_map.get("Op Name", "Op Name"), ""),
                "op_type": op_type,
                "task_type": row.get(col_map.get("Task Type", "Task Type"), ""),
                "task_duration_us": safe_float(row.get(dur_col, 0)),
                "op_state": row.get(col_map.get("OP State", "OP State"), ""),
                "input_shapes": shapes, "input_dtypes": dtypes,
                "aic_scalar_ratio": self._get_ratio(row, self.AIC_SCALAR_FIELDS),
                "aic_mac_ratio": self._get_ratio(row, self.MAC_FIELDS),
                "aic_mte2_ratio": self._get_ratio(row, self.MTE2_AIC_FIELDS),
                "cube_utilization": self._get_ratio(row, self.CUBE_UTIL_FIELDS),
                "aiv_scalar_ratio": self._get_ratio(row, self.AIV_SCALAR_FIELDS),
                "aiv_vec_ratio": self._get_ratio(row, self.VEC_FIELDS),
            })
        return top_ops

    def _check_scalar_ratio(self, row, is_aicore, is_aivector, ctx, result):
        """检查 scalar_ratio 问题（AICore 或 AIVector）"""
        fields = self.AIC_SCALAR_FIELDS if is_aicore else self.AIV_SCALAR_FIELDS
        scalar_ratio = self._get_ratio(row, fields)
        if scalar_ratio >= self.SCALAR_THRESHOLD:
            ctx["ptype"] = "scalar_ratio>=30%"
            ctx["pvalue"] = f"{scalar_ratio:.1%}"
            self._agg_problem(result["problem_ops_agg"], ctx, result.get("col_map"))
            result["scalar_ratio_ops"].append({
                "op_name": ctx["op_name"], "op_type": ctx["op_type"],
                "task_type": ctx["task_type"], "scalar_ratio": scalar_ratio,
                "task_duration_us": ctx["task_dur"],
            })

    def _check_ratio_problems(self, row, info, result):
        """检查算子 ratio 问题，info 含 op_name/op_type/task_type/task_dur/is_aicore/is_aivector/is_aicpu"""
        op_name = info["op_name"]
        op_type = info["op_type"]
        task_dur = info["task_dur"]
        is_aicore = info["is_aicore"]
        is_aivector = info["is_aivector"]
        is_aicpu = info["is_aicpu"]
        ctx = {"row": row, "op_name": op_name, "op_type": op_type,
               "task_type": info["task_type"], "task_dur": task_dur}
        if is_aicpu:
            ctx["ptype"] = "AICPU"
            ctx["pvalue"] = "-"
            self._agg_problem(result["problem_ops_agg"], ctx, result.get("col_map"))
            result["aicpu_ops"].append({
                "op_name": op_name, "op_type": op_type,
                "task_duration_us": task_dur,
            })
        if is_aicore or is_aivector:
            self._check_scalar_ratio(row, is_aicore, is_aivector, ctx, result)
        if is_aicore:
            cube_util = self._get_ratio(row, self.CUBE_UTIL_FIELDS)
            if 0 < cube_util < self.CUBE_UTIL_THRESHOLD:
                ctx["ptype"] = "cube_util<20%"
                ctx["pvalue"] = f"{cube_util:.1f}%"
                self._agg_problem(result["problem_ops_agg"], ctx, result.get("col_map"))
                result["low_cube_utilization_ops"].append({
                    "op_name": op_name, "op_type": op_type,
                    "cube_utilization": cube_util,
                    "task_duration_us": task_dur,
                })
        if is_aicore:
            mac_ratio = self._get_ratio(row, self.MAC_FIELDS)
            if 0 < mac_ratio < self.MAC_RATIO_THRESHOLD:
                ctx["ptype"] = "mac_ratio<30%"
                ctx["pvalue"] = f"{mac_ratio:.1%}"
                self._agg_problem(result["problem_ops_agg"], ctx, result.get("col_map"))
                result["low_mac_ratio_ops"].append({
                    "op_name": op_name, "op_type": op_type,
                    "mac_ratio": mac_ratio, "task_duration_us": task_dur,
                })

    def _detect_fusion_from_op_statistic(self, result: Dict[str, Any]):
        """从 op_statistic_*.csv 的 OP Type 列检测融合"""
        _ge_fusion_op_types = {"FusedAscBackend", "AscBackend", "FusedAscCpu"}
        stat_file = self.locator.get("op_statistic")
        if not stat_file:
            return
        _, rows = read_csv_safe(stat_file)
        fusion_op_types = set()
        for row in rows:
            op_type = str(row.get("OP Type", "")).strip()
            for prefix in self.FUSION_PREFIXES:
                if op_type.startswith(prefix):
                    fusion_op_types.add(op_type)
                    self._classify_fusion_prefix(prefix, result)
                    break
            if op_type in _ge_fusion_op_types:
                fusion_op_types.add(op_type)
                result["autofuse_enabled"] = True
        if fusion_op_types:
            result["fusion_op_types"] = sorted(fusion_op_types)
            result["total_fusion_count"] = len(fusion_op_types)

    def _detect_fusion_from_fusion_op(self, result: Dict[str, Any]):
        """从 fusion_op_*.csv 提取融合算子明细"""
        fusion_file = self.locator.get("fusion_op")
        if not fusion_file:
            return
        _, rows = read_csv_safe(fusion_file)
        for row in rows[:50]:
            result["fusion_ops"].append({
                "fusion_op": row.get("Fusion Op", ""),
                "original_ops": row.get("Original Ops", ""),
                "memory_total_kb": safe_float(row.get("Memory Total(KB)", 0)),
            })
        if result["total_fusion_count"] == 0:
            result["total_fusion_count"] = len(rows)

    def _detect_from_dump_graph(self, result: Dict[str, Any]) -> bool:
        """从 dump 图文件推断执行模式，返回 True 表示已判定"""
        if self.graph_file:
            result["mode"] = "graph"
            result["mode_label"] = "图模式 (有Dump图文件)"
            result["evidence"].append(f"提供了 dump 图文件: {os.path.basename(self.graph_file)}")
            return True
        return False

    def _run_analysis(self) -> Dict[str, Any]:
        """运行所有分析并返回结果字典"""
        analysis = {}
        analysis["file_discovery"] = {dt: path for dt, path in self.locator.files.items() if path}
        h2d = self.calc_h2d()
        d2h = self.calc_d2h()
        me = self.calc_model_execute()
        iter_count = me.get("iteration_count", 0)
        iter_avg = me.get("iteration_avg_us", 0.0)
        op_total = self.calc_op_total_time(iter_count if iter_count > 0 else 1)
        op_total_avg = op_total.get("op_total_avg_us", 0.0)
        if iter_avg == 0.0 and op_total_avg > 0:
            if iter_count > 0:
                iter_avg = op_total_avg
                me["iteration_avg_us"] = iter_avg
                me["source"] = (me.get("source", "") + " + op_total估算").strip(" +")
            elif iter_count == 0:
                iter_count = 1
                iter_avg = op_total_avg
                me["iteration_avg_us"] = iter_avg
                me["iteration_count"] = 1
                me["source"] = (me.get("source", "") or "op_total估算(单次)")
        dynamic = self.check_dynamic_shape()
        op_metrics = self.analyze_op_metrics()
        fusion = self.analyze_fusion()
        exec_mode = self.detect_execution_mode()
        host_overhead = self._analyze_host_overhead(iter_count)
        total_iter_us = h2d["h2d_avg_us"] + iter_avg + d2h["d2h_avg_us"]
        op_ratio = (op_total_avg / iter_avg * 100) if iter_avg > 0 else 0.0
        h2d_ratio = (h2d["h2d_avg_us"] / total_iter_us * 100) if total_iter_us > 0 else 0.0
        d2h_ratio = (d2h["d2h_avg_us"] / total_iter_us * 100) if total_iter_us > 0 else 0.0
        me_ratio = (iter_avg / total_iter_us * 100) if total_iter_us > 0 else 0.0
        summary = {
            "h2d_avg_us": h2d["h2d_avg_us"], "model_execute_avg_us": iter_avg,
            "d2h_avg_us": d2h["d2h_avg_us"], "total_iter_avg_us": total_iter_us,
            "op_total_avg_us": op_total_avg, "h2d_ratio_pct": round(h2d_ratio, 2),
            "model_execute_ratio_pct": round(me_ratio, 2), "d2h_ratio_pct": round(d2h_ratio, 2),
            "op_total_ratio_pct": round(op_ratio, 2), "has_dynamic_shape": dynamic["has_dynamic_shape"],
            "iteration_count": iter_count, "eager_summary": me.get("eager_summary"),
            "compile_iter_us": me.get("compile_iter_us", 0.0),
        }
        analysis.update({
            "h2d": h2d, "d2h": d2h, "model_execute": me, "op_total_time": op_total,
            "dynamic_shape": dynamic, "op_metrics": op_metrics, "fusion": fusion,
            "execution_mode": exec_mode, "host_overhead": host_overhead, "summary": summary,
        })
        return analysis

    def _format_summary_section(self, r: Dict[str, Any]) -> List[str]:
        """生成标题 + 文件发现 + 核心指标汇总"""
        lines = []
        s = r.get("summary", {})
        lines.append("# 推荐模型 NPU Profiling 性能分析报告\n")
        lines.append("## 1. Profiling 文件发现\n")
        lines.append("| 数据类型 | 文件名 |")
        lines.append("|----------|--------|")
        for dt, path in r.get("file_discovery", {}).items():
            lines.append(f"| {dt} | `{os.path.basename(path)}` |")
        lines.append("")
        lines.append("## 2. 核心指标汇总\n")
        h2d_note = (
            "> **H2D/D2H说明**: 客户通过 `aclrtMemcpy`/`aclrtMemcpyBatch`"
            " 自行实现H2D/D2H，无标准InputCopy/OutputCopy。"
        )
        if "Memcpy" in r.get("h2d", {}).get("source", ""):
            lines.append(h2d_note)
        lines.append("")
        lines.append("| 指标 | 耗时(us/iter) | 占比 | 数据来源 |")
        lines.append("|------|:-------------:|:----:|----------|")
        lines.extend(self._format_metrics_table(r, s))
        compile_us = s.get("compile_iter_us", 0.0)
        if compile_us > 0:
            lines.append(f"| 首次编译迭代耗时 | {compile_us:.2f} | - | 已排除，不计入平均值 |")
        lines.extend(self._format_eager_summary(s))
        lines.append("")
        return lines


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="推荐模型昇腾NPU Profiling性能分析脚本"
    )
    parser.add_argument(
        "profiling_dir",
        help="MindStudio Profiler 输出目录路径 (mindstudio_profiler_output 或 PROF_XXX 目录)"
    )
    parser.add_argument(
        "--graph-file", "-g",
        default=None,
        help="Dump 图文件路径 (pbtxt 或 fxgraph runnable.py)，用于辅助判断执行模式"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Markdown 报告输出路径 (默认仅输出到终端)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.profiling_dir):
        logging.error(f"错误: 目录不存在: {args.profiling_dir}")
        return 1

    logging.info(f"分析目录: {args.profiling_dir}\n")

    pp = ProfilingParser(args.profiling_dir, graph_file=args.graph_file)
    logging.info(pp.locator.summary())
    logging.info("")

    try:
        report = pp.generate_report()
        md_report = pp.format_text_report()
    except Exception as e:
        logging.error(f"错误: 分析过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logging.info(md_report)

    if args.output:
        _save_report_to_file(args.output, md_report)

    return 0


def _save_report_to_file(output_path: str, content: str):
    """保存报告内容到指定文件路径"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"\nMarkdown 报告已保存至: {output_path}")


if __name__ == "__main__":
    exit(main())
