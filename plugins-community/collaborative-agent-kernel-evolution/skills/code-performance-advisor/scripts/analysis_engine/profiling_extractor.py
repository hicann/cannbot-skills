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
动态profiling指标提取器

设计原则:
1. 不hardcode字段名
2. 根据Task Type动态过滤有效字段
3. 保留完整原始数据供LLM解读
4. 参考: references/standards/op_summary_header_guide.md
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProfilingData:
    """
    Profiling数据契约 - 灵活版

    设计理念:
    - raw_metrics: 所有CSV列,不假设字段集合
    - task_type: 用于指导相关字段筛选
    - relevant_metrics: 根据task_type过滤
    """
    task_type: str  # AI_VECTOR_CORE / AI_CORE / MIX_AIC / UNKNOWN
    raw_metrics: Dict[str, Any]  # 所有非空字段
    relevant_metrics: Dict[str, Any]  # 根据task_type过滤的相关字段

    # 基础字段(所有类型通用)
    task_duration_us: float
    task_wait_time_us: float = 0.0
    block_dim: int = 1

    # 元数据
    op_name: str = ""
    op_type: str = ""
    csv_source: str = ""

    def get_metric(self, name: str, default=None) -> Optional[float]:
        """安全查询指标,不假设字段存在"""
        return self.relevant_metrics.get(name, default)

    def get_utilization_ratio(self) -> Optional[float]:
        """
        根据Task Type自动选择正确的利用率指标

        Vector: aiv_vec_ratio
        Cube: aic_mac_ratio
        MIX: 取max(aiv_vec_ratio, aic_mac_ratio)
        """
        if "VECTOR" in self.task_type:
            return self.get_metric("aiv_vec_ratio")
        elif "CORE" in self.task_type and "VECTOR" not in self.task_type:
            return self.get_metric("aic_mac_ratio")
        elif "MIX" in self.task_type:
            aiv = self.get_metric("aiv_vec_ratio", 0)
            aic = self.get_metric("aic_mac_ratio", 0)
            return max(aiv, aic) if (aiv or aic) else None
        return None

    def classify_bottleneck(self) -> str:
        """
        Classify the dominant bottleneck type using a dominant-ratio approach.

        Strategy: build a {label: (ratio, evidence_str)} map for the relevant
        core type, then return whichever label has the highest ratio.  No fixed
        thresholds — relative dominance is what matters, because the same
        absolute ratio (e.g. vec=0.55) may be normal for one operator type and
        a clear bottleneck for another.

        The only hard guard is ratio < 0.10 across all candidates, which usually
        means the CSV row is from a non-kernel phase (host setup, etc.) rather
        than an actual AI-Core execution window.

        Returns: "<Class>: <evidence>" or "Unknown: <reason>"

        Examples:
          "Memory-bound: aiv_mte2_ratio=0.38+aiv_mte3_ratio=0.22=0.60 为最大比率"
          "Compute-bound: aiv_vec_ratio=0.71 为最大比率"
          "Instruction-bound: aiv_scalar_ratio=0.45 为最大比率"
          "Unknown: 所有比率均低于 0.10，可能为非 AI Core 执行行"
        """
        def _g(key: str) -> float:
            v = self.get_metric(key)
            if v is None or pd.isna(v):
                return 0.0
            return float(v)

        candidates: Dict[str, tuple] = {}   # label -> (ratio, evidence_str)

        if "VECTOR" in self.task_type and "MIX" not in self.task_type:
            vec = _g("aiv_vec_ratio")
            mte2 = _g("aiv_mte2_ratio")
            mte3 = _g("aiv_mte3_ratio")
            scal = _g("aiv_scalar_ratio")
            candidates = {
                "Compute-bound": (vec, f"aiv_vec_ratio={vec:.2f}"),
                "Memory-bound": (mte2 + mte3, f"aiv_mte2_ratio={mte2:.2f}+aiv_mte3_ratio={mte3:.2f}={mte2+mte3:.2f}"),
                "Instruction-bound": (scal, f"aiv_scalar_ratio={scal:.2f}"),
            }

        elif "CORE" in self.task_type and "VECTOR" not in self.task_type and "MIX" not in self.task_type:
            mac = _g("aic_mac_ratio")
            mte2 = _g("aic_mte2_ratio")
            scal = _g("aic_scalar_ratio")
            candidates = {
                "Compute-bound": (mac, f"aic_mac_ratio={mac:.2f}"),
                "Memory-bound": (mte2, f"aic_mte2_ratio={mte2:.2f}"),
                "Instruction-bound": (scal, f"aic_scalar_ratio={scal:.2f}"),
            }

        elif "MIX" in self.task_type:
            vec = _g("aiv_vec_ratio")
            mac = _g("aic_mac_ratio")
            mte2a = _g("aiv_mte2_ratio")
            mte3a = _g("aiv_mte3_ratio")
            mte2c = _g("aic_mte2_ratio")
            scal_v = _g("aiv_scalar_ratio")
            scal_c = _g("aic_scalar_ratio")
            compute = max(vec, mac)
            memory = max(mte2a + mte3a, mte2c)
            scalar = max(scal_v, scal_c)
            compute_ev = f"aiv_vec_ratio={vec:.2f}" if vec >= mac else f"aic_mac_ratio={mac:.2f}"
            memory_ev = (f"aiv_mte2_ratio={mte2a:.2f}+aiv_mte3_ratio={mte3a:.2f}={mte2a+mte3a:.2f}"
                         if mte2a + mte3a >= mte2c else f"aic_mte2_ratio={mte2c:.2f}")
            scalar_ev = f"aiv_scalar_ratio={scal_v:.2f}" if scal_v >= scal_c else f"aic_scalar_ratio={scal_c:.2f}"
            candidates = {
                "Compute-bound": (compute, compute_ev),
                "Memory-bound": (memory, memory_ev),
                "Instruction-bound": (scalar, scalar_ev),
            }

        if not candidates:
            return f"Unknown: task_type='{self.task_type}' 无法识别"

        dominant = max(candidates, key=lambda k: candidates.get(k, (0.0, ""))[0])
        dom_ratio, dom_ev = candidates.get(dominant, (0.0, ""))

        if dom_ratio < 0.10:
            return f"Unknown: 所有比率均低于 0.10（最高={dom_ratio:.2f}），可能为非 AI Core 执行行"

        return f"{dominant}: {dom_ev} 为最大比率"

    def get_bottleneck_hint(self) -> str:
        """
        基于Task Type和指标,给出瓶颈提示(供LLM参考)

        注意: 这不是确定性诊断,只是提示
        """
        hints = []

        def _safe(v) -> float:
            """Return 0.0 for None or NaN, else float(v)."""
            if v is None or pd.isna(v):
                return 0.0
            return float(v)

        if "VECTOR" in self.task_type:
            vec_ratio = _safe(self.get_metric("aiv_vec_ratio"))
            scalar_ratio = _safe(self.get_metric("aiv_scalar_ratio"))
            mte_ratio = _safe(self.get_metric("aiv_mte2_ratio")) + _safe(self.get_metric("aiv_mte3_ratio"))

            if vec_ratio > 0.6:
                hints.append("Compute-bound (aiv_vec_ratio>60%)")
            if scalar_ratio > 0.3:
                hints.append("Scalar/Control-heavy (aiv_scalar_ratio>30%)")
            if mte_ratio > 0.5:
                hints.append("Memory-bound (aiv_mte2+mte3>50%)")

        elif "CORE" in self.task_type:
            mac_ratio = _safe(self.get_metric("aic_mac_ratio"))
            scalar_ratio = _safe(self.get_metric("aic_scalar_ratio"))
            mte_ratio = _safe(self.get_metric("aic_mte2_ratio"))

            if mac_ratio > 0.6:
                hints.append("Compute-bound (aic_mac_ratio>60%)")
            if scalar_ratio > 0.3:
                hints.append("Scalar/Control-heavy (aic_scalar_ratio>30%)")
            if mte_ratio > 0.4:
                hints.append("Memory-bound (aic_mte2_ratio>40%)")

        return " | ".join(hints) if hints else "No obvious bottleneck pattern"

    def to_dict(self) -> Dict:
        """转为字典(用于JSON序列化)"""
        # 递归转换numpy/pandas类型为Python原生类型
        def convert_value(v):
            # 处理数值类型
            if hasattr(v, 'item'):  # numpy类型
                return v.item()
            elif isinstance(v, (int, float, str, bool, type(None))):
                return v
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(val) for val in v]
            else:
                # 尝试转换为基本类型
                try:
                    return float(v) if '.' in str(v) else int(v)
                except (ValueError, TypeError):
                    return str(v)

        return {
            "task_type": self.task_type,
            "task_duration_us": convert_value(self.task_duration_us),
            "task_wait_time_us": convert_value(self.task_wait_time_us),
            "block_dim": convert_value(self.block_dim),
            "op_name": self.op_name,
            "op_type": self.op_type,
            "raw_metrics": convert_value(self.raw_metrics),
            "relevant_metrics": convert_value(self.relevant_metrics),
            "bottleneck_class": self.classify_bottleneck(),
            "bottleneck_hint": self.get_bottleneck_hint(),
            "utilization_ratio": convert_value(self.get_utilization_ratio()) if self.get_utilization_ratio() is not None else None
        }


class ProfilingExtractor:
    """
    Profiling数据提取器 - 智能识别Task Type

    参考文档: references/standards/op_summary_header_guide.md
    """

    def __init__(self, csv_path: Path, target_op_name: Optional[str] = None):
        self.csv_path = csv_path
        self.target_op_name = target_op_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    def extract(self) -> ProfilingData:
        """提取profiling数据(动态适配Task Type)"""
        df = pd.read_csv(self.csv_path)

        if df.empty:
            raise ValueError(f"Empty CSV: {self.csv_path}")

        # 如果指定了target_op_name，智能匹配对应行
        if self.target_op_name:
            row = self._find_matching_op(df, self.target_op_name)
            if row is None:
                available = df["Op Name"].unique().tolist()
                # 检测 CSV 是否来自 framework 而非自定义算子（常见误操作：Model_* vs ModelNew_*）
                framework_ops = [op for op in available
                                 if op.lower().startswith("aclnn")
                                 or ("batchnorm" in op.lower() and "custom" not in op.lower())
                                 or op.startswith("BatchNormV")]
                if framework_ops:
                    hint = (
                        f"\n💡 该 CSV 包含 framework 算子（如 '{framework_ops[0]}'），"
                        f"而非自定义算子 profiling。\n"
                        f"   请使用 'ModelNew_*' 目录下的 op_summary CSV（自定义算子的 profiling），\n"
                        f"   而非 'Model_*' 目录（PyTorch 内置算子的数据）。\n"
                        f"   init_workspace.py 会自动按时间戳选择最新 CSV；"
                        f"请确认 CAKE2/output/{self.target_op_name}/profiling/ 下存在 ModelNew_* 目录。"
                    )
                else:
                    hint = (
                        f"\n💡 尝试：精确名称、大小写不敏感匹配，或加/去 'Custom' 后缀"
                    )
                raise ValueError(
                    f"Op '{self.target_op_name}' not found in CSV.\n"
                    f"Available ops: {available}"
                    f"{hint}"
                )
        else:
            # 未指定算子名，取第一行
            row = df.iloc[0]

        # 1. 提取所有非空字段(不假设字段名)
        raw_metrics = {k: v for k, v in row.items() if pd.notna(v)}

        # 2. 识别Task Type
        task_type = row.get("Task Type", "UNKNOWN")

        # 3. 根据Task Type过滤相关字段
        relevant_metrics = self._filter_relevant_fields(raw_metrics, task_type)

        # 4. 提取基础字段(所有类型通用)
        task_duration = float(row.get("Task Duration(us)", 0))
        task_wait = float(row.get("Task Wait Time(us)", 0))
        block_dim = int(row.get("Block Dim", 1))

        return ProfilingData(
            task_type=task_type,
            raw_metrics=raw_metrics,
            relevant_metrics=relevant_metrics,
            task_duration_us=task_duration,
            task_wait_time_us=task_wait,
            block_dim=block_dim,
            op_name=str(row.get("Op Name", "")),
            op_type=str(row.get("OP Type", "")),
            csv_source=str(self.csv_path)
        )

    @staticmethod
    def _filter_relevant_fields(
        raw: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """
        根据Task Type过滤有效字段

        规则(参考op_summary_header_guide.md):
        - AI_VECTOR_CORE: aiv_* 字段有效
        - AI_CORE (Cube): aic_* 字段有效
        - MIX: 两类字段都有效
        - UNKNOWN: 保留所有 *_ratio 和 *_time 字段
        """
        if "VECTOR" in task_type and "MIX" not in task_type:
            # 纯Vector算子
            return {k: v for k, v in raw.items() if k.startswith("aiv_")}

        elif "CORE" in task_type and "VECTOR" not in task_type:
            # 纯Cube算子
            return {k: v for k, v in raw.items() if k.startswith("aic_")}

        elif "MIX" in task_type:
            # 混合算子,保留两类
            return {
                k: v for k, v in raw.items()
                if k.startswith(("aiv_", "aic_"))
            }

        else:
            # 未知类型,保守策略:保留所有ratio/time字段
            filtered: Dict[str, Any] = {}
            for k, v in raw.items():
                if any(suffix in k for suffix in ["_ratio", "_time", "_rate"]):
                    filtered[k] = v
            return filtered

    @staticmethod
    def _best_row(rows: pd.DataFrame) -> pd.Series:
        """
        从多个匹配行中选取最具代表性的行

        策略：跳过第1行（冷启动/预热效应），对剩余行取 Task Duration(us) 的 **中位数**。
        中位数比 min 更稳定（对偶发抖动不敏感），比 mean 对异常值更鲁棒。

        两种 CSV 模式均适用：
        - Advanced 模式（dirty CSV）：每轮 MatMul+ReduceMax+目标算子，目标算子行已被
          _find_matching_op 过滤；第1行 NPU 频率仍在爬升，后续稳定。
        - Basic 模式（clean CSV）：profiling 前已有 10 次 warmup，所有 20 行基本稳定，
          跳过第1行略有保守但无害。
        - 若只有1行：直接返回（无法跳过）。
        """
        if len(rows) == 1:
            return rows.iloc[0]

        # 跳过第1行（冷启动），对剩余行取中位数最近行
        stable = rows.iloc[1:].copy()
        durations = stable["Task Duration(us)"].astype(float)
        median_val = durations.median()
        # 取与中位数最近的实际行（避免返回插值）
        closest_idx = (durations - median_val).abs().idxmin()
        return stable.loc[closest_idx]

    def _find_matching_op(self, df: pd.DataFrame, target_name: str) -> Optional[pd.Series]:
        """
        智能匹配算子名,支持大小写和Custom后缀变体

        匹配策略(按优先级):
        1. 精确匹配: target_name == op_name
        2. 大小写不敏感: target_name.lower() == op_name.lower()
        3. Custom后缀变体匹配
        4. snake_case → PascalCase 转换（含Custom后缀变体）
        5. 去下划线后大小写不敏感匹配（最宽松兜底）

        多行处理: 当同一算子名有多行（多次运行）时，跳过首行warmup，
        取剩余行中 Task Duration(us) 最小的行（best stable run）。

        Examples:
            fastgelu -> FastgeluCustom ✓
            FastGELU -> FastgeluCustom ✓
            fastgeluCustom -> FastgeluCustom ✓
            FastgeluCustom -> fastgelu ✓ (如果CSV中有fastgelu)
            aten_native_batch_norm -> AtenNativeBatchNormCustom ✓
        """
        op_names = df["Op Name"].unique()

        # 策略1: 精确匹配 (最快路径)
        if target_name in op_names:
            rows = df[df["Op Name"] == target_name]
            matched = self._best_row(rows)
            logger.info(
                f"  ✅ Exact match: '{target_name}' ({len(rows)} run(s), best={matched.get('Task Duration(us)')}us)")
            return matched

        # 策略2: 大小写不敏感匹配
        for name in op_names:
            if name.lower() == target_name.lower():
                rows = df[df["Op Name"] == name]
                matched = self._best_row(rows)
                logger.info(
                    f"  ✅ Case-insensitive match: '{target_name}' -> '{name}' ({len(rows)} run(s), best={matched['Task Duration(us)']}us)")
                return matched

        # 策略3: Custom后缀变体匹配
        variants = []

        # 尝试添加 Custom 后缀
        if not target_name.endswith("Custom"):
            variants.extend([
                target_name + "Custom",                      # fastgelu -> fastgeluCustom
                target_name.capitalize() + "Custom",         # fastgelu -> FastgeluCustom
                target_name.upper() + "Custom",              # fastgelu -> FASTGELUCustom
                target_name.lower() + "Custom"               # FastGelu -> fastgelucustom
            ])

        # 尝试去除 Custom 后缀
        if target_name.endswith("Custom") or target_name.lower().endswith("custom"):
            if target_name.endswith("Custom"):
                base = target_name[:-6]  # 去除 "Custom"
            else:
                base = target_name[:-6]  # 去除 "custom"

            variants.extend([
                base,
                base.lower(),
                base.capitalize(),
                base.upper()
            ])

        # 策略4: snake_case → PascalCase 转换（支持下划线分隔的算子名）
        if "_" in target_name:
            # aten_native_batch_norm → AtenNativeBatchNorm
            pascal = "".join(w.capitalize() for w in target_name.split("_"))
            variants.extend([
                pascal,                    # AtenNativeBatchNorm
                pascal + "Custom",         # AtenNativeBatchNormCustom
                pascal.lower() + "Custom",  # atennativebatchnormcustom
            ])

        # 尝试所有变体 (大小写不敏感)
        for variant in variants:
            for name in op_names:
                if name.lower() == variant.lower():
                    rows = df[df["Op Name"] == name]
                    matched = self._best_row(rows)
                    logger.info(
                        f"  ✅ Variant match: '{target_name}' -> '{name}' (via '{variant}', {len(rows)} run(s), best={matched['Task Duration(us)']}us)")
                    return matched

        # 策略5: 去下划线后大小写不敏感匹配（最宽松兜底）
        target_no_sep = target_name.lower().replace("_", "").replace("-", "")
        for name in op_names:
            name_no_sep = name.lower().replace("_", "").replace("-", "").replace("custom", "")
            if name_no_sep == target_no_sep or name_no_sep == target_no_sep.replace("custom", ""):
                rows = df[df["Op Name"] == name]
                matched = self._best_row(rows)
                logger.info(
                    f"  ✅ Normalized match: '{target_name}' -> '{name}' (no-separator comparison, {len(rows)} run(s), best={matched.get('Task Duration(us)')}us)")
                return matched

        # 未找到匹配
        return None


# 使用示例
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if len(sys.argv) < 2:
        logger.info("Usage: python profiling_extractor.py <csv_path>")
        sys.exit(1)

    csv = Path(sys.argv[1])
    extractor = ProfilingExtractor(csv)

    # 提取数据
    data = extractor.extract()

    logger.info(f"=== Operator Info ===")
    logger.info(f"Operator: {data.op_name}")
    logger.info(f"Type: {data.op_type}")
    logger.info(f"Task Type: {data.task_type}")
    logger.info(f"Task Duration: {data.task_duration_us} us")

    logger.info(f"\n=== Bottleneck Hint ===")
    logger.info(f"{data.get_bottleneck_hint()}")

    util = data.get_utilization_ratio()
    if util is not None:
        logger.info(f"\n=== Utilization ===")
        logger.info(f"Ratio: {util:.3f}")

    logger.info(f"\n=== Relevant Metrics ({len(data.relevant_metrics)}) ===")
    for k, v in list(data.relevant_metrics.items())[:10]:  # 显示前10个
        logger.info(f"  {k}: {v}")

    if len(data.relevant_metrics) > 10:
        logger.info(f"  ... and {len(data.relevant_metrics) - 10} more")

    logger.info(f"\n=== JSON Export ===")
    logger.info(json.dumps(data.to_dict(), indent=2, ensure_ascii=False, default=str))
