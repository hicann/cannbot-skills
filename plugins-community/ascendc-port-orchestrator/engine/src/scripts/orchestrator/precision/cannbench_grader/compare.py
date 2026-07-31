#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
张量精度对比引擎

职责：
1. 提供 compare_tensors() — 张量对比主入口（支持多输出）
2. SingleOutputResult / CompareResult 数据类
3. MERE/MARE 计算 + 小值域/相消兜底判定

从 utils/precision.py 拆分出来；阈值查询依赖 utils/thresholds.py。
"""

import logging
import math
import traceback
from typing import Union, Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch

from .thresholds import (
    get_threshold,
    get_small_value_threshold,
    get_small_value_error,
    get_cancel_boundary,
    get_cancel_zero_threshold,
)


_logger = logging.getLogger(__name__)


# Integer dtypes that should use absolute-difference / exact comparison
# rather than the floating-point MERE/MARE path. Must stay in sync with
# the int threshold entries in `thresholds.py` (which declares all 8).
_INTEGER_DTYPES = (
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.uint16, torch.uint32, torch.uint64,
)


@dataclass
class SingleOutputResult:
    """单个输出的对比结果（通用容器，各 checker 通过 metadata 扩展特有指标）"""
    index: int                      # 输出索引
    name: str = ""                  # 输出名称（可选）
    dtype: str = ""                 # 数据类型
    passed: bool = True
    mismatch_count: int = 0         # 不匹配元素数
    total_count: int = 0            # 总元素数
    error_msg: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'index': self.index,
            'name': self.name,
            'dtype': self.dtype,
            'passed': self.passed,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'error_msg': self.error_msg,
        }
        d.update(self.metadata)
        return d

    def format_summary(self) -> str:
        """格式化单输出判定摘要（通用实现，各 checker 子类覆盖此方法）"""
        dtype_str = f"{self.dtype}[{self.name or self.index}]"
        if self.passed:
            return f"{dtype_str}: ✅"
        else:
            return f"{dtype_str}: ❌ {self.error_msg}"


@dataclass
class CompareResult:
    """对比结果（支持多输出算子）"""
    passed: bool
    dtype: str
    threshold: float  # 精度阈值（聚合值）
    mere: float = 0.0  # 平均相对误差（聚合值）
    mare: float = 0.0  # 最大相对误差（聚合值）
    max_diff: float = 0.0
    mean_diff: float = 0.0
    mismatch_count: int = 0
    total_count: int = 0
    mismatch_ratio: float = 0.0
    small_value_error_count: int = 0  # 小值域 NPU 错误计数
    small_value_cpu_error_count: int = 0  # 小值域 CPU 错误计数
    small_value_total_count: int = 0  # 小值域总计数
    cancel_error_count: int = 0  # 相消位置 NPU 错误计数
    cancel_cpu_error_count: int = 0  # 相消位置 CPU 错误计数
    cancel_total_count: int = 0  # 相消位置总计数
    small_value_passed: bool = True  # 小值域兜底判定是否通过
    cancel_passed: bool = True       # 相消兜底判定是否通过
    normal_mismatch_count: int = 0   # ADDITIVE diagnostic exposure of the Stage-2 normal-region
                                     # over-threshold count (compare.py:~605). NOT a grading input —
                                     # the pass/fail verdict is byte-identical with/without this field.
                                     # See cannbench_grader/PROVENANCE.md (a5_ops additive patch).
    error_msg: Optional[str] = None
    output_results: List[SingleOutputResult] = field(default_factory=list)  # 各输出独立结果

    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'dtype': self.dtype,
            'threshold': self.threshold,
            'mere': self.mere,
            'mare': self.mare,
            'max_diff': self.max_diff,
            'mean_diff': self.mean_diff,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'mismatch_ratio': self.mismatch_ratio,
            'small_value_error_count': self.small_value_error_count,
            'small_value_cpu_error_count': self.small_value_cpu_error_count,
            'small_value_total_count': self.small_value_total_count,
            'cancel_error_count': self.cancel_error_count,
            'cancel_cpu_error_count': self.cancel_cpu_error_count,
            'cancel_total_count': self.cancel_total_count,
            'small_value_passed': self.small_value_passed,
            'cancel_passed': self.cancel_passed,
            'normal_mismatch_count': self.normal_mismatch_count,
            'error_msg': self.error_msg,
            'output_results': [r.to_dict() for r in self.output_results],
        }

    def format_all_outputs(self) -> str:
        """格式化所有输出判定结果（用于日志）"""
        lines = []
        for r in self.output_results:
            lines.append(f"  - {r.format_summary()}")
        return "\n".join(lines)


def _normalize_outputs(output: Any) -> List[torch.Tensor]:
    """将输出标准化为张量列表。

    F089: 旧版静默丢弃非 tensor 元素（str / None / int / 嵌套非 tensor），
    若 output 和 golden 各自 normalize 后 len 恰好相等但**对应位置不同**，
    后续 zip 比较会张冠李戴。改为保留 None 占位 + 丢弃元素超过 0 时 logger 警告。
    调用方拿到 None 占位时按"该位置无法对比"处理。
    """
    if isinstance(output, torch.Tensor):
        return [output]
    elif isinstance(output, (tuple, list)):
        result: List[Any] = []
        dropped = 0
        for item in output:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        result.append(sub_item)
                    else:
                        result.append(None)
                        dropped += 1
            else:
                # 保留 None 占位维持索引对齐
                result.append(None)
                if item is not None:
                    dropped += 1
        if dropped > 0:
            _logger.warning(
                "_normalize_outputs: 丢弃 %d 个非 tensor 元素（替换为 None 占位 "
                "维持索引对齐）。若 output / golden 含同类型非 tensor 数据需调用方"
                "在外层先做语义比较。",
                dropped,
            )
        return result
    else:
        return []


def _compare_single_tensor(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    dtype: str,
    native_output: Optional[torch.Tensor] = None,
) -> CompareResult:
    """对比单个张量（计算MERE/MARE + 小值域处理）

    生态算子精度标准：
    1. 平均相对误差（MERE）= avg(|actual - golden| / (|golden| + 1e-7))
    2. 最大相对误差（MARE）= max(|actual - golden| / (|golden| + 1e-7))
    3. 通过标准: MERE < threshold 且 MARE < 10 * threshold

    小值域处理（来自 docs/kernel_bench_design_v1.0.md）：
    当 |golden| < small_value_threshold 时，采用小值域通过标准：
    - ErrorCount = 统计满足 (|golden| < threshold 且 |actual - golden| > error) 的位置数
    - 通过标准: ErrorCount_npu / max(ErrorCount_native, 1) ≤ 2

    Args:
        output: NPU/算子输出张量
        golden: Golden参考输出（FP64精度）
        threshold: 精度阈值
        dtype: 数据类型字符串
        native_output: 同精度参考输出（可选）
                       如果不提供，使用 golden 截断到目标精度作为参考
    """
    # Golden runs on CPU while the AI op runs on NPU.
    # Normalize both sides to CPU so subtract/equal don't trip on mixed devices.
    if output.device.type != "cpu":
        output = output.cpu()
    if golden.device.type != "cpu":
        golden = golden.cpu()

    # 检查形状
    if output.shape != golden.shape:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=f"形状不匹配: output={output.shape}, golden={golden.shape}"
        )

    # Bit-exact 浮点路径: 当 threshold == 0 且为浮点 dtype 时, 通过 .view(int dtype)
    # 做字节级比较, 这样 +0.0 / -0.0 不会被 IEEE 754 相等性当作同值放过, NaN payload
    # 也按字节区分。整数 dtype 已经天然字节唯一, 走下方 torch.equal 路径即可。
    if threshold == 0 and output.is_floating_point():
        _BIT_VIEW = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }
        int_dtype = _BIT_VIEW.get(output.dtype)
        if int_dtype is not None:
            golden_cast = golden.to(output.dtype).contiguous()
            # F093: 检测 golden 在 cast 到更窄 output.dtype 时是否新产生 inf
            # (fp64 → fp16 时 |x|>65504 → inf)。若如此，bit-exact 比较的 fail
            # 实际是数据范围超 dtype，而非计算错误。给出明确告警避免误判方向。
            new_inf = torch.isinf(golden_cast) & ~torch.isinf(golden)
            if new_inf.any().item():
                _logger.warning(
                    "compare(bit-exact): golden cast 到 %s 时产生 %d 处新 inf "
                    "(数据范围超 output dtype)，bit-exact 失败可能源自数据溢出而非"
                    "计算错误。",
                    output.dtype, int(new_inf.sum().item()),
                )
            output_c = output.contiguous()
            out_bits = output_c.view(int_dtype)
            gold_bits = golden_cast.view(int_dtype)
            if torch.equal(out_bits, gold_bits):
                return CompareResult(
                    passed=True,
                    dtype=dtype,
                    threshold=threshold,
                    mere=0.0,
                    mare=0.0,
                    max_diff=0.0,
                    mean_diff=0.0,
                    mismatch_count=0,
                    total_count=output.numel(),
                    mismatch_ratio=0.0,
                    cancel_error_count=0,
                    cancel_cpu_error_count=0,
                    cancel_total_count=0,
                )
            mismatch_mask = out_bits != gold_bits
            mismatch_count = int(mismatch_mask.sum())
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=0.0,
                mean_diff=0.0,
                mismatch_count=mismatch_count,
                total_count=output.numel(),
                mismatch_ratio=mismatch_count / output.numel() if output.numel() > 0 else 0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
                error_msg=f"bit-exact 比较失败: {mismatch_count}/{output.numel()} 个元素字节不等 (包括 ±0.0、NaN payload、Inf 符号等差异)",
            )
        # 不支持 view 的浮点类型 (例如 fp8) 继续走下方 MERE/MARE 路径

    # 对于整数类型，使用绝对差值容差比较
    if output.dtype in _INTEGER_DTYPES:
        if torch.equal(output, golden):
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=0.0,
                mean_diff=0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        # 不完全相等时，检查差值是否在容差范围内
        diff = torch.abs(output.long() - golden.long())
        mismatch_mask = diff > max(threshold, 0)
        mismatch_count = int(mismatch_mask.sum())

        diff_float = diff.float()
        if mismatch_count == 0:
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
                mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
            mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
            mismatch_count=mismatch_count,
            total_count=output.numel(),
            mismatch_ratio=mismatch_count / output.numel() if output.numel() > 0 else 0.0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
            error_msg=f"整数类型差值超出容差({threshold}): {mismatch_count}/{output.numel()} 个元素不匹配",
        )

    # 浮点类型对比
    # Golden 主动截断到 output.dtype，模拟算子输出的精度限制
    target_dtype = output.dtype
    golden_truncated = golden.to(target_dtype).double()
    output_fp64 = output.double()

    # 处理 NaN
    if torch.any(torch.isnan(output_fp64)) or torch.any(torch.isnan(golden_truncated)):
        nan_out = torch.isnan(output_fp64)
        nan_gold = torch.isnan(golden_truncated)
        if not torch.all(nan_out == nan_gold):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="NaN位置不匹配"
            )

    # 处理 Inf
    # 饱和边界处理：当一方 inf 另一方有限值时，将 inf 替换为 dtype 最大有限值后继续比较。
    # 这处理 fp16 饱和边界场景（NPU fp32→fp16 截断到 inf，golden fp64→fp16 未越界）。
    # 替换后由 MRE/MARE 决定 pass/fail——只有底层数值本身足够接近才能通过。
    replaced_inf = False
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)  # 初始化 inf 匹配掩码
    if torch.any(torch.isinf(output_fp64)) or torch.any(torch.isinf(golden_truncated)):
        inf_out = torch.isinf(output_fp64)
        inf_gold = torch.isinf(golden_truncated)
        inf_mismatch = inf_out != inf_gold

        if torch.any(inf_mismatch):
            # 获取目标 dtype 的最大有限值
            if target_dtype == torch.float16:
                max_finite = float(torch.finfo(torch.float16).max)  # 65504.0
            elif target_dtype == torch.bfloat16:
                max_finite = float(torch.finfo(torch.bfloat16).max)  # ~3.389e38
            elif target_dtype == torch.float32:
                max_finite = float(torch.finfo(torch.float32).max)
            else:
                max_finite = float(torch.finfo(target_dtype).max)

            mismatch_count = int(inf_mismatch.sum())
            # F087: 旧版无条件 print 到 stdout，批量评测污染输出 / 干扰 JSON 管道。
            # 改 logging.info 让用户/CI 通过 LOG_LEVEL 控制是否显示。
            _logger.info(
                "[inf_sat] %d element(s) saturated to inf on one side only, "
                "replacing inf with %s and continuing comparison",
                mismatch_count, max_finite,
            )

            # 替换 inf 为最大有限值（保留符号）
            if torch.any(inf_out & ~inf_gold):
                mask = inf_out & ~inf_gold
                output_fp64[mask] = torch.sign(output_fp64[mask]) * max_finite
                replaced_inf = True
            if torch.any(inf_gold & ~inf_out):
                mask = inf_gold & ~inf_out
                golden_truncated[mask] = torch.sign(golden_truncated[mask]) * max_finite
                replaced_inf = True

        # 双方都 inf 且符号相同的位置，直接视为匹配并排除后续比较
        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_truncated[both_inf])):
                return CompareResult(
                    passed=False,
                    dtype=dtype,
                    threshold=threshold,
                    error_msg="Inf符号不匹配"
                )
            inf_match_mask[both_inf] = True  # 更新 inf 匹配掩码

    # 计算相对误差（标准公式）
    # 公式: |actual - golden| / (|golden| + 1e-7)
    diff = torch.abs(output_fp64 - golden_truncated)
    golden_abs = torch.abs(golden_truncated)
    denominator = golden_abs + 1e-7  # 防止除0

    relative_error = diff / denominator

    # 排除 NaN、Inf 和匹配的 Inf 位置
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    valid_relative_error = relative_error[valid_mask]

    if len(valid_relative_error) == 0:
        # F090: 旧版"全 NaN/Inf 匹配 = pass" 会掩盖"两个算子在同位置都产生 NaN
        # 但根因不同（一个溢出 / 一个除零）"的真实 bug。改为：
        # - 全部位置都是 NaN 且位置匹配 → 视为可疑（log warn，仍 passed=True
        #   维持兼容；调用方可基于 mismatch_count 决定是否进一步排查）
        # - 全部位置都是 Inf 且符号匹配 → 同上（已由 inf_match_mask 保证）
        # 若调用方需要更严格的判定，可在外层检查 (mere==0 and total>0)。
        has_nan = torch.isnan(output).any().item() if output.numel() > 0 else False
        if has_nan:
            _logger.warning(
                "compare: 所有有效位置均为 NaN/Inf 且匹配；两个算子均输出 NaN 的"
                "极端情况可能掩盖独立的数值 bug，请人工核查 case 输入分布。"
                " (total=%d, dtype=%s)",
                output.numel(), dtype,
            )
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=0.0,
            mean_diff=0.0,
            mismatch_count=0,
            total_count=output.numel(),
            mismatch_ratio=0.0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    mere = float(valid_relative_error.mean())
    mare = float(valid_relative_error.max())

    # 计算绝对差异统计
    valid_diff = diff[valid_mask]
    max_diff = float(valid_diff.max()) if len(valid_diff) > 0 else 0.0
    mean_diff = float(valid_diff.mean()) if len(valid_diff) > 0 else 0.0
    total_count = output.numel()

    # ============================================================
    # 第一阶段：整体相对误差判定（优先判定）
    # ============================================================
    # 原则：相对误差是主要判定标准，如果整体相对误差达标，直接通过
    # 小值域/相消判定只是为了处理"相对误差可能不合理"的特殊情况
    mare_threshold = 10 * threshold

    # 计算整体相对误差（包括所有有效位置）
    overall_mere = float(valid_relative_error.mean())
    overall_mare = float(valid_relative_error.max())

    # 如果整体相对误差通过，直接返回，不需要小值域/相消判定
    if overall_mere < threshold and overall_mare < mare_threshold:
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=overall_mere,
            mare=overall_mare,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=0,
            total_count=total_count,
            mismatch_ratio=0.0,
            small_value_error_count=0,
            small_value_cpu_error_count=0,
            small_value_total_count=0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    # ============================================================
    # 第二阶段：分析失败原因，使用兜底判定
    # ============================================================
    # 整体相对误差不通过，需要分析是哪些位置导致的，并判断是否属于特殊情况

    # 找出相对误差超标的点
    mismatch_mask = relative_error > mare_threshold
    mismatch_mask[~valid_mask] = False
    mismatch_count = int(mismatch_mask.sum())

    # === 小值域处理 ===
    # 获取小值域阈值
    small_value_threshold = get_small_value_threshold(dtype)
    small_value_error = get_small_value_error(dtype)

    # 筛选小值域位置: |golden| < small_value_threshold
    small_value_mask = golden_abs < small_value_threshold
    small_value_mask[~valid_mask] = False  # 排除NaN/Inf
    small_value_total_count = int(small_value_mask.sum())

    # 小值域 NPU 错误计数: |golden| < small_value_threshold 且 |output - golden| > small_value_error
    small_value_npu_error_mask = small_value_mask & (diff > small_value_error)
    small_value_error_count = int(small_value_npu_error_mask.sum())

    # 小值域同精度参考错误计数
    # 重要：native 和 NPU 的比较基准必须一致，都使用 golden_truncated（FP64 → target_dtype → FP64）
    # 这样才能公平比较两者在相同精度限制下与理论真值的误差差异
    if native_output is not None:
        # F088: native_output 应在 CPU 上但调用方契约不强制；防御性 .cpu() 后再
        # .double() 避免 NPU 上的 .double() 失败或与 golden_truncated 跨设备运算
        if native_output.device.type != "cpu":
            native_output = native_output.cpu()
        cpu_output_fp64 = native_output.double()
        # 同精度差异也与截断后的 golden 比较，保持基准一致
        cpu_diff = torch.abs(cpu_output_fp64 - golden_truncated)
    else:
        # golden 截断到目标精度后再升到 FP64，这就是同精度下的"理想"输出
        cpu_output_fp64 = golden.to(target_dtype).double()
        # 与截断后的 golden 比较（此时 cpu_output_fp64 == golden_truncated，diff = 0）
        cpu_diff = torch.abs(cpu_output_fp64 - golden_truncated)

    # 同精度小值域错误计数: |golden| < threshold 且 |native_output - golden_truncated| > error
    cpu_small_value_mask = small_value_mask  # 直接使用 NPU 的小值域 mask，保证一致
    cpu_small_value_error_mask = cpu_small_value_mask & (cpu_diff > small_value_error)
    small_value_cpu_error_count = int(cpu_small_value_error_mask.sum())

    # === 小值域兜底判定 ===
    # 小值域兜底判定：ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    # 设计意图：如果 CPU 在同精度下也会在小值域犯错，NPU 犯错可以容忍（ratio ≤ 2）
    # 但当 CPU 参考是完美截断（native_output=None 时 cpu_diff=0）且 CPU 无错误时，
    # NPU 的任何错误都不应被容忍，否则会出现误判（如 golden=1e-6, output=65504 仍通过）
    if small_value_total_count > 0:
        if small_value_cpu_error_count == 0:
            # CPU 无错误 → NPU 必须也无错误才算通过
            small_value_passed = (small_value_error_count == 0)
        else:
            max_cpu_error = max(small_value_cpu_error_count, 1)
            small_value_ratio = small_value_error_count / max_cpu_error
            small_value_passed = small_value_ratio <= 2
    else:
        small_value_passed = True

    # === 相消位置处理 ===
    # 获取相消阈值
    cancel_boundary = get_cancel_boundary(dtype)
    cancel_zero_threshold = get_cancel_zero_threshold(dtype)

    # 检测相消位置
    output_abs = torch.abs(output_fp64)
    output_near_zero = output_abs < cancel_zero_threshold
    golden_in_cancel_range = (golden_abs < cancel_boundary) & (golden_abs >= small_value_threshold)
    cancel_mask = output_near_zero & golden_in_cancel_range & valid_mask
    cancel_total_count = int(cancel_mask.sum())

    # 相消位置"错误"判断：相对误差超过 mare_threshold
    cancel_npu_error_mask = cancel_mask & (relative_error > mare_threshold)
    cancel_error_count = int(cancel_npu_error_mask.sum())

    # CPU 相对误差超标计数
    cpu_relative_error = cpu_diff / (golden_abs + 1e-7)
    cancel_cpu_error_mask = cancel_mask & (cpu_relative_error > mare_threshold)
    cancel_cpu_error_count = int(cancel_cpu_error_mask.sum())

    # 相消位置兜底判定：与小值域相同逻辑
    # 当 CPU 无错误时，NPU 的任何错误都不应被容忍
    if cancel_total_count > 0:
        if cancel_cpu_error_count == 0:
            cancel_passed = (cancel_error_count == 0)
        else:
            max_cpu_cancel_error = max(cancel_cpu_error_count, 1)
            cancel_ratio = cancel_error_count / max_cpu_cancel_error
            cancel_passed = cancel_ratio <= 2
    else:
        cancel_passed = True

    # === 分析失败原因 ===
    # 检查相对误差超标的点是否都在小值域/相消范围内
    mismatch_in_small_value = mismatch_mask & small_value_mask
    mismatch_in_cancel = mismatch_mask & cancel_mask
    mismatch_in_normal = mismatch_mask & ~small_value_mask & ~cancel_mask

    normal_mismatch_count = int(mismatch_in_normal.sum())

    # 最终判定逻辑：
    # 1. 如果正常值域位置有相对误差超标的点 → 直接失败（不是特殊情况）
    # 2. 如果只有小值域/相消位置超标 → 使用兜底判定
    if normal_mismatch_count > 0:
        # 正常值域位置有误差超标，直接失败
        passed = False
        # 计算排除小值域/相消后的 MERE/MARE 用于显示
        normal_mask = ~small_value_mask & ~cancel_mask & valid_mask
        normal_relative_error = relative_error[normal_mask]
        if len(normal_relative_error) > 0:
            display_mere = float(normal_relative_error.mean())
            display_mare = float(normal_relative_error.max())
        else:
            # 正常值域完全为空（所有位置都在小值域/相消），
            # 但 mismatch_in_normal > 0 说明有超标点不在小值域也不在相消范围——
            # 这只在 mask 定义有交叉遗漏时出现。fallback 到 overall 值避免 0 误差显示。
            display_mere = overall_mere
            display_mare = overall_mare
    else:
        # 只有特殊位置（小值域/相消）误差超标，使用兜底判定
        passed = small_value_passed and cancel_passed
        if passed:
            # 兜底判定通过时，显示排除小值域/相消后的 MERE/MARE，
            # 避免 overall 值被小值域的巨大相对误差拉高，造成"MARE=0.9 却通过"的误解
            normal_mask = ~small_value_mask & ~cancel_mask & valid_mask
            normal_relative_error = relative_error[normal_mask]
            if len(normal_relative_error) > 0:
                display_mere = float(normal_relative_error.mean())
                display_mare = float(normal_relative_error.max())
            else:
                display_mere = 0.0
                display_mare = 0.0
        else:
            # 兜底判定失败时，显示 overall MERE/MARE
            # 此时失败原因是小值域/相消位置的误差，而非正常值域误差。
            # 如果 normal 区域完全为空（所有位置都在小值域），normal 值会是 0，
            # 与 passed=False 矛盾——用户看到"零误差但失败"。
            # 使用 overall 值让用户看到实际误差大小，更符合直觉。
            display_mere = overall_mere
            display_mare = overall_mare

    return CompareResult(
        passed=passed,
        dtype=dtype,
        threshold=threshold,
        mere=display_mere,
        mare=display_mare,
        max_diff=max_diff,
        mean_diff=mean_diff,
        mismatch_count=mismatch_count,
        total_count=total_count,
        mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
        small_value_error_count=small_value_error_count,
        small_value_cpu_error_count=small_value_cpu_error_count,
        small_value_total_count=small_value_total_count,
        cancel_error_count=cancel_error_count,
        cancel_cpu_error_count=cancel_cpu_error_count,
        cancel_total_count=cancel_total_count,
        small_value_passed=small_value_passed,
        cancel_passed=cancel_passed,
        normal_mismatch_count=normal_mismatch_count,
    )


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    dtype: str = 'float32',
    threshold: Optional[float] = None,
    native_output: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ignore_output_indices: Optional[List[int]] = None,
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> CompareResult:
    """
    对比输出张量与Golden参考结果（采用MERE/MARE标准 + 小值域处理）

    Args:
        output: 算子输出（单个张量或多张量）
        golden: Golden参考输出（单个张量或多张量），通常为 FP64 精度
        dtype: 数据类型字符串
        threshold: 精度阈值（可选，默认根据dtype自动选择）
        native_output: 同精度参考输出（可选，用于小值域比较）
                       如果不提供，则使用 golden 截断到目标精度作为参考
        ignore_output_indices: 需要忽略对比的输出索引列表
        custom_thresholds: 自定义精度阈值表（优先级高于默认阈值）

    Returns:
        CompareResult: 对比结果

    通过条件:
        正常值域: MERE < threshold 且 MARE < 10 * threshold
        小值域: ErrorCount_npu / max(ErrorCount_native, 1) ≤ 2
    """
    # 获取阈值（优先使用自定义阈值）
    if custom_thresholds is None:
        custom_thresholds = {}

    def _get_output_threshold(dtype_str: str) -> float:
        """获取单个输出的阈值（优先自定义，其次默认）"""
        dtype_lower = dtype_str.lower()
        if dtype_lower in custom_thresholds:
            return custom_thresholds[dtype_lower]
        return get_threshold(dtype_str)

    if threshold is None:
        threshold = _get_output_threshold(dtype)

    try:
        # 处理多输出情况
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)
        native_outputs = _normalize_outputs(native_output) if native_output is not None else None

        if len(outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"输出数量不匹配: output={len(outputs)}, golden={len(goldens)}"
            )

        if native_outputs is not None and len(native_outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"同精度输出数量不匹配: native_output={len(native_outputs)}, golden={len(goldens)}"
            )

        # 逐个对比
        all_passed = True
        mere_sum = 0.0
        mare_max = 0.0
        max_diff = 0.0
        mean_diff = 0.0
        mismatch_count = 0
        total_count = 0
        small_value_error_count = 0
        small_value_cpu_error_count = 0
        small_value_total_count = 0
        cancel_error_count = 0
        cancel_cpu_error_count = 0
        cancel_total_count = 0
        all_small_value_passed = True
        all_cancel_passed = True
        total_normal_mismatch = 0

        # 记录每个输出的独立判定结果
        single_output_results: List[SingleOutputResult] = []

        for i, (out_tensor, gold_tensor) in enumerate(zip(outputs, goldens)):
            # 处理可选输出：双方均为 None 则跳过，否则报错
            if out_tensor is None or gold_tensor is None:
                both_none = out_tensor is None and gold_tensor is None
                single_output_results.append(SingleOutputResult(
                    index=i,
                    name="",
                    dtype="none",
                    passed=both_none,
                    error_msg=(
                        ""
                        if both_none
                        else (
                            f"output[{i}] is None (candidate="
                            f"{'None' if out_tensor is None else 'Tensor'}, "
                            f"golden={'None' if gold_tensor is None else 'Tensor'})"
                        )
                    ),
                    metadata={'dtype_category': 'none'},
                ))
                continue

            # 跳过不需要对比的输出
            if ignore_output_indices and i in ignore_output_indices:
                # 创建跳过标记的 SingleOutputResult
                single_output_results.append(SingleOutputResult(
                    index=i,
                    name="",  # 名称由调用方填充
                    dtype=str(out_tensor.dtype).replace('torch.', ''),
                    passed=True,  # 跳过的输出视为通过
                    error_msg="(跳过对比)",
                    metadata={'dtype_category': 'int' if out_tensor.dtype in _INTEGER_DTYPES else 'float'},
                ))
                continue

            # 根据每个输出的实际 dtype 获取阈值（优先自定义阈值）
            out_dtype_str = str(out_tensor.dtype).replace('torch.', '')
            out_threshold = _get_output_threshold(out_dtype_str)
            out_dtype_category = 'int' if out_tensor.dtype in _INTEGER_DTYPES else 'float'

            native_tensor = native_outputs[i] if native_outputs is not None else None
            result = _compare_single_tensor(out_tensor, gold_tensor, out_threshold, out_dtype_str, native_tensor)

            # 转换 CompareResult 到 SingleOutputResult
            single_result = SingleOutputResult(
                index=i,
                name="",  # 名称由调用方填充
                dtype=out_dtype_str,
                passed=result.passed,
                mismatch_count=result.mismatch_count,
                total_count=result.total_count,
                error_msg=result.error_msg or "",
                metadata={
                    'dtype_category': out_dtype_category,
                    'threshold': out_threshold,
                    'mere': result.mere,
                    'mare': result.mare,
                    'max_diff': result.max_diff,
                    'mean_diff': result.mean_diff,
                    'max_abs_diff': int(result.max_diff) if out_dtype_category == 'int' else 0,
                    'small_value_error_count': result.small_value_error_count,
                    'small_value_cpu_error_count': result.small_value_cpu_error_count,
                    'small_value_total_count': result.small_value_total_count,
                    'cancel_error_count': result.cancel_error_count,
                    'cancel_cpu_error_count': result.cancel_cpu_error_count,
                    'cancel_total_count': result.cancel_total_count,
                    'small_value_passed': result.small_value_passed,
                    'cancel_passed': result.cancel_passed,
                },
            )
            single_output_results.append(single_result)

            is_passed = result.passed
            all_passed = all_passed and is_passed
            mere_sum += result.mere * result.total_count
            mare_max = max(mare_max, result.mare)
            max_diff = max(max_diff, result.max_diff)
            mean_diff += result.mean_diff * result.total_count
            mismatch_count += result.mismatch_count
            total_count += result.total_count
            small_value_error_count += result.small_value_error_count
            small_value_cpu_error_count += result.small_value_cpu_error_count
            small_value_total_count += result.small_value_total_count
            cancel_error_count += result.cancel_error_count
            cancel_cpu_error_count += result.cancel_cpu_error_count
            cancel_total_count += result.cancel_total_count
            all_small_value_passed = all_small_value_passed and result.small_value_passed
            all_cancel_passed = all_cancel_passed and result.cancel_passed
            total_normal_mismatch += result.normal_mismatch_count

        if total_count > 0:
            mere = mere_sum / total_count
            mean_diff = mean_diff / total_count
        else:
            mere = 0.0

        # 最终通过条件
        passed = all_passed

        # 确定返回的 dtype 和 threshold
        # 如果有失败输出，返回第一个失败输出的阈值信息
        # 否则返回第一个输出的阈值信息
        result_dtype = dtype
        result_threshold = threshold
        for sr in single_output_results:
            if not sr.passed and not sr.error_msg.startswith("(跳过"):
                result_dtype = sr.dtype
                result_threshold = sr.metadata.get('threshold', threshold)
                break

        return CompareResult(
            passed=passed,
            dtype=result_dtype,
            threshold=result_threshold,
            mere=mere,
            mare=mare_max,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=mismatch_count,
            total_count=total_count,
            mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
            small_value_error_count=small_value_error_count,
            small_value_cpu_error_count=small_value_cpu_error_count,
            small_value_total_count=small_value_total_count,
            cancel_error_count=cancel_error_count,
            cancel_cpu_error_count=cancel_cpu_error_count,
            cancel_total_count=cancel_total_count,
            small_value_passed=all_small_value_passed,
            cancel_passed=all_cancel_passed,
            normal_mismatch_count=total_normal_mismatch,
            output_results=single_output_results,  # 新增：各输出独立结果
        )

    except Exception as e:
        # 顶层 except 会吞掉所有内部逻辑（in-place mutation、shape 比较、
        # MERE/MARE 计算等）的 traceback。把堆栈附在 error_msg 末尾，方便排查。
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def compare_with_custom_threshold(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold_dict: Dict[str, float] = None,
) -> CompareResult:
    """
    使用自定义阈值表对比

    Args:
        output: 算子输出张量
        golden: Golden参考输出张量
        threshold_dict: 自定义阈值表，格式为 {dtype: threshold}

    Returns:
        CompareResult: 对比结果
    """
    from .thresholds import PRECISION_THRESHOLDS as _DEFAULT_THRESHOLDS

    if threshold_dict is None:
        threshold_dict = _DEFAULT_THRESHOLDS

    # 从张量推断dtype
    dtype_str = str(output.dtype).replace('torch.', '')
    threshold = get_threshold(dtype_str)
    if dtype_str.lower() in threshold_dict:
        threshold = threshold_dict[dtype_str.lower()]

    return compare_tensors(output, golden, dtype_str, threshold)
