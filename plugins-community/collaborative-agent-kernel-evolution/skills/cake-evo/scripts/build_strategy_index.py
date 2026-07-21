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
Build strategy index from expert_ideas.json files.

Reads all IdeaPool/*/expert_ideas.json, groups optimization strategies by
theme, detects variants within each group, and writes:
  - references/meta_prompts/strategy_index.md   (master index table)
  - references/meta_prompts/strategies/*.md     (~22 detail files)
  - references/meta_prompts/strategies/_unmatched.md  (review queue)
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy group definitions
# Each group: (file_id, meta)
# meta keys: category, id, title, index_title, when_to_apply, keywords
# Ordered from MOST SPECIFIC to MOST GENERAL within each category;
# first keyword match wins.
# ---------------------------------------------------------------------------

DTYPE_STRATEGIES = [
    ("dtype_04_fp8_int4_conversion", {
        "category": "multi_dtype_support",
        "id": "D4", "id_sort": 4,
        "title": "FP8/INT4 Quantization Conversion (FP8/INT4量化输出类型)",
        "index_title": "FP8/INT4 quantization conversion",
        "when_to_apply": "Quantization output operators",
        "keywords": ["FP8", "INT4特殊", "INT4地址", "HIFLOAT8", "FLOAT8",
                     "INT4输出量化", "混合精度输出支持（INT8"],
    }),
    ("dtype_03_tilingkey_dispatch", {
        "category": "multi_dtype_support",
        "id": "D3", "id_sort": 3,
        "title": "TilingKey-Driven Type Dispatch (TilingKey驱动类型分发)",
        "index_title": "TilingKey-driven type dispatch",
        "when_to_apply": "Host-side multi-type selection",
        "keywords": ["TilingKey", "Tiling Key", "TILING_KEY",
                     "Mean/Rstd输出的类型适配", "数据类型转换的向量化优化",
                     "分层类型转换策略", "数据类型感知的 Tiling",
                     "类型转换精细化控制", "差异化数据类型转换策略",
                     "条件编译支持的TilingKey"],
    }),
    ("dtype_02_template_kernel", {
        "category": "multi_dtype_support",
        "id": "D2", "id_sort": 2,
        "title": "Template Kernel Type Dispatch (模板化内核类型分发)",
        "index_title": "Template kernel type dispatch",
        "when_to_apply": "Compile-time multi-type kernel",
        "keywords": ["模板化的Kernel", "通用Kernel架构", "CRTP",
                     "模板泛型编程", "条件模板特化",
                     "模板化的数据类型", "模板化数据类型",
                     "模板化多数据类型", "模板化多类型",
                     "模板化混合精度", "模板化的多类型",
                     "模板化的类型特化", "混合数据类型的独立队列",
                     "数据类型转换工具函数", "统一的数据类型转换框架",
                     "模板化编程与编译期", "模板多态架构与TilingKey",
                     "模板化架构设计", "编译时多态"],
    }),
    ("dtype_05_bf16_special_handling", {
        "category": "multi_dtype_support",
        "id": "D5", "id_sort": 5,
        "title": "BF16 & Platform-Specific Handling (BF16/多平台处理)",
        "index_title": "BF16-specific handling",
        "when_to_apply": "BF16 output with rounding control or multi-platform",
        "keywords": ["架构特定的数据类型", "多平台数据类型配置", "多平台硬件适配",
                     "多平台配置支持", "多SOC平台适配", "多硬件平台差异化配置",
                     "芯片差异化配置策略", "条件编译处理芯片差异",
                     "Kirin平台特定配置", "BFLOAT16特殊UB布局",
                     "16位数据的Buffer布局", "BF16特殊", "差异化RoundMode",
                     "差异化Round Mode", "FP16/BF16精度保持策略",
                     "差异化精度处理策略", "动态Shape支持配置",
                     "条件编译支持多硬件", "多硬件平台适配",
                     "ASCEND310P", "310P专用"],
    }),
    ("dtype_01_mixed_precision", {
        "category": "multi_dtype_support",
        "id": "D1", "id_sort": 1,
        "title": "Mixed Precision Architecture (混合精度架构)",
        "index_title": "Mixed precision architecture",
        "when_to_apply": "Any op with FP16/BF16 input",
        "keywords": ["多数据类型", "多精度数据类型", "数据类型矩阵",
                     "丰富的数据类型", "完整的多", "完整数据类型",
                     "多精度", "精度数据类型组合",
                     "混合精度计算架构", "混合精度输入",
                     "双精度输入支持", "数据类型与精度控制",
                     "数据类型支持", "数据类型覆盖", "数据类型组合",
                     "数据类型扩展", "数据类型多样化", "数据类型推导",
                     "数据类型统一", "全数据类型", "全面的数据类型",
                     "数据类型校验", "统一的类型检查", "数据类型感知的Buffer",
                     "动态输出数据类型", "混合精度计算路径",
                     "动态数据类型分发", "灵活的数据类型",
                     "数据类型自动推导", "可选输入灵活处理",
                     "Values标量", "动态输入列表", "动态Tensor List",
                     "Host端多类型Proto", "数据类型尺寸自适应",
                     "中间计算类型提升", "双输出动态量化",
                     "可选参数状态管理"],
    }),
]

PERF_STRATEGIES = [
    ("perf_03_small_d_optimization", {
        "category": "performance",
        "id": "P3", "id_sort": 3,
        "title": "Small-D Multi-Row Merging (小D多行合并优化)",
        "index_title": "Small-D multi-row merging",
        "when_to_apply": "Hidden size ≤ 640 or small D",
        "keywords": ["小D优化", "Small-D", "SMALLD", "ComputeSmallD",
                     "多行合并", "小维度排序优化", "长短H场景优化",
                     "SingleN极致单核优化"],
    }),
    ("perf_09_deterministic_output", {
        "category": "performance",
        "id": "P9", "id_sort": 9,
        "title": "Deterministic Output via Workspace (确定性输出)",
        "index_title": "Deterministic output (workspace)",
        "when_to_apply": "Training ops needing reproducibility",
        "keywords": ["确定性输出", "Deterministic Output", "fixed_output",
                     "双输出量化的Workspace优化",
                     "SetAtomicAdd处理多核竞争",
                     "16MB Workspace预留",
                     "确定性计算模式"],
    }),
    ("perf_11_tail_block_handling", {
        "category": "performance",
        "id": "P11", "id_sort": 11,
        "title": "Tail Block Handling (尾块与边界处理)",
        "index_title": "Tail block handling (GatherMask)",
        "when_to_apply": "Pooling/gather with uneven splits",
        "keywords": ["尾块", "GatherMask", "Tail Block",
                     "Kernel索引预生成与Mask向量化",
                     "Argmax索引转换优化",
                     "余数处理与边界对齐",
                     "Adaptive Pooling重叠检测"],
    }),
    ("perf_01_double_buffer", {
        "category": "performance",
        "id": "P1", "id_sort": 1,
        "title": "Double Buffering (双缓冲机制)",
        "index_title": "Double buffering (BUFFER_NUM=2)",
        "when_to_apply": "Any memory-bound kernel",
        "keywords": ["双缓冲", "Double Buffering", "Double Buffer",
                     "BUFFER_NUM", "DOUBLE_BUFFER", "Ping-Pong Buffer",
                     "Ping/Pong", "pingBuf", "pongBuf", "数据预取"],
    }),
    ("perf_06_multi_algo_selection", {
        "category": "performance",
        "id": "P6", "id_sort": 6,
        "title": "Multi-Algorithm Adaptive Selection (多算法自适应选择)",
        "index_title": "Multi-algorithm adaptive selection",
        "when_to_apply": "Normalization, reduction ops",
        "keywords": ["多算法策略", "多算法", "算法策略动态", "多种算法",
                     "Binary Reduction树", "Gamma/Beta可选",
                     "Welford算法的数值稳定实现",
                     "Welford在线算法优化",
                     "场景感知的算法分派",
                     "高效的Reduce操作实现",
                     "差异化Reduce策略",
                     "自定义 ReduceSum 算法",
                     "多策略分发", "Kernel多策略分发",
                     "多模式自适应计算策略",
                     "多模式自适应执行策略",
                     "多策略自适应调度",
                     "多Tiling Key策略实现多路径",
                     "Tiling Key多分支路由策略",
                     "Tiling Key策略与多分支编译",
                     "TILING_KEY多路径分发",
                     "TilingKey多模式分发",
                     "FastCompute vs SliceCompute",
                     "双路径梯度计算策略",
                     "分层量化流水线",
                     "架构特定的ReduceSum优化",
                     "分层Tiling策略（Norm",
                     "双模式动态处理策略"],
    }),
    ("perf_02_adaptive_tiling", {
        "category": "performance",
        "id": "P2", "id_sort": 2,
        "title": "Adaptive Tiling (自适应分块策略)",
        "index_title": "Adaptive tiling (Split-N vs Split-D)",
        "when_to_apply": "Variable row/column dimensions",
        "keywords": ["自适应Tiling", "Split-N", "Split-D",
                     "自适应分块", "输入维度自适应", "三级Tiling分层",
                     "Regbase架构优化", "分块计算与Tiling",
                     "三模式UB Tiling", "多策略Tiling系统",
                     "多层级 Tiling", "多维度Tiling",
                     "多级分块与并行", "自动Tiling",
                     "智能Tiling", "三层次Tiling",
                     "分块处理与边界优化",
                     "动态Tiling与UB感知分块",
                     "自适应多策略Tiling系统",
                     "Smart Tiling 与多维分块",
                     "多维度Tiling策略",
                     "基于UB大小的动态分块",
                     "基于UB容量的智能Tiling",
                     "双模板策略（KernelRmsNorm",
                     "平台自适应与CompileInfo",
                     "多维分块", "UB感知分块",
                     "多策略Tiling", "统一Buffer管理与多级内存"],
    }),
    ("perf_04_load_balance", {
        "category": "performance",
        "id": "P4", "id_sort": 4,
        "title": "Multi-Core Load Balancing (多核负载均衡)",
        "index_title": "Multi-core load balancing",
        "when_to_apply": "Uneven data distributions",
        "keywords": ["负载均衡", "核心负载", "多核负载",
                     "动态核数分配", "均衡分块算法",
                     "多核并行与原子操作", "精细的多核并行",
                     "多核细粒度负载", "精细化的多核负载",
                     "多核并行与任务分配",
                     "Per-Head量化", "合轴", "MergeAxis"],
    }),
    ("perf_05_pipeline_sync", {
        "category": "performance",
        "id": "P5", "id_sort": 5,
        "title": "Pipeline Synchronization (流水线同步控制)",
        "index_title": "Pipeline sync (PipeBarrier/events)",
        "when_to_apply": "All double-buffered kernels",
        "keywords": ["精细同步控制", "AtomicAdd跨核累加",
                     "精细化事件同步与流水线",
                     "精细的Pipe Barrier流水线",
                     "流水线屏障(PipeBarrier)", "流水线屏障与事件同步",
                     "流水线屏障同步",
                     "8事件ID驱动的异步流水线",
                     "显式流水线同步", "流水线显式同步",
                     "异步流水线同步", "精细的Pipe Barrier",
                     "Pipeline Barrier精确控制",
                     "数据类型转换的延迟隐藏",
                     "流水线同步优化", "PipeBarrier流水线同步",
                     "流水线优化与内存", "向量化计算与流水线"],
    }),
    ("perf_07_data_alignment", {
        "category": "performance",
        "id": "P7", "id_sort": 7,
        "title": "32B Alignment + DataCopyPad (数据对齐与填充)",
        "index_title": "32B alignment + DataCopyPad",
        "when_to_apply": "Non-aligned input shapes",
        "keywords": ["DataCopyPad", "数据对齐", "内存对齐",
                     "32字节对齐", "Padding策略",
                     "SafeDataCopy", "非对齐数据",
                     "数据对齐与块级拷贝", "向量化内存访问与对齐",
                     "数据对齐与向量化访问",
                     "内存对齐与访问模式", "内存对齐与数据搬运",
                     "内存对齐与Padding", "内存对齐与数据布局",
                     "数据对齐与向量化优化"],
    }),
    ("perf_10_vectorized_copy", {
        "category": "performance",
        "id": "P10", "id_sort": 10,
        "title": "Vectorized Data Copy & Instructions (向量化数据与指令)",
        "index_title": "Vectorized data copy",
        "when_to_apply": "CopyIn/CopyOut optimization",
        "keywords": ["向量化指令与数据", "向量化拷贝", "向量化传输",
                     "向量化数据拷贝",
                     "Transpose向量化优化",
                     "向量化Gather指令",
                     "Gather/Scatter 向量化",
                     "Axpy指令优化", "Axpy指令融合",
                     "纯向量化计算",
                     "DataCopyExt优化内存传输",
                     "优化 DataCopy 策略",
                     "MicroAPI高性能指令",
                     "MicroAPI 寄存器级优化",
                     "寄存器级向量化优化",
                     "寄存器级APT优化",
                     "RegBase策略的寄存器级优化",
                     "Transpose策略的内存访问优化",
                     "Transpose优化的内存访问",
                     "向量化 Reduce 优化",
                     "Adds专用指令优化",
                     "ScalarPow的向量化实现",
                     "向量化指令与数据对齐",
                     "SIMT向量化并行计算",
                     "负数索引向量化处理",
                     "ArithProgression序列生成",
                     "多维索引生成（ArithProgression）",
                     "Mask数据搬运模式优化",
                     "指令选择优化",
                     "Gamma预加载复用",
                     "内存访问封装DataCopyEx",
                     "内存布局优化与LoopMode",
                     "Gather/Scatter 向量化内存访问"],
    }),
    ("perf_08_ub_memory_mgmt", {
        "category": "performance",
        "id": "P8", "id_sort": 8,
        "title": "UB Memory Partitioning (UB内存分区管理)",
        "index_title": "UB memory partitioning",
        "when_to_apply": "Kernels with multiple UB tensors",
        "keywords": ["UB内存", "UB大小", "UB优化",
                     "IndexBuffer预计算",
                     "Buffer池化复用",
                     "内存复用与原地操作",
                     "统一UB大小管理",
                     "UB内存精细管理",
                     "UB内存精细化管理",
                     "UB大小感知的数据分块",
                     "动态UB", "原地修改检测"],
    }),
    ("perf_12_broadcast_mask", {
        "category": "performance",
        "id": "P12", "id_sort": 12,
        "title": "Broadcast & Mask Operations (广播与掩码操作)",
        "index_title": "Broadcast & mask operations",
        "when_to_apply": "Operators with mask inputs, broadcasting dimensions, or conditional selection",
        "keywords": ["MaskOffset", "SelectWithBytesMask", "BA/AB广播",
                     "Mask广播", "标量数据的广播", "广播模式识别",
                     "Mask处理的精度保证", "BroadcastTo"],
    }),
    ("perf_13_special_algorithms", {
        "category": "performance",
        "id": "P13", "id_sort": 13,
        "title": "Special Algorithms & High-Level AscendC APIs (特殊算法与高阶API)",
        "index_title": "Special algorithms & high-level AscendC APIs",
        "when_to_apply": "Complex control flow, irregular access, or domain-specific high-level APIs",
        "keywords": ["SIMT编程模型", "Magic Number快速", "DAG计算图",
                     "Bind<>", "DAGSch",
                     "奇偶排列", "数据布局自适应", "双线性插值",
                     "LastDim模式", "分层归约", "早停优化", "Early Exit",
                     "Softmax高阶API", "Exp-based Swish", "浮点除法转乘法",
                     "TensorList遍历", "动态Scalar加载", "二分法求",
                     "基类-派生类架构", "内存自动连续化",
                     "可选参数状态管理", "索引类型的向量化"],
    }),
]

ACC_STRATEGIES = [
    ("acc_06_high_precision_rsqrt", {
        "category": "accuracy",
        "id": "A6", "id_sort": 6,
        "title": "High-Precision rsqrt via Newton-Raphson (高精度rsqrt)",
        "index_title": "High-precision rsqrt (Newton-Raphson)",
        "when_to_apply": "Normalization ops needing rsqrt",
        "keywords": ["高精度rsqrt", "Newton-Raphson", "高精度rstd计算",
                     "rsqrt的Newton", "Newton迭代",
                     "牛顿迭代优化RMS", "Rsqrt牛顿迭代优化",
                     "rsqrt"],
    }),
    ("acc_02_welford_algorithm", {
        "category": "accuracy",
        "id": "A2", "id_sort": 2,
        "title": "Welford Numerically Stable Mean/Var (Welford数值稳定算法)",
        "index_title": "Welford numerically stable mean/var",
        "when_to_apply": "LayerNorm, BatchNorm, RMSNorm",
        "keywords": ["Welford算法", "在线均值",
                     "在线方差", "Welford增量", "WelfordUpdate",
                     "Welford数值", "Batch Variance无偏估计",
                     "Softmax梯度计算数值稳定性",
                     "AMSGrad变体的数值稳定性"],
    }),
    ("acc_05_softmax_stability", {
        "category": "accuracy",
        "id": "A5", "id_sort": 5,
        "title": "Numerical Safety & Special Value Handling (数值安全与特殊值处理)",
        "index_title": "Softmax numerical stability",
        "when_to_apply": "Softmax, attention score ops, any op with NaN/Inf risk",
        "keywords": ["Softmax数值", "softmax稳定",
                     "数值稳定性", "NaN 处理", "NaN处理",
                     "特殊值处理", "除零保护", "负无穷值",
                     "溢出控制", "Clip前置防止",
                     "边界条件处理与数值", "坐标裁剪与归一化",
                     "minScale精度保护", "动态量化Scale计算的数值稳定性",
                     "边界检查与数值稳定性",
                     "Ceil Mode 的边界修正",
                     "Padding 区域 Argmax 修正",
                     "Epsilon的灵活配置", "Epsilon 和 Alpha",
                     "Gemma 特殊语义"],
    }),
    ("acc_03_rounding_mode", {
        "category": "accuracy",
        "id": "A3", "id_sort": 3,
        "title": "Rounding Mode Control (舍入模式控制)",
        "index_title": "Rounding mode control (CAST_*)",
        "when_to_apply": "Any Cast to lower precision",
        "keywords": ["舍入模式", "多模式舍入", "Sqrt模式", "CAST_NONE",
                     "CAST_RINT", "舍入控制",
                     "多种 Rounding Mode",
                     "差异化RoundMode选择",
                     "量化范围灵活配置"],
    }),
    ("acc_04_pipeline_barrier", {
        "category": "accuracy",
        "id": "A4", "id_sort": 4,
        "title": "SetFlag/WaitFlag Event Sync (事件同步保证精度)",
        "index_title": "SetFlag/WaitFlag event sync",
        "when_to_apply": "Data dependencies across pipes",
        "keywords": ["Pipeline Barrier同步控制", "PipeBarrier同步控制",
                     "PipeBarrier流水线同步",
                     "Pipeline Barrier精确控制",
                     "PipeBarrier同步", "分阶段 rstd"],
    }),
    ("acc_01_fp32_intermediate", {
        "category": "accuracy",
        "id": "A1", "id_sort": 1,
        "title": "FP32 Intermediate Computation (FP32中间计算)",
        "index_title": "FP32 intermediate computation",
        "when_to_apply": "BF16/FP16 with precision requirement",
        "keywords": ["高精度中间计算", "FP32中间", "高精度ReduceSum",
                     "高精度reduce", "BF16的FP32",
                     "FP32 中间计算", "Float32中间计算",
                     "FP32 累加保护",
                     "低精度类型的浮点精度提升",
                     "统一计算类型",
                     "中间计算精度提升",
                     "高精度除法指令",
                     "BF16计算精度保护",
                     "FP16/BF16精度", "FP32精度",
                     "BF16高精度中间计算", "混合精度计算策略", "平均因子"],
    }),
    ("acc_07_index_boundary", {
        "category": "accuracy",
        "id": "A7", "id_sort": 7,
        "title": "Index & Boundary Safety (索引与边界安全处理)",
        "index_title": "Index & boundary safety",
        "when_to_apply": "Any operator with index inputs, gather/scatter, or user-controlled indices",
        "keywords": ["负索引正确处理", "索引越界", "Padding索引排除",
                     "Argmax索引类型", "int64索引的安全",
                     "assert断言保护", "边界条件严格检查",
                     "输入校验与边界", "边界范围检查",
                     "负索引", "边界检查与索引转换",
                     "精确的Shape推导"],
    }),
    ("acc_08_quant_precision", {
        "category": "accuracy",
        "id": "A8", "id_sort": 8,
        "title": "Quantization-Specific Precision (量化专用精度处理)",
        "index_title": "Quantization-specific precision",
        "when_to_apply": "Quantization output operators, custom float formats, dequantization with bias",
        "keywords": ["FP32到FP19", "FP19量化", "Scale/Offset打包",
                     "Int9范围", "INT32类型Bias",
                     "动态Range量化", "量化边界精确控制",
                     "Gemma Mode", "量化转换 - 位操作",
                     "位操作优化", "Uint64"],
    }),
]

# Build flat lookup: category -> ordered list of (file_id, meta)
STRATEGY_MAP = {
    "multi_dtype_support": DTYPE_STRATEGIES,
    "performance": PERF_STRATEGIES,
    "accuracy": ACC_STRATEGIES,
}

# Flat dict for all strategies
ALL_STRATEGIES = {
    _flat_sid: _flat_meta
    for _flat_lst in [DTYPE_STRATEGIES, PERF_STRATEGIES, ACC_STRATEGIES]
    for _flat_sid, _flat_meta in _flat_lst
}


# ---------------------------------------------------------------------------
# Helper: identifier extraction and Jaccard similarity
# ---------------------------------------------------------------------------

def extract_identifiers(code: str) -> set:
    """Extract C++ identifiers longer than 3 chars from a code snippet."""
    if not code:
        return set()
    tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{3,}\b', code)
    # Filter out common keywords
    stop = {'void', 'auto', 'const', 'true', 'false', 'nullptr', 'return',
            'uint', 'int32', 'int64', 'float', 'bool', 'else', 'this',
            'template', 'typename', 'constexpr', 'inline', 'class', 'struct',
            'static', 'size', 'uint32', 'uint64', 'uint16', 'int16'}
    return {t for t in tokens if t.lower() not in stop}


def jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ---------------------------------------------------------------------------
# Matching: assign each item to a strategy group
# ---------------------------------------------------------------------------

def match_item(item: dict, category: str) -> str | None:
    """Return strategy file_id, or None if unmatched."""
    name = item.get("name", "")
    analysis = item.get("analysis", "")[:300]
    text = name + " " + analysis

    # Cross-category matching: search all strategies regardless of category
    all_strats = DTYPE_STRATEGIES + PERF_STRATEGIES + ACC_STRATEGIES
    for sid, meta in all_strats:
        for kw in meta["keywords"]:
            if kw in name or kw in text:
                return sid
    return None


# ---------------------------------------------------------------------------
# Variant clustering within a strategy group
# ---------------------------------------------------------------------------

def cluster_variants(items: list) -> list[list]:
    """
    Cluster items into variants based on Jaccard similarity of
    expert_code_snippet identifiers.  Threshold: 0.50.
    Returns list of clusters (each cluster is a list of items).
    """
    if not items:
        return []

    id_sets = [extract_identifiers(it.get("expert_code_snippet", ""))
               for it in items]

    # Union-Find
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if jaccard(id_sets[i], id_sets[j]) > 0.50:
                union(i, j)

    clusters: dict[int, list] = defaultdict(list)
    for i, item in enumerate(items):
        clusters[find(i)].append(item)

    # Sort clusters largest first (most representative variant first)
    return sorted(clusters.values(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

VARIANT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _code_block(code: str, lang: str = "cpp") -> str:
    code = (code or "").strip()
    if not code or code.startswith("// 无") or code.startswith("// No"):
        return ""
    return f"```{lang}\n{code}\n```"


def _benefit_tradeoff(items: list) -> tuple[str, str]:
    """Aggregate benefit and trade_off from all items in a cluster."""
    benefits = []
    tradeoffs = []
    seen_b: set[str] = set()
    seen_t: set[str] = set()
    for it in items:
        b = (it.get("benefit") or "").strip()
        t = (it.get("trade_off") or "").strip()
        if b and b not in seen_b:
            benefits.append(b)
            seen_b.add(b)
        if t and t not in seen_t:
            tradeoffs.append(t)
            seen_t.add(t)
    return "; ".join(benefits), "; ".join(tradeoffs)


# ---------------------------------------------------------------------------
# Write one detail file
# ---------------------------------------------------------------------------

def write_detail_file(sid: str, meta: dict, items: list, out_dir: Path):
    """
    Write references/meta_prompts/strategies/{sid}.md
    """
    lines = []
    sid_label = meta["id"]  # e.g. "P1"
    title = meta["title"]

    lines.append(f"# {sid_label}: {title}\n")

    # Overview: use the longest / most informative analysis
    if items:
        best = max(items, key=lambda it: len(it.get("analysis", "")))
        overview = best.get("analysis", "").strip()
        # Truncate very long overviews to ~600 chars but keep whole sentences
        if len(overview) > 700:
            truncated = overview[:700]
            last_period = max(truncated.rfind("。"), truncated.rfind("."))
            if last_period > 400:
                overview = overview[:last_period + 1]

        lines.append("## Overview\n")
        lines.append(overview + "\n")

    # When to use and trade-off (aggregated across all items)
    all_benefits = []
    all_tradeoffs = []
    seen_b: set[str] = set()
    seen_t: set[str] = set()
    for it in items:
        b = (it.get("benefit") or "").strip()
        t = (it.get("trade_off") or "").strip()
        if b and b not in seen_b:
            all_benefits.append(b)
            seen_b.add(b)
        if t and t not in seen_t:
            all_tradeoffs.append(t)
            seen_t.add(t)

    lines.append("\n## When to Use\n")
    lines.append(f"- {meta['when_to_apply']}\n")
    if all_benefits:
        for b in all_benefits[:3]:
            lines.append(f"- {b}\n")

    if all_tradeoffs:
        lines.append("\n## Trade-off\n")
        for t in all_tradeoffs[:3]:
            lines.append(f"- {t}\n")

    # Source operators
    sources = sorted({it.get("_source_op", "?") for it in items})
    lines.append(f"\n**Source operators**: {', '.join(sources)}\n")

    # Variants
    if not items:
        lines.append("\n*No examples found in the 43-operator corpus.*\n")
    else:
        variants = cluster_variants(items)
        if len(variants) == 1:
            # Single variant — no label needed
            var_items = variants[0]
            # pick representative (longest analysis)
            rep = max(var_items, key=lambda it: len(it.get("analysis", "")))
            ops = sorted({it.get("_source_op", "?") for it in var_items})

            lines.append("\n---\n")
            lines.append(f"\n## Implementation Pattern\n")
            lines.append(f"Source: {', '.join(ops)}\n")

            expert_code = _code_block(rep.get("expert_code_snippet", ""))
            lingxi_code = _code_block(rep.get("lingxi_code_snippet", ""))

            if expert_code:
                lines.append("\n**Expert implementation:**\n")
                lines.append(expert_code + "\n")
            if lingxi_code:
                lines.append("\n**vs. baseline (lingxi-code):**\n")
                lines.append(lingxi_code + "\n")
        else:
            # Multiple variants
            for v_idx, var_items in enumerate(variants):
                label = VARIANT_LABELS[v_idx] if v_idx < len(VARIANT_LABELS) else str(v_idx + 1)
                rep = max(var_items, key=lambda it: len(it.get("analysis", "")))
                ops = sorted({it.get("_source_op", "?") for it in var_items})
                name = rep.get("name", "")

                lines.append("\n---\n")
                lines.append(f"\n## Variant {label}: {name}\n")
                lines.append(f"Source: {', '.join(ops)}\n")

                # Brief analysis for this variant (if different enough from overview)
                variant_analysis = rep.get("analysis", "").strip()
                if variant_analysis and len(variant_analysis) > 100:
                    lines.append("\n" + variant_analysis[:500] + "\n")

                expert_code = _code_block(rep.get("expert_code_snippet", ""))
                lingxi_code = _code_block(rep.get("lingxi_code_snippet", ""))

                if expert_code:
                    lines.append("\n**Expert implementation:**\n")
                    lines.append(expert_code + "\n")
                if lingxi_code:
                    lines.append("\n**vs. baseline (lingxi-code):**\n")
                    lines.append(lingxi_code + "\n")

                benefit, tradeoff = _benefit_tradeoff(var_items)
                if benefit:
                    lines.append(f"\nBenefit: {benefit}\n")
                if tradeoff:
                    lines.append(f"Trade-off: {tradeoff}\n")

    out_path = out_dir / f"{sid}.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    logger.info(f"  Wrote {out_path.name} ({len(items)} items, {len(cluster_variants(items)) if items else 0} variants)")


# ---------------------------------------------------------------------------
# Write _unmatched.md
# ---------------------------------------------------------------------------

def write_unmatched_file(unmatched: list, out_dir: Path):
    if not unmatched:
        (out_dir / "_unmatched.md").write_text(
            "# Unmatched Items\n\nAll items were successfully matched.\n",
            encoding="utf-8",
        )
        return

    lines = [f"# Unmatched Items ({len(unmatched)} total)\n\n",
             "These items did not match any strategy group. Review and add keywords.\n\n"]

    by_cat: dict[str, list] = defaultdict(list)
    for it in unmatched:
        by_cat[it.get("_category", "unknown")].append(it)

    for cat, cat_items in sorted(by_cat.items()):
        lines.append(f"\n## {cat} ({len(cat_items)} items)\n\n")
        for it in cat_items:
            lines.append(f"- **{it.get('_source_op', '?')}** — {it.get('name', '?')}\n")

    (out_dir / "_unmatched.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"  Wrote _unmatched.md ({len(unmatched)} unmatched items)")


# ---------------------------------------------------------------------------
# Write strategy_index.md
# ---------------------------------------------------------------------------

DTYPE_META = [meta for _, meta in DTYPE_STRATEGIES]
PERF_META = [meta for _, meta in PERF_STRATEGIES]
ACC_META = [meta for _, meta in ACC_STRATEGIES]


def write_strategy_index(meta_prompts_dir: Path, item_counts: dict):
    lines = []
    lines.append("# AscendC Kernel Optimization Strategy Index\n\n")
    lines.append(
        "Source: 43 production-grade operators. "
        "Select all applicable strategies for your operator, "
        "then **Read each referenced detail file** before writing AscendC code.\n\n"
    )

    # ---- Multi-dtype section ----
    lines.append("## 多数据类型支持策略 (Multi-Dtype Support)\n\n")
    lines.append("| ID | Strategy | When to Apply | Sources | Detail File |\n")
    lines.append("|----|----------|---------------|---------|-------------|\n")
    for sid, meta in sorted(DTYPE_STRATEGIES, key=lambda x: x[1]["id_sort"]):
        count = item_counts.get(sid, 0)
        detail = f"strategies/{sid}.md"
        lines.append(
            f"| {meta['id']} | {meta['index_title']} "
            f"| {meta['when_to_apply']} | {count} ops | {detail} |\n"
        )
    lines.append("\n")

    # ---- Performance section ----
    lines.append("## 性能优化策略 (Performance Optimization)\n\n")
    lines.append("| ID | Strategy | When to Apply | Sources | Detail File |\n")
    lines.append("|----|----------|---------------|---------|-------------|\n")
    for sid, meta in sorted(PERF_STRATEGIES, key=lambda x: x[1]["id_sort"]):
        count = item_counts.get(sid, 0)
        detail = f"strategies/{sid}.md"
        lines.append(
            f"| {meta['id']} | {meta['index_title']} "
            f"| {meta['when_to_apply']} | {count} ops | {detail} |\n"
        )
    lines.append("\n")

    # ---- Accuracy section ----
    lines.append("## 精度优化策略 (Precision Optimization)\n\n")
    lines.append("| ID | Strategy | When to Apply | Sources | Detail File |\n")
    lines.append("|----|----------|---------------|---------|-------------|\n")
    for sid, meta in sorted(ACC_STRATEGIES, key=lambda x: x[1]["id_sort"]):
        count = item_counts.get(sid, 0)
        detail = f"strategies/{sid}.md"
        lines.append(
            f"| {meta['id']} | {meta['index_title']} "
            f"| {meta['when_to_apply']} | {count} ops | {detail} |\n"
        )
    lines.append("\n")

    # ---- Usage instructions ----
    lines.append("## How to Use\n\n")
    lines.append("1. Identify your operator's characteristics (dtype, shape, op type)\n")
    lines.append("2. Select ALL applicable strategy IDs from the tables above\n")
    lines.append("3. Read each referenced `strategies/XXX.md` detail file\n")
    lines.append("4. Apply the selected patterns in `tiling_pass`, `init_pass`, "
                 "`process_pass`, `process_nonaligned_pass`\n\n")
    lines.append("### Quick Reference by Operator Type\n\n")
    lines.append("| Operator Type | Recommended Strategies |\n")
    lines.append("|---------------|------------------------|\n")
    lines.append("| LayerNorm / RMSNorm | D1, D2, P1, P2, P3, A1, A2, A6 |\n")
    lines.append("| Quantization ops | D1, D4, D3, P2, P4, A3, A8 |\n")
    lines.append("| Element-wise (foreach) | D1, D2, P1, P2, A1, A4 |\n")
    lines.append("| Softmax / Attention | D1, P1, P5, A3, A4, A5 |\n")
    lines.append("| Pooling / Gather | D1, P1, P2, P11, A7 |\n")
    lines.append("| Optimizer ops | D1, P1, P2, P9, A1 |\n")
    lines.append("| Broadcast / Mask ops | D1, P1, P12, A5 |\n")
    lines.append("| Index / Scatter ops | D1, P1, P7, A7 |\n")
    lines.append("| Special / Complex ops | P13, D2, D3 |\n")

    out_path = meta_prompts_dir / "strategy_index.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    logger.info(f"\n  Wrote strategy_index.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    base_dir = Path(__file__).parent
    idea_pool_dir = base_dir / "IdeaPool"
    meta_prompts_dir = base_dir.parent / "references" / "meta_prompts"
    strategies_dir = meta_prompts_dir / "strategies"

    strategies_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect all items ----
    all_items: dict[str, list] = defaultdict(list)
    unmatched: list = []
    total_files = 0

    for json_file in sorted(idea_pool_dir.glob("*/expert_ideas.json")):
        total_files += 1
        op_name = json_file.parent.name
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        strategies = data.get("optimization_strategies", {})
        for category in ["multi_dtype_support", "performance", "accuracy"]:
            for item in strategies.get(category, []):
                item = dict(item)  # shallow copy
                item["_source_op"] = op_name
                sid = match_item(item, category)
                if sid:
                    all_items[sid].append(item)
                else:
                    item["_category"] = category
                    unmatched.append(item)

    logger.info(f"Processed {total_files} JSON files")
    logger.info(f"Total matched items: {sum(len(v) for v in all_items.values())}")
    logger.info(f"Total unmatched items: {len(unmatched)}")
    logger.info("")

    # ---- Write detail files ----
    logger.info("Writing detail files...")
    item_counts: dict[str, int] = {}
    for sid, meta in list(DTYPE_STRATEGIES) + list(PERF_STRATEGIES) + list(ACC_STRATEGIES):
        items = all_items.get(sid, [])
        # Count unique source operators
        item_counts[sid] = len({it.get("_source_op") for it in items})
        write_detail_file(sid, meta, items, strategies_dir)

    # ---- Write unmatched ----
    write_unmatched_file(unmatched, strategies_dir)

    # ---- Write strategy index ----
    write_strategy_index(meta_prompts_dir, item_counts)

    logger.info("\nDone! Summary:")
    logger.info(f"  Detail files: {len(ALL_STRATEGIES)}")
    logger.info(f"  strategy_index.md: {meta_prompts_dir / 'strategy_index.md'}")
    logger.info(f"  Unmatched items: {len(unmatched)}")
    if unmatched:
        logger.info(f"  Review: {strategies_dir / '_unmatched.md'}")


if __name__ == "__main__":
    main()
