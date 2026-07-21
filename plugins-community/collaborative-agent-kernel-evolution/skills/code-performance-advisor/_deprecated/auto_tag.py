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
Auto Tag Generator - 启发式 Fallback（TAG 阶段最后保障）

═══════════════════════════════════════════════════════════════
优先级说明（请严格遵守）:
  1. code_tag subskill（LLM 驱动，基于 code + profiling + evidence）← 首选
  2. auto_tag.py（关键词启发式，仅在 code_tag 无法执行时使用）← 本脚本
═══════════════════════════════════════════════════════════════

标签格式: 严格遵循 references/standards/tag_taxonony.md
  - Domain:  U.* (执行单元), O.* (算子族), T.* (数据类型)
  - Symptom: S.* (性能瓶颈)
  - Context: C.* (形状/布局/架构约束)

局限性:
  - U.* 优先从 profiling CSV 的 task_type 列读取；无 CSV 时依赖代码模式，准确率低
  - S.* 优先从 profiling 指标推断；无 CSV 时只保留极高置信度代码证据
  - 无 profiling → S.* 标签大概率缺失（会影响规则匹配质量）
  - 生成后须通过 hooks/post_tag_generation.py 验证

Usage:
    python auto_tag.py --op fastgelu
    python auto_tag.py --op aten__fused_adamw_ --profiling workspace/inputs/.../op_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from path_manager import PathManager

# ─── 1. O.* 算子族：代码关键词 → taxonomy 标签 ─────────────────────────────
# 规则：关键词出现即标注（高置信度模式优先）
O_KEYWORD_MAP: list[tuple[list[str], str]] = [
    # 精确算子族（高置信度）
    (["MatMul", "BatchMatMul", "Mmad", "mmad"], "O.MatMul"),
    (["Conv2d", "Conv", "Convolution", "DepthwiseConv"], "O.Conv"),
    (["MaxPool", "AvgPool", "Pooling", "AdaptivePool"], "O.Pooling"),
    (["LayerNorm", "RmsNorm", "BatchNorm", "GroupNorm"], "O.Norm"),
    (["AdamW", "adamw", "Adam", " adam"], "O.Optim"),
    (["Gelu", "FastGelu", "Relu", "Sigmoid", "Swish", "SwiGLU", "Tanh"], "O.Activation"),
    (["ReduceSum", "ReduceMax", "ReduceMin", "Reduce("], "O.Reduce"),
    (["Gather", "GatherV2", "ScatterNd", "IndexAdd", "IndexSelect"], "O.Index"),
    (["Softmax", "LogSoftmax"], "O.Attention"),
    (["DataCopy", "DataMove"], "O.DataCopy"),
    # 兜底：如果有明确的 element-wise 但没有更具体的 O.*
    (["Add(", "Mul(", "Div(", "Sub(", "Pow(", "Sqrt(", "Exp(", "Log("], "O.Elementwise"),
]

# ─── 2. T.* 数据类型：类型名关键词 ───────────────────────────────────────────
T_TYPE_MAP: list[tuple[list[str], str]] = [
    (["bfloat16", "BFloat16"], "T.BF16"),
    (["half", "float16", "Float16"], "T.FP16"),
    (["float32", "Float32"], "T.FP32"),
    (["double", "float64"], "T.FP64"),
    (["int8", "Int8"], "T.INT8"),
    (["uint8", "UInt8"], "T.UINT8"),
    (["int32", "Int32"], "T.INT32"),
    (["int64", "Int64"], "T.INT64"),
    (["bool", "Bool"], "T.BOOL"),
]
# float 容易误匹配（bfloat16 也含 float），单独处理
T_FLOAT_PLAIN = "T.FP32"

# ─── 3. U.* 执行单元：代码 fallback（仅在无 profiling 时使用）───────────────
# 有 Cube API → U.Cube；有 Vector/DMA API → U.Vector；两者都有 → U.Mix
CUBE_CODE_PATTERNS = ["MatMul", "Mmad", "BatchMatMul", "L1Tensor", "L0CTensor"]
VECTOR_CODE_PATTERNS = ["Pipe(", "LocalTensor", "EnQue", "DeQue", "DataCopy", "SetMaskCount"]

# ─── 4. C.* 上下文：代码结构线索 ─────────────────────────────────────────────
C_CODE_SIGNALS: list[tuple[list[str], str, str]] = [
    (["ubSize", "UB_SIZE", "UbSize", "C.UB.Capacity"], "C.UB.Capacity",
     "ubSize/UB_SIZE pattern detected in tiling or buffer calculation"),
    (["ALIGN_NUM * 256", "256B", "256b", "AlignDown(", "AlignUp(", "blockAlign"],
     "C.Align.256B",
     "256-byte alignment constraint found in code"),
    (["Ascend910B", "910B2", "910D", "910C"],
     None,  # handled dynamically below
     ""),
]

ARCH_MAP = {
    "Ascend910B2": "C.Arch.910B2",
    "910B2": "C.Arch.910B2",
    "Ascend910D": "C.Arch.910D",
    "910D": "C.Arch.910D",
    "Ascend910C": "C.Arch.910C",
    "910C": "C.Arch.910C",
    "Ascend910B": "C.Arch.910B",
    "910B": "C.Arch.910B",
}


# ─── Profiling CSV 解析 ────────────────────────────────────────────────────────

def _parse_profiling_csv(csv_path: Path) -> Optional[dict]:
    """
    从 op_summary CSV 中提取与标签相关的字段。
    返回 None 表示解析失败或文件不存在。
    """
    if not csv_path or not csv_path.exists():
        return None
    try:
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        # 取第一行（通常算子只有一条记录，或取 task duration 最大的）
        row = max(rows, key=lambda r: float(r.get("task_duration(us)", "0") or "0"))
        return {k.strip(): v.strip() for k, v in row.items()}
    except Exception as exc:
        logger.debug("failed to parse profiling CSV %s: %s", csv_path, exc)
        return None


def _find_profiling_csv(profiling_dir: Path) -> Optional[Path]:
    """
    在 workspace/inputs/{op}/profiling/ 中找 CSV：
      1. profiling/op_summary.csv（init_workspace 生成的 flat 格式）
      2. profiling/profiling_csv/*.csv（旧式嵌套格式）
    """
    if not profiling_dir.exists():
        return None
    flat = profiling_dir / "op_summary.csv"
    if flat.exists():
        return flat
    nested = list((profiling_dir / "profiling_csv").glob("*.csv"))
    if nested:
        return nested[0]
    # 兜底：任意 CSV
    any_csv = list(profiling_dir.glob("*.csv"))
    return any_csv[0] if any_csv else None


# ─── 代码读取 ──────────────────────────────────────────────────────────────────

def _read_code(code_dir: Path) -> str:
    all_text: list[str] = []
    for pattern in ("*.cpp", "*.h", "*.cuh"):
        for f in code_dir.rglob(pattern):
            try:
                all_text.append(f.read_text(errors="ignore"))
            except Exception as exc:
                logger.debug("failed to read code file %s: %s", f, exc)
    return "\n".join(all_text)


# ─── 主标注逻辑 ───────────────────────────────────────────────────────────────

def extract_tags(code_dir: Path, profiling_csv: Optional[Path]) -> dict:
    """
    返回符合 tag_taxonony.md 格式的标签 dict。
    """
    code = _read_code(code_dir) if code_dir.exists() else ""
    prof = _parse_profiling_csv(profiling_csv)

    domain_tags: list[str] = []
    symptom_tags: list[str] = []
    context_tags: list[str] = []
    evidence: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────
    # U.* — 执行单元（优先 profiling task_type）
    # ──────────────────────────────────────────────────────────
    task_type_raw = ""
    if prof:
        # CSV 列名可能是 "Task Type" / "task_type" / "TaskType"
        for col in ("Task Type", "task_type", "TaskType", "Op Type", "op_type"):
            v = prof.get(col, "")
            if v:
                task_type_raw = v.upper()
                break

    u_tag_added = False
    if task_type_raw:
        if "VECTOR" in task_type_raw or "AIV" in task_type_raw:
            domain_tags.append("U.Vector")
            evidence["U.Vector"] = f"profiling task_type={task_type_raw}"
            u_tag_added = True
        elif "MIX" in task_type_raw:
            domain_tags.append("U.Mix")
            evidence["U.Mix"] = f"profiling task_type={task_type_raw}"
            u_tag_added = True
        elif "CUBE" in task_type_raw or "AIC" in task_type_raw:
            domain_tags.append("U.Cube")
            evidence["U.Cube"] = f"profiling task_type={task_type_raw}"
            u_tag_added = True
        elif "CPU" in task_type_raw:
            domain_tags.append("U.CPU")
            evidence["U.CPU"] = f"profiling task_type={task_type_raw}"
            u_tag_added = True
        elif "DMA" in task_type_raw:
            domain_tags.append("U.DMA")
            evidence["U.DMA"] = f"profiling task_type={task_type_raw}"
            u_tag_added = True

    if not u_tag_added and code:
        # Fallback：从代码关键词推断（低置信度）
        has_cube = any(kw in code for kw in CUBE_CODE_PATTERNS)
        has_vector = any(kw in code for kw in VECTOR_CODE_PATTERNS)
        if has_cube and has_vector:
            domain_tags.append("U.Mix")
            evidence["U.Mix"] = "code heuristic: both Cube and Vector API patterns detected"
        elif has_cube:
            domain_tags.append("U.Cube")
            evidence["U.Cube"] = "code heuristic: Cube API patterns detected"
        elif has_vector:
            domain_tags.append("U.Vector")
            evidence["U.Vector"] = "code heuristic: Vector/DMA API patterns detected"
        # else: cannot determine, no U.* tag added (safer than guessing)

    # ──────────────────────────────────────────────────────────
    # O.* — 算子族（代码关键词）
    # ──────────────────────────────────────────────────────────
    o_found: set[str] = set()
    for keywords, tag in O_KEYWORD_MAP:
        matched = [kw for kw in keywords if kw in code]
        if matched:
            if tag not in o_found:
                # 避免被更具体的 O.* 覆盖（如 O.MatMul 优先于 O.Elementwise）
                # O.Elementwise 只在没有其他 O.* 时添加
                if tag == "O.Elementwise" and len(o_found) > 0:
                    continue
                o_found.add(tag)
                domain_tags.append(tag)
                evidence[tag] = f"keyword match: {matched[0]}"

    if not o_found:
        domain_tags.append("O.General")
        evidence["O.General"] = "no specific operator pattern detected"

    # ──────────────────────────────────────────────────────────
    # T.* — 数据类型（代码关键词）
    # ──────────────────────────────────────────────────────────
    for keywords, tag in T_TYPE_MAP:
        if any(kw in code for kw in keywords):
            domain_tags.append(tag)
            evidence[tag] = f"type keyword: {keywords[0]}"

    # 特殊处理：plain "float" 但排除 float16/bfloat16 误匹配
    has_plain_float = "float" in code
    no_float_dtype_tagged = (
        "T.FP32" not in domain_tags
        and "T.FP16" not in domain_tags
        and "T.BF16" not in domain_tags
    )
    if has_plain_float and no_float_dtype_tagged:
        domain_tags.append("T.FP32")
        evidence["T.FP32"] = "keyword: float (inferred)"

    # ──────────────────────────────────────────────────────────
    # S.* — 性能瓶颈（优先 profiling 指标）
    # ──────────────────────────────────────────────────────────
    if prof:
        def _pct(col: str) -> Optional[float]:
            v = prof.get(col, "")
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        aiv_scalar = _pct("aiv_scalar_ratio")
        aiv_vec = _pct("aiv_vec_ratio")
        aiv_mte2 = _pct("aiv_mte2_ratio")
        aiv_mte3 = _pct("aiv_mte3_ratio")
        aic_scalar = _pct("aic_scalar_ratio")
        aic_mte1 = _pct("aic_mte1_ratio")
        aic_mte2 = _pct("aic_mte2_ratio")

        # Scalar 瓶颈
        scalar_ratio = aiv_scalar if aiv_scalar is not None else aic_scalar
        if scalar_ratio is not None and scalar_ratio > 0.3:
            symptom_tags.append("S.ScalarBound")
            evidence["S.ScalarBound"] = f"scalar_ratio={scalar_ratio:.1%} > 30%"
            if scalar_ratio > 0.5:
                symptom_tags.append("S.HighScalarRatio")
                evidence["S.HighScalarRatio"] = f"scalar_ratio={scalar_ratio:.1%} > 50%"

        # 内存/搬运瓶颈（Vector）
        if aiv_mte2 is not None and aiv_mte3 is not None:
            if aiv_mte2 + aiv_mte3 > 0.5:
                symptom_tags.append("S.MemoryBound")
                symptom_tags.append("S.TransferDominated")
                evidence["S.MemoryBound"] = f"aiv_mte2+mte3={aiv_mte2+aiv_mte3:.1%} > 50%"
            elif aiv_mte2 + aiv_mte3 > 0.35:
                symptom_tags.append("S.MteBusy")
                evidence["S.MteBusy"] = f"aiv_mte2+mte3={aiv_mte2+aiv_mte3:.1%} > 35%"

        # 内存/搬运瓶颈（Cube）
        if aic_mte1 is not None and aic_mte2 is not None:
            if aic_mte1 + aic_mte2 > 0.4:
                symptom_tags.append("S.MemoryBound")
                evidence.setdefault("S.MemoryBound",
                                    f"aic_mte1+mte2={aic_mte1+aic_mte2:.1%} > 40%")

        # 低利用率
        if aiv_vec is not None and aiv_vec < 0.3:
            symptom_tags.append("S.LowVecUtil")
            evidence["S.LowVecUtil"] = f"aiv_vec_ratio={aiv_vec:.1%} < 30%"
        low_vec_util = aiv_vec is not None and aiv_vec < 0.5
        low_scalar_util = aiv_scalar is not None and aiv_scalar < 0.3
        if low_vec_util and low_scalar_util:
            symptom_tags.append("S.LowComputeUtil")
            evidence["S.LowComputeUtil"] = f"vec_ratio={aiv_vec:.1%} and scalar_ratio={aiv_scalar:.1%} both low"

    else:
        # 无 profiling → 只保留极高置信度的代码证据
        # SyncAll/SyncBarrier 出现 → 可能有流水停顿（保守标注）
        sync_patterns = ["SyncAll()", "SyncBarrier()", "pipe.SyncCalc()", "pipe.SyncVec()"]
        if any(p in code for p in sync_patterns):
            symptom_tags.append("S.PipeStall")
            matched_sync = next(p for p in sync_patterns if p in code)
            evidence["S.PipeStall"] = f"code: explicit sync pattern '{matched_sync}' detected"

        # scalar/amsgrad/maximize 分支 → ScalarBound（保守）
        scalar_code_patterns = ["amsgrad", "maximize", "scalar<float>", "scalar<half>"]
        if any(p in code for p in scalar_code_patterns):
            symptom_tags.append("S.ScalarBound")
            matched_scalar = next(p for p in scalar_code_patterns if p in code)
            evidence["S.ScalarBound"] = f"code: scalar branch pattern '{matched_scalar}'"

    # ──────────────────────────────────────────────────────────
    # C.* — 上下文（代码约束）
    # ──────────────────────────────────────────────────────────

    # UB 容量约束
    ub_patterns = ["ubSize", "UB_SIZE", "UbSize", "ub_size", "ubTileSize"]
    if any(p in code for p in ub_patterns):
        context_tags.append("C.UB.Capacity")
        evidence["C.UB.Capacity"] = f"tiling parameter pattern detected"

    # 对齐约束 256B
    align_patterns = ["256", "ALIGN_NUM", "AlignDown", "AlignUp", "blockAlign"]
    if any(p in code for p in align_patterns):
        context_tags.append("C.Align.256B")
        evidence["C.Align.256B"] = "256-byte alignment related pattern in code"

    # 架构标签
    for kw, arch_tag in ARCH_MAP.items():
        if kw in code:
            context_tags.append(arch_tag)
            evidence[arch_tag] = f"arch keyword '{kw}' in code"
            break  # 只标注第一个匹配的架构

    # ──────────────────────────────────────────────────────────
    # 去重 + 排序
    # ──────────────────────────────────────────────────────────
    domain_tags = sorted(set(domain_tags))
    symptom_tags = sorted(set(symptom_tags))
    context_tags = sorted(set(context_tags))

    return {
        "version": "1.0-heuristic",
        "op": {
            "name": "",  # 由调用方填充
            "task_type": task_type_raw or "unknown"
        },
        "inputs_used": {
            "code_kernel": str(code_dir / "op_kernel") if code_dir.exists() else "",
            "code_host": str(code_dir / "op_host") if code_dir.exists() else "",
            "profiling_csv": str(profiling_csv) if profiling_csv else ""
        },
        "missing_inputs": (
            [] if profiling_csv else ["profiling_csv"]
        ) + (
            [] if code_dir.exists() else ["code_dir"]
        ),
        "domain_tags": domain_tags,
        "symptom_tags": symptom_tags,
        "context_tags": context_tags,
        "evidence": evidence,
        "auto_generated": True,
        "confidence": "low" if not profiling_csv else "medium-low",
        "notes": [
            "Generated by auto_tag.py (heuristic fallback).",
            "code_tag subskill (LLM-driven) is strongly preferred for accurate tagging.",
            "To replace: run code_tag subskill, then resume workflow with --force-retag."
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Auto Tag Generator — heuristic fallback for TAG phase.\n"
            "Preferred: code_tag subskill (LLM-driven, evidence-based)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--op", required=True, help="Operator name (e.g. fastgelu)")
    parser.add_argument("--profiling", default="", help="Override profiling CSV path")
    parser.add_argument("--output", default="", help="Override output tag file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    pm = PathManager(ROOT)
    code_dir = pm.get_input_code_dir(args.op)
    tag_file = Path(args.output) if args.output else pm.get_tag_file(args.op)

    # 确定 profiling CSV 路径
    # Bug 3 fix: 同时兼容 profiling/ 和 profiling_data/ 两种目录名
    if args.profiling:
        profiling_csv = Path(args.profiling)
    else:
        prof_dir = pm.get_input_profiling_dir(args.op)
        profiling_csv = _find_profiling_csv(prof_dir)
        if profiling_csv is None:
            # 尝试 profiling_data/ 目录（手动放置的原始格式）
            prof_dir_alt = pm.get_input_dir(args.op) / "profiling_data"
            profiling_csv = _find_profiling_csv(prof_dir_alt)

    logger.info(f"[AUTO_TAG] Operator   : {args.op}")
    logger.info(f"[AUTO_TAG] Code dir   : {code_dir}  (exists={code_dir.exists()})")
    logger.info(
        f"[AUTO_TAG] Profiling  : {profiling_csv}  (exists={profiling_csv.exists() if profiling_csv else False})")

    if not code_dir.exists():
        logger.info(f"[AUTO_TAG] ❌ Code directory not found. Run init_workspace.py --op {args.op} first.")
        sys.exit(1)

    tags = extract_tags(code_dir, profiling_csv)
    tags["op"]["name"] = args.op

    if not profiling_csv:
        logger.info(f"[AUTO_TAG] ⚠️  No profiling CSV — S.* symptom tags may be absent or inaccurate")

    # 写入文件
    tag_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tag_file, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, ensure_ascii=False)

    logger.info(f"[AUTO_TAG] ✅ Tag file : {tag_file}")
    logger.info(f"[AUTO_TAG]    domain_tags  ({len(tags['domain_tags'])}): {tags['domain_tags']}")
    logger.info(f"[AUTO_TAG]    symptom_tags ({len(tags['symptom_tags'])}): {tags['symptom_tags']}")
    logger.info(f"[AUTO_TAG]    context_tags ({len(tags['context_tags'])}): {tags['context_tags']}")
    logger.info(f"[AUTO_TAG] ⚠️  These are heuristic tags. "
                f"Run code_tag subskill + --force-retag for LLM-quality analysis.")


if __name__ == "__main__":
    main()
