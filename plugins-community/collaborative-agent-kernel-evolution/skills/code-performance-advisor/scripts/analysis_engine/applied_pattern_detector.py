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

"""应用模式检测器（默认采用 V2 的保守策略）

目标:
  识别 baseline 代码中“已经应用”的优化模式，用于在规则评分阶段过滤冗余规则。

设计原则（保守优先）:
  - 宁可漏检（false negative），也不误判已应用（false positive）。
  - 因为误判会导致把有效规则强降权（score × 0.05），代价更高。

实现要点:
  - 只启用具备独特签名的模式；对易误判模式默认禁用。
  - 只有置信度达到阈值（>= 0.9 / 0.95）才认为“已应用”。
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class PatternSignature:
    """优化模式的代码特征签名"""
    pattern_id: str
    name: str
    required: List[str]  # AND: 必须全部匹配
    typical: List[str]  # 加分项
    confidence_threshold: float
    enabled: bool = True
    description: str = ""


class AppliedPatternDetectorV2:
    """已应用优化模式检测器 V2（默认实现）。"""

    # ==================== Tier 1: 高置信度模式（可安全检测）====================
    PATTERNS: Dict[str, PatternSignature] = {
        "COUNTER_MODE": PatternSignature(
            pattern_id="COUNTER_MODE",
            name="Counter Mode Vectorization",
            required=[
                r"SetMaskCount\s*\(",
                r"MaskMode::COUNTER",
            ],
            typical=[
                r"MASK_PLACEHOLDER",
                r"ResetMask\s*\(",
            ],
            confidence_threshold=0.9,
            enabled=True,
            description="使用 Counter Mode 硬件加速，避免显式循环",
        ),

        "BUFFER_FUSION": PatternSignature(
            pattern_id="BUFFER_FUSION",
            name="Buffer Reuse/Fusion",
            required=[
                r"//.*reuse|//.*fusion|//.*merge",
                r"LocalTensor.*=.*Get<",
            ],
            typical=[
                r"shared\d+|tmp\d+",
            ],
            confidence_threshold=0.95,
            enabled=True,
            description="Buffer 复用或融合以减少内存占用",
        ),

        # ==================== Tier 2: 中等置信度（谨慎检测）====================
        "DOUBLE_BUFFER": PatternSignature(
            pattern_id="DOUBLE_BUFFER",
            name="Double Buffering",
            required=[
                r"InitBuffer.*?,\s*2\s*,",
                r"EnQue|DeQue",
            ],
            typical=[
                r"ping.*pong|double.*buffer",
                r"bufferIndex\s*[\^\=]",
            ],
            confidence_threshold=0.95,
            enabled=True,
            description="双缓冲重叠计算和 IO",
        ),

        "EXPLICIT_SYNC": PatternSignature(
            pattern_id="EXPLICIT_SYNC",
            name="Explicit Synchronization",
            required=[
                r"pipe\.SyncCalc",
                r"WaitEvent",
            ],
            typical=[
                r"Event\s+\w+",
                r"pipe\.barrier",
            ],
            confidence_threshold=0.9,
            enabled=True,
            description="显式控制计算和数据同步",
        ),

        "WELFORD_ALGORITHM": PatternSignature(
            pattern_id="WELFORD_ALGORITHM",
            name="Welford Online Algorithm",
            required=[
                r"\bM2\b",
                r"WelfordParallelUpdate|welford",
            ],
            typical=[
                r"count\s*\+=\s*1",
                r"delta.*mean|mean.*delta",
            ],
            confidence_threshold=0.9,
            enabled=True,
            description="Welford 在线算法：单次遍历计算均值和方差（Norm 类算子）",
        ),

        # ==================== Tier 3: 低置信度（默认禁用）====================
        "ASYNC_PIPELINE": PatternSignature(
            pattern_id="ASYNC_PIPELINE",
            name="Async 3-Stage Pipeline",
            required=[r"__ASYNC_PIPELINE_MARKER_NEVER_MATCH__"],
            typical=[],
            confidence_threshold=1.0,
            enabled=False,
            description="异步 3 级流水线（检测难度高，默认禁用）",
        ),
        "ITERATE_ONCE": PatternSignature(
            pattern_id="ITERATE_ONCE",
            name="Single Iteration Processing",
            required=[r"__ITERATE_ONCE_MARKER_NEVER_MATCH__"],
            typical=[],
            confidence_threshold=1.0,
            enabled=False,
            description="一次性处理所有数据（检测难度高，默认禁用）",
        ),
    }

    # 规则 ID → Pattern ID 映射（维护在此处，与 PATTERNS 定义同位，新增模式时一并更新）
    RULE_PATTERN_MAP: Dict[str, str] = {
        "R_COUNTER_MODE_VECTORIZATION": "COUNTER_MODE",
        "R_API_VECTOR_COUNTER_MODE": "COUNTER_MODE",
        "R_SYNC_EXPLICIT_PIPELINE": "EXPLICIT_SYNC",
        "R_PIPE_ITERATE_ASYNC": "ASYNC_PIPELINE",
        "R_PIPE_DOUBLE_BUFFER": "DOUBLE_BUFFER",
        "R_DOUBLE_BUFFER": "DOUBLE_BUFFER",
        "R_BUFFER_FUSION": "BUFFER_FUSION",
        "R_MEM_UB_FUSION": "BUFFER_FUSION",
        "R_ITERATE_ONCE": "ITERATE_ONCE",
        "R_NORM_WELFORD_ALGORITHM": "WELFORD_ALGORITHM",
    }

    def detect(self, code: str) -> List[Dict]:
        """检测代码中已应用的优化模式（返回已达阈值的模式）。"""
        detected: List[Dict] = []

        for pattern_id, signature in self.PATTERNS.items():
            if not signature.enabled:
                continue

            confidence, evidence = self._match_pattern(code, signature)
            if confidence >= signature.confidence_threshold:
                detected.append(
                    {
                        "pattern_id": pattern_id,
                        "pattern_name": signature.name,
                        "confidence": round(confidence, 3),
                        "evidence": evidence,
                        "description": signature.description,
                        "threshold": signature.confidence_threshold,
                    }
                )

        return detected

    def detect_from_file(self, file_path: Union[str, Path]) -> List[Dict]:
        """从文件读取代码并检测。"""
        path = Path(file_path)
        code = path.read_text(encoding="utf-8", errors="ignore")
        return self.detect(code)

    def detect_from_files(self, file_paths: List[Union[str, Path]]) -> List[Dict]:
        """从多个文件读取并检测（把多个文件内容拼接后统一检测）。"""
        parts: List[str] = []
        for p in file_paths:
            path = Path(p)
            if not path.exists() or not path.is_file():
                continue
            parts.append(f"\n// ===== FILE: {path.name} =====\n")
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        return self.detect("\n".join(parts))

    @staticmethod
    def _match_pattern(code: str, signature: PatternSignature) -> Tuple[float, List[str]]:
        evidence: List[str] = []

        # ========== 必需特征（AND）==========
        for pattern in signature.required:
            match = re.search(pattern, code, re.MULTILINE | re.IGNORECASE)
            if match:
                snippet = match.group(0)[:50]
                evidence.append(f"✓ Required: {snippet}")
            else:
                evidence.append(f"✗ Missing required: {pattern[:60]}")
                return (0.0, evidence)

        # 所有必需特征都匹配，基础置信度从 0.7 起步
        base_confidence = 0.7

        # ========== 典型特征（加分项）==========
        typical_matched = 0
        for pattern in signature.typical:
            match = re.search(pattern, code, re.MULTILINE | re.IGNORECASE)
            if match:
                typical_matched += 1
                snippet = match.group(0)[:50]
                evidence.append(f"+ Typical: {snippet}")

        if signature.typical:
            typical_bonus = min(0.3, (typical_matched / len(signature.typical)) * 0.3)
            base_confidence += typical_bonus

        return (base_confidence, evidence)


class AppliedPatternDetector(AppliedPatternDetectorV2):
    """兼容性包装器：保持旧 import 与类名不变，默认使用 V2 策略。"""

    def detect(self, code: str) -> List[Dict]:
        results = super().detect(code)
        # 形式兼容：历史上调用方常用 confidence>0.7 过滤。
        return [r for r in results if r.get("confidence", 0.0) > 0.7]


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Test pattern detection on a file")
    parser.add_argument("file", type=Path, help="Path to AscendC .cpp file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed evidence")
    args = parser.parse_args()

    if not args.file.exists():
        logger.info(f"Error: File not found: {args.file}")
        return 1

    detector = AppliedPatternDetectorV2()
    code = args.file.read_text(encoding="utf-8", errors="ignore")
    results = detector.detect(code)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pattern Detection Results: {args.file.name}")
    logger.info(f"{'='*60}\n")

    if not results:
        logger.info("✓ No patterns detected (no redundancy filtering needed)")
        logger.info("  All optimization rules will be considered.")
    else:
        logger.info(f"✓ Detected {len(results)} applied pattern(s):\n")
        for i, result in enumerate(results, 1):
            logger.info(f"[{i}] {result['pattern_name']}")
            logger.info(
                f"    Confidence: {result['confidence']:.1%} "
                f"(threshold: {result.get('threshold', 0.0):.1%})"
            )
            logger.info(f"    Description: {result.get('description', '')}")
            if args.verbose and result.get("evidence"):
                logger.info("    Evidence:")
                for ev in result["evidence"]:
                    logger.info(f"      {ev}")
            logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
