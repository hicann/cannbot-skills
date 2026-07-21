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
Evidence Extractor for Code Performance Advisor

Extracts quantitative evidence from profiling CSV data to support optimization suggestions.
"""

from __future__ import annotations

import csv
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ProfilingMetrics:
    """Profiling metrics for an operator."""
    operator_name: str
    task_type: str

    # Duration metrics
    task_duration_us: float
    task_wait_time_us: float

    # AICore metrics (AI_CORE task type)
    aicore_time_us: Optional[float] = None
    aic_mac_ratio: Optional[float] = None
    aic_scalar_ratio: Optional[float] = None
    aic_scalar_time_us: Optional[float] = None

    # AIVector metrics (AI_VECTOR_CORE or MIX_AIV)
    aiv_time_us: Optional[float] = None
    aiv_vec_ratio: Optional[float] = None
    aiv_vec_time_us: Optional[float] = None
    aiv_scalar_ratio: Optional[float] = None
    aiv_scalar_time_us: Optional[float] = None

    # Raw data
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def bottleneck_type(self) -> str:
        """
        Identify the primary bottleneck type.

        Returns:
            'scalar', 'vector_underutilized', 'memory', or 'balanced'
        """
        if self.task_type == 'AI_VECTOR_CORE':
            if self.aiv_scalar_ratio and self.aiv_scalar_ratio > 0.4:
                return 'scalar'
            elif self.aiv_vec_ratio and self.aiv_vec_ratio < 0.3:
                return 'vector_underutilized'
            else:
                return 'balanced'
        elif self.task_type == 'AI_CORE':
            if self.aic_scalar_ratio and self.aic_scalar_ratio > 0.4:
                return 'scalar'
            else:
                return 'balanced'
        else:
            return 'unknown'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operator_name': self.operator_name,
            'task_type': self.task_type,
            'task_duration_us': self.task_duration_us,
            'bottleneck_type': self.bottleneck_type,
            'aiv_vec_ratio': self.aiv_vec_ratio,
            'aiv_scalar_ratio': self.aiv_scalar_ratio,
            'aiv_vec_time_us': self.aiv_vec_time_us,
            'aiv_scalar_time_us': self.aiv_scalar_time_us,
        }


@dataclass
class Evidence:
    """Evidence extracted from profiling data."""
    operator_name: str
    metrics: ProfilingMetrics

    # Identified issues
    bottleneck_type: str
    thresholds_violated: List[str] = field(default_factory=list)
    key_observations: List[str] = field(default_factory=list)

    # Comparative data (if available)
    baseline_metrics: Optional[ProfilingMetrics] = None
    improvement_delta: Optional[float] = None

    def to_markdown(self) -> str:
        """
        Render evidence as markdown.

        Returns:
            Markdown formatted string
        """
        lines = []

        lines.append(f"**Current Performance** (from profiling CSV):")
        lines.append(f"- `Task Duration(us)`: {self.metrics.task_duration_us:.2f}us")
        lines.append(f"- `Task Type`: {self.metrics.task_type}")

        if self.metrics.task_type in ['AI_VECTOR_CORE', 'MIX_AIV']:
            if self.metrics.aiv_vec_ratio is not None:
                lines.append(f"- `aiv_vec_ratio`: {self.metrics.aiv_vec_ratio:.3f}")
            if self.metrics.aiv_scalar_ratio is not None:
                lines.append(f"- `aiv_scalar_ratio`: {self.metrics.aiv_scalar_ratio:.3f}")
            if self.metrics.aiv_vec_time_us is not None:
                lines.append(f"- `aiv_vec_time(us)`: {self.metrics.aiv_vec_time_us:.3f}us")
            if self.metrics.aiv_scalar_time_us is not None:
                lines.append(f"- `aiv_scalar_time(us)`: {self.metrics.aiv_scalar_time_us:.3f}us")

        lines.append(f"\n**Bottleneck Type**: {self.bottleneck_type}")

        if self.thresholds_violated:
            lines.append(f"\n**Thresholds Violated**:")
            for violation in self.thresholds_violated:
                lines.append(f"- {violation}")

        if self.key_observations:
            lines.append(f"\n**Key Observations**:")
            for obs in self.key_observations:
                lines.append(f"- {obs}")

        if self.baseline_metrics and self.improvement_delta is not None:
            lines.append(f"\n**Improvement vs Baseline**:")
            lines.append(f"- Performance delta: {self.improvement_delta:+.1%}")
            lines.append(f"- Baseline duration: {self.baseline_metrics.task_duration_us:.2f}us")
            lines.append(f"- Current duration: {self.metrics.task_duration_us:.2f}us")

        return '\n'.join(lines)


class EvidenceExtractor:
    """Extract evidence from profiling CSV files."""

    def __init__(self, csv_path: Path):
        """
        Initialize extractor with CSV file.

        Args:
            csv_path: Path to profiling CSV file (op_summary*.csv)
        """
        self.csv_path = csv_path
        self._data = None

    def load(self) -> None:
        """Load CSV data into memory."""
        if self._data is not None:
            return  # Already loaded

        self._data = []

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._data.append(row)

    def extract_for_operator(self, operator_name: str) -> Optional[ProfilingMetrics]:
        """
        Extract metrics for a specific operator.

        Args:
            operator_name: Name of the operator

        Returns:
            ProfilingMetrics object or None if not found
        """
        self.load()

        # Find row for operator (with fuzzy matching)
        row = None
        actual_op_name = None

        # Strategy 1: Exact match
        for r in self._data:
            if r.get('Op Name') == operator_name:
                row = r
                actual_op_name = r.get('Op Name')
                break

        # Strategy 2: Case-insensitive match
        if not row:
            for r in self._data:
                op_name_lower = r.get('Op Name', '').lower()
                if op_name_lower == operator_name.lower():
                    row = r
                    actual_op_name = r.get('Op Name')
                    break

        # Strategy 3: Partial match (operator_name is substring of Op Name)
        if not row:
            for r in self._data:
                op_name_lower = r.get('Op Name', '').lower()
                if operator_name.lower() in op_name_lower:
                    row = r
                    actual_op_name = r.get('Op Name')
                    break

        if not row:
            return None

        # Parse metrics (use actual operator name from CSV)
        def safe_float(value: str) -> Optional[float]:
            """Safely parse float from CSV."""
            if not value or value == 'N/A':
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        metrics = ProfilingMetrics(
            operator_name=actual_op_name or operator_name,  # Use actual name from CSV
            task_type=row.get('Task Type', ''),
            task_duration_us=safe_float(row.get('Task Duration(us)')) or 0.0,
            task_wait_time_us=safe_float(row.get('Task Wait Time(us)')) or 0.0,
            aicore_time_us=safe_float(row.get('aicore_time(us)')),
            aic_mac_ratio=safe_float(row.get('aic_mac_ratio')),
            aic_scalar_ratio=safe_float(row.get('aic_scalar_ratio')),
            aic_scalar_time_us=safe_float(row.get('aic_scalar_time(us)')),
            aiv_time_us=safe_float(row.get('aiv_time(us)')),
            aiv_vec_ratio=safe_float(row.get('aiv_vec_ratio')),
            aiv_vec_time_us=safe_float(row.get('aiv_vec_time(us)')),
            aiv_scalar_ratio=safe_float(row.get('aiv_scalar_ratio')),
            aiv_scalar_time_us=safe_float(row.get('aiv_scalar_time(us)')),
            raw_data=row
        )

        return metrics

    def extract_evidence(
        self,
        operator_name: str,
        matched_tags: Optional[List[str]] = None
    ) -> Optional[Evidence]:
        """
        Extract evidence for an operator with context.

        Args:
            operator_name: Operator name
            matched_tags: Tags matched by rule (for context)

        Returns:
            Evidence object or None if operator not found
        """
        metrics = self.extract_for_operator(operator_name)

        if not metrics:
            return None

        # Identify bottleneck
        bottleneck = metrics.bottleneck_type

        # Check thresholds
        thresholds_violated = []
        key_observations = []

        if metrics.task_type in ['AI_VECTOR_CORE', 'MIX_AIV']:
            # Vector utilization threshold
            if metrics.aiv_vec_ratio is not None:
                if metrics.aiv_vec_ratio < 0.3:
                    thresholds_violated.append(
                        f"aiv_vec_ratio={metrics.aiv_vec_ratio:.3f} < 0.30 (target: >0.50)"
                    )
                    key_observations.append(
                        "Low vector utilization indicates inefficient use of vector instructions"
                    )

            # Scalar overhead threshold
            if metrics.aiv_scalar_ratio is not None:
                if metrics.aiv_scalar_ratio > 0.4:
                    thresholds_violated.append(
                        f"aiv_scalar_ratio={metrics.aiv_scalar_ratio:.3f} > 0.40 (target: <0.20)"
                    )
                    key_observations.append(
                        "High scalar ratio indicates excessive control overhead"
                    )

            # Scalar vs vector time comparison
            if metrics.aiv_scalar_time_us and metrics.aiv_vec_time_us:
                ratio = metrics.aiv_scalar_time_us / metrics.aiv_vec_time_us
                if ratio > 2.0:
                    key_observations.append(
                        f"Scalar time ({metrics.aiv_scalar_time_us:.2f}us) is {ratio:.1f}x vector time ({metrics.aiv_vec_time_us:.2f}us)"
                    )

        evidence = Evidence(
            operator_name=operator_name,
            metrics=metrics,
            bottleneck_type=bottleneck,
            thresholds_violated=thresholds_violated,
            key_observations=key_observations
        )

        return evidence

    def compare_with_baseline(
        self,
        operator_name: str,
        baseline_csv: Path
    ) -> Optional[Evidence]:
        """
        Extract evidence with baseline comparison.

        Args:
            operator_name: Operator name
            baseline_csv: Path to baseline CSV

        Returns:
            Evidence with baseline comparison
        """
        # Current metrics
        evidence = self.extract_evidence(operator_name)

        if not evidence:
            return None

        # Baseline metrics
        baseline_extractor = EvidenceExtractor(baseline_csv)
        baseline_metrics = baseline_extractor.extract_for_operator(operator_name)

        if baseline_metrics:
            evidence.baseline_metrics = baseline_metrics

            # Calculate improvement
            if baseline_metrics.task_duration_us > 0:
                delta = (baseline_metrics.task_duration_us - evidence.metrics.task_duration_us) / \
                    baseline_metrics.task_duration_us
                evidence.improvement_delta = delta

        return evidence


# Utility functions

def extract_evidence(
    csv_path: Path,
    operator_name: str,
    matched_tags: Optional[List[str]] = None
) -> Optional[Evidence]:
    """
    Convenience function to extract evidence.

    Args:
        csv_path: Path to profiling CSV
        operator_name: Operator name
        matched_tags: Matched tags for context

    Returns:
        Evidence object or None
    """
    extractor = EvidenceExtractor(csv_path)
    return extractor.extract_evidence(operator_name, matched_tags)


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if len(sys.argv) < 3:
        logger.info("Usage: python evidence_extractor.py <csv_file> <operator_name>")
        logger.info("Example: python evidence_extractor.py profiling.csv FastgeluCustom")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    op_name = sys.argv[2]

    if not csv_file.exists():
        logger.info(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    logger.info(f"Extracting evidence for: {op_name}")
    logger.info(f"From: {csv_file}")
    logger.info("=" * 60)

    try:
        evidence = extract_evidence(csv_file, op_name)

        if not evidence:
            logger.info(f"\n❌ Operator '{op_name}' not found in CSV")
            sys.exit(1)

        logger.info(f"\n✅ Evidence extracted successfully!\n")
        logger.info(evidence.to_markdown())

        logger.info(f"\n📊 Summary:")
        logger.info(f"   Bottleneck: {evidence.bottleneck_type}")
        logger.info(f"   Violations: {len(evidence.thresholds_violated)}")
        logger.info(f"   Observations: {len(evidence.key_observations)}")

    except Exception as e:
        logger.info(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
