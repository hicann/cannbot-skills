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
Goal Configuration Loader for Code Performance Advisor

This module loads performance optimization goals from goal.md files
and provides defaults when goals are not configured.
"""

import logging
import sys

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GoalConfig:
    """Performance optimization goal configuration."""

    # Relative improvement threshold (0-1 scale, e.g., 0.2 = 20%)
    relative_improvement: float = 0.2

    # Absolute metric targets (optional)
    absolute_metrics: Dict[str, Any] = field(default_factory=dict)

    # Stop conditions for iterative optimization
    stop_conditions: Dict[str, int] = field(default_factory=lambda: {
        "max_iterations": 5,
        "consecutive_failures": 2
    })

    # Optional: custom notes from user
    notes: str = ""

    @property
    def improvement_percentage(self) -> str:
        """Return relative improvement as percentage string."""
        return f"{self.relative_improvement * 100:.0f}%"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoalConfig':
        """
        Create GoalConfig from dictionary.

        Args:
            data: Dictionary with goal configuration

        Returns:
            GoalConfig object
        """
        return cls(
            relative_improvement=data.get("relative_improvement", 0.2),
            absolute_metrics=data.get("absolute_metrics", {}),
            stop_conditions=data.get("stop_conditions", {
                "max_iterations": 5,
                "consecutive_failures": 2
            }),
            notes=data.get("notes", "")
        )

    def validate(self) -> bool:
        """Validate goal configuration values."""
        if not (0 <= self.relative_improvement <= 1):
            raise ValueError(
                f"relative_improvement must be in [0, 1], got {self.relative_improvement}"
            )

        if self.stop_conditions.get("max_iterations", 0) < 1:
            raise ValueError("max_iterations must be >= 1")

        if self.stop_conditions.get("consecutive_failures", 0) < 1:
            raise ValueError("consecutive_failures must be >= 1")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "relative_improvement": self.relative_improvement,
            "absolute_metrics": self.absolute_metrics,
            "stop_conditions": self.stop_conditions,
            "notes": self.notes
        }


def load_goal(op_dir: Path) -> GoalConfig:
    """
    Load performance goal configuration from goal.md file.

    Args:
        op_dir: Path to operator directory (e.g., workspace/inputs/fastgelu/)

    Returns:
        GoalConfig object with loaded or default values

    Example:
        >>> goal = load_goal(Path("workspace/inputs/fastgelu"))
        >>> print(goal.improvement_percentage)  # "20%"
    """
    goal_file = op_dir / "roofline" / "goal.md"

    # If goal file doesn't exist or is empty, return defaults
    if not goal_file.exists():
        return GoalConfig()

    if yaml is None:
        # Keep behavior safe and predictable in minimal environments.
        return GoalConfig()

    try:
        content = goal_file.read_text(encoding="utf-8")

        # Parse YAML front matter if exists (between --- markers)
        if content.strip().startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                if yaml_content:
                    data = yaml.safe_load(yaml_content)
                    if data:
                        return _parse_goal_data(data)

        # Try parsing the entire file as YAML
        data = yaml.safe_load(content)
        if data and isinstance(data, dict):
            return _parse_goal_data(data)

        # File exists but has no valid YAML, return defaults
        return GoalConfig()

    except Exception as e:
        # On parse error, return defaults and log warning
        logger.info(f"⚠️  Warning: Failed to parse goal.md: {e}")
        logger.info(f"   Using default goal configuration")
        return GoalConfig()


def _parse_goal_data(data: Dict[str, Any]) -> GoalConfig:
    """Parse goal configuration from dictionary."""
    goal = GoalConfig()

    # Load relative improvement
    if "relative_improvement" in data:
        val = data["relative_improvement"]
        # Handle percentage strings like "20%" or "0.2"
        if isinstance(val, str) and val.endswith("%"):
            goal.relative_improvement = float(val.rstrip("%")) / 100
        else:
            goal.relative_improvement = float(val)

    # Load absolute metrics
    if "absolute_metrics" in data and isinstance(data["absolute_metrics"], dict):
        goal.absolute_metrics = data["absolute_metrics"]

    # Load stop conditions
    if "stop_conditions" in data and isinstance(data["stop_conditions"], dict):
        goal.stop_conditions.update(data["stop_conditions"])

    # Load notes
    if "notes" in data:
        goal.notes = str(data["notes"])

    # Validate before returning
    goal.validate()

    return goal


def create_default_goal_md(op_dir: Path, op_name: str) -> Path:
    """
    Create a default goal.md template file.

    Args:
        op_dir: Path to operator directory
        op_name: Operator name (for template customization)

    Returns:
        Path to created goal.md file
    """
    goal_file = op_dir / "roofline" / "goal.md"
    goal_file.parent.mkdir(parents=True, exist_ok=True)

    template = f"""---
# Performance Optimization Goal Configuration
# Goal file auto-generated for operator {op_name}

# Target relative performance improvement (20% = 0.2)
relative_improvement: 0.2

# Optional: absolute metric targets.
# Add entries under absolute_metrics as "metric_name: comparison" pairs,
# e.g. task_duration_us with a "< 5.0" target (microseconds),
# aiv_vec_ratio with a "> 0.50" target (vector utilization ratio),
# or memory_bandwidth_util with a "> 0.7" target (bandwidth utilization ratio).
absolute_metrics:

# Stop conditions for iterative optimization
stop_conditions:
  max_iterations: 5          # Maximum optimization rounds
  consecutive_failures: 2    # Stop after N failed attempts
---

## Performance Goal

### Current Bottleneck
[Describe the current performance bottleneck, e.g., "Low vector utilization (13%)"]

### Optimization Target
- **Primary Goal**: Improve performance by {"> 20%"}
- **Secondary Goals**: [e.g., "Increase vector ratio to > 50%"]

### Constraints
- Must maintain numerical accuracy
- [Add any other constraints]

### Notes
[Additional context or requirements]
"""

    goal_file.write_text(template, encoding="utf-8")
    return goal_file


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if len(sys.argv) > 1:
        op_dir = Path(sys.argv[1])
        logger.info(f"Loading goal from: {op_dir}")

        goal = load_goal(op_dir)
        logger.info(f"\n✅ Goal Configuration:")
        logger.info(f"   Relative improvement: {goal.improvement_percentage}")
        logger.info(f"   Max iterations: {goal.stop_conditions['max_iterations']}")
        logger.info(f"   Stop after failures: {goal.stop_conditions['consecutive_failures']}")

        if goal.absolute_metrics:
            logger.info(f"   Absolute metrics: {goal.absolute_metrics}")
    else:
        logger.info("Usage: python goal_loader.py <op_dir>")
        logger.info("Example: python goal_loader.py workspace/inputs/fastgelu")
