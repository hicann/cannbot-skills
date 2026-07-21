# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

"""Metric comparison and constraint checking.

Pure data-shape and arithmetic logic — no I/O, no subprocess, no YAML.
The `EvalResult` dataclass is the contract eval_runner writes into;
downstream consumers (keep_or_discard, baseline_init, dashboard) read
from it.

What lives here:
  - `EvalOutcome`          — classification enum, single source of truth for
                             what happened (OK / kernel fail / infra fail).
  - `EvalResult`           — the result dataclass.
  - `is_improvement`       — current-vs-best comparison with relative-%
                             threshold and direction (`lower_is_better`).
  - `check_constraints`    — hard-constraint check
                             ({metric: (op_str, threshold)} →
                              list of violation strings).

Why a separate module: the comparison logic is the only piece of
task_config that has zero external dependencies and zero side effects;
splitting it out lets every other module that needs only EvalResult
import from here without dragging in YAML / urllib / tarfile.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvalOutcome(str, Enum):
    """What happened in eval. KERNEL_FAIL = agent can fix via PLAN→EDIT;
    INFRA_FAIL = operator only (broken --ref, missing env, transport down).
    """
    OK = "ok"
    KERNEL_FAIL = "kernel_fail"
    INFRA_FAIL = "infra_fail"


@dataclass
class EvalResult:
    outcome: EvalOutcome = EvalOutcome.INFRA_FAIL
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    raw_output: str = ""
    # "ref" → broken --ref file (the only sub-flavor of INFRA_FAIL the
    # downstream messages distinguish). None on success or other failures.
    error_source: Optional[str] = None
    # Path to the on-disk FAIL report (full per-case + complete log) the agent
    # opens with its file reader, instead of a truncated stdout dump.
    fail_report: Optional[str] = None
    # failure_extractor signals, already parsed by eval_bridge from the FULL log;
    # forwarded so pipeline doesn't re-parse a truncated tail.
    failure_signals: dict = field(default_factory=dict)

    @property
    def correctness(self) -> bool:
        return self.outcome == EvalOutcome.OK


ConstraintMap = dict[str, tuple[str, float]]
MetricName = str
DEFAULT_PRIMARY_METRIC: MetricName = "latency_ms"
DEFAULT_LOWER_IS_BETTER = True
DEFAULT_IMPROVEMENT_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Constraint check
# ---------------------------------------------------------------------------

_COMPARISONS = {
    "<=": lambda actual, limit: actual <= limit,
    ">=": lambda actual, limit: actual >= limit,
    "<": lambda actual, limit: actual < limit,
    ">": lambda actual, limit: actual > limit,
    "==": lambda actual, limit: actual == limit,
}


def _constraint_error(name: str, rule: tuple, metrics: dict) -> Optional[str]:
    relation, limit = rule
    comparison = _COMPARISONS.get(relation)
    if comparison is None:
        return f"{name}: unknown operator '{relation}'"

    sentinel = object()
    actual = metrics.get(name, sentinel)
    if actual is sentinel or actual is None:
        return f"{name}: metric missing (required {relation} {limit})"
    if not isinstance(actual, (int, float)):
        return f"{name}: non-numeric value {actual!r}"
    if comparison(actual, limit):
        return None
    return f"{name}: {actual} violates {relation} {limit}"


def check_constraints(result: EvalResult, constraints: ConstraintMap) -> list[str]:
    """Check hard constraints. Returns list of violation strings (empty = ok)."""
    checked = (
        _constraint_error(name, rule, result.metrics)
        for name, rule in constraints.items()
    )
    return [message for message in checked if message is not None]


# ---------------------------------------------------------------------------
# Improvement comparison
# ---------------------------------------------------------------------------

def is_improvement(
    current: EvalResult,
    best: EvalResult,
    metric: MetricName = DEFAULT_PRIMARY_METRIC,
    lower_is_better: bool = DEFAULT_LOWER_IS_BETTER,
    threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> bool:
    """Check if current result improves on best.

    threshold is a relative percentage (e.g. 2.0 = needs >2% improvement).
    """
    if current.outcome != EvalOutcome.OK:
        return False

    missing = object()
    candidate = current.metrics.get(metric, missing)
    baseline = best.metrics.get(metric, missing)
    if candidate is missing or candidate is None:
        return False
    if baseline is missing or baseline is None:
        return True

    direction = -1 if lower_is_better else 1
    signed_gain = direction * (candidate - baseline)
    if baseline == 0:
        return signed_gain > 0
    gain_percent = 100.0 * signed_gain / abs(baseline)
    return gain_percent > threshold
