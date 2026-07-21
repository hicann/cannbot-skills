# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Round-N (post-EDIT) eval recorder. `record_round` is called in-process
by engine/pipeline.py and returns {decision, best_metric, eval_rounds,
max_rounds, consecutive_failures}.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from op_autoresearch.utils.console import emit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (
    Progress,
    append_history,
    clear_intent,
    load_progress,
    load_state,
    save_state,
    state_transaction,
    write_intent,
)

# save_progress not imported — record_round writes progress fields
# directly into state.json as part of one atomic save_state, so it
# can also bundle in pending_settle + expected_history_round.
from task_config import (
    EvalOutcome,
    EvalResult,
    check_constraints,
    is_improvement,
    load_task_config,
)
from utils.git_utils import (
    auto_rollback,
    commit_in_task,
    current_head_short,
)

from .progress_reducer import (
    RoundProgressUpdate,
    eval_result_from_data,
    reduce_round_progress,
)


@dataclass(frozen=True)
class RoundContext:
    task_dir: str
    eval_data: dict
    description: str
    plan_item: Optional[str]
    config: Any
    progress: Progress
    evaluation: EvalResult

    @property
    def round_num(self) -> int:
        return self.progress.next_round


@dataclass(frozen=True)
class RoundOutcomeState:
    decision: str
    commit_hash: Optional[str]
    consecutive_failures: int
    best_metric: Optional[float]
    best_commit: Optional[str]
    best_speedup: Optional[float]


def _classify_round(context: RoundContext) -> str:
    evaluation = context.evaluation
    progress = context.progress
    config = context.config
    if not evaluation.correctness:
        emit("[record_round] FAIL: correctness check failed")
        return "FAIL"
    violations = (check_constraints(evaluation, config.constraints)
                  if config.constraints else [])
    if violations:
        emit(f"[record_round] FAIL: constraint violations: {violations}")
        return "FAIL"
    current = evaluation.metrics.get(config.primary_metric)
    if not isinstance(current, (int, float)) or current != current:
        emit(f"[record_round] FAIL: correctness=PASS but primary "
              f"metric '{config.primary_metric}' missing from "
              f"{sorted(evaluation.metrics)}")
        return "FAIL"
    if progress.best_metric is None:
        return "KEEP"
    best = EvalResult(
        outcome=EvalOutcome.OK,
        metrics={config.primary_metric: progress.best_metric},
    )
    improved = is_improvement(
        evaluation,
        best,
        metric=config.primary_metric,
        lower_is_better=config.lower_is_better,
        threshold=config.improvement_threshold,
    )
    return "KEEP" if improved else "DISCARD"


def _initial_outcome(context: RoundContext, decision: str,
                     ) -> RoundOutcomeState:
    progress = context.progress
    failures = (progress.consecutive_failures + 1
                if decision == "FAIL" else progress.consecutive_failures)
    return RoundOutcomeState(
        decision=decision,
        commit_hash=None,
        consecutive_failures=failures,
        best_metric=progress.best_metric,
        best_commit=progress.best_commit,
        best_speedup=progress.best_speedup,
    )


def _measured_speedup(context: RoundContext) -> Optional[float]:
    measured = context.evaluation.metrics.get("speedup_vs_ref")
    if isinstance(measured, (int, float)) and measured > 0:
        return float(measured)
    return context.progress.best_speedup


def _preserve_kept_round(context: RoundContext) -> RoundOutcomeState:
    config = context.config
    metric = context.evaluation.metrics.get(config.primary_metric)
    ok, info = commit_in_task(
        context.task_dir,
        config.editable_files,
        f"autoresearch: {context.description} | {config.primary_metric}={metric}",
    )
    if not ok:
        emit(f"[record_round] git commit failed: {info}; demoting "
              f"KEEP -> FAIL (kernel state not preserved)")
        auto_rollback(context.task_dir)
        return RoundOutcomeState(
            decision="FAIL",
            commit_hash=None,
            consecutive_failures=context.progress.consecutive_failures + 1,
            best_metric=context.progress.best_metric,
            best_commit=context.progress.best_commit,
            best_speedup=context.progress.best_speedup,
        )
    commit_hash = (current_head_short(context.task_dir)
                   or context.progress.best_commit) if info == "noop" else info
    return RoundOutcomeState(
        decision="KEEP",
        commit_hash=commit_hash,
        consecutive_failures=0,
        best_metric=metric,
        best_commit=commit_hash,
        best_speedup=_measured_speedup(context),
    )


def _settle_edit(context: RoundContext, decision: str) -> RoundOutcomeState:
    if decision == "KEEP":
        return _preserve_kept_round(context)
    auto_rollback(context.task_dir)
    emit(f"[record_round] {decision}: rolled back editable files")
    return _initial_outcome(context, decision)


def _reduce_round(context: RoundContext,
                  outcome: RoundOutcomeState) -> Progress:
    reduction = reduce_round_progress(
        context.progress,
        context.evaluation,
        RoundProgressUpdate(
            round_num=context.round_num,
            consecutive_failures=outcome.consecutive_failures,
            best_metric=outcome.best_metric,
            best_commit=outcome.best_commit,
            best_speedup=outcome.best_speedup,
        ),
    )
    if reduction.anchor.changed and reduction.anchor.message:
        emit(f"[record_round] {reduction.anchor.message} "
              f"from R{context.round_num}")
    return reduction.progress


def _round_result(context: RoundContext, progress: Progress,
                  outcome: RoundOutcomeState) -> dict:
    return {
        "decision": outcome.decision,
        "best_metric": progress.best_metric,
        "round_metric": context.evaluation.metrics.get(
            context.config.primary_metric),
        "eval_rounds": context.round_num,
        "max_rounds": progress.max_rounds or context.config.max_rounds,
        "consecutive_failures": progress.consecutive_failures,
        "plan_item": context.plan_item,
        "plan_version": progress.plan_version,
        "round": context.round_num,
    }


def _failure_evidence(context: RoundContext) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    signals = context.eval_data.get("failure_signals")
    signal_present = isinstance(signals, dict) and any(
        signals.get(key) for key in ("primary", "python_error", "signals")
    )
    if signal_present:
        evidence["failure_signals"] = signals
    tail = (context.eval_data.get("raw_output_tail") or "").strip()
    if tail:
        evidence["raw_output_tail"] = tail[-1500:]
    return evidence


def _history_record(context: RoundContext,
                    outcome: RoundOutcomeState) -> dict[str, Any]:
    record: dict[str, Any] = {
        "round": context.round_num,
        "plan_item": context.plan_item,
        "description": context.description,
        "decision": outcome.decision,
        "metrics": context.evaluation.metrics,
        "correctness": context.evaluation.correctness,
        "error": context.evaluation.error,
        "commit": outcome.commit_hash,
    }
    if outcome.decision == "FAIL":
        record.update(_failure_evidence(context))
    return record


def _persist_round(context: RoundContext, progress: Progress,
                   result: dict, history: dict[str, Any]) -> None:
    state_patch = {
        **progress.to_dict(),
        "pending_settle": result,
        "expected_history_round": context.round_num,
    }
    write_intent(context.task_dir, {
        "kind": "round",
        "round": context.round_num,
        "kd_json": result,
        "state_patch": state_patch,
    })
    append_history(context.task_dir, history)
    state = load_state(context.task_dir) or {}
    state.update(state_patch)
    save_state(context.task_dir, state)
    clear_intent(context.task_dir)


def record_round(task_dir: str, eval_data: dict,
                 description: str = "optimization round",
                 plan_item: Optional[str] = None) -> dict:
    with state_transaction(task_dir):
        return _record_round(
            task_dir, eval_data, description=description, plan_item=plan_item)


def _record_round(task_dir: str, eval_data: dict,
                  description: str = "optimization round",
                  plan_item: Optional[str] = None) -> dict:
    """Single library entry point for one round of EDIT settlement.

    Atomically commits progress fields + pending_settle + the
    expected_history_round marker in one save_state. Decision flow:
    correctness → constraints → primary-metric presence → improvement.
    """
    config = load_task_config(task_dir)
    if config is None:
        return {"decision": "ERROR", "error": "task.yaml not found"}

    progress = load_progress(task_dir) or Progress()
    eval_result = eval_result_from_data(eval_data)
    context = RoundContext(
        task_dir=task_dir,
        eval_data=eval_data,
        description=description,
        plan_item=plan_item,
        config=config,
        progress=progress,
        evaluation=eval_result,
    )
    outcome = _settle_edit(context, _classify_round(context))

    progress = _reduce_round(context, outcome)

    kd_json = _round_result(context, progress, outcome)

    hist = _history_record(context, outcome)

    _persist_round(context, progress, kd_json, hist)
    return kd_json
