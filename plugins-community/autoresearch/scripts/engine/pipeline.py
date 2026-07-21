#!/usr/bin/env python3
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

"""Post-edit pipeline — runs ALL mechanical steps after Claude Code edits code.

Claude Code does the LLM work (plan, edit, diagnose). Then calls this:
    python scripts/engine/pipeline.py <task_dir>

Steps inside `with open_task(td, role="agent")`:
    1. quick_check → fail? rollback, report
    2. eval → get metrics
    3. t.record_round → KEEP/DISCARD/FAIL (journals + history + state)
    4. t.settle_round → plan.md + atomic phase/pending_settle commit
    5. print status + next guidance

Output: human-readable status to stdout. Claude Code sees it and acts accordingly.

Recovery: open_task at entry calls replay_intent + consistency check.
Any in-flight round transaction (intent.json + body landed before
state save) is reconstructed before this script proceeds. If state
.pending_settle is non-null after replay, this script runs the
replay-only settle branch (skips quick_check/eval/record_round).
"""
import json
import os
import sys
from typing import Optional

from op_autoresearch.utils.console import emit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Logging is owned by op_autoresearch/__init__.py (root -> stdout) so the eval
# INFO chain stays chronological and the AR Phase guidance lands last; the
# imports below trigger it. No basicConfig here — single owner.

from phase_machine import EDIT, get_guidance
from quick_check import (
    _run_smoke_test as _run_smoke,
)
from quick_check import (
    check_editable_files,
    effective_edit_issue,
)
from task_config import run_eval
import task_handle as task_api
from utils import eval_summary as eval_summary_utils
from utils.settings import recorded_speedup


def _emit_settle_failure(error_tail: str) -> None:
    emit(f"[PIPELINE] SETTLE INCOMPLETE. plan.md may already contain "
          f"the idempotent settlement, but state.pending_settle is still "
          f"authoritative; re-running this script will RETRY SETTLE ONLY "
          f"(kd_json was persisted to state.pending_settle) — it "
          f"will NOT re-run quick_check/eval/record_round.\n"
          f"\n"
          f"Recovery options (do NOT hand-edit plan.md):\n"
          f"  1. Fix the underlying cause from the error tail below, "
          f"then re-run pipeline.py — the replay-only path will "
          f"retry settle on the same kd_json.\n"
          f"  2. If the failure is structural (plan.md malformed, "
          f"no (ACTIVE) item, etc.) and settle cannot recover, run "
          f"create_plan.py to write a fresh plan.md. While "
          f"state.pending_settle is non-null, hooks/guard_bash "
          f"allows create_plan.py in EDIT phase as a recovery path; "
          f"on successful create_plan validation hooks/post_bash "
          f"clears state.pending_settle. The orphan history.jsonl "
          f"row stays (audit trail).\n"
          f"\n"
          f"error: {error_tail}", file=sys.stderr)


def _print_round_summary(t, decision: str, settled_id: str,
                         next_phase: str) -> None:
    """Status line + guidance after a settled round.

    Commit hash is included on KEEP rounds — that's the only decision
    where best_commit changed this round, so reporting it here surfaces
    "what kernel just got committed" without a duplicate stderr print
    from record_round.
    """
    # Progress is guaranteed initialised here (we're in EDIT, which
    # implies baseline committed). Read via Task's typed accessor.
    progress = t.progress
    rounds = progress.eval_rounds
    max_rounds = progress.max_rounds
    best = progress.best_metric
    failures = progress.consecutive_failures

    # Stored geomean speedup (best_speedup); pct derived from it so both numbers
    # are tied. Empty when unset — never re-derive from baseline/best latencies.
    speedup = recorded_speedup(progress)
    improv = ""
    if speedup is not None:
        pct = (1.0 - 1.0 / speedup) * 100
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    best_str = f"{best:.2f}" if isinstance(best, (int, float)) else str(best)
    commit_str = ""
    if decision == "KEEP" and progress.best_commit:
        commit_str = f" | commit: {progress.best_commit}"

    emit(f"\n{'=' * 50}")
    emit(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | "
          f"Best: {best_str}{improv} | Failures: {failures}{commit_str}")
    emit(f"Phase -> {next_phase}")
    emit(f"{'=' * 50}")
    # Drain any pending stderr (e.g. tracebacks from settle path) so
    # the AR Phase guidance lands as the very last line in the capture.
    sys.stderr.flush()
    sys.stdout.flush()
    emit(get_guidance(t.task_dir))


def _run_with_task(task) -> int:
    """Run one pipeline round inside an already-open task transaction."""
    task.require_phase(EDIT, action="pipeline")
    replay_result = _retry_pending_settle(task)
    if replay_result is not None:
        return replay_result
    config = task.config
    if config is None:
        emit("[PIPELINE] ERROR: task.yaml not found")
        return 1

    active = task.active_plan_item()
    description = active["description"] if active else "optimization round"
    plan_item = active["id"] if active else None
    if not _quick_check(task, config):
        return 0
    eval_json, early_result = _evaluate_round(task, config)
    if early_result is not None:
        return early_result
    return _record_and_settle(
        task,
        eval_json,
        description,
        plan_item,
    )


def _retry_pending_settle(task) -> Optional[int]:
    pending = task.pending_settle
    if not pending:
        return None
    emit(
        "[PIPELINE] Retrying settle from state.pending_settle "
        "(skipping quick_check/eval/record_round).",
        flush=True,
    )
    try:
        result = task.settle_round()
    except task_api.TaskCorrupted as exc:
        _emit_settle_failure(str(exc))
        return 1
    _print_round_summary(
        task,
        pending.get("decision", "?"),
        result["settled_item"] or "?",
        result["next_phase"],
    )
    return 0


def _quick_check(task, config) -> bool:
    emit("[PIPELINE] Running quick_check...", flush=True)
    try:
        edit_issue = effective_edit_issue(task.task_dir, config)
        file_issues = [edit_issue] if edit_issue else []
        if not file_issues:
            file_issues = check_editable_files(task.task_dir, config)
        smoke_errors = _run_smoke(task.task_dir, config)
    except Exception as exc:
        file_issues = [{
            "file": "(internal)",
            "report": (
                f"quick_check crashed: {type(exc).__name__}: {exc}"
            ),
            "errors": [],
        }]
        smoke_errors = []
    if not file_issues and not smoke_errors:
        emit("[PIPELINE] Quick check PASS", flush=True)
        return True

    task.rollback_edit()
    details = {"ok": False}
    if file_issues:
        details["file_issues"] = file_issues
    if smoke_errors:
        details["smoke_errors"] = smoke_errors
    emit(
        "[PIPELINE] QUICK CHECK FAIL: "
        f"{json.dumps(details, ensure_ascii=False)[:200]}"
    )
    emit("[PIPELINE] Auto-rolled back. Fix and re-edit.")
    emit(get_guidance(task.task_dir))
    return False


def _evaluate_round(task, config) -> tuple[Optional[dict], Optional[int]]:
    emit("[PIPELINE] Running eval...", flush=True)
    try:
        result = run_eval(
            task.task_dir,
            config,
            current_step=task.progress.next_round,
        )
    except Exception as exc:
        task.rollback_edit()
        emit(
            f"[PIPELINE] EVAL ERROR: run_eval raised "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None, 1
    eval_json = eval_summary_utils.eval_result_to_dict(result)
    eval_summary_utils.print_eval_metrics(eval_json, "PIPELINE")
    if eval_json.get("outcome") == "infra_fail":
        task.rollback_edit()
        emit(
            f"[PIPELINE] INFRA_FAIL: "
            f"{eval_json.get('error', 'no data')}. "
            "Rolled back, not recording round.",
            flush=True,
        )
        return None, 0
    eval_summary_utils.print_failure_signals(eval_json, "PIPELINE")
    return eval_json, None


def _record_and_settle(
    task,
    eval_json: dict,
    description: str,
    plan_item: Optional[str],
) -> int:
    decision_record = task.record_round(
        eval_json,
        description=description,
        plan_item=plan_item,
    )
    if decision_record.get("decision") == "ERROR":
        emit(
            "[PIPELINE] KEEP/DISCARD ERROR: "
            f"{decision_record.get('error')}"
        )
        return 1
    try:
        result = task.settle_round()
    except task_api.TaskCorrupted as exc:
        _emit_settle_failure(str(exc))
        return 1
    _print_round_summary(
        task,
        decision_record.get("decision", "FAIL"),
        result["settled_item"] or "?",
        result["next_phase"],
    )
    return 0


def main():
    argv = sys.argv[1:]
    # --trace: keep the msprof trace dirs (timeline + CSVs) for analysis.
    if "--trace" in argv:
        os.environ["OP_AUTORESEARCH_PROF_KEEP_RES"] = "1"
        argv = [a for a in argv if a != "--trace"]
    if not argv:
        emit("Usage: python pipeline.py <task_dir> [--trace]")
        sys.exit(1)

    task_dir = os.path.abspath(argv[0])

    # The body returns rc; sys.exit happens AFTER the with-block so
    # SystemExit from a non-zero normal completion doesn't trip
    # __exit__'s release-on-exception path (which would unclaim the
    # task and break the next post_bash hook's get_task_dir()).
    rc = 1
    try:
        with task_api.open_task(task_dir, role=task_api.Role.AGENT) as t:
            rc = _run_with_task(t)
    except task_api.TaskConsistencyError as e:
        emit(f"[PIPELINE] REFUSING TO RUN — {e}", file=sys.stderr)
    except task_api.TaskOwnershipError as e:
        emit(f"[PIPELINE] cannot run: {e}", file=sys.stderr)
        rc = 2
    except task_api.TaskPhaseError as e:
        emit(f"[PIPELINE] refused: {e}", file=sys.stderr)
        rc = 2

    sys.exit(rc)


if __name__ == "__main__":
    main()
