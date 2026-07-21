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

"""Resume an existing autoresearch task.

Usage:
    python scripts/resume.py [task_dir] [--force]

If task_dir is omitted, auto-detects the most recently active task
(prefers the current session's task via the session index).

Opens a Task with role="agent" — heal + consistency check + claim
ownership happen in one place. Refusal to claim a fresh-but-foreign
task is the TaskOwnershipError path; --force takes over via Task's
force flag. The journal handles the partial-baseline crash window
(SEED row landed, state didn't commit progress_initialized=True)
transparently: replay_intent inside open_task rebuilds state before
this script reads anything.
"""
import os
import sys

from op_autoresearch.utils.console import emit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_machine import (
    edit_marker_path,
    find_active_task_dir,
    has_pending_items,
    plan_path,
)
from task_handle import (
    Role,
    TaskConsistencyError,
    TaskNotInitialized,
    TaskOwnershipError,
    open_task,
)


def _validate_resumable(t) -> tuple[bool, str]:
    """Post-open validation. open_task already did heal + consistency +
    claim; here we only check resume-specific shape (progress
    initialised, plan.md still valid if non-empty).
    """
    try:
        progress = t.progress
    except TaskNotInitialized:
        # Resumability is keyed on PHASE, not progress presence. A task
        # parked at BASELINE with no committed baseline is the legitimate
        # "baseline pending" state (the gate refused to commit because no
        # valid ref baseline) — resume re-runs baseline.py after the env/
        # ref/worker is fixed. Any other phase with no progress is a task
        # that was never initialised.
        from phase_machine import BASELINE, read_phase
        if read_phase(t.task_dir) == BASELINE:
            return True, ""
        return False, ("Baseline never committed and phase is not "
                       "BASELINE — task was never initialised. Run "
                       "/autoresearch without --resume to start fresh.")
    required_fields = {"task", "eval_rounds", "max_rounds"}
    missing = required_fields - set(progress.keys())
    if missing:
        return False, f"state.json progress fields missing: {missing}"
    # plan.md present + has pending items → must be structurally valid.
    # A fully-consumed plan (0 pending) is legal (phase_on_resume
    # routes it to REPLAN). validate_plan rejects 0-pending plans for
    # lack of an ACTIVE item, so only validate when items exist.
    if os.path.exists(plan_path(t.task_dir)) and has_pending_items(t.task_dir):
        from phase_machine import validate_plan
        ok, err = validate_plan(t.task_dir)
        if not ok:
            return False, f"plan.md invalid: {err}"
    return True, ""


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    positional = [arg for arg in args if arg != "--force"]
    task_dir = _resolve_task_dir(positional[0] if positional else None)
    if task_dir is None:
        return 1
    return _resume_task(task_dir, force)


def _resolve_task_dir(candidate: str | None) -> str | None:
    task_dir = (
        os.path.abspath(candidate)
        if candidate
        else find_active_task_dir() or ""
    )
    if not task_dir:
        emit(
            "[resume] ERROR: No existing task found in ar_tasks/",
            file=sys.stderr,
        )
        return None
    if not os.path.isdir(task_dir):
        emit(f"[resume] ERROR: Not a directory: {task_dir}", file=sys.stderr)
        return None
    if not os.path.exists(os.path.join(task_dir, "task.yaml")):
        emit(
            f"[resume] ERROR: Missing task.yaml in {task_dir}",
            file=sys.stderr,
        )
        return None
    return task_dir


def _resume_task(task_dir: str, force: bool) -> int:
    try:
        with open_task(task_dir, role=Role.AGENT, force=force) as task:
            valid, error = _validate_resumable(task)
            if not valid:
                emit(
                    f"[resume] ERROR: Cannot resume {task_dir}",
                    file=sys.stderr,
                )
                emit(f"[resume] {error}", file=sys.stderr)
                return 1
            _clean_stale_edit_marker(task_dir)
            _print_resume_summary(task, task_dir)
            return 0
    except TaskConsistencyError as exc:
        emit(
            f"[resume] ERROR: state inconsistent for {task_dir}",
            file=sys.stderr,
        )
        emit(f"[resume] {exc}", file=sys.stderr)
        return 1
    except TaskOwnershipError as exc:
        _print_ownership_error(exc)
        return 1


def _clean_stale_edit_marker(task_dir: str) -> None:
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    from utils.git_utils import is_working_tree_clean
    if not is_working_tree_clean(task_dir):
        return
    try:
        os.remove(marker)
        emit("[resume] Cleaned stale edit marker.", file=sys.stderr)
    except OSError:
        return


def _print_resume_summary(task, task_dir: str) -> None:
    summary = task.summary or {}
    emit(f"[resume] Task: {summary.get('task')}")
    emit(
        f"[resume] Round: {summary.get('eval_rounds')}/"
        f"{summary.get('max_rounds')}"
    )
    emit(
        f"[resume] Best: {summary.get('best_metric')} | "
        f"Baseline: {summary.get('baseline_metric')}"
    )
    emit(f"[resume] Phase: {summary.get('phase')}")
    emit(task_dir)


def _print_ownership_error(error: Exception) -> None:
    emit(f"[resume] ERROR: {error}", file=sys.stderr)
    emit(
        "[resume] Another Claude Code session may be running it.",
        file=sys.stderr,
    )
    emit(
        "[resume] If you're sure no other session is running, add --force:",
        file=sys.stderr,
    )
    emit("[resume]   /autoresearch --resume --force", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
