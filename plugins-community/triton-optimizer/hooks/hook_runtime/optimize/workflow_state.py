# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil

from hook_runtime.optimize.baseline import baseline_dir, load_baseline_state
from hook_runtime.optimize.resume import classify_optimize_workspace
from hook_runtime.skill_loader import load_skill_script_module


@lru_cache(maxsize=1)
def _state_machine_module():
    return load_skill_script_module("triton-npu-optimize", "optimize-state/state_manage/state_machine")


@dataclass(frozen=True)
class WorkflowStateBootstrapResult:
    mode: str
    state_path: Path


def bootstrap_optimize_workflow_state(
    state_path: Path,
    *,
    run_id: str,
    baseline_reused: bool,
) -> None:
    _state_machine_module().bootstrap_state(
        state_path,
        run_id=run_id,
        baseline_reused=baseline_reused,
    )


def prepare_or_restore_optimize_workflow_state(
    source_operator: Path | None,
    workdir: Path,
    *,
    state_path: Path,
    run_id: str,
) -> WorkflowStateBootstrapResult:
    if state_path.exists():
        _state_machine_module().load_state(state_path)
        return WorkflowStateBootstrapResult(
            mode="reused-existing-state",
            state_path=state_path,
        )

    inspection_operator, resolution_issue = _resolve_source_operator_hint(source_operator, workdir)
    inspection = (
        classify_optimize_workspace(inspection_operator, workdir)
        if inspection_operator is not None
        else _no_session_inspection()
    )
    if resolution_issue is not None and _has_optimize_session_residue(workdir):
        raise ValueError(resolution_issue)
    if inspection.state == "resumable-session":
        bootstrap_optimize_workflow_state(
            state_path,
            run_id=run_id,
            baseline_reused=True,
        )
        return WorkflowStateBootstrapResult(
            mode="rebuilt-from-durable-artifacts",
            state_path=state_path,
        )
    if inspection.state == "partial-session":
        raise ValueError(f"cannot rebuild optimize workflow state from partial session: {inspection.detail}")

    bootstrap_optimize_workflow_state(
        state_path,
        run_id=run_id,
        baseline_reused=False,
    )
    return WorkflowStateBootstrapResult(
        mode="bootstrapped-fresh-baseline",
        state_path=state_path,
    )


def mark_baseline_passed_in_workflow_state(state_path: Path | None) -> None:
    if state_path is None or not state_path.exists():
        return
    _state_machine_module().mark_baseline_passed(state_path)


def render_optimize_phase_summary(state_path: Path | None) -> str | None:
    if state_path is None or not state_path.exists():
        return None
    return str(_state_machine_module().render_phase_summary(state_path))


def archive_round_timings_from_state(state_path: Path | None, archive_path: Path) -> bool:
    if state_path is None or not state_path.exists():
        return False
    timing_dir = state_path.parent / "round-timings"
    if not timing_dir.is_dir():
        return False
    shutil.copytree(timing_dir, archive_path, dirs_exist_ok=True)
    return True


def _resolve_source_operator_hint(
    source_operator: Path | None,
    workdir: Path,
) -> tuple[Path | None, str | None]:
    if source_operator is not None:
        return source_operator, None

    try:
        baseline_state = load_baseline_state(workdir)
    except FileNotFoundError:
        return None, None
    except ValueError as exc:
        return None, f"cannot determine source operator from baseline/state.json: {exc}"

    baseline_dir_path = baseline_dir(workdir)
    declared = Path(baseline_state.source_operator)
    if declared.is_absolute():
        return declared, None
    return (baseline_dir_path / declared).resolve(), None


def _has_optimize_session_residue(workdir: Path) -> bool:
    if baseline_dir(workdir).exists():
        return True
    if (workdir / "opt-note.md").exists():
        return True
    return any(path.is_dir() for path in workdir.glob("opt-round-*"))


def _no_session_inspection():
    @dataclass(frozen=True)
    class _Inspection:
        state: str
        detail: str | None
        test_mode: str | None
        bench_mode: str | None

    return _Inspection(state="no-session", detail=None, test_mode=None, bench_mode=None)
