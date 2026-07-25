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

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hook_runtime.optimize.baseline import baseline_dir, baseline_gate_issues, load_baseline_state
from hook_runtime.skill_loader import load_skill_script_module


@dataclass(frozen=True)
class ResumeResolution:
    workspace_state: str
    resume_existing_session: bool
    test_mode: str | None
    bench_mode: str | None


@dataclass(frozen=True)
class WorkspaceInspection:
    state: str
    detail: str | None
    test_mode: str | None
    bench_mode: str | None


def classify_optimize_workspace(input_path: Path, workdir: Path) -> WorkspaceInspection:
    return _classify_optimize_workspace(input_path, workdir)


def reset_optimize_workspace(input_path: Path, workdir: Path) -> None:
    pt_result_files = [
        path
        for path in workdir.glob("*_result.pt")
        if path.is_file()
    ]
    artifact_paths = [
        workdir / "opt-note.md",
        workdir / "learned_lessons.md",
        baseline_dir(workdir),
        workdir / ".triton-agent",
        workdir / "triton-agent-logs",
        input_path.with_name(f"opt_{input_path.stem}.py"),
    ]
    round_dirs = [
        path
        for path in workdir.glob("opt-round-*")
        if path.is_dir()
    ]
    for path in [*pt_result_files, *artifact_paths, *round_dirs]:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _classify_optimize_workspace(
    input_path: Path,
    workdir: Path,
    *,
    allow_reusable_harnesses_without_session: bool = False,
) -> WorkspaceInspection:
    del allow_reusable_harnesses_without_session

    opt_note_path = workdir / "opt-note.md"
    has_opt_note = opt_note_path.exists()
    has_rounds = any(path.is_dir() for path in workdir.glob("opt-round-*"))
    test_harnesses = _existing_test_harnesses(input_path, workdir)
    bench_harness = _existing_bench_harness(input_path, workdir)
    has_bench = bench_harness.exists()
    has_baseline_dir = baseline_dir(workdir).exists()

    has_optimize_session_markers = has_opt_note or has_rounds
    if not has_optimize_session_markers:
        return _inspect_workspace_without_session(
            workdir,
            has_baseline_dir=has_baseline_dir,
        )
    return _inspect_existing_workspace_session(
        input_path,
        workdir,
        opt_note_path,
        test_harnesses,
        bench_harness,
        has_opt_note=has_opt_note,
        has_bench=has_bench,
    )


def _inspect_workspace_without_session(
    workdir: Path,
    *,
    has_baseline_dir: bool,
) -> WorkspaceInspection:
    if has_baseline_dir:
        baseline_issue = _baseline_issue(workdir)
        if baseline_issue is not None:
            return WorkspaceInspection("partial-session", baseline_issue, None, None)
    return WorkspaceInspection("no-session", None, None, None)


def _inspect_existing_workspace_session(
    input_path: Path,
    workdir: Path,
    opt_note_path: Path,
    test_harnesses: list[Path],
    bench_harness: Path,
    *,
    has_opt_note: bool,
    has_bench: bool,
) -> WorkspaceInspection:

    baseline_issue = _baseline_issue(workdir)
    if baseline_issue is not None:
        return WorkspaceInspection("partial-session", baseline_issue, None, None)

    if not has_opt_note:
        return WorkspaceInspection("partial-session", f"missing {opt_note_path.name}", None, None)
    if not test_harnesses:
        return WorkspaceInspection(
            "partial-session",
            f"missing generated test harness for {input_path.name}",
            None,
            None,
        )
    if len(test_harnesses) > 1:
        return WorkspaceInspection("partial-session", "multiple test harnesses exist", None, None)
    if not has_bench:
        return WorkspaceInspection("partial-session", f"missing {bench_harness.name}", None, None)

    test_mode = _parse_test_mode(test_harnesses[0])
    if test_mode is None:
        return WorkspaceInspection(
            "partial-session",
            f"unreadable test-mode metadata in {test_harnesses[0].name}",
            None,
            None,
        )

    try:
        bench_mode = _resolve_bench_mode_from_baseline(workdir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return WorkspaceInspection(
            "partial-session",
            f"unable to resolve bench mode from baseline/state.json: {exc}",
            None,
            None,
        )

    return WorkspaceInspection("resumable-session", None, test_mode, bench_mode)


def resolve_optimize_resume(
    input_path: Path,
    workdir: Path,
    *,
    resume_mode: str,
    reset_optimize: bool,
    requested_test_mode: str | None,
    requested_bench_mode: str | None,
) -> ResumeResolution:
    inspection = _classify_optimize_workspace(
        input_path,
        workdir,
        allow_reusable_harnesses_without_session=reset_optimize and resume_mode == "fresh",
    )
    if resume_mode == "fresh":
        if inspection.state != "no-session":
            raise ValueError(f"resume fresh refused because optimize artifacts already exist in {workdir}")
        return ResumeResolution(
            workspace_state="no-session",
            resume_existing_session=False,
            test_mode=requested_test_mode or "differential",
            bench_mode=requested_bench_mode or "torch-npu-profiler",
        )

    if resume_mode == "continue":
        return _require_resumable_session(
            input_path, workdir, inspection,
            requested_test_mode=requested_test_mode,
            requested_bench_mode=requested_bench_mode,
        )

    if inspection.state == "no-session":
        return ResumeResolution(
            workspace_state="no-session",
            resume_existing_session=False,
            test_mode=requested_test_mode or "differential",
            bench_mode=requested_bench_mode or "torch-npu-profiler",
        )
    if inspection.state == "partial-session":
        raise ValueError(f"resume auto found partial optimize state: {inspection.detail}")
    _validate_explicit_mode_assertions(inspection, requested_test_mode, requested_bench_mode)
    return ResumeResolution(
        workspace_state="resumable-session",
        resume_existing_session=True,
        test_mode=inspection.test_mode,
        bench_mode=inspection.bench_mode,
    )


def _validate_explicit_mode_assertions(
    inspection: WorkspaceInspection,
    requested_test_mode: str | None,
    requested_bench_mode: str | None,
) -> None:
    if requested_test_mode is not None and requested_test_mode != inspection.test_mode:
        raise ValueError(
            f"--test-mode {requested_test_mode} conflicts with existing harness "
            f"test-mode {inspection.test_mode}"
        )
    if requested_bench_mode is not None and requested_bench_mode != inspection.bench_mode:
        raise ValueError(
            f"--bench-mode {requested_bench_mode} conflicts with existing harness "
            f"bench-mode {inspection.bench_mode}"
        )


def _require_resumable_session(
    input_path: Path,
    workdir: Path,
    inspection: WorkspaceInspection,
    *,
    requested_test_mode: str | None = None,
    requested_bench_mode: str | None = None,
) -> ResumeResolution:
    opt_note_path = workdir / "opt-note.md"
    if not opt_note_path.exists():
        raise ValueError(f"resume continue requires existing opt-note.md: {opt_note_path}")

    baseline_issue = _baseline_issue(workdir)
    if baseline_issue is not None:
        raise ValueError(
            f"resume continue requires established baseline/: {workdir / 'baseline'} ({baseline_issue})"
        )

    test_harnesses = _existing_test_harnesses(input_path, workdir)
    if not test_harnesses:
        raise ValueError(
            f"resume continue requires an existing generated test harness for {input_path.name}"
        )
    if len(test_harnesses) > 1:
        raise ValueError(
            "resume continue found multiple test harnesses. Keep only the active optimize test harness."
        )

    bench_harness = _existing_bench_harness(input_path, workdir)
    if not bench_harness.exists():
        raise ValueError(
            f"resume continue requires an existing generated benchmark harness: {bench_harness}"
        )

    if inspection.test_mode is None:
        raise ValueError(
            f"resume continue requires readable 'test-mode' metadata: {test_harnesses[0]}"
        )
    if inspection.bench_mode is None:
        raise ValueError(
            f"resume continue requires readable 'bench-mode' metadata: {bench_harness}"
        )

    _validate_explicit_mode_assertions(inspection, requested_test_mode, requested_bench_mode)

    return ResumeResolution(
        workspace_state="resumable-session",
        resume_existing_session=True,
        test_mode=inspection.test_mode,
        bench_mode=inspection.bench_mode,
    )


def _existing_test_harnesses(input_path: Path, workdir: Path) -> list[Path]:
    declared_test = _declared_test_harness(input_path, workdir)
    candidates: list[Path] = []
    if declared_test is not None:
        candidates.append(declared_test)
    candidates.extend(
        [
            input_path.with_name(f"differential_test_{input_path.stem}.py"),
            input_path.with_name(f"test_{input_path.stem}.py"),
        ]
    )
    return _existing_unique_paths(candidates)


def _existing_bench_harness(input_path: Path, workdir: Path) -> Path:
    declared_bench = _declared_bench_harness(input_path, workdir)
    if declared_bench is not None and declared_bench.exists():
        return declared_bench
    return input_path.with_name(f"bench_{input_path.stem}.py")


def _resolve_baseline_state_path(relative_path: str, workdir: Path) -> Path:
    declared = Path(relative_path)
    baseline_dir_path = baseline_dir(workdir)
    if not declared.is_absolute():
        return baseline_dir_path / declared
    return declared


def _declared_test_harness(input_path: Path, workdir: Path) -> Path | None:
    state = _matching_baseline_state(input_path, workdir)
    if state is None:
        return None
    path = _resolve_baseline_state_path(state.test_file, workdir)
    if not path.exists():
        return None
    return path


def _declared_bench_harness(input_path: Path, workdir: Path) -> Path | None:
    state = _matching_baseline_state(input_path, workdir)
    if state is None:
        return None
    path = _resolve_baseline_state_path(state.bench_file, workdir)
    if not path.exists():
        return None
    return path


def _matching_baseline_state(input_path: Path, workdir: Path):
    try:
        state = load_baseline_state(workdir)
    except (ValueError, OSError):
        return None

    declared_source = _resolve_baseline_state_path(state.source_operator, workdir).resolve()
    if declared_source != input_path.resolve():
        return None
    return state


def _existing_unique_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            unique_paths.append(path)
    return unique_paths


def _parse_test_mode(test_file: Path) -> str | None:
    metadata = _test_runner_module().parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        return None
    return str(mode)


def _resolve_bench_mode_from_baseline(workdir: Path) -> str:
    state = load_baseline_state(workdir)
    return state.bench_mode


def _baseline_issue(workdir: Path) -> str | None:
    root = baseline_dir(workdir)
    if not root.exists():
        return "missing established baseline/"

    issues = baseline_gate_issues(workdir)
    if not issues:
        return None
    return issues[0]


@lru_cache(maxsize=1)
def _test_runner_module():
    return load_skill_script_module("triton-npu-optimize", "run-eval/test_runner")
