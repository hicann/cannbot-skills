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

from __future__ import annotations

__all__ = [
    "BootstrapResult",
    "bootstrap_runtime_state",
    "cleanup_runtime_tree",
    "compiler_source_read_root",
    "prepare_compiler_source_context",
    "resolve_workspace",
    "validate_existing_state",
]

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from hook_runtime.optimize.compiler_source import (
    CompilerSourceInfo,
    RunGit,
    existing_compiler_source_path,
    prepare_compiler_source,
)
from hook_runtime.optimize.workflow_state import prepare_or_restore_optimize_workflow_state


@dataclass(frozen=True)
class BootstrapResult:
    additional_context: str | None = None


def bootstrap_runtime_state(
    workspace: Path,
    *,
    compiler_source_enabled: bool | None = None,
    compiler_source_cache_dir: Path | None = None,
    run_git: RunGit | None = None,
) -> BootstrapResult:
    contexts: list[str] = []
    runtime_dir = workspace / ".triton-agent"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / "state.json"
    try:
        prepare_or_restore_optimize_workflow_state(
            None,
            workspace,
            state_path=state_path,
            run_id=_plugin_run_id(),
        )
    except (ValueError, OSError) as exc:
        contexts.append(_workflow_repair_guidance(str(exc)))

    compiler_context = prepare_compiler_source_context(
        compiler_source_enabled=compiler_source_enabled,
        compiler_source_cache_dir=compiler_source_cache_dir,
        run_git=run_git,
    )
    if compiler_context is not None:
        contexts.append(compiler_context)

    return BootstrapResult(_join_contexts(contexts))


def prepare_compiler_source_context(
    *,
    compiler_source_enabled: bool | None = None,
    compiler_source_cache_dir: Path | None = None,
    run_git: RunGit | None = None,
) -> str | None:
    if not _compiler_source_enabled(compiler_source_enabled):
        return None
    try:
        compiler_source = prepare_compiler_source(
            mode="auto",
            cache_dir=compiler_source_cache_dir,
            run_git=run_git,
        )
    except ValueError as exc:
        return (
            "Compiler source analysis is unavailable: "
            f"{exc}. Continue optimize work without compiler source unless the user fixes "
            "the checkout or network access."
        )
    if compiler_source is None:
        return None
    return _compiler_source_context(compiler_source)


def compiler_source_read_root(
    *,
    compiler_source_enabled: bool | None = None,
    compiler_source_cache_dir: Path | None = None,
) -> Path | None:
    if not _compiler_source_enabled(compiler_source_enabled):
        return None
    return existing_compiler_source_path(compiler_source_cache_dir)


def validate_existing_state(state_path: Path) -> BootstrapResult:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BootstrapResult(None)
    except json.JSONDecodeError as exc:
        return BootstrapResult(
            _workflow_repair_guidance(
                f"Existing optimize workflow state is malformed: {exc}."
            )
        )
    if not isinstance(payload, dict):
        return BootstrapResult(
            _workflow_repair_guidance(
                "Existing optimize workflow state must be a JSON object."
            )
        )
    try:
        prepare_or_restore_optimize_workflow_state(
            None,
            state_path.parent.parent,
            state_path=state_path,
            run_id=_plugin_run_id(),
        )
    except (ValueError, OSError) as exc:
        return BootstrapResult(
            _workflow_repair_guidance(
                f"Existing optimize workflow state is invalid: {exc}."
            )
        )
    return BootstrapResult(None)


def cleanup_runtime_tree(runtime_dir: Path) -> None:
    if runtime_dir.name != ".triton-agent":
        return
    if runtime_dir.is_symlink() or runtime_dir.is_file():
        runtime_dir.unlink()
        return
    if runtime_dir.is_dir():
        shutil.rmtree(runtime_dir)


def resolve_workspace(payload: dict[str, object]) -> Path | None:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd).expanduser().resolve()
    return None


def _plugin_run_id() -> str:
    return "claude-plugin-session"


def _compiler_source_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    configured = os.environ.get("TRITON_AGENT_CLAUDE_PLUGIN_COMPILER_SOURCE", "auto")
    return configured.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _compiler_source_context(compiler_source: CompilerSourceInfo) -> str:
    return "\n".join(
        [
            "Compiler source analysis is enabled.",
            f"Compiler source path: {compiler_source.path}",
            f"Compiler source commit: {compiler_source.commit}.",
            "Treat the compiler source checkout as read-only.",
            "Do not run git clone, git fetch, git pull, or modify files in the compiler source checkout.",
            "Use the bundled triton-npu-analyze-compiler-source skill only when compiler source evidence is needed.",
        ]
    )


def _join_contexts(contexts: list[str]) -> str | None:
    if not contexts:
        return None
    return "\n\n".join(contexts)


def _workflow_repair_guidance(problem: str) -> str:
    return (
        f"{problem} "
        "Use `triton-npu-optimize` `optimize-state submit-baseline` to repair session state, "
        "then use `optimize-state start-round` to reopen the intended `opt-round-N/` before "
        "continuing round edits, same-round state updates, or round submission."
    )


def _edit_blocked_workflow_guidance(problem: str) -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        + problem
    )
