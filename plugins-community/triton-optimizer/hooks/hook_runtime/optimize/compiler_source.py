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

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

COMPILER_SOURCE_REPO_URL = "https://gitcode.com/Ascend/AscendNPU-IR.git"
COMPILER_SOURCE_DIR_NAME = "AscendNPU-IR"
CompilerSourceMode = Literal["off", "auto"]
RunGit = Callable[[list[str], Optional[Path]], str]


@dataclass(frozen=True)
class CompilerSourceInfo:
    path: Path
    commit: str


def _compiler_source_cache_dir() -> Path:
    configured = os.environ.get("TRITON_AGENT_COMPILER_SOURCE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".triton-agent").resolve()


def default_compiler_source_path(cache_dir: Path | None = None) -> Path:
    root = cache_dir or _compiler_source_cache_dir()
    return root / "compiler-sources" / COMPILER_SOURCE_DIR_NAME


def existing_compiler_source_path(cache_dir: Path | None = None) -> Path | None:
    checkout = default_compiler_source_path(cache_dir)
    try:
        _validate_git_checkout(checkout)
    except ValueError:
        return None
    return checkout


def prepare_compiler_source(
    *,
    mode: CompilerSourceMode,
    cache_dir: Path | None = None,
    run_git: RunGit | None = None,
) -> CompilerSourceInfo | None:
    if mode == "off":
        return None
    if mode != "auto":
        raise ValueError(f"Unsupported compiler source analysis mode: {mode}")

    git_runner = run_git or _run_git
    checkout = default_compiler_source_path(cache_dir)

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        git_runner(
            [
                "git",
                "clone",
                "--depth",
                "1",
                COMPILER_SOURCE_REPO_URL,
                str(checkout),
            ],
            None,
        )

    _validate_git_checkout(checkout)
    commit = _inspect_commit(checkout, git_runner)
    return CompilerSourceInfo(path=checkout, commit=commit)


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"Failed to run {' '.join(args)}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise ValueError(f"Failed to run {' '.join(args)}: {detail}")
    return result.stdout


def _validate_git_checkout(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Compiler source path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Compiler source path is not a directory: {path}")
    if not (path / ".git").exists():
        raise ValueError(f"Compiler source path is not a git checkout: {path}")


def _inspect_commit(path: Path, run_git: RunGit) -> str:
    commit = run_git(["git", "rev-parse", "HEAD"], path).strip()
    if not commit:
        raise ValueError(f"Unable to resolve compiler source commit for {path}")
    return commit
