# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-213(b): was the harness pristine for the run being finalized?

The problem this exists for
---------------------------
Nothing asserted that the code doing the measuring was unmodified while it
measured. Concrete case (2026-07-17, self-disclosed — caught by no tool): a
kernel worker edited `phase_o5_runner.py` (the O5 push-list, i.e. the
mechanism that decides whether O5 can even SEE a backend's artifacts), left
it uncommitted, and an op was then finalized on that modified harness with
verdict VERIFIED. The fix was probably correct; the shape was not. The
subject modified the instrument, and the instrument then verified the
subject. Same ruler, or it is not a ruler.

So: sample the harness tree's git state and let the caller downgrade the
verdict (VERIFIED -> PROVISIONAL) when it is dirty.

What counts as "the harness" (the boundary that makes this usable)
------------------------------------------------------------------
Scope is the code that MEASURES, never the code that IS measured. A dirty
`workspace/<op>/` or `output/**` is the worker's own product — that is the
normal, expected state of every single run and flagging it would make this
check pure noise. Measured on the live tree during a real run (2026-07-17):

    any dirty file                       -> 30 entries  (24 of them output/)
    any dirty path under the harness     ->  1 entry
    harness CODE (this module's scope)   ->  0 entries

The middle row is the trap. That one entry was
`src/scripts/orchestrator/.kernel_worker_active` — a runtime marker the
orchestrator itself writes for every worker spawn (`agent_dispatch.py`
`_ACTIVE_AGENT_MARKERS`). Counting it would fire this check on ~every run
that ever spawned a worker. Hence RUNTIME_NOISE_* below, which follows the
exclusion convention `finalize_pipeline.EXCLUDE_PATTERNS_RE` already uses
(and whose comment already names `.kernel_worker_active` by hand).

UNKNOWN does not downgrade
---------------------------
A customer's unpacked bundle is not a git checkout, so `git status` cannot
answer there. That degrades to UNKNOWN, which is RECORDED but does not
downgrade the verdict: in a non-git bundle EVERY run would be UNKNOWN, and a
check that fires on everything is a check the team learns to ignore. Only a
positive DIRTY finding — we looked, and we saw modified harness code —
downgrades. Recorded-but-not-downgrading is not a silent pass; the state
lands in the report either way.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Repo root: .../<root>/src/scripts/orchestrator/harness_pristine.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The code that MEASURES. `plugins/` lives under orchestrator/, so it is
# already covered by the first entry. Deliberately NOT included:
# `workspace/`, `output/` (worker product), `docs/` (prose).
HARNESS_PATHS: tuple[str, ...] = (
    "src/scripts/orchestrator",
    "src/scripts/patches",
)

# Per-component exclusions — runtime state that lives inside the harness
# tree but is not harness code. Mirrors finalize_pipeline.EXCLUDE_PATTERNS_RE.
RUNTIME_NOISE_DIRS = frozenset({"__pycache__", ".cache", ".pytest_cache"})
RUNTIME_NOISE_SUFFIXES = (".pyc", ".pyo", ".bak", "~")

CLEAN = "CLEAN"
DIRTY = "DIRTY"
UNKNOWN = "UNKNOWN"


@dataclass
class HarnessState:
    """Git state of the measuring code at the moment it was sampled."""

    state: str  # CLEAN | DIRTY | UNKNOWN
    dirty_paths: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_dirty(self) -> bool:
        return self.state == DIRTY


def _is_runtime_noise(path: str) -> bool:
    """True if `path` is runtime state rather than harness code.

    Dotfiles/dotdirs are excluded wholesale (`.kernel_worker_active`,
    `.optimizer_active`, `.opgen_state.json`, ...) — the orchestrator writes
    these into its own directory as it runs.
    """
    for component in path.split("/"):
        if not component:
            continue
        if component.startswith("."):
            return True
        if component in RUNTIME_NOISE_DIRS:
            return True
    return path.endswith(RUNTIME_NOISE_SUFFIXES)


def _parse_porcelain_path(line: str) -> Optional[str]:
    """Extract the path from one `git status --porcelain` line.

    Format is `XY <path>`; renames/copies render as `XY <old> -> <new>`, in
    which case the destination is what exists on disk now.
    """
    if len(line) < 4:
        return None
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path or None


def harness_state(project_root: Optional[Path] = None) -> HarnessState:
    """Sample the git state of the harness paths. Never raises.

    Returns HarnessState(CLEAN|DIRTY|UNKNOWN). UNKNOWN whenever we cannot
    get a trustworthy answer (not a git checkout, git absent, git errored,
    timeout) — the caller records it but must not treat it as a failure.
    """
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    try:
        git_executable = str(Path(shutil.which("git") or "git").resolve())
        proc = subprocess.run(
            [git_executable, "-C", str(root), "status", "--porcelain", "--", *HARNESS_PATHS],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return HarnessState(UNKNOWN, reason="git executable not found")
    except subprocess.TimeoutExpired:
        return HarnessState(UNKNOWN, reason="git status timed out after 30s")
    except Exception as e:  # pragma: no cover - defensive
        return HarnessState(UNKNOWN, reason=f"git status failed: {e}")

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        detail = stderr[0][:160] if stderr else f"rc={proc.returncode}"
        return HarnessState(UNKNOWN, reason=f"not a git checkout or git error: {detail}")

    dirty: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if not line.strip():
            continue
        path = _parse_porcelain_path(line)
        if path and not _is_runtime_noise(path):
            dirty.append(path)

    if dirty:
        return HarnessState(
            DIRTY,
            dirty_paths=sorted(dirty),
            reason=(
                f"{len(dirty)} uncommitted harness file(s) — the code that "
                f"measures this run is modified relative to HEAD"
            ),
        )
    return HarnessState(CLEAN, reason="harness tree matches HEAD")
