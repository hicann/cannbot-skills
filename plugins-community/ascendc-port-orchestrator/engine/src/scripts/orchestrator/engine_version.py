# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Engine version drift guard (2026-08-27 — 'KeyError: binding_sha256' incident).

The orchestrator is a LONG-RUNNING process: its Python modules are loaded once
at start. Per-round child processes (O5 npubench runner, graybox runtime
staging) are spawned fresh from the worktree on disk. If the engine code
changes on disk mid-session (e.g. a collaborator pushes a new evidence scheme),
the parent and its children run different contract versions and fail with
confusing errors — on 2026-08-26, lanes started 19:10/19:47 kept an old
evidence scheme in memory while the O5 children picked up the v5/v6 scheme
committed 20:00-20:24, and five O5 rounds burned on 'KeyError:
binding_sha256' before the skew was found.

Guard: capture an engine fingerprint once at process start; the driver loop
re-checks it every iteration and exits with the dedicated code EXIT_DRIFT
when the engine moved on disk. Restarting the orchestrator (resume) loads the
current engine — that is the only correct recovery.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

# Dedicated process exit code: the engine moved on disk while this orchestrator
# was running. Operators/agents: resume the op to reload the current engine.
EXIT_DRIFT = 77


def _plugin_root() -> Path:
    # this file lives at <plugin_root>/engine/src/scripts/orchestrator/
    return Path(__file__).resolve().parents[4]


def _engine_dirs(plugin_root: Path):
    """Directories that define engine CONTRACT behavior (not KB data)."""
    engine = plugin_root / "engine" / "src" / "scripts"
    out = []
    for rel in ("orchestrator", "workflow", "okf", "patches"):
        candidate = engine / rel
        if candidate.is_dir():
            out.append(candidate)
    return out


def capture(plugin_root: Optional[Path] = None) -> dict[str, Any]:
    """Fingerprint the engine tree plus git HEAD. Content-addressed (not mtime)."""
    pr = Path(plugin_root) if plugin_root is not None else _plugin_root()
    digest = hashlib.sha256()
    files = []
    for directory in _engine_dirs(pr):
        for f in sorted(directory.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            try:
                data = f.read_bytes()
            except OSError:
                continue
            rel = f.relative_to(pr).as_posix()
            files.append(rel)
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(len(data)).encode("ascii"))
            digest.update(b"\0")
            digest.update(data)
            digest.update(b"\0")
    git_head: Optional[str] = None
    try:
        git_executable = str(Path(shutil.which("git") or "git").resolve())
        completed = subprocess.run(
            [git_executable, "-C", str(pr), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if completed.returncode == 0:
            git_head = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        git_head = None
    return {
        "plugin_root": str(pr),
        "engine_tree_sha256": digest.hexdigest(),
        "git_head": git_head,
        "file_count": len(files),
    }


def drift(captured: dict[str, Any], plugin_root: Optional[Path] = None) -> tuple[bool, str]:
    """Return (drifted, reason). Recompute the fingerprint and compare."""
    try:
        now = capture(Path(captured.get("plugin_root")) if plugin_root is None else plugin_root)
    except Exception as exc:  # noqa: BLE001 — fail-safe: report drift rather than crash
        return True, f"engine fingerprint recompute failed: {exc!r}"
    if captured.get("engine_tree_sha256") != now.get("engine_tree_sha256"):
        return True, (
            f"engine tree changed on disk since orchestrator start "
            f"(files {captured.get('file_count')} -> {now.get('file_count')}; "
            f"tree {str(captured.get('engine_tree_sha256'))[:12]} -> "
            f"{str(now.get('engine_tree_sha256'))[:12]})"
        )
    old_head = captured.get("git_head")
    new_head = now.get("git_head")
    if old_head and new_head and old_head != new_head:
        return True, f"engine git HEAD moved {old_head[:8]} -> {new_head[:8]}"
    return False, ""
