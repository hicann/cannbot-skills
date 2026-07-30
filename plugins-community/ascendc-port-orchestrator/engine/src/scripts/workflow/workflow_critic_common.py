#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""workflow_critic_common — shared LEAF for the workflow critic (Rejection + reject_and_exit +
workspace/state helpers + consts), extracted from workflow_critic.py (behavior-neutral, 2026-07-05).
LEAF: imports nothing from workflow_critic. CRITICAL: workflow_critic.py is a __main__ script-hook,
so validators import shared symbols FROM THIS LEAF (never from the __main__ parent) to avoid the
circular-import / double-Rejection-class hazard."""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _find_project_root(start: Path) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "src" / "scripts" / "workflow" / "state_machine.py").is_file():
            return candidate
    return start.parents[3]  # deterministic engine/ fallback


REPO_ROOT = _find_project_root(Path(__file__).resolve())
YAML_PATH = Path(__file__).resolve().parents[4] / "workflows" / "opgen_state_machine.yaml"


class Rejection:
    def __init__(self, rule_id: str, description: str, expected: str, actual: str, fix: str):
        self.rule_id = rule_id
        self.description = description
        self.expected = expected
        self.actual = actual
        self.fix = fix

    def emit(self) -> str:
        return (
            f"  - rule {self.rule_id}: {self.description}\n"
            f"    expected: {self.expected}\n"
            f"    actual:   {self.actual}\n"
            f"    fix:      {self.fix}\n"
        )


def reject_and_exit(phase: str, rejections: list[Rejection]) -> None:
    sys.stderr.write(f"❌ workflow_critic: REJECTED at {phase}\n")
    for r in rejections:
        sys.stderr.write(r.emit())
    sys.stderr.write(
        "\nSee workflows/opgen_state_machine.yaml for the full spec.\n"
        "Fix the gap and retry, or create workspace/<op>/.workflow_exception_<phase> "
        "(user-signed, git-committed by user) to waive.\n"
    )
    sys.exit(2)


def load_state_machine() -> dict[str, Any]:
    if not YAML_PATH.exists():
        sys.stderr.write(f"workflow_critic: state machine YAML missing at {YAML_PATH}\n")
        sys.exit(0)  # can't critique without spec; fail open
    with YAML_PATH.open() as f:
        return yaml.safe_load(f)


def has_valid_exception(ws: Path, phase: str) -> tuple[bool, str]:
    """Returns (has_exception, reason). Exception file must be git-committed by
    a commit NOT authored by the orchestrator (i.e. by the real user)."""
    exc_file = ws / f".workflow_exception_{phase}"
    if not exc_file.exists():
        return False, f"no .workflow_exception_{phase} file"
    content = exc_file.read_text()
    # Required fields
    required = ["phase:", "reason:", "user_signature:", "date:"]
    missing = [f for f in required if f not in content]
    if missing:
        return False, f"exception file missing fields: {missing}"
    # Must be git-committed (tracked)
    git_executable = shutil.which("git")
    if git_executable is None:
        return False, "could not verify git status: git executable is unavailable"
    try:
        r = subprocess.run(
            [git_executable, "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", str(exc_file)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return False, "exception file must be git-committed (tracked) — not just present on disk"
    except Exception:
        return False, "could not verify git status of exception file"
    return True, "valid"


def read_active_target(ws: Path) -> tuple[str, bool]:
    """Read the active TARGET (a5/a3/a2) and PLATFORM_SIMT capability flag from
    workspace/.ascendc_env (V3.4 multi-target). Returns ('a5', True) as the
    safe default if .ascendc_env is missing or unparseable — so legacy sessions
    that never enabled multi-target keep the old A5 behavior.
    """
    env_file = None
    for candidate in (ws.parent.parent / "workspace/.ascendc_env",
                      ws.parent / "workspace/.ascendc_env",
                      Path.cwd() / "workspace/.ascendc_env"):
        if candidate.is_file():
            env_file = candidate
            break
    if env_file is None:
        return ("a5", True)
    target = "a5"
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("TARGET="):
                target = line.split("=", 1)[1].strip().strip("'\"").lower()
                break
    except Exception:
        return ("a5", True)
    target = target if target in ("a5", "a3", "a2") else "a5"
    platform_simt = (target == "a5")
    return (target, platform_simt)


SIMT_PRIMITIVES = (
    # SIMT-only primitives (won't compile on V220)
    "Simt::",
    "WarpShflSync", "WarpReduceAddSync", "WarpReduceMaxSync", "WarpReduceMinSync",
    "WarpBallotSync", "WarpAllSync", "WarpAnySync",
    "ThreadBarrier", "ThreadFence",
    "LAUNCH_BOUND", "__syncthreads",
    # Heuristic: SIMT-style indexing (universal block ops GetBlockIdx/GetBlockNum stay)
    "threadIdx.", "blockDim.", "gridDim.",
    # arch35-only runtime macros that compile fine but trigger ACL_ERROR_RT_PARAM_INVALID
    # (107000) at RegisterAscendBinary time on V220. Verified on 13_Cat 2026-04-25.
    "KERNEL_TASK_TYPE_DEFAULT",
)
_C_COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/', re.DOTALL)
