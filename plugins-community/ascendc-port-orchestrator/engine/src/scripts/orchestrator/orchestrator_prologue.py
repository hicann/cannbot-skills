# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Scoped workspace-resolution, audit, and timing helpers."""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)
_ORCH_MARKER = "run_single_op"


def _orch():
    """Return the live orchestrator module so test monkeypatches still bite."""
    for name in ("orchestrator", "orchestrator.orchestrator", "__main__"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, _ORCH_MARKER):
            return module
    return sys.modules.get("orchestrator") or sys.modules["__main__"]


def _workspace_name_for_backend(op: str, backend: str = "ascendc") -> str:
    """Return the flat workspace name for the sole supported backend."""
    if backend != "ascendc":
        raise ValueError("only the AscendC backend is supported")
    return op


def _resolve_workspace(op: str, backend: str = "ascendc") -> Path:
    """Find a flat scoped workspace by exact, lowercase, or numbered name."""
    workspace_root = _orch().WORKSPACE_ROOT
    name = _workspace_name_for_backend(op, backend)
    direct = workspace_root / name
    if direct.exists():
        return direct
    lower = workspace_root / name.lower()
    if lower.exists():
        return lower
    if not workspace_root.exists():
        return direct
    base_lower = op.lower()
    for directory in workspace_root.iterdir():
        if not directory.is_dir():
            continue
        candidate = directory.name.lower()
        if candidate == base_lower or candidate.endswith("_" + base_lower):
            return directory
    return direct


def _audit_bump_caps(workspace: Path, cap_bumps: dict[str, int]) -> None:
    """Append an explicit iteration-cap adjustment to the workspace audit."""
    if not cap_bumps:
        return
    log_path = workspace / ".cap_bumps.jsonl"
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "bumps": cap_bumps,
        "actor": "user_cli",
        "rationale": "user-explicit --bump-cap; not auto-applied by LLM/agent",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as stream:
        stream.write(json.dumps(entry) + "\n")


def _generate_timing_report(workspace: Path, op: str) -> None:
    """Generate TIMING_REPORT.md at a terminal state when requested."""
    script = _orch().PROJECT_ROOT / "scripts" / "gen_timing_report.py"
    if not script.exists():
        log.info("--timing: gen_timing_report.py not found at %s; skipping", script)
        return
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, str(script), str(workspace), op],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            log.info("--timing: TIMING_REPORT.md written (%s)", workspace / "TIMING_REPORT.md")
        else:
            log.info(
                "--timing: report gen failed (exit %s): %s",
                result.returncode,
                result.stderr[:200],
            )
    except Exception as exc:
        log.info("--timing: failed to generate report: %r", exc)
