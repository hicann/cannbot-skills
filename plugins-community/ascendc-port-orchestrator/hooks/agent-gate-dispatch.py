#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Route plugin-level hooks to the gate set for the active aog-* agent.

Claude Code ignores ``hooks:`` embedded in agents delivered by a plugin.  The
plugin hook surface is therefore the single runtime registration point; this
dispatcher uses the hook payload's ``agent_type`` to preserve the original
per-agent SubagentStop and PreToolUse contracts without running every gate for
every agent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


HOOK_ROOT = Path(__file__).resolve().parent / "v3"
STOP_GATES = {
    "aog-kernel-worker": ("check_worker.sh", "check_progress_signed.sh"),
    "aog-kernel-optimizer": (
        "check_optimizer_artifacts.sh",
        "check_progress_signed.sh",
    ),
    "aog-fused-optimizer": (
        "check_fused_optimizer_artifacts.sh",
        "check_progress_signed.sh",
    ),
    "aog-precision-probe": ("check_probe_report.sh", "check_progress_signed.sh"),
}
PRETOOL_GATES = {"aog-kernel-worker": ("block_edit_on_infra.sh",)}


def _payload(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _agent_type(payload: dict) -> str:
    value = (
        payload.get("agent_type")
        or payload.get("agent_name")
        or os.environ.get("CLAUDE_AGENT_TYPE", "")
    )
    # Accept a plugin-qualified value while keeping the on-disk names canonical.
    return str(value).rsplit(":", 1)[-1]


def _run(script_name: str, raw: str) -> int:
    script = HOOK_ROOT / script_name
    if not script.is_file():
        print(f"missing packaged agent gate: {script}", file=sys.stderr)
        return 2
    command = [str(script)]
    if os.name == "nt":
        command = [str(HOOK_ROOT.parent / "run-hook.cmd"), f"v3/{script_name}"]
    result = subprocess.run(
        command,
        input=raw,
        text=True,
        cwd=os.getcwd(),
        env=os.environ.copy(),
        capture_output=True,
        check=False,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in {"stop", "pretool"}:
        print("usage: agent-gate-dispatch.py {stop|pretool}", file=sys.stderr)
        return 2
    raw = sys.stdin.read()
    agent = _agent_type(_payload(raw))
    gates = (STOP_GATES if args[0] == "stop" else PRETOOL_GATES).get(agent, ())
    for script_name in gates:
        rc = _run(script_name, raw)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
