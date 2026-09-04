# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Validation guards for migration mounts, repository state, and backward specs.

Mechanically extracted from orchestrator.py (god-file decomposition 2026-06-30, per
ORCHESTRATOR_REFACTOR_AND_UT_SPEC §1). Behavior unchanged — function bodies are VERBATIM.
Re-imported into orchestrator's namespace so existing call-sites and
`orchestrator.<name>` external access are preserved.

DAG: imports only stdlib (os/sys/re/subprocess/pathlib) — no orchestrator import → acyclic, imports
standalone. No module-level orchestrator globals consumed (self-contained).

Not moved (kept in core, see decompose_log.md):
- enforce_port_a3_target: carries a [baselined] CORE_MODE_LEAK
  (`opgen_mode == "port_a3_to_a5" and target != "a5"`); relocating trips the arch-lint ratchet.
- _is_legitimate_pipeline_exhaustion: [baselined] GOD_FUNCTION (206 lines); same ratchet reason.

MONKEYPATCH NOTE (durable — OL-160-class latent-coupling guard): the functions and module-level
constants/logger here are re-imported into orchestrator's namespace, which preserves
`orchestrator.<name>` attribute LOOKUP only — it does NOT rebind THIS module's own globals. A test
that overrides a symbol one of these functions reads must `monkeypatch.setattr(<this_module>,
'<name>', ...)` on THIS module, NOT on `orchestrator` (patching orchestrator silently misses the
binding used here). No current test patches these on orchestrator; this note prevents a future one
from a silent no-op.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _validate_a3_host_home_mount(a3_host: str, a3_container: str, expected_host_home: str,
                                 container_home: str = "/home/npu_user") -> int:
    """P129 (2026-05-17): docker-inspect gate validating sliced mount.

    `container_home` = the in-container canonical mount destination (the `.Destination`
    to inspect). CONFIG-DRIVEN — the caller passes `phase_o25_a3_ref._a3_container_home()`
    (reads `A3_CONTAINER_HOME` from `.ascendc_env`); the `/home/npu_user` default preserves
    the a5 behavior + the existing 3-arg callers/tests. Without this a scrubbed / non-npu_user
    deployment's docker-inspect would query a Destination the container doesn't have → false rc=18.
    (This module stays stdlib-only/self-contained — the config is INJECTED, not imported here.)

    The orchestrator's port_a3 phase scp-pushes workspace to
    ${A3_HOST_HOME}/workspace/a5_ops_a3_to_a5/workspace/<op> on the host,
    then ssh+docker-exec runs the runner expecting
    /home/npu_user/workspace/a5_ops_a3_to_a5/workspace/<op> inside the
    container. The two paths are the same physical location iff the
    container has mount `${A3_HOST_HOME}:/home/npu_user`. If someone
    recreated the container with `/home/npu_user:/home/npu_user`
    default mount instead, scp writes to a slice path the container
    can't see → runner cd fails with confusing error.

    Catch this at startup with an explicit `docker inspect` check.
    Returns 0 on success, 18 on mismatch (validation gate convention).
    """
    if not (a3_host and a3_container and expected_host_home):
        return 0  # caller already warned about missing fields
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8",
             "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{a3_host}",
             f"docker inspect {a3_container} --format "
             "'{{range .Mounts}}{{if eq .Destination \"" + container_home + "\"}}"
             "{{.Source}}{{end}}{{end}}'"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"WARN: P129 mount gate could not run docker inspect on "
                  f"{a3_host}:{a3_container} (rc={result.returncode}). "
                  f"Continuing — gate is advisory.")
            return 0
        actual = result.stdout.strip()
        if actual != expected_host_home:
            print(
                f"ERROR: P129 mount gate FAILED.\n"
                f"  Container '{a3_container}' on {a3_host} has\n"
                f"    {container_home} ← {actual!r}\n"
                f"  But .ascendc_env A3_HOST_HOME = {expected_host_home!r}.\n"
                f"\n"
                f"  Hint: someone recreated the container without the sliced "
                f"mount convention (see memory feedback_a3_per_agent_sliced_"
                f"container_convention). Either:\n"
                f"    (a) recreate container with `-v {expected_host_home}:{container_home}`\n"
                f"    (b) update workspace/.ascendc_env A3_HOST_HOME={actual} to match reality\n"
                f"\n"
                f"  See P126/P129 in ROADMAP and src/scripts/setup_a3_isolated_container.sh."
            )
            return 18
        return 0
    except subprocess.TimeoutExpired:
        print(f"WARN: P129 mount gate ssh timeout for {a3_host}. Continuing — gate advisory.")
        return 0
    except Exception as e:
        print(f"WARN: P129 mount gate raised {e!r}. Continuing — gate advisory.")
        return 0


def _refuse_if_detached() -> None:
    """Task #51 (2026-05-13): refuse to run as a detached/orphan process.

    Background: agents (both A5 and DS) repeatedly launched the orchestrator
    via `python3 orchestrator.py ... &` or `nohup python3 orchestrator.py ... &`,
    which reparents the process to init (PPID=1). The CC task tracker then
    can't see the orchestrator; the `& `-launched parent shell exits with 0
    and CC reports "completed" even though the work hasn't started. Result:
    no visibility, no `TaskStop`-ability, no ability for the user to inspect
    progress in the UI task list.

    Mechanical fix: at startup, if PPID==1 AND env knob ALLOW_DETACHED is
    unset, refuse to run with a clear error. This catches the error at the
    moment of launch regardless of agent discipline or hook state.

    Escape hatch: `ALLOW_DETACHED=1` env var allows truly-detached use
    (systemd / cron / debug). Default behavior is fail-loud.
    """
    if os.environ.get("ALLOW_DETACHED"):
        return
    if os.getppid() != 1:
        return
    print(
        "ERROR: orchestrator was launched as a detached/orphan process "
        "(PPID=1).",
        file=sys.stderr,
    )
    print(
        "Likely caused by trailing `&` or `nohup ...` in the launching "
        "shell command.",
        file=sys.stderr,
    )
    print(
        "Correct invocation from an agent:\n"
        "  Bash(command='python3 src/scripts/orchestrator/orchestrator.py "
        "--port-a3-ops <op_dir> --lane 0 --reference-source npubench "
        "--npubench-task <task.py>', run_in_background=True)\n"
        "  — NO trailing `&`, NO `nohup`, NO shell `> log 2>&1` redirect "
        "(bare --port-a3-ops without an explicit reference source fails closed).",
        file=sys.stderr,
    )
    print(
        "See the current customer entry Skills under ${CLAUDE_PLUGIN_ROOT}/skills/ "
        "for why this is the correct pattern (CC tracks the task across turns, no "
        "Bash-tool timeout applies to background tasks, process stays "
        "alive until natural completion or TaskStop).",
        file=sys.stderr,
    )
    print(
        "Override (truly-detached use like systemd / cron): set "
        "ALLOW_DETACHED=1 env var.",
        file=sys.stderr,
    )
    sys.exit(2)


def _spec_has_backward_contract(py_path: Path) -> bool:
    """True iff a .py defines BOTH a `forward(` and a `BACKWARD_SPEC` — i.e. a
    differentiable forward spec usable as the autograd reference for a backward port
    (the input phase_o25_backward.provision_backward_reference expects)."""
    try:
        txt = py_path.read_text()
    except Exception:
        return False
    return ("BACKWARD_SPEC" in txt) and bool(re.search(r"def\s+forward\b", txt))
