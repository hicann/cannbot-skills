# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Resolution helpers — env load, NPU lane detection, and op-name derivation.

Mechanically extracted from orchestrator.py (god-file decomposition 2026-06-30, per
ORCHESTRATOR_REFACTOR_AND_UT_SPEC §1). Behavior unchanged — function bodies are VERBATIM.
Re-imported into orchestrator's namespace (`from resolution import *names*`) so existing call-sites
and `orchestrator.<name>` external access are preserved.

DAG: this module imports only stdlib + the `logging_config` sibling, so it
imports standalone without pulling orchestrator.

The timeout constants are re-imported by the engine. Workspace identity is now
flat because only the AscendC backend is admitted.

MONKEYPATCH NOTE (durable — OL-160-class latent-coupling guard): the functions and module-level
constants/logger here are re-imported into orchestrator's namespace, which preserves
`orchestrator.<name>` attribute LOOKUP only — it does NOT rebind THIS module's own globals. A test
that overrides a symbol one of these functions reads must `monkeypatch.setattr(<this_module>,
'<name>', ...)` on THIS module, NOT on `orchestrator` (patching orchestrator silently misses the
binding used here). No current test patches these on orchestrator; this note prevents a future one
from a silent no-op.
"""
from __future__ import annotations
import logging

import re
import subprocess
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)

_DEFAULT_AGENT_TIMEOUT_SEC_A5 = 9000   # 150 min (Opus backend; was 5400 pre-input_gen)
_DEFAULT_AGENT_TIMEOUT_SEC_DS = 10800  # 180 min (V4 backend, P0aau 2026-05-09)


def _agent_timeout_for_target(target: str) -> int:
    """Return per-target agent timeout. DS backends get 3h; everything else 90min."""
    t = (target or "").lower()
    if "ds" in t or "v4" in t:
        return _DEFAULT_AGENT_TIMEOUT_SEC_DS
    return _DEFAULT_AGENT_TIMEOUT_SEC_A5


def _resolve_env():
    """Lazy-load the parsed workspace env for timeout dispatch."""
    try:
        from briefs._common import load_env
        return load_env()
    except Exception:
        return None


def _detect_max_lane() -> int:
    """P0aaz: dynamically detect max valid NPU lane via `npu-smi info`.

    Runs `npu-smi info` locally or via SSH (depending on target), parses
    the device table to count available NPUs, and returns the highest valid
    lane index (0-based). Falls back to per-target conservative defaults
    if `npu-smi` is unavailable.

    Local targets (host=localhost or empty): run directly.
    Remote targets: SSH + `npu-smi info`; if SSH fails, fall back to
    per-target default based on known hardware (a5=2, a3=2, a2=4).
    """
    try:
        from briefs._common import load_env as _env_loader
        env = _env_loader()
    except Exception:
        return 2  # conservative fallback: assume A5 (3 NPUs, max lane 2)

    target = (env.target or "a5").lower()
    if target.endswith("-ds"):
        target_base = target[:-3]
    else:
        target_base = target

    host = getattr(env, 'host', '') or ''
    user = getattr(env, 'user', '') or 'root'
    container = getattr(env, 'container', '') or ''

    # Precedence: try the target-specific env keys first, then fall back
    # to A5_* (legacy .ascendc_env format).
    try:
        target_upper = target_base.upper()
        host = (
            getattr(env, f"{target_upper}_HOST", None)
            or getattr(env, "A5_HOST", None)
            or host
        )
        user = (
            getattr(env, f"{target_upper}_USER", None)
            or getattr(env, "A5_USER", None)
            or user
        )
        container = (
            getattr(env, f"{target_upper}_CONTAINER", None)
            or getattr(env, "A5_CONTAINER", None)
            or container
        )
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )

    # Build command. Fix ①a (chair): the pre-existing probe ran bare
    # `npu-smi info` WITHOUT the NPU driver lib path, so on hosts that don't
    # already export it npu-smi fails to load `libdrvdsmi_host.so` → the probe
    # falls back to a WRONG lane cap. Prepend the driver lib dirs to
    # LD_LIBRARY_PATH. Prepend (NOT replace) and expand in the TARGET shell so
    # hosts where LD_LIBRARY_PATH already resolves npu-smi are unaffected
    # (behavior-neutral). NOTE: this only makes npu-smi RUNNABLE — it does NOT
    # touch the NPU-ID↔Phy-ID mapping (chair logged that as a separate harness
    # DEBT; do not attempt it here).
    _driver_libs = (
        "/usr/local/Ascend/driver/lib64/driver:"
        "/usr/local/Ascend/driver/lib64/common"
    )
    _smi = (
        f'LD_LIBRARY_PATH="{_driver_libs}'
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" npu-smi info'
    )
    is_local = not host or host in ("localhost", "127.0.0.1", "::1")

    if not is_local:
        ssh_args = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=no"]
        if user:
            ssh_args += [f"{user}@{host}"]
        else:
            ssh_args += [host]
        if container:
            remote = f"docker exec {container} bash -lc '{_smi}'"
        else:
            remote = f"bash -lc '{_smi}'"
        cmd = ssh_args + [remote]
    else:
        if container and container != "local":
            cmd = ["docker", "exec", container, "bash", "-lc", _smi]
        else:
            cmd = ["bash", "-lc", _smi]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"npu-smi exit {result.returncode}: {result.stderr[:200]}")
        output = result.stdout
    except Exception as e:
        # Fall back to per-target defaults
        # P135.LF (2026-05-18 task #16): A5 fallback bumped 2→4 (5 NPUs: 0-4)
        # because the current A5 hardware has 5 Ascend950PR NPUs but the
        # fallback assumed 3. When npu-smi probe fails (e.g., libc_sec.so
        # missing from orch host shell — observed 2026-05-18 with shared
        # A5 container), the cap=2 falsely blocked lanes 3+4 from being
        # used. erfinv cold-start hit this when lanes 0-2 were busy with
        # foreach_sqrt / foreach_neg / 11_DequantSwigluQuant.
        log.info(f"npu-smi probe failed ({e}); "
              f"falling back to default lane max for {target_base}")
        return {"a5": 4, "a3": 2, "a2": 4}.get(target_base, 2)

    # Parse `npu-smi info` output. The device table prints one device per
    # two rows; the FIRST row carries the numeric NPU ID and the chip-model
    # name. Two layouts exist across npu-smi versions:
    #   24.1.rc2 (older, no column pipe after ID):
    #       | 0     910B2C              | OK            | ...
    #   25.7.rc1 (newer, pipe-separated columns):
    #       | 0      | Ascend950PR      | OK            | ...
    # The old regex `^\|\s*(\d+)\s+\w+` only matched the first layout and
    # yielded 0 IDs on the second (the char after the ID's spaces is `|`,
    # not a word char) → "parse yielded 0 NPUs" → fallback cap of 4 → a
    # build-root lane like 6 was wrongly rejected (this bug). The regex
    # below accepts BOTH layouts: ID, spaces, an OPTIONAL column pipe, then
    # the NAME cell (everything up to the next `|`). Two constraints on the
    # name cell keep the match to EXACTLY the one device-name row per device
    # (verified: 1 match/device on all fixtures — no reliance on set-dedup):
    #   * must contain ≥1 ASCII letter  → the chip name (`Ascend950PR` /
    #     `910B2C`). Excludes a 25.7.rc1 process-table row whose 2nd column
    #     is a pure-numeric PID (`| 0 | 846718 |`).
    #   * must NOT contain ':'          → excludes the device's SECOND row
    #     whose 2nd column is the Bus-Id (`| 0 | 0000:DA:00.0 |`); hex
    #     letters in a Bus-Id would otherwise re-match the same ID. The ':'
    #     guard also rejects the "No running processes found" lines and any
    #     separator/header rows.
    npu_ids: set[int] = set()
    for line in output.splitlines():
        m = re.match(r'^\|\s*(\d+)\s+\|?\s*([^|:]*[A-Za-z][^|:]*)\|', line)
        if m:
            npu_ids.add(int(m.group(1)))
    if not npu_ids:
        log.info("npu-smi output parse yielded 0 NPUs; "
              "falling back to default lane max")
        return {"a5": 4, "a3": 2, "a2": 4}.get(target_base, 2)

    max_lane = max(npu_ids)
    log.info(f"detected {len(npu_ids)} NPU(s) via npu-smi; max lane = {max_lane}")
    return max_lane


def op_name_from_workspace(workspace: Path) -> str:
    """Return the operation name from a flat scoped workspace path."""
    return workspace.name
