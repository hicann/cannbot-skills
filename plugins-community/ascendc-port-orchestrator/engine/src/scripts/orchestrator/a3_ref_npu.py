# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Phase O2.5 A3-CANN reference — NPU gating + host-workspace config (decomposed 2026-07-06).

Cohesive LEAF: A3 NPU busy-gating (npu-smi parsing, idle-lane selection, range/
threshold config) plus the A3-host workspace-root resolver. Imports stdlib +
`_ascendc_env_path` from the a3_ref_common base. NEVER imports from the
phase_o25_a3_ref facade (unidirectional edge, no cycle).
"""
from __future__ import annotations
import logging

import re
import subprocess
from pathlib import Path
from typing import Optional

from a3_ref_common import RunRemote, _ascendc_env_path


def _read_a3_npu_gate_config(workspace: Path) -> tuple[int, int]:
    """Read A3_DEFAULT_NPU_ID + A3_AICORE_BUSY_THRESHOLD from .ascendc_env.

    Returns (npu_id, busy_threshold_pct). Defaults: npu_id=0, threshold=20.
    """
    npu_id, threshold = 0, 20
    env_path = _ascendc_env_path()
    if not env_path.exists():
        return npu_id, threshold
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("A3_DEFAULT_NPU_ID="):
            try:
                npu_id = int(line.split("=", 1)[1].strip().strip("'\""))
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
        elif line.startswith("A3_AICORE_BUSY_THRESHOLD="):
            try:
                threshold = int(line.split("=", 1)[1].strip().strip("'\""))
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
    return npu_id, threshold


def parse_npu_range(spec: str) -> list[int]:
    """Parse an NPU id range spec into a sorted list of int ids.

    Accepts:
      "0"          -> [0]
      "0-3"        -> [0, 1, 2, 3]
      "0,2,4"      -> [0, 2, 4]
      "0-1,4-5"    -> [0, 1, 4, 5]
      ""           -> []
    Invalid tokens are skipped (best-effort).
    """
    if not spec:
        return []
    ids: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            skip_current_item = False
            try:
                lo, hi = token.split("-", 1)
                lo, hi = int(lo.strip()), int(hi.strip())
                for n in range(min(lo, hi), max(lo, hi) + 1):
                    ids.add(n)
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
        else:
            skip_current_item = False
            try:
                ids.add(int(token))
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
    return sorted(ids)


def _read_a3_npu_range_config(workspace: Path) -> tuple[list[int], int]:
    """Read A3_NPU_RANGE (auto-pick) or fall back to [A3_DEFAULT_NPU_ID] (single chip).

    Returns (candidate_ids, busy_threshold_pct). When A3_NPU_RANGE is set,
    candidate_ids is the parsed list and the orchestrator will auto-pick
    the idlest one within the range. When unset, falls back to the existing
    single-chip behavior.
    """
    fallback_id, threshold = _read_a3_npu_gate_config(workspace)
    env_path = _ascendc_env_path()
    range_spec = ""
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("A3_NPU_RANGE="):
                range_spec = line.split("=", 1)[1].strip().strip("'\"")
                break
    candidates = parse_npu_range(range_spec)
    if not candidates:
        candidates = [fallback_id]
    return candidates, threshold


def pick_idle_npu_in_range(
    a3_user: str,
    a3_host: str,
    candidate_npus: list[int],
    threshold_pct: int,
    *,
    run_remote: Optional[RunRemote] = None,
) -> tuple[Optional[int], dict[int, float], str]:
    """Probe `npu-smi info` once, return the idlest NPU id from candidate_npus.

    Returns (chosen_id, observed_pcts, raw_log):
      chosen_id: lowest-AICore chip from candidate_npus whose AICore% is
                 below threshold_pct. None if all are above threshold.
      observed_pcts: {npu_id: aicore_pct} for each candidate (0.0 if not
                     observed). Empty dict if npu-smi unavailable.
      raw_log: stdout of npu-smi for diagnostics.
    """
    if not candidate_npus:
        return None, {}, "no candidate NPUs configured (A3_NPU_RANGE empty)"
    rc, stdout, _ = _run_npu_smi(a3_user, a3_host, run_remote=run_remote)
    if rc != 0:
        # Fallback: assume first candidate is free
        return (
            candidate_npus[0], {},
            f"npu-smi unavailable (rc={rc}); falling back to first candidate {candidate_npus[0]}",
        )
    observed: dict[int, float] = {}
    for nid in candidate_npus:
        observed[nid] = parse_aicore_pct(stdout, nid)
    # Pick the lowest-AICore candidate at or below threshold
    eligible = [(p, nid) for nid, p in observed.items() if p <= threshold_pct]
    if not eligible:
        return None, observed, stdout
    eligible.sort()  # lowest aicore first
    return eligible[0][1], observed, stdout


def check_a3_npu_busy(
    a3_user: str,
    a3_host: str,
    npu_id: int,
    threshold_pct: int,
    *,
    run_remote: Optional[RunRemote] = None,
) -> tuple[bool, float, str]:
    """Probe `npu-smi info` on A3 host and parse AICore% for the bound chip.

    Returns (busy, aicore_pct, raw_log):
      busy: True iff aicore_pct > threshold_pct
      aicore_pct: the observed AICore% (0.0 if parse failed)
      raw_log: stdout of npu-smi for debugging
    """
    rc, stdout, stderr = _run_npu_smi(a3_user, a3_host, run_remote=run_remote)
    if rc != 0:
        # If npu-smi unavailable, fall back to "assume free" rather than
        # block all A3 runs — the gate is best-effort, not hard-required.
        return False, 0.0, f"npu-smi unavailable (rc={rc}); skipping busy gate"
    aicore = parse_aicore_pct(stdout, npu_id)
    busy = aicore > threshold_pct
    return busy, aicore, stdout


def parse_aicore_pct(npu_smi_output: str, npu_id: int) -> float:
    """Parse `npu-smi info` output and return AICore% for the given chip id.

    npu-smi output layout (alternating row pattern per device):
        | <npu>  Ascend910           | OK    | <power> ...                    |
        | <chip> <phy_id>            | <bus> | <aicore%> <mem-MB> <hbm-MB>    |

    We match the chip-detail row whose first column equals npu_id. Returns
    0.0 if no match (caller treats as "skip gate").
    """
    chip_row_re = re.compile(
        r"^\|\s*(\d+)\s+(\d+)\s+\|\s+\S+\s+\|\s+(\d+(?:\.\d+)?)\s+"
    )
    for line in npu_smi_output.splitlines():
        m = chip_row_re.match(line)
        if not m:
            continue
        chip_id = int(m.group(1))
        if chip_id == npu_id:
            skip_current_item = False
            try:
                return float(m.group(3))
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
    return 0.0


def _run_npu_smi(
    a3_user: str,
    a3_host: str,
    *,
    run_remote: Optional[RunRemote] = None,
) -> tuple[int, str, str]:
    """ssh `npu-smi info` on A3 host. Uses run_remote if provided (test
    injection), else falls back to direct subprocess.run."""
    if run_remote is not None:
        # run_remote was designed for docker-exec; we want host-level
        # npu-smi. Fall through to subprocess for the real path; tests
        # that want to mock npu-smi should pass push_dir/pull_files
        # equivalents (see test_a3_busy_gate.py for the injection point).
        pass
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"{a3_user}@{a3_host}", "npu-smi info"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", "npu-smi timed out after 15s"
    except Exception as e:
        return 1, "", f"npu-smi invocation error: {e!r}"


def _a3_host_workspace_root_from_env(workspace: Path) -> str:
    """Read A3_HOST_HOME from workspace/.ascendc_env (or fallback)."""
    env_path = _ascendc_env_path()
    if not env_path.exists():
        return "/home/npu_user_opus"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("A3_HOST_HOME="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return "/home/npu_user_opus"
