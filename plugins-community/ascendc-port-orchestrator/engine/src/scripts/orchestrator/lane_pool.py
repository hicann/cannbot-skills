# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Lane pool — NPU lane allocation for parallel batch (DEBT-077 Track C #2).

A "lane" is an NPU id (0/1/2 by default — the original 3-NPU A5 box). Each
running orchestrator process owns one lane for its op's duration. The
lane_pool:

  1. Probes which NPUs are physically idle via npu-smi over ssh
  2. Tracks logical lane ownership in workspace/.lanes/lane_N/state files
     (JSON with {state, op, pid, ts})
  3. Atomically allocates a free lane (file rename = filesystem-level lock)
  4. Releases on op completion (or via cleanup script if process crashes)

Lane state file schema:
  workspace/.lanes/lane_<N>/state            (JSON)
  {
    "lane":   <int>,
    "state":  "free" | "busy",
    "op":     "<op name or null>",
    "pid":    <orchestrator pid or null>,
    "ts":     "<ISO 8601 Z>",
    "host":   "<a5_host>",
  }

Codex C5: "lane release best-effort at exit (wrap in try/except; if
lane_release.py not present, that's OK — orchestrator handles release on
its side)". Implemented via Python finally: clauses in batch dispatcher.

Phantom-lane policy (handover env-freeze): only the ids in VALID_LANES are
allocatable; anything else is refused for allocate / probe / list. The set
defaults to (0, 1, 2) for the 3-NPU A5 box and can be widened for machines
with more NPUs (e.g. an 8-NPU 950DT) via the AOG_LANE_IDS env var — a
comma-separated id list such as "0,1,2,3,4,5,6,7". A malformed value falls
back to the default rather than silently widening the pool.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_VALID_LANES = (0, 1, 2)


def _parse_valid_lanes(raw: str) -> tuple[int, ...]:
    """Parse the AOG_LANE_IDS override into the valid lane-id tuple.

    Accepts a comma-separated list of non-negative ids. Any malformed token
    or an empty result falls back to DEFAULT_VALID_LANES — never widen the
    pool on a configuration typo.
    """
    lanes: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            return DEFAULT_VALID_LANES
        lane = int(token)
        if lane not in lanes:
            lanes.append(lane)
    return tuple(sorted(lanes)) if lanes else DEFAULT_VALID_LANES


VALID_LANES = _parse_valid_lanes(os.environ.get("AOG_LANE_IDS", ""))


# ---------------------------------------------------------------------------
# Pool root resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent  # repo root
DEFAULT_POOL_ROOT = _PROJECT_ROOT / "src" / "workspace" / ".lanes"


def pool_root(override: Optional[Path] = None) -> Path:
    """Return the lane state-file root. Tests can override via arg."""
    return override or DEFAULT_POOL_ROOT


# ---------------------------------------------------------------------------
# npu-smi parsing — pure function, testable
# ---------------------------------------------------------------------------
@dataclass
class NpuStatus:
    npu_id: int
    util_pct: int
    has_running_process: bool


def parse_npu_smi(text: str) -> dict[int, NpuStatus]:
    """Parse `npu-smi info` stdout and return per-NPU status.

    Looks for two patterns:
      1. Util/memory line:  `| <id>      | Ascend...  | OK            | <power>     <temp>           ...`
                            `| .. | <id>:00.0 | <util>      <mem>...`
      2. Process table:     `| <id>                       | <pid>       | <name>...`
                            `| No running processes found in NPU <id>`

    Returns dict {npu_id: NpuStatus}.
    """
    result: dict[int, NpuStatus] = {}

    # Util line: appears 2 lines after the header; "| ... | <util>           <mem>..."
    # We just grab the first two-decimal integer after `|         | <bus> |` where bus
    # belongs to that NPU. Easier: NPU util always appears as "| <util>           <mem>"
    # following the header. Use stateful parsing.

    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\|\s*(\d+)\s+\|\s+Ascend", line)
        if not m:
            continue
        npu_id = int(m.group(1))
        if npu_id not in VALID_LANES:
            continue
        # Look at next 1-2 lines for the util row: "|        |  ...  | <util>     <mem> / <total>  ..."
        util_pct = 0
        for j in range(i + 1, min(i + 4, len(lines))):
            nl = lines[j]
            # Pattern: leading "|  ...  |  <bus> |  <util>           <mem-info>"
            um = re.match(r"^\|.*\|\s+\d+:\d+:\d+\.\d+\s*\|\s*(\d+)\s+", nl)
            if um:
                util_pct = int(um.group(1))
                break
        result[npu_id] = NpuStatus(npu_id=npu_id, util_pct=util_pct,
                                    has_running_process=False)

    # Process table
    proc_pat = re.compile(r"^\|\s*(\d+)\s+\|\s+(\d+)\s+\|", re.MULTILINE)
    no_proc_pat = re.compile(r"No running processes found in NPU\s+(\d+)")
    for m in proc_pat.finditer(text):
        npu_id = int(m.group(1))
        if npu_id in result:
            result[npu_id].has_running_process = True
    for m in no_proc_pat.finditer(text):
        npu_id = int(m.group(1))
        if npu_id in result:
            result[npu_id].has_running_process = False

    return result


def probe_idle_npus(text: str, *, util_threshold: int = 5) -> list[int]:
    """Return list of NPU ids that are idle (util ≤ threshold AND no running process).

    Pure-function variant — caller fetches the npu-smi text via ssh
    separately. Util threshold accommodates background measurement noise.
    """
    statuses = parse_npu_smi(text)
    return [
        npu_id for npu_id, s in sorted(statuses.items())
        if s.util_pct <= util_threshold and not s.has_running_process
    ]


def fetch_npu_smi(host: str, user: str, password: str,
                  *, timeout_sec: int = 30) -> str:
    """Run npu-smi info over ssh; return stdout.

    Raises subprocess.CalledProcessError on ssh failure. Caller decides
    how to handle (log + raise vs silently return empty).
    """
    cmd = [
        "sshpass", "-p", password,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{user}@{host}",
        "npu-smi info",
    ]
    completed = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec, check=True,
    )
    return completed.stdout


# ---------------------------------------------------------------------------
# Lane state files — atomic allocate/release
# ---------------------------------------------------------------------------
def _state_file(lane: int, root: Optional[Path] = None) -> Path:
    if lane not in VALID_LANES:
        raise ValueError(f"lane {lane} not in {VALID_LANES} (see AOG_LANE_IDS)")
    return pool_root(root) / f"lane_{lane}" / "state"


def _now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_state(lane: int, *, state: str, op: Optional[str], pid: Optional[int],
                host: Optional[str] = None, root: Optional[Path] = None) -> Path:
    """Write lane state file. Used by allocate/release; also useful in tests."""
    if state not in ("free", "busy"):
        raise ValueError(f"state must be free|busy, got {state!r}")
    sf = _state_file(lane, root)
    sf.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lane": lane, "state": state, "op": op, "pid": pid,
        "ts": _now_z(), "host": host,
    }
    sf.write_text(json.dumps(payload, indent=2))
    return sf


def read_state(lane: int, *, root: Optional[Path] = None) -> dict | None:
    """Read lane state. Returns None if file missing or malformed."""
    sf = _state_file(lane, root)
    if not sf.exists():
        return None
    try:
        return json.loads(sf.read_text())
    except Exception:
        return None


def list_lanes(root: Optional[Path] = None) -> dict[int, dict]:
    """Return {lane: state-dict} for all VALID_LANES.

    Lanes without state files are returned as {state: 'free', ...}.
    """
    out: dict[int, dict] = {}
    for ln in VALID_LANES:
        s = read_state(ln, root=root)
        if s is None:
            out[ln] = {"lane": ln, "state": "free", "op": None, "pid": None}
        else:
            out[ln] = s
    return out


def allocate_lane(op: str, *, idle_lanes: Optional[list[int]] = None,
                  pid: Optional[int] = None, host: Optional[str] = None,
                  root: Optional[Path] = None) -> Optional[int]:
    """Atomically allocate a free lane to `op`.

    Args:
        op: workspace name acquiring the lane.
        idle_lanes: NPU ids known to be hardware-idle (from probe_idle_npus).
                    If None, picks any logical-free lane regardless of NPU
                    occupancy (caller's risk).
        pid: orchestrator pid for the audit record. Defaults to os.getpid().
        host: A5 host for the audit record. Optional.
        root: pool root override (tests).

    Returns:
        the allocated lane id, or None if none available.

    Atomicity: writes state file with state="busy" only if it doesn't already
    exist or current state is "free". Race-condition window is small (single
    write); for true cross-process safety use os.O_EXCL with `open` (TODO if
    parallel batch from multiple machines becomes a thing).
    """
    if pid is None:
        pid = os.getpid()
    candidates = idle_lanes if idle_lanes is not None else list(VALID_LANES)
    for ln in candidates:
        if ln not in VALID_LANES:
            continue
        cur = read_state(ln, root=root)
        if cur is None or cur.get("state") == "free":
            write_state(ln, state="busy", op=op, pid=pid, host=host, root=root)
            return ln
    return None


def allocate_lanes(op: str, n: int, *, idle_lanes: Optional[list[int]] = None,
                   pid: Optional[int] = None, host: Optional[str] = None,
                   root: Optional[Path] = None) -> Optional[list[int]]:
    """Atomically allocate n free lanes to op (all-or-nothing).

    Collects n free candidates, then writes busy only if the full set is
    available. Fewer than n free candidates -> None with nothing written.
    Each allocated lane carries the same op/pid/host.

    n=1 delegates to allocate_lane; n=0 returns []. Single read-then-write
    window (same MVP caveat as allocate_lane -- a parallel multi-machine
    dispatcher needs os.O_EXCL / flock).
    """
    if n <= 0:
        return []
    if n == 1:
        ln = allocate_lane(op, idle_lanes=idle_lanes, pid=pid,
                           host=host, root=root)
        return [ln] if ln is not None else None
    candidates = idle_lanes if idle_lanes is not None else list(VALID_LANES)
    chosen: list[int] = []
    for ln in candidates:
        if ln not in VALID_LANES:
            continue
        if len(chosen) >= n:
            break
        cur = read_state(ln, root=root)
        if cur is None or cur.get("state") == "free":
            chosen.append(ln)
    if len(chosen) < n:
        return None
    if pid is None:
        pid = os.getpid()
    for ln in chosen:
        write_state(ln, state="busy", op=op, pid=pid, host=host, root=root)
    return chosen


def release_lane(lane: int, *, op: Optional[str] = None,
                 root: Optional[Path] = None) -> None:
    """Release a busy lane. Idempotent: releasing an already-free lane is fine.

    If op is given and doesn't match the recorded op, refuses to release
    (prevents accidental cross-op release on misconfigured calls).
    """
    cur = read_state(lane, root=root)
    if cur and op and cur.get("op") and cur["op"] != op:
        raise ValueError(
            f"release_lane: lane {lane} owned by op={cur['op']!r}, "
            f"caller passed op={op!r}; refusing"
        )
    write_state(lane, state="free", op=None, pid=None, root=root)


def release_lanes(lanes: list[int], *, op: Optional[str] = None,
                  root: Optional[Path] = None) -> None:
    """Release a group of lanes (inverse of allocate_lanes). Delegates to
    release_lane per lane (idempotent; op consistency checked per lane).
    """
    for ln in lanes:
        release_lane(ln, op=op, root=root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_status(root: Optional[Path] = None) -> int:
    lanes = list_lanes(root)
    print(f"{'lane':<6s} {'state':<8s} {'op':<30s} {'pid':<8s} {'ts':<25s}")
    print("-" * 80)
    for ln in sorted(lanes):
        s = lanes[ln]
        print(f"{ln:<6d} {s.get('state','?'):<8s} {str(s.get('op') or '-'):<30s} "
              f"{str(s.get('pid') or '-'):<8s} {s.get('ts','-'):<25s}")
    return 0


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="lane_pool — NPU lane allocator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show all lane state files")
    p = sub.add_parser("allocate", help="allocate a lane (test only)")
    p.add_argument("--op", required=True)
    p = sub.add_parser("release", help="release a lane")
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--op", default=None)
    args = ap.parse_args()

    if args.cmd == "status":
        sys.exit(_cli_status())
    elif args.cmd == "allocate":
        ln = allocate_lane(args.op)
        if ln is None:
            print("no free lane")
            sys.exit(2)
        print(f"allocated lane {ln}")
        sys.exit(0)
    elif args.cmd == "release":
        release_lane(args.lane, op=args.op)
        print(f"released lane {args.lane}")
        sys.exit(0)
