# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Orchestrator recovery — diagnose + clean up interrupted op-gen / opt runs.

Origin: 2026-05-04 batch5 op#1_gelu ko-2 quota crash exposed a zombie pattern
where:
  1. claude --print writes the result envelope (is_error=true / api_error_status=429
     for quota; or is_error=true for other transient errors) BUT the claude
     main process does NOT exit
  2. claude has long-lived MCP children (discord, playwright/bun) that block
     SIGCHLD propagation
  3. The orchestrator subprocess sits in pipe_read forever waiting for the
     SIGCHLD that never comes
  4. Parent batch.py polls the dead-but-alive orchestrator forever
  5. .agent_died_at_<state> markers from earlier failed runs accumulate and
     confuse resume.py's diagnose() (it reports agent_died for ops that have
     since had successful follow-up activity)

This module diagnoses + safely cleans up. It is called from the
`/aog-orchestrator-recover` skill.

Design principles:
- DIAGNOSE-FIRST: default mode is read-only. --kill / --clean-stale required
  to mutate state.
- IDEMPOTENT: running twice is safe. Cleanup detects already-cleaned items.
- AUDIT: every kill or marker-clear writes to workspace/.recover_log.jsonl
  (per workspace) AND a fleet-level src/workspace/.recover_log.jsonl

CLI:
  python3 -m recover                      # diagnose
  python3 -m recover --kill                # kill any zombies
  python3 -m recover --clean-stale-markers # clear stale .agent_died_at_*
  python3 -m recover --resume              # after kill, run resume.py --all
  python3 -m recover --json                # machine-readable
"""
from __future__ import annotations
import logging

import argparse
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import resume as resume_mod
import state_executor
from backends import get_backend

# recover's proc-identify/op-extract is a per-harness coupling (survey #2) → owned by the Backend,
# not hardcoded here. CCBackend.identify_cmd/parse_op_from_cmd mirror the prior inline logic verbatim.
_backend = get_backend()


class ProcKind(Enum):
    BATCH_DISPATCHER = "batch_dispatcher"      # python3 orchestrator.py --batch ...
    SINGLE_OP_ORCHESTRATOR = "single_op_orch"  # python3 orchestrator.py <op> ...
    HARNESS_AGENT = "harness_agent"            # active backend process (Claude Code/Codex/opencode)
    CLAUDE_AGENT = "harness_agent"             # backward-compatible alias
    OTHER = "other"


class ProcHealth(Enum):
    HEALTHY = "healthy"          # actively producing tool_use events / log lines
    QUIET = "quiet"              # alive but no recent activity (could be deep in agent thinking)
    ZOMBIE = "zombie"            # result envelope written but process not exited
    ORPHAN = "orphan"            # parent dead, child alive (rare)
    UNKNOWN = "unknown"


@dataclass
class ProcInfo:
    pid: int
    ppid: int
    cmd: str
    elapsed_sec: int
    kind: ProcKind
    health: ProcHealth
    op: Optional[str] = None        # op name if derivable
    workspace: Optional[Path] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class StaleMarker:
    workspace: Path
    op: str
    marker_path: Path
    marker_ts: str
    last_state_log_ts: Optional[str]
    age_minutes: float
    is_stale: bool                  # True if state log shows newer activity
    reason: str


@dataclass
class RecoverReport:
    procs: list[ProcInfo] = field(default_factory=list)
    stale_markers: list[StaleMarker] = field(default_factory=list)
    actions_taken: list[dict] = field(default_factory=list)
    resumable_ops: list[str] = field(default_factory=list)
    pending_user_decision: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Process scan
# ---------------------------------------------------------------------------
def _ps_axo() -> list[dict]:
    """ps aux as list of dicts. Cross-platform safe (linux ps fields)."""
    out = subprocess.run(
        ["ps", "-eo", "pid,ppid,etime,cmd", "--no-headers"],
        capture_output=True, text=True, check=False,
    )
    procs = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        elapsed = _parse_etime(parts[2])
        cmd = parts[3]
        procs.append({"pid": pid, "ppid": ppid, "elapsed": elapsed, "cmd": cmd})
    return procs


def _parse_etime(s: str) -> int:
    """ps etime format: [DD-]HH:MM:SS or MM:SS. Returns total seconds."""
    s = s.strip()
    if not s:
        return 0
    days = 0
    if "-" in s:
        d, rest = s.split("-", 1)
        days = int(d)
        s = rest
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = map(int, parts)
    elif len(parts) == 2:
        h, m, sec = 0, int(parts[0]), int(parts[1])
    else:
        return 0
    return days * 86400 + h * 3600 + m * 60 + sec


def _classify_proc(p: dict) -> ProcKind:
    cmd = p["cmd"]
    if _backend.identify_cmd(cmd):
        return ProcKind.HARNESS_AGENT
    if "orchestrator.py --batch" in cmd or "orchestrator.py  --batch" in cmd:
        return ProcKind.BATCH_DISPATCHER
    # Single-op: orchestrator.py <op_name> --lane N
    if re.search(r"orchestrator\.py\s+\S+\s+(--lane|-l)", cmd):
        return ProcKind.SINGLE_OP_ORCHESTRATOR
    return ProcKind.OTHER


def _extract_op_from_cmd(cmd: str) -> Optional[str]:
    """For orchestrator.py / claude --print, extract op name from cmd line."""
    # orchestrator.py <op> --lane N
    m = re.search(r"orchestrator\.py\s+(\S+)\s+--lane", cmd)
    if m:
        return m.group(1)
    # Backend-owned op-slug parsing (survey #2: recover coupling → Backend).
    return _backend.parse_op_from_cmd(cmd)


def _resolve_workspace(op: Optional[str]) -> Optional[Path]:
    if not op:
        return None
    root = _HERE.parent.parent.parent.parent / "workspace"
    direct = root / op
    if direct.exists():
        return direct
    for d in root.iterdir():
        if d.is_dir() and d.name.lower() == op.lower():
            return d
        if d.is_dir() and d.name.lower().endswith("_" + op.lower()):
            return d
    return None


def _classify_health(p: dict, kind: ProcKind, workspace: Optional[Path]) -> tuple[ProcHealth, list[str]]:
    """Classify health based on:
    - claude: result envelope written? + recent stream log activity?
    - orchestrator: events log activity? running with claude grandchild?
    - batch: any pending op subprocess?
    """
    notes: list[str] = []
    pid = p["pid"]
    elapsed = p["elapsed"]

    if kind == ProcKind.HARNESS_AGENT:
        # Check if result envelope already written to its stream log.
        # Heuristic: workspace has a .cc_stream_log_*.jsonl whose tail has type=result
        if workspace and workspace.exists():
            for sl in sorted(workspace.glob(".cc_stream_log_*.jsonl"),
                              key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    last_line = ""
                    with open(sl, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        if size == 0:
                            continue
                        f.seek(max(0, size - 8192))
                        last_line = f.read().decode("utf-8", errors="replace").splitlines()[-1]
                    if last_line.strip():
                        evt = json.loads(last_line)
                        if evt.get("type") == "result":
                            notes.append(
                                f"stream_log {sl.name} ends with result event "
                                f"(is_error={evt.get('is_error')}, "
                                f"api_error_status={evt.get('api_error_status')})"
                            )
                            return ProcHealth.ZOMBIE, notes
                except Exception as error:
                    logging.getLogger(__name__).debug(
                        "Recoverable operation failed.", exc_info=error
                    )
        # No result envelope but elapsed huge → likely just slow agent
        if elapsed > 5400:  # > 90 min
            notes.append(f"elapsed {elapsed}s > 90min wall-clock cap")
            return ProcHealth.QUIET, notes
        return ProcHealth.HEALTHY, notes

    if kind == ProcKind.SINGLE_OP_ORCHESTRATOR:
        # Healthy if has a claude grandchild
        try:
            children = subprocess.run(
                ["pgrep", "-P", str(pid)],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            for cpid_s in children:
                # Check if it's a claude process
                cmd = subprocess.run(
                    ["ps", "-p", cpid_s, "-o", "cmd", "--no-headers"],
                    capture_output=True, text=True, check=False,
                ).stdout
                if _backend.identify_cmd(cmd):
                    return ProcHealth.HEALTHY, ["has live backend grandchild"]
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
        notes.append("no backend grandchild")
        return ProcHealth.QUIET, notes

    if kind == ProcKind.BATCH_DISPATCHER:
        # Healthy = has live single-op orchestrator children
        try:
            children = subprocess.run(
                ["pgrep", "-P", str(pid)],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            for cpid_s in children:
                cmd = subprocess.run(
                    ["ps", "-p", cpid_s, "-o", "cmd", "--no-headers"],
                    capture_output=True, text=True, check=False,
                ).stdout
                if "orchestrator.py" in cmd and "--lane" in cmd:
                    return ProcHealth.HEALTHY, ["has live op orchestrator child"]
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
        notes.append("no live op orchestrator child")
        return ProcHealth.QUIET, notes

    return ProcHealth.UNKNOWN, notes


def scan_processes() -> list[ProcInfo]:
    """Scan ps for all relevant processes; classify kind + health."""
    procs = _ps_axo()
    out: list[ProcInfo] = []
    for p in procs:
        kind = _classify_proc(p)
        if kind == ProcKind.OTHER:
            continue
        op = _extract_op_from_cmd(p["cmd"])
        workspace = _resolve_workspace(op)
        health, notes = _classify_health(p, kind, workspace)
        out.append(ProcInfo(
            pid=p["pid"], ppid=p["ppid"],
            cmd=p["cmd"][:200],
            elapsed_sec=p["elapsed"],
            kind=kind, health=health,
            op=op, workspace=workspace, notes=notes,
        ))
    return out


# ---------------------------------------------------------------------------
# Stale marker detection
# ---------------------------------------------------------------------------
def find_stale_markers() -> list[StaleMarker]:
    """Walk workspace/ for .agent_died_at_* markers; classify stale vs current.

    Stale = state_transitions.jsonl has activity newer than the marker.
    Current = no newer activity (real failure standing).
    """
    root = _HERE.parent.parent.parent.parent / "workspace"
    if not root.exists():
        return []
    out: list[StaleMarker] = []
    for ws in sorted(root.iterdir()):
        if not ws.is_dir():
            continue
        for marker in ws.glob(".agent_died_at_*"):
            # Skip already-cleaned markers (idempotency)
            if ".cleaned-" in marker.name:
                continue
            try:
                payload = json.loads(marker.read_text())
                marker_ts = payload.get("ts", "")
            except Exception:
                marker_ts = ""
            mtime = marker.stat().st_mtime
            age_min = (time.time() - mtime) / 60

            log = ws / "state_transitions.jsonl"
            last_log_ts = None
            is_stale = False
            reason = ""
            if log.exists() and log.stat().st_mtime > mtime + 30:
                # state log activity AFTER the marker — stale
                is_stale = True
                reason = (f"state_transitions.jsonl mtime is "
                          f"{(log.stat().st_mtime - mtime):.0f}s newer than marker")
                # Get last log entry ts
                try:
                    lines = [l for l in log.read_text().splitlines() if l.strip()]
                    if lines:
                        last_log_ts = json.loads(lines[-1]).get("ts")
                except Exception as error:
                    logging.getLogger(__name__).debug(
                        "Recoverable operation failed.", exc_info=error
                    )
            elif age_min > 60 * 6:  # > 6h old marker
                is_stale = True
                reason = f"marker is {age_min:.0f}min old (> 6h threshold)"
            else:
                reason = "marker recent + no newer state log activity"

            out.append(StaleMarker(
                workspace=ws, op=ws.name, marker_path=marker,
                marker_ts=marker_ts, last_state_log_ts=last_log_ts,
                age_minutes=age_min, is_stale=is_stale, reason=reason,
            ))
    return out


# ---------------------------------------------------------------------------
# Actions: kill zombies + clean stale markers
# ---------------------------------------------------------------------------
def kill_zombie(proc: ProcInfo, *, dry_run: bool = False) -> dict:
    """SIGTERM zombie process tree.

    Strategy from batch5 finding: the backend CLI can fail to respond to
    SIGCHLD properly. Kill the bash-poller grandchild first (this often makes
    the backend process notice the result envelope was written and exit), then
    SIGTERM the backend process.
    """
    actions = []
    if proc.health != ProcHealth.ZOMBIE:
        return {"skipped": f"not a zombie: health={proc.health.value}"}

    if proc.kind != ProcKind.HARNESS_AGENT:
        return {"skipped": f"kill strategy only applies to harness_agent, got {proc.kind.value}"}

    # Find any bash polling grandchildren
    try:
        children = subprocess.run(
            ["pgrep", "-P", str(proc.pid)],
            capture_output=True, text=True, check=False,
        ).stdout.split()
        bash_pids = []
        for cpid_s in children:
            cmd = subprocess.run(
                ["ps", "-p", cpid_s, "-o", "cmd", "--no-headers"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            # bash polling pattern from batch5: until ... do sleep 5; done
            if "until" in cmd and "sleep" in cmd:
                bash_pids.append(int(cpid_s))
        for bp in bash_pids:
            actions.append(f"SIGTERM bash poller pid={bp}")
            if not dry_run:
                try:
                    os.kill(bp, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if bash_pids and not dry_run:
            time.sleep(2)  # let SIGCHLD propagate
    except Exception as e:
        actions.append(f"WARN: failed listing children of {proc.pid}: {e}")

    # SIGTERM the backend process itself.
    actions.append(f"SIGTERM backend pid={proc.pid}")
    if not dry_run:
        try:
            os.kill(proc.pid, signal.SIGTERM)
            time.sleep(2)
            # Check if still alive; SIGKILL if so
            try:
                os.kill(proc.pid, 0)
                actions.append(f"SIGKILL backend pid={proc.pid} (SIGTERM ignored)")
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # already dead
        except ProcessLookupError:
            actions.append("already dead by the time we got here")

    return {"pid": proc.pid, "kind": proc.kind.value, "actions": actions,
            "dry_run": dry_run}


def clean_stale_markers(stale: list[StaleMarker], *, dry_run: bool = False) -> list[dict]:
    """Remove stale .agent_died_at_* markers (move to .agent_died_at_*.cleaned-<ts>)."""
    actions = []
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for sm in stale:
        if not sm.is_stale:
            continue
        action = {
            "op": sm.op,
            "marker": str(sm.marker_path),
            "reason": sm.reason,
            "dry_run": dry_run,
        }
        if not dry_run:
            new_path = sm.marker_path.with_suffix(
                sm.marker_path.suffix + f".cleaned-{ts}"
            )
            try:
                sm.marker_path.rename(new_path)
                action["renamed_to"] = str(new_path)
            except Exception as e:
                action["error"] = str(e)
        actions.append(action)
    return actions


def _audit_log(workspace: Optional[Path], action: dict) -> None:
    """Append to workspace/.recover_log.jsonl (or fleet-level if no workspace)."""
    if workspace is None:
        log = _HERE.parent.parent.parent.parent / "src" / "workspace" / ".recover_log.jsonl"
    else:
        log = workspace / ".recover_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        **action,
    }
    with open(log, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Top-level recover()
# ---------------------------------------------------------------------------
def recover(*, kill: bool = False, clean_stale: bool = False,
            run_resume: bool = False, dry_run: bool = False) -> RecoverReport:
    rep = RecoverReport()

    # 1. Scan procs
    rep.procs = scan_processes()

    # 2. Find stale markers
    rep.stale_markers = find_stale_markers()

    # 3. Take action: kill zombies
    if kill:
        for p in rep.procs:
            if p.health == ProcHealth.ZOMBIE:
                action = kill_zombie(p, dry_run=dry_run)
                rep.actions_taken.append(action)
                _audit_log(p.workspace, {"action": "kill_zombie", **action})

    # 4. Take action: clean stale markers
    if clean_stale:
        cleaned = clean_stale_markers(rep.stale_markers, dry_run=dry_run)
        rep.actions_taken.extend(cleaned)
        for a in cleaned:
            ws = _resolve_workspace(a.get("op"))
            _audit_log(ws, {"action": "clean_stale_marker", **a})

    # 5. Identify resumable + pending-user-decision ops
    if run_resume or kill or clean_stale:
        statuses = resume_mod.scan_all()
        for s in statuses:
            if s.action in (resume_mod.ResumeAction.RESUMABLE,
                             resume_mod.ResumeAction.USER_DECISION_READY):
                rep.resumable_ops.append(s.op)
            elif s.action == resume_mod.ResumeAction.USER_DECISION_PENDING:
                rep.pending_user_decision.append(s.op)

    # 6. Optional: invoke resume.py --all (kicks off resumable ops)
    if run_resume and not dry_run and rep.resumable_ops:
        for op in rep.resumable_ops:
            sub_action = {"action": "resume.execute", "op": op}
            try:
                rc = resume_mod.execute(op, dry_run=False)
                sub_action["exit_code"] = rc
            except Exception as e:
                sub_action["error"] = str(e)
            rep.actions_taken.append(sub_action)

    return rep


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_report(rep: RecoverReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Orchestrator Recovery Report")
    lines.append("=" * 80)

    # Procs
    lines.append("")
    lines.append(f"Processes ({len(rep.procs)} total):")
    if not rep.procs:
        lines.append("  (none — no orchestrator/harness backend processes running)")
    else:
        lines.append(f"  {'pid':<8s} {'kind':<22s} {'health':<10s} {'elapsed':<10s} {'op':<25s} notes")
        for p in rep.procs:
            elap_h = f"{p.elapsed_sec//3600:d}:{(p.elapsed_sec%3600)//60:02d}:{p.elapsed_sec%60:02d}"
            lines.append(
                f"  {p.pid:<8d} {p.kind.value:<22s} {p.health.value:<10s} "
                f"{elap_h:<10s} {(p.op or '-'):<25s} {'; '.join(p.notes)[:60]}"
            )

    # Markers
    lines.append("")
    lines.append(f"Agent-died markers ({len(rep.stale_markers)} total, "
                 f"{sum(1 for m in rep.stale_markers if m.is_stale)} stale):")
    for m in rep.stale_markers:
        flag = "STALE" if m.is_stale else "current"
        lines.append(f"  [{flag}] {m.op} (age {m.age_minutes:.0f}min) — {m.reason}")

    # Actions taken
    if rep.actions_taken:
        lines.append("")
        lines.append("Actions taken:")
        for a in rep.actions_taken:
            lines.append(f"  - {json.dumps(a)[:200]}")

    # Resumable
    if rep.resumable_ops:
        lines.append("")
        lines.append(f"Resumable ops ({len(rep.resumable_ops)}):")
        for op in rep.resumable_ops:
            lines.append(f"  - {op}  (run: python3 -m orchestrator --resume {op})")

    if rep.pending_user_decision:
        lines.append("")
        lines.append(f"Pending user_decision.md ({len(rep.pending_user_decision)}):")
        for op in rep.pending_user_decision:
            ws = _resolve_workspace(op)
            lines.append(f"  - {op} (write: {ws}/user_decision.md)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Diagnose + recover stuck orchestrator runs")
    ap.add_argument("--kill", action="store_true",
                    help="kill zombie harness backend processes (writes audit log)")
    ap.add_argument("--clean-stale-markers", action="store_true",
                    help="rotate stale .agent_died_at_* markers (writes audit log)")
    ap.add_argument("--resume", action="store_true",
                    help="after kill+clean, invoke resume.py for each resumable op")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be done without doing it")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    args = ap.parse_args()

    rep = recover(
        kill=args.kill,
        clean_stale=args.clean_stale_markers,
        run_resume=args.resume,
        dry_run=args.dry_run,
    )

    if args.json:
        # Convert dataclasses to dicts (Path → str, Enum → value)
        out = {
            "procs": [
                {"pid": p.pid, "kind": p.kind.value, "health": p.health.value,
                 "op": p.op, "elapsed_sec": p.elapsed_sec, "notes": p.notes}
                for p in rep.procs
            ],
            "stale_markers": [
                {"op": m.op, "marker": str(m.marker_path), "is_stale": m.is_stale,
                 "age_minutes": m.age_minutes, "reason": m.reason}
                for m in rep.stale_markers
            ],
            "actions_taken": rep.actions_taken,
            "resumable_ops": rep.resumable_ops,
            "pending_user_decision": rep.pending_user_decision,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print(format_report(rep))

    # Exit code: 0 if clean, 2 if zombies/stale found (advisory)
    has_zombies = any(p.health == ProcHealth.ZOMBIE for p in rep.procs)
    has_stale = any(m.is_stale for m in rep.stale_markers)
    return 2 if (has_zombies or has_stale) and not (args.kill or args.clean_stale_markers) else 0


if __name__ == "__main__":
    sys.exit(main())
