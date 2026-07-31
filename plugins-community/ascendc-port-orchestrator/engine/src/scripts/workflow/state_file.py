#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""state_file.py — durable per-op + batch state for /ascendc-op-gen and /aog-op-batch.

Solves the cross-session resume problem: a session crash, org-usage-limit kill, NPU
disconnect, or context-compaction event leaves the orchestrator without explicit
"where was I" memory. Without this file, the next session must re-derive state from
PROGRESS.md tails + state_transitions.jsonl + lane_state — fragile and error-prone.

This module is the single source of truth for "what has been done / what remains" at
two scopes:

  - Per-op:  workspace/<op>/.opgen_state.json    (created/updated by /ascendc-op-gen)
  - Batch:   workspace/.batch_state.json         (created/updated by /aog-op-batch)

The two are independent files; batch state references per-op state by relative path.

Schema (per-op, .opgen_state.json):
  {
    "op": "26_AvgPool3d",
    "skill_invocation": "ascendc-op-gen",
    "skill_args": "26_AvgPool3d",
    "mode": "default" | "optimize" | "research-first" | "precision-probe",
    "target": "a5" | "a3" | "a3-ds" | "a2",
    "soc_version": "Ascend950PR_9579",
    "det_policy": "required" | "best_effort" | "n/a",
    "session_id_first_seen": "20260503T044100Z",
    "session_id_last_touch": "20260503T053200Z",
    "phases_completed": ["O0","O1","O1.5","O2","O2.5","O3"],
    "current_phase": "O4",
    "current_state_machine_state": "await_optimizer",
    "last_handoff": {
      "from_agent": "topktopp-kw-2",
      "to_agent": "@aog-kernel-optimizer",
      "summary": "Pass A 16/16, Pass B 47/50, perf 0.397x",
      "ts": "20260503T053200Z"
    },
    "iter_history": [
      {"agent_slug": "topktopp-kw-1", "iter": 1, "verdict": "PARTIAL_PERF", "ts": "..."},
      {"agent_slug": "topktopp-ko-1", "iter": 1, "verdict": "PARTIAL_PERF_PLATEAU", "ts": "..."}
    ],
    "iter_cap_remaining": {"kw": 2, "ko": 4, "pp": 1, "ar": 1},
    "build_archive_dir": "/data/build_archive/9_topktopp",
    "kb_merge_required": true,
    "kb_merged_marker_seen": false,
    "blockers": [],
    "version": 1
  }

Schema (batch, .batch_state.json):
  {
    "batch_id": "fix_partial_perf_20260503T044100Z",
    "skill_invocation": "aog-op-batch",
    "skill_args": "9_TopKTopP 22_Nonzero 26_AvgPool3d",
    "session_id_first_seen": "20260503T044100Z",
    "session_id_last_touch": "20260503T053200Z",
    "ops_planned": ["9_TopKTopP", "22_Nonzero", "26_AvgPool3d"],
    "ops_status": {
      "9_TopKTopP":  {"per_op_state_file": "workspace/9_topktopp/"
                       ".opgen_state.json", "lane": 1, "status": "in_progress"},
      "22_Nonzero":  {"per_op_state_file": "workspace/22_Nonzero/.opgen_state.json", "lane": 0, "status": "completed"},
      "26_AvgPool3d":{"per_op_state_file": "workspace/26_avgpool3d/.opgen_state.json", "lane": 2, "status": "pending"}
    },
    "lanes_assignment": {"0":"22_Nonzero","1":"9_TopKTopP","2":"26_AvgPool3d"},
    "blockers": ["org_usage_limit"],
    "session_log": ["20260503T044100Z init", "20260503T053200Z lane 1 spawn died"],
    "version": 1
  }

CLI usage (for shell scripts and post_agent_return.py):
  state_file.py per-op init    --op <op> --workspace <dir> --skill ascendc-op-gen --args <args> --target <a5|a3|...>
  state_file.py per-op update  --op <op> --workspace <dir> --field <dotted.path> --value <json>
  state_file.py per-op resume  --op <op> --workspace <dir>            # prints "what's next"
  state_file.py batch init     --batch-id <id> --ops <list> --workspace-root <dir>
  state_file.py batch update   --batch-id <id> --workspace-root <dir> --field <p> --value <v>
  state_file.py batch resume   --workspace-root <dir>                  # prints batch summary

Python usage (for orchestrator main-context):
  from state_file import OpgenState, BatchState
  s = OpgenState.load_or_init(workspace_dir, op="26_AvgPool3d", target="a5", ...)
  if s.is_resumable():
      print(s.resume_summary())   # one-line: where to pick up
  s.complete_phase("O3")
  s.set_current_state("await_worker")
  s.record_handoff(from_agent="...", to="@aog-kernel-optimizer", summary="...")
  s.save()
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False))
    os.replace(tmp, path)


def _set_dotted(d: dict, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


@dataclass
class OpgenState:
    """Per-op state file for /ascendc-op-gen."""
    workspace_dir: Path
    op: str
    skill_invocation: str = "ascendc-op-gen"
    skill_args: str = ""
    mode: str = "default"
    target: str = ""
    soc_version: str = ""
    det_policy: str = "best_effort"
    session_id_first_seen: str = field(default_factory=_now_iso)
    session_id_last_touch: str = field(default_factory=_now_iso)
    phases_completed: list[str] = field(default_factory=list)
    current_phase: str = "O0"
    current_state_machine_state: str = "init"
    last_handoff: dict = field(default_factory=dict)
    iter_history: list[dict] = field(default_factory=list)
    iter_cap_remaining: dict = field(default_factory=lambda: {"kw": 3, "ko": 5, "pp": 1, "ar": 1, "fo": 1, "da": 1})
    build_archive_dir: str = ""
    kb_merge_required: bool = False
    kb_merged_marker_seen: bool = False
    blockers: list[str] = field(default_factory=list)
    version: int = SCHEMA_VERSION

    @property
    def state_file_path(self) -> Path:
        return self.workspace_dir / ".opgen_state.json"

    @classmethod
    def load_or_init(cls, workspace_dir: str | Path, op: str, **kwargs) -> "OpgenState":
        wd = Path(workspace_dir)
        wd.mkdir(parents=True, exist_ok=True)
        sf = wd / ".opgen_state.json"
        if sf.is_file():
            try:
                data = json.loads(sf.read_text())
                # touch last-seen
                data["session_id_last_touch"] = _now_iso()
                inst = cls(workspace_dir=wd, op=data.get("op", op))
                for k, v in data.items():
                    if hasattr(inst, k):
                        setattr(inst, k, v)
                inst.workspace_dir = wd
                return inst
            except (json.JSONDecodeError, KeyError):
                # corrupt → fall through to fresh init
                pass
        # fresh init
        inst = cls(workspace_dir=wd, op=op, **{k: v for k, v in kwargs.items() if hasattr(cls, k)})
        return inst

    def is_resumable(self) -> bool:
        """True if state file shows prior partial work (phases completed > O0)."""
        return len(self.phases_completed) > 1 or self.current_phase not in ("O0", "init")

    def resume_summary(self) -> str:
        """One-line summary of where to resume."""
        parts = [f"op={self.op}",
                 f"phases_done={','.join(self.phases_completed) or '(none)'}",
                 f"current_phase={self.current_phase}",
                 f"sm_state={self.current_state_machine_state}"]
        if self.last_handoff:
            parts.append(
                f"last_handoff={self.last_handoff.get('from_agent','?')}→{self.last_handoff.get('to_agent','?')}")
        if self.blockers:
            parts.append(f"blockers={','.join(self.blockers)}")
        return " | ".join(parts)

    def complete_phase(self, phase: str) -> None:
        if phase not in self.phases_completed:
            self.phases_completed.append(phase)

    def set_current_phase(self, phase: str) -> None:
        self.current_phase = phase

    def set_current_state(self, sm_state: str) -> None:
        self.current_state_machine_state = sm_state

    def record_handoff(self, from_agent: str, to_agent: str, summary: str) -> None:
        self.last_handoff = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "summary": summary,
            "ts": _now_iso(),
        }

    def record_iter(self, agent_slug: str, iter_num: int, verdict: str, summary: str = "") -> None:
        self.iter_history.append({
            "agent_slug": agent_slug,
            "iter": iter_num,
            "verdict": verdict,
            "summary": summary,
            "ts": _now_iso(),
        })
        # decrement iter cap by agent type
        # slug pattern: <slug>-<code>-<n> → code is kw/ko/pp/ar/fo/da
        for code in ("kw", "ko", "pp", "ar", "fo", "da"):
            if f"-{code}-" in agent_slug and code in self.iter_cap_remaining:
                self.iter_cap_remaining[code] = max(0, self.iter_cap_remaining[code] - 1)
                break

    def add_blocker(self, blocker: str) -> None:
        if blocker not in self.blockers:
            self.blockers.append(blocker)

    def clear_blocker(self, blocker: str) -> None:
        if blocker in self.blockers:
            self.blockers.remove(blocker)

    def save(self) -> None:
        self.session_id_last_touch = _now_iso()
        data = {
            "op": self.op,
            "skill_invocation": self.skill_invocation,
            "skill_args": self.skill_args,
            "mode": self.mode,
            "target": self.target,
            "soc_version": self.soc_version,
            "det_policy": self.det_policy,
            "session_id_first_seen": self.session_id_first_seen,
            "session_id_last_touch": self.session_id_last_touch,
            "phases_completed": self.phases_completed,
            "current_phase": self.current_phase,
            "current_state_machine_state": self.current_state_machine_state,
            "last_handoff": self.last_handoff,
            "iter_history": self.iter_history,
            "iter_cap_remaining": self.iter_cap_remaining,
            "build_archive_dir": self.build_archive_dir,
            "kb_merge_required": self.kb_merge_required,
            "kb_merged_marker_seen": self.kb_merged_marker_seen,
            "blockers": self.blockers,
            "version": self.version,
        }
        _atomic_write_json(self.state_file_path, data)


@dataclass
class BatchState:
    """Batch state file for /aog-op-batch (wraps multiple per-op states)."""
    workspace_root: Path
    batch_id: str
    skill_invocation: str = "aog-op-batch"
    skill_args: str = ""
    session_id_first_seen: str = field(default_factory=_now_iso)
    session_id_last_touch: str = field(default_factory=_now_iso)
    ops_planned: list[str] = field(default_factory=list)
    ops_status: dict = field(default_factory=dict)  # op_name → {per_op_state_file, lane, status}
    lanes_assignment: dict = field(default_factory=dict)  # lane_id_str → op_name
    blockers: list[str] = field(default_factory=list)
    session_log: list[str] = field(default_factory=list)
    version: int = SCHEMA_VERSION

    @property
    def state_file_path(self) -> Path:
        return self.workspace_root / ".batch_state.json"

    @classmethod
    def load_or_init(cls, workspace_root: str | Path, batch_id: str | None = None, **kwargs) -> "BatchState":
        wr = Path(workspace_root)
        sf = wr / ".batch_state.json"
        if sf.is_file():
            try:
                data = json.loads(sf.read_text())
                data["session_id_last_touch"] = _now_iso()
                inst = cls(workspace_root=wr, batch_id=data.get("batch_id", batch_id or "unknown"))
                for k, v in data.items():
                    if hasattr(inst, k):
                        setattr(inst, k, v)
                inst.workspace_root = wr
                return inst
            except (json.JSONDecodeError, KeyError):
                pass
        if not batch_id:
            batch_id = f"batch_{_now_iso()}"
        inst = cls(workspace_root=wr, batch_id=batch_id, **{k: v for k, v in kwargs.items() if hasattr(cls, k)})
        return inst

    def is_resumable(self) -> bool:
        return any(s.get("status") in ("in_progress", "pending") for s in self.ops_status.values())

    def resume_summary(self) -> str:
        done = sum(1 for s in self.ops_status.values() if s.get("status") == "completed")
        in_prog = sum(1 for s in self.ops_status.values() if s.get("status") == "in_progress")
        pending = sum(1 for s in self.ops_status.values() if s.get("status") == "pending")
        failed = sum(1 for s in self.ops_status.values() if s.get("status") == "failed")
        return (f"batch={self.batch_id} | done={done} in_progress={in_prog} pending={pending} "
                f"failed={failed} | blockers={','.join(self.blockers) or 'none'}")

    def assign_op(self, op: str, lane: int, per_op_state_file: str) -> None:
        self.ops_status[op] = {
            "per_op_state_file": per_op_state_file,
            "lane": lane,
            "status": "pending",
            "assigned_ts": _now_iso(),
        }
        self.lanes_assignment[str(lane)] = op

    def update_op_status(self, op: str, status: str, note: str = "") -> None:
        if op not in self.ops_status:
            self.ops_status[op] = {}
        self.ops_status[op]["status"] = status
        self.ops_status[op]["last_update_ts"] = _now_iso()
        if note:
            self.ops_status[op]["note"] = note
        self.session_log.append(f"{_now_iso()} op={op} status={status} {note}".strip())

    def add_blocker(self, blocker: str) -> None:
        if blocker not in self.blockers:
            self.blockers.append(blocker)
            self.session_log.append(f"{_now_iso()} BLOCKER {blocker}")

    def clear_blocker(self, blocker: str) -> None:
        if blocker in self.blockers:
            self.blockers.remove(blocker)
            self.session_log.append(f"{_now_iso()} BLOCKER_CLEARED {blocker}")

    def save(self) -> None:
        self.session_id_last_touch = _now_iso()
        data = {
            "batch_id": self.batch_id,
            "skill_invocation": self.skill_invocation,
            "skill_args": self.skill_args,
            "session_id_first_seen": self.session_id_first_seen,
            "session_id_last_touch": self.session_id_last_touch,
            "ops_planned": self.ops_planned,
            "ops_status": self.ops_status,
            "lanes_assignment": self.lanes_assignment,
            "blockers": self.blockers,
            "session_log": self.session_log[-200:],  # keep last 200 lines
            "version": self.version,
        }
        _atomic_write_json(self.state_file_path, data)


def _cli():
    ap = argparse.ArgumentParser(prog="state_file.py", description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="scope", required=True)

    # per-op subcommands
    op_parser = sp.add_parser("per-op")
    op_sub = op_parser.add_subparsers(dest="action", required=True)
    for action in ("init", "update", "resume", "show"):
        sub = op_sub.add_parser(action)
        sub.add_argument("--op", required=True)
        sub.add_argument("--workspace", required=True)
        if action == "init":
            sub.add_argument("--skill", default="ascendc-op-gen")
            sub.add_argument("--args", default="")
            sub.add_argument("--mode", default="default")
            sub.add_argument("--target", default="")
            sub.add_argument("--soc-version", default="")
            sub.add_argument("--det-policy", default="best_effort")
        if action == "update":
            sub.add_argument("--field", required=True, help="dotted path, e.g. last_handoff.summary")
            sub.add_argument("--value", required=True, help="JSON-parseable value")

    # batch subcommands
    b_parser = sp.add_parser("batch")
    b_sub = b_parser.add_subparsers(dest="action", required=True)
    for action in ("init", "update", "resume", "show"):
        sub = b_sub.add_parser(action)
        sub.add_argument("--workspace-root", required=True)
        if action == "init":
            sub.add_argument("--batch-id", required=True)
            sub.add_argument("--ops", required=True, help="comma-separated op names")
            sub.add_argument("--skill", default="aog-op-batch")
            sub.add_argument("--args", default="")
        if action == "update":
            sub.add_argument("--field", required=True)
            sub.add_argument("--value", required=True)

    args = ap.parse_args()

    if args.scope == "per-op":
        if args.action == "init":
            s = OpgenState.load_or_init(args.workspace, op=args.op,
                                        skill_invocation=args.skill, skill_args=args.args,
                                        mode=args.mode, target=args.target,
                                        soc_version=args.soc_version, det_policy=args.det_policy)
            s.save()
            print(json.dumps({"ok": True, "path": str(s.state_file_path), "resumable": s.is_resumable(),
                              "resume_summary": s.resume_summary()}))
        elif args.action == "update":
            s = OpgenState.load_or_init(args.workspace, op=args.op)
            try:
                v = json.loads(args.value)
            except json.JSONDecodeError:
                v = args.value
            data = {**{k: getattr(s, k) for k in s.__dataclass_fields__ if k != "workspace_dir"}}
            _set_dotted(data, args.field, v)
            for k, val in data.items():
                if hasattr(s, k):
                    setattr(s, k, val)
            s.save()
            print(json.dumps({"ok": True, "field": args.field, "value": v}))
        elif args.action in ("resume", "show"):
            s = OpgenState.load_or_init(args.workspace, op=args.op)
            print(s.resume_summary())

    elif args.scope == "batch":
        if args.action == "init":
            ops = [o.strip() for o in args.ops.split(",") if o.strip()]
            b = BatchState.load_or_init(args.workspace_root, batch_id=args.batch_id,
                                        skill_invocation=args.skill, skill_args=args.args,
                                        ops_planned=ops)
            b.save()
            print(json.dumps({"ok": True, "path": str(b.state_file_path), "resumable": b.is_resumable(),
                              "resume_summary": b.resume_summary()}))
        elif args.action == "update":
            b = BatchState.load_or_init(args.workspace_root)
            try:
                v = json.loads(args.value)
            except json.JSONDecodeError:
                v = args.value
            data = {**{k: getattr(b, k) for k in b.__dataclass_fields__ if k != "workspace_root"}}
            _set_dotted(data, args.field, v)
            for k, val in data.items():
                if hasattr(b, k):
                    setattr(b, k, val)
            b.save()
            print(json.dumps({"ok": True, "field": args.field, "value": v}))
        elif args.action in ("resume", "show"):
            b = BatchState.load_or_init(args.workspace_root)
            print(b.resume_summary())


if __name__ == "__main__":
    _cli()
