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
"""batch_dispatcher.py — V3.8 Multi-op concurrent state-machine dispatcher.

Solves DEBT-067 (orchestrator-as-script) for the multi-concurrent-op case:
multiple ops can be at different state-machine progress simultaneously, each
following its own pipeline (kw → ko → fo → researcher → kw → ...) without
manual orchestrator routing decisions in main context.

Architecture:

  ┌─────────────────────────────────┐
  │  main context (LLM orchestrator)│
  │  ┌───────────────────────────┐  │
  │  │ /aog-op-batch SKILL loop  │  │
  │  │   while not all-terminal: │  │
  │  │     plan = dispatcher.    │  │
  │  │       compute_next_actions│  │
  │  │     for each action:      │  │
  │  │       Agent(...)          │  │ ← Agent tool only callable here
  │  │     wait_completions()    │  │
  │  │     update_state(returns) │  │
  │  └───────────────────────────┘  │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  batch_dispatcher.py (this file)│
  │  ┌───────────────────────────┐  │
  │  │ compute_next_actions(ws_  │  │
  │  │   root) → list[NextAction]│  │
  │  │ (read-only: scans .opgen_ │  │
  │  │  state.json + runs state_ │  │
  │  │  machine.next per op)     │  │
  │  │                           │  │
  │  │ allocate_lanes(actions,   │  │
  │  │   lanes) → assignments    │  │
  │  │                           │  │
  │  │ build_brief(op, action) → │  │
  │  │   prompt_text             │  │
  │  │                           │  │
  │  │ record_spawn(op,lane,agent│  │
  │  │ record_completion(op,res) │  │
  │  └───────────────────────────┘  │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  state_machine.py (existing)    │
  │  state_file.py (existing V3.7.9)│
  │  workflow_critic.py (existing)  │
  └─────────────────────────────────┘

Workflow per dispatcher invocation:

  $ python3 src/scripts/workflow/batch_dispatcher.py plan --workspace-root workspace
  → outputs JSON list of NextAction objects (one per non-terminal op)
  → main context reads JSON, spawns each via Agent

  Per-op decision tree (mechanical, no LLM judgment):
    1. Read workspace/<op>/.opgen_state.json (state_file.py)
    2. If terminal (finalize/abort): skip
    3. Otherwise: state_machine.py next --workspace workspace/<op> [--dry-run]
       returns next state + agent type
    4. If next state requires user/researcher input not yet present: skip (wait)
    5. Otherwise: emit NextAction(op, next_state, agent_type, brief_template)

Lane allocation (if batched-spawn mode):

  $ python3 src/scripts/workflow/batch_dispatcher.py allocate \
      --actions '[<json>]' \
      --max-lanes 4
  → outputs JSON dict mapping op_name → lane_id

Spawn record (called by main context after spawning):

  $ python3 src/scripts/workflow/batch_dispatcher.py record-spawn \
      --op <op> --lane <N> --agent <agent_type> --slug <slug>

Completion record (called by main context after agent returns):

  $ python3 src/scripts/workflow/batch_dispatcher.py record-completion \
      --op <op> --handoff '<handoff line>'
  → atomically updates .opgen_state.json + state_transitions.jsonl
  → returns next_state for the op (so main context can decide if more spawns)

This script is intentionally read-mostly. It does NOT spawn agents (that's
LLM-only via Agent tool). It does NOT make LLM-class decisions. All routing
is YAML-driven (state_machine.py) + workspace-state-driven (.opgen_state.json).

Created 2026-05-03 per user directive "立刻修复 debt 067，我们后续会大量使用 Batch".
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Project layout
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
STATE_MACHINE_PY = SCRIPT_DIR / "state_machine.py"
STATE_FILE_PY = SCRIPT_DIR / "state_file.py"
LANES_STATE_DIR = PROJECT_ROOT / "src" / "workspace" / ".lanes"

TERMINAL_STATES = {"finalize", "abort"}

# Map state-machine "agent_to_state" inverse — given a state, which agent runs there
STATE_TO_AGENT = {
    "await_worker": "aog-kernel-worker",
    "await_optimizer": "aog-kernel-optimizer",
    "await_probe": "aog-precision-probe",
    "await_fused_optimizer": "aog-fused-optimizer",
    "await_researcher": "aog-researcher",
    "await_det_analyzer": "aog-determinism-analyzer",
}

# Map agent → 2-letter slug code (per V3.3.1 G7 naming convention)
AGENT_TO_SLUG_CODE = {
    "aog-kernel-worker": "kw",
    "aog-kernel-optimizer": "ko",
    "aog-precision-probe": "pp",
    "aog-fused-optimizer": "fo",
    "aog-researcher": "ar",
    "aog-determinism-analyzer": "da",
}


@dataclass
class NextAction:
    """One agent-spawn decision for a single op, computed by the dispatcher."""
    op: str                          # op name (workspace dir basename)
    workspace: str                   # workspace path (relative to project root)
    current_state: str               # what state the op is in NOW
    next_state: str                  # what state the dispatcher recommends spawning
    agent_type: str                  # which agent type (e.g. "aog-kernel-worker")
    suggested_slug: str              # suggested per-spawn slug (per G7)
    skip: bool                       # True if this op has nothing to do (terminal/in-flight)
    skip_reason: str                 # human-readable
    last_handoff: str                # last handoff text from prior agent

    def asdict(self) -> dict:
        return asdict(self)


def _list_op_workspaces(ws_root: Path) -> list[Path]:
    """Discover op workspaces — directories with .opgen_state.json OR PROGRESS.md."""
    if not ws_root.exists():
        return []
    out = []
    for entry in sorted(ws_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if (entry / ".opgen_state.json").exists() or (entry / "PROGRESS.md").exists():
            out.append(entry)
    return out


def _read_opgen_state(ws: Path) -> dict | None:
    """Read .opgen_state.json (V3.7.9 state file). Returns None if absent/corrupt."""
    sf = ws / ".opgen_state.json"
    if not sf.is_file():
        return None
    try:
        return json.loads(sf.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_last_transition(ws: Path) -> dict | None:
    """Read the last entry from state_transitions.jsonl. Returns None if log missing/empty."""
    log = ws / "state_transitions.jsonl"
    if not log.is_file():
        return None
    try:
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except (json.JSONDecodeError, OSError):
        return None


def _detect_current_state(ws: Path) -> tuple[str, str]:
    """Determine op's current state. Returns (state, last_handoff_or_init_marker)."""
    last = _read_last_transition(ws)
    if last is None:
        # No log: starting fresh. Default to await_worker (initial state for benchmark/opgen modes)
        # but check PROGRESS.md for explicit Mode declaration first.
        progress = ws / "PROGRESS.md"
        if progress.exists():
            text = progress.read_text(errors="replace")
            for line in text.split("\n")[:15]:
                s = line.strip()
                if s.lower().startswith("mode:"):
                    mode = s.split(":", 1)[1].strip().split()[0].lower()
                    if mode == "optimize":
                        return ("await_optimizer", "init (optimize mode)")
                    break
        return ("await_worker", "init (default)")
    state = last.get("to_state", "")
    handoff = last.get("handoff", "")
    return (state, handoff)


def _next_spawn_index(ws: Path, agent_type: str) -> int:
    """Count prior spawns of this agent type for the op. Used for slug naming."""
    log = ws / "state_transitions.jsonl"
    if not log.is_file():
        return 1
    target_state = next((s for s, a in STATE_TO_AGENT.items() if a == agent_type), None)
    if target_state is None:
        return 1
    n = 0
    try:
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("to_state") == target_state:
                n += 1
    except OSError:
        return 1
    return n + 1  # next spawn is N+1


def _build_op_slug(op_name: str) -> str:
    """Convert op workspace dir name to a clean slug for description prefix."""
    s = op_name.lower()
    # Strip op-number prefix (e.g. "9_topktopp" → "topktopp", "12_kvrms" → "kvrms")
    if s and s[0].isdigit() and "_" in s:
        s = s.split("_", 1)[1]
    return s


def _is_in_flight(op_name: str) -> bool:
    """Heuristic: is an agent currently running for this op? Check lane state files."""
    if not LANES_STATE_DIR.exists():
        return False
    for lane_dir in LANES_STATE_DIR.iterdir():
        if not lane_dir.is_dir():
            continue
        sf = lane_dir / "state"
        if not sf.is_file():
            continue
        try:
            content = sf.read_text()
            if "state=BUSY" in content:
                # Match op name (handles op_<slug>_<agent><N> patterns)
                for line in content.split("\n"):
                    if line.startswith("op="):
                        busy_op = line.split("=", 1)[1].strip()
                        # heuristic: op name appears as substring
                        if op_name.replace("_", "").lower() in busy_op.replace("_", "").lower():
                            return True
                        if any(p in busy_op for p in [op_name, _build_op_slug(op_name)]):
                            return True
        except OSError:
            continue
    return False


def compute_next_actions(ws_root: Path = WORKSPACE_ROOT) -> list[NextAction]:
    """For each op workspace, determine what (if anything) should spawn next.

    Returns a list of NextAction objects, one per discovered op. Ops that are
    terminal (finalize/abort) or in-flight (lane state BUSY for them) are
    returned with skip=True.
    """
    actions: list[NextAction] = []
    workspaces = _list_op_workspaces(ws_root)

    for ws in workspaces:
        op = ws.name
        ws_rel = str(ws.relative_to(PROJECT_ROOT))
        current_state, last_handoff = _detect_current_state(ws)

        # Skip terminal
        if current_state in TERMINAL_STATES:
            actions.append(NextAction(
                op=op, workspace=ws_rel,
                current_state=current_state, next_state=current_state,
                agent_type="", suggested_slug="",
                skip=True, skip_reason=f"terminal state ({current_state})",
                last_handoff=last_handoff,
            ))
            continue

        # Skip in-flight
        if _is_in_flight(op):
            actions.append(NextAction(
                op=op, workspace=ws_rel,
                current_state=current_state, next_state=current_state,
                agent_type=STATE_TO_AGENT.get(current_state, ""),
                suggested_slug="",
                skip=True, skip_reason="agent already in-flight (lane BUSY)",
                last_handoff=last_handoff,
            ))
            continue

        # Compute next state via state_machine.py next --dry-run
        try:
            result = subprocess.run(
                ["python3", str(STATE_MACHINE_PY), "next",
                 "--workspace", str(ws),
                 "--current-state", current_state,
                 "--handoff", last_handoff,
                 "--dry-run"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                actions.append(NextAction(
                    op=op, workspace=ws_rel,
                    current_state=current_state, next_state="error",
                    agent_type="", suggested_slug="",
                    skip=True,
                    skip_reason=f"state_machine.py next failed: {result.stderr.strip()[:200]}",
                    last_handoff=last_handoff,
                ))
                continue
            sm_out = json.loads(result.stdout)
            next_state = sm_out.get("to_state") or sm_out.get("next_state", "")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
            actions.append(NextAction(
                op=op, workspace=ws_rel,
                current_state=current_state, next_state="error",
                agent_type="", suggested_slug="",
                skip=True, skip_reason=f"state_machine invocation error: {e}",
                last_handoff=last_handoff,
            ))
            continue

        # next_state is terminal: nothing to spawn
        if next_state in TERMINAL_STATES:
            actions.append(NextAction(
                op=op, workspace=ws_rel,
                current_state=current_state, next_state=next_state,
                agent_type="", suggested_slug="",
                skip=True, skip_reason=f"next_state is terminal ({next_state})",
                last_handoff=last_handoff,
            ))
            continue

        agent_type = STATE_TO_AGENT.get(next_state, "")
        if not agent_type:
            actions.append(NextAction(
                op=op, workspace=ws_rel,
                current_state=current_state, next_state=next_state,
                agent_type="", suggested_slug="",
                skip=True, skip_reason=f"no agent mapping for state {next_state}",
                last_handoff=last_handoff,
            ))
            continue

        # Build suggested slug per G7
        op_slug = _build_op_slug(op)
        code = AGENT_TO_SLUG_CODE.get(agent_type, "??")
        spawn_index = _next_spawn_index(ws, agent_type)
        suggested_slug = f"{op_slug}-{code}-{spawn_index}"

        actions.append(NextAction(
            op=op, workspace=ws_rel,
            current_state=current_state, next_state=next_state,
            agent_type=agent_type, suggested_slug=suggested_slug,
            skip=False, skip_reason="",
            last_handoff=last_handoff,
        ))

    return actions


def allocate_lanes(actions: list[NextAction], max_lanes: int = 4) -> dict[str, int]:
    """Assign free lanes to non-skip actions. Returns op_name → lane_id."""
    # Read current lane state
    busy_lanes = set()
    if LANES_STATE_DIR.exists():
        for lane_dir in LANES_STATE_DIR.iterdir():
            if not lane_dir.is_dir():
                continue
            try:
                lid = int(lane_dir.name.replace("lane_", ""))
            except ValueError:
                continue
            if lid >= max_lanes:
                continue
            sf = lane_dir / "state"
            if sf.is_file() and "state=BUSY" in sf.read_text():
                busy_lanes.add(lid)

    free_lanes = [l for l in range(max_lanes) if l not in busy_lanes]
    actionable = [a for a in actions if not a.skip]
    assignments = {}
    for action, lane in zip(actionable, free_lanes):
        assignments[action.op] = lane
    return assignments


def cmd_plan(args) -> int:
    actions = compute_next_actions(Path(args.workspace_root or WORKSPACE_ROOT))
    output = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_workspaces": len(actions),
        "n_actionable": sum(1 for a in actions if not a.skip),
        "n_terminal": sum(1 for a in actions if a.skip and "terminal" in a.skip_reason),
        "n_in_flight": sum(1 for a in actions if a.skip and "in-flight" in a.skip_reason),
        "actions": [a.asdict() for a in actions],
    }
    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0


def cmd_allocate(args) -> int:
    src = args.actions
    if src == "-":
        raw = sys.stdin.read()
    elif src.startswith("@"):
        raw = Path(src[1:]).read_text()
    elif src.startswith("{") or src.startswith("["):
        raw = src
    else:
        # treat as file path
        raw = Path(src).read_text()
    parsed = json.loads(raw)
    actions_data = parsed.get("actions", parsed) if isinstance(parsed, dict) else parsed
    actions = [NextAction(**a) for a in actions_data]
    allocations = allocate_lanes(actions, max_lanes=args.max_lanes)
    output = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_lanes": args.max_lanes,
        "n_allocated": len(allocations),
        "allocations": [
            {"op": op, "lane": lane,
             "agent_type": next((a.agent_type for a in actions if a.op == op), ""),
             "suggested_slug": next((a.suggested_slug for a in actions if a.op == op), ""),
             "workspace": next((a.workspace for a in actions if a.op == op), "")}
            for op, lane in allocations.items()
        ],
    }
    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0


def cmd_summary(args) -> int:
    """Human-readable summary table of all op workspaces' state."""
    actions = compute_next_actions(Path(args.workspace_root or WORKSPACE_ROOT))
    print(f"{'op':<32} {'state':<28} {'next-action':<28} {'note':<40}")
    print("-" * 132)
    for a in actions:
        if a.skip:
            note = f"SKIP: {a.skip_reason[:38]}"
            next_action = "—"
        else:
            note = f"spawn {a.suggested_slug}"
            next_action = f"→ {a.agent_type}"
        print(f"{a.op:<32} {a.current_state:<28} {next_action:<28} {note:<40}")
    print("-" * 132)
    n_actionable = sum(1 for a in actions if not a.skip)
    n_terminal = sum(1 for a in actions if a.skip and "terminal" in a.skip_reason)
    n_in_flight = sum(1 for a in actions if a.skip and "in-flight" in a.skip_reason)
    n_other_skip = sum(1 for a in actions if a.skip) - n_terminal - n_in_flight
    print(
        f"Total: {len(actions)} ops  |  actionable: {n_actionable}  |  "
        f"in-flight: {n_in_flight}  |  terminal: {n_terminal}  |  "
        f"other-skip: {n_other_skip}"
    )
    return 0


def main():
    ap = argparse.ArgumentParser(prog="batch_dispatcher", description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sp.add_parser("plan", help="Compute next-actions JSON for all op workspaces")
    p_plan.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))
    p_plan.add_argument("--pretty", action="store_true")

    p_alloc = sp.add_parser("allocate", help="Assign lanes to actionable ops")
    p_alloc.add_argument("--actions", required=True,
                         help="JSON array of NextAction objects (typically from `plan` output)")
    p_alloc.add_argument("--max-lanes", type=int, default=4)
    p_alloc.add_argument("--pretty", action="store_true")

    p_sum = sp.add_parser("summary", help="Human-readable summary of all op workspaces")
    p_sum.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))

    args = ap.parse_args()
    cmd_handlers = {
        "plan": cmd_plan,
        "allocate": cmd_allocate,
        "summary": cmd_summary,
    }
    sys.exit(cmd_handlers[args.cmd](args))


if __name__ == "__main__":
    main()
