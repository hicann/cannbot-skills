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
"""orchestrator_sweep.py — post-handoff multi-op state-machine sweep.

When any background Agent returns in /aog-op-batch parallel mode, this script
inventories all ops in the batch and emits the actionable next-step list for
each. Orchestrator calls this on every task-notification + uses the output
to dispatch agents / KB merges / archives.

Goal: take the manual "remember to advance state machine after each Agent
return" out of the orchestrator's head and into a declarative tool.

Usage:
    python3 src/scripts/workflow/orchestrator_sweep.py \\
        --batch-id 20260501T180400Z \\
        [--ops 2_SwiGLU,5_Cumsum] \\
        [--json]

If --batch-id is given, reads workspace/.batch_runs/<id>/plan.json for op list.
If --ops is given, sweeps those explicitly. At least one required.

Output (JSON if --json, else human-readable):
    {
      "batch_id": "...",
      "ops": [
        {
          "op": "5_Cumsum",
          "current_state": "await_probe" | "await_optimizer" | "finalize" | "abort" | ...,
          "lane": <int|null>,         # currently allocated lane, or null if released
          "next_action": "spawn_probe" | "spawn_optimizer" | "kb_merge" | "phase_o5" |
                         "archive" | "wait" | "abort" | "stuck",
          "next_action_args": {...},  # context-specific (e.g. agent type, brief hints)
          "knowledge_update_pending": <bool>,  # workspace has unmerged knowledge_update.md
          "handoff_signal": "<last handoff line from PROGRESS.md>"
        },
        ...
      ],
      "free_lanes": [0, 2, ...],
      "queued_ops": [...]   # ops not yet started (from plan.json minus active)
    }

The orchestrator reads next_action and:
- spawn_probe / spawn_optimizer / spawn_worker → run Agent(subagent_type=...)
- kb_merge → Skill(name="aog-knowledge-maintain", args="knowledge_update_path=...")
- phase_o5 → run delegation/anti-hack/anti-overfitting scans + self-critic + perf
- archive → cp workspace/{op}/* to output/{project}/src/kernels/{op}/, drop .kb_merged
- wait → no actionable state change yet (op's agent still running on its lane)
- abort → state machine reached terminal abort; surface to user
- stuck → contradictory or undefined state; needs investigation

This script does NOT spawn agents itself. It's an inventory tool. Orchestrator
acts on its output.
"""
from __future__ import annotations
import logging
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO / "workspace"
LANE_STATUS = REPO / "src" / "skills" / "aog-parallel-lane-orchestrator" / "scripts" / "lane_status.py"
STATE_MACHINE = REPO / "src" / "scripts" / "workflow" / "state_machine.py"


def _read_progress_handoff(op: str) -> str:
    """Read last handoff line from workspace/{op}/PROGRESS.md (or '')."""
    p = WORKSPACE / op / "PROGRESS.md"
    if not p.exists():
        return ""
    text = p.read_text(errors="replace")
    handoff_pattern = re.compile(r"^(→ orchestrator:|@aog-\S+:|@orchestrator:).*$", re.M)
    matches = handoff_pattern.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()


def _read_state_log_tail(op: str) -> tuple[str, dict]:
    """Read state_transitions.jsonl tail (to_state, full record). Returns ('', {}) if missing."""
    p = WORKSPACE / op / "state_transitions.jsonl"
    if not p.exists():
        return "", {}
    try:
        lines = [ln for ln in p.read_text().split("\n") if ln.strip()]
        if not lines:
            return "", {}
        last = json.loads(lines[-1])
        return last.get("to_state", ""), last
    except Exception:
        return "", {}


def _has_knowledge_update_unmerged(op: str) -> bool:
    """True if workspace/{op}/knowledge_update.md exists AND archive lacks .kb_merged for it."""
    src = WORKSPACE / op / "knowledge_update.md"
    if not src.exists() or src.stat().st_size < 50:
        return False
    # Find archive path candidates
    for project in ("a3_to_a5_port", "backward_ops"):
        archive = REPO / "output" / project / "src" / "kernels" / op
        if archive.exists():
            marker = archive / ".kb_merged"
            if marker.exists():
                try:
                    from kb_marker_verifier import verify_marker
                    from kb_tiering.adapters.cannbot_c import resolve_c_root
                except ImportError:
                    scripts_root = Path(__file__).resolve().parent.parent
                    sys.path.insert(0, str(scripts_root))
                    try:
                        from kb_marker_verifier import verify_marker
                        from kb_tiering.adapters.cannbot_c import resolve_c_root
                    except ImportError:
                        return True
                report = verify_marker(
                    archive,
                    project_root=REPO,
                    expected_c_root=resolve_c_root().resolve(),
                )
                if report.verdict == "OK":
                    return False  # already merged into durable provider state
            return True
    return True  # archive doesn't exist yet → merge pending


def _classify_next_action(op: str) -> dict[str, Any]:
    """Inventory one op and decide next action."""
    handoff = _read_progress_handoff(op)
    log_state, log_tail = _read_state_log_tail(op)

    # Knowledge update inventory
    kb_pending = _has_knowledge_update_unmerged(op)

    # Decide next action based on log_state
    out = {
        "op": op,
        "current_state": log_state or "(no log — fresh)",
        "handoff_signal": handoff,
        "knowledge_update_pending": kb_pending,
        "lane": None,
        "next_action": "wait",
        "next_action_args": {},
    }

    # Match log_state to action
    if log_state == "await_worker":
        # If handoff says worker just exited, dispatch next based on handoff
        if handoff:
            if handoff.startswith("→ orchestrator: done"):
                out["next_action"] = "phase_o5"
                out["next_action_args"] = {"reason": "worker exited done"}
            elif handoff.startswith("@aog-precision-probe"):
                out["next_action"] = "spawn_probe"
                out["next_action_args"] = {"agent_type": "aog-precision-probe"}
            elif handoff.startswith("@aog-kernel-optimizer"):
                out["next_action"] = "spawn_optimizer"
                out["next_action_args"] = {"agent_type": "aog-kernel-optimizer"}
            elif handoff.startswith("@aog-determinism-analyzer"):
                out["next_action"] = "spawn_det_analyzer"
                out["next_action_args"] = {"agent_type": "aog-determinism-analyzer"}
            elif handoff.startswith("@orchestrator: build stuck"):
                out["next_action"] = "stuck"
                out["next_action_args"] = {"reason": "build stuck"}
            else:
                # New op no worker run yet
                out["next_action"] = "spawn_worker"
                out["next_action_args"] = {"agent_type": "aog-kernel-worker"}
        else:
            out["next_action"] = "spawn_worker"
            out["next_action_args"] = {"agent_type": "aog-kernel-worker"}
    elif log_state == "await_probe":
        if handoff and ("probe done" in handoff.lower() or "applied" in handoff.lower()):
            # Probe done — needs perf measurement + finalize
            out["next_action"] = "phase_o5"
            out["next_action_args"] = {"reason": "probe applied fix in-place"}
        else:
            out["next_action"] = "spawn_probe"
            out["next_action_args"] = {"agent_type": "aog-precision-probe"}
    elif log_state == "await_optimizer":
        if handoff and "final perf" in handoff.lower():
            # Optimizer done
            out["next_action"] = "phase_o5"
            out["next_action_args"] = {"reason": "optimizer exited"}
        elif handoff and "requires rewrite" in handoff.lower():
            # Optimizer wrote V3 directive → respawn worker Kind 2
            out["next_action"] = "spawn_worker"
            out["next_action_args"] = {"agent_type": "aog-kernel-worker", "kind": 2}
        else:
            out["next_action"] = "spawn_optimizer"
            out["next_action_args"] = {"agent_type": "aog-kernel-optimizer"}
    elif log_state == "await_researcher":
        out["next_action"] = "spawn_researcher"
        out["next_action_args"] = {"agent_type": "aog-researcher"}
    elif log_state == "await_det_analyzer":
        out["next_action"] = "spawn_det_analyzer"
        out["next_action_args"] = {"agent_type": "aog-determinism-analyzer"}
    elif log_state == "await_fused_optimizer":
        out["next_action"] = "spawn_fused_optimizer"
        out["next_action_args"] = {"agent_type": "aog-fused-optimizer"}
    elif log_state == "finalize":
        # Determine finalize sub-step
        archive_exists = any(
            (REPO / "output" / proj / "src" / "kernels" / op).exists()
            for proj in ("a3_to_a5_port", "backward_ops")
        )
        verif = WORKSPACE / op / "verification.json"
        self_critic = WORKSPACE / op / "self_critic_report.md"

        if not self_critic.exists():
            out["next_action"] = "phase_o5"
            out["next_action_args"] = {"reason": "self_critic_report.md missing"}
        elif kb_pending:
            out["next_action"] = "kb_merge"
            out["next_action_args"] = {"path": str(WORKSPACE / op / "knowledge_update.md")}
        else:
            out["next_action"] = "archive"
            out["next_action_args"] = {"reason": "ready to archive"}
    elif log_state == "abort":
        out["next_action"] = "abort"
        out["next_action_args"] = {"reason": "state machine reached terminal abort"}
    elif not log_state:
        # Fresh workspace
        if (WORKSPACE / op / "PROGRESS.md").exists():
            # Pre-Phase-O3 setup done; ready to spawn first worker
            out["next_action"] = "spawn_worker"
            out["next_action_args"] = {"agent_type": "aog-kernel-worker", "is_first_spawn": True}
        else:
            out["next_action"] = "phase_o0_o3_setup"
            out["next_action_args"] = {"reason": "workspace not yet initialized"}

    return out


def _list_lanes() -> tuple[list[int], list[tuple[int, str]]]:
    """Return (free_lanes, busy_lane_ops)."""
    if not LANE_STATUS.exists():
        return [], []
    try:
        result = subprocess.run(
            ["python3", str(LANE_STATUS), "--json"],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        free = [l["lane"] for l in data["lanes"] if l["state"] == "FREE" and l.get("npu_free")]
        busy = [(l["lane"], l.get("op", "?")) for l in data["lanes"] if l["state"] == "BUSY"]
        return free, busy
    except Exception:
        return [], []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-id", help="batch ID; reads workspace/.batch_runs/<id>/plan.json")
    p.add_argument("--ops", help="comma-separated op list; alternative to --batch-id")
    p.add_argument("--json", action="store_true", help="output JSON")
    args = p.parse_args(argv)

    ops: list[str] = []
    batch_id = args.batch_id or ""

    if args.batch_id:
        plan = WORKSPACE / ".batch_runs" / args.batch_id / "plan.json"
        if plan.exists():
            try:
                data = json.loads(plan.read_text())
                ops = data.get("ops", [])
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
    if args.ops:
        ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not ops:
        print("ERROR: no ops to sweep (provide --batch-id or --ops)", file=sys.stderr)
        return 2

    free_lanes, busy = _list_lanes()
    busy_ops = {op: lane for lane, op in busy}

    op_inventory = []
    for op in ops:
        rec = _classify_next_action(op)
        rec["lane"] = busy_ops.get(op)
        # If op is currently running on a lane, downgrade actionable spawn → wait
        # (the agent is already running; respawning would race / duplicate work)
        if rec["lane"] is not None and rec["next_action"].startswith("spawn_"):
            rec["next_action_was_redirected_from"] = rec["next_action"]
            rec["next_action"] = "wait"
            rec["next_action_args"] = {"reason": f"agent already running on lane {rec['lane']}"}
        op_inventory.append(rec)

    out = {
        "batch_id": batch_id,
        "ops": op_inventory,
        "free_lanes": free_lanes,
        "busy_lanes": [(lane, op) for lane, op in busy],
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"=== Orchestrator Sweep — batch {batch_id or '(adhoc)'} ===")
        print(f"Free lanes: {free_lanes}")
        print(f"Busy: {busy}")
        print()
        for rec in op_inventory:
            kb_tag = " [kb_pending]" if rec["knowledge_update_pending"] else ""
            print(f"  {rec['op']:35s} state={rec['current_state']:20s} lane={rec['lane']} "
                  f"→ {rec['next_action']:20s}{kb_tag}")
            if rec["handoff_signal"]:
                print(f"    handoff: {rec['handoff_signal'][:120]}")
        print()
        spawnable = []
        for record in op_inventory:
            next_action = record["next_action"]
            if (
                next_action.startswith("spawn_")
                or next_action in ("kb_merge", "phase_o5", "archive")
            ):
                spawnable.append(record)
        if spawnable:
            print("ACTIONABLE next steps (orchestrator should dispatch immediately):")
            for r in spawnable:
                print(f"  - {r['op']}: {r['next_action']} ({r['next_action_args']})")
        else:
            print("(no actionable next steps — wait for in-flight agents)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
