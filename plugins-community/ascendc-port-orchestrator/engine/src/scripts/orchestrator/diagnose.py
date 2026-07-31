# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Diagnose CLI — fleet status + per-op forensic reports (V2 #1).

Aggregates the structured diagnostic files written by orchestrator
modules into one place:

  workspace/<op>/
    state_transitions.jsonl       (state_executor)
    .cc_envelope_log.jsonl        (agent_dispatch P0b)
    .schema_normalizations.log    (schema_norm)
    .critic_invoke_log.jsonl      (critic_invoke)
    .kb_merge_log.jsonl           (kb_invoke)
    orchestrator_events.jsonl     (events.py)
    verification.json             (worker)
    probe_result.json             (probe)

CLI:
    python3 diagnose.py                  # fleet table
    python3 diagnose.py --op 22_Nonzero  # detailed per-op report
    python3 diagnose.py --paths          # path-coverage matrix
    python3 diagnose.py --cost           # cost rollup by agent type
    python3 diagnose.py --denials        # any permission_denials in any op?

Foundation: pure file-system reads, no LLM. Adds value by aggregating across
ops and giving canonical formatting; doesn't replace `jq` for ad-hoc.
"""
from __future__ import annotations
import logging

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import resume as resume_mod
import state_executor


@dataclass
class OpReport:
    op: str
    workspace: Path
    current_state: str
    n_spawns: int = 0
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    n_envelope_denials: int = 0
    n_schema_rejects: int = 0
    n_critic_fired: int = 0
    n_critic_timed_out: int = 0
    n_kb_merges: int = 0
    transitions: list[tuple[str, str]] = field(default_factory=list)
    last_action: Optional[str] = None  # ResumeAction value
    perf_ratio: Optional[float] = None
    pass_a: Optional[str] = None
    pass_b: Optional[str] = None


def _resolve_workspace_root() -> Path:
    return _HERE.parent.parent.parent.parent / "workspace"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        skip_current_item = False
        try:
            out.append(json.loads(line))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
    return out


def build_report(op: str, *, workspace: Optional[Path] = None) -> OpReport:
    """Read all diagnostic files for one op + roll up."""
    if workspace is None:
        workspace = _resolve_workspace_root() / op
    if not workspace.exists():
        return OpReport(op=op, workspace=workspace, current_state="MISSING")

    rep = OpReport(op=op, workspace=workspace, current_state=state_executor.current_state(workspace))

    # Envelope rollup: cost + duration + denials
    envs = _read_jsonl(workspace / ".cc_envelope_log.jsonl")
    rep.n_spawns = len(envs)
    for e in envs:
        rep.total_cost_usd += float(e.get("cost_usd", 0))
        rep.total_duration_s += float(e.get("duration_ms", 0)) / 1000
        denials = e.get("permission_denials") or []
        rep.n_envelope_denials += len(denials) if isinstance(denials, list) else 0

    # Schema rejects
    norms = _read_jsonl(workspace / ".schema_normalizations.log")
    rep.n_schema_rejects = sum(1 for n in norms if n.get("category") == "TERMINAL_REJECT")

    # Critic + KB
    crits = _read_jsonl(workspace / ".critic_invoke_log.jsonl")
    rep.n_critic_fired = len(crits)
    rep.n_critic_timed_out = sum(1 for c in crits if c.get("timed_out"))
    kbs = _read_jsonl(workspace / ".kb_merge_log.jsonl")
    rep.n_kb_merges = len(kbs)

    # Transitions
    trans = _read_jsonl(workspace / "state_transitions.jsonl")
    rep.transitions = [(t.get("from_state", "?"), t.get("to_state", "?")) for t in trans]

    # Resume action (UNKNOWN / NONE_TERMINAL / RESUMABLE / etc.)
    try:
        status = resume_mod.diagnose(op, workspace=workspace)
        rep.last_action = status.action.value
    except Exception:
        rep.last_action = "unknown"

    # Verification rollup
    vj = workspace / "verification.json"
    if vj.exists():
        try:
            v = json.loads(vj.read_text())
            prec = v.get("precision", {}) or {}
            rep.pass_a = (prec.get("pass_a") or {}).get("status")
            rep.pass_b = (prec.get("pass_b") or {}).get("status")
            perf = v.get("performance", {}) or {}
            r = perf.get("ratio")
            if r is not None:
                try:
                    rep.perf_ratio = float(r)
                except Exception as error:
                    logging.getLogger(__name__).debug(
                        "Recoverable operation failed.", exc_info=error
                    )
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    return rep


def fleet_status(*, root: Optional[Path] = None) -> list[OpReport]:
    if root is None:
        root = _resolve_workspace_root()
    if not root.exists():
        return []
    out: list[OpReport] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "PROGRESS.md").exists():
            continue
        out.append(build_report(d.name, workspace=d))
    return out


def path_coverage(reports: list[OpReport]) -> dict:
    """Aggregate transition pairs across ops + report unique paths.

    Returns {
      "unique_pairs": [(from, to), ...],   # sorted
      "ops_per_pair": {(from, to): [op_names]},
      "ops_with_terminal_finalize": [...],
      "ops_with_optimizer_path": [...],    # exercised V3.8.4 escalation
      "ops_with_probe_path": [...],
    }
    """
    pairs: dict[tuple, list[str]] = {}
    for r in reports:
        for p in r.transitions:
            # Coerce None / missing states to "?" so sorting doesn't blow up
            f, t = p
            pairs.setdefault((f or "?", t or "?"), []).append(r.op)
    terminal_finalize_ops: set[str] = set()
    optimizer_path_ops: set[str] = set()
    probe_path_ops: set[str] = set()
    for report in reports:
        has_terminal_finalize = False
        has_optimizer_path = False
        has_probe_path = False
        for transition in report.transitions:
            if transition in {
                ("await_worker", "finalize"),
                ("await_optimizer", "finalize"),
                ("await_probe", "finalize"),
            }:
                has_terminal_finalize = True
            if transition == ("await_worker", "await_optimizer"):
                has_optimizer_path = True
            if transition in {
                ("await_worker", "await_probe"),
                ("await_probe", "await_worker"),
            }:
                has_probe_path = True
        if has_terminal_finalize:
            terminal_finalize_ops.add(report.op)
        if has_optimizer_path:
            optimizer_path_ops.add(report.op)
        if has_probe_path:
            probe_path_ops.add(report.op)
    return {
        "unique_pairs": sorted(pairs.keys()),
        "ops_per_pair": {p: sorted(set(ops)) for p, ops in pairs.items()},
        "ops_with_terminal_finalize": sorted(terminal_finalize_ops),
        "ops_with_optimizer_path": sorted(optimizer_path_ops),
        "ops_with_probe_path": sorted(probe_path_ops),
    }


def cost_rollup(reports: list[OpReport]) -> dict:
    total_cost = sum(r.total_cost_usd for r in reports)
    total_dur = sum(r.total_duration_s for r in reports)
    total_spawns = sum(r.n_spawns for r in reports)
    return {
        "n_ops": len(reports),
        "total_cost_usd": round(total_cost, 4),
        "total_duration_s": int(total_dur),
        "total_spawns": total_spawns,
        "avg_cost_per_op": round(total_cost / max(len(reports), 1), 4),
    }


def denial_audit(reports: list[OpReport]) -> list[dict]:
    """Return per-op denial counts where >0."""
    return [
        {"op": r.op, "n_denials": r.n_envelope_denials,
         "n_spawns": r.n_spawns}
        for r in reports if r.n_envelope_denials > 0
    ]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def format_fleet_table(reports: list[OpReport]) -> str:
    if not reports:
        return "(no ops with PROGRESS.md found)"
    lines = []
    h = (f"{'op':<35s} {'state':<22s} {'action':<22s} "
         f"{'cost$':<8s} {'dur(s)':<8s} {'spawns':<7s} {'pa':<6s} {'pb':<6s} {'perf':<7s}")
    lines.append(h)
    lines.append("-" * len(h))
    for r in reports:
        perf_s = f"{r.perf_ratio:.2f}" if r.perf_ratio is not None else "-"
        lines.append(
            f"{r.op:<35s} {r.current_state:<22s} {(r.last_action or '?'):<22s} "
            f"{r.total_cost_usd:<8.2f} {r.total_duration_s:<8.0f} {r.n_spawns:<7d} "
            f"{(r.pass_a or '-'):<6s} {(r.pass_b or '-'):<6s} {perf_s:<7s}"
        )
    return "\n".join(lines)


def format_op_detail(rep: OpReport) -> str:
    lines = [
        f"== {rep.op} ==",
        f"  workspace:        {rep.workspace}",
        f"  current_state:    {rep.current_state}",
        f"  resume_action:    {rep.last_action}",
        f"  spawns:           {rep.n_spawns}",
        f"  total_cost_usd:   {rep.total_cost_usd:.4f}",
        f"  total_duration_s: {rep.total_duration_s:.1f}",
        f"  permission_denials: {rep.n_envelope_denials}",
        f"  schema_rejects:   {rep.n_schema_rejects}",
        f"  critic_fired:     {rep.n_critic_fired} ({rep.n_critic_timed_out} timed_out)",
        f"  kb_merges:        {rep.n_kb_merges}",
        f"  precision:        pass_a={rep.pass_a or '-'}, pass_b={rep.pass_b or '-'}",
        f"  performance:      ratio={rep.perf_ratio if rep.perf_ratio is not None else '-'}",
        f"  transitions ({len(rep.transitions)}):",
    ]
    for f, t in rep.transitions:
        lines.append(f"    {f:>30s} → {t}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Diagnose orchestrator runs")
    ap.add_argument("--op", help="show detailed report for one op")
    ap.add_argument("--paths", action="store_true",
                    help="show path coverage matrix across all ops")
    ap.add_argument("--cost", action="store_true",
                    help="show cost rollup")
    ap.add_argument("--denials", action="store_true",
                    help="audit permission denials across ops (>0 only)")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of text table")
    args = ap.parse_args()

    if args.op:
        rep = build_report(args.op)
        if args.json:
            print(json.dumps(rep.__dict__, default=str, indent=2))
        else:
            print(format_op_detail(rep))
        return 0

    reports = fleet_status()

    if args.paths:
        pc = path_coverage(reports)
        if args.json:
            # tuple keys → strings for JSON
            print(json.dumps({
                "unique_pairs": [f"{a}->{b}" for a, b in pc["unique_pairs"]],
                "ops_per_pair": {f"{a}->{b}": v for (a, b), v in pc["ops_per_pair"].items()},
                "ops_with_terminal_finalize": pc["ops_with_terminal_finalize"],
                "ops_with_optimizer_path": pc["ops_with_optimizer_path"],
                "ops_with_probe_path": pc["ops_with_probe_path"],
            }, indent=2))
        else:
            print("=== unique transition pairs across fleet ===")
            for a, b in pc["unique_pairs"]:
                ops_str = ", ".join(pc["ops_per_pair"][(a, b)])
                print(f"  {a:>30s} → {b:<25s}  [{ops_str}]")
            print(f"\nops reaching finalize:    {pc['ops_with_terminal_finalize']}")
            print(f"ops with optimizer path:  {pc['ops_with_optimizer_path']}")
            print(f"ops with probe path:      {pc['ops_with_probe_path']}")
        return 0

    if args.cost:
        cr = cost_rollup(reports)
        print(json.dumps(cr, indent=2))
        return 0

    if args.denials:
        da = denial_audit(reports)
        if not da:
            print("no permission_denials across any op (clean)")
        else:
            print(json.dumps(da, indent=2))
        return 0

    # Default: fleet table
    if args.json:
        print(json.dumps([r.__dict__ for r in reports], default=str, indent=2))
    else:
        print(format_fleet_table(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
