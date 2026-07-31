# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Aggregate per-op cost + timing from orchestrator_events.jsonl logs.

Task #46 (2026-05-14): REPORT.md columns for "Agent 耗时" (sum of agent
spawn durations) and "端到端耗时" (wall-clock from orchestrator.start
to orchestrator.terminal) were being filled manually + inconsistently.
This module reads the canonical event logs and emits machine-accurate
rows.

Usage:
  python3 src/scripts/orchestrator/gen_e2e_cost_report.py
    # Defaults to scanning arch22→arch35 migration and backward outputs.

  python3 src/scripts/orchestrator/gen_e2e_cost_report.py --root output/a3_to_a5_port
  python3 src/scripts/orchestrator/gen_e2e_cost_report.py --op foreach_abs
  python3 src/scripts/orchestrator/gen_e2e_cost_report.py --format markdown
  python3 src/scripts/orchestrator/gen_e2e_cost_report.py --format json --out costs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ROOTS = [
    _PROJECT_ROOT / "output" / "a3_to_a5_port" / "src" / "kernels",
    _PROJECT_ROOT / "output" / "backward_ops" / "src" / "kernels",
]


@dataclass
class OpCost:
    op: str
    archive: str  # relative archive path
    project: str  # "a3_to_a5_port" / "backward_ops"
    start_ts: Optional[str] = None
    terminal_ts: Optional[str] = None
    terminal_state: Optional[str] = None
    e2e_wall_s: float = 0.0          # terminal_ts - start_ts
    agent_dur_s: float = 0.0         # sum(spawn.complete.duration_s)
    agent_cost_usd: float = 0.0      # sum(spawn.complete.cost_usd)
    spawn_count: int = 0
    spawns_by_type: dict[str, int] = field(default_factory=dict)
    invocations: int = 1             # orchestrator.start count
    errors: list[str] = field(default_factory=list)


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _format_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def scan_op(events_jsonl: Path, op: str, project: str, archive: str) -> OpCost:
    cost = OpCost(op=op, archive=archive, project=project)
    start_dt: Optional[datetime] = None
    terminal_dt: Optional[datetime] = None
    invocations = 0
    try:
        for line in events_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = d.get("event")
            data = d.get("data", {}) or {}
            ts = d.get("ts")
            if evt == "orchestrator.start":
                invocations += 1
                if start_dt is None:
                    cost.start_ts = ts
                    start_dt = _parse_ts(ts)
            elif evt == "orchestrator.spawn.complete":
                cost.spawn_count += 1
                cost.agent_dur_s += float(data.get("duration_s", 0) or 0)
                cost.agent_cost_usd += float(data.get("cost_usd", 0) or 0)
                a_type = data.get("agent_type", "?")
                cost.spawns_by_type[a_type] = cost.spawns_by_type.get(a_type, 0) + 1
            elif evt == "orchestrator.terminal":
                cost.terminal_ts = ts
                cost.terminal_state = data.get("state")
                terminal_dt = _parse_ts(ts)
    except FileNotFoundError:
        cost.errors.append(f"events file missing: {events_jsonl}")
        return cost
    cost.invocations = max(invocations, 1)
    if start_dt and terminal_dt:
        cost.e2e_wall_s = (terminal_dt - start_dt).total_seconds()
    elif start_dt:
        # No terminal — op still in flight or aborted without terminal event
        cost.errors.append("no orchestrator.terminal event — op may still be in flight")
    return cost


def _merge_workspace_terminal(cost: OpCost, op: str) -> None:
    """Supplement archive log with workspace log's later events.

    The archive log is captured at finalize promotion time, which happens
    BEFORE the orchestrator writes the `orchestrator.terminal` event to
    workspace (terminal lands ~20s after promotion). So archive-only
    scans return e2e_wall_s=0 + terminal_state=None. Fix: also read
    workspace/<op>/orchestrator_events.jsonl if present and patch in
    the terminal event when found.
    """
    ws_events = _PROJECT_ROOT / "workspace" / op / "orchestrator_events.jsonl"
    if not ws_events.is_file() or cost.terminal_ts is not None:
        return
    try:
        for line in ws_events.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "orchestrator.terminal":
                cost.terminal_ts = d.get("ts")
                cost.terminal_state = (d.get("data") or {}).get("state")
        start_dt = _parse_ts(cost.start_ts or "")
        term_dt = _parse_ts(cost.terminal_ts or "")
        if start_dt and term_dt:
            cost.e2e_wall_s = (term_dt - start_dt).total_seconds()
            # Drop the "no terminal" error if we now have one
            cost.errors = [e for e in cost.errors if "orchestrator.terminal" not in e]
    except OSError:
        pass


def scan_root(root: Path) -> list[OpCost]:
    """Walk a root looking for `*/orchestrator_events.jsonl`."""
    rows: list[OpCost] = []
    if not root.is_dir():
        return rows
    project = root.name if root.parent.name == "output" else root.parent.parent.name
    for op_dir in sorted(root.iterdir()):
        if not op_dir.is_dir():
            continue
        events = op_dir / "orchestrator_events.jsonl"
        if not events.is_file():
            continue
        try:
            archive_rel = str(op_dir.resolve().relative_to(_PROJECT_ROOT))
        except ValueError:
            archive_rel = str(op_dir)
        cost = scan_op(events, op=op_dir.name, project=project, archive=archive_rel)
        _merge_workspace_terminal(cost, op_dir.name)
        rows.append(cost)
    return rows


def format_markdown(rows: list[OpCost]) -> str:
    lines = [
        "| Op | Agent 耗时 | 端到端耗时 | Spawn 数 | Cost (USD) | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        spawn_breakdown = ",".join(
            f"{t.split('-')[-1]}={n}" for t, n in sorted(r.spawns_by_type.items())
        ) or "-"
        terminal = r.terminal_state or "(no terminal)"
        lines.append(
            f"| {r.op} | {_format_dur(r.agent_dur_s)} "
            f"| {_format_dur(r.e2e_wall_s)} "
            f"| {r.spawn_count} ({spawn_breakdown}) "
            f"| ${r.agent_cost_usd:.2f} "
            f"| {terminal} |"
        )
    # Totals row
    if rows:
        total_dur = sum(r.agent_dur_s for r in rows)
        total_wall = sum(r.e2e_wall_s for r in rows)
        total_spawn = sum(r.spawn_count for r in rows)
        total_cost = sum(r.agent_cost_usd for r in rows)
        lines.append(
            f"| **TOTAL ({len(rows)} ops)** | **{_format_dur(total_dur)}** "
            f"| **{_format_dur(total_wall)}** | **{total_spawn}** "
            f"| **${total_cost:.2f}** | — |"
        )
    return "\n".join(lines)


def format_json(rows: list[OpCost]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", action="append", default=None,
                   help="Archive root to scan (repeatable). "
                        f"Default: {[str(r.relative_to(_PROJECT_ROOT)) for r in _DEFAULT_ROOTS]}")
    p.add_argument("--op", default=None,
                   help="Filter to a single op name (matches archive dir name)")
    p.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p.add_argument("--out", default=None, help="Write output to file (default: stdout)")
    args = p.parse_args(argv)

    roots = [Path(r) for r in (args.root or [])] or _DEFAULT_ROOTS
    rows: list[OpCost] = []
    for root in roots:
        rows.extend(scan_root(root))
    if args.op:
        rows = [r for r in rows if r.op == args.op]
    rows.sort(key=lambda r: (r.project, r.op))

    if args.format == "markdown":
        out = format_markdown(rows)
    else:
        out = format_json(rows)

    if args.out:
        Path(args.out).write_text(out + "\n")
        print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    else:
        print(out)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
