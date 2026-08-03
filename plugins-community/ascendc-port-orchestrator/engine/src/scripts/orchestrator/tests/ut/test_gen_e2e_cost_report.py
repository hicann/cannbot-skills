# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for gen_e2e_cost_report — aggregates per-op cost + timing.

Task #46 (2026-05-14).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import gen_e2e_cost_report as gec  # noqa: E402


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _event(event: str, ts: str, **data) -> dict:
    return {"ts": ts, "event": event, "data": data}


def test_scan_op_aggregates_spawn_costs(tmp_path: Path) -> None:
    events = tmp_path / "orchestrator_events.jsonl"
    _write_events(events, [
        _event("orchestrator.start", "2026-05-14T05:00:00Z"),
        _event("orchestrator.spawn.complete", "2026-05-14T05:05:00Z",
               agent_type="aog-kernel-worker", duration_s=300.0, cost_usd=2.50),
        _event("orchestrator.spawn.complete", "2026-05-14T05:10:00Z",
               agent_type="aog-kernel-worker", duration_s=250.0, cost_usd=1.80),
        _event("orchestrator.terminal", "2026-05-14T05:15:00Z", state="done"),
    ])
    cost = gec.scan_op(events, op="myop", project="testproj", archive="output/testproj/myop")
    assert cost.spawn_count == 2
    assert cost.agent_dur_s == 550.0
    assert cost.agent_cost_usd == pytest.approx(4.30)
    assert cost.spawns_by_type == {"aog-kernel-worker": 2}
    assert cost.terminal_state == "done"
    assert cost.e2e_wall_s == 900.0  # 15 min


def test_scan_op_no_terminal_records_error(tmp_path: Path) -> None:
    events = tmp_path / "orchestrator_events.jsonl"
    _write_events(events, [
        _event("orchestrator.start", "2026-05-14T05:00:00Z"),
        _event("orchestrator.spawn.complete", "2026-05-14T05:05:00Z",
               agent_type="aog-kernel-worker", duration_s=100.0, cost_usd=1.00),
    ])
    cost = gec.scan_op(events, op="stalled", project="t", archive="x")
    assert cost.terminal_state is None
    assert cost.e2e_wall_s == 0.0
    assert any("terminal" in e for e in cost.errors)


def test_scan_op_counts_multiple_invocations(tmp_path: Path) -> None:
    """Resume re-fires orchestrator.start; tool should count invocations."""
    events = tmp_path / "orchestrator_events.jsonl"
    _write_events(events, [
        _event("orchestrator.start", "2026-05-14T05:00:00Z"),
        _event("orchestrator.spawn.complete", "2026-05-14T05:05:00Z",
               agent_type="aog-kernel-worker", duration_s=300.0, cost_usd=2.50),
        _event("orchestrator.start", "2026-05-14T05:30:00Z"),  # resume
        _event("orchestrator.spawn.complete", "2026-05-14T05:35:00Z",
               agent_type="aog-kernel-worker", duration_s=200.0, cost_usd=1.50),
        _event("orchestrator.terminal", "2026-05-14T05:40:00Z", state="done"),
    ])
    cost = gec.scan_op(events, op="o", project="t", archive="x")
    assert cost.invocations == 2
    assert cost.spawn_count == 2
    # start_ts must be the FIRST start (e2e wall starts from there)
    assert cost.start_ts == "2026-05-14T05:00:00Z"
    assert cost.e2e_wall_s == 2400.0  # 40 min


def test_format_markdown_with_totals(tmp_path: Path) -> None:
    rows = [
        gec.OpCost(op="op_a", archive="x", project="p",
                   agent_dur_s=600.0, e2e_wall_s=900.0,
                   agent_cost_usd=2.50, spawn_count=2,
                   spawns_by_type={"aog-kernel-worker": 2},
                   terminal_state="done"),
        gec.OpCost(op="op_b", archive="x", project="p",
                   agent_dur_s=1800.0, e2e_wall_s=2400.0,
                   agent_cost_usd=15.00, spawn_count=5,
                   spawns_by_type={"aog-kernel-worker": 3, "aog-precision-probe": 2},
                   terminal_state="abort"),
    ]
    md = gec.format_markdown(rows)
    assert "| op_a |" in md
    assert "| op_b |" in md
    assert "**TOTAL (2 ops)**" in md
    assert "$17.50" in md  # 2.50 + 15.00


def test_format_markdown_formats_duration_thresholds() -> None:
    rows = [
        gec.OpCost(
            op="short",
            archive="x",
            project="p",
            agent_dur_s=30,
            e2e_wall_s=75,
        ),
        gec.OpCost(
            op="long",
            archive="x",
            project="p",
            agent_dur_s=3725,
        ),
    ]

    markdown = gec.format_markdown(rows)

    assert "| short | 30s | 1m15s |" in markdown
    assert "| long | 1h02m | 0s |" in markdown


def test_scan_root_skips_non_dirs_and_missing_events(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    # Op A with events
    (root / "op_a").mkdir()
    _write_events(root / "op_a" / "orchestrator_events.jsonl", [
        _event("orchestrator.start", "2026-05-14T05:00:00Z"),
        _event("orchestrator.terminal", "2026-05-14T05:10:00Z", state="done"),
    ])
    # Op B without events
    (root / "op_b").mkdir()
    # Stray file (not a dir)
    (root / "stray.txt").write_text("ignore")
    rows = gec.scan_root(root)
    ops = sorted(r.op for r in rows)
    assert ops == ["op_a"], f"expected only op_a, got {ops}"
