# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test orchestrator events.py JSONL emitter (Track C #1)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import events  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    return tmp_path / "test_op"


def test_emit_writes_per_op_log(ws):
    events.emit(ws, "orchestrator.start", data={"op": "test"})
    log = ws / "orchestrator_events.jsonl"
    assert log.exists()
    content = log.read_text().strip()
    entry = json.loads(content)
    assert entry["event"] == "orchestrator.start"
    assert entry["data"] == {"op": "test"}
    assert entry["source"] == "orchestrator"


def test_emit_derives_op_from_workspace_name(ws):
    entry = events.emit(ws, "orchestrator.iter", data={"iter": 0})
    assert entry["op"] == "test_op"


def test_emit_explicit_op_overrides_workspace(ws):
    entry = events.emit(ws, "batch.op_dispatched", op="other_op")
    assert entry["op"] == "other_op"


def test_emit_appends_multiple_entries(ws):
    events.emit(ws, "orchestrator.start")
    events.emit(ws, "orchestrator.iter", data={"iter": 0})
    events.emit(ws, "orchestrator.terminal", data={"state": "finalize"})
    entries = events.read_events(ws)
    assert len(entries) == 3
    assert [e["event"] for e in entries] == [
        "orchestrator.start",
        "orchestrator.iter",
        "orchestrator.terminal",
    ]


def test_emit_rejects_invalid_source(ws):
    with pytest.raises(ValueError, match="source"):
        events.emit(ws, "orchestrator.start", source="bogus")


def test_emit_writes_to_global_log(tmp_path):
    glog = tmp_path / "global_events.jsonl"
    events.emit(None, "batch.start", source="batch", global_log=glog)
    assert glog.exists()
    entry = json.loads(glog.read_text().strip())
    assert entry["event"] == "batch.start"
    assert entry["source"] == "batch"
    assert entry["op"] is None


def test_emit_writes_both_per_op_and_global(ws, tmp_path):
    glog = tmp_path / "global.jsonl"
    events.emit(ws, "orchestrator.iter", data={"iter": 1}, global_log=glog)
    assert (ws / "orchestrator_events.jsonl").exists()
    assert glog.exists()


def test_read_events_empty_when_no_log(ws):
    assert events.read_events(ws) == []


def test_read_events_skips_malformed_lines(ws):
    log = ws / "orchestrator_events.jsonl"
    ws.mkdir()
    log.write_text(
        '{"event": "ok", "op": "x", "data": {}}\n'
        'this is not json\n'
        '{"event": "ok2", "op": "x", "data": {}}\n'
    )
    entries = events.read_events(ws)
    assert len(entries) == 2
    assert entries[0]["event"] == "ok"
    assert entries[1]["event"] == "ok2"


def test_filter_by_prefix(ws):
    events.emit(ws, "orchestrator.start")
    events.emit(ws, "orchestrator.iter")
    events.emit(ws, "lane.allocated", source="lane_pool")
    entries = events.read_events(ws)
    orch = events.filter_events(entries, prefix="orchestrator.")
    assert len(orch) == 2
    lane = events.filter_events(entries, prefix="lane.")
    assert len(lane) == 1


def test_filter_by_source(ws):
    events.emit(ws, "orchestrator.start", source="orchestrator")
    events.emit(ws, "lane.allocated", source="lane_pool")
    entries = events.read_events(ws)
    lp = events.filter_events(entries, source="lane_pool")
    assert len(lp) == 1
    assert lp[0]["event"] == "lane.allocated"


def test_emit_env_global_log_override(ws, tmp_path, monkeypatch):
    """ORCHESTRATOR_GLOBAL_EVENTS env var routes default global log."""
    glog = tmp_path / "env_global.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_GLOBAL_EVENTS", str(glog))
    events.emit(ws, "orchestrator.start")
    assert glog.exists()


def test_emit_lane_field(ws):
    entry = events.emit(ws, "orchestrator.spawn.start", lane=0)
    assert entry["lane"] == 0


def test_emit_ts_iso_format(ws):
    entry = events.emit(ws, "orchestrator.start")
    # ISO 8601 with Z suffix
    assert entry["ts"].endswith("Z")
    assert "T" in entry["ts"]
