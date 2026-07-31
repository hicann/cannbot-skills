# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Structured event stream for the Python orchestrator (DEBT-077 Track C).

Codex C3: orchestrator should NOT directly know about Discord (or any
specific notification channel). It emits structured JSONL events; consumer
processes tail the stream and forward as needed.

Two streams:
  - per-op:  workspace/<op>/orchestrator_events.jsonl
  - global:  src/workspace/.events.jsonl  (project-relative; created on
             demand for batch / cross-op aggregation)

Event schema (canonical):
  {
    "ts":      "<ISO 8601 Z>",
    "event":   "<dot-separated kind, e.g. orchestrator.spawn.complete>",
    "op":      "<op name or null for cross-op events>",
    "lane":    <int or null>,
    "data":    {...},     # event-specific payload
    "source":  "orchestrator" | "batch" | "lane_pool" | "resume",
  }

Event kinds (V1 starter set; extensible):
  orchestrator.start
  orchestrator.iter           {iter, state}
  orchestrator.spawn.start    {agent_type, lane, spawn_index}
  orchestrator.spawn.complete {agent_type, success, duration_s, cost_usd}
  orchestrator.spawn.failed   {agent_type, reason}
  orchestrator.transition     {from_state, to_state, rationale}
  orchestrator.kb_merge       {success}
  orchestrator.critic_fired   {trigger}
  orchestrator.pause          {state, reason}
  orchestrator.terminal       {state, exit_code}
  batch.start                 {ops, max_lanes}
  batch.op_dispatched         {op, lane}
  batch.op_complete           {op, exit_code}
  batch.complete              {n_ops, n_pass, n_fail}
  lane.allocated              {lane, op}
  lane.released               {lane}
  resume.replay               {n_events_replayed}

The fixed schema makes it easy to filter/grep — `jq 'select(.event |
startswith("orchestrator.spawn"))' events.jsonl`.
"""
from __future__ import annotations
import logging

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Optional


VALID_SOURCES = ("orchestrator", "batch", "lane_pool", "resume")


def emit(
    workspace: Optional[Path],
    event: str,
    *,
    data: Optional[dict[str, Any]] = None,
    op: Optional[str] = None,
    lane: Optional[int] = None,
    source: str = "orchestrator",
    global_log: Optional[Path] = None,
) -> dict:
    """Append one event to per-op + (optionally) global JSONL streams.

    Args:
        workspace: per-op workspace dir; if None, only writes to global_log.
        event: dot-separated kind. SHOULD start with one of the source prefixes
               so consumers can filter; not enforced.
        data: event-specific payload. None → empty dict.
        op: op name. If workspace is given and op is None, derived from
            workspace.name.
        lane: NPU id (0/1/2). None for cross-op events.
        source: actor that emitted the event (must be in VALID_SOURCES).
        global_log: explicit global stream path; if None, defaults to
                    src/workspace/.events.jsonl when present.

    Returns:
        the entry dict written (useful for tests + chaining).

    Raises:
        ValueError: if source not in VALID_SOURCES.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"events.emit: source={source!r} not in {VALID_SOURCES}")

    if op is None and workspace is not None:
        op = workspace.name

    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        "op": op,
        "lane": lane,
        "data": data or {},
        "source": source,
    }
    line = json.dumps(entry) + "\n"

    # Per-op log
    if workspace is not None:
        per_op = workspace / "orchestrator_events.jsonl"
        per_op.parent.mkdir(parents=True, exist_ok=True)
        with open(per_op, "a") as f:
            f.write(line)

    # Global log (resolved or explicit)
    if global_log is None:
        # Default: ${PROJECT_ROOT}/src/workspace/.events.jsonl
        # We use src/workspace/.lanes/ already; this is the global aggregator.
        env = os.environ.get("ORCHESTRATOR_GLOBAL_EVENTS")
        if env:
            global_log = Path(env)
    if global_log is not None:
        global_log.parent.mkdir(parents=True, exist_ok=True)
        with open(global_log, "a") as f:
            f.write(line)

    return entry


def read_events(workspace: Path) -> list[dict]:
    """Read per-op events. Empty list if file missing."""
    log = workspace / "orchestrator_events.jsonl"
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
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


def filter_events(events: list[dict], *, prefix: Optional[str] = None,
                  source: Optional[str] = None) -> list[dict]:
    """Lightweight filter — by event-kind prefix and/or source."""
    out = events
    if prefix is not None:
        out = [e for e in out if e.get("event", "").startswith(prefix)]
    if source is not None:
        out = [e for e in out if e.get("source") == source]
    return out
