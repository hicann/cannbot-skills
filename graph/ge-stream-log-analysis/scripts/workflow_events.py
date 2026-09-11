#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified, graph-centric GE stream log analysis.

The public CLI keeps the historical JSON compatibility view, while this
module owns one workflow per compiled graph.  Compile and runtime evidence is
attached to that workflow as soon as a unique identity can be established;
unbound V2 blocks are kept only as temporary candidates until the input has
been consumed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import heapq
from pathlib import Path
import re
from typing import Any, Iterable, Optional, Sequence

from workflow_correlation import _parse_timestamp


CORRELATION_WINDOW_SECONDS = 5.0
INIT_DEDUP_WINDOW_SECONDS = 0.001

INFO_LINE = re.compile(r"^\s*\[INFO\]")
LOG_PID = re.compile(r"\b[A-Z]+\((\d+)(?:,[^)]*)?\):")
LOG_TIMESTAMP = re.compile(
    r"(?P<base>\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac1>\d+))?(?:\.(?P<frac2>\d+))?"
)
LOG_SOURCE = re.compile(r"\[(?P<source>[^]:]+):(?P<line>\d+)\]")
LOG_FUNCTION = re.compile(
    r"\]\s*(?P<context>\d+)\s+(?P<function>(?:[A-Za-z_][A-Za-z0-9_]*::)*"
    r"[A-Za-z_][A-Za-z0-9_<>~]*)\s*:\s*(?P<message>.*)$"
)
BARE_FUNCTION = re.compile(
    r"^\s*\[INFO\]\s+(?P<function>(?:[A-Za-z_][A-Za-z0-9_]*::)*"
    r"[A-Za-z_][A-Za-z0-9_<>~]*)\s*:\s*(?P<message>.*)$"
)

GRAPH_BUILD = re.compile(r"Begin to build (known|unknown) shape graph\[([^]]+)\]", re.I)
GRAPH_SCOPE = re.compile(
    r"Start assign attached stream for graph\s+([^,\s]+)\s+with subgraph num:\s*(\d+)",
    re.I,
)
GRAPH_RELATION = re.compile(
    r"\[GraphRelation\].*?parent_graph:\s*([^,]+),\s*parent_node:\s*([^,]+),\s*"
    r"child_graph:\s*([^,]+),\s*subgraph_index:\s*(\d+)",
    re.I,
)
STATIC_EXIT = re.compile(
    r"At last, root graph:\s*([^,]+),\s*total stream num:\s*(\d+),\s*"
    r"main stream num:\s*(\d+),\s*attached stream num:\s*(\d+)",
    re.I,
)
STATIC_PHYSICAL_EXIT = re.compile(
    r"After SplitStreamAndRefreshTaskDef, graph:\s*([^,]+),\s*stream num:\s*(\d+),\s*"
    r"notify num:\s*(\d+),\s*event num:\s*(\d+)",
    re.I,
)
DYNAMIC_EXIT = re.compile(
    r"^Graph:\s*(?P<graph>[^,]+),\s*stream num:\s*(?P<stream>\d+),\s*"
    r"event num:\s*(?P<event>\d+)\.?\s*$",
    re.I,
)
DYNAMIC_ASSIGN = re.compile(
    r"Assign stream_id:\s*(-?\d+)\s+for engine:\s*([^,]+),\s*subgraph:\s*(.+?)\.\s*$",
    re.I,
)
DYNAMIC_SUBGRAPH = re.compile(
    r"\[Assign\]\[StreamId\]\s*(-?\d+)\s+for Subgraph\s+(.+?)\s+\(engine:\s*([^)]+)\)",
    re.I,
)
DYNAMIC_SUBGRAPH_GRAPH = re.compile(r"^(?P<root>.+)_sub_\d+_unknown?$", re.I)
DYNAMIC_REASSIGN = re.compile(
    r"Node:\s*([^,]+),\s*stream_label:\s*([^,]+),\s*reassign stream id:\s*(-?\d+)",
    re.I,
)
DYNAMIC_REFRESH = re.compile(
    r"Refresh stream by node ids of graph:\s*([^,]+),\s*stream_id:\s*(-?\d+),\s*"
    r"type:\s*([^,]+),\s*name:\s*(.+?)\.\s*$",
    re.I,
)
FINAL_OPERATOR = re.compile(
    r"Op\s+\[(?P<name>[^]]+)\]\s+OpType\s+\[(?P<type>[^]]+)\].*?"
    r"logic stream id is\s*(?P<stream>-?\d+)",
    re.I,
)
ATTACHED_STREAM = re.compile(
    r"Op\s+\[(?P<name>[^]]+)\]\s+OpType\s+\[(?P<type>[^]]+)\].*?"
    r"logic attached stream id is\s*(?P<streams>-?\d+(?:\s+-?\d+)*)\s*\.?\s*$",
    re.I,
)
STATIC_SPLIT_NODE = re.compile(
    r"op\s+name\s+\[(?P<name>[^]]+)\]\s+is\s+split\s+to\s+new\s+stream\s+"
    r"id\s+(?P<stream>-?\d+)\s*,\s*is\s+attached\s+stream:\s*(?P<attached>\d+)",
    re.I,
)
STATIC_SPLIT_LIMIT = re.compile(
    r"stream\[(?P<old>-?\d+)\].*?split\s+stream\s+to\s+(?:new\s+)?(?P<new>-?\d+),\s*"
    r"first\s+node\[name:\s*(?P<name>[^,\]]+),\s*type:\s*(?P<type>[^,\]]+),\s*"
    r"owner\s+graph:\s*(?P<graph>[^]]+)\]",
    re.I,
)
SYNC_EXIT = re.compile(
    r"After\s+InsertSyncNodesByLogicStream,\s*graph:\s*(?P<graph>[^,]+),\s*"
    r"stream\s+num:\s*(?P<stream>\d+),\s*notify\s+num:\s*(?P<notify>\d+),\s*"
    r"event\s+num:\s*(?P<event>\d+)",
    re.I,
)
NODE_STREAM = re.compile(
    r"node:\s*(.*?),\s*type\s+(.*?),\s*topo id:\s*-?\d+,\s*logical stream id:\s*(-?\d+)",
    re.I,
)
NODE_ASSIGN = re.compile(
    r"Node\s+(.+?)\s+(?:assigned stream|reassign to stream)\s+(-?\d+)", re.I
)
NODE_REFRESH = re.compile(
    r"(?:node|Node)\s+(.+?)\s+(?:set stream id from|refresh stream id from)\s+-?\d+\s+to\s+(-?\d+)",
    re.I,
)
NODE_UPDATED = re.compile(
    r"Stream of node\s+(.+?)\s+has been updated from\s+-?\d+\s+to\s+(-?\d+)",
    re.I,
)

RUNTIME_INIT = re.compile(
    r"InitRuntimeParams:.*?model_id=(\d+).*?stream_num\s*[:=]\s*(\d+),\s*"
    r"notify_num\s*[:=]\s*(\d+),\s*event_num\s*[:=]\s*(\d+),\s*"
    r"label_num\s*[:=]\s*(\d+)",
    re.I,
)
RUNTIME_CREATE = re.compile(
    r"Create new stream:\s*([^,]+),\s*rt stream id:\s*(\d+),\s*rt model id:\s*(\d+),\s*"
    r"priority:\s*(-?\d+),\s*stream flag:\s*(\d+),\s*task num:\s*(\d+)",
    re.I,
)
RUNTIME_BIND = re.compile(
    r"Logical stream index:\s*(\d+),\s*rtstream:\s*(\d+),\s*model:\s*(\d+),\s*"
    r"stream flag:\s*(\d+)",
    re.I,
)
RUNTIME_TOTAL = re.compile(
    r"model total stream num:\s*(\d+),\s*model stream num:\s*(\d+),\s*"
    r"(?:hccl\s+)?follow stream num:\s*(\d+)",
    re.I,
)
V2_EXECUTOR_ANCHOR = re.compile(
    r"Build\s+RT2\s+executor\s+for\s+root\s+compute\s+graph\[(?P<graph>[^]]+)\],\s*"
    r"model\[(?P<model>[^]]+)\]\.?",
    re.I,
)
V2_COLLECT_STREAM = re.compile(
    r"Collect\s+rt2\s+stream,\s*get\s+rts\s+stream\s+(?P<pointer>\S+)\s+"
    r"from\s+logical\s+stream\s+(?P<logical_index>\d+),\s*"
    r"rts_stream_id\s*:\s*(?P<rt_stream_id>-?\d+)",
    re.I,
)
RUNTIME_TASK = re.compile(
    r"KernelTaskInfo Init Success,(?:\s*node\s*:(.*?),)?\s*logic stream id:\s*(\d+),\s*stream:\s*([^\s.]+)",
    re.I,
)
RUNTIME_NODE_LOGIC = re.compile(
    r"(?:node\s*:?[ ]*|op:)([^,(:]+).*?logic(?:al)? stream id:\s*(\d+)", re.I
)
BATCH_SIZE = re.compile(r"aclmdlSetDynamicBatchSize.*?batchSize\[(\d+)\]", re.I)
BATCH_LABEL = re.compile(r"current batch label:([^\s,]+)", re.I)
MODEL_NAME = re.compile(r"model name set.*?name=([^\s,]+)", re.I)
GRAPH_NAME = re.compile(r"graph_name:\s*([^\s,]+)", re.I)
ACTIVE_STREAM = re.compile(
    r"StreamActive_(\d+).*?(?:active_stream_id=|active stream id:)\s*(\d+)", re.I
)

STREAM_NUM_SEPARATOR = r"(?:\s*(?:[:=]|\bis\b)\s*|\s+)"
V2_REUSABLE = re.compile(
    rf"\breusable[ _]stream[ _]num{STREAM_NUM_SEPARATOR}(\d+)", re.I
)
V2_ATTACHED = re.compile(
    rf"\battached[ _]stream[ _]num{STREAM_NUM_SEPARATOR}(\d+)", re.I
)
V2_ROOT_TOTAL = re.compile(
    rf"(?:Root\s+graph\s+total[ _]stream[ _]num|Root\s+model\s+[^,]+,\s*total[ _]stream[ _]num)"
    rf"{STREAM_NUM_SEPARATOR}(?P<value>\d+)",
    re.I,
)
V2_ROOT_MODEL = re.compile(
    r"Root\s+model\s+(?P<name>[^,]+),\s*total[ _]stream[ _]num", re.I
)
V2_SUBMODEL = re.compile(
    rf"Static\s+sub\s+model\s+(?P<name>[^,]+),\s*stream[ _]num{STREAM_NUM_SEPARATOR}(?P<stream>\d+)"
    rf"(?:,\s*event[ _]num{STREAM_NUM_SEPARATOR}(?P<event>\d+))?"
    rf"(?:,\s*notify[ _]num{STREAM_NUM_SEPARATOR}(?P<notify>\d+))?",
    re.I,
)
V2_EVENT = re.compile(rf"\bevent[ _]num{STREAM_NUM_SEPARATOR}(\d+)", re.I)
V2_NOTIFY = re.compile(rf"\bnotify[ _]num{STREAM_NUM_SEPARATOR}(\d+)", re.I)
KERNEL_STREAM = re.compile(
    r"\[KernelTrace\]\s*\[SplitRtStreams(?:_[^]]+)?\]\s*"
    r"Get\s+rts\s+stream\s+.*?from\s+logical\s+stream\s*(-?\d+)\s*,\s*"
    r"rts[ _]+stream[ _]*id\s*[:=]\s*(-?\d+)",
    re.I,
)
KERNEL_NOTIFY = re.compile(
    r"\[KernelTrace\].*?Get rts notify .*?from logical notify\s*(-?\d+)", re.I
)
KERNEL_SEND = re.compile(
    r"\[KernelTrace\].*?Sent event\s*(-?\d+)\s+RT event\s+(\S+)\s+from stream\s*(-?\d+)",
    re.I,
)
KERNEL_WAIT = re.compile(
    r"\[KernelTrace\].*?Waited event\s*(-?\d+)\s+RT event\s+(\S+)\s+at stream\s*(-?\d+)",
    re.I,
)


@dataclass
class LogRecord:
    path: Path
    line: int
    raw: str
    pid: Optional[str] = None
    context: Optional[str] = None
    timestamp: Optional[str] = None
    source: Optional[str] = None
    function: Optional[str] = None
    message: str = ""
    producer: Optional[str] = None
    kernel_tag: Optional[str] = None
    qualified: bool = False
    order: int = 0
    stage_hint: str = "generic"


@dataclass
class Event:
    kind: str
    record: LogRecord
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileSpec:
    """Identity and shape inputs used to initialise a compile session."""

    graph: str
    pid: Optional[str]
    context: Optional[str]
    timestamp: Optional[str]
    sequence: int
    shape: str


def evidence(record: LogRecord, keyword: str) -> dict[str, Any]:
    level = re.search(r"\[(DEBUG|INFO|WARNING|WARN|ERROR|EVENT)\]", record.raw)
    return {
        "file": str(record.path),
        "line": record.line,
        "keyword": keyword,
        "level": level.group(1) if level else "INFO",
        "excerpt": record.raw.rstrip()[:500],
    }


def _timestamp_from_match(match: re.Match[str]) -> Optional[str]:
    parts = [match.group("frac1"), match.group("frac2")]
    fraction = "".join(item for item in parts if item)[:6].ljust(6, "0")
    try:
        value = datetime.strptime(match.group("base"), "%Y-%m-%d-%H:%M:%S")
    except ValueError:
        return None
    return f"{value:%Y-%m-%d %H:%M:%S}.{fraction}"


def parse_record(
    path: Path, line_number: int, raw: str, order: int, stage: str
) -> LogRecord:
    pid_match = LOG_PID.search(raw)
    timestamp_match = LOG_TIMESTAMP.search(raw)
    source_match = LOG_SOURCE.search(raw)
    function_match = LOG_FUNCTION.search(raw)
    bare_match = BARE_FUNCTION.match(raw) if function_match is None else None
    function = None
    message = raw.strip()
    context = None
    if function_match:
        context = function_match.group("context")
        function = function_match.group("function")
        message = function_match.group("message").strip()
    elif bare_match:
        function = bare_match.group("function")
        message = bare_match.group("message").strip()
    context_match = re.search(r"\]\s*(\d+)\s+(?=[A-Za-z_\[])", raw)
    if context is None and context_match:
        context = context_match.group(1)
    producer_match = re.search(r"\[INFO\]\s+([A-Z][A-Z0-9_]*)\(", raw)
    producer = producer_match.group(1) if producer_match else None
    kernel_match = re.search(r"\[KernelTrace\]\s*\[([^]]+)\]", raw)
    return LogRecord(
        path=path,
        line=line_number,
        raw=raw,
        pid=pid_match.group(1) if pid_match else None,
        context=context,
        timestamp=_timestamp_from_match(timestamp_match) if timestamp_match else None,
        source=source_match.group("source") if source_match else None,
        function=function,
        message=message,
        producer=producer,
        kernel_tag=kernel_match.group(1) if kernel_match else None,
        qualified=bool(producer == "GE" and source_match and context and function),
        order=order,
        stage_hint=stage,
    )


def _function_matches(record: LogRecord, names: Sequence[str]) -> bool:
    if not record.function:
        return False
    short = record.function.rsplit("::", 1)[-1]
    return any(record.function == name or short == name for name in names)


def _allowed(
    record: LogRecord, functions: Sequence[str] = (), sources: Sequence[str] = ()
) -> bool:
    if not record.qualified:
        return True
    source = (record.source or "").lower()
    return (not functions or _function_matches(record, functions)) and (
        not sources or any(source.endswith(item.lower()) for item in sources)
    )


def _kernel_allowed(record: LogRecord, sources: Sequence[str]) -> bool:
    if record.producer not in (None, "GE"):
        return False
    source = (record.source or "").lower()
    return not source or any(source.endswith(item.lower()) for item in sources)


def _iter_path_records(path: Path, stage: str) -> Iterable[LogRecord]:
    """Yield accepted records from one file without retaining the file."""
    order = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            if not INFO_LINE.search(raw):
                continue
            producer = re.search(r"\[INFO\]\s+([A-Z][A-Z0-9_]*)\(", raw)
            if producer and producer.group(1) != "GE":
                continue
            if re.search(r"(?:\[RUNTIME\]|\bRUNTIME\s*\()", raw):
                continue
            if "ModelExecuteTask" in raw or "ConstructSqeForModelExecuteTask" in raw:
                continue
            yield parse_record(path, number, raw, order, stage)
            order += 1


def iter_records(paths: Sequence[Path], stages: dict[Path, str]) -> Iterable[LogRecord]:
    order = 0
    for path in paths:
        for record in _iter_path_records(path, stages.get(path, "generic")):
            record.order = order
            order += 1
            yield record


def iter_records_ordered(
    paths: Sequence[Path], stages: dict[Path, str]
) -> Iterable[LogRecord]:
    """Merge timestamped files with one-record-per-file memory usage.

    If a file has no timestamped GE record, retain the normal file/line order;
    this keeps compact legacy fixtures deterministic and avoids inventing a
    cross-file clock for unqualified lines.
    """
    iterators = []
    for path in paths:
        iterators.append(iter(_iter_path_records(path, stages.get(path, "generic"))))
    first = _first_records(iterators)
    if not first or any(record.timestamp is None for _, record, _ in first):
        yield from _replay_records(first)
        return
    yield from _timestamp_ordered_records(first)


def _first_records(iterators):
    first: list[tuple[int, LogRecord, Iterable[LogRecord]]] = []
    for index, iterator in enumerate(iterators):
        try:
            record = next(iterator)
        except StopIteration:
            continue
        first.append((index, record, iterator))
    return first


def _replay_records(first):
    """Replay peeked records without allocating the remainder of each log."""
    order = 0
    for _index, record, iterator in first:
        record.order = order
        order += 1
        yield record
        for later in iterator:
            later.order = order
            order += 1
            yield later


def _timestamp_ordered_records(first):
    heap: list[tuple[datetime, int, int, LogRecord, Iterable[LogRecord]]] = []
    serial = 0
    for index, record, iterator in first:
        heapq.heappush(
            heap,
            (
                _parse_timestamp(record.timestamp) or datetime.max,
                index,
                serial,
                record,
                iterator,
            ),
        )
        serial += 1
    order = 0
    while heap:
        _, index, _, record, iterator = heapq.heappop(heap)
        record.order = order
        order += 1
        yield record
        try:
            later = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                _parse_timestamp(later.timestamp) or datetime.max,
                index,
                serial,
                later,
                iterator,
            ),
        )
        serial += 1


def _v2_summary(text: str) -> Optional[dict[str, Any]]:
    root_total = V2_ROOT_TOTAL.search(text)
    root_model = V2_ROOT_MODEL.search(text)
    submodel = V2_SUBMODEL.search(text)
    if root_total or root_model:
        return _v2_root_summary(text, root_total, root_model)
    if submodel:
        return _v2_submodel_summary(submodel)
    return None


def _v2_root_summary(text, root_total, root_model):
    event = V2_EVENT.search(text)
    notify = V2_NOTIFY.search(text)
    reusable = V2_REUSABLE.search(text)
    attached = V2_ATTACHED.search(text)
    return {
        "scope": "root_model" if root_model else "root_graph",
        "name": root_model.group("name").strip() if root_model else None,
        "total_stream_count": (int(root_total.group("value")) if root_total else None),
        "stream_count": None,
        "reusable_stream_count": int(reusable.group(1)) if reusable else None,
        "attached_stream_count": int(attached.group(1)) if attached else None,
        "event_count": int(event.group(1)) if event else None,
        "notify_count": int(notify.group(1)) if notify else None,
    }


def _v2_submodel_summary(submodel):
    return {
        "scope": "static_submodel",
        "name": submodel.group("name").strip(),
        "total_stream_count": None,
        "stream_count": int(submodel.group("stream")),
        "reusable_stream_count": None,
        "attached_stream_count": None,
        "event_count": (
            int(submodel.group("event")) if submodel.group("event") else None
        ),
        "notify_count": (
            int(submodel.group("notify")) if submodel.group("notify") else None
        ),
    }


_V2_RESOURCE_FIELDS = (
    "scope",
    "name",
    "total_stream_count",
    "stream_count",
    "reusable_stream_count",
    "attached_stream_count",
    "event_count",
    "notify_count",
)


def _v2_resource_signature(summary: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(summary.get(field_name) for field_name in _V2_RESOURCE_FIELDS)


def _v2_resource_scope_key(summary: dict[str, Any]) -> tuple[Any, Any]:
    return summary.get("scope"), summary.get("name")


def _is_batch_graph(name: Optional[str]) -> bool:
    return bool(name and re.match(r"^batch_\d+(?:$|_)", name, re.I))


def _is_subgraph(name: Optional[str]) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return bool(
        lowered.startswith(("while", "case"))
        or _is_batch_graph(name)
        or "_sub_" in lowered
        or lowered.endswith("_subgraph")
        or "subgraph" in lowered
    )


def _dynamic_subgraph_root(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    match = DYNAMIC_SUBGRAPH_GRAPH.match(str(name).strip())
    return match.group("root") if match else None


def _has_compile_stream_evidence(state: dict[str, Any]) -> bool:
    return bool(
        state.get("dynamic_stream_count") is not None
        or state.get("logical_stream_count") is not None
        or state.get("final_model_stream_count") is not None
        or state.get("dynamic_logic_stream_ids")
        or state.get("streams")
        or state.get("operator_facts")
        or state.get("compile_stream_allocation_evidence")
    )


def _event(record: LogRecord, kind: str, **payload: Any) -> Event:
    return Event(kind=kind, record=record, payload=payload)


def eventize(record: LogRecord) -> list[Event]:
    """Convert one qualified GE INFO record into semantic facts."""
    text = record.raw
    message = record.message
    events: list[Event] = []
    _eventize_compile_facts(record, text, message, events)
    _eventize_runtime_facts(record, text, message, events)
    return events


def _eventize_compile_facts(record, text, message, events):
    _eventize_compile_entries(record, text, message, events)
    _eventize_compile_exits(record, text, message, events)
    _eventize_compile_physical_facts(record, text, message, events)
    _eventize_final_operator_facts(record, text, events)
    _eventize_node_operator_facts(record, text, message, events)
    _eventize_dynamic_assignment_facts(record, message, events)
    _eventize_static_split_facts(record, message, events)
    _eventize_graph_node_facts(record, message, events)
    _eventize_graph_relation_facts(record, text, message, events)
    _eventize_compile_markers(record, message, events)
    _eventize_batch_facts(record, message, events)
    _eventize_compile_timing(record, text, events)


def _eventize_compile_entries(record, text, message, events):
    graph_match = GRAPH_BUILD.search(text)
    if graph_match and _allowed(
        record,
        ("BuildForKnownShapeGraph", "BuildForUnknownShapeGraph"),
        ("graph_builder.cc",),
    ):
        shape, graph = graph_match.groups()
        events.append(
            _event(record, "GRAPH_BEGIN", shape=shape.lower(), graph=graph.strip())
        )
    if "Start to build BuildForDynamicShape" in message and _allowed(
        record, ("BuildForDynamicShapeGraph",), ("graph_builder.cc",)
    ):
        events.append(_event(record, "COMPILE_FACT", subtype="dynamic_outer_entry"))


def _eventize_compile_exits(record, text, message, events):
    static_exit = STATIC_EXIT.search(text)
    if static_exit and _allowed(
        record,
        ("Assign", "Run", "AssignLogicalStreams"),
        ("logical_stream_allocator.cc", "stream_allocator.cc"),
    ):
        graph, count, main, attached = static_exit.groups()
        events.append(
            _event(
                record,
                "COMPILE_END",
                subtype="static",
                graph=graph.strip(),
                stream_count=int(count),
                main_stream_count=int(main),
                attached_stream_count=int(attached),
            )
        )
    dynamic_exit = DYNAMIC_EXIT.match(message)
    if dynamic_exit and _allowed(
        record,
        ("AssignStreamsForDynamicShapeGraph",),
        ("dynamic_stream_allocator.cc",),
    ):
        events.append(
            _event(
                record,
                "COMPILE_END",
                subtype="dynamic",
                graph=dynamic_exit.group("graph").strip(),
                stream_count=int(dynamic_exit.group("stream")),
                event_count=int(dynamic_exit.group("event")),
            )
        )


def _eventize_compile_physical_facts(record, text, message, events):
    physical_exit = STATIC_PHYSICAL_EXIT.search(text)
    if physical_exit and _allowed(
        record,
        ("SplitStreamAndRefreshTaskDef",),
        ("stream_allocator.cc",),
    ):
        graph, count, notify, event_count = physical_exit.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="physical_exit",
                graph=graph.strip(),
                stream_count=int(count),
                notify_count=int(notify),
                event_count=int(event_count),
            )
        )
    sync_exit = SYNC_EXIT.search(text)
    if sync_exit and _allowed(
        record, ("InsertSyncNodesByLogicStream",), ("stream_allocator.cc",)
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="sync_exit",
                graph=sync_exit.group("graph").strip(),
                stream_count=int(sync_exit.group("stream")),
                notify_count=int(sync_exit.group("notify")),
                event_count=int(sync_exit.group("event")),
            )
        )


def _eventize_final_operator_facts(record, text, events):
    operator = FINAL_OPERATOR.search(text)
    if operator and _allowed(
        record,
        ("SetLogicStreamIdAttr",),
        ("stream_allocator.cc", "logical_stream_allocator.cc"),
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="operator",
                name=operator.group("name").strip(),
                op_type=operator.group("type").strip(),
                stream_id=int(operator.group("stream")),
            )
        )
        attached = ATTACHED_STREAM.search(text)
        attached_ids = (
            [int(value) for value in re.findall(r"-?\d+", attached.group("streams"))]
            if attached
            else []
        )
        attached_ids = [value for value in attached_ids if value >= 0]
        if attached_ids:
            events.append(
                _event(
                    record,
                    "COMPILE_FACT",
                    subtype="attached_operator",
                    name=attached.group("name").strip(),
                    op_type=attached.group("type").strip(),
                    stream_ids=attached_ids,
                )
            )


def _eventize_node_operator_facts(record, text, message, events):
    node_stream = NODE_STREAM.search(text)
    if node_stream and _allowed(
        record,
        ("SplitStreams", "SplitStreamAndRefreshTaskDef"),
        ("stream_allocator.cc", "logical_stream_allocator.cc"),
    ):
        name, op_type, stream_id = node_stream.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="operator",
                name=name.strip(),
                op_type=op_type.strip(),
                stream_id=int(stream_id),
            )
        )
    dynamic_refresh = DYNAMIC_REFRESH.search(message)
    if dynamic_refresh and _allowed(
        record, ("RefreshStreamsForGraphByNodeIds",), ("dynamic_stream_allocator.cc",)
    ):
        graph, stream_id, op_type, name = dynamic_refresh.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="dynamic_operator",
                graph=graph.strip(),
                name=name.strip(),
                op_type=op_type.strip(),
                stream_id=int(stream_id),
            )
        )


def _eventize_dynamic_assignment_facts(record, message, events):
    dynamic_assign = DYNAMIC_ASSIGN.search(message)
    if dynamic_assign and _allowed(
        record, ("AssignStreamForSubgraph",), ("dynamic_stream_allocator.cc",)
    ):
        stream_id, engine, subgraph = dynamic_assign.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="dynamic_assign",
                stream_id=int(stream_id),
                owner=f"subgraph:{subgraph.strip()}",
                engine=engine.strip(),
            )
        )
    dynamic_subgraph = DYNAMIC_SUBGRAPH.search(message)
    if dynamic_subgraph and _allowed(
        record, ("SetSubgraphStreamToNodes",), ("dynamic_stream_allocator.cc",)
    ):
        stream_id, subgraph, engine = dynamic_subgraph.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="dynamic_assign_final",
                stream_id=int(stream_id),
                owner=f"subgraph:{subgraph.strip()}",
                engine=engine.strip(),
            )
        )
    dynamic_reassign = DYNAMIC_REASSIGN.search(message)
    if dynamic_reassign and _allowed(
        record, ("ReassignStreamByStreamLabel",), ("dynamic_stream_allocator.cc",)
    ):
        node, label, stream_id = dynamic_reassign.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="dynamic_reassign",
                stream_id=int(stream_id),
                owner=f"node:{node.strip()}|label:{label.strip()}",
            )
        )


def _eventize_static_split_facts(record, message, events):
    split_node = STATIC_SPLIT_NODE.search(message)
    if split_node and _allowed(
        record,
        ("SplitStreamForOneNode", "SplitNodesToNewStream"),
        ("stream_allocator.cc",),
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="static_split",
                name=split_node.group("name").strip(),
                stream_id=int(split_node.group("stream")),
                attached=int(split_node.group("attached")),
            )
        )
    split_limit = STATIC_SPLIT_LIMIT.search(message)
    if split_limit and _allowed(
        record,
        ("SplitNodesToNewStream", "SplitStreams"),
        ("stream_allocator.cc",),
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="static_split_limit",
                old_stream_id=int(split_limit.group("old")),
                stream_id=int(split_limit.group("new")),
                name=split_limit.group("name").strip(),
                op_type=split_limit.group("type").strip(),
                graph=split_limit.group("graph").strip(),
            )
        )


def _eventize_graph_node_facts(record, message, events):
    node_assign = NODE_ASSIGN.search(message)
    if node_assign and _allowed(
        record, (), ("stream_allocator.cc", "logical_stream_allocator.cc")
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="operator",
                name=node_assign.group(1).strip(),
                op_type="unknown",
                stream_id=int(node_assign.group(2)),
            )
        )
    node_refresh = NODE_REFRESH.search(message) or NODE_UPDATED.search(message)
    if node_refresh and _allowed(
        record, (), ("stream_allocator.cc", "logical_stream_allocator.cc")
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="operator",
                name=node_refresh.group(1).strip(),
                op_type="unknown",
                stream_id=int(node_refresh.group(2)),
            )
        )


def _eventize_graph_relation_facts(record, text, message, events):
    scope = GRAPH_SCOPE.search(message)
    if scope and _allowed(
        record,
        ("Run", "StartAssignAttachedStream"),
        ("assign_attached_stream_pass.cc",),
    ):
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="graph_scope",
                graph=scope.group(1).strip(),
                subgraph_count=int(scope.group(2)),
            )
        )
    relation = GRAPH_RELATION.search(text)
    if relation and _allowed(record, (), ("graph", "builder", "partition")):
        parent, node, child, index = relation.groups()
        events.append(
            _event(
                record,
                "COMPILE_FACT",
                subtype="graph_relation",
                parent_graph=parent.strip(),
                parent_node=node.strip(),
                child_graph=child.strip(),
                subgraph_index=int(index),
            )
        )


def _eventize_compile_markers(record, message, events):
    if "Move split stream from ge to rts" in message and _allowed(
        record, (), ("stream_allocator.cc",)
    ):
        events.append(_event(record, "COMPILE_FACT", subtype="delegated_split"))
    if "Assign attached stream" in message and _allowed(
        record, (), ("stream_allocator.cc", "logical_stream_allocator.cc")
    ):
        events.append(_event(record, "COMPILE_FACT", subtype="attached_allocation"))
    allocation_markers = (
        "[Assign][NewStreamId]",
        "[Assign][StreamId]",
        "[Update][StreamId]",
        "[Reuse][Stream]",
    )
    if _allowed(
        record,
        (),
        (
            "logical_stream_allocator.cc",
            "stream_allocator.cc",
            "assign_attached_stream_pass.cc",
        ),
    ) and any(marker.lower() in message.lower() for marker in allocation_markers):
        events.append(_event(record, "COMPILE_FACT", subtype="allocation"))
    if (
        "ATTR_NAME_SEND_EVENT_IDS" in message or "ATTR_NAME_RECV_EVENT_IDS" in message
    ) and _allowed(
        record,
        (),
        ("stream_allocator.cc", "dynamic_stream_allocator.cc"),
    ):
        events.append(_event(record, "COMPILE_FACT", subtype="sync_attr"))


def _eventize_batch_facts(record, message, events):
    for phase, marker, sources in (
        (
            "dynamic_batch_option_observed",
            "Found dynamic batch, shape",
            ("multi_batch_options.cc",),
        ),
        (
            "dynamic_batch_output_shape_observed",
            "Start to get output dynamic batch shape message",
            ("multi_batch_copy_graph.cc",),
        ),
        (
            "dynamic_batch_graph_observed",
            "Add batch graph[",
            ("multi_batch_clone_pass.cc",),
        ),
    ):
        if marker.lower() in message.lower() and _allowed(record, (), sources):
            events.append(
                _event(
                    record,
                    "COMPILE_FACT",
                    subtype="batch",
                    phase=phase,
                    marker=marker,
                )
            )
    if "ascend_mbatch_shape_" in message.lower() and _allowed(
        record,
        (),
        (
            "multi_batch_copy_graph.cc",
            "multi_batch_clone_pass.cc",
            "stream_allocator.cc",
        ),
    ):
        events.append(
            _event(record, "COMPILE_FACT", subtype="batch", phase="batch_operator")
        )


def _eventize_compile_timing(record, text, events):
    if (
        "GEPERFTRACE" in text
        and "time cost of" in text
        and _allowed(record, (), ("model_builder.cc", "graph_builder.cc"))
    ):
        timing = (
            "logical_stream_assignment_timing_end"
            if "AssignLogicalStreams" in text
            else ("task_build_timing_end" if "BuildModelForGet" in text else None)
        )
        if timing:
            events.append(
                _event(record, "COMPILE_FACT", subtype="timing", phase=timing)
            )


def _eventize_runtime_facts(record, text, message, events):
    _eventize_v1_runtime_init(record, message, events)
    _eventize_v1_runtime_bindings(record, message, events)
    _eventize_v2_executor(record, message, events)
    _eventize_v2_collect(record, message, events)
    _eventize_v2_resources(record, text, message, events)
    _eventize_v2_kernel_events(record, text, events)
    _eventize_runtime_names(record, message, events)
    _eventize_runtime_operators(record, message, events)
    _eventize_runtime_batch(record, message, events)
    _eventize_runtime_context(record, message, events)
    return events


def _eventize_v1_runtime_init(record, message, events):
    runtime_init = RUNTIME_INIT.search(message)
    if runtime_init and _allowed(record, ("InitRuntimeParams",), ("davinci_model.cc",)):
        model_id, stream_count, notify, event_count, label = runtime_init.groups()
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="init",
                model_id=int(model_id),
                stream_count=int(stream_count),
                notify_count=int(notify),
                event_count=int(event_count),
                label_count=int(label),
            )
        )
    runtime_create = RUNTIME_CREATE.search(message)
    if runtime_create and _allowed(
        record, ("CreateNewStream",), ("reusable_stream_allocator.cc",)
    ):
        pointer, rt_id, model_id, priority, flag, task_num = runtime_create.groups()
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="create",
                pointer=pointer,
                rt_stream_id=int(rt_id),
                rt_model_id=int(model_id),
                priority=int(priority),
                flag=int(flag),
                task_num=int(task_num),
            )
        )


def _eventize_v1_runtime_bindings(record, message, events):
    runtime_bind = RUNTIME_BIND.search(message)
    if runtime_bind and _allowed(
        record, ("InitRuntimeResource",), ("davinci_model.cc",)
    ):
        index, rt_id, model_id, flag = runtime_bind.groups()
        events.append(
            _event(
                record,
                "RUNTIME_BIND",
                subtype="v1",
                logical_index=int(index),
                rt_stream_id=int(rt_id),
                model_id=int(model_id),
                flag=int(flag),
            )
        )
    runtime_total = RUNTIME_TOTAL.search(message)
    if runtime_total and _allowed(record, ("GetStreamNum",), ("model_executor.cc",)):
        total, model, follow = map(int, runtime_total.groups())
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="total",
                total_stream_count=total,
                stream_count=model,
                follow_stream_count=follow,
            )
        )


def _eventize_v2_executor(record, message, events):
    executor_anchor = V2_EXECUTOR_ANCHOR.search(message)
    if executor_anchor and _allowed(
        record,
        ("Build", "ModelV2ExecutorBuilder::Build"),
        ("model_v2_executor_builder.cc",),
    ):
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="v2_executor_anchor",
                graph=executor_anchor.group("graph").strip(),
                model=executor_anchor.group("model").strip(),
            )
        )


def _eventize_v2_collect(record, message, events):
    collected_stream = V2_COLLECT_STREAM.search(message)
    if collected_stream:
        logical_index = int(collected_stream.group("logical_index"))
        function_known = record.function is not None
        if function_known:
            if _function_matches(record, ("OccupyStreamResource",)):
                role = "main"
            elif _function_matches(record, ("AcquireStreams",)):
                role = "auxiliary"
            else:
                role = None
        else:
            # Synthetic/minimal fixtures may omit the function envelope.  The
            # Real GE INFO log paths are still gated by the source file when present.
            role = "main" if logical_index == 0 else "auxiliary"
        source = (record.source or "").lower()
        source_allowed = (
            not source
            or source.endswith("model_v2_executor.cc")
            or source.endswith("stream_allocator.cc")
        )
        if role is not None and source_allowed and _allowed(record):
            events.append(
                _event(
                    record,
                    "RUNTIME_BIND",
                    subtype="v2_collect",
                    role=role,
                    pointer=collected_stream.group("pointer"),
                    logical_index=logical_index,
                    rt_stream_id=int(collected_stream.group("rt_stream_id")),
                )
            )


def _eventize_v2_resources(record, text, message, events):
    summary = _v2_summary(message)
    if summary and _allowed(
        record,
        ("GetReusableStreamResourceNum", "GetNonRootModelResourceNum"),
        ("model_converter.cc",),
    ):
        events.append(
            _event(record, "RUNTIME_FACT", subtype="v2_resource", summary=summary)
        )
    kernel_stream = KERNEL_STREAM.search(text)
    if kernel_stream and _kernel_allowed(record, ("stream.cc",)):
        events.append(
            _event(
                record,
                "RUNTIME_BIND",
                subtype="v2",
                logical_index=int(kernel_stream.group(1)),
                rt_stream_id=int(kernel_stream.group(2)),
            )
        )
    kernel_notify = KERNEL_NOTIFY.search(text)
    if kernel_notify and _kernel_allowed(record, ("notify.cc",)):
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="v2_notify",
                logical_index=int(kernel_notify.group(1)),
            )
        )


def _eventize_v2_kernel_events(record, text, events):
    kernel_send = KERNEL_SEND.search(text)
    if kernel_send and _kernel_allowed(record, ("event.cc",)):
        event_id, rt_event, stream_id = kernel_send.groups()
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="v2_send",
                event_id=int(event_id),
                rt_event=rt_event,
                stream_id=int(stream_id),
            )
        )
    kernel_wait = KERNEL_WAIT.search(text)
    if kernel_wait and _kernel_allowed(record, ("event.cc",)):
        event_id, rt_event, stream_id = kernel_wait.groups()
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="v2_wait",
                event_id=int(event_id),
                rt_event=rt_event,
                stream_id=int(stream_id),
            )
        )


def _eventize_runtime_names(record, message, events):
    model_name = MODEL_NAME.search(message)
    if model_name and not _allowed(
        record, (), ("davinci_model.cc", "model_executor.cc")
    ):
        model_name = None
    if model_name:
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="model_name",
                name=model_name.group(1).strip(),
            )
        )
    graph_name = GRAPH_NAME.search(message)
    if graph_name and (
        not ("Known node" in message and "model_id" in message)
        or not _allowed(record, (), ("davinci_model.cc",))
    ):
        graph_name = None
    if graph_name:
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="graph_name",
                name=graph_name.group(1).strip(),
            )
        )


def _eventize_runtime_operators(record, message, events):
    task = RUNTIME_TASK.search(message)
    if task and not _allowed(
        record,
        ("KernelTaskInfo",),
        ("davinci_model.cc", "model_executor.cc", "task_info.cc"),
    ):
        task = None
    if task:
        node, logic_id, pointer = task.groups()
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="operator",
                node=(node or "unknown").strip(),
                logical_index=int(logic_id),
                pointer=pointer,
            )
        )
    if task is None:
        node_logic = RUNTIME_NODE_LOGIC.search(message)
        if node_logic and not _allowed(
            record,
            (),
            ("davinci_model.cc", "model_executor.cc", "task_info.cc"),
        ):
            node_logic = None
        if node_logic:
            events.append(
                _event(
                    record,
                    "RUNTIME_FACT",
                    subtype="operator",
                    node=node_logic.group(1).strip(),
                    logical_index=int(node_logic.group(2)),
                    pointer=None,
                )
            )


def _eventize_runtime_batch(record, message, events):
    batch_size = BATCH_SIZE.search(message)
    if batch_size:
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="batch_size",
                value=int(batch_size.group(1)),
            )
        )
    batch_label = BATCH_LABEL.search(message)
    if batch_label:
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="batch_label",
                value=batch_label.group(1),
            )
        )
    active = ACTIVE_STREAM.search(message)
    if active:
        events.append(
            _event(
                record,
                "RUNTIME_FACT",
                subtype="active_stream",
                branch=active.group(1),
                logical_index=int(active.group(2)),
            )
        )


def _eventize_runtime_context(record, message, events):
    for phase, marker in (
        ("hybrid_model_init_started", "Start to init hybrid model"),
        ("hybrid_model_init_rt2", "Succeed init hybrid model"),
        ("hybrid_execute_rt1", "HybridModel will execute in rt1.0"),
        ("hybrid_execute_pipeline", "HybridModel will execute in pipeline mode"),
        ("hybrid_execute_rt2", "HybridModel will execute in rt2.0 mode"),
        (
            "dynamic_model_executor_entry",
            "SyncExecuteModel via dynamic shape model executor",
        ),
    ):
        if marker.lower() in message.lower():
            events.append(
                _event(record, "RUNTIME_FACT", subtype="context", phase=phase)
            )


def _new_compile(spec: CompileSpec) -> dict[str, Any]:
    session_id = f"compile:{spec.pid or 'unknown'}:{spec.graph}:{spec.sequence}"
    return {
        **_new_compile_identity(spec, session_id),
        **_new_compile_streams(),
        **_new_compile_graph(spec, session_id),
        **_new_compile_details(spec.graph),
    }


def _new_compile_identity(spec: CompileSpec, session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "graph": spec.graph,
        "pid": spec.pid,
        "ge_context": spec.context,
        "window_state": "open",
        "graph_started_at": spec.timestamp,
        "root_graph_summary_at": None,
        "root_graph_completed_at": None,
        "root_graph_completion_evidence": None,
    }


def _new_compile_streams() -> dict[str, Any]:
    return {
        "dynamic_stream_count": None,
        "dynamic_event_count": None,
        "dynamic_stream_at": None,
        "dynamic_logic_stream_ids": [],
        "dynamic_initial_logic_stream_ids": [],
        "dynamic_current_assignments": {},
        "dynamic_owner_labels": {},
        "dynamic_assignment_history": [],
        "dynamic_stream_assignment_evidence": [],
        "compile_stream_allocation_evidence": [],
        "allocation_reasons": Counter(),
        "logical_stream_count": None,
        "main_stream_count": None,
        "attached_stream_count": None,
        "final_model_stream_count": None,
        "event_count": None,
        "notify_count": None,
        "streams": {},
        "operator_facts": {},
        "phases": {},
        "evidence": [],
        "physical_split": "unknown",
    }


def _new_compile_graph(spec: CompileSpec, session_id: str) -> dict[str, Any]:
    dynamic = spec.shape == "unknown"
    scope = "subgraph" if _is_subgraph(spec.graph) else "outer_graph"
    return {
        "graph_role": "unknown_shape_graph" if dynamic else "known_shape_graph",
        "graph_class": "dynamic_shape" if dynamic else "static_shape",
        "hybrid_static_subgraph": False,
        "compile_path": "dynamic_shape" if dynamic else "known_shape",
        "runtime_path": "unknown",
        "stream_scope": "submodel" if scope == "subgraph" else "graph",
        "graph_scope": scope,
        "session_role": "compile_subgraph" if scope == "subgraph" else "compile_graph",
        "outer_graph": spec.graph,
        "compile_family_id": session_id,
        "parent_graph": None,
        "parent_node": None,
        "subgraph_index": None,
        "subgraph_count": None,
        "relation_status": "unknown",
        "relation_evidence": [],
    }


def _new_compile_details(graph: str) -> dict[str, Any]:
    return {
        "sync_stream_count": None,
        "sync_notify_count": None,
        "sync_event_count": None,
        "sync_evidence": {"send_recv": 0, "event_attrs": 0},
        "attached_operator_mappings": [],
        "static_split_details": [],
        "dynamic_batch_markers": 0,
        "unassociated_dynamic_batch": [],
        "operator_mapping": "unknown",
        "operator_derived_stream_count": None,
        "stream_count_consistency": "unknown",
        "report_visibility": "visible",
        "compile_suppressed_reason": None,
        "dynamic_subgraph_root": _dynamic_subgraph_root(graph),
    }


def _new_runtime(
    pid: Optional[str], context: Optional[str], sequence: int
) -> dict[str, Any]:
    return {
        **_new_runtime_identity(pid, context, sequence),
        **_new_runtime_resources(),
        **_new_runtime_details(sequence),
    }


def _new_runtime_identity(
    pid: Optional[str], context: Optional[str], sequence: int
) -> dict[str, Any]:
    return {
        "session_id": f"runtime:{pid or 'unknown'}:{sequence}",
        "model_id": None,
        "pid": pid,
        "ge_context": context,
        "model_name": None,
        "init_runtime_at": None,
        "init_runtime_evidence": None,
        "init_runtime_evidence_all": [],
        "duplicate_init_count": 0,
        "duplicate_init_evidence": [],
        "runtime_started_at": None,
        "runtime_last_at": None,
        "runtime_anchor": None,
        "executor_graph": None,
        "executor_model": None,
        "executor_anchor_evidence": None,
        "runtime_graph_identity": None,
        "runtime_graph_class": None,
        "dynamic_subgraph_fallback": False,
        "runtime_backend": "unknown",
    }


def _new_runtime_resources() -> dict[str, Any]:
    return {
        "model_stream_count": None,
        "model_total_stream_count": None,
        "follow_stream_count": None,
        "hccl_stream_count": None,
        "event_count": None,
        "notify_count": None,
        "label_count": None,
        "requested_stream_count": None,
        "reusable_stream_count": None,
        "attached_stream_count": None,
        "v2_resource_summaries": [],
        "v2_resource_duplicate_evidence": [],
        "v2_resource_conflicts": [],
        "bindings": {},
        "runtime_logical_to_rt_bindings": [],
        "allocation_bindings": [],
        "execution_bindings": [],
        "v2_binding_conflicts": [],
        "kernel_trace_bindings": [],
        "kernel_trace_events": [],
        "kernel_trace_notifies": [],
        "created_streams": [],
        "auxiliary_streams": [],
        "operator_bindings": [],
    }


def _new_runtime_details(sequence: int) -> dict[str, Any]:
    return {
        "active_streams": {},
        "batch_size": None,
        "active_batch_label": None,
        "batch_observation": "not_observed",
        "hybrid_context_markers": [],
        "phases": {},
        "evidence": [],
        "files": [],
        "created_stream_count": 0,
        "bound_stream_count": 0,
        "unbound_stream_count": 0,
        "stream_count_consistency": "unknown",
        "observed_trace_stream_count": None,
        "runtime_order": sequence,
    }


def _add_phase(
    state: dict[str, Any], phase: Optional[str], item: dict[str, Any]
) -> None:
    if phase is None:
        return
    items = state.setdefault("phases", {}).setdefault(phase, [])
    key = (item.get("file"), item.get("line"), item.get("keyword"))
    if not any(
        (old.get("file"), old.get("line"), old.get("keyword")) == key for old in items
    ):
        items.append(item)


def _add_evidence(
    state: dict[str, Any], phase: Optional[str], item: dict[str, Any]
) -> None:
    _add_phase(state, phase, item)
    items = state.setdefault("evidence", [])
    key = (item.get("file"), item.get("line"), item.get("keyword"))
    if not any(
        (old.get("file"), old.get("line"), old.get("keyword")) == key for old in items
    ):
        items.append(item)


def _phase_evidence(record: LogRecord, keyword: str) -> dict[str, Any]:
    return evidence(record, keyword)


def _contains_v2_binding(values, index, rt_id):
    for item in values:
        same_index = int(item.get("logical_stream_index", -1)) == index
        same_rt_id = int(item.get("rt_stream_id", -1)) == rt_id
        if same_index and same_rt_id:
            return True
    return False


def _conflicting_v2_bindings(values, index, rt_id):
    conflicts = []
    for item in values:
        same_index = int(item.get("logical_stream_index", -1)) == index
        different_rt_id = int(item.get("rt_stream_id", -1)) != rt_id
        if same_index and different_rt_id:
            conflicts.append(item)
    return conflicts


def _effective_v2_bindings(runtime):
    effective = {}
    for item in runtime.get("runtime_logical_to_rt_bindings", []):
        index = item.get("logical_stream_index")
        if index is not None:
            effective[int(index)] = item
    return effective


def _same_scope_v2_summaries(summaries, summary):
    scope_key = _v2_resource_scope_key(summary)
    return [item for item in summaries if _v2_resource_scope_key(item) == scope_key]
