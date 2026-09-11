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
"""CLI adapter for the unified GE stream workflow analyzer.

The modules behind the ``workflow_core`` compatibility facade own log parsing
and graph/runtime correlation; this module keeps command-line compatibility
and renders the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence

# The CLI is executed as a script, so its sibling module is on ``sys.path``.
from workflow_core import analyze_workflows


LOGGER = logging.getLogger(__name__)
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)


REQUIRED_REPO_PATHS = (
    "compiler/graph/build/stream",
    "runtime",
    "docs/zh/design/features/stream_allocator.md",
    "docs/zh/design/constraints/stream_allocator.md",
)


@dataclass(frozen=True)
class StreamRecordContext:
    """Inputs used to render one graph/runtime stream record."""

    session: Dict[str, Any]
    compile_session: Optional[Dict[str, Any]]
    runtime_session: Optional[Dict[str, Any]]
    compile_info: Dict[str, Any]
    runtime_backend: str
    runtime_path: str
    runtime_count: Optional[int]
    request_info: Dict[str, Any]
    runtime_rows: list[Dict[str, Any]]
    mapping_rows: list[Dict[str, Any]]
    runtime_bind_evidence: list[Dict[str, Any]]
    create_evidence: list[Dict[str, Any]]
    correlation: str
    mapping_status: str
    missing: list[str]
    runtime_only_dynamic_family: bool


@dataclass(frozen=True)
class ReportContext:
    """Inputs collected before the final report is assembled."""

    repo_root: Optional[Path]
    missing: list[str]
    context_warnings: list[str]
    analysis_mode: str
    logs: Dict[str, list[Path]]
    generic_logs: list[Path]
    bundle: Dict[str, Any]
    compile_data: Optional[Dict[str, Any]]
    runtime_data: Optional[Dict[str, Any]]
    selected_sessions: list[Dict[str, Any]]


class MappingRowValues(NamedTuple):
    """Named cells rendered in one logical-stream mapping row."""

    graph: str
    model_id: str
    compile_stream_id: str
    runtime_stream_index: str
    rt_stream_id: str
    task_num: str
    evidence: str
    status: str


def find_repo_root(explicit: Optional[str]) -> tuple[Optional[Path], list[str]]:
    start = Path(explicit).resolve() if explicit else Path.cwd().resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        missing = [
            item for item in REQUIRED_REPO_PATHS if not (candidate / item).exists()
        ]
        if not missing:
            return candidate, []
    missing = [item for item in REQUIRED_REPO_PATHS if not (start / item).exists()]
    return None, missing


def _compile_count_values(compile_data: Optional[Dict[str, Any]]) -> set[int]:
    """Return observed compile-side stream-count candidates."""
    if not compile_data:
        return set()
    values: set[int] = set()
    for key in (
        "logical_stream_count",
        "final_model_stream_count",
        "dynamic_stream_count",
    ):
        value = compile_data.get(key)
        if value is not None:
            values.add(value)
    return values


def _is_scope_warning(message: str) -> bool:
    """Identify hierarchy warnings that belong in the detailed section."""
    lower = message.lower()
    for token in ("parent", "subgraph", "hierarch", "multiple compile graph scopes"):
        if token in lower:
            return True
    return False


def resolve_log_inputs(values: Optional[Sequence[str]]) -> list[Path]:
    """Resolve files/directories into a deterministic list of readable candidates."""
    paths: set[Path] = set()
    for value in values or []:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            paths.add(candidate)
        elif candidate.is_dir():
            paths.update(item for item in candidate.rglob("*") if item.is_file())
        else:
            raise FileNotFoundError(str(candidate))
    return sorted(paths)


def _select_compile_family(compile_data, runtime_data):
    sessions = (compile_data or {}).get("sessions", []) if compile_data else []
    root_name = (
        (compile_data or {}).get("graph")
        if compile_data and (compile_data or {}).get("graph")
        else (runtime_data or {}).get("runtime_graph_name")
    )
    root_session = None
    for item in reversed(sessions):
        same_root = item.get("graph") == root_name
        outer_scope = item.get("graph_scope", "outer_graph") == "outer_graph"
        if same_root and outer_scope:
            root_session = item
            break
    if root_session is None:
        return sessions, None, list(sessions[-1:])
    root_family = root_session.get("compile_family_id")
    root_graph = root_session.get("graph")
    family = []
    for item in sessions:
        if item is root_session:
            family.append(item)
            continue
        if item.get("graph_scope") != "subgraph":
            continue
        same_family = item.get("compile_family_id") == root_family
        same_outer = item.get("outer_graph") == root_graph
        if same_family or same_outer:
            family.append(item)
    return sessions, root_session, family


def _classify_root_graph(root_session, family_sessions, compile_sessions):
    if root_session is None:
        return "unknown", []
    root_class = root_session.get("graph_class", "unknown")
    root_family = root_session.get("compile_family_id")
    children = []
    for item in compile_sessions:
        if item.get("graph_scope") != "subgraph":
            continue
        if item.get("compile_family_id") == root_family or item.get(
            "outer_graph"
        ) == root_session.get("graph"):
            children.append(item)
    has_dynamic_child = any(
        item.get("graph_class") == "dynamic_shape" for item in children
    )
    has_known_child = any(
        item.get("graph_class") == "static_shape"
        and not is_batch_branch_graph(item.get("graph"))
        for item in children
    )
    evidence = []
    if root_class == "dynamic_shape" and has_known_child:
        return "hybrid_dynamic_static_subgraph", ["unknown_root_with_known_subgraph"]
    if root_class == "static_shape" and has_dynamic_child:
        return "hybrid_dynamic_static_subgraph", ["known_root_with_unknown_subgraph"]
    if root_class == "dynamic_shape":
        return "pure_dynamic_shape", ["unknown_root_without_known_subgraph"]
    if root_class == "static_shape":
        return "pure_static_shape", ["known_root_without_unknown_subgraph"]
    return root_class, evidence


def _runtime_stream_count(runtime_data):
    if not runtime_data:
        return None
    count = runtime_data.get("model_stream_count")
    if count is None:
        allocation = runtime_data.get("allocation_bindings") or []
        indexes = {
            item.get("logical_stream_index")
            for item in allocation
            if item.get("logical_stream_index") is not None
        }
        count = len(indexes) if indexes else None
    if count is None:
        count = runtime_data.get("observed_trace_stream_count")
    if count is None:
        count = runtime_data.get("model_total_stream_count")
    return count


def _has_batch_compile_markers(session: Dict[str, Any]) -> bool:
    if session.get("dynamic_batch_markers"):
        return True
    return any(
        stream.get("batch_ids") for stream in session.get("streams", {}).values()
    )


def classify(
    compile_data: Optional[Dict[str, Any]], runtime_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    from analyze_parts import run_classify

    return run_classify(compile_data, runtime_data)


def build_batch_mapping(
    compile_data: Optional[Dict[str, Any]], runtime_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not compile_data:
        return result
    for stream_id, stream in compile_data.get("streams", {}).items():
        for batch_id in stream.get("batch_ids", {}):
            _append_batch_stream(result, batch_id, stream_id, runtime_data)
    for key, item in result.items():
        if not key.startswith("batch_"):
            continue
        if runtime_data is None:
            item["mapping_status"] = "compile_only"
        elif len(item["rt_stream_ids"]) == len(item["logic_stream_ids"]):
            item["mapping_status"] = "complete"
        else:
            item["mapping_status"] = "partial"
    if runtime_data and runtime_data.get("active_batch_label"):
        active = runtime_data["active_batch_label"]
        result["active_branch"] = {"label": active, **result.get(active.lower(), {})}
    return result


def _append_batch_stream(result, batch_id, stream_id, runtime_data):
    item = result.setdefault(
        f"batch_{batch_id}", {"logic_stream_ids": [], "rt_stream_ids": []}
    )
    logic_id = int(stream_id)
    if logic_id not in item["logic_stream_ids"]:
        item["logic_stream_ids"].append(logic_id)
    binding = (runtime_data or {}).get("bindings", {}).get(stream_id)
    if binding:
        rt_id = binding.get("rt_stream_id")
        if rt_id is not None and rt_id not in item["rt_stream_ids"]:
            item["rt_stream_ids"].append(rt_id)
    item["mapping_status"] = "complete" if item["rt_stream_ids"] else "compile_only"


def build_graph_inventory(
    compile_data: Optional[Dict[str, Any]], sessions: Sequence[Dict[str, Any]]
) -> list[Dict[str, Any]]:
    """Expose graph-level classification without flattening model sessions."""
    if not compile_data:
        return []
    runtime_by_graph = {
        item.get("graph"): item for item in sessions if item.get("graph")
    }
    inventory = []
    for item in compile_data.get("sessions", []):
        graph = item.get("graph")
        model_session = runtime_by_graph.get(graph, {})
        inventory.append(
            {
                "graph": graph,
                "scenario_class": model_session.get(
                    "scenario_class", session_scenario_class(item)
                ),
                "graph_role": item.get("graph_role", "unknown"),
                "graph_class": item.get("graph_class", "unknown"),
                "compile_path": item.get("compile_path", "unknown"),
                "runtime_path": model_session.get(
                    "runtime_path", item.get("runtime_path", "unknown")
                ),
                "stream_scope": item.get("stream_scope", "graph"),
                "graph_scope": item.get("graph_scope", "unknown"),
                "outer_graph": item.get("outer_graph"),
                "compile_family_id": item.get("compile_family_id"),
                "session_role": item.get("session_role", "compile_graph"),
                "relation_status": item.get("relation_status", "unknown"),
            }
        )
    return inventory


def is_batch_branch_graph(graph: Optional[str]) -> bool:
    if not graph:
        return False
    return re.match(r"^batch_\d+(?:$|_)", graph.lower()) is not None


def session_scenario_class(
    compile_session: Optional[Dict[str, Any]],
    runtime_session: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the broad category for one model/compile session.

    This is intentionally separate from ``graph_class``.  A dynamic-batch
    model can have a known-shape root/branch and still be a dynamic scenario;
    conversely, a hybrid root keeps its dynamic/static boundary visible.
    """
    compile_session = compile_session or {}
    runtime_session = runtime_session or {}
    has_batch = bool(
        compile_session.get("dynamic_batch_markers")
        or runtime_session.get("batch_size") is not None
        or runtime_session.get("active_batch_label")
    )
    if compile_session.get("hybrid_static_subgraph"):
        return "hybrid"
    if has_batch or compile_session.get("graph_class") == "dynamic_shape":
        return "dynamic"
    if compile_session.get("graph_class") == "static_shape":
        return "static"
    return "unknown"


def _phase_items(
    session: Optional[Dict[str, Any]], phase_names: Sequence[str]
) -> list[Dict[str, Any]]:
    """Return de-duplicated evidence for the requested phases."""
    if not session:
        return []
    result: list[Dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    phases = session.get("phases", {})
    for phase_name in phase_names:
        for item in phases.get(phase_name, []):
            key = (item.get("file"), item.get("line"), item.get("keyword"))
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result


def _first_phase_item(
    session: Optional[Dict[str, Any]], phase_names: Sequence[str]
) -> Optional[Dict[str, Any]]:
    items = _phase_items(session, phase_names)
    return items[0] if items else None


def _evidence_text(item: Optional[Dict[str, Any]], verbose: bool = False) -> str:
    """Render one compact file/line evidence reference."""
    if not item:
        return "unknown"
    result = f"`{item.get('file', 'unknown')}:{item.get('line', '?')}`"
    keyword = item.get("keyword")
    if keyword:
        result += f" {keyword}"
    if verbose:
        excerpt = str(item.get("excerpt", "")).replace("|", "\\|").replace("\n", " ")
        if excerpt and excerpt != keyword:
            result += f"<br><small>{excerpt[:220]}</small>"
    return result


def _evidence_list_text(
    items: Sequence[Dict[str, Any]], verbose: bool = False, limit: int = 2
) -> str:
    if not items:
        return "unknown"
    shown_items = list(items[:limit])
    result = [_evidence_text(item, verbose) for item in shown_items]
    if len(items) > limit:
        result.append(f"(+{len(items) - limit})")
    return "<br>".join(result)


def _empty_compile_stream_record() -> Dict[str, Any]:
    return {
        "path": "unknown",
        "entry": None,
        "allocation": [],
        "exits": [],
        "stream_count": None,
        "logical_stream_count": None,
        "final_physical_stream_count": None,
        "stream_scope": "unknown",
        "stream_policy": "unknown",
        "stream_policy_scope": "unknown",
        "logical_stream_ids": [],
        "operator_streams": [],
        "operator_mapping": "unknown",
        "operator_derived_stream_count": None,
        "stream_count_consistency": "unknown",
        "physical_split": "unknown",
    }


def _compile_exit_data(compile_session, is_dynamic):
    entry = _first_phase_item(compile_session, ("graph_build_started",))
    exits: list[Dict[str, Any]] = []
    if is_dynamic:
        dynamic_exit = _first_phase_item(compile_session, ("dynamic_stream_assigned",))
        if dynamic_exit:
            exits.append(
                {
                    "kind": "dynamic",
                    "evidence": dynamic_exit,
                    "stream_count": compile_session.get("dynamic_stream_count"),
                    "event_count": compile_session.get("dynamic_event_count"),
                    "notify_count": None,
                }
            )
        stream_count = compile_session.get("dynamic_stream_count")
        stream_scope = "dynamic"
    else:
        logical_exit = _first_phase_item(compile_session, ("logical_stream_assigned",))
        if logical_exit:
            exits.append(
                {
                    "kind": "logical",
                    "evidence": logical_exit,
                    "stream_count": compile_session.get("logical_stream_count"),
                    "event_count": None,
                    "notify_count": None,
                }
            )
        physical_exit = _first_phase_item(
            compile_session, ("physical_stream_finalized",)
        )
        if physical_exit:
            exits.append(
                {
                    "kind": "physical",
                    "evidence": physical_exit,
                    "stream_count": compile_session.get("final_model_stream_count"),
                    "event_count": compile_session.get("event_count"),
                    "notify_count": compile_session.get("notify_count"),
                }
            )
        stream_count = compile_session.get("logical_stream_count")
        if stream_count is None:
            stream_count = compile_session.get("final_model_stream_count")
        stream_scope = "graph"

    return entry, exits, stream_count, stream_scope


def _compile_logical_ids(compile_session, is_dynamic):
    """Collect final logical IDs from allocator and operator evidence."""
    # DynamicStreamAllocator may expose the logical IDs it assigned to
    # subgraphs.  Prefer the current owner->ID map, which reflects final
    # reassignments; fall back to the historical final/initial lists only for
    # old logs that cannot identify an owner.
    dynamic_ids = []
    allocation = list(compile_session.get("compile_stream_allocation_evidence") or [])
    if is_dynamic:
        current_assignments = compile_session.get("dynamic_current_assignments") or {}
        dynamic_ids = sorted(
            {
                int(value)
                for value in current_assignments.values()
                if isinstance(value, int) and value >= 0
            }
        )
        if not dynamic_ids:
            dynamic_ids = list(compile_session.get("dynamic_logic_stream_ids") or [])
        if not dynamic_ids:
            dynamic_ids = list(
                compile_session.get("dynamic_initial_logic_stream_ids") or []
            )
        dynamic_allocation = list(
            compile_session.get("dynamic_stream_assignment_evidence") or []
        )
        if dynamic_allocation:
            allocation = dynamic_allocation

    logical_stream_ids: list[int] = []
    for stream_id in compile_session.get("streams") or {}:
        try:
            stream_value = int(stream_id)
        except (TypeError, ValueError):
            continue
        if stream_value not in logical_stream_ids:
            logical_stream_ids.append(stream_value)
    for stream_id in dynamic_ids:
        try:
            stream_value = int(stream_id)
        except (TypeError, ValueError):
            continue
        if stream_value not in logical_stream_ids:
            logical_stream_ids.append(stream_value)
    logical_stream_ids.sort()
    return logical_stream_ids, allocation


def _compile_operator_streams(compile_session, logical_stream_ids):

    # ``streams`` is the normalized compile-side operator mapping.  Keep it
    # alongside the stream evidence so the report can show one compact row per
    # logical stream without duplicating operators in every runtime mapping
    # row.  Dynamic allocator IDs may have no operator INFO record; retain an
    # explicit unknown row for those IDs instead of treating them as zero
    # operators.
    operator_streams: list[Dict[str, Any]] = []
    raw_streams = compile_session.get("streams") or {}
    for stream_id_text, stream in sorted(
        raw_streams.items(), key=lambda item: int(item[0])
    ):
        try:
            stream_id = int(stream_id_text)
        except (TypeError, ValueError):
            continue
        operators = list(stream.get("operators") or [])
        operator_streams.append(
            {
                "logic_stream_id": stream_id,
                "operator_count": stream.get("operator_count", len(operators)),
                "operators": operators,
                "batch_ids": dict(stream.get("batch_ids") or {}),
            }
        )
    known_operator_ids = {item["logic_stream_id"] for item in operator_streams}
    for stream_id in logical_stream_ids:
        if stream_id in known_operator_ids:
            continue
        operator_streams.append(
            {
                "logic_stream_id": stream_id,
                "operator_count": None,
                "operators": [],
                "batch_ids": {},
            }
        )
    operator_streams.sort(key=lambda item: item["logic_stream_id"])
    operator_ids = []
    for item in operator_streams:
        if item.get("operator_count") not in (None, 0):
            operator_ids.append(item["logic_stream_id"])
    return operator_streams, operator_ids


def _compile_stream_record(compile_session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize only the compile evidence needed by the report."""
    if not compile_session:
        return _empty_compile_stream_record()
    is_dynamic = compile_session.get("compile_path") == "dynamic_shape" or (
        compile_session.get("graph_class") == "dynamic_shape"
    )
    entry, exits, stream_count, stream_scope = _compile_exit_data(
        compile_session, is_dynamic
    )
    logical_stream_ids, allocation = _compile_logical_ids(compile_session, is_dynamic)
    operator_streams, operator_ids = _compile_operator_streams(
        compile_session, logical_stream_ids
    )
    operator_derived_count = max(operator_ids) + 1 if operator_ids else None
    logical_count = (
        compile_session.get("logical_stream_count")
        if not is_dynamic
        else compile_session.get("dynamic_stream_count")
    )
    policy_count = logical_count
    policy_scope = "logical"
    if policy_count is None:
        policy_scope = "unknown"
    stream_policy = _stream_policy_for_count(policy_count)
    operator_mapping = "observed" if operator_ids else "unknown"
    if logical_count is None or operator_derived_count is None:
        operator_consistency = "unknown"
    else:
        operator_consistency = (
            "passed" if operator_derived_count <= logical_count else "mismatch"
        )
    return {
        "path": compile_session.get("compile_path", "unknown"),
        "entry": entry,
        "allocation": allocation,
        "exits": exits,
        "stream_count": stream_count,
        "logical_stream_count": logical_count,
        "final_physical_stream_count": compile_session.get("final_model_stream_count"),
        "stream_scope": stream_scope,
        "stream_policy": stream_policy,
        "stream_policy_scope": policy_scope,
        "logical_stream_ids": logical_stream_ids,
        "operator_streams": operator_streams,
        "operator_mapping": operator_mapping,
        "operator_derived_stream_count": operator_derived_count,
        "stream_count_consistency": operator_consistency,
        "physical_split": compile_session.get("physical_split", "unknown"),
    }


def _phase_item_for_line(
    session: Optional[Dict[str, Any]], phase_names: Sequence[str], line: Any
) -> Optional[Dict[str, Any]]:
    if line is None:
        return None
    for item in _phase_items(session, phase_names):
        if item.get("line") == line:
            return item
    return None


def _v1_binding_rows(runtime_session, seen):
    rows = []
    for index, binding in sorted(
        (runtime_session.get("bindings") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        logic_id = int(index)
        rt_id = binding.get("rt_stream_id")
        evidence_item = _phase_item_for_line(
            runtime_session, ("logical_stream_bound",), binding.get("binding_line")
        )
        key = (logic_id, rt_id, evidence_item.get("line") if evidence_item else None)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "compile_logic_stream_id": None,
                "runtime_logic_stream_index": logic_id,
                "rt_stream_id": rt_id,
                "task_num": binding.get("task_num"),
                "evidence": evidence_item,
                "evidence_type": "Logical stream index",
            }
        )
    return rows


def _v2_binding_rows(runtime_session, seen):
    rows = []
    for binding in runtime_session.get("runtime_logical_to_rt_bindings", []):
        logic_id = binding.get("logical_stream_index")
        rt_id = binding.get("rt_stream_id")
        key = (logic_id, rt_id, binding.get("line"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "compile_logic_stream_id": None,
                "runtime_logic_stream_index": logic_id,
                "rt_stream_id": rt_id,
                "task_num": None,
                "evidence": {
                    "file": binding.get("file"),
                    "line": binding.get("line"),
                    "keyword": binding.get("keyword", "Get rts stream"),
                    "excerpt": binding.get("excerpt", ""),
                },
                "evidence_type": binding.get("evidence_type", "direct"),
            }
        )
    return rows


def _runtime_binding_rows(
    runtime_session: Optional[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Normalize V1 and V2 logical-to-RT bindings into one row shape."""
    if not runtime_session:
        return []
    seen: set[tuple[Any, Any, Any]] = set()
    rows = _v1_binding_rows(runtime_session, seen)
    rows.extend(_v2_binding_rows(runtime_session, seen))
    rows.sort(
        key=lambda item: (
            item.get("runtime_logic_stream_index")
            if item.get("runtime_logic_stream_index") is not None
            else 2**31,
            item.get("rt_stream_id") if item.get("rt_stream_id") is not None else 2**31,
        )
    )
    return rows


def _v2_request_info(runtime_session):
    values = []
    for label, field in (
        ("requested", "requested_stream_count"),
        ("reusable", "reusable_stream_count"),
        ("attached", "attached_stream_count"),
    ):
        value = runtime_session.get(field)
        if value is not None:
            values.append(f"{label}={value}")
    allocation = runtime_session.get("allocation_bindings") or []
    if not values and allocation:
        values.append(f"allocated={len(allocation)}")
    if not values and runtime_session.get("model_total_stream_count") is not None:
        values.append(f"total={runtime_session.get('model_total_stream_count')}")
    evidence_items = _phase_items(
        runtime_session,
        (
            "runtime_resource_summary",
            "stream_resource_requested",
            "runtime_stream_bound",
        ),
    )
    return {
        "text": "; ".join(values) if values else "unknown",
        "evidence": evidence_items,
    }


def _runtime_request_info(
    runtime_session: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not runtime_session:
        return {"text": "not_observed", "evidence": []}
    backend = runtime_session.get("runtime_backend", "unknown")
    v2 = backend in ("v2_rt2", "mixed") or bool(
        runtime_session.get("v2_resource_summaries")
        or runtime_session.get("requested_stream_count") is not None
    )
    if v2:
        return _v2_request_info(runtime_session)
    stream_count = runtime_session.get("model_stream_count")
    request_text = (
        f"stream_num={stream_count}" if stream_count is not None else "unknown"
    )
    total_count = runtime_session.get("model_total_stream_count")
    if total_count is not None and total_count != stream_count:
        request_text += f"; total={total_count}"
    request_evidence = _phase_items(runtime_session, ("runtime_params_initialized",))
    if total_count is not None and total_count != stream_count:
        request_evidence += _phase_items(runtime_session, ("runtime_stream_summary",))
    return {
        "text": request_text,
        "evidence": request_evidence,
    }


def _stream_count_text(compile_info: Dict[str, Any]) -> str:
    count = compile_info.get("stream_count")
    if count is None:
        return "unknown"
    ids = compile_info.get("logical_stream_ids", [])
    if ids:
        return f"{count} [{','.join(str(item) for item in ids)}]"
    scope = compile_info.get("stream_scope", "graph")
    return f"{count} ({scope}; IDs unknown)"


def _stream_count_summary_text(compile_info: Dict[str, Any]) -> str:
    """Return a count/scope summary; individual IDs live in the mapping table."""
    count = compile_info.get("stream_count")
    if count is None:
        return "unknown"
    scope = compile_info.get("stream_scope", "graph")
    return f"{count} ({scope})"


def _logical_stream_count_text(compile_info: Dict[str, Any]) -> str:
    count = compile_info.get("logical_stream_count")
    if count is not None:
        return str(count)
    return "unknown"


def _stream_policy_text(compile_info: Dict[str, Any]) -> str:
    policy = compile_info.get("stream_policy", "unknown")
    scope = compile_info.get("stream_policy_scope", "unknown")
    if policy == "unknown" or scope == "unknown":
        return policy
    return f"{policy} ({scope})"


def _stream_policy_for_count(count: Any) -> str:
    if count is None:
        return "unknown"
    if count > 1:
        return "multi_stream"
    if count == 1:
        return "single_stream"
    return "unknown"


def _operator_display_text(operator: Dict[str, Any], verbose: bool = False) -> str:
    """Render one operator as ``name (type)`` with optional source evidence."""
    name = str(operator.get("name") or operator.get("node") or "unknown")
    op_type = str(operator.get("type") or "unknown")
    value = f"{name} ({op_type})".replace("|", r"\|").replace("\n", " ")
    if verbose and operator.get("file"):
        value += f"<br><small>`{operator['file']}:{operator.get('line', '?')}`</small>"
    return value


def _operator_stream_text(
    stream: Dict[str, Any], operator_limit: Optional[int], verbose: bool = False
) -> str:
    operators = list(stream.get("operators") or [])
    if not operators:
        return "unknown"
    shown = operators if operator_limit is None else operators[:operator_limit]
    rendered = [_operator_display_text(item, verbose) for item in shown]
    total = stream.get("operator_count")
    if total is not None and total > len(shown):
        rendered.append(f"(+{total - len(shown)}，使用 --full-ops 查看全部)")
    return "<br>".join(rendered)


def _compile_exit_text(compile_info: Dict[str, Any], verbose: bool = False) -> str:
    parts = []
    for item in compile_info.get("exits", []):
        details = []
        if item.get("stream_count") is not None:
            details.append(f"stream={item['stream_count']}")
        if item.get("event_count") is not None:
            details.append(f"event={item['event_count']}")
        if item.get("notify_count") is not None:
            details.append(f"notify={item['notify_count']}")
        suffix = f" ({', '.join(details)})" if details else ""
        parts.append(
            f"{item.get('kind', 'stream')} exit: "
            f"{_evidence_text(item.get('evidence'), verbose)}{suffix}"
        )
    return "<br>".join(parts) if parts else "unknown"


def _compile_allocation_text(
    compile_info: Dict[str, Any], verbose: bool = False
) -> str:
    """Render observed compile-side allocation records, when available."""
    if compile_info.get("path") == "unknown":
        return "not_observed"
    allocation = compile_info.get("allocation") or []
    if not allocation:
        return "not_observed"
    shown_items = list(allocation[:2])
    rendered = []
    for item in shown_items:
        text = _evidence_text(item, verbose)
        stage = item.get("stage")
        rendered.append(f"{stage}: {text}" if stage else text)
    if len(allocation) > len(shown_items):
        rendered.append(f"(+{len(allocation) - len(shown_items)})")
    return "<br>".join(rendered)


def _build_mapping_rows(compile_ids, runtime_rows):
    rows = []
    matched = set()
    for compile_id in sorted(compile_ids):
        match = next(
            (
                item
                for item in runtime_rows
                if item.get("runtime_logic_stream_index") == compile_id
            ),
            None,
        )
        if match is None:
            rows.append(
                {
                    "compile_logic_stream_id": compile_id,
                    "runtime_logic_stream_index": None,
                    "rt_stream_id": None,
                    "task_num": None,
                    "evidence": None,
                    "evidence_type": "not_observed",
                }
            )
            continue
        row = dict(match)
        row["compile_logic_stream_id"] = compile_id
        rows.append(row)
        if match.get("runtime_logic_stream_index") is not None:
            matched.add(int(match["runtime_logic_stream_index"]))
    for item in runtime_rows:
        if item.get("runtime_logic_stream_index") not in matched:
            rows.append(dict(item))
    return rows or [dict(item) for item in runtime_rows]


def _mapping_status(compile_session, runtime_session, mapping_rows, compile_ids):
    if not compile_session:
        return "runtime_only" if mapping_rows else "not_observed"
    if not runtime_session:
        return "compile_only"
    if not mapping_rows:
        return "not_observed"
    complete = all(
        item.get("compile_logic_stream_id") is not None
        and item.get("rt_stream_id") is not None
        for item in mapping_rows
    )
    return (
        "complete" if complete and len(mapping_rows) == len(compile_ids) else "partial"
    )


def _missing_evidence(context: StreamRecordContext) -> list[str]:
    missing = []
    if context.compile_session:
        if context.compile_info.get("entry") is None:
            missing.append("compile_entry")
        if not context.compile_info.get("exits"):
            missing.append("compile_stream_exit")
        if context.compile_info.get("stream_count") is None:
            missing.append("compile_stream_count")
        elif not context.compile_info.get("logical_stream_ids"):
            missing.append("compile_logic_stream_ids")
    else:
        missing.append("compile_record")
    if context.runtime_session:
        if not context.request_info.get("evidence"):
            missing.append("runtime_stream_request")
        if not context.runtime_rows:
            missing.append("runtime_logic_to_rt_binding")
    else:
        missing.append("runtime_record")
    if context.compile_session and context.runtime_session:
        if context.session.get("correlation", "unknown") == "unknown":
            missing.append("compile_runtime_correlation")
        if (
            context.mapping_status == "partial"
            and "compile_logic_stream_ids" not in missing
        ):
            missing.append("logic_to_rt_mapping")
    return missing


def _runtime_record_count(runtime_session):
    if not runtime_session:
        return None
    count = runtime_session.get("model_stream_count")
    if count is None:
        count = runtime_session.get("requested_stream_count")
    if count is None:
        indexes = {
            item.get("logical_stream_index")
            for item in runtime_session.get("allocation_bindings") or []
            if item.get("logical_stream_index") is not None
        }
        count = len(indexes) if indexes else None
    return count


def _stream_compile_payload(context: StreamRecordContext) -> Dict[str, Any]:
    compile_info = context.compile_info
    return {
        **compile_info,
        "entry_text": _evidence_text(compile_info.get("entry")),
        "exit_text": _compile_exit_text(compile_info),
        "stream_text": _stream_count_text(compile_info),
    }


def _stream_runtime_payload(context: StreamRecordContext) -> Dict[str, Any]:
    runtime_session = context.runtime_session
    return {
        "backend": context.runtime_backend,
        "path": context.runtime_path,
        "stream_count": context.runtime_count,
        "request_text": context.request_info.get("text"),
        "request_evidence": context.request_info.get("evidence"),
        "create_evidence": context.create_evidence,
        "binding_evidence": context.runtime_bind_evidence,
        "bindings": context.runtime_rows,
        "mapping_rows": context.mapping_rows,
        "observed_binding_count": len(context.runtime_rows),
        "created_count": (
            runtime_session.get("created_stream_count") if runtime_session else None
        ),
        "bound_count": (
            runtime_session.get("bound_stream_count") if runtime_session else None
        ),
    }


def _stream_record_payload(context: StreamRecordContext) -> Dict[str, Any]:
    session = context.session
    compile_session = context.compile_session
    runtime_session = context.runtime_session
    return {
        "session_id": session.get("session_id"),
        "record_kind": session.get("session_type", "unknown"),
        "graph": (
            compile_session.get("graph")
            if compile_session
            else runtime_session.get("runtime_graph_name")
            if runtime_session
            else None
        ),
        "model_id": runtime_session.get("model_id") if runtime_session else None,
        "compile": _stream_compile_payload(context),
        "runtime": _stream_runtime_payload(context),
        "correlation": context.correlation,
        "mapping_status": context.mapping_status,
        "analysis_status": session.get("analysis_status", "partial"),
        "report_visibility": (
            "runtime_binding_only" if context.runtime_only_dynamic_family else "visible"
        ),
        "missing": context.missing,
        "sort_time": (
            (compile_session or {}).get("graph_started_at")
            or (runtime_session or {}).get("init_runtime_at")
            or ""
        ),
    }


def _record_correlation(session: Dict[str, Any]) -> str:
    correlation = session.get("correlation", "unknown")
    if session.get("correlation_inference"):
        correlation += f" / {session['correlation_inference']}"
    return correlation


def _record_runtime_path(session, runtime_session):
    runtime_backend = (
        runtime_session.get("runtime_backend", "unknown")
        if runtime_session
        else "not_observed"
    )
    runtime_path = (
        session.get("runtime_path")
        or (runtime_session or {}).get("runtime_path")
        or runtime_backend
    )
    return runtime_backend, runtime_path


def _build_stream_context(session: Dict[str, Any]) -> StreamRecordContext:
    compile_session = session.get("compile")
    runtime_session = session.get("runtime")
    compile_info = _compile_stream_record(compile_session)
    runtime_only_dynamic_family = bool(
        runtime_session
        and runtime_session.get("runtime_graph_identity")
        in ("dynamic_subgraph_family", "dynamic_root_compile_missing")
    )
    runtime_rows = _runtime_binding_rows(runtime_session)
    request_info = _runtime_request_info(runtime_session)
    runtime_bind_evidence = _phase_items(
        runtime_session, ("logical_stream_bound", "runtime_stream_bound")
    )
    create_evidence = _phase_items(runtime_session, ("stream_created",))
    compile_ids = set(compile_info["logical_stream_ids"])
    mapping_rows = _build_mapping_rows(compile_ids, runtime_rows)
    mapping_status = _mapping_status(
        compile_session, runtime_session, mapping_rows, compile_ids
    )
    runtime_backend, runtime_path = _record_runtime_path(session, runtime_session)
    return StreamRecordContext(
        session=session,
        compile_session=compile_session,
        runtime_session=runtime_session,
        compile_info=compile_info,
        runtime_backend=runtime_backend,
        runtime_path=runtime_path,
        runtime_count=_runtime_record_count(runtime_session),
        request_info=request_info,
        runtime_rows=runtime_rows,
        mapping_rows=mapping_rows,
        runtime_bind_evidence=runtime_bind_evidence,
        create_evidence=create_evidence,
        correlation=_record_correlation(session),
        mapping_status=mapping_status,
        missing=[],
        runtime_only_dynamic_family=runtime_only_dynamic_family,
    )


def _build_stream_record(session: Dict[str, Any]) -> Dict[str, Any]:
    context = _build_stream_context(session)
    missing = _missing_evidence(context)
    context = replace(context, missing=missing)
    return _stream_record_payload(context)


def build_stream_records(sessions: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Build the compact graph-to-runtime stream mapping view."""
    records = [_build_stream_record(session) for session in sessions]
    records.sort(
        key=lambda item: (
            0 if item.get("sort_time") else 1,
            item.get("sort_time", ""),
            item.get("graph") or "",
            item.get("session_id") or "",
        )
    )
    return records


def _record_runtime_request_text(record: Dict[str, Any], verbose: bool = False) -> str:
    runtime = record.get("runtime") or {}
    if runtime.get("backend") == "not_observed":
        return "not_observed"
    request_text = runtime.get("request_text", "unknown")
    request_evidence = _evidence_list_text(runtime.get("request_evidence", []), verbose)
    if request_evidence == "unknown":
        return request_text
    return f"{request_text}<br>{request_evidence}"


def _record_runtime_create_text(record: Dict[str, Any], verbose: bool = False) -> str:
    runtime = record.get("runtime") or {}
    if runtime.get("backend") == "not_observed":
        return "not_observed"
    evidence_items = runtime.get("create_evidence", [])
    if runtime.get("backend") == "v2_rt2":
        allocation = runtime.get("allocation_bindings") or []
        if allocation:
            return f"{len(allocation)} 条<br>{_evidence_list_text(allocation, verbose)}"
    if not evidence_items:
        if runtime.get("backend") == "v2_rt2":
            return "not_observed (V2 无专用创建日志)"
        return "unknown"
    count = runtime.get("created_count")
    count_text = count if count is not None else len(evidence_items)
    return f"{count_text} 条<br>{_evidence_list_text(evidence_items, verbose)}"


def _record_runtime_binding_text(record: Dict[str, Any]) -> str:
    runtime = record.get("runtime") or {}
    if runtime.get("backend") == "not_observed":
        return "not_observed"
    evidence_items = runtime.get("binding_evidence", [])
    observed_count = runtime.get("observed_binding_count")
    if observed_count is None:
        observed_count = len(evidence_items)
    if observed_count == 0 and not evidence_items:
        return "unknown"
    # Mapping rows may contain compile-only placeholders.  Count only direct
    # runtime bindings here; their line-level evidence is shown once below.
    return f"{observed_count} 条"


def _record_missing_text(missing: Sequence[str]) -> str:
    labels = {
        "compile_record": "编译记录",
        "compile_entry": "编译入口",
        "compile_stream_exit": "编译流分配出口",
        "compile_stream_count": "编译流数量",
        "compile_logic_stream_ids": "编译逻辑流 ID",
        "runtime_record": "运行记录",
        "runtime_stream_request": "运行 stream 申请",
        "runtime_logic_to_rt_binding": "逻辑流到 RT 流绑定",
        "compile_runtime_correlation": "编译/运行关联",
        "logic_to_rt_mapping": "逻辑流到物理流映射",
    }
    return ", ".join(labels.get(item, item) for item in missing)


def _record_graph_label(record: Dict[str, Any]) -> str:
    graph = record.get("graph")
    if graph:
        return str(graph)
    model_id = record.get("model_id")
    if model_id is not None:
        return f"model_id={model_id} (runtime-only)"
    return "unknown"


def _count_records(records, kinds, status=None):
    count = 0
    for item in records:
        if item.get("record_kind") not in kinds:
            continue
        if status is not None and item.get("mapping_status") != status:
            continue
        count += 1
    return count


def _render_operator_section(records, operator_limit, verbose):
    lines = [
        "",
        "## 逻辑流对应算子",
        "",
        "每条逻辑流只列一次编译期最终算子映射，格式为 `算子名 (算子类型)`；算子数来自全部已观察的 GE INFO 映射。默认显示 `--sample-limit` 条，使用 `--full-ops` 展开该流的全部算子。",
        "",
        "| 图名 | 逻辑流 ID | 算子数 | 算子（类型） |",
        "|---|---:|---:|---|",
    ]
    row_count = 0
    for record in records:
        compile_info = record.get("compile") or {}
        streams = compile_info.get("operator_streams") or []
        for stream in streams:
            row_count += 1
            count = stream.get("operator_count")
            values = (
                str(_record_graph_label(record)),
                str(stream.get("logic_stream_id", "unknown")),
                str(count) if count is not None else "unknown",
                _operator_stream_text(stream, operator_limit, verbose),
            )
            lines.append("| " + " | ".join(values) + " |")
    if row_count == 0:
        lines.append("| - | - | - | 未观察到编译期算子→逻辑流 INFO 映射 |")
    return lines


def _mapping_row_status(record, row):
    runtime_only = record.get("report_visibility") == "runtime_binding_only"
    compile_id = row.get("compile_logic_stream_id")
    runtime_index = row.get("runtime_logic_stream_index")
    has_runtime_binding = row.get("rt_stream_id") is not None
    if compile_id is not None and has_runtime_binding:
        return "complete"
    if runtime_only and runtime_index is not None and has_runtime_binding:
        return "runtime_only"
    if compile_id is not None and runtime_index is None:
        return "not_observed"
    return "partial"


def _mapping_row_values(record, row, verbose):
    runtime_only = record.get("report_visibility") == "runtime_binding_only"
    compile_id = row.get("compile_logic_stream_id")
    runtime_index = row.get("runtime_logic_stream_index")
    rt_stream_id = row.get("rt_stream_id")
    task_num = row.get("task_num")
    return MappingRowValues(
        str(_record_graph_label(record)),
        str(record.get("model_id")) if record.get("model_id") is not None else "-",
        str(compile_id)
        if compile_id is not None
        else ("-" if runtime_only else "unknown"),
        str(runtime_index) if runtime_index is not None else "unknown",
        str(rt_stream_id) if rt_stream_id is not None else "unknown",
        str(task_num) if task_num is not None else "unknown",
        _evidence_text(row.get("evidence"), verbose),
        _mapping_row_status(record, row),
    )


def _render_mapping_section(records, verbose):
    lines = [
        "",
        "## 逻辑流 → RT 流",
        "",
        "只有同时观察到编译逻辑流 ID 和运行期绑定证据时，才标记为 `complete`。",
        "编译逻辑流 ID 与运行逻辑索引按数值配对；这表示日志可观察到的 index-based 关系，不把两层 ID 当作同一类对象。",
        "",
        "| 图名 | model_id | 编译逻辑流 ID | 运行逻辑索引 | RT stream ID | task | 绑定证据 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    row_count = 0
    for record in records:
        for row in (record.get("runtime") or {}).get("mapping_rows", []):
            row_count += 1
            values = _mapping_row_values(record, row, verbose)
            lines.append("| " + " | ".join(values) + " |")
    if row_count == 0:
        lines.append(
            "| - | - | - | - | - | - | 未观察到运行期逻辑流绑定 | not_observed |"
        )
    return lines


def _render_missing_section(records):
    missing_records = []
    for item in records:
        if (
            item.get("missing")
            and item.get("report_visibility") != "runtime_binding_only"
        ):
            missing_records.append(item)
    lines = ["", "## 未完整证据", ""]
    if not missing_records:
        lines.append("未发现与流映射相关的缺失证据。")
        return lines
    lines += ["| 图名 | 缺失项 | 说明 |", "|---|---|---|"]
    grouped_missing: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for record in missing_records:
        grouped_missing[tuple(record.get("missing", []))].append(
            _record_graph_label(record)
        )
    for missing_tuple, graph_names in grouped_missing.items():
        missing = list(missing_tuple)
        note = "仅保留已观察到的阶段，不对缺失日志做推断。"
        if "compile_logic_stream_ids" in missing:
            note = "编译出口给出了流数量，但没有逐逻辑流 ID，无法宣称一一映射。"
        elif "logic_to_rt_mapping" in missing:
            note = "两侧都有流证据，但编译逻辑流 ID 与运行绑定不能逐项对应。"
        lines.append(
            f"| {'<br>'.join(graph_names)} | {_record_missing_text(missing)} | {note} |"
        )
    return lines


def _render_graph_evidence_rows(records, verbose):
    lines = []
    for record in records:
        if record.get("report_visibility") == "runtime_binding_only":
            continue
        compile_info = record.get("compile") or {}
        runtime_info = record.get("runtime") or {}
        graph = _record_graph_label(record)
        path = compile_info.get("path", "unknown")
        path = {
            "known_shape": "static / known_shape",
            "dynamic_shape": "dynamic / unknown_shape",
        }.get(path, path)
        runtime_session = "not_observed"
        if runtime_info.get("backend") != "not_observed":
            model_id = (
                record.get("model_id")
                if record.get("model_id") is not None
                else "unknown"
            )
            runtime_path = runtime_info.get(
                "path", runtime_info.get("backend", "unknown")
            )
            runtime_session = f"model_id={model_id} / {runtime_path}"
        values = (
            str(graph),
            path,
            _evidence_text(compile_info.get("entry"), verbose),
            _compile_allocation_text(compile_info, verbose),
            _compile_exit_text(compile_info, verbose),
            _logical_stream_count_text(compile_info),
            _stream_policy_text(compile_info),
            runtime_session,
            _record_runtime_request_text(record, verbose),
            _record_runtime_create_text(record, verbose),
            _record_runtime_binding_text(record),
            record.get("correlation", "unknown"),
            record.get("mapping_status", "unknown"),
        )
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report_warnings(report):
    relevant = []
    for item in report.get("warnings", []):
        if not _is_scope_warning(item):
            relevant.append(item)
    if not relevant:
        return []
    return ["", "## 流证据告警", ""] + [f"- {item}" for item in relevant]


def _render_overview_lines(report, mode, records):
    compile_count = _count_records(records, ("model", "compile_graph"))
    runtime_count = _count_records(records, ("model", "runtime_model"))
    paired_count = _count_records(records, ("model",))
    complete_count = _count_records(records, ("model",), status="complete")
    report_status = (
        "complete"
        if records and complete_count == len(records)
        else ("partial" if records else "no_stream_evidence")
    )
    selected_compile = report.get("compile") or {}
    logical_count = selected_compile.get("logical_stream_count")
    if logical_count is None:
        logical_count = selected_compile.get("dynamic_stream_count")
    policy = mode.get("stream_policy", "unknown")
    consistency = selected_compile.get("stream_count_consistency", "unknown")
    return [
        "# GE 流分配与运行流映射",
        "",
        "## 总览",
        "",
        "- 日志范围：GE INFO 日志（仅使用实际日志，不以源码函数名补充事件）",
        f"- 场景/Shape：`{mode.get('scenario_class', 'unknown')}` / `{mode.get('shape_mode', 'unknown')}`",
        f"- 编译图形态：`{mode.get('graph_form', 'unknown')}`",
        f"- 运行后端：`{mode.get('runtime_backend', 'unknown')}`",
        f"- 流策略（单流/多流）：`{policy}`",
        f"- 编译最终逻辑流数：`{logical_count if logical_count is not None else 'unknown'}`",
        f"- 算子映射/流数一致性：`{selected_compile.get('operator_mapping', 'unknown')}` / `{consistency}`",
        f"- 编译图记录/运行记录：`{compile_count}` / `{runtime_count}`",
        f"- 已关联图/完整逻辑流→RT流映射：`{paired_count}` / `{complete_count}`",
        f"- 选定模型分析状态：`{mode.get('analysis_status', 'unknown')}`",
        f"- 图级报告状态：`{report_status}`",
        "",
        "`unknown` 表示对应阶段没有观察到匹配的 GE INFO 日志证据；它不等价于 0 或单流。",
        "",
        "## 图级流证据",
        "",
        "每行对应一个编译图或运行记录，只展示编译入口/出口、流分配和运行绑定证据。",
        "",
        "| 图名 | 编译路径 | 编译窗口入口 | 编译分配入口/过程 | 编译流出口 | 最终逻辑流数 | 流策略 | 运行会话 | 运行申请 | 运行创建数 | 运行绑定数 | 关联 | 状态 |",
        "|---|---|---|---|---|---:|---|---|---|---|---:|---|---|",
    ]


def _render_graph_notes():
    return [
        "",
        (
            "`编译窗口入口`用于确定当前 known/unknown 图的日志范围。静态路径若出现 "
            "`[Assign][NewStreamId]`、`[Update][StreamId]` 或 `[Reuse][Stream]`，"
            "会在“编译分配入口/过程”列保留实际过程证据；静态 logical allocator "
            "没有稳定的独立开始日志时仍写 `not_observed`，不把 `AssignLogicalStreams` "
            "函数名或 timing 行冒充入口。动态图若出现 `Assign stream_id` 或 "
            "`[Assign][StreamId]`，同样保留实际证据，最终逻辑流数量仍以流分配出口为准。"
            "`运行创建数`统计实际 `Create new stream` 行（可能包含辅助流），"
            "`运行绑定数`只统计可映射的绑定行；逐条绑定日志只在下方映射表保留一次。"
        ),
    ]


def render_stream_mapping_markdown(
    report: Dict[str, Any], verbose: bool = False, operator_limit: Optional[int] = 5
) -> str:
    """Render the evidence-first Markdown report."""
    mode = report.get("mode", {})
    records = report.get("stream_records")
    if records is None:
        records = build_stream_records(report.get("sessions", []))

    lines = _render_overview_lines(report, mode, records)
    lines.extend(_render_graph_evidence_rows(records, verbose))

    lines += _render_graph_notes()

    lines.extend(_render_operator_section(records, operator_limit, verbose))
    lines.extend(_render_mapping_section(records, verbose))
    lines.extend(_render_missing_section(records))

    lines.extend(_render_report_warnings(report))
    return "\n".join(lines) + "\n"


def render_markdown(
    report: Dict[str, Any], sample_limit: Optional[int], verbose: bool = False
) -> str:
    return render_stream_mapping_markdown(
        report, verbose=verbose, operator_limit=sample_limit
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        help="generic GE log file; repeat for rotated/mixed logs",
    )
    parser.add_argument(
        "--log-dir", action="append", help="directory containing mixed/rotated GE logs"
    )
    parser.add_argument(
        "--compile",
        action="append",
        help="GE/atc compile log; repeat for rotated files",
    )
    parser.add_argument(
        "--runtime",
        action="append",
        help="GE/ACL runtime log; repeat for rotated files",
    )
    parser.add_argument(
        "--repo-root", help="GE repository root; otherwise discover from cwd"
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--full-ops", action="store_true", help="retain all final operators"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="include evidence excerpts in Markdown"
    )
    parser.add_argument(
        "--session", action="append", help="restrict report to session_id (repeatable)"
    )
    parser.add_argument("--sample-limit", type=int, default=5)
    return parser.parse_args()


def _load_input_logs(args):
    if not any((args.log, args.log_dir, args.compile, args.runtime)):
        LOGGER.error(
            "at least one of --log, --log-dir, --compile or --runtime is required",
        )
        return 2, None
    repo_root, missing = find_repo_root(args.repo_root)
    context_warnings: list[str] = []
    analysis_mode = "repo_verified"
    if repo_root is None:
        analysis_mode = "log_only"
        context_warnings.append(
            "GE repository context was not found; analysis uses GE INFO log fields only"
        )
    try:
        generic_logs = resolve_log_inputs((args.log or []) + (args.log_dir or []))
    except FileNotFoundError as error:
        LOGGER.error("log input is not readable: %s", error)
        return 4, None

    logs: Dict[str, list[Path]] = {}
    for kind, value in (("compile", args.compile), ("runtime", args.runtime)):
        if value:
            try:
                paths = resolve_log_inputs(value)
            except FileNotFoundError as error:
                LOGGER.error("%s log is not readable: %s", kind, error)
                return 4, None
            logs[kind] = paths

    if generic_logs:
        logs["compile"] = sorted(set(logs.get("compile", [])) | set(generic_logs))
        logs["runtime"] = sorted(set(logs.get("runtime", [])) | set(generic_logs))
    return 0, (repo_root, missing, context_warnings, analysis_mode, logs, generic_logs)


def _select_report_sessions(bundle, args):
    compile_data = bundle.get("compile")
    runtime_data = bundle.get("runtime")
    all_sessions = bundle.get("sessions", [])
    selected_sessions = all_sessions
    if not args.session:
        return 0, compile_data, runtime_data, selected_sessions
    selected_sessions = [
        item for item in all_sessions if item.get("session_id") in set(args.session)
    ]
    if not selected_sessions:
        LOGGER.error("requested session_id not found: " + ", ".join(args.session))
        return 5, None, None, []
    selected = selected_sessions[0]
    compile_data = selected.get("compile")
    runtime_data = selected.get("runtime")
    if runtime_data is not None:
        runtime_data["compile_runtime_correlation"] = selected.get(
            "correlation", "unknown"
        )
        runtime_data["correlation_evidence"] = selected.get("correlation_evidence", [])
        runtime_data["associated_graph"] = selected.get("graph")
        runtime_data["associated_compile_completed_at"] = (compile_data or {}).get(
            "root_graph_completed_at"
        )
    return 0, compile_data, runtime_data, selected_sessions


def _append_classification_warnings(warnings, mode, compile_data, runtime_data):
    if (
        mode.get("compile_runtime_correlation") == "unknown"
        and compile_data
        and runtime_data
    ):
        warnings.append(
            "compile/runtime stream counts are available, but graph name and model name "
            "do not prove the same model session"
        )
    if (
        mode.get("graph_hierarchy_status") == "unknown"
        and compile_data
        and compile_data.get("graph_scopes")
    ):
        warnings.append(
            "graph/subgraph parent relationship was not observed; subgraph num and graph "
            "name are not sufficient to infer parent_graph or parent_node"
        )
    if mode["analysis_status"] == "no_stream_evidence":
        warnings.append(
            "no stream evidence found; checked compile anchors "
            "AssignLogicalStreams/At last, root graph/SetLogicStreamIdAttr "
            "and runtime anchors InitRuntimeParams/Create new stream/Logical stream index/"
            "Build RT2 executor/Collect rt2 stream/SplitRtStreams"
        )


def _build_report(context: ReportContext) -> Dict[str, Any]:
    warnings = context.context_warnings + list(context.bundle.get("warnings", []))
    mode = classify(context.compile_data, context.runtime_data)
    _append_classification_warnings(
        warnings, mode, context.compile_data, context.runtime_data
    )
    return {
        "context": {
            "repo_root": (
                str(context.repo_root) if context.repo_root is not None else None
            ),
            "analysis_mode": context.analysis_mode,
            "repo_context_missing": (
                context.missing if context.repo_root is None else []
            ),
            "logs": {
                key: [str(item) for item in value]
                for key, value in context.logs.items()
            },
            "log_level_filter": "INFO",
            "input_mode": ("generic" if context.generic_logs else "phase_specific"),
        },
        "mode": mode,
        "model_overview": {
            "root_graph": (context.compile_data or {}).get("graph")
            or (context.runtime_data or {}).get("runtime_graph_name")
            or (context.runtime_data or {}).get("executor_graph"),
            "scenario_class": mode.get("scenario_class", "unknown"),
            "root_graph_class": mode.get("root_graph_class", "unknown"),
            "graph_form": mode.get("graph_form", "unknown"),
            "runtime_backend": mode.get("runtime_backend", "unknown"),
            "classification_evidence": mode.get("classification_evidence", []),
        },
        "graphs": build_graph_inventory(
            context.compile_data, context.selected_sessions
        ),
        "compile": context.compile_data,
        "runtime": context.runtime_data,
        "sessions": context.selected_sessions,
        "stream_records": build_stream_records(context.selected_sessions),
        "batch_mapping": build_batch_mapping(
            context.compile_data, context.runtime_data
        ),
        "warnings": warnings,
    }


def _emit_report(args, report):
    if args.format == "json":
        DATA_LOGGER.info(json.dumps(report, ensure_ascii=False, indent=2))
        return
    DATA_LOGGER.info(
        render_markdown(
            report, None if args.full_ops else args.sample_limit, verbose=args.verbose
        )
    )


def main() -> int:
    args = parse_args()
    status, input_data = _load_input_logs(args)
    if status:
        return status
    repo_root, missing, context_warnings, analysis_mode, logs, generic_logs = input_data
    # Compile and runtime evidence is collected in one ordered graph
    # workflow.  This prevents the old two-pass scanners from losing the
    # relationship between a graph's compile exit and its V2 trace-only
    # runtime block (which has no model_id or graph name).
    bundle = analyze_workflows(logs, args.full_ops, args.sample_limit)
    status, compile_data, runtime_data, selected_sessions = _select_report_sessions(
        bundle, args
    )
    if status:
        return status
    report = _build_report(
        ReportContext(
            repo_root=repo_root,
            missing=missing,
            context_warnings=context_warnings,
            analysis_mode=analysis_mode,
            logs=logs,
            generic_logs=generic_logs,
            bundle=bundle,
            compile_data=compile_data,
            runtime_data=runtime_data,
            selected_sessions=selected_sessions,
        )
    )
    _emit_report(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
