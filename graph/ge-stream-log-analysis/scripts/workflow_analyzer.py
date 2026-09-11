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
"""Stateful GE compile/runtime workflow correlation."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Iterable, Optional

from workflow_correlation import (
    _analysis_status,
    _candidate_score,
    _compile_count,
    _compile_counts,
    _compile_logic_stream_ids,
    _delta,
    _is_v2_trace_runtime,
    _runtime_count,
    _runtime_path,
    _runtime_trace_indexes,
    _scenario,
    _v2_runtime_fragments_compatible,
)
from workflow_events import (
    CORRELATION_WINDOW_SECONDS,
    INIT_DEDUP_WINDOW_SECONDS,
    CompileSpec,
    Event,
    LogRecord,
    _add_evidence,
    _add_phase,
    _conflicting_v2_bindings,
    _contains_v2_binding,
    _dynamic_subgraph_root,
    _effective_v2_bindings,
    _event,
    _has_compile_stream_evidence,
    _is_batch_graph,
    _is_subgraph,
    _new_compile,
    _new_runtime,
    _phase_evidence,
    _same_scope_v2_summaries,
    _v2_resource_signature,
    eventize,
)


def _allocation_reason(text):
    if "[Assign][NewStreamId" in text:
        return "stream_label"
    if "[Reuse][Stream]" in text:
        return "reuse"
    if "Assign attached stream" in text:
        return "attached"
    if "has user stream label" in text:
        return "user_stream_label"
    return "other"


def _record_dynamic_assignment(
    state: dict[str, Any],
    stream_id: int,
    item: dict[str, Any],
    owner: str,
    final: bool,
) -> None:
    if stream_id < 0:
        return
    assignments = state.setdefault("dynamic_current_assignments", {})
    old = assignments.get(owner)
    if owner.startswith("node:"):
        node = owner[5:].split("|", 1)[0]
        for old_owner in list(assignments):
            if not old_owner.startswith("node:") or old_owner == owner:
                continue
            if old_owner[5:].split("|", 1)[0] == node:
                assignments.pop(old_owner, None)
    assignments[owner] = stream_id
    if not final:
        ids = state.setdefault("dynamic_initial_logic_stream_ids", [])
        if stream_id not in ids:
            ids.append(stream_id)
    state["dynamic_logic_stream_ids"] = sorted(
        {
            value
            for value in assignments.values()
            if isinstance(value, int) and value >= 0
        }
    )
    transition = {
        "owner": owner,
        "stream_id": stream_id,
        "previous_stream_id": old,
        "stage": "final" if final else "initial",
        **item,
    }
    state.setdefault("dynamic_assignment_history", []).append(transition)
    state.setdefault("dynamic_stream_assignment_evidence", []).append(
        {"stage": "final" if final else "initial", **item}
    )
    _add_evidence(state, "dynamic_stream_assignment", item)


def _record_compile_end(state, event, item):
    subtype = event.payload.get("subtype")
    record = event.record
    if subtype == "static":
        state.update(
            logical_stream_count=event.payload["stream_count"],
            main_stream_count=event.payload["main_stream_count"],
            attached_stream_count=event.payload["attached_stream_count"],
            root_graph_summary_at=record.timestamp,
            root_graph_completed_at=record.timestamp,
            root_graph_completion_evidence=item,
            window_state="compiled",
        )
        _add_evidence(state, "logical_stream_assigned", item)
        return True
    if subtype == "dynamic":
        state.update(
            dynamic_stream_count=event.payload["stream_count"],
            dynamic_event_count=event.payload["event_count"],
            dynamic_stream_at=record.timestamp,
            root_graph_completed_at=record.timestamp,
            root_graph_completion_evidence=item,
            window_state="compiled",
        )
        _add_evidence(state, "dynamic_stream_assigned", item)
        return True
    if subtype == "physical_exit":
        state.update(
            final_model_stream_count=event.payload["stream_count"],
            notify_count=event.payload["notify_count"],
            event_count=event.payload["event_count"],
            physical_split=(
                "explicit_split"
                if event.payload["stream_count"]
                > (state.get("logical_stream_count") or 0)
                else "no_split_observed"
            ),
        )
        if state.get("root_graph_completed_at") is None:
            state["root_graph_completed_at"] = record.timestamp
            state["root_graph_completion_evidence"] = item
        _add_evidence(state, "physical_stream_finalized", item)
        return True
    return False


def _record_compile_sync(state, event, item):
    subtype = event.payload.get("subtype")
    if subtype == "sync_exit":
        state.update(
            sync_stream_count=event.payload["stream_count"],
            sync_notify_count=event.payload["notify_count"],
            sync_event_count=event.payload["event_count"],
        )
        _add_evidence(state, "sync_generated", item)
        return True
    if subtype == "sync_attr":
        sync_evidence = state.setdefault(
            "sync_evidence", {"send_recv": 0, "event_attrs": 0}
        )
        sync_evidence["event_attrs"] = sync_evidence.get("event_attrs", 0) + 1
        _add_evidence(state, "sync_generated", item)
        return True
    return False


def _record_compile_allocation(state, event, item):
    subtype = event.payload.get("subtype")
    if subtype == "allocation":
        state.setdefault("compile_stream_allocation_evidence", []).append(item)
        reasons = state.setdefault("allocation_reasons", Counter())
        reasons[_allocation_reason(event.record.raw)] += 1
        _add_evidence(state, "compile_stream_allocation", item)
        return True
    if subtype == "attached_allocation":
        state["attached_stream_count"] = state.get("attached_stream_count") or 0
        state["attached_stream_count"] += 1
        state.setdefault("allocation_reasons", Counter())["attached"] += 1
        state.setdefault("compile_stream_allocation_evidence", []).append(item)
        _add_evidence(state, "compile_stream_allocation", item)
        return True
    if subtype == "delegated_split":
        state["physical_split"] = "delegated_to_rts"
        _add_evidence(state, "physical_stream_finalized", item)
        return True
    return False


def _is_late_dynamic_scope(scope_graph, event):
    return bool(
        scope_graph.lower().endswith("_dynamic")
        and not _is_subgraph(scope_graph)
        and event.payload["subgraph_count"] > 0
    )


def _without_synthetic_build(items):
    filtered = []
    for entry in items:
        keyword = str(entry.get("keyword", ""))
        if "Begin to build known shape graph" not in keyword:
            filtered.append(entry)
    return filtered


def _is_dynamic_child(child, state, prefix):
    same_pid = child.get("pid") in (None, state.get("pid"))
    same_context = child.get("ge_context") in (None, state.get("ge_context"))
    same_prefix = str(child.get("graph", "")).startswith(prefix)
    return child is not state and same_pid and same_context and same_prefix


def _merge_dynamic_child(state, child):
    for field_name in (
        "dynamic_logic_stream_ids",
        "dynamic_initial_logic_stream_ids",
    ):
        for stream_id in child.get(field_name, []):
            if stream_id not in state[field_name]:
                state[field_name].append(stream_id)
    state["dynamic_current_assignments"].update(
        child.get("dynamic_current_assignments", {})
    )
    state["dynamic_assignment_history"].extend(
        child.get("dynamic_assignment_history", [])
    )
    state["dynamic_stream_assignment_evidence"].extend(
        child.get("dynamic_stream_assignment_evidence", [])
    )


def _record_compile_misc(state, event, item):
    subtype = event.payload.get("subtype")
    if subtype == "batch":
        state["dynamic_batch_markers"] = max(1, state.get("dynamic_batch_markers", 0))
        _add_evidence(state, event.payload.get("phase", "dynamic_batch"), item)
        return True
    if subtype == "timing":
        _add_evidence(state, event.payload.get("phase"), item)
        return True
    return False


def _keyword(event: Event) -> str:
    return (
        event.record.message.split(":", 1)[0].strip()
        if event.record.message
        else event.kind
    )


def _store_v2_binding(
    runtime: dict[str, Any], binding: dict[str, Any], bucket: str
) -> None:
    """Store a V2 binding while keeping allocation evidence authoritative."""
    index = int(binding["logical_stream_index"])
    rt_id = int(binding["rt_stream_id"])
    values = runtime.setdefault(bucket, [])
    if _contains_v2_binding(values, index, rt_id):
        return
    bucket_conflicts = _conflicting_v2_bindings(values, index, rt_id)
    if bucket_conflicts:
        runtime.setdefault("v2_binding_conflicts", []).append(
            {
                "logical_stream_index": index,
                "existing": bucket_conflicts,
                "incoming": binding,
            }
        )
    values.append(binding)

    effective = _effective_v2_bindings(runtime)
    current = effective.get(index)
    if (
        not bucket_conflicts
        and current is not None
        and int(current.get("rt_stream_id", -1)) != rt_id
    ):
        conflict = {
            "logical_stream_index": index,
            "existing": current,
            "incoming": binding,
        }
        conflicts = runtime.setdefault("v2_binding_conflicts", [])
        if conflict not in conflicts:
            conflicts.append(conflict)
    # New GE INFO log allocation evidence takes precedence over a legacy
    # SplitRtStreams fallback.  Conflicting allocation evidence is kept
    # in v2_binding_conflicts and does not overwrite the first fact.
    if current is None or (
        current.get("binding_source") == "SplitRtStreams"
        and bucket == "allocation_bindings"
    ):
        effective[index] = binding
    runtime["runtime_logical_to_rt_bindings"] = [
        effective[key] for key in sorted(effective)
    ]


def _store_v2_resource_summary(
    runtime: dict[str, Any], summary: dict[str, Any]
) -> None:
    summaries = runtime.setdefault("v2_resource_summaries", [])
    signature = _v2_resource_signature(summary)
    if any(_v2_resource_signature(item) == signature for item in summaries):
        duplicates = runtime.setdefault("v2_resource_duplicate_evidence", [])
        if summary not in duplicates:
            duplicates.append(summary)
        return
    same_scope = _same_scope_v2_summaries(summaries, summary)
    if same_scope:
        conflict = {
            "scope": summary.get("scope"),
            "name": summary.get("name"),
            "existing": same_scope,
            "incoming": summary,
        }
        conflicts = runtime.setdefault("v2_resource_conflicts", [])
        if conflict not in conflicts:
            conflicts.append(conflict)
    summaries.append(summary)


def _runtime_binding_at_index(runtime, index):
    for binding in runtime.get("runtime_logical_to_rt_bindings", []):
        if int(binding.get("logical_stream_index", -1)) == index:
            return binding
    return None


def _runtime_backend(subtype):
    v2_subtypes = {
        "v2",
        "v2_resource",
        "v2_notify",
        "v2_send",
        "v2_wait",
        "v2_executor_anchor",
        "v2_collect",
    }
    return "v2_rt2" if subtype in v2_subtypes else "v1_davinci_model"


def _reuse_v2_runtime(runtime):
    return bool(
        runtime
        and runtime.get("runtime_backend") == "v2_rt2"
        and runtime.get("model_id") is None
        and runtime.get("init_runtime_at") is None
    )


def _record_runtime_anchor(runtime, event, item):
    if event.payload.get("subtype") == "v2_executor_anchor":
        record = event.record
        runtime.update(
            executor_graph=event.payload["graph"],
            executor_model=event.payload["model"],
            executor_anchor_evidence=item,
            runtime_anchor=item,
            runtime_started_at=record.timestamp,
        )
        runtime["model_name"] = event.payload["graph"]
        _add_evidence(runtime, "v2_executor_anchor", item)
        return True
    return False


def _same_runtime_init(runtime, event):
    if runtime.get("model_id") != event.payload["model_id"]:
        return False
    started_at = runtime.get("init_runtime_at")
    if started_at is None:
        return False
    delta = _delta(started_at, event.record.timestamp)
    return delta is not None and 0 <= delta <= INIT_DEDUP_WINDOW_SECONDS


def _created_stream(runtime, rt_stream_id):
    for item in reversed(runtime.get("created_streams", [])):
        if item.get("rt_stream_id") == rt_stream_id:
            return item
    return {}


def _v2_trace_binding(event):
    record = event.record
    return {
        "logical_stream_index": event.payload["logical_index"],
        "rt_stream_id": event.payload["rt_stream_id"],
        "file": str(record.path),
        "line": record.line,
        "keyword": "Get rts stream",
        "excerpt": record.raw.rstrip()[:500],
        "evidence_type": "direct",
        "binding_source": "SplitRtStreams",
    }


def _observed_trace_count(runtime):
    indexes = set()
    for entry in runtime["runtime_logical_to_rt_bindings"]:
        indexes.add(entry["logical_stream_index"])
    return len(indexes)


def _is_named_root_summary(summary, runtime):
    return bool(
        summary.get("scope") == "root_model"
        and summary.get("name")
        and runtime.get("model_name") is None
    )


def _apply_v2_resource_counts(runtime, summary):
    if summary.get("scope") not in ("root_graph", "root_model"):
        return
    field_map = {
        "total_stream_count": "model_total_stream_count",
        "reusable_stream_count": "reusable_stream_count",
        "attached_stream_count": "attached_stream_count",
        "event_count": "event_count",
        "notify_count": "notify_count",
    }
    for source, target in field_map.items():
        if summary.get(source) is not None:
            runtime[target] = summary[source]


def _record_runtime_v2_auxiliary(runtime, event, item):
    subtype = event.payload.get("subtype")
    record = event.record
    if subtype == "v2_notify":
        notify = {
            "logical_notify_index": event.payload["logical_index"],
            "file": str(record.path),
            "line": record.line,
            "keyword": "Get rts notify",
            "excerpt": record.raw.rstrip()[:500],
        }
        runtime.setdefault("kernel_trace_notifies", []).append(notify)
        _add_evidence(runtime, "runtime_notify_bound", item)
        return True
    if subtype in ("v2_send", "v2_wait"):
        trace = {
            "logic_event_id": event.payload["event_id"],
            "rt_event": event.payload["rt_event"],
            "stream_id": event.payload["stream_id"],
            "file": str(record.path),
            "line": record.line,
            "keyword": "Sent event" if subtype == "v2_send" else "Waited event",
            "excerpt": record.raw.rstrip()[:500],
        }
        runtime.setdefault("kernel_trace_events", []).append(trace)
        _add_evidence(
            runtime,
            "runtime_event_sent" if subtype == "v2_send" else "runtime_event_waited",
            item,
        )
        return True
    return False


def _record_runtime_name(runtime, event, item):
    del item
    subtype = event.payload.get("subtype")
    if subtype == "model_name":
        # A compile-time graph_name must never populate this field.  At
        # this point the event is already in a runtime block.
        if (
            runtime.get("init_runtime_at") is not None
            or runtime.get("runtime_backend") == "v2_rt2"
        ):
            runtime["model_name"] = event.payload["name"]
        return True
    if subtype == "graph_name":
        if (
            runtime.get("model_id") is not None
            or runtime.get("init_runtime_at") is not None
        ):
            runtime["model_name"] = event.payload["name"].rstrip(".")
        return True
    return False


def _record_runtime_operator(runtime, event, item):
    del item
    if event.payload.get("subtype") != "operator":
        return False
    record = event.record
    index = event.payload["logical_index"]
    binding = runtime.get("bindings", {}).get(str(index), {})
    runtime.setdefault("operator_bindings", []).append(
        {
            "node": event.payload["node"],
            "logic_stream_id": index,
            "rt_stream_id": binding.get("rt_stream_id"),
            "stream_pointer": event.payload.get("pointer"),
            "pid": record.pid,
            "ge_context": record.context,
            "timestamp": record.timestamp,
            "file": str(record.path),
            "line": record.line,
        }
    )
    return True


def _record_runtime_batch(runtime, event, item):
    subtype = event.payload.get("subtype")
    if subtype == "batch_size":
        runtime["batch_size"] = event.payload["value"]
        runtime["batch_observation"] = "observed"
        _add_evidence(runtime, "batch_size_set", item)
        return True
    if subtype == "batch_label":
        runtime["active_batch_label"] = event.payload["value"]
        runtime["batch_observation"] = "observed"
        _add_evidence(runtime, "batch_label_observed", item)
        return True
    if subtype == "active_stream":
        runtime.setdefault("active_streams", {})[event.payload["branch"]] = (
            event.payload["logical_index"]
        )
        runtime["batch_observation"] = "observed"
        _add_evidence(runtime, "batch_stream_active", item)
        return True
    return False


def _record_runtime_context(runtime, event, item):
    if event.payload.get("subtype") != "context":
        return False
    marker = {"phase": event.payload["phase"], **item}
    runtime.setdefault("hybrid_context_markers", []).append(marker)
    _add_evidence(runtime, event.payload["phase"], item)
    return True


def _candidate_scope_matches(compile_state, runtime, v2_runtime):
    if v2_runtime and compile_state.get("graph_class") == "dynamic_shape":
        exact_graph = runtime.get("executor_graph") == compile_state.get("graph")
        return bool(
            compile_state.get("dynamic_stream_count") is not None or exact_graph
        )
    return bool(
        compile_state.get("graph_scope") == "outer_graph"
        and not _is_subgraph(compile_state.get("graph"))
    )


def _candidate_has_compile_exit(compile_state, runtime, v2_runtime):
    executor_match = bool(
        v2_runtime and runtime.get("executor_graph") == compile_state.get("graph")
    )
    return bool(
        compile_state.get("root_graph_completed_at") is not None
        or _compile_count(compile_state) is not None
        or executor_match
    )


def _v2_trace_candidate_score(workflow, runtime):
    compile_state = workflow["compile"]
    if not runtime.get("pid") or compile_state.get("pid") != runtime.get("pid"):
        return None
    trace_indexes = _runtime_trace_indexes(runtime)
    if not trace_indexes.issubset(_compile_logic_stream_ids(compile_state)):
        return None
    delta = _delta(
        compile_state.get("root_graph_completed_at"),
        runtime.get("runtime_started_at"),
    )
    if delta is not None and delta <= 0:
        return None
    return _candidate_score(workflow, runtime)


def _v2_candidate_named(compile_state, runtime):
    executor_graph = runtime.get("executor_graph")
    if executor_graph:
        return executor_graph == compile_state.get("graph")
    return bool(
        runtime.get("model_name")
        and runtime.get("model_name") == compile_state.get("graph")
    )


def _candidate_count_mismatch(compile_state, runtime):
    runtime_count, _count_kind = _runtime_count(runtime)
    compile_counts = _compile_counts(compile_state)
    return bool(
        runtime_count is not None
        and compile_counts
        and runtime_count not in compile_counts
    )


def _candidate_identity_complete(compile_state, runtime):
    return bool(
        runtime.get("pid")
        and compile_state.get("pid") == runtime.get("pid")
        and runtime.get("ge_context")
        and compile_state.get("ge_context") == runtime.get("ge_context")
    )


def _v2_preferred_candidates(candidates, runtime):
    named = []
    exact = []
    runtime_indexes = _runtime_trace_indexes(runtime)
    for item in candidates:
        compile_state = item["compile"]
        executor_match = bool(
            runtime.get("executor_graph")
            and runtime.get("executor_graph") == compile_state.get("graph")
        )
        model_match = bool(
            not runtime.get("executor_graph")
            and runtime.get("model_name") == compile_state.get("graph")
        )
        if executor_match or model_match:
            named.append(item)
        if runtime_indexes == _compile_logic_stream_ids(compile_state):
            exact.append(item)
    return named or exact or candidates


def _v2_correlation_evidence(compile_state, runtime):
    evidence_items: list[str] = []
    if runtime.get("pid") and runtime.get("pid") == compile_state.get("pid"):
        evidence_items.append("same_pid")
    if runtime.get("executor_graph") == compile_state.get("graph"):
        evidence_items.append("executor_graph_exact_match")
    if runtime.get("dynamic_subgraph_fallback"):
        fallback = runtime.get("correlation_evidence") or [
            "dynamic_subgraph_family_match"
        ]
        evidence_items.extend(fallback)
    if _is_v2_trace_runtime(runtime):
        evidence_items.append("runtime_logical_index_match")
    _add_context_and_compile_order(evidence_items, compile_state, runtime)
    return evidence_items


def _add_context_and_compile_order(evidence_items, compile_state, runtime):
    if runtime.get("ge_context") and runtime.get("ge_context") == compile_state.get(
        "ge_context"
    ):
        evidence_items.append("same_ge_context")
    delta = _delta(
        compile_state.get("root_graph_completed_at"),
        runtime.get("runtime_started_at"),
    )
    if delta is not None and delta > 0:
        evidence_items.append("compile_before_runtime")
    return delta


def _v1_correlation_evidence(compile_state, runtime):
    evidence_items = []
    if runtime.get("pid") and runtime.get("pid") == compile_state.get("pid"):
        evidence_items.append("same_pid")
    delta = _add_context_and_compile_order(evidence_items, compile_state, runtime)
    if delta is not None and delta <= CORRELATION_WINDOW_SECONDS:
        evidence_items.append("same_time_window")
    runtime_count, count_kind = _runtime_count(runtime)
    if runtime_count is not None and runtime_count in _compile_counts(compile_state):
        evidence_items.append(
            "trace_index_count_match" if count_kind == "trace" else "stream_count_match"
        )
    elif runtime_count is not None and runtime.get("model_name") == compile_state.get(
        "graph"
    ):
        evidence_items.append("stream_count_mismatch")
    evidence_items.append("no_intervening_model_boundary")
    return evidence_items


def _apply_dynamic_fallback(runtime, root, children, exact_roots):
    runtime["dynamic_subgraph_fallback"] = True
    runtime["runtime_graph_identity"] = (
        "dynamic_subgraph_family" if children else "dynamic_root_compile_missing"
    )
    runtime["runtime_graph_class"] = "dynamic_shape"
    runtime["runtime_graph_name"] = root
    runtime["compile_runtime_correlation"] = runtime["runtime_graph_identity"]
    runtime["correlation_evidence"] = [
        "executor_graph_exact_match" if exact_roots else "executor_graph_dynamic_root",
        "dynamic_subgraph_family_match"
        if children
        else "dynamic_root_compile_summary_missing",
        "runtime_single_stream_observed",
        "dynamic_compile_summary_missing",
    ]
    for workflow in [*(children or []), *exact_roots]:
        state = workflow["compile"]
        state["report_visibility"] = "suppressed"
        state["compile_suppressed_reason"] = (
            "single_stream_dynamic_subgraph_without_compile_stream_evidence"
        )


def _latest_compatible_v2_block(blocks, source):
    for target in reversed(blocks):
        if _v2_runtime_fragments_compatible(target, source):
            return target
    return None


def _preceding_v2_anchor(anchors, source):
    target = anchors[0]
    source_order = source.get("runtime_order", 0)
    for anchor in anchors:
        if anchor.get("runtime_order", 0) <= source_order:
            target = anchor
    return target


def _merge_v2_times(target, source):
    started_values = []
    for value in (
        target.get("runtime_started_at"),
        source.get("runtime_started_at"),
    ):
        if value is not None:
            started_values.append(value)
    last_values = []
    for value in (
        target.get("runtime_last_at"),
        source.get("runtime_last_at"),
    ):
        if value is not None:
            last_values.append(value)
    target["runtime_started_at"] = min(started_values)
    target["runtime_last_at"] = max(last_values)
    target["runtime_order"] = min(
        int(target.get("runtime_order", 0)), int(source.get("runtime_order", 0))
    )


def _merge_v2_scalar_fields(target, source):
    for field_name in (
        "model_id",
        "model_name",
        "model_stream_count",
        "model_total_stream_count",
        "requested_stream_count",
        "reusable_stream_count",
        "attached_stream_count",
        "event_count",
        "notify_count",
    ):
        if target.get(field_name) is None and source.get(field_name) is not None:
            target[field_name] = source[field_name]
    for field_name in (
        "executor_graph",
        "executor_model",
        "executor_anchor_evidence",
        "runtime_anchor",
    ):
        if target.get(field_name) is None and source.get(field_name) is not None:
            target[field_name] = source[field_name]


def _merge_v2_traces(target, source):
    trace_keys = set()
    for item in target.get("kernel_trace_bindings", []):
        trace_keys.add((item.get("logical_stream_index"), item.get("rt_stream_id")))
    for item in source.get("kernel_trace_bindings", []):
        key = (item.get("logical_stream_index"), item.get("rt_stream_id"))
        if key not in trace_keys:
            target.setdefault("kernel_trace_bindings", []).append(item)
            trace_keys.add(key)


def _merge_v2_evidence(target, source):
    for phase, items in source.get("phases", {}).items():
        for item in items:
            _add_phase(target, phase, item)
    for item in source.get("evidence", []):
        _add_evidence(target, None, item)


def _populate_compile_stream_placeholders(state):
    count = _compile_count(state)
    if count is not None:
        for stream_id in range(max(0, int(count))):
            state["streams"].setdefault(
                str(stream_id),
                {"operator_count": None, "operators": [], "batch_ids": {}},
            )


def _finalize_compile_metadata(state):
    state["operator_mapping"] = "observed" if state["operator_facts"] else "unknown"
    ids = [int(key) for key in state["streams"]]
    state["operator_derived_stream_count"] = max(ids) + 1 if ids else None
    logical_count = _compile_count(state)
    if logical_count is not None and state["operator_derived_stream_count"] is not None:
        state["stream_count_consistency"] = (
            "passed"
            if state["operator_derived_stream_count"] <= logical_count
            else "mismatch"
        )
    if state.get("window_state") == "open":
        state["window_state"] = "unknown"
    state["files"] = sorted(
        {item.get("file") for item in state.get("evidence", []) if item.get("file")}
    )


def _record_compile_operator(state, event, item):
    subtype = event.payload.get("subtype")
    record = event.record
    if subtype in ("operator", "dynamic_operator"):
        name = event.payload.get("name", "unknown")
        stream_id = int(event.payload.get("stream_id", -1))
        if stream_id < 0:
            return False
        op_type = event.payload.get("op_type", "unknown")
        state.setdefault("operator_facts", {})[name] = {
            "name": name,
            "type": op_type,
            "logic_stream_id": stream_id,
            "file": str(record.path),
            "line": record.line,
            "keyword": _keyword(event),
            "source_kind": "dynamic_refresh"
            if subtype == "dynamic_operator"
            else "compile",
        }
        _add_evidence(state, "operator_stream_mapping", item)
        if subtype == "dynamic_operator":
            # RefreshStreamsForGraphByNodeIds is the dynamic allocator's
            # final node assignment.  Treat it as a state replacement,
            # while retaining the transition in assignment history.
            _record_dynamic_assignment(
                state,
                stream_id,
                item,
                owner=f"node:{name}",
                final=True,
            )
        return True
    if subtype == "attached_operator":
        state.setdefault("attached_operator_mappings", []).append(
            {
                "name": event.payload["name"],
                "type": event.payload["op_type"],
                "stream_id": event.payload["stream_ids"][0],
                "logic_attached_stream_ids": list(event.payload["stream_ids"]),
                **item,
            }
        )
        _add_evidence(state, "attached_stream_mapping", item)
        return True
    return False


def _record_compile_assignment(state, event, item):
    subtype = event.payload.get("subtype")
    if subtype in ("dynamic_assign", "dynamic_assign_final", "dynamic_reassign"):
        stream_id = int(event.payload["stream_id"])
        record = event.record
        owner = event.payload.get("owner") or (f"line:{record.path}:{record.line}")
        _record_dynamic_assignment(
            state,
            stream_id,
            item,
            owner,
            final=subtype != "dynamic_assign",
        )
        return True
    return False


def _record_compile_split(state, event, item):
    subtype = event.payload.get("subtype")
    record = event.record
    if subtype == "static_split":
        state.setdefault("static_split_details", []).append({**event.payload, **item})
        state["physical_split"] = "explicit_split"
        _add_evidence(state, "physical_split_detail", item)
        return True
    if subtype == "static_split_limit":
        state.setdefault("static_split_details", []).append({**event.payload, **item})
        state["physical_split"] = "explicit_split"
        # The first node after a split is also a final operator fact.
        state.setdefault("operator_facts", {})[event.payload["name"]] = {
            "name": event.payload["name"],
            "type": event.payload["op_type"],
            "logic_stream_id": event.payload["stream_id"],
            "file": str(record.path),
            "line": record.line,
            "keyword": _keyword(event),
            "source_kind": "split_limit",
        }
        _add_evidence(state, "physical_split_detail", item)
        return True
    return False


def _record_runtime_v1(runtime, event, item):
    subtype = event.payload.get("subtype")
    record = event.record
    if subtype == "create":
        stream = {
            "pointer": event.payload["pointer"],
            "rt_stream_id": event.payload["rt_stream_id"],
            "rt_model_id": event.payload["rt_model_id"],
            "priority": event.payload["priority"],
            "flag": event.payload["flag"],
            "task_num": event.payload["task_num"],
            "role": "unclassified",
            "line": record.line,
            "file": str(record.path),
        }
        runtime.setdefault("created_streams", []).append(stream)
        _add_evidence(runtime, "stream_created", item)
        return True
    if subtype == "v1":
        index = event.payload["logical_index"]
        create = _created_stream(runtime, event.payload["rt_stream_id"])
        binding = {
            **create,
            "model_stream_index": index,
            "ge_model_id": event.payload["model_id"],
            "rt_stream_id": event.payload["rt_stream_id"],
            "flag": event.payload["flag"],
            "binding_line": record.line,
        }
        runtime.setdefault("bindings", {})[str(index)] = binding
        _add_evidence(runtime, "logical_stream_bound", item)
        return True
    if subtype == "total":
        runtime.update(
            model_total_stream_count=event.payload["total_stream_count"],
            model_stream_count=event.payload["stream_count"],
            follow_stream_count=event.payload["follow_stream_count"],
            hccl_stream_count=event.payload["follow_stream_count"],
        )
        _add_evidence(runtime, "runtime_stream_summary", item)
        return True
    return False


def _record_runtime_v2_binding(runtime, event, item):
    subtype = event.payload.get("subtype")
    record = event.record
    if subtype == "v2_collect":
        binding = {
            "logical_stream_index": int(event.payload["logical_index"]),
            "rt_stream_id": int(event.payload["rt_stream_id"]),
            "pointer": event.payload["pointer"],
            "role": event.payload["role"],
            "file": str(record.path),
            "line": record.line,
            "keyword": "Collect rt2 stream",
            "excerpt": record.raw.rstrip()[:500],
            "evidence_type": "allocation",
            "binding_source": (
                "OccupyStreamResource"
                if event.payload["role"] == "main"
                else "AcquireStreams"
            ),
        }
        _store_v2_binding(runtime, binding, "allocation_bindings")
        _add_evidence(runtime, "runtime_stream_bound", item)
        return True
    if subtype == "v2":
        trace = _v2_trace_binding(event)
        _store_v2_binding(runtime, trace, "execution_bindings")
        runtime.setdefault("kernel_trace_bindings", []).append(trace)
        runtime["observed_trace_stream_count"] = _observed_trace_count(runtime)
        _add_evidence(
            runtime,
            "runtime_stream_bound",
            _phase_evidence(record, "SplitRtStreams"),
        )
        return True
    return False


def _record_runtime_v2_resource(runtime, event, item):
    if event.payload.get("subtype") != "v2_resource":
        return False
    record = event.record
    summary = dict(event.payload["summary"])
    summary.update(
        file=str(record.path),
        line=record.line,
        keyword=_keyword(event),
        evidence=item,
    )
    _store_v2_resource_summary(runtime, summary)
    if _is_named_root_summary(summary, runtime):
        runtime["model_name"] = summary["name"]
    _apply_v2_resource_counts(runtime, summary)
    _add_evidence(runtime, "runtime_resource_summary", item)
    return True


def _v2_candidate_score(workflow, runtime):
    compile_state = workflow["compile"]
    executor_graph = runtime.get("executor_graph")
    if not executor_graph and (
        not runtime.get("pid") or compile_state.get("pid") != runtime.get("pid")
    ):
        return None
    if executor_graph and executor_graph != compile_state.get("graph"):
        return None
    delta = _delta(
        compile_state.get("root_graph_completed_at"),
        runtime.get("runtime_started_at"),
    )
    if delta is not None and delta <= 0:
        return None
    named = _v2_candidate_named(compile_state, runtime)
    if executor_graph and not named:
        return None
    if not named and _candidate_count_mismatch(compile_state, runtime):
        return None
    return _candidate_score(workflow, runtime)


def _select_runtime_candidate(candidates, runtime):
    if runtime.get("runtime_backend") == "v2_rt2":
        preferred = _v2_preferred_candidates(candidates, runtime)
        return preferred[0] if len(preferred) == 1 else None
    best = candidates[0]
    best_score = _candidate_score(best, runtime)
    equally_good = []
    for item in candidates:
        if _candidate_score(item, runtime) == best_score:
            equally_good.append(item)
    return best if len(equally_good) == 1 else None


def _attach_v2_runtime(workflow, runtime):
    compile_state = workflow["compile"]
    evidence_items = _v2_correlation_evidence(compile_state, runtime)
    workflow["runtime"] = runtime
    expected_indexes = _compile_logic_stream_ids(compile_state)
    trace_indexes = _runtime_trace_indexes(runtime)
    if trace_indexes and trace_indexes == expected_indexes:
        evidence_items.extend(["stream_count_match", "logical_index_set_match"])
    elif _runtime_count(runtime)[0] in _compile_counts(compile_state):
        evidence_items.append("stream_count_match")
    named = bool(
        runtime.get("executor_graph") == compile_state.get("graph")
        or runtime.get("model_name") == compile_state.get("graph")
    )
    workflow["correlation"] = "strong" if named else "same_session"
    combined = workflow.get("correlation_evidence", []) + evidence_items
    workflow["correlation_evidence"] = list(dict.fromkeys(combined))
    if workflow["correlation"] == "same_session":
        workflow["correlation_inference"] = "inferred"


def _attach_v1_runtime(workflow, runtime):
    compile_state = workflow["compile"]
    evidence_items = _v1_correlation_evidence(compile_state, runtime)
    workflow["runtime"] = runtime
    workflow["correlation"] = (
        "strong"
        if runtime.get("model_name") == compile_state.get("graph")
        else "same_session"
    )
    workflow["correlation_evidence"] = evidence_items
    if workflow["correlation"] == "same_session" and runtime.get("model_id") is None:
        workflow["correlation_inference"] = "inferred"


def _merge_v2_bindings(target, source):
    source_bucket_keys: set[tuple[int, int]] = set()
    for bucket in ("allocation_bindings", "execution_bindings"):
        for item in source.get(bucket, []):
            _store_v2_binding(target, item, bucket)
            source_bucket_keys.add(
                (int(item["logical_stream_index"]), int(item["rt_stream_id"]))
            )
    for item in source.get("runtime_logical_to_rt_bindings", []):
        key = (int(item["logical_stream_index"]), int(item["rt_stream_id"]))
        if key in source_bucket_keys:
            continue
        bucket = (
            "execution_bindings"
            if item.get("binding_source") == "SplitRtStreams"
            else "allocation_bindings"
        )
        _store_v2_binding(target, item, bucket)
    target["observed_trace_stream_count"] = len(
        target.get("runtime_logical_to_rt_bindings", [])
    )


def _merge_v2_details(target, source):
    for item in source.get("v2_resource_summaries", []):
        _store_v2_resource_summary(target, item)
    for field_name in (
        "v2_resource_duplicate_evidence",
        "v2_resource_conflicts",
        "kernel_trace_events",
        "kernel_trace_notifies",
        "v2_binding_conflicts",
    ):
        for item in source.get(field_name, []):
            if item not in target.setdefault(field_name, []):
                target[field_name].append(item)


def _merge_v2_runtime_fragment(target: dict[str, Any], source: dict[str, Any]) -> None:
    _merge_v2_times(target, source)
    _merge_v2_scalar_fields(target, source)
    _merge_v2_bindings(target, source)
    _merge_v2_traces(target, source)
    _merge_v2_details(target, source)
    _merge_v2_evidence(target, source)


def _merge_unanchored_v2_blocks(blocks):
    local = []
    for source in blocks:
        target = _latest_compatible_v2_block(local, source)
        if target is None:
            local.append(source)
        else:
            _merge_v2_runtime_fragment(target, source)
    return local


def _merge_anchored_v2_blocks(blocks, anchors):
    merged = list(anchors)
    for source in blocks:
        if source.get("executor_anchor_evidence"):
            continue
        target = _preceding_v2_anchor(anchors, source)
        if _v2_runtime_fragments_compatible(target, source):
            _merge_v2_runtime_fragment(target, source)
        else:
            merged.append(source)
    return merged


def _coalesce_v2_pid_blocks(blocks):
    anchors = []
    for item in blocks:
        if item.get("executor_anchor_evidence"):
            anchors.append(item)
    anchors.sort(key=lambda item: item.get("runtime_order", 0))
    if not anchors:
        return _merge_unanchored_v2_blocks(blocks)
    return _merge_anchored_v2_blocks(blocks, anchors)


class WorkflowAnalyzer:
    """Consume normalized events and assemble graph-centric workflows."""

    def __init__(self, full_ops: bool = False, sample_limit: int = 5):
        self.full_ops = full_ops
        self.sample_limit = sample_limit
        self.workflows: list[dict[str, Any]] = []
        self.runtime_blocks: list[dict[str, Any]] = []
        self.active_compile: dict[
            tuple[Optional[str], Optional[str]], dict[str, Any]
        ] = {}
        self.active_runtime: dict[
            tuple[Optional[str], Optional[str]], dict[str, Any]
        ] = {}
        self.graph_sequences: Counter[tuple[Optional[str], str]] = Counter()
        self.runtime_sequences: Counter[tuple[Optional[str], Optional[str]]] = Counter()
        self.dynamic_outer: dict[
            tuple[Optional[str], Optional[str]], dict[str, Any]
        ] = {}
        self.pending_batch: list[Event] = []
        self.pending_dynamic_outer: list[Event] = []
        self.unbound_graph_scopes: set[
            tuple[tuple[Optional[str], Optional[str]], Optional[str]]
        ] = set()
        self.pending_context: list[Event] = []
        self.graph_scopes: list[dict[str, Any]] = []
        self.graph_relations: list[dict[str, Any]] = []
        self.files: set[str] = set()
        self.warnings: list[str] = []
        self.input_stages: set[str] = set()
        self.global_compile_phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.global_compile_evidence: list[dict[str, Any]] = []
        self.pending_runtime_names: list[Event] = []

    @staticmethod
    def identity(record: LogRecord) -> tuple[Optional[str], Optional[str]]:
        return record.pid, record.context

    @staticmethod
    def _runtime_session(runtime: dict[str, Any]) -> dict[str, Any]:
        runtime_graph = runtime.get("runtime_graph_name") or runtime.get(
            "executor_graph"
        )
        is_dynamic_family = runtime.get("runtime_graph_identity") in (
            "dynamic_subgraph_family",
            "dynamic_root_compile_missing",
        )
        correlation = (
            runtime.get("runtime_graph_identity") if is_dynamic_family else "unknown"
        )
        return {
            "session_id": runtime["session_id"],
            "order": runtime.get("runtime_order", 0),
            "graph": runtime_graph,
            "model_id": runtime.get("model_id"),
            "session_type": "runtime_model",
            "session_role": "runtime_aggregate"
            if is_dynamic_family
            else "runtime_model",
            "correlation": correlation,
            "correlation_evidence": runtime.get("correlation_evidence", []),
            "correlation_inference": "inferred" if is_dynamic_family else None,
            "correlation_candidates": runtime.get("correlation_candidates", []),
            "scenario_class": "dynamic"
            if is_dynamic_family
            else _scenario(None, runtime),
            "graph_class": "dynamic_shape" if is_dynamic_family else "unknown",
            "compile_path": "dynamic_shape" if is_dynamic_family else "unknown",
            "runtime_path": runtime.get("runtime_backend", "unknown"),
            "stream_scope": "runtime_model",
            "compile": None,
            "runtime": runtime,
            "analysis_status": "partial",
            "report_visibility": "runtime_binding_only"
            if is_dynamic_family
            else "visible",
        }

    @staticmethod
    def _compile_session(workflow: dict[str, Any]) -> dict[str, Any]:
        state = workflow["compile"]
        return {
            "session_id": state["session_id"],
            "order": workflow.get("order", 0),
            "graph": workflow["graph"],
            "model_id": None,
            "session_type": "compile_graph",
            "session_role": state.get("session_role", "compile_graph"),
            "correlation": "unknown",
            "correlation_evidence": [],
            "correlation_inference": None,
            "correlation_candidates": [],
            "scenario_class": _scenario(state),
            "graph_class": state.get("graph_class", "unknown"),
            "compile_path": state.get("compile_path", "unknown"),
            "runtime_path": "unknown",
            "stream_scope": state.get("stream_scope", "graph"),
            "compile": state,
            "runtime": None,
            "analysis_status": "partial",
        }

    def consume(self, event: Event) -> None:
        self.files.add(str(event.record.path))
        if event.kind == "GRAPH_BEGIN":
            self._new_workflow(event)
        elif event.kind == "COMPILE_FACT":
            self._record_compile(event)
        elif event.kind == "COMPILE_END":
            self._record_compile(event)
        elif event.kind in ("RUNTIME_FACT", "RUNTIME_BIND"):
            self._record_runtime(event)

    def consume_records(self, records: Iterable[LogRecord]) -> None:
        for record in records:
            for event in eventize(record):
                self.consume(event)

    def finalize(self) -> list[dict[str, Any]]:
        for state in self._compile_states():
            self._finalize_compile(state)
        self._mark_hybrid_static_subgraphs()
        self._coalesce_v2_runtime_blocks()
        self._prepare_dynamic_subgraph_fallback()
        self._pair_runtime_blocks()
        return self._build_sessions()

    def _new_workflow(self, event: Event) -> dict[str, Any]:
        graph = str(event.payload["graph"])
        shape = str(event.payload.get("shape", "known"))
        key = (event.record.pid, graph)
        self.graph_sequences[key] += 1
        compile_state = _new_compile(
            CompileSpec(
                graph=graph,
                pid=event.record.pid,
                context=event.record.context,
                timestamp=event.record.timestamp,
                sequence=self.graph_sequences[key],
                shape=shape,
            )
        )
        workflow = {
            "workflow_id": f"workflow:{event.record.pid or 'unknown'}:{graph}:{self.graph_sequences[key]}",
            "order": event.record.order,
            "graph": graph,
            "compile": compile_state,
            "runtime": None,
            "correlation": "unknown",
            "correlation_evidence": [],
            "correlation_inference": None,
            "correlation_candidates": [],
        }
        begin_item = _phase_evidence(
            event.record, f"Begin to build {shape} shape graph"
        )
        _add_evidence(compile_state, "graph_build_started", begin_item)
        identity = self.identity(event.record)
        if (
            compile_state["graph_scope"] == "subgraph"
            and identity in self.dynamic_outer
        ):
            outer = self.dynamic_outer[identity]
            compile_state["outer_graph"] = outer["graph"]
            compile_state["compile_family_id"] = outer["compile_family_id"]
        elif compile_state["graph_scope"] == "outer_graph":
            self.dynamic_outer[identity] = (
                compile_state
                if shape == "unknown"
                else self.dynamic_outer.get(identity, compile_state)
            )
        self.workflows.append(workflow)
        self.active_compile[identity] = workflow
        self._attach_pending_batch(compile_state, event.record)
        self._attach_pending_dynamic_outer(compile_state, event.record)
        return workflow

    def _find_workflow(
        self, graph: Optional[str], record: LogRecord, shape: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        if graph is None and shape is None:
            active = self.active_compile.get(self.identity(record))
            if active is not None:
                return active
        candidates = []
        expected_class = "dynamic_shape" if shape == "unknown" else "static_shape"
        for item in self.workflows:
            compile_state = item["compile"]
            if graph is not None and item["graph"] != graph:
                continue
            if compile_state.get("pid") not in (None, record.pid):
                continue
            if compile_state.get("ge_context") not in (None, record.context):
                continue
            if shape is not None and compile_state.get("graph_class") != expected_class:
                continue
            candidates.append(item)
        if candidates:
            return candidates[-1]
        # Some supported GE INFO log versions omit the build-entry line while
        # retaining a graph-qualified allocator/exit fact.  Keep this legacy
        # recovery for those facts; attached-stream scopes and aggregate-only
        # records are filtered before reaching this helper.
        if graph:
            fake = _event(record, "GRAPH_BEGIN", shape=shape or "known", graph=graph)
            return self._new_workflow(fake)
        return self.active_compile.get(self.identity(record))

    def _current_compile(self, record: LogRecord) -> Optional[dict[str, Any]]:
        workflow = self.active_compile.get(self.identity(record))
        return workflow or (self.workflows[-1] if self.workflows else None)

    def _attach_pending_batch(
        self, compile_state: dict[str, Any], record: LogRecord
    ) -> None:
        for pending in list(self.pending_batch):
            if pending.record.pid not in (
                None,
                record.pid,
            ) or pending.record.context not in (None, record.context):
                continue
            delta = _delta(pending.record.timestamp, record.timestamp)
            if delta is not None and not 0 <= delta <= CORRELATION_WINDOW_SECONDS:
                continue
            item = _phase_evidence(
                pending.record,
                pending.payload.get(
                    "keyword", pending.payload.get("phase", "dynamic batch")
                ),
            )
            compile_state["dynamic_batch_markers"] = max(
                1, compile_state.get("dynamic_batch_markers", 0)
            )
            _add_evidence(
                compile_state, pending.payload.get("phase", "dynamic_batch"), item
            )
            self.pending_batch.remove(pending)

    def _attach_pending_dynamic_outer(
        self, compile_state: dict[str, Any], record: LogRecord
    ) -> None:
        """Attach an outer dynamic marker printed before its graph entry."""
        if (
            compile_state.get("graph_class") != "dynamic_shape"
            or compile_state.get("graph_scope") != "outer_graph"
        ):
            return
        for pending in list(self.pending_dynamic_outer):
            if pending.record.pid not in (
                None,
                record.pid,
            ) or pending.record.context not in (None, record.context):
                continue
            delta = _delta(pending.record.timestamp, record.timestamp)
            if delta is not None and not 0 <= delta <= CORRELATION_WINDOW_SECONDS:
                continue
            item = _phase_evidence(pending.record, "BuildForDynamicShapeGraph")
            _add_evidence(compile_state, "dynamic_outer_entry", item)
            self.pending_dynamic_outer.remove(pending)

    def _record_compile(self, event: Event) -> None:
        item = _phase_evidence(event.record, _keyword(event))
        workflow = self._compile_workflow(event, item)
        if workflow is None:
            return
        state = workflow["compile"]
        handlers = (
            _record_compile_end,
            _record_compile_sync,
            _record_compile_operator,
            _record_compile_assignment,
            _record_compile_split,
            _record_compile_allocation,
            self._record_compile_scope,
            self._record_compile_relation,
            _record_compile_misc,
        )
        for handler in handlers:
            if handler(state, event, item):
                return

    def _compile_workflow(self, event: Event, item: dict[str, Any]):
        record = event.record
        subtype = event.payload.get("subtype")
        graph = event.payload.get("graph")
        if subtype == "dynamic_outer_entry":
            self._store_dynamic_outer_entry(event, item)
            return None
        if subtype == "graph_scope":
            workflow = self._scoped_workflow(graph, record)
            if workflow is None:
                self.unbound_graph_scopes.add((self.identity(record), graph))
                return None
        else:
            if (
                subtype == "dynamic"
                and (
                    self.identity(record),
                    graph,
                )
                in self.unbound_graph_scopes
            ):
                return None
            shape = "unknown" if subtype == "dynamic" else None
            workflow = self._find_workflow(graph, record, shape)
        if workflow is None and subtype == "batch":
            self.pending_batch.append(event)
        elif workflow is None and subtype == "timing":
            self._store_global_compile_timing(event, item)
        return workflow

    def _store_dynamic_outer_entry(self, event: Event, item: dict[str, Any]) -> None:
        current = self.active_compile.get(self.identity(event.record))
        is_dynamic_outer = bool(
            current
            and current["compile"].get("graph_class") == "dynamic_shape"
            and current["compile"].get("graph_scope") == "outer_graph"
        )
        if is_dynamic_outer:
            _add_evidence(current["compile"], "dynamic_outer_entry", item)
        else:
            self.pending_dynamic_outer.append(event)

    def _scoped_workflow(self, graph, record):
        for candidate in reversed(self.workflows):
            compile_state = candidate["compile"]
            same_graph = candidate.get("graph") == graph
            same_pid = compile_state.get("pid") in (None, record.pid)
            same_context = compile_state.get("ge_context") in (None, record.context)
            if same_graph and same_pid and same_context:
                return candidate
        return None

    def _store_global_compile_timing(self, event, item):
        phase = event.payload.get("phase")
        if phase:
            self.global_compile_phases[phase].append(item)
            self.global_compile_evidence.append(item)

    def _record_compile_scope(self, state, event, item):
        subtype = event.payload.get("subtype")
        if subtype == "graph_scope":
            state["subgraph_count"] = event.payload["subgraph_count"]
            scope_graph = event.payload["graph"]
            self.graph_scopes.append(
                {
                    "graph": scope_graph,
                    "subgraph_count": event.payload["subgraph_count"],
                    "session_id": state["session_id"],
                    "evidence": item,
                }
            )
            if _is_late_dynamic_scope(scope_graph, event):
                self._promote_dynamic_outer(state, scope_graph, event.record)
            _add_evidence(state, "graph_scope_observed", item)
            return True
        return False

    def _promote_dynamic_outer(self, state, scope_graph, record):
        state.update(
            graph_role="unknown_shape_graph",
            graph_class="dynamic_shape",
            compile_path="dynamic_shape",
            graph_scope="outer_graph",
            stream_scope="graph",
            outer_graph=scope_graph,
            compile_family_id=state["session_id"],
        )
        state.get("phases", {}).pop("graph_build_started", None)
        state["evidence"] = _without_synthetic_build(state.get("evidence", []))
        self.dynamic_outer[self.identity(record)] = state
        self._reparent_dynamic_children(state, scope_graph)

    def _reparent_dynamic_children(self, state, scope_graph):
        prefix = f"{scope_graph}_sub_"
        for workflow in self.workflows:
            child = workflow["compile"]
            if not _is_dynamic_child(child, state, prefix):
                continue
            child.update(
                graph_scope="subgraph",
                stream_scope="submodel",
                outer_graph=scope_graph,
                compile_family_id=state["session_id"],
                session_role="compile_subgraph",
            )
            _merge_dynamic_child(state, child)
        assignments = state["dynamic_current_assignments"].values()
        state["dynamic_logic_stream_ids"] = sorted(
            value for value in assignments if isinstance(value, int) and value >= 0
        )

    def _record_compile_relation(self, state, event, item):
        del state
        subtype = event.payload.get("subtype")
        if subtype == "graph_relation":
            relation = {**event.payload, "evidence": item}
            self.graph_relations.append(relation)
            for candidate in self.workflows:
                if candidate["graph"] == event.payload["child_graph"]:
                    candidate["compile"].update(
                        parent_graph=event.payload["parent_graph"],
                        parent_node=event.payload["parent_node"],
                        subgraph_index=event.payload["subgraph_index"],
                        relation_status="explicit",
                        relation_evidence=[item],
                    )
            return True
        return False

    def _new_runtime_block(
        self, record: LogRecord, backend: str = "unknown"
    ) -> dict[str, Any]:
        identity = self.identity(record)
        self.runtime_sequences[identity] += 1
        state = _new_runtime(
            record.pid, record.context, self.runtime_sequences[identity]
        )
        state["runtime_order"] = record.order
        state["runtime_backend"] = backend
        state["runtime_started_at"] = record.timestamp
        state["runtime_anchor"] = _phase_evidence(record, "runtime evidence")
        self.runtime_blocks.append(state)
        self.active_runtime[identity] = state
        self._attach_pending_context(state, record)
        return state

    def _runtime_block(
        self, record: LogRecord, backend: str, force_new: bool = False
    ) -> dict[str, Any]:
        identity = self.identity(record)
        current = self.active_runtime.get(identity)
        if current is not None and not force_new:
            current_backend = current.get("runtime_backend", "unknown")
            if current_backend in ("unknown", backend) or backend == "unknown":
                current["runtime_last_at"] = record.timestamp
                if current.get("pid") is None:
                    current["pid"] = record.pid
                if current.get("ge_context") is None:
                    current["ge_context"] = record.context
                if current.get("runtime_started_at") is None:
                    current["runtime_started_at"] = record.timestamp
                if current_backend == "unknown" and backend != "unknown":
                    current["runtime_backend"] = backend
                return current
        return self._new_runtime_block(record, backend)

    def _attach_pending_context(
        self, runtime: dict[str, Any], record: LogRecord
    ) -> None:
        for pending in list(self.pending_context):
            if pending.record.pid not in (
                None,
                record.pid,
            ) or pending.record.context not in (None, record.context):
                continue
            delta = _delta(pending.record.timestamp, record.timestamp)
            if delta is not None and not 0 <= delta <= CORRELATION_WINDOW_SECONDS:
                continue
            item = _phase_evidence(
                pending.record, pending.payload.get("phase", "hybrid context")
            )
            marker = {"phase": pending.payload.get("phase"), **item}
            runtime.setdefault("hybrid_context_markers", []).append(marker)
            _add_evidence(runtime, pending.payload.get("phase"), item)
            self.pending_context.remove(pending)

    def _record_runtime(self, event: Event) -> None:
        if self._defer_runtime_name(event):
            return
        force_new = self._runtime_force_new(event)
        backend = _runtime_backend(event.payload.get("subtype"))
        runtime = self._runtime_for_event(event, backend, force_new)
        if runtime is None:
            return
        runtime["runtime_last_at"] = event.record.timestamp
        item = _phase_evidence(event.record, _keyword(event))
        handlers = (
            _record_runtime_anchor,
            self._record_runtime_init,
            _record_runtime_v1,
            _record_runtime_v2_binding,
            _record_runtime_v2_resource,
            _record_runtime_v2_auxiliary,
            _record_runtime_name,
            _record_runtime_operator,
            _record_runtime_batch,
            _record_runtime_context,
        )
        for handler in handlers:
            if handler(runtime, event, item):
                return

    def _defer_runtime_name(self, event):
        subtype = event.payload.get("subtype")
        if subtype not in ("graph_name", "model_name"):
            return False
        current = self.active_runtime.get(self.identity(event.record))
        if current is not None and current.get("model_id") is not None:
            return False
        self.pending_runtime_names.append(event)
        return True

    def _runtime_force_new(self, event):
        subtype = event.payload.get("subtype")
        if subtype == "v2_executor_anchor":
            return True
        if subtype not in ("v2", "v2_collect"):
            return False
        current = self.active_runtime.get(self.identity(event.record))
        if current is None or current.get("runtime_backend") != "v2_rt2":
            return False
        if current.get("executor_anchor_evidence") is not None:
            return False
        index = int(event.payload["logical_index"])
        previous = _runtime_binding_at_index(current, index)
        return bool(
            previous is not None
            and previous.get("rt_stream_id") != event.payload["rt_stream_id"]
        )

    def _runtime_for_event(self, event, backend, force_new):
        record = event.record
        subtype = event.payload.get("subtype")
        current = self.active_runtime.get(self.identity(record))
        if subtype == "context":
            if current is None:
                self.pending_context.append(event)
            return current
        if subtype != "init" or not _reuse_v2_runtime(current):
            return self._runtime_block(record, backend, force_new)
        current["session_id"] = (
            f"model:{record.pid or 'unknown'}:{event.payload['model_id']}"
        )
        return current

    def _record_runtime_init(self, runtime, event, item):
        if event.payload.get("subtype") != "init":
            return False
        record = event.record
        if _same_runtime_init(runtime, event):
            if runtime.get("model_stream_count") == event.payload["stream_count"]:
                runtime["duplicate_init_count"] += 1
                runtime.setdefault("duplicate_init_evidence", []).append(item)
                runtime.setdefault("init_runtime_evidence_all", []).append(item)
                return True
        if (
            runtime.get("model_id") is not None
            and runtime.get("init_runtime_evidence") is not None
        ):
            runtime = self._new_runtime_block(record, "v1_davinci_model")
        runtime.update(
            model_id=event.payload["model_id"],
            init_runtime_at=record.timestamp,
            init_runtime_evidence=item,
            model_stream_count=event.payload["stream_count"],
            notify_count=event.payload["notify_count"],
            event_count=event.payload["event_count"],
            label_count=event.payload["label_count"],
            runtime_anchor=item,
        )
        if runtime.get("runtime_started_at") is None:
            runtime["runtime_started_at"] = record.timestamp
        self._drop_early_runtime_names(record.order)
        runtime.setdefault("init_runtime_evidence_all", []).append(item)
        _add_evidence(runtime, "runtime_params_initialized", item)
        return True

    def _drop_early_runtime_names(self, order):
        retained = []
        for pending in self.pending_runtime_names:
            if pending.record.order > order:
                retained.append(pending)
        self.pending_runtime_names = retained

    def _compile_candidates(self, runtime: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for workflow in self.workflows:
            score = self._candidate_fit_score(workflow, runtime)
            if score is not None:
                candidates.append((score, workflow))
        candidates.sort(
            key=lambda item: (item[0], item[1].get("order", 0)), reverse=True
        )
        return [item[1] for item in candidates]

    def _candidate_fit_score(self, workflow, runtime):
        if workflow.get("runtime") is not None:
            return None
        compile_state = workflow["compile"]
        v2_runtime = runtime.get("runtime_backend") == "v2_rt2"
        if not _candidate_scope_matches(compile_state, runtime, v2_runtime):
            return None
        if not _candidate_has_compile_exit(compile_state, runtime, v2_runtime):
            return None
        if _is_v2_trace_runtime(runtime):
            return _v2_trace_candidate_score(workflow, runtime)
        if v2_runtime:
            return _v2_candidate_score(workflow, runtime)
        return self._v1_candidate_score(workflow, runtime)

    def _v1_candidate_score(self, workflow, runtime):
        compile_state = workflow["compile"]
        if runtime.get("pid") and compile_state.get("pid") not in (
            None,
            runtime.get("pid"),
        ):
            return None
        if runtime.get("ge_context") and compile_state.get("ge_context") not in (
            None,
            runtime.get("ge_context"),
        ):
            return None
        delta = _delta(
            compile_state.get("root_graph_completed_at"),
            runtime.get("runtime_started_at"),
        )
        if delta is not None and not 0 < delta <= CORRELATION_WINDOW_SECONDS:
            return None
        named = runtime.get("model_name") == compile_state.get("graph")
        identity_complete = _candidate_identity_complete(compile_state, runtime)
        timed = delta is not None and delta > 0
        if not named and not (identity_complete and timed):
            return None
        if not named and _candidate_count_mismatch(compile_state, runtime):
            return None
        if not named and identity_complete and timed:
            if self._has_intervening_boundary(compile_state, runtime):
                return None
        return _candidate_score(workflow, runtime)

    def _has_intervening_boundary(
        self, compile_state: dict[str, Any], runtime: dict[str, Any]
    ) -> bool:
        """Return whether another model boundary lies in this workflow gap."""
        compile_at = compile_state.get("root_graph_completed_at")
        runtime_at = runtime.get("runtime_started_at")
        if not compile_at or not runtime_at:
            return True
        total_gap = _delta(compile_at, runtime_at)
        if total_gap is None or total_gap <= 0:
            return True
        if self._compile_boundary_in_gap(compile_state, runtime, compile_at, total_gap):
            return True
        return self._runtime_boundary_in_gap(runtime, compile_at, total_gap)

    def _compile_boundary_in_gap(self, compile_state, runtime, compile_at, total_gap):
        for workflow in self.workflows:
            other = workflow["compile"]
            if other is compile_state:
                continue
            if other.get("graph_scope") != "outer_graph" or _is_subgraph(
                other.get("graph")
            ):
                continue
            if other.get("pid") not in (None, runtime.get("pid")):
                continue
            if other.get("ge_context") not in (None, runtime.get("ge_context")):
                continue
            event_at = other.get("root_graph_completed_at")
            if event_at:
                gap = _delta(compile_at, event_at)
                if gap is not None and 0 < gap < total_gap:
                    return True
        return False

    def _runtime_boundary_in_gap(self, runtime, compile_at, total_gap):
        for other in self.runtime_blocks:
            if other is runtime:
                continue
            if other.get("pid") not in (None, runtime.get("pid")):
                continue
            if other.get("ge_context") not in (None, runtime.get("ge_context")):
                continue
            # A V2 trace/resource block has no InitRuntimeParams boundary;
            # treating every such block as a hard boundary would reject a
            # valid graph when compile and runtime records are interleaved.
            event_at = other.get("init_runtime_at")
            if event_at:
                gap = _delta(compile_at, event_at)
                if gap is not None and 0 < gap < total_gap:
                    return True
        return False

    def _has_ambiguous_prior_compile(
        self, selected: dict[str, Any], runtime: dict[str, Any]
    ) -> bool:
        """Detect an equally plausible older compile with no runtime boundary.

        If several model graphs were compiled before a model-less V2 trace
        (or an InitRuntimeParams line), selecting only the newest graph would
        hide the ambiguity.  A later runtime boundary makes the older graph a
        completed session instead, so it is not treated as a competitor.
        """
        selected_state = selected["compile"]
        selected_at = selected_state.get("root_graph_completed_at")
        runtime_at = runtime.get("runtime_started_at")
        if not selected_at or not runtime_at:
            return False
        selected_count = _compile_counts(selected_state)
        for workflow in self.workflows:
            other = workflow["compile"]
            if other is selected_state:
                continue
            if other.get("graph_scope") != "outer_graph" or _is_subgraph(
                other.get("graph")
            ):
                continue
            if other.get("pid") != selected_state.get("pid") or other.get(
                "ge_context"
            ) != selected_state.get("ge_context"):
                continue
            other_at = other.get("root_graph_completed_at")
            gap = _delta(other_at, runtime_at)
            if gap is None or gap <= 0 or gap > CORRELATION_WINDOW_SECONDS:
                continue
            if selected_count and not (selected_count & _compile_counts(other)):
                continue
            if other_at >= selected_at:
                continue
            # If another runtime started after the older compile, that older
            # graph already has its own runtime boundary and is not ambiguous
            # for the current block.
            has_runtime_boundary = self._runtime_boundary_after_compile(
                runtime, other_at, gap
            )
            if not has_runtime_boundary:
                return True
        return False

    def _runtime_boundary_after_compile(self, runtime, compile_at, total_gap):
        for block in self.runtime_blocks:
            if block is runtime:
                continue
            if block.get("pid") != runtime.get("pid"):
                continue
            if block.get("ge_context") != runtime.get("ge_context"):
                continue
            boundary = _delta(compile_at, block.get("runtime_started_at"))
            if boundary is not None and 0 < boundary < total_gap:
                return True
        return False

    def _pair_runtime(self, runtime: dict[str, Any]) -> None:
        candidates = self._compile_candidates(runtime)
        runtime["correlation_candidates"] = [item["workflow_id"] for item in candidates]
        if not candidates:
            return
        best = _select_runtime_candidate(candidates, runtime)
        if best is None:
            return
        compile_state = best["compile"]
        if runtime.get(
            "runtime_backend"
        ) != "v2_rt2" and self._has_ambiguous_prior_compile(best, runtime):
            self._extend_correlation_candidates(runtime, compile_state)
            return
        if runtime.get("runtime_backend") == "v2_rt2":
            _attach_v2_runtime(best, runtime)
            return
        _attach_v1_runtime(best, runtime)

    def _extend_correlation_candidates(self, runtime, compile_state):
        candidate_ids = list(runtime.get("correlation_candidates", []))
        for item in self.workflows:
            if item["compile"] is not compile_state:
                candidate_ids.append(item["workflow_id"])
        runtime["correlation_candidates"] = list(dict.fromkeys(candidate_ids))

    def _mark_hybrid_static_subgraphs(self):
        for workflow in self.workflows:
            state = workflow["compile"]
            if (
                state.get("graph_scope") != "outer_graph"
                or state.get("graph_class") != "dynamic_shape"
            ):
                continue
            state["hybrid_static_subgraph"] = self._has_static_family_child(state)

    def _has_static_family_child(self, state):
        for item in self.workflows:
            child = item["compile"]
            if child.get("graph_scope") != "subgraph":
                continue
            if child.get("compile_family_id") != state.get("compile_family_id"):
                continue
            if child.get("graph_class") != "static_shape":
                continue
            if not _is_batch_graph(child.get("graph")):
                return True
        return False

    def _pair_runtime_blocks(self):
        for runtime in sorted(
            self.runtime_blocks, key=lambda item: item.get("runtime_order", 0)
        ):
            self._pair_runtime(runtime)

    def _build_sessions(self):
        sessions: list[dict[str, Any]] = []
        paired_runtime = set()
        for workflow in self.workflows:
            runtime = workflow.get("runtime")
            if workflow["compile"].get("report_visibility") == "suppressed":
                continue
            if runtime is not None:
                paired_runtime.add(id(runtime))
                sessions.append(self._model_session(workflow))
            else:
                sessions.append(self._compile_session(workflow))
        for runtime in self.runtime_blocks:
            if id(runtime) not in paired_runtime:
                sessions.append(self._runtime_session(runtime))
        sessions.sort(
            key=lambda item: (item.get("order", 0), item.get("session_id", ""))
        )
        return sessions

    def _prepare_dynamic_subgraph_fallback(self) -> None:
        """Recognize a single-stream runtime aggregate for silent dynamic children."""
        families = self._dynamic_fallback_families()
        for runtime in self.runtime_blocks:
            self._prepare_runtime_dynamic_fallback(runtime, families)

    def _dynamic_fallback_families(self):
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for workflow in self.workflows:
            state = workflow["compile"]
            root = _dynamic_subgraph_root(state.get("graph"))
            missing_dynamic_stream_evidence = (
                root is not None
                and state.get("graph_class") == "dynamic_shape"
                and state.get("dynamic_stream_count") is None
                and not _has_compile_stream_evidence(state)
            )
            if missing_dynamic_stream_evidence:
                families[root].append(workflow)
        return families

    def _prepare_runtime_dynamic_fallback(self, runtime, families):
        if runtime.get("runtime_backend") != "v2_rt2":
            return
        root = runtime.get("executor_graph")
        children = families.get(root)
        exact_roots = self._dynamic_fallback_roots(root)
        if not root or (not children and not exact_roots):
            return
        if runtime.get("v2_binding_conflicts"):
            return
        if _runtime_trace_indexes(runtime) != {0}:
            return
        _apply_dynamic_fallback(runtime, root, children, exact_roots)

    def _dynamic_fallback_roots(self, root):
        matches = []
        for workflow in self.workflows:
            state = workflow["compile"]
            if state.get("graph") != root:
                continue
            if state.get("graph_class") != "dynamic_shape":
                continue
            if state.get("graph_scope") != "outer_graph":
                continue
            if state.get("dynamic_stream_count") is None:
                if not _has_compile_stream_evidence(state):
                    matches.append(workflow)
        return matches

    def _coalesce_v2_runtime_blocks(self) -> None:
        """Join compatible V2 fragments before graph matching.

        A single RT2 execution commonly emits an executor/resource fragment
        and several binding fragments on different threads.  The executor
        graph anchor is therefore allowed to absorb unanchored fragments;
        two executor anchors remain separate model boundaries.
        """
        ordered = sorted(
            self.runtime_blocks, key=lambda item: item.get("runtime_order", 0)
        )
        merged: list[dict[str, Any]] = []
        v2_by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        v2_without_pid: list[dict[str, Any]] = []
        for item in ordered:
            if item.get("runtime_backend") != "v2_rt2":
                merged.append(item)
                continue
            pid = item.get("pid")
            if pid:
                v2_by_pid[pid].append(item)
            else:
                v2_without_pid.append(item)
        for _pid, blocks in v2_by_pid.items():
            merged.extend(_coalesce_v2_pid_blocks(blocks))
        merged.extend(v2_without_pid)
        self.runtime_blocks = sorted(
            merged, key=lambda item: item.get("runtime_order", 0)
        )

    def _compile_states(self) -> Iterable[dict[str, Any]]:
        return (item["compile"] for item in self.workflows)

    def _finalize_compile(self, state: dict[str, Any]) -> None:
        state["streams"] = self._compile_operator_streams(state)
        _populate_compile_stream_placeholders(state)
        _finalize_compile_metadata(state)

    def _compile_operator_streams(self, state):
        streams: dict[str, dict[str, Any]] = {}
        for fact in state.get("operator_facts", {}).values():
            stream_id = int(fact["logic_stream_id"])
            stream = streams.setdefault(
                stream_id, {"operator_count": 0, "operators": [], "batch_ids": {}}
            )
            stream["operator_count"] += 1
            if self.full_ops or len(stream["operators"]) < self.sample_limit:
                stream["operators"].append(fact)
            batch = re.search(r"ascend_mbatch_batch_(\d+)", fact.get("name", ""), re.I)
            if batch:
                stream["batch_ids"][batch.group(1)] = (
                    stream["batch_ids"].get(batch.group(1), 0) + 1
                )
        for stream_id in state.get("dynamic_logic_stream_ids", []):
            streams.setdefault(
                int(stream_id),
                {"operator_count": None, "operators": [], "batch_ids": {}},
            )
        return {
            str(key): value
            for key, value in sorted(streams.items(), key=lambda item: int(item[0]))
        }

    def _model_session(self, workflow: dict[str, Any]) -> dict[str, Any]:
        compile_state = workflow["compile"]
        runtime = workflow["runtime"]
        runtime_path = _runtime_path(compile_state, runtime)
        scenario = _scenario(compile_state, runtime)
        if scenario == "static":
            family = compile_state.get("compile_family_id")
            if any(
                item["compile"].get("compile_family_id") == family
                and item["compile"].get("dynamic_batch_markers")
                for item in self.workflows
            ):
                scenario = "dynamic"
        return {
            "session_id": f"{compile_state['session_id']}+{runtime['session_id']}",
            "order": workflow.get("order", 0),
            "graph": workflow["graph"],
            "model_id": runtime.get("model_id"),
            "session_type": "model",
            "session_role": "model",
            "correlation": workflow.get("correlation", "unknown"),
            "correlation_evidence": workflow.get("correlation_evidence", []),
            "correlation_inference": workflow.get("correlation_inference"),
            "correlation_candidates": workflow.get("correlation_candidates", []),
            "scenario_class": scenario,
            "graph_class": compile_state.get("graph_class", "unknown"),
            "compile_path": compile_state.get("compile_path", "unknown"),
            "runtime_path": runtime_path,
            "stream_scope": compile_state.get("stream_scope", "graph"),
            "compile": compile_state,
            "runtime": runtime,
            "analysis_status": _analysis_status(
                compile_state, runtime, workflow.get("correlation")
            ),
        }
