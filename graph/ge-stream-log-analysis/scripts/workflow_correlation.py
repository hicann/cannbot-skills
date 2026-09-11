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
"""Pure helpers for GE compile/runtime correlation decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
    except (TypeError, ValueError):
        return None


def _delta(earlier: Optional[str], later: Optional[str]) -> Optional[float]:
    first = _parse_timestamp(earlier)
    second = _parse_timestamp(later)
    if first is None or second is None:
        return None
    return (second - first).total_seconds()


def _compile_count(compile_state: Optional[dict[str, Any]]) -> Optional[int]:
    if not compile_state:
        return None
    if compile_state.get("graph_class") == "dynamic_shape":
        return compile_state.get("dynamic_stream_count")
    return next(
        (
            compile_state.get(name)
            for name in ("logical_stream_count", "final_model_stream_count")
            if compile_state.get(name) is not None
        ),
        None,
    )


def _compile_counts(compile_state: Optional[dict[str, Any]]) -> set[int]:
    """Return authoritative graph-local counts usable for correlation."""
    if not compile_state:
        return set()
    names = (
        ("dynamic_stream_count",)
        if compile_state.get("graph_class") == "dynamic_shape"
        else ("logical_stream_count", "final_model_stream_count")
    )
    return {
        int(compile_state[name])
        for name in names
        if compile_state.get(name) is not None
    }


def _runtime_count(runtime: Optional[dict[str, Any]]) -> tuple[Optional[int], str]:
    if not runtime:
        return None, "unknown"
    if runtime.get("model_stream_count") is not None:
        return int(runtime["model_stream_count"]), "model"
    # V2's root resource summary is the closest model-level request count.
    if runtime.get("model_total_stream_count") is not None:
        return int(runtime["model_total_stream_count"]), "v2_root"
    requested = runtime.get("requested_stream_count")
    if requested is not None:
        return int(requested), "requested"
    allocation_indexes = {
        int(item["logical_stream_index"])
        for item in runtime.get("allocation_bindings", [])
        if item.get("logical_stream_index") is not None
    }
    if allocation_indexes:
        return len(allocation_indexes), "allocation"
    indexes = {
        int(item["logical_stream_index"])
        for item in runtime.get("runtime_logical_to_rt_bindings", [])
        if item.get("logical_stream_index") is not None
    }
    if indexes:
        # A trace-only V2 block has no request summary.  The number of unique
        # logical indexes is usable as a pairing constraint, but is kept
        # separate from an authoritative requested/model count in evidence.
        return len(indexes), "trace"
    return None, "unknown"


def _is_v2_trace_runtime(runtime: Optional[dict[str, Any]]) -> bool:
    """Return whether runtime identity comes only from legacy V2 traces."""
    return bool(
        runtime
        and runtime.get("runtime_backend") == "v2_rt2"
        and not runtime.get("allocation_bindings")
        and runtime.get("runtime_logical_to_rt_bindings")
    )


def _runtime_trace_indexes(runtime: dict[str, Any]) -> set[int]:
    return {
        int(item["logical_stream_index"])
        for item in runtime.get("runtime_logical_to_rt_bindings", [])
        if item.get("logical_stream_index") is not None
    }


def _runtime_trace_map(runtime: dict[str, Any]) -> dict[int, int]:
    result = {}
    for item in runtime.get("runtime_logical_to_rt_bindings", []):
        logical_index = item.get("logical_stream_index")
        rt_stream_id = item.get("rt_stream_id")
        if logical_index is not None and rt_stream_id is not None:
            result[int(logical_index)] = int(rt_stream_id)
    return result


def _v2_runtime_fragments_compatible(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> bool:
    """Return whether two V2 fragments can describe the same graph runtime."""
    if (
        existing.get("runtime_backend") != "v2_rt2"
        or incoming.get("runtime_backend") != "v2_rt2"
    ):
        return False
    if not existing.get("pid") or existing.get("pid") != incoming.get("pid"):
        return False
    # Two executor anchors are explicit model boundaries.  An anchored block
    # and an unanchored resource/binding fragment, however, are normally parts
    # of the same execution and must be merged.
    if existing.get("executor_anchor_evidence") and incoming.get(
        "executor_anchor_evidence"
    ):
        return False
    for field_name in ("model_id", "executor_graph", "executor_model"):
        if (
            existing.get(field_name) is not None
            and incoming.get(field_name) is not None
            and existing[field_name] != incoming[field_name]
        ):
            return False
    # ``model_name`` is overloaded in legacy logs (it may contain either the
    # graph or the model label).  Once an executor graph anchor exists, do not
    # use this field as a fragment boundary.
    if not existing.get("executor_anchor_evidence") and not incoming.get(
        "executor_anchor_evidence"
    ):
        if (
            existing.get("model_name") is not None
            and incoming.get("model_name") is not None
            and existing["model_name"] != incoming["model_name"]
        ):
            return False
    existing_bindings = _runtime_trace_map(existing)
    incoming_bindings = _runtime_trace_map(incoming)
    return all(
        index not in existing_bindings or existing_bindings[index] == rt_stream_id
        for index, rt_stream_id in incoming_bindings.items()
    )


def _compile_logic_stream_ids(compile_state: dict[str, Any]) -> set[int]:
    ids = {
        int(key)
        for key in (compile_state.get("streams") or {})
        if str(key).lstrip("-").isdigit() and int(key) >= 0
    }
    ids.update(
        int(value)
        for value in compile_state.get("dynamic_logic_stream_ids", [])
        if isinstance(value, int) and value >= 0
    )
    count = _compile_count(compile_state)
    if count is not None:
        ids.update(range(max(0, int(count))))
    return ids


def _candidate_score(workflow: dict[str, Any], runtime: dict[str, Any]) -> int:
    compile_state = workflow["compile"]
    if _is_v2_trace_runtime(runtime):
        return _v2_trace_candidate_score(compile_state, runtime)
    return _standard_candidate_score(workflow, runtime)


def _v2_trace_candidate_score(compile_state, runtime):
    if not runtime.get("pid") or compile_state.get("pid") != runtime.get("pid"):
        return -1
    indexes = _runtime_trace_indexes(runtime)
    compile_indexes = _compile_logic_stream_ids(compile_state)
    if not indexes.issubset(compile_indexes):
        return -1
    delta = _delta(
        compile_state.get("root_graph_completed_at"),
        runtime.get("runtime_started_at"),
    )
    if delta is not None and delta <= 0:
        return -1
    score = 100 + len(indexes) * 10
    if indexes == compile_indexes:
        score += 100
    if delta is not None:
        score += max(0, int(10 - min(delta, 10)))
    return score


def _standard_candidate_score(workflow, runtime):
    compile_state = workflow["compile"]
    score = 0
    if runtime.get("executor_graph") and runtime.get("executor_graph") == workflow.get(
        "graph"
    ):
        score += 200
    if runtime.get("model_name") and runtime.get("model_name") == workflow.get("graph"):
        score += 100
    if runtime.get("pid") and runtime.get("pid") == compile_state.get("pid"):
        score += 20
    if runtime.get("ge_context") and runtime.get("ge_context") == compile_state.get(
        "ge_context"
    ):
        score += 20
    runtime_count, count_kind = _runtime_count(runtime)
    if runtime_count is not None and runtime_count in _compile_counts(compile_state):
        score += 20 if count_kind != "trace" else 10
    delta = _delta(
        compile_state.get("root_graph_completed_at"), runtime.get("runtime_started_at")
    )
    if delta is not None:
        score += max(0, int(10 - delta))
    return score


def _runtime_path(compile_state: dict[str, Any], runtime: dict[str, Any]) -> str:
    backend = runtime.get("runtime_backend", "unknown")
    if backend == "v2_rt2":
        return "v2_rt2"
    if backend == "v1_hybrid":
        return (
            "v1_hybrid_dynamic"
            if compile_state.get("graph_class") == "dynamic_shape"
            else "v1_hybrid_known_submodel"
        )
    if backend == "v1_davinci_model":
        return "v1_davinci"
    return backend


def _scenario(
    compile_state: Optional[dict[str, Any]], runtime: Optional[dict[str, Any]] = None
) -> str:
    compile_state = compile_state or {}
    runtime = runtime or {}
    if compile_state.get("hybrid_static_subgraph"):
        return "hybrid"
    dynamic_markers = (
        compile_state.get("graph_class") == "dynamic_shape",
        compile_state.get("dynamic_batch_markers"),
        runtime.get("batch_size") is not None,
        runtime.get("active_batch_label"),
    )
    if any(dynamic_markers):
        return "dynamic"
    if compile_state.get("graph_class") == "static_shape":
        return "static"
    return "unknown"


def _analysis_status(
    compile_state: dict[str, Any], runtime: dict[str, Any], correlation: Optional[str]
) -> str:
    if correlation not in ("strong", "same_session"):
        return "partial"
    if compile_state.get("root_graph_completed_at") is None:
        return "partial"
    if runtime.get("v2_binding_conflicts") or runtime.get("v2_resource_conflicts"):
        return "partial"
    compile_count = _compile_count(compile_state)
    runtime_count, _ = _runtime_count(runtime)
    if compile_count is None or runtime_count is None:
        return "partial"
    if runtime_count not in _compile_counts(compile_state):
        return "partial"
    rows = runtime.get("mapping_rows", [])
    if rows and all(
        row.get("compile_logic_stream_id") is not None
        and row.get("rt_stream_id") is not None
        for row in rows
    ):
        return "complete"
    # V2 traces may be the only runtime evidence.  If every compile index has
    # a direct trace binding, the graph workflow is complete even without a
    # V1 InitRuntimeParams line.
    if (
        runtime.get("runtime_backend") == "v2_rt2"
        and len(runtime.get("runtime_logical_to_rt_bindings", [])) >= compile_count
    ):
        return "complete"
    return "partial"
