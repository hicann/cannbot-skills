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
"""Normalize correlated GE workflows into the public report schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from workflow_analyzer import WorkflowAnalyzer
from workflow_correlation import (
    _analysis_status,
    _compile_counts,
    _runtime_count,
    _runtime_path,
)
from workflow_events import (
    CompileSpec,
    _is_subgraph,
    _new_compile,
    _new_runtime,
    iter_records_ordered,
)


def _copy_counter(value: Any) -> dict[str, int]:
    if not value:
        return {}
    return {str(key): int(count) for key, count in dict(value).items()}


def _mapping_rows(
    compile_state: dict[str, Any], runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    compile_ids = sorted(int(key) for key in (compile_state.get("streams") or {}))
    direct = _v1_mapping_rows(runtime, compile_ids)
    _add_v2_mapping_rows(direct, runtime, compile_ids)
    return _complete_mapping_rows(direct, compile_ids)


def _v1_mapping_rows(runtime, compile_ids):
    direct = {}
    for index, binding in (runtime.get("bindings") or {}).items():
        logic_index = int(index)
        direct[logic_index] = {
            "compile_logic_stream_id": logic_index
            if logic_index in compile_ids
            else None,
            "runtime_logic_stream_index": logic_index,
            "rt_stream_id": binding.get("rt_stream_id"),
            "task_num": binding.get("task_num"),
            "evidence": None,
            "evidence_type": "Logical stream index",
        }
    return direct


def _add_v2_mapping_rows(direct, runtime, compile_ids):
    for binding in runtime.get("runtime_logical_to_rt_bindings", []):
        logic_index = binding.get("logical_stream_index")
        if logic_index is None:
            continue
        logic_index = int(logic_index)
        direct[logic_index] = {
            "compile_logic_stream_id": logic_index
            if logic_index in compile_ids
            else None,
            "runtime_logic_stream_index": logic_index,
            "rt_stream_id": binding.get("rt_stream_id"),
            "task_num": None,
            "evidence": {
                "file": binding.get("file"),
                "line": binding.get("line"),
                "keyword": binding.get("keyword", "Get rts stream"),
                "excerpt": binding.get("excerpt", ""),
            },
            "evidence_type": "Get rts stream",
        }


def _complete_mapping_rows(direct, compile_ids):
    rows: list[dict[str, Any]] = []
    for logic_id in compile_ids:
        row = direct.get(logic_id)
        if row is None:
            row = {
                "compile_logic_stream_id": logic_id,
                "runtime_logic_stream_index": None,
                "rt_stream_id": None,
                "task_num": None,
                "evidence": None,
                "evidence_type": "compile-only",
            }
        rows.append(row)
    for logic_index in sorted(set(direct) - set(compile_ids)):
        rows.append(direct[logic_index])
    return rows


def _normalise_compile(
    state: dict[str, Any], analyzer: WorkflowAnalyzer
) -> dict[str, Any]:
    # Convert internal Counters/defaultdicts once at the compatibility edge.
    result = dict(state)
    result["allocation_reasons"] = _copy_counter(state.get("allocation_reasons"))
    streams = {}
    ordered_streams = sorted(
        (state.get("streams") or {}).items(), key=lambda item: int(item[0])
    )
    for key, value in ordered_streams:
        streams[str(key)] = dict(value)
    result["streams"] = streams
    result["dynamic_logic_stream_ids"] = sorted(
        set(state.get("dynamic_logic_stream_ids") or [])
    )
    result["dynamic_initial_logic_stream_ids"] = sorted(
        set(state.get("dynamic_initial_logic_stream_ids") or [])
    )
    result["dynamic_markers"] = {
        "dynamic_shape": 1 if state.get("graph_class") == "dynamic_shape" else 0,
        "dynamic_batch": state.get("dynamic_batch_markers", 0),
        "dynamic_batch_info": sum(
            1
            for item in state.get("phases", {}).get("dynamic_batch_option_observed", [])
        ),
        "batch_operator": sum(
            1 for item in state.get("phases", {}).get("batch_operator", [])
        ),
    }
    result["observed_graphs"] = [state.get("graph")] if state.get("graph") else []
    graph_scopes = []
    for item in analyzer.graph_scopes:
        if item.get("session_id") == state.get("session_id"):
            graph_scopes.append(item)
    result["graph_scopes"] = graph_scopes
    result["graph_relations"] = list(analyzer.graph_relations)
    result["hierarchy_status"] = (
        "explicit"
        if analyzer.graph_relations
        else ("unknown" if result["graph_scopes"] else "unknown")
    )
    result["sync_evidence"] = dict(
        state.get("sync_evidence") or {"send_recv": 0, "event_attrs": 0}
    )
    result["sync_summaries"] = []
    result["log_level_filter"] = "INFO"
    result["files"] = sorted(set(state.get("files") or []))
    return result


def _normalise_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    result = dict(runtime)
    result["bindings"] = dict(runtime.get("bindings") or {})
    result["created_streams"] = list(runtime.get("created_streams") or [])
    bound_ids = {item.get("rt_stream_id") for item in result["bindings"].values()}
    result["auxiliary_streams"] = [
        item
        for item in result["created_streams"]
        if item.get("rt_stream_id") not in bound_ids
    ]
    result["created_stream_count"] = len(result["created_streams"])
    result["bound_stream_count"] = max(
        len(result["bindings"]),
        len(result.get("allocation_bindings") or []),
    )
    result["unbound_stream_count"] = len(result["auxiliary_streams"])
    result["files"] = sorted(set(runtime.get("files") or []))
    result["observed_model_ids"] = sorted(
        {runtime["model_id"]} if runtime.get("model_id") is not None else set()
    )
    result.setdefault("session_role", "runtime_model")
    result["log_level_filter"] = "INFO"
    return result


def _empty_compile_data() -> dict[str, Any]:
    """Return the stable empty compile view used for compile-only scans."""
    state = _new_compile(
        CompileSpec(
            graph="", pid=None, context=None, timestamp=None, sequence=0, shape="known"
        )
    )
    state.update(
        graph=None,
        session_id=None,
        graph_role="unknown",
        graph_class="unknown",
        compile_path="unknown",
        graph_scope="outer_graph",
        session_role="compile_graph",
        outer_graph=None,
        compile_family_id=None,
        window_state="unknown",
        files=[],
        sessions=[],
        observed_graphs=[],
        graph_scopes=[],
        graph_relations=[],
        hierarchy_status="unknown",
        dynamic_markers={},
        log_level_filter="INFO",
    )
    state["allocation_reasons"] = {}
    return state


def _empty_runtime_data() -> dict[str, Any]:
    """Return the stable empty runtime view used for runtime-only scans."""
    state = _new_runtime(None, None, 0)
    state.update(
        session_id=None,
        runtime_order=0,
        sessions=[],
        files=[],
        runtime_backend="unknown",
        compile_runtime_correlation="unknown",
        correlation_evidence=[],
        correlation_inference=None,
        associated_graph=None,
        associated_compile_completed_at=None,
        observed_model_ids=[],
        log_level_filter="INFO",
    )
    return state


def _stage_data(
    analyzer: WorkflowAnalyzer, sessions: list[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    compile_states, suppressed_states = _normalised_compile_states(analyzer)
    runtime_states = [_normalise_runtime(item) for item in analyzer.runtime_blocks]
    runtime_by_source = _sync_runtime_normalisation(analyzer, runtime_states)
    _link_session_runtime_data(sessions, runtime_by_source)
    compile_data = _compile_stage_data(analyzer, compile_states, suppressed_states)
    runtime_data = _runtime_stage_data(
        analyzer, sessions, runtime_states, runtime_by_source
    )
    return compile_data, runtime_data


def _normalised_compile_states(analyzer):
    visible = []
    suppressed = []
    for item in analyzer.workflows:
        state = _normalise_compile(item["compile"], analyzer)
        if item["compile"].get("report_visibility") == "suppressed":
            suppressed.append(state)
        else:
            visible.append(state)
    return visible, suppressed


def _sync_runtime_normalisation(analyzer, runtime_states):
    for source, normalised in zip(analyzer.runtime_blocks, runtime_states):
        source.update(
            {
                "auxiliary_streams": normalised["auxiliary_streams"],
                "created_stream_count": normalised["created_stream_count"],
                "bound_stream_count": normalised["bound_stream_count"],
                "unbound_stream_count": normalised["unbound_stream_count"],
                "files": normalised["files"],
                "log_level_filter": "INFO",
            }
        )
    return {
        id(source): normalised
        for source, normalised in zip(analyzer.runtime_blocks, runtime_states)
    }


def _link_session_runtime_data(sessions, runtime_by_source):
    for session in sessions:
        runtime = session.get("runtime")
        compile_state = session.get("compile")
        if runtime is None or compile_state is None:
            continue
        runtime["mapping_rows"] = _mapping_rows(compile_state, runtime)
        runtime["runtime_logical_to_rt_bindings"] = list(
            runtime.get("runtime_logical_to_rt_bindings") or []
        )
        runtime_count, _count_kind = _runtime_count(runtime)
        consistency = (
            "passed"
            if runtime_count is not None
            and runtime_count in _compile_counts(compile_state)
            else runtime.get("stream_count_consistency", "unknown")
        )
        runtime["stream_count_consistency"] = consistency
        compile_state["runtime_path"] = _runtime_path(compile_state, runtime)
        session["analysis_status"] = _analysis_status(
            compile_state, runtime, session.get("correlation")
        )
        normalised = runtime_by_source.get(id(runtime))
        if normalised is not None:
            _update_normalised_runtime(normalised, runtime, session, compile_state)


def _update_normalised_runtime(normalised, runtime, session, compile_state):
    normalised["mapping_rows"] = list(runtime["mapping_rows"])
    normalised["runtime_logical_to_rt_bindings"] = list(
        runtime.get("runtime_logical_to_rt_bindings") or []
    )
    normalised["stream_count_consistency"] = runtime["stream_count_consistency"]
    normalised["compile_runtime_correlation"] = session.get("correlation", "unknown")
    normalised["correlation_evidence"] = list(session.get("correlation_evidence") or [])
    normalised["correlation_inference"] = session.get("correlation_inference")
    normalised["associated_graph"] = session.get("graph")
    normalised["associated_compile_completed_at"] = compile_state.get(
        "root_graph_completed_at"
    )


def _latest_compile_state(compile_states):
    model_compile = []
    for item in compile_states:
        if item.get("graph_scope") != "outer_graph":
            continue
        if not _is_subgraph(item.get("graph")):
            model_compile.append(item)
    if model_compile:
        return sorted(
            model_compile,
            key=lambda item: (
                item.get("root_graph_completed_at")
                or item.get("graph_started_at")
                or "",
                item.get("session_id", ""),
            ),
        )[-1]
    return compile_states[-1] if compile_states else None


def _compile_stage_data(analyzer, compile_states, suppressed_states):
    latest_compile = _latest_compile_state(compile_states)
    if latest_compile is not None:
        compile_data = dict(latest_compile)
        compile_data["sessions"] = compile_states
        compile_data["graph_scopes"] = list(analyzer.graph_scopes)
        compile_data["graph_relations"] = list(analyzer.graph_relations)
        compile_data["observed_graphs"] = [
            item.get("graph") for item in compile_states if item.get("graph")
        ]
        compile_data["files"] = sorted(analyzer.files)
        compile_data["hierarchy_status"] = _hierarchy_status(analyzer)
        compile_data["dynamic_markers"] = _compile_dynamic_markers(compile_states)
        compile_data["suppressed_compile_sessions"] = _suppressed_sessions(
            suppressed_states
        )
    elif "compile" in analyzer.input_stages:
        compile_data = _empty_compile_data()
        compile_data["graph_scopes"] = list(analyzer.graph_scopes)
        compile_data["graph_relations"] = list(analyzer.graph_relations)
        compile_data["suppressed_compile_sessions"] = _suppressed_sessions(
            suppressed_states
        )
    else:
        compile_data = None
    if compile_data is not None and analyzer.global_compile_phases:
        _merge_global_compile_evidence(analyzer, compile_data)
    return compile_data


def _hierarchy_status(analyzer):
    return "explicit" if analyzer.graph_relations else "unknown"


def _compile_dynamic_markers(compile_states):
    dynamic_shape = 0
    dynamic_batch = 0
    dynamic_batch_info = 0
    batch_operator = 0
    for item in compile_states:
        dynamic_shape += int(item.get("graph_class") == "dynamic_shape")
        dynamic_batch += int(item.get("dynamic_batch_markers") or 0)
        phases = item.get("phases", {})
        dynamic_batch_info += len(phases.get("dynamic_batch_option_observed", []))
        batch_operator += len(phases.get("batch_operator", []))
    return {
        "dynamic_shape": dynamic_shape,
        "dynamic_batch": dynamic_batch,
        "dynamic_batch_info": dynamic_batch_info,
        "batch_operator": batch_operator,
    }


def _suppressed_sessions(states):
    result = []
    for item in states:
        result.append(
            {
                "graph": item.get("graph"),
                "reason": item.get("compile_suppressed_reason"),
                "session_id": item.get("session_id"),
            }
        )
    return result


def _merge_global_compile_evidence(analyzer, compile_data):
    for phase, items in analyzer.global_compile_phases.items():
        compile_data.setdefault("phases", {}).setdefault(phase, []).extend(items)
    compile_data.setdefault("evidence", []).extend(analyzer.global_compile_evidence)


def _runtime_stage_data(analyzer, sessions, runtime_states, runtime_by_source):
    if runtime_states:
        latest_source_runtime = analyzer.runtime_blocks[-1]
        latest_runtime = runtime_by_source[id(latest_source_runtime)]
        runtime_data = dict(latest_runtime)
        runtime_data["sessions"] = runtime_states
        runtime_data["files"] = sorted(analyzer.files)
        backends = _runtime_backends(runtime_states)
        if len(backends) > 1:
            runtime_data["runtime_backend"] = "mixed"
        pair = _runtime_pair(
            sessions, latest_source_runtime, latest_runtime.get("session_id")
        )
        if pair is not None:
            _apply_runtime_pair(runtime_data, pair)
        else:
            _apply_empty_runtime_pair(runtime_data)
        return runtime_data
    return _empty_runtime_data() if "runtime" in analyzer.input_stages else None


def _runtime_backends(runtime_states):
    result = set()
    for item in runtime_states:
        backend = item.get("runtime_backend")
        if backend not in (None, "unknown"):
            result.add(backend)
    return result


def _runtime_pair(sessions, source_runtime, session_id):
    for item in sessions:
        if item.get("runtime") is source_runtime:
            return item
    for item in sessions:
        runtime = item.get("runtime")
        if runtime and runtime.get("session_id") == session_id:
            if item.get("compile") is not None:
                return item
    return None


def _apply_runtime_pair(runtime_data, pair):
    runtime_data["compile_runtime_correlation"] = pair.get("correlation", "unknown")
    runtime_data["correlation_evidence"] = pair.get("correlation_evidence", [])
    runtime_data["correlation_inference"] = pair.get("correlation_inference")
    runtime_data["associated_graph"] = pair.get("graph")
    runtime_data["associated_compile_completed_at"] = (pair.get("compile") or {}).get(
        "root_graph_completed_at"
    )


def _apply_empty_runtime_pair(runtime_data):
    runtime_data.setdefault("compile_runtime_correlation", "unknown")
    runtime_data.setdefault("correlation_evidence", [])
    runtime_data.setdefault("correlation_inference", None)
    runtime_data.setdefault("associated_graph", None)
    runtime_data.setdefault("associated_compile_completed_at", None)


def analyze_workflows(
    logs: dict[str, Sequence[Path]], full_ops: bool = False, sample_limit: int = 5
) -> dict[str, Any]:
    """Analyze all supplied files in one ordered pass.

    ``logs`` may contain ``compile`` and/or ``runtime`` paths.  A path listed
    in both roles is read once; event routing is based on the concrete log
    pattern, so mixed GE INFO log files do not require a caller-side split.
    """
    paths, stages = _ordered_log_paths(logs)
    analyzer = WorkflowAnalyzer(full_ops=full_ops, sample_limit=sample_limit)
    analyzer.input_stages = set(logs)
    analyzer.consume_records(iter_records_ordered(paths, stages))
    sessions = analyzer.finalize()
    _append_model_id_warning(analyzer)
    _append_runtime_stream_warnings(analyzer)
    compile_data, runtime_data = _stage_data(analyzer, sessions)
    return {
        "compile": compile_data,
        "runtime": runtime_data,
        "sessions": sessions,
        "files": sorted(analyzer.files),
        "warnings": list(analyzer.warnings),
    }


def _ordered_log_paths(logs):
    paths = []
    stages = {}
    for stage, stage_paths in logs.items():
        for path in stage_paths:
            path = Path(path)
            if path not in stages:
                paths.append(path)
                stages[path] = stage
            elif stages[path] != "generic":
                stages[path] = "mixed"
    paths.sort(key=str)
    return paths, stages


def _append_model_id_warning(analyzer):
    model_ids = set()
    for item in analyzer.runtime_blocks:
        if item.get("model_id") is not None:
            model_ids.add(item.get("model_id"))
    model_ids = sorted(model_ids)
    if len(model_ids) > 1:
        analyzer.warnings.append(
            f"multiple model ids observed {model_ids}; sessions are kept separate"
        )


def _append_runtime_stream_warnings(analyzer):
    for runtime in analyzer.runtime_blocks:
        limit = runtime.get("model_stream_count")
        if limit is None:
            continue
        invalid = _invalid_runtime_stream_ids(runtime, limit)
        if invalid:
            analyzer.warnings.append(
                f"runtime logic stream ids outside [0,{limit - 1}]: {invalid}"
            )


def _invalid_runtime_stream_ids(runtime, limit):
    invalid = set()
    for item in runtime.get("operator_bindings", []):
        logic_stream_id = item.get("logic_stream_id")
        if logic_stream_id is not None and int(logic_stream_id) >= limit:
            invalid.add(int(logic_stream_id))
    for index in runtime.get("bindings") or {}:
        if int(index) >= limit:
            invalid.add(int(index))
    return sorted(invalid)
