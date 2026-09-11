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

"""Bounded classification helpers for GE stream reports."""

from analyze_stream_logs import (
    _classify_root_graph,
    _compile_count_values,
    _has_batch_compile_markers,
    _runtime_stream_count,
    _select_compile_family,
    is_batch_branch_graph,
)


def _classify_part_01(state):
    compile_data = state.get("compile_data")
    runtime_data = state.get("runtime_data")
    compile_sessions, root_session, family_sessions = _select_compile_family(
        compile_data, runtime_data
    )
    state["compile_sessions"] = compile_sessions
    state["family_sessions"] = family_sessions
    state["root_session"] = root_session


def _classify_part_02(state):
    runtime_data = state.get("runtime_data")
    runtime_family_dynamic = bool(
        runtime_data
        and runtime_data.get("runtime_graph_identity")
        in ("dynamic_subgraph_family", "dynamic_root_compile_missing")
    )
    state["runtime_family_dynamic"] = runtime_family_dynamic


def _classify_part_03(state):
    family_sessions = state.get("family_sessions")
    root_batch_compile = any(
        _has_batch_compile_markers(item) for item in family_sessions
    )
    state["root_batch_compile"] = root_batch_compile


def _classify_part_04(state):
    root_batch_compile = state.get("root_batch_compile")
    state["batch_compile"] = root_batch_compile


def _classify_part_05(state):
    compile_data = state.get("compile_data")
    batch_compile = state.get("batch_compile")
    root_session = state.get("root_session")
    if root_session is None and compile_data:
        batch_compile = bool(
            compile_data.get("dynamic_markers", {}).get("batch_operator")
            or compile_data.get("dynamic_markers", {}).get("dynamic_batch")
        )
    state["batch_compile"] = batch_compile


def _classify_part_06(state):
    runtime_data = state.get("runtime_data")
    batch_runtime = bool(
        runtime_data
        and (
            runtime_data.get("batch_size") is not None
            or runtime_data.get("active_batch_label")
        )
    )
    state["batch_runtime"] = batch_runtime


def _classify_part_07(state):
    family_sessions = state.get("family_sessions")
    runtime_family_dynamic = state.get("runtime_family_dynamic")
    has_dynamic_graph = runtime_family_dynamic or any(
        item.get("graph_class") == "dynamic_shape" for item in family_sessions
    )
    state["has_dynamic_graph"] = has_dynamic_graph


def _classify_part_08(state):
    family_sessions = state.get("family_sessions")
    has_dynamic_child = False
    for item in family_sessions:
        if (
            item.get("graph_scope") == "subgraph"
            and item.get("graph_class") == "dynamic_shape"
        ):
            has_dynamic_child = True
            break
    state["has_dynamic_child"] = has_dynamic_child


def _classify_part_09(state):
    root_session = state.get("root_session")
    raw_root_class = root_session.get("graph_class") if root_session else None
    state["raw_root_class"] = raw_root_class


def _classify_part_10(state):
    family_sessions = state.get("family_sessions")
    has_known_non_batch_child = False
    for item in family_sessions:
        if item.get("graph_scope") != "subgraph":
            continue
        if item.get("graph_class") == "static_shape" and not is_batch_branch_graph(
            item.get("graph")
        ):
            has_known_non_batch_child = True
            break
    state["has_known_non_batch_child"] = has_known_non_batch_child


def _classify_part_11(state):
    has_dynamic_child = state.get("has_dynamic_child")
    has_known_non_batch_child = state.get("has_known_non_batch_child")
    raw_root_class = state.get("raw_root_class")
    root_session = state.get("root_session")
    hybrid = bool(
        root_session
        and (
            root_session.get("hybrid_static_subgraph")
            or (raw_root_class == "dynamic_shape" and has_known_non_batch_child)
            or (raw_root_class == "static_shape" and has_dynamic_child)
        )
    )
    state["hybrid"] = hybrid


def _classify_part_12(state):
    family_sessions = state.get("family_sessions")
    local_dynamic_marker = False
    for item in family_sessions:
        if item.get("graph_class") == "dynamic_shape" or item.get(
            "dynamic_batch_markers"
        ):
            local_dynamic_marker = True
            break
    state["local_dynamic_marker"] = local_dynamic_marker


def _classify_part_13(state):
    batch_compile = state.get("batch_compile")
    batch_runtime = state.get("batch_runtime")
    has_dynamic_graph = state.get("has_dynamic_graph")
    hybrid = state.get("hybrid")
    local_dynamic_marker = state.get("local_dynamic_marker")
    dynamic = (
        batch_compile
        or batch_runtime
        or hybrid
        or has_dynamic_graph
        or local_dynamic_marker
    )
    state["dynamic"] = dynamic


def _classify_part_14(state):
    state["logical_count"] = None


def _classify_part_15(state):
    state["count_scope"] = "unknown"


def _classify_part_16(state):
    compile_data = state.get("compile_data")
    count_scope = state.get("count_scope")
    dynamic = state.get("dynamic")
    logical_count = state.get("logical_count")
    root_session = state.get("root_session")
    if compile_data:
        # Select the count produced by the selected graph's own allocator.
        # RefreshInfoOfDynamicShapeGraph aggregate values are intentionally not
        # candidates here.
        if root_session and root_session.get("graph_class") == "dynamic_shape":
            logical_count = compile_data.get("dynamic_stream_count")
            count_scope = "dynamic" if logical_count is not None else "unknown"
        else:
            logical_count = compile_data.get("logical_stream_count")
            count_scope = "logical" if logical_count is not None else "unknown"
        if logical_count is None and root_session is None:
            logical_count = compile_data.get("logical_stream_count")
            if logical_count is not None:
                count_scope = "logical"
        if logical_count is None and dynamic:
            logical_count = compile_data.get("dynamic_stream_count")
            if logical_count is not None:
                count_scope = "dynamic"
    state["count_scope"] = count_scope
    state["logical_count"] = logical_count


def _classify_part_17(state):
    compile_data = state.get("compile_data")
    graph_phases = compile_data.get("phases", {}) if compile_data else {}
    state["graph_phases"] = graph_phases


def _classify_part_18(state):
    graph_phases = state.get("graph_phases")
    graph_entries = graph_phases.get("graph_build_started", [])
    state["graph_entries"] = graph_entries


def _classify_part_19(state):
    graph_entries = state.get("graph_entries")
    has_known_entry = False
    for item in graph_entries:
        if "known shape graph" in item["keyword"]:
            has_known_entry = True
            break
    state["has_known_entry"] = has_known_entry


def _classify_part_20(state):
    graph_entries = state.get("graph_entries")
    has_unknown_entry = False
    for item in graph_entries:
        if "unknown shape graph" in item["keyword"]:
            has_unknown_entry = True
            break
    state["has_unknown_entry"] = has_unknown_entry


def _classify_part_21(state):
    compile_data = state.get("compile_data")
    dynamic = state.get("dynamic")
    has_known_entry = state.get("has_known_entry")
    has_unknown_entry = state.get("has_unknown_entry")
    if dynamic or has_unknown_entry:
        shape_mode = "dynamic_shape"
    elif has_known_entry or (
        compile_data and compile_data.get("logical_stream_count") is not None
    ):
        shape_mode = "static_shape"
    else:
        shape_mode = "unknown"
    state["shape_mode"] = shape_mode


def _classify_part_22(state):
    batch_compile = state.get("batch_compile")
    batch_runtime = state.get("batch_runtime")
    hybrid = state.get("hybrid")
    if hybrid:
        graph_form = "hybrid_dynamic_static_subgraph"
    elif batch_compile or batch_runtime:
        graph_form = "dynamic_batch_static_branches"
    else:
        graph_form = "single_graph"
    state["graph_form"] = graph_form


def _classify_part_23(state):
    state["root_graph_class"] = "unknown"


def _classify_part_24(state):
    state["classification_evidence"] = []


def _classify_part_25(state):
    compile_data = state.get("compile_data")
    classification_evidence = state.get("classification_evidence")
    compile_sessions = state.get("compile_sessions")
    family_sessions = state.get("family_sessions")
    graph_form = state.get("graph_form")
    root_batch_compile = state.get("root_batch_compile")
    root_evidence = state.get("root_evidence")
    root_graph_class = state.get("root_graph_class")
    root_session = state.get("root_session")
    runtime_family_dynamic = state.get("runtime_family_dynamic")
    if root_session is not None:
        root_graph_class, root_evidence = _classify_root_graph(
            root_session, family_sessions, compile_sessions
        )
        classification_evidence.extend(root_evidence)
        if root_batch_compile:
            graph_form = "dynamic_batch_static_branches"
        if (
            graph_form == "dynamic_batch_static_branches"
            and root_session.get("graph_class") == "static_shape"
            and not any(
                item.get("graph_class") == "dynamic_shape" for item in family_sessions
            )
        ):
            root_graph_class = "pure_static_shape"
            classification_evidence.append("dynamic_batch_known_shape_compute_path")
    elif compile_data and compile_data.get("logical_stream_count") is not None:
        root_graph_class = "static_shape"
        classification_evidence.append("logical_stream_summary_without_graph_entry")
    elif runtime_family_dynamic:
        root_graph_class = "dynamic_shape"
        classification_evidence.append("dynamic_subgraph_family_runtime_aggregate")
    state["graph_form"] = graph_form
    state["root_evidence"] = root_evidence
    state["root_graph_class"] = root_graph_class


def _classify_part_26(state):
    batch_compile = state.get("batch_compile")
    batch_runtime = state.get("batch_runtime")
    graph_form = state.get("graph_form")
    root_batch_compile = state.get("root_batch_compile")
    root_graph_class = state.get("root_graph_class")
    root_session = state.get("root_session")
    shape_mode = state.get("shape_mode")
    if root_session is not None:
        # Global marker counters can contain evidence from an earlier model
        # in the same file.  Once the selected root is known, derive the
        # top-level shape/form from that root and its runtime session only.
        if root_graph_class == "hybrid_dynamic_static_subgraph":
            shape_mode = "dynamic_shape"
            graph_form = "hybrid_dynamic_static_subgraph"
        elif root_graph_class == "pure_dynamic_shape":
            shape_mode = "dynamic_shape"
            graph_form = "single_graph"
        elif root_graph_class == "pure_static_shape":
            # Dynamic Batch is a dynamic model scenario even though each
            # selected Batch_N compute branch is compiled as known-shape.
            # Keep the graph-level compile_path=known_shape, but do not make
            # the model look like an ordinary static model in the overview.
            shape_mode = (
                "dynamic_shape"
                if (batch_compile or batch_runtime or root_batch_compile)
                else "static_shape"
            )
            graph_form = (
                "dynamic_batch_static_branches"
                if (batch_compile or batch_runtime or root_batch_compile)
                else "single_graph"
            )
    state["graph_form"] = graph_form
    state["shape_mode"] = shape_mode


def _classify_part_27(state):
    logical_count = state.get("logical_count")
    stream_policy = (
        "multi_stream"
        if logical_count and logical_count > 1
        else ("single_stream" if logical_count == 1 else "unknown")
    )
    state["stream_policy"] = stream_policy


def _classify_part_28(state):
    logical_count = state.get("logical_count")
    state["final_logical_stream_count"] = logical_count


def _classify_part_29(state):
    compile_data = state.get("compile_data")
    physical = (
        compile_data.get("physical_split", "unknown") if compile_data else "unknown"
    )
    state["physical"] = physical


def _classify_part_30(state):
    graph_form = state.get("graph_form")
    physical = state.get("physical")
    shape_mode = state.get("shape_mode")
    stream_policy = state.get("stream_policy")
    if (
        shape_mode == "dynamic_shape"
        and graph_form == "single_graph"
        and stream_policy == "single_stream"
    ):
        physical = "not_applicable"
    state["physical"] = physical


def _classify_part_31(state):
    state["concrete_compile_phases"] = {
        "graph_build_started",
        "logical_stream_assigned",
        "dynamic_stream_assigned",
        "compile_stream_allocation",
        "physical_stream_finalized",
        "operator_stream_mapping",
    }


def _classify_part_32(state):
    compile_data = state.get("compile_data")
    concrete_compile_phases = state.get("concrete_compile_phases")
    compile_evidence = bool(
        compile_data
        and (
            compile_data.get("logical_stream_count") is not None
            or compile_data.get("dynamic_stream_count") is not None
            or any(
                compile_data.get("phases", {}).get(name)
                for name in concrete_compile_phases
            )
        )
    )
    state["compile_evidence"] = compile_evidence


def _classify_part_33(state):
    runtime_data = state.get("runtime_data")
    runtime_evidence = bool(
        runtime_data
        and (
            runtime_data.get("model_stream_count") is not None
            or runtime_data.get("allocation_bindings")
            or runtime_data.get("runtime_logical_to_rt_bindings")
            or runtime_data.get("phases")
        )
    )
    state["runtime_evidence"] = runtime_evidence


def _classify_part_34(state):
    runtime_data = state.get("runtime_data")
    runtime_count = _runtime_stream_count(runtime_data)
    state["runtime_count"] = runtime_count


def _classify_part_35(state):
    runtime_data = state.get("runtime_data")
    runtime_mapping_rows = (runtime_data or {}).get("mapping_rows", [])
    state["runtime_mapping_rows"] = runtime_mapping_rows


def _classify_part_36(state):
    runtime_mapping_rows = state.get("runtime_mapping_rows")
    runtime_mapping_complete = bool(
        runtime_mapping_rows
        and all(
            row.get("compile_logic_stream_id") is not None
            and row.get("rt_stream_id") is not None
            for row in runtime_mapping_rows
        )
    )
    state["runtime_mapping_complete"] = runtime_mapping_complete


def _classify_part_37(state):
    compile_data = state.get("compile_data")
    runtime_data = state.get("runtime_data")
    logical_count = state.get("logical_count")
    runtime_count = state.get("runtime_count")
    has_compile_runtime_counts = bool(
        compile_data
        and runtime_data
        and logical_count is not None
        and runtime_count is not None
    )
    state["has_compile_runtime_counts"] = has_compile_runtime_counts


def _classify_part_38(state):
    compile_data = state.get("compile_data")
    runtime_data = state.get("runtime_data")
    compile_evidence = state.get("compile_evidence")
    correlation = state.get("correlation")
    has_compile_runtime_counts = state.get("has_compile_runtime_counts")
    runtime_count = state.get("runtime_count")
    runtime_evidence = state.get("runtime_evidence")
    runtime_mapping_complete = state.get("runtime_mapping_complete")
    if not compile_evidence and not runtime_evidence:
        analysis_status = "no_stream_evidence"
    elif has_compile_runtime_counts:
        # Counts can be complete even when the log does not prove that the
        # compile graph and runtime model are the same object.  Keep the
        # overall result partial until that correlation is explicit.
        correlation = runtime_data.get("compile_runtime_correlation", "unknown")
        analysis_status = (
            "complete"
            if (
                correlation in ("strong", "same_session")
                and runtime_count in (_compile_count_values(compile_data))
                and (
                    runtime_data.get("model_stream_count") is not None
                    or runtime_mapping_complete
                )
            )
            else "partial"
        )
    else:
        analysis_status = "partial"
    state["analysis_status"] = analysis_status
    state["correlation"] = correlation


def _classify_part_39(state):
    compile_data = state.get("compile_data")
    runtime_data = state.get("runtime_data")
    logical_count = state.get("logical_count")
    runtime_count = state.get("runtime_count")
    runtime_mapping_complete = state.get("runtime_mapping_complete")
    count_status = (
        "complete"
        if compile_data
        and runtime_data
        and logical_count is not None
        and runtime_count is not None
        and runtime_count in _compile_count_values(compile_data)
        and (
            runtime_data.get("model_stream_count") is not None
            or runtime_mapping_complete
        )
        else "partial"
    )
    state["count_status"] = count_status


def _classify_part_40(state):
    compile_data = state.get("compile_data")
    hierarchy_status = (compile_data or {}).get("hierarchy_status", "unknown")
    state["hierarchy_status"] = hierarchy_status


def _classify_part_41(state):
    runtime_data = state.get("runtime_data")
    correlation_status = (
        runtime_data.get("compile_runtime_correlation", "unknown")
        if runtime_data
        else "unknown"
    )
    state["correlation_status"] = correlation_status


def _classify_part_42(state):
    compile_data = state.get("compile_data")
    runtime_data = state.get("runtime_data")
    same_model_name = bool(
        compile_data
        and runtime_data
        and compile_data.get("graph")
        and runtime_data.get("model_name") == compile_data.get("graph")
    )
    state["same_model_name"] = same_model_name


def _classify_part_43(state):
    correlation_status = state.get("correlation_status")
    same_model_name = state.get("same_model_name")
    if correlation_status == "unknown" and same_model_name:
        correlation_status = "strong"
    state["correlation_status"] = correlation_status


def _classify_part_44(state):
    runtime_data = state.get("runtime_data")
    runtime_backend = (
        runtime_data.get("runtime_backend", "unknown") if runtime_data else "unknown"
    )
    state["runtime_backend"] = runtime_backend


def _classify_part_45(state):
    root_graph_class = state.get("root_graph_class")
    runtime_backend = state.get("runtime_backend")
    if (
        root_graph_class == "hybrid_dynamic_static_subgraph"
        and runtime_backend == "v1_davinci_model"
    ):
        # An unknown-shape outer graph is loaded through HybridDavinciModel;
        # the observed DavinciModel records are its known-shape submodels.
        runtime_backend = "v1_hybrid"
    state["runtime_backend"] = runtime_backend


def _classify_part_46(state):
    dynamic = state.get("dynamic")
    graph_form = state.get("graph_form")
    hybrid = state.get("hybrid")
    root_graph_class = state.get("root_graph_class")
    if root_graph_class == "hybrid_dynamic_static_subgraph" or hybrid:
        scenario_class = "hybrid"
    elif dynamic or graph_form == "dynamic_batch_static_branches":
        scenario_class = "dynamic"
    elif root_graph_class == "pure_static_shape":
        scenario_class = "static"
    else:
        scenario_class = "unknown"
    state["scenario_class"] = scenario_class


def _classify_part_47(state):
    batch_compile = state.get("batch_compile")
    batch_runtime = state.get("batch_runtime")
    classification_evidence = state.get("classification_evidence")
    if batch_compile or batch_runtime:
        classification_evidence.append("dynamic_batch_scenario")


def _classify_part_48(state):
    compile_data = state.get("compile_data")
    runtime_data = state.get("runtime_data")
    state["result"] = {
        "shape_mode": state.get("shape_mode"),
        "scenario_class": state.get("scenario_class"),
        "graph_form": state.get("graph_form"),
        "stream_policy": state.get("stream_policy"),
        "final_logical_stream_count": state.get("final_logical_stream_count"),
        "final_physical_stream_count": (compile_data or {}).get(
            "final_model_stream_count"
        ),
        "operator_mapping": (compile_data or {}).get("operator_mapping", "unknown"),
        "operator_derived_stream_count": (compile_data or {}).get(
            "operator_derived_stream_count"
        ),
        "stream_count_consistency": (compile_data or {}).get(
            "stream_count_consistency", "unknown"
        ),
        "physical_split": state.get("physical"),
        "runtime_backend": state.get("runtime_backend"),
        "analysis_status": state.get("analysis_status"),
        "stream_count_status": state.get("count_status"),
        "stream_count_scope": state.get("count_scope"),
        "graph_hierarchy_status": state.get("hierarchy_status"),
        "compile_runtime_correlation": state.get("correlation_status"),
        "root_graph_class": state.get("root_graph_class"),
        "classification_evidence": state.get("classification_evidence"),
        "graph_count": len(state.get("compile_sessions", [])),
        "model_count": len(
            [
                item
                for item in (runtime_data or {}).get("sessions", [])
                if item.get("model_id") is not None
            ]
        ),
    }


def run_classify(compile_data, runtime_data):
    state = {"compile_data": compile_data, "runtime_data": runtime_data}
    for part in _CLASSIFY_PARTS:
        part(state)
    return state.get("result")


_CLASSIFY_PARTS = (
    _classify_part_01,
    _classify_part_02,
    _classify_part_03,
    _classify_part_04,
    _classify_part_05,
    _classify_part_06,
    _classify_part_07,
    _classify_part_08,
    _classify_part_09,
    _classify_part_10,
    _classify_part_11,
    _classify_part_12,
    _classify_part_13,
    _classify_part_14,
    _classify_part_15,
    _classify_part_16,
    _classify_part_17,
    _classify_part_18,
    _classify_part_19,
    _classify_part_20,
    _classify_part_21,
    _classify_part_22,
    _classify_part_23,
    _classify_part_24,
    _classify_part_25,
    _classify_part_26,
    _classify_part_27,
    _classify_part_28,
    _classify_part_29,
    _classify_part_30,
    _classify_part_31,
    _classify_part_32,
    _classify_part_33,
    _classify_part_34,
    _classify_part_35,
    _classify_part_36,
    _classify_part_37,
    _classify_part_38,
    _classify_part_39,
    _classify_part_40,
    _classify_part_41,
    _classify_part_42,
    _classify_part_43,
    _classify_part_44,
    _classify_part_45,
    _classify_part_46,
    _classify_part_47,
    _classify_part_48,
)
