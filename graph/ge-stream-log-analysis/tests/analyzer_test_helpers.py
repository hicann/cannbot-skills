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
"""Shared fixtures and assertions for the stream-log analyzer tests."""

import os
import tempfile
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_ROOT / "scripts" / "analyze_stream_logs.py"
REPO_ROOT = SKILL_ROOT.parents[1]


def _optional_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


GE_REPO_ROOT = _optional_path("GE_REPO_ROOT")
GE_LOG_ROOT = _optional_path("GE_LOG_ROOT")
if GE_LOG_ROOT is None and GE_REPO_ROOT is not None:
    GE_LOG_ROOT = GE_REPO_ROOT / "log_test"


def external_log(relative_name):
    """Return an explicitly configured external log, or None."""
    if GE_LOG_ROOT is None:
        return None
    candidate = (GE_LOG_ROOT / relative_name).resolve()
    return candidate if candidate.is_file() else None


def analysis_repo_root():
    return GE_REPO_ROOT if GE_REPO_ROOT is not None else REPO_ROOT


def sessions_of_type(data, session_type):
    return [item for item in data["sessions"] if item["session_type"] == session_type]


def _runtime_model(data, model_id):
    for item in data["sessions"]:
        if item["session_type"] == "runtime_model" and item["model_id"] == model_id:
            return item
    raise AssertionError(f"runtime model {model_id} not found")


def _runtime_init_line(stream_count=1):
    return (
        "[INFO] GE(100,python3):2026-08-27-10:00:00.300 [davinci_model.cc:1]2001 "
        f"InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:{stream_count}, "
        "notify_num:0, event_num:0, label_num:0.\n"
    )


def run_compile_and_runtime(test_case, content):
    """Analyze separate compile/runtime files using the common runtime fixture."""
    with tempfile.TemporaryDirectory() as directory:
        compile_log = Path(directory) / "compile.log"
        runtime_log = Path(directory) / "runtime.log"
        compile_log.write_text(content)
        runtime_log.write_text(_runtime_init_line())
        return test_case.run_json(
            "--compile",
            str(compile_log),
            "--runtime",
            str(runtime_log),
            "--repo-root",
            str(REPO_ROOT),
        )


def assert_dynamic_outer_overview(test_case, data):
    test_case.assertEqual(data["compile"]["graph"], "ge_default_20260824202021")
    test_case.assertEqual(data["compile"]["graph_scope"], "outer_graph")
    test_case.assertNotIn("aggregate_stream_count", data["compile"])
    test_case.assertIsNone(data["mode"]["final_logical_stream_count"])
    test_case.assertNotIn("aggregate_stream_count", data["mode"])
    test_case.assertNotIn("ignored_dynamic_aggregate_evidence", data["compile"])
    test_case.assertEqual(data["mode"]["graph_form"], "hybrid_dynamic_static_subgraph")
    test_case.assertEqual(data["mode"]["stream_policy"], "unknown")
    test_case.assertEqual(data["mode"]["runtime_backend"], "v1_hybrid")
    graph_inventory = {item["graph"]: item for item in data["graphs"]}
    test_case.assertEqual(
        graph_inventory["ge_default_20260824202021"]["stream_scope"], "graph"
    )
    test_case.assertEqual(
        graph_inventory["While0cond_control_KSkUzQkkVgg1"]["stream_scope"],
        "submodel",
    )
    test_case.assertEqual(data["runtime"]["model_id"], 2)
    test_case.assertEqual(data["runtime"]["duplicate_init_count"], 1)
    test_case.assertIsNone(data["runtime"]["associated_graph"])
    test_case.assertEqual(data["runtime"]["compile_runtime_correlation"], "unknown")
    test_case.assertIsNone(data["runtime"]["correlation_inference"])


def assert_dynamic_outer_sessions(test_case, data):
    dynamic_compile = None
    for item in data["sessions"]:
        if (
            item["session_type"] == "compile_graph"
            and item["graph"] == "ge_default_20260824202021"
        ):
            dynamic_compile = item
            break
    if dynamic_compile is None:
        raise AssertionError("dynamic compile session not found")
    test_case.assertIsNone(dynamic_compile["model_id"])
    test_case.assertEqual(dynamic_compile["correlation"], "unknown")
    runtime_model = _runtime_model(data, 2)
    test_case.assertIsNone(runtime_model["graph"])
    test_case.assertEqual(runtime_model["analysis_status"], "partial")
    compile_subgraphs = sessions_of_type(data, "compile_graph")
    for item in compile_subgraphs:
        if item["graph"] != "ge_default_20260824202021":
            test_case.assertEqual(item["session_role"], "compile_subgraph")
    test_case.assertIn(
        "ge_default_20260824202021",
        [item["graph"] for item in compile_subgraphs],
    )
    test_case.assertEqual(
        [
            item["line"]
            for item in data["runtime"]["phases"]["runtime_params_initialized"]
        ],
        [59],
    )
    test_case.assertEqual(
        {item["session_type"] for item in data["sessions"]},
        {"model", "compile_graph", "runtime_model"},
    )
    model_sessions = {
        item["model_id"]: item
        for item in data["sessions"]
        if item["session_type"] == "model"
    }
    test_case.assertEqual(set(model_sessions), {1})
    test_case.assertEqual(model_sessions[1]["runtime"]["model_total_stream_count"], 2)
    test_case.assertEqual(data["mode"]["analysis_status"], "partial")
    test_case.assertEqual(data["mode"]["compile_runtime_correlation"], "unknown")


def assert_v2_resource_result(test_case, data):
    models = sessions_of_type(data, "model")
    test_case.assertEqual(len(models), 1)
    test_case.assertEqual(models[0]["graph"], "graph_a")
    test_case.assertEqual(models[0]["correlation"], "strong")
    test_case.assertNotIn("same_pid", models[0]["correlation_evidence"])
    runtime = models[0]["runtime"]
    test_case.assertEqual(len(runtime["v2_resource_summaries"]), 1)
    test_case.assertEqual(len(runtime["v2_resource_duplicate_evidence"]), 1)
    test_case.assertEqual(len(runtime["allocation_bindings"]), 4)
    test_case.assertEqual(len(runtime["execution_bindings"]), 4)
    bindings = [
        (item["logical_stream_index"], item["rt_stream_id"])
        for item in runtime["runtime_logical_to_rt_bindings"]
    ]
    test_case.assertEqual(bindings, [(0, 63), (1, 56), (2, 57), (3, 58)])
    test_case.assertEqual(models[0]["analysis_status"], "complete")


def assert_classification_fixtures(test_case, fixture):
    _assert_dynamic_classification(test_case, fixture)
    _assert_batch_classification(test_case, fixture)
    _assert_v2_classification(test_case, fixture)


def _assert_dynamic_classification(test_case, fixture):
    pure_dynamic = test_case.run_json(
        "--compile",
        str(fixture / "compile_pure_dynamic.log"),
        "--repo-root",
        str(REPO_ROOT),
    )
    test_case.assertEqual(
        pure_dynamic["mode"]["root_graph_class"], "pure_dynamic_shape"
    )
    test_case.assertEqual(pure_dynamic["mode"]["scenario_class"], "dynamic")
    hybrid = test_case.run_json(
        "--compile",
        str(fixture / "compile_hybrid.log"),
        "--repo-root",
        str(REPO_ROOT),
    )
    test_case.assertEqual(
        hybrid["mode"]["root_graph_class"], "hybrid_dynamic_static_subgraph"
    )
    test_case.assertEqual(hybrid["mode"]["scenario_class"], "hybrid")


def _assert_batch_classification(test_case, fixture):
    batch = test_case.run_json(
        "--compile",
        str(fixture / "compile_dynamic_batch_static.log"),
        "--repo-root",
        str(REPO_ROOT),
    )
    test_case.assertEqual(batch["mode"]["graph_form"], "dynamic_batch_static_branches")
    test_case.assertEqual(batch["mode"]["scenario_class"], "dynamic")
    test_case.assertEqual(batch["mode"]["shape_mode"], "dynamic_shape")
    test_case.assertEqual(batch["mode"]["root_graph_class"], "pure_static_shape")
    test_case.assertNotIn("active_branch", batch["batch_mapping"])


def _assert_v2_classification(test_case, fixture):
    v2 = test_case.run_json(
        "--runtime",
        str(fixture / "runtime_v2_kernel_trace.log"),
        "--repo-root",
        str(REPO_ROOT),
    )
    test_case.assertEqual(v2["runtime"]["runtime_backend"], "v2_rt2")
    test_case.assertEqual(len(v2["runtime"]["runtime_logical_to_rt_bindings"]), 2)
    test_case.assertEqual(len(v2["runtime"]["kernel_trace_notifies"]), 1)
    test_case.assertEqual(len(v2["runtime"]["kernel_trace_events"]), 2)
    v2_report = test_case.run_markdown(
        "--runtime",
        str(fixture / "runtime_v2_kernel_trace.log"),
        "--repo-root",
        str(REPO_ROOT),
    )
    test_case.assertIn("not_observed (V2 无专用创建日志)", v2_report)
    test_case.assertNotIn("requested=2", v2_report)
    test_case.assertIn("unknown", v2_report)
    test_case.assertIn("Get rts stream", v2_report)
    test_case.assertNotIn("/ None /", v2_report)
    no_trace = test_case.run_json(
        "--runtime",
        str(fixture / "runtime_v2_no_kernel_trace.log"),
        "--repo-root",
        str(REPO_ROOT),
    )
    test_case.assertIsNone(no_trace["runtime"]["requested_stream_count"])
    test_case.assertEqual(no_trace["runtime"]["runtime_logical_to_rt_bindings"], [])


def normalize_streams(streams):
    normalized = {}
    fields = ("name", "type", "logic_stream_id", "source_kind")
    for stream_id, item in streams.items():
        operators = []
        for operator in item["operators"]:
            operators.append({key: operator.get(key) for key in fields})
        normalized[stream_id] = {
            "operator_count": item["operator_count"],
            "batch_ids": item["batch_ids"],
            "operators": operators,
        }
    return normalized


def normalize_bindings(bindings):
    fields = (
        "rt_stream_id",
        "rt_model_id",
        "priority",
        "flag",
        "task_num",
        "model_stream_index",
        "ge_model_id",
        "role",
    )
    normalized = {}
    for index, item in bindings.items():
        normalized[index] = {key: item.get(key) for key in fields}
    return normalized
