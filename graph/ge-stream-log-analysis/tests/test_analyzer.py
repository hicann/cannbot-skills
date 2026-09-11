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

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analyzer_test_helpers import (
    REPO_ROOT,
    SCRIPT,
    SKILL_ROOT,
    analysis_repo_root as _analysis_repo_root,
    assert_classification_fixtures as _assert_classification_fixtures,
    assert_dynamic_outer_overview as _assert_dynamic_outer_overview,
    assert_dynamic_outer_sessions as _assert_dynamic_outer_sessions,
    assert_v2_resource_result as _assert_v2_resource_result,
    external_log as _external_log,
    normalize_bindings as _normalize_bindings,
    normalize_streams as _normalize_streams,
    run_compile_and_runtime,
    sessions_of_type as _sessions_of_type,
)


class AnalyzerTest(unittest.TestCase):
    def run_json(self, *args, cwd=None):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args, "--format", "json"],
            cwd=cwd or REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def run_markdown(self, *args, cwd=None):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args, "--format", "markdown"],
            cwd=cwd or REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def assert_fact_summary_equal(self, left, right):
        self.assertEqual(left["mode"], right["mode"])
        self.assertEqual(left["compile"]["graph"], right["compile"]["graph"])
        self.assertEqual(
            left["compile"]["logical_stream_count"],
            right["compile"]["logical_stream_count"],
        )

        self.assertEqual(
            _normalize_streams(left["compile"]["streams"]),
            _normalize_streams(right["compile"]["streams"]),
        )

        self.assertEqual(
            _normalize_bindings(left["runtime"]["bindings"]),
            _normalize_bindings(right["runtime"]["bindings"]),
        )
        self.assertEqual(
            [
                (item["session_type"], item.get("graph"), item.get("model_id"))
                for item in left["sessions"]
            ],
            [
                (item["session_type"], item.get("graph"), item.get("model_id"))
                for item in right["sessions"]
            ],
        )

    def test_static_fixture(self):
        fixture = SKILL_ROOT / "tests" / "fixtures"
        data = self.run_json(
            "--compile",
            str(fixture / "compile_static.log"),
            "--runtime",
            str(fixture / "runtime_static.log"),
            "--repo-root",
            str(REPO_ROOT),
        )
        self.assertEqual(data["compile"]["logical_stream_count"], 2)
        self.assertEqual(data["mode"]["physical_split"], "explicit_split")
        self.assertEqual(data["runtime"]["bindings"]["1"]["rt_stream_id"], 11)
        self.assertEqual(data["runtime"]["operator_bindings"][0]["rt_stream_id"], 11)
        self.assertEqual(data["runtime"]["auxiliary_streams"][0]["rt_stream_id"], 12)
        self.assertNotIn("execute_entry_streams", data["runtime"])
        self.assertEqual(data["compile"]["allocation_reasons"]["stream_label"], 1)
        self.assertEqual(data["compile"]["allocation_reasons"]["attached"], 1)
        self.assertEqual(data["compile"]["sync_evidence"]["event_attrs"], 1)
        compile_session = data["compile"]["sessions"][0]
        self.assertTrue(compile_session["compile_stream_allocation_evidence"])

    def test_single_stream_policy_and_operator_type_are_reported(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:343]2001 Begin to build known shape graph[single].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [stream_allocator.cc:598]2001 SetLogicStreamIdAttr:Op [relu] OpType [Relu] logic stream id is 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [logical_stream_allocator.cc:870]2001 At last, root graph: single, total stream num: 1, main stream num: 1, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "single.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
            report = self.run_markdown(
                "--compile", str(log), "--repo-root", str(REPO_ROOT)
            )
        self.assertEqual(data["mode"]["stream_policy"], "single_stream")
        record = next(
            item for item in data["stream_records"] if item.get("graph") == "single"
        )
        self.assertEqual(record["compile"]["logical_stream_count"], 1)
        self.assertEqual(record["compile"]["stream_policy"], "single_stream")
        self.assertEqual(
            record["compile"]["operator_streams"][0]["operators"][0]["type"],
            "Relu",
        )
        self.assertIn("single_stream", report)
        self.assertIn("relu (Relu)", report)

    def test_full_ops_expands_stream_operator_table(self):
        operators = "\n".join(
            f"[INFO] SetLogicStreamIdAttr:Op [op_{index}] OpType [Type{index}] logic stream id is 0."
            for index in range(6)
        )
        content = (
            "[INFO] Begin to build known shape graph[full].\n"
            + operators
            + "\n[INFO] At last, root graph: full, total stream num: 1, main stream num: 1, attached stream num: 0.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "full.log"
            log.write_text(content)
            compact = self.run_markdown(
                "--compile", str(log), "--repo-root", str(REPO_ROOT)
            )
            expanded = self.run_markdown(
                "--compile",
                str(log),
                "--repo-root",
                str(REPO_ROOT),
                "--full-ops",
            )
        self.assertIn("(+1，使用 --full-ops 查看全部)", compact)
        self.assertNotIn("op_5 (Type5)", compact)
        self.assertIn("op_5 (Type5)", expanded)

    def test_stream_records_expose_compile_and_runtime_mapping_facts(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:343]2001 Begin to build known shape graph[report_graph].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [stream_allocator.cc:598]2001 SetLogicStreamIdAttr:Op [a] OpType [Add] logic stream id is 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [stream_allocator.cc:598]2001 SetLogicStreamIdAttr:Op [b] OpType [Add] logic stream id is 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.003 [logical_stream_allocator.cc:870]2001 At last, root graph: report_graph, total stream num: 2, main stream num: 2, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.004 [stream_allocator.cc:677]2001 After SplitStreamAndRefreshTaskDef, graph:report_graph, stream num:2, notify num:0, event num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [model_executor.cc:904]2001 model_id=2, model total stream num:2, model stream num:2, hccl follow stream num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.011 [davinci_model.cc:691]2001 InitRuntimeParams:InitRuntimeParams: model_id=2, stream_num:2, notify_num:0, event_num:0, label_num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.012 [reusable_stream_allocator.cc:35]2001 Create new stream: 0x1, rt stream id: 45, rt model id: 2, priority: 0, stream flag: 1, task num: 2.
[INFO] GE(100,python3):2026-08-27-10:00:00.013 [davinci_model.cc:1484]2001 Logical stream index: 0, rtstream: 45, model: 2, stream flag: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.014 [reusable_stream_allocator.cc:35]2001 Create new stream: 0x2, rt stream id: 44, rt model id: 2, priority: 0, stream flag: 1, task num: 3.
[INFO] GE(100,python3):2026-08-27-10:00:00.015 [davinci_model.cc:1484]2001 Logical stream index: 1, rtstream: 44, model: 2, stream flag: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "paired.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        record = next(
            item
            for item in data["stream_records"]
            if item.get("graph") == "report_graph"
        )
        self.assertEqual(record["compile"]["path"], "known_shape")
        self.assertEqual(record["compile"]["logical_stream_ids"], [0, 1])
        self.assertEqual(record["compile"]["logical_stream_count"], 2)
        self.assertEqual(record["compile"]["stream_policy"], "multi_stream")
        self.assertEqual(
            [
                (item["logic_stream_id"], item["operators"][0]["type"])
                for item in record["compile"]["operator_streams"]
            ],
            [(0, "Add"), (1, "Add")],
        )
        self.assertEqual(record["mapping_status"], "complete")
        self.assertEqual(
            [
                (item["compile_logic_stream_id"], item["rt_stream_id"])
                for item in record["runtime"]["mapping_rows"]
            ],
            [(0, 45), (1, 44)],
        )
        self.assertEqual(len(record["runtime"]["bindings"]), 2)
        self.assertEqual(record["runtime"]["created_count"], 2)
        self.assertIsNotNone(record["compile"]["entry"])
        self.assertEqual(
            [item["kind"] for item in record["compile"]["exits"]],
            ["logical", "physical"],
        )

    def test_dynamic_allocator_info_is_compile_allocation_evidence(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]2001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dyn].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [dynamic_stream_allocator.cc:180]2001 AssignStreamForSubgraph:Assign stream_id: 0 for engine: AIcoreEngine, subgraph: dyn_part0.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [dynamic_stream_allocator.cc:414]2001 SetSubgraphStreamToNodes:[Assign][StreamId] 0 for Subgraph dyn_part0 (engine: AIcoreEngine).
[INFO] GE(100,python3):2026-08-27-10:00:00.003 [dynamic_stream_allocator.cc:180]2001 AssignStreamForSubgraph:Assign stream_id: 1 for engine: AICPU, subgraph: dyn_part1.
[INFO] GE(100,python3):2026-08-27-10:00:00.004 [dynamic_stream_allocator.cc:414]2001 SetSubgraphStreamToNodes:[Assign][StreamId] 1 for Subgraph dyn_part1 (engine: AICPU).
[INFO] GE(100,python3):2026-08-27-10:00:00.005 [dynamic_stream_allocator.cc:461]2001 ReassignStreamByStreamLabel:Node: dyn_node, stream_label: label, reassign stream id: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.006 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: dyn, stream num: 2, event num: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
            report = self.run_markdown(
                "--compile", str(log), "--repo-root", str(REPO_ROOT)
            )
        session = next(
            item for item in data["compile"]["sessions"] if item["graph"] == "dyn"
        )
        self.assertEqual(session["dynamic_logic_stream_ids"], [0, 1])
        self.assertEqual(session["dynamic_initial_logic_stream_ids"], [0, 1])
        self.assertEqual(len(session["dynamic_stream_assignment_evidence"]), 5)
        record = next(
            item for item in data["stream_records"] if item.get("graph") == "dyn"
        )
        self.assertEqual(record["compile"]["logical_stream_ids"], [0, 1])
        self.assertEqual(len(record["compile"]["allocation"]), 5)
        self.assertIn("编译分配入口/过程", report)
        self.assertIn("Assign stream_id", report)

    def test_attached_stream_ids_keep_all_operator_ids(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:343]2001 BuildForKnownShapeGraph:Begin to build known shape graph[attached].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [stream_allocator.cc:598]2001 SetLogicStreamIdAttr:Op [hccl] OpType [HcomAllReduce] logic stream id is 0, logic attached stream id is 2 3.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [logical_stream_allocator.cc:870]2001 At last, root graph: attached, total stream num: 4, main stream num: 2, attached stream num: 2.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "attached.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        mappings = data["compile"]["sessions"][0]["attached_operator_mappings"]
        self.assertEqual(mappings[0]["logic_attached_stream_ids"], [2, 3])

    def test_dynamic_reassignment_replaces_stale_node_id_but_keeps_other_owner(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]2001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dyn].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [dynamic_stream_allocator.cc:180]2001 AssignStreamForSubgraph:Assign stream_id: 0 for engine: AIcoreEngine, subgraph: part0.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [dynamic_stream_allocator.cc:414]2001 SetSubgraphStreamToNodes:[Assign][StreamId] 0 for Subgraph part0 (engine: AIcoreEngine).
[INFO] GE(100,python3):2026-08-27-10:00:00.003 [dynamic_stream_allocator.cc:461]2001 ReassignStreamByStreamLabel:Node: dyn_node, stream_label: label, reassign stream id: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.004 [dynamic_stream_allocator.cc:461]2001 ReassignStreamByStreamLabel:Node: dyn_node, stream_label: label, reassign stream id: 2.
[INFO] GE(100,python3):2026-08-27-10:00:00.005 [dynamic_stream_allocator.cc:640]2001 RefreshStreamsForGraphByNodeIds:Refresh stream by node ids of graph: dyn, stream_id: 3, type: Add, name: dyn_node.
[INFO] GE(100,python3):2026-08-27-10:00:00.006 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: dyn, stream num: 2, event num: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_reassign.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        session = next(
            item for item in data["compile"]["sessions"] if item["graph"] == "dyn"
        )
        self.assertEqual(session["dynamic_logic_stream_ids"], [0, 3])
        self.assertNotIn(1, session["dynamic_logic_stream_ids"])
        self.assertNotIn(2, session["dynamic_logic_stream_ids"])
        self.assertEqual(session["dynamic_current_assignments"]["node:dyn_node"], 3)
        self.assertEqual(len(session["dynamic_assignment_history"]), 5)

    def test_dynamic_summary_requires_plural_allocator_function_and_exact_message(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [dynamic_stream_allocator.cc:73]2001 AssignStreamForDynamicShapeGraph:Graph: singular, stream num: 3, event num: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [dynamic_stream_allocator.cc:73]2001 SomeOtherFunction:AssignStreamsForDynamicShapeGraph:Graph: ghost, stream num: 4, event num: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: real, stream num: 2, event num: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_summary_strict.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        graphs = {item["graph"]: item for item in data["compile"]["sessions"]}
        self.assertNotIn("singular", graphs)
        self.assertNotIn("ghost", graphs)
        self.assertEqual(graphs["real"]["dynamic_stream_count"], 2)

    def test_dynamic_assignment_before_late_outer_scope_is_reparented(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]2001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[outer_dynamic_sub_0_unknow].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [dynamic_stream_allocator.cc:180]2001 AssignStreamForSubgraph:Assign stream_id: 0 for engine: AIcoreEngine, subgraph: part0.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [dynamic_stream_allocator.cc:414]2001 SetSubgraphStreamToNodes:[Assign][StreamId] 0 for Subgraph part0 (engine: AIcoreEngine).
[INFO] GE(100,python3):2026-08-27-10:00:00.003 [assign_attached_stream_pass.cc:35]2001 Start assign attached stream for graph outer_dynamic with subgraph num: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.004 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: outer_dynamic, stream num: 1, event num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "late_outer.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertNotIn(
            "outer_dynamic",
            {item["graph"] for item in data["compile"]["sessions"]},
        )
        subgraph = next(
            item
            for item in data["compile"]["sessions"]
            if item["graph"] == "outer_dynamic_sub_0_unknow"
        )
        self.assertEqual(subgraph["dynamic_logic_stream_ids"], [0])

    def test_markdown_does_not_render_non_stream_sections(self):
        fixture = SKILL_ROOT / "tests" / "fixtures"
        report = self.run_markdown(
            "--compile",
            str(fixture / "compile_static.log"),
            "--runtime",
            str(fixture / "runtime_static.log"),
            "--repo-root",
            str(REPO_ROOT),
        )
        self.assertIn("编译入口", report)
        self.assertIn("运行创建", report)
        self.assertIn("逻辑流 → RT 流", report)
        self.assertIn("编译图记录/运行记录：", report)
        self.assertIn("最终逻辑流数", report)
        self.assertIn("流策略（单流/多流）", report)
        self.assertIn("## 逻辑流对应算子", report)
        self.assertIn("data (Data)", report)
        self.assertIn("conv (Conv2D)", report)
        self.assertNotIn("图层级/父子关系", report)
        self.assertNotIn("父图/父节点", report)
        self.assertNotIn("动态 Batch 映射", report)

    def test_single_model_markdown_is_compact_by_default(self):
        fixture = SKILL_ROOT / "tests" / "fixtures"
        report = self.run_markdown(
            "--compile",
            str(fixture / "compile_static.log"),
            "--runtime",
            str(fixture / "runtime_static.log"),
            "--repo-root",
            str(REPO_ROOT),
        )
        self.assertIn("## 图级流证据", report)
        self.assertIn("## 逻辑流对应算子", report)
        self.assertIn("## 逻辑流 → RT 流", report)
        self.assertNotIn("## 图层级/父子关系", report)
        self.assertNotIn("### 模型会话流明细", report)

    def test_verbose_adds_evidence_excerpts(self):
        fixture = SKILL_ROOT / "tests" / "fixtures"
        report = self.run_markdown(
            "--compile",
            str(fixture / "compile_static.log"),
            "--runtime",
            str(fixture / "runtime_static.log"),
            "--repo-root",
            str(REPO_ROOT),
            "--verbose",
        )
        self.assertIn("<small>[INFO]", report)
        self.assertIn("## 逻辑流 → RT 流", report)
        self.assertNotIn("图层级/父子关系", report)

    def test_multiple_model_markdown_keeps_each_session_mapping(self):
        content = (
            "[INFO] InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, "
            "notify_num:0, event_num:0, label_num:0.\n"
            "[INFO] InitRuntimeParams:InitRuntimeParams: model_id=2, stream_num:2, "
            "notify_num:0, event_num:0, label_num:0.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime.log"
            log.write_text(content)
            report = self.run_markdown(
                "--runtime", str(log), "--repo-root", str(REPO_ROOT)
            )
        self.assertIn("## 图级流证据", report)
        self.assertIn("model_id=1 (runtime-only)", report)
        self.assertIn("model_id=2 (runtime-only)", report)
        self.assertNotIn("模型会话流明细", report)

    def test_rts_delegated_split(self):
        log = SKILL_ROOT / "tests/fixtures/compile_delegated.log"
        data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["physical_split"], "delegated_to_rts")

    def test_missing_ge_context_runs_in_log_only_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--compile",
                    str(SKILL_ROOT / "tests/fixtures/compile_static.log"),
                    "--repo-root",
                    directory,
                    "--format",
                    "json",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["context"]["analysis_mode"], "log_only")
        self.assertTrue(
            any("GE repository context" in item for item in data["warnings"])
        )

    def test_ge_context_only_checks_directory_markers(self):
        """A valid GE root enables context mode without reading source content."""
        required_paths = (
            "compiler/graph/build/stream",
            "runtime",
            "docs/zh/design/features/stream_allocator.md",
            "docs/zh/design/constraints/stream_allocator.md",
        )
        with tempfile.TemporaryDirectory() as directory:
            ge_root = Path(directory)
            for relative in required_paths:
                marker = ge_root / relative
                marker.parent.mkdir(parents=True, exist_ok=True)
                if marker.suffix:
                    marker.write_text("marker\n")
                else:
                    marker.mkdir(exist_ok=True)
            data = self.run_json(
                "--compile",
                str(SKILL_ROOT / "tests/fixtures/compile_static.log"),
                "--repo-root",
                str(ge_root),
            )
        self.assertEqual(data["context"]["analysis_mode"], "repo_verified")
        self.assertEqual(data["context"]["repo_root"], str(ge_root))

    def test_missing_log(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--compile",
                "/does/not/exist",
                "--repo-root",
                str(REPO_ROOT),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 4)

    def test_multiple_models_and_invalid_logic_id(self):
        content = """\
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 10, rt model id: 20, priority: 0, stream flag: 1, task num: 3.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 10, model: 1, stream flag: 1.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=2, stream_num:2, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x2, rt stream id: 11, rt model id: 21, priority: 0, stream flag: 1, task num: 4.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 11, model: 2, stream flag: 1.
[INFO] KernelTaskInfo Init Success, logic stream id: 4294967295, stream: (nil).
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["model_id"], 2)
        self.assertEqual(data["runtime"]["bindings"]["0"]["rt_stream_id"], 11)
        self.assertNotIn(
            10, [item["rt_stream_id"] for item in data["runtime"]["created_streams"]]
        )
        self.assertTrue(any("multiple model ids" in item for item in data["warnings"]))
        self.assertTrue(any("4294967295" in item for item in data["warnings"]))
        self.assertEqual(len(data["runtime"]["sessions"]), 2)
        self.assertEqual(
            {item["model_id"] for item in data["runtime"]["sessions"]}, {1, 2}
        )
        sessions = {item["model_id"]: item for item in data["runtime"]["sessions"]}
        self.assertEqual(set(sessions[1]["bindings"]), {"0"})
        self.assertEqual(set(sessions[2]["bindings"]), {"0"})
        self.assertTrue(
            all(
                "model_id=2" not in item["excerpt"]
                for item in sessions[1]["evidence"]
                if item["keyword"] == "InitRuntimeParams"
            )
        )

    def test_reordered_model_blocks_keep_independent_bindings(self):
        content = """\
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=2, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x2, rt stream id: 22, rt model id: 2, priority: 0, stream flag: 1, task num: 4.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 22, model: 2, stream flag: 1.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 21, rt model id: 1, priority: 0, stream flag: 1, task num: 3.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 21, model: 1, stream flag: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "reordered_runtime.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        sessions = {item["model_id"]: item for item in data["runtime"]["sessions"]}
        self.assertEqual(sessions[1]["bindings"]["0"]["rt_stream_id"], 21)
        self.assertEqual(sessions[2]["bindings"]["0"]["rt_stream_id"], 22)

    def test_rotated_compile_logs_and_phase_boundaries(self):
        fixture = SKILL_ROOT / "tests" / "fixtures"
        data = self.run_json(
            "--compile",
            str(fixture / "compile_static.log"),
            "--compile",
            str(fixture / "compile_delegated.log"),
            "--repo-root",
            str(REPO_ROOT),
        )
        self.assertEqual(len(data["compile"]["files"]), 2)
        self.assertIn("logical_stream_assigned", data["compile"]["phases"])
        self.assertIn("physical_stream_finalized", data["compile"]["phases"])

    def test_v2_source_acquire_name_without_ge_log_is_ignored(self):
        content = "[INFO] ModelV2Executor::OccupyStreamResource AcquireStreams(stream_num=4)\n"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["runtime_backend"], "unknown")
        self.assertEqual(data["mode"]["analysis_status"], "no_stream_evidence")
        self.assertEqual(data["runtime"]["sessions"], [])
        self.assertIsNone(data["runtime"]["requested_stream_count"])

    def test_source_function_names_are_not_runtime_log_evidence(self):
        content = """\
[INFO] LoweringAndSplitRtStreams
[INFO] SplitRtStreams
[INFO] ModelV2Executor::OccupyStreamResource
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "source_names_only.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["runtime_backend"], "unknown")
        self.assertEqual(data["runtime"]["runtime_logical_to_rt_bindings"], [])
        self.assertIsNone(data["runtime"]["requested_stream_count"])

    def test_mixed_v1_v2_logs_keep_backend_per_runtime_session(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.010 ModelV2Executor OccupyStreamResource AcquireStreams(stream_num=2)
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [KernelTrace][SplitRtStreams] Get rts stream 0x111 from logical stream 0, rts_stream_id: 41
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed_runtime.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        sessions = data["runtime"]["sessions"]
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["runtime_backend"], "v1_davinci_model")
        self.assertEqual(sessions[1]["runtime_backend"], "v2_rt2")
        self.assertEqual(data["runtime"]["runtime_backend"], "mixed")

    def test_reordered_model_blocks_do_not_cross_contaminate(self):
        content = """\\
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=2, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x2, rt stream id: 22, rt model id: 2, priority: 0, stream flag: 1, task num: 2.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 22, model: 2, stream flag: 1.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 11, rt model id: 1, priority: 0, stream flag: 1, task num: 3.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 11, model: 1, stream flag: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "reordered.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        sessions = {item["model_id"]: item for item in data["runtime"]["sessions"]}
        self.assertEqual(sessions[2]["bindings"]["0"]["rt_stream_id"], 22)
        self.assertEqual(sessions[1]["bindings"]["0"]["rt_stream_id"], 11)

    def test_v2_fields_are_not_populated_by_compile_attached_summary(self):
        content = "[INFO] At last, root graph: g, total stream num: 2, main stream num: 2, attached stream num: 0.\n"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["runtime_backend"], "unknown")
        self.assertIsNone(data["runtime"]["attached_stream_count"])

    def test_timing_records_are_not_reported_as_function_start(self):
        content = (
            "[INFO] BuildModelForGetTask:[GEPERFTRACE] The time cost of "
            "GraphBuilder::AssignLogicalStreams is [12] micro seconds.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertNotIn("task_build_started", data["compile"]["phases"])
        self.assertIn("logical_stream_assignment_timing_end", data["compile"]["phases"])

    def test_no_stream_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "unrelated.log"
            log.write_text("[INFO] unrelated message\n")
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["analysis_status"], "no_stream_evidence")
        self.assertTrue(any("no stream evidence" in item for item in data["warnings"]))

    def test_info_only_filter_ignores_debug_fact(self):
        content = """\
[DEBUG] At last, root graph: debug_graph, total stream num: 99, main stream num: 99, attached stream num: 0.
[INFO] At last, root graph: info_graph, total stream num: 2, main stream num: 2, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["compile"]["logical_stream_count"], 2)
        self.assertEqual(data["context"]["log_level_filter"], "INFO")
        self.assertEqual(data["mode"]["analysis_status"], "partial")

    def test_debug_only_stream_fact_is_not_analyzed(self):
        content = (
            "[DEBUG] At last, root graph: debug_graph, total stream num: 99, "
            "main stream num: 99, attached stream num: 0.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "debug.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertIsNone(data["compile"]["logical_stream_count"])
        self.assertEqual(data["mode"]["analysis_status"], "no_stream_evidence")

    def test_operator_mapping_unknown_when_info_mapping_absent(self):
        content = "[INFO] At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.\n"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["compile"]["operator_mapping"], "unknown")

    def test_generic_log_dir_routes_mixed_file_to_both_phases(self):
        content = """\
[INFO] Assign:[Assign][LogicalStream] At last, root graph: mixed, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 21, rt model id: 7, priority: 0, stream flag: 1, task num: 2.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 21, model: 7, stream flag: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "logs"
            log_dir.mkdir()
            (log_dir / "mixed.log.0").write_text(content)
            (log_dir / "unrelated.log.1").write_text("[INFO] unrelated message\n")
            data = self.run_json(
                "--log-dir", str(log_dir), "--repo-root", str(REPO_ROOT)
            )
        self.assertEqual(data["context"]["input_mode"], "generic")
        self.assertEqual(data["compile"]["logical_stream_count"], 1)
        self.assertEqual(data["runtime"]["bindings"]["0"]["rt_stream_id"], 21)

    def test_unrelated_info_lines_do_not_change_fact_summary(self):
        base = """\\
[INFO] Begin to build known shape graph[g].
[INFO] SetLogicStreamIdAttr:Op [a] OpType [Add] logic stream id is 0,
[INFO] At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 21, rt model id: 7, priority: 0, stream flag: 1, task num: 2.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 21, model: 7, stream flag: 1.
"""
        noisy = (
            "[INFO] unrelated compiler diagnostic\n[INFO] another unrelated line\n"
            + base
        )
        with tempfile.TemporaryDirectory() as directory:
            clean_log = Path(directory) / "clean.log"
            noisy_log = Path(directory) / "noisy.log"
            clean_log.write_text(base)
            noisy_log.write_text(noisy)
            clean = self.run_json(
                "--log", str(clean_log), "--repo-root", str(REPO_ROOT)
            )
            noisy_data = self.run_json(
                "--log", str(noisy_log), "--repo-root", str(REPO_ROOT)
            )
        self.assert_fact_summary_equal(clean, noisy_data)

    def test_split_compile_inputs_match_single_file_facts(self):
        first = """\\
[INFO] Begin to build known shape graph[g].
[INFO] SetLogicStreamIdAttr:Op [a] OpType [Add] logic stream id is 0,
"""
        second = """\\
[INFO] At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            first_log = Path(directory) / "compile.log.0"
            second_log = Path(directory) / "compile.log.1"
            single_log = Path(directory) / "compile.single.log"
            first_log.write_text(first)
            second_log.write_text(second)
            single_log.write_text(first + second)
            split = self.run_json(
                "--compile",
                str(first_log),
                "--compile",
                str(second_log),
                "--repo-root",
                str(REPO_ROOT),
            )
            single = self.run_json(
                "--compile", str(single_log), "--repo-root", str(REPO_ROOT)
            )
        self.assertEqual(split["compile"]["graph"], single["compile"]["graph"])
        self.assertEqual(
            split["compile"]["logical_stream_count"],
            single["compile"]["logical_stream_count"],
        )
        for stream_id in split["compile"]["streams"]:
            split_ops = split["compile"]["streams"][stream_id]["operators"]
            single_ops = single["compile"]["streams"][stream_id]["operators"]
            self.assertEqual(
                [
                    (item["name"], item["type"], item["logic_stream_id"])
                    for item in split_ops
                ],
                [
                    (item["name"], item["type"], item["logic_stream_id"])
                    for item in single_ops
                ],
            )

    def test_reordered_pid_model_blocks_do_not_cross_bind(self):
        compile_blocks = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[g100].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [stream_allocator.cc:1]2001 At last, root graph: g100, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(200,python3):2026-08-27-10:01:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[g200].
[INFO] GE(200,python3):2026-08-27-10:01:00.010 [stream_allocator.cc:1]2001 At last, root graph: g200, total stream num: 2, main stream num: 2, attached stream num: 0.
"""
        runtime_blocks = """\\
[INFO] GE(200,python3):2026-08-27-10:01:00.020 [davinci_model.cc:1]2001 InitRuntimeParams:InitRuntimeParams: model_id=2, stream_num:2, notify_num:0, event_num:0, label_num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [davinci_model.cc:1]2001 InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            compile_log = Path(directory) / "compile.log"
            runtime_log = Path(directory) / "runtime.log"
            compile_log.write_text(compile_blocks)
            runtime_log.write_text(runtime_blocks)
            data = self.run_json(
                "--compile",
                str(compile_log),
                "--runtime",
                str(runtime_log),
                "--repo-root",
                str(REPO_ROOT),
            )
        paired = {
            item["model_id"]: item["graph"]
            for item in data["sessions"]
            if item["session_type"] == "model"
        }
        self.assertEqual(paired, {1: "g100", 2: "g200"})

    def test_top_level_classification_uses_selected_latest_root_not_global_markers(
        self,
    ):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build unknown shape graph[dynamic_model].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [graph_builder.cc:1]2001 RefreshInfoOfDynamicShapeGraph:Total stream num: 2, event num: 1, notify num: 0.
[INFO] GE(200,python3):2026-08-27-10:01:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[static_model].
[INFO] GE(200,python3):2026-08-27-10:01:00.010 [stream_allocator.cc:1]2001 At last, root graph: static_model, total stream num: 1, main stream num: 1, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "multi_root.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["compile"]["graph"], "static_model")
        self.assertEqual(data["mode"]["root_graph_class"], "pure_static_shape")
        self.assertEqual(data["mode"]["shape_mode"], "static_shape")
        self.assertEqual(data["mode"]["graph_form"], "single_graph")

    def test_compile_sessions_keep_multiple_graphs(self):
        content = """\
[INFO] Begin to build known shape graph[graph_a].
[INFO] Assign:[Assign][LogicalStream] At last, root graph: graph_a, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] Begin to build known shape graph[graph_b].
[INFO] Assign:[Assign][LogicalStream] At last, root graph: graph_b, total stream num: 2, main stream num: 2, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            {item["graph"] for item in data["compile"]["sessions"]},
            {"graph_a", "graph_b"},
        )
        self.assertEqual(len(data["sessions"]), 2)
        self.assertTrue(
            all(
                item.get("session_type") == "compile_graph" for item in data["sessions"]
            )
        )

    def test_compile_graphs_are_not_reported_as_models(self):
        content = """\
[INFO] Begin to build known shape graph[root].
[INFO] At last, root graph: root, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] Begin to build unknown shape graph[root_subgraph].
[INFO] At last, root graph: root_subgraph, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            {item.get("session_type") for item in data["sessions"]},
            {"compile_graph", "runtime_model"},
        )
        self.assertNotIn(
            "root_subgraph",
            [
                item.get("graph")
                for item in data["sessions"]
                if item.get("session_type") in ("model", "runtime_model")
            ],
        )

    def test_while_graph_is_compile_subgraph_without_outer_marker(self):
        content = """\
[INFO] Begin to build known shape graph[While0_body].
[INFO] At last, root graph: While0_body, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=3, stream_num:1, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "while.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            [item["session_type"] for item in data["sessions"]],
            ["compile_graph", "runtime_model"],
        )

    def test_dynamic_log_selects_outer_graph_and_deduplicates_init(self):
        log = _external_log("2_动态图.txt")
        if log is None:
            self.skipTest("external GE log unavailable; set GE_LOG_ROOT")
        data = self.run_json(
            "--log", str(log), "--repo-root", str(_analysis_repo_root())
        )
        _assert_dynamic_outer_overview(self, data)
        _assert_dynamic_outer_sessions(self, data)

    def test_single_model_keeps_compile_subgraph_details_by_default(self):
        log = _external_log("1_动态图静态子图.txt")
        if log is None:
            self.skipTest("external GE log unavailable; set GE_LOG_ROOT")
        data = self.run_json(
            "--log", str(log), "--repo-root", str(_analysis_repo_root())
        )
        graphs = {item["graph"]: item for item in data["graphs"]}
        self.assertNotIn("ge_default_20260824201530_dynamic", graphs)
        self.assertEqual(
            graphs["ge_default_20260824201530_dynamic_sub_0_input"]["stream_scope"],
            "submodel",
        )
        self.assertTrue(all(f"Batch_{index}" not in graphs for index in (0, 1, 2)))
        report = self.run_markdown("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertIn("## 图级流证据", report)
        self.assertIn("## 逻辑流 → RT 流", report)
        self.assertIn("ge_default_20260824201530_dynamic_sub_0_input", report)
        self.assertIn("ge_default_20260824201530_dynamic_sub_1_unknow", report)
        self.assertNotIn("父图/父节点", report)
        self.assertNotIn("编译族", report)

    def test_top_level_compile_uses_latest_root_summary_session(self):
        content = """\
[INFO] Begin to build known shape graph[root].
[INFO] At last, root graph: root, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] SetLogicStreamIdAttr:Op [root_op] OpType [Add] logic stream id is 0,
[INFO] Begin to build unknown shape graph[root_subgraph].
[INFO] SetLogicStreamIdAttr:Op [sub_op] OpType [Add] logic stream id is 0,
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "scope.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["compile"]["graph"], "root")
        self.assertEqual(
            [item["name"] for item in data["compile"]["streams"]["0"]["operators"]],
            ["root_op"],
        )

    def test_batch_mapping_keeps_all_streams(self):
        content = """\
[INFO] Begin to build known shape graph[g].
[INFO] SetLogicStreamIdAttr:Op [ascend_mbatch_batch_0_a] OpType [Add] logic stream id is 0,
[INFO] SetLogicStreamIdAttr:Op [ascend_mbatch_batch_0_b] OpType [Add] logic stream id is 1,
[INFO] At last, root graph: g, total stream num: 2, main stream num: 2, attached stream num: 0.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:2, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 21, rt model id: 7, priority: 0, stream flag: 1, task num: 2.
[INFO] CreateNewStream:Create new stream: 0x2, rt stream id: 22, rt model id: 7, priority: 0, stream flag: 1, task num: 2.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 21, model: 7, stream flag: 1.
[INFO] InitRuntimeResource:Logical stream index: 1, rtstream: 22, model: 7, stream flag: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "batch.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["batch_mapping"]["batch_0"]["logic_stream_ids"], [0, 1])
        self.assertEqual(data["batch_mapping"]["batch_0"]["rt_stream_ids"], [21, 22])
        self.assertNotIn("logic_stream_id", data["batch_mapping"]["batch_0"])
        self.assertNotIn("rt_stream_id", data["batch_mapping"]["batch_0"])

    def test_batch_runtime_activation_is_session_scoped(self):
        content = """\\
[INFO] Begin to build known shape graph[g].
[INFO] SetLogicStreamIdAttr:Op [ascend_mbatch_batch_1_a] OpType [Add] logic stream id is 0,
[INFO] At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] CreateNewStream:Create new stream: 0x1, rt stream id: 21, rt model id: 7, priority: 0, stream flag: 1, task num: 2.
[INFO] InitRuntimeResource:Logical stream index: 0, rtstream: 21, model: 7, stream flag: 1.
[INFO] aclmdlSetDynamicBatchSize batchSize[1]
[INFO] current batch label:Batch_1
[INFO] StreamActive_1 active_stream_id=0
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "batch_runtime.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["batch_size"], 1)
        self.assertEqual(data["runtime"]["active_batch_label"], "Batch_1")
        self.assertEqual(data["runtime"]["active_streams"], {"1": 0})
        self.assertEqual(
            data["batch_mapping"]["active_branch"]["logic_stream_ids"], [0]
        )
        self.assertEqual(data["batch_mapping"]["active_branch"]["rt_stream_ids"], [21])

    def test_graph_scope_does_not_infer_parent_relationship(self):
        content = "[INFO] Start assign attached stream for graph wrapper with subgraph num: 2\n"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["compile"]["hierarchy_status"], "unknown")
        self.assertEqual(data["compile"]["graph_scopes"], [])
        self.assertEqual(data["compile"]["sessions"], [])

    def test_explicit_graph_relation_is_preserved(self):
        content = """\
[INFO] Begin to build known shape graph[child].
[INFO] [GraphRelation] parent_graph: root, parent_node: Case, child_graph: child, subgraph_index: 1
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["compile"]["hierarchy_status"], "explicit")
        self.assertEqual(data["compile"]["sessions"][0]["parent_graph"], "root")
        self.assertEqual(data["compile"]["sessions"][0]["parent_node"], "Case")

    def test_multiple_ge_info_operator_sources(self):
        content = """\
[INFO] Begin to build known shape graph[g].
[INFO] SetLogicStreamIdAttr:Op [a] OpType [Conv] logic stream id is 0,
[INFO] Refresh stream by node ids of graph: g, stream_id: 1, type: Add, name: b.
[INFO] Node a assigned stream 1 from stream 0.
[INFO] node c refresh stream id from 0 to 2
[INFO] node: d, type Relu, topo id: 4, logical stream id: 2, real stream size: 1, stream id to task size: {2:1}
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        streams = data["compile"]["streams"]
        self.assertEqual(streams["1"]["operator_count"], 2)
        self.assertEqual(streams["2"]["operator_count"], 2)

    def test_external_runtime_lines_are_ignored(self):
        content = """\
[INFO] GE(1): InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] RUNTIME(1): ModelExecuteTask stream_id=99, model_id=1.
[INFO] RUNTIME(1): KernelTaskInfo Init Success, node :external, logic stream id: 99, stream: 0x1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["model_id"], 1)
        self.assertEqual(data["runtime"]["operator_bindings"], [])
        self.assertNotIn("execute_entry_streams", data["runtime"])

    def test_hccl_follow_stream_summary(self):
        content = "[INFO] GE(1): model total stream num: 2, model stream num: 1, hccl follow stream num: 0\n"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["model_total_stream_count"], 2)
        self.assertEqual(data["runtime"]["follow_stream_count"], 0)

    def test_graph_name_proves_compile_runtime_correlation(self):
        content = """\
[INFO] Begin to build known shape graph[g].
[INFO] At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] Init:Known node:0, refreshable:0, model_id:7, graph_id:0, graph_name:g.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["model_name"], "g")
        self.assertEqual(data["mode"]["compile_runtime_correlation"], "strong")
        self.assertEqual(data["mode"]["analysis_status"], "complete")

    def test_same_session_uses_context_time_and_stream_count_without_name(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[g].
[INFO] GE(100,python3):2026-08-27-10:00:00.100 [stream_allocator.cc:1]2001 At last, root graph: g, total stream num: 2, main stream num: 2, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.200 [stream_allocator.cc:1]2001 After SplitStreamAndRefreshTaskDef, graph:g, stream num:2, notify num:0, event num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.300 [davinci_model.cc:1]2001 InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:2, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["compile_runtime_correlation"], "same_session")
        self.assertEqual(data["mode"]["analysis_status"], "complete")
        self.assertIsNone(data["runtime"]["model_name"])
        self.assertEqual(data["runtime"]["associated_graph"], "g")
        self.assertEqual(
            data["runtime"]["correlation_evidence"],
            [
                "same_pid",
                "same_ge_context",
                "compile_before_runtime",
                "same_time_window",
                "stream_count_match",
                "no_intervening_model_boundary",
            ],
        )

    def test_subgraph_completion_is_not_an_intervening_model_boundary(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[root].
[INFO] GE(100,python3):2026-08-27-10:00:00.100 [stream_allocator.cc:1]2001 At last, root graph: root, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.200 [stream_allocator.cc:1]2001 After SplitStreamAndRefreshTaskDef, graph:root, stream num:1, notify num:0, event num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.210 [graph_builder.cc:1]2001 Begin to build known shape graph[While0_body].
[INFO] GE(100,python3):2026-08-27-10:00:00.220 [stream_allocator.cc:1]2001 At last, root graph: While0_body, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.230 [stream_allocator.cc:1]2001 After SplitStreamAndRefreshTaskDef, graph:While0_body, stream num:1, notify num:0, event num:0.
"""
        data = run_compile_and_runtime(self, content)
        self.assertEqual(data["mode"]["compile_runtime_correlation"], "same_session")

    def test_graph_name_before_runtime_init_does_not_create_strong_match(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[g].
[INFO] GE(100,python3):2026-08-27-10:00:00.050 [compiler.cc:1]2001 Known node:0, graph_name:g.
[INFO] GE(100,python3):2026-08-27-10:00:00.100 [stream_allocator.cc:1]2001 At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.200 [stream_allocator.cc:1]2001 After SplitStreamAndRefreshTaskDef, graph:g, stream num:1, notify num:0, event num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.300 [davinci_model.cc:1]2001 InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertIsNone(data["runtime"]["model_name"])
        self.assertEqual(data["mode"]["compile_runtime_correlation"], "same_session")
        self.assertNotEqual(data["mode"]["compile_runtime_correlation"], "strong")

    def test_same_session_ambiguity_is_not_auto_merged(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[g1].
[INFO] GE(100,python3):2026-08-27-10:00:00.100 [stream_allocator.cc:1]2001 At last, root graph: g1, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.200 [stream_allocator.cc:1]2001 After SplitStreamAndRefreshTaskDef, graph:g1, stream num:1, notify num:0, event num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[g2].
[INFO] GE(100,python3):2026-08-27-10:00:00.200 [stream_allocator.cc:1]2001 At last, root graph: g2, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.200 [stream_allocator.cc:1]2001 After SplitStreamAndRefreshTaskDef, graph:g2, stream num:1, notify num:0, event num:0.
"""
        data = run_compile_and_runtime(self, content)
        self.assertEqual(data["mode"]["compile_runtime_correlation"], "unknown")
        runtime_sessions = _sessions_of_type(data, "runtime_model")
        self.assertEqual(len(runtime_sessions), 1)
        self.assertEqual(len(runtime_sessions[0]["correlation_candidates"]), 2)

    def test_same_graph_or_model_id_isolated_by_pid(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc] Begin to build known shape graph[g].
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [logical_stream_allocator.cc] At last, root graph: g, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(200,python3):2026-08-27-10:01:00.000 [graph_builder.cc] Begin to build known shape graph[g].
[INFO] GE(200,python3):2026-08-27-10:01:00.001 [logical_stream_allocator.cc] At last, root graph: g, total stream num: 2, main stream num: 2, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(len(data["compile"]["sessions"]), 2)
        self.assertEqual(
            {item["pid"] for item in data["compile"]["sessions"]}, {"100", "200"}
        )

    def test_runtime_model_sessions_include_pid(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] GE(200,python3):2026-08-27-10:01:00.000 InitRuntimeParams:InitRuntimeParams: model_id=1, stream_num:2, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(len(data["runtime"]["sessions"]), 2)
        self.assertEqual(
            {item["pid"] for item in data["runtime"]["sessions"]}, {"100", "200"}
        )

    def test_v2_kernel_trace_stream_binding_is_direct_evidence(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [model_v2_executor.cc:1]2001 ModelV2Executor OccupyStreamResource AcquireStreams(stream_num=2)
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [stream.cc:55]2001 [KernelTrace][SplitRtStreams] Get rts stream 0x111 from logical stream 0, rts stream_id: 41
[INFO] GE(100,python3):2026-08-27-10:00:00.011 [stream.cc:55]2001 [KernelTrace][SplitRtStreams] Get rts stream 0x222 from logical stream 1, rts stream_id: 42
[INFO] GE(100,python3):2026-08-27-10:00:00.012 [stream.cc:55]2001 [KernelTrace][SplitRtStreams] Get rts stream 0x333 from logical stream 0, rts_stream_id: 41
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["runtime_backend"], "v2_rt2")
        self.assertEqual(
            [
                {key: item[key] for key in ("logical_stream_index", "rt_stream_id")}
                for item in data["runtime"]["runtime_logical_to_rt_bindings"]
            ],
            [
                {"logical_stream_index": 0, "rt_stream_id": 41},
                {"logical_stream_index": 1, "rt_stream_id": 42},
            ],
        )
        self.assertEqual(
            data["runtime"]["phases"]["runtime_stream_bound"][0]["keyword"],
            "SplitRtStreams",
        )
        self.assertEqual(len(data["runtime"]["kernel_trace_bindings"]), 3)
        self.assertEqual(data["mode"]["analysis_status"], "partial")

    def test_v2_collect_rt2_stream_logs_are_primary_bindings(self):
        content = """\
[INFO] GE(700,python3):2026-09-01-10:00:00.000 [graph_builder.cc:460]5001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dynamic_graph]
[INFO] GE(700,python3):2026-09-01-10:00:00.010 [dynamic_stream_allocator.cc:70]5001 AssignStreamsForDynamicShapeGraph:Graph: dynamic_graph, stream num: 3, event num: 2.
[INFO] GE(700,python3):2026-09-01-10:00:00.020 [model_v2_executor_builder.cc:155]6001 Build:Build RT2 executor for root compute graph[dynamic_graph], model[model_1].
[INFO] GE(700,python3):2026-09-01-10:00:00.021 [model_v2_executor.cc:120]6001 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 50
[INFO] GE(700,python3):2026-09-01-10:00:00.022 [stream_allocator.cc:45]6001 AcquireStreams:Collect rt2 stream, get rts stream 0x101 from logical stream 1, rts_stream_id: 51
[INFO] GE(700,python3):2026-09-01-10:00:00.023 [stream_allocator.cc:45]6001 AcquireStreams:Collect rt2 stream, get rts stream 0x102 from logical stream 2, rts_stream_id: 52
[INFO] GE(700,python3):2026-09-01-10:00:00.024 [model_v2_executor.cc:120]6001 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 50
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_collect.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        model_sessions = [
            item for item in data["sessions"] if item.get("session_type") == "model"
        ]
        self.assertEqual(len(model_sessions), 1)
        runtime = model_sessions[0]["runtime"]
        self.assertEqual(runtime["executor_graph"], "dynamic_graph")
        self.assertEqual(runtime["executor_model"], "model_1")
        self.assertEqual(
            {
                (item["logical_stream_index"], item["rt_stream_id"])
                for item in runtime["allocation_bindings"]
            },
            {(0, 50), (1, 51), (2, 52)},
        )
        self.assertEqual(
            [
                (item["logical_stream_index"], item["rt_stream_id"])
                for item in runtime["runtime_logical_to_rt_bindings"]
            ],
            [(0, 50), (1, 51), (2, 52)],
        )
        self.assertEqual(model_sessions[0]["correlation"], "strong")
        self.assertEqual(model_sessions[0]["analysis_status"], "complete")

    def test_v2_executor_graph_is_primary_anchor_across_contexts(self):
        content = """\
[INFO] GE(500,python3):2026-09-01-10:00:00.000 [graph_builder.cc:460]5001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[graph_a]
[INFO] GE(500,python3):2026-09-01-10:00:00.001 [dynamic_stream_allocator.cc:70]5001 AssignStreamsForDynamicShapeGraph:Graph: graph_a, stream num: 1, event num: 0.
[INFO] GE(500,python3):2026-09-01-10:00:00.002 [graph_builder.cc:460]5001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[graph_b]
[INFO] GE(500,python3):2026-09-01-10:00:00.003 [dynamic_stream_allocator.cc:70]5001 AssignStreamsForDynamicShapeGraph:Graph: graph_b, stream num: 2, event num: 0.
[INFO] GE(700,python3):2026-09-01-10:00:00.010 [model_v2_executor_builder.cc:155]6001 Build:Build RT2 executor for root compute graph[graph_b], model[model_b].
[INFO] GE(700,python3):2026-09-01-10:00:00.011 [model_v2_executor.cc:120]6002 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 50
[INFO] GE(700,python3):2026-09-01-10:00:00.012 [stream_allocator.cc:45]6003 AcquireStreams:Collect rt2 stream, get rts stream 0x101 from logical stream 1, rts_stream_id: 51
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_graph_anchor.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [
            item for item in data["sessions"] if item.get("session_type") == "model"
        ]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["graph"], "graph_b")
        self.assertEqual(models[0]["correlation"], "strong")
        self.assertIn("executor_graph_exact_match", models[0]["correlation_evidence"])
        self.assertNotIn("same_pid", models[0]["correlation_evidence"])
        self.assertEqual(
            [
                (item["logical_stream_index"], item["rt_stream_id"])
                for item in models[0]["runtime"]["runtime_logical_to_rt_bindings"]
            ],
            [(0, 50), (1, 51)],
        )

    def test_v2_executor_windows_keep_same_pid_models_separate(self):
        content = """\
[INFO] GE(700,python3):2026-09-01-10:00:00.000 [graph_builder.cc:460]5001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[g1]
[INFO] GE(700,python3):2026-09-01-10:00:00.001 [dynamic_stream_allocator.cc:70]5001 AssignStreamsForDynamicShapeGraph:Graph: g1, stream num: 1, event num: 0.
[INFO] GE(700,python3):2026-09-01-10:00:00.002 [graph_builder.cc:460]5001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[g2]
[INFO] GE(700,python3):2026-09-01-10:00:00.003 [dynamic_stream_allocator.cc:70]5001 AssignStreamsForDynamicShapeGraph:Graph: g2, stream num: 1, event num: 0.
[INFO] GE(700,python3):2026-09-01-10:00:00.010 [model_v2_executor_builder.cc:155]6001 Build:Build RT2 executor for root compute graph[g1], model[m1].
[INFO] GE(700,python3):2026-09-01-10:00:00.011 [model_v2_executor.cc:120]6002 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 50
[INFO] GE(700,python3):2026-09-01-10:00:00.020 [model_v2_executor_builder.cc:155]6003 Build:Build RT2 executor for root compute graph[g2], model[m2].
[INFO] GE(700,python3):2026-09-01-10:00:00.021 [model_v2_executor.cc:120]6004 OccupyStreamResource:Collect rt2 stream, get rts stream 0x200 from logical stream 0, rts_stream_id: 60
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_executor_windows.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = {
            item["graph"]: item
            for item in data["sessions"]
            if item.get("session_type") == "model"
        }
        self.assertEqual(set(models), {"g1", "g2"})
        self.assertEqual(models["g1"]["runtime"]["mapping_rows"][0]["rt_stream_id"], 50)
        self.assertEqual(models["g2"]["runtime"]["mapping_rows"][0]["rt_stream_id"], 60)

    def test_v2_executor_coalesces_repeated_resources_and_binding_fragments(self):
        first = """\
[INFO] GE(500,atc):2026-09-01-10:29:48.100 [graph_builder.cc:460]5001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[graph_a]
[INFO] GE(500,atc):2026-09-01-10:29:48.110 [dynamic_stream_allocator.cc:70]5001 AssignStreamsForDynamicShapeGraph:Graph: graph_a, stream num: 4, event num: 0.
[INFO] GE(700,python3):2026-09-01-10:29:48.460 [model_v2_executor_builder.cc:155]6000 Build:Build RT2 executor for root compute graph[graph_a], model[model_a].
[INFO] GE(700,python3):2026-09-01-10:29:48.468 [model_converter.cc:366]6001 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(700,python3):2026-09-01-10:29:48.469 [stream_allocator.cc:45]6002 AcquireStreams:Collect rt2 stream, get rts stream 0x101 from logical stream 1, rts_stream_id: 56
[INFO] GE(700,python3):2026-09-01-10:29:48.470 [stream_allocator.cc:45]6002 AcquireStreams:Collect rt2 stream, get rts stream 0x102 from logical stream 2, rts_stream_id: 57
[INFO] GE(700,python3):2026-09-01-10:29:48.471 [stream_allocator.cc:45]6002 AcquireStreams:Collect rt2 stream, get rts stream 0x103 from logical stream 3, rts_stream_id: 58
"""
        second = """\
[INFO] GE(700,python3):2026-09-01-10:29:48.475 [model_converter.cc:366]6003 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(700,python3):2026-09-01-10:29:48.476 [model_v2_executor.cc:120]6004 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 63
[INFO] GE(700,python3):2026-09-01-10:29:48.480 [stream.cc:55]6005 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x100 from logical stream 0, rts_stream_id: 63
[INFO] GE(700,python3):2026-09-01-10:29:48.481 [stream.cc:55]6005 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x101 from logical stream 1, rts_stream_id: 56
[INFO] GE(700,python3):2026-09-01-10:29:48.482 [stream.cc:55]6005 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x102 from logical stream 2, rts_stream_id: 57
[INFO] GE(700,python3):2026-09-01-10:29:48.483 [stream.cc:55]6005 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x103 from logical stream 3, rts_stream_id: 58
"""
        with tempfile.TemporaryDirectory() as directory:
            first_log = Path(directory) / "35816.log"
            second_log = Path(directory) / "52174.log"
            first_log.write_text(first)
            second_log.write_text(second)
            data = self.run_json(
                "--log",
                str(first_log),
                "--log",
                str(second_log),
                "--repo-root",
                str(REPO_ROOT),
            )
        runtime_only = _sessions_of_type(data, "runtime_model")
        self.assertEqual(runtime_only, [])
        _assert_v2_resource_result(self, data)

    def test_v2_resource_blocks_merge_subset_and_superset_without_anchor(self):
        content = """\
[INFO] GE(303,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]7000 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dynamic_root].
[INFO] GE(303,python3):2026-08-27-10:00:00.010 [dynamic_stream_allocator.cc:73]7000 AssignStreamsForDynamicShapeGraph:Graph: dynamic_root, stream num:4, event num:0.
[INFO] GE(303,python3):2026-08-27-10:00:01.000 [model_converter.cc:366]7001 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(303,python3):2026-08-27-10:00:01.001 [stream_allocator.cc:45]7001 AcquireStreams:Collect rt2 stream, get rts stream 0x101 from logical stream 1, rts_stream_id: 56
[INFO] GE(303,python3):2026-08-27-10:00:01.002 [stream_allocator.cc:45]7001 AcquireStreams:Collect rt2 stream, get rts stream 0x102 from logical stream 2, rts_stream_id: 57
[INFO] GE(303,python3):2026-08-27-10:00:01.003 [stream_allocator.cc:45]7001 AcquireStreams:Collect rt2 stream, get rts stream 0x103 from logical stream 3, rts_stream_id: 58
[INFO] GE(303,python3):2026-08-27-10:00:01.004 [model_converter.cc:366]7002 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(303,python3):2026-08-27-10:00:01.005 [stream.cc:55]7002 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x100 from logical stream 0, rts_stream_id: 63
[INFO] GE(303,python3):2026-08-27-10:00:01.006 [stream.cc:55]7002 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x101 from logical stream 1, rts_stream_id: 56
[INFO] GE(303,python3):2026-08-27-10:00:01.007 [stream.cc:55]7002 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x102 from logical stream 2, rts_stream_id: 57
[INFO] GE(303,python3):2026-08-27-10:00:01.008 [stream.cc:55]7002 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x103 from logical stream 3, rts_stream_id: 58
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_subset_superset.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        self.assertEqual(len(models), 1)
        self.assertFalse(
            any(item["session_type"] == "runtime_model" for item in data["sessions"])
        )
        runtime = models[0]["runtime"]
        self.assertEqual(len(runtime["v2_resource_summaries"]), 1)
        self.assertEqual(
            [
                (item["logical_stream_index"], item["rt_stream_id"])
                for item in runtime["runtime_logical_to_rt_bindings"]
            ],
            [(0, 63), (1, 56), (2, 57), (3, 58)],
        )

    def test_v2_conflicting_resource_summaries_are_flagged_not_split(self):
        content = """\
[INFO] GE(700,python3):2026-09-01-10:00:00.000 [model_v2_executor_builder.cc:155]6000 Build:Build RT2 executor for root compute graph[graph_a], model[model_a].
[INFO] GE(700,python3):2026-09-01-10:00:00.001 [model_converter.cc:366]6001 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(700,python3):2026-09-01-10:00:00.002 [model_converter.cc:366]6002 GetReusableStreamResourceNum:Root graph total stream_num 5, reusable stream num 5, attached stream num 0, event_num 0, notify_num 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_conflicting_resources.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        sessions = [
            item
            for item in data["runtime"]["sessions"]
            if item.get("v2_resource_summaries")
        ]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(sessions[0]["v2_resource_summaries"]), 2)
        self.assertEqual(len(sessions[0]["v2_resource_conflicts"]), 1)

    def test_v2_kernel_trace_accepts_arbitrary_split_suffix(self):
        content = """\\
[INFO] GE(123,python3):2026-08-31-10:00:00.000 [model_converter.cc:366]2001 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 10, notify_num 0.
[INFO] GE(123,python3):2026-08-31-10:00:00.010 [davinci_model.cc:701]2001 InitRuntimeParams:InitRuntimeParams: model_id=2, session_id:0, device_id:0, stream_num:4, notify_num:0, event_num:10, label_num:0.
[INFO] GE(123,python3):2026-08-31-10:00:00.011 [stream.cc:55]2001 [KernelTrace][SplitRtStreams_35] Get rts stream 0x111 from logical stream 0, rts_stream_id: 41
[INFO] GE(123,python3):2026-08-31-10:00:00.012 [stream.cc:55]2001 [KernelTrace][SplitRtStreams_backend_v2] Get rts stream 0x222 from logical stream 1, rts_stream_id: 42
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_suffix.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        runtime = data["runtime"]
        self.assertEqual(runtime["runtime_backend"], "v2_rt2")
        self.assertEqual(runtime["pid"], "123")
        self.assertEqual(runtime["ge_context"], "2001")
        self.assertEqual(runtime["model_id"], 2)
        self.assertEqual(runtime["reusable_stream_count"], 4)
        self.assertEqual(runtime["attached_stream_count"], 0)
        self.assertEqual(runtime["model_total_stream_count"], 4)
        self.assertEqual(
            [
                (item["logical_stream_index"], item["rt_stream_id"])
                for item in runtime["runtime_logical_to_rt_bindings"]
            ],
            [(0, 41), (1, 42)],
        )
        self.assertEqual(len(runtime["sessions"]), 1)
        self.assertEqual(len(runtime["sessions"][0]["v2_resource_summaries"]), 1)

    def test_v2_pending_resource_summary_and_trace_share_unknown_session(self):
        content = """\\
[INFO] GE(123,python3):2026-08-31-10:00:00.000 [model_converter.cc:366]2001 GetReusableStreamResourceNum:Root graph total stream_num 4, reusable stream num 4, attached stream num 0, event_num 10, notify_num 0.
[INFO] GE(123,python3):2026-08-31-10:00:00.010 [stream.cc:55]2001 [KernelTrace][SplitRtStreams_backend_v2] Get rts stream 0x111 from logical stream 0, rts_stream_id: 41
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_pending.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        runtime = data["runtime"]
        self.assertEqual(len(runtime["sessions"]), 1)
        self.assertIsNone(runtime["model_id"])
        self.assertEqual(runtime["reusable_stream_count"], 4)
        self.assertEqual(
            runtime["runtime_logical_to_rt_bindings"][0]["rt_stream_id"], 41
        )

    def test_kernel_trace_from_unrelated_source_is_ignored(self):
        content = """\\
[INFO] GE(123,python3):2026-08-31-10:00:00.000 [unrelated.cc:55]2001 [KernelTrace][SplitRtStreams_35] Get rts stream 0x111 from logical stream 0, rts_stream_id: 41
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_wrong_source.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["runtime"]["runtime_logical_to_rt_bindings"], [])
        self.assertEqual(data["runtime"]["sessions"], [])

    def test_v2_resource_summary_separators_and_scopes_are_preserved(self):
        content = """\\
[INFO] GE(123,python3):2026-08-31-10:00:00.000 [model_converter.cc:60]2001 GetNonRootModelResourceNum:Static sub model child, stream_num 2, event_num is 3, notify_num:1.
[INFO] GE(123,python3):2026-08-31-10:00:00.001 [model_converter.cc:380]2001 GetReusableStreamResourceNum:Root model root_model, total_stream_num:4, attached_stream_num=0, event_num 10, notify_num is 0.
[INFO] GE(123,python3):2026-08-31-10:00:00.002 [model_v2_executor.cc:119]2001 AcquireStreams stream_num 4
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_resources.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        runtime = data["runtime"]
        self.assertEqual(runtime["runtime_backend"], "v2_rt2")
        self.assertIsNone(runtime["requested_stream_count"])
        self.assertEqual(runtime["reusable_stream_count"], None)
        self.assertEqual(runtime["attached_stream_count"], 0)
        self.assertEqual(runtime["model_total_stream_count"], 4)
        summary_sessions = [
            item for item in runtime["sessions"] if item.get("v2_resource_summaries")
        ]
        self.assertEqual(len(summary_sessions), 1)
        self.assertEqual(
            sorted(
                (
                    summary["scope"],
                    summary["name"],
                    summary.get("stream_count"),
                )
                for summary in summary_sessions[0]["v2_resource_summaries"]
            ),
            [("root_model", "root_model", None), ("static_submodel", "child", 2)],
        )

    def test_v2_preinit_trace_is_promoted_when_init_follows(self):
        content = """\\
[INFO] GE(123,python3):2026-08-31-10:00:00.000 [model_v2_executor.cc:119]2001 AcquireStreams stream_num 1
[INFO] GE(123,python3):2026-08-31-10:00:00.001 [stream.cc:55]2001 [KernelTrace][SplitRtStreams_backend_v2] Get rts stream 0x111 from logical stream 0, rts_stream_id: 41
[INFO] GE(123,python3):2026-08-31-10:00:00.002 [davinci_model.cc:701]2001 InitRuntimeParams:InitRuntimeParams: model_id=2, session_id:0, device_id:0, stream_num:1, notify_num:0, event_num:0, label_num:0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_preinit.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        runtime = data["runtime"]
        self.assertEqual(runtime["runtime_backend"], "v2_rt2")
        self.assertEqual(runtime["model_id"], 2)
        self.assertEqual(len(runtime["sessions"]), 1)
        self.assertEqual(runtime["sessions"][0]["session_id"], "model:123:2")
        self.assertEqual(
            runtime["runtime_logical_to_rt_bindings"][0]["rt_stream_id"], 41
        )

    def test_v2_kernel_trace_without_init_keeps_unknown_model_scope(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [model_v2_executor.cc:1]2001 ModelV2Executor OccupyStreamResource AcquireStreams(stream_num=1)
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [stream.cc:55]2001 [KernelTrace][SplitRtStreams] Get rts stream 0x111 from logical stream 0, rts stream_id: 41
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertIsNone(data["runtime"]["model_id"])
        self.assertIsNone(data["runtime"]["requested_stream_count"])
        self.assertEqual(
            data["runtime"]["runtime_logical_to_rt_bindings"][0]["rt_stream_id"], 41
        )
        self.assertEqual(
            data["runtime"]["sessions"][0]["session_role"], "runtime_model"
        )

    def test_v2_kernel_trace_events_and_notifies_are_preserved(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [KernelTrace][CreateNotifies] Get rts notify 0x111 from logical notify 2
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [KernelTrace][CallRtsSendEvent] Sent event 3 RT event 0x222 from stream 4
[INFO] GE(100,python3):2026-08-27-10:00:00.011 [KernelTrace][CallRtsWaitEvent] Waited event 3 RT event 0x222 at stream 5
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime_v2_events.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            data["runtime"]["kernel_trace_notifies"][0]["logical_notify_index"], 2
        )
        self.assertEqual(
            [item["logic_event_id"] for item in data["runtime"]["kernel_trace_events"]],
            [3, 3],
        )
        self.assertIn("runtime_event_sent", data["runtime"]["phases"])
        self.assertIn("runtime_event_waited", data["runtime"]["phases"])

    def test_dynamic_stream_summary_does_not_pollute_static_graph(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 Begin to build known shape graph[static_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 Graph: static_root, stream num: 9, event num: 8.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 At last, root graph: static_root, total stream num: 1, main stream num: 1, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "static.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        session = data["compile"]["sessions"][0]
        self.assertIsNone(session["dynamic_stream_count"])
        self.assertEqual(session["graph_class"], "static_shape")

    def test_graph_class_and_compile_path_are_saved_per_graph(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[root_dynamic].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: root_dynamic, stream num: 2, event num: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [graph_builder.cc:1]2001 BuildForKnownShapeGraph:Begin to build known shape graph[While0_body].
[INFO] GE(100,python3):2026-08-27-10:00:00.030 [logical_stream_allocator.cc:1]2001 At last, root graph: While0_body, total stream num: 1, main stream num: 1, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.040 [graph_builder.cc:1]2001 RefreshInfoOfDynamicShapeGraph:Total stream num: 3, event num: 1, notify num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "hybrid.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            data["mode"]["root_graph_class"], "hybrid_dynamic_static_subgraph"
        )
        graphs = {item["graph"]: item for item in data["compile"]["sessions"]}
        self.assertEqual(graphs["root_dynamic"]["graph_class"], "dynamic_shape")
        self.assertEqual(graphs["root_dynamic"]["compile_path"], "dynamic_shape")
        self.assertEqual(graphs["While0_body"]["graph_class"], "static_shape")
        self.assertEqual(graphs["While0_body"]["compile_path"], "known_shape")
        self.assertEqual(graphs["While0_body"]["session_role"], "compile_subgraph")
        self.assertNotIn(
            "logical_stream_assigned",
            graphs["root_dynamic"]["phases"],
        )
        self.assertIn("dynamic_stream_assigned", graphs["root_dynamic"]["phases"])
        self.assertNotIn("aggregate_stream_count", graphs["root_dynamic"])
        self.assertIn("logical_stream_assigned", graphs["While0_body"]["phases"])
        self.assertEqual(
            graphs["While0_body"]["compile_family_id"],
            graphs["root_dynamic"]["compile_family_id"],
        )

    def test_pure_dynamic_root_uses_dynamic_compile_path(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build unknown shape graph[dynamic_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: dynamic_root, stream num: 2, event num: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [graph_builder.cc:1]2001 RefreshInfoOfDynamicShapeGraph:Total stream num: 2, event num: 1, notify num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["root_graph_class"], "pure_dynamic_shape")
        self.assertEqual(data["mode"]["scenario_class"], "dynamic")
        self.assertEqual(data["compile"]["compile_path"], "dynamic_shape")
        self.assertEqual(data["compile"]["dynamic_stream_count"], 2)

    def test_dynamic_root_sub_total_aggregate_is_audit_only(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:738]2001 RefreshInfoOfDynamicShapeGraph:Root model: root, stream num: 2, event num: 1, notify num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.001 [graph_builder.cc:751]2001 RefreshInfoOfDynamicShapeGraph:Sub model: child, stream num: 1, event num: 2, notify num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.002 [graph_builder.cc:761]2001 RefreshInfoOfDynamicShapeGraph:Total stream num: 3, event num: 3, notify num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_aggregate_only.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertNotIn("ignored_dynamic_aggregate_evidence", data["compile"])
        self.assertIsNone(data["compile"]["dynamic_stream_count"])
        self.assertNotIn("aggregate_stream_count", data["mode"])
        self.assertEqual(data["mode"]["analysis_status"], "no_stream_evidence")

    def test_dynamic_allocator_summary_can_seed_missing_build_entry(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [dynamic_stream_allocator.cc:73]2001 AssignStreamsForDynamicShapeGraph:Graph: dynamic_root, stream num: 2, event num: 1.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_summary.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["shape_mode"], "dynamic_shape")
        self.assertEqual(data["mode"]["root_graph_class"], "pure_dynamic_shape")

    def test_classification_and_v2_fixture_files(self):
        fixture = SKILL_ROOT / "tests" / "fixtures"
        _assert_classification_fixtures(self, fixture)

    def test_dynamic_batch_known_branch_uses_static_compile_path(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 BuildForKnownShapeGraph:Begin to build known shape graph[batch_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [multi_batch_clone_pass.cc:1]2001 ascend_mbatch_shape_case ascend_mbatch_shape_data.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [logical_stream_allocator.cc:1]2001 At last, root graph: batch_root, total stream num: 2, main stream num: 2, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.030 [acl.cc:1]2001 aclmdlSetDynamicBatchSize batchSize[2]
[INFO] GE(100,python3):2026-08-27-10:00:00.040 [label.cc:1]2001 current batch label:Batch_1
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "batch.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["graph_form"], "dynamic_batch_static_branches")
        self.assertEqual(data["mode"]["scenario_class"], "dynamic")
        self.assertEqual(data["mode"]["shape_mode"], "dynamic_shape")
        self.assertEqual(data["mode"]["root_graph_class"], "pure_static_shape")
        self.assertEqual(data["compile"]["sessions"][0]["compile_path"], "known_shape")

    def test_dynamic_batch_info_log_marks_dynamic_scenario(self):
        """Use the real INFO marker emitted by multi_batch_options.cc."""
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 BuildForKnownShapeGraph:Begin to build known shape graph[batch_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [multi_batch_options.cc:527]2001 Found dynamic batch, shape [2]
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [logical_stream_allocator.cc:1]2001 At last, root graph: batch_root, total stream num: 2, main stream num: 2, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_batch_info.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["scenario_class"], "dynamic")
        self.assertEqual(data["mode"]["graph_form"], "dynamic_batch_static_branches")
        self.assertGreater(data["compile"]["dynamic_markers"]["dynamic_batch_info"], 0)
        self.assertIn(
            "dynamic_batch_option_observed",
            data["compile"]["phases"],
        )
        self.assertEqual(data["compile"]["sessions"][0]["compile_path"], "known_shape")

    def test_dynamic_batch_marker_before_graph_is_attached_to_next_root(self):
        content = """\\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [multi_batch_options.cc:527]2001 Found dynamic batch, shape [2]
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [graph_builder.cc:1]2001 BuildForKnownShapeGraph:Begin to build known shape graph[batch_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [logical_stream_allocator.cc:1]2001 At last, root graph: batch_root, total stream num: 2, main stream num: 2, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_batch_before_graph.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["scenario_class"], "dynamic")
        self.assertEqual(data["compile"]["sessions"][0]["dynamic_batch_markers"], 1)
        self.assertFalse(data["compile"]["unassociated_dynamic_batch"])

    def test_dynamic_batch_belongs_to_dynamic_category_but_uses_davinci_runtime(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [graph_builder.cc:1]2001 Begin to build known shape graph[dynamic_batch_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [multi_batch_clone_pass.cc:1]2001 ascend_mbatch_shape_case ascend_mbatch_shape_data.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [logical_stream_allocator.cc:1]2001 At last, root graph: dynamic_batch_root, total stream num: 2, main stream num: 2, attached stream num: 0.
[INFO] GE(100,python3):2026-08-27-10:00:00.030 [davinci_model.cc:1]2001 InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:2, notify_num:0, event_num:0, label_num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.040 [reusable_stream_allocator.cc:1]2001 Create new stream: 0x1, rt stream id: 41, rt model id: 7, priority: 0, stream flag: 257, task num: 1.
[INFO] GE(100,python3):2026-08-27-10:00:00.050 [davinci_model.cc:1]2001 Logical stream index: 0, rtstream: 41, model: 7, stream flag: 257.
[INFO] GE(100,python3):2026-08-27-10:00:00.060 [acl.cc:1]2001 aclmdlSetDynamicBatchSize batchSize[2]
[INFO] GE(100,python3):2026-08-27-10:00:00.070 [label.cc:1]2001 current batch label:Batch_1
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_batch_runtime.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["scenario_class"], "dynamic")
        self.assertEqual(data["mode"]["shape_mode"], "dynamic_shape")
        self.assertEqual(data["compile"]["compile_path"], "known_shape")
        model_sessions = [
            item for item in data["sessions"] if item["session_type"] == "model"
        ]
        self.assertEqual(len(model_sessions), 1)
        self.assertEqual(model_sessions[0]["runtime_path"], "v1_davinci")
        self.assertNotEqual(model_sessions[0]["runtime_path"], "v1_hybrid_dynamic")

    def test_hybrid_context_marker_is_optional_and_not_a_stream_fact(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 [hybrid_model.cc:66]2001 Start to init hybrid model 7, gert enabled: 0, single op: 0
[INFO] GE(100,python3):2026-08-27-10:00:00.010 [davinci_model.cc:1]2001 InitRuntimeParams:InitRuntimeParams: model_id=7, stream_num:1, notify_num:0, event_num:0, label_num:0.
[INFO] GE(100,python3):2026-08-27-10:00:00.020 [hybrid_model_rt_v1_executor.cc:173]2001 HybridModel will execute in rt1.0 singleline mode
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "hybrid_context.log"
            log.write_text(content)
            data = self.run_json("--runtime", str(log), "--repo-root", str(REPO_ROOT))
        session = data["runtime"]["sessions"][0]
        self.assertEqual(
            {item["phase"] for item in session["hybrid_context_markers"]},
            {"hybrid_model_init_started", "hybrid_execute_rt1"},
        )
        self.assertIn("hybrid_model_init_started", session["phases"])
        self.assertIn("hybrid_execute_rt1", session["phases"])
        self.assertEqual(session["runtime_backend"], "v1_davinci_model")
        self.assertIsNone(session["model_total_stream_count"])

    def test_known_shape_root_is_pure_static(self):
        content = """\
[INFO] GE(100,python3):2026-08-27-10:00:00.000 Begin to build known shape graph[static_root].
[INFO] GE(100,python3):2026-08-27-10:00:00.010 At last, root graph: static_root, total stream num: 1, main stream num: 1, attached stream num: 0.
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "static_root.log"
            log.write_text(content)
            data = self.run_json("--compile", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(data["mode"]["root_graph_class"], "pure_static_shape")
        self.assertEqual(data["mode"]["scenario_class"], "static")
        self.assertEqual(data["compile"]["compile_path"], "known_shape")

    def test_dynamic_log_regression_keeps_two_models_and_compile_subgraphs(self):
        log = _external_log("2_动态图.txt")
        if log is None:
            self.skipTest("external GE log unavailable; set GE_LOG_ROOT")
        data = self.run_json(
            "--log", str(log), "--repo-root", str(_analysis_repo_root())
        )
        model_sessions = [
            item for item in data["sessions"] if item["session_type"] == "model"
        ]
        self.assertEqual(
            {(item["graph"], item["model_id"]) for item in model_sessions},
            {
                ("ge_default_20260824202014", 1),
            },
        )
        self.assertEqual(data["mode"]["scenario_class"], "hybrid")
        subgraphs = [
            item for item in data["sessions"] if item["session_type"] == "compile_graph"
        ]
        self.assertTrue(subgraphs)
        self.assertTrue(
            all(
                item["session_role"] == "compile_subgraph"
                for item in subgraphs
                if item["graph"] != "ge_default_20260824202021"
            )
        )
        self.assertIn(
            "ge_default_20260824202021", {item["graph"] for item in subgraphs}
        )
        model_categories = {
            (item["graph"], item["model_id"]): item["scenario_class"]
            for item in model_sessions
        }
        self.assertEqual(model_categories["ge_default_20260824202014", 1], "static")
        runtime_only = next(
            item
            for item in data["sessions"]
            if item["session_type"] == "runtime_model" and item["model_id"] == 2
        )
        self.assertEqual(runtime_only["scenario_class"], "unknown")
        self.assertEqual(runtime_only["correlation"], "unknown")

        report = self.run_markdown(
            "--log", str(log), "--repo-root", str(_analysis_repo_root())
        )
        self.assertIn("hybrid_dynamic_static_subgraph", report)
        self.assertIn("ge_default_20260824202014", report)
        self.assertIn("ge_default_20260824202021", report)
        self.assertIn("While0cond_control_KSkUzQkkVgg1", report)

    def test_dynamic_batch_category_is_not_contaminated_by_earlier_dynamic_family(self):
        """A mixed file may contain an unrelated dynamic compile family first."""
        log = _external_log("1_动态图静态子图.txt")
        if log is None:
            self.skipTest("external GE log unavailable; set GE_LOG_ROOT")
        data = self.run_json(
            "--log", str(log), "--repo-root", str(_analysis_repo_root())
        )
        self.assertEqual(data["mode"]["scenario_class"], "dynamic")
        self.assertEqual(data["mode"]["graph_form"], "dynamic_batch_static_branches")
        self.assertEqual(data["mode"]["root_graph_class"], "pure_static_shape")
        model_sessions = [
            item for item in data["sessions"] if item["session_type"] == "model"
        ]
        self.assertEqual(len(model_sessions), 1)
        self.assertEqual(model_sessions[0]["scenario_class"], "dynamic")
        self.assertEqual(model_sessions[0]["runtime_path"], "v1_davinci")
        report = self.run_markdown(
            "--log", str(log), "--repo-root", str(_analysis_repo_root())
        )
        self.assertIn("场景/Shape：`dynamic`", report)

    def test_v2_trace_without_init_is_attached_to_compile_graph_by_index(self):
        """V2 SplitRtStreams has no model id, but can use a unique graph window."""
        content = """\
[INFO] GE(300,python3):2026-08-27-10:00:00.000 [graph_builder.cc:343]5001 BuildForKnownShapeGraph:Begin to build known shape graph[v2_graph].
[INFO] GE(300,python3):2026-08-27-10:00:00.005 [logical_stream_allocator.cc:873]5001 Assign:[Assign][LogicalStream] At last, root graph: v2_graph, total stream num:2, main stream num:2, attached stream num:0.
[INFO] GE(300,python3):2026-08-27-10:00:00.010 [stream_allocator.cc:677]5001 SplitStreamAndRefreshTaskDef:After SplitStreamAndRefreshTaskDef, graph:v2_graph, stream num:2, notify num:0, event num:0.
[INFO] GE(300,python3):2026-08-27-10:00:00.020 [stream.cc:55]5001 [KernelTrace][SplitRtStreams_backend_v2] Get rts stream 0x1 from logical stream 0, rts_stream_id: 41
[INFO] GE(300,python3):2026-08-27-10:00:00.021 [stream.cc:55]5001 [KernelTrace][SplitRtStreams_backend_v2] Get rts stream 0x2 from logical stream 1, rts_stream_id: 42
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_paired.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        self.assertEqual(len(models), 1)
        model = models[0]
        self.assertEqual(model["graph"], "v2_graph")
        self.assertIsNone(model["model_id"])
        self.assertEqual(model["correlation"], "same_session")
        self.assertEqual(model["correlation_inference"], "inferred")
        self.assertEqual(data["mode"]["analysis_status"], "complete")
        self.assertEqual(data["mode"]["stream_count_status"], "complete")
        record = next(
            item for item in data["stream_records"] if item["graph"] == "v2_graph"
        )
        self.assertEqual(
            [
                (row["compile_logic_stream_id"], row["rt_stream_id"])
                for row in record["runtime"]["mapping_rows"]
            ],
            [(0, 41), (1, 42)],
        )

    def test_v2_trace_pairs_by_pid_and_logical_index_without_time_context_gate(self):
        content = """\\
[INFO] GE(3200928,python3.7):2026-08-27-20:00:00.000 [graph_builder.cc:462]3201000 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dyn_graph].
[INFO] GE(3200928,python3.7):2026-08-27-20:00:00.010 [dynamic_stream_allocator.cc:73]3201000 AssignStreamsForDynamicShapeGraph:Graph: dyn_graph, stream num:4, event num:0.
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.692 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x1 from logical stream 0, rts_stream_id: 60
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.693 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x2 from logical stream 1, rts_stream_id: 61
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.694 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x3 from logical stream 2, rts_stream_id: 62
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.695 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x4 from logical stream 3, rts_stream_id: 63
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_dynamic_rotated.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["graph"], "dyn_graph")
        self.assertEqual(models[0]["correlation"], "same_session")
        self.assertIn("same_pid", models[0]["correlation_evidence"])
        self.assertIn("runtime_logical_index_match", models[0]["correlation_evidence"])
        self.assertEqual(
            [row["rt_stream_id"] for row in models[0]["runtime"]["mapping_rows"]],
            [60, 61, 62, 63],
        )

    def test_v2_dynamic_trace_fragments_merge_across_contexts_and_rotated_logs(self):
        content = """\
[INFO] GE(3200928,python3.7):2026-08-27-21:04:50.000 [graph_builder.cc:462]3201000 BuildForUnknownShapeGraph:Begin to build unknown shape graph[ge_default_20260827210450].
[INFO] GE(3200928,python3.7):2026-08-27-21:04:50.010 [dynamic_stream_allocator.cc:73]3201000 AssignStreamsForDynamicShapeGraph:Graph: ge_default_20260827210450, stream num:4, event num:0.
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.692 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x1 from logical stream 0, rts_stream_id: 60
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.693 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x2 from logical stream 1, rts_stream_id: 59
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.694 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x3 from logical stream 2, rts_stream_id: 58
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.695 [stream.cc:55]3201199 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x4 from logical stream 3, rts_stream_id: 57
[INFO] GE(3200928,python3.7):2026-08-27-21:05:11.692 [stream.cc:55]3201200 SplitRtStreams:[KernelTrace][SplitRtStreams_15]Get rts stream 0x1 from logical stream 0, rts_stream_id: 60
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_dynamic_rotated_overlap.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
            report = self.run_markdown("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        runtime_only = _sessions_of_type(data, "runtime_model")
        self.assertEqual(len(models), 1)
        self.assertEqual(runtime_only, [])
        self.assertEqual(models[0]["graph"], "ge_default_20260827210450")
        self.assertEqual(models[0]["correlation"], "same_session")
        self.assertEqual(
            [
                (row["compile_logic_stream_id"], row["rt_stream_id"])
                for row in models[0]["runtime"]["mapping_rows"]
            ],
            [(0, 60), (1, 59), (2, 58), (3, 57)],
        )
        self.assertIn("| ge_default_20260827210450 | - | 0 | 0 | 60 |", report)
        self.assertNotIn("| unknown | - | unknown | 0 | 60 |", report)

    def test_dynamic_subgraphs_without_allocator_summary_keep_runtime_single_stream(
        self,
    ):
        content = """\
[INFO] GE(100,atc):2026-09-04-17:52:51.000 [graph_builder.cc:460]1001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[ge_default_20260904175251_dynamic_sub_1_unknow]
[INFO] GE(100,atc):2026-09-04-17:52:51.001 [graph_builder.cc:460]1001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[ge_default_20260904175251_dynamic_sub_2_unknow]
[INFO] GE(100,atc):2026-09-04-17:52:51.002 [graph_builder.cc:460]1001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[ge_default_20260904175251_dynamic_sub_3_unknow]
[INFO] GE(200,python3):2026-09-04-17:53:57.700 [model_v2_executor_builder.cc:158]2001 Build:Build RT2 executor for root compute graph[ge_default_20260904175251_dynamic], model[ge_default_20260904175251_dynamic].
[INFO] GE(200,python3):2026-09-04-17:53:57.758 [model_v2_executor.cc:120]2002 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 53
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_subgraphs_single_stream.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
            report = self.run_markdown("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            [item["session_type"] for item in data["sessions"]], ["runtime_model"]
        )
        runtime_session = data["sessions"][0]
        self.assertEqual(runtime_session["graph"], "ge_default_20260904175251_dynamic")
        self.assertEqual(runtime_session["correlation"], "dynamic_subgraph_family")
        self.assertEqual(
            runtime_session["runtime"]["runtime_logical_to_rt_bindings"][0][
                "rt_stream_id"
            ],
            53,
        )
        self.assertEqual(data["compile"]["sessions"], [])
        self.assertEqual(len(data["compile"]["suppressed_compile_sessions"]), 3)
        self.assertIn("| ge_default_20260904175251_dynamic | - | - | 0 | 53 |", report)
        self.assertIn("Collect rt2 stream", report)
        self.assertIn("runtime_only", report)
        self.assertNotIn("unknown | 0 | 53", report)
        self.assertNotIn("dynamic_sub_1_unknow", report)

    def test_dynamic_subgraphs_without_summary_do_not_suppress_runtime_multistream(
        self,
    ):
        content = """\
[INFO] GE(100,atc):2026-09-04-17:52:51.000 [graph_builder.cc:460]1001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[ge_default_20260904175251_dynamic_sub_1_unknow]
[INFO] GE(200,python3):2026-09-04-17:53:57.700 [model_v2_executor_builder.cc:158]2001 Build:Build RT2 executor for root compute graph[ge_default_20260904175251_dynamic], model[ge_default_20260904175251_dynamic].
[INFO] GE(200,python3):2026-09-04-17:53:57.758 [model_v2_executor.cc:120]2002 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 53
[INFO] GE(200,python3):2026-09-04-17:53:57.759 [stream_allocator.cc:45]2003 AcquireStreams:Collect rt2 stream, get rts stream 0x101 from logical stream 1, rts_stream_id: 54
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_subgraphs_multistream.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertTrue(
            any(item["session_type"] == "compile_graph" for item in data["sessions"])
        )
        self.assertFalse(
            any(
                item.get("runtime", {}).get("dynamic_subgraph_fallback")
                for item in data["sessions"]
                if item.get("runtime")
            )
        )

    def test_dynamic_root_without_summary_matches_executor_by_graph_name(self):
        content = """\
[INFO] GE(100,atc):2026-09-04-17:52:51.000 [graph_builder.cc:460]1001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[ge_default_20260904175251_dynamic]
[INFO] GE(200,python3):2026-09-04-17:53:57.700 [model_v2_executor_builder.cc:158]2001 Build:Build RT2 executor for root compute graph[ge_default_20260904175251_dynamic], model[ge_default_20260904175251_dynamic].
[INFO] GE(200,python3):2026-09-04-17:53:57.758 [model_v2_executor.cc:120]2002 OccupyStreamResource:Collect rt2 stream, get rts stream 0x100 from logical stream 0, rts_stream_id: 53
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "dynamic_root_without_summary.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertEqual(
            [item["session_type"] for item in data["sessions"]], ["runtime_model"]
        )
        runtime = data["sessions"][0]
        self.assertEqual(runtime["graph"], "ge_default_20260904175251_dynamic")
        self.assertEqual(runtime["correlation"], "dynamic_root_compile_missing")
        self.assertIn("executor_graph_exact_match", runtime["correlation_evidence"])
        self.assertEqual(runtime["analysis_status"], "partial")

    def test_v2_dynamic_summary_graph_is_eligible_even_when_name_looks_like_subgraph(
        self,
    ):
        content = """\
[INFO] GE(302,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]6001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[root_dynamic_sub_1_unknow].
[INFO] GE(302,python3):2026-08-27-10:00:00.010 [dynamic_stream_allocator.cc:73]6001 AssignStreamsForDynamicShapeGraph:Graph: root_dynamic_sub_1_unknow, stream num:2, event num:0.
[INFO] GE(302,python3):2026-08-27-10:00:01.000 [stream.cc:55]6002 SplitRtStreams:[KernelTrace][SplitRtStreams_1]Get rts stream 0x1 from logical stream 0, rts_stream_id: 71
[INFO] GE(302,python3):2026-08-27-10:00:01.001 [stream.cc:55]6002 SplitRtStreams:[KernelTrace][SplitRtStreams_1]Get rts stream 0x2 from logical stream 1, rts_stream_id: 72
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_dynamic_named_subgraph.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["graph"], "root_dynamic_sub_1_unknow")
        self.assertEqual(
            [row["rt_stream_id"] for row in models[0]["runtime"]["mapping_rows"]],
            [71, 72],
        )

    def test_v2_resource_and_trace_fragments_merge_into_dynamic_workflow(self):
        content = """\
[INFO] GE(303,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]7001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dynamic_root].
[INFO] GE(303,python3):2026-08-27-10:00:00.010 [dynamic_stream_allocator.cc:73]7001 AssignStreamsForDynamicShapeGraph:Graph: dynamic_root, stream num:2, event num:0.
[INFO] GE(303,python3):2026-08-27-10:00:01.000 [model_converter.cc:366]7002 GetReusableStreamResourceNum:Root graph total stream_num 2, reusable stream num 2, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(303,python3):2026-08-27-10:00:01.010 [stream.cc:55]7003 SplitRtStreams:[KernelTrace][SplitRtStreams_1]Get rts stream 0x1 from logical stream 0, rts_stream_id: 81
[INFO] GE(303,python3):2026-08-27-10:00:01.011 [stream.cc:55]7003 SplitRtStreams:[KernelTrace][SplitRtStreams_1]Get rts stream 0x2 from logical stream 1, rts_stream_id: 82
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_dynamic_resource_and_trace.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        self.assertEqual(len(models), 1)
        self.assertFalse(
            any(item["session_type"] == "runtime_model" for item in data["sessions"])
        )
        self.assertEqual(models[0]["runtime"]["reusable_stream_count"], 2)
        self.assertEqual(
            [row["rt_stream_id"] for row in models[0]["runtime"]["mapping_rows"]],
            [81, 82],
        )

    def test_v2_same_pid_and_index_set_keeps_multiple_dynamic_candidates_ambiguous(
        self,
    ):
        content = """\
[INFO] GE(304,python3):2026-08-27-10:00:00.000 [graph_builder.cc:462]8001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dynamic_a].
[INFO] GE(304,python3):2026-08-27-10:00:00.010 [dynamic_stream_allocator.cc:73]8001 AssignStreamsForDynamicShapeGraph:Graph: dynamic_a, stream num:2, event num:0.
[INFO] GE(304,python3):2026-08-27-10:00:08.000 [graph_builder.cc:462]8001 BuildForUnknownShapeGraph:Begin to build unknown shape graph[dynamic_b].
[INFO] GE(304,python3):2026-08-27-10:00:08.010 [dynamic_stream_allocator.cc:73]8001 AssignStreamsForDynamicShapeGraph:Graph: dynamic_b, stream num:2, event num:0.
[INFO] GE(304,python3):2026-08-27-10:00:10.000 [stream.cc:55]8002 SplitRtStreams:[KernelTrace][SplitRtStreams_1]Get rts stream 0x1 from logical stream 0, rts_stream_id: 91
[INFO] GE(304,python3):2026-08-27-10:00:10.001 [stream.cc:55]8002 SplitRtStreams:[KernelTrace][SplitRtStreams_1]Get rts stream 0x2 from logical stream 1, rts_stream_id: 92
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_dynamic_ambiguous.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        self.assertFalse(
            any(item["session_type"] == "model" for item in data["sessions"])
        )
        runtime = next(
            item for item in data["sessions"] if item["session_type"] == "runtime_model"
        )
        self.assertEqual(len(runtime["correlation_candidates"]), 2)

    def test_multiple_v2_trace_blocks_keep_compile_graphs_separate(self):
        content = """\
[INFO] GE(301,python3):2026-08-27-10:00:00.000 [graph_builder.cc:343]5002 BuildForKnownShapeGraph:Begin to build known shape graph[g1].
[INFO] GE(301,python3):2026-08-27-10:00:00.010 [stream_allocator.cc:677]5002 SplitStreamAndRefreshTaskDef:After SplitStreamAndRefreshTaskDef, graph:g1, stream num:1, notify num:0, event num:0.
[INFO] GE(301,python3):2026-08-27-10:00:00.040 [model_converter.cc:366]5002 GetReusableStreamResourceNum:Root graph total stream_num 1, reusable stream num 1, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(301,python3):2026-08-27-10:00:00.041 [stream.cc:55]5002 [KernelTrace][SplitRtStreams_1] Get rts stream 0x1 from logical stream 0, rts_stream_id: 51
[INFO] GE(301,python3):2026-08-27-10:00:00.042 [graph_builder.cc:343]5002 BuildForKnownShapeGraph:Begin to build known shape graph[g2].
[INFO] GE(301,python3):2026-08-27-10:00:00.043 [stream_allocator.cc:677]5002 SplitStreamAndRefreshTaskDef:After SplitStreamAndRefreshTaskDef, graph:g2, stream num:2, notify num:0, event num:0.
[INFO] GE(301,python3):2026-08-27-10:00:00.050 [model_converter.cc:366]5002 GetReusableStreamResourceNum:Root graph total stream_num 2, reusable stream num 2, attached stream num 0, event_num 0, notify_num 0.
[INFO] GE(301,python3):2026-08-27-10:00:00.051 [stream.cc:55]5002 [KernelTrace][SplitRtStreams_2] Get rts stream 0x2 from logical stream 0, rts_stream_id: 61
[INFO] GE(301,python3):2026-08-27-10:00:00.052 [stream.cc:55]5002 [KernelTrace][SplitRtStreams_2] Get rts stream 0x3 from logical stream 1, rts_stream_id: 62
"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "v2_multi.log"
            log.write_text(content)
            data = self.run_json("--log", str(log), "--repo-root", str(REPO_ROOT))
        models = [item for item in data["sessions"] if item["session_type"] == "model"]
        self.assertEqual(
            {(item["graph"], item["correlation"]) for item in models},
            {("g1", "same_session"), ("g2", "same_session")},
        )
        records = {item["graph"]: item for item in data["stream_records"]}
        self.assertEqual(
            records["g1"]["runtime"]["mapping_rows"][0]["rt_stream_id"], 51
        )
        self.assertEqual(
            [row["rt_stream_id"] for row in records["g2"]["runtime"]["mapping_rows"]],
            [61, 62],
        )

    def test_markdown_omits_input_log_list(self):
        fixture = SKILL_ROOT / "tests" / "fixtures" / "compile_static.log"
        report = self.run_markdown(
            "--compile", str(fixture), "--repo-root", str(REPO_ROOT)
        )
        self.assertNotIn("输入日志：", report)


if __name__ == "__main__":
    unittest.main()
