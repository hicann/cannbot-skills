#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for utils/gen_batch_excel.py.

Run with: python3 -m pytest tests/test_gen_batch_excel.py -v
"""

import importlib.util
import json
import os
import sys
from dataclasses import dataclass

import pytest


def _load_module():
    """Load gen_batch_excel.py as a module (utils/ is not a package)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "utils", "gen_batch_excel.py")
    spec = importlib.util.spec_from_file_location("gen_batch_excel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_batch_excel"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gbe():
    return _load_module()


@dataclass
class PerfSpec:
    """perf_result.json 的构造参数。"""
    speedup: float
    passed: int = None
    total: int = None
    fw: float = None
    impl: float = None


def _write_perf(path, spec: PerfSpec):
    d = {"speedup_vs_torch": spec.speedup}
    if spec.passed is not None and spec.total is not None:
        d["passed_cases"] = spec.passed
        d["total_cases"] = spec.total
    d["framework"] = {"avg_latency_ms": spec.fw} if spec.fw is not None else {}
    d["implementation"] = {"avg_latency_ms": spec.impl} if spec.impl is not None else {}
    path.write_text(json.dumps(d))


def _write_verify(path, passed, total, failures=0):
    path.write_text(json.dumps({
        "op_name": "test_op",
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failures,
        "failures": [],
    }))


@dataclass
class IterSpec:
    """一个迭代目录的构造参数。"""
    name: str
    speedup: float
    passed: int = 50
    total: int = 50
    verify: bool = True


def _make_iter(root, spec: IterSpec):
    """建 output/<name>/：perf_result.json + verify/verify_result.json。"""
    it = root / "output" / spec.name
    it.mkdir(parents=True)
    if spec.verify:
        vd = it / "verify"
        vd.mkdir()
        _write_verify(vd / "verify_result.json", spec.passed, spec.total)
    _write_perf(it / "perf_result.json",
                PerfSpec(spec.speedup, spec.passed, spec.total))


class TestParseReportMd:
    @staticmethod
    def test_parses_pass_rate(gbe, tmp_path):
        md = tmp_path / "report.md"
        md.write_text(
            "# t\n\n"
            "**算子名称**: 97_Op\n"
            "**目标加速比**: 100.0\n"
            "**通过率**: 50/50 (100%)\n"
            "**框架平均延迟**: 0.0938 ms\n"
            "**实现平均延迟**: 0.1310 ms\n"
            "**几何平均加速比**: 0.7158x\n"
        )
        d = gbe.parse_report_md(md)
        assert d["op_name"] == "97_Op"
        assert d["passed_cases"] == 50
        assert d["total_cases"] == 50
        assert d["framework_latency_ms"] == "0.0938"
        assert d["geo_speedup"] == "0.7158"


class TestParsePerfResultJson:
    @staticmethod
    def test_missing_file_returns_empty(gbe, tmp_path):
        assert gbe.parse_perf_result_json(tmp_path / "nope.json") == {}

    @staticmethod
    def test_extracts_latency_and_speedup(gbe, tmp_path):
        path = tmp_path / "perf_result.json"
        path.write_text(json.dumps({
            "speedup_vs_torch": 2.5,
            "framework": {"avg_latency_ms": 0.2},
            "implementation": {"avg_latency_ms": 0.08},
        }))
        d = gbe.parse_perf_result_json(path)
        assert d == {
            "framework_latency_ms": 0.2,
            "impl_latency_ms": 0.08,
            "geo_speedup": 2.5,
        }


class TestParseSummaryJson:
    @staticmethod
    def test_parses_perf_data(gbe, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({
            "success": True,
            "target_reached": False,
            "perf_data": {
                "avg_latency_ms": 0.131,
                "passed_cases": 48,
                "total_cases": 50,
                "per_shape_results": [{"framework_avg_latency_ms": 0.09}],
            },
        }))
        d = gbe.parse_summary_json(path)
        assert d["passed_cases"] == 48
        assert d["total_cases"] == 50
        assert d["impl_latency_ms"] == 0.131
        assert d["framework_latency_ms"] == 0.09


class TestParseOutputFallback:
    @staticmethod
    def test_selects_max_speedup(gbe, tmp_path):
        _make_iter(tmp_path, IterSpec("iter_0", 1.2))
        _make_iter(tmp_path, IterSpec("opt_iter_0", 2.5))
        d, src = gbe.parse_output_fallback(tmp_path)
        assert src == "output/opt_iter_0/"
        assert d["geo_speedup"] == 2.5
        assert d["passed_cases"] == 50 and d["total_cases"] == 50

    @staticmethod
    def test_tie_prefers_newer_opt_iter(gbe, tmp_path):
        _make_iter(tmp_path, IterSpec("opt_iter_0", 2.5))
        _make_iter(tmp_path, IterSpec("opt_iter_1", 2.5))
        d, src = gbe.parse_output_fallback(tmp_path)
        assert src == "output/opt_iter_1/"

    @staticmethod
    def test_optimized_perf_uses_optimized_verify(gbe, tmp_path):
        it = tmp_path / "output" / "opt_iter_0"
        it.mkdir(parents=True)
        vd = it / "verify"
        vd.mkdir()
        # baseline 与 optimized 各有独立 verify
        _write_verify(vd / "verify_result_baseline.json", 30, 50)
        _write_verify(vd / "verify_result_optimized.json", 50, 50)
        _write_perf(it / "baseline_perf_result.json", PerfSpec(1.5, 50, 50))
        _write_perf(it / "optimized_perf_result.json", PerfSpec(3.0, 50, 50))

        d, src = gbe.parse_output_fallback(tmp_path)
        assert src == "output/opt_iter_0/"
        assert d["geo_speedup"] == 3.0
        # 精度取自 optimized 对应的 verify，而非 baseline
        assert d["passed_cases"] == 50 and d["total_cases"] == 50

    @staticmethod
    def test_baseline_verify_mapping_when_baseline_is_max(gbe, tmp_path):
        it = tmp_path / "output" / "opt_iter_0"
        it.mkdir(parents=True)
        vd = it / "verify"
        vd.mkdir()
        _write_verify(vd / "verify_result_baseline.json", 49, 50)
        _write_verify(vd / "verify_result_optimized.json", 50, 50)
        _write_perf(it / "baseline_perf_result.json", PerfSpec(5.0, 50, 50))
        _write_perf(it / "optimized_perf_result.json", PerfSpec(3.0, 50, 50))

        d, src = gbe.parse_output_fallback(tmp_path)
        assert d["geo_speedup"] == 5.0
        assert d["passed_cases"] == 49 and d["total_cases"] == 50

    @staticmethod
    def test_verify_missing_falls_back_to_perf_passed(gbe, tmp_path):
        it = tmp_path / "output" / "iter_0"
        it.mkdir(parents=True)
        _write_perf(it / "perf_result.json", PerfSpec(2.0, passed=40, total=40))
        d, src = gbe.parse_output_fallback(tmp_path)
        assert d["passed_cases"] == 40 and d["total_cases"] == 40

    @staticmethod
    def test_empty_output_returns_empty(gbe, tmp_path):
        assert gbe.parse_output_fallback(tmp_path) == ({}, "")
        empty = tmp_path / "output"
        empty.mkdir()
        assert gbe.parse_output_fallback(tmp_path) == ({}, "")

    @staticmethod
    def test_skips_perf_without_speedup(gbe, tmp_path):
        it = tmp_path / "output" / "iter_0"
        it.mkdir(parents=True)
        (it / "perf_result.json").write_text(json.dumps({"speedup_vs_torch": None}))
        assert gbe.parse_output_fallback(tmp_path) == ({}, "")

    @staticmethod
    def test_skips_malformed_perf(gbe, tmp_path):
        it = tmp_path / "output" / "iter_0"
        it.mkdir(parents=True)
        (it / "perf_result.json").write_text("{not json")
        assert gbe.parse_output_fallback(tmp_path) == ({}, "")


def _build_row(gbe, tmp_path, op_file, status="✅ 成功", elapsed=10):
    op = {"id": 1, "file": op_file, "status": status, "elapsed": elapsed}
    ctx = gbe.ReportContext(
        output_dir=tmp_path,
        header={"npu": "11", "npu_list": "11"},
        arch_full="ascend910b2",
        level="level4",
        target_speedup=2.0,
    )
    return gbe.build_row(op, ctx)


class TestBuildRowDataSource:
    @staticmethod
    def test_report_md_source(gbe, tmp_path):
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        (op_dir / "report.md").write_text(
            "# t\n**通过率**: 50/50 (100%)\n"
            "**框架平均延迟**: 0.1 ms\n**实现平均延迟**: 0.05 ms\n"
            "**几何平均加速比**: 2.0x\n"
        )
        row = _build_row(gbe, tmp_path, "op.py")
        assert row["data_source"] == "report.md"
        assert row["precision"] == "50/50"
        assert row["precision_ok"] is True
        assert row["speedup"] == 2.0

    @staticmethod
    def test_summary_json_source_when_no_report(gbe, tmp_path):
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        (op_dir / "summary.json").write_text(json.dumps({
            "perf_data": {
                "avg_latency_ms": 0.13,
                "passed_cases": 50,
                "total_cases": 50,
            },
        }))
        row = _build_row(gbe, tmp_path, "op.py")
        assert row["data_source"] == "summary.json"
        assert row["precision"] == "50/50"
        assert row["precision_ok"] is True

    @staticmethod
    def test_fallback_when_no_report_and_no_summary(gbe, tmp_path):
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        _make_iter(op_dir, IterSpec("iter_0", 1.2))
        _make_iter(op_dir, IterSpec("opt_iter_0", 2.6))
        row = _build_row(gbe, tmp_path, "op.py")
        assert row["data_source"] == "output/opt_iter_0/"
        assert row["speedup"] == 2.6
        assert row["precision"] == "50/50"
        assert row["precision_ok"] is True
        assert row["ref_latency_ms"] is None  # fallback 未提供延迟
        assert row["impl_latency_ms"] is None

    @staticmethod
    def test_fallback_fills_latency_from_perf(gbe, tmp_path):
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        it = op_dir / "output" / "iter_0"
        it.mkdir(parents=True)
        vd = it / "verify"
        vd.mkdir()
        _write_verify(vd / "verify_result.json", 50, 50)
        _write_perf(it / "perf_result.json", PerfSpec(2.5, 50, 50, fw=0.2, impl=0.08))
        row = _build_row(gbe, tmp_path, "op.py")
        assert row["data_source"] == "output/iter_0/"
        assert row["ref_latency_ms"] == 0.2
        assert row["impl_latency_ms"] == 0.08
        assert row["precision_ok"] is True

    @staticmethod
    def test_fallback_partial_precision_not_ok(gbe, tmp_path):
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        _make_iter(op_dir, IterSpec("iter_0", 2.0, passed=30, total=50))
        row = _build_row(gbe, tmp_path, "op.py")
        assert row["precision"] == "30/50"
        assert row["precision_ok"] is False

    @staticmethod
    def test_missing_op_dir_returns_empty_row(gbe, tmp_path):
        row = _build_row(gbe, tmp_path, "ghost.py")
        assert row["data_source"] == ""
        assert row["precision_ok"] is False
        assert row["speedup"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
