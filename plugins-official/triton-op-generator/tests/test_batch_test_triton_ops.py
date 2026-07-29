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

"""Unit tests for utils/batch_test_triton_ops.py.

Run with: python3 -m pytest tests/test_batch_test_triton_ops.py -v
"""

import csv
import importlib.util
import io
import json
import logging
import os
import sys
from contextlib import contextmanager

import pytest


def _load_module():
    """Load batch_test_triton_ops.py as a module (utils/ is not a package)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "utils", "batch_test_triton_ops.py")
    spec = importlib.util.spec_from_file_location("batch_test_triton_ops", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_test_triton_ops"] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _capture_log(target_logger, level=logging.INFO):
    """Attach a temporary StringIO handler to capture logger output in tests.

    batch_test_triton_ops 的 logger 在 import 时绑定了真实 stdout/stderr，
    redirect_stdout / capsys 无法捕获，需直接挂 handler。
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    target_logger.addHandler(handler)
    try:
        yield buf
    finally:
        target_logger.removeHandler(handler)


@pytest.fixture(scope="module")
def bt():
    return _load_module()


class TestWalkExcludingNoise:
    @staticmethod
    def test_skips_session_dir_and_pycache(bt, tmp_path):
        (tmp_path / "session_dir").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "keep").mkdir()
        (tmp_path / "keep" / "file.txt").write_text("x")

        visited = list(bt.walk_excluding_noise(str(tmp_path)))
        dirnames = {os.path.basename(d) for d, _, _ in visited}
        filenames = [f for _, _, fs in visited for f in fs]

        assert "session_dir" not in dirnames
        assert "__pycache__" not in dirnames
        assert "keep" in dirnames
        assert filenames == ["file.txt"]


class TestFindTaskFile:
    @staticmethod
    def test_single_py_file(bt, tmp_path):
        (tmp_path / "foo.py").write_text("x")
        assert bt.find_task_file(str(tmp_path)) == str(tmp_path / "foo.py")

    @staticmethod
    def test_prefer_dir_name_match(bt, tmp_path):
        op_dir = tmp_path / "mydir"
        op_dir.mkdir()
        (op_dir / "bar.py").write_text("x")
        (op_dir / "mydir.py").write_text("x")
        assert bt.find_task_file(str(op_dir)) == str(op_dir / "mydir.py")

    @staticmethod
    def test_prefer_model_or_get_inputs(bt, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("class Model:\n    pass\n")
        assert bt.find_task_file(str(tmp_path)) == str(tmp_path / "b.py")

    @staticmethod
    def test_skip_generated_and_triton_files(bt, tmp_path):
        (tmp_path / "op_generated.py").write_text("x")
        (tmp_path / "op_triton_ascend.py").write_text("x")
        (tmp_path / "op.py").write_text("x")
        assert bt.find_task_file(str(tmp_path)) == str(tmp_path / "op.py")

    @staticmethod
    def test_no_candidate_returns_none(bt, tmp_path):
        assert bt.find_task_file(str(tmp_path)) is None


class TestFindFallbackImpl:
    @staticmethod
    def test_opt_iter_uses_optimized_file(bt, tmp_path):
        task = tmp_path / "op.py"
        task.write_text("class Model:\n    pass\n")
        opt = tmp_path / "output" / "opt_iter_0"
        opt.mkdir(parents=True)
        verify_dir = opt / "verify"
        verify_dir.mkdir()
        (opt / "optimized_code.py").write_text("# opt")
        (verify_dir / "op_triton_optimized.py").write_text("# opt impl")
        (verify_dir / "verify_result_optimized.json").write_text(json.dumps({
            "total_cases": 2,
            "passed_cases": 2,
            "failed_cases": 0,
        }))
        (opt / "optimized_perf_result.json").write_text(json.dumps({
            "speedup_vs_torch": 3.0,
        }))

        info = bt.find_fallback_impl(str(tmp_path))
        assert info is not None
        assert "_triton_optimized.py" in info["gen_file"]

    @staticmethod
    def _make_iter(root, name="iter_0", passed=True, speedup=1.5):
        it = root / "output" / name
        it.mkdir(parents=True)
        verify_dir = it / "verify"
        verify_dir.mkdir()
        (it / "generated_code.py").write_text("# generated")
        (verify_dir / "op_triton_ascend_impl.py").write_text("# impl")
        (verify_dir / "verify_result.json").write_text(json.dumps({
            "total_cases": 4,
            "passed_cases": 4 if passed else 2,
            "failed_cases": 0 if passed else 2,
        }))
        (it / "perf_result.json").write_text(json.dumps({
            "speedup_vs_torch": speedup,
        }))
        return it

    def test_selects_best_passed_candidate(self, bt, tmp_path):
        task = tmp_path / "op.py"
        task.write_text("class Model:\n    pass\n")
        self._make_iter(tmp_path, "iter_0", passed=True, speedup=1.2)
        self._make_iter(tmp_path, "iter_1", passed=True, speedup=2.0)

        info = bt.find_fallback_impl(str(tmp_path))
        assert info is not None
        assert "iter_1" in info["gen_file"]
        assert info["op_name"] == os.path.basename(str(tmp_path))

    def test_skips_unpassed_candidates(self, bt, tmp_path):
        task = tmp_path / "op.py"
        task.write_text("class Model:\n    pass\n")
        self._make_iter(tmp_path, "iter_0", passed=False, speedup=5.0)
        self._make_iter(tmp_path, "iter_1", passed=True, speedup=1.0)

        info = bt.find_fallback_impl(str(tmp_path))
        assert info is not None
        assert "iter_1" in info["gen_file"]

    def test_returns_none_when_no_passed(self, bt, tmp_path):
        task = tmp_path / "op.py"
        task.write_text("class Model:\n    pass\n")
        self._make_iter(tmp_path, "iter_0", passed=False, speedup=1.0)

        assert bt.find_fallback_impl(str(tmp_path)) is None


class TestFindOperators:
    @staticmethod
    def test_direct_generated(bt, tmp_path):
        level = tmp_path / "level1"
        level.mkdir()
        op_dir = level / "31_op"
        op_dir.mkdir()
        (op_dir / "op_generated.py").write_text("# gen")
        (op_dir / "op.py").write_text("class Model:\n    pass\n")

        ops = bt.find_operators([str(level)])
        assert len(ops) == 1
        assert ops[0]["op_name"] == "op"
        assert ops[0]["gen_file"].endswith("op_generated.py")

    @staticmethod
    def test_nested_workdir(bt, tmp_path):
        level = tmp_path / "level1"
        level.mkdir()
        op_dir = level / "31_op"
        op_dir.mkdir()
        nested = op_dir / "op_20260723_1200_1234"
        nested.mkdir()
        (nested / "op_generated.py").write_text("# gen")
        (nested / "op.py").write_text("class Model:\n    pass\n")

        ops = bt.find_operators([str(level)])
        assert len(ops) == 1
        assert ops[0]["gen_file"].endswith("op_generated.py")

    @staticmethod
    def test_fallback_impl(bt, tmp_path):
        level = tmp_path / "level1"
        level.mkdir()
        op_dir = level / "31_op"
        op_dir.mkdir()
        (op_dir / "op.py").write_text("class Model:\n    pass\n")
        it = op_dir / "output" / "iter_0"
        it.mkdir(parents=True)
        verify_dir = it / "verify"
        verify_dir.mkdir()
        (it / "generated_code.py").write_text("# gen")
        (verify_dir / "verify_result.json").write_text(json.dumps({
            "total_cases": 2,
            "passed_cases": 2,
            "failed_cases": 0,
        }))
        (it / "perf_result.json").write_text(json.dumps({
            "speedup_vs_torch": 1.5,
        }))

        ops = bt.find_operators([str(level)])
        assert len(ops) == 1
        assert ops[0]["gen_file"].endswith("generated_code.py")

    @staticmethod
    def test_warns_when_no_candidate(bt, tmp_path):
        level = tmp_path / "level1"
        level.mkdir()
        op_dir = level / "31_op"
        op_dir.mkdir()

        with _capture_log(bt.logger, level=logging.WARNING) as buf:
            ops = bt.find_operators([str(level)])
        assert ops == []
        assert "未找到生成代码" in buf.getvalue()


class TestLoadJson:
    @staticmethod
    def test_missing_file_returns_none(bt, tmp_path):
        assert bt.load_json(str(tmp_path / "nope.json")) is None

    @staticmethod
    def test_invalid_json_returns_load_error(bt, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        data = bt.load_json(str(path))
        assert "_load_error" in data

    @staticmethod
    def test_valid_json_parsed(bt, tmp_path):
        path = tmp_path / "good.json"
        path.write_text(json.dumps({"a": 1, "b": [2, 3]}))
        assert bt.load_json(str(path)) == {"a": 1, "b": [2, 3]}


class TestSummarizeFailures:
    @staticmethod
    def test_empty_failures(bt):
        assert bt.summarize_failures({"failures": []}) == ""
        assert bt.summarize_failures(None) == ""

    @staticmethod
    def test_truncates_and_joins(bt):
        failures = [
            {"case_idx": 1, "error_type": "AccuracyError", "error_msg": "a" * 200},
            {"case_idx": 2, "error_type": "MLIRCompilationError", "error_msg": "compile\nerror"},
        ]
        summary = bt.summarize_failures({"failures": failures}, max_items=5)
        assert "case 1: AccuracyError" in summary
        assert "case 2: MLIRCompilationError" in summary
        assert "compile error" in summary
        assert "\n" not in summary

    @staticmethod
    def test_limits_items(bt):
        failures = [{"case_idx": i, "error_type": "E", "error_msg": "x"} for i in range(10)]
        summary = bt.summarize_failures({"failures": failures}, max_items=3)
        assert summary.count("case") == 3
        assert "7 more" in summary


class TestWriteCsv:
    @staticmethod
    def test_write_csv_creates_header_and_rows(bt, tmp_path):
        path = tmp_path / "out.csv"
        results = [
            {"level": "l1", "dir_name": "d1", "op_name": "o1", "extra": "ignored"},
            {"level": "l1", "dir_name": "d2", "op_name": "o2", "extra": "ignored"},
        ]
        bt.write_csv(results, str(path))

        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["op_name"] == "o1"
        assert rows[1]["op_name"] == "o2"
        assert "extra" not in rows[0]

    @staticmethod
    def test_write_csv_row_appends(bt, tmp_path):
        path = tmp_path / "out.csv"
        fieldnames = ["level", "op_name"]
        bt.write_csv_row({"level": "l1", "op_name": "o1"}, str(path), fieldnames)
        bt.write_csv_row({"level": "l2", "op_name": "o2"}, str(path), fieldnames)

        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["op_name"] for r in rows] == ["o1", "o2"]


class TestPrintSummary:
    @staticmethod
    def test_basic_summary(bt):
        results = [
            {
                "level": "l1", "dir_name": "d1", "op_name": "o1",
                "verify_status": "PASS", "benchmark_status": "PASS",
                "speedup_vs_torch": 1.5, "target_speedup": 0.8,
                "target_reached": True, "verify_error_summary": "",
            },
            {
                "level": "l1", "dir_name": "d2", "op_name": "o2",
                "verify_status": "FAIL", "benchmark_status": "NOT_RUN",
                "speedup_vs_torch": None, "target_speedup": 0.8,
                "target_reached": False, "verify_error_summary": "case 1: AccuracyError",
            },
        ]
        with _capture_log(bt.logger) as buf:
            bt.print_summary(results)
        out = buf.getvalue()
        assert "总算子数: 2" in out
        assert "精度验证通过: 1" in out
        assert "精度验证失败: 1" in out
        assert "d2: FAIL" in out

    @staticmethod
    def test_top_and_bottom_speedups(bt):
        results = [
            {
                "level": "l1", "dir_name": f"d{i}", "op_name": f"o{i}",
                "verify_status": "PASS", "benchmark_status": "PASS",
                "speedup_vs_torch": float(i), "target_speedup": 0.8,
                "target_reached": i >= 0.8, "verify_error_summary": "",
            }
            for i in range(1, 8)
        ]
        with _capture_log(bt.logger) as buf:
            bt.print_summary(results)
        out = buf.getvalue()
        assert "加速比 Top 5:" in out
        assert "加速比 Bottom 5:" in out
        assert "7.0000x" in out
        assert "1.0000x" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
