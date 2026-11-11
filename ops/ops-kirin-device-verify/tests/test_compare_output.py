#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""compare_output.py 的判定边界回归（合成数据，无 NPU 依赖）。

重点固化 bitwise / tolerance 两条判定路径的边界语义：
- bitwise：按位全一致，+0/-0 与 NaN payload 差异均判 FAIL
- tolerance：严格大于阈值才判错；atol/rtol 需同时超限；NaN 同位一致、Inf 须同号
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_output.py"
SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
SPEC = importlib.util.spec_from_file_location("compare_output", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def run_report(func, *args):
    """执行以 sys.exit 收尾的 report_*，返回 (退出码, stdout)。"""
    stream = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with contextlib.redirect_stdout(stream):
            with unittest.TestCase().assertRaises(SystemExit) as ctx:
                func(*args)
    return ctx.exception.code, stream.getvalue()


class AsBitsTest(unittest.TestCase):
    def test_view_width_matches_itemsize(self) -> None:
        cases = [
            (np.array([True, False], dtype=np.bool_), np.dtype(np.uint8)),
            (np.array([0, 255], dtype=np.uint8), np.dtype(np.uint8)),
            (np.array([-1, 1], dtype=np.int8), np.dtype(np.uint8)),
            (np.array([1.0], dtype=np.float16), np.dtype(np.uint16)),
            (np.array([-1, 1], dtype=np.int16), np.dtype(np.uint16)),
            (np.array([1.0], dtype=np.float32), np.dtype(np.uint32)),
            (np.array([-1, 1], dtype=np.int64), np.dtype(np.uint64)),
        ]
        for array, expected in cases:
            with self.subTest(dtype=array.dtype):
                self.assertEqual(MODULE.as_bits(array).dtype, expected)

    def test_non_contiguous_input_is_made_contiguous(self) -> None:
        source = np.arange(12, dtype=np.uint8).reshape(3, 4)[:, ::2]
        bits = MODULE.as_bits(source)
        self.assertTrue(bits.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(
            bits, np.array([[0, 2], [4, 6], [8, 10]], dtype=np.uint8)
        )


class LoadBinTest(unittest.TestCase):
    def test_reads_expected_element_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            np.array([1, 2, 3, 4], dtype=np.uint8).tofile(path)
            loaded = MODULE.load_bin(str(path), np.uint8, (2, 2))
        np.testing.assert_array_equal(loaded, np.array([1, 2, 3, 4], dtype=np.uint8))

    def test_size_mismatch_exits_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.bin"
            np.array([1, 2, 3], dtype=np.uint8).tofile(path)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                with self.assertRaises(SystemExit) as ctx:
                    MODULE.load_bin(str(path), np.uint8, (2, 2))
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("字节数自校验失败", stream.getvalue())


class BitwiseBoundaryTest(unittest.TestCase):
    def test_identical_is_pass(self) -> None:
        result = np.array([0, 1, 2, 255], dtype=np.uint8)
        golden = result.copy()
        code, out = run_report(
            MODULE.report_bitwise,
            result,
            golden,
            result.astype(np.float64),
            golden.astype(np.float64),
            result.size,
            "uint8",
        )
        self.assertEqual(code, 0)
        self.assertIn("PASS (bitwise 全一致)", out)

    def test_single_bit_flip_is_fail(self) -> None:
        result = np.array([0, 1, 2, 3], dtype=np.uint8)
        golden = np.array([0, 1, 2, 2], dtype=np.uint8)
        code, out = run_report(
            MODULE.report_bitwise,
            result,
            golden,
            result.astype(np.float64),
            golden.astype(np.float64),
            result.size,
            "uint8",
        )
        self.assertEqual(code, 1)
        self.assertRegex(out, r"反转元素数量\s*:\s*1")
        self.assertRegex(out, r"反转元素比例\s*:\s*0\.250000")
        self.assertIn("FAIL (1/4 不一致)", out)

    def test_bool_reversal_count(self) -> None:
        result = np.array([True, False, True], dtype=np.bool_)
        golden = np.array([True, True, False], dtype=np.bool_)
        code, out = run_report(
            MODULE.report_bitwise,
            result,
            golden,
            result.astype(np.float64),
            golden.astype(np.float64),
            result.size,
            "bool",
        )
        self.assertEqual(code, 1)
        self.assertRegex(out, r"反转元素数量\s*:\s*2")
        self.assertRegex(out, r"golden True\(1\)\s*:\s*2")
        self.assertRegex(out, r"output True\(1\)\s*:\s*2")

    def test_float_positive_and_negative_zero_differ(self) -> None:
        result = np.array([0.0], dtype=np.float32)
        golden = np.array([-0.0], dtype=np.float32)
        code, out = run_report(
            MODULE.report_bitwise,
            result,
            golden,
            result.astype(np.float64),
            golden.astype(np.float64),
            result.size,
            "float32",
        )
        self.assertEqual(code, 1)
        self.assertRegex(out, r"反转元素数量\s*:\s*1")

    def test_float_nan_payload_mismatch_fails(self) -> None:
        result = np.array([0x7E00], dtype=np.uint16).view(np.float16)
        golden = np.array([0x7C01], dtype=np.uint16).view(np.float16)
        code, _ = run_report(
            MODULE.report_bitwise,
            result,
            golden,
            result.astype(np.float64),
            golden.astype(np.float64),
            result.size,
            "float16",
        )
        self.assertEqual(code, 1)

    def test_float_identical_nan_payload_passes(self) -> None:
        value = np.array([0x7E00], dtype=np.uint16).view(np.float16)
        result = value.copy()
        golden = value.copy()
        code, _ = run_report(
            MODULE.report_bitwise,
            result,
            golden,
            result.astype(np.float64),
            golden.astype(np.float64),
            result.size,
            "float16",
        )
        self.assertEqual(code, 0)


class ToleranceBoundaryTest(unittest.TestCase):
    def _run(self, result, golden, rtol, atol):
        result = np.asarray(result, dtype=np.float64)
        golden = np.asarray(golden, dtype=np.float64)
        return run_report(
            MODULE.report_tolerance,
            result,
            golden,
            result,
            golden,
            result.size,
            rtol,
            atol,
        )

    def test_equal_to_both_thresholds_passes(self) -> None:
        code, out = self._run([1.125], [1.0], rtol=0.125, atol=0.125)
        self.assertEqual(code, 0)
        self.assertIn("结论               : PASS", out)

    def test_exceeding_both_thresholds_fails(self) -> None:
        code, out = self._run([1.25], [1.0], rtol=0.125, atol=0.125)
        self.assertEqual(code, 1)
        self.assertRegex(out, r"不匹配元素\s*:\s*1/1")

    def test_diff_over_atol_but_rel_within_rtol_passes(self) -> None:
        code, _ = self._run([1025.0], [1024.0], rtol=0.01, atol=0.5)
        self.assertEqual(code, 0)

    def test_diff_within_atol_but_rel_over_rtol_passes(self) -> None:
        code, _ = self._run([0.75], [0.5], rtol=0.1, atol=0.5)
        self.assertEqual(code, 0)

    def test_same_position_nan_passes(self) -> None:
        code, _ = self._run([np.nan], [np.nan], rtol=1e-3, atol=1e-3)
        self.assertEqual(code, 0)

    def test_one_sided_nan_fails(self) -> None:
        code, _ = self._run([np.nan], [1.0], rtol=1e-3, atol=1e-3)
        self.assertEqual(code, 1)

    def test_same_sign_inf_passes(self) -> None:
        code, _ = self._run([np.inf], [np.inf], rtol=1e-3, atol=1e-3)
        self.assertEqual(code, 0)

    def test_opposite_sign_inf_fails(self) -> None:
        code, _ = self._run([np.inf], [-np.inf], rtol=1e-3, atol=1e-3)
        self.assertEqual(code, 1)

    def test_inf_vs_finite_fails(self) -> None:
        code, _ = self._run([np.inf], [1.0], rtol=1e-3, atol=1e-3)
        self.assertEqual(code, 1)

    def test_positive_and_negative_zero_pass(self) -> None:
        code, _ = self._run([0.0], [-0.0], rtol=1e-3, atol=1e-3)
        self.assertEqual(code, 0)

    def test_zero_golden_uses_eps_floor(self) -> None:
        small_code, _ = self._run([1e-4], [0.0], rtol=0.1, atol=1e-3)
        large_code, _ = self._run([2e-3], [0.0], rtol=0.1, atol=1e-3)
        self.assertEqual(small_code, 0)
        self.assertEqual(large_code, 1)

    def test_negative_golden_uses_absolute_value(self) -> None:
        boundary_code, _ = self._run([-1.125], [-1.0], rtol=0.125, atol=0.125)
        exceed_code, _ = self._run([-1.25], [-1.0], rtol=0.125, atol=0.125)
        self.assertEqual(boundary_code, 0)
        self.assertEqual(exceed_code, 1)


class CliTest(unittest.TestCase):
    def _run_cli(self, result, golden, shape, dtype, mode):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "out.bin"
            golden_path = Path(tmp) / "golden.bin"
            np.asarray(result).tofile(result_path)
            np.asarray(golden).tofile(golden_path)
            cmd = [
                sys.executable,
                str(SCRIPT),
                str(result_path),
                str(golden_path),
                "--shape",
                shape,
                "--dtype",
                dtype,
                "--mode",
                mode,
            ]
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=SUBPROCESS_ENV,
                check=False,
            )

    def test_bitwise_mismatch_returns_1(self) -> None:
        proc = self._run_cli(
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([1, 2, 3, 5], dtype=np.uint8),
            "2,2",
            "uint8",
            "bitwise",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("FAIL", proc.stdout)

    def test_tolerance_pass_returns_0(self) -> None:
        proc = self._run_cli(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            "2",
            "float32",
            "tolerance",
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("PASS", proc.stdout)

    def test_size_mismatch_returns_1(self) -> None:
        proc = self._run_cli(
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([1, 2, 3], dtype=np.uint8),
            "2,2",
            "uint8",
            "bitwise",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("字节数自校验失败", proc.stdout)

    def test_missing_required_mode_returns_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "out.bin"
            golden_path = Path(tmp) / "golden.bin"
            np.array([1], dtype=np.uint8).tofile(result_path)
            np.array([1], dtype=np.uint8).tofile(golden_path)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(result_path),
                    str(golden_path),
                    "--shape",
                    "1",
                    "--dtype",
                    "uint8",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=SUBPROCESS_ENV,
                check=False,
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--mode", proc.stderr)


if __name__ == "__main__":
    unittest.main()
