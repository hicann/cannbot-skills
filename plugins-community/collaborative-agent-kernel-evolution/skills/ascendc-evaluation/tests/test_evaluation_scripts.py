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

"""Unit tests for ascendc-evaluation scripts.

Tests importability and CPU-only functional correctness of the
evaluation utility modules. No NPU/CANN required.

Reference: AscendC-skills/tests/integration/test_scaffold_utils.py
"""

import sys
from pathlib import Path

import pytest

# Add evaluation scripts to path (this file lives in <skill>/tests/, scripts in <skill>/scripts/)
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module", autouse=True)
def scripts_path():
    """Add evaluation scripts to sys.path."""
    p = str(SCRIPTS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    yield
    if p in sys.path:
        sys.path.remove(p)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    """All utility modules are importable."""

    @staticmethod
    def test_import_constants():
        from constants import DEFAULT_TOLERANCES, DEFAULT_ULP_CONFIG
        assert isinstance(DEFAULT_TOLERANCES, dict)
        assert isinstance(DEFAULT_ULP_CONFIG, dict)

    @staticmethod
    def test_import_ulp():
        from ulp import ulp_metrics, ulp_distance_bitwise
        assert callable(ulp_metrics)
        assert callable(ulp_distance_bitwise)

    @staticmethod
    def test_import_precision():
        from precision import dual_inspect, _compute_comparison, _count_element_stats
        assert callable(dual_inspect)
        assert callable(_compute_comparison)
        assert callable(_count_element_stats)

    @staticmethod
    def test_import_datagen():
        from datagen import resolve_dtype, parse_distr, make_tensor
        assert callable(resolve_dtype)
        assert callable(parse_distr)
        assert callable(make_tensor)

    @staticmethod
    def test_import_suites():
        from suites import try_op, compare_precision, smoke_test
        assert callable(try_op)
        assert callable(compare_precision)
        assert callable(smoke_test)

    @staticmethod
    def test_import_reporting():
        from reporting import make_suffix, CheckpointCollector
        assert callable(make_suffix)


# ---------------------------------------------------------------------------
# constants.py
# ---------------------------------------------------------------------------

class TestConstants:
    @staticmethod
    def test_float_tolerances():
        import torch
        from constants import DEFAULT_TOLERANCES
        for dt in [torch.float16, torch.bfloat16, torch.float32]:
            assert dt in DEFAULT_TOLERANCES, f"Missing tolerances for {dt}"
            tol = DEFAULT_TOLERANCES[dt]
            assert "atol" in tol
            assert "rtol" in tol
            assert "sv_th" in tol
            assert "svec_ratio_limit" in tol

    @staticmethod
    def test_int_tolerances():
        import torch
        from constants import DEFAULT_TOLERANCES
        for dt in [torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint8, torch.bool]:
            assert dt in DEFAULT_TOLERANCES, f"Missing tolerances for {dt}"
            tol = DEFAULT_TOLERANCES[dt]
            assert tol["atol"] == 0
            assert tol["rtol"] == 0.0

    @staticmethod
    def test_uint_tolerances():
        import torch
        from constants import DEFAULT_TOLERANCES
        for dt in [torch.uint16, torch.uint32, torch.uint64]:
            assert dt in DEFAULT_TOLERANCES, f"Missing tolerances for {dt}"


# ---------------------------------------------------------------------------
# datagen.py
# ---------------------------------------------------------------------------

class TestDatagen:
    @staticmethod
    def test_resolve_dtype():
        import torch
        from datagen import resolve_dtype
        assert resolve_dtype("float16") == torch.float16
        assert resolve_dtype("float32") == torch.float32
        assert resolve_dtype(torch.bfloat16) == torch.bfloat16

    @staticmethod
    def test_parse_distr_normal():
        import torch
        from datagen import parse_distr
        gen = parse_distr("normal(0,1)")
        t = gen((4, 4), torch.float32)
        assert t.shape == (4, 4)
        assert t.dtype == torch.float32

    @staticmethod
    def test_parse_distr_uniform():
        import torch
        from datagen import parse_distr
        gen = parse_distr("uniform(-1,1)")
        t = gen((8,), torch.float32)
        assert t.shape == (8,)
        assert t.min() >= -1.0
        assert t.max() <= 1.0

    @staticmethod
    def test_parse_distr_invalid():
        from datagen import parse_distr
        with pytest.raises(ValueError):
            parse_distr("invalid_format")

    @staticmethod
    def test_make_tensor_reproducible():
        import torch
        from datagen import make_tensor
        t1 = make_tensor((4, 4), dtype=torch.float16, seed=42)
        t2 = make_tensor((4, 4), dtype=torch.float16, seed=42)
        assert torch.equal(t1, t2)


# ---------------------------------------------------------------------------
# ulp.py
# ---------------------------------------------------------------------------

class TestUlp:
    @staticmethod
    def test_identical_tensors():
        import torch
        from ulp import ulp_metrics
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result = ulp_metrics(a, a.clone(), dtype=torch.float16)
        assert result["ulp_max"] == 0
        assert result["ulp_mean"] == 0.0

    @staticmethod
    def test_small_diff():
        import torch
        import numpy as np
        from ulp import ulp_metrics
        a = torch.tensor([1.0], dtype=torch.float16)
        a_np = np.float16(1.0)
        b_np = np.nextafter(a_np, np.float16(2.0))
        b = torch.tensor([float(b_np)], dtype=torch.float16)
        result = ulp_metrics(a, b, dtype=torch.float16)
        assert result["ulp_max"] <= 1


# ---------------------------------------------------------------------------
# precision.py
# ---------------------------------------------------------------------------

class TestPrecision:
    @staticmethod
    def test_dual_inspect_identical_float():
        import torch
        from precision import dual_inspect, ComponentResult
        a = torch.randn(16, 16, dtype=torch.float16)
        result = dual_inspect(y_ans=a, y_ref=a.clone(), y_golden=a.double(), name="test")
        assert isinstance(result, ComponentResult)
        assert result.passed
        assert result.ans_vs_ref.ae_max == 0.0
        assert result.ans_vs_ref.mismatch_rate == 0.0

    @staticmethod
    def test_dual_inspect_identical_int():
        """Integer outputs with identical values should pass."""
        import torch
        from precision import dual_inspect, ComponentResult
        a = torch.tensor([1, 2, 3, 100, -50], dtype=torch.int32)
        result = dual_inspect(y_ans=a, y_ref=a.clone(), y_golden=a.clone(), name="int_test")
        assert isinstance(result, ComponentResult)
        assert result.passed
        assert result.ans_vs_golden.mismatch_rate == 0.0
        assert result.ans_vs_golden.ae_max == 0.0
        assert result.ratios is None  # no ratios for int

    @staticmethod
    def test_dual_inspect_int_mismatch():
        """Integer outputs with differences should fail and produce diagnosis."""
        import torch
        from precision import dual_inspect
        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        b = torch.tensor([1, 2, 4], dtype=torch.int32)  # off-by-one on last element
        result = dual_inspect(y_ans=b, y_ref=a.clone(), y_golden=a.clone(), name="int_fail")
        assert not result.passed
        assert result.ans_vs_golden.mismatch_rate > 0.0
        assert result.ans_vs_golden.ae_max == 1.0
        assert result.diagnosis is not None

    @staticmethod
    def test_count_element_stats_int():
        """Integer tensor should have nan=None, inf=None, subnormal=None, small=None."""
        import torch
        from precision import _count_element_stats
        t = torch.tensor([0, 1, -1, 127, -128], dtype=torch.int8)
        stats = _count_element_stats(t)
        assert stats.nan is None
        assert stats.inf is None
        assert stats.subnormal is None
        assert stats.small is None
        assert stats.total == 5

    @staticmethod
    def test_int_exact_equality_large_values():
        """int32 values > 2^24 should compare correctly (not lose precision)."""
        import torch
        from precision import _compute_comparison
        big = 2**24 + 1  # 16777217 — not representable in float32
        a = torch.tensor([big], dtype=torch.int32)
        b = torch.tensor([big], dtype=torch.int32)
        result = _compute_comparison(a, b, b, None, None, 0, 0.0, None, torch.int32)
        assert result.mismatch_rate == 0.0

    @staticmethod
    def test_int_exact_equality_detects_diff():
        """Large int values differing by 1 should be detected."""
        import torch
        from precision import _compute_comparison
        big = 2**24 + 1
        a = torch.tensor([big], dtype=torch.int32)
        b = torch.tensor([big + 1], dtype=torch.int32)
        result = _compute_comparison(a, b, b, None, None, 0, 0.0, None, torch.int32)
        assert result.mismatch_rate > 0.0


# ---------------------------------------------------------------------------
# reporting.py
# ---------------------------------------------------------------------------

class TestApiDescriptionGuard:
    @staticmethod
    def test_resolve_api_description_path_prefers_explicit_arg(tmp_path):
        from evaluate import resolve_api_description_path

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "api_description.md").write_text("float16", encoding="utf-8")
        explicit = tmp_path / "external_api_description.md"
        explicit.write_text("bfloat16", encoding="utf-8")

        resolved = resolve_api_description_path(work_dir, str(explicit))
        assert resolved == explicit.resolve()

    @staticmethod
    def test_validate_api_description_dtype_guard_accepts_explicit_api_desc(tmp_path):
        from evaluate import validate_api_description_dtype_guard

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        explicit = tmp_path / "external_api_description.md"
        explicit.write_text("supported dtype: bfloat16", encoding="utf-8")

        test_cases = {
            "cases": [
                {"id": 1, "dtype": "bfloat16"},
            ]
        }

        validate_api_description_dtype_guard(work_dir, test_cases, api_desc=str(explicit))

    @staticmethod
    def test_validate_api_description_dtype_guard_rejects_mismatched_dtype(tmp_path):
        from evaluate import validate_api_description_dtype_guard

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        explicit = tmp_path / "external_api_description.md"
        explicit.write_text("supported dtype: float16", encoding="utf-8")

        test_cases = {
            "cases": [
                {"id": 1, "dtype": "bfloat16"},
            ]
        }

        with pytest.raises(ValueError, match="not mentioned by api_description"):
            validate_api_description_dtype_guard(work_dir, test_cases, api_desc=str(explicit))


class TestReporting:
    @staticmethod
    def test_make_suffix():
        from reporting import make_suffix
        s = make_suffix("abc123def456")
        assert s == "abc123de"

    @staticmethod
    def test_make_suffix_no_arg():
        from reporting import make_suffix
        s = make_suffix()
        assert len(s) > 0


# ---------------------------------------------------------------------------
# suites.py
# ---------------------------------------------------------------------------

class TestSuites:
    @staticmethod
    def test_try_op_simple_model():
        """try_op should run a simple nn.Module on CPU."""
        import torch
        from torch import nn
        from suites import try_op

        class DoubleModel(nn.Module):
            def forward(self, x):
                return x * 2

        inp = torch.tensor([1.0, 2.0, 3.0])
        result = try_op(DoubleModel, (inp,), (), role='ref')
        assert result is not None
        assert torch.allclose(result['output'], inp * 2)

    @staticmethod
    def test_try_op_non_tensor_inputs():
        """try_op should handle non-tensor inputs (scalars) without crash."""
        import torch
        from torch import nn
        from suites import try_op

        class ScaleModel(nn.Module):
            def forward(self, x, scale):
                return x * scale

        inp = torch.tensor([1.0, 2.0])
        result = try_op(ScaleModel, (inp, 3.0), (), role='ref')
        assert result is not None
        assert torch.allclose(result['output'], inp * 3.0)

    @staticmethod
    def test_try_op_golden_promotes_dtype():
        """Golden role should promote float tensors to fp64."""
        import torch
        from torch import nn
        from suites import try_op

        class Identity(nn.Module):
            def forward(self, x):
                return x

        inp = torch.tensor([1.0], dtype=torch.float16)
        result = try_op(Identity, (inp,), (), role='golden')
        assert result is not None
        assert result['output'].dtype == torch.float64

    @staticmethod
    def test_smoke_test_multi_output():
        """smoke_test should handle multi-output models."""
        import torch
        from torch import nn
        from suites import smoke_test

        class MultiOut(nn.Module):
            def forward(self, x):
                return x, x * 2

        inp = torch.tensor([1.0, 2.0])
        result = smoke_test("multi", MultiOut, (inp,))
        assert result["passed"]
        assert result["n_outputs"] == 2
