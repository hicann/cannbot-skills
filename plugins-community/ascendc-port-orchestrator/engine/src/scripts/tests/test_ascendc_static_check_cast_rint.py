# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression coverage for same-dtype CAST_RINT value corruption."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_CHECKER_PATH = Path(__file__).resolve().parent.parent / "ascendc_static_check.py"
_SPEC = importlib.util.spec_from_file_location("ascendc_static_check", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


def _violations(path: Path, source: str) -> list[dict]:
    path.write_text(source, encoding="utf-8")
    return checker.check_cast_rint_same_dtype(str(path), source.splitlines(True))


def test_rejects_explicit_float_to_float_cast_rint(tmp_path: Path):
    source = (
        "LocalTensor<float> src;\n"
        "LocalTensor<float> dst;\n"
        "Cast(dst, src, RoundMode::CAST_RINT, count);\n"
    )

    violations = _violations(tmp_path / "kernel.h", source)

    assert len(violations) == 1
    assert violations[0]["line"] == 3
    assert "fp32-to-fp32" in violations[0]["detail"]


def test_rejects_templated_namespaced_cast_rint(tmp_path: Path):
    source = (
        "LocalTensor<float> src;\n"
        "LocalTensor<float> dst;\n"
        "AscendC::Cast<float>(dst, src, RoundMode::CAST_RINT, count);\n"
    )

    assert len(_violations(tmp_path / "kernel.h", source)) == 1


def test_rejects_indexed_same_dtype_operands(tmp_path: Path):
    source = (
        "LocalTensor<float> src;\n"
        "LocalTensor<float> dst;\n"
        "Cast(dst[offset], src[offset], RoundMode::CAST_RINT, count);\n"
    )

    assert len(_violations(tmp_path / "kernel.h", source)) == 1


def test_rejects_generic_cast_when_kernel_instantiates_float(tmp_path: Path):
    source = """\
template <typename T>
class KernelGeluGrad {
  void Compute() {
    auto src = inQueue.DeQue<T>();
    auto dst = tmpBuffer.Get<float>();
    Cast(dst, src, RoundMode::CAST_RINT, count);
  }
};
"""
    (tmp_path / "kernels.cpp").write_text(
        "KernelGeluGrad<float> kernel;\n", encoding="utf-8"
    )

    violations = _violations(tmp_path / "kernel.h", source)

    assert len(violations) == 1
    assert "instantiated as float" in violations[0]["detail"]


def test_accepts_cast_in_mandated_non_float_branch(tmp_path: Path):
    source = """\
template <typename T>
class KernelGeluGrad {
  void Compute() {
    auto src = inQueue.DeQue<T>();
    auto dst = tmpBuffer.Get<float>();
    if constexpr (!std::is_same_v<T, float>) {
      Cast(dst, src, RoundMode::CAST_RINT, count);
    }
  }
};
"""
    (tmp_path / "kernels.cpp").write_text(
        "KernelGeluGrad<float> kernel;\n", encoding="utf-8"
    )

    assert _violations(tmp_path / "kernel.h", source) == []


def test_closed_guard_does_not_exempt_later_cast(tmp_path: Path):
    source = """\
template <typename T>
class KernelGeluGrad {
  void Compute() {
    auto src = inQueue.DeQue<T>();
    auto dst = tmpBuffer.Get<float>();
    if constexpr (!std::is_same_v<T, float>) {
      Adds(dst, src, 0.0f, count);
    }
    Cast(dst, src, RoundMode::CAST_RINT, count);
  }
};
"""
    (tmp_path / "kernels.cpp").write_text(
        "KernelGeluGrad<float> kernel;\n", encoding="utf-8"
    )

    assert len(_violations(tmp_path / "kernel.h", source)) == 1


def test_string_literal_brace_does_not_keep_guard_open(tmp_path: Path):
    source = """\
template <typename T>
class KernelGeluGrad {
  void Compute() {
    auto src = inQueue.DeQue<T>();
    auto dst = tmpBuffer.Get<float>();
    if constexpr (!std::is_same_v<T, float>) {
      Log("{");
    }
    Cast(dst, src, RoundMode::CAST_RINT, count);
  }
};
"""
    (tmp_path / "kernels.cpp").write_text(
        "KernelGeluGrad<float> kernel;\n", encoding="utf-8"
    )

    assert len(_violations(tmp_path / "kernel.h", source)) == 1


def test_accepts_real_half_to_float_conversion(tmp_path: Path):
    source = (
        "LocalTensor<half> src;\n"
        "LocalTensor<float> dst;\n"
        "Cast(dst, src, RoundMode::CAST_RINT, count);\n"
    )

    assert _violations(tmp_path / "kernel.h", source) == []


def test_ignores_commented_same_dtype_example(tmp_path: Path):
    source = (
        "LocalTensor<float> src;\n"
        "LocalTensor<float> dst;\n"
        "// Cast(dst, src, RoundMode::CAST_RINT, count);\n"
    )

    assert _violations(tmp_path / "kernel.h", source) == []
