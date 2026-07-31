# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression coverage for three-argument DataCopy count units."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_CHECKER_PATH = Path(__file__).resolve().parent.parent / "ascendc_static_check.py"
_SPEC = importlib.util.spec_from_file_location("ascendc_static_check", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


def _violations(source: str) -> list[dict]:
    lines = source.splitlines(True)
    return (
        checker.check_datacopy_byte_count("kernel.h", lines)
        + checker.check_datacopy_params_unit_mismatch("kernel.h", lines)
    )


def test_rejects_direct_byte_variable_as_element_count():
    """A byte-named variable cannot satisfy the element-count overload."""
    violations = _violations("DataCopy(out, in, copyBytes);\n")

    assert len(violations) == 1
    assert violations[0]["line"] == 1
    assert "element/operand count" in violations[0]["detail"]


def test_rejects_sizeof_product_in_multiline_call():
    """Multiplying by sizeof remains visible across a multiline call."""
    violations = _violations(
        "void Copy() {\n"
        "  DataCopy(out,\n"
        "           in,\n"
        "           curCount * sizeof(T));\n"
        "}\n"
    )

    assert len(violations) == 1
    assert violations[0]["line"] == 2


def test_rejects_arbitrary_division_of_byte_count():
    """Only an explicit sizeof conversion can turn bytes into elements."""
    violations = _violations("DataCopy(out, in, copyBytes / 3);\n")

    assert len(violations) == 1


def test_rejects_element_count_used_as_datacopyparams_burst_count():
    """An element count cannot be reinterpreted as one-block DMA bursts."""
    source = (
        "DataCopyParams cp;\n"
        "cp.blockLen = 1;\n"
        "cp.blockCount = curCount;\n"
        "DataCopy(out, in, cp);\n"
    )

    violations = _violations(source)

    assert len(violations) == 1
    assert "blockCount counts DMA bursts" in violations[0]["detail"]


def test_accepts_genuine_strided_datacopyparams_geometry():
    """A configured stride identifies genuine multi-burst geometry."""
    source = (
        "DataCopyParams cp;\n"
        "cp.blockLen = 1;\n"
        "cp.blockCount = curCount;\n"
        "cp.srcStride = sourceGap;\n"
        "DataCopy(out, in, cp);\n"
    )

    assert _violations(source) == []


def test_zero_strides_do_not_hide_element_count_as_burst_count():
    """Explicit zero gaps retain the oversized contiguous-copy finding."""
    source = (
        "DataCopyParams cp{};\n"
        "cp.blockLen = 1;\n"
        "cp.blockCount = curCount;\n"
        "cp.srcStride = 0;\n"
        "cp.dstStride = 0U;\n"
        "DataCopy(out, in, cp);\n"
    )

    assert len(_violations(source)) == 1


def test_suffixed_one_still_exposes_element_count_as_burst_count():
    """Integer suffixes do not change a one-block burst length."""
    source = (
        "DataCopyParams cp;\n"
        "cp.blockLen = 1U;\n"
        "cp.blockCount = curCount;\n"
        "DataCopy(out, in, cp);\n"
    )

    assert len(_violations(source)) == 1


def test_accepts_explicit_block_count_units():
    """A block-named count documents the required DMA unit."""
    source = (
        "DataCopyParams cp = {};\n"
        "cp.blockLen = 1;\n"
        "cp.blockCount = curBlockCount;\n"
        "DataCopy(out, in, cp);\n"
    )

    assert _violations(source) == []


def test_accepts_element_count_and_explicit_byte_conversion():
    """Element counts and byte-to-element conversion remain valid."""
    source = (
        "DataCopy(out, in, alignedElements);\n"
        "DataCopy(out, in, payloadBytes / sizeof(T));\n"
        "DataCopyPad(out, in, DataCopyExtParams{1, copyBytes, 0, 0, 0});\n"
    )

    assert _violations(source) == []


def test_ignores_commented_examples():
    """Commented anti-pattern examples must not fail generated source."""
    source = (
        "// DataCopy(out, in, copyBytes);\n"
        "/* DataCopy(out, in, curCount * sizeof(T)); */\n"
        "DataCopy(out, in, curCount);\n"
    )

    assert _violations(source) == []
