# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Focused practical C++ anti-delegation regression tests.

The scanner deliberately permits allocation and metadata marshalling in a
custom-kernel host bridge, while rejecting framework compute or moving a value
back to CPU.  These tests pin that narrow boundary without assuming a specific
plugin layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from scan_delegation_cheating import scan_cpp  # noqa: E402


def _scan(tmp_path: Path, body: str) -> list[dict]:
    source = tmp_path / "host.cpp"
    source.write_text(body)
    return scan_cpp(source, is_pybind=False)


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("auto out = at::sort(input, 0);\n", "at::sort()"),
        ("auto out = torch::native_layer_norm(input, shape);\n", "torch::native_layer_norm()"),
    ],
)
def test_cpp_unlisted_at_or_torch_call_is_framework_delegation(tmp_path, body, needle):
    hits = _scan(tmp_path, body)
    assert any(needle in hit["desc"] for hit in hits), hits


def test_cpp_metadata_and_allocation_calls_remain_allowed(tmp_path):
    hits = _scan(
        tmp_path,
        """
auto options = at::TensorOptions().dtype(torch::kFloat16);
auto out = torch::empty({count}, options);
auto workspace = at::empty({bytes}, options);
""",
    )
    assert hits == []


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("auto host = device_value.cpu();\n", "tensor.cpu() host fallback"),
        ("auto host = device_value.to(at::kCPU);\n", "tensor.to(...::kCPU) host fallback"),
        ("auto host = device_value.to(torch::kCPU);\n", "tensor.to(...::kCPU) host fallback"),
    ],
)
def test_cpp_host_fallback_is_rejected(tmp_path, body, needle):
    hits = _scan(tmp_path, body)
    assert any(needle in hit["desc"] for hit in hits), hits


def test_cpp_operator_specific_aten_header_is_rejected(tmp_path):
    hits = _scan(tmp_path, "#include <ATen/ops/topk.h>\n")
    assert any("ATen/ops/topk.h" in hit["desc"] for hit in hits), hits


def test_cpp_allocator_aten_header_and_comment_are_not_false_positives(tmp_path):
    hits = _scan(
        tmp_path,
        """
#include <ATen/ops/empty.h>
// #include <ATen/ops/topk.h>
const char* text = "at::sort(input, 0)";
""",
    )
    assert hits == []
