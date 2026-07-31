# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression (#448): phase_o5_runner resolves the
NPU-python bin via {TARGET}_NPU_PYTHON_BIN first, then generic NPU_PYTHON_BIN.

Bug: _run_verifier / _run_canonical_pass_a read only the generic
NPU_PYTHON_BIN, so an A3 op (TARGET=a3) used the A5 py311 path -> "No such file
or directory" exit 127 at phase_o5 verify. `target` is uppercase at the call
sites.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from phase_o5_runner import _resolve_npu_python_bin  # type: ignore  # noqa: E402


def test_target_specific_wins_over_generic():
    env = {
        "A3_NPU_PYTHON_BIN": "/usr/local/py39/bin",
        "NPU_PYTHON_BIN": "/root/miniconda3/envs/py311/bin",  # A5 — must NOT win for A3
    }
    assert _resolve_npu_python_bin(env, "A3") == "/usr/local/py39/bin"
    # And for A5 the generic (no A5-specific key) is used:
    assert _resolve_npu_python_bin(env, "A5") == "/root/miniconda3/envs/py311/bin"


def test_falls_back_to_generic_when_target_absent():
    env = {"NPU_PYTHON_BIN": "/opt/py/bin"}
    assert _resolve_npu_python_bin(env, "A3") == "/opt/py/bin"


def test_empty_when_neither_present():
    assert _resolve_npu_python_bin({}, "A5") == ""


def test_trailing_slash_stripped():
    assert _resolve_npu_python_bin({"A5_NPU_PYTHON_BIN": "/x/bin/"}, "A5") == "/x/bin"


def test_target_specific_empty_string_falls_through_to_generic():
    """An empty target-specific value must not shadow a real generic one
    (the `or` chain treats '' as falsy).
    """
    env = {"A3_NPU_PYTHON_BIN": "", "NPU_PYTHON_BIN": "/g/bin"}
    assert _resolve_npu_python_bin(env, "A3") == "/g/bin"
