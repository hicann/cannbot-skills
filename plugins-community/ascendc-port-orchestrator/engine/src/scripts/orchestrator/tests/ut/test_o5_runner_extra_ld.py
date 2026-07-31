# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression (2026-06-19, .171 npu_dev3 multi-CANN box): phase_o5_runner must let a box
inject an extra LD_LIBRARY_PATH prefix into the RUN env via {TARGET}_EXTRA_LD_LIBRARY_PATH
(then generic EXTRA_LD_LIBRARY_PATH), so the container LD line picks up runtime acl/hccl libs
that live in a DIFFERENT CANN than the compile toolkit.

Bug it guards: the container run-env LD only added the NPU-python's torch/torch_npu .so dirs +
`source {cann_path}/set_env.sh`. On a box where libacl_dvpp.so/libhccl.so live only in
8.5.0/x86_64-linux/lib64 (while arch35 compiles with 9.1.T500 bisheng), the verifier python aborts
with `libacl_dvpp.so: cannot open shared object file`. Same resolution shape as
_resolve_npu_python_bin (target-specific wins over generic). `target` is uppercase at call sites.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from phase_o5_runner import _resolve_extra_ld  # type: ignore  # noqa: E402
from phase_o5_verify import _container_npu_python_setup  # type: ignore  # noqa: E402


def test_target_specific_wins_over_generic():
    env = {
        "A5_EXTRA_LD_LIBRARY_PATH": "/usr/local/Ascend/8.5.0/x86_64-linux/lib64",
        "EXTRA_LD_LIBRARY_PATH": "/some/generic/lib",  # must NOT win for A5
    }
    assert _resolve_extra_ld(env, "A5") == "/usr/local/Ascend/8.5.0/x86_64-linux/lib64"
    # A3 has no A3-specific key -> generic is used:
    assert _resolve_extra_ld(env, "A3") == "/some/generic/lib"


def test_falls_back_to_generic_when_target_absent():
    env = {"EXTRA_LD_LIBRARY_PATH": "/g/lib"}
    assert _resolve_extra_ld(env, "A5") == "/g/lib"


def test_empty_when_neither_present():
    # Default: no extra LD -> empty string -> zero behavior change for single-CANN targets.
    assert _resolve_extra_ld({}, "A5") == ""


def test_trailing_colon_stripped():
    assert _resolve_extra_ld({"A5_EXTRA_LD_LIBRARY_PATH": "/x/lib:"}, "A5") == "/x/lib"


def test_target_specific_empty_string_falls_through_to_generic():
    """An empty target-specific value must not shadow a real generic one
    (the `or` chain treats '' as falsy).
    """
    env = {"A5_EXTRA_LD_LIBRARY_PATH": "", "EXTRA_LD_LIBRARY_PATH": "/g/lib"}
    assert _resolve_extra_ld(env, "A5") == "/g/lib"


def test_multi_dir_prefix_preserved():
    """A colon-joined multi-dir prefix is returned verbatim (only a single trailing colon stripped)."""
    val = "/usr/local/Ascend/8.5.0/x86_64-linux/lib64:/extra/lib"
    assert _resolve_extra_ld({"A5_EXTRA_LD_LIBRARY_PATH": val}, "A5") == val


def test_container_python_setup_keeps_cann_runtime_before_extra_ld():
    py, setup = _container_npu_python_setup(
        "/usr/local/Ascend/cann-9.1.T500",
        "/root/miniconda3/envs/py311/bin",
        "/usr/local/Ascend/8.5.0/x86_64-linux/lib64",
    )

    assert py == '"$PYBIN"'
    assert (
        "PYBIN='/root/miniconda3/envs/py311/bin/python3'; "
        "[ -x \"$PYBIN\" ] || PYBIN='/root/miniconda3/envs/py311/bin/python3.11'"
    ) in setup
    assert (
        "export PYTHONPATH=/usr/local/Ascend/cann-9.1.T500/python/site-packages:"
        "${PYTHONPATH:-}"
    ) in setup

    t500_idx = setup.index("/usr/local/Ascend/cann-9.1.T500/x86_64-linux/lib64")
    extra_idx = setup.index("/usr/local/Ascend/8.5.0/x86_64-linux/lib64")
    assert t500_idx < extra_idx
    assert (
        "$SP/torch/lib:$SP/torch_npu/lib:$PYROOT/lib:"
        "/usr/local/Ascend/8.5.0/x86_64-linux/lib64"
    ) in setup
