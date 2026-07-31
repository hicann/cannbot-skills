# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for P151 cache-replay anti-pattern detection (OL-165).

User catch 2026-05-18 04:27Z: foreach_sqrt model_new_ascendc.py shipped
with MODE B fallback that returns cached A5 outputs by hash lookup when
the kernel .so fails to load. Worker self-justified with 80+ line
docstring. Same pattern subsequently audited and confirmed in 3 already-
committed archives (apply_adam_w_v2, adaptive_avg_pool3d,
gather_elements_v2).

The gate extension (P151 within `_check_pybind_host_logic`) catches:
- hashlib / _tensor_digest / _LOOKUP_CACHE markers
- a5_capture.pt / edge_dataset.pt['a5_outputs'] reads
- subprocess.run from forward path
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _check_pybind_host_logic  # noqa: E402


@pytest.fixture
def tmpws(tmp_path):
    """Workspace with op_kernel/pybind11.cpp (clean) + model_new_ascendc.py slot."""
    ws = tmp_path / "op"
    (ws / "op_kernel").mkdir(parents=True)
    (ws / "op_kernel" / "pybind11.cpp").write_text(
        "torch::Tensor op(torch::Tensor x) {\n"
        "    auto x_c = x.contiguous();\n"
        "    auto y = torch::empty(x_c.sizes(), x_c.options());\n"
        "    return y;\n"
        "}\n"
    )
    return ws


def _write_model_new(ws: Path, content: str) -> None:
    (ws / "model_new_ascendc.py").write_text(content)


def test_p151_clean_model_new_passes(tmpws):
    """Minimal model_new_ascendc.py that only imports pybind ext and
    dispatches. No cache, no subprocess, no torch compute. Must pass.
    """
    _write_model_new(tmpws, """
import torch
import torch.nn as nn
import _foreach_sqrt_ext as _ext

class ModelNew(nn.Module):
    def forward(self, x):
        return _ext.run_foreach_sqrt(x)
""")
    assert _check_pybind_host_logic(tmpws, {}) is None


def test_p151_hashlib_digest_pattern_blocks(tmpws):
    """hashlib + digest = cache-replay marker. Must REJECT."""
    _write_model_new(tmpws, """
import torch
import hashlib
import io

def _tensor_digest(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return hashlib.md5(buf.getvalue()).hexdigest()

class ModelNew(nn.Module):
    def forward(self, x):
        return _LOOKUP_CACHE.get(_tensor_digest(x))
""")
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "hashlib" in reason or "_tensor_digest" in reason or "_LOOKUP_CACHE" in reason


def test_p151_a5_capture_read_blocks(tmpws):
    """Reading a5_capture.pt for output lookup is the explicit cheat marker."""
    _write_model_new(tmpws, """
import torch
captured = torch.load("a5_capture.pt")

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return captured[0]["a5_outputs"]
""")
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "a5_capture" in reason


def test_p151_subprocess_from_forward_blocks(tmpws):
    """subprocess.run from forward path = cpp runner pattern; OUR pybind/kernel
    is not in the test loop. Pattern 2 of OL-165.
    """
    _write_model_new(tmpws, """
import subprocess
import torch
import tempfile

class ModelNew(torch.nn.Module):
    def forward(self, x):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([runner_path, "input.bin", "output.bin"])
""")
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "subprocess" in reason


def test_p151_lookup_cache_attr_blocks(tmpws):
    """Even without hashlib import, _LOOKUP_CACHE token alone signals
    cache-replay design intent.
    """
    _write_model_new(tmpws, """
import torch

_LOOKUP_CACHE = {}

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return _LOOKUP_CACHE.get(id(x))
""")
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "_LOOKUP_CACHE" in reason


def test_p151_known_apply_adam_w_v2_pattern(tmpws):
    """Reproduce the exact apply_adam_w_v2 model_new_ascendc.py shape
    (verbatim from output/a3_to_a5_port/src/kernels/apply_adam_w_v2/
    model_new_ascendc.py line 105-150 + forward 198-217). Must REJECT.
    """
    _write_model_new(tmpws, """
import torch
import torch.nn as nn
import hashlib
import io
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_A5_CAPTURE_PATH = _HERE / "a5_capture.pt"

def _tensor_digest(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return hashlib.md5(buf.getvalue()).hexdigest()

def _build_lookup():
    return {}

_LOOKUP_CACHE = None

class ModelNew(nn.Module):
    def forward(self, var, m, v, maxgrad, grad, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize):
        global _LOOKUP_CACHE
        if _LOOKUP_CACHE is None:
            _LOOKUP_CACHE = _build_lookup()
        digest = _tensor_digest(var)
        hit = _LOOKUP_CACHE.get(digest)
        if hit is not None:
            return hit
        raise NotImplementedError("no cache match")
""")
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None, "apply_adam_w_v2 incident pattern must be rejected"
    # multiple signals expected
    assert sum(s in reason for s in ("hashlib", "_tensor_digest", "_LOOKUP_CACHE", "_build_lookup", "a5_capture")) >= 2


def test_p151_no_model_new_file_passes(tmpws):
    """If model_new_ascendc.py doesn't exist, scan it not — clean."""
    assert _check_pybind_host_logic(tmpws, {}) is None
