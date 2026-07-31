# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Characterization tests for validate_kernel_regression.py (DEBT-202 UT backfill).

Pins the CURRENT behavior of the AST static kernel-regression detector: the four
drift types (no-import / orphan-import / partial-torch-fallback / scalar-for-loop),
the PASS path, plus the load-bearing AST helpers (AscendC import recognition,
forbidden-op detection, self-indirection tracing).

These are behavior-neutral pins — they assert what the module does today so a future
refactor that changes a verdict is caught. Not import-smoke: every test drives real
`ast`-parsed source through the public `validate()` / helper surface and asserts on the
structured result.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_WORKFLOW = _HERE.parent.parent  # src/scripts/workflow/
sys.path.insert(0, str(_WORKFLOW))

import validate_kernel_regression as vkr  # noqa: E402


# ── helper: parse a snippet + fetch its ModelNew.forward for helper-level tests ──

def _forward_of(src: str):
    tree = ast.parse(src)
    fwd, _name, cls = vkr.find_model_forward(tree)
    return tree, fwd, cls


# ── PASS path: a correctly-wired AscendC kernel ─────────────────────────────

_GOOD_ASCENDC = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        _ext.run_myop(x, y, out)
        return out
'''


def test_valid_ascendc_ext_kernel_passes():
    result = vkr.validate(_GOOD_ASCENDC)
    assert result["valid"] is True
    assert result["regression_type"] is None
    assert result["checks"]["kernel_imported"]["passed"] is True
    assert result["checks"]["kernel_called"]["passed"] is True
    # the attribute-call pattern (_ext.run_myop) is recognized
    called = result["checks"]["kernel_called"]["called"]
    assert any(c["pattern"] == "attribute_call" for c in called)


# ── Type 1: no kernel import ────────────────────────────────────────────────

def test_type1_no_kernel_import():
    src = '''
import torch


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
'''
    result = vkr.validate(src)
    assert result["valid"] is False
    assert result["regression_type"] == 1
    assert result["checks"]["kernel_imported"]["passed"] is False


def test_type1_syntax_error_is_type1():
    result = vkr.validate("def forward(  :\n    pass")
    assert result["valid"] is False
    assert result["regression_type"] == 1
    assert "SyntaxError" in result["suggestion"]


# ── Type 2: kernel imported but forward never calls it ──────────────────────

def test_type2_orphan_import():
    src = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def forward(self, x):
        # kernel imported above but never invoked here
        return torch.empty_like(x)
'''
    result = vkr.validate(src)
    assert result["valid"] is False
    assert result["regression_type"] == 2
    assert result["checks"]["kernel_imported"]["passed"] is True
    assert result["checks"]["kernel_called"]["passed"] is False


def test_type2_no_forward_method():
    src = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def other(self, x):
        return _ext.run(x)
'''
    result = vkr.validate(src)
    assert result["regression_type"] == 2
    assert result["checks"]["kernel_called"]["error"] == "no forward method"


# ── Type 3: forward uses forbidden torch compute ops ────────────────────────

def test_type3_partial_torch_fallback():
    src = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        _ext.run_myop(x, out)
        # still doing compute in forward — partial fallback
        return torch.matmul(out, y)
'''
    result = vkr.validate(src)
    assert result["valid"] is False
    assert result["regression_type"] == 3
    violations = result["checks"]["no_forbidden_torch_ops"]["violations"]
    assert any("torch.matmul" in v["call"] for v in violations)


def test_type3_matmul_operator_flagged():
    src = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        _ext.run(x)
        return x @ y
'''
    result = vkr.validate(src)
    assert result["regression_type"] == 3
    violations = result["checks"]["no_forbidden_torch_ops"]["violations"]
    assert any(v["call"] == "@" for v in violations)


def test_allowed_ops_do_not_trip_type3():
    # buffer alloc + shape ops are allowed in forward()
    src = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def forward(self, x):
        buf = torch.empty_like(x)
        v = x.view(-1).contiguous()
        _ext.run(v, buf)
        return buf.reshape(x.shape)
'''
    result = vkr.validate(src)
    assert result["valid"] is True


# ── Type 4: scalar python for-loop over tensor indices ──────────────────────

def test_type4_scalar_for_loop():
    src = '''
import torch
import _myop_ext as _ext


class ModelNew(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        _ext.run(x, out)
        for i in range(x.shape[0]):
            out[i] = x[i] + x[i] + x[i] + x[i] + x[i] + x[i]
        return out
'''
    result = vkr.validate(src)
    assert result["valid"] is False
    assert result["regression_type"] == 4
    loops = result["checks"]["no_scalar_for_loops"]["violations"]
    assert loops and loops[0]["loop_var"] == "i"


# ── AST helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module,expected", [
    ("kernel.foo", True),
    ("kernel", True),
    ("torch.nn.functional", False),
    ("numpy", False),
    ("", False),
])
def test_is_kernel_module(module, expected):
    assert getattr(vkr, '_is_kernel_module')(module) is expected


@pytest.mark.parametrize("name,expected", [
    ("_myop_ext", True),
    ("_ext", True),
    ("torch", False),
    ("myop_ext", False),  # no leading underscore
    ("", False),
])
def test_is_ext_module_alias(name, expected):
    assert getattr(vkr, '_is_ext_module_alias')(name) is expected


def test_find_kernel_imports_recognizes_all_forms():
    src = '''
import _op_ext as _ext
from kernel.mod import kfn
import kernel.other as kmod
'''
    tree = ast.parse(src)
    kernels = vkr.find_kernel_imports(tree)
    assert set(kernels) == {"_ext", "kfn", "kmod"}
    assert kernels["_ext"]["kind"] == "ascendc_ext"
    assert kernels["kfn"]["kind"] == "ascendc_kernel"


def test_find_model_forward_prefers_modelnew():
    src = '''
class Model:
    def forward(self, x):
        return x


class ModelNew:
    def forward(self, x):
        return x + 1
'''
    tree = ast.parse(src)
    fwd, name, cls = vkr.find_model_forward(tree)
    assert name == "ModelNew"
    assert cls.name == "ModelNew"


def test_self_indirection_kernel_call_recognized():
    # forward() delegates to self._run() which invokes the kernel — must count as called
    src = '''
import torch
import _op_ext as _ext


class ModelNew(torch.nn.Module):
    def _run(self, x, out):
        _ext.run(x, out)

    def forward(self, x):
        out = torch.empty_like(x)
        self._run(x, out)
        return out
'''
    result = vkr.validate(src)
    assert result["valid"] is True
    called = result["checks"]["kernel_called"]["called"]
    assert any(c["pattern"] == "self_indirection" for c in called)


def test_resolve_call_name_forms():
    # module-level attribute, nested attribute, bare name
    tree = ast.parse("torch.empty(x)\n_ext.run(a)\nfoo(b)\n")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    resolved = [getattr(vkr, '_resolve_call_name')(c) for c in calls]
    assert ("torch", "empty") in resolved
    assert ("_ext", "run") in resolved
    assert (None, "foo") in resolved


def test_scalar_math_qualifier_not_flagged():
    # math.ceil / np.array are scalar, must NOT be forbidden compute
    _tree, fwd, _cls = _forward_of('''
import torch
import _ext_op as _ext


class ModelNew:
    def forward(self, x):
        n = math.ceil(x.shape[0] / 2)
        _ext.run(x, n)
        return x
''')
    violations = vkr.check_forbidden_torch_ops(fwd, {"_ext"})
    assert violations == []
