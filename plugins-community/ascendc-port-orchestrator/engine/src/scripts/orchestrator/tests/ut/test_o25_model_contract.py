# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""item① (controllable-harness / DEBT-199) — Phase-O2.5 model.py↔input_gen contract
validation (2026-07-04).

validate_model_contract runs model.forward on ONE sample case and FAILS LOUD with the
SPECIFIC mismatch (NPU-delegation / signature-mismatch / output-shape) so the 生态
cpu_truth golden path can never silently produce no/wrong golden (the archived-FA
model.py failure mode: it delegated to npu_fusion_attention instead of CPU math).
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o25_a3_ref as p25a3  # noqa: E402


def _write(ws: Path, model_src: str):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "model.py").write_text(dedent(model_src))


def _edge_inputs(ws: Path, cases):
    torch.save(cases, ws / "edge_inputs.pt")


# ---------------------------------------------------------------------------
# OK path
# ---------------------------------------------------------------------------
def test_ok_cpu_pure_kwargs_model(tmp_path):
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                return x * 2.0
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK"


def test_ok_wrapped_case_is_unwrapped(tmp_path):
    """A wrapped case {idx,name,shape,inputs,meta} must be unwrapped (DEBT-199 v2)."""
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                return x + 1
    """)
    _edge_inputs(tmp_path, [{"idx": 0, "name": "c0", "shape": [4],
                             "inputs": {"x": torch.ones(4)}, "meta": {}}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK"


def test_ok_get_input_groups_positional(tmp_path):
    _write(tmp_path, """
        import torch
        def get_input_groups():
            return [[torch.ones(4), torch.ones(4)]]
        class Model:
            def __call__(self, a, b):
                return a + b
    """)
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK"


# ---------------------------------------------------------------------------
# FAIL-LOUD: NPU delegation (static, no run needed)
# ---------------------------------------------------------------------------
def test_npu_delegation_torch_npu_import(tmp_path):
    _write(tmp_path, """
        import torch
        import torch_npu
        class Model:
            def __call__(self, x):
                return torch_npu.npu_fusion_attention(x)
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert not r.ok and r.reason_code == "NPU_DELEGATION"
    assert "torch_npu" in r.message


def test_npu_delegation_dot_npu_move(tmp_path):
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                return (x.npu() * 2).cpu()
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert not r.ok and r.reason_code == "NPU_DELEGATION"


def test_npu_delegation_ignores_comment(tmp_path):
    """A commented-out reference to torch_npu must NOT trip the static scan."""
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                # historically this used torch_npu.npu_fusion_attention
                return x * 2
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK"


def test_npu_delegation_ignores_module_docstring(tmp_path):
    """The exact archived-FA false-positive: MODULE DOCSTRING mentions
    `npu_fusion_attention` but the body is pure-CPU torch SDPA (no real
    torch_npu / npu_*() / .npu() CALL). Validation must PASS (no NPU_DELEGATION).
    """
    _write(tmp_path, '''
        """Pure-PyTorch CPU reference — the fp64-golden equivalent of
        npu_fusion_attention. Emits the same result as torch_npu.npu_fusion_attention
        but uses only plain torch SDPA math so it can produce a CPU fp64 golden."""
        import torch
        import torch.nn.functional as F
        class Model:
            def __call__(self, q, k, v):
                # equivalent to npu_fusion_attention, but CPU-only
                return F.scaled_dot_product_attention(q, k, v)
    ''')
    _edge_inputs(tmp_path, [{"q": torch.ones(1, 1, 2, 4),
                             "k": torch.ones(1, 1, 2, 4),
                             "v": torch.ones(1, 1, 2, 4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK", r.message


def test_npu_delegation_ignores_docstring_but_catches_real_call(tmp_path):
    """A vendor-op NAME in the docstring must be ignored, but a REAL
    torch_npu.npu_fusion_attention(...) CALL in the body must STILL be caught.
    """
    _write(tmp_path, '''
        """Reference doc: this mimics npu_fusion_attention semantics."""
        import torch
        import torch_npu
        class Model:
            def __call__(self, x):
                return torch_npu.npu_fusion_attention(x)
    ''')
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert not r.ok and r.reason_code == "NPU_DELEGATION"
    assert "torch_npu" in r.message


def test_npu_delegation_ignores_string_literal(tmp_path):
    """A vendor-op name inside a plain string literal (not a docstring) must NOT trip."""
    _write(tmp_path, '''
        import torch
        class Model:
            def __call__(self, x):
                note = "delegates to npu_fusion_attention on the device path"
                return x * 2
    ''')
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK", r.message


# ---------------------------------------------------------------------------
# FAIL-LOUD: signature mismatch — names the exact bad kwargs
# ---------------------------------------------------------------------------
def test_signature_mismatch_names_extra_and_rename(tmp_path):
    """case emits `input_layout`, forward wants `layout` → extra kwarg named + rename hint."""
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x, layout):
                return x
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4), "input_layout": "BSND"}])
    r = p25a3.validate_model_contract(tmp_path)
    assert not r.ok and r.reason_code == "SIGNATURE_MISMATCH"
    assert "input_layout" in r.message
    assert "layout" in r.message  # rename hint
    assert r.detail.get("extra_kwargs") == ["input_layout"]


def test_kwargs_star_accepts_any_keys(tmp_path):
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x, **kwargs):
                return x
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4), "anything": 1, "extra": 2}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK"


# ---------------------------------------------------------------------------
# FAIL-LOUD: output-shape mismatch vs a3_outputs
# ---------------------------------------------------------------------------
def test_output_shape_mismatch(tmp_path):
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                return torch.ones(2, 2)   # wrong shape vs a3 capture (4,)
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    torch.save({"inputs": [0], "a3_outputs": [torch.ones(4)]}, tmp_path / "edge_dataset.pt")
    r = p25a3.validate_model_contract(tmp_path)
    assert not r.ok and r.reason_code == "OUTPUT_SHAPE_MISMATCH"
    assert r.detail.get("model_shape") == [2, 2]


# ---------------------------------------------------------------------------
# Non-blocking skips
# ---------------------------------------------------------------------------
def test_skipped_no_model(tmp_path):
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "SKIPPED_NO_MODEL"


def test_skipped_no_inputs(tmp_path):
    _write(tmp_path, """
        class Model:
            def __call__(self, x):
                return x
    """)
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "SKIPPED_NO_INPUTS"


def test_forward_raised_generic(tmp_path):
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                raise RuntimeError("boom in reference")
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert not r.ok and r.reason_code == "FORWARD_RAISED"
    assert "boom" in r.message


def test_fp64_downcast_warning(tmp_path):
    """A forward that returns fp16 despite fp64-capable inputs → OK but with a warning
    (non-blocking: the cpu_truth golden wouldn't be truly fp64).
    """
    _write(tmp_path, """
        import torch
        class Model:
            def __call__(self, x):
                return (x.to(torch.float32)).to(torch.float16)
    """)
    _edge_inputs(tmp_path, [{"x": torch.ones(4, dtype=torch.float64)}])
    r = p25a3.validate_model_contract(tmp_path)
    assert r.ok and r.reason_code == "OK"
    assert "fp64_downcast_warning" in r.detail
