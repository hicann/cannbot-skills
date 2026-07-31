# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""B2-core increment 2: backward_cpu_truth.template.py real-caller + e2e.

Proves the autograd oracle primitive is used in the production reference-
generation pattern (NOT isolated — #269 lesson): the template's compute_one /
main drive the full edge_inputs.pt -> grad-truth flow on CPU, exercising
compute_backward_reference end-to-end.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_RP_DIR = Path(__file__).resolve().parent.parent  # src/scripts/reference_provider/
if str(_RP_DIR) not in sys.path:
    sys.path.insert(0, str(_RP_DIR))


def _load_template():
    """Load backward_cpu_truth.template.py (dotted filename → import by path)."""
    path = _RP_DIR / "backward_cpu_truth.template.py"
    spec = importlib.util.spec_from_file_location("backward_cpu_truth_template", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MUL_SCHEMA = {
    "op_name": "mul_grad",
    "forward": (lambda x, y: x * y),
    "wrt": ["x", "y"],
    "grad_outputs_fn": None,
}


def test_compute_one_mul_grad_matches_analytic():
    """Real-caller flow: compute_one drives the primitive; mul → dx=y, dy=x."""
    mod = _load_template()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    res = mod.compute_one({"inputs": {"x": x, "y": y}}, _MUL_SCHEMA)
    assert res["status"] == "ok"
    assert torch.allclose(res["x"].double(), y.double())   # dL/dx = y
    assert torch.allclose(res["y"].double(), x.double())   # dL/dy = x


def test_compute_one_custom_grad_outputs_from_input():
    """grad_outputs_fn can pull dy from a case input (the common backward shape)."""
    mod = _load_template()
    schema = {
        "op_name": "mul_grad_dy",
        "forward": (lambda x, y: x * y),
        "wrt": ["x"],
        "grad_outputs_fn": (lambda outs, inputs: inputs["dy"]),
    }
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    dy = torch.tensor([10.0, 100.0])
    res = mod.compute_one({"inputs": {"x": x, "y": y, "dy": dy}}, schema)
    assert res["status"] == "ok"
    assert torch.allclose(res["x"].double(), (y * dy).double())  # dL/dx = y*dy


def test_compute_one_scalar_input_passed_through():
    """Non-tensor scalars bind into forward via closure, not into the oracle."""
    mod = _load_template()
    schema = {
        "op_name": "scaled_mul",
        "forward": (lambda x, y, alpha: x * y * alpha),
        "wrt": ["x"],
        "grad_outputs_fn": None,
    }
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    res = mod.compute_one({"inputs": {"x": x, "y": y, "alpha": 2.0}}, schema)
    assert res["status"] == "ok"
    assert torch.allclose(res["x"].double(), (y * 2.0).double())


def test_compute_one_degenerate_skipped_not_error():
    """Non-finite fp64 truth → status 'skipped' (§5.4 guard), not 'error'/'ok'."""
    mod = _load_template()
    schema = {
        "op_name": "overflow",
        "forward": (lambda x: x * float("inf")),
        "wrt": ["x"],
        "grad_outputs_fn": None,
    }
    res = mod.compute_one({"inputs": {"x": torch.tensor([1.0])}}, schema)
    assert res["status"] == "skipped"


def test_main_end_to_end_edge_inputs_to_truth(tmp_path, monkeypatch):
    """Full file→file flow: main() reads edge_inputs.pt, writes cpu_truth_outputs.pt
    with correct per-case grads (the real pipeline path, not isolation).
    """
    mod = _load_template()
    monkeypatch.setattr(mod, "SCHEMA", _MUL_SCHEMA)

    edge = tmp_path / "edge_inputs.pt"
    out = tmp_path / "truth.pt"
    cases = [
        {"idx": 0, "name": "c0", "shape": [3],
         "inputs": {"x": torch.tensor([1.0, 2.0, 3.0]), "y": torch.tensor([4.0, 5.0, 6.0])}},
        {"idx": 1, "name": "c1", "shape": [2],
         "inputs": {"x": torch.tensor([7.0, 8.0]), "y": torch.tensor([9.0, 10.0])}},
    ]
    torch.save({"cases": cases}, edge)

    rc = mod.main(["--edge", str(edge), "--out", str(out)])
    assert rc == 0
    saved = torch.load(out, weights_only=False)
    assert saved["op"] == "mul_grad"
    assert saved["n_cases"] == 2
    by_idx = {o["idx"]: o for o in saved["outputs"]}
    assert by_idx[0]["status"] == "ok"
    # case0: dx = y = [4,5,6]
    assert torch.allclose(by_idx[0]["x"].double(), torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
    assert torch.allclose(by_idx[1]["y"].double(), torch.tensor([7.0, 8.0], dtype=torch.float64))  # dy=x
