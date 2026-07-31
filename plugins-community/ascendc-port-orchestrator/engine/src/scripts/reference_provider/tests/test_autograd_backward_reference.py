# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""B2-core: autograd backward reference oracle correctness (design §8 layer 4/5).

Layer 4 (oracle correctness): autograd reference matches the analytic
hand-derived gradient for known ops.
Layer 5 (multi-output): one grad tensor per differentiable input (`wrt`).
Plus the §5.4 degenerate-reference guard + error contracts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_RP_DIR = Path(__file__).resolve().parent.parent  # src/scripts/reference_provider/
if str(_RP_DIR) not in sys.path:
    sys.path.insert(0, str(_RP_DIR))

from autograd_backward_reference import (  # noqa: E402
    compute_backward_reference,
    DegenerateReferenceError,
)


# ── Layer 4: analytic-match correctness ──────────────────────────────────

def test_mul_grad_matches_analytic():
    """y = x*z → dy/dx = z, dy/dz = x (grad_outputs = ones)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    z = torch.tensor([4.0, 5.0, 6.0])
    grads = compute_backward_reference(lambda x, z: x * z, {"x": x, "z": z}, wrt=["x", "z"])
    assert torch.allclose(grads["x"], z.to(torch.float64))   # dL/dx = z
    assert torch.allclose(grads["z"], x.to(torch.float64))   # dL/dz = x


def test_add_grad_matches_analytic():
    """y = x+z → dy/dx = dy/dz = 1 (grad_outputs = ones)."""
    x = torch.randn(4)
    z = torch.randn(4)
    grads = compute_backward_reference(lambda x, z: x + z, {"x": x, "z": z}, wrt=["x", "z"])
    assert torch.allclose(grads["x"], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(grads["z"], torch.ones(4, dtype=torch.float64))


def test_matmul_grad_matches_analytic():
    """y = x @ W → dL/dx = grad_out @ W^T (custom grad_outputs)."""
    x = torch.randn(2, 3)
    weight = torch.randn(3, 5)
    go = torch.randn(2, 5)
    grads = compute_backward_reference(
        lambda x, W: x @ W, {"x": x, "W": weight}, wrt=["x"], grad_outputs=go,
    )
    expected = go.to(torch.float64) @ weight.to(torch.float64).T
    assert torch.allclose(grads["x"], expected, atol=1e-10)


def test_custom_grad_outputs_scale():
    """grad_outputs scales the grad linearly (dy/dx=z, scaled by go)."""
    x = torch.tensor([1.0, 2.0])
    z = torch.tensor([3.0, 4.0])
    go = torch.tensor([10.0, 100.0])
    grads = compute_backward_reference(
        lambda x, z: x * z, {"x": x, "z": z}, wrt=["x"], grad_outputs=go,
    )
    assert torch.allclose(grads["x"], (z * go).to(torch.float64))


# ── Layer 5: multi-output ─────────────────────────────────────────────────

def test_multi_output_returns_grad_per_wrt():
    x = torch.randn(3)
    z = torch.randn(3)
    w = torch.randn(3)
    grads = compute_backward_reference(
        lambda x, z, w: x * z + w, {"x": x, "z": z, "w": w}, wrt=["x", "z", "w"],
    )
    assert set(grads.keys()) == {"x", "z", "w"}
    assert torch.allclose(grads["x"], z.to(torch.float64))
    assert torch.allclose(grads["z"], x.to(torch.float64))
    assert torch.allclose(grads["w"], torch.ones(3, dtype=torch.float64))


def test_reference_is_fp64():
    x = torch.randn(2, dtype=torch.float32)
    z = torch.randn(2, dtype=torch.float32)
    grads = compute_backward_reference(lambda x, z: x * z, {"x": x, "z": z}, wrt=["x"])
    assert grads["x"].dtype == torch.float64


def test_non_wrt_int_input_passed_through():
    """Index/int inputs (not in wrt) are passed through, not cast to float."""
    idx = torch.tensor([0, 2, 1], dtype=torch.int64)
    table = torch.randn(4, 5)
    grads = compute_backward_reference(
        lambda table, idx: table[idx], {"table": table, "idx": idx}, wrt=["table"],
    )
    assert grads["table"].shape == table.shape  # gather backward = scatter-add


# ── §5.4 degenerate-reference guard + error contracts ────────────────────

def test_degenerate_reference_nonfinite_output_raises():
    def overflow_fwd(x):
        return x * float("inf")
    with pytest.raises(DegenerateReferenceError):
        compute_backward_reference(overflow_fwd, {"x": torch.tensor([1.0])}, wrt=["x"])


def test_grad_none_raises_when_not_differentiable():
    """wrt an input the output doesn't depend on → autograd None → RuntimeError."""
    with pytest.raises(RuntimeError):
        compute_backward_reference(
            lambda x, z: x * 2.0, {"x": torch.randn(3), "z": torch.randn(3)}, wrt=["z"],
        )


def test_unknown_wrt_raises_keyerror():
    with pytest.raises(KeyError):
        compute_backward_reference(lambda x: x * 2, {"x": torch.randn(2)}, wrt=["nope"])


def test_empty_wrt_raises():
    with pytest.raises(ValueError):
        compute_backward_reference(lambda x: x * 2, {"x": torch.randn(2)}, wrt=[])
