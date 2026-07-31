# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for backward_input_gen — determinism + edge-value properties."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_RP_DIR = Path(__file__).resolve().parent.parent
if str(_RP_DIR) not in sys.path:
    sys.path.insert(0, str(_RP_DIR))

from backward_input_gen import (  # noqa: E402
    DEFAULT_PROFILES,
    materialize_inputs,
    materialize_grad_outputs,
    resolve_shape,
)

_DECL = {"x": {"shape": ["N"], "dtype": "float32"}}
_BIND = {"N": 256}


def test_resolve_shape_symbolic_and_concrete():
    assert resolve_shape(["N", 64], {"N": 8}) == [8, 64]
    assert resolve_shape([1, 1, "S", 64], {"S": 128}) == [1, 1, 128, 64]


def test_reproducible_same_seed_idx():
    a, _ = materialize_inputs(_DECL, _BIND, torch.float32, "randn", 1234, 3)
    b, _ = materialize_inputs(_DECL, _BIND, torch.float32, "randn", 1234, 3)
    assert torch.equal(a["x"], b["x"])  # bit-identical reproduction


def test_distinct_idx_distinct_inputs():
    a, _ = materialize_inputs(_DECL, _BIND, torch.float32, "randn", 1234, 1)
    b, _ = materialize_inputs(_DECL, _BIND, torch.float32, "randn", 1234, 2)
    assert not torch.equal(a["x"], b["x"])  # different record idx → different draw


def test_zeros_profile_has_exact_zeros():
    out, _ = materialize_inputs(_DECL, _BIND, torch.float32, "zeros", 1234, 5)
    assert (out["x"] == 0).any(), "zeros profile must inject exact zeros"


def test_large_profile_bigger_than_randn():
    rn, _ = materialize_inputs(_DECL, _BIND, torch.float32, "randn", 1234, 7)
    lg, _ = materialize_inputs(_DECL, _BIND, torch.float32, "large", 1234, 7)
    assert lg["x"].abs().max() > rn["x"].abs().max() * 10


def test_small_profile_smaller_than_randn():
    sm, _ = materialize_inputs(_DECL, _BIND, torch.float32, "small", 1234, 9)
    assert sm["x"].abs().max() < 0.1


def test_boundary_profile_has_boundary_values():
    out, _ = materialize_inputs(_DECL, _BIND, torch.float32, "boundary", 1234, 11)
    x = out["x"]
    # contains at least one of the sprinkled boundary constants
    has = any((x == v).any().item() for v in (0.0, 1.0, -1.0, 0.5, -0.5))
    assert has


def test_all_default_profiles_materialize():
    for i, prof in enumerate(DEFAULT_PROFILES):
        out, shapes = materialize_inputs(_DECL, _BIND, torch.float32, prof, 1234, i)
        assert out["x"].shape == (256,)
        assert shapes["x"] == [256]


def test_grad_outputs_reproducible():
    def forward(x):
        return torch.abs(x)
    inp, _ = materialize_inputs(_DECL, _BIND, torch.float32, "randn", 1234, 2)
    g1, s1 = materialize_grad_outputs(forward, inp, 2, 1234)
    g2, s2 = materialize_grad_outputs(forward, inp, 2, 1234)
    assert torch.equal(g1, g2) and s1 == s2
