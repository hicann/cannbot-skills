# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for P149 pybind host-business-logic gate.

User catch 2026-05-18T03:18Z: pybind11.cpp MUST NOT contain business
logic. Two confirmed incident patterns:
  1. elu archive (commit 509b512a): TAIL_PAD output alloc + narrow+view
     crop to mask kernel writing past valid range
  2. clipped_swiglu archive: `group_index->to(at::kCPU)` CPU offload

DEBT-002 pybind purity hook was too permissive (ALLOWED list included
narrow/view/to(cpu)). P149 gate adds structural detection at finalize
time.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _check_pybind_host_logic  # noqa: E402


@pytest.fixture
def tmpws(tmp_path):
    """Workspace dir with op_kernel subdir."""
    ws = tmp_path / "op"
    (ws / "op_kernel").mkdir(parents=True)
    return ws


def _write_pybind(ws: Path, content: str) -> None:
    (ws / "op_kernel" / "pybind11.cpp").write_text(content)


def test_p149_clean_pybind_passes(tmpws):
    """Minimal clean pybind — contiguous input + torch::empty natural shape
    + data_ptr + kernel call. No business logic. Must pass.
    """
    _write_pybind(
        tmpws,
        """
        torch::Tensor my_op(torch::Tensor x) {
            auto x_c = x.contiguous();
            auto y = torch::empty(x_c.sizes(), x_c.options());
            launch_kernel(x_c.data_ptr(), y.data_ptr(), x_c.numel());
            return y;
        }
        """,
    )
    assert _check_pybind_host_logic(tmpws, {}) is None


def test_p149_elu_tail_pad_pattern_blocks(tmpws):
    """The exact elu incident pattern: TAIL_PAD_ELEMS alloc + narrow crop.
    Must REJECT.
    """
    _write_pybind(
        tmpws,
        """
        constexpr int64_t TAIL_PAD_ELEMS = 32;
        torch::Tensor elu(torch::Tensor x) {
            auto x_c = x.contiguous();
            auto numel = x_c.numel();
            auto raw = torch::empty({numel + TAIL_PAD_ELEMS}, x_c.options());
            launch_kernel(x_c.data_ptr(), raw.data_ptr(), numel);
            return raw.narrow(0, 0, numel).view(x_c.sizes());
        }
        """,
    )
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "P149" in reason
    assert "Output-alignment cleanup" in reason
    assert "TAIL_PAD" in reason or "narrow" in reason


def test_p149_clipped_swiglu_cpu_offload_blocks(tmpws):
    """The clipped_swiglu incident: group_index->to(at::kCPU). Must REJECT."""
    _write_pybind(
        tmpws,
        """
        torch::Tensor clipped_swiglu(torch::Tensor x, c10::optional<torch::Tensor> group_index) {
            auto x_c = x.contiguous();
            if (group_index.has_value()) {
                auto gi_cpu = group_index->to(at::kCPU).to(at::kLong).contiguous();
                /* use gi_cpu ... */
            }
            auto y = torch::empty(x_c.sizes(), x_c.options());
            return y;
        }
        """,
    )
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "P149" in reason
    assert "CPU offload" in reason


def test_p149_cpu_method_call_also_blocks(tmpws):
    """Same as above but via `.cpu()` method instead of `.to(at::kCPU)`."""
    _write_pybind(
        tmpws,
        """
        torch::Tensor op(torch::Tensor x) {
            auto x_cpu = x.cpu();  // CPU offload via .cpu() method
            auto y = torch::empty(x.sizes(), x.options());
            return y;
        }
        """,
    )
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "CPU offload" in reason


def test_p149_torch_cat_in_pybind_blocks(tmpws):
    """torch::cat in pybind = output assembly = host-side fusion. Reject."""
    _write_pybind(
        tmpws,
        """
        torch::Tensor op(torch::Tensor x) {
            auto a = torch::empty(x.sizes(), x.options());
            auto b = torch::empty(x.sizes(), x.options());
            launch_kernel(x.data_ptr(), a.data_ptr(), b.data_ptr(), x.numel());
            return torch::cat({a, b}, -1);
        }
        """,
    )
    reason = _check_pybind_host_logic(tmpws, {})
    assert reason is not None
    assert "Output assembly" in reason


def test_p149_comments_alone_dont_trigger(tmpws):
    """Comments mentioning forbidden patterns must NOT trigger — only
    actual code does. Critical: DEBT-002 hook header comment lists
    FORBIDDEN: torch::cat in EVERY archive pybind11.cpp. Gate must
    strip comments before scanning.
    """
    _write_pybind(
        tmpws,
        """
        // Pybind purity:
        //   ALLOWED: contiguous, options, data_ptr
        //   FORBIDDEN: torch::cat, torch::stack, .to(at::kCPU), .cpu(), TAIL_PAD, narrow(0,0,...)
        // (Above is just doc — actual code below is clean.)
        torch::Tensor my_op(torch::Tensor x) {
            auto x_c = x.contiguous();
            auto y = torch::empty(x_c.sizes(), x_c.options());
            launch_kernel(x_c.data_ptr(), y.data_ptr(), x_c.numel());
            return y;
        }
        """,
    )
    assert _check_pybind_host_logic(tmpws, {}) is None, (
        "Header doc comments listing forbidden patterns must not trigger gate"
    )


def test_p149_no_pybind_file_passes(tmpws):
    """No pybind11.cpp (e.g., aclnn-direct CLI runner mode) — gate skips."""
    # don't write any pybind file
    assert _check_pybind_host_logic(tmpws, {}) is None


def test_p149_allows_item_for_scalar_metadata(tmpws):
    """`.item<T>()` for scalar metadata extraction is explicitly allowed
    per OL-163 + brief. Must pass.
    """
    _write_pybind(
        tmpws,
        """
        torch::Tensor fatrelu_mul(torch::Tensor x, torch::Tensor threshold) {
            auto x_c = x.contiguous();
            float thr = threshold.contiguous().item<float>();  // OK: scalar metadata
            auto y = torch::empty(x_c.sizes(), x_c.options());
            launch_kernel(x_c.data_ptr(), y.data_ptr(), x_c.numel(), thr);
            return y;
        }
        """,
    )
    assert _check_pybind_host_logic(tmpws, {}) is None


def test_p149_legacy_pybind_at_root_also_detected(tmp_path):
    """Some archives put pybind11.cpp at op root instead of op_kernel/.
    Both locations must be scanned.
    """
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "pybind11.cpp").write_text("""
        torch::Tensor op(torch::Tensor x) {
            auto x_cpu = x.to(at::kCPU);
            return torch::empty(x.sizes(), x.options());
        }
    """)
    reason = _check_pybind_host_logic(ws, {})
    assert reason is not None
    assert "CPU offload" in reason


def test_p149_workspace_none_returns_none():
    """Test-fixture safety: None workspace → None (no false positives)."""
    assert _check_pybind_host_logic(None, {}) is None
