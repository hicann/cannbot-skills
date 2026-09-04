# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for scan_delegation_cheating __main__-block waiver.

Surfaced by a 20_Gather canary on 2026-05-20. The `if __name__ == "__main__":`
block typically contains CPU oracle code for human-visible bit-equality
smoke tests (e.g. `torch.gather(x_cpu, ...)`). That code is NEVER invoked
when the verifier imports ModelNew + calls forward() — so it categorically
cannot be a delegation vector on the kernel surface. Scanner must not flag.

Same shape as PR #21 tl.sort false-positive, but at block-context level
instead of call-pattern level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from scan_delegation_cheating import scan_python_wrapper  # noqa: E402


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


# ---- Bug reproduction: 20_Gather case ----


def test_torch_gather_in_main_block_is_waived(tmp_path):
    """20_Gather model_new_ascendc.py reproduction. torch.gather in __main__
    smoke test must NOT be flagged.
    """
    p = _write(tmp_path, "model_new_ascendc.py", """
import torch
from kernel.gather_launch import run_gather

class ModelNew(torch.nn.Module):
    def forward(self, x, dim, index, sparse_grad=False):
        return run_gather(x, int(dim), index, bool(sparse_grad))


if __name__ == "__main__":
    x_cpu = torch.randn(8, 16, 16, 16)
    idx_cpu = torch.randint(0, 8, (4, 16, 16, 16), dtype=torch.int64)
    dev = torch.device("npu:0")
    mn = ModelNew()
    out_npu = mn(x_cpu.to(dev), 0, idx_cpu.to(dev), False)
    out_cpu_truth = torch.gather(x_cpu, 0, idx_cpu, sparse_grad=False)
    diff = (out_npu.cpu().float() - out_cpu_truth.float()).abs()
    print(f"smoke out={out_npu.shape} max_abs={diff.max().item():.3e}")
""")
    hits = scan_python_wrapper(p)
    assert hits == [], f"Expected no hits, got {hits}"


# ---- Negative cases: REAL delegation OUTSIDE main block must still be flagged ----


def test_torch_gather_at_module_level_still_flagged(tmp_path):
    """If torch.gather is called at module level (or inside forward()),
    that IS real delegation; scanner must still flag it.
    """
    p = _write(tmp_path, "model_new_ascendc.py", """
import torch

class ModelNew(torch.nn.Module):
    def forward(self, x, dim, index, sparse_grad=False):
        return torch.gather(x, dim, index, sparse_grad=sparse_grad)
""")
    hits = scan_python_wrapper(p)
    assert len(hits) >= 1
    assert any("torch.gather" in (h.get("text") or "") or "gather" in (h.get("desc") or "").lower()
               for h in hits), f"Expected gather delegation flag, got {hits}"


@pytest.mark.parametrize(
    "expression",
    ["torch.add(x, y)", "torch.sub(x, y)", "torch.mul(x, y)", "torch.div(x, y)", "x.add(y)"],
)
def test_elementwise_torch_delegation_in_forward_is_flagged(tmp_path, expression):
    """Elementwise top-level and tensor-method shortcuts are still delegation."""
    p = _write(tmp_path, "model_new_ascendc.py", f"""
import torch

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        return {expression}
""")

    hits = scan_python_wrapper(p)

    assert hits, f"Expected elementwise delegation flag for {expression}, got {hits}"


def test_torch_gather_in_forward_method_still_flagged(tmp_path):
    """Delegation in forward() body must still be flagged even if file
    contains an unrelated __main__ block elsewhere.
    """
    p = _write(tmp_path, "model_new_ascendc.py", """
import torch
import torch_npu

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch_npu.npu_some_op(x)  # ← real delegation

if __name__ == "__main__":
    pass  # innocent __main__ block
""")
    hits = scan_python_wrapper(p)
    assert len(hits) >= 1
    assert any("torch_npu" in (h.get("text") or "") for h in hits)


# ---- Block exit semantics ----


def test_main_block_exits_at_top_level_statement(tmp_path):
    """A top-level statement at the SAME indent as `if __name__` exits
    the block. Subsequent delegation must still be flagged.
    """
    p = _write(tmp_path, "model_new_ascendc.py", """
import torch

if __name__ == "__main__":
    x = torch.randn(8)
    y = torch.gather(x, 0, torch.tensor([0]))  # ← waived (in main block)

def helper():
    z = torch.gather(x, 0, torch.tensor([0]))  # ← FLAGGED (back at top level)
""")
    hits = scan_python_wrapper(p)
    # Only the helper-function torch.gather should be flagged
    assert len(hits) >= 1
    assert all(h.get("line", 0) >= 8 for h in hits), f"Expected hits only after main block exit, got {hits}"


def test_nested_indent_inside_main_block_still_waived(tmp_path):
    """Code nested DEEPER than the if __name__ statement (e.g. inside a
    for loop in __main__) is still inside the block, still waived.
    """
    p = _write(tmp_path, "model_new_ascendc.py", """
import torch

if __name__ == "__main__":
    for i in range(3):
        x = torch.randn(8)
        y = torch.gather(x, 0, torch.tensor([i]))  # ← inside main block, waived
        for j in range(2):
            z = torch.gather(x, 0, torch.tensor([j]))  # ← also waived
""")
    hits = scan_python_wrapper(p)
    assert hits == [], f"Expected no hits in nested __main__ code, got {hits}"


def test_main_block_with_inline_comment_still_recognized(tmp_path):
    """The guard line can have a trailing comment; still must match."""
    p = _write(tmp_path, "model_new_ascendc.py", """
import torch

if __name__ == "__main__":  # smoke-test entry
    y = torch.gather(torch.randn(8), 0, torch.tensor([0]))
""")
    hits = scan_python_wrapper(p)
    assert hits == [], f"Expected guard-with-comment to still waive, got {hits}"
