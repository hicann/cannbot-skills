# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for FA-class structural pre-build gate (Test 5-bis).

Moved 2026-05-26 from port_fa_cv_agent plugin into shared
`src/scripts/orchestrator/plugins/_fa_class_gate.py` so any mode plugin
(port_a3 / backward) can apply
the gate when op_class is FA. Owner architectural correction: FA is an
op-class concern, not a mode.

Covers the 2026-05-26 three-way-convergent diagnosis:
- cv-agent stock FA (hand Mmad + cube/vec split + real WorkspaceQueue) must PASS
  — the old syntactic Check 2 (`grep Matmul<`) FALSELY rejected it.
- Real Antipattern A (monolithic / inline-flag-spam / empty-WQ-shell) must FAIL.
- The semantic WQ check must reject an empty `class WorkspaceQueue {}` decoy
  (same decoy class as the POC's empty MatmulKernel<> that gamed Check 2).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def _gate():
    """Return the pre-build gate function from the shared module."""
    from orchestrator.plugins._fa_class_gate import pre_build_check_test_5bis
    return pre_build_check_test_5bis


def _ws(files: dict[str, str]) -> Path:
    """Write {relpath: content} into a fresh temp workspace, return its path."""
    d = Path(tempfile.mkdtemp())
    for rel, content in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


# ── Reusable fixtures mirroring cv-agent stock structure ────────────────
_REAL_WQ = """
class WorkspaceQueue {
public:
    __aicore__ inline void Init(uint16_t pId, uint16_t cId) {}
    __aicore__ inline AscendC::GlobalTensor<float> ProducerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(consumerNotifyProducerId_);
    }
    __aicore__ inline void ProducerReleaseFix() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(producerNotifyConsumerId_);
    }
    __aicore__ inline AscendC::GlobalTensor<float> ConsumerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(producerNotifyConsumerId_);
    }
    __aicore__ inline void ConsumerReleaseMte2() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(consumerNotifyProducerId_);
    }
private:
    uint16_t producerNotifyConsumerId_;
    uint16_t consumerNotifyProducerId_;
};
"""

_CUBE_H = """
#include "matmul_tile.h"
class FlashAttentionCube {
    __aicore__ inline void Process() {
        MmadParams mp;
        Mmad(cL0, aL0, bL0, mp);
        WorkspaceQueue<float, RING_SLOTS> sQueue_;
        auto s = sQueue_.ProducerAcquire();
        sQueue_.ProducerReleaseFix();
    }
};
"""

_VEC_H = """
class FlashAttentionVec {
    __aicore__ inline void Process() {
        WorkspaceQueue<float, RING_SLOTS> pQueue_;
        auto p = pQueue_.ConsumerAcquire();
        pQueue_.ConsumerReleaseMte2();
    }
};
"""


def _cv_agent_stock() -> dict[str, str]:
    return {
        "kernel/workspace_queue.h": _REAL_WQ,
        "kernel/flash_attention_cube.h": _CUBE_H,
        "kernel/flash_attention_vec.h": _VEC_H,
    }


def test_cv_agent_stock_structure_passes():
    """cv-agent stock = hand Mmad + cube/vec split + real WQ → MUST pass."""
    ws = _ws(_cv_agent_stock())
    assert _gate()(ws) is None


def test_hand_mmad_no_matmul_library_passes():
    """Hand Mmad with 0 `Matmul<` MUST pass (old syntactic Check 2 wrongly
    rejected this; cv-agent itself uses hand Mmad).
    """
    files = _cv_agent_stock()
    assert "Matmul<" not in files["kernel/flash_attention_cube.h"]
    assert "Mmad(" in files["kernel/flash_attention_cube.h"]
    assert _gate()(_ws(files)) is None


def test_monolithic_no_split_fails():
    ws = _ws({"kernel/fusion_attention_kernel.h":
              "class FaKernel { void f(){ Mmad(a,b,c,p);"
              " WorkspaceQueue q; } };"})
    v = _gate()(ws)
    assert v is not None and "check1" in v


def test_cube_hard_gated_off_fails():
    files = _cv_agent_stock()
    files["kernel/flash_attention_cube.h"] += "\nconst bool cube_eligible = false && x;\n"
    v = _gate()(_ws(files))
    assert v is not None and "check2_test5bis" in v


def test_cube_bypass_zero_matmul_fails():
    """Pure-VEC FA (0 Mmad, 0 Matmul lib) = Antipattern B / OL-188."""
    files = {
        "kernel/fa_cube.h": "class FaCube { void f(){ /* no matmul */ } };",
        "kernel/fa_vec.h": _VEC_H,
        "kernel/workspace_queue.h": _REAL_WQ,
    }
    v = _gate()(_ws(files))
    assert v is not None and "check2_test5bis" in v and "cube-bypass" in v


def test_empty_shell_workspacequeue_decoy_fails():
    """SEMANTIC catch: empty `class WorkspaceQueue {}` wrapping inline flags
    must FAIL (same decoy class as POC empty MatmulKernel<>).
    """
    decoy_wq = "class WorkspaceQueue { /* empty shell, no Producer/Consumer */ };"
    files = {
        "kernel/fa_cube.h": "#include \"x.h\"\nclass FaCube { void f(){ Mmad(a,b,c,p);"
                            " CrossCoreSetFlag<0x2,PIPE_FIX>(0); } };\n" + decoy_wq,
        "kernel/fa_vec.h": "class FaVec { void f(){} };",
    }
    v = _gate()(_ws(files))
    assert v is not None and "check3_semantic" in v


def test_inline_flag_spam_with_real_wq_fails():
    """Even with real WQ, >5 file-level CrossCoreSetFlag = inline spam leaking
    outside the queue = Antipattern A.
    """
    files = _cv_agent_stock()
    files["kernel/flash_attention_cube.h"] += "\n" + "\n".join(
        f"CrossCoreSetFlag<0x2,PIPE_MTE3>({i});" for i in range(6))
    v = _gate()(_ws(files))
    assert v is not None and "check4" in v


def test_comments_mentioning_flags_do_not_trip_check4():
    """2026-05-26 review: doc comments with 'CrossCoreSetFlag' must NOT
    count toward check4. Real flags = 4 (in WQ); comments add mentions; raw
    grep would hit >5 and mis-fail. Comment-strip must keep this PASS.
    """
    files = _cv_agent_stock()
    files["kernel/workspace_queue.h"] += (
        "\n// NOTE: hand-off goes through CrossCoreSetFlag-based slot notify.\n"
        "// The producer's CrossCoreSetFlag and consumer's CrossCoreSetFlag\n"
        "// and another CrossCoreSetFlag mention here would be 3 comment hits.\n"
    )
    assert _gate()(_ws(files)) is None


def test_class_inside_block_comment_does_not_satisfy_check1():
    """Class defined only inside /* */ comment must NOT satisfy split (comment-
    stripping removes it).
    """
    files = {
        "kernel/fa_cube.h": "/* class FaCube {}; class FaVec {}; */\n"
                            "class RealMono { void f(){ Mmad(a,b,c,p); } };",
        "kernel/fa_vec.h": "/* placeholder */",
    }
    v = _gate()(_ws(files))
    assert v is not None and "check1" in v


def test_missing_kernel_dir_fails():
    v = _gate()(Path(tempfile.mkdtemp()))
    assert v is not None and "precond" in v
