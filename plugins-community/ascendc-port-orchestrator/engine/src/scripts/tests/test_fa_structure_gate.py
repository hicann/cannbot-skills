# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for fa_structure_gate (#265, path-(a) CLI diagnostic).

Covers the ALIGNED/DIVERGENT/NO_KERNEL verdicts AND documents the heuristic
false-pos/neg limits main flagged in audit (textual grep, no AST/comment scoping).
These are a DIAGNOSTIC's known limits — asserted so they're explicit, not hidden.
"""
import pathlib
import sys

_S = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_S))
import fa_structure_gate as g  # noqa: E402


def _mk(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(body)
    return f


# ---- correct-behavior cases ----

def test_aligned_two_matmul_plus_tiling(tmp_path):
    body = """
    MatmulImpl<AT, BT, CT> bmm1;
    MatmulImpl<AT, BT, CT> bmm2;
    for (uint32_t i = 0; i < sInnerLoopTimes; i++) { /* singleProcessSInnerSize tile */ }
    """
    r = g.gate(_mk(tmp_path, "k.h", body))
    assert r["verdict"] == "ALIGNED", r
    assert r["cube_total"] >= 2 and r["has_s2_tiling"]


def test_divergent_vec_only(tmp_path):
    body = """
    // pure-vec FA: ReduceMax / Exp / ReduceSum, no cube
    ReduceMax(...); Exp(...); ReduceSum(...);
    for (i=0;i<sInnerLoopTimes;i++) {}
    """
    r = g.gate(_mk(tmp_path, "k.h", body))
    assert r["verdict"] == "DIVERGENT" and r["cube_total"] == 0, r


def test_divergent_single_matmul(tmp_path):
    # only QK^T on cube, P@V on vec → 1 cube < 2 required → DIVERGENT
    body = "MatmulImpl<AT,BT,CT> bmm1;\nfor(i=0;i<sInnerLoopTimes;i++){}\n"
    r = g.gate(_mk(tmp_path, "k.h", body))
    assert r["verdict"] == "DIVERGENT" and r["cube_total"] == 1, r


def test_no_kernel_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    r = g.gate(tmp_path / "empty")
    assert r["verdict"] == "NO_KERNEL", r


# ---- documented heuristic LIMITS (false-pos/neg main flagged) ----

def test_known_limit_comment_matmul_overcounts(tmp_path):
    # FALSE-POSITIVE: a matmul mentioned only in a COMMENT still counts (no comment scoping).
    # Documents the limit; an auto-gate (path b) would need AST to avoid this.
    body = """
    // historically used MatmulImpl<A,B,C> for bmm1 and MatmulImpl<A,B,C> for bmm2 (now removed)
    ReduceMax(...);  // actually vec-only now
    for(i=0;i<sInnerLoopTimes;i++){}
    """
    r = g.gate(_mk(tmp_path, "k.h", body))
    # The gate OVER-counts the commented matmuls → falsely ALIGNED. Asserted as a KNOWN limit.
    assert r["cube_total"] == 2, "documents textual-count limit (comment mentions counted)"
    assert r["verdict"] == "ALIGNED", "KNOWN false-positive — diagnostic only, not auto-gate"


def test_known_limit_single_tiling_token_passes(tmp_path):
    # FALSE-POSITIVE: a single sInner* token (even in a comment) satisfies has_s2_tiling.
    body = "MatmulImpl<A,B,C> a;\nMatmulImpl<A,B,C> b;\n// note: singleProcessSInnerSize not actually used\n"
    r = g.gate(_mk(tmp_path, "k.h", body))
    assert r["has_s2_tiling"] is True, "documents single-token tiling-pass limit"


if __name__ == "__main__":
    import tempfile
    # minimal self-run
    import pytest  # noqa
    sys.exit(pytest.main([__file__, "-q"]))
