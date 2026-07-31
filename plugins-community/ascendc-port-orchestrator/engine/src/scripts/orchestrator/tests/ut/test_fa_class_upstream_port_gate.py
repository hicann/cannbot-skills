# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression: fa_class_structure gate accepts the upstream-faithful (wholeport/
regbase) FA-A5 architecture while still rejecting cube-bypass + monolithic.

Origin 2026-06-11: the gate's non-recursive glob missed kernel/wholeport/ and its
checks were calibrated to the cv-agent hand-roll (Cube/Vec naming + WorkspaceQueue
+ small headers), so it rejected EVERY faithfully-ported upstream kernel — incl.
the cv-agent's own finalized 3_FusionAttention reference. Owner ratified fixing the
gate (option a) to accept the upstream architecture (Set/WaitCrossCore sync, large
engines) without weakening the real anti-pattern guards.
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, "src/scripts/orchestrator/plugins")
from _fa_class_gate import pre_build_check_test_5bis  # noqa: E402


def _mk(files: dict) -> Path:
    d = Path(tempfile.mkdtemp()) / "ws"
    k = d / "kernel"
    k.mkdir(parents=True)
    for rel, content in files.items():
        p = k / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


def test_upstream_wholeport_passes():
    ws = _mk({
        "wholeport/wp_block_cube.h": "// cube engine\nvoid c(){ Mmad(a,b,c); }\n",
        "wholeport/wp_block_vec_base.h": "// vec engine\nvoid v(){ Softmax(x); ReduceMax(x); }\n",
        "regbase_buffer.h": "void sync(){ SetCrossCore(1); WaitCrossCore(1); }\n",
    })
    assert pre_build_check_test_5bis(ws) is None


def test_upstream_cube_bypass_rejected():
    # wholeport present but NO cube matmul (pure-vec masquerading as FA).
    ws = _mk({
        "wholeport/wp_block_vec_base.h": "void v(){ Softmax(x); }\n",
        "regbase_buffer.h": "void sync(){ SetCrossCore(1); }\n",
    })
    r = pre_build_check_test_5bis(ws)
    assert r is not None and "check2" in r


def test_upstream_monolithic_rejected():
    # cube + vec in the SAME file (no engine separation) → Antipattern A.
    ws = _mk({
        "wholeport/wp_block_all.h": "void all(){ Mmad(a,b,c); Softmax(x); }\n",
        "regbase_buffer.h": "void sync(){ SetCrossCore(1); }\n",
    })
    r = pre_build_check_test_5bis(ws)
    assert r is not None and "check1" in r


def test_upstream_no_sync_rejected():
    ws = _mk({
        "wholeport/wp_block_cube.h": "void c(){ Mmad(a,b,c); }\n",
        "wholeport/wp_block_vec_base.h": "void v(){ Softmax(x); }\n",
        "regbase_buffer.h": "void noop(){}\n",
    })
    r = pre_build_check_test_5bis(ws)
    assert r is not None and "check3" in r
