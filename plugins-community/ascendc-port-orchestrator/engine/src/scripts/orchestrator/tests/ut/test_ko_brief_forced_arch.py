# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Forced-architecture honor in ko_brief (2026-06-16, companion to the kw side).

After kw correctly implements a FORCED architecture (forced-SIMT / forced-SIMD
marker in op_classification.json) and precision passes, ko (the optimizer) must
NOT undo the mandate for performance: ko has an Outcome-B architecture-rewrite
path + the OL-54 reg-based-SIMD lever, so it could silently switch a forced-SIMT
op back to SIMD — defeating the forced-architecture mandate and the owner's
SIMT-vs-SIMD comparison.

This pins that ko, on a forced-arch op, is told to:
  - optimize ONLY WITHIN the forced architecture;
  - NOT take the Outcome-B architecture-rewrite path;
  - NOT apply the OL-54 reg-based-SIMD lever to SWITCH architecture;
  - report a forced-arch perf CEILING as a valid conclusion (the comparison
    datapoint), NOT silently switch;
  - treat architecture switch as owner-approval-only (out of ko autonomy).

Non-forced ops: ko behavior UNCHANGED (Outcome-B / OL-54 lever available).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # ../../ → orchestrator/

from briefs.ko_brief import (  # noqa: E402
    _forced_architecture_ko_block,
    _optimizer_phase_block,
)


def _ws(tmp_path: Path, cls: dict | None) -> Path:
    ws = tmp_path / "op"
    ws.mkdir(parents=True)
    if cls is not None:
        (ws / "op_classification.json").write_text(json.dumps(cls))
    return ws


def _assert_ko_forced_complete(block: str, arch: str) -> None:
    assert "ARCHITECTURE IS FIXED" in block
    assert arch in block
    # optimize within only
    assert "WITHIN" in block
    # no Outcome-B rewrite
    assert "Outcome-B" in block
    # no OL-54 arch switch
    assert "OL-54" in block
    # report ceiling as valid conclusion
    assert "ceiling" in block.lower()
    assert "KO_PERF_PLATEAU" in block
    # owner-approval-only switch
    assert "owner-approval" in block.lower()


def test_ko_forced_block_simt(tmp_path):
    ws = _ws(tmp_path, {"force_simt": True})
    _assert_ko_forced_complete(_forced_architecture_ko_block(ws), "SIMT")


def test_ko_forced_block_simd(tmp_path):
    ws = _ws(tmp_path, {"force_simd": True})
    _assert_ko_forced_complete(_forced_architecture_ko_block(ws), "SIMD")


def test_ko_forced_block_bare_tag(tmp_path):
    # Preserve the generic SIMT-force tag mechanism.
    ws = _ws(tmp_path, {"op_class_tags": ["SIMT", "scan", "migration"]})
    _assert_ko_forced_complete(_forced_architecture_ko_block(ws), "SIMT")


def test_ko_forced_block_empty_for_nonforced(tmp_path):
    ws = _ws(tmp_path, {"op_class_tags": ["migration", "reduction"]})
    assert _forced_architecture_ko_block(ws) == ""


def test_ko_forced_block_empty_when_no_classification(tmp_path):
    ws = tmp_path / "op"
    ws.mkdir(parents=True)
    assert _forced_architecture_ko_block(ws) == ""


def test_phase_block_forced_simt_carries_instruction(tmp_path):
    """The AscendC ko phase block for a forced-SIMT op carries the
    optimize-within-only / no-Outcome-B / no-OL-54-switch / report-ceiling /
    owner-approval instruction, AND still has the normal tuning phases.
    """
    ws = _ws(tmp_path, {"op_class_tags": ["SIMT", "scan", "migration"],
                        "force_simt": True})
    phases = _optimizer_phase_block(
        directive_text=None, handoff_from_worker=None, iter_cap=3,
        plugin=None, workspace=ws,
    )
    _assert_ko_forced_complete(phases, "SIMT")
    assert "PHASES (aog-kernel-optimizer)" in phases


def test_phase_block_nonforced_unchanged(tmp_path):
    """A non-forced op's ko phase block is byte-identical to the
    no-classification baseline (no forced block injected).
    """
    ws_forced = _ws(tmp_path / "a", {"op_class_tags": ["reduction"]})
    ws_plain = tmp_path / "b" / "op"
    ws_plain.mkdir(parents=True)  # no op_classification.json
    a = _optimizer_phase_block(None, None, 3, plugin=None, workspace=ws_forced)
    b = _optimizer_phase_block(None, None, 3, plugin=None, workspace=ws_plain)
    assert "ARCHITECTURE IS FIXED" not in a
    assert "ARCHITECTURE IS FIXED" not in b
    assert a == b  # forced-block injection is the ONLY difference, and it's absent


def test_phase_block_default_no_workspace_unchanged(tmp_path):
    """Back-compat: calling without workspace (old call shape) yields the plain
    AscendC phase block with no forced block.
    """
    block = _optimizer_phase_block(None, None, 3, plugin=None)
    assert "ARCHITECTURE IS FIXED" not in block
    assert "PHASES (aog-kernel-optimizer)" in block
