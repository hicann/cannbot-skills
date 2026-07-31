# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Forced-architecture handling in kw_brief.

When an op's architecture is FIXED at classification time (forced-SIMT /
forced-SIMD marker in op_classification.json), the kw_brief phase block MUST:
  - emit an "ARCHITECTURE IS FIXED" instruction block telling kw to implement
    the forced architecture, NOT run the SIMT_VS_SIMD decision tree, NOT
    override to another architecture, and that architecture-change is a
    ko-stage (post-precision, performance-driven) decision;
  - NOT point kw at the SIMT_VS_SIMD decision tree (KB manifest suppression).

Non-forced ops are UNCHANGED — they still get the cold-start phases without the
forced block (the SIMT_VS_SIMD decision tree remains available to them).

Root cause: kw was given a forced-SIMT classification, but during Phase A ran
the SIMT_VS_SIMD decision tree itself, re-classified selective_scan
recurrence→SIMD, and OVERRODE the forced SIMT — before precision was even
aligned, on a signal from its own decision logic (not ko, not perf). Overreach
+ wrong timing. Correct flow: architecture is FIXED at classification; kw only
implements; precision aligns first; only ko (post-precision, perf) may propose
an architecture change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # ../../ → orchestrator/

from briefs.kw_brief import (  # noqa: E402
    _detect_forced_architecture,
    _forced_architecture_block,
    _phase_instructions_block,
)
from briefs._common import kb_manifest_block  # noqa: E402


# --- marker detection ------------------------------------------------------

def _ws(tmp_path: Path, cls: dict | None) -> Path:
    ws = tmp_path / "op"
    ws.mkdir(parents=True)
    if cls is not None:
        (ws / "op_classification.json").write_text(json.dumps(cls))
    return ws


def _migration_env() -> SimpleNamespace:
    """Minimal environment for the supported arch22→arch35 worker route."""
    return SimpleNamespace(
        opgen_mode="port_a3_to_a5",
        target="a5",
        port_a3_source="/fixture/arch22/op",
        host="fixture-host",
        container="fixture-container",
    )


def test_detect_force_simt_bool_key(tmp_path):
    ws = _ws(tmp_path, {"op_class_tags": ["migration"], "force_simt": True})
    assert _detect_forced_architecture(ws) == "SIMT"


def test_detect_force_simd_bool_key(tmp_path):
    ws = _ws(tmp_path, {"op_class_tags": ["port_a3"], "force_simd": True})
    assert _detect_forced_architecture(ws) == "SIMD"


def test_detect_forced_arch_string_key(tmp_path):
    ws = _ws(tmp_path, {"op_class_tags": [], "forced_arch": "simt"})
    assert _detect_forced_architecture(ws) == "SIMT"


def test_detect_bare_simt_tag(tmp_path):
    # Preserve the generic SIMT-force tag mechanism.
    ws = _ws(tmp_path, {"op_class_tags": ["SIMT", "scan", "migration"]})
    assert _detect_forced_architecture(ws) == "SIMT"


def test_detect_none_when_unmarked(tmp_path):
    ws = _ws(tmp_path, {"op_class_tags": ["migration", "reduction"]})
    assert _detect_forced_architecture(ws) is None


def test_detect_none_when_no_classification(tmp_path):
    ws = _ws(tmp_path, None)
    assert _detect_forced_architecture(ws) is None


def test_detect_ambiguous_both_tags_is_none(tmp_path):
    # if both SIMT and SIMD appear, don't guess — treat as non-forced
    ws = _ws(tmp_path, {"op_class_tags": ["SIMT", "SIMD"]})
    assert _detect_forced_architecture(ws) is None


# --- forced block content --------------------------------------------------

def _assert_forced_block_complete(block: str, arch: str) -> None:
    assert "ARCHITECTURE IS FIXED" in block
    assert arch in block
    # do NOT run the decision tree
    assert "SIMT_VS_SIMD" in block
    assert "decision tree" in block.lower()
    # do NOT override
    assert "Do NOT override" in block or "do NOT override" in block.lower()
    # ko-only, post-precision, performance-driven
    assert "ko" in block and "aog-kernel-optimizer" in block
    assert "precision" in block.lower()
    # document concern but implement
    assert "analysis.md" in block


def test_forced_block_simt_content(tmp_path):
    ws = _ws(tmp_path, {"force_simt": True})
    block = _forced_architecture_block(ws)
    _assert_forced_block_complete(block, "SIMT")


def test_forced_block_simd_content(tmp_path):
    ws = _ws(tmp_path, {"force_simd": True})
    block = _forced_architecture_block(ws)
    _assert_forced_block_complete(block, "SIMD")


def test_forced_block_empty_for_nonforced(tmp_path):
    ws = _ws(tmp_path, {"op_class_tags": ["migration"]})
    assert _forced_architecture_block(ws) == ""


# --- phase block integration (the load-bearing path) -----------------------

def test_phase_block_forced_simt_emits_fixed_instruction(tmp_path):
    """Cold-start phase block for a forced-SIMT op carries the
    'architecture FIXED / do NOT override / do NOT run decision tree / ko-only'
    instruction.
    """
    ws = _ws(tmp_path, {"op_class_tags": ["SIMT", "scan", "migration"],
                        "force_simt": True})
    phases = _phase_instructions_block(
        "selective_scan", ws, iter_cap_remaining=8,
        directive_text=None, handoff_from_prior=None,
        env=_migration_env(),
    )
    _assert_forced_block_complete(phases, "SIMT")
    # The supported migration phases still follow the fixed-architecture block.
    assert "PHASES (port_from_a3_ascendc — arch22→arch35 port mode" in phases


def test_phase_block_nonforced_unchanged(tmp_path):
    """A non-forced op's phase block is byte-identical to the no-classification
    baseline (no forced block injected).
    """
    ws_forced = _ws(tmp_path / "a", {"op_class_tags": ["reduction"]})
    ws_plain = tmp_path / "b" / "op"
    ws_plain.mkdir(parents=True)  # no op_classification.json at all
    a = _phase_instructions_block("foo", ws_forced, iter_cap_remaining=8,
                                  directive_text=None, handoff_from_prior=None,
                                  env=_migration_env())
    b = _phase_instructions_block("foo", ws_plain, iter_cap_remaining=8,
                                  directive_text=None, handoff_from_prior=None,
                                  env=_migration_env())
    assert "ARCHITECTURE IS FIXED" not in a
    assert "ARCHITECTURE IS FIXED" not in b
    assert a == b  # forced-block injection is the ONLY difference, and it's absent


def test_kb_manifest_suppresses_decision_tree_for_forced(tmp_path):
    """KB manifest for a forced-arch op does NOT load SIMT_VS_SIMD_DECISION."""
    ws = _ws(tmp_path, {
        "op_class_tags": ["SIMT", "migration"],
        "force_simt": True,
        "kb_recommendations": [
            {"path": "target/ascendc/SIMT_VS_SIMD_DECISION.md"},
        ],
    })
    manifest = kb_manifest_block(
        "selective_scan", workspace=ws, target="a5", force_legacy_kb=True,
    )
    assert "SIMT_VS_SIMD_DECISION" not in manifest


def test_kb_manifest_keeps_decision_tree_for_nonforced(tmp_path):
    """Non-forced op still gets the SIMT_VS_SIMD decision tree if recommended."""
    ws = _ws(tmp_path, {
        "op_class_tags": ["scatter-gather"],
        "kb_recommendations": [
            {"path": "target/ascendc/SIMT_VS_SIMD_DECISION.md"},
        ],
    })
    manifest = kb_manifest_block(
        "some_scatter_op", workspace=ws, target="a5", force_legacy_kb=True,
    )
    assert "SIMT_VS_SIMD_DECISION" in manifest
