# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""The a3 FA-class brief delivers P-P116 + PB-55; the a5 brief must not (DEBT-222).

Two KB artifacts were merged for the a3/arch22 cube+vector MIX attention path but
were NOT wired into any worker brief:
  - **P-P116** (`fa_class_a3_mix_template.md`): the a3 hand-authored
    2-cube + softmax MIX attention STARTING SKELETON — the a3 counterpart of the
    a5-only P-P103 `fa_class_template.md`.
  - **PB-55**: the `MIX_AIC_1_2` reverse (AIV→AIC) handshake is per-subblock
    COUNTED, so BOTH AIV subblocks must set the reverse flag or the AIC deadlocks.

Both declare `applies_to: soc=Ascend910_9382` + `unverified_on: soc=Ascend950PR`,
so the delivery is a3-only and must NOT leak into the a5 brief (that would re-open
DEBT-208 — the a5 recipe stays anchored on its own P-P103 / two-paths route).

FIXTURE COVERS BOTH SoC FAMILIES — every claim is asserted against an a3 (220x)
target AND an a5 (351x) target. A single-SoC fixture certifies nothing here: the
whole claim is a DIFFERENCE between targets, satisfied by a composer that ignores
`target` entirely.

MUTATION PIN: deleting the `_fa_a3_mix_skeleton_block()` wiring (or the block
body) from `_fa_assembly_deadlock_warning_block` makes
`test_a3_brief_references_p116_and_pb55` go RED — the whole point of the delivery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # ../../ → orchestrator/

from briefs.kb_scope import (  # noqa: E402
    kb_entry_applies_to_target,
    kb_file_applies_to_target,
)
from briefs.kw_brief_fa import (  # noqa: E402
    _fa_assembly_deadlock_warning_block,
    _fa_class_template_assembly_block,
)

A3_TARGETS = ["a3", "a2"]  # 220x
A5_TARGETS = ["a5"]  # 351x

_A3_SKELETON_HEADING = "### a3 FA-CLASS STARTING SKELETON — P-P116"
_PB55_HEADING = "#### PB-55 — the REVERSE (AIV→AIC) handshake is per-subblock-COUNTED"
_A3_TEMPLATE_MD = "target/ascendc/patterns/domains/fa_class_a3_mix_template.md"


def _fa_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "op_classification.json").write_text(
        json.dumps({"op_class_tags": ["ATTENTION", "FUSED", "SOFTMAX"]})
    )
    return ws


# ---------------------------------------------------------------------------
# 0. The scope is read from the KB's own `applies_to`, not hardcoded.
# ---------------------------------------------------------------------------


def test_p116_and_pb55_scope_to_a3_only_from_the_kb():
    """P-P116 file + PB-55 entry both declare `soc=Ascend910_9382` → a3, not a5.

    If this breaks, the KB artifact's declared scope changed — check the entry
    before the composer. This is what makes the delivery a3-only STRUCTURAL, not a
    prose "do not over-apply".
    """
    for t in A3_TARGETS:
        assert kb_file_applies_to_target(_A3_TEMPLATE_MD, t) is True
        assert kb_entry_applies_to_target("PB-55", t) is True
    for t in A5_TARGETS:
        assert kb_file_applies_to_target(_A3_TEMPLATE_MD, t) is False
        assert kb_entry_applies_to_target("PB-55", t) is False


# ---------------------------------------------------------------------------
# 1. MUTATION TARGET — the a3 brief carries P-P116 + PB-55.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", A3_TARGETS)
def test_a3_brief_references_p116_and_pb55(target):
    """The a3 FA brief delivers BOTH artifacts, WITH their load-bearing content.

    Dies if the `_fa_a3_mix_skeleton_block()` wiring or body is removed — a no-op
    PR is worse than none, so this asserts the substantive facts, not just the ids.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    # P-P116 skeleton, pointed at the real file.
    assert _A3_SKELETON_HEADING in out
    assert "P-P116" in out
    assert "fa_class_a3_mix_template.md" in out
    # The a3↔a5 divergence a worker must not get wrong: OMIT the arch35-only macro.
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)" in out
    assert "an a3 worker OMITS it" in out
    assert "IterateAll<sync=true>" in out
    # PB-55 reverse-handshake rule — the both-subblocks-set fix, verbatim intent.
    assert _PB55_HEADING in out
    assert "BOTH AIV subblocks MUST `CrossCoreSetFlag(FLAG_P)`" in out
    assert "per-subblock-COUNTED" in out
    assert "single-setter reverse=DEADLOCK" in out
    # PB-55 also lands as a scoped fix-card.
    assert "PB-55 (reverse AIV→AIC handshake is per-subblock-COUNTED" in out


_A3_COMPILABLE_EXAMPLE_PATH = "examples/a3_mix_fa_min"


@pytest.mark.parametrize("target", A3_TARGETS)
def test_a3_brief_points_at_compilable_mix_reference(target):
    """The a3 brief carries the STRICT-BAR THREE-STAGE discipline (post #188/#189).

    The owner set a STRICT bar: an op-gen worker GENERATES from KB knowledge + the
    customer's SHIPPED libraries — it does NOT copy a liftable artifact. #188 reworked
    P-P116 into a reusable generate-from-knowledge template (with a shipped-library
    steering table); #189 thinned `examples/a3_mix_fa_min/` into a handshake-only
    SYNC-WITNESS whose compute is a zero-liftable PLACEHOLDER. The old "MATERIALIZE
    the full compilable reference" framing (#186) is now STALE — the reference is a
    sync demonstrator, the pattern lives in the template. This asserts the composed a3
    brief now points a worker through the three stages, in order: READ the template
    (shipped-library steering) → BUILD+RUN the witness (deadlock-free, placeholder,
    NOT copyable) → GENERATE your own with shipped-library primitives (do not lift).
    Removing/reverting the pointer block from `_fa_a3_mix_skeleton_block()` turns this
    RED — a stale "materialize a copyable op" pointer is exactly the delivery-failure
    this lane exists to fix.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    # The three-stage discipline is announced up front.
    assert (
        "THREE-STAGE DISCIPLINE — READ the template → BUILD+RUN the witness → "
        "GENERATE your own with shipped libraries" in out
    )
    assert "you do NOT copy a\nliftable artifact" in out

    # STAGE 1 — READ the reworked template P-P116 as reusable pattern KNOWLEDGE,
    # incl. its shipped-library steering table (USE the customer's CANN/catlass).
    assert "**READ the template P-P116**" in out
    assert "reusable pattern KNOWLEDGE" in out
    assert "**shipped-library steering table**" in out
    assert "USE CANN `MatmulImpl` / `AscendC::SoftMax` /" in out
    assert "catlass `CrossCore`" in out
    # Our hand-written helpers are op-glue the worker GENERATES, never lifts.
    assert "op-glue you GENERATE yourself, do NOT lift" in out

    # STAGE 2 — BUILD + RUN the SYNC-WITNESS: it demonstrates the AIC↔AIV handshake
    # closing deadlock-free; its compute is a PLACEHOLDER, so it is NOT a copyable op.
    assert _A3_COMPILABLE_EXAMPLE_PATH in out
    assert "src/skills/references/target/ascendc/examples/a3_mix_fa_min/" in out
    assert "**BUILD + RUN the SYNC-WITNESS**" in out
    assert "close\n" in out and "**deadlock-free** on device" in out
    assert "**PLACEHOLDER**" in out
    assert "sync DEMONSTRATOR, NOT a copyable op" in out

    # STAGE 3 — GENERATE your own kernel + host using SHIPPED-library primitives;
    # do NOT lift the witness body / our helpers / any non-shipped source.
    assert "**GENERATE your own kernel + host**" in out
    assert "using shipped-library primitives" in out
    assert (
        "Do **NOT** lift the witness body, our example's hand-written\n"
        "   helpers, or any non-shipped source." in out
    )
    # The customer-portability test + the anti-copy backstops.
    assert "does the customer's own CANN/catlass\n   BUILD it?" in out
    assert "DEBT-215" in out


@pytest.mark.parametrize("target", A5_TARGETS)
def test_a5_brief_does_not_gain_compilable_mix_reference(target):
    """The compilable MIX example is a3-only — it must NOT leak into the a5 brief.

    The reference is `arch22`/`Ascend910_9382` and makes NO claim on a5; it rides the
    same a3-only `_fa_a3_mix_skeleton_block()` gate as P-P116/PB-55, so an a5 brief
    that gained the pointer would be the DEBT-208 leak in the other direction.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert _A3_COMPILABLE_EXAMPLE_PATH not in out
    assert "a3_mix_fa_min" not in out


@pytest.mark.parametrize("target", A3_TARGETS)
def test_a3_brief_carries_p116_honest_scope(target):
    """The skeleton is delivered WITH its bounds — not as a finished flash-attention.

    Stripping the honest-scope bullets would be the same disease in the other
    direction (an unbounded "here's your FA skeleton" claim). All must survive.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert "seqlen ≤ 384" in out
    assert "single-pass NON-flash" in out
    assert "causal / attention mask (not wired" in out
    assert "vendor FA baseline" in out
    assert "unverified_on: soc=Ascend950PR" in out


# ---------------------------------------------------------------------------
# 2. The delivery is a3-only — it must not leak into the a5 brief (DEBT-208).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", A5_TARGETS)
def test_a5_brief_does_not_gain_p116_or_pb55(target):
    """P-P116 / PB-55 are `unverified_on: Ascend950PR` → ABSENT on a5.

    Dies if the a3 skeleton block is composed unconditionally (the DEBT-208 defect
    in the other direction: an a3-only card leaking to a5).
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert _A3_SKELETON_HEADING not in out
    assert _PB55_HEADING not in out
    assert "P-P116" not in out
    assert "fa_class_a3_mix_template" not in out
    assert "PB-55 (" not in out


def test_a5_brief_keeps_its_own_p103_route_unchanged():
    """The a5 branch still references P-P103 / its two-paths recipe — no regression."""
    a5 = _fa_class_template_assembly_block  # composed full brief below
    ws_tags = {"op_class_tags": ["ATTENTION", "FUSED", "SOFTMAX"]}
    import tempfile

    d = Path(tempfile.mkdtemp())
    (d / "op_classification.json").write_text(json.dumps(ws_tags))
    brief = a5("3_FusionAttention", d, target="a5")
    assert brief is not None
    assert "P-P103" in brief
    assert "fa_class_template.md" in brief
    assert "P-P116" not in brief
    assert "PICK EXACTLY ONE PATH" in brief


# ---------------------------------------------------------------------------
# 3. End-to-end through the real composer entry point.
# ---------------------------------------------------------------------------


def test_composed_a3_brief_delivers_p116_pb55_end_to_end(tmp_path: Path):
    """`_fa_class_template_assembly_block(op, ws, target='a3')` — the real path."""
    ws = _fa_ws(tmp_path)
    a3 = _fa_class_template_assembly_block("3_FusionAttention", ws, target="a3")
    a5 = _fa_class_template_assembly_block("3_FusionAttention", ws, target="a5")
    assert a3 is not None and a5 is not None
    # a3 gets the delivery; a5 does not.
    assert "P-P116" in a3 and _PB55_HEADING in a3
    assert "P-P116" not in a5 and "PB-55" not in a5
    # a5 keeps its own template reference.
    assert "fa_class_template.md" in a5
