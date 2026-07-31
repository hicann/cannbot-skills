# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Brief composers honor each KB entry's own `applies_to: soc=` (DEBT-208).

The defect: `kw_brief_fa` composed PB-34's MIX-KFC deadlock warning
UNCONDITIONALLY into the FA-class forward template-assembly brief, with the scope
carried only as PROSE inside the injected text ("V351/A5 scope bound — do NOT
over-apply"). On A5 that INVERTED the KB's own advice — PB-34 declares
`applies_to: soc=Ascend910_9382` (V220) and carries two
`verified_does_not_reproduce_on: Ascend950PR` witnesses (the GDN full-op
light-port ran 122/122 T1 PASS), and its Consequence line makes that light-port
the DEFAULT A5 route. The warning steered A5 workers off the recommended route;
an FA worker retreated to vector-only.

FIXTURE COVERS BOTH SoC FAMILIES — every scope assertion is made against a V351
target AND a V220 target. A single-SoC fixture would certify nothing here: the
whole claim is a DIFFERENCE between targets, and an A5-only (or V220-only)
fixture is satisfied by a composer that ignores `target` entirely.

The trap this feature had to avoid is pinned explicitly: PB-35's `applies_to`
said V220-only until 2026-07-17 while its own `confirmed_on` recorded an A5
deadlock, so honoring `applies_to` naively would have SUPPRESSED PB-35 on the one
SoC where it is CONFIRMED — turning an over-block fix into an under-block.
`test_pb35_still_reaches_an_a5_worker` is that pin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # ../../ → orchestrator/

from briefs.kb_scope import (  # noqa: E402
    applies_to_target,
    kb_entry_applies_to_target,
    kb_entry_soc_families,
    kb_section_applies_to_target,
    kb_section_soc_families,
    soc_family_for_target,
)
from briefs.kw_brief_fa import (  # noqa: E402
    _fa_assembly_deadlock_warning_block,
    _fa_class_template_assembly_block,
)

# The two SoC families the KB spells ~20 ways, by the target names that select
# them. Both halves of every scope claim below are exercised against this.
V351_TARGETS = ["a5"]
V220_TARGETS = ["a3", "a2"]

_PB34_HEADING = "### PB-34 — MIX cube+vec SILENT-HANG"
_PB35_HEADING = "### PB-35 — the Pattern-A trap"
_CCS_MD = "target/ascendc/fa_class/cross_core_sync.md"


def _fa_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "op_classification.json").write_text(
        json.dumps({"op_class_tags": ["ATTENTION", "FUSED", "SOFTMAX"]})
    )
    return ws


# ---------------------------------------------------------------------------
# 1. The scope is read from the KB, not hardcoded — against the REAL entries.
# ---------------------------------------------------------------------------


def test_real_kb_entries_parse_to_expected_families():
    """The predicate reads real `applies_to: soc=` lines, in the KB's own spellings.

    These are the four spellings that matter here, and they are all different:
    `soc=Ascend910_9382 (V220 A2/A3 single-die)` / `soc=Ascend910_9382,Ascend950PR_9579`
    / `soc=Ascend950PR (V351/A5)` / `soc=Ascend950PR`. If this test breaks, a KB
    entry's declared scope changed — check the entry before touching the parser.
    """
    assert kb_entry_soc_families("PB-34") == {"V220"}
    assert kb_entry_soc_families("PB-35") == {"V220", "V351"}
    assert kb_entry_soc_families("PB-45") == {"V351"}
    assert kb_entry_soc_families("OL-220") == {"V351"}
    assert kb_section_soc_families(_CCS_MD, "4") == {"V351"}


def test_unknown_entry_and_unknown_target_fail_open():
    """FAIL-OPEN: suppress only on a POSITIVE machine-readable exclusion.

    An entry we cannot find, or a target whose family we don't know, must keep the
    injection — this feature may only ever narrow an over-block, never invent a new
    under-block.
    """
    assert kb_entry_soc_families("PB-99999") is None
    assert kb_entry_applies_to_target("PB-99999", "a5") is True
    assert kb_entry_applies_to_target("PB-34", "some-future-soc") is True
    assert applies_to_target(None, "a5") is True
    assert applies_to_target({"*"}, "a3") is True  # soc=all
    assert soc_family_for_target(None) is None


def test_soc_family_map_covers_both_families():
    assert {soc_family_for_target(t) for t in V351_TARGETS} == {"V351"}
    assert {soc_family_for_target(t) for t in V220_TARGETS} == {"V220"}


# ---------------------------------------------------------------------------
# 2. MUTATION TARGET — the PB-34 predicate. Both SoC families asserted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", V351_TARGETS)
def test_pb34_warning_absent_on_v351(target):
    """V351/A5 → the V220-only warning is ABSENT.

    Dies if the `if kb_entry_applies_to_target("PB-34", target)` predicate is
    removed (unconditional injection = the DEBT-208 defect restored).
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert _PB34_HEADING not in out
    assert "PB-34 (root cause + Pattern A/B exclusivity)" not in out


@pytest.mark.parametrize("target", V220_TARGETS)
def test_pb34_warning_present_on_v220(target):
    """V220 → the warning is PRESENT (the recall-fix #126 closed a REAL gap).

    Dies if the predicate is inverted or made always-false — i.e. this is the
    guard against "fix the over-block by deleting the warning".
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert _PB34_HEADING in out
    assert "KFC slot contention" in out
    assert "fa_fused_mixed_fp16" in out


def test_pb34_scope_prose_is_no_longer_the_enforcement():
    """The V220 warning no longer ships the "do NOT over-apply" compliance plea.

    The bound is the composer's `if`, not an instruction the reading LLM must obey.
    """
    v220 = _fa_assembly_deadlock_warning_block("a3")
    assert "do NOT over-apply this warning" not in v220


# ---------------------------------------------------------------------------
# 3. THE TRAP — honoring applies_to must not become an UNDER-block.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", V351_TARGETS + V220_TARGETS)
def test_pb35_still_reaches_an_a5_worker(target):
    """PB-35 reaches BOTH SoCs — its `applies_to` names both.

    This is the pin for the trap DEBT-208 had to clear: PB-35 is `confirmed_on`
    A5, so suppressing it there would be strictly worse than the over-block being
    fixed. On A5 it is the mode that actually bites once PB-34 is scoped away.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert _PB35_HEADING in out
    assert "event_t(0)" in out


def test_a5_worker_is_not_left_without_cross_core_guidance():
    """Suppressing PB-34 on A5 must REPLACE it with the A5 recipe, not a hole."""
    out = _fa_assembly_deadlock_warning_block("a5")
    assert "PICK EXACTLY ONE PATH" in out
    assert "cross_core_sync.md` §4" in out or "§4" in out


# ---------------------------------------------------------------------------
# 4. Part 2 — TWO MUTUALLY EXCLUSIVE PATHS, never a menu.
# ---------------------------------------------------------------------------


def test_two_paths_surfaced_on_a5_and_scoped_off_v220():
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert "#### PATH B — library cube + KFC-IMPLICIT sync" in a5
    assert "#### PATH A — non-KFC library cube + the §4 manual handshake" in a5
    # The A5 recipe is anchored on two `soc=Ascend950PR` entries → not on V220.
    for t in V220_TARGETS:
        assert "PICK EXACTLY ONE PATH" not in _fa_assembly_deadlock_warning_block(t)


def test_paths_are_exclusive_not_a_menu():
    """Blending IS the bug — Path B must forbid §4 rather than offer it.

    A summary reading "library cube + §4 handshake" is an invitation to re-create
    `fa_fused_mixed_fp16`'s exact failing structure.
    """
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert "MUTUALLY EXCLUSIVE" in a5
    assert "Blending them IS the bug" in a5
    assert "§4 DOES NOT APPLY TO THIS PATH. Do NOT add it." in a5
    assert "ZERO manual `CrossCoreSetFlag`" in a5
    assert "Take ONE path WHOLE" in a5


def test_path_b_carries_its_full_op_witness():
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert "122/122 T1 PASS" in a5
    assert "OL-220" in a5


def test_path_a_bounds_are_not_stripped():
    """The precisions a paraphrase has stripped before — all three must survive."""
    a5 = _fa_assembly_deadlock_warning_block("a5")
    # (A)/(B)/(C) as §4 defines them
    assert "SYNC MODE 4, not mode 2" in a5
    assert "`id` / `id+16`" in a5 and "BOTH must be Set" in a5
    # (C) = consumer waits on PIPE_V, NOT the producer's PIPE_FIX
    assert "the consumer `Wait`s on **`PIPE_V`**, NOT" in a5
    assert "the producer's `PIPE_FIX`" in a5
    # §4's 64/64 witness is the abstraction, NOT a hand-roll
    assert "`Buffer<CROSS_CORE_SYNC_FORWARD>` abstraction, NOT a hand-roll" in a5
    # (C)'s bound: UB-resident only; via-GM needs more
    assert "UB-RESIDENT" in a5
    assert "507015" in a5
    # PB-35's event_t(0) rule reaches the A5 hand-roller
    assert "event_t(0)" in a5


def test_recipe_is_pointed_at_not_copied():
    """POINT, do not COPY — a copy drifts and a paraphrase strips the bounds."""
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert "cross_core_sync.md` §4 (`:209`" in a5
    assert "READ §4 ITSELF — do not work from this summary" in a5
    # Not a transcription of §4's code block.
    assert "AIV1_FLAG_OFFSET = 16" not in a5


def test_intra_aic_hand_roll_is_discouraged_on_both_socs_with_correct_anchors():
    """V220 = UNSOLVED (PLATFORM_BUGS.md:934); V351 = non-deterministic (PB-45)."""
    v220 = _fa_assembly_deadlock_warning_block("a3")
    assert "UNSOLVED in canonical KB" in v220
    assert "PLATFORM_BUGS.md:934" in v220
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert "### PB-45" in a5
    assert "non-deterministic" in a5


# ---------------------------------------------------------------------------
# 4b. The V220 worker is NOT steered to a vector fallback on a false premise.
# ---------------------------------------------------------------------------

_SHIPPED_CUBE_HEADING = "### A non-KFC LIBRARY CUBE HAS SHIPPED ON V220"


@pytest.mark.parametrize("target", V220_TARGETS)
def test_v220_no_longer_says_the_cube_workflow_has_not_landed(target):
    """MUTATION PIN: the false "until the canonical V220 cube workflow lands" steer.

    That trailer was false once DEBT-206 shipped (OL-275,
    `OPERATIONAL_KNOWLEDGE.md:11016` — first SHIPPED verified_on:a3 cube op), and it
    contradicted the PB-34 block's OWN `cube-only` bullet four lines above it. It was
    the clause written in plain imperative language, so it is the one a worker
    obeyed: an attention op went pure-vector because of it.

    Dies if the trailer is restored.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert "until the canonical V220 cube" not in out
    assert "stay on the AIV-only VEC fallback" not in out


@pytest.mark.parametrize("target", V220_TARGETS)
def test_v220_learns_a_non_kfc_cube_has_shipped_with_its_bounds(target):
    """The constructive half: the shipped route reaches the worker WITH its bounds.

    Deleting the block, or stripping the bounds that keep it honest, must go red —
    an unbounded "cube shipped!" claim would be the same disease in the other
    direction (DEBT-206 is cube-only and CANN 9.1.0, not an FA MIX solution).
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert _SHIPPED_CUBE_HEADING in out
    # The route, cited to a real artifact.
    assert "DEBT-206" in out and "OPERATIONAL_KNOWLEDGE.md:11016" in out
    assert "IterateAll<sync=true>" in out and "ASCEND_IS_AIC" in out
    assert "build_ascendc.py" in out
    # The bounds that must never be stripped.
    assert "CUBE-ONLY" in out
    assert "does **NOT** prove the cube↔vec" in out
    assert "CANN 9.1.0" in out and "9.0.0" in out
    # The honest gap is stated plainly rather than filled with an invented recipe.
    assert "NOT PROVEN" in out
    assert "No such recipe exists in the KB" in out


@pytest.mark.parametrize("target", V220_TARGETS)
def test_v220_vector_fallback_is_marked_ship_blocked(target):
    """The fallback is not merely discouraged — a real gate rejects it.

    `_check_architecture_class` (`finalize_checks_structural.py:447`) returns
    ARCHITECTURAL_HACK for a pure-vec kernel on a cube-required op, and
    `kw_brief_pa3_phases` already tells the same worker "NEVER to a pure-VEC
    fallback". The FA brief must not contradict the gate its own worker will hit.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert "SHIP-BLOCKED" in out
    assert "ARCHITECTURAL_HACK" in out


@pytest.mark.parametrize("target", V220_TARGETS)
def test_catlass_v220_is_labelled_unproven_not_recommended(target):
    """An unproven candidate must be labelled, never presented as a recipe.

    catlass `BlockMmadTla` is the vendor's default-arch (2201) V220 example, but the
    KB holds NO V220 execution witness. Writing it as a recipe would be the exact
    defect this change fixes.
    """
    out = _fa_assembly_deadlock_warning_block(target)
    assert "UNPROVEN CANDIDATE" in out
    assert "no V220 execution witness" in out


def test_shipped_cube_block_is_scoped_off_a5():
    """OL-275 is `soc=Ascend910_V220` → the block must not leak into the A5 brief.

    The A5 recipe is anchored on OL-220 ∧ cross_core_sync.md §4 and stays that way;
    this fix may not widen an A5 scope (that would re-create DEBT-208).
    """
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert _SHIPPED_CUBE_HEADING not in a5
    assert "DEBT-206" not in a5


# ---------------------------------------------------------------------------
# 5. Fix-card list — filtered per card, the class fix rather than a PB-34 patch.
# ---------------------------------------------------------------------------


def test_fix_cards_filtered_by_each_cards_own_scope():
    """Each card in the trailer is filtered by its OWN applies_to, not one flag.

    OL-275 is `soc=Ascend910_V220` and its own `unverified_on` says "do not assume
    transfer" to A5; EC-68/OL-220 are `soc=Ascend950PR`. The pre-fix list handed
    all four to every target.
    """
    a5 = _fa_assembly_deadlock_warning_block("a5")
    assert "OL-220 (" in a5 and "EC-68 (" in a5
    assert "OL-275 (" not in a5
    assert "PB-34 (" not in a5

    v220 = _fa_assembly_deadlock_warning_block("a3")
    assert "PB-34 (" in v220 and "OL-275 (" in v220
    assert "EC-68 (" not in v220
    assert "OL-220 (" not in v220


# ---------------------------------------------------------------------------
# 6. End-to-end through the composer + the brief builder.
# ---------------------------------------------------------------------------


def test_template_assembly_block_threads_target(tmp_path: Path):
    ws = _fa_ws(tmp_path)
    a5 = _fa_class_template_assembly_block("3_FusionAttention", ws, target="a5")
    v220 = _fa_class_template_assembly_block("3_FusionAttention", ws, target="a3")
    assert a5 is not None and v220 is not None
    assert _PB34_HEADING not in a5
    assert _PB34_HEADING in v220
    # Both still get the actual FA recipe they came for.
    for out in (a5, v220):
        assert "template-assembly" in out.lower()


def test_template_assembly_block_defaults_to_a5(tmp_path: Path):
    """Default matches the `kb_manifest_block(..., target="a5")` convention."""
    ws = _fa_ws(tmp_path)
    assert _fa_class_template_assembly_block(
        "3_FusionAttention", ws
    ) == _fa_class_template_assembly_block("3_FusionAttention", ws, target="a5")


def test_build_worker_brief_passes_env_target(tmp_path: Path):
    """The wiring that makes this real: `env.target` reaches the FA composer."""
    from briefs.kw_brief import _phase_instructions_block

    ws = _fa_ws(tmp_path)

    class _Env:
        target = "a3"
        opgen_mode = "backward"

    out = _phase_instructions_block(
        "3_FusionAttention", ws, 5, None, None, env=_Env()
    )
    assert _PB34_HEADING in out

    _Env.target = "a5"
    out_a5 = _phase_instructions_block(
        "3_FusionAttention", ws, 5, None, None, env=_Env()
    )
    assert _PB34_HEADING not in out_a5
    assert "PICK EXACTLY ONE PATH" in out_a5
