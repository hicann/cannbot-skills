#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Golden/import lock for the kw_brief god-file decomposition (DEBT-201, 2026-07-06).

kw_brief.py (2847 lines) was split into cohesive sibling modules:
  - kw_brief_fa.py          — FA-class predicates + template-assembly + backward stitch
  - kw_brief_shared.py      — _forced_architecture_block (shared leaf)
  - kw_brief_pa3_phases.py  — port_a3 Phase A/B/C body builders + context (leaf)
  - kw_brief_port_a3.py     — port_a3 orchestrator + Phase D/E/budget bodies

These prompt-string builders are behaviour-bearing: behaviour == the emitted
string. This test (a) proves each public builder is IMPORTABLE from its new
sibling-module home AND still re-exported from `briefs.kw_brief` (the public
surface external callers use), and (b) sha256-locks each builder's output so any
future drift fails here. Fills the direct-UT gap for builders that previously
only had transitive golden coverage via `_port_a3_phase_instructions_block`.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH = _HERE.parent.parent.parent  # tests/ut -> tests -> orchestrator/
_BRIEFS = _ORCH / "briefs"
sys.path.insert(0, str(_ORCH))


class _StubEnv:
    port_a3_source = "/src"
    host = "a5host.example"
    container = "npu_dev3"
    target = "a5"

    def __getattr__(self, name):  # any other env.X -> harmless empty string
        return ""


def _ws(tags):
    d = Path(tempfile.mkdtemp()) / "ws"
    d.mkdir(parents=True, exist_ok=True)
    (d / "op_classification.json").write_text(json.dumps({"op_class_tags": tags}))
    return d


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. The split modules are independently importable + the public surface is stable
# ---------------------------------------------------------------------------

def test_new_sibling_modules_are_importable():
    from briefs import kw_brief_fa  # noqa: F401
    from briefs import kw_brief_shared  # noqa: F401
    from briefs import kw_brief_pa3_phases  # noqa: F401
    from briefs import kw_brief_port_a3  # noqa: F401


def test_public_surface_reexports_stable():
    """External callers (BackwardPlugin, golden test, cube-mix test) import these
    from `briefs.kw_brief`; the decomposition must keep them re-exported there.
    """
    from briefs import kw_brief as k

    for sym in (
        # FA cluster (BackwardPlugin imports the first four)
        "_is_fa_class_backward",
        "_fused_fa_backward_requested",
        "_fa_class_backward_stitch_block",
        "_fa_class_backward_multilaunch_block",
        "_fa_class_template_assembly_block",
        "_fa_assembly_intro_block",
        "_fa_assembly_recipe_block",
        "_fa_assembly_compile_block",
        "_fa_assembly_verify_hard_block",
        "_fa_ge_host_gen_block",
        # shared leaf
        "_forced_architecture_block",
        # port_a3 orchestrator + cube-mix (imported by tests)
        "_port_a3_phase_instructions_block",
        "_port_a3_cube_class_mix_block",
        # parent-retained
        "build_worker_brief",
        "_phase_instructions_block",
        "_exit_handoff_block",
        "_backward_perf_c2_block",
    ):
        assert hasattr(k, sym), f"briefs.kw_brief lost re-export: {sym}"


def test_no_import_cycle_shared_leaf_is_leaf():
    """Leaf modules must NOT import the parent / orchestrator (acyclic invariant).

    Checks actual import statements via AST, not substring (docstrings mention
    the module names in prose).
    """
    import ast

    def _imported_modules(fname):
        tree = ast.parse((_BRIEFS / fname).read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
            elif isinstance(n, ast.Import):
                mods.update(a.name for a in n.names)
        return mods

    shared_imports = _imported_modules("kw_brief_shared.py")
    assert "briefs.kw_brief" not in shared_imports
    assert "briefs.kw_brief_port_a3" not in shared_imports

    # phases module is a leaf too (orchestrator imports it one-way, not vice-versa)
    phases_imports = _imported_modules("kw_brief_pa3_phases.py")
    assert "briefs.kw_brief" not in phases_imports
    assert "briefs.kw_brief_port_a3" not in phases_imports


# ---------------------------------------------------------------------------
# 2. FA-cluster builder goldens (sha256 byte-locks; captured post-split == pre-split)
# ---------------------------------------------------------------------------

_FA_NOARG_GOLDENS = {
    # Deliberate packaging wording update: the source is the current customer
    # entry request, not the removed legacy /ascendc-op-gen command.
    "_fa_assembly_recipe_block": "b78ad2630776f9eaa4c379b3d500291f154c17f677163b1e69ee4f067b34406c",
    # MIX cube+vec silent-hang warning for the FORWARD FA-assembly brief
    # (recall-fix 2026-07-16). Byte-locked like the other FA leaf-builders so the
    # warning content can't silently drift.
    # Hash updated DELIBERATELY 2026-07-16 (DEBT-210 a/b/c): the block used to tell
    # every FA worker the ACLRT stub "CANNOT supply" the FFTS descriptor and that the
    # route was closed to them. That was false — it is a three-line host<->kernel idiom
    # and catlass standalone 23/49 supplies it without GE. The block now splits the two
    # orthogonal 507014 failure modes (DEBT-210 sync-base-not-emitted, SoC-independent /
    # PB-34 KFC slot contention, V220-only). Content change is the point of the change.
    #
    # Hash updated DELIBERATELY AGAIN 2026-07-17 (DEBT-210 overcorrection fix): the
    # retraction above told workers the three lines were theirs to write, which
    # overshot. They do not compile from our host TU — the CANN runtime include is on
    # the DEVICE target and CMake PRIVATE does not propagate — so a worker following
    # the block hit a missing-header error and would reasonably conclude the KB lies.
    # The block now states the compile gap, marks the error EXPECTED, and routes to
    # escalation (DEBT-210(d′)) instead of a hand-patched build. Content change is
    # again the point; re-pinned rather than regenerated silently.
    #
    # Hash updated DELIBERATELY a THIRD time 2026-07-17 (causal retraction + PB-35):
    # two facts landed. (1) The sync-base causal arrow is REFUTED on V220, not merely
    # unproven — a single-variable flip on a known-good catlass MIX op (instrument-
    # checked: 9 real cross-core <0x2> flags, SetSyncBaseAddr genuinely wired) still
    # passed, so the block no longer says an unset sync base hangs; the confirmed
    # cause of our 507014 is PB-34, evidenced by 3_FusionAttention's own source.
    # (2) PB-35 (op_class=mixed_aic_aiv_pattern_a_tile_mmad, confirmed_on
    # Ascend950PR_9579/A5) attacks Pattern A itself, so "escape via Pattern A" was
    # incomplete guidance on A5; the block now points at PB-35 + cross_core_sync.md
    # §4's runnable handshake rather than copying the recipe. Content change is the
    # point; re-pinned deliberately, not regenerated.
    #
    # Hash updated DELIBERATELY a FOURTH time 2026-07-17 (DEBT-208 — SoC scoping):
    # the block is no longer one unconditional string. Each KB card it carries is now
    # emitted iff that card's OWN `applies_to: soc=` covers the target (read via
    # briefs.kb_scope, which reuses kb_index_audit's SoC parser), so this builder is
    # target-dependent. The no-arg golden below therefore pins the `target="a5"`
    # composition — the A5 worker's brief, which is the one the defect corrupted:
    # PB-34 (`soc=Ascend910_9382`, V220, with two Ascend950PR no-reproduce witnesses)
    # is now ABSENT on A5 and its slot is taken by the two mutually-exclusive proven
    # A5 routes (Path B = MatmulImpl light-port + KFC-implicit sync, 122/122 witness;
    # Path A = non-KFC catlass block cube + cross_core_sync.md §4's handshake). PB-35
    # (`applies_to` names BOTH SoCs, `confirmed_on` A5) still reaches A5 — suppressing
    # it was the trap. The V220 composition is byte-locked separately by
    # test_kw_brief_soc_scope_debt208.py, which also mutation-proves the predicate.
    # Content change is the point; re-pinned deliberately, not regenerated silently.
    "_fa_assembly_deadlock_warning_block": "225851330cff475eb30134893486ec14f9018305a1003097ec64c1c8a17a90bd",
    "_fa_assembly_compile_block": "0cdf20aa8b996c46fb2e0b9035db439e434aa984a0ec69bfbc4abc7d85c6b6d5",
    "_fa_assembly_verify_hard_block": "d9fe58d2c7e9af3d6a302a331527f4db5faafd7ad5dccf609ef01433d457b1a5",
    # cannbot re-pin (v3.13.0 re-sync): builder output == v3.13.0's with only
    # src/skills/references→kb/ relocation applied (proven reloc-equivalent, no other drift).
    "_fa_ge_host_gen_block": "ec5e47f327960386cd95c183dfd880f77998037f160542f6c21e9f53e4ab62c6",
}


@pytest.mark.parametrize("fn,sha", sorted(_FA_NOARG_GOLDENS.items()))
def test_fa_noarg_builders_byte_identical(fn, sha):
    from briefs import kw_brief_fa as fa

    assert _sha(getattr(fa, fn)()) == sha, f"{fn} emitted string drifted"


def test_fa_assembly_intro_byte_identical():
    from briefs import kw_brief_fa as fa

    got = _sha(getattr(fa, '_fa_assembly_intro_block')("flash_attention_score", "FA_CLASS ATTENTION"))
    assert got == "2649ca3fc8c4ef35cab3e5b41c971df76502ed14ab08789ae45fa23dcce875c8"


def test_fa_predicates():
    from briefs import kw_brief_fa as fa

    bw_ws = Path(tempfile.mkdtemp()) / "ws"
    bw_ws.mkdir(parents=True)
    (bw_ws / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "backward"}))
    assert getattr(fa, '_is_fa_class_backward')("flash_attention_score_grad", "FA_CLASS BACKWARD", bw_ws) is True
    assert getattr(fa, '_is_fa_class_backward')("abs", "ELEMENTWISE", _ws(["ELEMENTWISE"])) is False

    fused_ws = Path(tempfile.mkdtemp()) / "ws"
    fused_ws.mkdir(parents=True)
    (fused_ws / ".opgen_state.json").write_text(json.dumps({"fa_backward_arch": "fused"}))
    assert getattr(fa, '_fused_fa_backward_requested')("x", fused_ws) is True
    assert getattr(fa, '_fused_fa_backward_requested')("x", None) is False


# ---------------------------------------------------------------------------
# 3. Shared leaf: forced-architecture block
# ---------------------------------------------------------------------------

def test_forced_architecture_block():
    from briefs import kw_brief_shared as sh

    # forced marker = boolean key `force_simt` (or a bare SIMT/SIMD op_class_tag)
    forced_ws = Path(tempfile.mkdtemp()) / "ws"
    forced_ws.mkdir(parents=True)
    (forced_ws / "op_classification.json").write_text(
        json.dumps({"force_simt": True, "op_class_tags": ["a3_to_a5_port"]})
    )
    forced = getattr(sh, '_forced_architecture_block')(forced_ws)
    assert "ARCHITECTURE IS FIXED" in forced and "SIMT" in forced
    assert getattr(sh, '_forced_architecture_block')(_ws(["ELEMENTWISE"])) == ""


# ---------------------------------------------------------------------------
# 4. port_a3 phase-body builder goldens (previously only transitively covered)
# ---------------------------------------------------------------------------

_PA3_PHASE_GOLDENS = {
    "_pa3_context": "0a6601b70b0eec5516b2d2f5ff42765e7563f4409b5beeec924f8ff6673687ed",
    "_pa3_phase_a_1": "b8c6143f46af2094177df455519f4799d1dea08dd589c7dce3e2b163f042dd37",
    "_pa3_phase_a_2": "7de3e2d152eceb32955486c5e802e9d265a97b5b4afe9a6a606271b7354965e3",
    "_pa3_phase_a_3": "c5903dd6e58f8bac3f6dcee3c9dd0c2db3fcb4febdfd702a87992e68ac1f0d27",
    "_pa3_phase_b": "4d7022f4f05432c2db701d1a2eb62b3c7cc07ea010c2939f6c421313b6b3d2c5",
    "_pa3_phase_c": "40b46bd439f246e52b581c251ee55b3ec113c8a3d9e6a1db669a16d2196a7d25",
}
_PA3_ORCH_GOLDENS = {
    "_pa3_phase_d_1": "423c7bc19c126855796eb292265f164153c844bc572896fc27aed59da7e2c64e",
    "_pa3_phase_d_2": "036d6924809d64d979c72dc75e7d1678bc0477c6cf7d9d9d6682a238b0b7a719",
    "_pa3_phase_d_3": "68b823aba42c55f0adbbe2a63f4212a54e1b7858fca5536cf00dcbb72093e0a1",
    "_pa3_phase_e": "a92f72c6c2fb8bedafc18c7edd2685d57ca441f1b2b5be8b3db54291790638aa",
    "_pa3_iter_budget": "5625d5ab4c746624153b9387426ac53bb83c30beff4bf9b50aa626440c85fcb8",
}


def _pa3_kw():
    return dict(
        op="mat_mul_v3",
        workspace=_ws(["a3_to_a5_port", "CUBE_MIX"]),
        iter_cap_remaining=3,
        port_source="/src",
        aclnn_entry="aclnnFoo",
        gen_data_source="gen.py",
        peer_deps_line="(none)",
        env=_StubEnv(),
    )


@pytest.mark.parametrize("fn,sha", sorted(_PA3_PHASE_GOLDENS.items()))
def test_pa3_phase_builders_byte_identical(fn, sha):
    from briefs import kw_brief_pa3_phases as ph

    assert _sha(getattr(ph, fn)(**_pa3_kw())) == sha, f"{fn} emitted string drifted"


@pytest.mark.parametrize("fn,sha", sorted(_PA3_ORCH_GOLDENS.items()))
def test_pa3_orch_phase_builders_byte_identical(fn, sha):
    from briefs import kw_brief_port_a3 as po

    assert _sha(getattr(po, fn)(**_pa3_kw())) == sha, f"{fn} emitted string drifted"


def test_pa3_helper_blocks_byte_identical():
    from briefs import kw_brief_pa3_phases as ph

    ws = _ws(["a3_to_a5_port", "CUBE_MIX"])
    assert _sha(getattr(ph, '_migration_level_block')("mat_mul_v3", ws)) == \
        "c1e7ef6163fabf4007d204efed373769b616d8d3d1032cda55a9986a3f46a712"
    assert _sha(getattr(ph, '_port_a3_cube_class_mix_block')(ws)) == \
        "f2f5598fef7c87eabb038f142e5f9a700fe386fffdf1e7bb9aa0c73513b0142e"
    assert _sha(getattr(ph, '_port_a3_complete_deliverable_block')()) == \
        "5ee9954105c35627358a4a5a9b9a9113c19a1f67c3b7a95cc6bba50ddccf8337"
