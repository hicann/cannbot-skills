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
    #
    # Hash updated deliberately after the external-knowledge migration. The configured
    # composition is asserted separately because it embeds the fixture's tmpdir.
    # Updated again after removing the plugin-local fallback:
    # unconfigured mode no longer falls back to plugin-local cards, and the marker
    # text changed to 「cannbot-knowledge 未安装、配置无效或未收录」. Re-pinned after
    # diffing bb0bec32-vs-HEAD generated blocks: 4 changed lines, all carrying the
    # ⚠️ marker; non-marker lines 0.
    "_fa_assembly_deadlock_warning_block": "f5600c3f7a0c4d19f83d51aef95339585af98e7c0ea3c7fd7b8b4ee54e9e8692",
    "_fa_assembly_compile_block": "0cdf20aa8b996c46fb2e0b9035db439e434aa984a0ec69bfbc4abc7d85c6b6d5",
    "_fa_assembly_verify_hard_block": "d9fe58d2c7e9af3d6a302a331527f4db5faafd7ad5dccf609ef01433d457b1a5",
    # cannbot re-pin (v3.13.0 re-sync): builder output == v3.13.0's with only
    # src/skills/references→kb/ relocation applied (proven reloc-equivalent, no other drift).
    # Re-pinned 2026-08-31 (OKF-only 迁移): recipe pointer moved to plugin
    # `templates/fa_class/op_host/` (was `kb/target/ascendc/patterns/domains/fa_class/templates/op_host/`).
    "_fa_ge_host_gen_block": "69e763c303208bd09b64afde1124a8da83d3959e81a97e03f6e876b293008898",
}


@pytest.mark.parametrize("fn,sha", sorted(_FA_NOARG_GOLDENS.items()))
def test_fa_noarg_builders_byte_identical(fn, sha, no_external_kb):
    """Byte-lock the FA leaf builders with the external KB UNCONFIGURED.

    The deterministic mode — see the `_fa_assembly_deadlock_warning_block`
    re-pin note). Configured-mode behaviour is covered by
    `test_fa_builders_external_repo_mode`.
    """
    from briefs import kw_brief_fa as fa

    assert _sha(getattr(fa, fn)()) == sha, f"{fn} emitted string drifted"


def test_fa_builders_external_repo_mode(external_kb_fixture):
    """配置外部知识仓后：迁移卡的引用解析为外部仓绝对路径。

    且 scope 谓词恢复（a5 组合里 PB-34 缺席、§4 卡以 snake 名绝对路径出现）。
    """
    from briefs import kw_brief_fa as fa
    from briefs.kw_brief_fa import (
        _fa_assembly_deadlock_warning_block,
        _fa_assembly_intro_block,
    )

    ext = str(external_kb_fixture)
    intro = _fa_assembly_intro_block("flash_attention_score", "FA_CLASS ATTENTION")
    assert ext + "/knowledge/ops/ascendc/examples/fa_class_template.md" in intro

    block = _fa_assembly_deadlock_warning_block("a5")
    assert "### PB-34 — MIX cube+vec SILENT-HANG" not in block        # scope restored
    assert ext + "/knowledge/ops/ascendc/optimizations/fa_cross_core_sync_workspacequeue.md" in block
    assert "⚠️ 未找到" not in block                                    # every reference resolved


def test_fa_assembly_intro_byte_identical(no_external_kb):
    from briefs import kw_brief_fa as fa

    got = _sha(getattr(fa, '_fa_assembly_intro_block')("flash_attention_score", "FA_CLASS ATTENTION"))
    # Re-pinned 2026-09-17 (去双轨模式, bb167473): marker text changed to
    # 「cannbot-knowledge 未安装、配置无效或未收录」; bb0bec32-vs-HEAD diff = 1 line
    # (the P-P103 pointer's ⚠️ marker), non-marker lines 0.
    # Re-pinned 2026-09-17 (外部知识仓适配): the P-P103 pointer now goes through
    # `external_kb.kb_ref_display`; pinned here with the external repo
    # UNCONFIGURED (deterministic ⚠️-marker form). Configured-mode assertion in
    # `test_fa_builders_external_repo_mode`.
    assert got == "4a8e4feb99f4eb480ce4a28c4f78a05f0d2b075f222d94724f462f336a6f2ab5"


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
    # Re-pinned 2026-09-17 (去双轨模式, bb167473): `_pa3_context` 内嵌的
    # `_migration_level_block`/`_port_a3_cube_class_mix_block` 的 ⚠️ 标记文案改为
    # 「cannbot-knowledge 未安装、配置无效或未收录」。
    # Re-pinned 2026-09-17 (外部知识仓适配): `_pa3_context` 内嵌的
    # `_migration_level_block`/`_port_a3_cube_class_mix_block` 经 `kb_ref_display`
    # 输出；本 pin 为未配置外部仓的确定性形态（⚠️ 未找到 标记）。配置模式见
    # `test_pa3_blocks_external_repo_mode`。
    # `_pa3_phase_a_2`'s previous pin was already stale before this migration
    # (baseline drift) — re-pinned to current content, no text change from OKF.
    # Re-pinned 2026-09-02: `_pa3_phase_a_2` pin stale again on the 1358ec68
    # baseline (baseline drift — baseline and HEAD emit byte-identical output,
    # verified by diff); re-pinned to current content.
    "_pa3_context": "4537c979a98825f20ee4401fa2c25f3cf596dfedad287c42453cf045ec0ad97f",
    "_pa3_phase_a_1": "7a882d3c4b5b78080aa59f4f6e7193cbf8ee87f81984e8f7146d21e51c510e65",
    "_pa3_phase_a_2": "b039e18ed322c08515f682efc96a476d5e2f3df62d1dd28715d4badb8740fddd",
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
def test_pa3_phase_builders_byte_identical(fn, sha, no_external_kb):
    from briefs import kw_brief_pa3_phases as ph

    assert _sha(getattr(ph, fn)(**_pa3_kw())) == sha, f"{fn} emitted string drifted"


@pytest.mark.parametrize("fn,sha", sorted(_PA3_ORCH_GOLDENS.items()))
def test_pa3_orch_phase_builders_byte_identical(fn, sha, no_external_kb):
    from briefs import kw_brief_port_a3 as po

    assert _sha(getattr(po, fn)(**_pa3_kw())) == sha, f"{fn} emitted string drifted"


def test_pa3_helper_blocks_byte_identical(no_external_kb):
    from briefs import kw_brief_pa3_phases as ph

    # Re-pinned 2026-09-17 (去双轨模式, bb167473): 本函数下 3 个断言中的 2 个重钉——
    # `_migration_level_block` 与 `_port_a3_cube_class_mix_block` 的标记文案变化。
    # `_port_a3_complete_deliverable_block` 未变（不含 KB 路径），hash 保持。
    # bb0bec32-vs-HEAD 两棵树生成同一字符串做 diff 验证：差异全部落在 ⚠️ 标记行。
    # Re-pinned 2026-09-17 (外部知识仓适配): `_migration_level_block` 的
    # guides/subdirs 与 `_port_a3_cube_class_mix_block` 的 cube_vector_fusion
    # 指引改经 `kb_ref_display` 输出；本 pin 为未配置外部仓的确定性形态
    # （⚠️ 未找到 标记）。
    ws = _ws(["a3_to_a5_port", "CUBE_MIX"])
    assert _sha(getattr(ph, '_migration_level_block')("mat_mul_v3", ws)) == \
        "2efd53f64ff389235df515706da450d9615bc081deca865d804fe9e44101ea6e"
    assert _sha(getattr(ph, '_port_a3_cube_class_mix_block')(ws)) == \
        "6e49e7136c256935a143dd1c3fcdf1ad728f3fbe099200e85893c9507103bcbf"
    assert _sha(getattr(ph, '_port_a3_complete_deliverable_block')()) == \
        "caa6ba76a1250f6e8365576e9181da972dd5b962871de5617070229509ff2425"


def test_pa3_blocks_external_repo_mode(external_kb_fixture):
    """配置外部知识仓后，playbook guides 与模板解析为外部仓绝对路径。"""
    from briefs import kw_brief_pa3_phases as ph
    from briefs.kw_brief_pa3_phases import (
        _migration_level_block,
        _port_a3_cube_class_mix_block,
    )

    ext = str(external_kb_fixture)
    ws = _ws(["a3_to_a5_port", "CUBE_MIX"])
    ml = _migration_level_block("mat_mul_v3", ws)
    assert ext + "/knowledge/ops/ascendc/guides/cross_gen_migration_guide/l1_implementation.md" in ml
    assert "asc-devkit-vendored" not in ml
    mix = _port_a3_cube_class_mix_block(ws)
    assert ext + "/knowledge/ops/ascendc/examples/cube_vector_fusion.md" in mix
