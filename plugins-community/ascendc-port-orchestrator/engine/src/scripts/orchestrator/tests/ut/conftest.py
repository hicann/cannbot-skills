# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""UT-scoped hermetic guards.

The pre-commit gate runs ONLY this ``ut/`` suite and MUST be hermetic — no unit test may
depend on a live ``claude`` subprocess. ``phase_o17_classify`` binds its Backend at import
(``_backend = get_backend()`` → the ``claude_code`` backend) and ``_invoke_claude_skill``
shells out to it. Any ut test that drives ``run_single_op`` through the O17 classify phase
(e.g. the FSM-characterization suite, which already mocks ``spawn_for_state`` + ``fire_critic``
but leaves O17 as an unmocked seam) therefore spawns a real ``claude --print`` child and HANGS
off a provisioned box (headless CI / cannbot-dev / clean checkout). This is the SAME class of
headless-absent env dependency the parent ``tests/conftest.py`` already neutralizes for
``.ascendc_env`` via an autouse hermetic fixture.

This autouse fixture makes the O17 classify skill hermetic in ut: it returns the
"skill unavailable" result (``ok=False``) — exactly what ``_invoke_claude_skill`` yields on a
box without the claude CLI (its ``not_found`` branch) — so O17 deterministically takes its
documented fallback path (cached/intrinsic taxonomy in ``_try_existing_classification`` or an
error classification) instead of blocking on a live subprocess. It is scoped to ``ut/`` only;
backend-requiring integration tests live in ``it/`` where a real backend is expected. Tests that
want a specific O17 classification still monkeypatch ``_invoke_claude_skill`` themselves — the
per-test patch is applied after this fixture and wins.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_LOG = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _hermetic_o17_backend(monkeypatch):
    """Neutralize the O17 classify skill's live-backend dispatch for every ut test.

    Mirrors ``tests/conftest.py::_hermetic_ascendc_env`` (neutralize a headless-absent env
    dependency in setup only; tested code unchanged). Returns the honest claude-absent result
    so O17 exercises its real fallback path — this is behavior-preserving for any test that
    does not itself assert on a live O17 classification."""
    orch_dir = str(Path(__file__).resolve().parents[2])  # …/orchestrator
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    try:
        import phase_o17_classify as o17
    except ImportError:
        # A ut test that never imports the orchestrator package is unaffected.
        return
    monkeypatch.setattr(
        o17,
        "_invoke_claude_skill",
        lambda workspace, timeout=300: (
            False,
            "",
            "hermetic ut: O17 classify skill disabled (no live claude in the ut gate)",
        ),
        raising=True,
    )


@pytest.fixture(autouse=True)
def _hermetic_harness_pristine(monkeypatch):
    """DEBT-213(b): pin the harness-pristine probe to CLEAN for every ut test.

    Same hermeticity class as ``_hermetic_o17_backend`` above. The probe shells
    out to ``git status`` over the REAL repo, so O5's verdict would otherwise
    depend on whether the developer running pytest happens to have uncommitted
    harness edits — the ut gate would go red for everyone who is mid-change on
    the orchestrator, which is precisely the "check the team learns to ignore"
    outcome DEBT-213 is trying to avoid. Pinning to CLEAN keeps every existing
    O5 test asserting what it means to assert (claim vs re-measurement).

    The probe itself is NOT left untested: ``test_debt_213_harness_pristine.py``
    exercises the real implementation against real throwaway git repos, and
    re-patches ``harness_state`` per-test for the verdict-downgrade cases (the
    per-test patch is applied after this fixture and wins).
    """
    orch_dir = str(Path(__file__).resolve().parents[2])  # …/orchestrator
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    try:
        import harness_pristine as hp
    except ImportError:
        return
    monkeypatch.setattr(
        hp,
        "harness_state",
        lambda *a, **k: hp.HarnessState(hp.CLEAN, reason="hermetic ut: probe pinned CLEAN"),
        raising=True,
    )


# ---------------------------------------------------------------------------
# External cannbot-knowledge repo (OKF v0.2) hermetic fixtures
# ---------------------------------------------------------------------------
# The 672 OKF v1 cards moved out of the plugin (commit 223ee980) into the
# external cannbot-knowledge repo, configured via CANNBOT_KNOWLEDGE_ROOT and
# adapted in `briefs/external_kb.py`. Scope predicates (briefs/kb_scope.py) and
# brief display paths (`kb_ref_display`) now read through that repo, so tests
# of SoC-scoped composition need it twice over:
#   - `external_kb_fixture` builds a minimal OKF v0.2 repo in tmp_path with the
#     cards the composers gate on (entry aliases/platforms, the CCS §4 section
#     scope, the pattern templates' header applies_to) at their POST-migration
#     locations, and points CANNBOT_KNOWLEDGE_ROOT at it;
#   - `no_external_kb` explicitly isolates a test from any ambient external
#     config (env var or project `.cannbot/knowledge.env`), pinning the loud
#     missing-install marker used when the dependency is unconfigured.
# `briefs.external_kb._ROOT_CACHE` and `briefs.kb_scope._EXTERNAL_SCOPES_CACHE`
# are process-global, so an autouse reset keeps the two modes order-independent.


def _reset_external_kb_caches():
    try:
        from briefs.external_kb import _ROOT_CACHE

        _ROOT_CACHE.clear()
    except Exception as exc:
        _LOG.debug("external_kb root cache reset skipped: %s", exc)
    try:
        from briefs import kb_scope as _ks

        _ks.reset_external_scopes_cache()
    except Exception as exc:
        _LOG.debug("kb_scope entry-scope cache reset skipped: %s", exc)


@pytest.fixture(autouse=True)
def _external_kb_cache_isolation():
    """Reset the external-KB process caches around every ut test."""
    _reset_external_kb_caches()
    yield
    _reset_external_kb_caches()


@pytest.fixture
def no_external_kb(monkeypatch, tmp_path):
    """Explicitly unconfigure the external knowledge repo for this test."""
    monkeypatch.delenv("CANNBOT_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setenv("CANNBOT_PROJECT_ROOT", str(tmp_path))
    _reset_external_kb_caches()
    yield
    _reset_external_kb_caches()


def _v02_card(*, entry_id=None, applies_to=None, body=""):
    """Minimal OKF v0.2 card. `_external_entry_scopes` reads the legacy entry id
    from the body provenance line (「原 OKF v1 卡号：**ID**」) and the scope from
    the frontmatter `description:`-inline `applies_to: soc=` (same shape as the
    real migrated cards)."""
    lines = ["---", 'okf_version: "0.2"']
    if applies_to:
        lines.append('description: "applies_to: %s"' % applies_to)
    lines.append("---")
    lines.append("")
    if entry_id:
        lines.append("> 原 OKF v1 卡号：**%s**。" % entry_id)
        lines.append("")
    lines.append(body or "# fixture card\n")
    return "\n".join(lines) + "\n"


# The migrated cards the brief composers actually gate on, at their OKF v0.2
# (external-repo) locations. Scopes mirror the real cards: PB-34/OL-275/PB-55
# are V220-only, OL-220/OL-223/PB-45/EC-68 are V351-only, PB-35 names both.
_EXTERNAL_FIXTURE_CARDS = {
    "ops/ascendc/runbooks/compilation/pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md":
        _v02_card(entry_id="PB-34", applies_to="soc=Ascend910_9382 (V220 A2/A3 single-die)"),
    "ops/ascendc/runbooks/compilation/pb_35_event_t_0_for_cube_internal_pipe_sync_mte1_m_m_fix.md":
        _v02_card(entry_id="PB-35", applies_to="soc=Ascend910_9382,Ascend950PR_9579"),
    "ops/ascendc/runbooks/compilation/pb_55_mix_reverse_handshake_counted.md":
        _v02_card(entry_id="PB-55", applies_to="soc=Ascend910_9382"),
    "ops/ascendc/runbooks/compilation/ec_68_507015_setsysworkspaceforce_on_aclrt_launch_mix.md":
        _v02_card(entry_id="EC-68", applies_to="soc=Ascend950PR"),
    "ops/ascendc/optimizations/pb_45_tpipe_reset_frees_global_event_pool.md":
        _v02_card(entry_id="PB-45", applies_to="soc=Ascend950PR"),
    "ops/ascendc/optimizations/ol_220_cube_vec_mix_ascendc_library_build_recipe.md":
        _v02_card(entry_id="OL-220", applies_to="soc=Ascend950PR"),
    "ops/ascendc/optimizations/ol_223_reset_safe_cube_internal_l0_fences.md":
        _v02_card(entry_id="OL-223", applies_to="soc=Ascend950PR"),
    "ops/ascendc/optimizations/ol_275_managed_cube_kfc_lifecycle.md":
        _v02_card(entry_id="OL-275", applies_to="soc=Ascend910_V220"),
    # §4 section-scope anchor for `kb_section_soc_families(..., "4")`: the
    # applies_to line must be a PLAIN line (the section scanner does not see
    # through a blockquote `>`).
    "ops/ascendc/optimizations/fa_cross_core_sync_workspacequeue.md": (
        "# FA cross-core sync WorkspaceQueue\n"
        "\n"
        "### 4. RUNNABLE deadlock-avoiding handshake (PUBLIC-API)\n"
        "applies_to: soc=Ascend950PR (V351 / A5, Ascend950PR_9579)\n"
        "\n"
        "SYNC MODE 4 body.\n"
        "\n"
        "### 5. Other\n"
    ),
    # Pattern templates gated via `kb_file_applies_to_target` (header zone,
    # blockquote-tolerant).
    "ops/ascendc/examples/fa_class_a3_mix_template.md": (
        "# P-P116 a3 FA-class MIX skeleton\n"
        "> applies_to: soc=Ascend910_9382 (V220 A2/A3 single-die); cann=9.0.0+\n"
    ),
    "ops/ascendc/examples/gmm_swiglu_quant_a8w8_class_template.md": (
        "# GMM SwiGLU quant A8W8 template\n"
        "> applies_to: soc=Ascend950PR/V351 (arch35 only)\n"
    ),
    "ops/ascendc/examples/hkv_patterns.md": (
        "# HKV patterns\n"
        "\n"
        "No applies_to header — neutral template.\n"
    ),
    # fa_class_template.md: 12 frontmatter lines push `applies_to` past a naive
    # lines[:20] window — the DEBT window regression fixture (see
    # test_kb_scope.py::test_applies_to_survives_frontmatter_that_pushes_it_past_the_window).
    "ops/ascendc/examples/fa_class_template.md": (
        "---\n"
        + "".join("field_%d: v\n" % i for i in range(10))
        + 'okf_version: "0.2"\n'
        + "---\n"
        + "".join("filler body line %d\n" % i for i in range(14))
        + "> applies_to: soc=Ascend950PR (V351/A5)\n"
    ),
    "ops/ascendc/examples/cube_vector_fusion.md": "# P-P102 cube_vector_fusion\n",
    # migration_level playbook guides (referenced by `_migration_level_block`).
    "ops/ascendc/guides/cross_gen_migration_guide/l1_implementation.md": "# L1\n",
    "ops/ascendc/guides/cross_gen_migration_guide/l2_register_based.md": "# L2\n",
    "ops/ascendc/guides/cross_gen_migration_guide/l1_l2_implementation.md": "# L1+L2\n",
    "ops/ascendc/guides/cross_gen_migration_guide/l3_simt_optimization.md": "# L3\n",
    "ops/ascendc/guides/cross_gen_migration_guide/l4_simt_optimization.md": "# L4\n",
    "ops/ascendc/guides/cross_gen_migration_guide/l5_register_based.md": "# L5\n",
}


@pytest.fixture
def external_kb_fixture(monkeypatch, tmp_path):
    """A minimal external cannbot-knowledge repo (OKF v0.2) wired via env.

    Layout gates `external_kb.external_kb_root()`'s validation:
    Bundle index + Registry + the knowledge-query script placeholder. Returns the
    fixture ROOT (the dir containing `knowledge/`).
    """
    root = tmp_path / "cannbot-knowledge"
    (root / "knowledge").mkdir(parents=True)
    (root / "knowledge" / "index.md").write_text(
        '---\nokf_version: "0.2"\n---\n# fixture knowledge bundle\n'
    )
    registry = root / "governance" / "schemas" / "registries.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("domains: {}\n")
    kq = root / ".agents" / "skills" / "knowledge-query" / "scripts" / "knowledge_query.py"
    kq.parent.mkdir(parents=True)
    kq.write_text("# fixture placeholder\n")
    for rel, content in _EXTERNAL_FIXTURE_CARDS.items():
        p = root / "knowledge" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    monkeypatch.setenv("CANNBOT_KNOWLEDGE_ROOT", str(root))

    def _fixture_entry_search(query, target="a5", per_platform_k=10, card_type=None):
        """Simulate knowledge-query retrieval for entry-id lookups only.

        The fixture bundle has no sqlite index, and production
        `search_external_cards` fail-closes to [] without one — which left
        entry-scope resolution returning None for every fixture card. Entry
        scope (`kb_scope._external_entry_scope`) keys on legacy ids
        (``PB-34`` / ``OL-275`` / ...), so resolve those by scanning the
        fixture cards directly and let the real frontmatter parser
        (`_ids_and_scope_from_card`) do the rest. General-text queries keep
        returning [] so unconfigured/无命中 brief markers still render.
        """
        import re as _re

        q = str(query or "").strip()
        if not _re.fullmatch(r"[A-Z]{2}-\d+", q):
            return []
        hits = []
        for path in sorted((root / "knowledge").rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if q in text:
                hits.append({
                    "local_path": str(path),
                    "path": str(path.relative_to(root / "knowledge")),
                    "title": path.stem,
                    "type": "runbook",
                    "status": "stable",
                    "score": 1.0,
                })
        return hits[: max(int(per_platform_k or 10), 10)]

    # Patch through the SAME import statement the code under test uses. A
    # string-targeted monkeypatch resolves the module via pytest's
    # derive_importpath, which (with the per-test sys.modules snapshot/restore
    # in tests/conftest.py) can land on a different module instance than the
    # one `from briefs.external_kb import ...` sees inside kb_scope — the
    # second fixture test then ran the REAL search (order-dependent failures).
    from briefs import external_kb as _ekb

    monkeypatch.setattr(_ekb, "search_external_cards", _fixture_entry_search)
    _reset_external_kb_caches()
    yield root
    _reset_external_kb_caches()
