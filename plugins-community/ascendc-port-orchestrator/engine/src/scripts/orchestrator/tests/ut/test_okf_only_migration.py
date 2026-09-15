# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""OKF-only migration (2026-08-31) regression pins.

The legacy bundled KB layout (`kb/target/` + `kb/KB_INDEX.md`) was physically
deleted; the bundled b-tier is OKF cards under `kb/okf/**` and nothing else.
These tests pin the migration's observable contract:

  a. OKF-only startup — all seven brief builders (kw/pp/ko/ar/fo/da/cl) render
     with no legacy tree on disk and raise no file-not-found.
  b. No brief text references the removed layout (`kb/target`, `KB_INDEX`,
     `target/ascendc`).
  c. OKF retrieval failure is FAIL-LOUD — an empty/failed retrieval produces a
     loud warning marker in the brief, never a silent legacy fallback.
  d. The user-local c-tier (write / read / tombstone) is unaffected.
  e. Template/example asset paths exist and are non-empty.

The workspace fixture seeds an explicit `a3_live` reference binding so these
tests do NOT depend on the `ReferenceSourceError` baseline gap that currently
keeps test_p0aay_brief_smoke.py red (its fixtures predate the mandatory
durable reference binding).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # ../../ → orchestrator/
_SCRIPTS = _HERE.parent.parent.parent         # engine/src/scripts/
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from kb_paths import kb_root, plugin_root  # noqa: E402
from reference_source import explicit_a3_live_binding  # noqa: E402
from briefs import _common as bc  # noqa: E402
from briefs import brief_kb  # noqa: E402
from briefs.kw_brief import build_worker_brief  # noqa: E402
from briefs.pp_brief import build_probe_brief  # noqa: E402
from briefs.ko_brief import build_optimizer_brief  # noqa: E402
from briefs.ar_brief import build_researcher_brief  # noqa: E402
from briefs.fo_brief import build_fused_optimizer_brief  # noqa: E402
from briefs.da_brief import build_det_analyzer_brief  # noqa: E402
from briefs.cl_brief import build_cann_learner_brief  # noqa: E402

_BUILDERS = {
    "kw": build_worker_brief,
    "pp": build_probe_brief,
    "ko": build_optimizer_brief,
    "ar": build_researcher_brief,
    "fo": build_fused_optimizer_brief,
    "da": build_det_analyzer_brief,
    "cl": build_cann_learner_brief,
}

# Tokens that must never appear in a rendered brief after the OKF-only
# migration — each names a piece of the deleted legacy layout.
_LEGACY_LAYOUT_TOKENS = ("kb/target", "KB_INDEX", "target/ascendc")

# The loud marker `kb_manifest_block` emits when OKF retrieval returns empty
# (index not built / no hits / retrieval error) — the fail-loud contract.
_FAIL_LOUD_MARKER = "⚠️ OKF 检索无返回"


def _seed_env(tmp_path: Path) -> bc.AscendCEnv:
    env_path = tmp_path / ".ascendc_env"
    env_path.write_text(textwrap.dedent("""\
        A5_HOST=test
        A5_USER=root
        A5_PASSWORD='x'
        A5_CONTAINER=test
        CANN_PATH=/data/cann
        SOC_VERSION=Ascend950PR_9579
        BENCHMARK_ROOT=/root/bench
        LOCAL_BENCHMARK=/local/bench
        LOCAL_PROJECT=/proj
        TARGET=a5
        OPGEN_MODE=port_a3_to_a5
        BENCHMARK_BRANCH=main
    """))
    return bc.load_env(env_path)


def _seed_workspace(tmp_path: Path, op: str) -> Path:
    """Workspace with an explicit reference binding (no ReferenceSourceError)."""
    workspace = tmp_path / "workspace" / op
    workspace.mkdir(parents=True)
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": op,
        "opgen_mode": "port_a3_to_a5",
        "reference": explicit_a3_live_binding(),
    }))
    return workspace


# ---------------------------------------------------------------------------
# Layout pin: the legacy tree is physically gone, the OKF tree is real.
# ---------------------------------------------------------------------------


def test_legacy_kb_layout_is_physically_removed():
    root = kb_root()
    assert root.is_dir(), f"kb root missing: {root}"
    assert not (root / "target").exists(), "legacy kb/target tree reappeared"
    assert not (root / "KB_INDEX.md").exists(), "legacy kb/KB_INDEX.md reappeared"


def test_okf_layout_is_present_and_non_empty():
    for sub in ("okf/runbooks", "okf/reference"):
        directory = kb_root() / sub
        assert directory.is_dir(), f"OKF knowledge directory missing: {directory}"
        assert any(directory.rglob("*.md")), f"OKF knowledge directory empty: {directory}"


# ---------------------------------------------------------------------------
# (a) OKF-only startup: every brief builder renders without file-not-found.
# (b) No brief references the deleted legacy layout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_brief_builder_renders_okf_only(tmp_path, name):
    env = _seed_env(tmp_path)
    workspace = _seed_workspace(tmp_path, "5_Cumsum")
    brief = _BUILDERS[name](
        "5_Cumsum", workspace,
        lane=0, spawn_index=1, iter_cap_remaining=15,
        env=env,
    )
    assert isinstance(brief, str) and len(brief) > 1000, (
        f"{name} brief looks truncated: {len(brief)} chars"
    )


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_brief_has_no_legacy_target_paths(tmp_path, name):
    env = _seed_env(tmp_path)
    workspace = _seed_workspace(tmp_path, "5_Cumsum")
    brief = _BUILDERS[name](
        "5_Cumsum", workspace,
        lane=0, spawn_index=1, iter_cap_remaining=15,
        env=env,
    )
    for token in _LEGACY_LAYOUT_TOKENS:
        assert token not in brief, f"{name} brief references removed legacy layout: {token!r}"


# ---------------------------------------------------------------------------
# (c) OKF retrieval failure is fail-loud, with no silent legacy fallback.
# ---------------------------------------------------------------------------


def test_okf_retrieval_failure_is_fail_loud(monkeypatch):
    """Empty/failed OKF retrieval → loud marker, never a silent fallback.

    `_okf_reference_block` returns "" on every failure shape (index not built,
    query without hits, non-zero exit, unreadable payload) — patching it to ""
    is exactly the retrieval-failure seam, and the composer must answer with
    the loud marker and nothing from the retired legacy system.
    """
    monkeypatch.setattr(brief_kb, "_okf_reference_block", lambda *a, **k: "")
    out = brief_kb.kb_manifest_block("5_Cumsum", workspace=None, target="a5")
    assert _FAIL_LOUD_MARKER in out
    assert "无回退" in out
    for token in _LEGACY_LAYOUT_TOKENS:
        assert token not in out, f"fail-loud block silently fell back to {token!r}"


def test_okf_retrieval_failure_reaches_worker_brief(tmp_path, monkeypatch):
    """The loud marker is not swallowed one layer up: it lands in the kw brief."""
    monkeypatch.setattr(brief_kb, "_okf_reference_block", lambda *a, **k: "")
    env = _seed_env(tmp_path)
    workspace = _seed_workspace(tmp_path, "5_Cumsum")
    brief = build_worker_brief(
        "5_Cumsum", workspace,
        lane=0, spawn_index=1, iter_cap_remaining=15,
        env=env,
    )
    assert _FAIL_LOUD_MARKER in brief
    for token in _LEGACY_LAYOUT_TOKENS:
        assert token not in brief


# ---------------------------------------------------------------------------
# (d) user-local c-tier is unaffected by the bundled-tier removal.
# ---------------------------------------------------------------------------


def test_c_tier_write_read_tombstone_roundtrip(tmp_path):
    """c-tier basic path on the OKF-only tree: write → brief read → tombstone.

    Full c-tier coverage lives in kb_tiering/tests/test_cannbot_c.py and
    test_read_bridge.py; this pins that the OKF-only layout did not break the
    one writable tier the feedback loop depends on.
    """
    from kb_tiering.interface import Entry
    from kb_tiering.read_bridge import build_arbiter, inject_for_brief

    arbiter = build_arbiter(user_kb_root=str(tmp_path / "user_kb"))
    entry = Entry(
        id="customer:placeholder",  # adapter reassigns to the content-hash id on put
        tier="customer",
        role="user-local",
        kind="experience",
        claim="Always pass a non-empty CMAKE_BUILD_TYPE to a MIX ascendc_library build.",
    )
    status, entry_id = arbiter.write(entry, "customer")
    assert status == "WRITTEN", f"c-tier write rejected: {entry_id}"

    rows = inject_for_brief(arbiter, keywords=["CMAKE_BUILD_TYPE"])
    assert any(r["tier"] == "customer" for r in rows), "c-tier read bridge missed the entry"

    provider = arbiter.providers[0]
    provider.tombstone(entry.content_hash, "okf:OL-999")
    status, why = arbiter.write(entry, "customer")
    assert status == "REJECTED" and "tombstoned" in why, (
        f"tombstone did not close the resurrection loop: {status} {why}"
    )


# ---------------------------------------------------------------------------
# (e) Template/example asset paths exist and are non-empty.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", [
    "templates/fa_class/op_kernel",
    "templates/fa_class/op_host",
    "templates/precision-testing",
    "examples/a3_mix_fa_min",
])
def test_template_and_example_paths_are_readable(rel):
    directory = plugin_root() / rel
    assert directory.is_dir(), f"missing asset directory: {directory}"
    assert any(directory.iterdir()), f"empty asset directory: {directory}"
