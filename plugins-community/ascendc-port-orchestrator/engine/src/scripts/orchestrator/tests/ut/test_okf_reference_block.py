# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""M2 OKF read-path (_okf_reference_block / kb_manifest_block): OKF-only, fail-loud.

OKF 是唯一 b-tier 知识来源（OKF-only 迁移 2026-08：ASCENDC_PORT_OKF 开关与
force_legacy_kb 逃生门已移除，不存在 legacy manifest 回退分支）。Invariants locked here:
  1. kb_manifest_block 只产出 OKF 块（或响亮空标记），永远不产出 legacy "# KB MANIFEST"。
  2. exclusivity drops only KNOWLEDGE POINTERS — the source-agnostic discipline + per-target hw spec
     (ALWAYS_LOADED_RULES, ANTI-PRESSURE CHECKPOINT, the chip spec) STAY.
  3. fail-loud-no-fallback: 检索为空/失败 → marker, NEVER a silent legacy fallback.
  4. NEVER raises — any subprocess / JSON / shape failure returns "".
The mocked tests force the kbq/index preconditions True so the mocked subprocess is actually reached.
"""
import json
import subprocess
from pathlib import Path
import pytest
from briefs import _common


def _force_okf_ready(monkeypatch):
    """Make _okf_reference_block reach the (mocked) subprocess: pretend the query script + index exist.
    Also pin CANNBOT_OKF_ENGINE_ROOT so okf_engine's resolver returns on its FIRST (env) candidate under
    the is_file mock — keeps these behavior tests independent of the real filesystem's plugin layout.
    (Real resolver-discovery coverage — env vs sibling vs marketplace dir names, consumer/contributor
    split — lives in test_okf_engine.py against real tmpdirs.)"""
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setenv("CANNBOT_OKF_ENGINE_ROOT", "/fake/cannbot-knowledge")


def test_okf_block_empty_when_engine_missing(monkeypatch):
    """RFC #381: the engine is EXTERNAL (cannbot-knowledge). If it isn't installed, okf_engine resolves
    None → this returns "" and the caller emits the loud marker — NEVER a silent legacy fallback.
    is_file→False simulates cannbot-knowledge absent (no knowledge_query.py at any candidate root).
    """
    monkeypatch.delenv("CANNBOT_OKF_ENGINE_ROOT", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""
    # and end-to-end the manifest turns the empty result into the loud marker, not the legacy manifest.
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "OKF 检索无返回" in out and "# KB MANIFEST" not in out


# --- exclusivity + discipline, deterministic (mocked subprocess) -----------
def _mock_hits(monkeypatch, payload):
    class _CP:
        returncode = 0
        stdout = payload
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _CP())


def test_okf_block_populated_on_valid_hits(monkeypatch):
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        {"path": "runbooks/field-notes/build/ec-13.md", "title": "SyncFunc", "kind": "field_note", "score": 9}]}))
    blk = getattr(_common, '_okf_reference_block')("13_Cat", None, "a5")
    assert "# OKF 知识卡片" in blk and "ec-13.md" in blk


def test_kb_manifest_success_is_okf_with_scaffold(monkeypatch):
    """命中 → OKF block, NO legacy '# KB MANIFEST', discipline + per-target hw kept."""
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        {"path": "runbooks/field-notes/build/ec-13.md", "title": "t", "kind": "field_note", "score": 9}]}))
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "# OKF 知识卡片" in out and "# KB MANIFEST" not in out           # exclusive OKF
    assert "shared/ALWAYS_LOADED_RULES.md" in out                          # BLOCKER regression guard
    assert "ANTI-PRESSURE CHECKPOINT" in out                              # discipline kept
    assert "ascend950pr.md" in out                                        # per-target hw spec kept (a5)


def test_kb_manifest_target_routes_hw_spec(monkeypatch):
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        {"path": "runbooks/field-notes/build/ec-13.md", "title": "t", "kind": "field_note", "score": 9}]}))
    assert "ascend910c.md" in _common.kb_manifest_block("13_Cat", None, "a3")   # a3 → 910c
    assert "ascend910b.md" in _common.kb_manifest_block("13_Cat", None, "a2")   # a2 → 910b


# --- fail-loud, NO silent legacy fallback -----------------------------------
def test_kb_manifest_empty_is_marker_no_fallback(monkeypatch):
    """OKF 检索为空 → LOUD marker, NEVER the legacy manifest; discipline still there."""
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": []}))
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "OKF 检索无返回" in out            # loud marker
    assert "# KB MANIFEST" not in out            # NO silent legacy fallback
    assert "shared/ALWAYS_LOADED_RULES.md" in out  # discipline still present


# --- never raises (subprocess actually reached via _force_okf_ready) --------
@pytest.mark.parametrize("payload", ['{"hits":"x"}', '{"hits":[1,2]}', '[1,2,3]', 'not json', '{}'])
def test_okf_block_never_raises_on_malformed_output(monkeypatch, payload):
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, payload)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""


def test_okf_block_empty_on_subprocess_error(monkeypatch):
    _force_okf_ready(monkeypatch)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="knowledge_query", timeout=30)

    monkeypatch.setattr(subprocess, "run", _boom)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""


def test_kb_manifest_subprocess_error_is_marker_no_fallback(monkeypatch):
    """检索子进程异常 → still marker + no legacy fallback (fail-loud end-to-end)."""
    _force_okf_ready(monkeypatch)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="knowledge_query", timeout=30)

    monkeypatch.setattr(subprocess, "run", _boom)
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "OKF 检索无返回" in out and "# KB MANIFEST" not in out


# --- archived/deprecated 卡降权（过滤出 top-5 注入） ---------------------------
# 实测回归：pb-36-archived-deprecated-...（标题即"已归档废弃"）曾是 worker 读得最多
# 的卡。归档卡只是 audit trail（安全规则 5 保留不删），不得占用 brief 注入位。
def _hit(path, title="t", **kw):
    h = {"path": path, "title": title, "kind": "field_note", "score": 9}
    h.update(kw)
    return h


def test_okf_block_filters_archived_by_path_and_backfills_top5(monkeypatch):
    """路径命名约定（pb-36-archived-deprecated-...）命中 → 滤掉；第 6 张在役卡递补进 top-5。"""
    _force_okf_ready(monkeypatch)
    hits = [_hit("runbooks/field-notes/build/pb-36-archived-deprecated-2026-05-22-datacopy.md",
                 title="[ARCHIVED/DEPRECATED 2026-05-22] DataCopy srcStride")]
    hits += [_hit(f"runbooks/field-notes/build/ec-{i}.md") for i in range(6)]
    _mock_hits(monkeypatch, json.dumps({"hits": hits}))
    blk = getattr(_common, '_okf_reference_block')("13_Cat", None, "a5")
    assert "archived-deprecated" not in blk                       # 废弃卡被滤掉
    assert "ec-4.md" in blk                                      # 第 5 张在役卡递补进 top-5


def test_okf_block_filters_archived_by_title_marker_and_tags(monkeypatch):
    """路径干净但标题带 [ARCHIVED...] 前缀 / tags metadata 带 deprecated 词元 → 同样滤掉。"""
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        _hit("runbooks/field-notes/build/pb-99-old.md",
             title="[ARCHIVED 2026-01-01] superseded workaround"),
        _hit("runbooks/field-notes/build/pb-100-old.md", tags=["build", "deprecated"]),
        _hit("runbooks/field-notes/build/ec-13.md", title="SyncFunc"),
    ]}))
    blk = getattr(_common, '_okf_reference_block')("13_Cat", None, "a5")
    assert "pb-99-old.md" not in blk and "pb-100-old.md" not in blk
    assert "ec-13.md" in blk


def test_okf_block_keeps_live_card_mentioning_archived_in_title(monkeypatch):
    """误伤守卫：标题正文里提及 archived 的在役卡（OL-168 式）不得被滤掉。"""
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        _hit("runbooks/field-notes/build/ol-168-reference-spec-drift.md",
             title="Reference-spec drift — mtime pre-flight before treating archived kernel as starting material"),
    ]}))
    blk = getattr(_common, '_okf_reference_block')("13_Cat", None, "a5")
    assert "ol-168-reference-spec-drift.md" in blk


def test_okf_block_all_archived_falls_back_to_loud_marker(monkeypatch):
    """命中全是归档卡 → 滤后为空 → 响亮空标记（fail-loud），绝不注入废弃卡。"""
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        _hit("runbooks/field-notes/build/pb-36-archived-deprecated-2026-05-22-datacopy.md"),
    ]}))
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "OKF 检索无返回" in out and "# KB MANIFEST" not in out


# ---------------------------------------------------------------------------
# knowledge-query path roots (2026-09-05)
# ---------------------------------------------------------------------------
def test_okf_hit_path_resolves_reference_bundle_paths():
    """`reference/` cards come back bundle-relative; `kb/okf/<p>` names nothing for them.

    knowledge-query reports each card's path relative to its CONTENT ROOT, and the roots
    are not uniform: `runbooks/...` is okf-root-relative, but a reference card is
    bundle-relative (`porter/...`, `asc-devkit-vendored/...`). Concatenating "kb/okf/"
    verbatim therefore sent the worker to a nonexistent file for all 546 reference cards.
    Measured before the fix: 7 of 40 top-5 slots across 8 sampled queries were dead.

    The contract is stronger than "resolve the root": EVERY returned path is a verified
    existing file, and anything unverifiable returns None so the caller can say so.
    """
    from briefs.brief_kb import _okf_hit_path

    # ut -> tests -> orchestrator -> scripts -> src -> engine -> plugin root
    kb_root = Path(__file__).resolve().parents[6] / "kb" / "okf"
    assert kb_root.is_dir(), "kb/okf missing — PLUGIN_ROOT arithmetic is wrong"

    # runbooks: already okf-root-relative, must be left alone
    runbook = "runbooks/hardware/target-ascend950pr.md"
    assert (kb_root / runbook).is_file(), "fixture card moved; re-point this test"
    assert _okf_hit_path(kb_root, {"path": runbook}) == "kb/okf/" + runbook

    # reference bundle: must gain the `reference/` root
    ref = "porter/handbook/language_reference.md"
    assert (kb_root / "reference" / ref).is_file(), "fixture card moved; re-point this test"
    assert _okf_hit_path(kb_root, {"path": ref}) == "kb/okf/reference/" + ref

    # `local_path` (what knowledge-query actually sends) wins over the probe — but only
    # when it names the SAME card. A `local_path` that resolves to a different existing
    # file must NOT be used: that substitutes one card for another and the worker reads
    # the wrong knowledge, which existence checks alone cannot catch.
    assert _okf_hit_path(
        kb_root, {"path": ref, "local_path": str(kb_root / "reference" / ref)}
    ) == "kb/okf/reference/" + ref
    other = kb_root / "reference" / "porter" / "handbook" / "roofline_model.md"
    assert other.is_file(), "fixture card moved; re-point this test"
    assert _okf_hit_path(kb_root, {"path": ref, "local_path": str(other)}) \
        == "kb/okf/reference/" + ref, "a mismatched local_path must fall through to the probe"

    # A SAME-BASENAME `local_path` in a different directory is still a different card.
    # Matching only the basename let one card be served in place of another while every
    # existence check passed — the failure mode existence checks cannot see.
    same_name_elsewhere = kb_root / "reference" / "porter" / "toolchain" / "language_reference.md"
    assert not same_name_elsewhere.exists(), "fixture assumes this path is free"
    assert _okf_hit_path(kb_root, {"path": ref, "local_path": str(same_name_elsewhere)}) \
        == "kb/okf/reference/" + ref

    # ...but a STALE `local_path` must not be trusted. An index that has gone out of date
    # hands back paths whose files were deleted; taking them on faith reintroduces exactly
    # the dead-path bug this function exists to fix. Here the probe on `path` rescues it.
    assert _okf_hit_path(
        kb_root, {"path": ref, "local_path": str(kb_root / "reference" / "gone.md")}
    ) == "kb/okf/reference/" + ref

    # A `local_path` outside kb/okf (c-tier, or a relocated index) must not raise.
    assert _okf_hit_path(kb_root, {"path": runbook, "local_path": "/tmp/elsewhere.md"}) \
        == "kb/okf/" + runbook

    # kb_root given as a str must work — callers are not required to hand us a Path.
    assert _okf_hit_path(str(kb_root), {"path": runbook}) == "kb/okf/" + runbook


@pytest.mark.parametrize("hit,why", [
    ({}, "empty hit used to yield the bare directory `kb/okf/`"),
    ({"path": None}, "None path used to yield `kb/okf/`"),
    ({"path": ""}, "empty path"),
    ({"path": "zz/absent.md"}, "nothing resolves under either content root"),
    ({"path": "/etc/hosts"}, "absolute path: `Path(root) / '/etc/hosts'` is `/etc/hosts`"),
    ({"path": "../../../etc/hosts"}, "traversal escapes kb/okf"),
    ({"path": "okf/../../etc/hosts"}, "traversal in the middle"),
])
def test_okf_hit_path_returns_none_rather_than_an_unverified_path(hit, why):
    """Unverifiable input must produce None, never a fabricated `kb/okf/...` string.

    A brief line is an instruction to open a file. A path we cannot verify costs the
    worker a turn and returns nothing, so it is worse than admitting we have no path.
    Each case below produced a dead path before codex round 12.
    """
    from briefs.brief_kb import _okf_hit_path

    kb_root = Path(__file__).resolve().parents[6] / "kb" / "okf"
    assert _okf_hit_path(kb_root, hit) is None, why
