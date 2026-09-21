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


@pytest.fixture(autouse=True)
def _isolated_external_kb(no_external_kb):
    """Keep ambient project knowledge configuration out of these unit tests."""


def _force_okf_ready(monkeypatch):
    """Mark the separately installed knowledge dependency as configured."""
    from briefs import external_kb

    monkeypatch.setattr(
        external_kb, "external_kb_root", lambda: Path("/fake/knowledge")
    )


def test_okf_block_empty_when_knowledge_install_missing(monkeypatch):
    """A missing knowledge installation is loud and never falls back in-plugin."""
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "cannbot-knowledge 未安装或项目配置无效" in out
    assert "# KB MANIFEST" not in out


# --- exclusivity + discipline, deterministic (mocked subprocess) -----------
def _mock_hits(monkeypatch, payload):
    from briefs import external_kb

    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        parsed = {}
    hits = parsed.get("hits", []) if isinstance(parsed, dict) else []
    if not isinstance(hits, list):
        hits = []
    normalized = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        item = dict(hit)
        item.setdefault("local_path", "/fake/knowledge/knowledge/" + str(item.get("path", "")))
        normalized.append(item)
    monkeypatch.setattr(external_kb, "search_external_cards", lambda *a, **k: normalized)


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
    from briefs import external_kb

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="knowledge_query", timeout=30)

    monkeypatch.setattr(external_kb, "search_external_cards", _boom)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""


def test_kb_manifest_subprocess_error_is_marker_no_fallback(monkeypatch):
    """检索子进程异常 → still marker + no legacy fallback (fail-loud end-to-end)."""
    _force_okf_ready(monkeypatch)
    from briefs import external_kb

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="knowledge_query", timeout=30)

    monkeypatch.setattr(external_kb, "search_external_cards", _boom)
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
