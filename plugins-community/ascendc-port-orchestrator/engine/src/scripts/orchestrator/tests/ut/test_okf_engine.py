# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Real-filesystem coverage for okf_engine's external cannbot-knowledge resolver.

The hot-path tests in test_okf_reference_block.py mock Path.is_file globally, which cannot exercise the
resolver's actual discovery (candidate dir names, layout, consumer/contributor split). These tests use
REAL tmpdirs with fake marker files, isolating the plugin root + HOME so no real install interferes.
"""
import importlib.util
import json
from pathlib import Path

import pytest

# okf_engine lives at <plugin>/engine/src/scripts/okf/okf_engine.py; this test at
# <plugin>/engine/src/scripts/orchestrator/tests/ut/ → parents[3] == .../scripts.
_OKF_ENGINE = Path(__file__).resolve().parents[3] / "okf" / "okf_engine.py"
_spec = importlib.util.spec_from_file_location("okf_engine_ut", _OKF_ENGINE)
okf_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(okf_engine)

_TIERS = {
    "query": "skills/knowledge-query/scripts/knowledge_query.py",
    "lint": "skills/knowledge-lint/scripts/knowledge_lint.py",
    "graph": "skills/ops-knowledge-ingest/scripts/okf_graph.py",
}


def _make_ck(root: Path, tiers=("query",), *, scoped_package: bool = False) -> Path:
    """Create a fake cannbot-knowledge plugin dir with the requested skill tiers present."""
    for t in tiers:
        rel = _TIERS[t]
        if scoped_package:
            rel = rel.removeprefix("skills/")
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fake\n", encoding="utf-8")
    return root


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A plugin root with NO cannbot-knowledge siblings; empty tmp HOME; no env overrides."""
    pr = tmp_path / "market" / "ascendc-port-orchestrator"
    pr.mkdir(parents=True)
    monkeypatch.setattr(okf_engine, "_plugin_root", lambda: pr)
    home = tmp_path / "home"
    (home / ".claude" / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CANNBOT_OKF_ENGINE_ROOT", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return tmp_path, pr, home


def test_nothing_installed_resolves_none(isolated):
    assert okf_engine.resolve_engine_root() is None
    assert okf_engine.knowledge_query_script() is None
    assert okf_engine.knowledge_lint_script() is None
    assert okf_engine.okf_graph_dir() is None


def test_env_override_wins(isolated, monkeypatch):
    tmp_path, _, _ = isolated
    ck = _make_ck(tmp_path / "custom_ck", ("query", "lint", "graph"))
    monkeypatch.setenv("CANNBOT_OKF_ENGINE_ROOT", str(ck))
    assert okf_engine.resolve_engine_root() == ck
    assert okf_engine.knowledge_query_script() == ck / _TIERS["query"]
    assert okf_engine.knowledge_lint_script() == ck / _TIERS["lint"]
    assert okf_engine.okf_graph_dir() == (ck / _TIERS["graph"]).parent


def test_sibling_canonical_name(isolated):
    _, pr, _ = isolated
    ck = _make_ck(pr.parent / "cannbot-knowledge", ("query",))
    assert okf_engine.resolve_engine_root() == ck


def test_monorepo_community_plugin(isolated, monkeypatch):
    tmp_path, _, _ = isolated
    pr = tmp_path / "repo" / "plugins-community" / "ascendc-port-orchestrator"
    pr.mkdir(parents=True)
    monkeypatch.setattr(okf_engine, "_plugin_root", lambda: pr)
    ck = _make_ck(
        tmp_path / "repo" / "plugins-community" / "cannbot-knowledge", ("query",)
    )
    assert okf_engine.resolve_engine_root() == ck


def test_legacy_marketplace_prefixed_dir(isolated):
    _, _, home = isolated
    ck = _make_ck(
        home / ".claude" / "plugins" / "cannbot-knowledge-consumer-skills",
        ("query",),
    )
    assert okf_engine.resolve_engine_root() == ck


def test_marketplace_named_dir_and_no_consumer_shadow(isolated):
    """Marketplace installs the dir under the PACKAGE name (not literally 'cannbot-knowledge'), and a
    consumer-only install must NOT shadow a contributor install when resolving the lint tier.
    """
    _, _, home = isolated
    marketplace = home / ".claude" / "plugins" / "cache" / "cannbot"
    consumer = _make_ck(
        marketplace / "cannbot-knowledge-consumer-skills" / "1.0.0", ("query",)
    )
    # query is found via the prefixed dir name (validated by the marker file, not the name)
    assert okf_engine.knowledge_query_script() == consumer / _TIERS["query"]
    # consumer ships no lint → lint must resolve to None, not to the consumer dir
    assert okf_engine.knowledge_lint_script() is None
    # add a contributor install → lint/graph resolve to it despite the consumer coming first alphabetically
    contrib = _make_ck(
        marketplace / "cannbot-knowledge-contributor-skills" / "1.0.0",
        ("query", "lint", "graph"),
    )
    assert okf_engine.knowledge_lint_script() == contrib / _TIERS["lint"]
    assert okf_engine.okf_graph_dir() == (contrib / _TIERS["graph"]).parent


def test_installed_plugins_registry_precedes_stale_cache(isolated):
    """Use the registry's active install instead of an older cache version that still exists."""
    _, _, home = isolated
    plugins = home / ".claude" / "plugins"
    package = plugins / "cache" / "cannbot" / "cannbot-knowledge-consumer-skills"
    _make_ck(package / "9.9.9", ("query",))
    active = _make_ck(package / "1.0.0", ("query",))
    registry = {
        "version": 2,
        "plugins": {
            "cannbot-knowledge-consumer-skills@cannbot": [
                {
                    "scope": "user",
                    "version": "1.0.0",
                    "installPath": str(active),
                }
            ]
        },
    }
    (plugins / "installed_plugins.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    assert okf_engine.resolve_engine_root() == active


def test_registered_scoped_skills_package_resolves_query(isolated):
    """A skills-root marketplace source caches knowledge-query without skills/."""
    _, _, home = isolated
    plugins = home / ".claude" / "plugins"
    active = _make_ck(
        plugins / "cache/cannbot/cannbot-knowledge-consumer-skills/1.0.0",
        ("query",),
        scoped_package=True,
    )
    registry = {
        "plugins": {
            "cannbot-knowledge-consumer-skills@cannbot": [
                {"installPath": str(active)}
            ]
        }
    }
    (plugins / "installed_plugins.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    expected = active / "knowledge-query/scripts/knowledge_query.py"
    assert okf_engine.resolve_engine_root() == active
    assert okf_engine.knowledge_query_script() == expected


def test_relative_registry_install_path(isolated):
    _, _, home = isolated
    plugins = home / ".claude" / "plugins"
    relative = Path(
        "cache/cannbot/cannbot-knowledge-consumer-skills/1.0.0"
    )
    active = _make_ck(plugins / relative, ("query",))
    registry = {
        "plugins": {
            "cannbot-knowledge-consumer-skills@cannbot": [
                {"installPath": str(relative)}
            ]
        }
    }
    (plugins / "installed_plugins.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    assert okf_engine.resolve_engine_root() == active


def test_malformed_registry_does_not_use_stale_cache(isolated):
    _, _, home = isolated
    plugins = home / ".claude" / "plugins"
    _make_ck(
        plugins / "cache/cannbot/cannbot-knowledge-consumer-skills/9.9.9",
        ("query",),
    )
    (plugins / "installed_plugins.json").write_text("{broken", encoding="utf-8")

    assert okf_engine.resolve_engine_root() is None


def test_missing_registered_install_does_not_use_stale_cache(isolated):
    _, _, home = isolated
    plugins = home / ".claude" / "plugins"
    _make_ck(
        plugins / "cache/cannbot/cannbot-knowledge-consumer-skills/9.9.9",
        ("query",),
    )
    registry = {
        "plugins": {
            "cannbot-knowledge-consumer-skills@cannbot": [
                {"installPath": str(plugins / "missing-active")}
            ]
        }
    }
    (plugins / "installed_plugins.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    assert okf_engine.resolve_engine_root() is None


def test_invalid_registered_path_fails_closed(isolated):
    _, _, home = isolated
    plugins = home / ".claude" / "plugins"
    registry = {
        "plugins": {
            "cannbot-knowledge-consumer-skills@cannbot": [
                {"installPath": "~definitely_no_such_user_123/knowledge"}
            ]
        }
    }
    (plugins / "installed_plugins.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    assert okf_engine.resolve_engine_root() is None


def test_multiple_unregistered_cache_versions_are_ambiguous(isolated):
    _, _, home = isolated
    package = (
        home
        / ".claude/plugins/cache/cannbot/cannbot-knowledge-consumer-skills"
    )
    _make_ck(package / "1.0.0", ("query",))
    _make_ck(package / "9.9.9", ("query",))

    assert okf_engine.resolve_engine_root() is None


def test_custom_config_does_not_fallback_to_default_home(isolated, monkeypatch):
    tmp_path, _, home = isolated
    _make_ck(
        home
        / ".claude/plugins/cache/cannbot/cannbot-knowledge-consumer-skills/1.0.0",
        ("query",),
    )
    custom = tmp_path / "custom-claude"
    (custom / "plugins").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))

    assert okf_engine.resolve_engine_root() is None


def test_versioned_seed_cache_without_registry(isolated, monkeypatch):
    tmp_path, _, _ = isolated
    seed = tmp_path / "seed-plugins"
    port = (
        seed
        / "cache/cannbot/ascendc-port-orchestrator/1.0.0"
    )
    port.mkdir(parents=True)
    monkeypatch.setattr(okf_engine, "_plugin_root", lambda: port)
    ck = _make_ck(
        seed
        / "cache/cannbot/cannbot-knowledge-consumer-skills/1.0.0",
        ("query",),
    )

    assert okf_engine.resolve_engine_root() == ck


def test_versioned_seed_cache_honors_invalid_registry(isolated, monkeypatch):
    tmp_path, _, _ = isolated
    seed = tmp_path / "seed-plugins"
    port = (
        seed
        / "cache/cannbot/ascendc-port-orchestrator/1.0.0"
    )
    port.mkdir(parents=True)
    monkeypatch.setattr(okf_engine, "_plugin_root", lambda: port)
    _make_ck(
        seed
        / "cache/cannbot/cannbot-knowledge-consumer-skills/9.9.9",
        ("query",),
    )
    (seed / "installed_plugins.json").write_text("{broken", encoding="utf-8")

    assert okf_engine.resolve_engine_root() is None
