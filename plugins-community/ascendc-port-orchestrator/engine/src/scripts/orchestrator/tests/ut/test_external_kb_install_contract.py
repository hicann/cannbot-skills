# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Contract tests for consuming cannbot-knowledge master installation output."""

import os
from pathlib import Path

from briefs import external_kb

from _kb_fixture_helpers import _valid_root


def test_project_install_config_is_discovered_from_child_directory(
    tmp_path, monkeypatch
):
    root = _valid_root(tmp_path)
    project = tmp_path / "operator-project"
    nested = project / "src" / "op"
    nested.mkdir(parents=True)
    config = project / ".cannbot" / "knowledge.env"
    config.parent.mkdir()
    config.write_text(f"export CANNBOT_KNOWLEDGE_ROOT='{root}'\n")

    monkeypatch.delenv("CANNBOT_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.delenv("CANNBOT_PROJECT_ROOT", raising=False)
    # Inert .ascendc_env source: this test exercises the installer config path.
    monkeypatch.setenv(
        "ASCENDC_ENV_PATH", str(tmp_path / "absent" / ".ascendc_env")
    )
    monkeypatch.chdir(nested)
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert external_kb.external_kb_root() == root.resolve()
    assert os.environ["CANNBOT_KNOWLEDGE_ROOT"] == str(root.resolve())


def test_launcher_project_anchor_survives_engine_chdir(tmp_path, monkeypatch):
    root = _valid_root(tmp_path)
    project = tmp_path / "operator-project"
    config = project / ".cannbot" / "knowledge.env"
    config.parent.mkdir(parents=True)
    config.write_text(f"CANNBOT_KNOWLEDGE_ROOT={root}\n")
    engine = tmp_path / "plugin" / "engine"
    engine.mkdir(parents=True)

    monkeypatch.delenv("CANNBOT_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setenv("CANNBOT_PROJECT_ROOT", str(project))
    # Inert .ascendc_env source: this test exercises the installer config path.
    monkeypatch.setenv(
        "ASCENDC_ENV_PATH", str(tmp_path / "absent" / ".ascendc_env")
    )
    monkeypatch.chdir(engine)
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert external_kb.external_kb_root() == root.resolve()


def test_explicit_environment_overrides_project_config(tmp_path, monkeypatch):
    configured = _valid_root(tmp_path, "configured")
    explicit = _valid_root(tmp_path, "explicit")
    project = tmp_path / "operator-project"
    config = project / ".cannbot" / "knowledge.env"
    config.parent.mkdir(parents=True)
    config.write_text(f"CANNBOT_KNOWLEDGE_ROOT={configured}\n")

    monkeypatch.setenv("CANNBOT_PROJECT_ROOT", str(project))
    monkeypatch.setenv("CANNBOT_KNOWLEDGE_ROOT", str(explicit))
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert external_kb.external_kb_root() == explicit.resolve()


def test_ascendc_env_is_a_knowledge_configuration_source(tmp_path, monkeypatch):
    """`.ascendc_env` CANNBOT_KNOWLEDGE_ROOT is the user-facing config surface.

    O0 preflight requires it when neither an explicit process env var nor a
    project installer config provides the knowledge root.
    """
    root = _valid_root(tmp_path)
    ascendc_env = tmp_path / ".ascendc_env"
    ascendc_env.write_text(f"CANNBOT_KNOWLEDGE_ROOT={root}\n")
    empty_project = tmp_path / "operator-project"
    empty_project.mkdir()

    monkeypatch.delenv("CANNBOT_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setenv("CANNBOT_PROJECT_ROOT", str(empty_project))
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(ascendc_env))
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert external_kb.external_kb_root() == root.resolve()


def test_ascendc_env_overrides_project_install_config(tmp_path, monkeypatch):
    env_root = _valid_root(tmp_path, "from-ascendc-env")
    project_root = _valid_root(tmp_path, "from-project")
    ascendc_env = tmp_path / ".ascendc_env"
    ascendc_env.write_text(f"CANNBOT_KNOWLEDGE_ROOT={env_root}\n")
    project = tmp_path / "operator-project"
    config = project / ".cannbot" / "knowledge.env"
    config.parent.mkdir(parents=True)
    config.write_text(f"CANNBOT_KNOWLEDGE_ROOT={project_root}\n")

    monkeypatch.delenv("CANNBOT_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setenv("CANNBOT_PROJECT_ROOT", str(project))
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(ascendc_env))
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert external_kb.external_kb_root() == env_root.resolve()


def test_plugin_local_card_is_not_a_runtime_fallback(no_external_kb):
    from briefs.external_kb import _PLUGIN_ROOT

    rel = "reference/porter/handbook/language_reference.md"
    local = _PLUGIN_ROOT / "kb" / "okf" / rel
    assert local.is_file(), "fixture requires a retained plugin-local card"
    assert external_kb.resolve_kb_ref(rel) is None


def test_canonical_knowledge_path_resolves_in_external_checkout(tmp_path, monkeypatch):
    root = _valid_root(tmp_path)
    card = root / "knowledge" / "ops" / "ascendc" / "concepts" / "example.md"
    card.parent.mkdir(parents=True)
    card.write_text("# example\n")
    monkeypatch.setenv("CANNBOT_KNOWLEDGE_ROOT", str(root))
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert external_kb.resolve_kb_ref(
        "knowledge/ops/ascendc/concepts/example.md"
    ) == card.resolve()


def test_query_ref_displays_requests_api_cards(monkeypatch):
    calls = []

    def _search(query, target="a5", per_platform_k=10, card_type=None):
        calls.append((query, target, per_platform_k, card_type))
        return [{"local_path": "/external/knowledge/ops/ascendc/apis/foo.md"}]

    monkeypatch.setattr(external_kb, "search_external_cards", _search)
    assert external_kb.query_ref_displays(
        "register vector compute APIs", card_type="apis", limit=3
    ) == ["/external/knowledge/ops/ascendc/apis/foo.md"]
    assert calls == [("register vector compute APIs", "a5", 5, "apis")]
