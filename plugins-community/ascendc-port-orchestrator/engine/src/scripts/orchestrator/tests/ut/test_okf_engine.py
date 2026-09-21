# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Contract tests for the read-only cannbot-knowledge compatibility resolver."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

from _kb_fixture_helpers import _valid_root

_OKF_ENGINE = Path(__file__).resolve().parents[3] / "okf" / "okf_engine.py"
_spec = importlib.util.spec_from_file_location("okf_engine_ut", _OKF_ENGINE)
okf_engine = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(okf_engine)


def test_missing_project_install_fails_closed(no_external_kb):
    assert okf_engine.knowledge_root() is None
    assert okf_engine.knowledge_query_script() is None


def test_project_install_config_resolves_full_checkout(tmp_path, monkeypatch):
    root = _valid_root(tmp_path)
    project = tmp_path / "operator-project"
    config = project / ".cannbot" / "knowledge.env"
    config.parent.mkdir(parents=True)
    config.write_text(f"CANNBOT_KNOWLEDGE_ROOT='{root}'\n")
    monkeypatch.delenv("CANNBOT_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setenv("CANNBOT_PROJECT_ROOT", str(project))
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert okf_engine.knowledge_root() == root.resolve()
    assert okf_engine.knowledge_query_script() == (
        root
        / ".agents"
        / "skills"
        / "knowledge-query"
        / "scripts"
        / "knowledge_query.py"
    )


def test_marketplace_skill_package_is_not_a_knowledge_root(tmp_path, monkeypatch):
    skills = tmp_path / "cannbot-knowledge-consumer-skills"
    query = skills / "knowledge-query" / "scripts" / "knowledge_query.py"
    query.parent.mkdir(parents=True)
    query.write_text("# query only\n")
    monkeypatch.setenv("CANNBOT_KNOWLEDGE_ROOT", str(skills))
    from briefs.external_kb import _ROOT_CACHE

    _ROOT_CACHE.clear()

    assert okf_engine.knowledge_root() is None


def test_plugin_local_okf_tree_is_never_selected(no_external_kb):
    from briefs.external_kb import _PLUGIN_ROOT

    assert (_PLUGIN_ROOT / "kb" / "okf").is_dir()
    assert okf_engine.knowledge_root() is None


def test_shell_wrapper_queries_full_checkout_read_only(tmp_path):
    root = _valid_root(tmp_path)
    index = root / "artifacts" / "indexes" / "knowledge.sqlite3"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"fixture")
    wrapper = _OKF_ENGINE.with_name("okf_kb.sh")
    env = dict(os.environ, CANNBOT_KNOWLEDGE_ROOT=str(root))

    result = subprocess.run(
        ["/bin/bash", str(wrapper), "search", "--query", "vector reduction"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)
    assert argv[:3] == ["--knowledge-root", str(root.resolve()), "search"]
    assert "--domain" in argv and argv[argv.index("--domain") + 1] == "ops"
    assert "--technology" in argv and argv[argv.index("--technology") + 1] == "ascendc"
    assert argv[-2:] == ["--query", "vector reduction"]

    rejected = subprocess.run(
        ["/bin/bash", str(wrapper), "build"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert rejected.returncode == 64
    assert "read-only consumer" in rejected.stderr
