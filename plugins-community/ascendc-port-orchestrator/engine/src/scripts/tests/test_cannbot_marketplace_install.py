# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""The installer must work both from a checkout and a marketplace cache.

Product-owned Skills are bundled below ``<plugin>/skills``. Reusable ops Skills
and ``knowledge-query`` stay in their canonical packages and are resolved from
the repository checkout or declared marketplace dependencies.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


INIT = Path(__file__).resolve().parents[4] / "init.sh"
_BASH = shutil.which("bash")
_ENTRY_SKILLS = ("ascendc-cross-gen-port", "ascendc-backward-gen")


def _bash() -> str:
    if _BASH is None:
        pytest.skip("bash executable not found")
    return _BASH


def _src() -> str:
    assert INIT.is_file(), f"init.sh not found at {INIT} — this test would pass vacuously"
    return INIT.read_text(encoding="utf-8")


def _whitelist(var: str) -> list[str]:
    m = re.search(rf'^{var}="([^"]*)"', _src(), re.M)
    assert m, f"{var} whitelist not found in init.sh"
    return m.group(1).split()


def test_init_sh_is_present():
    assert len(_src()) > 1000, "init.sh unexpectedly tiny — check the path"


@pytest.mark.parametrize("skill_name", _ENTRY_SKILLS)
def test_entry_skill_engine_resolver_supports_source_tree(skill_name, tmp_path):
    base = INIT.parent / "skills" / skill_name
    resolver = base.resolve().parent.parent / "scripts" / "resolve_engine.py"
    env = dict(os.environ, CLAUDE_CONFIG_DIR=str(tmp_path / "empty-config"))

    result = subprocess.run(
        [sys.executable, str(resolver), "--base-dir", str(base)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == (INIT.parent / "engine").resolve()


def test_entry_skill_engine_resolver_accepts_existing_manifest_hook_path(tmp_path):
    config_root = tmp_path / "config"
    base = config_root / "skills" / "ascendc-cross-gen-port"
    base.mkdir(parents=True)
    engine = tmp_path / "installed-plugin" / "engine"
    sentinel = engine / "src" / "scripts" / "orchestrator" / "__main__.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("# fixture\n")
    settings = engine / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text("{}\n")
    (config_root / "cannbot-manifest.json").write_text(json.dumps({
        "plugin": "ascendc-port-orchestrator",
        "hooks_settings_engine": str(settings),
    }))
    resolver = INIT.parent / "scripts" / "resolve_engine.py"

    result = subprocess.run(
        [sys.executable, str(resolver), "--base-dir", str(base)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == engine.resolve()


def test_env_template_contains_resolver_required_fields_for_both_targets():
    template = INIT.parent / "engine" / "workspace" / ".ascendc_env.template"
    text = template.read_text(encoding="utf-8")
    for target in ("A5", "A3"):
        for field in ("HOST", "USER", "PASSWORD", "CONTAINER", "CANN_PATH", "SOC_VERSION"):
            assert f"{target}_{field}=" in text, f"template missing {target}_{field}"


def test_checkout_detection_resolves_shared_skills_from_ops():
    s = _src()
    assert 'SHARED_SKILL_ROOT="$PLUGIN_DIR/../../ops"' in s
    assert 'if [ -d "$SHARED_SKILL_ROOT" ]; then DIRECT_CHECKOUT=1' in s
    assert set(_whitelist("SHARED_SKILLS")) == {
        "ops-precision-standard", "ascendc-docs-search",
        "ascendc-simt-best-practices",
    }
    assert not set(_whitelist("SHARED_SKILLS")) & set(_whitelist("LOCAL_SKILLS"))


def test_only_product_owned_skills_use_the_plugin_local_linking_path():
    s = _src()
    assert 'LOCAL_SKILL_ROOT="$PLUGIN_DIR/skills"' in s
    assert 'for skill_dir in "$LOCAL_SKILL_ROOT"/*/' in s
    assert not {
        "ops-precision-standard", "ascendc-docs-search",
        "ascendc-simt-best-practices",
    } & set(_whitelist("LOCAL_SKILLS"))


def test_shared_skills_are_declared_as_one_minimal_marketplace_dependency():
    manifest = json.loads(
        (INIT.parent / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    package = "ascendc-port-orchestrator-shared-skills"
    assert package in manifest["dependencies"]

    repo_root = INIT.parent.parents[1]
    marketplace = json.loads(
        (repo_root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in marketplace["plugins"] if item["name"] == package)
    assert entry["source"] == "./ops"
    assert {Path(item).name for item in entry["skills"]} == set(
        _whitelist("SHARED_SKILLS")
    )


def test_health_check_accepts_installed_links_or_marketplace_dependencies():
    s = _src()
    assert "marketplace_skill_present" in s
    assert '[ -e "$CONFIG_ROOT/skills/$want" ] || marketplace_skill_present "$want"' in s


def _fake_marketplace_tree(tmp: Path, with_dependencies: bool) -> tuple[Path, dict[str, str]]:
    """Build a faithful cache layout that is deliberately outside a checkout."""
    plugin = tmp / "cache" / "cannbot" / "ascendc-port-orchestrator" / "0.1.4"
    plugin.mkdir(parents=True)
    src = INIT.parent
    shutil.copy2(INIT, plugin / "init.sh")
    for sub in ("agents", "engine", "hooks", "scripts", "skills"):
        if (src / sub).is_dir():
            shutil.copytree(
                src / sub,
                plugin / sub,
                symlinks=True,
                ignore=shutil.ignore_patterns("output", "__pycache__", "workspace"),
            )

    # workspace contains run artefacts, but its template is part of the product.
    ws = plugin / "engine" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    tmpl = src / "engine" / "workspace" / ".ascendc_env.template"
    assert tmpl.is_file(), f"env template missing at {tmpl} — fixture cannot be faithful"
    shutil.copy2(tmpl, ws / ".ascendc_env.template")

    ccd = tmp / "ccd"
    if with_dependencies:
        packages = {
            "ascendc-port-orchestrator-shared-skills": _whitelist("SHARED_SKILLS"),
            "cannbot-knowledge-consumer-skills": _whitelist("KNOWLEDGE_SKILLS"),
        }
        for package, skills in packages.items():
            comp = ccd / "plugins" / "cache" / "cannbot" / package / "1.0.0"
            for name in skills:
                skill_dir = comp / name
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    ccd.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CLAUDE_CONFIG_DIR=str(ccd),
        ASCENDC_PORT_USER_KB=str(tmp / "kb"),
    )
    return plugin, env


def _run(plugin: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), "init.sh", "global", "claude"],
        cwd=plugin,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_installer_completes_from_a_marketplace_layout():
    with tempfile.TemporaryDirectory() as t:
        plugin, env = _fake_marketplace_tree(Path(t), with_dependencies=True)
        r = _run(plugin, env)
        assert r.returncode == 0, (
            "init.sh failed from a marketplace-installed copy.\n"
            f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
        for name in _whitelist("LOCAL_SKILLS"):
            assert (Path(env["CLAUDE_CONFIG_DIR"]) / "skills" / name).is_symlink()
        for name in _whitelist("SHARED_SKILLS") + _whitelist("KNOWLEDGE_SKILLS"):
            assert not (Path(env["CLAUDE_CONFIG_DIR"]) / "skills" / name).exists()
        assert (plugin / "engine" / "workspace" / ".ascendc_env").is_file()
        manifest = json.loads(
            (Path(env["CLAUDE_CONFIG_DIR"]) / "cannbot-manifest.json").read_text()
        )
        assert Path(manifest["plugin_root"]) == plugin.resolve()
        assert Path(manifest["engine_root"]) == (plugin / "engine").resolve()
        for name in _ENTRY_SKILLS:
            installed_base = Path(env["CLAUDE_CONFIG_DIR"]) / "skills" / name
            resolver = (
                installed_base.resolve().parent.parent / "scripts" / "resolve_engine.py"
            )
            assert resolver.is_file()
            resolved = subprocess.run(
                [sys.executable, str(resolver), "--base-dir", str(installed_base)],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert resolved.returncode == 0, resolved.stderr
            assert Path(resolved.stdout.strip()) == (plugin / "engine").resolve()


@pytest.mark.parametrize(
    ("skill_name", "mode"),
    (("ascendc-cross-gen-port", "--port-a3"), ("ascendc-backward-gen", "--backward")),
)
def test_entry_skill_background_launch_is_self_contained(skill_name, mode):
    text = (INIT.parent / "skills" / skill_name / "SKILL.md").read_text()
    command_line = next(line for line in text.splitlines() if 'command="' in line)

    for token in (
        "SKILL_BASE=", "realpath", "../../scripts/resolve_engine.py",
        "ENGINE_DIR=", "--base-dir", "$ENGINE_DIR/workspace/.ascendc_env",
        'cd \\"$ENGINE_DIR\\"', mode,
    ):
        assert token in command_line


def test_missing_marketplace_dependencies_fails_loudly():
    """Bundled product skills must not mask missing shared dependencies."""
    with tempfile.TemporaryDirectory() as t:
        plugin, env = _fake_marketplace_tree(Path(t), with_dependencies=False)
        r = _run(plugin, env)
        assert r.returncode != 0, (
            "installer reported success with marketplace dependencies absent.\n"
            f"stdout:\n{r.stdout}"
        )
