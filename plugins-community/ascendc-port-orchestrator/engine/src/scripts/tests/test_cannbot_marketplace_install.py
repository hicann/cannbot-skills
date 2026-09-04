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
# Harness/session variables the shared launcher must re-derive itself; a test
# env inherits none of them from the developer's own shell.
_INHERITED_HARNESS_ENV = frozenset({
    "CLAUDE_CONFIG_DIR", "AOG_HARNESS_BACKEND", "OPENCODE", "OPENCODE_PID",
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
})


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
        "ascendc-simt-best-practices", "ascendc-api-best-practices",
    }
    assert not set(_whitelist("SHARED_SKILLS")) & set(_whitelist("LOCAL_SKILLS"))


def test_only_product_owned_skills_use_the_plugin_local_linking_path():
    s = _src()
    assert 'LOCAL_SKILL_ROOT="$PLUGIN_DIR/skills"' in s
    assert 'for skill_dir in "$LOCAL_SKILL_ROOT"/*/' in s
    assert not {
        "ops-precision-standard", "ascendc-docs-search",
        "ascendc-simt-best-practices", "ascendc-api-best-practices",
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


def test_knowledge_dependency_uses_a_manifest_free_skills_root():
    """Do not merge the knowledge team's plugin manifest into its skill bundle."""
    manifest = json.loads(
        (INIT.parent / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    package = "cannbot-knowledge-consumer-skills"
    assert package in manifest["dependencies"]

    repo_root = INIT.parent.parents[1]
    marketplace = json.loads(
        (repo_root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in marketplace["plugins"] if item["name"] == package)
    source_root = repo_root / entry["source"]
    assert source_root == repo_root / "plugins-community/cannbot-knowledge/skills"
    assert entry["skills"] == ["./knowledge-query"]
    assert not (source_root / ".claude-plugin" / "plugin.json").exists()


def test_health_check_accepts_installed_links_or_marketplace_dependencies():
    s = _src()
    assert "marketplace_skill_present" in s
    assert '[ -e "$CONFIG_ROOT/skills/$want" ] || marketplace_skill_present "$want"' in s


def _copy_packaged_plugin(tmp: Path) -> Path:
    """Copy the packaged plugin surface into a cache dir outside any checkout."""
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

    # The O0 safety contract is part of the packaged plugin, not an optional
    # checkout-only resource.  Keep the fixture representative of a complete
    # marketplace payload while allowing a dedicated test below to remove one.
    for rel in (
        "kb/shared/ANTI_PRESSURE_PROTOCOLS.md",
        "kb/KB_INDEX.md",
        "kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md",
    ):
        src_file = src / rel
        dst_file = plugin / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    # workspace contains run artefacts, but its template is part of the product.
    ws = plugin / "engine" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    tmpl = src / "engine" / "workspace" / ".ascendc_env.template"
    assert tmpl.is_file(), f"env template missing at {tmpl} — fixture cannot be faithful"
    shutil.copy2(tmpl, ws / ".ascendc_env.template")
    return plugin


def _seed_stale_direct_install_settings(plugin: Path) -> None:
    """Simulate a checkout-local direct-install artifact left in the payload.

    The real file contains absolute paths; the marketplace installer must
    discard the entire runtime-only surface.
    """
    stale_settings = plugin / "engine" / ".claude" / "settings.json"
    stale_settings.parent.mkdir(parents=True, exist_ok=True)
    stale_settings.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{
            "type": "command",
            "command": "python3 /Users/junming/stale/workflow_critic.py",
        }]}]}}),
        encoding="utf-8",
    )


def _seed_dependency_packages(ccd: Path) -> None:
    """Materialise the companion skill packages the installer depends on."""
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


def _fake_marketplace_tree(tmp: Path, with_dependencies: bool) -> tuple[Path, dict[str, str]]:
    """Build a faithful cache layout that is deliberately outside a checkout."""
    plugin = _copy_packaged_plugin(tmp)
    _seed_stale_direct_install_settings(plugin)
    ccd = tmp / "ccd"
    if with_dependencies:
        _seed_dependency_packages(ccd)
    ccd.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CLAUDE_CONFIG_DIR=str(ccd),
        ASCENDC_PORT_USER_KB=str(tmp / "kb"),
    )
    return plugin, env


def _run(plugin: Path, env: dict[str, str], tool: str = "claude") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), "init.sh", "global", tool],
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
            installed = Path(env["CLAUDE_CONFIG_DIR"]) / "skills" / name
            assert installed.is_symlink()
            assert (installed / "SKILL.md").is_file()
            cache_root = Path(env["CLAUDE_CONFIG_DIR"]) / "plugins" / "cache"
            assert installed.resolve().is_relative_to(cache_root.resolve())
        assert (plugin / "engine" / "workspace" / ".ascendc_env").is_file()
        # A copied developer checkout may carry the gitignored direct-checkout
        # settings surface.  Marketplace installs must remove it so Claude does
        # not load absolute hook paths from the source machine instead of the
        # plugin-native ${CLAUDE_PLUGIN_ROOT} hooks.json.
        assert not (plugin / "engine" / ".claude").exists()
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


def test_incomplete_marketplace_payload_fails_before_health_report():
    """Missing O0 KB guards must fail at install time, not on first agent run."""
    with tempfile.TemporaryDirectory() as t:
        plugin, env = _fake_marketplace_tree(Path(t), with_dependencies=True)
        (plugin / "kb" / "shared" / "ANTI_PRESSURE_PROTOCOLS.md").unlink()
        r = _run(plugin, env)
        assert r.returncode != 0
        assert "mandatory O0 KB" in r.stdout
        assert "kb/shared/ANTI_PRESSURE_PROTOCOLS.md" in r.stdout


def test_opencode_marketplace_cache_fails_loudly_without_reading_claude_dependencies():
    """OpenCode's no-Claude contract is explicit until its own dependency bundle exists."""
    with tempfile.TemporaryDirectory() as t:
        plugin, env = _fake_marketplace_tree(Path(t), with_dependencies=True)
        r = _run(plugin, env, tool="opencode")
        assert r.returncode != 0
        assert "full cannbot-skills checkout" in r.stdout
        assert "Claude marketplace cache" in r.stdout


@pytest.mark.parametrize(
    ("skill_name", "mode"),
    (("ascendc-cross-gen-port", "port-a3-ops"), ("ascendc-backward-gen", "backward")),
)
def test_entry_skill_delegates_to_the_shared_launcher(skill_name, mode):
    """The entry skill must hand off to the shared launcher, not re-derive the launch.

    This contract used to be checked against an inline shell snippet pasted into each
    SKILL.md. That snippet is now a single shared script (scripts/launch_orchestrator.sh)
    so a second harness cannot grow a third divergent copy; the assertions therefore moved
    to the launcher (see the test below), and what remains checkable here is that the skill
    delegates, passes its own base dir, and selects the right mode.
    """
    text = (INIT.parent / "skills" / skill_name / "SKILL.md").read_text()
    command_line = next(line for line in text.splitlines() if 'command="' in line)

    for token in ("launch_orchestrator.sh", "--skill-base", "--mode", mode):
        assert token in command_line, f"{skill_name} entry does not delegate: missing {token}"
    # The old inline form must not creep back in alongside the launcher.
    assert "python3 -m orchestrator" not in command_line, (
        "entry skill re-derives the launch instead of delegating to the shared launcher"
    )


def test_shared_launcher_enforces_self_containment():
    """The launcher owns the two settings whose absence fails SILENTLY.

    * AOG_HARNESS_BACKEND: unset simply falls back to Claude Code, so an opencode session
      would spawn `claude` with no error at all.
    * CLAUDE_CONFIG_DIR: unset makes a spawned worker fall back to ~/.claude, where a
      different operator-generation suite may be installed — wrong KB, broken
      self-containment, again with no error.
    Both must be exported by the launcher before the orchestrator process starts, because
    the dispatch sites bind their backend at module import time.
    """
    launcher = (INIT.parent / "scripts" / "launch_orchestrator.sh").read_text()

    for token in (
        "resolve_engine.py",            # engine resolved, never guessed from cwd
        "--base-dir",
        "workspace/.ascendc_env",       # config validated before launch
        "export AOG_HARNESS_BACKEND",   # harness selection made explicit
        "CLAUDE_CONFIG_DIR",            # self-containment for the CC path
        "python3 -m orchestrator",
    ):
        assert token in launcher, f"shared launcher lost: {token}"


def _fake_launcher_tree(root: Path) -> tuple[Path, Path]:
    """A minimal plugin+engine tree the real launcher can drive to completion.

    The engine's `orchestrator` package is a stub that prints the environment it was started
    with, so the launcher can be observed rather than read.
    """
    plugin = root / "plugin"
    scripts = plugin / "scripts"
    engine = plugin / "engine"
    orch = engine / "src" / "scripts" / "orchestrator"
    orch.mkdir(parents=True)
    (engine / "workspace").mkdir(parents=True)
    (engine / "workspace" / ".ascendc_env").write_text("TARGET=a5\n")
    (orch / "__main__.py").write_text(
        "import os, sys\n"
        "print('BACKEND=' + os.environ.get('AOG_HARNESS_BACKEND', ''))\n"
        "print('CCDIR=' + os.environ.get('CLAUDE_CONFIG_DIR', ''))\n"
        "print('ENVFILE=' + os.environ.get('ASCENDC_ENV_PATH', ''))\n"
        "print('ARGV=' + ' '.join(sys.argv[1:]))\n"
    )
    scripts.mkdir(parents=True)
    shutil.copy(INIT.parent / "scripts" / "launch_orchestrator.sh", scripts / "launch_orchestrator.sh")
    (scripts / "resolve_engine.py").write_text(
        "import sys\n"
        f"sys.stdout.write({str(engine)!r})\n"
    )
    return plugin, engine


def _run_launcher(
    plugin: Path,
    harness: str,
    home: Path,
    reference_args: list[str] | None = None,
    resume_op: str | None = None,
) -> dict:
    # Cleared so the launcher's own exports are the only source of these: an inherited
    # CLAUDE_CONFIG_DIR would make a launcher that never sets it look like one that does,
    # and inherited harness/host fingerprints would decide the backend for it.
    _cleared = ("CLAUDE_CONFIG_DIR", "AOG_HARNESS_BACKEND", "OPENCODE", "OPENCODE_PID",
                "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
    env = {k: v for k, v in os.environ.items() if k not in _cleared}
    env["HOME"] = str(home)
    command = [
        _bash(), str(plugin / "scripts" / "launch_orchestrator.sh"),
        "--skill-base", str(plugin / "skills" / "ascendc-cross-gen-port"),
        "--lane", "3",
        "--harness", harness,
    ]
    if resume_op is None:
        command.extend(["--mode", "port-a3-ops", "--source", "/tmp/src-op"])
    else:
        command.extend(["--resume", resume_op])
    if reference_args:
        command.extend(reference_args)
    proc = subprocess.run(
        command,
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, f"launcher failed: {proc.stdout}\n{proc.stderr}"
    observed = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in ("BACKEND", "CCDIR", "ENVFILE", "ARGV"):
            observed[key] = value
    return observed


def test_shared_launcher_actually_exports_what_the_orchestrator_reads():
    """Observe the launcher, do not read it.

    The token check above passes on a launcher that merely MENTIONS these names — deleting
    the `export CLAUDE_CONFIG_DIR=...` line leaves the name behind in the comment explaining
    it, so the grep stays green while self-containment is gone. Both settings fail silently
    at runtime, so the test for them has to be the one that runs the thing.
    """
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        plugin, _ = _fake_launcher_tree(root)
        home = root / "home"
        (home / ".claude").mkdir(parents=True)

        cc = _run_launcher(plugin, "claude_code", home)
        assert cc["BACKEND"] == "claude_code"
        assert cc["CCDIR"] == str(home / ".claude"), (
            "launcher did not export CLAUDE_CONFIG_DIR, so a spawned worker falls back to "
            f"whatever ~/.claude happens to hold: {cc['CCDIR']!r}"
        )
        assert cc["ARGV"] == "--port-a3-ops /tmp/src-op --lane 3"

        oc = _run_launcher(plugin, "opencode", home)
        assert oc["BACKEND"] == "opencode", (
            "launcher did not export AOG_HARNESS_BACKEND=opencode; the dispatch sites bind "
            "the backend at import time and would silently spawn `claude` instead"
        )


def test_shared_launcher_pins_plugin_workspace_env_over_stale_process_override():
    """A stale host env file must not redirect a marketplace run to SSH."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        plugin, engine = _fake_launcher_tree(root)
        home = root / "home"
        (home / ".claude").mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k not in _INHERITED_HARNESS_ENV}
        env.update({
            "HOME": str(home),
            "ASCENDC_ENV_PATH": "/tmp/stale-host-config",
            "ASCENDC_ENV_FILE": "/tmp/stale-host-config",
        })
        proc = subprocess.run(
            [
                _bash(), str(plugin / "scripts" / "launch_orchestrator.sh"),
                "--skill-base", str(plugin / "skills" / "ascendc-cross-gen-port"),
                "--mode", "port-a3-ops", "--source", "/tmp/src-op", "--lane", "0",
                "--harness", "claude_code",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"launcher failed: {proc.stdout}\n{proc.stderr}"
        env_lines = [line for line in proc.stdout.splitlines() if line.startswith("ENVFILE=")]
        assert env_lines == [f"ENVFILE={engine / 'workspace' / '.ascendc_env'}"]


def test_shared_launcher_passes_the_native_npubench_pair():
    """Native old-format task/root flags must survive the common entry layer."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        plugin, _ = _fake_launcher_tree(root)
        home = root / "home"
        (home / ".claude").mkdir(parents=True)
        task_root = root / "inputs" / "npu_benchmark"
        task = task_root / "level1" / "3_Add.py"
        observed = _run_launcher(
            plugin,
            "claude_code",
            home,
            [
                "--reference-source", "npubench",
                "--npubench-task", str(task),
                "--npubench-root", str(task_root),
            ],
        )

        assert observed["ARGV"] == (
            "--port-a3-ops /tmp/src-op --lane 3 "
            "--reference-source npubench "
            f"--npubench-task {task} --npubench-root {task_root}"
        )


def test_shared_launcher_refuses_npubench_without_its_task_before_launch():
    proc = subprocess.run(
        [
            _bash(), str(INIT.parent / "scripts" / "launch_orchestrator.sh"),
            "--skill-base", "/tmp/irrelevant-skill-base",
            "--mode", "port-a3-ops", "--source", "/tmp/irrelevant-source",
            "--reference-source", "npubench",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "requires --npubench-task" in proc.stderr


def test_shared_launcher_passes_resume_through_as_a_lifecycle_command():
    """--resume is an engine lifecycle command, not a new-run modifier.

    The engine CLI rejects ``--resume`` combined with ``--port-a3-ops`` (its operand
    is the scoped workspace name, not the source path), so the launcher must
    switch the invocation form instead of appending the flag to a new run.
    """
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        plugin, _ = _fake_launcher_tree(root)
        home = root / "home"
        (home / ".claude").mkdir(parents=True)

        observed = _run_launcher(plugin, "claude_code", home, resume_op="3_Add")

        assert observed["ARGV"] == "--resume 3_Add --lane 3"


def test_shared_launcher_refuses_resume_with_cold_start_before_launch():
    """The engine treats --resume and --cold-start as mutually exclusive."""
    proc = subprocess.run(
        [
            _bash(), str(INIT.parent / "scripts" / "launch_orchestrator.sh"),
            "--skill-base", "/tmp/irrelevant-skill-base",
            "--resume", "3_Add", "--cold-start",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "--resume and --cold-start are mutually exclusive" in proc.stderr


def test_shared_launcher_refuses_resume_with_creation_arguments_before_launch():
    """A resume must fail closed instead of silently dropping --mode/--source."""
    proc = subprocess.run(
        [
            _bash(), str(INIT.parent / "scripts" / "launch_orchestrator.sh"),
            "--skill-base", "/tmp/irrelevant-skill-base",
            "--resume", "3_Add", "--mode", "port-a3-ops", "--source", "/tmp/src-op",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "--resume cannot be combined with" in proc.stderr


def test_missing_marketplace_dependencies_fails_loudly():
    """Bundled product skills must not mask missing shared dependencies."""
    with tempfile.TemporaryDirectory() as t:
        plugin, env = _fake_marketplace_tree(Path(t), with_dependencies=False)
        r = _run(plugin, env)
        assert r.returncode != 0, (
            "installer reported success with marketplace dependencies absent.\n"
            f"stdout:\n{r.stdout}"
        )


def test_project_install_refuses_to_overwrite_its_own_command_templates():
    """`init.sh project opencode` run from the plugin directory must not eat the templates.

    At project level CONFIG_ROOT is `$PWD/.opencode`, so running it from inside the plugin
    makes destination and SOURCE the same file. `sed src > dst` truncates the redirect target
    before sed reads it, which empties the template — a tracked file in this repo — and the
    installer then counted it as installed and reported success, because its "did the
    placeholder get substituted?" check passes vacuously on an empty file.

    Run against a COPY of the plugin: the condition is only that CONFIG_ROOT and the template
    directory are the same path, which a copy reproduces exactly, and running the real
    installer in the checkout would litter it with install artefacts.
    """
    with tempfile.TemporaryDirectory() as t:
        plugin = Path(t) / "ascendc-port-orchestrator"
        shutil.copytree(
            INIT.parent, plugin, symlinks=True,
            ignore=shutil.ignore_patterns("output", "__pycache__", "node_modules", ".git"),
        )
        templates = sorted((plugin / ".opencode" / "command").glob("*.md"))
        assert templates, "no opencode command templates to protect"
        before = {t_: t_.read_bytes() for t_ in templates}
        assert all(before.values()), "fixture templates were already empty"

        proc = subprocess.run(
            [_bash(), str(plugin / "init.sh"), "project", "opencode"],
            cwd=str(plugin), capture_output=True, text=True, timeout=600,
        )

        after = {t_: t_.read_bytes() for t_ in templates}
        emptied = [t_.name for t_ in templates if not after[t_]]
        assert not emptied, f"installer emptied its own source templates: {emptied}"
        damaged = [t_.name for t_ in templates if after[t_] != before[t_]]
        assert not damaged, f"installer modified its own source templates: {damaged}"
        assert proc.returncode != 0, (
            "installer reported success while installing into the plugin's own source tree:\n"
            f"{proc.stdout[-2000:]}"
        )
