#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""cannbot-side equivalent of a5_ops #240 / DEBT-226 skill-scope boundary.

WHY THIS FILE EXISTS
--------------------
a5_ops ships `src/scripts/tests/test_skill_scope_boundary.py`, which asserts that
`src/deploy.sh` partitions `src/skills/*/SKILL.md` by their `scope: customer|dev-ops`
frontmatter. cannbot installs through `init.sh` instead: product-owned runtime
skills live in this community plugin's `skills/`, reusable ops Skills stay in
repository `ops/`, and `knowledge-query` is owned by the sibling knowledge plugin.

Excluding it WITHOUT a replacement would drop mechanical coverage of a security-adjacent
property ("a dev-ops skill must not ship to customers"). This file restores that coverage
against cannbot's actual mechanism: the `init.sh` ownership-tier whitelists and
their literal `INCLUDED_SKILLS` union.

Anchor: a force-pushed v3.18.0 re-sync incorrectly copied a5_ops' `scope: dev-ops`
classification onto cannbot's existing customer `aog-report-gen` capability. The
boundary test now catches that scope drift without silently deleting the local feature.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
PLUGIN_ROOT = _HERE.parents[4]          # ascendc-port-orchestrator/
REPO_ROOT = PLUGIN_ROOT.parents[1]      # cannbot repo root
INIT_SH = PLUGIN_ROOT / "init.sh"
OPS_ROOT = REPO_ROOT / "ops"
LOCAL_ROOT = PLUGIN_ROOT / "skills"
KNOWLEDGE_ROOT = REPO_ROOT / "plugins-community" / "cannbot-knowledge" / "skills"


def _whitelist(var: str) -> list[str]:
    text = INIT_SH.read_text(encoding="utf-8")
    m = re.search(rf'^{var}="([^"]+)"', text, re.M)
    assert m, f"{var} not found in {INIT_SH}"
    return m.group(1).split()


def _scope_of(skill_md: Path) -> str | None:
    try:
        head = skill_md.read_text(encoding="utf-8", errors="ignore").splitlines()[:15]
    except OSError:
        return None
    for line in head:
        m = re.match(r"^scope:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    return None


def _skill_md(name: str) -> Path:
    if name in _whitelist("LOCAL_SKILLS"):
        return LOCAL_ROOT / name / "SKILL.md"
    if name in _whitelist("SHARED_SKILLS"):
        return OPS_ROOT / name / "SKILL.md"
    if name in _whitelist("KNOWLEDGE_SKILLS"):
        return KNOWLEDGE_ROOT / name / "SKILL.md"
    raise AssertionError(f"{name} is not assigned to a skill ownership tier")


def test_init_sh_and_skill_roots_exist():
    """Guard the guard: if the layout moves, fail loudly instead of passing vacuously."""
    assert INIT_SH.is_file(), f"init.sh missing at {INIT_SH}"
    assert LOCAL_ROOT.is_dir(), f"plugin-local skills/ missing at {LOCAL_ROOT}"
    assert OPS_ROOT.is_dir(), f"provenance ops/ missing at {OPS_ROOT}"
    assert KNOWLEDGE_ROOT.is_dir(), f"knowledge plugin skills/ missing at {KNOWLEDGE_ROOT}"


def test_included_skills_is_the_literal_tier_union():
    """The generic repository validator reads INCLUDED_SKILLS without shell expansion."""
    tiered = (
        _whitelist("LOCAL_SKILLS")
        + _whitelist("SHARED_SKILLS")
        + _whitelist("KNOWLEDGE_SKILLS")
    )
    assert _whitelist("INCLUDED_SKILLS") == tiered


@pytest.mark.parametrize(
    "name",
    ["ops-precision-standard", "ascendc-docs-search", "ascendc-simt-best-practices"],
)
def test_shared_runtime_skill_has_one_canonical_copy(name):
    """Reusable Skills are dependency-owned; the plugin must not vendor snapshots."""
    source = OPS_ROOT / name
    assert (source / "SKILL.md").is_file(), f"canonical shared Skill missing: {source}"
    assert name in _whitelist("SHARED_SKILLS")
    assert not (LOCAL_ROOT / name).exists(), f"duplicate plugin snapshot remains: {name}"


def test_no_devops_skill_in_customer_whitelist():
    """THE security property: no `scope: dev-ops` skill may sit in INCLUDED_SKILLS.

    A source re-sync must not silently add a dev-ops skill to this customer product.
    """
    leaked = []
    for name in _whitelist("INCLUDED_SKILLS"):
        if _scope_of(_skill_md(name)) == "dev-ops":
            leaked.append(name)
    assert not leaked, (
        f"dev-ops skill(s) shipped to customers via INCLUDED_SKILLS: {leaked}. "
        "Remove them from init.sh, or (if cannbot genuinely dispatches them at customer "
        "runtime) reconcile the scope tag with a5_ops."
    )


def test_whitelisted_skills_exist_in_their_owned_roots():
    """Every customer skill must resolve from its declared ownership tier."""
    missing = [n for n in _whitelist("INCLUDED_SKILLS") if not _skill_md(n).is_file()]
    assert not missing, f"whitelisted skills absent from their owned roots: {missing}"


def test_conventional_skill_directory_is_exactly_the_customer_local_set():
    """Claude auto-discovers skills/*; an allowlist cannot hide an extra directory."""
    discovered = sorted(
        path.parent.name for path in LOCAL_ROOT.glob("*/SKILL.md") if path.is_file()
    )
    assert discovered == sorted(_whitelist("LOCAL_SKILLS"))
    leaked = [name for name in discovered if _scope_of(_skill_md(name)) == "dev-ops"]
    assert not leaked, f"dev-ops skill(s) auto-discovered from conventional skills/: {leaked}"


def test_report_gen_remains_a_customer_capability():
    """Protect the pre-force-push customer report feature and its local scope."""
    report = LOCAL_ROOT / "aog-report-gen" / "SKILL.md"
    assert report.is_file(), "customer aog-report-gen skill was unexpectedly removed"
    assert _scope_of(report) != "dev-ops"
    assert "aog-report-gen" in _whitelist("LOCAL_SKILLS")
    assert "aog-report-gen" in _whitelist("INCLUDED_SKILLS")
    assert not (LOCAL_ROOT / "knowledge-query").exists()
    assert not (PLUGIN_ROOT / "maintainer").exists()


def test_report_gen_agent_is_in_the_customer_closure():
    """The restored report Skill and its isolated executor must ship together."""
    import json

    assert "aog-report-gen" in _whitelist("INCLUDED_AGENTS")
    manifest = json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert "./agents/aog-report-gen.md" in manifest.get("agents", [])
    assert (PLUGIN_ROOT / "agents" / "aog-report-gen.md").is_file()


def test_conventional_agent_directory_matches_manifest_and_installer():
    """Prevent an unregistered or unintentionally auto-discovered customer agent."""
    import json

    discovered = sorted(path.stem for path in (PLUGIN_ROOT / "agents").glob("*.md"))
    manifest = json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())
    registered = sorted(
        Path(path).stem for path in manifest.get("agents", [])
        if path.startswith("./agents/")
    )
    assert discovered == registered == sorted(_whitelist("INCLUDED_AGENTS"))


@pytest.mark.parametrize(
    "name",
    ["aog-version-bump", "aog-roadmap-maintain", "aog-release-branch",
     "aog-session-retrospective", "architecture-review", "docs-maintenance",
     "aog-prompt-evolve", "aog-regression-check"],
)
def test_known_devops_skills_never_whitelisted(name):
    """a5_ops' dev-ops set (#240) must never appear in the cannbot customer whitelist."""
    assert name not in _whitelist("INCLUDED_SKILLS"), (
        f"{name} is an a5_ops dev-ops skill and must not ship to cannbot customers"
    )
