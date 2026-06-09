#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# dependency_validator.py - Dependency Graph Validator for CANN Skills
# =============================================================================
# Validates cross-references between marketplace.json, plugin.json, AGENTS.md,
# and agent .md files.
#
# Usage:
#   python3 dependency_validator.py <repo_root>
#
# Output: JSONL (one JSON object per finding), followed by a summary record.
# Exit code: 0 if no error-level findings, 1 otherwise.
#
# Rules:
#   DG-01: marketplace.json skills paths exist
#   DG-02: marketplace.json dependencies valid
#   DG-03: plugin.json agents paths exist
#   DG-04: plugin.json dependencies valid
#   DG-05: AGENTS.md skills references exist
#   DG-06: Agent .md skills references exist
#   DG-07: Orphaned skills detection (warn)
#   DG-08: Orphaned agents detection (warn)
#   DG-09: Circular dependency detection
#   DG-10: init.sh INCLUDED_SKILLS covers all marketplace-declared skills
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", ".opencode", ".claude", ".claude-plugin", "asc-devkit"}


def _is_inside_nested_git_repo(file_path: Path, repo_root: Path) -> bool:
    """Check if file resides inside a nested git repo (not the repo root itself)."""
    current = file_path.parent
    while current != repo_root and current != current.parent:
        if (current / ".git").exists() and current != repo_root:
            return True
        current = current.parent
    return False


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str | None]:
    """Parse YAML frontmatter from a markdown file. Returns (data, error_msg)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {}, f"cannot read {path}: {exc}"

    if not text.startswith("---"):
        return {}, "missing opening '---'"

    lines = text.splitlines(keepends=False)
    close_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx is None:
        return {}, "missing closing '---'"

    fm_text = "\n".join(lines[1:close_idx])
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        return {}, f"invalid YAML frontmatter: {exc}"
    if not isinstance(data, dict):
        return {}, "frontmatter is not a mapping"
    return data, None


class DependencyValidator:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.marketplace_path = self.repo_root / ".claude-plugin" / "marketplace.json"
        self.findings: list[dict[str, Any]] = []

        self.marketplace: dict = {}
        self.skill_packages: dict[str, dict] = {}
        self.development_teams: dict[str, dict] = {}
        self.all_skill_names: set[str] = set()
        self.all_skill_dirs: dict[str, Path] = {}
        self.all_agent_files: dict[str, Path] = {}
        self.referenced_skills: set[str] = set()
        self.referenced_agents: set[str] = set()

    def load_marketplace(self) -> bool:
        """Load and parse marketplace.json, populating skill_packages and development_teams."""
        if not self.marketplace_path.exists():
            self._emit("error", "DG-ENV", f"marketplace.json not found: {self.marketplace_path}")
            return False
        try:
            self.marketplace = json.loads(self.marketplace_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._emit("error", "DG-ENV", f"Failed to parse marketplace.json: {exc}")
            return False

        for plugin in self.marketplace.get("plugins", []):
            name = plugin.get("name", "")
            cat = plugin.get("category", "")
            if cat == "skills":
                self.skill_packages[name] = plugin
            elif cat == "development":
                self.development_teams[name] = plugin
        return True

    def discover_skills(self) -> None:
        """Discover all top-level SKILL.md files in the repository.

        Only considers SKILL.md files that are direct children of recognized
        skill root directories (ops/, graph/, model/, ops-lab/*/skills/).
        Skips nested SKILL.md files inside team directories or sub-skills.
        Also discovers skills that are renamed at install time via init.sh.
        """
        skill_root_dirs = {"ops", "graph", "model", "infra"}

        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d != "operators"]
            root_path = Path(root)
            if _is_inside_nested_git_repo(root_path, self.repo_root):
                dirs.clear()
                continue
            skill_md_files = [f for f in files if f.lower() == "skill.md"]
            if not skill_md_files:
                continue

            skill_dir = root_path
            rel = skill_dir.relative_to(self.repo_root)
            parts = rel.parts

            if len(parts) < 2:
                continue

            top_dir = parts[0]

            if top_dir in ("plugins-official", "plugins-community"):
                continue

            if top_dir in skill_root_dirs:
                if len(parts) != 2:
                    continue
            elif top_dir == "ops-lab":
                if len(parts) < 4 or parts[2] != "skills":
                    continue
            else:
                continue

            skill_name = skill_dir.name
            self.all_skill_names.add(skill_name)
            self.all_skill_dirs[skill_name] = skill_dir

        init_renames = self._discover_init_sh_renames()
        for installed_name, src_path in init_renames.items():
            src = Path(src_path)
            if src.exists() and (src / "SKILL.md").exists():
                self.all_skill_names.add(installed_name)
                self.all_skill_dirs[installed_name] = src

    def discover_agents(self) -> None:
        """Discover all agent .md files (flat layout: agents/<name>.md)."""
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if _is_inside_nested_git_repo(Path(root), self.repo_root):
                dirs.clear()
                continue
            if Path(root).name != "agents":
                continue
            self._collect_agent_files(Path(root), files)

    def validate_marketplace_skills_paths(self) -> None:
        """DG-01: Validate skills paths in marketplace.json exist on disk."""
        for pkg_name, pkg in self.skill_packages.items():
            source = pkg.get("source", "")
            skills_list = pkg.get("skills", [])
            if not source or not skills_list:
                continue
            source_dir = self.repo_root / source.lstrip("./")
            for skill_rel in skills_list:
                skill_dir_name = skill_rel.lstrip("./")
                skill_path = source_dir / skill_dir_name / "SKILL.md"
                if not skill_path.exists():
                    self._emit(
                        "error", "DG-01",
                        f"Skill path does not exist: {source}/{skill_dir_name}/SKILL.md "
                        f"(referenced by package '{pkg_name}')",
                        str(self.marketplace_path.relative_to(self.repo_root)),
                    )

    def validate_marketplace_dependencies(self) -> None:
        """DG-02: Validate team dependencies reference existing skill packages."""
        all_pkg_names = set(self.skill_packages.keys()) | set(self.development_teams.keys())
        for team_name, team in self.development_teams.items():
            deps = team.get("dependencies", [])
            for dep in deps:
                if dep not in all_pkg_names:
                    self._emit(
                        "error", "DG-02",
                        f"Dependency '{dep}' not found in marketplace.json "
                        f"(referenced by team '{team_name}')",
                        str(self.marketplace_path.relative_to(self.repo_root)),
                    )

    def validate_plugin_agents_paths(self) -> None:
        """DG-03: Validate agent paths in plugin.json files exist on disk."""
        for plugin_json, plugin in self._discover_plugin_json_files():
            team_dir = plugin_json.parent.parent
            plugin_name = plugin.get("name", team_dir.name)
            agents_list = plugin.get("agents", [])
            for agent_rel in agents_list:
                agent_path = team_dir / agent_rel.lstrip("./")
                if not agent_path.exists():
                    self._emit(
                        "error", "DG-03",
                        f"Agent path does not exist: {agent_rel} "
                        f"(referenced by plugin '{plugin_name}')",
                        str(plugin_json.relative_to(self.repo_root)),
                    )

    def validate_plugin_dependencies(self) -> None:
        """DG-04: Validate dependencies in plugin.json reference existing packages."""
        all_pkg_names = set(self.skill_packages.keys()) | set(self.development_teams.keys())
        for plugin_json, plugin in self._discover_plugin_json_files():
            plugin_name = plugin.get("name", plugin_json.parent.parent.name)
            deps = plugin.get("dependencies", [])
            for dep in deps:
                if dep not in all_pkg_names:
                    self._emit(
                        "error", "DG-04",
                        f"Dependency '{dep}' not found in marketplace.json "
                        f"(referenced by plugin.json of '{plugin_name}')",
                        str(plugin_json.relative_to(self.repo_root)),
                    )

    def validate_agents_md_skills(self) -> None:
        """DG-05: Validate skills references in AGENTS.md frontmatter exist."""
        agents_md_files = self._find_agents_md_files()

        for agents_md in agents_md_files:
            fm, err = _parse_frontmatter(agents_md)
            if err:
                continue
            skills_list = fm.get("skills", [])
            if not isinstance(skills_list, list):
                continue
            team_name = agents_md.parent.name
            for skill_name in skills_list:
                if not isinstance(skill_name, str):
                    continue
                self.referenced_skills.add(skill_name)
                if skill_name not in self.all_skill_names:
                    self._emit(
                        "error", "DG-05",
                        f"Skill '{skill_name}' not found "
                        f"(referenced in AGENTS.md of '{team_name}')",
                        str(agents_md.relative_to(self.repo_root)),
                    )

    def validate_agent_md_skills(self) -> None:
        """DG-06: Validate skills references in agent .md frontmatter exist."""
        for agent_name, agent_path in self.all_agent_files.items():
            fm, err = _parse_frontmatter(agent_path)
            if err:
                continue
            skills_list = fm.get("skills", [])
            if not isinstance(skills_list, list):
                continue
            for skill_name in skills_list:
                if not isinstance(skill_name, str):
                    continue
                self.referenced_skills.add(skill_name)
                if skill_name not in self.all_skill_names:
                    self._emit(
                        "error", "DG-06",
                        f"Skill '{skill_name}' not found "
                        f"(referenced in agent '{agent_name}')",
                        str(agent_path.relative_to(self.repo_root)),
                    )

    def detect_orphaned_skills(self) -> None:
        """DG-07: Detect skills not referenced by any agent, team, or marketplace package."""
        marketplace_skills = self._collect_marketplace_skill_names()
        plugin_json_agents_skills = self._collect_plugin_agent_skills()
        all_referenced = self.referenced_skills | marketplace_skills | plugin_json_agents_skills

        for skill_name in sorted(self.all_skill_names):
            if skill_name not in all_referenced:
                self._emit(
                    "warn", "DG-07",
                    f"Skill '{skill_name}' is not referenced by any agent, team, or marketplace package",
                    str(self.all_skill_dirs[skill_name].relative_to(self.repo_root)),
                )

    def detect_orphaned_agents(self) -> None:
        """DG-08: Detect agents not referenced by any plugin.json."""
        plugin_referenced_agents = self._collect_plugin_referenced_agents()

        for agent_name in sorted(self.all_agent_files.keys()):
            if agent_name not in plugin_referenced_agents:
                self._emit(
                    "warn", "DG-08",
                    f"Agent '{agent_name}' is not referenced by any plugin.json",
                    str(self.all_agent_files[agent_name].relative_to(self.repo_root)),
                )

    def detect_circular_dependencies(self) -> None:
        """DG-09: Detect circular dependency chains in marketplace.json."""
        graph: dict[str, list[str]] = {}
        for team_name, team in self.development_teams.items():
            graph[team_name] = team.get("dependencies", [])
        for pkg_name in self.skill_packages:
            graph.setdefault(pkg_name, [])

        cycles = self._find_cycles(graph)
        for cycle in cycles:
            self._emit(
                "error", "DG-09",
                f"Circular dependency detected: {cycle}",
                str(self.marketplace_path.relative_to(self.repo_root)),
            )

    def validate_init_sh_skills(self) -> None:
        """DG-10: Validate init.sh INCLUDED_SKILLS covers all marketplace-declared skills."""
        for team_name, team in self.development_teams.items():
            source = team.get("source", "")
            if not source:
                continue
            team_dir = self.repo_root / source.lstrip("./")
            if not team_dir.is_dir():
                continue

            # Collect all skills from dependency packages
            marketplace_skills: set[str] = set()
            for dep_name in team.get("dependencies", []):
                pkg = self.skill_packages.get(dep_name)
                if not pkg:
                    continue
                for skill_rel in pkg.get("skills", []):
                    skill_name = skill_rel.lstrip("./").split("/")[-1]
                    marketplace_skills.add(skill_name)

            if not marketplace_skills:
                continue

            # Extract init.sh skills
            init_vars = self._extract_init_sh_vars(team_dir)
            if init_vars is None:
                continue  # No installer script, skip

            init_skills = init_vars["skills"]
            if not init_skills:
                continue  # No INCLUDED_SKILLS defined, skip

            # Check for missing skills
            missing = marketplace_skills - init_skills
            for skill_name in sorted(missing):
                self._emit(
                    "error", "DG-10",
                    f"Skill '{skill_name}' declared in marketplace.json but missing from "
                    f"{init_vars['script_name']} INCLUDED_SKILLS (team '{team_name}')",
                    str((team_dir / init_vars["script_name"]).relative_to(self.repo_root)),
                )

    def run_all_validations(self) -> int:
        """Run all dependency graph validations and output results."""
        if not self.load_marketplace():
            self._output_findings()
            return 1

        self.discover_skills()
        self.discover_agents()

        self.validate_marketplace_skills_paths()
        self.validate_marketplace_dependencies()
        self.validate_plugin_agents_paths()
        self.validate_plugin_dependencies()
        self.validate_agents_md_skills()
        self.validate_agent_md_skills()
        self.detect_orphaned_skills()
        self.detect_orphaned_agents()
        self.detect_circular_dependencies()
        self.validate_init_sh_skills()

        self._output_findings()
        error_count = sum(1 for f in self.findings if f["level"] == "error")
        return 1 if error_count > 0 else 0

    def _collect_agent_files(self, agents_dir: Path, files: list[str]) -> None:
        """Extract agent .md files from an agents directory."""
        for f in files:
            if f.endswith(".md") and f != "AGENTS.md":
                agent_name = f[:-3]
                agent_path = agents_dir / f
                self.all_agent_files[agent_name] = agent_path

    def _emit(self, level: str, rule: str, msg: str, file: str = "") -> None:
        """Append a finding to the findings list."""
        self.findings.append({"level": level, "rule": rule, "msg": msg, "file": file})

    def _output_findings(self) -> None:
        """Output all findings as JSONL to stdout, followed by a summary."""
        for f in self.findings:
            sys.stdout.write(json.dumps(f, ensure_ascii=False) + "\n")

        error_count = sum(1 for f in self.findings if f["level"] == "error")
        warn_count = sum(1 for f in self.findings if f["level"] == "warn")
        summary = json.dumps({
            "summary": {
                "errors": error_count,
                "warnings": warn_count,
                "total_skills": len(self.all_skill_names),
                "total_agents": len(self.all_agent_files),
                "skill_packages": len(self.skill_packages),
                "development_teams": len(self.development_teams),
            }
        }, ensure_ascii=False)
        sys.stdout.write(summary + "\n")

    @staticmethod
    def _extract_init_sh_vars(team_dir: Path) -> dict | None:
        """Extract INCLUDED_SKILLS and VERSION from init.sh or install.sh.

        Returns dict with keys: 'skills' (set[str]), 'version' (str), 'script_name' (str).
        Returns None if no installer script found.
        """
        for script_name in ("init.sh", "install.sh"):
            script = team_dir / script_name
            if script.exists():
                try:
                    content = script.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                skills: set[str] = set()
                m = re.search(r'INCLUDED_SKILLS="([^"]*)"', content)
                if m:
                    skills = set(m.group(1).split())

                version = ""
                m = re.search(r'VERSION="([^"]*)"', content)
                if m:
                    version = m.group(1)

                return {
                    "skills": skills,
                    "version": version,
                    "script_name": script_name,
                }
        return None

    def _discover_init_sh_renames(self) -> dict[str, str]:
        """Scan init.sh files for symlink rename mappings.

        Parses patterns like:
            ln -sfn "$(realpath "$SCRIPT_DIR/workflow")" "$CONFIG_ROOT/skills/ops-registry-invoke-workflow"

        Returns dict mapping installed_name -> source_dir_relative_to_team.
        """
        renames: dict[str, str] = {}
        ln_pattern = re.compile(
            r'ln\s+-sfn\s+.*?SCRIPT_DIR/([\w-]+).*?skills/([\w-]+)'
        )
        for root, dirs, _ in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if _is_inside_nested_git_repo(Path(root), self.repo_root):
                dirs.clear()
                continue
            init_sh = Path(root) / "init.sh"
            if not init_sh.exists():
                continue
            try:
                content = init_sh.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in ln_pattern.finditer(content):
                src_dir = m.group(1)
                installed_name = m.group(2)
                renames[installed_name] = str(Path(root) / src_dir)
        return renames

    def _discover_plugin_json_files(self) -> list[tuple[Path, dict]]:
        """Discover all plugin.json files in the repository.

        Returns list of (plugin_json_path, parsed_data) tuples.
        """
        results: list[tuple[Path, dict]] = []
        for root, dirs, _ in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if _is_inside_nested_git_repo(Path(root), self.repo_root):
                dirs.clear()
                continue
            plugin_json = Path(root) / ".claude-plugin" / "plugin.json"
            if not plugin_json.exists():
                continue
            try:
                plugin = json.loads(plugin_json.read_text(encoding="utf-8"))
                results.append((plugin_json, plugin))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def _collect_marketplace_skill_names(self) -> set[str]:
        """Collect all skill names declared in marketplace skill packages."""
        names: set[str] = set()
        for pkg in self.skill_packages.values():
            source = pkg.get("source", "")
            for skill_rel in pkg.get("skills", []):
                names.add(skill_rel.lstrip("./").split("/")[-1])
        return names

    def _collect_plugin_agent_skills(self) -> set[str]:
        """Collect skill names referenced by agents in plugin.json files."""
        plugin_json_agents_skills: set[str] = set()
        for _, team in self.development_teams.items():
            source = team.get("source", "")
            if not source:
                continue
            team_dir = self.repo_root / source.lstrip("./")
            plugin_json = team_dir / ".claude-plugin" / "plugin.json"
            if not plugin_json.exists():
                continue
            try:
                plugin = json.loads(plugin_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for agent_rel in plugin.get("agents", []):
                agent_path = team_dir / agent_rel.lstrip("./")
                if agent_path.exists():
                    fm, _ = _parse_frontmatter(agent_path)
                    for s in fm.get("skills", []):
                        if isinstance(s, str):
                            plugin_json_agents_skills.add(s)
        return plugin_json_agents_skills

    def _collect_plugin_referenced_agents(self) -> set[str]:
        """Collect agent names referenced by plugin.json files."""
        plugin_referenced_agents: set[str] = set()
        for plugin_json, plugin in self._discover_plugin_json_files():
            for agent_rel in plugin.get("agents", []):
                agent_name = Path(agent_rel).stem
                plugin_referenced_agents.add(agent_name)
        return plugin_referenced_agents

    def _find_agents_md_files(self) -> list[Path]:
        """Find all AGENTS.md files in the repository."""
        agents_md_files: list[Path] = []
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d != "operators"]
            if _is_inside_nested_git_repo(Path(root), self.repo_root):
                dirs.clear()
                continue
            if "AGENTS.md" in files:
                agents_md_files.append(Path(root) / "AGENTS.md")
        return agents_md_files

    def _find_cycles(self, graph: dict[str, list[str]]) -> list[str]:
        """Find all circular dependency cycles in a dependency graph using DFS."""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[str] = []

        def _dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for dep in graph.get(node, []):
                if dep not in visited:
                    _dfs(dep, path)
                elif dep in rec_stack:
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    cycles.append(" -> ".join(cycle))
            path.pop()
            rec_stack.discard(node)

        for node in graph:
            if node not in visited:
                _dfs(node, [])
        return cycles


def main() -> int:
    """Entry point for dependency_validator.py."""
    if yaml is None:
        logging.error("PyYAML is required. Install with: pip install pyyaml")
        return 2

    if len(sys.argv) < 2:
        logging.error("Usage: dependency_validator.py <repo_root>")
        return 2

    repo_root = sys.argv[1]
    validator = DependencyValidator(repo_root)
    return validator.run_all_validations()


if __name__ == "__main__":
    sys.exit(main())