#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# skill_validator.py - YAML-aware validator for CANN Skills tests
# =============================================================================
# Emits one JSON object per finding on stdout (JSON Lines), followed by a
# {"summary": {...}} record. Shell drivers parse these and translate them
# into print_fail / print_error / print_warn calls.
#
# Subcommands:
#   parse <file>                                — dump frontmatter as JSON
#   validate-skill <file>                       — S-STR/S-CON rules for skills
#   validate-agent <file>                       — A-STR/A-CON rules for agents
#   validate-team <file>                        — T-STR/T-CON rules for teams
#   check-uniqueness <kind> <path...>           — cross-file name-uniqueness check
#
# Exit code is 0 unless the arguments are malformed. Callers must inspect the
# "level" field on each JSON line to decide pass/fail.
# =============================================================================

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # PyYAML 6.x
except ImportError as exc:  # pragma: no cover
    print(json.dumps({
        "level": "error",
        "rule": "ENV",
        "file": "",
        "msg": f"PyYAML is required: {exc}",
    }))
    sys.exit(2)


RULES_PATH = Path(__file__).with_name("rules.yaml")


# ---------------------------------------------------------------------------
# Rules loading
# ---------------------------------------------------------------------------

def load_rules() -> dict[str, Any]:
    with RULES_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


RULES = load_rules()

SKILL_KEYWORDS_RE = re.compile("|".join(RULES.get("skill_keywords", [])), re.IGNORECASE)
TEAM_KEYWORDS_RE = re.compile("|".join(RULES.get("team_keywords", [])), re.IGNORECASE)
TRIGGER_RE = re.compile(
    "|".join(RULES.get("trigger_condition_phrases", [])),
    re.IGNORECASE,
)
RESERVED_PREFIXES: list[str] = [p.lower() for p in RULES.get("reserved_name_prefixes", [])]
ANTI_PATTERN_RE = re.compile(
    "|".join(RULES.get("anti_pattern_phrases", [])),
    re.IGNORECASE,
) if RULES.get("anti_pattern_phrases") else None

NAME_FORMAT_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

MAX_NAME = int(RULES.get("skill_name_max", 64))
MAX_DESC = int(RULES.get("skill_description_max", 1024))
MAX_COMPAT = int(RULES.get("skill_compatibility_max", 500))
MAX_WORDS = int(RULES.get("skill_max_words", 5000))
LONG_LINE_THRESHOLD = int(RULES.get("skill_long_line_threshold", 200))
DESC_COMBINED_MAX = int(RULES.get("skill_description_combined_max", 1536))

ACTION_VERB_RE = re.compile(
    r"[A-Za-z\u4e00-\u9fff]+?("
    r"提供|Provides?|处理|Handles?|负责|生成|支持|Suppors?|"
    r"Designs?|实现|Implements?|设计|Optimizes?|优化|"
    r"创建|Creates?|分析|Analyzes?|验证|Validates?|"
    r"检查|Checks?|提取|Extracts?|定位|Locates?|"
    r"修复|Fixes?|调试|Debugs?|测试|Tests?|"
    r"管理|Manages?|配置|Configures?|部署|Deploys?|"
    r"重构|Refactors?|迁移|Migrates?|集成|Integrates?|"
    r"扩展|Extends?|封装|Wraps?|转换|Converts?)",
)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(path: Path) -> tuple[dict[str, Any], str, str | None]:
    """Return (frontmatter_dict, body, error_msg_or_none).

    Reads the full file. Frontmatter is delimited by a leading '---' line and
    a following '---' line (no arbitrary depth limit). If parsing fails, the
    tuple's third element carries a message.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, "", f"cannot read {path}: {exc}"

    if not text.startswith("---"):
        return {}, text, "missing opening '---'"

    # Locate closing '---'
    lines = text.splitlines(keepends=False)
    # Skip the first '---'
    close_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx is None:
        return {}, text, "missing closing '---'"

    fm_text = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1:])
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        return {}, body, f"invalid YAML frontmatter: {exc}"
    if not isinstance(data, dict):
        return {}, body, "frontmatter is not a mapping"
    return data, body, None


def raw_frontmatter_text(path: Path) -> str:
    """Return just the between-fences text (no YAML parse). Used for XML-tag scan."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    lines = text.splitlines(keepends=False)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i])
    return ""


# ---------------------------------------------------------------------------
# Body-empty guard (shared across skill/agent/team validators)
# ---------------------------------------------------------------------------


def _check_body_not_empty(rule_prefix: str, file_str: str, path: Path, body: str) -> bool:
    """Return True if body is empty and an error was emitted.

    An empty body usually means the closing '---' was placed after markdown
    content, causing it to be absorbed into the YAML frontmatter (e.g. '#
    headings' become YAML comments and are silently ignored).
    """
    if body.strip():
        return False

    raw = raw_frontmatter_text(path)
    detail = ""
    if raw:
        md_lines: list[str] = []
        for lineno, line in enumerate(raw.splitlines(), start=2):
            if line.startswith("#") or line.startswith("|"):
                md_lines.append(f"line {lineno}: {line.strip()}")
        if md_lines:
            joined = "; ".join(md_lines[:3])
            detail = (
                f" Markdown content was absorbed into frontmatter as YAML "
                f"comments ({joined}). The closing '---' must come BEFORE "
                f"any markdown headings, not after them."
            )
    emit("error", f"{rule_prefix}-STR-01", file_str,
         f"File body is empty (no content after closing '---').{detail}")
    return True


def _check_description_triggers(rule_prefix: str, desc: str, is_manual: bool,
                                file_str: str, mode: str | None = None) -> None:
    """Check description for trigger keywords, conditions, and anti-patterns.

    Validates S/A-CON-02 (keywords), S/A-CON-03 (trigger conditions),
    and S/A-CON-05 (anti-patterns).
    A-CON-03 is skipped for agents with mode=subagent (they are invoked by
    parent agents, not by user input).
    """
    kw_re = SKILL_KEYWORDS_RE  # agents use the same keyword set as skills

    # S/A-CON-02/03: skipped for manually-triggered skills
    if not is_manual:
        if not kw_re.search(desc):
            emit("error", f"{rule_prefix}-CON-02", file_str, "Description lacks trigger keywords")
        # A-CON-03: skip for subagents (invoked by parent agent, not user input)
        if rule_prefix == "A" and mode == "subagent":
            pass
        elif not TRIGGER_RE.search(desc):
            msg = "Description missing trigger conditions (e.g. 'Use when...' or '触发：...')"
            emit("error" if rule_prefix == "S" else "warn", f"{rule_prefix}-CON-03", file_str, msg)

    # S/A-CON-05: anti-pattern phrases
    if ANTI_PATTERN_RE and ANTI_PATTERN_RE.search(desc):
        emit("warn", f"{rule_prefix}-CON-05", file_str,
             "Description uses vague/lazy anti-pattern phrasing "
             "(see rules.yaml anti_pattern_phrases)")


def _check_content_quality(rule_prefix: str, file_str: str, text_all: str) -> None:
    """Check file body for progressive disclosure and file reference depth.

    Validates S/A-CON-04 (progressive disclosure) and S/A-CON-06 (file reference depth).
    """
    # S/A-CON-04: progressive disclosure (long files should link to supporting files)
    line_count = text_all.count("\n") + 1
    support_re = (
        r"references/|scripts/|assets/"
        r"|reference\.md|examples\.md|REFERENCE\.md|EXAMPLES\.md"
        r"|\{file:"
    )
    if line_count > LONG_LINE_THRESHOLD and not re.search(support_re, text_all):
        label = "SKILL.md" if rule_prefix == "S" else "Agent file"
        emit("warn", f"{rule_prefix}-CON-04", file_str,
             f"{label} is {line_count} lines but no supporting files linked "
             f"(use references/, scripts/, assets/, or separate .md files for progressive disclosure)")

    # S/A-CON-06: file reference depth (Agent Skills spec: keep references one level deep)
    nested_refs = re.findall(
        r"\]\((?:references/|scripts/|assets/)[^)]*/[^)]*/[^)]*\)", text_all
    )
    if nested_refs:
        emit("warn", f"{rule_prefix}-CON-06", file_str,
             f"Found {len(nested_refs)} deeply nested file reference(s) "
             f"(Agent Skills spec: keep file references one level deep from SKILL.md)")


# ---------------------------------------------------------------------------
# Finding emission
# ---------------------------------------------------------------------------

def emit(level: str, rule: str, file: str, msg: str) -> None:
    print(json.dumps({"level": level, "rule": rule, "file": file, "msg": msg}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Common field checks
# ---------------------------------------------------------------------------

def _name_checks(rule_prefix: str, name: Any, file_str: str) -> None:
    if not isinstance(name, str) or not name:
        emit("error", f"{rule_prefix}-02", file_str, "Missing 'name' field (or empty)")
        return
    if not (1 <= len(name) <= MAX_NAME):
        emit("error", f"{rule_prefix}-05", file_str,
             f"name length must be 1-{MAX_NAME} chars (got {len(name)})")
    if not NAME_FORMAT_RE.match(name):
        emit("error", f"{rule_prefix}-06", file_str,
             f"name must match ^[a-z0-9]+(-[a-z0-9]+)*$ (got '{name}')")
    # Reserved prefixes
    lname = name.lower()
    for prefix in RESERVED_PREFIXES:
        if lname.startswith(prefix):
            emit("warn", f"{rule_prefix}-14", file_str,
                 f"name uses reserved prefix '{prefix}*' (name='{name}')")
            break


def _description_checks(rule_prefix: str, desc: Any, file_str: str) -> str | None:
    if not isinstance(desc, str) or not desc:
        emit("error", f"{rule_prefix}-03", file_str, "Missing 'description' field (or empty)")
        return None
    if not (1 <= len(desc) <= MAX_DESC):
        emit("error", f"{rule_prefix}-07", file_str,
             f"description length must be 1-{MAX_DESC} chars (got {len(desc)})")
    return desc


def _check_compatibility(compat: Any, file_str: str) -> None:
    """Validate S-STR-07 compatibility field."""
    if isinstance(compat, str):
        if not (1 <= len(compat) <= MAX_COMPAT):
            emit("error", "S-STR-07", file_str,
                 f"compatibility length must be 1-{MAX_COMPAT} chars (got {len(compat)})")
    else:
        emit("error", "S-STR-07", file_str,
             f"compatibility must be a string (got {type(compat).__name__})")


def _check_metadata(metadata: Any, file_str: str) -> None:
    """Validate S-STR-16 metadata string->string mapping."""
    if not isinstance(metadata, dict):
        emit("error", "S-STR-16", file_str,
             f"metadata must be a mapping (got {type(metadata).__name__})")
    else:
        for k, v in metadata.items():
            if not isinstance(k, str) or not isinstance(v, str):
                emit("error", "S-STR-16", file_str,
                     f"metadata entries must be string->string (got '{k}': {type(v).__name__})")


def _validate_skill_structure(fm: dict, skill_file: Path, skill_dir: Path,
                              skill_name: str, file_str: str) -> None:
    """Run all S-STR-* structure checks. Emits findings via emit()."""
    _name_checks("S-STR", fm.get("name"), file_str)
    _description_checks("S-STR", fm.get("description"), file_str)

    # S-STR-07 extension: compatibility length
    compat = fm.get("compatibility")
    if compat is not None:
        _check_compatibility(compat, file_str)

    # S-STR-16: metadata must be string->string
    metadata = fm.get("metadata")
    if metadata is not None:
        _check_metadata(metadata, file_str)

    # S-STR-09: exact filename
    if skill_file.name != "SKILL.md":
        emit("error", "S-STR-09", file_str,
             f"File must be named 'SKILL.md' exactly (got '{skill_file.name}')")

    # S-STR-10: kebab-case dir name
    if not NAME_FORMAT_RE.match(skill_name):
        emit("error", "S-STR-10", file_str,
             f"Directory name must be kebab-case (got '{skill_name}')")

    # S-STR-11: no README.md inside skill dir
    if (skill_dir / "README.md").exists():
        emit("warn", "S-STR-11", file_str,
             "README.md not allowed inside skill directory (put content in SKILL.md or references/)")

    # S-STR-04: references dir must contain at least one .md file
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        has_md = any(p.suffix == ".md" for p in ref_dir.rglob("*.md"))
        if not has_md:
            emit("error", "S-STR-04", file_str, "references/ directory contains no .md files")

    # S-STR-12: XML tag injection in raw frontmatter
    raw = raw_frontmatter_text(skill_file)
    if raw and re.search(r"<[a-zA-Z_][a-zA-Z0-9_-]*>|</[a-zA-Z_][a-zA-Z0-9_-]*>", raw):
        emit("error", "S-STR-12", file_str,
             "Frontmatter contains XML tag pattern (security restriction)")

    # S-STR-17: description + when_to_use combined length (Claude Code truncates at 1536)
    when_to_use = fm.get("when_to_use")
    if when_to_use is not None:
        if not isinstance(when_to_use, str):
            emit("error", "S-STR-17", file_str,
                 f"when_to_use must be a string (got {type(when_to_use).__name__})")
        else:
            desc = fm.get("description", "")
            combined = len(desc) + len(when_to_use)
            if combined > DESC_COMBINED_MAX:
                emit("warn", "S-STR-17", file_str,
                     f"description + when_to_use combined is {combined} chars "
                     f"(Claude Code truncates at {DESC_COMBINED_MAX})")

    # S-STR-18: disable-model-invocation must be boolean
    dmi = fm.get("disable-model-invocation")
    if dmi is not None and not isinstance(dmi, bool):
        emit("error", "S-STR-18", file_str,
             f"disable-model-invocation must be boolean (got {type(dmi).__name__})")


def _validate_skill_content(fm: dict, body: str, skill_file: Path, skill_name: str, file_str: str) -> None:
    """Run all S-CON-* content checks. Emits findings via emit()."""
    # S-CON-01: name matches directory
    if fm.get("name") and fm["name"] != skill_name:
        emit("error", "S-CON-01", file_str,
             f"name '{fm['name']}' != directory '{skill_name}'")

    desc = fm.get("description") if isinstance(fm.get("description"), str) else None
    is_manual = fm.get("disable-model-invocation") is True
    if desc:
        _check_description_triggers("S", desc, is_manual, file_str)

    text_all = skill_file.read_text(encoding="utf-8", errors="replace")
    _check_content_quality("S", file_str, text_all)


# ---------------------------------------------------------------------------
# Skill validator
# ---------------------------------------------------------------------------

def validate_skill(skill_file: Path, subset: str = "all") -> None:
    """subset ∈ {all, structure, content}. Structure covers S-STR-*, content
    covers S-CON-*. 'all' emits both."""
    file_str = str(skill_file)
    skill_dir = skill_file.parent
    skill_name = skill_dir.name

    fm, body, err = parse_frontmatter(skill_file)
    if err:
        emit("error", "S-STR-01", file_str, err)
        return

    # S-STR-01 extension: body must not be empty.
    if _check_body_not_empty("S", file_str, skill_file, body):
        return

    do_structure = subset in ("all", "structure")
    do_content = subset in ("all", "content")

    if do_structure:
        _validate_skill_structure(fm, skill_file, skill_dir, skill_name, file_str)
    if do_content:
        _validate_skill_content(fm, body, skill_file, skill_name, file_str)


# ---------------------------------------------------------------------------
# Agent validator
# ---------------------------------------------------------------------------

def _extract_skills_list(fm: dict[str, Any]) -> list[str]:
    raw = fm.get("skills")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if isinstance(x, (str, int))]
    return []


def _validate_agent_structure(fm: dict, agent_file: Path, file_str: str, known_skill_names: set[str] | None) -> None:
    """Run all A-STR-* structure checks. Emits findings via emit()."""
    _name_checks("A-STR", fm.get("name"), file_str)
    _description_checks("A-STR", fm.get("description"), file_str)

    # A-STR-02/03: mode field
    mode = fm.get("mode")
    if mode is None:
        emit("error", "A-STR-02", file_str, "Missing 'mode' field")
    elif mode not in ("primary", "subagent", "all"):
        emit("error", "A-STR-03", file_str,
             f"Invalid mode '{mode}' (must be: primary, subagent, or all)")

    # A-STR-04: skills deps exist
    if known_skill_names is not None:
        for sk in _extract_skills_list(fm):
            if sk not in known_skill_names:
                emit("error", "A-STR-04", file_str, f"Missing skill dependency: {sk}")


def _validate_agent_content(fm: dict, agent_file: Path, agent_name: str, file_str: str) -> None:
    """Run all A-CON-* content checks. Emits findings via emit()."""
    if fm.get("name") and fm["name"] != agent_name:
        emit("error", "A-CON-01", file_str,
             f"name '{fm['name']}' != directory/file '{agent_name}'")

    desc = fm.get("description") if isinstance(fm.get("description"), str) else None
    is_manual = fm.get("disable-model-invocation") is True
    if desc:
        _check_description_triggers("A", desc, is_manual, file_str, mode=fm.get("mode"))

    text_all = agent_file.read_text(encoding="utf-8", errors="replace")
    _check_content_quality("A", file_str, text_all)


def validate_agent(agent_file: Path, known_skill_names: set[str] | None = None, subset: str = "all") -> None:
    file_str = str(agent_file)
    agent_name = agent_file.stem

    fm, _body, err = parse_frontmatter(agent_file)
    if err:
        emit("error", "A-STR-01", file_str, err)
        return

    if _check_body_not_empty("A", file_str, agent_file, _body):
        return

    do_structure = subset in ("all", "structure")
    do_content = subset in ("all", "content")

    if do_structure:
        _validate_agent_structure(fm, agent_file, file_str, known_skill_names)
    if do_content:
        _validate_agent_content(fm, agent_file, agent_name, file_str)


# ---------------------------------------------------------------------------
# Team validator
# ---------------------------------------------------------------------------

def _check_team_skill_deps(fm: dict, file_str: str, known_skill_names: set[str]) -> None:
    """Validate T-STR-04: team skill dependencies exist."""
    for sk in _extract_skills_list(fm):
        if sk not in known_skill_names:
            emit("error", "T-STR-04", file_str, f"Missing skill dependency: {sk}")


def _check_team_refs(team_file: Path, file_str: str) -> None:
    """Validate T-STR-05: references directory has .md files."""
    ref_dir = team_file.parent / "references"
    if ref_dir.is_dir():
        has_md = any(p.suffix == ".md" for p in ref_dir.rglob("*.md"))
        if not has_md:
            emit("error", "T-STR-05", file_str, "Empty references directory")


def _validate_team_structure(fm: dict, team_file: Path, file_str: str,
                             known_skill_names: set[str] | None) -> None:
    """Run all T-STR-* structure checks. Emits findings via emit()."""
    # Teams in this repo don't require 'name' field, but description is required.
    _description_checks("T-STR", fm.get("description"), file_str)

    mode = fm.get("mode")
    if mode is None:
        emit("error", "T-STR-02", file_str, "Missing 'mode' field")
    elif mode not in ("primary", "all"):
        emit("error", "T-STR-02", file_str, f"Invalid mode '{mode}' (must be: primary or all)")

    if "skills" not in fm:
        emit("error", "T-STR-03", file_str, "Missing 'skills' field")
    elif known_skill_names is not None:
        _check_team_skill_deps(fm, file_str, known_skill_names)

    _check_team_refs(team_file, file_str)


def _validate_team_content(fm: dict, team_name: str, file_str: str) -> None:
    """Run all T-CON-* content checks. Emits findings via emit()."""
    if not NAME_FORMAT_RE.match(team_name):
        emit("error", "T-CON-01", file_str,
             f"Directory name must match ^[a-z0-9]+(-[a-z0-9]+)*$ (got '{team_name}')")

    desc = fm.get("description") if isinstance(fm.get("description"), str) else None
    is_manual = fm.get("disable-model-invocation") is True
    if desc and not is_manual:
        if not TEAM_KEYWORDS_RE.search(desc):
            emit("error", "T-CON-02", file_str, "Description lacks trigger keywords")
        if not TRIGGER_RE.search(desc):
            emit("warn", "T-CON-03", file_str,
                 "Description missing trigger conditions (e.g. 'Use when...' or '当...时')")


def validate_team(team_file: Path, known_skill_names: set[str] | None = None, subset: str = "all") -> None:
    file_str = str(team_file)
    team_name = team_file.parent.name

    fm, _body, err = parse_frontmatter(team_file)
    if err:
        emit("error", "T-STR-01", file_str, err)
        return

    if _check_body_not_empty("T", file_str, team_file, _body):
        return

    do_structure = subset in ("all", "structure")
    do_content = subset in ("all", "content")

    if do_structure:
        _validate_team_structure(fm, team_file, file_str, known_skill_names)
    if do_content:
        _validate_team_content(fm, team_name, file_str)


# ---------------------------------------------------------------------------
# Uniqueness check
# ---------------------------------------------------------------------------

def check_uniqueness(kind: str, paths: list[Path]) -> None:
    """kind = skill|agent|team; paths = list of files to read."""
    rule_map = {"skill": "S-STR-15", "agent": "A-STR-09", "team": "T-STR-07"}
    rule = rule_map.get(kind, "U-STR-16")

    seen: dict[str, list[str]] = {}
    for p in paths:
        fm, _body, err = parse_frontmatter(p)
        if err:
            continue
        name = fm.get("name")
        if not isinstance(name, str) or not name:
            continue
        seen.setdefault(name, []).append(str(p))

    for name, locations in seen.items():
        if len(locations) > 1:
            for loc in locations:
                emit("error", rule, loc,
                     f"Duplicate {kind} name '{name}' also in: " +
                     ", ".join(x for x in locations if x != loc))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _known_skills(paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    for p in paths:
        fm, _b, err = parse_frontmatter(p)
        if err:
            continue
        n = fm.get("name")
        if isinstance(n, str):
            names.add(n)
    return names


def _parse_args(argv: list[str]) -> tuple[str, str, bool, list[str], list[str]]:
    """Parse command line arguments and return (cmd, subset, batch, skill_paths, filtered_argv)."""
    if len(argv) < 2:
        raise ValueError("usage: skill_validator.py <subcmd> [args...]")

    subset = "all"
    batch = False
    skill_paths_arg: list[str] = []
    filtered: list[str] = []
    
    for a in argv:
        if a.startswith("--subset="):
            subset = a.split("=", 1)[1]
        elif a == "--batch":
            batch = True
        elif a.startswith("--skill-paths="):
            skill_paths_arg.extend(a.split("=", 1)[1].split(","))
        else:
            filtered.append(a)
    
    return filtered[1], subset, batch, skill_paths_arg, filtered


def _cmd_parse(argv: list[str]) -> int:
    """Handle parse command."""
    fm, _body, err = parse_frontmatter(Path(argv[2]))
    out = {"frontmatter": fm, "error": err}
    print(json.dumps(out, ensure_ascii=False))
    return 0


def _cmd_validate_skill(argv: list[str], subset: str, batch: bool) -> int:
    """Handle validate-skill command."""
    if batch:
        for f in argv[2:]:
            print(f"FILE:{f}")
            validate_skill(Path(f), subset=subset)
    else:
        validate_skill(Path(argv[2]), subset=subset)
    return 0


def _cmd_validate_agent(argv: list[str], subset: str, batch: bool, skill_paths_arg: list[str]) -> int:
    """Handle validate-agent command."""
    if batch:
        skill_paths = [Path(p) for p in skill_paths_arg]
        known = _known_skills(skill_paths) if skill_paths else None
        for f in argv[2:]:
            print(f"FILE:{f}")
            validate_agent(Path(f), known, subset=subset)
    else:
        skill_paths = [Path(p) for p in argv[3:]]
        known = _known_skills(skill_paths) if skill_paths else None
        validate_agent(Path(argv[2]), known, subset=subset)
    return 0


def _cmd_validate_team(argv: list[str], subset: str, batch: bool, skill_paths_arg: list[str]) -> int:
    """Handle validate-team command."""
    if batch:
        skill_paths = [Path(p) for p in skill_paths_arg]
        known = _known_skills(skill_paths) if skill_paths else None
        for f in argv[2:]:
            print(f"FILE:{f}")
            validate_team(Path(f), known, subset=subset)
    else:
        skill_paths = [Path(p) for p in argv[3:]]
        known = _known_skills(skill_paths) if skill_paths else None
        validate_team(Path(argv[2]), known, subset=subset)
    return 0


def _cmd_check_uniqueness(argv: list[str]) -> int:
    """Handle check-uniqueness command."""
    if len(argv) < 4:
        print("usage: skill_validator.py check-uniqueness <kind> <path...>", file=sys.stderr)
        return 2
    kind = argv[2]
    paths = [Path(p) for p in argv[3:]]
    check_uniqueness(kind, paths)
    return 0


def main(argv: list[str]) -> int:
    """Main entry point - command dispatcher."""
    try:
        cmd, subset, batch, skill_paths_arg, argv = _parse_args(argv)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    
    handlers = {
        "parse": lambda: _cmd_parse(argv),
        "validate-skill": lambda: _cmd_validate_skill(argv, subset, batch),
        "validate-agent": lambda: _cmd_validate_agent(argv, subset, batch, skill_paths_arg),
        "validate-team": lambda: _cmd_validate_team(argv, subset, batch, skill_paths_arg),
        "check-uniqueness": lambda: _cmd_check_uniqueness(argv),
    }
    
    handler = handlers.get(cmd)
    if handler:
        return handler()
    
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
