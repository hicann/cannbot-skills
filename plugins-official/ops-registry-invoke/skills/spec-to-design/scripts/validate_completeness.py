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
"""Content completeness checks for spec-to-design DESIGN.md and PLAN.md.

Supports paradigm-specific manifest.yaml files:
  references/<paradigm>/manifest.yaml

Manifest rules are auto-discovered from spec.yaml's op.paradigms.
Other paradigms only need to add their own manifest.yaml — no Python changes.
"""

from __future__ import annotations

# Standard library
import argparse
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

# Third-party
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESOURCES_DIR = Path(__file__).resolve().parents[3] / "workflow" / "resources"
if _RESOURCES_DIR.is_dir() and str(_RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_DIR))

# Local (path-dependent; must follow the sys.path insertion above)
from frontmatter_utils import parse_yaml_frontmatter
import validate_design
from _output_log import get_logger

logger = logging.getLogger(__name__)
_OUTPUT_LOGGER = get_logger("ops_registry_invoke.validate_completeness")


SECTION_MIN_LINES = {
    "修订记录": 3,
    "1. 概述": 16,
    "2. 架构设计": 18,
    "3. 实现方案": 28,
    "4. 性能优化": 6,
    "5. 风险评估": 8,
    "6. 交付件清单": 6,
    "7. 迭代规划": 5,
    "8. Design Contract": 5,
}

SECTION_REQUIRED_TERMS = {
    "1. 概述": ["算子名称", "目标芯片", "目标架构", "spec.yaml 一致性映射"],
    "2. 架构设计": ["op_api", "op_host", "op_kernel", "op_graph"],
    "3. 实现方案": ["TilingData", "TilingKey", "API 映射", "API 验证记录", "数据流", "内存管理", "UB 容量验证"],
    "4. 性能优化": ["并行", "流水线"],
    "5. 风险评估": ["API 风险", "精度风险", "性能风险"],
    "7. 迭代规划": [],
    "8. Design Contract": [
        "Programming Model", "Compute Chain", "API Constraints",
        "Precision Strategy", "Memory Strategy",
    ],
}

SECTION_REQUIRED_TERMS_BY_ITERATION = {
    1: ["迭代一"],
    2: ["迭代一", "迭代二"],
    3: ["迭代一", "迭代二", "迭代三"],
}

PLAN_REQUIRED_TERMS_BASE = [
    "TilingKey",
    "Dtype",
    "Memory Strategy",
]

PLAN_REQUIRED_TERMS_BY_ITERATION = {
    1: [],
    2: ["迭代一", "迭代二"],
    3: ["迭代一", "迭代二", "迭代三"],
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_section(markdown: str, title: str) -> str:
    pattern = re.compile(rf"^## {re.escape(title)}\s*$", re.MULTILINE)
    match = pattern.search(markdown)
    if not match:
        return ""
    next_match = re.search(r"^## .+$", markdown[match.end():], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(markdown)
    return markdown[match.start():end].strip()


def body_line_count(section: str) -> int:
    lines = section.splitlines()[1:] if section else []
    return sum(1 for line in lines if line.strip())


def _heading_key(heading: str) -> str:
    """Normalize a heading to a lookup key: strip ###/#### prefix and whitespace."""
    return re.sub(r"^#{2,4}\s*", "", heading).strip()


def _find_blocks(text: str, heading: str) -> list[str]:
    """Find all blocks under a ### or #### heading, return their concatenated content."""
    key = re.escape(_heading_key(heading))
    pattern = re.compile(rf"^(?:###|####)\s+{key}\s*$", re.MULTILINE)
    blocks = []
    for match in pattern.finditer(text):
        rest = text[match.end():]
        next_match = re.search(r"^(?:###|####) ", rest, re.MULTILINE)
        end = match.end() + next_match.start() if next_match else len(text)
        blocks.append(text[match.start():end].strip())
    return blocks


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

_MANIFEST_CACHE: dict[str, dict] = {}
_GATE_FUNCS_CACHE: dict[str, list] = {}
_SKILL_HOMES_CACHE: dict[str, str] | None = None


def _find_skill_root() -> Path | None:
    """Locate the spec-to-design skill root (this script's parent directory)."""
    env = os.environ.get("SPEC_TO_DESIGN_SKILL_DIR", "")
    if env and Path(env).is_dir():
        return Path(env)
    # This script lives in <skill_root>/scripts/ — the parent directory is the skill root.
    return Path(__file__).resolve().parent.parent


def _append_unique(path: Path, roots: list[Path], seen: set[str]) -> None:
    """Append ``path`` to ``roots`` only if its string form is unseen."""
    key = str(path)
    if key not in seen:
        roots.append(path)
        seen.add(key)


def _collect_symlink_candidates(
    resolved: Path | None, roots: list[Path], seen: set[str]
) -> None:
    """Walk up from CWD, adding .claude/.opencode skill dirs (real + symlink)."""
    skill_name = resolved.name if resolved else "spec-to-design"
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        for prefix in (".claude", ".opencode"):
            candidate = parent / prefix / "skills" / skill_name
            if not candidate.is_dir():
                continue
            # Resolved target, then the symlink path itself (for sibling lookup).
            _append_unique(candidate.resolve(), roots, seen)
            _append_unique(candidate, roots, seen)


def _find_skill_dirs() -> list[Path]:
    """Find all possible skill root directories (resolved + symlink-based).

    When skills are symlinked from different source directories, the resolved
    path's parent may not contain sibling skills. This function also searches
    for .claude/skills/ and .opencode/skills/ directories from CWD upward.
    """
    roots: list[Path] = []
    seen: set[str] = set()

    # 1. Resolved path (always include)
    resolved = _find_skill_root()
    if resolved:
        _append_unique(resolved, roots, seen)

    # 2. Walk up from CWD looking for .claude/skills/ and .opencode/skills/
    _collect_symlink_candidates(resolved, roots, seen)

    return roots


def _find_in_skill_dirs(
    all_skill_dirs: list[Path],
    paradigm: str,
    skill_homes: dict[str, str],
    filename: str,
) -> Path | None:
    """Find a paradigm-specific file across all discovered skill directories.

    Search order:
    1. <skill_dir>/references/paradigms/<paradigm.lower()>/<filename> for each known skill dir
    2. <skill_dir.parent>/<skill_homes[paradigm]>/references/paradigms/<paradigm.lower()>/<filename>
    3. <repo_root>/ops/<skill_homes[paradigm]>/references/paradigms/<paradigm.lower()>/<filename>
    """
    para_lower = paradigm.lower()

    # Direct: each skill dir's own references/paradigms/
    for d in all_skill_dirs:
        candidate = d / "references" / "paradigms" / para_lower / filename
        if candidate.exists():
            return candidate

    home_name = skill_homes.get(paradigm, skill_homes.get("default"))
    if home_name:
        # Via skill_homes: sibling skill name from each skill dir's parent
        for d in all_skill_dirs:
            candidate = d.parent / home_name / "references" / "paradigms" / para_lower / filename
            if candidate.exists():
                return candidate

        # Via skill_homes: repo-root ops/ directory
        for d in all_skill_dirs:
            repo_root = d.parent.parent.parent.parent
            candidate = (
                repo_root / "ops" / home_name / "references"
                / "paradigms" / para_lower / filename
            )
            if candidate.exists():
                return candidate

    return None


def _load_skill_homes(skill_root: Path) -> dict[str, str]:
    """Load skill_homes mapping from paradigm-refs.yaml (cached)."""
    global _SKILL_HOMES_CACHE
    if _SKILL_HOMES_CACHE is not None:
        return _SKILL_HOMES_CACHE
    refs_file = skill_root / "references" / "paradigm-refs.yaml"
    if not refs_file.exists():
        _SKILL_HOMES_CACHE = {}
        return _SKILL_HOMES_CACHE
    try:
        data = yaml.safe_load(refs_file.read_text(encoding="utf-8"))
        _SKILL_HOMES_CACHE = data.get("skill_homes", {}) or {}
    except Exception:
        _SKILL_HOMES_CACHE = {}
    return _SKILL_HOMES_CACHE


def _resolve_design_paradigms(spec_path: Path) -> list[str] | None:
    """Resolve the design-driving paradigms declared in spec.yaml.

    Precedence: op.paradigm_groups (Elementwise kept) > op.paradigm_routing
    (backward compat) > op.paradigms with Elementwise stripped as fallback.

    Returns None when the spec is missing or cannot be parsed.
    """
    if not spec_path.exists():
        return None

    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        op = spec.get("op", {})
        paradigms = op.get("paradigms", [])
        paradigm_groups = op.get("paradigm_groups") or []
        routing = op.get("paradigm_routing")
    except Exception:
        return None

    if paradigm_groups and isinstance(paradigm_groups, list):
        # paradigm_groups mode: collect all paradigms from all groups
        active_set: set = set()
        for group in paradigm_groups:
            active_set.update(group.get("paradigms", []))
        return list(active_set) or paradigms

    if routing and isinstance(routing, dict):
        # Backward compat: paradigm_routing mode
        active_set = set()
        for case in routing.get("cases", []):
            active_set.update(case.get("active_paradigms", []))
        return list(active_set) or paradigms

    return [p for p in paradigms if p != "Elementwise"] or paradigms


def _rule_from_value(value: dict) -> dict | None:
    """Build a section rule dict from a manifest node that carries a heading."""
    heading = value.get("heading", "")
    if not heading:
        return None
    rule = {"heading": heading}
    for field in ("required_terms", "required_patterns", "min_lines"):
        if field in value:
            rule[field] = value[field]
    return rule


def _merge_manifest(manifest: dict, combined: dict) -> None:
    """Extend ``combined`` with section rules and gate_checks from ``manifest``."""
    for value in manifest.values():
        if not isinstance(value, dict):
            continue
        top_rule = _rule_from_value(value)
        if top_rule:
            combined["sections"].append(top_rule)
        for sub_value in value.values():
            if isinstance(sub_value, dict) and "heading" in sub_value:
                combined["sections"].append(_rule_from_value(sub_value))

    for gate_check in manifest.get("gate_checks", []):
        if isinstance(gate_check, dict):
            combined["gate_checks"].append(gate_check)


def _load_single_manifest(
    paradigm: str,
    skill_root: Path,
    all_skill_dirs: list[Path],
    skill_homes: dict[str, str],
) -> dict | None:
    """Locate and parse a paradigm's manifest.yaml, returning its dict or None."""
    manifest_path = skill_root / "references" / paradigm.lower() / "manifest.yaml"
    if not manifest_path.exists():
        manifest_path = _find_in_skill_dirs(
            all_skill_dirs, paradigm, skill_homes, "manifest.yaml"
        )
    if not manifest_path or not manifest_path.exists():
        return None
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("manifest: failed to parse %s", manifest_path, exc_info=True)
        return None
    return manifest if isinstance(manifest, dict) else None


def load_manifest_rules(spec_path: Path) -> dict:
    """Load manifest rules for the design-driving paradigm from spec.yaml.

    Returns {} if spec has no paradigms or no manifest files are found.
    Cached per paradigm combo to avoid repeated I/O.

    When op.paradigm_groups is present, manifest rules are loaded for all
    paradigms declared in groups (Elementwise is NOT stripped).
    Falls back to op.paradigm_routing for backward compat with old specs.
    """
    design_paradigms = _resolve_design_paradigms(spec_path)
    if design_paradigms is None:
        return {}

    cache_key = ",".join(design_paradigms)
    if cache_key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[cache_key]

    skill_root = _find_skill_root()
    if skill_root is None:
        return {}

    combined: dict = {"sections": [], "gate_checks": []}
    skill_homes = _load_skill_homes(skill_root)
    all_skill_dirs = _find_skill_dirs()

    for paradigm in design_paradigms:
        manifest = _load_single_manifest(
            paradigm, skill_root, all_skill_dirs, skill_homes
        )
        if manifest is not None:
            _merge_manifest(manifest, combined)

    _MANIFEST_CACHE[cache_key] = combined
    return combined


def _load_gate_module(gate_script: Path, paradigm: str):
    """Import a paradigm's validate_gates.py and return its module, or None."""
    try:
        mod_spec = importlib.util.spec_from_file_location(
            f"validate_gates_{paradigm.lower()}", str(gate_script)
        )
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    except Exception:
        logger.warning("gate funcs: failed to load %s", gate_script, exc_info=True)
        return None
    return module


def _resolve_gate_func(
    paradigm: str,
    skill_root: Path,
    all_skill_dirs: list[Path],
    skill_homes: dict[str, str],
):
    """Locate and load a paradigm's validate_gates callable, or None."""
    gate_script = skill_root / "references" / paradigm.lower() / "validate_gates.py"
    if not gate_script.exists():
        gate_script = _find_in_skill_dirs(
            all_skill_dirs, paradigm, skill_homes, "validate_gates.py"
        )
    if not gate_script or not gate_script.exists():
        logger.debug("gate funcs: no validate_gates.py for paradigm %s", paradigm)
        return None

    module = _load_gate_module(gate_script, paradigm)
    if module is None:
        return None
    if not hasattr(module, "validate_gates"):
        logger.warning("gate funcs: %s has no validate_gates attribute", gate_script)
        return None
    logger.info("gate funcs: loaded %s.validate_gates from %s", paradigm, gate_script)
    return module.validate_gates


def _load_paradigm_gate_funcs(spec_path: Path) -> list:
    """Discover per-paradigm validate_gates.py scripts and return their callables.

    Each references/<paradigm>/validate_gates.py must expose:
        def validate_gates(design_text: str) -> list[str]: ...
    """
    design_paradigms = _resolve_design_paradigms(spec_path)
    if design_paradigms is None:
        logger.debug("gate funcs: spec %s missing or unparseable", spec_path)
        return []

    cache_key = ",".join(design_paradigms)
    if cache_key in _GATE_FUNCS_CACHE:
        logger.debug("gate funcs: cache hit for %s", cache_key)
        return _GATE_FUNCS_CACHE[cache_key]

    skill_root = _find_skill_root()
    if skill_root is None:
        logger.debug("gate funcs: skill root not found")
        return []

    funcs: list = []
    skill_homes = _load_skill_homes(skill_root)
    all_skill_dirs = _find_skill_dirs()
    for paradigm in design_paradigms:
        func = _resolve_gate_func(paradigm, skill_root, all_skill_dirs, skill_homes)
        if func is not None:
            funcs.append(func)

    _GATE_FUNCS_CACHE[cache_key] = funcs
    return funcs


# ---------------------------------------------------------------------------
# Manifest rule application — only runs when manifest rules exist
# ---------------------------------------------------------------------------

def _check_section_rule(rule: dict, design_text: str) -> list[str]:
    """Check one manifest section rule against the design text."""
    heading = rule.get("heading", "")
    blocks = _find_blocks(design_text, heading)
    content = "\n".join(blocks) if blocks else ""
    if not content:
        return []

    errors: list[str] = []
    for term in rule.get("required_terms", []):
        if term not in content:
            errors.append(f"§{heading}: missing required term {term!r}")

    for pat in rule.get("required_patterns", []):
        if not isinstance(pat, dict):
            continue
        pattern = pat.get("pattern", "")
        desc = pat.get("description", pattern)
        if not re.search(pattern, content):
            errors.append(f"§{heading}: {desc}")

    min_ln = rule.get("min_lines", 0)
    if min_ln > 0:
        actual = body_line_count(content)
        if actual < min_ln:
            errors.append(f"§{heading}: too short ({actual} lines, need >= {min_ln})")

    return errors


def _check_gate_check(gate_check: dict, design_text: str) -> list[str]:
    """Check one automatable manifest gate_check against the design text."""
    check_str = gate_check.get("check", "")
    gc_id = gate_check.get("id", "?")
    gc_desc = gate_check.get("desc", "")

    # Automatable: grep -c "PATTERN" → 期望 N
    m = re.search(r'grep\s+-c\s+"([^"]+)"\s*[→>]\s*期望\s*(\d+)', check_str)
    if m:
        pattern = m.group(1)
        expected = int(m.group(2))
        count = len(re.findall(pattern, design_text))
        if count != expected:
            return [f"G{gc_id}: {gc_desc} (found {count}, expected {expected})"]
        return []

    # Automatable: §X 含 "TERM" (single or multiple terms)
    terms = re.findall(r'"([^"]+)"', check_str)
    missing = [t for t in terms if t not in design_text]
    if missing:
        return [f"G{gc_id}: {gc_desc} (missing: {', '.join(missing)})"]
    return []


def _apply_manifest_checks(design_text: str, rules: dict) -> list[str]:
    """Apply manifest section rules and automatable gate_checks."""
    errors: list[str] = []
    for rule in rules.get("sections", []):
        errors.extend(_check_section_rule(rule, design_text))
    for gate_check in rules.get("gate_checks", []):
        errors.extend(_check_gate_check(gate_check, design_text))
    return errors


# ---------------------------------------------------------------------------
# Main validation entry — original signature unchanged
# ---------------------------------------------------------------------------

def validate_design_completeness(design_path: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(design_path)

    iteration_count = 3
    plan_path = design_path.parent / "PLAN.md"
    if plan_path.exists():
        frontmatter, _ = parse_yaml_frontmatter(read_text(plan_path))
        ic = frontmatter.get("iteration_count")
        if ic in (1, 2, 3):
            iteration_count = ic

    for section, min_lines in SECTION_MIN_LINES.items():
        body = extract_section(text, section)
        if not body:
            errors.append(f"{section}: missing section")
            continue
        count = body_line_count(body)
        if count < min_lines:
            errors.append(f"{section}: too little content ({count} non-empty lines, need >= {min_lines})")
        required_terms = SECTION_REQUIRED_TERMS.get(section, [])
        if section == "7. 迭代规划":
            required_terms = SECTION_REQUIRED_TERMS_BY_ITERATION.get(iteration_count, [])
        for term in required_terms:
            if term not in body:
                errors.append(f"{section}: missing required term {term!r}")

    if "待补充" in text or "待验证" in text:
        if "风险" not in text and "需回到" not in text:
            errors.append("uncertain items found but no risk/follow-up explanation")

    return errors


def validate_design_completeness_with_manifest(
    design_path: Path, spec_path: Path
) -> list[str]:
    """Validate DESIGN.md with paradigm-specific rules (manifest + gates) then common checks."""
    manifest_rules = load_manifest_rules(spec_path)
    gate_funcs = _load_paradigm_gate_funcs(spec_path)

    if manifest_rules or gate_funcs:
        logger.info("running manifest+gate validation: %d section rule(s), %d gate func(s)",
                    len(manifest_rules.get("sections", [])), len(gate_funcs))
        errors = _apply_manifest_checks(read_text(design_path), manifest_rules)
        for validate_gates in gate_funcs:
            errors.extend(validate_gates(read_text(design_path)))
        errors.extend(validate_design_completeness(design_path))
    else:
        logger.info("no manifest rules or gate funcs found; running common completeness checks only")
        errors = validate_design_completeness(design_path)

    return errors


def validate_plan_completeness(plan_path: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(plan_path)

    frontmatter, body = parse_yaml_frontmatter(text)
    iteration_count = frontmatter.get("iteration_count", 3)
    if isinstance(iteration_count, str):
        iteration_count = int(iteration_count) if iteration_count.isdigit() else 3
    if iteration_count not in (1, 2, 3):
        iteration_count = 3

    required_terms = PLAN_REQUIRED_TERMS_BASE + PLAN_REQUIRED_TERMS_BY_ITERATION.get(iteration_count, [])
    for term in required_terms:
        if term not in body:
            errors.append(f"PLAN.md missing required term: {term}")

    min_table_rows = iteration_count * 4
    table_lines = [line for line in body.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < min_table_rows:
        errors.append(
            "PLAN.md has too few table rows for iteration planning "
            f"({len(table_lines)} rows, need >= {min_table_rows})"
        )
    return errors


def _resolve_log_file(cli_log_file: Path | None, spec_path: Path) -> Path | None:
    """Determine the log file path, honoring CLI flag > $VALIDATE_LOG_FILE > auto-derived.

    Returns None when logging-to-file is explicitly disabled (value '-') or when
    the auto-derived location cannot be created.
    """
    candidates: list[tuple[str, Path | None]] = [
        ("cli --log-file", cli_log_file),
        ("$VALIDATE_LOG_FILE", Path(os.environ["VALIDATE_LOG_FILE"]) if os.environ.get("VALIDATE_LOG_FILE") else None),
    ]
    for source, value in candidates:
        if value is None:
            continue
        if str(value) == "-":
            logger.debug("log file disabled via %s", source)
            return None
        return value

    # Auto-derive: place next to spec.yaml so logs travel with the operator docs.
    try:
        spec_dir = spec_path.resolve().parent
    except OSError:
        return None
    return spec_dir / ".validate_completeness.log"


def _configure_logging(verbose: bool, log_file: Path | None) -> None:
    """Configure root logger with stderr (always) + optional file handler.

    File always records DEBUG for full trace; stderr level depends on --verbose.
    """
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers filter

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
            # Announce on stderr so the agent (and user) sees where the trace went.
            stderr_handler.emit(logging.LogRecord(
                name=__name__, level=logging.INFO, pathname=__file__, lineno=0,
                msg="log file: %s", args=(log_file,), exc_info=None,
            ))
        except OSError as e:
            stderr_handler.emit(logging.LogRecord(
                name=__name__, level=logging.WARNING, pathname=__file__, lineno=0,
                msg="failed to open log file %s: %s; stderr only", args=(log_file, e), exc_info=None,
            ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--sections-dir", type=Path,
        help="optional generated sections directory; accepted for workflow compatibility",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable DEBUG-level logging for detailed gate discovery trace")
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="path to a log file. Defaults to <spec_dir>/.validate_completeness.log so "
             "the agent flow leaves a persistent trace next to spec.yaml. "
             "Override with $VALIDATE_LOG_FILE; set either to '-' to disable.",
    )
    args = parser.parse_args()

    log_file = _resolve_log_file(args.log_file, args.spec)
    _configure_logging(verbose=args.verbose, log_file=log_file)

    structural_args = argparse.Namespace(
        spec=args.spec,
        template=args.template,
        design=args.design,
        plan=args.plan,
    )
    errors = validate_design.validate_design(structural_args)
    op_name = validate_design.extract_op_name(read_text(args.spec))
    if args.plan:
        errors.extend(validate_design.validate_plan(args.plan, op_name))

    # Use manifest-enriched validation when spec is available
    errors.extend(
        validate_design_completeness_with_manifest(args.design, args.spec)
    )
    if args.plan:
        errors.extend(validate_plan_completeness(args.plan))

    if errors:
        for error in errors:
            _OUTPUT_LOGGER.info(f"ERROR: {error}")
        return 1

    _OUTPUT_LOGGER.info(f"OK: {args.design} -- completeness checks passed")
    if args.plan:
        _OUTPUT_LOGGER.info(f"OK: {args.plan} -- completeness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
