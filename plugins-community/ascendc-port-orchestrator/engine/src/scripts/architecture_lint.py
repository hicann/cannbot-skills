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
"""architecture_lint.py — mechanized architecture-conformance checks.

Implements the deterministic floor of docs/design/HARNESS_DESIGN_PHILOSOPHY.md §5: where a
design principle can be checked deterministically, it is a lint rather than left to
review attention. This catches *plugin-architecture rot* — the "god-function / a-bunch-of-
else" drift the owner flagged (2026-06-20): mode-specific branches leaking into core,
and oversized core functions.

Checks:

  1. CORE_MODE_LEAK — a `if/elif <mode|opgen_mode|backend|target> == "<lit>"` branch
     inside a CORE module (orchestrator/ top-level + orchestrator/briefs/ + workflow/,
     not under plugins/, not a test) means mode/target-specific logic has leaked out of
     its plugin. Per HARNESS_DESIGN_PHILOSOPHY §2/§4(a): mode-specific logic belongs IN
     the plugin; the core stays mode-agnostic (mirror of OL-160, where the safety net
     keys off the canonical contract, not mode-specific names). 2026-06-20 widening
     (lint b): the original regex only matched op-gen-mode literals on `mode`; it now
     matches ANY string literal on `mode|opgen_mode|backend|target` so `target == "a3"`
     and `backend == "ascendc"` are caught too — 0-tolerance for NEW occurrences; the
     existing ones are grandfathered into the baseline.

  2. PLUGIN_NAME_BRANCH — (lint a) `(_active_)?plugin.name == "<lit>"` branches in core,
     the strongest plugin-leak smell (a core module deciding behavior by comparing the
     active plugin's identity string instead of calling a plugin method). 0-NEW; the one
     existing occurrence (phase_o5_runner.py port_a3_to_a5) is grandfathered as known
     debt to convert to a plugin method.

  3. GOD_FUNCTION — a function longer than --max-func-lines (default 200) in a core module
     is a god-function candidate; per §4(a) depth-over-breadth and general maintainability.

  4. PLUGIN_REGISTRY_INCOMPLETE — (lint c) every registered plugin must implement the
     BasePlugin required-method set so a NEW backend has to register + implement rather
     than being core-if-branched. Derived from the protocol-conformance contract
     (plugins/tests/test_protocol_conformance.REQUIRED_METHODS).

  5. FSM_HARDCODED_TRANSITION — (lint e) a hardcoded next-state/goto string literal in
     workflow/state_machine.py that bypasses the `state_transitions.jsonl`-driven
     `next_state()` table (e.g. `next_state = "await_worker"`). Transitions must come from
     the YAML/data path (`trans.get("goto")`), never a Python literal. Clean today (0
     findings) — a forward ratchet against re-introducing hardcoded routing.

Re (d) dispatch-complexity: an if/elif-chain-length sub-check was evaluated and SKIPPED —
GOD_FUNCTION (function length) already covers the dispatch functions it would flag
(workflow_critic._dispatch's mode-elif chain is inside a function already length-bounded),
and the widened CORE_MODE_LEAK catches the per-branch literals directly. A separate
chain-length metric would double-count the same lines without catching anything new.

Default is REPORT-ONLY (exit 0) so it can be adopted incrementally — like
kb_index_audit.py --report-only. Pass --strict to fail (for pre-commit / CI once the
known baseline is burned down). A --baseline file allows grandfathering known findings
while preventing NEW ones.

Usage:
  python3 src/scripts/architecture_lint.py                 # report, exit 0
  python3 src/scripts/architecture_lint.py --strict        # fail on any finding
  python3 src/scripts/architecture_lint.py --json          # machine-readable
  python3 src/scripts/architecture_lint.py --baseline src/scripts/.arch_lint_baseline.json
  python3 src/scripts/architecture_lint.py --regenerate-baseline src/scripts/.arch_lint_baseline.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

# Op-gen modes (the plugin names + re-entry modes). A core branch on any of these = leak.
# `optimize` is the --optimize re-entry mode (DEBT-161/PR#11). The staleness-guard test
# (test_architecture_lint) asserts this allowlist ⊇ every registered plugin's mode name,
# so it cannot silently lag a new backend/mode plugin (OL-160).
OPGEN_MODES = ("port_a3_to_a5", "backward", "optimize")
# DEBT-161 Batch-2 precision narrowing: a `<ident> == "<lit>"` branch in core is only an
# op-gen leak when <lit> is a REAL op-gen mode / backend / arch-target. The identifiers
# `mode`/`target` are OVERLOADED for non-op-gen meanings that are NOT leaks:
#   mode=="foreground"               → spawn execution mode (agent_transport)
#   mode=="auto"/"pre_agent_spawn"   → critic phase (workflow_critic)
#   target=="__from_user_decision__" → FSM magic token (state_machine)
# Flagging those was a false positive of the lint-(b) widening. Allowlist-by-known-value
# fixes precision; the staleness-guard test prevents the allowlist lagging new values.
KNOWN_BACKENDS = ("ascendc",)
KNOWN_ARCH_TARGETS = ("a3", "a5")
_MODE_LEAK_ALLOWLIST = {
    "mode": OPGEN_MODES,
    "opgen_mode": OPGEN_MODES,
    "backend": KNOWN_BACKENDS,
    "target": KNOWN_ARCH_TARGETS,
}

# Lint (b): widened — flag ANY string-literal branch on a mode/target identifier, not
# just op-gen-mode literals. Captures the dispatch-identifier name (group 1) and the
# string literal it is compared against (group 2). `plugin.name` is handled by the
# dedicated PLUGIN_NAME_BRANCH check (lint a) below, so it is NOT in this alternation.
_MODE_BRANCH_RE = re.compile(
    r"^\s*(?:if|elif)\b.*\b(mode|opgen_mode|backend|target)\b\s*==\s*['\"]([^'\"]+)['\"]"
)

# Lint (a): plugin.name identity branch — the strongest plugin-leak smell. Matches
# `plugin.name == "x"` / `_active_plugin.name == "x"` (incl. inside a compound `if ... and
# _active_plugin.name == "x"`). Group 1 = the literal compared against.
_PLUGIN_NAME_BRANCH_RE = re.compile(
    r"^\s*(?:if|elif)\b.*\b(?:_active_plugin|plugin)\.name\s*==\s*['\"]([^'\"]+)['\"]"
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "src", "scripts")
CORE_DIR = os.path.join(SCRIPTS_DIR, "orchestrator")
# Core module roots scanned for leak/god-function rot. orchestrator/ top-level +
# orchestrator/briefs/ + workflow/ are all "core" per the grounding audit (2026-06-20):
# they hold the inline literal-branch debt. plugins/ and any tests/ dir are excluded.
_CORE_ROOTS = (
    CORE_DIR,
    os.path.join(CORE_DIR, "briefs"),
    os.path.join(SCRIPTS_DIR, "workflow"),
)
STATE_MACHINE_PY = os.path.join(SCRIPTS_DIR, "workflow", "state_machine.py")


def _core_py_files():
    """Core modules: *.py directly under each _CORE_ROOTS dir, excluding plugins/, tests/.

    Only top-level files of each root are scanned (not recursive) so plugins/ and
    per-module tests/ subdirs are naturally excluded — and the briefs/ + workflow/ roots
    are listed explicitly rather than walked, keeping the file set deterministic and the
    plugin layer (which is ALLOWED to branch on its own paradigm) out of scope.
    """
    out = []
    seen = set()
    for root in _CORE_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not (name.endswith(".py") and os.path.isfile(p)):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _line_findings(files, finding_for_line):
    """Collect non-None line findings from each requested source file."""
    findings = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, 1):
                finding = finding_for_line(file_path, line_number, line)
                if finding is not None:
                    findings.append(finding)
    return findings


def _mode_leak_finding(file_path, line_number, line):
    """Return the architecture finding for one real mode/backend/target branch."""
    match = _MODE_BRANCH_RE.match(line)
    if match is None:
        return None
    ident, literal = match.group(1), match.group(2)
    # DEBT-161 Batch-2: only a REAL op-gen mode/backend/arch-target literal is a
    # leak. Overloaded identifiers compared to non-op-gen values are not leaks.
    if literal not in _MODE_LEAK_ALLOWLIST.get(ident, ()):
        return None
    return {
        "check": "CORE_MODE_LEAK",
        "file": os.path.relpath(file_path, REPO_ROOT),
        "line": line_number,
        "ident": ident,
        # Keep key name `mode` for baseline back-compat.
        "mode": literal,
        "detail": line.strip()[:120],
    }


def check_core_mode_leak(files):
    """Find core branches on a real op-gen mode, backend, or architecture target."""
    return _line_findings(files, _mode_leak_finding)


def _plugin_name_branch_finding(file_path, line_number, line):
    """Return a plugin-identity branch finding for one line, if present."""
    match = _PLUGIN_NAME_BRANCH_RE.match(line)
    if match is None:
        return None
    return {
        "check": "PLUGIN_NAME_BRANCH",
        "file": os.path.relpath(file_path, REPO_ROOT),
        "line": line_number,
        "mode": match.group(1),  # the literal compared against
        "detail": line.strip()[:120],
    }


def check_plugin_name_branch(files):
    """Lint (a): `(_active_)?plugin.name == "<lit>"` identity branch in core."""
    return _line_findings(files, _plugin_name_branch_finding)


def _parse_python_file(file_path):
    """Parse a source file, returning None for syntax-invalid input as before."""
    try:
        with open(file_path, encoding="utf-8") as source_file:
            return ast.parse(source_file.read(), filename=file_path)
    except SyntaxError:
        return None


def _god_function_finding(file_path, node, max_lines):
    """Build the finding for one oversized function AST node, if any."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    end_line = getattr(node, "end_lineno", node.lineno)
    span = end_line - node.lineno + 1
    if span <= max_lines:
        return None
    return {
        "check": "GOD_FUNCTION",
        "file": os.path.relpath(file_path, REPO_ROOT),
        "line": node.lineno,
        "func": node.name,
        "lines": span,
    }


def _god_function_findings(file_path, max_lines):
    """Collect all oversized function findings in a parseable source file."""
    tree = _parse_python_file(file_path)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        finding = _god_function_finding(file_path, node, max_lines)
        if finding is not None:
            findings.append(finding)
    return findings


def check_god_functions(files, max_lines):
    """Find functions whose AST source span exceeds ``max_lines``."""
    findings = []
    for file_path in files:
        findings.extend(_god_function_findings(file_path, max_lines))
    return findings


# Required-method contract for lint (c). Kept in sync with
# plugins/tests/test_protocol_conformance.REQUIRED_METHODS — the same set the runtime
# protocol-conformance test asserts. A new backend MUST register + implement these (so it
# can't be wired in via a core if-branch instead).
REQUIRED_PLUGIN_METHODS = (
    "detect",
    "verify_files",
    "forbidden_patterns",
    "scanner_category",
    "check_binary_provenance",
    "check_verify_path_provenance",
    "archive_layout_mapping",
    "archive_project_subdir",
    "resolve_archive_target",
    "check_op_host_completeness",
    "extra_finalize_checks",
    "kb_subdirs",
    "kw_brief_phase_a",
    "kw_brief_phase_d",
)


def _import_plugin_registry():
    """Return the live registry callable or the same visible import-failure finding."""
    try:
        from plugins import all_plugins  # type: ignore
    except Exception as error:  # noqa: BLE001 — surface import failure as a finding
        return None, [{
            "check": "PLUGIN_REGISTRY_INCOMPLETE",
            "file": "src/scripts/orchestrator/plugins/__init__.py",
            "plugin": "<import>",
            "detail": f"could not import plugin registry: {type(error).__name__}: {error}",
        }]
    return all_plugins, []


def _plugin_missing_method_findings(plugin):
    """Return protocol-method findings for one registered plugin instance."""
    findings = []
    plugin_name = getattr(plugin, "name", repr(plugin))
    for method_name in REQUIRED_PLUGIN_METHODS:
        method = getattr(plugin, method_name, None)
        if not callable(method):
            findings.append({
                "check": "PLUGIN_REGISTRY_INCOMPLETE",
                "file": "src/scripts/orchestrator/plugins/",
                "plugin": plugin_name,
                "method": method_name,
                "detail": f"plugin {plugin_name!r} missing required method {method_name!r}",
            })
    return findings


def _registry_missing_method_findings(all_plugins):
    """Return every missing required-method finding from the live registry."""
    findings = []
    for plugin in all_plugins():
        findings.extend(_plugin_missing_method_findings(plugin))
    return findings


def check_plugin_registry_complete():
    """Lint (c): every registered plugin implements the required-method set.

    On import failure return a visible finding rather than crashing the lint, because the
    import path can be unavailable in some CI sandboxes.
    """
    registry_dir = CORE_DIR
    added_to_path = registry_dir not in sys.path
    if added_to_path:
        sys.path.insert(0, registry_dir)
    try:
        all_plugins, import_findings = _import_plugin_registry()
        if import_findings:
            return import_findings
        return _registry_missing_method_findings(all_plugins)
    finally:
        if added_to_path:
            try:
                sys.path.remove(registry_dir)
            except ValueError:
                pass


# Lint (e): names that, when assigned a bare string literal in state_machine.py, would
# bypass the data-driven transition table. The legitimate path resolves these from
# `trans.get("goto")` / `snap.get("user_decision_target")` (a call, not a literal), or
# compares against the `__from_user_decision__` magic token. A direct `next_state = "x"`
# or `goto = "await_worker"` hardcode is the regression this guards against.
_FSM_TARGET_NAMES = ("next_state", "to_state", "goto")
# Magic tokens / sentinels that are NOT real state hardcodes (allowed).
_FSM_ALLOWED_LITERALS = {"__from_user_decision__", "init", "abort"}


def check_fsm_hardcoded_transition():
    """Lint (e): flag `<next_state|to_state|goto> = "<state-literal>"` in state_machine.py.

    AST-based: catches an assignment whose target is one of _FSM_TARGET_NAMES and whose
    RHS is a plain string Constant (not a Call like `.get("goto")`, not the
    `__from_user_decision__` sentinel). That is a hardcoded routing decision that bypasses
    the jsonl/YAML-driven table. Comparisons (`if target == "..."`) are NOT flagged — only
    assignments that SET the transition target.
    """
    findings = []
    if not os.path.isfile(STATE_MACHINE_PY):
        return findings
    try:
        with open(STATE_MACHINE_PY, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=STATE_MACHINE_PY)
    except SyntaxError:
        return findings
    rel = os.path.relpath(STATE_MACHINE_PY, REPO_ROOT)
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        if value.value in _FSM_ALLOWED_LITERALS:
            continue
        for t in targets:
            name = getattr(t, "id", None) or getattr(t, "attr", None)
            if name in _FSM_TARGET_NAMES:
                findings.append({
                    "check": "FSM_HARDCODED_TRANSITION",
                    "file": rel,
                    "line": node.lineno,
                    "target": name,
                    "literal": value.value,
                    "detail": f"{name} = {value.value!r} bypasses data-driven transition table",
                })
    return findings


def _key(x):
    """Stable identity for baseline matching. Per-check disambiguation:
    - GOD_FUNCTION: (check, file, func)
    - CORE_MODE_LEAK / PLUGIN_NAME_BRANCH: (check, file, mode-literal)
    - PLUGIN_REGISTRY_INCOMPLETE: (check, plugin, method)
    - FSM_HARDCODED_TRANSITION: (check, file, target, literal)
    Line numbers are intentionally NOT part of the key so that adding/removing lines
    elsewhere in a file doesn't un-grandfather an existing finding.
    """
    c = x["check"]
    if c == "GOD_FUNCTION":
        return (c, x["file"], x.get("func"))
    if c in ("CORE_MODE_LEAK", "PLUGIN_NAME_BRANCH"):
        return (c, x["file"], x.get("mode"))
    if c == "PLUGIN_REGISTRY_INCOMPLETE":
        return (c, x.get("plugin"), x.get("method"))
    if c == "FSM_HARDCODED_TRANSITION":
        return (c, x["file"], x.get("target"), x.get("literal"))
    return (c, x.get("file"), x.get("func"), x.get("mode"))


def collect_findings(max_func_lines):
    files = _core_py_files()
    return (
        check_core_mode_leak(files)
        + check_plugin_name_branch(files)
        + check_god_functions(files, max_func_lines)
        + check_plugin_registry_complete()
        + check_fsm_hardcoded_transition()
    )


def _load_baseline(path):
    baseline = set()
    if path and os.path.exists(path):
        with open(path) as fh:
            for b in json.load(fh):
                baseline.add(_key(b))
    return baseline


def _baseline_record(f):
    """The minimal, stable subset of a finding persisted to the baseline file —
    everything _key() reads, so a regenerated baseline round-trips exactly."""
    rec = {"check": f["check"]}
    for fld in ("file", "func", "mode", "plugin", "method", "target", "literal"):
        if fld in f:
            rec[fld] = f.get(fld)
    return rec


def partition_by_changed(new_findings, changed_files):
    """Split non-baselined findings into (blocking, warn_only) for the --changed-only
    ratchet: a finding BLOCKS only if its file was changed by this commit; a finding in a
    file NOT in the changed set is pre-existing-from-this-commit's-view → warn, don't block.

    This is the non-bricking pre-commit semantic (DEBT-164): a contributor cannot ADD a new
    violation in a file they touch, but a NEW violation someone else left elsewhere in the
    repo does not brick every unrelated commit. `changed_files` are repo-relative paths
    (e.g. from `git diff --cached --name-only`), matching finding['file'].
    """
    changed = set(changed_files or ())
    blocking = [f for f in new_findings if f.get("file") in changed]
    warn_only = [f for f in new_findings if f.get("file") not in changed]
    return blocking, warn_only


def _argument_parser():
    """Build the stable command-line interface for this lint."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strict", action="store_true", help="exit 1 if any non-baselined finding")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--max-func-lines", type=int, default=200, help="god-function threshold (default 200)")
    parser.add_argument("--baseline", help="JSON file of grandfathered findings to ignore")
    parser.add_argument("--regenerate-baseline", metavar="PATH",
                        help="write ALL current findings to PATH as the new baseline (ratchet reset) and exit 0")
    parser.add_argument("--changed-only", nargs="*", metavar="FILE",
                        help="ratchet scope (DEBT-164): with --strict, block ONLY on new findings whose "
                             "file is in this set (e.g. `git diff --cached --name-only`). New findings in "
                             "files NOT in the set are warned-but-allowed — the non-bricking pre-commit gate.")
    return parser


def _baseline_sort_key(record):
    """Return the deterministic ordering key for regenerated baseline records."""
    return (
        record["check"], record.get("file") or "", record.get("plugin") or "",
        record.get("func") or "", record.get("method") or "", record.get("mode") or "",
        record.get("target") or "", record.get("literal") or "",
    )


def _regenerate_baseline(path, findings):
    """Write the current findings as a stable baseline and report success."""
    records = sorted((_baseline_record(finding) for finding in findings), key=_baseline_sort_key)
    with open(path, "w") as baseline_file:
        json.dump(records, baseline_file, indent=2)
        baseline_file.write("\n")
    print(f"architecture_lint: wrote {len(records)} grandfathered finding(s) to {path}")
    return 0


def _group_findings(findings):
    """Group known finding types in the report's established display order."""
    groups = {
        "CORE_MODE_LEAK": [],
        "PLUGIN_NAME_BRANCH": [],
        "GOD_FUNCTION": [],
        "PLUGIN_REGISTRY_INCOMPLETE": [],
        "FSM_HARDCODED_TRANSITION": [],
    }
    for finding in findings:
        group = groups.get(finding["check"])
        if group is not None:
            group.append(finding)
    return groups


def _baseline_tag(finding, baseline):
    """Return the unchanged text suffix marking a grandfathered finding."""
    return "" if _key(finding) not in baseline else "  [baselined]"


def _print_mode_leaks(findings, baseline):
    """Print CORE_MODE_LEAK records in the existing human report format."""
    for finding in findings:
        tag = _baseline_tag(finding, baseline)
        print(f"  CORE_MODE_LEAK       {finding['file']}:{finding['line']}  "
              f"{finding.get('ident','?')}=={finding['mode']!r}{tag}")
        print(f"      {finding['detail']}")


def _print_plugin_name_branches(findings, baseline):
    """Print PLUGIN_NAME_BRANCH records in the existing human report format."""
    for finding in findings:
        tag = _baseline_tag(finding, baseline)
        print(f"  PLUGIN_NAME_BRANCH   {finding['file']}:{finding['line']}  "
              f"plugin.name=={finding['mode']!r}{tag}")
        print(f"      {finding['detail']}")


def _print_god_functions(findings, baseline):
    """Print GOD_FUNCTION records ordered by descending source span."""
    for finding in sorted(findings, key=lambda item: -item["lines"]):
        tag = _baseline_tag(finding, baseline)
        print(f"  GOD_FUNCTION         {finding['file']}:{finding['line']}  "
              f"{finding['func']}() = {finding['lines']} lines{tag}")


def _print_registry_gaps(findings, baseline):
    """Print PLUGIN_REGISTRY_INCOMPLETE records in the human report format."""
    for finding in findings:
        tag = _baseline_tag(finding, baseline)
        print(f"  PLUGIN_REGISTRY_INCOMPLETE  {finding.get('plugin')}  {finding['detail']}{tag}")


def _print_fsm_hardcodes(findings, baseline):
    """Print FSM_HARDCODED_TRANSITION records in the human report format."""
    for finding in findings:
        tag = _baseline_tag(finding, baseline)
        print(f"  FSM_HARDCODED_TRANSITION  {finding['file']}:{finding['line']}  "
              f"{finding['detail']}{tag}")


def _finding_location(finding):
    """Return the display location used by changed-only warnings and blocks."""
    if not finding.get("file"):
        return "?"
    return f"{finding['file']}:{finding.get('line','?')}"


def _print_changed_only(blocking, warn_only):
    """Print changed-only warnings and blocking findings in their established order."""
    for finding in warn_only:
        print(f"  [warn, not blocking — pre-existing, file not in this commit] "
              f"{finding['check']} {_finding_location(finding)}")
    for finding in blocking:
        print(f"  [BLOCK — new violation in a file this commit changes] "
              f"{finding['check']} {_finding_location(finding)}")
    print(f"  changed-only ratchet: {len(blocking)} blocking, {len(warn_only)} warn-only")


def _print_text_report(findings, baseline, new_findings, max_func_lines, blocking, warn_only,
                       changed_only):
    """Print the report-only output without changing its command-line contract."""
    groups = _group_findings(findings)
    print(f"architecture_lint: {len(groups['CORE_MODE_LEAK'])} core mode-leak(s), "
          f"{len(groups['PLUGIN_NAME_BRANCH'])} plugin.name-branch(es), "
          f"{len(groups['GOD_FUNCTION'])} god-function(s) (threshold {max_func_lines}), "
          f"{len(groups['PLUGIN_REGISTRY_INCOMPLETE'])} registry-gap(s), "
          f"{len(groups['FSM_HARDCODED_TRANSITION'])} fsm-hardcode(s)")
    _print_mode_leaks(groups["CORE_MODE_LEAK"], baseline)
    _print_plugin_name_branches(groups["PLUGIN_NAME_BRANCH"], baseline)
    _print_god_functions(groups["GOD_FUNCTION"], baseline)
    _print_registry_gaps(groups["PLUGIN_REGISTRY_INCOMPLETE"], baseline)
    _print_fsm_hardcodes(groups["FSM_HARDCODED_TRANSITION"], baseline)
    print(f"\n  new (non-baselined): {len(new_findings)}")
    if changed_only is not None:
        _print_changed_only(blocking, warn_only)
    print("  ref: docs/design/HARNESS_DESIGN_PHILOSOPHY.md §5 (mechanized invariants)")


def _print_json_report(findings, new_findings, blocking, warn_only, changed_only):
    """Print the report's machine-readable representation."""
    output = {
        "findings": findings,
        "new": new_findings,
        "baselined": len(findings) - len(new_findings),
    }
    if changed_only is not None:
        output["blocking"] = blocking
        output["warn_only"] = warn_only
    print(json.dumps(output, indent=2))


def _print_report(args, findings, new_findings, baseline, blocking, warn_only):
    """Select the requested JSON or human-readable report renderer."""
    if args.json:
        _print_json_report(findings, new_findings, blocking, warn_only, args.changed_only)
        return
    _print_text_report(findings, baseline, new_findings, args.max_func_lines, blocking, warn_only,
                       args.changed_only)


def main(argv=None):
    """Run the architecture checks and return the documented process status."""
    args = _argument_parser().parse_args(argv)
    findings = collect_findings(args.max_func_lines)
    if args.regenerate_baseline:
        return _regenerate_baseline(args.regenerate_baseline, findings)
    baseline = _load_baseline(args.baseline)
    new_findings = [finding for finding in findings if _key(finding) not in baseline]
    blocking, warn_only = partition_by_changed(new_findings, args.changed_only) \
        if args.changed_only is not None else (new_findings, [])
    _print_report(args, findings, new_findings, baseline, blocking, warn_only)
    # With --changed-only, only violations in changed files block (DEBT-164).
    return 1 if args.strict and blocking else 0


if __name__ == "__main__":
    sys.exit(main())
