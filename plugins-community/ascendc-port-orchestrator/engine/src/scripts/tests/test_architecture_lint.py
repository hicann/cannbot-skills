# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for src/scripts/architecture_lint.py — the §5 mechanized invariants.

Covers the 2026-06-20 extension (lints a–e) on top of the original
CORE_MODE_LEAK + GOD_FUNCTION skeleton:

  (a) PLUGIN_NAME_BRANCH  — `(_active_)?plugin.name == "<lit>"` detection
  (b) widened CORE_MODE_LEAK — `(mode|opgen_mode|backend|target) == "<lit>"`
  (c) PLUGIN_REGISTRY_INCOMPLETE — every registered plugin implements the contract
  (e) FSM_HARDCODED_TRANSITION — hardcoded next-state literal in state_machine.py

Plus the ratchet contract:
  (i)   the widened regexes catch target==/backend==/plugin.name== samples
  (ii)  --strict passes on current code with the committed baseline
  (iii) a NEW injected violation is caught (non-zero exit)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parent.parent  # src/scripts/
sys.path.insert(0, str(_SCRIPTS))

import architecture_lint as al  # noqa: E402

_LINT_PY = _SCRIPTS / "architecture_lint.py"
_BASELINE = _SCRIPTS / ".arch_lint_baseline.json"


# ── (i) widened regexes catch the new identifier forms ──────────────────────

@pytest.mark.parametrize("line,ident,lit", [
    ('    if target == "a3":', "target", "a3"),
    ('    if backend == "ascendc":', "backend", "ascendc"),
    ('    elif mode == "backward":', "mode", "backward"),
    ('    if opgen_mode == "port_a3_to_a5" and target != "a5":', "opgen_mode", "port_a3_to_a5"),
])
def test_mode_branch_regex_widened(line, ident, lit):
    m = getattr(al, '_MODE_BRANCH_RE').match(line)
    assert m is not None, f"widened _MODE_BRANCH_RE should match {line!r}"
    assert m.group(1) == ident
    assert m.group(2) == lit


def test_mode_branch_regex_ignores_non_dispatch_compare():
    # A comparison that is NOT on a dispatch identifier must not match.
    assert getattr(al, '_MODE_BRANCH_RE').match('    if name == "foo":') is None
    assert getattr(al, '_MODE_BRANCH_RE').match('    x = mode == "a3"') is None  # not if/elif


@pytest.mark.parametrize("line,lit", [
    ('    if plugin.name == "backward":', "backward"),
    ('        if _active_plugin is not None and _active_plugin.name == "port_a3_to_a5":', "port_a3_to_a5"),
    ('    elif plugin.name == "unsupported_mode":', "unsupported_mode"),
])
def test_plugin_name_branch_regex(line, lit):
    m = getattr(al, '_PLUGIN_NAME_BRANCH_RE').match(line)
    assert m is not None, f"_PLUGIN_NAME_BRANCH_RE should match {line!r}"
    assert m.group(1) == lit


# ── (a)+(b) detection on synthetic files ────────────────────────────────────

def test_check_detects_target_backend_and_plugin_name(tmp_path):
    f = tmp_path / "fake_core.py"
    f.write_text(
        "def g(mode, backend, target, plugin):\n"
        '    if target == "a3":\n'
        "        return 1\n"
        '    if backend == "ascendc":\n'
        "        return 2\n"
        '    if plugin.name == "unsupported_mode":\n'
        "        return 3\n"
    )
    files = [str(f)]
    leaks = al.check_core_mode_leak(files)
    leak_lits = {x["mode"] for x in leaks}
    assert "a3" in leak_lits and "ascendc" in leak_lits
    pn = al.check_plugin_name_branch(files)
    assert len(pn) == 1 and pn[0]["mode"] == "unsupported_mode"
    # plugin.name is NOT double-counted by CORE_MODE_LEAK
    assert "unsupported_mode" not in leak_lits


# ── (c) registry-completeness on the live registry ──────────────────────────

def test_registry_completeness_passes_on_real_plugins():
    findings = al.check_plugin_registry_complete()
    assert findings == [], (
        f"all registered plugins should implement the required-method set; got: {findings}"
    )


def test_required_methods_match_conformance_test():
    # Lint (c)'s contract must stay in sync with the protocol-conformance test.
    sys.path.insert(0, str(_SCRIPTS / "orchestrator"))
    from plugins.tests import test_protocol_conformance as tpc  # type: ignore
    assert set(al.REQUIRED_PLUGIN_METHODS) == set(tpc.REQUIRED_METHODS), (
        "architecture_lint.REQUIRED_PLUGIN_METHODS drifted from "
        "test_protocol_conformance.REQUIRED_METHODS"
    )


# ── (e) FSM-hardcoded-transition ─────────────────────────────────────────────

def test_fsm_check_clean_on_current_state_machine():
    findings = al.check_fsm_hardcoded_transition()
    assert findings == [], (
        f"state_machine.py should derive transitions from the data table, not "
        f"hardcode them; got: {findings}"
    )


def test_fsm_check_flags_hardcoded_literal(tmp_path, monkeypatch):
    fake = tmp_path / "state_machine.py"
    fake.write_text(
        "def next_state(ws, cur, handoff):\n"
        "    target = trans.get('goto')          # legit: data-driven\n"
        "    if target == '__from_user_decision__':  # legit: magic token compare\n"
        "        pass\n"
        '    next_state = "await_worker"          # BAD: hardcoded routing\n'
        "    return next_state\n"
    )
    monkeypatch.setattr(al, "STATE_MACHINE_PY", str(fake))
    findings = al.check_fsm_hardcoded_transition()
    assert len(findings) == 1, findings
    assert findings[0]["target"] == "next_state"
    assert findings[0]["literal"] == "await_worker"


# ── (ii) --strict passes on current code with committed baseline ─────────────

def test_strict_passes_with_committed_baseline():
    r = subprocess.run(
        [sys.executable, str(_LINT_PY), "--strict", "--baseline", str(_BASELINE)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        f"--strict should pass on current code with the committed baseline.\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )


def test_baseline_grandfathers_every_current_finding():
    findings = al.collect_findings(200)
    baseline = getattr(al, '_load_baseline')(str(_BASELINE))
    new = [f for f in findings if getattr(al, '_key')(f) not in baseline]
    assert new == [], f"committed baseline must grandfather all current findings; new={new}"


def test_port_a3_plugin_name_debt_resolved():
    # DEBT-161 Batch-1 (2026-06-21): the one PLUGIN_NAME_BRANCH debt
    # (phase_o5_runner `_active_plugin.name == "port_a3_to_a5"`) was BURNED DOWN —
    # converted to a plugin method (BasePlugin.canonical_pass_a_skip_reason). It must
    # now be GONE: no PLUGIN_NAME_BRANCH finding anywhere, and absent from the baseline.
    findings = al.collect_findings(200)
    assert not [f for f in findings if f["check"] == "PLUGIN_NAME_BRANCH"], (
        "PLUGIN_NAME_BRANCH must be resolved (converted to a plugin method), not present: "
        f"{[f for f in findings if f['check'] == 'PLUGIN_NAME_BRANCH']}"
    )
    with open(_BASELINE, encoding="utf-8") as baseline_file:
        baseline_records = json.load(baseline_file)
    assert not any(r["check"] == "PLUGIN_NAME_BRANCH" for r in baseline_records), \
        "resolved PLUGIN_NAME_BRANCH must be dropped from the baseline (ratchet down)"


# ── (iii) a NEW injected violation is caught ─────────────────────────────────

def test_new_violation_caught_by_strict(tmp_path):
    # Inject a brand-new core leak into a copy of a real core file and confirm
    # --strict fails. Use a temp file added to the scan set via monkeypatch-free
    # approach: drop it under workflow/ would mutate the repo, so instead exercise
    # the in-process path with collect_findings against an injected file.
    # DEBT-161 Batch-2: the literal must be a supported target alias
    # (allowlist narrowing), so a core branch on `a3` is a leak.
    f = tmp_path / "injected_core.py"
    f.write_text('def h(target):\n    if target == "a3":\n        return 1\n')
    findings = al.check_core_mode_leak([str(f)])
    baseline = getattr(al, '_load_baseline')(str(_BASELINE))
    new = [x for x in findings if getattr(al, '_key')(x) not in baseline]
    assert len(new) == 1 and new[0]["mode"] == "a3"


def test_new_violation_caught_end_to_end(tmp_path):
    # End-to-end: append a new leak to a real core file, run the CLI --strict,
    # assert exit 1, then restore. Guarantees the regex + scan + baseline +
    # exit-code wiring all work together (not just the unit functions).
    target_file = _SCRIPTS / "orchestrator" / "events.py"
    original = target_file.read_text()
    try:
        # DEBT-161 Batch-2: literal must be a REAL backend (allowlist) — `ascendc` is a
        # known backend, so a NEW core branch on it IS a leak the strict gate must catch.
        target_file.write_text(original + '\nif backend == "ascendc":\n    pass\n')
        r = subprocess.run(
            [sys.executable, str(_LINT_PY), "--strict", "--baseline", str(_BASELINE)],
            capture_output=True, text=True,
        )
        assert r.returncode == 1, (
            f"--strict should fail on a NEW injected leak.\nstdout:\n{r.stdout}"
        )
        assert "ascendc" in r.stdout
    finally:
        target_file.write_text(original)


# ── DEBT-161 Batch-2: allowlist precision + staleness-guard ──────────────────

def test_overloaded_identifier_literals_not_flagged():
    """Precision: `mode`/`target` compared to NON-op-gen values are NOT leaks and must
    NOT be flagged (the false positives the allowlist narrowing removed): spawn mode,
    critic phase, FSM magic token.
    """
    import tempfile
    import os as _os
    with tempfile.TemporaryDirectory() as d:
        p = _os.path.join(d, "core.py")
        open(p, "w").write(
            'def f(mode, target):\n'
            '    if mode == "foreground": return 1\n'              # spawn mode — not op-gen
            '    if mode == "pre_agent_spawn": return 2\n'          # critic phase — not op-gen
            '    if target == "__from_user_decision__": return 3\n'  # FSM magic token — not arch
        )
        assert al.check_core_mode_leak([p]) == [], \
            "overloaded mode/target literals (non-op-gen) must not be flagged"


def test_genuine_mode_target_backend_literals_still_flagged():
    """Recall: REAL op-gen mode / backend / arch-target literals ARE still flagged."""
    import tempfile
    import os as _os
    with tempfile.TemporaryDirectory() as d:
        p = _os.path.join(d, "core.py")
        open(p, "w").write(
            'def f(mode, target, backend):\n'
            '    if mode == "port_a3_to_a5": return 1\n'
            '    if target == "a3": return 2\n'
            '    if backend == "ascendc": return 3\n'
            '    if mode == "optimize": return 4\n'
        )
        lits = {x["mode"] for x in al.check_core_mode_leak([p])}
        assert lits == {"port_a3_to_a5", "a3", "ascendc", "optimize"}, lits


def test_mode_leak_allowlist_superset_of_all_plugin_modes():
    """OL-160 staleness-guard: OPGEN_MODES must be a SUPERSET of every registered plugin's
    mode name — else a NEW backend/mode plugin's core leak would silently go unflagged (the
    allowlist lagging). Derived from the LIVE plugin registry, not a hand-list.
    """
    sys.path.insert(0, str(_SCRIPTS / "orchestrator"))
    from plugins import all_plugins  # noqa: E402
    plugin_modes = {p.name for p in all_plugins()}
    missing = plugin_modes - set(al.OPGEN_MODES)
    assert not missing, (
        f"OPGEN_MODES allowlist is STALE — missing registered plugin mode(s): {sorted(missing)}. "
        f"Add them to architecture_lint.OPGEN_MODES so core leaks on those modes stay flagged. "
        f"OPGEN_MODES={al.OPGEN_MODES}"
    )


# ── (iv) --changed-only non-bricking ratchet (DEBT-164) ──────────────────────

_REPO = _SCRIPTS.parent.parent  # repo root — findings are reported repo-relative from here


def test_partition_by_changed_splits_blocking_vs_warn():
    """Unit: a finding blocks only if its file is in the changed set; others warn."""
    new = [
        {"check": "GOD_FUNCTION", "file": "a/touched.py", "line": 1},
        {"check": "CORE_MODE_LEAK", "file": "b/untouched.py", "line": 2},
    ]
    blocking, warn = al.partition_by_changed(new, ["a/touched.py", "c/other.py"])
    assert [f["file"] for f in blocking] == ["a/touched.py"]
    assert [f["file"] for f in warn] == ["b/untouched.py"]


def test_partition_empty_changed_set_warns_everything():
    """Unit: with no changed files, nothing blocks — the non-bricking guarantee."""
    new = [{"check": "GOD_FUNCTION", "file": "x.py", "line": 1}]
    blocking, warn = al.partition_by_changed(new, [])
    assert blocking == [] and len(warn) == 1


def _run_changed_only(changed_files):
    return subprocess.run(
        [sys.executable, str(_LINT_PY), "--strict", "--baseline", str(_BASELINE),
         "--changed-only", *changed_files],
        capture_output=True, text=True, cwd=str(_REPO),
    )


def test_changed_only_blocks_when_violation_in_changed_file():
    """e2e: a NEW leak in events.py blocks when events.py is in the changed set."""
    target_file = _SCRIPTS / "orchestrator" / "events.py"
    rel = "src/scripts/orchestrator/events.py"
    original = target_file.read_text()
    try:
        # DEBT-161 Batch-2: literal must be an allowlisted backend (`ascendc`) so it IS a leak.
        target_file.write_text(original + '\nif backend == "ascendc":\n    pass\n')
        r = _run_changed_only([rel])
        assert r.returncode == 1, f"new leak in a changed file must block.\nstdout:\n{r.stdout}"
        assert "BLOCK" in r.stdout and "ascendc" in r.stdout
    finally:
        target_file.write_text(original)


def test_changed_only_allows_when_violation_not_in_changed_set():
    """e2e: the SAME new leak in events.py does NOT block when events.py is not in the
    changed set — a pre-existing violation elsewhere never bricks an unrelated commit.
    """
    target_file = _SCRIPTS / "orchestrator" / "events.py"
    original = target_file.read_text()
    try:
        # DEBT-161 Batch-2: allowlisted backend (`ascendc`) = a real leak; it's in events.py
        # which is NOT in the changed set, so it warns (doesn't block).
        target_file.write_text(original + '\nif backend == "ascendc":\n    pass\n')
        # changed set = some other real core file with no new violation
        r = _run_changed_only(["src/scripts/orchestrator/orchestrator.py"])
        assert r.returncode == 0, f"violation outside the changed set must NOT block.\nstdout:\n{r.stdout}"
        assert "warn" in r.stdout.lower() and "ascendc" in r.stdout
    finally:
        target_file.write_text(original)
