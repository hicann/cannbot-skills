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
r"""Guard: literal `kb/okf/...` and `kb/shared/...` paths named in code / prompts / cards
must exist on disk.

NOT every `kb/...` path: the trees this reorg DELETED (`kb/hardware/`, `kb/plugin-scope/`)
are deliberately out of the pattern. Measured 2026-09-05 —
`grep -rn "kb/hardware\|kb/plugin-scope"` over engine/agents/skills/workflows/hooks/kb/docs
(.py/.md/.sh/.yaml) finds 22 lines / 27 occurrences, all of them migration-ledger rows,
README history, `<!-- 迁移自 -->` provenance comments, or this docstring. Every one was
checked by hand; none is a live reference. Counts shift with the file types you include,
so quote the command with the number.
See `_SCAN_DIRS`/`_CANARIES` for what is actually covered and the exemption knobs below
for what is deliberately not.

WHY THIS EXISTS (2026-09-05). The `kb/okf/reference/` reorganization moved 545 cards.
Several consumers held the old paths hard-coded and were missed on the first pass:

  * TWO stale literals pointing at the pre-reorg `okf/reference/patterns/`:
    `kw_brief_fa.py:_A3_MIX_TEMPLATE_MD` (the path the composer asks for) and
    `kb_scope.py:_resolve_domain_template_path`'s hard-coded fallback (where the resolver
    looks when the direct lookup misses). Once the cards moved to
    `okf/reference/porter/patterns/`, both missed, so `kb_file_soc_families` returned None
    and the DEBT-208 SoC scope filter FAILED OPEN.
    Measured on each commit's own tree — `families=None, a3=True, a5=True` on e67c5561,
    9949b771 and 24f28622; `families={V220}, a5=False` from f4b41341 on. For the one card a
    composer actually gates (`fa_class_a3_mix_template.md`, a3-only) the leak therefore ran
    a3-template -> a5-worker.

    WHAT THIS GUARD DOES *NOT* CLAIM. An earlier draft of this docstring said nothing in
    the repo could report that leak. That was wrong on all three counts, measured on
    e67c5561: lint was FAIL (5452 blockers), the suite was 60 failed vs 45 on base, and
    `test_kw_brief_a3_fa_skeleton_delivery.py::test_a5_brief_does_not_gain_p116_or_pb55`
    — whose own docstring names this exact leak — was RED. Behavioural coverage existed
    and worked. What did not exist was anything that named the CAUSE: 45 pre-existing
    failures make 15 new ones easy to miss, and none of them says "this literal path is
    gone". That is the gap this file closes — a cheap, specific, cause-level signal, not
    a substitute for the behavioural test.
  * `plugins/port_a3/migration_level.py` — emitted `l1-implementation-guide.md` etc.,
    consumed by `kw_brief_pa3_phases.py` which prefixed `kb/okf/reference/migration/`.
  * `briefs/kw_brief_fa.py`, `briefs/kw_brief.py`, `workflow/workflow_critic_validators.py`,
    `skills/aog-op-classify/validate_kb_paths.py` — literal card paths in brief text.

None of these produce a lint error or a test failure on their own: the brief still renders,
it just tells the worker to read a file that is not there. This test makes that class of
breakage loud.

SCOPE: static scan of literal paths. It cannot see paths assembled at runtime from
non-literal parts — those still need their own tests (see `test_kb_scope.py` for the
domain-template resolver).
"""
import collections
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../engine/src/scripts/orchestrator/tests/ut -> plugin root (ut,tests,orchestrator,scripts,src,engine)
PLUGIN_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 6)))
KB_ROOT = os.path.join(PLUGIN_ROOT, "kb")

# Three literal forms actually emitted by consumers, each resolved against the bases below:
#   (A) plugin-root-relative `kb/okf/...` / `kb/shared/...` — prompts and brief prose.
#   (B) kb-root-relative `"okf/..."` — code, where the loader adds `kb/` (`phase_o0.py:55`).
#       This is the form that broke in the reorg (`kb_scope.py` held a bare `okf/reference/...`),
#       so a `kb/`-only regex would not have seen it.
#   (C) okf-root-relative `"runbooks/..."` — cards citing sibling cards without the `okf/` prefix
#       (e.g. `runbooks/hardware/probe-...md:108`).
# (B)/(C) are matched ONLY when fully delimited by a quote/backtick, so prose mentioning a path
# mid-sentence is not a false positive; that delimiting is also what makes an extensionless
# directory path safe to accept.
#
# `okf/` is NOT globally unambiguous — `engine/src/scripts/okf/` is a real source directory — so
# form (B) additionally requires the second segment to be an OKF tree (`reference`/`runbooks`/
# `ops`/`search`/`graph`) or `index.md`. That is what keeps `"okf/okf_kb.sh"` from being reported.
_PATH_RE_PREFIXED = re.compile(
    r"kb/((?:okf|shared)/[A-Za-z0-9._/\-]+?(?:\.(?:md|json|jsonl|txt|ya?ml)|/))(?=[\s`\"'),\]]|$)"
)
_PATH_RE_KB_RELATIVE = re.compile(
    r"""(?P<q>["'`])(okf/(?:(?:reference|runbooks|ops|search|graph)/[A-Za-z0-9._/\-]*?|index\.md))/?(?P=q)"""
)
# Form (C) is the card-to-card citation form and is scanned ONLY inside `kb/` cards.
# Elsewhere a bare `runbooks/...` is usually a synthesised test fixture id
# (`test_okf_reference_block.py` invents `runbooks/field-notes/build/pb-99-old.md`),
# which is data for the test, not a reference to a file that must exist.
_PATH_RE_OKF_RELATIVE = re.compile(
    r"""(?P<q>["'`])(runbooks/[A-Za-z0-9._/\-]+?)/?(?P=q)"""
)
# `shared/` is NOT unambiguous either: CANN's own source tree has a `shared/common/`, and FA-class
# prose quotes it (`porter/patterns/fa_class_template.md:560`). Restrict it to code, where a bare
# `shared/...` string is by construction a kb-relative path the loader prefixes.
_PATH_RE_KB_RELATIVE_SHARED = re.compile(
    r"""(?P<q>["'`])(shared/[A-Za-z0-9._/\-]+?)/?(?P=q)"""
)
_FORM_C_SOURCE_ROOT = "kb/"
# Deliberately a LINE-SHAPE test, not a substring test. Loosening it to `"迁移自" in line`
# would exempt any line that merely mentions provenance — including one that also carries a
# live path. Pinned below, because pinning `_SKIP_FILES`'s value does not pin this rule.
_PROVENANCE_LINE_PREFIX = "<!-- 迁移自"


def _is_provenance_line(line: str) -> bool:
    """Whole line is a provenance comment, so the deleted path it names is not a claim."""
    return line.lstrip().startswith(_PROVENANCE_LINE_PREFIX)


def _strip_provenance(line: str) -> str:
    """Drop a leading provenance comment, KEEPING whatever follows it on the same line.

    Skipping the whole line would exempt a live path that happens to share it —
    `<!-- 迁移自 kb/hardware/old.md --> 请读 kb/okf/reference/missing.md` must still
    report the second path.
    """
    if not _is_provenance_line(line):
        return line
    end = line.find("-->")
    return "" if end < 0 else line[end + 3:]

# CANN's own source tree also has a `shared/...`; card prose quotes it. Excluded by PREFIX,
# not by file: excluding a whole card would permanently unguard every real `kb/shared/X.md`
# reference later added to that card.
_CANN_SOURCE_PREFIXES = ("shared/common/",)


# kb/ and docs/ are in scope because prose there is load-bearing too: `kb/shared/GATE_CONTRACT.md`
# is injected into every port-a3 brief as MANDATORY (`briefs/kw_brief_port_a3.py:534`), so a stale
# path in it misroutes a worker exactly like a stale path in code does.
_SCAN_DIRS = ("engine/src/scripts", "agents", "skills", "workflows", "hooks", "kb", "docs")
_SCAN_EXT = (".py", ".sh", ".md", ".yaml", ".yml")

# Paths that are documented as historical / aspirational rather than live references.
_ALLOWED_ABSENT = (
    "kb/okf/search/",          # build product, gitignored
    "kb/okf/graph/",           # okf_graph judgment cache, absent until judge runs
)

# Basenames used as metavariables in docstrings / deliberately-absent test fixtures.
# Listed explicitly rather than pattern-guessed, so adding one is a visible decision.
# Each entry is a (source file suffix, basename) pair, scoped to the ONE file that
# legitimately names it, so the same basename elsewhere is still reported. The paths are
# exact and repo-relative: `endswith` matching would let a nested lookalike such as
# `agents/evil/briefs/kb_scope.py` inherit the exemption.
_DOC_EXAMPLES = {
    ("engine/src/scripts/orchestrator/briefs/kb_scope.py", "X.md"),
    ("engine/src/scripts/orchestrator/tests/ut/test_kb_scope.py", "does_not_exist.md")
}

# Files whose matches are provenance notes about where content CAME FROM.
# Matched as an exact path or a path PREFIX (see `_is_skipped_file`), never a substring.
# The one-shot `okf/convert_*.py` migration scripts used to need an exemption here (their
# string literals named the deleted source tree); they were removed 2026-09-05, and the
# exemption went with them — an exemption that outlives its subject is a blind spot.
_SKIP_FILES = (
    "engine/src/scripts/orchestrator/tests/ut/test_kb_paths_exist.py",
)


def _is_cann_source(kb_rel: str) -> bool:
    """CANN's own `shared/common/...` tree, not `kb/shared/`. Directory-boundary match:
    a bare `shared/common` (the regex strips the trailing slash) counts, but
    `shared/commonwealth/x.md` must not."""
    return any(kb_rel == pfx.rstrip("/") or kb_rel.startswith(pfx) for pfx in _CANN_SOURCE_PREFIXES)


def _is_skipped_file(rel_src: str) -> bool:
    """Substring matching here would skip any file whose path merely CONTAINS a marker,
    and a skipped file is not checked at all."""
    return any(rel_src == m or rel_src.startswith(m) for m in _SKIP_FILES)


def _iter_tree_source_files(base):
    """Scannable files under one root, skipping .git."""
    for root, _dirs, files in os.walk(base):
        if os.sep + ".git" in root:
            continue
        for fn in files:
            if fn.endswith(_SCAN_EXT):
                yield os.path.join(root, fn)


def _iter_source_files():
    for rel in _SCAN_DIRS:
        base = os.path.join(PLUGIN_ROOT, rel)
        if os.path.isdir(base):
            yield from _iter_tree_source_files(base)


def _collect():
    """{(source_file, lineno, kb_relpath, form)} for every literal KB path found."""
    out = set()
    for path in _iter_source_files():
        rel_src = os.path.relpath(path, PLUGIN_ROOT)
        if _is_skipped_file(rel_src):
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for kb_rel, form in _scan_line(rel_src, line):
                out.add((rel_src, lineno, kb_rel, form))
    return out


def _scan_line(rel_src: str, line: str):
    """(path, form) pairs in ONE line. Split out so the skip/exemption semantics can be
    asserted directly — mutating this call site is what escapes a helper-level test."""
    found = set()
    line = _strip_provenance(line)
    for m in _PATH_RE_PREFIXED.finditer(line):
        found.add((m.group(1), "A"))
    for m in _PATH_RE_KB_RELATIVE.finditer(line):
        found.add((m.group(2), "B"))
    for m in _PATH_RE_KB_RELATIVE_SHARED.finditer(line):
        if not _is_cann_source(m.group(2)):
            found.add((m.group(2), "S"))
    if rel_src.startswith(_FORM_C_SOURCE_ROOT):
        for m in _PATH_RE_OKF_RELATIVE.finditer(line):
            found.add((m.group(2), "C"))
    return found


# Every entry below WIDENS the guard's blind spot. Measured 2026-09-05: each of these knobs
# can be widened by one line to swallow a real dead path while all other assertions stay
# green — e.g. adding `any(c.isdigit() for c in kb_rel)` to `_is_placeholder` exempts 75/310
# matches. `test_exemption_surface_is_pinned` pins their VALUES;
# `test_every_exemption_rule_is_pinned_by_behaviour_not_just_its_constant` pins how they
# are APPLIED — codex rounds 12-13 showed that pinning only the values is not enough.
_PLACEHOLDER_MARKERS = ("{", "<", "*", "...")


def _is_placeholder(kb_rel):
    """A metavariable or elided path, not something that should exist on disk."""
    return any(marker in kb_rel for marker in _PLACEHOLDER_MARKERS)


# Which roots a form's paths may legitimately resolve against. Form (A) carries its own
# `kb/` prefix so it is plugin-root-relative; (B)/(S) are kb-root-relative; (C) is written
# relative to `kb/okf`. Giving every form both bases let `runbooks/X.md` be satisfied by a
# stray `kb/runbooks/X.md` that no consumer would ever load.
_FORM_BASES = {"A": ("",), "B": ("",), "S": ("",), "C": ("okf",)}


def _exists(kb_rel, form="A"):
    """True only for a path that resolves to a real entry INSIDE kb/, for that form.

    `os.path.isfile` alone accepts a `..` that the OS normalises back out of `kb/`, a
    symlink whose target sits outside it, and a directory named `foo.md`. All three would
    make a broken reference look fine.
    """
    # No consumer writes `..` in a KB path. Rejecting it outright is the real invariant:
    # `okf/../shared/X.md` normalises to a file that DOES exist under kb/, so a containment
    # check alone would wave it through while no loader would ever resolve it that way.
    if ".." in kb_rel.split("/"):
        return False
    kb_real = os.path.realpath(KB_ROOT)
    for base in _FORM_BASES.get(form, ("",)):
        target = os.path.join(KB_ROOT, base, kb_rel) if base else os.path.join(KB_ROOT, kb_rel)
        real = os.path.realpath(target)
        if real != kb_real and not real.startswith(kb_real + os.sep):
            continue  # `..` or a symlink escaped kb/
        # Forms (B)/(C) strip the trailing slash, so "looks like a directory" is
        # "ends with / OR its last segment has no extension".
        last = kb_rel.rstrip("/").rsplit("/", 1)[-1]
        if kb_rel.endswith("/") or "." not in last:
            if os.path.isdir(real):
                return True
        elif os.path.isfile(real):
            return True
    return False


def _missing_from(entries):
    """The filter+resolve half of the guard, isolated so a negative canary can drive it.

    `_CANARIES` only proves `_collect()` still matches. Nothing proved that what it
    collects is still CHECKED: making `_exists()` return True, or adding `kb/okf/` to
    `_ALLOWED_ABSENT`, or widening `_RESOLVE_BASES`, turns the assertion below into a
    tautology while every canary stays green. `test_a_dead_path_is_reported` closes that.
    """
    missing = []
    for rel_src, lineno, kb_rel, _form in sorted(entries):
        if any(kb_rel.startswith(a[len("kb/"):]) for a in _ALLOWED_ABSENT):
            continue
        if _is_placeholder(kb_rel):
            continue  # template placeholder / elided path, not a concrete one
        base = kb_rel.rsplit("/", 1)[-1]
        if (rel_src, base) in _DOC_EXAMPLES:
            continue
        if not _exists(kb_rel, _form):
            missing.append(f"{rel_src}:{lineno} -> kb/{kb_rel}")
    return missing


def test_every_literal_kb_path_exists_on_disk():
    """A brief that names a nonexistent KB file is a silent worker-side failure."""
    missing = _missing_from(_collect())
    assert not missing, (
        "%d KB path(s) named in source/prompts do not exist on disk:\n  %s"
        % (len(missing), "\n  ".join(missing))
    )


# Paths that certainly do not exist, one per shape the filter handles.
# The last entry catches ONE specific widening: adding "okf/reference" as a base makes it
# resolve. It does NOT pin the base list in general -- measured, adding "okf/runbooks" or
# "shared" leaves every assertion here green; `test_exemption_surface_is_pinned` does that.
_DEAD_FIXTURES = (
    ("agents/not-a-real-file.md", 1, "okf/reference/porter/patterns/zz_definitely_absent.md", "A"),
    ("agents/not-a-real-file.md", 2, "okf/reference/zz-absent-dir/", "A"),
    ("briefs/not_a_real_file.py", 3, "shared/ZZ_ABSENT.md", "S"),
    ("kb/not-a-real-card.md", 4, "runbooks/zz-absent/zz.md", "C"),
    ("agents/not-a-real-file.md", 5, "porter/index.md", "A"),
)


def test_exemption_surface_is_pinned():
    """Pin the constants that can make this guard blind. See the note above `_is_placeholder`.

    Split across three tests (this one, `test_scan_surface_is_pinned`,
    `test_regex_alternations_and_exemption_tables_are_pinned`) purely for size; together
    they are the one surface, and loosening any single knob must fail exactly one of them.
    """
    assert _FORM_BASES == {"A": ("",), "B": ("",), "S": ("",), "C": ("okf",)}, (
        "_FORM_BASES changed to %r. Each extra base per form makes more wrong-shaped "
        "paths 'resolve' — a form-(C) `runbooks/X.md` must not be satisfied by a stray "
        "`kb/runbooks/X.md`." % (_FORM_BASES,)
    )
    assert _PLACEHOLDER_MARKERS == ("{", "<", "*", "..."), (
        "_PLACEHOLDER_MARKERS widened to %r — every marker exempts real paths that "
        "happen to contain it." % (_PLACEHOLDER_MARKERS,)
    )
    assert _ALLOWED_ABSENT == ("kb/okf/search/", "kb/okf/graph/"), (
        "_ALLOWED_ABSENT widened to %r; only gitignored build products belong here."
        % (_ALLOWED_ABSENT,)
    )
    assert _SKIP_FILES == (
        "engine/src/scripts/orchestrator/tests/ut/test_kb_paths_exist.py",
    ), (
        "_SKIP_FILES widened to %r — a skipped file's paths are not checked at all."
        % (_SKIP_FILES,)
    )
    assert _PROVENANCE_LINE_PREFIX == "<!-- 迁移自", (
        "_PROVENANCE_LINE_PREFIX changed to %r — this is a line-START test on purpose."
        % (_PROVENANCE_LINE_PREFIX,)
    )
    assert _FORM_C_SOURCE_ROOT == "kb/", (
        "_FORM_C_SOURCE_ROOT narrowed to %r — the only form (C) canary lives under "
        "kb/okf/, so narrowing to that prefix drops every kb/shared card silently."
        % (_FORM_C_SOURCE_ROOT,)
    )


def test_scan_surface_is_pinned():
    """Which roots and extensions get scanned at all. Part 2 of the exemption surface."""
    assert _SCAN_DIRS == (
        "engine/src/scripts", "agents", "skills", "workflows", "hooks", "kb", "docs",
    ), (
        "_SCAN_DIRS changed to %r. Dropping a root scans nothing under it, and the per-root "
        "floors below only cover the roots that carry many paths — `workflows`/`hooks` have "
        "too few for a floor, so only this pin protects them." % (_SCAN_DIRS,)
    )
    assert _SCAN_EXT == (".py", ".sh", ".md", ".yaml", ".yml"), (
        "_SCAN_EXT changed to %r. Dropping an extension silently stops scanning that file "
        "type; no canary is anchored in a .sh/.yaml file, so nothing else would notice."
        % (_SCAN_EXT,)
    )


def test_regex_alternations_and_exemption_tables_are_pinned():
    """Each alternation branch pinned by NAME, plus the two (file, name) exemption tables.

    Part 3 of the exemption surface. A floor cannot see one branch of an alternation
    disappear (form B keeps its floor of 10 after losing `ops`), and a canary only covers
    the branch it happens to sit on.
    """
    assert set(_PATH_RE_PREFIXED.pattern.split("(?:md|")[1].split(")")[0].split("|")) == {
        "json", "jsonl", "txt", "ya?ml"
    }, "form (A) extension alternation changed — %s" % _PATH_RE_PREFIXED.pattern
    for tree in ("reference", "runbooks", "ops", "search", "graph"):
        assert "|%s|" % tree in "|%s|" % "|".join(
            _PATH_RE_KB_RELATIVE.pattern.split("(?:reference")[1].split(")/")[0].lstrip("|").split("|")
            + ["reference"]
        ), "form (B) lost the %r tree from its alternation" % tree
    assert _CANN_SOURCE_PREFIXES == ("shared/common/",), (
        "_CANN_SOURCE_PREFIXES widened to %r; only CANN's own source tree belongs here."
        % (_CANN_SOURCE_PREFIXES,)
    )
    assert _DOC_EXAMPLES == {
        ("engine/src/scripts/orchestrator/briefs/kb_scope.py", "X.md"),
        ("engine/src/scripts/orchestrator/tests/ut/test_kb_scope.py", "does_not_exist.md"),
    }, (
        "_DOC_EXAMPLES changed to %r — each entry exempts a (file, basename) pair. "
        "Pinning only its LENGTH was not enough: swapping an entry for "
        "('', 'zz-absent-card.md') keeps len == 2 while exempting that basename "
        "everywhere, and a real dead path with that name then goes unreported."
        % (sorted(_DOC_EXAMPLES),)
    )


def test_every_exemption_rule_is_pinned_by_behaviour_not_just_its_constant():
    """Pinning a constant never pins how it is APPLIED — the lesson of codex rounds 12-13.

    Each assertion below corresponds to a one-line loosening that leaves every constant
    untouched (so `test_exemption_surface_is_pinned` stays green) while swallowing a real
    dead path. They were all measured green before these assertions existed.
    """
    # `_SKIP_FILES` matched as a substring would skip any path merely CONTAINING a marker.
    assert _is_skipped_file(
        "engine/src/scripts/orchestrator/tests/ut/test_kb_paths_exist.py"
    )
    assert not _is_skipped_file(
        "agents/orchestrator/tests/ut/test_kb_paths_exist.py"
    ), "a live consumer whose path merely ENDS with the marker must still be scanned"

    # `_DOC_EXAMPLES` matched with endswith would let a nested lookalike inherit it.
    real = ("engine/src/scripts/orchestrator/briefs/kb_scope.py", 1,
            "okf/reference/porter/patterns/X.md", "A")
    fake = ("agents/evil/briefs/kb_scope.py", 1,
            "okf/reference/porter/patterns/X.md", "A")
    assert _missing_from({real}) == []
    assert _missing_from({fake}), "a nested lookalike path must not inherit the exemption"

    # Provenance must strip the COMMENT, not the line — asserted through `_scan_line`,
    # the actual call site. A helper-level assertion here stayed green when the call site
    # was mutated back to `continue`.
    shared_line = "<!-- 迁移自 kb/hardware/old.md --> 请读 `kb/okf/reference/missing.md`"
    assert ("okf/reference/missing.md", "A") in _scan_line("agents/x.md", shared_line), (
        "dropping the whole line exempts a live path that merely shares it with a "
        "provenance comment"
    )
    assert _scan_line("agents/x.md", "<!-- 迁移自 kb/hardware/old.md -->") == set(), (
        "a pure provenance comment must still contribute nothing"
    )

    # `_exists` must reject `..` outright. `okf/../shared/X.md` normalises to a file that
    # really is there, so a containment check alone waves it through — but no loader
    # resolves paths that way, so accepting it hides a broken reference.
    assert _exists("shared/GATE_CONTRACT.md", "A"), "sanity: the real file resolves"
    assert not _exists("okf/../shared/GATE_CONTRACT.md", "A")
    # ...and a symlink escaping kb/ must not count either (containment, still needed).
    assert not _exists("okf/zz-no-such-entry.md", "A")

    # `_CANN_SOURCE_PREFIXES` must match on the directory boundary.
    assert _is_cann_source("shared/common") and _is_cann_source("shared/common/util.h")
    assert not _is_cann_source("shared/commonwealth/x.md"), "prefix without a boundary"


def test_provenance_skip_is_a_line_shape_not_a_substring():
    """Pinning the marker's VALUE does not pin how it is APPLIED.

    Loosening `startswith` to `in` exempts any line that merely mentions provenance —
    including one that also carries a live path — and no constant changes, so
    `test_exemption_surface_is_pinned` stays green. This asserts the shape directly.
    """
    assert _is_provenance_line("<!-- 迁移自 kb/hardware/x.md -->")
    assert _is_provenance_line("    <!-- 迁移自 kb/hardware/x.md -->"), "leading space is fine"
    assert not _is_provenance_line(
        "See `kb/okf/reference/porter/index.md` <!-- 迁移自 kb/hardware/x.md -->"
    ), "a line that CONTAINS the marker but does not start with it still carries a claim"


def test_doc_example_exemption_is_scoped_to_its_one_file():
    """Pinning `_DOC_EXAMPLES`'s VALUE does not pin that it is applied per-(file, basename).

    Broadening the check to "basename is in _DOC_EXAMPLES" — with the constant untouched —
    would exempt a real dead card that happens to be named `X.md` anywhere in the tree.
    """
    exempt = ("engine/src/scripts/orchestrator/briefs/kb_scope.py", 1,
              "okf/reference/porter/patterns/X.md", "A")
    assert _missing_from({exempt}) == [], "the documented metavariable must stay exempt"
    elsewhere = ("agents/aog-kernel-worker.md", 1, "okf/reference/porter/patterns/X.md", "A")
    assert _missing_from({elsewhere}), (
        "the same basename in ANOTHER file must still be reported — the exemption is "
        "(file, basename), not basename"
    )


def test_symlink_escaping_kb_is_not_treated_as_existing(tmp_path):
    """A link inside kb/ pointing OUTSIDE it must not make a reference look valid.

    `os.path.isfile` follows links, so without the containment check a card could
    "exist" by pointing at any file on the machine.
    """
    outside = tmp_path / "outside.md"
    outside.write_text("not part of the KB", encoding="utf-8")
    link = os.path.join(KB_ROOT, "okf", "zz_escape_test_link.md")
    if os.path.lexists(link):
        pytest.skip("fixture name collides with a real path")
    try:
        os.symlink(str(outside), link)
    except OSError:
        pytest.skip("cannot create symlinks here")
    try:
        assert os.path.isfile(link), "sanity: the link resolves to a real file"
        assert not _exists("okf/zz_escape_test_link.md", "A"), (
            "a symlink whose target sits outside kb/ was accepted as existing"
        )
    finally:
        os.unlink(link)


def test_dangling_symlink_is_not_treated_as_existing(tmp_path):
    """`_exists` must use a following stat, never `lexists`.

    A card deleted out from under a symlink leaves the link behind; `os.path.lexists`
    would call that path present and the brief would send a worker to a broken link.
    """
    link = os.path.join(KB_ROOT, "okf", "zz_dangling_test_link.md")
    if os.path.lexists(link):
        pytest.skip("fixture name collides with a real path")
    try:
        # Target sits INSIDE kb/ so the containment check cannot mask the question:
        # the only thing separating "exists" from "does not" here is following the link.
        os.symlink(os.path.join(KB_ROOT, "okf", "zz_no_such_target.md"), link)
    except OSError:
        pytest.skip("cannot create symlinks here")
    try:
        assert os.path.lexists(link), "fixture did not take effect"
        assert not _exists("okf/zz_dangling_test_link.md"), (
            "a dangling symlink was reported as existing — `_exists` follows links by "
            "design; switching it to lexists() would silently accept broken links"
        )
    finally:
        os.unlink(link)


@pytest.mark.parametrize("entry", _DEAD_FIXTURES)
def test_a_dead_path_is_reported(entry):
    """Negative canary: the filter must still REJECT. Guards the half canaries can't see."""
    reported = _missing_from({entry})
    assert len(reported) == 1, (
        "a path that does not exist was not reported: %r -> %r. `_exists`, "
        "`_ALLOWED_ABSENT`, `_RESOLVE_BASES` or a placeholder rule went permissive, which "
        "makes test_every_literal_kb_path_exists_on_disk vacuous." % (entry, reported)
    )


# Concrete matches that MUST be found, one per form, each chosen to exercise the part of
# its regex most likely to be narrowed by a well-meaning edit. Floors alone are not enough:
# deleting the `-` from form (A)'s character class drops it 240 -> 185, which clears a floor
# of 50 while silently unguarding every hyphenated card path -- the pb, ol, target and
# asc-devkit-vendored prefixes, i.e. most of the KB.
#
# Why each entry earns its place. Kept OUT of the literal below because CodeCheck's
# commented-out-code heuristic reads a comment sitting inside a bracketed literal as dead
# code (PR 1005 r1 and r2 both reported one, the second right after the first was reworded):
#
#   B / aog-op-classify  the `runbooks` branch of form (B)'s alternation. Without it,
#                        deleting that one branch leaves the `reference` canary happy.
#   S / aog-op-classify  a MARKDOWN prompt. The code-only S canary stays green if form (S)
#                        is narrowed back to `_CODE_EXT`, which is exactly how a dead
#                        `hardware/HIASCEND_DOC_URLS` survived.
#   A / kw_brief.py      form (A)'s `shared` alternative: 34/240 of its matches, and the
#                        MANDATORY injection surface.
#   A / aog-kernel-worker  form (A)'s trailing-slash directory form: 91/240 -- and a reorg
#                        is precisely what deletes directories.
#   B / phase_o0.py      form (B)'s `index.md` alternative: 2 matches, but its own branch.
#   S / ALWAYS_LOADED_RULES  form (S) inside a CARD. Two kb/shared cards cite each other
#                        this way and both are MANDATORY brief injections; excluding all of
#                        `kb/` from form (S) unguards them while the canaries above stay green.
_CANARIES = (
    ("A", "agents/aog-kernel-optimizer.md", "okf/runbooks/hardware/target-ascend950pr.md"),
    ("B", "agents/aog-researcher.md", "okf/reference/porter/handbook/language_reference.md"),
    ("B", "skills/aog-op-classify/SKILL.md",
          "okf/runbooks/field-notes/precision/ol-103-npu-transcendentals-fp16-precision.md"),
    ("S", "briefs/op_taxonomy.py", "shared/ALWAYS_LOADED_RULES.md"),
    ("S", "skills/aog-op-classify/SKILL.md", "shared/HIASCEND_DOC_URLS.md"),
    ("C", "kb/okf/runbooks/hardware/probe-2026-04-21-q-instruction-cycles.md",
          "runbooks/hardware/target-ascend950pr.md"),
    ("A", "briefs/kw_brief.py", "shared/GATE_CONTRACT.md"),
    ("A", "agents/aog-kernel-worker.md", "okf/runbooks/"),
    ("B", "orchestrator/phase_o0.py", "okf/index.md"),
    ("S", "kb/shared/ALWAYS_LOADED_RULES.md", "shared/KERNEL_AUTHORING_GUARDS.md"),
)


@pytest.mark.parametrize("form,src_suffix,kb_rel", _CANARIES)
def test_canary_match_is_still_found(form, src_suffix, kb_rel):
    """A known match per form. Narrowing a regex until these vanish fails loudly."""
    found = {(src, path, f) for src, _l, path, f in _collect()}
    assert any(src.endswith(src_suffix) and path == kb_rel and f == form for src, path, f in found), (
        "form (%s) canary not found: %r in %s.\n"
        "EITHER the regex/skip rules for form (%s) were narrowed — in which case everything "
        "written in that form is now unguarded and the regex is the bug —\n"
        "OR that reference was legitimately edited away (these canaries point at live prose "
        "and KB cards, which `aog-knowledge-maintain` rewrites): then re-point the canary at "
        "another instance of the SAME form, do not delete it."
        % (form, kb_rel, src_suffix, form)
    )


def test_no_doubled_kb_prefix():
    """A doubled `kb/` prefix resolves (form A matches the inner half) but is a broken literal.

    Introduced by a path-rewrite pass that prefixed `kb/` onto an already-prefixed path;
    the existence check cannot see it, so it needs its own assertion.
    """
    bad = []
    needle = "kb/" + "kb/"   # split so this file's own lines are not a self-match
    for path in _iter_source_files():
        rel_src = os.path.relpath(path, PLUGIN_ROOT)
        if any(marker in rel_src for marker in _SKIP_FILES):
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if needle in line:
                bad.append(f"{rel_src}:{lineno}")
    assert not bad, "doubled kb/ prefix at:\n  " + "\n  ".join(bad)


def test_scan_actually_finds_paths():
    """Guard the guard: every form and every scan root must contribute, INDIVIDUALLY.

    These floors catch a form or a root collapsing to (near) zero. They CANNOT catch a
    regex that is narrowed but still matches a lot — `_CANARIES` above covers that.
    """
    found = _collect()
    by_form = collections.Counter(form for _s, _l, _p, form in found)
    by_root = collections.Counter(src.split("/")[0] for src, _l, _p, _f in found)

    # counts on 2026-09-05: A=240 B=49 S=15 C=6 (total 310); by root: engine 92, agents 77,
    # skills 50, kb 71, docs 19, workflows 1, hooks 0
    for form, floor in (("A", 50), ("B", 10), ("S", 3), ("C", 1)):
        assert by_form[form] >= floor, (
            "form (%s) contributed %d matches (floor %d) — that regex is broken. All: %r"
            % (form, by_form[form], floor, dict(by_form))
        )
    # A mistyped scan root would silently scan nothing; assert the directories are real.
    for root in _SCAN_DIRS:
        assert os.path.isdir(os.path.join(PLUGIN_ROOT, root)), \
            f"_SCAN_DIRS names {root!r}, which is not a directory — that root scans nothing"
    # Roots that carry KB references must keep carrying them. `hooks/` is deliberately absent:
    # it holds no KB path today, so a floor there would be a permanent false alarm.
    for root, floor in (("engine", 40), ("agents", 20), ("skills", 10), ("kb", 20), ("docs", 5)):
        assert by_root[root] >= floor, (
            "scan root %r contributed %d matches (floor %d) — a skip rule or _SCAN_DIRS "
            "regressed and paths under it are now unguarded. All: %r"
            % (root, by_root[root], floor, dict(by_root))
        )


@pytest.mark.parametrize("required", [
    "okf/index.md",
    "okf/reference/index.md",
    "shared/ALWAYS_LOADED_RULES.md",
    "shared/ANTI_PRESSURE_PROTOCOLS.md",
    "shared/HIASCEND_DOC_URLS.md",
    "CONVENTIONS.md",
])
def test_load_bearing_kb_files_present(required):
    """Files that briefs mark MANDATORY, plus the conventions rulebook."""
    assert os.path.isfile(os.path.join(KB_ROOT, required)), f"missing kb/{required}"
