# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SoC-scope predicate for KB-derived brief injections (DEBT-208).

Every KB entry already declares its own machine-readable scope:

    `applies_to: soc=Ascend910_9382 (V220 A2/A3 single-die); cann=9.0.0+; ...`

Brief composers used to hardcode injection and carry the scope as PROSE inside
the injected text ("V351/A5 scope bound — do NOT over-apply"), i.e. the bound
held only if the reading LLM complied. That is backwards from this repo's
structurally-enforced > compliance principle, and it inverted the KB's own
advice: PB-34 is `soc=Ascend910_9382` (V220) with two
`verified_does_not_reproduce_on: Ascend950PR` witnesses, yet its deadlock
warning was composed unconditionally into the A5 forward-FA brief — steering A5
workers away from the very light-port PB-34 recommends for them.

This module makes the scope a CODE predicate: a composer asks whether an entry
applies to the target it is briefing for, and the answer is read from the KB
entry itself. Adding a SoC bound to a KB entry is then enough to bound every
composer that cites it — no composer edit, no per-entry `if PB-34` patch.

Reused, NOT reimplemented — the SoC parsing comes from `kb_index_audit`
(`_applies_to_socs` / `_soc_family` / `_ANY_ENTRY_HEAD_RE` / `_APPLIES_FIELD_RE`).
That module's hard-failing `confirmed_on ⊆ applies_to` lint is the OTHER half of
this feature (it is what keeps an entry's declared scope honest, so honoring the
scope is safe), and it already solved the two hard parts:

  - **arch-family granularity** — the KB spells two families ~20 ways
    (`Ascend910_9382` / `Ascend910C` / `V220`; `Ascend950PR` / `Ascend950PR_9579`
    / `V351`). Exact-token comparison false-positives on every entry that
    declares `soc=Ascend950PR` and verifies on `Ascend950PR_9579`.
  - **identifying-position reads** — a naive prose scan red-flags `V220→A5 port`
    DIRECTIONS and cross-ref IDs.

A second, subtly-different parser here would be its own rot: the two would drift
and the lint would stop describing what the composers actually do.

FAIL-OPEN by construction (`kb_applies_to_target`): an injection is suppressed
ONLY on a positive, machine-readable exclusion — the entry declares a `soc=`
scope, the target's family is known, and the family is not in the scope.
Unknown entry / unparseable scope / `soc=all` / unknown target all keep the
injection, so this can only ever narrow an over-block, never create a new
under-block. The `applies_to`-vs-`confirmed_on` contradiction that PB-35 carried
until 2026-07-17 (declared V220-only, CONFIRMED on A5) is the shape that WOULD
turn this into an under-block; `kb_index_audit.check_soc_scope` now fails hard on
that shape, which is why honoring `applies_to` is safe to do mechanically.
"""
from __future__ import annotations
import logging

import re
import sys
from pathlib import Path
from typing import Optional

from kb_paths import kb_root

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent.parent  # → repo root

# Target name (env.target / `TARGET=`) → coarse SoC arch family, the granularity
# `kb_index_audit._soc_family` normalizes to. Mirrors op_taxonomy.TARGET_HW_SPEC_MAP
# (a5 → ascend950pr = V351 / arch35; a3 → ascend910c and a2 → ascend910b, both
# V220 single-die), which is the established target→hardware mapping here.
SOC_FAMILY_BY_TARGET: dict[str, str] = {
    "a5": "V351",
    "a3": "V220",
    "a2": "V220",
}

# `### 4. RUNNABLE deadlock-avoiding handshake ...` — doc-section anchors that
# carry their own `applies_to:` line but no EC/PB/OL-style entry id.
_SECTION_HEAD_RE = re.compile(r"^#{2,4}\s+(\d+)\.\s")


def _kb_index_audit():
    """Import `kb_index_audit` (lives in src/scripts/, not under orchestrator/).

    Lazy + sys.path-on-demand, matching the established cross-package precedent
    in `finalize_checks_structural._pp88_gate`. Returns None if unavailable so
    callers fail OPEN rather than dropping an injection on an importer error.
    """
    try:
        scripts_dir = _PROJECT_ROOT / "src" / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import kb_index_audit as _audit  # type: ignore

        return _audit
    except Exception:
        return None


def soc_family_for_target(target: Optional[str]) -> Optional[str]:
    """Coarse SoC family (`V220` / `V351`) for a build target, else None."""
    if not target:
        return None
    return SOC_FAMILY_BY_TARGET.get(str(target).strip().lower())


def _iter_kb_files(audit) -> list[Path]:
    """Every canonical KB markdown file, across backends."""
    out: list[Path] = []
    for kb_dir in audit.KB_DIRS.values():
        if kb_dir.is_dir():
            out.extend(sorted(kb_dir.rglob("*.md")))
    return out


def _applies_line_for(lines: list[str], start: int, end: int, audit):
    """First `applies_to:` line inside [start, end) → family set (or None)."""
    for line in lines[start:end]:
        applies_field_re = getattr(audit, "_APPLIES_FIELD_RE")
        m = applies_field_re.match(line.strip())
        if m:
            families, _raw = getattr(audit, "_applies_to_socs")(m.group(1))
            return families
    return None


def kb_entry_soc_families(entry_id: str) -> Optional[set[str]]:
    """SoC families a KB entry (`PB-34`, `OL-220`, `EC-68`, `P-P103`…) is scoped to.

    Returns the family set (`{"V220"}`), `{"*"}` for `soc=all`, or None when the
    entry is not found or declares no recognizable `soc=` scope.
    """
    audit = _kb_index_audit()
    if audit is None:
        return None
    for path in _iter_kb_files(audit):
        skip_current_item = False
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        entry_head_re = getattr(audit, "_ANY_ENTRY_HEAD_RE")
        heads = [i for i, line in enumerate(lines) if entry_head_re.match(line)]
        for n, start in enumerate(heads):
            if entry_head_re.match(lines[start]).group(1) != entry_id:
                continue
            end = heads[n + 1] if n + 1 < len(heads) else len(lines)
            return _applies_line_for(lines, start, end, audit)
    return None


def kb_section_soc_families(rel_path: str, section_no: str) -> Optional[set[str]]:
    """SoC families a numbered KB doc-section is scoped to.

    For reference docs whose sections carry an `applies_to:` line but no entry id
    — e.g. `fa_class/cross_core_sync.md` §4, whose
    `applies_to: soc=Ascend950PR (V351 / A5, Ascend950PR_9579)` makes its runnable
    handshake an A5 recipe.

    Args:
        rel_path: path under `src/skills/references/`, e.g.
            `target/ascendc/fa_class/cross_core_sync.md`.
        section_no: the section number as written, e.g. `"4"`.
    """
    audit = _kb_index_audit()
    if audit is None:
        return None
    path = kb_root() / rel_path
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    heads = [i for i, l in enumerate(lines) if _SECTION_HEAD_RE.match(l)]
    for n, start in enumerate(heads):
        if _SECTION_HEAD_RE.match(lines[start]).group(1) != section_no:
            continue
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        return _applies_line_for(lines, start, end, audit)
    return None


# Header zone of a domain-template file — where a prominent `applies_to:` lives
# (frontmatter or a top blockquote). Mirrors kb_index_audit._TEMPLATE_SCOPE_HEAD_LINES.
_TEMPLATE_HEADER_LINES = 20


def _resolve_domain_template_path(rel_path: str):
    """Resolve ANY path form the compose path can produce to the on-disk file.

    The brief manifest can name a domain template several ways depending on what
    the classifier emitted and whether `resolve_legacy_kb_path` normalized it:
      - `target/ascendc/patterns/domains/X.md`  (legacy-rewritten / canonical)
      - `patterns/domains/X.md`                 (raw classifier recommendation)
      - `domains/X.md`                          (bare, un-rewritten)
      - `src/skills/references/target/ascendc/patterns/domains/X.md` (full/abs)
    A filter that resolved only the canonical form would be inert for the others
    (wired but never firing — theater one layer deeper). All domain templates
    live at `target/ascendc/patterns/domains/<name>.md`, so any `…/domains/<name>.md`
    (or bare `domains/<name>.md`) is resolved by canonical basename. Returns a
    Path (may not exist) or None if `rel_path` names no domain-template file.
    """
    refs = kb_root()
    rel = rel_path.strip()
    marker = "src/skills/references/"
    if marker in rel:  # strip an absolute / repo-rooted prefix
        rel = rel[rel.rindex(marker) + len(marker):]
    direct = refs / rel
    if direct.is_file():
        return direct
    m = re.search(r"(?:^|/)domains/([^/]+\.md)$", rel)
    if m:
        cand = refs / "target" / "ascendc" / "patterns" / "domains" / m.group(1)
        if cand.is_file():
            return cand
    return None


def kb_file_soc_families(rel_path: str) -> Optional[set[str]]:
    """SoC families a whole domain-TEMPLATE file (`patterns/domains/*.md`) is scoped to.

    Reads the file's header-zone `applies_to: soc=` line, tolerating a leading
    blockquote `>` (the FA / GMM template convention keeps the tag inside a top
    `>` block). This is the file-level analogue of `kb_entry_soc_families`
    (per-entry) and `kb_section_soc_families` (per numbered section): a domain
    template is delivered to a worker as a WHOLE file (a brief-manifest path),
    so its scope is a whole-file property.

    `rel_path` is accepted in ANY of the path forms the compose path can emit
    (see `_resolve_domain_template_path`), NOT only the canonical
    `target/ascendc/patterns/domains/X.md` — a form-sensitive resolver would make
    the whole filter inert for the un-normalized forms.

    Returns the family set (`{"V351"}`), `{"*"}` for `soc=all`, or None when the
    file is absent or declares no machine-readable header `applies_to: soc=`.
    """
    audit = _kb_index_audit()
    if audit is None:
        return None
    path = _resolve_domain_template_path(rel_path)
    if path is None or not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for raw in lines[:_TEMPLATE_HEADER_LINES]:
        s = raw.strip()
        if s.startswith(">"):
            s = s[1:].strip()  # see through a top-of-file blockquote
        applies_field_re = getattr(audit, "_APPLIES_FIELD_RE")
        m = applies_field_re.match(s)
        if m:
            families, _raw = getattr(audit, "_applies_to_socs")(m.group(1))
            return families
    return None


def kb_file_applies_to_target(rel_path: str, target: Optional[str]) -> bool:
    """Should a composer inject a domain-TEMPLATE file into a `target` brief? FAIL-OPEN.

    True (inject) unless the template declares a concrete header `soc=` scope,
    the target's family is known, and that family is outside the scope. An
    untagged template, a `soc=all` template, or an unknown target all keep the
    template — this can only ever narrow an over-delivery (an a5-only template
    handed to a 220x/a3 worker), never invent a new under-delivery.
    """
    return applies_to_target(kb_file_soc_families(rel_path), target)


def applies_to_target(families: Optional[set[str]], target: Optional[str]) -> bool:
    """Does a parsed `applies_to` family set cover `target`? FAIL-OPEN.

    True (inject) unless the entry declares a concrete SoC scope, the target's
    family is known, and that family is outside the scope.
    """
    if not families or families == {"*"}:
        return True  # unscoped / universal → nothing to contradict
    fam = soc_family_for_target(target)
    if fam is None:
        return True  # unknown target → never silently drop knowledge
    return fam in families


def kb_entry_applies_to_target(entry_id: str, target: Optional[str]) -> bool:
    """Should a composer inject `entry_id`'s knowledge into a `target` brief?"""
    return applies_to_target(kb_entry_soc_families(entry_id), target)


def kb_section_applies_to_target(
    rel_path: str, section_no: str, target: Optional[str]
) -> bool:
    """Should a composer inject a numbered doc-section's recipe for `target`?"""
    return applies_to_target(kb_section_soc_families(rel_path, section_no), target)
