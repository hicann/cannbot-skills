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

SoC 解析逻辑（`_soc_family` / `_applies_to_socs` / `_ANY_ENTRY_HEAD_RE` /
`_APPLIES_FIELD_RE`）**内联自 `src/scripts/kb_index_audit.py`**（OKF-only
迁移，2026-08-31 摘掉对该模块的懒加载依赖——该模块随 legacy KB 一起退役；
实现原样拷贝）。它解决了两个难点：

  - **arch-family granularity** — the KB spells two families ~20 ways
    (`Ascend910_9382` / `Ascend910C` / `V220`; `Ascend950PR` / `Ascend950PR_9579`
    / `V351`). Exact-token comparison false-positives on every entry that
    declares `soc=Ascend950PR` and verifies on `Ascend950PR_9579`.
  - **identifying-position reads** — a naive prose scan red-flags `V220→A5 port`
    DIRECTIONS and cross-ref IDs.

条目来源：OKF 卡片（`kb/okf/**`，frontmatter `original_id: PB-34` +
`description:`/正文里的 `applies_to: soc=`）。OKF-only 迁移（2026-08-31）
前还覆盖 legacy `kb/target/**` 的 `## PB-34` 标题条目，该树已删除。
读不到时 FAIL-OPEN。

FAIL-OPEN by construction (`kb_applies_to_target`): an injection is suppressed
ONLY on a positive, machine-readable exclusion — the entry declares a `soc=`
scope, the target's family is known, and the family is not in the scope.
Unknown entry / unparseable scope / `soc=all` / unknown target all keep the
injection, so this can only ever narrow an over-block, never create a new
under-block.
"""
from __future__ import annotations
import logging

import re
from pathlib import Path
from typing import Optional

from kb_paths import kb_root

_HERE = Path(__file__).resolve()

# Target name (env.target / `TARGET=`) → coarse SoC arch family, the granularity
# `_soc_family` normalizes to. Mirrors op_taxonomy.TARGET_HW_SPEC_MAP
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

# ── Inlined SoC parsing (copied from kb_index_audit, 2026-08-31; see module note) ──

# Any SoC-naming token: Ascend9xx family names, or the V220/V351 arch shorthands.
_SOC_TOKEN_RE = re.compile(r"\b(Ascend9\d{2}[A-Za-z0-9_]*|V220|V351)\b")

_APPLIES_FIELD_RE = re.compile(r"^`?applies_to\s*:\s*(.*?)`?\s*$")

# Entry headings across every supported KB file class (legacy layout).
_ANY_ENTRY_HEAD_RE = re.compile(
    r"^#{2,4}\s+((?:EC|PB|OL|P-P|F-P|F-AP|CAND)[-A-Za-z0-9_]*)\b"
)

# OKF 卡片形态：frontmatter `original_id: PB-34` 携带 legacy 条目 id（卡片没有
# `## PB-34` 标题）；`applies_to` 常嵌在 `description:` 行内。
_OKF_ORIGINAL_ID_RE = re.compile(r"^original_id:\s*([A-Za-z0-9_-]+)\s*$")
_OKF_DESC_APPLIES_RE = re.compile(r'^description:\s*"?\s*applies_to\s*:\s*(.*?)"?\s*$')


def _soc_family(token: str) -> Optional[str]:
    """Normalize a SoC token to its coarse arch family (`V220` / `V351`)."""
    t = token.lower()
    if "950" in t or "v351" in t:
        return "V351"
    if "910" in t or "v220" in t:
        return "V220"
    return None


def _families_in(text: str) -> set[str]:
    """Every SoC family named anywhere in `text`."""
    return {f for f in (_soc_family(t) for t in _SOC_TOKEN_RE.findall(text)) if f}


def _applies_to_socs(value: str) -> tuple[Optional[set[str]], Optional[str]]:
    """Parse the `soc=` clause of an applies_to line into a family set.

    Returns `({"*"}, raw)` for `soc=all` / `soc=any` (universal), `(None, None)`
    when no `soc=` clause or no recognizable SoC token is present (unscoped).
    """
    m = re.search(r"soc\s*=\s*([^;`]*)", value)
    if not m:
        return None, None
    raw = m.group(1).strip()
    if re.match(r"^(all|any)\b", raw, re.IGNORECASE):
        return {"*"}, raw
    return (_families_in(raw) or None), raw


def soc_family_for_target(target: Optional[str]) -> Optional[str]:
    """Coarse SoC family (`V220` / `V351`) for a build target, else None."""
    if not target:
        return None
    return SOC_FAMILY_BY_TARGET.get(str(target).strip().lower())


# 条目扫描目录（相对 kb_root）：OKF-only 迁移（2026-08-31）后仅扫 OKF 卡片树；
# legacy target 树已删除。
_KB_SCAN_DIRS = ("okf",)


def _iter_kb_files() -> list[Path]:
    """Every canonical KB markdown file (OKF cards and reference docs)."""
    root = kb_root()
    out: list[Path] = []
    for sub in _KB_SCAN_DIRS:
        kb_dir = root / sub
        if kb_dir.is_dir():
            out.extend(sorted(kb_dir.rglob("*.md")))
    return out


def _applies_line_for(lines: list[str], start: int, end: int) -> Optional[set[str]]:
    """First `applies_to:` line inside [start, end) → family set (or None)."""
    for line in lines[start:end]:
        s = line.strip()
        m = _APPLIES_FIELD_RE.match(s) or _OKF_DESC_APPLIES_RE.match(s)
        if m:
            families, _raw = _applies_to_socs(m.group(1))
            return families
    return None


def kb_entry_soc_families(entry_id: str) -> Optional[set[str]]:
    """SoC families a KB entry (`PB-34`, `OL-220`, `EC-68`, `P-P103`…) is scoped to.

    Returns the family set (`{"V220"}`), `{"*"}` for `soc=all`, or None when the
    entry is not found or declares no recognizable `soc=` scope.
    """
    for path in _iter_kb_files():
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
        # OKF card `original_id: PB-34` frontmatter lines (`_ANY_ENTRY_HEAD_RE`
        # is retained for pre-migration heading forms inside migrated bodies).
        heads = [
            i for i, line in enumerate(lines)
            if _ANY_ENTRY_HEAD_RE.match(line) or _OKF_ORIGINAL_ID_RE.match(line)
        ]
        for n, start in enumerate(heads):
            head = lines[start]
            m = _ANY_ENTRY_HEAD_RE.match(head) or _OKF_ORIGINAL_ID_RE.match(head)
            if m.group(1) != entry_id:
                continue
            end = heads[n + 1] if n + 1 < len(heads) else len(lines)
            return _applies_line_for(lines, start, end)
    return None


def kb_section_soc_families(rel_path: str, section_no: str) -> Optional[set[str]]:
    """SoC families a numbered KB doc-section is scoped to.

    For reference docs whose sections carry an `applies_to:` line but no entry id
    — e.g. `fa_class/cross_core_sync.md` §4, whose
    `applies_to: soc=Ascend950PR (V351 / A5, Ascend950PR_9579)` makes its runnable
    handshake an A5 recipe.

    Args:
        rel_path: path under `kb/`, e.g.
            `okf/runbooks/operator-optimization/fa-cross-core-sync-workspacequeue.md`.
        section_no: the section number as written, e.g. `"4"`.
    """
    path = kb_root() / rel_path
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        # KNOWN FAIL-OPEN: the caller reads None as "no declared scope" and keeps the
        # card. The file exists (checked above), so this is an unreadable or non-UTF-8
        # card — rare, and narrowing the except at least stops unrelated bugs from being
        # swallowed into a scope decision.
        return None
    heads = [i for i, l in enumerate(lines) if _SECTION_HEAD_RE.match(l)]
    for n, start in enumerate(heads):
        if _SECTION_HEAD_RE.match(lines[start]).group(1) != section_no:
            continue
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        return _applies_line_for(lines, start, end)
    return None


# Header zone of a domain-template file — where a prominent `applies_to:` lives
# (frontmatter or a top blockquote). Formerly kb_index_audit._TEMPLATE_SCOPE_HEAD_LINES.
_TEMPLATE_HEADER_LINES = 20      # body lines scanned, counted AFTER frontmatter
_FRONTMATTER_MAX_LINES = 60      # cap the closing-`---` search so a stray `---` cannot run away


def _resolve_domain_template_path(rel_path: str):
    """Resolve ANY path form the compose path can produce to the on-disk file.

    The brief compose path can name a pattern/domain template several ways:
      - `okf/reference/porter/patterns/X.md`        (OKF canonical)
      - `patterns/domains/X.md` / `domains/X.md`   (raw classifier recommendation)
      - `src/skills/references/…/X.md`             (full/abs prefix form)
    A filter that resolved only the canonical form would be inert for the others
    (wired but never firing — theater one layer deeper). OKF-only 迁移后，未转卡的
    patterns 归一到 `kb/okf/reference/porter/patterns/<name>.md`（2026-09-05 目录重组后位于 porter bundle 下）（legacy 的
    `target/ascendc/patterns/domains/X.md` 形态已随 kb/target 删除）；已转卡（不在
    reference/patterns 下）或尚未搬迁的条目解析不到文件时返回 None，由调用方
    FAIL-OPEN（不因文件不存在而硬失败）。Returns a Path (may not exist) or None
    if `rel_path` names no domain-template file.
    """
    refs = kb_root()
    rel = rel_path.strip()
    marker = "src/skills/references/"
    if marker in rel:  # strip an absolute / repo-rooted prefix
        rel = rel[rel.rindex(marker) + len(marker):]
    direct = refs / rel
    if direct.is_file():
        return direct
    m = re.search(r"(?:^|/)(?:patterns|domains)/([^/]+\.md)$", rel)
    if m:
        cand = refs / "okf" / "reference" / "porter" / "patterns" / m.group(1)
        if cand.is_file():
            return cand
    return None


def _header_zone(lines: list[str]) -> list[str]:
    """The first `_TEMPLATE_HEADER_LINES` lines of BODY, i.e. after any YAML frontmatter.

    WHY the skip (2026-09-05). Making these cards OKF-compliant gave each ~12 lines of
    frontmatter, which pushes the machine-readable `applies_to: soc=` line further down.
    A plain `lines[:20]` window then reads past nothing and returns None, and the caller
    treats None as "no declared scope" — a SILENT fail-open (no lint error, no test
    failure). Measured over the shipped KB: 161 of 1326 cards parse differently before
    and after this change, every one of them in that direction.

    Scope of the claim, because the earlier version of this comment overstated it: the
    one card a composer actually gates today (`fa_class_a3_mix_template.md`, via
    `kw_brief_fa.py:341`) was NOT broken by the reorg — its tag sits at line 14, six
    lines inside the old window. So this is hardening against a real and measured
    fragility, not the repair of an observed mis-delivery.
    """
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), _FRONTMATTER_MAX_LINES)):
            if lines[i].strip() == "---":
                start = i + 1
                break
    return lines[start:start + _TEMPLATE_HEADER_LINES]


def kb_file_soc_families(rel_path: str) -> Optional[set[str]]:
    """SoC families a whole domain-TEMPLATE file is scoped to.

    Reads the file's header-zone `applies_to: soc=` line, tolerating a leading
    blockquote `>` (the FA / GMM template convention keeps the tag inside a top
    `>` block). This is the file-level analogue of `kb_entry_soc_families`
    (per-entry) and `kb_section_soc_families` (per numbered section): a domain
    template is delivered to a worker as a WHOLE file (a brief-manifest path),
    so its scope is a whole-file property.

    `rel_path` is accepted in ANY of the path forms the compose path can emit
    (see `_resolve_domain_template_path`), NOT only the canonical
    `okf/reference/porter/patterns/X.md` — a form-sensitive resolver would make
    the whole filter inert for the un-normalized forms.

    Returns the family set (`{"V351"}`), `{"*"}` for `soc=all`, or None when the
    file is absent or declares no machine-readable header `applies_to: soc=`.
    """
    path = _resolve_domain_template_path(rel_path)
    if path is None or not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for raw in _header_zone(lines):
        s = raw.strip()
        if s.startswith(">"):
            s = s[1:].strip()  # see through a top-of-file blockquote
        m = _APPLIES_FIELD_RE.match(s)
        if m:
            families, _raw = _applies_to_socs(m.group(1))
            return families
    return None


def kb_file_applies_to_target(rel_path: str, target: Optional[str]) -> bool:
    """Should a composer inject a domain-TEMPLATE file into a `target` brief? FAIL-OPEN.

    True (inject) unless the template declares a concrete header `soc=` scope,
    the target's family is known, and that family is outside the scope. An
    untagged template, a `soc=all` template, or an unknown target all keep the
    template — this can only ever narrow an over-delivery, never invent a new
    under-delivery. (The direction depends on the card: the only template a composer
    gates today, `fa_class_a3_mix_template.md`, is a3-only, so here the over-delivery
    being narrowed is an a3 template reaching an a5 worker.)
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
