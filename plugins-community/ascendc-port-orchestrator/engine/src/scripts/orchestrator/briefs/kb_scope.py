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

SoC 解析逻辑（`_soc_family` / `_applies_to_socs` /
`_APPLIES_FIELD_RE`）**内联自 `src/scripts/kb_index_audit.py`**（OKF-only
迁移，2026-08-31 摘掉对该模块的懒加载依赖——该模块随 legacy KB 一起退役；
实现原样拷贝）。它解决了两个难点：

  - **arch-family granularity** — the KB spells two families ~20 ways
    (`Ascend910_9382` / `Ascend910C` / `V220`; `Ascend950PR` / `Ascend950PR_9579`
    / `V351`). Exact-token comparison false-positives on every entry that
    declares `soc=Ascend950PR` and verifies on `Ascend950PR_9579`.
  - **identifying-position reads** — a naive prose scan red-flags `V220→A5 port`
    DIRECTIONS and cross-ref IDs.

条目来源：外部知识卡片（frontmatter `original_id: PB-34` +
`description:`/正文里的 `applies_to: soc=`）。
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

_LOG = logging.getLogger(__name__)

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

# `applies_to` may be embedded in a migrated card's `description:` line.
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


_OKF_V02_PROVENANCE_RE = re.compile(r"原 OKF v1 卡号：\*\*([A-Za-z0-9_-]+)\*\*")
_OKF_V02_ALIASES_INLINE_RE = re.compile(r"^aliases:\s*\[([^\]]*)\]")
_OKF_V02_ALIAS_ITEM_RE = re.compile(r"^\s*-\s*((?:EC|PB|OL|P-P|F-P|F-AP|CAND)[-A-Za-z0-9_]*)\s*$")
_OKF_V02_PLATFORMS_INLINE_RE = re.compile(r"^platforms:\s*\[([^\]]*)\]")
_OKF_V02_PLATFORMS_ITEM_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_\"']+)\s*$")
_ENTRY_ID_RE = re.compile(r"^(?:EC|PB|OL|P-P|F-P|F-AP|CAND)[-A-Za-z0-9_]*$")

# 知识仓平台注册值 → SoC family（与 SOC_FAMILY_BY_TARGET 同粒度：
# a3/a2/910* → V220；950（含 950PR/A5）→ V351；agnostic → 全平台通配）。
_FAMILY_BY_PLATFORM = {
    "a2": "V220",
    "a3": "V220",
    "910": "V220",
    "950": "V351",
}


def _families_from_platforms(platforms: list[str]) -> Optional[set[str]]:
    """OKF v0.2 frontmatter `platforms:` → SoC family 集合；空/未声明 → None。"""
    families: set[str] = set()
    for raw in platforms:
        p = str(raw).strip().strip('"').strip("'").lower()
        if not p:
            continue
        if p == "agnostic":
            return {"*"}
        fam = _FAMILY_BY_PLATFORM.get(p)
        if fam:
            families.add(fam)
    return families or None


_EXTERNAL_SCOPES_CACHE: dict[str, Optional[set[str]]] = {}


def _alias_items_from_head(head: str) -> list[str]:
    """`- ` alias items under an `aliases:` frontmatter block."""
    found: list[str] = []
    in_aliases = False
    for line in head.splitlines():
        stripped = line.strip()
        if stripped.startswith("aliases:"):
            in_aliases = True
            continue
        if not in_aliases:
            continue
        if not stripped.startswith("-"):
            in_aliases = False
            continue
        alias_match = _OKF_V02_ALIAS_ITEM_RE.match(line)
        if alias_match:
            found.append(alias_match.group(1))
    return found


def _card_ids_from_head(head: str) -> list[str]:
    """Legacy entry ids from one card head (provenance line + aliases)."""
    ids: list[str] = []
    match = _OKF_V02_PROVENANCE_RE.search(head)
    if match:
        ids.append(match.group(1))
    match = _OKF_V02_ALIASES_INLINE_RE.search(head)
    if match:
        ids.extend(
            alias.strip().strip('"').strip("'")
            for alias in match.group(1).split(",")
        )
    ids.extend(_alias_items_from_head(head))
    return [entry_id for entry_id in dict.fromkeys(ids) if _ENTRY_ID_RE.match(entry_id)]


def _platform_items_from_head(head: str) -> list[str]:
    """`- ` platform items under a `platforms:` frontmatter block."""
    found: list[str] = []
    in_platforms = False
    for line in head.splitlines():
        stripped = line.strip()
        if stripped.startswith("platforms:"):
            in_platforms = True
            continue
        if not in_platforms:
            continue
        if not stripped.startswith("-"):
            in_platforms = False
            continue
        item_match = _OKF_V02_PLATFORMS_ITEM_RE.match(line)
        if item_match:
            found.append(item_match.group(1))
    return found


def _card_platforms_from_head(head: str) -> list[str]:
    """Frontmatter platforms from one card head (inline list or `- ` items)."""
    platforms: list[str] = []
    platform_match = _OKF_V02_PLATFORMS_INLINE_RE.search(head)
    if platform_match:
        platforms.extend(item.strip() for item in platform_match.group(1).split(","))
        return platforms
    return _platform_items_from_head(head)


def _card_families_from_head(head: str, platforms: list[str]) -> Optional[set[str]]:
    """SoC families from platforms, falling back to an applies_to line."""
    families = _families_from_platforms(platforms)
    if families is not None:
        return families
    for line in head.splitlines():
        stripped = line.strip()
        applies_match = (
            _APPLIES_FIELD_RE.match(stripped)
            or _OKF_DESC_APPLIES_RE.match(stripped)
        )
        if applies_match:
            return _applies_to_socs(applies_match.group(1))[0]
    return None


def _ids_and_scope_from_card(path: Path) -> tuple[list[str], Optional[set[str]]]:
    """Read legacy IDs and SoC scope from one query-selected external card."""
    try:
        with path.open(encoding="utf-8") as fh:
            head = fh.read(8192)
    except (OSError, UnicodeDecodeError):
        return [], None
    ids = _card_ids_from_head(head)
    families = _card_families_from_head(head, _card_platforms_from_head(head))
    return ids, families



def _collect_entry_hits(entry_id: str) -> list[dict]:
    """Query both architecture families and dedup hits by local_path.

    Platform filtering must not hide a V220-only card while deciding whether
    it applies to V351, or vice versa.
    """
    from briefs.external_kb import search_external_cards

    hits: list[dict] = []
    seen_paths: set[str] = set()
    for query_target in ("a5", "a3"):
        for hit in search_external_cards(
            entry_id, target=query_target, per_platform_k=10
        ):
            path_key = str(hit.get("local_path") or "")
            if path_key and path_key not in seen_paths:
                seen_paths.add(path_key)
                hits.append(hit)
    return hits


def _external_entry_scope(entry_id: str) -> Optional[set[str]]:
    """Resolve one entry through knowledge-query; never scan plugin/local trees."""
    if entry_id in _EXTERNAL_SCOPES_CACHE:
        return _EXTERNAL_SCOPES_CACHE[entry_id]
    try:
        hits = _collect_entry_hits(entry_id)
        for hit in hits:
            local_path = hit.get("local_path")
            if not local_path:
                continue
            ids, families = _ids_and_scope_from_card(Path(str(local_path)))
            for found_id in ids:
                _EXTERNAL_SCOPES_CACHE.setdefault(found_id, families)
            if entry_id in ids:
                return families
    except Exception as exc:
        _LOG.debug("external entry scope lookup failed: %s err=%s", entry_id, exc)
    _EXTERNAL_SCOPES_CACHE[entry_id] = None
    return None


def reset_external_scopes_cache() -> None:
    """Clear the external entry-scope cache (ut test-isolation hook).

    Clear, never rebind to None: `_external_entry_scope` does
    `entry_id in _EXTERNAL_SCOPES_CACHE` before its try block, so a None
    cache raised TypeError on every soc-scope lookup and failed the
    branch's own kw_brief soc-scope/decomposition suites (65 ut
    regressions vs the merge-base). Rebind to a fresh dict when a
    previous botched reset left a non-dict behind.
    """
    global _EXTERNAL_SCOPES_CACHE
    if not isinstance(_EXTERNAL_SCOPES_CACHE, dict):
        _EXTERNAL_SCOPES_CACHE = {}
    _EXTERNAL_SCOPES_CACHE.clear()


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
    return _external_entry_scope(entry_id)


def kb_section_soc_families(rel_path: str, section_no: str) -> Optional[set[str]]:
    """SoC families a numbered KB doc-section is scoped to.

    For reference docs whose sections carry an `applies_to:` line but no entry id
    — e.g. `fa_class/cross_core_sync.md` §4, whose
    `applies_to: soc=Ascend950PR (V351 / A5, Ascend950PR_9579)` makes its runnable
    handshake an A5 recipe.

    Args:
        rel_path: canonical external knowledge path, e.g.
            `knowledge/ops/ascendc/optimizations/fa_cross_core_sync_workspacequeue.md`.
        section_no: the section number as written, e.g. `"4"`.
    """
    try:
        from briefs import external_kb as _ext
        path = _ext.resolve_kb_ref(rel_path)
    except Exception:
        path = None
    if path is None:
        return None
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
      - provider-relative porter pattern paths
      - `patterns/domains/X.md` / `domains/X.md`   (raw classifier recommendation)
      - `src/skills/references/…/X.md`             (full/abs prefix form)
    A filter that resolved only one form would be inert for the others
    (wired but never firing — theater one layer deeper). 未转卡的 patterns 归一到
    provider 的 porter patterns 目录；已转卡（不在
    reference/patterns 下）或尚未搬迁的条目解析不到文件时返回 None，由调用方
    FAIL-OPEN（不因文件不存在而硬失败）。Returns a Path (may not exist) or None
    if `rel_path` names no domain-template file.
    """
    rel = rel_path.strip()
    marker = "src/skills/references/"
    if marker in rel:  # strip an absolute / repo-rooted prefix
        rel = rel[rel.rindex(marker) + len(marker):]
    try:
        from briefs import external_kb as _ext
        resolved = _ext.resolve_kb_ref(rel)
    except Exception:
        resolved = None
    if resolved is not None and resolved.is_file():
        return resolved
    m = re.search(r"(?:^|/)(?:patterns|domains)/([^/]+\.md)$", rel)
    if m:
        # 外部知识仓（OKF v0.2）：porter patterns 已迁至 ops/ascendc/examples/。
        try:
            from briefs import external_kb as _ext
            resolved = _ext.resolve_kb_ref(
                "knowledge/ops/ascendc/examples/" + m.group(1).replace("-", "_")
            )
        except Exception:
            resolved = None
        if resolved is not None and resolved.is_file():
            return resolved
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
    `knowledge/ops/ascendc/examples/X.md` — a form-sensitive resolver would make
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
