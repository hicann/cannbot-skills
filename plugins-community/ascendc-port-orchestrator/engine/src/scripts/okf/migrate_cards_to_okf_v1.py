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

"""migrate_cards_to_okf_v1.py — idempotent migration of porter kb/okf cards to the okf.v1 card schema.

The port plugin's cards were authored to the vendored okf-9.1.0 schema (`type`+`timestamp`, `*_card` types).
The community cannbot-knowledge engine (RFC #381) enforces okf.v1: required frontmatter
`schema_version, kind, type, source_family, title, description, tags, created_at, updated_at`, with
`schema_version == okf.v1`, `kind` in the strict KNOWN_KINDS, no legacy `timestamp`, and `created_at`/
`updated_at` as UTC ISO-8601 `Z`. This script rewrites each content card IN PLACE with a minimal diff:
it adds/renames the required fields, canonicalizes porter `type` → (type, kind), converts the timestamp
to UTC-Z, drops the legacy `timestamp`, and strips structural tags. The card BODY is never touched.
Re-runnable: cards already at `schema_version: okf.v1` are left untouched.

porter type → (canonical type, kind):
  build_card     → implementation_trap  / implementation_trap   (build error + fix = a trap to avoid)
  precision_card → implementation_trap  / implementation_trap   (precision pitfall + fix = a trap)
  perf_card      → optimization_runbook / operator_optimization
  Runbook        → optimization_runbook / operator_optimization
  Guide          → programming_guide    / guide

The legacy converters import ``write_okf_v1_card`` from this module.  That
boundary canonicalizes generated cards before they reach disk and refuses to
replace a different existing ``okf.v1`` card.  Re-running a remote converter
therefore cannot silently downgrade or erase a locally migrated card.

Usage:  migrate_cards_to_okf_v1.py <kb_root>            # migrate in place
        migrate_cards_to_okf_v1.py <kb_root> --dry <file>   # print one migrated card to stdout, no write
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TYPE_MAP = {
    "build_card": ("implementation_trap", "implementation_trap"),
    "precision_card": ("implementation_trap", "implementation_trap"),
    "perf_card": ("optimization_runbook", "operator_optimization"),
    "Runbook": ("optimization_runbook", "operator_optimization"),
    "Guide": ("programming_guide", "guide"),
}
SOURCE_FAMILY = "curated"   # porter cards are distilled/curated knowledge (KNOWN_SOURCE_FAMILIES)
STRUCTURAL_TAGS = {"cann", "ascend-c", "asc-devkit", "op-dev-guide", "profiling", "example", "glossary", "index"}
DEFAULT_TS = "2026-07-14T00:00:00Z"   # deterministic fallback when a card carries no timestamp
HEAD_ORDER = ["schema_version", "kind", "type", "source_family", "title", "description"]
TAIL_ORDER = ["tags", "created_at", "updated_at"]
DROP = {"timestamp"}   # legacy field the lint blocks; its value is migrated into created_at/updated_at


class OKFCardWriteError(RuntimeError):
    """A converter output cannot be written without losing card state."""


def to_utc_z(val: str) -> str:
    v = (val or "").strip().strip("'\"")
    if not v:
        return DEFAULT_TS
    try:
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return DEFAULT_TS


def tags_tokens_from_entry(entry) -> list:
    """Extract tag tokens from a `tags:` frontmatter entry — handles BOTH the inline flow form
    (`tags: [a, b]`) AND the YAML block-list form (`tags:` then `  - a` / `  - b` continuation lines).
    Returning the wrong-but-empty list here was the silent-corruption risk flagged in review.
    """
    first = re.match(r"^tags:\s*(.*)$", entry[0])
    inline = (first.group(1).strip() if first else "")
    toks = []
    if inline.startswith("[") and inline.endswith("]"):
        toks = [t.strip() for t in inline[1:-1].split(",") if t.strip()]
    elif inline and inline not in ("|", ">", "|-", ">-", "|+", ">+"):
        toks = [t.strip() for t in inline.split(",") if t.strip()]   # rare bare-scalar inline
    for ln in entry[1:]:                                             # block-list continuation items
        m = re.match(r"^\s*-\s*(.*)$", ln)
        if m and m.group(1).strip():
            toks.append(m.group(1).strip())
    return toks


def clean_tags(tokens) -> str:
    """Drop structural tags + dedup (case-insensitive), preserving order; emit canonical inline flow form."""
    seen, out = set(), []
    for tok in tokens:
        t = (tok or "").strip()
        if not t:
            continue
        key = t.strip("'\"").lower()
        if key in STRUCTURAL_TAGS or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return "[" + ", ".join(out) + "]"


def parse_ordered(fm_block: str):
    """Parse frontmatter into ordered [(key, [lines...])]; a key's value may span continuation lines
    (e.g. the `signal:` block list). Non-key leading lines (rare) attach to a None key preserved verbatim.
    """
    entries, cur = [], None
    for ln in fm_block.split("\n"):
        if re.match(r"^\w+:", ln):
            if cur is not None:
                entries.append(cur)
            cur = [ln]
        else:
            if cur is None:
                entries.append([None, ln])   # preserve stray leading line verbatim
            else:
                cur.append(ln)
    if cur is not None:
        entries.append(cur)
    return entries


def migrate_text(raw: str):
    if not raw.startswith("---"):
        return None, "skip-nofm"
    end = raw.find("\n---", 3)
    if end == -1:
        return None, "skip-nofm"
    # opening fence is line "---\n" (chars 0..3 == "---" + newline); block is raw[4:end]; closing at raw[end+1:end+4]
    fm_block = raw[4:end]
    rest = raw[end + 4:]   # everything after "\n---"
    entries = parse_ordered(fm_block)

    def keyof(e):
        return None if e[0] is None else re.match(r"^(\w+):", e[0]).group(1)

    orig_by_key = {}   # key → its full entry (key line + continuation lines)
    stray = []         # non-key lines before the first key (rare) — preserved verbatim
    kv = {}            # key → first-line value (for scalar reads: schema_version/type/timestamp)
    for e in entries:
        k = keyof(e)
        if k is None:
            stray.append(e[1])
            continue
        orig_by_key.setdefault(k, e)
        if k not in kv:
            m = re.match(r"^\w+:\s*(.*)$", e[0])
            kv[k] = m.group(1) if m else ""

    if kv.get("schema_version", "").strip() == "okf.v1":
        return None, "already"
    ptype = kv.get("type", "").strip()
    if ptype not in TYPE_MAP:
        return None, "skip-unknown-type:" + (ptype or "<none>")

    canon_type, kind = TYPE_MAP[ptype]
    created = to_utc_z(kv.get("timestamp", ""))
    new_vals = {
        "schema_version": "okf.v1",
        "kind": kind,
        "type": canon_type,
        "source_family": SOURCE_FAMILY,
        "created_at": created,
        "updated_at": created,
    }
    if "tags" in orig_by_key:
        new_vals["tags"] = clean_tags(tags_tokens_from_entry(orig_by_key["tags"]))

    # emit: HEAD_ORDER first, then original middle keys (order preserved, minus head/tail/drop),
    # then TAIL_ORDER. REBUILT fields (in new_vals) become one canonical line; PRESERVED fields emit
    # their FULL entry — every continuation line (block scalars / `signal:` block list) intact.
    managed = set(HEAD_ORDER) | set(TAIL_ORDER) | DROP
    out_lines = list(stray)
    for k in HEAD_ORDER:
        if k in new_vals:
            out_lines.append("%s: %s" % (k, new_vals[k]))
        elif k in orig_by_key:
            out_lines.extend(orig_by_key[k])          # preserve multi-line title/description
    done = set()
    for e in entries:
        k = keyof(e)
        if k is None or k in managed or k in done:
            continue
        out_lines.extend(orig_by_key[k])              # full entry incl. continuations
        done.add(k)
    for k in TAIL_ORDER:
        if k in new_vals:
            out_lines.append("%s: %s" % (k, new_vals[k]))
        elif k in orig_by_key:
            out_lines.extend(orig_by_key[k])

    new_fm = "\n".join(out_lines)
    return "---\n" + new_fm + "\n---" + rest, "migrated:" + ptype


def _canonical_okf_v1(raw: str, source: str) -> tuple[str, str]:
    """Return ``raw`` as okf.v1 or fail instead of emitting a legacy card."""
    new, status = migrate_text(raw)
    if status == "already":
        return raw, status
    if new is not None and status.startswith("migrated:"):
        return new, status
    raise OKFCardWriteError(
        "cannot produce okf.v1 for %s (%s); refusing to write" % (source, status)
    )


def write_okf_v1_card(path, generated: str) -> str:
    """Safely write one converter-produced card in canonical okf.v1 form.

    Existing v1 content is immutable at this migration boundary: an identical
    regeneration is left byte-for-byte untouched, while a differing result is
    rejected for manual reconciliation.  A legacy target is upgraded only when
    its canonical form exactly matches the generated card, so this helper also
    avoids erasing unreviewed local edits in old-schema files.

    Returns ``created``, ``upgraded``, or ``unchanged``.
    """
    target = Path(path)
    candidate, _ = _canonical_okf_v1(generated, "generated card %s" % target)

    action = "created"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        existing_v1, existing_status = _canonical_okf_v1(
            existing, "existing card %s" % target
        )
        if existing_status == "already":
            if existing == candidate:
                return "unchanged"
            raise OKFCardWriteError(
                "refusing to overwrite existing okf.v1 card with regenerated content: %s"
                % target
            )
        if existing_v1 != candidate:
            raise OKFCardWriteError(
                "refusing to overwrite locally different legacy card: %s" % target
            )
        action = "upgraded"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate, encoding="utf-8")
    return action


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 64
    root = Path(sys.argv[1])
    if "--dry" in sys.argv:
        f = Path(sys.argv[sys.argv.index("--dry") + 1])
        new, status = migrate_text(f.read_text(encoding="utf-8"))
        sys.stderr.write("[%s] %s\n" % (status, f))
        if new:
            sys.stdout.write(new[:new.find("\n---", 3) + 4] + "\n")   # print migrated frontmatter only
        return 1 if status.startswith("skip") else 0   # --dry on a card it can't handle is a failure too
    # fail-closed: a wrong/typo'd root must NOT succeed silently on an empty scan (empty-set fail-open).
    if not root.is_dir():
        sys.stderr.write("FAIL: kb_root %r is not a directory\n" % str(root))
        return 2
    stats = {}
    changed = 0
    scanned = 0
    skipped_files = []
    for md in sorted(root.rglob("*.md")):
        if md.name == "index.md" or "/_migration/" in md.as_posix():
            continue
        scanned += 1
        new, status = migrate_text(md.read_text(encoding="utf-8"))
        tag = status.split(":")[0]
        stats[tag] = stats.get(tag, 0) + 1
        if new is not None:
            md.write_text(new, encoding="utf-8")
            changed += 1
        elif tag.startswith("skip"):   # skip-nofm / skip-unknown-type — a card the migration cannot handle
            skipped_files.append((status, md.as_posix()))
    sys.stderr.write("migrate okf.v1: scanned=%d changed=%d  %s\n" % (scanned, changed, dict(sorted(stats.items()))))
    if scanned == 0:
        sys.stderr.write("FAIL: no content cards found under %r — wrong root? (expected reference/ ops/ runbooks/)\n"
                         % str(root))
        return 2
    # fail-loud: on the real card set every card must be migrated (or already migrated). Any
    # no-frontmatter / unknown-type card is an UNEXPECTED state — exit nonzero, don't silently pass.
    if skipped_files:
        sys.stderr.write("FAIL: %d card(s) in an unexpected state (not migrated):\n" % len(skipped_files))
        for status, path in skipped_files[:20]:
            sys.stderr.write("  [%s] %s\n" % (status, path))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
