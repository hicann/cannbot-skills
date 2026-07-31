# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""NODE-21 Phase B — KB applies_to migration script.

Per docs/design/KB_TAG_INDEX_TAXONOMY_DESIGN.md Phase B.

Sweeps OPERATIONAL_KNOWLEDGE.md / ERROR_CORRECTIONS.md / PLATFORM_BUGS.md /
patterns/PATTERN_INDEX.md and adds structured `applies_to:` YAML blocks to
entries that don't have one, OR converts existing single-line
`applies_to: foo=bar; baz=qux` lines into the canonical YAML block form
declared in Phase A's `src/scripts/kb_schema.py`.

Inference rules (rule-based, deterministic, no LLM):
- File is under `target/ascendc/` → paradigm=ascendc by default
- Entry mentions arch35 / V300 (project-internal) → arch_family=arch35
- Entry mentions arch22 / V220 → arch_family=arch22
- Entry mentions arch38 / V310 → arch_family=arch38
- Entry mentions Ascend950PR → soc=Ascend950PR (T2 specificity)
- Entry mentions Ascend910_9382 or Ascend910B → soc per match
- Entry has `__NPU_ARCH__ == NNNN` → npu_arch=NNNN

Default fallback: paradigm=any, arch_family=any (universal scope).

Modes:
- `--dry-run` (default): print diff per entry, no file changes
- `--apply`: write changes
- `--batch <FILTER>`: only entries whose body matches FILTER regex
- `--report`: summary of what would change without per-entry diff

Usage:
  python3 src/scripts/migrations/kb_add_applies_to.py --dry-run
  python3 src/scripts/migrations/kb_add_applies_to.py --apply --batch "OL-19[3-6]"
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]  # engine/
# 2026-07-05: KB relocated to <plugin_root>/kb/ (_REPO_ROOT.parent == plugin_root).
_KB_ROOT = _REPO_ROOT.parent / "kb" / "target" / "ascendc"

# Files this script will touch.
_KB_FILES = [
    _KB_ROOT / "OPERATIONAL_KNOWLEDGE.md",
    _KB_ROOT / "ERROR_CORRECTIONS.md",
    _KB_ROOT / "PLATFORM_BUGS.md",
    _KB_ROOT / "patterns" / "PATTERN_INDEX.md",
]

# Entry header regex per file.
_ENTRY_HEADER = re.compile(r"^(## |### )(OL-\d+|EC-\d+|PB-\d+|P-P\d+)[:\b]", re.MULTILINE)


@dataclass
class Entry:
    """One KB entry's parsed boundaries + content."""
    file: Path
    id: str
    start_line: int       # 0-indexed; the header line itself
    end_line: int         # exclusive; line after the entry ends
    header: str           # full header line e.g. "## OL-195: title"
    body_lines: list[str] = field(default_factory=list)


def _split_entries(file_path: Path) -> list[Entry]:
    """Return every entry in this file with its boundaries."""
    text = file_path.read_text()
    lines = text.splitlines()
    entries: list[Entry] = []
    last_idx: Optional[int] = None
    last_id: Optional[str] = None
    last_header: Optional[str] = None
    for i, line in enumerate(lines):
        m = _ENTRY_HEADER.match(line)
        if m is None:
            continue
        if last_idx is not None and last_id is not None:
            entries.append(Entry(
                file=file_path,
                id=last_id,
                start_line=last_idx,
                end_line=i,
                header=last_header or "",
                body_lines=lines[last_idx:i],
            ))
        last_idx = i
        last_id = m.group(2)
        last_header = line
    if last_idx is not None and last_id is not None:
        entries.append(Entry(
            file=file_path,
            id=last_id,
            start_line=last_idx,
            end_line=len(lines),
            header=last_header or "",
            body_lines=lines[last_idx:len(lines)],
        ))
    return entries


def _infer_tags(entry: Entry) -> dict[str, str]:
    """Rule-based scope inference from entry body."""
    body = "\n".join(entry.body_lines)
    tags: dict[str, str] = {}

    # T0 paradigm — file path hint
    if "target/ascendc/" in str(entry.file):
        tags["paradigm"] = "ascendc"

    # T1 arch_family — DETERMINISTIC signals ONLY (task #41 mis-tag fix,
    # 2026-05-28). Loose prose keywords like "V300" / "V220" / "Ascend950PR"
    # appear in comparative asides ("unlike V300...") inside generic entries
    # (OL-7 process rule, OL-88 harness lesson) and produced a ~15-20% mis-tag
    # rate that would WRONGLY HIDE generic entries from the other arch's
    # targets. So arch_family is inferred ONLY from:
    #   (a) the __NPU_ARCH__ macro value (deterministic family map), or
    #   (b) an explicit literal `arch35`/`arch22`/`arch38` directory-style token.
    # Everything else gets NO arch_family (defaults to universal scope) and is
    # tagged correctly later in the reviewed V300-rename pass (NODE-18), where
    # each mention is read in context. Strict-but-correct beats broad-but-wrong.
    _ARCH_BY_NPU_ARCH = {"3510": "arch35", "5102": "arch35",
                         "3003": "arch22", "3113": "arch22", "3510x": "arch35"}
    _npu_macro = re.search(r"__NPU_ARCH__\s*==\s*(\d{3,4})\b", body)
    if _npu_macro and _npu_macro.group(1) in _ARCH_BY_NPU_ARCH:
        tags["arch_family"] = _ARCH_BY_NPU_ARCH[_npu_macro.group(1)]
    elif re.search(r"\barch35/\b|`arch35`|\barch35\b(?!\d)", body) and not re.search(r"\barch22\b", body):
        tags["arch_family"] = "arch35"
    elif re.search(r"\barch22/\b|`arch22`|\barch22\b(?!\d)", body) and not re.search(r"\barch35\b", body):
        tags["arch_family"] = "arch22"
    # else: no arch_family — defaults to universal; reviewed pass assigns it.

    # T2 soc — explicit canonical SOC names
    soc_match = re.search(
        r"\b(Ascend950PR_9579|Ascend950PR_9589|Ascend950PR|Ascend910_9382|Ascend910_9392|Ascend910B[1-4])\b",
        body,
    )
    if soc_match:
        tags["soc"] = soc_match.group(1)

    # T3 npu_arch — __NPU_ARCH__ macro value
    npu_arch_match = re.search(r"__NPU_ARCH__\s*==\s*(\d{4})\b", body)
    if npu_arch_match:
        tags["npu_arch"] = npu_arch_match.group(1)

    # cann version
    cann_match = re.search(r"\bCANN\s+(\d+\.\d+(?:\.\d+)?)\b", body)
    if cann_match:
        tags["cann"] = cann_match.group(1)

    return tags


def _has_applies_to(entry: Entry) -> bool:
    """True if the entry already carries ANY applies_to scope tag — either the
    single-line ``applies_to: foo=bar`` form OR a structured ```yaml ... ```
    block with an ``applies_to:`` key.

    Task #41 fix (2026-05-28): the migration is CONSERVATIVE — it only ADDS
    applies_to to entries that have none; it never rewrites an entry that
    already has any form. This avoids (a) inserting a duplicate YAML block
    when one already exists, and (b) mis-parsing a single-line value that
    contains ';' inside parentheses (e.g. OL-196's prose-in-value). Existing
    single-line forms already validate via kb_schema.parse_applies_to, so
    converting them is cosmetic and risky — skip.
    """
    for line in entry.body_lines:
        stripped = line.strip()
        # single-line form: `applies_to: ...` or applies_to: ...
        if stripped.startswith("`applies_to:") or stripped.startswith("applies_to:"):
            return True
        # YAML-block form: a bare `applies_to:` key line inside a ```yaml block
        if stripped == "applies_to:":
            return True
    return False


def _yaml_block(tags: dict[str, str]) -> str:
    """Render tags as a markdown YAML block."""
    if not tags:
        return ""
    lines = ["```yaml", "applies_to:"]
    for key in ("paradigm", "arch_family", "soc", "npu_arch", "cann",
                "bisheng", "dtype", "op_class"):
        if key in tags:
            lines.append(f"  {key}: {tags[key]}")
    lines.append("```")
    return "\n".join(lines)


def _process_entry(entry: Entry, *, paradigm_only: bool = True) -> Optional[str]:
    """Return updated entry text if a change is proposed; None to skip.

    Conservative policy (task #41): SKIP any entry that already has an
    applies_to (single-line or YAML block). Only inject for entries with none.

    Discriminating-tag policy (2026-05-28, after OL-173 catalog mis-tag):
    by default (`paradigm_only=True`) the bulk migration emits ONLY the
    `paradigm` tag — the single dimension derivable 100% reliably from file
    location. Every discriminating tag (arch_family / soc / npu_arch / cann)
    is keyword-inferred from the entry body and a *catalog* or *comparison*
    entry (e.g. OL-173 "macro values per arch", which mentions __NPU_ARCH__
    == 3003 AND Ascend950PR as examples) gets mis-scoped — and a mis-scope
    HIDES a needed entry from the target it should serve. Those tags are
    added in a reviewed per-entry pass (e.g. the V300-rename sweep), not by
    bulk keyword guess. Pass `paradigm_only=False` only for that reviewed path.
    """
    if _has_applies_to(entry):
        return None  # never touch entries that already carry a scope tag
    inferred = _infer_tags(entry)
    if paradigm_only:
        inferred = {k: v for k, v in inferred.items() if k == "paradigm"}
    if not inferred:
        return None

    yaml_block = _yaml_block(inferred)
    if not yaml_block:
        return None

    # Insert immediately after the header (line 0 of body).
    new_body = list(entry.body_lines)
    new_body.insert(1, "")
    new_body.insert(2, yaml_block)
    return "\n".join(new_body)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default: dry-run)")
    parser.add_argument("--batch", default=None,
                        help="Only entries whose ID matches this regex")
    parser.add_argument("--report", action="store_true",
                        help="Summary only, no per-entry diff")
    parser.add_argument("--full-infer", action="store_true",
                        help="Emit discriminating tags (arch_family/soc/npu_arch/cann) "
                             "in addition to paradigm. DEFAULT OFF — those need per-entry "
                             "review (catalog/comparison entries mis-scope; see OL-173). "
                             "Use only in the reviewed V300-rename pass.")
    args = parser.parse_args(argv)

    total_entries = 0
    total_proposed = 0
    summary_by_file: dict[str, int] = {}

    for fp in _KB_FILES:
        if not fp.exists():
            print(f"WARN: skipping missing {fp}", file=sys.stderr)
            continue
        entries = _split_entries(fp)
        original = fp.read_text().splitlines()
        new_lines = list(original)
        offset = 0
        per_file_count = 0
        for entry in entries:
            total_entries += 1
            if args.batch and not re.search(args.batch, entry.id):
                continue
            updated = _process_entry(entry, paradigm_only=not args.full_infer)
            if updated is None:
                continue
            total_proposed += 1
            per_file_count += 1
            if args.apply:
                # Replace original line range with updated body.
                start = entry.start_line + offset
                end = entry.end_line + offset
                updated_lines = updated.splitlines()
                new_lines[start:end] = updated_lines
                offset += len(updated_lines) - (end - start)
            elif not args.report:
                _inf = _infer_tags(entry)
                if not args.full_infer:
                    _inf = {k: v for k, v in _inf.items() if k == "paradigm"}
                print(f"\n--- {entry.id} ({fp.name}) ---")
                print("PROPOSED:")
                print(_yaml_block(_inf))
        summary_by_file[fp.name] = per_file_count
        if args.apply and per_file_count > 0:
            fp.write_text("\n".join(new_lines) + "\n")

    print("\n=== Summary ===")
    print(f"Entries scanned: {total_entries}")
    print(f"Proposed changes: {total_proposed}")
    for name, count in summary_by_file.items():
        print(f"  {name}: {count}")
    if not args.apply:
        print("\n(dry-run; pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
