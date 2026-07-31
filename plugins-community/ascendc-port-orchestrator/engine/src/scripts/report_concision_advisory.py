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

"""report_concision_advisory.py — front-prose concision ADVISORY for a project REPORT.md.

Owner direction (2026-06-19): "中文正式报告行文但不会太啰嗦。" Data belongs in tables; front prose
should be a short 关键读法, not a wall of text. This flags front-region prose blocks (including `>`
关键读法 blockquotes — those are exactly where over-long narration sneaks in) that exceed a soft length
threshold, with the advice "tighten — move data to a table, keep prose to a short 关键读法".

ADVISORY ONLY (never DRIFT): owner wants polish, not a hard gate that false-kills a legitimately
detailed report. **Always exits 0.** Phase R_audit / R2 surface it as a non-blocking note; concision
is a writing nudge, not a merge blocker (unlike jargon-lint / archive-link-lint, which are DRIFT
drivers + blocking write-time gates).

Mirrors report_not_log_lint's FRONT-region detection (first top-level `## ` section, intro excluded)
and its C2 paragraph-accumulation, but counts `>` blockquotes too (C2 skips them — this advisory does
NOT, because over-long 关键读法 pointers are the common over-writing site) and uses a higher threshold.

Usage:  python3 report_concision_advisory.py <REPORT.md> [--json]
Exit:   always 0. Prints a `report_concision_advisories=N` status token.
"""
import re
import sys
import json

_THRESHOLD = 400  # chars; soft — a front prose/blockquote block longer than this gets a tighten nudge.

# `归档:` archive-link pointer line (archive-link convention) — structural, not narrative prose. Skip.
_ARCHIVE_LINE_RE = re.compile(r"^\s*归档[（(:：]")


def _front_region(lines):
    h2 = [i for i, ln in enumerate(lines) if re.match(r"##\s+\S", ln) and not ln.startswith("###")]
    if not h2:
        return (0, len(lines))
    start = h2[0]
    end = h2[1] if len(h2) >= 2 else len(lines)
    return (start, end)


def advise(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    start, end = _front_region(lines)
    front = list(enumerate(lines))[start:end]
    notes = []
    # accumulate contiguous prose/blockquote blocks (NOT tables/headings/fences/blank)
    block, block_start = [], None

    def flush(b, b0):
        if not b:
            return
        # strip a leading '> ' from blockquote lines so the count is the readable text length
        txt = " ".join(re.sub(r"^\s*>\s?", "", x).strip() for x in b)
        if len(txt) > _THRESHOLD:
            notes.append((b0 + 1, "ADV:front-prose-too-long",
                          f"{len(txt)} chars (>{_THRESHOLD})", txt[:100]))

    for i, ln in front:
        s = ln.strip()
        is_struct = (not s) or s.startswith("|") or s.startswith(
            "#") or s.startswith("```") or _ARCHIVE_LINE_RE.match(ln)
        if is_struct:
            flush(block, block_start)
            block, block_start = [], None
        else:
            if block_start is None:
                block_start = i
            block.append(ln)
    flush(block, block_start)
    return notes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if not args:
        print("usage: report_concision_advisory.py <REPORT.md> [--json]", file=sys.stderr)
        sys.exit(64)
    notes = advise(args[0])
    if as_json:
        print(json.dumps({"report_concision_advisories": len(notes), "advisories": notes},
                         ensure_ascii=False, indent=2))
    else:
        print(f"report_concision_advisories={len(notes)}  ({args[0]})")
        for ln, kind, hit, ctx in notes:
            print(f"  advisory  L{ln}  {kind}  [{hit}]  tighten — data→table, prose→short 关键读法  {ctx}")
        if not notes:
            print("  clean — front prose is concise.")
    sys.exit(0)  # ADVISORY — never blocks.


if __name__ == "__main__":
    main()
