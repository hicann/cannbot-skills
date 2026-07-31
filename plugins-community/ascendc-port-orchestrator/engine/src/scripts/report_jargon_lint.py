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

"""report_jargon_lint.py — deterministic internal-harness-jargon lint for a project REPORT.md.

Owner's TOP report-quality concern (2026-06-19, repeated, SERIOUS — "客户觉得我们是 cannbot 套壳/
骗人"): the FRONT (核心数据 / 核心成果) section of a customer-facing report must NOT leak
internal-harness jargon. A customer reading §1-2 should see plain comparison data, never our
pipeline's private vocabulary (cannbot / 商用「标准」 framing / PARTIAL_PERSIST / OL-/DEBT- IDs /
competitor_mare / pass_a-pass_b / blackbox / lane-health / ratio-gate / bare commit-SHAs / task-IDs).
Internal verdicts stay in `verification.json`, not in the prose a customer reads.

Mirrors `report_not_log_lint.py` exactly: same FRONT-region detection (first top-level `## ` section
= 核心数据, intro before it excluded; later §implementation / §history sections ALLOWED to use
jargon — that is where internal vocabulary legitimately lives), `--json`, exit 2 on hits, a
`report_jargon_violations=N` status token consumed by Phase R_audit.

CONSERVATIVE (no-false-positive is main's HARD requirement):
  - PROSE only — table rows, blockquote pointers, headings, code fences are SKIPPED. Jargon in a
    blockquote 关键读法 pointer or a heading parenthetical (e.g. a backticked md5 ref) is not
    customer-facing prose.
  - Backtick code spans are STRIPPED before matching — legit technical tokens (verification.json,
    compare_cv, a md5-hash reference written in backticks) live in backticks and must not trip the lint.
  - The intro (before the first `## `) is excluded (allowed to summarize briefly).

Usage:  python3 report_jargon_lint.py <REPORT.md> [--json]
Exit:   0 = clean; 2 = jargon found in the front prose.
"""
import re
import sys
import json

# Denylist (owner 2026-06-19). Each entry is (label, compiled-regex). Tuned to fire on PROSE only —
# the caller already strips backtick spans + skips tables/blockquotes/headings before matching.
_DENY = [
    ("cannbot", re.compile(r"cannbot", re.IGNORECASE)),
    # 商用「标准」/② framing — the "commercial standard" jargon owner banned. Plain 商用 noun is rare
    # in a front section; flag it (front is data-only, no need for the word).
    ("商用", re.compile(r"商用")),
    ("PARTIAL_PERSIST", re.compile(r"PARTIAL_PERSIST")),
    ("OL-id", re.compile(r"\bOL-\d+\b")),
    ("DEBT-id", re.compile(r"\bDEBT-\d+\b")),
    ("competitor_mare", re.compile(r"competitor_mare")),
    # pass_a / pass_b internal verdict tokens (underscore/hyphen). NOT "P-P58" (KB pattern id, allowed).
    ("pass_a/pass_b", re.compile(r"\bpass[_-][ab]\b", re.IGNORECASE)),
    ("blackbox", re.compile(r"\bblackbox\b", re.IGNORECASE)),
    ("lane-health", re.compile(r"\blane[- ]health\b", re.IGNORECASE)),
    ("ratio-gate", re.compile(r"\bratio[- ]gate\b", re.IGNORECASE)),
    # task-IDs: a bare #<digits> (PR/issue ref) in prose. Not a markdown anchor (#section) — those are
    # word-prefixed; this requires the # to be preceded by whitespace/start and followed by digits.
    ("task-id #N", re.compile(r"(?:^|\s)#\d+\b")),
    # P0xxx internal phase/task tags (P0cc, P0aba, P128 …). Letter-or-digit suffix after P0 / P + digits.
    ("P0-tag", re.compile(r"\bP0[a-z]+\b")),
    ("P-phase", re.compile(r"\bP\d{2,}\b")),
    # bare commit-SHA in prose: a standalone 7-40 hex token (after backtick-stripping). Word-boundaried
    # and required to contain >=1 digit AND >=1 a-f letter to avoid flagging plain numbers / plain words.
    ("bare-SHA", re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")),
]

_BACKTICK_SPAN = re.compile(r"`[^`]*`")


def _front_region(lines):
    """(start, end) [0-based, end exclusive] of the front core-data region: the FIRST top-level
    `## ` section. Identical contract to report_not_log_lint._front_region (intro excluded)."""
    h2 = [i for i, ln in enumerate(lines) if re.match(r"##\s+\S", ln) and not ln.startswith("###")]
    if not h2:
        return (0, len(lines))
    start = h2[0]
    end = h2[1] if len(h2) >= 2 else len(lines)
    return (start, end)


def lint(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    start, end = _front_region(lines)
    front = list(enumerate(lines))[start:end]
    hits = []
    for i, ln in front:
        s = ln.strip()
        # PROSE only — skip tables, blockquote pointers, headings, code fences.
        if (not s) or s.startswith("|") or s.startswith(">") or s.startswith("#") or s.startswith("```"):
            continue
        # Strip backtick code spans — legit technical tokens live there.
        prose = _BACKTICK_SPAN.sub(" ", ln)
        for label, rx in _DENY:
            m = rx.search(prose)
            if m:
                hits.append((i + 1, f"J:{label}", m.group(0).strip(), s[:120]))
    return hits


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if not args:
        print("usage: report_jargon_lint.py <REPORT.md> [--json]", file=sys.stderr)
        sys.exit(64)
    hits = lint(args[0])
    if as_json:
        print(json.dumps({"report_jargon_violations": len(hits), "hits": hits},
                         ensure_ascii=False, indent=2))
    else:
        print(f"report_jargon_violations={len(hits)}  ({args[0]})")
        for ln, kind, hit, ctx in hits:
            print(f"  JARGON  L{ln}  {kind}  [{hit}]  {ctx}")
        if not hits:
            print("  clean — front section is jargon-free (customer-facing plain language).")
    sys.exit(2 if hits else 0)


if __name__ == "__main__":
    main()
