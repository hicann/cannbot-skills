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

"""report_not_log_lint.py — deterministic "Report != Log" lint for a project REPORT.md.

Mechanizes the aog-report-gen "Report != Log" cardinal rule (SKILL.md) that was previously
documented-only / agent-discretionary. Integration test (2026-06-20) showed audit agents only
*volunteered* the flag when they happened to read the prose; a one-line hand-edit that skipped the
skill got zero protection (the original §一-SIMD-as-a-log incident). This makes the check executable
+ a first-class DRIFT driver.

Scope: the FRONT (core-data) region only — from the doc top through the end of the FIRST top-level
`## ` section (the 核心数据 / 核心成果 section that holds the comparison tables). Later sections
(§二+ implementation, §三 history/investigation) are ALLOWED to be narrative — that's where the log
content belongs.

Checks (front only):
  C1 (DRIFT)   log-event / investigation phrases — a report states conclusions, a log narrates events.
  C2 (DRIFT)   long non-blockquote, non-table prose paragraph — narrative where a table row / a short
               `>` 关键读法 blockquote belongs.
  C3 (advisory) a table row with a bare device-time absolute (µs/us/ms) and NO comparison ratio (× / vs)
               in the same row — an absolute with no vs-baseline comparison is meaningless to a reader.

Usage:  python3 report_not_log_lint.py <REPORT.md> [--json]
Exit:   0 = clean; 2 = DRIFT (any C1/C2). C3 alone prints advisory but exits 0.
Prints a `report_not_log_violations=N` status token (consumed by Phase R_audit).
"""
import re
import sys
import json

# C1: investigation / log-event markers that belong in §三-history, not in the front core-data section.
_LOG_PHRASES = [
    r"已翻案", r"翻案", r"NO-?GO", r"复查发现", r"复查", r"原型坏", r"坏原型",
    r"优化手法", r"scalar-?bound", r"vec_ratio\s*[0-9.]+\s*[→\-]>", r"盘上有.*套",
    r"breakthrough", r"refuted\b", r"investigation",
]
# NOTE: deliberately NOT flagging "✅" — it is a legitimate per-dtype pass marker in data-table cells
# (e.g. "✅ 达标"), not an event-celebration. C1 only fires on PROSE lines (table rows are skipped).
_LOG_RE = re.compile("|".join(f"(?:{p})" for p in _LOG_PHRASES))

# Archive-link line (introduced by the archive-link convention, report_archive_link_lint.py): a
# `归档:` pointer line listing relative markdown links to kernel dirs / verification.json. It is
# STRUCTURAL (a per-table archive pointer), not narrative prose — skip it in C1/C2 so the
# archive-link convention does not trip the Report≠Log lint.
_ARCHIVE_LINE_RE = re.compile(r"^\s*归档[（(:：]")

# C3: a table cell device-time absolute (µs/us/ms); ratio markers that make it a valid comparison.
_TIME_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:µs|us|ms)\b")
_RATIO_RE = re.compile(r"[×x]\b|vs[\s\-]|ratio|比例|比.*快|0\.\d+\s*[×x]")


def _front_region(lines):
    """Return (start, end) line indices [0-based, end exclusive] of the front core-data region:
    from doc top through the end of the FIRST top-level '## ' section."""
    h2 = [i for i, ln in enumerate(lines) if re.match(r"##\s+\S", ln) and not ln.startswith("###")]
    if not h2:
        return (0, len(lines))
    # front = the FIRST top-level '## ' section (核心数据 / 核心成果 — holds the comparison tables),
    # from its heading to the start of the SECOND top-level section. The intro BEFORE the first '## '
    # (§1 顶部 1-句范围) is allowed to summarize the conclusion briefly, so it is excluded.
    start = h2[0]
    end = h2[1] if len(h2) >= 2 else len(lines)
    return (start, end)


def lint(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    start, end = _front_region(lines)
    front = list(enumerate(lines))[start:end]
    drift, advisory = [], []

    # C1 — log-event phrases
    for i, ln in front:
        # >blockquote pointer + table-cell pass-markers + 归档-link line are legit
        if ln.strip().startswith(">") or ln.lstrip().startswith("|") or _ARCHIVE_LINE_RE.match(ln):
            continue
        m = _LOG_RE.search(ln)
        if m:
            drift.append((i + 1, "C1 log-event-phrase", m.group(0), ln.strip()[:120]))

    # C2 — long non-blockquote, non-table prose paragraph in the front
    para, para_start = [], None

    def flush(p, p0):
        if not p:
            return
        txt = " ".join(x.strip() for x in p)
        if len(txt) > 250:
            drift.append((p0 + 1, "C2 narrative-where-table-belongs", f"{len(txt)} chars", txt[:120]))
    for i, ln in front:
        s = ln.strip()
        is_struct = (not s) or s.startswith("|") or s.startswith(">") or s.startswith(
            "#") or s.startswith("```") or _ARCHIVE_LINE_RE.match(ln)
        if is_struct:
            flush(para, para_start)
            para, para_start = [], None
        else:
            if para_start is None:
                para_start = i
            para.append(ln)
    flush(para, para_start)

    # C3 — table row with a bare device-time absolute and no comparison ratio (advisory)
    for i, ln in front:
        if not ln.lstrip().startswith("|") or set(ln.strip()) <= set("|-: "):
            continue
        if _TIME_RE.search(ln) and not _RATIO_RE.search(ln):
            advisory.append((i + 1, "C3 bare-absolute-no-ratio", _TIME_RE.search(ln).group(0), ln.strip()[:120]))

    return drift, advisory


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if not args:
        print("usage: report_not_log_lint.py <REPORT.md> [--json]", file=sys.stderr)
        sys.exit(64)
    drift, advisory = lint(args[0])
    if as_json:
        print(json.dumps({"report_not_log_violations": len(drift),
                          "advisories": len(advisory),
                          "drift": drift, "advisory": advisory}, ensure_ascii=False, indent=2))
    else:
        print(f"report_not_log_violations={len(drift)}  advisories={len(advisory)}  ({args[0]})")
        for ln, kind, hit, ctx in drift:
            print(f"  DRIFT  L{ln}  {kind}  [{hit}]  {ctx}")
        for ln, kind, hit, ctx in advisory:
            print(f"  advisory  L{ln}  {kind}  [{hit}]  {ctx}")
        if not drift and not advisory:
            print("  clean — front section is comparison-data only, no log narrative.")
    sys.exit(2 if drift else 0)


if __name__ == "__main__":
    main()
