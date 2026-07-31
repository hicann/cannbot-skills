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

"""report_archive_link_lint.py — deterministic "every core-results table carries its archive link" lint.

Owner direction (2026-06-19): "每个表格到 archive 的文件要有链接，早期报告有、我最近的漏了。" Each
core-results comparison table in the FRONT (核心数据 / 核心成果) section must carry a relative markdown
link to its archive — the kernel dir (and `verification.json` when present) — so a reader can click
through from a number to the artifact that produced it. Early reports had these; recent ones dropped them.

Convention enforced: within a few lines of each front core-results table (immediately after the table,
before the next table / heading), there is a line carrying a markdown link whose target is an archived
kernel path (`.../kernels/<op>/` or `.../verification.json`). Example, under a table:
    归档: [selective_scan_fwd_simd](src/kernels/selective_scan_fwd_simd/) ·
    [verification.json](src/kernels/selective_scan_fwd_simd/verification.json)

Mirrors `report_not_log_lint.py`: same FRONT-region detection (first top-level `## ` section, intro
excluded), `--json`, exit 2 on a table missing its link, a `report_archive_link_violations=N` status
token consumed by Phase R_audit.

CONSERVATIVE (no-false-positive is main's HARD requirement):
  - Only "core-results" tables are checked: a contiguous `|`-row block with a header + separator +
    at least 2 DATA rows. Tiny tables (a 1-row note / a 2-line key=value table) are skipped — they are
    not per-op/per-impl result tables that need an archive pointer.
  - The archive-link search window spans from the table's last row up to (and including) the line just
    before the NEXT table or heading — generous enough that a link in the table's `>` 关键读法 pointer
    or a `归档:` line a couple lines down counts. (A link anywhere in that contiguous post-table block
    satisfies the convention.)
  - Link recognized by markdown-link target containing `kernels/` or ending `verification.json`.

Usage:  python3 report_archive_link_lint.py <REPORT.md> [--json]
Exit:   0 = clean; 2 = a front core-results table is missing its archive link.
"""
import re
import sys
import json

# A markdown link whose target points at an archived kernel dir / verification.json.
_ARCHIVE_LINK = re.compile(r"\]\(([^)]*(?:kernels/[^)]*|verification\.json))\)")


def _front_region(lines):
    """(start, end) [0-based, end exclusive] of the front core-data region: the FIRST top-level
    `## ` section. Identical contract to report_not_log_lint._front_region (intro excluded)."""
    h2 = [i for i, ln in enumerate(lines) if re.match(r"##\s+\S", ln) and not ln.startswith("###")]
    if not h2:
        return (0, len(lines))
    start = h2[0]
    end = h2[1] if len(h2) >= 2 else len(lines)
    return (start, end)


def _is_table_row(s):
    return s.lstrip().startswith("|")


def _is_separator(s):
    return _is_table_row(s) and set(s.strip()) <= set("|-: ")


def _tables(front):
    """Yield (header_lineidx, last_row_lineidx) for each contiguous table block in `front`
    (list of (idx, line)). A 'core-results' table = header + separator + >=2 data rows."""
    i = 0
    n = len(front)
    while i < n:
        idx, ln = front[i]
        if _is_table_row(ln):
            # collect the contiguous |-row block
            j = i
            rows = []
            while j < n and _is_table_row(front[j][1]):
                rows.append(front[j])
                j += 1
            data_rows = [r for r in rows[1:] if not _is_separator(r[1])]
            has_sep = any(_is_separator(r[1]) for r in rows)
            if has_sep and len(data_rows) >= 2:
                yield (rows[0][0], rows[-1][0])
            i = j
        else:
            i += 1


def lint(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    start, end = _front_region(lines)
    front = list(enumerate(lines))[start:end]
    misses = []
    for hdr, last in _tables(front):
        # search window: lines after the table's last row, up to (not incl.) the next table/heading.
        win_start = last + 1
        win_lines = []
        for i in range(win_start, end):
            ln = lines[i]
            s = ln.strip()
            if _is_table_row(s) or s.startswith("#"):
                break
            win_lines.append(ln)
        block = "\n".join(win_lines)
        if not _ARCHIVE_LINK.search(block):
            misses.append((hdr + 1, "A:table-missing-archive-link", lines[hdr].strip()[:120]))
    return misses


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if not args:
        print("usage: report_archive_link_lint.py <REPORT.md> [--json]", file=sys.stderr)
        sys.exit(64)
    misses = lint(args[0])
    if as_json:
        print(json.dumps({"report_archive_link_violations": len(misses), "misses": misses},
                         ensure_ascii=False, indent=2))
    else:
        print(f"report_archive_link_violations={len(misses)}  ({args[0]})")
        for ln, kind, ctx in misses:
            print(f"  NO-LINK  L{ln}  {kind}  [table header]  {ctx}")
        if not misses:
            print("  clean — every front core-results table carries its archive link.")
    sys.exit(2 if misses else 0)


if __name__ == "__main__":
    main()
