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

"""convert_patterns_to_okf.py — migrate porter's P-P/F-P/F-AP pattern family to OKF runbook OPT cards.

Fills kb/okf/runbooks/operator-optimization/ (which was an empty scaffold). Per user decision
(2026-07-13): type=Runbook, keep the original id (P-P5/F-AP1) as original_id + filename, tag
positive(P-P/F-P)=optimization / anti(F-AP)=anti-pattern; faithful port of the porter body
(Severity → frontmatter, body = the pattern description). Deterministic, no LLM.

Sources: kb/target/ascendc/patterns/domains/*.md — each `### <ID>: <name>` + `**Severity**: X` + body.
Metadata rules (§3.2): confidence=single_run(floor), original_id preserved, no fabricated fields.

Usage: convert_patterns_to_okf.py <patterns/domains dir> <out_dir> [--migration-date ISO]
"""
import re
import os
import glob
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card

SEV_OK = {"low", "medium", "high", "critical"}


def slug(pid, name):
    s = re.sub(r"[^A-Za-z0-9]+", "-", re.sub(r"`", "", name)).strip("-").lower()
    return "%s-%s" % (pid.lower(), re.sub(r"-+", "-", s)[:48].strip("-"))


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--migration-date", default="2026-07-13T00:00:00+08:00")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    made, seen, recon = 0, set(), []
    for f in sorted(glob.glob(os.path.join(a.domains_dir, "*.md"))):
        domain = os.path.basename(f)[:-3]
        text = open(f, encoding="utf-8").read()
        parts = re.split(r"(?m)^#{2,4}\s+(P-P\d+|F-P\d+|F-AP\d+)\s*:\s*(.+?)\s*$", text)
        for i in range(1, len(parts), 3):
            pid, name, body = parts[i], parts[i + 1].strip(), parts[i + 2]
            if pid in seen:
                recon.append((pid, "skip:dup"))
                continue
            seen.add(pid)
            # pull severity out of the body -> frontmatter; strip meta lines from the body
            m = re.search(r"\*\*Severity\*\*\s*:\s*([A-Za-z]+)", body)
            sev = m.group(1).lower() if m else ""
            sev = sev if sev in SEV_OK else ""
            body_clean = re.sub(r"(?m)^\s*\*\*(Severity|Updated)\*\*\s*:.*$", "", body).strip()
            body_clean = re.sub(r"^\s*-{3,}\s*$", "", body_clean, flags=re.M).strip()
            # neutralize dead relative links [text](../x) / [text](x.md) — they point at porter's old tree
            body_clean = re.sub(r"\[([^\]]+)\]\((?:\.\.?/[^)]*|[^)]*\.md)\)", r"\1", body_clean)
            if not body_clean:
                recon.append((pid, "skip:empty-body"))
                continue
            desc = " ".join(re.sub(r"[`*|]", "", body_clean).split())[:200]
            anti = pid.startswith("F-AP")
            kindtag = "anti-pattern" if anti else "optimization"
            idents = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", name + " " + body_clean)
            tags = [domain, kindtag] + [x.lower() for x in idents[:5]] + [pid.lower(), "ascendc"]
            # dedup tags, keep order
            tt, tseen = [], set()
            for t in tags:
                if t.lower() not in tseen:
                    tseen.add(t.lower())
                    tt.append(t)
            fm = ["---", "type: Runbook", "title: " + yq(name),
                  "description: " + yq(desc)]
            if sev:
                fm.append("severity: " + sev)
            fm += ["confidence: single_run", "original_id: " + pid,
                   "tags: [%s]" % ", ".join(tt),
                   "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
            heading = "## 反模式" if anti else "## 优化点"
            out = (
                "\n".join(fm)
                + heading
                + "\n\n"
                + body_clean
                + (
                    "\n\n<!-- 迁移自 porter kb/target/ascendc/patterns/domains/"
                    "%s.md（%s，convert_patterns_to_okf.py）。confidence "
                    "未升格。 -->\n" % (domain, pid)
                )
            )
            write_okf_v1_card(os.path.join(a.out_dir, slug(pid, name) + ".md"), out)
            recon.append((pid, "migrated:%s(%s)" % (domain, kindtag)))
            made += 1

    # out_dir = <kb_root>/runbooks/operator-optimization → up 2 = <kb_root>
    mdir = os.path.join(os.path.dirname(os.path.dirname(a.out_dir.rstrip("/"))), "_migration")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "patterns_reconcile.md"), "w", encoding="utf-8") as fo:
        fo.write(
            "# P-P/F-P/F-AP → OKF runbook OPT 迁移核销 "
            "(convert_patterns_to_okf.py)\n\ntype=Runbook,正/反用 tag 区分,"
            "保留原 id。\n\n| id | disposition |\n|---|---|\n"
        )

        def key(r):
            m = re.search(r"\d+", r[0])
            return (r[0][:3], int(m.group()) if m else 0)
        for pid, d in sorted(recon, key=key):
            fo.write("| %s | %s |\n" % (pid, d))
    print(
        "patterns → operator-optimization: migrated %d (共 %d 条) -> "
        "_migration/patterns_reconcile.md" % (made, len(recon))
    )


if __name__ == "__main__":
    main()
