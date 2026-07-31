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

"""convert_ol_to_okf.py — migrate porter's OL family (OPERATIONAL_KNOWLEDGE.md) to OKF, Category-routed.

M1 converter for OL (see 迁移OKF-设计方案 §3.6). OL is the messiest family: it routes to MULTIPLE
destinations by the entry's `- **Category**:` field. This converter handles the DETERMINISTIC,
field-note-bound categories; SKIPs orchestration/discipline rules; and QUEUES everything else
(compound Category, no Category, runbook-OPT-bound, api_glossary) into a human-review manifest —
matching §3.6's "带 Category 确定性归类 + 其余进人工复核队列". Every OL gets a disposition.

Not auto-migrated here (queued for review): algorithm_selection/optimization_technique/pipeline_design/
kernel_design (→ runbook OPT, needs the operator-optimization card format), api_glossary (→ reference
bundle), compound categories, and entries with no Category. Deterministic, no LLM.

Metadata: type=<*_card>, confidence=single_run (floor, never raised), NO reproduce_count/severity
(OL gives neither), original_id=OL-N, timestamp=migration date + inferred flag.

Usage: convert_ol_to_okf.py <OPERATIONAL_KNOWLEDGE.md> <kb_root> [--migration-date ISO]
"""
import re
import os
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card

SKIP = "__skip__"
REVIEW = "__review__"
# category (single token, lowercased) -> (field-notes domain, okf type) | SKIP | REVIEW
ROUTE = {
    "process": SKIP, "process_rules": SKIP, "trust_calibration": SKIP, "workflow": SKIP,
    "orchestration": SKIP, "environment": SKIP,
    "platform_bug": ("build", "build_card"), "platform_compat": ("build", "build_card"),
    "platform_constraint": ("build", "build_card"), "arch_compat": ("build", "build_card"),
    "precision": ("precision", "precision_card"),
    "performance": ("perf", "perf_card"), "performance_analysis": ("perf", "perf_card"),
    "measurement": ("perf", "perf_card"), "profiling_interpretation": ("perf", "perf_card"),
    "algorithm_selection": REVIEW, "optimization_technique": REVIEW, "pipeline_design": REVIEW,
    "kernel_design": REVIEW, "api_glossary": REVIEW, "conditional_insight": REVIEW,
}
PHENOM = {"build_card": "build_failure", "precision_card": "precision_issue", "perf_card": "perf_regression"}


def slugify(title, n):
    s = re.sub(r"`", "", title)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    s = re.sub(r"-+", "-", s)[:50].strip("-")
    return "ol-%s-%s" % (n, s) if s else "ol-%s" % n


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def field(body, name):
    m = re.search(r"-\s+\*\*" + re.escape(name) + r"\*\*\s*:\s*(.*?)(?=\n-\s+\*\*|\n```|\n---|\Z)", body, re.S)
    return m.group(1).strip() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("kb_root")
    ap.add_argument("--migration-date", default="2026-07-10T00:00:00+08:00")
    a = ap.parse_args()
    text = open(a.src, encoding="utf-8").read()
    parts = re.split(r"(?m)^##\s+OL-(\d+)\s*:\s*(.+?)\s*$", text)

    counts = {"migrated": 0, "skip": 0, "review": 0}
    recon = []
    for i in range(1, len(parts), 3):
        n, title, body = parts[i], parts[i + 1].strip(), parts[i + 2]
        cat_raw = field(body, "Category").strip().lower()
        lesson = field(body, "Lesson")
        trigger = field(body, "Trigger")
        evidence = field(body, "Evidence")

        # routing
        if not cat_raw:
            recon.append((int(n), title, "review:no-category", ""))
            counts["review"] += 1
            continue
        if "/" in cat_raw:
            recon.append((int(n), title, "review:compound(%s)" % cat_raw, ""))
            counts["review"] += 1
            continue
        route = ROUTE.get(cat_raw)
        if route is None:
            recon.append((int(n), title, "review:unknown-category(%s)" % cat_raw, ""))
            counts["review"] += 1
            continue
        if route == SKIP:
            recon.append((int(n), title, "skip:orchestration-rule(%s)" % cat_raw, ""))
            counts["skip"] += 1
            continue
        if route == REVIEW:
            recon.append((int(n), title, "review:needs-runbook-or-reference(%s)" % cat_raw, ""))
            counts["review"] += 1
            continue

        domain, otype = route
        if not lesson:   # field-note needs substance
            recon.append((int(n), title, "review:no-lesson(%s)" % cat_raw, ""))
            counts["review"] += 1
            continue

        out_dir = os.path.join(a.kb_root, "runbooks", "field-notes", domain)
        os.makedirs(out_dir, exist_ok=True)
        signal = (trigger or lesson).splitlines()[0][:160]
        desc = " ".join(lesson.split())[:200]
        idents = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", title + " " + lesson)
        codes = re.findall(r"\b\d{6}\b", body)
        tags = []
        for t in list(codes) + [x.lower() for x in idents]:
            if t.lower() not in [z.lower() for z in tags]:
                tags.append(t)
        tags = tags[:7] + ["ascendc", cat_raw, "ol-%s" % n]
        fn = slugify(title, n) + ".md"
        fm = ["---", "type: " + otype, "title: " + yq(title), "description: " + yq(desc),
              "phenomenon: " + PHENOM[otype], "signal:", "  - " + yq(signal),
              "confidence: single_run", "original_id: OL-%s" % n,
              "tags: [%s]" % ", ".join(tags),
              "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
        b = ["## 现象 / 触发", trigger.strip() or "(见教训)", "",
             "## 教训 / 根因", lesson.strip()]
        if evidence:
            b += ["", "## 证据", evidence.strip()]
        b += [
            "",
            (
                "<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-%s（category=%s，"
                "convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count "
                "未升格。 -->" % (n, cat_raw)
            ),
        ]
        write_okf_v1_card(os.path.join(out_dir, fn), "\n".join(fm) + "\n".join(b) + "\n")
        recon.append((int(n), title, "migrated:%s/%s" % (domain, otype), "runbooks/field-notes/%s/%s" % (domain, fn)))
        counts["migrated"] += 1

    mdir = os.path.join(a.kb_root, "_migration")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "ol_reconcile.md"), "w", encoding="utf-8") as f:
        f.write("# OL → OKF 迁移核销 (convert_ol_to_okf.py, §3.6 Category-routed)\n\n")
        f.write("migrated=%d  skip(orchestration-rule)=%d  review-queue=%d  total=%d\n\n" %
                (counts["migrated"], counts["skip"], counts["review"], len(recon)))
        f.write("| OL | title | disposition | card |\n|---|---|---|---|\n")
        for n, t, d, c in sorted(recon):
            f.write("| OL-%s | %s | %s | %s |\n" % (n, t.replace("|", "\\|")[:55], d, c))
    print("[OL] migrated=%d skip=%d review=%d total=%d -> _migration/ol_reconcile.md" %
          (counts["migrated"], counts["skip"], counts["review"], len(recon)))


if __name__ == "__main__":
    main()
