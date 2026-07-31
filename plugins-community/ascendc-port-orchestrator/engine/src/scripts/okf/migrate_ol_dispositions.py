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

"""migrate_ol_dispositions.py — LLM-assisted second pass for OL review-queue entries.

The deterministic OL converter (convert_ol_to_okf.py §3.6) routes only Category-tagged entries and
queues the rest for review. The ol-review-queue-classifier workflow then PROPOSES a disposition for
each queued entry. This script materializes the field-note-bound proposals (build/precision/perf_card)
into OKF cards — marked `confidence: inferred` + `classified_by: llm-assisted` so a human can confirm
or delete. Non-field-note proposals (runbook_OPT / reference / not-migrate / ambiguous) are NOT
migrated here (they need the runbook/reference format or human decision).

Usage: migrate_ol_dispositions.py <OPERATIONAL_KNOWLEDGE.md> <kb_root> <dispositions.json> [--migration-date ISO]
"""
import re
import os
import json
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card

DOMAIN = {"build_card": "build", "precision_card": "precision", "perf_card": "perf"}
PHENOM = {"build_card": "build_failure", "precision_card": "precision_issue", "perf_card": "perf_regression"}


def slug(t, n):
    s = re.sub(r"[^A-Za-z0-9]+", "-", re.sub(r"`", "", t)).strip("-").lower()
    return ("ol-%s-%s" % (n, re.sub(r"-+", "-", s)[:50].strip("-"))) or ("ol-%s" % n)


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def field(body, name):
    m = re.search(r"-\s+\*\*" + re.escape(name) + r"\*\*\s*:\s*(.*?)(?=\n-\s+\*\*|\n```|\n---|\Z)", body, re.S)
    return m.group(1).strip() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("kb_root")
    ap.add_argument("dispositions")
    ap.add_argument("--migration-date", default="2026-07-11T00:00:00+08:00")
    a = ap.parse_args()
    with open(a.dispositions, encoding="utf-8") as dispositions_file:
        disp = json.load(dispositions_file)  # {"OL-97":"precision_card", ...}
    with open(a.src, encoding="utf-8") as source_file:
        text = source_file.read()
    parts = re.split(r"(?m)^##\s+OL-(\d+)\s*:\s*(.+?)\s*$", text)
    byid = {"OL-%s" % parts[i]: (parts[i + 1].strip(), parts[i + 2]) for i in range(1, len(parts), 3)}
    made, recon = 0, []
    for oid, otype in disp.items():
        if otype not in DOMAIN or oid not in byid:
            recon.append((oid, "skip:not-found-or-bad-type"))
            continue
        title, body = byid[oid]
        lesson, trigger, evidence = field(body, "Lesson"), field(body, "Trigger"), field(body, "Evidence")
        if not lesson:
            recon.append((oid, "skip:no-lesson"))
            continue
        n = oid.split("-")[1]
        domain = DOMAIN[otype]
        out_dir = os.path.join(a.kb_root, "runbooks", "field-notes", domain)
        os.makedirs(out_dir, exist_ok=True)
        idents = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", title + " " + lesson)
        codes = re.findall(r"\b\d{6}\b", body)
        tags, seen = [], set()
        for t in list(codes) + [x.lower() for x in idents]:
            if t.lower() not in seen:
                seen.add(t.lower())
                tags.append(t)
        tags = tags[:7] + ["ascendc", "ol-%s" % n, "llm-classified"]
        fm = ["---", "type: " + otype, "title: " + yq(title),
              "description: " + yq(" ".join(lesson.split())[:200]),
              "phenomenon: " + PHENOM[otype], "signal:", "  - " + yq((trigger or lesson).splitlines()[0][:160]),
              "confidence: inferred",                       # LLM-classified → lowest confidence
              "classified_by: llm-assisted",                # provenance: NOT deterministic Category
              "original_id: " + oid, "tags: [%s]" % ", ".join(tags),
              "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
        b = ["## 现象 / 触发", trigger.strip() or "(见教训)", "", "## 教训 / 根因", lesson.strip()]
        if evidence:
            b += ["", "## 证据", evidence.strip()]
        b += ["",
            "<!-- LLM-辅助分类迁移(convert 二次 pass, migrate_ol_dispositions.py)。confidence=inferred,待人工确认。原 OL-%s。 -->" % n]
        write_okf_v1_card(
            os.path.join(out_dir, slug(title, n) + ".md"),
            "\n".join(fm) + "\n".join(b) + "\n",
        )
        recon.append((oid, "migrated:%s/%s (inferred)" % (domain, otype)))
        made += 1
    with open(
        os.path.join(a.kb_root, "_migration", "ol_llm_assisted_reconcile.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "# OL LLM-辅助二次迁移核销 (migrate_ol_dispositions.py)\n\n"
            "confidence=inferred,待人工确认。\n\n| OL | disposition |\n|---|---|\n"
        )
        for oid, d in sorted(recon, key=lambda r: int(r[0].split("-")[1])):
            f.write("| %s | %s |\n" % (oid, d))
    print("LLM-assisted migrated %d/%d field-note-bound OL entries (confidence=inferred)" % (made, len(disp)))


if __name__ == "__main__":
    main()
