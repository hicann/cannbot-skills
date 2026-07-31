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

"""convert_family_to_okf.py — migrate porter's phenomenon-card families (EC / PB) to OKF build_cards.

M1 converter for the EC/PB families (see 迁移OKF-设计方案 §3.1/§3.2). Deterministic, no LLM. OL is NOT
handled here (needs Category-aware routing — separate converter).

FAITHFUL WHOLE-BODY PORT (rewritten 2026-07-13 after opus/codex review found the old field-extraction
approach (a) DROPPED 29 entries whose Root/Fix were bold *paragraphs* not `- **bullets**`, gated out as
"no-root/fix"; and (b) reconstructed each card from only the Error/Root/Fix/Note bullets, DISCARDING every
correcting/updating bullet — so e.g. EC-23 presented a since-refuted "DataCopyPad crashes on A5" as current
fact while the source's 3 later cross-op confirmations that "V351 works fine, guard is stale" were dropped).
Now: the ENTIRE source entry body is ported faithfully — ALL bullets kept (incl. corrections/cross-op
confirmations/evidence), with exactly TWO deterministic cleanups and nothing else: (1) strip `---` section
separators, (2) neutralize dead relative / scheme-less `.md` links (they point at porter's old tree and would
404 in kb/okf). It is NOT byte-verbatim (those two cleanups), but no knowledge content is altered or dropped.
Labelled bullets are used ONLY to derive the retrieval frontmatter (signal/description/tags), never to gate
migration or to truncate the body. Every non-empty entry migrates; nothing is silently dropped.

Metadata rules (§3.2, "只映射、绝不编造/升格"):
  - type=build_card, confidence=single_run (conservative floor; only lowered, never raised)
  - NO reproduce_count / severity unless in source (EC/PB give neither → omitted)
  - original_id = <FAM>-N preserved; tags include <fam>-N; timestamp = migration date + inferred flag

Usage: convert_family_to_okf.py --family EC|PB <SRC.md> <out_dir:.../runbooks/field-notes/build> [--migration-date ISO]
"""
import re
import sys
import os
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card

FAMILIES = {
    "EC": dict(idre=r"EC-(\d+)", signal=["Error pattern", "Error", "Symptom"], phenomenon="build_failure"),
    "PB": dict(idre=r"PB-(\d+)", signal=["Symptom", "Error pattern", "Error"], phenomenon="build_failure"),
}


def slugify(title, fam, n):
    s = re.sub(r"`", "", title)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    s = re.sub(r"-+", "-", s)[:50].strip("-")
    return "%s-%s-%s" % (fam.lower(), n, s) if s else "%s-%s" % (fam.lower(), n)


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def labelled(body, name):
    """First line of a labelled section — matches BOTH `- **Name**:` bullets AND `**Name (…)**:`
    bold paragraphs (the form the old regex missed). Used only for frontmatter, never to gate.
    """
    m = re.search(r"(?m)^\s*-?\s*\*\*" + re.escape(name) + r"[^*\n]*\*\*\s*:?\s*(.+)", body)
    return m.group(1).strip() if m else ""


def strip_fences(text):
    return re.sub(r"```[a-zA-Z0-9]*\n?|```", "", text).strip()


def clip(s, maxlen=160):
    return re.sub(r"[`*]", "", " ".join(s.split()))[:maxlen]


def clean_body(body):
    body = re.sub(r"(?m)^\s*-{3,}\s*$", "", body).strip()          # section separators
    body = re.sub(r"\[([^\]]+)\]\((?!\w+://)(?:\.\.?/[^)]*|[^):]*\.md)\)", r"\1", body)  # dead relative/.md links
    return body.strip()


def first_meaningful(body):
    for ln in strip_fences(body).splitlines():
        t = re.sub(r"^[\s\-*>]+", "", ln).strip()
        if t and not t.startswith(("```", "#", "applies_to", "verified_on", "source:")):
            return t
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--migration-date", default="2026-07-10T00:00:00+08:00")
    a = ap.parse_args()
    fam, cfg = a.family, FAMILIES[a.family]
    os.makedirs(a.out_dir, exist_ok=True)
    text = open(a.src, encoding="utf-8").read()

    parts = re.split(r"(?m)^###\s+" + cfg["idre"] + r"\s*:\s*(.+?)\s*$", text)
    recon, made = [], 0
    for i in range(1, len(parts), 3):
        n, title, raw = parts[i], parts[i + 1].strip(), parts[i + 2]
        body = clean_body(raw)
        if not body:
            recon.append((int(n), title, "skipped:empty-body", ""))
            continue

        # frontmatter (retrieval facets) — derived, never gating
        sig = ""
        for name in cfg["signal"]:
            sig = labelled(body, name)
            if sig:
                break
        signal = [clip(sig)] if sig else ([clip(first_meaningful(body))] or ["(see body)"])
        desc = clip(first_meaningful(body) or title, 200) or title
        codes = re.findall(r"\b\d{6}\b", body)
        idents = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", title + " " + body[:400])
        tags = []
        for t in list(codes) + [x.lower() for x in idents]:
            if t.lower() not in [z.lower() for z in tags]:
                tags.append(t)
        tags = tags[:8] + ["ascendc", "%s-%s" % (fam.lower(), n)]

        fn = slugify(title, fam, n) + ".md"
        fm = ["---", "type: build_card", "title: " + yq(title), "description: " + yq(desc),
              "phenomenon: " + cfg["phenomenon"], "signal:"] + ["  - " + yq(s) for s in signal] + [
              "confidence: single_run", "original_id: %s-%s" % (fam, n),
              "tags: [%s]" % ", ".join(tags),
              "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
        # FAITHFUL: whole entry body verbatim (all bullets incl. corrections/cross-op confirmations)
        out = (
            "\n".join(fm)
            + "## 条目正文（忠实搬运，含全部更正/佐证 bullet）\n\n"
            + body
            + (
                "\n\n<!-- 迁移自 porter kb/target/ascendc/（%s-%s，"
                "convert_family_to_okf.py，M1，整档忠实搬运）。confidence/"
                "severity/reproduce_count 未升格。 -->\n" % (fam, n)
            )
        )
        write_okf_v1_card(os.path.join(a.out_dir, fn), out)
        recon.append((int(n), title, "migrated", "runbooks/field-notes/build/" + fn))
        made += 1

    kb_root = os.path.dirname(os.path.dirname(os.path.dirname(a.out_dir.rstrip("/"))))
    mdir = os.path.join(kb_root, "_migration")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "%s_reconcile.md" % fam.lower()), "w", encoding="utf-8") as f:
        f.write("# %s → OKF 迁移核销 (convert_family_to_okf.py --family %s)\n\n" % (fam, fam))
        f.write("| %s | title | disposition | card |\n|---|---|---|---|\n" % fam)
        for n, t, d, c in sorted(recon):
            f.write("| %s-%s | %s | %s | %s |\n" % (fam, n, t.replace("|", "\\|")[:60], d, c))
    print("[%s] migrated %d/%d entries; reconciliation -> _migration/%s_reconcile.md" %
          (fam, made, len(recon), fam.lower()))


if __name__ == "__main__":
    main()
