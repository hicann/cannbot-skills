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

"""convert_cand_to_okf.py — migrate porter's unverified candidates (CAND-*) to OKF field-notes inferred tier.

Per user decision (2026-07-13): CAND → field-notes `inferred` tier — `status: stub` + `confidence: inferred`
so kb-query's default `--status verified` EXCLUDES them (they only surface with `--status all/stub`).
Faithful whole-body port (CAND entries use **Symptom**/**Trigger**/**Promote when**/**Risks** headings).
Deterministic, no LLM. Destination: runbooks/field-notes/inferred/ (bundle=runbooks → EMPTY_OK, no resource).

Usage: convert_cand_to_okf.py <candidates.md> <kb_root> [--migration-date ISO]
"""
import re
import os
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card


def slug(cid, title):
    s = re.sub(r"[^A-Za-z0-9]+", "-", re.sub(r"`", "", title)).strip("-").lower()
    return ("cand-%s-%s" % (cid.lower(), re.sub(r"-+", "-", s)[:42].strip("-"))).strip("-")


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def first_signal(body):
    for k in ("Symptom", "Trigger", "Recommendation"):
        m = re.search(r"\*\*" + k + r"\*\*[^\n]*?:?\s*(.+)", body)
        if m and m.group(1).strip():
            return re.sub(r"[`*]", "", m.group(1)).strip()[:160]
    for ln in body.splitlines():
        t = ln.strip()
        if t and not t.startswith(("```", "#", "-")):
            return re.sub(r"[`*]", "", t)[:160]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("kb_root")
    ap.add_argument("--migration-date", default="2026-07-13T00:00:00+08:00")
    a = ap.parse_args()
    text = open(a.src, encoding="utf-8").read()
    out_dir = os.path.join(a.kb_root, "runbooks", "field-notes", "inferred")
    os.makedirs(out_dir, exist_ok=True)
    parts = re.split(r"(?m)^##\s+CAND-([A-Za-z0-9][A-Za-z0-9-]*)\s*:\s*(.+?)\s*$", text)
    made, seen, recon = 0, set(), []
    for i in range(1, len(parts), 3):
        cid, title, body = parts[i], parts[i + 1].strip(), parts[i + 2]
        if cid in seen:
            recon.append((cid, "skip:dup"))
            continue
        seen.add(cid)
        body = re.sub(r"^\s*-{3,}\s*$", "", body, flags=re.M).strip()
        body = re.sub(r"\[([^\]]+)\]\((?:\.\.?/[^)]*|[^)]*\.md)\)", r"\1", body)  # kill dead relative links
        if not body:
            recon.append((cid, "skip:empty"))
            continue
        desc = " ".join(re.sub(r"[`*#>|]", "", body).split())[:200]
        idents = re.findall(r"`([A-Za-z_][A-Za-z0-9_.]{2,})`", title + " " + body)
        tags = ["candidate", "inferred"] + [x.lower() for x in idents[:5]] + ["cand-%s" % cid.lower()]
        tt, s = [], set()
        for t in tags:
            if t.lower() not in s:
                s.add(t.lower())
                tt.append(t)
        fm = ["---", "type: build_card", "title: " + yq(title), "description: " + yq(desc),
              "phenomenon: build_failure", "signal:", "  - " + yq(first_signal(body) or title),
              "confidence: inferred",          # 未验证候选:最低证据档
              "status: stub",                  # 默认 --status verified 会排除它
              "original_id: CAND-%s" % cid, "tags: [%s]" % ", ".join(tt),
              "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
        out = (
            "\n".join(fm)
            + "## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）\n\n"
            + body
            + (
                "\n\n<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/"
                "candidates.md（CAND-%s，convert_cand_to_okf.py）。status=stub "
                "未验证,待复现后 promote。 -->\n" % cid
            )
        )
        write_okf_v1_card(os.path.join(out_dir, slug(cid, title) + ".md"), out)
        recon.append((cid, "migrated:inferred(stub)"))
        made += 1

    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write(
            "# field-notes / inferred\n\n未验证候选(CAND-*),`status: stub` —— "
            "默认检索排除,复现够(reproduce_count≥2)后 promote 成正式卡。\n"
        )
    mdir = os.path.join(a.kb_root, "_migration")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "cand_reconcile.md"), "w", encoding="utf-8") as f:
        f.write(
            "# CAND → OKF field-notes/inferred 迁移核销 (status=stub, "
            "confidence=inferred)\n\n迁移 %d 条(默认检索排除,待复现 promote)。"
            "\n\n| CAND | disposition |\n|---|---|\n" % made
        )
        for cid, d in recon:
            f.write("| CAND-%s | %s |\n" % (cid, d))
    print("CAND → field-notes/inferred: migrated %d (共 %d 条,status=stub 默认排除)" % (made, len(recon)))


if __name__ == "__main__":
    main()
