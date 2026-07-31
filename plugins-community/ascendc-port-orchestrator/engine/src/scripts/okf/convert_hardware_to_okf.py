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

"""convert_hardware_to_okf.py — migrate porter's hardware FACT files to OKF reference/hardware cards.

Only the fact files migrate (chip specs, probe findings, source references) → reference/hardware/ as `type: Guide`
cards (faithful whole-file body). The hardware/ top-level PROCESS docs (EXPERT_QUESTIONS, HIASCEND_DOC_URLS,
HW_EXTRACTION_PROMPT, INDEX, INTERNAL_QUERY_QUEUE) are orchestration/registry — NOT migrated. Deterministic.

Note: porter's hardware cards have no upstream @40hex commit; reference/hardware is NOT a git bundle
(GIT_BUNDLES = {asc-devkit, ops}), so no `resource`/pin is required. reference cards DO require title+description.

Usage: convert_hardware_to_okf.py <hardware dir> <kb_root> [--migration-date ISO]
"""
import re
import os
import glob
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card

SRC_SUBDIRS = ["target", "probe_findings", "source"]


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def strip_frontmatter(text):
    m = re.match(r"\s*---\n.*?\n---\n", text, re.S)
    fm = {}
    if m:
        for line in m.group(0).splitlines():
            kv = re.match(r"([A-Za-z_]+):\s*(.+)", line)
            if kv:
                fm[kv.group(1)] = kv.group(2).strip()
        text = text[m.end():]
    return fm, text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hw_dir")
    ap.add_argument("kb_root")
    ap.add_argument("--migration-date", default="2026-07-13T00:00:00+08:00")
    a = ap.parse_args()
    # runbooks/hardware → bundle="runbooks" ∈ EMPTY_OK_BUNDLES → resource 可空(porter 内部硬件事实无上游 @commit)
    out_dir = os.path.join(a.kb_root, "runbooks", "hardware")
    os.makedirs(out_dir, exist_ok=True)
    made, recon = 0, []
    files = []
    for sub in SRC_SUBDIRS:
        files += sorted(glob.glob(os.path.join(a.hw_dir, sub, "*.md")))
    for f in files:
        if os.path.basename(f) == "index.md":
            continue
        srcfm, body = strip_frontmatter(open(f, encoding="utf-8").read())
        stem = os.path.basename(f)[:-3]
        h1 = re.search(r"(?m)^#\s+(.+)$", body)
        title = (h1.group(1).strip() if h1 else stem).replace("`", "")
        # description: first non-heading paragraph, else the platform/type frontmatter
        desc = ""
        for para in re.split(r"\n\s*\n", body):
            p = re.sub(r"[`*#>|]", "", para).strip()
            if p and not p.startswith("---"):
                desc = " ".join(p.split())[:200]
                break
        if not desc:
            desc = " ".join("%s=%s" % (k, v) for k, v in srcfm.items())[:200] or title
        body = re.sub(r"\[([^\]]+)\]\((?:\.\.?/[^)]*|[^)]*\.md)\)", r"\1", body)  # kill dead relative links
        plat = srcfm.get("platform", "") or srcfm.get("soc_version", "")
        tags = ["hardware", os.path.basename(os.path.dirname(f))] + ([plat.lower()]
                                             if plat else []) + [stem.lower().replace("_", "-")[:30]]
        tt, s = [], set()
        for t in tags:
            if t and t.lower() not in s:
                s.add(t.lower())
                tt.append(t)
        fm = ["---", "type: Guide", "title: " + yq(title), "description: " + yq(desc),
              "confidence: single_run", "original_id: hw/%s" % stem,
              "tags: [%s]" % ", ".join(tt),
              "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
        out = (
            "\n".join(fm)
            + (body if body.lstrip().startswith("#") else "# " + title + "\n\n" + body)
            + (
                "\n\n<!-- 迁移自 porter kb/hardware/%s（convert_hardware_to_okf.py，"
                "硬件事实→reference/hardware）。 -->\n" % os.path.relpath(f, a.hw_dir)
            )
        )
        # OKF forbids leading-numeric filenames; prefix with the source subdir (probe files start with a date)
        sub_short = {"probe_findings": "probe", "target": "target",
            "source": "src"}.get(os.path.basename(os.path.dirname(f)), "hw")
        fn = sub_short + "-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
        write_okf_v1_card(os.path.join(out_dir, fn + ".md"), out)
        recon.append((stem, "migrated:runbooks/hardware"))
        made += 1

    # bundle root index (reference cards live under a bundle; give it a nav index)
    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("# runbooks / hardware\n\n昇腾芯片规格 + 实测 probe + 来源架构参考（从 porter kb/hardware 迁入的硬件事实卡）。\n\n")
        for stem, _ in recon:
            f.write("- %s\n" % stem)
    mdir = os.path.join(a.kb_root, "_migration")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "hardware_reconcile.md"), "w", encoding="utf-8") as f:
        f.write(
            "# hardware → OKF runbooks/hardware 迁移核销\n\n只迁事实文件;"
            "顶层流程文档不迁。runbooks bundle EMPTY_OK 免 resource。"
            "\n\n| 文件 | disposition |\n|---|---|\n"
        )
        for stem, d in recon:
            f.write("| %s | %s |\n" % (stem, d))
    print("hardware: migrated %d 张 → runbooks/hardware(type: Guide, EMPTY_OK)" % made)


if __name__ == "__main__":
    main()
