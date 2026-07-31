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
"""convert_docs_to_okf.py — migrate porter's remaining WHOLE-DOC knowledge references to OKF Runbook cards.

Completeness pass (2026-07-13): the family converters covered id-indexed families (EC/PB/OL/patterns/…),
but several *whole-document* knowledge references were never carried over. This converter ports each such
doc as ONE faithful whole-body Runbook card (no field extraction, no LLM). Deterministic.

Scope decision (explicit allow-list below — nothing is auto-discovered):
  MIGRATE  = genuine transferable technical/methodology knowledge (fa_class lowering refs, exploration
             diagnostics, benchmark/regression methodology, cann-family→arch classification, build-system).
  NOT here = process/discipline docs (ALWAYS_LOADED_RULES, ANTI_PRESSURE, GATE_CONTRACT, KB_WRITING_DISCIPLINE,
             OUTPUT_PROJECT_LAYOUT), session retrospectives, agent configs, scope/layout, KB_USAGE_LOG,
             asc-devkit-derived big refs + migration/** (consume wiki), external_imports/** (vendored external
             snapshot). These are recorded as not-migrate in the traceability doc, NOT ported.

Home: runbooks/operator-optimization (bundle=runbooks → EMPTY_OK, no resource — porter-internal, no upstream URL).
These are MECHANICAL faithful ports (like hardware/patterns): confidence=single_run, NO classified_by (not LLM).

Usage: convert_docs_to_okf.py <kb_root> [--migration-date ISO]
"""
import re
import os
import argparse

from migrate_cards_to_okf_v1 import write_okf_v1_card

# (relpath under kb/, slug, title, tags, extra note) — explicit allow-list, faithful whole-body port.
MANIFEST = [
    ("target/ascendc/fa_class/cross_core_sync.md", "fa-cross-core-sync-workspacequeue",
     "Cross Core Sync 与 WorkspaceQueue 详细参考", ["fa-class", "cross-core-sync", "workspacequeue", "aic-aiv", "sync"]),
    ("target/ascendc/fa_class/cv_lowering.md", "fa-cv-fused-init-process-lowering",
     "C/V 融合计算总参考:Init 与 Process", ["fa-class", "cv-fusion", "aic-aiv", "lowering", "reference"]),
    (
        "target/ascendc/fa_class/cv_reference_concrete_params.md",
        "fa-concrete-lowering-params-507015",
        "FA 类具体 lowering 参数(507015 三处高危决策)",
        ["fa-class", "507015", "matmul-primitive", "softmax-online", "concrete-params"],
    ),
    (
        "target/ascendc/cann_classification/README.md",
        "cann-op-family-arch-classification",
        "CANN 算子族 → 架构分类(反纯 VEC 作弊门)",
        ["cann-classification", "architecture-gate", "anti-cheat", "cube-vec-fused", "ol-188"],
    ),
    ("target/ascendc/build_system/README.md", "build-system-pattern-bsp-subdir",
     "Build-System Pattern(BSP)子库说明", ["build-system", "bsp", "launch-glue"]),
    ("shared/exploration/STRUCTURAL_DIMENSIONS.md", "explore-structural-dimensions-5axes",
     "AscendC kernel 优化的 5 个结构维度", ["exploration", "optimization", "structural-dimensions", "methodology"]),
    (
        "shared/exploration/GROUNDING_CHAINS.md",
        "explore-grounding-chains-msprof-to-diagnosis",
        "Grounding Chains:msprof 指标 → 诊断 → 候选维度",
        ["exploration", "msprof", "diagnosis", "grounding-chains", "methodology"],
    ),
    ("shared/exploration/EXPLORATION_PROTOCOL.md", "explore-bounded-exploration-protocol-9step",
     "有界探索协议(模式穷尽后的 9 步找优化)", ["exploration", "protocol", "optimization", "methodology"]),
    ("shared/BENCHMARK_METHODOLOGY.md", "bench-methodology-profiling-first",
     "基准测试方法论(Profiling-First)", ["benchmark", "methodology", "profiling", "msprof", "precision"]),
    ("shared/REGRESSION_METHODOLOGY.md", "regression-methodology-3way-oracle",
     "重构/升级防 regression 方法论(3-way oracle)", ["regression", "methodology", "oracle", "verification"]),
]


def yq(s):
    return '"' + " ".join(str(s).split()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def strip_frontmatter(text):
    m = re.match(r"\s*---\n.*?\n---\n", text, re.S)
    return text[m.end():] if m else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kb_root")
    ap.add_argument("--migration-date", default="2026-07-13T00:00:00+08:00")
    a = ap.parse_args()
    kb_src = os.path.dirname(a.kb_root.rstrip("/"))  # kb/okf -> kb
    out_dir = os.path.join(a.kb_root, "runbooks", "operator-optimization")
    os.makedirs(out_dir, exist_ok=True)
    made, recon = 0, []
    for rel, slug, title, tags in MANIFEST:
        src = os.path.join(kb_src, rel)
        if not os.path.isfile(src):
            recon.append((rel, "skip:source-missing"))
            continue
        body = strip_frontmatter(open(src, encoding="utf-8").read()).strip()
        # neutralize dead relative links (they point at porter's old tree / .claude refs).
        # ONLY strip relative / scheme-less .md targets — never touch external http(s):// links
        # (over-stripping those would break "faithful port" + evidence traceability, codex 2026-07-13).
        body = re.sub(r"\[([^\]]+)\]\((?!\w+://)(?:\.\.?/[^)]*|[^):]*\.md)\)", r"\1", body)
        if not body:
            recon.append((rel, "skip:empty"))
            continue
        desc = " ".join(re.sub(r"[`*#>|]", "", body).split())[:200]
        tt, seen = [], set()
        for t in tags + ["ascendc"]:
            if t.lower() not in seen:
                seen.add(t.lower())
                tt.append(t)
        fm = ["---", "type: Runbook", "title: " + yq(title), "description: " + yq(desc),
              "confidence: single_run", "original_id: doc/%s" % rel,
              "tags: [%s]" % ", ".join(tt),
              "timestamp: '%s'" % a.migration_date, "timestamp_inferred: true", "---", ""]
        out = "\n".join(fm) + (body if body.lstrip().startswith("#") else "# " + title + "\n\n" + body) + \
              "\n\n<!-- 迁移自 porter kb/%s(整档忠实搬运,convert_docs_to_okf.py)。跨 op 参考/方法论知识,非机械家族。 -->\n" % rel
        write_okf_v1_card(os.path.join(out_dir, slug + ".md"), out)
        recon.append((rel, "migrated:operator-optimization"))
        made += 1

    # fail-closed: the MANIFEST is a fixed allow-list — a missing source is a completeness regression,
    # not a silent skip. Abort so the caller can't ship a partial completeness pass (codex 2026-07-13).
    missing = [rel for rel, d in recon if d.startswith("skip:")]
    if missing:
        import sys
        sys.stderr.write("FATAL: allow-list source(s) missing/empty, completeness pass incomplete: %s\n" %
                         ", ".join(missing))
        sys.exit(2)

    mdir = os.path.join(a.kb_root, "_migration")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "docs_reconcile.md"), "w", encoding="utf-8") as f:
        f.write("# 整档知识参考 → OKF Runbook 迁移核销 (convert_docs_to_okf.py)\n\n"
                "完整性盘点(2026-07-13)补迁:family 转换器只覆盖 id 索引家族,这些**整篇文档型**知识参考此前漏了。\n"
                "整档忠实搬运成 Runbook(operator-optimization,EMPTY_OK 免 resource)。\n\n| 源文档 | disposition |\n|---|---|\n")
        for rel, d in recon:
            f.write("| %s | %s |\n" % (rel, d))
    print("docs → operator-optimization: migrated %d (共 %d 条) -> _migration/docs_reconcile.md" % (made, len(recon)))


if __name__ == "__main__":
    main()
