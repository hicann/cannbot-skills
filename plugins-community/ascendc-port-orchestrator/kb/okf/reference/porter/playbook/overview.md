---
schema_version: okf.v1
kind: guide
type: migration_guide
source_family: curated
title: "arch22 → arch35 AscendC Migration and Backward-Generation Reference"
description: "arch22→arch35 迁移与反向生成的参考总纲：给出 L1~L5 分层迁移模型的适用条件与范围（L1 基础适配必做、L2 RegBase 重写、L3 SIMT、L4 需 IsRegbaseSocVersion 或 UB 不足时升级、L5 仅供参考），以及目录地图、被 brief 构建层消费的方式与来源可复现性说明。"
tags: [playbook]
resource: https://gitcode.com/cann/cannbot-skills/blob/master/plugins-community/ascendc-port-orchestrator/kb/okf/reference/porter/playbook/overview.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
# arch22 → arch35 AscendC Migration and Backward-Generation Reference

> **Source**: PR #103 `Ascend/agent-skills` — `ascendc-operator-A5-migration` skill (commit `66637919` on PR branch). Imported into our KB 2026-05-16 by user directive (P113).

This is reference material for arch22 → arch35 migration and backward-generation workers. The
internal route identifier remains `port_a3_to_a5` for compatibility.
It is **not** an executable skill of its own — it's KB content consumed by the
brief construction layer (P114).

## Layered migration model (L1 → L5)

Decision tree at `_SOURCE_SKILL.md`:

| Level | When to apply | Scope |
|---|---|---|
| **L1** | Always (all ops) | 基础适配: independent config + RegBase kernel entry + remove BF16 conditional compile |
| **L2** | Performance-critical / quant Cast chain / overflow control / FP8/HiFloat8 new dtypes | RegBase API rewrite (MicroAPI Register-based) |
| **L3** | Scatter/Gather + simple indexing + no UB transit + high parallelism | SIMT optimization (add SIMT kernel alongside L1 base) |
| **L4** | Tiling needs `IsRegbaseSocVersion` / UB shortage (40KB SIMT DCache) | Out of scope this skill — escalate |
| **L5** | (RegBase advanced features) | Reference only |

## Directory map

```
migration/
├── _SOURCE_SKILL.md                   # the original SKILL.md (workflow + L1/L2/L3 decision tree)
├── l1-implementation-guide.md         # L1: config + arch35 + remove conditional compile
├── l1-l2-implementation-guide.md      # L1+L2 combined walkthrough
├── l2-register-based-guide.md         # L2: MicroAPI / RegBase / RegTensor
├── l3-simt-optimization-guide.md      # L3: SIMT scatter/gather (V351 hw feature)
├── l4-simt-optimization-guide.md      # L4: deeper SIMT + UB DCache
├── l5-register-based-guide.md         # L5: advanced RegBase
├── api-overview/                      # API surface map per category
├── cube/                              # Cube unit ops (MatmulImpl etc.)
├── datacopy/                          # DataCopy / DataCopyPad / Pipe APIs
├── memory-base-vector/                # Memory-base VEC ops (Add/Sub/Mul/Exp/Ln/Sqrt/...)
├── reg-base-vector/                   # RegBase MicroAPI VEC ops
├── precision-testing/                 # vendor 2.1 precision testing patterns
├── migration/                         # case studies — concrete arch22→arch35 op migrations
├── simd/                              # SIMD mode reference
└── simt/                              # SIMT mode reference (V351-specific)
```

## How this content gets used (P114 — brief injection plugin)

Per Zheng directive 2026-05-16: brief construction must be **plugin form**, not
if-else. The plan:

1. `src/scripts/orchestrator/plugins/port_a3/migration_level.py` —
   plugin that maps `op_meta` → `MigrationLevel ∈ {L1, L2, L3, L4}`. Decision
   rules from `_SOURCE_SKILL.md` decision tree. **OCP**: new heuristics added
   inside the plugin, not new if-else at brief-construction call sites.

2. `kw_brief.py` injects, per resolved level, the corresponding `migration/
   l{N}-*.md` references into the worker brief's KB-references section.
   Level-specific subdirs (cube/ datacopy/ simt/ etc.) are injected per
   op-pattern lookup.

This way the brief is **per-op-tailored**: L1 op only gets L1 guide; L3-eligible
op gets L1 + L3 guides + simt/ subdir; etc. No worker sees 4.2 MB of irrelevant
material; no worker misses the L3 SIMT opportunity when applicable.

## Provenance + reproducibility

Source repo: `https://gitcode.com/Ascend/agent-skills`
PR: `#103` (新增A5迁移skill)
Branch: `feature/ascendc-A5-migration` (squashed to master)
Original author: ansen-changan
Date imported: 2026-05-16

If upstream updates, re-import via:
```bash
cd /tmp && git clone https://gitcode.com/Ascend/agent-skills.git pr103-skills
cd pr103-skills && git fetch origin merge-requests/103/head:pr103 && git checkout pr103
rsync -av --delete skills/ascendc-operator-A5-migration/references/ \
  <a5_ops>/src/skills/references/target/ascendc/migration/
```
