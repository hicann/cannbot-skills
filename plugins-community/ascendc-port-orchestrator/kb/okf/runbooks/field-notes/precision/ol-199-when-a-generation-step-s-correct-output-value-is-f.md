---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "When a generation step's correct output value is fixed by an upstream design artifact, ENFORCE it with a deterministic post-emit gate — do not rely on the (LLM) generator reproducing it"
description: "<!-- applies_to_backend: all -->"
phenomenon: precision_issue
signal:
  - "<!-- applies_to_backend: all -->"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-199
timestamp_inferred: true
tags: [ascendc, ol-199]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; scope=op-gen IL-chain / design->emit contract; instance=FA-class tile size`
`verified_on: FA-class IL designer->translator (a5_ops), tile_size_consistency finalize gate`

**Principle**: in a multi-stage generator where an upstream stage produces an authoritative value (a design parameter, a contract field, a shape) and a downstream LLM stage must lower it into the emitted artifact, **advisory prose alone does not make the lowering deterministic** — the LLM may silently substitute a different value (e.g. by verbatim-porting a sibling artifact, or improvising). The fix is a deterministic post-emit gate: mechanically parse the upstream authoritative value AND the emitted value, assert equality, and on mismatch REJECT with a re-emit directive (route back to the generating stage). This converts "hope the LLM complied" into "the harness forces compliance or fails visibly". Pair the hard gate with a self-consistency check (all downstream-derived values trace to the same constant, no residual literals). Whitebox-direct: the harness directs the generation to a known-correct target, it does not gamble.

**Concrete anchor**: FA-class `design/tile_level.block_N` (authoritative, perf-design) vs emitted kernel `FA_BLOCK_N`. Gate = parse design block_N (tuple `block_M,block_N=X,Y` + single forms) vs `constexpr ... (FA_)?BLOCK_N = <int>` in `kernel/*.h` (comment-stripped); mismatch -> violation string + rollback-to-translator directive; defensive (missing design/kernel -> no-op). Registered as an op-class-guarded `extra_finalize_checks` hook.

**Evidence**: FA-class IL `3_FusionAttention` (2026-05-29). Before the gate, disk had 0 kernels with block_N=128 vs 10 with block_N=64 — the designer's tile_level specified 128 but the translator always emitted 64 (no enforcement existed: no mechanical substitution, no finalize tile gate, only an advisory "grep for residual 64" note). Fix (a5_ops PR #263): deterministic `tile_size_consistency` gate + test (9 cases). After the gate (with OL-198 cold-start eviction), a clean run emitted `FA_BLOCK_N=128` in a single translate (disk-verified). Caps at EMIT — gate guarantees the tile matches design; the LLM must still produce a buildable kernel under the directive.

**Other instances (predicted)**: any IL-chain / design->emit contract field (block_M, ring-slot count, dtype dispatch, workspace sizing); any port mode where an upstream reference fixes a value the LLM must reproduce; backward/training op-gen lowering a design spec.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-199（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
