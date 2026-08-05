---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-class A3 (membase) tile-size — pick largest tile fitting L0 (block_N≥128); per-tile scalar+sync overhead × tile-count dominates device-time"
description: "applies_to: soc=Ascend910_9382(V220/A910C); cann=9.0.0; op_class=fa_class_A3_membase verified_on: soc=Ascend910_9382; cann=9.0.0 Pattern: FA-class A3 (membase) device-time is dominated by per-kv-tile"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382(V220/A910C); cann=9.0.0; op_class=fa_class_A3_membase"
confidence: inferred
status: stub
original_id: CAND-FA-TILESIZE-1
timestamp_inferred: true
tags: [candidate, inferred, aclnnflashattentionscore, npu_fusion_attention, softmaxflashv2tilingfunc, cand-fa-tilesize-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382(V220/A910C); cann=9.0.0; op_class=fa_class_A3_membase`
`verified_on: soc=Ascend910_9382; cann=9.0.0`

Pattern: FA-class A3 (membase) device-time is dominated by **per-kv-tile scalar-instruction issue + cross-core handshake + GM round-trip**, scaling ~O(S^1.8) vs vendor `aclnnFlashAttentionScore` ~O(S^0.75). The prototype designer's default `block_M=block_N=64` makes tile count (hence per-tile fixed overhead) 4× larger than necessary. **Choose the LARGEST tile that fits L0**: for D≤128 fp16, `block_N=128` (L0B `2×(BASE_K×block_N×2B)=64KB` = exactly L0B cap; L0A unchanged at block_M=64; UB vec inputQue depth-2 `block_M×block_N×4B=32KB×2` fits). Do NOT also raise block_M to 128 simultaneously — `block_M=block_N=128` overflows UB (vec inputQue depth-2 = 128KB).

Measured (3_FusionAttention_n9bis, NPU2, same-NPU A/B vs `npu_fusion_attention`): block_N 64→128 ⇒ S=1024 440→252µs (1.74×), S=512 133→82µs (1.62×), S=128 17.3→14.6µs (benchmark 1.17→**1.39× vs vendor = PASS**). **Precision unchanged** (max_abs 1.2e-4 fp16 1-ULP) — tile size is perf-only, no numeric effect. NEGATIVE result (measured, not assumed): hoisting per-tile `SoftMaxFlashV2TilingFunc` to Init = negligible (440→445µs) — the redundant tiling-func recompute is NOT the dominant scalar cost; tile-count is.

Template-assembly guidance: the FA-class A3 AscendC template should use `block_N` = max-fitting-L0 (≥128 for D≤128), not the `64,64` default. Remaining levers (large-S still 0.23× @S=1024, perf-polish not deliverable-blocking): vectorize per-row scale via `Brcb` (not scalar `GetValue` loops in `RowMulsImpl`/`RowDivsImpl`); reduce RING_SLOTS cross-core handshakes; keep scores/P on-chip to cut GM MTE2.

Source: independent prototype FA-class A3 perf whitebox 2026-05-29 (`docs/handovers/HISTORICAL_FA_CLASS_A3_PERF_WHITEBOX_2026_05_29.md`, PR #249).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-TILESIZE-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
