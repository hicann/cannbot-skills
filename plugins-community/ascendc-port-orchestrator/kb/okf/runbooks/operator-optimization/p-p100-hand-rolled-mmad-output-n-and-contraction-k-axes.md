---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Hand-rolled Mmad: output-n and contraction-k axes spanning >1 C0 block (>BASE_K=16) MUST be tiled in BASE_K chunks (+accumulate for k)"
description: "For hand-rolled tile-MMAD, a single LoadData+Mmad whose output-n or contraction-k axis spans more than one C0 block (>BASE_K=16, e.g. D=64 = 4 blocks) mis-packs the fractals: a too-wide n load spreads"
severity: high
confidence: single_run
original_id: P-P100
timestamp_inferred: true
tags: [platform_compat, optimization, loaddata, mmad, fixpipe, p-p100, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For hand-rolled tile-MMAD, a single `LoadData`+`Mmad` whose **output-n** or **contraction-k** axis spans more than one C0 block (>BASE_K=16, e.g. D=64 = 4 blocks) mis-packs the fractals: a too-wide **n** load spreads the output over a 2× stride (observable as alternating-zero columns in the output); a too-wide **k** in one Mmad mis-contracts. **Rule**: tile both axes in BASE_K(=C0) chunks. For **n** (output): loop `ni` over `nTiles=nAlign/BASE_K`, B-load one C0 col-block per tile, `Mmad` with `mp.n=BASE_K`, `Fixpipe` to `out[.., ni*BASE_K]`. For **k** (contraction): loop `ki` over `kTiles=kAlign/BASE_K`, slice each operand's L1 col-block at offset `ki*stride*BASE_K`, `Mmad` with `mp.k=BASE_K` and `cmatrixInitVal=(ki==0)` (init on first tile, accumulate after via the 4-arg `Mmad(c,a,b,c,mp)`). Both loops reduce to a single iteration (the prior single-shot path) when the axis ≤16, so small-dim cases are unaffected. This mirrors cv-agent FlashAttention: BMM1 tiles the head-dim contraction, BMM2 tiles the output-n. **Evidence**: lightning_indexer_grad (A3 V220, 2026-05-27) — all D=16 cases passed with single-shot loads; D=64 cases (4 C0 blocks) failed on every output. Tiling C1's k=D and C3a/C3b's n=D in BASE_K chunks (k with accumulate) fixed the D=64 cases while leaving D=16 bit-identical. **Cross-ref**: P-P99 (contraction-axis SOURCE — orthogonal; this is about axis SIZE). `applies_to: soc=Ascend910_9382 (V220/A3); cann=9.0.0; unverified_on: V351/A5`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P100，convert_patterns_to_okf.py）。confidence 未升格。 -->
