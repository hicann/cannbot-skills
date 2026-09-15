---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Manual-Mmad B-operand contraction axis = the L1 tile's C0/inner dim; A and B must source the contraction from the SAME axis"
description: "For hand-rolled tile-MMAD (Mmad + LoadData-based L1→L0 loads, NOT MatmulImpl<>): the 2D-params LoadNzL1ToZnL0B (LoadData2DParams, ifTranspose=false) makes the L0B contraction (k) axis = the L1 tile's"
severity: high
confidence: single_run
original_id: P-P99
timestamp_inferred: true
tags: [platform_compat, optimization, mmad, loaddata, loadnzl1toznl0b, loaddata2dparams, dvalue, p-p99, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For hand-rolled tile-MMAD (`Mmad` + `LoadData`-based L1→L0 loads, NOT `MatmulImpl<>`): the 2D-params `LoadNzL1ToZnL0B` (`LoadData2DParams`, `ifTranspose=false`) makes the L0B **contraction (k) axis = the L1 tile's C0/inner (column) dim** — i.e. the `dValue` used in the GM→L1 `Nd2Nz` load. The 2D-params `LoadNzL1ToZzL0A` does the same for A. A and B must therefore source the contraction from the **same physical axis** — either BOTH from C0 (plain loads), or BOTH from the ROW dim (matched transposed loads). A **mixed** pairing (A contracts over its C0 while B contracts over its ROW, or vice-versa) mis-aligns the L0A/L0B fractals → wrong/zero output even though the logical `mp.k` value matches. **Decision rule**: identify each operand's contraction axis relative to its L1 layout. (a) contraction == C0 dim → plain `LoadNz...` (no transpose); (b) contraction == ROW dim → either a transposed load (`LoadDataWithTranspose` / `LoadData2DParams{ifTranspose=true}`, V220-verified for the K^T side) OR the 3D `LoadData3DParamsV2` form (`mExtension`=contraction, `channelSize`/`kExtension`=output-n) which contracts over the L1 ROW axis. Whatever form A uses, B must use the **matching** form so both source k consistently. **Anti-pattern**: pairing a non-transposed A (k from C0) with a transposed/3D B (k from ROW) — the empirical failure was correct-magnitude-but-wrong dq. **Evidence**: lightning_indexer_grad (A3 V220, 2026-05-27) — dgk = dscores^T @ Q (contract over N1): A=dscores^T (trans, k from ROW) + B=Q (trans, k from ROW) → correct (dk ~0.1%). dq = dscores @ gk (contract over topK): A=dscores (plain, k from C0) + B=gk (trans, k from ROW) → wrong; matching both to the 3D form (the cv-agent FA BMM2 P@V mechanism: A=P k-from-C0, B=V k-from-ROW, both 3D) was required. **Cross-ref**: P-P100 (multi-C0-block tiling — orthogonal axis-size concern), CAND-FA1 (`LoadData2DParams{ifTranspose=true}` for K^T verified form). `applies_to: soc=Ascend910_9382 (V220/A3); cann=9.0.0; unverified_on: V351/A5`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P99，convert_patterns_to_okf.py）。confidence 未升格。 -->
