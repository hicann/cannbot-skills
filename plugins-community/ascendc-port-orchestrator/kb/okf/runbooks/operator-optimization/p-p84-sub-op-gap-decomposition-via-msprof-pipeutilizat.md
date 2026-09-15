---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Sub-op gap decomposition via msprof PipeUtilization anchor + analytical scalar-op counting (fused-optimizer methodology when standalone refs are unreliable)"
description: "For multi-stage fused ops where running per-sub-op standalone CANN refs (npu_top_k, npu_softmax, npu_cumsum, etc.) is unreliable due to EC-33-class instability, NPU lane contention, or shared-resource"
severity: high
confidence: single_run
original_id: P-P84
timestamp_inferred: true
tags: [platform_compat, optimization, npu_top_k, npu_softmax, npu_cumsum, aiv_scl_ratio, aiv_vec_ratio, p-p84, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

### Trigger
For multi-stage fused ops where running per-sub-op standalone CANN refs (`npu_top_k`, `npu_softmax`, `npu_cumsum`, etc.) is unreliable due to EC-33-class instability, NPU lane contention, or shared-resource concerns, decompose per-row dur into per-sub-op contributions analytically. **Procedure**: (1) take ONE high-quality msprof PipeUtilization profile of the dominant case — record `aiv_scl_ratio`, `aiv_vec_ratio`, `aiv_mte2_ratio`, `aiv_mte3_ratio`, total dur, rows-per-AIV; (2) per-row dur = total_dur / rows_per_aiv; per-pipe budget = pipe_ratio × per_row_dur; (3) for each sub-op, count scalar ops (GetValue/SetValue/Set+Get pairs) by code inspection — calibrate per-op latency from observed dur (typically 3–6 ns/op on Ascend950PR scalar pipe); (4) per-sub-op scalar contribution = scalar_op_count × calibrated_latency; vector contribution via cycle accounting (e.g. ReduceMax over N ≈ N/64 + log₂(64)); (5) sub-op classification: BOTTLENECK if contribution > 50% of dominant pipe budget; UNCOMPARABLE if no standalone ref exists for the sub-op's full role (e.g. row-max into shared accumulator across phases); NECESSARY if structurally required (e.g. emit phase doing full-row scatter). **When applicable**: fused op where `npu_<X>` standalone ref exists but cannot be measured cleanly; per-AIV serial work dominates (single-AIV-per-row); bottleneck dominated by scalar pipe (analytical counting gives high-confidence cost). **Limitations**: ±2× per-cell precision (qualitatively decisive for finding dominant cell, not for ranking near-equal cells); doesn't help when standalone refs ARE reliable — direct measurement is better. **Anti-pattern complement**: When the localized bottleneck is scalar-pipe-bound (`aiv_scl_ratio > 0.6`), buffer-aliasing optimizations have ~zero ROI even if the aliasing is correct — recovered UB doesn't help unless it unlocks a tile/chunk-size increase that itself unlocks vector work. Op#9 fo-1 found 19.4 KB recoverable from idle multi-AIV buffers + 8.8 KB from sortedVal/Idx ↔ mergeTmp aliasing — ~28 KB total. CHUNK_LEN expansion 2048→4096 would consume 16 KB more (fits!), but archive iter 2 already tried this and reverted for **precision** (Sort intrinsic drift on >2048 elements), independent of UB. So even with the alias fix, the obvious downstream lever isn't safe. Validated op#9 9_TopKTopP fo-1 (2026-05-03 Ascend950PR_9579): localized 81% of dur to Phase 1 inner loop (32 chunks × 1088 merge_cap × 6 scalar ops × ~5ns ≈ 740 us/row of 1264 us/row total). Saved 4× standalone-CANN benchmark calls that would have hit EC-33 truncation. Verdict: structural ceiling 0.385× confirmed via 3rd independent diagnostic angle. **Cross-ref**: MSPROF_AGENT_GUIDE.md (PipeUtilization extraction), EC-33 (sustained-call instability rationale for analytical decomposition), OL-82 (scalar_ratio thresholds).

<!-- 迁移自 porter patterns/PATTERN_INDEX.md P-P84 索引行（该条目无 domains 正文，索引行即全部内容；B2 手工补卡，format 对齐 convert_patterns_to_okf.py 输出）。confidence/severity 未升格。 -->
