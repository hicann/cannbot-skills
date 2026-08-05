---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Top-K buffer `effective_kept` vs `buffer_len` — always compute effective_kept separately"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel with oversized top-K buffer (e.g. TOPK_CAP > actual row's count of v >= threshold) produces wrong cutsum cumulation, cutoff decision lands far from thres"
confidence: single_run
original_id: EC-32
timestamp_inferred: true
tags: [effective_kept, buffer_len, ascendc, ec-32]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Kernel with oversized top-K buffer (e.g. TOPK_CAP > actual row's count of v >= threshold) produces wrong cutsum cumulation, cutoff decision lands far from threshold, precision all-fail with systemic offset.
- **Root cause**: Worker walks cumsum / reduce over the full TOPK_CAP-sized buffer assuming all positions contain valid data. When the row has fewer than TOPK_CAP values ≥ threshold (e.g. row with small N, or row where actual kept count < buffer cap), the walk processes uninitialized / padding positions, inflating the cum and preventing cutoff from triggering.
- **Fix**: After any top-K merge / compaction step, compute `effective_kept` separately:
  ```cpp
  int32_t effective_kept = 0;
  for (int32_t i = 0; i < TOPK_CAP; i++) {
    if (top_val[i] >= threshold) effective_kept++;
  }
  ```
  Then walk `[0 .. effective_kept - 1]` (or the equivalent ASC range), NOT `[0 .. TOPK_CAP - 1]`. Padding positions between effective_kept and TOPK_CAP must be held at a sentinel (e.g. -inf) that guarantees they would be rejected if processed — belt-and-suspenders.
- **Detection**: Small-N cases fail systemically while large-N cases (where effective_kept ≈ TOPK_CAP) pass. Or: cumsum values reach 1.0 far from the expected cutoff rank.
- **Evidence**: 9_TopKTopP cold-run Phase D iter 2 (2026-04-18). Worker walked from `topk_len - 1` down instead of `effective_kept - 1`; for rows with < TOPK_CAP kept values, iterated over non-kept pos → cum inflated → cutoff never set. Fix: explicit `effective_kept` scan, walk that range.
- **Related**: P-P59 (tied-threshold buffer truncation) — P-P59 assumes `effective_kept ≤ TOPK_CAP`; this EC is about the distinct implementation concern that `effective_kept` may be `< TOPK_CAP` for small rows.

<!-- 迁移自 porter kb/target/ascendc/（EC-32，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
