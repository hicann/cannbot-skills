---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SOC_VERSION Ascend910B2 causes 507035 on Ascend950PR hardware"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "ALL kernels crash with 507035 (vector core exception, error 259 = illegal instruction) at PC offset 0x80. Build succeeds but runtime crashes every time."
confidence: single_run
original_id: PB-12
timestamp_inferred: true
tags: [507035, ascendc, pb-12]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with three CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate, 22_HybridAttentionMaskPreparation). Do not downgrade.
- **Status**: CONFIRMED (2026-04-16)
- **Symptom**: ALL kernels crash with 507035 (vector core exception, error 259 = illegal instruction) at PC offset 0x80. Build succeeds but runtime crashes every time.
- **Root cause**: `build_ascendc.py` defaults to `-v Ascend910B2` if no SOC_VERSION specified. Ascend910B2 binary contains instructions not supported on Ascend950PR AIV cores.
- **Fix**: Always pass `-v Ascend950PR_9589` when building on A5 hardware.
- **Impact**: This explains ALL previous unexplained 507035 crashes. Workers MUST always specify SOC_VERSION.
- **Evidence**: 14_AdaptiveInstanceNormalization2DBackward (4th attempt): all kernel variants crashed identically until SOC_VERSION corrected.

<!-- 迁移自 porter kb/target/ascendc/（PB-12，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
