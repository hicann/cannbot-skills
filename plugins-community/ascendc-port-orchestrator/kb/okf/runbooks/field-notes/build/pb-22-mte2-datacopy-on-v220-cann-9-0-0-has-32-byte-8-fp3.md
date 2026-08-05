---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "MTE2 DataCopy on V220 CANN 9.0.0 has 32-byte (8 fp32 element) transfer limit per destination TBuf [V220]"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "DataCopy(dst_TBuf, gm, count) only writes the first 32 bytes (8 fp32 elements) into the destination TBuf. Elements beyond index 7 are always zero, regardless of"
confidence: single_run
original_id: PB-22
timestamp_inferred: true
tags: [ascendc, pb-22]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL (silent data corruption — elements beyond index 7 are zero without error)
- **Status**: CONFIRMED 2026-05-01 op#31 IOU pp-1 probe (13 iterations, V220 CANN 9.0.0)
- **Symptom**: `DataCopy(dst_TBuf, gm, count)` only writes the first 32 bytes (8 fp32 elements) into the destination TBuf. Elements beyond index 7 are always zero, regardless of count parameter (40, 64), chunking (single call vs serialized), or TBuf position (VECCALC vs VECIN).
- **Fix**: Use `TQue<QuePosition::VECIN, depth>` for input streaming instead of `TBuf + DataCopy`. TQue's EnQue/DeQue rotation uses a different MTE2 path. Or deinterleave in pybind host-side and pass column vectors directly.
- **Evidence**: op#31 IOU a3 V220 pp-1 (2026-05-01) — 13 probe iterations confirmed across multiple TBuf positions and chunking strategies. A5 working kernel uses TQue<VECIN> successfully.
  - op#30 NMS a3 ds kw-1 (2026-05-07): Used `TQue<VECIN,1>` for streaming box coordinate and score inputs (fp32), avoiding TBuf+DataCopy 32-byte limit. 31/31 bit-exact vs Python CPU reference. Confirms TQue<VECIN> as canonical V220 input path when element count per tile exceeds 8 fp32.
  - fatrelu_mul port_a3_to_a5 kw-1 (2026-05-17, **V351 sub-block confirmation**): case 7 has lastDim=2 → d=1 → DataCopyPad blockLen = 1×4 = 4 bytes (well below 32B alignment boundary). Both input (`DataCopyPadExtParams<float>{false,0,0,0}`) and output paths handled the sub-block transfer correctly — 0.0 max_abs_diff vs A3. Validates that **V351 (A5) DataCopyPad handles unaligned blockLen natively**; the 32-byte-limit failure mode in this PB-22 entry is V220-specific (the symptom does not transfer to V351 even on extremely small d=1 tiles).
- **Cross-ref**: OL-94 (TQue vs TBuf decision), OL-124 (TBuf→MTE3 coherence), PB-9 (UB→UB DataCopy), PB-11 (multi-iteration DataCopy)

<!-- 迁移自 porter kb/target/ascendc/（PB-22，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
