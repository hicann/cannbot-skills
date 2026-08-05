---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "UB aliasing cross-row V→MTE2 hazard — silent data corruption"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "When a fused op's ProcessRow() aliases two UB buffers (P-P65 pattern), if the alias target is (a) VEC-written near the end of ProcessRow AND (b) MTE2-written ne"
confidence: single_run
original_id: PB-17
timestamp_inferred: true
tags: [262144, ascendc, pb-17]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: When a fused op's ProcessRow() aliases two UB buffers (P-P65 pattern), if the alias target is (a) VEC-written near the end of ProcessRow AND (b) MTE2-written near the start of the next ProcessRow, with no explicit V→MTE2 sync, MTE2 overlaps the still-in-flight VEC writes → data corruption, precision FAIL. The kernel itself compiles and a single case may PASS — only batched runs surface the bug.
- **Affected**: Ascend950PR / CANN 9.0.0 / bisheng 2026-03-21 (expected to apply to all CANN versions; not probe-confirmed).
- **Root cause**: The AIV VEC pipe and MTE2 pipe run in parallel; AscendC queue's EnQue/DeQue only syncs MTE2→VEC, not V→MTE2. When the aliased physical slot is accessed by both pipes on different rows, the hardware provides no automatic barrier.
- **Workaround**: Insert `SetFlag<HardEvent::V_MTE2>` at the end of ProcessRow(), and `WaitFlag<HardEvent::V_MTE2>` at the start of the next row. Cost ~100-200 ns/row, which typically comes close to cancelling the savings from aliasing. If the sync cost offsets the benefit, the alias isn't worth it.
- **Status**: OPEN (architectural constraint, not a CANN bug).
- **Evidence**:
  - op#11 aog-fused-optimizer pilot Iter1 C5 attempt (2026-04-21): aliasing `fp16Buf_ ← tmpBuf_` → precision FAIL (6430/262144 int8 mismatch, max_abs_diff=147), REVERT; static dataflow audit identified a V→MTE2 hazard.
  - `workspace/dequantswigluquant/fused_analysis.md` §Iter 1 preliminary + §Handoff
- **Detection heuristic (for aog-fused-optimizer agents)**:
  - For each alias candidate, check the alias target's last write in the current row (VEC) and its first write in the next row (MTE2).
  - If there is no sync event between those two writes, this alias is a PB-17 risk.
  - Either add a sync (evaluate the net benefit) or drop this alias.
- **Cross-reference**: PB-47 (the cross-ITERATION variant — same V→MTE2 hazard class, but a per-tile buffer reloaded each chunk-loop iteration rather than two aliased buffers within `ProcessRow`; signature "every tile wrong except the last").

<!-- 迁移自 porter kb/target/ascendc/（PB-17，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
