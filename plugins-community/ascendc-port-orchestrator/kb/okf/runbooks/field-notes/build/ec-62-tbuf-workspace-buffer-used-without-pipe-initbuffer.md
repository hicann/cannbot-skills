---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "TBuf workspace buffer used without `pipe_.InitBuffer()` — unallocated UB access → 507035 vector core exception on V220"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=all"
phenomenon: build_failure
signal:
  - "Kernel compiles and links, crashes 507035 (subErrType:4, ADDR_MISALIGN) on every launch."
confidence: single_run
original_id: EC-62
timestamp_inferred: true
tags: [507035, ascendc, ec-62]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=all`

- **Severity**: CRITICAL (kernel crashes at runtime with 507035, compiles cleanly)
- **Status**: CONFIRMED 2026-05-23 1_RotaryMul a3-ds kw-1
- **Symptom**: Kernel compiles and links, crashes 507035 (subErrType:4, ADDR_MISALIGN) on every launch.
- **Root cause**: Worker init'd TQue buffers but forgot TBuf workspace InitBuffer. TBuf::Get<T>(n) on unallocated UB → hardware fault.
- **Fix**: Add `pipe_.InitBuffer(tbuf_name, size_bytes)` for every TBuf member. All TBufs must be explicitly initialized.
- **Detection**: grep `TBuf<.*> \w+_;` in kernel.h. For each, verify `pipe_.InitBuffer(name, ...)` exists. Now mandated by kw_brief Phase C self-audit.
- **Evidence**: 1_RotaryMul a3-ds kw-1 (2026-05-23, V220): 5 TBuf workspace buffers had no InitBuffer. Adding them fixed fp16+fp32 on A3 NPU0.
- **Cross-ref**: EC-60 (blockDim=0), EC-61 (scalar-pipe acc), PB-22 (DataCopy 32B limit). All four are "compiles, crashes" V220 classes kw_brief must preempt.

<!-- 迁移自 porter kb/target/ascendc/（EC-62，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
