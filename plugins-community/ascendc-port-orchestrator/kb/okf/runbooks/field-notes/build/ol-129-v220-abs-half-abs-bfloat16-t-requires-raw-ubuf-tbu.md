---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 `Abs<half>` / `Abs<bfloat16_t>` requires raw `__ubuf__*` TBuf pointer — use Cast→fp32 TBuf→Abs→Cast back for fp16/bf16 abs [V220]"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "paradigm: ascendc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-129
timestamp_inferred: true
tags: [ascendc, ol-129]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

**Principle**: On V220 (dav_c220), the VEC `Abs<half>` and `Abs<bfloat16_t>` APIs do not accept TQue-dequeued `LocalTensor<half>` — they require a raw `__ubuf__ half*` (bare TBuf) as the first parameter.

**Fix**: For fp16/bf16 abs, promote to fp32 via a TBuf, compute Abs in-place on fp32, then cast back to native type with CAST_RINT. This is bit-exact because fp32 can represent all fp16/bf16 values exactly.

**Evidence**: op#4 Abs DS kw-1 (2026-05-02): 1 build iter to diagnose. Cast→fp32→Abs→Cast workaround produces bit-exact results vs CPU truth (50/50 PASS, MERE=0/MARE=0).

**Cross-ref**: OL-127 (CANN API surface gaps V220), OL-81 (CAST_RINT for fp16/bf16 output).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-129（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
