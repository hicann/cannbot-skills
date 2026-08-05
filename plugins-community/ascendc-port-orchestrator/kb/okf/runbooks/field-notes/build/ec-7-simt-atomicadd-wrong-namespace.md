---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`Simt::atomicAdd` — wrong namespace"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-7
timestamp_inferred: true
tags: [atomicadd, ascendc, ec-7]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: no member named 'atomicAdd' in namespace 'AscendC::Simt'
  ```
  or:
  ```
  error: call to 'atomicAdd' is ambiguous
  ```
  (when both `Simt::atomicAdd` and global `atomicAdd` are attempted)
- **Root cause**: `atomicAdd` on AscendC is a **global built-in function**, not a member of the `AscendC::Simt` namespace. This differs from other Simt APIs like `Simt::VF_CALL`, `Simt::Dim3`, `Simt::WarpReduceAddSync` which are namespaced.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  Simt::atomicAdd(base + offset, value);       // ❌ not in Simt namespace
  AscendC::Simt::atomicAdd(base + offset, value);  // ❌ same error

  // AFTER (compiles):
  atomicAdd(base + offset, value);             // ✅ global built-in, no namespace
  ```
- **Supported types**: `float`, `half`, `bfloat16_t`, `int32_t` — all use the same unqualified `atomicAdd`.

<!-- 迁移自 porter kb/target/ascendc/（EC-7，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
