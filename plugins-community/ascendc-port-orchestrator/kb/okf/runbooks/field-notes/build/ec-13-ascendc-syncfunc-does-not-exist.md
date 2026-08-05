---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`AscendC::SyncFunc<>` does not exist"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-13
timestamp_inferred: true
tags: [setflag, waitflag, ascendc, ec-13]
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
  error: no member named 'SyncFunc' in namespace 'AscendC'
  ```
- **Root cause**: There is no `AscendC::SyncFunc` API. The generated code (from templates or LLM) may invent this API for pipe synchronization. The correct API uses `SetFlag`/`WaitFlag` with event IDs fetched from `GetTPipePtr()->FetchEventID()`.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

  // AFTER (compiles):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
  ```
- **Common sync events**: MTE2_S (GM→scalar), S_MTE3 (scalar→GM write), V_S (VEC→scalar), S_V (scalar→VEC)
- **Evidence**: Cumsum V1 build failure (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-13，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
