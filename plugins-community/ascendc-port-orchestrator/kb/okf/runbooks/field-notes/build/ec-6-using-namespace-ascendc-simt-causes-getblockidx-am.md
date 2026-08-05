---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-6
timestamp_inferred: true
tags: [getblockidx, getblocknum, ascendc, ec-6]
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
  error: call to 'GetBlockIdx' is ambiguous
  note: candidate function: int32_t AscendC::Simt::GetBlockIdx()
  note: candidate function: int64_t GetBlockIdx()
  ```
  (typically 20+ errors across a file since every `GetBlockIdx`/`GetBlockNum` call is ambiguous)
- **Root cause**: CANN defines TWO `GetBlockIdx()` functions — `AscendC::Simt::GetBlockIdx()` returning `int32_t` and a basic API `GetBlockIdx()` returning `int64_t`. Adding `using namespace AscendC::Simt;` pulls the Simt version into the same scope as the basic API version, making every unqualified call ambiguous.
- **Fix**:
  ```cpp
  // BEFORE (ambiguous):
  using namespace AscendC;
  using namespace AscendC::Simt;   // ❌ pulls in Simt::GetBlockIdx

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // ambiguous: Simt::GetBlockIdx vs basic_api
  }

  // AFTER (unambiguous):
  using namespace AscendC;         // ✅ only basic API GetBlockIdx (int64_t)
  // No "using namespace AscendC::Simt;" — dispatchers use qualified Simt::VF_CALL

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // resolves to basic_api int64_t version
    Simt::VF_CALL<kernel_vf<T>>(   // Simt:: qualified prefix for VF_CALL
        Simt::Dim3{THREAD_NUM}, ...);
  }
  ```
- **Note**: Kernel VF functions themselves don't call `GetBlockIdx` — they receive `block_index` as a parameter from the dispatcher. Only dispatchers need `GetBlockIdx`/`GetBlockNum`.
- **Related**: OL-14 (OPERATIONAL_KNOWLEDGE.md)

<!-- 迁移自 porter kb/target/ascendc/（EC-6，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
