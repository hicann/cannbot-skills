---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`block_num` / `block_index` macro collision in parameter names"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-12
timestamp_inferred: true
tags: [block_num, block_index, ascendc, ec-12]
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
  error: cannot initialize a parameter of type 'int64_t (*)(void)' with an rvalue of type 'int64_t'
  note: expanded from macro 'block_num'
  #define block_num get_block_num()
  ```
- **Root cause**: CANN defines `block_num` as a macro expanding to `get_block_num()` (a function). When used as a function parameter name, `int64_t block_num` becomes `int64_t get_block_num()` -- a function declaration, not a parameter. Similarly, `block_index` may collide with other CANN macros.
- **Fix**: Rename parameters to avoid CANN macro names:
  ```cpp
  // BEFORE (fails):
  void Init(GM_ADDR x, int64_t block_index, int64_t block_num) { ... }

  // AFTER (compiles):
  void Init(GM_ADDR x, int64_t blk_idx, int64_t blk_cnt) { ... }
  ```
- **CANN macros to avoid as identifiers**: `block_num`, `block_idx`, and any other identifier in `__clang_cce_aicore_builtin_vars.h`.
- **Related**: OL-14 (namespace ambiguity)

<!-- 迁移自 porter kb/target/ascendc/（EC-12，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
