---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`const LocalTensor<T>` variable blocks Sort / write-through VEC API overload resolution"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Compile error no matching function for call to 'Sort' ... 1st argument ('const LocalTensor<float>') would lose const qualifier when the caller declared destinat"
confidence: single_run
original_id: EC-34
timestamp_inferred: true
tags: [sort, ascendc, ec-34]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Compile error `no matching function for call to 'Sort' ... 1st argument ('const LocalTensor<float>') would lose const qualifier` when the caller declared destination tensor locals as `const LocalTensor<float> x = ...;` (a common "value hint" pattern from earlier kernels).
- **Root cause**: Hardware `Sort` (and most write-through VEC APIs: `Cast`, `Duplicate`, `Adds`, `Muls`, `Select`, etc.) take `LocalTensor<T>&` (non-const) for their destination parameter because the API writes to them. A `const LocalTensor` cannot bind to the non-const reference.
- **Fix**: Drop `const` from any LocalTensor variable that will be passed as a destination to a write-through VEC API. `LocalTensor<T>` is a view/handle type with copy semantics, so dropping `const` does not weaken safety meaningfully.
- **Detection**: Compile error at the API call site, blaming "const qualifier" or "argument would lose const". Check the variable declaration, not the API signature.
- **Related**: general C++ overload resolution rule; applies to any write-through `LocalTensor` API, not just Sort. Documented here because it's a common newcomer gotcha when porting patterns from pure read-only kernels.
- **Evidence**: 9_TopKTopP cold-run round 2, Phase C iter 1 (2026-04-18). Dropped const → compile OK.
- **Status**: Low-severity general-C++ gotcha. Useful as preventive guidance in Phase B for kernels using hardware Sort / write-through APIs.

<!-- 迁移自 porter kb/target/ascendc/（EC-34，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
