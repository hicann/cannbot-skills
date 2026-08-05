---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bare `is_same<>` unresolved in a standalone bisheng/ccec kernel TU despite `using namespace AscendC` — use `std::is_same` + `#include <type_traits>`"
description: "applies_to: soc=Ascend950PR (arch35/V351); cann=9.0.0; bisheng=9.0.0; op_class=all; kernel_type=standalone_verification_pybind"
phenomenon: build_failure
signal:
  - "a standalone verification/pybind kernel (authored fresh, not #include-ing the full op_kernel/arch35 template tree) fails to compile at a is_same<A, B>::value /"
confidence: single_run
original_id: EC-83
timestamp_inferred: true
tags: [ascendc, ec-83]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (arch35/V351); cann=9.0.0; bisheng=9.0.0; op_class=all; kernel_type=standalone_verification_pybind`

- **Symptom**: a standalone verification/pybind kernel (authored fresh, not `#include`-ing the full op_kernel/arch35 template tree) fails to compile at a `is_same<A, B>::value` / `is_same_v<...>` use site with an unresolved-name error, even though the TU has `using namespace AscendC;`. The same bare `is_same` compiles fine INSIDE the arch35 regbase headers.
- **Root cause**: the arch35 regbase headers that use bare `is_same` also pull in extra AscendC internal headers that bring the symbol into unqualified scope. A standalone kernel TU that only does `using namespace AscendC` does NOT transitively include those internals, so the unqualified name never resolves in the bisheng/ccec compile context.
- **Fix**: qualify with `std::is_same` (or `std::is_same_v`) and add `#include <type_traits>` at the top of the standalone TU:
  ```cpp
  #include <type_traits>
  ...
  if constexpr (std::is_same_v<T, half>) { ... }
  ```
- **Detection**: unresolved-`is_same` compile error in a kernel `.h`/`.cpp` you authored standalone (not carried verbatim from upstream); grep the TU for bare `is_same<` / `is_same_v<` without a `std::` qualifier.
- **Evidence**: rms_norm kw-1 (2026-07-02, A5 Ascend950PR_957b, port_a3_to_a5, CANN 9.0.0): `rms_norm_kernel.h:182` bare `is_same` → `std::is_same` + `#include <type_traits>` cleared the compile-fix iter-2 build. The arch35 regbase headers used bare `is_same` fine; the independently-authored verification kernel did not.
- **Cross-ref**: OL-267 (AscendC symbols split between global and `AscendC::` scope — same "which scope resolves this symbol" family, but here the resolution is `std::`, not an AscendC scope).

<!-- 迁移自 porter kb/target/ascendc/（EC-83，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
