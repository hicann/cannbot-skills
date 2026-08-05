---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-39
timestamp_inferred: true
tags: [matmulconfig, ascendc, ec-39]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom (compile error tail)**:
  ```
  matmul_utils.h:435:26: error: no member named 'usedCoreNum' in 'MatmulConfig'
      if constexpr (MM_CFG.usedCoreNum == -1) { ... }
  matmul_utils.h:438:26: error: no member named 'M' in 'MatmulConfig'
  matmul_utils.h:486:39: error: invalid operands ('const IterateOrder' and 'int')
      if constexpr (MM_CFG.iterateOrder == -1) { ... }
  ```
  …20+ such errors before `-ferror-limit` kicks in. The compiler is trying to compare `MatmulConfig` fields to `−1` for the constexpr-vs-GM tiling decision, but `MatmulConfig` does not have those fields — they live on `MatmulApiStaticTiling` (CANN 9.0.0 `include/adv_api/matmul/tiling.h:431`).
- **Trigger pattern**: building `MatmulImpl<AT, BT, CT, BIAS, MM_CFG = CFG_NORM>` and calling `mm.Init(__gm__ TCubeTiling*, TPipe*)` (or the non-`__gm__` overload).
- **Fix**: Use `MatmulApiStaticTiling` (a struct that wraps `MatmulConfig`) as `MM_CFG`:
  ```cpp
  static constexpr MatmulApiStaticTiling MM_CFG_RUNTIME = []() {
      MatmulApiStaticTiling t{};   // every shape field defaults to −1 → use runtime tiling
      t.cfg = CFG_NORM;            // plain MatmulConfig becomes the .cfg member
      return t;
  }();
  MatmulImpl<AT, BT, CT, /*BIAS=*/CT, MM_CFG_RUNTIME> mm;
  ```
  Setting individual fields (e.g. `t.baseM = 128`) makes those constexpr; leaving them −1 reads from runtime tiling. This is the Opt2 unlock — see OL-91 step 3.
- **Rejected workaround**: switching to `Init(const TCubeTiling*, TPipe*)` (non-`__gm__`) and copying tiling field-by-field per OL-77 is NOT necessary — `MatmulImpl::Init` has a dedicated `__gm__` overload that does the slice-copy internally. The actual issue is the `MM_CFG` type, not the `Init` overload.
- **Evidence**: 1_BatchMatmul (2026-04-28) Phase C iter 1 — first level-3 cube op hit this on first build. Now amortized across 4 cube ops (op#1/#4/#5/#3 all cite OL-91 step 3 in analysis.md and built clean from iter 0).

<!-- 迁移自 porter kb/target/ascendc/（EC-39，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
