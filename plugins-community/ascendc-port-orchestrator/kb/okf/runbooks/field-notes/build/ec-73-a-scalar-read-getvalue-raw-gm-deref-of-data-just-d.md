---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A scalar read (`GetValue` / raw `__gm__` deref) of data just `DataCopyPad`'d into UB needs `HardEvent::MTE2_S` — `MTE2_V` alone leaves the scalar pipe racing the DMA → garbage indices → OOB GM access → silent hang"
description: "<!-- applies_to_backend: all -->"
phenomenon: build_failure
signal:
  - "no compile error. At runtime the kernel reads plausible-looking but WRONG scalar values (e.g. sampling/gather indices) right after a DataCopyPad-to-UB, computes"
confidence: single_run
original_id: EC-73
timestamp_inferred: true
tags: [getvalue, __gm__, datacopypad, mte2_v, ascendc, ec-73]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — the MTE2→S dependency is general AscendC pipe semantics, but not retested here)`

- **Error pattern**: no compile error. At runtime the kernel reads plausible-looking but WRONG scalar values (e.g. sampling/gather indices) right after a `DataCopyPad`-to-UB, computes an out-of-range GM offset from them, and the OOB read wedges the device (silent hang, no fault). Confounds easily with a wedged-NPU artifact (OL-189) — but the MTE2_S fix is independently required.
- **Root cause**: `DataCopyPad` moves data on the **MTE2** pipe. A subsequent `GetValue`/`SetValue`/raw pointer read runs on the **scalar (S)** pipe. Only `HardEvent::MTE2_S` orders the DMA-completion against the scalar read. Syncing `MTE2_V` (the common reflex for "DMA then VEC compute") guards VEC consumers but NOT the scalar pipe — the scalar read still races the in-flight DMA and observes stale/garbage UB.
- **Fix**:
  ```cpp
  // BEFORE (scalar read races the DMA — garbage index → OOB → hang):
  DataCopyPad(idxLocal, idxGm, copyParams, padParams);
  SyncFunc<HardEvent::MTE2_V>();           // ← guards VEC, NOT the scalar read below
  int64_t k = idxLocal.GetValue(i);        // races MTE2

  // AFTER (scalar read ordered after the DMA):
  DataCopyPad(idxLocal, idxGm, copyParams, padParams);
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  SetFlag<HardEvent::MTE2_S>(ev);
  WaitFlag<HardEvent::MTE2_S>(ev);
  int64_t k = idxLocal.GetValue(i);        // safe
  ```
- **Evidence**: MultiScaleDeformableAttnFunction port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500): scalar reads of `DataCopyPad`'d sampling indices under MTE2_V-only sync produced garbage indices → OOB GM access → hang; adding the `MTE2_S` Set/Wait fixed it.
- **Other instances (predicted)**: any kernel that DMA-loads index/offset/shape metadata to UB and then reads it scalar-side to drive addressing — gather/scatter, deformable/sampling ops, dynamic-shape tiling readers, variable-length list ops. General rule: a scalar consumer of `DataCopyPad`/`DataCopy`-to-UB data must sync `MTE2_S`, not `MTE2_V`.
- **Related**: EC-13 (`SyncFunc` API form + the `MTE2_S (GM→scalar)` event list), PB-43-class scalar-pipe sync (`SetFlag`/`WaitFlag` vs unsupported `PipeBarrier<PIPE_S>`), OL-189 (wedged-NPU can mask this — verify the fix on a fresh card).

<!-- 迁移自 porter kb/target/ascendc/（EC-73，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
