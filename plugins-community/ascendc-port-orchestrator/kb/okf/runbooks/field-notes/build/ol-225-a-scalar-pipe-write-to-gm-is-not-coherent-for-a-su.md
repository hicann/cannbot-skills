---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A scalar-pipe write to GM is not coherent for a subsequent cube/MTE2 read — stage in UB then DataCopy (MTE3) to flush"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all (cube + scalar-staging)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all (cube + scalar-staging)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-225
timestamp_inferred: true
tags: [__gm__, ascendc, ol-225]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all (cube + scalar-staging)`
`verified_on: soc=Ascend950PR; cann=9.1.T500 (GDN chunk_gated_delta_rule catlass, 2026-06-16)`

### Principle

When the **scalar pipe** writes a value directly to a GM address (`GlobalTensor::SetValue` / scalar store into a `__gm__` pointer), that write is **not guaranteed visible** to a subsequent **cube (MTE2/L1 load) or other-core read** of the same GM — even after the local block has "finished" producing it. The scalar store goes through a path that is not fenced against the cube's GM read by ordinary pipe barriers. The cube then loads stale/garbage for that operand.

The coherent path is: build the data in **UB** (on-chip; scalar `SetValue` into a `LocalTensor` is fine on-chip), then **`DataCopy` UB→GM (MTE3)** with an explicit `V_MTE3` / `S_MTE3` → `MTE3_MTE2` fence. The MTE3 store is the GM-visible flush the cube's MTE2 load will see.

### Decision rule

- Producing a GM tensor that a cube GEMM (or another core) will read, where the producer is scalar-pipe element-by-element (e.g. a small scalar transpose, a hand-built index/coefficient table): **never `SetValue` straight to GM**. Stage in a UB `LocalTensor`, then `DataCopy` to GM.
- Pure scalar→GM→scalar (same core, scalar read) is a different, narrower question; this rule is specifically about a downstream **cube/MTE2** or **cross-core** consumer.

### Concrete anchor

```cpp
// S0 transpose: build in UB (on-chip scalar SetValue), then DataCopy to GM (MTE3 flush).
// Scalar SetValue directly to GM is NOT flushed for the cube to read -> must go via UB+DataCopy.
for (uint32_t r = 0; r < Dk; ++r)
  for (uint32_t col = 0; col < Dv; ++col)
    s0ub.SetValue((uint64_t)r*Dv + col, isub.GetValue((uint64_t)col*Dk + r)); // build in UB
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
CopyUbBf2Gm(gS0, s0ub, Dk * Dv);                       // MTE3 flush -> now cube-visible
```

### Evidence

GDN `chunk_gated_delta_rule` on catlass (A5/V351, CANN 9.1.T500, 2026-06-16): the initial-state transpose `S0[Dk,Dv] = init[Dv,Dk]ᵀ` was built by scalar `SetValue` directly into the S0 GM workspace. The cube `RunGemm` that consumes S0 (`qs@S0`, `k_cd@S0`) read stale data → wrong recurrence output. Routing the transpose through a UB `LocalTensor` + `DataCopy` (MTE3) made the cube read the correct S0 (`gdn.cpp:154-168`).

### Other instances (predicted)

Any hand-built small operand that the scalar pipe produces and a cube/MTE2 stage consumes: scalar transposes, per-head coefficient/scale tables, gathered index arrays staged for a fractal load, host-computed-then-scalar-stored tiling scratch. Distinct from OL-128 (V220 VEC-write→TBuf→MTE3 staleness, fixed via `TQue<VECOUT>`): that is the VEC pipe on V220; this is the scalar pipe → cube consumer on V351.

### Cross-references

- OL-128 (V220 TBuf→MTE3 stale data — VEC pipe, different SoC/consumer)
- OL-94 (TQue vs TBuf sync decision table)
- OL-224 (manual-cube tail/transpose correctness — same GDN cube)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-225（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
