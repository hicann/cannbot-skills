---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`LocalMemBar<MemType::UB>` replaces `SetFlag<MTE2_V>`+`WaitFlag` in A5 L2 MicroAPI [V351, microapi-sync]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=l2-microapi; phase=kernel-author verified_on: soc=Ascend950PR; cann=9.0.0 source: PR 103 l2-register-based-guide.md §250 Pattern: In A5 L"
confidence: single_run
original_id: P-P95
timestamp_inferred: true
tags: [patterns-index, optimization, waitflag, __vec_scope__, localmembar, memtype, l0a, p-p95, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=l2-microapi; phase=kernel-author`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l2-register-based-guide.md §250`

**Pattern**: In A5 L2 (Register-based MicroAPI) kernels, replace the A3 explicit pipe-event sync pair `SetFlag<HardEvent::MTE2_V>(EVENT_ID) + WaitFlag<HardEvent::MTE2_V>(EVENT_ID)` with the simpler `LocalMemBar<MemType::UB>()` memory barrier. The new barrier is type-targeted (UB), no event-id management, less error-prone.

**Concrete**:

```cpp
// ❌ A3 style — explicit pipe-event sync, error-prone event-id juggling
DataCopy(srcLocal, srcGm, count);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Mul(dstLocal, src1Local, src2Local, count);

// ✅ A5 L2 style — type-targeted barrier
DataCopy<LoadDist::DIST_UNPACK_B16>(reg0, src_addr);
LocalMemBar<MemType::UB>();         // wait for UB-affecting prior ops
MicroAPI::Mul(reg1, reg0, reg0, maskReg);
```

**Where**: L2 inner loops between data-load and compute, between compute and store. Inside `__VEC_SCOPE__` blocks, the compiler tracks dependencies per register so most intra-scope sync is elided; explicit `LocalMemBar` is needed at scope boundaries.

**`MemType` options**:
- `MemType::UB` — barrier against all UB-affecting prior operations
- (additional types in CANN headers — `L1`, `L0A`/`L0B`/`L0C` for cube-side L2 ports)

**When NOT to use**:
- L1 mechanical ports (Memory-based) — keep `SetFlag`/`WaitFlag` for compatibility with existing A3 sync chains
- L3 SIMT kernels — they live outside the UB-pipe model, sync is implicit per-thread

**Detection signature**:

```bash
# In arch35/ kernels (L2), find lingering A3-style sync pairs
grep -nE "SetFlag<HardEvent::|WaitFlag<HardEvent::" arch35/*.h
# Each pair is a substitution candidate.
```

**Why simpler is correct**:
- A3 `SetFlag`/`WaitFlag` requires manually picking EVENT_ID (0-7), threading it through compute, freeing it later. Per-instance bugs accumulate.
- A5 `LocalMemBar<MemType::UB>` has no event-id parameter — compiler tracks dependencies. Less code, fewer bugs.

**Evidence**: PR 103 l2-guide §250 row "内存屏障" entry.

**Other instances (predicted)**: every L2-classified op in cohort 2.

**Cross-reference**:
- OL-152 (Memory↔Register API mapping — this is one row)
- P-P96 (`__VEC_SCOPE__` inner loop — uses `LocalMemBar` at boundaries)

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P95，convert_patterns_to_okf.py）。confidence 未升格。 -->
