---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Automatic pipeline overlap with TQue (replacing PipeBarrier + manual Ping-Pong)"
description: "Scenario: SIMD kernel loop processing multiple data chunks (e.g. top_k experts); each iteration needs DataCopy(MTE2) read + Muls/Add(VEC) compute. Anti-pattern 1 (PipeBarrier<PIPE_ALL> serialization —"
confidence: single_run
original_id: P-P28
timestamp_inferred: true
tags: [memory_access, optimization, p-p28, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Scenario**: SIMD kernel loop processing multiple data chunks (e.g. top_k experts); each iteration needs DataCopy(MTE2) read + Muls/Add(VEC) compute.

**Anti-pattern 1** (PipeBarrier<PIPE_ALL> serialization — old approach, deprecated):
```cpp
// FORBIDDEN: PipeBarrier<PIPE_ALL> synchronizes all pipes; MTE2/VEC cannot parallelize
for (int k = 0; k < top_k; k++) {
  DataCopy(buf, inGm_[expert[k] * hdim], hdim);  // MTE2
  PipeBarrier<PIPE_ALL>();                         // wait all pipes → serial!
  Muls(tmp, buf, w[k], hdim);                     // VEC
  Add(accum, accum, tmp, hdim);                    // VEC
  PipeBarrier<PIPE_ALL>();                         // wait all pipes again → serial!
}
```

**Correct pattern** (Ping-Pong pipelining):
```cpp
// Two independent TBufs: ping and pong
DataCopy(ping, inGm_[expert[0] * hdim], hdim);   // prolog: load the first
PipeBarrier<PIPE_ALL>();

for (int k = 0; k < top_k - 1; k++) {
  int cur = k % 2, nxt = 1 - cur;
  // MTE2: prefetch next into the other buffer (parallel with VEC)
  DataCopy(nxt==0 ? ping : pong, inGm_[expert[k+1] * hdim], hdim);
  // VEC: compute current buffer (parallel with MTE2)
  Cast(expertF, cur==0 ? ping : pong, RoundMode::CAST_NONE, hdim);
  Muls(tmp, expertF, w[k], hdim);
  Add(accum, accum, tmp, hdim);
  PipeBarrier<PIPE_ALL>();  // sync: both buffers must be ready for the next iter
}
// epilog: process the last one
```

**Anti-pattern 2** (manual Ping-Pong + PipeBarrier — old approach E10-3):
```cpp
// WARNING: better than anti-pattern 1, but PipeBarrier<PIPE_ALL> still syncs all pipes
DataCopy(ping, inGm_[expert[0]], hdim);
PipeBarrier<PIPE_ALL>();
for (int k = 0; k < top_k - 1; k++) {
  DataCopy(pong, inGm_[expert[k+1]], hdim);  // MTE2: load next
  Muls(tmp, ping, w[k], hdim);               // VEC: compute current
  Add(accum, accum, tmp, hdim);
  PipeBarrier<PIPE_ALL>();                    // wait all pipes — including those that don't need it
  swap(ping, pong);
}
```

**Correct pattern** (TQue<VECIN,4> automatic pipeline overlap — E13):
```cpp
// OK: TQue EnQue/DeQue only syncs MTE2→VEC, does not block other pipes
// depth=4 lets MTE2 prefetch ahead; VEC never waits
pipe_.InitBuffer(xQueue_, 4, bufBytes);  // depth 4
pipe_.InitBuffer(yQueue_, 2, bufBytes);  // output depth 2

LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
Duplicate(yLocal, 0.0f, hdim);
for (int k = 0; k < top_k; k++) {
  LocalTensor<T> x = xQueue_.AllocTensor<T>();
  DataCopy(x, inGm_[expert[k] * hdim], hdim);   // MTE2
  xQueue_.EnQue(x);                              // auto-enqueue when MTE2 completes
  LocalTensor<T> xComp = xQueue_.DeQue<T>();     // wait for MTE2 (this pipe only)
  Muls(xComp, xComp, w[k], hdim);               // VEC (parallel with next iter's MTE2)
  Add(yLocal, yLocal, xComp, hdim);
  xQueue_.FreeTensor(xComp);
}
yQueue_.EnQue(yLocal);
LocalTensor<T> yOut = yQueue_.DeQue<T>();
DataCopy(outGm_[dst], yOut, hdim);               // MTE3
yQueue_.FreeTensor(yOut);
```

**Key difference**: TQue's EnQue/DeQue only syncs MTE2→VEC. PipeBarrier<PIPE_ALL> syncs all MTE2+VEC+MTE3+Scalar pipes. With depth=4, MTE2 can preload 3 buffers ahead and VEC never idles.

**Measured effect (SG forward, 2026-04-01)**:
- PipeBarrier → TQue: **1.6-2.3x** speedup (6 cases)
- OL-4 TQue bug resolved (CANN 9.0.0; backward has long validated TQue)

**Applicability**:
- SIMD kernel (has DataCopy + VEC compute loop)
- Loop iterations >= 2
- **Strongly prefer the TQue approach**; only fall back to PipeBarrier when TQue has a known bug
- **accum must also be managed by TQue** — E14 empirically showed TBuf accum + TQue input precision FAIL (max_diff=0.76). Root cause: TBuf has **no automatic sync** (confirmed by official docs); between VEC writing accum(TBuf) and MTE2 writing input(TQue) there is no sync → UB bus contention. **Fix**: move accum into TQue<VECOUT> (Pattern B in ASCENDC_LANGUAGE_REFERENCE.md), consistent with the forward yQueue_ pattern
- Detailed sync semantics: `src/skills/references/target/ascendc/LANGUAGE_REFERENCE.md` §2-3

**Combine with P-P22 (Persistent)**: TQue overlaps MTE2/VEC in the inner loop; Persistent reduces scheduling overhead in the outer loop. They are orthogonal and stackable.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P28，convert_patterns_to_okf.py）。confidence 未升格。 -->
