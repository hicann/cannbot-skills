---
applies_to: soc=Ascend950PR
reason: Warp-level shuffle (`Simt::WarpShflSync`, `Simt::WarpReduceAddSync`) only exists on arch35 / Ascend 950PR. A2 (910B) and A3 (910C) lack the SIMT path entirely — when targeting those chips, replace warp-cooperative patterns with UB-scratchpad block-reduce or sort-then-segment-reduce.
---

# Domain: Cooperative Group Programming
> Patterns for warp-level cooperative traversal, shuffle reduction, and value transport.
> Load when: Analyzer detects __shfl, __shfl_xor, GROUP_SIZE, or cooperative parallel loops.
> chip_scope: **a5-only** — see frontmatter; do NOT load when TARGET ∈ {a3, a2}.

---

## Patterns

### P-P13: Cooperative-group parallel traversal + shuffle reduction

**Severity**: High | **Source**: HKV hand-written version, CONFIRMED (generalized)

Transform single-thread sequential traversal into N-thread cooperative parallel + `__shfl_xor` divide-and-conquer reduction.

```cpp
auto rank = threadIdx.x % GROUP_SIZE;
for (uint32_t pos = rank; pos < array_size; pos += GROUP_SIZE) { ... }
for (int32_t offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    auto other = __shfl_xor(val, offset, GROUP_SIZE);
    if (other < val) val = other;  // or +=, max, etc.
}
```

Applicable to any scenario requiring min/max/sum within a warp.

---

### P-P16: Cooperative-group value transport

**Severity**: Medium | **Source**: HKV hand-written version, CONFIRMED (generalized)

GROUP_SIZE threads share the work of a large vector copy:
```cpp
for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
    dst[pos * dim + j] = src[idx * dim + j];
}
```

At dim=128: 128 stores → 16 threads each do 8 stores. Applicable to any large embedding/feature vector transport.

---

## P-P64: VEC↔Scalar interleaved compute with paired V_S / S_V flags

**Severity**: CRITICAL (precision-affecting if missing) | **Source**: CANN `ops-nn/optim/advance_step/op_kernel/advance_step_spec.h` lines 169-180, 235-251 (2026-04-24).

**Trigger**: a compute pass needs to (a) produce a VEC result, (b) extract per-element scalars from that result for indirect-index lookup or per-element conditional, (c) feed scalars back into a VEC operation.

**Pattern** (canonical 5-step):
```cpp
// 1. VEC produces result
Cast(bufferLocal, srcInt64, RoundMode::CAST_NONE, count);   // VEC: T -> int32

// 2. Wait for VEC writes to drain before scalar reads them
SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

// 3. Scalar pipe loop reads + mutates + writes back
for (i = 0; i < count; i++) {
    int32_t v = bufferLocal.GetValue(i);
    bufferLocal.SetValue(i, transform(v));    // or SetValue elsewhere
}

// 4. Wait for scalar writes to drain before VEC reads them
SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

// 5. Next VEC op
Add(out, bufferLocal, other, count);
```

**Why both flags are required**:
- `V_S` (VEC → Scalar fence): guarantees all VEC writes have committed to UB before scalar reads start. Without it, scalar may read stale data.
- `S_V` (Scalar → VEC fence): guarantees all scalar writes have committed before VEC reads start. Without it, VEC may read stale data.
- AscendC's `EnQue/DeQue` only syncs MTE2↔VEC and VEC↔MTE3 — it does NOT cover VEC↔Scalar. You MUST place these flags manually for the V↔S boundary.

**Failure mode without flags**: precision mismatch on cases where VEC and Scalar compute the same UB region in different orders. Often manifests as "passes when run alone, fails in batched run" — the most insidious symptom.

**Applicability**: any indirect-indexing op (gather/scatter where index needs scalar inspect), any per-row condition evaluation (e.g. `if (mask[i]) v += x`), any quantization step that reads scale-per-row scalars and applies them via VEC.

**Reference the helper template**: P-P64 cohabits naturally with the `SetWaitFlag<HardEvent>` template helper from CANN `advance_step_common.h`:
```cpp
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt) {
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}
```
This collapses 3 lines (FetchEventID + SetFlag + WaitFlag) to one call. Worth pasting into per-op kernel headers.
