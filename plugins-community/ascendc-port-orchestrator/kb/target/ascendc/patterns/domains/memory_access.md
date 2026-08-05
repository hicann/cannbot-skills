---
applies_to: soc=all
reason: vec4 loads, tiling, and DataCopy alignment are universal AscendC concerns across a5/a3/a2. UB-size assumptions (256KB on a5, 192KB on a3/a2) MUST be parameterized — see CMP-002 in cross-platform notes.
---

# Domain: Memory Access Optimization
> Patterns for vectorized loads, adaptive tiling, and inner-loop variable sizing.
> Load when: Analyzer detects strided memory access, tile size selection, or loop variable types.

---

## Patterns

### P-P3: vec4 vectorization path enablement condition

**Severity**: Medium

**Anti-pattern**: `if (hidden_dim % 4 == 0 && grid_y > 1)` — `grid_y > 1` is a redundant gate.

**Correct**: `if (hidden_dim % 4 == 0)` — when grid_y==1, block_y=0 is entirely correct.

---

### P-P11: Adaptive tile size

**Severity**: **High**

```
dim ≤ 256:  <BRE=32,  TI=16>    16 edges × 32 emb threads
dim ≤ 512:  <BRE=64,  TI=8>     8 edges × 64 emb threads
dim > 512:  <BRE=512, TI=1>     1 edge × 512 emb threads
```

On a 56-AIV target, 56 blocks × TI=16 covers 896 edges per step. Tune TI
from the target workload and UB budget instead of copying a fixed launch shape.

---

### P-P12: int32 for inner-loop counters instead of int64

**Severity**: Medium

Use `int` for variables whose range is within int32 (loop counter j, thread_idx_emb, etc.). `edge_in[i] * emb_dim` must stay int64. Effect: fwd -3%, bwd -2%.

---

### P-P23: Contiguous-block partition vs grid-stride (adjacent-element comparison scenario)

**Severity**: **High** | **Source**: E10-1 assign_edges optimization (2026-03-30) | **Platform**: Ascend950PR

**Problem**: when comparing adjacent elements (`arr[i] vs arr[i-1]`), a grid-stride loop causes every read of `arr[i-1]` to be a cache miss (distance `total_threads` elements, typically 28672).

**Anti-pattern** (grid-stride, cache miss):
```cpp
for (int64_t i = tid; i < n; i += total_threads) {
    if (i == 0 || arr[i] != arr[i - 1]) { ... }  // arr[i-1] is 28672 elements from arr[i] in HBM
}
```

**Correct pattern** (contiguous block, cache-friendly):
```cpp
int64_t chunk = (n + total_threads - 1) / total_threads;
int64_t start = tid * chunk;
int64_t end = min(start + chunk, n);
for (int64_t i = start; i < end; i++) {
    if (i == 0 || arr[i] != arr[i - 1]) { ... }  // arr[i-1] adjacent to arr[i]
}
```

**Measured**: assign_edges sorted scan from 123ms → 10ms (**12x**). Total optimization (including atomicCAS avoidance): 259ms → 10ms (**25.6x**).

**Trigger condition**: adjacent access `arr[i-1]` or `arr[i+1]` inside the loop + grid-stride loop → switch to contiguous block.

**Note**: contiguous-block partition is not appropriate for all scenarios. If each iteration's data is fully independent (no adjacent dependency), grid-stride's coalesced access is instead better. Use contiguous block only when **adjacent-element comparison / dependency is required**.

---

### P-P24: Sort-to-Reuse — eliminate GM read amplification from indirect addressing

**Severity**: **CRITICAL** | **Source**: E11 msprof analysis (2026-03-31) | **Platform**: Ascend950PR (no L2 cache)

**Problem**: multiple work items read the same GM data via indirect indexing; each access is an independent HBM read.

```
// Anti-pattern: per-token iteration, input[expert] is re-read N times by N tokens
for token in all_tokens:
    expert = index[token]
    for d in hidden_dim:
        val = input[expert * hdim + d]  // 128 tokens share the same expert, read 128 times!
```

**Root cause**: this Ascend NPU SIMT path does not provide an effective cache for the repeated scalar reads. Every scalar GM read reaches HBM; when N work items read the same row via `arr[index[i]]`, actual HBM read volume is N × row_size rather than 1 × row_size.

**msprof verification** (E11, SG backward xlarge):
- `vec_ratio = 0.99` **does not equal** 99% atomicAdd
- Removing atomicAdd only saves 4.4% (158us / 3605us)
- **95%+ of VEC time is scalar GM random reads** (SIMT scalar reads go through the VEC pipe)
- In SIMT mode, `msprof vec_ratio` = sum of GM reads + compute + atomicAdd; it cannot be separated

**Correct pattern**: Sort-to-Reuse — sort by the indirect index so shared data is loaded once.

```
// Step 1: counting sort edges by expert_index → sorted_edges[], expert_offsets[]
// Step 2: per-expert processing
for expert in all_experts:
    DataCopy(local_buf, input[expert * hdim], hdim)  // read 1 time, not 128
    for edge in expert_run:
        token = sorted_edges[edge]
        // compute with local_buf, no need to re-read input[expert]
```

**Effect (measured)**:
| | GM reads | Time |
|---|:-:|:-:|
| per-token SIMT | 32K × 4096 = **134M** reads | 3447us |
| per-expert SIMD sorted | 256 × 4096 = **1M** reads | 265us |
| **Read reduction** | **128x** | **13x speedup** |

**Trigger condition**: you see `arr[index[i]]`-form indirect GM reads + multiple work items sharing the same index value → consider sort-to-reuse.

**Applicability**:
- MoE scatter-gather: `input[expert_index]` shared by N tokens
- GNN message passing: `features[neighbor_id]` shared by multiple edges
- Embedding lookup: `embedding[token_id]` shared across multiple positions
- Any indirect addressing pattern with fan-out

**Not applicable**:
- Index is fully unique (no sharing) → sorting yields no reuse benefit
- Data size < UB capacity → load once; no sorting needed

**Relation to P-P21**: P-P21 (sorted-edge accumulation) focuses on eliminating atomicAdd write conflicts. P-P24 focuses on eliminating GM read amplification. They typically appear together (the same sort fixes both read and write), but P-P24's benefit is much larger than P-P21's (95% vs 4.4% for SG backward).

**Relation to msprof interpretation**: a high SIMT-mode `vec_ratio` is not necessarily an atomicAdd bottleneck. You must compare timings of "with atomicAdd" vs "without atomicAdd" kernels to confirm. If the difference is <10%, the real bottleneck is GM reads.

---

### P-P25: SetAtomicAdd + DataCopyPad — hardware atomic write in SIMD mode

**Severity**: **CRITICAL** | **Source**: E12 expert SIMD backward (2026-03-31) | **Platform**: Ascend950PR

**Problem**: in SIMD mode, scatter-add (e.g. `grad_in[expert] += weight * grad_out`) needs an atomic write, but SIMT `atomicAdd` goes through a VEC-pipe CAS loop (slow), and `SetValue` in AIV mode is unreliable (OL-19).

**Correct pattern**: `SetAtomicAdd<T>()` + `DataCopyPad` — hardware atomic on the MTE3 pipe

```cpp
// SIMD backward: write each token's grad_in contribution back to the expert slot
Muls(gradInLocal, gradOutLocal, expertWeight, hdim);
// EnQue + DeQue for pipeline sync
gradInOutQue_.EnQue(gradInLocal);
LocalTensor<float> gradInOut = gradInOutQue_.DeQue<float>();
// MTE3 atomic add: atomicity guaranteed by hardware, no VEC CAS
SetAtomicAdd<float>();
DataCopyPad(gradInGm_[expertIdx * hdim], gradInOut, copyParams);
SetAtomicNone();
gradInOutQue_.FreeTensor(gradInOut);
```

**Comparison**:
| Method | Pipe | Mechanism | Speed |
|--------|------|-----------|-------|
| SIMT `atomicAdd(ptr, val)` | VEC | CAS loop | Slow (serialized under contention) |
| SIMD `SetAtomicAdd` + `DataCopyPad` | **MTE3** | Hardware atomic DMA | **Fast** (bulk atomic add) |
| SIMD `SetValue` (GM) | Scalar | — | Unreliable (OL-19) |

**Key advantage**: DataCopyPad transfers the whole hdim vector in one atomic add, not per-element CAS.
**No sorting needed**: per-token SIMD + SetAtomicAdd writes back directly; no counting-sort preprocessing required.

---

### P-P26: SetFlag/WaitFlag fine-grained event sync (replacing PipeBarrier<PIPE_ALL>)

**Severity**: High | **Source**: E12 expert code (2026-03-31) | **Platform**: Ascend950PR

**Problem**: `PipeBarrier<PIPE_ALL>()` blocks all pipes, preventing MTE2/VEC/MTE3 parallelism.

**Correct pattern**: use `SetFlag`/`WaitFlag` + `HardEvent` to specify cross-pipe dependencies precisely.

```cpp
// MTE2→Scalar: can only GetValue from UB after DataCopy completes
event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
SetFlag<HardEvent::MTE2_S>(id);
WaitFlag<HardEvent::MTE2_S>(id);

// VEC→Scalar: can only GetValue the result after ReduceSum completes
event_t id2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(id2);
WaitFlag<HardEvent::V_S>(id2);

// Scalar→VEC: can only run Muls after SetValue completes
event_t id3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(id3);
WaitFlag<HardEvent::S_V>(id3);

// Scalar→MTE3: can only DataCopyPad back after SetValue completes
event_t id4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
SetFlag<HardEvent::S_MTE3>(id4);
WaitFlag<HardEvent::S_MTE3>(id4);
```

**Interaction with TQue**: TQue<VECIN,2>'s AllocTensor/EnQue/DeQue/FreeTensor manages double-buffering automatically. SetFlag/WaitFlag is for scalar↔vector sync outside of TQue.

**Note**: OL-4 (TQue data corruption) may be a specific buffer-size / configuration issue. E12 expert code uses TQue<VECIN,2> + SetFlag and works correctly.

---

### P-P28: Automatic pipeline overlap with TQue (replacing PipeBarrier + manual Ping-Pong)

**Severity**: **CRITICAL** | **Source**: expert E13 (2026-04-01) | **Effect**: SG fwd **1.6-2.3x** over PipeBarrier

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

---

### P-P29: Batch Preload Cache — eliminating scalar GM read bottleneck

**Severity**: CRITICAL | **Source**: expert E12 (2026-03-31) | **Effect**: scalar pipe 42% → estimated <10%

**Scenario**: SIMD kernel loop needs to read a small amount of scalar data (index/weight); each `GetValue()` reads from GM at ~100 cycles.

**msprof evidence**: scalar=42% (SG backward); most of it is GetValue GM scalar reads.

**Anti-pattern** (per-element GM read):
```cpp
for (int i = 0; i < actual_k; i++) {
  local_index[i] = indexGm_.GetValue(index_base + i);   // ~100 cycle per read
  local_weight[i] = weightGm_.GetValue(index_base + i); // ~100 cycle per read
}
```

**Correct pattern** (batch preload into UB cache):
```cpp
// Init: allocate cache buffers
static constexpr uint32_t CACHE_SIZE = 1024;
pipe_.InitBuffer(idxCacheBuf_, CACHE_SIZE * sizeof(int32_t));
pipe_.InitBuffer(wtCacheBuf_, CACHE_SIZE * sizeof(float));

// Cache accessor: batch load on cache miss
__aicore__ inline int32_t GetIndexCached(int64_t idx, int64_t endIdx) {
  if (idx >= idxCacheBase_ + idxCacheLen_) {
    uint32_t copyLen = min(endIdx - idx, CACHE_SIZE);
    DataCopyPad(cache, indexGm_[idx], {1, copyLen * sizeof(int32_t), 0, 0}, padNone);
    idxCacheBase_ = idx; idxCacheLen_ = copyLen;
    // MTE2→Scalar sync: GetValue only after DataCopyPad completes
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(ev);
    WaitFlag<HardEvent::MTE2_S>(ev);
  }
  return cache.GetValue(idx - idxCacheBase_);  // ~1 cycle from UB
}
```

**Principle**: one GM DMA loading 1024 elements (amortized ~0.1 cycle/element) replaces 1024 scalar GM reads (~100 cycle/each). Hit rate depends on whether consecutively accessed indices in the loop fall in the same cache block.

**Applicability**:
- GM scalars read sequentially in a loop (index, weight, offset, etc.)
- When total access > cache size, chunked load; accesses must be increasing
- Non-increasing accesses (e.g. indirect indexing) do not fit this pattern

**Combine with P-P28 (Ping-Pong)**: P-P29 eliminates the scalar read bottleneck (scalar pipe); P-P28 overlaps MTE2/VEC. They are orthogonal.

**Cache size selection**: 1024 is an expert empirical value. Too large wastes UB; too small increases miss frequency. Adjust based on UB headroom and top_k.

---

### P-P62: Row-Scalar VEC Multiply — avoiding per-row scalar multiply on the Scalar pipe

**Severity**: HIGH | **Source**: CANN ops-transformer flash-attention source scan (2026-04-21) + hiascend.com Brcb API ref | **Applicability**: VEC computations with one scalar coefficient per row (dequant scale, softmax row-max, attention score, etc.)

**Scenario**: in a SIMD kernel, each row has a scalar coefficient; the canonical form is `Muls(work_row, work_row, scale_k, H)`, with a different `scale_k` per row. A direct `GetValue(scale[i]) → Muls(...)` triggers the whole MTE2_S → S_V → V_S Scalar-VEC sync chain. Empirically on op#11, `scalar_ratio ~= 0.44` (close to Scalar-VEC serialization).

**Correct pattern (RowMuls pattern, from `ops-transformer/attention/incre_flash_attention/op_kernel/ifa_public_define.h`)**:

```cpp
// src1Ub shape: [dealRowCount, 8_fp32_or_16_fp16]
// Each row's scalar has been pre-filled into a 32B block (via Brcb or Duplicate)
BinaryRepeatParams params;
params.src0BlkStride = 1;
params.src1BlkStride = 0;       // KEY: src1 does not step across blocks (the whole block is a single scalar broadcast)
params.dstBlkStride  = 1;
params.src0RepStride = columnCount / blockElementNum;  // row_stride in blocks
params.src1RepStride = 1;       // each repeat advances by 1 block = the next row's scalar
params.dstRepStride  = columnCount / blockElementNum;

AscendC::Mul(dst, src0, src1Ub, /*elements_per_repeat*/ REPEAT_ELEMENT_NUM,
             /*repeatTimes*/ dealRowCount, params);
```

**How to fill src1Ub** (two choices):
- `Brcb(src1Ub, scalars_ub, dealRowCount, brcbParams)` — AscendC API ref 07_0089, a dedicated hardware "8 scalars → 8 × 32B blocks" instruction
- `Duplicate<T>(src1Ub_block_i, scalar_i, blockElementNum)` in a single loop (low overhead since it is only init-time)

**Principle**: `src1BlkStride=0` + `src1RepStride=1` make VEC `Mul` treat each row's 32B block as "one scalar broadcast across the row", completely on the VEC pipe without touching the Scalar pipe. Equivalent to "per-row scalar × whole-row vector" but latency ~= 1 VEC Mul instead of N Scalar-syncs.

**Anti-pattern**:
```cpp
for (int i = 0; i < N; i++) {
  T scale = scaleUb.GetValue(i);   // Scalar pipe, ~100 cycle
  Muls(work[i*H], work[i*H], scale, H);  // triggers S_V sync
}
```

**Evidence of CANN internal usage**:
- `ops-transformer/attention/incre_flash_attention/op_kernel/ifa_public_define.h` `RowMuls`
- `ops-transformer/attention/sparse_flash_attention/op_kernel/arch32/sparse_flash_attention_service_vector_mla.h`
- `ops-transformer/attention/nsa_selected_attention_infer/op_kernel/nsa_public_define.h`

Standard per-row scale pattern for flash-attention-class operators.

**op#11 opportunity**: DequantSwigluQuant has per-row dequant_scale_k; currently uses `GetValue→Muls` giving `scalar_ratio 0.44`. Switching to the RowMuls pattern should be significantly faster.

**Measured data** (`probe_findings/2026-04-21_Q_scalar_broadcast.md`, N=64 rows × H=8 fp32 in one block, warmup 10 + 20 measured, median):

| Variant | median time (us) | ratio vs K_base |
|---------|------------------|-----------------|
| `K_base` (per-row GetValue + Muls) | 758.53 | 1.00x |
| `K_muls_flex` ("Muls flexible scalar position" arg-order variant) | 778.52 | 0.97x — **useless**; argument order does not change the pipe path |
| `K_brcb` (single Brcb + single wide Mul with src1BlkStride=0/src1RepStride=1) | **29.95** | **25.3x** WINNER |

Brcb path wins decisively; this pattern is exactly that path. Note: the probe uses H=8 (block-width matched) which amplifies the ratio; at the realistic H=2048 the ratio will converge but is still expected to be significant (follow-up probe with shape-matched H TBD).

**Counter-lesson** (from this probe): the "flexible" in `Muls (flexible scalar position)` refers to **argument position** (dst, src, scalar vs dst, scalar, src), **not** scalar source (UB vs register). Both overloads go through the Scalar pipe; neither bypasses the sync chain. Names in the hiascend.com API list can mislead; you must read the signature.

**Applicability (critical — lesson from op#11 Kind-1 respawn 2026-04-21)**:
- Brcb's 25.3x comes from **one Brcb + 1 wide Mul covering N_ROWS rows**. If the kernel is currently `for r in N: process_one_row(r)` (1 row per iter), each iter has only 1 scalar; Brcb degenerates to "put a single scalar in 1 block then wide Mul" — equivalent to (actually **worse than**) the original `Muls(dst, src, scalar, H)`.
- The precondition of P-P62 is **the kernel has a multi-row parallel axis to amortize over**. A single-row loop does not satisfy it.
- If the kernel is single-row-per-iter, applying P-P62 requires a **Kind-2 architecture rewrite**: batch R ≥ 8 rows per iter. This means UB must hold R row buffers simultaneously (H=4992 fp32 × R=8 = 160 KB for work alone, plus other buffers can exceed 192 KB); usually not cost-effective.
- Actual op#11 result: static audit found only 2 scalar Muls in the kernel (`as_scalar` + `dyn_scale`); the other per-row operations are already H-wide vector ops. Brcb has no room to fold; Kind-1 retrofit failed.

---

### P-P65: Cross-phase buffer-liveness aliasing for fused ops (UB budget relief)

**Severity**: HIGH | **Source**: op#11 aog-fused-optimizer pilot (2026-04-21) | **Applicability**: UB budget overflow in fused operators; the incremental optimizer (aog-kernel-optimizer) often picks the wrong alias target

**Scenario**: a fused op's ProcessRow() has multiple phases (dequant / SwiGLU / quant / ...); each phase has its own scratch buffer. When the UB budget is tight you want to alias to save space, but the aog-kernel-optimizer's global view often picks the wrong target (aliasing onto a still-live buffer) → silent UB overwrite → precision FAIL.

**Method (fused-op liveness graph)**:
1. For each phase, list every UB buffer's live range (which phase reads, which writes; on which line the last read occurs)
2. Find buffers **"dead past phase N"** — i.e. not used after a certain phase reads them, but physically still occupying a slot
3. The slot can be aliased to a later phase's new scratch; no need to InitBuffer a new TBuf
4. Key: alias target must be a buffer **dead within a cross-phase window**; a buffer still alive in the same phase (e.g. the amax scratch tmpBuf still used by DynQuant) cannot be aliased

**Verification step**: precision FULL re-verify (not a single case) — a wrong liveness judgment produces silent errors.

**op#11 example**:
- `otherBuf_` is dead after the last Mul in SwiGLU (line 413)
- Previously, aog-kernel-optimizer Opt4 chose `tmpBuf_` as the alias target → `tmpBuf` is still alive in DynQuant's amax reduction → silent overflow → precision FAIL → REVERT
- aog-fused-optimizer C4 correctly chose `otherBuf_` → precision PASS on first try, 20 KB released
- This is the first empirical win of the fused-op view over the global view

**Anti-pattern (i.e. the trigger for PB-17)**:
- Aliasing a buffer that is **VEC-written near the end of ProcessRow** and **MTE2-written near the start of the next ProcessRow** creates a V→MTE2 cross-row hazard (no sync → silent corruption). See PLATFORM_BUGS PB-17.
- As a heuristic: the "dead range" of an alias target must be fully clear across rows, or explicitly `SetFlag<HardEvent::V_MTE2>` at the end of ProcessRow.

**When to use this pattern**:
- Fused op, UB budget near the limit, depth=2 / tile expansion blocked
- Need to free UB for architectural restructure
- aog-kernel-optimizer has plateaued or Opt-N tried alias but precision FAILs

---

### P-P33: SIMT→SIMD conversion (memory-bound elementwise kernel)

**Severity**: **HIGH** | **Source**: MXFP4 migration (2026-04-07) | **Status**: candidate (pending SIMD implementation verification)

**Trigger condition**: msprof shows MTE2=0% AND throughput < 50% of theoretical bandwidth

**Scenario**: a SIMT kernel performs elementwise/per-group ops; all GM accesses go through dcache (VEC pipe); the MTE2 DMA engine is completely idle.

**Diagnosis**:
```
msprof data:
  aiv_vec_ratio: high (>70% for large tensors)
  aiv_mte2_ratio: ~0%
  aiv_mte3_ratio: ~0%
  Throughput: actual 125 GB/s vs theoretical 400 GB/s (31%)
```

**Reason**: in SIMT mode, GM reads/writes go through dcache (128B cacheline), not through MTE2 DMA. The VEC pipe carries both compute and memory access, and the two cannot parallelize.

**Optimization**: switch to SIMD or hybrid mode:
```
Option A (pure SIMD): DataCopy(MTE2) → VEC compute → DataCopy(MTE3)
  - Applicable when the computation can be expressed with SIMD vector instructions
  - TQue<VECIN,4> + TQue<VECOUT,2> for automatic pipeline overlap

Option B (hybrid SIMT+SIMD):
  - SIMD DataCopy bulk-loads into UB
  - GetPhyAddr() obtains the UB physical address
  - SIMT VF_CALL performs irregular computation (e.g. bit ops)
  - SIMD DataCopy writes back to GM
  - Applicable when the computation contains operations SIMD does not support (e.g. reinterpret float↔int)
```

**Expected effect**: 2-3x throughput gain on large tensors (MTE2+VEC dual-pipe parallelism)

**Constraints**:
- MXFP4-specific analysis required: the PyTorch version's algorithm uses float math (log2, floor, pow2) and may be fully expressible in SIMD
- The source version uses bit ops (reinterpret cast, bit shift) and must use hybrid mode

**Important limitation (verified 2026-04-07)**:
the SIMD version of MXFP4 is **4-20x slower** than SIMT. Reason: MXFP4 quantization needs per-element x_exp and shift amount, which cannot be expressed with SIMD vector instructions (different shift per element). SIMD degenerates to per-element GetValue/SetValue scalar operations, far slower than SIMT's 128-thread parallelism.

**P-P33 applicability update**:
- Applies: computation fully expressible with SIMD vector instructions (Add, Muls, Cast — same op for every element)
- Applies: SG forward/backward (contiguous DataCopy + Muls + Add, all vector ops)
- Not applicable: per-element heterogeneous computation (e.g. MXFP4's per-element log2/pow2/shift)
- Not applicable: quantization / bit-ops requiring per-element conditional branching

**Decision criterion**: check whether the inner loop executes the **exact same instruction sequence** for every element (same Muls/Add/Cast). If each element needs different ops (different shift amount, different branch), SIMT is better.

**SIMD V4 "fast" experiment (2026-04-07)**:
tile-wide shared exponent (no per-group loop) is **1.08x faster than SIMT** on small tensors, proving SIMD itself is not slow.
But **precision is broken**: using a tile-level exponent instead of a per-32-group exponent lowers quantization precision (does not meet MXFP4 spec, cannot be used as production).

**Full comparison (same-NPU A/B)**:
| Version | 4K(ms) | 4M(ms) | Precision | Production-capable |
|:---:|:---:|:---:|:---:|:---:|
| SIMT (128 threads) | 0.018 | **0.253** | OK PyTorch exact | OK **production** |
| SIMD V3 (per-group vectorized) | 0.029 | 1.724 | OK PyTorch exact | NO, slower than SIMT |
| SIMD V4 fast (tile-wide) | **0.017** | 0.813 | WARN **precision degraded** | NO, violates spec |

**WARNING — precision**: SIMD V4 fast replaces the per-32-group exponent with a tile-wide shared exponent.
This means 1024 elements share one exponent, while the MXFP4 spec requires one per 32 elements.
When intra-tile values vary widely (some near 0, some large), small values underflow to 0.
**The A3 hand-written SIMD implementation has the same issue — that is the root cause of its precision bug.**

**P-P33 final conclusion**:
1. The SIMD perf bottleneck is not SIMD mode itself, but the **per-group serial loop**
2. Eliminating the per-group loop (tile-wide processing) lets SIMD be faster than SIMT
3. But eliminating the per-group loop = abandoning per-group precision = **violates spec**
4. For group-local quantization operators, **SIMT is the only approach that meets both precision and performance**
5. SIMD applies when group_size >= tile_size, or when per-group precision is not required

**Evidence**: MXFP4 full-chain verification (2026-04-07): msprof + SIMD V1/V2/V3/V4 A/B + PyTorch spec comparison


---

## P-P49: Per-uid 8-aligned GM scalar slot (multi-core scalar write)

**Trigger**: SIMD kernel where each AIV core needs to write **a single scalar value**
(per-block mean / rstd / sum / count) to a shared GM array indexed by core uid.

**Problem**: A5 `DataCopy(GM, UB, count)` has a 32B minimum granularity:
- fp32: writes a block of 8 elements
- fp16/bf16: writes a block of 16 elements

If multiple cores write to adjacent indices in a GM array (e.g., `out[uid] = value`),
each core's DataCopy writes its **entire 8/16 element block**, which **overwrites
neighboring cores' data**. Result: non-deterministic / partially zero output.

**Solution**: Allocate per-uid **isolated 8-aligned slots** in GM:
```cpp
// Host: allocate (num_uids * 8) elements instead of num_uids
auto mean_buf = torch::zeros({num_uids * 8}, ...);

// Kernel: each uid writes to its own dedicated slot
DataCopy(meanGm[uid * 8], localMean, 8);  // 8 = full alignment block

// Host extracts every 8th element via stride select:
auto mean = mean_buf.view({num_uids, 8}).select(1, 0).contiguous();
```

**Why it works**: Each core's write block (8 elements) lands in a non-overlapping
region. The first element in each block is the actual scalar; the other 7 are zero
padding. Pybind layer extracts the first element via `.select(1, 0)`.

**Cost analysis**: 8x GM bandwidth for the per-uid output (e.g., for 64 cores writing
mean+rstd, this is 64 * 8 * 4B * 2 = 4KB instead of 512B). Negligible vs the
input/output tensor sizes.

**When to use**:
- ✅ Per-block scalar reduction outputs (mean, max, count)
- ✅ Norm-style kernels with per-group statistics
- ✅ Any pattern where each core produces a single scalar destination

**When NOT to use**:
- ❌ When per-core output is already ≥ 8 elements (no race)
- ❌ When using WorkspaceMerge / accumulate-into-shared (different pattern)
- ❌ When atomicAdd is acceptable (use SetAtomicAdd directly)

**Related**: OL-70 (root cause), PB-9 (related UB→UB DataCopy pitfall), P-P25 (atomicAdd-based scatter)

**Evidence**: 2_GroupNormSwish (Level 2, 2026-04-15): mean/rstd outputs are per (N, group) scalars.
Initial direct write to `meanGm[uid]` produced sporadic zero outputs across cases. After per-uid
8-aligned slot conversion: 50/50 PASS, 2.05x speedup. Validated on fp16/fp32/bf16, 2D-6D shapes.

---

## P-P66: `TQueBind<VECIN, VECOUT, 1>` for in-place UB buffer reuse

**Severity**: Useful (saves ~tile size of UB) | **Source**: CANN `ops-nn/optim/advance_step/op_kernel/advance_step_spec.h` lines 57-58, 130, 147 (2026-04-24).

**Trigger**: a compute pass reads a buffer, mutates in-place, writes the same buffer back to GM. Separate `TQue<VECIN>` + `TQue<VECOUT>` would consume 2× the UB.

**Pattern**:
```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer1;

pipe_->InitBuffer(buffer1, /*depth=*/1, /*bytes=*/tileSize);

// CopyIn:
LocalTensor<T> local = buffer1.AllocTensor<T>();
DataCopyPad(local, srcGm[offset], params, padParams);
buffer1.EnQue(local);

// Compute (mutate in place):
LocalTensor<T> local = buffer1.DeQue<T>();
Cast(local, local, RoundMode::CAST_NONE, count);
Adds(local, local, 1, count);
buffer1.EnQue(local);

// CopyOut:
LocalTensor<T> local = buffer1.DeQue<T>();
DataCopyPad(dstGm[offset], local, params);
```

**Trade-off**:
- `TQueBind<VECIN, VECOUT, 1>` = 1 × tileSize UB, forced serial CopyIn → Compute → CopyOut on each tile.
- `TQue<VECIN, depth=2>` + `TQue<VECOUT, depth=2>` = 4 × tileSize UB, but CopyIn(iter+1) overlaps Compute(iter).

**When to use**:
- UB budget is tight (multi-tensor fused ops where every KB matters; cf. op#11 DequantSwigluQuant peak 194/192 KB).
- Op is memory-bound on small tiles (pipeline depth doesn't help — bandwidth saturates anyway).
- In-place compute fits in one VEC sequence (no need to keep src + dst alive simultaneously).

**When NOT to use**:
- Op is VEC-bound and benefits from depth ≥ 2 pipelining (don't trade throughput for UB).
- Input and output dtypes/sizes differ (would need 2 separate physical slots regardless).

**Related to PB-17 (P-P65)**: PB-17 is the aggressive form — manually aliasing two named TBufs onto the same physical UB slot, which the framework does not manage. `TQueBind` is the framework-managed version with proper EnQue/DeQue protocol; **use it first**, only escalate to manual aliasing if `TQueBind` doesn't fit the access pattern.

## P-P92: V220 PipeBarrier<PIPE_V> coalescing — remove redundant intra-pipe barriers

**Domain**: memory_access / sync
**Severity**: HIGH — observed 6.8× perf impact
**Arch**: V220 / arch22 (Ascend910B-series) only
**Companion**: P-P77 (PipeBarrier precision regression — 6/10 wrong outputs from extra barriers on unrelated V ops)

### When to apply

A SIMD elementwise/fused kernel that:
1. Uses TBuf-based pipeline with explicit `SetFlag`/`WaitFlag` at PIPE crossings
2. Emits > 5 `PipeBarrier<PIPE_V>()` calls per row-loop iteration
3. Profiler shows `aiv_vec_ratio < 60%` on what should be VEC-bound
4. Runs on V220 / arch22 (Ascend910B2C, Ascend910B4, etc.)

### Principle

On V220 arch22, the **VEC pipe is in-order within its own pipe**. Consecutive `PIPE_V` operations auto-serialize via data dependencies — the hardware tracks which VEC op reads which UB region and which VEC op wrote it last. An explicit `PipeBarrier<PIPE_V>()` between two back-to-back VEC ops performs ZERO useful synchronization: the data dependency already serializes them. Worse, each `PipeBarrier<PIPE_V>()` forces a **pipe drain** (~tens of cycles latency), flushing the VEC pipeline and waiting for all prior VEC ops to retire before proceeding.

In a row-loop kernel with N such barriers per row × M rows, the cumulative drain overhead **dominates total runtime**, turning a compute-bound kernel into a sync-overhead-bound one.

**What PipeBarrier<PIPE_V> does NOT protect against**: data races between PIPE_V and PIPE_MTE2 / PIPE_MTE3 / PIPE_S. Those are cross-pipe operations and the in-order guarantee does NOT cross pipe boundaries. `SetFlag`/`WaitFlag` at phase boundaries is still REQUIRED for those.

### Fix

```cpp
// BEFORE (anti-pattern): 24 PipeBarrier<PIPE_V> calls
Mul(buf, buf, scale, count);
PipeBarrier<PIPE_V>();       // ← REDUNDANT — Mul writes buf, Cast reads buf, data dep serializes
Cast(tmp, buf, CAST_NONE, count);
PipeBarrier<PIPE_V>();       // ← REDUNDANT
Sigmoid(buf, tmp, count);
PipeBarrier<PIPE_V>();       // ← REDUNDANT
// ... etc — 24 total per row

// AFTER (P-P92 fix): 0 PipeBarrier<PIPE_V> calls
Mul(buf, buf, scale, count);
// No barrier — VEC pipe auto-serializes Mul→Cast via buf data dependency
Cast(tmp, buf, CAST_NONE, count);
// No barrier — VEC pipe auto-serializes Cast→Sigmoid via tmp data dependency
Sigmoid(buf, tmp, count);
// ... cross-pipe sync (V→MTE3, V→S) still uses SetFlag/WaitFlag at phase boundaries
```

**RETAIN** these cross-pipe sync points:
- `SetFlag<HardEvent::V_MTE3>(ev)` / `WaitFlag<HardEvent::V_MTE3>(ev)` — ensure MTE3 sees VEC writes before DMA-out
- `SetFlag<HardEvent::MTE2_V>(ev)` / `WaitFlag<HardEvent::MTE2_V>(ev)` — ensure DMA-in completes before VEC reads
- `SetFlag<HardEvent::V_S>(ev)` / `WaitFlag<HardEvent::V_S>(ev)` — scalar pipe must see VEC broadcasts
- `SetFlag<HardEvent::S_V>(ev)` / `WaitFlag<HardEvent::S_V>(ev)` — VEC must see scalar writes

### Evidence

op#11 DequantSwigluQuant kw-3 H1 (2026-05-13):
- **Pre-H1**: 24 `PipeBarrier<PIPE_V>()` calls → 354 µs wall-clock on [128,4096] → 0.19× CANN (5.4× slower)
- **Post-H1**: 0 `PipeBarrier<PIPE_V>()` calls → 52 µs wall-clock → **6.8× speedup**, 1.27× CANN (faster)
- **Precision**: 49/50 PASS_T1 preserved bit-exact (H1 is sync-only, zero arithmetic change)
- **Determinism**: 50/50 identical preserved (VEC in-order semantics unchanged)
- **Researcher prediction was 1.5–2.0×; actual was 6.8×** — underestimated because the cumulative drain cost of 24 barriers per row × 456 rows wasn't modeled

### Limits

- **V220 only**: confirmed on Ascend910B2C (arch22). A5/V351 reg-based SIMD may have different pipe semantics — verify before applying.
- **Intra-VEC only**: only `PipeBarrier<PIPE_V>()` is safe to remove. Do NOT remove `PipeBarrier<PIPE_MTE2>()`, `PipeBarrier<PIPE_MTE3>()`, `PipeBarrier<PIPE_S>()`, or `PipeBarrier<PIPE_ALL>()`.
- **Phase-boundary flags stay**: SetFlag/WaitFlag at PIPE crossings remain REQUIRED. This pattern only removes intra-VEC-pipe barriers between consecutive VEC ops.
- **Detection pre-condition**: kernel must already have correct `SetFlag`/`WaitFlag` at phase boundaries. If phase-boundary sync is wrong, removing intra-VEC barriers won't fix it.
