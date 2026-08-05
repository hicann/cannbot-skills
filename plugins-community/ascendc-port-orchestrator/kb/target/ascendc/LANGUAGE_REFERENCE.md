# AscendC Language Reference

> Compiled from CANN 9.0.0-beta.2 official docs + CANN source code (ops-transformer, opbase, catlass).
> Last verified: 2026-04-07.

## 1. Memory Model: TPosition

AscendC abstracts physical memory into logical positions (TPosition). All vector-related positions map to **Unified Buffer (UB)** physically, but represent different pipeline stages.

| TPosition | Physical | Stage | Usage |
|-----------|----------|-------|-------|
| `VECIN` | UB | CopyIn (MTE2→VEC) | Input data from GM |
| `VECOUT` | UB | CopyOut (VEC→MTE3) | Output data to GM |
| `VECCALC` | UB | Compute only | Temporary variables |
| `A1`, `B1` | L1 Buffer | Cube input cache | Matrix operands |
| `A2`, `B2` | L0A/L0B | Cube compute | Small matrix blocks |
| `CO1`, `CO2` | L0C / UB | Cube output | Matrix result |

**Key insight**: VECIN, VECOUT, VECCALC are all physically UB. The distinction is for **pipeline synchronization**, not physical memory separation.

## 2. Memory Management: TQue vs TBuf

### TQue (Queue-based, with sync)

```cpp
TQue<QuePosition::VECIN, 4> xQueue_;   // depth=4, input pipeline
TQue<QuePosition::VECOUT, 2> yQueue_;  // depth=2, output pipeline
```

**API cycle**: `AllocTensor` → use → `EnQue` → `DeQue` → use → `FreeTensor`

**Synchronization semantics** (from official design principle doc):
- `EnQue` → emits hardware `set` signal = "data write is complete"
- `DeQue` → emits hardware `wait` signal = "wait for data write to finish"
- `AllocTensor` → emits `wait` = "wait for memory to be freed"
- `FreeTensor` → emits `set` = "signal memory is released"

**Sync targets by TPosition**:
- `TQue<VECIN>`: EnQue/DeQue syncs **MTE2→VEC** (load done → can compute)
- `TQue<VECOUT>`: EnQue/DeQue syncs **VEC→MTE3** (compute done → can store)

**DoubleBuffer**: `depth >= 2` enables pipeline overlap. When VEC processes buffer[0], MTE2 can load into buffer[1]. `depth=4` allows MTE2 to prefetch 3 buffers ahead of VEC.

### TBuf (Static buffer, NO sync)

```cpp
TBuf<TPosition::VECCALC> tmpBuf_;
```

**API**: `Get<T>()` → returns fixed LocalTensor. No Alloc/Free/EnQue/DeQue.

**Critical**: TBuf has **NO automatic synchronization**. When mixing TBuf with TQue, you MUST add explicit sync (PipeBarrier or SetFlag/WaitFlag) to prevent data hazards.

### TQueBind (Bidirectional queue)

```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
```

Binds VECIN and VECOUT so the same buffer can flow through both paths. Used in MoE operators for bidirectional data flow.

**Source**: `cann/ops-transformer/examples/fast_kernel_launch_example/csrc/moe_distribute_combine_v2/`

## 3. Synchronization Hierarchy (lightest → heaviest)

### Level 1: TQue EnQue/DeQue (automatic, recommended)

Targets a specific pipe pair. **Zero overhead** beyond the minimum sync needed.

```cpp
// MTE2→VEC sync (input pipeline)
xLocal = xQueue_.AllocTensor<float>();
DataCopy(xLocal, gmIn[offset], count);    // MTE2
xQueue_.EnQue(xLocal);                    // set: MTE2 done
xComp = xQueue_.DeQue<float>();           // wait: MTE2 done → VEC can proceed
Muls(xComp, xComp, w, count);            // VEC
xQueue_.FreeTensor(xComp);               // set: memory released

// VEC→MTE3 sync (output pipeline)
yLocal = yQueue_.AllocTensor<float>();
// ... VEC ops on yLocal ...
yQueue_.EnQue(yLocal);                    // set: VEC done
yOut = yQueue_.DeQue<float>();            // wait: VEC done → MTE3 can proceed
DataCopy(gmOut[offset], yOut, count);     // MTE3
yQueue_.FreeTensor(yOut);                 // set: memory released
```

### Level 2: SyncFunc (explicit, single-event)

Targets a specific pipe pair. Cleaner API than SetFlag/WaitFlag for single events.

```cpp
SyncFunc<AscendC::HardEvent::MTE2_V>();    // wait for MTE2 → VEC
SyncFunc<AscendC::HardEvent::V_MTE3>();    // wait for VEC → MTE3
SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // wait for MTE3 → MTE2
SyncFunc<AscendC::HardEvent::V_S>();       // wait for VEC → Scalar
SyncFunc<AscendC::HardEvent::S_V>();       // wait for Scalar → VEC
SyncFunc<AscendC::HardEvent::MTE2_S>();    // wait for MTE2 → Scalar
SyncFunc<AscendC::HardEvent::MTE3_S>();    // wait for MTE3 → Scalar
```

**Source**: `cann/ops-transformer/examples/fast_kernel_launch_example/csrc/moe_distribute_combine_v2/`

### Level 3: SetFlag/WaitFlag (explicit, multi-stream with flag IDs)

For complex pipelines with multiple concurrent data streams.

```cpp
// Flag IDs allow tracking multiple independent sync points
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);      // stream 0
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingPongFlag + 6);  // ping-pong
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingPongFlag + 6);
```

**Source**: `cann/ops-transformer/attention/nsa_compress_attention_infer/op_kernel/`

### Level 4: PipeBarrier (single-pipe barrier)

```cpp
PipeBarrier<PIPE_V>();    // barrier within VEC pipe only
PipeBarrier<PIPE_MTE2>(); // barrier within MTE2 pipe only
PipeBarrier<PIPE_MTE3>(); // barrier within MTE3 pipe only
```

**Use case**: When consecutive VEC ops on the **same** TBuf tensor have read-after-write dependency (e.g., in-place operations). Does NOT block other pipes.

**Source**: Widely used in CANN MoE quantization code for consecutive VEC ops.

### Level 5: PipeBarrier<PIPE_ALL> (all-pipe barrier, AVOID)

```cpp
PipeBarrier<PIPE_ALL>();  // syncs ALL pipes: MTE2 + VEC + MTE3 + Scalar + Cube
```

**Cost**: Drains all instruction queues. Prevents ALL pipeline overlap. **Measured 1.6-2.3x slowdown** vs TQue in SG forward (E13 data).

**When necessary**: Only when there's a true all-pipe dependency, e.g., DataCopy to GM (MTE3) must complete before the next DataCopy from same GM address (MTE2), AND VEC is also accessing the same UB region.

## 4. Common Patterns

### Pattern A: TQue Pipeline (CopyIn → Compute → CopyOut)

The standard AscendC paradigm. Each stage has its own TQue.

```cpp
// Members
TQue<QuePosition::VECIN, 2> inQueue_;
TQue<QuePosition::VECOUT, 2> outQueue_;

// Loop
for (int i = 0; i < n; i++) {
    // CopyIn
    auto x = inQueue_.AllocTensor<T>();
    DataCopy(x, gmIn[i * len], len);
    inQueue_.EnQue(x);
    // Compute
    auto xC = inQueue_.DeQue<T>();
    auto y = outQueue_.AllocTensor<T>();
    Abs(y, xC, len);
    inQueue_.FreeTensor(xC);
    outQueue_.EnQue(y);
    // CopyOut
    auto yO = outQueue_.DeQue<T>();
    DataCopy(gmOut[i * len], yO, len);
    outQueue_.FreeTensor(yO);
}
```

### Pattern B: Accumulator in TQue<VECOUT> (reduction loops)

For kernels that accumulate across iterations then write to GM. The accumulator is allocated from VECOUT queue, used for VEC ops, then EnQue'd for MTE3 write-back.

```cpp
// Members
TQue<QuePosition::VECIN, 4> xQueue_;   // input data (depth=4 for prefetch)
TQue<QuePosition::VECOUT, 2> yQueue_;  // accumulator + output

// Per-output loop
LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
Duplicate(yLocal, 0.0f, count);  // VEC op on VECOUT tensor — LEGAL before EnQue

for (int k = 0; k < num_inputs; k++) {
    LocalTensor<float> x = xQueue_.AllocTensor<float>();
    DataCopy(x, gmIn[src_offset], count);      // MTE2 (can overlap with prev VEC)
    xQueue_.EnQue(x);
    LocalTensor<float> xC = xQueue_.DeQue<float>();  // wait MTE2
    Muls(xC, xC, weight, count);                     // VEC (in-place on dequeued tensor)
    Add(yLocal, yLocal, xC, count);                   // VEC (accumulate)
    xQueue_.FreeTensor(xC);                           // release for reuse
}

yQueue_.EnQue(yLocal);                          // VEC → MTE3 sync
LocalTensor<float> yOut = yQueue_.DeQue<float>();
DataCopy(gmOut[dst_offset], yOut, count);       // MTE3
yQueue_.FreeTensor(yOut);
```

**Critical**: The accumulator (`yLocal`) is between `AllocTensor` and `EnQue` — all VEC ops (Duplicate, Muls, Add, Cast, etc.) are legal in this state. This is confirmed by:
- Official fusion operator example (LeakyRelu on VECOUT tensor before EnQue)
- Working SG forward PingPong code (E13)
- CANN MoE quantization code (BlockReduceMax, Cast, Muls, Div on VECOUT)

### Pattern C: Read-Modify-Write via TQue

When accumulating into an existing GM value (e.g., grad_in += local_accum):

```cpp
// Read existing GM value through xQueue_
LocalTensor<float> gi = xQueue_.AllocTensor<float>();
DataCopy(gi, gmGradIn[offset], count);
xQueue_.EnQue(gi);
LocalTensor<float> giC = xQueue_.DeQue<float>();
Add(yLocal, yLocal, giC, count);    // accumulate into VECOUT tensor
xQueue_.FreeTensor(giC);
// Then write back via yQueue_ (Pattern B epilogue)
```

### Pattern D: SyncFunc for TBuf operations

When TBuf is used (no queue sync), explicit sync is needed:

```cpp
TBuf<TPosition::VECCALC> tmpBuf_;
LocalTensor<float> tmp = tmpBuf_.Get<float>();

DataCopy(tmp, gmIn[offset], count);
SyncFunc<AscendC::HardEvent::MTE2_V>();   // explicit: MTE2 done → VEC can read tmp
Muls(tmp, tmp, scalar, count);
SyncFunc<AscendC::HardEvent::V_MTE3>();   // explicit: VEC done → MTE3 can write tmp
DataCopy(gmOut[offset], tmp, count);
```

## 5. Anti-Patterns

### Anti-Pattern 1: TBuf accumulator + TQue input (NO SYNC)

```cpp
// ❌ WRONG: TBuf has no sync, TQue sync only covers xQueue_ pipeline
TBuf<VECCALC> accumBuf_;               // NO sync
TQue<VECIN, 4> xQueue_;                // has MTE2→VEC sync

LocalTensor<float> accum = accumBuf_.Get<float>();
Duplicate(accum, 0.0f, count);
for (...) {
    auto x = xQueue_.AllocTensor<float>();
    DataCopy(x, gm[...], count);
    xQueue_.EnQue(x);
    auto xC = xQueue_.DeQue<float>();
    Muls(xC, xC, w, count);
    Add(accum, accum, xC, count);   // VEC writes accum (TBuf) — NO sync with MTE2!
    xQueue_.FreeTensor(xC);
    // Next iteration's DataCopy may overlap with this Add → UB bus contention
}
```

**Fix**: Move accumulator to `TQue<VECOUT>` (Pattern B).

### Anti-Pattern 2: PipeBarrier<PIPE_ALL> in hot loops

```cpp
// ❌ SLOW: syncs ALL pipes every iteration
for (...) {
    DataCopy(buf, gm[...], count);
    PipeBarrier<PIPE_ALL>();          // drains MTE2+VEC+MTE3+Scalar
    Muls(tmp, buf, w, count);
    Add(accum, accum, tmp, count);
    PipeBarrier<PIPE_ALL>();          // drains again
}
```

**Fix**: Use TQue (Pattern B) or SyncFunc (Pattern D).

### Anti-Pattern 3: Unnecessary PIPE_ALL when PIPE_V suffices

```cpp
// ❌ Overkill: only VEC→VEC dependency, but blocking all pipes
Duplicate(accum, 0.0f, count);
PipeBarrier<PIPE_ALL>();              // Only need VEC barrier
Add(accum, accum, tmp, count);
```

**Fix**: `PipeBarrier<PIPE_V>()` — only blocks VEC pipe.

## 6. HardEvent Reference

| HardEvent | From → To | Use Case |
|-----------|-----------|----------|
| `MTE2_V` | MTE2 → VEC | DataCopy from GM complete → VEC can read |
| `V_MTE3` | VEC → MTE3 | VEC compute complete → DataCopy to GM |
| `MTE3_MTE2` | MTE3 → MTE2 | Write-back complete → can load from same address |
| `MTE2_MTE1` | MTE2 → MTE1 | UB→L1 DMA coordination |
| `MTE1_MTE2` | MTE1 → MTE2 | L1→UB DMA coordination |
| `MTE1_M` | MTE1 → Cube | L1 data ready → Cube can compute |
| `M_MTE1` | Cube → MTE1 | Cube done → can load next |
| `V_S` | VEC → Scalar | VEC result needed by Scalar |
| `S_V` | Scalar → VEC | Scalar value needed by VEC |
| `S_MTE2` | Scalar → MTE2 | Address calc done → can start DMA |
| `S_MTE3` | Scalar → MTE3 | Address calc done → can start write-back |
| `MTE2_S` | MTE2 → Scalar | Load complete → Scalar can read |
| `MTE3_S` | MTE3 → Scalar | Write complete → Scalar can proceed |

## 7. SIMT Programming Model

AscendC supports SIMT (Single Instruction, Multiple Threads) alongside SIMD. On Ascend950PR, both models run on the same VEC core and can be mixed within a kernel.

### 7.1 SIMT Thread Model

```cpp
// Thread indexing (within a block)
uint32_t tid = Simt::GetThreadIdx();       // 1D flat index
uint32_t tid_x = Simt::GetThreadIdx<0>();  // X (innermost) dimension
uint32_t tid_y = Simt::GetThreadIdx<1>();  // Y (outer) dimension
uint32_t nthreads = Simt::GetThreadNum();
uint32_t nthreads_x = Simt::GetThreadNum<0>();

// Block indexing (across cores)
uint32_t bid = GetBlockIdx();
uint32_t nblocks = GetBlockNum();

// Launch kernel with thread dimensions
Simt::VF_CALL<my_kernel_vf>(Simt::Dim3{256}, args...);         // 256 threads, 1D
Simt::VF_CALL<my_kernel_vf>(Simt::Dim3{16, 32}, args...);      // 16x32 threads, 2D
```

### 7.2 SIMT Synchronization

| API | Scope | Use Case |
|-----|-------|----------|
| `__syncthreads()` / `Simt::ThreadBarrier()` | Thread block | Wait for all threads in block |
| `asc_threadfence()` | Global | Memory ordering (visibility, no blocking) |
| `asc_threadfence_block()` | Thread block | Memory ordering within block |
| `SyncAll<true/false>()` | Cross-core | All-core synchronization barrier |
| `CrossCoreBarrier<MODE, PIPE>()` | Cross-core | Catlass fine-grained cross-core sync |

```cpp
// Binary reduction pattern (from CANN ops-nn)
for (uint32_t k = nthreads_y / 2; k > 0; k /= 2) {
    if (tid_y < 2 * k) shared[tid_y * stride + tid_x] = value;
    Simt::ThreadBarrier();
    if (tid_y < k) value += shared[(tid_y + k) * stride + tid_x];
    Simt::ThreadBarrier();
}
```

**Source**: `cann/ops-nn/index/sorted_sparse_segment_mean_grad/op_kernel/arch35/`

### 7.3 SIMT Memory Model

| Memory | Qualifier | Physical | Access |
|--------|-----------|----------|--------|
| Global (GM) | `__gm__` | HBM | Through dcache + L2 cache |
| Shared / UB | `__local_mem__` / `__ubuf__` | UB SRAM | Direct (carved from UB) |
| dcache | (automatic) | UB SRAM | 128B cacheline, 32-128KB configurable |

```cpp
// dcache invalidation (thread 0 only)
if (Simt::GetThreadIdx<0>() + Simt::GetThreadIdx<1>() == 0) {
    __builtin_cce_dcci(nullptr, 1, 0);  // invalidate dcache
}
__syncthreads();

// Volatile GM access (prevent compiler optimization)
__gm__ volatile float* ptr = (__gm__ volatile float*)gm_addr;

// Access UB via physical address for SIMT thread access
__local_mem__ float* ubuf_ptr = (__local_mem__ float*)(tensor.GetPhyAddr());
```

### 7.4 SIMT Atomic Operations

```cpp
Simt::AtomicAdd(gm_ptr + offset, value);  // float, int32, half, bf16
// WARNING: GM atomicAdd goes through HBM, serialized. Avoid in hot loops.
// Use sorted-edge accumulation pattern (P-P21) instead.
```

### 7.5 SIMT + SIMD Mixed Mode

The VEC core can switch between SIMT and SIMD within a kernel. Data is exchanged via UB.

```cpp
// SIMD: allocate UB buffer via TQue
LocalTensor<float> buf = xQueue_.AllocTensor<float>();
DataCopy(buf, gmIn[offset], count);          // MTE2 DMA (SIMD)
xQueue_.EnQue(buf);

// Get physical UB address for SIMT
__local_mem__ float* ubuf = (__local_mem__ float*)(buf.GetPhyAddr());

// SIMT: irregular computation using threads
Simt::VF_CALL<my_simt_kernel>(Simt::Dim3{256}, ubuf, ...);

// SIMD: write back via TQue
LocalTensor<float> result = xQueue_.DeQue<float>();
DataCopy(gmOut[offset], result, count);      // MTE3 DMA (SIMD)
xQueue_.FreeTensor(result);
```

**Source**: `cann/ops-transformer/attention/mla_prolog/op_kernel/arch35/vf/`,华为内置算子 `diag_part_simt_simd.h`

### 7.6 SIMT Performance Patterns

**Grid-stride loop** (standard SIMT loop pattern):
```cpp
for (uint32_t i = bid * nthreads + tid; i < total; i += nblocks * nthreads) {
    output[i] = compute(input[i]);
}
```

**2D thread block decomposition**:
- Y threads → outer loop (rows/experts)
- X threads → inner loop (columns/hidden dims)

**Fast integer division** (avoid expensive modulo):
```cpp
uint32_t result = Simt::UintDiv(value, magic_number, shift);
```

### 7.7 Cross-Core Synchronization (Catlass)

```cpp
// From catlass/include/catlass/arch/cross_core_sync.hpp
constexpr uint8_t AIV_INTER_BLOCK_BARRIER = 8;
constexpr uint8_t AIC_INTER_BLOCK_BARRIER = 9;
constexpr int MAX_REVERSE_DEPTH = 15;  // prevents freeze in one-way sync

template <uint8_t MODE, pipe_t PIPE>
void CrossCoreBarrier() {
    constexpr FlagID flagId = BarrierFlag<MODE, g_coreType>::ID;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}
```

## 8. Documentation Resources

- **Official docs**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/
  - NOTE: JS-rendered site, use `dev-browser` skill for automated browsing
  - Key pages: "基于TPipe和TQue编程", "编程模型设计原理", "TBuf的使用", "使能DoubleBuffer"
- **CANN source code**: `~/workspace/cann/` (git fetch before reading)
  - `ops-transformer/examples/fast_kernel_launch_example/csrc/moe_*` — MoE with TQue/TQueBind/SyncFunc
  - `ops-transformer/attention/nsa_compress_attention_infer/` — SetFlag/WaitFlag multi-stream
  - `opbase/pkg_inc/op_common/atvoss/reduce/` — Reduce framework with TQue
- **AscendC API reference**: CANN 9.0 Ascend C API doc (separate doc set from operator dev guide)
