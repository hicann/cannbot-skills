# AscendC Error Corrections Reference

> Structured error→repair mappings for common AscendC SIMT compile errors.
> Load when: Generator encounters compile errors in Stage 1 compile-fix loop.
> Format: Error pattern → Root cause → Fix → Related pattern ID

> **`applies_to_backend:` tag** (P132): default is `ascendc`. EC entries
> are almost entirely AscendC-specific (bisheng / aclnn compile errors,
> AscendC SIMT/SIMD signatures). Cross-backend EC is rare. If an entry
> documents a hardware-runtime symptom that affects both backends (e.g.
> a CANN driver-side error, an aclnn dispatch failure), declare
> `<!-- applies_to_backend: all -->` immediately after the `##` header.
> See `OPERATIONAL_KNOWLEDGE.md` header note for full schema reference.

---

## Compile Errors

### EC-1: Missing `__aicore__` on helper function

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: calling a __host__ function("helper_func") from a __aicore__ function("kernel_vf") is not allowed
  ```
- **Root cause**: All functions called inside `__simt_vf__ __aicore__` kernel VF functions must themselves be decorated with `__aicore__`. Bisheng treats undecorated functions as `__host__`-only, and cross-domain calls are forbidden.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  inline float compute_weight(float x) { return x * 0.5f; }

  // AFTER (compiles):
  __aicore__ inline float compute_weight(float x) { return x * 0.5f; }
  ```
- **Note**: Template helper functions also need `__aicore__`:
  ```cpp
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }
  ```
- **Related**: None (basic AscendC requirement)

---
### EC-2: `GM_ADDR` needs typed pointer cast

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: cannot initialize a variable of type '__gm__ float *' with an lvalue of type 'GM_ADDR' (aka 'uint8_t * __attribute__((address_space(1)))')
  ```
  or:
  ```
  error: subscript of pointer to type '__gm__ uint8_t' ... is not allowed
  ```
- **Root cause**: `GM_ADDR` is `__gm__ uint8_t*`. Kernel VF functions receive all GM pointers as untyped `GM_ADDR`. To access data as a specific type, you must cast with `reinterpret_cast<__gm__ T*>`. The `__gm__` qualifier must be preserved through the cast.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  __gm__ float* input = input_gm;              // type mismatch
  float val = input_gm[i];                      // subscript on uint8_t*

  // AFTER (compiles):
  __gm__ float* input = reinterpret_cast<__gm__ float*>(input_gm);
  float val = input[i];                          // correct typed access

  // For const pointers:
  __gm__ const int* edge_in = reinterpret_cast<__gm__ const int*>(edge_in_gm);
  ```
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK — kernel launch pattern)

---
### EC-3: `LAUNCH_BOUND` value exceeds 512

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: 'LAUNCH_BOUND' attribute parameter 1024 exceeds maximum allowed value 512
  ```
  or at runtime: incorrect results / register spilling when LAUNCH_BOUND > 512 with complex kernel logic.
- **Root cause**: Ascend950PR supports LAUNCH_BOUND up to 2048 in theory, but **512 is the practical maximum** for kernels with non-trivial register usage. At 512 threads, each thread gets 64 registers (128KB register file / 512 threads / 4 bytes). Higher thread counts reduce per-thread registers, causing spills to slower memory and often incorrect codegen.
- **Fix**:
  ```cpp
  // BEFORE (risky or fails):
  LAUNCH_BOUND(1024) inline void kernel_vf(...) { ... }

  // AFTER (safe default):
  LAUNCH_BOUND(512) inline void kernel_vf(...) { ... }

  // Define as named constant:
  constexpr uint32_t OP_THREAD_NUM = 512;
  LAUNCH_BOUND(OP_THREAD_NUM) inline void kernel_vf(...) { ... }
  ```
- **Note**: source `__launch_bounds__(1024)` must be reduced to 512 when migrating. The dispatcher `Simt::Dim3{OP_THREAD_NUM}` must match the LAUNCH_BOUND value.
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK)

---
### EC-4: `simt_compat.h` conflicts in NPU mode

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: redefinition of 'blockDim' as different kind of symbol
  ```
  or:
  ```
  error: expected unqualified-id
  ```
  (when `#define blockDim` macro clashes with CANN's built-in `blockDim` in NPU mode)
- **Root cause**: `simt_compat.h` defines `blockDim` and `threadIdx` as macros that map to raw CPU-mode globals (`g_threadDimX`, `g_threadIdxX`). In NPU mode, CANN provides its own built-in `blockDim`/`threadIdx` — the macros collide with these built-ins. The header must only be included in CPU debug builds.
- **Fix**:
  ```cpp
  // BEFORE (fails on NPU):
  #include "simt_compat.h"    // unconditional include → macro conflicts

  // AFTER (conditional):
  #if defined(ASCENDC_CPU_DEBUG)
  #include "simt_compat.h"
  #endif
  ```
  The guard works because:
  - CPU debug mode: `ASCENDC_CPU_DEBUG` is defined by tikicpulib CMake target → macros active
  - NPU mode: `ASCENDC_CPU_DEBUG` is not defined → header skipped, CANN built-ins used
- **Related**: None (project-specific compatibility layer)

---
### EC-5: `static_cast<float>(bfloat16_t)` fails in bisheng

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: not support bf16 type cast
  ```
  or:
  ```
  error: static_cast from 'bfloat16_t' to 'float' is not allowed
  ```
- **Root cause**: Bisheng compiler (CANN 9.0.0 and 9.0.T501) does not support scalar `static_cast` between `bfloat16_t` and `float` in either direction. The `half` (fp16) type works fine with `static_cast`. This is a known bisheng limitation (PB-4 in PLATFORM_BUGS.md).
- **Fix (SIMT kernel — use bit-manipulation)**:
  ```cpp
  // BEFORE (fails):
  bfloat16_t val = input[i];
  float fval = static_cast<float>(val);    // ❌ bisheng rejects this

  // AFTER (bit-manipulation workaround):
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }

  template <>
  __aicore__ inline float simt_to_float<bfloat16_t>(bfloat16_t v) {
    uint16_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    uint32_t f32bits = static_cast<uint32_t>(bits) << 16;
    float result;
    __builtin_memcpy(&result, &f32bits, sizeof(result));
    return result;
  }

  // Reverse: float → bfloat16_t
  template <typename T>
  __aicore__ inline T simt_from_float(float v) { return static_cast<T>(v); }

  template <>
  __aicore__ inline bfloat16_t simt_from_float<bfloat16_t>(float v) {
    uint32_t f32bits;
    __builtin_memcpy(&f32bits, &v, sizeof(f32bits));
    uint16_t bits = static_cast<uint16_t>(f32bits >> 16);  // truncate
    bfloat16_t result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
  }
  ```
- **Fix (SIMD kernel — use Cast intrinsic)**:
  ```cpp
  // Cast(bf16→float) is lossless and works:
  Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);
  float w = floatBuf.GetValue(i);
  ```
- **WARNING**: `Cast(bf16→half)` is LOSSY — bf16 exponent=8bit overflows half exponent=5bit, producing `inf` for large values. Always cast bf16→float (lossless).
- **Related**: P-P27 (bf16 scalar via Cast + GetValue)

---
### EC-6: `using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: call to 'GetBlockIdx' is ambiguous
  note: candidate function: int32_t AscendC::Simt::GetBlockIdx()
  note: candidate function: int64_t GetBlockIdx()
  ```
  (typically 20+ errors across a file since every `GetBlockIdx`/`GetBlockNum` call is ambiguous)
- **Root cause**: CANN defines TWO `GetBlockIdx()` functions — `AscendC::Simt::GetBlockIdx()` returning `int32_t` and a basic API `GetBlockIdx()` returning `int64_t`. Adding `using namespace AscendC::Simt;` pulls the Simt version into the same scope as the basic API version, making every unqualified call ambiguous.
- **Fix**:
  ```cpp
  // BEFORE (ambiguous):
  using namespace AscendC;
  using namespace AscendC::Simt;   // ❌ pulls in Simt::GetBlockIdx

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // ambiguous: Simt::GetBlockIdx vs basic_api
  }

  // AFTER (unambiguous):
  using namespace AscendC;         // ✅ only basic API GetBlockIdx (int64_t)
  // No "using namespace AscendC::Simt;" — dispatchers use qualified Simt::VF_CALL

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // resolves to basic_api int64_t version
    Simt::VF_CALL<kernel_vf<T>>(   // Simt:: qualified prefix for VF_CALL
        Simt::Dim3{THREAD_NUM}, ...);
  }
  ```
- **Note**: Kernel VF functions themselves don't call `GetBlockIdx` — they receive `block_index` as a parameter from the dispatcher. Only dispatchers need `GetBlockIdx`/`GetBlockNum`.
- **Related**: OL-14 (OPERATIONAL_KNOWLEDGE.md)

---
### EC-7: `Simt::atomicAdd` — wrong namespace

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: no member named 'atomicAdd' in namespace 'AscendC::Simt'
  ```
  or:
  ```
  error: call to 'atomicAdd' is ambiguous
  ```
  (when both `Simt::atomicAdd` and global `atomicAdd` are attempted)
- **Root cause**: `atomicAdd` on AscendC is a **global built-in function**, not a member of the `AscendC::Simt` namespace. This differs from other Simt APIs like `Simt::VF_CALL`, `Simt::Dim3`, `Simt::WarpReduceAddSync` which are namespaced.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  Simt::atomicAdd(base + offset, value);       // ❌ not in Simt namespace
  AscendC::Simt::atomicAdd(base + offset, value);  // ❌ same error

  // AFTER (compiles):
  atomicAdd(base + offset, value);             // ✅ global built-in, no namespace
  ```
- **Supported types**: `float`, `half`, `bfloat16_t`, `int32_t` — all use the same unqualified `atomicAdd`.

---
### EC-8: Missing `#include <kernel_operator.h>`

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: unknown type name 'GM_ADDR'
  error: unknown type name '__gm__'
  error: use of undeclared identifier 'atomicAdd'
  error: unknown type name 'bfloat16_t'
  error: no member named 'VF_CALL' in namespace 'AscendC::Simt'
  ```
  (cascade of errors — types, macros, and functions all undefined)
- **Root cause**: `kernel_operator.h` is the master header for AscendC. It pulls in all CANN types (`GM_ADDR`, `__gm__`, `bfloat16_t`, `half`), SIMT APIs (`Simt::VF_CALL`, `Simt::Dim3`), SIMD APIs (`DataCopy`, `Cast`), atomics (`atomicAdd`), and platform macros (`LAUNCH_BOUND`, `__aicore__`). Without it, nothing AscendC-specific compiles.
- **Fix**:
  ```cpp
  // BEFORE (cascade of errors):
  #include <cstdint>
  // missing kernel_operator.h

  // AFTER (compiles):
  #include <kernel_operator.h>    // ✅ MUST be first AscendC include
  #include <cstdint>
  ```
- **Rule**: Every `.h` and `.cpp` file that uses any AscendC type or API must include `<kernel_operator.h>` as its first AscendC include. Standard library headers (`<cstdint>`, `<cstring>`) can come before or after.
- **Related**: None (basic AscendC requirement)

---
### EC-9: Missing `namespace ascendc_ops {}` wrapper

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: redefinition of 'ITER'
  error: redefinition of 'simt_to_float'
  error: use of undeclared identifier 'POOLING_FWD_THREAD_NUM'
  ```
  (name collisions between kernel files, or missing constant/helper definitions when files are compiled together)
- **Root cause**: All kernel code in this project must be wrapped in `namespace ascendc_ops { ... }`. Without the namespace: (1) macros like `ITER(x,y)` and helper templates like `simt_to_float` collide when multiple kernel headers are included in the same translation unit; (2) dispatcher `.cpp` files use `using namespace ascendc_ops;` to access kernel VF functions and constants — if the VF functions are in the global namespace, `using namespace ascendc_ops;` finds nothing.
- **Fix**:
  ```cpp
  // BEFORE (collisions, missing symbols):
  #include <kernel_operator.h>
  using namespace AscendC;

  #define ITER(x, y) (((x) + (y) - 1) / (y))

  template <typename T>
  __simt_vf__ __aicore__
  LAUNCH_BOUND(512) inline void my_kernel_vf(GM_ADDR input_gm, ...) { ... }

  // AFTER (namespaced):
  #include <kernel_operator.h>

  namespace ascendc_ops {
  using namespace AscendC;

  #define ITER(x, y) (((x) + (y) - 1) / (y))

  template <typename T>
  __simt_vf__ __aicore__
  LAUNCH_BOUND(512) inline void my_kernel_vf(GM_ADDR input_gm, ...) { ... }

  }  // namespace ascendc_ops
  ```
  Corresponding dispatcher file:
  ```cpp
  #include "my_kernel.h"
  using namespace ascendc_ops;

  extern "C" __global__ __aicore__ void my_kernel_fp32(GM_ADDR input_gm, ...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Simt::VF_CALL<my_kernel_vf<float>>(
        Simt::Dim3{512}, input_gm, ..., GetBlockIdx(), GetBlockNum());
  }
  ```
- **Note**: `extern "C" __global__` dispatcher functions are in the global namespace (required by CANN runtime). Only the VF functions, helpers, and constants go inside `namespace ascendc_ops`.
- **Related**: None (project convention for multi-file compilation)

---
### EC-10: aclrtlaunch Undefined Reference (Linker)

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `undefined reference to 'aclrtlaunch_xxx(...)'`
- **Root cause**: Auto-generated `host_stub.cpp` exports kernel launch functions as **C symbols** (no mangling). Test code that declares them without `extern "C"` gets C++ mangled names → linker mismatch.
- **Fix**:
  ```cpp
  // ❌ Wrong — C++ mangling
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);

  // ✅ Correct — C linkage
  extern "C" {
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);
  }
  ```
- **Related**: PB-8 in PLATFORM_BUGS.md

---
### EC-11: CANN Build Fails at merge_mix_obj.sh (95%)

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh`
- **Root cause**: `CMAKE_BUILD_TYPE` not set → cmake passes empty `--build-type` to `merge_mix_obj.sh` → `shift 2` fails
- **Fix**: Always pass `-DCMAKE_BUILD_TYPE=Release` to cmake
- **Related**: PB-7 in PLATFORM_BUGS.md

---
### EC-12: `block_num` / `block_index` macro collision in parameter names

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: cannot initialize a parameter of type 'int64_t (*)(void)' with an rvalue of type 'int64_t'
  note: expanded from macro 'block_num'
  #define block_num get_block_num()
  ```
- **Root cause**: CANN defines `block_num` as a macro expanding to `get_block_num()` (a function). When used as a function parameter name, `int64_t block_num` becomes `int64_t get_block_num()` -- a function declaration, not a parameter. Similarly, `block_index` may collide with other CANN macros.
- **Fix**: Rename parameters to avoid CANN macro names:
  ```cpp
  // BEFORE (fails):
  void Init(GM_ADDR x, int64_t block_index, int64_t block_num) { ... }

  // AFTER (compiles):
  void Init(GM_ADDR x, int64_t blk_idx, int64_t blk_cnt) { ... }
  ```
- **CANN macros to avoid as identifiers**: `block_num`, `block_idx`, and any other identifier in `__clang_cce_aicore_builtin_vars.h`.
- **Related**: OL-14 (namespace ambiguity)

---
### EC-13: `AscendC::SyncFunc<>` does not exist

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: no member named 'SyncFunc' in namespace 'AscendC'
  ```
- **Root cause**: There is no `AscendC::SyncFunc` API. The generated code (from templates or LLM) may invent this API for pipe synchronization. The correct API uses `SetFlag`/`WaitFlag` with event IDs fetched from `GetTPipePtr()->FetchEventID()`.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

  // AFTER (compiles):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
  ```
- **Common sync events**: MTE2_S (GM→scalar), S_MTE3 (scalar→GM write), V_S (VEC→scalar), S_V (scalar→VEC)
- **Evidence**: Cumsum V1 build failure (2026-04-09)

---
### EC-14: `TQue<..., 0>` — depth must be >= 1

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: static assertion failed: must use AllocTensor<LocalTensor&> api while tque's depth is zero
  ```
- **Root cause**: `TQue` template's second parameter is the depth (number of buffer slots). Depth 0 means "use pass-by-reference AllocTensor API" which has a completely different usage pattern. Standard AllocTensor/EnQue/DeQue/FreeTensor requires depth >= 1.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::TQue<AscendC::TPosition::VECIN, 0> xQueue_;

  // AFTER (works):
  AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_;
  ```
- **Evidence**: Cumsum V1 build failure (2026-04-09)

---
### EC-15: `PipeBarrier<PIPE_S>` not valid on Ascend950PR

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: the range of 1st parameter must be [4, 6]
  ```
  (from `kernel_reg.h`, triggered by `PipeBarrier<PIPE_S>()`)
- **Root cause**: On Ascend950PR, `pipe_barrier()` only accepts pipe values 4 (PIPE_MTE2), 5 (PIPE_V), 6 (PIPE_MTE3). The scalar pipe (PIPE_S) is not supported for PipeBarrier. To synchronize the scalar pipe, use `SetFlag`/`WaitFlag` with appropriate event types.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::PipeBarrier<PIPE_S>();

  // AFTER (S→MTE3 sync for scalar writes visible to MTE3 output):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(ev);

  // For S→V sync (scalar writes visible to VEC):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev);
  ```
- **Valid PipeBarrier pipes**: PIPE_MTE2 (4), PIPE_V (5), PIPE_MTE3 (6) — that's it.
- **Evidence**:
  - Sort V1 build failure (2026-04-09)
  - clipped_swiglu port_a3_to_a5 kw-1 (2026-05-17): scalar→vector gather pattern (interleaved A/B half extraction via `LocalTensor::SetValue(i, …)` feeding subsequent `Cast` / `Mins` / `Maxs`) tripped `kernel_reg.h:85` "range of 1st parameter must be [2,6],[10,10]" on AIC and "[4,6]" on AIV. Fixed in iter 2 by replacing `PipeBarrier<PIPE_S>()` with `SetFlag/WaitFlag<HardEvent::S_V>` at two sync sites (F32 interleaved branch + Half interleaved branch). 8/8 cases PASS post-fix. General trigger class: any kernel that scalar-gathers a strided/masked pattern (even/odd interleave, group indexing) into UB that subsequent VEC ops read — common in SwiGLU-family fused activations and small-shape Gather helpers.

---
### EC-16: DataCopy alignment overwrite in strided/chunked copies

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures on specific test cases where non-aligned chunk sizes cause data corruption in adjacent output regions
- **Root cause**: DataCopy requires 32-byte aligned element counts. When `chunk_size % ALIGN != 0`, naively aligning up writes extra elements past the chunk boundary, corrupting adjacent tensor data.
- **Fix (overlapping tail write)**:
  ```
  1. Copy floor_aligned(chunk) elements normally
  2. Copy last ALIGN elements starting at (chunk - ALIGN), overlapping with already-written region
  ```
  The overlap is harmless (same values re-written), and tail elements are placed correctly without overflow.
- **Condition**: Strided/chunked DMA with non-aligned chunk boundaries (e.g., cat along non-last dim)
- **Evidence**: Cat V1 failed 3/51 cases, fixed in V2 (2026-04-09)

---

## Quick Lookup Table

| EC | Error keyword | One-line fix |
|----|--------------|--------------|
| EC-1 | `calling a __host__ function from __aicore__` | Add `__aicore__ inline` to helper |
| EC-2 | `cannot initialize '__gm__ T*' with 'GM_ADDR'` | `reinterpret_cast<__gm__ T*>(gm_addr)` |
| EC-3 | `LAUNCH_BOUND exceeds maximum` | Reduce to 512 |
| EC-4 | `redefinition of 'blockDim'` | Guard with `#if defined(ASCENDC_CPU_DEBUG)` |
| EC-5 | `not support bf16 type cast` | Use `simt_to_float()` bit-manipulation (P-P27) |
| EC-6 | `call to 'GetBlockIdx' is ambiguous` | Remove `using namespace AscendC::Simt;` (OL-14) |
| EC-7 | `no member 'atomicAdd' in 'Simt'` | Use unqualified `atomicAdd()` (global built-in) |
| EC-8 | `unknown type name 'GM_ADDR'` | Add `#include <kernel_operator.h>` as first include |
| EC-9 | `redefinition of 'ITER'` / missing symbols | Wrap all code in `namespace ascendc_ops {}` |
| EC-10 | `undefined reference to 'aclrtlaunch_'` | Add `extern "C" {}` around declaration |
| EC-11 | `merge_mix_obj.sh Error 1` at 95% | Add `-DCMAKE_BUILD_TYPE=Release` |
| EC-12 | `cannot initialize 'int64_t (*)(void)'` + `expanded from macro 'block_num'` | Rename param: `blk_idx`/`blk_cnt` |
| EC-13 | `no member named 'SyncFunc' in namespace 'AscendC'` | Use `SetFlag`/`WaitFlag` with `FetchEventID` |
| EC-14 | `static assertion failed: must use AllocTensor...depth is zero` | Change TQue depth from 0 to ≥1 |
| EC-15 | `the range of 1st parameter must be [4, 6]` | No `PipeBarrier<PIPE_S>`, use SetFlag/WaitFlag for S pipe |
| EC-16 | Non-aligned chunk DataCopy corrupts adjacent data | Overlapping tail write: copy last ALIGN elems separately |
| EC-17 | Sub-align chunk overwrite in compact output | nblk=1 + padded alloc + narrow view |

---
### EC-17: Sub-alignment chunk overwrite in compact (tightly-packed) output

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures when chunk_size < DataCopy alignment AND output elements are tightly packed (no gaps between chunks from different outer iterations)
- **Root cause**: DataCopy writes aligned count of elements. When chunk < align, excess elements overwrite the next chunk's data. In Cat's output (strided with gaps), overlapping tail write works. In Split's output (compact), there are no gaps — adjacent chunks are immediately adjacent.
- **Fix (host-side)**: Detect `chunk < align && outer > 1`. Use `nblk=1` (serial execution — overwrites self-correct within one block) + allocate padded output + narrow to exact size.
- **Applicability**: Any kernel writing to compact output with non-aligned chunk boundaries
- **Evidence**: Split V1 failed 4/57 cases, fixed in V2 (2026-04-09)

---
### EC-18: Forward-overwrite data race in multi-block non-aligned DMA

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures in multi-block kernels where non-aligned DataCopy uses forward-overwrite technique (write ALIGN elements, let next iteration overwrite tail). When multiple blocks process different rows in parallel, the overwrites from different blocks race.
- **Root cause**: Block K-1's tail overwrite extends into Block K's write region. Without ordering, Block K may read before K-1's overwrite completes, or K's write may be overwritten by K-1's stale data.
- **Fix**: Two approaches:
  1. **Per-row overlap** (chunk >= ALIGN): re-copy last ALIGN elements from `chunk - ALIGN` offset. No cross-row overwrite. Safe for multi-block.
  2. **nblk=1 + padded alloc** (chunk < ALIGN): serialize to one block. Over-allocate output with ALIGN padding, narrow() after kernel.
- **Evidence**: Split V3 — V2 forward-overwrite caused 12 new failures, fixed with per-row overlap (2026-04-09)

---
### EC-19: PadTiling name conflict with CANN built-in

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `error: reference to 'PadTiling' is ambiguous`
- **Root cause**: CANN `kernel_tiling.h` defines `PadTiling` in `AscendC::tiling` namespace and imports it via `using`. Custom struct with same name conflicts.
- **Fix**: Rename custom tiling struct to unique name (e.g., `PadOpTiling`, `MyPadTiling`)
- **Evidence**: Pad V2 first build (2026-04-09)
### EC-20: Tiling CPU→NPU copy must happen AFTER all fields finalized

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Wrong results — tiling field has stale value on NPU
- **Root cause**: If pybind writes `tiling.field = X` after `tiling_npu = tiling_cpu.to(device)`, the NPU copy has old value
- **Fix**: Finalize ALL tiling fields, then copy once
- **Evidence**: Pad V2 mode routing bug (2026-04-09)
### EC-21: VECIN-only pipeline cannot do GM→UB→GM pass-through

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Data corruption or sync hang when doing DataCopy(UB←GM) then DataCopy(GM←UB) through VECIN queue only
- **Root cause**: VECIN syncs MTE2→VEC, but MTE3 store needs VEC→MTE3 sync. Without a VEC op and VECOUT queue, the pipeline has a sync gap.
- **Fix**: Split-queue pattern: VECIN for load + VECOUT for store + VEC identity op (Adds 0.0f) between them
- **Evidence**: Pad V2, Cat, Split all use this pattern (P-CAT-1)
### EC-22: Multi-block aligned DataCopy overwrite race

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures that disappear with nblk=1 but appear with nblk>1. Same elements fail deterministically. Mismatch ratio ~0.01-18%.
- **Root cause**: DataCopy requires aligned element counts. When `count % ALIGN != 0`, writing `ceil(count/ALIGN)*ALIGN` elements overwrites adjacent output positions. Single-block: next tile overwrites stale values. Multi-block: overwrite lands in another block's range → write-write race.
- **Fix**: Overlap-tail technique for ALL DataCopy calls with non-aligned counts. Write `floor(count/ALIGN)*ALIGN` normally, then re-write last ALIGN elements starting at `count - ALIGN`.
- **Diagnostic**: nblk=1 vs nblk=N A/B test (OL-43) — if nblk=1 passes, it's this bug.
- **Evidence**: Pad V3-V5 (2026-04-10): nblk=1 → 51/51 PASS, nblk=56 → 28/51 PASS
- **Fix approach 1 (partial)**: Row-level partitioning — ensures block boundaries at row boundaries, reducing but not eliminating races (28→30 PASS)
- **Fix approach 2 (partial)**: Pre-fill output with fill_value (torch::full) — does NOT help because overflow writes source data, not fill_value
- **Fix approach 3 (verified)**: 3-phase segment processing (fill-left → source → fill-right). Source phase overflow lands in fill-right area, immediately overwritten. Verified: case 38 (previously always FAIL) now PASS.
- **Fix approach 4 (NOT recommended)**: SafeWrite with overlap-tail `local[t-AL]` — triggers UB alignment error (error code 80). SafeWrite with scalar GetValue also triggers VEC alignment errors due to pipeline interference.
- **Generalized fix**: For any multi-block SIMD kernel doing DataCopy-to-GM with non-aligned tile counts, ensure processing order guarantees that overflow regions are overwritten by subsequent writes. 3-phase decomposition (pre-fill → source → post-fill) is the most reliable pattern.
- **V220 READ alignment evidence (ds agent 2026-05-13, op#3 Add 40/40 fix)**: DataCopy on V220 requires aligned element counts for **READS** as well as writes. Copying <8 fp32 elements from GM **reads garbage data** (no crash, just wrong values). Symptom: 1D small tensors consistently fail with large diff ~3-7. Fix: pad tile element count to SIMD boundary (`(curElems + 7) & ~7` for fp32) for all DataCopy calls (CopyIn AND CopyOut). Use actual element count for GM offset advancement (`cur += tile_elems` not `cur += aligned_elems`). Same alignment needed for fp16/bf16 (16-element boundary).
### EC-23: DataCopyPad UB→GM crashes on Ascend950PR (507035)

```yaml
applies_to:
  paradigm: ascendc
```

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: `DataCopyPad(gmTensor, localTensor, extParams)` for UB→GM direction causes 507035 vector core exception, even with properly aligned blockLen.
- **Root cause**: DataCopyPad in UB→GM direction is not supported (or buggy) on Ascend950PR. GM→UB direction works fine.
- **Fix**: Use `DataCopy` with aligned count instead. Handle non-aligned tails by pre-padding input to aligned size in the host (pybind) layer.
- **Evidence**: Sort kernel development (2026-04-14), confirmed on Ascend950PR_9589 CANN 9.0.0.
- **Cleaner mitigation (29_DynamicQuant kw-2, 2026-05-01)**: when using a tile-loop kernel with `TILE=N`, **pre-pad the input row stride to `align_up(D, N)` in the pybind layer** (pad output stride symmetrically). Every tile (including last) then uses plain `DataCopy(local, gm[r*in_stride+off], TILE)` — no DataCopyPad, no partial-count handling, no risk of EC-23. Pybind narrows back to original D after kernel via `.narrow(-1, 0, D_orig)`. Cost: a few KB of zero-padded GM per row. Benefit: kernel code becomes branchless across tile types and EC-23-immune by construction. Recommended for all tile-loop kernels touching variable D.
- **Non-crash precision-corruption variant (8_QuantScatter kw-1, 2026-05-03)**: when the writeback alignment exceeds the per-slot stride (e.g. `DataCopy<int8>(gm[i*D], ub, AlignUp(D, 32))` with `D < 32`), the **extra bytes silently corrupt the next row of the same logical tensor** — NO crash, NO 507035. Symptom signature: `mismatch_count = batches × overshoot_bytes` exact (e.g. 8 batches × 16-byte overshoot = 128/1024 mismatch on Pass B). When you see a precision Pass B failure with this exact factorization, check the writeback path's count argument vs the per-slot stride before suspecting algorithm bugs. **Same mitigation applies**: pad output GM stride to `align_up(D, 32)` in pybind, narrow back via `.narrow(-1, 0, D_orig)` before return.
- **Portability shim from CANN ops-nn (2026-05-12, CAND-A3A5-3 auto-merged via Mode 5 C37)**: cross-op evidence from `group_norm_silu_quant_base.h` lines 17-25 + `rms_norm_quant.cpp` line 245 (ReduceSum fork) shows CANN's own norm/quant kernels use a portability pattern: `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` branches use `DataCopyPad` for UB→GM (allowed on V220); the `#else` branch (V351 / arch35) falls back to `DataCopy` with caller-supplied `align_up(count, 8)` sizing. When porting any norm/quant op kernel with partial-block UB→GM writes, **reuse this shim instead of authoring a new one**. The shim is typically a small inline helper at the top of `<op>_base.h`; copy the pattern verbatim and parameterize over the dtype.
- **Cross-op confirmation that V351 DataCopyPad both directions work (group_norm_silu kw-1, 2026-05-24)**: port_a3_to_a5 V220→V351 AUTHORED-from-source cold start. Upstream V220 `group_norm_silu_base.h::IsDataCopyPadSupport()` returns `false` for `__NPU_ARCH__ == 3510` (V351/arch35) — only `220` / `3003` / `3113` are listed as supported. Empirically, DataCopyPad worked cleanly in BOTH directions on V351: GM→UB load of a 4-element fp32 (16 bytes, non-32B-aligned) AND UB→GM store of the same. Case 7 bit-exact PASS. Worker did NOT fall back to the portability shim; used DataCopyPad unconditionally. Combined with the 2026-05-18 task #22 aog-hardware-probe (UB→GM blockLen ∈ {31,33,47,63}) + OL-167 scope clarification, the upstream V220 `IsDataCopyPadSupport` guard is **stale for V351** — V351 workers SHOULD use DataCopyPad freely (per OL-167 / P-P98) and need not author / inherit the V220-style `#if __CCE_AICORE__ == 220` portability shim unless they explicitly target V220 as a co-build.
- **2nd cross-op confirmation (modulate kw-1, 2026-06-21, port_a3_to_a5 V220→arch35, A5 Ascend950PR_9579)**: a self-contained VEC affine kernel used DataCopyPad freely in BOTH directions (GM→UB load + UB→GM store) on V351 — no 507035, no portability shim — across 225/225 PASS (fp16/fp32/bf16). Adds a second port_a3 cross-op data point (after group_norm_silu) that the upstream V220 `IsDataCopyPadSupport`==false-for-3510 guard is stale for V351.
- **3rd cross-op confirmation, int64 output dtype (top_k_top_p_sample kw-4, 2026-06-24, port_a3_to_a5 V220→V351, Ascend950PR_9579)**: a sampling kernel used `DataCopyPad(outIdxGm[rowId], out, DataCopyExtParams{1, sizeof(int64_t), ...})` for a single-element **int64** (8-byte) UB→GM index write (`top_k_top_p_sample_kernel.h:324-325`) — no 507035, no portability shim — across a 16-case bit-exact PASS suite. Adds the **int64** dtype to the V351 both-directions-OK evidence base (prior confirmations: fp16/fp32/bf16 via group_norm_silu + modulate), reinforcing that the upstream V220 `IsDataCopyPadSupport`==false-for-3510 guard is stale for V351 across all common index/value dtypes.
### EC-24: SortConfig must be global constexpr for template NTTP

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `constexpr SortConfig cfg = {...}` inside a function body → "cannot use as template non-type parameter" compile error.
- **Root cause**: C++ template constraint — `const SortConfig&` NTTP requires the variable to have external linkage / global scope.
- **Fix**: Declare `SortConfig` at namespace/global scope, outside any function body.
- **Evidence**: Sort kernel development (2026-04-14).
### EC-25: Advanced Sort API is device-only (__NPU_ARCH__ guard)

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `Sort<T, isReuse, config>()` compilation fails with "SortConfig not found" or "SortType not found".
- **Root cause**: The advanced Sort API (`adv_api/sort/sort.h`) is only available when `__NPU_ARCH__` is defined during device compilation. Host build stubs don't include these declarations.
- **Fix**: Guard Sort API usage with `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ > 0)`. In host compilation, the function can be a stub or not compiled.
- **Evidence**: Sort kernel development (2026-04-14).
### EC-26: Duplicate / VEC op on non-aligned UB offset → runtime error 340

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Kernel compiles successfully but at runtime reports "UB address not aligned" / error code 340 / `aclrtSynchronizeStream` failure.
- **Trigger conditions**: `Duplicate(ubBuf[offset], 0.0f, count)` or similar VEC op, where `offset` does not meet dtype alignment:
  - fp32: `offset` must be a multiple of 8 (32B alignment)
  - fp16/bf16: `offset` must be a multiple of 16 (32B alignment)
- **Root cause**: AscendC VEC instructions require the LocalTensor's start address to be 32B aligned. If the code does a partial tile fill (`Duplicate(buf[orig_n], 0, pad_count)` to zero the tail), `orig_n` may not be aligned and triggers a hardware exception.
- **Fix options**:
  1. **Preferred (used by 29_TanhGatedResidualAddBackward)**: in the pybind layer, use `torch::zeros + .copy_()` to pre-fill the input tensor to the aligned size — the kernel no longer needs to manually zero padding (padded elements naturally produce 0 products, which do not affect reduce).
  2. Use `Duplicate(buf, 0, total_aligned_count)` to zero the entire buffer instead of just the tail.
  3. Switch to DataCopyPad for non-aligned GM↔UB transfers — but note EC-23 warns that DataCopyPad UB→GM crashes on A5.
- **Detection**: if perf tests report error code 340 and the kernel uses partial-tile / tail handling, first grep `Duplicate.*\[.*\]` to locate non-aligned start addresses.
- **Evidence**: 29_TanhGatedResidualAddBackward V1 used `Duplicate(wA[orig_in_tile], 0.0f, pad)` → error code 340. After switching to pybind pre-padding, 50/50 PASS. Worker tool count saved ~5 iterations (avoided manual debug).
  - 7_MoeGatingTopKSoftmax (2026-04-17): `Duplicate<float>(expFp32Local[N_], -INF, padCount)` where N_=5120/7168 is non-aligned → err 340. Fix: delete the padding Duplicate — since `x = -inf` input naturally produces `exp = 0`, no padding is needed. Same EC-26 reset pattern.
### EC-27: build_ascendc.py default SOC_VERSION causes 507035 on Ascend950PR

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Kernel builds OK but every run produces `aclrtSynchronizeStream failed, error code:507035` with "illegal instruction" at PC 0x80
- **Root cause**: `build_ascendc.py` defaults to `Ascend910B2` SOC if not specified. Binary is incompatible with Ascend950PR hardware.
- **Fix**: Always use `-v Ascend950PR_9589` flag. Our worker ENV already does this, but if any script omits it → instant 507035.
- **Detection**: If ALL test cases crash (not just some), and error is at very low PC offset (0x80), suspect SOC_VERSION mismatch.
- **Evidence**: 14_AdaptiveInstanceNormalization2DBackward (2026-04-16).
- **Sub-variant & casing clarification (2026-06-24, top_k_top_p_sample port_a3 kw-2)**: the Ascend950PR chip family is NOT a single SOC_VERSION string — CANN 9.0.0 `ascendc_kernel_cmake/legacy_modules/host_config.cmake` lists ~30 sibling variants under `ascend950_list` (`ascend950pr_9599`, `_958a`, `_9589`, `_958b`, `_9579`, `_957b`, `_957c`, `_957d`, plus `ascend950dt_*`), all mapping to arch `ascend950` (`opdesc_parser.py`). This refines the "Fix" above without contradicting it:
  1. **Case is normalized by cmake** — `host_config.cmake` does `string(TOLOWER "${SOC_VERSION}")` before matching the list, so `-v Ascend950PR_9589` (EC-27, build_ascendc.py PascalCase) and `ascend950pr_957b` (worker lowercase, port_a3 CANN cmake path) are BOTH accepted. Do not file PascalCase-vs-lowercase as a contradiction — cmake lowercases.
  2. **The `_95xx` sub-variant suffix is MANDATORY** — a bare family name `Ascend950PR` (or `ascend950pr`) is REJECTED (`FATAL_ERROR ... does not support`), because no `ascend950_list` entry lacks the suffix. The kw-2 note "`Ascend950PR` is rejected" = missing-suffix, NOT a casing bug.
  3. **Pick the variant that matches YOUR chip** — this A5 box (`npu-smi info -t board -i 0`: NPU Name `9579`, Chip `Ascend950PR` V100) built + ran 10/10 PASS with `ascend950pr_957b`; EC-27's anchor used `_9589`. Both are `ascend950`-arch siblings (binary-compatible for general compute). On a `507035`-style mismatch, confirm the variant via `npu-smi` and match the suffix rather than assuming one canonical string.
  - **Source**: CANN 9.0.0 install — `host_config.cmake:14` (`string(TOLOWER)` + `ascend950_list`), `opdesc_parser.py:53` (variant→arch map). Verified 2026-06-24. Resolves the regression-risk flag carried forward in two prior `top_k_top_p_sample` KB merges (they blocked `957b` vs `9589` as an EC-27 contradiction — it is not).
### EC-28: fp32 `-inf` sentinel must be `0xFF800000` (true IEEE -inf), NOT `-FLT_MAX`

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: All-pass on kernel side but verifier reports `max_abs_diff=3.40282e+38` on positions that were "masked/filtered" in both kernel and reference.
- **Root cause**: Kernel uses `constexpr float NEG_INF = -3.40282e38f` (i.e. `-FLT_MAX`) as sentinel for masked positions, while PyTorch reference's `masked_fill_(-float("inf"))` writes true IEEE -inf (`0xFF800000`). These are **different floats**: `-FLT_MAX = 0xFF7FFFFF ≠ 0xFF800000`. Verifier compares element-wise → counts them unequal even though semantically both say "filtered".
- **Fix**: Use a bit-cast helper:
  ```cpp
  __aicore__ inline float GetNegInfF32() {
      uint32_t bits = 0xFF800000u;
      float f;
      __builtin_memcpy(&f, &bits, sizeof(float));
      return f;
  }
  ```
  Then `Duplicate<float>(buf, GetNegInfF32(), len)`. For **bf16/fp16, also use explicit bit-pattern injection** — `static_cast<half|bfloat16_t>(fp32 -inf)` is NOT reliable on bisheng (see 2026-04-30 follow-up below):
  ```cpp
  if constexpr (std::is_same_v<T, half>) {
      uint16_t bits = 0xFC00u;  // fp16 -inf
      NEG_INF = *reinterpret_cast<half*>(&bits);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      uint16_t bits = 0xFF80u;  // bf16 -inf
      NEG_INF = *reinterpret_cast<bfloat16_t*>(&bits);
  }
  ```
- **Detection**: precision FAIL with `max_abs_diff ≈ 3.4e38` and mismatch positions all in "filtered" regions (e.g. positions that should be -inf per top-k/top-p mask). Compare ref output bit pattern vs kernel output at one mismatched position — if ref is `0xFF800000` and kernel is `0xFF7FFFFF`, this is EC-28.
- **Evidence**: 9_TopKTopP V2 iter 2 (2026-04-17). Worker had `NEG_INF_F32 = -FLT_MAX`, all fp32 N > 8192 cases showed `max_abs_diff=3.4e38` on every masked position. Fixed by bit-cast helper → fp32 cases went 0 → 17/17.
- **Evidence (2026-04-30 follow-up)**: 9_TopKTopP a3 cold-start (`topktopp-kw-1/kw-2`). Original "round-trip cleanly" KB note had it that `static_cast<T>(-__builtin_huge_valf())` would suffice for fp16/bf16 — actually 24/24 fp16+bf16 cases failed with same `max_abs_diff=3.4e38, mean_abs_diff=inf` symptom as the fp32 case. Explicit bit-pattern injection fixed the bit-pattern mismatch (22→23 PASS). KB updated above to require explicit bits for fp16/bf16 too. (Note: applied fix did NOT close all 28 remaining failures — separate shape-specific algorithm bug — see knowledge_update.md; original EC-28 sentinel-bit-pattern issue is now fully addressed.)
### EC-29: `SortConfig` device-side is 2-field, not 4-field (host tiling header mismatches)

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Kernel compile error `excess elements in struct initializer` on line declaring `constexpr SortConfig SORT_CFG = {...}`.
- **Root cause**: The hardware `Sort` API has **two** `SortConfig` definitions:
  - **Host-side tiling header** `sort_tiling_intf.h`: 4 fields `{type, isDescend, hasSrcIndex, hasDstIndex}`.
  - **Device-side impl** `sort_impl.h`: 2 fields `{type, isDescend}` — source/destination index options are fixed by overload choice, not config field.
- **Fix**: For device-side (the kernel), use 2-field initializer only: `constexpr SortConfig CFG = {SortType::RADIX_SORT, true};`. Use the simpler Sort overload `Sort<T, isReuse, cfg>(dst, dstIdx, src, tmp, count)` which auto-assigns default indices.
- **Detection**: First-line-of-kernel compile error, no other errors preceding.
- **Evidence**: 9_TopKTopP V2 iter 2 Phase C (2026-04-17). 4-field initializer copied from CANN host-side sample → device compile fails.
### EC-30: `deploy_to_a5.sh` syncs `current_task/` not `workspace/<op>/kernel/`

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Worker edits `workspace/<op>/kernel/topktopp_kernel.h`, runs `bash src/scripts/deploy_to_a5.sh --build`, build succeeds, precision result is **unchanged** from previous iteration despite the edit. Verification reports same FAIL signature as before the change.
- **Root cause**: `deploy_to_a5.sh` syncs `~/workspace/AscendOpGenAgent/current_task/` to A5 container for build. It does **not** read from `workspace/<op>/`. So edits to `workspace/<op>/kernel/*.h` are invisible to the build.
- **Fix**: Worker must manually copy before each build:
  ```bash
  cp $LOCAL_PROJECT/workspace/<op>/kernel/* \
     ${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task}/kernel/
  bash src/scripts/deploy_to_npu.sh --build
  ```
  Or fix the deploy script to accept `ASCENDC_WORKSPACE` env var and sync from that. Until fixed, worker/probe must always pre-copy.
- **Detection**: Verification run produces identical results across two consecutive iters despite kernel edits. Confirm by `md5sum workspace/<op>/kernel/topktopp_kernel.h` vs `md5sum <current_task on A5>/kernel/topktopp_kernel.h` — different md5 = phantom build.
- **Evidence**:
  - 9_TopKTopP V2 iter 1 (2026-04-17). Worker rewrote kernel, built, got identical 34/50 FAIL as v1 — phantom 30 minutes debugging before noticing A5 current_task/ had stale v1 kernel.
  - 9_TopKTopP cold-run probe (2026-04-18). Probe explicitly ran `cp workspace/topktopp_v31/kernel/* current_task/kernel/` before every `deploy_to_a5.sh --build` call — confirmed the workaround works and is load-bearing. Spawned inside aog-kernel-worker + aog-precision-probe specs as a standing requirement.
### EC-31: `Select(dst, mask, src0, src1)` mask polarity — mask bit=1 → src0 (NOT src1)

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Kernel precision all-fail or systemic mismatch. Worker wrote a "drop-mask" (e.g. `Compares(mask, v, threshold, LT)` → mask=1 where value < threshold, i.e. where position should be DROPPED), then `Select(dst, mask, workBuf, NEG_INF)`, assuming mask=1 means "use workBuf (keep)" and mask=0 means "use NEG_INF (drop)". Actual behavior inverts this: mask=1 picks src0 (workBuf) for the drop positions → keeps what should be dropped and vice versa.
- **Root cause**: `Select` semantics are documented as "mask bit=1 → pick src0, mask bit=0 → pick src1". Worker wrote mask as "drop-bit is 1" but treated Select as "keep-bit is 1". Semantic mismatch.
- **Fix**: ALWAYS build a positive **keep-mask**: the mask bit is 1 where the element should be KEPT (i.e. written to dst as src0), and 0 where it should be filled with sentinel (src1). For "keep v >= threshold", use `Compares(..., GE)` not `LT`. For complex conditions with ties, construct: `keep = (v >= threshold) AND (v > cutoff OR (v == cutoff AND idx > cutoff_idx))` using `Compares` + `And` + `Or`. Then `Select(dst, keep_mask, v, NEG_INF)`.
- **Alternative (safer)**: Use scalar `SetValue` per kept position in an iteration — no mask polarity ambiguity. Slower but bug-resistant. Prefer this for complex per-column emit logic until the mask-construction approach is well-tested.
- **Detection**: If a kernel's "Phase 4 emit" or similar produces 0/N PASS with all-positions-flipped signature (kept and dropped positions swapped), check mask polarity first. This is distinguishable from EC-28 (-inf sentinel value) because the count of mismatched positions is typically 50% of kept or 50% of all — not the "few boundary positions" signature of tie-break bugs.
- **Evidence**: 9_TopKTopP cold-run Phase D iter 1 (2026-04-18). Worker had `mask = (v < threshold)` + `Select(dst, mask, workBuf, -inf)` intending mask=1 = drop → selected workBuf for drops, -inf for keeps. Fix: rebuilt as keep-mask. iter went 0/50 → closed multi-bug chain.
### EC-32: Top-K buffer `effective_kept` vs `buffer_len` — always compute effective_kept separately

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Kernel with oversized top-K buffer (e.g. TOPK_CAP > actual row's count of v >= threshold) produces wrong cutsum cumulation, cutoff decision lands far from threshold, precision all-fail with systemic offset.
- **Root cause**: Worker walks cumsum / reduce over the full TOPK_CAP-sized buffer assuming all positions contain valid data. When the row has fewer than TOPK_CAP values ≥ threshold (e.g. row with small N, or row where actual kept count < buffer cap), the walk processes uninitialized / padding positions, inflating the cum and preventing cutoff from triggering.
- **Fix**: After any top-K merge / compaction step, compute `effective_kept` separately:
  ```cpp
  int32_t effective_kept = 0;
  for (int32_t i = 0; i < TOPK_CAP; i++) {
    if (top_val[i] >= threshold) effective_kept++;
  }
  ```
  Then walk `[0 .. effective_kept - 1]` (or the equivalent ASC range), NOT `[0 .. TOPK_CAP - 1]`. Padding positions between effective_kept and TOPK_CAP must be held at a sentinel (e.g. -inf) that guarantees they would be rejected if processed — belt-and-suspenders.
- **Detection**: Small-N cases fail systemically while large-N cases (where effective_kept ≈ TOPK_CAP) pass. Or: cumsum values reach 1.0 far from the expected cutoff rank.
- **Evidence**: 9_TopKTopP cold-run Phase D iter 2 (2026-04-18). Worker walked from `topk_len - 1` down instead of `effective_kept - 1`; for rows with < TOPK_CAP kept values, iterated over non-kept pos → cum inflated → cutoff never set. Fix: explicit `effective_kept` scan, walk that range.
- **Related**: P-P59 (tied-threshold buffer truncation) — P-P59 assumes `effective_kept ≤ TOPK_CAP`; this EC is about the distinct implementation concern that `effective_kept` may be `< TOPK_CAP` for small rows.
### EC-33: Advanced `Sort` 6-arg overload with `srcIndexTensor` triggers aivec 343 on Ascend950PR

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Runtime `aicore exception 507015 / errcode 343 / "Incorrectly sorted data entered by the VMS"` at the MrgSort step inside hardware radix sort, when invoked via the 6-argument `Sort` overload: `Sort<T, U, isReuse, CFG>(dstV, dstI, srcV, srcI, tmp, count)` — even when `srcIndexTensor` is correctly populated with increasing values `[0, 1, ..., count-1]`.
- **Root cause**: Unconfirmed, but reproducible on CANN 9.0.0 / Ascend950PR_9589. Likely a CANN-internal behavior of the srcIndex path in radix sort on this SOC. Same error signature (`aivec 343 "Incorrectly sorted data entered by the VMS"`) appears when `torch_npu.npu_top_k_top_p` runs under sustained calls — suggests a shared CANN-side issue with the radix sort VMS module under certain input patterns, independent of caller.
- **Fix**: Use the 5-argument overload `Sort<T, isReuse, CFG>(dstV, dstI, srcV, tmp, count)`, which auto-generates local indices `0..count-1` and avoids the srcIndex path. If you need global/external indices (e.g. chunk offsets), add the offset manually when reading `dstI` after the call:
  ```cpp
  // Chunked merge example: chunk starts at col_offset
  Sort<float, false, SORT_CFG_DESC>(svOut, siOut, svIn, stmp, count);
  for (int32_t i = 0; i < count; i++) {
      global_idx[i] = siOut.GetValue(i) + col_offset;  // manual offset
  }
  ```
- **Detection**: If you call the 6-arg Sort overload and hit `aivec 343`, switch to the 5-arg variant first before any other debugging.
- **Evidence**: 9_TopKTopP cold-run round 2, Phase C iter 2 (2026-04-18). 6-arg Sort with caller-provided srcIndex → 507015/343. Switching to 5-arg + external offset → OK. Same error signature also observed in CANN reference path (`torch_npu.npu_top_k_top_p` under performance.py sustained calls) on the same SOC — suggests shared internal cause.
- **Additional trigger (2026-04-18 V3.2 9_TopKTopP test 2)**: Even with the 5-arg Sort overload (no srcIndex) and 2-field `SortConfig`, using `SortType::RADIX_SORT` can still trigger VMS 343 at runtime on Ascend950PR under certain input patterns (exact conditions unconfirmed; may correlate with chunk count > 1 or specific value distributions). Switching to `SortType::MERGE_SORT` resolved the runtime crash in one case, and precision stayed at 50/50 PASS.
- **Mitigation — prefer MERGE_SORT by default**: Use `constexpr SortConfig CFG = {SortType::MERGE_SORT, isDescend}` as the default choice on Ascend950PR CANN 9.0.0. Only switch to `SortType::RADIX_SORT` if perf profiling demonstrates a significant improvement AND you can confirm no VMS 343 on representative inputs.
- **Status**: Reproducible on current session's A5 state; may be specific to CANN 9.0.0 + Ascend950PR_9589. Confirmation across different NPU state / CANN version desirable.
- **Benchmarking methodology for affected ops (2026-04-19 calibrated on 9_TopKTopP)**: When the reference itself (e.g. `torch_npu.npu_top_k_top_p`) is implicated in EC-33 VMS 343 crashes, the standard `utils/performance.py current_task all` (default warmup=5 repeat=10 = 750 ref calls for 50 cases) will reliably crash. Empirical threshold map on Ascend950PR CANN 9.0.0:

  | warmup × repeat | Total ref calls (50 cases) | Behavior |
  |-----------------|---------------------------|----------|
  | 0 × 1 | 50 | Never crashes; BUT cold-launch overhead dominates → inflated ratios on small-N cases (launch overhead ≈ compute time); not representative of warm production perf |
  | **1 × 2** | **150** | **Recommended default** — never crashed in 3/3 runs; warm kernel measurement; ratio represents steady-state |
  | 3 × 3 | 300 | Usually OK (1/1 observed) |
  | 5 × 5 | 500 | **Flaky** — 2/3 runs crashed in one experiment |
  | 5 × 10 (default) | 750 | Always crashes |

  **Methodology rules when hitting EC-33-affected ops**:
  1. **Default to `warmup=1 repeat=2` × 3 runs**, take median of per-run sum/median/geomean ratios. This gives warm-kernel numbers without tripping VMS 343.
  2. **Do NOT use `warmup=0 repeat=1`** as a primary measurement. Cold-launch overhead on small-N cases inflates the ratio artifact; numbers are not comparable across sessions. Acceptable only as a fallback when `warmup=1 repeat=2` also crashes.
  3. Worst-case: drop to `warmup=0 repeat=3` (150 calls but all cold) if ref is unusually fragile. Still warm-vs-warm between impls.
  4. Script ratio computation: `src/scripts/perf_ab.py` (promoted from /tmp) reads `performance.py` output files and emits sum/median/geomean ratios + distribution buckets.

  **Illustrative (9_TopKTopP R3b snapshot, 2026-04-19)**:
  - `warmup=0 repeat=1` (50 calls): sum 0.475x (cold artifact — inflated)
  - `warmup=1 repeat=2` × 3 runs (warm median): sum 0.222x (honest number)

  **Cross-session reinforcement (op#9 9_TopKTopP kw-1 + ko-1, 2026-05-02 Ascend950PR_9579)**:
  - `utils/performance.py` 50-case sweep with default warmup=5 repeat=10 → CANN reference hits VMS 343 around case 30–40 in BOTH kw-1 and ko-1 sessions on this SOC, independent of the test kernel; behavior consistent with the kw-1 measurement n=40/50 and the ko-1 measurement n=31/50 before the reference crashed
  - `warmup=1 repeat=2` wall-clock measurement completes for n=31–40 of 50 cases reliably; this remains the recommended methodology for any op whose reference is `torch_npu.npu_top_k_top_p` or another EC-33-affected fused op on Ascend950PR
  - Profiler-based timing (`utils/performance.py`) cannot complete because the profiler's overhead amplifies CANN's sustained-call instability — confirmed across 2 sessions
  - **op#9 9_TopKTopP pp-3 (2026-05-03 Ascend950PR_9579) — 4th data point**: pp-3 ran 5 sequential calls into `torch_npu.npu_top_k_top_p` (cases 8, 17, 26, 35, 44 via `Model().forward()` in a single Python session, NO profiler attached). All 5 calls completed clean. **Corroborates pp-2's "single-call or multi-call without profiler is safe" refinement** (line 685 above). EC-33 trigger appears to require profiler-induced sustained call patterns specifically, not just call count alone. Practical implication: Pass-B style harnesses that loop `Model().forward()` directly without msprof attachment can run the full 50-case sweep on EC-33-affected ops without VMS 343.

  **kw-3 RADIX_SORT retry — 2nd-data-point evidence narrows the trigger (op#9 9_TopKTopP, 2026-05-03 Ascend950PR_9579 + CANN 9.0.0 b103)**:
  - Switched `SortType::MERGE_SORT` → `SortType::RADIX_SORT` (1-line constexpr swap), re-ran 50 Pass B + 50 determinism cases. **VMS 343 did NOT trigger.** RADIX completed all 100 cases without `errcode 343 / aicore exception 507015 / "Incorrectly sorted data entered by the VMS"`.
  - Kernel uses CHUNK_LEN=2048 with up to 32 chunks per row for bf16 N=65536 — well above the originally hypothesized "chunk count > 1" trigger threshold, so the original threshold-correlation hypothesis is weakened.
  - **Perf cost of MERGE vs RADIX on op#9 is ~0% (within noise) on this SOC**, NOT the previously suspected ~38%. Median 0.385× (RADIX) vs 0.388× (MERGE) across n=49 cases. The historical R3b 0.610× archive number was likely a different harness shape mix or CANN sub-version.
  - **Implications**:
    - Trigger is narrower than originally documented — likely correlated with specific input distributions or specific CANN VMS-module state, not RADIX_SORT-on-Ascend950PR in general.
    - **MERGE_SORT remains the safer default** (defensive posture is cheap); op-level retests of RADIX_SORT under representative inputs are reasonable when perf profiling identifies a benefit.
    - When testing RADIX_SORT for a new op, a 50-case Pass B sweep + a determinism check is sufficient to confirm no VMS 343 on the chosen config.

  **Wall-clock truncation variance methodology refinement (op#9 kw-3, 2026-05-03)**:
  - Three back-to-back warmup=1 repeat=2 runs in one container session produced n=22, 40, 49 cases completed before VMS 343 — i.e. the truncation point itself is non-deterministic and depends on NPU sustained-call state.
  - **Don't average median ratios across runs** — each run is computed over a different shape subset; values are not comparable.
  - **Report the run with the largest `n`** as the representative measurement (covers the most shape diversity).
  - **Don't claim 0.6× threshold pass/fail off a single run** — re-baseline at session start, run A and B kernels in same session order with NPU re-init between, and use n=max for each.
  - Cross-session sanity: across 4 sessions of op#9 (kw-1 / ko-1 / kw-2 / kw-3), median ratios all fell in 0.27–0.42× — that range IS the structural ceiling, robust to the truncation-point variance.
  - Gap: 2.1x. The warm number is the correct one for product reporting.
### EC-34: `const LocalTensor<T>` variable blocks Sort / write-through VEC API overload resolution

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
### EC-35: AIV device binary does not link libm transcendentals — split SIMT + SIMD for per-element trig/exp/log

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**:
  ```
  ld.lld: error: undefined symbol: cosf
  >>> referenced by <kernel>.cpp:<line>
  >>>    .../device_aiv.o:(...<kernel>_vf..._simt_entry)
  ld.lld: error: undefined symbol: sinf
  ...
  ```
  Appears at the AIV device-object link step (`ld.lld -m aicorelinux`), NOT at compile. Applies identically to `std::cos` / `std::sin` / `__builtin_cosf` / `__builtin_sinf` / `expf` / `logf` / `sqrtf` / `tanf` / `powf` — all resolve to libm names at link.
- **Root cause**: AIV-only device binaries (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`) are linked against a minimal runtime WITHOUT libm. Per-element scalar transcendentals are unavailable inside `__simt_vf__` functions. The official AscendC API catalog lists transcendentals in §9.1 "Math Library" (数学计算库) as SIMD APIs only (`LocalTensor<T>` + `sharedTmpBuffer` operands) — no public SIMT scalar equivalent.
- **Fix (verified in op#28 MultimodalRopePositionComputationWithGridBasedIndexing)**: Split the kernel into two launches:
  1. **SIMT kernel** materializes the transcendental's *input* into a dedicated GM buffer (e.g., `emb[total_tokens, head_dim]` fp32).
  2. **SIMD kernel** reads that GM buffer with `DataCopy` into `TQue<VECIN, 2>`, applies AscendC's `Cos()` / `Sin()` / `Exp()` / `Log()` high-level API on `LocalTensor<T>`, writes via `DataCopy` to `TQue<VECOUT, 2>`.

  Pair outputs (e.g. Cos + Sin on same input) in one SIMD pass to share MTE2 load:
  ```cpp
  LocalTensor<float> x = inQ_.DeQue<float>();
  LocalTensor<float> cosT = cosQ_.AllocTensor<float>();
  LocalTensor<float> sinT = sinQ_.AllocTensor<float>();
  LocalTensor<uint8_t> tmp = tmpBuf_.Get<uint8_t>();
  Cos<float, false>(cosT, x, tmp, count);
  PipeBarrier<PIPE_V>();
  Sin<float, false>(sinT, x, tmp, count);
  PipeBarrier<PIPE_V>();
  ```
  Cost: one extra GM round-trip (SIMT → GM → SIMD read-back). Usually cheap — Cos/Sin polynomial eval is compute-bound, not MTE2-bound.
- **Detection**: grep `ld.lld` output for `undefined symbol: <trig/exp/log>f`. Check if a `__simt_vf__` function references any of these in scalar form.
- **Prevention (Phase B checklist)**: For ops with per-element trig / exp / log (RoPE, sinusoidal PE, softmax with scalar `exp`, gaussian activation, softcap, etc.): plan SIMT + SIMD split in Phase A, do NOT attempt pure-SIMT.
- **Related**: Structural AIV constraint, not a bug — AscendC API catalog §9.1 already says transcendentals are SIMD-only. This entry formalizes the Phase A / Phase C takeaway for ops that miss the catalog lookup.
- **Evidence**: op#28 Phase C iter 2 (2026-04-22). First version used `__builtin_cosf(f)` / `__builtin_sinf(f)` in `__simt_vf__`; compile OK, link fail at `ld.lld`. Split into `mrope_build_emb_vf` (SIMT → fp32 GM) + `MropeCosSinApply` (SIMD `Cos<float>()` + `Sin<float>()`). Link OK, precision 50/50, perf 10.3x sum.
- **Status**: OPEN (structural). Not a candidate for CANN fix.
- **Distinct path — do not over-generalize "transcendentals don't link in AIV"** (added 2026-06-22): this entry is the AIV-VECTOR device-object link path (`KERNEL_TYPE_AIV_ONLY` → libm names undefined at `ld.lld`). The SEPARATE SIMT-SCALAR path DOES have working transcendentals: `expf`/`logf`/`log1pf` are declared `__simt_callee__` in `simt_api/math_functions.h` and link/run inside a VF callee (see OL-242). So "split SIMT+SIMD for transcendentals" (this EC, vector path) and "call the SIMT scalar intrinsic directly" (OL-242, scalar path) coexist — identify which path your callee is on before choosing. Cross-ref OL-242.
### EC-37: K2 workspace must be pre-zeroed when K1 cores may skip their slot (fused scatter+reduce)

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: In a multi-kernel pipeline, K1 writes per-core local accumulations into `ws[nblk, H]`, and K2 sums along the nblk dimension. If K1 partitions by rows (or any axis with fewer logical units than nblk), some cores have `my_rows_=0` and `return` directly — their ws slices are never written, so K2 sum reads garbage.
- **Trigger conditions**:
  - Output shape `[nblk, H]` (per-core exclusive slot)
  - Logical work units `logical_work_units < nblk` (e.g., when BS < 56, rows do not fill all 56 cores)
  - K1 has `if (my_rows_ == 0) return;` fast-path
  - workspace comes from `torch::empty(...)` or similar uninitialised allocation
- **Root cause**: "per-core-exclusive workspace" does NOT imply "every core writes". Adjacent trap to OL-66 (torch::zeros not stream-ordered): OL-66 is about host-side zeros not being synchronised; this one is about device-side zeroing never happening at all.
- **Fix (3 options)**:
  1. **Preferred**: pybind calls `aclrtlaunch_memzero(ws_ptr, nblk*H*sizeof(T))` before K1, stream-ordered (automatically sequenced with K1/K2 on the same stream). Cost: one 56-core memzero launch (< 0.1 ms for 1 MB ws).
  2. K1 path guarantees every core writes (even if my_rows_=0, still Duplicate 0 into the core's own ws slot). Cost: every K1 variant must remember to handle this.
  3. Use `torch::zeros` instead of `torch::empty` (carries the OL-66 risk — host zeros ≠ stream-ordered, may overlap with K1).
- **Detection signal**: pass/fail flips across the N/nblk boundary — BS ≤ 32 PASS (some benchmark), BS ∈ (32, 2048) some cores idle → FAIL, BS = 2048 PASS again (all cores share work evenly). Failing case's output `[i][1]` (K2 product) has large-magnitude garbage (max_abs_diff ≥ 100), while `[i][0]` (K1 direct product) is completely correct.
- **Prevention (Phase B checklist)**: List every kernel's workspace; for each workspace, answer "do all cores in K1 write every slot? If not, who zeroes it before K2 reads?"
- **Evidence**: op#17 EmbeddingWithInitialLayernormBackward Phase D iter 1 (2026-04-23). `gnw_workspace[56, H]`, K1 partitioned by rows; when BS < 56, several cores skipped with my_rows_=0 → K2 sum read garbage → 30/57 FAIL. Fix: pybind added `K4_memzero(gnw_ws)` launch before K1 → 57/57 PASS.
- **Related**: OL-66 (torch::zeros not stream-ordered) — different failure mode same root theme.
### EC-36: `Cast<T, T>` same-dtype is a no-op — destination stays uninitialised (silent data corruption)

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: No compile error. At runtime, `Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, count)` where `dstLocal` and `srcLocal` have the SAME dtype T produces no hardware instruction. `dstLocal` retains whatever was in that UB region from prior kernel state (garbage / zeros / stale prior-tile data). Downstream ops using `dstLocal` silently use uninitialised values.
- **Root cause**: Cast's codegen optimises same-dtype casts to nothing (compiler assumes no conversion needed). Valid for in-place casts; catastrophic when the intent was to copy from src to dst.
- **Fix (two options)**:
  1. **Per-dtype Compute specialisation**: template-specialise so the fp32 path skips the Cast entirely and operates directly on the dequeued input tensor.
  2. **Explicit copy via arithmetic no-op**: `Adds(dst, src, 0.0f, count);` forces a real data movement. Or `Muls(dst, src, 1.0f, count);`.
- **Detection signal**: kernel produces output that looks like "some operands missing" — e.g., `out ≈ a + 0 · b` or `out ≈ b` when formula should be `out = c*a - d*b`. Hint: if tail of your compute chain uses a buffer that was "copied via Cast<T,T>", suspect this.
- **Prevention (Phase B checklist)**: any `Cast<T1, T2>` where T1 == T2 — audit. If intent is "copy from input to scratch", use `Adds(..., 0.0f, ...)` or avoid the scratch entirely.
- **Evidence**: op#16 Batched2DRopePositionEncodingBackward Phase D iter 2 (2026-04-22). fp32 path `Cast(gcF, gc, CAST_NONE, count)` where both are fp32 left `gcF` uninitialised → downstream `Mul(prodA, gcF, sinF)` produced `prodA ≈ 0` → `out = prodB - prodA = gs*cos(θ)` consistently (missing the `-gc*sin` term). Fix: switched fp32 path to skip Cast, operate on dequeued input directly. 50/50 PASS after.
- **Related**: P-P52 fp32 promotion — when promoting bf16/fp16 → fp32, the Cast IS needed (different dtype). Trap is only when dtypes happen to match.
### EC-38: ASCENDC_API_CATALOG.md miss ≠ API doesn't exist — must `ls` adv_api/ headers before falling back to manual decomposition

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Worker greps `ASCENDC_API_CATALOG.md` for an advanced API name (e.g. `Sigmoid` / `Silu` / `Swish` / `Tanh`) per OL-80 "API existence check". Catalog returns 0 hits → worker concludes "API not available on this chip". Worker falls back to manual decomposition (`Exp + Reciprocal + Mul` for sigmoid, hand-rolled polynomial for tanh). Resulting kernel: ULP-divergence from CANN reference, Pass A FAIL with `max_abs_diff ≈ 4e-7..1e-3` (polynomial-difference signature, not 1-ULP rounding drift).
- **Root cause**: `ASCENDC_API_CATALOG.md` is a HUMAN-MAINTAINED summary, not an auto-generated index — it lags behind real CANN releases. Many advanced API headers exist on the chip but are not (yet) listed in the catalog. The headers ARE present at `cann-{version}/aarch64-linux/asc/include/adv_api/<name>/kernel_operator_<name>_intf.h`. OL-80 grep is a cheap first-pass check, NOT a definitive existence check.
- **Fix (worker Phase A — MANDATORY when catalog grep miss)**: Before falling back to manual decomposition, run:
  ```bash
  # via /a3_op or /a5_op skill on the active container
  ls /usr/local/Ascend/cann-{ver}/aarch64-linux/asc/include/adv_api/ 2>/dev/null
  find /usr/local/Ascend/cann-{ver} -name "kernel_operator_<api>_intf.h" 2>/dev/null
  ```
  If the header exists → use the advanced API; expect bit-exact match against CANN reference (A-P35 advanced API regime). If the header genuinely doesn't exist → manual decomposition is the right path AND A-P35 contract softening (Pass B 1e-3 tolerance) applies.
- **Detection signal**: precision FAIL with residuals 1-ULP to 1e-3 in transcendental ops + worker's analysis.md cites OL-80 catalog grep but did NOT cite an `ls adv_api/` step — suspect catalog miss → manual decomposition trap.
- **Prevention (Phase A checklist addition)**: For every ascendc primitive used in the kernel, the analysis.md must list one of: (a) catalog §section it came from (existing OL-80 check), OR (b) actual `ls adv_api/<name>/` output showing the intf header.
- **Evidence**:
  - 2026-04-28 op#11 DequantSwigluQuant a3 cold-start: a different agent's worker grep'd catalog for `Sigmoid` → 0 hits → fell back to manual `Exp + Reciprocal + Mul` for silu → Pass A drifted from CANN reference. Fix: switched to advanced `Sigmoid()` API (header at `adv_api/sigmoid/kernel_operator_sigmoid_intf.h`) → bit-exact (a5 sibling op#11 archive kernel.h:309/328 is the precedent that surfaced the catalog gap).
  - General: 50+ ops over the project that touched activation / softmax / matmul historically had this trap; catalog gradually accreted §9.1 entries as ops surfaced them.
- **Related**: A-P35 (advanced API regime) — EC-38 is the discovery step that determines which A-P35 regime applies. OL-80 (API existence check) — EC-38 is the second-stage check after OL-80 grep miss. OL-91 / aog-self-critic C23 (bar-lowering verdicts without artifact evidence) — declaring "API doesn't exist" without `ls` is a C23 bar-lowering verdict labeled as authoritative narrative.
### EC-39: Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields

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
### EC-40: Cube tiling host POD size mismatch — `TCubeTiling` is 50 int32 (200 B), not 51

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom (compile error)**:
  ```
  batchmatmul_tiling.h:80:37: error: static assertion failed:
      <name>TilingHost size must match TCubeTiling
  ```
- **Trigger**: a host-side POD that mirrors `TCubeTiling` (so pybind can `torch::empty + .copy_()` the tiling into NPU memory) is declared with the wrong field count — typically estimated as 51 from incomplete docs, but the actual count for CANN 9.0.0 is **50 int32 fields = 200 bytes**.
- **Fix**: Inspect `kernel_tiling.h` `struct TCubeTiling` field-by-field and count exactly. For CANN 9.0.0: 50 int32 fields. Mirror with `static_assert(sizeof(TilingHost) == 50 * sizeof(int32_t))`. **Recount whenever CANN version changes** — this is brittle to upstream additions.
- **Better long-term fix (Opt2 path, OL-91 step 3)**: Eliminate the host POD entirely by using on-stack `TCubeTiling tiling{}` filled in the kernel from scalar args + the non-`__gm__` `Init(const TCubeTiling*, TPipe*)` overload. No host H2D, no size-match concern, ~5–10 µs faster per call.
- **Evidence**: 1_BatchMatmul (2026-04-28) Phase C iter 2. Once op#1 documented the exact count, op#4 / op#5 / op#3 all built clean on first try.
### EC-41: 32B-aligned `DataCopy(GM, ub, count)` to under-allocated `torch::empty({C}, ...)` overflows adjacent torch tensors

```yaml
applies_to:
  paradigm: ascendc
```

- **Symptom (silent precision corruption + occasional crash)**:
  ```
  random Pass-A failures across cases that share output ordering;
  Python finalize: "double free or corruption"
  ```
- **Root cause**: `DataCopy(gm, ub, count)` minimum write granularity is **32 B = 8 fp32 elements**. When pybind allocates a small output buffer via `torch::empty({C}, opts_f32)` with `C < 8`, the 32 B store overruns into adjacent torch-allocator slots. The OOB write often clobbers another tensor's first cache line, producing data-dependent precision drift in unrelated outputs and triggering allocator integrity checks at process exit. Generalizes the kernel-side rule (PB-9 / DataCopy 32 B granularity) to the **host allocation side** — the host buffer must be padded to ≥ 8 elements regardless of how many the kernel "logically" writes.
- **Fix (host side)**:
  ```cpp
  // BEFORE (fails — torch::empty({C}, ...) lays out only C * sizeof(T) bytes):
  auto out_small = torch::empty({C}, opts_f32);
  kernelDataCopy(gm_ptr, ub, /*count=*/C);  // 32 B store overruns

  // AFTER (host buffer padded to 8-element boundary):
  const int64_t C_pad = (C + 7) & ~7LL;     // RoundUp64(C, 8)
  auto out_small = torch::empty({C_pad}, opts_f32);
  // (Kernel still emits 32 B; the pad absorbs it. Caller slices [..., :C].)
  ```
  And on the kernel side: `gmBuf.SetGlobalBuffer(ptr, C_pad)` so the bounds check sees the padded region.
- **Symptoms it explains**:
  - "Tests pass in isolation, fail in batch" — order-of-allocation matters for which adjacent tensor gets clobbered.
  - "Adding an unrelated print fixes it" — the print's allocations shift the heap layout enough to make the OOB land in unused space.
  - "Works on smaller cases, fails on larger" — case_gen iterates output ordering; large counts happen to clobber a downstream tensor.
- **Anti-pattern**: round the **count** parameter of DataCopy up instead of the host allocation. The count round-up writes garbage into `[C, C_pad)` which the host buffer doesn't own. Always pad the **buffer**.
- **Related**: PB-9 (kernel-side 32 B granularity rule); P-P-buffer alignment patterns; CLAUDE.md "32 B alignment is real, not advisory".
- **Evidence**: op#27 27_MultiMaskAttentionAggregation (a3 V220, 2026-04-28) Pass-2 mask_sum buffer — `num_classes` was 2-5 across cases, `torch::empty({num_classes}, opts_f32)` allocated 8-20 B, kernel's 32 B `DataCopy` overflowed into adjacent torch tensor → random Pass-A failures + `double free or corruption` on Python exit. Fixed by `RoundUp64(C, 8)` host-side pad. Generalizable to any pass writing < 8 fp32 elements. <!-- terminology-ok (historical anchor) -->
- **⚠ Caveat — host over-alloc can WORSEN a larger-burst adjacent OOB, and this class can be a TEST-HARNESS artifact, not an op defect** (SFA a5/351x whitebox, 2026-06-22): for a **256 B-burst** `BroadCastAndCopyOut` `DataCopy` into a tightly-`at::empty`'d softmax-aux GM, host over-allocation did **not** help and in one variant made it **worse** (mssanitizer OOB count 14→20) by re-adjacency in the torch caching allocator's GM pool — the extra headroom just relocated which neighbor got clobbered. Before "fixing" such an OOB, prove **artifact-vs-defect**: run a **production-clean sibling op** (e.g. stock FA) under the identical tight-`at::empty` harness — if it trips the identical OOB + backtrace, the finding is a torch-caching-allocator tight-adjacency **test-harness artifact**, not your kernel's bug (four fixes were refuted this way: DataCopyPad exact-extent, clamp-to-bound, round-up alloc, full-burst headroom). The `RoundUp64` pad above is correct for the <8-element under-alloc case; it is NOT a universal remedy for burst-into-adjacent-pool OOBs.
### EC-42: AscendC autogen K_TYPE single-match-per-cpp — mixing AIV and AIC kernels in one .cpp silently mis-registers all but the first

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL (silent corruption, no runtime error)
- **Status**: OPEN (CANN 9.0.0 build pipeline)
- **Symptom**: build succeeds with no error, optionally a `[WARNING]: Multiple kernel functions are detected. It is recommended to define only one kernel function per file.` line in build log. At runtime, the kernels meant to be AIC (e.g. cube via `MatmulImpl`) silently produce no output — destination buffer reads as uninitialized memory. Same .cpp's first-declared kernel (whose `KERNEL_TASK_TYPE_DEFAULT(...)` ran first) runs correctly; subsequent kernels of a different task type write nothing.
- **Trigger pattern**: kernel author puts kernels of DIFFERENT task types (`KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_AIC_ONLY`) in the same .cpp file, expecting per-`extern "C"` `KERNEL_TASK_TYPE_DEFAULT(...)` to apply locally to each kernel. The macro looks like a per-function attribute but is actually a file-scope global.
- **Root cause**: `cann-9.0.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py::find_kernel_type_by_source` uses `re.search` (single-match) on `__enable_feature_for_compile_default = X;` (the macro injected by `KERNEL_TASK_TYPE_DEFAULT`). Only the FIRST match is taken and applied to ALL kernels in that file. Mixing types silently mis-registers all but the first.
- **Fix**: split kernels into separate .cpp files by task type. Build script `kernel_dir.glob("*.cpp")` (excluding `pybind11.cpp`) picks up multiple files independently, each with its own first-and-only `KERNEL_TASK_TYPE_DEFAULT(...)` line. Naming convention: `<op>_aiv_kernels.cpp` + `<op>_aic_kernels.cpp` per the op#7 ConvStandard2d precedent.
- **Detection rule**: if your kernel set has BOTH `KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_AIC_ONLY`, you MUST split. The build warning is logged but easy to miss.
- **Diagnostic when output is silent zeros**: re-deploy a minimal one-cube-call kernel in its own .cpp first to confirm the cube path itself works, before debugging algorithm. If isolated cube works but combined doesn't → K_TYPE trap is the root cause.
- **Evidence**:
  - op#7 ConvStandard2d Opt1 (aog-kernel-optimizer ko-1, 2026-04-29). Single combined `conv2d_kernels.cpp` with AIV im2col + AIC cube + AIV bias → 50/50 precision PASS appeared but cube output was zeros (debug `4×8 @ 8×16 = ones×8` returned zeros). Split into `conv2d_aiv_kernels.cpp` + `conv2d_aic_kernels.cpp` → cube ran correctly, returned 8.0 as expected, end-to-end 50/50 + 16/16 PASS, perf median 0.087× → 0.155× → 0.705×.
  - op#6 QuantMatmul kw (Phase E backfill 2026-05-07). Two-launch quant-matmul: AIC int8 GEMM → workspace int32 → AIV dequant → output T. Single `quantmatmul_kernels.cpp` with both `[aicore]` and `[aivec]` `extern "C"` entry points; `device_aiv.o` was empty post-build, AIV dequant never ran, output buffer was uninitialized. Detection: `ls kernels_aiv_device_dir/` empty when AIV launches were expected. Fix: split into `quantmatmul_kernels.cpp` (AIC) + `quantmatmul_aiv_kernels.cpp` (AIV) → 50/50 PASS. Confirms generalization beyond conv-shaped two-launch ops; applies to the textbook quant-matmul shape on Ascend950PR.
- **Cross-ref**: OL-91 (cube playbook), P-P68 (single-AIC GEMM template). Any kernel layering AIV stages around an AIC cube call needs split-cpp.
### EC-43: bisheng `half` macro undefined in `aic_obj` compile target — direct-include `kernel_operator.h` in .cpp

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: HIGH (compile-time blocker, error message mentions a similar-but-wrong type which sends debug down the wrong path)
- **Status**: CONFIRMED 2026-05-01 op#2 SwiGLU kw-2 build iter 1
- **Affected**: Ascend950PR / CANN 9.0.0 b103 / bisheng 2026-03-21. SIMD multi-core class kernel (`__global__ __aicore__` + `class.Init().Process()` pattern), kernel.h uses `__gm__ half*` reinterpret_cast.
- **Symptom**: `error: unknown type name 'half'; did you mean 'half2'?` at the `__gm__ half*` reinterpret_cast site, in the `aic_obj` build target — even though the op is `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` (no AIC kernels). Same pattern in older built kernels in `output/npukernelbench/src/kernels/*/kernel/` compiles fine, suggesting context-dependent.
- **Root cause hypothesis (not fully bisected)**:
  - `__clang_cce_types.h:19` defines `#define half __cce_half`
  - `__clang_cce_runtime_wrapper.h:139` does `#undef half` then `typedef __cce_half half;` immediately after — so `half` becomes an actual type alias.
  - For the `aic_obj` compile target, `runtime_wrapper.h`'s typedef path may not be reached (likely guarded by `#if !defined(CCE_NO_HALF)` or analogous macro polarity that differs per target). The `#undef` in `runtime_wrapper.h` ran BUT the typedef did not fire → `half` is left as undefined identifier.
- **Workaround (canonical, verified)**: add `#include "kernel_operator.h"` **directly** at the top of the `<op>_kernels.cpp` file (the dispatcher .cpp), not just transitively via `#include "<op>_kernel.h"`. The direct include reaches `runtime_wrapper.h` early enough in the `aic_obj` target's preprocessor stack that the typedef fires before the kernel.h's `__gm__ half*` site.
- **Detection rule**: if you see "unknown type name 'half'; did you mean 'half2'?" in `aic_obj` build output, FIRST add `#include "kernel_operator.h"` at top of the dispatcher .cpp. Do NOT chase the suggested 'half2' — it's a wrong rabbit hole.
- **Reference templates that have the direct include**: `output/npukernelbench/src/kernels/11_GroupNorm/kernel/groupnorm_kernels.cpp:1` (working), `output/npukernelbench/src/kernels/2_SwiGLU/kernel/swiglu_kernels.cpp:1` (post-fix). Static check `missing_kernel_operator` was added 2026-05-01 to flag .cpp files without the direct include.
- **Evidence**: op#2 SwiGLU kw-2 (2026-05-01) — initial Phase B kernel (synthesized from scratch, only `#include "<op>_kernel.h"` in .cpp) failed compile in `aic_obj` target with the unknown-type-name error. Workaround applied (direct include) → build PASS first try.
### EC-44: `deploy_to_npu.sh` TARGET case statement rejects `a3-ds` and other alias targets
`applies_to: soc=all; cann=all; bisheng=all; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0-beta.2`
- **Severity**: LOW (deploy-time blocker, workaround exists)
- **Status**: OPEN
- **Symptom**: `deploy_to_npu.sh`'s case statement only accepts `a5|a3|a2` for TARGET. When `.ascendc_env` sets `TARGET=a3-ds` (DS-backend isolation), the script exits with no matching case and deploy is blocked.
- **Workaround**: manual `tar+scp+docker exec` deploy, or temporarily change TARGET in .ascendc_env.
- **Fix**: add case aliases in the script (`a3-ds|a3_kimi|a3`) all mapping to the a3 code path.
- **Detection rule**: if `deploy_to_npu.sh` exits silently without deploying, check that TARGET matches one of the script's expected values.
- **Evidence**: op#30 NMS a3 ds kw-1 (2026-05-07) — manual tar+scp+extract deploy required because script rejected TARGET=a3-ds.

### EC-45: Container `CA_cann_9_b2_kevin` is compile-only — stub CANN libraries, `aclInit` returns 100039
`applies_to: soc=Ascend910_9382; cann=9.0.0-beta.2; bisheng=n/a; op_class=all`
- **Severity**: HIGH (runtime blocker — kernel builds but cannot execute)
- **Status**: CONFIRMED 2026-05-07 op#30 NMS ds kw-1
- **Symptom**: Kernel builds successfully and .so is produced, but `aclInit` returns error code 100039 ("stub library cannot be used for execution"). All subsequent ACL calls fail.
- **Root cause**: container image was built with stub/development CANN libraries, not full runtime. NPU passthrough is absent or the installed CANN lacks libascendcl.so with runtime backend.
- **Detection rule**: after build, run a minimal `aclInit` + `aclrtSetDevice` smoke test before proceeding to precision/perf verification. If `aclInit` returns non-zero, the container cannot run kernels — switch to a runtime-capable container or accept build-only verification.
- **Workaround**: runtime verification must use a container with full CANN runtime + NPU device passthrough (e.g., `--device=/dev/davinciX`).
- **Generalizes to**: any container where CANN was installed from a dev/stub package rather than the full runtime package. Check `aclInit` return code as the first step of any runtime verification.
- **Evidence**: op#30 NMS ds kw-1 (2026-05-07) — build succeeded (9.2MB .so), `aclInit` returned 100039 on CA_cann_9_b2_kevin. Runtime verification was performed on a different container/setup.

### EC-46: Explicit `kernel_operator_<adv>_intf.h` include fails — adv_api headers are pulled transitively via `kernel_operator.h`
`applies_to: soc=all; cann=9.0.0; bisheng=2026-03-21; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`
- **Severity**: MEDIUM (compile-time hard fail, well-defined fix)
- **Status**: CONFIRMED 2026-05-13 group_norm_silu_quant build iter 1
- **Symptom**: kernel explicitly adds `#include "kernel_operator_<adv>_intf.h"` (e.g. `kernel_operator_sigmoid_intf.h`) following ASCENDC_API_CATALOG.md's "Header: adv_api/<adv>/kernel_operator_<adv>_intf.h" line. Build fails first preprocessor pass: `'kernel_operator_sigmoid_intf.h' file not found`. The header DOES exist on disk at `cann-{ver}/<arch>-linux/asc/include/adv_api/<adv>/kernel_operator_<adv>_intf.h` (EC-38 confirms presence), but the default build include path does NOT reach `adv_api/` — adv_api headers are intended to be pulled in transitively, not directly included.
- **Root cause**: ASCENDC_API_CATALOG.md's "Header: ..." annotation documents WHERE the API is defined for human reference; it is NOT a recommendation to add an explicit `#include`. `kernel_operator.h` is the canonical entry point that pulls in all adv_api headers transitively. Adv_api intf.h files reference internal types/macros assuming kernel_operator.h was loaded first; including them standalone breaks even if the path were on the search list.
- **Fix**: keep ONLY `#include "kernel_operator.h"` (or `<kernel_operator.h>`). Call the adv API directly — `Sigmoid(dst, src, count)` / `Tanh(dst, src, count)` / etc. — and trust the transitive include.
- **Signature note (Sigmoid specifically)**: prefer the 2-arg form `Sigmoid(dst, src, count)` (sizes its tmp buffer internally from `srcTensor.GetSize()`) over the older 3-arg form `Sigmoid(dst, src, tmpBuf, count)`. Cousin op `11_DequantSwigluQuant` uses the 2-arg variant and is the reference template for sigmoid-bearing kernels.
- **Detection rule**: compile error `'kernel_operator_<adv>_intf.h' file not found` + the offending #include line traces back to a worker reading ASCENDC_API_CATALOG.md's "Header:" annotation. Fix is mechanical: delete the explicit #include line.
- **Catalog edit candidate**: ASCENDC_API_CATALOG.md should re-annotate "Header: ..." entries as "(transitive via `kernel_operator.h` — do NOT add explicit `#include`)" or drop the field entirely. Filed as catalog improvement; not blocking this EC.
- **Related**: EC-38 (catalog miss ≠ API doesn't exist; the `ls adv_api/<name>/` step that EC-38 mandates verifies the API exists — but EC-46 says you still don't need to include it explicitly once verified).
- **Evidence**: group_norm_silu_quant (2026-05-13, A5 fused GroupNorm+SiLU+Quant port) — iter 1 added `#include "kernel_operator_sigmoid_intf.h"` per catalog line 234 advice → build failed `file not found`. Iter 2 dropped the explicit include, used `Sigmoid(...)` directly with `<kernel_operator.h>` already in scope → build PASS, all 8 cases bit-exact on Pass A. Cousin op `11_DequantSwigluQuant` archive validated the no-explicit-include pattern is the project precedent.

### EC-47: `ToFloat<>` static_assert "only support bfloat16_t/hifloat8_t/fp8_*/fp4_*" after BF16 guard removal [V351, port_a3_to_a5]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0; op=ctc_loss_v3`
`source: PR 103 SKILL.md §455-471 (cites our ctc_loss_v3 as the canonical example)`

**Symptom**:
```
kernel_scalar_convert.h: error: static assertion failed: ToFloat only support
bfloat16_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/fp4x2_e1m2_t/fp4x2_e2m1_t data type on current device!
```
Call chain: `ToFloat<>(val)` → `Cast<>(bVal)` → `static_assert` fail.

**Root cause**: On A5, the templated `ToFloat<>` helper only specializes for the low-precision narrow-floats. When A3-source code removed its BF16 conditional-compile guards (per OL-142 / EC-49) but the underlying tensor type changed in the process, the call to `ToFloat<>` may now receive a wider type (`half` / `float`) that `ToFloat<>` refuses by design.

**Concrete example (ctc_loss_v3, cited in PR 103)**:
```cpp
// BEFORE — under #if guarded BF16 path, ToFloat sees bfloat16_t (OK)
logProbBlank = ToFloat(logProbBlankTensor.GetValue(0));

// AFTER guard removal — type chain shifts; ToFloat now sees half (REJECTED)
// Fix: explicit ReinterpretCast<bfloat16_t>() before ToFloat
logProbFirstChar = ToFloat(logProbFirstTensor
    .template ReinterpretCast<bfloat16_t>()
    .GetValue(0));
```

**Fix pattern** (universal):
- Before calling `ToFloat<>(x)`, insert `.template ReinterpretCast<bfloat16_t>()` if the underlying memory holds BF16-encoded bits but the static type is `half`/`float`/etc.
- For genuine `half` / `float` values, use plain `static_cast<float>(x)` instead of `ToFloat<>`.

**Detection signature**: search for `ToFloat<` calls in newly-ported arch35/ kernels; cross-check against `LocalTensor<T>` declarations to confirm T is in the allowed set OR a `ReinterpretCast` is present.

**Evidence**:
- ctc_loss_v3 (2026-05-13): we hit this during the L1 port; the PR 103 authors saw our archive and cite it explicitly in their fast-track table
- PR 103 EC table line 452: "ToFloat<> static_assert 失败 | A5 上 ToFloat 仅支持 BF16/FP8/HiFloat8 等新类型"

**Mitigation gate**: post-worker `aog-self-critic` should grep arch35/ kernel headers for `ToFloat<` and emit a soft-warning if the surrounding `LocalTensor` type is `half` / `float` without `ReinterpretCast`.

**Cross-reference**: EC-49 (BF16 guard removal) often causes this; the fix sequence is "remove guard → recompile → if static_assert fires, apply this fix".

---

### EC-48: `Exec format error: bisheng` / `OSError: [Errno 8]` — bisheng wrapper script missing shebang [V351, build-toolchain]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §441-477`

**Symptom**: Build fails with `Exec format error: bisheng` or `OSError: [Errno 8]` referring to `build/gen_bisheng_dir/bisheng`.

**Root cause**: The generated wrapper at `build/gen_bisheng_dir/bisheng` is a shell script but missing `#!/bin/bash` on first line. Most Linux loaders refuse to exec it. **The wrapper is regenerated by every `clean build`, so the fix must be reapplied each time.**

**Fix**:
```bash
sed -i '1i#!/bin/bash' build/gen_bisheng_dir/bisheng
```

**Detection signature**: grep first line of `build/gen_bisheng_dir/bisheng` for `#!` shebang; if absent, prepend.

**Evidence**:
- PR 103 lists as Trap row 1 of "常见编译错误速查" (most common build error)
- Re-fires after every `rm -rf build && cmake ...` cycle — not a one-time fix

**Mitigation**: orchestrator's `build_runner.sh` should auto-apply the sed fix as a post-cmake step. Add to `aog-a3-author` Path B build_runner.sh template.

---

### EC-49: BF16 `__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113` guards MUST be removed in `arch35/` ports [V351, port_a3_to_a5]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all_port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §289-303 + OL-142`

**Symptom**: Compile failure inside `arch35/` kernel header — a BF16-typed path that should compile is dead-code-eliminated, leading to "undefined function" errors at the call site OR (worse) silent fallback to wrong-dtype path.

**Root cause**: A3 source liberally wraps BF16 / new-dtype paths in negative guards `#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))`. On A5 (`__NPU_ARCH__ == 3510`) these conditions evaluate false → guard becomes `#if !false` = include. ACCIDENTALLY correct, but fragile — if a future A5 SoC variant gets a different numeric ID, the guard would silently exclude code. More importantly, A3 BF16 paths inside the guards are STILL written for V220 codegen; on A5 they may use deprecated APIs.

**Fix (mechanical, every port_a3_to_a5 L1 step)**:

```bash
# Identify in A3 source
grep -nE "__NPU_ARCH__\s*==\s*(3003|3113)" op_kernel/*.h op_kernel/*.cpp

# For each match: remove the guard, keep the body unconditional
```

```cpp
// BEFORE (A3 source)
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    BF16_PATH;
#endif

// AFTER (arch35/ port) — guard removed
BF16_PATH;
```

**Anti-pattern**: copy A3 source verbatim into `arch35/` without removing these guards. The build "works" by accident (3510 ≠ 3003 ≠ 3113) but the port is unstable.

**Detection signature**: after `arch35/` files are written, `grep -E "__NPU_ARCH__\s*==\s*(3003|3113)" arch35/*.h arch35/*.cpp` MUST return zero hits.

**Evidence**: PR 103 §289-303 codifies as canonical L1 step

**Mitigation gate**: `aog-self-critic` post-worker — auto-grep `arch35/` for residual 3003/3113 references; reject finalize if found.

**Cross-reference**: OL-142 (NPU_ARCH macros), EC-47 (ToFloat fix often needed after this removal).

### EC-50: Target prior-art JSON may contain duplicate `bin_filename` keys — regenerate a valid task-owned schema [arch22→arch35]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2.5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: workspace/foreach_reciprocal/knowledge_update.md (kw-1, 2026-05-16)`

**Symptom**: While mirroring upstream `op_host/config/ascend950/<op>_binary.json` into a port_a3_to_a5 workspace, the file contains a JSON object with TWO `bin_filename` entries on adjacent lines for the same dtype variant:

```json
{
    "bin_filename": "ForeachReciprocal_8d9f799857af3a32fcb6092255dfdab9",
    "bin_filename": "ForeachReciprocal_10f6ed20a89d7d8d379ca7132257bfa5",
    "inputs": [...]
}
```

**Root cause**: This is a JSON spec violation (RFC 8259 forbids duplicate keys in the same object) shipped in upstream's prebuilt config. The first hash is a stale entry from a prior build run that wasn't pruned; the second hash is the current build's effective bin_filename. Most JSON parsers (Python `json.loads`, C++ `nlohmann::json` default config, jq) implement **last-key-wins** semantics, so the runtime picks the second hash and the binary loads correctly.

**Fix**: treat the target JSON as advisory evidence, then generate a task-owned RFC-8259-valid schema
from the selected operator contract and current build outputs. If duplicate keys make the effective
value ambiguous, fail generation and require an explicit canonical value. Validate the chosen
`bin_filename` against the binary produced by the current clean build; byte identity with target prior
art is not an acceptance gate.

**Anti-patterns**:
- Copying the duplicate-key target file into the deliverable because the default parser happens to use
  last-key-wins.
- Normalizing it silently without recording which canonical value was selected and why.

**Escalation**: if the selected contract and current build cannot determine one canonical filename,
return a visible schema/provenance failure. Do not substitute the target file as a workaround.

**Historical evidence**: foreach_reciprocal 2026-05-16 target prior art carried two bf16
`bin_filename` keys; the old pipeline accepted last-key-wins. This demonstrates the ambiguity and
motivates strict task-owned schema generation; it does not justify copying the invalid JSON.

**Other instances (predicted)**: any upstream `<op>_binary.json` for ports where the build farm has multiple historical hashes. Quick detector: `python -c "import json; json.loads(open('<f>').read(), object_pairs_hook=lambda p: [None for k,_ in p if list(zip(*p))[0].count(k)>1])"` — flags duplicate keys at parse time.

**Cross-reference**: OL-141 (target artifacts are advisory, never a skip or byte-mirror verdict),
OL-157 (foreach unary family packaging patterns).

### EC-51: ops-nn-port cmake fails `Could not find a package configuration file provided by "ASC"` — ASCEND_CANN_PACKAGE_PATH not auto-exported

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: ops-nn-port `build.sh --pkg` or `--opkernel` fails at cmake configure with `Could not find a package configuration file provided by "ASC"`. Trace points to `cmake/dependencies.cmake` line ~75: `find_package(ASC REQUIRED HINTS ${ASCEND_CANN_PACKAGE_PATH}/.../tikcpp/ascendc_kernel_cmake)` resolving to an empty hint path.

**Root cause**: CANN's `set_env.sh` sets `ASCEND_HOME_PATH` (and `ASCEND_TOOLKIT_HOME`, etc.) but does NOT export `ASCEND_CANN_PACKAGE_PATH`. The ops-nn-port build chain reads `ASCEND_CANN_PACKAGE_PATH` directly, with no fallback to `ASCEND_HOME_PATH`. With it unset, the `HINTS` clause becomes a literal `/.../tikcpp/ascendc_kernel_cmake` path and `find_package` cannot locate the actually-shipped `asc-config.cmake`.

**Fix** — explicit pre-build export:
```bash
source /data/cann_b103/cann-9.0.0/set_env.sh    # NOT pipelined — see anti-pattern below
export ASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH
# Then run build.sh
```
The CANN install at `/data/cann_b103/cann-9.0.0` ships `asc-config.cmake` at BOTH:
- `${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc-config.cmake`
- `${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc-config.cmake`

Either path resolves; the `HINTS` lookup walks both.

**Anti-pattern** (separate bug class — observed in same incident): `source /data/cann_b103/cann-9.0.0/set_env.sh | tail -1`. The pipe puts `source` into a subshell — env vars set by `set_env.sh` (including `ASCEND_HOME_PATH`) NEVER reach the parent shell. Use `source ... 2>&1; tail` or `{ source ...; } && echo done` if filtering output is needed. **Symptom is identical** to the missing-export above: `ASCEND_HOME_PATH` is also unset, so `ASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH` exports an empty string.

**Evidence**: fatrelu_mul kw-1 port_a3_to_a5 (2026-05-17, A5 host 198.51.100.35 container npu_dev3, CANN 9.0.0 build b103). Iter-1 cmake failure trace; fix in iter-2 produced `Built target ascendc_impl_gen` and proceeded through `--opkernel` to 3× kernel ELF.
- GDN `chunk_gated_delta_rule` catlass/bisheng build (A5/V351, CANN 9.1.T500, 2026-06-16): an ICE-looking catlass build failure was actually CANN `set_env.sh` never sourced at all (`ASCEND_HOME_PATH` entirely unset, so the toolchain/`ASC` lookup resolved to empty hint paths) — same root class, one-line fix (`source .../set_env.sh` before build). A build error that LOOKS like a compiler ICE should first be triaged as "is the CANN env sourced?" before suspecting bisheng.

**Other instances (predicted)**: any ops-nn-port build on a fresh container/shell session; any orchestrator-spawned build subshell that re-sources `set_env.sh` without explicit `ASCEND_CANN_PACKAGE_PATH` follow-up export. Add to the canonical port_a3 build pre-step list.

**Cross-reference**: OL-158 (Phase C build interpretation — this EC is the prerequisite for getting to a state where `--pkg`/`--opkernel` artifact-set inspection is meaningful).

### EC-52: A5 port `<op>_apt.cpp` `fatal error: '<op>_tiling.h' file not found` — must not `#include` the host-side tiling header

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: opc compile of A5 port's `op_kernel/<op>_apt.cpp` fails with `fatal error: '<op>_tiling.h' file not found` even though `op_host/<op>_tiling.h` exists in the source tree.

**Root cause**: `op_host/` is NOT in the kernel-side compile include path. The tiling-data struct definition (e.g. `FatreluMulTilingData`) is auto-injected into the kernel translation unit by the build pipeline's tiling-data header generation — driven by the `BEGIN_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS` macros in `op_host/<op>_tiling.h`, processed host-side, then re-emitted as a synthesized header that `GET_TILING_DATA()` resolves at kernel compile time. An explicit `#include "<op>_tiling.h"` from the apt.cpp searches the wrong path and trips the compile.

**Fix**: remove `#include "<op>_tiling.h"` (or any include of `op_host/<op>_tiling.h`) from the `<op>_apt.cpp`. The `GET_TILING_DATA(<varname>, tiling)` macro inside `Process()` resolves the struct through the auto-injected header — no manual include needed.

**Reference convention**: the A3 upstream `op_kernel/<op>.cpp` (e.g. `cann/ops-nn/activation/fatrelu_mul/op_kernel/fatrelu_mul.cpp`) also does NOT include `op_host/<op>_tiling.h`. The A5 `<op>_apt.cpp` should follow the same convention — adding the include "defensively" is a worker-side mistake.

**Evidence**: fatrelu_mul kw-1 port_a3_to_a5 (2026-05-17). Iter-2 tripped on this exact symptom after worker added the include as a "make-sure-it-resolves" measure; removing the line let `--opkernel` proceed cleanly to ELF emission. The A3 upstream pattern was the authoritative counter-example.

**Other instances (predicted)**: any greenfield A5 port (Mode B / Mode B-simple / Mode B-mechanical per OL-141 / OL-158). Especially likely when worker generates apt.cpp from scratch rather than copying-and-editing an existing arch35 sibling. Add to W11 apt.cpp emission checklist.

**Cross-reference**: OL-141 (target `op_kernel/arch35/` include structure is advisory; derive the
task-owned include set from current public APIs), W11 (arch35 apt.cpp emission gate).

### EC-53: A5 V351 runtime error 507035 + EZ9999 `errcode:(95) MTE instruction DDR address out of range` — kernel GM extent overshoot (fault tree) [V351, port_a3_to_a5]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 V220 silently tolerated the same overshoot — that is precisely why the upstream bug survived)`

**Error pattern** (full A5 stack):
```
AclrtSynchronizeDeviceWithTimeout, error code is 507035
EZ9999: ... errcode:(95) errorStr: The DDR address of the MTE instruction is
out of range. subErrType: 0x4
```
Distinct from EC-23 (DataCopyPad UB→GM 507035 — pure direction-not-supported)
and EC-27 (SOC_VERSION-derived 507035 — illegal instruction at PC 0x80). EC-53
is specifically the **MTE-OOR sub-class** of 507035, identified by the
`errcode:(95)` line in the EZ9999 detail.

**Root cause family**: kernel issues a MTE2 (GM→UB) or MTE3 (UB→GM) transfer whose
DDR address range extends past the actual allocation of the targeted tensor. A5
V351's MTE hardware boundary-checks; A3 V220 did not, so the same kernel binary
ran "fine" on A3 (silently reading/writing past the buffer end) and traps on A5.

**Fault tree** (port_a3_to_a5 — check in order):
1. **Input-extent conflation (most common)**: kernel uses `SetGlobalBuffer((__gm__ T*)<X>, extent)`
   where `extent` is derived from input A's shape but `<X>` is input/output B with a
   smaller shape. Detect via: grep kernel.h for `SetGlobalBuffer` and audit each `extent`
   argument against the actual tensor it pairs with. Fix: pybind padding wrapper —
   see OL-162.
2. **InitOutput overshoot**: `InitOutput(<tensor>_gm, X, 0)` writes X elements when the
   actual allocation is only Y < X. Triggered same way as #1 (X computed from a
   different shape). Fix: same as #1 (pybind padding) OR cap X at `min(X, actual_size)`
   when both are host-side known.
3. **UB-budget tiling overshoot**: `per_core_do_block_num = UB_budget / one_block_size`
   sized for A5's larger UB exceeds the op's actual `block_num`; kernel loads past
   GM end on small-shape cases. This is **OL-158**'s territory — fix in host tiling
   with `std::min(ub_budget_blocks, block_num)`. Distinct from #1/#2 because the
   overshoot is derived from A5 vs A3 capacity delta, not from a source-level shape
   conflation.
4. **DataCopyPad stride > actual extent**: tile inner-loop with `blockLen + stride`
   exceeding tensor end. Rarer; usually paired with a custom non-aligned tail handler.
   Fix: tile-size cap or kernel-side bound check.

**Diagnostic checklist** (when EZ9999 95/0x4 appears on A5 but A3 was clean):
- Run the same kernel on shapes where all relevant dims are EQUAL (e.g. M==N for
  2-pointcloud ops). If MTE-OOR disappears, root cause is #1 or #2.
- Inspect `verification.json` failing-case shape vs the kernel's `SetGlobalBuffer`
  extents. If extent is derived from `xyz1` dim but applied to `xyz2` GM pointer,
  confirmed #1.
- Audit `InitOutput` arg #2 against output tensor's `numel()`; mismatch → #2.

**Evidence**:
- chamfer_distance_grad kw-1 port_a3_to_a5 (2026-05-17): cases 3 (B=2 M=1 N=128)
  and 6 (B=1 M=1 N=128) triggered errcode 95 on A5. Root cause: kernel uses xyz1's
  N as both batch-stride and `InitOutput(grad_xyz2_gm, B*N*2, 0)` extent for the
  grad_xyz2 tensor whose actual size is B*M*2. Case 6's `InitOutput` wrote 256
  fp32 into a 2-fp32 allocation. Fix: pybind padding wrapper per OL-162. After
  fix: 8/8 cases PASS_WITHIN_TOLERANCE on A5 (vs A3 capture, vs CPU truth).
  Same kernel binary executes cleanly on A3 V220 for those shapes — confirms A3
  hardware silently tolerates GM-out-of-allocation while A5 V351 traps.

**Other instances (predicted)**:
- Any 2-pointcloud / 2-feature-map op family (`loss/*_grad`, `pointcloud/*`)
  whose upstream kernel was authored for symmetric-shape test drivers and never
  hardened for N≠M.
- Any op with `aclrtMemset`-on-output → `InitOutput`-extent inheritance from a
  larger sibling input.
- Any A3→A5 port where the test driver historically used a single N for all
  related tensors. Make the asymmetric-shape sweep mandatory in edge case_gen.

**Cross-reference**:
- OL-162 — pybind padding wrapper (the fix for fault #1/#2 without modifying the
  L1-verbatim kernel body)
- OL-158 — host-tiling per-core-block cap (the fix for fault #3)
- OL-160 — canonical entry-point names (the pybind wrapper lives in pybind11.cpp
  attached to the canonical `model_new_ascendc.py`)
- EC-23 — different 507035 sub-class (DataCopyPad UB→GM, not GM extent overshoot)
- EC-27 — different 507035 sub-class (SOC_VERSION default, not MTE boundary check)

### EC-54: A3→A5 port — PR4778 ship artifacts (`<op>_apt.cpp`, `arch35/<op>.h`) placed under `kernel/` instead of `op_kernel/` → `build_ascendc.py` glob picks them up and compile fails with `unknown type name '<Op>TilingData'` [V351, port_a3_to_a5, build-layout]

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: workspace-side `build_ascendc.py` verify-only build (the per-op pybind/ACLRT_LAUNCH_KERNEL build, NOT the on-host CANN ops-nn-build pipeline) fails with:
```
kernel/arch35/<op>.h:NN:NN: error: unknown type name '<Op>TilingData'
kernel/<op>_apt.cpp:NN:NN: error: use of undeclared identifier 'tilingData'
```

**Root cause**: `build_ascendc.py` globs `workspace/<op>/kernel/*.cpp` for compilation. The `<Op>TilingData` struct + `GET_TILING_DATA` macro are emitted ONLY by the CANN ops-nn-build pipeline (`build.sh --pkg --ops=<op> --soc=ascend950`) — they do not exist at `build_ascendc.py` time. When the worker places PR4778 ship artifacts (`<op>_apt.cpp` and/or `arch35/<op>.h`) under `kernel/` instead of `op_kernel/`, the verify-build glob picks them up and tries to compile them outside the auto-tiling pipeline → struct + macro are unresolved → compile error.

This is distinct from EC-52 (`<op>_apt.cpp` mistakenly `#include`s `<op>_tiling.h`). EC-52 = include resolution bug; EC-54 = layout misplacement bug. Both surface the same `unknown type name '<Op>TilingData'` token but the fix paths are independent.

**Fix**: enforce PB-33 layout split. `kernel/` is verify-only (pybind11.cpp + `<op>_kernels.cpp` + `<op>_kernel.h` — what `build_ascendc.py` compiles); `op_kernel/` is ship-only (`arch35/<op>.h` + `<op>_apt.cpp` — what `build.sh --pkg --ops=<op>` compiles).

```
workspace/<op>/
  kernel/                 ← build_ascendc.py glob picks these up
    pybind11.cpp
    <op>_kernel.h
    <op>_kernels.cpp
  op_kernel/              ← ops-nn-build pipeline picks these up
    arch35/<op>.h
    <op>_apt.cpp
  op_host/                ← ops-nn-build pipeline picks these up
    <op>_def.cpp, <op>_tiling.cpp, <op>_tiling.h, CMakeLists.txt, config/ascend950/...
```

Mechanical fix when misplacement is detected:
```bash
mv workspace/<op>/kernel/arch35/        workspace/<op>/op_kernel/arch35/
mv workspace/<op>/kernel/<op>_apt.cpp   workspace/<op>/op_kernel/<op>_apt.cpp
```

**Detection signature** (pre-build audit):
```bash
# Should ONLY see pybind11.cpp + <op>_kernel.h + <op>_kernels.cpp under kernel/
ls workspace/<op>/kernel/*.cpp 2>/dev/null | grep -E '_apt\.cpp$' && echo "EC-54: apt.cpp misplaced under kernel/"
[ -d workspace/<op>/kernel/arch35 ] && echo "EC-54: arch35/ misplaced under kernel/"
```

**Evidence**:
- fatrelu_mul kw-1 port_a3_to_a5 (2026-05-17): worker initially emitted `kernel/arch35/fatrelu_mul.h` + `kernel/fatrelu_mul_apt.cpp` + `kernel/fatrelu_mul_kernels.cpp`. Build globbed all three .cpp files; apt.cpp compile failed with `unknown type name 'FatreluMulTilingData'`. Fix was `mv` to `op_kernel/`. After move, the kernel/ glob only saw `fatrelu_mul_kernels.cpp` + `pybind11.cpp` and build proceeded cleanly. Iter-1 build PASS after layout correction → 8/8 T1_BIT_EXACT precision.

**Other instances (predicted)**: every port_a3_to_a5 op that emits BOTH a pybind/ACLRT_LAUNCH_KERNEL verify path AND PR4778 ship artifacts. Especially likely when the worker generates apt.cpp + arch35/ from scratch without consulting PB-33 layout.

**Mitigation candidates** (out of scope for this entry — would require harness change):
- (a) `build_ascendc.py` exclude-by-name pattern: explicitly skip `*_apt.cpp` and `arch35/` subdir (already excludes `pybind11.cpp`).
- (b) `finalize_pipeline` pre-build-gate: warn if `workspace/<op>/kernel/arch35/` exists or `kernel/<op>_apt.cpp` exists, before invoking `build_ascendc.py`.

**Cross-reference**:
- PB-33 — archive layout contract (kernel/ vs op_kernel/ split); EC-54 is the build-time failure mode when the contract is violated at workspace level
- EC-52 — different cause for the same `unknown type name '<Op>TilingData'` token (defensive `#include` of `<op>_tiling.h` from a correctly-placed apt.cpp)
- OL-141 — target `op_kernel/arch35/` is advisory layout evidence, not a body mirror source
- OL-160 — canonical entry-point names (`kernel/pybind11.cpp` + `kernel/<op>_kernels.cpp` are the verify-path canonical names; misplacing ship artifacts under `kernel/` violates this naming invariant)


### EC-55: Pybind module import fails with a torch symbol — link `libtorch_python` before diagnosing ABI
`applies_to: soc=Ascend910_9382; cann=9.0.0; torch=2.9.0+cpu; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0; torch=2.9.0+cpu`
`unverified_on: soc=Ascend950PR (verify the installed torch build before assuming the same ABI)`

(ds agent extraction 2026-05-19 from earlier ds-branch commit e13d49db; originally numbered EC-47 in ds branch but main now has different EC-47 — renumbered to EC-55.)

- **Error pattern**: the pybind module links successfully, but `import <module>`
  fails with a symbol such as `_ZTVN5torch8autograd12AutogradMetaE` or
  `pybind11::detail::type_caster<at::Tensor>::cast`.
- **First diagnosis**: the `type_caster<at::Tensor>` symbols are provided by
  `libtorch_python.so`; the `AutogradMeta` vtable is defined by PyTorch's core
  `libtorch_cpu.so`, normally reached through `libtorch.so`. Missing `torch`
  and/or `torch_python` from the extension's link set is therefore a
  link-configuration defect, not evidence of an ABI mismatch.
- **Fix and diagnosis order**:
  1. Inspect the generated `target_link_libraries`. The current `torch::Tensor`
     binding contract requires `torch`, `torch_python`, and `torch_npu`.
  2. Verify the selected torch library directory contains
     `libtorch_python.so*`. If it does not, report the environment dependency as
     missing; do not disguise it with a different binding contract.
  3. Fix the persistent `build_ascendc.py` generator, delete the affected build
     directory, and rebuild. Do not patch only an auto-generated CMake file.
  4. Use `readelf -d <module>.so` to inspect `DT_NEEDED`. Run `ldd -r` in the
     runtime environment and filter for the target torch symbols; a standalone
     `ldd -r` can legitimately report Python C-API symbols that the interpreter
     supplies. The decisive check is a fresh interpreter that imports `torch`
     and `torch_npu` before importing the extension.
  5. Only if all three libraries are linked and their symbols resolve should an
     incompatible torch/torch_npu/extension ABI pairing be investigated.
- **Current contract**: generated bindings use `torch::Tensor`. Do not replace
  them with raw Python-object operations or `py::object`; that bypasses the
  worker hooks and changes the runtime interface.
- **Corrected evidence (2026-07-30)**: a backward-generation extension initially
  failed on `AutogradMeta` while its generated CMake linked only `torch_npu`.
  Linking `torch`, `torch_python`, and `torch_npu`, then rebuilding cleanly,
  produced an importable module with resolved dependencies.
- **Historical evidence, not a current recommendation**: op#3 Add (2026-05-11)
  bypassed the tensor binding with raw pybind11/Python operations and imported,
  but only 44/50 precision cases passed. That single run does not establish an
  ABI root cause and its workaround is outside the current contract.

### EC-56: Small NPU tensors (<32 bytes) trigger torch_npu Slice kernel crash 507035 — pad tiling buffer to ≥256 bytes
`applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 — torch_npu Slice kernel may not have this minimum-size constraint; verify on A5 before assuming pattern applies)`

(ds agent extraction 2026-05-19 from earlier ds-branch commit e13d49db; originally numbered EC-48 in ds branch but main now has different EC-48 — renumbered to EC-56.)

- **Error pattern**: creating a small NPU tensor (e.g., 12-byte tiling struct via `torch.empty(12, dtype=uint8).to(device)`) triggers a torch_npu internal Slice kernel crash: `aclrtLaunchKernelWithHostArgs failed, return: 507035`. The crash is in torch_npu's H2D transfer path, not in the user kernel.
- **Root cause**: torch_npu's internal `.to(device)` transfer path invokes a Slice kernel for sub-32-byte tensors, and the Slice kernel on V220 does not handle very small payloads correctly.
- **Fix**: pad the host tiling buffer to ≥256 bytes before `.to(device)`:
  ```python
  PAD_BYTES = 256
  tiling_bytes = struct.pack(...)  # e.g. 12 bytes
  padded = tiling_bytes + b'\x00' * (PAD_BYTES - len(tiling_bytes))
  tiling_tensor = torch.frombuffer(bytearray(padded), dtype=torch.uint8).to(device)
  ```
- **Detection**: if `torch::empty({N}, ...).to(npu_device)` crashes with 507035 and N < 32, suspect this. Pad to 256 bytes and retry.
- **Evidence**: op#3 Add ds kw-1 (2026-05-11, Ascend910_9382 V220, CANN 9.0.0): 12-byte AddTiling struct → `.to(device)` crashed with Slice kernel 507035. Padded to 256 bytes → transfer succeeded.
- **Cross-ref**: OL-77 (GM tiling struct byte-by-byte read — tiling design pattern that generates the small host POD this crash affects).

---

### EC-57: `REGIST_MATMUL_OBJ` lands NO-SYNC branch with single `-DASCENDC_MATMUL_AICORE` — runtime MPU 507015 once cube uses the matmul object [V220+V351, ALL_MODES, build-config + KFC-sync-activation]

`applies_to: soc=Ascend910_9382 (V220 A2/A3) + Ascend950PR_9579 (V351 A5); cann=9.0.0+; bisheng=ascendc.cmake DYNAMIC_MODE; op_class=mixed_aic_aiv_with_REGIST_MATMUL_OBJ; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: a5_ops:3_FusionAttention 2026-05-21T19:53Z (independent CANN-source read of kfc_register_obj.h L262/322/368, confirmed empirically); also reproduced on V351 by probe_a5_v300_fa_sync 2026-05-23 — DEBT-20 per-source-define gap is cross-arch`

- **Severity**: HIGH (build + register-binary both succeed, kernel launches, AIC then traps with MPU error 507015 on first matmul usage — symptom looks like an MPU bug but is actually a missing-define misconfiguration).
- **Symptom**: Mixed cube+vec kernel uses `REGIST_MATMUL_OBJ(...)` to instantiate the `MatmulImpl<>` library. Project sets a single global `-DASCENDC_MATMUL_AICORE` define (the "simple" DEBT-20 approach). Build succeeds. `aclrtlaunch_<op>(...)` returns clean. AIC stage traps with `aicore exception 507015` immediately when cube enters `mm.IterateAll` / `GetTensorC`. No AIV-side error.
- **Root cause** (per independent CANN-source read of `kfc_register_obj.h:262/322/368`):
  - L368: `REGIST_MATMUL_OBJ` macro expansion checks `#if defined(SPLIT_CORE_CUBE)` → if undefined, takes the **NO-SYNC branch** which just runs `InitCurObj(...)` — no KfcServer init on AIC, no KfcCommClient on AIV, no mailbox topology.
  - L262: real KFC server init (the AIC side: `KfcServer.Init() + while-isRun loop`) is gated on `#if defined(SPLIT_CORE_CUBE) && !defined(SPLIT_CORE_VEC)`.
  - L322: real KFC client init (the AIV side: `KfcCommClient + CrossCoreWaitFlag` mailbox poll) is gated on `#if !defined(SPLIT_CORE_CUBE) && defined(SPLIT_CORE_VEC)`.
  - Single global `-DASCENDC_MATMUL_AICORE` doesn't satisfy any of L262/L322 — both passes fall through to L368 NO-SYNC. Cube launches matmul state machine with no server to handshake with → MPU traps when matmul tries to wait for AIV consumer ack.
- **Fix** — per-compile-pass defines:
  - **AIC pass** must receive `-DSPLIT_CORE_CUBE=1` (and `-DASCENDC_MATMUL_AICORE` as before).
  - **AIV pass** must receive `-DSPLIT_CORE_VEC=1` (and `-DASCENDC_MATMUL_AICORE` as before).
  - In CMake (cmake ≥ 3.18 required): `set_source_files_properties(<file>.cpp PROPERTIES TARGET_DIRECTORY aic_obj COMPILE_DEFINITIONS "SPLIT_CORE_CUBE=1")` + symmetric for `aiv_obj` + `SPLIT_CORE_VEC=1`. Apply per kernel source file.
  - `build_ascendc.py` schema extension (landed `66a4d985`): `per_source_defines` now accepts per-pass dict form `{"global": [...], "aic": [...], "aiv": [...]}` in addition to legacy flat list form.
- **Detection** (pre-build static check): if kernel file uses `REGIST_MATMUL_OBJ(...)` AND `build_overrides.json` declares only flat `per_source_defines` (no `aic` / `aiv` keys) AND no `-DSPLIT_CORE_*` in global compile flags → guaranteed runtime 507015 fault.
- **Evidence**:
  - 3_FusionAttention 2026-05-21 (independent prototype ar-2 KFC dispatch crash): AIC 507015 with REGIST_MATMUL_OBJ + single `-DASCENDC_MATMUL_AICORE`. Direct CANN-source read confirmed L262/322/368 branch logic.
  - probe_a5_v300_fa_sync 2026-05-23 (main A5 probe attempt 2): same Pattern B build pattern reproduces the L368 NO-SYNC fall-through on V351 — DEBT-20 per-source-define plumbing applies cross-arch (V220 + V351 both need per-pass defines).
  - **9.1.0 correction (FA-a3 DEBT-36 white-box, 2026-07-12)**: on CANN 9.1.0 (driver package `V100R001C11B060`), `SPLIT_CORE_CUBE` / `SPLIT_CORE_VEC` are **auto-derived** from the compiler arch predefines `__DAV_CUBE__` / `__DAV_VEC__` (defined by `sys_macros.h` per compile pass), so **manual `-DSPLIT_CORE_CUBE` / `-DSPLIT_CORE_VEC` is NOT needed** on 9.1.0 — the L262/L322 branches select correctly from the predefines alone. The per-pass-define fix above is **9.0.0-era**; keep it for 9.0.0 builds (harmless if still supplied on 9.1.0). On 9.1.0 the 507015 NO-SYNC fall-through would only recur if the arch predefines are somehow stripped from a pass.
- **Cross-ref**: PB-34 (Pattern A V220 sync conflict — different failure mode in same MIX_AIC_1_2 mode), OL-176 (matmul_intf.h non-transitive include — sibling DEBT-20 family), DEBT-20.1 implementation commit `66a4d985`.

---

### EC-59: Phase O5 re-measurement disagrees with worker `pass_a_runner.py` `n_pass` count → orchestrator infinite finalize→await_worker loop when worker iter_cap exhausted [V220+V351, port_a3_to_a5, orchestrator-FSM]

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=any-with-non-deterministic-quant-or-stale-binary`
`verified_on: flat_quant 2026-05-23 (kw-9 worker: tier1_pass=4/8; Phase O5 re-run: tier1_pass=0/8; 84 phase_o5_mismatch rollbacks in 37min)`

- **Symptom**: `.rollback_history.jsonl` accumulates many identical `phase_o5_mismatch` entries (signature `phase_o5_mismatch::await_worker`) within minutes. `state_transitions.jsonl` shows ping-pong `await_worker → finalize → await_worker → finalize → ...` with iter_counts.worker climbing past cap (e.g. 84/9 → 91/9). Orchestrator never terminates.
- **Root cause**: Two compounding bugs:
  1. **Worker `pass_a_runner.py` `n_pass` is not reproducible across runs**: the runner counts a case bit-exact only when BOTH `ok_out` AND `ok_qscale` are true with strict equality. For ops with marginal precision (qscale floor near bf16 ULP, non-deterministic quantization order, or stale-binary re-deploy), worker's first-run count and O5's re-measurement count can disagree (e.g., 4/8 vs 0/8) without the kernel being wrong — just non-deterministic at the bit-exact threshold.
  2. **Orchestrator finalize→O5 MISMATCH branch unconditionally routed to await_worker** even when worker iter_cap was already exhausted. Combined with the P0y "legitimate exhaustion → finalize" route at orchestrator.py:1349, this created an infinite loop: finalize → O5 MISMATCH → await_worker (cap exhausted) → finalize again. No state change between iters → loop never escapes.
- **Fix**:
  1. **FSM loop-guard (orchestrator.py P0bb-loop-guard, 2026-05-23)**: in both `O5 MISMATCH` and `O5 RUNNER_FAILED` branches, check `state_executor.at_iter_cap(workspace, "await_worker")` before routing back. If exhausted, log a FATAL diagnostic naming the four hypotheses (stale binary / different inputs / non-determinism / fabrication) and `return 2` instead of looping.
  2. **Dual-count schema (P0cc, 2026-05-23 — closes schema gap, not just the loop)**: `pass_a_runner.py` MUST emit BOTH `tier1_pass` (strict bit-exact count) AND `tier1_pass_inclusive` (T1+T2-within-tolerance count). `phase_o5.py` honors `tier1_pass_inclusive` when worker-declared `precision.pass_a.status` is in `("PASS_WITHIN_TOLERANCE", "PARTIAL_PASS_WITHIN_TOLERANCE", "PARTIAL_PASS")`. `kw_brief.py` port_a3 phase block instructs worker to emit both counts. `verification.json` MUST include `tier1_pass_inclusive` field for these statuses (assertion in pre-done checklist). Why dual-count (not just loop-guard): the loop-guard makes the orchestrator FAIL cleanly instead of loop forever, but it doesn't fix the underlying schema gap — next customer running a similar marginal-tolerance op still hits MISMATCH → FATAL (clean terminal but no archive). Dual-count lets the FSM correctly promote when worker's T2 verdicts are within tolerance AND O5's inclusive re-measurement agrees. Per `feedback_no_patch_fix_harness_for_next_customer.md`: the harness is the product, per-archive intervention is a patch.
- **Detection** (without watching the orchestrator):
  - `wc -l workspace/<op>/.rollback_history.jsonl` > 20 → loop suspected
  - `grep -c phase_o5_mismatch workspace/<op>/.rollback_history.jsonl` > 5 → confirmed loop
  - `tail .opgen_state.json` shows `invocation_count` growing without commit-on-disk advancing
- **Evidence**:
  - flat_quant 2026-05-23: worker kw-9 emitted real per-case verdicts (4 T1_BIT_EXACT + 4 T2_PASS_WITHIN_TOLERANCE deterministic), pass_a_results.json showed `n_pass=4`. Phase O5 ran the SAME pass_a_runner.py via SSH and got `n_pass=0`. Loop ran 84 times (3:40→4:17Z UTC) before user intervention.
  - 1_BatchMatmul (DEBT-206, 2026-07-13, Ascend910_9382 / CANN 9.1.0, independent generated-op verification): the dual-count/INCLUSIVE-status contract fired the SAME way outside the migration route. Worker kw-1 emitted `precision.pass_a.status="PARTIAL"` (NOT in `INCLUSIVE_STATUSES`) with strict `tier1_pass=46`, so O5's canonical normalizer (which emits the inclusive `46 T1 + 4 T2 = 50`) compared 46 vs 50 → 1 spurious `phase_o5_mismatch` rollback (P0kk). Fix (kw-2, kernel byte-identical): set `pass_a.status="PARTIAL_PASS_WITHIN_TOLERANCE"` (an INCLUSIVE_STATUS) + emit `tier1_pass_inclusive=50` → O5 compares the inclusive field-set → VERIFIED. The mode-neutral lesson is to declare an INCLUSIVE status whenever the canonical grader returns any `PASS_T2 > 0` alongside a genuine FAIL.
- **Cross-ref**: P0kk (Phase O5 post-verify, orchestrator.py:994-1037), P0y (legitimate pipeline exhaustion → finalize, orchestrator.py:1349), P0bb-loop-guard (this entry's fix landed 2026-05-23).

### EC-58: `matmul_intf.h` is NOT included transitively from `kernel_operator.h` — must be explicit `#include` OR per-pass interface-library propagation [V220+V351, ALL_MODES, build-include + KFC-sync-activation]

`applies_to: soc=Ascend910_9382 (V220) + Ascend950PR_9579 (V351); cann=9.0.0+; bisheng=ascendc.cmake DYNAMIC_MODE; op_class=mixed_aic_aiv_with_REGIST_MATMUL_OBJ`
`verified_on: a5_ops:3_FusionAttention 2026-05-21 + probe_a5_v300_fa_sync 2026-05-23 — both architectures need explicit matmul_intf.h plumbing`

- **Symptom**: Build fails with `error: ‘MatmulImpl’ has not been declared` OR `‘REGIST_MATMUL_OBJ’ does not name a type` even though the kernel `#include <kernel_operator.h>`. Adding `#include <matmul_intf.h>` (or letting the per-pass `_aic_intf_pub` / `_aiv_intf_pub` interface libraries propagate it) resolves it.
- **Root cause**: `kernel_operator.h` does NOT transitively include `matmul_intf.h`. The matmul interface is a separate header that ships with the AscendC SDK under `tikcpp/tikcfw/interface/matmul_intf.h`. ascendc.cmake's `legacy_modules/device_preprocess_project/CMakeLists.txt` exposes it via `${BUILD_MODE}_aic_intf_pub` / `${BUILD_MODE}_aiv_intf_pub` INTERFACE libraries, which `target_link_libraries(<sub-target> PUBLIC <intf_pub>)` propagates only when the sub-target opts in.
- **Fix** — pick ONE of:
  1. **Explicit `#include "matmul_intf.h"` in every kernel file using `MatmulImpl<>` / `REGIST_MATMUL_OBJ`** (simplest, no CMake changes; recommended for project-local kernels).
  2. **Per-pass interface-library propagation** (cleaner, only used when ascendc.cmake's full DYNAMIC_MODE pipeline is active): rely on the `_aic_intf_pub` / `_aiv_intf_pub` link chain. Caveat: if `build_ascendc.py` bypasses the legacy_modules path (e.g., NPUKernelBench's slim build), the interface libraries aren't created and Option 1 is mandatory.
- **Detection** (build-fail signature): exact error `error: ‘REGIST_MATMUL_OBJ’ does not name a type` OR `error: ‘MatmulImpl’ does not name a type` — search the file for `kernel_operator.h` include without adjacent `matmul_intf.h` include.
- **Evidence**:
  - 3_FusionAttention 2026-05-21: independent Pattern B build attempts failed with this exact error before adding explicit `#include "matmul_intf.h"`.
  - probe_a5_v300_fa_sync 2026-05-23: main's A5 Pattern B probe also tripped on this until explicit include was added.
- **Cross-ref**: EC-57 (sibling KFC sync activation issue — both share DEBT-20/20.1 family root cause), DEBT-20 follow-up notes in `66a4d985`.

### EC-60: ACLRT_LAUNCH_KERNEL with blockDim=0 causes ACL_ERROR_RT_PARAM_INVALID (107000) on V220

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all`

- **Severity**: CRITICAL (kernel never launches, but compilation succeeds silently)
- **Status**: CONFIRMED 2026-05-22 26_AvgPool3d a3-ds
- **Symptom**: Kernel compiles cleanly (static_check 10/10 PASS), but every launch fails with `ACL_ERROR_RT_PARAM_INVALID (107000)`.
- **Root cause**: `ACLRT_LAUNCH_KERNEL(kernel_name)(0, stream, args...)` passes blockDim=0. ACL runtime rejects it; kernel's `GetBlockNum()` returns 0, causing division-by-zero in per-block work distribution.
- **Fix**: Replace `ACLRT_LAUNCH_KERNEL` with explicit `extern "C"` declarations + dynamic nblk computation (floor at 1, cap at 56 for V220).
- **Detection**: grep for `ACLRT_LAUNCH_KERNEL.*\(0,` in pybind11.cpp.
- **Evidence**: 26_AvgPool3d a3-ds kw-1 (2026-05-22, Ascend910_9382 V220).
- **Cross-ref**: PB-28 (KERNEL_TASK_TYPE_DEFAULT is arch35-only — also produces 107000 on V220).

### EC-61: Scalar-pipe accumulator array (float[]) ~20× slower than VEC-pipe accumulator (TBuf<VECCALC>) for per-element reduction ops on V220

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=reduction (per-element accumulator)`

- **Severity**: PERFORMANCE (kernel works correctly but 20-50× slower than achievable)
- **Status**: CONFIRMED 2026-05-22 27_MaxPool3d a3-ds (0.047× perf ratio, scalar pipe bottleneck)
- **Symptom**: Kernel produces correct output but perf ratio < 0.1×. `aiv_vec_ratio` near 0, high scalar pipe utilization.
- **Root cause**: `float acc[N]` (S-pipe scalar array). Every `acc[i] = val` is a scalar store (1 element/cycle) vs VEC pipe (8+ elements/cycle for fp32).
- **Fix**: Replace `float acc_[TILE_W]` with `TBuf<TPosition::VECCALC> accBuf_`. Use VEC `Duplicate` for init, `Max` for accumulation, `Cast` + direct `DataCopy` for output.
- **Evidence**: 27_MaxPool3d a3-ds kw-1 (2026-05-22, Ascend910_9382 V220).
- **Cross-ref**: P-P47 (VEC halving for reductions); OL-161 (V220 SIMD UB element duplication).

### EC-62: TBuf workspace buffer used without `pipe_.InitBuffer()` — unallocated UB access → 507035 vector core exception on V220

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=all`

- **Severity**: CRITICAL (kernel crashes at runtime with 507035, compiles cleanly)
- **Status**: CONFIRMED 2026-05-23 1_RotaryMul a3-ds kw-1
- **Symptom**: Kernel compiles and links, crashes 507035 (subErrType:4, ADDR_MISALIGN) on every launch.
- **Root cause**: Worker init'd TQue buffers but forgot TBuf workspace InitBuffer. TBuf::Get<T>(n) on unallocated UB → hardware fault.
- **Fix**: Add `pipe_.InitBuffer(tbuf_name, size_bytes)` for every TBuf member. All TBufs must be explicitly initialized.
- **Detection**: grep `TBuf<.*> \w+_;` in kernel.h. For each, verify `pipe_.InitBuffer(name, ...)` exists. Now mandated by kw_brief Phase C self-audit.
- **Evidence**: 1_RotaryMul a3-ds kw-1 (2026-05-23, V220): 5 TBuf workspace buffers had no InitBuffer. Adding them fixed fp16+fp32 on A3 NPU0.
- **Cross-ref**: EC-60 (blockDim=0), EC-61 (scalar-pipe acc), PB-22 (DataCopy 32B limit). All four are "compiles, crashes" V220 classes kw_brief must preempt.

### EC-63: `std::string` in pybind11 crashes with `basic_string null` SIGSEGV on V220 ARM64 (bisheng/GCC ABI mismatch)

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all (pybind11 host code)`

- **Severity**: CRITICAL (kernel compiles, pybind11 loads, crashes at first call with SIGSEGV)
- **Status**: CONFIRMED 2026-05-22 28_Interpolate a3-ds
- **Symptom**: pybind11 module loads successfully, but any call to the wrapped function crashes with `basic_string null` SIGSEGV (exit code 139). Same binary works on x86 host, crashes on V220 ARM64.
- **Root cause**: GCC (host, x86) and bisheng (device, ARM64) have different std::string ABIs. The `std::string` parameter in pybind11 function signature is constructed by pybind11 from Python str, but the underlying memory layout differs between host and device.
- **Fix**: Replace `const std::string&` parameters with `const char*` or `py::str` in pybind11 wrappers. Convert to C string before passing to kernel launch.
- **Detection**: grep for `std::string` in kernel/pybind11.cpp. Any match → replace with C-string alternative.
- **Evidence**: 28_Interpolate a3-ds (2026-05-22, V220): `interpolate_forward(..., const std::string& mode_str, ...)` crashed with SIGSEGV 139 on A3 NPU0. Kernel compiled and built successfully.
- **Cross-ref**: OL-180 (CANN env init for .so loading) — both are pybind11-level V220-specific host issues.

### EC-64: `EVENT_ID4..7` / `PIPE_FIX` undefined identifier in host_bisheng_obj preview compile — wrap kernel-side includes with `#if defined(__CCE_AICORE__)` guard

`applies_to: soc=Ascend950PR_9579 (V351); cann=9.0.0; bisheng=15.0.5; op_class=port_a3_to_a5 (V220-pure entry wrapping kernels using 8-event double-buffer pipelines)`
`verified_on: flat_quant 2026-05-23 kw-1 — wrapping `op_kernel/flat_quant.cpp` via `kernel/flat_quant_kernels.cpp` thin TU. host_bisheng_obj pass failed; `#if defined(__CCE_AICORE__)` guard around the algorithm-header `#include` + `extern "C" __global__ __aicore__` body fixed it; device aic_obj / aiv_obj passes were unaffected.`

- **Severity**: BUILD-BREAK (build aborts at host preview stage; device passes succeed independently — easy to misdiagnose as "kernel is broken")
- **Status**: CONFIRMED 2026-05-23 (flat_quant kw-1)
- **Symptom** (verbatim from build log):
  ```
  error: use of undeclared identifier 'EVENT_ID4'
  error: use of undeclared identifier 'EVENT_ID5'
  error: use of undeclared identifier 'EVENT_ID6'
  error: use of undeclared identifier 'EVENT_ID7'
  error: use of undeclared identifier 'PIPE_FIX'
  ```
  Source location: a kernel-side `#include "<staged_algorithm>.h"` line in the worker-authored `kernels.cpp` TU.
- **Root cause**: NPUKernelBench's `ascendc_library` build runs a `host_bisheng_obj` preview compile pass on `kernels.cpp` to validate it parses in host-preview mode. That pass does NOT define `__NPU_ARCH__=3510` (only device passes do). CANN 9.0.0 `tools/tikicpulib/lib/include/stub_fun.h` only declares the extended `event_t` enumerators (`EVENT_ID4..7`) and the `PIPE_FIX` pipe-id under one of the device `__NPU_ARCH__` macros:
  ```c
  } event_t;
  #if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201)
    || (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510)
    || (__NPU_ARCH__ == 5102))
      EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7,
  #endif
  ```
  Worker `.cpp` whose `#include`d kernel header references `EVENT_ID4..7` / `PIPE_FIX` in class member-default-initializers (typical of the V351-style 8-deep `DEvent<Pipe1,Pipe2>{EVENT_ID4, EVENT_ID5}` double-buffer template) → preview pass fails.
- **Fix**: wrap kernel-side `#include`s AND each `extern "C" __global__ __aicore__` function body with `#if defined(__CCE_AICORE__) … #endif`. The host preview pass doesn't define `__CCE_AICORE__` either, so the guarded region drops out of preview while remaining identical in the device aic_obj / aiv_obj compile.
  ```cpp
  #if defined(__CCE_AICORE__)
  #include "flat_quant_vec.h"
  #include "flat_quant_cube.h"
  // ... staged algorithm headers
  extern "C" __global__ __aicore__ void flat_quant(GM_ADDR x, GM_ADDR scale, ...) {
      // kernel body
  }
  #endif
  ```
- **Anti-pattern**: defining `EVENT_ID4` / `PIPE_FIX` yourself as a workaround. They ARE defined in device passes; the issue is host preview strictness. Guards are the load-bearing fix.
- **Scope note**: this fires specifically on V220+/V351 kernels using `EVENT_ID4..7` (8-event pipelines). Older V220 kernels using only `EVENT_ID0..3` don't hit this — they remain identifier-clean under host preview.
- **Detection** (grep signature):
  ```
  [host_bisheng_obj] use of undeclared identifier 'EVENT_ID4'
  [host_bisheng_obj] use of undeclared identifier 'PIPE_FIX'
  ```
  Build aborts at host_bisheng stage; device aic_obj / aiv_obj succeed independently.
- **Evidence**: flat_quant 2026-05-23 kw-1 (Ascend950PR V351): wrapping `op_kernel/flat_quant.cpp` via thin worker TU triggered all 5 error variants above; single `#if defined(__CCE_AICORE__)` guard around `#include`+body resolved on 2nd build, then PASS 8/8 T1 BIT_EXACT + 2.24× perf.
- **Extension — same guard rule applies to the regbase MicroAPI surface (2026-06-23, selective_scan_source_a5 perf-loop iter-2, Ascend950PR_957b, CANN 9.1.0.B060)**: a regbase `__simd_vf__` body + `using namespace AscendC::MicroAPI` ALSO requires the `#if defined(__CCE_AICORE__)` guard — same root cause (host preview pass lacks the device-only symbol surface), different symptom. On this CANN the host_bisheng preview pass does NOT expose `AscendC::MicroAPI` (`RegTensor`/`LoadAlign`/`StoreAlign` are absent from `include/ascendc/basic_api`; provided ONLY on the device pass). Unguarded → `error: expected namespace name` / `'MicroAPI' is not a namespace-name` on the host pass; guard wrap fixes it (device aic/aiv passes unaffected, identical to the EVENT_ID4 case above). Three companion build facts on this CANN, recorded for any regbase author: (1) `kernel_basic_intf.h` that the FA wholeport VF `#include`s is NOT shipped here — the FA wholeport VF would not build as-is; the MicroAPI surface is device-pass-only. (2) fp32 `VL = 64` (production `floatRepSize=64`), NOT the migration-doc's 128 — a regbase mask/tile loop sized for 128 mis-tiles fp32. (3) A regbase broadcast SCRATCH buffer must be sized to the full element count being broadcast (`lnElems`/CN), NOT a smaller chunk buffer (`LBUF_`/CH) — too-small → silent UB overflow → NaN (a control test with membase ops on the same rewiring also NaN'd, isolating the bug to the BUFFER, not the VF).
- **Cross-ref**: OL-132 (Mode A vs Mode B), OL-185 (V220→V351 port calibration anchor), OL-245 (regbase-default + its amortization boundary; the regbase VF this extension guards), `.upstream_prestaged.json` worker-authored dispatcher TU pattern.

### EC-65: V220 `op_kernel/*_base.h` `#define bfloat16_t int16_t` fallback silently disables bf16 on A5 — author local A5-safe copy under `workspace/kernel/`

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=norm-quant-port_a3-V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

- **Symptom**: A5 port of V220-pure port_a3 norm/quant op compiles + builds clean but bf16 inputs silently produce wrong results (or fp16-truncated results). Surface error message often absent — output values just don't match reference. Probe shows downstream casts producing int16-like values instead of bfloat16.
- **Root cause**: Upstream V220 `op_kernel/*_base.h` (norm and quant family base headers) contains a fallback `#define bfloat16_t int16_t` near the top, used when V220 toolchain bfloat16 support is absent. On A5 (Ascend950PR) with CANN 9.0.0 + bisheng that natively supports bfloat16, this fallback is wrong — bfloat16 should be the native type, not int16_t aliased. The fallback `#define` shadows the real type at compile time; downstream `Cast<bfloat16_t, ...>` becomes `Cast<int16_t, ...>`, silently producing wrong bit-patterns.
- **Affected**: port_a3 V220-pure norm/quant family ports (RMSNorm, LayerNorm, quant variants) that include `op_kernel/<op>_base.h` from upstream verbatim.
- **Workaround**: Author a local A5-safe copy of the base header under `workspace/<op>/kernel/` (e.g., `<op>_base_a5.h`) with the fallback `#define` removed, and `#include` the local copy instead of the upstream V220 header. Do NOT modify the upstream submodule.
- **Status**: OPEN (upstream-V220-side compatibility fallback that is hostile to A5)
- **Cross-ref**: KB_INDEX EC-1..EC-65 row; OL-185 V220→V351 calibration anchor.

### EC-66: Wrapper input materialization required — kernel reads zero on layout-conversion tensors (mechanism provisional)

`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=fa_class`
`source: DS 2026-05-27`

**Symptoms**:
- AscendC kernel output = all zeros for non-trivial inputs
- Adding `.abs().max().item()` or `print()` on input tensors fixes it
- Only affects non-identity layouts (BSH→BNSD, SBH→BNSD, BSND→BNSD)
- BNSD native layout works correctly (no async copy needed)
- Small shapes (Sq=2, Skv=2) more likely to trigger; large shapes may pass by timing luck

**Root cause**: NOT a simple cross-stream race. 2026-05-27 main C++ verify (198.51.100.92, CANN 9.0.0) tested both `aclrtSynchronizeStream` and `aclrtSynchronizeDevice` before pybind kernel launch — **neither fixed** the zero-output. Device-wide sync should have flushed any pending async copy regardless of stream. This rules out stream-ordering as the mechanism.

**Leading hypothesis (NOT confirmed)**: Python→pybind boundary crossing triggers tensor materialization that pure sync (Python or C++) and C++-side reads do not. The load-bearing invariant is: wrapper-side tensor op on inputs, before pybind call, at Python level. Same op moved into C++ pybind = INEFFECTIVE (main matrix #3).

**REFUTED hypotheses**: stream-ordering race (same-stream in source), sync (Python + C++ all INEFFECTIVE), generic device→host read (C++ read ineffective), record_stream (UNSTABLE 1-3/9), after-launch sync (INEFFECTIVE).

**2026-05-27 verify data** (DS + main + independent prototype, npu-a3@198.51.100.92):
| Method | Verdict |
|---|---|
| Baseline (no fix) | 2/9 (degenerate only); non-BNSD cand=0 |
| +aclrtSynchronizeStream before launch | INEFFECTIVE |
| +aclrtSynchronizeDevice before launch | INEFFECTIVE |
| Python `torch.npu.synchronize()` only (no read) | INEFFECTIVE (non-BNSD still zero) |
| Python `.abs().max().item()` read on inputs | Bug A resolved (9/9 non-zero) |

**Mechanism** (provisional, narrowed): A **wrapper-side** tensor op on the input tensors before the pybind call is required. The same read moved inside C++ pybind does NOT work (main matrix #3: `q.abs().max().to(CPU).item()` inside pybind = INEFFECTIVE). Pure `torch.npu.synchronize()` in wrapper also INEFFECTIVE. The load-bearing element is specifically "a Python-level operation on the input tensors at the wrapper boundary, before crossing into pybind" — not generic read, not sync. Candidate explanation: Python→pybind boundary crossing triggers tensor data materialization that pure stream sync and C++-side reads do not.

**Why cv-agent's pybind isn't the fix**: cv-agent stock pybind is byte-identical in stream handling — same `getCurrentNPUStream().stream(false)`, same `storage().data()` read, zero explicit sync. It works on its stock 16 cases by shape-timing luck (large shapes give async copy enough time to land). Copying cv-agent's pybind verbatim copies the same latent race.

**Confirmed fix (2026-05-27, main C++ verify matrix on npu-a3@92)**:
In the Python **wrapper** (`model_new_ascendc.py`), after `_to_bnsd()` and BEFORE the pybind kernel call, force tensor materialization:
```python
# After _to_bnsd, before kernel call:
_ = q_bnsd.abs().max().item()
_ = k_bnsd.abs().max().item()
_ = v_bnsd.abs().max().item()
```
This is 100% deterministic across runs. Fix lives in **wrapper emission**, NOT pybind C++.

**REFUTED (all pybind-side C++ approaches, main 2026-05-27 NPU 0 9-case verify)**:
- `aclrtSynchronizeStream` before launch: INEFFECTIVE
- `aclrtSynchronizeDevice` before launch: INEFFECTIVE
- C++ `.abs().max().to(CPU).item()` inside pybind: INEFFECTIVE
- `aclrtSynchronizeStream` AFTER launch: INEFFECTIVE
- `recordStream` for all tensors: UNSTABLE (1-3/9 across runs)
- device-sync before + recordStream after: INEFFECTIVE

**STEP_1.4 translator acceptance check**: verify emitted `model_new_ascendc.py` contains tensor materialization (e.g., `.abs().max().item()` or equivalent) after layout conversion and before kernel call. This is a wrapper-emission rule, not a pybind-emission rule.

**STEP_1.4 translator acceptance check**: grep emitted `model_new_ascendc.py` for input materialization (`.abs().max().item()` or equivalent device→host sync) after layout conversion and before pybind kernel call. Absence → `translator_block: wrapper_no_input_materialize`. This is a **wrapper-emission** rule (model_new_ascendc.py), NOT a pybind-emission rule.

**Detection**:
- **Bug pattern**: `model_new_ascendc.py` calls `_to_bnsd()` to convert layouts, immediately passes output to pybind kernel without any sync/materialize step. Kernel on small shapes reads zero.
- **Absence check**: grep `model_new_ascendc.py` for `.abs().max().item()` or `torch.npu.synchronize()` between `_to_bnsd` and the pybind kernel call. Absence = probable Bug A (wrapper-side fix needed).
- **NOT a pybind bug**: pybind11.cpp source shows kernel runs on `getCurrentNPUStream()` (same stream as `.contiguous()`). Cross-stream race is NOT the mechanism (confirmed by independent source read 2026-05-27).

**Evidence**: (B) autonomous chain emit on npu-a3@198.51.100.92, CANN 9.0.0, B=2/S=2/N=16/D=16 BSH fp16. DS + independent prototype Python-level materialization confirmed Bug A resolved (3-agent convergence). Main C++ verify matrix (2026-05-27, NPU 0, 7 variants tested): pybind-side approaches all REFUTED; wrapper-side Python `.abs().max().item()` after `_to_bnsd` = ONLY working fix (deterministic 9/9 non-zero across runs). 6/9 cases remain Bug-B-wrong (kernel=3.17-3.44 vs ref=2.67-3.10 — systematic-high, independent Bug B).

**Cross-ref**: R1-R4 host-API rules (PR #210/#212/#214), STEP_1.4 translator pre-build gate (PR #198), Bug B compute-diag (softmax/cube sub-16 alignment).

### EC-67: `__VEC_SCOPE__` for-loop induction variable MUST be `uint16_t`

```yaml
applies_to:
  paradigm: ascendc
  arch_family: arch35
  bisheng: 2026-03-21+
```

- **Empirical anchor**: HW probe sub-agent `acd7700cc8182c637` 2026-05-28 20:14Z on Ascend950PR_9579 / arch35.

- **Symptom**: bisheng emits compile error inside `__VEC_SCOPE__ { ... }` block when the for-loop induction variable is anything other than `uint16_t`:
  ```
  regbase_probe_kernel.cpp:69:18: error: Induction variable must have a type uint16_t. Example:
   (uint16_t $var=0; $var< bound; $var++))
              for (int32_t i = 0; i < loops; ++i) {
  ```

- **Root cause**: bisheng's BuiltIn-API legality checker on the SIMD VF surface treats the loop counter as a `MaskReg`/`AddrReg` index source. `RegTensor` / `MaskReg` index registers are 16-bit by hardware design. Wider integer types fail the legality check.

- **Fix**: rewrite the loop to use `uint16_t`. Example:
  ```cpp
  // BAD — bisheng rejects
  __VEC_SCOPE__ {
      for (int32_t i = 0; i < repeatTimes; ++i) { /* ... */ }
  }

  // GOOD — bisheng accepts
  __VEC_SCOPE__ {
      for (uint16_t i = 0; i < repeatTimes; ++i) { /* ... */ }
  }
  ```
  The induction-variable rule applies ONLY inside `__VEC_SCOPE__`. Regular `__aicore__` code outside the scope can use any int type.

- **Scope qualifier**: bisheng version sealed at 2026-03-21 build; future bisheng may relax the constraint. Applies to arch35 / V351x / Ascend950PR family only — V351x (Atlas 200I/500 A2) does not surface `__VEC_SCOPE__` per its arch spec doc.

- **Detection (worker / translator emit-time)**: grep emitted kernel for `__VEC_SCOPE__` block, then for any `for (` line inside it. If the induction declaration is not exactly `uint16_t`, fail the local emit-time check.

- **Cross-ref**: OL-196 (Membase vs Regbase + `__VEC_SCOPE__` programming entry).

---

### EC-68: A kernel that owns its workspace via `GetUserWorkspace` under an `ACLRT_LAUNCH` host stub must call `SetSysWorkspaceForce(workspaceGM)` FIRST, or temporaries land out-of-range → MTE "DDR addr out of range" err95 (`507015`)
`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; kernel_type=ACLRT_LAUNCH+GetUserWorkspace`
`verified_on: soc=Ascend950PR; cann=9.1.T500 — chunk_gated_delta_rule (GDN) light-port 2026-06-15 (122/122 T1 PASS after fix)`

- **Symptom**: a kernel launched through an `ACLRT_LAUNCH` host stub (host-stub-generated entry, NOT the full CANN GE op-build path) that obtains its own scratch GM via `GetUserWorkspace(workspaceGM)` aborts at runtime with an AIV MTE fault `"DDR addr out of range"` / error 95, host return code `507015`. The abort is FAST (not a hang) and fires the instant the kernel writes its first temporary, before any compute progress.
- **Root cause**: an `ACLRT_LAUNCH` host stub passes `workspaceGM` as a plain kernel argument but does NOT set the sys-workspace base. `GetUserWorkspace(ws)` does NOT use the pointer you hand it — it returns `g_sysWorkspaceReserved + RESERVED_WORKSPACE` (the 16MB sys region the runtime auto-allocates), NOT your large allocation. So the kernel writes temporaries into the runtime's small auto sys-workspace and runs off its end → DDR out-of-range. (Compare: the full CANN GE op-build path sets the sys-workspace base for you; the bare `ACLRT_LAUNCH` stub does not.)
- **Fix**: call `SetSysWorkspaceForce(workspaceGM)` as the FIRST statement of the kernel — before `TPipe` construction, before any `matmul`/`MatmulImpl` `Init`. Then `GetUserWorkspace` returns `workspaceGM + 16MB` and the matmul KFC path uses the same base. The host side must allocate ONE workspace sized `16MB(sys) + interWorkspaceSz + stageWorkspaceSz` and pass its base as `workspaceGM`.
  ```cpp
  extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, /*...*/ GM_ADDR workspaceGM, GM_ADDR tiling) {
      AscendC::SetSysWorkspaceForce(workspaceGM);   // FIRST — before TPipe / matmul Init
      GM_ADDR userWs = AscendC::GetUserWorkspace(workspaceGM);  // now = workspaceGM + 16MB
      // ... TPipe pipe; mm.Init(...); ...
  }
  ```
- **Detection**: grep the kernel for `GetUserWorkspace(` with NO preceding `SetSysWorkspaceForce(` in the same `__global__` body, when the launch path is `ACLRT_LAUNCH` (not GE op-build). Runtime smoking gun: clean build/launch, then `507015` / err95 "DDR addr out of range" on the first temporary write.
- **Note**: this also explains the earlier A5 FA-sync probe's identical err95 — that probe never set the sys-workspace base either; its "transpose org-shape" hypothesis was a red herring, the real cause was the missing sys-workspace base.
- **Cross-ref**: EC-60 (`ACLRT_LAUNCH_KERNEL blockDim=0`), CAND-KFC-standalone-bootstrap-teardown (the `SetSysWorkspaceForce` + `REGIST_MATMUL_OBJ` standalone-KFC bootstrap), PB-34 (the GDN light-port that surfaced this).

### EC-69: Variable names `AT`, `BT`, `CT`, `WT` collide with AscendC builtin enums in `cce_aicore_intrinsics.h`

`applies_to: soc=all; cann=all; bisheng=all; op_class=all`

- **Symptom**: compile error when a local variable, template parameter, or struct member is named `AT`, `BT`, `CT`, or `WT`. The error typically surfaces as an ambiguous reference or type-mismatch in code that otherwise looks valid — e.g. `error: expected ';' before '=' token` or `error: 'AT' was not declared in this scope` inside a function that included AscendC headers.

- **Root cause**: `cce_aicore_intrinsics.h` (included transitively by `kernel_operator.h`) defines `AT`, `BT`, `CT`, `WT` as enum values or macros in the global scope. Any user code that uses these names for variables or template parameters silently collides. The collision is namespace-less — both the builtin enum and the user symbol occupy the same name.

- **Fix**: rename user symbols away from the 4 reserved names. Recommended replacements: `MTA` (for A-transpose flag), `MTB` (for B-transpose flag), `MTC` (for C-type flag), `MTW` (for weight/workspace flag). Alternatives: prefix with a namespace-like tag (e.g. `kAT`, `kBT`).

```cpp
// BAD — collides with cce_aicore_intrinsics.h enum
bool AT = false, BT = true, CT = false, WT = false;

// GOOD — renamed away from the reserved names
bool MTA = false, MTB = true, MTC = false, MTW = false;
```

- **Detection**: grep kernel source for `\b(AT|BT|CT|WT)\b` used as a variable/parameter name (not inside a string or comment). The 4-letter token in a declaration context is the smoking gun.

- **Evidence**:
  - fused_quant_mat_mul kw-1 (2026-06-15): compile failed at `kernel/fused_quant_mat_mul_kernel.h:36` — `AT`, `BT`, `CT`, `WT` as bool flags collided with intrinsics enum. Renamed to `MTA`/`MTB`/`MTC`/`MTW` → compile passed.

### EC-70: `TPipe::InitBuffer(TQue<T>&)` requires 3 arguments — different from TBuf InitBuffer which takes 2

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 chip family — CANN version may differ)`

- **Symptom**: compile error when calling `pipe_.InitBuffer(tQue)` with only 2 arguments. Error message: `too few arguments to function call, expected 3` or similar.

- **Root cause**: `TPipe::InitBuffer` has different signatures for TQue vs TBuf:
  - `InitBuffer(TBuf<T>&, uint8_t num_buffers)` — 2 args
  - `InitBuffer(TQue<T>&, uint8_t num_buffers, uint32_t len_per_buffer)` — 3 args

  The TQue variant requires an explicit per-buffer element length (in bytes) as the third argument. Workers familiar with the TBuf 2-arg form naturally write the TQue form with 2 args → compile fails.

- **Fix**: always provide the third argument for TQue InitBuffer. The `len_per_buffer` is the byte size of one buffer slot: typically `tileLength * sizeof(dtype)`.

```cpp
// TBuf — 2 args (OK)
pipe_.InitBuffer(tBuf, depth);

// TQue — 3 args REQUIRED
pipe_.InitBuffer(tQue, depth, tileLength * sizeof(half));
```

- **Detection**: grep for `InitBuffer(` calls near TQue variable declarations. If the call has exactly 2 args and the first arg is a `TQue<...>`, the third arg is missing.

- **Evidence**:
  - fused_quant_mat_mul kw-1 (2026-06-15): `kernel/fused_quant_mat_mul_kernels.cpp:69-70` — TQue InitBuffer called with 2 args. Added `BUF_LEN * sizeof(half)` as third arg → compile passed.

### EC-71: `GET_TILING_DATA` macro unresolved (`use of undeclared identifier 'tilingData'`) in a GENERATED arch35 kernel compiled by the workspace verify-build → define a POD TilingData struct before the algorithm `#include` + load the GM tiling blob with the FA `CopyTiling<T>` byte-copy helper

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.1.T500`

**Symptom**: a workspace-side verify-build (NPUKernelBench / `build_ascendc.py` — the per-op pybind/`ACLRT_LAUNCH_KERNEL` build, NOT the on-host CANN ops-nn-build pipeline) of a *generated* arch35 kernel fails with `error: use of undeclared identifier 'tilingData'` at the `GET_TILING_DATA(...)` call site.

**Root cause**: `GET_TILING_DATA` resolves only through the synthesized tiling-data header that the CANN ops-nn-build pipeline emits from the `BEGIN_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS` macros in `op_host/<op>_tiling.h`. The workspace verify-build does NOT process op_host tiling registration, so the macro (and the `tilingData` symbol it expands to) does not exist in that translation unit.

This is distinct from EC-52 (apt.cpp mistakenly `#include`s `<op>_tiling.h` → file not found) and EC-54 (PR4778 ship artifacts misplaced under `kernel/` → move them to `op_kernel/`). Here the kernel is *meant* to build in the verify-build TU and genuinely needs the tiling fields there — so neither "remove the include" nor "move to op_kernel/" applies.

**Fix** (the FA-proven portable substitute): define a POD TilingData struct mirroring the tiling fields, place it in the build TU **before** the algorithm `#include`, and byte-copy the GM tiling blob into a stack instance with a `CopyTiling<T>` helper (reinterpret the GM blob int32-wise into the stack POD):

```cpp
struct MyTilingData { int32_t f0; int32_t f1; /* ... mirror op_host layout ... */ };

template <typename T>
__aicore__ inline void CopyTiling(T* dst, GM_ADDR tilingGM) {
    auto src = reinterpret_cast<__gm__ int32_t*>(tilingGM);
    auto d   = reinterpret_cast<int32_t*>(dst);
    for (uint32_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) d[i] = src[i];
}
// in the entry, BEFORE using fields:
MyTilingData td; CopyTiling(&td, tilingGM);
#include "<op>_algorithm.h"   // POD must be defined ABOVE this include
```

**Evidence**: recurrent_gated_delta_rule kw-1 iter-1 (2026-06-18, port_a3_to_a5, A5 Ascend950PR_957b, CANN 9.1.T500): generated AIV-only recurrent-decode kernel tripped `undeclared identifier tilingData` on the first verify-build; adopting the FA `CopyTiling<T>` byte-copy helper + POD-TilingData-before-include let the build proceed and the kernel reached 30/30 T1 PASS. The pattern is the FA `kernel_common.h` `CopyTiling<T>` helper reused unchanged.

**Other instances (predicted)**: any *generated* (non-ship-artifact) arch35 / port_a3 kernel that needs tiling fields and is compiled by the workspace verify-build rather than the ops-nn-build pipeline — recurrent / SSM / linear-attention family especially, where the kernel is authored from arch22 algorithm source.

**Cross-reference**: EC-52 (include-resolution variant), EC-54 (layout-misplacement variant), EC-68 (ACLRT_LAUNCH workspace base — same verify-build/host-stub class), OL-141 (L1 mechanical port).

---
### EC-72: `FixpipeParams<float>` member-field form (`fp.nSize=`/`fp.mSize=`) does NOT compile on V220 (arch22) — use the positional `FixpipeParamsV220(...)` ctor + templated `Fixpipe<dstT, srcT, CFG_ROW_MAJOR>` call

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=cube-l0c-to-gm`
`verified_on: soc=Ascend910_V220; cann=9.0.0`
`unverified_on: soc=Ascend950PR (arch35 has the OPPOSITE constraint — see note below)`

**Symptom**: a cube kernel staging an L0C(fp32) → GM(fp32) fixpipe writes the params with the member-assignment form illustrated in some reference docs (`FixpipeParams<float> fp; fp.nSize = ...; fp.mSize = ...;`) and fails to compile on V220 — the templated `FixpipeParams<T>` struct does not expose `nSize`/`mSize` members on arch22/2201.

**Root cause**: on V220 the FIX-pipe descriptor is the dedicated struct `FixpipeParamsV220` (`kernel_struct_fixpipe.h`), constructed positionally — NOT a generic templated `FixpipeParams<T>` with assignable fields. The member-field form is an arch35-flavored illustration that does not exist on arch22.

**Fix** (V220 L0C→GM ND, fp32→fp32, no cast):
```cpp
// V220 (arch22): positional ctor + templated Fixpipe call
Fixpipe<float, float, CFG_ROW_MAJOR>(
    gmDst, l0cSrc,
    FixpipeParamsV220(nSize, mSize, srcStride, dstStride, /*reluEn=*/false));
```
`CFG_ROW_MAJOR` selects the L0C(fp32)→GM(fp32) ND (row-major) layout. After the rewrite the kernel TU compiles (build PASS).

**arch35 has the OPPOSITE constraint** (do NOT copy this V220 form to A5): on arch35/950PR the L0C→GM fixpipe MUST use `FixpipeParamsC310` (NZ2ND) with an explicit `quantPre` cast mode (`F322BF16` for float→bf16, `F322F16` for float→half). The arch22 `FixpipeParamsV220` (no cast) raises device error 169 subErrType 0x4 on arch35. See `patterns/domains/fa_class/templates/op_kernel/matmul_tile.h` (2026-06-16 arch35 fixpipe note).

**Evidence**: fa_gqa_grad kw-1 iter-1 (2026-06-19, port_a3_to_a5, Ascend910_V220/arch22, CANN 9.0.0): a hand-`Mmad` cube GEMM staging L0C→GM hit the no-such-member compile failure on the doc's member-field form; rewriting to `FixpipeParamsV220(nSize, mSize, srcStride, dstStride, reluEn)` + `Fixpipe<float,float,CFG_ROW_MAJOR>(...)` → build PASS.

**Other instances (predicted)**: any V220 cube kernel that drains an L0C accumulator to GM via FIX — hand-written `Mmad` GEMM ladders (FA-class fwd/bwd, GroupedMatmul, custom matmul), regardless of op class.

**Cross-reference**: A-P34 (`KERNEL_TYPE_*_ONLY` arch-guard — same V220-vs-arch35 entry-form divergence class), `patterns/unverified/candidates.md` (FixpipeParamsV220 opaque-field workflow), `patterns/domains/fa_class/templates/op_kernel/matmul_tile.h` (arch35 `FixpipeParamsC310` complementary note).

---
### EC-73: A scalar read (`GetValue` / raw `__gm__` deref) of data just `DataCopyPad`'d into UB needs `HardEvent::MTE2_S` — `MTE2_V` alone leaves the scalar pipe racing the DMA → garbage indices → OOB GM access → silent hang
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

---
### EC-74: A device helper called from a `__simt_vf__` kernel must be `__simt_callee__` on CANN ≥9.1.T500 — plain `__aicore__ inline` (which EC-1 prescribes) no longer suffices
<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=simt-l3`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: cann<9.1.T500 (older CANN compiled the same code with plain `__aicore__ inline` helpers — see 28_Interpolate)`

- **Error pattern**:
  ```
  note: candidate function not viable: simt_vf function can only call simt_callee function
  ```
  (the call site inside the `__simt_vf__` kernel fails to resolve the helper; the helper itself is already `__aicore__ inline`, so EC-1's fix is already in place yet the build still fails)
- **Root cause**: CANN ≥9.1.T500 tightened the SIMT calling convention — a function invoked from a `__simt_vf__` kernel must carry the `__simt_callee__` marker, not merely `__aicore__`. EC-1 (`__aicore__` on helpers) is necessary but no longer sufficient on this CANN. The upstream arch35 SIMT source already marks every helper `__simt_callee__ __aicore__ __attribute__((always_inline))`; older CANN versions (e.g. the one that compiled the finalized 28_Interpolate) accepted plain `__aicore__ inline` helpers, masking the requirement.
- **Fix**: add `__simt_callee__` (keep `__aicore__`; `__attribute__((always_inline))` recommended to match upstream) to EVERY device helper reachable from a `__simt_vf__` kernel:
  ```cpp
  // BEFORE (fails on CANN 9.1.T500: "simt_vf function can only call simt_callee function"):
  template <typename T>
  __aicore__ inline float gs_to_float(T v) { return static_cast<float>(v); }

  // AFTER (compiles):
  template <typename T>
  __simt_callee__ __aicore__ __attribute__((always_inline)) inline float gs_to_float(T v) { return static_cast<float>(v); }
  ```
- **Evidence**: grid_sample port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500): all VF-called helpers (`gs_to_float`/`gs_from_float`/`gs_fetch`/`gs_clip`/...) initially `__aicore__ inline` → compile failed with the `simt_callee` note; one compile iter to add `__simt_callee__ __attribute__((always_inline))` to each → build PASS, 29/29 T1 precision.
- **Other instances (predicted)**: any greenfield or ported SIMT (L3) kernel on CANN ≥9.1.T500 whose `__simt_vf__` body calls per-thread scalar helpers (gather/scatter index math, coordinate clamp, dtype cast helpers). When porting an older SIMT scaffold (e.g. 28_Interpolate) forward to 9.1.T500, expect this even though the original compiled clean.
- **Related**: EC-1 (the prior, weaker `__aicore__`-on-helper requirement this supersedes for SIMT VF callees), OL-150 (SIMT programming model — `__simt_vf__`/`LAUNCH_BOUND`/`Simt::VF_CALL`), OL-151 (SIMT helper APIs).

### EC-75: `.ascendc_env` `A5_CANN_PATH` pointed at the toolkit symlink-root (only `latest` + `set_env.sh` symlinks, no `tools/tikcpp`) → cmake `ascendc_kernel_cmake does not exist` — point it at the complete `.../ascend-toolkit/latest`
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: `build_ascendc.py` (verify-only pybind/ACLRT_LAUNCH build) fails at cmake configure with a missing-directory error naming `ascendc_kernel_cmake` (e.g. `ascendc_kernel_cmake does not exist` / `add_subdirectory given source ".../ascendc_kernel_cmake" which is not an existing directory`). The build does not even reach a compile step.

**Root cause**: the `A5_CANN_PATH` value in `workspace/.ascendc_env` was set to a CANN *symlink-root* directory that contains only the `latest` symlink and `set_env.sh` — NOT the actual toolkit tree. `build_ascendc.py` resolves the AscendC cmake module relative to `A5_CANN_PATH` (expects `tools/tikcpp/ascendc_kernel_cmake` under it), but a symlink-root has no `tools/`, so the path resolves to a non-existent directory.

**Fix**: set `A5_CANN_PATH` to the **complete toolkit directory** `.../ascend-toolkit/latest` (the resolved toolkit, which actually contains `tools/tikcpp/ascendc_kernel_cmake`), not the parent symlink-root. Verify with:
```bash
ls "$A5_CANN_PATH/tools/tikcpp/ascendc_kernel_cmake" || echo "EC-75: A5_CANN_PATH is not a complete toolkit"
```

**Distinct from EC-51**: EC-51 is `ASCEND_CANN_PACKAGE_PATH` unset → `find_package(ASC)` fails in the on-host ops-nn-build pipeline. EC-75 is the *workspace verify-build* `A5_CANN_PATH` config var pointing at the wrong (symlink-root) directory. Same `ascendc_kernel_cmake` token can appear in the trace, but the misconfigured variable and the fix are independent.

**Evidence**: iou_v2 kw-1 (2026-06-21, port_a3_to_a5, A5 Ascend950PR): `A5_CANN_PATH` at the toolkit symlink-root → cmake `ascendc_kernel_cmake does not exist`; repointing to `.../ascend-toolkit/latest` resolved it. (Same session also corrected two non-cmake `.ascendc_env`/lane setup issues: `A5_DEPLOY_STAGE_HOST/_CONTAINER` held host-IP/container-name where directory paths belong → silent empty deploy; and lane6 `BENCHMARK_ROOT` was missing `utils/build_ascendc.py` → `setup_lanes.sh` should copy `utils/` when provisioning a lane.)

**Other instances (predicted)**: any fresh A5 host bring-up or new lane where `.ascendc_env` is hand-filled — `A5_CANN_PATH` is the most error-prone field because the symlink-root and the resolved toolkit dir look interchangeable but only the latter has `tools/tikcpp`. Add the `ls .../tools/tikcpp/ascendc_kernel_cmake` check to lane/host preflight.

**Cross-reference**: EC-51 (different `ascendc_kernel_cmake`/ASC cmake failure — ops-nn-build pipeline, `ASCEND_CANN_PACKAGE_PATH`), OL-234 (build-only A5 host needs a complete-CANN runtime container — adjacent host-setup gotcha).

### EC-76: A `__VEC_SCOPE__` regbase VF that hardcodes its element/tile count to a fixed inner-dim constant produces garbage above that constant — derive the count from the RUNTIME inner dim
`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all`

A regbase MicroAPI VF that processes a `[outer, inner]` chunk must compute its loop/tile count from the **runtime** inner dim, not a value that happened to equal the inner dim on the shape it was first written for. A VF written for a fixed `inner=16` and coded as `ntFull = ceil(outer*16 / VL)` only covers the first 16 inner-lanes; for `inner>16` the upper `outer*(inner-16)` elements are NEVER written → the consumer reads uninitialized UB → garbage output. The bug is **masked whenever runtime inner == the hardcoded constant** (e.g. the original `dstate=16` customer), so it ships silently and only surfaces when a later caller uses a larger inner dim.

**Concrete anchor** (selective_scan fwd regbase build/prodC VFs): `uint16_t ntFull = (uint16_t)((cl * N + VLf - 1) / VLf);` — `N` is the runtime `dstate`, NOT a literal `16`. Tail over-process (when `cl*N` is not a multiple of VL=64) only touches the +64 buffer padding → identical semantics.

**Evidence**: selective_scan fwd-SIMD (2026-06-24, PR #52). `cl*16` → N=32 upper half garbage; `cl*N` → N∈{8,16,24,32,48,64} all 0-wrong at dtype floor. N=16 (customer) was correct either way (16==N).

**Other instances (predicted)**: any regbase/MicroAPI VF parameterized over a runtime inner/state/head dim — attention head-dim VFs, normalization feature-dim VFs, any `[L, D]` chunk VF — where an early single-shape author bakes the first D as a literal.

### EC-77: A chunked scan's cross-chunk carry fold needs a FULL `PipeBarrier<PIPE_ALL>` (not `PIPE_V`) before the wide intra-chunk scan reads it — at large state width the carry write has not drained
`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=scan`

When an L-chunked scan carries recurrence state across chunk boundaries by folding the incoming carry into position 0 of the chunk buffer (`Add(Bscan[0:N], dA[0:N]*xst)`) and then runs a WIDE parallel (Hillis-Steele) scan that reads that buffer, a `PipeBarrier<PIPE_V>` between the fold and the scan is **insufficient at large state width N** → deterministic carry corruption. Distinctive signature: **chunk 0 is correct** (its carry-in is zero, the fold is a no-op) while **chunk 1+ are wrong**; the corruption is deterministic and N-dependent (masked at small N where the working set drains within pipeline time); and a pure-fp64 simulation of the exact algorithm is CORRECT (so it is an on-device execution/fence issue, not an algorithm bug). Fix = `PipeBarrier<PIPE_ALL>` at the fold.

**Method note (saves a dead end)**: a sub-granule VEC-offset-alignment hypothesis (HS `off = stride*N` non-64-aligned at N>16) was A/B-REFUTED by a contiguous control; the real cause was the RAW fence. When chunk0-right/chunk1-wrong + algorithm-numpy-correct + N-only, suspect the carry-path fence before suspecting alignment.

**Evidence**: selective_scan fwd-SIMD L-chunk (2026-06-24, PR #52). N=32 multi-chunk (L≥257) deterministic-wrong with `PIPE_V`; `PIPE_ALL` at the fold → N∈{32,64} multi-chunk all 0-wrong. Sibling of the cross-row / cross-iteration V→MTE2 fences (PB-47, PB-49).

**Other instances (predicted)**: any chunked/tiled scan or recurrence that folds a cross-tile carry into a buffer immediately consumed by a wide multi-pass vector op — prefix-sum tiling, segmented scan, attention online-softmax running-stat carry.


### EC-78: SIMD elementwise kernel segfault (exit 139) with blockDim > 1 on A5 — multi-core-first diagnostic, NBLK=1 as fallback

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise,fused-elementwise`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A2/A3 — elementwise SIMD multi-core may work on A3; A5-specific trigger unconfirmed)`
`UPDATED 2026-06-24`: Originally observed as "NBLK=1 required" (swi_glu V1). Revised to "multi-core-first diagnostic" after V5 proved GetBlockIdx() outer_blocks partitioning works correctly and delivers 2.11× geo_mean improvement.

- **Symptom**: SIMD elementwise kernel launched with `blockDim=56` (data-parallel — all cores run identical code on identical data) crashes with exit 139. **Distinct from** a kernel that uses `GetBlockIdx()` to partition independent outer_blocks across cores — that pattern works correctly (see OL-254).
- **Root cause**: The crashing pattern uses `blockDim=N` data-parallel launch (all N cores run the SAME tile loop on the SAME data — no per-core work partition). This triggers a launch-time crash on A5 CANN 9.0.0. The crash is specific to data-parallel (identical-work) multi-core launch, NOT to multi-core in general.
- **Fix (diagnostic)**:
  1. FIRST verify the kernel uses GetBlockIdx() to partition work (see OL-254). If NOT — add per-core partitioning. This is the preferred fix and enables multi-core speedup.
  2. If the kernel ALREADY uses GetBlockIdx() partitioning AND still crashes → set `NBLK=1` as a diagnostic probe. If NBLK=1 fixes it, the bug is in the partition logic.
  3. NBLK=1 as a PERMANENT choice ONLY when outer_blocks = 1 (physically cannot partition).
- **Detection signal**: kernel compiles clean, pybind launches, process exits 139 immediately (before any kernel-side computation). If the crash is at launch time (not during kernel execution), suspect this.
- **Evidence**:
  - swi_glu V1 (2026-06-23, Ascend950PR_957b CANN 9.0.0): data-parallel blockDim=56 → segfault. NBLK=1 resolved → 50/50 bit-exact PASS, but perf 0.72× geo_mean with 0.07-0.16× on large shapes.
  - swi_glu V5 (2026-06-24, same hardware): GetBlockIdx() outer_blocks partitioning with nblk=32 → 50/50 bit-exact PASS, perf 1.52× geo_mean (2.11× vs V1), 41/50 faster-than-ref. Large-shape geo_mean 1.35× (vs V1 0.16× = 8.5× improvement). Proves multi-core elementwise IS correct and performant when using per-core work partition.
- **Cross-ref**: EC-17 (nblk=1 for sub-alignment chunk overwrite — different root cause), OL-214 (single-core-first testing methodology — NBLK=1 as diagnostic probe), OL-254 (multi-core outer_blocks partitioning — the preferred pattern), P-P114 (multi-core outer_blocks template).

### EC-79: DataCopy with raw `__gm__` pointer silently returns zeros on Ascend950PR pure-AIV kernels

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (any pure-AIV SIMD kernel using DataCopy for GM->UB reads)`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 — GM->UB DataCopy with raw pointers on V220 unconfirmed)`

- **Error pattern**: `DataCopy(local, reinterpret_cast<__gm__ float*>(gm_addr), byte_count)` compiles cleanly (no warning, no error) but at runtime the destination `local` tensor contains **all zeros** — the DataCopy reads zero from GM instead of the actual tensor data. All downstream computation produces zero/near-zero output. No runtime error, no 507035 — silent data corruption.
- **Root cause**: On Ascend950PR, `DataCopy` (the high-level AscendC API) expects a `GlobalTensor<T>` (or `LocalTensor<T>` for UB->UB) as the source/destination argument, not a raw `__gm__ T*` pointer. When passed a `reinterpret_cast<__gm__ T*>(gm_addr)`, the template resolves to a path that does not actually perform the GM->UB DMA — it produces zero-filled output without any diagnostic. The raw `__gm__` pointer path works in SIMT VF kernels (scalar pipe) but fails silently in pure-AIV SIMD class kernels.
- **Fix**: Use `GlobalTensor<T>` with `SetGlobalBuffer` + `operator[]` for all GM addressing in pure-AIV kernels:
  ```cpp
  // BEFORE (silently returns zeros):
  __gm__ float* gm_ptr = reinterpret_cast<__gm__ float*>(gm_addr);
  DataCopy(local, gm_ptr, N);  // compiles, reads zeros

  // AFTER (works correctly):
  GlobalTensor<float> gm_tensor;
  gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_addr));
  DataCopy(local, gm_tensor[offset], N);  // correct GM->UB DMA
  ```
  This is the proven pattern from FusedAddRmsnorm and other pure-AIV class kernels.
- **Distinct from PB-20**: PB-20 covers GM WRITE paths (`GlobalTensor::SetValue` silent no-op + raw `__gm__` pointer writes failing in pure-AIV). EC-79 covers the GM->UB READ path via `DataCopy` with a raw pointer — different API call, different direction, same root class (raw pointer vs GlobalTensor).
- **Detection**: kernel compiles and produces **all-zero output across ALL cases** (not just edge cases). If the entire output tensor is zero/epsilon and the kernel uses `DataCopy` with `reinterpret_cast<__gm__ T*>` arguments, suspect this first. A one-line probe: replace one `DataCopy(local, raw_gm_ptr, N)` with `DataCopy(local, gm_tensor[0], N)` and check if output becomes non-zero.
- **Evidence**: add_rms_norm_quant (2026-06-23, Ascend950PR_9579, CANN 9.0.0): aog-precision-probe iter 0 — kernel produced all-zero x_out and y1 across all 196 cases. Switching to GlobalTensor + SetGlobalBuffer + operator[] fixed immediately. FusedAddRmsnorm (earlier) used the GlobalTensor pattern from the start — no such issue.
- **Cross-reference**: PB-20 (GlobalTensor::SetValue silent no-op — same raw-pointer-vs-GlobalTensor class but WRITE direction), OL-77 (byte-copy loop workaround for reinterpret_cast GM->non-GM — adjacent API-surface mismatch).

---
### EC-80: `Duplicate(dst, src, count)` broadcasts a 32B BLOCK (8 fp32 / 16 fp16), NOT a single scalar — compiles clean but produces garbage output
<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise,normalization`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A2/A3 — Duplicate semantics are general AscendC API, not arch-specific)`

- **Error pattern**: kernel compiles and launches clean. At runtime, output values are completely wrong — e.g., a GroupNorm+SiLU kernel where all D>1 output slices are identical (matching only the last d slice's reference), or scalar-broadcast produces garbage ~1e5× the expected magnitude.
- **Root cause**: `Duplicate(dst, src, count)` replicates an entire **32-byte BLOCK** (8 fp32 elements, 16 fp16/bf16 elements) at a time, NOT a single scalar. For fp32, `Duplicate(dst, scalar_src, 1)` copies 8 consecutive fp32 values starting from the `scalar_src` UB address — the intended scalar plus 7 garbage neighbors. When used in a per-d loop to broadcast per-channel scale/bias, every output slice gets the same block of garbage data.
- **Fix** (scalar broadcast alternatives):
  ```cpp
  // BEFORE (WRONG — Duplicate broadcasts 8 fp32 values, not 1):
  Duplicate(scaleLocal, gammaLocal[d], 1);   // copies gamma[d] + 7 garbage neighbors

  // AFTER (correct — SIMD Brcb scalar broadcast):
  // Option A: Brcb (block-broadcast scalar to all vector lanes)
  Brcb(scaleLocal, gammaLocal[d], hwNumAligned_);

  // Option B: Inline scalar × vector (preferred when you already have a Mul/Add op):
  float scale = gammaFp32.GetValue(c);
  Muls(tmpFp32[ubOff], xFp32[ubOff], scale, hwNum_);  // scalar × vector in one op
  // No Duplicate needed — AscendC SIMD Mul/Add accept scalar right-hand-side directly.
  ```
- **Key insight**: AscendC SIMD VEC ops (`Muls`, `Adds`, `Divs`, `Subs`, `Mins`, `Maxs`) accept a **scalar** second operand directly — no need to broadcast the scalar into a tensor first. The `Duplicate` API exists for replicating multi-element BLOCKS (e.g., duplicating a row of weights across a batch), not for scalar→vector conversion.
- **Evidence**: group_norm_silu precision fix (2026-06-26, A5/Ascend950PR, CANN 9.0.0): D>1 mode used `Duplicate` to broadcast per-channel gamma/beta scalars into UB tensors. All D output slices were identical to the last d's reference because Duplicate copied 8-element blocks containing garbage. Fixed by removing Duplicate entirely and using inline `Muls`/`Adds` with float scalars directly.
- **Other instances (predicted)**: any kernel that uses `Duplicate` intending to broadcast a single scalar — per-channel affine transforms, bias-add loops, per-head attention scaling, per-expert MoE gating. The fix is always: either use SIMD VEC ops with scalar RHS (no broadcast needed), or if a full tensor IS needed, use `Brcb` for proper scalar-to-vector broadcast.
- **Related**: OL-260 (member shadowing in Init() — the SAME group_norm_silu session had BOTH bugs; Duplicate was a red herring once the shadowing fix was in), P-P4 (Dynamic block size), `ASCENDC_API_CATALOG.md` (Duplicate API signature).


### EC-81: `SToMTE3Sync()` after V-pipe Cast/Muls/Adds → deterministic garbage on V351 — must use `VToMTE3Sync()`
<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=all (any vec-side kernel storing fp16/bf16 results to GM after V-pipe compute)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A2/A3 — V220 likely has implicit V→S forwarding that masks the wrong sync)`

- **Error pattern**: fp32 outputs are fine (bit-exact to reference), but fp16 and bf16 outputs are **deterministic garbage** — fp16 max_abs_error ~7.3e+0 (completely wrong), bf16 max_abs_error ~4.8e+5 (overflow-scale). The error is consistent across runs (same input → same wrong output). Only affects half-precision paths; fp32 path (which skips Cast) is unaffected.
- **Root cause**: On A5/V351, the `Cast` intrinsic writes its result to UB via the V (vector) pipe. To transfer this data to GM via MTE3 (`DataCopyPad`), the sync primitive must come from the **V pipe** (`VToMTE3Sync()`), NOT from the scalar pipe (`SToMTE3Sync()`). The `SToMTE3Sync()` synchronizes the S pipe with MTE3, but the Cast result hasn't been drained from V yet → MTE3 reads **stale UB contents** (prior-iteration leftover or uninitialized). On V220, this may have been benign due to implicit V→S forwarding inside the chip, but V351 enforces strict pipe separation.
- **Fix**:
  ```cpp
  // BEFORE (V220-compatible but V351-broken):
  Cast(halfOut, fp32Buf, RoundMode::CAST_ROUND, count);
  SToMTE3Sync();                   // Scalar→MTE3 sync — WRONG pipe
  DataCopyPad(gmOut[offset], halfOut, count);

  // AFTER (V220+V351 correct):
  Cast(halfOut, fp32Buf, RoundMode::CAST_ROUND, count);
  PipeBarrier<PIPE_V>();           // drain V pipe first
  VToMTE3Sync();                   // V→MTE3 sync — CORRECT pipe
  DataCopyPad(gmOut[offset], halfOut, count);
  ```
- **Key insight**: The sync primitive naming follows `{srcPipe}ToMTE3Sync()` — `SToMTE3Sync` syncs S→MTE3, `VToMTE3Sync` syncs V→MTE3. A `Cast` (or `Muls`/`Adds`/any VEC op) output lives on V. Using `SToMTE3Sync` to guard the MTE3 store after a VEC op is always wrong — the correct primitive is `VToMTE3Sync` + `PipeBarrier<PIPE_V>`.
- **Bisect method**: if fp32 path passes but fp16/bf16 produce garbage on V351, immediately grep for `SToMTE3Sync` at every site following a `Cast` / `Muls` / `Adds` / VEC op. The fp32 path typically skips the Cast (direct DataCopyPad), so the wrong-sync half/bf16 path is the only one with the Cast→SToMTE3Sync→DataCopyPad chain.
- **Distinction from existing Cast+V-pipe OL**: the broader OL entry (Cast→PipeBarrier<PIPE_V>→DataCopy) diagnoses `PipeBarrier<PIPE_V>` as insufficient because it doesn't sync V→MTE3. This EC-81 is the lower-level variant: `SToMTE3Sync()` is the explicitly-wrong primitive (different pipe), not just an insufficient barrier. Both produce the same class of failure (stale UB → GM), but the grep target differs: search for `SToMTE3Sync` after any VEC op, not just `PipeBarrier<PIPE_V>`.
- **Evidence**: ctc_loss_v3 port_a3_to_a5 (2026-06-25, A5/V351, CANN 9.0.0): `CopyOutNegLogLikelihood` and `CopyOutAlphaTensor` used `Cast→SToMTE3Sync→DataCopyPad` for half/bf16 paths. fp32 path (no Cast) was fine. Fix: `VToMTE3Sync+PipeBarrier<PIPE_V>` — fp16 went from max_abs=7.31 (garbage) to max_abs=0.068 (within fp16 tolerance); bf16 went from max_abs=4.8e+5 (garbage) to max_abs=0.57 (within bf16 tolerance for non-T≥100 cases). 15/15 PASS post-fix.
- **Other instances (predicted)**: ANY V220→V351 port that has `Cast(half, fp32) → SToMTE3Sync → DataCopyPad` — this is the most natural V220 code pattern (V220's S pipe might implicitly forward V results). Ports from V220 `op_kernel/*.h` where the output epilogue contains half/bf16 Cast-to-GM paths. Grep for `SToMTE3Sync` in any port_a3_to_a5 workspace kernel source.
- **Related**: unnumbered OL (Cast→PipeBarrier<PIPE_V>→DataCopy produces garbage on V351 — same root class, different API surface); OL-260 (member shadowing — the ctc_loss_v3 session also had both bugs).

---
### EC-82: V220/CANN 8.5.1 `TPipe::InitBuffer(TQue&, depth)` 2-arg form does not compile — use the 3-arg `(TQue&, depth, buf_size_bytes)` form

`applies_to: soc=Ascend910_V220; cann=8.5.1; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910B2C (V220); cann=8.5.1`

- **Symptom**: build error `no matching member function for call to 'InitBuffer'` on a `pipe->InitBuffer(que, depth)` (2-arg) call that compiles fine on newer CANN.
- **Fix**: pass the explicit buffer size as the third argument:
  ```cpp
  // BAD on V220/CANN 8.5.1 — 2-arg overload absent
  pipe->InitBuffer(inQueue_, PIPE_DEPTH);
  // GOOD — 3-arg form
  pipe->InitBuffer(inQueue_, PIPE_DEPTH, TILE_ELEMS * sizeof(float));
  ```
- **Scope**: distinct from EC-62 (TBuf with NO `InitBuffer` at all → 507035 vector core exception). Here the call IS present; only the arg count is wrong. The 2-arg overload may exist on newer CANN — this is the V220/8.5.1 fallback. Applies to `TQue` buffers; `TBuf` workspace uses the 2-arg `(name, size_bytes)` form as usual (EC-62).
- **Detection**: build log shows `no matching member function for call to 'InitBuffer'` pointing at a `InitBuffer(<TQue>, <int>)` line. Fix by adding the byte-size third arg.
- **Evidence**: 1_GELU kw-1 re-spawn (2026-06-23, Ascend910B2C V220, CANN 8.5.1): the 2-arg form was rejected by the compiler; switching to the 3-arg form cleared the build.
- **Cross-ref**: EC-62 (missing InitBuffer → 507035), OL-63 (TQue depth=4 for elementwise bandwidth), P-P28 (TQue depth + explicit buf-size pairing).


### EC-83: bare `is_same<>` unresolved in a standalone bisheng/ccec kernel TU despite `using namespace AscendC` — use `std::is_same` + `#include <type_traits>`
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

### EC-85: cannbot `grade_batch` returns numpy scalar types + full per-case arrays → `TypeError: not JSON serializable` and a 100s-of-MB verdict — coerce recursively to native + trim to a scalar summary before persisting
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (grading-harness plumbing, SoC-independent)`
`verified_on: soc=Ascend950PR (selective_scan_full_grad kw-1 2026-06-18 / A5)`

- **Symptom**: persisting `verification.json` after grading a batch through cannbot `grade_batch` raises `TypeError: Object of type bool_ / float64 is not JSON serializable`; separately, the raw grade result embeds full per-case arrays that bloat the verdict file (observed ~329 MB) with data that does not belong in a verdict artifact.
- **Root cause**: `grade_batch` returns numpy scalar types (`np.bool_`, `np.float64`) and full per-case numpy arrays inside its result dict; the stdlib `json` encoder cannot serialize numpy scalars, and the arrays are unbounded in size.
- **Fix**: (1) recursively coerce the result with a `_native()` helper before `json.dump` — `np.generic → .item()`, `np.ndarray → summary stats or drop`, recurse into `dict`/`list`; (2) trim per-case arrays to a scalar summary (pass/total counts + representative stats), not the raw arrays.
- **Detection**: a `json.dump(grade_result)` on a cannbot output raising the numpy-type `TypeError`, or a `verification.json` in the 100s-of-MB range.
- **Cross-ref**: OL-272 (backward-mode pybind build/deploy — same class of backward-mode harness plumbing carve-outs the worker must handle itself). backend=ascendc.

### EC-86: uninitialized `RunInfo<true>` (isInfer) fields read UNCONDITIONALLY in `CalcS1Coord` → stack-garbage query/kv-row shift — LATENT in the shared fa_class template until an inference-mode op instantiates it; fix = struct-default at declaration (one point beats per-consumption-site `if constexpr` guards)
`applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.x; op_class=fa_class (isInfer=true path — MLA / sparse-FA / any inference-mode FA); kernel_type=fa_class template (wholeport)`
`verified_on: static source-trace vs latest main e3e4b051 (2026-07-24); originally hit at runtime in the SFA forward a3→a5 port (2026-06-22, case-0 +1 query-row shift, branch scan/home commits 89a5241e→ce372ccb)`

- **Symptom**: an **inference-mode** FA-class op (a kernel instantiated with `isInfer=true` — MLA, sparse-FA, decode) produces a **+1 query-row (or kv-row) coordinate shift** → wrong output, and it is **build-fragile**: one lucky build passes, an independent clean rebuild fails (classic uninitialized-read signature). A pure forward/training FA op (`isInfer=false`) does NOT exhibit it.
- **Root cause**: in `fa_class/.../op_kernel/wholeport/wp_util_regbase.h`, `struct RunInfo<true>` (the isInfer specialization) declares four coordinate fields with **NO default initializer**: `preTokensPerBatch` / `nextTokensPerBatch` (via the `COMMON_RUN_INFO` macro, `wp_util_regbase.h:177-178`) and `queryLeftPaddingSize` / `kvLeftPaddingSize` (`wp_util_regbase.h:201-202`). `CalcS1Coord` (`wp_block_cube.h:493/495/506`) then reads all four **unconditionally** to offset `s1Coord`/`s2Coord`. But the ONLY assignment (`wp_kernel_train.h:69-70`) is fenced under `if constexpr (hasAtten) { if ASCEND_IS_AIV { … } }` and covers **only** `preTokens/nextTokens` — `queryLeftPaddingSize`/`kvLeftPaddingSize` are **NEVER assigned anywhere in the template tree**. So on the AIC (or any `hasAtten=false` inference op), those reads are pure stack garbage → garbage coordinate offset.
- **Why LATENT in current main**: as of `e3e4b051` **no op in the repo instantiates `isInfer=true`** — `RunInfo<true>` appears only as its struct definition (`wp_util_regbase.h:196`), so the shared template ships this bug dormant. It bites the FIRST inference-mode FA op built on the template (that is exactly how the SFA a3→a5 port surfaced it).
- **Fix**: give the fields a default **at the struct/macro declaration** — `int64_t queryLeftPaddingSize = 0;` etc. (and `preTokensPerBatch`/`nextTokensPerBatch` a semantically-correct default). One declaration-site default covers **every** consumption site at once and is codegen-neutral. This **supersedes** the tempting per-consumption-site `if constexpr (hasAtten)` guard (which is fragile: it has to be replicated at every read and misses the AIC / hasAtten=false paths — the SFA port's partial guard `89a5241e` was itself superseded by the struct-default `ce372ccb`).
- **Detection**: grep the fa_class wholeport template for `RunInfo<true>` / `COMMON_RUN_INFO` fields that are read in `CalcS1Coord`/attenmask/pse but lack a declaration default; specifically flag any struct field read unconditionally on a template branch that is only assigned under an `if constexpr(...)` guard. **Two INDEPENDENT clean builds expose it where within-build determinism does not** — a single build can leave lucky-zero stack, so require bit-identical output across two fresh builds (a sharper form of the "verify the md5 of the source the build actually compiled" rule, OL — statically-verified-build family).
- **Evidence**: SFA forward a3→a5 port (2026-06-22, Ascend950PR, `scan/home` branch): case-0 query-row shift traced to these uninitialized fields; per-site `if constexpr` guard (`89a5241e`) was partial → struct-default-init of 4 fields (`ce372ccb`) grounded a committed-byte 6/6 floor-PASS. Re-verified structurally against latest main `e3e4b051` (2026-07-24): the four no-default declarations + unconditional reads + never-assigned `query/kvLeftPaddingSize` are all still present in the shipped fa_class template. **The SFA operator itself was never merged (superseded), but this latent template hazard is live in main today.**
- **Cross-ref**: EC-36 (`Cast<T,T>` no-op uninit family), EC-37 (K2 workspace uninit), OL-161 (conditional-write tail uninit) — same uninitialized-read class, but none cover the "struct-declared-no-default field read on a not-yet-instantiated `isInfer=true` template branch"; `patterns/domains/fa_class_template.md` K1/K2 (host-rule↔kernel-instance consistency — this is the RunInfo-field analogue on the coordinate path). backend=ascendc.
