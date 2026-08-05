---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: "Ascend910C (DaVinci V220, dual-die) — Target Platform"
description: "Ascend910C (DaVinci V220, dual-die) — Target Platform"
confidence: single_run
original_id: hw/ascend910c
timestamp_inferred: true
tags: [hardware, target, ascend910c]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
# Ascend910C (DaVinci V220, dual-die) — Target Platform

> A3 platform = two `Ascend910B` dies in one package. CANN treats it as a member
> of the `ascend910b_list` family (NOT a separate `ascend910c_list`); compiler
> selects `BUILD_MODE=c220` and arch dir `arch22`, same as A2/910B.
>
> **SIMD-only.** No SIMT (no `threadIdx`, no `Simt::` namespace, no
> `WarpShflSync`/`ThreadBarrier`/`LAUNCH_BOUND`). Kernels written for A5 (950PR)
> in SIMT mode will NOT compile for A3 — they must be rewritten using the
> `TPipe / TQue / DataCopy` SIMD pattern.
>
> Verified data on 192.0.2.210 container `npu-a3` (CANN 9.0.0_beta.2).
> Many memory/atomic numbers below are inherited from CANN source +
> public references and **need empirical confirmation on actual A3 hardware**
> (probe queue: `references/hardware/INTERNAL_QUERY_QUEUE.md`).

## SOC version reference

| Variant | SOC string | Notes |
|---------|-----------|-------|
| 9392 | `Ascend910_9392` | **The chip on our verified A3 server** (npu-smi `Ascend910`) |
| Other 93xx | `Ascend910_9391`, `Ascend910_9381`, `Ascend910_9382`, `Ascend910_9372`, `Ascend910_9362` | Other A3 sub-models — same `c220` build path |

`acl.get_soc_name()` returns the active SOC string at runtime — query before
choosing build flags.

## Compute (per package = 2 dies)

| Parameter | Value | Source |
|-----------|-------|--------|
| AICore total | **40 ~ 48** (= 2× 20-24 per die) | comparison doc; verify with npu-smi |
| AIV per AICore | 2 | CANN `GetAicAivTaskRation()` |
| **AIV total** | **80 ~ 96** | AICore × 2 |
| AIC total (Cube) | 40 ~ 48 | AICore × 1 |
| **Programming model** | **SIMD only** | DavidV100 manual §25.1.1; arch22 has no SIMT path |
| `threadIdx` / `blockDim` (SIMT) | **NOT available** | arch22 lacks SIMT path |
| `Simt::` namespace | **NOT available** | (ditto) |
| Warp / LAUNCH_BOUND | **N/A** | (ditto) |
| `GetBlockNum()` / `GetBlockIdx()` | available (block-level only) | universal AscendC |
| Block scheduling | Software time-slicing | shared with A2/A5 family |
| Clock frequency | TBD | needs verification |

### What WORKS on A3 (SIMD path)

- `TPipe`, `TQue<QuePosition::VECIN/VECOUT>`, `TBuf`
- `DataCopy(LocalTensor, GlobalTensor, len)`
- Vector ops: `Add`, `Mul`, `Muls`, `Adds`, `Cast`, `Exp`, `Reduce*` (SIMD form)
- Cube ops: `Mmad` family
- Block-level parallelism via `GetBlockNum() / GetBlockIdx()`
- Atomic ops on int32/uint32 (CAS, EXCH)

### What FAILS on A3 (SIMT-only patterns)

- Any `Simt::WarpShflSync`, `Simt::WarpReduceAddSync`, etc.
- `ThreadBarrier()`, `ThreadFence()`
- `__syncthreads()` equivalents
- Per-thread allocation patterns relying on `threadIdx.x`
- `LAUNCH_BOUND(...)` annotations

If an arch35 implementation uses warp shuffle, scatter-add via per-thread atomics,
or block-wide barriers, the A3 port MUST replace those patterns with SIMD
equivalents (e.g. block-wide reduction via UB scratchpad + DataCopy, or
sort-then-segment-reduce instead of unsorted scatter-add).

## Memory (per package)

| Parameter | Value | Source |
|-----------|-------|--------|
| HBM type | HBM2e / HBM3 (mixed by SKU) | public; comparison doc |
| **HBM capacity** | **128 GB** (dual-die total; varies by SKU) | comparison doc |
| **HBM bandwidth (peak)** | **~1.6 TB/s** (sum of two dies) | comparison doc |
| **L2 cache** | ~64 MB per die (TBD if shared cross-die) | comparison doc; **needs probe** |
| **UB / AIV** | **192 KB** | CANN `hardware.h` (V220 default) |
| UB / AICore | 384 KB (= 2 × 192 KB) | derived |
| **L1 Buffer / AIC** | **512 KB** | CANN `arch.h` L1_SIZE (universal) |
| L0A Buffer / AIC | 64 KB | CANN `arch.h` |
| L0B Buffer / AIC | 64 KB | CANN `arch.h` |
| **L0C Buffer / AIC** | **128 KB** (vs A5's 256 KB — half size!) | CANN `arch.h`: `__NPU_ARCH__==3510`→256KB, else→128KB (**was "3101", corrected 2026-06-18 vs arch.h:29**) |
| **BT (bias table) / AIC** | **1 KB (1024)** | cannbot npu-arch §2.3 DAV_2201 `bt_size` (A5/950PR=4KB — differs) |
| **Structured sparsity 4:2** | **supported** (`sparsity=1`) | cannbot npu-arch §2.3 DAV_2201 (A5/950PR `sparsity=0` — NOT supported; don't assume 4:2 on A5) |
| Per-AICore AXI | TBD (probe required) | comparison doc |
| Cross-die interconnect | TBD | comparison doc |

> **Cross-platform delta vs A5:** UB is 192 KB (not 256 KB) and L0C is 128 KB
> (not 256 KB). Tiling code that assumes 256 KB UB **will OOM at compile time**
> on A3. Always derive tile sizes from `GetUBSizeInBytes()` (runtime) or
> a `target` parameter (compile-time).

## Atomics

| Operation | Status | Source |
|-----------|--------|--------|
| `AtomicCas` U32/S32 | Supported | universal V220 |
| atomicAdd S32/U32 | Supported | universal V220 |
| **atomicAdd FP32** | **TBD — needs probe** | comparison doc; not yet verified |
| **atomicAdd FP16** | **TBD — needs probe** | comparison doc; high-risk for FP16 scatter-add ports |
| **atomicAdd BF16** | **TBD — needs probe** | comparison doc |
| atomicMax / atomicMin FP32 | TBD — needs probe | comparison doc |

> **Scatter-add risk:** A5 has FP16/BF16 atomicAdd. If A3 lacks them, ports of
> source scatter-add backwards (e.g. SparseGather) need a cast-to-FP32 wrapper or
> a sort-then-reduce restructure. **First A3 port that touches scatter-add
> MUST run a probe to confirm.**

## Pipelines

A3 inherits the V220 pipeline structure (CUBE, VEC, MTE1, MTE2, MTE3, FIXP +
independent Scalar Unit). The A5 doc's pipeline guidance applies *except*:

- No SIMT pipeline / SIMT DCache / SIMT shared memory (those are A5-only)
- Cube path (MMAD) and Vector path (SIMD) are present and behave like A5

## Cross-platform porting summary

| What | A5 (950PR) | A3 (910C) |
|------|-----------|-----------|
| Programming model | SIMT + SIMD | **SIMD only** |
| `__NPU_ARCH__` | 3510 | 2201 |
| arch dir | arch35 | arch22 |
| AIV count | 56 | 80 ~ 96 |
| UB per AIV | 256 KB | **192 KB** |
| L0C per AIC | 256 KB | **128 KB** |
| FP16/BF16 atomicAdd | supported | **TBD** |
| HBM bandwidth | 1.6 TB/s | ~1.6 TB/s (dual-die sum) |
| Block scheduling | software time-slicing | software time-slicing |

## Build flags (CMake)

```cmake
# In your project's CMakeLists.txt:
set(SOC_VERSION "Ascend910_9392" CACHE STRING "...")  # or other 93xx variant
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann" CACHE STRING "...")
# CANN 9.0 layout — note the new aarch64-linux/ prefix:
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
```

The CANN 8.x path (`compiler/tikcpp/...`) does NOT exist in CANN 9.0 — the
multi-target CMakeLists.txt searches both layouts and picks the one that exists.

## Open verification questions (PROBE QUEUE)

1. FP16/BF16 atomicAdd support — write a kernel that calls atomicAdd on a FP16
   GM tensor and verify result vs reference (probe `Q_atomic_fp16_a3.md`)
2. L2 cache total size and cross-die sharing semantics
3. Cross-die HBM access latency / bandwidth penalty
4. Actual frequency, register file size per AIV
5. SIMD vector unit width (likely 256B × 2, same as A5/A2; needs confirmation)

## References

- `docs/design/PLUGIN_PARADIGM_NOTES.md#ascend-chip-comparison` — authoritative A2/A3/A5 comparison
- CANN source (`code_channel_infer.py`, `arch.h`, `hardware.h`)
- HiAscend AscendC manual 9.0.0 (Chinese version) — see TODO probe queue for
  extracted A3-specific items
- `~/workspace/a3/doc/Hi1980B&CV100 Davinci Cloud芯片 用户指南_01.docx`
  (硬件指南) and `Hi1980B&CV100 Davinci Cloud芯片 硬件指南_01.docx` —
  primary hardware doc (Chinese)

<!-- 迁移自 porter kb/hardware/target/ascend910c.md（convert_hardware_to_okf.py，硬件事实→reference/hardware）。 -->
