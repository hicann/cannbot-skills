---
applies_to: soc=all
reason: Most launch/build practices (grid count, loop unroll, benchmark setup) are universal. Patterns that cite SIMT-specific `LAUNCH_BOUND` or `threadIdx`-based block shaping must declare per-pattern `chip_scope: a5-only` inline; default file-level scope is `all`.
---

# Domain: Kernel Launch & Build Practices
> Patterns for launch bounds, grid configuration, loop unrolling, and benchmark methodology.
> Load when: Analyzer detects kernel launch configuration, pragma unroll, or benchmark setup code.

---

## Patterns

### P-P5: LAUNCH_BOUND + LAUNCH_CHECK

**Severity**: Medium

`LAUNCH_BOUND(1024)` >= the maximum number of threads any dispatcher may launch. Use `LAUNCH_CHECK` to inspect the return value of each launch.

---

### P-P6: grid_y and thread count consistency

**Severity**: Medium

The host-side grid_y computation must match the actual thread count used by the dispatcher. Inconsistency between the two causes incorrect work distribution.

---

### P-P7: #pragma unroll scope

**Severity**: Low

Only use on small loops with a compile-time-inferrable upper bound. Loops containing `WarpReduceAddSync + ThreadBarrier` must not be unrolled.

---

### P-P8: Host benchmark best practice

**Severity**: Low

Warm up 3 times, time 10 NPU iterations and take the mean, compare precision
against audited CPU truth, then run boundary tests (`edges=0`, `dim=1`, `dim=3`).

---

### P-P9: SIMD vs SIMT selection

**Severity**: High | **Updated**: 2026-04-02 Batch 14 SIMT/SIMD crossover experiment

**Core rule**: **Prefer SIMD unless scatter-write (atomicAdd) is required**

| Scenario | Choice | Reason |
|------|------|------|
| scatter-write (atomicAdd to random addresses) | **SIMT** | SIMD's SetAtomicAdd requires alignment and is functionally limited |
| indirect-read + weighted sum (e.g., SG Forward) | **SIMD** | DataCopy block transfer + 4-pipeline parallelism |
| Contiguous aligned read/write | **SIMD** | Natural pipeline-overlap advantage |
| scatter-read + scatter-write mixed | **SIMT** | SIMD cannot orchestrate when both ends are irregular |

**~~Old rule~~ (deprecated)**: ~~"indirect indexing / random access -> SIMT"~~

**Reason for deprecation (msprof evidence)**: SG Forward has indirect addressing (each token reads a different expert), but the expert rows themselves are contiguous in memory. SIMD DataCopy batch-transports expert rows to UB (via MTE2), which is 2-7x faster than SIMT thread-scalar scattered reads. SIMT's VEC pipe gets stalled by GM read latency (vec=0.95+, mte2=0.000); SIMD runs 4 pipes simultaneously (vec+scl+mte2+mte3 each at 30-90%).

**Key insight**: "Indirect addressing" must distinguish the addressing layer from the data layer. SG Forward's addressing layer is indirect (which expert), but the data layer is contiguous (expert rows are contiguous memory). SIMD's DataCopy handles the data layer while scalar computes the addressing. indirect-read != must-be-SIMT.

**SIMT architecture limitation**: In SIMT mode, GM access only uses the VEC pipe's load/store units; the MTE2/MTE3 DMA engines do not participate. 4-pipeline parallelism is unachievable — this is a hardware limitation, not a code problem.

---

### P-P19: Kernel development must include UT (boundary dim + sorted/original consistency)

**Severity**: High | **Source**: Expert B6 feedback (accum[12] overflow not caught by tests)

Every new kernel or kernel variant must have a corresponding CPU reference test covering:
1. **Large-dim boundary**: dim=512, 1024, 4096 (validates BRE=512 path + accum fallback)
2. **Sorted/unsorted consistency**: element-wise comparison of sorted output vs original output
3. **Production-grade stress test**: edges > 10K, dim > 256
4. **Boundary values**: edges=0, edges=1, dim=1

**Anti-pattern**: Considering a kernel correct just because it "runs through" production data — the large-dim path has never been tested.

**Correct pattern**: First write the CPU reference implementation + UT -> compile and run in CPU mode -> only deploy to NPU after passing.
