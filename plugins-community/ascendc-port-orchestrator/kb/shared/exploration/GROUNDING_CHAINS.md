# Grounding Chains: Observable Data → Hypothesis

Each chain maps an observable msprof metric pattern to a diagnosis and candidate dimensions to explore.

## Chain Definitions

### GC-1: Under-Utilization
- **Observable**: All pipes < 30% utilization (vec_ratio, mte2_ratio, scalar_ratio all low)
- **Diagnosis**: Each core dispatch does too little work — cores are idle between dispatches
- **Candidate Dimensions**: D2 (increase work granularity), D1 (restructure to batch more per dispatch)
- **Example**: SG prod_a (vec=25%, mte2=12%, scalar=18%) → persistent kernel or batched tokens

### GC-2: Compute-Bound on VEC
- **Observable**: vec_ratio > 90%, scalar_ratio < 5%
- **Diagnosis**: VEC pipe is the bottleneck — computation itself is the limit
- **Candidate Dimensions**: D1 (reduce redundant compute via loop reorder), D5 (tile size to maximize VEC throughput)

### GC-3: Scalar/Control-Flow Overhead
- **Observable**: scalar_ratio > 20%
- **Diagnosis**: Too much time in scalar operations (loop control, address calculation, conditionals)
- **Candidate Dimensions**: D2 (persistent kernel eliminates re-dispatch overhead), D3 (cache preload reduces GM scalar reads)
- **Example**: Non-persistent kernel with per-token setup → P-P22 persistent kernel

### GC-4: DMA-Bound
- **Observable**: mte2_ratio > 50%
- **Diagnosis**: Waiting for data movement from GM to UB
- **Candidate Dimensions**: D3 (increase prefetch depth for better overlap), D1 (reorder loops to maximize data reuse)
- **Example**: Expert loaded 512 times → expert-major loop loads it once

### GC-5: Pipeline Bubbles
- **Observable**: vec_ratio < 50% AND mte2_ratio < 50% AND scalar < 20%
- **Diagnosis**: No pipe is saturated but none is efficient — pipes are waiting for each other
- **Candidate Dimensions**: D4 (better sync to overlap pipes), D3 (TQue with deeper depth)
- **Example**: PipeBarrier causing full stalls → TQue auto-sync (E13: 1.6-2.3x speedup)

### GC-6: Per-Dispatch Overhead Dominates Small Cases
- **Observable**: Large gap between performance on small vs large workloads (e.g., small=1.04 but medium=0.68)
- **Diagnosis**: Fixed dispatch/setup overhead dominates when work is small
- **Candidate Dimensions**: D2 (persistent kernel to amortize dispatch cost)
- **Note**: This chain often co-occurs with GC-1

### GC-7: GM Read Amplification (Fan-Out)
- **Observable**: N work items share same GM read (detectable by code analysis, not just msprof)
- **Diagnosis**: Same data loaded from GM multiple times instead of being reused
- **Candidate Dimensions**: D1 (sort-to-reuse: group items that share data)
- **Example**: 64 experts, each referenced ~512 times → expert-major loop eliminates 512x redundant reads
- **Related pattern**: P-P24 (sort-to-reuse for backward fan-in). GC-7 is the general principle covering both directions.

## Usage Protocol

1. Run msprof, extract `aiv_vec_ratio`, `aiv_mte2_ratio`, `aiv_scalar_ratio`, `task_duration`
2. Check each chain's trigger condition
3. Multiple chains may match — that's normal, it narrows the search
4. For each matching chain, look up candidate dimensions in STRUCTURAL_DIMENSIONS.md
5. Cross-reference with PATTERN_INDEX.md to filter already-applied patterns

### GC-8: SIMT MTE2 Starvation (Memory-Bound Elementwise)
- **Observable**: mte2_ratio ≈ 0% AND throughput < 50% of theoretical bandwidth AND kernel is SIMT
- **Diagnosis**: SIMT mode bypasses MTE2 DMA entirely — all GM access through dcache (VEC pipe). VEC carries both compute and memory loads.
- **Candidate Dimensions**: **SIMD conversion** (P-P33), mixed SIMT+SIMD mode
- **Note**: This is NOT fixable within SIMT — requires architectural switch to SIMD/mixed mode
- **Example**: MXFP4 SIMT kernel: MTE2=0%, vec=77%, throughput=125 GB/s (31% of 400 GB/s)
- **Evidence**: MXFP4 msprof (2026-04-07), H2/H3 both failed to improve

## Hopeless Case Detector

Before starting exploration, check ALL of these conditions. If ALL are true, do **NOT** explore — escalate to human:

1. All pipe utilizations > 80% (kernel is compute-saturated)
2. Target-side theoretical compute throughput matches the measured result within the accepted margin

If all are true, the target implementation is at its measured architectural limit; record the evidence before stopping exploration.
