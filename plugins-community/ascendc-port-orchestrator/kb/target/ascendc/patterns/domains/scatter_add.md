---
applies_to: soc=all
reason: The scatter-add optimization domain (reduce atomicAdd contention via
  sort-then-accumulate / register accumulation) is a general technique that applies
  across a5/a3/a2. NOTE the lead pattern P-P2 uses SIMT warp primitives
  (`Simt::WarpReduceAddSync`, `threadIdx`) that only exist on arch35/a5 — such
  SIMT-specific patterns carry their own per-pattern a5-only scope inline; the
  file-level default is `all` (same convention as precision.md / thread_utilization.md).
---

# Domain: Scatter-Add Optimization
> Patterns for kernels with atomicAdd scatter-write patterns.
> Load when: Analyzer detects atomicAdd in loop with indirect write target.

---

## Patterns

### P-P2: WarpReduceAddSync + warp-lane-0 atomic

**Severity**: High (7x) | **Source**: Codex

```cpp
// Anti-pattern: per-thread atomicAdd → 512 calls/block
// Correct: warp reduce then lane-0 atomicAdd → 16 calls/block
float warp_sum = Simt::WarpReduceAddSync(partial_sum);
Simt::ThreadBarrier();
if (threadIdx.x % 32 == 0 && warp_sum != 0.0f)
    atomicAdd(dst, warp_sum);
```

**Precondition**: dst must be pre-zeroed on the host side. Measured SG backward: 0.865ms → 0.121ms (7.1x).

---

### P-P10: Block Oversubscription

**Severity**: **High** | **Platform**: A5 verified | **Applicability**: scatter-add class operators

nblk > physical AIV core count (56) → disperses atomicAdd contention.

**Applicability conditions (key)**: Only effective when the kernel has atomicAdd contention.
- **Unsorted backward** (has atomicAdd contention): nblk=224 → bwd 2.0x speedup ✅
- **Sorted backward** (register accumulation, no atomicAdd): nblk=56 is strictly optimal. Oversubscription monotonically gets worse: 112 (+3.8%), 224 (+9.8%), 448 (+20.8%) ❌
- **Forward**: nblk=56 is always optimal (pipe-bound; oversubscription gives no benefit)

**Root cause**: Sorted + register accumulation eliminates atomicAdd contention → the benefit of dispersing contention via oversubscription disappears → only the register-pressure cost remains. Multiple blocks compete for the same AIV core's register file, and accumulator variables get spilled to HBM.

**E9-2 measurement confirmed** (2026-03-30, 61 clusters, NPU idle): see E10 exploration results.

---

### P-P17: Prefix-sum + block-level atomicAdd aggregation

**Severity**: High | **Source**: HKV hand-written version, CONFIRMED | **Applicability**: scatter-add class operators

Three-level aggregation to reduce global atomics:
1. **In-group prefix sum** (`__shfl_up`): each thread obtains local_offset
2. **Group leader → UB atomicAdd** (`__ubuf__`): once per group (512/32 = 16 times)
3. **Block leader → global atomicAdd**: once per whole block

512 global atomicAdd → 1. **Directly applicable to Pooling backward atomicAdd optimization.**

---

### P-P21: Scatter-add Sort + Register Accumulation (reduce atomicAdd count)

**Severity**: **High** | **Source**: AI-authored + msprof-driven (Batch 6, 2026-03-26)
**Applicability**: any scatter-add kernel (multiple inputs write the same output address)

**Problem**: In scatter-add every edge does one atomicAdd. When many edges target the same output (high fan-in), atomicAdd serialization becomes the bottleneck. msprof signature: `aiv_vec_ratio=1.0` but HBM bandwidth utilisation < 1% (pipe is jammed by atomicAdd).

**Optimization idea**: Pre-sort edges so writes for the same target are contiguous → register accumulate → single atomicAdd.

**Anti-pattern** (one atomicAdd per edge):
```cpp
for (int i = 0; i < edge_length; i++) {
    atomicAdd(&output[edge_out[i] * dim + d], input[edge_in[i] * dim + d]);
    // 100,000 edges pointing to the same target → 100,000 atomicAdd ops queued up
}
```

**Correct pattern** (register accumulation after sort):
```cpp
// Precondition: edges are sorted by edge_out
float accum = 0;
int prev_target = -1;
for (int i = 0; i < edge_length; i++) {
    int target = edge_out[i];
    if (target != prev_target) {
        if (prev_target >= 0) atomicAdd(&output[prev_target * dim + d], accum);  // write only when target changes
        accum = input[edge_in[i] * dim + d];
        prev_target = target;
    } else {
        accum += input[edge_in[i] * dim + d];  // register accumulation, no atomicAdd
    }
}
// flush final
if (prev_target >= 0) atomicAdd(&output[prev_target * dim + d], accum);
```

**Effect**: atomicAdd count drops from `edge_length` to `unique_targets`. With average fan-in=100, this is a 100x reduction.

**A5 measurement**: Pooling backward 100.53ms → 14.53ms (**-86%**), forward 15.66ms → 11.29ms (**-28%**). Bwd improvement is larger because atomicAdd contention is more severe (scatter-write vs scatter-read). Fwd improvement comes from register accum after sort reducing atomicAdd count (from `edge_length` to `unique_targets`).

**Sort overhead**: host `std::sort` 1147ms, NPU counting sort 405ms. Sort is a one-shot preprocessing step (no re-sort needed while the graph structure is unchanged).

**Trigger condition (generator must check)**:
- See `atomicAdd` inside a loop → check if it is a scatter-add pattern (multiple inputs write the same output)
- msprof shows `HBM_util < 1%` but `aiv_vec_ratio = 1.0` → atomicAdd bottleneck confirmed
- **The generation stage should already hint "this kernel has scatter-add; recommend providing a sorted variant"** — do not wait until the optimization stage.

**Notes**:
- ~~Sort only benefits backward~~ **Correction (E9-1 measurement, 61 clusters)**: forward benefits too. D_Fwd (sorted + BRE=emb_dim + register accum) is **1.39x** faster than B_Fwd (unsorted) overall (15.66ms → 11.29ms). For small dim (≤33), **1.45x**; for dim=1 + high fan-in, **7x** (cluster 5: 0.507 → 0.072ms). But for dim > 128, D actually becomes slower (BRE=emb_dim leaves too few index_threads); fall back to C (template sorted, BRE=32) in that case.
- Must be combined with P-P20 (BRE=emb_dim) so iter_emb=1 and a single scalar accum is sufficient.
- Precision: the order of float register accumulation differs from per-edge atomicAdd, but differences remain within atol (61/61 PASS).
- **dim cutoff**: dim ≤ ~128 use D variant (BRE=emb_dim); dim > 128 use C variant (BRE=32, template sorted).

#### Multi-Dim Accumulator Array (iter_emb > 1)

When BRE < emb_dim (large dims where BRE=emb_dim is not feasible), `iter_emb > 1` and a single scalar `accum` is insufficient. Use an accumulator array:

```cpp
constexpr int MAX_ACCUM = 16;  // max dim positions per thread
float accum[MAX_ACCUM] = {0};
int prev_target = -1;
int my_emb_lane = threadIdx.x % BRE;  // cache once

// Loop nesting: INDEX OUTER, EMBEDDING INNER (CRITICAL — never invert!)
for (int i = iter_start; i < iter_end; i++) {
    int target = edge_out[sorted_idx[i]];
    if (target != prev_target) {
        // Flush all accumulators for previous target
        if (prev_target >= 0) {
            for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
                if (accum[j] != 0.0f)
                    atomicAdd(&output[prev_target * dim + my_emb_lane + j * BRE], accum[j]);
                accum[j] = 0.0f;
            }
        }
        prev_target = target;
    }
    // Accumulate ALL embedding positions for this edge
    for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
        int64_t src_offset = static_cast<int64_t>(edge_in[sorted_idx[i]]) * dim + my_emb_lane + j * BRE;
        accum[j] += simt_to_float_generic<DATA_TYPE>(input[src_offset]);
    }
}
// Final flush
for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
    if (prev_target >= 0 && accum[j] != 0.0f)
        atomicAdd(&output[prev_target * dim + my_emb_lane + j * BRE], accum[j]);
}

// FALLBACK: when iter_emb > MAX_ACCUM, fall back to per-element atomicAdd
if (iter_emb > MAX_ACCUM) {
    // Use baseline atomicAdd path for the overflow portion
}
```

**CRITICAL loop nesting rule**: Index MUST be the outer loop, embedding inner. Inverting causes:
1. `prev_target` state reset between embedding iterations → redundant atomicAdd
2. iter_emb × redundant GM reads on edge_in/edge_out arrays
3. Defeats the entire purpose of sorted accumulation for multi-dim

---

### P-P32: Sorted-Edge First-Occurrence Dedup (atomicCAS-Free)

**Severity**: High | **Source**: E10-1 hand optimization (2026-03-30) | **Applicability**: dedup / first-occurrence detection

**Problem**: `generate_assign_edges`-class kernels use `atomicCAS` to detect first occurrence — one atomic per edge. When edge count is large, atomicCAS serialization becomes the bottleneck.

**Precondition**: edges are sorted by target (coordinate with P-P21 sort preprocessing).

**Anti-pattern** (atomicCAS per edge):
```cpp
// One atomicCAS per edge — checks whether this is the first write to the target
int old = Simt::AtomicCas(&assign_edges[target], INVALID, source);
if (old == INVALID) {
    // first occurrence
}
```

**Correct pattern** (adjacent comparison):
```cpp
// Precondition: edges sorted by edge_out
// Compare adjacent edges' targets — a change of target marks "first occurrence"
int prev_target = (tid > 0) ? edge_out[tid - 1] : -1;
int cur_target = edge_out[tid];
if (cur_target != prev_target) {
    // first occurrence of this target — no atomic needed!
    assign_edges[cur_target] = edge_in[tid];
}
```

**Advantages**:
- Zero atomics (pure load/store, no CAS contention)
- O(1) per element (vs atomicCAS's O(contention))
- Handles block boundaries: the first thread needs to compare with the previous block's last element (cross-block boundary)

**Measurement**: Pooling assign_edges: 259ms → 10ms (**25.6x**), combined with the overall sorted pipeline.

**Trigger condition**: any first-occurrence / dedup operation where the input is already sorted or can be pre-sorted.

---

### P-P39: Fan-In Threshold for Sort Decision (CANN-derived)

**Severity**: **HIGH** | **Source**: CANN ops-nn/index/scatter_add/ knowledge extraction (2026-04-14) | **Applicability**: any scatter-add class operator

**Problem**: P-P21 describes the sort + register-accumulation pattern but does not give a quantitative threshold for when sorting should be enabled. Sorting has overhead (O(N log N)); at low fan-in it is actually slower.

**Decision rule** (CANN scatter_add_tiling_base.cpp):
- `fan_in_ratio = indicesNum / varShape[0]` (input index count / output row count)
- **fan_in > 10:1** → enable sort (general threshold)
- **fan_in > 3:1 AND embDim * dtype_bytes >= 100K** → also enable sort (at large dim each atomicAdd is more expensive, lowering the threshold for sorting)
- **fan_in ≤ 3:1** → do not sort; atomicAdd directly

**Generator must check**: when generating a scatter-add kernel, analyze the benchmark's shape spec:
1. Compute fan_in_ratio = index_count / unique_target_count
2. fan_in > 10 → generate a variant with sort
3. fan_in 3-10 and embDim is large → generate a variant with sort
4. fan_in < 3 → baseline atomicAdd is sufficient

**Evidence**: CANN scatter_add_tiling_base.cpp:204-211 (`isSort_ = indicesNum_ > varShape_[0] * TEN`), E1 level (inferred from source).

**Stop condition**: when fan_in < 3 the sort overhead exceeds the savings. When index is already pre-sorted, skip sort and accumulate directly.

---

### P-P40: Double-Buffer Accumulation with Flush Threshold

**Severity**: **HIGH** | **Source**: CANN embedding_dense_grad_v2 knowledge extraction (2026-04-14) | **Applicability**: UB accumulation stage of sorted scatter-add

**Problem**: When accumulating many grad rows for the same index after sorting, if the accumulation count is large (high fan-in), fp16/bf16 precision is gradually lost.

**Pattern**: Maintain two UB accumulation buffers (addRes[0] and addRes[1]) and switch them when the index changes:
```
switchId = false
for each (index, grad) in sorted_data:
    if index == lastIndex:
        Add(addRes[switchId], addRes[switchId], grad)  // UB vector add
        count++
        if count >= FLUSH_LIMIT:  // fp16: 10, fp32: can be larger
            CopyOut(addRes[switchId], GM_output)  // uses SetAtomicAdd
            Duplicate(addRes[switchId], 0)  // clear
            count = 0
    else:
        CopyOut(addRes[switchId], GM_output)  // flush old buffer
        switchId = !switchId
        Duplicate(addRes[switchId], 0)  // clear new buffer
        Copy(addRes[switchId], grad)  // begin new accumulation
        lastIndex = index
        count = 1
```

**Flush threshold**: **LIMIT_COUNT_NUM = 10** (fp16/bf16). After accumulating > 10 fp16 values, precision loss starts to appear.

**Reason for dual buffers**: while buffer A is flushing to GM (MTE3 pipe), buffer B can start accumulating the next index (VEC pipe), achieving MTE3/VEC overlap.

**Evidence**: CANN embedding_dense_grad_v2.h:255-332, LIMIT_COUNT_NUM=10 (line 26). E1 level.

**Stop condition**: When the index is random (unsorted), it degrades to flushing after every single accumulation. Requires `embDim * sizeof(CT) * 2 < UB available space`.

---

### P-P41: 2D Tiling for Large Embedding Scatter

**Severity**: **MEDIUM** | **Source**: CANN scatter_add_tiling_base knowledge extraction (2026-04-14) | **Applicability**: scatter operators where embDim is too large to fit in UB alongside the sort buffer

**Problem**: UB 256KB must simultaneously hold: index array + sort buffer + accumulation buffer + update data. When embDim is very large (e.g., embDim=16384, fp32 = 64KB/row), a single row occupies a large UB footprint.

**Pattern**: Split [indicesNum, embDim] work into a 2D grid of rowTileNum × colTileNum:
1. Row partitioning: different cores handle different index ranges
2. Column partitioning: different embDim positions of the same index are handled by different cores
3. UB budget: `ubFactorRow * ubFactorCol * sizeof(T)` + index + sort buffer must be < UB size
4. Column-split cores finally merge results via atomicAdd (different columns of the same index are written to different GM addresses by different cores — no contention)

**Column-split threshold**: Only consider column splitting when `embDim * dtype_bytes > 4096`. Below that, fit the whole row into UB.

**Evidence**: CANN scatter_add_tiling_base.cpp:290-393 (FindUniqueCut + SimdTiling + DoBlockTiling). E1 level.

**Stop condition**: When embDim * dtype_bytes ≤ 4096, column splitting is not needed. Column splitting adds tiling complexity and should not be used for small-dim cases.

---

### P-P48: Per-Block Private Histogram in UB (atomicAdd Reduction)

**Severity**: **HIGH** | **Source**: Histc optimization (2026-04-14), histogram_v2 SIMT rewrite (2026-06-22) | **A5 verified**: ✅ 0.28x→0.53x (Histc, 1.9×), 0.44x→0.95x (histogram_v2, 2.15× overall, 31-element kernel speedup 14–35×) | **Applicability**: histogram / scatter-count class operators

**Problem**: N threads do GM atomicAdd against B bins (N >> B); atomicAdd serialization becomes the bottleneck. 28672 threads hitting 50-256 bins spend > 99% of time queueing.

**Pattern**: Each block maintains a private histogram in `__ubuf__`. Threads do UB atomicAdd (much faster than GM). After a ThreadBarrier, merge to GM via SIMT per-thread atomicAdd:

**Pattern** (Histc, NPUKernelBench):
```cpp
__ubuf__ int32_t local_hist[MAX_BINS];
// clear
for (uint32_t b = threadIdx.x; b < bins; b += THREAD_NUM) local_hist[b] = 0;
Simt::ThreadBarrier();
// local accumulation
for (...) atomicAdd(&local_hist[bin_idx], 1);
Simt::ThreadBarrier();
// merge to GM — SIMT per-thread atomicAdd
for (uint32_t b = threadIdx.x; b < bins; b += THREAD_NUM) {
    if (local_hist[b] > 0) atomicAdd(&gm_counts[b], local_hist[b]);
}
```

**Effect**: GM atomicAdd count drops from `total_elems` to `bins × nblk`. 65536 elements, 128 bins, 56 blocks: ~33K → 7168 (4.6x reduction).

**UB requirement**: MAX_BINS × 4 bytes = 1024 bytes (negligible).

**Trigger condition**: any kernel with `atomicAdd(&output[computed_index], ...)` where output is much smaller than input (histogram, voting, binning, etc.).

**Evidence**:
- Histc (2026-04-14): 0.28x → 0.53x. E3 level.
- histogram_v2 (2026-06-22): 0.44x → 0.95x overall wallclock ratio (2.15× improvement). Kernel time for 1M elements: 894μs → 5.8–28.6μs (14–35× profiler speedup). 31/31 precision PASS (30/31 bit-exact). SIMT UB atomicAdd with per-block histogram. E4 level — validated on 2 independent ops (Histc + histogram_v2) with strong quantitative evidence.

**Stop condition**: When bins > UB available space / sizeof(int32) (theoretically 256KB / 4 = 64K bins, far larger than anything realistic).

---

### P-P67: PyTorch-UB-class scatter-overwrite — NPU reference is non-deterministic, kernel cannot chase

**Severity**: CRITICAL (precision-correctness boundary) | **Source**: op#19 IndexPut aog-precision-probe pp-2 (2026-04-27) | **Applicability**: any scatter-class op with PyTorch-UB on duplicate indices

**Problem class**: `torch.index_put_(accumulate=False)` and `torch.scatter_(reduce=None)` ("assignment scatter") on inputs **with duplicate indices** are PyTorch-undefined-behavior per docs. NPU `torch_npu` resolves duplicate writes via parallel hardware-thread scheduling — winner per dup slot is **non-deterministic across runs** when contention is high enough.

**Concrete signature** (op#19 pp-2 measurement, fp16/bf16/fp32, K ∈ {64..16384}, fan_in=K/N≈0.5 with allow_dup_indices):
- 5-run NPU bit-eq check: 3/17 small-K cases (K ≤ 128) deterministic; 14/17 K ≥ 256 cases NON-deterministic
- Pairwise max abs diff across 5 NPU runs: 3.2 .. 4.39 (fp16, K=2560)
- ~1.2 % of slots flip identity between any two runs

**Anti-pattern (what NOT to do)**:
- ❌ Don't write a deterministic kernel rule (last-wins / first-wins / wave-chunk / AIV-block) and expect it to match. Best rule pp-2 found: WF32 (wave-firstwins W=32) at 61% mean / 31% min match. Far below the 90% acceptance bar.
- ❌ Don't use atomicAdd / SetAtomicAdd / SyncAll / multi-core scatter to "match NPU's parallelism". This re-introduces kernel non-determinism (DEBT-053 motivator); fails determinism gate; STILL doesn't bit-match the ref because the ref's randomness is HW-scheduling-dependent, not algorithm-dependent.
- ❌ Don't add case-specific predicates (`if K > threshold use rule X else rule Y`). OL-85 anti-overfitting violation.

**Correct response (what TO do)**:
1. **Precision-probe Step 1** (mandatory): replicate verifier's exact input gen, run NPU 5× on cloned buffers with explicit `torch.npu.synchronize()`, save pairwise diff matrix. Probe template: `workspace/indexput/probes/pp2_step1_dup_det.py`, `pp2_step1b_case33_isolated.py`.
2. **Precision-probe Step 2** (mandatory before classifying): exhaustive structured-rule search (R1 last/first, R3/R4 wave-chunk × W∈{8..512}, R5 sort-stable, R6 AIV-block × A∈{1,2,4,8,16,20,40}, R7 round-robin). Record per-case match rates. Templates: `pp2_step2_rule_search.py`, `pp2_step3_aiv_rule.py`.
3. **If best rule ≥ 99% on every case** → kernel-implementable, write it as deterministic single-thread algorithm.
4. **If best rule ≤ 90% mean OR 5-run pairwise diff > 0** → **REQUIREMENT** verdict per OL-90. Kernel ships with deterministic single-core single-source-order scan (matches CPU torch semantics, not NPU torch). Failures are spec-level UB, not kernel bugs.

**Verifier-side mitigations** (OL-90 lists 4): alt-ref hook (CPU-torch fallback for UB-trigger cases), case-gen `allow_dup_indices=False` default, dual-reference REPORT row, per-case tolerance loosen with documented rationale.

**Cross-reference**:
- **OL-90** (PyTorch-UB-class detection + verifier-side mitigation) — full canonical entry
- **OL-85** (logic-first, anti-overfitting) — forbids case-specific predicate hacks
- **OL-88** (ref non-det preflight) — sibling class; OL-88 = CANN op internal race, P-P67 = spec-level user-input UB
- **P-P61** (kernel runtime determinism) — kernel side stays deterministic regardless of ref behavior; non-det ref is no excuse for non-det kernel
- **DEBT-053** (op#19 sequential-reset) — the motivator that drove discovering this pattern

**Evidence summary**:
- op#19 IndexPut 29/46 PASS (22/22 acc=True + 7/7 acc=False no-dup + 0/17 acc=False with-dup); deterministic-kernel by construction (single AIV core, THREAD_NUM=1, no atomic, 5-run gate 46/46 IDENTICAL)
- pp-2 probes: `workspace/indexput/probes/pp2_npu_dup_det_5run.json` (Step 1), `pp2_step2_rule_match.json` (Step 2 R1-R5), `pp2_step3_aiv_rule.json` (Step 2 R6-R7)
- pp-2 verdict: REQUIREMENT — no deterministic kernel can match a non-deterministic ref. Published 17/46 gap is the canonical result until verifier-side alt-ref lands (DEBT recommendation).
