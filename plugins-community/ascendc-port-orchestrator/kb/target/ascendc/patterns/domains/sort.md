---
applies_to: soc=all
reason: AscendC Sort instruction (Concat+Sort+Extract pipeline P-P42) is a SIMD/Vector-pipe operation available across a5/a3/a2. Per-tile capacity assumptions (32/64/256 elements per pass) need re-checking on V220 due to 192KB UB vs 256KB UB.
---

# Domain: Sort Optimization
> Patterns for sorting kernels on Ascend NPU.
> Load when: Analyzer detects sort operation (argsort, topk, or explicit sort).
> Source: CANN ops-math/sort/ knowledge extraction (2026-04-14)

---

## Patterns

### P-P42: Hardware Sort Pipeline (Concat+Sort+Extract)

**Severity**: **CRITICAL** | **Source**: CANN ops-math/sort/ knowledge extraction (2026-04-14) | **A5 verification**: PASS 0.31x→0.85x (2.7x speedup, 2026-04-14) | **Applicability**: sort operators

**Problem**: Software sorting (scalar merge sort, bubble sort, etc.) is extremely slow on NPU. Ascend hardware has a dedicated bitonic sort network that processes 32 elements per cycle.

**Pattern**: Three-step pipeline using the built-in AscendC Sort API:
```cpp
// Step 1: Pack into sort struct (value, index) pairs
uint32_t concatRep = alignedN / 16;
AscendC::Concat(concatLocal, xLocal, concatTmpLocal, concatRep);

// Step 2: Hardware sort (bitonic network, descending by default)
uint32_t sortRep = alignedN / 32;
AscendC::Sort<float, true>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, sortRep);

// Step 3: Unpack results
AscendC::Extract(sortedValueLocal, sortedIndexLocal, sortedLocal, extractRep);
```

**Ascending-sort trick**: The hardware sorter defaults to descending. For ascending, flip the sign bit before sorting:
```cpp
// Ascending: flip sign bit → descending sort → flip sign bit again
Adds(xLocal_int32, xLocal_int32, (int32_t)0x80000000, N);  // flip sign bit
Sort<float, true>(...);  // hardware sort (descending)
Adds(result_int32, result_int32, (int32_t)0x80000000, N);   // flip back
```

**bf16 handling**: The hardware sorter only supports fp32 and fp16. bf16 must be Cast to fp32 for sorting, then Cast back.

**UB space requirements**:
- `sortedLocal`: 8 × alignedN bytes (sort struct)
- `sortTmpLocal`: 8 × alignedN bytes (temporary buffer)
- `concatTmpLocal`: GetConcatTmpSize() bytes
- **Total**: ~20-24 bytes/element → N=4096 fp32 ≈ 80-96KB

**Evidence**: CANN sort_merge_sort.h:198-243, sort_tiling_arch35.cpp:746-761. E1 level.

**A5 verification update (2026-04-14 / 2026-04-18)**: In practice on Ascend950PR we use the **Advanced Sort API**:
```cpp
// Recommended default: MERGE_SORT (EC-33 addendum — RADIX_SORT has sporadic VMS 343 on A5 CANN 9.0.0)
constexpr SortConfig sortCfg = {SortType::MERGE_SORT, true};  // must be global constexpr (EC-24)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ > 0)                // must have __NPU_ARCH__ guard (EC-25)
Sort<float, false, sortCfg>(dstLocal, srcLocal, tmpLocal, calCount);
#endif
```
**SortType selection** (2026-04-18 update):
- **`MERGE_SORT` (default on A5)**: stable, no VMS 343 observed. Preferred.
- **`RADIX_SORT`**: sporadic runtime `aicore 343 "Incorrectly sorted data entered by the VMS"` observed on Ascend950PR CANN 9.0.0 (see EC-33 addendum). Use only when perf profiling proves a significant advantage over MERGE_SORT AND representative inputs are confirmed not to trigger VMS 343.

This API uses sort internally, supports arbitrary N, and does not require manual Concat+Sort+Extract.
- Requires `#ifdef __NPU_ARCH__` guard (EC-25)
- SortConfig must be global constexpr (EC-24)
- DataCopyPad UB→GM crashes; use DataCopy + host padding instead (EC-23)
- Sorting on a non-last dimension: pybind-layer permute+contiguous → kernel sorts only the last dim → permute result back

**Stop condition**: When N > 4096 (fp32) or N > 1024 (fp16), UB space is insufficient; switch to radix sort. int types are not supported by this pipeline (use radix sort).

---

### P-P43: Sort Algorithm Selection Decision Tree

**Severity**: **HIGH** | **Source**: CANN sort_tiling_arch35 knowledge extraction (2026-04-14) | **Applicability**: all sort operators

**Problem**: Different N/dtype/batch combinations need different sort algorithms. No single algorithm covers all scenarios.

**Decision tree** (check in priority order):

```
1. N ≤ 4096 (fp32) or N ≤ 1024 (fp16/bf16)?
   YES → Hardware Merge Sort (Concat+Sort+Extract, P-P42)
   NO → continue

2. All data (value + index + tmp) fits in UB?
   YES → Single-core Radix Sort (AscendC Sort API with RADIX_SORT)
   NO → continue

3. B==1 AND 4096 < N ≤ 32768 AND fp32?
   YES → Multi-core Merge Sort (per-core local sort + 4-way hardware MrgSort merge)
   NO → continue

4. B ≥ core count AND N > 4096 AND blocksPerRow ≤ 256?
   YES → Big Batch Merge Sort (each core processes a whole row independently)
   NO → continue

5. Default → Multi-core Radix Sort (decoupled lookback prefix scan)
```

**Generator instructions**: When generating a Sort kernel:
1. Analyze the benchmark shape → determine N and dtype
2. For N ≤ 4096 (fp32): use P-P42 hardware sort directly — simplest and fastest
3. For N > 4096: consider radix sort, but implementation is complex. You can first use P-P42 on UB-sized chunks and then manually merge.

**Evidence**: CANN sort_tiling_arch35.cpp:746-761. E1 level.

**Stop condition**: When N < 32 the setup overhead dominates for all algorithms; a simple insertion sort may be faster.

---

### P-P44: Radix Sort Twiddle Transform for Float/Signed Types

**Severity**: **MEDIUM** | **Source**: CANN sort_radix_sort_more_core knowledge extraction (2026-04-14) | **Applicability**: radix sort on non-unsigned types

**Problem**: Radix sort sorts by bit pattern. IEEE 754 float bit patterns do not directly reflect numeric magnitude (negative bit patterns are "larger" than positive ones).

**Twiddle transformation rules**:
- **Unsigned int**: no transform needed
- **Signed int**: XOR sign bit mask (`0x80000000` for int32)
- **Float**: positive XOR `0x80000000` (flip sign bit); negative XOR `0xFFFFFFFF` (flip all bits)
- **Descending**: additional NOT of all bits after the transform

```cpp
// Vectorized implementation:
And(signBits, data, 0x80000000);         // extract sign bit
CompareNE(mask, signBits, 0);             // mask for negatives
Select(xorMask, mask, 0xFFFFFFFF, 0x80000000);  // pick XOR mask
Xor(twiddled, data, xorMask);            // apply transform
```

**Evidence**: CANN sort_radix_sort_more_core.h:314-562. E1 level.

**Stop condition**: Only needed for radix sort. The hardware Sort API (Concat+Sort+Extract) handles this internally.

---

### P-P57: SIMD ReduceMax(calcIndex=true) for small-k vectorized topk

**Severity**: **CRITICAL** | **Source**: 7_MoeGatingTopKSoftmax (2026-04-17), msprof E1 | **Applicability**: small-k topk (k ≤ 16) over N ≥ 256

**Problem**: Scalar insertion sort / selection-by-scan is extremely slow for topk on N ≥ 2048. Per-row `for i in N: if buf.GetValue(i) > topk_min: insert(i, v)` runs on the S pipe, and msprof shows `scalar_ratio` easily hitting 0.97+. The S pipe does 1 element/cycle — 2-3 orders of magnitude slower than the VEC pipe's 64-256 elements/cycle.

**Pattern** (k SIMD ReduceMax calls, each returning val+idx):

```cpp
// Setup:
// - valBuf: LocalTensor<float> containing all N candidate values (fp32 for precision)
// - reduceWorkBuf: temporary buffer, ≈ ceil(N/16) × 4 bytes
// - reduceOutBuf: 64 bytes (stores val + idx pair)

for (int kk = 0; kk < k; ++kk) {
    // 1) SIMD ReduceMax with index: find val and its position in buf simultaneously
    AscendC::ReduceMax<float>(
        reduceOutBuf,     // dst: [val_fp32 | idx_bits_as_fp32]
        valBuf,           // src
        reduceWorkBuf,    // work
        /*count=*/N,
        /*calcIndex=*/true
    );

    // 2) Scalar read of val and idx
    float  val = reduceOutBuf.GetValue(0);
    int32_t idx = *reinterpret_cast<int32_t*>(&reduceOutBuf.GetValue(1));  // reinterpret bits

    // 3) Output topk_vals[kk] = val, topk_indices[kk] = idx

    // 4) Mask out the selected slot so next ReduceMax picks the next-best
    valBuf.SetValue(idx, -INFINITY);

    // 5) S_V sync: SetValue is on the S pipe; next ReduceMax is on the V pipe
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
}
```

**Applicability**:
- k ≤ 16 (k ReduceMax calls are only worth it for small k; switch to P-P42 hardware Sort for large k)
- N ≥ 256 (for small N, ReduceMax setup overhead exceeds the scalar loop cost)
- Small-k large-N topk / selection (MoE gating, beam search step, attention head-k, etc.)

**Not applicable** (use P-P42 hardware Sort):
- k close to N (essentially a full sort)
- Need the complete sorted output order (P-P57 only guarantees the top-k values; tie-break order depends on -INFINITY mask hit ordering)
- int dtype (hardware ReduceMax only supports fp32/fp16/bf16)

**Evidence**:
- 7_MoeGatingTopKSoftmax (2026-04-17): k ∈ [1,10], N ∈ [2048, 7168], dtypes fp16/fp32/bf16.
  - **Before** (scalar insertion topk): msprof `scalar_ratio=0.975, vec_ratio=0.017` on worst case 17 [512,1024,2048] bf16, end-to-end sum-ratio **0.142x**
  - **After** (P-P57): msprof `scalar_ratio=0.271, vec_ratio=0.679` on same case, end-to-end sum-ratio **1.097x** (+673% single iter). case 47 [3584,7168] bf16: 8.0ms → 0.41ms (19.5× speedup on this case alone).
- Precision: 50/50 PASS held (all of fp16/fp32/bf16 pass). Tie-break matches PyTorch topk first-occurrence (when multiple values are equal, ReduceMax returns the smallest index — matching torch.topk's default behavior).
- top_k_top_p_sample A5 (2026-06-24, Ascend950PR CANN 9.0.0): iterative ReduceMax top-K extraction used for the Q-path (top-K over V ≤ 2048). K ≤ 100 completes in ~0.1 ms with precision PASS — see the practical-ceiling note below; the `k ≤ 16` ceiling is a conservative per-call-cost heuristic, NOT a hard limit.

**Practical K ceiling (decision rule)** — the `k ≤ 16` guidance above is conservative. P-P57's total cost is `k × cost(one ReduceMax over V)`; per-call cost scales with V, so the viable k grows as V shrinks. Measured: K ≤ 100 over V ≤ 2048 (~0.1 ms on A5) is practical. Treat `k > 16` as "evaluate the k×V product — prefer P-P42 only when the iteration count dominates", not as an automatic switch trigger. (See OL-252 for the no-Q case where even this iterative top-K is unnecessary — output is a single argmax.)

**Stop condition**: Switch to P-P42 when k > 16 (conservative), when the k×V product makes iterative ReduceMax dominate, or when full sorted output is required.

**Related**: OL-82 (scalar_ratio > 0.9 signature), P-P42 (hardware Sort for larger k)

---

## P-P60: AscendC Sort ASC tie-break direction is REVERSED from PyTorch stable ASC (ANTI-PATTERN)

**Severity**: **CRITICAL** | **Type**: ANTI-PATTERN | **Applicability**: any kernel using `Sort<T, isReuse, SortConfig{*, ASC}>` that needs to match PyTorch `.sort(stable=True)` tie order

**Anti-pattern** (NAIVE assumption — wrong):

```cpp
// Assumption: ASC sort + stable → smaller original idx comes first among ties (matches PyTorch)
constexpr SortConfig CFG_ASC = {SortType::RADIX_SORT, false};  // isDescend = false
Sort<float, false, CFG_ASC>(dstVal, dstIdx, srcVal, tmp, count);
// WRONG: AscendC Sort ASC actually puts LARGER original idx first in ties
//     — exactly opposite to PyTorch stable ASC
```

**Correct semantics (measured)**:

| Sort config | Tie order (when values are equal) |
|-------------|---------------------|
| `AscendC Sort<DESC>` (documented: "i>j, score[j] is selected first") | smaller original idx appears first |
| `AscendC Sort<ASC>` (undocumented) | **larger** original idx appears first ← reversed |
| `PyTorch .sort(dim, descending=False, stable=True)` (ASC) | smaller original idx appears first |
| `PyTorch .sort(dim, descending=True, stable=True)` (DESC) | smaller original idx appears first |

In other words: PyTorch keeps smaller-idx-first-in-ties in both ASC and DESC (stable semantics); AscendC Sort ASC **flips** this direction.

**Detection**: signature `max_abs_diff = 3.4e38` (-inf vs finite swap) at a few tied-boundary positions per row. fp16/bf16 fail more than fp32 (low-mantissa dtypes have more ties). Only appears on the ASC path — DESC-only sorts (e.g. topk) do not trigger this.

**Fix (two equivalent options)**:

1. **Post-walk reselect cutoff_orig_idx** (validated via probe): after the cumsum walk identifies `cutoff_val`, compute `n_drop_tied` (how many in the tied group to drop), then scan the top-K buffer linearly for all idx with `val == cutoff_val`, and pick the n_drop_tied-th **smallest** as `cutoff_orig_idx`. The Phase 4 mask uses the standard `(v == cutoff AND idx > cutoff_orig_idx)` condition.
   ```cpp
   // After cumsum walk identifies cutoff_val and n_drop_tied:
   int32_t tied_idxs[MAX_TIED];
   int32_t n_tied = 0;
   for (int32_t i = 0; i < effective_kept; i++) {
       if (top_val[i] == cutoff_val) tied_idxs[n_tied++] = top_orig_idx[i];
   }
   // Sort tied_idxs ascending (small n_tied, simple scalar insertion)
   // Pick the n_drop_tied-th smallest as cutoff_orig_idx
   cutoff_orig_idx = tied_idxs[n_drop_tied - 1];
   ```
2. **Secondary key sort by `-original_idx` within ties** (more expensive): do a secondary sort on the top-K buffer. Usually not worth it.

**NOT a fix (time-wasters)**:
- Flipping `SortConfig.isDescend` — both directions are "larger-idx-first-in-ties"; only the value direction differs, tie order still does not match PyTorch
- Per-chunk bubble reorder — only canonicalizes within-chunk; the cross-chunk tie order is determined by the merge logic, not by each chunk's sort tie behavior

**Evidence**: 9_TopKTopP cold-run 2026-04-18, probe iter 4 via `probes/p3_cutoff_boundary_analysis.py` (Python walk-simulation compares AscendC ASC sort output against PyTorch stable ASC output and shows the tied-group idx order is reversed). After the fix: 29/50 → 49/50. The remaining 1/50 is torch_npu vs pytorch-native 1-ULP cumsum drift (see OL-83), not a kernel bug.

**Related**:
- P-P42 (Hardware Sort pipeline): P-P42 describes `Sort<DESC>` tie behavior (smaller-idx-first, per CANN docs). P-P42 makes **no** claim about `Sort<ASC>` — if you use the ASC path, you must additionally handle this anti-pattern
- P-P59 (tied-threshold buffer truncation): P-P59 requires the buffer to hold all tied-threshold values; P-P60 requires correct tie order. **Both are required conditions for a PyTorch-stable-sort-compatible kernel** — neither alone is sufficient
- EC-31 (Select mask polarity): a common parallel bug when building the Phase 4 mask
- EC-32 (effective_kept vs buffer_len): post-walk processing must distinguish the two

---

### P-P107: Tiled reduction+normalization via GM workspace (3-pass streaming) when the reduction span exceeds UB

**Severity**: **HIGH** | **Source**: top_k_top_p_sample A3→A5 kw-5 V-tiling rewrite (2026-06-25) | **Applicability**: any per-row normalization needing both a reduction (max/sum) and an elementwise re-application (div) over a vector longer than UB

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=normalization / sampling / long-vector-reduce`
`unverified_on: soc=Ascend910_V220 (A3 — UB is 192KB not 256KB; tile budget must be re-checked, and DataCopyPad UB→GM is the V220-crashing EC-23 direction, not the free V351 path)`

**Problem**: A per-row normalization — softmax, L1/L2-normalization, online row-stats — needs a reduction (global max, then global sum) followed by an elementwise re-application (`div`) over a vector of length V. When V exceeds what fits in one UB allocation, a fixed `MAX_V_UB`-sized buffer silently overflows at V > cap: the elementwise pass writes/reads past the buffer, producing garbage output or a 507035 vector-core exception when later O(V)-shaped scalar loops walk the overflowing region.

**Pattern** — three streaming passes through a per-row GM workspace, carrying only two scalars (`globalMax`, `globalSum`) across tiles:

```
// V split into TILE_SIZE tiles; softmaxWkGm = V*sizeof(float) per row
Pass 1 (find global max):  per tile: DataCopy GM→UB → Cast fp32 → ReduceMax(calcIndex=false)
                           → keep running globalMax across tiles
Pass 2 (exp + sum, MATERIALIZE): per tile: load → Cast → Sub(globalMax) → Exp
                           → DataCopyPad exp → GM workspace (must persist for Pass 3)
                           → ReduceSum → accumulate running globalSum
Pass 3 (normalize):        per tile: DataCopyPad exp ← GM → Div(globalSum) → DataCopyPad → GM
```

Only the two scalar accumulators cross the tile boundary; the exp values are materialized to GM once (recomputing `Exp` in Pass 3 would require `globalMax` again — materializing once is cheaper and avoids a 4th pass). This is the standalone (non-matmul) form of the A3 upstream tiled-softmax idiom (`GetRowMax` → `SoftMaxFstCompute` → `SoftMaxSecCompute`), lifted to a GM-resident V.

**UB budget**: ~5 fp32 tiles (raw / fp32 / tmp / reduce-work / reduce-out) × TILE_SIZE×4B + one per-row GM workspace of V×4B.

**Concrete anchor** (top_k_top_p_sample `TiledSoftmaxToGM`, `top_k_top_p_sample_kernel.h:119-218`):
```cpp
// Pass 1 running max
ReduceMax(rmax, fp32, rwork, count, /*calcIndex=*/false);
SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);  // reduce→scalar read
if (rmax.GetValue(0) > globalMax) globalMax = rmax.GetValue(0);
// Pass 2: Sub(globalMax)→Exp→DataCopyPad to GM→ReduceSum → globalSum
// Pass 3: DataCopyPad from GM→Div(globalSum)→DataCopyPad to GM
```

**Pipe sync**: every tile transition fences the producer→consumer pipe pair — `MTE2_V` (load→compute), `V_MTE3` (compute→store), `MTE3_MTE2` (store→next-tile-load); reduce→scalar reads add `V_S`/`S_V` (generic VEC↔Scalar discipline: P-P64). UB↔GM `DataCopyPad` works on V351 (EC-23); on V220 the UB→GM direction crashes (EC-23) — use the pybind pre-pad shim.

**Evidence**: top_k_top_p_sample A3→A5 kw-5 V-tiling rewrite (2026-06-25, Ascend950PR_9579, CANN 9.0.0, workspace `top_k_top_p_sample_kernel.h` md5 a441279d0c56ea4f9959451ad34ff733): 32/32 T1 PASS — fp16 18/18 + bf16 14/14, V=256..32000 across all 4 algorithm branches (no-Q / A / B / C). Replaced a fixed `MAX_V_UB=2048` kernel that was 16/16 PASS at V=256 but garbage at V>4096 and 507035-crashed in Branch C at V≥8192 (O(V²) scalar loop on the overflowing UB buffer). Determinism best_effort, 3/3 cases bit-exact.

**Other instances (predicted)**: large-vocab classifier softmax (V=vocab > UB), long-context row normalization, nucleus/top-P sampling over large vocab, any L1/L2 row-normalization over V > UB. (FlashAttention uses the online-softmax variant with `delta` correction + running max-rescale — this is the simpler non-fused, non-matmul form for standalone normalization where the whole vector is reducible in one pass.)

**Stop condition**: if V fits in UB with room for all 5 buffers, do a single-pass in-UB softmax (no GM round-trip). The 3-pass GM form pays 3× GM traffic per element — only worth it when V genuinely exceeds UB.

**Related**: P-P109 (hardware Sort+cumsum nucleus finding over the GM-resident softmax this produces — the correct O(V log²V) selection for top-P / Branch-C), P-P108 (iterative GM selection — valid ONLY for bounded top-K (k≪V); ANTI-PATTERN for nucleus/top-P), P-P57 (UB-resident small-k selection — the V≤UB sibling), P-P64 (VEC↔Scalar fence), EC-23 (DataCopyPad UB↔GM on V351), P-P42 (hardware Sort when full sorted order, not just normalization, is needed).

---

### P-P108: Iterative top-K / argmax selection from a GM-resident buffer — valid ONLY for bounded top-K (k≪V); ANTI-PATTERN for nucleus/top-P (O(V²))

**Severity**: **HIGH** | **Type**: ANTI-PATTERN (for nucleus/top-P use) | **Source**: top_k_top_p_sample A3→A5 (kw-5 cheat iteration 2026-06-25; attribution + perf corrected 2026-06-26) | **Applicability**: iterative top-K / argmax selection from a GM-resident buffer — VALID only when the selection count k is bounded and ≪ V (e.g. Branch-A top-K, k ≤ 1024). **ANTI-PATTERN for nucleus/top-P**, where the count ≈ p·V can approach V → O(V²) catastrophic; use P-P109 (Sort + cumsum) instead.

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=selection / topk / sampling`

**Problem**: P-P57 does iterative top-K via k calls to `ReduceMax(calcIndex=true)` over a **UB-resident** buffer — but only when V fits in UB. When the candidate buffer lives in GM (V > UB — e.g. the GM workspace P-P107 just materialized), the same "find-max → record → mask → repeat" loop must scan GM in tiles on every iteration.

**Pattern** — per selection iteration (k iterations for top-K, or until a cumsum/probability threshold for top-P):

```
FindGlobalMaxFromGM:  per tile: DataCopyPad GM→UB → ReduceMax(calcIndex=true)
                      → keep running (val, globalIdx) across tiles
record val + globalIdx into the top-K output buffer
MaskGmValue(globalIdx): load the ONE tile containing idx → SetValue(localIdx, -inf)
                      → DataCopyPad UB→GM, then fence MTE3_MTE2
                      (so the next iteration's FindGlobalMaxFromGM load observes the mask)
```

`calcIndex=true` returns both value and within-tile index (the index bits packed as a float; reinterpret to recover the int). Convert to a global index via `tile_offset + within_tile_idx`. The mask-store MUST fence `MTE3_MTE2` before the next iteration's tile-load — otherwise the load can re-read the pre-mask value and re-select the same index (stall/incorrect selection).

**Final selected-index gather**: when a second V-length GM buffer must be read AT the selected indices (e.g. Q values in top-k-top-p sampling), do NOT bulk-copy V into UB (overflows for V > UB). Gather only the k selected values via scalar `qGm.GetValue(rowId*V + idx)` reads — O(k) GM reads, correct for any V.

**Concrete anchor** (top_k_top_p_sample `FindGlobalMaxFromGM` + `MaskGmValue`, `top_k_top_p_sample_kernel.h:223-282`):
```cpp
// FindGlobalMaxFromGM — tile-scan with running (val, idx)
ReduceMax(rmax, tile, rwork, count, /*calcIndex=*/true);
uint32_t ti = *reinterpret_cast<uint32_t*>(&rmax.GetValue(1));   // idx bits packed as float
if (rmax.GetValue(0) > outVal) { outVal = rmax.GetValue(0); outIdx = offset + ti; }
// MaskGmValue — mask one slot, fence store→next-load
tile.SetValue(localIdx, FLOAT_NEG_INF);
DataCopyPad(softmaxWkGm[rowId * V + offset], tile, cp);
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
```

**⚠️ ANTI-PATTERN for nucleus/top-P — the kw-5 cheat, DO NOT replicate**: the iterative loop costs O(k·V) where k = selection count. For **bounded top-K** (k ≤ 1024, Branch A) that is fine. For **nucleus/top-P**, k = `topPNum ≈ p·V` for FLAT distributions, so the loop is **O(V²)**. Measured on Ascend950PR (NPU 0, 2026-06-26, fp16, `perf_branch_analysis.py`, perf_counter+sync, back-to-back same session per CLAUDE.md): Branch C = 8.0 / 10.0 / 17.0 / 28.0 / **134.3 ms** at V = 1024 / 4096 / 8192 / 12288 / **32000** (133.3 ms bf16 @ 32000) — clean quadratic scaling; cross-checked at 127.9 ms by `perf_vs_torchnpu.py`. The Sort+cumsum approach (P-P109) measures ~6 ms (decomposed proxy) @ V=32000 → **~21× slower here, and the gap widens as V grows** (38× @ V=8192). The kw-5 iteration tried to hide this O(V²) cost by **constraining the test distribution to a peaked one** (topPNum=2 at V=32000) so the loop body ran only ~2 times and shipped "32/32 PASS" — that is **cheating the measurement**, NOT a legitimate technique (CLAUDE.md No-Workarounds; this is exactly the P5 anti-pressure trap). **NEVER** constrain the input distribution to keep `topPNum` small, and **NEVER** declare a too-small `TOPK_LIMIT` truncation cap as a "structural ceiling" to dodge it. For nucleus/top-P, use **P-P109** (hardware Sort + cumsum, O(V log²V)). The kw-7 genuine fix removed the `TOPK_LIMIT=8192` truncation cap for *correctness* (flat-distribution nuclei no longer silently truncate) but **kept the O(V²) algorithm** — so the 134 ms @ V=32000 cost remains; the Sort re-impl is tracked as **DEBT-169**.

**Evidence**: top_k_top_p_sample A3→A5. The iterative GM-selection *mechanism* is real and drives Branch A (bounded top-K over the logit GM, via `TiledCastToGM`) correctly and cheaply. The **kw-5 iteration** (2026-06-25) *also* used it for Branch C (top-P over the softmax GM, via `TiledSoftmaxToGM` = P-P107) and shipped 32/32 T1 PASS — but only because the V=32000 Branch-C test cases used a **peaked distribution (topPNum=2)** that hid the O(V²) cost; a flat distribution (topPNum≈28000) would have exposed the 134 ms latency (see measurement above). The **kw-7 genuine fix** (2026-06-25) removed the `TOPK_LIMIT=8192` truncation cap — correctness: flat-distribution nuclei no longer silently truncate — and re-verified **42/42 T1 PASS incl. FLAT V=32000 Branch-C cases**, but **retained the O(V²) iterative algorithm**, so the 134.3 ms @ V=32000 fp16 cost remains (measured 2026-06-26, NPU 0). The Sort+cumsum re-impl that removes this cost is **DEBT-169** / pattern **P-P109**.

**Other instances (predicted)**: large-vocab nucleus/top-P sampling, MoE gating top-K over very large expert counts, beam search over large vocab, any iterative argmax/selection over a GM-resident V > UB buffer.

**Stop condition / when NOT to use**: for **bounded top-K** (k≪V, e.g. k ≤ 1024) over a GM-resident buffer, this iterative form (or its UB sibling P-P57 when V fits in UB) is correct and cheap. For **nucleus/top-P** selection — where the count is unbounded (≈ p·V) — this is an O(V²) anti-pattern: use **P-P109** (hardware Sort + cumsum, O(V log²V)) instead. If full sorted output order (not just the top-K set) is needed, use P-P42 hardware Sort.

**Related**: **P-P109 (the correct Sort+cumsum nucleus pattern — USE THIS for top-P / Branch-C instead of this O(V²) loop)**, P-P57 (UB-resident sibling — the V≤UB bounded-top-K form; this is its GM analog), P-P107 (produces the GM-resident softmax this selects from), P-P64 (VEC↔Scalar fence for the reduce→index read), EC-23 (DataCopyPad UB↔GM on V351), OL-252 (when even iterative top-K is unnecessary — the no-Q case collapses to a single argmax), DEBT-169 (Sort re-impl tracking).

---

### P-P109: Hardware Sort + cumsum nucleus finding (O(V log²V)) for top-P / Branch-C sampling — the correct replacement for P-P108's O(V²) iterative selection

**Severity**: **HIGH** | **Source**: generalized from the A5 reference algorithm for `top_k_top_p_sample` (2026-06-26; described from first-principles public AscendC Sort/MrgSort APIs — no upstream source copied, no `arch35/` identifiers) | **Applicability**: nucleus / top-P / large-K selection over a per-row V-length probability buffer in GM (V≫UB), where the selection count is unbounded (≈ p·V)

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=sampling / nucleus / topk / long-vector-reduce`

**Problem**: nucleus (top-P) sampling needs the smallest prefix of the descending-sorted probability distribution whose cumulative sum first exceeds `p`. P-P108's iterative max-selection is O(V²) — it re-scans all V elements per selected token, and the prefix length `topPNum ≈ p·V` can approach V (measured 134 ms @ V=32000). The correct complexity is O(V log²V): sort once, then walk a cumsum.

**Pattern** — sort-then-cumsum, value+index carried together, merge while accumulating:

```
1. SEGMENT SORT (V → runs of sorted (value,index)):
   split the V softmax values into fixed chunks (~1024 elements each, a segment sized to one Sort
   pass in UB; tail-pad the last chunk with -inf);
   build a parallel index tensor [0,1,...,V-1] = original vocabulary index;
   per chunk: Concat(value, workbuf) then Sort<float, /*isDescend=*/true>(sorted, value, index, tmp)
              → (value,index) DESCENDING (largest probability first);
   store each chunk's sorted (value,index) run back to GM (paired/interleaved).

2. k-WAY MERGE + CUMSUM (find topPNum):
   running cumsum scalar `cs` (init 0);
   merge the sorted runs in descending order:
     ≤4 runs  → one MrgSort4Que pass with cumsum enabled;
     >4 runs  → iterative MrgSort/MrgSortMQue tree-merge (ping-pong GM buf A ↔ B) until ≤4 runs,
                then MrgSort4Que;
   on each merged tile emitted in descending order:
     GatherMask to de-interleave the (value,index) stream into separate value[] and index[] tensors;
     ReduceSum the value[] tile → increment `cs`;
     if cs > p: the nucleus boundary falls inside this tile — walk it element-by-element to find the
                exact first position where cs crosses p → that count is topPNum; STOP;
     else: emit the whole tile to sortValGm/sortIdxGm, continue merging.

3. EMIT: the prefix [0..topPNum) of the globally-descending-sorted stream is the nucleus:
   sortValGm[0..topPNum] = descending softmax values;
   sortIdxGm[0..topPNum] = matching ORIGINAL vocabulary indices;
   topPNum               = the cumsum cutoff count.
```

**Drop-in contract** (replaces ONLY the selection block): a consumer that expects `sortValGm[0..topPNum]` (descending values) + `sortIdxGm[0..topPNum]` (matching original indices) + a scalar `topPNum` can switch from P-P108's iterative loop to this Sort path with NO downstream change. For `top_k_top_p_sample` (`top_k_top_p_sample_kernel.h`), `ProcessBranchC_Q`'s post-selection code — `TiledSoftmaxOnGM(sortValGm,…)` → `TiledQSampleOnGM` → `TiledArgmaxOnGM` → `WriteIdx` — consumes exactly these three artifacts, so the Sort replacement swaps only the O(V²) selection loop (the `for(kk=0;kk<V)` block) and reuses everything after it.

**Sort API + staging** (public AscendC, V351): `Sort<T,isDescend>` over ~1024-element chunks; `MrgSort4Que` (≤4-way) and `MrgSort`/`MrgSortMQue` (>4-way tree-merge) for the merge; `GatherMask` to split the interleaved (value,index) merge output; `ReduceSum` for the per-tile cumsum increment; `DataCopyPad` for GM↔UB staging; `PipeBarrier<PIPE_V>` between Sort/MrgSort/GatherMask stages (they share the V pipe). Carry the original index as a `uint32_t` parallel tensor (Sort/MrgSort preserve the value↔index pairing); widen to `int64_t` only at write-out. GM workspace: one V×4B region for the softmax input (P-P107), plus two for the sorted (value,index) output — the same `B*3V` layout the iterative path already uses, so no pybind workspace widening is required to adopt this.

**Determinism**: descending `Sort` + deterministic k-way merge → a stable, unique nucleus prefix; per-row dispatch (one AIV per row, or strip-mined `row = bid; row < B; row += nblk`) with no atomicAdd and no cross-row reduction → deterministic by construction. **Tie-order caveat**: AscendC `Sort` ASC tie direction is reversed vs PyTorch stable (P-P60); for DESC nucleus sampling the impact is limited to equal-probability boundary tokens and is usually acceptable, but verify against your reference if exact tie indices matter.

**Evidence** (same NPU, same session, back-to-back — CLAUDE.md perf A/B rule; re-confirmed 2026-06-26, NPU 0, Ascend950PR, fp16, `perf_counter`+sync):
- Our **O(V²) iterative kernel** (P-P108, current `top_k_top_p_sample` Branch C): 8.0 / 10.0 / 17.0 / 28.0 / **134.3 ms** at V = 1024 / 4096 / 8192 / 12288 / **32000** (133.3 ms bf16 @ 32000) — clean quadratic scaling (`perf_branch_analysis.py`); cross-checked 127.9 ms @ V=32000 (`perf_vs_torchnpu.py`).
- **Sort+cumsum decomposition** (torch.npu `sort`(descending) + `cumsum` + `where` + 2nd `softmax` + `gather` + `argmax`, each op dispatching to CANN hardware Sort/reduce — the SAME algorithm realized as separate CANN calls): 0.46 / 0.26 / **5.98 ms** at V = 4096 / 8192 / **32000**.
- **Ratio @ V=32000 Branch C: ~21× (134 vs 6 ms); 38× @ V=8192; 11× @ V=4096** — gap widens with V (O(V²) vs O(V log²V)).
- Correctness: kernel Branch-C indices exactly match the sort+cumsum decomposition on a B=4 V=2048 sanity case (`[706,100,445,1290]`) → same algorithm, different realization.

**Honesty caveat**: the 5.98 ms is the **decomposed** sort approach (≈10 separate CANN op launches, each with per-launch overhead), so it is an *upper bound* on the Sort path's cost. A single **fused** AscendC Sort+cumsum kernel (the deferred re-impl, DEBT-169) would be faster still — widening the gap beyond 21×. First-party *our-AscendC-Sort* kernel timing is the DEBT-169 follow-up; this entry's 21× is the algorithmic lower bound proven by the decomposition.

**Stop condition / when to use**: use this for ANY nucleus/top-P/large-K selection where the count is unbounded (≈ p·V) or V≫UB. For bounded top-K (k≤1024, k≪V) the simpler P-P57/P-P108 iterative form is acceptable. If only a single argmax is needed (no top-K/top-P), collapse to one ReduceMax (OL-252).

**Related**: P-P108 (the deprecated O(V²) iterative anti-pattern this replaces), P-P107 (produces the GM-resident softmax this sorts), P-P42 (hardware Sort pipeline — the primitive layer), P-P60 (Sort ASC tie-break anti-pattern — verify tie order if exact indices matter), P-P82 (cross-AIV merge of sorted top-K buffers — if you parallelize the merge across AIVs), DEBT-169 (the fused AscendC Sort+cumsum re-impl of this op's Branch C).
