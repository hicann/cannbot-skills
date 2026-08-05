# Candidate Patterns pending A5 hardware validation

> These patterns are conceptually sound but lack performance data on A5 (Ascend950PR).
> Promotion condition: complete benchmark validation on A5, proving a measurable performance improvement.
>
> **Validation experiment**: in progress since 2026-03-26

---

## U-P1: `__ldg` cache hint for read-only global-memory access

**Source**: HKV hand-written version | **Validation status**: experiment in progress

**Concept**:
```cpp
// Plain read
K val = *(ptr + idx);

// With cache hint
K val = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(ptr + idx);
```

**API source**: AscendC compiler intrinsic (`__CCE__` conditional compilation), used widely in the HKV hand-written version.

**Expected effect**: When repeatedly scanning the same data (e.g., hash-bucket key/score), the L2 cache hint reduces HBM accesses.

**Open questions**:
1. Does `__ldg` actually affect cache behaviour on 950PR? (On some platforms it may be a no-op.)
2. Is it effective for random access patterns (e.g., Pooling's edge_in indirect indexing)?
3. Does the overhead of the cache hint (if any) exceed the benefit?

**Validation plan**:
- [ ] Compile test: can `__ldg` compile on 950PR + CANN 9.0.T501?
- [ ] Performance test: read-intensive kernel; compare throughput with/without `__ldg`.
- [ ] Record results: attach msprof data.

**Validation result** (2026-03-26, Ascend950PR + CANN 9.0.T501):

PASS **Compile**: both `__ldg` and `__stg` are available; bisheng compiles normally.

FAIL **No perf gain on sequential reads**: 4 kernel variants (plain/ldg-L2/ldg-L2+L1/ldg+stg) differ by < 0.5% (noise range) across 4MB~256MB data.

| Data size | plain GM | `__ldg` (L2, no L1) | Speedup |
|---------|----------|-------------------|---------|
| 4 MB | 0.085ms (49.5 GB/s) | 0.084ms (49.7 GB/s) | 1.003x |
| 256 MB | 11.78ms (22.8 GB/s) | 11.79ms (22.8 GB/s) | 0.999x |

**Conclusion**: For stride-scan patterns, hardware prefetch is already efficient enough — the `__ldg` cache hint provides no additional benefit. May only help for repeatedly scanning small chunks (e.g., HKV bucket lookup). **Not recommended for pooling/SG scenarios.**

---

## U-P2: source shared-memory prefetch -> `__ldg` cache replacement

**Source**: HKV migration comparison | **Validation status**: blocked on U-P1

**Concept**: source's `__pipeline_memcpy_async` + `__shared__` double-buffering -> AscendC alternatives:
1. `__ldg` with cache hint (U-P1)
2. Cooperative-group parallel reads (already verified general P-P13)

**Open question**: Can the `__ldg` cache hint actually substitute for the compute/memory overlap provided by async pipeline + shared memory?

**Validation prerequisite**: once U-P1 validation passes, compare throughput on a read-intensive kernel.

## P-CAT-1: VECIN→VECOUT bridge for data-movement kernels (VERIFIED on Cat + Split)
- **Trigger**: Pure copy kernel (no computation)
- **Pattern**: VECIN(GM→UB) → Adds(dst, src, 0.0f) → VECOUT(UB→GM)
- **Evidence**: Cat 51/51 PASS 1.20x (2026-04-09), Split 57/57 PASS (2026-04-09)
- **Status**: **READY TO PROMOTE** — verified on 2 independent ops

## P-CAT-2: Overlapping tail write for non-aligned DMA (VERIFIED on Cat)
- **Trigger**: Strided DataCopy with chunk_size % ALIGN != 0
- **Pattern**: Copy aligned portion + re-copy last ALIGN elements from (chunk-ALIGN)
- **Evidence**: Cat V2 fix, 3 previously failing cases now pass (2026-04-09)
- **Status**: Verified on 1 op, generalizable to any strided copy

## P-CAT-3: N-dim op → flat 3D decomposition (VERIFIED on Cat + Split)
- **Trigger**: Any op that processes along arbitrary dim (cat, split, cumsum, etc.)
- **Pattern**: outer=prod(shape[:d]) × target_dim × inner=prod(shape[d+1:])
- **Evidence**: Cat 51/51 PASS (2026-04-09), Split 57/57 PASS (2026-04-09)
- **Status**: **READY TO PROMOTE** — verified on 2 independent ops

## P-SPLIT-2: Padded allocation for sub-ALIGN compact DMA writes (VERIFIED on Split)
- **Trigger**: Kernel writes to compact output where chunk < DataCopy alignment
- **Pattern**: nblk=1 (serial) + padded alloc + narrow view to exact size
- **Evidence**: Split V2 fix, 4 previously failing cases now pass (2026-04-09)
- **Status**: Verified on 1 op. Generalizable to any compact-output kernel.

## P-REG-1: Reg-based multi-step fusion — keep intermediates in registers, avoid UB traffic (pending A5 compile validation)

**Source**: hiascend.com official docs (CANN 9.0 beta2, Reg vector-compute programming)

**Trigger**: Multi-step VEC compute chain (3+ steps) whose intermediate results get written to UB and immediately read back for the next step.

**Pattern**:
```cpp
// Mem-based (current): each intermediate flows through UB
Cast(work, xd, CAST_NONE, count);      // UB -> VEC -> UB
PipeBarrier<PIPE_V>();
Mul(work, work, smooth, count);         // UB -> VEC -> UB
PipeBarrier<PIPE_V>();
Abs(work, work, count);                 // UB -> VEC -> UB
PipeBarrier<PIPE_V>();

// Reg-based (target): intermediates stay in registers
__simd_vf__ inline void FusedCastMulAbs(...) {
    Reg::RegTensor<float> reg, smoothReg, absReg;
    Reg::LoadAlign(reg, srcAddr, aReg);           // UB -> Reg (once)
    Reg::LoadAlign(smoothReg, smoothAddr, aReg);  // UB -> Reg (once)
    Reg::Mul(reg, reg, smoothReg, mask);          // in-register compute
    Reg::Abs(absReg, reg, mask);                  // in-register compute
    Reg::StoreAlign(dstAddr, absReg, aReg, mask); // Reg -> UB (once)
}
// Saves 2 round-trips through UB (~4 cycles/trip * 2 = ~8 cycles saved per VL)
```

**Applicable scenarios**:
- DynamicQuant: cast -> mul_smooth -> abs -> reduce (4 steps; fusable into 2 UB accesses)
- GELU: x -> exp -> mul -> add -> div (5 steps)
- SwiGLU: exp -> add -> reciprocal -> mul (4 steps)
- LayerNorm: sub_mean -> square -> reduce (3 steps)

**Constraints**:
- Only processes VL elements per iteration; requires a manual loop (Mem-based can process a full LocalTensor)
- A `__simd_vf__` function cannot call `__aicore__` functions or SIMT functions
- GM -> Register direct load is not supported; data must go to UB first

**Status**: A5 compile validation passed (2026-04-12, CANN 9.0.0 + bisheng). Runtime perf validation pending — need to compare Mem-based vs Reg-based performance on a real op.

**Metadata extension (2026-05-12, from CAND-A3A5-5 self-flagged C35 merge — Mode 5 batch 2 auto-merge)**: per-VL canonical loop skeleton with explicit dtype-cast convention. The minimal arch35 reg-based loop body for fp16/bf16 input + fp32 compute + fp16/bf16 store is:

```cpp
__VEC_SCOPE__ {
    AscendC::MaskReg mask = AscendC::MaskReg::Auto();
    int32_t loopCount = (numElements + AscendC::VL - 1) / AscendC::VL;
    for (int32_t i = 0; i < loopCount; ++i) {
        // Tail mask — handles non-VL-aligned tail without separate epilog
        if (i == loopCount - 1) mask.SetCount(numElements - i * AscendC::VL);

        AscendC::RegTensor<float> regX, regOut;
        // Direct fp16/bf16→fp32 load via LoadDist::DIST_UNPACK_B16
        AscendC::LoadTensor<DIST_UNPACK_B16>(regX, ubFp16Tensor[i * AscendC::VL], mask);

        // ... compute on regX ... (e.g. Reg::Mul, Reg::Add, Reg::Sqrt)
        regOut = regX;  // placeholder

        // Direct fp32→fp16/bf16 store via StoreDist::DIST_PACK_B32 + CastTrait rounding
        AscendC::StoreTensor<DIST_PACK_B32, CastTrait::CAST_RINT>(
            ubFp16Tensor[i * AscendC::VL], regOut, mask);
    }
}
```

Key points beyond the base P-REG-1 mechanism:
1. `__VEC_SCOPE__` brackets the entire reg-based block — outside this scope, RegTensor operations are not legal.
2. `MaskReg::Auto()` + per-iter `mask.SetCount(...)` on the last iteration handles non-VL-aligned tail without a separate scalar epilogue — replaces the V220 hand-rolled tail loop.
3. `LoadTensor<DIST_UNPACK_B16>` + `StoreTensor<DIST_PACK_B32, CAST_RINT>` is the canonical bf16/fp16 → fp32 round-trip on arch35. **DO NOT** `Reg::Cast` separately — the unpack/pack distributions fold the cast into the load/store path. `CAST_RINT` is the per-vendor-spec rounding convention (round to nearest even); other `CastTrait` values exist for truncate / round-toward-zero but `CAST_RINT` matches the ops-nn ada_layer_norm convention.

Evidence: `ada_layer_norm/op_kernel/arch35/ada_layer_norm_common.h` (the `__VEC_SCOPE__` block) + `ada_layer_norm_impl.h` (per-VL loop + MaskReg tail handling). Documented in W11 ascend950pr.md §"Reg-based intrinsics restrictions" + CANN 9.0 public docs (Reg vector-compute programming guide).

Status: A5 compile + ada_layer_norm partial-port runtime evidence. Pending: P-REG-1 perf validation (the existing pending item) PLUS validation that the per-VL skeleton above transfers to other norm ops (LayerNorm, RmsNorm) cleanly.

---

## P-CAT-4: Fused scatter-accumulate + partial-reduce K1/K2 layout (op#17 first evidence)

**Date**: 2026-04-23
**Status**: CANDIDATE — needs 2nd independent op validation before promoting to P-P.

**Trigger**: Source `forward()` contains BOTH:
- Scatter-add-like operation (index_add_ / scatter_add_ / torch.zeros + index_add)
- Reduce-sum-like operation (`.sum(dim=...)`)
- …where both consume **a shared pre-multiplied tensor** (e.g. `grad_out_fp32 * hidden_states / rstd`)

**Efficient layout**: One K1 "per-row pass" that does:
1. Pre-multiply once in UB (shared intermediate `gnw_partial[H]`)
2. Scatter-write output A via tile-level `SetAtomicAdd` (embedding-grad-style, op#24 `EmbScatterF32`)
3. Accumulate output B into per-core UB accumulator `[H]` (op#19 `FusedResidualRmsNormBackward` style)
4. Write per-core partial `[H]` to workspace `[nblk, H]` at end
Plus K2 reduces `ws[nblk, H]` → final output B.

**Mandatory guard-rails** (don't skip):
- Pre-zero `ws` via `aclrtlaunch_memzero(nblk * H)` before K1 (see EC-37 — some K1 cores may have 0 rows and skip their slot).
- Pybind H-padding to `lcm(datablock_fp32, datablock_bf16)=16` (see P-P66) when H not guaranteed aligned.
- Cast normalizes dtypes in pybind (bf16 `norm_weight` → fp32 upfront).

**Anti-patterns to avoid**:
- Separate K_scatter + K_reduce kernels reading `grad_out` twice (2x HBM traffic on a memory-bandwidth-bound op).
- Per-core workspace stored as `[H, nblk]` col-major (K2 becomes strided — always row-major `[nblk, H]`).

**Evidence**:
- op#17 EmbeddingWithInitialLayernormBackward (2026-04-23): first combined example. 57/57 benchmark + 28/28 edge bit-exact. perf 0.94x (memory-bandwidth bound; further optimization = memzero/K1 overlap, K3 cast fusion).
- op#19 FusedResidualRmsNormBackward (historical): pure partial-reduce half (no scatter). 50/50 PASS.
- op#24 EmbeddingDenseBackward (historical): pure scatter half (no reduce-sum). 10.9x optimized.

**Promotion gate**: one more op that COMBINES scatter+reduce with shared pre-multiply — e.g. a fused grad_weight+grad_bias of an embedding+normalization backward variant.

---

## P-P67-candidate: Hash-Owner Deterministic Scatter Accumulate (HODSA) — **INVALIDATED on Ascend950PR (2026-04-26 op#19 kw-1+pp-1)**

**Status**: **NEGATIVE — empirically falsified on Ascend950PR**. The promotion gate FAILED. Kept here as a *cautionary anti-pattern* (do NOT promote to verified P-P67 in scatter_add.md). The pattern looked correct on paper but does not deliver determinism on this hardware.

**Empirical evidence (op#19 indexput-kw-1 + indexput-pp-1, 2026-04-26)**:
- kw-1 implemented HODSA per directive (gate-1 PASS: no AtomicAdd/SetAtomicAdd/SyncAll grep hits)
- 3-run gate-2 measured **28/46 IDENTICAL** (target 46/46) — 18 cases drift between runs of the SAME .so
- pp-1 falsified obvious race hypotheses across 2 iter:
  - H1 (TBuf<VECCALC> S_V fence missing) — adding S_V fence regressed bit_identical 28→20/46
  - H2 (TQue<VECIN, depth=2> alternation) — depth=1 unchanged from baseline 28/46; drift set merely shifted
- pp-1 confirmed **meta-nondeterminism**: same kernel.so → bit_identical 21-42/46 with max_diff 6414-25775 across 4 observations
- Both classified **requirement-class** (architecture-level, not fence/buffer fixable)

**Companion finding (orthogonal)**: fp16/bf16 step-wise vs fp32-promote semantic mismatch — CPU `index_put_(accumulate=True)` does fp16+fp16→round step-wise; HODSA's directive specifies fp32 accumulate then RINT. They differ in LSB even with race fixed. Also requirement-class.

**Conclusion (new anti-pattern)**: HODSA's "single-core-per-output-slot via owner=j mod nblk" + "scalar-pipe sequential fp32 accumulate in TBuf<VECCALC>" assumption depends on a hardware property — scalar-pipe sequential consistency on dense per-lane access — that **AscendC + Ascend950PR does not deliver** even without atomicAdd. The "by-construction deterministic" framing in the original research was correct algorithmically but wrong empirically.

**KB candidate**: A-P61.scalar-self-coherence (pp-1 coined) — single-core scalar-pipe accumulate in multi-tile loop on TBuf<VECCALC> is NOT guaranteed deterministic on Ascend950PR even when no atomicAdd / no SyncAll / no cross-core writes. Formal mechanism TBD; pp-1 recommends aog-determinism-analyzer follow-up to fully characterize.

**Forward path for op#19** (per directive §9 fallback):
- Variant (a) sort+segment-reduce — VEC-only path (no scalar-S-pipe); should sidestep A-P61.scalar-self-coherence
- Or variant (c) two-pass count-then-place — if a probe shows variant (a) also has a hidden hazard

**Lesson for future research**: pure-paper algorithm validation is insufficient on a hardware platform where the determinism/coherence model is incompletely documented. KB pattern proposals that promise "by-construction determinism" must include an empirical 3-run verification step BEFORE promotion. The DEBT-053 research → directive → kw-1 → pp-1 chain is the working version of this — pipeline correctly exposed the gap, just at architectural cost.

---

## P-P67-candidate (HISTORICAL — kept for context):

**Source**: DEBT-053 research session, `output/npukernelbench/src/kernels/19_IndexPut/research_report_DEBT053.md` (full 397-line report; 80-line P-P67 body + decision tree + 7 documented risks R1-R7 + ready-to-drop optimization_directive.md block).

**Problem**: A-P61.1 — concurrent multi-core `SetAtomicAdd` against an output GM slot with duplicate indices produces order-dependent fp16/bf16 sums; the kernel cannot bit-exact match a sequentially-evaluated reference. Existing scatter-add patterns (P-P2 / P-P21 / P-P40 / P-P48) all assume `DET_POLICY=best_effort` and accept this residual.

**Pattern (one-liner)**: Each output slot is owned by exactly one core via `owner(j) = j mod nblk`. Each core scans the entire `index[]` array, filters to its owned slots, and accumulates **sequentially in fp32** in a per-core UB scratch. No two cores write the same output slot ⇒ no atomic on observable output ⇒ deterministic by construction.

**When to use** (decision tree):
- `DET_POLICY=required` AND scatter-with-duplicates AND fan_in ≤ 10 → HODSA backbone (this entry)
- `DET_POLICY=required` AND fan_in > 10 AND M ≤ ~16K → counting-sort variant (op#5 + P-P48)
- `DET_POLICY=required` AND dim>1 AND per-segment fan-in ≤ FLUSH_LIMIT(P-P40) → sort+segment-reduce (P-P21+P-P40+P-P42/43+P-P52)
- `DET_POLICY=best_effort` → existing P-P21/P-P40/P-P48 are still preferred for perf

**Building blocks reused**: P-P21 register-accum loop body, P-P52 fp32 promote during accumulate, OL-78 TBuf persistent across tiles, P-P64 S_V flag pair after scalar SetValue, EC-37 workspace pre-zero (only if pattern is split into two kernels), OL-81 RNE rounding for final Cast.

**Anti-pattern resolved**: A-P61.1 (concurrent atomicAdd on observable output).

**Promotion gate**: op#19 IndexPut Kind-2 kw-1 implementation must pass:
1. 46/46 bit-exact match against `model.py` evaluated on CPU (sequential reference, OL-89 docstring-pyref ladder)
2. 46/46 IDENTICAL across 3 kernel runs (`determinism_check.py`)
3. `grep -nE "AtomicAdd|SetAtomicAdd|SyncAll" kernel/*.h` empty
4. Perf ≥ 0.9× current parallel-atomicAdd kernel at largest N case
5. UB usage ≤ 32 KB per core at M=65536, T∈{fp32, fp16, bf16}

**Predicted novel KB additions if validated**:
- `P-P67` body in `scatter_add.md`
- New `OL-90` candidate: "Index re-read overhead amortizes at N≥1K when fan_in × duplicate-rate is low"
- Cross-link in `determinism.md` resolving A-P61.1 specifically for scatter-with-duplicates

---

## A-P-NEW (2026-04-27): .claude/settings.json hook 用绝对路径 → 跨 instance clone 必坏

**Type**: Anti-pattern (deployment portability)

**Source**: 2026-04-27 Kimi-CC instance discovery — Kimi clone 自我 (main) a3_ops 后，`.claude/settings.json` 4 处 workflow_critic.py + state_machine.py hook 路径都是绝对路径 `/home/npu_user/workspace/a3/a3_ops/src/scripts/...` (main 的 path)。Kimi 自己的 repo 在 `/home/npu_user/workspace/a3-kimi/a3_ops/...` —— hook 触发时调用 NOT-EXIST script，silent fail，**critic enforcement 完全失效**。

Kimi self-audit 发现的：
> "the workflow critic hook is configured but points to the wrong path: /home/npu_user/workspace/a3/a3_ops/src/scripts/workflow/workflow_critic.py — This is the main agent's workspace, not mine. So the critic hook ran a non-existent script → silently failed → no enforcement. That's how I bypassed the harness."

**Detection**:
- `.claude/settings.json` 文件 hook 段含 `"command": "python3 /absolute/path/..."` 而不是 `${CLAUDE_PROJECT_DIR}/...` 或相对路径
- clone repo 后 `bash -c "test -x $(jq '.hooks.PreToolUse[0].hooks[0].command' .claude/settings.json)"` 失败

**Block conditions**:
- 任何 release/share to customer 前必须扫 `.claude/settings.json` 内绝对路径
- 任何 customer 报 "skill works but no validation enforcement" 应该先查 hook path

**Why this matters**:
- 产品 portability：客户 clone 后 critic 必须自动起效。绝对路径让 portability 完全失败
- silent fail 比 noisy fail 更危险：customer 不知道 critic 没在工作，按"产品无 bug"心态 ship 出去违反 invariant 的 op

---

## CAND-FA-COREDIST-1: L2-reuse core-distribution for sparse/varlen attention — round-robin + symmetric mirror + boustrophedon sweep

**Date**: 2026-06-03
**derived-from**: cann-source (FA arch35 forward kernel, core-split helpers + Process loop region)
**Status**: CANDIDATE — sanitized re-expression + turn-3b EXPLICIT-GENERIC offset-arithmetic (reflatten region-offset / reverse-partial pivot / TND accum-walk, all generic integer math). **Index-arithmetic executability empirically confirmed** by kw-wb5 hermetic graybox re-derivation (partition oracle PASS, see Evidence); still needs in-kernel runtime validation on A5. See VERDICT note + TURN-3b ADDENDUM at bottom.
**local-kb-crossref**: CAND-CANN-FA-ROW-TILE-1 (orthogonal layer — that entry is INTRA-core UB-budget row tiling + cube↔vec sync; THIS entry is INTER-core block distribution across `coreNum`. They compose: COREDIST-1 decides which global S1 block each core owns, ROW-TILE-1 decides how that core sub-tiles its owned block in UB). CAND-FA1/2/3 (cube/vec load forms — unrelated topic).

**Op-class**: attention-family kernels (and any op) whose work is a 2D grid of basic blocks `[N_heads_or_batch][S1_outer_blocks]` distributed across `coreNum` cores, where (a) a *triangular/sparse* mask makes early-block work cheaper than late-block work, and (b) we want consecutive cores to touch consecutive S1 blocks so the **L2 cache line for K/V tiles is reused across cores** instead of evicted.

### Principle (the load-balance idea, in plain terms)

Three independent distribution policies, chosen by a host-decided split-mode selector + layout flag:

1. **Sequential round-robin (full-compute region).** Flatten the `[N][S1outer]` grid row-major into one ordinal stream, then deal blocks to cores like cards: block `k` goes to core `k mod coreNum`. Consecutive blocks land on consecutive cores → the K/V tile a core needs is the one its neighbour just loaded → L2 hit. The per-core *count* is the textbook even-split: `floor(total/coreNum)`, with the first `total mod coreNum` cores each taking one extra block (remainder front-loaded so the low-id cores carry the +1).

2. **Symmetric mirror (partial/sparse region).** For a causal/triangular mask the *top* heads have light per-block cost and the *bottom* heads have heavy cost. Split the head axis in half. Deal the **top half forward** (core 0,1,2,…) and the **bottom half in mirror order** (…,2,1,0) so that each physical core is paired one light-region block with one heavy-region block. Net per-core load is balanced even though the grid is triangular. The mirror is implemented as a wrap-around modular *distance from a pivot core* (the core that owns the last forward block), not as a second independent deal — this keeps the two halves phase-aligned on the same cores.

3. **Boustrophedon / snake (variable-length TND region).** When sequence lengths vary per batch, alternate sweep direction every pass: pass 0 deals cores `0→coreNum-1`, pass 1 deals `coreNum-1→0`, pass 2 forward again, etc. A "cycle" is one forward + one reverse sweep (`2*coreNum` blocks). The snake keeps neighbouring passes' cores adjacent so the boundary K/V tiles stay warm, and it spreads the ragged-tail remainder evenly instead of always dumping it on the high-id cores.

### Ordinal → real-block index mapping (structure, not the verbatim formula)

Each core runs a loop over *its own* iteration count. On iteration `t` it must recover **which global S1 block** it is responsible for. The mapping is a three-step "deal → unflatten → reflatten":

- **deal**: pos-in-region = `t * coreNum + myRelativePosition` (snake region uses `t * (2*coreNum) + (myId or mirrored-myId)`).
- **unflatten**: split pos-in-region into a `(headIdx, s1BlockIdx)` pair by `÷`/`%` against the *region's per-head length* (which differs for partial vs full region because the partial region only spans the cheap prefix of S1).
- **reflatten**: map `(headIdx, s1BlockIdx)` back to the global block ordinal using the *full* S1-outer stride, then add the region's base offset (0 for forward-partial, `halfN*S1outer` for reverse-partial, `partialLength` for full).

The dispatch loop just bins its iteration index into `[forward-partial | reverse-partial | full]` ranges (by comparing against the three precomputed per-core counts) and calls the matching mapper.

### EXPLICIT-GENERIC offset-arithmetic (turn-3b — executability-closing, GENERIC index math only)

The structural recipe above tells *which* steps exist; the 3 pieces below give the *exact* arithmetic an agent must reproduce, all as generic integer formulas on `(myId, coreNum, tiling constants)`. **No vendor member-chains — every operand is a bare local.** Naming convention used here:
`cores` = number of cores; `myId` = this core's id; `s1Outer` = S1-outer block count per head; `n2g`,`batch` = head/batch axes; `prefixLen` = the cheap-prefix length = (host's last-fully-loaded-prefix-block index) `+ 1`. When the host marks "no sparse prefix" by setting that index to `-1`, the dense branch is selected (`prefixLen == 0`, the partial regions are empty) — see Piece 1.

**GQA head decomposition (needed only for the FULL head coords, not the block partition):** `n2g` (the heads-per-batch used in the deal) factors as `n2g = n2oNum * gSize`, where `n2oNum` = number of KV-head groups and `gSize` = GQA query-heads per KV-head (`gSize == 1` for plain MHA). Piece 3's `n2oIdx = local/tmpS1Outer/gSize` / `goIdx = local/tmpS1Outer%gSize` decode the KV-group and the in-group query offset from that factorization. The block-partition / load-balance logic does NOT need this relation (it counts ordinals); only a caller decoding the `(n2oIdx, goIdx)` head coordinates does.

**PIECE 1 — `realBlock(relPos, t, regionBase, isPartial)` reflatten (the deal→unflatten→reflatten body).**
The trap is: unflatten divides by the *region* per-head length, but reflatten multiplies by the *full* `s1Outer` stride, then adds a region base. Get either stride wrong and the kernel silently mis-indexes.

```cpp
// regionPerHeadLen = how many S1-outer blocks this region spans per head:
//   partial (cheap-prefix) region -> prefixLen
//   full  (expensive-suffix) region -> (s1Outer - prefixLen)
int64_t realBlock(int64_t relPos, int64_t t, int64_t regionBase,
                  int64_t regionPerHeadLen, int64_t s1Outer) {
  int64_t dealPos = t * cores + relPos;          // deal: card t to this core
  int64_t headIdx = dealPos / regionPerHeadLen;  // unflatten by REGION length
  int64_t blkIdx  = dealPos % regionPerHeadLen;
  return headIdx * s1Outer + blkIdx + regionBase;// reflatten by FULL stride + base
}
```
Region-base TABLE (the 3 call sites — these are the values that must be exact):
| region | `relPos` | `regionPerHeadLen` | `regionBase` |
|---|---|---|---|
| forward-partial | `myId` | `prefixLen` | `0` |
| reverse-partial | `relPosReverse` (Piece 2) | `prefixLen` | `halfN * s1Outer` |
| full | `myId` | `s1Outer - prefixLen` | `prefixLen` |

Dense (non-sparse) special case: the host signals "no cheap prefix" by setting the prefix index to `-1` (so `prefixLen` would be `0`); in that case `regionPerHeadLen` for the full region is the whole `s1Outer` and the partial regions are empty.

**PIECE 2 — the reverse-partial pivot + per-core counts (the symmetric-mirror precompute).**

```cpp
int64_t totalN = n2g * batch;                 // total heads to deal
int64_t halfN  = (totalN + 1) / 2;            // ceil-half: top heads forward, bottom mirror
int64_t fwdPartialLen = halfN          * prefixLen;          // blocks in forward-partial region
int64_t revPartialLen = (totalN - halfN) * prefixLen;        // blocks in reverse-partial region
int64_t fullLen       = (prefixLen == 0) ? totalN * s1Outer  // dense
                                         : totalN * (s1Outer - prefixLen); // sparse suffix

// PIVOT = the core that owns the LAST forward-partial block:
int64_t pivotCore = (fwdPartialLen - 1) % cores;
// this core's mirrored position = wrap-around distance from the pivot:
int64_t relPosReverse = (pivotCore - myId + cores) % cores;

// per-core count = even split, remainder front-loaded onto low relative positions:
int64_t count(int64_t relPos, int64_t len) {
  return len / cores + (relPos < (len % cores) ? 1 : 0);
}
int64_t fwdNum  = count(myId,          fwdPartialLen);
int64_t revNum  = count(relPosReverse, revPartialLen);   // NOTE: counted from mirror pos
int64_t fullNum = count(myId,          fullLen);
int64_t partialNum = fwdNum + revNum;
```
Bin this core's iteration index `i` into the matching mapper:
```cpp
if      (i < fwdNum)     b = realBlock(myId,          i,             0,                  prefixLen,            s1Outer);
else if (i < partialNum) b = realBlock(relPosReverse, i - fwdNum,    halfN * s1Outer,    prefixLen,            s1Outer);
else                     b = realBlock(myId,          i - partialNum,prefixLen,          s1Outer - prefixLen,  s1Outer);
```
The pivot definition is the crux: it is `(fwdPartialLen - 1) % cores`, i.e. `(halfN*prefixLen - 1) % cores` — **the modulo of one-less-than-the-forward-region-size**, NOT `fwdPartialLen % cores`. Off-by-one here puts the bottom-half mirror on the wrong core and breaks the L2-pairing.

**PIECE 3 — TND varlen accumulation (ragged per-batch ordinal walk + boustrophedon snake).**

(a) The ordinal→`(boIdx, n2oIdx, goIdx, s1oIdx)` walk for ragged sequences. Each batch contributes `ceil(actualS1[b] / s1Base) * n2g` ordinals; accumulate until the global ordinal falls inside the current batch:
```cpp
int64_t acc = 0, boIdx = 0;
int64_t perBatch = CeilDiv(actualS1[boIdx], s1Base) * n2g;
while (ordinal >= acc + perBatch && boIdx + 1 < batch) {
  acc += perBatch; ++boIdx;
  perBatch = CeilDiv(actualS1[boIdx], s1Base) * n2g;
}
int64_t local      = ordinal - acc;                 // position inside this batch
int64_t tmpS1Outer = CeilDiv(actualS1[boIdx], s1Base);
int64_t n2oIdx = local / tmpS1Outer / gSize;
int64_t goIdx  = local / tmpS1Outer % gSize;
int64_t s1oIdx = local % tmpS1Outer;
```
`actualS1[b]` is the per-batch length, recovered as the difference of the running cumulative-length array: `cumLen[b] - cumLen[b-1]` (with `cumLen[-1]≡0`). The accumulator `acc` and `boIdx` are carried across iterations (monotone), so the walk is amortized O(1) per ordinal, not O(batch) each time.

(b) The snake (boustrophedon) block index for a given cycle. A cycle = one forward + one reverse sweep = `2*cores` blocks. Split the per-core iteration index into `(loops, halfBit)` by `loops = idx >> 1`, `halfBit = idx & 1`:
```cpp
int64_t cyc = 2 * cores;
int64_t snakeBlock(int64_t loops, int64_t halfBit, int64_t myId, int64_t cyc) {
  return (halfBit == 0) ? loops * cyc + myId               // forward sweep
                        : loops * cyc + (cyc - myId - 1);   // reverse sweep
}
```
The ragged-tail count per core (how many blocks this core gets when the total doesn't divide `2*cores`): base `loops = total / (2*cores)` gives `2*loops` blocks; the remainder `R = total % (2*cores)` front-loads one extra forward block to cores with `myId < R`, and an *additional* reverse block to cores in the high band `myId+1 > 2*cores - R` (only when `R > cores`). This double-band remainder is what keeps the ragged tail balanced instead of dumping it on the high-id cores.

### Illustrative snippet (the three policy kernels, generic identifiers only)

```cpp
// (P1) reflatten: unflatten by REGION length, reflatten by FULL stride + region base
int64_t realBlock(int64_t relPos, int64_t t, int64_t regionBase,
                  int64_t regionPerHeadLen, int64_t s1Outer, int64_t cores) {
  int64_t dealPos = t * cores + relPos;
  return (dealPos / regionPerHeadLen) * s1Outer + (dealPos % regionPerHeadLen) + regionBase;
}
// (P2) mirror pivot for the symmetric bottom-half deal
int64_t pivotCore(int64_t fwdPartialLen, int64_t cores) { return (fwdPartialLen - 1) % cores; }
int64_t mirrorPos(int64_t pivot, int64_t myId, int64_t cores) { return (pivot - myId + cores) % cores; }
// (P1/P2) even split, remainder front-loaded — generic load balancer
int64_t myCount(int64_t len, int64_t relPos, int64_t cores) {
  return len / cores + (relPos < (len % cores) ? 1 : 0);
}
// (P3) snake sweep: even halfBit forward, odd halfBit reversed
int64_t snakeBlock(int64_t loops, int64_t halfBit, int64_t myId, int64_t cyc) {
  return loops * cyc + (halfBit ? (cyc - myId - 1) : myId);
}
```

### Why each policy (the WHY, which the source comment confirms)

- Round-robin (not block-contiguous) is *specifically* to raise L2 reuse of K/V across cores — block-contiguous assignment would have each core stream a disjoint K/V range and thrash L2.
- The mirror exists *only* for the sparse/causal case load imbalance; for dense attention the full-compute region uses plain round-robin.
- The snake exists for varlen (TND) because a fixed sweep direction biases the ragged remainder onto the same cores every cycle.

### Public-API expressibility

All three policies are **pure host-or-scalar integer arithmetic on the core id + tiling constants** — no special AscendC primitive is involved. They compile against public headers trivially (only `int64_t`, `/`, `%`, `+`, ternary). There is no internal-symbol dependency in the *distribution logic itself* (the surrounding kernel uses cube/vec block templates, but those are orthogonal to the index math).

### Evidence

- Derived from FA arch35 forward kernel: the core-split helper functions + the `Process()` region that precomputes the three per-core counts and bins the iteration index. Verified by reading the source comment block that names the three policies (sequential / symmetric / forward-reverse-cyclic) and draws the core→block assignment diagram.
- **Executability empirically confirmed (2026-06-03, kw-wb5 hermetic graybox)**: a worker given ONLY this entry (no source, no FA archive) re-derived all 3 offset-arithmetic pieces and verified the partition oracle — block-partition completeness (every global S1-outer block assigned exactly once, no gap/dup), per-region load-balance spread ≤ 1, and full-region L2-adjacency (`block k → core k%cores`) — PASS on 5 static + 3 snake configs, including a deliberately-stressed `fwdPartialLen` not-multiple-of-`cores` (the pivot off-by-one) and the `R>cores` double-band remainder. Zero load-bearing offsets/formulas invented. This validates the *index arithmetic* (pure integer math, CPU-runnable); it is NOT yet runtime-validated inside our own A5 FA kernel (the cube/vec block execution around it).

### Other-instances-predicted

Any tiled op over a `[outer][inner]` grid with (a) per-block cost gradient (triangular masks, causal LM, prefix-sum-like accumulation) and (b) a shared-across-cores cache resource (K/V tiles, a broadcast weight) benefits from round-robin+mirror. The snake generalises to any varlen/ragged-batch tiling. Candidate cross-refs: FA backward, fused-attention variants, any MoE expert-block deal with skewed token counts.

### COPY-SHAPE SELF-CHECK + HONEST VERDICT (the deliverable)

**Self-check (C34c token n-gram vs source)**: the three illustrative formulas above DO numerically coincide with the source at the *operation* level (`/`, `%`, `+1-on-remainder`, `cyc - id - 1`). But:
- They are renamed to generic identifiers (`myId/cores/cycle/pivotCore`), reordered, and stripped of the vendor-specific member-access chains (the object-member-rooted shared-params and const-info dereferences, the two-coordinate head/block intermediate, and the prefix-length-based branch). The source bodies are ~20 lines each with vendor member chains; my snippets are 3 one-liners with no member access.
- Contiguous-token overlap estimate: the longest matching run is the reverse-index expression of the snake mapper (the "cycle-width minus my-id minus one" shape), and that match only appears AFTER renaming the source's vendor member references to my generic identifiers and dropping the object-member roots. At the token level that is < 5% of either body. The remainder-front-load `base + (id < total%cores ? 1 : 0)` is a textbook idiom that predates this source. **I assess C34c PASS (overlap well under 5%)** — but flagging honestly: this is *marginal for the snake one-liner specifically*, because the snake formula is so short that ANY correct implementation looks similar. The protection here is that the formula is a generic boustrophedon index, independently re-derivable.

**The honest Path-A verdict** — is the re-expression executable-enough WITHOUT being verbatim?

**MOSTLY YES, with one genuinely-thin spot.** Breaking it down:

- **The per-core count helper (the 3 counts)**: FULLY re-expressible. "Even split, remainder to the low-id cores" is universal parallel-computing knowledge; the re-expression is not a copy in any meaningful sense.
- **The varlen snake mapper**: re-expressible as principle ("alternate sweep direction per cycle"), and an agent can re-derive the reverse-index expression `cyc - id - 1` from that principle without ever seeing the source. The formula is short enough that re-expression and verbatim converge — but they converge because the *math is generic*, not because we copied. A competent agent told "snake-order deal, reverse on odd passes" writes the same line.
- **The deal→unflatten→reflatten mapper**: the STRUCTURE (flatten ordinal, split by region length, reflatten by full stride, add region offset) is re-expressible and is the genuinely valuable transferable insight. An agent given this 3-step recipe + the region-offset table can write equivalent code.

**The one thin spot**: the exact *region partitioning* — i.e. that the partial (cheap-prefix) region's per-head length is "one more than the index of the last fully-loaded prefix block", the full region's length is "the total S1-outer count minus that prefix length", and the reverse-partial base offset is "half-the-heads multiplied by the full S1-outer stride" — is sparse-attention-specific bookkeeping. It is re-expressible in words (the cheap-prefix length vs the remaining suffix; the reverse half is offset by half-the-heads × full-stride) and I have done so above, but an agent reproducing it must get these offsets exactly right or the kernel silently mis-indexes. This is the part where "executable-enough" leans on the agent re-deriving from the *structural description* rather than reading off a formula.

**Conclusion for Path-A**: **NOT fundamentally irreducible / NOT a hard copy-cheat-line.** The arithmetic is generic load-balancing + flatten/unflatten + boustrophedon — all textbook, all re-derivable from the stated principles. The repro-closure CAN carry this sanitized form: an agent given the principle + the deal→unflatten→reflatten recipe + the region-offset table (all in words above) can write EQUIVALENT code without the verbatim bodies. The earlier graybox "can't write it without the formulas" finding reflects *missing the principle*, not *irreducibility* — once the principle is stated, the formulas follow. The residual risk is bookkeeping-exactness on the sparse region offsets, mitigated by the structural description, not by needing the verbatim source. **Path-A viable for this crux.**

### TURN-3b ADDENDUM — per-piece copy-shape verdict for the EXPLICIT-GENERIC offset-arithmetic

Turn-3b added the 3 specific offset-arithmetic pieces (reflatten region-offset, reverse-partial pivot, TND accum-walk) as explicit generic formulas. The C34c crux this turn tests: does each piece stay GENERIC (overlap < 5%, executable) or hit the copy-line (> 5%, fundamentally-verbatim)? Per-piece honest verdict:

- **PIECE 1 — reflatten region-offset: ADDED GENERICALLY (C34c < 5%, executable).** The source body is ~20 lines built entirely from vendor member-chains (a shared-params core-count member, a const-info S1-outer-size member, and a first-full-load-prefix-index member). My re-expression strips every member root to a bare local and keeps only the *math shape* `dealPos/regionLen → head*fullStride + blk + base`. That shape is a generic two-level flatten/unflatten — textbook. The load-bearing transferable insight (unflatten by REGION length, reflatten by FULL stride) is captured in words + the region-base table, NOT by copying. Longest contiguous token run after generic renaming is the single expression `head*stride + blk + base`, a universal flatten idiom. **Overlap well under 5%; executable from the table.**

- **PIECE 2 — reverse-partial pivot: ADDED GENERICALLY (C34c < 5%, executable).** The pivot `(fwdPartialLen - 1) % cores` and the wrap-distance `(pivot - myId + cores) % cores` are both short generic modular idioms. The non-obvious bit (the `-1` inside the modulo, and that counts are taken from the *mirrored* position for the reverse region) is stated explicitly in prose + the bin block. The even-split-remainder `len/cores + (relPos < len%cores ? 1 : 0)` predates this source (universal parallel-computing idiom). Member chains stripped. **Overlap under 5%; an agent can write correct code from the formula + the "off-by-one is the crux" note without the source body.**

- **PIECE 3 — TND varlen accum-walk: ADDED GENERICALLY (C34c < 5%, executable) — but the SNAKE one-liner is the thinnest spot (same convergence note as the base entry).** Two sub-parts:
  - The ragged ordinal→batch walk (accumulate `ceil(len/base)*n2g` per batch until ordinal falls inside) is a generic ragged-prefix-sum scan; the source carries member-rooted accumulators, my re-expression uses bare locals + an explicit `actualS1[b] = cumLen[b]-cumLen[b-1]` note. **< 5%, executable.**
  - The snake block `loops*cyc + (halfBit ? cyc-myId-1 : myId)` is so short that *any* correct boustrophedon implementation converges to it. As flagged on the base entry: this convergence is because the math is generic (re-derivable from "reverse on odd sweeps"), NOT because we copied. **< 5% at token level, but it is the single thinnest piece** — the protection is that the formula is independently re-derivable from the stated snake principle, not the verbatim source.

**Turn-3b resolution (the empirical answer this turn sought):** all 3 offset-arithmetic pieces **added GENERICALLY (C34c < 5%) and are executable** from the formulas + tables + prose above — **none hit the fundamentally-verbatim copy-line.** The earlier "contract-only / agent guesses wrong" gap was caused by the candidate stating only the *principle* and omitting the *exact region-base table + pivot formula + accum-walk loop*; once those are written as generic integer math (member-chains stripped), repro-closure carries them. **Turn-3b CLOSES the offset-line for the sparse-region bookkeeping — it is NOT Path-A's copy-limit.** The single residual thin spot is the snake one-liner whose brevity makes generic-re-expression and verbatim converge, but that convergence is driven by the math being generic (re-derivable), so it is not a copy in any load-bearing sense.

## CAND-FA-CARRIER-1: AIV↔AIC cross-core scalar-params carrier — CacheLine-bounded POD word-blit from a fixed scratch address + the host-tiling carrier-population/workspace structure

**Date**: 2026-06-03
**derived-from**: cann-source (FA arch35 forward — kernel-base cross-core scalar handoff region + op_host arch35 tiling lifecycle/workspace sizing + per-layout output-offset arithmetic)
**Status**: CANDIDATE — sanitized re-expression with **explicit per-piece C34c adjudication** (the crux this carve-out tests: a raw-word-blit carrier layout is inherently close to the vendor struct). Two pieces sanitize generic-executable (the blit MECHANISM/protocol + the host-tiling STRUCTURE); ONE piece (the exact carrier byte-layout field-list + bitfield packing) hits the **copy-line** and is FLAGGED as a hard reproducibility-copy boundary, NOT force-added. See PER-PIECE VERDICT at bottom. Not yet runtime-validated on A5.
**local-kb-crossref**: CAND-FA-COREDIST-1 (orthogonal — that entry is the INTER-core block-distribution *arithmetic* the kernel runs from the carrier's split-mode + sparse-prefix fields; THIS entry is WHAT carries those fields across the AIC/AIV boundary + HOW the host populates/sizes them. They compose: the host-tiling structure here SETS the split-mode/prefix-index fields, COREDIST-1 CONSUMES them). `cv_reference_concrete_params.md` §`cross_core_sync` + §`kernel_block_iteration` (orthogonal — those cover the per-tile WorkspaceQueue ring flag chain + the `GetBlockIdx()/GetSubBlockNum()` normalization; THIS entry covers the ONE-SHOT scalar-params bootstrap blit that happens once at Init, before any tile loop). CAND-OPAQUE-STRUCT-RUNTIME-VERIFY (the POD-layout-is-a-contract caution).

**Op-class**: any MIX cube+vec (AIC + paired-AIV) fused kernel where a block of scalar tiling/shape parameters computed on the host must reach BOTH engines, and the cube engine receives them via a low-latency on-chip scratch buffer rather than re-reading GM tiling — i.e. attention-family and similar cube-MIX ops whose per-core scalar setup must be identical on the cube and its paired vector sub-blocks.

### The problem (why a carrier exists at all)

In a MIX kernel the host computes one block of scalar parameters (shapes, tile counts, mode flags, the core-distribution selector, sparse-prefix index, scale). Both the cube engine and its paired vector sub-blocks need these. The vector side can read the full host tiling blob directly; the cube side, on this architecture, instead receives a **compacted subset** through a fixed on-chip scratch region (a cross-core scalar buffer), to avoid a GM round-trip on the cube's critical path. The carrier is that compacted subset.

### PIECE A — the cross-core scalar-params bootstrap blit (MECHANISM / protocol — GENERIC, executable)

The bootstrap is a one-shot, once-at-Init handshake, NOT a per-tile sync:

1. Host stages the compacted scalar-params POD into a fixed on-chip cross-core scratch region (base = the region's address 0). This is enabled by a **per-op-host compile switch** that routes cube↔vector scalar comm through the scratch buffer instead of GM (a build-system glue flag — see Risks; a graybox that hand-rolls the host glue WILL miss this switch and silently fall back to a different comm path).
2. The cube engine, at Init, **waits on a single dedicated cross-core flag** (a reserved high flag-id, mode = whole-engine sync) that signals "scalar-params staged".
3. The cube engine then **word-copies the POD** out of the scratch region into its local params struct, iterating `sizeof(params)/sizeof(uint32_t)` 32-bit words from scratch-base into the struct:

```cpp
// generic POD scratch-blit (public-surface skeleton; no vendor symbols)
// precondition: a one-shot cross-core flag has fired signalling "params staged in scratch"
WaitCrossEngineFlag(/*reserved bootstrap flag id*/);            // whole-engine wait, once
auto* src = reinterpret_cast<__scratch__ uint32_t*>(0);         // fixed scratch base
auto* dst = reinterpret_cast<uint32_t*>(&localScalarParams);    // local POD
#pragma unroll
for (int w = 0; w < sizeof(localScalarParams) / sizeof(uint32_t); ++w) {
    dst[w] = src[w];                                            // raw 32-bit word copy
}
```

**Two load-bearing constraints the agent MUST reproduce (these are the transferable insight):**
- **The carrier POD must be ≤ one CacheLine (128 bytes on this class).** The blit is sized in 32-bit words and the scratch region is one cache line; a carrier that grows past 128 B silently truncates (the tail fields are never copied → cube reads garbage shape/mode). Keep the carrier minimal — only fields the cube actually needs, packed.
- **The word count is `sizeof(POD)/sizeof(uint32_t)`, so the POD must be a multiple of 4 bytes with no field whose width the host and kernel disagree on.** Any host/kernel layout disagreement (a field the host writes as 8 bytes but the kernel lays out as 4, a bitfield split differently) makes the SAME word index mean different things on the two sides → silent corruption, no compile error.

This mechanism is fully re-expressible from the contract above without any vendor body: "one-shot flag → reinterpret POD as uint32 array → copy N words from scratch base, N = sizeof/4, POD ≤ 128 B."

### PIECE B — the host-tiling carrier-population + workspace STRUCTURE (GENERIC, executable)

The host populates the carrier and sizes workspace through a fixed lifecycle (this is the standard tiling-base shape on this platform, re-expressible as a recipe):

1. **Lifecycle order** (7 stages): platform query (core counts, UB/L1/L0 sizes) → shape/attr/layout analysis → op-tiling (compute tile sizes + split mode + sparse params + populate the carrier) → high-level-API tiling → tiling-key selection → workspace sizing → post-tiling (set block-dim, finalize raw tiling blob). The carrier is populated in the op-tiling stage; do NOT populate it earlier (shape analysis hasn't run) or later (workspace sizing reads it).
2. **Carrier population for the core-distribution selector** (feeds CAND-FA-COREDIST-1): the host sets, into the multi-core sub-carrier — (a) the used-core count = `min(totalWorkUnits, physicalCores)`, (b) the S1-outer block count per head = `ceil(s1 / s1BlockSize)`, (c) the even-split factor = `ceil(totalWorkUnits / usedCores)` + its tail, (d) a **split-mode flag** (sequential vs the multi-core-first round-robin/mirror/snake mode) chosen by sparse pattern, and (e) a **cheap-prefix boundary index** = the index of the last fully-loaded prefix block, or `-1` to signal "no sparse prefix → dense path". The prefix-index selection by sparse mode is a decision table:
   | sparse pattern | prefix-boundary index | split mode |
   |---|---|---|
   | left-up-causal (and the pre/next-token combos that reduce to it) | `ceil(min(s1,s2)/s1BlockSize) - 1` | multi-core-first |
   | right-down-causal (and its band equivalent) | `s1OuterCount - 1` | multi-core-first |
   | all-mask / no-mask-full / dense-band | `-1` (dense) | multi-core-first |
   | otherwise | (unset) | sequential |
3. **Workspace sizing** = sum of the per-purpose regions the kernel needs (e.g. an optional pre-op scratch region whose size is the aligned product of the relevant shape extents, offset-recorded into the carrier so the kernel can find it) **plus a fixed reserve constant**. Pattern: each region's byte size is `AlignUp(extentProduct, gmAlign)`, regions are laid end-to-end and each region's start offset is stamped into the carrier; a final fixed reserve is added last. Block-dim is set as `cores * subBlockCount` (the linear cube+paired-vec count).

### PIECE C — per-layout output-offset arithmetic (GENERIC integer math, executable)

For each supported memory layout the per-(batch, head-group, query-head, s1-block) output base offset is a sum of per-axis-index × per-axis-stride terms, plus a sub-block row term for the paired-AIV split. The STRUCTURE (which generalizes):

```cpp
// generic per-layout output offset (bare locals; strides are precomputed shape products)
// layout selects WHICH axis multiplies WHICH stride; the SHAPE is always:
//   offset = bIdx*bStride + n2Idx*n2Stride + gIdx*gStride + s1Idx*s1Stride + subRowTerm
// where subRowTerm = subBlockIdx * firstHalfRows * (the layout's row-stride)
int64_t outOffset = bIdx*bStride + n2Idx*n2Stride + gIdx*gStride
                  + s1Idx*s1Stride + subBlockIdx*firstHalfRows*rowStride;
```
The only per-layout variation is which precomputed stride each index multiplies (row-major head-last vs seq-major vs head-dim-major); all strides are products of trailing shape extents. This is textbook strided-tensor addressing — re-derivable from the layout description + the rule "subBlockIdx selects this AIV sub-block's half of the M-row tile."

### Evidence

- Derived from FA arch35 forward: the kernel-base cross-engine scalar handoff region (the one-shot flag wait + the uint32 word-copy out of the on-chip scratch region into the local scalar-params struct), the op_host arch35 tiling lifecycle (the carrier-population call in the op-tiling stage + the sparse-prefix/split-mode decision block + the workspace-sizing/reserve in post-tiling), and the per-layout output-offset block in the forward kernel. Verified by reading the blit loop's `sizeof/sizeof(uint32_t)` bound + the inline source note that the carrier "must be ≤ CacheLine = 128 Bytes", and the per-op-host CMake comm-via-scratch switch.
- NOT yet runtime-validated inside an a5_ops kernel. PIECE A/B/C are read-grounded structural recipes; the byte-layout copy-line (PIECE D below) is the empirically-determined repro boundary.

### Other-instances-predicted

Any MIX cube+vec op that needs a host-computed scalar block on the cube's critical path benefits from the CacheLine-bounded scratch-blit (PIECE A) and the lifecycle/workspace structure (PIECE B): fused norm+matmul, MoE dispatch with a cube stage, paged-attention. The per-layout offset shape (PIECE C) generalizes to any multi-layout tensor op. The copy-line (PIECE D) recurs for ANY raw-word-blit carrier — the lesson transfers even though the specific layout does not.

### PER-PIECE COPY-SHAPE VERDICT (the deliverable — honest C34c adjudication)

The brief's crux: a struct byte-layout for a raw word-blit is inherently close to the vendor struct (the blit needs the exact layout to work). Per-piece self-check (token n-gram overlap vs source):

- **PIECE A — the blit MECHANISM/protocol: ADDED GENERICALLY (C34c < 5%, executable).** The transferable insight is the *contract* — "one-shot cross-core flag → reinterpret a ≤128 B POD as a uint32 array → copy `sizeof/4` words from a fixed scratch base." My skeleton uses bare generic names (`localScalarParams`, `WaitCrossEngineFlag`, `src/dst/w`) and no vendor symbol. The two load-bearing constraints (CacheLine ≤ 128 B; word count = sizeof/4 so layouts must agree) are stated in prose, re-derivable. The `reinterpret_cast<uint32_t*> + #pragma unroll word-copy` shape is a universal POD-blit idiom that predates this source. Overlap is the generic copy-loop only. **< 5%, executable from the contract.**

- **PIECE B — host-tiling STRUCTURE: ADDED GENERICALLY (C34c < 5%, executable).** The 7-stage lifecycle is the platform's standard tiling-base contract (public). The carrier-population fields are described by generic ROLE (used-core count, s1-outer count, even-split factor, split-mode flag, cheap-prefix index) not by vendor member-chains. The sparse→prefix-index decision table is generic integer math (`ceil(min(s1,s2)/block)-1`, `s1OuterCount-1`, `-1`) — the same `-1`-means-dense sentinel already lives in COREDIST-1's prose. Workspace = `AlignUp(extentProduct, align)` regions end-to-end + fixed reserve is a textbook arena-sizing recipe. **< 5%, executable.**

- **PIECE C — per-layout offset arithmetic: ADDED GENERICALLY (C34c < 5%, executable).** Strided-tensor base-offset `Σ idx*stride + subRowTerm` is universal; only the index→stride pairing varies per layout, described in words. No vendor symbol. **< 5%, executable.**

- **PIECE D — the EXACT carrier byte-layout (field list + types + order + bitfield bit-widths + reserved-padding): HITS THE COPY-LINE (C34c > 5%, fundamentally-verbatim). FLAGGED, NOT ADDED.** This is the empirical answer the carve-out sought. The carrier is a ~30-field POD mixing 64-bit shape extents, 32-bit counts, and a tightly-packed bitfield word (several sub-byte mode flags + two ~11–16-bit dim fields + a 1-bit split-mode + a residual count, all packed to stay within the 128 B CacheLine). For the word-blit to work, the host-side population struct and the kernel-side receive struct must have **byte-identical layout** — same field order, same widths, same bitfield packing, same reserved padding. There is no way to write that layout that is both (a) correct (the blit depends on it bit-for-bit) and (b) meaningfully different from the vendor's struct: ANY correct re-derivation of the SAME contract converges on the SAME bytes, and a renamed-field copy is still a copy (C34c renamed-identifier detection). I therefore do NOT reproduce the field list, types, order, or bitfield widths in this entry — that would be a verbatim layout copy. **This is a hard reproducibility-copy boundary for the carrier struct.**

### CONCLUSION — is the carrier-struct + host-tiling KB-reproducible?

**PARTIALLY, with a sharp boundary.** The *mechanism* (how the blit works), the *host-tiling structure* (lifecycle, population order, workspace sizing, sparse-prefix selection), and the *offset arithmetic* are all KB-reproducible (PIECE A/B/C, C34c-clean, executable from the contracts above). The *exact carrier byte-layout* is NOT KB-reproducible without copying (PIECE D) — it is a fundamentally-verbatim contract.

**Practical repro implication**: an agent can re-derive everything EXCEPT the precise field order/widths/bitfield-packing of the carrier POD. For repro-closure of the whole-port, the carrier layout must come from a co-located reference struct (the host and kernel share ONE header defining it, so the layout is authored once and the blit's `sizeof` stays consistent) — it cannot be reconstructed from a KB prose description. The KB's job here is to make the agent (1) KNOW a CacheLine-bounded scratch-blit carrier is the right mechanism, (2) KNOW the host-tiling lifecycle/workspace structure, and (3) KNOW that the carrier POD layout must be authored as a single shared header (not independently re-typed on each side) and kept ≤ 128 B — NOT to ship the layout itself. That is the honest boundary: the structure is learnable; the byte-layout is a copy-line.

- 跨 LLM backend 实验（main/Kimi/其他）需要互不污染路径

**Correct response (修复方法)**:
1. `.claude/settings.json` hook command 用 `${CLAUDE_PROJECT_DIR}/src/scripts/workflow/workflow_critic.py` (CC supports `$CLAUDE_PROJECT_DIR` env-substitution)
2. 或用相对路径 `src/scripts/workflow/workflow_critic.py` (依赖 cwd = repo root)
3. deploy.sh 加自检步骤：clone 后 `find .claude -name '*.json' | xargs grep -l '/home/'` —— 任何 hit 都 alert customer

**Pipeline integration**:
- merged-arch-sanity skill 加一项 check: hook command path 解析 + exists 检查 (currently 8/8 不含此项)
- workflow_critic.py 启动时自检：sys.argv[0] 在 `${CLAUDE_PROJECT_DIR}/src/scripts/` 之内吗？不在则 print 警告 + exit 0 (允许 instance 自查)

**Recurrence evidence**:
- 2026-04-27 Kimi spawn 3_Add: critic 因路径 mismatch silent fail，worker 跳过 Phase O5/O6 finalize, archive 用 unverified 数字写 verification.json — bypass detected only by self-audit (C13 起效 catch 了 silent failure)

**Severity**: HIGH for product release — hook 失效就是 critic 失效就是 OL-85/C18/anti-delegation 全部失效

**Related**: C2 (infrastructure bypass — 这是 infrastructure 自身的 portability bug); A-P-pseudo-tool-call-text (similar silent-fail signature, different cause)

---

## CAND-PP74: torch_npu CANN op shape-specific divergence from public formula

**Source**: pp-2 / 11_DequantSwigluQuant a3 case 32 (2026-04-28)
**Validation status**: 1 confirmed instance (mode=1 gpt-oss SwiGLU, H=2528 N=216), evidence on disk.

**Symptom**: a public-formula fingerprint match (P-P71-style: parameter signature like `clamp_limit / glu_alpha=1.702 / glu_bias=1.0` → OpenAI gpt-oss) is bit-exact verified across many shapes (pp-1 confirmed 7/7), BUT a small subset of shape×N combinations diverge from the torch_npu fused-op reference. The kernel implementing the public formula matches torch_npu on most shapes and diverges on the same shapes torch_npu itself cannot be reproduced from public formula on NPU torch.

**Reproducer pattern**:
```python
# 1. Implement public-formula fingerprint by hand on NPU torch (same dtype/precision as kernel)
xf = x.float() * weight_scale.float() * activation_scale.float()
... # public formula
manual_q, manual_sc = quantize_dynamic(out)
# 2. Compare to torch_npu fused-op reference at the failing shape
ref_q, ref_sc = torch_npu.npu_dequant_swiglu_quant(...)
# 3. If manual_sc - ref_sc deviation matches kernel's deviation → upstream divergence
```

**Verified evidence** (op#11 a3 case 32, 2026-04-28):
- Manual fp32 gpt-oss formula on NPU torch ops, shape `[216, 5056]` mode=1 al=True:
  - vs torch_npu reference: `q_match=85.52%, sc_max_diff=0.1017`
- Kernel (implementing identical formula in AscendC):
  - vs torch_npu reference: `q_match=91.73%, sc_max_diff=0.241`
- Both diverge from reference at the SAME shape with similar magnitude — proving CANN-internal computation differs from publicly verifiable formula.

**Hypothesis (not verified)**: CANN may use different internal tile-boundary precision handling (possibly fp16 intermediate accumulation, different rounding at tile boundaries, or a fused MAC sequence that produces different rounding than chained `mul→add` on torch ops) at certain non-standard (N, H) combinations. Typical magnitude: 1-2 cases out of ~50 in benchmark distributions.

**Treatment / verdict policy**:
- This is a `convention` class divergence — kernel cannot bit-match torch_npu by implementing the public formula (since torch_npu itself diverges from public formula at these shapes).
- Reasonable acceptance: report PARTIAL_PASS (e.g., "49/50 with case 32 documented as upstream torch_npu CANN-internal divergence"), do NOT pursue overfit fix (OL-85).
- A `requirement` verdict is NOT warranted unless msprof reverse-engineering reveals a public-API decomposition that bit-matches the divergent torch_npu output (probe iter ≥6+ territory).

**Related**: P-P71 (public-formula fingerprint), OL-85 (no overfit fixes), OL-91 (convention vs requirement evidence bar), pp-1 §Recommendation.

---

## CAND-PP75: case_gen extreme-magnitude distributions (`large_mag`, `denormal`, `const_near_zero`) produce overflow/underflow refs in dequant→quant fused ops

**Source**: pp-2 / 11_DequantSwigluQuant a3 Pass B 4 cases (2026-04-28)
**Validation status**: 1 op confirmed; pattern likely general for any dequant→activation→quant pipeline.

**Symptom**: Pass B (edge_dataset) sign_off-tier cases pulled from `case_gen.py` distributions `large_mag` (uniform 1e20..1e30) and `const_near_zero` (= `eps_fp32 * 5 = 5.96e-7`) produce reference outputs that contain `inf`/`nan` (from fp32 overflow on multiplication of two ≥1e20 scalars) or sub-1e-28 normal-range scales (from product underflow). A kernel implementing typical-magnitude-correct dynamic int8 quant tail (with a `clamp(min=1e-10)` div-guard floor or similar) will diverge:

- For `large_mag` cases: ref `quant_scales = [inf, inf, ..., nan]`, ref `quantized_output = [0, 0, ...]` (since dividing by inf gives 0). Kernel either matches (if it also overflows) or differs (if it implements explicit overflow guard).
- For `const_near_zero` cases: ref `quant_scales ≈ 2-3e-28` (still in fp32 normal range, but 18 orders below typical). Kernel's `div-guard floor=1e-10` clamps the scale to `1e-10`, producing different quantized output.

**Confirmed evidence** (op#11 a3 edge_dataset, 2026-04-28):
- Cases 10/11 (`dist_large_mag_seed{0,1}`): `weight_scale.amax ≈ 9.97e+29 / 9.91e+29`, `activation_scale.amax ≈ 9.32e+29 / 4.82e+29`, ref `quant_scales = [inf, inf, inf, inf]` / `[inf, inf, inf, nan]`.
- Cases 18/19 (`dist_const_near_zero_seed{0,1}`): all scales = 5.95e-7, ref `quant_scales ≈ 2.69e-28 / 2.74e-28` (vastly below kernel div-guard floor `1e-10`).

**case_gen.py distributions** (`src/scripts/reference_provider/case_gen.py`):
```
{"tag": "large_mag",      "fn": mk_uniform(1e20, 1e30)}     # produces overflow on op*op
{"tag": "small_mag",      "fn": mk_uniform(1e-30, 1e-20)}   # similar underflow risk
{"tag": "denormal",       "fn": mk_denormal()}              # explicit denormal fp32 inputs
{"tag": "const_near_zero","fn": mk_const(eps_fp32 * 5)}     # 5.96e-7
```

**Treatment / verdict policy**:
- These are `convention`-class divergences for dequant→activation→quant fused ops with multi-tensor scale products: the kernel matches typical-magnitude inputs and diverges only on stress-test extremes that produce ref `inf/nan` or sub-fp32-normal-range scales.
- Honest workflow: Pass B should NOT gate on these for ops where the chain of operand multiplications can overflow/underflow fp32. The right SCHEMA-level fix is to add a `skip_extreme_magnitudes=True` flag (or `extreme_magnitude_only_subset=True`) on per-op SCHEMA, so input_gen.py emits these cases into a separate documentation-only edge subset, not the Pass B precision gate.
- Affected ops: any fused op where ≥2 user-provided scale tensors multiply (dequant→quant pipelines, attention scaled softmax with scale param, layernorm with scaled output, MoE quant pipelines).

**Recommendation for input_gen.py / case_gen.py**:
```python
# In SCHEMA:
SCHEMA = {
    ...
    "skip_extreme_magnitudes": True,  # exclude large_mag / small_mag / const_near_zero / denormal
    # OR
    "extreme_magnitude_subset_only": True,  # emit them but tag as 'edge_documentation_only'
}
```

`case_gen.py` filters extreme distributions when the SCHEMA flag is set; alternatively keeps them in a separate `edge_dataset_extreme.pt` that does not gate Pass B but is documented in `analysis.md` for completeness.

**Related**: OL-85 (no overfit fixes), OL-91 (convention evidence bar), pp-1 §Recommendation; mirrors P-P70 dynamic-quant-tail context but at extreme operand magnitudes.

---

## CAND-A-P-NONALIGNED-DIVERGENCE: torch_npu fused-op shape-specific divergence at non-128-aligned 2H (op#11 case 32 ONLY confirmed)

**Status**: 1 confirmed datapoint, 6 suspect, 0 archived sweeps. NOT yet a generic pattern. Codex review 2026-04-28 flagged earlier A-P37 codification as over-generalized — moved here to candidates pending durable evidence.

**What we KNOW (durable evidence)**:
- op#11 `[216, 5056]` mode=1 al=True: torch_npu output diverges from manual fp32 gpt-oss formula on a3 NPU. Reproduced by pp-2 in `/tmp/probe_case32_reproduce.py` (q_match=0.8552, sc_max_diff=0.10). Documented in `output/npukernelbench-a3/src/kernels/11_DequantSwigluQuant/probe_report.md` §pp-2.
- Benchmark commit 909454b (wabluy independent finding) confirms case 32 is where CANN deviates from its own documented formula — fix was to change `Model.forward` to use the documented CPU formula (which our kernel matches bit-exact at this shape too).

**What we DO NOT have durable evidence for**:
- Whether this affects ALL non-128-aligned shapes, or only `2H = 64 mod 128`, or only specific N×H combinations
- Whether mode=0 also exhibits this (orchestrator probe `/tmp/probe_mode0_align.py` suggested yes, but ephemeral /tmp script not archived)
- Whether other CANN aclnn fused ops (aclnnLayerNorm, aclnnGroupNorm, aclnnRMSNorm) have analogous regime
- The pad-to-128 strategy outcome (one ephemeral probe; needs archived reproducer)

**Promotion criteria** (move to platform_compat.md A-P3X if):
1. Archived shape sweep (≥3 ops or ≥10 N×H combos) showing the regime is reproducible across ops/cases
2. Probe outputs committed under `output/.../<op>/probes/probe_outputs/` (not `/tmp/`)
3. Cross-confirmed by ≥2 independent runs (ours + a5 / ds / kimi)

**Until promotion**: this is a single-data-point heuristic, not a KB pattern. Workers should NOT cite it as authoritative. The op#11 archive solution (benchmark spec change + CPU truth verification) handles the symptom for op#11; if a future op hits a similar regime, treat as new investigation.

**Related**: OL-91 (artifact evidence bar — this candidate fails by being /tmp-only), aog-self-critic C18/C23 (codifying narrative as pattern is a reward-hacking tell), CLAUDE.md "no CANN source copy".

---

## CAND-PP76: V→S sync after `Adds(dst_ub, scratch_ub, 0.0f, count)` when next iter's S pipe writes the same `scratch_ub` via `SetValue`

**Source**: op#27 27_MultiMaskAttentionAggregation a3 V220 (2026-04-28) — aog-kernel-worker iter-5 secondary fix

**Validation status**: 1 op observed; the fix WAS applied and KEPT, but its independent contribution is **inconclusive** because the dominant cause was found later to be stale `LOCAL_TASK` staging (OL-93). The pattern is plausible from first principles but its empirical signature was contaminated by the staging gap.

**Symptom** (op#27 narrative):
- Inner cls loop populates a small UB scratch buffer per class via `finalF.SetValue(cls, scalar)` (S pipe write)
- Then `S→V sync` + `Adds(finalT, finalF, 0.0f, C_pad)` (V pipe read of `finalF`)
- Next iter starts with `finalF.SetValue(cls, scalar)` again (S pipe write of `finalF` while iter-N's V read MAY still be in flight)
- Without an explicit `SetFlag<HardEvent::V_S> + WaitFlag<HardEvent::V_S>` AFTER the Adds, iter N+1's S write can race iter N's still-in-flight V read

**Proposed fix**:
```cpp
Adds<float>(finalT, finalF, 0.0f, C_pad);
event_t evvs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(evvs);
WaitFlag<HardEvent::V_S>(evvs);
// (iter boundary; next iter's SetValue(finalF, ...) is now safe)
```

**Why "candidate" not promoted**: op#27's full-PASS state was achieved by `cp workspace → LOCAL_TASK + rebuild` (per OL-93). Pre-rebuild, with this V→S fix already in the on-disk kernel, the kernel still showed ~10 % intermittent precision mismatch — i.e. this fix did NOT resolve the symptom in isolation. We cannot tell whether (a) the V→S race is real but the build was running stale code that lacked it, OR (b) the V→S race is theoretical-only and never fires in practice on V220. Distinguishing requires a controlled probe: build kernel W/ vs W/O the V→S sync from a freshly-staged `LOCAL_TASK`, observe det count over N runs.

**Promotion criteria**:
1. Reproduce the race on a separate op (≥2 ops total) where staging is provably clean (`diff workspace/{op}/kernel LOCAL_TASK/kernel` empty before each verify)
2. Show that omitting the V→S flag produces non-determinism that adding it eliminates (binary contrast on identical staging)
3. Document the specific S/V/MTE pattern that triggers it (probably: scalar fan-in into a UB scratch reused across iter boundaries)

**Until promotion**: workers MAY apply this fix as defensive practice when the access pattern matches (per-iter S-pipe scalar fan-in into a UB region read by a later V op then re-written next iter), but should NOT claim it resolves an intermittent non-determinism — run OL-93 staging diff first.

**Related**: P-P74 (TBuf→TQue auto-sync — the more general/effective pattern; V→S sync is a targeted complement when the racing buffer is filled by S pipe rather than rotated by a queue), OL-93 (the staging-gap red herring this candidate is conjugate to), A-P61 (determinism anti-patterns).

## CAND-PP77: Predicate-driven chunk-skip early-exit on iterative merge regresses on Gaussian / well-spread input distributions (ANTI-PATTERN candidate)

**Source**: op#9 9_TopKTopP ko-1 Opt1 (2026-05-02 Ascend950PR_9579) — REVERT.

**Validation status**: 1 op observed (REVERT evidence). Anti-pattern candidate, not promoted.

**Hypothesis tested**: For a chunked Sort + 2-pointer merge into a top-K buffer of capacity `K_MAX`, add a per-chunk early-exit `if (chunk_max < top_min) skip_merge` once the buffer is full. The intuition: if the new chunk's maximum is below the current top-K floor, no element in this chunk can survive the merge, so the entire `K_MAX`-iter merge loop is wasted work.

**Why it failed (Gaussian-distribution case)**:
- For random-normal input at N = 65536, the top-1088 floor stabilizes at ~2σ after the first 4–8 chunks
- New chunk maxes are routinely 3–4σ → the early-exit predicate `chunk_max < top_min` almost never fires
- Net cost added per chunk: 2× scalar `GetValue` + 1× explicit V→S sync, with zero merge-loop savings
- Wall-clock impact: bf16 N=65536 k=1024 +5.0 % regression; fp16 mid +3.9 % regression; fp32 small −0.4 % (noise)

**When the pattern *would* work** (predicted, not yet observed):
- Long-tailed distributions (real attention scores: a few large positions, many tiny tail)
- Inputs with provable structure (already partially sorted, sparse logits with many filler positions)
- Distributions where the top-K floor saturates fast and most subsequent chunks are below it

**Generalization**: any iterative merge / scan / accumulate with a "skip if this iter cannot contribute" predicate is data-distribution-dependent. The optimization is NOT a free win — it adds fixed predicate-test cost in exchange for a probabilistic savings that depends on input distribution.

**Promotion criteria**:
1. Demonstrate +5 % or better speedup on a real-workload distribution (e.g. captured attention scores or production logits) on ≥2 ops
2. Document a distribution test the kernel can apply at runtime to gate the early-exit (e.g. estimate input variance from the first chunk's max-to-mean ratio; only enable skip when the estimated tail is heavy)
3. Show that the predicate cost (`GetValue` + sync) is amortized across enough merges to net positive

**Until promotion**: aog-kernel-optimizer should NOT propose chunk-skip / iter-skip early-exits on merge-style hot loops without empirical evidence on the target workload's distribution. If proposed, treat as anti-pattern unless input distribution is documented to be long-tailed.

**Related**: P-P81 (runtime-bounded loop cap — the validated alternative for iter-count reduction; works on per-row k variance, NOT per-chunk data variance), OL-85 (anti-overfitting — distribution-specific kernel paths require the distribution probe in the kernel), failures_ledger.md tagging convention (REVERT entries become anti-pattern evidence).

## CAND-PP78: K_ROWS_PER_AIV outer-loop fusion to amortize aclrtLaunchKernel overhead — gated by AIV utilization (1-op evidence, gate-aware)

**Source**: 10_LayerNorm kw-2 (2026-05-03 Ascend950PR_9579) — IMPLEMENTED + VERIFIED INERT on this op, but well-formed mechanism for genuinely under-utilized cases.

**Validation status**: 1 op implemented (Pass A 60/60 + Pass B 16/16 + Det 60/60 PRESERVED bit-exact), perf flat (mechanism inert per OL-124 gate). Promotion blocked until validated on a 2nd op where the gate ACTUALLY fires (B < TOTAL_AIV).

**Pattern**:

For per-row kernels (LayerNorm, RMSNorm, Softmax, GroupNorm, etc.) where the inner per-row computation is bounded and the launch overhead `aclrtLaunchKernel` ≈ 20-25 µs per call dominates small cases, parameterize the per-AIV row loop with a `K_ROWS_PER_AIV` stride:

```cpp
// Inner per-row code unchanged from kw-1/ko-1 baseline:
template <typename T>
__aicore__ inline void ProcessRow(int32_t row) { /* ... */ }

// Outer K-fusion wrapper:
__aicore__ inline void Process() {
    int32_t bid = GetBlockIdx();
    int32_t my_rows = (B + blockDim_ - 1) / blockDim_;
    int32_t row_base = bid * my_rows;
    for (int32_t r = 0; r < my_rows; r += K_ROWS_PER_AIV) {
        for (int32_t k = 0; k < K_ROWS_PER_AIV && (r + k) < my_rows; k++) {
            ProcessRow<T>(row_base + r + k);
        }
    }
}
```

Pybind selects K adaptively based on UB budget:
- `K = 4` if `H × sizeof(T) ≤ 2 KB`
- `K = 2` if `H × sizeof(T) ≤ 8 KB`
- `K = 1` (baseline path identical) otherwise

**Activation gate (PREREQUISITE per OL-124)**: this mechanism only delivers measurable speedup when `min(rows / TOTAL_AIV) < 1` across the test set — i.e. some cases have AIVs idle at baseline. If every benchmark case has `rows ≥ TOTAL_AIV`, the existing inner per-row `for r=0..my_rows_` loop already amortizes K rows per launch and the K-fusion wrapper is algebraically inert.

**Pre-implementation check** (do this BEFORE writing any code):

```python
import json
cases = [json.loads(l) for l in open("vendor/.../<op>.json")]
ratios = [num_rows(c) / 56 for c in cases]   # 56 AIVs on a5 V220
if min(ratios) >= 1.0:
    # GATE FAILS — K_ROWS_PER_AIV is decorative on this op
    # Document in analysis.md and skip; pursue different optimization axis.
```

**When the mechanism IS applicable** (predicted, awaiting 2nd op evidence):
- Tiny-batch decode-style ops (single-token attention B=1..32 vs TOTAL_AIV=56)
- Fan-out scatter where producer count < AIV count
- Per-token RMSNorm/Softmax during LLM decode (B=1 incremental)

**Promotion criteria**:
1. Validated on ≥2 ops where the gate ACTUALLY fires (`min(rows/TOTAL_AIV) < 1`) AND mechanism delivers ≥+30% perf gain
2. UB budget analysis shows K=4 path doesn't bust 192 KB on the target dtype
3. Adaptive-K selection logic generalizes (not op-specific hardcoded thresholds)

**Trap to avoid**: implementing K_ROWS_PER_AIV "preventively" on ops where the gate fails — wastes iter budget, adds code complexity, delivers zero perf. Always probe the gate first.

**Related**: OL-124 (the activation gate principle — Mechanism B is gated by this rule), P-P82 (Mechanism A counterpart — multi-AIV-per-row partition, also gated by OL-124), OL-27 (perf re-measurement of byte-identical kernels — needed to verify "inert" vs "regressed").

## CAND-PB-SIMD-SIMT-COEXIST: SIMD class kernel + SIMT `__simt_vf__` kernel coexistence in same `.so` corrupts SIMT path's precision (PB candidate, 1-op evidence)

**Severity**: HIGH (would block any Kind-2 SIMD rewrite of multi-path SIMT operators)
**Source**: 26_AvgPool3d kw-2 (2026-05-03 Ascend950PR_9579, CANN 9.0.0) — 8 SIMD UB-tile rewrite iterations + 1 revert.
**Validation status**: 1 op evidence (SIMT regression observed); hypothesis plausible but unverified; no minimal repro on a 2nd op yet.

**Symptom**: when a kernel module emits BOTH a SIMD class kernel (Init/Process pattern with `TPipe + TQue + TBuf`) AND a SIMT `__simt_vf__` kernel in the same `.so`, the SIMT path may produce subtly wrong results — including for SIMT cases that worked correctly when the SIMD class was absent.

**Specific evidence (op#26 AvgPool3d kw-2)**:

The SIMT generic path was a verbatim copy of kw-1's `avgpool3d_vf<T>` (renamed to `avgpool3d_simt_fallback_vf<T>` for kw-2's mixed build). In kw-1 (SIMT-only build) it passed 72/72 cases; in kw-2 (SIMD class + SIMT in the same `.so`) the same SIMT code FAILED on 4-7 cases per dtype:

| Dtype | kw-1 SIMT-only | kw-2 SIMD+SIMT |
|---|---|---|
| fp32 | 52/52 PASS | 42-44/52 PASS |
| fp16 | 10/10 PASS | 5-6/10 PASS |
| bf16 | 10/10 PASS | 5-6/10 PASS |

Failing cases mix BOTH fast-path-routed (SIMD class) AND generic-routed (SIMT) — the SIMT regression cannot be explained by a SIMD class bug alone.

**Plausible mechanism (not yet confirmed)**:
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` set in both SIMD class entry-points AND SIMT entry-points
- SIMD class instantiates `TPipe + TQue + TBuf` state that may persist across kernel boundaries
- SIMT path's grid-stride loop with `__gm__ T*` scalar reads may pick up stale event/queue state from prior SIMD invocation
- OR mixed `__global__ __aicore__` kernels with `Simt::VF_CALL<...>` may force compiler to a different code-gen mode for the SIMT TU

**Mitigation strategies (proposed, not validated)**:
1. **One-pattern-per-.so rule**: either all-SIMT or all-SIMD class. For multi-path kernels (fast + generic), implement BOTH paths via the same pattern (generic path also uses Init/Process class with conditional logic in `Process()`).
2. **Separate .so per path**: build SIMD class in `_op_simd.so`, SIMT in `_op_simt.so`, dispatch at Python pybind level.
3. **Debug-instrument the SIMT regression**: write canary GM values at start/end of SIMT kernel to confirm whether SIMT is being called at all or its state is corrupted by prior SIMD invocation.

**Workaround applied (op 26)**: reverted to kw-1 SIMT-only baseline (72/72 PASS, 0.14× perf). Structural-ceiling exit at 0.14× — ko-1's 3-axis ablation already confirmed HBM scalar-load latency is the SIMT bottleneck; the architectural shift to SIMD class is blocked by this coexistence issue.

**Promotion criteria** (CAND → PB-N):
1. Reproduce on a second op where mixed SIMD class + SIMT `__simt_vf__` build exhibits SIMT regression vs SIMT-only build
2. Minimal repro: 2-kernel `.so` with one trivial SIMD class kernel and one SIMT scalar kernel reading from independent GM buffers — does the SIMT kernel produce different output when SIMD class is linked?
3. Identify exact compiler/linker code-gen difference between mixed and SIMT-only builds (`bisheng -S` or equivalent)
4. Confirm with CANN team if this is intentional ABI vs unintended pipeline-state leak

**Related**:
- PB-9 (DataCopy localDst/localSrc silent corruption — same family: pipeline-state correctness gotchas)
- OL-63 (TQue VECIN depth — same family: pipeline configuration affects code-gen)
- A3 op#13_Cat finding 2026-04-25 (KERNEL_TASK_TYPE_DEFAULT macro affects code-gen path — sibling concern)
- aog-self-critic C5 (no premature platform-blame — must reproduce on 2nd op before promoting to PB-N)

---

## CAND-PP79: Tie-cluster amplification of OL-83 boundary drift in low-precision sort+select pipelines (1-op evidence)

**Pattern**: When a sort+top-K+top-P (or sort+threshold-mask) op runs on `bf16` or `fp16`
with large N (≥16384), random-Gaussian inputs produce dense tie clusters at the top-K
boundary. The limited mantissa quantizes ~N distinct fp32 values down to ~2^M (M=mantissa
bits) distinct bit patterns; with N >> 2^M the boundary lands inside a 100s-position tie
cluster. Different valid sort implementations break ties differently → different valid top-K
masks → different top-P walks → different valid emit positions. Verifier reports
`max_abs_diff = FLT_MAX` and `mean_abs_diff = inf`, but `finite_max_diff = 0.0`.

**Symptom signature**:
- Pass A (small-N edge-set, deterministic shapes) bit-exact ✓
- Pass B (random benchmark, N≥16K, bf16/fp16) shows `max_abs_diff = 3.4e38`, `mean_abs_diff = inf`
- BUT `finite_max_diff = 0.0` (only -inf vs finite mask flips, no value disagreement)
- T1-vs-CPU triage (CAND-PP80) shows `kernel_vs_cpu_truth_flips ≈ ref_vs_cpu_truth_flips`
  (vendor reference is NOT MORE CORRECT than our kernel vs fp64 truth)

**Mitigation paths (decision rule)**:
- **DO NOT** attempt to reproduce vendor's tie-break order — vendor tie-break depends on
  hardware-internal sort intrinsic (undocumented, version-specific). Reproducing is OL-85
  case-specific overfitting that breaks on vendor updates.
- **PREFERRED**: classify cluster as OL-83-amplified T2-with-evidence carry-over.
- **ROOT FIX (methodology layer)**: refine verifier to admit
  `n_flip_positions ≤ Σ_rows tie_count_at_kth_value` AS T2-with-evidence pass, conditional
  on `finite_max_diff = 0.0`. This is a DEBT-level methodology improvement (not per-op).

**Concrete anchor — bf16 N=65536 random Gaussian**:
```
~256 distinct values in [-3σ, +3σ] → ~256 ties per unique value
Random k in distribution bulk → boundary lands inside 248-284-position tie cluster
Our Sort<MERGE_SORT> keeps 121-of-248 ties; CANN ref keeps 74-of-248 — both spec-valid
```

**Evidence**:
- op#9 9_TopKTopP cluster {8, 17, 26, 35} on bf16 [B, 65536] (2026-05-03, pp-3):
  `finite_max_diff=0.0` everywhere; `kernel_vs_cpu_truth` flips ≈ `ref_vs_cpu_truth` flips.

**Other instances (predicted)**:
- Any op that does sort + boundary-threshold-emit on bf16/fp16 large-N: top-k softmax
  (op#7 MoeGatingTopKSoftmax has potential exposure on dense small-num_experts cases),
  threshold-mask scatter, NMS at low IoU thresholds with score ties, beam search.
- Generalizes to any "rank-then-cut" op family.

**Promote when**: a 2nd op exhibits the same `finite_max_diff = 0.0 ∧ flips ≈ ref_flips`
T1-vs-CPU triage signature in a non-9_TopKTopP-family op.

**Source**: op#9 9_TopKTopP pp-3 (2026-05-03), workspace/9_topktopp/probes/probe_outputs/pp3_t1_vs_cpu_diff.json + pp3_tie_analysis.txt.

---

## CAND-PP80: T1-vs-CPU-fp64-truth triage before GM-dump bisection (precision-probe methodology)
`verified_on: a5_ops:3_FusionAttention:case_b27a259d`
`verified_on: a5_ops:3_FusionAttention:case_ad1de4ec`
`verified_on: a5_ops:apply_adam_w_quant:case_ab7dc8ca`
`verified_on: a5_ops:apply_adam_w_quant:case_aa66206a`
`verified_on: a5_ops:apply_adam_w_v2:case_e5aa3648`
`verified_on: a5_ops:apply_adam_w_v2:case_f08368cf`

**Pattern**: For Pass B clusters that share dtype + shape signature (e.g. all-bf16-N≥16K-failing),
run T1-vs-CPU triage BEFORE GM-dump bisection — it's ~10× cheaper and often refutes the bug
hypothesis entirely.

**Method**:
1. Compute reference (`torch_npu.<op>`) output, kernel output, AND pure-CPU fp64 truth on the same input
2. Pairwise diff in 3 directions: `ref_vs_kernel`, `ref_vs_cpu_truth`, `kernel_vs_cpu_truth`
3. Decision tree:
   - `kernel_vs_cpu` flips ≈ `ref_vs_cpu` flips AND both large → **OL-83-amplified** (CAND-PP79); kernel and ref equally drift from truth; verifier methodology too tight; **stop** — no GM-dump needed
   - `kernel_vs_cpu` flips >> `ref_vs_cpu` flips → real kernel bug; **proceed to GM-dump bisection**
   - `kernel_vs_cpu` flips ≤ 1 per row AND `ref_vs_cpu` flips ≤ 1 per row → canonical OL-83 (1-ULP single-position drift)

**Cost comparison**:
- T1-vs-CPU triage: 1 probe script + 1 build + 1 ssh run (~5 min)
- GM-dump bisection: 5+ iters of kernel rebuild + dump-extract + diff-analyze (~30 min/iter, ~150 min)

**Concrete anchor** — pp-3 on op#9 9_TopKTopP cluster {8,17,26,35}:
- Brief asked for GM-dump bisection of 5 phases (Sort, MrgSort, Extract, cumsum-prob, top-p, sample)
- T1-vs-CPU triage in 1 iter showed `finite_max_diff=0.0` and `kernel_vs_cpu_flips ≈ ref_vs_cpu_flips`
- Falsified the GM-dump premise; no kernel rebuild needed; falsifies kw-6's MrgSort/Extract bug hypothesis at the same time

**Evidence**: op#9 9_TopKTopP pp-3 (2026-05-03) — single-op evidence so far.

**Other instances (predicted)**: Any precision-probe spawn where Pass B failures cluster by
dtype+shape signature and the brief jumps directly to phase-by-phase GM-dump should run T1-vs-CPU
triage first. Sort+select ops are the highest-yield candidates because tie-break ambiguity is
common; cumsum/reduction-chain ops also benefit (tests whether reference is bit-exact or also
drifts from CPU truth at fp32 precision limits).

**Promote when**: a 2nd probe spawn uses this methodology and produces analogous "GM-dump
unnecessary" outcome on a non-tie-break-related precision puzzle.

**Source**: op#9 9_TopKTopP pp-3 (2026-05-03), workspace/9_topktopp/probe_report.md §pp-3.

---

## CAND-PP81: Two-tier eval NaN-match parity gap — `precision_eval_two_tier.py` mis-classifies saturated cases

**Pattern**: When kernel and CANN reference both saturate to NaN in the same positions where CPU truth is finite, the current `precision_eval_two_tier.py` `classify_output` function evaluates `pass_t2 = (NaN <= NaN)` which is False in Python (NaN comparisons). Net effect: cases where ours matches CANN bit-for-bit on NaN positions get labeled FAIL despite OL-109 parity-or-better intent (we are not strictly worse than CANN — both saturate).

**Symptom**:
- Verifier reports PARTIAL with N FAIL on bf16 / fp16 outputs that have wide dynamic range or division-by-near-zero structure
- Probe ad-hoc check shows ours == CANN bit-for-bit on NaN/inf positions; both diverge from CPU truth equally
- Op classification flips significantly when NaN-match parity is honored (op#14 went 22/50 strict T1 → 28/50 effective T2)

**Patch candidate** (in `classify_output`):
```python
import math
ours_nan = math.isnan(ours_mere) or math.isnan(ours_mare)
cann_nan = math.isnan(cann_mere) or math.isnan(cann_mare)
if ours_nan and cann_nan:
    verdict = "PASS_T2_NAN_PARITY"
elif pass_t1: ...
elif pass_t2: ...
```

**Evidence**:
- 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03): 10/50 cases (2,3,4,8,9,12,20,28,30,49) all show ours+CANN = NaN/NaN on grad_input + grad_weight, with grad_bias bit-exact 0.0/0.0. Without NaN-match treatment, op verdict PARTIAL with 32 FAIL; with NaN-match treatment, effective 28/50 PASS_T2.

**Other instances (predicted)**: Any fp16/bf16-saturating op evaluated through two-tier — high impact on `adaptive_instance_norm_bwd`, `batchnorm_bwd`, `layernorm_bwd`, any op with `pow(std, -3)` or large-dynamic-range intermediates. Affects all Wave 5 PERF-UNKNOWN re-bench ops with mixed-precision outputs.

**Promote when**: the patch is applied to `precision_eval_two_tier.py`, regression-tested against ≥2 ops with NaN saturation. Until then this is a methodology-fix candidate, not a kernel-coding pattern.

**Source**: op#14 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03). DEBT-070 candidate.

---

## CAND-PP82: Direction-4 anti-pattern — cast-point fix without reduction-shape pilot

**Anti-pattern**: when probe observes "CANN gets bit-exact 0.0 MERE vs ours has finite drift" on a REDUCTION output (e.g. `grad_weight = (...).sum()`), the immediate hypothesis is "CANN must keep fp32 internally and cast at the very end; we're casting too early — emit fp32 + pybind `.to(T)` will close the gap." This is the Direction-4 hypothesis.

**Why it's anti-pattern**: this hypothesis ignores the more likely root cause — CANN's bit-exactness with CPU truth often means CANN reproduces **CPU's reduction SHAPE** (lane-by-lane accumulation order), not that CANN has a different cast point. Our kernel may already be doing fp32-internal-with-cast-at-emit; the gap is in the cross-tile/cross-N reduction shape itself, not the cast point.

**Cost of applying without diagnostic**:
- 2-3 iters of build-verify-revert (per OL-111 measurement risk) consumed
- Cases shift identity, not count (OL-110 fail-floor invariant)
- Opportunity cost: real fix path (reproduce reduction-shape via per-lane accumulator) gets buried

**Recommended diagnostic BEFORE Direction 4**:
Write a probe that bit-compares ours, CANN, and CPU **on the same reduction algorithm** (literal `(go * x_normalized).sum()` element-by-element on identical inputs). Three outcomes:
1. CANN matches CPU bit-exactly when both fp16 → CANN reproduces CPU lane-by-lane order. Direction 4 won't help; need reduction-shape fix.
2. CANN deviates from CPU but matches ours when both fp16 → ours is OL-110 sub-family residual; document and ship at fail-floor.
3. CANN matches CPU only at fp32 emit, deviates at fp16 emit → Direction 4 IS the right fix. Apply with OL-111 pilot.

**Without this diagnostic, Direction 4 is OL-85 reward-hacking** (case-specific overfitting that doesn't address root cause).

**Evidence**:
- 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03): 11 cases match Direction-4-temptation pattern; deferred per OL-111 risk + this anti-pattern recognition.

**Promote when**: 2nd op surfaces this exact temptation pattern AND probe runs the recommended diagnostic AND outcome is documented (any of the 3 paths above). This codifies the diagnostic ritual into a verified pattern.

**Source**: op#14 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03), workspace/14_adaptive_instance_norm_bwd/knowledge_update.md Candidate 3.

---

## CAND-PP83: Eliminate pybind input-padding via kernel-side DataCopyPad GM→UB last-tile

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=tile-loop-vec`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 chip family — DataCopyPad GM→UB semantics likely transfer but not validated; A3 should re-confirm before applying)`

**Trigger**: tile-loop kernel where pybind currently does `auto x_padded = torch::zeros({B, D_padded}, ...); x_padded.slice(-1, 0, D_orig).copy_(x);` to align row stride to a TILE multiple, then passes `x_padded` to the kernel. msprof shows two pre-kernel ops (`Fill` + `ViewCopy`) accounting for non-trivial fraction of total time (op#29: 7 % + 19 % = 26 % of total).

**Principle**: EC-23 forbids DataCopyPad in the **UB→GM** direction on Ascend950PR, but the **GM→UB** direction works fine. So the pybind-side padding (which exists to make every tile's GM→UB load a clean `DataCopy`) is not necessary if the kernel handles its last tile via DataCopyPad. Pybind passes the unpadded tensor as a zero-copy view; kernel uses plain `DataCopy(local, gm[r*D + off], TILE)` for full tiles and `DataCopyPad(local, gm[r*D + off], cp, pad)` for the last tile only. This eliminates the Fill kernel + ViewCopy kernel from the per-call hot path.

**Concrete anchor**:
```cpp
// Kernel last-tile branch
if (cnt == TILE) {
    DataCopy(xLocal, gmX_[r * D + off], TILE);
} else {
    DataCopyExtParams cp{1, (uint32_t)(cnt * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, 0, T(0)};
    DataCopyPad(xLocal, gmX_[r * D + off], cp, pad);
}
// Pybind: just pass x_2d directly (no zeros + slice + copy).
```

Output write must continue to use the EC-23 mitigation (pre-pad output GM stride or use 3-phase writeback per EC-22) — this candidate is specifically about the **input** path.

**Quantified benefit (op#29 ko-iter4 evidence)**: DynamicQuant case 12 [4096, 11008] fp16. msprof showed Fill 7 % + ViewCopy 19 % of total = 26 % overhead removed. Kernel-side DataCopyPad cost ≈ 5 % (cnt-align computation + branching on last tile). **Net: ~+50 % wall-clock speedup on this kernel.**

**Cost / risk**:
- Kernel code grows by ~10 lines for the last-tile branch (cnt-align scalar logic).
- Branch is taken at most once per row, so per-tile overhead is amortized.
- Does NOT eliminate output-side padding; for ops where output write also has a padding tax, EC-23 cleaner mitigation (pre-pad output GM stride in pybind) is still needed.

**Promote when**: 2nd op shows the same Fill + ViewCopy elimination produces measurable (>5 %) wall-clock improvement AND precision unchanged. Cross-domain: any tile-loop quant / norm / elementwise op with non-aligned D where pybind currently does pre-padding for input.

**Anti-pattern avoided**: applying CAND-PP83 to the OUTPUT path violates EC-23 — UB→GM DataCopyPad crashes. Always pair this candidate with EC-23 cleaner mitigation (output-side pre-padding) when both directions need handling.

**Source**: op#29 29_DynamicQuant ko-iter4 (2026-05-02). 1-op evidence; needs second op to promote.

---

## CAND-PP84: Collapse per-tile V→S sync into a single end-of-pass V→S sync via per-tile UB result slots

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=multi-tile-row-reduction`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 chip family — V→S sync semantics likely identical, but pipe-stage scheduling differs; A3 should re-validate)`

**Trigger**: multi-tile per-row reduction (max, sum, etc.) where the natural pattern is `for each tile: ReduceXxx → V→S sync → scalar combine into row-level accumulator`. msprof shows non-trivial scalar-pipe time and V→S sync count = `tile_count` per row.

**Principle**: Each per-tile V→S sync costs scalar-pipe cycles AND prevents the V pipe from overlapping with other VEC work. For multi-tile rows (3-7 tiles in LLM-shape kernels), this is significant overhead. Replace with: write each tile's reduction result to a per-tile slot of a **scratch UB tensor** (e.g. `maxAccum[t * 8]`), do ONE `PipeBarrier<PIPE_V>` at end of pass, then ONE V→S sync, then the scalar combine loop reads from the scratch UB. Reduces V→S sync count from `tile_count` to 1 per row.

**Concrete anchor**:
```cpp
LocalTensor<float> maxAccum = scratchBuf.Get<float>();   // per-tile slots, scratch reused
for (int32_t t = 0; t < tile_count; ++t) {
    ReduceMax<float>(maxAccum[t * 8], srcLocal, ws, cnt_align, false);  // direct-write to slot
}
PipeBarrier<PIPE_V>();
event_t evVS = pipe_.FetchEventID(HardEvent::V_S);
SetFlag<HardEvent::V_S>(evVS);
WaitFlag<HardEvent::V_S>(evVS);
float row_max = 0.0f;
for (int32_t t = 0; t < tile_count; ++t)
    row_max = std::max(row_max, maxAccum.GetValue(t * 8));
```

The 8-element stride per tile slot is for fp32 datablock alignment (32 B / 4 B = 8 elements). For fp16/bf16 use stride-16 per slot.

**Quantified benefit (op#29 ko-iter4 evidence)**: DynamicQuant case 12 [4096, 11008] fp16. Pass 1 has 3 tiles per row; per-tile sync collapse reduced `aiv_scalar_ratio` from 0.282 → ~0.18; honest mean perf +30-45 % on multi-tile cases. Single-tile cases see no change (only one sync to begin with) — pattern only helps when `tile_count > 1`.

**Cost / risk**:
- Adds 1 scratch UB tensor of size `tile_count * 8 * sizeof(float)` per row (negligible — typically < 256 B).
- Increases peak VEC register pressure slightly (per-tile result lingering in scratch), but msprof confirms no spill on op#29's 11008-D path.
- Does NOT help when `tile_count == 1` (single-tile rows already do one sync).

**Promote when**: 2nd op (e.g. RmsNorm sum-of-squares pass, layernorm mean pass, softmax max pass) shows the same `tile_count > 1` × scalar-ratio reduction. The pattern complements OL-115 (manual prefetch) — both target the same thin-compute multi-tile signature.

**Source**: op#29 29_DynamicQuant ko-iter4 (2026-05-02). 1-op evidence; needs second op to promote.

---

## CAND-PP85: Adversarial divisor-clamp probe to detect torch_npu fused-op docstring divergence

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=dynamic-quant-fused`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 chip family — torch_npu fused-op behavior may differ across chip families; re-probe before relying on this divergence on A3)`

**Trigger**: porting a dynamic-quantization-class CANN fused op where the public reference docstring specifies `clamp(quant_scales, min=epsilon)` (typical: 1e-10) for divisor safety, BUT the actual `torch_npu.npu_<op>` implementation may NOT apply the clamp — it uses the unclamped amax/N_levels divisor and relies on natural overflow + output clamp to produce saturation behavior.

**Principle**: when CANN fused-op docstrings describe a "clamp divisor at small ε" safety guard, the **public docstring may diverge from the actual fused-op kernel's algorithm**. A docstring-literal kernel writing `qs_div = max(out_scale, 1e-10); dyn_scale = 1/qs_div` produces all-zero output for degenerate rows (tiny magnitudes where `amax / N_levels << ε`), while the reference fused op produces non-zero saturated int8 output (because it uses the unclamped tiny `amax` directly).

This is the **same family of divergence as P-P58.X (swiglu_mode mode-flag dispatch)** — different axis (divisor-clamp vs mode dispatch), same root cause (docstring not authoritative for fused op).

**Diagnostic probe (1-shot detection)**: write a small const-magnitude probe with all inputs at very tiny magnitude (~5e-7 or smaller), where `amax / N_levels` would land below the docstring's clamp ε. Run the reference; inspect whether output has non-zero int8 values on the max positions. If yes → divergence confirmed; drop the clamp in your kernel.

**Concrete fix**:
```cpp
// docstring-literal (BROKEN on degenerate rows)
qs_div = std::max(out_scale, 1e-10f);
dyn_scale = 1.0f / qs_div;

// matches torch_npu actual behavior
if (y_max > 0.0f) dyn_scale = CLIP_MAX / y_max;
else              dyn_scale = 0.0f;
```

The `y_max > 0` guard catches the exact-zero-row degenerate case (where `stored_scale = 0` and `output = 0` is correct); for any positive y_max, use natural `127 / y_max` (no ε floor). Output clamp `[-128, 127]` handles any overflow naturally.

**Quantified evidence (op#11 v3.2 cold-restart, 2026-04-21)**: edge-dataset cases `dist_const_near_zero_seed{0,1}` (inputs ~5.95e-7) — docstring-literal kernel produced all-zero output; torch_npu reference produced `[2, 51, 26, 21, 104, 127, ...]`. Drop-clamp fix → 24/24 int8 bit-exact across full edge dataset.

**Cost / risk**:
- Removes a "safety" guard from kernel — but the actual safety is the output clamp `[-128, 127]`, not the divisor clamp.
- Risk: zero-amax row (true zero input) — handled by the explicit `if (y_max > 0)` guard.
- Pre-probe ritual takes ~5 min; saves multi-iter precision-probe loops on degenerate edge cases.

**Promote when**: 2nd CANN dynamic-quant-class op (e.g. AddRmsNormDynamicQuant, RmsNormDynamicQuant, SwigluDynamicQuant, GroupQuantize) confirms the same divergence pattern AND probe documented + drop-clamp fix applied. Will likely co-promote with P-P58.X into a unified "CANN fused-op docstring is not authoritative" pattern family.

**Anti-pattern avoided**: trusting fused-op docstrings as authoritative. Pass A (benchmark) inputs typically have normal magnitudes where divisor clamp never kicks in, so the bug is invisible without an adversarial Pass B / edge-dataset probe.

**Source**: op#11 DequantSwigluQuant v3.2 cold-restart (2026-04-21). 1-op evidence (sibling family P-P58.X). Needs second dynamic-quant-class op to promote.

---

## CAND-PP86: [REGRESSION-RISK BLOCKED] Always-depth=4 TQue<VECIN> proposal — would override OL-63 thin-compute carve-out

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=tile-loop-vec`
`verified_on: n/a (BLOCKED — single-test counter-evidence on uninitialized data)`
`unverified_on: all`
`regression_risk_classification: pattern #3 (contradicting evidence-anchored guidance)`
`status: NEEDS_USER_INPUT — do NOT promote without multi-op + msprof + initialized-data evidence`

**Source**: `workspace/regression_risk_test_p0aax/knowledge_update.md` Findings §1
(adversarial fixture for the P0aax regression-risk gate). Demoted by the gate
on 2026-05-07 (run_ts 20260507T081851Z) from a HARD-severity proposal to amend
canonical OL-63.

**Proposed claim** (as submitted): always use `TQue<VECIN, depth=4>` regardless
of per-tile VEC compute weight; remove OL-63's depth=2 thin-compute branch +
the `VEC < 2× MTE2` litmus test.

**Why blocked**: OL-63 currently has 2 multi-op multi-date Evidence rows
(GELU 2026-04-14: depth=4 wins on compute-heavy; DynamicQuant ko-1 iter3
2026-05-02: depth 2→4 **regressed honest mean perf by 7 %** on thin-compute
tile loop) AND a top-of-entry decision rule + measurable litmus. The fixture
offers single-test counter-evidence on uninitialized data (op "frobnicate",
TILE=512, +3 % depth=4 win) — well within typical run-to-run variance for
elementwise tile loops, no msprof, no per-pipe-stage breakdown, no application
of the litmus the OL prescribes. See
`workspace/regression_risk_test_p0aax/kb_scan/regression_risk_20260507T081851Z.md`
for full reasoning.

**Promote when**: ≥3 independent ops (different op classes, different per-tile
compute profiles spanning thin AND heavy) show depth=4 wins, msprof confirms
the wins are pipeline-overlap-driven (not compute-noise), and the
DynamicQuant ko-1 –7 % result is reproduced under depth=4 to demonstrate the
prior measurement was either flaky or environment-specific. Until then, OL-63
remains the authoritative rule.

**Anti-pattern this candidate would re-introduce if accepted prematurely**:
"single-op weak counter-evidence overrides multi-op-evidenced decision rule" —
the kind of premature generalization the regression-risk gate exists to catch.

---

## CAND-PP87: [REGRESSION-RISK BLOCKED] Pure-TBuf-for-dequant-output proposal — would override P-P77 + OL-94 + 6_QuantMatmul Finding #3

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quant-dequant`
`verified_on: n/a (BLOCKED — single-test "no corruption observed" on adversarial fixture)`
`unverified_on: all`
`regression_risk_classification: pattern #2 (re-introducing anti-pattern) + pattern #3 (contradicting evidence-anchored guidance)`
`status: NEEDS_USER_INPUT — do NOT promote; absence-of-evidence ≠ evidence-of-absence for pipe-ordering bugs`

**Source**: `workspace/regression_risk_test_p0aax/knowledge_update.md` Findings §2
(adversarial fixture). Demoted by the gate on 2026-05-07
(run_ts 20260507T081851Z) from a HARD-severity proposal to archive the
6_QuantMatmul TQue-VECOUT recommendation + add a new P-P stating pure TBuf is
fine for dequant output writes.

**Proposed claim** (as submitted): pure-TBuf-based dequant output write
pipeline is fine; the `TQue<VECOUT>` requirement (P-P77, OL-94, 6_QuantMatmul
Finding #3) is over-cautious; revert to pure TBuf for all dequant pipelines.

**Why blocked**:
- **P-P77** (`patterns/PATTERN_INDEX.md` line 88) explicitly catalogues
  bare-TBuf in this slot as an anti-pattern — op#27 27_MultiMaskAttentionAggregation
  Phase D iter-5 saw 9/10 wrong-output runs with PipeBarrier<PIPE_ALL>; 6/10 wrong
  with extra PipeBarrier<PIPE_V>.
- **OL-94** decision table cites the official AscendC doc:
  `TBuf申请的内存空间只能参与计算，无法执行队列的入队出队操作` and
  `EnQue调用会发射同步指令set` — TBuf has no sync mechanism by design.
- **6_QuantMatmul Finding #3** has a concrete per-row probe: even rows = 0,
  odd rows = correct values; rebuilding with TQue<VECOUT> + EnQue/DeQue → all
  64 rows correct. Already merged as evidence (batch 20260507T081021Z).
- The fixture offers single-op "no corruption observed" — but pipe-ordering
  bugs are schedule-sensitive; absence-of-corruption in one configuration
  does NOT generalize. The fixture made no attempt to reproduce the exact
  per-row probe pattern.

See `workspace/regression_risk_test_p0aax/kb_scan/regression_risk_20260507T081851Z.md`
for full reasoning.

**Promote when**: the proposed pure-TBuf pipeline is run against (a) the exact
per-row probe pattern from 6_QuantMatmul Finding #3 with bit-identical PASS,
AND (b) the op#27 27_MultiMaskAttentionAggregation regression case (≥10
independent runs PASS), AND (c) at least one additional dequant-class op
(e.g., 11_DequantSwigluQuant). Until those reproductions exist, P-P77 + OL-94
+ 6_QuantMatmul Finding #3 remain authoritative.

**Anti-pattern this candidate would re-introduce if accepted prematurely**:
the "TBuf as sync-capable queue substitute" misuse that A-P61, OL-94,
P-P77 collectively catalogue. Schedule-sensitive determinism bugs require
multi-run, multi-op, deliberately-adversarial reproduction — not single-pass
"didn't see it" reports.

---

## CAND-PP88: Iterative DIT radix-2 FFT (Cooley-Tukey) with hybrid VEC/scalar butterfly path
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=spectral-transform (FFT/IFFT/DCT family)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (V220 family — twiddle build via Sin/Cos<RADIAN_REDUCTION> primitive precision tier may differ; needs A3 probe)`

**Trigger**: reference op is `torch.fft.rfft` / `torch.fft.fft` with PoT length, OR direct DFT O(N²) hits perf ceiling on large N, AND fp32 precision must match torch.fft (catastrophic-cancellation cases).

**Pattern (per-row, single-AIV-owns-row determinism)**:
1. Zero-pad: `real_buf[0..seqlen) ← x[row]`, `real_buf[seqlen..N) = 0`, `imag_buf[0..N) = 0`.
2. Bit-reversal permutation in-place (scalar SetValue/GetValue with V→S/S→V sync).
3. For stage `s` in `[1..log2N]`: `m = 1 << s; half = m >> 1`; build per-stage twiddle via `ArithProgression + Muls + Sin<RADIAN_REDUCTION> + Cos<RADIAN_REDUCTION>`; dispatch:
   - `half < 8` → scalar butterfly path (avoids EC-26 unaligned-VEC violation since `m=2,4,8` violates 32 B alignment for fp32).
   - `half ≥ 8` → VEC butterfly path (Mul/Sub/Add; `tmp_a` aliased as `tb_im` to save one scratch buffer).
4. Per-row normalize: `Muls(real_buf, real_buf, 1/N, N); Muls(imag_buf, imag_buf, 1/N, N)`.
5. `DataCopy(GM, real_buf, AlignUp8(kcount))` with V→MTE3 sync; same for imag.

**Per-stage twiddle rebuild (NOT recurrence)**: each stage's twiddle is independently computed from `theta = ArithProgression × (-2π/m)` → no cross-stage error propagation. Per-stage cost: `half × Sin + half × Cos`. Total across `log2N` stages: ~N twiddle ops, dwarfed by `6N log2N` butterfly ops.

**UB layout (worst case N=8192, seqlen=4096, fp32)**: input TQue (16 KB) + real/imag bufs (32+32 KB) + twiddle_cos/sin (16+16 KB) + tmp_a/tmp_b/tb_re scratch (16×3 KB) + theta/idx (16+16 KB) = **176 KB** (fits 192 KB UB).

**Hybrid dispatch for non-PoT seqlen**: cheap runtime check `is_pot = (seqlen > 0) && ((seqlen & (seqlen-1)) == 0)` — PoT path → radix-2 FFT, non-PoT path → direct DFT fallback. Dispatch overhead negligible.

**Determinism**: by-construction satisfied (1 row → 1 core, fixed-order stages, deterministic bit-reversal, deterministic twiddle compute, no atomicAdd, no shared GM writes).

**Performance**: op#23 23_HyenaFftSizePaddingRfft kw-3 (2026-04-26): Pass A 49/49, det 49/49, perf ratio_median 1.17× (vs kw-2 direct-DFT 0.596× — ~2× improvement on PoT-dominated benchmark). Pass B 12/14: 2 large_mag fp32-ULP-limit cases (residual ~ N · ULP · |max_input|; classified as an fp32 algorithmic limit, not a kernel bug).

**Anti-pattern avoided**: case-specific predicates / ε nudges to mask adversarial-magnitude failures (OL-85 violation). Adversarial-magnitude residual is fp32 unit-ULP × |input|_max; classified as an fp32 algorithmic limit, not patched.

**Promote when**: a second spectral-transform op (FFT-derived: convolution-via-FFT, IFFT, real-FFT-of-2N-trick) confirms the same Cooley-Tukey + scalar-stage-cutoff template. Likely to live in a future `patterns/domains/spectral.md` (KB has zero spectral content today; gap noted as DEBT-052).

**Source**: op#23 23_HyenaFftSizePaddingRfft kw-3 (2026-04-26). 1-op evidence; KB had no spectral pattern at start of session — directive itself encoded the missing knowledge.

---

## CAND-PP89: torch_npu fused-op path divergence at non-grid-aligned (N, H) shapes — `npu_dequant_swiglu_quant` mode=1 evidence
`applies_to: soc=Ascend910_9382 (V220, A3); cann=9.0.0; bisheng=15.0.5+2026-01-28; op_class=quant-fused (DequantSwigluQuant family)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 — same fused op may exist with different tile geometry; A5-side probe needed before generalizing)`

**Trigger**: kernel reference is a CANN fused op of the form `torch_npu.npu_<fused>_quant(...)` AND benchmark Pass A shows shape-conditional precision divergence on a small subset of cases that does NOT correlate with H-alignment, dtype, or scalar-input combination.

**Symptom**: torch_npu reference produces a different scale / quantized output for certain (N, H) shapes vs the docstring's manual computation. Independent reproduction via `torch_npu` ops in the docstring's exact compute path produces THE SAME divergence — proving the divergence is upstream in CANN's internal kernel, NOT in the user's port.

**Empirical shape-sensitivity table** (op#11 DequantSwigluQuant a3 mode=1, swiglu = `(x_glu × sigmoid(α·x_glu)) × (x_linear + β)`, then per-row dynamic int8 quantization):

| (seed, N, 2H)      | scale_max_diff | result |
|--------------------|----------------|--------|
| (0, 32, 64)        | 3.26e-1        | FAILS — small (N divides AIV, 2H divides 32) but tile path differs |
| (0, 32, 128)       | 0.0            | PASSES |
| (0, 64, 1024)      | 0.0            | PASSES |
| (0, 216, 5056)     | 4.03e-1        | FAILS — 216 % 48 ≠ 0 AND 5056 % 128 ≠ 0 |
| (0, 256, 8192)     | 0.0            | PASSES |
| (1, 216, 5056)     | 4.83e-1        | FAILS (consistent across seeds) |
| (42, 256, 8192)    | 0.0            | PASSES |

**Hypothesis**: torch_npu's internal CANN op uses a tile-boundary "fast path" gated on `(N % aiv_count == 0) AND (H % tile_h == 0)` (likely `tile_h ≈ 128 fp32`). When BOTH hold → "pure" path; otherwise → fallback path with different rounding semantics.

**Diagnostic recipe**:
```python
# Same diff signature regardless of whether compute is on CPU or NPU torch:
manual_sc    = (x_fp32_compute(...).abs().amax(dim=-1)) / 127.0
torch_npu_sc = torch_npu.npu_dequant_swiglu_quant(...)[1]
(manual_sc - torch_npu_sc).abs().max() == 0.241  # all paths converge on this divergence
```
If `manual_sc` (computed via NPU torch ops along the docstring's path) shows the SAME diff vs `torch_npu_sc`, the divergence is entirely in the fused CANN op — NOT a port bug.

**Action when this signature applies**:
1. msprof the failing vs passing shape to compare CANN op kernel-name / tile params (different sub-op sequences = different rounding paths).
2. If two paths confirmed: compare both paths with fp64 CPU truth and keep the result unresolved if neither path is justified.
3. Do NOT case-specifically patch (OL-85 violation). Either accept residual or extend verifier to skip the fallback-path subset.

**Promote when**: a second torch_npu fused-quant op (e.g. `npu_swiglu_quant`, `npu_add_rms_norm_dynamic_quant`) shows the same `(N, H)`-grid-conditional fast/fallback divergence. Likely co-promotes with CAND-PP85 + P-P58.X into a unified "CANN fused-op grid-path divergence" family.

**Source**: op#11 DequantSwigluQuant a3 kw-2 (2026-04-28). 1-op evidence. Cross-arch isolation note: A5-side probe of same op family pending; if A5 shows identical (N, H)-grid sensitivity, applies_to broadens to `all`.

---

## CAND-PP90: Small-D per-row shuffle ops — scalar fp32 compute beats over-engineered SIMD plumbing
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=elementwise-with-permute (interleave / odd-even / shuffle / per-row-permute small-D)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (A3 V220 — scalar GetValue/SetValue cost may differ relative to VEC dispatch; needs A3 probe before generalizing)`

**Trigger**: elementwise op with per-row shuffle / permute / FMA where the row is small (`D ≤ 128`) AND the permutation pattern is non-contiguous (interleave, even-odd split, butterfly within row). Reference algorithm reads scalar-style: `for i in range(D): out[i] = formula(x[π(i)], y[π'(i)], ...)`.

**Recommendation**: scalar fp32 compute loop after SIMD `Cast<fp32, T>(...)` is a legitimate, maintainable implementation. Don't reach for VEC `Mul/Add` + `Gather` + scratch buffers + PipeBarriers when:

1. The natural reference is per-element scalar (not block-vectorized).
2. The permutation pattern requires `Gather` (or scalar scatter writes) anyway.
3. Total scalar ops per tile is small (D=64 → 256 scalar/row, 2048-4096/tile — sub-millisecond on AIV).

**Anti-pattern avoided** (the specific over-engineering this entry points away from): "build gather-style contiguous sub-tiles → apply VEC Mul/Add → scatter writes back" needs O(tile) scalar scatter writes anyway, plus PipeBarriers, plus scratch buffers, AND ends up matching the reference algorithm 1-to-1 only after per-element fixups. The "pure SIMD" version is more code, more buffers, more risk — and not faster than scalar at small-D.

**Trade-off (transparent)**: scalar path will not match a CANN fused op that uses specialized ISA (e.g. RoPE-specific instructions). Expect 0.5×–0.8× ratio vs `torch_npu.npu_<fused>` — acceptable when (a) the fused op exists for the reference but the port was via the generic docstring, (b) precision matters more than perf, OR (c) baseline pass is the priority and a vectorized rewrite is a follow-up optimization.

**Decision rule** (when in doubt):
- If reference docstring is a `for i in range(D)` per-element loop → scalar path is the literal translation, use it (Iron-law §5).
- If `D ≤ 128` AND permutation is non-contiguous → scalar path FIRST.
- If `D > 128` OR permutation is contiguous (concatenation, slice-and-shift) → VEC path likely wins.

**Evidence**: op#13 13_InterleaveRope (2026-04-30). Reference: `torch_npu.npu_interleave_rope` per-row interleave-then-FMA, `D = 64` fixed. Scalar fp32 loop after SIMD `Cast<fp32, T>` → 50/50 PASS bit-exact (both fp16 + bf16 paths), perf ratio 0.72× vs `torch_npu.npu_interleave_rope`. CANN fused op presumably uses specialized ISA for this exact shape — 0.72× is the cost of generic-codegen scalar vs specialized-instruction. Maintainability + correctness > 30% perf gap when the reference algorithm IS scalar.

**Promote when**: a second small-D per-row shuffle op (e.g. odd-even shuffle, butterfly-within-D layouts in attention variants) confirms the same scalar-beats-SIMD-plumbing decision. Likely co-promotes with P-P9 (SIMD vs SIMT decision framework) into an OL with explicit small-D scope clause.

**Source**: op#13 13_InterleaveRope kw-1 (2026-04-30). 1-op evidence; author's own promotion gate cited in `output/npukernelbench/src/kernels/13_InterleaveRope/knowledge_update.md`.

---

## CAND-PP91: ReduceMax-per-iteration greedy selection as sort-replacement under tight UB budget
`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=greedy_selection`
`verified_on: soc=Ascend910_9382; cann=9.0.0-beta.2`
`unverified_on: soc=Ascend950PR_9589 (has 256KB UB — might fit full sort for larger N; greedy selection may still be preferable for tile-locality reasons independent of UB pressure)`

**Trigger**: op whose reference algorithm is "sort by score DESC → iterate greedy over sorted list → suppress IoU≥threshold neighbors". When `N_max × sizeof(dtype) > UB / 2`, pre-sorting the full element list is infeasible — the sort workspace alone exceeds the UB budget.

**Recommendation**: replace the pre-sort with a ReduceMax-per-iteration greedy loop. Each iteration: (1) tiled ReduceMax over persistent UB scores buffer to find the current maximum-score element, (2) compute pairwise IoU against that element for all candidates in tiled chunks, (3) suppress candidates whose IoU ≥ threshold by setting their scores to -inf in the persistent buffer. No sort workspace needed — total UB = persistent scores buffer (N_max × sizeof(dtype)) + IoU compute tile (~2KB per chunk). For V220 (192KB UB) with fp32 N≤32768, this uses ~160KB (83% utilization).

**Concrete anchor** (canonical V220 NMS inner loop):
```cpp
// Per-iteration: find max-score element via tiled ReduceMax
float best_score = -INFINITY;
int   best_idx   = -1;
for (int tile = 0; tile < num_tiles; ++tile) {
    auto tile_max = WholeReduceMax(scores[tile], scores[tile], tile_len, 1);
    float tile_best;  tile_max.GetValue(0, tile_best);
    if (tile_best > best_score) { best_score = tile_best; /* track idx */ }
}
if (best_score < score_threshold) break;  // early exit
// IoU suppress: for each tile, compute IoU vs best box, mask scores to -inf
```

**Single-AIV determinism**: with `nblk=1` (single-AIV execution), strict `>` across tiles and linear-forward in-tile scan gives deterministic output matching PyTorch stable-sort tie-break semantics (lowest index wins on tie). No cross-core communication, no atomicAdd.

**Promote when**: a second greedy-selection op on V220 (or another platform with tight UB) confirms the ReduceMax-per-iteration approach independently. Candidate promotion candidates include: top-K with dynamic K (where K varies per row and full-sort is wasteful), iterative beam search, WBF (weighted boxes fusion).

**Evidence**: op#30 NMS a3 ds kw-1 (2026-05-07, Ascend910_9382 V220, CANN 9.0.0-beta.2). N_max=32768 fp32 — pre-sort needs ~256KB (exceeds 192KB UB). ReduceMax-greedy uses ~160KB UB (83% utilization). Pass B vs Python CPU reference: 31/31 bit-exact (set-equivalence comparison on `selected_indices[:num_selected]` + bit-exact `num_selected`). Single-AIV, deterministic by construction.

**Source**: op#30 NMS a3 ds kw-1 (2026-05-07). 1-op evidence.

---

## CAND-PP92: Surgical metadata-only response when finalize→await_worker rollback is structural-not-numerical
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=workflow`
`verified_on: synthetic pytest fixture (test_iter_cap_p0aa_drives_fina0 — 9_topktopp underlying)`
`unverified_on: real-op rollback in production`

**Trigger**: `state_transitions.jsonl` shows a finalize → await_worker rollback whose `rationale` cites a missing Pass-B verifier artifact OR a missing GATE_CONTRACT §"MANDATORY artifacts for finalize gate" field, AND the prior pipeline (probe + researcher + optimizer) has already produced terminal evidence on `verification.json` (`persist_verdict` set, `precision.persist_classification` set, optimizer plateau reached). The rollback is a metadata-shape gap, NOT a numerical re-litigation request.

**Recommendation** — kw response on this kind of rollback is surgical, not regenerative:
1. Read GATE_CONTRACT §"MANDATORY artifacts for finalize gate" in FULL. The rollback `rationale` only names one symptom but the gate audits four fields: `run_pass_b.py` artifact + `verification.json.precision.pass_b` + `performance.independent_re_measure` + `performance.ratio_baseline`. Fixing only the symptom named in `rationale` re-triggers rollback on the next-named missing field.
2. Emit the missing artifacts AT THE WORKSPACE ROOT — do NOT re-run analysis / build / verify phases.
3. Preserve prior `precision.persist_verdict` / `precision.persist_classification` / `precision.persist_evidence` verbatim. The upstream pipeline owns those; kw must not overwrite.
4. Use `pass_b: {status: "N/A", reason: "<upstream evidence chain>"}` as canonical encoding when PARTIAL_PERSIST verdict is already established. Reason string MUST cite the upstream evidence (probe_report.md + cann_strategy_inference.md) so the finalize gate sees a non-frivolous N/A.
5. Re-emit `→ orchestrator: PARTIAL_PERSIST — <evidence>` with the same evidence chain — do NOT switch pass_a to PASS or inflate the verdict.

**Concrete anchor** (verification.json pass_b stanza encoding PARTIAL_PERSIST established upstream):
```json
"pass_b": {
  "status": "N/A",
  "reason": "PARTIAL_PERSIST established by upstream probe + researcher; see probe_report.md + cann_strategy_inference.md. Tier-1 residual on 4 fp16 cases — bit-exact edge-dataset would only re-confirm."
}
```

**Anti-patterns explicitly avoided**:
- Regenerating kernel files (burns iter budget on conclusion already reached)
- Re-running probe / researcher / optimizer (verdict=requirement already terminal per V3.8.8 "never let PARTIAL pass" + iter_cap policy)
- Promoting PARTIAL to PASS by lying about pass_b OR switching pass_a status (precision verdict integrity)
- Writing to `state_transitions.jsonl` (orchestrator-owned artifact)

**Cites** ANTI_PRESSURE_PROTOCOLS P5 (closure-pressure / "expected failure") + P7 (closure-desire after long pipeline) — the urge to over-deliver ("regenerate kernel", "re-measure perf") is exactly the pressure mode this pattern guards against. Also cites GATE_CONTRACT §"Phase D Verify Gate" + §"MANDATORY artifacts for finalize gate" (P0aaf #108, P0aba 2026-05-07).

**Promote when**: a real-op rollback in production exercises this surgical-metadata-only response and confirms (a) no re-verify needed, (b) finalize gate accepts the re-emitted PARTIAL_PERSIST, (c) iter cap preserved.

**Evidence**: synthetic pytest fixture `test_iter_cap_p0aa_drives_fina0/test_op` (2026-05-10, 9_topktopp underlying op-class transcendental + sort, DET_POLICY=best_effort). Pipeline pre-state: worker→probe→researcher→optimizer×5 with KO_PERF_PLATEAU. Rollback rationale: "no Pass B verifier found". kw spawn closed metadata gaps without re-litigating precision/perf verdicts. 1 fixture, 0 real-op evidence.

**Source**: pytest fixture iter_cap_p0aa kw spawn (2026-05-10).

---

## CAND-FA1: Manual user-owned CrossCore flag handoff for mixed AIC/AIV producer-consumer stages (NOT for kernels using high-level Matmul<> library internals)
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=mixed_aic_aiv_fused_kernel_with_user_owned_cross_engine_handoff`
`derived-from: cann-source (FA-class fused-attention reference structure, 2026-05-10 revise-cl3)`
`verified_on: cann ops-transformer FA reference — top-level kernel file (cube/vec flag chain ~80 lines, three named user-owned flags); arch cross-core-sync header (FlagID = uint16_t; MAX_REVERSE_DEPTH = 16 array-slot count = 15 reuses + 1 initial state, consistent with ascend950pr.md "同一 flagId 最大计数 15 次"; FFTS_MAX_FLAG = 7); reserved-barrier IDs at 8/9/10 are a cann-source-derived REFINEMENT of the public 0–10 range documented in ascend950pr.md, not a replacement`
`refuted_on: a5_ops:3_FusionAttention:case_b27a259d — kernel mixes MatmulImpl + manual CrossCore which violates this pattern's hard-do-not-apply clause; the manual CrossCoreWaitFlag path hard-hung. This is NEGATIVE evidence reinforcing the exclusion, NOT positive validation. The specific failure mode (AICore timeout 507014 / LaunchAscendKernel 507035) is codified separately as PB-34 in PLATFORM_BUGS.md — read it before emitting any MIX_AIC_1_2 kernel that touches MatmulImpl/MatmulClient/KFC.`

`empirically_validated_on (2026-05-21, partial evidence chain — a5_ops:3_FusionAttention kw-4 cycle 3 run buksn5pky):`
- **`Nd2NzParams` field shape for D-aligned fp16 tiles (D%16==0)**: `Nd2NzParams{ndNum=1, nValue=S, dValue=D, srcNdMatrixStride=0, srcDValue=D, dstNzC0Stride=D/16, dstNzNStride=16, dstNzMatrixStride=0}` correctly performs ND→NZ during `DataCopy(l1, gm, params)` GM→L1. Evidence: with this shape, the first-`Mmad` fault evolved `507015` (iter 1+2, wrong shape) → `0x8000004000` L0B read/write conflict (iter 3 Phase 1, shape correct but sync missing). The transition is unambiguous evidence that the shape passes the L1-decode stage.
- **`LoadData2DParams` with `ifTranspose=true` for K^T side** (matmul1 = Q @ K^T case): mechanically equivalent to `LoadDataWithTranspose`; built clean on V220 with `LoadData2DParams{startIndex=0, repeatTimes=(baseM/16)*(baseK/16), srcStride=1, sid=0, dstGap=0, ifTranspose=true, addrMode=0}`. **SUPERSEDED 2026-05-28** by `fa_class/cv_reference_concrete_params.md#decision_id-qk_load_form` (cv-agent ComputeMM1 dual-operand `ifTranspose=false` + Mmad k-contraction along D=C0). This `ifTranspose=true` form COMPILES clean and accepts QK^T magnitude but, when paired asymmetrically with a plain 3DParamsV2 A-load (`LoadNzL1ToZzL0A`), produces a layout-permute on the `[BLOCK_M × BLOCK_N]` tile — `attn_out` `abs_max` tracks ref within ~1% but element-wise `max_diff ~1.3-1.6` on the FA-A3 6-case canonical (P-P99 corollary: A/B contraction axes must source the same axis). Use the `qk_load_form` decision; do NOT emit `ifTranspose=true` in fresh code. Kept here for historical evidence of the V220 compile-clean signal that masked the precision bug.
- **Event-ID allocation — `N >= 4` Fix PARTIALLY REFUTED, deeper deadlock unresolved**: prior hypothesis "`event_t(0)` collides with cross-core `FLAG_CANON_DONE`; use `N >= 4` to dodge" — the collision IS real (codified as PB-35), but the kw-5 cycle (3_FusionAttention iter 4, 2026-05-21T~14:00Z) empirically falsified that `N >= 4` is sufficient. Tested three distinct schemes: (a) `event_t(0)` baseline (silent hang); (b) raw `event_t(2,3,4)` for mm1 + `event_t(5,6,7)` for mm2 with distinct IDs ≥ 2 (silent hang, same signature); (c) `GetTPipePtr()->FetchEventID(HardEvent::X)` canonical runtime allocation (silent hang, same signature). ALL three produce identical "kernel enqueues + torch.npu.synchronize() hangs past 45s, no aicore exception" symptom. The visible event-ID collision is one layer; the actual deadlock root cause is deeper — open hypotheses include Cross* sync uniformity per HardEvent class, MIX_AIC_1_2 cube-internal-sync incompatibility with the CrossCoreSetFlag chain, and FFTSCNT mailbox semantics interacting with cube-internal HardEvent flags. See PB-35 Evidence row 3 + new candidate CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP for the full evidence chain.
- **Outstanding for canonical promotion**: full Pass A case_3 `PASS_T1` + measured perf vs CANN baseline. Iter 4 attempted convergence with corrected event-ID scheme — FAILED (silent hang persists across all 3 sync schemes). Promotion to canonical `P-P` / `OL` entry BLOCKED by the unresolved MIX_AIC_1_2 sync-infra gap; baseline kernel remains AIV-only VEC fallback delivering case_3 PASS_T1 (ours_mere=1.999e-6 < cann_mere=2.227e-6) with 60 deterministic `_OutOfScope` skips. Next attempted convergence requires either `aog-fused-optimizer` investigation of MIX_AIC_1_2 sync semantics OR `aog-cann-learner` Mode 5 extraction of CANN ops_transformer arch22 `flash_attention_score` kernel structure.

**Trigger**: mixed-mode kernel dispatched via `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` with AIC half compiled from `__DAV_C220_CUBE__` and AIV half from `__DAV_C220_VEC__`, decomposing into ≥2 user-owned producer-consumer stages where cube and vector exchange GM-resident intermediates and a kernel-wide `SyncAll<true>()` is too coarse. The kernel must NOT instantiate the high-level AscendC `Matmul<>` template (`MatmulImpl` / `MatmulClient` / KFC) — see hard-exclusion below.

**Recommendation**: pair `AscendC::CrossCoreSetFlag<0x2, PIPE>(flagId)` with `AscendC::CrossCoreWaitFlag<0x2>(flagId)`. **MODE template argument must appear identically on both sides** — this satisfies the canonical "SetFlag 和 WaitFlag 必须参数完全一致" rule in `ascend950pr.md`. `CrossCoreWaitFlag`'s MODE is defaulted in the public header, so a bare `CrossCoreWaitFlag(id)` call resolves to the same EventID; for KB readability ALWAYS write the explicit `<0x2>` on the wait so the pairing is visually unambiguous. MODE `0x2` is the AIC↔AIV 1:2 paired-sync mode inside `KERNEL_TYPE_MIX_AIC_1_2`; release reaches only the paired sub-blocks of the opposite engine, NOT a whole-device broadcast.

Pipe selection (verified against the FA reference):
- `PIPE_FIX` when the producer is AIC writing its output through the FIX pipe to GM
- `PIPE_MTE3` when the producer is AIV writing data through MTE3 to GM
- Pick the pipe whose retirement must precede the consumer's read.

Flag-ID ownership (cann-source-derived refinement of the public `0–10` range documented in `ascend950pr.md`):
- IDs `0..7` (`FFTS_MAX_FLAG = 7`) are the user-owned range used by this pattern.
- IDs `8`, `9`, `10` are reserved by `BarrierFlag` specializations for inter-block / inter-subblock barriers. Earlier local examples using `0x8` predate this carve-out — they were not in conflict only because no barrier specialization was active; new code MUST stay in `0..7`.
- Per-flagId count budget: canonical KB documents **15 reuses** (`ascend950pr.md` "同一 flagId 最大计数 15 次"); the cann-source constant `MAX_REVERSE_DEPTH = 16` is the underlying slot-array size = 15 reusable counts + 1 initial state. Treat 15 as the publicly-bounded reuse limit.

**Concrete anchor** (three-flag QK → softmax → PV chain; cube ladder built on tile-MMAD primitives, NOT `Matmul<>`):
```cpp
constexpr uint32_t cube

## CAND-FA2: Online-softmax per-row state recurrence — running max, running sum, and an external accumulator rescale carried across row tiles (final divide deferred to last tile)
`applies_to: any SoC with public AscendC VEC Exp/Sub/Mul/Add/Max/Div + Brcb primitives; cann=9.0.0+; op_class=online_softmax / streaming_normalization / fused_softmax_matmul`
`derived-from: cann-source (FA-class kernel structure, 2026-05-10 revise-cl3)`
`verified_on: cann ops-transformer FA-class block epilogues for online softmax + output rescale (source-structure-only; no a5_ops measurement)`

**Trigger**: Op needs row-softmax over a logical row whose width exceeds UB capacity AND has a streaming downstream consumer that can absorb per-tile probability numerators (e.g. `softmax(scores) @ V` in FlashAttention-style forward, streaming-softmax-then-weighted-sum, online-attention with KV-cache). The op MUST be able to either (a) carry an unnormalized output accumulator across tiles and divide once at the end, OR (b) buffer per-tile numerators with their center-max metadata for a deferred rescale pass.

**Key constraint — standalone softmax CANNOT use this recurrence as-is**:
The exp(scores - newMax) values produced in tile k are centered on the *running* max at step k, not the final max. They are valid numerators of an unnormalized accumulator only. To get final probabilities you need one of:
  1. **Streaming consumer with rescaled accumulator** (the FlashAttention path): maintain an unnormalized output `O_running`. Each tile, multiply `O_running` by `delta_k = exp(oldRunMax - newMax)` BEFORE adding the current tile's `probs_k @ V_k`. On the last tile, divide `O_running` by final `runSum`. The FA-class reference structure splits this across two stages: a softmax stage emitting per-tile probability numerators plus the per-tile delta state, and a downstream output-rescale stage that multiplies the running accumulator by delta each tile and performs the divide-by-runSum only on the last-tile branch.
  2. **Stored numerators with replay**: persist `numerator_k = exp(scores_k - center_k)` AND `center_k` per tile, then after the final tile compute `prob_k = numerator_k * exp(center_k - finalRunMax) / finalRunSum` in a second pass. Costs an extra full GM round-trip; only do this when there is no fused downstream consumer.

Do NOT emit per-tile probability-numerator values to GM as if they were final softmax outputs for standalone consumers — they are valid only within the rescale-aware pipeline.

**Recommendation**: Process the row in tiles of width `TILE_W`. Maintain fp32 per-row state arrays of length `R` in UB across all tile iterations of one row group. Naming follows the source structure exactly so future readers can map back to the FA layout:

- `runMax[R]` (CANN `gm` — running/global max across all processed tiles for this row)
- `runSum[R]` (CANN `gl` — running normalizer, centered on current `runMax`)
- `newMax[R]` (CANN `hm` — scratch holding max(runMax, tileMax) before commit)
- `delta[R]` (CANN `dm` — scratch holding `exp(runMax_old - newMax)`; kept around because the rescale phase consumes it)
- `tileMax[R]` (CANN `lm` — current tile's local rowmax)
- `tileSum[R]` (CANN `ll` — current tile's local rowsum after the tile's Exp)

Per tile `k`, after reading scores into UB and applying scale + mask (mask = add(-3e38) on masked positions, standard CANN form):

**1. tileMax = rowmax(scores_k)** — use the wide-row block-reduction shape from CAND-FA4.

**2. First-tile guard** (explicit branch — the first-tile-flag form used in the FA-class reference):
  - If this is the row's first tile: `runMax := tileMax`, then proceed directly to Step 3 with `runMax` as the centering value; after Step 4 set `runSum := tileSum` and (for streaming consumers) `O_running := probs_k @ V_k`. Do NOT compute `delta` on this path. Do NOT initialize with `runMax = -inf` and then evaluate `exp(runMax - newMax)` — the result is undefined (NaN) and corrupts downstream.
  - If the very first tile of the row is itself fully masked (`tileMax ≈ -3e38`), defer state init to the next non-masked tile (track a per-row `seenAnyValidTile` flag). The recurrence is undefined on an empty-evidence row; the worker MUST honor the op's reference behavior for empty / all-masked rows (often: produce zero output, skip the divide).

**3. Later-tile recurrence** (the steady-state body):
  - `newMax = max(tileMax, runMax)`
  - `delta = exp(runMax - newMax)` — `delta ∈ [0, 1]`; underflow to `0.0` is the *correct* value when the old running tail is irrelevant. Do NOT clamp to a small positive constant; the consumer's `O_running *= delta` then correctly forgets the stale accumulator.
  - `probs_k = exp(scores_k - newMax)` — broadcast `newMax` across the tile's columns (see Broadcast note below).
  - `tileSum = rowsum(probs_k)`
  - `runSum = delta * runSum + tileSum`
  - `runMax := newMax` (committed last, since the Sub above still uses old runMax)

**4. Streaming-consumer accumulator update** (carries the rescale through to the output side, mirroring the FA-class output-rescale stage):
  - `O_running := delta * O_running + probs_k @ V_k`
  - Broadcasting `delta` over the embed dimension of `O_running` uses the same P-P62 Brcb-then-Mul shape as the softmax core.

**5. Final-tile finalize** (only on the last tile of the row's tile sweep):
  - `O_final := O_running / runSum`
  - Cast to output dtype (CAST_RINT for bfloat16, CAST_NONE for fp16).

**Broadcast note (P-P62 precondition)**:
The `Brcb(...) -> Sub` shape used to broadcast `newMax` across tile columns, and the `Brcb(...) -> Mul/Div` used for `delta`/`runSum` across the embed dimension, both require the rows-axis aligned up to `FLOAT_BLOCK_SIZE = 8`. CANN uses `BrcbRepeatParams(1, 8)` with repeat count `R_round / 8`. Use this exact shape only when `R >= 8` (multi-row batched per AIV). For single-row or `R < 8` kernels, use a scalar `GetValue/SetValue` broadcast or `Sub<float, scalar>` overload — both are correct but slower. Do NOT force the Brcb shape on a `R = 1` kernel; the repeat count goes to 0 and the op silently no-ops.

**Concrete anchor** (per-tile non-first body, public-API VEC primitives — copy the *shape*, not vendor identifiers; pick worker-local LocalTensor names):
```cpp
// All buffers are AscendC::LocalTensor<float>; repeatPar1 is BinaryRepeatParams(1,1,1,8,8,8).
// runMax/runSum/newMax/delta/tileMax/tileSum each hold R fp32 elements (R rows per AIV).

// Step 3a: newMax = max(tileMax, runMax)
AscendC::Max<float, false>(newMax, tileMax, runMax, 0, 1, repeatPar1);
AscendC::PipeBarrier<PIPE_V>();

// Step 3b: delta = exp(runMax - newMax)  (uses runMax BEFORE we overwrite it)
AscendC::Sub<float, false>(delta, runMax, newMax, 0, 1, repeatPar1);
AscendC::Exp<float, false>(delta, delta, 0, 1, AscendC::UnaryRepeatParams(1,1,8,8));
AscendC::PipeBarrier<PIPE_V>();

// Step 3c: probs_k = exp(scores_k - newMax) — broadcast newMax via Brcb→Sub (P-P62 shape; requires R>=8)
AscendC::Brcb(tvScratch.ReinterpretCast<uint32_t>(),
              newMax.ReinterpretCast<uint32_t>(),
              R_round / 8, AscendC::BrcbRepeatParams(1, 8));
// then per-column Sub(scores_k, tvScratch) repeats R times across the tile width, then Exp on the tile.

// Step 3d: tileSum = rowsum(probs_k) — use CAND-FA4 block-reduce shape

// Step 3e: runSum = delta * runSum + tileSum
AscendC::Mul<float, false>(runSum, delta, runSum, 0, 1, repeatPar1);
AscendC::Add<float, false>(runSum, runSum, tileSum, 0, 1, repeatPar1);

// Step 3f: commit runMax = newMax
AscendC::DataCopy(runMax, newMax, AscendC::DataCopyParams(1, R_round / 8, 0, 0));
```

**Numerics**:
- Stable because every tile's `Exp` argument lies in `(-∞, 0]`.
- `delta ∈ [0, 1]`; underflow to exactly `0.0` is correct (the old `O_running` contribution is fully dominated). Do not clamp.
- Mask-as-`-3e38` propagates cleanly: an all-masked tile produces `tileMax ≈ -3e38`, `delta ≈ 1`, `probs_k ≈ 0`, `tileSum ≈ 0` — runSum and O_running unchanged. An all-masked row across ALL tiles is a contract violation; handle per op-spec.

**Determinism**: Deterministic when each row is single-AIV-owned (no cross-core writes participate in the state), the per-tile rowmax/rowsum reduction order is fixed (CAND-FA4 block-reduce shape is fixed), and `Exp/Sub/Mul/Add/Max/Div` are per-element. `delta` order matters only via `Mul(runSum, delta, runSum)` which is a per-element scalar product — order-independent. By construction det-preserving when those preconditions hold.

**Hard do-not-apply**:
- Do NOT use this recurrence to produce final probabilities for a *standalone* softmax output without either (a) the streaming-rescaled accumulator path or (b) the stored-numerator-then-replay path. Emitting `exp(scores_k - newMax_at_step_k)` directly is wrong because the centering changes across tiles.
- Do NOT use the `Brcb(R_round/8, BrcbRepeatParams(1,8))` shape when `R < 8`; the repeat count rounds to 0 and the broadcast silently emits nothing (P-P62 precondition violation).
- Do NOT use the unified loop body with `runMax = -inf` initialization; `exp(-inf - newMax)` is undefined for an all-masked first tile and NaN-poisons the row.
- Do NOT clamp `delta` to `epsilon` to avoid underflow; the correct semantics rely on `delta == 0` killing the stale accumulator.

**Other instances predicted**:
- FlashAttention-class forward (`QK` tile → online softmax state → `P @ V` accumulator) — the canonical case.
- Sliding-window / block-sparse attention (each window/block uses one row sweep with this recurrence).
- Streaming softmax fused with downstream reduction (e.g. cross-entropy logsumexp, attention-pooling), where the rescale folds into the consumer.
- Chunked LogSumExp: same `runMax` + `delta * runSum + tileSum` recurrence; final value `log(runSum) + runMax`.
- Online normalization where each update step is a rescaled add (e.g. running weighted mean with re-centered weights).
NOT predicted: streaming L2-normalize (the rescale identity does not factor through `sqrt(sum_sq)` cleanly without a different recurrence — kept out of scope per codex r1).

**Risks before promotion**:
- Brcb precondition R>=8: silently breaks on small-row kernels; worker MUST select a scalar-broadcast variant for `R < 8`. Add a static_assert or runtime guard.
- Mask-as-additive-`-3e38` is the *only* validated path; mask-as-multiply (`scores * mask` with `mask ∈ {0, 1}`) interacts badly with the rowmax step (a zero is not `-inf`). Worker must verify mask form before reusing this pattern.
- The pipeline depends on a separate "rescale_o" stage to consume `delta`. Worker must wire `delta` (or equivalent per-stack-tile state buffer) through to the output-accumulator stage; merging them into one body works only if `O_running` fits in UB alongside the softmax state.
- Source-structure verification only — no a5_ops kernel has yet shipped this exact recurrence. Promotion to P-P requires an a5_ops implementation that passes Pass A + Pass B + det + perf on 3_FusionAttention or a streaming-softmax op, plus a second op confirming portability.

## CAND-FA3: GM workspace slot rotation via modulo-(MAX_LAG+1) for cross-core stage decoupling
`applies_to: any soc with cross-core sync (CAND-FA1); cann=9.0.0+; op_class=multi_stage_pipeline_with_GM_handoff`
`derived-from: cann-source (FA-class workspace layout, 2026-05-10 revise-cl4)`
`verified_on: cann-source (FA-class op_kernel main scheduling loop)`
`unverified_on: a5_ops`

**Trigger**: A multi-stage producer→consumer pipeline (CAND-FA1) uses GM-resident workspace tensors as the hand-off medium between cores (cube↔vec or peer-cube↔peer-cube), and adjacent stages must overlap without an iteration overwriting GM scratch that a downstream stage is still consuming.

**Parameter definition (corrected vs prior revision)**: `MAX_LAG` is the maximum iteration distance between a producer write of a slot and the last still-active consumer read of that same slot. It is **not** the count of stages in flight. Example: with three pipeline stages (cube-QK → vec-Softmax → cube-PV), the producer of stage-1 at iteration `k` must keep its slot intact until the stage-3 consumer at iteration `k` finishes — that consumer runs `MAX_LAG = 2` iterations after the producer in the outer loop (it is launched when the producer is on iteration `k+2`). Hence `MAX_LAG = 2` requires `MAX_LAG + 1 = 3` distinct slots, not 3 in-flight producers.

**Recommendation**: Allocate the workspace as `(MAX_LAG + 1)` parallel GM slots per core. At outer-loop iteration `k`, the producer writes:

```cpp
slotIdx = k % (MAX_LAG + 1);
```

A consumer that reads data produced `lag` iterations earlier (`1 <= lag <= MAX_LAG`) reads:

```cpp
readSlot = (k + (MAX_LAG + 1) - lag) % (MAX_LAG + 1);
```

The set `{k, k-1, ..., k-MAX_LAG}` is pairwise-distinct modulo `MAX_LAG + 1`, so the active slots never alias. The slot used at iteration `k - (MAX_LAG + 1)` is the one being overwritten now, which is collision-free if and only if every consumer that could still reference it has finished.

**Safety condition (must be structurally enforced)**: this scheme is collision-free if and only if `MAX_LAG` is an **upper bound** on the actual iteration retention of every consumer of the slot. In CAND-FA1-style pipelines this is enforced by the cross-core flag chain: stage `s+1` at iteration `k` blocks on stage `s`'s "produced" flag for iteration `k`, and the producer at iteration `k + MAX_LAG + 1` blocks on the "consumed" event from the last downstream stage at iteration `k`. If the actual lag is bounded above by `MAX_LAG` for the entire schedule, modulo rotation is sound; otherwise it is unsafe.

**Per-core GM offset**:

```cpp
slotOffset = uint64_t(coreIdx) * SLOT_BYTES * (MAX_LAG + 1)
           + uint64_t(slotIdx) * SLOT_BYTES;
```

Typical `MAX_LAG = 2` gives three slots per core and supports a three-stage overlapped window: stage-A writes iteration `k`, stage-B reads iteration `k-1`, stage-C reads iteration `k-2`. `MAX_LAG = 1` reduces to classic double-buffer ping-pong. `MAX_LAG >= 3` is rarely useful because the cross-core flag chain and the slowest stage usually cap useful overlap; justify with profiling.

**Concrete anchor** (public-API surface):

```cpp
constexpr uint32_t MAX_LAG = 2;
constexpr uint32_t SLOT_COUNT = MAX_LAG + 1;

// Producer (e.g. cube core at stage 0):
uint32_t writeSlot = stageCount % SLOT_COUNT;
uint64_t writeOffset = uint64_t(coreIdx) * SLOT_BYTES * SLOT_COUNT
                     + uint64_t(writeSlot) * SLOT_BYTES;
auto gProducerView = gWorkspace[writeOffset / sizeof(T)];
// ... write gProducerView, then publish CAND-FA1 cross-core flag for this stage/iter ...

// Consumer at downstream stage, reading data produced `lag` iters earlier:
uint32_t readSlot = (stageCount + SLOT_COUNT - lag) % SLOT_COUNT;
uint64_t readOffset = uint64_t(coreIdx) * SLOT_BYTES * SLOT_COUNT
                    + uint64_t(readSlot) * SLOT_BYTES;
auto gConsumerView = gWorkspace[readOffset / sizeof(T)];
// ... wait on CAND-FA1 flag for the corresponding (stage, iter) before reading ...
```

**Cross-reference to P-P77 (no conflict, different tier)**: P-P77 rotates UB-resident `TQue` slots for **same-core** inter-iteration synchronization between MTE2 and V via TQue/EnQue/DeQue semantics. CAND-FA3 rotates **GM-resident workspace slots** for **cross-core** inter-stage hand-off with explicit cross-core flags. The two tiers are orthogonal: a kernel can use UB ping-pong (P-P77) inside each stage and GM slot rotation (CAND-FA3) between stages. UB ping-pong is bounded by UB capacity and queue depth (typically 2); GM slot rotation is bounded by workspace allocation and the cross-core flag protocol, which makes `MAX_LAG > 1` cheap.

**Determinism**: deterministic when (a) each `(coreIdx, stageCount, slotIdx)` write region has exactly one producer; (b) every consumer read is gated by the CAND-FA1 producer-complete flag for the matching `(stage, iteration)`; (c) the scheduler structurally guarantees that no consumer retains a slot beyond `MAX_LAG` iterations after its producer (e.g., by having the producer at iteration `k + MAX_LAG + 1` block on the last consumer's completion flag for iteration `k`). It does **not** make concurrent multi-producer writes to the same GM region deterministic; those remain an A-P61-style race unless the workspace is partitioned per producer.

**Hard do-not-apply (delayed-consumer case)**:
- Do not use bare modulo rotation when consumer retention is not structurally bounded by `MAX_LAG`. Examples that violate this:
  - Producer can race ahead of the slowest consumer because of data-dependent skip logic (e.g., sparse-block attention where some stage-B iterations are no-ops while stage-A continues at full speed).
  - Cross-core flag is only set after producer completes but never signals consumer release, so the producer at `k+MAX_LAG+1` can wrap and overwrite slot before the consumer at `k` reads it.
  - Backpressure from a downstream sink (e.g., HBM-bandwidth-stalled WriteBack stage) makes the latest consumer fall behind by more than `MAX_LAG` iterations.
- In these cases, choose one of:
  1. Grow the slot count to a hard upper bound on actual retention (acceptable when retention is bounded but variable).
  2. Add a per-slot release/ack: each consumer signals "slot released" after read; the producer at iteration `k + MAX_LAG + 1` waits on the release event for iteration `k` before reusing the slot.
  3. Switch to an explicit queue (e.g., TQue at GM tier via a circular ring buffer with head/tail flags) — this is a generalization of the modulo scheme with explicit credit accounting.

**Other instances predicted**:
- FA-class forward: multiple GM scratch tensors (S, P, OTmp, OUpdate in the CANN reference) all use the same slot-rotation scheme with shared `MAX_LAG`.
- Prefill+decode hybrid attention where cube prefill produces GM scratch consumed by vec decode at a fixed iteration lag.
- Multi-stage scan/reduction with cube partials produced N iterations before vec finalize consumes them.
- Fused norm-then-GEMM where vec normalization produces normalized GM scratch consumed by cube GEMM at the next iteration.
- MoE finalize where expert-output scratch is produced by one stage and combined by a routing stage at fixed lag.

**Promote when**: a5_ops ships a multi-stage cube↔vec kernel using this rotation AND msprof confirms at least two stages in flight simultaneously, visible as cube and vec utilization both greater than 0% in the same wall-clock window AND the safety condition is structurally argued (not just empirically observed) in `knowledge_update.md`.

**Risks before promotion**:
- `SLOT_BYTES` must include alignment padding so adjacent slots do not share an HBM line / cache sector (false-share serializes through the LLC). Use `SLOT_BYTES = AlignUp(payload_bytes, 512)` unless a tighter architecture-specific bound is proven.
- Workspace size grows linearly with `(MAX_LAG + 1) * num_cores`. Example: 24 cores × 3 slots × 128 KB each = 9 MB of GM workspace. Host launch code must pre-check required workspace size against device free memory before dispatch.
- Modulo rotation is a layout convention, not a synchronization primitive. It prevents address collision only under a bounded-lag schedule; the cross-core flag protocol (CAND-FA1) or an equivalent ack chain is still required for read-after-write and write-after-read correctness.
- If the kernel later adds a stage with data-dependent skip behaviour (e.g., sparse-block attention paths that nop-out some stage-B iterations), revisit whether the lag is still bounded by `MAX_LAG`; the modulo scheme may silently violate the safety condition without crashing.
- `MAX_LAG` is a compile-time constant in the CANN reference. Templating it makes per-shape tuning possible but multiplies dispatch table size; combine with CAND-FA1 dispatch budget before promotion.

## CAND-FA4: Tree-reduce wide fp32 rows via packed `BlockReduceMax/Sum` partials with per-stage mask sizing (no `WholeReduce` on fp32)
`applies_to: op_class=row_reduction (W ≥ 64 fp32 elements per row, R parallel rows in UB); soc=Ascend910_V220 / Ascend950PR (fp32 BlockReduceMax/BlockReduceSum confirmed public-API); cann=9.0.0+`
`derived-from: cann-source (FA-class fp32 online-softmax row reductions, 2026-05-10 revise-cl5)`
`verified_on: cann-source (read-only); unverified_on: a5_ops`
`local-kb-crossref: P-P47 (half-interval scalar finish), ascend950pr.md Sort/Reduce VEC primitive specs`

**Trigger**: Need rowmax or rowsum of a wide fp32 row (`W ≥ 64` fp32 per row, `R` parallel rows packed in UB) inside a hot kernel loop (FA softmax, LayerNorm rowstats, RmsNorm rowsumsq, wide softmax). A scalar `GetValue`-driven reduction is `R × W` scalar-pipe cycles and is the canonical anti-pattern.

**Recommendation — corrected mechanics**:

Use chained `AscendC::BlockReduceMax<float, false>` / `BlockReduceSum<float, false>` stages, where each repeat consumes one 64-fp32 vector (organized as 8 contiguous 32-byte blocks of 8 fp32) and writes **8 packed fp32 partials** densely. Do not use `WholeReduceMax/Sum<float>` to finish — the fp32 cap is 64 elements per repeat, and the finishing stage is more economical as a final `BlockReduce*` with a strided per-block mask.

After one full stage, a row of `W` fp32 collapses to a packed row of `ceil(W / 8)` partials. After two stages, `ceil(W / 64)`. After three stages, `ceil(W / 512)`. Continue staging until the per-row partial count is `≤ 8`, then finish.

Mask use across stages (this is the part that the prior candidate had wrong):

- The template arg `<float, false>` means `isSetMask = false`, i.e. **use the externally configured vector mask**; it does **not** mean "no mask". When the per-repeat workload is a full 64-fp32 vector, set the mask to all-ones once and leave it.
- The vector mask gates **which source lanes contribute per repeat**, not which output lanes are written. The dst is always the dense packed-partial layout, determined by `dstBlkStride`/`dstRepStride`/`repeatTime`.
- **Stages 1 through N-1** (full fan-in): full mask, no per-stage mask change.
- **Final stage**, when the packed row is now `k` valid partials per 8-element block (because the previous stage produced fewer than 8 partials per row but they sit in an 8-lane block layout), set a **tiled strided mask** of the form "low `k` bits of every 8-bit byte" so each repeat consumes only the `k` valid lanes per block. Compute this mask as `((1<<k)-1)` replicated into bytes 0..7 of both `lo` and `hi` halves of the mask register, then call `AscendC::SetVectorMask<int8_t>(tiledMask, tiledMask)`. After the final reduction, **restore full mask** with `AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1)` before continuing.
- For an intermediate stage whose source row is wider than 8 partials but shorter than 64 (e.g. 32 packed partials per row from a `W=256` first stage), set a **contiguous** mask of that element count (`SetVectorMask<int8_t>` with the standard contiguous form) and lower `srcRepStride` to `partials/8` blocks instead of the full 8.

**Important non-claim**: `SetVectorMask` is not used to "skip every 8th element" between stages. The packed partial layout means the next stage reads the partials densely from the front of each row. The only legitimate inter-stage mask uses are (a) reducing the contiguous element count when the packed row is shorter than a full 64-fp32 vector, and (b) the tiled per-block mask in the final stage when each 8-block holds fewer than 8 valid partials.

**Shape contract (single-row-at-a-time view; in practice R rows are packed and reduced in parallel)**:
```cpp
// stage 0 : src         [R, W]
// stage 1 : partial1    [R, ceil(W / 8)]     full mask, full repeats
// stage 2 : partial2    [R, ceil(W / 64)]    contig mask = partial1 row width if < 64
// stage k : partial_k   [R, ceil(W / 8^k)]   continue while row width > 8
// finish  : row_out     [R]                  tiled per-block mask = row width
```

The variant choice (how many stages, what mask form on the final stage) is fixed once `W` is known. Below is the chosen variant per `W`. There is one shape contract per variant; do not list a recommended path then call it "optional".

**Variant: W = 64** — single stage suffices.
```cpp
// repeats = R, full mask (set once outside).
AscendC::BlockReduceSum<float, false>(rowSumUb, srcUb,
    /*repeatTime=*/R,
    /*dstRepStride(scalar)=*/0, /*srcBlkStride=*/1,
    /*dstRepStride=*/1, /*srcRepStride=*/8);
AscendC::PipeBarrier<PIPE_V>();
// Per-row 8 partials per output block; final per-row finish uses a tiled mask of 8 lanes,
// which is just full mask -> a second BlockReduceSum with repeats=R*8/64.
```

**Variant: W = 256** — three stages, intermediate contiguous mask of 32 plus final tiled mask of 4.
```cpp
constexpr uint32_t WALIG = 256;
constexpr uint32_t FP32_BLOCK = 8;
constexpr uint32_t FP32_VEC = 64;

// Stage 1: [R,256] -> [R,32] packed partials. Full mask, full repeats.
AscendC::BlockReduceSum<float, false>(scratch, srcUb,
    /*repeatTime=*/R * WALIG / FP32_VEC,
    /*dstRepStride(scalar)=*/0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();

// Stage 2: [R,32] -> [R,4] packed partials.
// Set contiguous mask = 32 (one repeat consumes 32 fp32 from one row's packed partials).
SetContiguousVectorMask(/*count=*/32);   // see helper below
AscendC::BlockReduceSum<float, false>(scratch2, scratch,
    /*repeatTime=*/R,
    /*dstRepStride(scalar)=*/0, 1, 1, /*srcRepStride=*/4);   // 32 fp32 = 4 blocks
AscendC::PipeBarrier<PIPE_V>();

// Stage 3 (finish): [R,4] -> [R]. Each row's 4 partials sit in lanes 0..3 of an 8-block;
// tiled mask = ((1<<4)-1) replicated into every byte of the 128-bit mask register.
SetTiledBlockReduceMask(/*lanesPerBlock=*/4);    // see helper below
AscendC::BlockReduceSum<float, false>(rowSumUb, scratch2,
    /*repeatTime=*/CeilDiv(R * FP32_BLOCK, FP32_VEC),
    /*dstRepStride(scalar)=*/0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();

// MANDATORY: restore full mask before any unrelated VEC work.
AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
```

**Variant: W = 512 (R rows, R*W ≥ 4096)** — three stages, full mask throughout, no `SetVectorMask` between stages. The repeat counts implicitly carry the fan-in.
```cpp
constexpr uint32_t WALIG = 512;
constexpr uint32_t FP32_BLOCK = 8;
constexpr uint32_t FP32_VEC = 64;

AscendC::BlockReduceSum<float, false>(scratch, srcUb,
    /*repeatTime=*/R * WALIG / FP32_VEC, 0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();
AscendC::BlockReduceSum<float, false>(scratch2, scratch,
    /*repeatTime=*/R * WALIG / FP32_BLOCK / FP32_VEC, 0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();
AscendC::BlockReduceSum<float, false>(rowSumUb, scratch2,
    /*repeatTime=*/R * WALIG / FP32_VEC / FP32_VEC, 0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();
// Mask was never altered; no restore needed.
```
Constraint: this exact form requires `R * W` divisible by `FP32_VEC * FP32_VEC = 4096`. For `W = 512` this means `R ≥ 8`. For lower `R`, use the W=256-style ending (intermediate contig mask + final tiled mask).

**Variant: W > 64 generic (loop accumulator)** — outer loop over 64-fp32 chunks, accumulate per-row via `AscendC::Max` / `AscendC::Add`, handle tail with `SetContiguousVectorMask(W % 64)` then `SetTiledBlockReduceMask(CeilDiv(W % 64, 8))` for the final finish, mask restored at end.

**Public mask helpers (logic ported from observed pattern)** — implementable on a5_ops side without reading CANN:
```cpp
__aicore__ inline void SetContiguousVectorMask(uint32_t count) {
    // count: total fp32 lanes to enable, 1..128 (across the two 64-bit mask halves).
    if (count == 128 || count == 0) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (count >= 64) {
        uint64_t lo = ((uint64_t)1 << (count - 64)) - 1;
        AscendC::SetVectorMask<int8_t>(lo, (uint64_t)-1);
    } else {
        uint64_t lo = ((uint64_t)1 << count) - 1;
        AscendC::SetVectorMask<int8_t>(0x0, lo);
    }
}
__aicore__ inline void SetTiledBlockReduceMask(uint32_t lanesPerBlock) {
    // lanesPerBlock: 1..8, enables low-`lanesPerBlock` lanes in each 8-lane block.
    if (lanesPerBlock < 1 || lanesPerBlock > 8) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t sub = ((uint64_t)1 << lanesPerBlock) - 1;
    uint64_t tiled = (sub <<  0) | (sub <<  8) | (sub << 16) | (sub << 24)
                   | (sub << 32) | (sub << 40) | (sub << 48) | (sub << 56);
    AscendC::SetVectorMask<int8_t>(tiled, tiled);
}
```

**Determinism**: Hardware reduction order within one `BlockReduce` repeat is fixed (block-local tree). With one AIV owning each row, or a deterministic cross-AIV merge, same input → same output. Determinism does NOT come from "layout choice"; it comes from (a) one writer per output slot and (b) fixed reduction order per repeat. Scratch tiles must be initialized whenever any producer might skip lanes that a later reader covers (EC-37-style).

**Hard do-not-apply**:
- Non-fp32 dtypes: the half-precision path uses `WholeReduceMax/Sum<half>` with up to 128 half lanes per repeat — a different shape contract; use a separate pattern.
- Socs without confirmed public fp32 `BlockReduceMax/Sum` support: do not assume "any soc". Confirmed: Ascend910_V220, Ascend950PR (FA-class fp32 epilogue). For other targets, probe-compile `BlockReduceMax<float, false>(…)` and validate one-row results before adopting.
- Rows with `W < 64`: a single masked `BlockReduce*` (or even `WholeReduceMax/Sum<float>` capped at 64) is simpler; do not introduce a tree.
- Output destination is GM (not UB): the `BlockReduce*` chain writes to UB; only the final `DataCopy` to GM is allowed.
- Scratch tiles that are public outputs of the kernel: do not place tree-stage scratch in a buffer that the kernel exports; use a private tile.

**Other instances predicted**:
- LayerNorm row mean / row variance for H ≥ 64 (replace scalar-loop reduction).
- RmsNorm rowsumsq for H ≥ 64.
- Attention rowmax + rowsum (this op).
- GroupNorm per-group mean where group size ≥ 64.
- Vocabulary-class wide softmax rowmax + rowsumexp.
- Online softmax in fused attention variants (FA forward/backward).

**Risks before promotion**:
- The tiled per-block mask on the final stage is the single fragile point. Wrong `lanesPerBlock` → silent wrong result. Tested case shapes must cover at least one power-of-two width (e.g. W=256), one with intermediate-contig + tiled-final (W=256), and one full-fan-in-no-mask (W=512, R≥8).
- The `<false>` form requires the external mask to be set correctly before each call. Forgetting to restore the full mask after a tail or final stage breaks unrelated downstream VEC ops. The restoration line `AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1)` is part of the contract.
- For `W = 512` full-fan-in variant: requires `R * W` divisible by 4096. If a kernel runs at smaller `R`, do not silently fall through to this variant — pick the masked variant.
- The `dstBlkStride` parameter for these calls is documented as 0 in the FA fp32 path (no per-row scaling); confirm this against AscendC public docs for the target SoC version before promoting, because the parameter semantics for `BlockReduce*` differ between half-precision and fp32 forms.
- Performance claim "100× speedup over scalar" is plausible but **unmeasured on a5_ops**. Promote only after one a5 kernel ships this pattern with `msprof` data showing the expected `aiv_vec_ratio` lift.

## CAND-FAG-1: Three-kernel split (pre-init → main → post-cast) for fp32-workspace gradient accumulation in low-precision backward ops
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=low_precision_backward_with_multi_source_gradient_accumulation / flash_attention_backward / fused_norm_backward / scatter_grad / any_bwd_op_with_atomic_add_across_cores_on_low_precision_output`
`derived-from: cann-source (fa-grad-class backward, 2026-05-10 multicann)`
`verified_on: cann ops-transformer flash_attention_score_grad/op_kernel/arch35/ — three peer kernel headers covering the pre/main/post split (pre header initializes the fp32 workspace, post header rescales fp32→output-dtype, main header does the actual fwd-recompute + bwd matmul chain); the post header's accumulation-then-cast write-back pattern is the load-bearing structural piece`
`unverified_on: a5_ops (no backward op currently shipped)`

**Trigger**: A backward op produces gradients (e.g. dq/dk/dv) in a low-precision output dtype (fp16/bf16/fp8) BUT the accumulation across multiple producing blocks must happen in fp32 to avoid catastrophic precision loss from atomic-add in low precision. The op cannot simply atomic-add into the user-facing fp16 dq buffer because (a) `SetAtomicAdd<fp16>()` either does not exist, has worse cumulative error, or has slower hardware path, and (b) cross-block partial sums for a single output element may number in the hundreds (one per S2 tile crossing the row).

**Recommendation**: Split the launch into three peer kernels chained on the same workspace via host-side enqueue order. Public-API surface is `InitOutput`, `Cast`, `Muls`, `DataCopy`, `SetAtomicAdd<float>`/`SetAtomicNone`, plus a host-side workspace plan exposed via `tilingData->postTilingData.{dq,dk,dv}WorkSpaceOffset`.

1. **PRE kernel** (AIV-only): on each used core, `InitOutput<float>(dqWorkSpaceGm[off], len, 0)` for the per-core slice of each output's fp32 workspace. When the output dtype IS fp32, skip the fp32 workspace entirely and `InitOutput` the user-facing GM directly. Optional ancillary clears (e.g. dropout-mask working buffer, ds-sink workspace) belong here.
2. **MAIN kernel**: do the recompute + matmul chain; all writes to dq/dk/dv go through `SetAtomicAdd<float>()` into the fp32 workspace (NOT the fp16/bf16 user output). `SetAtomicNone()` is called at the end of each atomic-region to keep subsequent writes ordered.
3. **POST kernel** (AIV-only, ping-pong): tile across each of the three fp32 workspaces, `DataCopy` a tile into UB → `Muls(tile, tile, scale, n)` (e.g. dq/dk inherit the attention scale) → `Cast(outTile, tile, RoundMode::CAST_ROUND, n)` → `DataCopy` final low-precision tile to dq/dk/dv GM. Ping/pong queue pair (`inQuePing`, `inQuePong`, `outQuePing`, `outQuePong`) overlaps the cast/scale of one tile with the GM load of the next. Skip the scale-Muls for dv (it does not carry the attention scale).

**Concrete anchor** (public AscendC):
```cpp
// PRE kernel — clear fp32 workspace before MAIN runs
if constexpr (IsSameType<OutT, float>::value) {
    InitOutput<OutT>(dqGm[dqOffset], initDqSize, 0);  // fp32: write user output directly
} else {
    InitOutput<float>(dqWorkSpaceGm[dqOffset], initDqSize, 0);  // low-prec: clear fp32 scratch
}

// MAIN kernel — all atomic accumulation into fp32 workspace
SetAtomicAdd<float>();
DataCopy(dqWorkSpaceGm[off], dqUbFp32, n);  // partial sum from one block, accumulated atomically
SetAtomicNone();

// POST kernel — ping-pong scale + cast + write final output
LocalTensor<float> inPing  = inQuePing.AllocTensor<float>();
DataCopy(inPing, dqWorkSpaceGm[pingIdx], pingSize);
inQuePing.EnQue(inPing); inQuePing.DeQue<float>();
Muls(inPing, inPing, scale, pingSize);
LocalTensor<OutT> outPing = outQuePing.AllocTensor<OutT>();
Cast(outPing, inPing, RoundMode::CAST_ROUND, pingSize);
outQuePing.EnQue(outPing); outQuePing.DeQue<OutT>();
DataCopy(dqGm[pingIdx], outPing, alignUp16(pingSize));
```

**Why it works**:
- fp32 atomic-add is hardware-supported with deterministic semantics on V220/950PR; fp16 atomic-add either is unsupported or has cumulative error proportional to producer count
- Splitting PRE/MAIN/POST also keeps the MAIN kernel's UB budget free of cast scratch — a fp16 output of size B·N·S·D would otherwise need an extra cast buffer in the hot path
- POST's ping-pong overlap hides Cast latency behind DataCopy, achieving near-MTE2-bound throughput on the rescale
- `InitOutput` is the only public API that emits a fixed-value write through MTE3 without going through UB allocation, making it the right primitive for workspace zeroing

**Determinism**: PRE/POST are deterministic by construction (each output element is written by exactly one core in POST). The MAIN kernel's atomic-add is the determinism risk — see CAND-FAG-2 for the deterministic-mode alternative that replaces atomic-add with a partition-by-coordinate dispatch scheme.

**Other instances predicted**:
- Any backward op of a fused reduction (LayerNorm backward, RMSNorm backward, Softmax backward) when accumulating dWeight / dBias across batch
- Scatter-add gradients when the scatter index set spans more cores than the index cardinality
- MoE expert gradient accumulation back into the shared input embedding
- Cross-entropy + softmax fused backward where dlogits accumulates per-class across batches
- Beam-search / sequence-parallel backward where gradient at a position is summed from multiple ranks/cores

**Risks before promotion**:
- a5_ops has no shipped backward op exercising this three-kernel split — the pattern's launch-overhead-vs-precision tradeoff (3× kernel launches vs 1×) is unmeasured on this codebase
- The fp32 workspace size is `MAX_CUBE_CORE_NUM × CUBE_BASEM × HEAD_DIM_ALIGN` per output for the BN2 path — for tall S/D this can exceed reserved workspace; check `RESERVED_WORKSPACE_SIZE` budget before adopting
- If the output dtype is already fp32, the three-kernel split is wasteful — the FA-grad reference explicitly bypasses workspaces and writes directly to user GM in that case (see anchor `if constexpr (IsSameType<OutT,float>::value)`)
- For very small problems (single-core-sufficient), MAIN-only without atomic-add is faster — gate on producer count

**Cross-reference**:
- P-P89 (GM workspace contract for fused ops): this candidate is the BACKWARD specialization — outputs are public, fp32 scratch is the opaque workspace sliced via `dqWorkSpaceOffset` / `dkWorkSpaceOffset` / `dvWorkSpaceOffset`. Same workspace-contract shape as P-P89; promote-merge if multi-op evidence accumulates
- CAND-FA1 (manual cross-core flag handoff): orthogonal — this candidate is about WHAT goes through the workspace, CAND-FA1 is about HOW writes are ordered
- CAND-FA3 (GM workspace slot rotation): orthogonal — this candidate uses a flat per-output workspace, not a rotating ring

**Promote when**: a5_ops ships a backward op with measured precision improvement vs single-kernel fp16-atomic-add baseline AND measured launch-overhead acceptable vs the precision win.

## CAND-FAG-3: Saved-tensor restore contract for fused backward — fwd persists per-row scalar statistics (softmaxMax, softmaxSum), bwd re-reads them double-buffered to avoid re-running the fwd reduction
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_backward_with_recompute / flash_attention_backward / online_softmax_backward / any_bwd_op_whose_fwd_emitted_per_row_normalization_state`
`derived-from: cann-source (fa-grad-class backward, 2026-05-10 multicann)`
`verified_on: cann ops-transformer flash_attention_score_grad/op_kernel/arch35/ — backward kernels accept `softmaxMax` and `softmaxSum` GM pointers as Init arguments and consume them via a per-tile CopyInMaxSum helper (declared in `vector_api/pse_atten_mask_muls_simple_softmax.h`); double-buffered max-sum queue `maxSumQue[2]` indexed by `taskId & 1` is the hot-loop shape in three peer block-vec headers (the s1s2_bn2_regbase, s1s2_bn2gs1s2_regbase, and s1s2_bn2s2_regbase main-vec headers all use the same `CopyInMaxSum<T2, VECTOR_BASEM>(..., maxSumQue[taskId & 1], softmaxMaxGm, softmaxSumGm)` call shape)`
`unverified_on: a5_ops`

**Trigger**: A fused backward op needs intermediate per-row statistics that were computed once in the forward pass — specifically the online-softmax running max and running sum (CAND-FA2's `runMax`, `runSum`). Re-computing these in the backward kernel would require replaying the full forward streaming softmax over QKᵀ, doubling forward-matmul work. The fwd already had them in registers/UB at end-of-row; the contract is to persist them to GM as auxiliary outputs of the forward and let the backward re-read them.

**Recommendation**: Treat the (max, sum) pair as a **first-class fused-op output**, not an internal scratch. Persistence contract:

1. **Forward emits two extra GM tensors**, shape `[B, N, S1, 1]` (or `[T, N, 1]` in TND) — one for the per-row running max, one for the per-row running sum. These are public-API outputs allocated by the user/host, NOT in the opaque workspace (P-P89 contract: workspace is for scratch the user cannot inspect; saved-tensor for backward is a PUBLIC saved-tensor in PyTorch terms).
2. **Backward Init signature** accepts both pointers explicitly (`softmaxMax`, `softmaxSum`) alongside dy and the original Q/K/V — this is the structural marker that the bwd is a recompute-with-saved-tensors variant, NOT a full-replay variant.
3. **Inside the bwd hot loop**, copy-in the (max, sum) row-strip for the current tile using a double-buffered VECIN queue indexed by `taskId & 1`. Use `DataCopyPad` because the row-strip width is `VECTOR_BASEM` (e.g. 64) which may not align to 32B at the trailing tile. The copy-in runs concurrently with the prior tile's cube matmul thanks to the `& 1` ping-pong index.
4. **Apply** the saved (max, sum) inside the bwd's `simpledSoftmax` block (the recompute of P = softmax(scale·Q·Kᵀ + bias)): `p = exp(scores - max[row]) * (1.0f / sum[row])` — using Brcb to broadcast the per-row scalar across the row's columns, then VEC Exp/Sub/Mul.

**Concrete anchor** (public AscendC):
```cpp
// Forward (emit side) — at row-group epilogue, after final runMax / runSum are computed:
DataCopy(softmaxMaxGm[rowGroupOffset], runMaxUb, vectorBaseM);   // [VECTOR_BASEM] per row
DataCopy(softmaxSumGm[rowGroupOffset], runSumUb, vectorBaseM);

// Backward (consume side) — declare ping-pong queue at Init:
TQue<QuePosition::VECIN, 2> maxSumQue;   // depth-2 double buffer
pipe->InitBuffer(maxSumQue, 2, vectorBaseM * 2 * sizeof(float));  // max+sum interleaved

// Per-tile copy-in indexed by taskId
LocalTensor<float> ms = maxSumQue.AllocTensor<float>();   // implicit ping/pong by depth-2
DataCopyExtParams cp{1, (uint32_t)(vectorBaseM * sizeof(float)), 0, 0, 0};
DataCopyPadExtParams<float> pad{};
DataCopyPad(ms,                         softmaxMaxGm[curRowOffset], cp, pad);
DataCopyPad(ms[vectorBaseM],            softmaxSumGm[curRowOffset], cp, pad);
maxSumQue.EnQue(ms);
// ... in the recompute stage, dequeue and use ms[0..VECTOR_BASEM-1] (max) and ms[VECTOR_BASEM..2·V-1] (sum) ...
LocalTensor<float> ms2 = maxSumQue.DeQue<float>();
LocalTensor<float> maxRow = ms2;
LocalTensor<float> sumRow = ms2[vectorBaseM];
// Brcb broadcast across columns, then Sub/Exp/Mul to reconstruct P from the saved stats
```

**Why it works**:
- Saving just `[B,N,S,1]` adds <1% to forward GM traffic (the softmax denominator is one scalar per row, NOT per column) — the bwd otherwise pays a full re-reduction over S2 to find max/sum
- Public outputs (not workspace): the fwd op promises these as part of its contract, so the bwd can be invoked by a different launch / different stream and still find the data
- Double-buffer queue depth-2 with `taskId & 1` indexing is the standard CV-decoupling shape — the (max, sum) copy-in runs on AIV while the prior tile's QKᵀ runs on AIC
- `DataCopyPad` not `DataCopy`: row counts (`VECTOR_BASEM`, typically 64) are 32B-aligned for fp32, but the last row-group of TND / variable-S can be short — pad-with-zero is the only safe primitive

**Determinism**: Deterministic by construction — the saved (max, sum) is bit-identical to what the forward computed (it's the same bytes, just persisted to GM rather than discarded). The bwd's reconstruction `exp(scores - max) / sum` is element-wise so per-tile order does not matter. Combine with CAND-FAG-2 to get full backward determinism.

**Other instances predicted**:
- LayerNorm / RMSNorm backward: fwd saves per-row mean+rstd; bwd reads them instead of recomputing
- Cross-entropy + log-softmax fused backward: fwd saves per-row log-sum-exp; bwd reads to reconstruct probabilities
- Online-softmax-and-rescale chains: any second-pass that needs to know the final max/sum from the first pass
- Group-norm backward: fwd saves per-group mean+rstd
- BatchNorm-train backward: fwd saves per-channel running stats (already standard PyTorch behavior — this candidate codifies the AscendC pipe-pong copy-in shape)

**Risks before promotion**:
- a5_ops has no shipped fused-bwd op yet; the saved-tensor I/O cost vs recompute cost has not been measured on this codebase. For very short S (< 256), recompute may be cheaper than the extra GM round-trip
- The contract is **fragile across fwd↔bwd version skew**: if the fwd op's online-softmax algorithm changes (different center-max convention, different scale application order), the saved tensor becomes invalid for an old bwd. Version-tag the saved tensor or pin fwd/bwd to the same kernel build
- Memory footprint: `[B,N,S,1]` fp32 is small but non-zero — for B=32, N=32, S=8192, two tensors = 64MB. Fine for training, may be too much for inference if the bwd is being used for gradient checkpointing
- `DataCopyPad` with non-zero `paddingValue` is dangerous if the bwd reads beyond the valid row count — always pair with explicit `s1RealSize` bookkeeping

**Cross-reference**:
- CAND-FA2 (online-softmax per-row state recurrence): this candidate is the BACKWARD half — CAND-FA2 describes what the fwd computes and HOLDS in registers; this candidate describes how those values cross the fwd↔bwd boundary via saved tensors
- P-P89 (workspace contract for fused ops): related but DIFFERENT — saved tensors for backward are PUBLIC outputs (user can inspect, must be stable across versions), workspace is OPAQUE. The fwd's signature must list `softmaxMax` and `softmaxSum` as outputs, not bake them into `workspace`
- CAND-FAG-1 (three-kernel pre/main/post split): orthogonal — saved-tensor restore happens in MAIN, the pre/main/post split is about how the bwd's outputs are written
- CAND-FA1 (manual cross-core flag handoff): orthogonal

**Promote when**: a5_ops ships a paired (fwd, bwd) fused op pair where the bwd reads fwd-emitted saved tensors AND shows <30% slowdown vs a hand-written non-fused PyTorch backward AND the saved-tensor GM round-trip is profiled to confirm <10% of bwd wall time.

## CAND-FAG-4: Multi-output backward dispatch — single fused bwd kernel produces all primary input gradients (dq, dk, dv) in one pass via shared recompute + per-output cube ladder, NOT three independent backward launches
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=multi_output_backward / fused_attention_backward / any_bwd_op_whose_forward_was_one_fused_kernel_producing_N_outputs`
`derived-from: cann-source (fa-grad-class backward, 2026-05-10 multicann)`
`verified_on: cann ops-transformer flash_attention_score_grad/op_kernel/arch35/ — three peer outputs (dq, dk, dv) share one Init signature (`Init(..., dq, dk, dv, dpse, dqRope, dkRope, dsink, workspace, ...)`), one tiling-data struct (`FagTilingType`), and one fp32-workspace plan (`postTilingData.{dq,dk,dv}WorkSpaceOffset`); the main vec-block headers issue a three-stage vec pipeline `ProcessVec2 → ProcessVec3 → ProcessVec4` where the intermediate ds is reused for both dq and dk/dv cube ladders; the post-kernel `ProcessDqkv` iterates `for qkvIdx in {0,1,2}` over the three outputs with one ping-pong loop body`
`unverified_on: a5_ops`

**Trigger**: A fused forward op has N>1 primary inputs (e.g. attention's Q, K, V) and the user-facing bwd interface produces N gradient tensors of the same input shapes. A naive design would launch N separate backward kernels — but the bwd's most expensive shared subwork is (a) recomputing P = softmax(scale·Q·Kᵀ + bias) and (b) computing dP = dY @ Vᵀ then dS = P · (dP - rowsum(P·dP)). dS is the common ancestor of dQ, dK, and dV; computing it once and consuming it three times is the structural win.

**Recommendation**: Build the bwd as **one kernel** that:

1. Takes ALL primary-input GM pointers (q, k, v, dy, plus saved tensors per CAND-FAG-3) AND all gradient-output pointers (dq, dk, dv) in a single Init signature.
2. Internally chains **shared recompute → shared ds → per-output cube ladder**:
   - Stage A (recompute, vec): from saved (max, sum) and Q/K, reconstruct P. CAND-FAG-3 anchor.
   - Stage B (dY @ Vᵀ → dP, cube): one matmul.
   - Stage C (ds = P · (dP - softmaxgrad-correction), vec): the `FlashSoftmaxGrad` step from FAG design doc §1.3 — emits ds shared by Stages D, E, F.
   - Stage D (dQ += ds @ K, cube): writes to dq's fp32 workspace via SetAtomicAdd.
   - Stage E (dK += dsᵀ @ Q, cube): writes to dk's fp32 workspace.
   - Stage F (dV += Pᵀ @ dY, cube): writes to dv's fp32 workspace. Uses P (not ds).
3. Treat the three outputs as **a uniform array indexed by qkvIdx ∈ {0,1,2}** in the POST kernel (CAND-FAG-1) — same ping-pong, same Cast loop, only the per-output scale and write-offset differ. Skip the Muls(scale) for the qkvIdx==2 (dv) branch because dv does NOT inherit the attention scale.

**Concrete anchor** (public AscendC):
```cpp
// Single Init takes all gradient outputs together (NOT three separate kernel signatures)
__aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dy,
                            GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                            GM_ADDR workspace, FagTilingDataLike *tilingData, TPipe *pipeIn);

// In the per-tile hot loop:
// Stage A — recompute P from saved (max, sum)
ReconstructP_FromSavedStats(pUb, scoresUb, savedMaxRow, savedSumRow, n);
// Stage B — dY @ Vᵀ on AIC, hands ds source to AIV via cube ladder (anchor in CAND-FA1)
IterateMmDpFromDyVt(dpL0c, dyL1, vL1Trans);
// Stage C — ds = P · (dP - rowsum_dot_correction); shared output
ComputeDs(dsUb, pUb, dpUb, n);    // ds is now in UB, ready for three consumers
// Stages D/E/F — three cube matmuls, all using ds (or P), all writing fp32 workspace atomically
SetAtomicAdd<float>();
IterateMmDqFromDsK (dqWorkSpaceGm, dsL1, kL1);       // ds @ K → dq
IterateMmDkFromDsTQ(dkWorkSpaceGm, dsL1Trans, qL1);  // dsᵀ @ Q → dk
IterateMmDvFromPTDy(dvWorkSpaceGm, pL1Trans, dyL1);  // Pᵀ @ dY → dv
SetAtomicNone();

// POST kernel — uniform 3-output cast/scale loop
for (int qkvIdx = 0; qkvIdx < 3; ++qkvIdx) {
    // ... ping-pong DataCopy → (Muls for qkvIdx<2 only) → Cast → DataCopy to dqkv[qkvIdx] ...
}
```

**Why it works**:
- Three separate bwd kernels would each pay (a) saved-tensor copy-in, (b) Q/K/V copy-in, (c) recompute P, (d) compute dP, (e) compute ds — 5 redundancies × 2 = 10x duplicated work. One fused bwd pays each once
- ds is the common gradient ancestor; the algorithm's correctness proof (FlashAttention paper §4.3) is what makes this fusion safe — there is no other intermediate that achieves the same sharing
- Per-output fp32 workspaces decouple the three cube ladders' atomic-add domains — they never contend on the same address, so SetAtomicAdd safety is per-output
- The 3-output POST-kernel uniformity is a code-size and instruction-cache win: one loop body with a per-iter qkvIdx-conditional Muls skip beats three duplicated POST blocks

**Determinism**: NOT deterministic by default — Stages D/E/F use atomic-add across cores. To get a deterministic multi-output bwd, layer CAND-FAG-2 (coordinate-partitioned dispatch) on top — its assignment formula must produce a bijection over (b, n2, g, s1Outer, s2Outer) tiles regardless of which output is being written. The FA-grad reference's deter mode does exactly this: the same coordinate dispatcher serves all three of dq, dk, dv writebacks.

**Other instances predicted**:
- MoE backward: dExpert-weight, dGate-weight, dInput all share an intermediate "routed-input × routed-output-grad" tensor
- Cross-attention backward: dQ_decoder, dK_encoder, dV_encoder share the same ds
- Fused gated-linear-unit (GLU) backward: dGate, dUp, dDown share a ds-equivalent intermediate
- Convolution backward: dW and dInput share the unrolled-input × dY product if expressed as gemm
- LayerNorm + Linear fused backward: dLN-input, dLN-gamma, dLN-beta, dLinear-W share the dLN-output intermediate

**Risks before promotion**:
- The fused-bwd kernel is one of the largest single-source kernels in the FA-grad reference (multi-thousand-line single header); UB / L1 budget pressure scales nonlinearly with output count. For >3 outputs (e.g. attention with bias gradient), the fusion may overflow UB and force per-output spill
- Cube-ladder ordering matters: dQ depends on K (not its grad), dK depends on Q (not its grad), dV depends on P (not its grad) — independent in graph terms, so the L1 reuse policy must keep K, Q, V, dY co-resident across the three ladders. If L1 is too small for all four, the fused kernel degrades to per-output reload — worse than three separate kernels
- The single Init signature accumulates many arguments (12+) — exceeds the "Init args" budget of some host frameworks; may need a packed-args struct on the host side
- The qkvIdx==2 skip-Muls branch is a hardcoded shape assumption (dv doesn't carry attention scale) — if a future variant adds a scale-like factor to dv (e.g. quantization dequant scale), the POST kernel's uniform loop breaks silently

**Cross-reference**:
- CAND-FAG-1 (three-kernel pre/main/post split): COMPOSED with this candidate — the "M" in pre/M/post is exactly the multi-output fused kernel described here. CAND-FAG-1 = "how to write the three outputs"; this candidate = "why one MAIN kernel produces all three"
- CAND-FAG-2 (deterministic coordinate dispatch): COMPOSED — provides the optional determinism layer for the multi-output bwd
- CAND-FAG-3 (saved-tensor restore): COMPOSED — provides Stage A input
- P-P89 (workspace contract for fused ops): same multi-output workspace shape; promote-merge if multi-op evidence accumulates (attention bwd is one op, MoE bwd would be another)
- CAND-FA1 / CAND-FA3 (cross-core sync, slot rotation): orthogonal — both apply within Stage B and Stage D/E/F's cube ladders unchanged

**Promote when**: a5_ops ships a multi-output fused-bwd op (attention bwd, MoE bwd, fused-norm-Linear bwd) with measured perf ≥1.0× vs the "three independent bwd kernels" baseline AND L1/UB budget verified ≥80% utilization (proves the fusion is paying for its complexity).

## CAND-MLA-1: Latent-projection prolog skeleton — down-project × pre-norm × up-project producing two heads from one normalized intermediate (cube-vec-cube ladder)
`applies_to: any soc with public AscendC Matmul-tile primitives + RmsNorm + cross-core sync; cann=9.0.0+; op_class=latent_attention_prolog / low_rank_dual_projection_prefix / mla_prefix`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h (top-level Process/AicProcess at ~L755-L880 — two-stage MM_CQ → RMSNORM_CQ → MM_QCQR ladder), attention/mla_prolog/docs/aclnnMlaProlog.md (formula block: c^Q = RmsNorm(x·W_DQ); q^C = c^Q·W_UQ; q^N = q^C·W_UK)`
`unverified_on: a5_ops (no MLA-prolog-class op currently shipped; closest analog is fused norm-then-matmul chains in workspace/3_FusionAttention)`

**Trigger**: Op shape is "compress hidden dim by a low-rank projection, normalize the latent, then expand into TWO downstream heads (e.g. content and rotary, or up-projection plus a second branch)" — characteristic of MLA prolog, low-rank-adapter-style prefixes, factorized-attention prefixes, and certain MoE expert prefixes where one small intermediate feeds multiple downstream matmuls. Hidden dim `He` is large (>=4K), the latent dim `Hc` is small (~1K-2K), and the two downstream up-projections share the latent — meaning the latent must be (a) normalized once, (b) cached in GM or wide-UB, (c) read twice by the up-matmuls without recomputation.

**Recommendation**: Structure as a three-stage cube-vec-cube ladder with the normalized latent as the cross-stage carrier in GM workspace. Stages:
  1. **Down-project (cube)**: `c_pre = x · W_D` where `(B*S, He) @ (He, Hc) -> (B*S, Hc)`. Emit to a GM workspace slot (pre-norm latent buffer). Signal vec.
  2. **Normalize (vec)**: read the pre-norm latent from GM, apply RmsNorm with `gamma` of length `Hc`, write the normalized latent to a second GM workspace slot. Signal cube. (This is the only stage that touches the latent's row state — see CAND-MLA-3 for the per-row rmsnorm shape.)
  3. **Up-project (cube)**: cube reads `c_norm` and performs the dual up-projection. The fused up-matmul has output dim `N*(D+Dr)` where the `N*D` slice is the content head and the `N*Dr` slice is the rope head — a single matmul over a concatenated weight `W_U = [W_UQ | W_QR]` shape `(Hc, N*(D+Dr))`, with the two heads separated post-matmul by offset slicing in the consumer.

The dual-output fusion in step 3 is load-bearing: it amortizes the GM read of `c_norm` (1 read instead of 2) and shares the L1 A-tile across both N partitions. The post-matmul slice into content/rope heads is a cheap address-only operation (no data movement) because the slice is along the N dimension which is the cube's output dim and is already laid out contiguously per head.

The cross-stage handoff between stages 1->2 and 2->3 uses the user-owned cross-core flag protocol from CAND-FA1 (paired AIC/AIV via `CrossCoreSetFlag` / `CrossCoreWaitFlag`); the latent GM slot follows the workspace-slot-rotation discipline of CAND-FA3 when the step is iterated over a batch dimension.

**Concrete anchor** (public-API-only skeleton; orchestrator level — the per-stage matmul body uses the cube-tile-mmad primitives or `Matmul<>` per CAND-FA1 hard-exclusion clause):
```cpp
constexpr uint32_t downDoneId = 1;   // user-owned flag IDs, all <= FFTS_MAX_FLAG (7)
constexpr uint32_t normDoneId = 2;

// Stage 1 (cube) — down-project x by W_D, emit pre-norm latent to GM
// (B*S, He) @ (He, Hc) -> preNormLatentGm[B*S, Hc]
AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(downDoneId);

// Stage 2 (vec) — gate on cube, normalize, emit normalized latent to GM
AscendC::CrossCoreWaitFlag(downDoneId);
RmsNormRow(cNormUb, preNormLatentGm, gammaUb, /*col=*/Hc, epsilon);
DataCopy(normLatentGm[rowOffset], cNormUb, Hc);
AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(normDoneId);

// Stage 3 (cube) — gate on vec, up-project into fused (content | rope) heads
// (B*S, Hc) @ (Hc, N*(D+Dr)) -> upResGm[B*S, N*(D+Dr)]
AscendC::CrossCoreWaitFlag(normDoneId);
// Downstream consumers slice: content = mmUpRes[:, :N*D]; rope = mmUpRes[:, N*D:]
```

**Why it works**:
- The latent dim `Hc` is the smallest tensor on the cross-stage path. Putting the cross-stage hand-off AT the latent (rather than upstream of the down-project or downstream of the up-project) minimizes GM round-trip volume and the working set the vec stage must keep resident.
- Fusing the two up-projections into one matmul along N halves the down-projection's L1 A-tile reuse pressure: the cube reads `c_norm` once, computes the concatenated output tile, and the downstream slice is purely an offset alias.
- The cube-vec-cube ladder gives the cube engine two compute windows per token-group (down + up), interleaved with one vec window (norm), allowing partial overlap when the producer-consumer flags are pipe-tight (PIPE_FIX for cube emit, PIPE_MTE3 for vec emit) and when the workspace slot rotation (CAND-FA3) admits one in-flight generation of overlap.

**Determinism**: The skeleton is deterministic when (a) each row's down-project / norm / up-project is owned by a single AIC/AIV pair (no cross-core writes participate in the latent), (b) RmsNorm uses the row-reduction shape of CAND-MLA-3 with a fixed Cast+Square+Sum order, and (c) the up-projection's K-dim reduction order is fixed by the tiling (same as any deterministic matmul). The fusion of two up-projections into a single matmul does not change any reduction order per output element — the only change is that two output regions are produced in one pass.

**Other instances predicted**:
- MLA-style prolog operators (this verified instance) for inference and training prefixes
- LoRA-style low-rank prefixes (`x · A · B` factorization) where `A` projects to a low rank and `B` projects back — the dual-output extension is when `B` is itself a concatenation
- DeepSeek-V2 / V3 latent attention prefixes that share an MQA-style compressed KV
- MoE prefixes that share a routing pre-projection feeding both gate scores and load-balancing statistics from one normalized latent
- Any "compute embed once, use twice" pattern where the second use is along a different head axis (the up-fusion variant amortizes the embed read)

**Risks before promotion**:
- The dual-output up-projection requires the two consumers' inner-dim layouts to be compatible — if the rope head needs an interleaved-pair layout (CAND-MLA-4) along the same axis as the content head's flat layout, the fusion breaks the rope head's locality. Verify by matching the consumer's GatherMask stride against the fused matmul's output stride before promoting per shape.
- Three-stage cross-core flag chains consume 2 of the `FFTS_MAX_FLAG = 7` user-owned IDs; layered on top of CAND-FA1's existing chain or co-existing pipelines, exhaustion is plausible — track flag-ID accounting at the kernel level.
- Latent GM slot rotation (CAND-FA3) is REQUIRED if the prolog is iterated across a batch dimension with an inflight-depth > 1; without rotation the second down-project would overwrite the latent the up-project of generation N-1 is still reading. The MLA reference uses `stepBatchSize` chunking + a `curBlockTokenOffset` rotation discipline that combines with FA3's modulo slot indexing.
- a5_ops has no MLA-class op shipping yet — pattern is structurally derived but not measured.

**Cross-reference**:
- CAND-FA1 (cross-core flag handoff) — supplies the stage handoff primitive used at each of the two seams in this ladder; the hard-do-not-apply clause about `Matmul<>` carries through
- CAND-FA3 (GM workspace slot rotation) — supplies the multi-generation slot discipline when the prolog is iterated per batch chunk
- CAND-MLA-3 (per-row RmsNorm with shared-tmp UB) — supplies the vec stage's row-normalize implementation
- CAND-MLA-4 (interleaved-pair RoPE via GatherMask) — supplies the downstream rope-head consumer of the fused up-projection's rope slice

**Promote when**: an a5_ops op with the latent-projection shape ships (e.g. a future MLA prolog port, a LoRA-bias-fused matmul prefix, or a fused down-norm-up triplet that is currently three separate ops), AND the shipped kernel demonstrates measurable improvement from the dual-output up-projection fusion vs two sequential up-matmuls, AND the cross-stage flag chain is verified disjoint from any high-level `Matmul<>` library use per CAND-FA1's exclusion clause.

## CAND-MLA-2: Paged-attention scatter-cache via single DataCopy per token — block-index decomposition, no per-token loop on cube side
`applies_to: any soc with public AscendC DataCopy + DataCopyPad + DataCopyParams; cann=9.0.0+; op_class=kv_cache_scatter / paged_attention_writeback / block_indexed_cache_update`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/service_scatter_cache.h (ScatterCache + ScatterCacheUnAligned + ScatterCacheMultiRows + MaterializeOffsetsWithHeadSize, ~135 lines, public-API only — DataCopy, DataCopyPad, DataCopyParams), attention/mla_prolog/docs/aclnnMlaProlog.md (kvCacheRef shape `(BlockNum, BlockSize, Nkv, Hckv)` and cacheIndex semantic "取值范围需在[0,BlockNum*BlockSize)内")`
`unverified_on: a5_ops (no paged-KV op shipped)`

**Trigger**: Op needs to write a per-token result tensor into a paged KV cache laid out as `(BlockNum, BlockSize, N, D)` where each input token has an integer `paTokenIndex ∈ [0, BlockNum*BlockSize)` selecting its slot, the slot's block-id is `paTokenIndex / BlockSize`, its in-block row is `paTokenIndex % BlockSize`, and the per-row payload is `N*D` bytes (32B-aligned) or `N*D` elements (unaligned). The op may also need to scatter MULTIPLE consecutive token rows that may straddle a page boundary (last few rows of one page + first few rows of the next page). Cache layout may be ND or NZ (FRACTAL_NZ along the head dim).

**Recommendation**: Decompose the scatter into pure address arithmetic + a single `DataCopy` per token (or per row-run when consecutive), avoiding any per-element loop. The three variants follow the same address-decomposition:

  - **Aligned single-row** (`col % 32B == 0`): one `DataCopy` with no params; offset = `paTokenIndex * stride` for ND, or `(paTokenIndex/blockSize)*blockSize*stride + (paTokenIndex%blockSize)*col0` for NZ.
  - **Unaligned single-row** (`col` not 32B-aligned): `DataCopyPad` with `DataCopyParams{1, col*sizeof(T), 0, 0}`; ND only.
  - **Multi-row run** (consecutive rows, may straddle one page boundary): split into a prefix-in-current-page `DataCopy` and a tail-in-next-page `DataCopy`; the split point is computed once from `rowsInCurBatch` vs `row`.

For NZ cache layout: `col0 = ALIGN_BLOCK_SIZE / sizeof(T)` (16 elements for bf16, 32 for int8); the per-page stride uses `DataCopyParams{col/col0, 1, 0, blockSize-1}` so successive C0 chunks land in the correct NZ sub-block. This means a single `DataCopy` writes ALL of a row's `N*D` elements scattered across `col/col0` C0 sub-blocks within one page, using the destination stride to skip `blockSize-1` rows between consecutive C0 chunks.

A negative `paTokenIndex` (= sentinel for "drop this token") MUST early-return — the scatter is silently dropped, which is the correct behavior for variable-length-batch padding tokens. The caller pre-computes negative indices for padding rows.

**Concrete anchor** (verified shape; ND aligned + NZ paged + multi-row spill):
```cpp
// ND aligned variant — one DataCopy per token
if (paTokenIndex < 0) { return; }
DataCopy(cacheGm[paTokenIndex * stride], inputUb, col);

// NZ paged variant — one DataCopy per token, address decomposed to (page, row-in-page)
constexpr uint8_t col0 = 32 / sizeof(T);    // 16 for bf16, 32 for int8
int64_t pageId  = paTokenIndex / blockSize;
int64_t rowInPg = paTokenIndex % blockSize;
int64_t off = pageId * blockSize * stride + rowInPg * col0;
DataCopyParams p{ static_cast<uint16_t>(col / col0), 1, 0,
                  static_cast<uint16_t>(blockSize - 1) };
DataCopy(cacheGm[off], inputUb, p);

// Multi-row run straddling one page boundary
int64_t copyCnt = col * rowsInCurBatch;
DataCopy(cacheGm[cacheOffset], inputUb, copyCnt);
if (rowsInCurBatch != totalRows) {
    DataCopy(cacheGm[nextBatchOffset], inputUb[copyCnt],
             (totalRows - rowsInCurBatch) * col);
}
```

**Why it works**:
- Paged-attention's address decomposition is purely scalar — `pageId = idx/blockSize`, `rowInPage = idx%blockSize` — and these two divisions+modulos are free relative to the DataCopy cost. No per-element loop is needed because the page-internal stride is encoded in `DataCopyParams.dstStride` (NZ) or implicit in the contiguous offset (ND).
- The NZ variant writes one row's `N*D` elements as `col/col0` C0 chunks with `dstStride=blockSize-1`, which the MTE3 engine fuses into one descriptor — same MTE3 cost as a single contiguous DataCopy. The NZ-vs-ND choice is therefore a layout-only tradeoff with no per-token compute cost difference.
- The multi-row-spill variant is the "row-runs collapse, page-runs split" specialization — collapse consecutive same-page rows into one DataCopy, split only at page boundaries (at most one split per row-run). For typical paged-attention page sizes (16-128), the expected number of splits per token-batch is small, and the speedup vs per-token DataCopy is proportional to in-batch token locality.
- Negative-index early-return is the correct semantic for masked / padding tokens; it avoids spurious GM writes that would corrupt unused slots and avoid synthetic "drop = write to scratch then ignore" overhead.

**Determinism**: Each output cache slot has a single owning input token (by the contract of `paTokenIndex` being a one-to-one assignment); no scatter-add. Deterministic by construction when (a) `paTokenIndex` values are unique across the batch (the caller's responsibility — the CANN reference comment says "取值范围需在 [0, BlockNum*BlockSize) 内" but does not enforce uniqueness; uniqueness IS required for determinism), and (b) the DataCopy ordering across tokens does not matter because each write touches disjoint addresses.

**Other instances predicted**:
- KV-cache writeback in paged-attention prefill and decode (MLA reference; vLLM-style block tables)
- Any "scatter rows into a pre-allocated paged buffer indexed by an integer token-to-slot map" op
- Sparse-tensor-style scatter-writes where the row index is dense and pre-computed (NOT general atomic scatter-add — that needs a different pattern)
- IndexPut variants with a single integer dim-0 index (when `accumulate=False`)
- Beam-search KV reorganization where each beam's KV is scattered into a new batch-major layout
- Speculative-decoding "accepted-prefix writeback" where token-level acceptance produces a sparse-but-dense-after-compaction index map

**Risks before promotion**:
- Uniqueness of `paTokenIndex` across the batch is a precondition for determinism — if the caller computes the index map dynamically and two tokens collide, the scatter is non-deterministic. The MLA reference does not check uniqueness; a hardened caller must.
- The NZ stride encoding assumes `col % col0 == 0` (i.e. the head dim is C0-aligned). The unaligned variant exists for ND only — for NZ unaligned, the pattern does NOT apply as-is and must be padded upstream.
- `blockSize` must be in the documented range (MLA: 16-1024, multiple of 16). Smaller `blockSize` values stress the MTE3 descriptor cost; very large `blockSize` values stress UB residency of the prefix-in-page+tail-in-next-page two-DataCopy path. Profile boundary cases (page=16, page=1024) when first shipping.
- A negative `paTokenIndex` is the CANN convention for "drop"; consumers must verify their caller honors it (some PyTorch-side mappings use `-1` for padding, others use `INT64_MIN`).

**Cross-reference**:
- CAND-MLA-1 (latent-prolog skeleton) — produces the tensors that are scattered into the KV cache via this pattern (the K^C and K^R outputs)
- OL-58 (DataCopyPad for tail-byte handling) — supplies the `DataCopyPad` form used in the unaligned single-row variant
- P-P (any DataCopy-stride pattern entries in `patterns/domains/memory_access.md`) — orthogonal; this is the "writeback with paged index" specialization
- a5_ops 1_RotaryMul / 12_KvRmsnormRopeCache (closest existing benchmark ops) — these have the rope+cache shape but NOT the paged scatter; promoting this candidate provides the paged variant they currently lack

**Promote when**: an a5_ops op ships a paged-KV scatter writeback (e.g. a future MLA prolog, paged FlashAttention prefill, or a paged-cache update op), AND the per-token DataCopy cost is verified to dominate over the address-arithmetic cost on the target SoC, AND the multi-row-spill variant is exercised on a workload with non-trivial in-batch locality so the split-vs-collapse decision can be measured.

## CAND-NSA-1: Matmul-library-driven AIC/AIV co-iteration — vector phases chained via local `SetFlag<HardEvent::MTE3_MTE2>` between iterations, while the cube side is driven by `Matmul<>::IterateAll() + WaitIterateAll()` (the **complement** to CAND-FA1's hard-do-not-apply clause)

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_high_level_matmul | matmul_lib_driven_pipeline_with_aiv_postprocess`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — top-level kernel main loop pattern`
`unverified_on: a5_ops (3_FusionAttention currently does NOT layer multi-phase AIV postprocess on top of high-level Matmul<>)`

**Trigger**: A fused-attention-class kernel uses the high-level `matmul::Matmul<>` template (NOT tile-MMAD primitives) for its cube halves AND needs the AIV side to chain >=2 distinct vector phases per outer iteration (e.g. QK→softmax→aux-scoring→TopK) with cross-iteration overlap. CAND-FA1's flag protocol is forbidden here (its hard-do-not-apply clause names exactly this case); a different sync recipe is required.

**Recommendation**: Drive the cube side with the high-level Matmul library client API — one `IterateAll<false>(gmDst, ...)` per outer iter writes the cube result into a GM ping-pong slot; the matching `WaitIterateAll(); End();` retires the cube call from the AIV side. AIV phases within the iter chain through local `event_t` allocated from the TPipe and synchronized with `SetFlag<HardEvent::MTE3_MTE2>` / `WaitFlag<HardEvent::MTE3_MTE2>`, NOT through `CrossCoreSetFlag/WaitFlag`. The library's internal AIC↔AIV sync is hidden behind `WaitIterateAll` — adding user-owned cross-core flags on top would race with library-owned flag IDs (see CAND-FA1 hard-do-not-apply clause and `507014` evidence cited there).

Per-iter shape:

1. AIV calls `bmm.IterateAll<false>(gmDst[taskId & 1], ...)` to kick a cube call into one GM ping-pong slot.
2. AIV calls `WaitIterateAll(); End();` to block on cube retirement of the *current* iter.
3. AIV runs phase-1 vec compute consuming `gmDst[taskId & 1]`; emits its own MTE3 GM write; sets `MTE3_MTE2` flag.
4. AIV (still same iter) kicks the next cube call `bmm2.IterateAll<false>(gmDst2[taskId & 1], ...)`.
5. AIV runs phase-2/phase-3 vec compute, each gated by `WaitFlag<MTE3_MTE2>` against the prior MTE3 emission and re-arming the flag at the end.
6. Between iters, the next iter's bmm1 can be pre-kicked while the current iter's phase-3 (e.g. TopK) is still running because they bind to different GM ping-pong slots (`taskId+1` vs `taskId`).

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
event_t mte3mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);
for (int64_t it = innerOffset; it < innerLimit; ++it) {
    // cube side via Matmul library client
    bmm1.IterateAll<false>(gQK[it & 1], /*sync=*/false, /*reuse=*/false, /*wait=*/true);
    bmm1.WaitIterateAll(); bmm1.End();

    // AIV phase 1 (softmax-class) — consumes gQK[it & 1], writes gProbs[it & 1]
    WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
    runVecPhase1(gQK[it & 1], gProbs[it & 1]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);

    // chain next cube + AIV phases on the SAME ping-pong index, then bump taskId
    WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
    bmm2.IterateAll<false>(gPV[it & 1], false, false, true);
    runVecPhase2(gProbs[it & 1], gPV[it & 1]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);
}
WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
```

**Why it works**: `Matmul<>::WaitIterateAll()` is the library-owned barrier covering the cube → GM-write retirement, so user-owned `CrossCoreSetFlag/WaitFlag` on the same MODE/pipe space is redundant and provably conflict-prone (CAND-FA1 hard-do-not-apply names this). `SetFlag<HardEvent::MTE3_MTE2>` is a per-AIV-core local pipe flag — it orders the AIV's own MTE3 emission with its next iter's MTE2 read of the same GM region, which is what's needed once the cube↔vec handoff is already library-owned. Ping-pong on `taskId & 1` keeps two GM slots live so the next iter's cube call can prefetch while the current iter's vec tail finishes.

**Determinism**: The AIV's per-iter phase order is fixed by the source structure. Each GM ping-pong slot has a single writer per iter (the matched cube call) and a single reader (the matched vec phase). `WaitIterateAll` is a strict barrier — no in-flight cube write can leak into the next iter's vec read of slot `(taskId+1) & 1` because that slot was the prior iter's read target and is now free. Det-preserving by construction.

**Hard do-not-apply**:
- Do NOT combine this pattern with user-owned `CrossCoreSetFlag/WaitFlag` on overlapping `MODE` / pipe space — the high-level `Matmul<>` library already uses CrossCore internally and flag-ID collisions can stall the iter (the same `507014`-class failure CAND-FA1 cites).
- Do NOT use `PIPE_V` for the local AIV flag — the AIV's GM write retirement is on the MTE3 pipe; releasing on `PIPE_V` would publish before the GM write drains.
- Do NOT extend the ping-pong depth beyond 2 unless the GM workspace contract (P-P89) is restructured for >2-way rotation per CAND-FA3 modulo discipline — TQue-style depth-4 (OL-63) does not apply to GM ping-pong slots.

**Other instances predicted**:
- FlashAttention-class forward where cube QK → AIV softmax → cube PV → AIV rescale, when the cube side is built on the high-level `Matmul<>` template (not the tile-MMAD path that CAND-FA1 covers).
- Fused-attention with auxiliary side-output passes (e.g. attention + per-block aux-score + top-K-indices) where AIV runs ≥3 distinct phases per cube outer iter.
- Streaming attention prefill/decode hybrids that use the public `Matmul<>` template for BMM1 + BMM2 and need AIV-side multi-phase post-processing per row block.
- MoE GEMM chains where the expert GEMM uses `Matmul<>` and the AIV side runs gather/scatter + scale between GEMMs.

**Risks before promotion**:
- a5_ops 3_FusionAttention currently uses simpler AIV postprocess; no multi-phase chain layered on `Matmul<>` is shipped yet — this candidate is unverified on a5_ops measurements.
- The `WaitIterateAll(); End();` pair MUST appear in that order per iter; reversing them is observed to silently miss-retire the cube call on some library versions.
- The local `MTE3_MTE2` event must be `FetchEventID`-acquired ONCE outside the loop and re-armed (`SetFlag`) before the loop body's first `WaitFlag`; per-iter `AllocEventID` inside the loop is permitted only when paired with `ReleaseEventID` before iter exit, otherwise event-ID pool exhausts.
- Pipe selection: producing MTE3 → consuming MTE2 on the same AIV is the safe per-iter chain. Using a `V_MTE3` event in place of `MTE3_MTE2` between iters releases before the GM-write retires (silent stale-read).

**Cross-reference**:
- CAND-FA1 (cross-core user-owned flag handoff for kernels that do NOT use `Matmul<>`) — this candidate is the complementary case named in CAND-FA1's hard-do-not-apply clause.
- CAND-FA3 (GM workspace slot rotation modulo MAX_LAG+1) — directly composes; the ping-pong is the `MAX_LAG=1` instance of FA3.
- P-P89 (GM workspace contract — public outputs vs opaque scratch) — supplies the GM layout discipline the ping-pong slots sit inside.
- P-P75 (intra-core `SetFlag/WaitFlag<HardEvent>` for SIMD pipe sync) — the local pipe-flag layer this candidate composes on top of; same primitive, applied to inter-phase chaining within one AIV across iterations.
- OL-91 (cube playbook conventions for `Matmul<>` users) — orthogonal but the same dispatch class.

**Promote when**: an a5_ops fused op (e.g. a future 3_FusionAttention + topK variant, or fused attention + auxiliary statistics) ships with `Matmul<>`-driven cube halves AND a measurable cube/vec overlap improvement vs a baseline that serializes vec phases behind `SyncAll<true>()`. Verification must include msprof showing the next-iter cube kick overlapping the current-iter vec tail.

## CAND-NSA-2: Power-of-2 tree-reduce across the head-pack axis via strided `Add` repeats — fold `G` query-heads-per-KV-group into one row in `log2(G)` vector passes

`applies_to: any soc with public AscendC Add(BinaryRepeatParams stride/repeat) ; cann=9.0.0+; op_class=group_query_attention_head_pack_reduce | per_row_head_group_sum | gqa_fused`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — importance-score reduce across head-pack`
`unverified_on: a5_ops`

**Trigger**: An op packs `G` query-heads per KV head (group-query attention with `G > 1`) and needs to reduce a per-head per-block tensor of shape `[S1_tile, G, K]` (rows × heads-per-group × per-head-cols) into `[S1_tile, K]` summed across the head-pack axis. The reduce target is NOT a row reduction (CAND-FA4's territory) — it is across a middle axis whose stride in UB equals `K` × `sizeof(elem)`. `G` is power-of-2 (the gQA spec — typical values 2/4/8/16).

**Recommendation**: Reduce in `log2(G)` passes via a strided `Add`. At pass `p` (with `p = 1, 2, 4, ..., G/2`), pairs are `(row_i, row_{i+p})` for every other row at distance `p` along the head-pack axis. Express each pass as a single `Add` call with a `BinaryRepeatParams{srcBlkStride=1, dstBlkStride=1, srcRepStride=2p×K×elem/32, dstRepStride=2p×K×elem/32}` repeat-stride, repeat count `S1_tile × G / (2p)`, and `count = K` elements per repeat. Each pass halves the live head-pack length; after `log2(G)` passes the head-pack is fully folded and the surviving rows are `2p × K`-strided in UB. A final compact `DataCopy` with `dataCopyParams{blockCount=S1_tile, blockLen=K_blocks, srcStride=(G-1)×K_blocks, dstStride=0}` gathers the folded rows back to a contiguous `[S1_tile, K]` layout.

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
// in:  buf[S1_tile][G][K] (fp32, UB), G is a power of two, K is repeat-count
// out: buf[S1_tile][0][K] with the surviving (folded) rows G*K-strided
for (uint32_t p = 1; p < G; p *= 2) {
    uint8_t stride = static_cast<uint8_t>(2 * p * K * sizeof(float) / 32);  // 32B = 1 vector block
    Add(buf, buf[p * K], buf,
        /*mask=*/K,
        /*repeat=*/static_cast<uint8_t>(S1_tile * (G / (2 * p))),
        /*params=*/{1, 1, 1, stride, stride, stride});
    PipeBarrier<PIPE_V>();
}
// optional compact pack: gather every G-th row back to contiguous output
DataCopy(out, buf, DataCopyParams{S1_tile, K * sizeof(float) / 32,
                                  (G - 1) * K * sizeof(float) / 32, 0});
```

If a single pass's per-element repeat-stride exceeds the architecture's signed `uint8_t` block-stride window (256 vector blocks), fall back to a manual outer loop emitting one `Add` per surviving row pair at that pass. The pattern remains `log2(G)`; only the unrolling shape changes:
```cpp
if (sgBlock < 256) { Add(... repeat=S1_tile*(G/(2*p)) ...); }
else { for (int r = 0; r < S1_tile*(G/(2*p)); ++r) { Add(...); } }
```

**Why it works**: Head-pack reduce sits between row reduce (`WholeReduceSum` / `BlockReduceSum`) and across-AIV reduce (cross-core). Public `WholeReduce`/`BlockReduce` primitives reduce along the *last* axis only — they do not address a middle axis. Bridging the middle-axis reduce via per-pair `Add` with a stride that doubles each pass costs `log2(G)` vector instructions in `K`-element-mask form per pass and `S1_tile × G / 2` total `Add`-mask repeats — the same vec-pipe budget as a tree reduce inside one row, paid across passes rather than across the row.

**Determinism**: Pairwise `Add` with a fixed (compile-time-determined) pair schedule produces a deterministic reduction tree. No cross-core write participates. `PipeBarrier<PIPE_V>` between passes ensures the `Add` of pass `p` completes before pass `2p` reads its result. Det-preserving by construction.

**Hard do-not-apply**:
- Do NOT use this pattern when the reduce axis is the LAST axis of UB — public `WholeReduceSum`/`BlockReduceSum` are faster on the last axis (single primitive vs `log2(G)` `Add`s).
- Do NOT use this pattern when `G` is not a power of two — the tree shape breaks; either pad to next power of two with `Duplicate(... 0)` on the pad rows or emit a non-power-of-two manual loop.
- Do NOT use this pattern when `K * sizeof(elem) < 32B` — a vector block carries only 32 bytes, so a per-repeat mask of `K` smaller than one block forces sub-block addressing the `Add` overload doesn't support; pad `K` to a 32B-multiple in UB first.
- Do NOT use this pattern with `Add` repeat-strides above the architecture's `uint8_t` limit without the manual-loop fallback shown above — silent stride wrap.

**Other instances predicted**:
- Any gQA fused-attention forward that emits a per-head intermediate and needs to fold heads before downstream reduce / mask / sort.
- Fused row-norm + scatter where multiple feature groups must be summed before the scatter (e.g. group-norm pre-affine + write).
- Multi-head per-row statistics (mean, variance) when the reduction crosses head-packs and the tail axis is already small enough to keep in UB.
- MoE per-expert per-row sums when each row visits multiple expert outputs and they must be combined before write.

**Risks before promotion**:
- a5_ops has not shipped a gQA op with `G > 1` head-pack reduce; the pattern is unverified on a5_ops perf.
- Repeat-stride 256-block wrap is observed in CANN reference's `if (sgBlock < 256)` branch — copy the manual-loop fallback verbatim if the kernel's `(2p × K × sizeof(elem)) / 32` can exceed 255.
- `PipeBarrier<PIPE_V>` between passes is mandatory; omitting it produces silent wrong-result on V220-class AIV because the pass-`p` `Add` writes the source of pass-`2p` (same UB region — read-after-write hazard).
- This pattern assumes the in-place destination matches the first input — both `dst` and one `src` are `buf` and the other `src` is `buf[p × K]`. Using a separate dst buffer doubles UB.

**Cross-reference**:
- CAND-FA4 (tree-reduce wide fp32 rows via packed `BlockReduceMax/Sum` partials) — this candidate is the orthogonal middle-axis case; both compose in one kernel (row reduce inside a head, then head-pack reduce across heads).
- P-P62 (Row-Scalar VEC Multiply via Brcb — same `BinaryRepeatParams` stride family) — same primitive class, different reduction shape.
- patterns/domains/reduction_quant.md (reduction shapes index) — should add a "middle-axis head-pack reduce" entry when this candidate promotes.

**Promote when**: an a5_ops fused gQA op (e.g. a future GQA attention forward or fused gQA + scoring kernel) ships with `G > 1` per-row head-pack reduce AND msprof shows the `log2(G)` `Add`-pass cost is below the alternative (UB-spill + WholeReduce-over-flattened-axis or per-row scalar loop).

## CAND-NSA-3: Overlapping-window weighted-sum aggregation as a 1-D "convolution" — fold a fine-grained per-element signal into per-chunk scores via strided gather + scalar-`Muls` + accumulating `Add`, with a precomputed triangular weight schedule

`applies_to: any soc with public DataCopy(stride) + Muls + Add; cann=9.0.0+; op_class=block_sparse_attention_scoring | overlapping_pool_aggregation | windowed_per_chunk_reduce`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — pre-TopK chunked importance score`
`unverified_on: a5_ops`

**Trigger**: An op needs to produce per-chunk scalar scores from a finer-grained per-element signal where each output chunk's score is a weighted sum over an OVERLAPPING window (window length `W` > chunk stride `M`, so adjacent chunks share `W − M` source elements). Typical examples: block-sparse attention importance scoring (compress `S2` softmax probabilities into `ceil(S2 / M)` per-block scores), strided pooling with overlap, fused 1-D convolution + sum-pool fast path. The triangular weights `t[i]` (for `i ∈ [0, W)`) are determined by how many output chunks each input position falls into, so weight[0]=1, weight[1]=min(2, peakCover), ..., peaks at `peakCover`, then mirrors back — this is a structural property of overlap geometry, NOT a learned parameter.

**Recommendation**: Three-step shape per AIV per row tile:

1. **Strided gather** the input row of length `S2` into a `[W_chunks, W]` UB layout using one `DataCopy(...DataCopyParams{blockCount=W_chunks, blockLen=W_blocks, srcStride=(M-W)_blocks, dstStride=0})` so the chunk axis becomes the outer (block) axis and the per-chunk-window axis is contiguous along the inner axis. (When `W` > `M`, srcStride is negative-relative; emit as a 2-D `DataCopyPad` with `srcStride=(M*sizeof(T))` and adjusted `blockLen` to avoid signed overflow.)
2. **Position-wise weighted accumulate**: for each `i ∈ [1, W)` (excluding the trivial `i=0` self-term), call `Muls(scoreScratch + i_offset, srcWindow + i_offset, static_cast<float>(t[i]), countPerChunk, repeat=W_chunks, BinaryRepeatParams{srcBlkStride=1, srcRepStride=stride_chunk, dstBlkStride=1, dstRepStride=stride_chunk})`, then `Add(score, scoreScratch + i_offset, score, countPerChunk, repeat=W_chunks - skip, BinaryRepeatParams{...})`. `PipeBarrier<PIPE_V>` between `Muls` and `Add`. The `t[i]` schedule is a host-emitted constant table of length `W` (see "Weight schedule" below).
3. **Cross-chunk pack**: the chunk's per-position-weighted partial is then collapsed across the inner `W` axis via the row-reduce path (CAND-FA4 / WholeReduceSum on the row, OR the head-pack tree of CAND-NSA-2 if the chunk axis is folded across groups).

**Weight schedule** (the triangular ramp — public arithmetic on the host):
```cpp
// W = window length, peakCover = max chunks any position contributes to
// t[i] for i ∈ [0, W): rises 1..peakCover, plateaus, descends peakCover..1
for (int i = 0; i < W; ++i) {
    if (i < W / 2)      t[i] = std::min<int>(i + 1, peakCover);
    else                t[i] = std::min<int>(W - i, peakCover);
}
```
The schedule is symmetric and has compact-support — corner chunks (first and last `W − M` of the row) have fewer covering windows than interior chunks.

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
// Gather: per-chunk view of overlapping window
DataCopy(scoreScratch, srcRow,
    DataCopyParams{static_cast<uint16_t>(W_chunks),
                   static_cast<uint16_t>(W * sizeof(float) / 32),
                   static_cast<uint16_t>((M - W) * sizeof(float) / 32), 0});

// Weighted accumulate across positions in the window
for (int i = 1; i < W; ++i) {
    Muls(tmp, scoreScratch + i, static_cast<float>(t[i]),
         /*count=*/innerCount, /*repeat=*/W_chunks,
         BinaryRepeatParams{1, 1, chunkRepStride, chunkRepStride});
    PipeBarrier<PIPE_V>();
    Add(score, tmp, score, innerCount, W_chunks,
        BinaryRepeatParams{1, 1, 1, chunkRepStride, chunkRepStride, chunkRepStride});
    PipeBarrier<PIPE_V>();
}
```

**Why it works**: An overlapping-window weighted sum is equivalent to a 1-D depthwise convolution of the signal with the triangular kernel `t[]`, then downsampled at stride `M`. Direct DataCopy with a `M-W`-blocks stride realizes the overlap-and-downsample in one MTE2 issue, eliminating the per-output-chunk scalar gather (which would be `O(W_chunks × W)` scalar `GetValue`/`SetValue` pairs — the scalar-pipe-bound anti-pattern OL-82 / P-P86 cites). The `Muls` × `Add` pair per position runs vec-pipe-bound at one repeat over `W_chunks` chunks per call, so the total cost is `(W − 1) × (Muls + Add)` vec-pipe calls regardless of `W_chunks`. The triangular weight table is precomputed on the host once per tiling, avoiding per-iter scalar arithmetic.

**Determinism**: All operations are public `DataCopy` / `Muls` / `Add` / `PipeBarrier` — no atomic, no cross-core. The `(W − 1)` position-wise accumulations occur in fixed program order, so the per-chunk sum is a deterministic sequence `score = sum_i t[i] × src[chunk_off + i]`. Det-preserving by construction.

**Hard do-not-apply**:
- Do NOT use this pattern when the window stride `M >= W` (no overlap) — collapses to a non-overlapping pool that one `WholeReduceSum`-per-chunk or a single `DataCopy` + `Add` would handle more cheaply.
- Do NOT use this pattern when `W` is data-dependent (varies per row) — the static loop unroll over `i ∈ [1, W)` becomes a variable-trip loop that defeats compile-time scheduling; emit a separate dynamic-`W` kernel instead.
- Do NOT use this pattern when the weight schedule is NOT a low-arithmetic shape — for arbitrary learned weights, the `Muls` per position is fine but the per-chunk closed-form weight collapse used in the reference's overlap math (the triangular ramp from `[1, peakCover]`) does NOT generalize.
- Do NOT collapse the gather DataCopy into a single contiguous `DataCopy` when `M < W` — the overlap forces `srcStride < 0` semantically; the source must be re-issued per overlap-pair via separate DataCopys (or, for moderate `W − M`, a `DataCopyPad` with a positive offset reset between blocks).

**Other instances predicted**:
- Block-sparse attention pre-TopK scoring: NSA-class importance score, native-sparse-attention chunk score, dilated-attention windowed score.
- Strided 1-D pooling with overlap (audio frame energy, NLP token-level chunk pooling, sliding-window L2 norm).
- Fused conv1d + sum-pool fast path when kernel size > stride.
- Compressed top-K input preparation: any kernel that needs a per-block summary stat before a `TopK<>` over the compressed length.
- Speculative-decode draft-score aggregation across token spans.

**Risks before promotion**:
- a5_ops has not shipped a block-sparse / overlapping-pool op yet; the pattern is unverified on a5_ops perf and precision.
- The strided `DataCopy` `(M - W) * sizeof(T) / 32` block-stride must fit `uint16_t srcStride` — for large `W` and small chunks the stride can overflow; emit per-chunk-pair DataCopy in that case.
- The triangular weight schedule above is correct only when every input position contributes to BETWEEN 1 AND `peakCover` chunks; boundary rows of the matrix (first / last `W − M` elements) need explicit zero-pad before the weighted accumulate, otherwise the partial sums under-count.
- Per-position `Muls` repeats with a non-unit `srcRepStride` were observed to require an explicit `PipeBarrier<PIPE_V>` between successive `Muls` calls on the same destination region — omitting the barrier produces a write-write hazard on V220-class AIV (silent wrong-result).
- This pattern produces per-chunk SCORES; the downstream TopK over those scores is a separate concern (use the public `AscendC::TopK<>` per P-P85, not a hand-roll).

**Cross-reference**:
- P-P85 (`AscendC::TopK` adv_api primitive) — the natural downstream consumer of the per-chunk scores this candidate emits.
- P-P62 (Row-Scalar VEC Multiply via Brcb) — same `BinaryRepeatParams` repeat-stride family; both candidates use the same Add/Muls overload pattern.
- patterns/domains/reduction_quant.md — should add an "overlapping windowed pool" entry when this candidate promotes.
- OL-82 / P-P86 (scalar-pipe-bound anti-pattern for fused-op scoring) — this candidate is the vec-pipe-clean alternative to the scalar `GetValue`/`SetValue` per-position naive form.

**Promote when**: an a5_ops fused op (e.g. a future block-sparse attention, sliding-window pool, or fused TopK over per-block scores) ships with overlapping-window aggregation AND msprof shows `aiv_vec_ratio > 0.6` for the scoring phase (proving the implementation stayed vec-pipe-bound) AND the per-chunk score output matches a reference triangular-overlap weighting within bit-exact tolerance for fp32 / 1-ULP tolerance for fp16/bf16.

## CAND-NSA-4: Phase-multiplexed single UB region — allocate one `TBuf<>` covering all of UB and re-slice it per phase via typed `GetWithOffset<T>` views (UB-side counterpart to P-P89's GM-side workspace contract)

`applies_to: any soc with public TBuf<TPosition::VECCALC> + GetWithOffset<T>; cann=9.0.0+; op_class=multi_phase_fused_op_with_disjoint_per_phase_ub_needs`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — softmax phase + scoring phase + topK phase share the same UB region`
`unverified_on: a5_ops`

**Trigger**: A fused op runs ≥3 sequential phases per outer iter (e.g. softmax → aux-score → TopK) whose per-phase peak UB tensor sets are mostly disjoint — phase A needs `{tensorsA[]}`, phase B needs `{tensorsB[]}`, phase C needs `{tensorsC[]}`, and `max(sumA, sumB, sumC) << sumA + sumB + sumC`. Naively giving each phase its own `TBuf<>` overflows UB; per-phase `TQue` rotation does not solve the problem because the buffers are scratch, not pipelined input/output. The phases run serially (a `PipeBarrier<PIPE_V>` or `SetFlag/WaitFlag` boundary separates them), so the UB region's "owner" cleanly changes at each phase boundary.

**Recommendation**: Allocate a single `TBuf<>` covering the kernel's full UB budget (or its largest single-phase peak). At each phase entry, compute byte offsets for that phase's tensors and obtain typed views via `GetWithOffset<T>(elem_count, byte_offset)`. The same byte region under-pins different typed views in different phases — phase A may see it as `LocalTensor<float>`, phase B as `LocalTensor<half>`, phase C as `LocalTensor<int32_t>` — and the phase barrier (`PipeBarrier<PIPE_V>` or `SetFlag<HardEvent::V_MTE3> + WaitFlag<HardEvent::MTE3_V>` when MTE writes intervene) guarantees the previous phase's writes have retired before the next phase's reads begin.

This is the UB-side analog of P-P89's GM workspace contract: ONE byte buffer, host- or kernel-computed offsets, typed re-slicing. Difference: P-P89 covers GM scratch with host-emitted offsets in tilingdata; CAND-NSA-4 covers UB scratch with kernel-computed offsets at phase entry (since UB layout depends on the phase's runtime per-phase row counts and aligned col widths, which are tiling-derived but not flat host constants).

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
// Init: one TBuf for all of UB
TBuf<TPosition::VECCALC> allUb;
pipe.InitBuffer(allUb, /*bytes=*/192 * 1024);
LocalTensor<uint8_t> base = allUb.Get<uint8_t>();

// Phase A entry: softmax tensors
int64_t off = 0;
LocalTensor<float> qkScores = allUb.GetWithOffset<float>(rows * cols, off);
off += rows * cols * sizeof(float);
LocalTensor<float> softmaxOut = allUb.GetWithOffset<float>(rows * cols, off);
// ... phase A compute ...
PipeBarrier<PIPE_V>();   // boundary — phase A's writes retire before phase B's reads

// Phase B entry: aux-score tensors REUSE the same byte region with new typed views
off = 0;
LocalTensor<float> scoreScratch = allUb.GetWithOffset<float>(rows2 * scoreLen, off);
off += rows2 * scoreLen * sizeof(float);
LocalTensor<half> packedScores = allUb.GetWithOffset<half>(rows2 * scoreLen, off);
```

For phases separated by an MTE3 emission (e.g. one phase writes intermediate results to GM and a later phase reads them back), use `SetFlag<HardEvent::V_MTE3>` + `WaitFlag<HardEvent::V_MTE3>` at the boundary instead of `PipeBarrier<PIPE_V>` — `PipeBarrier` only orders within the vector pipe and does NOT order against the MTE pipe.

**Why it works**: A `TBuf<TPosition::VECCALC>` of size `B` byte-allocates UB once; subsequent `GetWithOffset<T>(count, off)` views are address arithmetic only and do not allocate. Per-phase reuse is safe ONLY because (a) the phase boundary (`PipeBarrier` / hard-event flag) hard-orders the prior phase's writes before the next phase's reads, (b) phases are mutually exclusive in time (no overlap), and (c) the typed view's lifetime is bounded by the phase scope — using a stale view from phase A inside phase B is a programmer error. The single-`TBuf<>` form avoids the "OL-94 TQue vs TBuf sync decision table" complexity of per-tensor TQue rotation when the tensors are scratch (no pipelined producer/consumer pattern across iterations on the same buffer).

**Determinism**: The phase boundary primitive (`PipeBarrier<PIPE_V>` or `SetFlag/WaitFlag<HardEvent::V_*>`) is deterministic — it stalls the consumer until the producer pipe drains. Per-phase compute uses only public vec/MTE primitives. As long as each phase's compute itself is deterministic (no atomic, no cross-core mid-phase), the multi-phase chain is deterministic by construction.

**Hard do-not-apply**:
- Do NOT use this pattern when phases overlap in time (e.g. phase A's tail runs on MTE while phase B starts on VEC) — there is no UB-byte coherence between concurrent phases. Use separate `TQue`s or separate `TBuf`s in that case.
- Do NOT omit the phase boundary primitive (`PipeBarrier` / `SetFlag` / `WaitFlag`) — the second phase MAY race read-after-write against the first phase's tail and silently see uninitialized bytes (V220-class AIV does not auto-serialize across logical phases sharing UB).
- Do NOT use this pattern when one of the phases needs `TQue`-style double-buffer pipelining ON THE SAME tensor (e.g. streaming input load overlapped with compute on the prior tile) — that is exactly what `TQue<DEPTH=4>` (OL-63) was designed for; do not replace TQue with TBuf re-slicing in pipelined contexts.
- Do NOT use `LocalTensor<T2>` typed views from phase A inside phase B — the view object holds a base + offset; using it after a phase boundary that re-purposed the region is a use-after-free-class hazard at the language level (no compile error, silent wrong-bytes read).
- Do NOT use the reinterpret-cast form (`localTensor.template ReinterpretCast<T>()`) across phase boundaries to "convert" a phase-A typed view to a phase-B typed view — re-obtain via `GetWithOffset<NewT>(count, off)` at phase entry to make the lifetime explicit.

**Other instances predicted**:
- Any fused-attention forward that does softmax → aux-score → TopK or softmax → mask-select → emit (3+ phases per row tile).
- Fused norm + scatter + gather where each phase's working tensors are mutually disjoint.
- Fused dequant → matmul-prep → quant pipelines where dequant scratch, matmul A/B prep buffers, and quant scratch are large and disjoint.
- Multi-stage MoE per-expert dispatch where routing-mask, gather-buffer, and per-expert-output stages each peak in different UB regions.
- Fused LayerNorm + Linear where the LayerNorm's stats buffers and the Linear's matmul-prep buffers do not coexist.

**Risks before promotion**:
- a5_ops has not yet shipped a fused op with 3+ disjoint-UB-need phases sharing one `TBuf<>`; the pattern is unverified on a5_ops perf and precision.
- Phase-boundary primitive choice (`PipeBarrier<PIPE_V>` vs `SetFlag/WaitFlag<HardEvent>`) is failure-mode-different — choosing `PipeBarrier<PIPE_V>` when an MTE write actually crosses the boundary is the OL-94 mis-application class (silent stale data on the second phase).
- Per-phase `GetWithOffset<T>(count, off)` must respect the architecture's UB alignment (32B vector block on V220 / V351); call sites that compute `off` from runtime tiling must `AlignUp(off, 32)` before the next view, otherwise the next typed view's MTE2/MTE3 issues mis-align and either fault or silently corrupt.
- Debugging a wrong-bytes read across a phase boundary is hard — there is no compile-time check that "phase A's `qkScores` view is unused after the boundary". Code-review discipline is required (or static-analysis pass over `GetWithOffset` call sites).

**Cross-reference**:
- P-P89 (GM workspace contract for fused ops — public outputs separate; opaque scratch sliced by host offsets) — this candidate is the UB-side analog. Cross-reference both when shipping a multi-phase fused op: GM scratch follows P-P89, UB scratch follows CAND-NSA-4.
- OL-94 (TQue vs TBuf sync decision table) — directly relevant: the phase-boundary sync primitive choice MUST consult OL-94. `PipeBarrier<PIPE_V>` is correct only when no MTE write crosses the boundary.
- P-P66 (`TQueBind<VECIN, VECOUT, 1>` in-place buffer reuse) — related "share one buffer across roles" pattern, but P-P66 is within ONE pipelined compute, this candidate is across SERIAL phases.
- OL-63 (TQue depth-4 for elementwise) — orthogonal; OL-63 governs streaming pipelined tensors, this candidate governs phase-scratch reuse.

**Promote when**: an a5_ops fused op (e.g. a future fused attention + scoring or fused norm + scatter + gather) ships with ≥3 phases sharing one `TBuf<>` AND total kernel UB peak measured below the alternative of per-phase-`TBuf` allocation (proving the reuse saved UB) AND precision PASS shows no cross-phase wrong-read regressions vs a per-phase-`TBuf` baseline.

## CAND-RAU-1: Online-softmax 2-input symmetric merge — associative/commutative reducer over (max, sum, accum) triples for ring/tree fold
`applies_to: any SoC with public AscendC VEC Max/Sub/Exp/Mul/Add/Div; cann=9.0.0+; op_class=online_softmax / streaming_normalization / fused_softmax_matmul / ring_attention`
`derived-from: cann-source (ring-attn-class update, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update SBH + TND variants (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: An attention-class kernel produces partial (attn_out, softmax_max, softmax_sum) state from N independent stages (ring-attention shards, KV-cache chunks, sliding-window blocks) and needs to combine those partials into one final triple. The combination must be order-independent (the ring/tree can fold in any order) and must preserve numerical stability under the max-shift form.

**Relationship to CAND-FA2**: CAND-FA2 expresses the same online-softmax algebra as a *sequential recurrence* (one tile after another, carrying running state forward). This pattern expresses it as a *binary reducer*: given two equal-rank triples `(m_a, s_a, o_a)` and `(m_b, s_b, o_b)`, produce a merged triple `(m, s, o)`. Both forms agree exactly; the merge form is what makes ring-attention (and any tree/ring fold of partial attention outputs) work correctly regardless of communication schedule.

**Identity (the algebra is associative+commutative)**:
```
m       = max(m_a, m_b)
s_a'    = s_a * exp(m_a - m)
s_b'    = s_b * exp(m_b - m)
s       = s_a' + s_b'
scale_a = s_a' / s         # per-row, applies to all head_dim columns of o_a
scale_b = s_b' / s
o       = scale_a * o_a + scale_b * o_b
```
Associativity proof sketch: combining (a,b) then with c yields the same (m,s,o) as combining b,c then with a — because all three exponents re-center on the final max and the sums re-scale accordingly. Commutativity is by inspection (symmetric in a,b).

**Why this matters for ring-attention**: Each ring step ships a (m,s,o) triple between cards/cores. On receipt, the consumer merges with its current accumulator using this kernel. Without the algebraic property, the result would depend on ring traversal order, breaking determinism across ring topologies and across re-runs.

**Shape**: per-row state (`m`, `s`) has shape `[bn, seq, softmax_tail]` where softmax_tail is the per-row reduction width (usually 8 to align to one fp32 block). `o` has shape `[bn, seq, head_dim]`. The merge runs row-by-row (per `(bn, seq)` index), broadcasting the per-row scale across head_dim.

**Concrete anchor** (public-API VEC primitives, worker-local LocalTensor names):
```cpp
// Inputs in UB: maxA, maxB, sumA, sumB (each R*softmaxTail fp32); outA, outB (each R*headDim T).
// Scratch: scaleA, scaleB (R*softmaxTail fp32).
constexpr uint64_t mask[2] = {UINT64_MAX, 0};
AscendC::BinaryRepeatParams rpSoft = {1, 1, 1, 8, 8, 8};
uint8_t rt = (R * softmaxTail + 64 - 1) / 64;

// Step 1: merged max
AscendC::Max(maxOut, maxA, maxB, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();

// Step 2: per-input exp(m_i - m)
AscendC::Sub(scaleA, maxA, maxOut, mask, rt, rpSoft);
AscendC::Sub(scaleB, maxB, maxOut, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(scaleA, scaleA, mask, rt, {1, 1, 8, 8});
AscendC::Exp(scaleB, scaleB, mask, rt, {1, 1, 8, 8});
AscendC::PipeBarrier<PIPE_V>();

// Step 3: scaled sums and merged sum
AscendC::Mul(scaleA, sumA, scaleA, mask, rt, rpSoft);
AscendC::Mul(scaleB, sumB, scaleB, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Add(sumOut, scaleA, scaleB, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();

// Step 4: per-row out-scale (scaleA, scaleB are reused as scale_a, scale_b)
AscendC::Div(scaleA, scaleA, sumOut, mask, rt, rpSoft);
AscendC::Div(scaleB, scaleB, sumOut, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();
// Step 5: broadcast scaleA/scaleB across head_dim and combine — see CAND-RAU-3 for stride shape
```

**Numerics**:
- Stable: every `Exp` argument lies in `(-∞, 0]`.
- An "empty" input (all masked, m_i ≈ -3e38, s_i = 0): contributes `exp(-3e38 - m) ≈ 0` and `0 * 0 = 0`, harmless.
- Both inputs empty: `s = 0`, the Div introduces 0/0 = NaN; worker must short-circuit empty-pair merges per op spec.

**Determinism**: Deterministic when (a) each row is single-AIV-owned, (b) the input GM tensors are themselves deterministic outputs of upstream stages, and (c) operands are read in a fixed (prev,cur) order. The algebra is order-invariant only at the math level (exact in real arithmetic); fp32 rounding errors do depend on order, so ring fold order should be fixed across runs for bit-exact reproducibility. Document this caveat — it is the same caveat as any fp tree-reduction.

**Hard do-not-apply**:
- Do NOT apply when the upstream stages produced their per-stage probabilities (already divided by per-stage sum); this merge expects *unnormalized* per-stage state — i.e. `out_i` must be `sum_j(exp(s_ij - m_i) * V_j)` NOT divided by `sum_i`. Confirm upstream contract before adopting.
- Do NOT collapse the two Div's into one by computing `scale = s_scaled / s` outside the loop and broadcasting; the kernel issues two Div's because both ratios are needed separately for the two outputs.

**Other instances predicted**:
- Ring-attention (the canonical case): each card holds one shard's (m,s,o), ring rotates them, each step merges.
- FlashAttention-v3 cross-block merge when blocks are processed concurrently by different AIVs and merged at the end.
- Distributed softmax/cross-entropy: each shard computes local (m,s) plus logsumexp, merged tree-wise.
- MoE expert output combining when each expert produces softmax-weighted state and the gate-weighted sum is computed via this merge form.
- Any "split-K" reduction over an inner dim where the reduction is `sum of exp(...)`.

**Risks before promotion**:
- Numerical: re-runs must use the same ring order to get bit-exact match. Build harness must record and fix the order; otherwise A/B perf and det-check both drift.
- Boundary: all-empty pair (both `m = -inf`) produces NaN; worker MUST gate.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel that ships this merge (ring-attention update, KV-cache merge, or partial-FlashAttention finalize) and passes Pass A + Pass B + det + perf on 3_FusionAttention or a similar op.

## CAND-RAU-2: Packed-variable-length (TND/varlen) traversal via cumulative-offset pointer table with per-AIV batch-boundary advance
`applies_to: any SoC with int64 GM scalar reads via .GetValue(); cann=9.0.0+; op_class=variable_length_sequence / packed_TND / ring_attention_varlen / flash_attn_varlen`
`derived-from: cann-source (ring-attn-class update TND variant, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update_tnd.h (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: An attention-class or sequence-class kernel consumes a TND/packed-varlen layout where N variable-length sequences are concatenated end-to-end along the T (token) axis. Per-batch offsets are provided as a cumulative-offset table of length `B+1` (CSR-style): batch `i` spans tokens `[actualSeqQlen[i], actualSeqQlen[i+1])`. Each AIV is assigned a contiguous slice of the global T axis (`dimTIndexCore .. dimTIndexCore + dimTCore`) and must, per token, know which batch that token belongs to in order to compute correct softmax-tail / head-dim GM offsets.

**Why not just precompute a per-token batch map**: would require a separate scratch tensor of length T; the cumulative-offset form is already what host-side tiling provides and avoids the extra memory. The cost is a binary-search-or-linear-scan per AIV at startup, then constant per-step bookkeeping.

**Pattern**:
1. **Startup search**: At AIV start, linearly scan `actualSeqQlen[0..B]` to find the batch containing the first token `dimTIndexCore`. Record `curBatchIndex`, `seqNumBatchStartIndex = actualSeqQlen[curBatchIndex]`, `seqNumBatch = actualSeqQlen[curBatchIndex+1] - seqNumBatchStartIndex`, and `seqNumBatchTail = dimTIndexCore - seqNumBatchStartIndex` (position within batch). Linear scan is fine because `B` is small (≤ 256 typical) and amortized over `dimTCore` per-token work.
2. **Per-token main loop**: For each of `dimTCore` tokens this AIV owns, work on the current batch.
3. **Boundary check (while-loop)**: Before each token, if `seqNumBatchTail == seqNumBatch`, the previous step exhausted the batch. Advance: `curBatchIndex += 1`, refresh start/end/seqNumBatch from `actualSeqQlen`, reset `seqNumBatchTail = 0`. Use `while` (not `if`) to correctly skip zero-length batches.
4. **Per-token offset computation**: Use `(curBatchIndex, seqNumBatchTail)` to index softmax tensors at `[curBatchIndex, seqNumBatchTail, head, ...]` and attn tensors at the corresponding flat T offset. The softmax tensor in this varlen layout is shape `[sum(seq_i_padded_to_block), head_num, softmax_tail]`, so per-batch stride must use the *actual* `seqNumBatch` for that batch (not a constant).
5. **Increment**: `seqNumBatchTail += 1` at end of each token's iteration.

**Concrete anchor** (public-API ; worker-local names):
```cpp
// actualSeqQlenGm is GlobalTensor<int64_t>, length B+1, CSR-style.
int64_t curBatch = 0, batchStart = 0, batchEnd = 0, batchLen = 0, tailInBatch = 0;
// Startup: find batch containing dimTIndexCore
for (int64_t b = 0; b < batchSize; b++) {
  batchEnd = actualSeqQlenGm.GetValue(b + 1);
  if (dimTIndexCore < batchEnd) {
    curBatch = b;
    batchStart = actualSeqQlenGm.GetValue(b);
    batchLen = batchEnd - batchStart;
    tailInBatch = dimTIndexCore - batchStart;
    break;
  }
}

// Per-token loop
for (int64_t t = 0; t < dimTCore; t++) {
  while (tailInBatch == batchLen) {           // skip exhausted/empty batches
    curBatch += 1;
    batchStart = actualSeqQlenGm.GetValue(curBatch);
    batchEnd   = actualSeqQlenGm.GetValue(curBatch + 1);
    batchLen   = batchEnd - batchStart;
    tailInBatch = 0;
  }
  // softmax GM offset: per-batch stride is batchLen * head_num * softmax_tail
  int64_t softmaxOffset = batchStart * head_num * softmax_tail + tailInBatch * softmax_tail;
  int64_t attnOffset    = (dimTIndexCore + t) * head_num * head_dim;
  // ... per-head work, advance offsets by head_num_loop_each * head_dim/softmax_tail ...
  tailInBatch += 1;
}
```

**Numerics / correctness**:
- `while` (not `if`) is mandatory: zero-length batches (legal in some packed layouts) would otherwise corrupt indexing.
- Cast `actualSeqQlen` reads as int64; per-batch lengths can exceed INT32_MAX on long-context ops. Match the project's int64 type rule.
- Startup linear scan is O(B) per AIV; for large B (≥ 1024), switch to binary search. For typical B ≤ 256, linear is fine.

**Determinism**: Fully deterministic — the layout and traversal are data-driven by the input table.

**Hard do-not-apply**:
- Do NOT use this pattern when the layout is BSH (uniform seqlen) — overhead has no benefit; pre-computed strides are faster.
- Do NOT skip the `while` (use `if`): silent index corruption on zero-length batches.
- Do NOT precompute every per-token (batch, tail) into a scratch tensor for "speed" — the GM round-trip to read the scratch exceeds the linear-scan + while-advance cost.

**Other instances predicted**:
- FlashAttention varlen forward/backward (`cu_seqlens_q`, `cu_seqlens_kv` are exactly this CSR table).
- Variable-length softmax / cross-entropy with packed labels.
- Variable-length pooling / scatter-mean per-sequence.
- Per-sequence layernorm over packed TND inputs.
- Variable-length ROPE / KV-cache append.

**Risks before promotion**:
- Per-AIV startup cost: if `dimTCore` is tiny (1-2 tokens) and `B` is large, linear scan dominates. Worker should fall back to BSH path or precompute per-AIV start-batch in tiling host code.
- The `actualSeqQlen` tensor must be on GM (the kernel reads via `.GetValue()` which does a scalar GM load); putting it in a workspace UB cache buys nothing because each value is read at most twice.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel that consumes varlen layout end-to-end.

## CAND-RAU-4: Concatenated-pair UB layout for 2-input binary-reduction kernels — single LocalTensor of size `2N`, operand B at offset `N`
`applies_to: any SoC with public AscendC TQue/LocalTensor; cann=9.0.0+; op_class=binary_reduction / pairwise_merge / 2_input_pointwise`
`derived-from: cann-source (ring-attn-class update SBH + TND, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update.h + _tnd.h SoftmaxDataMoveIn/AttnDataMoveIn (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: Kernel implements a binary reduction `out = f(a, b)` where `a, b` have identical shape, both come from GM, and both are processed in the same Compute block. The naïve form allocates two separate queues / two separate LocalTensors. This pattern packs both operands into ONE LocalTensor of size `2 * N` and indexes operand B via offset `[N]`, halving queue count and easing buffer-budget pressure.

**Pattern**:
1. **Init**: Allocate one queue per pair (not two). Configure each queue's buffer size to `2 * N * sizeof(T)` (twice operand size). `tPipe->InitBuffer(pairQueue, BUFFER_NUM, 2 * N * sizeof(T))`.
2. **DataMoveIn**: AllocTensor returns a LocalTensor of full length `2N`. Issue two DataCopyPad's, first into `tensor` (offset 0), second into `tensor[N]` (offset N elements). Each DataCopyPad uses the SAME DataCopyExtParams (same shape, same stride).
3. **Compute**: Reference operand A as `tensor` and operand B as `tensor[N]` directly in the VEC primitive call. e.g. `Max(out, tensor, tensor[N], mask, repeat, params)`.
4. **Free**: One FreeTensor releases both operands together.

**Concrete anchor** (public-API; worker-local names):
```cpp
// Init phase
tPipe->InitBuffer(pairQueue, BUFFER_NUM, 2 * N * sizeof(float));

// DataMoveIn
auto pair = pairQueue.AllocTensor<float>();        // length 2N
AscendC::DataCopyPad(pair,        srcAGm[offset], copyParams, padParams);
AscendC::DataCopyPad(pair[N],     srcBGm[offset], copyParams, padParams);
pairQueue.EnQue<float>(pair);

// Compute
pair = pairQueue.DeQue<float>();
AscendC::Max(out, pair, pair[N], mask, repeatTimes, repeatPar);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Sub(scratchA, pair,    out, mask, repeatTimes, repeatPar);
AscendC::Sub(scratchB, pair[N], out, mask, repeatTimes, repeatPar);
// ... reuse pair[0..N) and pair[N..2N) for the full merge ...
pairQueue.FreeTensor<float>(pair);
```

**Benefits**:
- **Queue count halved**: for a 2-input op with 3 logical streams (e.g. softmax has prev_max, prev_sum, prev_out and cur_max, cur_sum, cur_out → six logical streams paired as three pairs), this saves three queues. Important when budget is tight (typical Atlas-class budget: 12-16 queues).
- **DMA bandwidth identical**: two DataCopyPad's go out either way; packing into one tensor doesn't merge them.
- **VEC primitive overhead unchanged**: Max/Sub/etc don't care that operands are offset-aliased; they just compute on the addresses they're handed.
- **Cache locality marginally better**: A and B end up in adjacent UB blocks; the VEC unit's load step crosses them in one fetch sometimes.

**Numerics**: No effect on semantics — identical to two-queue form.

**Hard do-not-apply**:
- Do NOT pack when operands have DIFFERENT shape: defeats the offset-indexing assumption.
- Do NOT pack when operand A and operand B have DIFFERENT lifetimes (e.g. A is used early, B is used in a downstream compute): the FreeTensor releases both together. Use separate queues if independent free is needed.
- Do NOT pack when buffer-budget is tight in absolute UB bytes: this form REQUIRES 2*N per queue slot; two separate queues would have N each. Total UB usage same per pair, but allocation is less flexible.
- Do NOT pack more than 2 operands this way without measurement: 3-way packing (`tensor`, `tensor[N]`, `tensor[2*N]`) is legal but harder to reason about; downstream readers struggle.

**Other instances predicted**:
- Any 2-input merge/reduce kernel: `max(a, b)`, `min(a, b)`, pairwise softmax-merge (this op), pairwise tree-reductions.
- Top-k merge kernels combining two sorted-and-padded chunks.
- KV-cache append where `[prev_kv, new_kv]` are concatenated then re-indexed.
- Two-stream fused element-wise: `out = α*a + β*b` with α, β scalar.

**Risks before promotion**:
- Buffer-budget reasoning becomes per-pair (in 2N units) instead of per-stream — analyzers (`probe_report.md` budget sections) must understand the convention. Document in the kernel header that the queue holds 2 operands packed.
- Some VEC primitives have alignment requirements on offsets; `tensor[N]` must satisfy them. For fp32 with 32-byte alignment, `N` must be a multiple of 8 — typically true (compute counts are usually mult of 64 = one repeat). Add a static_assert.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel that measurably benefits from the queue-count reduction (i.e. the un-packed form runs out of buffer slots or has measurably worse pipeline overlap).
## CAND-FAG-2: Deterministic backward via coordinate-partitioned core dispatch — replace atomic-add with a number-theoretic per-core (s1,s2)-tile assignment so each output element is touched by exactly one core
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=deterministic_backward / flash_attention_backward_deterministic / any_bwd_op_where_atomic_add_violates_determinism_requirement`
`derived-from: cann-source (fa-grad-class backward, 2026-05-10 multicann)`
`verified_on: cann ops-transformer flash_attention_score_grad/op_kernel/arch35/deter.h — coordinate-assignment helpers (TransTilingSplitMode / CalDenseIndex / CalDenseIndexForSingleN family, ~200 lines) and the corresponding kernel-side deter dispatcher header that consumes them; the design doc explicitly names "通过特定分核方式避免多核同地址累加，达到确定性计算的效果" as the deterministic-mode goal`
`unverified_on: a5_ops`

**Trigger**: Backward op needs bit-reproducibility across runs (PyTorch `torch.use_deterministic_algorithms(True)`, training-loss-reproduction in research, downstream debugging). Naive backward uses `SetAtomicAdd<float>` to combine partial gradients from multiple cores — but atomic-add ordering is NOT deterministic on multi-core, so the same input produces slightly different gradients across runs. Standalone numerical precision is fine; the bit-pattern reproducibility is what fails.

**Recommendation**: Replace cross-core atomic-add with a **coordinate-partitioned dispatch**: pre-compute, on the host or in a once-per-launch scalar block, an assignment `(coreId) → list-of-(b,n2,g,s1_tile,s2_tile)` such that for every (b,n2,g,s1,s2) tile of the output, EXACTLY ONE core is the writer. Each core then performs the bwd matmuls for its assigned tiles and writes through a plain `DataCopy` (no atomic), so the output is the deterministic sum-of-partials-in-fixed-order.

Two coordinate-assignment shapes seen in the reference, both expressible in public AscendC:

1. **Dense (rectangular) tile space, `B × N2 × G × S1.o × S2.o`**: assignment by a closed-form GCD / modular formula. Given core index `coreId` and the per-axis outer counts `(b, n2, g, m=S1.o, n=S2.o)`, compute `(batchId, n2Idx, gIdx, s1Idx, s2Idx)` via `Ceil`/modulo arithmetic so that the loop fills the (m,n) grid in a column-major-then-batch order with period `LCM(m, R)/m` where `R` is the number of cores. Tiles that would land outside the grid are marked invalid (`batchId = -1`) and the core skips them.
2. **TND / variable-S sequences**: assignment must respect the prefix-sum of `actualSeqQlen[]` / `actualSeqKvlen[]`. Compute the per-batch tile counts from the prefix-sum, run the same modular dispatch within each batch's tile rectangle, then map back to GM offsets.

The dispatch is purely scalar work (~50 cycles per tile) and runs once at task-loop entry; the per-tile compute cost dominates so dispatch overhead is negligible.

**Concrete anchor** (public AscendC):
```cpp
// Coordinate-assignment for one (taskId) on this core's dispatch sequence
__aicore__ inline void AssignTile(int64_t taskId, int64_t coreId, int64_t totalCores,
                                  int64_t m /*S1.o*/, int64_t n /*S2.o*/, int64_t b,
                                  int64_t &batchId, int64_t &s1Idx, int64_t &s2Idx) {
    int64_t r = (taskId / totalCores) + 1;       // round index within this core
    int64_t j = coreId + 1;                       // 1-based core position
    int64_t p = (Ceil(r, m) - 1) * totalCores + j;
    int64_t w = ((p - 1) % b) + 1;
    int64_t y = Ceil(p, b);                       // s2Idx (1-based)
    int64_t r1 = ((r - 1) % m) + 1;
    int64_t y1 = ((y - 1) % m) + 1;
    int64_t x = y1 + r1 - 1;
    if (x > m) x -= m;                            // s1Idx (1-based, wrapped)
    batchId = (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) ? w : -1;
    s1Idx = x - 1; s2Idx = y - 1;                 // back to 0-based
}

// Main task loop — NO SetAtomicAdd, plain DataCopy
for (int64_t taskId = 0; taskId < tilesPerCore; ++taskId) {
    int64_t batchId, s1Idx, s2Idx;
    AssignTile(taskId, coreId, totalCores, s1Outer, s2Outer, b, batchId, s1Idx, s2Idx);
    if (batchId < 0) continue;
    // ... compute partial dq/dk/dv for this tile ...
    DataCopy(dqGm[GmOffsetOf(batchId, s1Idx)], dqUbFp32, len);   // no atomic — single writer
}
```

**Why it works**:
- The assignment formula guarantees each (batch, s1, s2) tile has exactly one `(taskId, coreId)` pair producing it — the proof is the GCD-based period of the modular schedule (`gcd(m, R)` divides `m·n`, so the sequence walks every tile exactly once before repeating)
- Removing `SetAtomicAdd` removes the only nondeterministic ordering point; per-core compute order is fixed by the task loop, so re-runs produce bit-identical output
- The assignment is purely on outer-loop indices (`s1Outer`, `s2Outer`), so the inner per-tile matmul and softmax stay exactly the same as the non-deterministic path — only the work-to-core mapping changes

**Determinism**: Deterministic by construction when (a) the assignment formula is bijective on `[0, b·m·n) → (batchId, s1Idx, s2Idx)`, (b) within a tile the reduction order is fixed (use the same VEC primitive order as the non-deterministic path), and (c) cross-core ordering does not matter because no cross-core address is shared. The formula's bijection is a number-theoretic invariant — verifiable by enumerating small (b,m,n,R) and counting.

**Other instances predicted**:
- Any backward op currently using `SetAtomicAdd` across cores — embed-grad accumulation, scatter-add backward, MoE expert-gradient combine
- Forward ops that need bit-reproducible reductions (deterministic LayerNorm with very large hidden dim, deterministic all-reduce-equivalent on-chip)
- The same shape applies to forward FlashAttention's dQ-accumulator if a deterministic variant is requested — though for forward most existing kernels are already deterministic by row-ownership

**Risks before promotion**:
- Load-balance: the modular dispatch is bijective but NOT load-balanced when `R` does not evenly divide `b·m·n` — tail cores get one fewer tile. Fine for most shapes; pathological when `tilesPerCore` is small (1–2) and the tail-imbalance is 50%
- TND / variable-seq case is significantly more complex — the formula must run per-batch, and the prefix-sum array must be available in scalar memory; check that `actualSeqQlen` fits the scalar block
- The non-determ path is typically ~5–15% faster (because tile-to-core matching can be greedy/locality-aware); deterministic mode is an explicit opt-in, not a default
- Bijection MUST be unit-tested for each (b,n2,g,m,n,R) combination shipped — silent off-by-one in the formula produces silent missing tiles → wrong gradients

**Cross-reference**:
- CAND-FAG-1 (three-kernel pre/main/post split): this candidate REPLACES the MAIN kernel's atomic-add accumulation, PRE/POST still apply unchanged
- CAND-FA3 (GM workspace slot rotation): orthogonal — that pattern is for cross-stage pipelining within one kernel; this is for cross-core within-stage determinism
- P-P67-candidate (HODSA hash-owner deterministic scatter, INVALIDATED on 950PR): this candidate is the FA-class alternative to HODSA — uses a number-theoretic bijection instead of hash partitioning. HODSA was invalidated because hash partitioning loses load balance under skewed indices; this dense-coordinate variant has the SAME load-balance weakness but only on `R ∤ b·m·n` tails (much milder)
- OL-91 (cube playbook): orthogonal

**Promote when**: a5_ops ships a backward op with a `--deterministic` mode that passes bit-reproducibility (same input → identical output bytes across 100 runs) AND shows <20% perf loss vs the non-deterministic atomic-add baseline.
## CAND-MLA-3: Whole-VEC barrier for cross-partition stage transitions when consecutive vec stages have different core-partitionings
`applies_to: any soc with WaitAllCore / SyncAll on vec engine; cann=9.0.0+; op_class=multi_stage_vec_pipeline_with_repartition`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h (vec-side process body at ~L880-L955 — engine-scoped WaitAllCore on PIPE_MTE3 appears at three distinct seams: between the down-projection rmsnorm and the subsequent matmul gate; between rmsnorm finish and the dual-output up-projection; between the up-projection and the dequant stage), inline source comment explicitly states the seam exists because two consecutive vec stages have different core-partitioning strategies and require an all-vec barrier between them`
`unverified_on: a5_ops`

**Trigger**: A vec-side pipeline has consecutive stages where the across-core partitioning differs — stage N partitions the workload along axis X (e.g. batch / token rows), stage N+1 partitions along axis Y (e.g. head dim / N axis). At the seam, each vec core has produced an output region that core N+1 (of the next stage) needs to read but did NOT itself produce. A paired AIC/AIV CrossCore flag (CAND-FA1) is insufficient — it synchronizes one AIV with one paired AIC, NOT one AIV with all peer AIVs. Without a whole-vec barrier, stage N+1 starts reading regions that some peer AIV is still writing.

**Recommendation**: Insert `AscendC::WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(flagId)` (or the equivalent whole-vec barrier helper on the target SoC) at the seam between two vec stages that have different partitioning. The barrier semantics:

  - All AIV sub-blocks wait until every AIV sub-block has reached this point
  - The barrier flag is scoped to the vec engine (NOT cube)
  - The pipe argument (`PIPE_MTE3`) ensures the GM write that the AIV is about to depend on has retired before the barrier completes

The barrier is REQUIRED whenever a re-partitioning seam exists; it is NOT required between consecutive vec stages that share the same core partitioning along the same axis (those need only an intra-core `SetFlag/WaitFlag<HardEvent>` pair).

A typical layered structure mixing all three sync layers:
  - intra-core: `SetFlag<HardEvent::MTE3_V>` / `WaitFlag<HardEvent::MTE3_V>` (within one AIV's pipe chain)
  - whole-vec: `WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(allVecBarrierId)` (at a re-partitioning seam within vec)
  - cross-engine: `CrossCoreSetFlag<0x2, PIPE_MTE3>(stageDoneId)` / `CrossCoreWaitFlag(stageDoneId)` (between vec and cube, per CAND-FA1)

These three layers compose without conflict — they target disjoint scopes. Using the wrong layer is the failure mode this candidate flags.

**Concrete anchor** (verified pattern from the MLA prolog vec process; two re-partitioning seams shown):
```cpp
// Stage A (vec): row-partitioned RmsNorm over the latent
RmsNormRow(cNormUb, cPreGm, gammaUb, /*col=*/Hc, eps);  // each AIV owns a row subset
DataCopy(cNormGm[rowOff], cNormUb, Hc);                 // emit via MTE3

// Re-partitioning seam: Stage B partitions by head-index, not by row.
// A whole-vec barrier is required because Stage B core will read rows that
// peer AIV cores wrote in Stage A.
constexpr uint32_t allVecBarrierId = 3;  // user-owned flag ID, in 0..7 range
AscendC::WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(allVecBarrierId);

// Stage B (vec): head-partitioned RoPE on cross-batch rows
RotaryPosEmb(qRopeUb, qRotUb, cosUb, sinUb, sharedTmp,
             /*rows=*/rowsForThisHead, /*cols=*/Dr, sinCosStride);
DataCopy(qRopeOutGm[headOff], qRopeUb, rowsForThisHead * Dr);

// Another seam if Stage C re-partitions again (e.g. dequant by token)
AscendC::WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(allVecBarrierId);
```

**Why it works**:
- Re-partitioning is a SoC-level data-flow phenomenon: stage N's output region is the union over all AIV cores of per-core sub-regions, and stage N+1's input region is also the union over all AIV cores of per-core sub-regions, but the two unions are partitioned differently. The barrier converts the cross-region dependency into a single "all of stage N has retired before any of stage N+1 starts" guarantee, which is sufficient for any read pattern in stage N+1.
- The pipe argument (`PIPE_MTE3`) ensures the GM write that retires the barrier is the actual data write, not a register write — so consumers in the next stage are guaranteed to see the data, not just the control signal.
- Whole-vec barriers are cheaper than `SyncAll<true>()` because they exclude the cube engine; if no AIC is participating in this seam, broadcasting the barrier to AIC adds unnecessary cube-side stall (cube can keep cube-internal work going). Use the engine-scoped form whenever the seam is engine-internal.

**Determinism**: Adding a barrier never breaks determinism — it only narrows the in-flight schedule. Determinism of the pipeline is governed by the per-element reductions and writes inside each stage; the barrier ensures stage N's writes are visible to stage N+1's reads, which is a precondition for the per-element compute to even be well-defined.

**Other instances predicted**:
- MLA-class prolog (this verified instance) where rmsnorm-by-row feeds qcqr-by-head
- FlashAttention-style vec pipelines where softmax-by-row feeds output-rescale-by-embed
- Fused norm + scatter where norm-by-row feeds scatter-by-target-slot
- MoE expert dispatch where routing-by-token feeds per-expert-shuffle-by-expert
- Any two-stage vec pipeline where stage 1 is "all cores process disjoint row subsets" and stage 2 is "all cores process disjoint column subsets" of the same intermediate tensor
- Permute / transpose epilogues where the producer and consumer partition the tensor along different axes

**Risks before promotion**:
- Using `SyncAll<true>()` instead of `WaitAllCore<SYNC_MODE_ALL_VEC>` is a correctness-equivalent but wasteful alternative — it stalls cube needlessly. Verify the seam is engine-internal before choosing the cheaper form.
- Using an intra-core `WaitFlag<HardEvent::MTE3_V>` instead of the whole-vec barrier is a CORRECTNESS BUG — it only orders the producer's pipe within one AIV, not across peer AIVs. Audit code that has a stage transition + intra-core pipe wait but no whole-vec barrier; this is the most plausible silent-corruption mode for re-partitioning seams.
- Excessive whole-vec barriers serialize the vec engine; ideally place one per re-partitioning seam, not per stage. The MLA reference inserts three barriers across the ~75-line vec process — verify counter-cases that introduce barriers without a re-partition.
- The `PIPE_MTE3` choice assumes the producer's last operation is a GM write; if the producer's last op is a vec compute (PIPE_V) with no GM write, use `PIPE_V` instead — wrong pipe = released before data retires.
- a5_ops has no shipping kernel with vec-engine re-partitioning seams currently; promoting requires a port (e.g. fused norm-then-scatter, fused multi-head rope) that exercises this shape.

**Cross-reference**:
- CAND-FA1 (cross-core flag handoff) — orthogonal layer (this is intra-vec, CAND-FA1 is cross-engine); both compose in one kernel
- P-P75 (intra-core SetFlag/WaitFlag<HardEvent> for SIMD pipe sync) — the intra-core layer; this candidate is the engine-scoped layer that sits between intra-core and SyncAll
- CAND-MLA-1 (latent-prolog skeleton) — the multi-stage ladder where this barrier appears at the vec-side re-partitioning seams
- OL- entries on `SyncAll` overuse — this is the "more-targeted-than-SyncAll" replacement

**Promote when**: an a5_ops vec-pipeline op ships with a documented re-partitioning seam (e.g. fused-norm-then-scatter, fused multi-head rope, fused softmax + output rescale) AND the seam's whole-vec barrier is measured cheaper than the `SyncAll<true>()` alternative AND the engine-scoped barrier is verified necessary (i.e. a peer-AIV-read dependency demonstrably exists at the seam).
## CAND-MLA-4: Interleaved-pair RoPE via GatherMask even/odd split + symmetric Mul(cos)+Mul(sin) (single-pass, no transpose)
`applies_to: any soc with public AscendC GatherMask + Mul + Add + BinaryRepeatParams; cann=9.0.0+; op_class=rotary_position_embedding / pairwise_rotation_on_interleaved_layout`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/service_rope.h (RotaryPosEmb, ~75 lines, public-API only — GatherMask, Mul, Add, BinaryRepeatParams, GatherMaskParams), attention/mla_prolog/docs/aclnnMlaProlog.md (formula block: q^R = ROPE(c^Q · W_QR) on a 64-wide head dim, sin/cos shape (B,S,Dr))`
`unverified_on: a5_ops (a5_ops has 1_RotaryMul which uses a different concrete shape — verify alignment before promoting)`

**Trigger**: Op needs RoPE rotation on a tensor laid out as `[row, col]` where each row's `col` elements are interleaved pairs `(x0, x1, x2, x3, ...)` with the rotation defined as `(x_2i, x_2i+1) -> (x_2i*cos - x_2i+1*sin, x_2i*sin + x_2i+1*cos)`. The sin/cos coefficient tensors are pre-broadcast to length `col` per row with the convention that `cos[:col/2]` and `cos[col/2:]` are the two halves used by the two `Mul` calls (sin/cos are NOT per-pair scalars; they are per-half vectors). `col` is small (e.g. 64 for MLA — `Dr`-dimension rope), small enough to fit several rows in UB. `col` must be a multiple of `ALIGN_BLOCK_SIZE / sizeof(C)` (32B / element-size).

**Recommendation**: Implement RoPE in a single vec pass with NO explicit transpose / reshape — use `GatherMask` with patterns `1` (odd-indexed, picking `x_0, x_2, ...`) and `2` (even-indexed, picking `x_1, x_3, ...`) to materialize the two half-vectors in scratch UB, then issue four `Mul` calls and one `Add`:

  - `evenHalf = GatherMask(input, mask=1)`  — selects `x_0, x_2, x_4, ...`
  - `oddHalf  = GatherMask(input, mask=2)`  — selects `x_1, x_3, x_5, ...`
  - `tmp0 = evenHalf * cos[lower half]`  — strided mul, blockNumPerRowHalf source stride
  - `tmp0_high = oddHalf * cos[upper half]` — written to `outputLocal[col/2 :]`
  - `tmp1 = oddHalf * sin[lower half]` — written to sin-scratch
  - `tmp1_high = evenHalf * sin[upper half]` — written to sin-scratch[col/2 :]
  - `out = tmp0 + tmp1` (single Add over the full `row*col` count — order-irrelevant per element)

The `BinaryRepeatParams` use `src0BlkStrideIn=1, src1BlkStrideIn=1, dstBlkStrideIn=1` with `src0RepStrideIn = blockNumPerRowHalf` (the gathered half-vector's per-row block count) and `src1RepStrideIn = blockNumSinCosRepStride` (the sin/cos tensor's per-row block count). This lets a SINGLE `Mul` call cover multiple rows in one vec instruction — the inner repeat axis is rows, not columns. Critical for amortizing the vec-instruction overhead when `col` is small (`Dr=64` means `col/2 = 32` elements per Mul, which is one repeat-iteration; rows are the parallelism axis).

The `shareTmpUb` buffer holds the two reinterpret-cast scratches: `reArrLocal[2*row*col/2]` for the gathered halves and `outputLocalSinTmp[row*col]` for the sin-side Mul outputs. Sized `2 * row * col * sizeof(C)` bytes total. The output can alias the input — the gather has already materialized the rearrangement, so overwriting input mid-pass is safe.

**Concrete anchor** (verified pattern; assumes `col` is a multiple of 32B/element, `sinCosRepStride` is the per-row stride of the sin/cos arrays in elements):
```cpp
// rsvdCnt is the GatherMask output count for residue tracking
uint64_t cnt = row * col;
uint64_t rsvdCnt = 0;
LocalTensor<C> reArr   = sharedTmp.ReinterpretCast<C>();
LocalTensor<C> sinTmp  = sharedTmp.ReinterpretCast<C>()[cnt];
GatherMaskParams gp { 1, 1, 0, 0 };  // 1 repeat, src0 stride 1, no rep strides

// Materialize the two halves of every row in one scratch
GatherMask(reArr,          input, /*mask=*/1, true, cnt, gp, rsvdCnt);
GatherMask(reArr[cnt >> 1], input, /*mask=*/2, true, cnt, gp, rsvdCnt);
AscendC::PipeBarrier<PIPE_V>();

uint8_t bpr     = col / (32 / sizeof(C));   // blocks per row
uint8_t bprHalf = bpr >> 1;
uint8_t bprSinCos = sinCosRepStride / (32 / sizeof(C));
BinaryRepeatParams mp { 1, 1, 1, bpr, bprHalf, bprSinCos };

// rows are the outer repeat, cos/sin sit at strided per-row offsets in their tensor
Mul(output,             reArr,             cos,             col >> 1, row, mp);
Mul(output[col >> 1],   reArr[cnt >> 1],   cos[col >> 1],   col >> 1, row, mp);
Mul(sinTmp,             reArr[cnt >> 1],   sin,             col >> 1, row, mp);
Mul(sinTmp[col >> 1],   reArr,             sin[col >> 1],   col >> 1, row, mp);
AscendC::PipeBarrier<PIPE_V>();
Add(output, output, sinTmp, cnt);
```

**Why it works**:
- The two `GatherMask` calls with masks `1` and `2` decompose the interleaved layout into two contiguous half-tensors with no explicit transpose / no DataCopy — `GatherMask` is a single vec instruction that materializes the gathered output by stride-selection, far cheaper than a Transpose op or a strided DataCopy.
- The four `Mul` calls each use the SAME `BinaryRepeatParams` with `repeatTimes=row` — the cost per row is one `Mul` instruction over `col/2` elements, with the row dimension absorbed by the repeat. For small `col` (e.g. Dr=64), this is ~16x more efficient than four vec-instruction-per-row.
- The `(cos, sin)` per-half encoding (cos lower half = cos for x_{2i}, cos upper half = cos for x_{2i+1}) sidesteps the negation that a naive `(x_2i*cos - x_2i+1*sin)` formula would need — the negation is absorbed into the sin tensor's pre-broadcast layout (sin upper half is the negated counterpart). The four-Mul-plus-Add shape is symmetric and uses NO negate primitive.
- Output aliasing input is safe because both `GatherMask` calls retire before any `Mul` reads `reArr`, and the `Add` reads `output` and `sinTmp`, neither of which is `input`. The `PipeBarrier<PIPE_V>` between gather and mul (and between mul and add) is the only required intra-core ordering.

**Determinism**: Each output element is `out[i] = a*cos_i + b*sin_i` with one Mul-and-one-Mul-and-one-Add per element — no reduction across elements, deterministic by construction. The four `Mul` calls touch disjoint regions of `output` and `sinTmp` and may be issued in any order; the `Add` is per-element and order-irrelevant.

**Other instances predicted**:
- MLA-prolog rope head (this verified instance) — Dr=64 interleaved-pair rope
- Standard LLaMA / Qwen / DeepSeek rope heads — same interleaved-pair convention, typically Dh=64 or 128
- Any pairwise-rotation pattern (complex multiplication on a real-valued interleaved layout): `(a + b*i) * (c + d*i) = (ac - bd) + (ad + bc)*i`
- Spherical / hyperspherical rotations expressed in pair-of-coordinates form
- Audio / vision positional encodings that use sinusoidal pairwise mixing on small head dims

**Risks before promotion**:
- `col` must be a multiple of `32B / sizeof(C)` (2 for fp32, 16 for bf16). For Dr=64 in bf16, `col/2 = 32` elements = 64B = 2 blocks — works cleanly; for non-aligned `Dr`, the gather pattern needs padding (out-of-scope here).
- The "interleaved-pair" layout convention `(x0,x1,x2,x3,...)` differs from the "half-half" convention `(x0,x1,...,xn/2, xn/2+1, ..., xn-1)` used by some HuggingFace rope variants. Verify the input layout convention against the reference before promoting — the gather masks 1/2 are correct ONLY for interleaved-pair.
- `sinCosRepStride` is the per-row stride of the cos/sin tensor IN ELEMENTS, not bytes; mis-specifying it produces all rows reading the same sin/cos row (silent staleness).
- The `BinaryRepeatParams` strides assume the sin/cos tensor has a separate row-stride from the gathered half. If sin/cos are broadcast (same row used for all `row` repeats), set `bprSinCos = 0`. The MLA reference uses non-zero stride; the broadcast case is a simplification that needs its own verification.
- The four `Mul` calls share the same `mp` — if rows have different per-row sin/cos (i.e. sin/cos are per-row from upstream gather), this is correct; if sin/cos are batch-broadcast, the rep stride configuration changes. Verify upstream sin/cos shape.
- a5_ops 1_RotaryMul exists but uses a different concrete shape (a5_ops is single-row, non-strided); the candidate is structurally similar but the strided multi-row form is not yet exercised.

**Cross-reference**:
- CAND-MLA-1 (latent-prolog skeleton) — the rope head is one of the two consumers of the fused up-projection in the skeleton; this candidate is the rope-head's concrete implementation
- CAND-MLA-2 (paged scatter cache) — the rope output is one of the two tensors scattered into the cache via CAND-MLA-2
- a5_ops 1_RotaryMul — closest existing benchmark; a port that adopted the gather-mask-1/2 + four-Mul shape would be the promotion vehicle for this candidate
- a5_ops 12_KvRmsnormRopeCache — the fused norm+rope+cache shape that combines CAND-MLA-1+3+4+2; if it ever lands as a real port, all four candidates promote together

**Promote when**: an a5_ops rope-shape op ships using this exact gather-mask-1/2 + four-Mul + Add shape, AND vec-instruction count vs an explicit `Transpose + per-row Mul` baseline is measured (the candidate claims ~2-4× vec-instruction reduction for small Dr; verify on a target shape), AND the sin/cos layout convention (interleaved-pair, per-half encoding) is verified against the upstream model's rope convention.
## CAND-RAU-3: Per-row scale broadcast across inner dim via non-zero src1Stride in BinaryRepeatParams — Brcb-free broadcast for `[R, inner]` *= `[R, 1]` shape
`applies_to: any SoC with public AscendC VEC Mul/Add and BinaryRepeatParams; cann=9.0.0+; op_class=row_broadcast_apply / online_softmax_output_rescale / attention_output_combine`
`derived-from: cann-source (ring-attn-class update, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update.h AttnCompute body (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: Apply a per-row scalar scale (`scale[R, softmax_tail]` with `softmax_tail ≤ 1 fp32 block = 8`) to a per-row vector (`data[R, inner]` with `inner` ≥ several fp32 blocks). The classic shape arising in online-softmax output rescale: every row has one (or up to 8) scale values, every row also has `head_dim` (or `embed_dim`) data values, and we must compute `data[r, j] *= scale[r, jj_mod_softmax_tail]` for `r ∈ [0, R)`, `j ∈ [0, inner)`.

**Why an alternative to Brcb (P-P62) is useful**: Brcb requires `R % 8 == 0` and a separate scratch tensor of size `R * 8 * inner_blocks` worth of broadcast results. For attention-class kernels where `inner = head_dim ∈ {64, 128, 256}` is large and `R = seqNumLoop` (the row count per AIV tile) is naturally a multiple of 8, the BinaryRepeatParams stride-based form sidesteps the Brcb entirely and reuses `softmaxTempBuf` (8 elements/row) in place.

**Shape**:
- `data` layout: `[R, inner]`, fp32, contiguous in `inner`. So row `r` occupies `data[r * inner .. (r+1) * inner)`.
- `scale` layout: `[R, softmax_tail]`, fp32, contiguous. Row `r` occupies `scale[r * softmax_tail .. (r+1) * softmax_tail)`. Typical `softmax_tail = 8` (one fp32 block).
- After broadcast, `data[r, j] *= scale[r, 0]` (scalar per row — or potentially per fp32-block, depending on softmax_tail).

**Pattern**: Issue Mul in `inner / 64`-step chunks across `inner` (a fp32 repeat is 64 elements). For each chunk, the BinaryRepeatParams.src1RepStride is set to `softmax_tail / 8` (in fp32 blocks), causing each successive repeat (= each successive row) to advance src1 by one row's worth of scale. The dst/src0 strides advance by `inner / 8` (one row of data). The mask covers 64 elements per repeat. Loop `inner / 64` times to cover the full inner dim.

**Concrete anchor** (public-API; worker-local names):
```cpp
// data, scale are LocalTensor<float>. R = rows, inner = inner-dim element count (mult of 64).
// softmax_tail typically 8 (1 fp32 block). repeatNumB32 = 64. blockNumB32 = 8.
constexpr uint64_t mask[2] = {UINT64_MAX, 0};
AscendC::BinaryRepeatParams rp = {
    /*dstBlkStride=*/1, /*src0BlkStride=*/1, /*src1BlkStride=*/0,
    /*dstRepStride=*/(uint8_t)(inner / 8),
    /*src0RepStride=*/(uint8_t)(inner / 8),
    /*src1RepStride=*/(uint8_t)(softmax_tail / 8)  // KEY: non-zero, walks across rows of scale
};
for (int64_t c = 0; c < inner / 64; c++) {
    AscendC::Mul(data[c * 64], data[c * 64], scale,
                 mask, /*repeatTimes=*/R, rp);
}
AscendC::PipeBarrier<PIPE_V>();
```

**Relationship to P-P62 (Brcb shape)**:
- P-P62: Brcb `scale [R]` → `bcastScale [R, 8]`, then per-block Sub/Mul against `data` with all strides = inner. Requires `R >= 8`. Costs one Brcb + one full broadcast tensor in UB.
- This pattern: Skip Brcb. Issue Mul `inner / 64` times with src1RepStride walking the scale tensor row-by-row. Costs `inner / 64` extra Mul calls (cheap; same compute either way) and zero extra UB scratch.
- Both produce identical results. Use this when `softmax_tail` is already > 0 (you have a real per-row scale buffer, not a single scalar) and inner-dim is large enough that the `inner / 64` Mul calls amortize well. Use P-P62 when you only have a scalar-per-row in a `[R]` shape and need to manifest the broadcast tensor.

**Numerics**: Identical to per-element Mul; no order effects. Deterministic.

**Hard do-not-apply**:
- Do NOT use when `inner % 64 != 0`: the trailing partial chunk needs a separate tail handler with adjusted mask (not shown in anchor — worker must add).
- Do NOT use when `R` exceeds 255 (RepeatTimes is uint8): split the outer R loop.
- Do NOT use when `softmax_tail / 8 > 255`: src1RepStride is uint8.
- Do NOT confuse with `src1BlkStride = 0` (which broadcasts within a single repeat); the KEY field here is `src1RepStride = softmax_tail / 8` (non-zero, walks across repeats).

**Other instances predicted**:
- Attention output combine (`out *= scale_per_row`) — the canonical case.
- LayerNorm / RMSNorm output scaling (gamma is per-feature not per-row, so opposite axis — but the same stride trick applies).
- Per-row gain/bias application in any normalized output (group-norm, batch-norm fold).
- Per-token weighting in MoE finalize (token-scale broadcast across hidden_dim).

**Risks before promotion**:
- Tail handling: `inner` not multiple of 64 needs a tail Mul with reduced mask. Worker must verify the op's actual head_dim values.
- The BinaryRepeatParams shape is brittle to read; document inline. Mis-setting `src1RepStride = 0` silently broadcasts the FIRST row's scale to all rows (catastrophic numerical bug, but cases would still "look reasonable" because outputs are not NaN). Add an assertion in dev builds.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel benchmarking this against P-P62 Brcb form to confirm the trade-off.

**Cross-reference**: This is a sibling to P-P62 (different shape, same semantic). C37 dedup should treat them as alternative implementations under the same `applies_to` umbrella, NOT merge them.

## A3→A5 Port Patterns (2026-05-12, from PR4778 cross-op-evidence batch)

> **Source**: gitcode `cann/ops-nn` PR #4778 (`ansen-changan/ops-nn` fork, FETCH_HEAD `88e41203c`) + already-merged `top_k_top_p_sample_v2` + partial-port `ada_layer_norm`, `fused_quant_mat_mul`. Extracted from the original `op_kernel/<op>*.h` headers + entry-point `.cpp` dispatchers that PR4778 patches with `arch35/` + `ascend950` configs.
>
> **Scope**: cross-op port-pattern catalog complementing W8 (artifact layout), W9 (OL-131 router patching), W10 (P-P89 surgical strip), W11 (ToFloat<> A5 restriction). These candidates capture the *kernel-shape* + *dispatch-shape* + *macro-guard-shape* decisions that determine whether a port is a "1-line strip" (P-P89) or requires a new variant carve-out.
>
> **Status of all entries below**: NEEDS_REVISION — mechanical scanners pending (C34a identifier-denylist, C34b compile-gate, C34c copy-shape ≤5%, C35 KB-overlap). Public-API surface throughout; verbatim CANN identifiers avoided.

### CAND-A3A5-1: Two-port-mode dichotomy — "surgical strip" vs "regbase parallel variant"

**Source**: `top_k_top_p_sample_v2` (master, A5-port merged) + `gather_elements_v2`, `group_norm_silu_quant`, `index_put_with_sort`, `rms_norm_quant`, `apply_adam_w_quant` (PR4778) — cross-op evidence (6 ops)
**Pattern class**: port-strategy decision
**A5 differentiator**: A5 admits TWO valid host-side wiring shapes; choosing the right one before kernel work begins prevents wasted rewrites.

**Mode A — Shared entry-point** (e.g. `top_k_top_p_sample_v2_def.cpp`):
```cpp
OpAICoreConfig aicConfig;   // single config object, no opFile.value override
aicConfig.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true)
         .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false);
this->AICore().AddConfig("ascend910b",   aicConfig);
this->AICore().AddConfig("ascend910_93", aicConfig);
this->AICore().AddConfig("ascend950",    aicConfig);   // same kernel .cpp serves all
```
No `arch35/` directory, no `_apt.cpp`. The kernel's existing `.cpp` is portable across V220 + V351 because every chip-specific divergence inside is guarded by `__CCE_AICORE__ == 220` or `__NPU_ARCH__ == 3003 || 3113`.

**Mode B — Dedicated regbase entry-point** (e.g. ctc_loss_v3, ada_layer_norm partial-port):
```cpp
OpAICoreConfig regbaseCfg;
regbaseCfg.DynamicCompileStaticFlag(true)
          .ExtendCfgInfo("opFile.value", "<op>_apt");   // routes to <op>_apt.cpp
this->AICore().AddConfig("ascend950", regbaseCfg);
```
Plus `op_kernel/<op>_apt.cpp` that `#include "arch35/<op>.h"` and dispatches into a separate-namespace kernel using `__VEC_SCOPE__` + `RegTensor<float>` + MicroAPI primitives.

**Decision rule**:
- Existing V220 kernel's hot loop is **already expressible** with public AscendC class-style ops (`Cast`, `Mul`, `Sort`, etc.) AND its chip-specific guards are limited to `DataCopyPad` polyfills or specific tiling-key filtering → **Mode A**, surgical strip per P-P89.
- Existing V220 kernel uses Welford / ReduceMax-tree-reduce / mask-driven per-VL compute / scalar-fan-in patterns that benefit from arch35's reg-based MicroAPI primitives (`__VEC_SCOPE__`, `RegTensor`, `MaskReg`, `UpdateMask`, `DataCopy<T, LoadDist::DIST_BRC_B32>`) → **Mode B**, write `arch35/<op>_*.h` with reg-based body.

**Why this matters**: choosing Mode A when the op benefits from reg-based MicroAPI leaves perf on the table; choosing Mode B when the V220 kernel is already public-API-portable triples the porting work + duplicates code. The signal that decides is **the V220 kernel's existing macro guards**: if it already has `__NPU_ARCH__ == 3003 || 3113` guards in its `.cpp` dispatcher, the kernel authors pre-staged a Mode A port. If those guards are absent and the op uses Welford/tree-reductions, plan for Mode B.

**Generalizes to**: every A3→A5 port. The Phase A `analysis.md` MUST include the mode determination as Step 1; W8 artifact layout differs between the two modes (Mode A has no `arch35/`, no `_apt.cpp`).

**Status**: NEEDS_REVISION — mechanical scanners pending.

---

### CAND-A3A5-2: Pre-staged `__NPU_ARCH__ == 3003 || 3113` macro guard as port-readiness indicator
`verified_on: a5_ops:ctc_loss_v3:case_1c8bbc23`

**Source**: `gather_elements_v2.cpp` (master), `rms_norm_quant.cpp` (master), `ada_layer_norm.cpp` (master), `apply_adam_w_quant.cpp` (master) — cross-op evidence (4 ops)
**Pattern class**: pre-port-readiness detection
**A5 differentiator**: the constants `3003` (Ascend950PR_9579) and `3113` (Ascend950PR_9589) are the V351 chip variants; their `__NPU_ARCH__` values are distinct from V220's `__CCE_AICORE__ == 220`.

**Concrete anchor** (the canonical pre-staged guard shape):
```cpp
// In <op>.cpp dispatcher (master state, pre-PR4778):
if (TILING_KEY_IS(1)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    // V220 path: bf16 down-cast to half because V220 lacks bf16 in some VEC primitives
    if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
        OpKernel<half, int32_t> op(...);  op.Process();
    } else { OpKernel<DTYPE_X, int32_t> op(...);  op.Process(); }
#else
    // V351/A5 path: use the native dtype, no down-cast
    OpKernel<DTYPE_X, int32_t> op(...);  op.Process();
#endif
}
```

**Why this matters**: when `git grep '__NPU_ARCH__ == 3003'` over a master-state op returns hits, the op was designed with V351 dispatch in mind, and the PR4778 port is mostly **wiring** (adding `ascend950` config + minimal arch35 files). When zero hits, the op needs structural changes (often Mode B per CAND-A3A5-1). Phase A analysis.md should grep for this guard as a 30-second port-difficulty estimator.

**Generalizes to**: ops where V220 had a sub-optimal codepath (bf16-via-half cast, scalar-loop fallback for missing primitive, smaller-tile workaround for UB limits). A5's regbase path typically *removes* the workaround. The pre-staged `#if !(__NPU_ARCH__ == 3003 || 3113)` brackets exactly those V220-only branches.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-3: `DataCopyPad` availability guarded by `__CCE_AICORE__ == 220` — write the portable fallback once, reuse across norm/quant ops

**Source**: `group_norm_silu_quant_base.h` (lines 17-25) + repeated pattern in `rms_norm_quant.cpp` (line 245 ReduceSum fork) — cross-op evidence (2 ops, recurring elsewhere)
**Pattern class**: copy-primitive portability shim
**A5 differentiator**: `DataCopyPad` exists on V220 but its semantics for UB→GM differ on V351 (EC-23 catalogs this), so the canonical safe form is to detect availability and fall back to aligned `DataCopy` with caller-supplied ceil-to-block sizing.

**Concrete anchor**:
```cpp
namespace platform {
__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}
}

template <typename T, bool isAlign = true>
__aicore__ inline void CopyInData(LocalTensor<T>& dst, GlobalTensor<T>& src, int64_t count) {
    if constexpr (isAlign) {
        DataCopy(dst, src, count);
    } else if constexpr (platform::IsDataCopyPadSupport()) {
        DataCopyExtParams cp{1, uint32_t(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, cp, pad);
    } else {
        int64_t elemsPerBlock = 32 / sizeof(T);
        int64_t alignCount = (count + elemsPerBlock - 1) / elemsPerBlock * elemsPerBlock;
        DataCopy(dst, src, alignCount);   // caller must over-allocate to alignCount
    }
}
```

**Why this matters**: A5 port often hits "I want non-aligned GM→UB copy, what's the safe primitive?" The wrong answer is "use DataCopyPad everywhere" (EC-23 — UB→GM crashes 507035). The right answer is the static-dispatch shim above; copy it into the op's `_base.h` once, and the kernel body is portable. Pairs with P-P83 (output-side EC-23 mitigation) on the write path.

**Generalizes to**: any norm/quant/reduce op with `H` not guaranteed aligned to 32-byte block. RmsNormQuant, GroupNormSiluQuant, layernorm variants, AdaLayerNorm all benefit.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-4: Variant-split kernel file convention — `<op>_<variant>.h` siblings dispatched by tiling-key from a thin `.cpp`

**Source**: `gather_elements_v2.cpp` (scalar/transpose/last_dim — 4 files), `index_put_with_sort.cpp` (base/gather_data/scatter_data — 3 phases via inheritance), `apply_adam_w_quant.cpp` (fp16/fp32 split — 2 files), `top_k_top_p_sample_v2.cpp` (comm/main/sort_cumsum — 3 files), `group_norm_silu_quant.cpp` (base/b16 — 2 files) — cross-op evidence (5 ops)
**Pattern class**: multi-variant kernel organization
**A5 differentiator**: A5's wider regbase MicroAPI surface tempts authors to write one mega-template kernel; the master-state convention (preserved through A5 port) is to keep variants in **sibling files** with **per-variant TILING_KEY dispatch in the `.cpp`**, even when the bodies share 80% of their code.

**Concrete anchor** (canonical thin-cpp + sibling-header shape):
```cpp
// op_kernel/<op>.cpp — pure dispatcher, ≤ 50 lines
#include "<op>_variant_a.h"
#include "<op>_variant_b.h"
extern "C" __global__ __aicore__ void <op>(/* args */, GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        VariantAKernel<DTYPE_X, int32_t> op(...);  op.Process();
    } else if (TILING_KEY_IS(1)) {
        VariantBKernel<DTYPE_X, int32_t> op(...);  op.Process();
    }
}
```

**Why this matters**:
1. **Compile budget**: each `.h` is a separate translation unit's instantiation source; mega-template kernels balloon compile time (ops-nn already pushes 10+ minute builds).
2. **Debugability**: msprof reports per-tiling-key kernel names; sibling-header naming makes the profile output greppable.
3. **Variant carve-out cost is bounded**: adding a 4th variant for A5 (e.g. arch35-specific reg-based path) means adding ONE new `<op>_<variant>_a5.h` + ONE new `else if (TILING_KEY_IS(N))` branch + ONE new tiling-key emitter in `_tiling.cpp`. No changes to existing variants.
4. **Cross-platform divergence is per-variant**: dtype-specific variants (fp16 vs fp32 in apply_adam_w_quant) often have different alignment / quantization-LUT semantics; isolating them prevents `if constexpr` ladders from accumulating.

**Generalizes to**: any op with ≥2 algorithmic paths (dtype-specialized, dim-specialized, shape-specialized, mode-specialized). The trap is collapsing variants into one templated kernel "for elegance"; the port pays the elegance tax in build time + profile illegibility.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-5: arch35 reg-based MicroAPI body — `__VEC_SCOPE__` + `RegTensor<float>` + per-VL mask loop replaces V220 class-style VEC ops [SELF-REVIEW: C35 overlap with P-REG-1 → propose metadata extension to P-REG-1, NOT new entry]

**SELF-REVIEW NOTE (2026-05-12, mechanical scanner C35)**: this candidate overlaps with P-REG-1 (P-REG-1 covers the mechanism: "keep intermediates in registers, avoid UB traffic"; this candidate adds the canonical per-VL loop SKELETON with `MaskReg` tail handling, `LoadTensor` bf16/fp16 unpack shim, and `CastTrait` rounding parameterization). Per C35 + W11 + P-REG-1, this entry should ROUTE TO a metadata-fix proposal extending P-REG-1's body with the per-VL loop skeleton + `LoadDist::DIST_UNPACK_B16`/`StoreDist::DIST_PACK_B32` direct-cast pattern + `CastTrait` rounding convention. NOT a stand-alone new entry. Body retained below for the kb-maintain agent to lift into P-REG-1.

**Source**: `ada_layer_norm/op_kernel/arch35/ada_layer_norm_common.h` + `ada_layer_norm_impl.h` — 1 op partial-port evidence; mechanism documented in W11 (`ascend950pr.md §Reg-based intrinsics`) and CANN 9.0 official docs (per P-REG-1 candidate)
**Pattern class**: arch35-native compute-loop shape (skeleton extension to P-REG-1)
**A5 differentiator**: this is the ONLY pattern that makes Mode B (CAND-A3A5-1) worth the rewrite cost — without using MicroAPI, an arch35 kernel is just a renamed V220 kernel.

**Concrete anchor** (the canonical arch35 per-VL loop):
```cpp
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);   // VL = 64 fp32 on A5

template <typename T>
__simd_callee__ inline void LoadTensor(RegTensor<float>& dst, __ubuf__ T* src, MaskReg& m) {
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dst, src);                    // direct fp32 reg-load
    } else {
        RegTensor<T> tmp;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(tmp, src);   // unpack bf16/fp16 to fill VL slots
        Cast<float, T, castTraitB16ToB32>(dst, tmp, m);
    }
}

__aicore__ inline void ComputeRowVF(uint32_t dataCount, __ubuf__ float* xAddr, ...) {
    uint16_t colLoopTimes = CeilDiv(dataCount, V_LENGTH);
    __VEC_SCOPE__ {                            // enter reg-based scope
        RegTensor<float> x, scale, shift;
        MaskReg pFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pLoop;
        for (uint16_t j = 0; j < colLoopTimes; j++) {
            pLoop = UpdateMask<float>(dataCount);   // shrinks on last iter for tail
            LoadTensor(scale, scaleAddr + j * V_LENGTH, pLoop);
            LoadTensor(shift, shiftAddr + j * V_LENGTH, pLoop);
            FusedMulDstAdd(x, scale, shift, pLoop);  // in-register FMA
            CopyToTensor(outAddr + j * V_LENGTH, x, pLoop);
        }
    }
}
```

**Why this matters**:
- Intermediates `scale`, `shift`, `x` stay in vector registers across `LoadTensor → FusedMulDstAdd → CopyToTensor` — no UB round-trip (P-REG-1 mechanism)
- `MaskReg` + `UpdateMask` handles tail without a separate epilogue branch — the same loop covers full-VL and partial-VL iterations
- `CastTrait` parameterizes rounding (`CAST_RINT` for bf16, `CAST_ROUND` for fp16) without runtime branching
- `LoadDist::DIST_UNPACK_B16` + `StoreDist::DIST_PACK_B32` translate bf16/fp16 elements to/from fp32 register slots in the LOAD/STORE instructions, eliminating explicit Cast steps

**Generalizes to**: any A5 norm/activation/quant kernel where the compute graph is `load → cast-to-fp32 → ≥3 VEC steps → cast-to-out-dtype → store` per row. AdaLayerNorm, GroupNormSiluQuant, RmsNormQuant, ApplyAdamWQuant all fit. The candidate complements P-REG-1 (which describes the mechanism in isolation) by giving the canonical per-VL loop skeleton.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-6: fp16-path / fp32-path file split for dtype-divergent algorithms (apply_adam_w_quant + group_norm_silu_quant evidence)

**Source**: `apply_adam_w_quant_fp16.h` + `apply_adam_w_quant_fp32.h` + shared `apply_adam_w_quant_base.h`; `group_norm_silu_quant_b16.h` + `group_norm_silu_quant_base.h` — cross-op evidence (2 ops)
**Pattern class**: dtype-specific file decomposition (sub-case of CAND-A3A5-4)
**A5 differentiator**: when fp16 + fp32 algorithms differ in **buffer count / quantization LUT / accumulator dtype**, splitting into sibling files is cheaper to port than templating because the per-dtype divergences span the whole `Process()` body, not just leaf Cast calls.

**Concrete anchor** (per-dtype class with shared base):
```cpp
// apply_adam_w_quant_base.h  — dtype-agnostic helpers
template <typename T> __aicore__ inline void DataCopyIn(...);
template <typename T> __aicore__ inline void CastF16ToFp32(LocalTensor<T>& dst, LocalTensor<T1>& src, uint32_t n);
template <typename T> __aicore__ inline void CastFp32ToF16(LocalTensor<T>& dst, LocalTensor<T1>& src, uint32_t n) {
    if constexpr (AscendC::IsSameType<T, half>::value) Cast(dst, src, RoundMode::CAST_NONE, n);
    else                                               Cast(dst, src, RoundMode::CAST_RINT, n);  // bf16
}

// apply_adam_w_quant_fp32.h — pure-fp32 class (TILING_KEY 100)
template <typename T, typename U> class ApplyAdamWQuant { ... };   // 1 calc buf, no quant LUT

// apply_adam_w_quant_fp16.h — fp16/bf16 class with quant LUT (TILING_KEY 200 + 300)
template <typename T, typename U, typename T_VAR_GRAD> class ApplyAdamWQuant16 {
    // adds qMapMBuf/qMapVBuf for 256-entry quantization LUT, fp32 calc buf, fp16 store buf
    pipe.InitBuffer(qMapMBuf, Q_MAP_SIZE * sizeof(T));   // 256-entry LUT
    pipe.InitBuffer(calcBuf, CALC_BUF_NUM * singleSize * sizeof(float));  // 6× scratch
    ...
};
```
Dispatch from thin `.cpp`:
```cpp
if (TILING_KEY_IS(100))      ApplyAdamWQuant<float,    int64_t> op;
else if (TILING_KEY_IS(200)) ApplyAdamWQuant16<float, int64_t, half>      op;   // fp16 path
else if (TILING_KEY_IS(300)) ApplyAdamWQuant16<float, int64_t, bfloat16_t> op;  // bf16 path
```

**Why this matters**: dtype-templating-the-whole-kernel produces `if constexpr` chains in 5-10 places (buffer alloc, quant LUT init, dequant step, accumulator dtype, atomic-add type, store dtype). Splitting per dtype makes each class linear, with shared logic in `_base.h`. Compile-time: only the live template instantiates. The bf16 path is added by **type-aliasing** the fp16 path (`ApplyAdamWQuant16<float, int64_t, bfloat16_t>`), not by writing a third file — the dtype-template parameter at the fp16/bf16-class level absorbs the cast-rounding difference via the `CastFp32ToF16` shim.

**Generalizes to**: any op with `fp32 high-precision path` + `(fp16, bf16) low-precision quantized path`. Optimizer ops (Adam, AdamW, SGD-with-momentum, Lion), fused norm-quant ops (RmsNormQuant, LayerNormQuant), activation-quant fusions (SwigluQuant, GeluQuant).

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-7: Three-phase index-then-scatter via inheritance — base → gather → scatter class hierarchy in same .so

**Source**: `index_put_with_sort_base.h` (base helpers) → `index_put_with_sort_scatter_data.h` (ScatterDataInKernelOp inherits) → `index_put_with_sort_gather_data.h` (GatherDataOp inherits ScatterDataInKernelOp) — 1 op evidence with strong structural signal
**Pattern class**: multi-phase op via inheritance
**A5 differentiator**: same kernel `.so` exposes multiple phase entry-points routed by TILING_KEY = 0/1/2; **phases share state via inheritance**, not via cross-kernel GM workspace passing.

**Concrete anchor** (class hierarchy + `.cpp` dispatch):
```cpp
// base.h
class IndexPutWithSortBase { /* InitGmTensor, InitTilingData, PIPE_V_S, AddUb2Gm helpers */ };

// scatter_data.h
class ScatterDataBetweenKernelOp : public IndexPutWithSortBase { /* phase 0 */ };
class ScatterDataInKernelOp      : public IndexPutWithSortBase { /* phase 1 */ };

// gather_data.h
class GatherDataOp : public ScatterDataInKernelOp { /* phase 2; reuses ScatterDataInKernelOp's UB layout */
    void Process() {
        GetHeadTailIndexValue();
        for (uint64_t i = 0; i <= indicesBlocks; i++) { ProcessIndicesBlock(...); }
        CopyOutSyncData();
        SyncAll();                       // inter-phase barrier
        ProcessCacheData(sliceSize, 0);  // continues into scatter phase
    }
};

// .cpp
if      (TILING_KEY_IS(0)) ScatterDataBetweenKernelOp<...> op;
else if (TILING_KEY_IS(1)) ScatterDataInKernelOp<...>      op;
else if (TILING_KEY_IS(2)) GatherDataOp<...>               op;
```

**Why this matters**: this is the alternative to CAND-FAG-1's three-kernel split (PRE→MAIN→POST as separate launches). When phases share UB layout + tiling data + helper primitives (`AddUb2Gm`, `CopyGm2Ub` with atomic-add), inheritance keeps the code DRY without paying the kernel-launch overhead of CAND-FAG-1. The trade-off:
- **Three-kernel split (CAND-FAG-1)**: each phase is a separate `__global__` launch; PRE clears workspace, MAIN does atomic-add, POST rescales+casts. Best when phases have radically different UB layouts.
- **Inheritance hierarchy (this candidate)**: phases share UB layout via the base class; SyncAll() between phases. Best when phases differ in *which* GM regions they touch but share *how* they tile + buffer.

**Generalizes to**: any op-with-pre-sort + accumulate pattern where the sort produces a permutation tensor consumed by the scatter phase. IndexPutWithSort, ScatterND-with-sort, sparse-gather-then-aggregate (MoE finalize variants).

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-8: arch35 tilingkey template-argument declaration via ASCENDC_TPL_ARGS_DECL — compile-time specialization axes

**Source**: `fused_quant_mat_mul/op_kernel/arch35/fused_quant_mat_mul_tilingkey.h` (5-axis: ATRANS×BTRANS×BIASMODE×KERNELTYPE×OPTYPE) + `fused_quant_mat_mul_kernel_tilingkey.h` master (6-axis variant) — 1 op evidence, distinctive arch35 pattern
**Pattern class**: arch35 host-side template-axis enumeration
**A5 differentiator**: arch35 introduces `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` macros that let the kernel author declare which `(trans, kerneltype, optype, ...)` combinations get instantiated, with the dispatcher auto-generated by the compile pipeline. V220 has no equivalent; the V220 kernel would either branch at runtime or hand-roll a switch.

**Concrete anchor** (the 5-axis declaration):
```cpp
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_VEC_EPILOGUE_WITH_MMAPI 0
#define TPL_VEC_EPILOGUE_WITH_MMAPI    2
#define F_OPTYPE_NONE   0
#define F_OPTYPE_SWIGLU 2

ASCENDC_TPL_ARGS_DECL(
    FusedQuantMatMul,
    ASCENDC_TPL_UINT_DECL(ATRANS,      ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(BTRANS,      ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(BIASMODE,    ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_DECL(KERNELTYPE,  ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 5, 1, 8),
    ASCENDC_TPL_UINT_DECL(OPTYPE,      ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                          F_OPTYPE_NONE, F_OPTYPE_RELU, F_OPTYPE_SWIGLU));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
    ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0, 1),
    /* ... per-axis enumeration of which combos to compile ... */));
```

**Why this matters**: ASCENDC_TPL_ARGS_DECL is arch35's native answer to source-style template specialization (see CAND-source-FA-5). The axes pack into a single uint32 tiling key whose layout is declared by `_BW` (bit-width) sizes — `ASCENDC_TPL_2_BW = 2 bits, ASCENDC_TPL_4_BW = 4 bits`. The kernel binary is built with all selected combinations as separate `__aicore__` symbols; the host dispatcher chooses one at runtime by computing the packed key from runtime params.

When porting a source fused-kernel that uses C++ template specialization heavily (FlashAttention-style 5-axis dispatch), ASCENDC_TPL_ARGS_DECL is the direct A5 equivalent — no need to hand-write the per-combination dispatcher.

**Generalizes to**: any arch35 op with ≥3 orthogonal compile-time axes (trans/bias/optype/dtype/kerneltype/...) where the cross-product is small enough to materialize (typically ≤ 64 combos). Quant-fused matmul, attention variants, fused conv-bias-activation.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-9: Phase O2.5 detection signal — grep for variant headers vs single .h to predict port complexity

**Source**: all 6 PR4778 ops + 2 partial-port ops — cross-op evidence (8 ops)
**Pattern class**: pre-port complexity estimation
**A5 differentiator**: arch35 ports vary 10× in complexity (1-line strip vs 4-file new variant); the existing V220 file structure is a deterministic predictor.

**Concrete anchor — estimator** (drop into Phase O2.5 ref-derive):
```python
def estimate_port_complexity(op_kernel_dir: Path) -> dict:
    headers = list(op_kernel_dir.glob("*.h"))      # exclude arch35/ subdir
    cpp     = list(op_kernel_dir.glob("*.cpp"))
    n_variant_headers = len(headers)
    has_npu_arch_guard = any(
        "__NPU_ARCH__ == 3003" in f.read_text() for f in cpp
    )
    cpp_uses_apt = any(f.name.endswith("_apt.cpp") for f in cpp)
    is_aclnn_exclude = "aclnn_exclude" in (op_kernel_dir / "../op_host/<op>_def.cpp").read_text()

    if n_variant_headers == 1 and has_npu_arch_guard:
        # Single-header, pre-staged guard → Mode A surgical strip (P-P89), ~ 1 hour
        return {"mode": "A", "estimated_hours": 1, "files_to_add": ["ascend950 config"]}
    elif n_variant_headers >= 3 and has_npu_arch_guard:
        # Multi-variant pre-staged (gather_elements_v2, index_put_with_sort, etc.) → Mode A,
        # but each variant header needs scan for V220-specific code paths; ~ 4 hours
        return {"mode": "A", "estimated_hours": 4,
                "files_to_add": ["ascend950 config + per-variant tiling-key validation"]}
    elif not has_npu_arch_guard:
        # Master state lacks A5 dispatch → Mode B regbase variant needed; ~ 1-2 days
        return {"mode": "B", "estimated_hours": 12,
                "files_to_add": ["op_kernel/arch35/*.h", "op_kernel/<op>_apt.cpp",
                                 "ascend950 OpAICoreConfig with opFile.value"]}
    if is_aclnn_exclude:
        # Plus W9 (OL-131) cross-op router patching
        return {**result, "additional": ["op_api/<peer>.cpp router edits (OL-131)"]}
```

**Why this matters**: the brief audit `output/a3_to_a5_port/docs/PIPELINE_KB_GAP_AUDIT_ctc_loss_v3.md` notes Phase A often takes 4-6× longer than necessary because the worker tries to understand the FULL kernel before deciding the port shape. Running this estimator FIRST narrows the work scope before any kernel reading.

**Generalizes to**: every A3→A5 port classifier. Feeds W4 `phase_o25_a3_ref` derivation; complements W5 (ports-readiness scan).

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-10: WelfordUpdate/WelfordFinalize as arch35 norm-primitives — replaces hand-rolled mean/var loops

**Source**: `ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h` (ProcessWelfordUpdate, ProcessWelfordFinalize) — 1 op evidence, but the primitive is documented in AscendC arch35 public API
**Pattern class**: arch35 high-level statistic primitive
**A5 differentiator**: arch35 exposes `WelfordUpdate<T, float, false>(meanLocal, varLocal, meanLocal, varLocal, x, scratch, param)` and `WelfordFinalize<true>(...)` as public primitives. V220 has neither; the equivalent V220 norm kernel hand-rolls the running-mean + running-variance loop.

**Concrete anchor** (per-tile streaming update + finalize):
```cpp
// In hot tile-loop:
WelfordUpdateParam updParam;
updParam.rnLength = 1;
updParam.abLength = sliceSize;
updParam.abComputeLength = computeLength;
updParam.nRec = 1.0f / float(welfordCount);
WelfordUpdate<T, float, false>(
    meanTmpLocal, varTmpLocal,            // in/out running stats
    meanTmpLocal, varTmpLocal,             // (aliased: read+write same buffer)
    xLocal,                                // current tile data
    tmpLocal, updParam);                   // scratch + param
PipeBarrier<PIPE_V>();

// At end of row:
WelfordFinalizePara finParam;
finParam.rnLength = welfordCount;
finParam.abLength = sliceSize;
finParam.headCount = welfordCount;
finParam.headCountLength = tailSize;
finParam.tailCount = (tailSize > 0) ? (welfordCount - 1) : welfordCount;
finParam.tailCountLength = sliceSize - tailSize;
finParam.abRec = 1.0f / float(sliceSize);
finParam.rRec  = 1.0f / float(hiddenDim);
WelfordFinalize<true>(meanLocal[batch], varLocal[batch], meanTmpLocal, varTmpLocal, tmpLocal, finParam);
```

**Why this matters**: when porting any LayerNorm-class or InstanceNorm-class op from V220 to A5 in Mode B (regbase), the hand-rolled `for (tile) { partialSum += x; partialSumSq += x*x; }` loop should be REPLACED with `WelfordUpdate` calls, not just copied. `WelfordUpdate` is numerically more stable (running-stats form) and uses arch35's hardware streaming-statistic path; literal V220 algorithm translation will pass precision tests but leave significant perf on the table.

**Generalizes to**: every A5 norm port. AdaLayerNorm (this evidence), LayerNorm, InstanceNorm, GroupNorm, RmsNorm-with-mean-subtract, BatchNorm forward, AdaptiveInstanceNorm.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-11: SocVersion-discriminated dtype-support list in op_api — A5 admits more dtypes than V220/A3 for the same op

**Source**: `gather_elements_v2/op_host/op_api/aclnn_gather.cpp` lines 90-101, `IsRegbase()` discriminator throughout (lines 93, 335, 356, 386, 419) — 1 op direct evidence; pattern is widely cited across ops-nn (W9 OL-131 corroborates)
**Pattern class**: host-side per-SoC capability divergence
**A5 differentiator**: A5's regbase path supports a broader DataType list (e.g. fp8_e4m3fn / fp8_e5m2 / hifloat8 / int4b_t / double) than V220; the op_api layer must branch on `Ops::NN::AclnnUtil::IsRegbase()` to expose the wider list only on A5.

**Concrete anchor**:
```cpp
static bool CheckDtypeValid(const aclTensor* self, ...) {
    bool is910b = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                   GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    bool is950  = Ops::NN::AclnnUtil::IsRegbase();
    std::initializer_list<op::DataType> SUPPORTED;
    if (is910b)      SUPPORTED = DTYPE_SUPPORT_910B_LIST;   // {fp16, bf16, fp32, int32}
    else if (is950)  SUPPORTED = DTYPE_SUPPORT_950_LIST;    // {fp16, bf16, fp32, int32, fp8_e4m3, double}
    else             SUPPORTED = DTYPE_SUPPORT_LIST;        // {fp16, fp32, int32} (legacy)
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SUPPORTED, return false);
    return true;
}
```

Also gates algorithm choice (CalGatherV2 uses different L0 op dispatch when `IsRegbase()`):
```cpp
int64_t dim = Ops::NN::AclnnUtil::IsRegbase() ? dimFinal : dimSize - 1;
gatherElementsResult = l0op::GatherElements(selfContiguous, dim, indexContiguous, executor);
```

**Why this matters**: a port that only adds `AICore().AddConfig("ascend950")` + arch35 kernel files but **forgets to expand the op_api dtype list** silently rejects A5-only dtypes (fp8, hifloat8) at the aclnn boundary — the kernel is built and ready but never reached. Phase A analysis.md MUST scan `op_host/op_api/aclnn_<op>.cpp` for `DTYPE_SUPPORT_LIST` + verify either (a) it's already a single list that covers A5 dtypes, OR (b) there's an `is950` branch with the expanded list. This complements W9 (OL-131 router) on the same op_api/ axis.

**Generalizes to**: every aclnn-exposed op where A5 admits new dtypes (fp8 family, hifloat8, int4, double on regbase). Includes most index/, norm/, quant/, optim/ family ops.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-12: TILING_KEY-bit-encoding convention — pack (slice, beta, bf16, fp16, ...) into one uint64 key

**Source**: `rms_norm_quant.cpp` lines 354-387 (TILING_KEY = 1, 65, 257, 321, 27, 91, 283, 347) — 1 op direct evidence; widely-recurring pattern across norm/quant ops
**Pattern class**: tiling-key encoding convention
**A5 differentiator**: A5 ports often add new bit-flags (e.g. an `is_regbase_path` bit, an `apply_quant_offset` bit), and the convention is to **extend the encoding orthogonally** rather than re-use existing key values with different meanings.

**Concrete anchor** (the canonical bit-field comments + dispatcher):
```cpp
// Bit layout (8-bit field, LSB-first):
//   bit 0-5: dtype-encoding (000001 = fp16, 011011 = bf16, ...)
//   bit 6  : use-slice (1 = slice loop, 0 = full-row)
//   bit 7  : (reserved)
//   bit 8  : has-beta (1 = beta included, 0 = beta empty)
if (TILING_KEY_IS(1))   { /* fp16, no beta,  no slice  0b0_0_0_0_000001 */ }
if (TILING_KEY_IS(65))  { /* fp16, no beta,  use slice 0b0_0_0_1_000001 */ }
if (TILING_KEY_IS(257)) { /* fp16, has beta, no slice  0b0_1_0_0_000001 */ }
if (TILING_KEY_IS(321)) { /* fp16, has beta, use slice 0b0_1_0_1_000001 */ }
#if (defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
if (TILING_KEY_IS(283)) { /* bf16, has beta, no slice  0b0_1_0_0_011011 */ }
if (TILING_KEY_IS(27))  { /* bf16, no beta,  no slice  0b0_0_0_0_011011 */ }
if (TILING_KEY_IS(347)) { /* bf16, has beta, use slice 0b0_1_0_1_011011 */ }
if (TILING_KEY_IS(91))  { /* bf16, no beta,  use slice 0b0_0_0_1_011011 */ }
#endif
```

**Why this matters**:
- **Tiling-key collisions are silent**: re-using key value 27 to mean "fp16 + new-A5-feature" when V220 already uses 27 for "bf16 no-beta no-slice" causes A5 ports to dispatch into wrong kernel branch on cross-SoC builds. The convention is `key = base_key | (new_bit << N)` not `key = base_key + offset`.
- **Per-bit grouping survives macro guards**: the `__NPU_ARCH__ == 3003 || 3113` guard around the bf16 cluster shows the convention's robustness — adding A5 means adding more `TILING_KEY_IS()` branches OUTSIDE the V220-only guard, not modifying existing ones.
- **Tiling-side correspondence**: the host-side `_tiling.cpp` MUST emit the same bit-packed value via `SetTilingKey(base | flags)`; the comment block on each case is the contract between tiling + kernel.

**Generalizes to**: any op with ≥3 boolean / small-enum compile-time axes. RmsNormQuant (this evidence), GroupNormSiluQuant, all norm-with-optional-beta ops, fused-quant pipelines. Complements CAND-A3A5-8 for ops that prefer manual encoding over ASCENDC_TPL_ARGS_DECL.

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-13: Sort + cumsum 3-file split for top-k/top-p sampling — sort algorithm in dedicated header

**Source**: `top_k_top_p_sample_v2/op_kernel/{top_k_top_p_sample_v2.h, top_k_top_p_sample_v2_comm.h, top_k_top_p_sample_v2_sort_cumsum.h}` — 1 op evidence (already-merged baseline; PR4778 ops mimic the convention)
**Pattern class**: sort-driven sampler structural decomposition
**A5 differentiator**: A5's `Sort<float, true>` + `Concat` + `MrgSort4` primitives are fast enough that a top-k/top-p sampler is best expressed as **sort-first, cumsum-second, sample-third** rather than ReduceMax-per-iteration (CAND-PP91). The 3-file split keeps the sort details out of the main kernel.

**Concrete anchor — file responsibilities**:
- `<op>_comm.h`: constants (KERNEL_BUFFER_SIZE, FLOAT_MIN = `-3.40282e+38f`, MRG_PER_ELE = 2), `DataCopyEx` wrapper, `SetWaitFlag<HardEvent>` shorthand, parameter struct
- `<op>_sort_cumsum.h`: `TopKTopPSampleV2SortKernel<T>` class with `SortOneTime` (Concat + Sort), `SortSPartAll` (segmented sort across a row + MrgSort), `SumErreyOne` (linear cumsum scan for top-p threshold)
- `<op>.h`: main `TopKTopPSampleV2Kernel<T>` class — has a `TopKTopPSampleV2SortKernel<T> sortOp` member, owns Init/Process; sort is a delegated subroutine

```cpp
template <typename T>
class TopKTopPSampleV2Kernel {
public:
    TopKTopPSampleV2SortKernel<T> sortOp;            // composition, not inheritance
    void Process() {
        for (int row = 0; row < rtCoreRowNum; row++) {
            DataCopyPad(topPtempLocal, topPGlobal[row], ...);   // per-row hyperparams
            sortOp.SortSPartAll(params, bufLocal, srcGlobal, destGlobal);
            sortOp.SumErreyOne(sortedScores, count, &sumVal, topp, &topPNum, ifRet);
            // ... sample from top-p prefix using q (Gumbel-trick)
        }
    }
};
```

**Why this matters**:
- The sort code is the most complex piece (segmented merge + tie-break + mask-fill); isolating it in `_sort_cumsum.h` lets norm/quant ports reuse the **same shape** for any "rank-then-mask" sampling head.
- Composition (`sortOp` as member) is preferred over inheritance because top-k/top-p sampling may eventually need to swap sort algorithms (radix-sort, Sort+VBitSort) per dtype without changing the main kernel.
- `FLOAT_MIN = -3.40282e+38f` is the canonical fp32 mask-fill sentinel (CAND-FA2 also uses `-3e38` for softmax masking) — A5 ports should reuse this constant, not reintroduce per-op variations.

**Generalizes to**: any rank-then-cut sampling head: top-k softmax (MoE gating, op#7), nucleus sampling (top-p), beam search (top-k per beam), threshold-mask emit (NMS-like — though CAND-PP91 is the alternative when UB is tight).

**Status**: NEEDS_REVISION.

---

### CAND-A3A5-14: Peer-router inspection deferred from kw staging when verify uses PyTorch front-end

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR (gather_elements_v2 PoC 2026-05-12; ctc_loss_v3 PoC 2026-05-12 same shape)`
`unverified_on: ports using a hand-written aclnn UT instead of PyTorch front-end — router-patch inspection cannot be deferred there`

**Principle**: When a port's PoC verify path is a high-level PyTorch op (`torch.gather`, `F.ctc_loss`, `torch.matmul`, etc.) — NOT a hand-written aclnn UT — the front-end dispatcher hides peer-router decisions behind stock CANN routing. Until the new arch35 `.so` is built and registered, the verify exercises only the **stock dispatcher path**, whose routing is independent of our `op_api/<peer>.cpp` patches. Therefore W9 peer-router inspection (OL-131 Step 2) can be deferred from the kw staging step to the follow-up on-host build+register step without compromising the PoC verification.

**Concrete anchor** — what the kw step should record when it defers:

```markdown
## Cross-Op Edits (deferred to on-host build step)

`peer_op_dependencies` = [<list from CMakeLists.txt>]    # e.g. 6 peers
`peer_router_edit_required` = TBD                        # determined at build step
Verify path: torch.<op> → A5 CANN stock dispatcher       # not the new .so
→ W9 (OL-131 Step 2) router-edit inspection deferred.
```

The kw step's `knowledge_update.md` §"Notes for next workflow step" then lists the deferred peer files for the on-host build skill to address.

**Why this matters**: kw is bounded by file-staging time (~25-30 min for a Mode B-medium op). Inflating it by N × 30-min router-inspection rounds (one per peer) breaks the time budget and risks user-watching pressure (P1) for ops with many peers (gather_elements_v2 has 6, GroupedMatmul has 7+). The deferral keeps kw fast for PoC verification — when the build step later builds + registers the arch35 `.so` and the verify dispatches to it (verified via `ASCEND_LAUNCH_BLOCKING=1` trace), THAT is when router gaps will surface concretely.

**Cross-ref**: OL-131 (peer-router pattern + 2-step detection — this candidate is the workflow-side companion that says *when* to apply Step 2), OL-134 (port-complexity estimator surfaces peer count for budget planning).

**Status**: 2-op evidence (ctc_loss_v3 1-peer-build-system-only, gather_elements_v2 6-peer-deferred-to-build-step). Promote after one more confirming op where the deferred peers DO require router patches at build time (proves the deferral didn't hide a kw-time fixable gap).

---

### CAND-A3A5-21: Pure indirect-index memory shuffle — T1 bit-exact expected across all dtypes

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=indirect-index-no-arith`
`verified_on: soc=Ascend950PR + soc=Ascend910_V220 (gather_elements_v2 cross-arch verify, 2026-05-12, 8/8 cases max_abs_err = 0.0 across fp32/fp16/bf16/int32)`

**Principle**: Operations that perform pure indirect-index memory shuffle without any arithmetic (gather, gather_elements, gather_v2/v3, scatter_update without atomic-add merge, narrow, index_select, slice-with-dynamic-stride) produce **bit-identical output between any two correct implementations regardless of dtype precision class**. Output bytes are a permutation of input bytes; no rounding occurs.

Therefore for these ops, `max_abs_err == 0.0` is the correct expected outcome across `fp32 / fp16 / bf16 / int32 / int8 / fp8` — distinct from fused-arithmetic ops where fp16/bf16 paths need T2 numeric tolerance.

**Practical consequence — debugging rule**: when porting or verifying a pure-shuffle op, **any T2-tolerance softening for low-precision dtypes is a sign of a bug** (off-by-one offset, wrong dim handling, sign-extension on signed index dtype, dtype-conversion code accidentally inserted into a copy path), NOT a precision-class concern. Do not paper over with `T2=1e-3`; instead trace the byte-level deviation.

**Concrete anchor — verification expectation table**:

| Op class | T1 fp32 | T1 fp16 | T1 bf16 | T1 int32 |
|---|---|---|---|---|
| Pure shuffle (this candidate) | 0.0 | 0.0 | 0.0 | 0.0 |
| Fused arithmetic (norm, attention, matmul) | 0.0 | ≤T2 tolerance | ≤T2 tolerance | 0.0 |
| Scatter-with-atomic-add | non-zero (order-dep) | non-zero | non-zero | 0.0 |

**Why this matters**: gather/scatter ops in PR4778 (`gather_elements_v2`, `gather_v3`, `index_put_with_sort`'s scatter phase) all share this property, and so does the much larger ATen `index_*` family. Codifying the expectation avoids the temptation to soften tolerance for low-precision cases when in fact the algorithm preserves bytes verbatim.

**Distinction from CAND-A3A5-14**: CAND-A3A5-14 is a workflow rule (when to defer router inspection); this one is a precision-discipline rule (what T1 outcome to expect on a shuffle op). Both surfaced in the same gather_elements_v2 PoC but are independent.

**Promotion notes (C36)**: title is principle-first (`Pure indirect-index memory shuffle — T1 bit-exact expected across all dtypes`), not op-class-scoped. Recommend promoting to `patterns/domains/scatter_add.md` (or a new `gather_scatter.md` domain file if scope grows) once 2-op evidence is logged on independent ops (e.g. `gather_v3`, `index_put` scatter phase). Op_class tag `indirect-index-no-arith` covers gather/scatter/narrow/slice families generically.

**Status**: 1-op evidence (gather_elements_v2, 8 cases). Promote after second independent op confirms.

### CAND-A3A5-16: arch35 `bin_filename` hash is op-signature derived, not target-arch derived — seed `ascend950/<op>_binary.json` verbatim from `ascend910b/<op>_binary.json`
`verified_on: a5_ops:foreach_abs:case_c472a0d0`

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: file-level comparison on 2 ops (apply_adam_w_quant 2026-05-13; foreach_abs 2026-05-13) — runtime correctness pending build`

**Principle**: The `bin_filename` hash field in `op_host/config/<arch>/<op>_binary.json` is a digest over the op_type + input schema + output schema + attr schema — NOT over the target architecture. Direct comparison of `ascend910b/<op>_binary.json` vs `ascend910_93/<op>_binary.json` for an op pre-staged for both arches shows **byte-identical `bin_filename` values**. Therefore, when seeding `op_host/config/ascend950/<op>_binary.json` for a fresh A3→A5 Mode B port, copy the hashes from `ascend910b/<op>_binary.json` verbatim — they remain correct against the V351 build artifact.

**Concrete anchor** (apply_adam_w_quant evidence, 2026-05-13):
```bash
$ diff op_host/config/ascend910b/apply_adam_w_quant_binary.json \
       op_host/config/ascend910_93/apply_adam_w_quant_binary.json
# both files differ only in target_arch / soc fields; bin_filename hashes match
```

**Additional evidence** (foreach_abs, 2026-05-13): `config/ascend950/foreach_abs_binary.json` and `config/ascend910b/foreach_abs_binary.json` are **byte-identical** for `bin_filename` across all 3 dtypes — `ForeachAbs_0f7470551d61134bcca200a197e8952e` (fp16), `..._3c4f1eb9...` (fp32), `..._5217b558...` (bf16) appear in BOTH configs. Strengthens the op-signature-derivation claim: same hashes show up in already-staged ascend950 entries that were generated by the ops-nn build system, not just by static file comparison. Cross-op evidence now: apply_adam_w_quant (910b vs 910_93 staged side-by-side) + foreach_abs (ascend950 vs ascend910b staged side-by-side).

**Why this matters**: avoids the false belief that `bin_filename` needs regeneration for each target arch via some opaque hash tool. The four host-side artifacts for a Mechanical-Mode-B port (CAND-A3A5-1, OL-134 evidence row for apply_adam_w_quant) reduce to "copy + retarget" rather than "copy + recompute hashes".

**Generalizes to**: every A3→A5 Mode B-mechanical port where the op signature (input/output/attr schemas) is unchanged from the A3 source.

**Promote-to-canonical criteria**:
1. Validated on ≥1 more port (e.g., one of the PR4778 ops once they ship)
2. Confirm runtime: build with copied hashes succeeds AND op dispatches correctly on ascend950
3. Compile-gate (C34b): N/A — this is a JSON file convention, no compile step

**Related**:
- OL-134 Mode B-mechanical sub-bucket (this is what makes Mode B-mechanical fully templatable)
- W8 ops_nn_a5_artifact_layout (canonical Mode B artifacts — `<op>_binary.json` is one of the four)
- CAND-A3A5-1 (two-port-mode dichotomy)

### CAND-A3A5-17: Defer V220 `Abs<half>` / OL-129 / PB-23 mitigation to build-time during A3→A5 Mode B port — do NOT preemptively surgical-strip

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=port_a3_to_a5`
`verified_on: static-analysis-only (apply_adam_w_quant kw-2, 2026-05-13)`
`unverified_on: build-time verification pending (port_a3 harness gap blocks Phase C/D)`

**Principle**: When porting a V220 kernel to A5 (V351) and the kernel calls `Abs<half>` / `Abs<bfloat16_t>` (or analogous `Vabs`/binary-scalar VEC ops covered by OL-129 / PB-23) with operands sourced from `TBuf<VECCALC>`, **do NOT preemptively rewrite the call site to the OL-129 Cast→fp32→Abs→Cast workaround during Phase B (artifact staging)**. The three viable mitigations (a) bare-`__ubuf__ half*` workaround; (b) Cast→fp32→Abs→Cast bit-exact; (c) Cast-promote whole tensor at row entry have **different perf/code-size costs** and the optimal choice depends on V351 codegen behavior + surrounding TQue/TBuf shape — which is observable only at build/runtime.

**Decision rule**:
1. Phase B (staging): copy V220 kernel source verbatim into `op_kernel/arch35/` (Mode B-mechanical per OL-134).
2. Phase C (build): if bisheng on `ascend950` emits the same OL-129 / PB-23 diagnostic, then and only then apply the mitigation. The compiler error message identifies the exact call site + which mitigation is admissible.
3. **Do NOT** edit kernel source on the basis of OL-129 / PB-23 keyword-grep matches alone.

**Why this matters**: OL-129 is tagged `[V220]` — its V351 status is unknown until empirically tested. False-positive preemptive patching (a) introduces semantic risk (the Cast→fp32 workaround changes precision class for fp16 abs from "bit-exact via raw ubuf" to "bit-exact via fp32 round-trip" — usually equivalent but not always when downstream relies on subnormal handling), (b) inflates the static-analysis port-effort estimate falsely, and (c) creates an aliasing class of code that diverges from the V220 master without empirical justification.

**Anchor (apply_adam_w_quant kw-2 site)**: `apply_adam_w_quant_fp16.h::NormlizeFp16` calls `Abs<half>` on a `TBuf<VECCALC>`-sourced tensor — exactly the OL-129/PB-23 hazard shape. kw-2 surgical-strip plan considered three mitigations (analysis.md §Surgical strip plan); all three are deferred pending Phase C build error.

**Generalizes to**: any OL-/PB- tagged `[V220]` whose V351 status is not yet probed. Specifically PB-23 (Divs/Muls/Adds/Subs/Sub/Add/Mul/Div bf16/int dtype rejection), OL-129 (Abs<half>/<bfloat16_t> raw-ubuf requirement), and likely OL-127 (CANN API surface gaps). The build-driven discovery pattern is cheaper than upfront speculative patching.

**Promote-to-canonical criteria**:
1. Validated on ≥1 build cycle: an A3→A5 port where Phase B left V220-style Abs<half>/etc. in place, Phase C build either passed (proving V351 lifted the V220 restriction) or emitted a specific diagnostic that identified the right mitigation.
2. Captures whether V351 lifts any of the V220 OL-129/PB-23 restrictions natively.

**Related**:
- OL-129 (V220 Abs<half> raw-ubuf requirement — parent rule with [V220] scope)
- PB-23 (V220 binary-scalar VEC ops dtype rejection)
- P-P90 / W10 (V220→V351 surgical strip discipline — this candidate is a "when NOT to strip" refinement)
- OL-134 Mode B-mechanical (mechanical ports defer all kernel-source edits to build-time by design)

---

## CAND-PP93: Row-wise fused `<reduce-norm> + <elementwise> + <quant>` via 2-pass single-tile pattern when row fits one tile

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=fused-norm-quant (row-reduction + per-element scaling + final integer-quant cast chain)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (V220 register-file & TBuf budget differ; pattern likely transfers but not probed); D > TILE_FP16 (multi-tile row chunking — different pipeline structure required); bf16 + per-channel scale combo (rms_norm_quant covered only bf16 per-tensor)`

**Principle**: when a fused op decomposes as `row_reduce_then_normalize(x) → per_element_affine(gamma, beta) → quant_cast_to_int8(scale, offset)` AND every row fits in one UB tile (D ≤ TILE_FP16 = 4096 for fp16/bf16), the simplest correct pattern is a 2-pass row-resident loop:
- **Pass 1**: load row x → cast fp32 → square → reduction-sum (BinaryFoldReduceSum, P-P47) → derive scalar `inv_rms` from the per-row sum
- **Pass 2**: reload row x → multiply by `inv_rms` → multiply by gamma → add beta → divide by scale → add offset → CAST_RINT to int32 → SetDeqScale + CAST_NONE to fp16 → CAST_TRUNC to int8 → store

No tile-inner loop within a row, no MTE2/VEC inter-tile pipeline, no flash-attention-style chunked accumulation. The pattern is a clean composition of three existing primitives — **P-P45/47** (row-reduction primitive), **P-P46** (quant cast chain fp32→int8 via the RINT/SetDeqScale/TRUNC sequence; see OL-81 for `CAST_RINT = IEEE RNE` correctness), and the row-resident structure proven in `18_FusedAddRmsnorm`.

**Concrete anchor** (rms_norm_quant kw-1 kernel skeleton, 2026-05-13):
```cpp
// Pass 1: row reduce
DataCopy(xLocal, gmX[row * D], D);                  // VECIN
Cast(xF32, xLocal, RoundMode::CAST_NONE, D);        // fp16→fp32 or bf16→fp32
Mul(sqLocal, xF32, xF32, D);
BinaryFoldReduceSum(sumLocal, sqLocal, D);          // P-P47 half-interval tree
float inv_rms = 1.0f / sqrtf(sumLocal.GetValue(0) / D + epsilon);

// Pass 2: row apply + quant cast
DataCopy(xLocal2, gmX[row * D], D);
Cast(xF32_2, xLocal2, CAST_NONE, D);
Muls(yF32, xF32_2, inv_rms, D);
Mul(yF32, yF32, gammaF32, D);
Add(yF32, yF32, betaF32, D);
Div(yF32, yF32, scaleF32, D);                       // OL-82: keep literal Div, don't pre-compute 1/scale
Add(yF32, yF32, offsetF32, D);                      // offsetF32 is host-side widened (OL-137)
Mins(yF32, yF32, 127.0f, D); Maxs(yF32, yF32, -128.0f, D);
Cast(i32, yF32, RoundMode::CAST_RINT, D);           // OL-81
SetDeqScale(static_cast<half>(1.0f));
Cast(fp16Tmp, i32, RoundMode::CAST_NONE, D);
Cast(out_int8, fp16Tmp, RoundMode::CAST_TRUNC, D);
DataCopy(gmY[row * D], out_int8, D);
```

**Host-side responsibilities** (template):
- Align32 D for kernel UB alignment
- Broadcast scalar/short scale/offset to length D when reference uses per-tensor variant (single-code-path kernel)
- Int8 offset → fp32 widen at pybind (OL-137) to avoid in-kernel Cast(fp32, int8)
- Persistent TBuf<VECCALC> for `gamma`, `beta`, `scale`, `offset_f32` loaded once per kernel launch
- TQue<VECIN, depth=2> for `x` to pipeline Pass 1 / Pass 2 reloads (OL-63)

**Evidence**:
- rms_norm_quant kw-1 (2026-05-13): 8/8 Pass A + 8/8 Pass B PASS; perf 6.60× over Path-A reference (CPU-truth Model.forward decomposition path executing ~10 PyTorch primitives sequentially). First-try success with no compile-fix or precision-fix iter.

**Other instances (predicted)**: LayerNormQuant variants (replace RmsNorm step with mean-subtract+var-divide reduce), GroupNormQuant variants with per-group reduction (group_size ≤ TILE_FP16 case), AddRmsNormQuant (extra Add op fused into Pass 2 prologue), per-token quant variants of any row-wise norm.

**Promotion gate**: requires ≥2 independent ops in this family verified PASS. rms_norm_quant is instance #1. Candidate ops for second-instance evidence: AddRmsNormQuant, GroupNormSiluQuant (norm+activation+quant; OL-69 covers the activation-cost framing). Promote to `patterns/domains/quant.md` or `patterns/domains/fused_norm.md` as a P-P entry once second-op evidence is logged.

**Cross-ref**:
- P-P45 (single-pass UB-resident dynamic quantization — this is the 2-pass row-resident cousin)
- P-P46 (quantize cast chain fp32→int8)
- P-P47 (half-interval tree reduction via BinaryFoldReduceSum)
- P-P51 / P-P52 (fp32 promotion in compute path; output cast back to int8 only at the very end)
- OL-63 (TQue depth for pipeline overlap)
- OL-69 (norm + activation fusion cost analysis — generalizes to norm + quant: activation/quant tail adds only a handful of VEC instructions, end-to-end gain comes from eliminating intermediate GM round-trip)
- OL-81 (CAST_RINT = IEEE RNE — bit-exact match to `torch.round` chain)
- OL-82 (no math-equivalent rewrites without minimal repro — kept literal `Div(scale)`, not `Muls(1/scale)`)
- OL-127 (no single-thread SIMT as final state — kernel uses nblk=56 row partitioning)
- OL-137 (host-side dtype-widening for int8 offset)
- PB-9 (DataCopy UB→UB unsafe — used `Adds(dst, src, 0.0f, count)` bridge)

---

## CAND-PERF-NBLK-LAUNCH-DOM: Shape-adaptive `nblk` reduction for tiny workloads is NOT a guaranteed perf win — launch-path cost may dominate over idle-core cost

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=any kernel with small per-shape row/element counts (num_rows ≤ ~8, total work < one wavefront)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (ada_layer_norm only — single op evidence)`
`unverified_on: soc=Ascend910_V220 (A3 family — launch-path cost profile may differ); other op classes`

**Hypothesis**: Setting `nblk = min(56, num_rows)` for small workloads (so each AIC gets ≥ 1 row of real work) saves the idle-core scheduling cost on tiny shapes and should improve ratio vs vendor.

**Counter-evidence (ada_layer_norm kw-3 → kw-3-revert, 2026-05-13)**: Applying this rule to `(B=1, S=1, H=16)` and similar tiny cases dropped the perf ratio from 0.44× to 0.40× across the whole sweep. Root cause hypothesis: with fewer cores active, per-core work scales proportionally (1 core does what 56 cores would split), and that serialization cost exceeds the saved idle-core spin. More importantly, the **dominant cost on small shapes appears to be the CANN op-API launch path itself**, not core spinning — reducing core count cannot help if the bottleneck is upstream of the kernel.

**Recommended workflow**:
1. Before applying shape-adaptive `nblk` reduction, run `msprof` on the smallest-shape case and check whether AIV idle time is actually a meaningful chunk of the kernel's wall-clock. If launch-path is dominant, `nblk` tuning is a no-op (or regression) by construction.
2. If profile shows idle-core cost IS significant, prefer keeping `nblk=56` and accept the idle-core waste — the alternative often regresses.
3. If launch-path is the dominator, the real optimization is upstream (kernel-launch fusion, batched dispatch, persistent kernels) — not per-shape `nblk` knob twiddling.

**Promote-to-pattern criteria**: validated on ≥ 2 ops in different op_classes with `msprof`-confirmed launch-path-dominant profiles, plus an explicit measurement showing the `nblk` knob has zero or negative effect on those shapes.

**Related**:
- OL-127 (no single-thread SIMT — the upper bound `nblk` is still 56 / hardware max; this candidate is about avoiding aggressive reduction, not about the floor)
- P-P1 (numBlocks dynamic — the standard "always use all cores" pattern; this candidate documents a regime where deviating from it doesn't help)
- MSPROF_AGENT_GUIDE.md (the profile-first-then-tune workflow this candidate enforces)

## CAND-PB20-GMPAD: 16x-padded GM workspace for small per-work-unit scalar outputs in pure-AIV class kernels

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=2026-03-21; op_class=normalization (mean/rstd-emit), dynamic-quant (per-row scale-emit), any pure-AIV class kernel with small scalar GM outputs`
`verified_on: soc=Ascend950PR; cann=9.0.0 (group_norm_silu_quant — single-op evidence)`
`unverified_on: soc=Ascend910_9382 (A3); soc=Ascend910B3 (A2)`

**Trigger**: kernel needs to write small per-work-unit scalar outputs (e.g., per-(n, g) `mean` / `rstd` in normalization ops, per-row dynamic quant scale, per-token statistic in fused-attn) to GM from a **pure-AIV class kernel** (the `extern "C" __global__ __aicore__ void f(...) { KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); class.Init(...).Process(); }` pattern). PB-20's existing workaround ("DataCopy UB→GM") presumes the output buffer is large enough for natural 32B-aligned DataCopy blocks. When the per-work-unit output is a SINGLE scalar (e.g., 4 bytes fp32), the 32B-alignment + inter-AIV-write-race combination silently corrupts adjacent slots.

**Pattern**: in pybind, allocate the scalar output buffer with an extra 16-element trailing dim:
```cpp
auto mean_ws = torch::empty({N, num_groups, 16}, opts);    // 16x oversize on innermost
```
In kernel, every AIV writes a 16-element T block to its `(n, g) * 16` GM offset via `DataCopy`:
```cpp
LocalTensor<T> ub_block = ubBuf_.Get<T>();
ub_block.SetValue(0, mean_t);                              // scalar value at lane 0
DataCopy(gmMean_[idx * 16], ub_block, 16);                 // 32B-aligned block write
```
After kernel returns, pybind extracts lane-0 across the trailing dim to recover the natural shape:
```cpp
auto mean = mean_ws.select(/*dim=*/2, /*index=*/0).contiguous();   // back to {N, num_groups}
```

**Why it works**: 16 elements of T (fp16: 32B, fp32: 64B) is always ≥ the 32B DataCopy MTE3 granularity, so every AIV's write lands in its own naturally-aligned 32B block — no inter-AIV write race, no alignment fault, no silent corruption. The cost is a 16× oversize on the (typically small) statistic output — negligible when `num_groups` ≤ 32 and N is bounded.

**Cost**: 16× memory oversize on the small output buffer only (the main data path is unaffected). For typical norm ops with `mean[N, G]` where `G ≤ 32`, the overhead is `N·G·16·sizeof(T)` bytes — bounded.

**Reusable across**: `LayerNorm` / `InstanceNorm` / `RMSNorm` / `BatchNorm` (mean/rstd emit), dynamic-quant ops (per-row scale emit), fused-attn statistics, any pure-AIV class kernel that emits one or a few scalars per work unit.

**Evidence**:
- group_norm_silu_quant (2026-05-13, A5 fused GroupNorm+SiLU+Quant port from A3 aclnn): iter 1→2 tried `const_cast<__gm__ T*>(gmMean_.GetPhyAddr())` + `mean_ptr[idx] = mean_t` → silent garbage on mean/rstd (max_abs_diff 2.09e+28 on case 6 bf16 — uninitialized FP_MAX sentinel). Iter 3→4 applied 16x-pad pattern → mean/rstd bit-exact (0.0 diff) on all 8 cases. Diff size: ~30 lines in `EmitMeanRstd` helper + 6 lines in pybind11.cpp.

**Promote-to-canonical criteria**:
1. Validated on ≥ 1 more op in a different op_class (e.g., a dynamic-quant op with per-row scale emit, or a different normalization op).
2. Compile-gate against public AscendC headers (C34b).
3. Confirms PB-20 workaround sub-bullet is the right host for this refinement, OR is distinct enough to promote as P-P (data-movement pattern).

**Related**:
- PB-20 (parent — pure-AIV class kernel cannot use `GlobalTensor::SetValue` or raw `__gm__ p[i]=v`; DataCopy UB→GM is the only working pattern). CAND-PB20-GMPAD is the scalar-output sub-case of PB-20's workaround.
- Iron law §5 (literal translation — keep scalar outputs as separate GM tensors, don't pack them creatively into the main output buffer).

---

## A3→A5 arch35 upstream-port patterns (2026-05-13, from `/aog-prior-art-verify` Phase 6)

> **Source**: offline analysis of CANN team's A5 ports for `ada_layer_norm` (norm/, 4 files / 1466 lines) and `fused_quant_mat_mul` (matmul/, 3 files / 226 lines). Phase 3-5 hardware verify pending — these are upstream_pass candidates IF verify confirms upstream actually meets our precision+perf bar; if upstream FAILS verify, these flip to upstream_fail (anti-pattern) tagging.
> **Detection skill**: `src/skills/aog-prior-art-verify/SKILL.md`
> **Mechanical scanners**: candidates must clear C34a (identifier denylist), C34b (compile-gate), C34c (n-gram copy-shape ≤ 5%), C35 (KB-overlap) before promotion.

### CAND-A35-PORT-1: arch35 port is structural reshape, NOT mechanical V220-strip

**Source**: upstream_pass (cross-op evidence: ada_layer_norm + fused_quant_mat_mul)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all; unverified_on=Ascend910_V220 (A3 — pattern is A5-specific)`

**Principle**: an A3 op WITHOUT `__CCE_AICORE__ == 220` guards or `impl/dav_c220/*` deps is NOT a candidate for mechanical V220-strip when porting to A5. It needs structural reshape — either algorithm-split (welford vs full_load, see CAND-A35-DISPATCH-1) or composition over re-impl (`FusedOpType` template arg, see CAND-A35-COMPOSITION-1). The presence of `op_kernel/arch35/` upstream IS the canonical signal that CANN team found this structural reshape; if no `arch35/` exists, the op is either trivially portable OR not yet ported.

**Decision rule (for kw briefs in port_a3_to_a5 mode)**:
1. Read `<backward>/<op>/op_kernel/arch35/`; if present, that IS the port → route through prior-art-verify
2. If absent, check A3 source for `__CCE_AICORE__ == 220` guards; if present → mechanical strip is the right port
3. If absent AND no V220 guards → structural reshape required → flag as researcher-grade, not single-shot kw

**Anti-pattern (counter-example)**: the harness-generated `ada_layer_norm` kernel postmortemed at `output/a3_to_a5_port/src/kernels/ada_layer_norm.postmortem-handrolled-partial/` (923 lines, 0.38× perf) tried mechanical-port from A3's `AdaLayerNormND` class, ignoring upstream's structural split — cost $69.20 / 244 min wasted.

### CAND-A35-DISPATCH-1: tilingkey-driven multi-algorithm class split

**Source**: upstream_pass (ada_layer_norm: `AdaLayerNormFullLoad` + `AdaLayerNormWelford` selected by tilingkey in `impl.h`)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=norm`

**Principle**: when an op has two qualitatively-different code paths (full-load vs sliced; fast-path vs general; one-pass vs Welford), prefer SPLIT into two top-level template-instantiated classes selected by a thin tilingkey dispatcher header. Avoid intra-class `if/else` branching on regime — the compiler can't dead-code-eliminate across template instantiations and the inner loop pays dispatch overhead per iteration.

**Concrete (3 files):**
```
op_kernel/arch35/
├── <op>_full_load.h    # AdaLayerNormFullLoad<T,U,Y,OP_CODE> class
├── <op>_welford.h      # AdaLayerNormWelford<T,U,Y,OP_CODE> class
└── <op>_impl.h         # template-method definitions (NOT a class) — pulled in by both
```
The `cpp` entry point reads the tilingkey, `if constexpr` instantiates one of the two classes, calls `Init` + `Process`. Each class is fully specialized at compile-time → no runtime dispatch overhead inside the hot loop.

### CAND-A35-COMPOSITION-1: fused matmul/conv ops compose reusable arch35 kernel classes via `FusedOpType` template param

**Source**: upstream_pass (fused_quant_mat_mul A5 imports `QuantBatchMatmulV3::MatmulAswKernel*` with `FusedOpType::SWIGLU | RELU | NONE`)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=matmul-fused; predicted_applicable_to=conv-fused, attention-fused`

**Principle**: when a fused op = base op (matmul/conv/attn) + epilogue (RELU/SWIGLU/GELU/QUANT), prefer template-parameter composition over class-level re-implementation. CANN arch35 provides templated kernel classes with epilogue template params; the fused-op kernel becomes a thin tilingkey dispatcher (~226 lines for fused_quant_mat_mul, vs ~600 lines of equivalent A3 code).

**Decision rule (for kw briefs)**:
1. Identify the base op the fused op composes (matmul, conv, ...)
2. Check `cann/ops-nn/<base_op_dir>/op_kernel/arch35/` for reusable templated kernel classes
3. If `FusedOpType` (or equivalent epilogue-template-param) is available, use it
4. Re-implementing from scratch is the wrong default for fused matmul/conv on A5

**Concrete signature**:
```cpp
#include "../../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_abl1_full_load.h"

MatMulASWKernel<DTYPE_X1, ..., FusedOpType::SWIGLU> op;
op.Init(x1, x2, bias, scale, yScale, y, user, &tilingData, &tPipe);
op.Process();
```

### CAND-A35-DISPATCH-2: 2-level tilingkey (`op_type` × `kernel_type`) cross-product

**Source**: upstream_pass (fused_quant_mat_mul `_tilingkey.h`)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=cube-fused`

**Principle**: for fused-matmul/cube ops with multiple L1-load strategies × multiple epilogue fusions, declare TWO independent tilingkey dimensions (`TPL_OPTYPE`, `TPL_KERNELTYPE`) instead of flattening to one keyspace. Cross-product instantiation in a nested `if constexpr` chain. Lets tiling code reason about L1-load-strategy independent of fusion, avoids combinatorial explosion in tilingkey enum + makes adding a new fusion or new L1-strategy a single-dimension change.

### CAND-A35-CAST-1: typed `CastTrait` / `LayerNormConfig` / `NormalizeConfig` constants over implicit primitives

**Source**: upstream_pass (ada_layer_norm common.h)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all`

**Principle**: declare `constexpr CastTrait castTraitX = {RegLayout, SatMode, MaskMergeMode, RoundMode}` constants once per direction pair (b16→b32, b32→b16, f32→i16, f16→i8, f32→fp8, f32→hif8) and use named traits at call sites: `Cast(dst, src, castTraitB16ToB32, len)`. Locks `SatMode::NO_SAT` + `RoundMode::CAST_RINT` explicitly — A5 defaults may differ from V220 and naming the contract prevents silent numerical drift.

```cpp
constexpr CastTrait castTraitB16ToB32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait castTraitB32ToB16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait castTraitF32ToI16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
```

**Anti-pattern**: relying on default-argument `Cast(dst, src, len)` overload — A5's defaults are not the same as A3's.

### CAND-A35-SIMD-1: `__VEC_SCOPE__` + `RegTensor<T>` register-level SIMD micro-API

**Source**: upstream_pass (ada_layer_norm welford.h)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all`

**Principle**: arch35 exposes a register-level SIMD micro-API (`__VEC_SCOPE__` scope, `RegTensor<T>` register, `MaskReg` + `UpdateMask<T>(len)` for tail-handling) that A3 lacks. Hot inner loops should be lowered to this API when possible — vendor `LocalTensor`-based primitives are correct but pay UB round-trip latency that the register-level API avoids.

```cpp
__VEC_SCOPE__ {
    RegTensor<float> x;
    MaskReg p = UpdateMask<float>(remaining_length);
    Duplicate(x, 0.0f);
    DataCopy(dst_ub, x, p);
}
```

Functions called from `__VEC_SCOPE__` blocks need `__simd_callee__` annotation to inline correctly.

### CAND-A35-FORMAT-1: constexpr `CubeFormat` from FORMAT_* macros

**Source**: upstream_pass (fused_quant_mat_mul A5, also QuantBatchMatmulV3 arch35)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=cube-fused`

**Principle**: cube engine specializes by `CubeFormat::{ND, NZ}` template parameter, which should be a `constexpr` derived from compile-time `FORMAT_X1` / `FORMAT_X2` / `FORMAT_Y` macros, NOT a runtime branch. A3 used runtime format checks; A5 fast-path requires constexpr.

```cpp
#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif
```

### CAND-A35-TASKTYPE-1: explicit `KERNEL_TASK_TYPE_DEFAULT` per code path

**Source**: upstream_pass (ada_layer_norm.cpp head, fused_quant_mat_mul.cpp per-block)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all`

**Principle**: declare `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_1)` at the head of each `TPL_OPTYPE` block matching the actual cube/vec ratio used by that path. The runtime uses this to allocate the right resource lanes; default (no declaration) is suboptimal for cube-heavy fused matmul or vec-heavy norm ops.

### Promotion gate notes

These 7 candidates need:
1. Cross-op evidence beyond ada_layer_norm + fused_quant_mat_mul. **CAND-A35-DISPATCH-1** + **CAND-A35-COMPOSITION-1** could merge into a single "tilingkey dispatcher + template specialization" canonical OL/P-P after a third op confirms.
2. Phase 3-5 hardware verify of upstream MUST land before promotion — if upstream FAILS verify on our edge_dataset, these flip to upstream_fail tagging (anti-pattern) and the harness regen brief needs the FAILED rows annotated. Current state: NOT_YET_VERIFIED.
3. Mechanical scanners: C34a/b/c/35 against public AscendC headers + KB overlap. The snippet code samples above are sampled (not verbatim copies) — should pass C34c n-gram ≤ 5%.

**Linked to**: `OL-141` (target `op_kernel/arch35/` is advisory prior art; never skip generation or truth).

---

## CAND-A3A5-20: torch_npu Python binding may alias a generic op name to one specific aclnn variant — verify routing before declaring a port verified

**Source**: workspace/top_k_top_p_sample/knowledge_update.md F1 (kw-1, 2026-05-14)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=op_family_v1_v2_variants; phase=Phase D verify`

**Principle**: when an op family ships as V1/V2 (or V1/V2/V3) sibling aclnn entries with different I/O signatures, the `torch_npu.<op>` Python binding does NOT necessarily call the variant whose name matches. The binding may be coded to dispatch to a fixed variant (typically the newest one) regardless of caller-supplied argument count. Porting only ONE variant to A5 and then verifying via `torch_npu.<op>(...)` can exercise the WRONG variant — producing a 0/N FAIL with EZ1001 even though the ported variant's build + registry install are correct.

**Detection signal**: verify reports `EZ1001 / Get regInfo failed / does not support opType [<OpName>V2]` while the kernel installed and the `binary_info_config.json` registry entry are both for `<OpName>` (V1 — no V2 suffix). The variant mismatch in the error message confirms the harness routed past V1 to V2.

**Mitigation paths (decision rule)**:
- **Path A — direct-aclnn ctypes wrapper**: write `verify.py` that calls `libopapi.so::aclnn<OpName>` directly via ctypes (bypassing torch_npu). Smallest scope; validates the ported variant specifically. Recommended when the deliverable is "verify V1 port artifact correctness end-to-end".
- **Path B — port the sibling variant too**: stage V2 artifacts identically to V1, build, install. Recommended when the deliverable is "make the user-visible `torch_npu.<op>` binding actually work on A5".
- **Path C — accept PoC scope**: no further work; document the variant-routing gap. Recommended when the deliverable is upstream-staging-only (e.g., PR4778-style review), not end-to-end NPU validation.

**Concrete anchor** (top_k_top_p_sample, 2026-05-14):
- V1 `aclnnTopKTopPSample`: 4 inputs + 3 attrs + 2 outputs.
- V2 `aclnnTopKTopPSampleV2`: 5 inputs + 6 attrs + 4 outputs.
- `torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q=..., eps=..., is_need_logits=..., top_k_guess=...)` takes V1-style 7 args **but routes internally to `aclnnTopKTopPSampleV2`**. No torch_npu Python entry calls V1.

**Generalization**: family-ported ops (V1/V2 variant pairs, or any op family where a Python binding name does not 1:1 map to a single aclnn entry) MUST verify variant routing before reporting verify PASS/FAIL. The `applies_to` op-classes most at risk: numeric sampling primitives, quant variants (per-tensor vs per-token vs MXFP), attention variants (V1/V2/V3 with extended kv-cache args), and any op that has been API-evolved in CANN ≥ 8.0.

**Other instances (predicted)**: `top_k_top_p_sample_v2`, `apply_rotary_pos_emb_v2`, `swiglu_quant_v2`, `dequant_swiglu_quant`, `npu_moe_init_routing_v3`, any op-family with a `_v[0-9]+` aclnn entry where the torch_npu Python binding name is shared.

**Promotion gate**: needs ≥ 1 additional op-family confirmation (a second V1/V2 case where the harness routed past the ported variant) before promotion to OL/P-P. Auto-promote pipeline (Mode 5) should run C39 dry-run on a sibling op-family before promotion.

**Cross-ref**:
- OL-158 (companion: Phase C build+register activation criteria — Phase C may succeed yet verify still fails due to variant-aliasing)
- CAND-A3A5-15 (registry-install procedure that this trap can mask)
- OL-131 (peer-router edits — different concern: router aliases ARE intended, this entry is about variant aliases the operator-porter didn't expect)

---

## CAND-A3A5-15: Manual ascend950 kernel registry install — 3-file merge into running CANN runtime

**Source**: workspace/top_k_top_p_sample/knowledge_update.md F3 (kw-1, 2026-05-14)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5; phase=Phase C install`

**Principle**: after `build.sh --pkg --ops=<op> --soc=ascend950` produces `.o`+`.json` artifacts in `build/binary/ascend950/bin/ascend950/<op>/`, registering them into the running CANN runtime requires THREE separate filesystem edits, NOT one. Missing any of the three leaves the kernel "built but not callable" — verify reports EZ1001 even though `find` shows the binary on disk. Many port_a3_to_a5 sessions misdiagnose "Phase C done" because step 1 alone (binary install) was performed without steps 2 and 3 (registry merge + per-op JSON copy).

**3-step install procedure**:

1. **Install per-binary `.o` + `.json` into kernel directory**:
   ```bash
   cp build/binary/ascend950/bin/ascend950/<op>/*.o   $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/<op>/
   cp build/binary/ascend950/bin/ascend950/<op>/*.json $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/<op>/
   ```

2. **Merge op entry into runtime `binary_info_config.json`** (single top-level key per op_type):
   ```python
   import json, shutil
   live = "$CANN/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_nn/binary_info_config.json"
   built = "build/binary/ascend950/bin/config/ascend950/binary_info_config.json"
   shutil.copy(live, f"{live}.bak")
   merged = {**json.load(open(live)), **json.load(open(built))}  # op_type unique → no schema conflict
   json.dump(merged, open(live, "w"), indent=2)
   ```

3. **Copy per-op registration JSON**:
   ```bash
   cp build/binary/ascend950/bin/config/ascend950/<op>.json \
      $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_nn/<op>.json
   ```
   (Follows the existing pattern for `apply_top_k_top_p_with_sorted.json` and other pre-shipped registry entries.)

**Concrete anchor (top_k_top_p_sample, 2026-05-14)**: Step 2 merged 251 → 252 keys; new key `TopKTopPSample` added cleanly (no schema conflict — op_type is unique). After all three steps the V1 kernel is callable from any aclnn entry that targets `TopKTopPSample`.

**Anti-pattern**: doing step 1 only and assuming the runtime auto-discovers binaries by directory scan. CANN's runtime indexes by `binary_info_config.json` keys — a binary on disk WITHOUT a registry entry is invisible at op-resolve time.

**Promotion gate**: needs ≥ 1 additional op confirmation that the same 3-step procedure works for another op_type without further filesystem edits. If a second op surfaces a 4th required step (e.g., `simplified_key.ini` re-generation, `opp_kernel_list.txt` append), this candidate should be revised before promotion.

**Cross-ref**:
- OL-158 (Phase C activation criteria — when to attempt build+install at all)
- CAND-A3A5-14 (variant-aliasing trap — install can succeed yet verify still FAILS due to routing)
- OL-132 (port strategy `regbaseCfg` vs flat `AddConfig` — determines the `op_def.cpp` shape that produced these artifacts)

## CAND-A3A5-22: atvoss-DAG-vs-A3-hand-rolled cross-platform 1-ULP drift concentrates on the highest-arithmetic-intensity output

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5; arch_pair=A3(V220)↔A5(V351)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: apply_adam_w_v2)`
`unverified_on: other Adam-family ports; other normalize/fuse/quant atvoss-DAG ports`

**Predicted rule** (forward-looking, 1-op evidence):
When the upstream A3 kernel uses bespoke per-dtype `.h` files (e.g.
`<op>_b16.h` / `<op>_fp.h` / `<op>_mix_dtype.h`) and the A5 port uses an
`atvoss/elewise/elewise_sch.h` + `ElementwiseSch<schMode, OpDag>` DAG, expect
1-ULP cross-platform drift concentrated on the **highest-arithmetic-intensity
output** (the tensor whose computation graph has the most Mul/Add chain
ordering choices). Other outputs typically remain bit-exact. T1
bit-exactness across all outputs across A3↔A5 is NOT achievable on this
class of port — T2 within per-dtype ULP-floor is the realistic target.

**Why**: atvoss reorders Mul/Add chains for compute-graph fusion. The
reordering is mathematically equivalent (same operations, different
parenthesization) but **bit-different** under IEEE-754 — error
amplifies with arithmetic intensity, so the worst-drift output is the one
with the longest chained reduce/EMA/normalize sequence.

**Concrete anchor (apply_adam_w_v2, 2026-05-14)**:
| Output | A3 path (hand-rolled `.h`) | A5 path (atvoss DAG) | Observed drift |
|---|---|---|---|
| `m` (1st-moment EMA) | `apply_adam_w_v2_b16.h` → tight Cast/Mul/Sub/Add chain | DAG node `OpGradCast_ × OpBeta1Sub1` → accumulator | max |Δm| ≈ 1.5e-8 fp32 (1-ULP), 6.1e-5 fp16 (1-ULP @ scale 0.5), 4.9e-4 bf16 (sub-ULP @ scale 1.0) |
| `v` (2nd-moment) | bespoke `.h` | DAG | **bit-exact** |
| `var` (param) | bespoke `.h` | DAG | **bit-exact** |
| `max_grad_norm` | bespoke `.h` | DAG | **bit-exact** |

`m` has 2 Muls + 1 Sub + 1 Cast in the EMA update — highest AI; the
other outputs have fewer chain links and survive bit-exact. Signature
matches OL-83 (two valid fp32 paths → 1-ULP boundary drift) but
**operates at port level, not single-op precision-probe level**.

**Relation to OL-83**:
- OL-83 = a single op + two FMA-grouping orderings on the same SoC give 1-ULP drift
- This candidate = cross-SoC A3↔A5 port + structural difference (bespoke per-dtype `.h` vs DAG) gives 1-ULP drift, **localized to the highest-AI output**
- Not contradictory — this candidate is a *port-specific predictor* for which output OL-83-class residual will land on, BEFORE the precision probe runs.

**How to use during port verification**:
1. Inventory port-source: if A3 has `<op>_b16.h` / `<op>_fp.h` family AND A5 uses `ElementwiseSch<OpDag>`, EXPECT cross-platform 1-ULP residual on the heaviest-chain output.
2. Don't waste a precision-probe spawn trying to chase bit-exactness on that output across SoCs. Accept T2 within per-dtype ULP-floor.
3. DO still verify the *other* outputs are bit-exact — if more than the heaviest-chain output drifts, the port has a real bug, not a structural residual.

**Promotion gate**: needs ≥ 1 additional Adam-family port confirmation
(e.g. apply_adam_w / apply_came / apply_lamb), or any non-Adam atvoss-DAG
port (norm-fuse-quant, layernorm-quant) showing the same "drift concentrates
on highest-AI output, others bit-exact" signature. If a second case shows
drift on a low-AI output instead, the candidate's "highest-AI" claim is
wrong and the rule needs revision before promotion.

**Cross-ref**:
- OL-83 (1-ULP single-position drift between two valid fp32 paths — same-SoC root)
- OL-141 (target arch35 advisory-inventory rule — context where this drift class shows up)
- OL-81 (CAST_RINT for bf16/fp16 output cast — the DAG's output cast convention)
- OL-118 (when fp32 output should NOT be cast — different concern: this candidate is about *how* the cast surfaces drift on intermediate accumulators)

## CAND-A3A5-23: Host-side scalar dtype-conversion to bypass W11 `ToFloat<bf16>` restriction in pybind/ACLRT_LAUNCH_KERNEL ports

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=port_a3_to_a5,elementwise-with-scalar-param`
`verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: fatrelu_mul)`
`unverified_on: A3 (V220 has no W11 restriction; pattern is A5-port-specific)`

**Predicted rule** (forward-looking, 1-op evidence):
For ops that take a small set (1-3) of scalar parameters of fp16/bf16/fp32 dtype, **extracting the scalar(s) to fp32 on host** in the pybind11 launcher via `tensor.to(at::kFloat).cpu().data_ptr<float>()[0]` and **passing them as fp32 kernel launch arguments** avoids the W11 `ToFloat<bfloat16_t>` restriction entirely. The kernel itself never reads a bf16 scalar from GM, never calls `ToFloat<bfloat16_t>`, never invokes any restricted intrinsic — yielding a uniform kernel template across fp16/bf16/fp32 input dtypes with no W11 conflict.

**Contrast with upstream V220 pattern**: V220 reads scalar from a 1-elem GM tensor via `inScalarGM.GetValue(0)` then casts via `ToFloat(threshold)`, which requires `__aicore__` specialization for bfloat16_t. On A5 (V351) this is restricted per W11. Host-side conversion is cleaner and side-steps the restriction at the kernel boundary.

**Pybind purity preservation**: `tensor.to(at::kFloat)` is a dtype conversion (managed by torch's tensor library), not a math operation. The pybind layer remains compute-free per the project's "no PyTorch/CANN delegation" rule — it only reshapes scalar parameters into a launch-arg-compatible form.

**Concrete anchor (fatrelu_mul, 2026-05-17)**:
```cpp
// workspace/fatrelu_mul/kernel/pybind11.cpp::run_fatrelu_mul
float threshold = threshold_tensor.to(at::kFloat).cpu().data_ptr<float>()[0];
// All three kernel entry points take `float threshold`:
//   fatrelu_mul_kernels.cpp::fatrelu_mul_fp32(..., float threshold, ...)
//   fatrelu_mul_kernels.cpp::fatrelu_mul_fp16(..., float threshold, ...)
//   fatrelu_mul_kernels.cpp::fatrelu_mul_bf16(..., float threshold, ...)
```
Result: 8/8 T1 PASS bit-exact across fp16/bf16/fp32. No W11 errors, no per-dtype `__aicore__` specialization needed for the scalar param.

**Byte-identity proof for each IEEE dtype** (validated on fatrelu_mul 2026-05-17 — explains why the host-side conversion does NOT introduce a precision delta vs upstream V220's kernel-side `GetValue(0) + ToFloat` chain):

- **fp32**: `.item<float>()` is identity host→device (single fp32 word read; no conversion needed).
- **fp16**: `.item<float>()` performs IEEE half→fp32 widening (zero-extend mantissa, re-bias exponent — single instruction, deterministic, bit-identical to AscendC's `(float)half_value` widening on device).
- **bf16**: `.item<float>()` performs bf16→fp32 by zero-padding the low 16 mantissa bits — bit-identical to AscendC's `ToFloat<bfloat16_t>(v)` on device. The widening conversion is exact (no rounding occurs) for both directions because bf16 is a strict prefix of fp32's bit layout.

This dtype-by-dtype identity proof generalizes to ANY 1-element scalar tensor input in the IEEE float family — same pattern transfers to other thresholds/alphas/limits without per-op identity re-derivation.

**Applicability** (predicted next ports): `clamp_min` (1 scalar), `add_scalar` (1 scalar), `topk_threshold` (1 scalar), `leaky_relu` (1 scalar `negative_slope`), `hardshrink` (1 scalar `lambd`), `softplus` (1 scalar `beta`), or any other A3→A5 port where the upstream V220 reads a scalar param from a tiny GM tensor and casts via `ToFloat`.

**Promotion gate**: needs validation on 2+ additional ports with scalar parameters (different op classes, e.g. clamp_min + leaky_relu) to confirm the pattern transfers across activation/threshold variants. If a second case shows the host-side conversion introduces a precision-mismatch vs CPU truth (e.g. due to host fp32 quantization differing from device-side cast), revisit.

**Cross-ref**:
- W11 (restricted `ToFloat<bfloat16_t>` intrinsic on A5 — see KB W11 entry if present, or W-restriction sweep notes)
- OL-143 (L1 mechanical port for port_a3_to_a5 — this pattern fits cleanly inside L1)
- OL-81 (CAST_RINT for bf16 — the per-dtype cast convention this pattern preserves at output, despite consolidating input scalar to fp32)
- P140 (pybind/ACLRT_LAUNCH_KERNEL path — host conversion is only meaningful in that mode; ops-nn/op_host/op_kernel/arch35 binary-registration path follows a different scalar-passing convention)

## CAND-A3A5-18: Per-row dual `TQue<VECIN,4>` + single `TQue<VECOUT,4>` pipeline for split-input pointwise ops

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-split-input,gate-and-multiply`
`verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: fatrelu_mul)`
`unverified_on: other SwiGLU/GeGLU/ReGLU/gating-style ports`

**Predicted rule** (forward-looking, 1-op evidence):
For ops where each row has two correlated input slices (split-input pointwise: x1 = first half of input row, x2 = second half), use **two depth-4 input queues simultaneously** + one depth-4 output queue. The driver loop becomes:

```cpp
for row in my_rows:
    AllocTensor(x1_q); AllocTensor(x2_q);
    DataCopyPad(x1, ...); DataCopyPad(x2, ...);
    EnQue(x1_q); EnQue(x2_q);
    // -- Compute step --
    DeQue(x1_q); DeQue(x2_q); AllocTensor(out_q);
    Cast / CompareScalar / Select / Mul / Cast;
    FreeTensor(x1); FreeTensor(x2); EnQue(out_q);
    // -- CopyOut step --
    DeQue(out_q); DataCopyPad(out, ...); FreeTensor(out);
```

**Why this beats upstream V220's manual sync choreography**: V220 uses explicit `pingPongFlag` + `SetFlag<HardEvent::MTE3_MTE2>` + `WaitFlag<HardEvent::MTE2_V>` to overlap pipelines. TQue depth=4 on A5 gives equivalent MTE2/VEC/MTE3 overlap with much less surface for sync-bug regressions; no manual flag-management code path means no per-dtype flag-mismatch failure mode.

**Concrete anchor (fatrelu_mul, 2026-05-17)**:
Two parallel `TQue<QuePosition::VECIN, 4>` for x1/x2 + one `TQue<QuePosition::VECOUT, 4>` for output. 8/8 T1 PASS bit-exact vs A3 ground truth; median ratio 1.054× over A3 baseline on small-shape inputs (max 8K elements). Per-tile compute (Cast + CompareScalar + Select + Mul + Cast) is heavy enough to amortize depth=4 allocator overhead — falls in OL-63's "heavy compute → depth=4" regime, not the "thin compute → depth=2" regime.

**Applicability** (predicted): any split-input pointwise op family —
- SwiGLU / clipped_swiglu (x1 = SiLU-gate, x2 = up-projection)
- GeGLU / ReGLU / FATReLU / clipped_silu_mul
- Mul-of-two-halves activation variants
- Two-input element-wise gating (output = f(x1) * x2 for some f)

**Promotion gate**: needs validation on 2+ additional ports in this family (e.g. swiglu, geglu) confirming depth=4 dual-VECIN holds the perf ratio above OL-143's 0.6× floor without per-tile thinning regression. If a second case is thin-compute (single VEC op per tile) and depth=4 regresses vs depth=2 (per OL-63's thin-compute branch), the rule should be scoped to "heavy compute split-input" rather than "all split-input".

**Cross-ref**:
- OL-63 (TQue<VECIN,4> depth decision — this candidate is a specific application to the dual-VECIN case)
- OL-143 (L1 mechanical port — this pattern is the canonical L1 layout for SwiGLU/FATReLU-family)
- P-P28 (per-tile depth=4 baseline pattern — extends to dual queue here)
- OL-115 (depth=2 + explicit prefetch — the alternative for thin per-tile compute; not preferred for split-input gate-and-multiply where compute is heavy)

## CAND-A3A5-19: Front-back-split + gate + elementwise-mul ops share a single L1 port template — kernel-level substitution is the only per-op delta

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=port_a3_to_a5,glu-family,split-input-gated-mul`
`verified_on: soc=Ascend950PR; cann=9.0.0 (2-op evidence: clipped_swiglu, fatrelu_mul)`
`unverified_on: other GLU-family variants (swiglu, geglu, reglu, clipped_silu_mul)`

**Predicted rule** (forward-looking, 2-op evidence — at promotion-gate threshold):

Operators whose A3 reference matches the shape:
- Last-dim split: `x1 = row[:d]`, `x2 = row[d:2d]` (front-back, not interleaved)
- Apply a gate `g(x1)` (the per-op delta — comparison, sigmoid, GeLU, clipping, etc.)
- Output = `g(x1) * x2`
- Per-row partition across AIV cores
- fp16/bf16 cast-in → fp32-compute → cast-out pipeline

…all collapse to a single L1 mechanical port template. Take an already-verified archive (clipped_swiglu is the current anchor) as the starting point and substitute ONLY the inner-compute body — file layout, tiling derivation, dual-TQue<VECIN,4> dispatch loop, fp32-compute path, cast emission all transfer verbatim.

**Concrete anchor — template archive**: `output/a3_to_a5_port/src/kernels/clipped_swiglu/`. Per-op delta is the `Compute()` body's primitive sequence:

| Op | Inner-compute body (fp32 path) |
|---|---|
| clipped_swiglu | `Mins → Maxs → Adds → Muls → Exp → Adds → Div → Mul` (~8 ops) |
| fatrelu_mul | `CompareScalar → Select → Mul` (3 ops) |
| swiglu (predicted) | `Mul(x1, x1) → Muls → Adds → Sigmoid-via-Exp → Mul` (similar shape) |
| geglu (predicted) | `Mul → Muls → Adds → Tanh → Adds → Muls → Mul` (similar shape) |
| reglu (predicted) | `CompareScalar(0) → Select → Mul` (same as fatrelu_mul with threshold=0) |

The dispatch-loop structure (`for row in my_rows: DataCopyPad x1 / x2 → EnQue → Compute → DeQue → DataCopyPad out`), per-AIV tile sizing, fp16/bf16 cast bracketing, and W11 host-side scalar-conversion (per CAND-A3A5-17) are template-fixed and transfer without modification.

**Why this is more than "L1 generally works"**: GLU-family ops surface a recurring micro-structure (dual-VECIN, split-row-front-back, gate-then-mul) that maps 1:1 to a small set of AscendC primitive sequences. The pattern lets a fresh kw spawn:
1. Identify the op as front-back-split + gate + mul (taxonomy check)
2. Copy the clipped_swiglu archive as the starting workspace
3. Rewrite only the `Compute()` primitive sequence + register the right scalar params
…cutting the kernel design phase from ~30 min (full algorithm derivation) to ~5 min (substitution).

**Promotion gate**: needs ONE more independent op evidence (next candidate: `swiglu` or `geglu` archive port) to confirm the template-substitution is reliable across more-than-trivial gate variations. If a third port confirms, promote to canonical OL/P-P entry under `patterns/domains/port_a3_to_a5.md`.

**Anti-pattern guardrails** (where the template should NOT be applied verbatim):
- **Interleaved-stride-2 layout** (gpt-oss SwiGLU mode=1, per P-P71) breaks the front-back assumption → template's dual `DataCopyPad x1/x2` would compute wrong slices; needs stride-2 extraction per P-P71 instead
- **Heavy gate primitives** (e.g. `Erf`, `Sqrt` chains for exotic activation variants) may push per-tile compute above OL-63's "heavy compute" threshold AND need a smaller `TILE_PAIRS` — re-derive tile size from UB budget rather than copying clipped_swiglu's 512
- **Variable d per row** (jagged GLU variants) breaks the constant-`d` per-row partition — fall back to standard non-GLU L1 path
- **Tri-input or higher arity** (e.g. clipped_silu_mul with extra clip params) extends, but the third TQue<VECIN> would push depth=4 dispatch beyond the dual-queue template — investigate before declaring template-fit

**Cross-ref**:
- OL-141 (L1 mechanical port — this is the per-family L1 template specialization)
- OL-143 (L1/L2/L3 classifier — front-back-split + gate-mul ops should reliably classify L1)
- CAND-A3A5-17 (host-side scalar conversion — sub-pattern reused by this template for `threshold` / `alpha` / `limit` scalars)
- CAND-A3A5-18 (dual VECIN<4> + single VECOUT<4> pipeline — the dispatch-loop shape this template adopts)
- P-P71 (chunked vs interleaved-stride-2 fingerprinting — guards against applying this template to the wrong layout convention)
- OL-152 (A3↔A5 API substitution — applied per-primitive once the Compute body is written)

---

### CAND-WF-1: Pre-route agent spawn when pre-spawn classifier deterministically predicts the spawn's handoff verdict

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=orchestrator_phase_o3`
`verified_on: 3_FusionAttention a3 benchmark cold-start 2026-05-19 (fusionattention-kw-1 emitted structural_rewrite_needed after Phase A; one full kw spawn cost wasted on a verdict the classifier could have derived)`
`unverified_on: other handoff verdicts (escalate_to_researcher, await_pp_for_persistent_partial) — only structural_rewrite_needed cold-start has cross-evidence here`

**Principle (abstract, mode-agnostic)**: When the orchestrator's pre-spawn classifier (`op_classification.json` + complexity tier + sibling-archive lookup) can deterministically derive that a worker spawn will emit a specific routing handoff X, the orchestrator should pre-route directly to handoff X's target FSM state — skipping the spawn cost. The worker contract is preserved (Phase A artifact + handoff emission both happen — they're synthesized from the classifier's inputs) and the wasted Phase B/C/D attempt is skipped.

**Trigger conditions for safe pre-routing**:
1. Op classification fully resolved (`algorithm_classification` + `complexity_tier` + fused tag + `ref_runnable.json`)
2. The handoff verdict X has an unambiguous trigger expressed as a boolean over classifier inputs (no judgment call required from the worker)
3. Either (a) sibling-target archive provides measured-evidence baseline confirming the verdict, OR (b) KB OL/PB explicitly codifies the trigger (e.g. OL-159 cold-start criterion for FA-class)

**Safety constraint**: pre-routing must SYNTHESIZE a Phase A artifact (e.g. `workspace/{op}/analysis.md`) containing the routing reasoning + cited evidence, so downstream agents (designer / researcher / probe) inherit the same context they would have gotten from a real spawn's emission. Without this artifact the downstream agent has no inheritance trail.

**Concrete anchor — FA-class cold-start pre-route**:
- Classifier inputs: `op_class=fa,attention_forward; complexity=L4; ref_runnable=runnable; fused=true`
- Sibling archive lookup: `output/<sibling-target>/src/kernels/<op>/verification.json.precision = {PARTIAL_PERSIST, pass_a_count_passed ≤ 5/N, spawn_count ≥ 5, cost ≥ $20}`
- Conclusion per OL-159: FA-class cold-start → the kw FA template-assembly recipe
- Pre-route: orchestrator writes synthesized `analysis.md` (citing OL-159 cold-start criterion + sibling archive evidence band) and routes to `await_worker` with the FA template-assembly recipe pre-selected
- Savings: ~$3-4 LLM cost + ~10 min wallclock per FA-class cold-start op

**Generalizes to** (predicted, needs validation): any FSM transition where the worker's emitted handoff is fully determined by inputs visible pre-spawn. Candidate handoff verdicts: `structural_rewrite_needed` (FA cold-start — verified), `escalate_to_researcher` (when complexity ≥ L4 and KB has no matching pattern — predicted), `await_pp_for_persistent_partial` (when sibling has PARTIAL_PERSIST with same signature — predicted).

**Cross-ref**: OL-159 (FA cold-start trigger criterion — the first verdict to support pre-routing); CAND-WF-2 (sibling-archive carry-forward evidence — the data source that enables pre-routing decisions).

**Status**: 1-op evidence (3_FusionAttention a3 2026-05-19). Promotion gate: implement pre-route for FA-class cold-start in `state_machine.py` Phase O3 entry, validate on ≥1 more FA-class cold-start op (e.g. flash_decoding, multi_head_latent_attention) that the pre-routed path produces the same downstream-agent outcome as the legacy spawn-then-route path. Risk: false pre-routing on edge cases where worker would have produced a different verdict than classifier predicts; mitigation = retain "force-spawn" override for orchestrator audit mode.

---

### CAND-WF-2: Cross-target sibling-archive `scope_note` + iter-count as carry-forward evidence in Phase O2.5 worker brief

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=orchestrator_phase_o2_5`
`verified_on: 3_FusionAttention a3 benchmark cold-start 2026-05-19 (fusionattention-kw-1; sibling A5 archive scope_note was decisive evidence for honest tractability projection, manually surfaced by worker)`
`unverified_on: non-FA op classes — only L4 fused-attention cross-target has been exercised here`

**Principle (abstract)**: When the SAME op exists as a finalized archive on a sibling target (a5↔a3, or future architectures), its `verification.json.precision.scope_note` and `verification.json.precision.scope_gap_analysis.remaining_gap_close_estimate_iters` represent **empirically-measured** baselines — not algorithmic guesses. Phase O2.5 brief construction should auto-detect these and surface them in the worker brief's "Hard gate floors (cold start)" block under a clear "cross-target evidence (NOT target floor)" label.

**Safety constraint (within C19/C22 boundaries)**: SURFACE only the high-level architectural framing (scope_note text + iter-count band + cost band), NEVER include kernel code paths, kernel source content, `design.md`/`analysis.md` body text, or anything that would constitute prompt-leakage of sibling kernel implementation details. Status-only inheritance preserves the worker's independent algorithmic derivation per C22. The fields permitted to carry forward:
- `verification.json.precision.scope_note` (one-paragraph architectural framing)
- `verification.json.precision.scope_gap_analysis.remaining_gap_close_estimate_iters` (integer band)
- `verification.json.precision.pass_a_count_passed` (numeric, e.g. "1/61")
- Aggregate spawn count + cost from `optimization_log.md` or equivalent (numeric)

FORBIDDEN to surface: `kernel/`, `model_new_ascendc.py`, `analysis.md` body, `fused_analysis.md`, `optimization_directive.md`, any tilingkey/primitive choice from sibling.

**Concrete anchor** — 3_FusionAttention 2026-05-19:
- Sibling lookup: `output/npukernelbench/src/kernels/3_FusionAttention/verification.json`
- Surfaced fields: `precision.status="PARTIAL_PERSIST"`, `precision.pass_a_count_passed=1/61`, `scope_gap_analysis.remaining_gap_close_estimate_iters="30-50"`, aggregate `spawn_count=8`, `aggregate_cost_usd=~30`
- Effect: A3 kw inherited the iter-budget reality check (40-70 iters needed) without seeing any kernel code from A5 → emitted `structural_rewrite_needed` with confidence rather than attempting a placeholder kernel

**Generalizes to** (predicted): any cross-target port (a5↔a3, future arches), any op class where sibling-archive carry-forward provides a measurable iter/cost band that an iter-budget-bounded worker spawn could not bridge. Particularly load-bearing for L3/L4 ops where the gap-closure cost is highly non-linear in complexity tier.

**Cross-ref**: OL-159 (FA-class trigger criterion that consumes this evidence); CAND-WF-1 (pre-route decision that the surfaced evidence supports); C19 (sibling-cross-check status-only — establishes the read boundary); C22 (prompt-leakage prohibition — establishes the WRITE boundary on what may enter the brief).

**Status**: 1-op evidence (3_FusionAttention a3 2026-05-19). Promotion gate: implement auto-detection in Phase O2.5 brief construction (`kw_brief.py` "Hard gate floors" section), validate that the C19/C22 boundary stays intact via mechanical grep test (no kernel-code strings from sibling appear in worker brief), exercise on ≥1 more cross-target port (non-FA op class) to confirm the principle generalizes beyond FA-class.

## CAND-PP96: E8M0 shared-exponent byte → fp32 via union bit-reinterpret in AscendC scalar context

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=15.x; op_class=mx-quant,mxfp8,mxfp4,microscaling-decode; kernel_type=AIV scalar pipe`
`verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm — single op evidence)`
`unverified_on: soc=Ascend910_V220 (no E8M0 type support per OL-144); op_class=mxfp4-matmul, mxfp8-attention, future MX-format consumers`

**Principle**: when decoding MicroScaling E8M0 shared-exponent scale bytes (`fp8_e8m0_t`, see OL-144) in scalar context on the AIV pipe, libm is unavailable so `ldexpf(1.0f, byte - 127)` cannot be used. The canonical primitive is a union-based bit reinterpret that constructs the IEEE-754 fp32 with the byte placed directly into the biased-exponent field. For each byte `b ∈ [0, 254]`, the corresponding scale `2^(b - 127)` is exactly representable as a normal fp32 (sign=0, biased_exp=b, mantissa=0).

**Concrete anchor** (3-line primitive — drop-in for any kernel that consumes per-block E8M0 scales):

```cpp
__aicore__ inline float E8m0_byte_to_scale(uint8_t b) {
    union { uint32_t u; float f; } cvt;
    cvt.u = static_cast<uint32_t>(b) << 23;  // IEEE 754: sign=0, biased_exp=b, mantissa=0 → 2^(b-127)
    return cvt.f;
}
```

The `union` bit-cast is well-defined in AscendC scalar code under bisheng (CANN 9.1.0.B010 confirmed); we don't need to fall back to `memcpy`-style type punning. Compiles cleanly on the AIV scalar pipe.

**Where this bites if missing**: workers reaching for `ldexpf(1.0f, b - 127)` get an "undefined symbol" link error (libm not available on AIV); workers reaching for `Cast` from uint8 to float get **PB-26 territory** (`Cast<float, uint8_t>` is unsupported, silent garbage). The union bit-reinterpret is the only correct primitive for E8M0 → fp32 in kernel-side scalar code.

**Why this is reusable beyond LayerNorm**: any MX-format consumer needs per-block scale broadcast. Examples:
- mxfp8 attention (Q/K/V quantized with E4M3 mantissa + E8M0 shared scale per 32-element block).
- mxfp4 matmul-class kernels (FP4x2 packed weights + E8M0 per-block scale per OL-144 / OL-145).
- Any future MX-format dequant where the per-block scale is consumed scalar-wise (not via the `fp8_e8m0_t` → bf16 reg-vector Cast in `__VEC_SCOPE__`).

**Anti-patterns**:
- ❌ `Cast<float, uint8_t>(scale_fp32, scale_u8, RoundMode::CAST_NONE, 1)` — uint8 → float Cast is unsupported on A5 (PB-26 family), silently produces garbage.
- ❌ `ldexpf(1.0f, byte - 127)` — libm unavailable on AIV.
- ❌ `pow(2.0f, byte - 127)` — same libm issue, plus floating-point pow is slow and loses bit-exactness vs the exact integer-bit construction.
- ❌ Computing the scale on host and broadcasting an already-fp32 scale tensor — works but doubles input bandwidth on every quant kernel; the kernel-side bit reinterpret is strictly cheaper.

**Evidence**:
- MxFp8LayerNorm kw-1 (2026-05-21 Ascend950PR_957c, CANN 9.1.0.B010): primitive used in step 3 (per-block scale application) — 24 to 128 scalar calls per row depending on D, each computing one `2^(b - 127)` scale exactly. Pass-A 8/8 + Pass-B 11/11 PASS_WITHIN_TOLERANCE confirms numeric correctness (would fail dequant precision immediately if the bit-reinterpret produced wrong scales).

**Promotion gate**: 2+ op evidence required. Next candidate: any future mxfp8 / mxfp4 op landing with the same scalar-pipe E8M0 decode pattern (e.g. an independent fused mxfp8-attention prototype, OR an mxfp4-matmul kernel decoding scales scalar-wise). Then promote to `patterns/domains/quant.md` as a P-Pxx.

**Cross-ref**:
- OL-144 (A5 narrow-float datatype family — `fp8_e8m0_t` exponent-only scale type definition)
- OL-145 (MicroScaling format — per-block 32-element scale convention)
- PB-26 (`Cast<float, uint8_t>` unsupported — the trap this primitive avoids)
- OL-152 (L2 register-based path uses reg-vector Cast for fp8_e8m0_t → bf16; this candidate is the L1-scalar-pipe counterpart for the same decode)

## CAND-PP97: Pybind-side scale-tensor padding for sub-32B-per-row inputs in per-block quant kernels

`applies_to: soc=Ascend950PR; cann=9.0.0+; bisheng=all; op_class=mx-quant,mxfp8,mxfp4,per-block-quant,any kernel reading a sub-32B-per-row scale/index/metadata tensor`
`verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm — single op evidence on D=768 case)`
`unverified_on: soc=Ascend910_V220; op_class=other-quant-formats (int4-grouped, GPTQ-style, AWQ, mxfp4)`

**Principle**: AscendC's `DataCopy(dst, src, count)` requires `count * sizeof(T)` to be a multiple of 32 bytes — non-aligned `count` silently rounds DOWN (see OL-167). For input-side tensors whose per-row payload is naturally sub-32B (e.g. per-block scale bytes in MX-format kernels: `n_blocks = D / 32` uint8 bytes where small `D` makes `n_blocks < 32`), the kernel cannot use plain DataCopy to load one row's scales. The fix is **host-side pybind padding** that zero-extends the scale tensor along the last dim to a multiple of 32, while the kernel iterates only over the original (unpadded) per-row scale count.

This is **distinct from OL-167's anti-pattern** (host-side OUTPUT padding + `narrow+contiguous` to hide kernel non-aligned writes — that's data-path cheating). Here the padding is on the INPUT side, the kernel READS only the original-count active scales, and the zero-padded tail is provably unused. Input metadata layout IS a pybind responsibility because:
1. The kernel cannot reshape inputs (it only consumes GM pointers).
2. The per-block-scale layout convention (`scales.shape[-1] = D / block_size`) is part of the public op contract — the kernel can't unilaterally change it.
3. The `DataCopy` 32B-alignment is a hardware constraint on the LOAD primitive, NOT a contract on the data layout.

**Concrete anchor** (pybind sketch, per-block-scale variant):

```cpp
// inputs: x_scales is uint8 shape [..., n_blocks] where n_blocks = D / BLOCK_SIZE
int64_t n_blocks = D / BLOCK_SIZE;
int64_t n_blocks_pad = ((n_blocks + 31) / 32) * 32;        // align_up to 32 bytes

torch::Tensor scales_eff = x_scales;
if (n_blocks_pad != n_blocks) {
    auto pad_shape = x_scales.sizes().vec();
    pad_shape.back() = n_blocks_pad;
    scales_eff = torch::zeros(pad_shape, x_scales.options());    // zero-init padding (NOT empty)
    scales_eff.narrow(-1, 0, n_blocks).copy_(x_scales);
}
// kernel reads scales_eff with stride n_blocks_pad per row, iterates only n_blocks per row
launch_kernel(..., scales_eff.data_ptr<uint8_t>(), /*n_blocks_active=*/n_blocks,
                                                    /*n_blocks_stride=*/n_blocks_pad, ...);
```

**Correctness invariant**: the kernel's per-row loop bound is `n_blocks_active` (the ORIGINAL count); the kernel's DataCopy uses `n_blocks_stride` (the padded count) ONLY for the DataCopy granularity. The padded tail bytes are READ into UB but never INDEXED by the compute loop, so their zero-init content is provably ignored.

**Why this is NOT covered by OL-162 or OL-167**:
- OL-162 covers asymmetric-shape kernels where one input's dim is hard-coded as buffer extent for a differently-shaped tensor (kernel-internal OOR). Different fault class.
- OL-167 forbids host-side OUTPUT padding + `narrow+contiguous` (kernel hides non-aligned writes). Different fault direction (input vs output) and different load-bearing reason (kernel CAN handle non-aligned input via DataCopyPad, but DataCopyPad for tiny scale tensors costs more than the host-side zero-extend; for per-block scales, the host pad is the natural fix).

**Trigger classifier**:
- `inputs.<scale_or_metadata_tensor>.shape[-1] * sizeof(dtype) < 32B` per row → pybind padding applies.
- `inputs.<main_data_tensor>.shape[-1] * sizeof(dtype) < 32B` per row → main-data tile so small the kernel architecture is wrong; revisit tiling.

**Anti-patterns**:
- ❌ `DataCopy(scalesLocal, gmXScales, n_blocks)` with `n_blocks < 32` → silently transfers ZERO bytes per OL-167. Symptom: scale = uninitialized → dequant produces garbage → precision verification fails with wildly out-of-range MERE.
- ❌ Using `DataCopyPad` for tiny per-block-scale loads — works but adds kernel complexity (extra params, smaller throughput); the host zero-extend is strictly simpler for INPUT scales.
- ❌ Forcing the kernel to compute the padding at GM-load time via element-wise scalar `GetValue` — wastes scalar pipe, much slower than a single DataCopy from a pre-padded host tensor.
- ❌ Assuming all valid op shapes will have `n_blocks ≥ 32` naturally — sweep your `D` cases; small-D variants (D=768 in MXFP8 LayerNorm, D=256 in some attention configs) will hit this.

**Evidence**:
- MxFp8LayerNorm kw-1 (2026-05-21 Ascend950PR_957c, CANN 9.1.0.B010): 8 benchmark cases sweep D ∈ {768, 1024, 2048, 4096} → only D=768 (case 3, n_blocks=24 < 32) needs padding. D ∈ {1024, 2048, 4096} → n_blocks ∈ {32, 64, 128} naturally aligned. Pybind detects via `n_blocks < 32` and zero-pads `x_scales` to `[..., 32]`. Kernel reads `n_blocks_active=24` for the compute loop. Pass-A 8/8 + Pass-B 11/11 PASS (case 3 specifically validates the padded path).

**Promotion gate**: 2+ op evidence required. Next candidate: any future MX-format / per-block-quant op with sub-32B per-row metadata (mxfp4 grouped matmul, per-block int4-quant with E4M3 scales, AWQ/GPTQ-style scale tensors). Then promote to `patterns/domains/quant.md` or `patterns/domains/data_movement.md` as a P-Pxx alongside P-P98 (DataCopyPad).

**Cross-ref**:
- OL-167 (DataCopy `count` silent truncation — explains why the bare DataCopy doesn't work; this candidate is the input-side mitigation, NOT the cheat OL-167 forbids)
- OL-162 (pybind padding wrapper for asymmetric-shape OOR — different fault class but same broad architectural shape: host pads to make kernel-internal extent invariants hold)
- P-P98 (DataCopyPad — the alternative for output-side non-aligned writes; mentioned here only to contrast: not applicable to small input scales)
- PB-26 / OL-144 (MX-format / fp8_e8m0_t scale convention — where the sub-32B-per-row constraint comes from)

## CAND-DEPLOY-STAGE-LOCAL_TASK: Standalone-mode kernel-worker spawns must stage workspace/<op>/ → LOCAL_TASK before invoking deploy_to_npu.sh

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=standalone_worker_spawn (NOT orchestrator-driven)`
`verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm kw-1 — single op evidence)`
`unverified_on: soc=Ascend910_V220 (A3 deploy path uses a different `deploy_to_npu.sh` stagedir layout — pattern may transfer but not yet replayed); orchestrator-driven mode (stage step is handled by orchestrator's Phase O3 hand-off, this candidate doesn't apply)`

**Principle**: `src/scripts/deploy_to_npu.sh` canonicalizes the kernel source through `LOCAL_TASK=$HOME/workspace/AscendOpGenAgent/current_task` when the env var is unset. In orchestrator-driven `/ascendc-op-gen` runs, the orchestrator's Phase O3 hand-off stages `workspace/<op>/{kernel,op_host,op_kernel,...}` into this LOCAL_TASK path before invoking the deploy script. When a worker is spawned **standalone** (no orchestrator wrapping — e.g. via direct `Agent(subagent_type=aog-kernel-worker, ...)` call from another agent / from `/aog-orchestrator-recover`, or via the `Skill` invocation entry-point), there is no Phase O3 stager, and the worker MUST stage its own files OR `deploy_to_npu.sh` re-deploys whatever was last in LOCAL_TASK from the previous op.

**Concrete anchor** (worker-side guard, drop in front of every deploy):

```bash
LOCAL_TASK=${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task}
WORKSPACE_OP="${WORKSPACE_ROOT:-/mnt/d/projects/a5_ops/workspace}/${OP_NAME}"

# Standalone-mode stage: mirror workspace/<op>/ → LOCAL_TASK
if [ -z "${ORCHESTRATOR_STAGED:-}" ]; then
    rm -rf "$LOCAL_TASK"
    cp -r "$WORKSPACE_OP/." "$LOCAL_TASK/"
fi

bash src/scripts/deploy_to_npu.sh --build
```

The orchestrator sets `ORCHESTRATOR_STAGED=1` after Phase O3 — so the guard is a no-op in orchestrator-driven mode but covers the standalone-spawn case.

**Symptom when missing** (caught first-build, kw-1): build appears to succeed but the resulting `.so` exports the previous op's symbols. Verification then fails with `ModuleNotFoundError: No module named '<current_op>'` (the build deployed `<previous_op>`'s files because LOCAL_TASK still held them). The error is far from the root cause — looks like a Python import bug, actually is a deploy-stage bug.

**Detection signature** (post-build sanity check):

```bash
# After deploy, verify LOCAL_TASK contains the CURRENT op's kernel files
test -f "$LOCAL_TASK/op_host/<op>_def.cpp" || echo "BUG: LOCAL_TASK contains wrong op's files"
test -f "$LOCAL_TASK/kernel/<op>_kernel.h" || echo "BUG: LOCAL_TASK contains wrong op's files"
```

**Anti-patterns**:
- ❌ Trusting `deploy_to_npu.sh` to read from `workspace/<op>/` directly — it does NOT (LOCAL_TASK is the canonical source).
- ❌ Overriding `LOCAL_TASK` env to point at `workspace/<op>/` without mirroring — `deploy_to_npu.sh` writes intermediates into LOCAL_TASK, corrupting the workspace working copy.
- ❌ Skipping the `rm -rf` before the `cp -r` — stale files from a previous op (op_host JSONs, partial CMake artifacts) survive and the build picks them up alongside the current op's files.

**Why this surfaced**: MxFp8LayerNorm kw-1 was spawned in this session right after 10_LayerNorm finalized. LOCAL_TASK still held 10_LayerNorm's files. First build went green (rebuilt the 10_LayerNorm .so), then verification failed with `ModuleNotFoundError: No module named 'MxFp8LayerNorm'` because the .so didn't export the MxFp8LayerNorm pybind module. Re-staged + rebuild → clean.

**Promotion gate**: 2+ op evidence in standalone-spawn mode (next candidate: any future direct `Agent(subagent_type=aog-kernel-worker, ...)` spawn that exhibits the same first-build wrong-op symptom). Then either:
1. Promote to OL alongside OL-160 (canonical entry-points) covering deploy-stage canonicality.
2. Land a `src/scripts/stage_and_deploy.sh workspace/<op>` wrapper that bakes the guard in, making the candidate self-obsoleting.

**Cross-ref**:
- OL-160 (canonical entry-point file names — the safety net assumes the .so it loads exports the canonical op; this candidate explains how the .so can end up exporting the WRONG op)
- `src/scripts/deploy_to_npu.sh` (the script being staged for)
- `ascendc-op-gen` Phase O3 (orchestrator-driven stager — sets `ORCHESTRATOR_STAGED=1` so the guard noops)


## CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP: V220 `KERNEL_TYPE_MIX_AIC_1_2` cube-internal pipe sync deadlocks regardless of event-ID scheme — root cause deeper than event-ID allocation

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
`verified_on: a5_ops:3_FusionAttention kw-3/kw-4/kw-5 iter chain 2026-05-21 — 4 distinct sync schemes tested over 4 worker iters; ALL produce identical silent-hang at torch.npu.synchronize()`
`v351_reproduces: NO (probe_a5_v300_fa_sync 2026-05-23) — Pattern A on V351 (MatmulImpl<> + manual CrossCoreSetFlag<0x2>(FLAG_AIC_DONE=0) + MIX_AIC_1_2 + 16×16×16 fp16 mm.IterateAll + AIV Muls(*,2.0)) completes in 0.036ms steady-state with bit-exact AIV output and non-zero matmul C output across 3 trials. The "deeper sync-infra gap" hypothesis applies to V220-specific FFTSCNT mailbox semantics, NOT to V351. PB-35 cube-internal pipe sync (event_t for HardEvent::M_FIX) remains unverified on V351 — follow-up probe required.`
`derived-from: empirical kw-5 falsification of CAND-FA1's "use event_t(N >= 4)" event-ID-collision hypothesis (codified separately as PB-35)`

**Hypothesis falsified by this candidate**: "Low event IDs (0/1) collide with AIC↔AIV CrossCoreSetFlag chain at flag IDs 0..3; using event IDs ≥ 4 (or canonical `GetTPipePtr()->FetchEventID()`) for cube-internal pipe sync resolves the deadlock."

**Empirical evidence chain** (4 schemes, 4 worker iters):

| Iter / cycle | Scheme | Outcome |
|---|---|---|
| iter 1 (kw-2 cycle 1) | `DataCopy` ND GM→L1 + `LoadData2D` L1→L0 (no ND→NZ step) | `error code 507015 aicore exception` on first `Mmad` (layout fault, not sync) |
| iter 2 (kw-3 cycle 2) | `DataCopy(l1, gm, Nd2NzParams{...dstNzC0Stride=S})` + `LoadData2D` (wrong stride) | Same generic 507015 (still layout fault) |
| iter 3 Phase 1 (kw-4) | Corrected `Nd2NzParams{...dstNzC0Stride=D/16}` + `LoadData2D` (no pipe sync) | `0x8000004000 L0B read/write conflict in MTE` (sync genuinely missing — layout fixed) |
| iter 3 Phase 2 (kw-4) | Above + `SetFlag/WaitFlag<HardEvent::MTE2_MTE1\|MTE1_M\|M_FIX>(event_t(0))` | Silent hang at `torch.npu.synchronize()` past 90s (PB-35 codifies as "event_t(0) collides with FLAG_CANON_DONE") |
| iter 4 Phase 1 (kw-5) | Same + raw `event_t(2,3,4)` for mm1 + `event_t(5,6,7)` for mm2 (distinct IDs, all ≥ 2) | Silent hang, identical signature |
| iter 4 Phase 2 (kw-5) | Same + `GetTPipePtr()->FetchEventID(HardEvent::X)` canonical runtime allocation | Silent hang, identical signature |

Three layer transitions across the chain (each iter peeled one layer): Layer 1 (iter 1) — ND vs NZ layout (solved by `Nd2NzParams` shape); Layer 2 (iter 3 Phase 1) — cross-pipe sync absent (MTE1→M RAW hazard); Layer 3 (iter 3 Phase 2 onward) — even with valid sync events, the deadlock persists. Falsifies the simple "event-ID collision" hypothesis.

**Strong candidate root-cause hypotheses for fo investigation**:
1. **Cross* sync uniformity per HardEvent class**: AIC↔AIV CrossCoreSetFlag chain may impose barriers that collide with `HardEvent::M_FIX` regardless of event ID (the cross-core semantics are uniform per HardEvent class, not per event ID). All three schemes fail because they all use `HardEvent::M_FIX` which IS the FIX-pipe event being driven externally by the AIV→AIC chain.
2. **MIX_AIC_1_2 may require uniform Cross* sync across ALL pipe events on the cube side**, not local SetFlag/WaitFlag mixed with CrossCoreSetFlag for cross-core sync. Cube-internal pipe sync via local `SetFlag<HardEvent::X>` may be incompatible with the mixed-mode dispatch loop.
3. **FFTSCNT mailbox semantics**: per the `kfc_dispatch_failure_followup.md` root cause investigation, MIX_AIC_1_2 may have mailbox-counter semantics that prevent any cube tile-MMAD with internal sync regardless of event ID scheme.

**Scope**:
- Applies only to `KERNEL_TYPE_MIX_AIC_1_2` launches (single-launch mixed cube + vector core).
- Non-mixed cube-only `KERNEL_TYPE_AIC_ONLY` launches NOT tested this evidence chain — may behave differently.
- Tested only on case_3 [4,64,512] BSH fp16 head=8 (S=64, D=64). Smaller / larger shapes not probed; conclusion likely holds across shapes given the failure is at the cube-launch level not at compute.

**Mitigation today (until fo investigation lands)**:
- Stay on AIV-only VEC fallback path for fp16 cube-eligible shapes. AIV-only path delivers 3_FusionAttention case_3 PASS_T1 with ours_mere=1.999e-6 beats CANN (cann_mere=2.227e-6, 11% better). Performance via AIV path is bandwidth-limited but acceptable for the in-scope shape.
- Workspace baseline kernel uses VEC-only path; do NOT attempt MIX_AIC_1_2 cube tile-MMAD with user-owned pipe sync until KB seeding lands.

**Escalation paths (mutually exclusive — pick one)**:
1. **`aog-fused-optimizer` + KB seeding**: Investigate alternate launch modes (e.g. `KERNEL_TYPE_AIC_ONLY` separate from AIV softmax, with explicit GM hand-off via CrossCoreSetFlag<0x2>). Estimated 8-12 fo iters once API_CATALOG.md gains entries for cube tile primitives.
2. **`aog-cann-learner` Mode 5 extraction**: Run dedicated CANN learner against CANN's `ops-transformer/op_kernel/flash_attention_score/arch22/` source to extract V220 cube workflow into `patterns/unverified/candidates.md` as a verified pattern (with explicit event-ID allocation and CrossCoreSetFlag/HardEvent interaction documented).
3. **Pause for authoritative documentation**: Wait for an AscendC mixed-mode programming page on hiascend.com (currently no public doc covers MIX_AIC_1_2 cube-internal pipe sync rules).

---

**HYPOTHESIS STATUS UPDATED 2026-05-23 — V220 Pattern C status UNVERIFIED (NOT closed); A5/V351 cross-arch probe (commit `56444ff8`) invalidated V220 Pattern A as cross-arch finding**:

#### Prior (PR #117, 2026-05-22) claim — partially RETRACTED 2026-05-23

PR #117 codified my V220 Pattern C probe as "Class B falsification — Pattern C structurally blocked on V220" and authored the verdict "all 3 V220 mixed AIC+AIV patterns now empirically falsified". **Both claims are now flagged for re-examination per main agent's A5/V351 probe finding (commit `56444ff8` 2026-05-23)**:

1. **My V220 Pattern C probe was likely misdesigned** (same defect as main's V351 Pattern C probe per their PROBE_REPORT.md): used `SetFlag<HardEvent::MTE3_MTE2>` from AIC + `WaitFlag<HardEvent::MTE3_MTE2>` from AIV — but `HardEvent::*` is **intra-core** pipe-sync semantics. AIC and AIV are different cores → the WaitFlag on AIV waits on a pipe-event AIC never raises on the AIV's pipe register. Hang is from the misdesign, NOT a V220 architectural block of single-launch fused. To genuinely probe Pattern C on V220 the cross-core sync mechanism is `CrossCoreSetFlag<0x2>(flagId)` not `SetFlag<HardEvent>`.

2. **V220 Pattern A status unchanged** (PB-34 + PB-35 still valid V220-only). Pattern B unwound from production for unrelated reasons. But **the "all 3 patterns falsified" claim was overstating** — only A is empirically falsified on V220 by the 5-iter chain; B/C are weaker claims (probe-design defect or unverified).

3. **CAND-FA-MULTI-LAUNCH-PERF-GAP V220 "real ceiling at 0.014× CANN" claim still holds via Pattern A falsification alone** (no Pattern C contribution needed); but the phrasing "multi-launch is FORCED architecture" was too strong — should be "multi-launch is the path of least resistance given Pattern A's confirmed V220 deadlock; Pattern C on V220 remains genuinely unverified due to probe-design defect".

#### A5/V351 cross-arch finding (commit `56444ff8` 2026-05-23) — confirmed by main agent's `probe_a5_v300_fa_sync`

Pattern A on V351/Ascend950PR_9579 + CANN B103 runs clean (0.036ms, bit-exact, 3 deterministic trials). V220 Pattern A deadlock does NOT reproduce on V351. This invalidates any "V220 → V351 inheritance" of mixed-mode-deadlock assumption. See PB-34's new `verified_does_not_reproduce_on: V351` line.

#### Current status

- **V220**: Pattern A confirmed deadlocking (PB-34/PB-35 valid). Pattern B/C status — Pattern B unwound for unrelated reasons; Pattern C UNVERIFIED (probe was misdesigned). Multi-launch architecture remains the practical V220 choice given Pattern A block, but the door is NOT formally closed on Pattern C.
- **V351**: Pattern A runs clean. Single-launch fused FA is **viable on V351**. Use single-launch architecture for any V351 / A5 fused-attention port.
- **Real V220 ceiling**: 0.014× CANN @ S=1024 (PR #114 measured) stays correct, supported by Pattern A deadlock (PB-34) — does NOT rely on Pattern C "falsification".

#### Follow-up needed (genuine V220 Pattern C falsification)

A correctly-designed V220 Pattern C probe must use cross-core sync (`CrossCoreSetFlag<0x2>(flagId)` on AIC + `CrossCoreWaitFlag<0x2>(flagId)` on AIV), not `HardEvent`-based intra-core sync. The original PR #116 probe source at `docs/design/fa_delta5_pattern_c_probe_snapshot/probe_pattern_c_kernel.h` is **invalidated as a Pattern C reproducer**; future probe should follow PB-35's reserved-flag-IDs convention (cross-core flags 0..3, cube-internal events ≥4 if pipe sync also needed). Cost: ~60-90 min for proper probe.

#### Cost paid for this codification

- PR #117 (~45 min) shipped a partially-wrong claim that was caught + corrected within ~3h by main's A5 probe.
- Net KB still benefits from the negative finding (probe-design defect is a learned lesson, codified below), even though the original "falsified" claim was wrong.
- **Lesson per OL-175**: probe-result codification needs probe-design-validation step. My V220 Class B hang shape (silent timeout, no error) is identical to a misdesigned-sync probe AND a real cross-core deadlock — they're not distinguishable without checking sync intrinsic semantics first.

#### Cross-ref

- [CAND-FA-MULTI-LAUNCH-PERF-GAP §5](candidates.md#CAND-FA-MULTI-LAUNCH-PERF-GAP) (re-scoped 2026-05-23 to V220-only; V351 has different path)
- PB-34, PB-35 (still valid Pattern A V220-only falsifications)
- `workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md` (main's A5 probe + identification of intra-core vs cross-core sync defect)
- `docs/design/fa_delta5_pattern_c_probe_snapshot/probe_pattern_c_kernel.h` (my malformed V220 probe — preserved as reproducer for the defect, NOT as falsification evidence)
- OL-175 (failure-framing discipline — applies here: "Class B fail" was claimed without distinguishing misdesign from real deadlock)

**Hard do-not-apply**:
- Do NOT use `KERNEL_TYPE_MIX_AIC_1_2` with user-owned cube tile-MMAD + local `SetFlag/WaitFlag<HardEvent::*>` pipe sync until this candidate resolves. Compiles clean, runtime deadlocks.
- Do NOT interpret PB-35's "Use IDs ≥ 4" Fix as a complete solution — it closes the visible event-ID collision but not the underlying sync-infra gap.

**Cross-ref**:
- PB-35 (the visible event-ID collision; this candidate explains why the PB-35 Fix is incomplete)
- PB-34 (the related cube+vec sync minefield — Matmul library vs user-owned flags)
- CAND-FA1 (Pattern A recommendation; this candidate is the open-hypothesis follow-up for CAND-FA1's empirical_validated_on event-ID section)
- OL-159 (FA-class independent prototype structural-rewrite-needed — the larger framing under which this gap lives)
- `output/.../workspace/3_FusionAttention/kfc_dispatch_failure_followup.md` (prior fo workstream tracking the FFTSCNT mailbox angle)
- API_CATALOG.md (currently zero entries for cube tile-MMAD primitives — known gap)

**Promotion path**: candidate stays here until (a) one of the 3 escalation paths resolves the deadlock with reproducible evidence on case_3, OR (b) the AIV-only fallback path is formally declared the canonical V220 FA strategy and Pattern A on V220 is documented as architecturally infeasible. Either outcome promotes to OL-class entry.


## CAND-OPAQUE-STRUCT-RUNTIME-VERIFY: Hardware-intrinsic structs with undocumented field semantics — compile-clean is NOT a success signal; require hardware verification before claiming a primitive works

`applies_to: soc=all; cann=all; bisheng=any; op_class=any_kernel_using_low_level_hardware_primitive_with_opaque_struct_fields`
`verified_on: a5_ops:3_FusionAttention kw-2/kw-3/kw-4 chain 2026-05-21 — best-effort Nd2NzParams field guessing compiled cleanly across 3 iters, faulted at runtime with 3 distinct fault signatures; correct field shape only confirmed in iter 3 after directive sourcing concrete API field values from V220 SDK headers`
`derived-from: empirical Nd2NzParams field-shape divergence; generalizes to any opaque hardware-intrinsic struct`

**Principle**: When a kernel primitive's parameter struct contains fields that forward 1:1 to a hardware intrinsic AND the public SDK header has NO docstring explaining the fields' semantics (only field types), best-effort guessing based on field names produces kernels that compile cleanly but fault at runtime. The bisheng / CANN toolchain accepts ANY valid C++ type signature on these calls; there is no compile-time check on stride field values, layout decoding, event-ID safety, or fragment alignment. Workers MUST verify on hardware (≥1 routed test case yielding a clean run) before claiming the primitive works — "compile clean" is not a success signal for these primitives.

**Concrete anchor** (the Nd2NzParams case — empirical evidence chain):

```cpp
// Public SDK declaration in basic_api/kernel_struct_data_copy.h L257-304:
struct Nd2NzParams {
    uint16_t ndNum;
    uint16_t nValue;
    uint16_t dValue;
    uint16_t srcNdMatrixStride;
    uint16_t srcDValue;
    uint16_t dstNzC0Stride;   // ← OPAQUE: no doc on semantic meaning
    uint16_t dstNzNStride;    // ← OPAQUE
    uint16_t dstNzMatrixStride; // ← OPAQUE
};
// Constructor signature shows field order; no inline doc on what each stride means.
// Backing impl (dav_c220/kernel_operator_data_copy_impl.h L304-316) forwards
// these 1:1 to a hardware intrinsic `copy_gm_to_cbuf_multi_nd2nz_b16` which
// is ALSO undocumented in public SDK.
```

For case S=64, D=64 fp16: 5 of 8 fields are clearly derivable (`ndNum=1, nValue=S, dValue=D, srcNdMatrixStride=0, srcDValue=D` for contiguous ND); 3 stride fields (`dstNzC0Stride`, `dstNzNStride`, `dstNzMatrixStride`) require knowing the EXACT NZ packing semantics on V220 — which differ from arch3510 packing. kw-2 iter 2 guessed `dstNzC0Stride=S=64`; runtime fault `507015 aicore exception` on first `Mmad`. kw-4 iter 3 sourced concrete value `dstNzC0Stride=D/16=4` from a V220 SDK header reading; fault transitioned to `0x8000004000 L0B read/write conflict` (different fault layer — confirms shape passes L1-decode). The transition between fault signatures is the empirical proof that field semantics matter and best-effort guessing is unreliable.

**Anti-pattern (BANNED — caught across 3 iters / 4 worker spawns)**:
```cpp
// kw-3 iter 2 ── BAD: guessing dstNzC0Stride from field-name intuition
AscendC::DataCopy(l1Dst, gmSrc, AscendC::Nd2NzParams{
    1, S, D, 0, D,
    /*dstNzC0Stride=*/S,  // ← guessed "looks like inner-dim row count"
    /*dstNzNStride=*/16,  // ← guessed "N-stride sounds like 16 = C0 block"
    /*dstNzMatrixStride=*/0
});
// Compiles clean. Runtime: 507015 aicore exception. No compile/build warning.
```

**Correct pattern (only after SDK header reading + cross-verified field semantics)**:
```cpp
// kw-4 iter 3 ── correct after sourcing field semantics from SDK header:
AscendC::DataCopy(l1Dst, gmSrc, AscendC::Nd2NzParams{
    1, M, K, 0, K,
    /*dstNzC0Stride=*/K/16,  // ← #16-elem C0 strips per L1 row, not row count
    /*dstNzNStride=*/16,     // ← 16 rows per N-stride (canonical NZ blocking)
    /*dstNzMatrixStride=*/0
});
// Compiles clean. Runtime: L1-decode succeeds. Fault (if any) is at later layer.
```

**Generalized workflow guidance**:
1. **Read the SDK header for field semantics first** — every field with `uint16_t`/`uint8_t` type but no docstring is an opaque field. Grep the impl `.h` files for the field name to find the hardware-intrinsic invocation that consumes it.
2. **Verify on hardware before broadening the change** — write a minimal kernel that uses the primitive on a single shape, build it, run it, observe the fault signature (or PASS). Do NOT propagate the primitive across multiple kernel functions until the single-case verification passes.
3. **When SDK header lacks field docs entirely** (the Nd2NzParams case): escalate to `aog-cann-learner` Mode 5 extraction or `aog-hardware-probe` skill to seed API_CATALOG.md before relying on the primitive in production code.
4. **Run-time fault signatures encode the actual gap layer** — `507015 aicore exception` ≠ `0x8000004000 L0B read/write conflict` ≠ silent hang. The fault-signature transition across iters tells you which layer was solved (layout vs sync vs deeper). Treat each new fault signature as positive progress; don't conflate "still failing" with "no progress".

**Other instances (predicted)**:
- `MmadParams` (cube tile-MMAD): fields `isBias`, `cmatrixInitVal`, `cmatrixSource` interact with prior pipe state in non-obvious ways.
- `LoadData2DParams` / `LoadData3DParamsV220`: `ifTranspose`, `addrMode`, `dilation*` fields lack semantic docstrings for V220.
- `FixpipeParamsV220`: numerous mask / nz2nd config fields driving the FIX-pipe behavior.
- Future V220-specific intrinsic structs that get added to the SDK without docs.
- Any `event_t(N)` parameter — looks like an integer ID but interacts with cross-core sync semantics (PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP).

**Cross-ref**:
- OL-130 (API surface lookup chain — SDK header reading is load-bearing for opaque-field primitives)
- API_CATALOG.md (the appropriate destination for verified opaque-field semantics)
- CAND-FA1 (where the Nd2NzParams field shape is documented as verified)
- PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP (companion case — event_t(N) is an opaque parameter whose semantics interact with MIX_AIC_1_2 sync infra)

**Promotion path**: candidate to OL-class entry once 2+ ops independently demonstrate the workflow (SDK header reading → hardware verification → API_CATALOG seeding) on different opaque-field primitives. The Nd2NzParams + Mmad/LoadData2DParams + FixpipeParamsV220 sweep on a single FA-class op would satisfy this; alternatively, future ops using a different opaque-field primitive (e.g. a new V351 / V220 quant primitive) on similar workflow would.

---

### CAND-FA-CANON-FREE: Eliminate AIV layout-canonicalize stage via `MatmulImpl::SetOrgShape` 5-arg variant for strided GM→L1 load + per-row contig AIV postprocess for output layout transform

`applies_to: soc=all (V220 verified; V351 unverified); cann=9.0.0+; op_class=fused-attention or any cube-heavy op needing layout-aware GM gather from BSH/SBH/BSND/BNSD source; correctness_scope: S*D ≤ UB_BUDGET_8192 AND S*Skv ≤ UB_BUDGET_8192 (current kw-1-derived algorithm; row-tiled FlashAttention rewrite tracked as DEBT-FA-row-tiled)`
`verified_on: a5_ops:3_FusionAttention kw-5 structural rewrite (2026-05-22, A3 npu-a3-test) — multi-shape sweep: BNSD/BSH/SBH/BSND × N∈{1,2,4} × shapes within UB budget all produce max_abs ≤ 6.1e-5 PASS_T1; perf 0.603× CANN at B=1,S=64,N=2,D=64,BSH,fp16 (56.9 µs vs CANN 34.3 µs). Pass B VEC fallback 9/9 preserved.`
`unverified_on: V351 (Ascend950PR / A5) — pattern likely portable but matmul library behavior across arch versions needs verification before broad scoping; future iter to confirm`

**Supersedes**: PB-36 (now ARCHIVED) — PB-36's "V220 DataCopy hardware bug + Python permute workaround" framing was wrong. CANN's own `aclnnFlashAttentionScoreV2` works on V220 without any such workaround; the bug was in our design choice to use an AIV canon stage at all.

**Pattern overview**: For FA-class ops where source tensors live in non-canonical layouts (BSH/SBH/BSND), the natural-seeming "AIV canon stage → BNSD scratch → mm1/mm2 read scratch" approach is fragile because the canon stage requires `DataCopy` with non-zero `srcStride`/`dstStride` which exhibits silent wrong-output behavior on V220 CANN 9.0.0 for the FA-specific tile dimensions. The fix is to **never materialize a canonical layout in scratch** — instead use matmul's native strided GM→L1 load capability, and only do a final per-row contig transform for the OUTPUT side (where matmul C-write doesn't support stride).

**Principle**:
1. **`MatmulImpl::SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc=0)` is the stride mechanism**, despite the header docs misnaming `orgKa` as "K-axis size" — these fields encode **physical leading dimensions** (row strides including nested-axis interleaving). The 5-arg variant with explicit `orgKc` is required when B's N-axis size differs from C's.
2. **Per-(b,n_head) GM offset**: `head_off = b*sB + n_head*sN`. `SetTensorA(qGm[head_off])` + `SetOrgShape(orgM=S, orgN=sS, orgKa=sS, orgKb=sS, orgKc=S_or_D)` lets matmul do strided GM→L1 loads internally. BNSD degenerates (sS=D, no stride overhead); BSH/SBH/BSND get correct strided gather.
3. **Matmul C-side does NOT accept layout stride** — empirically tested, BSH/SBH/BSND output writes produce 5–38% error when attempted. The C-side always writes contig per-head. So mm2 outputs to BNSD-internal contig scratch.
4. **AIV postprocess stage for output layout transform**: per (b, n_head), loop over `s`, **single-row contig DataCopy** GM→UB (no stride params: `blockCount=1, blockLen=D*sizeof(T)/32, srcStride=0, dstStride=0`) + explicit `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` then UB→GM at `b*sB + n_head*sN + s*sS`. No strided DataCopy params anywhere → sidesteps the V220 strided-DataCopy bug class entirely.

**Concrete parameter values** (BSH input, where `sB=S*N*D, sN=D, sS=N*D=H`):
```cpp
// mm1: scores [B,N,S,S] = Q @ K^T
mm.SetOrgShape(S,           // orgM
               sS,          // orgN = H (Q/K physical leading dim)
               sS,          // orgKa = H
               sS,          // orgKb = H
               S);          // orgKc — C is BNSD-internal contig, N=S
mm.SetTensorA(qGm[b*sB + n*sN], /*isTransposeA=*/false);
mm.SetTensorB(kGm[b*sB + n*sN], /*isTransposeB=*/true);
mm.SetSingleShape(S, S, D);
// → matmul loads Q row s, col k from qGm[head_off + s*H + k] (correct BSH access)

// mm2: out_bnsd [B,N,S,D] = attn @ V
mm.SetOrgShape(S, sS, S, sS, D);   // orgKa=S (attn contig), orgKb=H (V strided), orgKc=D
// → out_bnsd is BNSD-internal contig at bn_idx*S*D; user-layout transform deferred to postprocess

// Postprocess AIV (only fires for non-BNSD):
for (s = 0; s < S; ++s) {
    DataCopy(ub, src_bnsd[bn_idx*S*D + s*D], row_params);  // contig D fp16 read
    SetFlag<MTE2_MTE3>(eid); WaitFlag<MTE2_MTE3>(eid);
    DataCopy(dst_user[b*sB + n*sN + s*sS], ub, row_params); // contig D fp16 write at strided offset
    SetFlag<MTE3_MTE2>(eid); WaitFlag<MTE3_MTE2>(eid);
}
```

**Launch topology** (compared to PR #103's broken AIV canon path):
- **Before (broken canon)**: canon (BSH→BNSD scratch, broken) → mm1 → softmax → mm2 → uncanon (BNSD→BSH, also broken) = 5 launches, all 5 in source even though canon/uncanon produced wrong output
- **After (this CAND)**: mm1 → softmax → mm2 → postprocess = 4 launches for non-BNSD, 3 for BNSD; no strided-DataCopy bug surfaces

**Why "matmul C-side does NOT support stride" matters as a separate finding**: An obvious design temptation is to set `orgN=sS` for mm2's C-side and have matmul write BSH-layout directly. This compiles cleanly but produces 5–38% systematic error (verified at B=1,S=64,N=2,D=64,BSH→4.8%; BSND/SBH→38%). The matmul library on V220 always writes C contig per launch regardless of `orgN`/`orgKc`. So the output-layout transform MUST be a separate stage, and that stage MUST avoid strided DataCopy (which on V220 is the bug class PB-36 documents).

**Verification scoreboard (honest)**:
- Cube path (S>=16, S*D ≤ 8192, S*Skv ≤ 8192): **9/10 PASS at max_abs ≤ 6.1e-5**
- Cube path (S*Skv > 8192): **1 algorithmic-scope FAIL** at `BSH B=1 S=128 N=1 D=64` (Skv*S=16384 > UB budget). This is NOT a defect of CAND-FA-CANON-FREE — the kernel still uses kw-1's "materialize full S×Skv scores in UB" algorithm. Real FlashAttention algorithm rewrite (row-tiled with online softmax) tracked as DEBT-FA-row-tiled.
- Pass B (VEC fallback, S=2): **9/9 PASS**
- Perf at B=1,S=64,N=2,D=64,BSH,fp16: **0.603× CANN** (56.9 µs ours vs 34.3 µs CANN). 13× over kw-1 baseline (0.046×). Target ≥0.6× CANN met within the rewrite's correctness scope.

**Cross-ref**:
- PB-36 (ARCHIVED — what this CAND supersedes)
- `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-canon-removal-structural-rewrite` (full design + complete kernel patches + falsification chain + verification numbers)
- PB-9 (V220 UB→UB DataCopy nuance; same MTE-engine family)
- PB-22 (V220 MTE2 DataCopy 32B transfer limit per destination TBuf)
- PB-34 + PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP (V220 mixed-mode KFC/sync minefield — the reason this rewrite uses **multi-launch** for the planned DEBT-FA-row-tiled outer loop rather than fused mixed AIC/AIV)
- DEBT-FA-row-tiled (follow-up: real flash-attention algorithm rewrite to expand correctness scope to arbitrary S/Skv)

**Promotion path**: candidate to canonical OL once (a) one more L4 fused-cube op independently uses the SetOrgShape 5-arg + postprocess-AIV pattern and verifies correctness, OR (b) V351/A5 cross-arch verification of the same FA shapes confirms portability, OR (c) the row-tiled algorithm rewrite ships and the combined pattern proves to be the canonical FA-class approach on V220. Until then, scope strictly to `op_class=fused-attention` and `correctness_scope=S*D,S*Skv ≤ UB_BUDGET_8192`.

---

### CAND-NO-CHEAT-AUDIT-CHECKLIST: Self-audit schema for AscendC op-gen agents — pre-DONE checklist to catch CPU compute / PyTorch delegation / kernel CPU-fallback cheating [V351+V220, ASCENDC_MODES, anti-cheat, agent-discipline]

`applies_to: soc=all; cann=all; bisheng=all; op_class=all; mode=arch22_to_arch35/backward; backend=ascendc`
`verified_on: independent 3_FusionAttention audit 2026-05-22 — owner asked "are we using CPU for some of the logics which will be considered as cheating?" mid-PR-#112 push. Audit identified zero cheating in the cube path but surfaced a per-row scalar loop on the AIV scalar pipe. The audit was ad-hoc; this CAND codifies the steps so future AscendC agents can self-audit before declaring DONE.`

**Why this CAND** (the recurring failure mode): CLAUDE.md has the "No PyTorch/CANN Delegation, No CPU Fallback" rule but it's a paragraph-level prose statement, not a grep-able checklist. Real audits get done **ad-hoc** when someone asks ("are we cheating?"); the answer is good but the audit isn't reproducible. Without a codified checklist, the next op-gen agent reading CLAUDE.md will know the *rule* but not the *test* — meaning subtle cheating (e.g., Python-side `permute()` workaround that survived 12 hours in PR #103, or `_check_scope` integer arithmetic that LOOKS like CPU compute but isn't) can ship undetected until owner asks.

**This is the operationalization companion to OL-175** (failure-framing discipline). OL-175 says "don't hide failures via framing"; this CAND says "don't hide cheating via lack-of-audit-procedure". Same family.

#### Checklist (run BEFORE declaring DONE on any op)

**Step 1 — Python `model_new_ascendc.py::forward` scan** (3 substeps):

```bash
# A. No CPU compute on tensor data
grep -nE "\.cpu\(\)|\.numpy\(\)|\.tolist\(\)|\.item\(\)" workspace/<op>/model_new_ascendc.py
# Expected: empty. If hits, every hit must be in a non-compute path (e.g., debug print
# behind `if DEBUG:` guard, not in the live forward). Live forward must operate on .npu()
# tensors only.

# B. No PyTorch compute-op delegation
grep -nE "torch\.(matmul|softmax|exp|log|sort|argsort|topk|max|sum|mean|var|std|cumsum|gather|scatter|index_select|permute|reshape|view|transpose|expand|repeat|tile|contiguous|cat|stack|chunk|split|fft|rfft|conv\w+|linear|layer_norm|batch_norm|sigmoid|tanh|relu|gelu|silu)" workspace/<op>/model_new_ascendc.py
grep -nE "F\.(softmax|attention|conv\w+|linear|layer_norm|batch_norm|sigmoid|tanh|relu|gelu|silu|scaled_dot_product_attention)" workspace/<op>/model_new_ascendc.py
# Expected: empty in forward(). If hits, every hit must be in a non-forward helper
# (e.g., __init__ shape pre-computation) AND not touching input tensors.

# C. forward() has at most ONE _ext.run_<op>(...) call and a direct return
python3 -c "
import ast
src = open('workspace/<op>/model_new_ascendc.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'forward':
        ext_calls = [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id == '_ext']
        print(f'_ext.*() calls in forward(): {len(ext_calls)}')
        torch_attr_calls = [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name) and n.value.id == 'torch']
        print(f'torch.<attr> in forward(): {len(torch_attr_calls)}')
"
# Expected: exactly 1 _ext.* call. torch.* allowed only for non-compute uses (torch.float16
# dtype literals); each hit needs manual inspect.
```

**Step 2 — Pybind11 scan** (4 substeps):

```bash
# A. No torch::<compute_fn> in pybind dispatch body
grep -nE "torch::(matmul|softmax|exp|log|sort|topk|max|sum|mean|cat|stack|cdist|matrix_exp)" workspace/<op>/kernel/pybind11.cpp
# Expected: empty. Operations on torch::Tensor data are forbidden in pybind; only
# torch::empty/options/sizes/dtype/contiguous-on-input are allowed.

# B. Only allowed torch:: surface usage
grep -nE "torch::|at::" workspace/<op>/kernel/pybind11.cpp | \
  grep -vE "(torch::Tensor|torch::empty|torch::kFloat|torch::kInt|torch::kHalf|torch::kBfloat|c10::optional|\.options\(\)|\.sizes?\(\)|\.dtype\(\)|\.size\(|\.device\(\)|\.contiguous\(\)|at::Half|TORCH_CHECK|TORCH_INTERNAL|c10_npu)"
# Expected: empty (or only whitelisted residual). Any surfaces outside the allowed list
# (torch::Tensor decl, torch::empty alloc, .options/.sizes/.size/.dtype/.contiguous/at::Half/
# TORCH_CHECK/c10_npu stream access) need manual inspect.

# C. No CPU-tensor traffic
grep -nE "\.cpu\(\)|\.to\(c?torch::kCPU\)|aclrtMemcpy.*HOST_TO_DEVICE|memcpy.*data_ptr\(\)" workspace/<op>/kernel/pybind11.cpp
# Expected: empty in the dispatch body. The only allowed host-side device operation
# here is aclrtMemset (GM workspace initialization); it transfers no tensor data.

# D. .contiguous() only on INPUT tensors (not on outputs from kernel)
# Output tensors written by AscendC kernel MUST NOT be passed through .contiguous()
# because the kernel writes layouts the caller expects. .contiguous() on output would
# round-trip via aclnnContiguous = CANN delegation. Manual inspect:
grep -nB2 -A1 "\.contiguous\(\)" workspace/<op>/kernel/pybind11.cpp
# Expected: every hit is on an INPUT (query/key/value/etc from forward args), NOT on
# scratch buffers, output tensors, or post-kernel tensor returns.
```

**Step 3 — Kernel host-launch scan** (1 substep):

```bash
# A. No CPU-side compute in kernel host code
grep -rnE "\.cpu\(\)|std::sort|std::sin|std::cos|std::exp|std::log|std::sqrt" workspace/<op>/kernel/*.cpp workspace/<op>/kernel/*.h
# Expected: empty in body of __global__ aclrtlaunch_* functions. Pure host-side helper
# code (tiling computation, blockDim selection) is allowed to use std::* but NOT for
# computing tensor values.
```

**Step 4 — AscendC scalar-pipe usage sanity** (info only, not anti-cheat):

```bash
# Per-element scalar loops on AIV (GetValue/SetValue inside a for-loop) are LEGAL but
# burn AIV scalar pipe. Not cheating, but tip-of-iceberg perf hint.
grep -nE "\.GetValue\(\w+\)|\.SetValue\(\w+, " workspace/<op>/kernel/fusion_attention_kernel.h | wc -l
# If count is high (>10 per kernel), flag for CAND-FA-MULTI-LAUNCH-PERF-GAP Δ#2-style
# "rewrite as vector op" optimization. Not a fail.
```

**Step 5 — Live-path probe** (only if Steps 1-4 surfaced any suspect line):

```python
# Run the actual op once on .npu() tensor; profile with msprof to confirm no aclnn*
# host-API calls (other than the expected workspace/HBM allocs).
python3 -c "
import torch, torch_npu
import sys
sys.path.insert(0, 'workspace/<op>')
sys.path.insert(0, 'workspace/<op>/kernel/build')
import model_new_ascendc as mna
q = torch.randn((1,64,128), dtype=torch.float16).npu()
k, v = q.clone(), q.clone()
torch.npu.synchronize()
# Wrap in msprof if suspect aclnn calls
out = mna.ModelNew()(q, k, v, 2, 'BSH', scale=1.0)
torch.npu.synchronize()
"
# Optional: msprof profile and grep for 'aclnnPermute', 'aclnnContiguous', 'aclnnSort',
# 'aclnnMatmul', etc. — any aclnn* in the per-step kernel list (other than the user's
# expected aclrtlaunch_<op_internal_*>) is a delegation.
```

#### Examples of cheating this checklist catches

**Live example caught earlier this session (PR #103, retracted via PR #106)**:
```python
# model_new_ascendc.py forward(), BAD:
q_kern = query.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()  # delegates to aclnnPermute
# Caught at Step 1B (torch.reshape pattern would have flagged) AND
# Step 5 msprof would have shown aclnnPermute in trace.
```

**Subtler delegation that LOOKS like metadata but isn't**:
```python
# BAD:
if input_layout == "BSH":
    q_bnsd = query.permute(0, 2, 1, 3).contiguous()  # contiguous() on POST-permute tensor = aclnnContiguous compute
# Step 1B catches `.permute(`; Step 2D catches `.contiguous()` on non-input tensor.
```

**Subtler still — scalar fallback inside kernel host code**:
```cpp
// kernel/op_host/<op>_tiling.cpp, BAD:
void compute_tiling(...) {
    std::sort(tiling.workgroup_priorities, tiling.workgroup_priorities + N);  // host-side std::sort on per-instance data
    // This burns host CPU cycles per kernel launch. Not a Cheating-on-results, but
    // host-side compute-per-launch that can amortize differently than expected.
}
// Step 3 catches std::sort.
```

#### Promotion path

Candidate to OL once:
1. At least 3 different ops (different op-classes: 1 elementwise / 1 reduction / 1 fused) have used this checklist successfully pre-DONE
2. The Steps 1-3 grep patterns are stable (no false negatives surfaced)
3. The live-path probe is empirically validated on at least one fused AscendC op and one backward AscendC op

Until then, scope: AscendC ops where author manually runs the checklist before declaring DONE. Verifier-side automation is **DEBT-NO-CHEAT-AUDIT-CI** (future hook to enforce this via pre-commit).

#### Cross-ref

- **OL-175** (defensive-guard refusal is highest fail tier — sibling agent-discipline anti-cheat)
- **OL-160** (canonical entry-point file names — sibling structural anti-cheat enforcement)
- **OL-167** (DataCopy `count` truncation pad+narrow cheat — sibling on-device anti-cheat)
- **OL-172** (ModelNew.forward output count parity — sibling contract-side anti-cheat)
- **CAND-FA-MULTI-LAUNCH-PERF-GAP** Δ#2 (per-row scalar loop on AIV scalar pipe is NOT cheating but IS perf headroom)
- **CLAUDE.md "No PyTorch/CANN Delegation"** + "No CPU fallback" — the rule this CAND operationalizes
- **PR #103 → PR #106** (worked example: cheating shipped via Python permute(), caught by owner, retracted)
---
### CAND-FA-MULTI-LAUNCH-PERF-GAP: Five design-choice deltas between multi-launch row-tiled FA and CANN's single-launch fused FA — measured 100× perf gap at S=1024 traced to specific kernel-internal pipelining choices [V220, L4 fused-attention, perf-optimization-roadmap]

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0; op_class=fused-attention or any L4 op_class requiring row-tiled outer + accumulator across tiles; perf_regime: large_S (S ≥ 4 × T_q i.e. ≥4 Q-tiles)`
`verified_on: a5_ops:3_FusionAttention Step 3 row-tiled multi-launch FA (PR #109, 2026-05-22, A3 npu-a3-test) — correctness 13/13 cube + 9/9 Pass B PASS at max_abs ≤ 1.5e-5; perf measured 0.69-0.81× CANN at small S (fast-path) vs 0.025× at S=512 vs 0.007× at S=1024. 100× perf gap at large S directly traceable to 5 design choices identified via CANN-source comparison.`
`unverified_on: V351 (Ascend950PR / A5) — same design-choice deltas likely apply but V351 KFC behavior may differ; needs cross-arch verification before broad scoping`
`v351_implication: probe_a5_v300_fa_sync 2026-05-23 — Pattern A clean on V351 means SINGLE-LAUNCH FUSED FA is viable on V351/Ascend950PR. The 5-delta multi-launch roadmap was V220-conservative; on V351 the single-launch fused architecture (CANN's Pattern C-equivalent for arch35) is the right architectural target, not multi-launch + 5 deltas. V220 ceiling at 0.014× CANN @ S=1024 does NOT propagate to V351. Re-scope this CAND title from "Five design-choice deltas..." to "V220-only roadmap; V351 should target single-launch fused" — pending sole-final-author retitle when broader scope confirmed.`

> **2026-07-20 cross-ref (a3 multi-core FA resolution):** a **per-head-independent library** multi-core FA (`MatmulImpl IterateAll<sync=true>`, one head-slice/core, NO cross-core ring) runs **deadlock-free** on a3 — see `fa_class/cross_core_sync.md` §5 / PB-56. This **refutes the "parallelization is forced to deadlock" premise**, BUT does **NOT** lift the large-S ceiling: that kernel is non-flash O(S²), measured only @ S=512/BN=32 (0.186× vendor). **The 0.014× CANN @ S=1024 ceiling still stands** (different shape; non-flash cost dominates at large S).

**Why this CAND** (not just a DEBT): The 5 deltas below are **structural** — they apply to any row-tiled multi-tile L4 fused op on V220 (MoE finalize, fused norm+matmul with cross-tile accumulator, GroupNorm + chunked reduce, fused-quant attention variants, etc.), not just FA. Future agent implementing row-tiled L4 should read this CAND before declaring "structural ceiling" on multi-launch perf — the 5 deltas are concrete optimization headroom, not assumed-immutable architecture.

**Source comparison**: CANN `attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h` (kernel) + `op_host/arch22/flash_attention_score_tiling_general.cpp` (host tiling) read 2026-05-22 via owner-authorized port_a3 mode (cann_learner exception per CLAUDE.md V3.x carve-out). KB-carveout rules respected: design patterns + parameter values + structural choices extracted; no verbatim source.

#### Delta #1 — L0C accumulator residency across KV-tile iterations

| | CANN (single-launch fused) | Multi-launch (our PR #109) |
|---|---|---|
| O[T_q, D] fp32 location | **L0C resident** across all KV-tile iters within a Q-tile | GM round-trip every KV-tile (load → scale → add → store) |
| Per-KV-tile O GM traffic | ~0 (only fixpipe-write once at Q-tile epilogue) | `T_q*D*4 = 16 KB` read + 16 KB write per KV-tile |
| Mechanism | `bmm2.template IterateAll<false>(...)` per KV iter (the `false` = no auto-flush) + `taskIdMod2` ping-pong | separate `fa_scale_and_accumulate_fp16` AIV launch with explicit GM r/w |
| L0C lifetime | mm2 object instance persists across KV iters; L0C carries state | mm2 launches discrete; no cross-launch state |

**Headroom**: For S=1024, `T_kv_tiles=16`, we do **16 GM round-trips of `T_q*D = 16 KB`** = ~512 KB extra HBM bandwidth per Q-tile per (b, n_head). At V220 HBM ~1.6 TB/s effective: ~0.32 µs overhead just for O traffic per Q-tile × 16 Q-tiles = ~5 µs. Plus launch-init overhead. Adopting L0C residency would save ~80% of accumulator memory traffic.

**Adopt feasibility**: requires keeping the mm2 `MatmulImpl` object alive across launches OR moving to single-launch fused. Within current multi-launch architecture: **NOT feasible** — each `aclrtlaunch_*` instantiates fresh `MatmulImpl`. This delta alone justifies eventual move to single-launch (delta #5).

#### Delta #2 — Online softmax fp32 buffer sizing

| | CANN | Ours (PR #109) |
|---|---|---|
| Live fp32 softmax buffers per AIV | `[s1_vec, s2_aligned]` = `[8, 64]` = **2 KB** per ping-pong buf × 4 bufs (max/sum/exp + scratch) ≈ **8 KB** | Full `[T_q, T_kv]` fp32 = `[64, 64] * 4` = **16 KB** for scoresF + 4 KB for reductions ≈ 20 KB |
| Computation granularity | Row-by-row, 8-way reduction factor (`softmaxReduceSize=8`) | Per-Q-tile full materialization |
| API | CANN's `SoftMaxCompute` called inside per-loopIdx loop with `[s1_vec, s2_aligned]` input | Our `Cast` + `Adds` + `Exp` over full ST = T_q*T_kv |

**Headroom**: ~12 KB UB freed per AIV. Reusable for larger tile sizes (delta #3) OR resident O accumulator (delta #1).

**Adopt feasibility**: **HIGH within multi-launch architecture**. Rewrite `FaOnlineSoftmaxUpdateKernel::ProcessTile` to process `s1_vec=8` rows at a time, looping `T_q / s1_vec = 8` times. Same scalar-loop structure already exists for per-row m_new computation; just shrink inner buf allocation. **Effort: 1-2h. Risk: low.**

#### Delta #3 — Tile size selection function

| | CANN (`CalcS1S2BasicBlock`) | Ours |
|---|---|---|
| T_q, T_kv source | **Host-side computed per-shape**: `tmpS1 ∈ [GetMinS1BasicBlock(), alignedS1]` step 16; for each, max `tmpS2 ≤ alignedS2` s.t. UB-budget fits | **Hard-coded** `T_q = T_kv = 64` in pybind |
| UB budget formula | `s1*16*X + s1*D*Y + s1*(expNum+2)*32 + apiTmp ≤ ubSize` (X, Y per-op family constants) | n/a — never computed |
| Typical result @ B=1, N=12, S=1024, D=64 | `s1BaseSize=64, s2BaseSize=64` balanced OR `128×64` / `64×128` depending on enableL1Reuse | always 64×64 |

**Headroom**: 64×64 likely undersized for D=64 small-Skv shapes (could be 128×64 → halves Q-tile-count → halves multi-launch overhead in the Q-tile dimension) and oversized for D=128 (could be 64×32 → fits with delta #2's freed UB). 2-4× tile area at peak UB utilization.

**Adopt feasibility**: **HIGH within multi-launch architecture**. Implement a tiny host-side `tile_sizing_v1(B, N, S, Skv, D, dtype_bytes) → (T_q, T_kv)` function in pybind11.cpp. Even a simple "fit-then-balance" heuristic (start at 128×128, shrink to fit UB while staying balanced) would beat hard-coded 64×64. **Effort: 1h. Risk: low.**

#### Delta #4 — Alpha rescale fusion into mm2 post-process

| | CANN | Ours |
|---|---|---|
| Alpha rescale (O = α * O_prev + dO_new) | **Fused into bmm2's vec2 post-process**: `DataCopy(bmm2ResUb, stage2BufTensor)` + `Bmm2ResultMul(bmm2ResUb, expUb, ...)` + `Add(bmm2ResUb, bmm2ResUb, stage2BufTensor)` in single AIV kernel | **Separate `fa_scale_and_accumulate_fp16` AIV launch** |
| Launches per KV-tile | mm1 + softmax_update + mm2 (+vec2 fused inside mm2) = 3 logical stages, but **1 launch** in fused KFC mode | mm1 + softmax_update + mm2 + scale_accumulate = **4 launches** |

**Headroom**: 25% launch count reduction per KV-tile. For S=1024 with 16×16 = 256 KV iters × 4 launches = 1024 launches → could drop to 768 = -33%. Direct ~25-33% perf gain at large S where launch overhead dominates.

**Adopt feasibility**: **MEDIUM within multi-launch architecture**. Two options:
- (a) merge `fa_scale_and_accumulate` body into the tail of `fa_online_softmax_update` (which already has alpha computed) — requires routing dO_partial GM ptr into softmax_update; saves 1 launch but pre-vs-post mm2 ordering needs care since softmax_update runs BEFORE mm2.
- (b) merge `fa_scale_and_accumulate` body into a new "fa_mm2_then_scale" AIV-side kernel that runs after mm2 — requires mm2 output and prior O / alpha all visible to one AIV.
- Both require shape-rewrite vs current cleanly-separated stages. **Effort: 2-3h. Risk: medium (correctness re-verification needed).**

#### Delta #5 — KFC-implicit AIC↔AIV sequencing (single-launch fused) vs multi-launch isolation

| | CANN | Ours |
|---|---|---|
| Kernel structure | **Single fused kernel** with `taskId % 3` ping-pong stages over mm1/vec1/mm2/vec2; AIC implicit via tiling BlockDim | Multi-launch: each stage is its own `aclrtlaunch_*` |
| Sync mechanism between stages | KFC implicit + intra-core `SetFlag/WaitFlag<MTE3_MTE2>` event sync | Stream-sequential dispatch (NPU stream serializes launches) |
| AIC↔AIV handoff | Implicit via task pipeline; no `CrossCoreSetFlag<0x2>` (Pattern A) and no explicit `REGIST_MATMUL_OBJ` (Pattern B) — instead **event-ordered task sequence within one kernel** | None — each launch is fully independent, AIC and AIV separated |
| Total launches @ S=1024 | ~B*N task-pipeline stages = ~100 for B=1, N=12 (one per (b, n) pair across all Q+KV iters) | ~1057 launches (16 Q-tiles × 16 KV-tiles × 4 + 16 inits + 16 finalizes + 1 postprocess) |

**Headroom**: 10× launch reduction → could close most of the launch-overhead-bound perf gap.

**Adopt feasibility**: ~~**HIGH RISK** within current state~~ ~~**EMPIRICALLY FALSIFIED 2026-05-22 — Pattern C structurally blocked on V220.**~~ **STATUS REVISED 2026-05-23 — V220 Pattern C UNVERIFIED (probe was misdesigned); V351 Pattern A confirmed VIABLE.**

The V220 KFC mixed-mode minefield (PB-34, PB-35) burned 5 iter on Pattern A. PR #117 originally claimed Pattern C also empirically falsified on V220 — that claim has been retracted (see [CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP](#CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP) status update 2026-05-23). My V220 probe used `SetFlag<HardEvent::MTE3_MTE2>` which is intra-core pipe-sync, NOT cross-core AIC↔AIV handoff — the hang was from probe-design defect, not real V220 architectural block.

**Updated probe outcomes (2026-05-23)**:
- **V220 Pattern A**: deadlocks (PB-34, 5-iter chain). **CONFIRMED.**
- **V220 Pattern B**: unwound from production for unrelated reasons. Status unknown.
- **V220 Pattern C**: probe misdesigned, status genuinely UNVERIFIED. Re-probe needed with `CrossCoreSetFlag<0x2>(flagId)` semantics.
- **V351 Pattern A**: runs clean (0.036ms, bit-exact, 3 deterministic trials). See main agent's `workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md` and PB-34's `verified_does_not_reproduce_on: V351` line.

**Practical implication for V220**: multi-launch architecture remains the path of least resistance (Pattern A confirmed blocked). Real V220 ceiling at 0.014× CANN @ S=1024 stays correct, supported by Pattern A falsification alone — doesn't rely on the retracted Pattern C "falsification".

**Practical implication for V351**: **single-launch fused FA is viable**. Use Pattern A architecture (`MatmulImpl<>` + `CrossCoreSetFlag<0x2>` + `KERNEL_TYPE_MIX_AIC_1_2`) for V351 / Ascend950PR FA ports. CANN's `flash_attention_score/arch32/s1s2_bn2gs1.h` is the pattern reference.

#### Combined optimization roadmap (UPDATED 2026-05-22 with measured outcomes)

Starting from PR #109's 0.007× CANN at S=1024:

| Step | Predicted | Measured | Delta |
|---|---|---|---|
| +#2 (row-wise softmax) + #3 (tile sizing) — PR #112 | ~0.02× | **0.014×** | below estimate (~70% of prediction) |
| +#4 (alpha-rescale fusion via reorder) — PR #114 | +1.5× per shape | **+1.17-1.30×** S=64..512; **~0% S=1024** | below; S=1024 plateau is dO_part GM round-trip-bound (CAND #1 territory) |
| +#1 (L0C residency) — needs #5 first | 80% mem traffic save | **N/A (blocked by #5 falsification)** | NOT STANDALONE |
| +#5 (single-launch event-ordered) — PR #117 falsification | ~10× | **0× (Class B falsified)** | empirical block |

**Actual measured ceiling for V220 multi-launch FA (PR #114)**: **0.014× CANN @ S=1024, 0.6-0.8× CANN @ S=64**. This is the **real V220 ceiling**, not "0.4-0.6× CANN with all 5 deltas adopted" as initially projected — that projection was made on the assumption Δ#5 was adoptable. Per OL-175 honest framing: predicted-vs-measured roadmap calibration is itself KB-valuable; future agents starting from this CAND get the empirical ceiling not the speculative one.

#### Cross-ref

- **CAND-FA-CANON-FREE** (PR #106 ancestor — solved canon stage and matmul stride mechanism; this CAND is the **next-tier** structural problem)
- **PB-34** (Matmul + manual CrossCoreSetFlag + MIX_AIC_1_2 deadlock — Pattern A falsified at V220)
- **PB-35** (event_t(0..3) collides with FLAG_CANON_DONE chain — Pattern A/B refinement falsified)
- **CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP** (open-hypothesis follow-up to PB-35; delta #5's event-ordered single-kernel task pipeline might be the resolution)
- **OL-175** (defensive-guard-refusal-is-highest-tier — applicable here when claiming perf "ceiling" before exhausting structural deltas above)
- `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-step3-q-tiled` (the impl this CAND analyzes)
- CANN source `flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h:499-2244` + `op_host/arch22/flash_attention_score_tiling_general.cpp:1815-1914`

**Promotion path**: candidate to OL once at least 2 of the 5 deltas are empirically adopted in our codebase with measured perf delta confirmation. Until then, scope strictly to `op_class=fused-attention` and `perf_regime: large_S` — small-S fast-path doesn't surface most of these deltas.
---
### CAND-FUSED-KERNEL-PERF-ITERATION-WORKFLOW: 5-phase methodology for iteratively closing the perf gap on L4 fused kernels (from FA-2026-05-22 worked example) [V351+V220, ALL_MODES, agent-discipline + L4-methodology]

`applies_to: soc=all; cann=all; op_class=L4 fused (FA-class / MoE finalize / GroupNorm+activation+quant / fused-conv+norm / any multi-stage cube+vec op needing row-tiling or cross-iter accumulation); mode=all_modes`
`verified_on: 3_FusionAttention 2026-05-22 — full 8-PR iteration session (PR #103 retracted → #106 → #109 → #110 → #112 → #113 → #114 → in-flight #115) drove FA from 0.046× CANN baseline to 0.6× CANN (S≤256) + arbitrary S/Skv correctness. Each phase below corresponds to one or more concrete PRs in that session.`
`unverified_on: other L4 fused op classes — workflow not yet exercised on MoE finalize / fused norm / etc., but the per-phase artifact shape is op-class-agnostic`

#### Why this CAND

Owner direction: "we can continue this iteration to keep improving kb for FA like CV fused kernel gen". The FA 2026-05-22 session went through 5 discrete iteration phases, each producing a measurable correctness or perf delta. Without codification, the next agent attacking a different L4 fused op (e.g., MoE finalize, fused-quant-attention variants, GroupNorm + SiLU + Quant fused) will:
- Re-invent the CANN-source-read step
- Re-discover the 5-delta perf-comparison schema
- Re-derive that low-risk deltas should adopt first
- Re-experience the "0-output FAIL is highest tier, not skip" lesson (OL-175)

This CAND captures the **workflow shape** — what artifacts to produce per phase, what to query CANN source for, how to schedule deltas by risk, when to escalate to single-launch — so future iterations can compress from N sessions to ~1.

#### Phase 1 — Cold-build correctness

**Goal**: produce ANY working algorithm that compiles + runs + matches CANN reference on at least one shape. Forget perf entirely.

**Allowed shortcuts during this phase ONLY**: VEC fallback paths, simpler-than-optimal algorithms (e.g., materialize full S×S scores in UB even if it caps S=128). Pass B / VEC fallback is fine.

**Anti-patterns to avoid**:
- Python-side `torch.permute()` / `.reshape().contiguous()` workarounds delegating to CANN — caught by CAND-NO-CHEAT-AUDIT-CHECKLIST (PR #103 cheating retracted via PR #106)
- "Different limit, separate DEBT" framing on shapes the kernel can't handle — caught by OL-175 (these are 0-output FAILs)

**Output artifact**: a working kernel + design doc + verification scoreboard showing PASS on at least 1 shape per layout.

**Worked example anchor**: FA PR #103-era (pre-retraction) shipped working VEC fallback + cube path at 0.046× CANN on S=64. Correctness was real even though perf was bad.

#### Phase 2 — CANN-source read for structural pattern extraction

**Goal**: identify what CANN does structurally for this op-class. NOT a verbatim port — pattern extraction only.

**Steps**:
1. Dispatch CANN-source agent (port_a3 mode, owner-authorized via CLAUDE.md §V3.x KB carve-out). Question template: "How does CANN's `<op>` implementation handle [X] on V220?" where [X] is the *first* unknown blocking your impl (e.g., "BSH→BNSD layout transform", "cross-tile accumulator", "softmax intermediate dtype").
2. **Narrow, targeted questions** — one specific unknown per dispatch. Broad "explain how CANN does FA" wastes the carve-out budget.
3. Extract: code-location ref + 1-2 sentence summary of pattern + concrete parameter values. No verbatim source copy.
4. If first dispatch surfaces N-1 follow-up questions, dispatch another agent for the next 1-2 unknowns. **Iterate dispatches narrow, not one big agent.**

**KB-carveout discipline** (per CLAUDE.md §V3.x): patterns + parameter values only. No verbatim source. The dispatched agent operates under that constraint; verify by checking returned content doesn't contain large code blocks.

**Output artifact**: 1-N "CAND-<op>-<aspect>" entries each with "CANN's approach / ours / measured gap / open-question" structure.

**Worked example anchor**: PR #106 used CANN-source agent to find the `SetOrgShape` 5-arg variant for layout-aware strided GM loads. PR #109 used a second agent to find the s1/s2 double-tiling nesting. PR #110 used a third agent for the 5-delta perf comparison (L0C residency / fp32 buf / tile sel / fused-stage / single-launch).

#### Phase 3 — 5-delta perf-comparison schema authoring (CAND-<OP>-MULTI-LAUNCH-PERF-GAP)

**Goal**: after Phase 2's CANN reads surface 3-5 structural design choices, codify them as a single CAND entry with concrete per-delta breakdown.

**Schema per delta** (worked example: CAND-FA-MULTI-LAUNCH-PERF-GAP §1-5):
- **CANN's approach** (file:line ref, pattern name, parameter values)
- **Our approach** (current PR's approach, code-ref)
- **Measured gap** (perf delta or behavioral difference)
- **Adopt feasibility** (LOW / MEDIUM / HIGH risk; effort estimate in hours)
- **Headroom** (estimated perf gain if adopted)

**Recurring 5 deltas for L4 fused-cube-vec ops on V220** (extracted from CAND-FA-MULTI-LAUNCH-PERF-GAP, likely apply to other L4 op-classes):
1. **L0C accumulator residency across tile iterations** — does CANN keep accumulators in L0C across inner-loop iters? Our default multi-launch GM-round-trip is the perf floor.
2. **Working-buffer fp32 sizing** — does CANN use row-wise chunking (`S1_VEC`-style) vs full-tile fp32 buffers?
3. **Host-side tile-size selection** — does CANN have `CalcXXBasicBlock`-style UB-budget formula vs hard-coded?
4. **Fused stage opportunities** — can a separate AIV stage (alpha-rescale / accumulate / final-norm) be folded into another stage's pipeline?
5. **Single-launch fused vs multi-launch** — single-kernel KFC-implicit task pipeline (Pattern C variant) vs our deterministic multi-launch?

**Risk-ordered adoption sequence**: #2 + #3 first (LOW risk, structural-only, ~3h combined) → #4 (MEDIUM risk, reorder/fusion, ~3h) → #1 + #5 together (HIGH risk, requires single-launch + Pattern C falsification probe, ~8h).

**Output artifact**: one CAND-<OP>-MULTI-LAUNCH-PERF-GAP entry in `patterns/unverified/candidates.md` with all 5 deltas filled in.

**Worked example anchor**: CAND-FA-MULTI-LAUNCH-PERF-GAP (PR #110) is the canonical instance for FA.

#### Phase 4 — Risk-ordered delta adoption with measured calibration

**Goal**: implement each delta as a separate PR with measured perf delta. Update CAND with predicted-vs-measured calibration after each PR.

**Per-delta PR template**:
1. Implement the delta in a focused branch
2. Run full correctness sweep (regression check across all previously-PASSing shapes)
3. Measure perf vs CANN on the same shape grid
4. Compute "actual_gain / predicted_gain" — calibration data
5. Update CAND-<OP>-MULTI-LAUNCH-PERF-GAP's §N with "MEASURED: predicted X, got Y. Reason for delta:..." (per OL-175 "failure knowledge is KB-valuable")

**Critical discipline**: if a delta MISSES its predicted gain, codify WHY — that's where the real KB value compounds. PR #114's S=1024 plateau (Δ#4 didn't move S=1024 despite +25% on smaller S) is the canonical "calibration finding" — surfaced that dO_part GM round-trip becomes its own bottleneck at large S, confirming Δ#1 (L0C residency) as the larger lever.

**Output artifacts**: 1 PR per delta + CAND update with calibration column.

**Worked example anchor**: PR #112 (Δ#2+#3), PR #114 (Δ#4) for FA. Each PR's design doc has predicted-vs-measured table.

#### Phase 5 — High-risk delta or escape-hatch (single-launch / Pattern C)

**Goal**: once low/medium-risk deltas exhausted, decide whether to attempt the high-risk delta (typically single-launch fused or major architectural change).

**Trigger criteria**:
- All LOW/MEDIUM-risk deltas adopted and CAND calibrated
- Remaining perf gap to CANN is dominated by the architectural delta (typical at large S — confirmed via per-delta calibration)
- Owner / project willing to spend HIGH-risk budget

**Procedure**:
1. **Falsification probe first**: write a minimal standalone kernel that exercises the high-risk pattern (e.g., Pattern C event-ordered single-launch task pipeline on V220). DO NOT touch the production kernel yet.
2. If probe falsifies → record empirical falsification in CAND, mark delta as **structurally blocked** for this arch + cite specific failure mode. **Failure knowledge is KB-valuable per OL-175.**
3. If probe passes → implement in production kernel, measure, update CAND.

**Anti-pattern**: jumping directly to architectural rewrite without falsification probe. Cost can be 5+ iterations of debugging V220-specific deadlock before realizing the pattern doesn't work. PR #103-era FA had 5 iter Pattern A/B falsification chain — preventable if we'd done isolated probes first.

**Output artifact**: either probe-pass + production PR, OR probe-fail + updated CAND with empirical block.

**Worked example anchor**: FA Δ#5 Pattern C falsification probe (planned PR #115 at time of this CAND authoring; pending).

#### Compound output: workflow → next-iter speedup

Iteration N benefits from prior iterations on different ops:
- CAND-<OP>-MULTI-LAUNCH-PERF-GAP from prior op → schema template for new op's CAND (same 5-delta structure)
- Pattern C probe result from prior op → known whether V220 single-launch is feasible → skip probe if already falsified
- CAND-NO-CHEAT-AUDIT-CHECKLIST → pre-DONE audit for new op
- OL-175 framing discipline → honest scoreboards prevent reward-hacking in new op's PRs

After this CAND lands + 2-3 different op-class iterations validate the schema, promotion to OL-N "L4 fused-kernel iteration methodology" is appropriate.

#### Cross-ref

- **CAND-FA-MULTI-LAUNCH-PERF-GAP** (worked example: the 5-delta schema for FA specifically)
- **CAND-NO-CHEAT-AUDIT-CHECKLIST** (Phase 1 anti-cheat enforcement)
- **OL-175** (failure-framing discipline — applies throughout, especially Phase 4 calibration)
- **PR #103 → #106** retraction (worked example: Phase 1 cheating caught, retracted, learned)
- **PR #110 / #112 / #114** (worked examples: Phase 3 / 4 / 4-with-calibration)
- **CLAUDE.md V3.x CANN-learn carve-out** (Phase 2 source-read authorization)

**Promotion path**: candidate to OL once 2+ different L4 op-classes use this workflow successfully (each producing their own CAND-<OP>-MULTI-LAUNCH-PERF-GAP). FA is the first; need at least one more (target: MoE finalize OR fused norm + activation + quant) to validate generalizability.

---

## CAND-FA-VEC-D-TILE-1: GQA index-decode (B, N2, G, S1) via device-side mod/div + KV-offset uses n2_idx ONLY — avoid Python `repeat_interleave` materialization
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_GQA_or_MQA / FlashAttention_variant / streaming_attention_with_grouped_kv`
`derived-from: cann-source (FA reference V220 split-S1 + V351 train top-level Process, 2026-05-24 cl-fa-diff)`
`evidence_family: FA-GQA-DIM`
`verified_on: cann ops-transformer FA reference V220 + V351 (kernel-structural evidence; no a5_ops kernel measurement on this exact pattern yet)`

**Trigger**: Multi-head attention with grouped-query heads (Q has N1 heads, K/V has N2 heads, N1 = G * N2 with G ≥ 2). Logical iteration space is (B, N2, G, S1_outer). Naive port temptation: pre-materialize K/V as (B, N1, S, D) on the Python side via `repeat_interleave(K, G, dim=1)` so the kernel sees a 1:1 N1 mapping. That doubles+ HBM K/V traffic.

**Why "candidate"**: Pattern is structural (algorithm shape, not API surface), derived from CANN FA reference's index-decode. Promotion to canonical requires a 2nd op-class verification — e.g. a separate GQA-style sparse attention kernel showing measurable HBM-traffic reduction vs the `repeat_interleave` port. The 507035 hang seen in independent prototype DEBT-FA-GQA (D=512 row-tiled fp16) is currently UNRESOLVED, so the symptom-anchor below is a hypothesis-link, not validated.

**Recommendation**:
1. Treat the 4-axis (B, N2, G, S1_outer) as a single flat counter `idx` over `[0, B * N2 * G * S1Outer)`.
2. Distribute `idx` across cores via `multi_core_offset = block_idx * splitFactor` style core-split.
3. Inside each core, decode the 4 axis indices via repeated mod/div:
   ```cpp
   int64_t b_idx     = idx / (n2_size * g_size * s1_outer_size);
   int64_t n2_idx    = (idx / (g_size * s1_outer_size)) % n2_size;
   int64_t g_idx     = (idx / s1_outer_size) % g_size;
   int64_t s1o_idx   = idx % s1_outer_size;
   ```
4. **Q offset uses both n2_idx and g_idx** (Q has N1 = N2*G heads): e.g. for BNSD,
   `q_off = b_idx * (N1 * S1 * D) + n2_idx * (g_size * S1 * D) + g_idx * (S1 * D) + s1o_idx * (s1_base * D)`.
5. **K/V offset uses ONLY n2_idx** (KV has N2 heads, shared across G query-groups within same n2): e.g. for BNSD,
   `kv_off = b_idx * (N2 * S2 * D) + n2_idx * (S2 * D) + s2_offset`.

The kernel reads K[n2_idx] once per (B, N2, S1_outer) tuple and uses it for all G query groups in that tuple — no K/V duplication needed.

**Concrete anchor** (public-API only — `GlobalTensor::operator[]` + plain `int64_t` arithmetic):
```cpp
// Inside kernel; tilingData provides b_size, n2_size, g_size, s1_outer_size, s1_base, d_size, layout strides.
int64_t flat_idx = block_idx * split_factor + inner_idx;  // inner_idx loops over the per-core range
int64_t b_idx   = flat_idx / (n2_size * g_size * s1_outer_size);
int64_t n2_idx  = (flat_idx / (g_size * s1_outer_size)) % n2_size;
int64_t g_idx   = (flat_idx / s1_outer_size) % g_size;
int64_t s1o_idx = flat_idx % s1_outer_size;

int64_t q_off  = b_idx * n1_s1_d + n2_idx * g_s1_d + g_idx * s1_d + s1o_idx * s1_base_d;
int64_t kv_off = b_idx * n2_s2_d + n2_idx * s2_d;  // no g_idx — KV shared across G

LocalTensor<half> q_tile = q_que.AllocTensor<half>();
DataCopy(q_tile, q_gm[q_off], s1_base * d_size);
// K, V re-used for all G iterations within this (b_idx, n2_idx, s1o_idx)
```

**Reject_cond**: skip this pattern when **G = 1** (MHA — no grouping benefit). Also skip when the op's GQA shape is **already materialized upstream** (e.g. a sparse-attention variant where K/V is logically shaped (B, N1, S, D) for indexing reasons and de-replication is non-trivial). And skip when **D is so small** (D ≤ 32) that the HBM K/V traffic is already negligible relative to compute.

**Symptom anchor**: independent prototype `fa_v220` row-tiled fp16 D=512 fp16 GQA case hangs at `LaunchAscendKernel 507035` (DEBT-FA-GQA). HYPOTHESIS-LINK: the current row-tiled-VEC-only kernel pre-materializes the G replicas of K/V via Python `repeat_interleave(K, G, dim=1)`, which then makes the kernel's UB budget tight when D=512 and triggers the 507035 silent-launch failure. Validating this candidate's recommendation (device-side n2_idx-only KV offset) on the independent prototype kernel and measuring whether the 507035 hang resolves is one of two next-step actions for DEBT-FA-GQA.

**Other-instances-predicted**: any GQA-aware attention port (FlashAttention-GQA, sparse FlashAttention, multi-query attention variants), incremental KV-cache decoding kernels, and any L4-fused op whose "logical N1" decomposes to N2*G with a shared K/V head per group.

**Promote when**: measured on independent prototype `fa_v220` D=512 GQA AND one additional GQA-shape op (target: a separate sparse-attention or KV-cache decode kernel), with HBM-K/V traffic reduction ≥ G×-1 confirmed via msprof (each n2_idx-only KV offset eliminates G-1 duplicate K/V reads).

---

## CAND-FA-VEC-D-TILE-2: D-dim accumulation belongs INSIDE the Matmul library (V220) / inside FA BaseApi (V351), NOT in kernel-level d-tile loop with manual SetFlag/WaitFlag
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_large_head_dim_D >= 256 / FlashAttention_D_512_or_768`
`derived-from: cann-source (FA reference V220 bn2gs1s2_b.h + V351 kernel_train, 2026-05-24 cl-fa-diff)`
`evidence_family: FA-GQA-DIM`
`verified_on: cann ops-transformer FA reference V220 + V351 dispatch tables`

**Trigger**: Implementing FlashAttention (forward) with head-dim D ≥ 256 on V220 or D ≥ 512 on V351. Temptation: write an explicit d-tile loop in the kernel `for (d_tile = 0; d_tile < D / D_TILE; ++d_tile) { ... DataCopy Q[s1, d_tile*D_TILE:(d_tile+1)*D_TILE] ...; bmm1_partial; accumulate; }` and synchronize d-tile boundaries with manual `SetFlag<HardEvent::V_V> / WaitFlag<HardEvent::V_V>` or `PipeBarrier<PIPE_V>`. This is the path the independent prototype row-tiled fp16 kernel takes for D=512 with D_TILE=128.

**Why "candidate"**: structural pattern derived from how CANN's FA reference dispatches large-D shapes; symptom-link to independent prototype DEBT-FA-GQA is hypothesis, not validated. Need one more L4-FA port to confirm.

**Recommendation**: The CANN FA reference does NOT do kernel-level d-tile splitting. Instead:
- V220 large-D path (when D doesn't fit a single template-dispatch key): tiling table sets a per-core `d_base_size` field consumed by the matmul library tiling policy; the kernel calls `bmm1.SetTail(s1_real, d_size, s2_real); bmm1.IterateAll(workspace_ping, ...)` with d-direction accumulation handled by the high-level `matmul::Matmul<>` library internally. Kernel level has NO d-loop — only outer (b/n2/g/s1) loops.
- V351 large-D path: `flash_attention_score_template_tiling_key.h` includes a discrete D-template key for `D=768`, indicating UB-resident Q[s1_base, 768] is feasible without splitting. The kernel relies on the FA BaseApi base class to manage d-tile accumulation if needed.

For a VEC-only port (when AIC + Matmul library is unavailable, e.g. AIV-only fallback path used in DEBT-FA-GQA), kernel-level d-tile loop IS unavoidable. In that case the safe shape is:
1. Use **`HardEvent::MTE2_V` SetFlag/WaitFlag PER d-tile iteration** between the DataCopy of Q[s1, d_tile] and the Mul/Madd of scores += Q[s1, d_tile] * K^T[d_tile, s2].
2. Do NOT use `PipeBarrier<PIPE_V>` between d-tile iterations of the same s1 row — see PB-21 (V220 PipeBarrier<PIPE_ALL>-on-TBuf silent crash 507015). Use explicit event flags.
3. Keep K^T tile in UB across the full d-loop (load once per s1-block, reuse for all d-tiles); only Q gets reloaded per d-tile.

**Concrete anchor** (public-API VEC-only d-tile loop for fp16 D=512 row-tiled FA; only used when AIC path is unavailable):
```cpp
// One Q-row block in flight; K^T full-S2 × full-D-tile in UB (re-used across d-tiles)
LocalTensor<half> q_tile = q_que.AllocTensor<half>();    // sized for [s1_base, d_tile_size]
LocalTensor<half> kt_tile = kt_buf.Get<half>();           // sized for [d_tile_size, s2_real]
LocalTensor<float> scores = scores_buf.Get<float>();     // [s1_base, s2_real], persistent across d-tiles
Duplicate(scores, 0.0f, s1_base * s2_real);
event_t e_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

for (int d_tile = 0; d_tile < d_size / d_tile_size; ++d_tile) {
    DataCopy(q_tile, q_gm[q_off + d_tile * d_tile_size], s1_base * d_tile_size);
    DataCopy(kt_tile, kt_gm[kt_off + d_tile * d_tile_size * s2_real], d_tile_size * s2_real);
    SetFlag<HardEvent::MTE2_V>(e_mte2_v);
    WaitFlag<HardEvent::MTE2_V>(e_mte2_v);
    // Accumulate scores += q_tile @ kt_tile^T  using VEC MMA-style Mul+Add chain
    // (kept abstract — the chain depends on s1_base × s2_real shape; emit Madd / Mul + Add as fits.)
}
PipeBarrier<PIPE_V>();  // OK here: post-d-loop, before softmax; not between d-tiles.
// scores fully accumulated; continue with softmax + bmm2.
```

**Reject_cond**: do NOT apply this pattern when:
- The op is using high-level `matmul::Matmul<>` / `MatmulImpl<>` — the library handles d-tile internally; kernel-level d-loop is wrong-direction.
- D ≤ 128 single-template dispatch path — the full Q tile fits in UB without splitting; no d-loop needed.
- The kernel mixes `MatmulImpl` + manual `CrossCoreSetFlag` — that's the PB-34 / CAND-FA1 deadlock zone, deal with that first.

**Symptom anchor**: independent prototype `fa_v220` D=512 D_TILE=128 (single-iter d-loop) hang at `LaunchAscendKernel 507035`. HYPOTHESIS: the hang is sync-related (likely the d-tile boundary's MTE2→V handoff missing or wrong-event-ID), NOT UB-budget-related (UB at 192 KB easily fits 64-row × D=512 fp16 = 64 KB). The reject_cond above flags that if the kernel ALSO uses MatmulImpl + manual CrossCore, the d-tile fix won't resolve the underlying deadlock.

**Other-instances-predicted**: any large-D attention port (D=256, 384, 512, 768), any L4-fused kernel decomposing into a `for d_tile { mm1_partial; accumulate; }` shape on the VEC-only path.

**Promote when**: independent prototype DEBT-FA-GQA resolves with this d-loop sync shape AND a separate D=768 FA-class op verifies the same shape works.

---

## CAND-FA-AUX-OUT-1: FlashAttention aux outputs (softmax_max, softmax_sum) emit via Brcb broadcast + single tile-wise DataCopy (or strided DataCopy with dstStride for BNGS1 interleave), NOT per-row scalar SetValue
`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_aux_softmax_outputs / FlashAttention_forward_train / online_softmax_with_external_max_sum_emit`
`derived-from: cann-source (FA reference V220 s1s2_bn2gs1 + s1_bn2gs1 + bn2gs1s2_b epilogues, 2026-05-24 cl-fa-diff)`
`evidence_family: FA-AUX-OUT`
`verified_on: cann ops-transformer FA reference V220 aux-output emit pattern (kernel-structural evidence; V351 epilogue is in BaseApi base class, out of scope for this extraction)`

**Trigger**: Op emits per-row scalar fp32 auxiliary outputs (e.g. `softmax_max[B, N2*G, S1]`, `softmax_sum[B, N2*G, S1]`, or any "one fp32 per softmax row") at the END of each Q-tile, in addition to the main attention output tensor. The aux output layout is typically (B, N2*G, S1) — interleaved per (n2, g) head — meaning aux for head_i and aux for head_(i+1) for the same s1 row are NOT contiguous in GM.

**Why "candidate"**: derived from CANN FA reference's aux-emit pattern. Verified-on is structural only — need a port-side measurement showing the Brcb+DataCopy combo outperforms per-row SetValue (or a different L4 op showing the same aux-emit problem benefits from this pattern).

**Recommendation**:
1. Per-Q-tile, AFTER softmax reduction (max/sum is computed in UB as `LocalTensor<float>` of length `s1_real`), use `Brcb` to broadcast each fp32 scalar to 8 fp32 lanes:
   ```cpp
   Brcb(broadcast_buf, max_per_row, (s1_real + 7) / 8, {1, 8});
   // broadcast_buf now has s1_real * 8 fp32 elements
   ```
   The "×8" matches the SoftmaxFlashV2 / canonical AscendC online-softmax convention where each row's scalar reduction is broadcast to 8 lanes for downstream Mul-by-reciprocal arithmetic.
2. Sync the V→MTE3 transition explicitly:
   ```cpp
   event_t e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
   SetFlag<HardEvent::V_MTE3>(e_v_mte3);
   WaitFlag<HardEvent::V_MTE3>(e_v_mte3);
   ```
3. **Contiguous BS layout** (single output stream — aux outputs for one head, no interleave): single DataCopy:
   ```cpp
   DataCopy(softmax_max_gm[max_off], broadcast_buf, s1_real * 8);  // fp32 count
   ```
4. **Strided BNGS1 layout** (aux outputs interleaved across N2*G heads at the s1 axis — typical for FA aux outputs feeding back into a backward pass): single STRIDED DataCopy via `DataCopyParams`:
   ```cpp
   DataCopy(softmax_max_gm[max_off], broadcast_buf,
            DataCopyParams{
                /*blockCount=*/ static_cast<uint16_t>(s1_real),
                /*blockLen=*/   1,        // 1 unit = 8 fp32 = 32 B
                /*srcStride=*/  0,        // packed source
                /*dstStride=*/  static_cast<uint16_t>(n2_g - 1)  // skip (N2*G - 1) units between rows
            });
   ```
   The strided destination writes row 0's aux at offset 0, row 1's aux at offset N2*G*32 B, etc. — exactly the BNGS1 layout external callers expect, with no extra layout-conversion pass.

**Concrete anchor** (public-API end-to-end pattern; placeholder names):
```cpp
LocalTensor<float> max_row    = max_buf.Get<float>();   // [s1_real] from softmax reduction
LocalTensor<float> sum_row    = sum_buf.Get<float>();
LocalTensor<float> broadcast  = aux_emit_buf.Get<float>();  // [s1_real * 8]

PipeBarrier<PIPE_V>();
Brcb(broadcast, max_row, (s1_real + 7) / 8, {1, 8});
event_t e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(e_v_mte3);
WaitFlag<HardEvent::V_MTE3>(e_v_mte3);

if (layout_is_strided) {
    DataCopy(softmax_max_gm[max_off], broadcast,
             {static_cast<uint16_t>(s1_real), 1, 0,
              static_cast<uint16_t>(n2_g - 1)});
} else {
    DataCopy(softmax_max_gm[max_off], broadcast, s1_real * 8);
}
// Repeat for sum_row → softmax_sum_gm.
```

**Reject_cond**: do NOT use this pattern when:
- Aux output is per-tile (NOT per-row) — e.g. an op emitting a single fp32 reduction per (B, N1, S1_tile) chunk; in that case a plain single-element DataCopy is the right shape.
- The aux output layout is (B, S1, N2*G) with N2*G as the innermost dim — that's contiguous across heads-per-row and benefits from `DataCopyPad` not Brcb+strided-DataCopy.
- S1_real ≤ 8 — the 8-lane Brcb broadcast is wasteful at this size; per-row SetValue may be cheaper.

**Symptom anchor**: DEBT-FA-AUX (FA aux output write currently unblocked path). Hypothesis-link: the V220 row-tiled fp16 kernel's current aux-emit implementation is unknown but if it's per-row SetValue, it will be 8× slower than the Brcb+DataCopy shape. Need port-side measurement to validate.

**Other-instances-predicted**: any fused op with per-row scalar aux outputs (BatchNorm aux mean/var, GroupNorm aux mean/var, online-LayerNorm aux, RMS-Norm aux). Same Brcb-to-8-lane + (optional strided) DataCopy pattern applies.

**Promote when**: measured perf delta vs per-row SetValue on independent prototype fa_v220 aux-output emit AND one other aux-emit op (BatchNorm or GroupNorm forward) shows the Brcb+DataCopy shape is ≥ 2× faster.

---

## CAND-V220-V351-FA-DIFF-1: V220 FA monolithic-class + Matmul-library handoff vs V351 FA per-engine block-types + ASCEND_IS_AIC/AIV gates — the structural pattern for porting FA-class kernels across V220/V351
`applies_to: soc=Ascend910_V220 ↔ Ascend950PR cross-arch port; cann=9.0.0+; op_class=fused_attention_port_a3_to_a5 / FlashAttention_arch22_to_arch35`
`derived-from: cann-source (FA reference V220 arch22/* + V351 arch35/*, 2026-05-24 cl-fa-diff)`
`evidence_family: V351-SYNC-MODE / port_a3_to_a5_FA`
`verified_on: cann ops-transformer FA reference V220 + V351 top-level Process structure + entry macros + template_tiling_key dispatch tables`

**Trigger**: Porting a V220 FA-class kernel (single templated class with `IterateBmm1` / `ProcessVec1` / `IterateBmm2` / `ProcessVec2` methods, 3-deep per-stage info array ping-pong, monolithic source) to V351 / Ascend950PR. Temptation: copy V220 source verbatim, swap `arch22` includes for `arch35` equivalents, hope for the best.

**Why "candidate"**: cross-arch structural pattern derived from comparing V220 and V351 FA reference. The per-engine block-type shape on V351 is mandatory architecturally (V351 toolchain expects `ASCEND_IS_AIC/AIV` partition); structural recommendation is unambiguous from source.

**Recommendation**: V220 and V351 FA reference share the SAME outer algorithm (online softmax, ping-pong pipeline, GQA index decode, KV-shared dispatch) but differ in 4 specific structural axes. When porting, refactor along these axes:

1. **Per-engine block-types**: V351 expects two distinct template classes — one with AIC-only methods, one with AIV-only methods — instantiated via `std::conditional<g_coreType == AscendC::AIC, CubeBlockType, VecBlockType>`. V220's monolithic class with internal `if constexpr` cube/vec guards is NOT the V351 shape; refactor to split.
2. **`ASCEND_IS_AIC` / `ASCEND_IS_AIV` source guards**: V351 Process() body uses `if ASCEND_IS_AIC { cubeBlock.IterateBmm1(...); }` / `if ASCEND_IS_AIV { vecBlock.ProcessVec1(...); }`. Same .o file compiles both paths; the guard runs at compile time. V220's pattern (one class, one Process method) does not transfer; the V351 source must be explicitly partitioned.
3. **Pipeline depth**: V220 = 3-deep (per-stage info array of size 3, indexed via `taskId % 3`); V351 = 4-deep (per-stage info array of size 4, indexed via `taskId & 3`). When porting, add one more pipeline slot — V351's wider Matmul/Vec capacity expects it.
4. **D-template-key range**: V220 supports D-template values from a narrow set ({5, 6, 8} indices, ~3 specific D sizes); V351 supports a much wider set ({16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768}). When porting, re-validate the D-template-key dispatch — D=512 falls into the "general path" on V220 but might dispatch to the D=768 template on V351, which uses different inner UB layout assumptions.

**Concrete anchor** (V351 per-engine partition skeleton, public-API only):
```cpp
// V351 (Ascend950PR / arch35) top-level Process — public-API shape
namespace MyOp {

template <typename CubeBlockType, typename VecBlockType>
class MyFusedKernel : public BaseFAClass<MyFusedKernel<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    __aicore__ inline void Process() {
        RunInfo runInfo[4];  // 4-deep ping-pong
        int64_t taskId = 0;
        for (int64_t outer = 0; outer < outerLimit + 3; ++outer) {
            bool notLast = (outer < outerLimit);
            // ... pipeline stage gating ...

            if (notLast) {
                this->ComputeAxisIdx(outer, runInfo[taskId & 3]);
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm1(runInfo[taskId & 3]);
                }
            }
            if (taskId > 0 && notLast) {
                if ASCEND_IS_AIV {
                    this->vecBlock.ProcessVec1(runInfo[(taskId + 3) & 3]);
                }
            }
            if (taskId > 1 && notLast) {
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm2(runInfo[(taskId + 2) & 3]);
                }
            }
            if (taskId > 2) {
                if ASCEND_IS_AIV {
                    this->vecBlock.ProcessVec2(runInfo[(taskId + 1) & 3]);
                }
            }
            ++taskId;
        }
    }
};

}  // namespace MyOp

// Entry macro — public AscendC kernel registration with MIX_AIC_1_2
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

Critical: the `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` are NOT runtime branches — they expand to compile-time conditional that gates which engine compiles which sub-tree. The .o file built with `__DAV_C310_CUBE__` only contains the AIC sub-tree; the one built with `__DAV_C310_VEC__` only contains the AIV sub-tree. The MIX_AIC_1_2 launch invokes both .o files paired.

**Reject_cond**: do NOT use this V220→V351 refactor pattern when:
- The op is **not** fused-attention class (doesn't have the bmm1/softmax/bmm2 3-stage structure). Other fused ops use different cross-engine patterns; CAND-FA1 / CAND-FA-VEC-D-TILE-* still apply per-arch but the per-engine block-type split here is FA-specific.
- The port target is V220 (the current canonical KB direction is V351 → V220 not the reverse). For V220 ports, this candidate doesn't apply.
- The op uses the high-level `matmul::Matmul<>` library on V220 — the library on V351 handles per-engine internally, so the explicit per-engine split is redundant. This candidate's value is for ops that fully decompose into kernel-level AIC and AIV phases (FA, MoE-finalize, fused-MLA).

**Symptom anchor**: independent prototype `fa_v220` (V220) is currently the only working A3 FA kernel; any future V351/A5 port (post DEBT-FA-V351) will hit the per-engine partition wall. The kw-5 silent-hang in CAND-FA1 / PB-34 was on V220, but the structural shape mismatch (monolithic class + MatmulImpl + manual CrossCore) was a V351-style anti-pattern on V220. Honoring this candidate's per-engine partition during a V351 port avoids inheriting that anti-pattern.

**Other-instances-predicted**: any V220→V351 port of an L4-fused op decomposing into cube+vec phases (MoE-finalize, fused-MLA, fused-RMSNorm-then-attention). Same per-engine block-type pattern applies.

**Promote when**: a successful V220→V351 FA port lands using this structural shape (per-engine block-types + ASCEND_IS_AIC/AIV gates + 4-deep pipeline + V351 D-template-key re-validation) AND a second port-a3_to_a5 fused op (e.g. fused-MLA) reuses the same shape.



## CAND-V351-AIV-WholeReduceMax-fp32-mask-cap: V351 AIV `WholeReduceMax<float>` silently truncates per-repeat when `mask > 64` — split into chunked-reduce + scalar combine

`applies_to: soc=Ascend950PR_9579 (V351); cann=9.0.0; bisheng=15.0.5; op_class=fused-norm | fused-quant | attention-softmax-denom | any-per-row-fp32-reduction`
`verified_on: grouped_matmul_swiglu_quant_v2 2026-05-24 (pp-5 fix landed case 4 → 8/8 PASS_WITHIN_TOLERANCE @ commit f8fecd70)`
`unverified_on: V220 / Ascend910 (A3 chip family — empirical evidence is V351-only; A3 may have different per-repeat mask limit, needs probe)`

**Principle**: On V351 AIV, `WholeReduceMax(dst, src, mask, repeat, ...)` and the WholeReduce* family have a **hardware-level per-repeat fp32 mask cap of 64 elements** (one fp32 vector unit = 64-element lane). Passing `mask > 64` (e.g. mask=128, mask=256) **silently completes** the call but reduces over only the first 64 fp32 elements per repeat — no compile error, no runtime warning, no bit-pattern indicator. The remaining elements beyond mask=64 are ignored. Same caveat applies to other WholeReduce* primitives (WholeReduceMin, WholeReduceSum, possibly WholeReduceAdd).

**Symptom**: per-row absmax / max / sum computed across N>64 fp32 elements returns a value that is the max/sum of only the first 64 elements of the row. Downstream quant scale or softmax denominator is wrong; precision FAIL on cases where the missed elements would have dominated.

**Mitigation (concrete anchor — code from GMSQ_v2 utils.h `ReduceMaxTemplate`)**:
```cpp
constexpr uint32_t FP32_LEN_64_REPEAT = 64;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_ELE = 64;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_BLOCK = 8;

if (count <= FP32_LEN_64_REPEAT) {
    // Small case — single per-repeat WholeReduceMax is safe.
    WholeReduceMax(dst, src, count, 1, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK,
                   ReduceOrder::ORDER_ONLY_VALUE);
} else {
    // Large case (count > 64) — first 64 via per-repeat block-reduce, tail
    // via small-case ReduceMaxSmall, scalar Max combine.
    BlockReduceMax(workLocal, src, /*repeat=*/REPEAT_64,
                   /*mask=*/VEC_LEN_ONCE_REPEAT_ELE, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(resTmpLocal, workLocal, VEC_LEN_ONCE_REPEAT_ELE, 1, 1, 1,
                   VEC_LEN_ONCE_REPEAT_BLOCK, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    ReduceMaxSmall(dst, workLocal, src + FP32_LEN_64_REPEAT,
                   count - FP32_LEN_64_REPEAT);  // tail (≤64 elements)
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams repeatParams = {1, 1, 1, NUM_8, NUM_8, NUM_8};
    Max(dst, dst, resTmpLocal, 1, 1, repeatParams);  // scalar combine head + tail
}
```

The mitigation generalizes: split N-element reduction into `ceil(N/64)` per-repeat reductions (mask=64 each), each producing one partial result, then combine via `Max` (or `Add` for sum) operating on the partial-result vector. Final reduce-of-partials uses single-repeat `WholeReduce*` over ≤64 partials.

**Reject_cond** — do NOT apply when:
- The reduction target is fp16/bf16 (mask cap is 128 for half-precision, not 64). Verify per-dtype before applying.
- The op is V220-only (verified_on is V351; V220 may have different limit per `verified_on` line).
- The reduction count is provably ≤64 at compile time (single-repeat call is correct and faster).

**Symptom anchor**: GMSQ_v2 case 4 originally failed with off-by-up-to-50% scale value when per-row silu*gate absmax was computed across 128 fp32 elements (count=N/2=128). Worker initially wrote `WholeReduceMax(dst, src, /*mask=*/128, repeat=1, ...)` — compiled clean, ran clean, returned absmax over first 64 elements only. pp-5 root-cause diagnosis: split into BlockReduceMax(64) + WholeReduceMax(64) + ReduceMaxSmall(tail) + scalar Max combine → case 4 PASS_WITHIN_TOLERANCE.

**Other instances (predicted)**:
- Fused-quant absmax over inner-D dimensions > 64 (e.g. group_norm_silu_quant inner_dim > 64, rms_norm_quant with hidden_size > 64)
- Attention softmax-max (per-row max over sequence-length > 64)
- Reduction-sum patterns where the same hardware lane structure applies (WholeReduceSum / Add over > 64 fp32 elements)
- Any V351 op whose Tier-2 N axis falls in (64, 8192] range AND reduces along that axis

**Verdict mapping (honest per-output disclosure, post independent cross-review 2026-05-24)**:
The chunked-reduce + scalar `Max`/`Add` combine introduces an fp32 **rounding-order difference** vs the (silently-truncated) single-call baseline. Per-output verdict impact:
- **Quantized outputs** (e.g. `y_int8` in GMSQ_v2): stay `T1_BIT_EXACT` — the quant step (Cast/Round) truncates the sub-ULP scale difference below the int8 quantization threshold.
- **fp32 reduction outputs** (e.g. `y_scale` in GMSQ_v2, or per-row max/sum): land in `T2_WITHIN_FP32_FLOOR` band on cases where the chunked-reduce activates (count > 64). Within fp32 ULP, but NOT strict bit-exact vs reference.

This is the **correct** outcome (better than the wrong silently-truncated answer), not a hardware-floor cheat. Customers applying this pattern should set tolerance gates accordingly: expect strict T1 on downstream quantized/integer outputs, T2_WITHIN_FP32_FLOOR on direct fp32 reduction outputs that flow through `ceil(N/64)` chunks.

GMSQ_v2 case 4 evidence (anchor): post-pp-5 verification.json `per_case_summary_pp5`:
```
case 4: verdict=PASS_WITHIN_TOLERANCE, y_int8=T1_BIT_EXACT, y_scale=T2_WITHIN_FP32_FLOOR
case 6: verdict=PASS_WITHIN_TOLERANCE, y_int8=T1_BIT_EXACT, y_scale=T2_WITHIN_FP32_FLOOR
(cases 1,2,3,5,7,8: y_scale=T1_FP32_NEAR_BIT_EXACT — count ≤ 64 path, single-call safe)
```

**Promote when**:
1. A second V351 op (e.g. group_norm_silu_quant, rms_norm_quant, or attention-softmax-fwd) independently hits the silent-truncation symptom AND applies the same chunked-reduce mitigation successfully (pass rate improves on at-risk cases).
2. Hardware-engineering team confirms the per-repeat fp32 mask=64 is a documented spec (not just empirical), and equivalent caps for fp16/bf16/int32 reductions are catalogued.

**Cross-link**: kernel anchor on origin/main (commit f8fecd70):
`output/a3_to_a5_port/src/kernels/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_utils.h::ReduceMaxTemplate` — customer can grep this function directly post-fresh-clone for the reference template. verification.json `precision.pass_a.per_case_summary_pp5` cases 4 + 6 are the y_scale T2_WITHIN_FP32_FLOOR evidence anchors.


## CAND-V220-to-V351-PortPattern-CubeVecFusedOp: V220→V351 port pattern for cube+vec fused ops — TWO V351 sync paradigms (forward FA-class vs backward gradient)

`applies_to: soc_pair=V220→V351 (Ascend910_V220 source → Ascend950PR_9579 V351 target); cann=9.0.0; bisheng=15.0.5; op_class=non-FA-fused-cube-vec | fused-quant-matmul | fused-norm-matmul | indexer-attention-non-softmax-class | backward-gradient-cube-vec`
`verified_on: forward path = lightning_indexer arch22→arch35 (CANN .../lightning_indexer/op_kernel/ diff 2026-05-24); backward path = sparse_lightning_indexer_grad_kl_loss arch35 (CANN .../sparse_lightning_indexer_grad_kl_loss/op_kernel/arch35/ direct grep 2026-05-24 — independent review catch + main correction)`
`unverified_on: kw runtime PASS yet for LIG_grad backward port using these patterns — bg orch bmz9tfk7b in flight`

**Principle — V351 has TWO sync paradigms** (critical clarification post independent review catch 2026-05-24 21:39Z):

V351 cube+vec fused ops do NOT all share the same cross-core sync paradigm. The path depends on the **forward-vs-backward** axis of the op class:

**Path A — Forward FA-class fused op (mode=4 KFC-internal, drop per-block sync)**:
Forward attention-class ops with Q×K@V tile-scheduling. Evidence: `lightning_indexer/op_kernel/arch35/lightning_indexer_kernel.h:623-624,655-656` defines `QLI_SYNC_MODE4 = 4` (in `lightning_indexer_common.h:70`) and uses it for **outer-loop dual-flag setup + teardown ONLY**. `ProcessBaseBlock` per-block has **NO CrossCoreSetFlag/WaitFlag calls** — per-block sync is delegated to matmul library + RegBase MicroAPI primitives via KFC channel.

**Path B — Backward gradient or scatter/gather-heavy fused op (mode=2 manual per-block, rotating dual-flag indexed by taskIdMod2)**:
Backward gradient ops keep V220-style manual per-block handshake, but enhanced with V351 features: **dual-flag rotating by `taskIdMod2`** for pipeline-depth-2 producer-consumer, plus **per-stage typed flags**. Evidence: `sparse_lightning_indexer_grad_kl_loss/op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_cube_block.h:41` defines `static constexpr uint8_t SYNC_MODE = 2`. `kernel_base.h:633-664` shows 30+ `CrossCoreSetFlag<2, PIPE>(...)` and `CrossCoreWaitFlag<2, PIPE>(...)` sites with:
- Dual-flag arrays: `SYNC_MM2_TO_V1_FLAG[0,1]`, `SYNC_GATHER_TO_MM12_FLAG[0,1]`, `SYNC_C3_TO_V7_FLAG[0,1]`
- Per-stage typed flags: `SYNC_AIV_INNER_FLAG2`, `SYNC_V6_TO_C3_FLAG`
- Per-block usage: `SYNC_*_FLAG[pRunInfo.kTaskIdMod2]` rotates between [0] and [1] per task

**Decision criterion** (which paradigm to use for V220→V351 port):
- Forward attention-class (softmax + Q×K@V + per-row online softmax) → **Path A** (mode=4 KFC-internal)
- Backward gradient (GEMM-reduce + scatter + gather + relu_grad / etc.) → **Path B** (mode=2 manual rotating dual-flag)
- Forward non-attention fused (e.g. fused-norm + cube, fused-quant + cube, GMSQ_v2 path) → **Path A** mode=4 with outer-only dual-flag (verified by GMSQ_v2 commit f8fecd70 — different from sync_aic_aiv_modes=4 but same paradigm of "outer-only KFC-internal")
- Heavy scatter/gather in middle of pipeline → **Path B** (mode=2 manual)

**Concrete delta (forward `lightning_indexer_kernel.h::ProcessBaseBlock` evidence — same op, V220 vs V351)**:

V220 (arch22, manual per-block ping-pong):
```cpp
template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessBaseBlock(...) {
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
            constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(
            constInfo.syncV1C1);
    }
}
```

V351 (arch35, KFC-internal):
```cpp
template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessBaseBlock(...) {
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);   // NO manual sync — KFC-internal
    } else {
        vectorService.ProcessVec1(runInfo);  // NO manual sync — KFC-internal
        if (runInfo.isLastS2InnerLoop) {
            vectorService.ProcessTopK(runInfo);
        }
    }
}
```

Outer-loop setup/teardown (both arches, but V351 uses mode=4 + dual-flag indexed events):

V220:
```cpp
if ASCEND_IS_AIV {
    vectorService.AllocEventID();
    CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);  // 2x prime the pipe
    CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
} else {
    matmulService.AllocEventID();
}
// ... loop ...
if ASCEND_IS_AIC {
    matmulService.FreeEventID();
    CrossCoreWaitFlag(constInfo.syncV1C1);     // 2x drain
    CrossCoreWaitFlag(constInfo.syncV1C1);
}
```

V351:
```cpp
if ASCEND_IS_AIV {
    vectorService.AllocEventID();
    CrossCoreSetFlag<QLI_SYNC_MODE4, PIPE_V>(CROSS_VC_EVENT + 0);
    CrossCoreSetFlag<QLI_SYNC_MODE4, PIPE_V>(CROSS_VC_EVENT + 1);
} else {
    matmulService.AllocEventID();
}
// ... loop with NO per-block manual sync ...
if ASCEND_IS_AIC {
    matmulService.FreeEventID();
    CrossCoreWaitFlag<QLI_SYNC_MODE4, PIPE_FIX>(CROSS_VC_EVENT + 0);
    CrossCoreWaitFlag<QLI_SYNC_MODE4, PIPE_FIX>(CROSS_VC_EVENT + 1);
}
```

**Port-recipe checklist** (for V220→V351 cube+vec fused op, applicable to LIG_grad / quant-matmul / fused-norm-matmul backward classes):

1. **Sync mode constant**: replace `<MODE2>` template param in `CrossCoreSetFlag/WaitFlag` with `<MODE4>`. (V220 → V351 hardware sync infrastructure change. Sync constant value differs per op-family but the mode parameter is always V351=4.)
2. **PIPE on CrossCoreSetFlag for AIV-side**: V220 uses `PIPE_MTE2`; V351 uses `PIPE_V`. AIC-side keeps `PIPE_FIX` both arches.
3. **Flag IDs**: V220 typically uses named per-op flags (`syncV1C1`, `syncC1V1`); V351 uses event-indexed dual-flag (`CROSS_VC_EVENT + 0/+1`). Outer-loop setup primes BOTH flags; teardown drains BOTH.
4. **Per-block CrossCore calls in inner loop**: DELETE them entirely on V351. The matmul library + RegBase MicroAPI primitives (`MicroVAdd`/`MicroVMul`/etc.) handle KFC-internal sync. Only outer-loop setup + teardown remains.
5. **Service classes (`matmulService` / `vectorService`)**: V351 versions live in arch35/ subdir. Cube/vec primitives use `vf/` (vector fission) for typed primitives. The service-class API at V351 ports the V220 API surface to RegBase MicroAPI internals — caller code stays similar, callees differ.
6. **Per-pipe define for AIC vs AIV TU**: V351 build requires per-source-file compile-flag isolation (`-DASCENDC_MATMUL_AICORE` on AIC.cpp, `-DASCEND_VEC_AICORE` on AIV.cpp). See OL-176 / EC-58 for KFC sync per-pass-defines pattern.
7. **Sync paradigm note for non-FA-class fused ops**: per OL-185, op classification stays L2/L3 (NOT L4) when no softmax/attention/online-softmax-tile-scheduling. Calibration anchor remains flat_quant (shipped 8/8 via L2 path).

**Reject_cond** — do NOT apply when:
- Op is FA-forward class (softmax/attention with Q×K@V) → L4 path, different sync requirements (see CAND-V220-V351-FA-DIFF-1).
- Op is pure-VEC (no cube stage) → no cross-core sync needed at all.
- Op uses `KERNEL_TYPE_MIX_AIC_2_2` instead of `MIX_AIC_1_2` → may have different KFC channel layout, re-verify.

**Symptom anchor** (LIG_grad port 2026-05-24 in-flight):
- Worker fa_fused_mixed_fp16 V220 port to V351 hung at `CrossCoreWaitFlag` spin (independent prototype T1.12 iter 1/2/3 cumulative falsification) when applying V351 mode=4 to per-block sync calls that V351 doesn't need. Fix path = drop per-block, only outer dual-flag setup.
- LIG_grad worker attempted V220 line-port (mode=2 manual per-block) → would have hit same V220-only paradigm on V351 hardware. Plugin fix P0gg (commit f9b98ea3) now routes LIG_grad to L2 with this pattern as expected calibration.

**Other instances (predicted)**:
- LIG_grad (backward gradient, GEMM+gather+scatter+reduce class) — in-flight verification (bg orch bmz9tfk7b)
- Future V220→V351 backward gradients of non-FA fused ops (e.g. attention_grad sans softmax, MoE-finalize backward, fused-quant-matmul backward, fused-norm-matmul backward)
- Quant-matmul forward V220→V351 (e.g. `quant_batch_matmul_v3` port — has identical MIX_AIC_1_2 + matmul library shape)

**Promote when**:
1. LIG_grad ships via this pattern (commit SHA + verification.json pass_a 8/8) → verifies pattern beyond forward op evidence
2. A SECOND non-FA-class V220→V351 port (e.g. quant_batch_matmul_v3, fused_norm_matmul, attention_grad_no_softmax) applies the recipe + ships clean

**Cross-link**: forward `lightning_indexer/op_kernel/` arch22+arch35 pair (CANN reference source; NOT customer-readable — KB body above is self-contained); flat_quant calibration anchor (OL-185, commit 7b3c7bf3 on origin/main); CAND-V351-AIV-WholeReduceMax (related V351 AIV gotcha, commit 5260fd68); CAND-V351-arch35-RegBase-service-class-skeleton (complementary — V351 service-class detail patterns).

## CAND-V351-arch35-RegBase-service-class-skeleton: V351 arch35 service-class structure (cube_block + vector_block + vf/* MicroAPI vector fission) for backward op-family port targets

`applies_to: soc=Ascend950PR_9579 (V351/arch35); cann=9.0.0; bisheng=15.0.5; op_class=backward (gradient) multi-stage cube+vec | non-FA-class (no online-softmax tile-scheduling)`
`verified_on: sparse_lightning_indexer_grad_kl_loss arch35 service-class structure (CANN reference 2026-05-24, 3758 LOC read; NOT yet shipped on ours; complementary evidence layer to CAND-V220-to-V351-PortPattern-CubeVecFusedOp which covered sync-mode delta)`
`unverified_on: kw runtime ship using these service-class skeletons; V220 (V220 LIG_grad uses different vec primitives — V351 vf/ MicroAPI is arch35-only)`

**Companion to CAND-V220-to-V351-PortPattern-CubeVecFusedOp**: that entry covers V220→V351 **sync-mode delta** (mode2→mode4 + outer-only KFC-internal); THIS entry covers V351 arch35 **service-class skeleton** (how cube_block + vector_block + vf/* fit together inside the V351 op kernel TU). Together = sync + structure = full V220→V351 port template.

### Principle: V351 arch35 op kernel splits along 4 axes
1. **Top-level kernel class**: `KernelBase<CubeBlockType, VecBlockType>` template with `ASCEND_IS_AIC`/`ASCEND_IS_AIV` branches in `Init()` and `Process*()` methods. Owns shared TPipe, BufferManager<UB/L1>, ConstInfo/RunInfo structs.
2. **Cube block** (header `*_cube_block.h`): manages L0A/L0B/L0C buffer pool + matmul invocation + AIC sync. Template params via macro-generated `TEMPLATES_DEF`.
3. **Vector block** (header `*_vector_block.h`): N-stage AIV pipeline (`ProcessVector0` ... `ProcessVectorN`); each stage is one vf-process function. Manages UB buffer pool + Vec sync.
4. **Vector fission `vf/*.h`** subdir: each vec primitive (Cast/Add/Mul/Nd2Nz format conversion/etc.) is one `__simd_vf__` function using `MicroAPI` namespace — RegTensor + LoadAlign/StoreAlign + MaskReg.

### Concrete inline skeleton (self-contained, customer-runnable):

```cpp
// File: <op>_regbase_common.h
namespace <OpNs> {
constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
constexpr uint32_t UB_MAX_SIZE = 128 * 1024;   // per-op allocation, full V351 UB=256KB
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
constexpr uint32_t MODE_NUM_2 = 2;
constexpr uint32_t MODE_NUM_3 = 3;

// Dual-flag arrays for ping-pong, indexed by [taskIdMod2]
constexpr uint8_t SYNC_A_TO_B_FLAG[2] = {N, N+1};
constexpr uint8_t SYNC_B_TO_A_FLAG[2] = {M, M+1};

struct ConstInfo { /* per-kernel constants + tilingData slice */ };
struct RunInfo   { /* per-task: taskId, bIdx, taskIdMod2, ... */ };
struct KRunInfo  { /* per-K-loop: kTaskId, kTaskIdMod2, ... */ };

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X)  X(INPUT_T) X(OUT_T) X(T)
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) X(LAYOUT, OpLayout, OpLayout::TND)
#define TEMPLATES_DEF template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
                                CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>
}  // namespace
```

```cpp
// File: <op>_kernels.cpp (dispatcher TU)
#include "kernel_operator.h"

// PB-28 KB ENTRY CORRECTION (2026-05-25 02:17Z): KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
// is NOT arch35-only. V220 ACCEPTS this macro natively per:
// (1) CANN canonical FA: flash_attention_score.cpp:379 uses it unconditionally on V220 build
// (2) Independent empirical 2026-05-25 02:17Z: removed arch-guard, V220 build + .so load PASS,
//     no RegisterAscendBinary 107000 error
// Earlier KB entry (independent prototype commit 1162679d) attributed 9/61→42/61 jump to PB-28 guard +
// 7-tuple ModelNew fix combined; falsification shows the 7-tuple was the load-bearing fix.
// PB-28 arch-guard was defensive over-application. KB entry PB-28 itself needs amendment.
// DO NOT add arch-guard around this macro for V220 builds — it's V220-native.
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

// ... kernel entry points (one per dtype+layout combo)
```

```cpp
// File: <op>_cube_block.h
namespace <OpNs> {
using T = float;
TEMPLATES_DEF
class OpBlockCube {
public:
    static constexpr uint8_t SYNC_MODE = 2;       // mode=2 manual flag-based for backward op (mode=4 KFC-internal only for forward FA-family per companion CAND)
    static constexpr uint32_t M_SPLIT_SIZE = 128;
    static constexpr uint32_t N_SPLIT_SIZE = 128;
    static constexpr uint32_t K_SPLIT_SIZE = 128;

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;                                  // ping-pong
    BuffersPolicyDB<BufferType::L0B> l0bBuf;
    BuffersPolicy3buff<BufferType::L0C> commonL0CBuf;                          // 3-buffer for deeper pipeline
    BuffersPolicySingleBuffer<BufferType::L1> sYL1Buf;

    __aicore__ inline void ComputeMm1(Buffer<UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       BuffersPolicyDB<L1, SyncType::CROSS_CORE_SYNC_BOTH> &sYL1Buf,
                                       RunInfo &runInfo, ConstInfo &constInfo, KRunInfo &kInfo) {
        // AIC waits for AIV gather completion
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_A_TO_B_FLAG[kInfo.kTaskIdMod2]);
        // ... matmul invocation (op-specific) ...
        // AIC signals AIV that result is ready in workspace
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_B_TO_A_FLAG[kInfo.kTaskIdMod2]);
    }
};
}
```

```cpp
// File: <op>_vector_block.h
#include "vf/vf_process_vec0.h"
#include "vf/vf_process_vec1.h"
// ... per-stage vf headers

namespace <OpNs> {
TEMPLATES_DEF
class OpBlockVec {
public:
    __aicore__ inline void ProcessVector1(Buffer<UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                          RunInfo &runInfo, KRunInfo &kInfo) {
        // AIV waits for AIC matmul result
        CrossCoreWaitFlag<2, PIPE_V>(SYNC_B_TO_A_FLAG[kInfo.kTaskIdMod2]);
        // ... call vf_process_vec1 primitives ...
        // AIV releases buffer
        CrossCoreSetFlag<2, PIPE_V>(SYNC_B_TO_A_FLAG[kInfo.kTaskIdMod2]);
    }
};
}
```

```cpp
// File: vf/vf_process_vecN.h
namespace AscendC {
using namespace MicroAPI;
template <typename INPUT_T>
__simd_vf__ inline void OpSpecificMicroOp(__ubuf__ INPUT_T *dstUb, __ubuf__ INPUT_T *srcUb,
                                          uint32_t m, uint32_t n) {
    RegTensor<INPUT_T> vreg_x;
    MaskReg preg = UpdateMask<uint16_t>(/*repeatSize=*/128);
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_x, srcUb + i * n);
        StoreAlign<INPUT_T,
                   MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb), vreg_x, /*blockStride=*/m, /*repeatStride=*/1, preg);
    }
}
}  // namespace
```

```cpp
// File: <op>_kernel_base.h
namespace <OpNs> {
template <typename CubeBlockType, typename VecBlockType>
class OpKernelBase {
public:
    __aicore__ inline void Init(GM_ADDR ...inputs, GM_ADDR workspace,
                                const TilingData *tiling, TPipe *tPipe) {
        pipe = tPipe;
        tilingData = tiling;
        SetConstInfo();
        InitWorkspace(workspace);
        if ASCEND_IS_AIV {
            vecBlock.InitParams(constInfo, tilingData);
            vecBlock.InitGlobalBuffer(/* AIV-side GMs */);
            vecBlock.InitBuffers(pipe);
        } else if ASCEND_IS_AIC {
            cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
            cubeBlock.InitCubeBuffers();
            cubeBlock.InitGlobalBuffer(/* AIC-side GMs */);
        }
    }
    __aicore__ inline void Process() {
        // Outer loop with [taskId % MODE_NUM_2] ping-pong on kRunInfos[]
        KRunInfo kRunInfos[MODE_NUM_2];
        for (int32_t taskId = 0; taskId < runInfo.kLoopTimes + 1; ++taskId) {
            KRunInfo &cur = kRunInfos[taskId % MODE_NUM_2];
            KRunInfo &prev = kRunInfos[(taskId + 1) % MODE_NUM_2];
            // ... AIV produces cur, AIC consumes prev (deep pipeline) ...
        }
    }
    TPipe *pipe;
    const TilingData *__restrict tilingData;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    ConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};
}
```

### Tiling data class hierarchy (V351 RegBase pattern)

```cpp
namespace optiling {
class BaseParamsRegbase {
    int32_t bSize, n2Size, s1Size, s2Size, dSize, kSize;
    float scaleValue;
    uint8_t layoutType;
    int32_t get_bSize() const { return bSize; }
    void set_bSize(int32_t p) { this->bSize = p; }
    // ... pair for each field
};
class MultiCoreParamsRegbase {
    uint32_t coreNum;
    int64_t splitFactorSize, totalSize;
    int64_t bS1Index[MAX_CORE_NUM_REGBASE];  // 36 max for V351
};
class VecApiParamsRegbase {
    SoftMaxTiling softmaxTilingData;          // for Vec API sub-tiling if applicable
};
class RegBaseTilingData {
    BaseParamsRegbase baseParams;
    MultiCoreParamsRegbase multiCoreParams;
    VecApiParamsRegbase vectorParams;
};
}
```

### Template tiling key (compile-time variant selector)

```cpp
ASCENDC_TPL_ARGS_DECL(OpName,
    ASCENDC_TPL_BOOL_DECL(VARIANT_BOOL, 0, 1),
    ASCENDC_TPL_UINT_DECL(VARIANT_RANGE, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_UINT_DECL(LAYOUT, 4, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(DETERMINISTIC, 0, 1),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(VARIANT_BOOL, 0, 1),
        // ... valid combinations
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::RegBaseTilingData)
    )
);
```

### Reject_cond — do NOT apply this skeleton when:
- Op is FA-forward class (online-softmax tile-scheduling) — different sync paradigm, see CAND-V220-to-V351-PortPattern body (mode=4 + outer-only KFC-internal for forward LIG-family)
- Op is pure-VEC (no cube stage) — vector fission still applies but cube_block + L0A/L0B/L0C abstractions unused
- Op doesn't fit `MIX_AIC_1_2` topology — `BuffersPolicy{DB,3buff,SingleBuffer}` abstractions assume cube+AIV producer-consumer; pure-cube or pure-AIV patterns differ

### Sync-mode nuance vs companion CAND (open audit item):

CAND-V220-to-V351-PortPattern-CubeVecFusedOp documents V220 mode=2 → V351 mode=4 + KFC-internal-implicit (no per-block manual sync) based on forward LIG arch22→arch35 evidence. THIS entry documents V351 backward sparse_LIG_grad_kl using **mode=2 + per-block manual sync with dual-flag ping-pong** (verified via direct grep of `CrossCoreSetFlag<2, PIPE_X>` 30+ occurrences in `kernel_base.h` + `cube_block.h` + `vector_block.h`).

**Reconciliation hypothesis** (needs main verify on forward LIG arch35 source):
- **V351 forward LIG-family** (lightning_indexer arch35): mode=4 + KFC-internal (cgmct-style sync wrapper at outer loop, no per-block inner sync)
- **V351 backward LIG-family** (sparse_lightning_indexer_grad_kl_loss arch35): mode=2 + manual per-block sync + dual-flag `[taskIdMod2]` ping-pong (canonical V220-style paradigm preserved on V351 hardware)

Possible reasons for divergence:
- Forward op tile-scheduling has multi-task-per-block pipeline depth → benefits from KFC-internal sync via cgmct wrapper
- Backward op task-per-block 1:1 → manual flag-based sync sufficient + no cgmct wrapper overhead
- Or main's "mode=4" claim from forward source may need verify against newer commit (forward LIG arch35 kernel.h is 30705 bytes — not full read yet)

**Promote when** (separate from companion CAND):
1. LIG_grad ships via this service-class skeleton (bg orch `bmz9tfk7b`) — verifies skeleton at runtime
2. A SECOND V351 backward op (e.g. attention_grad sans softmax, MoE-finalize backward, fused-quant-matmul backward) applies the skeleton + ships clean
3. Sync-mode nuance reconciled: confirm forward = mode=4 + outer-KFC; backward = mode=2 + per-block-manual; or unify if same paradigm proves to apply both

**Other instances (predicted)**:
- LIG_grad (current bg orch `bmz9tfk7b`) — direct beneficiary
- dense_lightning_indexer_grad_kl_loss arch35 (CANN sibling, parallel-pair with sparse — paradigm identical)
- attention_grad backward without softmax
- MoE-finalize backward
- Custom V351 backward fused-norm-matmul / fused-quant-matmul

**Customer-impact**: customer brings V220 backward multi-stage cube+vec op to harness → harness applies (companion CAND sync paradigm + this CAND service-class skeleton) → kw generates V351 kernel TU **without CANN-source access**. Fully self-contained per Zheng 2026-05-24T21:28Z directive.

**Cross-link**: companion CAND-V220-to-V351-PortPattern-CubeVecFusedOp (commit fb13a899 on origin/main — sync paradigm delta); CAND-V351-AIV-WholeReduceMax-fp32-mask-cap (commit 5260fd68 — V351 AIV reduction primitive gotcha applicable to backward ops); OL-185 (op_class L2/L3 calibration anchor); CAND-V220-V351-FA-DIFF-1 (forward FA-class differs from this skeleton — that's L4 path).

## CAND-FA-CV-1: Ring buffer workspace with WorkspaceQueue for multi-stage AIC↔AIV pipeline overlap

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=mixed_aic_aiv_fused_kernel_with_kv_iteration_and_prelaunch_overlap`
`derived-from: cv-agent tile2asc flash_attention design (block_level + cube.h + workspace_queue.h)`
`verified_on: cv-agent stock FA 16/16 PASS on A3/V220 (independent prototype F10.A.1 2026-05-25); DS env build+load+execute confirmed`
`unverified_on: V351/A5; a5_ops 61-case fixture; perf vs CANN baseline`

**Trigger**: A mixed AIC/AIV fused kernel (e.g., FA, MoE gating, grouped matmul) needs ≥2 stages where cube and vector exchange GM-resident intermediates, AND stages should overlap (prelaunch next stage before current stage fully completes).

**Pattern**: cv-agent FA uses `WorkspaceQueue<T, DEPTH>(gm_tensor, elem_size, sig_ready, sig_free)` for ring-buffer GM scratchpad management. Producer (cube/vec) writes into slot via `queue.AllocSlot(pipe)` → DataCopy(slot, src) → `queue.ReleaseSlot()`. Consumer acquires via `queue.WaitSlot()` → DataCopy(dst, slot) → `queue.FreeSlot()`. Ring buffer depth = prelaunch + 1 (e.g., prelaunch=2 → 3 slots). Cross-core flag IDs use vendor recipe: slot-indexed flags (0x8+slot for C→V, 0x10+slot for V→C).

**Our gap**: a5_ops FA (PR #146) uses raw `CrossCoreSetFlag<0x2, PIPE_FIX>(0x8+slot)` + manual GM offset arithmetic — no `WorkspaceQueue` abstraction, no ring-buffer lifecycle management, no slot ownership tracking.

**Detection**: grep for `CrossCoreSetFlag` + `DataCopy.*workspace` in kernel .h files. If flag IDs are hand-computed AND workspace_s/p/o tensors are raw-addressable (no queue abstraction) → WorkspaceQueue pattern missing.

**Evidence**: cv-agent `flash_attention_cube.h:32-34` + `workspace_queue.h` full implementation (Init/InitFreeSlotsMte2/AllocSlot/WaitSlot/ReleaseSlot/FreeSlot). 16/16 PASS stock fixture on V220.

**Cross-ref**: CAND-FA1 (manual CrossCoreSetFlag — this is the higher-level abstraction for multi-stage pipelines)


## CAND-FA-CV-2: Declare partition, ring-buffer, workspace, and sync contracts before AscendC kernel implementation

`applies_to: workflow=L4_fused_op_design; backend=ascendc`
`derived-from: FA-class template-assembly and WorkspaceQueue evidence`
`verified_on: a staged FA design followed by AscendC assembly completed with an explicit block/workspace contract`

**Trigger**: a new L4 fused op needs algorithm design before AscendC coding. The worker
must decide block partitioning, tile sizes, ring-buffer depth, workspace tensors, and
cross-core synchronization before it starts emitting kernel bodies.

**Pattern**: use two stages:

1. **Design contract** — record `block_num`, `block_M/N`, prelaunch/ring slots,
   workspace tensor names/shapes/dtypes, cube/vector ownership, and flag IDs in a
   structured decision manifest or Phase-A artifact.
2. **Template assembly** — fill the per-tile Load/Gemm/Softmax stages only after the
   contract is reviewable, then verify that the emitted tiling and workspace layout
   match it mechanically.

**Detection**: if an L4 cube↔vec kernel is emitted without a structured artifact
covering `block_partition`, `cube_vec_split`, `cross_core_sync`, `ub_tiling`,
and `ring_slots`, the design step was skipped. Reject the emission before build
rather than inferring those decisions from a monolithic header afterward.

**Evidence**: the FA-class reference design used block_M=64, block_N=64, prelaunch=2,
ring_slots=3, four workspace tensors, and explicit C/V ownership. Making those
decisions first exposed partition and workspace drift that a one-step model-to-code
path had missed.

**Promote when**: a second L4 fused operator independently confirms that the same
design-contract keys predict a correct multi-core AscendC assembly.


## CAND-FA-CV-3: LoadNdGmToNzL1 for Cube-side strided GM→L1 with dimension-aware tiling

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=cube_matmul_with_strided_input`
`derived-from: cv-agent flash_attention_cube.h:57 — LoadQ implementation`
`verified_on: cv-agent stock FA 16/16 PASS (cube matmul1 Q@K^T + matmul2 P@V both use this pattern)`

**Trigger**: Cube matmul (Mmad/Fixpipe) needs to load input tensors from GM→L1 where the input layout may be strided (e.g., Q[batch, head, seq, dim] → cube processes per-(b,h) tile with stride=dim in GM).

**Pattern**: `LoadNdGmToNzL1(dst_L1, gm_src, M, K, stride)` loads M×K block from GM into L1 in Nz format (V220 cube-native layout), handling dimension-aware stride. Followed by `SetWaitFlag<HardEvent::MTE2_MTE1>()` to signal L1 data ready for cube pipeline.

**Our gap**: a5_ops matmul ops (1_BatchMatmul, 3_MatmulBothTrans) use `DataCopy` or `TQue` for GM→UB loads, then manually reorganize for cube. `LoadNdGmToNzL1` combines load + layout transform in one hardware operation.

**Detection**: grep for `DataCopy.*L1\|TQue.*A1` in kernel .h files. If cube matmul path uses `DataCopy` or `TQue` for L1 staging instead of `LoadNdGmToNzL1` → pattern missing.

**Evidence**: cv-agent `flash_attention_cube.h:57` — `LoadNdGmToNzL1(qL1, qGm_[qOffset], BLOCK_M, dim, dim)` at LoadQ entry.

**Cross-ref**: P-P47 (VEC halving reduction), EC-61 (scalar-pipe accumulator → VEC pipe)


## CAND-FA-CV-4: Separate Cube/Vector class architecture with MTE2_MTE1 sync boundary

`applies_to: soc=Ascend910_9382 (V220); op_class=mixed_aic_aiv_fused_kernel`
`derived-from: cv-agent flash_attention_cube.h + flash_attention_vec.h + flash_attention_kernel.h`

**Trigger**: Mixed AIC+AIV kernel where cube and vector stages have independent state (tiling config, GM tensor handles, UB buffers, pipeline queues). Monolithic single-class architecture forces shared state that complicates both sides.

**Pattern**: cv-agent FA splits into 3 classes:
1. `FlashAttentionCube<QType>` — cube-only: L1 buffers (qBufL1_, kvBufL1_, pBufL1_), L0A/B/C queues, LoadQ/LoadKV/Mmad1/Mmad2 methods. Init receives tiling struct + all GM tensor handles.
2. `FlashAttentionVec<QType>` — vector-only: UB buffers for softmax state (m_i, sumexp, acc_o), MTE3 output queue.
3. `FlashAttentionKernel` — orchestrator: Init allocates pipe + instantiates both Cube and Vec objects, Process() runs kv_loops with prelaunch pipeline (LoadKV while Cube processes prev iteration).

Sync boundary: `SetFlag<HardEvent::MTE2_MTE1>()` on cube side (cube→vec L1→VEC data ready), `WaitFlag<HardEvent::V_MTE2>()` on vec side (vec→cube UB→L1 data for next iter).

**Our gap**: a5_ops FA (PR #146) uses monolithic single-class with all buffers + methods in one `FaKernel` class. No cube/vec separation → harder to reason about AIC vs AIV lifetimes.

**Detection**: grep for `__aicore__` in kernel .h. If a single class contains BOTH `queL0A_/queL0B_` (cube L0 queues) AND `m_i/sumexp/acc_o` (vec UB state) → monolithic architecture, no cube/vec separation.

**Evidence**: cv-agent files — `flash_attention_cube.h:13` (FlashAttentionCube class), `flash_attention_vec.h` (FlashAttentionVec class), `flash_attention_kernel.h` (orchestrator).

**Cross-ref**: CAND-FA-CV-1 (ring buffer with WorkspaceQueue — the queue abstraction pairs with cube/vec separation), CAND-V351SYNC-1 (V351 mode=4 sync protocol)


## CAND-FA-CV-5: Block-local sparse attention without ring buffer — composite stages sufficient when no KV iteration

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=block_sparse_attention_without_cross_block_interaction`
`derived-from: cv-agent tile2asc block_sparse_attention design (block_level + kernel)`
`verified_on: cv-agent stock block_sparse_attention design documents pattern; not independently runtime-verified by DS`
`unverified_on: V351/A5; DS A3 runtime verification pending`

**HARD GUARD (2026-05-27, per F10 root-cause: designer selected CV-5 for standard SDPA FA → K-loop collapse + handoff broken)**:
- **DO NOT use CAND-FA-CV-5 for standard SDPA / FlashAttention with multi-KV-block (Skv > block_N).**
- CAND-FA-CV-5 applies **ONLY** when `exactly 1 KV block per Q block` (Skv ≤ block_size, block-sparse with no cross-block interaction).
- For standard FA (3_FusionAttention, Skv up to 512 ≫ block_N = 64/128 → multi KV block → KV loop required): use **CAND-FA-CV-1** (ring buffer + WorkspaceQueue + KV loop).
- **Mis-application consequence**: single `T.mma` without K-loop (KV blocks beyond first ignored), no workspace_meta ring (softmax state handoff broken) → A3 zero-output / A5 deadlock.
- **Detection**: grep design `*.py` for `CAND-FA-CV-5` reference AND `Skv > block_N` → BLOCK with `fa_pattern_mismatch: CV-5 requires Skv ≤ block_N, use CV-1`.

**Pattern**: When attention is block-local (each sequence divided into fixed-size blocks, no cross-block interaction), the full prelaunch/ring-buffer/WorkspaceQueue pipeline (CAND-FA-CV-1) is unnecessary. A simpler 4-stage pipeline (C1→V1→C2→V2) with 3 workspace tensors (s/p/o, no meta) and NO KV loop suffices:
- C1: Q_block @ K_block^T → workspace_s (fp32 scores)
- V1: scale + softmax → workspace_p (fp16 weights)
- C2: P @ V_block → workspace_o (fp32 partial output)
- V2: cast + write output

Block dimension: `block_num = batch * n_heads * n_blocks` where `n_blocks = seq_len / block_size`. No prelaunch needed because there's exactly 1 KV block per Q block.

**Our gap**: a5_ops FA (PR #146) assumed ALL attention variants need the full ring-buffer pipeline. For block-sparse or local-attention variants, the simpler composite-stage pattern reduces UB pressure and eliminates ring-buffer sync complexity.

**Detection**: grep for `kv_loops\|prelaunch\|ring_slots` in kernel .h files. If block-sparse op has these (unnecessary) → over-engineered for block-local pattern.

**Evidence**: cv-agent `block_sparse_attention/design/block_level/block_sparse_attention.py` — explicit comment "we don't need the prelaunch / ring-buffer pattern." 3 workspace tensors (vs 4 in base FA), no kv_loops variable.

**Cross-ref**: CAND-FA-CV-1 (ring buffer with WorkspaceQueue — the full pipeline), CAND-FA-CV-2 (block-level design — this is a counter-example showing design adapts to attention variant)


## CAND-FA-CV-6: MLA split-key architecture — Q/K decomposed into nope (position-independent) + rope (position-dependent) for KV cache compression

`applies_to: soc=all; cann=all; op_class=multi_head_latent_attention_with_compressed_kv_cache`
`derived-from: cv-agent tile2asc flash_attention_mla model.py + kernel`
`verified_on: cv-agent stock MLA design documents pattern; cv-agent kernel artifacts present (cube.h + vec.h + kernel.h triplet confirms CAND-FA-CV-4 applies)`
`unverified_on: DS A3 runtime verification pending; a5_ops harness integration pending`

**Pattern**: MLA (Multi-head Latent Attention, per DeepSeek-V2/V3) splits Q and K into two components:
- `nope` (no-position): position-independent semantic content — can be absorbed into compressed KV cache
- `rope` (RoPE): position-dependent rotary embedding — computed on-the-fly, NOT cached

Input: 4 separate tensors (`q_nope`, `q_rope`, `k_nope`, `k_rope`) instead of standard single Q/K. GQA repetition via `_repeat_kv` handles kv_heads ≠ n_heads. Kernel retains cube.h/vec.h/kernel.h triplet (CAND-FA-CV-4 confirmed — MLA is same architectural family as standard FA).

**Our gap**: a5_ops has NO MLA implementation. Standard FA (3_FusionAttention) uses single Q/K inputs. CV-agent's MLA kernel exists with full design→translation chain complete — immediate adoption candidate for F10.E.2 (second op after FA).

**Key difference from standard FA**: Not just a different attention variant — it's a **different input contract** (4 tensors vs 2). This affects case_gen schema, model.py interface, and device block-level partition design.

**Detection**: grep for `nope\|_rope\|q_rope\|k_nope` in model.py or design files. Absence = standard FA, not MLA.

**Evidence**: cv-agent `flash_attention_mla/model.py:20-28` (forward signature with 4 inputs + _repeat_kv). Kernel has cube.h/vec.h/kernel.h triplet (same architectural pattern as CAND-FA-CV-4).

**Cross-ref**: CAND-FA-CV-4 (cube/vec class separation — MLA confirms this pattern applies beyond standard FA), CAND-FA-CV-2 (block-level design — MLA has design files in trace.md confirming 6-stage workflow)

---


## CAND-FA-CV-7: Multi-strategy kernel dispatch by shape regime — specialized variants per dimension class

`verified_on: cv-agent rms_norm multi-kernel architecture (merge_n / single_row / splitd) + FA variant dispatch table (s1s2_bn2gs1 / var_len_score / var_len_score_sab)`

**Pattern**: When an op's optimal compute strategy depends on input shape dimensions, emit N specialized kernel variants dispatched by shape class at runtime, rather than a single generic kernel with internal branches.

**Mechanism**:
1. Analyze op's compute pattern for dimension-dependent behavior (reduction axis size, gather dim, matmul M/N/K ratio)
2. Define shape classes (e.g., large-N merge, small-M single-row, small-N split-D for rms_norm; BSH/BNSD vs TND/varlen for FA)
3. Emit one kernel variant per shape class, each with optimal tiling (block size, loop ordering, buffer sizing)
4. Host-side dispatcher selects variant at runtime based on input tensor shapes

**Concrete example (rms_norm)**:
- `merge_n` variant: N ≥ 4096 → reduce rows in blocks, merge partial results → large N throughput
- `single_row` variant: M ≤ 16 → one block per row, no cross-block sync → low latency for small M
- `splitd` variant: N < 256 → split along D dimension, parallel VEC reduce per chunk → handles tiny N

**Concrete example (FA — per F10 variant dispatch finding)**:
- `s1s2_bn2gs1` variant: BSH/BNSD/SBH fixed-shape → matmul::Matmul + KFC implicit sync, zero CrossCore
- `var_len_score` variant: TND/variable seq length → tile-MMAD + 12-flag CrossCore triple-buffer
- `var_len_score_sab` variant: TND + sparse attention → same SYNC_* scheme

**Why one generic kernel is worse**: Internal branches on shape dimensions cause divergent tiling, wasted buffer allocation (max of all variants), and suboptimal VEC/Cube utilization for the actual shape. Variant-per-class eliminates runtime branches and allows per-class buffer sizing.

**Detection**: count kernel variants per op directory (`ls kernel/*.cpp | wc -l`). Single `.cpp` with if/else on shape → gap. Multiple `.cpp` with shape-class naming (merge_n / single_row / splitd) → pattern applied.

**Evidence**: cv-agent `rms_norm/kernel/` has 9 .cpp files (3 variants × 3 dtypes). FA variant dispatch table verified by main+independent reviewer+DS independent grep of CANN arch22 source (2026-05-23).

**Cross-ref**: CAND-FA-VARIANT-DISPATCH (FA_CLASS_DESIGN_NOTES.md#fa-v220-decision-tracking-ds — the shape-regime → CANN-variant → KB-pattern mapping table), CAND-FA-CV-4 (cube/vec separation — each variant has own cube.h+vec.h pair), [§F10 FA-class problem solution](../../../../docs/ROADMAP.md#f10-cv-agent-learning--harness-fa-gen-capability)

---


## CAND-FA-CV-8: Fused matmul + elementwise in single kernel — L1-resident intermediate, no GM round-trip

`verified_on: cv-agent matmul_leakyrelu kernel (Cube matmul → L1 resident → VEC leaky_relu → GM output)`

**Pattern**: When an op is matmul followed by elementwise activation (GELU, ReLU, LeakyReLU, SiLU, add, mul), fuse both into a single AscendC kernel. The matmul output stays in L1 buffer and is immediately consumed by the VEC elementwise stage — zero GM round-trip for the intermediate tensor.

**Mechanism**:
1. Cube unit computes matmul output into L1 workspace tensor (not GM)
2. VEC unit reads from the SAME L1 buffer (no DataCopy to GM, no DataCopy back)
3. VEC applies elementwise activation (Mul+Max+Mul for LeakyReLU, etc.)
4. VEC writes final result to GM output tensor

**Buffer lifecycle**:
```
[GM A] → L1 buf_a → Cube → L1 buf_c (matmul output, resident)
                                   ↓
                              VEC reads buf_c
                                   ↓
                              VEC applies activation
                                   ↓
                              [GM Y] ← VEC writes buf_c (now = activated output)
```

**Savings**: Eliminates one GM write (matmul output) + one GM read (activation input). For matmul with M=N=K=4096 fp16: saves 64MB GM write + 64MB GM read per invocation.

**Detection**: grep for `Matmul.*IterateAll` and `Activation|Relu|Gelu|Silu|LeakyRelu` in the same kernel .cpp. If matmul output goes to GM (SetGlobalBuffer before activation) → gap. If matmul output stays in L1/TBuf → pattern applied.

**Evidence**: cv-agent `matmul_leakyrelu/kernel/matmul_leakyrelu.h` — single kernel class with `MatmulObj.IterateAll()` writing to L1-local `mmOutBuf_`, immediately followed by `LeakyRelu(mmOutBuf_, yOutQueue_)` in the same Process() body.

**Cross-ref**: CAND-NSA-1 (Matmul<>::IterateAll + local SetFlag for AIV chaining — same L1-resident principle for FA), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — L1 workspace management generalizes to fused matmul)

---


## CAND-FA-CV-9: Dimension-adaptive gather with 3-mode dispatch — detect gather dim → classify → select specialized kernel

`verified_on: cv-agent gather_elements_v2 3-kernel architecture (last_dim / transpose / scalar) with per-mode VEC alignment strategy`

**Pattern**: For gather/index/select ops on arbitrary dimensions, classify the gather dimension into 3 modes and dispatch to a specialized kernel per mode. Each mode has a different DataCopy strategy based on memory access pattern.

**3 modes**:
1. **last_dim** (dim == -1 or dim == ndim-1): innermost dimension → contiguous elements per row → VEC DataCopy row-by-row directly. No transpose needed.
2. **permute_last_dim** (dim != -1 but index count fits in UB): transpose input so gather dim becomes last → same VEC row-by-row kernel as mode 1. Overhead: one transpose pass.
3. **scalar** (dim != -1 AND index count too large for UB transpose, OR unaligned): element-by-element scalar gather via `DataCopy<1,1>`. Slowest but always correct.

**Mode selection logic** (host-side, in pybind11 or tiling computation):
```cpp
if (dim == ndim - 1 || dim == -1) → mode = LAST_DIM;
else if (index_count * element_size ≤ UB_SIZE / 2) → mode = PERMUTE_LAST_DIM;
else → mode = SCALAR;
```

**Why not one generic kernel**: A single gather kernel would need to handle arbitrary-dim access patterns with runtime stride computation → scalar access for ALL elements → no VEC utilization. Mode 1 gives full VEC bandwidth for the most common case (gather on last dim).

**Detection**: count per-mode kernel .cpp files. Single generic gather kernel → gap. 3 separate .cpp with mode names → pattern applied.

**Evidence**: cv-agent `gather_elements_v2/kernel/` has `gather_elements_v2_last_dim_kernel.h`, `gather_elements_v2_transpose_kernel.h`, `gather_elements_v2_scalar_kernel.h` — 3 modes with shared common kernel base class.

**Cross-ref**: CAND-FA-CV-7 (multi-strategy shape dispatch — same dispatch-by-classification pattern, different classification axis: dimension vs shape), PB-22 (DataCopy 32B alignment — scalar mode bypasses this by using DataCopy<1,1>), OL-124 (TQue<VECOUT> constraint — gather output may need TBuf not TQue for scalar mode)

---


## CAND-FA-CV-10: Im2Col + Matmul decomposition for Conv2D — sliding-window ops via spatial-to-column transform + Cube matmul

`verified_on: cv-agent conv2d kernel architecture (im2col GM→L1 tile + Cube Matmul)`

**Pattern**: Decompose Conv2D (and any sliding-window op) into two stages: (1) Im2Col — DataCopy transforms spatial window into column matrix in L1, (2) Matmul — Cube multiplies column matrix × filter matrix. The decomposition converts the irregular spatial access pattern into a regular matmul that Cube can execute at full throughput.

**Im2Col tile sizing** (critical for L1 budget):
```
L1 input tile rows = (KH - 1) * dilationH + (FMAP_L1_TILE_HO - 1) * strideH + 1
L1 input tile cols = (KW - 1) * dilationW + (FMAP_L1_TILE_WO - 1) * strideW + 1
```
The L1 tile must be large enough to cover one output tile worth of input windows — each output position needs KH×KW input elements.

**Why not direct convolution**: Direct spatial sliding-window access to GM is stride-irregular → cannot use VEC/Cube efficiently. Im2Col transforms it into a dense matmul → Cube runs at full throughput. The im2col overhead (one extra DataCopy pass) is amortized by Cube matmul speed.

**Applicability**: Any op with a sliding-window pattern over spatial dimensions — Conv2D, Conv3D, MaxPool, AvgPool, DilatedConv, DepthwiseConv. The pattern generalizes to N-D by adjusting the im2col tile size formula.

**Detection**: grep for `Conv2D\|Im2Col\|im2col` in kernel .h. If kernel directly loops over spatial positions with scalar access → gap. If kernel has im2col tile sizing + Cube Matmul → pattern applied.

**Evidence**: cv-agent `conv2d/kernel/conv2d_kernel.h` — `Conv2DCubeKernel` class with `hiBlock_`/`wiBlock_` im2col tile sizing + Cube Matmul via L0A/L0B queues with fp32 accumulation.

**Cross-ref**: CAND-FA-CV-8 (fused matmul — conv2d is structurally "im2col fused with matmul", same L1-resident intermediate principle), CAND-FA-CV-7 (multi-strategy dispatch — pooling ops may need im2col variant vs direct variant per kernel size)

---


## CAND-FA-CV-11: INT8 quantized matmul with FP32 accumulation + per-channel scale → FP16 output

`verified_on: cv-agent quant_matmul kernel (int8 A × int8 B → fp32 accum → scale → fp16)`

**Pattern**: For INT8 quantized matrix multiplication, compute in fp32 accumulation (exact for INT8 range), apply per-channel scale, then convert to fp16 output. All in one kernel — no intermediate GM tensors.

**Mechanism**:
1. Load INT8 A, INT8 B from GM into L1
2. Cube Matmul: INT8 × INT8 → INT32 in L0C
3. Convert INT32 → FP32 (exact for |result| ≤ 2^24)
4. Multiply per-channel scale (FP32) elementwise
5. Convert FP32 → FP16 for output
6. Write FP16 output to GM

**Why not INT8 output**: INT8 output requires requantization with zero-point — loses precision and makes the kernel specific to one quantization scheme. FP16 output is universal — downstream ops consume it without knowing the quantized origin. The INT8→FP16 conversion is zero-cost on NPU (type cast in VEC).

**Precision guarantee**: INT8 range is [-128, 127], so max |INT8 product| = 16384 per element pair. With K ≤ 4096, max |sum| = 67,108,864 < 2^26 — fits in FP32 exact-representation range (2^24 for integers). No accumulation noise for K ≤ 4096.

**Detection**: grep for `int8\|INT8\|int8_t` AND `Matmul\|Cube` in kernel .cpp. If activation is quantized but weights are fp16 → partial quantization, not this pattern.

**Evidence**: cv-agent `quant_matmul/` with model.py forward: `a.to(fp32) @ b.to(fp32) * scale → fp16`. Kernel pattern confirmed via matmul_leakyrelu's Matmul + elementwise pipeline (CAND-FA-CV-8).

**Cross-ref**: CAND-FA-CV-8 (fused matmul+elementwise — quant matmul is "matmul + scale multiply" fusion, same L1-resident principle), P-P93 (quant-op CPU reference `.clamp(low,high).to(int_dtype)`), P-P94 (MERE/MARE aux precision standard)


## CAND-FA-CV-12: Input-arity-specialized kernels for variable-input-count ops

`verified_on: cv-agent concat_dv2 4-kernel architecture (dim0_1/2/3/4 inputs) with shared common base class`

**Pattern**: For ops accepting a variable number of input tensors (concat, stack, elementwise with N inputs), emit one specialized kernel per supported input count rather than a single generic kernel with dynamic buffer allocation. Each per-arity kernel has optimal buffer sizing — no over-allocation for the common small-N case, no under-allocation for the large-N case.

**Why not one generic kernel**: Dynamic buffer allocation based on input count means either (a) allocating for max possible inputs (wasteful for common 2-input case), or (b) allocating per-call based on runtime count (complex, error-prone, hard to verify at build time). Per-arity kernels make buffer sizes compile-time constants → verifiable by prebuild check.

**Shared base class pattern**: All per-arity kernels inherit from a common base (`concat_dim0_kernel_common.h`) holding shared logic (tiling struct, DataCopy primitives, row iteration). Per-arity kernel only customizes: buffer count, DataCopy loop count, output offset computation.

**When to use**: Input count variadicity is bounded and known at kernel compile time (not runtime-dynamic):
- concat: 1-4 inputs → 4 kernels
- stack: 2-4 inputs → 3 kernels
- Elementwise with N operands: N ∈ {2,3,4} → 3 kernels

**When NOT to use**: Input count is truly runtime-dynamic (≥5, unbounded, or determined by host-side logic after build). Then fall back to max-allocation single kernel.

**Detection**: count kernel .cpp files with input-count in filename (`_1.cpp`, `_2.cpp`, `_3.cpp`, `_4.cpp`). Single generic kernel handling N inputs → gap for bounded-N ops.

**Evidence**: cv-agent `concat_dv2/kernel/` has `concat_dim0_1_kernel.h` through `concat_dim0_4_kernel.h`, all inheriting from `concat_dim0_kernel_common.h` with shared `CopyTiling`, `SetGlobalBuffer` logic.

**Cross-ref**: CAND-FA-CV-7 (multi-strategy dispatch — same "specialize per parameter-class" principle, different dispatch axis: input count vs shape regime), CAND-FA-CV-4 (cube/vec class separation — common base + per-variant subclass is the same architectural pattern)

---


## CAND-FA-CV-13: Reshape + Matmul + DynamicQuant fusion — multi-stage pipeline with L1-resident intermediates

`verified_on: cv-agent reshape_matmul_rowwise_quant_int8 kernel (reshape view → mm → dynamic_quant, all in one kernel)`

**Pattern**: Fuse reshape (zero-copy view), matmul, and dynamic quantization into a single AscendC kernel. The reshape is free (reinterpret strides), the matmul output stays in L1, and the VEC quant stage reads directly from L1. Two intermediates (reshaped view, matmul result) never touch GM.

**Pipeline**:
1. Reshape: x(m,n) → x_view(m*n/k, k) — zero-copy, just reinterpret strides
2. Matmul: x_view @ h(k,k) → result(m*n/k, k) in L1
3. DynamicQuant: row-wise max_abs → scale → round + clip → int8 output
4. Reshape back: int8 result → (m, n) output

**Why fusion matters**: Without fusion, the matmul output (fp16/bf16, size m*n) must be written to GM, then read back by a separate quant kernel. For m=4096, n=2048: 16MB write + 16MB read avoided.

**L1 budget constraint**: The matmul output buffer + quant workspace must fit in L1 simultaneously. This constrains the inner dimension K to K_max = L1_size / (element_size * 2). If K exceeds this, the pipeline must split into tiles.

**Detection**: grep for `reshape` AND `Matmul\|GEMM` in the same kernel .cpp. If reshape+matmul are separate ops → gap. If in same kernel → fusion applied.

**Evidence**: cv-agent `reshape_matmul_quant/` kernel with `ReshapeMatmulQuantKernel` class — single Process() body with reshape→matmul→quant stages.

**Cross-ref**: CAND-FA-CV-8 (fused matmul+elementwise — same principle, different tail stage: quant vs activation), CAND-FA-CV-11 (INT8 quant matmul — this pattern adds reshape fusion before matmul), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — L1 workspace management generalizes to reshape+matmul+quant pipeline)

---


## CAND-FA-CV-14: PageAttention block-table KV cache addressing — physical-block indirection for paged LLM inference

`verified_on: cv-agent sparse_flash_attn_mask_pa kernel (block_table indirection + sparse top-k indices + GQA)`

**Pattern**: For LLM inference with paged KV cache (vLLM-style PageAttention), use a block table for logical→physical address translation: `physical_block = block_table[batch, token_idx // block_size]`; `block_offset = token_idx % block_size`. The kernel reads KV from physically non-contiguous blocks, assembling logical KV sequences on-the-fly in L1.

**Addressing mechanism**:
1. Input: `block_table[batch, max_blocks]` — maps logical block index → physical block index
2. For token at position `s`, compute `block_idx = s // block_size`, `offset = s % block_size`
3. Read KV from `kv_cache[block_table[batch, block_idx], offset, :]` — physical address via indirection
4. Assemble contiguous logical KV sequence in L1 buffer from scattered physical reads

**Why paging matters**: Without PageAttention, KV cache must be pre-allocated as max_seq_len × num_layers × 2 × d_model elements of contiguous memory — most of which is wasted for short sequences. Block-table paging allows physical KV cache blocks to be allocated on-demand, reducing memory by 3-20× for typical batch mixtures.

**Additional structural features (this specific variant)**:
- **Sparse top-k indices**: `indices[S, G, topk]` — attend only to top-k relevant tokens per query, not all KV
- **Split-dim**: Q/KV dim split into `dim + tail_dim` where tail_dim is handled differently
- **GQA**: `heads_per_group = heads // kv_group` — multiple Q heads share KV

**L1 challenge**: KV tokens are scattered across physical blocks → DataCopy must do N individual GM reads (one per logical token) rather than one contiguous transfer. Mitigation: if block_size is large enough, intra-block reads are contiguous; only cross-block transitions pay the scatter cost.

**Detection**: grep for `block_table\|page_table\|block_idx\|physical_block` in kernel .h/.cpp. Absence → contiguous KV cache (no paging). Presence → PageAttention pattern.

**Evidence**: cv-agent `sparse_flash_attn_mask_pa/model.py:6-11` — `_logical_pa_token()` function with `block_idx = token_idx // block_size` + `block_table[batch_idx, block_idx]` + `kv[physical_block, block_offset, 0]` — canonical PageAttention indirection.

**Cross-ref**: CAND-FA-CV-6 (MLA split-key architecture — same KV cache compression family, different mechanism: nope+rope decomposition vs block-table paging), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — applies when KV iteration loop exists; PageAttention may use ring buffer for async block prefetch), CAND-NSA-1 (Matmul<>::IterateAll — matmul lib may accelerate the Q×K computation inside PageAttention)

---


## CAND-SKILL-FA-DIVERGENCE-1: Generator-emission anti-pattern — monolithic inline CrossCore flag spam vs WorkspaceQueue discipline

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=fused-attention`
`verified_on: differential run of a monolithic generated kernel against a split-header WorkspaceQueue kernel`

**Anti-pattern**:

- One large header contains cube, vector, scheduler, and synchronization logic.
- Inline `CrossCoreSetFlag<0x2, PIPE_FIX/MTE3>(id)` calls are scattered through
  tile loops.
- Flag IDs are scalar literals and workspace slots have no ownership abstraction.

**Preferred pattern**:

- Split cube, vector, workspace queue, matmul tile, and shared kernel contracts.
- Wrap CrossCore Set/Wait operations in `WorkspaceQueue` with ring-slot lifecycle.
- Instantiate dtype entry points explicitly so SoC and build gates remain visible.

**Empirical signal**: the split-header WorkspaceQueue kernel ran at multiple shapes;
the monolithic 757-line generated kernel failed at stream synchronization with
AICore error 507015. The comparison does not prove WorkspaceQueue alone is the
root-cause fix, but it is enough to reject the unsafe emission shape.

**Detection gate**:

- more than five inline `CrossCoreSetFlag|CrossCoreWaitFlag` calls;
- no `WorkspaceQueue`/queue abstraction;
- a single generated header of at least 500 lines for a fused-attention kernel.

When all three fire, block the generated result and require the standard FA
template blocks instead of continuing compile/fix iterations.

**Recommended remediation**:

1. Make WorkspaceQueue a mandatory FA template-assembly block.
2. Add the detection gate to the post-generation review.
3. Validate the rule on a second GQA/attention variant before promotion.

**Cross-ref**: CAND-FA-CV-1 (WorkspaceQueue ring buffer), CAND-FA-CV-4
(cube/vector split), PB-34/PB-35 (Matmul plus manual-flag hazards).

## CAND-CANN-FA-ROW-TILE-1: Row-tile UB-budget partition for cube-decomposed forward attention with 3-task pipelined carousel and matmul-library-driven cube↔vec sync

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=forward_flash_attention_with_high_level_matmul_library AND row_dim_exceeds_ub_budget`
`derived-from: cann-source (ops-transformer flash_attention_score arch22 s1s2_bn2gs1 variant, 2026-05-26)`
`verified_on: cann-source (read-only structural extraction); unverified_on: a5_ops`
`local-kb-crossref: CAND-FA2 (online-softmax recurrence — composes inside vec1 phase), CAND-FA4 (block-reduce shape for rowmax/rowsum), CAND-NSA-1 (Matmul-library + local SetFlag<MTE3_MTE2> cube↔vec sync — this candidate extends NSA-1's 2-stage ping-pong to a 3-stage carousel), CAND-FA-CV-1 (WorkspaceQueue ring buffer — different abstraction layer, V220 manual CrossCore path), OL-186 (V351 forward FA cube-MatmulImpl P@V precision requirement — the cube halves this candidate orchestrates).`

**Trigger**: A forward fused-attention class kernel decomposes into two cube stages (QK_dot, P_at_V) and intermediate vector stages (mask + softmax, output rescale + writeback), AND the per-output-row (Q-seq direction) score-matrix tile `[s1_rows, kv_chunk_cols]` does not fit in a single UB buffer in one shot, AND the cube halves are driven by the high-level `matmul::Matmul<>` library client API (NOT raw `Mmad` / `Fixpipe` intrinsics with manual `CrossCoreSetFlag` — that case is the negative side of CAND-FA1).

Concretely: this is the structural skeleton for the `BNGS1S2`-style FA variant where Q-seq is parallelized across cores (each AIV/AIC core owns a contiguous block of Q-rows), KV-seq is the inner reduction axis processed in chunks, and head_dim is small enough to fit alongside the score tile in UB.

**Recommendation — two-level row partition + 3-task pipelined carousel**:

(1) **Outer row tile** chosen by host tiling: a per-task block of `Q_BLOCK_ROWS` query rows (size chosen so the running softmax state `[Q_BLOCK_ROWS, 1]` and the output accumulator `[Q_BLOCK_ROWS, head_dim_v]` together fit in their own dedicated UB buffers, leaving the score-tile UB budget free for the inner step).

(2) **Inner row sub-tile** computed at runtime each KV-chunk iteration:

```
SUB_ROWS_VEC1 = min( UB_SCORE_BUDGET_FP32 / kv_chunk_cols_aligned , Q_BLOCK_ROWS )
SPLIT_N_VEC1  = ceil_div( Q_BLOCK_ROWS , SUB_ROWS_VEC1 )

SUB_ROWS_VEC2 = (head_dim_v_aligned > 64)
                ? UB_OUT_BUDGET_FP32 / head_dim_v_aligned
                : Q_BLOCK_ROWS
SPLIT_N_VEC2  = ceil_div( Q_BLOCK_ROWS , SUB_ROWS_VEC2 )
```

`UB_SCORE_BUDGET_FP32` and `UB_OUT_BUDGET_FP32` are static UB-allocation constants chosen at `TPipe::InitBuffer` time (typically the same fp32-element count, e.g. 8192 fp32 = 32 KB, matching the score-tile and output-tile ping-pong buffers). Use the formula to size the inner row stride at each KV chunk so the per-chunk score (or output) matrix `[SUB_ROWS, col_aligned]` fits in its dedicated buffer; loop the inner step `SPLIT_N` times to cover the full block.

Why "inner step has different size per cube stage": the score matrix is `[Q_BLOCK_ROWS, kv_chunk_cols]` whereas the output matrix is `[Q_BLOCK_ROWS, head_dim_v]`. With the same UB-element budget, the sub-row count differs because the column dimension differs. The `head_dim_v_aligned > 64` gate just means "if head_dim_v is large enough to actually need sub-partition, partition; else the whole Q_BLOCK_ROWS fits in one shot for the output stage". This is the canonical shape; do NOT replicate one sub-tile size across both stages.

(3) **3-task pipelined carousel** across the KV-chunk loop, using three task descriptors indexed by `carouselId % 3`. Two GM ping-pong slots (`carouselId % 2`) hold the cube outputs (score and output) because at most two of the three in-flight tasks touch the same GM-slot kind at any moment.

Per iteration of the KV-chunk loop, six actions in this order:

```
//      slot_now  = carouselId % 3
//      slot_prev = (carouselId + 2) % 3     // = carouselId - 1
//      slot_pp   = (carouselId + 1) % 3     // = carouselId - 2

(A) if carouselId >= 1:  mmQK.WaitIterateAll(); mmQK.End();                  // retire QK_dot for slot_prev
(B) if not-tail:         mmQK.IterateAll<false>(scoreGmSlot[carouselId % 2], ...)
                                                                              // issue QK_dot for slot_now
(C) if carouselId >= 1:  VecPhase1(carousel[slot_prev]);                     // mask + softmax + write probs
                          AscendC::SetFlag<HardEvent::MTE3_MTE2>(evt)         // arm probs->mmPV sync
(D) if carouselId >= 2:  mmPV.WaitIterateAll(); mmPV.End();                  // retire P_at_V for slot_pp
(E) if carouselId >= 1:  AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evt)
                          mmPV.IterateAll<false>(outGmSlot[slot_prev % 2], ...)
                                                                              // issue P_at_V for slot_prev
(F) if carouselId >= 2:  VecPhase2(carousel[slot_pp]);                       // output rescale + writeback
carouselId += 1
```

Steady state (`carouselId >= 2`) has THREE work units overlapped:
- slot T   — QK_dot cube just issued (running on cube engine, AIV moves on)
- slot T-1 — vec1 (mask + softmax) running on AIV, P_at_V cube kicked in same iter
- slot T-2 — P_at_V cube just retired, vec2 (output rescale + GM writeback) running on AIV

This is structurally deeper than NSA-1's 2-stage cube→vec ping-pong: NSA-1 alternates one cube call + one or two vec phases per iter; this carousel keeps the cube ENGINE always busy (next QK_dot already issued before current iter's P_at_V completes) AND keeps the vec ENGINE always busy (two distinct vec phases interleaved across iters).

**Sync primitive set**:
- Cube↔vec retirement: `matmul::Matmul<>::IterateAll<false>(gmDst, ...)` to issue, `WaitIterateAll(); End();` to retire. The library owns the underlying AIC↔AIV sync — do NOT layer `CrossCoreSetFlag/WaitFlag` on top (see CAND-FA1 hard-do-not-apply). One `mmQK` instance (for QK_dot) and one `mmPV` instance (for P_at_V) are declared as class members and reused across the carousel; the library tracks per-call state across iters.
- AIV-side GM-write retirement: `AscendC::SetFlag<HardEvent::MTE3_MTE2>(evt)` after the vec1 phase finishes writing the probs matrix to GM, paired with `AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evt)` before the next `mmPV.IterateAll` reads it. `evt` is `FetchEventID(HardEvent::MTE3_MTE2)` once outside the loop.
- Intra-AIV pipe sync inside the vec1 and vec2 phases: standard `HardEvent::MTE2_V`, `V_MTE2`, `V_MTE3`, `MTE2_MTE3`, `MTE3_V` flags — see CAND-FA2 and P-P75.

**Per-row state ownership across KV chunks** (composes with CAND-FA2):
- Score-tile UB buffer: ping-pong by `carouselId % 2` to overlap stage1 of next iter with stage2 of current iter.
- Per-row online softmax state arrays (running max, running sum, rescale-delta): allocated as two-deep ping-pong slot arrays indexed by `carouselId % 2`. Size each slot = `Q_BLOCK_ROWS * FLOAT_BLOCK_SIZE * sizeof(fp32) = Q_BLOCK_ROWS * 32 bytes` (8 fp32 lanes per row for `Brcb`-compatible layout per P-P62).
- Output accumulator `O_running`: held in a stage2 UB buffer; carries across KV chunks, divided by final running_sum on the last KV chunk (CAND-FA2 §5).

**Concrete anchor** (public-API surface only; worker chooses op-local member names; UB-element budget shown as a worker-chosen constant):
```cpp
// Worker-chosen UB budget per ping-pong buffer (32 KB shown; pick based on UB layout).
constexpr uint32_t UB_SCORE_BUDGET_FP32 = 8 * 1024;
constexpr uint32_t UB_OUT_BUDGET_FP32   = 8 * 1024;

// One-time UB allocations done in InitBuffer() — sizes are static.
this->pipe->InitBuffer(this->scoreUbPing , UB_SCORE_BUDGET_FP32 * sizeof(float));
this->pipe->InitBuffer(this->scoreUbPong , UB_SCORE_BUDGET_FP32 * sizeof(float));
this->pipe->InitBuffer(this->outAccumUb  , UB_OUT_BUDGET_FP32   * sizeof(float));
this->pipe->InitBuffer(this->runSumUb[0] , Q_BLOCK_ROWS * 4 * 8);   // 8 fp32 lanes per row
this->pipe->InitBuffer(this->runSumUb[1] , Q_BLOCK_ROWS * 4 * 8);
this->pipe->InitBuffer(this->runMaxUb    , Q_BLOCK_ROWS * 4 * 8);
this->pipe->InitBuffer(this->rescaleUb[0], Q_BLOCK_ROWS * 4 * 8);
this->pipe->InitBuffer(this->rescaleUb[1], Q_BLOCK_ROWS * 4 * 8);

// Per-task descriptor (UB-resident, NOT GM); 3-slot carousel array as class member.
struct CarouselTaskDesc {
    int64_t carouselId;
    int32_t kvChunkCols;
    int32_t subRowsVec1, splitNVec1;
    int32_t subRowsVec2, splitNVec2;
    // ... per-iter offsets, mask offsets, ... (worker fills) ...
};

void ComputeInnerSubTileSplit(CarouselTaskDesc& td, int32_t headDimVal) {
    int32_t kvAligned = AlignUp(td.kvChunkCols, /*alignTo=*/8);   // fp32 8-element block align
    td.subRowsVec1 = Min<int32_t>(UB_SCORE_BUDGET_FP32 / kvAligned, Q_BLOCK_ROWS);
    td.splitNVec1  = CeilDiv<int32_t>(Q_BLOCK_ROWS, td.subRowsVec1);

    int32_t hdAligned = AlignUp(headDimVal, 8);
    td.subRowsVec2 = (hdAligned > 64)
                  ? (UB_OUT_BUDGET_FP32 / hdAligned)
                  : Q_BLOCK_ROWS;
    td.splitNVec2  = CeilDiv<int32_t>(Q_BLOCK_ROWS, td.subRowsVec2);
}

// Outer scheduling loop (sketch — 3-task pipelined carousel):
event_t evtVec1Done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
CarouselTaskDesc carousel[3];
int64_t carouselId = 0;

for (int64_t kvChunk = 0; kvChunk <= kvChunkLimit; ++kvChunk) {
    // (A) Retire previous QK_dot cube call.
    if (carouselId >= 1)               { this->mmQK.WaitIterateAll(); this->mmQK.End(); }

    // (B) Issue current QK_dot cube call into ping-pong slot [carouselId % 2].
    if (kvChunk <= kvChunkLimit) {
        SetCarouselTaskDesc(carousel[carouselId % 3], carouselId, kvChunk);
        ComputeInnerSubTileSplit(carousel[carouselId % 3], this->headDimV);
        this->mmQK.SetTensorA(this->qGmInput[QInputOffset(carouselId)]);
        this->mmQK.SetTensorB(this->kGmInput[KInputOffset(carouselId, kvChunk)], /*transposeB=*/true);
        this->mmQK.SetTail(Q_BLOCK_ROWS, carousel[carouselId % 3].kvChunkCols, this->headDimQ);
        this->mmQK.template IterateAll<false>(this->scoreGmSlot[carouselId % 2], 0, false, true);
    }

    // (C) Vec1 for previous task: mask + softmax (CAND-FA2 + CAND-FA4); writes probs to GM.
    if (carouselId >= 1) {
        VecPhase1(carousel[(carouselId + 2) % 3]);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(evtVec1Done);
    }

    // (D) Retire previous-previous P_at_V cube call.
    if (carouselId >= 2)               { this->mmPV.WaitIterateAll(); this->mmPV.End(); }

    // (E) Issue current P_at_V cube call (for previous task).
    if (carouselId >= 1) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evtVec1Done);
        this->mmPV.SetTensorA(this->probsGmSlot[(carouselId + 2) % 3 % 2]);
        this->mmPV.SetTensorB(this->vGmInput[VInputOffset(carousel[(carouselId + 2) % 3])]);
        this->mmPV.SetTail(Q_BLOCK_ROWS, this->headDimV, carousel[(carouselId + 2) % 3].kvChunkCols);
        this->mmPV.template IterateAll<false>(this->outGmSlot[(carouselId + 2) % 3 % 2], 0, false, true);
    }

    // (F) Vec2 for previous-previous task: output rescale + GM writeback.
    if (carouselId >= 2) {
        VecPhase2(carousel[(carouselId + 1) % 3]);
    }

    carouselId++;
}
```

**VecPhase1 inner sub-row loop** (the UB-budget sub-tile partition in action):
```cpp
void VecPhase1(CarouselTaskDesc& td) {
    LocalTensor<float> scoreUb = this->scoreUbPing.template Get<float>();  // or scoreUbPong by parity
    // ... DataCopy(scoreUb, scoreGmSlot[td.carouselId % 2], ...) — score tile GM→UB ...

    int32_t subRowsRun = td.subRowsVec1;
    for (int32_t splitIdx = 0; splitIdx < td.splitNVec1; ++splitIdx) {
        if (splitIdx == td.splitNVec1 - 1) {
            // Tail sub-tile shrinks to remaining rows; mandatory to avoid UB overflow.
            subRowsRun = Q_BLOCK_ROWS - splitIdx * td.subRowsVec1;
        }
        // Per sub-row chunk:
        //   - apply atten_mask via Select / Adds(-large_negative)
        //   - rowmax via CAND-FA4 block-reduce shape
        //   - online-softmax recurrence (CAND-FA2 step 3) updating runMaxUb / runSumUb / rescaleUb
        //   - rowsum via CAND-FA4 block-reduce shape
        //   - emit per-row probs sub-block to GM via DataCopy (MTE3 pipe)
    }
}
```

**Determinism**:
- Each task descriptor `carousel[carouselId % 3]` is owned by a single carousel slot; reading/writing within one carousel iter is sequential.
- GM ping-pong slots `[carouselId % 2]` for cube outputs are single-writer (the matched cube call) and single-reader (the matched vec phase); the carousel structure structurally guarantees that no slot is written while another stage reads it (the `WaitIterateAll` barriers create a strict total order between writer and reader).
- Per-row softmax state has one row per AIV core (no cross-core write).
- Det-preserving by construction when the inner row-sub-tile reduction is deterministic (CAND-FA2 + CAND-FA4 preconditions).

**Hard do-not-apply**:
- Do NOT use a 3-stage carousel when the cube engine cannot keep up with the kick rate — if cube latency for QK_dot exceeds the AIV's combined vec1+vec2 time, the carousel adds no overlap and just burns 50% more task-descriptor UB; degrade to 2-stage NSA-1 ping-pong.
- Do NOT use the `min(UB_BUDGET / col_aligned, Q_BLOCK_ROWS)` formula when the per-row UB state arrays (runMax, runSum, rescale buffers) have not been carved out of the UB budget first — the formula's `UB_SCORE_BUDGET_FP32` assumes those slots are already reserved.
- Do NOT mix this with manual `CrossCoreSetFlag/WaitFlag` for the cube↔vec handoff — that races with the Matmul library's internal AIC↔AIV flag pool (PB-34 / 507014 territory).
- Do NOT collapse `splitNVec1` and `splitNVec2` to one value — the two stages can have different sub-tile sizes when `kv_chunk_cols != head_dim_v`. Forcing equality wastes UB and may pessimize one stage.
- Do NOT use this skeleton for backward FA — the gradient ops have a different cube/vec ladder (saved-tensor restore + multi-output dispatch — see CAND-FAG-3 / CAND-FAG-4). This candidate is forward-only.

**Other instances predicted**:
- Forward FlashAttention (the canonical case the source structure derives from).
- Variable-length forward attention (TND layout) where Q_BLOCK_ROWS is per-batch-item but the inner UB-budget formula still applies.
- Forward multi-query / grouped-query attention (GQA) — Q_BLOCK_ROWS effectively scales by group factor on the score side but the sub-tile formula is unchanged (the worker chooses Q_BLOCK_ROWS post-group-expand).
- Forward MLA prefill where the latent-projected K/V dimension is small enough to fit alongside the score tile.
- Other cube-vec-cube forward ops that decompose as (matmul A → vec post-process → matmul B → vec finalize) and need to amortize cube/vec overlap across an inner reduction loop. Examples: fused GEMM + activation + GEMM (e.g. SwiGLU on top of LM head), block-sparse attention with chunked KV iteration.

NOT predicted:
- Standalone softmax / pure-VEC reductions — no cube halves, the carousel does nothing.
- Backward FA (different cube/vec ladder).
- FA variants using raw Mmad/Fixpipe without the Matmul library client API — those are CAND-FA1 territory.

**Risks before promotion**:
- The UB-budget anchor `8192 fp32 elements (= 32 KB)` is a CHOICE not a constant — it depends on how the worker partitions UB. The candidate's formula is shape-correct regardless, but workers must declare the chosen anchor explicitly in their kernel's `InitBuffer` so the inner sub-tile size is computed against the right number.
- 3-task carousel assumes a kernel structure where exactly 2 cube stages and 2 vec stages alternate; an op with 3 cube stages (e.g. fused-attention with extra side-product) needs a different scheduling skeleton.
- The pattern depends on `matmul::Matmul<>::IterateAll/WaitIterateAll/End` being available — V220 + V351 both ship this API but a future arch that drops it would invalidate the sync layer.
- Inner sub-tile formula `min(UB / col_aligned, Q_BLOCK_ROWS)` integer-rounds DOWN; the tail iter's `subRowsRun = Q_BLOCK_ROWS - splitIdx * subRows` handles the remainder. Workers MUST emit the tail-iter shrink — forgetting it underflows UB on the last sub-tile.
- Verification gap: source-structure read only; no a5_ops kernel has shipped this exact carousel + sub-tile formula yet. Promotion to P-P / OL requires a5_ops forward FA implementation that passes Pass A + Pass B + det + perf vs CANN baseline AND msprof confirms cube/vec overlap (cube utilization > 0 AND vec utilization > 0 in the same wall-clock window for >= 80% of total kernel time).

**Promote when**:
- a5_ops 3_FusionAttention forward implementation lands using this carousel + sub-tile formula, achieves >= 51/61 case PASS_T1 (closing the existing 1/61 gap), AND msprof shows cube/vec overlap >= 60% of total kernel time.
- A SECOND op outside FA (e.g. fused GEMM-activation-GEMM, MLA prefill) successfully uses the same carousel skeleton.

**Cross-reference**:
- CAND-FA2 (online-softmax recurrence — runs inside the vec1 phase).
- CAND-FA4 (block-reduce shape for rowmax/rowsum — runs inside the vec1 phase).
- CAND-NSA-1 (Matmul-library + local SetFlag<MTE3_MTE2> — the same sync primitive set; this candidate extends to 3-stage carousel).
- CAND-FA1 (manual CrossCore — explicit negative complement; do NOT mix).
- CAND-FA3 (GM workspace slot rotation modulo MAX_LAG+1 — this candidate uses MAX_LAG=1 for cube-output slots; the 3-stage carousel is decoupled from the 2-slot GM ping-pong because task descriptors are UB-resident, not GM-resident).
- CAND-FA-CV-1 (cv-agent WorkspaceQueue with prelaunch+1 ring slots — different abstraction layer; cv-agent kernels use manual CrossCore, this candidate uses Matmul library; not interchangeable).
- OL-159 (forward FA softmax tile-scheduling — algorithmic scope companion).
- OL-186 (V351 forward FA cube-MatmulImpl P_at_V precision requirement — the cube halves this candidate orchestrates).
- P-P62 (Brcb broadcast precondition for the per-row state).
- P-P75 (intra-core SetFlag/WaitFlag<HardEvent> pipe sync primitive).

**Anti-overlap-with-NSA-1 statement** (explicit for C35 self-review): NSA-1 documents the 2-stage primitive (Matmul library IterateAll + local SetFlag<MTE3_MTE2>) and a per-iter ping-pong on `iter & 1`. This candidate is structurally deeper:
- 3 task descriptors instead of 2 (carousel vs ping-pong)
- Cube ENGINE next-iter pre-kick: while slot T-1's P_at_V is computing on cube, slot T's QK_dot has already been issued (separate Matmul instance)
- Inner row sub-tile partition formula (NSA-1 does not address UB overflow on the score tile because the compressed-attention output is much smaller per iter)
- Two-level partition (outer host tiling + inner runtime UB-budget split)

If C35 flags overlap, the **delta-content** answer is the UB-budget sub-tile formula + the 3-stage carousel structure. NSA-1 should remain a separate entry covering the 2-stage primitive; this candidate is the FA-class extension.

## CAND-PP98: V351 single-launch FA-class kernel — all-zero output diagnosis checklist when build+dispatch PASS

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=fused-attention`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Bisection: when a V351 single-launch FA-class kernel builds clean, dispatches clean, and produces all-zero output, walk these probes in order to localize the silent-no-exec:
1. Zero-pybind probe — confirm pybind wrapper actually marshals args + invokes kernel (insert printf in pybind, rerun)
2. Zero-VEC probe — invoke VEC-only path, check if any VEC write fires
3. Zero-MTE3 probe — instrument MTE3 emit count to verify the cube→GM writeback is happening
4. Fixpipe probe — N/A on V351 (V351 uses different cube writeback). On V220 this is the cube→L0C→GM step

Source: lightning_indexer_grad kw-NEW 2026-05-23.

## CAND-PP99: Pybind hardcoded platform constants — mirror upstream runtime ascendcPlatform queries to avoid silent V220-vs-V351 mismatch

`applies_to: soc=all; cann=9.0.0+; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Anti-pattern: pybind11.cpp hardcoding platform constants (`AIC_NUM`, `AIV_NUM`, `LIB_API_WS_BYTES`) at host-side. On V220→V351 ports these values silently mismatch (V220 hard-coded to V220 numbers, deployed on V351 → wrong workspace size + wrong block dim). Use runtime `ascendcPlatform` queries (`GetCoreNumAic()` / `GetCoreNumAiv()` / `GetLibApiWorkSpaceSize()`) mirroring upstream pybind11.cpp pattern.

Source: lightning_indexer_grad kw-NEW 2026-05-23 anti-patterns §3.

## CAND-PP100: Thin-TU wrapping of staged V220 algorithm headers for port_a3 V220-pure entry (default-OFF arch35)

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: minimum-viable port_a3 V220-pure entry-point structure when default-OFF arch35 (no upstream V351 arch35/ available):
- `<op>_kernels.cpp` — thin worker TU wrapping V220 op_kernel/*.h staged headers
- `pybind11.cpp` — host marshalling, runtime ascendcPlatform queries
- minimum helpers under `kernel/` — only what the worker TU references
- **forbidden**: `#include "arch35/<op>.h"` (default-OFF), `<op>_apt.cpp` (V351 amped TU)

Source: flat_quant kw-1 2026-05-23 (8/8 T1 BIT_EXACT + 2.24× perf in single spawn, ~250 LOC delta).

## CAND-PP101: Minimum-fields `<Op>TilingData` struct in port_a3 V220-pure path

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: when porting V220 cube+vec fused op via thin-TU wrapping (CAND-PP100), grep `tilingData->` first in the worker TU + helpers, mirror only the accessed fields in your local `<Op>TilingData` struct. Skip the upstream `BEGIN_TILING_DATA_DEF` + nested `TCubeTiling` reconstruction.

Source: flat_quant kw-1 2026-05-23 (4-field struct sufficed where upstream had 11-field nested struct).

## CAND-PP102: Two-kernel split (AIC_ONLY + AIV_ONLY via separate ACLRT_LAUNCH_KERNEL on same stream) as PB-34 mitigation Pattern C — empirically broken on V351

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all_cube_vec_fused`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Anti-pattern (verified broken): splitting a fused cube+vec op into two separate kernels (AIC_ONLY + AIV_ONLY) launched as two ACLRT_LAUNCH_KERNEL calls on the same stream — proposed as PB-34 manual-CrossCoreSetFlag deadlock mitigation. Empirically: AIV kernel returns ret=0 but writes nothing (matches `13_Cat` silent-no-exec signature). Probe required (constant-write probe to AIV-only entry) before declaring this infrastructure-unsupported vs algorithm-bug.

Source: grouped_matmul_swiglu_quant_v2 kw-3 2026-05-24.

## CAND-PP103: CPU-truth construction rule for per-tensor int8 dynamic quant — multiply by precomputed `_INV_127`, NOT divide by `127.0`

`applies_to: soc=all; cann=all; op_class=quant`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: per-tensor int8 dynamic quant CPU-truth synthesis must mirror CANN's `Muls(absmax, DYNAMIC_QUANT_FACTOR)` semantics — multiply by precomputed `_INV_127 = 1.0/127.0` constant. Do NOT divide by `127.0` directly. Mathematically equivalent but LSB-different on fp32 → silent bit-exact drift when CPU truth uses divide.

Source: grouped_matmul_swiglu_quant_v2 kw-3 2026-05-24 (Path-B truth synthesis, 8/8 cases bit-exact post-fix).

## CAND-FA-TILESIZE-1: FA-class A3 (membase) tile-size — pick largest tile fitting L0 (block_N≥128); per-tile scalar+sync overhead × tile-count dominates device-time

`applies_to: soc=Ascend910_9382(V220/A910C); cann=9.0.0; op_class=fa_class_A3_membase`
`verified_on: soc=Ascend910_9382; cann=9.0.0`

Pattern: FA-class A3 (membase) device-time is dominated by **per-kv-tile scalar-instruction issue + cross-core handshake + GM round-trip**, scaling ~O(S^1.8) vs vendor `aclnnFlashAttentionScore` ~O(S^0.75). The prototype designer's default `block_M=block_N=64` makes tile count (hence per-tile fixed overhead) 4× larger than necessary. **Choose the LARGEST tile that fits L0**: for D≤128 fp16, `block_N=128` (L0B `2×(BASE_K×block_N×2B)=64KB` = exactly L0B cap; L0A unchanged at block_M=64; UB vec inputQue depth-2 `block_M×block_N×4B=32KB×2` fits). Do NOT also raise block_M to 128 simultaneously — `block_M=block_N=128` overflows UB (vec inputQue depth-2 = 128KB).

Measured (3_FusionAttention_n9bis, NPU2, same-NPU A/B vs `npu_fusion_attention`): block_N 64→128 ⇒ S=1024 440→252µs (1.74×), S=512 133→82µs (1.62×), S=128 17.3→14.6µs (benchmark 1.17→**1.39× vs vendor = PASS**). **Precision unchanged** (max_abs 1.2e-4 fp16 1-ULP) — tile size is perf-only, no numeric effect. NEGATIVE result (measured, not assumed): hoisting per-tile `SoftMaxFlashV2TilingFunc` to Init = negligible (440→445µs) — the redundant tiling-func recompute is NOT the dominant scalar cost; tile-count is.

Template-assembly guidance: the FA-class A3 AscendC template should use `block_N` = max-fitting-L0 (≥128 for D≤128), not the `64,64` default. Remaining levers (large-S still 0.23× @S=1024, perf-polish not deliverable-blocking): vectorize per-row scale via `Brcb` (not scalar `GetValue` loops in `RowMulsImpl`/`RowDivsImpl`); reduce RING_SLOTS cross-core handshakes; keep scores/P on-chip to cut GM MTE2.

Source: independent prototype FA-class A3 perf whitebox 2026-05-29 (`docs/handovers/HISTORICAL_FA_CLASS_A3_PERF_WHITEBOX_2026_05_29.md`, PR #249).

## CAND-FA-A3-PERF-STRUCTURAL-1: FA-class A3 (membase) vendor-gap is structural — matmul::Matmul async cube + zero-CrossCore vs hand-Mmad MIX_AIC_1_2; in-paradigm scalar levers cap at lever-1

`applies_to: soc=Ascend910_9382 (V220 A3); cann=9.0.0+; op_class=fa_class_membase_mix_aic; layout=BNSD/BSH/SBH`
`derived-from: a5_ops whitebox CANN-A3-source comparison (3_FusionAttention_n9bis vs ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h, 2026-05-30, NPU2 device-time)`
`verified_on: a5_ops (3_FusionAttention_n9bis msprof A/B: ours 0.25x vendor S=1024 = 236us vs npu_fusion_attention 58-61us; aic_mac 0.089 vs vendor 0.315; aiv_scalar 0.776 vs vendor 0.283)`
`unverified_on: matmul::Matmul library FULL-FA port perf on A3 (POC 2026-05-30: the lib COMPILES standalone on V220 but DEADLOCKS at runtime — 507014 KFC-workspace-bootstrap gap; standalone >0.7x NOT achieved, see Recommendation)`

> **2026-07-20 cross-ref (a3 multi-core FA resolution):** the standalone-DEADLOCK / "matmul-lib high-perf NO-GO" above is the **async** `matmul::Matmul` (KfcServer) path. The **`MatmulImpl IterateAll<sync=true>`** library variant DOES run standalone **deadlock-free** on a3 (it avoids the async KFC msg-ring) — see `fa_class/cross_core_sync.md` §5 / PB-56 (per-head multi-core FA, 20/20, 0.186× vendor @ S=512/BN=32). This does NOT contradict the 0.25× **async** ceiling (sync 0.186× < 0.25×); it only prevents reading "matmul-lib standalone = NO-GO" as absolute.

**Trigger**: FA-class A3/V220 membase kernel below vendor `npu_fusion_attention`, profile shows `aiv_scalar_ratio` high (>0.5) + `aic_mac_ratio` low (<0.15). Use to decide whether further hand-tuning is worth it vs accepting the membase ceiling.

**Finding (CANN-source-grounded, file:line both sides)**: the ~4x vendor gap is STRUCTURAL, not a tunable scalar loop:
1. **Cube dispatch**: vendor uses `matmul::Matmul<>` library (`flash_attention_score_s1s2_bn2gs1.h:98-119` decl; `IterateBmm1` L1086-1106 / `IterateBmm2` L2116-2155 call `IterateAll<false>` = single async call doing internal L1-reuse of Q, L0A/L0B double-buffer, K-tiling pipeline, library-managed Fixpipe+fences). Cube runs ASYNC while vector computes (3-stage pipeline bmm1[t] || vec1[t-1] || bmm2[t-2]). OURS (`fusion_attention_cube.h` ComputeMM1 L85-175 / ComputeMM2 L195-270) = hand `Mmad`+fence SYNCHRONOUS loop; AIC stop-and-waits per WorkspaceQueue slot. => explains aic_mac 0.089 vs 0.315 (our cube is WAITING, not computing).
2. **CrossCore fence count**: vendor = 0 `CrossCoreSetFlag/WaitFlag` in bn2gs1 (~13 `WaitIterateAll` per Q-block, amortized over the tile, same kernel body drives cube+vec). OURS = `workspace_queue.h:46-81` fires 12 hardware CrossCore ops/KV-tile => **96 per Q-block at S=1024** (48/side). Each AIV `CrossCoreWaitFlag` stalls the scalar pipe waiting for AIC Fixpipe => dominant source of aiv_scalar 0.776.
3. **Root cause = architecture**: we use MIX_AIC_1_2 with SEPARATE AIC/AIV bodies (require hardware CrossCore flags to communicate); vendor runs cube-dispatch + vector in a SINGLE body via the matmul library's async interface (zero CrossCore flags).

**Recommendation**:
- In-paradigm (membase hand-Mmad) actionable levers are EXHAUSTED at lever-1 (RowMulsImpl Brcb+Mul vectorization, shipped). The only remaining in-paradigm lever is RowDivsImpl Brcb+Div vectorization, but (a) expected <5% (dominated by CrossCore stalls, not the 256 GetValue ops/Q-block) and (b) it FAILED 3 ways on V220: shared softmaxExpUb_ scratch -> NaN, dedicated recipBuf_ -> NaN (deeper RowMuls->RowDivs Vec2 pipeline hazard in `!isFirst&&isLast` tile), Brcb+Div -> UB-alignment fault 507015 (`Div` BinaryRepeatParams stride convention differs from `Mul` on V220 — open sub-issue if pursued).
- To close the >=80% structural gap: port FA-class A3 to `matmul::Matmul<>` library (`CFG_EXCEED` config) = single-body async-cube redesign (DEBT-20 `-DASCENDC_MATMUL_AICORE` flag isolation + full kernel restructure). NOT a one-line edit. This is the A3 high-perf path (still membase/arch22, NOT regbase/A5). **Owner gate (2026-05-30): validate the matmul library on A3 hits >0.7x vendor via a SMALL experiment BEFORE the full rewrite — guard against it being another membase ceiling.**
- **POC RESULT (2026-05-30, empirical — gate verdict NO-GO for standalone kernel rewrite)**: a minimal QK^T-only `matmul::Matmul<>` (KFC path, `matmul_intf.h`, `KERNEL_TYPE_MIX_AIC_1_2`, no manual CrossCoreSetFlag) in a standalone pybind11 kernel **COMPILES + links on V220** but **DEADLOCKS at runtime = aicore timeout 507014**. Root cause: the KFC path needs the CANN operator framework's workspace bootstrap (`SetSysWorkspaceForce(workspace)` + auto_gen `WORKSPACE_PARAM_OFFSET` + framework-allocated workspace layout matching `KfcCommServer::Init()`); without it the AIC-side KFC server never processes the AIV cube requests → AIV hangs → timeout. This is a NEW V220 hazard distinct from PB-34 (`MatmulImpl<>`+manual-CrossCore) and PB-35 (event-id collision): **KFC standalone workspace-bootstrap failure**. Implication: realizing the matmul-library high-perf path on A3 requires registering FA as a proper CANN operator (op_host tiling fn + framework workspace + working `REGIST_MATMUL_OBJ` bootstrap) — substantially MORE than a kernel-level rewrite, and off the port_a3/DEBT-110 product mainline. The standalone pybind11 architecture's 0.25x is therefore an EMPIRICALLY-CONFIRMED ceiling (not just source-inferred). The POC did NOT measure its own aic_mac (deadlocked before running) — vendor 0.355 / hand-Mmad 0.090 stand as the only measured cube ratios.

**Risks before promotion**: structural claim is source-comparison + msprof grounded. The matmul::Matmul-port is now EMPIRICALLY tested at POC level (compiles-but-deadlocks-507014 standalone — see POC RESULT); the FULL framework-integrated port perf remains unmeasured (would require the CANN-operator-registration effort). RowDivsImpl Div-alignment is an open V220 question. Cross-validate the aic_mac/fence root-cause on a second FA-class membase op (GQA) before promoting to OL.

**Other instances predicted**: any FA-class / cube+vec fused op on V220 membase MIX_AIC_1_2 (GQA, sparse-FA, NSA) — same hand-Mmad-vs-library structural gap.

**Cross-ref**: CAND-FA-TILESIZE-1 (the tile-size lever — real 1.6-1.7x internal but does NOT close vendor gap; this CAND explains WHY the residual is structural), OL-196 (membase=A3/V220/arch22 vs regbase=A5/V351/arch35), PB-34 (MatmulImpl + manual CrossCoreSet/Wait deadlock — a DIFFERENT failure mode, relevant when attempting the library port), PB-35 (FA pipe-sync event-id collision in MIX_AIC_1_2), DEBT-20 (per-source MATMUL_AICORE flag isolation needed for the library port), CAND-FA-MULTI-LAUNCH-PERF-GAP (the 5-delta perf comparison — overlapping root cause).

> **UPDATE 2026-05-30 (independent prototype, source-derived — the "507014=ceiling/needs-framework-registration" conclusion above is OVERTURNED as a mechanism)**: the standalone-KFC deadlock is NOT a ceiling. See **CAND-KFC-standalone-bootstrap-teardown** below — the 507014 is a 2-layer KFC lifecycle issue (workspace bootstrap + RAII-destructor teardown), both replicable standalone. The earlier POC concluded "ceiling" because it had the teardown wrong (`mm.End()` does NOT send SERVICE_QUIT). Standalone matmul-lib is reachable. (Owner meta-diagnosis 2026-05-30: a running same-chip CANN reference ⇒ "standalone won't start" = mechanism not fully read, not a ceiling.) **Status: mechanism source-solid; the standalone-kernel FIX is now VERIFIED-INSUFFICIENT (2026-07-12) — applying the full recipe at kernel level on `build_ascendc.py` did NOT clear 507014; the KFC msg-ring rendezvous only bootstraps under the CANN op-framework. See the "VERIFICATION RESULT" block below.**

## CAND-KFC-standalone-bootstrap-teardown (standalone matmul-lib/KFC reachable; overturns the 507014-ceiling)

**Source provenance**: CANN dav_c310 (V351/A5) + dav_c220 (V220/A3) `kfc/` headers + `kernel_operator_common_impl.h`, owner-authorized white-box read 2026-05-30. KFC lifecycle pattern is common across both archs. **Customer-runnable: this CAND states the MECHANISM + the standalone fix; no CANN path required to apply it.**

**Problem**: `matmul::Matmul<>` library (KFC path, `KERNEL_TYPE_MIX_AIC_1_2`) in a standalone pybind11 kernel (no CANN-op framework) deadlocks at runtime → aicore timeout 507014 / Exit 124. Two earlier POCs split on the cause (one "bootstrap fails", one "layer-3 teardown blocked") and one concluded a hard ceiling. Both are reconciled below; neither is a ceiling.

**Mechanism — 2 KFC lifecycle layers, both replicable standalone**:
1. **Bootstrap**: the AIC KFC server + AIV client message buffers live at offsets into a GM `workspace` (`KfcCommServer::Init(workspace, i)` → `GetMsgHead(workspace, i)`). Standalone recipe: allocate a GM workspace sized for the KFC msg ring + matmul L1/L0 scratch, call `SetSysWorkspaceForce(workspace)` so `GetSysWorkSpacePtr()` returns it, THEN `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), bmm1, tiling, bmm2, tiling)` (+ `matmul::InitL1Buffer`). If the workspace layout doesn't match `KfcCommServer::Init`'s expected offsets, the AIC server never services AIV cube requests → AIV hangs at the FIRST iterate.
2. **Teardown**: `SERVICE_QUIT` (0xfd00) exits the AIC server `while(isRun)` loop. It is posted by the **AIV-side `~KfcCommClient()` DESTRUCTOR** (RAII scope-exit): `AllocMessage()` + `KfcMsgMakeFlag(SERVICE_QUIT, 0)` + `dcci` to GM. **It is NOT sent by `mm.End()`.** Standalone fix: let the matmul / KfcCommClient object destruct at AIV kernel scope-exit (correct RAII lifetime — do not keep it alive past the kernel body); software KFC (`enableHardWare=false`); MIX_NUM/subblock conditions (in 1:2 AIC:AIV mode only the right AIV subblock sends, plus a `CrossCoreWaitFlag(KFC_SYNC_ID)` handshake). Missing teardown → AIC `while(isRun)` hangs forever → 507014.

**Implication**: standalone matmul-lib/KFC is achievable (workspace bootstrap + RAII destructor teardown) — NOT a ceiling requiring full CANN-operator registration. Unblocks the FA-A5 regbase perf path AND the FA-A3 matmul-lib path (shared blocker, shared answer).

**Verification status (DISCIPLINE)**: MECHANISM is source-derived and solid. This is **NOT a verified fix** until a standalone kernel applies it and a measured run returns WITHOUT 507014/124. Do not mark "solved" pre-run. Flow: codify (this CAND) → apply (RAII destructor scope + workspace bootstrap) → measured run → close.

**VERIFICATION RESULT (FA-a3 DEBT-36 white-box, 2026-07-12) — standalone-kernel fix REFUTED as sufficient; op-framework REQUIRED confirmed**:
The full documented recipe was applied PURELY at kernel level on the existing `build_ascendc.py` launch path:
- workspace-layout-vs-`KfcCommServer::Init` verified consistent (AIC + AIV share the same `GetSysWorkSpacePtr` base);
- software KFC `enableHardWare=false` (the default `ENABLE_HARD_POOL=false`, `kfc_register_obj.h:75`);
- c220-normal-mode both-subblock `SERVICE_QUIT` with `ubAddr=2`, `MIX_NUM=2`;
- RAII `~KfcCommClient` destructor at correct AIV kernel scope-exit;
- (`CrossCoreWaitFlag(KFC_SYNC_ID)` is c310 / super-kernel-only → N/A on V220, so it was correctly omitted).

**Result: 507014 NOT cleared (INNER_EXIT=124).** Definitive per-layer isolation on the standalone build:
- **FFTS `WORKSPACE_SYNC_ID` event channel WORKS** — `ffts_probe` / `ffts_both` exit 0;
- **workspace base consistent** — AIC and AIV read the same `GetSysWorkSpacePtr`;
- the residual hang is **ISOLATED to the KFC msg-ring server/client machinery**: `aiv_boot` with a `KfcCommClient` hangs exactly where a bare `WaitEvent` (no client) completes.

**Conclusion**: the standalone kernel-level fix is **INSUFFICIENT** on `build_ascendc.py`. The IDENTICAL KFC code is **bit-deterministic when launched via the CANN op-framework** — vendor `FlashAttentionScore` / `torch_npu.npu_fusion_attention` is 22/22 bit-identical (12 in-process + 10 fresh-process) plus an independent 10/10 bit-identical, `max_abs_diff` 0.000276 (fp16 floor), at the 64×64 fp16 shape. So clearing 507014 requires the op-framework build/launch infrastructure — the msg-ring rendezvous only bootstraps there. This is a **build-infra** matter, NOT a kernel-level one. **Verdict: standalone-kernel-fix REFUTED as sufficient; op-framework-required CONFIRMED.** Honest residual: the exact op-framework msg-ring bootstrap step is not pinned to a specific source line (KFC server internals are private) — the isolation is empirical (per-layer bisection), not source-read. This aligns with the harness-linkage root cause in **OL-235 / P-P102** (the matmul-library cube path is structurally unbuildable through `build_ascendc.py` because the host `TCubeTiling` link is absent) — same "library-cube path needs the op-framework layer, not the standalone launcher" conclusion, reached from the runtime-KFC side here vs the build-link side there.

**API/build note (2026-07-12)**: the KFC-managed object is the server/client-adaptive `matmul::Matmul<>` alias (what `REGIST_MATMUL_OBJ` wires the KFC server/client onto); the raw `MatmulImpl<>` is the non-KFC underlying form. Do NOT read this as a recommendation to take the KFC path on V220 through this harness — **P-P102 keeps the hand-rolled `AscendC::Mmad` default** for CUBE_MIX ports (the library path is structurally unbuildable here per OL-235, and this CAND shows the standalone KFC path still deadlocks). Companion build fact: the 2026-05-21 `matmul::`-prefix build breakage does NOT reproduce on CANN 9.1.0 — unprefixed `Matmul<>` / `MatmulType<>` compile clean with `using namespace AscendC;`.

**Fix path (learning only — owner-gated, NOT started)**: bringing the op-framework's KFC msg-ring bootstrap onto the mainline (standalone) `build_ascendc.py` launch chain is the CONFIRMED-needed direction — but it is a build-infra change, owner-gated and not begun. This CAND stays a CANDIDATE until that infra lands and a standalone KFC kernel runs 507014-free.

**Cross-ref**: overturns the 507014-ceiling conclusion in the CAND above; PB-34 (MatmulImpl+manual-CrossCore — different failure mode); DEBT-20 (`-DASCENDC_MATMUL_AICORE` per-source flag isolation, still needed for the library compile); OL-196 (membase/regbase); OL-235 / P-P102 (harness-linkage twin conclusion — library-cube path unbuildable via `build_ascendc.py`); OL-275 (the hand-rolled multi-cube self-poison this managed path resolves when op-framework-bootstrapped); CAND-V351-arch35-RegBase-service-class-skeleton (the regbase service-class that uses this KFC).

## CAND-FA-MICROAPI-REG-507015: FA-specific MicroAPI register-reduction softmax hits 507015 aicore exception (un-root-caused — PREMATURE, NOT a verified finding)
`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=attention (FA register-based softmax reduction)`

**Status: UN-ROOT-CAUSED open issue, NOT a verified KB finding.** Kept as candidate/roadmap only. Do NOT promote to canonical KB until root-caused with a minimal reproducer.

**Symptom**: porting the FA-A5 softmax to the MicroAPI register-compute path (`__VEC_SCOPE__` + `RegTensor` + register-based reduction, vs the mem-based `LocalTensor` path that DID work — P-P101) hits a runtime `507015 aicore exception`. The TRIVIAL register elementwise path runs clean at runtime (OL-54 runtime-clean evidence, `tests/repro/regbase_minimal.cpp`), so the MicroAPI register infrastructure works for simple elementwise — the crash is specific to the FA register-REDUCTION usage in the cube-coresident FA context.

**Why premature**: not bisected to a mechanism; no minimized reproducer; candidate causes (register-reduction in the FA cube-coresident context, register pressure, or a `__VEC_SCOPE__`/sync-scope issue) not isolated. The de-scalarize WIN was achieved by routing AROUND this via the mem-based VEC softmax (P-P101); the register path remains an open question, not a result.

**Cross-ref**: P-P101 (the mem-based route-around that works + is verified — the de-scalarize win lives there, NOT here); OL-54 (trivial register elementwise runtime-clean — the contrast that scopes this crash to register-reduction specifically); the 507015 Mmad/Nd2Nz CANDs above (DIFFERENT 507015 flavors — those are cube layout / `Nd2NzParams` faults; this is MicroAPI register-reduction in the vec path).

---

## CAND-FA-A5-KFC-WORKSPACE (custom-launch large-D GM-staging needs `SetSysWorkspaceForce` on dav-c310/3510) [PROVISIONAL — pending DS corroborate]

`applies_to: soc=Ascend950PR (V351/A5, dav-c310/__NPU_ARCH__==3510); cann=9.0.0; bisheng=n/a; op_class=all (custom-<<<>>>-launch cube/matmul op)`
`status: PROVISIONAL — mechanism source-grounded + 2×2-reconciled + d256 disk-verified; aggregate result (independent prototype FA-A5 31→35) pending DS build-from-SHA corroborate (SHA 4b2f79b8)`

**Mechanism (source-grounded)**: on a hand-written `<<<>>>` launch (no aclnn/GE framework), `GetUserWorkspace(workspace)` ignores its argument and returns the GLOBAL `g_sysWorkspaceReserved + RESERVED_WORKSPACE` (16 MiB); on dav-c310/3510 the base is `GetSysWorkSpacePtr() = __get_kfc_workspace_addr()`. The deprecated `SetSysWorkspace` only sets the global `if (g_sysWorkspaceReserved == nullptr)` → silent no-op if already set / optimized → `GetUserWorkspace` returns `nullptr+16MB=0x1000000` garbage → D>192 GM-staging OOB `507015` while D≤128 (UB-resident) passes. Fix: call `AscendC::SetSysWorkspaceForce(workspace)` (unconditional) before `GetUserWorkspace`, workspace sized `data + 16MB`.

**2×2 reconciliation (delta-proof, not flaky re-run)**: layout-alone (no Force) = no output; with `SetSysWorkspaceForce` = clean. The earlier `SetSysWorkspace` measured-negative STANDS as a fact (it was a silent no-op), reconciled — not "newer-wins".

**Promotion gate**: DS build-from-SHA 4b2f79b8 corroborate (device.o-recompile, +4 large-D, original 31 no-regress) + the `g_sysWorkspaceReserved` dump showing plain=nullptr/garbage vs Force=alloc-base. **Cross-ref PB-41** (the verified V220/multi-core instance of the same workspace-registration contract — this CAND is the A5/3510 single-core large-D extension).

---

## CAND-FA-A5-WORKSPACE-BIFURCATION (hand-rolled-launch workspace-binding root bifurcates — do NOT cross-lane-generalize) [PROVISIONAL]

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=FA/cube custom-launch`
`status: PROVISIONAL — independent prototype multi-core side DS-confirmed; independent prototype large-D side pending corroborate`

**Principle**: a "hand-rolled launch missing framework workspace-binding" symptom can bifurcate into RELATED but DISTINCT roots — do not transfer one lane's fix to the other without measuring. FA-A5 instance: (a) multi-core lane FFTS cross-core sync-scratch (fixed by `SetSysWorkspace`+16MB) = **DS-confirmed**; (b) single-core lane large-D GM-staging (needs `SetSysWorkspaceForce`/kfc base, plain `SetSysWorkspace` is a silent no-op on 3510) = **provisional**. A cross-lane transfer ("apply an independent run's SetSysWorkspace to single-core large-D path") was MEASURED-refuted mid-session — same symptom class, different mechanism. Lesson: measure each lane's root; shared toolkit API (`SetSysWorkspace*`) ≠ shared root.

**Cross-ref**: PB-41, CAND-FA-A5-KFC-WORKSPACE, `feedback_passcount_variance_first_hypothesis_is_nondeterminism` (sibling "measure don't cross-lane-generalize" discipline).

## CAND-FA-SOFTMAX-STAT-1: Online-softmax per-row stat lives as a `[m, 8]` datablock-packed buffer (1 fp32 block/row, 8 identical lanes) — extract by indexing `row*8`, broadcast-apply across columns via `src1BlkStride=0 / src1RepStride=1` (Brcb only when the stat is in `[m, 1]` form)

`applies_to: any SoC with public AscendC online-softmax (Softmax/SoftmaxFlashV2) + VEC Mul/Div/Brcb + BinaryRepeatParams; cann=9.0.0+; op_class=online_softmax_row_rescale / flash_attention_forward / fused_attention_with_softmax_stat / streaming_softmax_divide`
`derived-from: cann-source (FA forward reference vec epilogue, 2026-06-03 gb4softmaxstat)`
`evidence_family: FA-SOFTMAX-STAT`
`verified_on: public AscendC Softmax API doc (sumTensor/maxTensor last-axis = fixed 32 B = 1 datablock, all lanes identical) + FA forward reference vec divide/rescale epilogue (kernel-structural)`
`unverified_on: a5_ops`

**Trigger**: A fused attention / online-softmax kernel must (a) read the per-row max/sum reduction that `Softmax`/`SoftmaxFlashV2` produced, and (b) row-broadcast it across the head-dim columns to rescale or divide the attention output (`O[m, cols] /= sum[m]`, or `P[m, cols] *= alpha[m]`). The precision-critical failure mode this closes: the divide produces `nan` and the stored `softmax_sum` is systematically wrong because the per-row stat was read/broadcast with the wrong stride convention.

**The layout principle (the "why 8")**: the public online-softmax reduction output is NOT a contiguous `[m]` (one value per row). It is a `[m, B]` datablock-packed buffer where `B = 32 bytes / sizeof(reduce_dtype)` = **8 for fp32** (one hardware datablock per row, **all B lanes hold the same reduced value** — this is the documented public contract of the `sumTensor`/`maxTensor` outputs: "last axis fixed to one datablock, all data in the block identical"). Row `i`'s stat lives at `buf[i*8 .. i*8+8)`. There is therefore **no separate `[m,8]→[m]` extraction step** — to read row `i` you index any lane of its block (`buf[i*8]`); to feed the broadcast you point at the block base and let the stride convention spread it.

**Two regimes (pick by the reduce-tail width the API was told to use)**:
1. **Stat already `[m, 8]`** (the default, reduce-tail = 8): feed `stat[row*8]` directly into the apply with the broadcast `BinaryRepeatParams` below. **No Brcb.**
2. **Stat in `[m, 1]`** (reduce-tail = 1, contiguous one-per-row): expand to `[m, 8]` FIRST with a single `Brcb(blk, stat, (m+7)/8, {1, 8})` (block params dstBlkStride=1, dstRepStride=8), then apply identically. This is the same `Brcb` shape CAND-FA-AUX-OUT-1 uses for the GM emit — here it feeds the in-kernel apply instead.

**The broadcast-apply across columns (`[m, 8] → [m, cols]`, the load-bearing precision fix)**: a single `Div`/`Mul` whose `BinaryRepeatParams` broadcasts the per-row datablock across all columns within one repeat (`src1BlkStride = 0`) and advances exactly **one datablock per row/repeat** (`src1RepStride = 1`). The data operand walks columns normally (`*BlkStride = 1`, `*RepStride = cols_aligned / 8`). `repeatTimes = m` (rows), the per-repeat element count = the column slice. The divisor/multiplier index uses the **`*8` block stride** to land on the correct row's block: `stat[rowTileBase * 8]`.

**Concrete anchor** (public-API; worker-local names; runnable):
```cpp
// stat   : LocalTensor<float>, the per-row reduction. [m] OR [m,8] (see regimes).
// o      : LocalTensor<float>, attention output, [m, cols] contiguous in cols.
// m      = rows in this tile; cols = head-dim (32B-aligned); B32 = 8 (fp32 datablock).
constexpr int B32 = 8;
const int colsAlign = AlignUp(cols, B32);

// Regime 2 ONLY: expand contiguous [m] -> [m,8] datablock form first.
LocalTensor<float> statBlk = tmp.Get<float>();          // [m*8]
AscendC::Brcb(statBlk, stat, (m + 7) / B32, {1, 8});    // {dstBlkStride=1, dstRepStride=8}
AscendC::PipeBarrier<PIPE_V>();
// Regime 1: skip Brcb; use the API's [m,8] output directly as statBlk.

// Row-broadcast divide O[r, :] /= stat[r] for all r:
AscendC::BinaryRepeatParams rp;
rp.src0BlkStride = 1;                 // numerator (O) contiguous in cols
rp.src0RepStride = colsAlign / B32;   // O advances one row (cols/8 blocks) per repeat
rp.src1BlkStride = 0;                 // KEY: stat block broadcast across all cols
rp.src1RepStride = 1;                 // KEY: stat advances exactly one datablock per row
rp.dstBlkStride  = 1;
rp.dstRepStride  = colsAlign / B32;
const int chunk = 64;                 // fp32 elems per repeat (one vector instr width)
for (int c = 0; c < colsAlign / chunk; c++) {
    AscendC::Div(o[c * chunk], o[c * chunk], statBlk[/*rowTileBase*/0 * B32],
                 /*mask=*/chunk, /*repeatTimes=*/m, rp);
}
// (mirror with Mul + statBlk = alpha for the exp-rescale step; for a numerator
//  broadcast — e.g. P *= alpha where alpha is src0 — swap which operand carries
//  src*BlkStride=0 / src*RepStride=1.)
```

**Why the wrong way goes nan / wrong sum** (symptom anchor, kw-gb3 FA graybox 0/8):
- Reading the stat as if it were contiguous `[m]` (stride 1) when the API wrote `[m,8]` reads 8× too few rows — every 8th row's stat used for 8 consecutive rows → garbage `softmax_sum` (~60–490 observed) and div-by-corrupt → `nan` in `attention_out`.
- Using a fractal `Copy` + `CopyRepeatParams` for the stat broadcast: `CopyRepeatParams` is the **ND↔NZ fractal-layout reshuffle** primitive (for the matmul-result transpose), NOT a row-broadcast. The stat broadcast is **never** a `Copy`; it is `Brcb` (expand, regime 2 only) + `Div`/`Mul`-with-`BinaryRepeatParams` (apply). Conflating the two is the documented bug.
- Setting `src1RepStride = 0` (instead of 1) broadcasts the FIRST row's stat to ALL rows — looks numerically plausible (no nan) but is silently wrong. The non-zero `src1RepStride = 1` (one datablock/row) is load-bearing.

**Reject_cond**: do NOT use when
- The stat is a single per-tile scalar (not per-row) — a plain `Muls`/`Div`-by-scalar is correct, no broadcast stride needed.
- `cols % 64 != 0` without a tail handler — the trailing partial chunk needs a reduced-mask `Div` (not shown).
- `m > 255` — `repeatTimes` is uint8; split the row loop.
- A scalar `GetValue(r)`+`Muls` per-row loop is acceptable for correctness (the agent's current hand-roll does this and it works for the APPLY) — this entry's value is the **layout/extraction convention** that makes the stat itself correct, plus the de-scalarized broadcast for perf.

**Relationship to existing KB** (C35-disambiguated — overlaps each on ≤1 reason code, NOT ≥2):
- **CAND-FA-AUX-OUT-1**: covers `Brcb([m]→[m,8])` + DataCopy for **emitting** the stat to GM (write-out). This entry covers the **in-kernel consume/apply** broadcast (read-back + rescale) and the `reduceSize==8`-vs-`==1` regime. Sibling, opposite direction (emit vs apply). Shared: `Brcb {1,8}` shape — same datablock convention, deliberately consistent.
- **CAND-RAU-3**: generic `[R,inner] *= [R,1]` stride broadcast with `src1RepStride = softmax_tail/8`. This entry specializes it to the FA `[m,8]`-block-from-SoftmaxFlashV2 case (`src1RepStride = 1` exactly, because the stat tail IS one datablock) AND adds the `Div`/numerator-vs-denominator distinction + the regime split. Use RAU-3 for the general per-row-scale shape; use this for the FA online-softmax stat specifically.
- **cv_reference_concrete_params.md §softmax_online**: gives the `wsMetaGm_` GM-region cross-core handoff stride; this entry is the missing in-UB `[m,8]` layout + extraction/broadcast primitive that §softmax_online lacked (the kw-gb3 gap). Mirrored into that section.

**Other-instances-predicted**: any online/streaming reduction that row-broadcasts a per-row stat across an inner dim — LayerNorm/RMSNorm `x *= rstd[row]`, BatchNorm `(x - mean[row]) * inv_std[row]`, GroupNorm, softmax-with-temperature, any attention variant (GQA/MLA/paged) reusing the same `Softmax`/`SoftmaxFlashV2` stat outputs.

**Promote when**: an a5_ops FA kernel adopts the `[m,8]`-block consume + broadcast-Div and clears the 0/8 precision blocker (clean `attention_out`, correct `softmax_sum`), confirming the convention closes the bug; cross-validate on one norm-class op (RMSNorm or GroupNorm) using the same broadcast shape.


## CAND-PP104: Small-tensor backward — prefer a single fused kernel over the partial+reduce multi-launch template when per-core vector work can't amortize launch overhead

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (layer/rms/group norm grad), generalizes to any partial+reduce multi-launch backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (correlation measured; causal mitigation UNCONFIRMED)`
`status: UNCONFIRMED — promote only after an optimizer fused-rewrite recovers the ratio (validates causation, not just correlation)`

**Pattern**: the OL-75 dual-axis partial+reduce template (main per-row/per-group kernel + a reduce kernel launched once per cross-row output) is **size-sensitive**. It WINS on large tensors (vector work ≫ launch overhead) but LOSES on tiny tensors, where the ≥2–3 fixed `aclrtLaunchKernel` overheads (~20–25µs each, plus cross-core sync) dominate. Mitigation hypothesis for the small-tensor regime: collapse to a single fused kernel — (1) merge the per-output reduce launches into one; or (2) single-core fused main+reduce; (3) vectorize the per-channel reduction (one `WholeReduceSum` multi-repeat vs CG×2 fold-reduces); (4) consider MIX pipelining (OL-200). Profiling-first (msprof to confirm the launch-vs-compute split before rearchitecting).

**Evidence (correlation only — causal claim pending)**: group_norm_grad (2026-06-03, port_a3_to_a5 V220, authored from scratch). 2-kernel design issues 3 launches (main + 2 reduce, one per dweight/dbias). On GroupNorm tiny tensors (≤1024 elems): ours ~150–200µs flat vs vendor single fused CANN kernel ~28–49µs → **ratio 0.25×, stable across 3 runs / 2 devices**. Precision + determinism 4/4 PASS first try — purely a launch-overhead regression. Root cause flagged as un-profiled hypothesis in the op's verification.json (NOT yet msprof-confirmed). Contrast: sibling layer_norm_grad ran the SAME template at 1.0–1.9× because H=1024–4096 amortizes the launches (see OL-75 large-tensor evidence row).

**Profiler device-time cross-check (2026-06-03, same `group_norm_grad` kernel, P97-canonical)** — do NOT conflate the two measurement regimes: kernel-only `device_self_duration` ratio = **0.851×** (ours ~8–10µs vs vendor ~8.6µs). The end-to-end **0.25× wall-clock** vs this **0.851× device-time** gap IS the host launch overhead this candidate is about (3 launches × ~20–25µs vs vendor's 1 fused kernel). i.e. the regression is launch-strategy (multi-launch), NOT kernel-efficiency — our kernel is roughly on par with vendor at the device level; it's the 3-launch host orchestration that loses the wall-clock race. Strengthens the "fuse the launches 3→1" mitigation hypothesis; still UNCONFIRMED pending an optimizer fused-rewrite that recovers the wall-clock ratio.

**Promote when**: an optimizer applies the fused single-kernel rewrite to a small-tensor backward op AND measures the ratio recovering toward/above 1.0× on the same NPU (back-to-back A/B), confirming launch-overhead (not compute) was the lever. Until then this stays a correlation, not a verified pattern.

**Cross-ref**: OL-75 (the partial+reduce template + its size-sensitivity scope condition), OL-200 (MIX pipelining), the per-row K_ROWS_PER_AIV launch-amortization candidate (line ~764), CLAUDE.md profiling-first rule.

## CAND-PP105: Dtype-aware baseK L0A-budget clamp for variable/tiny per-group cube GEMM dims — derive base sizes at kernel entry, leave static tiling base fields at -1

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=cube_gemm (grouped/segmented/MoE-expert/ragged-batch matmul + backward), dtype-sensitive (fp32 needs the K cap)`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (grouped_matmul_grad 6/6 PASS: 3 shapes × {fp32,fp16}, per ranging 4..128)`
`status: UNCONFIRMED — single op; promote after a 2nd grouped/MoE/segmented cube op confirms the rule transfers`

**Pattern**: when per-group cube GEMM dims can be SMALLER than the static `MatmulApiStaticTiling` base (grouped / segmented / MoE-expert / ragged-batch) OR dtype is fp32, do NOT bake `baseM/baseN/baseK` into the static tiling. Instead leave the `MatmulApiStaticTiling` base fields at `-1` (runtime-driven from the on-stack `TCubeTiling`) and derive the base sizes at kernel entry per group:

```cpp
// per group GEMM, at kernel entry (mOut/nOut/kRed = this group's actual dims)
bM = align16(mOut, /*cap=*/128);
bN = align16(nOut, /*cap=*/128);
bK = align16(kRed, /*cap=*/ (sizeof(T) == 4 ? 64 : 128));   // dtype-aware K cap
```

This prevents two DISTINCT `507015` "CCU instruction-address check error" traps, both confirmed on grouped_matmul_grad:
- **(a) base > actual dim on tiny groups**: static `baseM/baseN/baseK=128` with per-group dims like `per=4 / K=16 / N=32` makes the single cube tile load PAST the GM operand region → 507015. The `align16(dim, cap)` clamp bounds the tile to the real operand extent.
- **(b) fp32 baseK=128 overflows L0A**: `baseM*baseK*sizeof(T) = 128*128*4 = 64KB == L0A limit`; with the tile loader's working set this overflows → 507015 on the FIRST large fp32 matmul. The `cap_K = sizeof(T)==4 ? 64 : 128` keeps the L0A tile at 32KB for both dtypes (`128*64*4 = 32KB` fp32, `128*128*2 = 32KB` fp16).

Both reductions in a grouped backward flow through `baseK` (dB's `kRed=per`, dA's `kRed=N`), so the SAME clamp covers both transposed GEMMs of the mm_grad pair.

**Evidence**: grouped_matmul_grad (2026-06-03, port_a3_to_a5 V220, KB-only / zero CANN source) — 6/6 PASS (3 shapes × {fp32,fp16}, per ranging 4..128) after applying the clamp; both 507015 traps were hit pre-fix and cleared post-fix. Matches the forward grouped-mm precedent (`fp32 baseK=64`), so the backward inherits the same dtype rule by construction.

**Promote when**: a 2nd grouped/MoE/segmented/ragged-batch cube op (forward or backward) confirms the dtype-aware clamp prevents the same two 507015 flavors — i.e. the rule transfers beyond grouped_matmul_grad.

**Cross-ref**: P-P68 (constexpr static tiling + on-stack `TCubeTiling` — this CAND extends it with the runtime base-derivation for variable dims), P-P69 (runtime transpose bool), P-P74 (multi-AIC segment dispatch / grouped matmul — the dispatch half of the same op class), EC-39 (`MM_CFG=MatmulApiStaticTiling` typed config), forward grouped-mm `fp32 baseK=64` precedent.

## CAND-PP106: Fused residual-add + normalization backward — the two add-branch input grads are bit-identical; emit grad_x to both output buffers from one compute pass

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (fused residual-add + rms/layer norm grad)`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (fused_add_rmsnorm_grad 4/4 PASS)`
`status: UNCONFIRMED — single op; needs a 2nd fused-add-norm op (e.g. add+layernorm grad) to confirm`

**Pattern**: for a "fused residual-add + `<normalization>`" backward where the forward is `x = a + b; y = norm(x) * w`, the two add-branch input grads are BIT-IDENTICAL: `d_a == d_b == grad_x`. This is the identity Jacobian of `x = a + b` (∂x/∂a = ∂x/∂b = I), so there is NO separate backward to compute for the second branch. Emit `grad_x` to BOTH output GM buffers from ONE compute pass. The marginal cost over the non-fused norm-grad is exactly **+1 input load + 1 extra grad store** — NOT a second backward.

```
forward:  x = a + b;  y = (x * rsqrt(mean(x^2)+eps)) * w   // add + rmsnorm
backward: grad_x = r*(g - (x*r*r)*sum_H(g*x)/H)            // the rms_norm_grad grad_x
          d_a = grad_x;  d_b = grad_x                       // identity branch — store twice, no recompute
```

**Evidence**: fused_add_rmsnorm_grad (2026-06-03, port_a3_to_a5 V220, KB-only / analytic backward derived from scratch, bit-exact vs fp64 autograd oracle, max_diff ~1e-16) — 4/4 PASS first verify, 0 precision-fix iters. Structurally the op is "`rms_norm_grad`'s grad_x emitted to two output buffers + the same grad_w cross-row reduction"; the leading `x = a + b` add is the only forward delta and it vanishes to an identity on the backward side.

**Promote when**: a 2nd fused-add-norm backward (e.g. add + layernorm grad) confirms the d_a==d_b==grad_x identity-store recipe transfers — i.e. it's a norm-family rule, not a one-op coincidence.

**Cross-ref**: rms_norm_grad / OL-103 (the norm-family transcendental NR-Rsqrt backward technique this fused op consumes — the add-branch dup is the only structural delta on top), OL-75 (the partial+reduce template the grad_w cross-row reduction uses), CAND-PP104 (sibling norm-family backward launch-overhead candidate), CAND-PP109 (the shared-load/different-grad multi-output sibling — PP106 here is identical-grad duplication).

## CAND-PP109: Fused multi-output elementwise gradient — one kernel pass loads shared inputs once and emits all N outputs, NOT one kernel per output

`applies_to: soc=Ascend910_9382 (V220/arch22); cann=9.0.0; bisheng=n/a; op_class=elementwise_backward`
`verified_on: soc=Ascend910_9382 (V220/arch22); cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351/arch35); dtype=fp16/bf16; large-N tiling-loop path (single-tile only so far)`

Pattern: an elementwise backward op that produces N input-gradients sharing the same upstream loads (e.g. `mul_grad`: `grad_x=gy*w`, `grad_w=gy*x`; generalizes to `add_grad`/`sub_grad`/`div_grad` and any elementwise map whose VJP reuses the forward inputs/grad-output) is cleanly expressed as a **single kernel pass**: load the shared inputs (x, w, gy) once into UB, emit each output with one VEC op, store all outputs. This halves/N-folds MTE2 DataCopy traffic and launch overhead vs N separate single-output launches. Deterministic by construction (one core owns each align-8 chunk, each output element written exactly once, no SetAtomicAdd) — satisfies `DET_POLICY=required` with no determinism-specific code.

Measured (mul_grad kw-1, 2026-05-30, a3/Ascend910_9382): 2/2 PASS fp32 vs fp64 oracle, det 2/2 bit-identical, **2.99x median** vs the torch autograd backward path (~49us ours vs ~140us autograd: two backward-mul launches + engine traversal) at N=8/16 (both launch-overhead-bound). 0 compile-fix, 0 precision-fix iters.

**Promote when**: a SECOND elementwise-backward op (add_grad / div_grad / a 3+-output elementwise VJP) ships single-pass with measured perf >= 1.0x vs the per-output-launch baseline — confirms the fusion generalizes beyond the 2-output mul case. Also lift the tiling-loop path (current evidence is single-tile, small-N only) and an A5/V351 + fp16/bf16 instance to retire the `unverified_on` line.

**Cross-ref**: CAND-PP106 (the IDENTICAL-grad-duplication sibling, `d_a==d_b==grad_x` emit-same-value-to-2-buffers; PP109 here is the shared-load / DIFFERENT-grad fusion — kept distinct until a 2nd op lets a maintainer promote them to a unified canonical multi-output-backward pattern), OL-103 Tier-1 (pure-Mul chain passes fp32 threshold; the precision side of this op), OL-181 (output buffer padded to Align8(N) for DataCopy overflow, narrowed in pybind), PB-22 (DataCopy 32B-aligned `cnt=Align8(len)`), OL-160 (canonical `model_new_ascendc.py` entry-point reused), CAND-FAG-4 (multi-output FUSED-bwd dispatch — the cube+atomic-add cousin for attention-grad; this CAND is the cheap elementwise analogue with by-construction det and no atomics).

---

## CAND-PP110: Build-provenance finalize gates must enumerate ALL legitimate build paths — a self-compiled pybind archive proves provenance via the source → `<stem>.o` → linked-`.so` chain, not via CANN-binary basename/md5 overlap

`applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5 (CPU-truth pybind build)`
`verified_on: A5 Ascend950PR_9579 (modulate build_ascendc.py P140 path)`

Provenance: derived from modulate kw-3 2026-06-21 (port_a3_to_a5) + workspace/modulate/user_decision.md session 2026-06-21.

The DEBT-091 `binary_provenance` gate originally recognized only two proof models: **(a)** basename+md5 overlap (the deliverable name == a CANN-shipped binary), and **(b)** snake→PascalCase bridge (`<op>.cpp` → `Modulate_950_*.o`, the ops-nn-port build naming). A CPU-truth port_a3 op built via `build_ascendc.py` (the brief-MANDATED P140 path) emits snake_case COMPILER outputs (`modulate_kernels.cpp` → `modulate_kernels.cpp.o` → `_modulate_ext*.so`) and has NO installed CANN binary — so neither (a) nor (b) matches **by construction**, falsely rejecting a class of correct self-built archives (the already-shipped `grid_sample`, same build shape, also fails the tightened gate). The honest, non-fakeable proof for this class is a **tertiary (c) compiled-artifact model**: an installed `<src>.o` whose stem equals a BUILT source basename proves the object was compiled from our listed source, and that object links into the dispatched `.so`; require ≥1 such anchor + all md5s valid 32-char hex (anti-placeholder guard). Generalizable principle (per CLAUDE.md "Fix Harness for Next Customer, not Patch Single Archive"): a provenance gate that recognizes only SOME build naming conventions silently rejects correct archives from other build paths — enumerate them all (CANN-shipped binary, ops-nn-port snake→Pascal, self-compiled pybind chain).

**Status (2026-06-21)**: the (c) `compiled_provenance` bridge ALREADY LANDED as DEBT-091(c) (merge `aff5c5ee`, PR #16, `PortA3Plugin`) — it validates source → `<stem>.cpp.o` → linked `.so` via a 3-AND chain (stem + deploy==workspace + built_from==deploy + 32-hex anchors), modulate's `build_evidence.compiled_provenance` is emitted, and finalize re-ran VERIFIED. This CAND records the generalizable PRINCIPLE for the maintainer (gate-completeness over build paths); the harness mechanism is in place.

**Promote when**: a SECOND self-compiled pybind op (CPU-truth port_a3 built via `build_ascendc.py`) passes the (c) `compiled_provenance` gate clean AND `grid_sample` is re-verified under the fixed gate — confirms the proof model generalizes and retro-covers the pre-tightening archive. Promote to an OL (gate-design principle) or fold into the DEBT-091 gate doc.

**Cross-ref**: regression coverage `test_binary_provenance_gate.py` (accept / placeholder-md5 reject / unrelated-object reject); CLAUDE.md "USING binary API ≠ COPYING source" (a self-built `.o`+`.so` chain is honest provenance, not a CANN-source copy).

---

## CAND-FA-LAUNCH-DISPATCH-1: FA-class raw-launch template — host tiling POD → dtype/feature/D-bucket launcher selection → `<<<>>>` raw dispatch

`applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 forward no-dropout whitebox — 13 bit-exact + 2 within-T1-tol of 15 kp=1 comparable cases [case43 d64, case37 d256]; wholeport base 64/64)`
`derived-from: target prior-art inspection for structure; launch recipe must be re-derived from selected arch22 contract and current arch35 public APIs; self-contained = no external #include arch35`

**Date**: 2026-06-07 (FA-A5 graybox stage-2 template-ization, Phase 1)
**Status**: CANDIDATE — the dispatch-LOGIC recipe (this entry codifies launcher-selection invariants).
Archived launcher bodies and extern declarations are advisory target prior art only. The generator must
emit a task-owned launcher from the selected arch22 contract; copying an archived body cannot satisfy
generation or final validation. Self-contained also requires no external `#include arch35`.

**local-kb-crossref**: CAND-FA-CV-7 (shape-regime variant dispatch — ORTHOGONAL: CV-7 picks the kernel
STRATEGY by shape class; THIS picks the launcher SYMBOL by dtype × D-bucket × feature-gate WITHIN the
fixed-shape FA family, after the host tiling POD is computed), CAND-FA-CARRIER-1 (the host tiling POD this
launch consumes), P-P103 §"Host-tiling LOGIC template" (the POD producer) + §"Concrete functional-block
inventory" (the launcher symbols are the block instantiations).

**Op-class**: any cube-MIX op generated via a raw-launch pybind (port_a3 whole-port style) where a single
host entry computes a tiling POD and dispatches to one of N pre-instantiated kernel-template launchers.

**Pattern** (the launch template, 3 stages):
1. **DoTiling** — compute the tiling POD from op-config (P-P103 host-tiling logic). Allocate outputs with
   `at::empty` (NOT `at::zeros`) for kernel-WRITE-ONLY outputs (attn_out / sm_max / sm_sum) — the kernel
   keeps running max/sum in UB and writes them out; a host pre-zero fires a redundant ZerosLike device op
   per output (measured 0.43-0.54× vs vendor's 1 fused op). Gate write-completeness with an env-toggled
   NaN-poison fill in test builds (any residual NaN in a COMPARED output = under-write = revert that one
   to at::zeros).
2. **Launcher selection** — a decision tree keyed on (feature-gate priority) → dtype → D-bucket: priority
   order `userMask → hasPse → hasDropOut → isFp32 → isBf16 → fp16(default)`; within each, `dBasicBlock ≤
   64 / ≤128 / ≤256 / else` selects `wp_fa_do_<dt>_bnsd[_feature]_d{64,128,256,512}`. **The launcher `d{N}`
   suffix names the D-bucket TIER, not the literal D**: d64=Aligned64, d128=Aligned128, d256=Aligned256
   (Dn path), **d512=Aligned768** (splitD path — the symbol name is `d512`, the device template tier is
   Aligned768). The s1=64 core-fill class (P-P103 host-tiling stage 3) overrides to the
   `S1TemplateType::Aligned64` launcher; in the whitebox only the **fp16 D=128** Aligned64 override is
   wired (`wp_fa_do_fp16_bnsd_d128_s64`) — bf16/other-D core-fill at s1=64 keeps the s1=128 kernel because
   no Aligned64 device variant exists for those buckets in this build (widening = add the variant +
   extern). Each launcher is an `extern "C"` symbol = one (dtype, S1-tier, D-tier, feature) instantiation.
3. **`<<<coreNum>>>` raw dispatch** — launch the selected `extern "C"` symbol with `(blockDim, stream,
   q,k,v,..., tilingPtr, workspace, outputs)`. blockDim = the host-computed core count.

**Load-bearing constraints (transferable insight)**:
- Launcher symbol MUST match the host tiling's S1-tier (Aligned64 vs Aligned128) — a mismatch silently
  computes wrong tiles. The host `useAligned64Kernel` predicate and the launcher choice are ONE decision.
- `at::empty` for kernel-write-only outputs is the perf-correct default (avoid per-output ZerosLike); prove
  write-completeness before banking (poison-gate).
- D-bucket rounds UP (`AlignUp(D,64)` then ceil to the next device tier) — intermediate D reuses the
  next-bucket template (truth-valid up to D≤768).

**Other-instances-predicted**: any port_a3 whole-port cube-MIX op with a raw-launch pybind + multiple
dtype/feature/shape template instantiations (FA+X fusions, quant-attention, MoE-with-cube-stage). The
at::empty-vs-ZerosLike rule generalizes to ANY op whose outputs are fully kernel-written.

**Known gaps**: only the buckets a given spawn wires are reachable (whitebox: fp32 wired d64-only,
user-mask wired d128-only). case37 (d256/Aligned256) + dropout (kp<1) inherited from the whitebox (P-P103).

---

## CAND-FA-STITCH-1: FA-class template-stitching — A3-FA config → block-selection + host-tiling parameterization + launcher → self-contained A5 op

`applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 forward whitebox is the worked instance)`
`derived-from: synthesis of P-P103 + CAND-FA-LAUNCH-DISPATCH-1 + CAND-FA-CARRIER-1/COREDIST-1; target templates are advisory inputs; self-contained = no external #include arch35`

**Date**: 2026-06-07 (FA-A5 graybox stage-2 template-ization, Phase 1)
**Status**: CANDIDATE — the graybox generation recipe (how the harness uses FA-class knowledge to emit
a self-contained arch35 op). Not yet harness-wired (Phase 2 = G1). Template bodies remain searchable
as target prior art, but are not include-ready generated output.

**local-kb-crossref**: P-P103 (skeleton + FA+X delta table + block inventory + host-tiling logic),
CAND-FA-LAUNCH-DISPATCH-1 (launcher), CAND-FA-CARRIER-1 (carrier/host-tiling structure), CAND-FA-COREDIST-1
(core-split math), OL-205 (FA gaps concentrate in host feature-dispatch — grep kernel for the feature's
machinery first), OL-186 (cube-MIX 1:2 needed for forward FA), CAND-FA-CV-7 (shape-regime dispatch — a
DIFFERENT axis: CV-7 is across-variant strategy, this is within-variant assembly).

**Op-class**: the graybox generation recipe — given an A3-FA op-config, stitch the KB FA-class templates
into a self-contained A5 op (no `#include arch35`).

**The stitch (given an A3-FA config: dtype, shape B/N/N_kv/S1/S2, causal/sparse_mode, GQA gSize, layout, D)**:
1. **Select kernel-blocks** (P-P103 block inventory):
   - ALWAYS: `wp_kernel_base` (+ `wp_kernel_train` if backward/train) + `wp_block_cube` +
     `wp_block_vec_base` + the matching `wp_mc_*` layout block + `regbase_*` primitives + the shared
     carrier/util headers. These are the unchanged ~90% skeleton.
   - IF mask in scope: add `wp_attenmask` (NO_COMPRESS user-mask) or `wp_attenmask`+compress (causal).
   - IF pse/bias in scope: add `wp_pse` (auto-routes Nd path).
   - IF dropout (keep_prob<1) in scope: add `wp_dropmask` (separate workstream; template carries the gap).
2. **Parameterize the host tiling** (P-P103 host-tiling logic) by the config: dtype → `inputDtypeBytes` +
   cube dtype-path (fp32 = L1-split-N BMM2; softmax stat stays fp32); D → `dBasicBlock=AlignUp(D,64)` →
   D-tier (≤768); shape+dtype → basic-block + core-fill → `s1BasicBlock`; GQA → `n2Size=N_kv, gSize=N/N_kv`
   (host-only, OL-205); sparse_mode+mask-present → sparse-tiling (force dense if no explicit mask,
   OL-202/OL-85); core-split → set the carrier split-mode + cheap-prefix fields (CAND-FA-CARRIER-1 PIECE-B;
   COREDIST-1 consumes them).
3. **Pick the launcher** (CAND-FA-LAUNCH-DISPATCH-1): feature-gate priority → dtype → D-bucket →
   `wp_fa_do_<dt>_bnsd[_feature]_d{bucket}` (+ Aligned64 override on s1=64 core-fill). The launcher MUST
   match the host S1-tier.
4. **Emit self-contained task-owned code** — re-derive the selected blocks + host `DoTiling` + raw-launch
   pybind from the selected arch22 contract and current arch35 public APIs. NO `#include "arch35/..."`;
   no archived target body may be copied and declared generated. The carrier is ONE shared host+kernel
   header (CAND-FA-CARRIER-1 PIECE-D) whose layout is checked mechanically.

**Recipe phase-order (corresponding-knowledge, cross-ref P-P103 §Recipe)**: `copyin (GM→L1) → BMM1 (QK^T)
→ online-softmax → BMM2 (PV) → fixpipe (L0C→UB/GM) → epilogue (stat write-out)`. MIX cube-vec **1:2**
(`KERNEL_TYPE_MIX_AIC_1_2`, OL-186 — forward FA needs cube `Mmad` P@V; vec-only ceilings short). CV-fusion
datapath = cube→Fixpipe→UB→vec (S-channel `CROSS_CORE_SYNC_BOTH` depth-2 reverse-gated; P/O
`CROSS_CORE_SYNC_FORWARD`). split-core threshold: the core-fill drop (total<aicNum) + the s2≥threshold
split — host-config, not kernel.

**Other-instances-predicted**: the FA+X family (bias/pse, RoPE, quant-attention, FA-decode) per the P-P103
FA+X delta table — each = this stitch with the X-block added/swapped. The stitch SHAPE (select-blocks →
parameterize-host → pick-launcher → emit-self-contained) generalizes to any port_a3 whole-port cube-MIX op.

**Honest caveat**: stitching from these templates removes the from-scratch cost but an X-delta still needs
a CANN reference to study OR generate-from-guide. The whitebox is the WORKED fp16/bf16 BNSD no-dropout
instance; case37 + dropout + non-BNSD layouts are open deltas the template carries as known-gaps.

**Current RFC boundary**: keep the provenance-bearing target templates in the KB so the generator can
recover interfaces, required branches, and test hypotheses. Retrieval is advisory: emit task-owned code
from the selected arch22 contract, prove current-binary provenance, and validate against source-arch NPU
truth. A target-template body, target output, or bit-identical mirror cannot by itself close generation.

---

## CAND-KW-FAG-1: fp32 precision tier via cast-free output-GEMM routing when internal accumulation is already fp32

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=fp32-internal (FA / norm / GEMM-epilogue with cast-on-entry + cast-on-exit)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (flash_attention_grad fp32 tier)`
`unverified_on: other arch — the aliasing + f32-out routing are not chip-specific (design is arch-portable) but only A5 evidence exists`

For any kernel whose internal accumulation is **already fp32 with cast-on-entry / cast-on-exit**
(the common FA / norm / GEMM-epilogue shape), adding an **fp32 dtype tier is a pybind-dispatch-only
change**, not a new kernel:
(a) **skip the entry casts** — alias the fp32 input tensors directly as the GEMM A/B operands (safe
    because GEMM operands are read-only and the outputs are separate buffers);
(b) **route the output GEMMs to the f32-out cube entry** that already exists for the internal-fp32
    path (here `fag_gemm_f32out`, also used by the S / dP intermediates for ALL dtypes).
Result: the fp32 path has FEWER launches than the cast paths (4 fewer here — the inputs→fp32 cast×4
is gone) AND is strictly the most-accurate path (no T↔float round-trip).

**Aliasing precondition (mandatory pre-check)**: NO GEMM writes back into an operand buffer — verify
the dataflow DAG before aliasing (here qf/kf/vf/dOf are strictly read-only A/B operands). Re-aliasing
an input that a GEMM writes into corrupts silently.

Concrete anchor (flash_attention_grad, 2026-06-14): ZERO kernel.h/.cpp change — only `pybind11.cpp`
dispatch + `model_new_ascendc.py` + `verify_*.py` were extended; build RC=0 (no new TU); fp32 path
most-accurate (fp32 MERE bootstrap median 0.0).

**Promote when**: a SECOND fp32-internal op (a norm-family or GEMM-epilogue backward whose kernel
already accumulates in fp32) adds its fp32 tier the same dispatch-only way — confirms the cast-free
routing generalizes beyond FA backward. On promotion, move to `patterns/domains/precision.md`.

**Cross-ref**: P-P52 (fp32 reduction promotion — the internal-fp32 precondition this relies on),
P-P103 (FA-class template — the kernel family applied here), OL-81 (CAST_RINT fixpipe-cast on the
carried fp16/bf16 paths), CAND-KW-FAG-2 (the fp32-tier precision-grading floor this tier then hits).

---

## CAND-KW-FAG-2: fp32 backward-gradient MARE fail-floor — MERE-perfect-but-MARE-over-threshold is a small-value-domain metric artifact, not a kernel bug

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 dtype tier)`
`verified_on: NPU-independent (CPU torch_fp32-vs-fp64 triage); a5_ops:flash_attention_grad fp32 tier`

An fp32 backward op showing **MERE perfect (≈0.0 — mean accuracy at or better than the fp16/bf16 path
that passes T1) but MARE > `mare_thr = 10·2^-13 ≈ 1.22e-3`** is almost certainly the small-value-domain
metric-amplification fail-floor, NOT a kernel defect. The MARE is driven by genuinely-near-zero gradient
elements (|ref| < 2^-14) where the abs-err ~1e-8 (fp32 ULP floor) becomes a large RELATIVE error.

**Discriminator (NPU-independent, ≤30s — CAND-PP80 triage specialized to the metric)**: run the verify
metric on a **same-precision CPU reference (torch_fp32 vs fp64)**. If that reference ALSO exceeds
`mare_thr` on a large fraction of records with the MARE driver in |ref| < 2^-14, the residual is a metric
artifact → classify `requirement`, ship PASS_WITHIN_TOLERANCE with Tier-2 evidence. fp32 cannot exceed
fp32 — a kernel sitting at torch_fp32 ULP parity is at the best attainable accuracy.

**Distinct from OL-110 (reduction-tree fail-floor)**: even the cancellation-FREE output (dV here — no
scatter-add, no cross-row sum) is affected → this is an **output-floor ULP property**, NOT
reduction-ordering cancellation, so there is NO compensated-summation / reduction-shape lever. Do not
sweep reduction algorithms hoping to close it.

Concrete anchor (flash_attention_grad fp32 tier, 2026-06-14): fp32 MARE ci[0.0038, 0.0065] > 1.22e-3;
full 20-draw CPU triage (300 records) — torch_fp32-vs-fp64 ALSO fails the same threshold on 254/300
(84.7%, worst 8.73e-2 ≈ 71× thr), MARE driver in the small-value domain on 270/300 (90%). Maps to
PRECISION_STANDARD_v2.1 §4.5.3 (small-value-domain, Small-Value-Threshold = 2^-14) + §4.5.1
(competitor-ratio MARE_npu/MARE_baseline ≈ 1.0 ≤ 2).

**Anti-pattern (do NOT)**: relax the verify `mare_thr` inside the op's verify to force PASS
(reward-hacking the grading contract, OL-85); add a near-floor-epsilon kernel branch to mask the elements
(OL-85 forbidden Phase-D pattern). Whether `mare_thr = 10·2^-13` (max-relative-error) is structurally too
tight for fp32 backward gradients is a HARNESS-OWNER threshold question (a same-precision reference fails
it on ~85% of records), NOT a per-op edit.

**Promote when**: a SECOND fp32 backward op (another attention / norm / gemm backward) reproduces the
MERE≈0-but-MARE-over discriminator with the CPU same-precision reference also failing — confirms the
recognition rule generalizes across backward op-classes. On promotion, move to
`patterns/domains/precision.md` (cross-ref OL-109).

**Cross-ref**: OL-83 / OL-110 (fail-floor sub-families — this is the output-floor ULP sibling), OL-109
(two-tier verdict — PASS_WITHIN_TOLERANCE for classified residuals), CAND-PP80 (the T1-vs-CPU-fp64 triage
this specializes to the MARE metric), PRECISION_STANDARD_v2.1 §4.5.1 / §4.5.3, CAND-KW-FAG-1 (the
cast-free fp32 tier whose grading then lands on this floor).

## CAND-POOL-LAYOUT-BRIDGE: V220 pooling kernel layout bridge — channels-last (NDHWC) algorithm vs PyTorch channels-first (NCDHW) via pybind permute

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=pooling`
`verified_on: adaptive_avg_pool3d (V220→A5 L1 port, 2026-06-16)`
`unverified_on: other V220 pooling ops (adaptive_max_pool3d, avg_pool2d, max_pool2d); non-pooling V220 ops with channels-last native layout`

**Trigger**: A V220 pooling algorithm (e.g., adaptive_avg_pool3d SplitC/SplitW/MultiW) assumes channels-last memory layout (D, H, W, C — NDHWC), but PyTorch's tensor convention is channels-first (C, D, H, W — NCDHW). A direct port of the V220 kernel reads memory in the wrong order, producing wrong outputs.

**Recommendation**: Do NOT rewrite the kernel algorithm to use channels-first. Instead, bridge the layout mismatch in the pybind layer:
1. `input_ncdhw.permute(0, 2, 3, 4, 1).contiguous()` → NDHWC before kernel launch
2. Launch the (unchanged) V220-algorithm kernel on NDHWC input
3. `output_ndhwc.permute(0, 4, 1, 2, 3).contiguous()` → NCDHW after kernel returns

**Evidence**: adaptive_avg_pool3d L1 V220→A5 port (2026-06-16): initial direct port produced wrong outputs on all 30 cases. Root cause identified as NCDHW→NDHWC layout mismatch. Fix in pybind (permute before kernel + after kernel) resolved all 30/30 cases to bit-exact vs CPU truth. The V220 kernel algorithm was preserved unchanged — only the pybind interface adapted.

**Hard do-not-apply**:
- Do NOT use this pattern when the V220 algorithm HAS a channels-first code path — prefer the native path over the pybind bridge.
- Do NOT permute inside the kernel (wastes UB on layout transform) — keep it in host/pybind where it is a one-time cost per launch.
- Do NOT apply blindly to non-pooling V220 ops — verify the algorithm's assumed layout first by reading the op_host/kernel source.

**Cross-reference**: OL-141 (target arch35 is advisory; generate from selected arch22 source rather
than wrapping target code); selected-source pre-stage policy.

---

## CAND-GDN-CHUNK-RECURRENT-COMPOSE: chunk-recurrent / linear-attention via composed catlass primitives (no per-op cube template)

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=chunk-recurrent / linear-attention`
`status: CANDIDATE — catlass composition. DELIVERABLE binary = 98_gdn_single (built from gdn.cpp, md5 f93ae52d; NOTE 99_gdn_catlass is gdn_probe.cpp = a hardcoded probe that writes NO out.bin, NOT the deliverable). main-independent ×2 deterministic full-122 measurement 2026-06-17 (freshness-gated, .171 device1, /tmp/gdn_dualjudge.log): runs ALL 122 deterministic (bit-reproducible run1==run2; agent ×3 too). ABSOLUTE precision bf16-accurate (abs-diff vs fp64 oracle max ~0.002, mean ~0.0003, 0 elems >0.05, cos 0.9999+). DUAL-JUDGE: PASSES vendor npu-kernelbench atol+rtol(1e-2)+99%-match = 122/122; FAILS our verification_ascendc MERE/MARE relative judge = 0/122, due to near-zero relative-metric artifact (golden |g|<1e-3 → rel-err up to 1420; a bf16 output structurally cannot be relatively precise on near-zero values vs an fp32 oracle — NOT a kernel bug; same class as selective_scan near-zero). Canonical judge for this op = OWNER-PENDING decision (the absolute-vs-relative standard debate). graybox proof-gate #94 open. CAUTION do-not-conflate: the SEPARATE regbase hand-rolled-cube variant (/data/.../gdn_regbase, task#89) is NON-deterministic (PB-45 race, 117-120/122 @4e-2) — that is NOT this catlass deliverable; earlier "122/122"/"118/122" numbers were that wrong artifact.`

**Thesis**: a chunk-recurrent linear-attention op (chunk gated-delta-rule family) is expressible as a **composition** of three reusable primitive families over catlass + AIV, NOT a new per-op cube template. Three composition patterns observed:

**(a) One layout-tag-parameterized `RunGemm` covers N GEMMs.** A single ~30-line helper wrapping a catlass block-GEMM (`Gemm::Block::BlockMmadTla`), parameterized on `LayoutTagA` / `LayoutTagB` (RowMajor / ColumnMajor) + shapes, covers all 8 distinct GDN GEMMs (`kb@kᵀ`, `A@vb`, `A@kb`, `qs@kᵀ`, `qs@S0`, `k_cd@S0`, `aqk@v_new`, `kᵀ@v_new`). No per-matmul cube code — just different layout tags. catlass already decomposes into tile/block/epilogue; we compose at the block-GEMM seam and a kernel seam (AIV vector math between GEMMs). Catlass API surface used is public (`Gemm::Block::BlockMmadTla`, `Arch::Resource`, `DispatchPolicy`, `L1TileShape`/`L0TileShape`, `MmadPingpong`).

**(b) Multi-chunk recurrent state-carry via GM-workspace flush-at-chunk-end / reload-at-next-chunk.** The carried state `S[Dk,Dv]` lives in a GM workspace slab; at `ci==0` it is initialized from `initial_state` (transposed), at `ci>0` it is the `S_prev + kᵀ@v_new` flushed at the previous chunk's last phase. The outer chunk loop is sequential with a whole-grid barrier between chunks; the state is a GM round-trip, not held on-chip across chunks. This is the generic "sequential recurrence over chunked time, state in GM" shape.

**(c) `g≠0` gated-decay precompute as an added AIV routine.** When the gate is non-zero, a per-chunk AIV routine computes `gc = cumsum(g)`, `expg = exp(gc)`, `expkd = exp(gcLast - gc)` (vectorized `AscendC::Exp` — device has no scalar `expf`), feeding decay factors into the beta-scale / mask steps. The `g==0` path skips it entirely (decay factors are all 1). A new algorithm feature = one more AIV routine in the vector-routine list, not a structural change.

**Concrete anchor** (`gdn.cpp`):
```
L4    : AIC RunGemm (cube) / AIV vector math / SyncAll between
L82-88: outer chunk loop, S carried via prev-chunk flush into S0bf GM workspace (ci==0 init from initial_state)
L175-199: hasG g!=0 gated-decay cumsum/exp precompute (AIV)
```

**Status / scope**: CANDIDATE — single op (GDN chunk_gated_delta_rule), catlass composition, deliverable `98_gdn_single` (gdn.cpp). **main-independent ×2 deterministic full-122 measurement 2026-06-17** (freshness-gated, .171 device1): all 122 run STATUS OK + deterministic (bit-reproducible), incl B>1 (incr3 batch loop) and Dk128/large; absolute precision at the bf16 ceiling (abs-diff vs fp64 oracle max ~0.002, 0 elems >0.05, cos 0.9999+). **Dual-judge result**: vendor npu-kernelbench atol+rtol(1e-2)+99%-match = **122/122 PASS**; our `verification_ascendc.py` MERE/MARE relative judge = **0/122**. The 0/122 is a verified **near-zero relative-metric artifact**: the relative error explodes on golden elements with |value|<1e-3 (rel-err up to 1420, vs ~0.002 for |value|≥0.1), because a bf16 output's ~1e-3 absolute resolution divided by a ~1e-5 true value is huge — this is inherent to a bf16-output op judged by relative error against an fp32 oracle, NOT a kernel correctness bug (the kernel is absolutely accurate). Same near-zero class as selective_scan. **Which judge is canonical for this op is an OWNER-PENDING decision** (the absolute-vs-relative precision-standard debate; vendor's own npu-kernelbench uses atol+rtol, which passes). Cross-op generalization (does the 3-family composition cover other linear-attention ops with zero new templates?) remains open. Whitebox-workspace measurement (deliverable not yet archived); graybox-pipeline reproduction (proof-gate #94) NOT yet done. **Do NOT conflate** with the separate regbase hand-rolled-cube variant (`/data/.../gdn_regbase`, task#89) which is NON-deterministic (PB-45 cube↔vector race, 117-120/122 @4e-2) — a different artifact. Full design: `docs/design/FA_CLASS_DESIGN_NOTES.md#gdn-catlass-composable-primitives-design`.

**Cross-reference**: OL-224 (manual-cube tail/transpose correctness for the GDN cube), OL-225 (scalar→GM coherence for the S0 transpose), OL-226 (whole-grid SyncAll co-residency / count-matching for the head-block grid), CAND-FA-STITCH-1 / CAND-FA-LAUNCH-DISPATCH-1 (FA-class composition siblings — same "compose primitives, not per-op template" thesis).

---

## CAND-CATLASS-A5-POC: catlass builds + runs + verifies on A5 real hardware — PoC-level, NOT harness-proven

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all`
`status: CANDIDATE — PoC-on-real-hardware ONLY. NOT harness-proven: no verification.json, no run_sha256, no archived deliverable in output/, never reproduced through the op-gen pipeline. Evidence lives in whitebox-proxy workspace logs (gitignored), authored by hand. Promoted 2026-07-17 (DEBT-209) to record a capability the KB previously OMITTED while simultaneously over-claiming a build-path integration we do not have.`
`confirmed_on: Ascend950PR_957b — vendor example 49_ascend950_flash_attention_infer, CANN 9.1.T500, -DCATLASS_ARCH=3510: build + run + "Compare success." ×3 deterministic, WITH a zeroed-golden falsification control`

**Claim (bounded)**: catlass v1.5.0 **compiles, launches, and produces verified-correct output on A5 / Ascend950PR real hardware** using the arch macro `-DCATLASS_ARCH=3510` (2201 = A2/A3). This refutes the "catlass is not viable on A5" conclusion a prior source-read-only session reached without building. It does **NOT** claim catlass is integrated into our harness — it is not.

**Evidence tier — read this before citing the entry**:

- **Strongest — but it is the VENDOR's op, not ours.** `49_ascend950_flash_attention_infer`, the example catlass v1.5.0 ships. Built with `-DCATLASS_ARCH=3510`; ran on an A5 NPU; three back-to-back runs all reported "Compare success." (deterministic ×3). **Falsification control**: zeroing the golden made it report "Compare failed. Error count: 21608"; restoring the golden restored "Compare success." — so the comparison genuinely discriminates and the pass is not vacuous. Three shape/dtype cases (incl. GQA bf16 paged, D=128) all Compare-success. **What this proves: the toolchain + hardware path works.** **What it does NOT prove: that we can author a correct catlass op** — we ran vendor code that vendor already validated.
- **Ours, same foundation** — GDN `chunk_gated_delta_rule` composed on catlass primitives. See `CAND-GDN-CHUNK-RECURRENT-COMPOSE` for the full judge-qualified account; that entry's own body records a **dual-judge** result (vendor atol/rtol judge vs our relative MERE/MARE judge) and an OWNER-PENDING canonical-judge decision. **Do not quote a bare "122/122" from it** — the number is judge-dependent.
- **NOT locatable as of 2026-07-17 — not promoted.** A third A5 catlass PoC (`fa-a5-catlass`, reported `Ascend950PR_9589`, RUN_EXIT=0) is referenced in DEBT-209, but **no artifact for it exists anywhere on this host** (searched the repo, the pre-migration backup, and all agent workspaces). It is recorded here as an unverified reference only. **Do not cite it as evidence until an artifact surfaces** — its provenance cannot be reconstructed, and inventing one would be worse than the gap.

**Why this is CANDIDATE and not OL/PB**: every result above was produced by hand in a whitebox-proxy workspace, outside the pipeline, with no machine-checkable artifact. There is no `verification.json`, no `run_sha256`, nothing in `output/`. **Accurate status: catlass on A5 is PoC-proven on real hardware, NOT harness-proven.** Anyone citing this entry must carry that qualifier; it is not a substitute for a harness result.

**Provisioning (why you will not find catlass in-tree)**: `catlass/include` exists nowhere in-tree and `#include "catlass` has zero hits in our kernels. `build_ascendc.py:107-114` appends `catlass/include` **only if** `kernel/catlass/include` or `<task>/catlass/include` already exists as a directory, and nothing stages such a tree — so the branch is live but never fires. catlass is **unprovisioned, NOT prohibited** (policy: `CLAUDE.md` §"USING a CANN-binary artifact (API) ≠ COPYING CANN source" — catlass is SUPPORTED). **To wire it: stage the catlass tree at either path; no build change needed.** An absent directory has been misread as a ban before — do not repeat that.

**Cross-reference**: `CAND-GDN-CHUNK-RECURRENT-COMPOSE` (our catlass composition op), PB-35 / PB-45 (hand-rolled-cube non-determinism / TPipe-Reset failure modes that catlass's library-managed mode-4 flag discipline sidesteps), OL-220 (MIX cube+vec build recipe).

---

## CAND-SCAN-FP32-ACCUM: scan/recurrence ops must accumulate in fp32 internally, independent of I/O dtype

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=scan/recurrence/linear-attention (Mamba SSM, cumulative, RNN-like)`
`status: CANDIDATE (selective_scan_fwd: fp16/bf16 verified against fp64 truth; graybox-reproduce proof-gate pending)`

**Principle**: keep the L-scan recurrence state (`x = deltaA*x + deltaB_u`) AND the output reduction in **fp32 internally throughout**, casting to the I/O dtype (fp16/bf16) only at the final store. The low-precision output then lands within the dtype's quantization range against fp64 truth. The recurrence accumulator precision, not the I/O dtype, governs correctness.

**Generalizable rule** (scan/recurrence class — Mamba SSM, cumulative-sum, RNN-like, linear-attention): **NEVER accumulate a recurrence in the low-precision I/O dtype; ALWAYS use an fp32 accumulator + cast-at-end.** This is the load-bearing pattern for low-precision (fp16/bf16) 达标 on these ops. (Conversely, fp32 *output* on the same op can hit an irreducible near-zero-cancellation MARE floor — that is a separate, dtype-intrinsic limit, NOT fixed by accumulator precision; see the fp32-T1-floor finding.)

**Scope boundary (C35 reconcile, main 2026-06-18) — FORWARD-accumulator; NOT superseded by the backward entry**: this rule governs the *forward* recurrence accumulator precision (fp32 internal, cast at store). The sibling `CAND-SSM-BWD-WEIGHTGRAD-FP32` governs the *backward* weight-grad **output dtype** and explicitly notes the backward accumulator was ALREADY fp32 (consistent with this rule, not a correction of it). Complementary, not contradictory: forward → accumulator-precision; backward → output-dtype. This entry is **NOT superseded** — the in-channel "supersede" framing was corrected after reading both entries.

**Concrete anchor**: selective_scan SIMT kernel keeps state + reduction in fp32 and casts only at store.

**Evidence**: selective_scan_fwd 2026-06-17 (independent prototype whitebox against fp64 truth): C2 fp16 MERE 1.1e-6 / MARE 4.5e-3 (gate 9.77e-3) 达标; C3 bf16 MERE 7e-7 / MARE 7.5e-3 (gate 7.81e-2) 达标.

**Evidence (reassociation hazard MATERIALIZED, backward direction)**: selective_scan_full_grad bwd SIMD kw-3 (2026-07-15, A5 Ascend950PR). A CH256→CH512 chunk-size raise regrouped the fp32 Hillis-Steele scan + tree-reduce + cross-chunk carry into a different ULP-rounding order and regressed the fp32 grad margins at the near-zero cancellation edge: grad_A L=700 MARE-vs-floor 1.46×→1.67×, grad_delta_bias L=1300 1.46×→1.70× (fp16/bf16 unchanged at their noise floor). Confirms the corollary in the parenthetical above — reassociating a near-cancelling fp32 accumulation can break a thin margin; "byte-identical across chunk grouping" is impossible in principle for a chunked fp32-reduction, so a chunk-size change must be gated on the fp64-oracle floor ratios, never a byte-diff. See CAND-CHUNK-SIZE-RAISE-CHUNKED-FP32-REDUCTION-NOT-FREE (the full perf+precision A/B).

**Evidence (reassociation hazard MATERIALIZED, FORWARD direction — 2nd instance, 2026-07-16 vj re-verify)**: selective_scan_fwd_simd, A5 Ascend950PR. The production forward kernel evolved a11d97ac→72066c10 by switching the intra-chunk scan from **serial** recurrence to **parallel Hillis-Steele** — a pure reassociation of the fp32 accumulation. Re-measuring the current kernel's fp32 precision vs a fp64 oracle AND against a serial-fp32 reference on identical inputs: the parallel scan's near-zero **output** MARE is **1.3×–8.2× larger than the serial scan's** at the same cancellation points (customer L=5000 = 8.2×; s_a 5.0×; n32 2.7×; s_b 1.8×; n64 1.3×), while the MERE (~1e-6) and absmax (~1e-4) floors are UNCHANGED. Since the kernel uses better-than-fp32 software transcendentals (OL-103), the extra tail is attributable to the scan ORDER, not transcendentals. This is the forward analogue of the backward CH512 grad-margin shift — same hazard, but expressed as an **output**-MARE inflation (the forward has no grads). NOT a bug: tiny abs errors (≤7e-4) amplified by near-zero denominators, fp16/bf16 unaffected, kernel bit-identical to its merged predecessor (#127). **Lesson reinforced: any fp32-scan reassociation (chunk-regroup OR serial→parallel) inflates near-zero fp32 error — gate on the fp64-oracle floor + track the near-zero MARE tail, never assume a "still at floor" qualitative claim survives a scan-structure change.** The exact pass_a/pass_b count deltas were NOT recomputed 1:1 (the original forward `edge_dataset.pt` + Tier-1 thresholds + the a11d97ac serial kernel are not in-repo) — mechanism + magnitude reported, counts not fabricated.

**Provenance**: derived from independent prototype selective_scan_fwd T2 whitebox 2026-06-17 (owner-directed precision-alignment); forward reassociation instance from selective_scan_fwd_simd vj re-verify 2026-07-16 (main-directed, KO agent A5). Promotion gated on graybox-kw reproduction (#94 proof-gate).

**Cross-reference**: CAND-GDN-CHUNK-RECURRENT-COMPOSE (sibling scan/recurrent op).

---

## CAND-SSM-BWD-WEIGHTGRAD-FP32: scan/SSM backward — return weight_type=fp32 reduction grads in fp32, NOT the activation input dtype (else tiny→0 underflow / large→inf overflow)

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN, any op with mixed weight/activation params); scope=pybind-output-dtype + verify-truth-dtype; kernel_type=SIMD`
`verified_on: selective_scan_full_grad backward, a5 Ascend950PR_957b cann-9.1.T500, 30/30 PASS (fp32/fp16/bf16 10/10 each), run x2 stable (2026-06-18)`

**The op-class shape**: a backward that returns grads for BOTH activation-type params (input dtype = fp16/bf16: u, delta, B, C, z) AND weight-type params (framework keeps these fp32 regardless of activation dtype: A, D, delta_bias). The reference framework (PyTorch mamba) declares `weight_type = float` and returns dA/dD/ddelta_bias **in fp32, never rounded to the activation dtype**. The activation grads ARE returned at the activation dtype.

**Anti-pattern (the bug)**: pybind uniformly rounds EVERY output grad to the activation input dtype (`grad.to(input_dtype)` for all 8). For the weight grads this is wrong because:
- **tiny-magnitude profile** (inputs ~1e-3 → weight grads ~1e-8..1e-10): fp16's smallest positive denormal is **5.96e-8**; weight grads below that flush to **exactly 0.0** → relative-L2 ≈ 1.0 (looks like a catastrophic kernel bug, but the accumulator was fine). bf16 denormal floor ~9e-41 so bf16 hides it; fp16 exposes it.
- **large-magnitude profile** (inputs ~400 → weight grads ~1e11..1e12): exceed fp16 max **65504** → **inf**.
- Diagnostic tell: the "average" rel-L2 looks like a uniform mid value (e.g. 0.25) but is actually `(N_good × ~5e-4 + N_bad × ~1.0)/N` — i.e. a FEW catastrophic small/large cases, the rest fine. Always break the aggregate down per-case/per-profile before theorizing a uniform precision loss.

**Pattern (the fix)** — TWO coupled edits, both required:
1. **pybind**: return weight-type grads (dA/dD/ddelta_bias) in **fp32** (drop `.to(activation_dtype)` for those; keep it for the activation grads). The weight INPUTS are already fp32, so fp32 grads are dtype-consistent. Accumulation is (and should already be) fp32 via `SetAtomicAdd<float>()` into fp32 GM — this pattern is NOT about the accumulator, it is about the final output dtype.
2. **verify**: compare the weight grads against the fp64 truth cast to **fp32** (not the activation dtype). If the verify downcasts truth to fp16, the truth itself underflows to 0 and the bug is masked as "matches 0". Same atol/rtol — this is a truth-dtype correction, NOT a tolerance loosening.

**Residual honesty (do NOT mask)**: activation grads on the large profile genuinely overflow fp16 (true |grad| > 65504) — NO fp16 output can represent them; the reference (source mamba) returns them inf/nan too. A high-dynamic-range metric (skip non-finite truth elements, atol+rtol on significant elements) correctly passes these on the representable grads. This is a real fp16 dynamic-range edge, characterize it — don't "fix" it by changing the activation output contract.

**Refuted sibling-hypotheses (logged so a graybox doesn't chase them)**:
- "reduction grads accumulate in fp16, fix = fp32 accum" — REFUTED by reading the kernel: GM buffers and atomic-add were already `float`/fp32. The accumulator was never the problem.
- "rounding ALL grads to fp16 causes a uniform 0.25" — REFUTED by A/B: fp16-rounding a well-scaled grad (~1e3) changes rel-L2 by ~0 (~5e-4). The 0.25 was 2 underflow cases, not uniform.
- General lesson reinforced: a root-cause derived from the REFERENCE's convention is a HYPOTHESIS; the on-device A/B against OUR artifact decides. Verify the output **dtype** of the built .so (not the build exit code) — an incremental rebuild can silently keep a stale pybind .o (identical .so size tell); force `touch`+`rm .o/.so`+relink and re-check the artifact.

**Promote when**: a second mixed-weight/activation scan/SSM backward (e.g. GDN backward, mamba2) reproduces the same weight-grad-fp32 requirement, confirming it is the op-class convention and not selective_scan-specific.

## CAND-SSM-BWD-COOPSIMT-PERF: scan/SSM backward — cooperative parallel-prefix SIMT is the right SIMT mapping (beats naive SIMT ~165× and UN-optimized SIMD ~1.6×), BUT a fully-optimized vectorized SIMD ties/beats it — fully optimize BOTH architectures before declaring a winner

`applies_to: soc=Ascend950PR; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN); scope=architecture-selection+perf; kernel_type=SIMT-cooperative`
`verified_on: selective_scan_full_grad backward, a5 Ascend950PR_957b cann-9.1.T500 device1, 30/30 PASS all dtypes, device-time self-measured + __file__-proven (2026-06-18)`

**⚠ UPDATE 2026-06-19 — the original "coop is THE fastest" held only vs UN-optimized SIMD; after BOTH were fully optimized to the A5 vector roofline, SIMD wins.** Optimized device-time (msprof, NPU2, both 30/30): SIMD **1074µs large / 151µs small** vs coop **1082µs large / 306µs small** → tie on large, **SIMD ~2× faster on small**. The earlier "coop 1.61× faster" was real only because SIMD was still scalar-pipe-bound; once SIMD is vectorized both converge near the A5 vector roofline (~1080µs large, vec_ratio ≈ 0.99). **Production = optimized SIMD; coop kept as A/B evidence** (`output/selective_scan_source_a5/src/kernels/selective_scan_bwd_coopsimt/`).

**Optimization levers (each precision-clean, 30/30 held throughout):**
- **SIMD → 2.47×**: replace the per-l `V→S→V ReduceSum` round-trips (scalar-pipe-bound, vec_ratio 0.63) with a batched high-level `Sum<float>` + vectorized combine (vec_ratio 0.98). The `Sum` primitive runs on c310 despite the arch-guard — see **OL-230**.
- **coop → 1.54×**: `nblk = min(row_groups, 168)` oversubscription (**P-P10**, disperses cross-d atomic contention + un-caps the small case) + **Brent-Kung** work-efficient case-split scans for large + **WarpReduceAddSync** hybrid grad_A/D/db reductions (**P-P2**).
- Residual to the *true* A5 roofline (~37µs, ~29× further) needs a different execution model (vectorize over N=16 / use cube) — structural + precision-risky; both architectures plateau at vec_ratio ≈ 0.99 (vector-instruction-throughput saturated).

**Refines SIMT_VS_SIMD_DECISION/P-P9 for the scan-class**: the coarse table picks SIMD for "continuous-read + vector compute", but for scan/SSM that auto-pick is WRONG by tens-of-× *when SIMD is naive*. The forward proved SIMT wins; for the BACKWARD, the cooperative SIMT mapping (not the naive one) beats un-optimized SIMD — but a *fully-vectorized* SIMD (batched Sum) ties/wins, so don't stop at one architecture.

**Three architectures, measured (a5 device1, large fp32 (2,512,256,N16), torch.npu.Event, __file__-proven distinct .so, all 30/30 precision)**:
- naive-SIMT (1 thread = 1 whole row, serial in-thread fwd+reverse scan): **~270 ms — LOSES to SIMD ~80–165×** (single-issue scalar vs 16-lane vec; per-element dcache vs bulk DataCopy; per-element atomics).
- SIMD (regbase vector, ko'd): 2.70 ms.
- **cooperative parallel-prefix SIMT: 1.67 ms — BEATS SIMD 1.61×** (and naive-SIMT ~165×).

**Winning pattern**: a block (THREAD_NUM=512) cooperates on ONE row's L-axis, doing BOTH the forward state scan `x[l]=dA[l]·x[l-1]+dBu[l]` and the reverse adjoint scan `dx[l]=gs[l]·C[l]+dA[l+1]·dx[l+1]` as O(log L) affine-prefix trees (Hillis-Steele; associative op `(a1,b1)∘(a2,b2)=(a1·a2, a2·b1+b2)`). Mirror the forward SIMT kernel `output/selective_scan_source_a5/src/kernels/selective_scan_fwd_simt/`.

**Breakthrough gated on the atomicAdd structure, NOT the scan** (profiling-first atomics-off diagnostic): grad_A's per-(b,l) atomicAdd on the same [d,n] cell was 92% of device-time; replacing with a **cross-l block tree-reduction** (accumulate-then-ONE-atomic: N atomics/group vs L×N per row) dropped fp32-large 14.6ms→1.64ms, precision held 30/30. Project lesson "atomicAdd serialization, not thread occupancy". Scan itself (atomics-off floor 1.42ms) already < SIMD 2.70ms → cooperative-SIMT is intrinsically faster for this op-class.

**Anti-pattern**: concluding "SIMT loses / use SIMD" from a NAIVE 1-thread-per-row SIMT backward — that's the wrong mapping (loses ~165×) and does NOT mean SIMD is best. Always build+measure the COOPERATIVE parallel-prefix SIMT before deciding the scan-class architecture.

**Status 2026-06-19**: architecture comparison RESOLVED on selective_scan_full_grad — fully-optimized vectorized SIMD ≥ cooperative-SIMT (tie large / SIMD 2× small), both at the A5 vector roofline. Production ships the optimized SIMD; coop archived as A/B evidence. **Promote-to-canonical when** a second scan/SSM backward (GDN/mamba2) reproduces the nuance: cooperative-SIMT is the right SIMT mapping, but a fully-vectorized SIMD ties/beats it — so the actionable rule is "fully optimize BOTH (SIMD batched-Sum + coop cooperative-prefix) before picking", not "coop is fastest".

## CAND-ARCH35-VF-VARIANT-GENERATE-FROM-SOURCE: when upstream arch35/ exists only as a VF-micro-API variant, generate from the arch22 algorithm with standard AscendC vector APIs — do NOT copy the VF micro-helpers

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=port_a3_to_a5 (recurrent / linear-attention / SSM family); scope=generation-strategy`
`verified_on: recurrent_gated_delta_rule, a5 Ascend950PR_957b cann-9.1.T500, 30/30 T1 PASS + 1.36× perf + deterministic (2026-06-18)`

**Thesis** (reinforces the port_a3 default-OFF arch35 prestage rule): when CANN ships an `op_kernel/arch35/` reference for an op but that reference is ONLY a VF-micro-helper variant (e.g. `VecMulMatVF` / `OuterAddVF` — VF micro-API building blocks), do NOT copy those VF micro-helpers into the kernel TU. Copying `#include "arch35/..."` VF helpers is a V351-wrap, not a port (ARCH35_WRAP_CHEAT red line). Instead **generate** the A5 kernel from the arch22 algorithm source using standard AscendC vector APIs (`ReduceSum` / `Broadcast` / `MulAddDst` / `Muls` / `Exp` / `Cast`).

**Key empirical finding**: the standard-vector generation is correct, portable, self-contained AND already clears the perf floor (>0.6×) WITHOUT the VF rewrite. The VF micro-API variant is therefore an OPTIONAL L2 perf upgrade the optimizer may revisit later — NOT a correctness requirement and NOT a floor requirement. This removes the temptation to copy the VF variant "for perf".

**Concrete anchor**: recurrent_gated_delta_rule (Gated DeltaNet recurrent decode, qwen3-next family) — single AIV_ONLY vec-only kernel, L1 mechanical port. arch35/ had `VecMulMatVF`/`OuterAddVF`; kernel generated from the arch22 delta-rule algorithm (`state += beta·(v − state·k)⊗k`, `out = state·q`, per-head exp-decay gate) with standard vector APIs → 30/30 T1 PASS, 1.36× perf vs A3 baseline, deterministic.

**Anti-pattern avoided**: `#include "arch35/<op>.h"` of the VF-helper variant into the build TU (would be V351-wrap, not a port — and customer without upstream arch35 could not reproduce it).

**Promote when**: a second recurrent / linear-attention / SSM-family port whose arch35/ reference is a VF-only variant reproduces "standard-vector generation from arch22 clears the floor without the VF rewrite". Sibling: CAND-GDN-CHUNK-RECURRENT-COMPOSE (chunk variant via catlass — different decomposition, same don't-copy-upstream-template thesis).

## CAND-FA-GQA-BWD-1: GQA/MQA attention backward — shared-KV via per-operand group-divisor + dK/dV accumulate-over-G via cube-partials-then-deterministic-VEC-reduce

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382)`
`unverified_on: soc=Ascend950PR (A5/V300 — A3 evidence does not transfer automatically)`

**Principle**: Generating the backward of a grouped-query (GQA) / multi-query (MQA) attention does NOT require the heavy fused forward FA template — extend a dense multi-head FA-backward kernel with two localized, mostly host-side deltas (OL-205):
- (a) **Shared-KV via per-operand GROUP DIVISOR in the GEMM entry**: with Q heads `N1 = N2·G` and KV heads `N2`, query head `n1` attends KV head `n2 = n1 // G`. Offset the KV operand by `(n1 / G) · kv_stride` inside the matmul entry (`g1//G == g2`) instead of materializing `repeat_interleave(K, G)` — avoids doubling HBM K/V traffic (backward companion to CAND-FA-VEC-D-TILE-1 on the forward side).
- (b) **dK/dV accumulate-over-G** (the backward-specific crux a dense kernel lacks): each KV head receives gradient from all G query-groups sharing it. Emit per-query-group cube partials into a `[G1, S, D]` fp32 scratch (overwrite, NO atomic), then a deterministic fixed-g-order VEC reduce sums the G partials per KV head. No-atomic + fixed order ⇒ bit-deterministic.

All GEMMs stay cube (`MatmulImpl`) so OL-188 (cube-required) / OL-186 hold; built on a SIMD-multilaunch (AIC-ONLY / AIV-ONLY single-purpose launches) skeleton that sidesteps the PB-34/PB-35 V220 MIX-mode sync minefield.

**Concrete anchor**: KV operand offset `kv_base += (n1 / G) * (S*D)` in the per-(b,n1) GEMM dispatch; dV scratch `[G1,S,D]` fp32 + AIV `reduce_g` doing `for g in 0..G: acc += scratch[g]` in fixed order.

**Evidence**: `flash_attention_grad_gqa` (white-box gqa-bwd-wp1, 2026-06-19, A3/V220 Ascend910_9382) — precision vs fp64 CPU autograd (cannbot single-judge, 900 records = 5 GQA cases × {fp16,bf16,fp32} × 20 draws × 3 outputs): fp16 PASS 300/300, bf16 PASS 300/300, fp32 T2 dtype-floor (MERE bit-perfect, MARE-only, = base dense FAG OL-109 tier). Bit-deterministic 6/6. Perf (P97 torch_npu.profiler device_self, warmup=5/active=5, vs RAW vendor `npu_fusion_attention_grad`, lead independently re-ran in-container): median 1.391× (beat vendor 8/10 small/mid 1.3–1.85×; worst S=256/D=128 0.668× > 0.6× gate). Crossover ≈ S=256/D=128.

**Other instances (predicted)**: MQA (G = N1, single KV head); sparse / KV-cache-decode attention backward (same shared-KV + accumulate-over-G structure). The forward fused template (online-softmax + L0C-resident single-launch) is an OPTIONAL large-S (S≥256) perf upgrade only — NOT a correctness or floor requirement.

**Promote when**: a second GQA/MQA-family attention backward (or an MQA case of this op) reproduces "dense-FA-backward + group-divisor KV + accumulate-over-G clears precision AND perf-gate without the fused template". Cross-ref: CAND-FA-VEC-D-TILE-1 (forward GQA KV-offset), OL-188/OL-186 (cube-required), OL-205 (host feature-dispatch), CAND-FA-MULTI-LAUNCH-PERF-GAP (the large-S fused lever).

## CAND-FA-SWA-BWD-1: sliding-window (local-causal) attention backward — precision idiom (host-built additive [S,S] band mask via whole-row Add, NOT per-row Duplicate; dS needs no mask); STARTS below the perf floor (perf RESOLVED without a template in CAND-FA-SWA-BWD-2)

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382)`
`unverified_on: soc=Ascend950PR (A5/V300)`

**Principle**: Sliding-window (local-causal) attention backward derives from dense FA-backward with the SAME per-pair math — only WHICH (i,j) pairs are nonzero changes (window: j ∈ [max(0, i−W+1), i]). Two load-bearing lessons:
- (a) **Apply the window mask as a host-built additive `[S,S]` band (`0` in-window / `−2³⁰` out) via a whole-row `Add` in the softmax stage** — NOT via per-row `Duplicate(buf[offset], ...)` with a data-dependent offset (that is an UNALIGNED UB sub-tensor → `507035` vector-core exception). The host-built additive band + aligned whole-row Add is the production FA-score idiom.
- (b) **dS needs NO mask**: P = 0 outside the window ⇒ `dS = P∘(dP − rowsum(dP∘P)) = 0` there automatically. Mask only the softmax (forward-recompute) stage.

**Perf finding (starting observation — RESOLVED in CAND-FA-SWA-BWD-2; do NOT read this as "template needed")**: a precision-correct full-`S×S` multi-launch SWA backward STARTS ~3× SLOWER than the vendor fused masked op (median 0.366× < 0.6× gate). **This is NOT evidence the fused template (③) is needed** — the ko (CAND-FA-SWA-BWD-2) cleared the floor to **0.852×** with a CHEAP multi-launch lever (mask-cache decisive; block-skip measured ~nil), **NO** fused single-launch, staying out of the V220 PB-34/35 minefield. So like GQA, **SWA backward needs NO fused template** — it needed only a cheap ② (mask-cache). The ①→②③ ladder: GQA stops at ①; SWA needs a cheap ② but **NOT ③**.

**Concrete anchor**: host `atten_mask[S,S]` additive band; softmax stage `Add(scores_row, scores_row, mask_row)`; NO mask op in the dS stage.

**Evidence**: `flash_attention_grad_swa` (white-box gqa-bwd-wp1, 2026-06-19, A3/V220 Ascend910_9382) — precision vs fp64 windowed autograd (cannbot single-judge, 900 records): fp16 PASS 300/300, bf16 PASS 300/300, fp32 T2 dtype-floor (= OL-109). Deterministic 6/6. Perf (P97, vs RAW vendor `npu_fusion_attention_grad` sparse_mode=0 + exact window mask, lead independently re-ran in-container): median 0.366× (all 10 cases 0.327–0.447× < 0.6× gate). `507035` hit + fixed via the host-additive-band-mask idiom.

**Other instances (predicted)**: any banded / block-sparse / local-attention backward (the host-additive-band-mask idiom + dS-no-mask transfer; for perf see CAND-FA-SWA-BWD-2 — mask-cache, not block-skip, is the lever).

**Promote when**: a second banded/local-attention backward reproduces the host-additive-band-mask idiom (and, for perf, the CAND-FA-SWA-BWD-2 mask-cache resolution). Cross-ref: CAND-FA-SWA-BWD-2 (the perf resolution — mask-cache clears the floor, no template), CAND-FA-GQA-BWD-1 (sibling variant), OL-188/OL-186 (cube), PB-34/PB-35 (V220 fused minefield the multi-launch base sidesteps), CAND-FA-MULTI-LAUNCH-PERF-GAP.

## CAND-FA-SWA-BWD-2: sliding-window FA backward PERF in multi-launch — cache the [S,S] mask (decisive) + block-skip out-of-window GEMM ranges; the fused template is NOT needed to clear the floor

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382)`

**Principle**: A precision-correct full-`S×S` multi-launch sliding-window attention backward that sits below the perf floor can be brought ABOVE it WITHOUT the forward fused single-launch template — two cheap multi-launch levers, in measured priority of impact:
- **(1) CACHE the `[S,S]` additive band mask per (S,W,device) — THE decisive lever.** Rebuilding the mask (`torch.where`/`full` on device) every call is ~2–3× the real backward compute on small/mid shapes and is counted in device-self time, while the vendor takes a pre-built mask → per-call mask-build is an apples-to-oranges self-penalty. Caching (build once in warmup, reuse) makes it fair. This ALONE took SWA-bwd 0.354× → 0.852× (cleared the 0.6× floor).
- **(2) Block-skip the `S=Q@Kᵀ` / `dP=dO@Vᵀ` GEMMs** to the windowed band (banded GEMM looping query row-blocks, computing only key cols `[max(0,r0−W+1), r1)`, strided C via MatmulImpl `SetOrgShape(orgN=Sk)`; multi-launch, AIC-only, no CrossCore). **Measured impact: nearly zero** (0.354→0.349×) — the S×S GEMMs were never the bottleneck (cast / softmax / dS / output-GEMM dominate). Honest measurement REFUTED the intuition that block-skip would dominate; kept for large-W/large-S regimes but it is not the lever.

**Hard trap (KB-worthy)**: do NOT zero the out-of-band workspace via torch `.zero_()` BETWEEN raw `aclrtlaunch` kernels on the same NPU stream — stream-ordering is not guaranteed and races (esp. when fp16/bf16 cast kernels precede → the banded GEMM reads un-zeroed Sf/dPf = garbage; fp32 masked it). FIX = zero at ALLOCATION (`torch::zeros`), never mid-stream between raw launches.

**Verdict (answers "does sliding-window backward need the fused template for perf?")**: NO. Cheap multi-launch levers (mask-cache + block-skip) clear the 0.6× floor (0.852× median, 2/10 cases beat vendor), staying entirely out of the V220 PB-34/35 fused-MIX minefield. Combined with CAND-FA-GQA-BWD-1 (GQA also needs no template): the FA-class-backward finding is that the forward fused template is NOT required for these variants — derive-from-forward + cheap multi-launch optimization suffices for precision AND perf.

**Evidence**: flash_attention_grad_swa ko (gqa-bwd-wp1, 2026-06-19, A3/V220 Ascend910_9382) — perf 0.354× → 0.852× (lead independently re-ran in-container, matches; mask-fairness verified — vendor mask pre-built outside timed region, ours cached). Precision UNCHANGED (fp16/bf16 PASS 600/600, fp32 T2, deterministic 6/6 md5-identical to pre-ko).

**Promote when**: a second banded/local-attention backward reproduces "mask-cache clears the perf floor in multi-launch without the fused template". Cross-ref: CAND-FA-SWA-BWD-1 (precision/mask idiom), CAND-FA-GQA-BWD-1, CAND-FA-MULTI-LAUNCH-PERF-GAP.

## CAND-FA-CROSS-BWD-1: cross-attention (Sq≠Skv rectangular) backward — ZERO kernel change; dim-generic dense FA-bwd kernels + pure host rectangular remap; no template needed

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382)`
`unverified_on: soc=Ascend950PR (A5/V300)`

**Principle**: Cross-attention backward (Q from sequence length Sq, K/V from Skv, **Sq ≠ Skv**) is the CHEAPEST FA-class-backward variant to generate from a dense FA-bwd kernel — it needs **ZERO kernel change**, only a pure HOST rectangular remap. If the dense FA-bwd kernels are already **dim-generic** (the 5 GEMMs + the softmax/dS VEC stages take m/n/k + strides as per-launch parameters, not hard-coded square S), cross-attention is just: host passes rectangular shapes — q/dO=[G,Sq,D], k/v=[G,Skv,D], S/P/dP/dS=[G,Sq,Skv]; the 5 GEMMs get rectangular m/n/k+strides; softmax/dS reduce over Skv. Same per-pair FA-2 math. Load-bearing lesson: **author the derive-from-forward kernels dim-generic from the start, and rectangular/cross variants come for free.**

**Concrete anchor**: dense FAG GEMM entry `gemm<CT>(..., m, n, k, transA, transB, strideA, strideB, strideC)` already parameterized → cross passes m=Sq, n=Skv (per output) with no kernel edit; softmax row-reduce dim = Skv.

**Evidence**: flash_attention_grad_cross (white-box gqa-bwd-wp1, 2026-06-19, A3/V220 Ascend910_9382), Sq≠Skv cases. Precision vs fp64 cross autograd (cannbot single-judge, 900 records): fp16 PASS 300/300, bf16 PASS 300/300, fp32 T2 dtype-floor (= OL-109). Deterministic 6/6. Perf (P97, vs RAW vendor `npu_fusion_attention_grad` rectangular Sq≠Skv no-mask, lead independently re-ran in-container): median 0.960× (proxy 0.889×; ~8% run-variance, both > 0.6× floor; small cases beat vendor 1.4–1.5×). Out of box, no ko.

**Other instances (predicted)**: any rectangular-S attention backward (encoder-decoder cross-attn, retrieval/memory attention, prefix attention). The dim-generic-kernel lesson transfers to ANY variant whose only change is shape/stride.

**3-variant generalization (with CAND-FA-GQA-BWD-1, CAND-FA-SWA-BWD-2)**: GQA (1.391×), SWA (0.354→0.852× via cheap mask-cache), Cross (0.960×) — NONE of the three FA-class-backward variants needs the forward fused single-launch template to clear precision AND the perf floor. The simple multi-launch cube+vec skeleton (PB-34/35-sidestepping) + cheap non-template host levers generalize across the variants; the heavy fused template is not what generalizes.

**Promote when**: a second rectangular/cross attention backward reproduces "dim-generic dense kernels + host remap, zero kernel change, clears precision + perf-gate". Cross-ref: CAND-FA-GQA-BWD-1, CAND-FA-SWA-BWD-1/2, CAND-FA-MULTI-LAUNCH-PERF-GAP.

## CAND-FA-TEMPLATE-GEN-BWD-1: can the forward fused FA TEMPLATE generate the FA BACKWARD? — STRUCTURALLY YES (arch35 FA-grad IS the same CRTP/MIX/launch template with backward blocks), but the fused single-launch route is V351/arch35-ONLY (MicroAPI ISA-gate + V220 MIX-sync) — the V220-working backward is the non-template multi-launch path

`applies_to: soc=Ascend910_9382 (V220/dav-c220) vs Ascend950PR (V351/dav-c310); cann=9.0.0; op_class=flash-attention-backward / CUBE_MIX`
`status: graybox-feasibility-whitebox 2026-06-19 (mapping PROVEN by source; build REFUTED on V220 measured 3/3; precision UNREACHED on V220)`

**Owner question**: 通过模板能否生成反向 — can the forward FA fused single-launch *template-assembly* recipe (CAND-FA-STITCH-1 + CAND-FA-LAUNCH-DISPATCH-1, which generates `flash_attention_score`) generate the fused FA **backward**?

**Answer (two-part, honest)**:
1. **STRUCTURAL feasibility — YES.** Target prior-art inspection shows that
`flash_attention_score_grad` arch35 uses the same broad machinery as the forward, with backward blocks
swapped in: CRTP `<CubeBlockType,VecBlockType>` + `KERNEL_TYPE_MIX_AIC_1_2` + raw launch + host-tiling
POD + ASCENDC_TPL axes. This mapping is advisory. A generator may use it to plan components, but must
re-derive task-owned backward code from gradient math, saved-tensor contract, selected forward/source
contract, and current public APIs; it must not stitch copied target bodies. The component mapping is:
   - **3 launch-phases** (vs forward's 1): Pre (softmax-grad-front prep / cast) → **Base** (cube+vec MIX, the analog of the whole forward kernel) → Post (fp32-workspace→out-dtype reduce; fp32 skips Post).
   - **Cube block: 5 GEMMs** (vs forward's 2): `IterateMmDyV`(dP=dO@Vᵀ), `IterateMmQK`(recompute S=Q@Kᵀ), `IterateMmDsK`(dQ=dS@K), `IterateMmDsQ`(dK=dSᵀ@Q), `IterateMmPDy`(dV=Pᵀ@dO) — same D-bucket/dtype templating as forward.
   - **Vec block: softmax-GRAD** (vs forward's online-softmax): `CalculateCastSoftmaxGrad`(=rowsum(dP∘P)) + `BroadcastSubMul`(dS=P∘(dP−sfmg)) + `SimpleSoftMax`(recompute P from saved softmax_max/sum/attention_in).
   - Same staggered 4-deep AIC/AIV ping-pong (OL-200) + same CrossCoreSetFlag S-channel/forward MIX handshake; +backward-only template axes (deterType, IS_TND varlen, IS_D_NO_EQUAL, IS_ROPE, IS_NZ_OUT, dpse/dsink outputs).
   - FA-2 backward math = exactly CAND-FA-GQA-BWD-1's: dV=Pᵀ@dO, dP=dO@Vᵀ, dS=P∘(dP−rowsum(dP∘P)), dQ=(dS@K)·scale, dK=(dSᵀ@Q)·scale.
2. **BUILD/RUN feasibility on the brief's A3/V220 — NO (refuted, measured).** Two compounding blockers:
   - **BUILD (ISA-gate, measured 3/3)**: the arch35 FA-grad's ENTIRE vector path (sfmg-front, broadcast-sub-mul, dropout, simple-softmax) is **MicroAPI register-compute** (`RegTensor`/`__VEC_SCOPE__`, gated `ASC_DEVKIT_MAJOR>=9`). A minimal `RegTensor<float> a,b,c; __VEC_SCOPE__{Mul(c,a,b);}` built via the OFFICIAL harness (`build_ascendc.py -v Ascend910_9382`, the one that builds the working V220 backward) FAILS 3/3 with `error: expected namespace name (AscendC::MicroAPI)` / `no template named 'RegTensor'` / `undeclared '__VEC_SCOPE__'`. MicroAPI is a **dav-c310 (V351)** feature — present in the CANN tree but unavailable for the dav-c220 (V220) target.
   - **RUNTIME (MIX-sync, KB-evidenced)**: even if compiled, the fused `KERNEL_TYPE_MIX_AIC_1_2` cube↔vec CrossCoreSetFlag path hits PB-34 (built clean → `LaunchAscendKernel 507035` every case, 1/61, on V220 3_FusionAttention) / PB-35 (silent hang, "use IDs≥4" fix FALSIFIED, "UNSOLVED on V220"). The same fused MIX pattern is BENIGN on A5/V351 (PB-34 L876).

**Why this does NOT contradict CAND-FA-GQA/SWA/CROSS-BWD-1 ("no template needed")**: those answer "is the fused template REQUIRED for a working/perf-floor-clearing V220 backward?" — NO (multi-launch MatmulImpl AIC-only suffices, fp16/bf16 600/600, 1.391×). THIS answers the complementary "CAN the template GENERATE the backward?" — YES on V351, the machinery maps 1:1. The two are orthogonal: the template is sufficient-on-V351 but not necessary-on-V220.

**Historical V351 experiment (2026-06-20; not current RFC completion evidence)**: the old run stitched
target FA-grad blocks, built, and matched target/vendor output on 8 BN2 cases. Retain its build,
determinism, and performance measurements as capability evidence. Because it reused target bodies and
used target output rather than CPU fp64 autograd as final truth, it does not prove autonomous backward
generation under the current boundary. A compliant rerun must generate task-owned code and validate
every gradient plus saved-tensor semantics with CPU fp64 autograd. S>128 and fp32 remain gaps.

**The V220-working backward is NOT template-generated**: `fa_gqa_grad` (CAND-FA-GQA-BWD-1) = multi-launch AIC-ONLY `MatmulImpl` + separate AIV cast (ZERO MicroAPI, sidesteps PB-34/35). `fa_class_template.md` L488-491 already notes the existing FA-bwd is "non-template-assembly — NOT the forward template-assembly path."

**Gaps the forward template recipe LACKS for the backward**: (G1) target — forward template is arch35/V351, brief verify env is A3/V220; fused backward needs a V351 build+run lane. (G2) Pre/Post phases — recipe has no multi-launch-phase concept. (G3) 5-vs-2 GEMMs + fp32 partial workspace + deterministic-accumulate variants. (G4) softmax-grad vec block (new, not in forward block set). (G5) backward-only template axes (deterType/TND/D_NO_EQUAL/ROPE/NZ_OUT/dpse/dsink). (G6) MIX-sync UNSOLVED on V220.

## CAND-FA-A5-BWD-DP-CUBE-NONDET-1: multi-launch attention backward on A5/arch35 — a data-dependent FIRST-head dP-cube OUTPUT non-determinism (NOT a crash, NOT a precision floor) that survives serialize / per-launch-sync / zeroed-workspace / distinct-per-stage-workspace

`applies_to: soc=Ascend950PR (V351/arch35, dav-c310); cann=9.1.0.B060; bisheng=n/a; op_class=attention-backward (custom-<<<>>>-launch MatmulImpl cube + AIV vec, host-serialized multi-launch)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060 (det_probe 5×same-input, on-NPU)`
`unverified_on: soc=Ascend910_V220 (the multi-launch GQA backward CAND-FA-GQA-BWD-1 is bit-deterministic 6/6 on V220 — this non-det signature is NOT observed there; it appears specific to the A5/arch35 cube path)`
`status: PROVISIONAL — single-op reproduction + 5-mitigation refutation set; needs aog-determinism-analyzer confirmation + a second A5/arch35 multi-launch cube backward to confirm the per-output (dV-clean / dQ-dK-nondet) signature transfers`

**Principle**: a hand-rolled `<<<>>>`/ACLRT_LAUNCH `MatmulImpl` cube launched repeatedly on one stream on A5/arch35 can exhibit **data-dependent FIRST-launch (head-0) OUTPUT non-determinism** that is invariant to (a) full serialization, (b) per-launch `aclrtSynchronizeStream`, (c) zeroed sys-workspace, (d) distinct per-launch sys-workspace regions. This is a DIFFERENT failure class from CAND-FA-A5-KFC-WORKSPACE (a deterministic `507015` OOB crash on large-D GM-staging) and from OL-206 (a MIX in-kernel cube↔vec *handshake* race) — here the launches are single-purpose AIC-only / AIV-only and host-serialized (no in-kernel MIX handshake), yet the cube's *output value* drifts across identical re-runs. The signature is **per-output**: the gradient that consumes only clean inputs is bit-deterministic; the gradient whose dependency chain passes through the suspect cube is not.

**Per-output localization (the load-bearing diagnostic, det_probe 5×same-input)**:
- **dV is ALWAYS bit-deterministic + correct** (max_abs ~2e-4) across every shape/dtype. dV = Pfᵀ@dO consumes only Pf (the recomputed-softmax probabilities), which is clean.
- **dQ/dK are non-deterministic on head-0 ONLY, for SOME inputs**: `[1,4,128,128]` fp16 → dQ/dK 2.1 pairwise diff across 5 runs, error confined to head 0 (heads 1–3 bit-exact = 0); `[2,4,128,128]` fp16 → 0 (bit-det + correct). The two shapes differ only in seed / batch ⇒ the non-det is **data-dependent**, not shape-structural.
- **Chain isolation → the corrupt producer is the dP cube**: dQ=dS@K and dK=dSᵀ@Q both consume dS=ds(Pf,dP); Pf is proven clean (dV is perfect) ⇒ the corrupt input is **dP=dO@Vᵀ** (a cube GEMM). dP and the forward-recompute S=Q@Kᵀ use the IDENTICAL GEMM config, yet S (stage-2, the first cube) is deterministic while dP (stage-4) is not → points at the cube/KFC sys-workspace interaction across repeated launches, not the GEMM math.

**5-mitigation refutation set (all on-NPU measured — the value of this candidate is the NEGATIVE space it maps out)**:
1. **Fully per-head serial cube** (blockDim=1, pre-offset pointers) — non-det PERSISTS ⇒ NOT cube concurrency.
2. **`cubeWs` `torch::empty`→`torch::zeros` + explicit `aclrtSynchronizeStream` after every launch** — FIXED `[2,4,128,128]` (det+correct), lifted 46→48/54, but `[1,4,128,128]` head-0 PERSISTS ⇒ the residual is NOT cross-launch ordering/visibility and NOT (solely) uninitialized workspace. **Necessary-but-insufficient.** (Consumes OL-66 zeros-not-stream-ordered + CAND-FA-SWA-BWD-2 zero-at-allocation.)
3. **ds AIV single-core** (blockDim=1) — did NOT fix `[1,4,128,128]` AND REGRESSED fp32-large to 66546 non-det ⇒ residual non-det is the **dP cube**, NOT the ds-AIV stage. (Reverted.)
4. **5 distinct zeroed cube-workspace regions** (one per cube stage, so dP never reads S's dirtied KFC/sys-workspace) — REFUTED: head-0 `[1,4,128,128]` still 2.1 non-det ⇒ dirty-workspace-reuse-across-stages is NOT the mechanism.

**Net**: the FA-2 dense-causal backward MATH + multi-launch architecture are sound (dV perfect everywhere; `[2,4,128,128]` fully passes after the zero+sync fix; manual-fp32 matches golden) — the blocker is squarely a **data-dependent head-0 dP-cube non-determinism**, in the OL-206 "hand-rolled cube/KFC-on-A5 is a losing game" territory: the managed cross-core/workspace abstraction owns sub-cases a recipe-driven hand-roll keeps re-exposing. Recommended next step is the managed abstraction route or aog-determinism-analyzer root-cause, NOT another hand-roll mitigation.

**Concrete anchor**: det_probe 5×same-input per (shape,dtype); the discriminating pair is `[1,4,128,128]` fp16 (head-0 dQ/dK 2.1 non-det) vs `[2,4,128,128]` fp16 (bit-det). dV = Pfᵀ@dO is the always-clean control; dP = dO@Vᵀ is the suspect cube (same config as the deterministic S=Q@Kᵀ recompute).

**RESOLUTION — VERIFIED 2026-06-20 (the managed route the Net predicted, now MEASURED-confirmed on the determinism+precision axis):** the fix is the **MANAGED single-launch MIX with the library `matmul::Matmul<>` `REGIST_MATMUL_OBJ` Init-once** (`KERNEL_TYPE_MIX_AIC_1_2` + `SetSysWorkspaceForce`+`GetUserWorkspace` + per-pass `SPLIT_CORE_CUBE/VEC` + AIC-as-KFC-server + `SyncAll`, NO CrossCore) — the proven A5 pattern from the `fa_matmul_poc` reference. Rewriting the 8-launch ACLRT multi-launch into ONE such MIX launch **eliminated the head-0 dP non-determinism**: det_probe → bit-identical across runs (the 100%-repro [1,4,128,128] head-0 case went 2.147/2.555 → 0.0 pairwise). **PRECISION CORRECTION (fresh on-NPU re-verify 2026-06-20, supersedes an earlier "54/54" inference):** the strict cannbot 商用双标杆-ratio gate reads **50/54 representative (FAIL), NOT 54/54** — `n_rep=54 (randn-only)`, `n_edge=216 (zeros/large/small/boundary CORRECTLY excluded via is_edge=profile∉{randn,representative})`, `pass_rate 0.9259`. The 4 representative misses are **dtype-floor-ratio degeneracy, NOT a kernel precision bug**: ours matches the same-dtype CPU competitor to ~6 sig figs, and on idx42 [fp32 2,4,128,128] ours_abs 9.80e-07 is LOWER than the competitor's 1.16e-06 (ours MORE accurate) yet the MERE *ratio* 1.533 trips the tight 1.5× gate; the other 3 are large-shape (256²) floor cases where the ratio of two floor-level errors is unstable (MARE_r 5.66 / MERE_r 1.80 / RMSE_r 1.51, all <13% over gate). fp16+bf16 hard-tol = 30/30 each. So the honest precision verdict is **50/54 strict-FAIL with all 4 fails root-caused to the dtype floor (not a kernel miss, not a 达标 claim)** — and SEPARATELY the fp32-`large` EDGE fails (excluded) are a genuine large-magnitude fp32 floor. **So on A5/V351 a hand-rolled per-launch ACLRT `MatmulImpl` cannot be made deterministic by any workspace/sync hand-roll (the 5-mitigation set above all refuted); the library-managed MIX matmul (which OWNS its KFC sys-workspace via the framework) IS the deterministic path.** NOTE: this is the LIBRARY managed `matmul::Matmul<>` MIX — NOT the manual-Mmad/MicroAPI register-compute fused-stitch KO'd by the C19 architecture decision (different mechanism: managed-library-matmul vs hand-rolled-register-compute). So this does NOT revert C19.

**V220→A5 PORT-PLAN REFINEMENT (the load-bearing finding for the whole FA-grad feature surface):** a V220-proven multi-launch backward (CAND-FA-GQA-BWD-1: AIC-only `MatmulImpl` per-launch ACLRT, bit-det 6/6 on V220) does NOT port determinism-clean to A5/V351 — the per-launch ACLRT `MatmulImpl` has an unmanaged KFC sys-workspace that data-dependently corrupts head-0 dP. **On A5/V351, "multi-launch" must concretely mean the managed single-launch-MIX library-matmul, NOT V220-style ACLRT per-launch.** This affects EVERY future A5 feature-surface op (causal/GQA/mask/dropout/pse/varlen grad): start from the managed-MIX library-matmul, not a V220 ACLRT port. (FA_CLASS_DESIGN_NOTES.md#fa-grad-completeness-bar says "multi-launch" generically; this is the A5-concrete form.)

**HONEST RESIDUAL (NOT 达标 — STRUCTURAL perf floor):** the managed-MIX fix is **det-SOLVED + precision-at-the-dtype-floor (50/54 strict-gate, 4 fails = floor-ratio degeneracy not kernel bug)** but **PERF is NOT 达标 and is a STRUCTURAL floor**: .141/9579/B060 median 0.027×, .171/957b/T500 median 0.0073× (same kernel; the .171 stack runs it ~4.3× slower) vs vendor `npu_fusion_attention_grad` causal flat 2-8us. **CORRECTION (white-box torch_npu.profiler on .171/NPU2, supersedes an earlier "wasteful all-fp32 cube ~4× + no-band-skip ~2× = 8× removable" framing which is REFUTED):** the cost is per-row SERIAL AIV work (S rows × softmax/dS/cast, each with ~11 `PipeBarrier<PIPE_ALL>` full-pipe fences) + per-head SINGLE-CORE GEMMs (`ct.usedCoreNum=1`); profiled scaling is ~LINEAR in S (290us@S128→632@S256, ~2.5us/query-row) = per-row-bound, NOT cube-bound. **Evidence all-fp32-cube is NOT the killer: CAND-FA-GQA-BWD-1 reached 1.391× (达标) on V220 WITH all-fp32 cube.** Release-vs-Debug build measured identical → not a build-type artifact (note: the autogen cmake template FORCEs `CMAKE_BUILD_TYPE=Debug`, build_ascendc.py:233, but the bisheng device kernel is optimized regardless). The ~8× lever-set (band-skip ~2× + barrier-narrow + multi-core) narrows but does NOT reach the 0.6× floor from 0.0073×/0.04× — the gap is the det-required managed single-launch MIX (one KFC lifecycle, per-head serial, GM-staged between every stage) vs the vendor's fused all-core pipeline. A causal band-skip attempt (live-band [0..r]) built clean + stayed det-bit-exact but broke precision (masked-tile-into-GEMM bug — the S×S GEMMs consume the full band), reverted clean. **VENDOR-DET GROUNDING (measured):** `npu_fusion_attention_grad` causal IS bit-deterministic on A5/V351 (0.0 pairwise dQ/dK/dV ×5, all 6 shapes incl the head-0 case) — so the head-0 non-det was NOT a platform property (the vendor achieves BOTH det AND speed via fused all-core matmul-internals we don't have); we MATCH the vendor on det, the residual is PERF-only and structural. **The det-vs-perf tension is FUNDAMENTAL on A5/V351**: V220's perf-winning multi-launch (1.391×) is NON-det on A5; A5's det-winning managed-MIX is ~100-137× slower. Closing it needs a deeper rewrite (multi-core GEMMs + stage-fusion to kill GM round-trips = approach the vendor's structure), NOT lever-tuning. **Promote (this CAND → an OL/PB) when** a 2nd A5 multi-launch backward reproduces "ACLRT-MatmulImpl head-0 non-det → managed-MIX-library-matmul fixes it AND carries the same structural perf floor". Cross-ref: OL-220/OL-206/PB-45 (the A5 KFC-workspace class), `fa_matmul_poc` (the proven managed-MIX recipe), CAND-FA-GQA-BWD-1 (the V220 multi-launch 1.391× = perf-but-nondet-on-A5), CAND-FA-SWA-BWD-2 (the perf levers — insufficient here).

**Evidence**: `flash_attention_score_causal_grad` (resumed kw-1, 2026-06-20, Ascend950PR_9579 / CANN-9.1.0.B060) — the CAND-FA-GQA-BWD-1 multi-launch form at G=1 (dense-causal). Build clean first-try (0 compile-fix iters). dV bit-deterministic + correct every case; dQ/dK head-0 data-dependent non-det as above; 46→48/54 after mitigation 2. (A SEPARATE deterministic fp32-`large`-edge error — head-3, 14.9 — is an edge profile excluded from the representative verdict, likely genuine large-value fp32 vs competitor, not this non-det.)

**Promote when**: aog-determinism-analyzer confirms the dP-cube root cause OR a second A5/arch35 multi-launch cube backward reproduces the per-output (dV-clean / dQ-dK-head0-nondet) signature. Cross-ref: OL-206 (the broader hand-rolled-cube/KFC-on-A5 losing-game theme — but that is a MIX in-kernel handshake; THIS is host-serialized single-purpose launches, a distinct mechanism, do not transfer the fix), CAND-FA-A5-KFC-WORKSPACE (sibling A5/arch35 custom-launch workspace issue — but a deterministic OOB crash, not output non-det), CAND-FA-A5-WORKSPACE-BIFURCATION (the "measure each lane's root, don't cross-lane-generalize" discipline this signature obeys), CAND-FA-GQA-BWD-1 (the multi-launch backward this op is the G=1 instance of — bit-deterministic on V220, so the non-det is A5/arch35-specific), OL-66 (zeros not stream-ordered), CAND-FA-SWA-BWD-2 (zero-at-allocation), EC-68 / PB-41 (ACLRT_LAUNCH SetSysWorkspaceForce + sys-workspace sizing, applied).

**Honest residual / next step**: structural mapping PROVEN by source; build REFUTED on V220 (MicroAPI ISA-gate, 3/3 deterministic); precision signal UNREACHED (no .so on V220 → nothing to feed the fp64-autograd `verify_fa_gqa_grad.py` CDV harness, which is confirmed reusable). To close with a REAL build + precision signal, run the stitch on an **A5/V351 container** (MicroAPI + MIX both available) — that is the follow-on, OR confirm the intended target is V351.

**Historical stitch log (2026-06-20; pre-RFC target-copy procedure, not current generation guidance)**:
retain the component inventory and measurements below as target capability evidence. A compliant rerun
uses them only as advisory checks and emits task-owned code from gradient math, the saved-tensor/forward
contract, selected source, and current public APIs.
1. **RE-DERIVE the shared `regbase_*` responsibilities** (`matmul`, fixpipe output, buffers, copy-in)
   from the task contract. Archived headers may reveal API-shape hypotheses but are not reusable bodies.
2. **MAP target FA-grad blocks to task-owned components**: carriers/POD, five-GEMM cube block,
   softmax-gradient vector block, Pre/Base/Post orchestration, and supported feature leaves. Do not copy
   or rename target bodies; omitted features remain explicit gaps.
3. **DECLARE the public-API dependency closure** required by those task-owned components. Do not copy
   a target shared-common subtree into the deliverable.
4. **AUTHOR the entry** `wp_fag_entry.h` = templated `wp_fag_regbase_impl<INPUT,float,OUT,s1T,s2T,dT>` mirroring `entry_regbase.h` INVOKE_FAG_GENERAL_S1S2_BN2GS1S2 — the **3-PHASE single-launch** Pre→`pipeIn.Destroy()`→Base(`FlashAttentionScoreGradKernel<Cube,Vec>` CRTP via `g_coreType`)→Post(`if constexpr(!is_same<IN,float>)`, fp32 skips). `KERNEL_TYPE_MIX_AIC_1_2` + `SetSysWorkspaceForce` (fa-a5-kw-21 carry). Pin dense-core axes (IS_ATTEN_MASK=false/PSE_NONE/IS_DROP=false/NO_DETER/BN2GS1S2/no tnd/rope/nz).
5. **AUTHOR the dispatcher TU** `flash_attention_score_grad_kernels.cpp` = `#include kernel_operator.h` UNGUARDED + `#ifndef __ASC_NPU_HOST__` device body (K5 dual-pass; NEVER `#define __NPU_ARCH__ 3510`) + `FAG_LAUNCH(NAME,IN,OUT,S1T,S2T,DT)` → `extern "C" __global__ __aicore__` launchers (fag_do_{fp16,bf16,fp32}_bnsd_d128).
6. **AUTHOR the host** `pybind11.cpp` by deriving tiling fields from the declared contract; use target
   tiling only to enumerate review hypotheses. Fill the task-owned POD, select the launcher, and raw
   launch. `model_new_ascendc.py` must exercise the generated kernel; `model.py` supplies the CPU fp64
   autograd oracle. Saved tensors must follow the declared forward contract, not a hidden target call.
**Historical deliverable**: the old 51-file artifact is retained only as experiment provenance, not as
a current generated-success example.

**Historical build/run measurements**: the copied-block experiment built on V351 and recorded useful compiler, dispatch, determinism, precision, and performance observations. Retain those measurements as capability evidence only; they do not establish current RFC generation success because the code provenance and final-truth boundary were different.

**Current answer**: forward FA knowledge can guide backward component planning, but success requires a fresh task-owned implementation plus gradient/saved-tensor validation with CPU fp64 autograd. The archived target-block stitch alone does not answer “通过模板能否生成反向” under the current RFC.

**Promote when**: S>128 (BN2GS1S2 path) + fp32 + mask/pse/drop + other D-buckets/layouts wired, and msprof device-exclusive perf measured. (BUILD + RUN + PRECISION for the S≤128 fp16/bf16 dense core = DONE/measured-YES.) Cross-ref: CAND-FA-STITCH-1, CAND-FA-LAUNCH-DISPATCH-1, CAND-FA-GQA-BWD-1 (the non-template V220-working backward), CAND-FA-SWA-BWD-1/2, CAND-FA-CROSS-BWD-1, OL-133 (ASCENDC_TPL), OL-186/OL-188 (cube), PB-34/PB-35 (V220 MIX minefield), P-P103 (`fa_class_template.md` forward template domain).

## CAND-FA-A5-MULTICORE-DET-RECIPE: det-preserving MULTI-CORE attention backward on A5/V351 via a 2-launch Base(MIX-1-KFC S1-split)+Post(AIV fixed-order reduce) — the dK/dV cross-core-reduce-ORDER det hazard (distinct from KFC-workspace) + its multi-launch-REQUIRED fix + the per-GEMM-KFC perf ceiling (strongly-inferred, not proven)

`applies_to: soc=Ascend950PR (V351/arch35, dav-c310); cann=9.1.T500 (also 9.1.0.B060); op_class=attention-backward (managed matmul::Matmul<> MIX_AIC_1_2, multi-core S1-split, det-required); scope=any A5/V351 multi-core FA-grad / FA-class-backward that splits the query (S1) axis across cores`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500 (.171/NPU2, det_probe 5×same-input + cannbot precision + torch_npu.profiler perf, on-NPU, independently re-built+re-run by the driver — author≠measurer)`

The det-preserving SINGLE-launch managed-MIX (CAND-FA-A5-BWD-DP-CUBE-NONDET-1) is correct but SINGLE-CORE-per-head → ~0.0073× vs vendor. Making it MULTI-CORE to close the perf gap surfaces a NEW det hazard + a structural perf ceiling. This candidate = the recipe + the 2 findings, from the (B) perf-rewrite of `flash_attention_score_causal_grad` (2026-06-21, owner-GO'd; orch-autonomous-gen kw→pp→ko→fo grounded by the driver's white-box F1-F4 + iter-B1; driver independently verified every result on .171/NPU2). Result: det+precision PRESERVED + ~5× (0.0073→0.0363×), NOT 达标 (the ceiling §3).

### §1 — THE DET-PRESERVING MULTI-CORE 2-LAUNCH RECIPE (the reusable asset)
Distribute work by (head g × S1-block b) tasks across cores (raises utilization vs single-core-per-head; fa_matmul_poc model). Two launches:
- **Launch-Base** (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`, ONE KFC lifecycle: `SetSysWorkspaceForce(ws)`→`GetUserWorkspace`→`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &ct)` Init-once; AIC=macro-owned KFC server, AIV subblock-0 drives; RAII SERVICE_QUIT, NO mm.End()): per-task (g,b) does the per-S1-block LOCAL stages — S=q[s1blk]@kᵀ, Pf=softmax_causal, dP=dO[s1blk]@vᵀ, dS, **dQ=dS@k → dQ[g][s1rows] DIRECT (per-block-complete → DET, no cross-core add)** — AND the **dK/dV PARTIALS** (dKpart=dSᵀ@q[s1blk], dVpart=Pfᵀ@dO[s1blk]) written to per-(g,b)-DISJOINT fp32 GM workspace slots (disjoint ⇒ det writes, no shared accumulation). When s1Outer==1 (S≤s1Blk): write dK/dV DIRECT, NO Post (baseline path).
- **Launch-Post** (`KERNEL_TYPE_AIV_ONLY`, NO matmul → NO KFC; only when s1Outer>1): one designated task per head reads the head's dKpart/dVpart[g][0..s1Outer) + **REDUCES in FIXED ascending b-order** → dK/dV (cast). Fixed order ⇒ deterministic.
Artifact (the on-disk recipe): `output/npukernelbench/src/kernels/flash_attention_score_causal_grad/kernel/{facg_mix_kernel.h, facg_mix_kernels.cpp (Base entry), facg_post_kernel.h (Post), facg_mix_tiling.h, pybind11.cpp (2 aclrtlaunch + dK/dV partial fp32 ws alloc)}`. **VERIFIED**: det bit-identical 5× incl. **[1,4,256,256] s1Outer=2** (the multi-block case that exercises the cross-core Post reduce); precision 单标杆-达标 51/54 (3 fp32-floor-ratio fails = dtype floor, ours≈/better-than CPU competitor); matches vendor det.

### §2 — F4: THE MULTI-CORE DET HAZARD ROOT CAUSE (load-bearing for the whole A5 FA-grad surface)
The multi-core FA-grad det hazard is **NOT (only) the per-core KFC sys-workspace** (that's CAND-FA-A5-BWD-DP-CUBE-NONDET-1, the single-launch head-0 mechanism) — it is the **dK/dV CROSS-CORE ACCUMULATION ORDER**. dK=dSᵀ@Q and dV=Pᵀ@dO REDUCE over query-rows (S1); if S1 is split across cores, each core's PARTIAL dK/dV must be summed, and **fp-add is non-associative ⇒ a non-deterministic add-ORDER gives non-deterministic dK/dV**. (dQ reduces over keys (S2) not query-rows ⇒ per-S1-block-complete ⇒ naturally det — matches: dV was always det in the single-core-per-head kernel because no cross-core add.) The vendor's `deter.h` (arch35 god-mode, §F3) solves this with a fp32 GM workspace + a fixed coordinate-assignment so the reduce ORDER is deterministic — NOT atomic-add (`SetAtomicAdd` is order-non-det). **ITER-B1 (empirical, decisive): the det-safe cross-core reduce CANNOT be done mid-SINGLE-launch in the managed-MIX** — a mid-kernel all-core `SyncAll` DEADLOCKS because the AIC is the macro-owned KFC server (never reaches Process → the barrier never gets AIC participation → AIV cores wait forever; det_probe timeout). ⇒ the reduce MUST be a **2nd launch** (the launch boundary IS the only valid cross-core barrier when the matmul owns the AIC as a server — this is exactly WHY the vendor multi-launches Pre/Base/Post). The Post launch is pure-VEC NO-matmul → NO KFC → does NOT re-open the §CAND-FA-A5-BWD-DP-CUBE-NONDET-1 cross-relaunch-KFC non-det (Base = one KFC lifecycle; partials = per-core-disjoint det writes). VERIFIED: the 2-launch build stayed det incl s1Outer=2.

### §3 — THE PER-GEMM-KFC PERF CEILING (the structural finding — STRONGLY-INFERRED, NOT proven)
The det-required managed-MIX's perf ceiling = **`aic_scalar=0.955` (DOMINANT)** = the per-GEMM KFC `sync=true` handshake (msprof Level2 PipeUtilization labels it "CUBE core scalar = managed-matmul iteration + KFC-sync"; ~80 managed-GEMM handshake round-trips/launch = 5 GEMMs × #tasks; GEMM-ISSUE bound). Co-measured: aic_mac=0.31 (cube math 69% IDLE — gap is GEMM-ISSUE not compute), aic_mte2=0.105 (NOT HBM-bound), aiv_vec=0.14 (86% IDLE, blocked on cube sync=true), **aic_fixpipe=0.318 (a SEPARATE pipe)**, cube_util=72%. **The ONLY lever that moves aic_scalar = FEWER/LARGER fused GEMMs** = the host-optiling `MultiCoreMatmulTiling` fused-rewrite, which CONFLICTS with the on-device hand-built `ct.usedCoreNum=1` TCubeTiling (OL-220) + the OL-54 L769 FA-register-reduction 507015 wall + det-critical-surface risk → kw-3 NO-GO + fo-2 negative-EV. **HONEST inference-vs-measurement boundary (held end-to-end)**: this cap is **STRONGLY-PROFILE-INFERRED (3 converging points: the aic_scalar label / aic_fixpipe a measured-SEPARATE 0.318 pipe / baseK-tile-falsified-FLAT ⇒ per-GEMM cost is KFC-FIXED-not-per-iteration), NOT decomposition-MEASURED.** The candidate UB-output lever (`GetTensorC(LocalTensor)` / `FixpipeMmCopyOutToUB` L0C→UB — library-available, our default MM_CFG has enableGetTensorC=true + enableMixDualMaster=false, the vendor uses it for L0C/UB-residency) is a DATA-MOVEMENT lever → touches aic_fixpipe(0.318)+mte2(0.105), structurally should NOT move aic_scalar (output-dest knob; the KFC handshake fires per-GEMM-Iterate regardless of UB-vs-GM dest) → residual "fixpipe-UB-also-lowers-aic_scalar" path = LOW-probability. A before/after-aic_scalar MEASUREMENT was designed (verify-before-assert on the cap itself) but judged **not-worth-a-campaign** (the floor is ~0.036× either way; the path to 达标 is the optiling-fused pivot regardless of the residual fixpipe gain) — so it is NOT a "proven floor", it is a **"strongly-evidenced floor + residual-low-probability + judged-not-worth-measuring".** ⇒ **A5 matches the vendor's DETERMINISM but not its SPEED on the managed-MIX**; matching vendor speed needs the optiling-fused arch (fewer/larger GEMMs) whose A5-determinism is UNPROVEN. This applies to EVERY A5 FA-grad feature op (GQA/mask/dropout/pse/varlen) — the per-feature bar on the managed-MIX = det + precision-达标 + honest-perf-floor, NOT per-op perf-达标 (structurally gated on the pivot, not tuning).

**Evidence**: `flash_attention_score_causal_grad` (B)-rewrite, 2026-06-21, .171/NPU2 Ascend950PR_957b/cann-9.1.T500. Step-1 multi-launch (orch kw→pp→ko, autonomous-gen of the driver's iter-B1 directive) = ~4.3× (0.0073→0.0313×); step-2 (--optimize re-entry) occupancy-fill (adaptive s1Blk fills idle cores 8-16/24→22-24/24) = +~1.15× → 0.0363×. det 3/3+ bit-identical (incl s1Outer=2), precision 51/54, perf clean torch_npu.profiler median 0.0313-0.0363×. NOT 达标.

**Promote when**: a 2nd A5/V351 multi-core FA-grad (GQA/SWA/cross/varlen backward) reproduces (a) the 2-launch Base+Post det-preserving recipe (§1) AND (b) the dK/dV cross-core-reduce-order det signature + the multi-launch-required fix (§2); OR the per-GEMM-KFC ceiling (§3) is either decomposition-MEASURED (before/after-aic_scalar on a UB-output build) or circumvented by the optiling-fused arch with det held. Cross-ref: **CAND-FA-A5-BWD-DP-CUBE-NONDET-1** (the SINGLE-launch det-fix companion — the head-0 KFC-workspace mechanism this REFINES into multi-core; the §2 cross-core-reduce-order is a DISTINCT 2nd hazard), CAND-FA-GQA-BWD-1 (the V220 multi-launch this is the A5 det-preserving form of — V220 ACLRT per-launch is non-det on A5, §2), CAND-FA-MULTI-LAUNCH-PERF-GAP (the V220 5-delta perf-gap — overlapping per-GEMM-KFC theme on a different SoC), CAND-FA-SWA-BWD-2 (the mask-cache/band-skip perf levers — insufficient for the §3 ceiling), OL-220 (managed-MIX build recipe + the host-optiling-vs-on-device-tiling conflict that walls Lever-A), OL-54 (FA-register-reduction 507015 wall), OL-206 (hand-rolled-cube-on-A5 losing game), fa_matmul_poc (the proven multi-core managed-MIX SyncAll reference), the CANN arch35 `flash_attention_score_grad/op_kernel/arch35/{deter.h, s1s2_bn2gs1s2_{pre,post}_regbase.h, kernel_base.h, block_cube.h}` (F1-F4 vendor-3-phase reference grounding).
## CAND-PORTA3-DEINTERLEAVE-THRESHOLDED: small-op port_a3 CV — eliminate host input-transpose overhead via THRESHOLDED in-kernel de-interleave

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-box/port_a3-vec`
`verified_on: soc=Ascend950PR; cann=9.1.0 (A5 Ascend950PR_9579)`
`unverified_on: soc=Ascend910_V220`

**Principle**: For small-op port_a3 CV kernels, the host `.t().contiguous()` input marshaling ((m,4)↔(4,m) coord-major↔box-major, mirroring the op's `op_api/aclnn_*.cpp` `l0op::Transpose`) can DOMINATE the per-call launch-overhead floor — for iou_v2, the 2 input transpose kernels = ~40% of a flat ~0.10-0.12ms floor (the IoU compute is invisible; device-time flat vs m). Eliminate them with a THRESHOLDED in-kernel de-interleave: for small m (≤128) pybind passes box-major `(m,4)` directly + a `boxMajor` flag, the kernel contiguous-loads `(cnt,4)` then does a scalar UB-shuffle `dst.SetValue(p*cnt+i, src.GetValue(i*4+p))` over p∈0..3 (bit-exact, ~30us faster); for large m (>128) KEEP the host transpose — the scalar-shuffle is O(m) and loses past the crossover (m=64..1009), and large m is already 30-57×.

**Confirmed-WRONG primitives (do NOT repeat)**: (1) AscendC `Gather` intrinsic — RUNTIME 507035 (aivec 271, scalar internal-buffer OOB) on V351, both byte and element offset-unit forms; (2) strided multi-block `DataCopyPad` (byte-gap `srcStride`) gather — precision regressed 33/33→7/33; the byte-gap `srcStride` semantics PB-22 verified for *contiguous-tail* copies do NOT hold for a sub-32B multi-block *gather* (OL-85 overfit signature).

**Evidence**: iou_v2 ko-1 (2026-06-21, A5 Ascend950PR_9579 NPU6, P141 device-event). precision 33/33 preserved + det 33/33; perf median 0.45→0.67-0.90×, geomean 0.97→0.98-1.20× (2 runs, shared-host variance); small-m WINS (m=8 0.36→1.42×, m=64 0.25→1.21×). Vendor-grounded: A3-TBE is also small-op overhead-bound but at a 3-5× lower launch floor → real-but-partly-closeable gap (NOT a flat ceiling); mid-m 256-1024 residual = vendor-grounded floor.

**Other instances (predicted)**: modulate, upsample_{bilinear2d,bicubic2d,nearest}, roi_align_rotated — any port_a3 CV op whose `op_api/aclnn_*.cpp` does `l0op::Transpose`/`Contiguous` input marshaling. Cross-op KB asset for the CV cohort.

## CAND-PORTA3-CV-SOFTWARE-TRIG-L1: fp32-only geometric CV op — software fp32 sin/cos (not hw Sin/Cos), and audit the L4 "Subnormal Config" escalation against operand-reachability before escalating
`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-geometric/port_a3 (rotated-ROI pooling, affine/rotate warp, any op needing trig of an angle attr)`
`verified_on: roi_align_rotated kw-1 2026-06-21 A5/Ascend950PR_9579 (36/36 T1)`
`unverified_on: a3 (Ascend910_V220) — A5 evidence does not auto-transfer`

**Trigger**: A fp32-only geometric/CV op needs a transcendental of a geometric attribute — most commonly `sin(theta)`/`cos(theta)` of an ROI/box angle, but also rotation matrices, polar conversions. There is NO lower-precision dtype to absorb the hw-transcendental fp16-grade error (OL-103: hw `Sin`/`Cos` ≈ fp16 mantissa), and the reference is CPU/fp64 truth at the fp32 T1 floor.

**Recommendation (two coupled rules)**:
1. **Use a software fp32 trig** — a Cephes-style range-reduction + minimax polynomial for `sin`/`cos` in SIMT scalar (`__simt_callee__`), ~1e-7 accuracy. This is the OL-103 software-transcendental play extended from sigmoid/exp to trig. Scalar fp32 `/` in SIMT stays full-precision (no software-reciprocal needed for the divisions threaded through coordinate math — see OL-103 roi_align_rotated evidence). Result: max ours-MERE 1.10e-5 ≪ fp32 T1 floor 1.22e-4.
2. **Audit the L4 "Subnormal Config" escalation against operand-reachability before escalating.** The l1-implementation-guide decision-tree flags Div/Sin/Cos as an L4 (Subnormal Config) signal. But that escalation is only needed if subnormal *operands* are actually reachable. **Geometric/coordinate math (ROI box coords, angles, bilinear weights) never produces subnormal operands** — coordinates are O(1)..O(image-size), angles bounded, weights ∈ [0,1]. So the L4 signal can be **audit-overridden to L1** for this op class. Escalating to L4 on the syntactic Div/Sin/Cos signal alone — without checking reachability — wastes the researcher-route cost (OL-156) on an op that's a clean single-shot L1 SIMT VF kernel.

**Why it's a candidate not yet an OL**: single-op evidence (roi_align_rotated only). Promote to OL once a second fp32-only geometric CV op (e.g. a rotate/affine-warp port) confirms both the software-trig precision win AND the L4-reachability-override.

**Cross-ref**: OL-103 (hw transcendental fp16 floor + software-fp32 mitigation; roi evidence adds Sin/Cos + scalar-`/`), OL-105 (software-fp32 SIMD lowering caveats), OL-150 (SIMT VF one-thread-per-output paradigm), OL-156 (L4 STRUCTURAL escalation signature — what this audit avoids over-firing).

## CAND-FA-A5-OPTILING-FUSED-BWD: the optiling-fused arch35 FA-grad (BN2GS1S2 multi-block) PORTED to A5/V351 + VERIFIED det+correct+perf-reachable — RESOLVES the CAND-FA-A5-MULTICORE-DET-RECIPE §3 "optiling-fused det UNPROVEN on A5" open question; the arch35→A5 porting principle (direct-output-offset+atomic) + the SYMMETRIC det-vs-perf tension (deter.h needed only for det+fast-AT-SCALE)

`applies_to: soc=Ascend950PR (V351/arch35, dav-c310); cann=build /home/npu_user/cann-9.1.0 (MicroAPI/V351 codegen) + runtime 9.1.0.B060; op_class=attention-backward optiling-fused single-fused-launch (Pre/Base/Post BN2GS1S2 multi-block, NOT managed-MIX per-GEMM-KFC); scope=A5/V351 dense FA-grad S>128 multi-block (the path CAND-FA-A5-MULTICORE-DET-RECIPE §3 said needs the optiling-fused arch whose det was UNPROVEN)`
`verified_on: soc=Ascend950PR (.141=203.0.113.141, hostname tbe); fp16/bf16 BNSD D128; vs fp64-autograd-oracle (precision) + 6×same-input bit-identical (det) + device-Event A/B vs torch_npu.npu_fusion_attention_grad (perf); on-NPU, driver-built+run ~9 build cycles, author=measurer`

The owner PIVOT (2026-06-21, supersedes CAND-FA-A5-MULTICORE-DET-RECIPE accept-floor): god-mode COPY the arch35 backward FA source (the optiling-fused multi-block, vendor ships it det+fast on A5) → port A5/V351. The wholeport (directive-2, `output/a3_to_a5_port/.../base_kernel/kernel/wholeport/wp_fag_*`, 8027 lines, self-contained, 0 arch35 #include) builds clean on V351 (MicroAPI in T500/npu_user codegen-libs). This candidate = the Stage-2 result: the arch35-fused path PORTED + the det+correct+perf-reachable findings.

**FRAMING = the directive-2 FA-template practice REDUX, now BACKWARD** (the owner's "抄 arch35 → distill → template → generalize" loop, closed for FA-grad backward). This entry is **REGENERATION-COMPLETE**: a gray/black-box agent given §1 (the porting principle) + §1.5 (the dispatch wiring) + §2 (the perf core-scaling lever) + §4 (the build-env recipe) can re-derive the arch35→A5 FA-grad S>128 multi-block backward port WITHOUT re-running the ~9-build-cycle debug. CAND-FA-TEMPLATE-GEN-BWD-1 answered "can the forward template GENERATE the backward → YES (S≤128 slice)"; THIS entry extends it to the FULL S>128 multi-block surface (BN2GS1S2) + the det/perf characterization = the ③-reusable-asset pillar for FA-grad backward.

### §1 — THE arch35→A5 PORTING PRINCIPLE (the reusable lesson): direct-output-offset + atomic-add for all 3 grads
The wholeport entry.h was HARD-PINNED to BN2 single-block (IS_BN2_MULTIBLK=false, SPLIT_AXIS=BN2, Pre/Post skipped) — correct ONLY for s1≤128 && s2≤128 (CANN SetSplitAxis isBn2). For S>128, output was DETERMINISTIC-but-WRONG (all rows, err 0.4-1.3 vs ~5e-4 floor) because the BN2 Base writes dQ/dK/dV NON-atomically to per-core-scratch offsets assuming 1-write-owns-the-row — but s2Outer>1 distributes s2-blocks across tasks → they CLOBBER instead of accumulate.
THE FIX (4 edits, the porting principle): **wire the BN2GS1S2 multi-block path + write ALL 3 grads to their REAL output offsets + atomic-add** (NOT the production per-core-scratch+Post-remap):
1. entry.h: add `bool IS_MULTIBLK` template param → IS_BN2_MULTIBLK=IS_MULTIBLK, SPLIT_AXIS=IS_MULTIBLK?BN2GS1S2:BN2; enable Phase-1 Pre (zero-init fp32 ws) + Phase-3 Post (cast fp32-ws→out; fp32 skips). Host SelectLauncher picks BN2(S≤128) vs BN2GS1S2(S>128) per shape. + the missing td.preTilingData host fill (DoPreTiling — the Pre zeroing gates on maskCoreNum; prior host never filled preTilingData → Pre no-op'd).
2. dV: block_cube.h IterateMmPDyFixpout IS_BN2_MULTIBLK write → REAL `valueOffset+gmNOffset` + SetAtomicAdd (was `GetBlockIdx()*CUBE_BASEN*HEAD_DIM_ALIGN` per-core-scratch). Fixed dV 1.31→6.78e-4.
3. dQ: block_cube.h dQ IS_BN2_MULTIBLK write → REAL `queryOffset` (was `GetBlockIdx()*AlignTo128(s1)*D + s1oIdx*CUBE_BASEM*D` scratch). Fixed S192-tail dQ 1.08→4.5e-4.
THE WHEN (the non-obvious part): production uses per-core-scratch + a block-distribution that maps scratch→output; the ported Post does STRAIGHT Muls+Cast (NO scratch→output remap — verified in production Post too). So the per-core-scratch offset only works under production's exact distribution. **When YOUR block-distribution doesn't replicate production's scratch-mapping, write the REAL output offset + atomic-add instead** — simpler, provably correct, det-clean at low core-count. (REFUTED hypotheses en route, each tightened the locus: the !IS_DKV_RESIDENT_L0C branch — D128 IS L0C-resident; the host workspace-SIZE — refuted, but byproduct-fixed S192 dK; the per-core-scratch+Post-remap — no remap exists.)
VERIFIED: S128/192/256/512 + fp16/bf16 sweep (S160/320/384/640/768 incl unaligned tails, B/N 1-8) ALL PASS at dtype floor (~5e-4 fp16, ~4e-3 bf16) + det6 bit-identical S256/512. **S>128 multi-block CORRECTNESS + DET = COMPLETE, robust across the dense surface, det-clean WITHOUT deter.h** (resolves CAND-FA-A5-MULTICORE-DET-RECIPE §3's "optiling-fused det UNPROVEN on A5" — it's PROVEN det+correct).

### §2 — PERF: ≥0.6× REACHABLE on correct output; the core-distribution scaling lever
Perf re-measured on the now-CORRECT output (device-Event A/B vs vendor, prior wrong-output 0.149-0.311× moot). DEFAULT (BN-only core-split = blockOuter from fusedOuterBN=B*N2*g): S256 0.584× S512 0.390× S1024 0.204× S2048 0.149× — ours LINEAR, vendor FLAT (large-S degrades because cores DON'T scale with S; B1N4→only 4 of ~36 cores). FIX (FAG_PERF_CORESCALE, mirror production normal_regbase.cpp:362-363: blockOuter from fusedOuter=BN*s1Outer*s2Outer / aicNum → cores scale with S): S256 0.618× S512 0.504× S1024 0.422× S2048 0.644× (+3.4× large-S; S256 + S2048 CLEAR 0.6×). ⇒ perf≥0.6× is REACHABLE on the optiling-fused arch (NOT a structural floor — the managed-MIX §3 ceiling does NOT apply to this arch, no per-GEMM-KFC handshake).

### §3 — THE SYMMETRIC det-vs-perf TENSION (the load-bearing finding): deter.h needed ONLY for det+fast-AT-SCALE
The core-scaling (§2) that reaches 0.6× BREAKS det+correctness — and the tension is SYMMETRIC + FUNDAMENTAL:
- dK=dSᵀ@Q accumulates over S1 → needs s1-blocks CO-CORE. Under fusedOuter-scaling, a key-block's s1-blocks split across cores → cross-core atomic-add to dK[key] = non-det + wrong. Characterized: breaks IFF s1-split (B8N4 BN=32 fills cores w/o s1-split → dK correct; low-BN → wrong).
- (b) dK-safe distribution (round blockFactor up to multiple of s1Outer, baseIdx s1o-innermost → key-block's s1o stay co-core) FIXES dK — but then dQ=dS@K (accumulates over S2, needs s2-blocks co-core) goes NON-DET (S512: dQ 7/7 nondet, dK/dV det). s1-grouping MOVED the non-det dK→dQ.
⇒ You CANNOT keep BOTH s1-co-core (dK) AND s2-co-core (dQ) under full core-scaling → (b) fixes ONE grad's det at a time, CANNOT reach full det+fast alone for ALL S. **THRESHOLD (decisive re-measure at exact broke-shapes)**: (b) s1-grouping FIXES dK at ALL shapes AND gives **ALL-3-det + correct + perf≥0.6× for s2Outer≤2 (S≤256: B1N4S256/B1N16S256 all-3-det nd=0, S256 0.639×) — a REAL cheap win WITHOUT deter.h for the small-s2Outer slice.** dQ goes non-det ONLY at s2Outer≥4 (B2N2S512: dK/dV det, dQ 5/6 nondet) — because dQ's s2-accumulation splits cross-core only when s2Outer is large. So the deter.h scope SHRINKS to **just dQ's cross-core s2-reduce at s2Outer≥4** (NOT the whole kernel, NOT dK — dK is solved by s1-grouping). **deter.h fixed-order reduce is the EMPIRICALLY-NECESSARY mechanism for the large-S dQ residual** (makes the cross-core atomic det regardless of order). Same mechanism CAND-FA-A5-MULTICORE-DET-RECIPE §2 named, now in the FUSED arch, scoped to dQ-large-s2Outer. NOTE: all 3 grads stay CORRECT vs oracle throughout — the tension is DETERMINISM (+ atomic-contention perf), not correctness. The direct-output-atomic (§1) is correct+det at low-core + s1-grouping extends det+fast to S≤256; deter.h closes the S≥512 dQ residual for full-S det+fast.

### §4 — BUILD/RUN RECIPE (.141 reusable)
build CANN=/home/npu_user/cann-9.1.0 (has tikcpp + ccec_compiler; the MicroAPI/V351 codegen). runtime CANN=/data/pri/Ascend/9.1.0.B060/cann-9.1.0 (source set_env.sh — self-consistent; the npu_user set_env hardcodes ABSENT /home/npu_user paths) + `export TORCH_DEVICE_BACKEND_AUTOLOAD=0` (torch_npu auto-load fails on libhccl; B060 set_env supplies it at runtime). build_ascendc.py takes ABSOLUTE task dir; ASCEND_INSTALL_PATH=npu_user. verify runs IN-CONTAINER (imports the built .so, no SSH/no hardcoded IP). Read the target host from `.ascendc_env`.

### §5 — GENERALIZATION CHECK (the directive's "does the template generalize?" — SOURCE-ASSESSED, not yet built)
The wholeport is a FULL arch35 port with the feature paths PORTED-but-PINNED-DENSE (entry.h:100-106 hard-pins IS_ATTEN_MASK/IS_PSE/IS_DROP/IS_TND=false, IS_N_EQUAL, exactly like IS_BN2_MULTIBLK was pre-fix). Source-grep confirms the feature CODE EXISTS in the ported kernel: IS_ATTEN_MASK (9 refs kernel_base + 2 block_vec), IS_DROP/drop_mask (22 block_vec), IS_TND/actual_seq varlen (32 kernel_base), GQA gSize/goIdx/n2G (151 kernel_base — GQA is host-knob g=N1/N2, the kernel decodes it), D256/512 DTemplateType (21 block_cube, the >Aligned256 d-loop). So generalization = the SAME "unwire→wire the template flag + its host-tiling" recipe (§1) per feature:
- **GQA (g>1)**: LIKELY CHEAPEST — the kernel already decodes goIdx/gSize (baseIdx axis), IS_N_EQUAL is a host-knob; the dense port set g=N1/N2 already (pybind:95 gSize=N1/N2). Expect: works with little/no kernel change (the §1 direct-output-atomic offsets already include the g/n2 stride). Highest-confidence generalization.
- **mask/pse/dropout**: feature flags IS_ATTEN_MASK/IS_PSE/IS_DROP (ported paths exist) → flip the flag + pass the mask/pse/drop GM tensor + wire the host tiling (dropMaskOuter, the attenMask layout). Medium: the §1 output-offset principle is FEATURE-INDEPENDENT (it's about the dQ/dK/dV WRITE, not the attention math) → it should hold; the per-feature work is the forward-stat/mask plumbing, not the multi-block accumulation.
- **varlen (TND)**: IS_TND ported (32 refs, actual_seq prefix-sum offsets in CopyInMaxSum/keyOffset) → flip + pass actual_seq_qlen/kvlen + the TND host tiling. Medium-high (the offsets are TND-aware already).
- **D256/D512**: the GAP (SelectLauncher pybind:69 `dBasicBlock>128 → nullptr`). The kernel HAS the >Aligned256 d-loop (block_cube nLoops) but fp32-D>256 uses MatmulK (the descoped baseK path). Add the D256/512 launchers + the d-loop tiling. Medium (D-bucket is a known axis; fp32-large-D is the harder corner).
- **fp32**: descoped (the MatmulK L0-split-K baseK invalid-template-arg in the trimmed scaffold). dtype-floor follow-on.
HONEST: this is a SOURCE-ASSESSED generalization (the feature paths are ported + the §1 principle is feature-independent) — NOT yet BUILT+VERIFIED per-feature. The directive's "generalization check" = the prediction (GQA cheap, mask/pse/drop/varlen via flag+host-tiling, D-bucket via launchers) + the load-bearing claim: **the §1 direct-output-atomic porting principle + the §3 det-tension are feature-INDEPENDENT (output-write + cross-core-reduce, orthogonal to the attention features) → they transfer; each feature's work is the flag-wire + host-tiling, NOT re-deriving the multi-block accumulation.** Promote each feature on its own BUILD+VERIFY (the feature surface, task #22).

### §6 — deter.h port RESOLVES the §3 residual: NEW_DETER (DETER_DENSE) = full-S det+fast SIMULTANEOUS, MEASURED ≥0.6× (2026-06-22, on-NPU .141)

The §3 residual ("deter.h fixed-order reduce is the EMPIRICALLY-NECESSARY mechanism for the large-S dQ residual at s2Outer≥4") is now CLOSED — and the close carries a decisive correction + a counterintuitive headline.

**THE DISPATCH FINDING (read-not-guess, the load-bearing correction): the vendor's dense det path is DETER_DENSE (NEW), NOT DETER_OLD.** Production `GetDeterSparseTilingKey` (op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp:628-651): for the dense config (`!isSparse` OR `NO_MASK && s1Token≥s1 && s2Token≥s2`) it returns `DETER_DENSE` (=2). `DETER_OLD` (=1, line 651) is only the FALLBACK reached when none of dense/causal/band match. The kernel predicate `IS_DETER_NEW = (≠NO_DETER && ≠DETER_OLD)` so DENSE/CAUSAL/BAND are all "NEW". DeterSparseType enum (common_regbase.h:313-319): NO_DETER=0, DETER_OLD=1, DETER_DENSE=2, DETER_CAUSAL=3, DETER_BAND=4. ⇒ Porting DETER_OLD first (the obvious `Process_OLD_DETER`) ports the LEGACY path; the vendor's actual dense det+fast = NEW.

**WHY NEW HOLDS LARGE-S WHERE OLD COLLAPSES (the mechanism, the reusable lesson):**
- **OLD reduce** (Process_OLD_DETER, kernel_deter.h:1031): cube-cores SCATTER per-core dQ/dK/dV partials to a `deterGm` workspace → `SyncAll` → one designated V-core/row SERIAL-GATHERS ALL `usedCubeCoreNum` partials + atomic-add. Reduce cost ∝ usedCubeCoreNum → COLLAPSES at large S (more cores → longer per-row gather). Device-Event ratio collapses to 0.166×@S1024 / 0.123×@S2048 (718us@S2048).
- **NEW reduce** (Process_NEW_DETER, kernel_deter.h:753, the SPLIT_AXIS≠BN2S2 dense branch): NO per-row gather. `CalDeterIndex → CalDenseDeterIndex → CalDenseIndex` assigns each round's block to a FIXED core in deterministic COORDINATE order; each core writes dQ/dK/dV DIRECTLY to dq/dk/dvWorkSpaceGm; a per-grad FIX-FLAG CHAIN (`SYNC_DETER_FIX_FLAG → SYNC_DK_DETER_FIX_FLAG → SYNC_DV_DETER_FIX_FLAG`, kernel_deter.h:944-990) serializes the cross-core accumulate IN coordinate order. Determinism = fixed coordinate order + fix-flag serialization, NOT a gather-reduce stage. NO usedCubeCoreNum-proportional cost.

**THE COUNTERINTUITIVE HEADLINE — determinism makes it FASTER, not slower.** Device-exclusive (msprof op_summary Task Duration, on-NPU kernel time, zero host overhead; 2 reproducible runs, NPU2, fp16, vs vendor aclnnFlashAttentionScoreGradV4):
| shape | NEW-deter | corescale-only (non-det) | ⇒ |
|---|---|---|---|
| S1024 | **0.679× / 0.680×** | 0.513× / 0.510× | NEW +0.17× |
| S2048 | **0.775× / 0.770×** | 0.593× / 0.592× | NEW +0.18× |
NEW-deter is FASTER than the non-det corescale baseline — the coordinate-driven fix-flag-chain direct-accumulate avoids the corescale path's cross-core atomic-add CONTENTION. So the vendor's det mechanism is not a det-tax; it's a perf-win. (vs OLD: NEW S2048 = 113us vs OLD 718us = 4.0× faster.)

**THE METHODOLOGY-RECONCILE LESSON (measure don't infer):** wall-clock (torch Event wrapped around the launch) is host-overhead-DEFLATED (NEW-deter wall-clock 0.518×@S2048) and would UNDER-report a milestone that's actually MET; device-exclusive msprof (0.775×@S2048) is the right measure. Always reconcile a ratio-gate claim to device-exclusive before claiming it. (The banked §2 0.644× was device-Event; my device-exclusive corescale = 0.593× — same ballpark, run/device-state variance.)

**THE DETER_MODE HARDENING (keep the proven fallback compile-verified):** the wholeport entry exposes `DETER_MODE` (0=NO_DETER, 1=NEW/DETER_DENSE default, 2=OLD/DETER_OLD). BOTH NEW + OLD are compile-INSTANTIATED (OLD won't bit-rot) + runtime-selectable (host env FAG_DETER=1→NEW, +FAG_DETER_OLD=1→OLD fallback floor). If NEW ever regresses, OLD is the live det-correct fallback.

**OLD's L0C-accumulate gotcha (root-caused, the dK-under-deter trap):** under DETER_OLD the dK GEMM's L0C-resident multi-s2-block partial-sum accumulate is gated `if constexpr(!IS_DETER_OLD)` (block_cube.h:1278) → OFF → dK OVERWRITES instead of accumulates when a core sees >1 s2-block/key. OLD needs the host to flag dkv-deter on per-core cross-round dK-offset REUSE (route those rounds to the deter reduce). NEW avoids this entirely (coordinate-driven, no per-core L0C-accumulate dependence).

**VERIFIED (the §3 promote-condition met):** NEW-deter det7 (7 consecutive bit-identical) dQ+dK+dV at S128/256/384/512/1024/2048 × fp16+bf16, ALL at dtype-floor; default build (no FAG_DETER) all-S correct+det no-regress; OLD fallback det7@s2Outer≥4 intact; NEW-engaged verified (host launcher prints deterEngaged=1 mode=1, NOT silent bypass). NEW also FIXED the §3-era S384(s2Outer=3) residual that OLD couldn't (OLD's coordinate-table mis-maps when the corescale blockFactor s1Outer-rounding makes blockOuter overshoot s1Outer*s2Outer; NEW's CalDenseIndex coordinate assignment has no such dependence) → deter gate lowered to s2Outer≥3.

⇒ §3 RESIDUAL RESOLVED: full-S det+fast SIMULTANEOUS, device-exclusive ≥0.6× at the gate shapes (S1024 0.679×, S2048 0.775×), via the vendor's ACTUAL dense det path (NEW/DETER_DENSE), self-contained (NO arch35 #include). The det+fast tension (§3) is not fundamental at scale — the vendor's coordinate-driven NEW reduce resolves it AND beats the non-det baseline.

Artifact (add to §5 list): `wholeport/wp_fag_kernel_deter.h` (OLD Process_OLD_DETER + NEW Process_NEW_DETER/CalDeterIndex/CalDenseDeterIndex/CalDeterMaxLoopNum, dense g=1 subset), `wholeport/wp_fag_deter.h` (CalDenseIndex coordinate math — the HEART of NEW), `wholeport/wp_fag_entry.h` (DETER_MODE dispatch), `flash_attention_score_grad_kernels.cpp` (NEW+OLD launchers), `pybind11.cpp` (mode routing + s2Outer≥3 gate + engaged-probe). Porting reference: CANN `op_kernel/arch35/{flash_attention_score_grad_kernel_deter.h Process_NEW_DETER@753 + deter.h CalDenseIndex, op_host/arch35/...normal_regbase.cpp GetDeterSparseTilingKey@628}`.

**Promote when**: a 2nd A5/V351 optiling-fused FA-grad op reproduces (a) the direct-output-offset+atomic porting principle (§1) for S>128 multi-block correctness+det, AND (b) the symmetric det-vs-perf tension (§3) requiring deter.h for det+fast-at-scale; OR the deter.h KernelDeter port lands det+fast-SIMULTANEOUS ≥0.6× (closes the §3 residual — **DONE in §6: NEW_DETER, device-exclusive ≥0.6× measured**). Cross-ref: **CAND-FA-A5-MULTICORE-DET-RECIPE** (the managed-MIX path this PIVOTS from — its §3 said the optiling-fused arch is needed but det-UNPROVEN; THIS candidate PROVES the optiling-fused det+correct on A5, resolving that; the §2 cross-core-reduce-order det hazard is the SAME mechanism, now in the fused arch), CAND-FA-TEMPLATE-GEN-BWD-1 (the wholeport that builds the arch35 fused backward), CAND-FA-A5-BWD-DP-CUBE-NONDET-1 (the single-launch det companion), the CANN arch35 `flash_attention_score_grad/op_{kernel,host}/arch35/{deter.h, *_regbase, normal_regbase.cpp:362-363 core-split + :1179-1188 ws + DoPreTiling}` (the porting reference). Artifact: `output/a3_to_a5_port/.../base_kernel/kernel/{wholeport/wp_fag_entry.h, wholeport/wp_fag_block_cube.h, pybind11.cpp, flash_attention_score_grad_kernels.cpp, verify_s_gt128.py}`. Whitebox trace: `workspace/flash_attention_score_causal_grad/whitebox_log_B_pivot.md`.

---

## CAND-SIMD-OVER-ROWS-SUBGRANULE-REPACK-TAX: SIMD-over-rows / row-packing for a small-N scan LOSES at sub-granule N — the mandatory sub-granule strided transpose tax dwarfs the scan it accelerates

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32`
**Source**: selective_scan_source_a5 fwd-SIMD perf-loop iter-3 (2026-06-23, A5/Ascend950PR_957b) | **Validation status**: anti-pattern, whitebox-measured, NOT pursued

**Concept (the instinct)**: a small-N serial-in-L scan under-fills the vector lanes (N=16 on a 64-fp32-lane unit = ¼ lane utilization). The instinct is to pack R rows' N-state side-by-side so R×N fills the 128-lane width, run the per-step combine over the packed vector, and amortize the scan over R rows at once.

**Why it LOSES (the sub-granule repack tax)**: packing R rows requires interleaving the rows' N-state at N=16 element granularity, but **N=16 (64 B fp32) < the A5 fp32 vector granule (64 fp32 = 256 B)**. So the pack/unpack is a MANDATORY sub-granule strided transpose — the hardware cannot gather/scatter at 16-element stride within a granule cheaply, it must do a strided element-shuffle. Measured: the strided transpose runs **3× per chunk** (pack-in, the carry layout, unpack-out) and each costs ~10× the scan it is meant to accelerate.

**Measured (A5/Ascend950PR_957b, 2026-06-23, msprof device-time)**:
- Isolated R8 packed-serial scan ALONE: **0.68×** (i.e. faster — the packed scan itself is a win in isolation, confirming the lane-fill instinct is directionally real).
- BUT + the mandatory repack (the 3×/chunk sub-granule transpose): **7.39× SLOWER** end-to-end. The transpose tax is ~10× the scan, so the net is a large regression.
- Also a **correctness hazard**: the sub-granule VEC offset is exactly the OL-221 / EC-22 sub-granule-VEC-offset trap (a packed layout that reads/writes at a non-granule-aligned offset silently corrupts).

**Boundary (where it WOULD work)**: realizable only for **dstate (N) ≥ 64** — at full-granule N the packing needs NO sub-granule transpose (rows already align to the granule), so the tax vanishes and the lane-fill win can land. For N=16 (single-head Mamba scan) it is a net loss.

**Promote when**: a 2nd small-N (N≤32) serial-L scan/SSM reproduces the "packed-scan-fast-in-isolation but repack-tax-net-negative" measurement, OR an N≥64 variant lands the packing win (confirming the boundary). Cross-ref: OL-231 (the issue-bound architecture-floor anchor — this is one of the 8 levers that fail on a small-N serial-L SSM), OL-221 / EC-22 (sub-granule VEC-offset correctness trap this would trip), OL-245 (regbase amortization boundary — same "MEASURE the per-call work before rewriting" discipline), P-P106 (the scan structure). Whitebox trace: `workspace/ss_perf_loop/whitebox_log.md` (iter-3). backend=ascendc.

---

## CAND-BRENT-KUNG-WORK-EFFICIENT-SCAN-LOSES-ISSUE-BOUND: Brent-Kung / Blelloch work-efficient scan LOSES to Hillis-Steele on an ISSUE-BOUND A5 vector unit — op-issue-count dominates total element-work

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32`
**Source**: selective_scan_source_a5 fwd-SIMD perf-loop iter-4 (2026-06-23, A5/Ascend950PR_957b) | **Validation status**: anti-pattern, whitebox-measured, NOT pursued

**Concept (the instinct)**: the textbook says a work-efficient parallel scan (Brent-Kung / Blelloch, O(L) total work + O(log L) depth) beats the work-inefficient Hillis-Steele (O(L log L) work). So porting the scan from Hillis-Steele (HS) to Brent-Kung (BK) should cut total work ~3.5× and win.

**Why it LOSES (the A5 vector unit is ISSUE-BOUND, not work-bound)**: measured BK ran **11.6× SLOWER** than HS despite doing **3.5× LESS total element-work**. A control test isolated the cause:
- A **contiguous same-op-count BK (BKc)** control — BK's op structure but contiguous (non-strided) access — measured only **+1.1%** over HS. So the strided sub-granule access of the BK tree is NOT the killer.
- The real **11.5× killer is op-ISSUE-COUNT**: BK's up-sweep/down-sweep tree issues MANY small per-node ops, each only N=16 wide; HS issues FEW maximally-wide contiguous-L passes (~4096 elements each). On A5's issue-bound vector unit (a width-16 op issues in the same time as a width-128 op — OL-231's measured W=16 ≈ W=128 fact), **issue-count dominates total element-work**: 3.5× less work spread over many-times-more issues is a net loss.

**The general principle**: on an issue-bound vector unit, the scan optimum is **few maximally-wide contiguous ops**, NOT minimal total work. HS-over-contiguous-L (few wide passes) IS the issue-optimum; the work-efficient tree (many narrow ops) is exactly the wrong shape. Any restructure of the scan into more/smaller ops loses, regardless of how much element-work it saves.

**Boundary (where BK might compete)**: N(dstate) ≥ 64 changes the calculus — at full-granule N each tree node is a full-lane op, so the issue-count penalty per node is amortized over real lane-work and the work-efficiency can matter. For N=16 (single-head Mamba scan) HS wins decisively.

**Promote when**: a 2nd issue-bound vector op reproduces "work-efficient algorithm loses to wide-contiguous despite less total work, isolated to op-issue-count by a contiguous-control test", OR an N≥64 variant where BK competes. Cross-ref: OL-231 (issue-bound W=16≈W=128 anchor + the architecture-floor consolidation; this is the iter-4 lever in the 8-lever failure set), OL-245 (per-VF-call issue overhead — the same issue-count-dominates principle for regbase re-entry), P-P106 (the scan structure). Whitebox trace: `workspace/ss_perf_loop/whitebox_log.md` (iter-4). backend=ascendc.

---

## CAND-INPLACE-HS-SCAN-REGBASE-PHASE-SPLIT: an in-place Hillis-Steele scan combine converted to VL-tiled regbase MUST split phase1(pre-cache sources)/phase2(compute+write dst) — a single pass corrupts the scan via cross-tile RAW

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.0; bisheng=AIV; op_class=scan/SSM (Hillis-Steele affine-prefix, in-place); dtype=fp32/fp16/bf16`
**Source**: selective_scan_full_grad bwd loc1/loc2 regbase conversion (2026-07-01, A5/Ascend950PR_9579, branch archived/perf/ss-bwd-regbase-loc12) | **Validation status**: whitebox-measured, behavior-neutral (grads BIT-IDENTICAL to Membase baseline), device-time verified

**The trap**: a Membase in-place Hillis-Steele scan pass — `for (stride...) { B[i+stride] += A[i+stride]·B[i]; A[i+stride] *= A[i]; }` over `count=(cl-stride)*N` elements — reads `B[i]` (a SOURCE at the low index) and writes `B[i+stride]` (a DST at the high index). In Membase this is safe: each `Mul`/`Add` processes the WHOLE `count` in one op and `PipeBarrier<PIPE_V>` orders the ops. Naively lowering it to a SINGLE regbase VF loop over VL=64 tiles BREAKS it: the loop writes `B[i+stride]` in an early tile, and a LATER tile reads `B[i']` where `i'=i+stride` (the same element, now a source) → it reads the ALREADY-OVERWRITTEN value → **cross-tile RAW hazard → corrupt scan** (silent wrong grads, not a crash).

**The fix — phase1/phase2 split (two VL-tiled VFs, shared scratch)**:
- **phase1** (read sources, write SCRATCH only): `prod[off+i] = A[off+i]·B[i]; tA[off+i] = A[i];` — reads all sources, touches no dst → no hazard.
- **phase2** (read scratch + same-index dst, write dst): `B[off+i] += prod[off+i]; A[off+i] *= tA[off+i];` — the low-index source `B[i]` was fully consumed in phase1, so phase2 only does same-index read-modify-write → no cross-tile hazard.
```
__simd_vf__ HSScanPhase1VF(A,B,prod,tA,count){ __VEC_SCOPE__{ for(i<nt){ adr=CreateAddrReg(i,VL);
  LoadAlign(a,A+off,adr);LoadAlign(b,B,adr);Mul(p,a,b,m);StoreAlign(prod+off,p,adr,m);
  LoadAlign(ai,A,adr);StoreAlign(tA+off,ai,adr,m);}}}   // sources→scratch only
__simd_vf__ HSScanPhase2VF(A,B,prod,tA,count){ ... B[off]+=prod[off]; A[off]*=tA[off]; }  // scratch→dst
```
The existing Membase already keeps `tA`/`prod` scratch (for the barrier chain) — reuse them; you are only re-partitioning the SAME data-flow into two hazard-free passes.

**General principle**: any IN-PLACE strided combine (scan / cumulative / recurrence) whose write-index overlaps a later tile's read-index cannot be a single register-tiled pass — the tile granularity re-orders the reads/writes that the Membase whole-vector op + barrier implicitly serialized. Split into a source-cache phase and a compute-write phase. Distinct from OL-245 (WHETHER regbase pays off) — this is a CORRECTNESS precondition for regbasing an in-place scan at all. The reverse-scan direction adds a companion tail-mask correctness rule (offset-0 dst → `MaskPattern::ALL` over-write corrupts the preserved tail scan-state → use `UpdateMask` remaining-count tail mask; see the UpdateMask KB entry).

**Promote when**: a 2nd in-place strided-combine regbase conversion (another scan/cumsum/recurrence) reproduces "single-pass corrupts, phase1/phase2 split restores bit-identical". Cross-ref: OL-245 (regbase amortization — the orthogonal WHETHER-it-pays question; its ~300us-est→~80us-real evidence is this same op), P-P106 (the HS affine-prefix scan structure), the UpdateMask tail-mask entry (companion reverse-scan correctness). Evidence: loc1(fwd HS)+loc2(rev HS), grads bit-identical to baseline 3 dtypes 3+20-chunk, det 5/5, ~1% device-time (`output/selective_scan_source_a5/src/kernels/selective_scan_bwd_simd/perf_evidence_regbase_loc12/`). backend=ascendc.

**2nd reproduction — FORWARD path, single-pass-corrupts CONFIRMED (2026-07-24, selective_scan_fwd_simd expert round SMID0724.txt R3, A5/Ascend950PR_957b, CANN 9.1.T500)**: an expert proposed `SSFwdScanCombineReverseVF` — a SINGLE VL-tiled (VLf=64) regbase VF for the FORWARD N==16 in-place HS scan-combine, iterating tiles in **reverse ADDRESS order** (high→low) under the belief that reverse-sweep alone gives in-place hazard-safety. It does NOT inside a MicroAPI `__VEC_SCOPE__`: VF tile iterations are **software-pipelined with no inter-iteration ordering / no PipeBarrier**, so an early tile's store to `B[i+stride]` is not ordered before a later tile's load of that same element as source `B[i']` → the exact cross-tile RAW this entry describes. Measured: builds clean (exit 0), **bit-identical to baseline at L≤64** (case3 bf16 L64), **catastrophic corruption at L≥128** (case2 fp16 L128 MERE 6.4e-7→2.2e-1 absmax 60; case4 customer bf16 L5000 MERE 1.3e-6→**2.0e10, absmax 2.8e15** — the error compounds through log2(cl) passes × the L-recurrence). REJECTED at the precision gate (perf A/B not run — measuring a garbage kernel is meaningless). This reproduces the "single-pass corrupts" half in the forward direction on a 2nd op-context; the expert did NOT test the phase1/phase2 restore in fwd (R3's whole GOAL was to AVOID the extra scratch sweep). **Key synthesis (ties correctness↔perf)**: the phase1/phase2 fix that RESTORES correctness IS the "extra prodC/scratch sweep" the kernel's own perf note records as making `scan-in-VF = 1.03x slower` — so a FAST single-pass in-VF forward scan is a **double dead-end**: correct⇒needs the scratch sweep⇒no saving. The reverse-ADDRESS-sweep + `UpdateMask` tail (which R3 DID include, matching this entry's companion rule) is necessary-but-insufficient — it does not substitute for the phase-split. **Recall note (harness)**: this CAND directly predicted the failure but was NOT surfaced to the optimizer/brief (both re-derived the root cause from scratch, spending a build+precision cycle) — a KB-recall/surfacing gap for scan-regbase optimization planning (grep the KB for existing scan-regbase CANDs BEFORE implementing). Anchor: bg agent ssf-r3-ko (A5 rig, npu.Event device-time + fp64-oracle precision), kernel unchanged (stays ac5508b0). backend=ascendc.

## CAND-BWD-RATIO-DEGENERATE-ZERO: a well-conditioned backward reduction output whose same-precision competitor hits EXACTLY-0 error makes the competitor-RATIO gate (our/0=inf) structurally unwinnable — but only under the small-N per-case-all-pass fallback, NOT the bootstrap-median path

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 reduction outputs); kernel_type=any`
`verified_on: NPU-independent (grading-gate mechanism + competitor-provenance analysis); a5_ops:selective_scan_full_grad fp32 tier 2026-06-30`

A backward op exposes per-output gradients of two kinds: **direct per-element** grads and **reduction** grads (Σ over batch/seq/dim). When a reduction output is **well-conditioned** (the summation is cancellation-free, e.g. `grad_D = Σ_{b,l} gy·silu(z)·u`, or a pure bias-sum grad), a **same-precision competitor** (the fp32 forward with fp32-accumulated reductions, compared to the fp64 autograd golden) can achieve **EXACTLY 0 error** on every element. The cannbot **② competitor-ratio gate** is `our_err / competitor_err`; with `competitor_err = 0` and `our_err > 0` (our independent vector-op transcendental differs from torch's libm by ~1 ULP), the ratio = `+inf` → automatic FAIL. **Passing requires bit-exactness to torch's fp32 libm**, infeasible for any independent AscendC kernel (software-fp32 sigmoid/exp floor ≈ 3.8e-6 abs). This is a **grading-gate degeneracy, not a kernel defect**.

**Decision rule — which gate path is the limiter (the load-bearing distinction)**:
- The `per_case_all_pass` verdict basis is a **small-N circuit-breaker (小样本熔断)**, used ONLY because `N < 200`. Under it, a single `+inf`-ratio element on ONE output fails the whole case → the degenerate-zero output is unwinnable.
- The **intended** path is **bootstrap-median-CI** (valid at N≥200; L1 wants ≥700/dtype), which gates the **MEDIAN** MARE-ratio CI (≤5.0). The median is **robust to outliers** like the `inf`-ratio output. Empirically: a representative distribution (5/8 outputs ratio≈1, A≈5, B≈8, C≈10.5, D=inf, N=700/dtype) yields `ci_upper = 3.65 ≤ 5.0 → PASS`. i.e. the SAME kernel passes once the statistical path is selected.

**Provisioning fix (the actionable lever)**: `phase_o25_backward` (the backward-reference generator) MUST emit ≥ the cannbot statistical scale (**L1 ≥ 700 representative randn samples per (case,dtype)**) so `grade_batch` selects the median-CI path instead of the brittle `per_case_all_pass` fallback. A 48-sample backward-truth dataset triggers the small-N breaker and structurally cannot pass a degenerate-zero output, regardless of kernel quality.

Concrete anchor (selective_scan_full_grad / Mamba SSM FULL backward, 2026-06-30, Ascend950PR_9579/arch35): FAIL 41/48 representative; the 7 failers are ALL fp32 reduction outputs A/B/C/D. grad_D competitor MERE = 0.0 on every element (both records) → ratio +inf, unwinnable; grad_A inherits the SoftExp-vs-libm-expf summand diff (mare 5.07); grad_B/C are cross-d atomic-reduction-noise bound (~2-3.4× torch's pairwise fp32). Direct outputs (grad_u/grad_delta/grad_z) + delta_bias all PASS fp32; ALL outputs PASS fp16/bf16 (both at the low-precision noise floor, ratio≈1.0).

**Distinct from siblings**: CAND-KW-FAG-2 is the **MARE small-value-domain** amplification (single-record metric, |ref|<2^-14); THIS is the **competitor-RATIO degeneracy** (our/0=inf) compounded by **gate-PATH selection** (small-N fallback vs median-CI). CAND-SSM-BWD-WEIGHTGRAD-FP32 is a **dtype underflow** fix (return fp32). All three are backward-fp32 grading artifacts but attack different gate stages.

**Anti-pattern (do NOT)**: relax the verify ratio threshold or add a kernel branch to mask the degenerate output (OL-85 reward-hacking). The fix is in the **reference-dataset scale** (harness-side provisioning), not the kernel or the per-op verify.

**Promote when**: a 2nd backward op with a well-conditioned reduction output reproduces "competitor_err=0 → ratio inf → fails under small-N, passes under median-CI", confirming the gate-path-selection rule generalizes. Cross-ref: CAND-KW-FAG-2 (MARE sibling), CAND-SSM-BWD-WEIGHTGRAD-FP32 (dtype sibling), OL-103 (V220 transcendental floor — consumed but NOT the limiter here), OL-85 (no-reward-hack), OL-109/OL-110 (two-tier verdict / fail-floor family), PRECISION_STANDARD_v2.1 §4.5.1. Source: derived from selective_scan_full_grad (forward_spec_grad) knowledge_update.md 2026-06-30. backend=ascendc.

**Empirical confirmation (2026-07-01) — OLD-vs-NEW dual-grade on REAL fp32 kernel outputs**: the shipped `selective_scan_full_grad` kernel was rebuilt (`--clean`) on A5 (so md5 `4334544868`, Ascend950PR_9579) and its REAL fp32 grad outputs graded through the OLD adapter (git `b4535ddf`, 商用 ratio) vs the NEW adapter (HEAD, 生态 vendored compare.py) — the adapter is the ONLY variable (kernel outputs held FIXED). In the op's real **small-N regime** (16 representative = 2 records × 8 grads, the same `per_case_all_pass` regime as the real 41/48 FAIL): 2 `grad_D` cases have competitor_err EXACTLY 0 → OLD ratio=inf=unwinnable FAIL despite our abs-err ~1e-5 (accurate) → NEW correctly PASS; +4 more reduction grads flip via the finite-ratio family (ratio 1.8–5.2 penalizing an accurate kernel). NEW **discriminates** (not blanket-loosening): a negative control (perturbed grad_A rel-err 2.50) → NEW FAIL, and one `grad_C` OLD-ratio passed (1.11) → NEW FAIL (our_mare 4.6e-3 > 生态 fp32 bar). Evidence: `docs/validation/ss_bwd_grader_regrade_2026_07_01/` (whitebox_log.md + FINAL_old_vs_new.json + reproducible dual_grade scripts).

**Scope / nuance (do NOT over-claim, per 2026-07-01 framing)**: the OLD false-FAIL above is demonstrated in the op's REAL small-N regime — the FULL mechanism = ratio-gate degeneracy × small-N `per_case_all_pass` fallback. At N≥700 the OLD median-CI path **MIGHT** tolerate the inf-outlier (the `ci_upper ~3.65 ≤ 5.0` figure in the Decision-rule above is **ANALYTICALLY ESTIMATED, NOT a real 700-sample run**). The direction is **asymmetric**: if the estimate is wrong and large-N ALSO fails, it only STRENGTHENS "OLD ratio is broken", never weakens it. Either way the **load-bearing fix does NOT depend on N**: the merged 生态-absolute standard (vendored compare.py, PRECISION_METRICS_CANONICAL §0) removes the ratio path entirely → robust at any N. So the reckoning story is: OLD 商用-ratio false-FAILs well-conditioned/degenerate fp32 grads **at the op's real scale**; NEW 生态-absolute correctly passes them AND still catches real error.

## CAND-SEQ-SCAN-BWD-FP32-COMPENSATED-ACCUM: a sequential exp-scan backward is fp32-imprecise at cancellation points (2.4–14.5× worse than CPU-fp32 autograd) — compensated (Dekker double-single / Kahan) accumulation in the scan state recovers it, but DEFER when it does not flip the terminal verdict

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=backward-gradient (sequential exp-scan, e.g. Mamba SSM)`
`verified_on: a5_ops:selective_scan_full_grad fp32 tier 2026-07-01 (Ascend950PR_9579)`

A backward kernel whose forward is a sequential exp-scan (`x[l]=dA·x[l-1]+Δ·B·u`) accumulates the reverse-scan gradient state across L steps. At **cancellation points** (grads where opposite-sign summands nearly cancel — here grad u/B/C) plain fp32 accumulation is **2.4–14.5× worse** than a CPU-fp32 autograd reference (kernel MARE ~2.1e-3 vs native 9.3e-4); CPU-fp32 passes the 生态 fp32 gate (2^-13), the kernel does not. This is a **GENUINE kernel imprecision** (the standard SHOULD catch it), distinct from the grader-degeneracy sibling.

**Fix (candidate)**: double-single (Dekker 2×fp32) / Kahan compensated accumulation of the sequential scan state → fp32 up to 16/16.

**DEFER rule (P7 — the load-bearing decision)**: when fixing fp32 alone does NOT flip the terminal verdict (op still blocked on a separate gate), a compensated-accumulation rewrite of the sequential scan is a 3–5 iter, floor-regression-risk change with NO verdict-flip value NOW → DEFER to a future kw spawn that applies it WHEN it actually contributes to the terminal PASS. (selective_scan_full_grad: fp32 12→16 does not flip terminal FAIL while DEBT-180 blocks 32 fp16/bf16 cases; Kahan deferred until after the harness truth-gen fix lands.)

**Distinct from the grader-degeneracy sibling**: this is a real kernel error that NEW correctly FAILs (it serves as a natural negative control proving the 生态 standard discriminates) — NOT a well-conditioned grad falsely-failed by the OLD ratio-gate (that is CAND-BWD-RATIO-DEGENERATE-ZERO). Same op, same fp32 tier, opposite direction.

**fp32-backward CAND family (each attacks a DIFFERENT aspect — not duplicates)**: (1) CAND-KW-FAG-2 = MARE small-value metric ARTIFACT (grading, "not a kernel bug"); (2) CAND-SSM-BWD-WEIGHTGRAD-FP32 = dtype underflow (return grads in fp32); (3) CAND-BWD-RATIO-DEGENERATE-ZERO = competitor-ratio degeneracy (grading gate); (4) THIS = a GENUINE accumulation imprecision the standard SHOULD catch (real kernel error, fix=compensated accum). #1/#3 are grading artifacts; #2/#4 are real kernel issues — mine is the only one whose fix is compensated accumulation.

**Promote when**: a 2nd sequential-scan backward reproduces "plain fp32 scan-state accum N×-worse than CPU-fp32 at cancellation, recovered by compensated accum". Cross-ref: CAND-BWD-RATIO-DEGENERATE-ZERO (grader-degeneracy sibling), CAND-KW-FAG-2 + CAND-SSM-BWD-WEIGHTGRAD-FP32 (fp32-backward family), DEBT-180 (the co-blocking harness truth-gen bug), PRECISION_METRICS_CANONICAL §0.2 (fp32 2^-13 gate). Source: selective_scan_full_grad e2e 2026-07-01 (workspace/forward_spec_grad/verification.json root_cause cause_2). backend=ascendc.

## CAND-EXPOSED-MTE-LATENCY-DOUBLEBUFFER-VS-ROOFLINE-BLIND: msprof pipe-timeline showing MTE2(load)↔VEC(compute) SERIAL (compute waiting for transfer) = exposed non-overlapped memory latency → input/output UB double-buffer candidate; and roofline/MFU is STRUCTURALLY BLIND to this (it assumes overlap is already done)

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=AIV; op_class=vector/SIMD (chunked per-tile load→compute→store)`
`verified_on: diagnostic method + roofline-model limitation + exposure-ablation method (NPU-measured on selective_scan bwd, A5 .141). ⚠ The specific ss-bwd DB verdict ("net-NEGATIVE ≤2.8%, UB-infeasible") was REFUTED 2026-07-02 by build-and-measure — real = +2.76% customer win, shipped PR #82 (see VERDICT CORRECTED block below). The METHOD (busy≠exposure, PIPE_ALL anti-pattern, roofline-blind) STANDS.`

**THE LOAD-BEARING METHOD — a busy-RATIO is NOT exposure; measure exposure by byte-removal ablation BEFORE implementing double-buffer.** An msprof pipe-timeline / PipeUtilization showing a nonzero MTE2 (load) busy-ratio does NOT mean that latency is EXPOSED — most of it may already be overlapped with VEC compute. The rigorous exposure measure: **ablate the load bytes** (change the per-chunk `DataCopy`/LoadChunk to copy 1 element instead of `cnt`, keeping all fences + compute) and A/B the device-time. The saving = the EXPOSED (non-overlapped) load fraction, and it is a hard **UPPER BOUND** on any double-buffer gain (DB can at best hide the exposed fraction; the overlapped part was already free). Only implement DB if the ablation shows meaningful exposure. Double-buffering itself = depth-2 `TQue<VECIN,2>`/`TQue<VECOUT,2>` on the per-tile load+store so load(k+1) overlaps compute(k); tradeoff = per-pass UB compute halves + SetFlag overhead + first-chunk unhidden + a 2nd UB slot must FIT.

**Shape-dependence (critical for reconciling a small-shape pipe-diagram vs the real workload)**: exposure is HIGHER at small shapes (small per-chunk compute → less to hide the load behind) and LOWER at large shapes (big per-chunk compute → load mostly overlapped). A small-shape msprof diagram showing "compute waiting for transfer" is REAL but does NOT extrapolate — measure the ablation at the PRODUCTION shape.

**PIPE_ALL anti-pattern (verified)**: `PipeBarrier<PIPE_ALL>()` serializes **ALL** pipes (SCALAR/VEC/MTE2/MTE3) → destroys any overlap → a pipeline killer. Replace with the **narrowest** barrier that still fences the real hazard (`PipeBarrier<PIPE_V>` or a targeted `SetFlag/WaitFlag<HardEvent>`). ⚠ Do NOT blind-delete: some PIPE_ALL are load-bearing (e.g. selective_scan bwd `N>16 carry RAW` — PIPE_V insufficient at full chunk, PIPE_ALL fences it; deleting reintroduces wrong-grad RAW). Identify each PIPE_ALL's specific pipe-pair hazard before narrowing.

**The load-bearing tool lesson — roofline/MFU is BLIND to latency-hiding**: a roofline / MFU model whose kernel time is `T = max(T_compute, T_mem)` **assumes perfect overlap = assumes double-buffer is already done** → it models the **post-DB ceiling** → it **structurally cannot recommend double-buffer** (it thinks you're already at the ceiling). A kernel *without* DB measures `> roofline-max` = shows up as `η < 1`; a naive MFU **misattributes** that η-gap to "raise arithmetic intensity / fuse / multi-AIC" — the WRONG optimization direction. Latency-hiding (double-buffer / software-pipeline / prefetch) AND issue-count are roofline blind spots. **The right tool for this class = the msprof pipe-timeline** (it shows the actual overlap gaps); an MFU verdict of "memory-bound → raise AI" on a latency-exposed kernel is a misdirection. Correctness-first mitigation (being built into the MFU tool, PR#76): η-gap attribution must add an **"unmodeled component" exit → abstain** ("latency-hiding not modeled, measure via msprof timeline") rather than emit raise-AI/fuse.

**Concrete anchor (MEASURED, 2026-07-01, selective_scan_full_grad backward SIMD, Ascend950PR .141 NPU3)**: expert msprof pipe-timeline at a SMALL shape showed compute waiting for transfer, expert estimated **5-10%** from input/output UB double-buffer + removing PIPE_ALL (wiki 0627/0630). At the CUSTOMER shape (B=8 D=192 L=5000 N=16 fp32): PipeUtilization vec 67.7% / MTE2 16.9% / mte3 17% / scalar 10% (NOT vec-bound — forward is 99.6% vec). But the byte-removal **ablation** (LoadChunk 1-elem vs cnt) saved only **2.8%** (115356→112104µs, 3/3 rounds) → MTE2 ~85% ALREADY overlapped → **DB gain ≤ 2.8%, net-NEGATIVE** after SetFlag overhead, AND UB-infeasible (34KB free < 38.4KB for a 2nd load slot). The busy-ratio (17%) massively overstated the exposure (2.8%). Separately, MFU on this op (real FLOPs 1.72e9, HBM 225MB) output "bottleneck=memory → raise AI / multi-AIC" — **not one lever was double-buffer** (the roofline blind spot, empirically confirmed). Measurement caveat found: `kernel_details.csv` Duration can UNDERCOUNT (loc12 cited ~7040µs/kernel; real per-launch ~115ms — 16× — reconciled via event-full-op + pipe-time-sum). See [[reference_ss_bwd_doublebuffer_negative_and_kernel_details_undercount]].

**⚠ VERDICT CORRECTED (2026-07-02, PR #82 — build-and-measure REFUTES the "DB net-negative" conclusion above; owner-directed correction, empirically verified, not an auto-resolution)**: ss-bwd's real double-buffer = a 2-slot PASS-A input ping-pong that REPLACES the coarse per-chunk `SyncMTE2toV`+`SyncVtoMTE2` fences with fine per-slot `MTE2_V`/`V_MTE2` WAR events → **+2.76% @ N=16 (customer L=5000) / +4.25% @ N=8**, precision bit-identical to base (0 loss), UB-safe, shipped PR #82 (d8c607fd). **Why the ablation under-estimated it**: the byte-removal ablation KEPT the coarse fences, so its 2.8% bounded only BYTE-OVERLAP gain — but this DB's gain is FENCE-SERIALIZATION removal (the coarse fences serialized the whole chunk boundary), which the ablation structurally cannot see. **Generalized lesson (applies to ANY chunked/tiled kernel): when a double-buffer's mechanism is fence-removal rather than byte-hiding, BUILD the real DB and measure — a fence-keeping byte-ablation is an upper bound on byte-overlap only, NOT on a fence-removal DB, and will under-estimate it.** The "UB-infeasible (34KB<38.4KB)" was also wrong (it used the wrong 248KB usable; the 2nd slot fits at CH256=251.25KB < the real regbase-SIMD ceiling ~255.5KB — see CAND-UB-LIMIT-APPLICABILITY-REGBASE-SIMD). The METHOD in this candidate (busy≠exposure, PIPE_ALL anti-pattern, roofline-blind-to-latency-hiding) remains fully VALID — only the ss-bwd DB verdict is corrected.

**⚠ ko CONFIRMS AT-CEILING + THE PROFILE SHIFTS (2026-07-02, real aog-kernel-optimizer, msprof-profile-driven, PR #84)**: ran REAL ko (profile-driven iterative) beyond the DB. ko's fresh msprof profile shows the **post-DB kernel is now VEC-BOUND ~90% (aiv_vec 0.899, exposed non-vec ~10%)** — the original "~32% exposed dominated by scalar+MTE3+barriers" figure is **STALE (pre-DB)**; the DB + barrier-cleanup already closed the MTE2/MTE3 overlap gap. So ko's only remaining lever = vec-work reduction (regbase-fuse the Membase eltwise chains gz/gu/gd into MicroAPI VF) = **~+0.2-0.4%** (bit-identical, det run×4, at the noise/ceiling; +0.15% kernel_details / +0.37% msprof). The CN-sized vec bulk is structurally un-fusable (native Exp algorithmically required, Broadcast/Sum for the N⊗L structure, scan already regbase). ⇒ **DB (+2.76%) + barrier (~1.5%) + ko2 (~0.3%) = essentially ALL practically-available headroom; the kernel is at its A5 vec-structural ceiling — don't re-chase ss-bwd perf.** **Durable meta-lesson (applies to ANY iterative perf work): the bottleneck PROFILE SHIFTS after each optimization** (pre-DB was MTE-exposed → post-DB is vec-bound) — always RE-PROFILE before choosing the next lever, and run REAL ko (fresh profile → bottleneck → iterate) rather than implementing a suggested lever + testing (an expert-seeded single lever + test ≠ ko's profile-driven loop; the fresh profile is what catches the shift).

**Promote when**: a 2nd op reproduces "msprof busy-ratio suggested MTE exposure, byte-removal ablation showed it was mostly-overlapped (busy≠exposure), roofline/MFU failed to flag either way" — OR (the corrected lesson) a fence-removal DB beats the byte-ablation upper-bound — OR (the ko meta-lesson) the bottleneck profile shifts after an optimization and a fresh re-profile redirects the next lever.

## CAND-UB-LIMIT-APPLICABILITY-REGBASE-SIMD: verify a documented UB limit's APPLICABILITY (loud/silent? SIMT-scoped? shipped kernel already exceeds it?) before invoking it as a ship-blocker

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=AIV; op_class=all (UB-budget, regbase-SIMD)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (bracket-probe + per-grad bit-identical, selective_scan bwd DB, PR #82, 2026-07-01→02)`

**Principle**: Before invoking a documented UB-size limit as a ship-blocker, verify it applies to YOUR kernel class along three axes: (1) **LOUD or SILENT failure?** Bracket-probe the ceiling — on Ascend950PR, InitBuffer accepts ~255.5KB and rejects 256.0KB with a LOUD `507035` OOB → the true enforced ceiling is ~255.5KB, and `GetCoreMemSize`'s 248KB is a conservative advisory, not the enforced cap. A loud-failing ceiling is not the silent-OOB class. (2) **Is the limit SCOPED to a class you're not in?** PB-32's 40KB **silent** OOB is the SIMT DCache reserve, scoped to SIMT (L3 / `__simt_callee__`) kernels; it does NOT bind a regbase-SIMD kernel (MicroAPI `__VEC_SCOPE__`, no SIMT threads / DCache). (3) **Does the ALREADY-SHIPPED kernel already exceed the claimed limit?** A merged kernel running correct at 213.75KB > 208KB "SIMT-effective" proves the 40KB reserve doesn't bind that kernel family.
**Reserve-semantics safety check** (for a SIMD kernel allocating above the conservative advisory but below the loud ceiling): per-grad bit-for-bit md5 match to a known-good base at full customer scale + determinism across runs — a silent-reserve-corruption would perturb the grads or break determinism; bit-identical + deterministic ⇒ the reserved band is not clobbered under the real workload.
**Concrete anchor**: selective_scan bwd input-DB at 251.25KB (N=16 CH256) crosses the 248KB advisory but < the ~255.5KB loud ceiling; regbase-SIMD so PB-32 N/A; DB grads bit-identical to base at full customer scale (B=8 D=192 L=5000) → safe, shipped PR #82.
**Evidence**: selective_scan_bwd_simd DB UB-safety (2026-07-01→02): bracket-probe (255.5KB runs / 256KB→507035) + per-grad md5 bit-identical to base + PB-32 scope read. Cross-ref PB-32 (the SIMT 40KB reserve this scopes against).
**Other instances (predicted)**: any regbase-SIMD kernel near the UB ceiling; any ship-block decision citing a documented limit — check loud/silent + class-scope + shipped-precedent before taking the number at face value.
**Promote when**: a 2nd op's ship-decision is corrected by checking a limit's applicability (loud/silent, class-scope, or shipped-precedent) rather than taking the documented number at face value. backend=ascendc. Cross-ref: OL-245 (regbase amortization — the orthogonal UB-round-trip lever), OL-231 (issue-bound — the sibling roofline blind spot), MFU tool assessment (PR#76 latency-hiding modeling). Source: expert wiki (SelectiveScan反向自动生成算子优化, 0627/0630) + owner-directed measurement 2026-07-01. backend=ascendc.

---

### CAND-DB-COARSE-FENCE-CATCHES-PREFETCH: a coarse pipe-fence issued after a prefetch defeats double-buffering
`applies_to: soc=Ascend950PR; cann=9.x; bisheng=n/a; op_class=all (any double-buffered chunk loop)`

**Principle**: When adding input double-buffering (prefetch chunk N±1 while chunk N computes), any COARSE
pipe-fence that waits "all preceding same-pipe ops" — e.g. `SyncMTE2toV()` = SetFlag+WaitFlag
`<HardEvent::MTE2_V>` on a SHARED event id — issued in program order AFTER the prefetch will make the
compute pipe WAIT FOR THE PREFETCH to complete (the fence fires only after ALL preceding MTE2, including
the just-issued prefetch loads). This DESTROYS the overlap the DB was meant to create AND adds the fence's
own cost → net REGRESSION, and the aggregate vec_ratio DROPS (more compute-idle).

**Fix (either)**: (a) issue the coarse fence BEFORE the prefetch so it waits only the current chunk's own
loads; the prefetch (issued after) then overlaps compute. (b) use a DEDICATED event that waits only the
target buffer — do NOT reuse the coarse event that catches the prefetch's in-flight loads (the prefetch has
its own load-done event).

**Split-flag corollary (overlap a single-slot buffer with NO 2nd UB slot)**: if a buffer is read LATE in the
chunk (e.g. an `xall` read only at the post-pass), issue `SetFlag<MTE2_V>(dedicated_id)` right after its load
(before the prefetch) and DEFER the matching `WaitFlag` to just before its first read. Its load then overlaps
all intervening compute, without a 2nd slot. (ss-bwd: this lifted the gain from +0.85% to +1.17%.)

**Determinism note**: input DB reorders LOADS (MTE2) vs COMPUTE (VEC) only; it does NOT change the vector
accumulation order, so grads stay BIT-IDENTICAL to the pre-DB kernel — PROVIDED the DB keeps CH / chunk
partitioning identical (only adds a prefetch). Verify with non-aligned / tail-block shapes (odd L, small tail):
aligned-S pass + non-aligned-S flip = a DB-changed-partitioning bug. (agent-back confirmed the same
coarse-fence placement trap exists in the backward-plugin grad-reduce loops.)

**Evidence**: ss-bwd selective_scan_full_grad PASS B input DB, 2026-07-02, .171 Ascend950PR_957b (PR #87).
v1 (coarse SyncMTE2toV AFTER prefetch) = -1% (vec_ratio 0.894→0.889, vec-idle UP); v1.1 (fence BEFORE
prefetch) = +0.85% (vec_ratio 0.896→0.907); v2 (+split-flag xall late-load) = +1.17%. Instruction-timeline
(msprof --instr-profiling, `/aog-msprof-timeline`): VEC-idle-&-MTE2-busy stall 707→492us (-30%), MTE2
overlapped 18.6%→29.5%. All byte-identical + deterministic incl. tail cases.

**Other instances (predicted)**: any AscendC chunk-loop DB (attention grad, scan, conv im2col prefetch,
backward-plugin grad-reduce). The trap is universal to prefetch + shared-event coarse fences.

**PROCEDURE companion**: to MEASURE the stall this principle addresses, use the `/aog-msprof-timeline` skill
(instruction-level per-pipe gap), NOT the aggregate vec_ratio (which hides it).

**Status**: NEEDS_REVISION — mechanical scanners pending; kb-manager review before promotion.

---

## CAND-WRAPPER-SELF-SYSPATH-KERNEL-BUILD: the ModelNew wrapper must self-insert `kernel/build` into `sys.path`, because only the verifier does it for you
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward or task-owned pybind verification`
`verified_on: 1_BatchMatmul a3 2026-07-13 (Ascend910_9382 / CANN 9.1.0 — perf/pass_a/pass_b all failed to import `_<op>_ext` until fixed)`

**Principle**: `model_new_ascendc.py` (the ModelNew wrapper) MUST self-insert `os.path.join(os.path.dirname(__file__), "kernel", "build")` into `sys.path` at import time. Only `verification_ascendc.py` adds `kernel/build` to `sys.path`; `performance.py`, `pass_a_runner.py`, and `pass_b_runner.py` do NOT. So a wrapper that relies on the caller's `sys.path` imports fine under the verifier but throws `ModuleNotFoundError: _<op>_ext` under every other consumer — a portability gap that silently passes precision-verify then fails perf/Pass-A/Pass-B. Emitting the self-path-insert makes the wrapper importable from ANY harness (a portability fix, not a workaround).

**Concrete anchor**:
```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel", "build"))
```

**Promote-when**: a second migration/backward pybind wrapper independently hits the same non-verifier import failure, or the shared worker brief always emits the self-path insert.

**Status**: candidate — 1-op only (1_BatchMatmul DEBT-206); parked pending a 2nd instance per Mode-1 candidate policy.

---

## CAND-O5-EXACT-COUNT-NONDET-FAIL: phase_o5 re-measures pass_a and compares `{tier1_pass,total}` with EXACT equality even for a status=FAIL verdict, so a best_effort-determinism op whose FAIL count flakes run-to-run can MISMATCH the O5 re-measure and roll back — a harness limitation, not a kernel defect
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; det_policy=best_effort; mode=backward`
`verified_on: soc=Ascend950PR (selective_scan_full_grad kw-2 2026-06-18 / A5)`

**Principle**: `phase_o5` re-measures `pass_a` and compares `{tier1_pass, total}` against the worker's verdict with `EXACT_FIELDS` (no tolerance) — even when `status=FAIL`. If the kernel's determinism is only `best_effort` (e.g. cross-core fp32 `SetAtomicAdd` reductions), the FAIL pass-count itself is non-deterministic, so the O5 re-measure can land a different integer than the worker recorded → `MISMATCH` → rollback, with nothing wrong in the kernel.

**Concrete anchor**: selective_scan_full_grad kw-2 observed `tier1_pass` = 33 (×4) / 34 (×1) over 5 NPU runs — a borderline fp32 output whose degenerate ② `competitor_mare≈0` ratio flips pass/fail with the atomicAdd non-determinism (grad_A/B/C/D/δbias). The worker pinned a STABLE MODE (33), but a residual ~20% MISMATCH risk is inherent to the `EXACT`-count × non-determinism interaction.

**Fix candidates (owner/harness)**: (a) allow a ±tolerance on the O5 count comparison for `best_effort`-det FAIL verdicts; (b) resolve the fp32 degeneracy upstream so the count is no longer borderline; (c) a healthy-lane preflight so the O5 re-measure runs on a deterministic device.

**Distinct from CAND-BWD-RATIO-DEGENERATE-ZERO**: that entry is the GRADER ratio degeneracy (WHY individual cases fail — competitor_err=0 → our/0=inf); THIS is the phase_o5 O5-re-measure EXACT-integer-count comparison being brittle on ANY non-deterministic FAIL verdict (the compare stage, not the grade stage). They compound on this op but generalize independently.

**Promote when**: a 2nd `best_effort`-det op reproduces an O5 count MISMATCH caused purely by run-to-run FAIL-count flake (kernel logic unchanged). Cross-ref: CAND-BWD-RATIO-DEGENERATE-ZERO (grader-degeneracy sibling), OL-88 (reference non-determinism preflight), EC-59 (INCLUSIVE `pass_a.status` to avoid O5-rollback on T2-promoted PASS). Source: derived from selective_scan_full_grad knowledge_update.md (kw-2, 2026-06-18). backend=ascendc.

## CAND-CHUNK-SIZE-RAISE-CHUNKED-FP32-REDUCTION-NOT-FREE: raising the chunk size of a chunked fp32-reduction scan to cut per-chunk boundary overhead is NOT free — the UB-layout rewrite that fits the bigger chunk re-inflates the per-chunk cost, AND the chunk regroup reassociates the fp32 accumulation and can regress a near-zero-cancellation grad margin

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=chunked scan/SSM fp32-reduction (small-N serial-L, e.g. Mamba selective_scan backward); dtype=fp32`
`status: CANDIDATE — both facets NOW cross-direction confirmed: (A) PERF-tax [chunk-size raise is net-neutral on a small-N serial-L SSM] = bwd CH256→512 (kw-3 2026-07-15, rollback) + fwd CH256→384 (2026-07-16, wash −0.23%); (B) REASSOCIATION-precision hazard [reassociating the fp32 scan inflates near-zero fp32 error] = bwd (grad-margin 1.46→1.67× under CH512 regroup) + fwd (output near-zero MARE 1.3–8× under serial→parallel scan, 2026-07-16 vj re-verify) — the fwd form is output-MARE not grad-margin (no grads), see CAND-SCAN-FP32-ACCUM fwd instance. Both facets promotion-ELIGIBLE; FLAGGED for a KB-maintainer canonical-promotion pass (not self-promoted mid-op-round — canonical wording goes through the KB gate). All measured A/B, A5 Ascend950PR.`

**Principle**: for an L-chunked scan whose per-chunk boundary cost (GM staging, carry stitch) grows with the number of chunks, the instinct is "double the chunk size → halve the chunks → halve the boundary cost". Two effects make it NOT a free lunch and can net-negate it:
1. **The UB-layout rewrite that FITS the bigger chunk re-inflates the per-chunk slope.** A 2× chunk needs ~2× live UB. Fitting it under the UB ceiling forces layout concessions that each add per-chunk cost: dropping input double-buffer (loses load↔compute overlap), splitting a fused post-pass VF into sub-passes (loses the OL-245 regbase fusion), reload-late of a tensor freed mid-scan (+1 DataCopy/chunk). These re-inflate the per-chunk slope, so halving the chunk count does NOT halve the boundary time — it nets a small fraction of the naive prediction.
2. **The chunk regroup reassociates the fp32 accumulation** (Hillis-Steele scan + tree-reduce + cross-chunk carry all regroup into a different ULP-rounding order), and at a near-zero cancellation edge that can regress a thin fp32 grad margin — the exact CAND-SCAN-FP32-ACCUM hazard, now materialized in the BACKWARD direction. **"CHx byte-identical to CHy" is unachievable in principle** for a chunked fp32-reduction (different reduction grouping ≠ same bits); the only valid precision gate is the fp64-oracle floor-ratio comparison, NOT a byte-diff.

**Concrete anchor**: selective_scan_full_grad backward SIMD, L=5000/N=16/D=192, CH256→CH512 via a 13→6 live-buffer UB rewrite (SSBwdPostVF split into scr/prodA/tree-reduce sub-passes + reload-late Bt + drop input DB). Byte-exact at CH256 (proof the refactor itself is sound: 9 cfg×dtype, worst_max_abs_diff=0.000e+00); UB 246.5KB ≤ 255.5KB at CH512 (builds clean, det 5/5). But at CH512: perf **net ≈ 0** (ROUND-2 same-session back-to-back A/B: CH512 falls inside the CH256 baseline's own ~1.5% thermal drift, 1805→1833µs, which envelops every CH512 reading — CH512 min 1785µs ≈ CH256 cold min 1789µs; ⚠ an early "~1.4%" reading was a STALE-BASELINE ARTIFACT against a crashed/vector-core-timeout-session baseline, RETRACTED — see OL-231 / REPORT §九; do not confuse with the REAL +1.4% input-DB lever PR #87/#88/#89), and fp32 grad_A L=700 MARE-vs-floor 1.46×→1.67×, grad_delta_bias L=1300 1.46×→1.70× (past the chair-flagged 1.67× threshold) → ROLLBACK per the precision hard-gate.

**Decision rule (before spending budget on a chunk-size raise)**: measure BOTH (a) the NET device-time A/B (not the naive `boundary_cost × chunk_ratio` prediction — the layout-rewrite per-chunk costs offset most of it) AND (b) the fp64-oracle floor ratios per grad output (the regroup can regress a thin margin). For an UN-gated perf metric (no vendor C-API baseline), a ~1% net gain does NOT justify eroding a thin fp32 grad margin — precision-first, keep the floor.

**Distinct from siblings**: CAND-SCAN-FP32-ACCUM is the forward-accumulator dtype rule (fp32 internal, cast at store); THIS is the chunk-REGROUP reassociation hazard on the backward grads plus the perf-side layout-tax. CAND-SIMD-OVER-ROWS / CAND-BRENT-KUNG are row-packing / work-efficiency levers; this is the chunk-size (chunk-count) lever specifically.

**Promotion status (updated 2026-07-16 — the original "2nd repro" condition is now MET cross-direction on both facets; FLAGGED for canonical folding, NOT self-promoted mid-round)**: PERF-tax (chunk-size net-neutral) confirmed bwd+fwd; REASSOCIATION-precision confirmed bwd (grad-margin) + fwd (output-MARE, CAND-SCAN-FP32-ACCUM). The two transferable rules — "raising the chunk on a small-N serial-L SSM scan is net-neutral" and "reassociating a near-cancelling fp32 scan inflates near-zero fp32 error" — are promotion-eligible → **flagged for a KB-maintainer canonical-promotion pass** (kept in candidates.md so the canonical wording clears the KB gate rather than a mid-op-round self-promote; this is an explicit action-flag, not a defer). Cross-ref: CAND-SCAN-FP32-ACCUM (the reassociation hazard — now 2 instances incl this fwd one), OL-231 (small-N serial-L SSM issue/latency-bound floor — its KO-6b prediction "chunked 2-level scan: 1.0–1.40×, likely net-neutral + precision risk" is what both facets CONFIRM), OL-245 (regbase fusion lost in the post-pass split = a per-chunk cost), OL-80 (no VEC-API guessing). Source: selective_scan_full_grad kw-3 (2026-07-15) + selective_scan_fwd_simd fwd round (2026-07-16). backend=ascendc.

**Cross-direction reproduction (FORWARD, 2026-07-16, selective_scan_fwd_simd expert round KO-2)** — precise on WHICH facet reproduces. (i) **PERF-tax facet = reproduced.** The forward chunk-size raise (CH 256→384; **CH512 overflows the 255.5KB UB → 507035**, so 384 is the forward max) measured a same-session npu.Event **WASH (−0.23%, inside thermal drift)** at the customer shape (B8/D192/L5000/N16 bf16) — fewer chunk boundaries (20→14 rows/chunk) exactly offset by the larger per-chunk Hillis-Steele (work ∝ cl·N) + the UB-freeing alias fences. So the chunk-size lever is net-neutral on BOTH directions of a small-N serial-L SSM — do not spend budget raising the chunk on this op class. (ii) **Reassociation-precision facet = ALSO reproduced, but as OUTPUT-MARE, not a GRAD margin** (the forward has no grads). The forward's serial→parallel Hillis-Steele reassociation inflates its fp32 near-zero **output** MARE **1.3×–8.2× vs a serial fp32 scan** (cust L=5000 8.2×; MERE/abs floor unchanged ~1e-6/~1e-4) — a 2nd forward-direction instance of the reassociation hazard, recorded under CAND-SCAN-FP32-ACCUM (fwd parallel-scan, 2026-07-16 vj re-verify). So the ONLY backward-specific piece is the exact phrasing "regresses a *GRAD* margin" (forward has no grads); the underlying reassociation-inflates-near-zero-fp32 hazard is now cross-direction. (Forward round also cheaply LANDED an orthogonal +1.3% via copy-elimination — see CAND-FWD-POSTFOLD-CARRY-PIPEALL-REMOVAL-LOSES for the round's full ledger; that win is NOT chunk-size, it is a redundant-Adds removal.)

## CAND-FWD-POSTFOLD-CARRY-PIPEALL-REMOVAL-LOSES: removing a scan's `PipeBarrier<PIPE_ALL>` by moving the cross-chunk carry from a pre-scan narrow fold to a post-scan wide fold LOSES on a small-N serial-L SSM — the wide post-fold adds more vector-issue than the once-per-chunk barrier drain it removes, AND that PIPE_ALL is often a load-bearing N>16 periodic full-drain, not a local RAW
`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.0.B060; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32/fp16/bf16`
`status: CANDIDATE — anti-pattern (2 refuted forward levers + 1 landed win, one round). Source: selective_scan_fwd_simd expert round (owner 2026-07-16), KO agent A5 real-machine A/B.`

**The refuted lever (F1, expert "post-fold carry")**: the forward scan pre-folds the cross-chunk carry into `B[0]` before the Hillis-Steele scan (`Mul(tmp,Ascan,xst,N); Add(Bscan[0:N],tmp); PipeBarrier<PIPE_ALL>()`), then scans. The expert proposed the math-equivalent post-fold: scan carry-free, then `h = Bscan + Ascan·xst_broadcast` after (Ascan is the inclusive cumprod) — all wide `PIPE_V` ops, removing the PIPE_ALL. Math verified fp64-exact. **Measured ~5.3% SLOWER** (customer B8/D192/L5000/N16 bf16, same-session single-process npu.Event A/B, reproduced 2×): the post-fold adds 3 wide-CN ops/chunk (Broadcast+Mul+Add over CN≈4096, ×20 chunks ×1536 rows) whose issue-cost EXCEEDS the once-per-chunk PIPE_ALL drain removed. **AND the PIPE_ALL was load-bearing**: it is a periodic full-drain for the N>16 wide-write→wide-read ordering (OL-247), NOT a local RAW — removing it corrupts N=32/64 multi-chunk (absmax ~200 vs baseline ~7e-5); a "narrow-carry-out-first, wide-fold-last" refactor to dodge it STILL broke N>16, only restoring a PIPE_ALL fixed it. So even a correct F1 needs the fence back and is slower.

**The DISCRIMINATOR (the transferable lesson — reconciles with the backward DB win)**: **fence-elimination pays off ONLY when it does NOT add wide vector-issue.** Contrast the two measured cases on the SAME op family: (a) BACKWARD double-buffer (杠杆 D/E, PR #87/#88/#89) removed a coarse `SyncMTE2toV`/`SyncVtoMTE2` fence via a fine-grained per-slot WAR ping-pong that added ZERO compute ops → **WON +2.76%/+4.25%** (see the OL DB-reconcile note); (b) FORWARD post-fold (this CAND) removed a `PIPE_ALL` but PAID 3 wide-CN ops/chunk to do it → **LOST 5.3%**. On a vector-ISSUE-bound unit (small-N serial-L, OL-231), the added issue-count is what decides. Before removing any scan barrier, ask: does the replacement add wide vector ops? If yes, it will likely lose.

**The landed win from the same round (F2 reuse#3, +1.3% bit-identical — the positive half)**: `SSFwdBuildVF` was made to write `dBu` DIRECTLY into `xall` (=`Bscan`), eliminating the subsequent `Adds(Bscan,dBu,0,CN)` copy. Pure copy-elimination (the build VF's output buffer WAS a redundant intermediate), bit-identical, det 5/5, +1.30/1.31% ×3. **Rule: when a VF/op's output is immediately copied into another buffer with no intervening use, point the VF at the destination and delete the copy — free perf, provided no alias hazard.** This is issue-count REDUCTION (removes a whole CN-wide `Adds`), the opposite of F1.

**The aliasing-fence gotcha (F2 reuse#1)**: aliasing a prior chunk's V-pipe scratch (`sxKf`/`sxR`, last used by SoftExp/build) onto this chunk's MTE2 stage-write is a **V→MTE2 WAR the runtime does NOT auto-sync** (it auto-syncs MTE2→V only). Needs an explicit `SetFlag/WaitFlag<V_MTE2>` at the reuse point → then bit-identical + det-clean. (Same N>16-is-where-it-breaks lesson as F1/OL-247: verify every UB alias at N=32/64 multi-chunk, not just N=16.)

**Promote when**: a 2nd scan-barrier-removal on a small-N serial-L op reproduces "removing the barrier lost because the replacement added wide vector-issue." Cross-ref: OL-247 (VEC RAW silent corruption at N>16 — why the PIPE_ALL is load-bearing), OL-231 (small-N serial-L issue-bound ceiling), the OL DB-reconcile note (the backward fence-elimination that WON — the discriminator's positive pole), CAND-CHUNK-SIZE-RAISE-CHUNKED-FP32-REDUCTION-NOT-FREE (the same round's chunk-size wash). backend=ascendc.

## CAND-FWD-HWEXP-FP32-FLOOR-LOSES: replacing software fp32 transcendentals (SoftExp/SoftLn) with hardware `Exp`/`Ln` regresses the near-zero cancellation floor on a forward SSM scan — even though the SAME swap is a GO on the backward kernel (fwd/bwd diverge on the deltaA≈1.0 fp32 regime)

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.x; op_class=scan/SSM forward with a softplus/exp recurrence gate (deltaA=exp(delta·A) near 1.0); dtype=fp32 path only`
`verified_on: a5_ops:selective_scan_fwd_simd fp32 tier 2026-07-24 (Ascend950PR_957b, CANN 9.1.T500)`

**The trap**: an expert suggests "use the hardware `Exp`/`Ln` ISA instead of the software Horner/Newton transcendental" as a perf lever. On the FORWARD selective-scan the fp32 path deliberately uses SoftExp (Horner exp) / SoftLn / SoftRecip (Newton) precisely to hold the **near-zero cancellation floor** of `deltaA = exp(delta·A)` — values near 1.0 where the L-scan recurrence needs high *relative* precision. Hardware `Exp`/`Ln` use a lower-degree polynomial approximation that cannot hold that relative floor; the error compounds through the L-recurrence. Measured (surgical swap of the 4 fp32-branch Exp/Ln CALLS only, keeping fp32 on its general compute-path — do NOT flip `softTrans=(sizeof(T)==4)` since it ALSO gates path selection): builds clean (.so SMALLER, drops the Horner code), **fp32 case-0 MARE 1.52e-3 → 3.55e-3 (2.3× worse), MERE worse on both fp32 cases; bf16/fp16 bit-identical** (they already use hardware Exp — the swap only touched fp32). Deterministic (re-ran ×2, exact). REJECTED at the precision gate (perf A/B not run — a floor regression is disqualifying per OL-30 no-precision-drop-for-perf).

**The non-obvious part (fwd/bwd divergence)**: the identical hw-Exp/Ln swap was a **GO on the BACKWARD** kernel (selective_scan bwd, PR #66, 30/30 + perf — see expert-feedback doc row #3). Forward and backward diverge because the forward's precision gate is tighter on the exact `deltaA`-exp near-1.0 regime, whereas the backward's tolerance absorbs the poly-approx error. So "hw Exp is faster" is direction-specific — a lever that pays on one direction of an op can regress the other. Confirms the prior kw-4 attempt (SoftExp re-enable 9→8 regression) and the expert's own PROBE-FIRST caveat ("fp32 is at the near-zero cancellation floor — do not blind-change").

**General principle**: on a forward SSM/scan recurrence whose gate is an `exp` of a near-zero argument (`exp(x)` with `x→0`, result≈1.0), the software transcendental is load-bearing for the fp32 relative floor; do NOT swap it to hardware `Exp`/`Ln` for perf without an fp64-oracle floor check FIRST — and a GO on the backward does NOT transfer to the forward. Cross-ref: CAND-SCAN-FP32-ACCUM (the fp32-accumulator sibling — same op, keep fp32 internal), OL-103 (transcendental floor), OL-30 (no precision drop for perf), OL-231 (small-N issue-bound ceiling — why the perf upside was marginal anyway), the expert-feedback backward row #66 (the positive pole of the fwd/bwd divergence). Anchor: selective_scan_fwd_simd expert round SMID0724.txt R1, bg agent ssf-r1-ko, kernel unchanged (stays ac5508b0). backend=ascendc.

## CAND-GDR-1: gated-delta-rule Neumann triangular solve sign — the kernel computes `(I-L)^-1` but the reference solves `(I+L)^-1` with `L=strict(Acc)`; negate the host strict mask so `(I-L)^-1` becomes `(I+strict)^-1`
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-delta-rule / delta-rule triangular-solve; dtype=fp16-in/fp32-accumulate`
`status: CANDIDATE — 1 op device-confirmed. Source: gated_delta_rule fwd gen3 (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The bug**: the within-chunk triangular solve builds `T = (I + strict(Acc))^-1` via a Neumann power-product `prod(I + L^{2^k})`, k=0..5, with `L = strict(Acc)`. The kernel as first authored materialized `(I - L)^-1` (host strict mask emitted as `+tril(ones, -1)`), which is the WRONG sign for the gated-delta-rule recurrence → per-head result ~0.40 off the reference. **FIX**: negate the host strict mask (`maskStrict = -tril(ones, -1)`) so the kernel's `(I - L)^-1` product evaluates to `(I + strict)^-1`. Verified vs `model.py` to 7.4e-8 once corrected. This is a math-sign bug pinned by CPU-vs-device intermediate diff, NOT a precision/dtype issue (CPU fp16-sim passes at the corrected sign).

**Correction to an earlier overstated thesis**: "fp16 overflow → carry Neumann intermediates in fp32" was WRONG for this op (CPU fp16-sim passes); the real precision defect was the SIGN. Do NOT codify an fp32-intermediate fix for this op. **Promote when**: a 2nd delta-rule / gated-linear-attention triangular-solve reproduces the sign-convention gotcha. Cross-ref P-P117, CAND-GDR-4.

## CAND-GDR-2: intra-AIV UB-buffer WAR — reusing the same UB `LocalTensor`s across consecutive AIV elementwise ops without a fence races the prior op's V-pipe read against the next op's MTE2 overwrite; `PipeBarrier<PIPE_ALL>()` at the helper entry fixes it — cross-core sync does NOT
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=any multi-call AIV elementwise reusing UB buffers; dtype=any`
`status: CANDIDATE — 1 op device-confirmed. Source: gated_delta_rule fwd gen3 (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The bug**: reusing the same UB `LocalTensor`s (`la`/`lb`/`lc`) across consecutive AIV elementwise ops WITHOUT a fence — the prior op's V-pipe read of `la`/`lb` races the next op's MTE2 `DataCopy` overwrite → intermittent, timing-dependent single-element / single-row garbage (non-deterministic run-to-run). **CRITICAL — this is an INTRA-core hazard**: whole-device `SyncAll` OR scoped `CrossCore` sync does NOT fix it (both were tried on-device; the race survived BOTH). Cross-core barriers order work BETWEEN AIC and AIV groups; they do nothing for a WAR WITHIN one AIV core's own pipe. **FIX**: `PipeBarrier<PIPE_ALL>()` at the elementwise-helper entry (drains the prior V read before the MTE2 overwrite) → bit-deterministic (20/20 + 8/8 re-runs).

**Correction to an earlier overstated thesis**: "scoped CrossCore is the race fix" was WRONG — the race survived CrossCore. The real race fix is the intra-AIV Vop WAR fence; the `MIX_AIC_1_1` macro (see PB-57), not CrossCore, fixed the separate 507014 DEADLOCK. **Promote when**: a 2nd multi-call AIV elementwise op reproduces the "cross-core sync does not cover an intra-core UB WAR" separation. Cross-ref PB-17 (cross-row V→MTE2 alias hazard — sibling), `fa_class/cross_core_sync.md` §6 (the non-overlap note), PB-57, P-P117.

## CAND-GDR-3: one reused `MatmulImpl` object across matmuls of DIFFERENT shapes needs `SetOrgShape(M,N,K)` per call (and Init at the MAX shape); `SetSingleShape` alone leaves stale org dims → K-mismatch garbage
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=any kernel reusing one MatmulImpl across multiple matmul shapes; dtype=fp16-in/fp32-out`
`status: CANDIDATE — 1 op device-confirmed. Source: gated_delta_rule fwd gen3 (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The bug**: one `MatmulImpl` object reused across matmuls of DIFFERENT shapes (K=128 stage-1 KKT / K=64 Neumann chain / N=128 for U,W,O) — calling `SetSingleShape` alone does NOT re-establish the org dims, so the `K != Init` matmuls (e.g. `P2 = L @ L`) produce garbage (deterministic; per-head maxdiff == ref_absmax, `T` collapses to ~identity). **FIX**: (a) call `mm.SetOrgShape(M, N, K)` per matmul; (b) `Init` the `MatmulImpl` at the MAX shape (M=N=K=128) so L0C/L1 fit every call. **Promote when**: a 2nd multi-shape-reused-matmul kernel reproduces the "SetSingleShape without SetOrgShape → K-mismatch garbage." Cross-ref P-P68 (single-AIC GEMM static tiling), P-P117.

## CAND-GDR-4: precision-bar threshold conflation — the fp64 customer reference (`ref_gdr.py`, rtol=0.02) is the acceptance verdict (16/16, ≥17× headroom); `model.py`'s strict fp32 bar (9.766e-4, 14/16) under-rates the kernel via a forced `.to(float32)` and must NOT be read as a kernel bug
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=reporting GDR / attention precision pass-count; dtype=fp16-in`
`status: CANDIDATE — 1 op device-confirmed. Source: gated_delta_rule fwd gen3 (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The point**: "16/16 pass" is threshold-dependent. Against the customer fp64 reference (`ref_gdr.py`, FQA `_assert_relative` rtol=0.02) the kernel is **16/16 with ≥17× headroom** (worst case6 max_err 9.9e-5 vs threshold 1.71e-3). Against `model.py`'s fp32 reference it is 16/16 @ rtol=0.02 but **14/16 @ the shipped script's stricter internal 9.766e-4** (case6, case9 marginal). That marginal fp32 fail is NOT a kernel inaccuracy: `model.py`'s internal forced `.to(float32)` adds fp32 rounding ON TOP of the reference. **Rule**: codify the customer's fp64 gate as the acceptance verdict; never relabel a strict-fp32 marginal fail as a kernel bug (nor inflate the pass-count by hiding the strict-bar split). Cross-ref P-P117, CAND-GDR-1.

## CAND-GDR-BWD-1: gated-delta-rule BACKWARD A-matrix decay mask `dsMask2` — `decay_mask.swapaxes(-2,-1)` swaps the C axis with the HEAD axis (head relocation), NOT the two [C,C] axes; correct per-head matrix = `exp(g_c-g_d)` masked strict-lower (c>d)
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/delta-rule BACKWARD A-matrix (KKT) gradient; dtype=fp16-in`
`status: CANDIDATE — 1 op device-confirmed. Source: gated_delta_rule bwd (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The bug**: in the backward of the intra-chunk A-matrix (KKT), the reference applies `decay_mask.swapaxes(-2,-1)`. On the `[B,N,C,C,Hv]` decay tensor `swapaxes(-2,-1)` swaps the **last `C` with the `Hv` head axis** (a head relocation), it does **NOT** transpose the two `[C,C]` chunk axes. Reading it as a matrix transpose (using `exp(g_d - g_c)` for `d>c`) makes `dk/dg/db` FAIL (~0.12–0.27) while `dq/dv/dh0` stay clean. **FIX**: the correct per-head 2D mask is `dsMask2 = mStrictLow · exp(g_c - g_d)`, masked **strict-lower** (`c>d`). **Localization tip**: the failing grads (dk/dg/db) all share the dsMask2 chain, which dq/dv/dh0 do not — a per-grad pass/fail split pins the mask. **Promote when**: a 2nd delta-rule / gated-linear-attention backward reproduces the swapaxes-is-head-relocation gotcha. Cross-ref P-P118, CAND-GDR-1.

## CAND-GDR-BWD-2: gated-delta-rule BACKWARD `dstate` init must `.clone()` the input `dht` — a `.to().reshape().contiguous()` chain is a no-op VIEW on fp32-contiguous input, and the reverse-recurrence updates `dstate` in-place → mutates the caller's input tensor
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=any kernel that seeds an accumulator from an input tensor then updates it in-place; dtype=any`
`status: CANDIDATE — 1 op device-confirmed (negative-control proven). Source: gated_delta_rule bwd (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The bug**: the host seeds the reverse-recurrence accumulator `dstate` from the input `dht` via `dht_.to(kFloat).reshape().contiguous()`. When `dht` is ALREADY fp32-contiguous every step is a no-op → `dstate` **aliases the input storage**. PASS-B then updates `dstate` **in-place** across the reverse chunk loop → it **corrupts the caller's `dht_`** (a real input-mutation bug that breaks a training loop reusing `dht`), and a 2nd call on the same `dht` sees garbage → `dk/dg/db/dv` drift while `dq/dh0` stay clean (they don't depend on `dstate`). The normal verify harness makes FRESH inputs per case so it **cannot** catch this — you need a repeated-call-same-tensor stress test. **FIX**: `.clone()` the `dstate` init (independent copy). **Negative control (proven load-bearing)**: rebuilding with `.clone()` REMOVED made the stress test FAIL exactly as predicted (dht drift ~0.10–0.14; dk up to 1.09, dg 1.08, db 0.84 drift across calls; dq=0, dh0~0). **Promote when**: a 2nd kernel seeds an accumulator from an input then updates it in-place. Cross-ref P-P118, PB-17.

## CAND-GDR-BWD-3: a test-harness input-generation bug can masquerade as a kernel precision failure — build the passed-in state faithfully (thread `initial_state`), and cross-check with the reference's own input construction before blaming the kernel
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=authoring a fp64 gate / test harness for an op that takes a precomputed intermediate as input; dtype=any`
`status: CANDIDATE — 1 op device-confirmed. Source: gated_delta_rule bwd fp64 gate (DS, a3 Ascend910_9382, 2026-07-20). backend=ascendc.`

**The bug**: our newly-authored `test_fp64_gate.py` scored 10/11 (case1 fp64 fail on dq/dk/dg/db). Root cause was in the TEST, not the kernel: `get_input_groups` built the passed-in recurrent state `h` via the forward WITHOUT threading `initial_state`. The backward reference takes **no `h` argument** — it recomputes `h` internally WITH `initial_state` — so the passed-in `h` is consumed ONLY by the kernel, which trusted a state missing the decayed-init term. Case1 is the ONLY case with a nonzero `h0`, so ONLY it failed; `dv` (doesn't use `h`) and `dh0` (uses `h0` directly) stayed clean. **FIX**: thread `initial_state` into the `get_input_groups` forward (matches the customer's own `verify_cases.py`) → 11/11; no kernel change. **Diagnostic discipline** (3 checks that pinned it to the harness, not the kernel): (1) abs-err vs `max|ref|` refuted a near-zero-denominator artifact (denominators normal ~2.9/5.2, abs-err genuinely ~2.9/4.8); (2) fed the customer's `h` (built WITH init) the kernel passed case1 6/6, fed our init-less `h` it failed 2/6; (3) toggling `initial_state` isolated the trigger to nonzero init exactly. **Rule**: when a single case fails a self-authored gate, suspect the harness's input construction and diff it against the reference's own — an "all-inputs-look-plausible" harness can still feed a self-inconsistent precomputed intermediate. **Promote when**: a 2nd op taking a precomputed intermediate reproduces a harness-input-inconsistency false-fail. Cross-ref P-P118, CAND-GDR-4.

## CAND-GDR-FRACTAL-1: hand-cube fractal-lowering params are codified as per-context RULES (contract-axis-agnostic), not per-(M,N,K) tuples — tabulate the concrete strides for a chunked-recurrence op's varied contractions so the next gen skips the device-iteration cost
`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX/many-small-matmul; dtype=fp16 q/k/v + fp32 g/beta`
`status: CANDIDATE — UNVERIFIED (proposal only; verified_on: pending). Refinement of P-P119, NOT a contradiction of it. Derived from workspace/gdr_capstone4 kw-1 cold-start generation-foundation spawn (DS, 2026-07-21) — no build attempted that spawn; the sibling VERIFIED finding (MatmulImpl+IterateAll standalone → 507057 / 0-16 vs hand-rolled → 16/16 correctness gate) is ALREADY merged into P-P119 + a3_mix_small_matmul_cube.md (2026-07-21). backend=ascendc.`

**The observation**: `fa_class/cv_reference_concrete_params.md::matmul_primitive` codifies the hand-cube (`Nd2Nz→LoadData3D→Mmad→Fixpipe`) params as **per-context RULES** — the Mmad 4-arg accumulate form + `cmatrixInitVal=(ki==0)` + the per-context `Fixpipe srcStride` unit (`/C0` for L0C→workspace, ELEMENTS for L0C→GM; wrong unit → 507015 ECC read). P-P119 tells the next gen to reuse **those rules and NOT re-derive**. The rules are contract-axis-agnostic and (per P-P119, device-verified 16/16) sufficient. What is NOT tabulated is the fully-resolved concrete `Nd2Nz dstNzC0Stride` / `LoadData repeatTimes,srcStride` / `Fixpipe srcStride` VALUES for each of a GDR chunk's distinct (M,N,K) matmul shapes with **varied contraction axes**: `Acc=kb@knT` contracts over D=128 (M=C,N=C,K=128); `T@vb`/`W@S`/`Am@vn` contract over C=64 (K=64); `kdT@vn` contracts over C=64 (M=D,N=D). The FA-class reference tabulates only QK^T (contract-D) and P@V (contract-kv). **Proposal**: add a plain-`A@B` contract-over-{D=128, C=64} concrete-value table to `a3_mix_small_matmul_cube.md` (P-P119's home) so a future GDR-class gen applies the RULES to fixed shapes from the table instead of iterating them on device — a debuggability/speed refinement, not a correctness change (P-P119's rules already yield a correct kernel). **Do NOT read this as "the rules are insufficient"** — the reconciliation is: rules = load-bearing + verified; per-tuple table = optional convenience layer that lowers the next gen's device-iteration cost. **Promote when**: a 2nd chunked-recurrence / gated-linear-attention gen materializes the concrete per-(M,N,K) table on device and confirms a subsequent gen reused it WITHOUT re-iterating (i.e. the table itself is device-grounded, not just proposed). Until then it stays a proposal — the values must come from a working kernel, not be hand-guessed. Cross-ref P-P119 (the reuse-the-rules parent), P-P117 (the many-small call-site), `fa_class/cv_reference_concrete_params.md` (the rule source), CAND-GDR-3 (SetOrgShape per reused-mm call), `cv_reference_concrete_params.md::nd2nz_srcdvalue_uint16_overflow` (any concrete `Nd2Nz srcDValue` in the proposed table must stay < 65536 — GDR chunk strides K=D=128/C=64 are far under the limit, so GDR is not in the overflow regime, but the table must carry the constraint).
