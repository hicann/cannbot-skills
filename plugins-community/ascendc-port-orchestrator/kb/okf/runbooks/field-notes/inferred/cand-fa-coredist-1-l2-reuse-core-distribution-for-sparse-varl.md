---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "L2-reuse core-distribution for sparse/varlen attention — round-robin + symmetric mirror + boustrophedon sweep"
description: "Date: 2026-06-03 derived-from: cann-source (FA arch35 forward kernel, core-split helpers + Process loop region) Status: CANDIDATE — sanitized re-expression + turn-3b EXPLICIT-GENERIC offset-arithmetic"
phenomenon: build_failure
signal:
  - "Date: 2026-06-03"
confidence: inferred
status: stub
original_id: CAND-FA-COREDIST-1
timestamp_inferred: true
tags: [candidate, inferred, corenum, partiallength, cores, myid, cand-fa-coredist-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-COREDIST-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
