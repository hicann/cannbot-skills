---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Deterministic backward via coordinate-partitioned core dispatch — replace atomic-add with a number-theoretic per-core (s1,s2)-tile assignment so each output element is touched by exactly one core"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=deterministic_backward / flash_attention_backward_deterministic / any_bwd_op_where_atomic_add_violates_determinism_requirement deriv"
phenomenon: build_failure
signal:
  - "Backward op needs bit-reproducibility across runs (PyTorch torch.use_deterministic_algorithms(True), training-loss-reproduction in research, downstream debuggin"
confidence: inferred
status: stub
original_id: CAND-FAG-2
timestamp_inferred: true
tags: [candidate, inferred, datacopy, coreid, ceil, setatomicadd, s1outer, cand-fag-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FAG-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
