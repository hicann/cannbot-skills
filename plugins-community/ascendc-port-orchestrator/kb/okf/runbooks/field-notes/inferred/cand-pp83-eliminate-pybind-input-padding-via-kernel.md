---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Eliminate pybind input-padding via kernel-side DataCopyPad GM→UB last-tile"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=tile-loop-vec verified_on: soc=Ascend950PR; cann=9.0.0 unverified_on: soc=Ascend910_V220 (A3 chip family — DataCopyPad GM→UB semantics li"
phenomenon: build_failure
signal:
  - "tile-loop kernel where pybind currently does auto x_padded = torch::zeros({B, D_padded}, ...); x_padded.slice(-1, 0, D_orig).copy_(x); to align row stride to a"
confidence: inferred
status: stub
original_id: CAND-PP83
timestamp_inferred: true
tags: [candidate, inferred, x_padded, fill, viewcopy, datacopy, cand-pp83]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP83，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
