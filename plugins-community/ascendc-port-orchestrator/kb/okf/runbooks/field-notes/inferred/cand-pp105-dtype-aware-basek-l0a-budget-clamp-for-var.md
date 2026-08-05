---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Dtype-aware baseK L0A-budget clamp for variable/tiny per-group cube GEMM dims — derive base sizes at kernel entry, leave static tiling base fields at -1"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=cube_gemm (grouped/segmented/MoE-expert/ragged-batch matmul + backward), dtype-sensitive (fp32 needs the K cap) verified_on: soc=Ascend910_V220; ca"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=cube_gemm (grouped/segmented/MoE-expert/ragged-batch matmul + backward), dtype-sensitive (fp32 needs the K"
confidence: inferred
status: stub
original_id: CAND-PP105
timestamp_inferred: true
tags: [candidate, inferred, matmulapistatictiling, tcubetiling, basek, cand-pp105]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP105，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
