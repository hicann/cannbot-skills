---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "chunk-recurrent / linear-attention via composed catlass primitives (no per-op cube template)"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=chunk-recurrent / linear-attention status: CANDIDATE — catlass composition. DELIVERABLE binary = 98_gdn_single (built from gdn.cpp, md"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=chunk-recurrent / linear-attention"
confidence: inferred
status: stub
original_id: CAND-GDN-CHUNK-RECURRENT-COMPOSE
timestamp_inferred: true
tags: [candidate, inferred, rungemm, layouttaga, layouttagb, dispatchpolicy, l1tileshape, cand-gdn-chunk-recurrent-compose]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-GDN-CHUNK-RECURRENT-COMPOSE，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
