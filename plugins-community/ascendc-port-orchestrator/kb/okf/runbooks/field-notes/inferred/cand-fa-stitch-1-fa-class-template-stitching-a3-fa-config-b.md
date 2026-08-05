---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-class template-stitching — A3-FA config → block-selection + host-tiling parameterization + launcher → self-contained A5 op"
description: "applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 forward whitebox is the worked instance) derived-from: synthe"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/CUBE_MIX"
confidence: inferred
status: stub
original_id: CAND-FA-STITCH-1
timestamp_inferred: true
tags: [candidate, inferred, wp_kernel_base, wp_kernel_train, wp_block_cube, wp_block_vec_base, wp_attenmask, cand-fa-stitch-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-STITCH-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
