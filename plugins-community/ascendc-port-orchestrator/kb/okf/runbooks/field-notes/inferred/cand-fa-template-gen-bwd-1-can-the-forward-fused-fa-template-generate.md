---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "can the forward fused FA TEMPLATE generate the FA BACKWARD? — STRUCTURALLY YES (arch35 FA-grad IS the same CRTP/MIX/launch template with backward blocks), but the fused single-launch route is V351/arch35-ONLY (MicroAPI ISA-gate + V220 MIX-sync) — the V220-working backward is the non-template multi-launch path"
description: "applies_to: soc=Ascend910_9382 (V220/dav-c220) vs Ascend950PR (V351/dav-c310); cann=9.0.0; op_class=flash-attention-backward / CUBE_MIX status: graybox-feasibility-whitebox 2026-06-19 (mapping PROVEN"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382 (V220/dav-c220) vs Ascend950PR (V351/dav-c310); cann=9.0.0; op_class=flash-attention-backward / CUBE_MIX"
confidence: inferred
status: stub
original_id: CAND-FA-TEMPLATE-GEN-BWD-1
timestamp_inferred: true
tags: [candidate, inferred, flash_attention_score, flash_attention_score_grad, entry_regbase.h, kernel_type_mix_aic_1_2, regbasefag, cand-fa-template-gen-bwd-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220/dav-c220) vs Ascend950PR (V351/dav-c310); cann=9.0.0; op_class=flash-attention-backward / CUBE_MIX`
`status: graybox-feasibility-whitebox 2026-06-19 (mapping PROVEN by source; build REFUTED on V220 measured 3/3; precision UNREACHED on V220)`

**Owner question**: 通过模板能否生成反向 — can the forward FA fused single-launch *template-assembly* recipe (CAND-FA-STITCH-1 + CAND-FA-LAUNCH-DISPATCH-1, which generates `flash_attention_score`) generate the fused FA **backward**?

**Answer (two-part, honest)**:
1. **STRUCTURAL feasibility — YES.** Target prior-art inspection shows that
`flash_attention_score_grad` arch35 uses the same broad machinery as the forward, with backward blocks
swapped in. This mapping is advisory: re-derive task-owned backward code from gradient math,
saved-tensor contract, selected forward/source contract, and current public APIs; do not stitch copied
target bodies. The component mapping is:
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
target FA-grad blocks, built, and matched target/vendor output on 8 BN2 cases. Retain its measurements
as capability evidence. Because it reused target bodies and used target output rather than CPU fp64
autograd as final truth, it does not prove autonomous backward generation under the current boundary.

**The V220-working backward is NOT template-generated**: `fa_gqa_grad` (CAND-FA-GQA-BWD-1) = multi-launch AIC-ONLY `MatmulImpl` + separate AIV cast (ZERO MicroAPI, sidesteps PB-34/35). `fa_class_template.md` L488-491 already notes the existing FA-bwd is "non-template-assembly — NOT the forward template-assembly path."

**Gaps the forward template recipe LACKS for the backward**: (G1) target — forward template is arch35/V351, brief verify env is A3/V220; fused backward needs a V351 build+run lane. (G2) Pre/Post phases — recipe has no multi-launch-phase concept. (G3) 5-vs-2 GEMMs + fp32 partial workspace + deterministic-accumulate variants. (G4) softmax-grad vec block (new, not in forward block set). (G5) backward-only template axes (deterType/TND/D_NO_EQUAL/ROPE/NZ_OUT/dpse/dsink). (G6) MIX-sync UNSOLVED on V220.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-TEMPLATE-GEN-BWD-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
