---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "sliding-window FA backward PERF in multi-launch — cache the [S,S] mask (decisive) + block-skip out-of-window GEMM ranges; the fused template is NOT needed to clear the floor"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382) Principle: A precision-correct full-S×S multi-launc"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward"
confidence: inferred
status: stub
original_id: CAND-FA-SWA-BWD-2
timestamp_inferred: true
tags: [candidate, inferred, torch.where, full, aclrtlaunch, cand-fa-swa-bwd-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382)`

**Principle**: A precision-correct full-`S×S` multi-launch sliding-window attention backward that sits below the perf floor can be brought ABOVE it WITHOUT the forward fused single-launch template — two cheap multi-launch levers, in measured priority of impact:
- **(1) CACHE the `[S,S]` additive band mask per (S,W,device) — THE decisive lever.** Rebuilding the mask (`torch.where`/`full` on device) every call is ~2–3× the real backward compute on small/mid shapes and is counted in device-self time, while the vendor takes a pre-built mask → per-call mask-build is an apples-to-oranges self-penalty. Caching (build once in warmup, reuse) makes it fair. This ALONE took SWA-bwd 0.354× → 0.852× (cleared the 0.6× floor).
- **(2) Block-skip the `S=Q@Kᵀ` / `dP=dO@Vᵀ` GEMMs** to the windowed band (banded GEMM looping query row-blocks, computing only key cols `[max(0,r0−W+1), r1)`, strided C via MatmulImpl `SetOrgShape(orgN=Sk)`; multi-launch, AIC-only, no CrossCore). **Measured impact: nearly zero** (0.354→0.349×) — the S×S GEMMs were never the bottleneck (cast / softmax / dS / output-GEMM dominate). Honest measurement REFUTED the intuition that block-skip would dominate; kept for large-W/large-S regimes but it is not the lever.

**Hard trap (KB-worthy)**: do NOT zero the out-of-band workspace via torch `.zero_()` BETWEEN raw `aclrtlaunch` kernels on the same NPU stream — stream-ordering is not guaranteed and races (esp. when fp16/bf16 cast kernels precede → the banded GEMM reads un-zeroed Sf/dPf = garbage; fp32 masked it). FIX = zero at ALLOCATION (`torch::zeros`), never mid-stream between raw launches.

**Verdict (answers "does sliding-window backward need the fused template for perf?")**: NO. Cheap multi-launch levers (mask-cache + block-skip) clear the 0.6× floor (0.852× median, 2/10 cases beat vendor), staying entirely out of the V220 PB-34/35 fused-MIX minefield. Combined with CAND-FA-GQA-BWD-1 (GQA also needs no template): the FA-class-backward finding is that the forward fused template is NOT required for these variants — derive-from-forward + cheap multi-launch optimization suffices for precision AND perf.

**Evidence**: flash_attention_grad_swa ko (gqa-bwd-wp1, 2026-06-19, A3/V220 Ascend910_9382) — perf 0.354× → 0.852× (lead independently re-ran in-container, matches; mask-fairness verified — vendor mask pre-built outside timed region, ours cached). Precision UNCHANGED (fp16/bf16 PASS 600/600, fp32 T2, deterministic 6/6 md5-identical to pre-ko).

**Promote when**: a second banded/local-attention backward reproduces "mask-cache clears the perf floor in multi-launch without the fused template". Cross-ref: CAND-FA-SWA-BWD-1 (precision/mask idiom), CAND-FA-GQA-BWD-1, CAND-FA-MULTI-LAUNCH-PERF-GAP.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-SWA-BWD-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
