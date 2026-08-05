---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "sliding-window (local-causal) attention backward — precision idiom (host-built additive [S,S] band mask via whole-row Add, NOT per-row Duplicate; dS needs no mask); STARTS below the perf floor (perf RESOLVED without a template in CAND-FA-SWA-BWD-2)"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382) unverified_on: soc=Ascend950PR (A5/V300) Principle:"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward"
confidence: inferred
status: stub
original_id: CAND-FA-SWA-BWD-1
timestamp_inferred: true
tags: [candidate, inferred, add, flash_attention_grad_swa, npu_fusion_attention_grad, cand-fa-swa-bwd-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-SWA-BWD-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
