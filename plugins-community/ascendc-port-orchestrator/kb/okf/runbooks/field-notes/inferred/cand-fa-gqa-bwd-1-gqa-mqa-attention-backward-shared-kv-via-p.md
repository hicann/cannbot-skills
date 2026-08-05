---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "GQA/MQA attention backward — shared-KV via per-operand group-divisor + dK/dV accumulate-over-G via cube-partials-then-deterministic-VEC-reduce"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382) unverified_on: soc=Ascend950PR (A5/V300 — A3 eviden"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward"
confidence: inferred
status: stub
original_id: CAND-FA-GQA-BWD-1
timestamp_inferred: true
tags: [candidate, inferred, matmulimpl, reduce_g, flash_attention_grad_gqa, npu_fusion_attention_grad, cand-fa-gqa-bwd-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

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

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-GQA-BWD-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
