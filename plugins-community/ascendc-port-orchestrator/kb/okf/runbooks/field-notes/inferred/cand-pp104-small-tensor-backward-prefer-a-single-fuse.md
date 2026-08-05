---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Small-tensor backward — prefer a single fused kernel over the partial+reduce multi-launch template when per-core vector work can't amortize launch overhead"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (layer/rms/group norm grad), generalizes to any partial+reduce multi-launch backward verified_on: soc=Ascend910_V220; cann=9.0"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (layer/rms/group norm grad), generalizes to any partial+reduce multi-launch backward"
confidence: inferred
status: stub
original_id: CAND-PP104
timestamp_inferred: true
tags: [candidate, inferred, aclrtlaunchkernel, wholereducesum, group_norm_grad, device_self_duration, cand-pp104]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (layer/rms/group norm grad), generalizes to any partial+reduce multi-launch backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (correlation measured; causal mitigation UNCONFIRMED)`
`status: UNCONFIRMED — promote only after an optimizer fused-rewrite recovers the ratio (validates causation, not just correlation)`

**Pattern**: the OL-75 dual-axis partial+reduce template (main per-row/per-group kernel + a reduce kernel launched once per cross-row output) is **size-sensitive**. It WINS on large tensors (vector work ≫ launch overhead) but LOSES on tiny tensors, where the ≥2–3 fixed `aclrtLaunchKernel` overheads (~20–25µs each, plus cross-core sync) dominate. Mitigation hypothesis for the small-tensor regime: collapse to a single fused kernel — (1) merge the per-output reduce launches into one; or (2) single-core fused main+reduce; (3) vectorize the per-channel reduction (one `WholeReduceSum` multi-repeat vs CG×2 fold-reduces); (4) consider MIX pipelining (OL-200). Profiling-first (msprof to confirm the launch-vs-compute split before rearchitecting).

**Evidence (correlation only — causal claim pending)**: group_norm_grad (2026-06-03, port_a3_to_a5 V220, authored from scratch). 2-kernel design issues 3 launches (main + 2 reduce, one per dweight/dbias). On GroupNorm tiny tensors (≤1024 elems): ours ~150–200µs flat vs vendor single fused CANN kernel ~28–49µs → **ratio 0.25×, stable across 3 runs / 2 devices**. Precision + determinism 4/4 PASS first try — purely a launch-overhead regression. Root cause flagged as un-profiled hypothesis in the op's verification.json (NOT yet msprof-confirmed). Contrast: sibling layer_norm_grad ran the SAME template at 1.0–1.9× because H=1024–4096 amortizes the launches (see OL-75 large-tensor evidence row).

**Profiler device-time cross-check (2026-06-03, same `group_norm_grad` kernel, P97-canonical)** — do NOT conflate the two measurement regimes: kernel-only `device_self_duration` ratio = **0.851×** (ours ~8–10µs vs vendor ~8.6µs). The end-to-end **0.25× wall-clock** vs this **0.851× device-time** gap IS the host launch overhead this candidate is about (3 launches × ~20–25µs vs vendor's 1 fused kernel). i.e. the regression is launch-strategy (multi-launch), NOT kernel-efficiency — our kernel is roughly on par with vendor at the device level; it's the 3-launch host orchestration that loses the wall-clock race. Strengthens the "fuse the launches 3→1" mitigation hypothesis; still UNCONFIRMED pending an optimizer fused-rewrite that recovers the wall-clock ratio.

**Promote when**: an optimizer applies the fused single-kernel rewrite to a small-tensor backward op AND measures the ratio recovering toward/above 1.0× on the same NPU (back-to-back A/B), confirming launch-overhead (not compute) was the lever. Until then this stays a correlation, not a verified pattern.

**Cross-ref**: OL-75 (the partial+reduce template + its size-sensitivity scope condition), OL-200 (MIX pipelining), the per-row K_ROWS_PER_AIV launch-amortization candidate (line ~764), CLAUDE.md profiling-first rule.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP104，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
