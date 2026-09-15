---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "FlashAttention-class MIX template (a3/arch22) — hand-authored cube+vector MIX_AIC_1_2 attention starting skeleton, device-proven"
description: "For any a3 (Ascend910_9382, arch22) FA-class / attention-fwd op needing a hand-authored cube+vector MIX kernel — the a3 counterpart of P-P102/P-P103 (which are a5/arch35 ONLY). Device-proven skeleton"
severity: high
confidence: single_run
original_id: P-P116
timestamp_inferred: true
tags: [flash_attention, optimization, famix, famix_mh, aclrtlaunch, rtgetc2cctrladdr, p-p116, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For any a3 (Ascend910_9382, arch22) FA-class / attention-fwd op needing a hand-authored cube+vector MIX kernel — the **a3 counterpart** of P-P102/P-P103 (which are a5/arch35 ONLY). Device-proven skeleton (DS `famix`/`famix_mh`, Ascend910_9382): `S=Q@Kᵀ`(cube#1, AIC) → `P=softmax(S/√d)`(vector, AIV, fp32 row-wise) → `O=P@V`(cube#2, AIC). **Dispatch**: MIX_AIC_1_2 via standard AscendC AIC+AIV device objects + `aclrtlaunch` (FFTS descriptor auto-supplied via `rtGetC2cCtrlAddr`); **do NOT** emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` on arch22 (arch35-only macro → 107000; the a5-native macro P-P102/103 KEEP, a3 must OMIT). **Cube**: `MatmulImpl<> IterateAll<sync=true>` (NEVER async KfcServer `Iterate()/GetTensor()` → PB-34), runtime transpose-B (static ISTRANS false). **Handshake**: `CrossCoreSetFlag<MODE2,PIPE_FIX>(FLAG_S=4)` fwd (broadcast) / **BOTH AIV subblocks** `CrossCoreSetFlag<MODE2,PIPE_MTE3>(FLAG_P=5)` reverse — reverse is per-subblock-COUNTED (PB-55: single-setter DEADLOCKS); flag ids distinct, NEVER 0 (PB-35); MODE2 suffices (NOT arch35 §4 mode-4). **Scratch**: `S`,`P` `[seq,seq]` fp16 in GM; softmax one row at a time in UB; multi-head = device loop over `nheads=B*H`, one reused S/P pair (FLAG chain serializes). Genuine cube both matmuls (pure-VEC = OL-188 hack, forbidden). **HONEST SCOPE** device-verified: seq≤384, d=64, fp16, single-pass NON-flash softmax, multi-head, cos 0.999999, deterministic; NOT proven: flash online-softmax KV-tiling / causal mask / perf. Closes the a3 half of DEBT-222. Full body: `patterns/domains/fa_class_a3_mix_template.md`. Cross-ref PB-55/PB-34/PB-35/OL-188/OL-235/P-P101/P-P102/P-P103. `applies_to: soc=Ascend910_9382; cann=9.1.0; op_class=attention-fwd/CUBE_MIX (a3 hand-authored); verified_on=Ascend910_9382 famix/famix_mh (DS 2026-07-18); unverified_on: Ascend950PR`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P116，convert_patterns_to_okf.py）。confidence 未升格。 -->
