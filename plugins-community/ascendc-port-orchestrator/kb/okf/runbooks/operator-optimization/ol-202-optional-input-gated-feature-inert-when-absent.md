---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "A reference feature gated on an OPTIONAL input tensor is inert when that input is absent — the benchmark's input config, not the op's documented capability, defines the truth"
description: "A reference feature gated on an optional input's presence (not a mode flag alone) is inert if that input is absent; the benchmark's input config, not documented capability, is the truth."
confidence: single_run
original_id: OL-202
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-202, optional-input-gate, benchmark-truth, anti-cheat]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
When a reference op conditionally activates a feature based on the **presence of an optional input tensor** (not on a mode-flag attribute alone), setting that mode/attribute WITHOUT supplying the gating input is **inert** — the op silently computes the default. Consequently the **benchmark's actual input configuration**, not the op's documented general capability, defines the ground-truth behavior to match. A whole "axis" of mode-flag values can be **phantom** (all collapse to the default) if the benchmark never supplies the gating input.

**Verification protocol (do BOTH)**: (a) read the `op_host`/tiling source for the gate that decides activation; (b) confirm empirically by varying the mode flag with the gating input absent — output must be bit-identical across all mode values.

**Concrete anchor**: arch35 FlashAttentionScore `op_host` `flash_attention_score_tiling_regbase.cpp` `GetSparseInfo` L1347-49:
```cpp
if (!hasAttenMask) { return true; }   // sparseType stays at default SparseEnum::ALL (dense)
```
The entire `sparse_mode` dispatch (causal / band / prefix) is AFTER that guard, and `hasAttenMask` is set purely from whether the optional `atten_mask` tensor is passed (non-empty shape). Benchmark `3_FusionAttention`: **0/50 cases supply a required atten_mask** → all 50 compute **dense** attention regardless of `sparse_mode` (0–5). The perceived "41-case sparse_mode axis" was phantom; `sparse_mode` is NOT broken — geometry is driven by the **mask parameter** the benchmark doesn't provide.

**Anti-pattern (inference-substituted-for-measurement)**: (i) reading a KERNEL's mode-dispatch and asserting "implement the feature" WITHOUT reading the `op_host` GATE that decides whether it activates — *dispatch-existing ≠ it-activates* (main read `kernel_train.h` SparseModeEnum dispatch, missed the `op_host` `if(!hasAttenMask)` gate, sent a peer to implement inert masking); (ii) extrapolating a downstream-kernel claim ("the kernel over-masks → N cases recoverable") from a correct upstream conclusion WITHOUT measuring the actual kernel — refuted by reading the committed artifact's per-case `keep_prob` field (all 20 FAIL cases were kp<1.0 dropout, **zero** kp=1.0 → zero over-mask recovery). The fix for both: ground in the gate/source/disk, not a plausible inference.

Verified on soc=Ascend950PR, cann=9.0.0 (FA-A5 `3_FusionAttention`, 3-way converged 2026-06-01).
