---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "L2+L3 hybrid is a first-class tier, not a tie-break of OL-143"
description: "A5-port sweep shows tier classification must be MULTI-LABEL: 5 of 14 upstream ports run L2 MicroAPI hot loop AND L3 SIMT scatter/gather in one binary. Load both playbooks, do not pick one tier."
confidence: single_run
original_id: OL-153
classified_by: llm-assisted
timestamp_inferred: true
tags: [tier-classification, optimization, ol-153, port-a3-to-a5, multi-label, simt-microapi]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=all_port_a3_to_a5, phase=O2.5_classifier`.
Source: empirical sweep of 14 upstream A5 ports (`docs/analysis/UPSTREAM_A5_VALIDATION_SWEEP_2026_05_14.md`, 5 of 14 observed).

**Rule**: OL-143's decision tree treats L1 / L2 / L3 as mutually exclusive — pick exactly one. The
sweep shows **5 ops are L2+L3 hybrid**: they use MicroAPI Register-based compute (L2) for the hot
inner loop AND SIMT (L3) for a sub-phase (scatter / gather / shared-state accumulator) within the
SAME kernel binary. The tier model must be **multi-label, not single-selection**.

**Why it matters**:
- Single-tier classification routes the worker to ONE playbook (L1 mechanical OR L2 MicroAPI OR L3
  SIMT). Hybrid ops need BOTH the L2 and L3 playbooks loaded.
- Forcing one tier wastes iters: kw classifies L2, writes MicroAPI compute, then discovers the
  scatter/gather phase needs `Simt::AtomicAdd` — an out-of-budget rewrite. Multi-label upfront
  prevents this.
- The 5 hybrid ops include the most structurally important ports in the backlog:
  `moe_init_routing_v3` (17 files / 4671L, MoE routing core), `group_norm_silu`,
  `add_rms_norm_quant`, `repeat_interleave_v2`, `masked_select_v3`.

**Updated classifier** — replaces OL-143's exclusive tree with independent labels
`{L1, L2, L3, L4}`; L1 always true, the rest can fire together:
- **L2 fires** if `op_class ∈ {rmsnorm, rope, softmax, attention, layernorm, groupnorm, welford}`
  OR kernel src matches `__VEC_SCOPE__ | RegTensor< | MaskReg | CastTrait`
  OR `_def.cpp` matches `DT_FLOAT8 | DT_HIFLOAT8 | DT_FLOAT4`.
- **L3 fires** (independent of L2) if kernel src matches
  `__simt_vf__ | LAUNCH_BOUND | Simt::(GetThreadIdx|VF_CALL|AtomicAdd|UintDiv)`.
- **L4 fires** via structural signature (OL-156), not "has IsRegbase".

**Concrete hybrid example**: `moe_init_routing_v3` uses L2 for the dynamic-quant Cast chain
(MicroAPI) and L3 `Simt::AtomicAdd` for the expert-token count plus `__local_mem__` for first-index
state — both in the same kernel.
