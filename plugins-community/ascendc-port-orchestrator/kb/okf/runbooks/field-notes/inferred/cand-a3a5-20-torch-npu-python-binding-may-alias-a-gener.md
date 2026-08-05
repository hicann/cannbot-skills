---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "torch_npu Python binding may alias a generic op name to one specific aclnn variant — verify routing before declaring a port verified"
description: "Source: workspace/top_k_top_p_sample/knowledge_update.md F1 (kw-1, 2026-05-14) Scope: soc=Ascend950PR; cann=9.0.0; op_class=op_family_v1_v2_variants; phase=Phase D verify Principle: when an op family"
phenomenon: build_failure
signal:
  - "Source: workspace/top_k_top_p_sample/knowledge_update.md F1 (kw-1, 2026-05-14)"
confidence: inferred
status: stub
original_id: CAND-A3A5-20
timestamp_inferred: true
tags: [candidate, inferred, binary_info_config.json, verify.py, aclnntopktoppsample, aclnntopktoppsamplev2, cand-a3a5-20]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: workspace/top_k_top_p_sample/knowledge_update.md F1 (kw-1, 2026-05-14)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=op_family_v1_v2_variants; phase=Phase D verify`

**Principle**: when an op family ships as V1/V2 (or V1/V2/V3) sibling aclnn entries with different I/O signatures, the `torch_npu.<op>` Python binding does NOT necessarily call the variant whose name matches. The binding may be coded to dispatch to a fixed variant (typically the newest one) regardless of caller-supplied argument count. Porting only ONE variant to A5 and then verifying via `torch_npu.<op>(...)` can exercise the WRONG variant — producing a 0/N FAIL with EZ1001 even though the ported variant's build + registry install are correct.

**Detection signal**: verify reports `EZ1001 / Get regInfo failed / does not support opType [<OpName>V2]` while the kernel installed and the `binary_info_config.json` registry entry are both for `<OpName>` (V1 — no V2 suffix). The variant mismatch in the error message confirms the harness routed past V1 to V2.

**Mitigation paths (decision rule)**:
- **Path A — direct-aclnn ctypes wrapper**: write `verify.py` that calls `libopapi.so::aclnn<OpName>` directly via ctypes (bypassing torch_npu). Smallest scope; validates the ported variant specifically. Recommended when the deliverable is "verify V1 port artifact correctness end-to-end".
- **Path B — port the sibling variant too**: stage V2 artifacts identically to V1, build, install. Recommended when the deliverable is "make the user-visible `torch_npu.<op>` binding actually work on A5".
- **Path C — accept PoC scope**: no further work; document the variant-routing gap. Recommended when the deliverable is upstream-staging-only (e.g., PR4778-style review), not end-to-end NPU validation.

**Concrete anchor** (top_k_top_p_sample, 2026-05-14):
- V1 `aclnnTopKTopPSample`: 4 inputs + 3 attrs + 2 outputs.
- V2 `aclnnTopKTopPSampleV2`: 5 inputs + 6 attrs + 4 outputs.
- `torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q=..., eps=..., is_need_logits=..., top_k_guess=...)` takes V1-style 7 args **but routes internally to `aclnnTopKTopPSampleV2`**. No torch_npu Python entry calls V1.

**Generalization**: family-ported ops (V1/V2 variant pairs, or any op family where a Python binding name does not 1:1 map to a single aclnn entry) MUST verify variant routing before reporting verify PASS/FAIL. The `applies_to` op-classes most at risk: numeric sampling primitives, quant variants (per-tensor vs per-token vs MXFP), attention variants (V1/V2/V3 with extended kv-cache args), and any op that has been API-evolved in CANN ≥ 8.0.

**Other instances (predicted)**: `top_k_top_p_sample_v2`, `apply_rotary_pos_emb_v2`, `swiglu_quant_v2`, `dequant_swiglu_quant`, `npu_moe_init_routing_v3`, any op-family with a `_v[0-9]+` aclnn entry where the torch_npu Python binding name is shared.

**Promotion gate**: needs ≥ 1 additional op-family confirmation (a second V1/V2 case where the harness routed past the ported variant) before promotion to OL/P-P. Auto-promote pipeline (Mode 5) should run C39 dry-run on a sibling op-family before promotion.

**Cross-ref**:
- OL-158 (companion: Phase C build+register activation criteria — Phase C may succeed yet verify still fails due to variant-aliasing)
- CAND-A3A5-15 (registry-install procedure that this trap can mask)
- OL-131 (peer-router edits — different concern: router aliases ARE intended, this entry is about variant aliases the operator-porter didn't expect)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-20，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
