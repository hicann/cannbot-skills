---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "An aclnn tiling-error code (561002 class) is a HOST tiling-function REJECTION, not a missing-kernel-binary — read `ASCEND_GLOBAL_LOG_LEVEL=3` FUNC/FILE/LINE first; AND before concluding \"the vendor op can't do this benchmark shape,\" verify the benchmark def comes from the correct/latest source (a vendor op may implement only ONE variant the WRONG/stale def doesn't match — the def, not the op, is the bug)"
description: "<!-- applies_to_backend: ascendc -->"
phenomenon: build_failure
signal:
  - "<!-- applies_to_backend: ascendc -->"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-219
timestamp_inferred: true
tags: [561002, ascendc, ol-219]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR; cann=9.1.T500,9.0.0; bisheng=n/a; op_class=vendor-op prior-art; mode=arch22_to_arch35 research`

**Principle**: when a vendor `aclnn*` op fails at runtime with a tiling-class error (e.g. `561002 = ACLNN_ERR_INNER_TILING_ERROR`, host tiling returned -1), the FIRST move is **NOT** to assume "the prebuilt kernel binary is missing this shape's tilingKey and we can recompile it." Set **`ASCEND_GLOBAL_LOG_LEVEL=3`** and read the `EZ9999 ... [FUNC:<name>][FILE:<op>_tiling.cpp][LINE:N]` TraceBack — it is the discriminator:
- FUNC = `CheckRequiredInOutExistence` / `CheckRopeExistence` / `CompareShape` / `Check*Shape` → **host tiling REJECTS the input combination** (it has no code path for it). The kernel is never reached → recompiling a kernel `.o` cannot help.
- Literal "tiling key <N> not found in kernel binary" / "kernel not registered" → recompiling for that key MAY help.

**Variant + def-source corollary (the load-bearing lesson)**: a CANN vendor op can implement only ONE variant of a named op — forward `aclnnSparseFlashAttention` on A5 is **MLA-only** (tiling unconditionally requires `queryRope`; nope-dim 512 + rope-dim 64; kv_head=1; attention_mode=2 = DeepSeek-MLA sparse attention). When a benchmark def doesn't match that variant you get 561002. **But that does NOT mean "the op is unportable" — first verify the benchmark def is from the CORRECT/LATEST source.** Here the two sources disagreed: `vendor/AscendOpGenAgent` (v1, github fork — provides util) had a **wrong** SFA def (`[1,64,2,64]` fp16 no-rope GQA → 561002), while `vendor/ascendc-kernelgen-data` (v2, the OFFICIAL gitcode npukernelbench) had the **correct** MLA def (`q[1,1,128,512]`+`query_rope[1,1,128,64]`, kv_head=1, attn_mode=2). The shipped vendor op **runs the v2 def cleanly** (verified on .171 NPU1 → out `[1,1,128,512]` bf16, finite, no 561002). So the **v1 def was the bug, not the op** — no kernel build / no re-author needed. Only when the def is confirmed-correct-and-latest AND still unsupported does authorship (`/ascendc-op-gen`) become the path. (Forward `sparse_flash_attention` tiling source IS compiled-only in `libophost_transformer.so` + not open-sourced on gitcode — relevant only IF a genuine rebuild were needed, which here it is not.) **General rule: when an orchestrator pulls op defs from one repo but the official defs live in another, audit the def source before trusting any "vendor op missing" verdict.**

**Concrete anchor** (diagnosis recipe — core of the CANN-op-compile skill):
```bash
export ASCEND_GLOBAL_LOG_LEVEL=3            # turns opaque 561002 into FILE:LINE + human reason
# benchmark 7_SparseFlashAttention fp16 [1,64,2,64] no-rope, sparse_mode=3 →
#   EZ9999: Shape of queryRope is nullptr [FUNC:CheckRequiredInOutExistence][FILE:sparse_flash_attention_tiling.cpp][LINE:1539]
# enumerate AOT prebuilt tilingKeys (proves key-universe is NOT the issue):
#   ls <CANN>/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/<op>/*.json → kernelList[*].tilingKey
# decode host constraints from the compiled .so when source absent:
#   strings .../libophost_transformer.so | grep -iE 'queryRope|CheckRequired|only support|must be'
```

## Evidence
- FA cohort `7_SparseFlashAttention` 2026-06-12 (owner-directed SFA probe + owner def-source challenge): (1) prior-session + a first echo framed 561002 as "only 2 AOT-prebuilt shape variants → tilingKey-not-found"; `ASCEND_GLOBAL_LOG_LEVEL=3` **disproved** it — failure is `CheckRequiredInOutExistence` L1539 (queryRope nullptr) before any key selection → 561002 is a tiling-rejection. (2) A first verdict CANNOT_BUILD was then ALSO wrong because it tested the **v1** def — the owner challenged "did you check v2?"; the **v2** (official `ascendc-kernelgen-data`) SFA def is the MLA shape (`[1,1,128,512]`+rope64, kv_head=1, attn_mode=2), and the shipped vendor op **runs it clean on .171 NPU1** (out `[1,1,128,512]` bf16, finite). → The v1 def was a benchmark bug; SFA is NOT unportable; no build/authorship needed. Double lesson: read the FUNC name for 561002, AND verify the def source before any "vendor op missing" verdict. Report: `workspace/7_SparseFlashAttention/SFA_COMPILE_PROBE_REPORT.md`; v2 wiring = ROADMAP/task #43. Cross-ref OL-188 (op-family→architecture class), `feedback_ground_in_source_of_truth_not_our_own_artifacts`, `feedback_simplest_probe_first`, memory `reference_sfa_aclnn_mla_only_561002`.

## Other instances (predicted)
- Any benchmark/op-gen reference that wraps a `torch_npu` vendor op (FIA `npu_fused_infer_attention_score`, incre-FA, paged-attn) failing with a 56xxxx tiling code — diagnose tiling-reject vs kernel-missing the same way. Any "compile a CANN op from source" attempt: first confirm the op's tiling/proto/kernel are on-disk-and-open-source vs compiled-only-closed before promising a rebuild.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-219（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
