---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Reference-op run-to-run non-determinism — Phase O2.5 pre-flight + scope-by-construction"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "any benchmark/opgen op whose reference is torch_npu.npu_X or any CANN fused op"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-88
timestamp_inferred: true
tags: [ascendc, ol-88]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: pipeline / preflight / reference-determinism
- **Loaded by**: orchestrator (Phase O2.5 step 4 mandatory), aog-kernel-worker (Phase A KB load when `ref_determinism.json` exists in workspace)
- **Trigger**: any benchmark/opgen op whose reference is `torch_npu.npu_X` or any CANN fused op
- **Mandatory action**: orchestrator MUST run `python3 src/scripts/check_ref_determinism.py --op {slug} --workspace workspace/{op}/ --edge-inputs ... --model-py ...` AFTER edge_runner produces edge_dataset.pt and BEFORE spawning aog-kernel-worker. The script fires the reference 3× on identical inputs and emits `workspace/{op}/ref_determinism.json` listing per-output deterministic-vs-not verdicts.

- **Why this exists**: torch_npu CANN ops on Ascend950PR are not always run-to-run deterministic. If the worker writes a deterministic kernel against a non-deterministic reference, the verifier's bit-exact compare will FAIL no matter how correct the kernel is — and the worker can spend hours debugging "why doesn't my kernel match" before realizing the reference is the moving target.

- **Three confirmed non-det classes** (op#12 KvRmsnormRopeCache, 2026-04-24, 3-run probes):
  1. **Paged-attention block layouts** (`PA_BLK_BNSD`, `PA_BLK_NZ`): scatter races at the block-position level. 3-run max_diff ≈ 10-15.
  2. **`is_output_kv=False` semantic violation**: docstring says "returns None" for k_embed_ret/y_ret but CANN returns 4-tuple with uninitialized buffers — sometimes zero, sometimes garbage from prior NPU memory. 3-run pattern: r0=0, r1=6.78, r2=0 (clearly cold-buffer leak).
  3. **Norm cache_mode with duplicate indices**: `k_cache_out[b,n,idx,:] = k[b,n,s,:]` races when multiple (b,s) target same idx. 3-run max_diff ≈ 8.

- **What worker does with `ref_determinism.json`**:
  1. Phase A: read it before designing the kernel
  2. If `summary_per_output[<output_name>].verdict == "NON_DETERMINISTIC"`: drop that output from comparison OR zero-fill (matches common cold-buffer behavior). Document the scope decision in analysis.md `§"Reference determinism scope"`.
  3. If all outputs deterministic: proceed normally. Worker brief should still mention `ref_determinism.json` was checked (audit trail).

- **Pattern enforcement**: `case_gen.py` `index_range:<N>` invariant now defaults to **unique-by-default** (uses `randperm` + slice when n ≤ upper). Opt-in to duplicates via `tensor_inputs[i]["allow_dup_indices"] = True` only when the op semantically tests scatter-race behavior. This eliminates non-det class (3) at the case-generation source.

- **Cross-reference**:
  - OL-83 (torch_npu vs pytorch-native drift): static algorithm difference. OL-88 covers DYNAMIC run-to-run non-determinism — orthogonal class.
  - OL-87 (preflight benchmark reference run): companion preflight; OL-87 catches schema/crash issues, OL-88 catches non-det issues.
  - OL-66 (torch::zeros not stream-ordered): different but related — non-determinism from kernel side.

- **Evidence**: op#12 cold-start without OL-88 → worker spent 1 hour iterating on PARTIAL 18/50 + 7/27 before identifying CANN ref non-det as root cause. With OL-88 pre-flight, this should be discovered in 30 seconds before kernel design starts.

- **2026-04-27 empirical refinement — DO NOT extrapolate class-1 to all `torch_npu` inplace-scatter ops**: op#19 IndexPut Track 1 ref-determinism probe (18 cases of `torch.index_put_(accumulate=True)` at fp16/bf16, M ∈ {31..65536}) found **0/18 non-deterministic**, max_diff=0.0 across 5 runs. This contradicts an earlier extrapolation in `output/npukernelbench/src/kernels/19_IndexPut/verification_HODSA_attempt.json` which framed `index_put_` failures as OL-88 class-1. Lesson: class-1 was confirmed only on PA_BLK_BNSD/PA_BLK_NZ at op#12; do not generalize the verdict to other torch_npu inplace-scatter ops without a fresh `check_ref_determinism.py` probe. Probe report: `workspace/indexput/track1_falsification_report.md`.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-88（category=pipeline / preflight / reference-determinism，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
