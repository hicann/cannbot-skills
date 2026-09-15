---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "KB-sufficiency audit on SIMT scatter — op#21_Scatter doesn't apply P-P67/OL-67 (KB knows, archive doesn't)"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "investigating PARTIAL/FAIL ops that should be solvable per current KB"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-107
timestamp_inferred: true
tags: [ascendc, ol-107]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — per-archive audit
- **Category**: process / KB-coverage
- **Loaded by**: Lead/Reviewer when checking whether existing archives apply current KB
- **Trigger**: investigating PARTIAL/FAIL ops that should be solvable per current KB

### Lesson (gap-finding via existing-archive audit)

When an op's PARTIAL/FAIL signature matches a KB pattern's symptom exactly,
the question is not "what's missing in KB" but "did the archive apply the
pattern". For op#21_Scatter (29/47 PARTIAL):

**KB says** (already in repo, no gap):
- P-P67 (CRITICAL): scatter on duplicates is PyTorch-UB — kernel MUST NOT
  use atomic/multi-core scatter; ship single-core sequential scan
- OL-67: Two-phase int32 atomicAdd pattern (deterministic by construction)
- OL-90: Verifier-side alt-ref mitigation for non-deterministic refs

**Archive does**:
- Direct parallel `atomicAdd(output_fp32, val)` per line 99 — non-deterministic
  fp summation order → MARE jitter on duplicate-index cases
- Two-phase scratch but still fp32 atomicAdd → same problem at phase 2

**Why this matters for KB process**:
- 29/47 fail isn't "KB unable to fix"; it's "archive predates pattern adoption"
- A regenerated kernel via /ascendc-op-gen with current KB SHOULD apply P-P67/OL-67
- Validation: re-run /ascendc-op-gen on op#21 spec; expected: kernel uses int32
  atomicAdd two-phase OR single-core sequential

### Generalizable principle

For PARTIAL/FAIL ops in our 70-archive set:
1. **Map each fail signature to existing KB entries**
   (atomicAdd-non-det → P-P67/OL-67; hw transcendental → OL-103; etc.)
2. **Audit archive for pattern application** (grep for the pattern's primitives)
3. If archive applies it → real KB gap (escalate to OL-106-style PENDING entry)
4. If archive doesn't apply it → **regenerate** the kernel with /ascendc-op-gen
   (current KB now active) — fix should apply automatically
5. Codify the RE-GENERATION outcome as evidence for the pattern entry

### Concrete next-session candidates by audit class

Per this audit, op#21 likely fix path = re-generation, not KB enrichment.
Other archives likely in same class:
- op#19_FusedResidualRmsNormBackward — pre-OL-101 (host-precompute) era
- op#28 was already host-precompute-applied (FIXED this session)
- ops with hw transcendentals + fp16 PARTIAL — pre-OL-103 framework

### Self-critic per V3.3.13

1. Empirical: ✅ grep-confirmed P-P67's primitives ARE in KB, NOT in op#21 archive.
2. Achievable bar: ✅ **VALIDATED 2026-04-29 same day** via `scatter_kbtest-kw-1`
   aog-kernel-worker regen. Reached **45/47 (95.74%)** under CPU-truth (vs archive
   29/47), **+16 cases** improvement. KB scan surfaced P-P67/OL-67/OL-90 all
   3 via KB_INDEX.md. OL-67 was the active rule (generalized to fp32 scratch
   for float scatter-add; archive used native fp16 atomicAdd). P-P67/OL-90
   correctly NOT applied (op#21 is reduce='add' commutative-deterministic,
   not reduce=None PyTorch-UB). 2 residual fp32 MARE failures are intrinsic
   parallel-vs-sequential atomicAdd reorder at near-zero outputs (1 fp32
   ULP absolute = 9.54e-7) — out of scope for KB-pickup audit.
3. Reward-hack scan: ✅ NOT magic-fixing op#21; honest finding archive
   predates 2026-04-{12→29} KB additions, regeneration applies them.
4. Sealed reproducibility: ✅ Agent PROGRESS.md at
   `workspace/scatter_kbtest/PROGRESS.md`; per-case MERE/MARE in
   `precision_results.json`; kernel binary on npu_dev3.
5. Generalization: ✅ Methodology validated. Other pre-pattern archive candidates:
   - op#19 FusedResidualRmsNormBackward (pre-OL-101 host-precompute era)
   - op#16 Batched2DRopePos (pre-OL-103 Tier framework — but that's Tier-2-bound,
     different class)
   - Most Q4 cann-better ops in `cann_vs_ours_cpu_truth_report_2026_04_29.md`
     are Tier-2-bound (different fix path: OL-103 sw fp32)

### Validation outcome 2026-04-29 (same day as codification)

OL-107 prediction CONFIRMED on 3 axes:
- KB **surfacing** sound — KB_INDEX.md scan finds relevant entries
- KB **applicability decision** sound — agent correctly distinguishes
  reduce='add' commutative case from reduce=None PyTorch-UB case
- KB **action mapping** sound — OL-67 generalized correctly from int32
  histogram form to fp32 scratch for float scatter-add

**No KB enrichment needed for KB-pickup mechanism**. Recommended next
ROI: regenerate other suspect-pre-pattern archives. Tier-2-bound ops are
a separate path (OL-103 sw fp32 transcendentals, gated on OL-105
standalone debug methodology).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-107（category=process / kb-coverage，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
