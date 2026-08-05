---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "T1-vs-CPU-fp64-truth triage before GM-dump bisection (precision-probe methodology)"
description: "verified_on: a5_ops:3_FusionAttention:case_b27a259d verified_on: a5_ops:3_FusionAttention:case_ad1de4ec verified_on: a5_ops:apply_adam_w_quant:case_ab7dc8ca verified_on: a5_ops:apply_adam_w_quant:case"
phenomenon: build_failure
signal:
  - "verified_on: a5_ops:3_FusionAttention:case_b27a259d"
confidence: inferred
status: stub
original_id: CAND-PP80
timestamp_inferred: true
tags: [candidate, inferred, ref_vs_kernel, ref_vs_cpu_truth, kernel_vs_cpu_truth, kernel_vs_cpu, ref_vs_cpu, cand-pp80]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: a5_ops:3_FusionAttention:case_b27a259d`
`verified_on: a5_ops:3_FusionAttention:case_ad1de4ec`
`verified_on: a5_ops:apply_adam_w_quant:case_ab7dc8ca`
`verified_on: a5_ops:apply_adam_w_quant:case_aa66206a`
`verified_on: a5_ops:apply_adam_w_v2:case_e5aa3648`
`verified_on: a5_ops:apply_adam_w_v2:case_f08368cf`

**Pattern**: For Pass B clusters that share dtype + shape signature (e.g. all-bf16-N≥16K-failing),
run T1-vs-CPU triage BEFORE GM-dump bisection — it's ~10× cheaper and often refutes the bug
hypothesis entirely.

**Method**:
1. Compute reference (`torch_npu.<op>`) output, kernel output, AND pure-CPU fp64 truth on the same input
2. Pairwise diff in 3 directions: `ref_vs_kernel`, `ref_vs_cpu_truth`, `kernel_vs_cpu_truth`
3. Decision tree:
   - `kernel_vs_cpu` flips ≈ `ref_vs_cpu` flips AND both large → **OL-83-amplified** (CAND-PP79); kernel and ref equally drift from truth; verifier methodology too tight; **stop** — no GM-dump needed
   - `kernel_vs_cpu` flips >> `ref_vs_cpu` flips → real kernel bug; **proceed to GM-dump bisection**
   - `kernel_vs_cpu` flips ≤ 1 per row AND `ref_vs_cpu` flips ≤ 1 per row → canonical OL-83 (1-ULP single-position drift)

**Cost comparison**:
- T1-vs-CPU triage: 1 probe script + 1 build + 1 ssh run (~5 min)
- GM-dump bisection: 5+ iters of kernel rebuild + dump-extract + diff-analyze (~30 min/iter, ~150 min)

**Concrete anchor** — pp-3 on op#9 9_TopKTopP cluster {8,17,26,35}:
- Brief asked for GM-dump bisection of 5 phases (Sort, MrgSort, Extract, cumsum-prob, top-p, sample)
- T1-vs-CPU triage in 1 iter showed `finite_max_diff=0.0` and `kernel_vs_cpu_flips ≈ ref_vs_cpu_flips`
- Falsified the GM-dump premise; no kernel rebuild needed; falsifies kw-6's MrgSort/Extract bug hypothesis at the same time

**Evidence**: op#9 9_TopKTopP pp-3 (2026-05-03) — single-op evidence so far.

**Other instances (predicted)**: Any precision-probe spawn where Pass B failures cluster by
dtype+shape signature and the brief jumps directly to phase-by-phase GM-dump should run T1-vs-CPU
triage first. Sort+select ops are the highest-yield candidates because tie-break ambiguity is
common; cumsum/reduction-chain ops also benefit (tests whether reference is bit-exact or also
drifts from CPU truth at fp32 precision limits).

**Promote when**: a 2nd probe spawn uses this methodology and produces analogous "GM-dump
unnecessary" outcome on a non-tie-break-related precision puzzle.

**Source**: op#9 9_TopKTopP pp-3 (2026-05-03), workspace/9_topktopp/probe_report.md §pp-3.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP80，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
