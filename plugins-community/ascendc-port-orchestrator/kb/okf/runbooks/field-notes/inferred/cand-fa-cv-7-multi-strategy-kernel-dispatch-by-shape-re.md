---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Multi-strategy kernel dispatch by shape regime — specialized variants per dimension class"
description: "verified_on: cv-agent rms_norm multi-kernel architecture (merge_n / single_row / splitd) + FA variant dispatch table (s1s2_bn2gs1 / var_len_score / var_len_score_sab) Pattern: When an op's optimal com"
phenomenon: build_failure
signal:
  - "verified_on: cv-agent rms_norm multi-kernel architecture (merge_n / single_row / splitd) + FA variant dispatch table (s1s2_bn2gs1 / var_len_score / var_len_scor"
confidence: inferred
status: stub
original_id: CAND-FA-CV-7
timestamp_inferred: true
tags: [candidate, inferred, merge_n, single_row, splitd, s1s2_bn2gs1, var_len_score, cand-fa-cv-7]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent rms_norm multi-kernel architecture (merge_n / single_row / splitd) + FA variant dispatch table (s1s2_bn2gs1 / var_len_score / var_len_score_sab)`

**Pattern**: When an op's optimal compute strategy depends on input shape dimensions, emit N specialized kernel variants dispatched by shape class at runtime, rather than a single generic kernel with internal branches.

**Mechanism**:
1. Analyze op's compute pattern for dimension-dependent behavior (reduction axis size, gather dim, matmul M/N/K ratio)
2. Define shape classes (e.g., large-N merge, small-M single-row, small-N split-D for rms_norm; BSH/BNSD vs TND/varlen for FA)
3. Emit one kernel variant per shape class, each with optimal tiling (block size, loop ordering, buffer sizing)
4. Host-side dispatcher selects variant at runtime based on input tensor shapes

**Concrete example (rms_norm)**:
- `merge_n` variant: N ≥ 4096 → reduce rows in blocks, merge partial results → large N throughput
- `single_row` variant: M ≤ 16 → one block per row, no cross-block sync → low latency for small M
- `splitd` variant: N < 256 → split along D dimension, parallel VEC reduce per chunk → handles tiny N

**Concrete example (FA — per F10 variant dispatch finding)**:
- `s1s2_bn2gs1` variant: BSH/BNSD/SBH fixed-shape → matmul::Matmul + KFC implicit sync, zero CrossCore
- `var_len_score` variant: TND/variable seq length → tile-MMAD + 12-flag CrossCore triple-buffer
- `var_len_score_sab` variant: TND + sparse attention → same SYNC_* scheme

**Why one generic kernel is worse**: Internal branches on shape dimensions cause divergent tiling, wasted buffer allocation (max of all variants), and suboptimal VEC/Cube utilization for the actual shape. Variant-per-class eliminates runtime branches and allows per-class buffer sizing.

**Detection**: count kernel variants per op directory (`ls kernel/*.cpp | wc -l`). Single `.cpp` with if/else on shape → gap. Multiple `.cpp` with shape-class naming (merge_n / single_row / splitd) → pattern applied.

**Evidence**: cv-agent `rms_norm/kernel/` has 9 .cpp files (3 variants × 3 dtypes). FA variant dispatch table verified by main+independent reviewer+DS independent grep of CANN arch22 source (2026-05-23).

**Cross-ref**: CAND-FA-VARIANT-DISPATCH (FA_CLASS_DESIGN_NOTES.md#fa-v220-decision-tracking-ds — the shape-regime → CANN-variant → KB-pattern mapping table), CAND-FA-CV-4 (cube/vec separation — each variant has own cube.h+vec.h pair), §F10 FA-class problem solution

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-7，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
