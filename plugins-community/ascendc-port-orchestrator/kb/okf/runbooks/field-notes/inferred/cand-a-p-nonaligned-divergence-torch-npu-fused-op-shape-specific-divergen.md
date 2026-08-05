---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "torch_npu fused-op shape-specific divergence at non-128-aligned 2H (op#11 case 32 ONLY confirmed)"
description: "Status: 1 confirmed datapoint, 6 suspect, 0 archived sweeps. NOT yet a generic pattern. Codex review 2026-04-28 flagged earlier A-P37 codification as over-generalized — moved here to candidates pendin"
phenomenon: build_failure
signal:
  - "Status: 1 confirmed datapoint, 6 suspect, 0 archived sweeps. NOT yet a generic pattern. Codex review 2026-04-28 flagged earlier A-P37 codification as over-gener"
confidence: inferred
status: stub
original_id: CAND-A-P-NONALIGNED-DIVERGENCE
timestamp_inferred: true
tags: [candidate, inferred, model.forward, cand-a-p-nonaligned-divergence]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Status**: 1 confirmed datapoint, 6 suspect, 0 archived sweeps. NOT yet a generic pattern. Codex review 2026-04-28 flagged earlier A-P37 codification as over-generalized — moved here to candidates pending durable evidence.

**What we KNOW (durable evidence)**:
- op#11 `[216, 5056]` mode=1 al=True: torch_npu output diverges from manual fp32 gpt-oss formula on a3 NPU. Reproduced by pp-2 in `/tmp/probe_case32_reproduce.py` (q_match=0.8552, sc_max_diff=0.10). Documented in `output/npukernelbench-a3/src/kernels/11_DequantSwigluQuant/probe_report.md` §pp-2.
- Benchmark commit 909454b (wabluy independent finding) confirms case 32 is where CANN deviates from its own documented formula — fix was to change `Model.forward` to use the documented CPU formula (which our kernel matches bit-exact at this shape too).

**What we DO NOT have durable evidence for**:
- Whether this affects ALL non-128-aligned shapes, or only `2H = 64 mod 128`, or only specific N×H combinations
- Whether mode=0 also exhibits this (orchestrator probe `/tmp/probe_mode0_align.py` suggested yes, but ephemeral /tmp script not archived)
- Whether other CANN aclnn fused ops (aclnnLayerNorm, aclnnGroupNorm, aclnnRMSNorm) have analogous regime
- The pad-to-128 strategy outcome (one ephemeral probe; needs archived reproducer)

**Promotion criteria** (move to platform_compat.md A-P3X if):
1. Archived shape sweep (≥3 ops or ≥10 N×H combos) showing the regime is reproducible across ops/cases
2. Probe outputs committed under `output/.../<op>/probes/probe_outputs/` (not `/tmp/`)
3. Cross-confirmed by ≥2 independent runs (ours + a5 / ds / kimi)

**Until promotion**: this is a single-data-point heuristic, not a KB pattern. Workers should NOT cite it as authoritative. The op#11 archive solution (benchmark spec change + CPU truth verification) handles the symptom for op#11; if a future op hits a similar regime, treat as new investigation.

**Related**: OL-91 (artifact evidence bar — this candidate fails by being /tmp-only), aog-self-critic C18/C23 (codifying narrative as pattern is a reward-hacking tell), CLAUDE.md "no CANN source copy".

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A-P-NONALIGNED-DIVERGENCE，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
