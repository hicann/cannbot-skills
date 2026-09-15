---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Performance claims must be based on same-condition A/B data — never on speculation"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "any performance-related claim (\"no regression\", \"+X% improvement\", \"perf unchanged\")"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-27
timestamp_inferred: true
tags: [ascendc, ol-27]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process
- **Loaded by**: Builder, QA, Optimizer, Lead
- **Trigger**: any performance-related claim ("no regression", "+X% improvement", "perf unchanged")
- **Lesson**: After the E14 TQue refactor, CC claimed "no perf regression". In reality: (1) before/after ran on different NPUs (NPU 1 vs NPU 0), data not comparable; (2) F16/BF16 non-PingPong kernels had no perf data at all; (3) this conclusion was written into EXPERT_FEEDBACK.md and the session summary as "fact". Without user correction, this false conclusion would have become the basis for subsequent decisions.
  **Severity**: this is not a "slightly off" issue — it is **publishing a conclusion as fact with zero evidence**. In production this leads to:
  - Perf regressions hidden until discovered in production
  - Subsequent optimizations making decisions against a wrong baseline
  - Team trust in AI assistance tools collapses
  **Hard rules**:
  1. Performance claims must include **same-NPU, same-session, back-to-back** A/B data
  2. Every modified kernel must have a corresponding perf-data row
  3. If benchmark does not cover a given kernel, must label **"perf unverified"**
  4. **Never substitute "should be fine", "trend is consistent", or "numbers are close" for an A/B comparison**
  5. Cross-NPU / cross-session data can only be labeled "reference", not used for perf claims
- **Evidence**: E14 session (2026-04-07), user corrected three times

### Evidence (continued)

- **10_LayerNorm kw-1 (2026-05-02)**: prior archive PROGRESS.md (workspace/layernorm/, 2026-04-13) claimed 0.62× mean / 0.59× median; today's honest measurement on byte-identical kernel is **0.16× overall — 3.9× regression vs prior claim**. NPU 0 idle at measurement (no contention). Same kernel byte-for-byte. Suspect: CANN/bisheng version drift (CANN 9.0.0 b103 today vs whatever was active 2026-04-13), or per-AIV scheduling change in driver 25.7.rc1, or different reference-side optimization in the harness (newer aclnnLayerNorm fast path). **Implication: even byte-identical kernels can drift 3-4× across sessions. Historical perf numbers MUST be re-measured before being cited as a baseline.** "Do not trust archived `0.62×` without re-measuring" — this rule applies to ALL ops, not just LayerNorm.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-27（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
