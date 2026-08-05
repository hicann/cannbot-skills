---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "K2 workspace must be pre-zeroed when K1 cores may skip their slot (fused scatter+reduce)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "In a multi-kernel pipeline, K1 writes per-core local accumulations into ws[nblk, H], and K2 sums along the nblk dimension. If K1 partitions by rows (or any axis"
confidence: single_run
original_id: EC-37
timestamp_inferred: true
tags: [return, ascendc, ec-37]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: In a multi-kernel pipeline, K1 writes per-core local accumulations into `ws[nblk, H]`, and K2 sums along the nblk dimension. If K1 partitions by rows (or any axis with fewer logical units than nblk), some cores have `my_rows_=0` and `return` directly — their ws slices are never written, so K2 sum reads garbage.
- **Trigger conditions**:
  - Output shape `[nblk, H]` (per-core exclusive slot)
  - Logical work units `logical_work_units < nblk` (e.g., when BS < 56, rows do not fill all 56 cores)
  - K1 has `if (my_rows_ == 0) return;` fast-path
  - workspace comes from `torch::empty(...)` or similar uninitialised allocation
- **Root cause**: "per-core-exclusive workspace" does NOT imply "every core writes". Adjacent trap to OL-66 (torch::zeros not stream-ordered): OL-66 is about host-side zeros not being synchronised; this one is about device-side zeroing never happening at all.
- **Fix (3 options)**:
  1. **Preferred**: pybind calls `aclrtlaunch_memzero(ws_ptr, nblk*H*sizeof(T))` before K1, stream-ordered (automatically sequenced with K1/K2 on the same stream). Cost: one 56-core memzero launch (< 0.1 ms for 1 MB ws).
  2. K1 path guarantees every core writes (even if my_rows_=0, still Duplicate 0 into the core's own ws slot). Cost: every K1 variant must remember to handle this.
  3. Use `torch::zeros` instead of `torch::empty` (carries the OL-66 risk — host zeros ≠ stream-ordered, may overlap with K1).
- **Detection signal**: pass/fail flips across the N/nblk boundary — BS ≤ 32 PASS (some benchmark), BS ∈ (32, 2048) some cores idle → FAIL, BS = 2048 PASS again (all cores share work evenly). Failing case's output `[i][1]` (K2 product) has large-magnitude garbage (max_abs_diff ≥ 100), while `[i][0]` (K1 direct product) is completely correct.
- **Prevention (Phase B checklist)**: List every kernel's workspace; for each workspace, answer "do all cores in K1 write every slot? If not, who zeroes it before K2 reads?"
- **Evidence**: op#17 EmbeddingWithInitialLayernormBackward Phase D iter 1 (2026-04-23). `gnw_workspace[56, H]`, K1 partitioned by rows; when BS < 56, several cores skipped with my_rows_=0 → K2 sum read garbage → 30/57 FAIL. Fix: pybind added `K4_memzero(gnw_ws)` launch before K1 → 57/57 PASS.
- **Related**: OL-66 (torch::zeros not stream-ordered) — different failure mode same root theme.

<!-- 迁移自 porter kb/target/ascendc/（EC-37，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
