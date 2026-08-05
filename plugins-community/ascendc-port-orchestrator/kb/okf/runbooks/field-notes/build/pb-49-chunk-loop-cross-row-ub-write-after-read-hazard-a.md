---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "chunk-loop CROSS-ROW UB write-after-read hazard — a SHARED per-tile buffer written by V in one grid-stride ROW and reloaded via MTE2 at the top of the NEXT row needs a `V→MTE2` fence at the ROW boundary (AscendC auto-syncs only MTE2→V)"
description: "<!-- applies_to_backend: all -->"
phenomenon: build_failure
signal:
  - "a kernel processing multiple rows per block via a grid-stride row loop, reusing the SAME UB working buffers each row, produces wrong output for some rows — dist"
confidence: single_run
original_id: PB-49
timestamp_inferred: true
tags: [ascendc, pb-49]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=AIV; op_class=all (any SIMD/AIV kernel whose grid-stride row loop reuses shared UB working buffers)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — same MTE2→V-only auto-sync model expected, no cross-arch witness)`

- **Severity**: HIGH — silent, timing-dependent wrong output in a kernel that compiles and runs. The single-row path PASSES; only the multi-row-per-block (rows = B·D > nblk) case corrupts.
- **Symptom**: a kernel processing multiple rows per block via a grid-stride row loop, reusing the SAME UB working buffers each row, produces wrong output for some rows — distinctively, **only the LAST row per block is correct** (it has no successor to clobber its in-flight writes). The corruption WIDENS with the per-row working-set size (state width N): masked at small N (the buffers drain within pipeline time), grows to ~2/3 of rows wrong at larger N.
- **Root cause**: the prior row's last V-pipe writes to the shared UB buffers are still in flight when the next row's first MTE2 reload (e.g. an `Af`/`B`/`C` `DataCopy`) issues. AscendC auto-inserts only the `MTE2→V` dependency (reload-then-compute), NEVER the reverse `V→MTE2` (compute-then-next-reload) across the loop boundary → the next row's reload races the prior row's compute. A `PipeBarrier<PIPE_V>` does NOT help (it orders V-vs-V, not V-vs-MTE2).
- **Fix (minimal)**: ONE `SetFlag/WaitFlag<HardEvent::V_MTE2>` at the row-loop top for `r>0` — NOT a heavy `PipeBarrier<PIPE_ALL>` (which also works but is a superset). Orders the prior row's V writes before this row's first MTE2 reload.
- **Distinct from PB-47/PB-48**: PB-47 = same-iteration UB reload→consume (intra-row). PB-48 = SIMT GM cross-iteration coherency (different memory space + no-fence remedy). PB-49 = SIMD/AIV UB shared-buffer reuse across ROW iterations, fixed by a row-boundary `V→MTE2` fence. Whole-row variant of PB-47.
- **Status**: architectural (AscendC auto-sync covers MTE2→V only).
- **Evidence**: selective_scan fwd-SIMD (2026-06-24, A5/Ascend950PR_957b/CANN 9.1.T500, PR #50). Pristine kernel: N=32 rows=114 → 76/114 wrong (only last-row-per-block correct); N=16 rows=114 → 13/114 wrong (reaches N=16 too, timing-fragile); one `V_MTE2` row-boundary fence → N∈{16,32,64} rows>56 all 0-wrong, customer N=16 B8/D192/L5000 0/1536, perf-neutral (+0.4%). NOTE: at L=5000 large-CH the customer shape did NOT manifest the hazard (latent — confirmed correct pre-fix); the leak only bit small-L (≲256) multi-row.
- **Detection heuristic**: any SIMD/AIV kernel with a grid-stride ROW loop reusing shared UB buffers, where multi-row-per-block output is wrong but single-row is correct AND only the last row per block survives → add a `V→MTE2` fence at the row-loop top.
- **Cross-reference**: PB-47 (intra-iteration sibling), PB-48 (SIMT GM sibling), EC-77 (the carry-fold RAW fence in the same op), OL-253 (N-adaptive chunk in the same op).

<!-- 迁移自 porter kb/target/ascendc/（PB-49，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
