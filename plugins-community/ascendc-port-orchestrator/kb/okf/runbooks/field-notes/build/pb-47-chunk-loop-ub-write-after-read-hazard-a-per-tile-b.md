---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "chunk-loop UB write-after-read hazard — a per-tile buffer RELOADED via MTE2 at the top of each loop iteration and consumed by V in the SAME iteration needs a `V→MTE2` fence at the ITERATION BOUNDARY"
description: "<!-- applies_to_backend: all -->"
phenomenon: build_failure
signal:
  - "every iteration wrong EXCEPT the last (the last iteration has no successor reload to overwrite its buffer). The corrupted value is the buffer contents from a LA"
confidence: single_run
original_id: PB-47
timestamp_inferred: true
tags: [ascendc, pb-47]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=AIV; op_class=all (chunked/tiled kernels reloading a per-tile input via MTE2)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — not retested; same AIV V/MTE2 parallel-pipe architecture, expected to apply, no cross-arch witness yet)`

- **Severity**: HIGH — deterministic, input-dependent silent data corruption. The kernel compiles and a single chunk may PASS; only multi-chunk runs corrupt.
- **Symptom (distinctive signature)**: **every iteration wrong EXCEPT the last** (the last iteration has no successor reload to overwrite its buffer). The corrupted value is the buffer contents from a LATER iteration that the next MTE2 reload has already written.
- **Root cause**: when a UB buffer is RELOADED via MTE2 at the TOP of each iteration of an outer loop (e.g. an L-chunk loop) AND consumed by a V-pipe op LATER in the same iteration, the iteration boundary must carry a `V→MTE2` fence. AscendC's queue EnQue/DeQue only orders MTE2→V, never V→MTE2; without an explicit boundary fence the NEXT iteration's MTE2 reload overwrites the buffer before the CURRENT iteration's V op has read it. This is the CROSS-ITERATION analogue of PB-17 (which is the cross-ROW alias variant within a single fused `ProcessRow`).
- **Distinct from PB-17**: PB-17 is the P-P65 alias case — two UB buffers aliased within `ProcessRow`, V-write near end of row vs MTE2-write near start of NEXT row. PB-47 is NOT an alias — it is the SAME buffer reloaded each loop iteration. Same underlying hazard class (the AIV V and MTE2 pipes run in parallel and AscendC auto-syncs only MTE2→V), but the trigger context (chunk-loop reload vs intra-row alias) and the diagnostic signature ("every chunk wrong except last") differ enough to catalogue separately.
- **Workaround**: insert a `V→MTE2` fence at the chunk-loop TOP for every non-first iteration — e.g. `if (l0 > 0) SyncVtoMTE2();` (`SetFlag`/`WaitFlag<HardEvent::V_MTE2>`). ~8 lines, cheap relative to the per-chunk compute.
- **Status**: OPEN (architectural — AIV pipe parallelism, not a CANN bug).
- **Evidence**: selective_scan_source_a5 bwd_simd ③ grad_z bug (2026-06-22, A5/Ascend950PR_957b/CANN 9.1.T500). PASS-A L-chunk loop loaded `Ct` (MTE2) and consumed it at the ysc `Mul(prodC, xall, Ct, CN)` (V pipe); the boundary only fenced MTE3→V, so every non-final chunk read the LAST chunk's `Ct` (at L=512, chunk0 read `C[l=256]`). Root-caused by UB-instrumentation (dump UB→GM-scratch, read back from host) — smoking gun: `Ct in UB == truth C[l=256], diff 0.0`. Fix = `if(l0>0) SyncVtoMTE2();` at the chunk-loop top → grad_z MERE 1.499 → 1.85e-7. Whitebox-derived (UB-instrument), not guessed.
- **Evidence (fence survives vectorization)**: selective_scan_full_grad bwd 2.69× scan-vectorization (PR#37, merged main `bda9cb3c`, 2026-06-22, same env). The 2.69× opt re-wrote PASS A's per-chunk scan into the `[l*N+n]` Hillis-Steele form (P-P106) and added a reverse-suffix HS on PASS B — re-exercising the SAME chunk-loop that carries this `V→MTE2` boundary fence. Precision held 30/30 truth-backed after the rewrite, confirming the fence is still required and correctly placed under the vectorized scan (a missing/misplaced fence would have re-surfaced the "every chunk wrong except last" signature). The per-chunk-boundary fence and the intra-chunk scan are independent — both needed.
- **Detection heuristic**: for any multi-chunk / multi-tile loop that reloads a per-tile input buffer via MTE2, check whether that buffer is consumed by a V-pipe op LATER in the same iteration with no `V→MTE2` event between the V consume and the next iteration's MTE2 reload. If the precision signature is "all tiles wrong except the last", suspect PB-47 first.
- **Cross-reference**: PB-17 (the cross-ROW alias sibling — same V→MTE2 hazard class), EC-13 (HardEvent sync list), P-P106 (the L-chunk + parallel-scan pattern this surfaced under).

<!-- 迁移自 porter kb/target/ascendc/（PB-47，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
