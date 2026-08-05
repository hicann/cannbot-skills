---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Manual cube Mmad correctness on tail / rectangular tiles — k must be the REAL contraction extent (not kAlign), and a realM→mStep override must be gated to the NON-transpose load branch"
description: "In a hand-rolled cube Mmad, pass the REAL contraction extent K (not kAlign), and gate a realM→mStep override to the non-transpose load branch — both are silent on aligned/square tiles."
confidence: single_run
original_id: OL-224
classified_by: llm-assisted
timestamp_inferred: true
tags: [user-cube, optimization, ol-224, mmad, tail-tile, transpose, correctness]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR (V351/A5, arch35) / CANN 9.1.T500 / user-cube manual Mmad. Verified on Ascend950PR_957b.

**Principle (two distinct manual-cube tile bugs the library hides)**:

1. **Tail-K**: the fractal LOAD may use an aligned `kStep` (`kAlign = AlignUp(K, C0)`) for the NZ block layout, but `Mmad`'s `k` argument MUST be the REAL contraction extent `K`. Passing `kAlign` makes Mmad sum the `kAlign − K` padding rows (garbage/uninit L1) → wrong accumulation. Silent on K-aligned tiles (`K % C0 == 0`); only bites when `K < C0` or `K` is not a C0 multiple (e.g. a chunk-recurrent tail `curChunkSize = 4`). The library `MatmulImpl::SetSingleShape` uses the real `sk`, so only hand-rolled Mmad hits this.

2. **Transpose-A `realM` override**: a `if (realM != 0) mStep = ceil(realM/16)` refinement is only consistent when `mStep` tracks the M dim (the NON-transpose load branch). In the transpose-A load branch `mStep` is K-based, so overriding it with `realM` (an M quantity) corrupts the transposed load's K-fragment count. Silent on square tiles (`M == K` ⇒ M-based == K-based); corrupts rectangular tiles (`M != K`). FA's MAIN cube path (`flash_attention_score_cube.h`) uses `isLeftTranspose=false` everywhere, leaving the transpose-A path effectively untested — the first heavy transpose-A user (a state-update matmul `transA=true`) exposes it.

**CAUTION (regression scope, verified on disk 2026-06-16)**: FA's *wholeport* path (`wp_block_cube.h`) passes `isLeftTranspose=useDn`, which CAN be `true` — so fixing this `realM`-gating bug in the SHARED `regbase_matmul.h` template is NOT FA-neutral: it must be FA-regression-gated (especially wholeport + `useDn=true` cases) before landing, NOT bundled with a docs/KB change. (This is why the template fix is deferred to its own PR, not the KB entry.)

**Concrete anchor**:
```cpp
uint32_t kAlign = AlignUp(K, C0);               // fractal NZ block layout uses kAlign
Mmad(l0c, l0a, l0b, {.m = realM, .n = N, .k = K /* REAL, not kAlign */, ...});
if (!ifTranspose && mmParam.realM != 0) { mStep = CeilDiv(mmParam.realM, 16); }  // gate to non-transpose
```

**Evidence**: GDN `chunk_gated_delta_rule` regbase, 2026-06-16. Tail-K fix (`Mmad k = K`): the 4 `T=4` tail cases (idx 38–41) `final_state` went fs_ok 0.155 → 1.000. Transpose-`realM` gating: the entire 10-case rectangular cluster (`Dk=64, Dv=32`, idx 80/81/108/109/114/115/118–121) → out_ok=1.000 fs_ok=1.000; it also fixed the apparent "no-gamma out" failures (the "separate no-gamma bug" hypothesis was refuted by the data — the same transpose-A corruption fed both paths). Together: 108/122 → 118/122.

**Other instances (predicted)**: any hand-rolled cube on non-square / tail-shaped tiles — chunk-recurrent state updates, grouped/segmented matmuls.
