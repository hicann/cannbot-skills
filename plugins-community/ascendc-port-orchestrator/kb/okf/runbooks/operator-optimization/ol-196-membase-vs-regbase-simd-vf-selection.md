---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Membase vs Regbase selection + Reg vector __simd_vf__ programming entry (CANN 9.0.beta1 doc-distill)"
description: "AscendC SIMD has two models: Membase (LocalTensor↔UB per API) and Regbase (RegTensor register-resident chains, tiled by VL). Regbase suits hot vector chains; Membase suits bulk streaming."
confidence: single_run
original_id: OL-196
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-196, simd, regbase, membase]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
AscendC SIMD vector code can be written in **two orthogonal programming models** that differ in where the per-iteration intermediate values live:

1. **Membase (memory-based, historical/default)** — every basic API (`Add`, `Mul`, `Exp`, `WholeReduceMax`, …) reads its inputs as `LocalTensor` from Unified Buffer and writes its output as `LocalTensor` back to UB; intermediate results round-trip UB between two API calls. A whole `LocalTensor` (potentially much larger than one vector-register width) is processed per call, with no user-visible tiling.

2. **Regbase (register-based, new Reg vector API)** — per-iteration values live in explicit registers (`RegTensor<T>`, `MaskReg`, `AddrReg`, `UnalignRegForLoad/Store`). Compute APIs (`AscendC::MicroAPI::Add(dst, src0, src1, mask)`, …) operate register→register; load/store APIs (`LoadAlign`/`StoreAlign`, `LoadUnAlign`/`StoreUnAlign`) move data UB↔registers under explicit user control. Intermediates stay in the register file across a compute chain — **no UB round-trip** — but the user MUST tile by `VL` (Vector Length), since a `RegTensor` holds exactly `VL/sizeof(T)` elements per register.

The two compose at the call-graph level: on Regbase-enabled archs a high-level or basic API may itself be built on the Reg API, and user code may call the Reg API directly inside a `__simd_vf__` function.

**Selection criteria** (from the Reg矢量计算编程 doc §"Regbase和Membase编程调用层级"):

| Op characteristic | Pick Membase | Pick Regbase |
|---|---|---|
| Multiple back-to-back vector ops on the same buffer (e.g. `Sub→Exp→Mul→Add` row-softmax body) | — | ✅ register-resident chain skips UB load/store between steps |
| Data fits whole-`LocalTensor` API calls, no per-VL tiling needed | ✅ simpler code, no manual `repeatTimes` loop | — |
| Hot inner loop, compute-bound, vector pipe is the bottleneck | — | ✅ eliminating UB round-trips frees vector-pipe slots |
| Bulk streaming (one Load → one compute → one Store, no intermediate reuse) | ✅ no Regbase benefit, and Regbase forces VL-tiling overhead | — |

**Scope / caveats**: Regbase availability is **per-arch**. Doc-only distillation from the CANN 9.0-beta1 programming guide (NODE-13 markdown extraction); NOT yet correlated to a measured kernel speedup in this KB, and bit-pattern equivalence between a Membase-form and a Regbase-form of the same op on Ascend950PR_9579 (arch35/V351) is left to a future empirical OL.
