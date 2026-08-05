---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Deterministic cross-core reduction via per-core private GM slots + fixed-order fold, not shared-cell atomic-add"
description: "SetAtomicAdd into a shared GM cell is run-to-run non-deterministic; write each core's partial to its own GM slot then fold in fixed order for bit-identical output. Cost ≈ +4-9% device-time."
confidence: single_run
original_id: OL-252
classified_by: llm-assisted
timestamp_inferred: true
tags: [determinism, optimization, ol-252, cross-core-reduction, atomic-add]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** any reduction that accumulates partials from many cores into one GM cell, where strict determinism is required (owner mandate "all ops deterministic", or bit-reproducible gradients).

**The problem.** Accumulating per-core partials into one GM cell via `SetAtomicAdd` is **non-deterministic run-to-run** — the hardware atomic-add order varies, so the fp sum varies below the last bit.

**The fix.** Replace the shared-cell atomic with:
1. **Each core writes its partial into its OWN private GM slot** (`workspace[core_id * stride + idx]`, no contention), then
2. **A fixed-order accumulation** folds the per-core slots — a host-side `.sum(0)`, or a second deterministic-order reduce kernel.

Result: bit-identical across runs (`max_run_diff == 0`). Because only the summation order changes vs an accepted baseline, precision is structurally unchanged.

**Cost (grade by device-time, not wall).** In selective_scan bwd: msprof DEVICE-TIME ≈ **+4% on the grad kernel / +9% full-op**. The det path adds ~5 device-ops/launch (per-core-slot zero-init + the host `.sum` reduce). An early ~7.6% wall figure sits between the two device-time numbers. A second **on-device ordered-reduce kernel** recovers most of the +4%→+9% gap vs a host `.sum`.

**Scope test — which grads/outputs need this:** any output written by >1 core with a reduction over the core-partitioned axis. **Audit ALL such outputs, not just the obvious ones** — in selective_scan bwd the non-determinism was in grad_B/grad_C (reduce over `d`) AND grad_A/grad_D/grad_dbias (reduce over `b`), i.e. 5 of 8 grads, not the 2 first suspected.

### Concrete anchor
```cpp
// BEFORE (shared cell, non-deterministic):
SetAtomicAdd<float>();
DataCopy(gmOut[idx], partial);
SetAtomicNone();

// AFTER (own slot + fixed-order fold):
DataCopy(gmWorkspace[blockIdx * outLen + idx], partial);
// then host: grad = workspace.view(num_cores, outLen).sum(0)   // or a fixed-order reduce kernel
```

### Evidence
selective_scan bwd-SIMD (Ascend950PR, CANN 9.1.T500, bisheng=AIV, 2026-06-24, PR #53). All 8 grads `max_run_diff = 0` at N∈{16,32,64} × fp32/fp16/bf16 including customer B8/D192/L5000. Cost (msprof device-time) ≈ +4% grad-kernel / +9% full-op (honest, recoverable; early ~7.6% was wall).

### Other instances (predicted)
Any backward op with cross-core grad reduction (layernorm/softmax/attention grads); any forward reduction (Σ over a core-split axis) needing bit-reproducibility; histogram/scatter-add accumulators.
