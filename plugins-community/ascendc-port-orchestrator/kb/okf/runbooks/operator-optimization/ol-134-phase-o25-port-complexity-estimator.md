---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Phase O2.5 port-complexity estimator from V220 file structure"
description: "arch35 ports vary ~10x in complexity; the V220 file structure (macro guard, kernel-file count, peer deps, arch35 presence) is a deterministic predictor used to populate Phase O2.5 cost estimates before spawning the worker."
original_id: OL-134
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-134, port_a3_to_a5, cost-estimation, phase-o25]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

arch35 ports vary ~10× in complexity (a 1-line strip vs a 4-file new variant). The existing
V220 file structure of an op is a deterministic predictor of how big the port work will be —
use it to populate Phase O2.5 cost estimates BEFORE spawning the worker.

**Applies to** `soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5`. Do NOT apply to
V220-native kernel creation or backward port modes: the estimator's signal
axes (V220 macro guard, peer count from CMakeLists, arch35 dir presence) only exist when
porting an existing V220 kernel.

### Estimator rules

| Signal in V220 master | Estimated port complexity |
|---|---|
| `__NPU_ARCH__ == 3003 \|\| 3113` macro guard present in kernel `.cpp` | Mode A — 15 min (def.cpp + binary.json only) |
| Single `op_kernel/<op>.h` (~< 800 lines) + no macro guard | Mode B-simple — 1-2h (strip dav_c220 + write apt.cpp + binary.json) |
| Variant-split: 2-3 sibling `.h` files in `op_kernel/` | Mode B-medium — 3-4h (replicate variant split in arch35/, individual TILING_KEY dispatch) |
| Variant-split: 4+ sibling `.h` files OR cube+vec fused (matmul + softmax + quant in one kernel) | Mode B-complex — 4-8h (write an explicit AscendC template-assembly plan before implementation) |
| `DEPENDENCIES <peer_op>` in CMakeLists.txt with peer count > 0 | + cross-op router patch per peer per OL-131 — 30 min/peer |
| `op_graph/<op>_proto.h` present | + GE proto contract check (target-agnostic; usually no extra work) |

### Concrete anchor (use in Phase A analysis.md)

```bash
op=<op_name>
op_path=~/workspace/cann/ops-nn/<cat>/${op}
n_kernel_files=$(ls ${op_path}/op_kernel/*.h 2>/dev/null | wc -l)
has_macro_guard=$(grep -l "__NPU_ARCH__ == 3003" ${op_path}/op_kernel/*.{cpp,h} 2>/dev/null | wc -l)
peer_deps=$(grep -oE "DEPENDENCIES [a-z_]+" ${op_path}/op_host/CMakeLists.txt 2>/dev/null | wc -l)
echo "complexity: $n_kernel_files files, macro_guard=$has_macro_guard, peer_deps=$peer_deps"
```

### Evidence (cross-op)

- `top_k_top_p_sample_v2`: 3 kernel files + macro guard present → Mode A (~15 min host-only).
- `ctc_loss_v3`: 1 kernel file (830 lines) + no guard + 1 peer dep → predicted Mode B-simple,
  ~2h router-patch + staging. **Measured 2026-05-12 (PR4778 finalize, kw-1)**: ~30 min actual
  staging — zero router patches (the peer dep was build-system-only per OL-131 Step-2) AND
  zero kernel-side edits (PR4778's arch35/<op>.h was already cleanly authored; P-P90 audit
  returned 0 matches). The 2h budget is conservative when PR4778 is the ready-made source;
  ~30 min covers staging the 5 files + def.cpp patch + 1 peer CMakeLists edit.
- `gather_elements_v2`: 4 kernel files... (source text truncated here).

Source: cann-learner CAND-A3A5-9, promoted 2026-05-12 (8 ops cross-op evidence).
