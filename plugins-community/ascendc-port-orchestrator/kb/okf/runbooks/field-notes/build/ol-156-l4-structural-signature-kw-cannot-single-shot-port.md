---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "L4 STRUCTURAL signature — kw cannot single-shot port these; route to researcher FIRST [V351, port_a3_to_a5_escalation]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all_port_a3_to_a5; phase=O2.5_classifier"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all_port_a3_to_a5; phase=O2.5_classifier"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-156
timestamp_inferred: true
tags: [ascendc, ol-156]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all_port_a3_to_a5; phase=O2.5_classifier`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: docs/analysis/UPSTREAM_A5_VALIDATION_SWEEP_2026_05_14.md — L4 false-positive in OL-143`
`supersedes: OL-143 §L4 trigger row`

**Rule**: OL-143's L4 trigger ("tiling needs `IsRegbaseSocVersion()` decision") is **defective** — 13 of 14 ACTIONABLE ports have ≥ 1 `IsRegbaseSocVersion()` call in host tiling. The check is near-universal for any A5-aware port, not exceptional.

The TRUE L4 signal is **structural complexity** that exceeds single-iteration kw scope. An op fires L4 when ANY of the following observed:

| Signal | Threshold | Rationale |
|---|---|---|
| arch35 file count | ≥ 5 | Multi-phase decomposition required; each phase has own dispatch + state |
| arch35 line count | ≥ 1500 total | Beyond single-iter kw context budget for confident author |
| Per-strategy file separation (OL-154) | ≥ 4 `_*_quant.h` / `_*_variant.h` files | Multiple specialization axes; tiling-key cross-product dispatcher |
| L2+L3 hybrid (OL-153) | ≥ 3 trigger categories per tier | Worker must reason about hot-loop MicroAPI + sub-phase SIMT in same kernel |
| Cross-phase `__local_mem__` shared state | ≥ 2 phases reading same UB region | Inter-phase ordering needed; kw single-spawn can't reason about both |
| Overflow-mode SPR usage (OL-148) | `SetCtrlSpr<60>` present | Op is in the perf-extreme tier; perf optimization is load-bearing |
| Cross-op DEPENDENCIES (W9) | ≥ 1 peer op router needs patching | Multi-op edit campaign; kw single spawn doesn't co-edit peers |

**Observed L4 ops in our backlog** (per sweep):

| Op | Triggers fired (count) | Files / Lines |
|---|---|---|
| `moe_init_routing_v3` | ALL 7 (filecount + linecount + per-quant + L2+L3 + `__local_mem__` cross-phase + `SetCtrlSpr` + cross-op deps) | 17 / 4671 |
| `group_norm_silu` | 4 (filecount + linecount + L2+L3 + `__local_mem__`) | 6 / 3495 |
| `add_rms_norm_quant` | 3 (linecount + L2+L3 + `__local_mem__`) | 4 / 1728 |
| `flash_attention_score` | 3 (filecount + linecount + cross-phase deps; subkernels not in arch35/ — likely 4+ when included) | 5+ / 1521+ |
| `repeat_interleave_v2` | 2 (L2+L3 + `__local_mem__`) — BORDERLINE | 3 / 1444 |
| `rope_with_sin_cos_cache` | 0 — NOT L4 despite large (3f/1538L, L1-mechanical) | 3 / 1538 |

**Action when L4 is detected**:

1. **DO NOT spawn kw first.** Spawning kw on an L4 op without researcher pre-analysis is the canonical waste pattern (cf. ada_layer_norm $69.20 postmortem).
2. **Route to `aog-researcher` first** with a structural-analysis brief asking:
   - What are the discrete phases? (preprocess / main / epilogue / etc.)
   - Which phases share UB state via `__local_mem__`?
   - Which quant-strategy axes apply? (per OL-154)
   - What's the tilingkey-bit layout?
3. **Researcher produces a structural-decomposition directive** (markdown in workspace/) listing each phase, its tier (L1/L2/L3), and the dispatch interface.
4. **Then spawn kw** with the directive — kw now has per-phase scope, doesn't have to discover structure.

**Why this changes orchestrator routing**:

- Current state machine: `await_worker` → kw spawn → (if PARTIAL or BLOCKED) → `await_researcher` → researcher → back to kw with directive
- Proposed for L4: `phase_o25_a3_ref` detects upstream + L4 → **`await_researcher` BEFORE any kw spawn** → researcher's structural-decomp directive → `await_worker` → kw

The pre-routing saves ≥ 1 kw spawn cycle ($5-15) per L4 op. Across the 3-5 clear L4 ops in cohort 2 (`moe_init_routing_v3`, `group_norm_silu`, `add_rms_norm_quant`, `flash_attention_score`, `repeat_interleave_v2`-borderline), the savings sum to $20-75.

**Detection at scan-time** (workflow Batch 4 follow-up — pending implementation):

```python
def is_l4_structural(op_dir: Path) -> dict:
    arch35 = op_dir / "op_kernel" / "arch35"
    files = list(arch35.glob("*.h")) + list(arch35.glob("*.cpp"))
    n_files = len(files)
    n_lines = sum(sum(1 for _ in f.open()) for f in files)
    blob = "\n".join(f.read_text(errors="ignore") for f in files)

    quant_strategy_files = sum(
        1 for f in files
        if re.search(r"_(dynamic|static|pertensor|pertoken|hif8|mxfp8|mxfp4)_quant\.h$", f.name)
    )
    has_l2 = bool(re.search(r"__VEC_SCOPE__|RegTensor", blob))
    has_l3 = bool(re.search(r"__simt_vf__|Simt::", blob))
    has_local_mem = bool(re.search(r"__local_mem__", blob))
    has_spr = bool(re.search(r"SetCtrlSpr<\s*60|GLOBAL_OVERFLOW_MODE_CTRL", blob))

    signals = {
        "filecount_5": n_files >= 5,
        "linecount_1500": n_lines >= 1500,
        "per_quant_4": quant_strategy_files >= 4,
        "l2_l3_hybrid": has_l2 and has_l3,
        "local_mem": has_local_mem,
        "spr_overflow": has_spr,
    }
    is_l4 = sum(signals.values()) >= 2  # ≥ 2 signals fire
    return {"is_l4": is_l4, "signals": signals}
```

**Anti-pattern**: classifying an op L4 just because it has any single complex feature. The threshold is ≥ 2 signals — a single trigger (e.g. `__local_mem__` alone) is consistent with L3 single-tier.

**Other instances (predicted)**: any future MoE / multi-quant / attention-with-state op; especially mxfp8 attention variants and grouped-matmul-fused-with-norm patterns.

**Cross-reference**:
- OL-143 (predecessor — L4 trigger row is now SUPERSEDED by this entry)
- OL-153 (L2+L3 hybrid — `l2_l3_hybrid` is one L4 signal)
- OL-154 (per-quant-strategy files — `per_quant_4` is one signal)
- OL-148 (SPR overflow — `spr_overflow` is one signal)
- OL-151 (`__local_mem__` — `local_mem` is one signal)
- `aog-researcher` skill — receives L4 structural-decomp briefs

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-156（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
