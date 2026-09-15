---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Hardware-accelerator typed-config struct — use the typed config, not the generic policy"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "about to call <Accelerator>Impl<…, MM_CFG>::Init(...) or analogous typed-init signature on a hardware-accelerator API"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-91
timestamp_inferred: true
tags: [init, ascendc, ol-91]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: platform_compat
- **Loaded by**: aog-kernel-worker when target op uses a specialized hardware accelerator API (cube unit, sort engine, scan engine, etc.) whose `Init` signature takes a typed config struct rather than the underlying policy enum/struct
- **Trigger**: about to call `<Accelerator>Impl<…, MM_CFG>::Init(...)` or analogous typed-init signature on a hardware-accelerator API
- **Principle**: Hardware-accelerator APIs commonly have **two related types**:
  - A generic *policy* type (a config struct describing the operation regime; e.g. `MatmulConfig CFG_NORM`)
  - A typed *static-config* struct that **wraps** the policy and adds shape/sizing fields (e.g. `MatmulApiStaticTiling`)

  The `<Op>Impl<MM_CFG>` template instantiates against the **typed-config struct, not the policy**. Passing the policy directly (`MatmulImpl<…, CFG_NORM>` style) fails with "no member named '<field>' in '<PolicyType>'" because the impl's static_asserts and `if constexpr (MM_CFG.<field> == −1)` checks reach for fields the policy doesn't have.

  **The fix is structural**: instantiate against the typed-config struct, set `.cfg = <policy>` for the operation regime, and leave shape-runtime fields as a sentinel value (typically `-1`) so the impl reads them from runtime tiling instead. The sentinel-vs-constant split lets the compiler eliminate GM reads for any field set to a constant at compile time — a per-field perf unlock independent of the typed-config issue.
- **Three-step pattern**:
  1. **Use the typed-config struct in `<Op>Impl` instantiation**, not the policy directly. The error message "no member named X in `<PolicyType>`" is the canonical sign you've made this mistake.
  2. **Set policy as a member**: `t.cfg = <POLICY>` (e.g. `t.cfg = CFG_NORM`).
  3. **Lift shape-independent fields to compile time**: set values for fields that are constant per kernel (block sizes, depth, step, double-buffer flags, iterate order); leave shape-runtime fields (`M/N/K/...`) at the sentinel (typically `-1`) so they read from runtime tiling. Each non-sentinel field eliminates one GM read in the hot path.
- **By-construction determinism rider**: when the typed config + runtime tiling pattern is single-instance-per-output (one accelerator core per output tile, no atomic write, fixed iteration order via constant tiling), `DET_POLICY=required` is satisfied by construction.
- **Static-check escape hatch**: tools that detect "kernel has computation" by counting primitive markers (`TQue/TBuf/DataCopy/VEC_op/GlobalTensor/LocalTensor`) may not recognize hardware-accelerator-impl symbols (e.g. `MatmulImpl::IterateAll`). Add the accelerator's iteration symbol to the marker set, or as a workaround allocate a no-op scratch buffer of an existing primitive type (the long-term fix is extending the marker set, not the workaround).
- **Cube-unit instance** (current evidence base): cube unit has typed config `MatmulApiStaticTiling` wrapping `MatmulConfig`. Code template + cube-specific gotchas (`MatmulApiStaticTiling.batchM/batchN` don't exist; factory needs `__aicore__ inline constexpr`; local `constexpr auto X = factory<T>()` inside templated function needs `static constexpr` for clang; `TCubeTiling` is exactly 50 int32 fields = 200 bytes for CANN 9.0.0) live in `patterns/domains/platform_compat.md §P-P68`. EC-39 / EC-40 capture the specific build errors.
- **Other instances (predicted)**: sort engine (if it has a `SortApiStaticConfig` style wrapper), scan/reduction engines, MXU/quant engines on newer SOC. Verify by reading the impl's `Init` signature: if the second template arg is a typed struct (not a plain enum/policy), this OL applies.
- **Evidence**: 1_BatchMatmul (2026-04-28 cube cold-start, 3 build iters — first encounter with the typed-config trap, then Opt2 0.515×→1.27× via constexpr+on-stack); 4_MatmulTransA (1.36× median, 0 build iters — pattern carried); 5_MatmulTransB (1.29×, 0 build); 3_MatmulBothTrans (1.45×, 0+0); 2_GroupedMatmul (1.05× median, 0+0 — extends to multi-instance dispatch, see OL-93). Five cube ops; pattern stable.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-91（category=platform_compat，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
