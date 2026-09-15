---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-variant kernel organization — sibling `<op>_<variant>.h` files dispatched by tiling-key from a thin `.cpp`"
description: "> Source: cann-learner CAND-A3A5-4 (promoted 2026-05-12 from PR4778 cross-op-evidence batch, 5 ops). applies_to: soc=all; cann=9.0.0; bisheng=15.0.5; op_class=all verified_on: soc=Ascend950PR (PR4778"
confidence: single_run
original_id: P-P91
timestamp_inferred: true
tags: [patterns-index, optimization, gather_elements_v2, index_put_with_sort, apply_adam_w_quant, top_k_top_p_sample_v2, group_norm_silu_quant, p-p91, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

> **Source**: cann-learner CAND-A3A5-4 (promoted 2026-05-12 from PR4778 cross-op-evidence batch, 5 ops).

`applies_to: soc=all; cann=9.0.0; bisheng=15.0.5; op_class=all`
`verified_on: soc=Ascend950PR (PR4778 arch35 ports — 5 ops with variant-split confirmed); soc=Ascend910_V220 (the SAME 5 ops' V220 master state ALSO uses this convention — pattern pre-dates A5 port; PR4778 preserves it)`
`unverified_on: soc=Ascend910_V220 in a3/a2 from-scratch op-gen context (we observed it in PORTED ops; if a3 kw decides to write a SINGLE mega-template kernel that violates this convention, the precision/perf cost is unknown — no counter-evidence yet)`
`note: this pattern is a CODE-ORGANIZATION rule, not a hardware-tied rule. Applies wherever multiple algorithmic variants live in the same op_kernel/ directory regardless of target SoC. The V220 master state already follows it; A5 arch35 port preserves it. From-scratch a3/a2 op-gen via aog-kernel-worker should follow it too — but we have not validated the negative case empirically.`

**Principle**: when a kernel has multiple algorithmic variants (different dim layouts, dtype-divergent codepaths, scatter vs gather phases, etc.), keep variants in **sibling header files** under `op_kernel/` (or `op_kernel/arch35/` for A5 ports) and dispatch via TILING_KEY in a thin top-level `.cpp`. **Don't merge variants into one mega-template kernel** even when bodies share 80% of code.

A5's wider regbase MicroAPI surface tempts authors to write one mega-template kernel that switches via `if constexpr` on every axis. The master-state convention (preserved through A5 port) is more readable AND produces better object-file structure (per-variant `.o` files, smaller per-launch binary).

**Concrete anchor**:
```cpp
// op_kernel/<op>.cpp (top-level dispatcher — thin)
#include "<op>_scalar.h"
#include "<op>_transpose.h"
#include "<op>_last_dim.h"
#include "<op>_common.h"
extern "C" __global__ __aicore__ void op(GM_ADDR ..., GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    if (TILING_KEY_IS(0)) { OpScalar<...> op; op.Init(...); op.Process(); }
    else if (TILING_KEY_IS(1)) { OpTranspose<...> op; op.Init(...); op.Process(); }
    else if (TILING_KEY_IS(2)) { OpLastDim<...> op; op.Init(...); op.Process(); }
}
```

**Evidence** (cross-op, 5 ops):
- `gather_elements_v2`: 4 variant files (scalar, transpose, last_dim, common)
- `index_put_with_sort`: 3 phase files via inheritance (base, gather_data, scatter_data)
- `apply_adam_w_quant`: 2 dtype-split files (fp16, fp32) + shared base
- `top_k_top_p_sample_v2`: 3 files (main, comm, sort_cumsum) — see also OL-133
- `group_norm_silu_quant`: 2 files (base, b16)

**Sub-patterns**:
- **Dtype-split (sub-case)**: when fp16 + fp32 algorithms differ in buffer count / quantization LUT / accumulator dtype, splitting by dtype is cheaper to port than templating because per-dtype divergences span the whole Process() body. Evidence: `apply_adam_w_quant_fp16.h` + `apply_adam_w_quant_fp32.h`; `group_norm_silu_quant_b16.h` + `group_norm_silu_quant_base.h`. (Originally CAND-A3A5-6, folded into P-P91.)
- **Sort + cumsum split**: sort algorithm in dedicated header (`<op>_sort_cumsum.h`); main flow in `<op>.h`; shared types in `<op>_comm.h`. Evidence: `top_k_top_p_sample_v2`. (Originally CAND-A3A5-13, folded into P-P91.)

**Anti-pattern (DO NOT)**:
- Single mega-template kernel with `if constexpr` on every axis — harder to debug, larger per-launch binary, defeats per-variant TILING_KEY dispatch.
- Mega-template using `template<bool IsTranspose, bool IsLastDim, int Dtype>` — A5's regbase MicroAPI surface MAKES THIS POSSIBLE but the master-state convention says don't.

**Cross-ref**: OL-133 (`ASCENDC_TPL_ARGS_DECL` for compile-time axis enumeration — complementary to P-P91; use TPL_ARGS_DECL to declare WHICH variants exist, P-P91 to organize WHERE they live); P-P122 (inheritance-based variant of variant-split — use P-P122 when phases share state via class-hierarchy, P-P91 when variants are independent dtype/layout dispatches).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P91，convert_patterns_to_okf.py）。confidence 未升格。 -->
