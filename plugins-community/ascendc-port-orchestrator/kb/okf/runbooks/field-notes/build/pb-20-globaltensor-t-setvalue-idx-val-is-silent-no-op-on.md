---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`GlobalTensor<T>::SetValue(idx, val)` is silent no-op on Ascend950PR"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "kernel compiles, runs, returns success. Output GM tensor contains uninitialized values (torch::empty garbage). No diagnostic output. Precision FAIL on every cas"
confidence: single_run
original_id: PB-20
timestamp_inferred: true
tags: [ascendc, pb-20]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Status**: CONFIRMED (2026-04-22 op#5; 2026-04-24 cross-checked against CANN source)
- **Affected**: Ascend950PR / CANN 9.0.0 / bisheng 2026-03-21. SIMT-AIV kernel context. Only `GlobalTensor<T>::SetValue(uint64_t idx, T val)` (scalar GM write at indexed position) — bulk `DataCopy(GM, UB)` works fine.
- **Symptom**: kernel compiles, runs, returns success. Output GM tensor contains uninitialized values (`torch::empty` garbage). No diagnostic output. Precision FAIL on every case.
- **Confirmation against CANN source (2026-04-24, op#3)**: CANN's own `cann/ops-nn/optim/advance_step/op_kernel/advance_step.h` writes every output via `GlobalTensor<int64_t>::SetValue(idx, val)` — exactly the broken pattern. Crucially: that op's `op_host/advance_step_def.cpp` only registers `ascend910b` + `ascend910_93` AICore configs; **A5 is excluded specifically because the kernel pattern doesn't work there**. So PB-20 is not a worker quirk — it's a fundamental CANN-vs-A5 SIMT-AIV write-path mismatch that CANN itself works around by simply not shipping A5 binaries.
- **Workaround — context-dependent decision tree** (refined 2026-04-30 op#22 Nonzero):
  - **SIMT VF kernel** (`Simt::VF_CALL<f>(Simt::Dim3{N}, ...)` wrapped functions, `LAUNCH_BOUND(K)` annotated): use raw `__gm__ T*` pointer indirect writes via `reinterpret_cast<__gm__ T*>(GM_ADDR_arg)` then `gm[i] = val;`. Different opcode path (scalar pipe direct GM access). Reference templates: `output/npukernelbench/src/kernels/19_IndexPut/`, `output/npukernelbench/src/kernels/3_AdvanceStepFlashattn/`.
  - **Pure-AIV class kernel** (plain `extern "C" __global__ __aicore__ void f(...) { KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); class.Init(...).Process(); }` — the multi-core SIMD style used by MoeInitRouting / FusedAddRmsnorm): **DataCopy from UB LocalTensor to GM** via MTE3 pipe. Empirical: in this context, BOTH `GlobalTensor::SetValue` AND raw `__gm__ p[i]=v` silently fail (op#22 sentinel test, 2026-04-30, 8 hypotheses by kw-2 + minimal-sentinel reproduction by orchestrator). Pattern:
    ```cpp
    GlobalTensor<int32_t> g;  g.SetGlobalBuffer(...);
    LocalTensor<int32_t> ub = ub_buf.Get<int32_t>();
    ub.SetValue(0, my_val);              // UB scalar SetValue OK
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(g[off], ub, count);          // UB→GM via MTE3 — WORKS
    ```
  - Reference template for pure-AIV class kernel: `output/npukernelbench/src/kernels/5_MoeInitRouting/kernel/moeinitrouting_kernel.h` — every GM write goes through `DataCopy(...)` or `AtomicAdd(...)`, NEVER `GlobalTensor::SetValue`.
- **Implication for aog-kernel-worker / optimizer**: when porting from CANN reference kernels (especially scatter / in-place ops), grep the CANN source for `SetValue(` calls on `GlobalTensor` — every one of those needs to be rewritten per the kernel-context above. **Do not** assume "CANN does it this way, so we can too". When designing a new multi-core kernel, prefer the pure-AIV class pattern AND use DataCopy(UB→GM) for GM writes.
- **Implication for benchmark reference**: if a benchmark op's CANN kernel uses `SetValue` on GM, attempting to build that kernel from source against A5 will produce a binary that compiles but silently produces wrong output. This is "OL-68 Case B" — see OL-68 sub-case taxonomy.
- **Evidence**:
  - op#5 MoeInitRouting probe (2026-04-22) — sort kernel Pass 3 had this pattern → 0/50 PASS, all output uninitialized. After switching to UB TBuf staging + `DataCopy UB→GM` + `SetFlag/WaitFlag<HardEvent::S_MTE3>` sync (or alternatively raw `__gm__` pointer writes), → 53/53 PASS.
  - op#3 AdvanceStepFlashattn (2026-04-24) — used raw `__gm__ int64_t*` pattern from the start (per IndexPut precedent), 50/50 + 28/28 PASS.
  - op#22 Nonzero V2 (2026-04-30) — kw-2 attempted multi-core SIMD class kernel, K=0 silent-fail on all 50 cases despite 8 hypotheses (rename ws→ws_buf to dodge HAVE_WORKSPACE codegen, raw `__gm__` writes, GlobalTensor::SetValue, MTE2→S sync, fixed nblk=56, minimal sentinel `SetValue(0,42)`, bypass class entirely). Orchestrator reproduced K=0 with minimal sentinel kernel; switching to `DataCopy(UB→GM)` immediately worked. Confirms raw `__gm__` workaround does NOT generalize from SIMT VF to pure-AIV class context.
  - op#22 Nonzero V4 (2026-04-30, second probe-before-edit success) — kw-5 wrote `workspace/22_Nonzero/probes/gathermask_probe/` BEFORE attempting V4 SIMD-emit kernel, verified `GatherMask + ReinterpretCast<uint32_t>(CompareScalar bitmask)` API contract on a5 in ~3 minutes. V4 then built first try, no compile-fix iters, no precision-fix iters, 50/50 + 10/10 + det 50/50 PASS. Second consecutive op (after V2 sentinel) where probe-first prevented worker churn. Recommendation: when an API has thin documentation (single-line catalog entry, zero codebase precedent), run a 1-block probe before committing kernel architecture.
  - group_norm_silu_quant (2026-05-13, A5 fused GroupNorm+SiLU+Quant port from A3 aclnn) — iter 1→2 attempted `const_cast<__gm__ T*>(gmMean_.GetPhyAddr())` + `mean_ptr[idx] = mean_t` for per-(n,g) scalar mean/rstd outputs from pure-AIV class kernel. Compiled clean, ran clean, mean/rstd contained uninitialized garbage (max_abs_diff up to 2.09e+28 / FP_MAX sentinel on case 6 bf16). Exactly the second bullet of the decision tree above. Fix (iter 3→4): switched to `DataCopy(gmMean_[idx*16], ub_block, 16)` with 16x-padded GM workspace (each `(n,g)` element gets its own 32B-aligned slot), pybind extracts `mean_ws.select(2, 0).contiguous()` post-kernel → mean/rstd bit-exact (0.0 diff) on all 8 cases. **Sub-pattern for small per-work-unit scalar outputs**: see CAND-PB20-GMPAD candidate — wraps the `DataCopy(UB→GM)` workaround with the 16x-pad allocator so scalar-per-(n,g) outputs satisfy 32B alignment without inter-AIV races.

<!-- 迁移自 porter kb/target/ascendc/（PB-20，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
