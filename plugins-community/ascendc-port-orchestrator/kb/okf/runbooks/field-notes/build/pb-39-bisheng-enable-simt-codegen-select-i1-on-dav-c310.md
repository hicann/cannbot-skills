---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bisheng `--enable-simt` codegen — `select i1` on `dav-c310-vec` fires `Copy register different width` [V351/A5, --enable-simt-only]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=2026-04-03+ through 2026-04-30; mode=--npu-arch=dav-3510 --enable-simt"
phenomenon: build_failure
signal:
  - "fatal error: error in backend: Copy one register into another with a different width from bisheng --npu-arch=dav-3510 --enable-simt. Larger kernels (5K+ lines o"
confidence: single_run
original_id: PB-39
timestamp_inferred: true
tags: [ascendc, pb-39]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=2026-04-03+ through 2026-04-30; mode=--npu-arch=dav-3510 --enable-simt`
`also_affects: cann=9.1.0-beta.1 — partial fix only`
`verified_on: bisheng 2026-04-03 (CANN 9.0.0), 2026-04-28 (per user report), 2026-04-30 (CANN 9.1.0-beta.1)`

- **Severity**: HIGH for any code path emitting `--enable-simt` IR (generated pure-SIMT ports, hand-written pure-SIMT kernels, possibly some compiler-generated SIMT lowering paths). NOT relevant to ordinary AscendC operator development that uses SIMD intrinsics + `Select<T>()`.
- **Status**: OPEN. CANN 9.1.0-beta.1 ships a narrow codegen patch that handles the simple multi-`select`-with-shared-cond pattern, but kernels with richer i1 fan-out still trip.
- **Symptom**: `fatal error: error in backend: Copy one register into another with a different width` from `bisheng --npu-arch=dav-3510 --enable-simt`. Larger kernels (5K+ lines of generated SSA) segfault inside `HiTPE DAG->DAG Pattern Instruction Selection` instead — same defect reached one pass earlier.
- **Trigger**: LLVM `select i1 %cond, T %a, T %b` instructions where the i1 predicate is routed through a width-mismatched register MOV during SDAG→MachineInstr lowering. Multiple selects sharing one i1 condition (the SROA'd form of an aggregate ternary) trigger reliably for most result-type/fan-out combinations.
- **Secondary bug — surfaces only after the primary is patched**: `fatal error: error in backend: MaxThreads out of range!` — backend requires `!"simt-max-threads", i32 2048` annotation on every function in the module that holds (transitively) a 3-arg scoped atomic intrinsic.
- **Workaround**: 6-step IR-rewrite pipeline. Reproducible end-to-end from `gitcode.com/example/bisheng-crash-repro-cann-9` commit `0e89c0d` (private; share key on request). The pipeline:
  1. **Source**: change `&&` → `&` in `assert(...)` calls in any `array.h`-style bounds-check header (eliminates `select i1, i1, i1` from short-circuit AND).
  2. **Emit O3 LLVM bitcode** from the source: `bisheng --npu-arch=dav-3510 --enable-simt -DWP_ENABLE_ASCEND -O3 -emit-llvm -c <src>.asc`
  3. **Disassemble**: `bisheng -cc1 -triple hiipu64-hisilicon-cce -x ir -S -emit-llvm <bc>` → text IR.
  4. **IR rewrite**: replace each `select i1 %c, T %a, T %b` with a `br + phi` diamond, preserving SSA dominance (relabel every successor-phi predecessor reference from the original block label to the new tail-block label; entry-block's implicit numeric label is the param count). For `T==i1` widen through `zext i1→i32 / phi i32 / icmp ne` to avoid emitting `phi i1` (also broken).
  5. **IR metadata**: append `!annotation !<id>` (id → `!{!"simt-max-threads", i32 2048}`) on every function definition and matching `{ptr @<fn>, !"simt-max-threads", i32 2048}` entries in `!hivm.annotations`. Brute-force every-function form needed — annotating only obvious helpers leaves siblings unannotated.
  6. **Compile + link**: `bisheng -cc1 -triple hiipu64-hisilicon-cce -x ir -O0 -emit-obj` (-O0 mandatory — any -O1+ pass re-folds the diamonds back into `select i1`), then `ld.lld -shared -Bsymbolic` (-Bsymbolic needed because `g_sysSimtPrintFifoSpace` is weak and the backend emits PC-relative relocs for it).
- **Source-level workarounds that DON'T work**: per-component scalar select, `if/else` + out-param, `-fno-vectorize -fno-slp-vectorize`, `-DNDEBUG`, `__attribute__((noinline))/((optnone))/((flatten))`, packed struct, raw float pointer instead of struct, `volatile` tmp. All hit the same backend error because the optimizer re-fuses to the buggy pattern before codegen.
- **Evidence**:
  - Minimal repro: 30 lines, no Warp deps, parameterised on `SLEN` for struct length. CANN 9.0.0 N=1/3/5/6/7/8 FAIL at -O3; CANN 9.1.0 all N OK.
- **Recommended report to Huawei**: minimal repro `repro_min.asc` (30 lines, no proprietary deps) is sufficient to drive the bisheng team's diagnosis. The secondary "MaxThreads" diagnostic should also be improved to name the offending function.

<!-- 迁移自 porter kb/target/ascendc/（PB-39，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
