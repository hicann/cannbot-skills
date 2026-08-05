---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`DataCopy(UB, TBuf<TPosition::A1>)` in pure-AIV kernel — silent miscompile → runtime illegal instruction"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel compiles with NO warnings or errors. At runtime, first DataCopy touching an L1-backed LocalTensor (TPosition A1) from a pure-AIV kernel faults with aivec"
confidence: single_run
original_id: PB-16
timestamp_inferred: true
tags: [507035, datacopy, localtensor, ascendc, pb-16]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Symptom**: Kernel compiles with NO warnings or errors. At runtime, first `DataCopy` touching an L1-backed `LocalTensor` (TPosition A1) from a pure-AIV kernel faults with `aivec error exception, core id is 0, error code = 259, subErrType: 0x4, "Illegal instruction, which is usually caused by unaligned UUB addresses"`. Runtime status 507035. Error persists regardless of explicit `SetFlag<HardEvent::MTE3_MTE1>` sync — the fault is at the opcode level, not a sync bug.
- **Affected**: CANN 9.0.0 (innerversion V100R001C10SPC001B218), bisheng 2026-03-21, Ascend950PR_9589. Pure-AIV kernels (no Cube/Mmad). Both `DataCopy(UB, L1)` and `DataCopy(L1, UB)` overloads.
- **Misleading error message**: The "unaligned UUB" wording is not the actual root cause. Probe used 4 KB buffer, 32 B-aligned LocalTensors, element count a multiple of block — no alignment issue. The message is the generic err 259 text.
- **Workaround**: Do not use `TBuf<TPosition::A1>` (or A2/B1/B2/C1/C2/CO1/CO2) in a pure-AIV kernel on this toolchain. To use the hardware UB↔L1 channel documented in the 351x arch page, either (a) wait for CANN to expose a dedicated AIV-scope intrinsic (e.g. `CopyUbufToL1` variant), or (b) run the kernel as a mixed AIC+AIV task so a Cube context exists. For UB-budget-overflow optimization, use alternative axes: smaller tiles, fp16 intermediate with fp32 compute, split kernel into 2 launches.
- **Status**: OPEN. Escalate to CANN team asking whether a dedicated pure-AIV UB↔L1 intrinsic is planned. Re-probe after CANN version upgrade.
- **Evidence**: `src/skills/references/hardware/probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md` — full probe report with build logs + runtime stderr from 2 iterations (PipeBarrier + explicit MTE3→MTE1 SetFlag/WaitFlag variants both fail identically).
- **Severity**: HIGH. The silent-compile behavior is particularly dangerous: any optimizer that reads the public AscendC API ref (TPosition includes A1 as valid; TBuf accepts any TPosition; TBufPool docs explicitly say L1 is a managed resource) would reasonably conclude this is a legitimate optimization path and burn a full aog-kernel-worker iteration budget before discovering the runtime fault.
- **2026-04-21 update — CANN source cross-check confirms constraint**:
  - Low-level intrinsic `DataCopyUB2L1Impl((__cbuf__ T*)dst, (__ubuf__ T*)src, DataCopyParams)` exists at `ops-nn/matmul/common/cmct/tile/copy_ub_to_l1.h` + catlass parallels. This IS the functional UB→L1 DMA primitive.
  - Every single `TPosition::A1` usage across `ops-transformer / ops-nn / opbase / catlass / graph-autofusion` (grep -rlI) is inside matmul/Cube kernels. **Zero hits in pure-AIV kernels.**
  - Interpretation: the generic `DataCopy(LocalTensor<A1>, LocalTensor<UB>)` template does NOT route to `DataCopyUB2L1Impl` in pure-AIV compile context — it resolves to a no-op or placeholder opcode. The correct path is either (a) use `DataCopyUB2L1Impl` directly with memory-space-tagged raw pointers (but this is an internal API, not in public AscendC ref), or (b) run the kernel as a mixed AIC+AIV task so the Cube lowering is active.
  - Therefore: **PB-16 is not a bisheng bug in the "miscompile" sense — it's an undocumented constraint ("TPosition::A1 is Cube-context only"). Bisheng should warn/error, that part IS a bug. The runtime behavior is expected.**

<!-- 迁移自 porter kb/target/ascendc/（PB-16，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
