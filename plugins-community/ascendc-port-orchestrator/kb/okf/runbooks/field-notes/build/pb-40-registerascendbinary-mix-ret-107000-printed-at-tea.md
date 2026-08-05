---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`RegisterAscendBinary mix ret 107000` printed at teardown on `KERNEL_TYPE_MIX_AIC_1_2` V220 — non-fatal, independent of matmul-primitive [V220, mixed-mode-binary-register]"
description: "applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; kernel_type=KERNEL_TYPE_MIX_AIC_1_2"
phenomenon: build_failure
signal:
  - "stdout (NOT stderr) prints exactly one line RegisterAscendBinary mix ret 107000 AFTER the kernel's RAN_OK latency_ms=... + output comparison. A bare [ERROR] lin"
confidence: single_run
original_id: PB-40
timestamp_inferred: true
tags: [107000, kernel_type_mix_aic_1_2, ascendc, pb-40]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; kernel_type=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: npu-a3@198.51.100.92 (NPU 0 idle), bisheng/910C, CANN 9.0.0, 2026-05-27 independent prototype firsthand`
`unverified_on: V351/A5 (not tested)`

- **Severity (UNDER REVIEW 2026-05-27)**: initially documented LOW based on "kernel produces output before 107000". independent prototype blackbox D-sweep 2026-05-27 02:22Z refuted that — at D≥64 main shapes `cand_absmax=0.000` (output ZERO). Then 02:27Z independent prototype surfaced **target mismatch**: the FSM-emit kernel is A5/V351-targeted (per `PROGRESS.md` + UB-budget 248KB + `kernel.h:114` "A5 V351 cross-core" comment) and was being tested on A3/V220 — i.e. all V220-side observations are running an A5-incompatible binary on V220 hardware. A5-side re-verify (02:41Z) on `Ascend950PR_957b` (correct target) showed the SAME kernel **deadlocks** (kernel launch >60s, no return, PYEXIT=124). So on V220 the kernel fails-to-compute (this PB's 107000), on A5 it deadlocks (separate failure). **Open question**: is 107000 a real V220 platform bug for multi-entry `MIX_AIC_1_2`, OR is it V220's loader correctly rejecting an A5-format binary? Cannot distinguish without testing a V220-CORRECT multi-entry `MIX_AIC_1_2` kernel on V220. Severity stays UNDER REVIEW pending that disambiguating run.
- **Symptom**: stdout (NOT stderr) prints exactly one line `RegisterAscendBinary mix ret 107000` AFTER the kernel's `RAN_OK latency_ms=...` + output comparison. A bare `[ERROR]` line with no message follows. No `aiv ret 0` / `aic ret 0` companion success lines. Kernel still produces output (numerical correctness of that output is a separate concern).
- **Trigger**: A `.cpp` file contains TWO or more `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` entry-pairs where only ONE is invoked at runtime (e.g. `fusion_attention_custom_fp16_nopse` + `fusion_attention_custom_fp16_pse`, dispatched by `has_pse` in pybind; case0 uses `pse=None` → only `nopse` entry called → `pse` binary's deferred registration fails at teardown).
- **Cross-ref**: PB-28 (V220 + `KERNEL_TYPE_AIV_ONLY` + fatal-never-register). Scope distinct: PB-28 is `AIV_ONLY` + fatal-never-register; PB-40 is `MIX_AIC_1_2` + non-fatal deferred-register-at-teardown.
- **Independence from matmul-primitive**: confirmed across attempt-2 (Matmul-lib×3 + Mmad×5 emit) and attempt-3 (manual Mmad-only emit, post PR #191 `83f0bf0a`). Both produced the identical 107000 signature on the same V220 host. Therefore NOT coupled to cube primitive choice — it is a mix-binary registration property.
- **Detection**: stdout grep `RegisterAscendBinary mix ret 107000` on V220 builds containing multiple `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` entries. Pre-build heuristic: `grep -cE "KERNEL_TASK_TYPE_DEFAULT\(KERNEL_TYPE_MIX_AIC_1_2\)" workspace/<op>/kernel/*.cpp` returning > 1 with same-architecture entry pairs flags the risk.
- **Recommended action (proposed, NOT verified)**:
  1. Emit only ONE `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` per `.cpp`; collapse `_nopse` / `_pse` codepath divergence via runtime conditional inside a single entry, OR
  2. Split into multiple `.cpp` files (one entry-pair each) so the unused binary isn't co-loaded with the used one.
  Owner verification still required before promoting either to a hard rule.
- **Evidence**: independent prototype 2026-05-27 A3 firsthand on FSM-emit 3_FusionAttention attempt-3 (post PR #191 `83f0bf0a`). Disk artifact: `workspace/3_FusionAttention/kernel/fusion_attention_fp16.cpp` (two `MIX_AIC_1_2` entries). Timing trace: `RAN_OK latency_ms=79.572` → output emitted → `RegisterAscendBinary mix ret 107000` → bare `[ERROR]`. Discord context: 2026-05-27 01:32Z (timing trace) + 02:17Z (scope-2 independence verdict after matmul-primitive change).
- **Other instances (predicted)**: any V220 archive that registers >1 entry-pair of the same `MIX_AIC_*` type in a single .cpp where some entries are unused per dispatch — generic to mixed-mode + multi-entry-per-source-file emission patterns.

<!-- 迁移自 porter kb/target/ascendc/（PB-40，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
