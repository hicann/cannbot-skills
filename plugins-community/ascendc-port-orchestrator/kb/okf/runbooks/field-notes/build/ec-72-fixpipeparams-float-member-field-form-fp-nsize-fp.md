---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`FixpipeParams<float>` member-field form (`fp.nSize=`/`fp.mSize=`) does NOT compile on V220 (arch22) — use the positional `FixpipeParamsV220(...)` ctor + templated `Fixpipe<dstT, srcT, CFG_ROW_MAJOR>` call"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=cube-l0c-to-gm"
phenomenon: build_failure
signal:
  - "a cube kernel staging an L0C(fp32) → GM(fp32) fixpipe writes the params with the member-assignment form illustrated in some reference docs (FixpipeParams<float>"
confidence: single_run
original_id: EC-72
timestamp_inferred: true
tags: [ascendc, ec-72]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=cube-l0c-to-gm`
`verified_on: soc=Ascend910_V220; cann=9.0.0`
`unverified_on: soc=Ascend950PR (arch35 has the OPPOSITE constraint — see note below)`

**Symptom**: a cube kernel staging an L0C(fp32) → GM(fp32) fixpipe writes the params with the member-assignment form illustrated in some reference docs (`FixpipeParams<float> fp; fp.nSize = ...; fp.mSize = ...;`) and fails to compile on V220 — the templated `FixpipeParams<T>` struct does not expose `nSize`/`mSize` members on arch22/2201.

**Root cause**: on V220 the FIX-pipe descriptor is the dedicated struct `FixpipeParamsV220` (`kernel_struct_fixpipe.h`), constructed positionally — NOT a generic templated `FixpipeParams<T>` with assignable fields. The member-field form is an arch35-flavored illustration that does not exist on arch22.

**Fix** (V220 L0C→GM ND, fp32→fp32, no cast):
```cpp
// V220 (arch22): positional ctor + templated Fixpipe call
Fixpipe<float, float, CFG_ROW_MAJOR>(
    gmDst, l0cSrc,
    FixpipeParamsV220(nSize, mSize, srcStride, dstStride, /*reluEn=*/false));
```
`CFG_ROW_MAJOR` selects the L0C(fp32)→GM(fp32) ND (row-major) layout. After the rewrite the kernel TU compiles (build PASS).

**arch35 has the OPPOSITE constraint** (do NOT copy this V220 form to A5): on arch35/950PR the L0C→GM fixpipe MUST use `FixpipeParamsC310` (NZ2ND) with an explicit `quantPre` cast mode (`F322BF16` for float→bf16, `F322F16` for float→half). The arch22 `FixpipeParamsV220` (no cast) raises device error 169 subErrType 0x4 on arch35. See `patterns/domains/fa_class/templates/op_kernel/matmul_tile.h` (2026-06-16 arch35 fixpipe note).

**Evidence**: fa_gqa_grad kw-1 iter-1 (2026-06-19, port_a3_to_a5, Ascend910_V220/arch22, CANN 9.0.0): a hand-`Mmad` cube GEMM staging L0C→GM hit the no-such-member compile failure on the doc's member-field form; rewriting to `FixpipeParamsV220(nSize, mSize, srcStride, dstStride, reluEn)` + `Fixpipe<float,float,CFG_ROW_MAJOR>(...)` → build PASS.

**Other instances (predicted)**: any V220 cube kernel that drains an L0C accumulator to GM via FIX — hand-written `Mmad` GEMM ladders (FA-class fwd/bwd, GroupedMatmul, custom matmul), regardless of op class.

**Cross-reference**: A-P34 (`KERNEL_TYPE_*_ONLY` arch-guard — same V220-vs-arch35 entry-form divergence class), `patterns/unverified/candidates.md` (FixpipeParamsV220 opaque-field workflow), `patterns/domains/fa_class/templates/op_kernel/matmul_tile.h` (arch35 `FixpipeParamsC310` complementary note).

<!-- 迁移自 porter kb/target/ascendc/（EC-72，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
