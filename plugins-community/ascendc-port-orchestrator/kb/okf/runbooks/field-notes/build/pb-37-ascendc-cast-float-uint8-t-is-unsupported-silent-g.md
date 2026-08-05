---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC `Cast<float, uint8_t>` is unsupported — silent garbage, not an error"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=all (any kernel that needs uint8/bool tensor in fp32 compute)"
phenomenon: build_failure
signal:
  - "Cast(fp32_dst, u8_src, RoundMode::CAST_NONE, count) produces an instruction that does NOT do what the call site implies. The destination LocalTensor is left in"
confidence: single_run
original_id: PB-37
timestamp_inferred: true
tags: [ascendc, pb-37]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=all (any kernel that needs uint8/bool tensor in fp32 compute)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (A3 V220 — pattern not yet probed there)`
- **Severity**: HIGH (silent miscompile — output values 10×–100× reference magnitude; no compile error, no runtime error, no NaN/Inf signal).
- **Symptom**: `Cast(fp32_dst, u8_src, RoundMode::CAST_NONE, count)` produces an instruction that does NOT do what the call site implies. The destination LocalTensor is left in an indeterminate state — observed values uncorrelated with input (large-magnitude garbage even when input is bool 0/1).
- **Root cause**: AscendC `Cast` only supports a specific table of (src_dtype, dst_dtype) pairs. Per official docs, supported uint8 SRC pairs are: `uint8 → half`, `uint8 → uint16_t`, `uint8 → uint32_t`. NOT in the table (and silently produce garbage):
  - `uint8 → float` ✗
  - `uint8 → bfloat16_t` ✗
  - `uint8 → int8 / int16 / int32` ✗
- **Workaround — canonical two-step lift through half**:
  ```cpp
  __aicore__ inline void CastU8ToFp32(
      const LocalTensor<float>& dst,
      const LocalTensor<uint8_t>& src,
      const LocalTensor<half>& tmp_half,   // VECCALC scratch, count-aligned to 16
      int32_t count)
  {
      Cast(tmp_half, src,        RoundMode::CAST_NONE, count);  // u8 → half (exact for 0/1 mask values)
      PipeBarrier<PIPE_V>();
      Cast(dst,      tmp_half,   RoundMode::CAST_NONE, count);  // half → fp32 (exact for 0/1)
      PipeBarrier<PIPE_V>();
  }
  ```
  Both legs ARE in the supported pair table. For boolean masks the two-cast chain is bit-exact (0/1 representable in half exactly).
- **Detection**: precision `max_abs_diff` shows large-magnitude divergence with no obvious per-element correlation; mismatch count not factorable into alignment-overshoot. Before assuming algorithm bug, **grep the kernel for `Cast(.*float.*uint8_t.*)` or `Cast(<float-LT>, <u8-LT>, ...)` patterns** — even one such call short-circuits to PB-26.
- **Prevention (Phase B mandatory check)**: any `Cast<DST, SRC>` in a kernel must be cross-checked against the AscendC Cast precision-conversion table before the build. OL-80 + OL-84 already mandate "check docs before Cast"; PB-26 is the named instance that catches the cost of skipping it.
- **Evidence**: op#25 MaskedSoftmaxWithAttentionDropoutBackward kw-2 (2026-04-24). Boolean mask `attention_mask` (uint8) and dropout mask (uint8) needed in fp32 compute path. Direct `Cast<float, uint8_t>(maskFp32, maskU8, CAST_NONE, count)` produced output values 10×–100× reference. ~1.5 hours of precision debugging — kw-1 had diagnosed three other root causes (alignment, broadcast, scalar fusion) but missed this. Two-step lift via `tmp_half` fixed the path; rest of the kernel was already correct.
- **Related**: OL-80 / OL-84 (always check API surface before assuming an op exists), EC-23 (DataCopy alignment — orthogonal), P-P52 (fp32 promotion — different problem; P-P52 assumes the cast itself is supported).

<!-- 迁移自 porter kb/target/ascendc/（PB-37，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
